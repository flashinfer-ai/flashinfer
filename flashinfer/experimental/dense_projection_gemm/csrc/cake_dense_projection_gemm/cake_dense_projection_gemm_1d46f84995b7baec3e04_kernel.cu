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
#define NUM_TMA_PIPE_STAGES 9
#define NUM_ACC_PIPE_STAGES 2
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 24576
#define SMEM_SMEM_B_OFF 17408
#define SMEM_SMEM_B_STAGE_BYTES 8192
#define SMEM_SMEM_B_STRIDE 24576
#define SMEM_TOTAL 222208
#define THREADS 320

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

__device__ __forceinline__ void mbarrier_wait_cluster_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER_HINT:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_CLUSTER_HINT;\n\t"
        "bra.uni LAB_WAIT_CLUSTER_HINT;\n\t"
        "DONE_CLUSTER_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
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



__device__ __forceinline__ void tcgen05_commit_cg2_multicast_leader(
    int mbar_addr, uint16_t cta_mask, uint32_t leader) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "setp.ne.b32 leader, %2, 0;\n\t"
        "@leader tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;\n\t"
        "}\n"
        :: "r"(mbar_addr), "h"(cta_mask), "r"(leader) : "memory");
}

extern "C" {

__global__ __launch_bounds__(320) __cluster_dims__(2,1,1) void
kernel_cake_dense_projection_gemm_1d46f84995b7baec3e04(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, __nv_bfloat16* __restrict__ out, float* __restrict__ ws, unsigned int* __restrict__ counters, int M, int N, int K, int num_l, int num_splits, int ldo, int out_l)
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
    #define mma_done_addr (mbar_base + 72)
    #define acc_full_addr (mbar_base + 144)
    #define acc_empty_addr (mbar_base + 160)
    #define tmem_free_addr (mbar_base + 176)

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

    // Mbarrier init (5 pipeline groups, 0 ordered-sequence groups, 23 barriers)
    // Mbarriers at smem_raw[0..184)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full: 9 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            // mma_done: 9 barriers, init_count=1
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            // --- pipeline 'acc_pipe' ---
            // acc_full: 2 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            // acc_empty: 2 barriers, init_count=16
            mbarrier_init(smem + 160, 16);
            mbarrier_init(smem + 168, 16);
            // tmem_free: 1 barriers, init_count=32
            mbarrier_init(smem + 176, 32);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 184);
    if (warp == 1) {
        int _tmem_hold = smem + 184;
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
            int m_tiles = (M + 256 - 1) / 256;
            int tiles_l = m_tiles * ((N + 128 - 1) / 128);
            int num_tiles = num_l * tiles_l;
            int k_iters = (K + 64 - 1) / 64;
            unsigned int _phase_mma_done = 1;
            if (elect_sync()) {
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A))) : "memory");
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B))) : "memory");
                #pragma unroll 1
                for (unsigned int item = cluster_id; item < num_tiles * num_splits; item += num_clusters) {
                    int split = (int)item / num_tiles;
                    int tile = (int)item - split * num_tiles;
                    int l_idx = tile / tiles_l;
                    int in_l = tile - l_idx * tiles_l;
                    int n_tile = in_l / m_tiles;
                    int m_tile = in_l - n_tile * m_tiles;
                    int k_begin = split * k_iters / num_splits;
                    int k_end = (split + 1) * k_iters / num_splits;
                    int a_row = m_tile * 256 + cta_rank * 128;
                    int b_col = n_tile * 128 + cta_rank * 64;
                    #pragma unroll 1
                    for (int k_block = k_begin; k_block < k_end; k_block++) {
                        mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                        int k0 = k_block * 64;
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(smem_a_addr + load_stage * 24576), "l"((&A)), "r"(a_row), "r"(k0), "r"(l_idx),
                               "r"(((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF)), "l"(0x12F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(smem_a_addr + load_stage * 24576 + 8192), "l"((&A)), "r"(a_row + 64), "r"(k0), "r"(l_idx),
                               "r"(((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF)), "l"(0x12F0000000000000ULL) : "memory");
                        tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 24576, (&B), b_col, k0, l_idx, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        if (cta_rank == 0) {
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(49152)) : "memory");
                        }
                        load_stage += 1;
                        if (load_stage == 9) { load_stage = 0; _phase_mma_done ^= 1; }
                    }
                }
                #pragma unroll
                for (int _drain = 0; _drain < 9; _drain++) {
                    mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                    load_stage += 1;
                    if (load_stage == 9) { load_stage = 0; _phase_mma_done ^= 1; }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            unsigned int mma_stage = 0;
            unsigned int mma_acc_buf = 0;
            int m_tiles_m = (M + 256 - 1) / 256;
            int tiles_l_m = m_tiles_m * ((N + 128 - 1) / 128);
            int num_tiles_m = num_l * tiles_l_m;
            int k_iters_m = (K + 64 - 1) / 64;
            unsigned int _phase_acc_empty = 1;
            unsigned int _phase_tma_full = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (unsigned int item_1 = cluster_id; item_1 < num_tiles_m * num_splits; item_1 += num_clusters) {
                    int split_1 = (int)item_1 / num_tiles_m;
                    int tile_1 = (int)item_1 - split_1 * num_tiles_m;
                    int l_idx_1 = tile_1 / tiles_l_m;
                    int in_l_1 = tile_1 - l_idx_1 * tiles_l_m;
                    int n_tile_1 = in_l_1 / m_tiles_m;
                    int m_tile_1 = in_l_1 - n_tile_1 * m_tiles_m;
                    int k_begin_1 = split_1 * k_iters_m / num_splits;
                    int k_end_1 = (split_1 + 1) * k_iters_m / num_splits;
                    mbarrier_wait(acc_empty_addr + (mma_acc_buf) * 8, _phase_acc_empty);
                    #pragma unroll 1
                    for (int k_block_1 = k_begin_1; k_block_1 < k_end_1; k_block_1++) {
                        mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int init_flag = ((k_block_1 == k_begin_1) ? 1 : 0);
#if __CUDA_ARCH__ == 1070
                        int a_lo = ((((smem_a_addr) >> 4) & 0x7FFF) | 0x2000000) + (mma_stage) * 1536;
                        int b_lo = ((((smem_b_addr) >> 4) & 0x7FFF) | 0x2000000) + (mma_stage) * 1536;
#else
                        int a_lo = ((((smem_a_addr) >> 4) & 0x3FFF) | 0x2000000) + (mma_stage) * 1536;
                        int b_lo = ((((smem_b_addr) >> 4) & 0x3FFF) | 0x2000000) + (mma_stage) * 1536;
#endif
                        uint32_t _elected_0 = elect_sync();
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
                    "mov.b32 id, 270632080;\n\t"
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
                    :: "r"(a_lo), "r"(b_lo), "r"((tmem_accum + (mma_acc_buf * 256))), "r"(((init_flag) ? 0 : 1)), "r"(_elected_0) : "memory");
                        tcgen05_commit_cg2_multicast_leader(mma_done_addr + (mma_stage) * 8, (uint16_t)(3), _elected_0);
                        mma_stage += 1;
                        if (mma_stage == 9) { mma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    elect_commit_cg2_multicast(acc_full_addr + (mma_acc_buf) * 8, (uint16_t)(3));
                    mma_acc_buf += 1;
                    if (mma_acc_buf == 2) { mma_acc_buf = 0; _phase_acc_empty ^= 1; }
                }
                #pragma unroll
                for (int _drain_acc = 0; _drain_acc < 2; _drain_acc++) {
                    mbarrier_wait(acc_empty_addr + (mma_acc_buf) * 8, _phase_acc_empty);
                    mma_acc_buf += 1;
                    if (mma_acc_buf == 2) { mma_acc_buf = 0; _phase_acc_empty ^= 1; }
                }
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(tmem_free_addr), "r"(1) : "memory");
            }
            if (cta_rank != 0) {
                mbarrier_wait_cluster_hint(tmem_free_addr, 0, 10000000);
            }
            asm volatile("tcgen05.fence::after_thread_sync;");
            asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 2 && warp <= 9) {
        { // epilogue_main
            unsigned int acc_buf = 0;
            const int quarter = warp % 4;
            const int half = (warp - 2) / 4;
            int m_tiles_e = (M + 256 - 1) / 256;
            int tiles_l_e = m_tiles_e * ((N + 128 - 1) / 128);
            int num_tiles_e = num_l * tiles_l_e;
            int k_iters_e = (K + 64 - 1) / 64;
            int slice_id = cta_rank * 8 + (warp - 2);
            unsigned int last_ticket = (unsigned int)(num_splits - 1);
            float vals[64];
            float part[32];
            unsigned int _phase_acc_full = 0;
            #pragma unroll 1
            for (unsigned int item_2 = cluster_id; item_2 < num_tiles_e * num_splits; item_2 += num_clusters) {
                int split_2 = (int)item_2 / num_tiles_e;
                int tile_2 = (int)item_2 - split_2 * num_tiles_e;
                int l_idx_2 = tile_2 / tiles_l_e;
                int in_l_2 = tile_2 - l_idx_2 * tiles_l_e;
                int n_tile_2 = in_l_2 / m_tiles_e;
                int m_tile_2 = in_l_2 - n_tile_2 * m_tiles_e;
                int k_begin_2 = split_2 * k_iters_e / num_splits;
                int k_end_2 = (split_2 + 1) * k_iters_e / num_splits;
                int row = m_tile_2 * 256 + cta_rank * 128 + quarter * 32 + lane;
                int col_base = n_tile_2 * 128 + half * 64;
                unsigned int lane_addr = taddr + (unsigned int)(quarter * 32 << 16) + acc_buf * 256 + (unsigned int)(half * 64);
                mbarrier_wait(acc_full_addr + (acc_buf) * 8, _phase_acc_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(vals[0]), "=f"(vals[1]), "=f"(vals[2]), "=f"(vals[3]), "=f"(vals[4]), "=f"(vals[5]), "=f"(vals[6]), "=f"(vals[7]), "=f"(vals[8]), "=f"(vals[9]), "=f"(vals[10]), "=f"(vals[11]), "=f"(vals[12]), "=f"(vals[13]), "=f"(vals[14]), "=f"(vals[15]), "=f"(vals[16]), "=f"(vals[17]), "=f"(vals[18]), "=f"(vals[19]), "=f"(vals[20]), "=f"(vals[21]), "=f"(vals[22]), "=f"(vals[23]), "=f"(vals[24]), "=f"(vals[25]), "=f"(vals[26]), "=f"(vals[27]), "=f"(vals[28]), "=f"(vals[29]), "=f"(vals[30]), "=f"(vals[31])
                    : "r"(lane_addr));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(vals[32]), "=f"(vals[33]), "=f"(vals[34]), "=f"(vals[35]), "=f"(vals[36]), "=f"(vals[37]), "=f"(vals[38]), "=f"(vals[39]), "=f"(vals[40]), "=f"(vals[41]), "=f"(vals[42]), "=f"(vals[43]), "=f"(vals[44]), "=f"(vals[45]), "=f"(vals[46]), "=f"(vals[47]), "=f"(vals[48]), "=f"(vals[49]), "=f"(vals[50]), "=f"(vals[51]), "=f"(vals[52]), "=f"(vals[53]), "=f"(vals[54]), "=f"(vals[55]), "=f"(vals[56]), "=f"(vals[57]), "=f"(vals[58]), "=f"(vals[59]), "=f"(vals[60]), "=f"(vals[61]), "=f"(vals[62]), "=f"(vals[63])
                    : "r"(lane_addr + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                if (elect_sync()) {
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((acc_empty_addr + (acc_buf) * 8) & 0xFEFFFFFF) : "memory");
                }
                if (num_splits == 1) {
                    long long base_t = (long long)l_idx_2 * (long long)out_l + (long long)row;
                    if (row < M) {
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)col_base * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[0])), "r"((unsigned int)(col_base < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col_base + 1) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[1])), "r"((unsigned int)(col_base + 1 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col_base + 2) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[2])), "r"((unsigned int)(col_base + 2 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col_base + 3) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[3])), "r"((unsigned int)(col_base + 3 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col_base + 4) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[4])), "r"((unsigned int)(col_base + 4 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col_base + 5) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[5])), "r"((unsigned int)(col_base + 5 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col_base + 6) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[6])), "r"((unsigned int)(col_base + 6 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col_base + 7) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[7])), "r"((unsigned int)(col_base + 7 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col_base + 8) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[8])), "r"((unsigned int)(col_base + 8 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col_base + 9) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[9])), "r"((unsigned int)(col_base + 9 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col_base + 10) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[10])), "r"((unsigned int)(col_base + 10 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col_base + 11) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[11])), "r"((unsigned int)(col_base + 11 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col_base + 12) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[12])), "r"((unsigned int)(col_base + 12 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col_base + 13) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[13])), "r"((unsigned int)(col_base + 13 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col_base + 14) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[14])), "r"((unsigned int)(col_base + 14 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col_base + 15) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[15])), "r"((unsigned int)(col_base + 15 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col_base + 16) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[16])), "r"((unsigned int)(col_base + 16 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col_base + 17) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[17])), "r"((unsigned int)(col_base + 17 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col_base + 18) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[18])), "r"((unsigned int)(col_base + 18 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col_base + 19) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[19])), "r"((unsigned int)(col_base + 19 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col_base + 20) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[20])), "r"((unsigned int)(col_base + 20 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col_base + 21) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[21])), "r"((unsigned int)(col_base + 21 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col_base + 22) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[22])), "r"((unsigned int)(col_base + 22 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col_base + 23) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[23])), "r"((unsigned int)(col_base + 23 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col_base + 24) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[24])), "r"((unsigned int)(col_base + 24 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col_base + 25) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[25])), "r"((unsigned int)(col_base + 25 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col_base + 26) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[26])), "r"((unsigned int)(col_base + 26 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col_base + 27) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[27])), "r"((unsigned int)(col_base + 27 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col_base + 28) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[28])), "r"((unsigned int)(col_base + 28 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col_base + 29) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[29])), "r"((unsigned int)(col_base + 29 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col_base + 30) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[30])), "r"((unsigned int)(col_base + 30 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t + (long long)(col_base + 31) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[31])), "r"((unsigned int)(col_base + 31 < N)) : "memory");
                    }
                    long long base_t_0 = (long long)l_idx_2 * (long long)out_l + (long long)row;
                    if (row < M) {
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_0 + (long long)(col_base + 32) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[32])), "r"((unsigned int)(col_base + 32 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_0 + (long long)(col_base + 32 + 1) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[33])), "r"((unsigned int)(col_base + 32 + 1 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_0 + (long long)(col_base + 32 + 2) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[34])), "r"((unsigned int)(col_base + 32 + 2 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_0 + (long long)(col_base + 32 + 3) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[35])), "r"((unsigned int)(col_base + 32 + 3 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_0 + (long long)(col_base + 32 + 4) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[36])), "r"((unsigned int)(col_base + 32 + 4 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_0 + (long long)(col_base + 32 + 5) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[37])), "r"((unsigned int)(col_base + 32 + 5 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_0 + (long long)(col_base + 32 + 6) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[38])), "r"((unsigned int)(col_base + 32 + 6 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_0 + (long long)(col_base + 32 + 7) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[39])), "r"((unsigned int)(col_base + 32 + 7 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_0 + (long long)(col_base + 32 + 8) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[40])), "r"((unsigned int)(col_base + 32 + 8 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_0 + (long long)(col_base + 32 + 9) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[41])), "r"((unsigned int)(col_base + 32 + 9 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_0 + (long long)(col_base + 32 + 10) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[42])), "r"((unsigned int)(col_base + 32 + 10 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_0 + (long long)(col_base + 32 + 11) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[43])), "r"((unsigned int)(col_base + 32 + 11 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_0 + (long long)(col_base + 32 + 12) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[44])), "r"((unsigned int)(col_base + 32 + 12 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_0 + (long long)(col_base + 32 + 13) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[45])), "r"((unsigned int)(col_base + 32 + 13 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_0 + (long long)(col_base + 32 + 14) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[46])), "r"((unsigned int)(col_base + 32 + 14 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_0 + (long long)(col_base + 32 + 15) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[47])), "r"((unsigned int)(col_base + 32 + 15 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_0 + (long long)(col_base + 32 + 16) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[48])), "r"((unsigned int)(col_base + 32 + 16 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_0 + (long long)(col_base + 32 + 17) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[49])), "r"((unsigned int)(col_base + 32 + 17 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_0 + (long long)(col_base + 32 + 18) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[50])), "r"((unsigned int)(col_base + 32 + 18 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_0 + (long long)(col_base + 32 + 19) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[51])), "r"((unsigned int)(col_base + 32 + 19 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_0 + (long long)(col_base + 32 + 20) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[52])), "r"((unsigned int)(col_base + 32 + 20 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_0 + (long long)(col_base + 32 + 21) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[53])), "r"((unsigned int)(col_base + 32 + 21 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_0 + (long long)(col_base + 32 + 22) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[54])), "r"((unsigned int)(col_base + 32 + 22 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_0 + (long long)(col_base + 32 + 23) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[55])), "r"((unsigned int)(col_base + 32 + 23 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_0 + (long long)(col_base + 32 + 24) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[56])), "r"((unsigned int)(col_base + 32 + 24 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_0 + (long long)(col_base + 32 + 25) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[57])), "r"((unsigned int)(col_base + 32 + 25 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_0 + (long long)(col_base + 32 + 26) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[58])), "r"((unsigned int)(col_base + 32 + 26 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_0 + (long long)(col_base + 32 + 27) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[59])), "r"((unsigned int)(col_base + 32 + 27 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_0 + (long long)(col_base + 32 + 28) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[60])), "r"((unsigned int)(col_base + 32 + 28 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_0 + (long long)(col_base + 32 + 29) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[61])), "r"((unsigned int)(col_base + 32 + 29 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_0 + (long long)(col_base + 32 + 30) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[62])), "r"((unsigned int)(col_base + 32 + 30 < N)) : "memory");
                        asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_0 + (long long)(col_base + 32 + 31) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[63])), "r"((unsigned int)(col_base + 32 + 31 < N)) : "memory");
                    }
                } else {
                    long long slot_floats = 32768;
                    long long slice_base = (long long)tile_2 * (long long)num_splits * slot_floats + (long long)(slice_id * 2048 + lane * 4);
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.f32 [%0], {%1, %2, %3, %4}; }" :: "l"(ws + (slice_base + (long long)split_2 * slot_floats)), "f"(vals[(0) + 0]), "f"(vals[(0) + 1]), "f"(vals[(0) + 2]), "f"(vals[(0) + 3]), "r"((unsigned int)(col_base < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.f32 [%0], {%1, %2, %3, %4}; }" :: "l"(ws + (slice_base + (long long)split_2 * slot_floats + 128)), "f"(vals[(4) + 0]), "f"(vals[(4) + 1]), "f"(vals[(4) + 2]), "f"(vals[(4) + 3]), "r"((unsigned int)(col_base + 4 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.f32 [%0], {%1, %2, %3, %4}; }" :: "l"(ws + (slice_base + (long long)split_2 * slot_floats + 256)), "f"(vals[(8) + 0]), "f"(vals[(8) + 1]), "f"(vals[(8) + 2]), "f"(vals[(8) + 3]), "r"((unsigned int)(col_base + 8 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.f32 [%0], {%1, %2, %3, %4}; }" :: "l"(ws + (slice_base + (long long)split_2 * slot_floats + 384)), "f"(vals[(12) + 0]), "f"(vals[(12) + 1]), "f"(vals[(12) + 2]), "f"(vals[(12) + 3]), "r"((unsigned int)(col_base + 12 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.f32 [%0], {%1, %2, %3, %4}; }" :: "l"(ws + (slice_base + (long long)split_2 * slot_floats + 512)), "f"(vals[(16) + 0]), "f"(vals[(16) + 1]), "f"(vals[(16) + 2]), "f"(vals[(16) + 3]), "r"((unsigned int)(col_base + 16 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.f32 [%0], {%1, %2, %3, %4}; }" :: "l"(ws + (slice_base + (long long)split_2 * slot_floats + 640)), "f"(vals[(20) + 0]), "f"(vals[(20) + 1]), "f"(vals[(20) + 2]), "f"(vals[(20) + 3]), "r"((unsigned int)(col_base + 20 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.f32 [%0], {%1, %2, %3, %4}; }" :: "l"(ws + (slice_base + (long long)split_2 * slot_floats + 768)), "f"(vals[(24) + 0]), "f"(vals[(24) + 1]), "f"(vals[(24) + 2]), "f"(vals[(24) + 3]), "r"((unsigned int)(col_base + 24 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.f32 [%0], {%1, %2, %3, %4}; }" :: "l"(ws + (slice_base + (long long)split_2 * slot_floats + 896)), "f"(vals[(28) + 0]), "f"(vals[(28) + 1]), "f"(vals[(28) + 2]), "f"(vals[(28) + 3]), "r"((unsigned int)(col_base + 28 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.f32 [%0], {%1, %2, %3, %4}; }" :: "l"(ws + (slice_base + (long long)split_2 * slot_floats + 1024)), "f"(vals[(32) + 0]), "f"(vals[(32) + 1]), "f"(vals[(32) + 2]), "f"(vals[(32) + 3]), "r"((unsigned int)(col_base + 32 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.f32 [%0], {%1, %2, %3, %4}; }" :: "l"(ws + (slice_base + (long long)split_2 * slot_floats + 1152)), "f"(vals[(36) + 0]), "f"(vals[(36) + 1]), "f"(vals[(36) + 2]), "f"(vals[(36) + 3]), "r"((unsigned int)(col_base + 32 + 4 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.f32 [%0], {%1, %2, %3, %4}; }" :: "l"(ws + (slice_base + (long long)split_2 * slot_floats + 1280)), "f"(vals[(40) + 0]), "f"(vals[(40) + 1]), "f"(vals[(40) + 2]), "f"(vals[(40) + 3]), "r"((unsigned int)(col_base + 32 + 8 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.f32 [%0], {%1, %2, %3, %4}; }" :: "l"(ws + (slice_base + (long long)split_2 * slot_floats + 1408)), "f"(vals[(44) + 0]), "f"(vals[(44) + 1]), "f"(vals[(44) + 2]), "f"(vals[(44) + 3]), "r"((unsigned int)(col_base + 32 + 12 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.f32 [%0], {%1, %2, %3, %4}; }" :: "l"(ws + (slice_base + (long long)split_2 * slot_floats + 1536)), "f"(vals[(48) + 0]), "f"(vals[(48) + 1]), "f"(vals[(48) + 2]), "f"(vals[(48) + 3]), "r"((unsigned int)(col_base + 32 + 16 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.f32 [%0], {%1, %2, %3, %4}; }" :: "l"(ws + (slice_base + (long long)split_2 * slot_floats + 1664)), "f"(vals[(52) + 0]), "f"(vals[(52) + 1]), "f"(vals[(52) + 2]), "f"(vals[(52) + 3]), "r"((unsigned int)(col_base + 32 + 20 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.f32 [%0], {%1, %2, %3, %4}; }" :: "l"(ws + (slice_base + (long long)split_2 * slot_floats + 1792)), "f"(vals[(56) + 0]), "f"(vals[(56) + 1]), "f"(vals[(56) + 2]), "f"(vals[(56) + 3]), "r"((unsigned int)(col_base + 32 + 24 < N)) : "memory");
                    asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %5, 0; @p_ st.global.v4.f32 [%0], {%1, %2, %3, %4}; }" :: "l"(ws + (slice_base + (long long)split_2 * slot_floats + 1920)), "f"(vals[(60) + 0]), "f"(vals[(60) + 1]), "f"(vals[(60) + 2]), "f"(vals[(60) + 3]), "r"((unsigned int)(col_base + 32 + 28 < N)) : "memory");
                    uint32_t arrival;
                    asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                        : "=r"(arrival) : "l"(&counters[(tile_2 * 16 + slice_id) * 32 + lane]), "r"(static_cast<uint32_t>(last_ticket)) : "memory");
                    if (arrival == last_ticket) {
                        if (col_base < N) {
                            #pragma unroll 4
                            for (int step_0 = 1; step_0 < num_splits; step_0++) {
                                int ahead_0 = split_2 + step_0;
                                int other_0 = ((ahead_0 < num_splits) ? ahead_0 : ahead_0 - num_splits);
                                {
                                    float4 _v4 = *reinterpret_cast<const float4*>(ws + (slice_base + (long long)other_0 * slot_floats) + 0);
                                    part[0 + 0] = _v4.x;
                                    part[0 + 1] = _v4.y;
                                    part[0 + 2] = _v4.z;
                                    part[0 + 3] = _v4.w;
                                }
                                {
                                    float4 _v4 = *reinterpret_cast<const float4*>(ws + (slice_base + (long long)other_0 * slot_floats + 128) + 0);
                                    part[4 + 0] = _v4.x;
                                    part[4 + 1] = _v4.y;
                                    part[4 + 2] = _v4.z;
                                    part[4 + 3] = _v4.w;
                                }
                                {
                                    float4 _v4 = *reinterpret_cast<const float4*>(ws + (slice_base + (long long)other_0 * slot_floats + 256) + 0);
                                    part[8 + 0] = _v4.x;
                                    part[8 + 1] = _v4.y;
                                    part[8 + 2] = _v4.z;
                                    part[8 + 3] = _v4.w;
                                }
                                {
                                    float4 _v4 = *reinterpret_cast<const float4*>(ws + (slice_base + (long long)other_0 * slot_floats + 384) + 0);
                                    part[12 + 0] = _v4.x;
                                    part[12 + 1] = _v4.y;
                                    part[12 + 2] = _v4.z;
                                    part[12 + 3] = _v4.w;
                                }
                                {
                                    float4 _v4 = *reinterpret_cast<const float4*>(ws + (slice_base + (long long)other_0 * slot_floats + 512) + 0);
                                    part[16 + 0] = _v4.x;
                                    part[16 + 1] = _v4.y;
                                    part[16 + 2] = _v4.z;
                                    part[16 + 3] = _v4.w;
                                }
                                {
                                    float4 _v4 = *reinterpret_cast<const float4*>(ws + (slice_base + (long long)other_0 * slot_floats + 640) + 0);
                                    part[20 + 0] = _v4.x;
                                    part[20 + 1] = _v4.y;
                                    part[20 + 2] = _v4.z;
                                    part[20 + 3] = _v4.w;
                                }
                                {
                                    float4 _v4 = *reinterpret_cast<const float4*>(ws + (slice_base + (long long)other_0 * slot_floats + 768) + 0);
                                    part[24 + 0] = _v4.x;
                                    part[24 + 1] = _v4.y;
                                    part[24 + 2] = _v4.z;
                                    part[24 + 3] = _v4.w;
                                }
                                {
                                    float4 _v4 = *reinterpret_cast<const float4*>(ws + (slice_base + (long long)other_0 * slot_floats + 896) + 0);
                                    part[28 + 0] = _v4.x;
                                    part[28 + 1] = _v4.y;
                                    part[28 + 2] = _v4.z;
                                    part[28 + 3] = _v4.w;
                                }
                                vals[0] = vals[0] + part[0];
                                vals[1] = vals[1] + part[1];
                                vals[2] = vals[2] + part[2];
                                vals[3] = vals[3] + part[3];
                                vals[4] = vals[4] + part[4];
                                vals[5] = vals[5] + part[5];
                                vals[6] = vals[6] + part[6];
                                vals[7] = vals[7] + part[7];
                                vals[8] = vals[8] + part[8];
                                vals[9] = vals[9] + part[9];
                                vals[10] = vals[10] + part[10];
                                vals[11] = vals[11] + part[11];
                                vals[12] = vals[12] + part[12];
                                vals[13] = vals[13] + part[13];
                                vals[14] = vals[14] + part[14];
                                vals[15] = vals[15] + part[15];
                                vals[16] = vals[16] + part[16];
                                vals[17] = vals[17] + part[17];
                                vals[18] = vals[18] + part[18];
                                vals[19] = vals[19] + part[19];
                                vals[20] = vals[20] + part[20];
                                vals[21] = vals[21] + part[21];
                                vals[22] = vals[22] + part[22];
                                vals[23] = vals[23] + part[23];
                                vals[24] = vals[24] + part[24];
                                vals[25] = vals[25] + part[25];
                                vals[26] = vals[26] + part[26];
                                vals[27] = vals[27] + part[27];
                                vals[28] = vals[28] + part[28];
                                vals[29] = vals[29] + part[29];
                                vals[30] = vals[30] + part[30];
                                vals[31] = vals[31] + part[31];
                            }
                        }
                        if (col_base + 32 < N) {
                            #pragma unroll 4
                            for (int step_1 = 1; step_1 < num_splits; step_1++) {
                                int ahead_1 = split_2 + step_1;
                                int other_1 = ((ahead_1 < num_splits) ? ahead_1 : ahead_1 - num_splits);
                                {
                                    float4 _v4 = *reinterpret_cast<const float4*>(ws + (slice_base + (long long)other_1 * slot_floats + 1024) + 0);
                                    part[0 + 0] = _v4.x;
                                    part[0 + 1] = _v4.y;
                                    part[0 + 2] = _v4.z;
                                    part[0 + 3] = _v4.w;
                                }
                                {
                                    float4 _v4 = *reinterpret_cast<const float4*>(ws + (slice_base + (long long)other_1 * slot_floats + 1152) + 0);
                                    part[4 + 0] = _v4.x;
                                    part[4 + 1] = _v4.y;
                                    part[4 + 2] = _v4.z;
                                    part[4 + 3] = _v4.w;
                                }
                                {
                                    float4 _v4 = *reinterpret_cast<const float4*>(ws + (slice_base + (long long)other_1 * slot_floats + 1280) + 0);
                                    part[8 + 0] = _v4.x;
                                    part[8 + 1] = _v4.y;
                                    part[8 + 2] = _v4.z;
                                    part[8 + 3] = _v4.w;
                                }
                                {
                                    float4 _v4 = *reinterpret_cast<const float4*>(ws + (slice_base + (long long)other_1 * slot_floats + 1408) + 0);
                                    part[12 + 0] = _v4.x;
                                    part[12 + 1] = _v4.y;
                                    part[12 + 2] = _v4.z;
                                    part[12 + 3] = _v4.w;
                                }
                                {
                                    float4 _v4 = *reinterpret_cast<const float4*>(ws + (slice_base + (long long)other_1 * slot_floats + 1536) + 0);
                                    part[16 + 0] = _v4.x;
                                    part[16 + 1] = _v4.y;
                                    part[16 + 2] = _v4.z;
                                    part[16 + 3] = _v4.w;
                                }
                                {
                                    float4 _v4 = *reinterpret_cast<const float4*>(ws + (slice_base + (long long)other_1 * slot_floats + 1664) + 0);
                                    part[20 + 0] = _v4.x;
                                    part[20 + 1] = _v4.y;
                                    part[20 + 2] = _v4.z;
                                    part[20 + 3] = _v4.w;
                                }
                                {
                                    float4 _v4 = *reinterpret_cast<const float4*>(ws + (slice_base + (long long)other_1 * slot_floats + 1792) + 0);
                                    part[24 + 0] = _v4.x;
                                    part[24 + 1] = _v4.y;
                                    part[24 + 2] = _v4.z;
                                    part[24 + 3] = _v4.w;
                                }
                                {
                                    float4 _v4 = *reinterpret_cast<const float4*>(ws + (slice_base + (long long)other_1 * slot_floats + 1920) + 0);
                                    part[28 + 0] = _v4.x;
                                    part[28 + 1] = _v4.y;
                                    part[28 + 2] = _v4.z;
                                    part[28 + 3] = _v4.w;
                                }
                                vals[32] = vals[32] + part[0];
                                vals[33] = vals[33] + part[1];
                                vals[34] = vals[34] + part[2];
                                vals[35] = vals[35] + part[3];
                                vals[36] = vals[36] + part[4];
                                vals[37] = vals[37] + part[5];
                                vals[38] = vals[38] + part[6];
                                vals[39] = vals[39] + part[7];
                                vals[40] = vals[40] + part[8];
                                vals[41] = vals[41] + part[9];
                                vals[42] = vals[42] + part[10];
                                vals[43] = vals[43] + part[11];
                                vals[44] = vals[44] + part[12];
                                vals[45] = vals[45] + part[13];
                                vals[46] = vals[46] + part[14];
                                vals[47] = vals[47] + part[15];
                                vals[48] = vals[48] + part[16];
                                vals[49] = vals[49] + part[17];
                                vals[50] = vals[50] + part[18];
                                vals[51] = vals[51] + part[19];
                                vals[52] = vals[52] + part[20];
                                vals[53] = vals[53] + part[21];
                                vals[54] = vals[54] + part[22];
                                vals[55] = vals[55] + part[23];
                                vals[56] = vals[56] + part[24];
                                vals[57] = vals[57] + part[25];
                                vals[58] = vals[58] + part[26];
                                vals[59] = vals[59] + part[27];
                                vals[60] = vals[60] + part[28];
                                vals[61] = vals[61] + part[29];
                                vals[62] = vals[62] + part[30];
                                vals[63] = vals[63] + part[31];
                            }
                        }
                        long long base_t_1 = (long long)l_idx_2 * (long long)out_l + (long long)row;
                        if (row < M) {
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1 + (long long)col_base * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[0])), "r"((unsigned int)(col_base < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1 + (long long)(col_base + 1) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[1])), "r"((unsigned int)(col_base + 1 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1 + (long long)(col_base + 2) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[2])), "r"((unsigned int)(col_base + 2 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1 + (long long)(col_base + 3) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[3])), "r"((unsigned int)(col_base + 3 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1 + (long long)(col_base + 4) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[4])), "r"((unsigned int)(col_base + 4 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1 + (long long)(col_base + 5) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[5])), "r"((unsigned int)(col_base + 5 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1 + (long long)(col_base + 6) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[6])), "r"((unsigned int)(col_base + 6 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1 + (long long)(col_base + 7) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[7])), "r"((unsigned int)(col_base + 7 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1 + (long long)(col_base + 8) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[8])), "r"((unsigned int)(col_base + 8 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1 + (long long)(col_base + 9) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[9])), "r"((unsigned int)(col_base + 9 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1 + (long long)(col_base + 10) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[10])), "r"((unsigned int)(col_base + 10 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1 + (long long)(col_base + 11) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[11])), "r"((unsigned int)(col_base + 11 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1 + (long long)(col_base + 12) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[12])), "r"((unsigned int)(col_base + 12 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1 + (long long)(col_base + 13) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[13])), "r"((unsigned int)(col_base + 13 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1 + (long long)(col_base + 14) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[14])), "r"((unsigned int)(col_base + 14 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1 + (long long)(col_base + 15) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[15])), "r"((unsigned int)(col_base + 15 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1 + (long long)(col_base + 16) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[16])), "r"((unsigned int)(col_base + 16 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1 + (long long)(col_base + 17) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[17])), "r"((unsigned int)(col_base + 17 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1 + (long long)(col_base + 18) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[18])), "r"((unsigned int)(col_base + 18 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1 + (long long)(col_base + 19) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[19])), "r"((unsigned int)(col_base + 19 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1 + (long long)(col_base + 20) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[20])), "r"((unsigned int)(col_base + 20 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1 + (long long)(col_base + 21) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[21])), "r"((unsigned int)(col_base + 21 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1 + (long long)(col_base + 22) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[22])), "r"((unsigned int)(col_base + 22 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1 + (long long)(col_base + 23) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[23])), "r"((unsigned int)(col_base + 23 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1 + (long long)(col_base + 24) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[24])), "r"((unsigned int)(col_base + 24 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1 + (long long)(col_base + 25) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[25])), "r"((unsigned int)(col_base + 25 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1 + (long long)(col_base + 26) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[26])), "r"((unsigned int)(col_base + 26 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1 + (long long)(col_base + 27) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[27])), "r"((unsigned int)(col_base + 27 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1 + (long long)(col_base + 28) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[28])), "r"((unsigned int)(col_base + 28 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1 + (long long)(col_base + 29) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[29])), "r"((unsigned int)(col_base + 29 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1 + (long long)(col_base + 30) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[30])), "r"((unsigned int)(col_base + 30 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1 + (long long)(col_base + 31) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[31])), "r"((unsigned int)(col_base + 31 < N)) : "memory");
                        }
                        long long base_t_1_1 = (long long)l_idx_2 * (long long)out_l + (long long)row;
                        if (row < M) {
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1_1 + (long long)(col_base + 32) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[32])), "r"((unsigned int)(col_base + 32 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1_1 + (long long)(col_base + 32 + 1) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[33])), "r"((unsigned int)(col_base + 32 + 1 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1_1 + (long long)(col_base + 32 + 2) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[34])), "r"((unsigned int)(col_base + 32 + 2 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1_1 + (long long)(col_base + 32 + 3) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[35])), "r"((unsigned int)(col_base + 32 + 3 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1_1 + (long long)(col_base + 32 + 4) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[36])), "r"((unsigned int)(col_base + 32 + 4 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1_1 + (long long)(col_base + 32 + 5) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[37])), "r"((unsigned int)(col_base + 32 + 5 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1_1 + (long long)(col_base + 32 + 6) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[38])), "r"((unsigned int)(col_base + 32 + 6 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1_1 + (long long)(col_base + 32 + 7) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[39])), "r"((unsigned int)(col_base + 32 + 7 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1_1 + (long long)(col_base + 32 + 8) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[40])), "r"((unsigned int)(col_base + 32 + 8 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1_1 + (long long)(col_base + 32 + 9) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[41])), "r"((unsigned int)(col_base + 32 + 9 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1_1 + (long long)(col_base + 32 + 10) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[42])), "r"((unsigned int)(col_base + 32 + 10 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1_1 + (long long)(col_base + 32 + 11) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[43])), "r"((unsigned int)(col_base + 32 + 11 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1_1 + (long long)(col_base + 32 + 12) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[44])), "r"((unsigned int)(col_base + 32 + 12 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1_1 + (long long)(col_base + 32 + 13) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[45])), "r"((unsigned int)(col_base + 32 + 13 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1_1 + (long long)(col_base + 32 + 14) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[46])), "r"((unsigned int)(col_base + 32 + 14 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1_1 + (long long)(col_base + 32 + 15) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[47])), "r"((unsigned int)(col_base + 32 + 15 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1_1 + (long long)(col_base + 32 + 16) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[48])), "r"((unsigned int)(col_base + 32 + 16 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1_1 + (long long)(col_base + 32 + 17) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[49])), "r"((unsigned int)(col_base + 32 + 17 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1_1 + (long long)(col_base + 32 + 18) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[50])), "r"((unsigned int)(col_base + 32 + 18 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1_1 + (long long)(col_base + 32 + 19) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[51])), "r"((unsigned int)(col_base + 32 + 19 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1_1 + (long long)(col_base + 32 + 20) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[52])), "r"((unsigned int)(col_base + 32 + 20 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1_1 + (long long)(col_base + 32 + 21) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[53])), "r"((unsigned int)(col_base + 32 + 21 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1_1 + (long long)(col_base + 32 + 22) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[54])), "r"((unsigned int)(col_base + 32 + 22 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1_1 + (long long)(col_base + 32 + 23) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[55])), "r"((unsigned int)(col_base + 32 + 23 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1_1 + (long long)(col_base + 32 + 24) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[56])), "r"((unsigned int)(col_base + 32 + 24 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1_1 + (long long)(col_base + 32 + 25) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[57])), "r"((unsigned int)(col_base + 32 + 25 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1_1 + (long long)(col_base + 32 + 26) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[58])), "r"((unsigned int)(col_base + 32 + 26 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1_1 + (long long)(col_base + 32 + 27) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[59])), "r"((unsigned int)(col_base + 32 + 27 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1_1 + (long long)(col_base + 32 + 28) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[60])), "r"((unsigned int)(col_base + 32 + 28 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1_1 + (long long)(col_base + 32 + 29) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[61])), "r"((unsigned int)(col_base + 32 + 29 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1_1 + (long long)(col_base + 32 + 30) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[62])), "r"((unsigned int)(col_base + 32 + 30 < N)) : "memory");
                            asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + (base_t_1_1 + (long long)(col_base + 32 + 31) * (long long)ldo))), "h"(__bfloat16_as_ushort((__nv_bfloat16)vals[63])), "r"((unsigned int)(col_base + 32 + 31 < N)) : "memory");
                        }
                    }
                }
                acc_buf += 1;
                if (acc_buf == 2) { acc_buf = 0; _phase_acc_full ^= 1; }
            }
        }
    }

    // Cleanup
}

} // extern "C"
