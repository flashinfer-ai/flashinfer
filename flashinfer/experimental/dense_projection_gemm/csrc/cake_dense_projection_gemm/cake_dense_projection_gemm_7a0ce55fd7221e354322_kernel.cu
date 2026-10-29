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
#define TMEM_CHUNK_ACC_OFFSET 0
#define TMEM_SMALL_ACC_OFFSET 256
#define NUM_TMA_PIPE_STAGES 4
#define NUM_CHUNK_PIPE_STAGES 2
#define NUM_MAINLOOP_PIPE_STAGES 2
#define NUM_WORK_PIPE_STAGES 4
#define SMEM_A32_OFF 1024
#define SMEM_A32_STAGE_BYTES 16384
#define SMEM_A32_STRIDE 53248
#define SMEM_A1_OFF 17408
#define SMEM_A1_STAGE_BYTES 8192
#define SMEM_A1_STRIDE 53248
#define SMEM_A2_OFF 25600
#define SMEM_A2_STAGE_BYTES 8192
#define SMEM_A2_STRIDE 53248
#define SMEM_A3_OFF 33792
#define SMEM_A3_STAGE_BYTES 8192
#define SMEM_A3_STRIDE 53248
#define SMEM_B1_OFF 41984
#define SMEM_B1_STAGE_BYTES 4096
#define SMEM_B1_STRIDE 53248
#define SMEM_B2_OFF 46080
#define SMEM_B2_STAGE_BYTES 4096
#define SMEM_B2_STRIDE 53248
#define SMEM_B3_OFF 50176
#define SMEM_B3_STAGE_BYTES 4096
#define SMEM_B3_STRIDE 53248
#define SMEM_WORK_RESPONSE_OFF 214016
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_TOTAL 214144
#define THREADS 448

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



__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
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

__global__ __launch_bounds__(448) __cluster_dims__(2,1,1) void
kernel_cake_dense_projection_gemm_7a0ce55fd7221e354322(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, float* __restrict__ out32, int M, int N, int m_tiles, int n_tiles, int k_iters_split, int ldo, int out_l, int num_cluster_tiles)
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
    #define raw_full_addr (mbar_base + 0)
    #define parts_full_addr (mbar_base + 32)
    #define mma_done_addr (mbar_base + 64)
    #define chunk_full_addr (mbar_base + 96)
    #define chunk_empty_addr (mbar_base + 112)
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
    float* a32 = reinterpret_cast<float*>(smem_raw + 1024);
    const int a32_addr = smem + 1024;
    __nv_bfloat16* a1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int a1_addr = smem + 17408;
    __nv_bfloat16* a2 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 25600);
    const int a2_addr = smem + 25600;
    __nv_bfloat16* a3 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int a3_addr = smem + 33792;
    __nv_bfloat16* b1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 41984);
    const int b1_addr = smem + 41984;
    __nv_bfloat16* b2 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 46080);
    const int b2_addr = smem + 46080;
    __nv_bfloat16* b3 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 50176);
    const int b3_addr = smem + 50176;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 214016);
    const int work_response_addr = smem + 214016;

    // Mbarrier init (9 pipeline groups, 0 ordered-sequence groups, 28 barriers)
    // Mbarriers at smem_raw[0..224)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // raw_full: 4 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // parts_full: 4 barriers, init_count=8
            mbarrier_init(smem + 32, 8);
            mbarrier_init(smem + 40, 8);
            mbarrier_init(smem + 48, 8);
            mbarrier_init(smem + 56, 8);
            // mma_done: 4 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // --- pipeline 'chunk_pipe' ---
            // chunk_full: 2 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            // chunk_empty: 2 barriers, init_count=16
            mbarrier_init(smem + 112, 16);
            mbarrier_init(smem + 120, 16);
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
            // work_empty: 4 barriers, init_count=802
            mbarrier_init(smem + 192, 802);
            mbarrier_init(smem + 200, 802);
            mbarrier_init(smem + 208, 802);
            mbarrier_init(smem + 216, 802);
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
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_chunk_acc = taddr;
    const int tmem_small_acc = taddr + 256;

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
                            :: "r"(work_full_addr + (work_stage) * 8), "r"(0), "r"((uint32_t)(16)) : "memory");
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                            "}"
                            :: "r"(work_full_addr + (work_stage) * 8), "r"(1), "r"((uint32_t)(16)) : "memory");
                        asm volatile(
                            "fence.proxy.async.shared::cta;\n\t"
                            "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                                " [%0], [%1];"
                            :: "r"(work_response_addr + work_stage * 16 + 0 * 16), "r"(work_full_addr + work_stage * 8)
                            : "memory");
                    }
                    int pair = this_bid / 2;
                    int rank = this_bid % 2;
                    int pairs_per_l = m_tiles / 2 * n_tiles;
                    int l = pair / pairs_per_l;
                    int prem = pair - l * pairs_per_l;
                    int pair_m = prem / n_tiles;
                    int bid_n = prem % n_tiles;
                    int off_m = (pair_m * 2 + rank) * 128;
                    int off_n = bid_n * 128;
                    int b_col = off_n + cta_rank * 64;
                    int k_first = l * k_iters_split;
                    #pragma unroll 1
                    for (int iter_k = 0; iter_k < k_iters_split; iter_k++) {
                        mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                        int k0 = (k_first + iter_k) * 32;
                        #pragma unroll
                        for (int r = 0; r < 4; r++) {
                            tma_3d_gmem2smem(a32_addr + load_stage * 53248 + (unsigned int)(r * 4096), (&A), k0, off_m + 32 * r, 0, raw_full_addr + (load_stage) * 8);
                        }
                        #pragma unroll
                        for (int p = 0; p < 2; p++) {
                            tma_3d_gmem2smem(b1_addr + load_stage * 53248 + (unsigned int)(p * 2048), (&B), b_col + 32 * p, k0, 0, raw_full_addr + (load_stage) * 8);
                            tma_3d_gmem2smem(b2_addr + load_stage * 53248 + (unsigned int)(p * 2048), (&B), b_col + 32 * p, k0, 1, raw_full_addr + (load_stage) * 8);
                            tma_3d_gmem2smem(b3_addr + load_stage * 53248 + (unsigned int)(p * 2048), (&B), b_col + 32 * p, k0, 2, raw_full_addr + (load_stage) * 8);
                        }
                        mbarrier_arrive_expect_tx(raw_full_addr + (load_stage) * 8, 28672);
                        load_stage += 1;
                        if (load_stage == 4) { load_stage = 0; _phase_mma_done ^= 1; }
                    }
                    mbarrier_wait(work_full_addr + (work_stage) * 8, _phase_work_full);
                    uint32_t clc_valid = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %0, 1, 0, p1;\n\t"
                        "}\n"
                        : "=r"(clc_valid)
                        : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                        : "memory");
                    uint32_t clc_ctaid = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "+r"(clc_ctaid)
                        : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(work_empty_addr + (work_stage) * 8), "r"(0) : "memory");
                    work_stage += 1;
                    if (work_stage == 4) { work_stage = 0; _phase_work_empty ^= 1; _phase_work_full ^= 1; }
                    if (clc_valid == 0) {
                        break;
                    }
                    this_bid = clc_ctaid + (unsigned int)cta_rank;
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            unsigned int mma_tma_stage = 0;
            unsigned int chunk_stage = 0;
            unsigned int mma_epi_stage = 0;
            unsigned int work_stage_1 = 0;
            unsigned int _phase_epilogue_done = 1;
            unsigned int _phase_chunk_empty = 1;
            unsigned int _phase_parts_full = 0;
            unsigned int _phase_work_full_1 = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (unsigned int _tile_iter_1 = 0; _tile_iter_1 < num_cluster_tiles; _tile_iter_1++) {
                    mbarrier_wait(epilogue_done_addr + (mma_epi_stage) * 8, _phase_epilogue_done);
                    #pragma unroll 1
                    for (int iter_k_1 = 0; iter_k_1 < k_iters_split; iter_k_1++) {
                        int kc = iter_k_1 % 2;
                        if (kc == 0) {
                            mbarrier_wait(chunk_empty_addr + (chunk_stage) * 8, _phase_chunk_empty);
                        }
                        mbarrier_wait(parts_full_addr + (mma_tma_stage) * 8, _phase_parts_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int big_init = ((kc == 0) ? 1 : 0);
                        int small_init = ((iter_k_1 == 0) ? 1 : 0);
#if __CUDA_ARCH__ == 1070
                        int _mma_a_lo_0 = (((a1_addr) >> 4) & 0x7FFF) + (mma_tma_stage) * 3328;
                        int _mma_b_lo_0 = ((((b1_addr) >> 4) & 0x7FFF) | 0x800000) + (mma_tma_stage) * 3328;
#else
                        int _mma_a_lo_0 = (((a1_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 3328;
                        int _mma_b_lo_0 = ((((b1_addr) >> 4) & 0x3FFF) | 0x800000) + (mma_tma_stage) * 3328;
#endif
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x80004020;\n\t"
                    "mov.b32 bdhi, 0x80004020;\n\t"
                    "mov.b32 id, 270599312;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_chunk_acc + (chunk_stage * 128))), "r"(((big_init) ? 0 : 1)) : "memory");
#if __CUDA_ARCH__ == 1070
                        int _mma_a_lo_1 = (((a1_addr) >> 4) & 0x7FFF) + (mma_tma_stage) * 3328;
                        int _mma_b_lo_1 = ((((b2_addr) >> 4) & 0x7FFF) | 0x800000) + (mma_tma_stage) * 3328;
#else
                        int _mma_a_lo_1 = (((a1_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 3328;
                        int _mma_b_lo_1 = ((((b2_addr) >> 4) & 0x3FFF) | 0x800000) + (mma_tma_stage) * 3328;
#endif
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x80004020;\n\t"
                    "mov.b32 bdhi, 0x80004020;\n\t"
                    "mov.b32 id, 270599312;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"((tmem_small_acc + (mma_epi_stage * 128))), "r"(((small_init) ? 0 : 1)) : "memory");
#if __CUDA_ARCH__ == 1070
                        int _mma_a_lo_2 = (((a2_addr) >> 4) & 0x7FFF) + (mma_tma_stage) * 3328;
                        int _mma_b_lo_2 = ((((b1_addr) >> 4) & 0x7FFF) | 0x800000) + (mma_tma_stage) * 3328;
#else
                        int _mma_a_lo_2 = (((a2_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 3328;
                        int _mma_b_lo_2 = ((((b1_addr) >> 4) & 0x3FFF) | 0x800000) + (mma_tma_stage) * 3328;
#endif
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x80004020;\n\t"
                    "mov.b32 bdhi, 0x80004020;\n\t"
                    "mov.b32 id, 270599312;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_2), "r"(_mma_b_lo_2), "r"((tmem_small_acc + (mma_epi_stage * 128))), "r"(1) : "memory");
#if __CUDA_ARCH__ == 1070
                        int _mma_a_lo_3 = (((a1_addr) >> 4) & 0x7FFF) + (mma_tma_stage) * 3328;
                        int _mma_b_lo_3 = ((((b3_addr) >> 4) & 0x7FFF) | 0x800000) + (mma_tma_stage) * 3328;
#else
                        int _mma_a_lo_3 = (((a1_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 3328;
                        int _mma_b_lo_3 = ((((b3_addr) >> 4) & 0x3FFF) | 0x800000) + (mma_tma_stage) * 3328;
#endif
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x80004020;\n\t"
                    "mov.b32 bdhi, 0x80004020;\n\t"
                    "mov.b32 id, 270599312;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_3), "r"(_mma_b_lo_3), "r"((tmem_small_acc + (mma_epi_stage * 128))), "r"(1) : "memory");
#if __CUDA_ARCH__ == 1070
                        int _mma_a_lo_4 = (((a2_addr) >> 4) & 0x7FFF) + (mma_tma_stage) * 3328;
                        int _mma_b_lo_4 = ((((b2_addr) >> 4) & 0x7FFF) | 0x800000) + (mma_tma_stage) * 3328;
#else
                        int _mma_a_lo_4 = (((a2_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 3328;
                        int _mma_b_lo_4 = ((((b2_addr) >> 4) & 0x3FFF) | 0x800000) + (mma_tma_stage) * 3328;
#endif
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x80004020;\n\t"
                    "mov.b32 bdhi, 0x80004020;\n\t"
                    "mov.b32 id, 270599312;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_4), "r"(_mma_b_lo_4), "r"((tmem_small_acc + (mma_epi_stage * 128))), "r"(1) : "memory");
#if __CUDA_ARCH__ == 1070
                        int _mma_a_lo_5 = (((a3_addr) >> 4) & 0x7FFF) + (mma_tma_stage) * 3328;
                        int _mma_b_lo_5 = ((((b1_addr) >> 4) & 0x7FFF) | 0x800000) + (mma_tma_stage) * 3328;
#else
                        int _mma_a_lo_5 = (((a3_addr) >> 4) & 0x3FFF) + (mma_tma_stage) * 3328;
                        int _mma_b_lo_5 = ((((b1_addr) >> 4) & 0x3FFF) | 0x800000) + (mma_tma_stage) * 3328;
#endif
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x80004020;\n\t"
                    "mov.b32 bdhi, 0x80004020;\n\t"
                    "mov.b32 id, 270599312;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_5), "r"(_mma_b_lo_5), "r"((tmem_small_acc + (mma_epi_stage * 128))), "r"(1) : "memory");
                        elect_commit_cg2_multicast(mma_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                        mma_tma_stage += 1;
                        if (mma_tma_stage == 4) { mma_tma_stage = 0; _phase_parts_full ^= 1; }
                        int chunk_end = ((kc == 1) ? 1 : 0);
                        if (iter_k_1 == k_iters_split - 1) {
                            chunk_end = 1;
                        }
                        if (chunk_end == 1) {
                            elect_commit_cg2_multicast(chunk_full_addr + (chunk_stage) * 8, (uint16_t)(3));
                            chunk_stage += 1;
                            if (chunk_stage == 2) { chunk_stage = 0; _phase_chunk_empty ^= 1; }
                        }
                    }
                    elect_commit_cg2_multicast(mainloop_done_addr + (mma_epi_stage) * 8, (uint16_t)(3));
                    mma_epi_stage += 1;
                    if (mma_epi_stage == 2) { mma_epi_stage = 0; _phase_epilogue_done ^= 1; }
                    mbarrier_wait(work_full_addr + (work_stage_1) * 8, _phase_work_full_1);
                    uint32_t clc_valid_1 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %0, 1, 0, p1;\n\t"
                        "}\n"
                        : "=r"(clc_valid_1)
                        : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    uint32_t clc_ctaid_1 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "+r"(clc_ctaid_1)
                        : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(work_empty_addr + (work_stage_1) * 8), "r"(0) : "memory");
                    work_stage_1 += 1;
                    if (work_stage_1 == 4) { work_stage_1 = 0; _phase_work_full_1 ^= 1; }
                    if (clc_valid_1 == 0) {
                        break;
                    }
                }
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 2 && warp <= 9) {
        { // epilogue_main
            unsigned int epi_stage = 0;
            unsigned int chunk_stage_1 = 0;
            unsigned int work_stage_2 = 0;
            const int epi_warp = warp % 4;
            const int col_part = (warp - 2) / 4;
            const int local_row = epi_warp * 32 + lane;
            int lane_base = taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)(col_part * 64);
            unsigned int this_bid_1 = bid;
            unsigned int _phase_chunk_full = 0;
            unsigned int _phase_mainloop_done = 0;
            unsigned int _phase_work_full_2 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_2 = 0; _tile_iter_2 < num_cluster_tiles; _tile_iter_2++) {
                int pair_1 = this_bid_1 / 2;
                int rank_1 = this_bid_1 % 2;
                int pairs_per_l_1 = m_tiles / 2 * n_tiles;
                int l_1 = pair_1 / pairs_per_l_1;
                int prem_1 = pair_1 - l_1 * pairs_per_l_1;
                int pair_m_1 = prem_1 / n_tiles;
                int bid_n_1 = prem_1 % n_tiles;
                int off_m_1 = (pair_m_1 * 2 + rank_1) * 128;
                int off_n_1 = bid_n_1 * 128;
                int num_chunks = (k_iters_split + 1) / 2;
                float run[64];
                #pragma unroll
                for (int j = 0; j < 64; j++) {
                    run[j] = 0.0f;
                }
                #pragma unroll 1
                for (int _c = 0; _c < num_chunks; _c++) {
                    mbarrier_wait(chunk_full_addr + (chunk_stage_1) * 8, _phase_chunk_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int big_addr = (unsigned int)lane_base + chunk_stage_1 * 128;
                    #pragma unroll
                    for (int piece = 0; piece < 2; piece++) {
                        float _tmem_load_0[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                            : "r"(big_addr + piece * 32));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        #pragma unroll
                        for (int j_1 = 0; j_1 < 32; j_1++) {
                            run[piece * 32 + j_1] = run[piece * 32 + j_1] + _tmem_load_0[j_1];
                        }
                    }
                    if (elect_sync()) {
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((chunk_empty_addr + (chunk_stage_1) * 8) & 0xFEFFFFFF) : "memory");
                    }
                    chunk_stage_1 += 1;
                    if (chunk_stage_1 == 2) { chunk_stage_1 = 0; _phase_chunk_full ^= 1; }
                }
                mbarrier_wait(mainloop_done_addr + (epi_stage) * 8, _phase_mainloop_done);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int small_addr = (unsigned int)(lane_base + 256) + epi_stage * 128;
                #pragma unroll
                for (int piece_1 = 0; piece_1 < 2; piece_1++) {
                    float _tmem_load_1[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31])
                        : "r"(small_addr + piece_1 * 32));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int j_2 = 0; j_2 < 32; j_2++) {
                        run[piece_1 * 32 + j_2] = run[piece_1 * 32 + j_2] + _tmem_load_1[j_2];
                    }
                }
                if (elect_sync()) {
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((epilogue_done_addr + (epi_stage) * 8) & 0xFEFFFFFF) : "memory");
                }
                epi_stage += 1;
                if (epi_stage == 2) { epi_stage = 0; _phase_mainloop_done ^= 1; }
                int global_row = off_m_1 + local_row;
                int col_base = off_n_1 + col_part * 64;
                unsigned long long batch_off = (unsigned long long)l_1 * (unsigned long long)out_l;
                unsigned long long row_base = batch_off + (unsigned long long)global_row * (unsigned long long)ldo + (unsigned long long)col_base;
                unsigned long long row_bits = (unsigned long long)(out32 + row_base);
                if (global_row < M) {
                    if ((row_bits & 31) == 0) {
                        #pragma unroll
                        for (int q = 0; q < 8; q++) {
                            if (col_base + 8 * q + 8 <= N) {
                                {
                                    unsigned _stv8_0_0 = __float_as_uint(run[8 * q + 0]);
                                    unsigned _stv8_0_1 = __float_as_uint(run[8 * q + 1]);
                                    unsigned _stv8_0_2 = __float_as_uint(run[8 * q + 2]);
                                    unsigned _stv8_0_3 = __float_as_uint(run[8 * q + 3]);
                                    unsigned _stv8_0_4 = __float_as_uint(run[8 * q + 4]);
                                    unsigned _stv8_0_5 = __float_as_uint(run[8 * q + 5]);
                                    unsigned _stv8_0_6 = __float_as_uint(run[8 * q + 6]);
                                    unsigned _stv8_0_7 = __float_as_uint(run[8 * q + 7]);
                                    asm volatile(
                                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                        :: "l"((void*)(out32 + (row_base + (unsigned long long)(8 * q)) + (0))), "r"(_stv8_0_0), "r"(_stv8_0_1), "r"(_stv8_0_2), "r"(_stv8_0_3), "r"(_stv8_0_4), "r"(_stv8_0_5), "r"(_stv8_0_6), "r"(_stv8_0_7) : "memory");
                                }
                            }
                        }
                    } else {
                        #pragma unroll
                        for (int q_1 = 0; q_1 < 16; q_1++) {
                            if (col_base + 4 * q_1 + 4 <= N) {
                                {
                                    float4 _v4 = make_float4(run[4 * q_1 + 0], run[4 * q_1 + 1], run[4 * q_1 + 2], run[4 * q_1 + 3]);
                                    *reinterpret_cast<float4*>((out32 + (row_base + (unsigned long long)(4 * q_1))) + 0) = _v4;
                                }
                            }
                        }
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_2) * 8, _phase_work_full_2);
                uint32_t clc_valid_2 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(clc_valid_2)
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                    : "memory");
                uint32_t clc_ctaid_2 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(clc_ctaid_2)
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + (work_stage_2) * 8), "r"(0) : "memory");
                work_stage_2 += 1;
                if (work_stage_2 == 4) { work_stage_2 = 0; _phase_work_full_2 ^= 1; }
                if (clc_valid_2 == 0) {
                    break;
                }
                this_bid_1 = clc_ctaid_2 + (unsigned int)cta_rank;
            }
        }
    }
    // ---- Role: producer ----
    if (warp >= 10 && warp <= 13) {
        { // producer_main
            unsigned int p_stage = 0;
            unsigned int work_stage_3 = 0;
            const int tid_1 = (warp - 10) * 32 + lane;
            const int row = tid_1;
            const int src_off = tid_1 * 128;
            const int dst_off = 0;
            const int swz8 = row & 7;
            unsigned int this_bid_2 = bid;
            unsigned int _phase_raw_full = 0;
            unsigned int _phase_work_full_3 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_3 = 0; _tile_iter_3 < num_cluster_tiles; _tile_iter_3++) {
                #pragma unroll 1
                for (int iter_k_2 = 0; iter_k_2 < k_iters_split; iter_k_2++) {
                    mbarrier_wait(raw_full_addr + (p_stage) * 8, _phase_raw_full);
                    unsigned int src_row = a32_addr + p_stage * 53248 + (unsigned int)src_off;
                    unsigned int base1 = a1_addr + p_stage * 53248 + (unsigned int)dst_off;
                    unsigned int base2 = a2_addr + p_stage * 53248 + (unsigned int)dst_off;
                    unsigned int base3 = a3_addr + p_stage * 53248 + (unsigned int)dst_off;
                    #pragma unroll
                    for (int h = 0; h < 2; h++) {
                        unsigned int w0[4];
                        unsigned int w1[4];
                        unsigned int w2[4];
                        unsigned int w3[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w0[0])), "=r"(*reinterpret_cast<uint32_t*>(&w0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w0[(0) + 3]))
                            : "r"(src_row + (unsigned int)((4 * h ^ swz8) * 16)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w1[0])), "=r"(*reinterpret_cast<uint32_t*>(&w1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w1[(0) + 3]))
                            : "r"(src_row + (unsigned int)((4 * h + 1 ^ swz8) * 16)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w2[0])), "=r"(*reinterpret_cast<uint32_t*>(&w2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w2[(0) + 3]))
                            : "r"(src_row + (unsigned int)((4 * h + 2 ^ swz8) * 16)));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&w3[0])), "=r"(*reinterpret_cast<uint32_t*>(&w3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&w3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&w3[(0) + 3]))
                            : "r"(src_row + (unsigned int)((4 * h + 3 ^ swz8) * 16)));
                        unsigned int ph[8];
                        unsigned int pm[8];
                        unsigned int pl[8];
                        float v0 = __uint_as_float(w0[0]);
                        float v1 = __uint_as_float(w0[1]);
                        unsigned int pk = __as_u32(__float22bfloat162_rn(make_float2(v0, v1)));
                        float r0 = v0 - __uint_as_float(pk << 16);
                        float r1 = v1 - __uint_as_float(pk & 4294901760u);
                        unsigned int m0 = __as_u32(r0) & 4294901760u;
                        unsigned int m1 = __as_u32(r1) & 4294901760u;
                        float s0 = r0 - __uint_as_float(m0);
                        float s1 = r1 - __uint_as_float(m1);
                        ph[0] = pk;
                        uint32_t _prmt_0;
                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_0) : "r"(m0), "r"(m1));
                        pm[0] = _prmt_0;
                        uint32_t _prmt_1;
                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_1) : "r"(__as_u32(s0)), "r"(__as_u32(s1)));
                        pl[0] = _prmt_1;
                        float v0_0 = __uint_as_float(w0[2]);
                        float v1_0 = __uint_as_float(w0[3]);
                        unsigned int pk_0 = __as_u32(__float22bfloat162_rn(make_float2(v0_0, v1_0)));
                        float r0_0 = v0_0 - __uint_as_float(pk_0 << 16);
                        float r1_0 = v1_0 - __uint_as_float(pk_0 & 4294901760u);
                        unsigned int m0_0 = __as_u32(r0_0) & 4294901760u;
                        unsigned int m1_0 = __as_u32(r1_0) & 4294901760u;
                        float s0_0 = r0_0 - __uint_as_float(m0_0);
                        float s1_0 = r1_0 - __uint_as_float(m1_0);
                        ph[1] = pk_0;
                        uint32_t _prmt_2;
                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_2) : "r"(m0_0), "r"(m1_0));
                        pm[1] = _prmt_2;
                        uint32_t _prmt_3;
                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_3) : "r"(__as_u32(s0_0)), "r"(__as_u32(s1_0)));
                        pl[1] = _prmt_3;
                        float v0_1 = __uint_as_float(w1[0]);
                        float v1_1 = __uint_as_float(w1[1]);
                        unsigned int pk_1 = __as_u32(__float22bfloat162_rn(make_float2(v0_1, v1_1)));
                        float r0_1 = v0_1 - __uint_as_float(pk_1 << 16);
                        float r1_1 = v1_1 - __uint_as_float(pk_1 & 4294901760u);
                        unsigned int m0_1 = __as_u32(r0_1) & 4294901760u;
                        unsigned int m1_1 = __as_u32(r1_1) & 4294901760u;
                        float s0_1 = r0_1 - __uint_as_float(m0_1);
                        float s1_1 = r1_1 - __uint_as_float(m1_1);
                        ph[2] = pk_1;
                        uint32_t _prmt_4;
                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_4) : "r"(m0_1), "r"(m1_1));
                        pm[2] = _prmt_4;
                        uint32_t _prmt_5;
                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_5) : "r"(__as_u32(s0_1)), "r"(__as_u32(s1_1)));
                        pl[2] = _prmt_5;
                        float v0_2 = __uint_as_float(w1[2]);
                        float v1_2 = __uint_as_float(w1[3]);
                        unsigned int pk_2 = __as_u32(__float22bfloat162_rn(make_float2(v0_2, v1_2)));
                        float r0_2 = v0_2 - __uint_as_float(pk_2 << 16);
                        float r1_2 = v1_2 - __uint_as_float(pk_2 & 4294901760u);
                        unsigned int m0_2 = __as_u32(r0_2) & 4294901760u;
                        unsigned int m1_2 = __as_u32(r1_2) & 4294901760u;
                        float s0_2 = r0_2 - __uint_as_float(m0_2);
                        float s1_2 = r1_2 - __uint_as_float(m1_2);
                        ph[3] = pk_2;
                        uint32_t _prmt_6;
                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_6) : "r"(m0_2), "r"(m1_2));
                        pm[3] = _prmt_6;
                        uint32_t _prmt_7;
                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_7) : "r"(__as_u32(s0_2)), "r"(__as_u32(s1_2)));
                        pl[3] = _prmt_7;
                        float v0_3 = __uint_as_float(w2[0]);
                        float v1_3 = __uint_as_float(w2[1]);
                        unsigned int pk_3 = __as_u32(__float22bfloat162_rn(make_float2(v0_3, v1_3)));
                        float r0_3 = v0_3 - __uint_as_float(pk_3 << 16);
                        float r1_3 = v1_3 - __uint_as_float(pk_3 & 4294901760u);
                        unsigned int m0_3 = __as_u32(r0_3) & 4294901760u;
                        unsigned int m1_3 = __as_u32(r1_3) & 4294901760u;
                        float s0_3 = r0_3 - __uint_as_float(m0_3);
                        float s1_3 = r1_3 - __uint_as_float(m1_3);
                        ph[4] = pk_3;
                        uint32_t _prmt_8;
                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_8) : "r"(m0_3), "r"(m1_3));
                        pm[4] = _prmt_8;
                        uint32_t _prmt_9;
                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_9) : "r"(__as_u32(s0_3)), "r"(__as_u32(s1_3)));
                        pl[4] = _prmt_9;
                        float v0_4 = __uint_as_float(w2[2]);
                        float v1_4 = __uint_as_float(w2[3]);
                        unsigned int pk_4 = __as_u32(__float22bfloat162_rn(make_float2(v0_4, v1_4)));
                        float r0_4 = v0_4 - __uint_as_float(pk_4 << 16);
                        float r1_4 = v1_4 - __uint_as_float(pk_4 & 4294901760u);
                        unsigned int m0_4 = __as_u32(r0_4) & 4294901760u;
                        unsigned int m1_4 = __as_u32(r1_4) & 4294901760u;
                        float s0_4 = r0_4 - __uint_as_float(m0_4);
                        float s1_4 = r1_4 - __uint_as_float(m1_4);
                        ph[5] = pk_4;
                        uint32_t _prmt_10;
                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_10) : "r"(m0_4), "r"(m1_4));
                        pm[5] = _prmt_10;
                        uint32_t _prmt_11;
                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_11) : "r"(__as_u32(s0_4)), "r"(__as_u32(s1_4)));
                        pl[5] = _prmt_11;
                        float v0_5 = __uint_as_float(w3[0]);
                        float v1_5 = __uint_as_float(w3[1]);
                        unsigned int pk_5 = __as_u32(__float22bfloat162_rn(make_float2(v0_5, v1_5)));
                        float r0_5 = v0_5 - __uint_as_float(pk_5 << 16);
                        float r1_5 = v1_5 - __uint_as_float(pk_5 & 4294901760u);
                        unsigned int m0_5 = __as_u32(r0_5) & 4294901760u;
                        unsigned int m1_5 = __as_u32(r1_5) & 4294901760u;
                        float s0_5 = r0_5 - __uint_as_float(m0_5);
                        float s1_5 = r1_5 - __uint_as_float(m1_5);
                        ph[6] = pk_5;
                        uint32_t _prmt_12;
                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_12) : "r"(m0_5), "r"(m1_5));
                        pm[6] = _prmt_12;
                        uint32_t _prmt_13;
                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_13) : "r"(__as_u32(s0_5)), "r"(__as_u32(s1_5)));
                        pl[6] = _prmt_13;
                        float v0_6 = __uint_as_float(w3[2]);
                        float v1_6 = __uint_as_float(w3[3]);
                        unsigned int pk_6 = __as_u32(__float22bfloat162_rn(make_float2(v0_6, v1_6)));
                        float r0_6 = v0_6 - __uint_as_float(pk_6 << 16);
                        float r1_6 = v1_6 - __uint_as_float(pk_6 & 4294901760u);
                        unsigned int m0_6 = __as_u32(r0_6) & 4294901760u;
                        unsigned int m1_6 = __as_u32(r1_6) & 4294901760u;
                        float s0_6 = r0_6 - __uint_as_float(m0_6);
                        float s1_6 = r1_6 - __uint_as_float(m1_6);
                        ph[7] = pk_6;
                        uint32_t _prmt_14;
                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_14) : "r"(m0_6), "r"(m1_6));
                        pm[7] = _prmt_14;
                        uint32_t _prmt_15;
                        asm("prmt.b32 %0, %1, %2, 0x7632;" : "=r"(_prmt_15) : "r"(__as_u32(s0_6)), "r"(__as_u32(s1_6)));
                        pl[7] = _prmt_15;
                        #pragma unroll
                        for (int q_2 = 0; q_2 < 2; q_2++) {
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((base1 + (unsigned int)(row * 64 + (h * 32 + q_2 * 16) ^ (row * 64 + (h * 32 + q_2 * 16) >> 7 & 3) << 4))), "r"(ph[4 * q_2]), "r"(ph[4 * q_2 + 1]), "r"(ph[4 * q_2 + 2]), "r"(ph[4 * q_2 + 3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((base2 + (unsigned int)(row * 64 + (h * 32 + q_2 * 16) ^ (row * 64 + (h * 32 + q_2 * 16) >> 7 & 3) << 4))), "r"(pm[4 * q_2]), "r"(pm[4 * q_2 + 1]), "r"(pm[4 * q_2 + 2]), "r"(pm[4 * q_2 + 3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((base3 + (unsigned int)(row * 64 + (h * 32 + q_2 * 16) ^ (row * 64 + (h * 32 + q_2 * 16) >> 7 & 3) << 4))), "r"(pl[4 * q_2]), "r"(pl[4 * q_2 + 1]), "r"(pl[4 * q_2 + 2]), "r"(pl[4 * q_2 + 3]) : "memory");
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    if (elect_sync()) {
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((parts_full_addr + (p_stage) * 8) & 0xFEFFFFFF) : "memory");
                    }
                    p_stage += 1;
                    if (p_stage == 4) { p_stage = 0; _phase_raw_full ^= 1; }
                }
                mbarrier_wait(work_full_addr + (work_stage_3) * 8, _phase_work_full_3);
                uint32_t clc_valid_3 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(clc_valid_3)
                    : "r"(work_response_addr + work_stage_3 * 16 + 0 * 16)
                    : "memory");
                uint32_t clc_ctaid_3 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(clc_ctaid_3)
                    : "r"(work_response_addr + work_stage_3 * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + (work_stage_3) * 8), "r"(0) : "memory");
                work_stage_3 += 1;
                if (work_stage_3 == 4) { work_stage_3 = 0; _phase_work_full_3 ^= 1; }
                if (clc_valid_3 == 0) {
                    break;
                }
                this_bid_2 = clc_ctaid_3 + (unsigned int)cta_rank;
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
