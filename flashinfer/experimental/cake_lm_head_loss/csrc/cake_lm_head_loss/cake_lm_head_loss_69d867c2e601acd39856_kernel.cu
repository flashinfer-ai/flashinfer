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
#define NUM_CLC_PIPE_STAGES 2
#define NUM_THR_PIPE_STAGES 2
#define SMEM_SMEM_V1_OFF 1024
#define SMEM_SMEM_V1_STAGE_BYTES 16384
#define SMEM_SMEM_V1_STRIDE 49152
#define SMEM_SMEM_V0_OFF 17408
#define SMEM_SMEM_V0_STAGE_BYTES 16384
#define SMEM_SMEM_V0_STRIDE 49152
#define SMEM_SMEM_V0_0_OFF 33792
#define SMEM_SMEM_V0_0_STAGE_BYTES 16384
#define SMEM_SMEM_V0_0_STRIDE 49152
#define SMEM_CLC_RESP_OFF 197632
#define SMEM_CLC_RESP_STAGE_BYTES 16
#define SMEM_CLC_RESP_STRIDE 16
#define SMEM_TOTAL 197760
#define THREADS 224

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





__device__ __forceinline__ void tma_3d_gmem2smem_cta2(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
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

__global__ __launch_bounds__(224) __cluster_dims__(2,1,1) void
kernel_cake_lm_head_loss_69d867c2e601acd39856(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, float* __restrict__ C, float* __restrict__ STATS_OUT, int M, int m_tiles, int k_iters, int first_chunk, float* __restrict__ WS, int ws_slab, int ldc, int k_slices, int k_slice_iters, int group_m, int group_n)
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
    #define clc_full_addr (mbar_base + 80)
    #define clc_empty_addr (mbar_base + 96)
    #define thr_full_addr (mbar_base + 112)
    #define thr_empty_addr (mbar_base + 128)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    __nv_bfloat16* smem_v1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_v1_addr = smem + 1024;
    __nv_bfloat16* smem_v0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int smem_v0_addr = smem + 17408;
    __nv_bfloat16* smem_v0_0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int smem_v0_addr_0 = smem + 33792;
    unsigned int* clc_resp = reinterpret_cast<unsigned int*>(smem_raw + 197632);
    const int clc_resp_addr = smem + 197632;

    // Mbarrier init (8 pipeline groups, 0 ordered-sequence groups, 18 barriers)
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
            // epilogue_done: 1 barriers, init_count=8
            mbarrier_init(smem + 72, 8);
            // --- pipeline 'clc_pipe' ---
            // clc_full: 2 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // clc_empty: 2 barriers, init_count=416
            mbarrier_init(smem + 96, 416);
            mbarrier_init(smem + 104, 416);
            // --- pipeline 'thr_pipe' ---
            // thr_full: 2 barriers, init_count=32
            mbarrier_init(smem + 112, 32);
            mbarrier_init(smem + 120, 32);
            // thr_empty: 2 barriers, init_count=32
            mbarrier_init(smem + 128, 32);
            mbarrier_init(smem + 136, 32);
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
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            unsigned int load_stage = 0;
            unsigned int ld_cluster_id = bid / 2;
            unsigned int ld_num_clusters = num_bids / 2;
            unsigned int load_clc_stage = 0;
            unsigned int load_thr_stage = 0;
            unsigned int ld_tile = ld_cluster_id;
            unsigned int _phase_thr_empty = 1;
            unsigned int _phase_mma_done = 1;
            unsigned int _phase_clc_full = 0;
            #pragma unroll 1
            for (unsigned int _lp = 0; _lp < ldc / 128 / 2 * ((M + 511) / 512) * k_slices; _lp++) {
                if (cta_rank == 0) {
                    mbarrier_wait(thr_empty_addr + (load_thr_stage) * 8, _phase_thr_empty);
                    mbarrier_arrive(thr_full_addr + (load_thr_stage) * 8);
                    load_thr_stage += 1;
                    if (load_thr_stage == 2) { load_thr_stage = 0; _phase_thr_empty ^= 1; }
                }
                if (elect_sync()) {
                    int ld_item_i = ld_tile;
                    int ld_slice = ld_item_i / (ldc / 128 / 2 * ((M + 511) / 512));
                    int ld_pair_item = ld_item_i % (ldc / 128 / 2 * ((M + 511) / 512));
                    int ld_k_base = ld_slice * k_slice_iters;
                    int ld_k_count = ((ld_slice < k_slices - 1) ? k_slice_iters : k_iters - (k_slices - 1) * k_slice_iters);
                    int ld_pair_bid = ld_pair_item * 2 + cta_rank;
                    int group = ld_pair_bid / (group_m * ((M + 511) / 512));
                    int first_m = group * group_m;
                    int remaining = ldc / 128 - first_m;
                    int group_size = ((remaining >= group_m) ? group_m : remaining);
                    int local = ld_pair_bid % (group_m * ((M + 511) / 512));
                    int bid_m = first_m + local % group_size;
                    int bid_n = local / group_size;
                    int ld_off_h = bid_m * 128;
                    int ld_tok0 = bid_n * 512;
                    int sw_rem = M - ld_tok0;
                    int sw_rem0 = ((sw_rem >= 256) ? 256 : sw_rem);
                    int sw_rem1 = ((sw_rem > 256) ? sw_rem - 256 : 0);
                    int sw_rem1c = ((sw_rem1 >= 256) ? 256 : sw_rem1);
                    int sw_n0 = (sw_rem0 + 15) / 16 * 16;
                    int sw_n1 = (sw_rem1c + 15) / 16 * 16;
                    int ld_row0 = ld_tok0 + cta_rank * (sw_n0 / 2);
                    int ld_row1 = ld_tok0 + 256 + cta_rank * (sw_n1 / 2);
                    int ld_tx_s = ((sw_n1 > 0) ? 49152 : 32768);
                    #pragma unroll 1
                    for (int ld_k = 0; ld_k < ld_k_count; ld_k++) {
                        mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                        int ld_kk = ld_k_base + ld_k;
                        int ld_k0 = ld_kk * 64;
                        tma_3d_gmem2smem_cta2(smem_v1_addr + load_stage * 49152, (&B), ld_off_h, ld_k0, 0, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_v1_addr + load_stage * 49152 + 8192, (&B), ld_off_h + 64, ld_k0, 0, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_v0_addr + load_stage * 49152, (&A), 0, ld_row0, ld_kk, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        if (sw_n1 > 0) {
                            tma_3d_gmem2smem_cta2(smem_v0_addr_0 + load_stage * 49152, (&A), 0, ld_row1, ld_kk, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        }
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(ld_tx_s)) : "memory");
                        load_stage += 1;
                        if (load_stage == 4) { load_stage = 0; _phase_mma_done ^= 1; }
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
                    : "+r"(_clc_ctaid_0)
                    : "r"(clc_resp_addr + load_clc_stage * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(clc_empty_addr + (load_clc_stage) * 8), "r"(0) : "memory");
                load_clc_stage += 1;
                if (load_clc_stage == 2) { load_clc_stage = 0; _phase_clc_full ^= 1; }
                if (_clc_valid_1 == 0) {
                    break;
                }
                ld_tile = _clc_ctaid_0 / 2;
            }
            if (elect_sync()) {
                #pragma unroll 1
                for (int _ = 0; _ < 4; _++) {
                    mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                    load_stage += 1;
                    if (load_stage == 4) { load_stage = 0; _phase_mma_done ^= 1; }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            unsigned int mma_tma_stage = 0;
            unsigned int mma_cluster_id = bid / 2;
            unsigned int mma_num_clusters = num_bids / 2;
            unsigned int mma_clc_stage = 0;
            unsigned int mm_tile = mma_cluster_id;
            unsigned int _phase_epilogue_done = 1;
            unsigned int _phase_tma_full = 0;
            unsigned int _phase_clc_full_1 = 0;
            #pragma unroll 1
            for (unsigned int _mp = 0; _mp < ldc / 128 / 2 * ((M + 511) / 512) * k_slices; _mp++) {
                if (cta_rank == 0) {
                    int mm_item_i = mm_tile;
                    int mm_slice = mm_item_i / (ldc / 128 / 2 * ((M + 511) / 512));
                    int mm_k_last = ((mm_slice < k_slices - 1) ? k_slice_iters : k_iters - (k_slices - 1) * k_slice_iters);
                    int mm_pair_item = mm_item_i % (ldc / 128 / 2 * ((M + 511) / 512));
                    int mm_pair_bid = mm_pair_item * 2;
                    int group_1 = mm_pair_bid / (group_m * ((M + 511) / 512));
                    int first_m_1 = group_1 * group_m;
                    int remaining_1 = ldc / 128 - first_m_1;
                    int group_size_1 = ((remaining_1 >= group_m) ? group_m : remaining_1);
                    int local_1 = mm_pair_bid % (group_m * ((M + 511) / 512));
                    int bid_m_1 = first_m_1 + local_1 % group_size_1;
                    int bid_n_1 = local_1 / group_size_1;
                    int sw_rem_1 = M - bid_n_1 * 512;
                    int sw_rem0_1 = ((sw_rem_1 >= 256) ? 256 : sw_rem_1);
                    int sw_rem1_1 = ((sw_rem_1 > 256) ? sw_rem_1 - 256 : 0);
                    int sw_rem1c_1 = ((sw_rem1_1 >= 256) ? 256 : sw_rem1_1);
                    int sw_n0_1 = (sw_rem0_1 + 15) / 16 * 16;
                    int sw_n1_1 = (sw_rem1c_1 + 15) / 16 * 16;
                    mbarrier_wait(epilogue_done_addr, _phase_epilogue_done);
                    #pragma unroll 1
                    for (int mm_k = 0; mm_k < mm_k_last; mm_k++) {
                        mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int init_flag = ((mm_k == 0) ? 1 : 0);
                        int _mma_a_lo_0 = ((((smem_v1_addr) >> 4) & 0x3FFF) | 0x2000000) + (mma_tma_stage) * 3072;
                        int _mma_b_lo_0 = ((((smem_v0_addr) >> 4) & 0x3FFF) | 0x4000000) + (mma_tma_stage) * 3072;
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
                    "mov.b32 id, %4;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_accum), "r"(((init_flag) ? 0 : 1)), "r"(((0x10408490U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(sw_n0_1) >> 3) << 17))) : "memory");
                        if (sw_n1_1 > 0) {
                            int _mma_a_lo_1 = ((((smem_v1_addr) >> 4) & 0x3FFF) | 0x2000000) + (mma_tma_stage) * 3072;
                            int _mma_b_lo_1 = ((((smem_v0_addr_0) >> 4) & 0x3FFF) | 0x4000000) + (mma_tma_stage) * 3072;
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
                    "mov.b32 id, %4;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"((tmem_accum + (256))), "r"(((init_flag) ? 0 : 1)), "r"(((0x10408490U & ~(0x3fU << 17)) | ((static_cast<uint32_t>(sw_n1_1) >> 3) << 17))) : "memory");
                        }
                        elect_commit_cg2_multicast(mma_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                        mma_tma_stage += 1;
                        if (mma_tma_stage == 4) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    elect_commit_cg2_multicast(mainloop_done_addr, (uint16_t)(3));
                    _phase_epilogue_done ^= 1;
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
                    : "+r"(_clc_ctaid_1)
                    : "r"(clc_resp_addr + mma_clc_stage * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(clc_empty_addr + (mma_clc_stage) * 8), "r"(0) : "memory");
                mma_clc_stage += 1;
                if (mma_clc_stage == 2) { mma_clc_stage = 0; _phase_clc_full_1 ^= 1; }
                if (_clc_valid_2 == 0) {
                    break;
                }
                mm_tile = _clc_ctaid_1 / 2;
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 2 && warp <= 5) {
        { // epilogue_main
            const int epi_warp = warp % 4;
            const int local_row = epi_warp * 32 + lane;
            const int epi_col0 = (warp - 2) / 4 * 256;
            unsigned int ep_cluster_id = bid / 2;
            unsigned int ep_num_clusters = num_bids / 2;
            unsigned int epi_clc_stage = 0;
            unsigned int ep_tile = ep_cluster_id;
            unsigned int _phase_clc_full_2 = 0;
            unsigned int _phase_mainloop_done = 0;
            #pragma unroll 1
            for (unsigned int _ep = 0; _ep < ldc / 128 / 2 * ((M + 511) / 512) * k_slices; _ep++) {
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
                    : "+r"(_clc_ctaid_2)
                    : "r"(clc_resp_addr + epi_clc_stage * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(clc_empty_addr + (epi_clc_stage) * 8), "r"(0) : "memory");
                epi_clc_stage += 1;
                if (epi_clc_stage == 2) { epi_clc_stage = 0; _phase_clc_full_2 ^= 1; }
                int ep_item_i = ep_tile;
                int ep_slice = ep_item_i / (ldc / 128 / 2 * ((M + 511) / 512));
                int ep_pair_item = ep_item_i % (ldc / 128 / 2 * ((M + 511) / 512));
                int ep_pair_bid = ep_pair_item * 2 + cta_rank;
                int group_2 = ep_pair_bid / (group_m * ((M + 511) / 512));
                int first_m_2 = group_2 * group_m;
                int remaining_2 = ldc / 128 - first_m_2;
                int group_size_2 = ((remaining_2 >= group_m) ? group_m : remaining_2);
                int local_2 = ep_pair_bid % (group_m * ((M + 511) / 512));
                int bid_m_2 = first_m_2 + local_2 % group_size_2;
                int bid_n_2 = local_2 / group_size_2;
                int ep_h = bid_m_2 * 128 + local_row;
                int ep_tok0 = bid_n_2 * 512;
                int sw_rem_2 = M - ep_tok0;
                int sw_rem0_2 = ((sw_rem_2 >= 256) ? 256 : sw_rem_2);
                int sw_rem1_2 = ((sw_rem_2 > 256) ? sw_rem_2 - 256 : 0);
                int sw_rem1c_2 = ((sw_rem1_2 >= 256) ? 256 : sw_rem1_2);
                int sw_n0_2 = (sw_rem0_2 + 15) / 16 * 16;
                int sw_n1_2 = (sw_rem1c_2 + 15) / 16 * 16;
                int lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16);
                mbarrier_wait(mainloop_done_addr, _phase_mainloop_done);
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (ep_slice == 0) {
                    #pragma unroll 1
                    for (int sw_c = 0; sw_c < 16; sw_c++) {
                        int sw_col = sw_c * 16;
                        if (sw_n0_2 > sw_c * 16) {
                            float tmem_load[16];
                            tmem_ld_x16(&tmem_load[0], lane_addr + sw_col);
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            int sw_tok = ep_tok0 + sw_col;
                            unsigned long long sw_out = (unsigned long long)sw_tok * (unsigned long long)ldc + (unsigned long long)ep_h;
                            #pragma unroll
                            for (int sj = 0; sj < 16; sj++) {
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(C + (sw_out + (unsigned long long)(sj * ldc)))), "f"(tmem_load[(sj) + 0]), "r"((unsigned int)(sw_tok + sj < M)) : "memory");
                            }
                        }
                    }
                    #pragma unroll 1
                    for (int sw_c_1 = 0; sw_c_1 < 16; sw_c_1++) {
                        int sw_col_1 = 256 + sw_c_1 * 16;
                        if (sw_n1_2 > sw_c_1 * 16) {
                            float tmem_load_1[16];
                            tmem_ld_x16(&tmem_load_1[0], lane_addr + sw_col_1);
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            int sw_tok_1 = ep_tok0 + sw_col_1;
                            unsigned long long sw_out_1 = (unsigned long long)sw_tok_1 * (unsigned long long)ldc + (unsigned long long)ep_h;
                            #pragma unroll
                            for (int sj_1 = 0; sj_1 < 16; sj_1++) {
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(C + (sw_out_1 + (unsigned long long)(sj_1 * ldc)))), "f"(tmem_load_1[(sj_1) + 0]), "r"((unsigned int)(sw_tok_1 + sj_1 < M)) : "memory");
                            }
                        }
                    }
                } else {
                    int ep_slab = ep_slice - 1;
                    unsigned long long ws_off = (unsigned long long)ep_slab * (unsigned long long)ws_slab;
                    #pragma unroll 1
                    for (int sw_c_2 = 0; sw_c_2 < 16; sw_c_2++) {
                        int sw_col_2 = sw_c_2 * 16;
                        if (sw_n0_2 > sw_c_2 * 16) {
                            float tmem_load_2[16];
                            tmem_ld_x16(&tmem_load_2[0], lane_addr + sw_col_2);
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            int sw_tok_2 = ep_tok0 + sw_col_2;
                            unsigned long long sw_out_2 = (unsigned long long)sw_tok_2 * (unsigned long long)ldc + (unsigned long long)ep_h;
                            #pragma unroll
                            for (int sj_2 = 0; sj_2 < 16; sj_2++) {
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(WS + (ws_off + (sw_out_2 + (unsigned long long)(sj_2 * ldc))))), "f"(tmem_load_2[(sj_2) + 0]), "r"((unsigned int)(sw_tok_2 + sj_2 < M)) : "memory");
                            }
                        }
                    }
                    #pragma unroll 1
                    for (int sw_c_3 = 0; sw_c_3 < 16; sw_c_3++) {
                        int sw_col_3 = 256 + sw_c_3 * 16;
                        if (sw_n1_2 > sw_c_3 * 16) {
                            float tmem_load_3[16];
                            tmem_ld_x16(&tmem_load_3[0], lane_addr + sw_col_3);
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            int sw_tok_3 = ep_tok0 + sw_col_3;
                            unsigned long long sw_out_3 = (unsigned long long)sw_tok_3 * (unsigned long long)ldc + (unsigned long long)ep_h;
                            #pragma unroll
                            for (int sj_3 = 0; sj_3 < 16; sj_3++) {
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.f32 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(WS + (ws_off + (sw_out_3 + (unsigned long long)(sj_3 * ldc))))), "f"(tmem_load_3[(sj_3) + 0]), "r"((unsigned int)(sw_tok_3 + sj_3 < M)) : "memory");
                            }
                        }
                    }
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                if (elect_sync()) {
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((epilogue_done_addr) & 0xFEFFFFFF) : "memory");
                }
                _phase_mainloop_done ^= 1;
                if (_clc_valid_3 == 0) {
                    break;
                }
                ep_tile = _clc_ctaid_2 / 2;
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
                #pragma unroll 1
                for (unsigned int _sp = 0; _sp < ldc / 128 / 2 * ((M + 511) / 512) * k_slices; _sp++) {
                    mbarrier_wait(thr_full_addr + (sched_thr_stage) * 8, _phase_thr_full);
                    mbarrier_arrive(thr_empty_addr + (sched_thr_stage) * 8);
                    sched_thr_stage += 1;
                    if (sched_thr_stage == 2) { sched_thr_stage = 0; _phase_thr_full ^= 1; }
                    mbarrier_wait(clc_empty_addr + (sched_clc_stage) * 8, _phase_clc_empty);
                    if (lane < 2) {
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                            "}"
                            :: "r"(clc_full_addr + (sched_clc_stage) * 8), "r"(lane), "r"((uint32_t)(16)) : "memory");
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
                        :: "r"(clc_empty_addr + (sched_clc_stage) * 8), "r"(0) : "memory");
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
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
