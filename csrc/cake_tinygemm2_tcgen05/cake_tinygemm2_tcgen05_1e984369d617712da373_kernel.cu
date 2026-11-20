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
#define TMEM_NCOLS 128
#define TMEM_ACC0_OFFSET 0
#define TMEM_ACC1_OFFSET 32
#define TMEM_ACC2_OFFSET 64
#define TMEM_ACC3_OFFSET 96
#define NUM_TMA_PIPE_STAGES 18
#define NUM_EPI_PIPE_STAGES 1
#define SMEM_SMEM_A_OFF 148480
#define SMEM_SMEM_A_STAGE_BYTES 4096
#define SMEM_SMEM_A_STRIDE 4096
#define SMEM_SMEM_B_OFF 1024
#define SMEM_SMEM_B_STAGE_BYTES 8192
#define SMEM_SMEM_B_STRIDE 8192
#define SMEM_TOTAL 222208
#define THREADS 256

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

extern "C" {

__global__ __launch_bounds__(256) void
kernel_cake_tinygemm2_tcgen05_1e984369d617712da373(const __grid_constant__ CUtensorMap W, const __grid_constant__ CUtensorMap X, __nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ bias, int M, int N, int K)
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
    #define tma_empty_addr (mbar_base + 144)
    #define mma_epi_full_addr (mbar_base + 288)
    #define tmem_free_addr (mbar_base + 296)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 148480);
    const int smem_a_addr = smem + 148480;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_b_addr = smem + 1024;
    if (tid == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&W))) : "memory"); }
    if (tid == 32) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&X))) : "memory"); }

    // Mbarrier init (4 pipeline groups, 0 ordered-sequence groups, 38 barriers)
    // Mbarriers at smem_raw[0..304)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full: 18 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            mbarrier_init(smem + 8, 2);
            mbarrier_init(smem + 16, 2);
            mbarrier_init(smem + 24, 2);
            mbarrier_init(smem + 32, 2);
            mbarrier_init(smem + 40, 2);
            mbarrier_init(smem + 48, 2);
            mbarrier_init(smem + 56, 2);
            mbarrier_init(smem + 64, 2);
            mbarrier_init(smem + 72, 2);
            mbarrier_init(smem + 80, 2);
            mbarrier_init(smem + 88, 2);
            mbarrier_init(smem + 96, 2);
            mbarrier_init(smem + 104, 2);
            mbarrier_init(smem + 112, 2);
            mbarrier_init(smem + 120, 2);
            mbarrier_init(smem + 128, 2);
            mbarrier_init(smem + 136, 2);
            // tma_empty: 18 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            mbarrier_init(smem + 232, 1);
            mbarrier_init(smem + 240, 1);
            mbarrier_init(smem + 248, 1);
            mbarrier_init(smem + 256, 1);
            mbarrier_init(smem + 264, 1);
            mbarrier_init(smem + 272, 1);
            mbarrier_init(smem + 280, 1);
            // --- pipeline 'epi_pipe' ---
            // mma_epi_full: 1 barriers, init_count=1
            mbarrier_init(smem + 288, 1);
            // tmem_free: 1 barriers, init_count=4
            mbarrier_init(smem + 296, 4);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // TMEM alloc (128 columns, 128 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 304);
    if (warp == 2) {
        int _tmem_hold = smem + 304;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    if (warp >= 2 && warp <= 7) {
        asm volatile("barrier.sync.aligned %0, %1;" :: "r"(1), "r"(192) : "memory");
        asm volatile("tcgen05.fence::after_thread_sync;");
    }
    const int taddr = (warp >= 2 && warp <= 7) ? tmem_addr_storage[0] : 0;

    // Kernel post-init ops
    const int tmem_acc0 = taddr;
    const int tmem_acc1 = taddr + 32;
    const int tmem_acc2 = taddr + 64;
    const int tmem_acc3 = taddr + 96;

    // ---- Role: dma_a ----
    if (warp == 0) {
        { // dma_a_main
            unsigned int a_stage = 0;
            int m_idx = blockIdx.x * 32;
            int n_groups = (K + 64 - 1) / 64;
            int n_rounds = ((n_groups + 4 - 1) / 4 + 4 - 1) / 4;
            unsigned int _phase_tma_empty = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int rho = 0; rho < n_rounds; rho++) {
                    int g0_15_0 = rho * 4 * 4;
                    if (g0_15_0 < n_groups) {
                        mbarrier_wait(tma_empty_addr + (a_stage) * 8, _phase_tma_empty);
                        tma_2d_gmem2smem(smem_a_addr + a_stage * 4096, (&W), g0_15_0 * 64, m_idx, tma_full_addr + (a_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (a_stage) * 8, 4096);
                        a_stage += 1;
                        if (a_stage == 18) { a_stage = 0; _phase_tma_empty ^= 1; }
                    }
                    int g0_15_1 = rho * 4 * 4 + 1;
                    if (g0_15_1 < n_groups) {
                        mbarrier_wait(tma_empty_addr + (a_stage) * 8, _phase_tma_empty);
                        tma_2d_gmem2smem(smem_a_addr + a_stage * 4096, (&W), g0_15_1 * 64, m_idx, tma_full_addr + (a_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (a_stage) * 8, 4096);
                        a_stage += 1;
                        if (a_stage == 18) { a_stage = 0; _phase_tma_empty ^= 1; }
                    }
                    int g0_15_2 = rho * 4 * 4 + 2;
                    if (g0_15_2 < n_groups) {
                        mbarrier_wait(tma_empty_addr + (a_stage) * 8, _phase_tma_empty);
                        tma_2d_gmem2smem(smem_a_addr + a_stage * 4096, (&W), g0_15_2 * 64, m_idx, tma_full_addr + (a_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (a_stage) * 8, 4096);
                        a_stage += 1;
                        if (a_stage == 18) { a_stage = 0; _phase_tma_empty ^= 1; }
                    }
                    int g0_15_3 = rho * 4 * 4 + 3;
                    if (g0_15_3 < n_groups) {
                        mbarrier_wait(tma_empty_addr + (a_stage) * 8, _phase_tma_empty);
                        tma_2d_gmem2smem(smem_a_addr + a_stage * 4096, (&W), g0_15_3 * 64, m_idx, tma_full_addr + (a_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (a_stage) * 8, 4096);
                        a_stage += 1;
                        if (a_stage == 18) { a_stage = 0; _phase_tma_empty ^= 1; }
                    }
                    int g0_15_4 = (rho * 4 + 1) * 4;
                    if (g0_15_4 < n_groups) {
                        mbarrier_wait(tma_empty_addr + (a_stage) * 8, _phase_tma_empty);
                        tma_2d_gmem2smem(smem_a_addr + a_stage * 4096, (&W), g0_15_4 * 64, m_idx, tma_full_addr + (a_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (a_stage) * 8, 4096);
                        a_stage += 1;
                        if (a_stage == 18) { a_stage = 0; _phase_tma_empty ^= 1; }
                    }
                    int g0_15_5 = (rho * 4 + 1) * 4 + 1;
                    if (g0_15_5 < n_groups) {
                        mbarrier_wait(tma_empty_addr + (a_stage) * 8, _phase_tma_empty);
                        tma_2d_gmem2smem(smem_a_addr + a_stage * 4096, (&W), g0_15_5 * 64, m_idx, tma_full_addr + (a_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (a_stage) * 8, 4096);
                        a_stage += 1;
                        if (a_stage == 18) { a_stage = 0; _phase_tma_empty ^= 1; }
                    }
                    int g0_15_6 = (rho * 4 + 1) * 4 + 2;
                    if (g0_15_6 < n_groups) {
                        mbarrier_wait(tma_empty_addr + (a_stage) * 8, _phase_tma_empty);
                        tma_2d_gmem2smem(smem_a_addr + a_stage * 4096, (&W), g0_15_6 * 64, m_idx, tma_full_addr + (a_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (a_stage) * 8, 4096);
                        a_stage += 1;
                        if (a_stage == 18) { a_stage = 0; _phase_tma_empty ^= 1; }
                    }
                    int g0_15_7 = (rho * 4 + 1) * 4 + 3;
                    if (g0_15_7 < n_groups) {
                        mbarrier_wait(tma_empty_addr + (a_stage) * 8, _phase_tma_empty);
                        tma_2d_gmem2smem(smem_a_addr + a_stage * 4096, (&W), g0_15_7 * 64, m_idx, tma_full_addr + (a_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (a_stage) * 8, 4096);
                        a_stage += 1;
                        if (a_stage == 18) { a_stage = 0; _phase_tma_empty ^= 1; }
                    }
                    int g0_15_8 = (rho * 4 + 2) * 4;
                    if (g0_15_8 < n_groups) {
                        mbarrier_wait(tma_empty_addr + (a_stage) * 8, _phase_tma_empty);
                        tma_2d_gmem2smem(smem_a_addr + a_stage * 4096, (&W), g0_15_8 * 64, m_idx, tma_full_addr + (a_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (a_stage) * 8, 4096);
                        a_stage += 1;
                        if (a_stage == 18) { a_stage = 0; _phase_tma_empty ^= 1; }
                    }
                    int g0_15_9 = (rho * 4 + 2) * 4 + 1;
                    if (g0_15_9 < n_groups) {
                        mbarrier_wait(tma_empty_addr + (a_stage) * 8, _phase_tma_empty);
                        tma_2d_gmem2smem(smem_a_addr + a_stage * 4096, (&W), g0_15_9 * 64, m_idx, tma_full_addr + (a_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (a_stage) * 8, 4096);
                        a_stage += 1;
                        if (a_stage == 18) { a_stage = 0; _phase_tma_empty ^= 1; }
                    }
                    int g0_15_10 = (rho * 4 + 2) * 4 + 2;
                    if (g0_15_10 < n_groups) {
                        mbarrier_wait(tma_empty_addr + (a_stage) * 8, _phase_tma_empty);
                        tma_2d_gmem2smem(smem_a_addr + a_stage * 4096, (&W), g0_15_10 * 64, m_idx, tma_full_addr + (a_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (a_stage) * 8, 4096);
                        a_stage += 1;
                        if (a_stage == 18) { a_stage = 0; _phase_tma_empty ^= 1; }
                    }
                    int g0_15_11 = (rho * 4 + 2) * 4 + 3;
                    if (g0_15_11 < n_groups) {
                        mbarrier_wait(tma_empty_addr + (a_stage) * 8, _phase_tma_empty);
                        tma_2d_gmem2smem(smem_a_addr + a_stage * 4096, (&W), g0_15_11 * 64, m_idx, tma_full_addr + (a_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (a_stage) * 8, 4096);
                        a_stage += 1;
                        if (a_stage == 18) { a_stage = 0; _phase_tma_empty ^= 1; }
                    }
                    int g0_15_12 = (rho * 4 + 3) * 4;
                    if (g0_15_12 < n_groups) {
                        mbarrier_wait(tma_empty_addr + (a_stage) * 8, _phase_tma_empty);
                        tma_2d_gmem2smem(smem_a_addr + a_stage * 4096, (&W), g0_15_12 * 64, m_idx, tma_full_addr + (a_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (a_stage) * 8, 4096);
                        a_stage += 1;
                        if (a_stage == 18) { a_stage = 0; _phase_tma_empty ^= 1; }
                    }
                    int g0_15_13 = (rho * 4 + 3) * 4 + 1;
                    if (g0_15_13 < n_groups) {
                        mbarrier_wait(tma_empty_addr + (a_stage) * 8, _phase_tma_empty);
                        tma_2d_gmem2smem(smem_a_addr + a_stage * 4096, (&W), g0_15_13 * 64, m_idx, tma_full_addr + (a_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (a_stage) * 8, 4096);
                        a_stage += 1;
                        if (a_stage == 18) { a_stage = 0; _phase_tma_empty ^= 1; }
                    }
                    int g0_15_14 = (rho * 4 + 3) * 4 + 2;
                    if (g0_15_14 < n_groups) {
                        mbarrier_wait(tma_empty_addr + (a_stage) * 8, _phase_tma_empty);
                        tma_2d_gmem2smem(smem_a_addr + a_stage * 4096, (&W), g0_15_14 * 64, m_idx, tma_full_addr + (a_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (a_stage) * 8, 4096);
                        a_stage += 1;
                        if (a_stage == 18) { a_stage = 0; _phase_tma_empty ^= 1; }
                    }
                    int g0_15_15 = (rho * 4 + 3) * 4 + 3;
                    if (g0_15_15 < n_groups) {
                        mbarrier_wait(tma_empty_addr + (a_stage) * 8, _phase_tma_empty);
                        tma_2d_gmem2smem(smem_a_addr + a_stage * 4096, (&W), g0_15_15 * 64, m_idx, tma_full_addr + (a_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (a_stage) * 8, 4096);
                        a_stage += 1;
                        if (a_stage == 18) { a_stage = 0; _phase_tma_empty ^= 1; }
                    }
                }
            }
        }
    }
    // ---- Role: dma_b ----
    if (warp == 1) {
        { // dma_b_main
            unsigned int b_stage = 0;
            int n_idx = blockIdx.y * 64;
            int n_groups_b = (K + 64 - 1) / 64;
            int n_rounds_b = ((n_groups_b + 4 - 1) / 4 + 4 - 1) / 4;
            unsigned int _phase_tma_empty_1 = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int rho_1 = 0; rho_1 < n_rounds_b; rho_1++) {
                    int g0b_15_0 = rho_1 * 4 * 4;
                    if (g0b_15_0 < n_groups_b) {
                        mbarrier_wait(tma_empty_addr + (b_stage) * 8, _phase_tma_empty_1);
                        tma_2d_gmem2smem(smem_b_addr + b_stage * 8192, (&X), g0b_15_0 * 64, n_idx, tma_full_addr + (b_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (b_stage) * 8, 8192);
                        b_stage += 1;
                        if (b_stage == 18) { b_stage = 0; _phase_tma_empty_1 ^= 1; }
                    }
                    int g0b_15_1 = rho_1 * 4 * 4 + 1;
                    if (g0b_15_1 < n_groups_b) {
                        mbarrier_wait(tma_empty_addr + (b_stage) * 8, _phase_tma_empty_1);
                        tma_2d_gmem2smem(smem_b_addr + b_stage * 8192, (&X), g0b_15_1 * 64, n_idx, tma_full_addr + (b_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (b_stage) * 8, 8192);
                        b_stage += 1;
                        if (b_stage == 18) { b_stage = 0; _phase_tma_empty_1 ^= 1; }
                    }
                    int g0b_15_2 = rho_1 * 4 * 4 + 2;
                    if (g0b_15_2 < n_groups_b) {
                        mbarrier_wait(tma_empty_addr + (b_stage) * 8, _phase_tma_empty_1);
                        tma_2d_gmem2smem(smem_b_addr + b_stage * 8192, (&X), g0b_15_2 * 64, n_idx, tma_full_addr + (b_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (b_stage) * 8, 8192);
                        b_stage += 1;
                        if (b_stage == 18) { b_stage = 0; _phase_tma_empty_1 ^= 1; }
                    }
                    int g0b_15_3 = rho_1 * 4 * 4 + 3;
                    if (g0b_15_3 < n_groups_b) {
                        mbarrier_wait(tma_empty_addr + (b_stage) * 8, _phase_tma_empty_1);
                        tma_2d_gmem2smem(smem_b_addr + b_stage * 8192, (&X), g0b_15_3 * 64, n_idx, tma_full_addr + (b_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (b_stage) * 8, 8192);
                        b_stage += 1;
                        if (b_stage == 18) { b_stage = 0; _phase_tma_empty_1 ^= 1; }
                    }
                    int g0b_15_4 = (rho_1 * 4 + 1) * 4;
                    if (g0b_15_4 < n_groups_b) {
                        mbarrier_wait(tma_empty_addr + (b_stage) * 8, _phase_tma_empty_1);
                        tma_2d_gmem2smem(smem_b_addr + b_stage * 8192, (&X), g0b_15_4 * 64, n_idx, tma_full_addr + (b_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (b_stage) * 8, 8192);
                        b_stage += 1;
                        if (b_stage == 18) { b_stage = 0; _phase_tma_empty_1 ^= 1; }
                    }
                    int g0b_15_5 = (rho_1 * 4 + 1) * 4 + 1;
                    if (g0b_15_5 < n_groups_b) {
                        mbarrier_wait(tma_empty_addr + (b_stage) * 8, _phase_tma_empty_1);
                        tma_2d_gmem2smem(smem_b_addr + b_stage * 8192, (&X), g0b_15_5 * 64, n_idx, tma_full_addr + (b_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (b_stage) * 8, 8192);
                        b_stage += 1;
                        if (b_stage == 18) { b_stage = 0; _phase_tma_empty_1 ^= 1; }
                    }
                    int g0b_15_6 = (rho_1 * 4 + 1) * 4 + 2;
                    if (g0b_15_6 < n_groups_b) {
                        mbarrier_wait(tma_empty_addr + (b_stage) * 8, _phase_tma_empty_1);
                        tma_2d_gmem2smem(smem_b_addr + b_stage * 8192, (&X), g0b_15_6 * 64, n_idx, tma_full_addr + (b_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (b_stage) * 8, 8192);
                        b_stage += 1;
                        if (b_stage == 18) { b_stage = 0; _phase_tma_empty_1 ^= 1; }
                    }
                    int g0b_15_7 = (rho_1 * 4 + 1) * 4 + 3;
                    if (g0b_15_7 < n_groups_b) {
                        mbarrier_wait(tma_empty_addr + (b_stage) * 8, _phase_tma_empty_1);
                        tma_2d_gmem2smem(smem_b_addr + b_stage * 8192, (&X), g0b_15_7 * 64, n_idx, tma_full_addr + (b_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (b_stage) * 8, 8192);
                        b_stage += 1;
                        if (b_stage == 18) { b_stage = 0; _phase_tma_empty_1 ^= 1; }
                    }
                    int g0b_15_8 = (rho_1 * 4 + 2) * 4;
                    if (g0b_15_8 < n_groups_b) {
                        mbarrier_wait(tma_empty_addr + (b_stage) * 8, _phase_tma_empty_1);
                        tma_2d_gmem2smem(smem_b_addr + b_stage * 8192, (&X), g0b_15_8 * 64, n_idx, tma_full_addr + (b_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (b_stage) * 8, 8192);
                        b_stage += 1;
                        if (b_stage == 18) { b_stage = 0; _phase_tma_empty_1 ^= 1; }
                    }
                    int g0b_15_9 = (rho_1 * 4 + 2) * 4 + 1;
                    if (g0b_15_9 < n_groups_b) {
                        mbarrier_wait(tma_empty_addr + (b_stage) * 8, _phase_tma_empty_1);
                        tma_2d_gmem2smem(smem_b_addr + b_stage * 8192, (&X), g0b_15_9 * 64, n_idx, tma_full_addr + (b_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (b_stage) * 8, 8192);
                        b_stage += 1;
                        if (b_stage == 18) { b_stage = 0; _phase_tma_empty_1 ^= 1; }
                    }
                    int g0b_15_10 = (rho_1 * 4 + 2) * 4 + 2;
                    if (g0b_15_10 < n_groups_b) {
                        mbarrier_wait(tma_empty_addr + (b_stage) * 8, _phase_tma_empty_1);
                        tma_2d_gmem2smem(smem_b_addr + b_stage * 8192, (&X), g0b_15_10 * 64, n_idx, tma_full_addr + (b_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (b_stage) * 8, 8192);
                        b_stage += 1;
                        if (b_stage == 18) { b_stage = 0; _phase_tma_empty_1 ^= 1; }
                    }
                    int g0b_15_11 = (rho_1 * 4 + 2) * 4 + 3;
                    if (g0b_15_11 < n_groups_b) {
                        mbarrier_wait(tma_empty_addr + (b_stage) * 8, _phase_tma_empty_1);
                        tma_2d_gmem2smem(smem_b_addr + b_stage * 8192, (&X), g0b_15_11 * 64, n_idx, tma_full_addr + (b_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (b_stage) * 8, 8192);
                        b_stage += 1;
                        if (b_stage == 18) { b_stage = 0; _phase_tma_empty_1 ^= 1; }
                    }
                    int g0b_15_12 = (rho_1 * 4 + 3) * 4;
                    if (g0b_15_12 < n_groups_b) {
                        mbarrier_wait(tma_empty_addr + (b_stage) * 8, _phase_tma_empty_1);
                        tma_2d_gmem2smem(smem_b_addr + b_stage * 8192, (&X), g0b_15_12 * 64, n_idx, tma_full_addr + (b_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (b_stage) * 8, 8192);
                        b_stage += 1;
                        if (b_stage == 18) { b_stage = 0; _phase_tma_empty_1 ^= 1; }
                    }
                    int g0b_15_13 = (rho_1 * 4 + 3) * 4 + 1;
                    if (g0b_15_13 < n_groups_b) {
                        mbarrier_wait(tma_empty_addr + (b_stage) * 8, _phase_tma_empty_1);
                        tma_2d_gmem2smem(smem_b_addr + b_stage * 8192, (&X), g0b_15_13 * 64, n_idx, tma_full_addr + (b_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (b_stage) * 8, 8192);
                        b_stage += 1;
                        if (b_stage == 18) { b_stage = 0; _phase_tma_empty_1 ^= 1; }
                    }
                    int g0b_15_14 = (rho_1 * 4 + 3) * 4 + 2;
                    if (g0b_15_14 < n_groups_b) {
                        mbarrier_wait(tma_empty_addr + (b_stage) * 8, _phase_tma_empty_1);
                        tma_2d_gmem2smem(smem_b_addr + b_stage * 8192, (&X), g0b_15_14 * 64, n_idx, tma_full_addr + (b_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (b_stage) * 8, 8192);
                        b_stage += 1;
                        if (b_stage == 18) { b_stage = 0; _phase_tma_empty_1 ^= 1; }
                    }
                    int g0b_15_15 = (rho_1 * 4 + 3) * 4 + 3;
                    if (g0b_15_15 < n_groups_b) {
                        mbarrier_wait(tma_empty_addr + (b_stage) * 8, _phase_tma_empty_1);
                        tma_2d_gmem2smem(smem_b_addr + b_stage * 8192, (&X), g0b_15_15 * 64, n_idx, tma_full_addr + (b_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (b_stage) * 8, 8192);
                        b_stage += 1;
                        if (b_stage == 18) { b_stage = 0; _phase_tma_empty_1 ^= 1; }
                    }
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 2) {
        { // mma_main
            unsigned int mma_stage = 0;
            int n_groups_m = (K + 64 - 1) / 64;
            int n_rounds_m = ((n_groups_m + 4 - 1) / 4 + 4 - 1) / 4;
            unsigned int _phase_tma_full = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int rho_2 = 0; rho_2 < n_rounds_m; rho_2++) {
                    int g0m_15_0 = rho_2 * 4 * 4;
                    if (g0m_15_0 < n_groups_m) {
                        mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int init_flag_1_0 = ((rho_2 == 0) ? 1 : 0);
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_0 = (((smem_b_addr) >> 4) & 0x7FFF) + (mma_stage) * 512;
                        int _smem_desc_lo_1 = (((smem_a_addr) >> 4) & 0x7FFF) + (mma_stage) * 256;
#else
                        int _smem_desc_lo_0 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_stage) * 512;
                        int _smem_desc_lo_1 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage) * 256;
#endif
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
                    "mov.b32 id, 67634320;\n\t"
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
                    "}\n"
                    :: "r"(_smem_desc_lo_0), "r"(_smem_desc_lo_1), "r"(tmem_acc0), "r"(((init_flag_1_0) ? 0 : 1)) : "memory");
                        tcgen05_commit(tma_empty_addr + (mma_stage) * 8);
                        mma_stage += 1;
                        if (mma_stage == 18) { mma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    int g0m_0 = rho_2 * 4 * 4 + 1;
                    if (g0m_0 < n_groups_m) {
                        mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_2 = (((smem_b_addr) >> 4) & 0x7FFF) + (mma_stage) * 512;
                        int _smem_desc_lo_3 = (((smem_a_addr) >> 4) & 0x7FFF) + (mma_stage) * 256;
#else
                        int _smem_desc_lo_2 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_stage) * 512;
                        int _smem_desc_lo_3 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage) * 256;
#endif
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
                    "mov.b32 id, 67634320;\n\t"
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
                    "}\n"
                    :: "r"(_smem_desc_lo_2), "r"(_smem_desc_lo_3), "r"(tmem_acc0), "r"(1) : "memory");
                        tcgen05_commit(tma_empty_addr + (mma_stage) * 8);
                        mma_stage += 1;
                        if (mma_stage == 18) { mma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    int g0m_1 = rho_2 * 4 * 4 + 2;
                    if (g0m_1 < n_groups_m) {
                        mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_4 = (((smem_b_addr) >> 4) & 0x7FFF) + (mma_stage) * 512;
                        int _smem_desc_lo_5 = (((smem_a_addr) >> 4) & 0x7FFF) + (mma_stage) * 256;
#else
                        int _smem_desc_lo_4 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_stage) * 512;
                        int _smem_desc_lo_5 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage) * 256;
#endif
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
                    "mov.b32 id, 67634320;\n\t"
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
                    "}\n"
                    :: "r"(_smem_desc_lo_4), "r"(_smem_desc_lo_5), "r"(tmem_acc0), "r"(1) : "memory");
                        tcgen05_commit(tma_empty_addr + (mma_stage) * 8);
                        mma_stage += 1;
                        if (mma_stage == 18) { mma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    int g0m_2 = rho_2 * 4 * 4 + 3;
                    if (g0m_2 < n_groups_m) {
                        mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_6 = (((smem_b_addr) >> 4) & 0x7FFF) + (mma_stage) * 512;
                        int _smem_desc_lo_7 = (((smem_a_addr) >> 4) & 0x7FFF) + (mma_stage) * 256;
#else
                        int _smem_desc_lo_6 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_stage) * 512;
                        int _smem_desc_lo_7 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage) * 256;
#endif
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
                    "mov.b32 id, 67634320;\n\t"
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
                    "}\n"
                    :: "r"(_smem_desc_lo_6), "r"(_smem_desc_lo_7), "r"(tmem_acc0), "r"(1) : "memory");
                        tcgen05_commit(tma_empty_addr + (mma_stage) * 8);
                        mma_stage += 1;
                        if (mma_stage == 18) { mma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    int g0m_15_1 = (rho_2 * 4 + 1) * 4;
                    if (g0m_15_1 < n_groups_m) {
                        mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int init_flag_1_1 = ((rho_2 == 0) ? 1 : 0);
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_8 = (((smem_b_addr) >> 4) & 0x7FFF) + (mma_stage) * 512;
                        int _smem_desc_lo_9 = (((smem_a_addr) >> 4) & 0x7FFF) + (mma_stage) * 256;
#else
                        int _smem_desc_lo_8 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_stage) * 512;
                        int _smem_desc_lo_9 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage) * 256;
#endif
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
                    "mov.b32 id, 67634320;\n\t"
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
                    "}\n"
                    :: "r"(_smem_desc_lo_8), "r"(_smem_desc_lo_9), "r"(tmem_acc1), "r"(((init_flag_1_1) ? 0 : 1)) : "memory");
                        tcgen05_commit(tma_empty_addr + (mma_stage) * 8);
                        mma_stage += 1;
                        if (mma_stage == 18) { mma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    int g0m_3 = (rho_2 * 4 + 1) * 4 + 1;
                    if (g0m_3 < n_groups_m) {
                        mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_10 = (((smem_b_addr) >> 4) & 0x7FFF) + (mma_stage) * 512;
                        int _smem_desc_lo_11 = (((smem_a_addr) >> 4) & 0x7FFF) + (mma_stage) * 256;
#else
                        int _smem_desc_lo_10 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_stage) * 512;
                        int _smem_desc_lo_11 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage) * 256;
#endif
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
                    "mov.b32 id, 67634320;\n\t"
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
                    "}\n"
                    :: "r"(_smem_desc_lo_10), "r"(_smem_desc_lo_11), "r"(tmem_acc1), "r"(1) : "memory");
                        tcgen05_commit(tma_empty_addr + (mma_stage) * 8);
                        mma_stage += 1;
                        if (mma_stage == 18) { mma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    int g0m_4 = (rho_2 * 4 + 1) * 4 + 2;
                    if (g0m_4 < n_groups_m) {
                        mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_12 = (((smem_b_addr) >> 4) & 0x7FFF) + (mma_stage) * 512;
                        int _smem_desc_lo_13 = (((smem_a_addr) >> 4) & 0x7FFF) + (mma_stage) * 256;
#else
                        int _smem_desc_lo_12 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_stage) * 512;
                        int _smem_desc_lo_13 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage) * 256;
#endif
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
                    "mov.b32 id, 67634320;\n\t"
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
                    "}\n"
                    :: "r"(_smem_desc_lo_12), "r"(_smem_desc_lo_13), "r"(tmem_acc1), "r"(1) : "memory");
                        tcgen05_commit(tma_empty_addr + (mma_stage) * 8);
                        mma_stage += 1;
                        if (mma_stage == 18) { mma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    int g0m_5 = (rho_2 * 4 + 1) * 4 + 3;
                    if (g0m_5 < n_groups_m) {
                        mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_14 = (((smem_b_addr) >> 4) & 0x7FFF) + (mma_stage) * 512;
                        int _smem_desc_lo_15 = (((smem_a_addr) >> 4) & 0x7FFF) + (mma_stage) * 256;
#else
                        int _smem_desc_lo_14 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_stage) * 512;
                        int _smem_desc_lo_15 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage) * 256;
#endif
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
                    "mov.b32 id, 67634320;\n\t"
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
                    "}\n"
                    :: "r"(_smem_desc_lo_14), "r"(_smem_desc_lo_15), "r"(tmem_acc1), "r"(1) : "memory");
                        tcgen05_commit(tma_empty_addr + (mma_stage) * 8);
                        mma_stage += 1;
                        if (mma_stage == 18) { mma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    int g0m_15_2 = (rho_2 * 4 + 2) * 4;
                    if (g0m_15_2 < n_groups_m) {
                        mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int init_flag_1_2 = ((rho_2 == 0) ? 1 : 0);
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_16 = (((smem_b_addr) >> 4) & 0x7FFF) + (mma_stage) * 512;
                        int _smem_desc_lo_17 = (((smem_a_addr) >> 4) & 0x7FFF) + (mma_stage) * 256;
#else
                        int _smem_desc_lo_16 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_stage) * 512;
                        int _smem_desc_lo_17 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage) * 256;
#endif
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
                    "mov.b32 id, 67634320;\n\t"
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
                    "}\n"
                    :: "r"(_smem_desc_lo_16), "r"(_smem_desc_lo_17), "r"(tmem_acc2), "r"(((init_flag_1_2) ? 0 : 1)) : "memory");
                        tcgen05_commit(tma_empty_addr + (mma_stage) * 8);
                        mma_stage += 1;
                        if (mma_stage == 18) { mma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    int g0m_6 = (rho_2 * 4 + 2) * 4 + 1;
                    if (g0m_6 < n_groups_m) {
                        mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_18 = (((smem_b_addr) >> 4) & 0x7FFF) + (mma_stage) * 512;
                        int _smem_desc_lo_19 = (((smem_a_addr) >> 4) & 0x7FFF) + (mma_stage) * 256;
#else
                        int _smem_desc_lo_18 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_stage) * 512;
                        int _smem_desc_lo_19 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage) * 256;
#endif
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
                    "mov.b32 id, 67634320;\n\t"
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
                    "}\n"
                    :: "r"(_smem_desc_lo_18), "r"(_smem_desc_lo_19), "r"(tmem_acc2), "r"(1) : "memory");
                        tcgen05_commit(tma_empty_addr + (mma_stage) * 8);
                        mma_stage += 1;
                        if (mma_stage == 18) { mma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    int g0m_7 = (rho_2 * 4 + 2) * 4 + 2;
                    if (g0m_7 < n_groups_m) {
                        mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_20 = (((smem_b_addr) >> 4) & 0x7FFF) + (mma_stage) * 512;
                        int _smem_desc_lo_21 = (((smem_a_addr) >> 4) & 0x7FFF) + (mma_stage) * 256;
#else
                        int _smem_desc_lo_20 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_stage) * 512;
                        int _smem_desc_lo_21 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage) * 256;
#endif
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
                    "mov.b32 id, 67634320;\n\t"
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
                    "}\n"
                    :: "r"(_smem_desc_lo_20), "r"(_smem_desc_lo_21), "r"(tmem_acc2), "r"(1) : "memory");
                        tcgen05_commit(tma_empty_addr + (mma_stage) * 8);
                        mma_stage += 1;
                        if (mma_stage == 18) { mma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    int g0m_8 = (rho_2 * 4 + 2) * 4 + 3;
                    if (g0m_8 < n_groups_m) {
                        mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_22 = (((smem_b_addr) >> 4) & 0x7FFF) + (mma_stage) * 512;
                        int _smem_desc_lo_23 = (((smem_a_addr) >> 4) & 0x7FFF) + (mma_stage) * 256;
#else
                        int _smem_desc_lo_22 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_stage) * 512;
                        int _smem_desc_lo_23 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage) * 256;
#endif
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
                    "mov.b32 id, 67634320;\n\t"
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
                    "}\n"
                    :: "r"(_smem_desc_lo_22), "r"(_smem_desc_lo_23), "r"(tmem_acc2), "r"(1) : "memory");
                        tcgen05_commit(tma_empty_addr + (mma_stage) * 8);
                        mma_stage += 1;
                        if (mma_stage == 18) { mma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    int g0m_15_3 = (rho_2 * 4 + 3) * 4;
                    if (g0m_15_3 < n_groups_m) {
                        mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int init_flag_1_3 = ((rho_2 == 0) ? 1 : 0);
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_24 = (((smem_b_addr) >> 4) & 0x7FFF) + (mma_stage) * 512;
                        int _smem_desc_lo_25 = (((smem_a_addr) >> 4) & 0x7FFF) + (mma_stage) * 256;
#else
                        int _smem_desc_lo_24 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_stage) * 512;
                        int _smem_desc_lo_25 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage) * 256;
#endif
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
                    "mov.b32 id, 67634320;\n\t"
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
                    "}\n"
                    :: "r"(_smem_desc_lo_24), "r"(_smem_desc_lo_25), "r"(tmem_acc3), "r"(((init_flag_1_3) ? 0 : 1)) : "memory");
                        tcgen05_commit(tma_empty_addr + (mma_stage) * 8);
                        mma_stage += 1;
                        if (mma_stage == 18) { mma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    int g0m_9 = (rho_2 * 4 + 3) * 4 + 1;
                    if (g0m_9 < n_groups_m) {
                        mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_26 = (((smem_b_addr) >> 4) & 0x7FFF) + (mma_stage) * 512;
                        int _smem_desc_lo_27 = (((smem_a_addr) >> 4) & 0x7FFF) + (mma_stage) * 256;
#else
                        int _smem_desc_lo_26 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_stage) * 512;
                        int _smem_desc_lo_27 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage) * 256;
#endif
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
                    "mov.b32 id, 67634320;\n\t"
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
                    "}\n"
                    :: "r"(_smem_desc_lo_26), "r"(_smem_desc_lo_27), "r"(tmem_acc3), "r"(1) : "memory");
                        tcgen05_commit(tma_empty_addr + (mma_stage) * 8);
                        mma_stage += 1;
                        if (mma_stage == 18) { mma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    int g0m_10 = (rho_2 * 4 + 3) * 4 + 2;
                    if (g0m_10 < n_groups_m) {
                        mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_28 = (((smem_b_addr) >> 4) & 0x7FFF) + (mma_stage) * 512;
                        int _smem_desc_lo_29 = (((smem_a_addr) >> 4) & 0x7FFF) + (mma_stage) * 256;
#else
                        int _smem_desc_lo_28 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_stage) * 512;
                        int _smem_desc_lo_29 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage) * 256;
#endif
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
                    "mov.b32 id, 67634320;\n\t"
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
                    "}\n"
                    :: "r"(_smem_desc_lo_28), "r"(_smem_desc_lo_29), "r"(tmem_acc3), "r"(1) : "memory");
                        tcgen05_commit(tma_empty_addr + (mma_stage) * 8);
                        mma_stage += 1;
                        if (mma_stage == 18) { mma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    int g0m_11 = (rho_2 * 4 + 3) * 4 + 3;
                    if (g0m_11 < n_groups_m) {
                        mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_30 = (((smem_b_addr) >> 4) & 0x7FFF) + (mma_stage) * 512;
                        int _smem_desc_lo_31 = (((smem_a_addr) >> 4) & 0x7FFF) + (mma_stage) * 256;
#else
                        int _smem_desc_lo_30 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_stage) * 512;
                        int _smem_desc_lo_31 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage) * 256;
#endif
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
                    "mov.b32 id, 67634320;\n\t"
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
                    "}\n"
                    :: "r"(_smem_desc_lo_30), "r"(_smem_desc_lo_31), "r"(tmem_acc3), "r"(1) : "memory");
                        tcgen05_commit(tma_empty_addr + (mma_stage) * 8);
                        mma_stage += 1;
                        if (mma_stage == 18) { mma_stage = 0; _phase_tma_full ^= 1; }
                    }
                }
                tcgen05_commit(mma_epi_full_addr);
            }
        }
    }
    // ---- Role: freer ----
    if (warp == 3) {
        { // freer_main
            unsigned int _phase_tmem_free_0 = 0;
            mbarrier_wait(tmem_free_addr, _phase_tmem_free_0);
            _phase_tmem_free_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(128));
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 4 && warp <= 7) {
        { // epilogue_main
            const int epi_warp_s = warp % 4;
            int n_chunks_s = ((K + 64 - 1) / 64 + 4 - 1) / 4;
            int my_n = blockIdx.y * 64 + epi_warp_s * 16 + lane % 16;
            int m_run = blockIdx.x * 32 + lane / 16 * 16;
            float bias_run[16];
            int _min_0 = ((m_run + 0) < (M - 1) ? (m_run + 0) : (M - 1));
            bias_run[0] = (float)bias[_min_0];
            int _min_1 = ((m_run + 1) < (M - 1) ? (m_run + 1) : (M - 1));
            bias_run[1] = (float)bias[_min_1];
            int _min_2 = ((m_run + 2) < (M - 1) ? (m_run + 2) : (M - 1));
            bias_run[2] = (float)bias[_min_2];
            int _min_3 = ((m_run + 3) < (M - 1) ? (m_run + 3) : (M - 1));
            bias_run[3] = (float)bias[_min_3];
            int _min_4 = ((m_run + 4) < (M - 1) ? (m_run + 4) : (M - 1));
            bias_run[4] = (float)bias[_min_4];
            int _min_5 = ((m_run + 5) < (M - 1) ? (m_run + 5) : (M - 1));
            bias_run[5] = (float)bias[_min_5];
            int _min_6 = ((m_run + 6) < (M - 1) ? (m_run + 6) : (M - 1));
            bias_run[6] = (float)bias[_min_6];
            int _min_7 = ((m_run + 7) < (M - 1) ? (m_run + 7) : (M - 1));
            bias_run[7] = (float)bias[_min_7];
            int _min_8 = ((m_run + 8) < (M - 1) ? (m_run + 8) : (M - 1));
            bias_run[8] = (float)bias[_min_8];
            int _min_9 = ((m_run + 9) < (M - 1) ? (m_run + 9) : (M - 1));
            bias_run[9] = (float)bias[_min_9];
            int _min_10 = ((m_run + 10) < (M - 1) ? (m_run + 10) : (M - 1));
            bias_run[10] = (float)bias[_min_10];
            int _min_11 = ((m_run + 11) < (M - 1) ? (m_run + 11) : (M - 1));
            bias_run[11] = (float)bias[_min_11];
            int _min_12 = ((m_run + 12) < (M - 1) ? (m_run + 12) : (M - 1));
            bias_run[12] = (float)bias[_min_12];
            int _min_13 = ((m_run + 13) < (M - 1) ? (m_run + 13) : (M - 1));
            bias_run[13] = (float)bias[_min_13];
            int _min_14 = ((m_run + 14) < (M - 1) ? (m_run + 14) : (M - 1));
            bias_run[14] = (float)bias[_min_14];
            int _min_15 = ((m_run + 15) < (M - 1) ? (m_run + 15) : (M - 1));
            bias_run[15] = (float)bias[_min_15];
            unsigned int _phase_mma_epi_full_0 = 0;
            mbarrier_wait(mma_epi_full_addr, _phase_mma_epi_full_0);
            _phase_mma_epi_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            unsigned int tmem_addr_s = taddr + (unsigned int)(epi_warp_s * 32 << 16);
            float chains[64];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                : "=r"(*reinterpret_cast<uint32_t*>(&chains[0])), "=r"(*reinterpret_cast<uint32_t*>(&chains[1])), "=r"(*reinterpret_cast<uint32_t*>(&chains[2])), "=r"(*reinterpret_cast<uint32_t*>(&chains[3])), "=r"(*reinterpret_cast<uint32_t*>(&chains[4])), "=r"(*reinterpret_cast<uint32_t*>(&chains[5])), "=r"(*reinterpret_cast<uint32_t*>(&chains[6])), "=r"(*reinterpret_cast<uint32_t*>(&chains[7])), "=r"(*reinterpret_cast<uint32_t*>(&chains[8])), "=r"(*reinterpret_cast<uint32_t*>(&chains[9])), "=r"(*reinterpret_cast<uint32_t*>(&chains[10])), "=r"(*reinterpret_cast<uint32_t*>(&chains[11])), "=r"(*reinterpret_cast<uint32_t*>(&chains[12])), "=r"(*reinterpret_cast<uint32_t*>(&chains[13])), "=r"(*reinterpret_cast<uint32_t*>(&chains[14])), "=r"(*reinterpret_cast<uint32_t*>(&chains[15]))
                : "r"(tmem_addr_s));
            asm volatile(
                "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                : "=r"(*reinterpret_cast<uint32_t*>(&chains[16])), "=r"(*reinterpret_cast<uint32_t*>(&chains[17])), "=r"(*reinterpret_cast<uint32_t*>(&chains[18])), "=r"(*reinterpret_cast<uint32_t*>(&chains[19])), "=r"(*reinterpret_cast<uint32_t*>(&chains[20])), "=r"(*reinterpret_cast<uint32_t*>(&chains[21])), "=r"(*reinterpret_cast<uint32_t*>(&chains[22])), "=r"(*reinterpret_cast<uint32_t*>(&chains[23])), "=r"(*reinterpret_cast<uint32_t*>(&chains[24])), "=r"(*reinterpret_cast<uint32_t*>(&chains[25])), "=r"(*reinterpret_cast<uint32_t*>(&chains[26])), "=r"(*reinterpret_cast<uint32_t*>(&chains[27])), "=r"(*reinterpret_cast<uint32_t*>(&chains[28])), "=r"(*reinterpret_cast<uint32_t*>(&chains[29])), "=r"(*reinterpret_cast<uint32_t*>(&chains[30])), "=r"(*reinterpret_cast<uint32_t*>(&chains[31]))
                : "r"(tmem_addr_s + 32));
            asm volatile(
                "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                : "=r"(*reinterpret_cast<uint32_t*>(&chains[32])), "=r"(*reinterpret_cast<uint32_t*>(&chains[33])), "=r"(*reinterpret_cast<uint32_t*>(&chains[34])), "=r"(*reinterpret_cast<uint32_t*>(&chains[35])), "=r"(*reinterpret_cast<uint32_t*>(&chains[36])), "=r"(*reinterpret_cast<uint32_t*>(&chains[37])), "=r"(*reinterpret_cast<uint32_t*>(&chains[38])), "=r"(*reinterpret_cast<uint32_t*>(&chains[39])), "=r"(*reinterpret_cast<uint32_t*>(&chains[40])), "=r"(*reinterpret_cast<uint32_t*>(&chains[41])), "=r"(*reinterpret_cast<uint32_t*>(&chains[42])), "=r"(*reinterpret_cast<uint32_t*>(&chains[43])), "=r"(*reinterpret_cast<uint32_t*>(&chains[44])), "=r"(*reinterpret_cast<uint32_t*>(&chains[45])), "=r"(*reinterpret_cast<uint32_t*>(&chains[46])), "=r"(*reinterpret_cast<uint32_t*>(&chains[47]))
                : "r"(tmem_addr_s + 64));
            asm volatile(
                "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                : "=r"(*reinterpret_cast<uint32_t*>(&chains[48])), "=r"(*reinterpret_cast<uint32_t*>(&chains[49])), "=r"(*reinterpret_cast<uint32_t*>(&chains[50])), "=r"(*reinterpret_cast<uint32_t*>(&chains[51])), "=r"(*reinterpret_cast<uint32_t*>(&chains[52])), "=r"(*reinterpret_cast<uint32_t*>(&chains[53])), "=r"(*reinterpret_cast<uint32_t*>(&chains[54])), "=r"(*reinterpret_cast<uint32_t*>(&chains[55])), "=r"(*reinterpret_cast<uint32_t*>(&chains[56])), "=r"(*reinterpret_cast<uint32_t*>(&chains[57])), "=r"(*reinterpret_cast<uint32_t*>(&chains[58])), "=r"(*reinterpret_cast<uint32_t*>(&chains[59])), "=r"(*reinterpret_cast<uint32_t*>(&chains[60])), "=r"(*reinterpret_cast<uint32_t*>(&chains[61])), "=r"(*reinterpret_cast<uint32_t*>(&chains[62])), "=r"(*reinterpret_cast<uint32_t*>(&chains[63]))
                : "r"(tmem_addr_s + 96));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            __syncwarp();
            if (elect_sync()) {
                mbarrier_arrive(tmem_free_addr);
            }
            float run[16];
            run[0] = ((n_chunks_s > 0) ? chains[0] : 0.0f) + ((n_chunks_s > 1) ? chains[16] : 0.0f) + ((n_chunks_s > 2) ? chains[32] : 0.0f) + ((n_chunks_s > 3) ? chains[48] : 0.0f);
            run[0] = run[0] + bias_run[0];
            run[1] = ((n_chunks_s > 0) ? chains[1] : 0.0f) + ((n_chunks_s > 1) ? chains[17] : 0.0f) + ((n_chunks_s > 2) ? chains[33] : 0.0f) + ((n_chunks_s > 3) ? chains[49] : 0.0f);
            run[1] = run[1] + bias_run[1];
            run[2] = ((n_chunks_s > 0) ? chains[2] : 0.0f) + ((n_chunks_s > 1) ? chains[18] : 0.0f) + ((n_chunks_s > 2) ? chains[34] : 0.0f) + ((n_chunks_s > 3) ? chains[50] : 0.0f);
            run[2] = run[2] + bias_run[2];
            run[3] = ((n_chunks_s > 0) ? chains[3] : 0.0f) + ((n_chunks_s > 1) ? chains[19] : 0.0f) + ((n_chunks_s > 2) ? chains[35] : 0.0f) + ((n_chunks_s > 3) ? chains[51] : 0.0f);
            run[3] = run[3] + bias_run[3];
            run[4] = ((n_chunks_s > 0) ? chains[4] : 0.0f) + ((n_chunks_s > 1) ? chains[20] : 0.0f) + ((n_chunks_s > 2) ? chains[36] : 0.0f) + ((n_chunks_s > 3) ? chains[52] : 0.0f);
            run[4] = run[4] + bias_run[4];
            run[5] = ((n_chunks_s > 0) ? chains[5] : 0.0f) + ((n_chunks_s > 1) ? chains[21] : 0.0f) + ((n_chunks_s > 2) ? chains[37] : 0.0f) + ((n_chunks_s > 3) ? chains[53] : 0.0f);
            run[5] = run[5] + bias_run[5];
            run[6] = ((n_chunks_s > 0) ? chains[6] : 0.0f) + ((n_chunks_s > 1) ? chains[22] : 0.0f) + ((n_chunks_s > 2) ? chains[38] : 0.0f) + ((n_chunks_s > 3) ? chains[54] : 0.0f);
            run[6] = run[6] + bias_run[6];
            run[7] = ((n_chunks_s > 0) ? chains[7] : 0.0f) + ((n_chunks_s > 1) ? chains[23] : 0.0f) + ((n_chunks_s > 2) ? chains[39] : 0.0f) + ((n_chunks_s > 3) ? chains[55] : 0.0f);
            run[7] = run[7] + bias_run[7];
            run[8] = ((n_chunks_s > 0) ? chains[8] : 0.0f) + ((n_chunks_s > 1) ? chains[24] : 0.0f) + ((n_chunks_s > 2) ? chains[40] : 0.0f) + ((n_chunks_s > 3) ? chains[56] : 0.0f);
            run[8] = run[8] + bias_run[8];
            run[9] = ((n_chunks_s > 0) ? chains[9] : 0.0f) + ((n_chunks_s > 1) ? chains[25] : 0.0f) + ((n_chunks_s > 2) ? chains[41] : 0.0f) + ((n_chunks_s > 3) ? chains[57] : 0.0f);
            run[9] = run[9] + bias_run[9];
            run[10] = ((n_chunks_s > 0) ? chains[10] : 0.0f) + ((n_chunks_s > 1) ? chains[26] : 0.0f) + ((n_chunks_s > 2) ? chains[42] : 0.0f) + ((n_chunks_s > 3) ? chains[58] : 0.0f);
            run[10] = run[10] + bias_run[10];
            run[11] = ((n_chunks_s > 0) ? chains[11] : 0.0f) + ((n_chunks_s > 1) ? chains[27] : 0.0f) + ((n_chunks_s > 2) ? chains[43] : 0.0f) + ((n_chunks_s > 3) ? chains[59] : 0.0f);
            run[11] = run[11] + bias_run[11];
            run[12] = ((n_chunks_s > 0) ? chains[12] : 0.0f) + ((n_chunks_s > 1) ? chains[28] : 0.0f) + ((n_chunks_s > 2) ? chains[44] : 0.0f) + ((n_chunks_s > 3) ? chains[60] : 0.0f);
            run[12] = run[12] + bias_run[12];
            run[13] = ((n_chunks_s > 0) ? chains[13] : 0.0f) + ((n_chunks_s > 1) ? chains[29] : 0.0f) + ((n_chunks_s > 2) ? chains[45] : 0.0f) + ((n_chunks_s > 3) ? chains[61] : 0.0f);
            run[13] = run[13] + bias_run[13];
            run[14] = ((n_chunks_s > 0) ? chains[14] : 0.0f) + ((n_chunks_s > 1) ? chains[30] : 0.0f) + ((n_chunks_s > 2) ? chains[46] : 0.0f) + ((n_chunks_s > 3) ? chains[62] : 0.0f);
            run[14] = run[14] + bias_run[14];
            run[15] = ((n_chunks_s > 0) ? chains[15] : 0.0f) + ((n_chunks_s > 1) ? chains[31] : 0.0f) + ((n_chunks_s > 2) ? chains[47] : 0.0f) + ((n_chunks_s > 3) ? chains[63] : 0.0f);
            run[15] = run[15] + bias_run[15];
            unsigned int packed[8];
            if (my_n < N) {
                if (m_run + 16 <= M && M % 16 == 0) {
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(run[_lp*2 + 0], run[_lp*2+1 + 0]));
                        packed[_lp] = *(uint32_t*)&_bf2;
                    }
                    {
                        const unsigned* _raw_stv8_16 = reinterpret_cast<const unsigned*>(packed);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(out + (my_n * M + m_run))), "r"(_raw_stv8_16[0]), "r"(_raw_stv8_16[1]), "r"(_raw_stv8_16[2]), "r"(_raw_stv8_16[3]), "r"(_raw_stv8_16[4]), "r"(_raw_stv8_16[5]), "r"(_raw_stv8_16[6]), "r"(_raw_stv8_16[7]) : "memory");
                    }
                } else {
                    if (m_run + 0 < M) {
                        *(reinterpret_cast<__nv_bfloat16*>(out + (my_n * M + m_run + 0)) + (0)) = (__nv_bfloat16)run[0];
                    }
                    if (m_run + 1 < M) {
                        *(reinterpret_cast<__nv_bfloat16*>(out + (my_n * M + m_run + 1)) + (0)) = (__nv_bfloat16)run[1];
                    }
                    if (m_run + 2 < M) {
                        *(reinterpret_cast<__nv_bfloat16*>(out + (my_n * M + m_run + 2)) + (0)) = (__nv_bfloat16)run[2];
                    }
                    if (m_run + 3 < M) {
                        *(reinterpret_cast<__nv_bfloat16*>(out + (my_n * M + m_run + 3)) + (0)) = (__nv_bfloat16)run[3];
                    }
                    if (m_run + 4 < M) {
                        *(reinterpret_cast<__nv_bfloat16*>(out + (my_n * M + m_run + 4)) + (0)) = (__nv_bfloat16)run[4];
                    }
                    if (m_run + 5 < M) {
                        *(reinterpret_cast<__nv_bfloat16*>(out + (my_n * M + m_run + 5)) + (0)) = (__nv_bfloat16)run[5];
                    }
                    if (m_run + 6 < M) {
                        *(reinterpret_cast<__nv_bfloat16*>(out + (my_n * M + m_run + 6)) + (0)) = (__nv_bfloat16)run[6];
                    }
                    if (m_run + 7 < M) {
                        *(reinterpret_cast<__nv_bfloat16*>(out + (my_n * M + m_run + 7)) + (0)) = (__nv_bfloat16)run[7];
                    }
                    if (m_run + 8 < M) {
                        *(reinterpret_cast<__nv_bfloat16*>(out + (my_n * M + m_run + 8)) + (0)) = (__nv_bfloat16)run[8];
                    }
                    if (m_run + 9 < M) {
                        *(reinterpret_cast<__nv_bfloat16*>(out + (my_n * M + m_run + 9)) + (0)) = (__nv_bfloat16)run[9];
                    }
                    if (m_run + 10 < M) {
                        *(reinterpret_cast<__nv_bfloat16*>(out + (my_n * M + m_run + 10)) + (0)) = (__nv_bfloat16)run[10];
                    }
                    if (m_run + 11 < M) {
                        *(reinterpret_cast<__nv_bfloat16*>(out + (my_n * M + m_run + 11)) + (0)) = (__nv_bfloat16)run[11];
                    }
                    if (m_run + 12 < M) {
                        *(reinterpret_cast<__nv_bfloat16*>(out + (my_n * M + m_run + 12)) + (0)) = (__nv_bfloat16)run[12];
                    }
                    if (m_run + 13 < M) {
                        *(reinterpret_cast<__nv_bfloat16*>(out + (my_n * M + m_run + 13)) + (0)) = (__nv_bfloat16)run[13];
                    }
                    if (m_run + 14 < M) {
                        *(reinterpret_cast<__nv_bfloat16*>(out + (my_n * M + m_run + 14)) + (0)) = (__nv_bfloat16)run[14];
                    }
                    if (m_run + 15 < M) {
                        *(reinterpret_cast<__nv_bfloat16*>(out + (my_n * M + m_run + 15)) + (0)) = (__nv_bfloat16)run[15];
                    }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
