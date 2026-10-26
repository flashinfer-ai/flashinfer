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
#define TMEM_NCOLS 256
#define TMEM_ACC0_OFFSET 0
#define TMEM_ACC1_OFFSET 64
#define TMEM_ACC2_OFFSET 128
#define TMEM_ACC3_OFFSET 192
#define NUM_TMA_PIPE_STAGES 18
#define NUM_EPI_PIPE_STAGES 1
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 8192
#define SMEM_SMEM_A_STRIDE 4096
#define SMEM_SMEM_B_OFF 78848
#define SMEM_SMEM_B_STAGE_BYTES 8192
#define SMEM_SMEM_B_STRIDE 8192
#define SMEM_TOTAL 226304
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
kernel_cake_tinygemm2_tcgen05_46f583fba87d7a0249be(const __grid_constant__ CUtensorMap W, const __grid_constant__ CUtensorMap X, __nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ bias, int M, int N, int K)
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

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 78848);
    const int smem_b_addr = smem + 78848;
    if (tid == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&W))) : "memory"); }
    if (tid == 32) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&X))) : "memory"); }

    // Mbarrier init (3 pipeline groups, 0 ordered-sequence groups, 37 barriers)
    // Mbarriers at smem_raw[0..296)

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
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 296);
    if (warp == 2) {
        int _tmem_hold = smem + 296;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    if (warp == 2 || warp == 4 || warp == 5 || warp == 6 || warp == 7) {
        asm volatile("barrier.sync.aligned %0, %1;" :: "r"(1), "r"(160) : "memory");
        asm volatile("tcgen05.fence::after_thread_sync;");
    }
    const int taddr = (warp == 2 || warp == 4 || warp == 5 || warp == 6 || warp == 7) ? tmem_addr_storage[0] : 0;

    // Kernel post-init ops
    const int tmem_acc0 = taddr;
    const int tmem_acc1 = taddr + 64;
    const int tmem_acc2 = taddr + 128;
    const int tmem_acc3 = taddr + 192;

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
                    int g0 = rho * 4 * 4;
                    if (g0 < n_groups) {
                        mbarrier_wait(tma_empty_addr + (a_stage) * 8, _phase_tma_empty);
                        tma_2d_gmem2smem(smem_a_addr + a_stage * 4096, (&W), g0 * 64, m_idx, tma_full_addr + (a_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (a_stage) * 8, 4096);
                        a_stage += 1;
                        if (a_stage == 18) { a_stage = 0; _phase_tma_empty ^= 1; }
                    }
                    int g0_0 = rho * 4 * 4 + 1;
                    if (g0_0 < n_groups) {
                        mbarrier_wait(tma_empty_addr + (a_stage) * 8, _phase_tma_empty);
                        tma_2d_gmem2smem(smem_a_addr + a_stage * 4096, (&W), g0_0 * 64, m_idx, tma_full_addr + (a_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (a_stage) * 8, 4096);
                        a_stage += 1;
                        if (a_stage == 18) { a_stage = 0; _phase_tma_empty ^= 1; }
                    }
                    int g0_1 = rho * 4 * 4 + 2;
                    if (g0_1 < n_groups) {
                        mbarrier_wait(tma_empty_addr + (a_stage) * 8, _phase_tma_empty);
                        tma_2d_gmem2smem(smem_a_addr + a_stage * 4096, (&W), g0_1 * 64, m_idx, tma_full_addr + (a_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (a_stage) * 8, 4096);
                        a_stage += 1;
                        if (a_stage == 18) { a_stage = 0; _phase_tma_empty ^= 1; }
                    }
                    int g0_2 = rho * 4 * 4 + 3;
                    if (g0_2 < n_groups) {
                        mbarrier_wait(tma_empty_addr + (a_stage) * 8, _phase_tma_empty);
                        tma_2d_gmem2smem(smem_a_addr + a_stage * 4096, (&W), g0_2 * 64, m_idx, tma_full_addr + (a_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (a_stage) * 8, 4096);
                        a_stage += 1;
                        if (a_stage == 18) { a_stage = 0; _phase_tma_empty ^= 1; }
                    }
                    int g0_3 = (rho * 4 + 1) * 4;
                    if (g0_3 < n_groups) {
                        mbarrier_wait(tma_empty_addr + (a_stage) * 8, _phase_tma_empty);
                        tma_2d_gmem2smem(smem_a_addr + a_stage * 4096, (&W), g0_3 * 64, m_idx, tma_full_addr + (a_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (a_stage) * 8, 4096);
                        a_stage += 1;
                        if (a_stage == 18) { a_stage = 0; _phase_tma_empty ^= 1; }
                    }
                    int g0_4 = (rho * 4 + 1) * 4 + 1;
                    if (g0_4 < n_groups) {
                        mbarrier_wait(tma_empty_addr + (a_stage) * 8, _phase_tma_empty);
                        tma_2d_gmem2smem(smem_a_addr + a_stage * 4096, (&W), g0_4 * 64, m_idx, tma_full_addr + (a_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (a_stage) * 8, 4096);
                        a_stage += 1;
                        if (a_stage == 18) { a_stage = 0; _phase_tma_empty ^= 1; }
                    }
                    int g0_5 = (rho * 4 + 1) * 4 + 2;
                    if (g0_5 < n_groups) {
                        mbarrier_wait(tma_empty_addr + (a_stage) * 8, _phase_tma_empty);
                        tma_2d_gmem2smem(smem_a_addr + a_stage * 4096, (&W), g0_5 * 64, m_idx, tma_full_addr + (a_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (a_stage) * 8, 4096);
                        a_stage += 1;
                        if (a_stage == 18) { a_stage = 0; _phase_tma_empty ^= 1; }
                    }
                    int g0_6 = (rho * 4 + 1) * 4 + 3;
                    if (g0_6 < n_groups) {
                        mbarrier_wait(tma_empty_addr + (a_stage) * 8, _phase_tma_empty);
                        tma_2d_gmem2smem(smem_a_addr + a_stage * 4096, (&W), g0_6 * 64, m_idx, tma_full_addr + (a_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (a_stage) * 8, 4096);
                        a_stage += 1;
                        if (a_stage == 18) { a_stage = 0; _phase_tma_empty ^= 1; }
                    }
                    int g0_7 = (rho * 4 + 2) * 4;
                    if (g0_7 < n_groups) {
                        mbarrier_wait(tma_empty_addr + (a_stage) * 8, _phase_tma_empty);
                        tma_2d_gmem2smem(smem_a_addr + a_stage * 4096, (&W), g0_7 * 64, m_idx, tma_full_addr + (a_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (a_stage) * 8, 4096);
                        a_stage += 1;
                        if (a_stage == 18) { a_stage = 0; _phase_tma_empty ^= 1; }
                    }
                    int g0_8 = (rho * 4 + 2) * 4 + 1;
                    if (g0_8 < n_groups) {
                        mbarrier_wait(tma_empty_addr + (a_stage) * 8, _phase_tma_empty);
                        tma_2d_gmem2smem(smem_a_addr + a_stage * 4096, (&W), g0_8 * 64, m_idx, tma_full_addr + (a_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (a_stage) * 8, 4096);
                        a_stage += 1;
                        if (a_stage == 18) { a_stage = 0; _phase_tma_empty ^= 1; }
                    }
                    int g0_9 = (rho * 4 + 2) * 4 + 2;
                    if (g0_9 < n_groups) {
                        mbarrier_wait(tma_empty_addr + (a_stage) * 8, _phase_tma_empty);
                        tma_2d_gmem2smem(smem_a_addr + a_stage * 4096, (&W), g0_9 * 64, m_idx, tma_full_addr + (a_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (a_stage) * 8, 4096);
                        a_stage += 1;
                        if (a_stage == 18) { a_stage = 0; _phase_tma_empty ^= 1; }
                    }
                    int g0_10 = (rho * 4 + 2) * 4 + 3;
                    if (g0_10 < n_groups) {
                        mbarrier_wait(tma_empty_addr + (a_stage) * 8, _phase_tma_empty);
                        tma_2d_gmem2smem(smem_a_addr + a_stage * 4096, (&W), g0_10 * 64, m_idx, tma_full_addr + (a_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (a_stage) * 8, 4096);
                        a_stage += 1;
                        if (a_stage == 18) { a_stage = 0; _phase_tma_empty ^= 1; }
                    }
                    int g0_11 = (rho * 4 + 3) * 4;
                    if (g0_11 < n_groups) {
                        mbarrier_wait(tma_empty_addr + (a_stage) * 8, _phase_tma_empty);
                        tma_2d_gmem2smem(smem_a_addr + a_stage * 4096, (&W), g0_11 * 64, m_idx, tma_full_addr + (a_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (a_stage) * 8, 4096);
                        a_stage += 1;
                        if (a_stage == 18) { a_stage = 0; _phase_tma_empty ^= 1; }
                    }
                    int g0_12 = (rho * 4 + 3) * 4 + 1;
                    if (g0_12 < n_groups) {
                        mbarrier_wait(tma_empty_addr + (a_stage) * 8, _phase_tma_empty);
                        tma_2d_gmem2smem(smem_a_addr + a_stage * 4096, (&W), g0_12 * 64, m_idx, tma_full_addr + (a_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (a_stage) * 8, 4096);
                        a_stage += 1;
                        if (a_stage == 18) { a_stage = 0; _phase_tma_empty ^= 1; }
                    }
                    int g0_13 = (rho * 4 + 3) * 4 + 2;
                    if (g0_13 < n_groups) {
                        mbarrier_wait(tma_empty_addr + (a_stage) * 8, _phase_tma_empty);
                        tma_2d_gmem2smem(smem_a_addr + a_stage * 4096, (&W), g0_13 * 64, m_idx, tma_full_addr + (a_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (a_stage) * 8, 4096);
                        a_stage += 1;
                        if (a_stage == 18) { a_stage = 0; _phase_tma_empty ^= 1; }
                    }
                    int g0_14 = (rho * 4 + 3) * 4 + 3;
                    if (g0_14 < n_groups) {
                        mbarrier_wait(tma_empty_addr + (a_stage) * 8, _phase_tma_empty);
                        tma_2d_gmem2smem(smem_a_addr + a_stage * 4096, (&W), g0_14 * 64, m_idx, tma_full_addr + (a_stage) * 8);
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
                asm volatile("griddepcontrol.wait;" ::: "memory");
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
                #pragma unroll 1
                for (int rho_1 = 0; rho_1 < n_rounds_b; rho_1++) {
                    int g0b = rho_1 * 4 * 4;
                    if (g0b < n_groups_b) {
                        mbarrier_wait(tma_empty_addr + (b_stage) * 8, _phase_tma_empty_1);
                        tma_2d_gmem2smem(smem_b_addr + b_stage * 8192, (&X), g0b * 64, n_idx, tma_full_addr + (b_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (b_stage) * 8, 8192);
                        b_stage += 1;
                        if (b_stage == 18) { b_stage = 0; _phase_tma_empty_1 ^= 1; }
                    }
                    int g0b_0 = rho_1 * 4 * 4 + 1;
                    if (g0b_0 < n_groups_b) {
                        mbarrier_wait(tma_empty_addr + (b_stage) * 8, _phase_tma_empty_1);
                        tma_2d_gmem2smem(smem_b_addr + b_stage * 8192, (&X), g0b_0 * 64, n_idx, tma_full_addr + (b_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (b_stage) * 8, 8192);
                        b_stage += 1;
                        if (b_stage == 18) { b_stage = 0; _phase_tma_empty_1 ^= 1; }
                    }
                    int g0b_1 = rho_1 * 4 * 4 + 2;
                    if (g0b_1 < n_groups_b) {
                        mbarrier_wait(tma_empty_addr + (b_stage) * 8, _phase_tma_empty_1);
                        tma_2d_gmem2smem(smem_b_addr + b_stage * 8192, (&X), g0b_1 * 64, n_idx, tma_full_addr + (b_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (b_stage) * 8, 8192);
                        b_stage += 1;
                        if (b_stage == 18) { b_stage = 0; _phase_tma_empty_1 ^= 1; }
                    }
                    int g0b_2 = rho_1 * 4 * 4 + 3;
                    if (g0b_2 < n_groups_b) {
                        mbarrier_wait(tma_empty_addr + (b_stage) * 8, _phase_tma_empty_1);
                        tma_2d_gmem2smem(smem_b_addr + b_stage * 8192, (&X), g0b_2 * 64, n_idx, tma_full_addr + (b_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (b_stage) * 8, 8192);
                        b_stage += 1;
                        if (b_stage == 18) { b_stage = 0; _phase_tma_empty_1 ^= 1; }
                    }
                    int g0b_3 = (rho_1 * 4 + 1) * 4;
                    if (g0b_3 < n_groups_b) {
                        mbarrier_wait(tma_empty_addr + (b_stage) * 8, _phase_tma_empty_1);
                        tma_2d_gmem2smem(smem_b_addr + b_stage * 8192, (&X), g0b_3 * 64, n_idx, tma_full_addr + (b_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (b_stage) * 8, 8192);
                        b_stage += 1;
                        if (b_stage == 18) { b_stage = 0; _phase_tma_empty_1 ^= 1; }
                    }
                    int g0b_4 = (rho_1 * 4 + 1) * 4 + 1;
                    if (g0b_4 < n_groups_b) {
                        mbarrier_wait(tma_empty_addr + (b_stage) * 8, _phase_tma_empty_1);
                        tma_2d_gmem2smem(smem_b_addr + b_stage * 8192, (&X), g0b_4 * 64, n_idx, tma_full_addr + (b_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (b_stage) * 8, 8192);
                        b_stage += 1;
                        if (b_stage == 18) { b_stage = 0; _phase_tma_empty_1 ^= 1; }
                    }
                    int g0b_5 = (rho_1 * 4 + 1) * 4 + 2;
                    if (g0b_5 < n_groups_b) {
                        mbarrier_wait(tma_empty_addr + (b_stage) * 8, _phase_tma_empty_1);
                        tma_2d_gmem2smem(smem_b_addr + b_stage * 8192, (&X), g0b_5 * 64, n_idx, tma_full_addr + (b_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (b_stage) * 8, 8192);
                        b_stage += 1;
                        if (b_stage == 18) { b_stage = 0; _phase_tma_empty_1 ^= 1; }
                    }
                    int g0b_6 = (rho_1 * 4 + 1) * 4 + 3;
                    if (g0b_6 < n_groups_b) {
                        mbarrier_wait(tma_empty_addr + (b_stage) * 8, _phase_tma_empty_1);
                        tma_2d_gmem2smem(smem_b_addr + b_stage * 8192, (&X), g0b_6 * 64, n_idx, tma_full_addr + (b_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (b_stage) * 8, 8192);
                        b_stage += 1;
                        if (b_stage == 18) { b_stage = 0; _phase_tma_empty_1 ^= 1; }
                    }
                    int g0b_7 = (rho_1 * 4 + 2) * 4;
                    if (g0b_7 < n_groups_b) {
                        mbarrier_wait(tma_empty_addr + (b_stage) * 8, _phase_tma_empty_1);
                        tma_2d_gmem2smem(smem_b_addr + b_stage * 8192, (&X), g0b_7 * 64, n_idx, tma_full_addr + (b_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (b_stage) * 8, 8192);
                        b_stage += 1;
                        if (b_stage == 18) { b_stage = 0; _phase_tma_empty_1 ^= 1; }
                    }
                    int g0b_8 = (rho_1 * 4 + 2) * 4 + 1;
                    if (g0b_8 < n_groups_b) {
                        mbarrier_wait(tma_empty_addr + (b_stage) * 8, _phase_tma_empty_1);
                        tma_2d_gmem2smem(smem_b_addr + b_stage * 8192, (&X), g0b_8 * 64, n_idx, tma_full_addr + (b_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (b_stage) * 8, 8192);
                        b_stage += 1;
                        if (b_stage == 18) { b_stage = 0; _phase_tma_empty_1 ^= 1; }
                    }
                    int g0b_9 = (rho_1 * 4 + 2) * 4 + 2;
                    if (g0b_9 < n_groups_b) {
                        mbarrier_wait(tma_empty_addr + (b_stage) * 8, _phase_tma_empty_1);
                        tma_2d_gmem2smem(smem_b_addr + b_stage * 8192, (&X), g0b_9 * 64, n_idx, tma_full_addr + (b_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (b_stage) * 8, 8192);
                        b_stage += 1;
                        if (b_stage == 18) { b_stage = 0; _phase_tma_empty_1 ^= 1; }
                    }
                    int g0b_10 = (rho_1 * 4 + 2) * 4 + 3;
                    if (g0b_10 < n_groups_b) {
                        mbarrier_wait(tma_empty_addr + (b_stage) * 8, _phase_tma_empty_1);
                        tma_2d_gmem2smem(smem_b_addr + b_stage * 8192, (&X), g0b_10 * 64, n_idx, tma_full_addr + (b_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (b_stage) * 8, 8192);
                        b_stage += 1;
                        if (b_stage == 18) { b_stage = 0; _phase_tma_empty_1 ^= 1; }
                    }
                    int g0b_11 = (rho_1 * 4 + 3) * 4;
                    if (g0b_11 < n_groups_b) {
                        mbarrier_wait(tma_empty_addr + (b_stage) * 8, _phase_tma_empty_1);
                        tma_2d_gmem2smem(smem_b_addr + b_stage * 8192, (&X), g0b_11 * 64, n_idx, tma_full_addr + (b_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (b_stage) * 8, 8192);
                        b_stage += 1;
                        if (b_stage == 18) { b_stage = 0; _phase_tma_empty_1 ^= 1; }
                    }
                    int g0b_12 = (rho_1 * 4 + 3) * 4 + 1;
                    if (g0b_12 < n_groups_b) {
                        mbarrier_wait(tma_empty_addr + (b_stage) * 8, _phase_tma_empty_1);
                        tma_2d_gmem2smem(smem_b_addr + b_stage * 8192, (&X), g0b_12 * 64, n_idx, tma_full_addr + (b_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (b_stage) * 8, 8192);
                        b_stage += 1;
                        if (b_stage == 18) { b_stage = 0; _phase_tma_empty_1 ^= 1; }
                    }
                    int g0b_13 = (rho_1 * 4 + 3) * 4 + 2;
                    if (g0b_13 < n_groups_b) {
                        mbarrier_wait(tma_empty_addr + (b_stage) * 8, _phase_tma_empty_1);
                        tma_2d_gmem2smem(smem_b_addr + b_stage * 8192, (&X), g0b_13 * 64, n_idx, tma_full_addr + (b_stage) * 8);
                        mbarrier_arrive_expect_tx(tma_full_addr + (b_stage) * 8, 8192);
                        b_stage += 1;
                        if (b_stage == 18) { b_stage = 0; _phase_tma_empty_1 ^= 1; }
                    }
                    int g0b_14 = (rho_1 * 4 + 3) * 4 + 3;
                    if (g0b_14 < n_groups_b) {
                        mbarrier_wait(tma_empty_addr + (b_stage) * 8, _phase_tma_empty_1);
                        tma_2d_gmem2smem(smem_b_addr + b_stage * 8192, (&X), g0b_14 * 64, n_idx, tma_full_addr + (b_stage) * 8);
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
                    int g0m = rho_2 * 4 * 4;
                    if (g0m < n_groups_m) {
                        mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int init_flag = ((rho_2 == 0) ? 1 : 0);
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_0 = (((smem_a_addr) >> 4) & 0x7FFF) + (mma_stage) * 256;
                        int _smem_desc_lo_1 = (((smem_b_addr) >> 4) & 0x7FFF) + (mma_stage) * 512;
#else
                        int _smem_desc_lo_0 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage) * 256;
                        int _smem_desc_lo_1 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_stage) * 512;
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
                    "mov.b32 id, 68158608;\n\t"
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
                    :: "r"(_smem_desc_lo_0), "r"(_smem_desc_lo_1), "r"(tmem_acc0), "r"(((init_flag) ? 0 : 1)) : "memory");
                        tcgen05_commit(tma_empty_addr + (mma_stage) * 8);
                        mma_stage += 1;
                        if (mma_stage == 18) { mma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    int g0m_0 = rho_2 * 4 * 4 + 1;
                    if (g0m_0 < n_groups_m) {
                        mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_2 = (((smem_a_addr) >> 4) & 0x7FFF) + (mma_stage) * 256;
                        int _smem_desc_lo_3 = (((smem_b_addr) >> 4) & 0x7FFF) + (mma_stage) * 512;
#else
                        int _smem_desc_lo_2 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage) * 256;
                        int _smem_desc_lo_3 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_stage) * 512;
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
                    "mov.b32 id, 68158608;\n\t"
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
                        int _smem_desc_lo_4 = (((smem_a_addr) >> 4) & 0x7FFF) + (mma_stage) * 256;
                        int _smem_desc_lo_5 = (((smem_b_addr) >> 4) & 0x7FFF) + (mma_stage) * 512;
#else
                        int _smem_desc_lo_4 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage) * 256;
                        int _smem_desc_lo_5 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_stage) * 512;
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
                    "mov.b32 id, 68158608;\n\t"
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
                        int _smem_desc_lo_6 = (((smem_a_addr) >> 4) & 0x7FFF) + (mma_stage) * 256;
                        int _smem_desc_lo_7 = (((smem_b_addr) >> 4) & 0x7FFF) + (mma_stage) * 512;
#else
                        int _smem_desc_lo_6 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage) * 256;
                        int _smem_desc_lo_7 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_stage) * 512;
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
                    "mov.b32 id, 68158608;\n\t"
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
                    int g0m_3 = (rho_2 * 4 + 1) * 4;
                    if (g0m_3 < n_groups_m) {
                        mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int init_flag_1 = ((rho_2 == 0) ? 1 : 0);
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_8 = (((smem_a_addr) >> 4) & 0x7FFF) + (mma_stage) * 256;
                        int _smem_desc_lo_9 = (((smem_b_addr) >> 4) & 0x7FFF) + (mma_stage) * 512;
#else
                        int _smem_desc_lo_8 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage) * 256;
                        int _smem_desc_lo_9 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_stage) * 512;
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
                    "mov.b32 id, 68158608;\n\t"
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
                    :: "r"(_smem_desc_lo_8), "r"(_smem_desc_lo_9), "r"(tmem_acc1), "r"(((init_flag_1) ? 0 : 1)) : "memory");
                        tcgen05_commit(tma_empty_addr + (mma_stage) * 8);
                        mma_stage += 1;
                        if (mma_stage == 18) { mma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    int g0m_4 = (rho_2 * 4 + 1) * 4 + 1;
                    if (g0m_4 < n_groups_m) {
                        mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_10 = (((smem_a_addr) >> 4) & 0x7FFF) + (mma_stage) * 256;
                        int _smem_desc_lo_11 = (((smem_b_addr) >> 4) & 0x7FFF) + (mma_stage) * 512;
#else
                        int _smem_desc_lo_10 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage) * 256;
                        int _smem_desc_lo_11 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_stage) * 512;
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
                    "mov.b32 id, 68158608;\n\t"
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
                    int g0m_5 = (rho_2 * 4 + 1) * 4 + 2;
                    if (g0m_5 < n_groups_m) {
                        mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_12 = (((smem_a_addr) >> 4) & 0x7FFF) + (mma_stage) * 256;
                        int _smem_desc_lo_13 = (((smem_b_addr) >> 4) & 0x7FFF) + (mma_stage) * 512;
#else
                        int _smem_desc_lo_12 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage) * 256;
                        int _smem_desc_lo_13 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_stage) * 512;
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
                    "mov.b32 id, 68158608;\n\t"
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
                    int g0m_6 = (rho_2 * 4 + 1) * 4 + 3;
                    if (g0m_6 < n_groups_m) {
                        mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_14 = (((smem_a_addr) >> 4) & 0x7FFF) + (mma_stage) * 256;
                        int _smem_desc_lo_15 = (((smem_b_addr) >> 4) & 0x7FFF) + (mma_stage) * 512;
#else
                        int _smem_desc_lo_14 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage) * 256;
                        int _smem_desc_lo_15 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_stage) * 512;
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
                    "mov.b32 id, 68158608;\n\t"
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
                    int g0m_7 = (rho_2 * 4 + 2) * 4;
                    if (g0m_7 < n_groups_m) {
                        mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int init_flag_2 = ((rho_2 == 0) ? 1 : 0);
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_16 = (((smem_a_addr) >> 4) & 0x7FFF) + (mma_stage) * 256;
                        int _smem_desc_lo_17 = (((smem_b_addr) >> 4) & 0x7FFF) + (mma_stage) * 512;
#else
                        int _smem_desc_lo_16 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage) * 256;
                        int _smem_desc_lo_17 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_stage) * 512;
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
                    "mov.b32 id, 68158608;\n\t"
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
                    :: "r"(_smem_desc_lo_16), "r"(_smem_desc_lo_17), "r"(tmem_acc2), "r"(((init_flag_2) ? 0 : 1)) : "memory");
                        tcgen05_commit(tma_empty_addr + (mma_stage) * 8);
                        mma_stage += 1;
                        if (mma_stage == 18) { mma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    int g0m_8 = (rho_2 * 4 + 2) * 4 + 1;
                    if (g0m_8 < n_groups_m) {
                        mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_18 = (((smem_a_addr) >> 4) & 0x7FFF) + (mma_stage) * 256;
                        int _smem_desc_lo_19 = (((smem_b_addr) >> 4) & 0x7FFF) + (mma_stage) * 512;
#else
                        int _smem_desc_lo_18 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage) * 256;
                        int _smem_desc_lo_19 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_stage) * 512;
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
                    "mov.b32 id, 68158608;\n\t"
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
                    int g0m_9 = (rho_2 * 4 + 2) * 4 + 2;
                    if (g0m_9 < n_groups_m) {
                        mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_20 = (((smem_a_addr) >> 4) & 0x7FFF) + (mma_stage) * 256;
                        int _smem_desc_lo_21 = (((smem_b_addr) >> 4) & 0x7FFF) + (mma_stage) * 512;
#else
                        int _smem_desc_lo_20 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage) * 256;
                        int _smem_desc_lo_21 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_stage) * 512;
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
                    "mov.b32 id, 68158608;\n\t"
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
                    int g0m_10 = (rho_2 * 4 + 2) * 4 + 3;
                    if (g0m_10 < n_groups_m) {
                        mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_22 = (((smem_a_addr) >> 4) & 0x7FFF) + (mma_stage) * 256;
                        int _smem_desc_lo_23 = (((smem_b_addr) >> 4) & 0x7FFF) + (mma_stage) * 512;
#else
                        int _smem_desc_lo_22 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage) * 256;
                        int _smem_desc_lo_23 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_stage) * 512;
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
                    "mov.b32 id, 68158608;\n\t"
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
                    int g0m_11 = (rho_2 * 4 + 3) * 4;
                    if (g0m_11 < n_groups_m) {
                        mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int init_flag_3 = ((rho_2 == 0) ? 1 : 0);
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_24 = (((smem_a_addr) >> 4) & 0x7FFF) + (mma_stage) * 256;
                        int _smem_desc_lo_25 = (((smem_b_addr) >> 4) & 0x7FFF) + (mma_stage) * 512;
#else
                        int _smem_desc_lo_24 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage) * 256;
                        int _smem_desc_lo_25 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_stage) * 512;
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
                    "mov.b32 id, 68158608;\n\t"
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
                    :: "r"(_smem_desc_lo_24), "r"(_smem_desc_lo_25), "r"(tmem_acc3), "r"(((init_flag_3) ? 0 : 1)) : "memory");
                        tcgen05_commit(tma_empty_addr + (mma_stage) * 8);
                        mma_stage += 1;
                        if (mma_stage == 18) { mma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    int g0m_12 = (rho_2 * 4 + 3) * 4 + 1;
                    if (g0m_12 < n_groups_m) {
                        mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_26 = (((smem_a_addr) >> 4) & 0x7FFF) + (mma_stage) * 256;
                        int _smem_desc_lo_27 = (((smem_b_addr) >> 4) & 0x7FFF) + (mma_stage) * 512;
#else
                        int _smem_desc_lo_26 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage) * 256;
                        int _smem_desc_lo_27 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_stage) * 512;
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
                    "mov.b32 id, 68158608;\n\t"
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
                    int g0m_13 = (rho_2 * 4 + 3) * 4 + 2;
                    if (g0m_13 < n_groups_m) {
                        mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_28 = (((smem_a_addr) >> 4) & 0x7FFF) + (mma_stage) * 256;
                        int _smem_desc_lo_29 = (((smem_b_addr) >> 4) & 0x7FFF) + (mma_stage) * 512;
#else
                        int _smem_desc_lo_28 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage) * 256;
                        int _smem_desc_lo_29 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_stage) * 512;
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
                    "mov.b32 id, 68158608;\n\t"
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
                    int g0m_14 = (rho_2 * 4 + 3) * 4 + 3;
                    if (g0m_14 < n_groups_m) {
                        mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_30 = (((smem_a_addr) >> 4) & 0x7FFF) + (mma_stage) * 256;
                        int _smem_desc_lo_31 = (((smem_b_addr) >> 4) & 0x7FFF) + (mma_stage) * 512;
#else
                        int _smem_desc_lo_30 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage) * 256;
                        int _smem_desc_lo_31 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_stage) * 512;
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
                    "mov.b32 id, 68158608;\n\t"
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
    // ---- Role: epilogue ----
    if (warp >= 4 && warp <= 7) {
        { // epilogue_main
            const int epi_warp = warp % 4;
            const int epi_tid = epi_warp * 32 + lane;
            int m_idx_e = blockIdx.x * 32;
            int n_idx_e = blockIdx.y * 64;
            int n_chunks = ((K + 64 - 1) / 64 + 4 - 1) / 4;
            int row_in = lane % 16;
            int col_half = lane / 16;
            int my_m_local = epi_warp * 16 + row_in;
            int my_m = m_idx_e + my_m_local;
            unsigned int _phase_mma_epi_full_0 = 0;
            mbarrier_wait(mma_epi_full_addr, _phase_mma_epi_full_0);
            _phase_mma_epi_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            unsigned int tmem_addr = taddr + (unsigned int)(epi_warp * 32 << 16);
            float _tmem_load_0[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x32bx2.x8.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], 8;"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7]))
                : "r"(tmem_addr));
            float _tmem_load_1[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x32bx2.x8.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], 8;"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7]))
                : "r"(tmem_addr + 64));
            float _tmem_load_2[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x32bx2.x8.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], 8;"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7]))
                : "r"(tmem_addr + 128));
            float _tmem_load_3[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x32bx2.x8.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], 8;"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[7]))
                : "r"(tmem_addr + 192));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            int n_col0_0 = n_idx_e + col_half * 8;
            if (epi_tid < 64) {
                float p0_0_0 = ((n_chunks > 0) ? _tmem_load_0[0] : 0.0f);
                float p0_1_0 = ((n_chunks > 1) ? _tmem_load_1[0] : 0.0f);
                float p0_2_0 = ((n_chunks > 2) ? _tmem_load_2[0] : 0.0f);
                float p0_3_0 = ((n_chunks > 3) ? _tmem_load_3[0] : 0.0f);
                float t0_0 = p0_0_0 + p0_1_0 + p0_2_0 + p0_3_0;
                if (n_col0_0 < N) {
                    if (my_m < M && my_m_local < 32) {
                        *(reinterpret_cast<__nv_bfloat16*>(out + (my_m + n_col0_0 * M)) + (0)) = (__nv_bfloat16)t0_0;
                    }
                }
                float p0_0_1 = ((n_chunks > 0) ? _tmem_load_0[1] : 0.0f);
                float p0_1_1 = ((n_chunks > 1) ? _tmem_load_1[1] : 0.0f);
                float p0_2_1 = ((n_chunks > 2) ? _tmem_load_2[1] : 0.0f);
                float p0_3_1 = ((n_chunks > 3) ? _tmem_load_3[1] : 0.0f);
                float t0_1 = p0_0_1 + p0_1_1 + p0_2_1 + p0_3_1;
                if (n_col0_0 + 1 < N) {
                    if (my_m < M && my_m_local < 32) {
                        *(reinterpret_cast<__nv_bfloat16*>(out + (my_m + (n_col0_0 + 1) * M)) + (0)) = (__nv_bfloat16)t0_1;
                    }
                }
                float p0_0_2 = ((n_chunks > 0) ? _tmem_load_0[2] : 0.0f);
                float p0_1_2 = ((n_chunks > 1) ? _tmem_load_1[2] : 0.0f);
                float p0_2_2 = ((n_chunks > 2) ? _tmem_load_2[2] : 0.0f);
                float p0_3_2 = ((n_chunks > 3) ? _tmem_load_3[2] : 0.0f);
                float t0_2 = p0_0_2 + p0_1_2 + p0_2_2 + p0_3_2;
                if (n_col0_0 + 2 < N) {
                    if (my_m < M && my_m_local < 32) {
                        *(reinterpret_cast<__nv_bfloat16*>(out + (my_m + (n_col0_0 + 2) * M)) + (0)) = (__nv_bfloat16)t0_2;
                    }
                }
                float p0_0_3 = ((n_chunks > 0) ? _tmem_load_0[3] : 0.0f);
                float p0_1_3 = ((n_chunks > 1) ? _tmem_load_1[3] : 0.0f);
                float p0_2_3 = ((n_chunks > 2) ? _tmem_load_2[3] : 0.0f);
                float p0_3_3 = ((n_chunks > 3) ? _tmem_load_3[3] : 0.0f);
                float t0_3 = p0_0_3 + p0_1_3 + p0_2_3 + p0_3_3;
                if (n_col0_0 + 3 < N) {
                    if (my_m < M && my_m_local < 32) {
                        *(reinterpret_cast<__nv_bfloat16*>(out + (my_m + (n_col0_0 + 3) * M)) + (0)) = (__nv_bfloat16)t0_3;
                    }
                }
                float p0_0_4 = ((n_chunks > 0) ? _tmem_load_0[4] : 0.0f);
                float p0_1_4 = ((n_chunks > 1) ? _tmem_load_1[4] : 0.0f);
                float p0_2_4 = ((n_chunks > 2) ? _tmem_load_2[4] : 0.0f);
                float p0_3_4 = ((n_chunks > 3) ? _tmem_load_3[4] : 0.0f);
                float t0_4 = p0_0_4 + p0_1_4 + p0_2_4 + p0_3_4;
                if (n_col0_0 + 4 < N) {
                    if (my_m < M && my_m_local < 32) {
                        *(reinterpret_cast<__nv_bfloat16*>(out + (my_m + (n_col0_0 + 4) * M)) + (0)) = (__nv_bfloat16)t0_4;
                    }
                }
                float p0_0_5 = ((n_chunks > 0) ? _tmem_load_0[5] : 0.0f);
                float p0_1_5 = ((n_chunks > 1) ? _tmem_load_1[5] : 0.0f);
                float p0_2_5 = ((n_chunks > 2) ? _tmem_load_2[5] : 0.0f);
                float p0_3_5 = ((n_chunks > 3) ? _tmem_load_3[5] : 0.0f);
                float t0_5 = p0_0_5 + p0_1_5 + p0_2_5 + p0_3_5;
                if (n_col0_0 + 5 < N) {
                    if (my_m < M && my_m_local < 32) {
                        *(reinterpret_cast<__nv_bfloat16*>(out + (my_m + (n_col0_0 + 5) * M)) + (0)) = (__nv_bfloat16)t0_5;
                    }
                }
                float p0_0_6 = ((n_chunks > 0) ? _tmem_load_0[6] : 0.0f);
                float p0_1_6 = ((n_chunks > 1) ? _tmem_load_1[6] : 0.0f);
                float p0_2_6 = ((n_chunks > 2) ? _tmem_load_2[6] : 0.0f);
                float p0_3_6 = ((n_chunks > 3) ? _tmem_load_3[6] : 0.0f);
                float t0_6 = p0_0_6 + p0_1_6 + p0_2_6 + p0_3_6;
                if (n_col0_0 + 6 < N) {
                    if (my_m < M && my_m_local < 32) {
                        *(reinterpret_cast<__nv_bfloat16*>(out + (my_m + (n_col0_0 + 6) * M)) + (0)) = (__nv_bfloat16)t0_6;
                    }
                }
                float p0_0_7 = ((n_chunks > 0) ? _tmem_load_0[7] : 0.0f);
                float p0_1_7 = ((n_chunks > 1) ? _tmem_load_1[7] : 0.0f);
                float p0_2_7 = ((n_chunks > 2) ? _tmem_load_2[7] : 0.0f);
                float p0_3_7 = ((n_chunks > 3) ? _tmem_load_3[7] : 0.0f);
                float t0_7 = p0_0_7 + p0_1_7 + p0_2_7 + p0_3_7;
                if (n_col0_0 + 7 < N) {
                    if (my_m < M && my_m_local < 32) {
                        *(reinterpret_cast<__nv_bfloat16*>(out + (my_m + (n_col0_0 + 7) * M)) + (0)) = (__nv_bfloat16)t0_7;
                    }
                }
            }
            float _tmem_load_4[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x32bx2.x8.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], 8;"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[7]))
                : "r"(tmem_addr + 16));
            float _tmem_load_5[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x32bx2.x8.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], 8;"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[7]))
                : "r"(tmem_addr + 64 + 16));
            float _tmem_load_6[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x32bx2.x8.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], 8;"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[7]))
                : "r"(tmem_addr + 128 + 16));
            float _tmem_load_7[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x32bx2.x8.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], 8;"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[7]))
                : "r"(tmem_addr + 192 + 16));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            int n_col0_1 = n_idx_e + 16 + col_half * 8;
            if (epi_tid < 64) {
                float p1_0_0 = ((n_chunks > 0) ? _tmem_load_4[0] : 0.0f);
                float p1_1_0 = ((n_chunks > 1) ? _tmem_load_5[0] : 0.0f);
                float p1_2_0 = ((n_chunks > 2) ? _tmem_load_6[0] : 0.0f);
                float p1_3_0 = ((n_chunks > 3) ? _tmem_load_7[0] : 0.0f);
                float t1_0 = p1_0_0 + p1_1_0 + p1_2_0 + p1_3_0;
                if (n_col0_1 < N) {
                    if (my_m < M && my_m_local < 32) {
                        *(reinterpret_cast<__nv_bfloat16*>(out + (my_m + n_col0_1 * M)) + (0)) = (__nv_bfloat16)t1_0;
                    }
                }
                float p1_0_1 = ((n_chunks > 0) ? _tmem_load_4[1] : 0.0f);
                float p1_1_1 = ((n_chunks > 1) ? _tmem_load_5[1] : 0.0f);
                float p1_2_1 = ((n_chunks > 2) ? _tmem_load_6[1] : 0.0f);
                float p1_3_1 = ((n_chunks > 3) ? _tmem_load_7[1] : 0.0f);
                float t1_1 = p1_0_1 + p1_1_1 + p1_2_1 + p1_3_1;
                if (n_col0_1 + 1 < N) {
                    if (my_m < M && my_m_local < 32) {
                        *(reinterpret_cast<__nv_bfloat16*>(out + (my_m + (n_col0_1 + 1) * M)) + (0)) = (__nv_bfloat16)t1_1;
                    }
                }
                float p1_0_2 = ((n_chunks > 0) ? _tmem_load_4[2] : 0.0f);
                float p1_1_2 = ((n_chunks > 1) ? _tmem_load_5[2] : 0.0f);
                float p1_2_2 = ((n_chunks > 2) ? _tmem_load_6[2] : 0.0f);
                float p1_3_2 = ((n_chunks > 3) ? _tmem_load_7[2] : 0.0f);
                float t1_2 = p1_0_2 + p1_1_2 + p1_2_2 + p1_3_2;
                if (n_col0_1 + 2 < N) {
                    if (my_m < M && my_m_local < 32) {
                        *(reinterpret_cast<__nv_bfloat16*>(out + (my_m + (n_col0_1 + 2) * M)) + (0)) = (__nv_bfloat16)t1_2;
                    }
                }
                float p1_0_3 = ((n_chunks > 0) ? _tmem_load_4[3] : 0.0f);
                float p1_1_3 = ((n_chunks > 1) ? _tmem_load_5[3] : 0.0f);
                float p1_2_3 = ((n_chunks > 2) ? _tmem_load_6[3] : 0.0f);
                float p1_3_3 = ((n_chunks > 3) ? _tmem_load_7[3] : 0.0f);
                float t1_3 = p1_0_3 + p1_1_3 + p1_2_3 + p1_3_3;
                if (n_col0_1 + 3 < N) {
                    if (my_m < M && my_m_local < 32) {
                        *(reinterpret_cast<__nv_bfloat16*>(out + (my_m + (n_col0_1 + 3) * M)) + (0)) = (__nv_bfloat16)t1_3;
                    }
                }
                float p1_0_4 = ((n_chunks > 0) ? _tmem_load_4[4] : 0.0f);
                float p1_1_4 = ((n_chunks > 1) ? _tmem_load_5[4] : 0.0f);
                float p1_2_4 = ((n_chunks > 2) ? _tmem_load_6[4] : 0.0f);
                float p1_3_4 = ((n_chunks > 3) ? _tmem_load_7[4] : 0.0f);
                float t1_4 = p1_0_4 + p1_1_4 + p1_2_4 + p1_3_4;
                if (n_col0_1 + 4 < N) {
                    if (my_m < M && my_m_local < 32) {
                        *(reinterpret_cast<__nv_bfloat16*>(out + (my_m + (n_col0_1 + 4) * M)) + (0)) = (__nv_bfloat16)t1_4;
                    }
                }
                float p1_0_5 = ((n_chunks > 0) ? _tmem_load_4[5] : 0.0f);
                float p1_1_5 = ((n_chunks > 1) ? _tmem_load_5[5] : 0.0f);
                float p1_2_5 = ((n_chunks > 2) ? _tmem_load_6[5] : 0.0f);
                float p1_3_5 = ((n_chunks > 3) ? _tmem_load_7[5] : 0.0f);
                float t1_5 = p1_0_5 + p1_1_5 + p1_2_5 + p1_3_5;
                if (n_col0_1 + 5 < N) {
                    if (my_m < M && my_m_local < 32) {
                        *(reinterpret_cast<__nv_bfloat16*>(out + (my_m + (n_col0_1 + 5) * M)) + (0)) = (__nv_bfloat16)t1_5;
                    }
                }
                float p1_0_6 = ((n_chunks > 0) ? _tmem_load_4[6] : 0.0f);
                float p1_1_6 = ((n_chunks > 1) ? _tmem_load_5[6] : 0.0f);
                float p1_2_6 = ((n_chunks > 2) ? _tmem_load_6[6] : 0.0f);
                float p1_3_6 = ((n_chunks > 3) ? _tmem_load_7[6] : 0.0f);
                float t1_6 = p1_0_6 + p1_1_6 + p1_2_6 + p1_3_6;
                if (n_col0_1 + 6 < N) {
                    if (my_m < M && my_m_local < 32) {
                        *(reinterpret_cast<__nv_bfloat16*>(out + (my_m + (n_col0_1 + 6) * M)) + (0)) = (__nv_bfloat16)t1_6;
                    }
                }
                float p1_0_7 = ((n_chunks > 0) ? _tmem_load_4[7] : 0.0f);
                float p1_1_7 = ((n_chunks > 1) ? _tmem_load_5[7] : 0.0f);
                float p1_2_7 = ((n_chunks > 2) ? _tmem_load_6[7] : 0.0f);
                float p1_3_7 = ((n_chunks > 3) ? _tmem_load_7[7] : 0.0f);
                float t1_7 = p1_0_7 + p1_1_7 + p1_2_7 + p1_3_7;
                if (n_col0_1 + 7 < N) {
                    if (my_m < M && my_m_local < 32) {
                        *(reinterpret_cast<__nv_bfloat16*>(out + (my_m + (n_col0_1 + 7) * M)) + (0)) = (__nv_bfloat16)t1_7;
                    }
                }
            }
            float _tmem_load_8[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x32bx2.x8.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], 8;"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[7]))
                : "r"(tmem_addr + 32));
            float _tmem_load_9[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x32bx2.x8.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], 8;"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[7]))
                : "r"(tmem_addr + 64 + 32));
            float _tmem_load_10[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x32bx2.x8.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], 8;"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[7]))
                : "r"(tmem_addr + 128 + 32));
            float _tmem_load_11[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x32bx2.x8.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], 8;"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[7]))
                : "r"(tmem_addr + 192 + 32));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            int n_col0_2 = n_idx_e + 32 + col_half * 8;
            if (epi_tid < 64) {
                float p2_0_0 = ((n_chunks > 0) ? _tmem_load_8[0] : 0.0f);
                float p2_1_0 = ((n_chunks > 1) ? _tmem_load_9[0] : 0.0f);
                float p2_2_0 = ((n_chunks > 2) ? _tmem_load_10[0] : 0.0f);
                float p2_3_0 = ((n_chunks > 3) ? _tmem_load_11[0] : 0.0f);
                float t2_0 = p2_0_0 + p2_1_0 + p2_2_0 + p2_3_0;
                if (n_col0_2 < N) {
                    if (my_m < M && my_m_local < 32) {
                        *(reinterpret_cast<__nv_bfloat16*>(out + (my_m + n_col0_2 * M)) + (0)) = (__nv_bfloat16)t2_0;
                    }
                }
                float p2_0_1 = ((n_chunks > 0) ? _tmem_load_8[1] : 0.0f);
                float p2_1_1 = ((n_chunks > 1) ? _tmem_load_9[1] : 0.0f);
                float p2_2_1 = ((n_chunks > 2) ? _tmem_load_10[1] : 0.0f);
                float p2_3_1 = ((n_chunks > 3) ? _tmem_load_11[1] : 0.0f);
                float t2_1 = p2_0_1 + p2_1_1 + p2_2_1 + p2_3_1;
                if (n_col0_2 + 1 < N) {
                    if (my_m < M && my_m_local < 32) {
                        *(reinterpret_cast<__nv_bfloat16*>(out + (my_m + (n_col0_2 + 1) * M)) + (0)) = (__nv_bfloat16)t2_1;
                    }
                }
                float p2_0_2 = ((n_chunks > 0) ? _tmem_load_8[2] : 0.0f);
                float p2_1_2 = ((n_chunks > 1) ? _tmem_load_9[2] : 0.0f);
                float p2_2_2 = ((n_chunks > 2) ? _tmem_load_10[2] : 0.0f);
                float p2_3_2 = ((n_chunks > 3) ? _tmem_load_11[2] : 0.0f);
                float t2_2 = p2_0_2 + p2_1_2 + p2_2_2 + p2_3_2;
                if (n_col0_2 + 2 < N) {
                    if (my_m < M && my_m_local < 32) {
                        *(reinterpret_cast<__nv_bfloat16*>(out + (my_m + (n_col0_2 + 2) * M)) + (0)) = (__nv_bfloat16)t2_2;
                    }
                }
                float p2_0_3 = ((n_chunks > 0) ? _tmem_load_8[3] : 0.0f);
                float p2_1_3 = ((n_chunks > 1) ? _tmem_load_9[3] : 0.0f);
                float p2_2_3 = ((n_chunks > 2) ? _tmem_load_10[3] : 0.0f);
                float p2_3_3 = ((n_chunks > 3) ? _tmem_load_11[3] : 0.0f);
                float t2_3 = p2_0_3 + p2_1_3 + p2_2_3 + p2_3_3;
                if (n_col0_2 + 3 < N) {
                    if (my_m < M && my_m_local < 32) {
                        *(reinterpret_cast<__nv_bfloat16*>(out + (my_m + (n_col0_2 + 3) * M)) + (0)) = (__nv_bfloat16)t2_3;
                    }
                }
                float p2_0_4 = ((n_chunks > 0) ? _tmem_load_8[4] : 0.0f);
                float p2_1_4 = ((n_chunks > 1) ? _tmem_load_9[4] : 0.0f);
                float p2_2_4 = ((n_chunks > 2) ? _tmem_load_10[4] : 0.0f);
                float p2_3_4 = ((n_chunks > 3) ? _tmem_load_11[4] : 0.0f);
                float t2_4 = p2_0_4 + p2_1_4 + p2_2_4 + p2_3_4;
                if (n_col0_2 + 4 < N) {
                    if (my_m < M && my_m_local < 32) {
                        *(reinterpret_cast<__nv_bfloat16*>(out + (my_m + (n_col0_2 + 4) * M)) + (0)) = (__nv_bfloat16)t2_4;
                    }
                }
                float p2_0_5 = ((n_chunks > 0) ? _tmem_load_8[5] : 0.0f);
                float p2_1_5 = ((n_chunks > 1) ? _tmem_load_9[5] : 0.0f);
                float p2_2_5 = ((n_chunks > 2) ? _tmem_load_10[5] : 0.0f);
                float p2_3_5 = ((n_chunks > 3) ? _tmem_load_11[5] : 0.0f);
                float t2_5 = p2_0_5 + p2_1_5 + p2_2_5 + p2_3_5;
                if (n_col0_2 + 5 < N) {
                    if (my_m < M && my_m_local < 32) {
                        *(reinterpret_cast<__nv_bfloat16*>(out + (my_m + (n_col0_2 + 5) * M)) + (0)) = (__nv_bfloat16)t2_5;
                    }
                }
                float p2_0_6 = ((n_chunks > 0) ? _tmem_load_8[6] : 0.0f);
                float p2_1_6 = ((n_chunks > 1) ? _tmem_load_9[6] : 0.0f);
                float p2_2_6 = ((n_chunks > 2) ? _tmem_load_10[6] : 0.0f);
                float p2_3_6 = ((n_chunks > 3) ? _tmem_load_11[6] : 0.0f);
                float t2_6 = p2_0_6 + p2_1_6 + p2_2_6 + p2_3_6;
                if (n_col0_2 + 6 < N) {
                    if (my_m < M && my_m_local < 32) {
                        *(reinterpret_cast<__nv_bfloat16*>(out + (my_m + (n_col0_2 + 6) * M)) + (0)) = (__nv_bfloat16)t2_6;
                    }
                }
                float p2_0_7 = ((n_chunks > 0) ? _tmem_load_8[7] : 0.0f);
                float p2_1_7 = ((n_chunks > 1) ? _tmem_load_9[7] : 0.0f);
                float p2_2_7 = ((n_chunks > 2) ? _tmem_load_10[7] : 0.0f);
                float p2_3_7 = ((n_chunks > 3) ? _tmem_load_11[7] : 0.0f);
                float t2_7 = p2_0_7 + p2_1_7 + p2_2_7 + p2_3_7;
                if (n_col0_2 + 7 < N) {
                    if (my_m < M && my_m_local < 32) {
                        *(reinterpret_cast<__nv_bfloat16*>(out + (my_m + (n_col0_2 + 7) * M)) + (0)) = (__nv_bfloat16)t2_7;
                    }
                }
            }
            float _tmem_load_12[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x32bx2.x8.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], 8;"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[7]))
                : "r"(tmem_addr + 48));
            float _tmem_load_13[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x32bx2.x8.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], 8;"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[7]))
                : "r"(tmem_addr + 64 + 48));
            float _tmem_load_14[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x32bx2.x8.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], 8;"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[7]))
                : "r"(tmem_addr + 128 + 48));
            float _tmem_load_15[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x32bx2.x8.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8], 8;"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[7]))
                : "r"(tmem_addr + 192 + 48));
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            int n_col0_3 = n_idx_e + 48 + col_half * 8;
            if (epi_tid < 64) {
                float p3_0_0 = ((n_chunks > 0) ? _tmem_load_12[0] : 0.0f);
                float p3_1_0 = ((n_chunks > 1) ? _tmem_load_13[0] : 0.0f);
                float p3_2_0 = ((n_chunks > 2) ? _tmem_load_14[0] : 0.0f);
                float p3_3_0 = ((n_chunks > 3) ? _tmem_load_15[0] : 0.0f);
                float t3_0 = p3_0_0 + p3_1_0 + p3_2_0 + p3_3_0;
                if (n_col0_3 < N) {
                    if (my_m < M && my_m_local < 32) {
                        *(reinterpret_cast<__nv_bfloat16*>(out + (my_m + n_col0_3 * M)) + (0)) = (__nv_bfloat16)t3_0;
                    }
                }
                float p3_0_1 = ((n_chunks > 0) ? _tmem_load_12[1] : 0.0f);
                float p3_1_1 = ((n_chunks > 1) ? _tmem_load_13[1] : 0.0f);
                float p3_2_1 = ((n_chunks > 2) ? _tmem_load_14[1] : 0.0f);
                float p3_3_1 = ((n_chunks > 3) ? _tmem_load_15[1] : 0.0f);
                float t3_1 = p3_0_1 + p3_1_1 + p3_2_1 + p3_3_1;
                if (n_col0_3 + 1 < N) {
                    if (my_m < M && my_m_local < 32) {
                        *(reinterpret_cast<__nv_bfloat16*>(out + (my_m + (n_col0_3 + 1) * M)) + (0)) = (__nv_bfloat16)t3_1;
                    }
                }
                float p3_0_2 = ((n_chunks > 0) ? _tmem_load_12[2] : 0.0f);
                float p3_1_2 = ((n_chunks > 1) ? _tmem_load_13[2] : 0.0f);
                float p3_2_2 = ((n_chunks > 2) ? _tmem_load_14[2] : 0.0f);
                float p3_3_2 = ((n_chunks > 3) ? _tmem_load_15[2] : 0.0f);
                float t3_2 = p3_0_2 + p3_1_2 + p3_2_2 + p3_3_2;
                if (n_col0_3 + 2 < N) {
                    if (my_m < M && my_m_local < 32) {
                        *(reinterpret_cast<__nv_bfloat16*>(out + (my_m + (n_col0_3 + 2) * M)) + (0)) = (__nv_bfloat16)t3_2;
                    }
                }
                float p3_0_3 = ((n_chunks > 0) ? _tmem_load_12[3] : 0.0f);
                float p3_1_3 = ((n_chunks > 1) ? _tmem_load_13[3] : 0.0f);
                float p3_2_3 = ((n_chunks > 2) ? _tmem_load_14[3] : 0.0f);
                float p3_3_3 = ((n_chunks > 3) ? _tmem_load_15[3] : 0.0f);
                float t3_3 = p3_0_3 + p3_1_3 + p3_2_3 + p3_3_3;
                if (n_col0_3 + 3 < N) {
                    if (my_m < M && my_m_local < 32) {
                        *(reinterpret_cast<__nv_bfloat16*>(out + (my_m + (n_col0_3 + 3) * M)) + (0)) = (__nv_bfloat16)t3_3;
                    }
                }
                float p3_0_4 = ((n_chunks > 0) ? _tmem_load_12[4] : 0.0f);
                float p3_1_4 = ((n_chunks > 1) ? _tmem_load_13[4] : 0.0f);
                float p3_2_4 = ((n_chunks > 2) ? _tmem_load_14[4] : 0.0f);
                float p3_3_4 = ((n_chunks > 3) ? _tmem_load_15[4] : 0.0f);
                float t3_4 = p3_0_4 + p3_1_4 + p3_2_4 + p3_3_4;
                if (n_col0_3 + 4 < N) {
                    if (my_m < M && my_m_local < 32) {
                        *(reinterpret_cast<__nv_bfloat16*>(out + (my_m + (n_col0_3 + 4) * M)) + (0)) = (__nv_bfloat16)t3_4;
                    }
                }
                float p3_0_5 = ((n_chunks > 0) ? _tmem_load_12[5] : 0.0f);
                float p3_1_5 = ((n_chunks > 1) ? _tmem_load_13[5] : 0.0f);
                float p3_2_5 = ((n_chunks > 2) ? _tmem_load_14[5] : 0.0f);
                float p3_3_5 = ((n_chunks > 3) ? _tmem_load_15[5] : 0.0f);
                float t3_5 = p3_0_5 + p3_1_5 + p3_2_5 + p3_3_5;
                if (n_col0_3 + 5 < N) {
                    if (my_m < M && my_m_local < 32) {
                        *(reinterpret_cast<__nv_bfloat16*>(out + (my_m + (n_col0_3 + 5) * M)) + (0)) = (__nv_bfloat16)t3_5;
                    }
                }
                float p3_0_6 = ((n_chunks > 0) ? _tmem_load_12[6] : 0.0f);
                float p3_1_6 = ((n_chunks > 1) ? _tmem_load_13[6] : 0.0f);
                float p3_2_6 = ((n_chunks > 2) ? _tmem_load_14[6] : 0.0f);
                float p3_3_6 = ((n_chunks > 3) ? _tmem_load_15[6] : 0.0f);
                float t3_6 = p3_0_6 + p3_1_6 + p3_2_6 + p3_3_6;
                if (n_col0_3 + 6 < N) {
                    if (my_m < M && my_m_local < 32) {
                        *(reinterpret_cast<__nv_bfloat16*>(out + (my_m + (n_col0_3 + 6) * M)) + (0)) = (__nv_bfloat16)t3_6;
                    }
                }
                float p3_0_7 = ((n_chunks > 0) ? _tmem_load_12[7] : 0.0f);
                float p3_1_7 = ((n_chunks > 1) ? _tmem_load_13[7] : 0.0f);
                float p3_2_7 = ((n_chunks > 2) ? _tmem_load_14[7] : 0.0f);
                float p3_3_7 = ((n_chunks > 3) ? _tmem_load_15[7] : 0.0f);
                float t3_7 = p3_0_7 + p3_1_7 + p3_2_7 + p3_3_7;
                if (n_col0_3 + 7 < N) {
                    if (my_m < M && my_m_local < 32) {
                        *(reinterpret_cast<__nv_bfloat16*>(out + (my_m + (n_col0_3 + 7) * M)) + (0)) = (__nv_bfloat16)t3_7;
                    }
                }
            }
        }
    }

    // Cleanup
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 2) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
    }
}

} // extern "C"
