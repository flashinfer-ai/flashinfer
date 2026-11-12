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
#define TMEM_NCOLS 64
#define TMEM_ACC_OFFSET 0
#define NUM_TMA_PIPE_STAGES 4
#define NUM_ONE_PIPE_STAGES 1
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 16384
#define SMEM_SMEM_B_OFF 132096
#define SMEM_SMEM_B_STAGE_BYTES 8192
#define SMEM_SMEM_B_STRIDE 8192
#define SMEM_RED_OFF 197632
#define SMEM_RED_STAGE_BYTES 8192
#define SMEM_RED_STRIDE 8192
#define SMEM_TOTAL 205824
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


__device__ __forceinline__ void tmem_st_x32_f32(int tmem_addr, const float* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x32.b32"
        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8,"
        "  %9, %10, %11, %12, %13, %14, %15, %16,"
        "  %17, %18, %19, %20, %21, %22, %23, %24,"
        "  %25, %26, %27, %28, %29, %30, %31, %32};"
        :: "r"(tmem_addr),
           "f"(src[0]),  "f"(src[1]),  "f"(src[2]),  "f"(src[3]),
           "f"(src[4]),  "f"(src[5]),  "f"(src[6]),  "f"(src[7]),
           "f"(src[8]),  "f"(src[9]),  "f"(src[10]), "f"(src[11]),
           "f"(src[12]), "f"(src[13]), "f"(src[14]), "f"(src[15]),
           "f"(src[16]), "f"(src[17]), "f"(src[18]), "f"(src[19]),
           "f"(src[20]), "f"(src[21]), "f"(src[22]), "f"(src[23]),
           "f"(src[24]), "f"(src[25]), "f"(src[26]), "f"(src[27]),
           "f"(src[28]), "f"(src[29]), "f"(src[30]), "f"(src[31]));
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

__global__ __launch_bounds__(256) void
kernel_cake_tinygemm2_tcgen05_eb55bfbe52f5ff00b048(const __grid_constant__ CUtensorMap W, const __grid_constant__ CUtensorMap X, __nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ bias, int M, int N, int K)
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
    #define tma_empty_addr (mbar_base + 32)
    #define acc_zero_addr (mbar_base + 64)
    #define mma_epi_full_addr (mbar_base + 72)
    #define tmem_free_addr (mbar_base + 80)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 132096);
    const int smem_b_addr = smem + 132096;
    float* red = reinterpret_cast<float*>(smem_raw + 197632);
    const int red_addr = smem + 197632;
    if (tid == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&W))) : "memory"); }
    if (tid == 32) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&X))) : "memory"); }

    // Mbarrier init (5 pipeline groups, 0 ordered-sequence groups, 11 barriers)
    // Mbarriers at smem_raw[0..88)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full: 4 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            mbarrier_init(smem + 8, 2);
            mbarrier_init(smem + 16, 2);
            mbarrier_init(smem + 24, 2);
            // tma_empty: 4 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // --- pipeline 'one_pipe' ---
            // acc_zero: 1 barriers, init_count=4
            mbarrier_init(smem + 64, 4);
            // mma_epi_full: 1 barriers, init_count=1
            mbarrier_init(smem + 72, 1);
            // tmem_free: 1 barriers, init_count=4
            mbarrier_init(smem + 80, 4);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // TMEM alloc (64 columns, 64 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 88);
    if (warp == 2) {
        int _tmem_hold = smem + 88;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(64) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    if (warp >= 2 && warp <= 7) {
        asm volatile("barrier.sync.aligned %0, %1;" :: "r"(1), "r"(192) : "memory");
        asm volatile("tcgen05.fence::after_thread_sync;");
    }
    const int taddr = (warp >= 2 && warp <= 7) ? tmem_addr_storage[0] : 0;

    // Kernel post-init ops
    const int tmem_acc = taddr;

    // ---- Role: dma_a ----
    if (warp == 0) {
        { // dma_a_main
            unsigned int a_stage = 0;
            int m0 = blockIdx.x * 32;
            int n_groups = (K + 64 - 1) / 64;
            int n_rounds = (n_groups + 16 - 1) / 16;
            unsigned int _phase_tma_empty = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int rho = 0; rho < n_rounds; rho++) {
                    int s0_1_0 = rho * 16;
                    if (s0_1_0 < n_groups) {
                        mbarrier_wait(tma_empty_addr + (a_stage) * 8, _phase_tma_empty);
                        tma_2d_gmem2smem(smem_a_addr + a_stage * 2 * 16384, (&W), rho * 4 * 4 * 64, m0, tma_full_addr + (a_stage) * 8);
                        tma_2d_gmem2smem(smem_a_addr + a_stage * 2 * 16384 + 4096, (&W), (rho * 4 + 1) * 4 * 64, m0, tma_full_addr + (a_stage) * 8);
                        tma_2d_gmem2smem(smem_a_addr + a_stage * 2 * 16384 + 8192, (&W), (rho * 4 + 2) * 4 * 64, m0, tma_full_addr + (a_stage) * 8);
                        tma_2d_gmem2smem(smem_a_addr + a_stage * 2 * 16384 + 12288, (&W), (rho * 4 + 3) * 4 * 64, m0, tma_full_addr + (a_stage) * 8);
                        if (n_groups > s0_1_0 + 1) {
                            tma_2d_gmem2smem(smem_a_addr + (a_stage * 2 + 1) * 16384, (&W), (rho * 4 * 4 + 1) * 64, m0, tma_full_addr + (a_stage) * 8);
                            tma_2d_gmem2smem(smem_a_addr + (a_stage * 2 + 1) * 16384 + 4096, (&W), ((rho * 4 + 1) * 4 + 1) * 64, m0, tma_full_addr + (a_stage) * 8);
                            tma_2d_gmem2smem(smem_a_addr + (a_stage * 2 + 1) * 16384 + 8192, (&W), ((rho * 4 + 2) * 4 + 1) * 64, m0, tma_full_addr + (a_stage) * 8);
                            tma_2d_gmem2smem(smem_a_addr + (a_stage * 2 + 1) * 16384 + 12288, (&W), ((rho * 4 + 3) * 4 + 1) * 64, m0, tma_full_addr + (a_stage) * 8);
                        }
                        int _min_0 = ((n_groups - s0_1_0) < (2) ? (n_groups - s0_1_0) : (2));
                        mbarrier_arrive_expect_tx(tma_full_addr + (a_stage) * 8, _min_0 * 16384);
                        a_stage += 1;
                        if (a_stage == 4) { a_stage = 0; _phase_tma_empty ^= 1; }
                    }
                    int s0_1_1 = rho * 16 + 2;
                    if (s0_1_1 < n_groups) {
                        mbarrier_wait(tma_empty_addr + (a_stage) * 8, _phase_tma_empty);
                        tma_2d_gmem2smem(smem_a_addr + a_stage * 2 * 16384, (&W), (rho * 4 * 4 + 2) * 64, m0, tma_full_addr + (a_stage) * 8);
                        tma_2d_gmem2smem(smem_a_addr + a_stage * 2 * 16384 + 4096, (&W), ((rho * 4 + 1) * 4 + 2) * 64, m0, tma_full_addr + (a_stage) * 8);
                        tma_2d_gmem2smem(smem_a_addr + a_stage * 2 * 16384 + 8192, (&W), ((rho * 4 + 2) * 4 + 2) * 64, m0, tma_full_addr + (a_stage) * 8);
                        tma_2d_gmem2smem(smem_a_addr + a_stage * 2 * 16384 + 12288, (&W), ((rho * 4 + 3) * 4 + 2) * 64, m0, tma_full_addr + (a_stage) * 8);
                        if (n_groups > s0_1_1 + 1) {
                            tma_2d_gmem2smem(smem_a_addr + (a_stage * 2 + 1) * 16384, (&W), (rho * 4 * 4 + 2 + 1) * 64, m0, tma_full_addr + (a_stage) * 8);
                            tma_2d_gmem2smem(smem_a_addr + (a_stage * 2 + 1) * 16384 + 4096, (&W), ((rho * 4 + 1) * 4 + 2 + 1) * 64, m0, tma_full_addr + (a_stage) * 8);
                            tma_2d_gmem2smem(smem_a_addr + (a_stage * 2 + 1) * 16384 + 8192, (&W), ((rho * 4 + 2) * 4 + 2 + 1) * 64, m0, tma_full_addr + (a_stage) * 8);
                            tma_2d_gmem2smem(smem_a_addr + (a_stage * 2 + 1) * 16384 + 12288, (&W), ((rho * 4 + 3) * 4 + 2 + 1) * 64, m0, tma_full_addr + (a_stage) * 8);
                        }
                        int _min_1 = ((n_groups - s0_1_1) < (2) ? (n_groups - s0_1_1) : (2));
                        mbarrier_arrive_expect_tx(tma_full_addr + (a_stage) * 8, _min_1 * 16384);
                        a_stage += 1;
                        if (a_stage == 4) { a_stage = 0; _phase_tma_empty ^= 1; }
                    }
                }
            }
        }
    }
    // ---- Role: dma_b ----
    if (warp == 1) {
        { // dma_b_main
            unsigned int b_stage = 0;
            int n0 = blockIdx.y * 8;
            int n_groups_b = (K + 64 - 1) / 64;
            int n_rounds_b = (n_groups_b + 16 - 1) / 16;
            unsigned int _phase_tma_empty_1 = 1;
            if (elect_sync()) {
                #pragma unroll 1
                for (int rho_1 = 0; rho_1 < n_rounds_b; rho_1++) {
                    int s0b_1_0 = rho_1 * 16;
                    if (s0b_1_0 < n_groups_b) {
                        mbarrier_wait(tma_empty_addr + (b_stage) * 8, _phase_tma_empty_1);
                        tma_2d_gmem2smem(smem_b_addr + b_stage * 2 * 8192, (&X), rho_1 * 4 * 4 * 64, n0, tma_full_addr + (b_stage) * 8);
                        tma_2d_gmem2smem(smem_b_addr + b_stage * 2 * 8192 + 2048, (&X), (rho_1 * 4 + 1) * 4 * 64, n0, tma_full_addr + (b_stage) * 8);
                        tma_2d_gmem2smem(smem_b_addr + b_stage * 2 * 8192 + 4096, (&X), (rho_1 * 4 + 2) * 4 * 64, n0, tma_full_addr + (b_stage) * 8);
                        tma_2d_gmem2smem(smem_b_addr + b_stage * 2 * 8192 + 6144, (&X), (rho_1 * 4 + 3) * 4 * 64, n0, tma_full_addr + (b_stage) * 8);
                        if (n_groups_b > s0b_1_0 + 1) {
                            tma_2d_gmem2smem(smem_b_addr + (b_stage * 2 + 1) * 8192, (&X), (rho_1 * 4 * 4 + 1) * 64, n0, tma_full_addr + (b_stage) * 8);
                            tma_2d_gmem2smem(smem_b_addr + (b_stage * 2 + 1) * 8192 + 2048, (&X), ((rho_1 * 4 + 1) * 4 + 1) * 64, n0, tma_full_addr + (b_stage) * 8);
                            tma_2d_gmem2smem(smem_b_addr + (b_stage * 2 + 1) * 8192 + 4096, (&X), ((rho_1 * 4 + 2) * 4 + 1) * 64, n0, tma_full_addr + (b_stage) * 8);
                            tma_2d_gmem2smem(smem_b_addr + (b_stage * 2 + 1) * 8192 + 6144, (&X), ((rho_1 * 4 + 3) * 4 + 1) * 64, n0, tma_full_addr + (b_stage) * 8);
                        }
                        int _min_0_1 = ((n_groups_b - s0b_1_0) < (2) ? (n_groups_b - s0b_1_0) : (2));
                        mbarrier_arrive_expect_tx(tma_full_addr + (b_stage) * 8, _min_0_1 * 4096);
                        b_stage += 1;
                        if (b_stage == 4) { b_stage = 0; _phase_tma_empty_1 ^= 1; }
                    }
                    int s0b_1_1 = rho_1 * 16 + 2;
                    if (s0b_1_1 < n_groups_b) {
                        mbarrier_wait(tma_empty_addr + (b_stage) * 8, _phase_tma_empty_1);
                        tma_2d_gmem2smem(smem_b_addr + b_stage * 2 * 8192, (&X), (rho_1 * 4 * 4 + 2) * 64, n0, tma_full_addr + (b_stage) * 8);
                        tma_2d_gmem2smem(smem_b_addr + b_stage * 2 * 8192 + 2048, (&X), ((rho_1 * 4 + 1) * 4 + 2) * 64, n0, tma_full_addr + (b_stage) * 8);
                        tma_2d_gmem2smem(smem_b_addr + b_stage * 2 * 8192 + 4096, (&X), ((rho_1 * 4 + 2) * 4 + 2) * 64, n0, tma_full_addr + (b_stage) * 8);
                        tma_2d_gmem2smem(smem_b_addr + b_stage * 2 * 8192 + 6144, (&X), ((rho_1 * 4 + 3) * 4 + 2) * 64, n0, tma_full_addr + (b_stage) * 8);
                        if (n_groups_b > s0b_1_1 + 1) {
                            tma_2d_gmem2smem(smem_b_addr + (b_stage * 2 + 1) * 8192, (&X), (rho_1 * 4 * 4 + 2 + 1) * 64, n0, tma_full_addr + (b_stage) * 8);
                            tma_2d_gmem2smem(smem_b_addr + (b_stage * 2 + 1) * 8192 + 2048, (&X), ((rho_1 * 4 + 1) * 4 + 2 + 1) * 64, n0, tma_full_addr + (b_stage) * 8);
                            tma_2d_gmem2smem(smem_b_addr + (b_stage * 2 + 1) * 8192 + 4096, (&X), ((rho_1 * 4 + 2) * 4 + 2 + 1) * 64, n0, tma_full_addr + (b_stage) * 8);
                            tma_2d_gmem2smem(smem_b_addr + (b_stage * 2 + 1) * 8192 + 6144, (&X), ((rho_1 * 4 + 3) * 4 + 2 + 1) * 64, n0, tma_full_addr + (b_stage) * 8);
                        }
                        int _min_1_1 = ((n_groups_b - s0b_1_1) < (2) ? (n_groups_b - s0b_1_1) : (2));
                        mbarrier_arrive_expect_tx(tma_full_addr + (b_stage) * 8, _min_1_1 * 4096);
                        b_stage += 1;
                        if (b_stage == 4) { b_stage = 0; _phase_tma_empty_1 ^= 1; }
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
            int n_rounds_m = (n_groups_m + 16 - 1) / 16;
            unsigned int _phase_acc_zero_0 = 0;
            mbarrier_wait(acc_zero_addr, _phase_acc_zero_0);
            _phase_acc_zero_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            unsigned int _phase_tma_full = 0;
            if (elect_sync()) {
                #pragma unroll 1
                for (int rho_2 = 0; rho_2 < n_rounds_m; rho_2++) {
                    int s0m_1_0 = rho_2 * 16;
                    if (s0m_1_0 < n_groups_m) {
                        mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_0 = (((smem_a_addr) >> 4) & 0x7FFF) + (mma_stage * 2) * 1024;
                        int _smem_desc_lo_1 = (((smem_b_addr) >> 4) & 0x7FFF) + (mma_stage * 2) * 512;
#else
                        int _smem_desc_lo_0 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage * 2) * 1024;
                        int _smem_desc_lo_1 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_stage * 2) * 512;
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
                    "mov.b32 id, 135267472;\n\t"
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
                    :: "r"(_smem_desc_lo_0), "r"(_smem_desc_lo_1), "r"(tmem_acc), "r"(1) : "memory");
                        if (n_groups_m > s0m_1_0 + 1) {
#if __CUDA_ARCH__ == 1070
                            int _smem_desc_lo_2 = (((smem_a_addr) >> 4) & 0x7FFF) + (mma_stage * 2 + 1) * 1024;
                            int _smem_desc_lo_3 = (((smem_b_addr) >> 4) & 0x7FFF) + (mma_stage * 2 + 1) * 512;
#else
                            int _smem_desc_lo_2 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage * 2 + 1) * 1024;
                            int _smem_desc_lo_3 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_stage * 2 + 1) * 512;
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
                    "mov.b32 id, 135267472;\n\t"
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
                    :: "r"(_smem_desc_lo_2), "r"(_smem_desc_lo_3), "r"(tmem_acc), "r"(1) : "memory");
                        }
                        tcgen05_commit(tma_empty_addr + (mma_stage) * 8);
                        mma_stage += 1;
                        if (mma_stage == 4) { mma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    int s0m_1_1 = rho_2 * 16 + 2;
                    if (s0m_1_1 < n_groups_m) {
                        mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
#if __CUDA_ARCH__ == 1070
                        int _smem_desc_lo_4 = (((smem_a_addr) >> 4) & 0x7FFF) + (mma_stage * 2) * 1024;
                        int _smem_desc_lo_5 = (((smem_b_addr) >> 4) & 0x7FFF) + (mma_stage * 2) * 512;
#else
                        int _smem_desc_lo_4 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage * 2) * 1024;
                        int _smem_desc_lo_5 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_stage * 2) * 512;
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
                    "mov.b32 id, 135267472;\n\t"
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
                    :: "r"(_smem_desc_lo_4), "r"(_smem_desc_lo_5), "r"(tmem_acc), "r"(1) : "memory");
                        if (n_groups_m > s0m_1_1 + 1) {
#if __CUDA_ARCH__ == 1070
                            int _smem_desc_lo_6 = (((smem_a_addr) >> 4) & 0x7FFF) + (mma_stage * 2 + 1) * 1024;
                            int _smem_desc_lo_7 = (((smem_b_addr) >> 4) & 0x7FFF) + (mma_stage * 2 + 1) * 512;
#else
                            int _smem_desc_lo_6 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage * 2 + 1) * 1024;
                            int _smem_desc_lo_7 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_stage * 2 + 1) * 512;
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
                    "mov.b32 id, 135267472;\n\t"
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
                    :: "r"(_smem_desc_lo_6), "r"(_smem_desc_lo_7), "r"(tmem_acc), "r"(1) : "memory");
                        }
                        tcgen05_commit(tma_empty_addr + (mma_stage) * 8);
                        mma_stage += 1;
                        if (mma_stage == 4) { mma_stage = 0; _phase_tma_full ^= 1; }
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
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(64));
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 4 && warp <= 7) {
        { // epilogue_main
            const int epi_warp = warp % 4;
            const int epi_tid = epi_warp * 32 + lane;
            int m0_e = blockIdx.x * 32;
            int n0_e = blockIdx.y * 8;
            unsigned int tmem_warp = taddr + (unsigned int)(epi_warp * 32 << 16);
            float zeros[32] = {0};
            tmem_st_x32_f32(tmem_warp, zeros);
            tmem_st_x32_f32(tmem_warp + 32, zeros);
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            if (elect_sync()) {
                mbarrier_arrive(acc_zero_addr);
            }
            int row_in = lane;
            int col_half = 0;
            unsigned int _phase_mma_epi_full_0 = 0;
            mbarrier_wait(mma_epi_full_addr, _phase_mma_epi_full_0);
            _phase_mma_epi_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float _tmem_load_0[16];
            tmem_ld_x16(&_tmem_load_0[0], tmem_warp + (unsigned int)(epi_warp * 16));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            asm volatile("tcgen05.fence::before_thread_sync;");
            __syncwarp();
            if (elect_sync()) {
                mbarrier_arrive(tmem_free_addr);
            }
            int red_base = epi_warp * 512 + col_half * 16 * 32 + row_in;
            if (row_in < 32) {
                red[red_base] = _tmem_load_0[0];
                red[red_base + 32] = _tmem_load_0[1];
                red[red_base + 64] = _tmem_load_0[2];
                red[red_base + 96] = _tmem_load_0[3];
                red[red_base + 128] = _tmem_load_0[4];
                red[red_base + 160] = _tmem_load_0[5];
                red[red_base + 192] = _tmem_load_0[6];
                red[red_base + 224] = _tmem_load_0[7];
                red[red_base + 256] = _tmem_load_0[8];
                red[red_base + 288] = _tmem_load_0[9];
                red[red_base + 320] = _tmem_load_0[10];
                red[red_base + 352] = _tmem_load_0[11];
                red[red_base + 384] = _tmem_load_0[12];
                red[red_base + 416] = _tmem_load_0[13];
                red[red_base + 448] = _tmem_load_0[14];
                red[red_base + 480] = _tmem_load_0[15];
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("barrier.sync 2, 128;" ::: "memory");
            int elem0_1_0 = epi_tid;
            if (elem0_1_0 < 512) {
                int r_e0_1_0 = elem0_1_0 % 32;
                int col_e0_1_0 = elem0_1_0 / 32;
                float s01_1 = red[elem0_1_0] + red[elem0_1_0 + 512];
                float s012_1 = s01_1 + red[elem0_1_0 + 1024];
                float tot0_1_0 = s012_1 + red[elem0_1_0 + 1536];
                int my_m0_1_0 = m0_e + r_e0_1_0;
                if (my_m0_1_0 < M && n0_e + col_e0_1_0 < N) {
                    *(reinterpret_cast<__nv_bfloat16*>(out + (my_m0_1_0 + (n0_e + col_e0_1_0) * M)) + (0)) = (__nv_bfloat16)tot0_1_0;
                }
            }
            int elem0_1_1 = epi_tid + 128;
            if (elem0_1_1 < 512) {
                int r_e0_1_1 = elem0_1_1 % 32;
                int col_e0_1_1 = elem0_1_1 / 32;
                float s01_2 = red[elem0_1_1] + red[elem0_1_1 + 512];
                float s012_2 = s01_2 + red[elem0_1_1 + 1024];
                float tot0_1_1 = s012_2 + red[elem0_1_1 + 1536];
                int my_m0_1_1 = m0_e + r_e0_1_1;
                if (my_m0_1_1 < M && n0_e + col_e0_1_1 < N) {
                    *(reinterpret_cast<__nv_bfloat16*>(out + (my_m0_1_1 + (n0_e + col_e0_1_1) * M)) + (0)) = (__nv_bfloat16)tot0_1_1;
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
