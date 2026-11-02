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

#define CAKE_INF CUDART_INF_F
#define NUM_PIPE_STAGES 5
#define SMEM_SW_OFF 1024
#define SMEM_SW_STAGE_BYTES 32768
#define SMEM_SW_STRIDE 40960
#define SMEM_SX_OFF 33792
#define SMEM_SX_STAGE_BYTES 8192
#define SMEM_SX_STRIDE 40960
#define SMEM_TOTAL 205824
#define THREADS 160
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


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(160, LAUNCH_MIN_BLOCKS) void
kernel_cake_tinygemm2_wgmma_164886d9e1894a3eb342(const __grid_constant__ CUtensorMap W, const __grid_constant__ CUtensorMap X, __nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ bias, int M, int N, int K)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define full_addr (mbar_base + 0)
    #define empty_addr (mbar_base + 40)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* sw = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int sw_addr = smem + 1024;
    __nv_bfloat16* sx = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int sx_addr = smem + 33792;
    if (tid == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&W))) : "memory"); }
    if (tid == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&X))) : "memory"); }

    // Mbarrier init (2 pipeline groups, 0 ordered-sequence groups, 10 barriers)
    // Mbarriers at smem_raw[0..80)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'pipe' ---
            // full: 5 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            // empty: 5 barriers, init_count=4
            mbarrier_init(smem + 40, 4);
            mbarrier_init(smem + 48, 4);
            mbarrier_init(smem + 56, 4);
            mbarrier_init(smem + 64, 4);
            mbarrier_init(smem + 72, 4);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // Kernel post-init ops
    int n0 = blockIdx.x * 16;
    int m0 = blockIdx.y * 64;
    int rounds = (K + 1023) / 1024;
    int nstages = ((rounds * 4) < ((K + 63) / 64 - 12 * (rounds - 1)) ? (rounds * 4) : ((K + 63) / 64 - 12 * (rounds - 1)));

    // ---- Role: mma ----
    if (warp <= 3) {
        { // mma_main
            float acc_0[8] = {0};
            float acc_1[8] = {0};
            float acc_2[8] = {0};
            float acc_3[8] = {0};
            int g = lane / 4;
            int t = lane % 4;
            int _base = make_warp_uniform((unsigned int)smem);
            unsigned int base = _base;
            int slot = 0;
            asm volatile("" : "+f"(acc_0[0]) :: "memory");
            asm volatile("" : "+f"(acc_0[1]) :: "memory");
            asm volatile("" : "+f"(acc_0[2]) :: "memory");
            asm volatile("" : "+f"(acc_0[3]) :: "memory");
            asm volatile("" : "+f"(acc_0[4]) :: "memory");
            asm volatile("" : "+f"(acc_0[5]) :: "memory");
            asm volatile("" : "+f"(acc_0[6]) :: "memory");
            asm volatile("" : "+f"(acc_0[7]) :: "memory");
            asm volatile("" : "+f"(acc_1[0]) :: "memory");
            asm volatile("" : "+f"(acc_1[1]) :: "memory");
            asm volatile("" : "+f"(acc_1[2]) :: "memory");
            asm volatile("" : "+f"(acc_1[3]) :: "memory");
            asm volatile("" : "+f"(acc_1[4]) :: "memory");
            asm volatile("" : "+f"(acc_1[5]) :: "memory");
            asm volatile("" : "+f"(acc_1[6]) :: "memory");
            asm volatile("" : "+f"(acc_1[7]) :: "memory");
            asm volatile("" : "+f"(acc_2[0]) :: "memory");
            asm volatile("" : "+f"(acc_2[1]) :: "memory");
            asm volatile("" : "+f"(acc_2[2]) :: "memory");
            asm volatile("" : "+f"(acc_2[3]) :: "memory");
            asm volatile("" : "+f"(acc_2[4]) :: "memory");
            asm volatile("" : "+f"(acc_2[5]) :: "memory");
            asm volatile("" : "+f"(acc_2[6]) :: "memory");
            asm volatile("" : "+f"(acc_2[7]) :: "memory");
            asm volatile("" : "+f"(acc_3[0]) :: "memory");
            asm volatile("" : "+f"(acc_3[1]) :: "memory");
            asm volatile("" : "+f"(acc_3[2]) :: "memory");
            asm volatile("" : "+f"(acc_3[3]) :: "memory");
            asm volatile("" : "+f"(acc_3[4]) :: "memory");
            asm volatile("" : "+f"(acc_3[5]) :: "memory");
            asm volatile("" : "+f"(acc_3[6]) :: "memory");
            asm volatile("" : "+f"(acc_3[7]) :: "memory");
            unsigned int _phase_full = 0;
            #pragma unroll 1
            for (int st = 0; st < nstages - 1; st++) {
                mbarrier_wait(full_addr + (slot) * 8, _phase_full);
                asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
                unsigned int sb = base + (unsigned int)(slot * 40960);
                uint64_t d_b_0 = (((uint64_t)(((sb + (33792u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                uint64_t d_a_0 = (((uint64_t)(((sb + (1024u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                uint64_t d_b_1 = (((uint64_t)(((sb + (33792u + 2048u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                uint64_t d_a_1 = (((uint64_t)(((sb + (1024u + 8192u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                uint64_t d_b_2 = (((uint64_t)(((sb + (33792u + 4096u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                uint64_t d_a_2 = (((uint64_t)(((sb + (1024u + 16384u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                uint64_t d_b_3 = (((uint64_t)(((sb + (33792u + 6144u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                uint64_t d_a_3 = (((uint64_t)(((sb + (1024u + 24576u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3]), "+f"(acc_0[4]), "+f"(acc_0[5]), "+f"(acc_0[6]), "+f"(acc_0[7])
                    : "l"(d_a_0), "l"(d_b_0)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3]), "+f"(acc_1[4]), "+f"(acc_1[5]), "+f"(acc_1[6]), "+f"(acc_1[7])
                    : "l"(d_a_1), "l"(d_b_1)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_2[0]), "+f"(acc_2[1]), "+f"(acc_2[2]), "+f"(acc_2[3]), "+f"(acc_2[4]), "+f"(acc_2[5]), "+f"(acc_2[6]), "+f"(acc_2[7])
                    : "l"(d_a_2), "l"(d_b_2)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_3[0]), "+f"(acc_3[1]), "+f"(acc_3[2]), "+f"(acc_3[3]), "+f"(acc_3[4]), "+f"(acc_3[5]), "+f"(acc_3[6]), "+f"(acc_3[7])
                    : "l"(d_a_3), "l"(d_b_3)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3]), "+f"(acc_0[4]), "+f"(acc_0[5]), "+f"(acc_0[6]), "+f"(acc_0[7])
                    : "l"(d_a_0 + 2), "l"(d_b_0 + 2)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3]), "+f"(acc_1[4]), "+f"(acc_1[5]), "+f"(acc_1[6]), "+f"(acc_1[7])
                    : "l"(d_a_1 + 2), "l"(d_b_1 + 2)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_2[0]), "+f"(acc_2[1]), "+f"(acc_2[2]), "+f"(acc_2[3]), "+f"(acc_2[4]), "+f"(acc_2[5]), "+f"(acc_2[6]), "+f"(acc_2[7])
                    : "l"(d_a_2 + 2), "l"(d_b_2 + 2)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_3[0]), "+f"(acc_3[1]), "+f"(acc_3[2]), "+f"(acc_3[3]), "+f"(acc_3[4]), "+f"(acc_3[5]), "+f"(acc_3[6]), "+f"(acc_3[7])
                    : "l"(d_a_3 + 2), "l"(d_b_3 + 2)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3]), "+f"(acc_0[4]), "+f"(acc_0[5]), "+f"(acc_0[6]), "+f"(acc_0[7])
                    : "l"(d_a_0 + 4), "l"(d_b_0 + 4)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3]), "+f"(acc_1[4]), "+f"(acc_1[5]), "+f"(acc_1[6]), "+f"(acc_1[7])
                    : "l"(d_a_1 + 4), "l"(d_b_1 + 4)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_2[0]), "+f"(acc_2[1]), "+f"(acc_2[2]), "+f"(acc_2[3]), "+f"(acc_2[4]), "+f"(acc_2[5]), "+f"(acc_2[6]), "+f"(acc_2[7])
                    : "l"(d_a_2 + 4), "l"(d_b_2 + 4)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_3[0]), "+f"(acc_3[1]), "+f"(acc_3[2]), "+f"(acc_3[3]), "+f"(acc_3[4]), "+f"(acc_3[5]), "+f"(acc_3[6]), "+f"(acc_3[7])
                    : "l"(d_a_3 + 4), "l"(d_b_3 + 4)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3]), "+f"(acc_0[4]), "+f"(acc_0[5]), "+f"(acc_0[6]), "+f"(acc_0[7])
                    : "l"(d_a_0 + 6), "l"(d_b_0 + 6)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3]), "+f"(acc_1[4]), "+f"(acc_1[5]), "+f"(acc_1[6]), "+f"(acc_1[7])
                    : "l"(d_a_1 + 6), "l"(d_b_1 + 6)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_2[0]), "+f"(acc_2[1]), "+f"(acc_2[2]), "+f"(acc_2[3]), "+f"(acc_2[4]), "+f"(acc_2[5]), "+f"(acc_2[6]), "+f"(acc_2[7])
                    : "l"(d_a_2 + 6), "l"(d_b_2 + 6)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_3[0]), "+f"(acc_3[1]), "+f"(acc_3[2]), "+f"(acc_3[3]), "+f"(acc_3[4]), "+f"(acc_3[5]), "+f"(acc_3[6]), "+f"(acc_3[7])
                    : "l"(d_a_3 + 6), "l"(d_b_3 + 6)
                    : "memory");
                asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
                asm volatile("" : "+f"(acc_0[0]) :: "memory");
                asm volatile("" : "+f"(acc_0[1]) :: "memory");
                asm volatile("" : "+f"(acc_0[2]) :: "memory");
                asm volatile("" : "+f"(acc_0[3]) :: "memory");
                asm volatile("" : "+f"(acc_0[4]) :: "memory");
                asm volatile("" : "+f"(acc_0[5]) :: "memory");
                asm volatile("" : "+f"(acc_0[6]) :: "memory");
                asm volatile("" : "+f"(acc_0[7]) :: "memory");
                asm volatile("" : "+f"(acc_1[0]) :: "memory");
                asm volatile("" : "+f"(acc_1[1]) :: "memory");
                asm volatile("" : "+f"(acc_1[2]) :: "memory");
                asm volatile("" : "+f"(acc_1[3]) :: "memory");
                asm volatile("" : "+f"(acc_1[4]) :: "memory");
                asm volatile("" : "+f"(acc_1[5]) :: "memory");
                asm volatile("" : "+f"(acc_1[6]) :: "memory");
                asm volatile("" : "+f"(acc_1[7]) :: "memory");
                asm volatile("" : "+f"(acc_2[0]) :: "memory");
                asm volatile("" : "+f"(acc_2[1]) :: "memory");
                asm volatile("" : "+f"(acc_2[2]) :: "memory");
                asm volatile("" : "+f"(acc_2[3]) :: "memory");
                asm volatile("" : "+f"(acc_2[4]) :: "memory");
                asm volatile("" : "+f"(acc_2[5]) :: "memory");
                asm volatile("" : "+f"(acc_2[6]) :: "memory");
                asm volatile("" : "+f"(acc_2[7]) :: "memory");
                asm volatile("" : "+f"(acc_3[0]) :: "memory");
                asm volatile("" : "+f"(acc_3[1]) :: "memory");
                asm volatile("" : "+f"(acc_3[2]) :: "memory");
                asm volatile("" : "+f"(acc_3[3]) :: "memory");
                asm volatile("" : "+f"(acc_3[4]) :: "memory");
                asm volatile("" : "+f"(acc_3[5]) :: "memory");
                asm volatile("" : "+f"(acc_3[6]) :: "memory");
                asm volatile("" : "+f"(acc_3[7]) :: "memory");
                asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
                asm volatile("" : "+f"(acc_0[0]) :: "memory");
                asm volatile("" : "+f"(acc_0[1]) :: "memory");
                asm volatile("" : "+f"(acc_0[2]) :: "memory");
                asm volatile("" : "+f"(acc_0[3]) :: "memory");
                asm volatile("" : "+f"(acc_0[4]) :: "memory");
                asm volatile("" : "+f"(acc_0[5]) :: "memory");
                asm volatile("" : "+f"(acc_0[6]) :: "memory");
                asm volatile("" : "+f"(acc_0[7]) :: "memory");
                asm volatile("" : "+f"(acc_1[0]) :: "memory");
                asm volatile("" : "+f"(acc_1[1]) :: "memory");
                asm volatile("" : "+f"(acc_1[2]) :: "memory");
                asm volatile("" : "+f"(acc_1[3]) :: "memory");
                asm volatile("" : "+f"(acc_1[4]) :: "memory");
                asm volatile("" : "+f"(acc_1[5]) :: "memory");
                asm volatile("" : "+f"(acc_1[6]) :: "memory");
                asm volatile("" : "+f"(acc_1[7]) :: "memory");
                asm volatile("" : "+f"(acc_2[0]) :: "memory");
                asm volatile("" : "+f"(acc_2[1]) :: "memory");
                asm volatile("" : "+f"(acc_2[2]) :: "memory");
                asm volatile("" : "+f"(acc_2[3]) :: "memory");
                asm volatile("" : "+f"(acc_2[4]) :: "memory");
                asm volatile("" : "+f"(acc_2[5]) :: "memory");
                asm volatile("" : "+f"(acc_2[6]) :: "memory");
                asm volatile("" : "+f"(acc_2[7]) :: "memory");
                asm volatile("" : "+f"(acc_3[0]) :: "memory");
                asm volatile("" : "+f"(acc_3[1]) :: "memory");
                asm volatile("" : "+f"(acc_3[2]) :: "memory");
                asm volatile("" : "+f"(acc_3[3]) :: "memory");
                asm volatile("" : "+f"(acc_3[4]) :: "memory");
                asm volatile("" : "+f"(acc_3[5]) :: "memory");
                asm volatile("" : "+f"(acc_3[6]) :: "memory");
                asm volatile("" : "+f"(acc_3[7]) :: "memory");
                if (st > 0) {
                    if (lane == 0) {
                        mbarrier_arrive(empty_addr + ((int)((unsigned int)(st - 1) % 5)) * 8);
                    }
                }
                slot += 1;
                if (slot == 5) { slot = 0; _phase_full ^= 1; }
            }
            mbarrier_wait(full_addr + (slot) * 8, _phase_full);
            asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
            unsigned int sb_1 = base + (unsigned int)(slot * 40960);
            uint64_t d_b_4 = (((uint64_t)(((sb_1 + (33792u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
            uint64_t d_a_4 = (((uint64_t)(((sb_1 + (1024u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
            uint64_t d_b_5 = (((uint64_t)(((sb_1 + (33792u + 2048u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
            uint64_t d_a_5 = (((uint64_t)(((sb_1 + (1024u + 8192u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
            uint64_t d_b_6 = (((uint64_t)(((sb_1 + (33792u + 4096u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
            uint64_t d_a_6 = (((uint64_t)(((sb_1 + (1024u + 16384u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
            uint64_t d_b_7 = (((uint64_t)(((sb_1 + (33792u + 6144u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
            uint64_t d_a_7 = (((uint64_t)(((sb_1 + (1024u + 24576u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3]), "+f"(acc_0[4]), "+f"(acc_0[5]), "+f"(acc_0[6]), "+f"(acc_0[7])
                : "l"(d_a_4), "l"(d_b_4)
                : "memory");
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3]), "+f"(acc_1[4]), "+f"(acc_1[5]), "+f"(acc_1[6]), "+f"(acc_1[7])
                : "l"(d_a_5), "l"(d_b_5)
                : "memory");
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                : "+f"(acc_2[0]), "+f"(acc_2[1]), "+f"(acc_2[2]), "+f"(acc_2[3]), "+f"(acc_2[4]), "+f"(acc_2[5]), "+f"(acc_2[6]), "+f"(acc_2[7])
                : "l"(d_a_6), "l"(d_b_6)
                : "memory");
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                : "+f"(acc_3[0]), "+f"(acc_3[1]), "+f"(acc_3[2]), "+f"(acc_3[3]), "+f"(acc_3[4]), "+f"(acc_3[5]), "+f"(acc_3[6]), "+f"(acc_3[7])
                : "l"(d_a_7), "l"(d_b_7)
                : "memory");
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3]), "+f"(acc_0[4]), "+f"(acc_0[5]), "+f"(acc_0[6]), "+f"(acc_0[7])
                : "l"(d_a_4 + 2), "l"(d_b_4 + 2)
                : "memory");
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3]), "+f"(acc_1[4]), "+f"(acc_1[5]), "+f"(acc_1[6]), "+f"(acc_1[7])
                : "l"(d_a_5 + 2), "l"(d_b_5 + 2)
                : "memory");
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                : "+f"(acc_2[0]), "+f"(acc_2[1]), "+f"(acc_2[2]), "+f"(acc_2[3]), "+f"(acc_2[4]), "+f"(acc_2[5]), "+f"(acc_2[6]), "+f"(acc_2[7])
                : "l"(d_a_6 + 2), "l"(d_b_6 + 2)
                : "memory");
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                : "+f"(acc_3[0]), "+f"(acc_3[1]), "+f"(acc_3[2]), "+f"(acc_3[3]), "+f"(acc_3[4]), "+f"(acc_3[5]), "+f"(acc_3[6]), "+f"(acc_3[7])
                : "l"(d_a_7 + 2), "l"(d_b_7 + 2)
                : "memory");
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3]), "+f"(acc_0[4]), "+f"(acc_0[5]), "+f"(acc_0[6]), "+f"(acc_0[7])
                : "l"(d_a_4 + 4), "l"(d_b_4 + 4)
                : "memory");
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3]), "+f"(acc_1[4]), "+f"(acc_1[5]), "+f"(acc_1[6]), "+f"(acc_1[7])
                : "l"(d_a_5 + 4), "l"(d_b_5 + 4)
                : "memory");
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                : "+f"(acc_2[0]), "+f"(acc_2[1]), "+f"(acc_2[2]), "+f"(acc_2[3]), "+f"(acc_2[4]), "+f"(acc_2[5]), "+f"(acc_2[6]), "+f"(acc_2[7])
                : "l"(d_a_6 + 4), "l"(d_b_6 + 4)
                : "memory");
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                : "+f"(acc_3[0]), "+f"(acc_3[1]), "+f"(acc_3[2]), "+f"(acc_3[3]), "+f"(acc_3[4]), "+f"(acc_3[5]), "+f"(acc_3[6]), "+f"(acc_3[7])
                : "l"(d_a_7 + 4), "l"(d_b_7 + 4)
                : "memory");
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3]), "+f"(acc_0[4]), "+f"(acc_0[5]), "+f"(acc_0[6]), "+f"(acc_0[7])
                : "l"(d_a_4 + 6), "l"(d_b_4 + 6)
                : "memory");
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3]), "+f"(acc_1[4]), "+f"(acc_1[5]), "+f"(acc_1[6]), "+f"(acc_1[7])
                : "l"(d_a_5 + 6), "l"(d_b_5 + 6)
                : "memory");
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                : "+f"(acc_2[0]), "+f"(acc_2[1]), "+f"(acc_2[2]), "+f"(acc_2[3]), "+f"(acc_2[4]), "+f"(acc_2[5]), "+f"(acc_2[6]), "+f"(acc_2[7])
                : "l"(d_a_6 + 6), "l"(d_b_6 + 6)
                : "memory");
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                : "+f"(acc_3[0]), "+f"(acc_3[1]), "+f"(acc_3[2]), "+f"(acc_3[3]), "+f"(acc_3[4]), "+f"(acc_3[5]), "+f"(acc_3[6]), "+f"(acc_3[7])
                : "l"(d_a_7 + 6), "l"(d_b_7 + 6)
                : "memory");
            asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
            asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
            asm volatile("" : "+f"(acc_0[0]) :: "memory");
            asm volatile("" : "+f"(acc_0[1]) :: "memory");
            asm volatile("" : "+f"(acc_0[2]) :: "memory");
            asm volatile("" : "+f"(acc_0[3]) :: "memory");
            asm volatile("" : "+f"(acc_0[4]) :: "memory");
            asm volatile("" : "+f"(acc_0[5]) :: "memory");
            asm volatile("" : "+f"(acc_0[6]) :: "memory");
            asm volatile("" : "+f"(acc_0[7]) :: "memory");
            asm volatile("" : "+f"(acc_1[0]) :: "memory");
            asm volatile("" : "+f"(acc_1[1]) :: "memory");
            asm volatile("" : "+f"(acc_1[2]) :: "memory");
            asm volatile("" : "+f"(acc_1[3]) :: "memory");
            asm volatile("" : "+f"(acc_1[4]) :: "memory");
            asm volatile("" : "+f"(acc_1[5]) :: "memory");
            asm volatile("" : "+f"(acc_1[6]) :: "memory");
            asm volatile("" : "+f"(acc_1[7]) :: "memory");
            asm volatile("" : "+f"(acc_2[0]) :: "memory");
            asm volatile("" : "+f"(acc_2[1]) :: "memory");
            asm volatile("" : "+f"(acc_2[2]) :: "memory");
            asm volatile("" : "+f"(acc_2[3]) :: "memory");
            asm volatile("" : "+f"(acc_2[4]) :: "memory");
            asm volatile("" : "+f"(acc_2[5]) :: "memory");
            asm volatile("" : "+f"(acc_2[6]) :: "memory");
            asm volatile("" : "+f"(acc_2[7]) :: "memory");
            asm volatile("" : "+f"(acc_3[0]) :: "memory");
            asm volatile("" : "+f"(acc_3[1]) :: "memory");
            asm volatile("" : "+f"(acc_3[2]) :: "memory");
            asm volatile("" : "+f"(acc_3[3]) :: "memory");
            asm volatile("" : "+f"(acc_3[4]) :: "memory");
            asm volatile("" : "+f"(acc_3[5]) :: "memory");
            asm volatile("" : "+f"(acc_3[6]) :: "memory");
            asm volatile("" : "+f"(acc_3[7]) :: "memory");
            int m_e_0 = m0 + warp * 16 + g;
            int n_e_0 = n0 + 2 * t;
            float v_e_0 = acc_0[0] + acc_1[0] + acc_2[0] + acc_3[0];
            if (m_e_0 < M && n_e_0 < N) {
                *(reinterpret_cast<__nv_bfloat16*>(out + (m_e_0 + n_e_0 * M)) + (0)) = (__nv_bfloat16)v_e_0;
            }
            int m_e_1 = m0 + warp * 16 + g;
            int n_e_1 = n0 + 2 * t + 1;
            float v_e_1 = acc_0[1] + acc_1[1] + acc_2[1] + acc_3[1];
            if (m_e_1 < M && n_e_1 < N) {
                *(reinterpret_cast<__nv_bfloat16*>(out + (m_e_1 + n_e_1 * M)) + (0)) = (__nv_bfloat16)v_e_1;
            }
            int m_e_2 = m0 + warp * 16 + g + 8;
            int n_e_2 = n0 + 2 * t;
            float v_e_2 = acc_0[2] + acc_1[2] + acc_2[2] + acc_3[2];
            if (m_e_2 < M && n_e_2 < N) {
                *(reinterpret_cast<__nv_bfloat16*>(out + (m_e_2 + n_e_2 * M)) + (0)) = (__nv_bfloat16)v_e_2;
            }
            int m_e_3 = m0 + warp * 16 + g + 8;
            int n_e_3 = n0 + 2 * t + 1;
            float v_e_3 = acc_0[3] + acc_1[3] + acc_2[3] + acc_3[3];
            if (m_e_3 < M && n_e_3 < N) {
                *(reinterpret_cast<__nv_bfloat16*>(out + (m_e_3 + n_e_3 * M)) + (0)) = (__nv_bfloat16)v_e_3;
            }
            int m_e_4 = m0 + warp * 16 + g;
            int n_e_4 = n0 + 8 + 2 * t;
            float v_e_4 = acc_0[4] + acc_1[4] + acc_2[4] + acc_3[4];
            if (m_e_4 < M && n_e_4 < N) {
                *(reinterpret_cast<__nv_bfloat16*>(out + (m_e_4 + n_e_4 * M)) + (0)) = (__nv_bfloat16)v_e_4;
            }
            int m_e_5 = m0 + warp * 16 + g;
            int n_e_5 = n0 + 8 + 2 * t + 1;
            float v_e_5 = acc_0[5] + acc_1[5] + acc_2[5] + acc_3[5];
            if (m_e_5 < M && n_e_5 < N) {
                *(reinterpret_cast<__nv_bfloat16*>(out + (m_e_5 + n_e_5 * M)) + (0)) = (__nv_bfloat16)v_e_5;
            }
            int m_e_6 = m0 + warp * 16 + g + 8;
            int n_e_6 = n0 + 8 + 2 * t;
            float v_e_6 = acc_0[6] + acc_1[6] + acc_2[6] + acc_3[6];
            if (m_e_6 < M && n_e_6 < N) {
                *(reinterpret_cast<__nv_bfloat16*>(out + (m_e_6 + n_e_6 * M)) + (0)) = (__nv_bfloat16)v_e_6;
            }
            int m_e_7 = m0 + warp * 16 + g + 8;
            int n_e_7 = n0 + 8 + 2 * t + 1;
            float v_e_7 = acc_0[7] + acc_1[7] + acc_2[7] + acc_3[7];
            if (m_e_7 < M && n_e_7 < N) {
                *(reinterpret_cast<__nv_bfloat16*>(out + (m_e_7 + n_e_7 * M)) + (0)) = (__nv_bfloat16)v_e_7;
            }
        }
    }
    // ---- Role: load ----
    if (warp == 4) {
        { // load_main
            int pre = ((5) < (nstages) ? (5) : (nstages));
            if (elect_sync()) {
                #pragma unroll 1
                for (int st_1 = 0; st_1 < pre; st_1++) {
                    mbarrier_arrive_expect_tx(full_addr + ((int)((unsigned int)st_1 % 5)) * 8, 40960);
                    int kb = (st_1 / 4 * 16 + st_1 % 4) * 64;
                    unsigned int dst = sw_addr + (unsigned int)(st_1 % 5 * 40960);
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3}], [%4], %5;"
                        :: "r"(dst + 0), "l"((&W)), "r"(kb), "r"(m0),
                           "r"(full_addr + ((int)((unsigned int)st_1 % 5)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3}], [%4], %5;"
                        :: "r"(dst + 8192), "l"((&W)), "r"(kb + 256), "r"(m0),
                           "r"(full_addr + ((int)((unsigned int)st_1 % 5)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3}], [%4], %5;"
                        :: "r"(dst + 16384), "l"((&W)), "r"(kb + 512), "r"(m0),
                           "r"(full_addr + ((int)((unsigned int)st_1 % 5)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3}], [%4], %5;"
                        :: "r"(dst + 24576), "l"((&W)), "r"(kb + 768), "r"(m0),
                           "r"(full_addr + ((int)((unsigned int)st_1 % 5)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                }
            }
            asm volatile("griddepcontrol.wait;" ::: "memory");
            if (lane == 0) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
            if (elect_sync()) {
                #pragma unroll 1
                for (int st_2 = 0; st_2 < pre; st_2++) {
                    int kbx = (st_2 / 4 * 16 + st_2 % 4) * 64;
                    unsigned int dstx = sx_addr + (unsigned int)(st_2 % 5 * 40960);
                    tma_2d_gmem2smem(dstx + 0, (&X), kbx, n0, full_addr + ((int)((unsigned int)st_2 % 5)) * 8);
                    tma_2d_gmem2smem(dstx + 2048, (&X), kbx + 256, n0, full_addr + ((int)((unsigned int)st_2 % 5)) * 8);
                    tma_2d_gmem2smem(dstx + 4096, (&X), kbx + 512, n0, full_addr + ((int)((unsigned int)st_2 % 5)) * 8);
                    tma_2d_gmem2smem(dstx + 6144, (&X), kbx + 768, n0, full_addr + ((int)((unsigned int)st_2 % 5)) * 8);
                }
                #pragma unroll 1
                for (int st_3 = pre; st_3 < nstages; st_3++) {
                    int _stage_empty = (int)((unsigned int)st_3 % 5);
                    int _parity_empty = (int)((unsigned int)st_3 / 5 % 2) ^ 1;
                    mbarrier_wait(empty_addr + (_stage_empty) * 8, _parity_empty);
                    mbarrier_arrive_expect_tx(full_addr + ((int)((unsigned int)st_3 % 5)) * 8, 40960);
                    int kb_1 = (st_3 / 4 * 16 + st_3 % 4) * 64;
                    unsigned int dst_1 = sw_addr + (unsigned int)(st_3 % 5 * 40960);
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3}], [%4], %5;"
                        :: "r"(dst_1 + 0), "l"((&W)), "r"(kb_1), "r"(m0),
                           "r"(full_addr + ((int)((unsigned int)st_3 % 5)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3}], [%4], %5;"
                        :: "r"(dst_1 + 8192), "l"((&W)), "r"(kb_1 + 256), "r"(m0),
                           "r"(full_addr + ((int)((unsigned int)st_3 % 5)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3}], [%4], %5;"
                        :: "r"(dst_1 + 16384), "l"((&W)), "r"(kb_1 + 512), "r"(m0),
                           "r"(full_addr + ((int)((unsigned int)st_3 % 5)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3}], [%4], %5;"
                        :: "r"(dst_1 + 24576), "l"((&W)), "r"(kb_1 + 768), "r"(m0),
                           "r"(full_addr + ((int)((unsigned int)st_3 % 5)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    int kbx_1 = (st_3 / 4 * 16 + st_3 % 4) * 64;
                    unsigned int dstx_1 = sx_addr + (unsigned int)(st_3 % 5 * 40960);
                    tma_2d_gmem2smem(dstx_1 + 0, (&X), kbx_1, n0, full_addr + ((int)((unsigned int)st_3 % 5)) * 8);
                    tma_2d_gmem2smem(dstx_1 + 2048, (&X), kbx_1 + 256, n0, full_addr + ((int)((unsigned int)st_3 % 5)) * 8);
                    tma_2d_gmem2smem(dstx_1 + 4096, (&X), kbx_1 + 512, n0, full_addr + ((int)((unsigned int)st_3 % 5)) * 8);
                    tma_2d_gmem2smem(dstx_1 + 6144, (&X), kbx_1 + 768, n0, full_addr + ((int)((unsigned int)st_3 % 5)) * 8);
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
