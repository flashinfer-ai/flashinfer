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
#define NUM_PIPE_STAGES 2
#define SMEM_SW_OFF 1024
#define SMEM_SW_STAGE_BYTES 22528
#define SMEM_SW_STRIDE 24576
#define SMEM_SX_OFF 17408
#define SMEM_SX_STAGE_BYTES 8192
#define SMEM_SX_STRIDE 24576
#define SMEM_TOTAL 50176
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


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(160, LAUNCH_MIN_BLOCKS) void
kernel_cake_tinygemm2_wgmma_68afbe92e2300b660538(const __grid_constant__ CUtensorMap W, const __grid_constant__ CUtensorMap X, __nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ bias, int M, int N, int K)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define full_addr (mbar_base + 0)
    #define empty_addr (mbar_base + 16)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* sw = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int sw_addr = smem + 1024;
    __nv_bfloat16* sx = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int sx_addr = smem + 17408;
    if (tid == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&W))) : "memory"); }
    if (tid == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&X))) : "memory"); }

    // Mbarrier init (2 pipeline groups, 0 ordered-sequence groups, 4 barriers)
    // Mbarriers at smem_raw[0..32)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'pipe' ---
            // full: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            // empty: 2 barriers, init_count=4
            mbarrier_init(smem + 16, 4);
            mbarrier_init(smem + 24, 4);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // Kernel post-init ops
    int rank = 0;
    int n0 = blockIdx.x * 16;
    int m0 = blockIdx.y * 32;
    int rounds = (K + 1023) / 1024;
    int nslices = ((rounds * 4) < ((K + 63) / 64 - 12 * (rounds - 1)) ? (rounds * 4) : ((K + 63) / 64 - 12 * (rounds - 1)));
    int nstages = (nslices + 4 - 1) / 4;

    // ---- Role: mma ----
    if (warp <= 3) {
        { // mma_main
            float acc_0[8] = {0};
            float acc_1[8] = {0};
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
            unsigned int _phase_full = 0;
            mbarrier_wait(full_addr + (slot) * 8, _phase_full);
            asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
            unsigned int sb = base + (unsigned int)(slot * 24576);
            uint64_t d_b_0 = (((uint64_t)(((sb + (17408u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
            uint64_t d_a_0 = (((uint64_t)(((sb + (1024u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
            uint64_t d_a_1 = (((uint64_t)(((sb + (1024u + 8192u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3]), "+f"(acc_0[4]), "+f"(acc_0[5]), "+f"(acc_0[6]), "+f"(acc_0[7])
                : "l"(d_a_0), "l"(d_b_0)
                : "memory");
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3]), "+f"(acc_1[4]), "+f"(acc_1[5]), "+f"(acc_1[6]), "+f"(acc_1[7])
                : "l"(d_a_1), "l"(d_b_0)
                : "memory");
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3]), "+f"(acc_0[4]), "+f"(acc_0[5]), "+f"(acc_0[6]), "+f"(acc_0[7])
                : "l"(d_a_0 + 2), "l"(d_b_0 + 2)
                : "memory");
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3]), "+f"(acc_1[4]), "+f"(acc_1[5]), "+f"(acc_1[6]), "+f"(acc_1[7])
                : "l"(d_a_1 + 2), "l"(d_b_0 + 2)
                : "memory");
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3]), "+f"(acc_0[4]), "+f"(acc_0[5]), "+f"(acc_0[6]), "+f"(acc_0[7])
                : "l"(d_a_0 + 4), "l"(d_b_0 + 4)
                : "memory");
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3]), "+f"(acc_1[4]), "+f"(acc_1[5]), "+f"(acc_1[6]), "+f"(acc_1[7])
                : "l"(d_a_1 + 4), "l"(d_b_0 + 4)
                : "memory");
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3]), "+f"(acc_0[4]), "+f"(acc_0[5]), "+f"(acc_0[6]), "+f"(acc_0[7])
                : "l"(d_a_0 + 6), "l"(d_b_0 + 6)
                : "memory");
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3]), "+f"(acc_1[4]), "+f"(acc_1[5]), "+f"(acc_1[6]), "+f"(acc_1[7])
                : "l"(d_a_1 + 6), "l"(d_b_0 + 6)
                : "memory");
            if (1 < nslices) {
                uint64_t d_b_1 = (((uint64_t)(((sb + (17408u + 2048u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                uint64_t d_a_2 = (((uint64_t)(((sb + (1024u + 2048u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                uint64_t d_a_3 = (((uint64_t)(((sb + (1024u + 10240u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3]), "+f"(acc_0[4]), "+f"(acc_0[5]), "+f"(acc_0[6]), "+f"(acc_0[7])
                    : "l"(d_a_2), "l"(d_b_1)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3]), "+f"(acc_1[4]), "+f"(acc_1[5]), "+f"(acc_1[6]), "+f"(acc_1[7])
                    : "l"(d_a_3), "l"(d_b_1)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3]), "+f"(acc_0[4]), "+f"(acc_0[5]), "+f"(acc_0[6]), "+f"(acc_0[7])
                    : "l"(d_a_2 + 2), "l"(d_b_1 + 2)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3]), "+f"(acc_1[4]), "+f"(acc_1[5]), "+f"(acc_1[6]), "+f"(acc_1[7])
                    : "l"(d_a_3 + 2), "l"(d_b_1 + 2)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3]), "+f"(acc_0[4]), "+f"(acc_0[5]), "+f"(acc_0[6]), "+f"(acc_0[7])
                    : "l"(d_a_2 + 4), "l"(d_b_1 + 4)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3]), "+f"(acc_1[4]), "+f"(acc_1[5]), "+f"(acc_1[6]), "+f"(acc_1[7])
                    : "l"(d_a_3 + 4), "l"(d_b_1 + 4)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3]), "+f"(acc_0[4]), "+f"(acc_0[5]), "+f"(acc_0[6]), "+f"(acc_0[7])
                    : "l"(d_a_2 + 6), "l"(d_b_1 + 6)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3]), "+f"(acc_1[4]), "+f"(acc_1[5]), "+f"(acc_1[6]), "+f"(acc_1[7])
                    : "l"(d_a_3 + 6), "l"(d_b_1 + 6)
                    : "memory");
            }
            if (2 < nslices) {
                uint64_t d_b_2 = (((uint64_t)(((sb + (17408u + 4096u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                uint64_t d_a_4 = (((uint64_t)(((sb + (1024u + 4096u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                uint64_t d_a_5 = (((uint64_t)(((sb + (1024u + 12288u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3]), "+f"(acc_0[4]), "+f"(acc_0[5]), "+f"(acc_0[6]), "+f"(acc_0[7])
                    : "l"(d_a_4), "l"(d_b_2)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3]), "+f"(acc_1[4]), "+f"(acc_1[5]), "+f"(acc_1[6]), "+f"(acc_1[7])
                    : "l"(d_a_5), "l"(d_b_2)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3]), "+f"(acc_0[4]), "+f"(acc_0[5]), "+f"(acc_0[6]), "+f"(acc_0[7])
                    : "l"(d_a_4 + 2), "l"(d_b_2 + 2)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3]), "+f"(acc_1[4]), "+f"(acc_1[5]), "+f"(acc_1[6]), "+f"(acc_1[7])
                    : "l"(d_a_5 + 2), "l"(d_b_2 + 2)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3]), "+f"(acc_0[4]), "+f"(acc_0[5]), "+f"(acc_0[6]), "+f"(acc_0[7])
                    : "l"(d_a_4 + 4), "l"(d_b_2 + 4)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3]), "+f"(acc_1[4]), "+f"(acc_1[5]), "+f"(acc_1[6]), "+f"(acc_1[7])
                    : "l"(d_a_5 + 4), "l"(d_b_2 + 4)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3]), "+f"(acc_0[4]), "+f"(acc_0[5]), "+f"(acc_0[6]), "+f"(acc_0[7])
                    : "l"(d_a_4 + 6), "l"(d_b_2 + 6)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3]), "+f"(acc_1[4]), "+f"(acc_1[5]), "+f"(acc_1[6]), "+f"(acc_1[7])
                    : "l"(d_a_5 + 6), "l"(d_b_2 + 6)
                    : "memory");
            }
            if (3 < nslices) {
                uint64_t d_b_3 = (((uint64_t)(((sb + (17408u + 6144u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                uint64_t d_a_6 = (((uint64_t)(((sb + (1024u + 6144u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                uint64_t d_a_7 = (((uint64_t)(((sb + (1024u + 14336u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3]), "+f"(acc_0[4]), "+f"(acc_0[5]), "+f"(acc_0[6]), "+f"(acc_0[7])
                    : "l"(d_a_6), "l"(d_b_3)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3]), "+f"(acc_1[4]), "+f"(acc_1[5]), "+f"(acc_1[6]), "+f"(acc_1[7])
                    : "l"(d_a_7), "l"(d_b_3)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3]), "+f"(acc_0[4]), "+f"(acc_0[5]), "+f"(acc_0[6]), "+f"(acc_0[7])
                    : "l"(d_a_6 + 2), "l"(d_b_3 + 2)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3]), "+f"(acc_1[4]), "+f"(acc_1[5]), "+f"(acc_1[6]), "+f"(acc_1[7])
                    : "l"(d_a_7 + 2), "l"(d_b_3 + 2)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3]), "+f"(acc_0[4]), "+f"(acc_0[5]), "+f"(acc_0[6]), "+f"(acc_0[7])
                    : "l"(d_a_6 + 4), "l"(d_b_3 + 4)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3]), "+f"(acc_1[4]), "+f"(acc_1[5]), "+f"(acc_1[6]), "+f"(acc_1[7])
                    : "l"(d_a_7 + 4), "l"(d_b_3 + 4)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3]), "+f"(acc_0[4]), "+f"(acc_0[5]), "+f"(acc_0[6]), "+f"(acc_0[7])
                    : "l"(d_a_6 + 6), "l"(d_b_3 + 6)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n16k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7}, %8, %9, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_1[0]), "+f"(acc_1[1]), "+f"(acc_1[2]), "+f"(acc_1[3]), "+f"(acc_1[4]), "+f"(acc_1[5]), "+f"(acc_1[6]), "+f"(acc_1[7])
                    : "l"(d_a_7 + 6), "l"(d_b_3 + 6)
                    : "memory");
            }
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
            if (warp == 0) {
                float v_e_0 = acc_0[0];
                float _add_0;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_0) : "f"(v_e_0), "f"(0.0f));
                v_e_0 = _add_0;
                float _add_1;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_1) : "f"(v_e_0), "f"(0.0f));
                v_e_0 = _add_1;
                float _add_2;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_2) : "f"(v_e_0), "f"(0.0f));
                v_e_0 = _add_2;
                int m_e_0 = m0 + g;
                int n_e_0 = n0 + 2 * t;
                if (m_e_0 < M && n_e_0 < N) {
                    *(reinterpret_cast<__nv_bfloat16*>(out + (m_e_0 + n_e_0 * M)) + (0)) = (__nv_bfloat16)v_e_0;
                }
                float v_e_1 = acc_0[1];
                float _add_3;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_3) : "f"(v_e_1), "f"(0.0f));
                v_e_1 = _add_3;
                float _add_4;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_4) : "f"(v_e_1), "f"(0.0f));
                v_e_1 = _add_4;
                float _add_5;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_5) : "f"(v_e_1), "f"(0.0f));
                v_e_1 = _add_5;
                int m_e_1 = m0 + g;
                int n_e_1 = n0 + 2 * t + 1;
                if (m_e_1 < M && n_e_1 < N) {
                    *(reinterpret_cast<__nv_bfloat16*>(out + (m_e_1 + n_e_1 * M)) + (0)) = (__nv_bfloat16)v_e_1;
                }
                float v_e_2 = acc_0[2];
                float _add_6;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_6) : "f"(v_e_2), "f"(0.0f));
                v_e_2 = _add_6;
                float _add_7;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_7) : "f"(v_e_2), "f"(0.0f));
                v_e_2 = _add_7;
                float _add_8;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_8) : "f"(v_e_2), "f"(0.0f));
                v_e_2 = _add_8;
                int m_e_2 = m0 + g + 8;
                int n_e_2 = n0 + 2 * t;
                if (m_e_2 < M && n_e_2 < N) {
                    *(reinterpret_cast<__nv_bfloat16*>(out + (m_e_2 + n_e_2 * M)) + (0)) = (__nv_bfloat16)v_e_2;
                }
                float v_e_3 = acc_0[3];
                float _add_9;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_9) : "f"(v_e_3), "f"(0.0f));
                v_e_3 = _add_9;
                float _add_10;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_10) : "f"(v_e_3), "f"(0.0f));
                v_e_3 = _add_10;
                float _add_11;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_11) : "f"(v_e_3), "f"(0.0f));
                v_e_3 = _add_11;
                int m_e_3 = m0 + g + 8;
                int n_e_3 = n0 + 2 * t + 1;
                if (m_e_3 < M && n_e_3 < N) {
                    *(reinterpret_cast<__nv_bfloat16*>(out + (m_e_3 + n_e_3 * M)) + (0)) = (__nv_bfloat16)v_e_3;
                }
                float v_e_4 = acc_0[4];
                float _add_12;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_12) : "f"(v_e_4), "f"(0.0f));
                v_e_4 = _add_12;
                float _add_13;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_13) : "f"(v_e_4), "f"(0.0f));
                v_e_4 = _add_13;
                float _add_14;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_14) : "f"(v_e_4), "f"(0.0f));
                v_e_4 = _add_14;
                int m_e_4 = m0 + g;
                int n_e_4 = n0 + 8 + 2 * t;
                if (m_e_4 < M && n_e_4 < N) {
                    *(reinterpret_cast<__nv_bfloat16*>(out + (m_e_4 + n_e_4 * M)) + (0)) = (__nv_bfloat16)v_e_4;
                }
                float v_e_5 = acc_0[5];
                float _add_15;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_15) : "f"(v_e_5), "f"(0.0f));
                v_e_5 = _add_15;
                float _add_16;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_16) : "f"(v_e_5), "f"(0.0f));
                v_e_5 = _add_16;
                float _add_17;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_17) : "f"(v_e_5), "f"(0.0f));
                v_e_5 = _add_17;
                int m_e_5 = m0 + g;
                int n_e_5 = n0 + 8 + 2 * t + 1;
                if (m_e_5 < M && n_e_5 < N) {
                    *(reinterpret_cast<__nv_bfloat16*>(out + (m_e_5 + n_e_5 * M)) + (0)) = (__nv_bfloat16)v_e_5;
                }
                float v_e_6 = acc_0[6];
                float _add_18;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_18) : "f"(v_e_6), "f"(0.0f));
                v_e_6 = _add_18;
                float _add_19;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_19) : "f"(v_e_6), "f"(0.0f));
                v_e_6 = _add_19;
                float _add_20;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_20) : "f"(v_e_6), "f"(0.0f));
                v_e_6 = _add_20;
                int m_e_6 = m0 + g + 8;
                int n_e_6 = n0 + 8 + 2 * t;
                if (m_e_6 < M && n_e_6 < N) {
                    *(reinterpret_cast<__nv_bfloat16*>(out + (m_e_6 + n_e_6 * M)) + (0)) = (__nv_bfloat16)v_e_6;
                }
                float v_e_7 = acc_0[7];
                float _add_21;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_21) : "f"(v_e_7), "f"(0.0f));
                v_e_7 = _add_21;
                float _add_22;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_22) : "f"(v_e_7), "f"(0.0f));
                v_e_7 = _add_22;
                float _add_23;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_23) : "f"(v_e_7), "f"(0.0f));
                v_e_7 = _add_23;
                int m_e_7 = m0 + g + 8;
                int n_e_7 = n0 + 8 + 2 * t + 1;
                if (m_e_7 < M && n_e_7 < N) {
                    *(reinterpret_cast<__nv_bfloat16*>(out + (m_e_7 + n_e_7 * M)) + (0)) = (__nv_bfloat16)v_e_7;
                }
                float v_e_8 = acc_1[0];
                float _add_24;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_24) : "f"(v_e_8), "f"(0.0f));
                v_e_8 = _add_24;
                float _add_25;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_25) : "f"(v_e_8), "f"(0.0f));
                v_e_8 = _add_25;
                float _add_26;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_26) : "f"(v_e_8), "f"(0.0f));
                v_e_8 = _add_26;
                int m_e_8 = m0 + 16 + g;
                int n_e_8 = n0 + 2 * t;
                if (m_e_8 < M && n_e_8 < N) {
                    *(reinterpret_cast<__nv_bfloat16*>(out + (m_e_8 + n_e_8 * M)) + (0)) = (__nv_bfloat16)v_e_8;
                }
                float v_e_9 = acc_1[1];
                float _add_27;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_27) : "f"(v_e_9), "f"(0.0f));
                v_e_9 = _add_27;
                float _add_28;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_28) : "f"(v_e_9), "f"(0.0f));
                v_e_9 = _add_28;
                float _add_29;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_29) : "f"(v_e_9), "f"(0.0f));
                v_e_9 = _add_29;
                int m_e_9 = m0 + 16 + g;
                int n_e_9 = n0 + 2 * t + 1;
                if (m_e_9 < M && n_e_9 < N) {
                    *(reinterpret_cast<__nv_bfloat16*>(out + (m_e_9 + n_e_9 * M)) + (0)) = (__nv_bfloat16)v_e_9;
                }
                float v_e_10 = acc_1[2];
                float _add_30;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_30) : "f"(v_e_10), "f"(0.0f));
                v_e_10 = _add_30;
                float _add_31;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_31) : "f"(v_e_10), "f"(0.0f));
                v_e_10 = _add_31;
                float _add_32;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_32) : "f"(v_e_10), "f"(0.0f));
                v_e_10 = _add_32;
                int m_e_10 = m0 + 16 + g + 8;
                int n_e_10 = n0 + 2 * t;
                if (m_e_10 < M && n_e_10 < N) {
                    *(reinterpret_cast<__nv_bfloat16*>(out + (m_e_10 + n_e_10 * M)) + (0)) = (__nv_bfloat16)v_e_10;
                }
                float v_e_11 = acc_1[3];
                float _add_33;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_33) : "f"(v_e_11), "f"(0.0f));
                v_e_11 = _add_33;
                float _add_34;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_34) : "f"(v_e_11), "f"(0.0f));
                v_e_11 = _add_34;
                float _add_35;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_35) : "f"(v_e_11), "f"(0.0f));
                v_e_11 = _add_35;
                int m_e_11 = m0 + 16 + g + 8;
                int n_e_11 = n0 + 2 * t + 1;
                if (m_e_11 < M && n_e_11 < N) {
                    *(reinterpret_cast<__nv_bfloat16*>(out + (m_e_11 + n_e_11 * M)) + (0)) = (__nv_bfloat16)v_e_11;
                }
                float v_e_12 = acc_1[4];
                float _add_36;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_36) : "f"(v_e_12), "f"(0.0f));
                v_e_12 = _add_36;
                float _add_37;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_37) : "f"(v_e_12), "f"(0.0f));
                v_e_12 = _add_37;
                float _add_38;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_38) : "f"(v_e_12), "f"(0.0f));
                v_e_12 = _add_38;
                int m_e_12 = m0 + 16 + g;
                int n_e_12 = n0 + 8 + 2 * t;
                if (m_e_12 < M && n_e_12 < N) {
                    *(reinterpret_cast<__nv_bfloat16*>(out + (m_e_12 + n_e_12 * M)) + (0)) = (__nv_bfloat16)v_e_12;
                }
                float v_e_13 = acc_1[5];
                float _add_39;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_39) : "f"(v_e_13), "f"(0.0f));
                v_e_13 = _add_39;
                float _add_40;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_40) : "f"(v_e_13), "f"(0.0f));
                v_e_13 = _add_40;
                float _add_41;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_41) : "f"(v_e_13), "f"(0.0f));
                v_e_13 = _add_41;
                int m_e_13 = m0 + 16 + g;
                int n_e_13 = n0 + 8 + 2 * t + 1;
                if (m_e_13 < M && n_e_13 < N) {
                    *(reinterpret_cast<__nv_bfloat16*>(out + (m_e_13 + n_e_13 * M)) + (0)) = (__nv_bfloat16)v_e_13;
                }
                float v_e_14 = acc_1[6];
                float _add_42;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_42) : "f"(v_e_14), "f"(0.0f));
                v_e_14 = _add_42;
                float _add_43;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_43) : "f"(v_e_14), "f"(0.0f));
                v_e_14 = _add_43;
                float _add_44;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_44) : "f"(v_e_14), "f"(0.0f));
                v_e_14 = _add_44;
                int m_e_14 = m0 + 16 + g + 8;
                int n_e_14 = n0 + 8 + 2 * t;
                if (m_e_14 < M && n_e_14 < N) {
                    *(reinterpret_cast<__nv_bfloat16*>(out + (m_e_14 + n_e_14 * M)) + (0)) = (__nv_bfloat16)v_e_14;
                }
                float v_e_15 = acc_1[7];
                float _add_45;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_45) : "f"(v_e_15), "f"(0.0f));
                v_e_15 = _add_45;
                float _add_46;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_46) : "f"(v_e_15), "f"(0.0f));
                v_e_15 = _add_46;
                float _add_47;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_47) : "f"(v_e_15), "f"(0.0f));
                v_e_15 = _add_47;
                int m_e_15 = m0 + 16 + g + 8;
                int n_e_15 = n0 + 8 + 2 * t + 1;
                if (m_e_15 < M && n_e_15 < N) {
                    *(reinterpret_cast<__nv_bfloat16*>(out + (m_e_15 + n_e_15 * M)) + (0)) = (__nv_bfloat16)v_e_15;
                }
            }
        }
    }
    // ---- Role: load ----
    if (warp == 4) {
        { // load_main
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(full_addr, (nslices + 3) / 4 * 24576);
                tma_3d_gmem2smem(sw_addr + 0, (&W), 0, m0, 4 * rank, full_addr);
                tma_3d_gmem2smem(sw_addr + 8192, (&W), 0, m0 + 16, 4 * rank, full_addr);
            }
            if (elect_sync()) {
                tma_3d_gmem2smem(sx_addr + 0, (&X), 0, n0, 4 * rank, full_addr);
            }
        }
    }

    // Cleanup
}

} // extern "C"
