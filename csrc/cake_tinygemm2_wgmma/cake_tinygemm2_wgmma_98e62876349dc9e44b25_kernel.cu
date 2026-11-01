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
#define NUM_PIPE_STAGES 4
#define SMEM_SW_OFF 1024
#define SMEM_SW_STAGE_BYTES 32768
#define SMEM_SW_STRIDE 49152
#define SMEM_SX_OFF 33792
#define SMEM_SX_STAGE_BYTES 16384
#define SMEM_SX_STRIDE 49152
#define SMEM_RED_OFF 197632
#define SMEM_RED_STAGE_BYTES 2048
#define SMEM_RED_STRIDE 2048
#define SMEM_TOTAL 199680
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
kernel_cake_tinygemm2_wgmma_98e62876349dc9e44b25(const __grid_constant__ CUtensorMap W, const __grid_constant__ CUtensorMap X, __nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ bias, int M, int N, int K)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define full_addr (mbar_base + 0)
    #define empty_addr (mbar_base + 32)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* sw = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int sw_addr = smem + 1024;
    __nv_bfloat16* sx = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int sx_addr = smem + 33792;
    float* red = reinterpret_cast<float*>(smem_raw + 197632);
    const int red_addr = smem + 197632;
    if (tid == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&W))) : "memory"); }
    if (tid == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&X))) : "memory"); }

    // Mbarrier init (2 pipeline groups, 0 ordered-sequence groups, 8 barriers)
    // Mbarriers at smem_raw[0..64)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'pipe' ---
            // full: 4 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // empty: 4 barriers, init_count=4
            mbarrier_init(smem + 32, 4);
            mbarrier_init(smem + 40, 4);
            mbarrier_init(smem + 48, 4);
            mbarrier_init(smem + 56, 4);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // Kernel post-init ops
    int n0 = blockIdx.x * 8;
    int m0 = blockIdx.y * 16;
    int rounds = (K + 1023) / 1024;
    int nslices = ((rounds * 4) < ((K + 63) / 64 - 12 * (rounds - 1)) ? (rounds * 4) : ((K + 63) / 64 - 12 * (rounds - 1)));
    int nstages = (nslices + 4 - 1) / 4;
    float bv_0 = 0.0f;
    if (tid < 128) {
        int _min_0 = ((m0 + tid % 16) < (M - 1) ? (m0 + tid % 16) : (M - 1));
        bv_0 = bias[_min_0];
    }

    // ---- Role: mma ----
    if (warp <= 3) {
        { // mma_main
            float acc_0[16] = {0};
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
            asm volatile("" : "+f"(acc_0[8]) :: "memory");
            asm volatile("" : "+f"(acc_0[9]) :: "memory");
            asm volatile("" : "+f"(acc_0[10]) :: "memory");
            asm volatile("" : "+f"(acc_0[11]) :: "memory");
            asm volatile("" : "+f"(acc_0[12]) :: "memory");
            asm volatile("" : "+f"(acc_0[13]) :: "memory");
            asm volatile("" : "+f"(acc_0[14]) :: "memory");
            asm volatile("" : "+f"(acc_0[15]) :: "memory");
            unsigned int _phase_full = 0;
            #pragma unroll 1
            for (int st = 0; st < nstages - 1; st++) {
                mbarrier_wait(full_addr + (slot) * 8, _phase_full);
                asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
                unsigned int sb = base + (unsigned int)(slot * 49152);
                uint64_t d_b_0 = (((uint64_t)(((sb + (33792u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                uint64_t d_a_0 = (((uint64_t)(((sb + (1024u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3]), "+f"(acc_0[4]), "+f"(acc_0[5]), "+f"(acc_0[6]), "+f"(acc_0[7]), "+f"(acc_0[8]), "+f"(acc_0[9]), "+f"(acc_0[10]), "+f"(acc_0[11]), "+f"(acc_0[12]), "+f"(acc_0[13]), "+f"(acc_0[14]), "+f"(acc_0[15])
                    : "l"(d_a_0), "l"(d_b_0)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3]), "+f"(acc_0[4]), "+f"(acc_0[5]), "+f"(acc_0[6]), "+f"(acc_0[7]), "+f"(acc_0[8]), "+f"(acc_0[9]), "+f"(acc_0[10]), "+f"(acc_0[11]), "+f"(acc_0[12]), "+f"(acc_0[13]), "+f"(acc_0[14]), "+f"(acc_0[15])
                    : "l"(d_a_0 + 2), "l"(d_b_0 + 2)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3]), "+f"(acc_0[4]), "+f"(acc_0[5]), "+f"(acc_0[6]), "+f"(acc_0[7]), "+f"(acc_0[8]), "+f"(acc_0[9]), "+f"(acc_0[10]), "+f"(acc_0[11]), "+f"(acc_0[12]), "+f"(acc_0[13]), "+f"(acc_0[14]), "+f"(acc_0[15])
                    : "l"(d_a_0 + 4), "l"(d_b_0 + 4)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3]), "+f"(acc_0[4]), "+f"(acc_0[5]), "+f"(acc_0[6]), "+f"(acc_0[7]), "+f"(acc_0[8]), "+f"(acc_0[9]), "+f"(acc_0[10]), "+f"(acc_0[11]), "+f"(acc_0[12]), "+f"(acc_0[13]), "+f"(acc_0[14]), "+f"(acc_0[15])
                    : "l"(d_a_0 + 6), "l"(d_b_0 + 6)
                    : "memory");
                uint64_t d_b_1 = (((uint64_t)(((sb + (33792u + 4096u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                uint64_t d_a_1 = (((uint64_t)(((sb + (1024u + 8192u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3]), "+f"(acc_0[4]), "+f"(acc_0[5]), "+f"(acc_0[6]), "+f"(acc_0[7]), "+f"(acc_0[8]), "+f"(acc_0[9]), "+f"(acc_0[10]), "+f"(acc_0[11]), "+f"(acc_0[12]), "+f"(acc_0[13]), "+f"(acc_0[14]), "+f"(acc_0[15])
                    : "l"(d_a_1), "l"(d_b_1)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3]), "+f"(acc_0[4]), "+f"(acc_0[5]), "+f"(acc_0[6]), "+f"(acc_0[7]), "+f"(acc_0[8]), "+f"(acc_0[9]), "+f"(acc_0[10]), "+f"(acc_0[11]), "+f"(acc_0[12]), "+f"(acc_0[13]), "+f"(acc_0[14]), "+f"(acc_0[15])
                    : "l"(d_a_1 + 2), "l"(d_b_1 + 2)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3]), "+f"(acc_0[4]), "+f"(acc_0[5]), "+f"(acc_0[6]), "+f"(acc_0[7]), "+f"(acc_0[8]), "+f"(acc_0[9]), "+f"(acc_0[10]), "+f"(acc_0[11]), "+f"(acc_0[12]), "+f"(acc_0[13]), "+f"(acc_0[14]), "+f"(acc_0[15])
                    : "l"(d_a_1 + 4), "l"(d_b_1 + 4)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3]), "+f"(acc_0[4]), "+f"(acc_0[5]), "+f"(acc_0[6]), "+f"(acc_0[7]), "+f"(acc_0[8]), "+f"(acc_0[9]), "+f"(acc_0[10]), "+f"(acc_0[11]), "+f"(acc_0[12]), "+f"(acc_0[13]), "+f"(acc_0[14]), "+f"(acc_0[15])
                    : "l"(d_a_1 + 6), "l"(d_b_1 + 6)
                    : "memory");
                uint64_t d_b_2 = (((uint64_t)(((sb + (33792u + 8192u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                uint64_t d_a_2 = (((uint64_t)(((sb + (1024u + 16384u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3]), "+f"(acc_0[4]), "+f"(acc_0[5]), "+f"(acc_0[6]), "+f"(acc_0[7]), "+f"(acc_0[8]), "+f"(acc_0[9]), "+f"(acc_0[10]), "+f"(acc_0[11]), "+f"(acc_0[12]), "+f"(acc_0[13]), "+f"(acc_0[14]), "+f"(acc_0[15])
                    : "l"(d_a_2), "l"(d_b_2)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3]), "+f"(acc_0[4]), "+f"(acc_0[5]), "+f"(acc_0[6]), "+f"(acc_0[7]), "+f"(acc_0[8]), "+f"(acc_0[9]), "+f"(acc_0[10]), "+f"(acc_0[11]), "+f"(acc_0[12]), "+f"(acc_0[13]), "+f"(acc_0[14]), "+f"(acc_0[15])
                    : "l"(d_a_2 + 2), "l"(d_b_2 + 2)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3]), "+f"(acc_0[4]), "+f"(acc_0[5]), "+f"(acc_0[6]), "+f"(acc_0[7]), "+f"(acc_0[8]), "+f"(acc_0[9]), "+f"(acc_0[10]), "+f"(acc_0[11]), "+f"(acc_0[12]), "+f"(acc_0[13]), "+f"(acc_0[14]), "+f"(acc_0[15])
                    : "l"(d_a_2 + 4), "l"(d_b_2 + 4)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3]), "+f"(acc_0[4]), "+f"(acc_0[5]), "+f"(acc_0[6]), "+f"(acc_0[7]), "+f"(acc_0[8]), "+f"(acc_0[9]), "+f"(acc_0[10]), "+f"(acc_0[11]), "+f"(acc_0[12]), "+f"(acc_0[13]), "+f"(acc_0[14]), "+f"(acc_0[15])
                    : "l"(d_a_2 + 6), "l"(d_b_2 + 6)
                    : "memory");
                uint64_t d_b_3 = (((uint64_t)(((sb + (33792u + 12288u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                uint64_t d_a_3 = (((uint64_t)(((sb + (1024u + 24576u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3]), "+f"(acc_0[4]), "+f"(acc_0[5]), "+f"(acc_0[6]), "+f"(acc_0[7]), "+f"(acc_0[8]), "+f"(acc_0[9]), "+f"(acc_0[10]), "+f"(acc_0[11]), "+f"(acc_0[12]), "+f"(acc_0[13]), "+f"(acc_0[14]), "+f"(acc_0[15])
                    : "l"(d_a_3), "l"(d_b_3)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3]), "+f"(acc_0[4]), "+f"(acc_0[5]), "+f"(acc_0[6]), "+f"(acc_0[7]), "+f"(acc_0[8]), "+f"(acc_0[9]), "+f"(acc_0[10]), "+f"(acc_0[11]), "+f"(acc_0[12]), "+f"(acc_0[13]), "+f"(acc_0[14]), "+f"(acc_0[15])
                    : "l"(d_a_3 + 2), "l"(d_b_3 + 2)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3]), "+f"(acc_0[4]), "+f"(acc_0[5]), "+f"(acc_0[6]), "+f"(acc_0[7]), "+f"(acc_0[8]), "+f"(acc_0[9]), "+f"(acc_0[10]), "+f"(acc_0[11]), "+f"(acc_0[12]), "+f"(acc_0[13]), "+f"(acc_0[14]), "+f"(acc_0[15])
                    : "l"(d_a_3 + 4), "l"(d_b_3 + 4)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3]), "+f"(acc_0[4]), "+f"(acc_0[5]), "+f"(acc_0[6]), "+f"(acc_0[7]), "+f"(acc_0[8]), "+f"(acc_0[9]), "+f"(acc_0[10]), "+f"(acc_0[11]), "+f"(acc_0[12]), "+f"(acc_0[13]), "+f"(acc_0[14]), "+f"(acc_0[15])
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
                asm volatile("" : "+f"(acc_0[8]) :: "memory");
                asm volatile("" : "+f"(acc_0[9]) :: "memory");
                asm volatile("" : "+f"(acc_0[10]) :: "memory");
                asm volatile("" : "+f"(acc_0[11]) :: "memory");
                asm volatile("" : "+f"(acc_0[12]) :: "memory");
                asm volatile("" : "+f"(acc_0[13]) :: "memory");
                asm volatile("" : "+f"(acc_0[14]) :: "memory");
                asm volatile("" : "+f"(acc_0[15]) :: "memory");
                asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
                asm volatile("" : "+f"(acc_0[0]) :: "memory");
                asm volatile("" : "+f"(acc_0[1]) :: "memory");
                asm volatile("" : "+f"(acc_0[2]) :: "memory");
                asm volatile("" : "+f"(acc_0[3]) :: "memory");
                asm volatile("" : "+f"(acc_0[4]) :: "memory");
                asm volatile("" : "+f"(acc_0[5]) :: "memory");
                asm volatile("" : "+f"(acc_0[6]) :: "memory");
                asm volatile("" : "+f"(acc_0[7]) :: "memory");
                asm volatile("" : "+f"(acc_0[8]) :: "memory");
                asm volatile("" : "+f"(acc_0[9]) :: "memory");
                asm volatile("" : "+f"(acc_0[10]) :: "memory");
                asm volatile("" : "+f"(acc_0[11]) :: "memory");
                asm volatile("" : "+f"(acc_0[12]) :: "memory");
                asm volatile("" : "+f"(acc_0[13]) :: "memory");
                asm volatile("" : "+f"(acc_0[14]) :: "memory");
                asm volatile("" : "+f"(acc_0[15]) :: "memory");
                if (st > 0) {
                    if (lane == 0) {
                        mbarrier_arrive(empty_addr + ((int)((unsigned int)(st - 1) % 4)) * 8);
                    }
                }
                slot += 1;
                if (slot == 4) { slot = 0; _phase_full ^= 1; }
            }
            mbarrier_wait(full_addr + (slot) * 8, _phase_full);
            asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
            unsigned int sb_1 = base + (unsigned int)(slot * 49152);
            uint64_t d_b_4 = (((uint64_t)(((sb_1 + (33792u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
            uint64_t d_a_4 = (((uint64_t)(((sb_1 + (1024u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 1, 1, 1, 0, 0;\n}\n"
                : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3]), "+f"(acc_0[4]), "+f"(acc_0[5]), "+f"(acc_0[6]), "+f"(acc_0[7]), "+f"(acc_0[8]), "+f"(acc_0[9]), "+f"(acc_0[10]), "+f"(acc_0[11]), "+f"(acc_0[12]), "+f"(acc_0[13]), "+f"(acc_0[14]), "+f"(acc_0[15])
                : "l"(d_a_4), "l"(d_b_4)
                : "memory");
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 1, 1, 1, 0, 0;\n}\n"
                : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3]), "+f"(acc_0[4]), "+f"(acc_0[5]), "+f"(acc_0[6]), "+f"(acc_0[7]), "+f"(acc_0[8]), "+f"(acc_0[9]), "+f"(acc_0[10]), "+f"(acc_0[11]), "+f"(acc_0[12]), "+f"(acc_0[13]), "+f"(acc_0[14]), "+f"(acc_0[15])
                : "l"(d_a_4 + 2), "l"(d_b_4 + 2)
                : "memory");
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 1, 1, 1, 0, 0;\n}\n"
                : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3]), "+f"(acc_0[4]), "+f"(acc_0[5]), "+f"(acc_0[6]), "+f"(acc_0[7]), "+f"(acc_0[8]), "+f"(acc_0[9]), "+f"(acc_0[10]), "+f"(acc_0[11]), "+f"(acc_0[12]), "+f"(acc_0[13]), "+f"(acc_0[14]), "+f"(acc_0[15])
                : "l"(d_a_4 + 4), "l"(d_b_4 + 4)
                : "memory");
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 1, 1, 1, 0, 0;\n}\n"
                : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3]), "+f"(acc_0[4]), "+f"(acc_0[5]), "+f"(acc_0[6]), "+f"(acc_0[7]), "+f"(acc_0[8]), "+f"(acc_0[9]), "+f"(acc_0[10]), "+f"(acc_0[11]), "+f"(acc_0[12]), "+f"(acc_0[13]), "+f"(acc_0[14]), "+f"(acc_0[15])
                : "l"(d_a_4 + 6), "l"(d_b_4 + 6)
                : "memory");
            if (1 < nslices - (nstages - 1) * 4) {
                uint64_t d_b_5 = (((uint64_t)(((sb_1 + (33792u + 4096u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                uint64_t d_a_5 = (((uint64_t)(((sb_1 + (1024u + 8192u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3]), "+f"(acc_0[4]), "+f"(acc_0[5]), "+f"(acc_0[6]), "+f"(acc_0[7]), "+f"(acc_0[8]), "+f"(acc_0[9]), "+f"(acc_0[10]), "+f"(acc_0[11]), "+f"(acc_0[12]), "+f"(acc_0[13]), "+f"(acc_0[14]), "+f"(acc_0[15])
                    : "l"(d_a_5), "l"(d_b_5)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3]), "+f"(acc_0[4]), "+f"(acc_0[5]), "+f"(acc_0[6]), "+f"(acc_0[7]), "+f"(acc_0[8]), "+f"(acc_0[9]), "+f"(acc_0[10]), "+f"(acc_0[11]), "+f"(acc_0[12]), "+f"(acc_0[13]), "+f"(acc_0[14]), "+f"(acc_0[15])
                    : "l"(d_a_5 + 2), "l"(d_b_5 + 2)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3]), "+f"(acc_0[4]), "+f"(acc_0[5]), "+f"(acc_0[6]), "+f"(acc_0[7]), "+f"(acc_0[8]), "+f"(acc_0[9]), "+f"(acc_0[10]), "+f"(acc_0[11]), "+f"(acc_0[12]), "+f"(acc_0[13]), "+f"(acc_0[14]), "+f"(acc_0[15])
                    : "l"(d_a_5 + 4), "l"(d_b_5 + 4)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3]), "+f"(acc_0[4]), "+f"(acc_0[5]), "+f"(acc_0[6]), "+f"(acc_0[7]), "+f"(acc_0[8]), "+f"(acc_0[9]), "+f"(acc_0[10]), "+f"(acc_0[11]), "+f"(acc_0[12]), "+f"(acc_0[13]), "+f"(acc_0[14]), "+f"(acc_0[15])
                    : "l"(d_a_5 + 6), "l"(d_b_5 + 6)
                    : "memory");
            }
            if (2 < nslices - (nstages - 1) * 4) {
                uint64_t d_b_6 = (((uint64_t)(((sb_1 + (33792u + 8192u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                uint64_t d_a_6 = (((uint64_t)(((sb_1 + (1024u + 16384u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3]), "+f"(acc_0[4]), "+f"(acc_0[5]), "+f"(acc_0[6]), "+f"(acc_0[7]), "+f"(acc_0[8]), "+f"(acc_0[9]), "+f"(acc_0[10]), "+f"(acc_0[11]), "+f"(acc_0[12]), "+f"(acc_0[13]), "+f"(acc_0[14]), "+f"(acc_0[15])
                    : "l"(d_a_6), "l"(d_b_6)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3]), "+f"(acc_0[4]), "+f"(acc_0[5]), "+f"(acc_0[6]), "+f"(acc_0[7]), "+f"(acc_0[8]), "+f"(acc_0[9]), "+f"(acc_0[10]), "+f"(acc_0[11]), "+f"(acc_0[12]), "+f"(acc_0[13]), "+f"(acc_0[14]), "+f"(acc_0[15])
                    : "l"(d_a_6 + 2), "l"(d_b_6 + 2)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3]), "+f"(acc_0[4]), "+f"(acc_0[5]), "+f"(acc_0[6]), "+f"(acc_0[7]), "+f"(acc_0[8]), "+f"(acc_0[9]), "+f"(acc_0[10]), "+f"(acc_0[11]), "+f"(acc_0[12]), "+f"(acc_0[13]), "+f"(acc_0[14]), "+f"(acc_0[15])
                    : "l"(d_a_6 + 4), "l"(d_b_6 + 4)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3]), "+f"(acc_0[4]), "+f"(acc_0[5]), "+f"(acc_0[6]), "+f"(acc_0[7]), "+f"(acc_0[8]), "+f"(acc_0[9]), "+f"(acc_0[10]), "+f"(acc_0[11]), "+f"(acc_0[12]), "+f"(acc_0[13]), "+f"(acc_0[14]), "+f"(acc_0[15])
                    : "l"(d_a_6 + 6), "l"(d_b_6 + 6)
                    : "memory");
            }
            if (3 < nslices - (nstages - 1) * 4) {
                uint64_t d_b_7 = (((uint64_t)(((sb_1 + (33792u + 12288u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                uint64_t d_a_7 = (((uint64_t)(((sb_1 + (1024u + 24576u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3]), "+f"(acc_0[4]), "+f"(acc_0[5]), "+f"(acc_0[6]), "+f"(acc_0[7]), "+f"(acc_0[8]), "+f"(acc_0[9]), "+f"(acc_0[10]), "+f"(acc_0[11]), "+f"(acc_0[12]), "+f"(acc_0[13]), "+f"(acc_0[14]), "+f"(acc_0[15])
                    : "l"(d_a_7), "l"(d_b_7)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3]), "+f"(acc_0[4]), "+f"(acc_0[5]), "+f"(acc_0[6]), "+f"(acc_0[7]), "+f"(acc_0[8]), "+f"(acc_0[9]), "+f"(acc_0[10]), "+f"(acc_0[11]), "+f"(acc_0[12]), "+f"(acc_0[13]), "+f"(acc_0[14]), "+f"(acc_0[15])
                    : "l"(d_a_7 + 2), "l"(d_b_7 + 2)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3]), "+f"(acc_0[4]), "+f"(acc_0[5]), "+f"(acc_0[6]), "+f"(acc_0[7]), "+f"(acc_0[8]), "+f"(acc_0[9]), "+f"(acc_0[10]), "+f"(acc_0[11]), "+f"(acc_0[12]), "+f"(acc_0[13]), "+f"(acc_0[14]), "+f"(acc_0[15])
                    : "l"(d_a_7 + 4), "l"(d_b_7 + 4)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n32k16.f32.bf16.bf16 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, %16, %17, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3]), "+f"(acc_0[4]), "+f"(acc_0[5]), "+f"(acc_0[6]), "+f"(acc_0[7]), "+f"(acc_0[8]), "+f"(acc_0[9]), "+f"(acc_0[10]), "+f"(acc_0[11]), "+f"(acc_0[12]), "+f"(acc_0[13]), "+f"(acc_0[14]), "+f"(acc_0[15])
                    : "l"(d_a_7 + 6), "l"(d_b_7 + 6)
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
            asm volatile("" : "+f"(acc_0[8]) :: "memory");
            asm volatile("" : "+f"(acc_0[9]) :: "memory");
            asm volatile("" : "+f"(acc_0[10]) :: "memory");
            asm volatile("" : "+f"(acc_0[11]) :: "memory");
            asm volatile("" : "+f"(acc_0[12]) :: "memory");
            asm volatile("" : "+f"(acc_0[13]) :: "memory");
            asm volatile("" : "+f"(acc_0[14]) :: "memory");
            asm volatile("" : "+f"(acc_0[15]) :: "memory");
            if (warp == 0) {
                red[g * 8 + 2 * t] = acc_0[0];
                red[g * 8 + 2 * t + 1] = acc_0[1];
                red[(g + 8) * 8 + 2 * t] = acc_0[2];
                red[(g + 8) * 8 + 2 * t + 1] = acc_0[3];
            }
            if (warp == 1) {
                red[(16 + g) * 8 + 2 * t] = acc_0[4];
                red[(16 + g) * 8 + 2 * t + 1] = acc_0[5];
                red[(16 + g + 8) * 8 + 2 * t] = acc_0[6];
                red[(16 + g + 8) * 8 + 2 * t + 1] = acc_0[7];
            }
            if (warp == 2) {
                red[(32 + g) * 8 + 2 * t] = acc_0[8];
                red[(32 + g) * 8 + 2 * t + 1] = acc_0[9];
                red[(32 + g + 8) * 8 + 2 * t] = acc_0[10];
                red[(32 + g + 8) * 8 + 2 * t + 1] = acc_0[11];
            }
            if (warp == 3) {
                red[(48 + g) * 8 + 2 * t] = acc_0[12];
                red[(48 + g) * 8 + 2 * t + 1] = acc_0[13];
                red[(48 + g + 8) * 8 + 2 * t] = acc_0[14];
                red[(48 + g + 8) * 8 + 2 * t + 1] = acc_0[15];
            }
        }
    }
    // ---- Role: load ----
    if (warp == 4) {
        { // load_main
            int pre = ((4) < (nstages) ? (4) : (nstages));
            if (elect_sync()) {
                #pragma unroll 1
                for (int st_1 = 0; st_1 < pre; st_1++) {
                    int nv_w = ((4) < (nslices - st_1 * 4) ? (4) : (nslices - st_1 * 4));
                    mbarrier_arrive_expect_tx(full_addr + ((int)((unsigned int)st_1 % 4)) * 8, nv_w * 12288);
                    int sl_0 = st_1 * 4;
                    int kb_0 = (sl_0 / 4 * 16 + sl_0 % 4) * 64;
                    unsigned int dst_0 = sw_addr + (unsigned int)(st_1 % 4 * 49152);
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3}], [%4], %5;"
                        :: "r"(dst_0 + 0), "l"((&W)), "r"(kb_0), "r"(m0),
                           "r"(full_addr + ((int)((unsigned int)st_1 % 4)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    int sl_1 = st_1 * 4;
                    int kb_1 = (sl_1 / 4 * 16 + sl_1 % 4) * 64;
                    unsigned int dst_1 = sw_addr + (unsigned int)(st_1 % 4 * 49152);
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3}], [%4], %5;"
                        :: "r"(dst_1 + 2048), "l"((&W)), "r"(kb_1 + 256), "r"(m0),
                           "r"(full_addr + ((int)((unsigned int)st_1 % 4)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    int sl_2 = st_1 * 4;
                    int kb_2 = (sl_2 / 4 * 16 + sl_2 % 4) * 64;
                    unsigned int dst_2 = sw_addr + (unsigned int)(st_1 % 4 * 49152);
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3}], [%4], %5;"
                        :: "r"(dst_2 + 4096), "l"((&W)), "r"(kb_2 + 512), "r"(m0),
                           "r"(full_addr + ((int)((unsigned int)st_1 % 4)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    int sl_3 = st_1 * 4;
                    int kb_3 = (sl_3 / 4 * 16 + sl_3 % 4) * 64;
                    unsigned int dst_3 = sw_addr + (unsigned int)(st_1 % 4 * 49152);
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3}], [%4], %5;"
                        :: "r"(dst_3 + 6144), "l"((&W)), "r"(kb_3 + 768), "r"(m0),
                           "r"(full_addr + ((int)((unsigned int)st_1 % 4)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    if (1 < nv_w) {
                        int sl_4 = st_1 * 4 + 1;
                        int kb_4 = (sl_4 / 4 * 16 + sl_4 % 4) * 64;
                        unsigned int dst_4 = sw_addr + (unsigned int)(st_1 % 4 * 49152 + 8192);
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3}], [%4], %5;"
                            :: "r"(dst_4 + 0), "l"((&W)), "r"(kb_4), "r"(m0),
                               "r"(full_addr + ((int)((unsigned int)st_1 % 4)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    }
                    if (1 < nv_w) {
                        int sl_5 = st_1 * 4 + 1;
                        int kb_5 = (sl_5 / 4 * 16 + sl_5 % 4) * 64;
                        unsigned int dst_5 = sw_addr + (unsigned int)(st_1 % 4 * 49152 + 8192);
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3}], [%4], %5;"
                            :: "r"(dst_5 + 2048), "l"((&W)), "r"(kb_5 + 256), "r"(m0),
                               "r"(full_addr + ((int)((unsigned int)st_1 % 4)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    }
                    if (1 < nv_w) {
                        int sl_6 = st_1 * 4 + 1;
                        int kb_6 = (sl_6 / 4 * 16 + sl_6 % 4) * 64;
                        unsigned int dst_6 = sw_addr + (unsigned int)(st_1 % 4 * 49152 + 8192);
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3}], [%4], %5;"
                            :: "r"(dst_6 + 4096), "l"((&W)), "r"(kb_6 + 512), "r"(m0),
                               "r"(full_addr + ((int)((unsigned int)st_1 % 4)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    }
                    if (1 < nv_w) {
                        int sl_7 = st_1 * 4 + 1;
                        int kb_7 = (sl_7 / 4 * 16 + sl_7 % 4) * 64;
                        unsigned int dst_7 = sw_addr + (unsigned int)(st_1 % 4 * 49152 + 8192);
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3}], [%4], %5;"
                            :: "r"(dst_7 + 6144), "l"((&W)), "r"(kb_7 + 768), "r"(m0),
                               "r"(full_addr + ((int)((unsigned int)st_1 % 4)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    }
                    if (2 < nv_w) {
                        int sl_8 = st_1 * 4 + 2;
                        int kb_8 = (sl_8 / 4 * 16 + sl_8 % 4) * 64;
                        unsigned int dst_8 = sw_addr + (unsigned int)(st_1 % 4 * 49152 + 16384);
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3}], [%4], %5;"
                            :: "r"(dst_8 + 0), "l"((&W)), "r"(kb_8), "r"(m0),
                               "r"(full_addr + ((int)((unsigned int)st_1 % 4)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    }
                    if (2 < nv_w) {
                        int sl_9 = st_1 * 4 + 2;
                        int kb_9 = (sl_9 / 4 * 16 + sl_9 % 4) * 64;
                        unsigned int dst_9 = sw_addr + (unsigned int)(st_1 % 4 * 49152 + 16384);
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3}], [%4], %5;"
                            :: "r"(dst_9 + 2048), "l"((&W)), "r"(kb_9 + 256), "r"(m0),
                               "r"(full_addr + ((int)((unsigned int)st_1 % 4)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    }
                    if (2 < nv_w) {
                        int sl_10 = st_1 * 4 + 2;
                        int kb_10 = (sl_10 / 4 * 16 + sl_10 % 4) * 64;
                        unsigned int dst_10 = sw_addr + (unsigned int)(st_1 % 4 * 49152 + 16384);
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3}], [%4], %5;"
                            :: "r"(dst_10 + 4096), "l"((&W)), "r"(kb_10 + 512), "r"(m0),
                               "r"(full_addr + ((int)((unsigned int)st_1 % 4)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    }
                    if (2 < nv_w) {
                        int sl_11 = st_1 * 4 + 2;
                        int kb_11 = (sl_11 / 4 * 16 + sl_11 % 4) * 64;
                        unsigned int dst_11 = sw_addr + (unsigned int)(st_1 % 4 * 49152 + 16384);
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3}], [%4], %5;"
                            :: "r"(dst_11 + 6144), "l"((&W)), "r"(kb_11 + 768), "r"(m0),
                               "r"(full_addr + ((int)((unsigned int)st_1 % 4)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    }
                    if (3 < nv_w) {
                        int sl_12 = st_1 * 4 + 3;
                        int kb_12 = (sl_12 / 4 * 16 + sl_12 % 4) * 64;
                        unsigned int dst_12 = sw_addr + (unsigned int)(st_1 % 4 * 49152 + 24576);
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3}], [%4], %5;"
                            :: "r"(dst_12 + 0), "l"((&W)), "r"(kb_12), "r"(m0),
                               "r"(full_addr + ((int)((unsigned int)st_1 % 4)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    }
                    if (3 < nv_w) {
                        int sl_13 = st_1 * 4 + 3;
                        int kb_13 = (sl_13 / 4 * 16 + sl_13 % 4) * 64;
                        unsigned int dst_13 = sw_addr + (unsigned int)(st_1 % 4 * 49152 + 24576);
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3}], [%4], %5;"
                            :: "r"(dst_13 + 2048), "l"((&W)), "r"(kb_13 + 256), "r"(m0),
                               "r"(full_addr + ((int)((unsigned int)st_1 % 4)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    }
                    if (3 < nv_w) {
                        int sl_14 = st_1 * 4 + 3;
                        int kb_14 = (sl_14 / 4 * 16 + sl_14 % 4) * 64;
                        unsigned int dst_14 = sw_addr + (unsigned int)(st_1 % 4 * 49152 + 24576);
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3}], [%4], %5;"
                            :: "r"(dst_14 + 4096), "l"((&W)), "r"(kb_14 + 512), "r"(m0),
                               "r"(full_addr + ((int)((unsigned int)st_1 % 4)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    }
                    if (3 < nv_w) {
                        int sl_15 = st_1 * 4 + 3;
                        int kb_15 = (sl_15 / 4 * 16 + sl_15 % 4) * 64;
                        unsigned int dst_15 = sw_addr + (unsigned int)(st_1 % 4 * 49152 + 24576);
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3}], [%4], %5;"
                            :: "r"(dst_15 + 6144), "l"((&W)), "r"(kb_15 + 768), "r"(m0),
                               "r"(full_addr + ((int)((unsigned int)st_1 % 4)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    }
                }
            }
            asm volatile("griddepcontrol.wait;" ::: "memory");
            if (lane == 0) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
            if (elect_sync()) {
                #pragma unroll 1
                for (int st_2 = 0; st_2 < pre; st_2++) {
                    int slx_0 = st_2 * 4;
                    int kbx_0 = (slx_0 / 4 * 16 + slx_0 % 4) * 64;
                    unsigned int dstx_0 = sx_addr + (unsigned int)(st_2 % 4 * 49152);
                    tma_2d_gmem2smem(dstx_0 + 0, (&X), kbx_0, n0, full_addr + ((int)((unsigned int)st_2 % 4)) * 8);
                    int slx_1 = st_2 * 4;
                    int kbx_1 = (slx_1 / 4 * 16 + slx_1 % 4) * 64;
                    unsigned int dstx_1 = sx_addr + (unsigned int)(st_2 % 4 * 49152);
                    tma_2d_gmem2smem(dstx_1 + 1024, (&X), kbx_1 + 256, n0, full_addr + ((int)((unsigned int)st_2 % 4)) * 8);
                    int slx_2 = st_2 * 4;
                    int kbx_2 = (slx_2 / 4 * 16 + slx_2 % 4) * 64;
                    unsigned int dstx_2 = sx_addr + (unsigned int)(st_2 % 4 * 49152);
                    tma_2d_gmem2smem(dstx_2 + 2048, (&X), kbx_2 + 512, n0, full_addr + ((int)((unsigned int)st_2 % 4)) * 8);
                    int slx_3 = st_2 * 4;
                    int kbx_3 = (slx_3 / 4 * 16 + slx_3 % 4) * 64;
                    unsigned int dstx_3 = sx_addr + (unsigned int)(st_2 % 4 * 49152);
                    tma_2d_gmem2smem(dstx_3 + 3072, (&X), kbx_3 + 768, n0, full_addr + ((int)((unsigned int)st_2 % 4)) * 8);
                    int _min_0_1 = ((4) < (nslices - st_2 * 4) ? (4) : (nslices - st_2 * 4));
                    if (1 < _min_0_1) {
                        int slx_4 = st_2 * 4 + 1;
                        int kbx_4 = (slx_4 / 4 * 16 + slx_4 % 4) * 64;
                        unsigned int dstx_4 = sx_addr + (unsigned int)(st_2 % 4 * 49152 + 4096);
                        tma_2d_gmem2smem(dstx_4 + 0, (&X), kbx_4, n0, full_addr + ((int)((unsigned int)st_2 % 4)) * 8);
                    }
                    int _min_1 = ((4) < (nslices - st_2 * 4) ? (4) : (nslices - st_2 * 4));
                    if (1 < _min_1) {
                        int slx_5 = st_2 * 4 + 1;
                        int kbx_5 = (slx_5 / 4 * 16 + slx_5 % 4) * 64;
                        unsigned int dstx_5 = sx_addr + (unsigned int)(st_2 % 4 * 49152 + 4096);
                        tma_2d_gmem2smem(dstx_5 + 1024, (&X), kbx_5 + 256, n0, full_addr + ((int)((unsigned int)st_2 % 4)) * 8);
                    }
                    int _min_2 = ((4) < (nslices - st_2 * 4) ? (4) : (nslices - st_2 * 4));
                    if (1 < _min_2) {
                        int slx_6 = st_2 * 4 + 1;
                        int kbx_6 = (slx_6 / 4 * 16 + slx_6 % 4) * 64;
                        unsigned int dstx_6 = sx_addr + (unsigned int)(st_2 % 4 * 49152 + 4096);
                        tma_2d_gmem2smem(dstx_6 + 2048, (&X), kbx_6 + 512, n0, full_addr + ((int)((unsigned int)st_2 % 4)) * 8);
                    }
                    int _min_3 = ((4) < (nslices - st_2 * 4) ? (4) : (nslices - st_2 * 4));
                    if (1 < _min_3) {
                        int slx_7 = st_2 * 4 + 1;
                        int kbx_7 = (slx_7 / 4 * 16 + slx_7 % 4) * 64;
                        unsigned int dstx_7 = sx_addr + (unsigned int)(st_2 % 4 * 49152 + 4096);
                        tma_2d_gmem2smem(dstx_7 + 3072, (&X), kbx_7 + 768, n0, full_addr + ((int)((unsigned int)st_2 % 4)) * 8);
                    }
                    int _min_4 = ((4) < (nslices - st_2 * 4) ? (4) : (nslices - st_2 * 4));
                    if (2 < _min_4) {
                        int slx_8 = st_2 * 4 + 2;
                        int kbx_8 = (slx_8 / 4 * 16 + slx_8 % 4) * 64;
                        unsigned int dstx_8 = sx_addr + (unsigned int)(st_2 % 4 * 49152 + 8192);
                        tma_2d_gmem2smem(dstx_8 + 0, (&X), kbx_8, n0, full_addr + ((int)((unsigned int)st_2 % 4)) * 8);
                    }
                    int _min_5 = ((4) < (nslices - st_2 * 4) ? (4) : (nslices - st_2 * 4));
                    if (2 < _min_5) {
                        int slx_9 = st_2 * 4 + 2;
                        int kbx_9 = (slx_9 / 4 * 16 + slx_9 % 4) * 64;
                        unsigned int dstx_9 = sx_addr + (unsigned int)(st_2 % 4 * 49152 + 8192);
                        tma_2d_gmem2smem(dstx_9 + 1024, (&X), kbx_9 + 256, n0, full_addr + ((int)((unsigned int)st_2 % 4)) * 8);
                    }
                    int _min_6 = ((4) < (nslices - st_2 * 4) ? (4) : (nslices - st_2 * 4));
                    if (2 < _min_6) {
                        int slx_10 = st_2 * 4 + 2;
                        int kbx_10 = (slx_10 / 4 * 16 + slx_10 % 4) * 64;
                        unsigned int dstx_10 = sx_addr + (unsigned int)(st_2 % 4 * 49152 + 8192);
                        tma_2d_gmem2smem(dstx_10 + 2048, (&X), kbx_10 + 512, n0, full_addr + ((int)((unsigned int)st_2 % 4)) * 8);
                    }
                    int _min_7 = ((4) < (nslices - st_2 * 4) ? (4) : (nslices - st_2 * 4));
                    if (2 < _min_7) {
                        int slx_11 = st_2 * 4 + 2;
                        int kbx_11 = (slx_11 / 4 * 16 + slx_11 % 4) * 64;
                        unsigned int dstx_11 = sx_addr + (unsigned int)(st_2 % 4 * 49152 + 8192);
                        tma_2d_gmem2smem(dstx_11 + 3072, (&X), kbx_11 + 768, n0, full_addr + ((int)((unsigned int)st_2 % 4)) * 8);
                    }
                    int _min_8 = ((4) < (nslices - st_2 * 4) ? (4) : (nslices - st_2 * 4));
                    if (3 < _min_8) {
                        int slx_12 = st_2 * 4 + 3;
                        int kbx_12 = (slx_12 / 4 * 16 + slx_12 % 4) * 64;
                        unsigned int dstx_12 = sx_addr + (unsigned int)(st_2 % 4 * 49152 + 12288);
                        tma_2d_gmem2smem(dstx_12 + 0, (&X), kbx_12, n0, full_addr + ((int)((unsigned int)st_2 % 4)) * 8);
                    }
                    int _min_9 = ((4) < (nslices - st_2 * 4) ? (4) : (nslices - st_2 * 4));
                    if (3 < _min_9) {
                        int slx_13 = st_2 * 4 + 3;
                        int kbx_13 = (slx_13 / 4 * 16 + slx_13 % 4) * 64;
                        unsigned int dstx_13 = sx_addr + (unsigned int)(st_2 % 4 * 49152 + 12288);
                        tma_2d_gmem2smem(dstx_13 + 1024, (&X), kbx_13 + 256, n0, full_addr + ((int)((unsigned int)st_2 % 4)) * 8);
                    }
                    int _min_10 = ((4) < (nslices - st_2 * 4) ? (4) : (nslices - st_2 * 4));
                    if (3 < _min_10) {
                        int slx_14 = st_2 * 4 + 3;
                        int kbx_14 = (slx_14 / 4 * 16 + slx_14 % 4) * 64;
                        unsigned int dstx_14 = sx_addr + (unsigned int)(st_2 % 4 * 49152 + 12288);
                        tma_2d_gmem2smem(dstx_14 + 2048, (&X), kbx_14 + 512, n0, full_addr + ((int)((unsigned int)st_2 % 4)) * 8);
                    }
                    int _min_11 = ((4) < (nslices - st_2 * 4) ? (4) : (nslices - st_2 * 4));
                    if (3 < _min_11) {
                        int slx_15 = st_2 * 4 + 3;
                        int kbx_15 = (slx_15 / 4 * 16 + slx_15 % 4) * 64;
                        unsigned int dstx_15 = sx_addr + (unsigned int)(st_2 % 4 * 49152 + 12288);
                        tma_2d_gmem2smem(dstx_15 + 3072, (&X), kbx_15 + 768, n0, full_addr + ((int)((unsigned int)st_2 % 4)) * 8);
                    }
                }
                #pragma unroll 1
                for (int st_3 = pre; st_3 < nstages; st_3++) {
                    int nv_s = ((4) < (nslices - st_3 * 4) ? (4) : (nslices - st_3 * 4));
                    int _stage_empty = (int)((unsigned int)st_3 % 4);
                    int _parity_empty = (int)((unsigned int)st_3 / 4 % 2) ^ 1;
                    mbarrier_wait(empty_addr + (_stage_empty) * 8, _parity_empty);
                    mbarrier_arrive_expect_tx(full_addr + ((int)((unsigned int)st_3 % 4)) * 8, nv_s * 12288);
                    int sl_16 = st_3 * 4;
                    int kb_16 = (sl_16 / 4 * 16 + sl_16 % 4) * 64;
                    unsigned int dst_16 = sw_addr + (unsigned int)(st_3 % 4 * 49152);
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3}], [%4], %5;"
                        :: "r"(dst_16 + 0), "l"((&W)), "r"(kb_16), "r"(m0),
                           "r"(full_addr + ((int)((unsigned int)st_3 % 4)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    int sl_17 = st_3 * 4;
                    int kb_17 = (sl_17 / 4 * 16 + sl_17 % 4) * 64;
                    unsigned int dst_17 = sw_addr + (unsigned int)(st_3 % 4 * 49152);
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3}], [%4], %5;"
                        :: "r"(dst_17 + 2048), "l"((&W)), "r"(kb_17 + 256), "r"(m0),
                           "r"(full_addr + ((int)((unsigned int)st_3 % 4)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    int sl_18 = st_3 * 4;
                    int kb_18 = (sl_18 / 4 * 16 + sl_18 % 4) * 64;
                    unsigned int dst_18 = sw_addr + (unsigned int)(st_3 % 4 * 49152);
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3}], [%4], %5;"
                        :: "r"(dst_18 + 4096), "l"((&W)), "r"(kb_18 + 512), "r"(m0),
                           "r"(full_addr + ((int)((unsigned int)st_3 % 4)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    int sl_19 = st_3 * 4;
                    int kb_19 = (sl_19 / 4 * 16 + sl_19 % 4) * 64;
                    unsigned int dst_19 = sw_addr + (unsigned int)(st_3 % 4 * 49152);
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3}], [%4], %5;"
                        :: "r"(dst_19 + 6144), "l"((&W)), "r"(kb_19 + 768), "r"(m0),
                           "r"(full_addr + ((int)((unsigned int)st_3 % 4)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    if (1 < nv_s) {
                        int sl_20 = st_3 * 4 + 1;
                        int kb_20 = (sl_20 / 4 * 16 + sl_20 % 4) * 64;
                        unsigned int dst_20 = sw_addr + (unsigned int)(st_3 % 4 * 49152 + 8192);
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3}], [%4], %5;"
                            :: "r"(dst_20 + 0), "l"((&W)), "r"(kb_20), "r"(m0),
                               "r"(full_addr + ((int)((unsigned int)st_3 % 4)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    }
                    if (1 < nv_s) {
                        int sl_21 = st_3 * 4 + 1;
                        int kb_21 = (sl_21 / 4 * 16 + sl_21 % 4) * 64;
                        unsigned int dst_21 = sw_addr + (unsigned int)(st_3 % 4 * 49152 + 8192);
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3}], [%4], %5;"
                            :: "r"(dst_21 + 2048), "l"((&W)), "r"(kb_21 + 256), "r"(m0),
                               "r"(full_addr + ((int)((unsigned int)st_3 % 4)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    }
                    if (1 < nv_s) {
                        int sl_22 = st_3 * 4 + 1;
                        int kb_22 = (sl_22 / 4 * 16 + sl_22 % 4) * 64;
                        unsigned int dst_22 = sw_addr + (unsigned int)(st_3 % 4 * 49152 + 8192);
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3}], [%4], %5;"
                            :: "r"(dst_22 + 4096), "l"((&W)), "r"(kb_22 + 512), "r"(m0),
                               "r"(full_addr + ((int)((unsigned int)st_3 % 4)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    }
                    if (1 < nv_s) {
                        int sl_23 = st_3 * 4 + 1;
                        int kb_23 = (sl_23 / 4 * 16 + sl_23 % 4) * 64;
                        unsigned int dst_23 = sw_addr + (unsigned int)(st_3 % 4 * 49152 + 8192);
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3}], [%4], %5;"
                            :: "r"(dst_23 + 6144), "l"((&W)), "r"(kb_23 + 768), "r"(m0),
                               "r"(full_addr + ((int)((unsigned int)st_3 % 4)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    }
                    if (2 < nv_s) {
                        int sl_24 = st_3 * 4 + 2;
                        int kb_24 = (sl_24 / 4 * 16 + sl_24 % 4) * 64;
                        unsigned int dst_24 = sw_addr + (unsigned int)(st_3 % 4 * 49152 + 16384);
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3}], [%4], %5;"
                            :: "r"(dst_24 + 0), "l"((&W)), "r"(kb_24), "r"(m0),
                               "r"(full_addr + ((int)((unsigned int)st_3 % 4)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    }
                    if (2 < nv_s) {
                        int sl_25 = st_3 * 4 + 2;
                        int kb_25 = (sl_25 / 4 * 16 + sl_25 % 4) * 64;
                        unsigned int dst_25 = sw_addr + (unsigned int)(st_3 % 4 * 49152 + 16384);
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3}], [%4], %5;"
                            :: "r"(dst_25 + 2048), "l"((&W)), "r"(kb_25 + 256), "r"(m0),
                               "r"(full_addr + ((int)((unsigned int)st_3 % 4)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    }
                    if (2 < nv_s) {
                        int sl_26 = st_3 * 4 + 2;
                        int kb_26 = (sl_26 / 4 * 16 + sl_26 % 4) * 64;
                        unsigned int dst_26 = sw_addr + (unsigned int)(st_3 % 4 * 49152 + 16384);
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3}], [%4], %5;"
                            :: "r"(dst_26 + 4096), "l"((&W)), "r"(kb_26 + 512), "r"(m0),
                               "r"(full_addr + ((int)((unsigned int)st_3 % 4)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    }
                    if (2 < nv_s) {
                        int sl_27 = st_3 * 4 + 2;
                        int kb_27 = (sl_27 / 4 * 16 + sl_27 % 4) * 64;
                        unsigned int dst_27 = sw_addr + (unsigned int)(st_3 % 4 * 49152 + 16384);
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3}], [%4], %5;"
                            :: "r"(dst_27 + 6144), "l"((&W)), "r"(kb_27 + 768), "r"(m0),
                               "r"(full_addr + ((int)((unsigned int)st_3 % 4)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    }
                    if (3 < nv_s) {
                        int sl_28 = st_3 * 4 + 3;
                        int kb_28 = (sl_28 / 4 * 16 + sl_28 % 4) * 64;
                        unsigned int dst_28 = sw_addr + (unsigned int)(st_3 % 4 * 49152 + 24576);
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3}], [%4], %5;"
                            :: "r"(dst_28 + 0), "l"((&W)), "r"(kb_28), "r"(m0),
                               "r"(full_addr + ((int)((unsigned int)st_3 % 4)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    }
                    if (3 < nv_s) {
                        int sl_29 = st_3 * 4 + 3;
                        int kb_29 = (sl_29 / 4 * 16 + sl_29 % 4) * 64;
                        unsigned int dst_29 = sw_addr + (unsigned int)(st_3 % 4 * 49152 + 24576);
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3}], [%4], %5;"
                            :: "r"(dst_29 + 2048), "l"((&W)), "r"(kb_29 + 256), "r"(m0),
                               "r"(full_addr + ((int)((unsigned int)st_3 % 4)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    }
                    if (3 < nv_s) {
                        int sl_30 = st_3 * 4 + 3;
                        int kb_30 = (sl_30 / 4 * 16 + sl_30 % 4) * 64;
                        unsigned int dst_30 = sw_addr + (unsigned int)(st_3 % 4 * 49152 + 24576);
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3}], [%4], %5;"
                            :: "r"(dst_30 + 4096), "l"((&W)), "r"(kb_30 + 512), "r"(m0),
                               "r"(full_addr + ((int)((unsigned int)st_3 % 4)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    }
                    if (3 < nv_s) {
                        int sl_31 = st_3 * 4 + 3;
                        int kb_31 = (sl_31 / 4 * 16 + sl_31 % 4) * 64;
                        unsigned int dst_31 = sw_addr + (unsigned int)(st_3 % 4 * 49152 + 24576);
                        asm volatile(
                            "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3}], [%4], %5;"
                            :: "r"(dst_31 + 6144), "l"((&W)), "r"(kb_31 + 768), "r"(m0),
                               "r"(full_addr + ((int)((unsigned int)st_3 % 4)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    }
                    int slx_16 = st_3 * 4;
                    int kbx_16 = (slx_16 / 4 * 16 + slx_16 % 4) * 64;
                    unsigned int dstx_16 = sx_addr + (unsigned int)(st_3 % 4 * 49152);
                    tma_2d_gmem2smem(dstx_16 + 0, (&X), kbx_16, n0, full_addr + ((int)((unsigned int)st_3 % 4)) * 8);
                    int slx_17 = st_3 * 4;
                    int kbx_17 = (slx_17 / 4 * 16 + slx_17 % 4) * 64;
                    unsigned int dstx_17 = sx_addr + (unsigned int)(st_3 % 4 * 49152);
                    tma_2d_gmem2smem(dstx_17 + 1024, (&X), kbx_17 + 256, n0, full_addr + ((int)((unsigned int)st_3 % 4)) * 8);
                    int slx_18 = st_3 * 4;
                    int kbx_18 = (slx_18 / 4 * 16 + slx_18 % 4) * 64;
                    unsigned int dstx_18 = sx_addr + (unsigned int)(st_3 % 4 * 49152);
                    tma_2d_gmem2smem(dstx_18 + 2048, (&X), kbx_18 + 512, n0, full_addr + ((int)((unsigned int)st_3 % 4)) * 8);
                    int slx_19 = st_3 * 4;
                    int kbx_19 = (slx_19 / 4 * 16 + slx_19 % 4) * 64;
                    unsigned int dstx_19 = sx_addr + (unsigned int)(st_3 % 4 * 49152);
                    tma_2d_gmem2smem(dstx_19 + 3072, (&X), kbx_19 + 768, n0, full_addr + ((int)((unsigned int)st_3 % 4)) * 8);
                    if (1 < nv_s) {
                        int slx_20 = st_3 * 4 + 1;
                        int kbx_20 = (slx_20 / 4 * 16 + slx_20 % 4) * 64;
                        unsigned int dstx_20 = sx_addr + (unsigned int)(st_3 % 4 * 49152 + 4096);
                        tma_2d_gmem2smem(dstx_20 + 0, (&X), kbx_20, n0, full_addr + ((int)((unsigned int)st_3 % 4)) * 8);
                    }
                    if (1 < nv_s) {
                        int slx_21 = st_3 * 4 + 1;
                        int kbx_21 = (slx_21 / 4 * 16 + slx_21 % 4) * 64;
                        unsigned int dstx_21 = sx_addr + (unsigned int)(st_3 % 4 * 49152 + 4096);
                        tma_2d_gmem2smem(dstx_21 + 1024, (&X), kbx_21 + 256, n0, full_addr + ((int)((unsigned int)st_3 % 4)) * 8);
                    }
                    if (1 < nv_s) {
                        int slx_22 = st_3 * 4 + 1;
                        int kbx_22 = (slx_22 / 4 * 16 + slx_22 % 4) * 64;
                        unsigned int dstx_22 = sx_addr + (unsigned int)(st_3 % 4 * 49152 + 4096);
                        tma_2d_gmem2smem(dstx_22 + 2048, (&X), kbx_22 + 512, n0, full_addr + ((int)((unsigned int)st_3 % 4)) * 8);
                    }
                    if (1 < nv_s) {
                        int slx_23 = st_3 * 4 + 1;
                        int kbx_23 = (slx_23 / 4 * 16 + slx_23 % 4) * 64;
                        unsigned int dstx_23 = sx_addr + (unsigned int)(st_3 % 4 * 49152 + 4096);
                        tma_2d_gmem2smem(dstx_23 + 3072, (&X), kbx_23 + 768, n0, full_addr + ((int)((unsigned int)st_3 % 4)) * 8);
                    }
                    if (2 < nv_s) {
                        int slx_24 = st_3 * 4 + 2;
                        int kbx_24 = (slx_24 / 4 * 16 + slx_24 % 4) * 64;
                        unsigned int dstx_24 = sx_addr + (unsigned int)(st_3 % 4 * 49152 + 8192);
                        tma_2d_gmem2smem(dstx_24 + 0, (&X), kbx_24, n0, full_addr + ((int)((unsigned int)st_3 % 4)) * 8);
                    }
                    if (2 < nv_s) {
                        int slx_25 = st_3 * 4 + 2;
                        int kbx_25 = (slx_25 / 4 * 16 + slx_25 % 4) * 64;
                        unsigned int dstx_25 = sx_addr + (unsigned int)(st_3 % 4 * 49152 + 8192);
                        tma_2d_gmem2smem(dstx_25 + 1024, (&X), kbx_25 + 256, n0, full_addr + ((int)((unsigned int)st_3 % 4)) * 8);
                    }
                    if (2 < nv_s) {
                        int slx_26 = st_3 * 4 + 2;
                        int kbx_26 = (slx_26 / 4 * 16 + slx_26 % 4) * 64;
                        unsigned int dstx_26 = sx_addr + (unsigned int)(st_3 % 4 * 49152 + 8192);
                        tma_2d_gmem2smem(dstx_26 + 2048, (&X), kbx_26 + 512, n0, full_addr + ((int)((unsigned int)st_3 % 4)) * 8);
                    }
                    if (2 < nv_s) {
                        int slx_27 = st_3 * 4 + 2;
                        int kbx_27 = (slx_27 / 4 * 16 + slx_27 % 4) * 64;
                        unsigned int dstx_27 = sx_addr + (unsigned int)(st_3 % 4 * 49152 + 8192);
                        tma_2d_gmem2smem(dstx_27 + 3072, (&X), kbx_27 + 768, n0, full_addr + ((int)((unsigned int)st_3 % 4)) * 8);
                    }
                    if (3 < nv_s) {
                        int slx_28 = st_3 * 4 + 3;
                        int kbx_28 = (slx_28 / 4 * 16 + slx_28 % 4) * 64;
                        unsigned int dstx_28 = sx_addr + (unsigned int)(st_3 % 4 * 49152 + 12288);
                        tma_2d_gmem2smem(dstx_28 + 0, (&X), kbx_28, n0, full_addr + ((int)((unsigned int)st_3 % 4)) * 8);
                    }
                    if (3 < nv_s) {
                        int slx_29 = st_3 * 4 + 3;
                        int kbx_29 = (slx_29 / 4 * 16 + slx_29 % 4) * 64;
                        unsigned int dstx_29 = sx_addr + (unsigned int)(st_3 % 4 * 49152 + 12288);
                        tma_2d_gmem2smem(dstx_29 + 1024, (&X), kbx_29 + 256, n0, full_addr + ((int)((unsigned int)st_3 % 4)) * 8);
                    }
                    if (3 < nv_s) {
                        int slx_30 = st_3 * 4 + 3;
                        int kbx_30 = (slx_30 / 4 * 16 + slx_30 % 4) * 64;
                        unsigned int dstx_30 = sx_addr + (unsigned int)(st_3 % 4 * 49152 + 12288);
                        tma_2d_gmem2smem(dstx_30 + 2048, (&X), kbx_30 + 512, n0, full_addr + ((int)((unsigned int)st_3 % 4)) * 8);
                    }
                    if (3 < nv_s) {
                        int slx_31 = st_3 * 4 + 3;
                        int kbx_31 = (slx_31 / 4 * 16 + slx_31 % 4) * 64;
                        unsigned int dstx_31 = sx_addr + (unsigned int)(st_3 % 4 * 49152 + 12288);
                        tma_2d_gmem2smem(dstx_31 + 3072, (&X), kbx_31 + 768, n0, full_addr + ((int)((unsigned int)st_3 % 4)) * 8);
                    }
                }
            }
        }
    }

    // Kernel teardown ops
    __syncthreads();
    int el_0 = tid;
    if (tid < 128 && el_0 < 128) {
        int col_0 = el_0 / 16;
        int row_0 = el_0 % 16;
        float c0_0 = red[row_0 * 8 + col_0];
        float s01_0 = c0_0 + red[(16 + row_0) * 8 + col_0];
        float s012_0 = s01_0 + red[(32 + row_0) * 8 + col_0];
        float tot_0 = s012_0 + red[(48 + row_0) * 8 + col_0];
        if (m0 + row_0 < M && n0 + col_0 < N) {
            *(reinterpret_cast<__nv_bfloat16*>(out + (m0 + row_0 + (n0 + col_0) * M)) + (0)) = (__nv_bfloat16)(tot_0 + bv_0);
        }
    }

    // Cleanup
}

} // extern "C"
