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
#define NUM_PIPE_STAGES 8
#define SMEM_SW_OFF 1024
#define SMEM_SW_STAGE_BYTES 8192
#define SMEM_SW_STRIDE 12288
#define SMEM_SX_OFF 9216
#define SMEM_SX_STAGE_BYTES 4096
#define SMEM_SX_STRIDE 12288
#define SMEM_RED_OFF 99328
#define SMEM_RED_STAGE_BYTES 2048
#define SMEM_RED_STRIDE 2048
#define SMEM_TOTAL 101376
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
kernel_cake_tinygemm2_wgmma_52dd13ad918199aca550(const __grid_constant__ CUtensorMap W, const __grid_constant__ CUtensorMap X, __nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ bias, int M, int N, int K)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define full_addr (mbar_base + 0)
    #define empty_addr (mbar_base + 64)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* sw = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int sw_addr = smem + 1024;
    __nv_bfloat16* sx = reinterpret_cast<__nv_bfloat16*>(smem_raw + 9216);
    const int sx_addr = smem + 9216;
    float* red = reinterpret_cast<float*>(smem_raw + 99328);
    const int red_addr = smem + 99328;
    if (tid == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&W))) : "memory"); }
    if (tid == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&X))) : "memory"); }

    // Mbarrier init (2 pipeline groups, 0 ordered-sequence groups, 16 barriers)
    // Mbarriers at smem_raw[0..128)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'pipe' ---
            // full: 8 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // empty: 8 barriers, init_count=4
            mbarrier_init(smem + 64, 4);
            mbarrier_init(smem + 72, 4);
            mbarrier_init(smem + 80, 4);
            mbarrier_init(smem + 88, 4);
            mbarrier_init(smem + 96, 4);
            mbarrier_init(smem + 104, 4);
            mbarrier_init(smem + 112, 4);
            mbarrier_init(smem + 120, 4);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // Kernel post-init ops
    int n0 = blockIdx.x * 8;
    int m0 = blockIdx.y * 16;
    int rounds = (K + 1023) / 1024;
    int nslices = ((rounds * 4) < ((K + 63) / 64 - 12 * (rounds - 1)) ? (rounds * 4) : ((K + 63) / 64 - 12 * (rounds - 1)));
    int nstages = nslices + 1 - 1;
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
                unsigned int sb = base + (unsigned int)(slot * 12288);
                uint64_t d_b_0 = (((uint64_t)(((sb + (9216u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
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
                        mbarrier_arrive(empty_addr + ((int)((unsigned int)(st - 1) % 8)) * 8);
                    }
                }
                slot += 1;
                if (slot == 8) { slot = 0; _phase_full ^= 1; }
            }
            mbarrier_wait(full_addr + (slot) * 8, _phase_full);
            asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
            unsigned int sb_1 = base + (unsigned int)(slot * 12288);
            uint64_t d_b_1 = (((uint64_t)(((sb_1 + (9216u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
            uint64_t d_a_1 = (((uint64_t)(((sb_1 + (1024u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
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
            int pre = ((8) < (nstages) ? (8) : (nstages));
            if (elect_sync()) {
                #pragma unroll 1
                for (int st_1 = 0; st_1 < pre; st_1++) {
                    int nv_w = ((1) < (nslices - st_1) ? (1) : (nslices - st_1));
                    mbarrier_arrive_expect_tx(full_addr + ((int)((unsigned int)st_1 % 8)) * 8, nv_w * 12288);
                    int sl_0 = st_1;
                    int kb_0 = (sl_0 / 4 * 16 + sl_0 % 4) * 64;
                    unsigned int dst_0 = sw_addr + (unsigned int)(st_1 % 8 * 12288);
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3}], [%4], %5;"
                        :: "r"(dst_0 + 0), "l"((&W)), "r"(kb_0), "r"(m0),
                           "r"(full_addr + ((int)((unsigned int)st_1 % 8)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    int sl_1 = st_1;
                    int kb_1 = (sl_1 / 4 * 16 + sl_1 % 4) * 64;
                    unsigned int dst_1 = sw_addr + (unsigned int)(st_1 % 8 * 12288);
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3}], [%4], %5;"
                        :: "r"(dst_1 + 2048), "l"((&W)), "r"(kb_1 + 256), "r"(m0),
                           "r"(full_addr + ((int)((unsigned int)st_1 % 8)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    int sl_2 = st_1;
                    int kb_2 = (sl_2 / 4 * 16 + sl_2 % 4) * 64;
                    unsigned int dst_2 = sw_addr + (unsigned int)(st_1 % 8 * 12288);
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3}], [%4], %5;"
                        :: "r"(dst_2 + 4096), "l"((&W)), "r"(kb_2 + 512), "r"(m0),
                           "r"(full_addr + ((int)((unsigned int)st_1 % 8)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    int sl_3 = st_1;
                    int kb_3 = (sl_3 / 4 * 16 + sl_3 % 4) * 64;
                    unsigned int dst_3 = sw_addr + (unsigned int)(st_1 % 8 * 12288);
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3}], [%4], %5;"
                        :: "r"(dst_3 + 6144), "l"((&W)), "r"(kb_3 + 768), "r"(m0),
                           "r"(full_addr + ((int)((unsigned int)st_1 % 8)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                }
            }
            if (elect_sync()) {
                #pragma unroll 1
                for (int st_2 = 0; st_2 < pre; st_2++) {
                    int slx_0 = st_2;
                    int kbx_0 = (slx_0 / 4 * 16 + slx_0 % 4) * 64;
                    unsigned int dstx_0 = sx_addr + (unsigned int)(st_2 % 8 * 12288);
                    tma_2d_gmem2smem(dstx_0 + 0, (&X), kbx_0, n0, full_addr + ((int)((unsigned int)st_2 % 8)) * 8);
                    int slx_1 = st_2;
                    int kbx_1 = (slx_1 / 4 * 16 + slx_1 % 4) * 64;
                    unsigned int dstx_1 = sx_addr + (unsigned int)(st_2 % 8 * 12288);
                    tma_2d_gmem2smem(dstx_1 + 1024, (&X), kbx_1 + 256, n0, full_addr + ((int)((unsigned int)st_2 % 8)) * 8);
                    int slx_2 = st_2;
                    int kbx_2 = (slx_2 / 4 * 16 + slx_2 % 4) * 64;
                    unsigned int dstx_2 = sx_addr + (unsigned int)(st_2 % 8 * 12288);
                    tma_2d_gmem2smem(dstx_2 + 2048, (&X), kbx_2 + 512, n0, full_addr + ((int)((unsigned int)st_2 % 8)) * 8);
                    int slx_3 = st_2;
                    int kbx_3 = (slx_3 / 4 * 16 + slx_3 % 4) * 64;
                    unsigned int dstx_3 = sx_addr + (unsigned int)(st_2 % 8 * 12288);
                    tma_2d_gmem2smem(dstx_3 + 3072, (&X), kbx_3 + 768, n0, full_addr + ((int)((unsigned int)st_2 % 8)) * 8);
                }
                #pragma unroll 1
                for (int st_3 = pre; st_3 < nstages; st_3++) {
                    int nv_s = ((1) < (nslices - st_3) ? (1) : (nslices - st_3));
                    int _stage_empty = (int)((unsigned int)st_3 % 8);
                    int _parity_empty = (int)((unsigned int)st_3 / 8 % 2) ^ 1;
                    mbarrier_wait(empty_addr + (_stage_empty) * 8, _parity_empty);
                    mbarrier_arrive_expect_tx(full_addr + ((int)((unsigned int)st_3 % 8)) * 8, nv_s * 12288);
                    int sl_4 = st_3;
                    int kb_4 = (sl_4 / 4 * 16 + sl_4 % 4) * 64;
                    unsigned int dst_4 = sw_addr + (unsigned int)(st_3 % 8 * 12288);
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3}], [%4], %5;"
                        :: "r"(dst_4 + 0), "l"((&W)), "r"(kb_4), "r"(m0),
                           "r"(full_addr + ((int)((unsigned int)st_3 % 8)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    int sl_5 = st_3;
                    int kb_5 = (sl_5 / 4 * 16 + sl_5 % 4) * 64;
                    unsigned int dst_5 = sw_addr + (unsigned int)(st_3 % 8 * 12288);
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3}], [%4], %5;"
                        :: "r"(dst_5 + 2048), "l"((&W)), "r"(kb_5 + 256), "r"(m0),
                           "r"(full_addr + ((int)((unsigned int)st_3 % 8)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    int sl_6 = st_3;
                    int kb_6 = (sl_6 / 4 * 16 + sl_6 % 4) * 64;
                    unsigned int dst_6 = sw_addr + (unsigned int)(st_3 % 8 * 12288);
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3}], [%4], %5;"
                        :: "r"(dst_6 + 4096), "l"((&W)), "r"(kb_6 + 512), "r"(m0),
                           "r"(full_addr + ((int)((unsigned int)st_3 % 8)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    int sl_7 = st_3;
                    int kb_7 = (sl_7 / 4 * 16 + sl_7 % 4) * 64;
                    unsigned int dst_7 = sw_addr + (unsigned int)(st_3 % 8 * 12288);
                    asm volatile(
                        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                        " [%0], [%1, {%2, %3}], [%4], %5;"
                        :: "r"(dst_7 + 6144), "l"((&W)), "r"(kb_7 + 768), "r"(m0),
                           "r"(full_addr + ((int)((unsigned int)st_3 % 8)) * 8), "l"(0x12F0000000000000ULL) : "memory");
                    int slx_4 = st_3;
                    int kbx_4 = (slx_4 / 4 * 16 + slx_4 % 4) * 64;
                    unsigned int dstx_4 = sx_addr + (unsigned int)(st_3 % 8 * 12288);
                    tma_2d_gmem2smem(dstx_4 + 0, (&X), kbx_4, n0, full_addr + ((int)((unsigned int)st_3 % 8)) * 8);
                    int slx_5 = st_3;
                    int kbx_5 = (slx_5 / 4 * 16 + slx_5 % 4) * 64;
                    unsigned int dstx_5 = sx_addr + (unsigned int)(st_3 % 8 * 12288);
                    tma_2d_gmem2smem(dstx_5 + 1024, (&X), kbx_5 + 256, n0, full_addr + ((int)((unsigned int)st_3 % 8)) * 8);
                    int slx_6 = st_3;
                    int kbx_6 = (slx_6 / 4 * 16 + slx_6 % 4) * 64;
                    unsigned int dstx_6 = sx_addr + (unsigned int)(st_3 % 8 * 12288);
                    tma_2d_gmem2smem(dstx_6 + 2048, (&X), kbx_6 + 512, n0, full_addr + ((int)((unsigned int)st_3 % 8)) * 8);
                    int slx_7 = st_3;
                    int kbx_7 = (slx_7 / 4 * 16 + slx_7 % 4) * 64;
                    unsigned int dstx_7 = sx_addr + (unsigned int)(st_3 % 8 * 12288);
                    tma_2d_gmem2smem(dstx_7 + 3072, (&X), kbx_7 + 768, n0, full_addr + ((int)((unsigned int)st_3 % 8)) * 8);
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
