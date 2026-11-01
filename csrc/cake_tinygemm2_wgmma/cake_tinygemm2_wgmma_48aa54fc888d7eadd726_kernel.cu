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
#define SMEM_SW_STAGE_BYTES 22528
#define SMEM_SW_STRIDE 24576
#define SMEM_SX_OFF 17408
#define SMEM_SX_STAGE_BYTES 8192
#define SMEM_SX_STRIDE 24576
#define SMEM_RED_OFF 197632
#define SMEM_RED_STAGE_BYTES 1536
#define SMEM_RED_STRIDE 1536
#define SMEM_TOTAL 199168
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

__global__ __launch_bounds__(160, LAUNCH_MIN_BLOCKS) __cluster_dims__(4,1,1) void
kernel_cake_tinygemm2_wgmma_48aa54fc888d7eadd726(const __grid_constant__ CUtensorMap W, const __grid_constant__ CUtensorMap X, __nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ bias, int M, int N, int K)
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
    #define red_full_addr (mbar_base + 128)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 4;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 4;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    __nv_bfloat16* sw = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int sw_addr = smem + 1024;
    __nv_bfloat16* sx = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int sx_addr = smem + 17408;
    float* red = reinterpret_cast<float*>(smem_raw + 197632);
    const int red_addr = smem + 197632;
    if (tid == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&W))) : "memory"); }
    if (tid == 0) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&X))) : "memory"); }

    // Mbarrier init (3 pipeline groups, 0 ordered-sequence groups, 17 barriers)
    // Mbarriers at smem_raw[0..136)

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
            // red_full: 1 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");

    // Kernel post-init ops
    int rank = cta_rank;
    int n0 = blockIdx.x / 4 * 8;
    int m0 = blockIdx.y * 16;
    int rounds = (K + 1023) / 1024;
    int nslices = ((rounds * 4) < ((K + 63) / 64 - 12 * (rounds - 1)) ? (rounds * 4) : ((K + 63) / 64 - 12 * (rounds - 1)));
    int nstages = (nslices + 8 - 1) / 8;

    // ---- Role: mma ----
    if (warp <= 3) {
        { // mma_main
            float acc_0[4] = {0};
            int g = lane / 4;
            int t = lane % 4;
            int _base = make_warp_uniform((unsigned int)smem);
            unsigned int base = _base;
            int slot = 0;
            asm volatile("" : "+f"(acc_0[0]) :: "memory");
            asm volatile("" : "+f"(acc_0[1]) :: "memory");
            asm volatile("" : "+f"(acc_0[2]) :: "memory");
            asm volatile("" : "+f"(acc_0[3]) :: "memory");
            unsigned int _phase_full = 0;
            #pragma unroll 1
            for (int st = 0; st < nstages - 1; st++) {
                mbarrier_wait(full_addr + (slot) * 8, _phase_full);
                asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
                unsigned int sb = base + (unsigned int)(slot * 24576);
                uint64_t d_b_0 = (((uint64_t)(((sb + (17408u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                uint64_t d_a_0 = (((uint64_t)(((sb + (1024u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_0), "l"(d_b_0)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_0 + 2), "l"(d_b_0 + 2)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_0 + 4), "l"(d_b_0 + 4)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_0 + 6), "l"(d_b_0 + 6)
                    : "memory");
                uint64_t d_b_1 = (((uint64_t)(((sb + (17408u + 1024u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                uint64_t d_a_1 = (((uint64_t)(((sb + (1024u + 2048u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_1), "l"(d_b_1)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_1 + 2), "l"(d_b_1 + 2)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_1 + 4), "l"(d_b_1 + 4)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_1 + 6), "l"(d_b_1 + 6)
                    : "memory");
                uint64_t d_b_2 = (((uint64_t)(((sb + (17408u + 2048u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                uint64_t d_a_2 = (((uint64_t)(((sb + (1024u + 4096u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_2), "l"(d_b_2)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_2 + 2), "l"(d_b_2 + 2)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_2 + 4), "l"(d_b_2 + 4)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_2 + 6), "l"(d_b_2 + 6)
                    : "memory");
                uint64_t d_b_3 = (((uint64_t)(((sb + (17408u + 3072u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                uint64_t d_a_3 = (((uint64_t)(((sb + (1024u + 6144u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_3), "l"(d_b_3)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_3 + 2), "l"(d_b_3 + 2)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_3 + 4), "l"(d_b_3 + 4)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_3 + 6), "l"(d_b_3 + 6)
                    : "memory");
                uint64_t d_b_4 = (((uint64_t)(((sb + (17408u + 4096u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                uint64_t d_a_4 = (((uint64_t)(((sb + (1024u + 8192u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_4), "l"(d_b_4)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_4 + 2), "l"(d_b_4 + 2)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_4 + 4), "l"(d_b_4 + 4)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_4 + 6), "l"(d_b_4 + 6)
                    : "memory");
                uint64_t d_b_5 = (((uint64_t)(((sb + (17408u + 5120u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                uint64_t d_a_5 = (((uint64_t)(((sb + (1024u + 10240u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_5), "l"(d_b_5)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_5 + 2), "l"(d_b_5 + 2)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_5 + 4), "l"(d_b_5 + 4)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_5 + 6), "l"(d_b_5 + 6)
                    : "memory");
                uint64_t d_b_6 = (((uint64_t)(((sb + (17408u + 6144u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                uint64_t d_a_6 = (((uint64_t)(((sb + (1024u + 12288u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_6), "l"(d_b_6)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_6 + 2), "l"(d_b_6 + 2)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_6 + 4), "l"(d_b_6 + 4)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_6 + 6), "l"(d_b_6 + 6)
                    : "memory");
                uint64_t d_b_7 = (((uint64_t)(((sb + (17408u + 7168u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                uint64_t d_a_7 = (((uint64_t)(((sb + (1024u + 14336u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_7), "l"(d_b_7)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_7 + 2), "l"(d_b_7 + 2)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_7 + 4), "l"(d_b_7 + 4)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_7 + 6), "l"(d_b_7 + 6)
                    : "memory");
                asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
                asm volatile("" : "+f"(acc_0[0]) :: "memory");
                asm volatile("" : "+f"(acc_0[1]) :: "memory");
                asm volatile("" : "+f"(acc_0[2]) :: "memory");
                asm volatile("" : "+f"(acc_0[3]) :: "memory");
                asm volatile("wgmma.wait_group.sync.aligned 1;" ::: "memory");
                asm volatile("" : "+f"(acc_0[0]) :: "memory");
                asm volatile("" : "+f"(acc_0[1]) :: "memory");
                asm volatile("" : "+f"(acc_0[2]) :: "memory");
                asm volatile("" : "+f"(acc_0[3]) :: "memory");
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
            unsigned int sb_1 = base + (unsigned int)(slot * 24576);
            uint64_t d_b_8 = (((uint64_t)(((sb_1 + (17408u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
            uint64_t d_a_8 = (((uint64_t)(((sb_1 + (1024u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                : "l"(d_a_8), "l"(d_b_8)
                : "memory");
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                : "l"(d_a_8 + 2), "l"(d_b_8 + 2)
                : "memory");
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                : "l"(d_a_8 + 4), "l"(d_b_8 + 4)
                : "memory");
            asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                : "l"(d_a_8 + 6), "l"(d_b_8 + 6)
                : "memory");
            if (1 < nslices - (nstages - 1) * 8) {
                uint64_t d_b_9 = (((uint64_t)(((sb_1 + (17408u + 1024u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                uint64_t d_a_9 = (((uint64_t)(((sb_1 + (1024u + 2048u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_9), "l"(d_b_9)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_9 + 2), "l"(d_b_9 + 2)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_9 + 4), "l"(d_b_9 + 4)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_9 + 6), "l"(d_b_9 + 6)
                    : "memory");
            }
            if (2 < nslices - (nstages - 1) * 8) {
                uint64_t d_b_10 = (((uint64_t)(((sb_1 + (17408u + 2048u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                uint64_t d_a_10 = (((uint64_t)(((sb_1 + (1024u + 4096u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_10), "l"(d_b_10)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_10 + 2), "l"(d_b_10 + 2)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_10 + 4), "l"(d_b_10 + 4)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_10 + 6), "l"(d_b_10 + 6)
                    : "memory");
            }
            if (3 < nslices - (nstages - 1) * 8) {
                uint64_t d_b_11 = (((uint64_t)(((sb_1 + (17408u + 3072u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                uint64_t d_a_11 = (((uint64_t)(((sb_1 + (1024u + 6144u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_11), "l"(d_b_11)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_11 + 2), "l"(d_b_11 + 2)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_11 + 4), "l"(d_b_11 + 4)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_11 + 6), "l"(d_b_11 + 6)
                    : "memory");
            }
            if (4 < nslices - (nstages - 1) * 8) {
                uint64_t d_b_12 = (((uint64_t)(((sb_1 + (17408u + 4096u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                uint64_t d_a_12 = (((uint64_t)(((sb_1 + (1024u + 8192u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_12), "l"(d_b_12)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_12 + 2), "l"(d_b_12 + 2)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_12 + 4), "l"(d_b_12 + 4)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_12 + 6), "l"(d_b_12 + 6)
                    : "memory");
            }
            if (5 < nslices - (nstages - 1) * 8) {
                uint64_t d_b_13 = (((uint64_t)(((sb_1 + (17408u + 5120u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                uint64_t d_a_13 = (((uint64_t)(((sb_1 + (1024u + 10240u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_13), "l"(d_b_13)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_13 + 2), "l"(d_b_13 + 2)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_13 + 4), "l"(d_b_13 + 4)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_13 + 6), "l"(d_b_13 + 6)
                    : "memory");
            }
            if (6 < nslices - (nstages - 1) * 8) {
                uint64_t d_b_14 = (((uint64_t)(((sb_1 + (17408u + 6144u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                uint64_t d_a_14 = (((uint64_t)(((sb_1 + (1024u + 12288u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_14), "l"(d_b_14)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_14 + 2), "l"(d_b_14 + 2)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_14 + 4), "l"(d_b_14 + 4)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_14 + 6), "l"(d_b_14 + 6)
                    : "memory");
            }
            if (7 < nslices - (nstages - 1) * 8) {
                uint64_t d_b_15 = (((uint64_t)(((sb_1 + (17408u + 7168u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                uint64_t d_a_15 = (((uint64_t)(((sb_1 + (1024u + 14336u))) >> 4) & 0x3FFFULL) | ((uint64_t)(0) << 16) | ((uint64_t)(64) << 32) | (1ULL << 62));
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_15), "l"(d_b_15)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_15 + 2), "l"(d_b_15 + 2)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_15 + 4), "l"(d_b_15 + 4)
                    : "memory");
                asm volatile("{\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0, %1, %2, %3}, %4, %5, 1, 1, 1, 0, 0;\n}\n"
                    : "+f"(acc_0[0]), "+f"(acc_0[1]), "+f"(acc_0[2]), "+f"(acc_0[3])
                    : "l"(d_a_15 + 6), "l"(d_b_15 + 6)
                    : "memory");
            }
            asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
            asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
            asm volatile("" : "+f"(acc_0[0]) :: "memory");
            asm volatile("" : "+f"(acc_0[1]) :: "memory");
            asm volatile("" : "+f"(acc_0[2]) :: "memory");
            asm volatile("" : "+f"(acc_0[3]) :: "memory");
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
            if (warp == 0) {
                if (rank > 0) {
                    uint32_t _cluster_mapa_0;
                    asm volatile(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_cluster_mapa_0) : "r"(red_addr + (unsigned int)((rank - 1) * 512 + 16 * lane)), "r"(0));
                    uint32_t _mbar_peer_1;
                    asm(
                        "mapa.shared::cluster.u32 %0, %1, %2;"
                        : "=r"(_mbar_peer_1) : "r"(red_full_addr), "r"(0));
                    asm volatile(
                        "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                        :: "r"(_cluster_mapa_0), "r"(__float_as_uint(acc_0[0])), "r"(__float_as_uint(acc_0[1])), "r"(__float_as_uint(acc_0[2])), "r"(__float_as_uint(acc_0[3])), "r"(_mbar_peer_1) : "memory");
                } else {
                    if (lane == 0) {
                        mbarrier_arrive_expect_tx(red_full_addr, 1536);
                    }
                    mbarrier_wait_cluster_hint(red_full_addr, 0, 10000000);
                    float v_e_0 = acc_0[0];
                    v_e_0 = v_e_0 + red[4 * lane];
                    v_e_0 = v_e_0 + red[128 + 4 * lane];
                    v_e_0 = v_e_0 + red[256 + 4 * lane];
                    int m_e_0 = m0 + g;
                    int n_e_0 = n0 + 2 * t;
                    if (m_e_0 < M && n_e_0 < N) {
                        *(reinterpret_cast<__nv_bfloat16*>(out + (m_e_0 + n_e_0 * M)) + (0)) = (__nv_bfloat16)v_e_0;
                    }
                    float v_e_1 = acc_0[1];
                    v_e_1 = v_e_1 + red[4 * lane + 1];
                    v_e_1 = v_e_1 + red[128 + 4 * lane + 1];
                    v_e_1 = v_e_1 + red[256 + 4 * lane + 1];
                    int m_e_1 = m0 + g;
                    int n_e_1 = n0 + 2 * t + 1;
                    if (m_e_1 < M && n_e_1 < N) {
                        *(reinterpret_cast<__nv_bfloat16*>(out + (m_e_1 + n_e_1 * M)) + (0)) = (__nv_bfloat16)v_e_1;
                    }
                    float v_e_2 = acc_0[2];
                    v_e_2 = v_e_2 + red[4 * lane + 2];
                    v_e_2 = v_e_2 + red[128 + 4 * lane + 2];
                    v_e_2 = v_e_2 + red[256 + 4 * lane + 2];
                    int m_e_2 = m0 + g + 8;
                    int n_e_2 = n0 + 2 * t;
                    if (m_e_2 < M && n_e_2 < N) {
                        *(reinterpret_cast<__nv_bfloat16*>(out + (m_e_2 + n_e_2 * M)) + (0)) = (__nv_bfloat16)v_e_2;
                    }
                    float v_e_3 = acc_0[3];
                    v_e_3 = v_e_3 + red[4 * lane + 3];
                    v_e_3 = v_e_3 + red[128 + 4 * lane + 3];
                    v_e_3 = v_e_3 + red[256 + 4 * lane + 3];
                    int m_e_3 = m0 + g + 8;
                    int n_e_3 = n0 + 2 * t + 1;
                    if (m_e_3 < M && n_e_3 < N) {
                        *(reinterpret_cast<__nv_bfloat16*>(out + (m_e_3 + n_e_3 * M)) + (0)) = (__nv_bfloat16)v_e_3;
                    }
                }
            }
        }
    }
    // ---- Role: load ----
    if (warp == 4) {
        { // load_main
            int pre = ((8) < (nstages) ? (8) : (nstages));
            int early = ((1) < (pre) ? (1) : (pre));
            if (elect_sync()) {
                #pragma unroll 1
                for (int st_1 = 0; st_1 < early; st_1++) {
                    int nv_w = ((8) < (nslices - st_1 * 8) ? (8) : (nslices - st_1 * 8));
                    mbarrier_arrive_expect_tx(full_addr + ((int)((unsigned int)st_1 % 8)) * 8, (nv_w + 3) / 4 * 12288);
                    tma_3d_gmem2smem(sw_addr + (unsigned int)(st_1 % 8 * 24576), (&W), 0, m0, 4 * (st_1 * 8 + rank), full_addr + ((int)((unsigned int)st_1 % 8)) * 8);
                    if (4 < nv_w) {
                        tma_3d_gmem2smem(sw_addr + (unsigned int)(st_1 % 8 * 24576 + 8192), (&W), 0, m0, 4 * (st_1 * 8 + 4 + rank), full_addr + ((int)((unsigned int)st_1 % 8)) * 8);
                    }
                }
            }
            asm volatile("griddepcontrol.wait;" ::: "memory");
            if (lane == 0) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
            if (elect_sync()) {
                #pragma unroll 1
                for (int st_2 = 0; st_2 < early; st_2++) {
                    tma_3d_gmem2smem(sx_addr + (unsigned int)(st_2 % 8 * 24576), (&X), 0, n0, 4 * (st_2 * 8 + rank), full_addr + ((int)((unsigned int)st_2 % 8)) * 8);
                    int _min_0 = ((8) < (nslices - st_2 * 8) ? (8) : (nslices - st_2 * 8));
                    if (4 < _min_0) {
                        tma_3d_gmem2smem(sx_addr + (unsigned int)(st_2 % 8 * 24576 + 4096), (&X), 0, n0, 4 * (st_2 * 8 + 4 + rank), full_addr + ((int)((unsigned int)st_2 % 8)) * 8);
                    }
                }
                #pragma unroll 1
                for (int st_3 = early; st_3 < pre; st_3++) {
                    int nv_p = ((8) < (nslices - st_3 * 8) ? (8) : (nslices - st_3 * 8));
                    mbarrier_arrive_expect_tx(full_addr + ((int)((unsigned int)st_3 % 8)) * 8, (nv_p + 3) / 4 * 12288);
                    tma_3d_gmem2smem(sx_addr + (unsigned int)(st_3 % 8 * 24576), (&X), 0, n0, 4 * (st_3 * 8 + rank), full_addr + ((int)((unsigned int)st_3 % 8)) * 8);
                    if (4 < nv_p) {
                        tma_3d_gmem2smem(sx_addr + (unsigned int)(st_3 % 8 * 24576 + 4096), (&X), 0, n0, 4 * (st_3 * 8 + 4 + rank), full_addr + ((int)((unsigned int)st_3 % 8)) * 8);
                    }
                    tma_3d_gmem2smem(sw_addr + (unsigned int)(st_3 % 8 * 24576), (&W), 0, m0, 4 * (st_3 * 8 + rank), full_addr + ((int)((unsigned int)st_3 % 8)) * 8);
                    if (4 < nv_p) {
                        tma_3d_gmem2smem(sw_addr + (unsigned int)(st_3 % 8 * 24576 + 8192), (&W), 0, m0, 4 * (st_3 * 8 + 4 + rank), full_addr + ((int)((unsigned int)st_3 % 8)) * 8);
                    }
                }
                #pragma unroll 1
                for (int st_4 = pre; st_4 < nstages; st_4++) {
                    int nv_s = ((8) < (nslices - st_4 * 8) ? (8) : (nslices - st_4 * 8));
                    int _stage_empty = (int)((unsigned int)st_4 % 8);
                    int _parity_empty = (int)((unsigned int)st_4 / 8 % 2) ^ 1;
                    mbarrier_wait(empty_addr + (_stage_empty) * 8, _parity_empty);
                    mbarrier_arrive_expect_tx(full_addr + ((int)((unsigned int)st_4 % 8)) * 8, (nv_s + 3) / 4 * 12288);
                    tma_3d_gmem2smem(sx_addr + (unsigned int)(st_4 % 8 * 24576), (&X), 0, n0, 4 * (st_4 * 8 + rank), full_addr + ((int)((unsigned int)st_4 % 8)) * 8);
                    if (4 < nv_s) {
                        tma_3d_gmem2smem(sx_addr + (unsigned int)(st_4 % 8 * 24576 + 4096), (&X), 0, n0, 4 * (st_4 * 8 + 4 + rank), full_addr + ((int)((unsigned int)st_4 % 8)) * 8);
                    }
                    tma_3d_gmem2smem(sw_addr + (unsigned int)(st_4 % 8 * 24576), (&W), 0, m0, 4 * (st_4 * 8 + rank), full_addr + ((int)((unsigned int)st_4 % 8)) * 8);
                    if (4 < nv_s) {
                        tma_3d_gmem2smem(sw_addr + (unsigned int)(st_4 % 8 * 24576 + 8192), (&W), 0, m0, 4 * (st_4 * 8 + 4 + rank), full_addr + ((int)((unsigned int)st_4 % 8)) * 8);
                    }
                }
            }
            asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        }
    }
}

} // extern "C"
