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
#define TMEM_NCOLS 256
#define TMEM_ACCUM_OFFSET 0
#define NUM_K_PIPE_STAGES 8
#define NUM_MMA_PIPE_STAGES 2
#define NUM_WORK_PIPE_STAGES 3
#define NUM_THROTTLE_PIPE_STAGES 3
#define NUM_DRAIN_PIPE_STAGES 1
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 16384
#define SMEM_SMEM_B_OFF 132096
#define SMEM_SMEM_B_STAGE_BYTES 8192
#define SMEM_SMEM_B_STRIDE 8192
#define SMEM_EPI_STAGING_OFF 197632
#define SMEM_EPI_STAGING_STAGE_BYTES 4096
#define SMEM_EPI_STAGING_STRIDE 4096
#define SMEM_WORK_RESPONSE_OFF 201728
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_FAST_DRAIN_RESPONSE_OFF 201776
#define SMEM_FAST_DRAIN_RESPONSE_STAGE_BYTES 64
#define SMEM_FAST_DRAIN_RESPONSE_STRIDE 64
#define SMEM_TOTAL 201856
#define THREADS 512
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




union MmaSmemDesc {
    uint64_t u64;
    uint32_t u32[2];
};


__device__ __forceinline__ void mma_ss_step_cg2(
    int a_lo, int b_lo, int taddr, uint32_t i_desc, int enable_d,
    uint32_t a_dhi, uint32_t b_dhi) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader, p;\n\t"
        ".reg .b32 adhi, bdhi, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
        ".reg .b64 da, db;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\t"
        "mov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
        "mov.b32 adhi, %5;\n\t"
        "mov.b32 bdhi, %6;\n\t"
        "mov.b64 da, {%0, adhi};\n\t"
        "mov.b64 db, {%1, bdhi};\n\t"
        "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, %3, "
        "{m0, m1, m2, m3, m4, m5, m6, m7}, p;\n\t"
        "}\n"
        :: "r"(a_lo), "r"(b_lo), "r"(taddr), "r"(i_desc), "r"(enable_d), "r"(a_dhi), "r"(b_dhi));
}


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


__device__ __forceinline__ void mbarrier_arrive_expect_tx(int mbar_addr, uint32_t bytes) {
    asm volatile(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
        :: "r"(mbar_addr), "r"(bytes) : "memory");
}



__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float approx_rcp(float x) {
    float y;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
}









__device__ __forceinline__ void tma_gather4_gmem2smem_mc_cta2(
    int dst, const void *tmap_ptr,
    int col_idx, int row0, int row1, int row2, int row3,
    int mbar_addr, unsigned short cta_mask) {
    // Multicast + cta_group::2 variant; see tma_gather4_gmem2smem_mc.
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
        ".mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
        :: "r"(dst), "l"(tmap_ptr), "r"(col_idx),
           "r"(row0), "r"(row1), "r"(row2), "r"(row3),
           "r"(mbar_addr), "h"(cta_mask) : "memory");
}


__device__ __forceinline__ void tma_store_4d(
    const void *tmap, int x, int y, int z, int w, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3, %4}], [%5];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(w), "r"(smem_addr) : "memory");
}


extern "C" {

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_cake_stepfun_moe_f073ef60ad781c16e95a(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap C, int* __restrict__ route_map, int* __restrict__ tile_expert, int* __restrict__ tile_mn_limit, int* __restrict__ num_non_exiting_ctas, float* __restrict__ scale_c, float* __restrict__ scale_gate, float* __restrict__ clamp_limit, float* __restrict__ act_alpha, float* __restrict__ act_beta, int M_out, int K, int grid_m, int grid_n, int K_tiles)
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
    #define a_full_addr (mbar_base + 0)
    #define b_full_addr (mbar_base + 64)
    #define k_done_addr (mbar_base + 128)
    #define mma_full_addr (mbar_base + 192)
    #define mma_free_addr (mbar_base + 208)
    #define work_full_addr (mbar_base + 224)
    #define work_empty_addr (mbar_base + 248)
    #define throttle_full_addr (mbar_base + 272)
    #define throttle_empty_addr (mbar_base + 296)
    #define drain_full_addr (mbar_base + 320)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + 132096);
    const int smem_b_addr = smem + 132096;
    uint16_t* epi_staging = reinterpret_cast<uint16_t*>(smem_raw + 197632);
    const int epi_staging_addr = smem + 197632;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 201728);
    const int work_response_addr = smem + 201728;
    unsigned int* fast_drain_response = reinterpret_cast<unsigned int*>(smem_raw + 201776);
    const int fast_drain_response_addr = smem + 201776;
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if ((int)blockIdx.y >= num_non_exiting_ctas[0]) return;

    // Mbarrier init (10 pipeline groups, 0 ordered-sequence groups, 41 barriers)
    // Mbarriers at smem_raw[0..328)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'k_pipe' ---
            // a_full: 8 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            mbarrier_init(smem + 8, 2);
            mbarrier_init(smem + 16, 2);
            mbarrier_init(smem + 24, 2);
            mbarrier_init(smem + 32, 2);
            mbarrier_init(smem + 40, 2);
            mbarrier_init(smem + 48, 2);
            mbarrier_init(smem + 56, 2);
            // b_full: 8 barriers, init_count=2
            mbarrier_init(smem + 64, 2);
            mbarrier_init(smem + 72, 2);
            mbarrier_init(smem + 80, 2);
            mbarrier_init(smem + 88, 2);
            mbarrier_init(smem + 96, 2);
            mbarrier_init(smem + 104, 2);
            mbarrier_init(smem + 112, 2);
            mbarrier_init(smem + 120, 2);
            // k_done: 8 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            // --- pipeline 'mma_pipe' ---
            // mma_full: 2 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            // mma_free: 2 barriers, init_count=256
            mbarrier_init(smem + 208, 256);
            mbarrier_init(smem + 216, 256);
            // --- pipeline 'work_pipe' ---
            // work_full: 3 barriers, init_count=1
            mbarrier_init(smem + 224, 1);
            mbarrier_init(smem + 232, 1);
            mbarrier_init(smem + 240, 1);
            // work_empty: 3 barriers, init_count=960
            mbarrier_init(smem + 248, 960);
            mbarrier_init(smem + 256, 960);
            mbarrier_init(smem + 264, 960);
            // --- pipeline 'throttle_pipe' ---
            // throttle_full: 3 barriers, init_count=32
            mbarrier_init(smem + 272, 32);
            mbarrier_init(smem + 280, 32);
            mbarrier_init(smem + 288, 32);
            // throttle_empty: 3 barriers, init_count=32
            mbarrier_init(smem + 296, 32);
            mbarrier_init(smem + 304, 32);
            mbarrier_init(smem + 312, 32);
            // --- pipeline 'drain_pipe' ---
            // drain_full: 1 barriers, init_count=1
            mbarrier_init(smem + 320, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 328);
    if (warp == 0) {
        int _tmem_hold = smem + 328;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
    }

    // ---- Role: epilogue ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 160;");
        { // epilogue_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            const int warp_0 = warp;
            const int lane_1 = lane;
            unsigned int acc_stage = 0;
            unsigned int work_stage = 0;
            unsigned int m_tile = blockIdx.x;
            unsigned int n_tile = blockIdx.y;
            int bound = num_non_exiting_ctas[0];
            float quad[4] = {0};
            unsigned int _phase_mma_full = 0;
            unsigned int _phase_work_full = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter = 0; _tile_iter < grid_m / 2 * grid_n; _tile_iter++) {
                if (m_tile >= (unsigned int)grid_m || n_tile >= (unsigned int)bound) {
                    break;
                }
                int expert = tile_expert[n_tile];
                int valid_rows = (unsigned int)tile_mn_limit[n_tile] - n_tile * 128;
                float sc = scale_c[expert];
                float sg = scale_gate[expert];
                float cl = clamp_limit[expert];
                float neg_cl = -cl;
                float fused = 1.4426950216293335f * sg;
                mbarrier_wait(mma_full_addr + (acc_stage) * 8, _phase_mma_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int acc_offset = acc_stage * 128;
                float _tmem_load_0[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[31]))
                    : "r"(taddr + (unsigned int)acc_offset));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float _tmem_load_1[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[31]))
                    : "r"(taddr + 1048576 + (unsigned int)acc_offset));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int base_row = warp_0 * 16 + lane_1 / 4 * 2;
                asm volatile("cp.async.bulk.wait_group.read 0;");
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                int local_token0 = lane_1 % 4 * 2;
                int local_token1 = local_token0 + 1;
                float x0_00 = _tmem_load_0[0];
                float x0_01 = _tmem_load_0[1];
                float x1_00 = _tmem_load_0[2];
                float x1_01 = _tmem_load_0[3];
                float x0_10 = _tmem_load_1[0];
                float x0_11 = _tmem_load_1[1];
                float x1_10 = _tmem_load_1[2];
                float x1_11 = _tmem_load_1[3];
                float _max_0 = max_noftz(x0_00, neg_cl);
                float _min_0 = fminf(_max_0, cl);
                float x0c_00 = _min_0;
                float _max_1 = max_noftz(x0_01, neg_cl);
                float _min_1 = fminf(_max_1, cl);
                float x0c_01 = _min_1;
                float _max_2 = max_noftz(x0_10, neg_cl);
                float _min_2 = fminf(_max_2, cl);
                float x0c_10 = _min_2;
                float _max_3 = max_noftz(x0_11, neg_cl);
                float _min_3 = fminf(_max_3, cl);
                float x0c_11 = _min_3;
                float _fma_0 = __fmaf_rn(x0c_00, sg, 0.0f);
                float lin_00 = _fma_0;
                float _fma_1 = __fmaf_rn(x0c_01, sg, 0.0f);
                float lin_01 = _fma_1;
                float _fma_2 = __fmaf_rn(x0c_10, sg, 0.0f);
                float lin_10 = _fma_2;
                float _fma_3 = __fmaf_rn(x0c_11, sg, 0.0f);
                float lin_11 = _fma_3;
                float _exp2_0 = approx_exp2(-(x1_00 * fused));
                float _rcp_0 = approx_rcp(1.0f + _exp2_0);
                float sig_00 = _rcp_0;
                float _exp2_1 = approx_exp2(-(x1_01 * fused));
                float _rcp_1 = approx_rcp(1.0f + _exp2_1);
                float sig_01 = _rcp_1;
                float _exp2_2 = approx_exp2(-(x1_10 * fused));
                float _rcp_2 = approx_rcp(1.0f + _exp2_2);
                float sig_10 = _rcp_2;
                float _exp2_3 = approx_exp2(-(x1_11 * fused));
                float _rcp_3 = approx_rcp(1.0f + _exp2_3);
                float sig_11 = _rcp_3;
                float act_00 = x1_00 * sig_00;
                float act_01 = x1_01 * sig_01;
                float act_10 = x1_10 * sig_10;
                float act_11 = x1_11 * sig_11;
                float _min_4 = fminf(act_00, cl);
                act_00 = _min_4;
                float _min_5 = fminf(act_01, cl);
                act_01 = _min_5;
                float _min_6 = fminf(act_10, cl);
                act_10 = _min_6;
                float _min_7 = fminf(act_11, cl);
                act_11 = _min_7;
                float y_00 = lin_00 * act_00;
                float y_10 = lin_10 * act_10;
                float y_01 = lin_01 * act_01;
                float y_11 = lin_11 * act_11;
                quad[0] = y_00 * sc;
                quad[1] = y_10 * sc;
                quad[2] = y_01 * sc;
                quad[3] = y_11 * sc;
                uint32_t _fp8_0[1];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(quad[0]), "f"(quad[1]),
                                           "f"(quad[2]), "f"(quad[3]));
                    _fp8_0[0] = _packed;
                }
                int off0 = local_token0 * 64 + base_row;
                int off1 = local_token1 * 64 + base_row;
                int swz0 = off0 ^ (off0 >> 7 & 3) << 4;
                int swz1 = off1 ^ (off1 >> 7 & 3) << 4;
                epi_staging[swz0 >> 1] = _fp8_0[0] & 65535;
                epi_staging[swz1 >> 1] = _fp8_0[0] >> 16;
                int local_token0_0 = lane_1 % 4 * 2 + 8;
                int local_token1_1 = local_token0_0 + 1;
                float x0_00_2 = _tmem_load_0[4];
                float x0_01_3 = _tmem_load_0[5];
                float x1_00_4 = _tmem_load_0[6];
                float x1_01_5 = _tmem_load_0[7];
                float x0_10_6 = _tmem_load_1[4];
                float x0_11_7 = _tmem_load_1[5];
                float x1_10_8 = _tmem_load_1[6];
                float x1_11_9 = _tmem_load_1[7];
                float _max_4 = max_noftz(x0_00_2, neg_cl);
                float _min_8 = fminf(_max_4, cl);
                float x0c_00_10 = _min_8;
                float _max_5 = max_noftz(x0_01_3, neg_cl);
                float _min_9 = fminf(_max_5, cl);
                float x0c_01_11 = _min_9;
                float _max_6 = max_noftz(x0_10_6, neg_cl);
                float _min_10 = fminf(_max_6, cl);
                float x0c_10_12 = _min_10;
                float _max_7 = max_noftz(x0_11_7, neg_cl);
                float _min_11 = fminf(_max_7, cl);
                float x0c_11_13 = _min_11;
                float _fma_4 = __fmaf_rn(x0c_00_10, sg, 0.0f);
                float lin_00_14 = _fma_4;
                float _fma_5 = __fmaf_rn(x0c_01_11, sg, 0.0f);
                float lin_01_15 = _fma_5;
                float _fma_6 = __fmaf_rn(x0c_10_12, sg, 0.0f);
                float lin_10_16 = _fma_6;
                float _fma_7 = __fmaf_rn(x0c_11_13, sg, 0.0f);
                float lin_11_17 = _fma_7;
                float _exp2_4 = approx_exp2(-(x1_00_4 * fused));
                float _rcp_4 = approx_rcp(1.0f + _exp2_4);
                float sig_00_18 = _rcp_4;
                float _exp2_5 = approx_exp2(-(x1_01_5 * fused));
                float _rcp_5 = approx_rcp(1.0f + _exp2_5);
                float sig_01_19 = _rcp_5;
                float _exp2_6 = approx_exp2(-(x1_10_8 * fused));
                float _rcp_6 = approx_rcp(1.0f + _exp2_6);
                float sig_10_20 = _rcp_6;
                float _exp2_7 = approx_exp2(-(x1_11_9 * fused));
                float _rcp_7 = approx_rcp(1.0f + _exp2_7);
                float sig_11_21 = _rcp_7;
                float act_00_22 = x1_00_4 * sig_00_18;
                float act_01_23 = x1_01_5 * sig_01_19;
                float act_10_24 = x1_10_8 * sig_10_20;
                float act_11_25 = x1_11_9 * sig_11_21;
                float _min_12 = fminf(act_00_22, cl);
                act_00_22 = _min_12;
                float _min_13 = fminf(act_01_23, cl);
                act_01_23 = _min_13;
                float _min_14 = fminf(act_10_24, cl);
                act_10_24 = _min_14;
                float _min_15 = fminf(act_11_25, cl);
                act_11_25 = _min_15;
                float y_00_26 = lin_00_14 * act_00_22;
                float y_10_27 = lin_10_16 * act_10_24;
                float y_01_28 = lin_01_15 * act_01_23;
                float y_11_29 = lin_11_17 * act_11_25;
                quad[0] = y_00_26 * sc;
                quad[1] = y_10_27 * sc;
                quad[2] = y_01_28 * sc;
                quad[3] = y_11_29 * sc;
                uint32_t _fp8_1[1];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(quad[0]), "f"(quad[1]),
                                           "f"(quad[2]), "f"(quad[3]));
                    _fp8_1[0] = _packed;
                }
                int off0_30 = local_token0_0 * 64 + base_row;
                int off1_31 = local_token1_1 * 64 + base_row;
                int swz0_32 = off0_30 ^ (off0_30 >> 7 & 3) << 4;
                int swz1_33 = off1_31 ^ (off1_31 >> 7 & 3) << 4;
                epi_staging[swz0_32 >> 1] = _fp8_1[0] & 65535;
                epi_staging[swz1_33 >> 1] = _fp8_1[0] >> 16;
                int local_token0_34 = lane_1 % 4 * 2 + 16;
                int local_token1_35 = local_token0_34 + 1;
                float x0_00_36 = _tmem_load_0[8];
                float x0_01_37 = _tmem_load_0[9];
                float x1_00_38 = _tmem_load_0[10];
                float x1_01_39 = _tmem_load_0[11];
                float x0_10_40 = _tmem_load_1[8];
                float x0_11_41 = _tmem_load_1[9];
                float x1_10_42 = _tmem_load_1[10];
                float x1_11_43 = _tmem_load_1[11];
                float _max_8 = max_noftz(x0_00_36, neg_cl);
                float _min_16 = fminf(_max_8, cl);
                float x0c_00_44 = _min_16;
                float _max_9 = max_noftz(x0_01_37, neg_cl);
                float _min_17 = fminf(_max_9, cl);
                float x0c_01_45 = _min_17;
                float _max_10 = max_noftz(x0_10_40, neg_cl);
                float _min_18 = fminf(_max_10, cl);
                float x0c_10_46 = _min_18;
                float _max_11 = max_noftz(x0_11_41, neg_cl);
                float _min_19 = fminf(_max_11, cl);
                float x0c_11_47 = _min_19;
                float _fma_8 = __fmaf_rn(x0c_00_44, sg, 0.0f);
                float lin_00_48 = _fma_8;
                float _fma_9 = __fmaf_rn(x0c_01_45, sg, 0.0f);
                float lin_01_49 = _fma_9;
                float _fma_10 = __fmaf_rn(x0c_10_46, sg, 0.0f);
                float lin_10_50 = _fma_10;
                float _fma_11 = __fmaf_rn(x0c_11_47, sg, 0.0f);
                float lin_11_51 = _fma_11;
                float _exp2_8 = approx_exp2(-(x1_00_38 * fused));
                float _rcp_8 = approx_rcp(1.0f + _exp2_8);
                float sig_00_52 = _rcp_8;
                float _exp2_9 = approx_exp2(-(x1_01_39 * fused));
                float _rcp_9 = approx_rcp(1.0f + _exp2_9);
                float sig_01_53 = _rcp_9;
                float _exp2_10 = approx_exp2(-(x1_10_42 * fused));
                float _rcp_10 = approx_rcp(1.0f + _exp2_10);
                float sig_10_54 = _rcp_10;
                float _exp2_11 = approx_exp2(-(x1_11_43 * fused));
                float _rcp_11 = approx_rcp(1.0f + _exp2_11);
                float sig_11_55 = _rcp_11;
                float act_00_56 = x1_00_38 * sig_00_52;
                float act_01_57 = x1_01_39 * sig_01_53;
                float act_10_58 = x1_10_42 * sig_10_54;
                float act_11_59 = x1_11_43 * sig_11_55;
                float _min_20 = fminf(act_00_56, cl);
                act_00_56 = _min_20;
                float _min_21 = fminf(act_01_57, cl);
                act_01_57 = _min_21;
                float _min_22 = fminf(act_10_58, cl);
                act_10_58 = _min_22;
                float _min_23 = fminf(act_11_59, cl);
                act_11_59 = _min_23;
                float y_00_60 = lin_00_48 * act_00_56;
                float y_10_61 = lin_10_50 * act_10_58;
                float y_01_62 = lin_01_49 * act_01_57;
                float y_11_63 = lin_11_51 * act_11_59;
                quad[0] = y_00_60 * sc;
                quad[1] = y_10_61 * sc;
                quad[2] = y_01_62 * sc;
                quad[3] = y_11_63 * sc;
                uint32_t _fp8_2[1];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(quad[0]), "f"(quad[1]),
                                           "f"(quad[2]), "f"(quad[3]));
                    _fp8_2[0] = _packed;
                }
                int off0_64 = local_token0_34 * 64 + base_row;
                int off1_65 = local_token1_35 * 64 + base_row;
                int swz0_66 = off0_64 ^ (off0_64 >> 7 & 3) << 4;
                int swz1_67 = off1_65 ^ (off1_65 >> 7 & 3) << 4;
                epi_staging[swz0_66 >> 1] = _fp8_2[0] & 65535;
                epi_staging[swz1_67 >> 1] = _fp8_2[0] >> 16;
                int local_token0_68 = lane_1 % 4 * 2 + 24;
                int local_token1_69 = local_token0_68 + 1;
                float x0_00_70 = _tmem_load_0[12];
                float x0_01_71 = _tmem_load_0[13];
                float x1_00_72 = _tmem_load_0[14];
                float x1_01_73 = _tmem_load_0[15];
                float x0_10_74 = _tmem_load_1[12];
                float x0_11_75 = _tmem_load_1[13];
                float x1_10_76 = _tmem_load_1[14];
                float x1_11_77 = _tmem_load_1[15];
                float _max_12 = max_noftz(x0_00_70, neg_cl);
                float _min_24 = fminf(_max_12, cl);
                float x0c_00_78 = _min_24;
                float _max_13 = max_noftz(x0_01_71, neg_cl);
                float _min_25 = fminf(_max_13, cl);
                float x0c_01_79 = _min_25;
                float _max_14 = max_noftz(x0_10_74, neg_cl);
                float _min_26 = fminf(_max_14, cl);
                float x0c_10_80 = _min_26;
                float _max_15 = max_noftz(x0_11_75, neg_cl);
                float _min_27 = fminf(_max_15, cl);
                float x0c_11_81 = _min_27;
                float _fma_12 = __fmaf_rn(x0c_00_78, sg, 0.0f);
                float lin_00_82 = _fma_12;
                float _fma_13 = __fmaf_rn(x0c_01_79, sg, 0.0f);
                float lin_01_83 = _fma_13;
                float _fma_14 = __fmaf_rn(x0c_10_80, sg, 0.0f);
                float lin_10_84 = _fma_14;
                float _fma_15 = __fmaf_rn(x0c_11_81, sg, 0.0f);
                float lin_11_85 = _fma_15;
                float _exp2_12 = approx_exp2(-(x1_00_72 * fused));
                float _rcp_12 = approx_rcp(1.0f + _exp2_12);
                float sig_00_86 = _rcp_12;
                float _exp2_13 = approx_exp2(-(x1_01_73 * fused));
                float _rcp_13 = approx_rcp(1.0f + _exp2_13);
                float sig_01_87 = _rcp_13;
                float _exp2_14 = approx_exp2(-(x1_10_76 * fused));
                float _rcp_14 = approx_rcp(1.0f + _exp2_14);
                float sig_10_88 = _rcp_14;
                float _exp2_15 = approx_exp2(-(x1_11_77 * fused));
                float _rcp_15 = approx_rcp(1.0f + _exp2_15);
                float sig_11_89 = _rcp_15;
                float act_00_90 = x1_00_72 * sig_00_86;
                float act_01_91 = x1_01_73 * sig_01_87;
                float act_10_92 = x1_10_76 * sig_10_88;
                float act_11_93 = x1_11_77 * sig_11_89;
                float _min_28 = fminf(act_00_90, cl);
                act_00_90 = _min_28;
                float _min_29 = fminf(act_01_91, cl);
                act_01_91 = _min_29;
                float _min_30 = fminf(act_10_92, cl);
                act_10_92 = _min_30;
                float _min_31 = fminf(act_11_93, cl);
                act_11_93 = _min_31;
                float y_00_94 = lin_00_82 * act_00_90;
                float y_10_95 = lin_10_84 * act_10_92;
                float y_01_96 = lin_01_83 * act_01_91;
                float y_11_97 = lin_11_85 * act_11_93;
                quad[0] = y_00_94 * sc;
                quad[1] = y_10_95 * sc;
                quad[2] = y_01_96 * sc;
                quad[3] = y_11_97 * sc;
                uint32_t _fp8_3[1];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(quad[0]), "f"(quad[1]),
                                           "f"(quad[2]), "f"(quad[3]));
                    _fp8_3[0] = _packed;
                }
                int off0_98 = local_token0_68 * 64 + base_row;
                int off1_99 = local_token1_69 * 64 + base_row;
                int swz0_100 = off0_98 ^ (off0_98 >> 7 & 3) << 4;
                int swz1_101 = off1_99 ^ (off1_99 >> 7 & 3) << 4;
                epi_staging[swz0_100 >> 1] = _fp8_3[0] & 65535;
                epi_staging[swz1_101 >> 1] = _fp8_3[0] >> 16;
                int local_token0_102 = lane_1 % 4 * 2 + 32;
                int local_token1_103 = local_token0_102 + 1;
                float x0_00_104 = _tmem_load_0[16];
                float x0_01_105 = _tmem_load_0[17];
                float x1_00_106 = _tmem_load_0[18];
                float x1_01_107 = _tmem_load_0[19];
                float x0_10_108 = _tmem_load_1[16];
                float x0_11_109 = _tmem_load_1[17];
                float x1_10_110 = _tmem_load_1[18];
                float x1_11_111 = _tmem_load_1[19];
                float _max_16 = max_noftz(x0_00_104, neg_cl);
                float _min_32 = fminf(_max_16, cl);
                float x0c_00_112 = _min_32;
                float _max_17 = max_noftz(x0_01_105, neg_cl);
                float _min_33 = fminf(_max_17, cl);
                float x0c_01_113 = _min_33;
                float _max_18 = max_noftz(x0_10_108, neg_cl);
                float _min_34 = fminf(_max_18, cl);
                float x0c_10_114 = _min_34;
                float _max_19 = max_noftz(x0_11_109, neg_cl);
                float _min_35 = fminf(_max_19, cl);
                float x0c_11_115 = _min_35;
                float _fma_16 = __fmaf_rn(x0c_00_112, sg, 0.0f);
                float lin_00_116 = _fma_16;
                float _fma_17 = __fmaf_rn(x0c_01_113, sg, 0.0f);
                float lin_01_117 = _fma_17;
                float _fma_18 = __fmaf_rn(x0c_10_114, sg, 0.0f);
                float lin_10_118 = _fma_18;
                float _fma_19 = __fmaf_rn(x0c_11_115, sg, 0.0f);
                float lin_11_119 = _fma_19;
                float _exp2_16 = approx_exp2(-(x1_00_106 * fused));
                float _rcp_16 = approx_rcp(1.0f + _exp2_16);
                float sig_00_120 = _rcp_16;
                float _exp2_17 = approx_exp2(-(x1_01_107 * fused));
                float _rcp_17 = approx_rcp(1.0f + _exp2_17);
                float sig_01_121 = _rcp_17;
                float _exp2_18 = approx_exp2(-(x1_10_110 * fused));
                float _rcp_18 = approx_rcp(1.0f + _exp2_18);
                float sig_10_122 = _rcp_18;
                float _exp2_19 = approx_exp2(-(x1_11_111 * fused));
                float _rcp_19 = approx_rcp(1.0f + _exp2_19);
                float sig_11_123 = _rcp_19;
                float act_00_124 = x1_00_106 * sig_00_120;
                float act_01_125 = x1_01_107 * sig_01_121;
                float act_10_126 = x1_10_110 * sig_10_122;
                float act_11_127 = x1_11_111 * sig_11_123;
                float _min_36 = fminf(act_00_124, cl);
                act_00_124 = _min_36;
                float _min_37 = fminf(act_01_125, cl);
                act_01_125 = _min_37;
                float _min_38 = fminf(act_10_126, cl);
                act_10_126 = _min_38;
                float _min_39 = fminf(act_11_127, cl);
                act_11_127 = _min_39;
                float y_00_128 = lin_00_116 * act_00_124;
                float y_10_129 = lin_10_118 * act_10_126;
                float y_01_130 = lin_01_117 * act_01_125;
                float y_11_131 = lin_11_119 * act_11_127;
                quad[0] = y_00_128 * sc;
                quad[1] = y_10_129 * sc;
                quad[2] = y_01_130 * sc;
                quad[3] = y_11_131 * sc;
                uint32_t _fp8_4[1];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(quad[0]), "f"(quad[1]),
                                           "f"(quad[2]), "f"(quad[3]));
                    _fp8_4[0] = _packed;
                }
                int off0_132 = local_token0_102 * 64 + base_row;
                int off1_133 = local_token1_103 * 64 + base_row;
                int swz0_134 = off0_132 ^ (off0_132 >> 7 & 3) << 4;
                int swz1_135 = off1_133 ^ (off1_133 >> 7 & 3) << 4;
                epi_staging[swz0_134 >> 1] = _fp8_4[0] & 65535;
                epi_staging[swz1_135 >> 1] = _fp8_4[0] >> 16;
                int local_token0_136 = lane_1 % 4 * 2 + 40;
                int local_token1_137 = local_token0_136 + 1;
                float x0_00_138 = _tmem_load_0[20];
                float x0_01_139 = _tmem_load_0[21];
                float x1_00_140 = _tmem_load_0[22];
                float x1_01_141 = _tmem_load_0[23];
                float x0_10_142 = _tmem_load_1[20];
                float x0_11_143 = _tmem_load_1[21];
                float x1_10_144 = _tmem_load_1[22];
                float x1_11_145 = _tmem_load_1[23];
                float _max_20 = max_noftz(x0_00_138, neg_cl);
                float _min_40 = fminf(_max_20, cl);
                float x0c_00_146 = _min_40;
                float _max_21 = max_noftz(x0_01_139, neg_cl);
                float _min_41 = fminf(_max_21, cl);
                float x0c_01_147 = _min_41;
                float _max_22 = max_noftz(x0_10_142, neg_cl);
                float _min_42 = fminf(_max_22, cl);
                float x0c_10_148 = _min_42;
                float _max_23 = max_noftz(x0_11_143, neg_cl);
                float _min_43 = fminf(_max_23, cl);
                float x0c_11_149 = _min_43;
                float _fma_20 = __fmaf_rn(x0c_00_146, sg, 0.0f);
                float lin_00_150 = _fma_20;
                float _fma_21 = __fmaf_rn(x0c_01_147, sg, 0.0f);
                float lin_01_151 = _fma_21;
                float _fma_22 = __fmaf_rn(x0c_10_148, sg, 0.0f);
                float lin_10_152 = _fma_22;
                float _fma_23 = __fmaf_rn(x0c_11_149, sg, 0.0f);
                float lin_11_153 = _fma_23;
                float _exp2_20 = approx_exp2(-(x1_00_140 * fused));
                float _rcp_20 = approx_rcp(1.0f + _exp2_20);
                float sig_00_154 = _rcp_20;
                float _exp2_21 = approx_exp2(-(x1_01_141 * fused));
                float _rcp_21 = approx_rcp(1.0f + _exp2_21);
                float sig_01_155 = _rcp_21;
                float _exp2_22 = approx_exp2(-(x1_10_144 * fused));
                float _rcp_22 = approx_rcp(1.0f + _exp2_22);
                float sig_10_156 = _rcp_22;
                float _exp2_23 = approx_exp2(-(x1_11_145 * fused));
                float _rcp_23 = approx_rcp(1.0f + _exp2_23);
                float sig_11_157 = _rcp_23;
                float act_00_158 = x1_00_140 * sig_00_154;
                float act_01_159 = x1_01_141 * sig_01_155;
                float act_10_160 = x1_10_144 * sig_10_156;
                float act_11_161 = x1_11_145 * sig_11_157;
                float _min_44 = fminf(act_00_158, cl);
                act_00_158 = _min_44;
                float _min_45 = fminf(act_01_159, cl);
                act_01_159 = _min_45;
                float _min_46 = fminf(act_10_160, cl);
                act_10_160 = _min_46;
                float _min_47 = fminf(act_11_161, cl);
                act_11_161 = _min_47;
                float y_00_162 = lin_00_150 * act_00_158;
                float y_10_163 = lin_10_152 * act_10_160;
                float y_01_164 = lin_01_151 * act_01_159;
                float y_11_165 = lin_11_153 * act_11_161;
                quad[0] = y_00_162 * sc;
                quad[1] = y_10_163 * sc;
                quad[2] = y_01_164 * sc;
                quad[3] = y_11_165 * sc;
                uint32_t _fp8_5[1];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(quad[0]), "f"(quad[1]),
                                           "f"(quad[2]), "f"(quad[3]));
                    _fp8_5[0] = _packed;
                }
                int off0_166 = local_token0_136 * 64 + base_row;
                int off1_167 = local_token1_137 * 64 + base_row;
                int swz0_168 = off0_166 ^ (off0_166 >> 7 & 3) << 4;
                int swz1_169 = off1_167 ^ (off1_167 >> 7 & 3) << 4;
                epi_staging[swz0_168 >> 1] = _fp8_5[0] & 65535;
                epi_staging[swz1_169 >> 1] = _fp8_5[0] >> 16;
                int local_token0_170 = lane_1 % 4 * 2 + 48;
                int local_token1_171 = local_token0_170 + 1;
                float x0_00_172 = _tmem_load_0[24];
                float x0_01_173 = _tmem_load_0[25];
                float x1_00_174 = _tmem_load_0[26];
                float x1_01_175 = _tmem_load_0[27];
                float x0_10_176 = _tmem_load_1[24];
                float x0_11_177 = _tmem_load_1[25];
                float x1_10_178 = _tmem_load_1[26];
                float x1_11_179 = _tmem_load_1[27];
                float _max_24 = max_noftz(x0_00_172, neg_cl);
                float _min_48 = fminf(_max_24, cl);
                float x0c_00_180 = _min_48;
                float _max_25 = max_noftz(x0_01_173, neg_cl);
                float _min_49 = fminf(_max_25, cl);
                float x0c_01_181 = _min_49;
                float _max_26 = max_noftz(x0_10_176, neg_cl);
                float _min_50 = fminf(_max_26, cl);
                float x0c_10_182 = _min_50;
                float _max_27 = max_noftz(x0_11_177, neg_cl);
                float _min_51 = fminf(_max_27, cl);
                float x0c_11_183 = _min_51;
                float _fma_24 = __fmaf_rn(x0c_00_180, sg, 0.0f);
                float lin_00_184 = _fma_24;
                float _fma_25 = __fmaf_rn(x0c_01_181, sg, 0.0f);
                float lin_01_185 = _fma_25;
                float _fma_26 = __fmaf_rn(x0c_10_182, sg, 0.0f);
                float lin_10_186 = _fma_26;
                float _fma_27 = __fmaf_rn(x0c_11_183, sg, 0.0f);
                float lin_11_187 = _fma_27;
                float _exp2_24 = approx_exp2(-(x1_00_174 * fused));
                float _rcp_24 = approx_rcp(1.0f + _exp2_24);
                float sig_00_188 = _rcp_24;
                float _exp2_25 = approx_exp2(-(x1_01_175 * fused));
                float _rcp_25 = approx_rcp(1.0f + _exp2_25);
                float sig_01_189 = _rcp_25;
                float _exp2_26 = approx_exp2(-(x1_10_178 * fused));
                float _rcp_26 = approx_rcp(1.0f + _exp2_26);
                float sig_10_190 = _rcp_26;
                float _exp2_27 = approx_exp2(-(x1_11_179 * fused));
                float _rcp_27 = approx_rcp(1.0f + _exp2_27);
                float sig_11_191 = _rcp_27;
                float act_00_192 = x1_00_174 * sig_00_188;
                float act_01_193 = x1_01_175 * sig_01_189;
                float act_10_194 = x1_10_178 * sig_10_190;
                float act_11_195 = x1_11_179 * sig_11_191;
                float _min_52 = fminf(act_00_192, cl);
                act_00_192 = _min_52;
                float _min_53 = fminf(act_01_193, cl);
                act_01_193 = _min_53;
                float _min_54 = fminf(act_10_194, cl);
                act_10_194 = _min_54;
                float _min_55 = fminf(act_11_195, cl);
                act_11_195 = _min_55;
                float y_00_196 = lin_00_184 * act_00_192;
                float y_10_197 = lin_10_186 * act_10_194;
                float y_01_198 = lin_01_185 * act_01_193;
                float y_11_199 = lin_11_187 * act_11_195;
                quad[0] = y_00_196 * sc;
                quad[1] = y_10_197 * sc;
                quad[2] = y_01_198 * sc;
                quad[3] = y_11_199 * sc;
                uint32_t _fp8_6[1];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(quad[0]), "f"(quad[1]),
                                           "f"(quad[2]), "f"(quad[3]));
                    _fp8_6[0] = _packed;
                }
                int off0_200 = local_token0_170 * 64 + base_row;
                int off1_201 = local_token1_171 * 64 + base_row;
                int swz0_202 = off0_200 ^ (off0_200 >> 7 & 3) << 4;
                int swz1_203 = off1_201 ^ (off1_201 >> 7 & 3) << 4;
                epi_staging[swz0_202 >> 1] = _fp8_6[0] & 65535;
                epi_staging[swz1_203 >> 1] = _fp8_6[0] >> 16;
                int local_token0_204 = lane_1 % 4 * 2 + 56;
                int local_token1_205 = local_token0_204 + 1;
                float x0_00_206 = _tmem_load_0[28];
                float x0_01_207 = _tmem_load_0[29];
                float x1_00_208 = _tmem_load_0[30];
                float x1_01_209 = _tmem_load_0[31];
                float x0_10_210 = _tmem_load_1[28];
                float x0_11_211 = _tmem_load_1[29];
                float x1_10_212 = _tmem_load_1[30];
                float x1_11_213 = _tmem_load_1[31];
                float _max_28 = max_noftz(x0_00_206, neg_cl);
                float _min_56 = fminf(_max_28, cl);
                float x0c_00_214 = _min_56;
                float _max_29 = max_noftz(x0_01_207, neg_cl);
                float _min_57 = fminf(_max_29, cl);
                float x0c_01_215 = _min_57;
                float _max_30 = max_noftz(x0_10_210, neg_cl);
                float _min_58 = fminf(_max_30, cl);
                float x0c_10_216 = _min_58;
                float _max_31 = max_noftz(x0_11_211, neg_cl);
                float _min_59 = fminf(_max_31, cl);
                float x0c_11_217 = _min_59;
                float _fma_28 = __fmaf_rn(x0c_00_214, sg, 0.0f);
                float lin_00_218 = _fma_28;
                float _fma_29 = __fmaf_rn(x0c_01_215, sg, 0.0f);
                float lin_01_219 = _fma_29;
                float _fma_30 = __fmaf_rn(x0c_10_216, sg, 0.0f);
                float lin_10_220 = _fma_30;
                float _fma_31 = __fmaf_rn(x0c_11_217, sg, 0.0f);
                float lin_11_221 = _fma_31;
                float _exp2_28 = approx_exp2(-(x1_00_208 * fused));
                float _rcp_28 = approx_rcp(1.0f + _exp2_28);
                float sig_00_222 = _rcp_28;
                float _exp2_29 = approx_exp2(-(x1_01_209 * fused));
                float _rcp_29 = approx_rcp(1.0f + _exp2_29);
                float sig_01_223 = _rcp_29;
                float _exp2_30 = approx_exp2(-(x1_10_212 * fused));
                float _rcp_30 = approx_rcp(1.0f + _exp2_30);
                float sig_10_224 = _rcp_30;
                float _exp2_31 = approx_exp2(-(x1_11_213 * fused));
                float _rcp_31 = approx_rcp(1.0f + _exp2_31);
                float sig_11_225 = _rcp_31;
                float act_00_226 = x1_00_208 * sig_00_222;
                float act_01_227 = x1_01_209 * sig_01_223;
                float act_10_228 = x1_10_212 * sig_10_224;
                float act_11_229 = x1_11_213 * sig_11_225;
                float _min_60 = fminf(act_00_226, cl);
                act_00_226 = _min_60;
                float _min_61 = fminf(act_01_227, cl);
                act_01_227 = _min_61;
                float _min_62 = fminf(act_10_228, cl);
                act_10_228 = _min_62;
                float _min_63 = fminf(act_11_229, cl);
                act_11_229 = _min_63;
                float y_00_230 = lin_00_218 * act_00_226;
                float y_10_231 = lin_10_220 * act_10_228;
                float y_01_232 = lin_01_219 * act_01_227;
                float y_11_233 = lin_11_221 * act_11_229;
                quad[0] = y_00_230 * sc;
                quad[1] = y_10_231 * sc;
                quad[2] = y_01_232 * sc;
                quad[3] = y_11_233 * sc;
                uint32_t _fp8_7[1];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(quad[0]), "f"(quad[1]),
                                           "f"(quad[2]), "f"(quad[3]));
                    _fp8_7[0] = _packed;
                }
                int off0_234 = local_token0_204 * 64 + base_row;
                int off1_235 = local_token1_205 * 64 + base_row;
                int swz0_236 = off0_234 ^ (off0_234 >> 7 & 3) << 4;
                int swz1_237 = off1_235 ^ (off1_235 >> 7 & 3) << 4;
                epi_staging[swz0_236 >> 1] = _fp8_7[0] & 65535;
                epi_staging[swz1_237 >> 1] = _fp8_7[0] >> 16;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                if (warp == 0) {
                    if (elect_sync()) {
                        int padding_rows = (128 - valid_rows % 128) % 128;
                        tma_store_4d((&C), m_tile * 64, padding_rows, 1073741824, n_tile * 128 - (unsigned int)padding_rows + 1073741824, epi_staging_addr);
                    }
                }
                asm volatile("cp.async.bulk.commit_group;");
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                int acc_offset_238 = acc_stage * 128 + 64;
                float _tmem_load_2[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[31]))
                    : "r"(taddr + (unsigned int)acc_offset_238));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float _tmem_load_3[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[31]))
                    : "r"(taddr + 1048576 + (unsigned int)acc_offset_238));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int base_row_239 = warp_0 * 16 + lane_1 / 4 * 2;
                asm volatile("cp.async.bulk.wait_group.read 0;");
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                int local_token0_240 = lane_1 % 4 * 2;
                int local_token1_241 = local_token0_240 + 1;
                float x0_00_242 = _tmem_load_2[0];
                float x0_01_243 = _tmem_load_2[1];
                float x1_00_244 = _tmem_load_2[2];
                float x1_01_245 = _tmem_load_2[3];
                float x0_10_246 = _tmem_load_3[0];
                float x0_11_247 = _tmem_load_3[1];
                float x1_10_248 = _tmem_load_3[2];
                float x1_11_249 = _tmem_load_3[3];
                float _max_32 = max_noftz(x0_00_242, neg_cl);
                float _min_64 = fminf(_max_32, cl);
                float x0c_00_250 = _min_64;
                float _max_33 = max_noftz(x0_01_243, neg_cl);
                float _min_65 = fminf(_max_33, cl);
                float x0c_01_251 = _min_65;
                float _max_34 = max_noftz(x0_10_246, neg_cl);
                float _min_66 = fminf(_max_34, cl);
                float x0c_10_252 = _min_66;
                float _max_35 = max_noftz(x0_11_247, neg_cl);
                float _min_67 = fminf(_max_35, cl);
                float x0c_11_253 = _min_67;
                float _fma_32 = __fmaf_rn(x0c_00_250, sg, 0.0f);
                float lin_00_254 = _fma_32;
                float _fma_33 = __fmaf_rn(x0c_01_251, sg, 0.0f);
                float lin_01_255 = _fma_33;
                float _fma_34 = __fmaf_rn(x0c_10_252, sg, 0.0f);
                float lin_10_256 = _fma_34;
                float _fma_35 = __fmaf_rn(x0c_11_253, sg, 0.0f);
                float lin_11_257 = _fma_35;
                float _exp2_32 = approx_exp2(-(x1_00_244 * fused));
                float _rcp_32 = approx_rcp(1.0f + _exp2_32);
                float sig_00_258 = _rcp_32;
                float _exp2_33 = approx_exp2(-(x1_01_245 * fused));
                float _rcp_33 = approx_rcp(1.0f + _exp2_33);
                float sig_01_259 = _rcp_33;
                float _exp2_34 = approx_exp2(-(x1_10_248 * fused));
                float _rcp_34 = approx_rcp(1.0f + _exp2_34);
                float sig_10_260 = _rcp_34;
                float _exp2_35 = approx_exp2(-(x1_11_249 * fused));
                float _rcp_35 = approx_rcp(1.0f + _exp2_35);
                float sig_11_261 = _rcp_35;
                float act_00_262 = x1_00_244 * sig_00_258;
                float act_01_263 = x1_01_245 * sig_01_259;
                float act_10_264 = x1_10_248 * sig_10_260;
                float act_11_265 = x1_11_249 * sig_11_261;
                float _min_68 = fminf(act_00_262, cl);
                act_00_262 = _min_68;
                float _min_69 = fminf(act_01_263, cl);
                act_01_263 = _min_69;
                float _min_70 = fminf(act_10_264, cl);
                act_10_264 = _min_70;
                float _min_71 = fminf(act_11_265, cl);
                act_11_265 = _min_71;
                float y_00_266 = lin_00_254 * act_00_262;
                float y_10_267 = lin_10_256 * act_10_264;
                float y_01_268 = lin_01_255 * act_01_263;
                float y_11_269 = lin_11_257 * act_11_265;
                quad[0] = y_00_266 * sc;
                quad[1] = y_10_267 * sc;
                quad[2] = y_01_268 * sc;
                quad[3] = y_11_269 * sc;
                uint32_t _fp8_8[1];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(quad[0]), "f"(quad[1]),
                                           "f"(quad[2]), "f"(quad[3]));
                    _fp8_8[0] = _packed;
                }
                int off0_270 = local_token0_240 * 64 + base_row_239;
                int off1_271 = local_token1_241 * 64 + base_row_239;
                int swz0_272 = off0_270 ^ (off0_270 >> 7 & 3) << 4;
                int swz1_273 = off1_271 ^ (off1_271 >> 7 & 3) << 4;
                epi_staging[swz0_272 >> 1] = _fp8_8[0] & 65535;
                epi_staging[swz1_273 >> 1] = _fp8_8[0] >> 16;
                int local_token0_274 = lane_1 % 4 * 2 + 8;
                int local_token1_275 = local_token0_274 + 1;
                float x0_00_276 = _tmem_load_2[4];
                float x0_01_277 = _tmem_load_2[5];
                float x1_00_278 = _tmem_load_2[6];
                float x1_01_279 = _tmem_load_2[7];
                float x0_10_280 = _tmem_load_3[4];
                float x0_11_281 = _tmem_load_3[5];
                float x1_10_282 = _tmem_load_3[6];
                float x1_11_283 = _tmem_load_3[7];
                float _max_36 = max_noftz(x0_00_276, neg_cl);
                float _min_72 = fminf(_max_36, cl);
                float x0c_00_284 = _min_72;
                float _max_37 = max_noftz(x0_01_277, neg_cl);
                float _min_73 = fminf(_max_37, cl);
                float x0c_01_285 = _min_73;
                float _max_38 = max_noftz(x0_10_280, neg_cl);
                float _min_74 = fminf(_max_38, cl);
                float x0c_10_286 = _min_74;
                float _max_39 = max_noftz(x0_11_281, neg_cl);
                float _min_75 = fminf(_max_39, cl);
                float x0c_11_287 = _min_75;
                float _fma_36 = __fmaf_rn(x0c_00_284, sg, 0.0f);
                float lin_00_288 = _fma_36;
                float _fma_37 = __fmaf_rn(x0c_01_285, sg, 0.0f);
                float lin_01_289 = _fma_37;
                float _fma_38 = __fmaf_rn(x0c_10_286, sg, 0.0f);
                float lin_10_290 = _fma_38;
                float _fma_39 = __fmaf_rn(x0c_11_287, sg, 0.0f);
                float lin_11_291 = _fma_39;
                float _exp2_36 = approx_exp2(-(x1_00_278 * fused));
                float _rcp_36 = approx_rcp(1.0f + _exp2_36);
                float sig_00_292 = _rcp_36;
                float _exp2_37 = approx_exp2(-(x1_01_279 * fused));
                float _rcp_37 = approx_rcp(1.0f + _exp2_37);
                float sig_01_293 = _rcp_37;
                float _exp2_38 = approx_exp2(-(x1_10_282 * fused));
                float _rcp_38 = approx_rcp(1.0f + _exp2_38);
                float sig_10_294 = _rcp_38;
                float _exp2_39 = approx_exp2(-(x1_11_283 * fused));
                float _rcp_39 = approx_rcp(1.0f + _exp2_39);
                float sig_11_295 = _rcp_39;
                float act_00_296 = x1_00_278 * sig_00_292;
                float act_01_297 = x1_01_279 * sig_01_293;
                float act_10_298 = x1_10_282 * sig_10_294;
                float act_11_299 = x1_11_283 * sig_11_295;
                float _min_76 = fminf(act_00_296, cl);
                act_00_296 = _min_76;
                float _min_77 = fminf(act_01_297, cl);
                act_01_297 = _min_77;
                float _min_78 = fminf(act_10_298, cl);
                act_10_298 = _min_78;
                float _min_79 = fminf(act_11_299, cl);
                act_11_299 = _min_79;
                float y_00_300 = lin_00_288 * act_00_296;
                float y_10_301 = lin_10_290 * act_10_298;
                float y_01_302 = lin_01_289 * act_01_297;
                float y_11_303 = lin_11_291 * act_11_299;
                quad[0] = y_00_300 * sc;
                quad[1] = y_10_301 * sc;
                quad[2] = y_01_302 * sc;
                quad[3] = y_11_303 * sc;
                uint32_t _fp8_9[1];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(quad[0]), "f"(quad[1]),
                                           "f"(quad[2]), "f"(quad[3]));
                    _fp8_9[0] = _packed;
                }
                int off0_304 = local_token0_274 * 64 + base_row_239;
                int off1_305 = local_token1_275 * 64 + base_row_239;
                int swz0_306 = off0_304 ^ (off0_304 >> 7 & 3) << 4;
                int swz1_307 = off1_305 ^ (off1_305 >> 7 & 3) << 4;
                epi_staging[swz0_306 >> 1] = _fp8_9[0] & 65535;
                epi_staging[swz1_307 >> 1] = _fp8_9[0] >> 16;
                int local_token0_308 = lane_1 % 4 * 2 + 16;
                int local_token1_309 = local_token0_308 + 1;
                float x0_00_310 = _tmem_load_2[8];
                float x0_01_311 = _tmem_load_2[9];
                float x1_00_312 = _tmem_load_2[10];
                float x1_01_313 = _tmem_load_2[11];
                float x0_10_314 = _tmem_load_3[8];
                float x0_11_315 = _tmem_load_3[9];
                float x1_10_316 = _tmem_load_3[10];
                float x1_11_317 = _tmem_load_3[11];
                float _max_40 = max_noftz(x0_00_310, neg_cl);
                float _min_80 = fminf(_max_40, cl);
                float x0c_00_318 = _min_80;
                float _max_41 = max_noftz(x0_01_311, neg_cl);
                float _min_81 = fminf(_max_41, cl);
                float x0c_01_319 = _min_81;
                float _max_42 = max_noftz(x0_10_314, neg_cl);
                float _min_82 = fminf(_max_42, cl);
                float x0c_10_320 = _min_82;
                float _max_43 = max_noftz(x0_11_315, neg_cl);
                float _min_83 = fminf(_max_43, cl);
                float x0c_11_321 = _min_83;
                float _fma_40 = __fmaf_rn(x0c_00_318, sg, 0.0f);
                float lin_00_322 = _fma_40;
                float _fma_41 = __fmaf_rn(x0c_01_319, sg, 0.0f);
                float lin_01_323 = _fma_41;
                float _fma_42 = __fmaf_rn(x0c_10_320, sg, 0.0f);
                float lin_10_324 = _fma_42;
                float _fma_43 = __fmaf_rn(x0c_11_321, sg, 0.0f);
                float lin_11_325 = _fma_43;
                float _exp2_40 = approx_exp2(-(x1_00_312 * fused));
                float _rcp_40 = approx_rcp(1.0f + _exp2_40);
                float sig_00_326 = _rcp_40;
                float _exp2_41 = approx_exp2(-(x1_01_313 * fused));
                float _rcp_41 = approx_rcp(1.0f + _exp2_41);
                float sig_01_327 = _rcp_41;
                float _exp2_42 = approx_exp2(-(x1_10_316 * fused));
                float _rcp_42 = approx_rcp(1.0f + _exp2_42);
                float sig_10_328 = _rcp_42;
                float _exp2_43 = approx_exp2(-(x1_11_317 * fused));
                float _rcp_43 = approx_rcp(1.0f + _exp2_43);
                float sig_11_329 = _rcp_43;
                float act_00_330 = x1_00_312 * sig_00_326;
                float act_01_331 = x1_01_313 * sig_01_327;
                float act_10_332 = x1_10_316 * sig_10_328;
                float act_11_333 = x1_11_317 * sig_11_329;
                float _min_84 = fminf(act_00_330, cl);
                act_00_330 = _min_84;
                float _min_85 = fminf(act_01_331, cl);
                act_01_331 = _min_85;
                float _min_86 = fminf(act_10_332, cl);
                act_10_332 = _min_86;
                float _min_87 = fminf(act_11_333, cl);
                act_11_333 = _min_87;
                float y_00_334 = lin_00_322 * act_00_330;
                float y_10_335 = lin_10_324 * act_10_332;
                float y_01_336 = lin_01_323 * act_01_331;
                float y_11_337 = lin_11_325 * act_11_333;
                quad[0] = y_00_334 * sc;
                quad[1] = y_10_335 * sc;
                quad[2] = y_01_336 * sc;
                quad[3] = y_11_337 * sc;
                uint32_t _fp8_10[1];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(quad[0]), "f"(quad[1]),
                                           "f"(quad[2]), "f"(quad[3]));
                    _fp8_10[0] = _packed;
                }
                int off0_338 = local_token0_308 * 64 + base_row_239;
                int off1_339 = local_token1_309 * 64 + base_row_239;
                int swz0_340 = off0_338 ^ (off0_338 >> 7 & 3) << 4;
                int swz1_341 = off1_339 ^ (off1_339 >> 7 & 3) << 4;
                epi_staging[swz0_340 >> 1] = _fp8_10[0] & 65535;
                epi_staging[swz1_341 >> 1] = _fp8_10[0] >> 16;
                int local_token0_342 = lane_1 % 4 * 2 + 24;
                int local_token1_343 = local_token0_342 + 1;
                float x0_00_344 = _tmem_load_2[12];
                float x0_01_345 = _tmem_load_2[13];
                float x1_00_346 = _tmem_load_2[14];
                float x1_01_347 = _tmem_load_2[15];
                float x0_10_348 = _tmem_load_3[12];
                float x0_11_349 = _tmem_load_3[13];
                float x1_10_350 = _tmem_load_3[14];
                float x1_11_351 = _tmem_load_3[15];
                float _max_44 = max_noftz(x0_00_344, neg_cl);
                float _min_88 = fminf(_max_44, cl);
                float x0c_00_352 = _min_88;
                float _max_45 = max_noftz(x0_01_345, neg_cl);
                float _min_89 = fminf(_max_45, cl);
                float x0c_01_353 = _min_89;
                float _max_46 = max_noftz(x0_10_348, neg_cl);
                float _min_90 = fminf(_max_46, cl);
                float x0c_10_354 = _min_90;
                float _max_47 = max_noftz(x0_11_349, neg_cl);
                float _min_91 = fminf(_max_47, cl);
                float x0c_11_355 = _min_91;
                float _fma_44 = __fmaf_rn(x0c_00_352, sg, 0.0f);
                float lin_00_356 = _fma_44;
                float _fma_45 = __fmaf_rn(x0c_01_353, sg, 0.0f);
                float lin_01_357 = _fma_45;
                float _fma_46 = __fmaf_rn(x0c_10_354, sg, 0.0f);
                float lin_10_358 = _fma_46;
                float _fma_47 = __fmaf_rn(x0c_11_355, sg, 0.0f);
                float lin_11_359 = _fma_47;
                float _exp2_44 = approx_exp2(-(x1_00_346 * fused));
                float _rcp_44 = approx_rcp(1.0f + _exp2_44);
                float sig_00_360 = _rcp_44;
                float _exp2_45 = approx_exp2(-(x1_01_347 * fused));
                float _rcp_45 = approx_rcp(1.0f + _exp2_45);
                float sig_01_361 = _rcp_45;
                float _exp2_46 = approx_exp2(-(x1_10_350 * fused));
                float _rcp_46 = approx_rcp(1.0f + _exp2_46);
                float sig_10_362 = _rcp_46;
                float _exp2_47 = approx_exp2(-(x1_11_351 * fused));
                float _rcp_47 = approx_rcp(1.0f + _exp2_47);
                float sig_11_363 = _rcp_47;
                float act_00_364 = x1_00_346 * sig_00_360;
                float act_01_365 = x1_01_347 * sig_01_361;
                float act_10_366 = x1_10_350 * sig_10_362;
                float act_11_367 = x1_11_351 * sig_11_363;
                float _min_92 = fminf(act_00_364, cl);
                act_00_364 = _min_92;
                float _min_93 = fminf(act_01_365, cl);
                act_01_365 = _min_93;
                float _min_94 = fminf(act_10_366, cl);
                act_10_366 = _min_94;
                float _min_95 = fminf(act_11_367, cl);
                act_11_367 = _min_95;
                float y_00_368 = lin_00_356 * act_00_364;
                float y_10_369 = lin_10_358 * act_10_366;
                float y_01_370 = lin_01_357 * act_01_365;
                float y_11_371 = lin_11_359 * act_11_367;
                quad[0] = y_00_368 * sc;
                quad[1] = y_10_369 * sc;
                quad[2] = y_01_370 * sc;
                quad[3] = y_11_371 * sc;
                uint32_t _fp8_11[1];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(quad[0]), "f"(quad[1]),
                                           "f"(quad[2]), "f"(quad[3]));
                    _fp8_11[0] = _packed;
                }
                int off0_372 = local_token0_342 * 64 + base_row_239;
                int off1_373 = local_token1_343 * 64 + base_row_239;
                int swz0_374 = off0_372 ^ (off0_372 >> 7 & 3) << 4;
                int swz1_375 = off1_373 ^ (off1_373 >> 7 & 3) << 4;
                epi_staging[swz0_374 >> 1] = _fp8_11[0] & 65535;
                epi_staging[swz1_375 >> 1] = _fp8_11[0] >> 16;
                int local_token0_376 = lane_1 % 4 * 2 + 32;
                int local_token1_377 = local_token0_376 + 1;
                float x0_00_378 = _tmem_load_2[16];
                float x0_01_379 = _tmem_load_2[17];
                float x1_00_380 = _tmem_load_2[18];
                float x1_01_381 = _tmem_load_2[19];
                float x0_10_382 = _tmem_load_3[16];
                float x0_11_383 = _tmem_load_3[17];
                float x1_10_384 = _tmem_load_3[18];
                float x1_11_385 = _tmem_load_3[19];
                float _max_48 = max_noftz(x0_00_378, neg_cl);
                float _min_96 = fminf(_max_48, cl);
                float x0c_00_386 = _min_96;
                float _max_49 = max_noftz(x0_01_379, neg_cl);
                float _min_97 = fminf(_max_49, cl);
                float x0c_01_387 = _min_97;
                float _max_50 = max_noftz(x0_10_382, neg_cl);
                float _min_98 = fminf(_max_50, cl);
                float x0c_10_388 = _min_98;
                float _max_51 = max_noftz(x0_11_383, neg_cl);
                float _min_99 = fminf(_max_51, cl);
                float x0c_11_389 = _min_99;
                float _fma_48 = __fmaf_rn(x0c_00_386, sg, 0.0f);
                float lin_00_390 = _fma_48;
                float _fma_49 = __fmaf_rn(x0c_01_387, sg, 0.0f);
                float lin_01_391 = _fma_49;
                float _fma_50 = __fmaf_rn(x0c_10_388, sg, 0.0f);
                float lin_10_392 = _fma_50;
                float _fma_51 = __fmaf_rn(x0c_11_389, sg, 0.0f);
                float lin_11_393 = _fma_51;
                float _exp2_48 = approx_exp2(-(x1_00_380 * fused));
                float _rcp_48 = approx_rcp(1.0f + _exp2_48);
                float sig_00_394 = _rcp_48;
                float _exp2_49 = approx_exp2(-(x1_01_381 * fused));
                float _rcp_49 = approx_rcp(1.0f + _exp2_49);
                float sig_01_395 = _rcp_49;
                float _exp2_50 = approx_exp2(-(x1_10_384 * fused));
                float _rcp_50 = approx_rcp(1.0f + _exp2_50);
                float sig_10_396 = _rcp_50;
                float _exp2_51 = approx_exp2(-(x1_11_385 * fused));
                float _rcp_51 = approx_rcp(1.0f + _exp2_51);
                float sig_11_397 = _rcp_51;
                float act_00_398 = x1_00_380 * sig_00_394;
                float act_01_399 = x1_01_381 * sig_01_395;
                float act_10_400 = x1_10_384 * sig_10_396;
                float act_11_401 = x1_11_385 * sig_11_397;
                float _min_100 = fminf(act_00_398, cl);
                act_00_398 = _min_100;
                float _min_101 = fminf(act_01_399, cl);
                act_01_399 = _min_101;
                float _min_102 = fminf(act_10_400, cl);
                act_10_400 = _min_102;
                float _min_103 = fminf(act_11_401, cl);
                act_11_401 = _min_103;
                float y_00_402 = lin_00_390 * act_00_398;
                float y_10_403 = lin_10_392 * act_10_400;
                float y_01_404 = lin_01_391 * act_01_399;
                float y_11_405 = lin_11_393 * act_11_401;
                quad[0] = y_00_402 * sc;
                quad[1] = y_10_403 * sc;
                quad[2] = y_01_404 * sc;
                quad[3] = y_11_405 * sc;
                uint32_t _fp8_12[1];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(quad[0]), "f"(quad[1]),
                                           "f"(quad[2]), "f"(quad[3]));
                    _fp8_12[0] = _packed;
                }
                int off0_406 = local_token0_376 * 64 + base_row_239;
                int off1_407 = local_token1_377 * 64 + base_row_239;
                int swz0_408 = off0_406 ^ (off0_406 >> 7 & 3) << 4;
                int swz1_409 = off1_407 ^ (off1_407 >> 7 & 3) << 4;
                epi_staging[swz0_408 >> 1] = _fp8_12[0] & 65535;
                epi_staging[swz1_409 >> 1] = _fp8_12[0] >> 16;
                int local_token0_410 = lane_1 % 4 * 2 + 40;
                int local_token1_411 = local_token0_410 + 1;
                float x0_00_412 = _tmem_load_2[20];
                float x0_01_413 = _tmem_load_2[21];
                float x1_00_414 = _tmem_load_2[22];
                float x1_01_415 = _tmem_load_2[23];
                float x0_10_416 = _tmem_load_3[20];
                float x0_11_417 = _tmem_load_3[21];
                float x1_10_418 = _tmem_load_3[22];
                float x1_11_419 = _tmem_load_3[23];
                float _max_52 = max_noftz(x0_00_412, neg_cl);
                float _min_104 = fminf(_max_52, cl);
                float x0c_00_420 = _min_104;
                float _max_53 = max_noftz(x0_01_413, neg_cl);
                float _min_105 = fminf(_max_53, cl);
                float x0c_01_421 = _min_105;
                float _max_54 = max_noftz(x0_10_416, neg_cl);
                float _min_106 = fminf(_max_54, cl);
                float x0c_10_422 = _min_106;
                float _max_55 = max_noftz(x0_11_417, neg_cl);
                float _min_107 = fminf(_max_55, cl);
                float x0c_11_423 = _min_107;
                float _fma_52 = __fmaf_rn(x0c_00_420, sg, 0.0f);
                float lin_00_424 = _fma_52;
                float _fma_53 = __fmaf_rn(x0c_01_421, sg, 0.0f);
                float lin_01_425 = _fma_53;
                float _fma_54 = __fmaf_rn(x0c_10_422, sg, 0.0f);
                float lin_10_426 = _fma_54;
                float _fma_55 = __fmaf_rn(x0c_11_423, sg, 0.0f);
                float lin_11_427 = _fma_55;
                float _exp2_52 = approx_exp2(-(x1_00_414 * fused));
                float _rcp_52 = approx_rcp(1.0f + _exp2_52);
                float sig_00_428 = _rcp_52;
                float _exp2_53 = approx_exp2(-(x1_01_415 * fused));
                float _rcp_53 = approx_rcp(1.0f + _exp2_53);
                float sig_01_429 = _rcp_53;
                float _exp2_54 = approx_exp2(-(x1_10_418 * fused));
                float _rcp_54 = approx_rcp(1.0f + _exp2_54);
                float sig_10_430 = _rcp_54;
                float _exp2_55 = approx_exp2(-(x1_11_419 * fused));
                float _rcp_55 = approx_rcp(1.0f + _exp2_55);
                float sig_11_431 = _rcp_55;
                float act_00_432 = x1_00_414 * sig_00_428;
                float act_01_433 = x1_01_415 * sig_01_429;
                float act_10_434 = x1_10_418 * sig_10_430;
                float act_11_435 = x1_11_419 * sig_11_431;
                float _min_108 = fminf(act_00_432, cl);
                act_00_432 = _min_108;
                float _min_109 = fminf(act_01_433, cl);
                act_01_433 = _min_109;
                float _min_110 = fminf(act_10_434, cl);
                act_10_434 = _min_110;
                float _min_111 = fminf(act_11_435, cl);
                act_11_435 = _min_111;
                float y_00_436 = lin_00_424 * act_00_432;
                float y_10_437 = lin_10_426 * act_10_434;
                float y_01_438 = lin_01_425 * act_01_433;
                float y_11_439 = lin_11_427 * act_11_435;
                quad[0] = y_00_436 * sc;
                quad[1] = y_10_437 * sc;
                quad[2] = y_01_438 * sc;
                quad[3] = y_11_439 * sc;
                uint32_t _fp8_13[1];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(quad[0]), "f"(quad[1]),
                                           "f"(quad[2]), "f"(quad[3]));
                    _fp8_13[0] = _packed;
                }
                int off0_440 = local_token0_410 * 64 + base_row_239;
                int off1_441 = local_token1_411 * 64 + base_row_239;
                int swz0_442 = off0_440 ^ (off0_440 >> 7 & 3) << 4;
                int swz1_443 = off1_441 ^ (off1_441 >> 7 & 3) << 4;
                epi_staging[swz0_442 >> 1] = _fp8_13[0] & 65535;
                epi_staging[swz1_443 >> 1] = _fp8_13[0] >> 16;
                int local_token0_444 = lane_1 % 4 * 2 + 48;
                int local_token1_445 = local_token0_444 + 1;
                float x0_00_446 = _tmem_load_2[24];
                float x0_01_447 = _tmem_load_2[25];
                float x1_00_448 = _tmem_load_2[26];
                float x1_01_449 = _tmem_load_2[27];
                float x0_10_450 = _tmem_load_3[24];
                float x0_11_451 = _tmem_load_3[25];
                float x1_10_452 = _tmem_load_3[26];
                float x1_11_453 = _tmem_load_3[27];
                float _max_56 = max_noftz(x0_00_446, neg_cl);
                float _min_112 = fminf(_max_56, cl);
                float x0c_00_454 = _min_112;
                float _max_57 = max_noftz(x0_01_447, neg_cl);
                float _min_113 = fminf(_max_57, cl);
                float x0c_01_455 = _min_113;
                float _max_58 = max_noftz(x0_10_450, neg_cl);
                float _min_114 = fminf(_max_58, cl);
                float x0c_10_456 = _min_114;
                float _max_59 = max_noftz(x0_11_451, neg_cl);
                float _min_115 = fminf(_max_59, cl);
                float x0c_11_457 = _min_115;
                float _fma_56 = __fmaf_rn(x0c_00_454, sg, 0.0f);
                float lin_00_458 = _fma_56;
                float _fma_57 = __fmaf_rn(x0c_01_455, sg, 0.0f);
                float lin_01_459 = _fma_57;
                float _fma_58 = __fmaf_rn(x0c_10_456, sg, 0.0f);
                float lin_10_460 = _fma_58;
                float _fma_59 = __fmaf_rn(x0c_11_457, sg, 0.0f);
                float lin_11_461 = _fma_59;
                float _exp2_56 = approx_exp2(-(x1_00_448 * fused));
                float _rcp_56 = approx_rcp(1.0f + _exp2_56);
                float sig_00_462 = _rcp_56;
                float _exp2_57 = approx_exp2(-(x1_01_449 * fused));
                float _rcp_57 = approx_rcp(1.0f + _exp2_57);
                float sig_01_463 = _rcp_57;
                float _exp2_58 = approx_exp2(-(x1_10_452 * fused));
                float _rcp_58 = approx_rcp(1.0f + _exp2_58);
                float sig_10_464 = _rcp_58;
                float _exp2_59 = approx_exp2(-(x1_11_453 * fused));
                float _rcp_59 = approx_rcp(1.0f + _exp2_59);
                float sig_11_465 = _rcp_59;
                float act_00_466 = x1_00_448 * sig_00_462;
                float act_01_467 = x1_01_449 * sig_01_463;
                float act_10_468 = x1_10_452 * sig_10_464;
                float act_11_469 = x1_11_453 * sig_11_465;
                float _min_116 = fminf(act_00_466, cl);
                act_00_466 = _min_116;
                float _min_117 = fminf(act_01_467, cl);
                act_01_467 = _min_117;
                float _min_118 = fminf(act_10_468, cl);
                act_10_468 = _min_118;
                float _min_119 = fminf(act_11_469, cl);
                act_11_469 = _min_119;
                float y_00_470 = lin_00_458 * act_00_466;
                float y_10_471 = lin_10_460 * act_10_468;
                float y_01_472 = lin_01_459 * act_01_467;
                float y_11_473 = lin_11_461 * act_11_469;
                quad[0] = y_00_470 * sc;
                quad[1] = y_10_471 * sc;
                quad[2] = y_01_472 * sc;
                quad[3] = y_11_473 * sc;
                uint32_t _fp8_14[1];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(quad[0]), "f"(quad[1]),
                                           "f"(quad[2]), "f"(quad[3]));
                    _fp8_14[0] = _packed;
                }
                int off0_474 = local_token0_444 * 64 + base_row_239;
                int off1_475 = local_token1_445 * 64 + base_row_239;
                int swz0_476 = off0_474 ^ (off0_474 >> 7 & 3) << 4;
                int swz1_477 = off1_475 ^ (off1_475 >> 7 & 3) << 4;
                epi_staging[swz0_476 >> 1] = _fp8_14[0] & 65535;
                epi_staging[swz1_477 >> 1] = _fp8_14[0] >> 16;
                int local_token0_478 = lane_1 % 4 * 2 + 56;
                int local_token1_479 = local_token0_478 + 1;
                float x0_00_480 = _tmem_load_2[28];
                float x0_01_481 = _tmem_load_2[29];
                float x1_00_482 = _tmem_load_2[30];
                float x1_01_483 = _tmem_load_2[31];
                float x0_10_484 = _tmem_load_3[28];
                float x0_11_485 = _tmem_load_3[29];
                float x1_10_486 = _tmem_load_3[30];
                float x1_11_487 = _tmem_load_3[31];
                float _max_60 = max_noftz(x0_00_480, neg_cl);
                float _min_120 = fminf(_max_60, cl);
                float x0c_00_488 = _min_120;
                float _max_61 = max_noftz(x0_01_481, neg_cl);
                float _min_121 = fminf(_max_61, cl);
                float x0c_01_489 = _min_121;
                float _max_62 = max_noftz(x0_10_484, neg_cl);
                float _min_122 = fminf(_max_62, cl);
                float x0c_10_490 = _min_122;
                float _max_63 = max_noftz(x0_11_485, neg_cl);
                float _min_123 = fminf(_max_63, cl);
                float x0c_11_491 = _min_123;
                float _fma_60 = __fmaf_rn(x0c_00_488, sg, 0.0f);
                float lin_00_492 = _fma_60;
                float _fma_61 = __fmaf_rn(x0c_01_489, sg, 0.0f);
                float lin_01_493 = _fma_61;
                float _fma_62 = __fmaf_rn(x0c_10_490, sg, 0.0f);
                float lin_10_494 = _fma_62;
                float _fma_63 = __fmaf_rn(x0c_11_491, sg, 0.0f);
                float lin_11_495 = _fma_63;
                float _exp2_60 = approx_exp2(-(x1_00_482 * fused));
                float _rcp_60 = approx_rcp(1.0f + _exp2_60);
                float sig_00_496 = _rcp_60;
                float _exp2_61 = approx_exp2(-(x1_01_483 * fused));
                float _rcp_61 = approx_rcp(1.0f + _exp2_61);
                float sig_01_497 = _rcp_61;
                float _exp2_62 = approx_exp2(-(x1_10_486 * fused));
                float _rcp_62 = approx_rcp(1.0f + _exp2_62);
                float sig_10_498 = _rcp_62;
                float _exp2_63 = approx_exp2(-(x1_11_487 * fused));
                float _rcp_63 = approx_rcp(1.0f + _exp2_63);
                float sig_11_499 = _rcp_63;
                float act_00_500 = x1_00_482 * sig_00_496;
                float act_01_501 = x1_01_483 * sig_01_497;
                float act_10_502 = x1_10_486 * sig_10_498;
                float act_11_503 = x1_11_487 * sig_11_499;
                float _min_124 = fminf(act_00_500, cl);
                act_00_500 = _min_124;
                float _min_125 = fminf(act_01_501, cl);
                act_01_501 = _min_125;
                float _min_126 = fminf(act_10_502, cl);
                act_10_502 = _min_126;
                float _min_127 = fminf(act_11_503, cl);
                act_11_503 = _min_127;
                float y_00_504 = lin_00_492 * act_00_500;
                float y_10_505 = lin_10_494 * act_10_502;
                float y_01_506 = lin_01_493 * act_01_501;
                float y_11_507 = lin_11_495 * act_11_503;
                quad[0] = y_00_504 * sc;
                quad[1] = y_10_505 * sc;
                quad[2] = y_01_506 * sc;
                quad[3] = y_11_507 * sc;
                uint32_t _fp8_15[1];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(quad[0]), "f"(quad[1]),
                                           "f"(quad[2]), "f"(quad[3]));
                    _fp8_15[0] = _packed;
                }
                int off0_508 = local_token0_478 * 64 + base_row_239;
                int off1_509 = local_token1_479 * 64 + base_row_239;
                int swz0_510 = off0_508 ^ (off0_508 >> 7 & 3) << 4;
                int swz1_511 = off1_509 ^ (off1_509 >> 7 & 3) << 4;
                epi_staging[swz0_510 >> 1] = _fp8_15[0] & 65535;
                epi_staging[swz1_511 >> 1] = _fp8_15[0] >> 16;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                if (warp == 0) {
                    if (elect_sync()) {
                        int padding_rows_1 = (128 - valid_rows % 128) % 128;
                        tma_store_4d((&C), m_tile * 64, padding_rows_1 + 64, 1073741824, n_tile * 128 - (unsigned int)padding_rows_1 + 1073741824, epi_staging_addr);
                    }
                }
                asm volatile("cp.async.bulk.commit_group;");
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((mma_free_addr + (acc_stage) * 8) & 0xFEFFFFFF) : "memory");
                acc_stage += 1;
                if (acc_stage == 2) { acc_stage = 0; _phase_mma_full ^= 1; }
                mbarrier_wait(work_full_addr + (work_stage) * 8, _phase_work_full);
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
                    : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_6 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(_clc_ctaid_6)
                    : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_7 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(_clc_ctaid_7)
                    : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + work_stage * 8), "r"(0) : "memory");
                work_stage += 1;
                if (work_stage == 3) { work_stage = 0; _phase_work_full ^= 1; }
                unsigned int ok = _clc_valid_3;
                if (_clc_ctaid_7 >= (unsigned int)bound) {
                    ok = 0;
                }
                if (ok == 0) {
                    break;
                }
                m_tile = _clc_ctaid_6 + (unsigned int)cta_rank;
                n_tile = _clc_ctaid_7;
            }
        }
    }
    // ---- Role: load_b ----
    if (warp >= 4 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
        { // load_b_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int stage = 0;
            unsigned int work_stage_1 = 0;
            unsigned int m_tile_1 = blockIdx.x;
            unsigned int n_tile_1 = blockIdx.y;
            int bound_1 = num_non_exiting_ctas[0];
            int warp_local = warp - 4;
            int route_base = 0;
            int routed[8];
            unsigned int cta_mask = 1 << cta_rank;
            unsigned int _phase_k_done = 1;
            unsigned int _phase_work_full_1 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_1 = 0; _tile_iter_1 < grid_m / 2 * grid_n; _tile_iter_1++) {
                if (m_tile_1 >= (unsigned int)grid_m || n_tile_1 >= (unsigned int)bound_1) {
                    break;
                }
                route_base = n_tile_1 * 128 + (unsigned int)(cta_rank * 64) + (unsigned int)(warp_local * 4);
                for (int row = 0; row < 4; row++) {
                    routed[row] = route_map[route_base + row];
                }
                route_base = n_tile_1 * 128 + (unsigned int)(cta_rank * 64) + (unsigned int)((8 + warp_local) * 4);
                for (int row_1 = 0; row_1 < 4; row_1++) {
                    routed[4 + row_1] = route_map[route_base + row_1];
                }
                #pragma unroll 1
                for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                    mbarrier_wait(k_done_addr + (stage) * 8, _phase_k_done);
                    if (elect_sync()) {
                        tma_gather4_gmem2smem_mc_cta2(smem_b_addr + stage * 8192 + (unsigned int)(warp_local * 512), (&B), iter_k * 128, routed[0], routed[1], routed[2], routed[3], ((b_full_addr + (stage) * 8) & 0xFEFFFFFF), cta_mask);
                        tma_gather4_gmem2smem_mc_cta2(smem_b_addr + stage * 8192 + (unsigned int)((8 + warp_local) * 512), (&B), iter_k * 128, routed[4], routed[5], routed[6], routed[7], ((b_full_addr + (stage) * 8) & 0xFEFFFFFF), cta_mask);
                    }
                    if (warp == 4) {
                        if (elect_sync()) {
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((b_full_addr + (stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(8192)) : "memory");
                        }
                    }
                    stage += 1;
                    if (stage == 8) { stage = 0; _phase_k_done ^= 1; }
                }
                mbarrier_wait(work_full_addr + (work_stage_1) * 8, _phase_work_full_1);
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
                    : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
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
                    : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_3 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(_clc_ctaid_3)
                    : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + work_stage_1 * 8), "r"(0) : "memory");
                work_stage_1 += 1;
                if (work_stage_1 == 3) { work_stage_1 = 0; _phase_work_full_1 ^= 1; }
                unsigned int ok_1 = _clc_valid_1;
                if (_clc_ctaid_3 >= (unsigned int)bound_1) {
                    ok_1 = 0;
                }
                if (ok_1 == 0) {
                    break;
                }
                m_tile_1 = _clc_ctaid_2 + (unsigned int)cta_rank;
                n_tile_1 = _clc_ctaid_3;
            }
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
    }
    // ---- Role: load_a ----
    if (warp == 12) {
        { // load_a_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int stage_1 = 0;
            unsigned int work_stage_2 = 0;
            unsigned int throttle_stage = 0;
            unsigned int m_tile_2 = blockIdx.x;
            unsigned int n_tile_2 = blockIdx.y;
            int bound_2 = num_non_exiting_ctas[0];
            unsigned int cta_mask_1 = 1 << cta_rank;
            unsigned int _phase_throttle_empty = 1;
            unsigned int _phase_k_done_1 = 1;
            unsigned int _phase_work_full_2 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_2 = 0; _tile_iter_2 < grid_m / 2 * grid_n; _tile_iter_2++) {
                if (m_tile_2 >= (unsigned int)grid_m || n_tile_2 >= (unsigned int)bound_2) {
                    break;
                }
                int expert_1 = tile_expert[n_tile_2];
                int sole_reader = 1;
                if ((unsigned int)bound_2 > n_tile_2 + 1) {
                    if (tile_expert[n_tile_2 + 1] == expert_1) {
                        sole_reader = 0;
                    }
                }
                if (n_tile_2 > 0) {
                    if (tile_expert[n_tile_2 - 1] == expert_1) {
                        sole_reader = 0;
                    }
                }
                if (cta_rank == 0) {
                    mbarrier_wait(throttle_empty_addr + (throttle_stage) * 8, _phase_throttle_empty);
                    mbarrier_arrive(throttle_full_addr + (throttle_stage) * 8);
                    throttle_stage += 1;
                    if (throttle_stage == 3) { throttle_stage = 0; _phase_throttle_empty ^= 1; }
                }
                #pragma unroll 1
                for (int iter_k_1 = 0; iter_k_1 < K_tiles; iter_k_1++) {
                    mbarrier_wait(k_done_addr + (stage_1) * 8, _phase_k_done_1);
                    if (elect_sync()) {
                        if (sole_reader == 1) {
                            asm volatile(
                                "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4, %5}], [%6], %7, %8;"
                                :: "r"(smem_a_addr + stage_1 * 16384), "l"((&A)), "r"(0), "r"(m_tile_2 * 128), "r"(iter_k_1), "r"(expert_1),
                                   "r"(((a_full_addr + (stage_1) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(cta_mask_1)), "l"(0x12F0000000000000ULL) : "memory");
                        } else {
                            asm volatile(
                                "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                                :: "r"(smem_a_addr + stage_1 * 16384), "l"((&A)), "r"(0), "r"(m_tile_2 * 128), "r"(iter_k_1), "r"(expert_1),
                                   "r"(((a_full_addr + (stage_1) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(cta_mask_1)) : "memory");
                        }
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((a_full_addr + (stage_1) * 8) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                    }
                    stage_1 += 1;
                    if (stage_1 == 8) { stage_1 = 0; _phase_k_done_1 ^= 1; }
                }
                mbarrier_wait(work_full_addr + (work_stage_2) * 8, _phase_work_full_2);
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
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
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
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_1 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(_clc_ctaid_1)
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + work_stage_2 * 8), "r"(0) : "memory");
                work_stage_2 += 1;
                if (work_stage_2 == 3) { work_stage_2 = 0; _phase_work_full_2 ^= 1; }
                unsigned int ok_2 = _clc_valid_0;
                if (_clc_ctaid_1 >= (unsigned int)bound_2) {
                    ok_2 = 0;
                }
                if (ok_2 == 0) {
                    break;
                }
                m_tile_2 = _clc_ctaid_0 + (unsigned int)cta_rank;
                n_tile_2 = _clc_ctaid_1;
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 13) {
        { // mma_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int _phase_mma_free = 1;
            unsigned int _phase_a_full = 0;
            unsigned int _phase_b_full = 0;
            unsigned int _phase_work_full_3 = 0;
            if (cta_rank == 0) {
                unsigned int stage_2 = 0;
                unsigned int acc_stage_1 = 0;
                unsigned int work_stage_3 = 0;
                unsigned int m_tile_3 = blockIdx.x;
                unsigned int n_tile_3 = blockIdx.y;
                int bound_3 = num_non_exiting_ctas[0];
                #pragma unroll 1
                for (unsigned int _tile_iter_3 = 0; _tile_iter_3 < grid_m / 2 * grid_n; _tile_iter_3++) {
                    if (m_tile_3 >= (unsigned int)grid_m || n_tile_3 >= (unsigned int)bound_3) {
                        break;
                    }
                    mbarrier_wait(mma_free_addr + (acc_stage_1) * 8, _phase_mma_free);
                    #pragma unroll 1
                    for (int iter_k_2 = 0; iter_k_2 < K_tiles; iter_k_2++) {
                        mbarrier_wait(a_full_addr + (stage_2) * 8, _phase_a_full);
                        mbarrier_wait(b_full_addr + (stage_2) * 8, _phase_b_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_0 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_2) * 1024;
                        int _mma_b_lo_0 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_2) * 512;
                        mma_ss_step_cg2(_mma_a_lo_0, _mma_b_lo_0, (tmem_accum + (acc_stage_1 * 128)), 270532624, ((((1) ? ((iter_k_2 == 0) ? 1 : 0) : 0)) ? 0 : 1), 0x40004040U, 0x40004040U);
                        int _mma_a_lo_1 = (((smem_a_addr + 32) >> 4) & 0x3FFF) + (stage_2) * 1024;
                        int _mma_b_lo_1 = (((smem_b_addr + 32) >> 4) & 0x3FFF) + (stage_2) * 512;
                        mma_ss_step_cg2(_mma_a_lo_1, _mma_b_lo_1, (tmem_accum + (acc_stage_1 * 128)), 270532624, ((((0) ? ((iter_k_2 == 0) ? 1 : 0) : 0)) ? 0 : 1), 0x40004040U, 0x40004040U);
                        int _mma_a_lo_2 = (((smem_a_addr + 64) >> 4) & 0x3FFF) + (stage_2) * 1024;
                        int _mma_b_lo_2 = (((smem_b_addr + 64) >> 4) & 0x3FFF) + (stage_2) * 512;
                        mma_ss_step_cg2(_mma_a_lo_2, _mma_b_lo_2, (tmem_accum + (acc_stage_1 * 128)), 270532624, ((((0) ? ((iter_k_2 == 0) ? 1 : 0) : 0)) ? 0 : 1), 0x40004040U, 0x40004040U);
                        int _mma_a_lo_3 = (((smem_a_addr + 96) >> 4) & 0x3FFF) + (stage_2) * 1024;
                        int _mma_b_lo_3 = (((smem_b_addr + 96) >> 4) & 0x3FFF) + (stage_2) * 512;
                        mma_ss_step_cg2(_mma_a_lo_3, _mma_b_lo_3, (tmem_accum + (acc_stage_1 * 128)), 270532624, ((((0) ? ((iter_k_2 == 0) ? 1 : 0) : 0)) ? 0 : 1), 0x40004040U, 0x40004040U);
                        elect_commit_cg2_multicast(k_done_addr + (stage_2) * 8, (uint16_t)(3));
                        if (iter_k_2 + 1 == K_tiles) {
                            elect_commit_cg2_multicast(mma_full_addr + (acc_stage_1) * 8, (uint16_t)(3));
                        }
                        stage_2 += 1;
                        if (stage_2 == 8) { stage_2 = 0; _phase_a_full ^= 1; _phase_b_full ^= 1; }
                    }
                    acc_stage_1 += 1;
                    if (acc_stage_1 == 2) { acc_stage_1 = 0; _phase_mma_free ^= 1; }
                    mbarrier_wait(work_full_addr + (work_stage_3) * 8, _phase_work_full_3);
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
                        : "r"(work_response_addr + work_stage_3 * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_4 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "+r"(_clc_ctaid_4)
                        : "r"(work_response_addr + work_stage_3 * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_5 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "+r"(_clc_ctaid_5)
                        : "r"(work_response_addr + work_stage_3 * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(work_empty_addr + work_stage_3 * 8), "r"(0) : "memory");
                    work_stage_3 += 1;
                    if (work_stage_3 == 3) { work_stage_3 = 0; _phase_work_full_3 ^= 1; }
                    unsigned int ok_3 = _clc_valid_2;
                    if (_clc_ctaid_5 >= (unsigned int)bound_3) {
                        ok_3 = 0;
                    }
                    if (ok_3 == 0) {
                        break;
                    }
                    m_tile_3 = _clc_ctaid_4 + (unsigned int)cta_rank;
                    n_tile_3 = _clc_ctaid_5;
                }
            }
        }
    }
    // ---- Role: work_id ----
    if (warp == 14) {
        { // work_id_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int work_stage_4 = 0;
            unsigned int throttle_stage_1 = 0;
            int bound_4 = num_non_exiting_ctas[0];
            unsigned int drain_stage = 0;
            unsigned int _phase_throttle_full = 0;
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_work_full_4 = 0;
            unsigned int _phase_drain_full = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (unsigned int _tile_iter_4 = 0; _tile_iter_4 < grid_m / 2 * grid_n; _tile_iter_4++) {
                    mbarrier_wait(throttle_full_addr + (throttle_stage_1) * 8, _phase_throttle_full);
                    mbarrier_arrive(throttle_empty_addr + (throttle_stage_1) * 8);
                    throttle_stage_1 += 1;
                    if (throttle_stage_1 == 3) { throttle_stage_1 = 0; _phase_throttle_full ^= 1; }
                    mbarrier_wait_cluster_hint(work_empty_addr + (work_stage_4) * 8, _phase_work_empty, 10000000);
                    if (lane < 2) {
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                            "}"
                            :: "r"(work_full_addr + work_stage_4 * 8), "r"(lane), "r"((uint32_t)(16)) : "memory");
                    }
                    if (elect_sync()) {
                        asm volatile(
                            "fence.proxy.async.shared::cta;\n\t"
                            "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                                " [%0], [%1];"
                            :: "r"(work_response_addr + work_stage_4 * 16 + 0 * 16), "r"(work_full_addr + work_stage_4 * 8)
                            : "memory");
                    }
                    mbarrier_wait(work_full_addr + (work_stage_4) * 8, _phase_work_full_4);
                    uint32_t _clc_valid_4 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %0, 1, 0, p1;\n\t"
                        "}\n"
                        : "=r"(_clc_valid_4)
                        : "r"(work_response_addr + work_stage_4 * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_8 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "+r"(_clc_ctaid_8)
                        : "r"(work_response_addr + work_stage_4 * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_9 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "+r"(_clc_ctaid_9)
                        : "r"(work_response_addr + work_stage_4 * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(work_empty_addr + work_stage_4 * 8), "r"(0) : "memory");
                    work_stage_4 += 1;
                    if (work_stage_4 == 3) { work_stage_4 = 0; _phase_work_empty ^= 1; _phase_work_full_4 ^= 1; }
                    unsigned int ok_4 = _clc_valid_4;
                    if (_clc_ctaid_9 >= (unsigned int)bound_4) {
                        ok_4 = 0;
                    }
                    if (ok_4 == 0) {
                        if (_clc_valid_4 != 0) {
                            #pragma unroll 1
                            for (unsigned int _drain_iter = 0; _drain_iter < (grid_m / 2 * grid_n + 4 - 1) / 4 + 1; _drain_iter++) {
                                if (elect_sync()) {
                                    mbarrier_arrive_expect_tx(drain_full_addr + (drain_stage) * 8, 64);
                                    asm volatile(
                                        "fence.proxy.async.shared::cta;\n\t"
                                        "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                            ".mbarrier::complete_tx::bytes.b128"
                                            " [%0], [%1];"
                                        :: "r"(fast_drain_response_addr + drain_stage * 64 + 0 * 16), "r"(drain_full_addr + drain_stage * 8)
                                        : "memory");
                                    asm volatile(
                                        "fence.proxy.async.shared::cta;\n\t"
                                        "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                            ".mbarrier::complete_tx::bytes.b128"
                                            " [%0], [%1];"
                                        :: "r"(fast_drain_response_addr + drain_stage * 64 + 1 * 16), "r"(drain_full_addr + drain_stage * 8)
                                        : "memory");
                                    asm volatile(
                                        "fence.proxy.async.shared::cta;\n\t"
                                        "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                            ".mbarrier::complete_tx::bytes.b128"
                                            " [%0], [%1];"
                                        :: "r"(fast_drain_response_addr + drain_stage * 64 + 2 * 16), "r"(drain_full_addr + drain_stage * 8)
                                        : "memory");
                                    asm volatile(
                                        "fence.proxy.async.shared::cta;\n\t"
                                        "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                            ".mbarrier::complete_tx::bytes.b128"
                                            " [%0], [%1];"
                                        :: "r"(fast_drain_response_addr + drain_stage * 64 + 3 * 16), "r"(drain_full_addr + drain_stage * 8)
                                        : "memory");
                                }
                                mbarrier_wait(drain_full_addr + (drain_stage) * 8, _phase_drain_full);
                                unsigned int canceled = 0;
                                uint32_t _clc_valid_5 = 0;
                                asm volatile(
                                    "{\n\t"
                                    ".reg .pred p1;\n\t"
                                    ".reg .b128 clc_r;\n\t"
                                    "ld.shared.b128 clc_r, [%1];\n\t"
                                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                                    "selp.u32 %0, 1, 0, p1;\n\t"
                                    "}\n"
                                    : "=r"(_clc_valid_5)
                                    : "r"(fast_drain_response_addr + drain_stage * 64 + 0 * 16)
                                    : "memory");
                                canceled += _clc_valid_5;
                                uint32_t _clc_valid_6 = 0;
                                asm volatile(
                                    "{\n\t"
                                    ".reg .pred p1;\n\t"
                                    ".reg .b128 clc_r;\n\t"
                                    "ld.shared.b128 clc_r, [%1];\n\t"
                                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                                    "selp.u32 %0, 1, 0, p1;\n\t"
                                    "}\n"
                                    : "=r"(_clc_valid_6)
                                    : "r"(fast_drain_response_addr + drain_stage * 64 + 1 * 16)
                                    : "memory");
                                canceled += _clc_valid_6;
                                uint32_t _clc_valid_7 = 0;
                                asm volatile(
                                    "{\n\t"
                                    ".reg .pred p1;\n\t"
                                    ".reg .b128 clc_r;\n\t"
                                    "ld.shared.b128 clc_r, [%1];\n\t"
                                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                                    "selp.u32 %0, 1, 0, p1;\n\t"
                                    "}\n"
                                    : "=r"(_clc_valid_7)
                                    : "r"(fast_drain_response_addr + drain_stage * 64 + 2 * 16)
                                    : "memory");
                                canceled += _clc_valid_7;
                                uint32_t _clc_valid_8 = 0;
                                asm volatile(
                                    "{\n\t"
                                    ".reg .pred p1;\n\t"
                                    ".reg .b128 clc_r;\n\t"
                                    "ld.shared.b128 clc_r, [%1];\n\t"
                                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                                    "selp.u32 %0, 1, 0, p1;\n\t"
                                    "}\n"
                                    : "=r"(_clc_valid_8)
                                    : "r"(fast_drain_response_addr + drain_stage * 64 + 3 * 16)
                                    : "memory");
                                canceled += _clc_valid_8;
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                _phase_drain_full ^= 1;
                                if (canceled == 0) {
                                    break;
                                }
                            }
                        }
                        break;
                    }
                }
            }
        }
    }
    // ---- Role: padding ----
    if (warp == 15) {
        { // padding_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int work_stage_5 = 0;
            unsigned int m_tile_4 = blockIdx.x;
            unsigned int n_tile_4 = blockIdx.y;
            int bound_5 = num_non_exiting_ctas[0];
            unsigned int _phase_work_full_5 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_5 = 0; _tile_iter_5 < grid_m / 2 * grid_n; _tile_iter_5++) {
                if (m_tile_4 >= (unsigned int)grid_m || n_tile_4 >= (unsigned int)bound_5) {
                    break;
                }
                mbarrier_wait(work_full_addr + (work_stage_5) * 8, _phase_work_full_5);
                uint32_t _clc_valid_9 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_9)
                    : "r"(work_response_addr + work_stage_5 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_10 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(_clc_ctaid_10)
                    : "r"(work_response_addr + work_stage_5 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_11 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(_clc_ctaid_11)
                    : "r"(work_response_addr + work_stage_5 * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + work_stage_5 * 8), "r"(0) : "memory");
                work_stage_5 += 1;
                if (work_stage_5 == 3) { work_stage_5 = 0; _phase_work_full_5 ^= 1; }
                unsigned int ok_5 = _clc_valid_9;
                if (_clc_ctaid_11 >= (unsigned int)bound_5) {
                    ok_5 = 0;
                }
                if (ok_5 == 0) {
                    break;
                }
                m_tile_4 = _clc_ctaid_10 + (unsigned int)cta_rank;
                n_tile_4 = _clc_ctaid_11;
            }
        }
    }

    // Cleanup
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
    }
}

} // extern "C"
