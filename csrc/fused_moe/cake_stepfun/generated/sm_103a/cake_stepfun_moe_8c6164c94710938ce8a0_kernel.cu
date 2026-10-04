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
#define TMEM_NCOLS 128
#define TMEM_ACCUM_OFFSET 0
#define NUM_K_PIPE_STAGES 5
#define NUM_MMA_PIPE_STAGES 2
#define NUM_WORK_PIPE_STAGES 3
#define NUM_THROTTLE_PIPE_STAGES 3
#define NUM_DRAIN_PIPE_STAGES 1
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 32768
#define SMEM_SMEM_A_STRIDE 32768
#define SMEM_SMEM_B_OFF 164864
#define SMEM_SMEM_B_STAGE_BYTES 8192
#define SMEM_SMEM_B_STRIDE 8192
#define SMEM_EPI_STAGING_OFF 205824
#define SMEM_EPI_STAGING_STAGE_BYTES 4096
#define SMEM_EPI_STAGING_STRIDE 4096
#define SMEM_WORK_RESPONSE_OFF 209920
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_FAST_DRAIN_RESPONSE_OFF 209968
#define SMEM_FAST_DRAIN_RESPONSE_STAGE_BYTES 64
#define SMEM_FAST_DRAIN_RESPONSE_STRIDE 64
#define SMEM_TOTAL 210048
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
kernel_cake_stepfun_moe_8c6164c94710938ce8a0(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap C, int* __restrict__ route_map, int* __restrict__ tile_expert, int* __restrict__ tile_mn_limit, int* __restrict__ num_non_exiting_ctas, float* __restrict__ scale_c, float* __restrict__ scale_gate, float* __restrict__ clamp_limit, float* __restrict__ act_alpha, float* __restrict__ act_beta, int M_out, int K, int grid_m, int grid_n, int K_tiles)
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
    #define b_full_addr (mbar_base + 40)
    #define k_done_addr (mbar_base + 80)
    #define mma_full_addr (mbar_base + 120)
    #define mma_free_addr (mbar_base + 136)
    #define work_full_addr (mbar_base + 152)
    #define work_empty_addr (mbar_base + 176)
    #define throttle_full_addr (mbar_base + 200)
    #define throttle_empty_addr (mbar_base + 224)
    #define drain_full_addr (mbar_base + 248)

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
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + 164864);
    const int smem_b_addr = smem + 164864;
    uint16_t* epi_staging = reinterpret_cast<uint16_t*>(smem_raw + 205824);
    const int epi_staging_addr = smem + 205824;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 209920);
    const int work_response_addr = smem + 209920;
    unsigned int* fast_drain_response = reinterpret_cast<unsigned int*>(smem_raw + 209968);
    const int fast_drain_response_addr = smem + 209968;
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if ((int)blockIdx.y >= num_non_exiting_ctas[0]) return;

    // Mbarrier init (10 pipeline groups, 0 ordered-sequence groups, 32 barriers)
    // Mbarriers at smem_raw[0..256)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'k_pipe' ---
            // a_full: 5 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            mbarrier_init(smem + 8, 2);
            mbarrier_init(smem + 16, 2);
            mbarrier_init(smem + 24, 2);
            mbarrier_init(smem + 32, 2);
            // b_full: 5 barriers, init_count=2
            mbarrier_init(smem + 40, 2);
            mbarrier_init(smem + 48, 2);
            mbarrier_init(smem + 56, 2);
            mbarrier_init(smem + 64, 2);
            mbarrier_init(smem + 72, 2);
            // k_done: 5 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            // --- pipeline 'mma_pipe' ---
            // mma_full: 2 barriers, init_count=1
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            // mma_free: 2 barriers, init_count=256
            mbarrier_init(smem + 136, 256);
            mbarrier_init(smem + 144, 256);
            // --- pipeline 'work_pipe' ---
            // work_full: 3 barriers, init_count=1
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            // work_empty: 3 barriers, init_count=960
            mbarrier_init(smem + 176, 960);
            mbarrier_init(smem + 184, 960);
            mbarrier_init(smem + 192, 960);
            // --- pipeline 'throttle_pipe' ---
            // throttle_full: 3 barriers, init_count=32
            mbarrier_init(smem + 200, 32);
            mbarrier_init(smem + 208, 32);
            mbarrier_init(smem + 216, 32);
            // throttle_empty: 3 barriers, init_count=32
            mbarrier_init(smem + 224, 32);
            mbarrier_init(smem + 232, 32);
            mbarrier_init(smem + 240, 32);
            // --- pipeline 'drain_pipe' ---
            // drain_full: 1 barriers, init_count=1
            mbarrier_init(smem + 248, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (128 columns, 128 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 256);
    if (warp == 0) {
        int _tmem_hold = smem + 256;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
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
                int valid_rows = (unsigned int)tile_mn_limit[n_tile] - n_tile * 64;
                float sc = scale_c[expert];
                float sg = scale_gate[expert];
                float cl = clamp_limit[expert];
                float neg_cl = -cl;
                float fused = 1.4426950216293335f * sg;
                mbarrier_wait(mma_full_addr + (acc_stage) * 8, _phase_mma_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int acc_offset = acc_stage * 64;
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
                float x0s_00 = x0c_00 * sc;
                float x0s_01 = x0c_01 * sc;
                float x0s_10 = x0c_10 * sc;
                float x0s_11 = x0c_11 * sc;
                float lin_00 = x0s_00 * sg;
                float lin_01 = x0s_01 * sg;
                float lin_10 = x0s_10 * sg;
                float lin_11 = x0s_11 * sg;
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
                quad[0] = lin_00 * act_00;
                quad[1] = lin_10 * act_10;
                quad[2] = lin_01 * act_01;
                quad[3] = lin_11 * act_11;
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
                float x0s_00_14 = x0c_00_10 * sc;
                float x0s_01_15 = x0c_01_11 * sc;
                float x0s_10_16 = x0c_10_12 * sc;
                float x0s_11_17 = x0c_11_13 * sc;
                float lin_00_18 = x0s_00_14 * sg;
                float lin_01_19 = x0s_01_15 * sg;
                float lin_10_20 = x0s_10_16 * sg;
                float lin_11_21 = x0s_11_17 * sg;
                float _exp2_4 = approx_exp2(-(x1_00_4 * fused));
                float _rcp_4 = approx_rcp(1.0f + _exp2_4);
                float sig_00_22 = _rcp_4;
                float _exp2_5 = approx_exp2(-(x1_01_5 * fused));
                float _rcp_5 = approx_rcp(1.0f + _exp2_5);
                float sig_01_23 = _rcp_5;
                float _exp2_6 = approx_exp2(-(x1_10_8 * fused));
                float _rcp_6 = approx_rcp(1.0f + _exp2_6);
                float sig_10_24 = _rcp_6;
                float _exp2_7 = approx_exp2(-(x1_11_9 * fused));
                float _rcp_7 = approx_rcp(1.0f + _exp2_7);
                float sig_11_25 = _rcp_7;
                float act_00_26 = x1_00_4 * sig_00_22;
                float act_01_27 = x1_01_5 * sig_01_23;
                float act_10_28 = x1_10_8 * sig_10_24;
                float act_11_29 = x1_11_9 * sig_11_25;
                float _min_12 = fminf(act_00_26, cl);
                act_00_26 = _min_12;
                float _min_13 = fminf(act_01_27, cl);
                act_01_27 = _min_13;
                float _min_14 = fminf(act_10_28, cl);
                act_10_28 = _min_14;
                float _min_15 = fminf(act_11_29, cl);
                act_11_29 = _min_15;
                quad[0] = lin_00_18 * act_00_26;
                quad[1] = lin_10_20 * act_10_28;
                quad[2] = lin_01_19 * act_01_27;
                quad[3] = lin_11_21 * act_11_29;
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
                float x0s_00_48 = x0c_00_44 * sc;
                float x0s_01_49 = x0c_01_45 * sc;
                float x0s_10_50 = x0c_10_46 * sc;
                float x0s_11_51 = x0c_11_47 * sc;
                float lin_00_52 = x0s_00_48 * sg;
                float lin_01_53 = x0s_01_49 * sg;
                float lin_10_54 = x0s_10_50 * sg;
                float lin_11_55 = x0s_11_51 * sg;
                float _exp2_8 = approx_exp2(-(x1_00_38 * fused));
                float _rcp_8 = approx_rcp(1.0f + _exp2_8);
                float sig_00_56 = _rcp_8;
                float _exp2_9 = approx_exp2(-(x1_01_39 * fused));
                float _rcp_9 = approx_rcp(1.0f + _exp2_9);
                float sig_01_57 = _rcp_9;
                float _exp2_10 = approx_exp2(-(x1_10_42 * fused));
                float _rcp_10 = approx_rcp(1.0f + _exp2_10);
                float sig_10_58 = _rcp_10;
                float _exp2_11 = approx_exp2(-(x1_11_43 * fused));
                float _rcp_11 = approx_rcp(1.0f + _exp2_11);
                float sig_11_59 = _rcp_11;
                float act_00_60 = x1_00_38 * sig_00_56;
                float act_01_61 = x1_01_39 * sig_01_57;
                float act_10_62 = x1_10_42 * sig_10_58;
                float act_11_63 = x1_11_43 * sig_11_59;
                float _min_20 = fminf(act_00_60, cl);
                act_00_60 = _min_20;
                float _min_21 = fminf(act_01_61, cl);
                act_01_61 = _min_21;
                float _min_22 = fminf(act_10_62, cl);
                act_10_62 = _min_22;
                float _min_23 = fminf(act_11_63, cl);
                act_11_63 = _min_23;
                quad[0] = lin_00_52 * act_00_60;
                quad[1] = lin_10_54 * act_10_62;
                quad[2] = lin_01_53 * act_01_61;
                quad[3] = lin_11_55 * act_11_63;
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
                float x0s_00_82 = x0c_00_78 * sc;
                float x0s_01_83 = x0c_01_79 * sc;
                float x0s_10_84 = x0c_10_80 * sc;
                float x0s_11_85 = x0c_11_81 * sc;
                float lin_00_86 = x0s_00_82 * sg;
                float lin_01_87 = x0s_01_83 * sg;
                float lin_10_88 = x0s_10_84 * sg;
                float lin_11_89 = x0s_11_85 * sg;
                float _exp2_12 = approx_exp2(-(x1_00_72 * fused));
                float _rcp_12 = approx_rcp(1.0f + _exp2_12);
                float sig_00_90 = _rcp_12;
                float _exp2_13 = approx_exp2(-(x1_01_73 * fused));
                float _rcp_13 = approx_rcp(1.0f + _exp2_13);
                float sig_01_91 = _rcp_13;
                float _exp2_14 = approx_exp2(-(x1_10_76 * fused));
                float _rcp_14 = approx_rcp(1.0f + _exp2_14);
                float sig_10_92 = _rcp_14;
                float _exp2_15 = approx_exp2(-(x1_11_77 * fused));
                float _rcp_15 = approx_rcp(1.0f + _exp2_15);
                float sig_11_93 = _rcp_15;
                float act_00_94 = x1_00_72 * sig_00_90;
                float act_01_95 = x1_01_73 * sig_01_91;
                float act_10_96 = x1_10_76 * sig_10_92;
                float act_11_97 = x1_11_77 * sig_11_93;
                float _min_28 = fminf(act_00_94, cl);
                act_00_94 = _min_28;
                float _min_29 = fminf(act_01_95, cl);
                act_01_95 = _min_29;
                float _min_30 = fminf(act_10_96, cl);
                act_10_96 = _min_30;
                float _min_31 = fminf(act_11_97, cl);
                act_11_97 = _min_31;
                quad[0] = lin_00_86 * act_00_94;
                quad[1] = lin_10_88 * act_10_96;
                quad[2] = lin_01_87 * act_01_95;
                quad[3] = lin_11_89 * act_11_97;
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
                float x0s_00_116 = x0c_00_112 * sc;
                float x0s_01_117 = x0c_01_113 * sc;
                float x0s_10_118 = x0c_10_114 * sc;
                float x0s_11_119 = x0c_11_115 * sc;
                float lin_00_120 = x0s_00_116 * sg;
                float lin_01_121 = x0s_01_117 * sg;
                float lin_10_122 = x0s_10_118 * sg;
                float lin_11_123 = x0s_11_119 * sg;
                float _exp2_16 = approx_exp2(-(x1_00_106 * fused));
                float _rcp_16 = approx_rcp(1.0f + _exp2_16);
                float sig_00_124 = _rcp_16;
                float _exp2_17 = approx_exp2(-(x1_01_107 * fused));
                float _rcp_17 = approx_rcp(1.0f + _exp2_17);
                float sig_01_125 = _rcp_17;
                float _exp2_18 = approx_exp2(-(x1_10_110 * fused));
                float _rcp_18 = approx_rcp(1.0f + _exp2_18);
                float sig_10_126 = _rcp_18;
                float _exp2_19 = approx_exp2(-(x1_11_111 * fused));
                float _rcp_19 = approx_rcp(1.0f + _exp2_19);
                float sig_11_127 = _rcp_19;
                float act_00_128 = x1_00_106 * sig_00_124;
                float act_01_129 = x1_01_107 * sig_01_125;
                float act_10_130 = x1_10_110 * sig_10_126;
                float act_11_131 = x1_11_111 * sig_11_127;
                float _min_36 = fminf(act_00_128, cl);
                act_00_128 = _min_36;
                float _min_37 = fminf(act_01_129, cl);
                act_01_129 = _min_37;
                float _min_38 = fminf(act_10_130, cl);
                act_10_130 = _min_38;
                float _min_39 = fminf(act_11_131, cl);
                act_11_131 = _min_39;
                quad[0] = lin_00_120 * act_00_128;
                quad[1] = lin_10_122 * act_10_130;
                quad[2] = lin_01_121 * act_01_129;
                quad[3] = lin_11_123 * act_11_131;
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
                float x0s_00_150 = x0c_00_146 * sc;
                float x0s_01_151 = x0c_01_147 * sc;
                float x0s_10_152 = x0c_10_148 * sc;
                float x0s_11_153 = x0c_11_149 * sc;
                float lin_00_154 = x0s_00_150 * sg;
                float lin_01_155 = x0s_01_151 * sg;
                float lin_10_156 = x0s_10_152 * sg;
                float lin_11_157 = x0s_11_153 * sg;
                float _exp2_20 = approx_exp2(-(x1_00_140 * fused));
                float _rcp_20 = approx_rcp(1.0f + _exp2_20);
                float sig_00_158 = _rcp_20;
                float _exp2_21 = approx_exp2(-(x1_01_141 * fused));
                float _rcp_21 = approx_rcp(1.0f + _exp2_21);
                float sig_01_159 = _rcp_21;
                float _exp2_22 = approx_exp2(-(x1_10_144 * fused));
                float _rcp_22 = approx_rcp(1.0f + _exp2_22);
                float sig_10_160 = _rcp_22;
                float _exp2_23 = approx_exp2(-(x1_11_145 * fused));
                float _rcp_23 = approx_rcp(1.0f + _exp2_23);
                float sig_11_161 = _rcp_23;
                float act_00_162 = x1_00_140 * sig_00_158;
                float act_01_163 = x1_01_141 * sig_01_159;
                float act_10_164 = x1_10_144 * sig_10_160;
                float act_11_165 = x1_11_145 * sig_11_161;
                float _min_44 = fminf(act_00_162, cl);
                act_00_162 = _min_44;
                float _min_45 = fminf(act_01_163, cl);
                act_01_163 = _min_45;
                float _min_46 = fminf(act_10_164, cl);
                act_10_164 = _min_46;
                float _min_47 = fminf(act_11_165, cl);
                act_11_165 = _min_47;
                quad[0] = lin_00_154 * act_00_162;
                quad[1] = lin_10_156 * act_10_164;
                quad[2] = lin_01_155 * act_01_163;
                quad[3] = lin_11_157 * act_11_165;
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
                float x0s_00_184 = x0c_00_180 * sc;
                float x0s_01_185 = x0c_01_181 * sc;
                float x0s_10_186 = x0c_10_182 * sc;
                float x0s_11_187 = x0c_11_183 * sc;
                float lin_00_188 = x0s_00_184 * sg;
                float lin_01_189 = x0s_01_185 * sg;
                float lin_10_190 = x0s_10_186 * sg;
                float lin_11_191 = x0s_11_187 * sg;
                float _exp2_24 = approx_exp2(-(x1_00_174 * fused));
                float _rcp_24 = approx_rcp(1.0f + _exp2_24);
                float sig_00_192 = _rcp_24;
                float _exp2_25 = approx_exp2(-(x1_01_175 * fused));
                float _rcp_25 = approx_rcp(1.0f + _exp2_25);
                float sig_01_193 = _rcp_25;
                float _exp2_26 = approx_exp2(-(x1_10_178 * fused));
                float _rcp_26 = approx_rcp(1.0f + _exp2_26);
                float sig_10_194 = _rcp_26;
                float _exp2_27 = approx_exp2(-(x1_11_179 * fused));
                float _rcp_27 = approx_rcp(1.0f + _exp2_27);
                float sig_11_195 = _rcp_27;
                float act_00_196 = x1_00_174 * sig_00_192;
                float act_01_197 = x1_01_175 * sig_01_193;
                float act_10_198 = x1_10_178 * sig_10_194;
                float act_11_199 = x1_11_179 * sig_11_195;
                float _min_52 = fminf(act_00_196, cl);
                act_00_196 = _min_52;
                float _min_53 = fminf(act_01_197, cl);
                act_01_197 = _min_53;
                float _min_54 = fminf(act_10_198, cl);
                act_10_198 = _min_54;
                float _min_55 = fminf(act_11_199, cl);
                act_11_199 = _min_55;
                quad[0] = lin_00_188 * act_00_196;
                quad[1] = lin_10_190 * act_10_198;
                quad[2] = lin_01_189 * act_01_197;
                quad[3] = lin_11_191 * act_11_199;
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
                float x0s_00_218 = x0c_00_214 * sc;
                float x0s_01_219 = x0c_01_215 * sc;
                float x0s_10_220 = x0c_10_216 * sc;
                float x0s_11_221 = x0c_11_217 * sc;
                float lin_00_222 = x0s_00_218 * sg;
                float lin_01_223 = x0s_01_219 * sg;
                float lin_10_224 = x0s_10_220 * sg;
                float lin_11_225 = x0s_11_221 * sg;
                float _exp2_28 = approx_exp2(-(x1_00_208 * fused));
                float _rcp_28 = approx_rcp(1.0f + _exp2_28);
                float sig_00_226 = _rcp_28;
                float _exp2_29 = approx_exp2(-(x1_01_209 * fused));
                float _rcp_29 = approx_rcp(1.0f + _exp2_29);
                float sig_01_227 = _rcp_29;
                float _exp2_30 = approx_exp2(-(x1_10_212 * fused));
                float _rcp_30 = approx_rcp(1.0f + _exp2_30);
                float sig_10_228 = _rcp_30;
                float _exp2_31 = approx_exp2(-(x1_11_213 * fused));
                float _rcp_31 = approx_rcp(1.0f + _exp2_31);
                float sig_11_229 = _rcp_31;
                float act_00_230 = x1_00_208 * sig_00_226;
                float act_01_231 = x1_01_209 * sig_01_227;
                float act_10_232 = x1_10_212 * sig_10_228;
                float act_11_233 = x1_11_213 * sig_11_229;
                float _min_60 = fminf(act_00_230, cl);
                act_00_230 = _min_60;
                float _min_61 = fminf(act_01_231, cl);
                act_01_231 = _min_61;
                float _min_62 = fminf(act_10_232, cl);
                act_10_232 = _min_62;
                float _min_63 = fminf(act_11_233, cl);
                act_11_233 = _min_63;
                quad[0] = lin_00_222 * act_00_230;
                quad[1] = lin_10_224 * act_10_232;
                quad[2] = lin_01_223 * act_01_231;
                quad[3] = lin_11_225 * act_11_233;
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
                        int padding_rows = (64 - valid_rows % 64) % 64;
                        tma_store_4d((&C), m_tile * 64, padding_rows, 1073741824, n_tile * 64 - (unsigned int)padding_rows + 1073741824, epi_staging_addr);
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
            int routed[4];
            unsigned int cta_mask = 1 << cta_rank;
            unsigned int _phase_k_done = 1;
            unsigned int _phase_work_full_1 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_1 = 0; _tile_iter_1 < grid_m / 2 * grid_n; _tile_iter_1++) {
                if (m_tile_1 >= (unsigned int)grid_m || n_tile_1 >= (unsigned int)bound_1) {
                    break;
                }
                route_base = n_tile_1 * 64 + (unsigned int)(cta_rank * 32) + (unsigned int)(warp_local * 4);
                for (int row = 0; row < 4; row++) {
                    routed[row] = route_map[route_base + row];
                }
                #pragma unroll 1
                for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                    mbarrier_wait(k_done_addr + (stage) * 8, _phase_k_done);
                    if (elect_sync()) {
                        tma_gather4_gmem2smem_mc_cta2(smem_b_addr + stage * 8192 + (unsigned int)(warp_local * 512), (&B), iter_k * 256, routed[0], routed[1], routed[2], routed[3], ((b_full_addr + (stage) * 8) & 0xFEFFFFFF), cta_mask);
                        tma_gather4_gmem2smem_mc_cta2(smem_b_addr + stage * 8192 + 4096 + (unsigned int)(warp_local * 512), (&B), iter_k * 256 + 128, routed[0], routed[1], routed[2], routed[3], ((b_full_addr + (stage) * 8) & 0xFEFFFFFF), cta_mask);
                    }
                    if (warp == 4) {
                        if (elect_sync()) {
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((b_full_addr + (stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(8192)) : "memory");
                        }
                    }
                    stage += 1;
                    if (stage == 5) { stage = 0; _phase_k_done ^= 1; }
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
                                :: "r"(smem_a_addr + stage_1 * 32768), "l"((&A)), "r"(0), "r"(m_tile_2 * 128), "r"(iter_k_1 * 2), "r"(expert_1),
                                   "r"(((a_full_addr + (stage_1) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(cta_mask_1)), "l"(0x12F0000000000000ULL) : "memory");
                        } else {
                            asm volatile(
                                "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                                :: "r"(smem_a_addr + stage_1 * 32768), "l"((&A)), "r"(0), "r"(m_tile_2 * 128), "r"(iter_k_1 * 2), "r"(expert_1),
                                   "r"(((a_full_addr + (stage_1) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(cta_mask_1)) : "memory");
                        }
                        if (sole_reader == 1) {
                            asm volatile(
                                "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4, %5}], [%6], %7, %8;"
                                :: "r"(smem_a_addr + stage_1 * 32768 + 16384), "l"((&A)), "r"(0), "r"(m_tile_2 * 128), "r"(iter_k_1 * 2 + 1), "r"(expert_1),
                                   "r"(((a_full_addr + (stage_1) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(cta_mask_1)), "l"(0x12F0000000000000ULL) : "memory");
                        } else {
                            asm volatile(
                                "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                                :: "r"(smem_a_addr + stage_1 * 32768 + 16384), "l"((&A)), "r"(0), "r"(m_tile_2 * 128), "r"(iter_k_1 * 2 + 1), "r"(expert_1),
                                   "r"(((a_full_addr + (stage_1) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(cta_mask_1)) : "memory");
                        }
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((a_full_addr + (stage_1) * 8) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                    }
                    stage_1 += 1;
                    if (stage_1 == 5) { stage_1 = 0; _phase_k_done_1 ^= 1; }
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
                        int _mma_a_lo_0 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_2) * 2048;
                        int _mma_b_lo_0 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_2) * 512;
                        mma_ss_step_cg2(_mma_a_lo_0, _mma_b_lo_0, (tmem_accum + (acc_stage_1 * 64)), 269484048, ((((1) ? ((iter_k_2 == 0) ? 1 : 0) : 0)) ? 0 : 1), 0x40004040U, 0x40004040U);
                        int _mma_a_lo_1 = (((smem_a_addr + 32) >> 4) & 0x3FFF) + (stage_2) * 2048;
                        int _mma_b_lo_1 = (((smem_b_addr + 32) >> 4) & 0x3FFF) + (stage_2) * 512;
                        mma_ss_step_cg2(_mma_a_lo_1, _mma_b_lo_1, (tmem_accum + (acc_stage_1 * 64)), 269484048, ((((0) ? ((iter_k_2 == 0) ? 1 : 0) : 0)) ? 0 : 1), 0x40004040U, 0x40004040U);
                        int _mma_a_lo_2 = (((smem_a_addr + 64) >> 4) & 0x3FFF) + (stage_2) * 2048;
                        int _mma_b_lo_2 = (((smem_b_addr + 64) >> 4) & 0x3FFF) + (stage_2) * 512;
                        mma_ss_step_cg2(_mma_a_lo_2, _mma_b_lo_2, (tmem_accum + (acc_stage_1 * 64)), 269484048, ((((0) ? ((iter_k_2 == 0) ? 1 : 0) : 0)) ? 0 : 1), 0x40004040U, 0x40004040U);
                        int _mma_a_lo_3 = (((smem_a_addr + 96) >> 4) & 0x3FFF) + (stage_2) * 2048;
                        int _mma_b_lo_3 = (((smem_b_addr + 96) >> 4) & 0x3FFF) + (stage_2) * 512;
                        mma_ss_step_cg2(_mma_a_lo_3, _mma_b_lo_3, (tmem_accum + (acc_stage_1 * 64)), 269484048, ((((0) ? ((iter_k_2 == 0) ? 1 : 0) : 0)) ? 0 : 1), 0x40004040U, 0x40004040U);
                        int _mma_a_lo_4 = (((smem_a_addr + 16384) >> 4) & 0x3FFF) + (stage_2) * 2048;
                        int _mma_b_lo_4 = (((smem_b_addr + 4096) >> 4) & 0x3FFF) + (stage_2) * 512;
                        mma_ss_step_cg2(_mma_a_lo_4, _mma_b_lo_4, (tmem_accum + (acc_stage_1 * 64)), 269484048, ((((0) ? ((iter_k_2 == 0) ? 1 : 0) : 0)) ? 0 : 1), 0x40004040U, 0x40004040U);
                        int _mma_a_lo_5 = (((smem_a_addr + 16416) >> 4) & 0x3FFF) + (stage_2) * 2048;
                        int _mma_b_lo_5 = (((smem_b_addr + 4128) >> 4) & 0x3FFF) + (stage_2) * 512;
                        mma_ss_step_cg2(_mma_a_lo_5, _mma_b_lo_5, (tmem_accum + (acc_stage_1 * 64)), 269484048, ((((0) ? ((iter_k_2 == 0) ? 1 : 0) : 0)) ? 0 : 1), 0x40004040U, 0x40004040U);
                        int _mma_a_lo_6 = (((smem_a_addr + 16448) >> 4) & 0x3FFF) + (stage_2) * 2048;
                        int _mma_b_lo_6 = (((smem_b_addr + 4160) >> 4) & 0x3FFF) + (stage_2) * 512;
                        mma_ss_step_cg2(_mma_a_lo_6, _mma_b_lo_6, (tmem_accum + (acc_stage_1 * 64)), 269484048, ((((0) ? ((iter_k_2 == 0) ? 1 : 0) : 0)) ? 0 : 1), 0x40004040U, 0x40004040U);
                        int _mma_a_lo_7 = (((smem_a_addr + 16480) >> 4) & 0x3FFF) + (stage_2) * 2048;
                        int _mma_b_lo_7 = (((smem_b_addr + 4192) >> 4) & 0x3FFF) + (stage_2) * 512;
                        mma_ss_step_cg2(_mma_a_lo_7, _mma_b_lo_7, (tmem_accum + (acc_stage_1 * 64)), 269484048, ((((0) ? ((iter_k_2 == 0) ? 1 : 0) : 0)) ? 0 : 1), 0x40004040U, 0x40004040U);
                        elect_commit_cg2_multicast(k_done_addr + (stage_2) * 8, (uint16_t)(3));
                        if (iter_k_2 + 1 == K_tiles) {
                            elect_commit_cg2_multicast(mma_full_addr + (acc_stage_1) * 8, (uint16_t)(3));
                        }
                        stage_2 += 1;
                        if (stage_2 == 5) { stage_2 = 0; _phase_a_full ^= 1; _phase_b_full ^= 1; }
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
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(128));
    }
}

} // extern "C"
