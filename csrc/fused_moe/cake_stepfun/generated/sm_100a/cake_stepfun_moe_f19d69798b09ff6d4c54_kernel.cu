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
#define TMEM_NCOLS 512
#define TMEM_ACCUM_OFFSET 0
#define NUM_K_PIPE_STAGES 6
#define NUM_MMA_PIPE_STAGES 2
#define NUM_WORK_PIPE_STAGES 3
#define NUM_THROTTLE_PIPE_STAGES 3
#define NUM_DRAIN_PIPE_STAGES 1
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 16384
#define SMEM_SMEM_B_OFF 99328
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 16384
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
kernel_cake_stepfun_moe_f19d69798b09ff6d4c54(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap C, int* __restrict__ route_map, int* __restrict__ tile_expert, int* __restrict__ tile_mn_limit, int* __restrict__ num_non_exiting_ctas, float* __restrict__ scale_c, float* __restrict__ scale_gate, float* __restrict__ clamp_limit, float* __restrict__ act_alpha, float* __restrict__ act_beta, int M_out, int K, int grid_m, int grid_n, int K_tiles)
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
    #define b_full_addr (mbar_base + 48)
    #define k_done_addr (mbar_base + 96)
    #define mma_full_addr (mbar_base + 144)
    #define mma_free_addr (mbar_base + 160)
    #define work_full_addr (mbar_base + 176)
    #define work_empty_addr (mbar_base + 200)
    #define throttle_full_addr (mbar_base + 224)
    #define throttle_empty_addr (mbar_base + 248)
    #define drain_full_addr (mbar_base + 272)

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
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + 99328);
    const int smem_b_addr = smem + 99328;
    uint16_t* epi_staging = reinterpret_cast<uint16_t*>(smem_raw + 197632);
    const int epi_staging_addr = smem + 197632;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 201728);
    const int work_response_addr = smem + 201728;
    unsigned int* fast_drain_response = reinterpret_cast<unsigned int*>(smem_raw + 201776);
    const int fast_drain_response_addr = smem + 201776;
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if ((int)blockIdx.y >= num_non_exiting_ctas[0]) return;

    // Mbarrier init (10 pipeline groups, 0 ordered-sequence groups, 35 barriers)
    // Mbarriers at smem_raw[0..280)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'k_pipe' ---
            // a_full: 6 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            mbarrier_init(smem + 8, 2);
            mbarrier_init(smem + 16, 2);
            mbarrier_init(smem + 24, 2);
            mbarrier_init(smem + 32, 2);
            mbarrier_init(smem + 40, 2);
            // b_full: 6 barriers, init_count=2
            mbarrier_init(smem + 48, 2);
            mbarrier_init(smem + 56, 2);
            mbarrier_init(smem + 64, 2);
            mbarrier_init(smem + 72, 2);
            mbarrier_init(smem + 80, 2);
            mbarrier_init(smem + 88, 2);
            // k_done: 6 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            // --- pipeline 'mma_pipe' ---
            // mma_full: 2 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            // mma_free: 2 barriers, init_count=256
            mbarrier_init(smem + 160, 256);
            mbarrier_init(smem + 168, 256);
            // --- pipeline 'work_pipe' ---
            // work_full: 3 barriers, init_count=1
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            mbarrier_init(smem + 192, 1);
            // work_empty: 3 barriers, init_count=960
            mbarrier_init(smem + 200, 960);
            mbarrier_init(smem + 208, 960);
            mbarrier_init(smem + 216, 960);
            // --- pipeline 'throttle_pipe' ---
            // throttle_full: 3 barriers, init_count=32
            mbarrier_init(smem + 224, 32);
            mbarrier_init(smem + 232, 32);
            mbarrier_init(smem + 240, 32);
            // throttle_empty: 3 barriers, init_count=32
            mbarrier_init(smem + 248, 32);
            mbarrier_init(smem + 256, 32);
            mbarrier_init(smem + 264, 32);
            // --- pipeline 'drain_pipe' ---
            // drain_full: 1 barriers, init_count=1
            mbarrier_init(smem + 272, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 280);
    if (warp == 0) {
        int _tmem_hold = smem + 280;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
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
                int valid_rows = (unsigned int)tile_mn_limit[n_tile] - n_tile * 256;
                float sc = scale_c[expert];
                float sg = scale_gate[expert];
                float cl = clamp_limit[expert];
                float neg_cl = -cl;
                float fused = 1.4426950216293335f * sg;
                mbarrier_wait(mma_full_addr + (acc_stage) * 8, _phase_mma_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int acc_offset = acc_stage * 256;
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
                        int padding_rows = (256 - valid_rows % 256) % 256;
                        tma_store_4d((&C), m_tile * 64, padding_rows, 1073741824, n_tile * 256 - (unsigned int)padding_rows + 1073741824, epi_staging_addr);
                    }
                }
                asm volatile("cp.async.bulk.commit_group;");
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                int acc_offset_238 = acc_stage * 256 + 64;
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
                float x0s_00_254 = x0c_00_250 * sc;
                float x0s_01_255 = x0c_01_251 * sc;
                float x0s_10_256 = x0c_10_252 * sc;
                float x0s_11_257 = x0c_11_253 * sc;
                float lin_00_258 = x0s_00_254 * sg;
                float lin_01_259 = x0s_01_255 * sg;
                float lin_10_260 = x0s_10_256 * sg;
                float lin_11_261 = x0s_11_257 * sg;
                float _exp2_32 = approx_exp2(-(x1_00_244 * fused));
                float _rcp_32 = approx_rcp(1.0f + _exp2_32);
                float sig_00_262 = _rcp_32;
                float _exp2_33 = approx_exp2(-(x1_01_245 * fused));
                float _rcp_33 = approx_rcp(1.0f + _exp2_33);
                float sig_01_263 = _rcp_33;
                float _exp2_34 = approx_exp2(-(x1_10_248 * fused));
                float _rcp_34 = approx_rcp(1.0f + _exp2_34);
                float sig_10_264 = _rcp_34;
                float _exp2_35 = approx_exp2(-(x1_11_249 * fused));
                float _rcp_35 = approx_rcp(1.0f + _exp2_35);
                float sig_11_265 = _rcp_35;
                float act_00_266 = x1_00_244 * sig_00_262;
                float act_01_267 = x1_01_245 * sig_01_263;
                float act_10_268 = x1_10_248 * sig_10_264;
                float act_11_269 = x1_11_249 * sig_11_265;
                float _min_68 = fminf(act_00_266, cl);
                act_00_266 = _min_68;
                float _min_69 = fminf(act_01_267, cl);
                act_01_267 = _min_69;
                float _min_70 = fminf(act_10_268, cl);
                act_10_268 = _min_70;
                float _min_71 = fminf(act_11_269, cl);
                act_11_269 = _min_71;
                quad[0] = lin_00_258 * act_00_266;
                quad[1] = lin_10_260 * act_10_268;
                quad[2] = lin_01_259 * act_01_267;
                quad[3] = lin_11_261 * act_11_269;
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
                float x0s_00_288 = x0c_00_284 * sc;
                float x0s_01_289 = x0c_01_285 * sc;
                float x0s_10_290 = x0c_10_286 * sc;
                float x0s_11_291 = x0c_11_287 * sc;
                float lin_00_292 = x0s_00_288 * sg;
                float lin_01_293 = x0s_01_289 * sg;
                float lin_10_294 = x0s_10_290 * sg;
                float lin_11_295 = x0s_11_291 * sg;
                float _exp2_36 = approx_exp2(-(x1_00_278 * fused));
                float _rcp_36 = approx_rcp(1.0f + _exp2_36);
                float sig_00_296 = _rcp_36;
                float _exp2_37 = approx_exp2(-(x1_01_279 * fused));
                float _rcp_37 = approx_rcp(1.0f + _exp2_37);
                float sig_01_297 = _rcp_37;
                float _exp2_38 = approx_exp2(-(x1_10_282 * fused));
                float _rcp_38 = approx_rcp(1.0f + _exp2_38);
                float sig_10_298 = _rcp_38;
                float _exp2_39 = approx_exp2(-(x1_11_283 * fused));
                float _rcp_39 = approx_rcp(1.0f + _exp2_39);
                float sig_11_299 = _rcp_39;
                float act_00_300 = x1_00_278 * sig_00_296;
                float act_01_301 = x1_01_279 * sig_01_297;
                float act_10_302 = x1_10_282 * sig_10_298;
                float act_11_303 = x1_11_283 * sig_11_299;
                float _min_76 = fminf(act_00_300, cl);
                act_00_300 = _min_76;
                float _min_77 = fminf(act_01_301, cl);
                act_01_301 = _min_77;
                float _min_78 = fminf(act_10_302, cl);
                act_10_302 = _min_78;
                float _min_79 = fminf(act_11_303, cl);
                act_11_303 = _min_79;
                quad[0] = lin_00_292 * act_00_300;
                quad[1] = lin_10_294 * act_10_302;
                quad[2] = lin_01_293 * act_01_301;
                quad[3] = lin_11_295 * act_11_303;
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
                float x0s_00_322 = x0c_00_318 * sc;
                float x0s_01_323 = x0c_01_319 * sc;
                float x0s_10_324 = x0c_10_320 * sc;
                float x0s_11_325 = x0c_11_321 * sc;
                float lin_00_326 = x0s_00_322 * sg;
                float lin_01_327 = x0s_01_323 * sg;
                float lin_10_328 = x0s_10_324 * sg;
                float lin_11_329 = x0s_11_325 * sg;
                float _exp2_40 = approx_exp2(-(x1_00_312 * fused));
                float _rcp_40 = approx_rcp(1.0f + _exp2_40);
                float sig_00_330 = _rcp_40;
                float _exp2_41 = approx_exp2(-(x1_01_313 * fused));
                float _rcp_41 = approx_rcp(1.0f + _exp2_41);
                float sig_01_331 = _rcp_41;
                float _exp2_42 = approx_exp2(-(x1_10_316 * fused));
                float _rcp_42 = approx_rcp(1.0f + _exp2_42);
                float sig_10_332 = _rcp_42;
                float _exp2_43 = approx_exp2(-(x1_11_317 * fused));
                float _rcp_43 = approx_rcp(1.0f + _exp2_43);
                float sig_11_333 = _rcp_43;
                float act_00_334 = x1_00_312 * sig_00_330;
                float act_01_335 = x1_01_313 * sig_01_331;
                float act_10_336 = x1_10_316 * sig_10_332;
                float act_11_337 = x1_11_317 * sig_11_333;
                float _min_84 = fminf(act_00_334, cl);
                act_00_334 = _min_84;
                float _min_85 = fminf(act_01_335, cl);
                act_01_335 = _min_85;
                float _min_86 = fminf(act_10_336, cl);
                act_10_336 = _min_86;
                float _min_87 = fminf(act_11_337, cl);
                act_11_337 = _min_87;
                quad[0] = lin_00_326 * act_00_334;
                quad[1] = lin_10_328 * act_10_336;
                quad[2] = lin_01_327 * act_01_335;
                quad[3] = lin_11_329 * act_11_337;
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
                float x0s_00_356 = x0c_00_352 * sc;
                float x0s_01_357 = x0c_01_353 * sc;
                float x0s_10_358 = x0c_10_354 * sc;
                float x0s_11_359 = x0c_11_355 * sc;
                float lin_00_360 = x0s_00_356 * sg;
                float lin_01_361 = x0s_01_357 * sg;
                float lin_10_362 = x0s_10_358 * sg;
                float lin_11_363 = x0s_11_359 * sg;
                float _exp2_44 = approx_exp2(-(x1_00_346 * fused));
                float _rcp_44 = approx_rcp(1.0f + _exp2_44);
                float sig_00_364 = _rcp_44;
                float _exp2_45 = approx_exp2(-(x1_01_347 * fused));
                float _rcp_45 = approx_rcp(1.0f + _exp2_45);
                float sig_01_365 = _rcp_45;
                float _exp2_46 = approx_exp2(-(x1_10_350 * fused));
                float _rcp_46 = approx_rcp(1.0f + _exp2_46);
                float sig_10_366 = _rcp_46;
                float _exp2_47 = approx_exp2(-(x1_11_351 * fused));
                float _rcp_47 = approx_rcp(1.0f + _exp2_47);
                float sig_11_367 = _rcp_47;
                float act_00_368 = x1_00_346 * sig_00_364;
                float act_01_369 = x1_01_347 * sig_01_365;
                float act_10_370 = x1_10_350 * sig_10_366;
                float act_11_371 = x1_11_351 * sig_11_367;
                float _min_92 = fminf(act_00_368, cl);
                act_00_368 = _min_92;
                float _min_93 = fminf(act_01_369, cl);
                act_01_369 = _min_93;
                float _min_94 = fminf(act_10_370, cl);
                act_10_370 = _min_94;
                float _min_95 = fminf(act_11_371, cl);
                act_11_371 = _min_95;
                quad[0] = lin_00_360 * act_00_368;
                quad[1] = lin_10_362 * act_10_370;
                quad[2] = lin_01_361 * act_01_369;
                quad[3] = lin_11_363 * act_11_371;
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
                float x0s_00_390 = x0c_00_386 * sc;
                float x0s_01_391 = x0c_01_387 * sc;
                float x0s_10_392 = x0c_10_388 * sc;
                float x0s_11_393 = x0c_11_389 * sc;
                float lin_00_394 = x0s_00_390 * sg;
                float lin_01_395 = x0s_01_391 * sg;
                float lin_10_396 = x0s_10_392 * sg;
                float lin_11_397 = x0s_11_393 * sg;
                float _exp2_48 = approx_exp2(-(x1_00_380 * fused));
                float _rcp_48 = approx_rcp(1.0f + _exp2_48);
                float sig_00_398 = _rcp_48;
                float _exp2_49 = approx_exp2(-(x1_01_381 * fused));
                float _rcp_49 = approx_rcp(1.0f + _exp2_49);
                float sig_01_399 = _rcp_49;
                float _exp2_50 = approx_exp2(-(x1_10_384 * fused));
                float _rcp_50 = approx_rcp(1.0f + _exp2_50);
                float sig_10_400 = _rcp_50;
                float _exp2_51 = approx_exp2(-(x1_11_385 * fused));
                float _rcp_51 = approx_rcp(1.0f + _exp2_51);
                float sig_11_401 = _rcp_51;
                float act_00_402 = x1_00_380 * sig_00_398;
                float act_01_403 = x1_01_381 * sig_01_399;
                float act_10_404 = x1_10_384 * sig_10_400;
                float act_11_405 = x1_11_385 * sig_11_401;
                float _min_100 = fminf(act_00_402, cl);
                act_00_402 = _min_100;
                float _min_101 = fminf(act_01_403, cl);
                act_01_403 = _min_101;
                float _min_102 = fminf(act_10_404, cl);
                act_10_404 = _min_102;
                float _min_103 = fminf(act_11_405, cl);
                act_11_405 = _min_103;
                quad[0] = lin_00_394 * act_00_402;
                quad[1] = lin_10_396 * act_10_404;
                quad[2] = lin_01_395 * act_01_403;
                quad[3] = lin_11_397 * act_11_405;
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
                float x0s_00_424 = x0c_00_420 * sc;
                float x0s_01_425 = x0c_01_421 * sc;
                float x0s_10_426 = x0c_10_422 * sc;
                float x0s_11_427 = x0c_11_423 * sc;
                float lin_00_428 = x0s_00_424 * sg;
                float lin_01_429 = x0s_01_425 * sg;
                float lin_10_430 = x0s_10_426 * sg;
                float lin_11_431 = x0s_11_427 * sg;
                float _exp2_52 = approx_exp2(-(x1_00_414 * fused));
                float _rcp_52 = approx_rcp(1.0f + _exp2_52);
                float sig_00_432 = _rcp_52;
                float _exp2_53 = approx_exp2(-(x1_01_415 * fused));
                float _rcp_53 = approx_rcp(1.0f + _exp2_53);
                float sig_01_433 = _rcp_53;
                float _exp2_54 = approx_exp2(-(x1_10_418 * fused));
                float _rcp_54 = approx_rcp(1.0f + _exp2_54);
                float sig_10_434 = _rcp_54;
                float _exp2_55 = approx_exp2(-(x1_11_419 * fused));
                float _rcp_55 = approx_rcp(1.0f + _exp2_55);
                float sig_11_435 = _rcp_55;
                float act_00_436 = x1_00_414 * sig_00_432;
                float act_01_437 = x1_01_415 * sig_01_433;
                float act_10_438 = x1_10_418 * sig_10_434;
                float act_11_439 = x1_11_419 * sig_11_435;
                float _min_108 = fminf(act_00_436, cl);
                act_00_436 = _min_108;
                float _min_109 = fminf(act_01_437, cl);
                act_01_437 = _min_109;
                float _min_110 = fminf(act_10_438, cl);
                act_10_438 = _min_110;
                float _min_111 = fminf(act_11_439, cl);
                act_11_439 = _min_111;
                quad[0] = lin_00_428 * act_00_436;
                quad[1] = lin_10_430 * act_10_438;
                quad[2] = lin_01_429 * act_01_437;
                quad[3] = lin_11_431 * act_11_439;
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
                float x0s_00_458 = x0c_00_454 * sc;
                float x0s_01_459 = x0c_01_455 * sc;
                float x0s_10_460 = x0c_10_456 * sc;
                float x0s_11_461 = x0c_11_457 * sc;
                float lin_00_462 = x0s_00_458 * sg;
                float lin_01_463 = x0s_01_459 * sg;
                float lin_10_464 = x0s_10_460 * sg;
                float lin_11_465 = x0s_11_461 * sg;
                float _exp2_56 = approx_exp2(-(x1_00_448 * fused));
                float _rcp_56 = approx_rcp(1.0f + _exp2_56);
                float sig_00_466 = _rcp_56;
                float _exp2_57 = approx_exp2(-(x1_01_449 * fused));
                float _rcp_57 = approx_rcp(1.0f + _exp2_57);
                float sig_01_467 = _rcp_57;
                float _exp2_58 = approx_exp2(-(x1_10_452 * fused));
                float _rcp_58 = approx_rcp(1.0f + _exp2_58);
                float sig_10_468 = _rcp_58;
                float _exp2_59 = approx_exp2(-(x1_11_453 * fused));
                float _rcp_59 = approx_rcp(1.0f + _exp2_59);
                float sig_11_469 = _rcp_59;
                float act_00_470 = x1_00_448 * sig_00_466;
                float act_01_471 = x1_01_449 * sig_01_467;
                float act_10_472 = x1_10_452 * sig_10_468;
                float act_11_473 = x1_11_453 * sig_11_469;
                float _min_116 = fminf(act_00_470, cl);
                act_00_470 = _min_116;
                float _min_117 = fminf(act_01_471, cl);
                act_01_471 = _min_117;
                float _min_118 = fminf(act_10_472, cl);
                act_10_472 = _min_118;
                float _min_119 = fminf(act_11_473, cl);
                act_11_473 = _min_119;
                quad[0] = lin_00_462 * act_00_470;
                quad[1] = lin_10_464 * act_10_472;
                quad[2] = lin_01_463 * act_01_471;
                quad[3] = lin_11_465 * act_11_473;
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
                float x0s_00_492 = x0c_00_488 * sc;
                float x0s_01_493 = x0c_01_489 * sc;
                float x0s_10_494 = x0c_10_490 * sc;
                float x0s_11_495 = x0c_11_491 * sc;
                float lin_00_496 = x0s_00_492 * sg;
                float lin_01_497 = x0s_01_493 * sg;
                float lin_10_498 = x0s_10_494 * sg;
                float lin_11_499 = x0s_11_495 * sg;
                float _exp2_60 = approx_exp2(-(x1_00_482 * fused));
                float _rcp_60 = approx_rcp(1.0f + _exp2_60);
                float sig_00_500 = _rcp_60;
                float _exp2_61 = approx_exp2(-(x1_01_483 * fused));
                float _rcp_61 = approx_rcp(1.0f + _exp2_61);
                float sig_01_501 = _rcp_61;
                float _exp2_62 = approx_exp2(-(x1_10_486 * fused));
                float _rcp_62 = approx_rcp(1.0f + _exp2_62);
                float sig_10_502 = _rcp_62;
                float _exp2_63 = approx_exp2(-(x1_11_487 * fused));
                float _rcp_63 = approx_rcp(1.0f + _exp2_63);
                float sig_11_503 = _rcp_63;
                float act_00_504 = x1_00_482 * sig_00_500;
                float act_01_505 = x1_01_483 * sig_01_501;
                float act_10_506 = x1_10_486 * sig_10_502;
                float act_11_507 = x1_11_487 * sig_11_503;
                float _min_124 = fminf(act_00_504, cl);
                act_00_504 = _min_124;
                float _min_125 = fminf(act_01_505, cl);
                act_01_505 = _min_125;
                float _min_126 = fminf(act_10_506, cl);
                act_10_506 = _min_126;
                float _min_127 = fminf(act_11_507, cl);
                act_11_507 = _min_127;
                quad[0] = lin_00_496 * act_00_504;
                quad[1] = lin_10_498 * act_10_506;
                quad[2] = lin_01_497 * act_01_505;
                quad[3] = lin_11_499 * act_11_507;
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
                        int padding_rows_1 = (256 - valid_rows % 256) % 256;
                        tma_store_4d((&C), m_tile * 64, padding_rows_1 + 64, 1073741824, n_tile * 256 - (unsigned int)padding_rows_1 + 1073741824, epi_staging_addr);
                    }
                }
                asm volatile("cp.async.bulk.commit_group;");
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                int acc_offset_512 = acc_stage * 256 + 128;
                float _tmem_load_4[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[31]))
                    : "r"(taddr + (unsigned int)acc_offset_512));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float _tmem_load_5[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[31]))
                    : "r"(taddr + 1048576 + (unsigned int)acc_offset_512));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int base_row_513 = warp_0 * 16 + lane_1 / 4 * 2;
                asm volatile("cp.async.bulk.wait_group.read 0;");
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                int local_token0_514 = lane_1 % 4 * 2;
                int local_token1_515 = local_token0_514 + 1;
                float x0_00_516 = _tmem_load_4[0];
                float x0_01_517 = _tmem_load_4[1];
                float x1_00_518 = _tmem_load_4[2];
                float x1_01_519 = _tmem_load_4[3];
                float x0_10_520 = _tmem_load_5[0];
                float x0_11_521 = _tmem_load_5[1];
                float x1_10_522 = _tmem_load_5[2];
                float x1_11_523 = _tmem_load_5[3];
                float _max_64 = max_noftz(x0_00_516, neg_cl);
                float _min_128 = fminf(_max_64, cl);
                float x0c_00_524 = _min_128;
                float _max_65 = max_noftz(x0_01_517, neg_cl);
                float _min_129 = fminf(_max_65, cl);
                float x0c_01_525 = _min_129;
                float _max_66 = max_noftz(x0_10_520, neg_cl);
                float _min_130 = fminf(_max_66, cl);
                float x0c_10_526 = _min_130;
                float _max_67 = max_noftz(x0_11_521, neg_cl);
                float _min_131 = fminf(_max_67, cl);
                float x0c_11_527 = _min_131;
                float x0s_00_528 = x0c_00_524 * sc;
                float x0s_01_529 = x0c_01_525 * sc;
                float x0s_10_530 = x0c_10_526 * sc;
                float x0s_11_531 = x0c_11_527 * sc;
                float lin_00_532 = x0s_00_528 * sg;
                float lin_01_533 = x0s_01_529 * sg;
                float lin_10_534 = x0s_10_530 * sg;
                float lin_11_535 = x0s_11_531 * sg;
                float _exp2_64 = approx_exp2(-(x1_00_518 * fused));
                float _rcp_64 = approx_rcp(1.0f + _exp2_64);
                float sig_00_536 = _rcp_64;
                float _exp2_65 = approx_exp2(-(x1_01_519 * fused));
                float _rcp_65 = approx_rcp(1.0f + _exp2_65);
                float sig_01_537 = _rcp_65;
                float _exp2_66 = approx_exp2(-(x1_10_522 * fused));
                float _rcp_66 = approx_rcp(1.0f + _exp2_66);
                float sig_10_538 = _rcp_66;
                float _exp2_67 = approx_exp2(-(x1_11_523 * fused));
                float _rcp_67 = approx_rcp(1.0f + _exp2_67);
                float sig_11_539 = _rcp_67;
                float act_00_540 = x1_00_518 * sig_00_536;
                float act_01_541 = x1_01_519 * sig_01_537;
                float act_10_542 = x1_10_522 * sig_10_538;
                float act_11_543 = x1_11_523 * sig_11_539;
                float _min_132 = fminf(act_00_540, cl);
                act_00_540 = _min_132;
                float _min_133 = fminf(act_01_541, cl);
                act_01_541 = _min_133;
                float _min_134 = fminf(act_10_542, cl);
                act_10_542 = _min_134;
                float _min_135 = fminf(act_11_543, cl);
                act_11_543 = _min_135;
                quad[0] = lin_00_532 * act_00_540;
                quad[1] = lin_10_534 * act_10_542;
                quad[2] = lin_01_533 * act_01_541;
                quad[3] = lin_11_535 * act_11_543;
                uint32_t _fp8_16[1];
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
                    _fp8_16[0] = _packed;
                }
                int off0_544 = local_token0_514 * 64 + base_row_513;
                int off1_545 = local_token1_515 * 64 + base_row_513;
                int swz0_546 = off0_544 ^ (off0_544 >> 7 & 3) << 4;
                int swz1_547 = off1_545 ^ (off1_545 >> 7 & 3) << 4;
                epi_staging[swz0_546 >> 1] = _fp8_16[0] & 65535;
                epi_staging[swz1_547 >> 1] = _fp8_16[0] >> 16;
                int local_token0_548 = lane_1 % 4 * 2 + 8;
                int local_token1_549 = local_token0_548 + 1;
                float x0_00_550 = _tmem_load_4[4];
                float x0_01_551 = _tmem_load_4[5];
                float x1_00_552 = _tmem_load_4[6];
                float x1_01_553 = _tmem_load_4[7];
                float x0_10_554 = _tmem_load_5[4];
                float x0_11_555 = _tmem_load_5[5];
                float x1_10_556 = _tmem_load_5[6];
                float x1_11_557 = _tmem_load_5[7];
                float _max_68 = max_noftz(x0_00_550, neg_cl);
                float _min_136 = fminf(_max_68, cl);
                float x0c_00_558 = _min_136;
                float _max_69 = max_noftz(x0_01_551, neg_cl);
                float _min_137 = fminf(_max_69, cl);
                float x0c_01_559 = _min_137;
                float _max_70 = max_noftz(x0_10_554, neg_cl);
                float _min_138 = fminf(_max_70, cl);
                float x0c_10_560 = _min_138;
                float _max_71 = max_noftz(x0_11_555, neg_cl);
                float _min_139 = fminf(_max_71, cl);
                float x0c_11_561 = _min_139;
                float x0s_00_562 = x0c_00_558 * sc;
                float x0s_01_563 = x0c_01_559 * sc;
                float x0s_10_564 = x0c_10_560 * sc;
                float x0s_11_565 = x0c_11_561 * sc;
                float lin_00_566 = x0s_00_562 * sg;
                float lin_01_567 = x0s_01_563 * sg;
                float lin_10_568 = x0s_10_564 * sg;
                float lin_11_569 = x0s_11_565 * sg;
                float _exp2_68 = approx_exp2(-(x1_00_552 * fused));
                float _rcp_68 = approx_rcp(1.0f + _exp2_68);
                float sig_00_570 = _rcp_68;
                float _exp2_69 = approx_exp2(-(x1_01_553 * fused));
                float _rcp_69 = approx_rcp(1.0f + _exp2_69);
                float sig_01_571 = _rcp_69;
                float _exp2_70 = approx_exp2(-(x1_10_556 * fused));
                float _rcp_70 = approx_rcp(1.0f + _exp2_70);
                float sig_10_572 = _rcp_70;
                float _exp2_71 = approx_exp2(-(x1_11_557 * fused));
                float _rcp_71 = approx_rcp(1.0f + _exp2_71);
                float sig_11_573 = _rcp_71;
                float act_00_574 = x1_00_552 * sig_00_570;
                float act_01_575 = x1_01_553 * sig_01_571;
                float act_10_576 = x1_10_556 * sig_10_572;
                float act_11_577 = x1_11_557 * sig_11_573;
                float _min_140 = fminf(act_00_574, cl);
                act_00_574 = _min_140;
                float _min_141 = fminf(act_01_575, cl);
                act_01_575 = _min_141;
                float _min_142 = fminf(act_10_576, cl);
                act_10_576 = _min_142;
                float _min_143 = fminf(act_11_577, cl);
                act_11_577 = _min_143;
                quad[0] = lin_00_566 * act_00_574;
                quad[1] = lin_10_568 * act_10_576;
                quad[2] = lin_01_567 * act_01_575;
                quad[3] = lin_11_569 * act_11_577;
                uint32_t _fp8_17[1];
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
                    _fp8_17[0] = _packed;
                }
                int off0_578 = local_token0_548 * 64 + base_row_513;
                int off1_579 = local_token1_549 * 64 + base_row_513;
                int swz0_580 = off0_578 ^ (off0_578 >> 7 & 3) << 4;
                int swz1_581 = off1_579 ^ (off1_579 >> 7 & 3) << 4;
                epi_staging[swz0_580 >> 1] = _fp8_17[0] & 65535;
                epi_staging[swz1_581 >> 1] = _fp8_17[0] >> 16;
                int local_token0_582 = lane_1 % 4 * 2 + 16;
                int local_token1_583 = local_token0_582 + 1;
                float x0_00_584 = _tmem_load_4[8];
                float x0_01_585 = _tmem_load_4[9];
                float x1_00_586 = _tmem_load_4[10];
                float x1_01_587 = _tmem_load_4[11];
                float x0_10_588 = _tmem_load_5[8];
                float x0_11_589 = _tmem_load_5[9];
                float x1_10_590 = _tmem_load_5[10];
                float x1_11_591 = _tmem_load_5[11];
                float _max_72 = max_noftz(x0_00_584, neg_cl);
                float _min_144 = fminf(_max_72, cl);
                float x0c_00_592 = _min_144;
                float _max_73 = max_noftz(x0_01_585, neg_cl);
                float _min_145 = fminf(_max_73, cl);
                float x0c_01_593 = _min_145;
                float _max_74 = max_noftz(x0_10_588, neg_cl);
                float _min_146 = fminf(_max_74, cl);
                float x0c_10_594 = _min_146;
                float _max_75 = max_noftz(x0_11_589, neg_cl);
                float _min_147 = fminf(_max_75, cl);
                float x0c_11_595 = _min_147;
                float x0s_00_596 = x0c_00_592 * sc;
                float x0s_01_597 = x0c_01_593 * sc;
                float x0s_10_598 = x0c_10_594 * sc;
                float x0s_11_599 = x0c_11_595 * sc;
                float lin_00_600 = x0s_00_596 * sg;
                float lin_01_601 = x0s_01_597 * sg;
                float lin_10_602 = x0s_10_598 * sg;
                float lin_11_603 = x0s_11_599 * sg;
                float _exp2_72 = approx_exp2(-(x1_00_586 * fused));
                float _rcp_72 = approx_rcp(1.0f + _exp2_72);
                float sig_00_604 = _rcp_72;
                float _exp2_73 = approx_exp2(-(x1_01_587 * fused));
                float _rcp_73 = approx_rcp(1.0f + _exp2_73);
                float sig_01_605 = _rcp_73;
                float _exp2_74 = approx_exp2(-(x1_10_590 * fused));
                float _rcp_74 = approx_rcp(1.0f + _exp2_74);
                float sig_10_606 = _rcp_74;
                float _exp2_75 = approx_exp2(-(x1_11_591 * fused));
                float _rcp_75 = approx_rcp(1.0f + _exp2_75);
                float sig_11_607 = _rcp_75;
                float act_00_608 = x1_00_586 * sig_00_604;
                float act_01_609 = x1_01_587 * sig_01_605;
                float act_10_610 = x1_10_590 * sig_10_606;
                float act_11_611 = x1_11_591 * sig_11_607;
                float _min_148 = fminf(act_00_608, cl);
                act_00_608 = _min_148;
                float _min_149 = fminf(act_01_609, cl);
                act_01_609 = _min_149;
                float _min_150 = fminf(act_10_610, cl);
                act_10_610 = _min_150;
                float _min_151 = fminf(act_11_611, cl);
                act_11_611 = _min_151;
                quad[0] = lin_00_600 * act_00_608;
                quad[1] = lin_10_602 * act_10_610;
                quad[2] = lin_01_601 * act_01_609;
                quad[3] = lin_11_603 * act_11_611;
                uint32_t _fp8_18[1];
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
                    _fp8_18[0] = _packed;
                }
                int off0_612 = local_token0_582 * 64 + base_row_513;
                int off1_613 = local_token1_583 * 64 + base_row_513;
                int swz0_614 = off0_612 ^ (off0_612 >> 7 & 3) << 4;
                int swz1_615 = off1_613 ^ (off1_613 >> 7 & 3) << 4;
                epi_staging[swz0_614 >> 1] = _fp8_18[0] & 65535;
                epi_staging[swz1_615 >> 1] = _fp8_18[0] >> 16;
                int local_token0_616 = lane_1 % 4 * 2 + 24;
                int local_token1_617 = local_token0_616 + 1;
                float x0_00_618 = _tmem_load_4[12];
                float x0_01_619 = _tmem_load_4[13];
                float x1_00_620 = _tmem_load_4[14];
                float x1_01_621 = _tmem_load_4[15];
                float x0_10_622 = _tmem_load_5[12];
                float x0_11_623 = _tmem_load_5[13];
                float x1_10_624 = _tmem_load_5[14];
                float x1_11_625 = _tmem_load_5[15];
                float _max_76 = max_noftz(x0_00_618, neg_cl);
                float _min_152 = fminf(_max_76, cl);
                float x0c_00_626 = _min_152;
                float _max_77 = max_noftz(x0_01_619, neg_cl);
                float _min_153 = fminf(_max_77, cl);
                float x0c_01_627 = _min_153;
                float _max_78 = max_noftz(x0_10_622, neg_cl);
                float _min_154 = fminf(_max_78, cl);
                float x0c_10_628 = _min_154;
                float _max_79 = max_noftz(x0_11_623, neg_cl);
                float _min_155 = fminf(_max_79, cl);
                float x0c_11_629 = _min_155;
                float x0s_00_630 = x0c_00_626 * sc;
                float x0s_01_631 = x0c_01_627 * sc;
                float x0s_10_632 = x0c_10_628 * sc;
                float x0s_11_633 = x0c_11_629 * sc;
                float lin_00_634 = x0s_00_630 * sg;
                float lin_01_635 = x0s_01_631 * sg;
                float lin_10_636 = x0s_10_632 * sg;
                float lin_11_637 = x0s_11_633 * sg;
                float _exp2_76 = approx_exp2(-(x1_00_620 * fused));
                float _rcp_76 = approx_rcp(1.0f + _exp2_76);
                float sig_00_638 = _rcp_76;
                float _exp2_77 = approx_exp2(-(x1_01_621 * fused));
                float _rcp_77 = approx_rcp(1.0f + _exp2_77);
                float sig_01_639 = _rcp_77;
                float _exp2_78 = approx_exp2(-(x1_10_624 * fused));
                float _rcp_78 = approx_rcp(1.0f + _exp2_78);
                float sig_10_640 = _rcp_78;
                float _exp2_79 = approx_exp2(-(x1_11_625 * fused));
                float _rcp_79 = approx_rcp(1.0f + _exp2_79);
                float sig_11_641 = _rcp_79;
                float act_00_642 = x1_00_620 * sig_00_638;
                float act_01_643 = x1_01_621 * sig_01_639;
                float act_10_644 = x1_10_624 * sig_10_640;
                float act_11_645 = x1_11_625 * sig_11_641;
                float _min_156 = fminf(act_00_642, cl);
                act_00_642 = _min_156;
                float _min_157 = fminf(act_01_643, cl);
                act_01_643 = _min_157;
                float _min_158 = fminf(act_10_644, cl);
                act_10_644 = _min_158;
                float _min_159 = fminf(act_11_645, cl);
                act_11_645 = _min_159;
                quad[0] = lin_00_634 * act_00_642;
                quad[1] = lin_10_636 * act_10_644;
                quad[2] = lin_01_635 * act_01_643;
                quad[3] = lin_11_637 * act_11_645;
                uint32_t _fp8_19[1];
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
                    _fp8_19[0] = _packed;
                }
                int off0_646 = local_token0_616 * 64 + base_row_513;
                int off1_647 = local_token1_617 * 64 + base_row_513;
                int swz0_648 = off0_646 ^ (off0_646 >> 7 & 3) << 4;
                int swz1_649 = off1_647 ^ (off1_647 >> 7 & 3) << 4;
                epi_staging[swz0_648 >> 1] = _fp8_19[0] & 65535;
                epi_staging[swz1_649 >> 1] = _fp8_19[0] >> 16;
                int local_token0_650 = lane_1 % 4 * 2 + 32;
                int local_token1_651 = local_token0_650 + 1;
                float x0_00_652 = _tmem_load_4[16];
                float x0_01_653 = _tmem_load_4[17];
                float x1_00_654 = _tmem_load_4[18];
                float x1_01_655 = _tmem_load_4[19];
                float x0_10_656 = _tmem_load_5[16];
                float x0_11_657 = _tmem_load_5[17];
                float x1_10_658 = _tmem_load_5[18];
                float x1_11_659 = _tmem_load_5[19];
                float _max_80 = max_noftz(x0_00_652, neg_cl);
                float _min_160 = fminf(_max_80, cl);
                float x0c_00_660 = _min_160;
                float _max_81 = max_noftz(x0_01_653, neg_cl);
                float _min_161 = fminf(_max_81, cl);
                float x0c_01_661 = _min_161;
                float _max_82 = max_noftz(x0_10_656, neg_cl);
                float _min_162 = fminf(_max_82, cl);
                float x0c_10_662 = _min_162;
                float _max_83 = max_noftz(x0_11_657, neg_cl);
                float _min_163 = fminf(_max_83, cl);
                float x0c_11_663 = _min_163;
                float x0s_00_664 = x0c_00_660 * sc;
                float x0s_01_665 = x0c_01_661 * sc;
                float x0s_10_666 = x0c_10_662 * sc;
                float x0s_11_667 = x0c_11_663 * sc;
                float lin_00_668 = x0s_00_664 * sg;
                float lin_01_669 = x0s_01_665 * sg;
                float lin_10_670 = x0s_10_666 * sg;
                float lin_11_671 = x0s_11_667 * sg;
                float _exp2_80 = approx_exp2(-(x1_00_654 * fused));
                float _rcp_80 = approx_rcp(1.0f + _exp2_80);
                float sig_00_672 = _rcp_80;
                float _exp2_81 = approx_exp2(-(x1_01_655 * fused));
                float _rcp_81 = approx_rcp(1.0f + _exp2_81);
                float sig_01_673 = _rcp_81;
                float _exp2_82 = approx_exp2(-(x1_10_658 * fused));
                float _rcp_82 = approx_rcp(1.0f + _exp2_82);
                float sig_10_674 = _rcp_82;
                float _exp2_83 = approx_exp2(-(x1_11_659 * fused));
                float _rcp_83 = approx_rcp(1.0f + _exp2_83);
                float sig_11_675 = _rcp_83;
                float act_00_676 = x1_00_654 * sig_00_672;
                float act_01_677 = x1_01_655 * sig_01_673;
                float act_10_678 = x1_10_658 * sig_10_674;
                float act_11_679 = x1_11_659 * sig_11_675;
                float _min_164 = fminf(act_00_676, cl);
                act_00_676 = _min_164;
                float _min_165 = fminf(act_01_677, cl);
                act_01_677 = _min_165;
                float _min_166 = fminf(act_10_678, cl);
                act_10_678 = _min_166;
                float _min_167 = fminf(act_11_679, cl);
                act_11_679 = _min_167;
                quad[0] = lin_00_668 * act_00_676;
                quad[1] = lin_10_670 * act_10_678;
                quad[2] = lin_01_669 * act_01_677;
                quad[3] = lin_11_671 * act_11_679;
                uint32_t _fp8_20[1];
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
                    _fp8_20[0] = _packed;
                }
                int off0_680 = local_token0_650 * 64 + base_row_513;
                int off1_681 = local_token1_651 * 64 + base_row_513;
                int swz0_682 = off0_680 ^ (off0_680 >> 7 & 3) << 4;
                int swz1_683 = off1_681 ^ (off1_681 >> 7 & 3) << 4;
                epi_staging[swz0_682 >> 1] = _fp8_20[0] & 65535;
                epi_staging[swz1_683 >> 1] = _fp8_20[0] >> 16;
                int local_token0_684 = lane_1 % 4 * 2 + 40;
                int local_token1_685 = local_token0_684 + 1;
                float x0_00_686 = _tmem_load_4[20];
                float x0_01_687 = _tmem_load_4[21];
                float x1_00_688 = _tmem_load_4[22];
                float x1_01_689 = _tmem_load_4[23];
                float x0_10_690 = _tmem_load_5[20];
                float x0_11_691 = _tmem_load_5[21];
                float x1_10_692 = _tmem_load_5[22];
                float x1_11_693 = _tmem_load_5[23];
                float _max_84 = max_noftz(x0_00_686, neg_cl);
                float _min_168 = fminf(_max_84, cl);
                float x0c_00_694 = _min_168;
                float _max_85 = max_noftz(x0_01_687, neg_cl);
                float _min_169 = fminf(_max_85, cl);
                float x0c_01_695 = _min_169;
                float _max_86 = max_noftz(x0_10_690, neg_cl);
                float _min_170 = fminf(_max_86, cl);
                float x0c_10_696 = _min_170;
                float _max_87 = max_noftz(x0_11_691, neg_cl);
                float _min_171 = fminf(_max_87, cl);
                float x0c_11_697 = _min_171;
                float x0s_00_698 = x0c_00_694 * sc;
                float x0s_01_699 = x0c_01_695 * sc;
                float x0s_10_700 = x0c_10_696 * sc;
                float x0s_11_701 = x0c_11_697 * sc;
                float lin_00_702 = x0s_00_698 * sg;
                float lin_01_703 = x0s_01_699 * sg;
                float lin_10_704 = x0s_10_700 * sg;
                float lin_11_705 = x0s_11_701 * sg;
                float _exp2_84 = approx_exp2(-(x1_00_688 * fused));
                float _rcp_84 = approx_rcp(1.0f + _exp2_84);
                float sig_00_706 = _rcp_84;
                float _exp2_85 = approx_exp2(-(x1_01_689 * fused));
                float _rcp_85 = approx_rcp(1.0f + _exp2_85);
                float sig_01_707 = _rcp_85;
                float _exp2_86 = approx_exp2(-(x1_10_692 * fused));
                float _rcp_86 = approx_rcp(1.0f + _exp2_86);
                float sig_10_708 = _rcp_86;
                float _exp2_87 = approx_exp2(-(x1_11_693 * fused));
                float _rcp_87 = approx_rcp(1.0f + _exp2_87);
                float sig_11_709 = _rcp_87;
                float act_00_710 = x1_00_688 * sig_00_706;
                float act_01_711 = x1_01_689 * sig_01_707;
                float act_10_712 = x1_10_692 * sig_10_708;
                float act_11_713 = x1_11_693 * sig_11_709;
                float _min_172 = fminf(act_00_710, cl);
                act_00_710 = _min_172;
                float _min_173 = fminf(act_01_711, cl);
                act_01_711 = _min_173;
                float _min_174 = fminf(act_10_712, cl);
                act_10_712 = _min_174;
                float _min_175 = fminf(act_11_713, cl);
                act_11_713 = _min_175;
                quad[0] = lin_00_702 * act_00_710;
                quad[1] = lin_10_704 * act_10_712;
                quad[2] = lin_01_703 * act_01_711;
                quad[3] = lin_11_705 * act_11_713;
                uint32_t _fp8_21[1];
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
                    _fp8_21[0] = _packed;
                }
                int off0_714 = local_token0_684 * 64 + base_row_513;
                int off1_715 = local_token1_685 * 64 + base_row_513;
                int swz0_716 = off0_714 ^ (off0_714 >> 7 & 3) << 4;
                int swz1_717 = off1_715 ^ (off1_715 >> 7 & 3) << 4;
                epi_staging[swz0_716 >> 1] = _fp8_21[0] & 65535;
                epi_staging[swz1_717 >> 1] = _fp8_21[0] >> 16;
                int local_token0_718 = lane_1 % 4 * 2 + 48;
                int local_token1_719 = local_token0_718 + 1;
                float x0_00_720 = _tmem_load_4[24];
                float x0_01_721 = _tmem_load_4[25];
                float x1_00_722 = _tmem_load_4[26];
                float x1_01_723 = _tmem_load_4[27];
                float x0_10_724 = _tmem_load_5[24];
                float x0_11_725 = _tmem_load_5[25];
                float x1_10_726 = _tmem_load_5[26];
                float x1_11_727 = _tmem_load_5[27];
                float _max_88 = max_noftz(x0_00_720, neg_cl);
                float _min_176 = fminf(_max_88, cl);
                float x0c_00_728 = _min_176;
                float _max_89 = max_noftz(x0_01_721, neg_cl);
                float _min_177 = fminf(_max_89, cl);
                float x0c_01_729 = _min_177;
                float _max_90 = max_noftz(x0_10_724, neg_cl);
                float _min_178 = fminf(_max_90, cl);
                float x0c_10_730 = _min_178;
                float _max_91 = max_noftz(x0_11_725, neg_cl);
                float _min_179 = fminf(_max_91, cl);
                float x0c_11_731 = _min_179;
                float x0s_00_732 = x0c_00_728 * sc;
                float x0s_01_733 = x0c_01_729 * sc;
                float x0s_10_734 = x0c_10_730 * sc;
                float x0s_11_735 = x0c_11_731 * sc;
                float lin_00_736 = x0s_00_732 * sg;
                float lin_01_737 = x0s_01_733 * sg;
                float lin_10_738 = x0s_10_734 * sg;
                float lin_11_739 = x0s_11_735 * sg;
                float _exp2_88 = approx_exp2(-(x1_00_722 * fused));
                float _rcp_88 = approx_rcp(1.0f + _exp2_88);
                float sig_00_740 = _rcp_88;
                float _exp2_89 = approx_exp2(-(x1_01_723 * fused));
                float _rcp_89 = approx_rcp(1.0f + _exp2_89);
                float sig_01_741 = _rcp_89;
                float _exp2_90 = approx_exp2(-(x1_10_726 * fused));
                float _rcp_90 = approx_rcp(1.0f + _exp2_90);
                float sig_10_742 = _rcp_90;
                float _exp2_91 = approx_exp2(-(x1_11_727 * fused));
                float _rcp_91 = approx_rcp(1.0f + _exp2_91);
                float sig_11_743 = _rcp_91;
                float act_00_744 = x1_00_722 * sig_00_740;
                float act_01_745 = x1_01_723 * sig_01_741;
                float act_10_746 = x1_10_726 * sig_10_742;
                float act_11_747 = x1_11_727 * sig_11_743;
                float _min_180 = fminf(act_00_744, cl);
                act_00_744 = _min_180;
                float _min_181 = fminf(act_01_745, cl);
                act_01_745 = _min_181;
                float _min_182 = fminf(act_10_746, cl);
                act_10_746 = _min_182;
                float _min_183 = fminf(act_11_747, cl);
                act_11_747 = _min_183;
                quad[0] = lin_00_736 * act_00_744;
                quad[1] = lin_10_738 * act_10_746;
                quad[2] = lin_01_737 * act_01_745;
                quad[3] = lin_11_739 * act_11_747;
                uint32_t _fp8_22[1];
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
                    _fp8_22[0] = _packed;
                }
                int off0_748 = local_token0_718 * 64 + base_row_513;
                int off1_749 = local_token1_719 * 64 + base_row_513;
                int swz0_750 = off0_748 ^ (off0_748 >> 7 & 3) << 4;
                int swz1_751 = off1_749 ^ (off1_749 >> 7 & 3) << 4;
                epi_staging[swz0_750 >> 1] = _fp8_22[0] & 65535;
                epi_staging[swz1_751 >> 1] = _fp8_22[0] >> 16;
                int local_token0_752 = lane_1 % 4 * 2 + 56;
                int local_token1_753 = local_token0_752 + 1;
                float x0_00_754 = _tmem_load_4[28];
                float x0_01_755 = _tmem_load_4[29];
                float x1_00_756 = _tmem_load_4[30];
                float x1_01_757 = _tmem_load_4[31];
                float x0_10_758 = _tmem_load_5[28];
                float x0_11_759 = _tmem_load_5[29];
                float x1_10_760 = _tmem_load_5[30];
                float x1_11_761 = _tmem_load_5[31];
                float _max_92 = max_noftz(x0_00_754, neg_cl);
                float _min_184 = fminf(_max_92, cl);
                float x0c_00_762 = _min_184;
                float _max_93 = max_noftz(x0_01_755, neg_cl);
                float _min_185 = fminf(_max_93, cl);
                float x0c_01_763 = _min_185;
                float _max_94 = max_noftz(x0_10_758, neg_cl);
                float _min_186 = fminf(_max_94, cl);
                float x0c_10_764 = _min_186;
                float _max_95 = max_noftz(x0_11_759, neg_cl);
                float _min_187 = fminf(_max_95, cl);
                float x0c_11_765 = _min_187;
                float x0s_00_766 = x0c_00_762 * sc;
                float x0s_01_767 = x0c_01_763 * sc;
                float x0s_10_768 = x0c_10_764 * sc;
                float x0s_11_769 = x0c_11_765 * sc;
                float lin_00_770 = x0s_00_766 * sg;
                float lin_01_771 = x0s_01_767 * sg;
                float lin_10_772 = x0s_10_768 * sg;
                float lin_11_773 = x0s_11_769 * sg;
                float _exp2_92 = approx_exp2(-(x1_00_756 * fused));
                float _rcp_92 = approx_rcp(1.0f + _exp2_92);
                float sig_00_774 = _rcp_92;
                float _exp2_93 = approx_exp2(-(x1_01_757 * fused));
                float _rcp_93 = approx_rcp(1.0f + _exp2_93);
                float sig_01_775 = _rcp_93;
                float _exp2_94 = approx_exp2(-(x1_10_760 * fused));
                float _rcp_94 = approx_rcp(1.0f + _exp2_94);
                float sig_10_776 = _rcp_94;
                float _exp2_95 = approx_exp2(-(x1_11_761 * fused));
                float _rcp_95 = approx_rcp(1.0f + _exp2_95);
                float sig_11_777 = _rcp_95;
                float act_00_778 = x1_00_756 * sig_00_774;
                float act_01_779 = x1_01_757 * sig_01_775;
                float act_10_780 = x1_10_760 * sig_10_776;
                float act_11_781 = x1_11_761 * sig_11_777;
                float _min_188 = fminf(act_00_778, cl);
                act_00_778 = _min_188;
                float _min_189 = fminf(act_01_779, cl);
                act_01_779 = _min_189;
                float _min_190 = fminf(act_10_780, cl);
                act_10_780 = _min_190;
                float _min_191 = fminf(act_11_781, cl);
                act_11_781 = _min_191;
                quad[0] = lin_00_770 * act_00_778;
                quad[1] = lin_10_772 * act_10_780;
                quad[2] = lin_01_771 * act_01_779;
                quad[3] = lin_11_773 * act_11_781;
                uint32_t _fp8_23[1];
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
                    _fp8_23[0] = _packed;
                }
                int off0_782 = local_token0_752 * 64 + base_row_513;
                int off1_783 = local_token1_753 * 64 + base_row_513;
                int swz0_784 = off0_782 ^ (off0_782 >> 7 & 3) << 4;
                int swz1_785 = off1_783 ^ (off1_783 >> 7 & 3) << 4;
                epi_staging[swz0_784 >> 1] = _fp8_23[0] & 65535;
                epi_staging[swz1_785 >> 1] = _fp8_23[0] >> 16;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                if (warp == 0) {
                    if (elect_sync()) {
                        int padding_rows_2 = (256 - valid_rows % 256) % 256;
                        tma_store_4d((&C), m_tile * 64, padding_rows_2 + 128, 1073741824, n_tile * 256 - (unsigned int)padding_rows_2 + 1073741824, epi_staging_addr);
                    }
                }
                asm volatile("cp.async.bulk.commit_group;");
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                int acc_offset_786 = acc_stage * 256 + 192;
                float _tmem_load_6[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[31]))
                    : "r"(taddr + (unsigned int)acc_offset_786));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float _tmem_load_7[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[31]))
                    : "r"(taddr + 1048576 + (unsigned int)acc_offset_786));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int base_row_787 = warp_0 * 16 + lane_1 / 4 * 2;
                asm volatile("cp.async.bulk.wait_group.read 0;");
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                int local_token0_788 = lane_1 % 4 * 2;
                int local_token1_789 = local_token0_788 + 1;
                float x0_00_790 = _tmem_load_6[0];
                float x0_01_791 = _tmem_load_6[1];
                float x1_00_792 = _tmem_load_6[2];
                float x1_01_793 = _tmem_load_6[3];
                float x0_10_794 = _tmem_load_7[0];
                float x0_11_795 = _tmem_load_7[1];
                float x1_10_796 = _tmem_load_7[2];
                float x1_11_797 = _tmem_load_7[3];
                float _max_96 = max_noftz(x0_00_790, neg_cl);
                float _min_192 = fminf(_max_96, cl);
                float x0c_00_798 = _min_192;
                float _max_97 = max_noftz(x0_01_791, neg_cl);
                float _min_193 = fminf(_max_97, cl);
                float x0c_01_799 = _min_193;
                float _max_98 = max_noftz(x0_10_794, neg_cl);
                float _min_194 = fminf(_max_98, cl);
                float x0c_10_800 = _min_194;
                float _max_99 = max_noftz(x0_11_795, neg_cl);
                float _min_195 = fminf(_max_99, cl);
                float x0c_11_801 = _min_195;
                float x0s_00_802 = x0c_00_798 * sc;
                float x0s_01_803 = x0c_01_799 * sc;
                float x0s_10_804 = x0c_10_800 * sc;
                float x0s_11_805 = x0c_11_801 * sc;
                float lin_00_806 = x0s_00_802 * sg;
                float lin_01_807 = x0s_01_803 * sg;
                float lin_10_808 = x0s_10_804 * sg;
                float lin_11_809 = x0s_11_805 * sg;
                float _exp2_96 = approx_exp2(-(x1_00_792 * fused));
                float _rcp_96 = approx_rcp(1.0f + _exp2_96);
                float sig_00_810 = _rcp_96;
                float _exp2_97 = approx_exp2(-(x1_01_793 * fused));
                float _rcp_97 = approx_rcp(1.0f + _exp2_97);
                float sig_01_811 = _rcp_97;
                float _exp2_98 = approx_exp2(-(x1_10_796 * fused));
                float _rcp_98 = approx_rcp(1.0f + _exp2_98);
                float sig_10_812 = _rcp_98;
                float _exp2_99 = approx_exp2(-(x1_11_797 * fused));
                float _rcp_99 = approx_rcp(1.0f + _exp2_99);
                float sig_11_813 = _rcp_99;
                float act_00_814 = x1_00_792 * sig_00_810;
                float act_01_815 = x1_01_793 * sig_01_811;
                float act_10_816 = x1_10_796 * sig_10_812;
                float act_11_817 = x1_11_797 * sig_11_813;
                float _min_196 = fminf(act_00_814, cl);
                act_00_814 = _min_196;
                float _min_197 = fminf(act_01_815, cl);
                act_01_815 = _min_197;
                float _min_198 = fminf(act_10_816, cl);
                act_10_816 = _min_198;
                float _min_199 = fminf(act_11_817, cl);
                act_11_817 = _min_199;
                quad[0] = lin_00_806 * act_00_814;
                quad[1] = lin_10_808 * act_10_816;
                quad[2] = lin_01_807 * act_01_815;
                quad[3] = lin_11_809 * act_11_817;
                uint32_t _fp8_24[1];
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
                    _fp8_24[0] = _packed;
                }
                int off0_818 = local_token0_788 * 64 + base_row_787;
                int off1_819 = local_token1_789 * 64 + base_row_787;
                int swz0_820 = off0_818 ^ (off0_818 >> 7 & 3) << 4;
                int swz1_821 = off1_819 ^ (off1_819 >> 7 & 3) << 4;
                epi_staging[swz0_820 >> 1] = _fp8_24[0] & 65535;
                epi_staging[swz1_821 >> 1] = _fp8_24[0] >> 16;
                int local_token0_822 = lane_1 % 4 * 2 + 8;
                int local_token1_823 = local_token0_822 + 1;
                float x0_00_824 = _tmem_load_6[4];
                float x0_01_825 = _tmem_load_6[5];
                float x1_00_826 = _tmem_load_6[6];
                float x1_01_827 = _tmem_load_6[7];
                float x0_10_828 = _tmem_load_7[4];
                float x0_11_829 = _tmem_load_7[5];
                float x1_10_830 = _tmem_load_7[6];
                float x1_11_831 = _tmem_load_7[7];
                float _max_100 = max_noftz(x0_00_824, neg_cl);
                float _min_200 = fminf(_max_100, cl);
                float x0c_00_832 = _min_200;
                float _max_101 = max_noftz(x0_01_825, neg_cl);
                float _min_201 = fminf(_max_101, cl);
                float x0c_01_833 = _min_201;
                float _max_102 = max_noftz(x0_10_828, neg_cl);
                float _min_202 = fminf(_max_102, cl);
                float x0c_10_834 = _min_202;
                float _max_103 = max_noftz(x0_11_829, neg_cl);
                float _min_203 = fminf(_max_103, cl);
                float x0c_11_835 = _min_203;
                float x0s_00_836 = x0c_00_832 * sc;
                float x0s_01_837 = x0c_01_833 * sc;
                float x0s_10_838 = x0c_10_834 * sc;
                float x0s_11_839 = x0c_11_835 * sc;
                float lin_00_840 = x0s_00_836 * sg;
                float lin_01_841 = x0s_01_837 * sg;
                float lin_10_842 = x0s_10_838 * sg;
                float lin_11_843 = x0s_11_839 * sg;
                float _exp2_100 = approx_exp2(-(x1_00_826 * fused));
                float _rcp_100 = approx_rcp(1.0f + _exp2_100);
                float sig_00_844 = _rcp_100;
                float _exp2_101 = approx_exp2(-(x1_01_827 * fused));
                float _rcp_101 = approx_rcp(1.0f + _exp2_101);
                float sig_01_845 = _rcp_101;
                float _exp2_102 = approx_exp2(-(x1_10_830 * fused));
                float _rcp_102 = approx_rcp(1.0f + _exp2_102);
                float sig_10_846 = _rcp_102;
                float _exp2_103 = approx_exp2(-(x1_11_831 * fused));
                float _rcp_103 = approx_rcp(1.0f + _exp2_103);
                float sig_11_847 = _rcp_103;
                float act_00_848 = x1_00_826 * sig_00_844;
                float act_01_849 = x1_01_827 * sig_01_845;
                float act_10_850 = x1_10_830 * sig_10_846;
                float act_11_851 = x1_11_831 * sig_11_847;
                float _min_204 = fminf(act_00_848, cl);
                act_00_848 = _min_204;
                float _min_205 = fminf(act_01_849, cl);
                act_01_849 = _min_205;
                float _min_206 = fminf(act_10_850, cl);
                act_10_850 = _min_206;
                float _min_207 = fminf(act_11_851, cl);
                act_11_851 = _min_207;
                quad[0] = lin_00_840 * act_00_848;
                quad[1] = lin_10_842 * act_10_850;
                quad[2] = lin_01_841 * act_01_849;
                quad[3] = lin_11_843 * act_11_851;
                uint32_t _fp8_25[1];
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
                    _fp8_25[0] = _packed;
                }
                int off0_852 = local_token0_822 * 64 + base_row_787;
                int off1_853 = local_token1_823 * 64 + base_row_787;
                int swz0_854 = off0_852 ^ (off0_852 >> 7 & 3) << 4;
                int swz1_855 = off1_853 ^ (off1_853 >> 7 & 3) << 4;
                epi_staging[swz0_854 >> 1] = _fp8_25[0] & 65535;
                epi_staging[swz1_855 >> 1] = _fp8_25[0] >> 16;
                int local_token0_856 = lane_1 % 4 * 2 + 16;
                int local_token1_857 = local_token0_856 + 1;
                float x0_00_858 = _tmem_load_6[8];
                float x0_01_859 = _tmem_load_6[9];
                float x1_00_860 = _tmem_load_6[10];
                float x1_01_861 = _tmem_load_6[11];
                float x0_10_862 = _tmem_load_7[8];
                float x0_11_863 = _tmem_load_7[9];
                float x1_10_864 = _tmem_load_7[10];
                float x1_11_865 = _tmem_load_7[11];
                float _max_104 = max_noftz(x0_00_858, neg_cl);
                float _min_208 = fminf(_max_104, cl);
                float x0c_00_866 = _min_208;
                float _max_105 = max_noftz(x0_01_859, neg_cl);
                float _min_209 = fminf(_max_105, cl);
                float x0c_01_867 = _min_209;
                float _max_106 = max_noftz(x0_10_862, neg_cl);
                float _min_210 = fminf(_max_106, cl);
                float x0c_10_868 = _min_210;
                float _max_107 = max_noftz(x0_11_863, neg_cl);
                float _min_211 = fminf(_max_107, cl);
                float x0c_11_869 = _min_211;
                float x0s_00_870 = x0c_00_866 * sc;
                float x0s_01_871 = x0c_01_867 * sc;
                float x0s_10_872 = x0c_10_868 * sc;
                float x0s_11_873 = x0c_11_869 * sc;
                float lin_00_874 = x0s_00_870 * sg;
                float lin_01_875 = x0s_01_871 * sg;
                float lin_10_876 = x0s_10_872 * sg;
                float lin_11_877 = x0s_11_873 * sg;
                float _exp2_104 = approx_exp2(-(x1_00_860 * fused));
                float _rcp_104 = approx_rcp(1.0f + _exp2_104);
                float sig_00_878 = _rcp_104;
                float _exp2_105 = approx_exp2(-(x1_01_861 * fused));
                float _rcp_105 = approx_rcp(1.0f + _exp2_105);
                float sig_01_879 = _rcp_105;
                float _exp2_106 = approx_exp2(-(x1_10_864 * fused));
                float _rcp_106 = approx_rcp(1.0f + _exp2_106);
                float sig_10_880 = _rcp_106;
                float _exp2_107 = approx_exp2(-(x1_11_865 * fused));
                float _rcp_107 = approx_rcp(1.0f + _exp2_107);
                float sig_11_881 = _rcp_107;
                float act_00_882 = x1_00_860 * sig_00_878;
                float act_01_883 = x1_01_861 * sig_01_879;
                float act_10_884 = x1_10_864 * sig_10_880;
                float act_11_885 = x1_11_865 * sig_11_881;
                float _min_212 = fminf(act_00_882, cl);
                act_00_882 = _min_212;
                float _min_213 = fminf(act_01_883, cl);
                act_01_883 = _min_213;
                float _min_214 = fminf(act_10_884, cl);
                act_10_884 = _min_214;
                float _min_215 = fminf(act_11_885, cl);
                act_11_885 = _min_215;
                quad[0] = lin_00_874 * act_00_882;
                quad[1] = lin_10_876 * act_10_884;
                quad[2] = lin_01_875 * act_01_883;
                quad[3] = lin_11_877 * act_11_885;
                uint32_t _fp8_26[1];
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
                    _fp8_26[0] = _packed;
                }
                int off0_886 = local_token0_856 * 64 + base_row_787;
                int off1_887 = local_token1_857 * 64 + base_row_787;
                int swz0_888 = off0_886 ^ (off0_886 >> 7 & 3) << 4;
                int swz1_889 = off1_887 ^ (off1_887 >> 7 & 3) << 4;
                epi_staging[swz0_888 >> 1] = _fp8_26[0] & 65535;
                epi_staging[swz1_889 >> 1] = _fp8_26[0] >> 16;
                int local_token0_890 = lane_1 % 4 * 2 + 24;
                int local_token1_891 = local_token0_890 + 1;
                float x0_00_892 = _tmem_load_6[12];
                float x0_01_893 = _tmem_load_6[13];
                float x1_00_894 = _tmem_load_6[14];
                float x1_01_895 = _tmem_load_6[15];
                float x0_10_896 = _tmem_load_7[12];
                float x0_11_897 = _tmem_load_7[13];
                float x1_10_898 = _tmem_load_7[14];
                float x1_11_899 = _tmem_load_7[15];
                float _max_108 = max_noftz(x0_00_892, neg_cl);
                float _min_216 = fminf(_max_108, cl);
                float x0c_00_900 = _min_216;
                float _max_109 = max_noftz(x0_01_893, neg_cl);
                float _min_217 = fminf(_max_109, cl);
                float x0c_01_901 = _min_217;
                float _max_110 = max_noftz(x0_10_896, neg_cl);
                float _min_218 = fminf(_max_110, cl);
                float x0c_10_902 = _min_218;
                float _max_111 = max_noftz(x0_11_897, neg_cl);
                float _min_219 = fminf(_max_111, cl);
                float x0c_11_903 = _min_219;
                float x0s_00_904 = x0c_00_900 * sc;
                float x0s_01_905 = x0c_01_901 * sc;
                float x0s_10_906 = x0c_10_902 * sc;
                float x0s_11_907 = x0c_11_903 * sc;
                float lin_00_908 = x0s_00_904 * sg;
                float lin_01_909 = x0s_01_905 * sg;
                float lin_10_910 = x0s_10_906 * sg;
                float lin_11_911 = x0s_11_907 * sg;
                float _exp2_108 = approx_exp2(-(x1_00_894 * fused));
                float _rcp_108 = approx_rcp(1.0f + _exp2_108);
                float sig_00_912 = _rcp_108;
                float _exp2_109 = approx_exp2(-(x1_01_895 * fused));
                float _rcp_109 = approx_rcp(1.0f + _exp2_109);
                float sig_01_913 = _rcp_109;
                float _exp2_110 = approx_exp2(-(x1_10_898 * fused));
                float _rcp_110 = approx_rcp(1.0f + _exp2_110);
                float sig_10_914 = _rcp_110;
                float _exp2_111 = approx_exp2(-(x1_11_899 * fused));
                float _rcp_111 = approx_rcp(1.0f + _exp2_111);
                float sig_11_915 = _rcp_111;
                float act_00_916 = x1_00_894 * sig_00_912;
                float act_01_917 = x1_01_895 * sig_01_913;
                float act_10_918 = x1_10_898 * sig_10_914;
                float act_11_919 = x1_11_899 * sig_11_915;
                float _min_220 = fminf(act_00_916, cl);
                act_00_916 = _min_220;
                float _min_221 = fminf(act_01_917, cl);
                act_01_917 = _min_221;
                float _min_222 = fminf(act_10_918, cl);
                act_10_918 = _min_222;
                float _min_223 = fminf(act_11_919, cl);
                act_11_919 = _min_223;
                quad[0] = lin_00_908 * act_00_916;
                quad[1] = lin_10_910 * act_10_918;
                quad[2] = lin_01_909 * act_01_917;
                quad[3] = lin_11_911 * act_11_919;
                uint32_t _fp8_27[1];
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
                    _fp8_27[0] = _packed;
                }
                int off0_920 = local_token0_890 * 64 + base_row_787;
                int off1_921 = local_token1_891 * 64 + base_row_787;
                int swz0_922 = off0_920 ^ (off0_920 >> 7 & 3) << 4;
                int swz1_923 = off1_921 ^ (off1_921 >> 7 & 3) << 4;
                epi_staging[swz0_922 >> 1] = _fp8_27[0] & 65535;
                epi_staging[swz1_923 >> 1] = _fp8_27[0] >> 16;
                int local_token0_924 = lane_1 % 4 * 2 + 32;
                int local_token1_925 = local_token0_924 + 1;
                float x0_00_926 = _tmem_load_6[16];
                float x0_01_927 = _tmem_load_6[17];
                float x1_00_928 = _tmem_load_6[18];
                float x1_01_929 = _tmem_load_6[19];
                float x0_10_930 = _tmem_load_7[16];
                float x0_11_931 = _tmem_load_7[17];
                float x1_10_932 = _tmem_load_7[18];
                float x1_11_933 = _tmem_load_7[19];
                float _max_112 = max_noftz(x0_00_926, neg_cl);
                float _min_224 = fminf(_max_112, cl);
                float x0c_00_934 = _min_224;
                float _max_113 = max_noftz(x0_01_927, neg_cl);
                float _min_225 = fminf(_max_113, cl);
                float x0c_01_935 = _min_225;
                float _max_114 = max_noftz(x0_10_930, neg_cl);
                float _min_226 = fminf(_max_114, cl);
                float x0c_10_936 = _min_226;
                float _max_115 = max_noftz(x0_11_931, neg_cl);
                float _min_227 = fminf(_max_115, cl);
                float x0c_11_937 = _min_227;
                float x0s_00_938 = x0c_00_934 * sc;
                float x0s_01_939 = x0c_01_935 * sc;
                float x0s_10_940 = x0c_10_936 * sc;
                float x0s_11_941 = x0c_11_937 * sc;
                float lin_00_942 = x0s_00_938 * sg;
                float lin_01_943 = x0s_01_939 * sg;
                float lin_10_944 = x0s_10_940 * sg;
                float lin_11_945 = x0s_11_941 * sg;
                float _exp2_112 = approx_exp2(-(x1_00_928 * fused));
                float _rcp_112 = approx_rcp(1.0f + _exp2_112);
                float sig_00_946 = _rcp_112;
                float _exp2_113 = approx_exp2(-(x1_01_929 * fused));
                float _rcp_113 = approx_rcp(1.0f + _exp2_113);
                float sig_01_947 = _rcp_113;
                float _exp2_114 = approx_exp2(-(x1_10_932 * fused));
                float _rcp_114 = approx_rcp(1.0f + _exp2_114);
                float sig_10_948 = _rcp_114;
                float _exp2_115 = approx_exp2(-(x1_11_933 * fused));
                float _rcp_115 = approx_rcp(1.0f + _exp2_115);
                float sig_11_949 = _rcp_115;
                float act_00_950 = x1_00_928 * sig_00_946;
                float act_01_951 = x1_01_929 * sig_01_947;
                float act_10_952 = x1_10_932 * sig_10_948;
                float act_11_953 = x1_11_933 * sig_11_949;
                float _min_228 = fminf(act_00_950, cl);
                act_00_950 = _min_228;
                float _min_229 = fminf(act_01_951, cl);
                act_01_951 = _min_229;
                float _min_230 = fminf(act_10_952, cl);
                act_10_952 = _min_230;
                float _min_231 = fminf(act_11_953, cl);
                act_11_953 = _min_231;
                quad[0] = lin_00_942 * act_00_950;
                quad[1] = lin_10_944 * act_10_952;
                quad[2] = lin_01_943 * act_01_951;
                quad[3] = lin_11_945 * act_11_953;
                uint32_t _fp8_28[1];
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
                    _fp8_28[0] = _packed;
                }
                int off0_954 = local_token0_924 * 64 + base_row_787;
                int off1_955 = local_token1_925 * 64 + base_row_787;
                int swz0_956 = off0_954 ^ (off0_954 >> 7 & 3) << 4;
                int swz1_957 = off1_955 ^ (off1_955 >> 7 & 3) << 4;
                epi_staging[swz0_956 >> 1] = _fp8_28[0] & 65535;
                epi_staging[swz1_957 >> 1] = _fp8_28[0] >> 16;
                int local_token0_958 = lane_1 % 4 * 2 + 40;
                int local_token1_959 = local_token0_958 + 1;
                float x0_00_960 = _tmem_load_6[20];
                float x0_01_961 = _tmem_load_6[21];
                float x1_00_962 = _tmem_load_6[22];
                float x1_01_963 = _tmem_load_6[23];
                float x0_10_964 = _tmem_load_7[20];
                float x0_11_965 = _tmem_load_7[21];
                float x1_10_966 = _tmem_load_7[22];
                float x1_11_967 = _tmem_load_7[23];
                float _max_116 = max_noftz(x0_00_960, neg_cl);
                float _min_232 = fminf(_max_116, cl);
                float x0c_00_968 = _min_232;
                float _max_117 = max_noftz(x0_01_961, neg_cl);
                float _min_233 = fminf(_max_117, cl);
                float x0c_01_969 = _min_233;
                float _max_118 = max_noftz(x0_10_964, neg_cl);
                float _min_234 = fminf(_max_118, cl);
                float x0c_10_970 = _min_234;
                float _max_119 = max_noftz(x0_11_965, neg_cl);
                float _min_235 = fminf(_max_119, cl);
                float x0c_11_971 = _min_235;
                float x0s_00_972 = x0c_00_968 * sc;
                float x0s_01_973 = x0c_01_969 * sc;
                float x0s_10_974 = x0c_10_970 * sc;
                float x0s_11_975 = x0c_11_971 * sc;
                float lin_00_976 = x0s_00_972 * sg;
                float lin_01_977 = x0s_01_973 * sg;
                float lin_10_978 = x0s_10_974 * sg;
                float lin_11_979 = x0s_11_975 * sg;
                float _exp2_116 = approx_exp2(-(x1_00_962 * fused));
                float _rcp_116 = approx_rcp(1.0f + _exp2_116);
                float sig_00_980 = _rcp_116;
                float _exp2_117 = approx_exp2(-(x1_01_963 * fused));
                float _rcp_117 = approx_rcp(1.0f + _exp2_117);
                float sig_01_981 = _rcp_117;
                float _exp2_118 = approx_exp2(-(x1_10_966 * fused));
                float _rcp_118 = approx_rcp(1.0f + _exp2_118);
                float sig_10_982 = _rcp_118;
                float _exp2_119 = approx_exp2(-(x1_11_967 * fused));
                float _rcp_119 = approx_rcp(1.0f + _exp2_119);
                float sig_11_983 = _rcp_119;
                float act_00_984 = x1_00_962 * sig_00_980;
                float act_01_985 = x1_01_963 * sig_01_981;
                float act_10_986 = x1_10_966 * sig_10_982;
                float act_11_987 = x1_11_967 * sig_11_983;
                float _min_236 = fminf(act_00_984, cl);
                act_00_984 = _min_236;
                float _min_237 = fminf(act_01_985, cl);
                act_01_985 = _min_237;
                float _min_238 = fminf(act_10_986, cl);
                act_10_986 = _min_238;
                float _min_239 = fminf(act_11_987, cl);
                act_11_987 = _min_239;
                quad[0] = lin_00_976 * act_00_984;
                quad[1] = lin_10_978 * act_10_986;
                quad[2] = lin_01_977 * act_01_985;
                quad[3] = lin_11_979 * act_11_987;
                uint32_t _fp8_29[1];
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
                    _fp8_29[0] = _packed;
                }
                int off0_988 = local_token0_958 * 64 + base_row_787;
                int off1_989 = local_token1_959 * 64 + base_row_787;
                int swz0_990 = off0_988 ^ (off0_988 >> 7 & 3) << 4;
                int swz1_991 = off1_989 ^ (off1_989 >> 7 & 3) << 4;
                epi_staging[swz0_990 >> 1] = _fp8_29[0] & 65535;
                epi_staging[swz1_991 >> 1] = _fp8_29[0] >> 16;
                int local_token0_992 = lane_1 % 4 * 2 + 48;
                int local_token1_993 = local_token0_992 + 1;
                float x0_00_994 = _tmem_load_6[24];
                float x0_01_995 = _tmem_load_6[25];
                float x1_00_996 = _tmem_load_6[26];
                float x1_01_997 = _tmem_load_6[27];
                float x0_10_998 = _tmem_load_7[24];
                float x0_11_999 = _tmem_load_7[25];
                float x1_10_1000 = _tmem_load_7[26];
                float x1_11_1001 = _tmem_load_7[27];
                float _max_120 = max_noftz(x0_00_994, neg_cl);
                float _min_240 = fminf(_max_120, cl);
                float x0c_00_1002 = _min_240;
                float _max_121 = max_noftz(x0_01_995, neg_cl);
                float _min_241 = fminf(_max_121, cl);
                float x0c_01_1003 = _min_241;
                float _max_122 = max_noftz(x0_10_998, neg_cl);
                float _min_242 = fminf(_max_122, cl);
                float x0c_10_1004 = _min_242;
                float _max_123 = max_noftz(x0_11_999, neg_cl);
                float _min_243 = fminf(_max_123, cl);
                float x0c_11_1005 = _min_243;
                float x0s_00_1006 = x0c_00_1002 * sc;
                float x0s_01_1007 = x0c_01_1003 * sc;
                float x0s_10_1008 = x0c_10_1004 * sc;
                float x0s_11_1009 = x0c_11_1005 * sc;
                float lin_00_1010 = x0s_00_1006 * sg;
                float lin_01_1011 = x0s_01_1007 * sg;
                float lin_10_1012 = x0s_10_1008 * sg;
                float lin_11_1013 = x0s_11_1009 * sg;
                float _exp2_120 = approx_exp2(-(x1_00_996 * fused));
                float _rcp_120 = approx_rcp(1.0f + _exp2_120);
                float sig_00_1014 = _rcp_120;
                float _exp2_121 = approx_exp2(-(x1_01_997 * fused));
                float _rcp_121 = approx_rcp(1.0f + _exp2_121);
                float sig_01_1015 = _rcp_121;
                float _exp2_122 = approx_exp2(-(x1_10_1000 * fused));
                float _rcp_122 = approx_rcp(1.0f + _exp2_122);
                float sig_10_1016 = _rcp_122;
                float _exp2_123 = approx_exp2(-(x1_11_1001 * fused));
                float _rcp_123 = approx_rcp(1.0f + _exp2_123);
                float sig_11_1017 = _rcp_123;
                float act_00_1018 = x1_00_996 * sig_00_1014;
                float act_01_1019 = x1_01_997 * sig_01_1015;
                float act_10_1020 = x1_10_1000 * sig_10_1016;
                float act_11_1021 = x1_11_1001 * sig_11_1017;
                float _min_244 = fminf(act_00_1018, cl);
                act_00_1018 = _min_244;
                float _min_245 = fminf(act_01_1019, cl);
                act_01_1019 = _min_245;
                float _min_246 = fminf(act_10_1020, cl);
                act_10_1020 = _min_246;
                float _min_247 = fminf(act_11_1021, cl);
                act_11_1021 = _min_247;
                quad[0] = lin_00_1010 * act_00_1018;
                quad[1] = lin_10_1012 * act_10_1020;
                quad[2] = lin_01_1011 * act_01_1019;
                quad[3] = lin_11_1013 * act_11_1021;
                uint32_t _fp8_30[1];
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
                    _fp8_30[0] = _packed;
                }
                int off0_1022 = local_token0_992 * 64 + base_row_787;
                int off1_1023 = local_token1_993 * 64 + base_row_787;
                int swz0_1024 = off0_1022 ^ (off0_1022 >> 7 & 3) << 4;
                int swz1_1025 = off1_1023 ^ (off1_1023 >> 7 & 3) << 4;
                epi_staging[swz0_1024 >> 1] = _fp8_30[0] & 65535;
                epi_staging[swz1_1025 >> 1] = _fp8_30[0] >> 16;
                int local_token0_1026 = lane_1 % 4 * 2 + 56;
                int local_token1_1027 = local_token0_1026 + 1;
                float x0_00_1028 = _tmem_load_6[28];
                float x0_01_1029 = _tmem_load_6[29];
                float x1_00_1030 = _tmem_load_6[30];
                float x1_01_1031 = _tmem_load_6[31];
                float x0_10_1032 = _tmem_load_7[28];
                float x0_11_1033 = _tmem_load_7[29];
                float x1_10_1034 = _tmem_load_7[30];
                float x1_11_1035 = _tmem_load_7[31];
                float _max_124 = max_noftz(x0_00_1028, neg_cl);
                float _min_248 = fminf(_max_124, cl);
                float x0c_00_1036 = _min_248;
                float _max_125 = max_noftz(x0_01_1029, neg_cl);
                float _min_249 = fminf(_max_125, cl);
                float x0c_01_1037 = _min_249;
                float _max_126 = max_noftz(x0_10_1032, neg_cl);
                float _min_250 = fminf(_max_126, cl);
                float x0c_10_1038 = _min_250;
                float _max_127 = max_noftz(x0_11_1033, neg_cl);
                float _min_251 = fminf(_max_127, cl);
                float x0c_11_1039 = _min_251;
                float x0s_00_1040 = x0c_00_1036 * sc;
                float x0s_01_1041 = x0c_01_1037 * sc;
                float x0s_10_1042 = x0c_10_1038 * sc;
                float x0s_11_1043 = x0c_11_1039 * sc;
                float lin_00_1044 = x0s_00_1040 * sg;
                float lin_01_1045 = x0s_01_1041 * sg;
                float lin_10_1046 = x0s_10_1042 * sg;
                float lin_11_1047 = x0s_11_1043 * sg;
                float _exp2_124 = approx_exp2(-(x1_00_1030 * fused));
                float _rcp_124 = approx_rcp(1.0f + _exp2_124);
                float sig_00_1048 = _rcp_124;
                float _exp2_125 = approx_exp2(-(x1_01_1031 * fused));
                float _rcp_125 = approx_rcp(1.0f + _exp2_125);
                float sig_01_1049 = _rcp_125;
                float _exp2_126 = approx_exp2(-(x1_10_1034 * fused));
                float _rcp_126 = approx_rcp(1.0f + _exp2_126);
                float sig_10_1050 = _rcp_126;
                float _exp2_127 = approx_exp2(-(x1_11_1035 * fused));
                float _rcp_127 = approx_rcp(1.0f + _exp2_127);
                float sig_11_1051 = _rcp_127;
                float act_00_1052 = x1_00_1030 * sig_00_1048;
                float act_01_1053 = x1_01_1031 * sig_01_1049;
                float act_10_1054 = x1_10_1034 * sig_10_1050;
                float act_11_1055 = x1_11_1035 * sig_11_1051;
                float _min_252 = fminf(act_00_1052, cl);
                act_00_1052 = _min_252;
                float _min_253 = fminf(act_01_1053, cl);
                act_01_1053 = _min_253;
                float _min_254 = fminf(act_10_1054, cl);
                act_10_1054 = _min_254;
                float _min_255 = fminf(act_11_1055, cl);
                act_11_1055 = _min_255;
                quad[0] = lin_00_1044 * act_00_1052;
                quad[1] = lin_10_1046 * act_10_1054;
                quad[2] = lin_01_1045 * act_01_1053;
                quad[3] = lin_11_1047 * act_11_1055;
                uint32_t _fp8_31[1];
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
                    _fp8_31[0] = _packed;
                }
                int off0_1056 = local_token0_1026 * 64 + base_row_787;
                int off1_1057 = local_token1_1027 * 64 + base_row_787;
                int swz0_1058 = off0_1056 ^ (off0_1056 >> 7 & 3) << 4;
                int swz1_1059 = off1_1057 ^ (off1_1057 >> 7 & 3) << 4;
                epi_staging[swz0_1058 >> 1] = _fp8_31[0] & 65535;
                epi_staging[swz1_1059 >> 1] = _fp8_31[0] >> 16;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                if (warp == 0) {
                    if (elect_sync()) {
                        int padding_rows_3 = (256 - valid_rows % 256) % 256;
                        tma_store_4d((&C), m_tile * 64, padding_rows_3 + 192, 1073741824, n_tile * 256 - (unsigned int)padding_rows_3 + 1073741824, epi_staging_addr);
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
            int routed[16];
            unsigned int cta_mask = 1 << cta_rank;
            unsigned int _phase_k_done = 1;
            unsigned int _phase_work_full_1 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_1 = 0; _tile_iter_1 < grid_m / 2 * grid_n; _tile_iter_1++) {
                if (m_tile_1 >= (unsigned int)grid_m || n_tile_1 >= (unsigned int)bound_1) {
                    break;
                }
                route_base = n_tile_1 * 256 + (unsigned int)(cta_rank * 128) + (unsigned int)(warp_local * 4);
                for (int row = 0; row < 4; row++) {
                    routed[row] = route_map[route_base + row];
                }
                route_base = n_tile_1 * 256 + (unsigned int)(cta_rank * 128) + (unsigned int)((8 + warp_local) * 4);
                for (int row_1 = 0; row_1 < 4; row_1++) {
                    routed[4 + row_1] = route_map[route_base + row_1];
                }
                route_base = n_tile_1 * 256 + (unsigned int)(cta_rank * 128) + (unsigned int)((16 + warp_local) * 4);
                for (int row_2 = 0; row_2 < 4; row_2++) {
                    routed[8 + row_2] = route_map[route_base + row_2];
                }
                route_base = n_tile_1 * 256 + (unsigned int)(cta_rank * 128) + (unsigned int)((24 + warp_local) * 4);
                for (int row_3 = 0; row_3 < 4; row_3++) {
                    routed[12 + row_3] = route_map[route_base + row_3];
                }
                #pragma unroll 1
                for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                    mbarrier_wait(k_done_addr + (stage) * 8, _phase_k_done);
                    if (elect_sync()) {
                        tma_gather4_gmem2smem_mc_cta2(smem_b_addr + stage * 16384 + (unsigned int)(warp_local * 512), (&B), iter_k * 128, routed[0], routed[1], routed[2], routed[3], ((b_full_addr + (stage) * 8) & 0xFEFFFFFF), cta_mask);
                        tma_gather4_gmem2smem_mc_cta2(smem_b_addr + stage * 16384 + (unsigned int)((8 + warp_local) * 512), (&B), iter_k * 128, routed[4], routed[5], routed[6], routed[7], ((b_full_addr + (stage) * 8) & 0xFEFFFFFF), cta_mask);
                        tma_gather4_gmem2smem_mc_cta2(smem_b_addr + stage * 16384 + (unsigned int)((16 + warp_local) * 512), (&B), iter_k * 128, routed[8], routed[9], routed[10], routed[11], ((b_full_addr + (stage) * 8) & 0xFEFFFFFF), cta_mask);
                        tma_gather4_gmem2smem_mc_cta2(smem_b_addr + stage * 16384 + (unsigned int)((24 + warp_local) * 512), (&B), iter_k * 128, routed[12], routed[13], routed[14], routed[15], ((b_full_addr + (stage) * 8) & 0xFEFFFFFF), cta_mask);
                    }
                    if (warp == 4) {
                        if (elect_sync()) {
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((b_full_addr + (stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                        }
                    }
                    stage += 1;
                    if (stage == 6) { stage = 0; _phase_k_done ^= 1; }
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
                    if (stage_1 == 6) { stage_1 = 0; _phase_k_done_1 ^= 1; }
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
                        int _mma_b_lo_0 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_2) * 1024;
                        mma_ss_step_cg2(_mma_a_lo_0, _mma_b_lo_0, (tmem_accum + (acc_stage_1 * 256)), 272629776, ((((1) ? ((iter_k_2 == 0) ? 1 : 0) : 0)) ? 0 : 1), 0x40004040U, 0x40004040U);
                        int _mma_a_lo_1 = (((smem_a_addr + 32) >> 4) & 0x3FFF) + (stage_2) * 1024;
                        int _mma_b_lo_1 = (((smem_b_addr + 32) >> 4) & 0x3FFF) + (stage_2) * 1024;
                        mma_ss_step_cg2(_mma_a_lo_1, _mma_b_lo_1, (tmem_accum + (acc_stage_1 * 256)), 272629776, ((((0) ? ((iter_k_2 == 0) ? 1 : 0) : 0)) ? 0 : 1), 0x40004040U, 0x40004040U);
                        int _mma_a_lo_2 = (((smem_a_addr + 64) >> 4) & 0x3FFF) + (stage_2) * 1024;
                        int _mma_b_lo_2 = (((smem_b_addr + 64) >> 4) & 0x3FFF) + (stage_2) * 1024;
                        mma_ss_step_cg2(_mma_a_lo_2, _mma_b_lo_2, (tmem_accum + (acc_stage_1 * 256)), 272629776, ((((0) ? ((iter_k_2 == 0) ? 1 : 0) : 0)) ? 0 : 1), 0x40004040U, 0x40004040U);
                        int _mma_a_lo_3 = (((smem_a_addr + 96) >> 4) & 0x3FFF) + (stage_2) * 1024;
                        int _mma_b_lo_3 = (((smem_b_addr + 96) >> 4) & 0x3FFF) + (stage_2) * 1024;
                        mma_ss_step_cg2(_mma_a_lo_3, _mma_b_lo_3, (tmem_accum + (acc_stage_1 * 256)), 272629776, ((((0) ? ((iter_k_2 == 0) ? 1 : 0) : 0)) ? 0 : 1), 0x40004040U, 0x40004040U);
                        elect_commit_cg2_multicast(k_done_addr + (stage_2) * 8, (uint16_t)(3));
                        if (iter_k_2 + 1 == K_tiles) {
                            elect_commit_cg2_multicast(mma_full_addr + (acc_stage_1) * 8, (uint16_t)(3));
                        }
                        stage_2 += 1;
                        if (stage_2 == 6) { stage_2 = 0; _phase_a_full ^= 1; _phase_b_full ^= 1; }
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
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
