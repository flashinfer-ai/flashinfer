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
#define NUM_K_PIPE_STAGES 6
#define NUM_MMA_PIPE_STAGES 2
#define NUM_WORK_PIPE_STAGES 3
#define NUM_THROTTLE_PIPE_STAGES 3
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 16384
#define SMEM_SMEM_B_OFF 99328
#define SMEM_SMEM_B_STAGE_BYTES 8192
#define SMEM_SMEM_B_STRIDE 8192
#define SMEM_EPI_STAGING_OFF 148480
#define SMEM_EPI_STAGING_STAGE_BYTES 4096
#define SMEM_EPI_STAGING_STRIDE 4096
#define SMEM_EPI_STAGING_U32_OFF 148480
#define SMEM_EPI_STAGING_U32_STAGE_BYTES 4096
#define SMEM_EPI_STAGING_U32_STRIDE 4096
#define SMEM_WORK_RESPONSE_OFF 152576
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_TOTAL 152704
#define THREADS 384
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

__global__ __launch_bounds__(384, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_cake_stepfun_moe_9667e601028923995bfb(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap C, int* __restrict__ route_map, int* __restrict__ tile_expert, int* __restrict__ tile_mn_limit, int* __restrict__ num_non_exiting_ctas, int M_out, int K, int grid_m, int grid_n, int K_tiles, float* __restrict__ clamp_limit)
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

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    __nv_bfloat16* smem_a = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
    const int smem_b_addr = smem + 99328;
    __nv_bfloat16* epi_staging = reinterpret_cast<__nv_bfloat16*>(smem_raw + 148480);
    const int epi_staging_addr = smem + 148480;
    unsigned int* epi_staging_u32 = reinterpret_cast<unsigned int*>(smem_raw + 148480);
    const int epi_staging_u32_addr = smem + 148480;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 152576);
    const int work_response_addr = smem + 152576;
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if ((int)blockIdx.y >= num_non_exiting_ctas[0]) return;

    // Mbarrier init (9 pipeline groups, 0 ordered-sequence groups, 34 barriers)
    // Mbarriers at smem_raw[0..272)

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
            // work_empty: 3 barriers, init_count=704
            mbarrier_init(smem + 200, 704);
            mbarrier_init(smem + 208, 704);
            mbarrier_init(smem + 216, 704);
            // --- pipeline 'throttle_pipe' ---
            // throttle_full: 3 barriers, init_count=32
            mbarrier_init(smem + 224, 32);
            mbarrier_init(smem + 232, 32);
            mbarrier_init(smem + 240, 32);
            // throttle_empty: 3 barriers, init_count=32
            mbarrier_init(smem + 248, 32);
            mbarrier_init(smem + 256, 32);
            mbarrier_init(smem + 264, 32);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 272);
    if (warp == 0) {
        int _tmem_hold = smem + 272;
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
    if (warp >= 8 && warp <= 11) {
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
            int base_row = warp_0 * 16 + lane_1 / 4 * 2;
            int feature_chunk = base_row / 8;
            int feature_word = base_row % 8 / 2;
            float pair[2];
            unsigned int word[1];
            unsigned int _phase_mma_full = 0;
            unsigned int _phase_work_full = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter = 0; _tile_iter < grid_m / 2 * grid_n; _tile_iter++) {
                if (m_tile >= (unsigned int)grid_m || n_tile >= (unsigned int)num_non_exiting_ctas[0]) {
                    break;
                }
                int expert = tile_expert[n_tile];
                int valid_rows = (unsigned int)tile_mn_limit[n_tile] - n_tile * 128;
                float cl = clamp_limit[expert];
                float _sub_0;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_0) : "f"(0.0f), "f"(cl));
                float neg_cl = _sub_0;
                mbarrier_wait(mma_full_addr + (acc_stage) * 8, _phase_mma_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int acc_offset = acc_stage * 128;
                float _tmem_load_0[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15]))
                    : "r"(taddr + (unsigned int)acc_offset));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float _tmem_load_1[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15]))
                    : "r"(taddr + 1048576 + (unsigned int)acc_offset));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                asm volatile("cp.async.bulk.wait_group.read 0;");
                asm volatile("barrier.sync 7, 128;" ::: "memory");
                int local_token0 = lane_1 % 4 * 2;
                int local_token1 = local_token0 + 1;
                float x00 = _tmem_load_0[2];
                float x01 = _tmem_load_0[3];
                float x10 = _tmem_load_1[2];
                float x11 = _tmem_load_1[3];
                float _mul_0;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_0) : "f"(x00), "f"(-1.4426950408889634f));
                float _exp2_0 = approx_exp2(_mul_0);
                float _add_0;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_0) : "f"(1.0f), "f"(_exp2_0));
                float _rcp_0 = approx_rcp(_add_0);
                float sig00 = _rcp_0;
                float _mul_1;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_1) : "f"(x01), "f"(-1.4426950408889634f));
                float _exp2_1 = approx_exp2(_mul_1);
                float _add_1;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_1) : "f"(1.0f), "f"(_exp2_1));
                float _rcp_1 = approx_rcp(_add_1);
                float sig01 = _rcp_1;
                float _mul_2;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_2) : "f"(x10), "f"(-1.4426950408889634f));
                float _exp2_2 = approx_exp2(_mul_2);
                float _add_2;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_2) : "f"(1.0f), "f"(_exp2_2));
                float _rcp_2 = approx_rcp(_add_2);
                float sig10 = _rcp_2;
                float _mul_3;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_3) : "f"(x11), "f"(-1.4426950408889634f));
                float _exp2_3 = approx_exp2(_mul_3);
                float _add_3;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_3) : "f"(1.0f), "f"(_exp2_3));
                float _rcp_3 = approx_rcp(_add_3);
                float sig11 = _rcp_3;
                float _mul_4;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_4) : "f"(x00), "f"(sig00));
                float g00 = _mul_4;
                float _mul_5;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_5) : "f"(x01), "f"(sig01));
                float g01 = _mul_5;
                float _mul_6;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_6) : "f"(x10), "f"(sig10));
                float g10 = _mul_6;
                float _mul_7;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_7) : "f"(x11), "f"(sig11));
                float g11 = _mul_7;
                float lin00 = _tmem_load_0[0];
                float lin01 = _tmem_load_0[1];
                float lin10 = _tmem_load_1[0];
                float lin11 = _tmem_load_1[1];
                float _max_0;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_0) : "f"(lin00), "f"(neg_cl));
                float _min_0;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_0) : "f"(_max_0), "f"(cl));
                lin00 = _min_0;
                float _max_1;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_1) : "f"(lin01), "f"(neg_cl));
                float _min_1;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_1) : "f"(_max_1), "f"(cl));
                lin01 = _min_1;
                float _max_2;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_2) : "f"(lin10), "f"(neg_cl));
                float _min_2;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_2) : "f"(_max_2), "f"(cl));
                lin10 = _min_2;
                float _max_3;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_3) : "f"(lin11), "f"(neg_cl));
                float _min_3;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_3) : "f"(_max_3), "f"(cl));
                lin11 = _min_3;
                float _min_4;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_4) : "f"(g00), "f"(cl));
                g00 = _min_4;
                float _min_5;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_5) : "f"(g01), "f"(cl));
                g01 = _min_5;
                float _min_6;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_6) : "f"(g10), "f"(cl));
                g10 = _min_6;
                float _min_7;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_7) : "f"(g11), "f"(cl));
                g11 = _min_7;
                float _mul_8;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_8) : "f"(lin00), "f"(g00));
                float value00 = _mul_8;
                float _mul_9;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_9) : "f"(lin01), "f"(g01));
                float value01 = _mul_9;
                float _mul_10;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_10) : "f"(lin10), "f"(g10));
                float value10 = _mul_10;
                float _mul_11;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_11) : "f"(lin11), "f"(g11));
                float value11 = _mul_11;
                pair[0] = value00;
                pair[1] = value10;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token0 * 32 + (feature_chunk ^ local_token0 % 8) * 4 + feature_word] = word[0];
                pair[0] = value01;
                pair[1] = value11;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token1 * 32 + (feature_chunk ^ local_token1 % 8) * 4 + feature_word] = word[0];
                int local_token0_0 = lane_1 % 4 * 2 + 8;
                int local_token1_1 = local_token0_0 + 1;
                float x00_2 = _tmem_load_0[6];
                float x01_3 = _tmem_load_0[7];
                float x10_4 = _tmem_load_1[6];
                float x11_5 = _tmem_load_1[7];
                float _mul_12;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_12) : "f"(x00_2), "f"(-1.4426950408889634f));
                float _exp2_4 = approx_exp2(_mul_12);
                float _add_4;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_4) : "f"(1.0f), "f"(_exp2_4));
                float _rcp_4 = approx_rcp(_add_4);
                float sig00_6 = _rcp_4;
                float _mul_13;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_13) : "f"(x01_3), "f"(-1.4426950408889634f));
                float _exp2_5 = approx_exp2(_mul_13);
                float _add_5;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_5) : "f"(1.0f), "f"(_exp2_5));
                float _rcp_5 = approx_rcp(_add_5);
                float sig01_7 = _rcp_5;
                float _mul_14;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_14) : "f"(x10_4), "f"(-1.4426950408889634f));
                float _exp2_6 = approx_exp2(_mul_14);
                float _add_6;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_6) : "f"(1.0f), "f"(_exp2_6));
                float _rcp_6 = approx_rcp(_add_6);
                float sig10_8 = _rcp_6;
                float _mul_15;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_15) : "f"(x11_5), "f"(-1.4426950408889634f));
                float _exp2_7 = approx_exp2(_mul_15);
                float _add_7;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_7) : "f"(1.0f), "f"(_exp2_7));
                float _rcp_7 = approx_rcp(_add_7);
                float sig11_9 = _rcp_7;
                float _mul_16;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_16) : "f"(x00_2), "f"(sig00_6));
                float g00_10 = _mul_16;
                float _mul_17;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_17) : "f"(x01_3), "f"(sig01_7));
                float g01_11 = _mul_17;
                float _mul_18;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_18) : "f"(x10_4), "f"(sig10_8));
                float g10_12 = _mul_18;
                float _mul_19;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_19) : "f"(x11_5), "f"(sig11_9));
                float g11_13 = _mul_19;
                float lin00_14 = _tmem_load_0[4];
                float lin01_15 = _tmem_load_0[5];
                float lin10_16 = _tmem_load_1[4];
                float lin11_17 = _tmem_load_1[5];
                float _max_4;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_4) : "f"(lin00_14), "f"(neg_cl));
                float _min_8;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_8) : "f"(_max_4), "f"(cl));
                lin00_14 = _min_8;
                float _max_5;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_5) : "f"(lin01_15), "f"(neg_cl));
                float _min_9;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_9) : "f"(_max_5), "f"(cl));
                lin01_15 = _min_9;
                float _max_6;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_6) : "f"(lin10_16), "f"(neg_cl));
                float _min_10;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_10) : "f"(_max_6), "f"(cl));
                lin10_16 = _min_10;
                float _max_7;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_7) : "f"(lin11_17), "f"(neg_cl));
                float _min_11;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_11) : "f"(_max_7), "f"(cl));
                lin11_17 = _min_11;
                float _min_12;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_12) : "f"(g00_10), "f"(cl));
                g00_10 = _min_12;
                float _min_13;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_13) : "f"(g01_11), "f"(cl));
                g01_11 = _min_13;
                float _min_14;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_14) : "f"(g10_12), "f"(cl));
                g10_12 = _min_14;
                float _min_15;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_15) : "f"(g11_13), "f"(cl));
                g11_13 = _min_15;
                float _mul_20;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_20) : "f"(lin00_14), "f"(g00_10));
                float value00_18 = _mul_20;
                float _mul_21;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_21) : "f"(lin01_15), "f"(g01_11));
                float value01_19 = _mul_21;
                float _mul_22;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_22) : "f"(lin10_16), "f"(g10_12));
                float value10_20 = _mul_22;
                float _mul_23;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_23) : "f"(lin11_17), "f"(g11_13));
                float value11_21 = _mul_23;
                pair[0] = value00_18;
                pair[1] = value10_20;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token0_0 * 32 + (feature_chunk ^ local_token0_0 % 8) * 4 + feature_word] = word[0];
                pair[0] = value01_19;
                pair[1] = value11_21;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token1_1 * 32 + (feature_chunk ^ local_token1_1 % 8) * 4 + feature_word] = word[0];
                int local_token0_22 = lane_1 % 4 * 2 + 16;
                int local_token1_23 = local_token0_22 + 1;
                float x00_24 = _tmem_load_0[10];
                float x01_25 = _tmem_load_0[11];
                float x10_26 = _tmem_load_1[10];
                float x11_27 = _tmem_load_1[11];
                float _mul_24;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_24) : "f"(x00_24), "f"(-1.4426950408889634f));
                float _exp2_8 = approx_exp2(_mul_24);
                float _add_8;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_8) : "f"(1.0f), "f"(_exp2_8));
                float _rcp_8 = approx_rcp(_add_8);
                float sig00_28 = _rcp_8;
                float _mul_25;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_25) : "f"(x01_25), "f"(-1.4426950408889634f));
                float _exp2_9 = approx_exp2(_mul_25);
                float _add_9;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_9) : "f"(1.0f), "f"(_exp2_9));
                float _rcp_9 = approx_rcp(_add_9);
                float sig01_29 = _rcp_9;
                float _mul_26;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_26) : "f"(x10_26), "f"(-1.4426950408889634f));
                float _exp2_10 = approx_exp2(_mul_26);
                float _add_10;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_10) : "f"(1.0f), "f"(_exp2_10));
                float _rcp_10 = approx_rcp(_add_10);
                float sig10_30 = _rcp_10;
                float _mul_27;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_27) : "f"(x11_27), "f"(-1.4426950408889634f));
                float _exp2_11 = approx_exp2(_mul_27);
                float _add_11;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_11) : "f"(1.0f), "f"(_exp2_11));
                float _rcp_11 = approx_rcp(_add_11);
                float sig11_31 = _rcp_11;
                float _mul_28;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_28) : "f"(x00_24), "f"(sig00_28));
                float g00_32 = _mul_28;
                float _mul_29;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_29) : "f"(x01_25), "f"(sig01_29));
                float g01_33 = _mul_29;
                float _mul_30;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_30) : "f"(x10_26), "f"(sig10_30));
                float g10_34 = _mul_30;
                float _mul_31;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_31) : "f"(x11_27), "f"(sig11_31));
                float g11_35 = _mul_31;
                float lin00_36 = _tmem_load_0[8];
                float lin01_37 = _tmem_load_0[9];
                float lin10_38 = _tmem_load_1[8];
                float lin11_39 = _tmem_load_1[9];
                float _max_8;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_8) : "f"(lin00_36), "f"(neg_cl));
                float _min_16;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_16) : "f"(_max_8), "f"(cl));
                lin00_36 = _min_16;
                float _max_9;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_9) : "f"(lin01_37), "f"(neg_cl));
                float _min_17;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_17) : "f"(_max_9), "f"(cl));
                lin01_37 = _min_17;
                float _max_10;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_10) : "f"(lin10_38), "f"(neg_cl));
                float _min_18;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_18) : "f"(_max_10), "f"(cl));
                lin10_38 = _min_18;
                float _max_11;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_11) : "f"(lin11_39), "f"(neg_cl));
                float _min_19;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_19) : "f"(_max_11), "f"(cl));
                lin11_39 = _min_19;
                float _min_20;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_20) : "f"(g00_32), "f"(cl));
                g00_32 = _min_20;
                float _min_21;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_21) : "f"(g01_33), "f"(cl));
                g01_33 = _min_21;
                float _min_22;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_22) : "f"(g10_34), "f"(cl));
                g10_34 = _min_22;
                float _min_23;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_23) : "f"(g11_35), "f"(cl));
                g11_35 = _min_23;
                float _mul_32;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_32) : "f"(lin00_36), "f"(g00_32));
                float value00_40 = _mul_32;
                float _mul_33;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_33) : "f"(lin01_37), "f"(g01_33));
                float value01_41 = _mul_33;
                float _mul_34;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_34) : "f"(lin10_38), "f"(g10_34));
                float value10_42 = _mul_34;
                float _mul_35;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_35) : "f"(lin11_39), "f"(g11_35));
                float value11_43 = _mul_35;
                pair[0] = value00_40;
                pair[1] = value10_42;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token0_22 * 32 + (feature_chunk ^ local_token0_22 % 8) * 4 + feature_word] = word[0];
                pair[0] = value01_41;
                pair[1] = value11_43;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token1_23 * 32 + (feature_chunk ^ local_token1_23 % 8) * 4 + feature_word] = word[0];
                int local_token0_44 = lane_1 % 4 * 2 + 24;
                int local_token1_45 = local_token0_44 + 1;
                float x00_46 = _tmem_load_0[14];
                float x01_47 = _tmem_load_0[15];
                float x10_48 = _tmem_load_1[14];
                float x11_49 = _tmem_load_1[15];
                float _mul_36;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_36) : "f"(x00_46), "f"(-1.4426950408889634f));
                float _exp2_12 = approx_exp2(_mul_36);
                float _add_12;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_12) : "f"(1.0f), "f"(_exp2_12));
                float _rcp_12 = approx_rcp(_add_12);
                float sig00_50 = _rcp_12;
                float _mul_37;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_37) : "f"(x01_47), "f"(-1.4426950408889634f));
                float _exp2_13 = approx_exp2(_mul_37);
                float _add_13;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_13) : "f"(1.0f), "f"(_exp2_13));
                float _rcp_13 = approx_rcp(_add_13);
                float sig01_51 = _rcp_13;
                float _mul_38;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_38) : "f"(x10_48), "f"(-1.4426950408889634f));
                float _exp2_14 = approx_exp2(_mul_38);
                float _add_14;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_14) : "f"(1.0f), "f"(_exp2_14));
                float _rcp_14 = approx_rcp(_add_14);
                float sig10_52 = _rcp_14;
                float _mul_39;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_39) : "f"(x11_49), "f"(-1.4426950408889634f));
                float _exp2_15 = approx_exp2(_mul_39);
                float _add_15;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_15) : "f"(1.0f), "f"(_exp2_15));
                float _rcp_15 = approx_rcp(_add_15);
                float sig11_53 = _rcp_15;
                float _mul_40;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_40) : "f"(x00_46), "f"(sig00_50));
                float g00_54 = _mul_40;
                float _mul_41;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_41) : "f"(x01_47), "f"(sig01_51));
                float g01_55 = _mul_41;
                float _mul_42;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_42) : "f"(x10_48), "f"(sig10_52));
                float g10_56 = _mul_42;
                float _mul_43;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_43) : "f"(x11_49), "f"(sig11_53));
                float g11_57 = _mul_43;
                float lin00_58 = _tmem_load_0[12];
                float lin01_59 = _tmem_load_0[13];
                float lin10_60 = _tmem_load_1[12];
                float lin11_61 = _tmem_load_1[13];
                float _max_12;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_12) : "f"(lin00_58), "f"(neg_cl));
                float _min_24;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_24) : "f"(_max_12), "f"(cl));
                lin00_58 = _min_24;
                float _max_13;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_13) : "f"(lin01_59), "f"(neg_cl));
                float _min_25;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_25) : "f"(_max_13), "f"(cl));
                lin01_59 = _min_25;
                float _max_14;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_14) : "f"(lin10_60), "f"(neg_cl));
                float _min_26;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_26) : "f"(_max_14), "f"(cl));
                lin10_60 = _min_26;
                float _max_15;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_15) : "f"(lin11_61), "f"(neg_cl));
                float _min_27;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_27) : "f"(_max_15), "f"(cl));
                lin11_61 = _min_27;
                float _min_28;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_28) : "f"(g00_54), "f"(cl));
                g00_54 = _min_28;
                float _min_29;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_29) : "f"(g01_55), "f"(cl));
                g01_55 = _min_29;
                float _min_30;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_30) : "f"(g10_56), "f"(cl));
                g10_56 = _min_30;
                float _min_31;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_31) : "f"(g11_57), "f"(cl));
                g11_57 = _min_31;
                float _mul_44;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_44) : "f"(lin00_58), "f"(g00_54));
                float value00_62 = _mul_44;
                float _mul_45;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_45) : "f"(lin01_59), "f"(g01_55));
                float value01_63 = _mul_45;
                float _mul_46;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_46) : "f"(lin10_60), "f"(g10_56));
                float value10_64 = _mul_46;
                float _mul_47;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_47) : "f"(lin11_61), "f"(g11_57));
                float value11_65 = _mul_47;
                pair[0] = value00_62;
                pair[1] = value10_64;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token0_44 * 32 + (feature_chunk ^ local_token0_44 % 8) * 4 + feature_word] = word[0];
                pair[0] = value01_63;
                pair[1] = value11_65;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token1_45 * 32 + (feature_chunk ^ local_token1_45 % 8) * 4 + feature_word] = word[0];
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 7, 128;" ::: "memory");
                if (warp == 0) {
                    if (elect_sync()) {
                        int padding_rows = (128 - valid_rows % 128) % 128;
                        tma_store_4d((&C), m_tile * 64, padding_rows, 1073741824, n_tile * 128 - (unsigned int)padding_rows + 1073741824, epi_staging_addr);
                    }
                }
                asm volatile("cp.async.bulk.commit_group;");
                asm volatile("barrier.sync 7, 128;" ::: "memory");
                int acc_offset_66 = acc_stage * 128 + 32;
                float _tmem_load_2[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[15]))
                    : "r"(taddr + (unsigned int)acc_offset_66));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float _tmem_load_3[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[15]))
                    : "r"(taddr + 1048576 + (unsigned int)acc_offset_66));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                asm volatile("cp.async.bulk.wait_group.read 0;");
                asm volatile("barrier.sync 7, 128;" ::: "memory");
                int local_token0_67 = lane_1 % 4 * 2;
                int local_token1_68 = local_token0_67 + 1;
                float x00_69 = _tmem_load_2[2];
                float x01_70 = _tmem_load_2[3];
                float x10_71 = _tmem_load_3[2];
                float x11_72 = _tmem_load_3[3];
                float _mul_48;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_48) : "f"(x00_69), "f"(-1.4426950408889634f));
                float _exp2_16 = approx_exp2(_mul_48);
                float _add_16;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_16) : "f"(1.0f), "f"(_exp2_16));
                float _rcp_16 = approx_rcp(_add_16);
                float sig00_73 = _rcp_16;
                float _mul_49;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_49) : "f"(x01_70), "f"(-1.4426950408889634f));
                float _exp2_17 = approx_exp2(_mul_49);
                float _add_17;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_17) : "f"(1.0f), "f"(_exp2_17));
                float _rcp_17 = approx_rcp(_add_17);
                float sig01_74 = _rcp_17;
                float _mul_50;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_50) : "f"(x10_71), "f"(-1.4426950408889634f));
                float _exp2_18 = approx_exp2(_mul_50);
                float _add_18;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_18) : "f"(1.0f), "f"(_exp2_18));
                float _rcp_18 = approx_rcp(_add_18);
                float sig10_75 = _rcp_18;
                float _mul_51;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_51) : "f"(x11_72), "f"(-1.4426950408889634f));
                float _exp2_19 = approx_exp2(_mul_51);
                float _add_19;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_19) : "f"(1.0f), "f"(_exp2_19));
                float _rcp_19 = approx_rcp(_add_19);
                float sig11_76 = _rcp_19;
                float _mul_52;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_52) : "f"(x00_69), "f"(sig00_73));
                float g00_77 = _mul_52;
                float _mul_53;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_53) : "f"(x01_70), "f"(sig01_74));
                float g01_78 = _mul_53;
                float _mul_54;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_54) : "f"(x10_71), "f"(sig10_75));
                float g10_79 = _mul_54;
                float _mul_55;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_55) : "f"(x11_72), "f"(sig11_76));
                float g11_80 = _mul_55;
                float lin00_81 = _tmem_load_2[0];
                float lin01_82 = _tmem_load_2[1];
                float lin10_83 = _tmem_load_3[0];
                float lin11_84 = _tmem_load_3[1];
                float _max_16;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_16) : "f"(lin00_81), "f"(neg_cl));
                float _min_32;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_32) : "f"(_max_16), "f"(cl));
                lin00_81 = _min_32;
                float _max_17;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_17) : "f"(lin01_82), "f"(neg_cl));
                float _min_33;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_33) : "f"(_max_17), "f"(cl));
                lin01_82 = _min_33;
                float _max_18;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_18) : "f"(lin10_83), "f"(neg_cl));
                float _min_34;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_34) : "f"(_max_18), "f"(cl));
                lin10_83 = _min_34;
                float _max_19;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_19) : "f"(lin11_84), "f"(neg_cl));
                float _min_35;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_35) : "f"(_max_19), "f"(cl));
                lin11_84 = _min_35;
                float _min_36;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_36) : "f"(g00_77), "f"(cl));
                g00_77 = _min_36;
                float _min_37;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_37) : "f"(g01_78), "f"(cl));
                g01_78 = _min_37;
                float _min_38;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_38) : "f"(g10_79), "f"(cl));
                g10_79 = _min_38;
                float _min_39;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_39) : "f"(g11_80), "f"(cl));
                g11_80 = _min_39;
                float _mul_56;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_56) : "f"(lin00_81), "f"(g00_77));
                float value00_85 = _mul_56;
                float _mul_57;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_57) : "f"(lin01_82), "f"(g01_78));
                float value01_86 = _mul_57;
                float _mul_58;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_58) : "f"(lin10_83), "f"(g10_79));
                float value10_87 = _mul_58;
                float _mul_59;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_59) : "f"(lin11_84), "f"(g11_80));
                float value11_88 = _mul_59;
                pair[0] = value00_85;
                pair[1] = value10_87;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token0_67 * 32 + (feature_chunk ^ local_token0_67 % 8) * 4 + feature_word] = word[0];
                pair[0] = value01_86;
                pair[1] = value11_88;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token1_68 * 32 + (feature_chunk ^ local_token1_68 % 8) * 4 + feature_word] = word[0];
                int local_token0_89 = lane_1 % 4 * 2 + 8;
                int local_token1_90 = local_token0_89 + 1;
                float x00_91 = _tmem_load_2[6];
                float x01_92 = _tmem_load_2[7];
                float x10_93 = _tmem_load_3[6];
                float x11_94 = _tmem_load_3[7];
                float _mul_60;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_60) : "f"(x00_91), "f"(-1.4426950408889634f));
                float _exp2_20 = approx_exp2(_mul_60);
                float _add_20;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_20) : "f"(1.0f), "f"(_exp2_20));
                float _rcp_20 = approx_rcp(_add_20);
                float sig00_95 = _rcp_20;
                float _mul_61;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_61) : "f"(x01_92), "f"(-1.4426950408889634f));
                float _exp2_21 = approx_exp2(_mul_61);
                float _add_21;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_21) : "f"(1.0f), "f"(_exp2_21));
                float _rcp_21 = approx_rcp(_add_21);
                float sig01_96 = _rcp_21;
                float _mul_62;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_62) : "f"(x10_93), "f"(-1.4426950408889634f));
                float _exp2_22 = approx_exp2(_mul_62);
                float _add_22;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_22) : "f"(1.0f), "f"(_exp2_22));
                float _rcp_22 = approx_rcp(_add_22);
                float sig10_97 = _rcp_22;
                float _mul_63;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_63) : "f"(x11_94), "f"(-1.4426950408889634f));
                float _exp2_23 = approx_exp2(_mul_63);
                float _add_23;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_23) : "f"(1.0f), "f"(_exp2_23));
                float _rcp_23 = approx_rcp(_add_23);
                float sig11_98 = _rcp_23;
                float _mul_64;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_64) : "f"(x00_91), "f"(sig00_95));
                float g00_99 = _mul_64;
                float _mul_65;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_65) : "f"(x01_92), "f"(sig01_96));
                float g01_100 = _mul_65;
                float _mul_66;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_66) : "f"(x10_93), "f"(sig10_97));
                float g10_101 = _mul_66;
                float _mul_67;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_67) : "f"(x11_94), "f"(sig11_98));
                float g11_102 = _mul_67;
                float lin00_103 = _tmem_load_2[4];
                float lin01_104 = _tmem_load_2[5];
                float lin10_105 = _tmem_load_3[4];
                float lin11_106 = _tmem_load_3[5];
                float _max_20;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_20) : "f"(lin00_103), "f"(neg_cl));
                float _min_40;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_40) : "f"(_max_20), "f"(cl));
                lin00_103 = _min_40;
                float _max_21;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_21) : "f"(lin01_104), "f"(neg_cl));
                float _min_41;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_41) : "f"(_max_21), "f"(cl));
                lin01_104 = _min_41;
                float _max_22;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_22) : "f"(lin10_105), "f"(neg_cl));
                float _min_42;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_42) : "f"(_max_22), "f"(cl));
                lin10_105 = _min_42;
                float _max_23;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_23) : "f"(lin11_106), "f"(neg_cl));
                float _min_43;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_43) : "f"(_max_23), "f"(cl));
                lin11_106 = _min_43;
                float _min_44;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_44) : "f"(g00_99), "f"(cl));
                g00_99 = _min_44;
                float _min_45;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_45) : "f"(g01_100), "f"(cl));
                g01_100 = _min_45;
                float _min_46;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_46) : "f"(g10_101), "f"(cl));
                g10_101 = _min_46;
                float _min_47;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_47) : "f"(g11_102), "f"(cl));
                g11_102 = _min_47;
                float _mul_68;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_68) : "f"(lin00_103), "f"(g00_99));
                float value00_107 = _mul_68;
                float _mul_69;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_69) : "f"(lin01_104), "f"(g01_100));
                float value01_108 = _mul_69;
                float _mul_70;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_70) : "f"(lin10_105), "f"(g10_101));
                float value10_109 = _mul_70;
                float _mul_71;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_71) : "f"(lin11_106), "f"(g11_102));
                float value11_110 = _mul_71;
                pair[0] = value00_107;
                pair[1] = value10_109;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token0_89 * 32 + (feature_chunk ^ local_token0_89 % 8) * 4 + feature_word] = word[0];
                pair[0] = value01_108;
                pair[1] = value11_110;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token1_90 * 32 + (feature_chunk ^ local_token1_90 % 8) * 4 + feature_word] = word[0];
                int local_token0_111 = lane_1 % 4 * 2 + 16;
                int local_token1_112 = local_token0_111 + 1;
                float x00_113 = _tmem_load_2[10];
                float x01_114 = _tmem_load_2[11];
                float x10_115 = _tmem_load_3[10];
                float x11_116 = _tmem_load_3[11];
                float _mul_72;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_72) : "f"(x00_113), "f"(-1.4426950408889634f));
                float _exp2_24 = approx_exp2(_mul_72);
                float _add_24;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_24) : "f"(1.0f), "f"(_exp2_24));
                float _rcp_24 = approx_rcp(_add_24);
                float sig00_117 = _rcp_24;
                float _mul_73;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_73) : "f"(x01_114), "f"(-1.4426950408889634f));
                float _exp2_25 = approx_exp2(_mul_73);
                float _add_25;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_25) : "f"(1.0f), "f"(_exp2_25));
                float _rcp_25 = approx_rcp(_add_25);
                float sig01_118 = _rcp_25;
                float _mul_74;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_74) : "f"(x10_115), "f"(-1.4426950408889634f));
                float _exp2_26 = approx_exp2(_mul_74);
                float _add_26;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_26) : "f"(1.0f), "f"(_exp2_26));
                float _rcp_26 = approx_rcp(_add_26);
                float sig10_119 = _rcp_26;
                float _mul_75;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_75) : "f"(x11_116), "f"(-1.4426950408889634f));
                float _exp2_27 = approx_exp2(_mul_75);
                float _add_27;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_27) : "f"(1.0f), "f"(_exp2_27));
                float _rcp_27 = approx_rcp(_add_27);
                float sig11_120 = _rcp_27;
                float _mul_76;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_76) : "f"(x00_113), "f"(sig00_117));
                float g00_121 = _mul_76;
                float _mul_77;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_77) : "f"(x01_114), "f"(sig01_118));
                float g01_122 = _mul_77;
                float _mul_78;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_78) : "f"(x10_115), "f"(sig10_119));
                float g10_123 = _mul_78;
                float _mul_79;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_79) : "f"(x11_116), "f"(sig11_120));
                float g11_124 = _mul_79;
                float lin00_125 = _tmem_load_2[8];
                float lin01_126 = _tmem_load_2[9];
                float lin10_127 = _tmem_load_3[8];
                float lin11_128 = _tmem_load_3[9];
                float _max_24;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_24) : "f"(lin00_125), "f"(neg_cl));
                float _min_48;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_48) : "f"(_max_24), "f"(cl));
                lin00_125 = _min_48;
                float _max_25;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_25) : "f"(lin01_126), "f"(neg_cl));
                float _min_49;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_49) : "f"(_max_25), "f"(cl));
                lin01_126 = _min_49;
                float _max_26;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_26) : "f"(lin10_127), "f"(neg_cl));
                float _min_50;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_50) : "f"(_max_26), "f"(cl));
                lin10_127 = _min_50;
                float _max_27;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_27) : "f"(lin11_128), "f"(neg_cl));
                float _min_51;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_51) : "f"(_max_27), "f"(cl));
                lin11_128 = _min_51;
                float _min_52;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_52) : "f"(g00_121), "f"(cl));
                g00_121 = _min_52;
                float _min_53;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_53) : "f"(g01_122), "f"(cl));
                g01_122 = _min_53;
                float _min_54;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_54) : "f"(g10_123), "f"(cl));
                g10_123 = _min_54;
                float _min_55;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_55) : "f"(g11_124), "f"(cl));
                g11_124 = _min_55;
                float _mul_80;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_80) : "f"(lin00_125), "f"(g00_121));
                float value00_129 = _mul_80;
                float _mul_81;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_81) : "f"(lin01_126), "f"(g01_122));
                float value01_130 = _mul_81;
                float _mul_82;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_82) : "f"(lin10_127), "f"(g10_123));
                float value10_131 = _mul_82;
                float _mul_83;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_83) : "f"(lin11_128), "f"(g11_124));
                float value11_132 = _mul_83;
                pair[0] = value00_129;
                pair[1] = value10_131;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token0_111 * 32 + (feature_chunk ^ local_token0_111 % 8) * 4 + feature_word] = word[0];
                pair[0] = value01_130;
                pair[1] = value11_132;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token1_112 * 32 + (feature_chunk ^ local_token1_112 % 8) * 4 + feature_word] = word[0];
                int local_token0_133 = lane_1 % 4 * 2 + 24;
                int local_token1_134 = local_token0_133 + 1;
                float x00_135 = _tmem_load_2[14];
                float x01_136 = _tmem_load_2[15];
                float x10_137 = _tmem_load_3[14];
                float x11_138 = _tmem_load_3[15];
                float _mul_84;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_84) : "f"(x00_135), "f"(-1.4426950408889634f));
                float _exp2_28 = approx_exp2(_mul_84);
                float _add_28;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_28) : "f"(1.0f), "f"(_exp2_28));
                float _rcp_28 = approx_rcp(_add_28);
                float sig00_139 = _rcp_28;
                float _mul_85;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_85) : "f"(x01_136), "f"(-1.4426950408889634f));
                float _exp2_29 = approx_exp2(_mul_85);
                float _add_29;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_29) : "f"(1.0f), "f"(_exp2_29));
                float _rcp_29 = approx_rcp(_add_29);
                float sig01_140 = _rcp_29;
                float _mul_86;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_86) : "f"(x10_137), "f"(-1.4426950408889634f));
                float _exp2_30 = approx_exp2(_mul_86);
                float _add_30;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_30) : "f"(1.0f), "f"(_exp2_30));
                float _rcp_30 = approx_rcp(_add_30);
                float sig10_141 = _rcp_30;
                float _mul_87;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_87) : "f"(x11_138), "f"(-1.4426950408889634f));
                float _exp2_31 = approx_exp2(_mul_87);
                float _add_31;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_31) : "f"(1.0f), "f"(_exp2_31));
                float _rcp_31 = approx_rcp(_add_31);
                float sig11_142 = _rcp_31;
                float _mul_88;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_88) : "f"(x00_135), "f"(sig00_139));
                float g00_143 = _mul_88;
                float _mul_89;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_89) : "f"(x01_136), "f"(sig01_140));
                float g01_144 = _mul_89;
                float _mul_90;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_90) : "f"(x10_137), "f"(sig10_141));
                float g10_145 = _mul_90;
                float _mul_91;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_91) : "f"(x11_138), "f"(sig11_142));
                float g11_146 = _mul_91;
                float lin00_147 = _tmem_load_2[12];
                float lin01_148 = _tmem_load_2[13];
                float lin10_149 = _tmem_load_3[12];
                float lin11_150 = _tmem_load_3[13];
                float _max_28;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_28) : "f"(lin00_147), "f"(neg_cl));
                float _min_56;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_56) : "f"(_max_28), "f"(cl));
                lin00_147 = _min_56;
                float _max_29;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_29) : "f"(lin01_148), "f"(neg_cl));
                float _min_57;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_57) : "f"(_max_29), "f"(cl));
                lin01_148 = _min_57;
                float _max_30;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_30) : "f"(lin10_149), "f"(neg_cl));
                float _min_58;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_58) : "f"(_max_30), "f"(cl));
                lin10_149 = _min_58;
                float _max_31;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_31) : "f"(lin11_150), "f"(neg_cl));
                float _min_59;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_59) : "f"(_max_31), "f"(cl));
                lin11_150 = _min_59;
                float _min_60;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_60) : "f"(g00_143), "f"(cl));
                g00_143 = _min_60;
                float _min_61;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_61) : "f"(g01_144), "f"(cl));
                g01_144 = _min_61;
                float _min_62;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_62) : "f"(g10_145), "f"(cl));
                g10_145 = _min_62;
                float _min_63;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_63) : "f"(g11_146), "f"(cl));
                g11_146 = _min_63;
                float _mul_92;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_92) : "f"(lin00_147), "f"(g00_143));
                float value00_151 = _mul_92;
                float _mul_93;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_93) : "f"(lin01_148), "f"(g01_144));
                float value01_152 = _mul_93;
                float _mul_94;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_94) : "f"(lin10_149), "f"(g10_145));
                float value10_153 = _mul_94;
                float _mul_95;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_95) : "f"(lin11_150), "f"(g11_146));
                float value11_154 = _mul_95;
                pair[0] = value00_151;
                pair[1] = value10_153;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token0_133 * 32 + (feature_chunk ^ local_token0_133 % 8) * 4 + feature_word] = word[0];
                pair[0] = value01_152;
                pair[1] = value11_154;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token1_134 * 32 + (feature_chunk ^ local_token1_134 % 8) * 4 + feature_word] = word[0];
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 7, 128;" ::: "memory");
                if (warp == 0) {
                    if (elect_sync()) {
                        int padding_rows_1 = (128 - valid_rows % 128) % 128;
                        tma_store_4d((&C), m_tile * 64, padding_rows_1 + 32, 1073741824, n_tile * 128 - (unsigned int)padding_rows_1 + 1073741824, epi_staging_addr);
                    }
                }
                asm volatile("cp.async.bulk.commit_group;");
                asm volatile("barrier.sync 7, 128;" ::: "memory");
                int acc_offset_155 = acc_stage * 128 + 64;
                float _tmem_load_4[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[15]))
                    : "r"(taddr + (unsigned int)acc_offset_155));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float _tmem_load_5[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[15]))
                    : "r"(taddr + 1048576 + (unsigned int)acc_offset_155));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                asm volatile("cp.async.bulk.wait_group.read 0;");
                asm volatile("barrier.sync 7, 128;" ::: "memory");
                int local_token0_156 = lane_1 % 4 * 2;
                int local_token1_157 = local_token0_156 + 1;
                float x00_158 = _tmem_load_4[2];
                float x01_159 = _tmem_load_4[3];
                float x10_160 = _tmem_load_5[2];
                float x11_161 = _tmem_load_5[3];
                float _mul_96;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_96) : "f"(x00_158), "f"(-1.4426950408889634f));
                float _exp2_32 = approx_exp2(_mul_96);
                float _add_32;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_32) : "f"(1.0f), "f"(_exp2_32));
                float _rcp_32 = approx_rcp(_add_32);
                float sig00_162 = _rcp_32;
                float _mul_97;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_97) : "f"(x01_159), "f"(-1.4426950408889634f));
                float _exp2_33 = approx_exp2(_mul_97);
                float _add_33;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_33) : "f"(1.0f), "f"(_exp2_33));
                float _rcp_33 = approx_rcp(_add_33);
                float sig01_163 = _rcp_33;
                float _mul_98;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_98) : "f"(x10_160), "f"(-1.4426950408889634f));
                float _exp2_34 = approx_exp2(_mul_98);
                float _add_34;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_34) : "f"(1.0f), "f"(_exp2_34));
                float _rcp_34 = approx_rcp(_add_34);
                float sig10_164 = _rcp_34;
                float _mul_99;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_99) : "f"(x11_161), "f"(-1.4426950408889634f));
                float _exp2_35 = approx_exp2(_mul_99);
                float _add_35;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_35) : "f"(1.0f), "f"(_exp2_35));
                float _rcp_35 = approx_rcp(_add_35);
                float sig11_165 = _rcp_35;
                float _mul_100;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_100) : "f"(x00_158), "f"(sig00_162));
                float g00_166 = _mul_100;
                float _mul_101;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_101) : "f"(x01_159), "f"(sig01_163));
                float g01_167 = _mul_101;
                float _mul_102;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_102) : "f"(x10_160), "f"(sig10_164));
                float g10_168 = _mul_102;
                float _mul_103;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_103) : "f"(x11_161), "f"(sig11_165));
                float g11_169 = _mul_103;
                float lin00_170 = _tmem_load_4[0];
                float lin01_171 = _tmem_load_4[1];
                float lin10_172 = _tmem_load_5[0];
                float lin11_173 = _tmem_load_5[1];
                float _max_32;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_32) : "f"(lin00_170), "f"(neg_cl));
                float _min_64;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_64) : "f"(_max_32), "f"(cl));
                lin00_170 = _min_64;
                float _max_33;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_33) : "f"(lin01_171), "f"(neg_cl));
                float _min_65;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_65) : "f"(_max_33), "f"(cl));
                lin01_171 = _min_65;
                float _max_34;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_34) : "f"(lin10_172), "f"(neg_cl));
                float _min_66;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_66) : "f"(_max_34), "f"(cl));
                lin10_172 = _min_66;
                float _max_35;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_35) : "f"(lin11_173), "f"(neg_cl));
                float _min_67;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_67) : "f"(_max_35), "f"(cl));
                lin11_173 = _min_67;
                float _min_68;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_68) : "f"(g00_166), "f"(cl));
                g00_166 = _min_68;
                float _min_69;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_69) : "f"(g01_167), "f"(cl));
                g01_167 = _min_69;
                float _min_70;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_70) : "f"(g10_168), "f"(cl));
                g10_168 = _min_70;
                float _min_71;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_71) : "f"(g11_169), "f"(cl));
                g11_169 = _min_71;
                float _mul_104;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_104) : "f"(lin00_170), "f"(g00_166));
                float value00_174 = _mul_104;
                float _mul_105;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_105) : "f"(lin01_171), "f"(g01_167));
                float value01_175 = _mul_105;
                float _mul_106;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_106) : "f"(lin10_172), "f"(g10_168));
                float value10_176 = _mul_106;
                float _mul_107;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_107) : "f"(lin11_173), "f"(g11_169));
                float value11_177 = _mul_107;
                pair[0] = value00_174;
                pair[1] = value10_176;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token0_156 * 32 + (feature_chunk ^ local_token0_156 % 8) * 4 + feature_word] = word[0];
                pair[0] = value01_175;
                pair[1] = value11_177;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token1_157 * 32 + (feature_chunk ^ local_token1_157 % 8) * 4 + feature_word] = word[0];
                int local_token0_178 = lane_1 % 4 * 2 + 8;
                int local_token1_179 = local_token0_178 + 1;
                float x00_180 = _tmem_load_4[6];
                float x01_181 = _tmem_load_4[7];
                float x10_182 = _tmem_load_5[6];
                float x11_183 = _tmem_load_5[7];
                float _mul_108;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_108) : "f"(x00_180), "f"(-1.4426950408889634f));
                float _exp2_36 = approx_exp2(_mul_108);
                float _add_36;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_36) : "f"(1.0f), "f"(_exp2_36));
                float _rcp_36 = approx_rcp(_add_36);
                float sig00_184 = _rcp_36;
                float _mul_109;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_109) : "f"(x01_181), "f"(-1.4426950408889634f));
                float _exp2_37 = approx_exp2(_mul_109);
                float _add_37;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_37) : "f"(1.0f), "f"(_exp2_37));
                float _rcp_37 = approx_rcp(_add_37);
                float sig01_185 = _rcp_37;
                float _mul_110;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_110) : "f"(x10_182), "f"(-1.4426950408889634f));
                float _exp2_38 = approx_exp2(_mul_110);
                float _add_38;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_38) : "f"(1.0f), "f"(_exp2_38));
                float _rcp_38 = approx_rcp(_add_38);
                float sig10_186 = _rcp_38;
                float _mul_111;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_111) : "f"(x11_183), "f"(-1.4426950408889634f));
                float _exp2_39 = approx_exp2(_mul_111);
                float _add_39;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_39) : "f"(1.0f), "f"(_exp2_39));
                float _rcp_39 = approx_rcp(_add_39);
                float sig11_187 = _rcp_39;
                float _mul_112;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_112) : "f"(x00_180), "f"(sig00_184));
                float g00_188 = _mul_112;
                float _mul_113;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_113) : "f"(x01_181), "f"(sig01_185));
                float g01_189 = _mul_113;
                float _mul_114;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_114) : "f"(x10_182), "f"(sig10_186));
                float g10_190 = _mul_114;
                float _mul_115;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_115) : "f"(x11_183), "f"(sig11_187));
                float g11_191 = _mul_115;
                float lin00_192 = _tmem_load_4[4];
                float lin01_193 = _tmem_load_4[5];
                float lin10_194 = _tmem_load_5[4];
                float lin11_195 = _tmem_load_5[5];
                float _max_36;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_36) : "f"(lin00_192), "f"(neg_cl));
                float _min_72;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_72) : "f"(_max_36), "f"(cl));
                lin00_192 = _min_72;
                float _max_37;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_37) : "f"(lin01_193), "f"(neg_cl));
                float _min_73;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_73) : "f"(_max_37), "f"(cl));
                lin01_193 = _min_73;
                float _max_38;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_38) : "f"(lin10_194), "f"(neg_cl));
                float _min_74;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_74) : "f"(_max_38), "f"(cl));
                lin10_194 = _min_74;
                float _max_39;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_39) : "f"(lin11_195), "f"(neg_cl));
                float _min_75;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_75) : "f"(_max_39), "f"(cl));
                lin11_195 = _min_75;
                float _min_76;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_76) : "f"(g00_188), "f"(cl));
                g00_188 = _min_76;
                float _min_77;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_77) : "f"(g01_189), "f"(cl));
                g01_189 = _min_77;
                float _min_78;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_78) : "f"(g10_190), "f"(cl));
                g10_190 = _min_78;
                float _min_79;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_79) : "f"(g11_191), "f"(cl));
                g11_191 = _min_79;
                float _mul_116;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_116) : "f"(lin00_192), "f"(g00_188));
                float value00_196 = _mul_116;
                float _mul_117;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_117) : "f"(lin01_193), "f"(g01_189));
                float value01_197 = _mul_117;
                float _mul_118;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_118) : "f"(lin10_194), "f"(g10_190));
                float value10_198 = _mul_118;
                float _mul_119;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_119) : "f"(lin11_195), "f"(g11_191));
                float value11_199 = _mul_119;
                pair[0] = value00_196;
                pair[1] = value10_198;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token0_178 * 32 + (feature_chunk ^ local_token0_178 % 8) * 4 + feature_word] = word[0];
                pair[0] = value01_197;
                pair[1] = value11_199;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token1_179 * 32 + (feature_chunk ^ local_token1_179 % 8) * 4 + feature_word] = word[0];
                int local_token0_200 = lane_1 % 4 * 2 + 16;
                int local_token1_201 = local_token0_200 + 1;
                float x00_202 = _tmem_load_4[10];
                float x01_203 = _tmem_load_4[11];
                float x10_204 = _tmem_load_5[10];
                float x11_205 = _tmem_load_5[11];
                float _mul_120;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_120) : "f"(x00_202), "f"(-1.4426950408889634f));
                float _exp2_40 = approx_exp2(_mul_120);
                float _add_40;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_40) : "f"(1.0f), "f"(_exp2_40));
                float _rcp_40 = approx_rcp(_add_40);
                float sig00_206 = _rcp_40;
                float _mul_121;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_121) : "f"(x01_203), "f"(-1.4426950408889634f));
                float _exp2_41 = approx_exp2(_mul_121);
                float _add_41;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_41) : "f"(1.0f), "f"(_exp2_41));
                float _rcp_41 = approx_rcp(_add_41);
                float sig01_207 = _rcp_41;
                float _mul_122;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_122) : "f"(x10_204), "f"(-1.4426950408889634f));
                float _exp2_42 = approx_exp2(_mul_122);
                float _add_42;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_42) : "f"(1.0f), "f"(_exp2_42));
                float _rcp_42 = approx_rcp(_add_42);
                float sig10_208 = _rcp_42;
                float _mul_123;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_123) : "f"(x11_205), "f"(-1.4426950408889634f));
                float _exp2_43 = approx_exp2(_mul_123);
                float _add_43;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_43) : "f"(1.0f), "f"(_exp2_43));
                float _rcp_43 = approx_rcp(_add_43);
                float sig11_209 = _rcp_43;
                float _mul_124;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_124) : "f"(x00_202), "f"(sig00_206));
                float g00_210 = _mul_124;
                float _mul_125;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_125) : "f"(x01_203), "f"(sig01_207));
                float g01_211 = _mul_125;
                float _mul_126;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_126) : "f"(x10_204), "f"(sig10_208));
                float g10_212 = _mul_126;
                float _mul_127;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_127) : "f"(x11_205), "f"(sig11_209));
                float g11_213 = _mul_127;
                float lin00_214 = _tmem_load_4[8];
                float lin01_215 = _tmem_load_4[9];
                float lin10_216 = _tmem_load_5[8];
                float lin11_217 = _tmem_load_5[9];
                float _max_40;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_40) : "f"(lin00_214), "f"(neg_cl));
                float _min_80;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_80) : "f"(_max_40), "f"(cl));
                lin00_214 = _min_80;
                float _max_41;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_41) : "f"(lin01_215), "f"(neg_cl));
                float _min_81;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_81) : "f"(_max_41), "f"(cl));
                lin01_215 = _min_81;
                float _max_42;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_42) : "f"(lin10_216), "f"(neg_cl));
                float _min_82;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_82) : "f"(_max_42), "f"(cl));
                lin10_216 = _min_82;
                float _max_43;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_43) : "f"(lin11_217), "f"(neg_cl));
                float _min_83;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_83) : "f"(_max_43), "f"(cl));
                lin11_217 = _min_83;
                float _min_84;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_84) : "f"(g00_210), "f"(cl));
                g00_210 = _min_84;
                float _min_85;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_85) : "f"(g01_211), "f"(cl));
                g01_211 = _min_85;
                float _min_86;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_86) : "f"(g10_212), "f"(cl));
                g10_212 = _min_86;
                float _min_87;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_87) : "f"(g11_213), "f"(cl));
                g11_213 = _min_87;
                float _mul_128;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_128) : "f"(lin00_214), "f"(g00_210));
                float value00_218 = _mul_128;
                float _mul_129;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_129) : "f"(lin01_215), "f"(g01_211));
                float value01_219 = _mul_129;
                float _mul_130;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_130) : "f"(lin10_216), "f"(g10_212));
                float value10_220 = _mul_130;
                float _mul_131;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_131) : "f"(lin11_217), "f"(g11_213));
                float value11_221 = _mul_131;
                pair[0] = value00_218;
                pair[1] = value10_220;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token0_200 * 32 + (feature_chunk ^ local_token0_200 % 8) * 4 + feature_word] = word[0];
                pair[0] = value01_219;
                pair[1] = value11_221;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token1_201 * 32 + (feature_chunk ^ local_token1_201 % 8) * 4 + feature_word] = word[0];
                int local_token0_222 = lane_1 % 4 * 2 + 24;
                int local_token1_223 = local_token0_222 + 1;
                float x00_224 = _tmem_load_4[14];
                float x01_225 = _tmem_load_4[15];
                float x10_226 = _tmem_load_5[14];
                float x11_227 = _tmem_load_5[15];
                float _mul_132;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_132) : "f"(x00_224), "f"(-1.4426950408889634f));
                float _exp2_44 = approx_exp2(_mul_132);
                float _add_44;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_44) : "f"(1.0f), "f"(_exp2_44));
                float _rcp_44 = approx_rcp(_add_44);
                float sig00_228 = _rcp_44;
                float _mul_133;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_133) : "f"(x01_225), "f"(-1.4426950408889634f));
                float _exp2_45 = approx_exp2(_mul_133);
                float _add_45;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_45) : "f"(1.0f), "f"(_exp2_45));
                float _rcp_45 = approx_rcp(_add_45);
                float sig01_229 = _rcp_45;
                float _mul_134;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_134) : "f"(x10_226), "f"(-1.4426950408889634f));
                float _exp2_46 = approx_exp2(_mul_134);
                float _add_46;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_46) : "f"(1.0f), "f"(_exp2_46));
                float _rcp_46 = approx_rcp(_add_46);
                float sig10_230 = _rcp_46;
                float _mul_135;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_135) : "f"(x11_227), "f"(-1.4426950408889634f));
                float _exp2_47 = approx_exp2(_mul_135);
                float _add_47;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_47) : "f"(1.0f), "f"(_exp2_47));
                float _rcp_47 = approx_rcp(_add_47);
                float sig11_231 = _rcp_47;
                float _mul_136;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_136) : "f"(x00_224), "f"(sig00_228));
                float g00_232 = _mul_136;
                float _mul_137;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_137) : "f"(x01_225), "f"(sig01_229));
                float g01_233 = _mul_137;
                float _mul_138;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_138) : "f"(x10_226), "f"(sig10_230));
                float g10_234 = _mul_138;
                float _mul_139;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_139) : "f"(x11_227), "f"(sig11_231));
                float g11_235 = _mul_139;
                float lin00_236 = _tmem_load_4[12];
                float lin01_237 = _tmem_load_4[13];
                float lin10_238 = _tmem_load_5[12];
                float lin11_239 = _tmem_load_5[13];
                float _max_44;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_44) : "f"(lin00_236), "f"(neg_cl));
                float _min_88;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_88) : "f"(_max_44), "f"(cl));
                lin00_236 = _min_88;
                float _max_45;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_45) : "f"(lin01_237), "f"(neg_cl));
                float _min_89;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_89) : "f"(_max_45), "f"(cl));
                lin01_237 = _min_89;
                float _max_46;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_46) : "f"(lin10_238), "f"(neg_cl));
                float _min_90;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_90) : "f"(_max_46), "f"(cl));
                lin10_238 = _min_90;
                float _max_47;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_47) : "f"(lin11_239), "f"(neg_cl));
                float _min_91;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_91) : "f"(_max_47), "f"(cl));
                lin11_239 = _min_91;
                float _min_92;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_92) : "f"(g00_232), "f"(cl));
                g00_232 = _min_92;
                float _min_93;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_93) : "f"(g01_233), "f"(cl));
                g01_233 = _min_93;
                float _min_94;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_94) : "f"(g10_234), "f"(cl));
                g10_234 = _min_94;
                float _min_95;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_95) : "f"(g11_235), "f"(cl));
                g11_235 = _min_95;
                float _mul_140;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_140) : "f"(lin00_236), "f"(g00_232));
                float value00_240 = _mul_140;
                float _mul_141;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_141) : "f"(lin01_237), "f"(g01_233));
                float value01_241 = _mul_141;
                float _mul_142;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_142) : "f"(lin10_238), "f"(g10_234));
                float value10_242 = _mul_142;
                float _mul_143;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_143) : "f"(lin11_239), "f"(g11_235));
                float value11_243 = _mul_143;
                pair[0] = value00_240;
                pair[1] = value10_242;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token0_222 * 32 + (feature_chunk ^ local_token0_222 % 8) * 4 + feature_word] = word[0];
                pair[0] = value01_241;
                pair[1] = value11_243;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token1_223 * 32 + (feature_chunk ^ local_token1_223 % 8) * 4 + feature_word] = word[0];
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 7, 128;" ::: "memory");
                if (warp == 0) {
                    if (elect_sync()) {
                        int padding_rows_2 = (128 - valid_rows % 128) % 128;
                        tma_store_4d((&C), m_tile * 64, padding_rows_2 + 64, 1073741824, n_tile * 128 - (unsigned int)padding_rows_2 + 1073741824, epi_staging_addr);
                    }
                }
                asm volatile("cp.async.bulk.commit_group;");
                asm volatile("barrier.sync 7, 128;" ::: "memory");
                int acc_offset_244 = acc_stage * 128 + 96;
                float _tmem_load_6[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[15]))
                    : "r"(taddr + (unsigned int)acc_offset_244));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float _tmem_load_7[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[15]))
                    : "r"(taddr + 1048576 + (unsigned int)acc_offset_244));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                asm volatile("cp.async.bulk.wait_group.read 0;");
                asm volatile("barrier.sync 7, 128;" ::: "memory");
                int local_token0_245 = lane_1 % 4 * 2;
                int local_token1_246 = local_token0_245 + 1;
                float x00_247 = _tmem_load_6[2];
                float x01_248 = _tmem_load_6[3];
                float x10_249 = _tmem_load_7[2];
                float x11_250 = _tmem_load_7[3];
                float _mul_144;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_144) : "f"(x00_247), "f"(-1.4426950408889634f));
                float _exp2_48 = approx_exp2(_mul_144);
                float _add_48;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_48) : "f"(1.0f), "f"(_exp2_48));
                float _rcp_48 = approx_rcp(_add_48);
                float sig00_251 = _rcp_48;
                float _mul_145;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_145) : "f"(x01_248), "f"(-1.4426950408889634f));
                float _exp2_49 = approx_exp2(_mul_145);
                float _add_49;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_49) : "f"(1.0f), "f"(_exp2_49));
                float _rcp_49 = approx_rcp(_add_49);
                float sig01_252 = _rcp_49;
                float _mul_146;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_146) : "f"(x10_249), "f"(-1.4426950408889634f));
                float _exp2_50 = approx_exp2(_mul_146);
                float _add_50;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_50) : "f"(1.0f), "f"(_exp2_50));
                float _rcp_50 = approx_rcp(_add_50);
                float sig10_253 = _rcp_50;
                float _mul_147;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_147) : "f"(x11_250), "f"(-1.4426950408889634f));
                float _exp2_51 = approx_exp2(_mul_147);
                float _add_51;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_51) : "f"(1.0f), "f"(_exp2_51));
                float _rcp_51 = approx_rcp(_add_51);
                float sig11_254 = _rcp_51;
                float _mul_148;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_148) : "f"(x00_247), "f"(sig00_251));
                float g00_255 = _mul_148;
                float _mul_149;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_149) : "f"(x01_248), "f"(sig01_252));
                float g01_256 = _mul_149;
                float _mul_150;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_150) : "f"(x10_249), "f"(sig10_253));
                float g10_257 = _mul_150;
                float _mul_151;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_151) : "f"(x11_250), "f"(sig11_254));
                float g11_258 = _mul_151;
                float lin00_259 = _tmem_load_6[0];
                float lin01_260 = _tmem_load_6[1];
                float lin10_261 = _tmem_load_7[0];
                float lin11_262 = _tmem_load_7[1];
                float _max_48;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_48) : "f"(lin00_259), "f"(neg_cl));
                float _min_96;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_96) : "f"(_max_48), "f"(cl));
                lin00_259 = _min_96;
                float _max_49;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_49) : "f"(lin01_260), "f"(neg_cl));
                float _min_97;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_97) : "f"(_max_49), "f"(cl));
                lin01_260 = _min_97;
                float _max_50;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_50) : "f"(lin10_261), "f"(neg_cl));
                float _min_98;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_98) : "f"(_max_50), "f"(cl));
                lin10_261 = _min_98;
                float _max_51;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_51) : "f"(lin11_262), "f"(neg_cl));
                float _min_99;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_99) : "f"(_max_51), "f"(cl));
                lin11_262 = _min_99;
                float _min_100;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_100) : "f"(g00_255), "f"(cl));
                g00_255 = _min_100;
                float _min_101;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_101) : "f"(g01_256), "f"(cl));
                g01_256 = _min_101;
                float _min_102;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_102) : "f"(g10_257), "f"(cl));
                g10_257 = _min_102;
                float _min_103;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_103) : "f"(g11_258), "f"(cl));
                g11_258 = _min_103;
                float _mul_152;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_152) : "f"(lin00_259), "f"(g00_255));
                float value00_263 = _mul_152;
                float _mul_153;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_153) : "f"(lin01_260), "f"(g01_256));
                float value01_264 = _mul_153;
                float _mul_154;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_154) : "f"(lin10_261), "f"(g10_257));
                float value10_265 = _mul_154;
                float _mul_155;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_155) : "f"(lin11_262), "f"(g11_258));
                float value11_266 = _mul_155;
                pair[0] = value00_263;
                pair[1] = value10_265;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token0_245 * 32 + (feature_chunk ^ local_token0_245 % 8) * 4 + feature_word] = word[0];
                pair[0] = value01_264;
                pair[1] = value11_266;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token1_246 * 32 + (feature_chunk ^ local_token1_246 % 8) * 4 + feature_word] = word[0];
                int local_token0_267 = lane_1 % 4 * 2 + 8;
                int local_token1_268 = local_token0_267 + 1;
                float x00_269 = _tmem_load_6[6];
                float x01_270 = _tmem_load_6[7];
                float x10_271 = _tmem_load_7[6];
                float x11_272 = _tmem_load_7[7];
                float _mul_156;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_156) : "f"(x00_269), "f"(-1.4426950408889634f));
                float _exp2_52 = approx_exp2(_mul_156);
                float _add_52;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_52) : "f"(1.0f), "f"(_exp2_52));
                float _rcp_52 = approx_rcp(_add_52);
                float sig00_273 = _rcp_52;
                float _mul_157;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_157) : "f"(x01_270), "f"(-1.4426950408889634f));
                float _exp2_53 = approx_exp2(_mul_157);
                float _add_53;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_53) : "f"(1.0f), "f"(_exp2_53));
                float _rcp_53 = approx_rcp(_add_53);
                float sig01_274 = _rcp_53;
                float _mul_158;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_158) : "f"(x10_271), "f"(-1.4426950408889634f));
                float _exp2_54 = approx_exp2(_mul_158);
                float _add_54;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_54) : "f"(1.0f), "f"(_exp2_54));
                float _rcp_54 = approx_rcp(_add_54);
                float sig10_275 = _rcp_54;
                float _mul_159;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_159) : "f"(x11_272), "f"(-1.4426950408889634f));
                float _exp2_55 = approx_exp2(_mul_159);
                float _add_55;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_55) : "f"(1.0f), "f"(_exp2_55));
                float _rcp_55 = approx_rcp(_add_55);
                float sig11_276 = _rcp_55;
                float _mul_160;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_160) : "f"(x00_269), "f"(sig00_273));
                float g00_277 = _mul_160;
                float _mul_161;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_161) : "f"(x01_270), "f"(sig01_274));
                float g01_278 = _mul_161;
                float _mul_162;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_162) : "f"(x10_271), "f"(sig10_275));
                float g10_279 = _mul_162;
                float _mul_163;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_163) : "f"(x11_272), "f"(sig11_276));
                float g11_280 = _mul_163;
                float lin00_281 = _tmem_load_6[4];
                float lin01_282 = _tmem_load_6[5];
                float lin10_283 = _tmem_load_7[4];
                float lin11_284 = _tmem_load_7[5];
                float _max_52;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_52) : "f"(lin00_281), "f"(neg_cl));
                float _min_104;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_104) : "f"(_max_52), "f"(cl));
                lin00_281 = _min_104;
                float _max_53;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_53) : "f"(lin01_282), "f"(neg_cl));
                float _min_105;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_105) : "f"(_max_53), "f"(cl));
                lin01_282 = _min_105;
                float _max_54;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_54) : "f"(lin10_283), "f"(neg_cl));
                float _min_106;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_106) : "f"(_max_54), "f"(cl));
                lin10_283 = _min_106;
                float _max_55;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_55) : "f"(lin11_284), "f"(neg_cl));
                float _min_107;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_107) : "f"(_max_55), "f"(cl));
                lin11_284 = _min_107;
                float _min_108;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_108) : "f"(g00_277), "f"(cl));
                g00_277 = _min_108;
                float _min_109;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_109) : "f"(g01_278), "f"(cl));
                g01_278 = _min_109;
                float _min_110;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_110) : "f"(g10_279), "f"(cl));
                g10_279 = _min_110;
                float _min_111;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_111) : "f"(g11_280), "f"(cl));
                g11_280 = _min_111;
                float _mul_164;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_164) : "f"(lin00_281), "f"(g00_277));
                float value00_285 = _mul_164;
                float _mul_165;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_165) : "f"(lin01_282), "f"(g01_278));
                float value01_286 = _mul_165;
                float _mul_166;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_166) : "f"(lin10_283), "f"(g10_279));
                float value10_287 = _mul_166;
                float _mul_167;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_167) : "f"(lin11_284), "f"(g11_280));
                float value11_288 = _mul_167;
                pair[0] = value00_285;
                pair[1] = value10_287;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token0_267 * 32 + (feature_chunk ^ local_token0_267 % 8) * 4 + feature_word] = word[0];
                pair[0] = value01_286;
                pair[1] = value11_288;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token1_268 * 32 + (feature_chunk ^ local_token1_268 % 8) * 4 + feature_word] = word[0];
                int local_token0_289 = lane_1 % 4 * 2 + 16;
                int local_token1_290 = local_token0_289 + 1;
                float x00_291 = _tmem_load_6[10];
                float x01_292 = _tmem_load_6[11];
                float x10_293 = _tmem_load_7[10];
                float x11_294 = _tmem_load_7[11];
                float _mul_168;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_168) : "f"(x00_291), "f"(-1.4426950408889634f));
                float _exp2_56 = approx_exp2(_mul_168);
                float _add_56;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_56) : "f"(1.0f), "f"(_exp2_56));
                float _rcp_56 = approx_rcp(_add_56);
                float sig00_295 = _rcp_56;
                float _mul_169;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_169) : "f"(x01_292), "f"(-1.4426950408889634f));
                float _exp2_57 = approx_exp2(_mul_169);
                float _add_57;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_57) : "f"(1.0f), "f"(_exp2_57));
                float _rcp_57 = approx_rcp(_add_57);
                float sig01_296 = _rcp_57;
                float _mul_170;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_170) : "f"(x10_293), "f"(-1.4426950408889634f));
                float _exp2_58 = approx_exp2(_mul_170);
                float _add_58;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_58) : "f"(1.0f), "f"(_exp2_58));
                float _rcp_58 = approx_rcp(_add_58);
                float sig10_297 = _rcp_58;
                float _mul_171;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_171) : "f"(x11_294), "f"(-1.4426950408889634f));
                float _exp2_59 = approx_exp2(_mul_171);
                float _add_59;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_59) : "f"(1.0f), "f"(_exp2_59));
                float _rcp_59 = approx_rcp(_add_59);
                float sig11_298 = _rcp_59;
                float _mul_172;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_172) : "f"(x00_291), "f"(sig00_295));
                float g00_299 = _mul_172;
                float _mul_173;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_173) : "f"(x01_292), "f"(sig01_296));
                float g01_300 = _mul_173;
                float _mul_174;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_174) : "f"(x10_293), "f"(sig10_297));
                float g10_301 = _mul_174;
                float _mul_175;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_175) : "f"(x11_294), "f"(sig11_298));
                float g11_302 = _mul_175;
                float lin00_303 = _tmem_load_6[8];
                float lin01_304 = _tmem_load_6[9];
                float lin10_305 = _tmem_load_7[8];
                float lin11_306 = _tmem_load_7[9];
                float _max_56;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_56) : "f"(lin00_303), "f"(neg_cl));
                float _min_112;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_112) : "f"(_max_56), "f"(cl));
                lin00_303 = _min_112;
                float _max_57;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_57) : "f"(lin01_304), "f"(neg_cl));
                float _min_113;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_113) : "f"(_max_57), "f"(cl));
                lin01_304 = _min_113;
                float _max_58;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_58) : "f"(lin10_305), "f"(neg_cl));
                float _min_114;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_114) : "f"(_max_58), "f"(cl));
                lin10_305 = _min_114;
                float _max_59;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_59) : "f"(lin11_306), "f"(neg_cl));
                float _min_115;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_115) : "f"(_max_59), "f"(cl));
                lin11_306 = _min_115;
                float _min_116;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_116) : "f"(g00_299), "f"(cl));
                g00_299 = _min_116;
                float _min_117;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_117) : "f"(g01_300), "f"(cl));
                g01_300 = _min_117;
                float _min_118;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_118) : "f"(g10_301), "f"(cl));
                g10_301 = _min_118;
                float _min_119;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_119) : "f"(g11_302), "f"(cl));
                g11_302 = _min_119;
                float _mul_176;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_176) : "f"(lin00_303), "f"(g00_299));
                float value00_307 = _mul_176;
                float _mul_177;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_177) : "f"(lin01_304), "f"(g01_300));
                float value01_308 = _mul_177;
                float _mul_178;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_178) : "f"(lin10_305), "f"(g10_301));
                float value10_309 = _mul_178;
                float _mul_179;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_179) : "f"(lin11_306), "f"(g11_302));
                float value11_310 = _mul_179;
                pair[0] = value00_307;
                pair[1] = value10_309;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token0_289 * 32 + (feature_chunk ^ local_token0_289 % 8) * 4 + feature_word] = word[0];
                pair[0] = value01_308;
                pair[1] = value11_310;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token1_290 * 32 + (feature_chunk ^ local_token1_290 % 8) * 4 + feature_word] = word[0];
                int local_token0_311 = lane_1 % 4 * 2 + 24;
                int local_token1_312 = local_token0_311 + 1;
                float x00_313 = _tmem_load_6[14];
                float x01_314 = _tmem_load_6[15];
                float x10_315 = _tmem_load_7[14];
                float x11_316 = _tmem_load_7[15];
                float _mul_180;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_180) : "f"(x00_313), "f"(-1.4426950408889634f));
                float _exp2_60 = approx_exp2(_mul_180);
                float _add_60;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_60) : "f"(1.0f), "f"(_exp2_60));
                float _rcp_60 = approx_rcp(_add_60);
                float sig00_317 = _rcp_60;
                float _mul_181;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_181) : "f"(x01_314), "f"(-1.4426950408889634f));
                float _exp2_61 = approx_exp2(_mul_181);
                float _add_61;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_61) : "f"(1.0f), "f"(_exp2_61));
                float _rcp_61 = approx_rcp(_add_61);
                float sig01_318 = _rcp_61;
                float _mul_182;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_182) : "f"(x10_315), "f"(-1.4426950408889634f));
                float _exp2_62 = approx_exp2(_mul_182);
                float _add_62;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_62) : "f"(1.0f), "f"(_exp2_62));
                float _rcp_62 = approx_rcp(_add_62);
                float sig10_319 = _rcp_62;
                float _mul_183;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_183) : "f"(x11_316), "f"(-1.4426950408889634f));
                float _exp2_63 = approx_exp2(_mul_183);
                float _add_63;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_63) : "f"(1.0f), "f"(_exp2_63));
                float _rcp_63 = approx_rcp(_add_63);
                float sig11_320 = _rcp_63;
                float _mul_184;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_184) : "f"(x00_313), "f"(sig00_317));
                float g00_321 = _mul_184;
                float _mul_185;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_185) : "f"(x01_314), "f"(sig01_318));
                float g01_322 = _mul_185;
                float _mul_186;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_186) : "f"(x10_315), "f"(sig10_319));
                float g10_323 = _mul_186;
                float _mul_187;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_187) : "f"(x11_316), "f"(sig11_320));
                float g11_324 = _mul_187;
                float lin00_325 = _tmem_load_6[12];
                float lin01_326 = _tmem_load_6[13];
                float lin10_327 = _tmem_load_7[12];
                float lin11_328 = _tmem_load_7[13];
                float _max_60;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_60) : "f"(lin00_325), "f"(neg_cl));
                float _min_120;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_120) : "f"(_max_60), "f"(cl));
                lin00_325 = _min_120;
                float _max_61;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_61) : "f"(lin01_326), "f"(neg_cl));
                float _min_121;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_121) : "f"(_max_61), "f"(cl));
                lin01_326 = _min_121;
                float _max_62;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_62) : "f"(lin10_327), "f"(neg_cl));
                float _min_122;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_122) : "f"(_max_62), "f"(cl));
                lin10_327 = _min_122;
                float _max_63;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_63) : "f"(lin11_328), "f"(neg_cl));
                float _min_123;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_123) : "f"(_max_63), "f"(cl));
                lin11_328 = _min_123;
                float _min_124;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_124) : "f"(g00_321), "f"(cl));
                g00_321 = _min_124;
                float _min_125;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_125) : "f"(g01_322), "f"(cl));
                g01_322 = _min_125;
                float _min_126;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_126) : "f"(g10_323), "f"(cl));
                g10_323 = _min_126;
                float _min_127;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_127) : "f"(g11_324), "f"(cl));
                g11_324 = _min_127;
                float _mul_188;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_188) : "f"(lin00_325), "f"(g00_321));
                float value00_329 = _mul_188;
                float _mul_189;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_189) : "f"(lin01_326), "f"(g01_322));
                float value01_330 = _mul_189;
                float _mul_190;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_190) : "f"(lin10_327), "f"(g10_323));
                float value10_331 = _mul_190;
                float _mul_191;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_191) : "f"(lin11_328), "f"(g11_324));
                float value11_332 = _mul_191;
                pair[0] = value00_329;
                pair[1] = value10_331;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token0_311 * 32 + (feature_chunk ^ local_token0_311 % 8) * 4 + feature_word] = word[0];
                pair[0] = value01_330;
                pair[1] = value11_332;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token1_312 * 32 + (feature_chunk ^ local_token1_312 % 8) * 4 + feature_word] = word[0];
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 7, 128;" ::: "memory");
                if (warp == 0) {
                    if (elect_sync()) {
                        int padding_rows_3 = (128 - valid_rows % 128) % 128;
                        tma_store_4d((&C), m_tile * 64, padding_rows_3 + 96, 1073741824, n_tile * 128 - (unsigned int)padding_rows_3 + 1073741824, epi_staging_addr);
                    }
                }
                asm volatile("cp.async.bulk.commit_group;");
                asm volatile("barrier.sync 7, 128;" ::: "memory");
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
                unsigned int in_bound = ((_clc_ctaid_7 < (unsigned int)num_non_exiting_ctas[0]) ? 1 : 0);
                unsigned int ok = _clc_valid_3 * in_bound;
                if (ok == 0) {
                    break;
                }
                m_tile = _clc_ctaid_6 + (unsigned int)cta_rank;
                n_tile = _clc_ctaid_7;
            }
        }
    }
    // ---- Role: load_b ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
        { // load_b_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int stage = 0;
            unsigned int work_stage_1 = 0;
            unsigned int m_tile_1 = blockIdx.x;
            unsigned int n_tile_1 = blockIdx.y;
            int warp_local = warp - 4;
            int route_base = 0;
            int routed[16];
            unsigned int cta_mask = 1 << cta_rank;
            unsigned int _phase_k_done = 1;
            unsigned int _phase_work_full_1 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_1 = 0; _tile_iter_1 < grid_m / 2 * grid_n; _tile_iter_1++) {
                if (m_tile_1 >= (unsigned int)grid_m || n_tile_1 >= (unsigned int)num_non_exiting_ctas[0]) {
                    break;
                }
                route_base = n_tile_1 * 128 + (unsigned int)(cta_rank * 64) + (unsigned int)(warp_local * 4);
                for (int row = 0; row < 4; row++) {
                    routed[row] = route_map[route_base + row];
                }
                route_base = n_tile_1 * 128 + (unsigned int)(cta_rank * 64) + (unsigned int)((4 + warp_local) * 4);
                for (int row_1 = 0; row_1 < 4; row_1++) {
                    routed[4 + row_1] = route_map[route_base + row_1];
                }
                route_base = n_tile_1 * 128 + (unsigned int)(cta_rank * 64) + (unsigned int)((8 + warp_local) * 4);
                for (int row_2 = 0; row_2 < 4; row_2++) {
                    routed[8 + row_2] = route_map[route_base + row_2];
                }
                route_base = n_tile_1 * 128 + (unsigned int)(cta_rank * 64) + (unsigned int)((12 + warp_local) * 4);
                for (int row_3 = 0; row_3 < 4; row_3++) {
                    routed[12 + row_3] = route_map[route_base + row_3];
                }
                #pragma unroll 1
                for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                    mbarrier_wait(k_done_addr + (stage) * 8, _phase_k_done);
                    if (elect_sync()) {
                        tma_gather4_gmem2smem_mc_cta2(smem_b_addr + stage * 8192 + (unsigned int)(warp_local * 512), (&B), iter_k * 64, routed[0], routed[1], routed[2], routed[3], ((b_full_addr + (stage) * 8) & 0xFEFFFFFF), cta_mask);
                        tma_gather4_gmem2smem_mc_cta2(smem_b_addr + stage * 8192 + (unsigned int)((4 + warp_local) * 512), (&B), iter_k * 64, routed[4], routed[5], routed[6], routed[7], ((b_full_addr + (stage) * 8) & 0xFEFFFFFF), cta_mask);
                        tma_gather4_gmem2smem_mc_cta2(smem_b_addr + stage * 8192 + (unsigned int)((8 + warp_local) * 512), (&B), iter_k * 64, routed[8], routed[9], routed[10], routed[11], ((b_full_addr + (stage) * 8) & 0xFEFFFFFF), cta_mask);
                        tma_gather4_gmem2smem_mc_cta2(smem_b_addr + stage * 8192 + (unsigned int)((12 + warp_local) * 512), (&B), iter_k * 64, routed[12], routed[13], routed[14], routed[15], ((b_full_addr + (stage) * 8) & 0xFEFFFFFF), cta_mask);
                    }
                    if (warp == 4) {
                        if (elect_sync()) {
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((b_full_addr + (stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(8192)) : "memory");
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
                unsigned int in_bound_1 = ((_clc_ctaid_3 < (unsigned int)num_non_exiting_ctas[0]) ? 1 : 0);
                unsigned int ok_1 = _clc_valid_1 * in_bound_1;
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
    if (warp == 8) {
        { // load_a_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int stage_1 = 0;
            unsigned int work_stage_2 = 0;
            unsigned int throttle_stage = 0;
            unsigned int m_tile_2 = blockIdx.x;
            unsigned int n_tile_2 = blockIdx.y;
            unsigned int cta_mask_1 = 1 << cta_rank;
            unsigned int _phase_throttle_empty = 1;
            unsigned int _phase_k_done_1 = 1;
            unsigned int _phase_work_full_2 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_2 = 0; _tile_iter_2 < grid_m / 2 * grid_n; _tile_iter_2++) {
                if (m_tile_2 >= (unsigned int)grid_m || n_tile_2 >= (unsigned int)num_non_exiting_ctas[0]) {
                    break;
                }
                int expert_1 = tile_expert[n_tile_2];
                int sole_reader = 1;
                if (n_tile_2 + 1 < (unsigned int)num_non_exiting_ctas[0]) {
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
                unsigned int in_bound_2 = ((_clc_ctaid_1 < (unsigned int)num_non_exiting_ctas[0]) ? 1 : 0);
                unsigned int ok_2 = _clc_valid_0 * in_bound_2;
                if (ok_2 == 0) {
                    break;
                }
                m_tile_2 = _clc_ctaid_0 + (unsigned int)cta_rank;
                n_tile_2 = _clc_ctaid_1;
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 9) {
        { // mma_main
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
                #pragma unroll 1
                for (unsigned int _tile_iter_3 = 0; _tile_iter_3 < grid_m / 2 * grid_n; _tile_iter_3++) {
                    if (m_tile_3 >= (unsigned int)grid_m || n_tile_3 >= (unsigned int)num_non_exiting_ctas[0]) {
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
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 270533776;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_accum + (acc_stage_1 * 128))), "r"(((((iter_k_2 == 0) ? 1 : 0)) ? 0 : 1)));
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
                    unsigned int in_bound_3 = ((_clc_ctaid_5 < (unsigned int)num_non_exiting_ctas[0]) ? 1 : 0);
                    unsigned int ok_3 = _clc_valid_2 * in_bound_3;
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
    if (warp == 10) {
        { // work_id_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int work_stage_4 = 0;
            unsigned int throttle_stage_1 = 0;
            unsigned int _phase_throttle_full = 0;
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_work_full_4 = 0;
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
                    unsigned int in_bound_4 = ((_clc_ctaid_9 < (unsigned int)num_non_exiting_ctas[0]) ? 1 : 0);
                    unsigned int ok_4 = _clc_valid_4 * in_bound_4;
                    if (ok_4 == 0) {
                        break;
                    }
                }
            }
        }
    }
    // ---- Role: padding ----
    if (warp == 11) {
        { // padding_main
            unsigned int work_stage_5 = 0;
            unsigned int m_tile_4 = blockIdx.x;
            unsigned int n_tile_4 = blockIdx.y;
            unsigned int _phase_work_full_5 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_5 = 0; _tile_iter_5 < grid_m / 2 * grid_n; _tile_iter_5++) {
                if (m_tile_4 >= (unsigned int)grid_m || n_tile_4 >= (unsigned int)num_non_exiting_ctas[0]) {
                    break;
                }
                mbarrier_wait(work_full_addr + (work_stage_5) * 8, _phase_work_full_5);
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
                unsigned int in_bound_5 = ((_clc_ctaid_11 < (unsigned int)num_non_exiting_ctas[0]) ? 1 : 0);
                unsigned int ok_5 = _clc_valid_5 * in_bound_5;
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
