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
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 16384
#define SMEM_SMEM_B_OFF 99328
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 16384
#define SMEM_EPI_STAGING_OFF 197632
#define SMEM_EPI_STAGING_STAGE_BYTES 4096
#define SMEM_EPI_STAGING_STRIDE 4096
#define SMEM_EPI_STAGING_U32_OFF 197632
#define SMEM_EPI_STAGING_U32_STAGE_BYTES 4096
#define SMEM_EPI_STAGING_U32_STRIDE 4096
#define SMEM_WORK_RESPONSE_OFF 201728
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_TOTAL 201856
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
kernel_cake_stepfun_moe_1e1034c2888d379ca0c7(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap C, int* __restrict__ route_map, int* __restrict__ tile_expert, int* __restrict__ tile_mn_limit, int* __restrict__ num_non_exiting_ctas, int M_out, int K, int grid_m, int grid_n, int K_tiles, float* __restrict__ clamp_limit)
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
    __nv_bfloat16* epi_staging = reinterpret_cast<__nv_bfloat16*>(smem_raw + 197632);
    const int epi_staging_addr = smem + 197632;
    unsigned int* epi_staging_u32 = reinterpret_cast<unsigned int*>(smem_raw + 197632);
    const int epi_staging_u32_addr = smem + 197632;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 201728);
    const int work_response_addr = smem + 201728;
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

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 272);
    if (warp == 0) {
        int _tmem_hold = smem + 272;
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
                int valid_rows = (unsigned int)tile_mn_limit[n_tile] - n_tile * 256;
                float cl = clamp_limit[expert];
                float _sub_0;
                asm("sub.rn.ftz.f32 %0, %1, %2;" : "=f"(_sub_0) : "f"(0.0f), "f"(cl));
                float neg_cl = _sub_0;
                mbarrier_wait(mma_full_addr + (acc_stage) * 8, _phase_mma_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int acc_offset = acc_stage * 256;
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
                        int padding_rows = (256 - valid_rows % 256) % 256;
                        tma_store_4d((&C), m_tile * 64, padding_rows, 1073741824, n_tile * 256 - (unsigned int)padding_rows + 1073741824, epi_staging_addr);
                    }
                }
                asm volatile("cp.async.bulk.commit_group;");
                asm volatile("barrier.sync 7, 128;" ::: "memory");
                int acc_offset_66 = acc_stage * 256 + 32;
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
                        int padding_rows_1 = (256 - valid_rows % 256) % 256;
                        tma_store_4d((&C), m_tile * 64, padding_rows_1 + 32, 1073741824, n_tile * 256 - (unsigned int)padding_rows_1 + 1073741824, epi_staging_addr);
                    }
                }
                asm volatile("cp.async.bulk.commit_group;");
                asm volatile("barrier.sync 7, 128;" ::: "memory");
                int acc_offset_155 = acc_stage * 256 + 64;
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
                        int padding_rows_2 = (256 - valid_rows % 256) % 256;
                        tma_store_4d((&C), m_tile * 64, padding_rows_2 + 64, 1073741824, n_tile * 256 - (unsigned int)padding_rows_2 + 1073741824, epi_staging_addr);
                    }
                }
                asm volatile("cp.async.bulk.commit_group;");
                asm volatile("barrier.sync 7, 128;" ::: "memory");
                int acc_offset_244 = acc_stage * 256 + 96;
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
                        int padding_rows_3 = (256 - valid_rows % 256) % 256;
                        tma_store_4d((&C), m_tile * 64, padding_rows_3 + 96, 1073741824, n_tile * 256 - (unsigned int)padding_rows_3 + 1073741824, epi_staging_addr);
                    }
                }
                asm volatile("cp.async.bulk.commit_group;");
                asm volatile("barrier.sync 7, 128;" ::: "memory");
                int acc_offset_333 = acc_stage * 256 + 128;
                float _tmem_load_8[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_8[15]))
                    : "r"(taddr + (unsigned int)acc_offset_333));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float _tmem_load_9[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_9[15]))
                    : "r"(taddr + 1048576 + (unsigned int)acc_offset_333));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                asm volatile("cp.async.bulk.wait_group.read 0;");
                asm volatile("barrier.sync 7, 128;" ::: "memory");
                int local_token0_334 = lane_1 % 4 * 2;
                int local_token1_335 = local_token0_334 + 1;
                float x00_336 = _tmem_load_8[2];
                float x01_337 = _tmem_load_8[3];
                float x10_338 = _tmem_load_9[2];
                float x11_339 = _tmem_load_9[3];
                float _mul_192;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_192) : "f"(x00_336), "f"(-1.4426950408889634f));
                float _exp2_64 = approx_exp2(_mul_192);
                float _add_64;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_64) : "f"(1.0f), "f"(_exp2_64));
                float _rcp_64 = approx_rcp(_add_64);
                float sig00_340 = _rcp_64;
                float _mul_193;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_193) : "f"(x01_337), "f"(-1.4426950408889634f));
                float _exp2_65 = approx_exp2(_mul_193);
                float _add_65;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_65) : "f"(1.0f), "f"(_exp2_65));
                float _rcp_65 = approx_rcp(_add_65);
                float sig01_341 = _rcp_65;
                float _mul_194;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_194) : "f"(x10_338), "f"(-1.4426950408889634f));
                float _exp2_66 = approx_exp2(_mul_194);
                float _add_66;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_66) : "f"(1.0f), "f"(_exp2_66));
                float _rcp_66 = approx_rcp(_add_66);
                float sig10_342 = _rcp_66;
                float _mul_195;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_195) : "f"(x11_339), "f"(-1.4426950408889634f));
                float _exp2_67 = approx_exp2(_mul_195);
                float _add_67;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_67) : "f"(1.0f), "f"(_exp2_67));
                float _rcp_67 = approx_rcp(_add_67);
                float sig11_343 = _rcp_67;
                float _mul_196;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_196) : "f"(x00_336), "f"(sig00_340));
                float g00_344 = _mul_196;
                float _mul_197;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_197) : "f"(x01_337), "f"(sig01_341));
                float g01_345 = _mul_197;
                float _mul_198;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_198) : "f"(x10_338), "f"(sig10_342));
                float g10_346 = _mul_198;
                float _mul_199;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_199) : "f"(x11_339), "f"(sig11_343));
                float g11_347 = _mul_199;
                float lin00_348 = _tmem_load_8[0];
                float lin01_349 = _tmem_load_8[1];
                float lin10_350 = _tmem_load_9[0];
                float lin11_351 = _tmem_load_9[1];
                float _max_64;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_64) : "f"(lin00_348), "f"(neg_cl));
                float _min_128;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_128) : "f"(_max_64), "f"(cl));
                lin00_348 = _min_128;
                float _max_65;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_65) : "f"(lin01_349), "f"(neg_cl));
                float _min_129;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_129) : "f"(_max_65), "f"(cl));
                lin01_349 = _min_129;
                float _max_66;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_66) : "f"(lin10_350), "f"(neg_cl));
                float _min_130;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_130) : "f"(_max_66), "f"(cl));
                lin10_350 = _min_130;
                float _max_67;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_67) : "f"(lin11_351), "f"(neg_cl));
                float _min_131;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_131) : "f"(_max_67), "f"(cl));
                lin11_351 = _min_131;
                float _min_132;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_132) : "f"(g00_344), "f"(cl));
                g00_344 = _min_132;
                float _min_133;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_133) : "f"(g01_345), "f"(cl));
                g01_345 = _min_133;
                float _min_134;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_134) : "f"(g10_346), "f"(cl));
                g10_346 = _min_134;
                float _min_135;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_135) : "f"(g11_347), "f"(cl));
                g11_347 = _min_135;
                float _mul_200;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_200) : "f"(lin00_348), "f"(g00_344));
                float value00_352 = _mul_200;
                float _mul_201;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_201) : "f"(lin01_349), "f"(g01_345));
                float value01_353 = _mul_201;
                float _mul_202;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_202) : "f"(lin10_350), "f"(g10_346));
                float value10_354 = _mul_202;
                float _mul_203;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_203) : "f"(lin11_351), "f"(g11_347));
                float value11_355 = _mul_203;
                pair[0] = value00_352;
                pair[1] = value10_354;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token0_334 * 32 + (feature_chunk ^ local_token0_334 % 8) * 4 + feature_word] = word[0];
                pair[0] = value01_353;
                pair[1] = value11_355;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token1_335 * 32 + (feature_chunk ^ local_token1_335 % 8) * 4 + feature_word] = word[0];
                int local_token0_356 = lane_1 % 4 * 2 + 8;
                int local_token1_357 = local_token0_356 + 1;
                float x00_358 = _tmem_load_8[6];
                float x01_359 = _tmem_load_8[7];
                float x10_360 = _tmem_load_9[6];
                float x11_361 = _tmem_load_9[7];
                float _mul_204;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_204) : "f"(x00_358), "f"(-1.4426950408889634f));
                float _exp2_68 = approx_exp2(_mul_204);
                float _add_68;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_68) : "f"(1.0f), "f"(_exp2_68));
                float _rcp_68 = approx_rcp(_add_68);
                float sig00_362 = _rcp_68;
                float _mul_205;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_205) : "f"(x01_359), "f"(-1.4426950408889634f));
                float _exp2_69 = approx_exp2(_mul_205);
                float _add_69;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_69) : "f"(1.0f), "f"(_exp2_69));
                float _rcp_69 = approx_rcp(_add_69);
                float sig01_363 = _rcp_69;
                float _mul_206;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_206) : "f"(x10_360), "f"(-1.4426950408889634f));
                float _exp2_70 = approx_exp2(_mul_206);
                float _add_70;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_70) : "f"(1.0f), "f"(_exp2_70));
                float _rcp_70 = approx_rcp(_add_70);
                float sig10_364 = _rcp_70;
                float _mul_207;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_207) : "f"(x11_361), "f"(-1.4426950408889634f));
                float _exp2_71 = approx_exp2(_mul_207);
                float _add_71;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_71) : "f"(1.0f), "f"(_exp2_71));
                float _rcp_71 = approx_rcp(_add_71);
                float sig11_365 = _rcp_71;
                float _mul_208;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_208) : "f"(x00_358), "f"(sig00_362));
                float g00_366 = _mul_208;
                float _mul_209;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_209) : "f"(x01_359), "f"(sig01_363));
                float g01_367 = _mul_209;
                float _mul_210;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_210) : "f"(x10_360), "f"(sig10_364));
                float g10_368 = _mul_210;
                float _mul_211;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_211) : "f"(x11_361), "f"(sig11_365));
                float g11_369 = _mul_211;
                float lin00_370 = _tmem_load_8[4];
                float lin01_371 = _tmem_load_8[5];
                float lin10_372 = _tmem_load_9[4];
                float lin11_373 = _tmem_load_9[5];
                float _max_68;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_68) : "f"(lin00_370), "f"(neg_cl));
                float _min_136;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_136) : "f"(_max_68), "f"(cl));
                lin00_370 = _min_136;
                float _max_69;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_69) : "f"(lin01_371), "f"(neg_cl));
                float _min_137;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_137) : "f"(_max_69), "f"(cl));
                lin01_371 = _min_137;
                float _max_70;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_70) : "f"(lin10_372), "f"(neg_cl));
                float _min_138;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_138) : "f"(_max_70), "f"(cl));
                lin10_372 = _min_138;
                float _max_71;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_71) : "f"(lin11_373), "f"(neg_cl));
                float _min_139;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_139) : "f"(_max_71), "f"(cl));
                lin11_373 = _min_139;
                float _min_140;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_140) : "f"(g00_366), "f"(cl));
                g00_366 = _min_140;
                float _min_141;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_141) : "f"(g01_367), "f"(cl));
                g01_367 = _min_141;
                float _min_142;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_142) : "f"(g10_368), "f"(cl));
                g10_368 = _min_142;
                float _min_143;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_143) : "f"(g11_369), "f"(cl));
                g11_369 = _min_143;
                float _mul_212;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_212) : "f"(lin00_370), "f"(g00_366));
                float value00_374 = _mul_212;
                float _mul_213;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_213) : "f"(lin01_371), "f"(g01_367));
                float value01_375 = _mul_213;
                float _mul_214;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_214) : "f"(lin10_372), "f"(g10_368));
                float value10_376 = _mul_214;
                float _mul_215;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_215) : "f"(lin11_373), "f"(g11_369));
                float value11_377 = _mul_215;
                pair[0] = value00_374;
                pair[1] = value10_376;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token0_356 * 32 + (feature_chunk ^ local_token0_356 % 8) * 4 + feature_word] = word[0];
                pair[0] = value01_375;
                pair[1] = value11_377;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token1_357 * 32 + (feature_chunk ^ local_token1_357 % 8) * 4 + feature_word] = word[0];
                int local_token0_378 = lane_1 % 4 * 2 + 16;
                int local_token1_379 = local_token0_378 + 1;
                float x00_380 = _tmem_load_8[10];
                float x01_381 = _tmem_load_8[11];
                float x10_382 = _tmem_load_9[10];
                float x11_383 = _tmem_load_9[11];
                float _mul_216;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_216) : "f"(x00_380), "f"(-1.4426950408889634f));
                float _exp2_72 = approx_exp2(_mul_216);
                float _add_72;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_72) : "f"(1.0f), "f"(_exp2_72));
                float _rcp_72 = approx_rcp(_add_72);
                float sig00_384 = _rcp_72;
                float _mul_217;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_217) : "f"(x01_381), "f"(-1.4426950408889634f));
                float _exp2_73 = approx_exp2(_mul_217);
                float _add_73;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_73) : "f"(1.0f), "f"(_exp2_73));
                float _rcp_73 = approx_rcp(_add_73);
                float sig01_385 = _rcp_73;
                float _mul_218;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_218) : "f"(x10_382), "f"(-1.4426950408889634f));
                float _exp2_74 = approx_exp2(_mul_218);
                float _add_74;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_74) : "f"(1.0f), "f"(_exp2_74));
                float _rcp_74 = approx_rcp(_add_74);
                float sig10_386 = _rcp_74;
                float _mul_219;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_219) : "f"(x11_383), "f"(-1.4426950408889634f));
                float _exp2_75 = approx_exp2(_mul_219);
                float _add_75;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_75) : "f"(1.0f), "f"(_exp2_75));
                float _rcp_75 = approx_rcp(_add_75);
                float sig11_387 = _rcp_75;
                float _mul_220;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_220) : "f"(x00_380), "f"(sig00_384));
                float g00_388 = _mul_220;
                float _mul_221;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_221) : "f"(x01_381), "f"(sig01_385));
                float g01_389 = _mul_221;
                float _mul_222;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_222) : "f"(x10_382), "f"(sig10_386));
                float g10_390 = _mul_222;
                float _mul_223;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_223) : "f"(x11_383), "f"(sig11_387));
                float g11_391 = _mul_223;
                float lin00_392 = _tmem_load_8[8];
                float lin01_393 = _tmem_load_8[9];
                float lin10_394 = _tmem_load_9[8];
                float lin11_395 = _tmem_load_9[9];
                float _max_72;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_72) : "f"(lin00_392), "f"(neg_cl));
                float _min_144;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_144) : "f"(_max_72), "f"(cl));
                lin00_392 = _min_144;
                float _max_73;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_73) : "f"(lin01_393), "f"(neg_cl));
                float _min_145;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_145) : "f"(_max_73), "f"(cl));
                lin01_393 = _min_145;
                float _max_74;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_74) : "f"(lin10_394), "f"(neg_cl));
                float _min_146;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_146) : "f"(_max_74), "f"(cl));
                lin10_394 = _min_146;
                float _max_75;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_75) : "f"(lin11_395), "f"(neg_cl));
                float _min_147;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_147) : "f"(_max_75), "f"(cl));
                lin11_395 = _min_147;
                float _min_148;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_148) : "f"(g00_388), "f"(cl));
                g00_388 = _min_148;
                float _min_149;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_149) : "f"(g01_389), "f"(cl));
                g01_389 = _min_149;
                float _min_150;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_150) : "f"(g10_390), "f"(cl));
                g10_390 = _min_150;
                float _min_151;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_151) : "f"(g11_391), "f"(cl));
                g11_391 = _min_151;
                float _mul_224;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_224) : "f"(lin00_392), "f"(g00_388));
                float value00_396 = _mul_224;
                float _mul_225;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_225) : "f"(lin01_393), "f"(g01_389));
                float value01_397 = _mul_225;
                float _mul_226;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_226) : "f"(lin10_394), "f"(g10_390));
                float value10_398 = _mul_226;
                float _mul_227;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_227) : "f"(lin11_395), "f"(g11_391));
                float value11_399 = _mul_227;
                pair[0] = value00_396;
                pair[1] = value10_398;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token0_378 * 32 + (feature_chunk ^ local_token0_378 % 8) * 4 + feature_word] = word[0];
                pair[0] = value01_397;
                pair[1] = value11_399;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token1_379 * 32 + (feature_chunk ^ local_token1_379 % 8) * 4 + feature_word] = word[0];
                int local_token0_400 = lane_1 % 4 * 2 + 24;
                int local_token1_401 = local_token0_400 + 1;
                float x00_402 = _tmem_load_8[14];
                float x01_403 = _tmem_load_8[15];
                float x10_404 = _tmem_load_9[14];
                float x11_405 = _tmem_load_9[15];
                float _mul_228;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_228) : "f"(x00_402), "f"(-1.4426950408889634f));
                float _exp2_76 = approx_exp2(_mul_228);
                float _add_76;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_76) : "f"(1.0f), "f"(_exp2_76));
                float _rcp_76 = approx_rcp(_add_76);
                float sig00_406 = _rcp_76;
                float _mul_229;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_229) : "f"(x01_403), "f"(-1.4426950408889634f));
                float _exp2_77 = approx_exp2(_mul_229);
                float _add_77;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_77) : "f"(1.0f), "f"(_exp2_77));
                float _rcp_77 = approx_rcp(_add_77);
                float sig01_407 = _rcp_77;
                float _mul_230;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_230) : "f"(x10_404), "f"(-1.4426950408889634f));
                float _exp2_78 = approx_exp2(_mul_230);
                float _add_78;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_78) : "f"(1.0f), "f"(_exp2_78));
                float _rcp_78 = approx_rcp(_add_78);
                float sig10_408 = _rcp_78;
                float _mul_231;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_231) : "f"(x11_405), "f"(-1.4426950408889634f));
                float _exp2_79 = approx_exp2(_mul_231);
                float _add_79;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_79) : "f"(1.0f), "f"(_exp2_79));
                float _rcp_79 = approx_rcp(_add_79);
                float sig11_409 = _rcp_79;
                float _mul_232;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_232) : "f"(x00_402), "f"(sig00_406));
                float g00_410 = _mul_232;
                float _mul_233;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_233) : "f"(x01_403), "f"(sig01_407));
                float g01_411 = _mul_233;
                float _mul_234;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_234) : "f"(x10_404), "f"(sig10_408));
                float g10_412 = _mul_234;
                float _mul_235;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_235) : "f"(x11_405), "f"(sig11_409));
                float g11_413 = _mul_235;
                float lin00_414 = _tmem_load_8[12];
                float lin01_415 = _tmem_load_8[13];
                float lin10_416 = _tmem_load_9[12];
                float lin11_417 = _tmem_load_9[13];
                float _max_76;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_76) : "f"(lin00_414), "f"(neg_cl));
                float _min_152;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_152) : "f"(_max_76), "f"(cl));
                lin00_414 = _min_152;
                float _max_77;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_77) : "f"(lin01_415), "f"(neg_cl));
                float _min_153;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_153) : "f"(_max_77), "f"(cl));
                lin01_415 = _min_153;
                float _max_78;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_78) : "f"(lin10_416), "f"(neg_cl));
                float _min_154;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_154) : "f"(_max_78), "f"(cl));
                lin10_416 = _min_154;
                float _max_79;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_79) : "f"(lin11_417), "f"(neg_cl));
                float _min_155;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_155) : "f"(_max_79), "f"(cl));
                lin11_417 = _min_155;
                float _min_156;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_156) : "f"(g00_410), "f"(cl));
                g00_410 = _min_156;
                float _min_157;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_157) : "f"(g01_411), "f"(cl));
                g01_411 = _min_157;
                float _min_158;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_158) : "f"(g10_412), "f"(cl));
                g10_412 = _min_158;
                float _min_159;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_159) : "f"(g11_413), "f"(cl));
                g11_413 = _min_159;
                float _mul_236;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_236) : "f"(lin00_414), "f"(g00_410));
                float value00_418 = _mul_236;
                float _mul_237;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_237) : "f"(lin01_415), "f"(g01_411));
                float value01_419 = _mul_237;
                float _mul_238;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_238) : "f"(lin10_416), "f"(g10_412));
                float value10_420 = _mul_238;
                float _mul_239;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_239) : "f"(lin11_417), "f"(g11_413));
                float value11_421 = _mul_239;
                pair[0] = value00_418;
                pair[1] = value10_420;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token0_400 * 32 + (feature_chunk ^ local_token0_400 % 8) * 4 + feature_word] = word[0];
                pair[0] = value01_419;
                pair[1] = value11_421;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token1_401 * 32 + (feature_chunk ^ local_token1_401 % 8) * 4 + feature_word] = word[0];
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 7, 128;" ::: "memory");
                if (warp == 0) {
                    if (elect_sync()) {
                        int padding_rows_4 = (256 - valid_rows % 256) % 256;
                        tma_store_4d((&C), m_tile * 64, padding_rows_4 + 128, 1073741824, n_tile * 256 - (unsigned int)padding_rows_4 + 1073741824, epi_staging_addr);
                    }
                }
                asm volatile("cp.async.bulk.commit_group;");
                asm volatile("barrier.sync 7, 128;" ::: "memory");
                int acc_offset_422 = acc_stage * 256 + 160;
                float _tmem_load_10[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[15]))
                    : "r"(taddr + (unsigned int)acc_offset_422));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float _tmem_load_11[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[15]))
                    : "r"(taddr + 1048576 + (unsigned int)acc_offset_422));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                asm volatile("cp.async.bulk.wait_group.read 0;");
                asm volatile("barrier.sync 7, 128;" ::: "memory");
                int local_token0_423 = lane_1 % 4 * 2;
                int local_token1_424 = local_token0_423 + 1;
                float x00_425 = _tmem_load_10[2];
                float x01_426 = _tmem_load_10[3];
                float x10_427 = _tmem_load_11[2];
                float x11_428 = _tmem_load_11[3];
                float _mul_240;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_240) : "f"(x00_425), "f"(-1.4426950408889634f));
                float _exp2_80 = approx_exp2(_mul_240);
                float _add_80;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_80) : "f"(1.0f), "f"(_exp2_80));
                float _rcp_80 = approx_rcp(_add_80);
                float sig00_429 = _rcp_80;
                float _mul_241;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_241) : "f"(x01_426), "f"(-1.4426950408889634f));
                float _exp2_81 = approx_exp2(_mul_241);
                float _add_81;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_81) : "f"(1.0f), "f"(_exp2_81));
                float _rcp_81 = approx_rcp(_add_81);
                float sig01_430 = _rcp_81;
                float _mul_242;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_242) : "f"(x10_427), "f"(-1.4426950408889634f));
                float _exp2_82 = approx_exp2(_mul_242);
                float _add_82;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_82) : "f"(1.0f), "f"(_exp2_82));
                float _rcp_82 = approx_rcp(_add_82);
                float sig10_431 = _rcp_82;
                float _mul_243;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_243) : "f"(x11_428), "f"(-1.4426950408889634f));
                float _exp2_83 = approx_exp2(_mul_243);
                float _add_83;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_83) : "f"(1.0f), "f"(_exp2_83));
                float _rcp_83 = approx_rcp(_add_83);
                float sig11_432 = _rcp_83;
                float _mul_244;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_244) : "f"(x00_425), "f"(sig00_429));
                float g00_433 = _mul_244;
                float _mul_245;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_245) : "f"(x01_426), "f"(sig01_430));
                float g01_434 = _mul_245;
                float _mul_246;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_246) : "f"(x10_427), "f"(sig10_431));
                float g10_435 = _mul_246;
                float _mul_247;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_247) : "f"(x11_428), "f"(sig11_432));
                float g11_436 = _mul_247;
                float lin00_437 = _tmem_load_10[0];
                float lin01_438 = _tmem_load_10[1];
                float lin10_439 = _tmem_load_11[0];
                float lin11_440 = _tmem_load_11[1];
                float _max_80;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_80) : "f"(lin00_437), "f"(neg_cl));
                float _min_160;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_160) : "f"(_max_80), "f"(cl));
                lin00_437 = _min_160;
                float _max_81;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_81) : "f"(lin01_438), "f"(neg_cl));
                float _min_161;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_161) : "f"(_max_81), "f"(cl));
                lin01_438 = _min_161;
                float _max_82;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_82) : "f"(lin10_439), "f"(neg_cl));
                float _min_162;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_162) : "f"(_max_82), "f"(cl));
                lin10_439 = _min_162;
                float _max_83;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_83) : "f"(lin11_440), "f"(neg_cl));
                float _min_163;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_163) : "f"(_max_83), "f"(cl));
                lin11_440 = _min_163;
                float _min_164;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_164) : "f"(g00_433), "f"(cl));
                g00_433 = _min_164;
                float _min_165;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_165) : "f"(g01_434), "f"(cl));
                g01_434 = _min_165;
                float _min_166;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_166) : "f"(g10_435), "f"(cl));
                g10_435 = _min_166;
                float _min_167;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_167) : "f"(g11_436), "f"(cl));
                g11_436 = _min_167;
                float _mul_248;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_248) : "f"(lin00_437), "f"(g00_433));
                float value00_441 = _mul_248;
                float _mul_249;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_249) : "f"(lin01_438), "f"(g01_434));
                float value01_442 = _mul_249;
                float _mul_250;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_250) : "f"(lin10_439), "f"(g10_435));
                float value10_443 = _mul_250;
                float _mul_251;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_251) : "f"(lin11_440), "f"(g11_436));
                float value11_444 = _mul_251;
                pair[0] = value00_441;
                pair[1] = value10_443;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token0_423 * 32 + (feature_chunk ^ local_token0_423 % 8) * 4 + feature_word] = word[0];
                pair[0] = value01_442;
                pair[1] = value11_444;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token1_424 * 32 + (feature_chunk ^ local_token1_424 % 8) * 4 + feature_word] = word[0];
                int local_token0_445 = lane_1 % 4 * 2 + 8;
                int local_token1_446 = local_token0_445 + 1;
                float x00_447 = _tmem_load_10[6];
                float x01_448 = _tmem_load_10[7];
                float x10_449 = _tmem_load_11[6];
                float x11_450 = _tmem_load_11[7];
                float _mul_252;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_252) : "f"(x00_447), "f"(-1.4426950408889634f));
                float _exp2_84 = approx_exp2(_mul_252);
                float _add_84;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_84) : "f"(1.0f), "f"(_exp2_84));
                float _rcp_84 = approx_rcp(_add_84);
                float sig00_451 = _rcp_84;
                float _mul_253;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_253) : "f"(x01_448), "f"(-1.4426950408889634f));
                float _exp2_85 = approx_exp2(_mul_253);
                float _add_85;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_85) : "f"(1.0f), "f"(_exp2_85));
                float _rcp_85 = approx_rcp(_add_85);
                float sig01_452 = _rcp_85;
                float _mul_254;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_254) : "f"(x10_449), "f"(-1.4426950408889634f));
                float _exp2_86 = approx_exp2(_mul_254);
                float _add_86;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_86) : "f"(1.0f), "f"(_exp2_86));
                float _rcp_86 = approx_rcp(_add_86);
                float sig10_453 = _rcp_86;
                float _mul_255;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_255) : "f"(x11_450), "f"(-1.4426950408889634f));
                float _exp2_87 = approx_exp2(_mul_255);
                float _add_87;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_87) : "f"(1.0f), "f"(_exp2_87));
                float _rcp_87 = approx_rcp(_add_87);
                float sig11_454 = _rcp_87;
                float _mul_256;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_256) : "f"(x00_447), "f"(sig00_451));
                float g00_455 = _mul_256;
                float _mul_257;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_257) : "f"(x01_448), "f"(sig01_452));
                float g01_456 = _mul_257;
                float _mul_258;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_258) : "f"(x10_449), "f"(sig10_453));
                float g10_457 = _mul_258;
                float _mul_259;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_259) : "f"(x11_450), "f"(sig11_454));
                float g11_458 = _mul_259;
                float lin00_459 = _tmem_load_10[4];
                float lin01_460 = _tmem_load_10[5];
                float lin10_461 = _tmem_load_11[4];
                float lin11_462 = _tmem_load_11[5];
                float _max_84;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_84) : "f"(lin00_459), "f"(neg_cl));
                float _min_168;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_168) : "f"(_max_84), "f"(cl));
                lin00_459 = _min_168;
                float _max_85;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_85) : "f"(lin01_460), "f"(neg_cl));
                float _min_169;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_169) : "f"(_max_85), "f"(cl));
                lin01_460 = _min_169;
                float _max_86;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_86) : "f"(lin10_461), "f"(neg_cl));
                float _min_170;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_170) : "f"(_max_86), "f"(cl));
                lin10_461 = _min_170;
                float _max_87;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_87) : "f"(lin11_462), "f"(neg_cl));
                float _min_171;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_171) : "f"(_max_87), "f"(cl));
                lin11_462 = _min_171;
                float _min_172;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_172) : "f"(g00_455), "f"(cl));
                g00_455 = _min_172;
                float _min_173;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_173) : "f"(g01_456), "f"(cl));
                g01_456 = _min_173;
                float _min_174;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_174) : "f"(g10_457), "f"(cl));
                g10_457 = _min_174;
                float _min_175;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_175) : "f"(g11_458), "f"(cl));
                g11_458 = _min_175;
                float _mul_260;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_260) : "f"(lin00_459), "f"(g00_455));
                float value00_463 = _mul_260;
                float _mul_261;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_261) : "f"(lin01_460), "f"(g01_456));
                float value01_464 = _mul_261;
                float _mul_262;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_262) : "f"(lin10_461), "f"(g10_457));
                float value10_465 = _mul_262;
                float _mul_263;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_263) : "f"(lin11_462), "f"(g11_458));
                float value11_466 = _mul_263;
                pair[0] = value00_463;
                pair[1] = value10_465;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token0_445 * 32 + (feature_chunk ^ local_token0_445 % 8) * 4 + feature_word] = word[0];
                pair[0] = value01_464;
                pair[1] = value11_466;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token1_446 * 32 + (feature_chunk ^ local_token1_446 % 8) * 4 + feature_word] = word[0];
                int local_token0_467 = lane_1 % 4 * 2 + 16;
                int local_token1_468 = local_token0_467 + 1;
                float x00_469 = _tmem_load_10[10];
                float x01_470 = _tmem_load_10[11];
                float x10_471 = _tmem_load_11[10];
                float x11_472 = _tmem_load_11[11];
                float _mul_264;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_264) : "f"(x00_469), "f"(-1.4426950408889634f));
                float _exp2_88 = approx_exp2(_mul_264);
                float _add_88;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_88) : "f"(1.0f), "f"(_exp2_88));
                float _rcp_88 = approx_rcp(_add_88);
                float sig00_473 = _rcp_88;
                float _mul_265;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_265) : "f"(x01_470), "f"(-1.4426950408889634f));
                float _exp2_89 = approx_exp2(_mul_265);
                float _add_89;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_89) : "f"(1.0f), "f"(_exp2_89));
                float _rcp_89 = approx_rcp(_add_89);
                float sig01_474 = _rcp_89;
                float _mul_266;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_266) : "f"(x10_471), "f"(-1.4426950408889634f));
                float _exp2_90 = approx_exp2(_mul_266);
                float _add_90;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_90) : "f"(1.0f), "f"(_exp2_90));
                float _rcp_90 = approx_rcp(_add_90);
                float sig10_475 = _rcp_90;
                float _mul_267;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_267) : "f"(x11_472), "f"(-1.4426950408889634f));
                float _exp2_91 = approx_exp2(_mul_267);
                float _add_91;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_91) : "f"(1.0f), "f"(_exp2_91));
                float _rcp_91 = approx_rcp(_add_91);
                float sig11_476 = _rcp_91;
                float _mul_268;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_268) : "f"(x00_469), "f"(sig00_473));
                float g00_477 = _mul_268;
                float _mul_269;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_269) : "f"(x01_470), "f"(sig01_474));
                float g01_478 = _mul_269;
                float _mul_270;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_270) : "f"(x10_471), "f"(sig10_475));
                float g10_479 = _mul_270;
                float _mul_271;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_271) : "f"(x11_472), "f"(sig11_476));
                float g11_480 = _mul_271;
                float lin00_481 = _tmem_load_10[8];
                float lin01_482 = _tmem_load_10[9];
                float lin10_483 = _tmem_load_11[8];
                float lin11_484 = _tmem_load_11[9];
                float _max_88;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_88) : "f"(lin00_481), "f"(neg_cl));
                float _min_176;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_176) : "f"(_max_88), "f"(cl));
                lin00_481 = _min_176;
                float _max_89;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_89) : "f"(lin01_482), "f"(neg_cl));
                float _min_177;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_177) : "f"(_max_89), "f"(cl));
                lin01_482 = _min_177;
                float _max_90;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_90) : "f"(lin10_483), "f"(neg_cl));
                float _min_178;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_178) : "f"(_max_90), "f"(cl));
                lin10_483 = _min_178;
                float _max_91;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_91) : "f"(lin11_484), "f"(neg_cl));
                float _min_179;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_179) : "f"(_max_91), "f"(cl));
                lin11_484 = _min_179;
                float _min_180;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_180) : "f"(g00_477), "f"(cl));
                g00_477 = _min_180;
                float _min_181;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_181) : "f"(g01_478), "f"(cl));
                g01_478 = _min_181;
                float _min_182;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_182) : "f"(g10_479), "f"(cl));
                g10_479 = _min_182;
                float _min_183;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_183) : "f"(g11_480), "f"(cl));
                g11_480 = _min_183;
                float _mul_272;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_272) : "f"(lin00_481), "f"(g00_477));
                float value00_485 = _mul_272;
                float _mul_273;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_273) : "f"(lin01_482), "f"(g01_478));
                float value01_486 = _mul_273;
                float _mul_274;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_274) : "f"(lin10_483), "f"(g10_479));
                float value10_487 = _mul_274;
                float _mul_275;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_275) : "f"(lin11_484), "f"(g11_480));
                float value11_488 = _mul_275;
                pair[0] = value00_485;
                pair[1] = value10_487;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token0_467 * 32 + (feature_chunk ^ local_token0_467 % 8) * 4 + feature_word] = word[0];
                pair[0] = value01_486;
                pair[1] = value11_488;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token1_468 * 32 + (feature_chunk ^ local_token1_468 % 8) * 4 + feature_word] = word[0];
                int local_token0_489 = lane_1 % 4 * 2 + 24;
                int local_token1_490 = local_token0_489 + 1;
                float x00_491 = _tmem_load_10[14];
                float x01_492 = _tmem_load_10[15];
                float x10_493 = _tmem_load_11[14];
                float x11_494 = _tmem_load_11[15];
                float _mul_276;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_276) : "f"(x00_491), "f"(-1.4426950408889634f));
                float _exp2_92 = approx_exp2(_mul_276);
                float _add_92;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_92) : "f"(1.0f), "f"(_exp2_92));
                float _rcp_92 = approx_rcp(_add_92);
                float sig00_495 = _rcp_92;
                float _mul_277;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_277) : "f"(x01_492), "f"(-1.4426950408889634f));
                float _exp2_93 = approx_exp2(_mul_277);
                float _add_93;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_93) : "f"(1.0f), "f"(_exp2_93));
                float _rcp_93 = approx_rcp(_add_93);
                float sig01_496 = _rcp_93;
                float _mul_278;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_278) : "f"(x10_493), "f"(-1.4426950408889634f));
                float _exp2_94 = approx_exp2(_mul_278);
                float _add_94;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_94) : "f"(1.0f), "f"(_exp2_94));
                float _rcp_94 = approx_rcp(_add_94);
                float sig10_497 = _rcp_94;
                float _mul_279;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_279) : "f"(x11_494), "f"(-1.4426950408889634f));
                float _exp2_95 = approx_exp2(_mul_279);
                float _add_95;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_95) : "f"(1.0f), "f"(_exp2_95));
                float _rcp_95 = approx_rcp(_add_95);
                float sig11_498 = _rcp_95;
                float _mul_280;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_280) : "f"(x00_491), "f"(sig00_495));
                float g00_499 = _mul_280;
                float _mul_281;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_281) : "f"(x01_492), "f"(sig01_496));
                float g01_500 = _mul_281;
                float _mul_282;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_282) : "f"(x10_493), "f"(sig10_497));
                float g10_501 = _mul_282;
                float _mul_283;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_283) : "f"(x11_494), "f"(sig11_498));
                float g11_502 = _mul_283;
                float lin00_503 = _tmem_load_10[12];
                float lin01_504 = _tmem_load_10[13];
                float lin10_505 = _tmem_load_11[12];
                float lin11_506 = _tmem_load_11[13];
                float _max_92;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_92) : "f"(lin00_503), "f"(neg_cl));
                float _min_184;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_184) : "f"(_max_92), "f"(cl));
                lin00_503 = _min_184;
                float _max_93;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_93) : "f"(lin01_504), "f"(neg_cl));
                float _min_185;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_185) : "f"(_max_93), "f"(cl));
                lin01_504 = _min_185;
                float _max_94;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_94) : "f"(lin10_505), "f"(neg_cl));
                float _min_186;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_186) : "f"(_max_94), "f"(cl));
                lin10_505 = _min_186;
                float _max_95;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_95) : "f"(lin11_506), "f"(neg_cl));
                float _min_187;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_187) : "f"(_max_95), "f"(cl));
                lin11_506 = _min_187;
                float _min_188;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_188) : "f"(g00_499), "f"(cl));
                g00_499 = _min_188;
                float _min_189;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_189) : "f"(g01_500), "f"(cl));
                g01_500 = _min_189;
                float _min_190;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_190) : "f"(g10_501), "f"(cl));
                g10_501 = _min_190;
                float _min_191;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_191) : "f"(g11_502), "f"(cl));
                g11_502 = _min_191;
                float _mul_284;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_284) : "f"(lin00_503), "f"(g00_499));
                float value00_507 = _mul_284;
                float _mul_285;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_285) : "f"(lin01_504), "f"(g01_500));
                float value01_508 = _mul_285;
                float _mul_286;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_286) : "f"(lin10_505), "f"(g10_501));
                float value10_509 = _mul_286;
                float _mul_287;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_287) : "f"(lin11_506), "f"(g11_502));
                float value11_510 = _mul_287;
                pair[0] = value00_507;
                pair[1] = value10_509;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token0_489 * 32 + (feature_chunk ^ local_token0_489 % 8) * 4 + feature_word] = word[0];
                pair[0] = value01_508;
                pair[1] = value11_510;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token1_490 * 32 + (feature_chunk ^ local_token1_490 % 8) * 4 + feature_word] = word[0];
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 7, 128;" ::: "memory");
                if (warp == 0) {
                    if (elect_sync()) {
                        int padding_rows_5 = (256 - valid_rows % 256) % 256;
                        tma_store_4d((&C), m_tile * 64, padding_rows_5 + 160, 1073741824, n_tile * 256 - (unsigned int)padding_rows_5 + 1073741824, epi_staging_addr);
                    }
                }
                asm volatile("cp.async.bulk.commit_group;");
                asm volatile("barrier.sync 7, 128;" ::: "memory");
                int acc_offset_511 = acc_stage * 256 + 192;
                float _tmem_load_12[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_12[15]))
                    : "r"(taddr + (unsigned int)acc_offset_511));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float _tmem_load_13[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_13[15]))
                    : "r"(taddr + 1048576 + (unsigned int)acc_offset_511));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                asm volatile("cp.async.bulk.wait_group.read 0;");
                asm volatile("barrier.sync 7, 128;" ::: "memory");
                int local_token0_512 = lane_1 % 4 * 2;
                int local_token1_513 = local_token0_512 + 1;
                float x00_514 = _tmem_load_12[2];
                float x01_515 = _tmem_load_12[3];
                float x10_516 = _tmem_load_13[2];
                float x11_517 = _tmem_load_13[3];
                float _mul_288;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_288) : "f"(x00_514), "f"(-1.4426950408889634f));
                float _exp2_96 = approx_exp2(_mul_288);
                float _add_96;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_96) : "f"(1.0f), "f"(_exp2_96));
                float _rcp_96 = approx_rcp(_add_96);
                float sig00_518 = _rcp_96;
                float _mul_289;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_289) : "f"(x01_515), "f"(-1.4426950408889634f));
                float _exp2_97 = approx_exp2(_mul_289);
                float _add_97;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_97) : "f"(1.0f), "f"(_exp2_97));
                float _rcp_97 = approx_rcp(_add_97);
                float sig01_519 = _rcp_97;
                float _mul_290;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_290) : "f"(x10_516), "f"(-1.4426950408889634f));
                float _exp2_98 = approx_exp2(_mul_290);
                float _add_98;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_98) : "f"(1.0f), "f"(_exp2_98));
                float _rcp_98 = approx_rcp(_add_98);
                float sig10_520 = _rcp_98;
                float _mul_291;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_291) : "f"(x11_517), "f"(-1.4426950408889634f));
                float _exp2_99 = approx_exp2(_mul_291);
                float _add_99;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_99) : "f"(1.0f), "f"(_exp2_99));
                float _rcp_99 = approx_rcp(_add_99);
                float sig11_521 = _rcp_99;
                float _mul_292;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_292) : "f"(x00_514), "f"(sig00_518));
                float g00_522 = _mul_292;
                float _mul_293;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_293) : "f"(x01_515), "f"(sig01_519));
                float g01_523 = _mul_293;
                float _mul_294;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_294) : "f"(x10_516), "f"(sig10_520));
                float g10_524 = _mul_294;
                float _mul_295;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_295) : "f"(x11_517), "f"(sig11_521));
                float g11_525 = _mul_295;
                float lin00_526 = _tmem_load_12[0];
                float lin01_527 = _tmem_load_12[1];
                float lin10_528 = _tmem_load_13[0];
                float lin11_529 = _tmem_load_13[1];
                float _max_96;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_96) : "f"(lin00_526), "f"(neg_cl));
                float _min_192;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_192) : "f"(_max_96), "f"(cl));
                lin00_526 = _min_192;
                float _max_97;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_97) : "f"(lin01_527), "f"(neg_cl));
                float _min_193;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_193) : "f"(_max_97), "f"(cl));
                lin01_527 = _min_193;
                float _max_98;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_98) : "f"(lin10_528), "f"(neg_cl));
                float _min_194;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_194) : "f"(_max_98), "f"(cl));
                lin10_528 = _min_194;
                float _max_99;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_99) : "f"(lin11_529), "f"(neg_cl));
                float _min_195;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_195) : "f"(_max_99), "f"(cl));
                lin11_529 = _min_195;
                float _min_196;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_196) : "f"(g00_522), "f"(cl));
                g00_522 = _min_196;
                float _min_197;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_197) : "f"(g01_523), "f"(cl));
                g01_523 = _min_197;
                float _min_198;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_198) : "f"(g10_524), "f"(cl));
                g10_524 = _min_198;
                float _min_199;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_199) : "f"(g11_525), "f"(cl));
                g11_525 = _min_199;
                float _mul_296;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_296) : "f"(lin00_526), "f"(g00_522));
                float value00_530 = _mul_296;
                float _mul_297;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_297) : "f"(lin01_527), "f"(g01_523));
                float value01_531 = _mul_297;
                float _mul_298;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_298) : "f"(lin10_528), "f"(g10_524));
                float value10_532 = _mul_298;
                float _mul_299;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_299) : "f"(lin11_529), "f"(g11_525));
                float value11_533 = _mul_299;
                pair[0] = value00_530;
                pair[1] = value10_532;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token0_512 * 32 + (feature_chunk ^ local_token0_512 % 8) * 4 + feature_word] = word[0];
                pair[0] = value01_531;
                pair[1] = value11_533;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token1_513 * 32 + (feature_chunk ^ local_token1_513 % 8) * 4 + feature_word] = word[0];
                int local_token0_534 = lane_1 % 4 * 2 + 8;
                int local_token1_535 = local_token0_534 + 1;
                float x00_536 = _tmem_load_12[6];
                float x01_537 = _tmem_load_12[7];
                float x10_538 = _tmem_load_13[6];
                float x11_539 = _tmem_load_13[7];
                float _mul_300;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_300) : "f"(x00_536), "f"(-1.4426950408889634f));
                float _exp2_100 = approx_exp2(_mul_300);
                float _add_100;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_100) : "f"(1.0f), "f"(_exp2_100));
                float _rcp_100 = approx_rcp(_add_100);
                float sig00_540 = _rcp_100;
                float _mul_301;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_301) : "f"(x01_537), "f"(-1.4426950408889634f));
                float _exp2_101 = approx_exp2(_mul_301);
                float _add_101;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_101) : "f"(1.0f), "f"(_exp2_101));
                float _rcp_101 = approx_rcp(_add_101);
                float sig01_541 = _rcp_101;
                float _mul_302;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_302) : "f"(x10_538), "f"(-1.4426950408889634f));
                float _exp2_102 = approx_exp2(_mul_302);
                float _add_102;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_102) : "f"(1.0f), "f"(_exp2_102));
                float _rcp_102 = approx_rcp(_add_102);
                float sig10_542 = _rcp_102;
                float _mul_303;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_303) : "f"(x11_539), "f"(-1.4426950408889634f));
                float _exp2_103 = approx_exp2(_mul_303);
                float _add_103;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_103) : "f"(1.0f), "f"(_exp2_103));
                float _rcp_103 = approx_rcp(_add_103);
                float sig11_543 = _rcp_103;
                float _mul_304;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_304) : "f"(x00_536), "f"(sig00_540));
                float g00_544 = _mul_304;
                float _mul_305;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_305) : "f"(x01_537), "f"(sig01_541));
                float g01_545 = _mul_305;
                float _mul_306;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_306) : "f"(x10_538), "f"(sig10_542));
                float g10_546 = _mul_306;
                float _mul_307;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_307) : "f"(x11_539), "f"(sig11_543));
                float g11_547 = _mul_307;
                float lin00_548 = _tmem_load_12[4];
                float lin01_549 = _tmem_load_12[5];
                float lin10_550 = _tmem_load_13[4];
                float lin11_551 = _tmem_load_13[5];
                float _max_100;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_100) : "f"(lin00_548), "f"(neg_cl));
                float _min_200;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_200) : "f"(_max_100), "f"(cl));
                lin00_548 = _min_200;
                float _max_101;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_101) : "f"(lin01_549), "f"(neg_cl));
                float _min_201;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_201) : "f"(_max_101), "f"(cl));
                lin01_549 = _min_201;
                float _max_102;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_102) : "f"(lin10_550), "f"(neg_cl));
                float _min_202;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_202) : "f"(_max_102), "f"(cl));
                lin10_550 = _min_202;
                float _max_103;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_103) : "f"(lin11_551), "f"(neg_cl));
                float _min_203;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_203) : "f"(_max_103), "f"(cl));
                lin11_551 = _min_203;
                float _min_204;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_204) : "f"(g00_544), "f"(cl));
                g00_544 = _min_204;
                float _min_205;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_205) : "f"(g01_545), "f"(cl));
                g01_545 = _min_205;
                float _min_206;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_206) : "f"(g10_546), "f"(cl));
                g10_546 = _min_206;
                float _min_207;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_207) : "f"(g11_547), "f"(cl));
                g11_547 = _min_207;
                float _mul_308;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_308) : "f"(lin00_548), "f"(g00_544));
                float value00_552 = _mul_308;
                float _mul_309;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_309) : "f"(lin01_549), "f"(g01_545));
                float value01_553 = _mul_309;
                float _mul_310;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_310) : "f"(lin10_550), "f"(g10_546));
                float value10_554 = _mul_310;
                float _mul_311;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_311) : "f"(lin11_551), "f"(g11_547));
                float value11_555 = _mul_311;
                pair[0] = value00_552;
                pair[1] = value10_554;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token0_534 * 32 + (feature_chunk ^ local_token0_534 % 8) * 4 + feature_word] = word[0];
                pair[0] = value01_553;
                pair[1] = value11_555;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token1_535 * 32 + (feature_chunk ^ local_token1_535 % 8) * 4 + feature_word] = word[0];
                int local_token0_556 = lane_1 % 4 * 2 + 16;
                int local_token1_557 = local_token0_556 + 1;
                float x00_558 = _tmem_load_12[10];
                float x01_559 = _tmem_load_12[11];
                float x10_560 = _tmem_load_13[10];
                float x11_561 = _tmem_load_13[11];
                float _mul_312;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_312) : "f"(x00_558), "f"(-1.4426950408889634f));
                float _exp2_104 = approx_exp2(_mul_312);
                float _add_104;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_104) : "f"(1.0f), "f"(_exp2_104));
                float _rcp_104 = approx_rcp(_add_104);
                float sig00_562 = _rcp_104;
                float _mul_313;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_313) : "f"(x01_559), "f"(-1.4426950408889634f));
                float _exp2_105 = approx_exp2(_mul_313);
                float _add_105;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_105) : "f"(1.0f), "f"(_exp2_105));
                float _rcp_105 = approx_rcp(_add_105);
                float sig01_563 = _rcp_105;
                float _mul_314;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_314) : "f"(x10_560), "f"(-1.4426950408889634f));
                float _exp2_106 = approx_exp2(_mul_314);
                float _add_106;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_106) : "f"(1.0f), "f"(_exp2_106));
                float _rcp_106 = approx_rcp(_add_106);
                float sig10_564 = _rcp_106;
                float _mul_315;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_315) : "f"(x11_561), "f"(-1.4426950408889634f));
                float _exp2_107 = approx_exp2(_mul_315);
                float _add_107;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_107) : "f"(1.0f), "f"(_exp2_107));
                float _rcp_107 = approx_rcp(_add_107);
                float sig11_565 = _rcp_107;
                float _mul_316;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_316) : "f"(x00_558), "f"(sig00_562));
                float g00_566 = _mul_316;
                float _mul_317;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_317) : "f"(x01_559), "f"(sig01_563));
                float g01_567 = _mul_317;
                float _mul_318;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_318) : "f"(x10_560), "f"(sig10_564));
                float g10_568 = _mul_318;
                float _mul_319;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_319) : "f"(x11_561), "f"(sig11_565));
                float g11_569 = _mul_319;
                float lin00_570 = _tmem_load_12[8];
                float lin01_571 = _tmem_load_12[9];
                float lin10_572 = _tmem_load_13[8];
                float lin11_573 = _tmem_load_13[9];
                float _max_104;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_104) : "f"(lin00_570), "f"(neg_cl));
                float _min_208;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_208) : "f"(_max_104), "f"(cl));
                lin00_570 = _min_208;
                float _max_105;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_105) : "f"(lin01_571), "f"(neg_cl));
                float _min_209;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_209) : "f"(_max_105), "f"(cl));
                lin01_571 = _min_209;
                float _max_106;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_106) : "f"(lin10_572), "f"(neg_cl));
                float _min_210;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_210) : "f"(_max_106), "f"(cl));
                lin10_572 = _min_210;
                float _max_107;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_107) : "f"(lin11_573), "f"(neg_cl));
                float _min_211;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_211) : "f"(_max_107), "f"(cl));
                lin11_573 = _min_211;
                float _min_212;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_212) : "f"(g00_566), "f"(cl));
                g00_566 = _min_212;
                float _min_213;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_213) : "f"(g01_567), "f"(cl));
                g01_567 = _min_213;
                float _min_214;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_214) : "f"(g10_568), "f"(cl));
                g10_568 = _min_214;
                float _min_215;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_215) : "f"(g11_569), "f"(cl));
                g11_569 = _min_215;
                float _mul_320;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_320) : "f"(lin00_570), "f"(g00_566));
                float value00_574 = _mul_320;
                float _mul_321;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_321) : "f"(lin01_571), "f"(g01_567));
                float value01_575 = _mul_321;
                float _mul_322;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_322) : "f"(lin10_572), "f"(g10_568));
                float value10_576 = _mul_322;
                float _mul_323;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_323) : "f"(lin11_573), "f"(g11_569));
                float value11_577 = _mul_323;
                pair[0] = value00_574;
                pair[1] = value10_576;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token0_556 * 32 + (feature_chunk ^ local_token0_556 % 8) * 4 + feature_word] = word[0];
                pair[0] = value01_575;
                pair[1] = value11_577;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token1_557 * 32 + (feature_chunk ^ local_token1_557 % 8) * 4 + feature_word] = word[0];
                int local_token0_578 = lane_1 % 4 * 2 + 24;
                int local_token1_579 = local_token0_578 + 1;
                float x00_580 = _tmem_load_12[14];
                float x01_581 = _tmem_load_12[15];
                float x10_582 = _tmem_load_13[14];
                float x11_583 = _tmem_load_13[15];
                float _mul_324;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_324) : "f"(x00_580), "f"(-1.4426950408889634f));
                float _exp2_108 = approx_exp2(_mul_324);
                float _add_108;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_108) : "f"(1.0f), "f"(_exp2_108));
                float _rcp_108 = approx_rcp(_add_108);
                float sig00_584 = _rcp_108;
                float _mul_325;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_325) : "f"(x01_581), "f"(-1.4426950408889634f));
                float _exp2_109 = approx_exp2(_mul_325);
                float _add_109;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_109) : "f"(1.0f), "f"(_exp2_109));
                float _rcp_109 = approx_rcp(_add_109);
                float sig01_585 = _rcp_109;
                float _mul_326;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_326) : "f"(x10_582), "f"(-1.4426950408889634f));
                float _exp2_110 = approx_exp2(_mul_326);
                float _add_110;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_110) : "f"(1.0f), "f"(_exp2_110));
                float _rcp_110 = approx_rcp(_add_110);
                float sig10_586 = _rcp_110;
                float _mul_327;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_327) : "f"(x11_583), "f"(-1.4426950408889634f));
                float _exp2_111 = approx_exp2(_mul_327);
                float _add_111;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_111) : "f"(1.0f), "f"(_exp2_111));
                float _rcp_111 = approx_rcp(_add_111);
                float sig11_587 = _rcp_111;
                float _mul_328;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_328) : "f"(x00_580), "f"(sig00_584));
                float g00_588 = _mul_328;
                float _mul_329;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_329) : "f"(x01_581), "f"(sig01_585));
                float g01_589 = _mul_329;
                float _mul_330;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_330) : "f"(x10_582), "f"(sig10_586));
                float g10_590 = _mul_330;
                float _mul_331;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_331) : "f"(x11_583), "f"(sig11_587));
                float g11_591 = _mul_331;
                float lin00_592 = _tmem_load_12[12];
                float lin01_593 = _tmem_load_12[13];
                float lin10_594 = _tmem_load_13[12];
                float lin11_595 = _tmem_load_13[13];
                float _max_108;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_108) : "f"(lin00_592), "f"(neg_cl));
                float _min_216;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_216) : "f"(_max_108), "f"(cl));
                lin00_592 = _min_216;
                float _max_109;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_109) : "f"(lin01_593), "f"(neg_cl));
                float _min_217;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_217) : "f"(_max_109), "f"(cl));
                lin01_593 = _min_217;
                float _max_110;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_110) : "f"(lin10_594), "f"(neg_cl));
                float _min_218;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_218) : "f"(_max_110), "f"(cl));
                lin10_594 = _min_218;
                float _max_111;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_111) : "f"(lin11_595), "f"(neg_cl));
                float _min_219;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_219) : "f"(_max_111), "f"(cl));
                lin11_595 = _min_219;
                float _min_220;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_220) : "f"(g00_588), "f"(cl));
                g00_588 = _min_220;
                float _min_221;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_221) : "f"(g01_589), "f"(cl));
                g01_589 = _min_221;
                float _min_222;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_222) : "f"(g10_590), "f"(cl));
                g10_590 = _min_222;
                float _min_223;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_223) : "f"(g11_591), "f"(cl));
                g11_591 = _min_223;
                float _mul_332;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_332) : "f"(lin00_592), "f"(g00_588));
                float value00_596 = _mul_332;
                float _mul_333;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_333) : "f"(lin01_593), "f"(g01_589));
                float value01_597 = _mul_333;
                float _mul_334;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_334) : "f"(lin10_594), "f"(g10_590));
                float value10_598 = _mul_334;
                float _mul_335;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_335) : "f"(lin11_595), "f"(g11_591));
                float value11_599 = _mul_335;
                pair[0] = value00_596;
                pair[1] = value10_598;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token0_578 * 32 + (feature_chunk ^ local_token0_578 % 8) * 4 + feature_word] = word[0];
                pair[0] = value01_597;
                pair[1] = value11_599;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token1_579 * 32 + (feature_chunk ^ local_token1_579 % 8) * 4 + feature_word] = word[0];
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 7, 128;" ::: "memory");
                if (warp == 0) {
                    if (elect_sync()) {
                        int padding_rows_6 = (256 - valid_rows % 256) % 256;
                        tma_store_4d((&C), m_tile * 64, padding_rows_6 + 192, 1073741824, n_tile * 256 - (unsigned int)padding_rows_6 + 1073741824, epi_staging_addr);
                    }
                }
                asm volatile("cp.async.bulk.commit_group;");
                asm volatile("barrier.sync 7, 128;" ::: "memory");
                int acc_offset_600 = acc_stage * 256 + 224;
                float _tmem_load_14[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_14[15]))
                    : "r"(taddr + (unsigned int)acc_offset_600));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float _tmem_load_15[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_15[15]))
                    : "r"(taddr + 1048576 + (unsigned int)acc_offset_600));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                asm volatile("cp.async.bulk.wait_group.read 0;");
                asm volatile("barrier.sync 7, 128;" ::: "memory");
                int local_token0_601 = lane_1 % 4 * 2;
                int local_token1_602 = local_token0_601 + 1;
                float x00_603 = _tmem_load_14[2];
                float x01_604 = _tmem_load_14[3];
                float x10_605 = _tmem_load_15[2];
                float x11_606 = _tmem_load_15[3];
                float _mul_336;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_336) : "f"(x00_603), "f"(-1.4426950408889634f));
                float _exp2_112 = approx_exp2(_mul_336);
                float _add_112;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_112) : "f"(1.0f), "f"(_exp2_112));
                float _rcp_112 = approx_rcp(_add_112);
                float sig00_607 = _rcp_112;
                float _mul_337;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_337) : "f"(x01_604), "f"(-1.4426950408889634f));
                float _exp2_113 = approx_exp2(_mul_337);
                float _add_113;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_113) : "f"(1.0f), "f"(_exp2_113));
                float _rcp_113 = approx_rcp(_add_113);
                float sig01_608 = _rcp_113;
                float _mul_338;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_338) : "f"(x10_605), "f"(-1.4426950408889634f));
                float _exp2_114 = approx_exp2(_mul_338);
                float _add_114;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_114) : "f"(1.0f), "f"(_exp2_114));
                float _rcp_114 = approx_rcp(_add_114);
                float sig10_609 = _rcp_114;
                float _mul_339;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_339) : "f"(x11_606), "f"(-1.4426950408889634f));
                float _exp2_115 = approx_exp2(_mul_339);
                float _add_115;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_115) : "f"(1.0f), "f"(_exp2_115));
                float _rcp_115 = approx_rcp(_add_115);
                float sig11_610 = _rcp_115;
                float _mul_340;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_340) : "f"(x00_603), "f"(sig00_607));
                float g00_611 = _mul_340;
                float _mul_341;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_341) : "f"(x01_604), "f"(sig01_608));
                float g01_612 = _mul_341;
                float _mul_342;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_342) : "f"(x10_605), "f"(sig10_609));
                float g10_613 = _mul_342;
                float _mul_343;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_343) : "f"(x11_606), "f"(sig11_610));
                float g11_614 = _mul_343;
                float lin00_615 = _tmem_load_14[0];
                float lin01_616 = _tmem_load_14[1];
                float lin10_617 = _tmem_load_15[0];
                float lin11_618 = _tmem_load_15[1];
                float _max_112;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_112) : "f"(lin00_615), "f"(neg_cl));
                float _min_224;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_224) : "f"(_max_112), "f"(cl));
                lin00_615 = _min_224;
                float _max_113;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_113) : "f"(lin01_616), "f"(neg_cl));
                float _min_225;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_225) : "f"(_max_113), "f"(cl));
                lin01_616 = _min_225;
                float _max_114;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_114) : "f"(lin10_617), "f"(neg_cl));
                float _min_226;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_226) : "f"(_max_114), "f"(cl));
                lin10_617 = _min_226;
                float _max_115;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_115) : "f"(lin11_618), "f"(neg_cl));
                float _min_227;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_227) : "f"(_max_115), "f"(cl));
                lin11_618 = _min_227;
                float _min_228;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_228) : "f"(g00_611), "f"(cl));
                g00_611 = _min_228;
                float _min_229;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_229) : "f"(g01_612), "f"(cl));
                g01_612 = _min_229;
                float _min_230;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_230) : "f"(g10_613), "f"(cl));
                g10_613 = _min_230;
                float _min_231;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_231) : "f"(g11_614), "f"(cl));
                g11_614 = _min_231;
                float _mul_344;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_344) : "f"(lin00_615), "f"(g00_611));
                float value00_619 = _mul_344;
                float _mul_345;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_345) : "f"(lin01_616), "f"(g01_612));
                float value01_620 = _mul_345;
                float _mul_346;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_346) : "f"(lin10_617), "f"(g10_613));
                float value10_621 = _mul_346;
                float _mul_347;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_347) : "f"(lin11_618), "f"(g11_614));
                float value11_622 = _mul_347;
                pair[0] = value00_619;
                pair[1] = value10_621;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token0_601 * 32 + (feature_chunk ^ local_token0_601 % 8) * 4 + feature_word] = word[0];
                pair[0] = value01_620;
                pair[1] = value11_622;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token1_602 * 32 + (feature_chunk ^ local_token1_602 % 8) * 4 + feature_word] = word[0];
                int local_token0_623 = lane_1 % 4 * 2 + 8;
                int local_token1_624 = local_token0_623 + 1;
                float x00_625 = _tmem_load_14[6];
                float x01_626 = _tmem_load_14[7];
                float x10_627 = _tmem_load_15[6];
                float x11_628 = _tmem_load_15[7];
                float _mul_348;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_348) : "f"(x00_625), "f"(-1.4426950408889634f));
                float _exp2_116 = approx_exp2(_mul_348);
                float _add_116;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_116) : "f"(1.0f), "f"(_exp2_116));
                float _rcp_116 = approx_rcp(_add_116);
                float sig00_629 = _rcp_116;
                float _mul_349;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_349) : "f"(x01_626), "f"(-1.4426950408889634f));
                float _exp2_117 = approx_exp2(_mul_349);
                float _add_117;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_117) : "f"(1.0f), "f"(_exp2_117));
                float _rcp_117 = approx_rcp(_add_117);
                float sig01_630 = _rcp_117;
                float _mul_350;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_350) : "f"(x10_627), "f"(-1.4426950408889634f));
                float _exp2_118 = approx_exp2(_mul_350);
                float _add_118;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_118) : "f"(1.0f), "f"(_exp2_118));
                float _rcp_118 = approx_rcp(_add_118);
                float sig10_631 = _rcp_118;
                float _mul_351;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_351) : "f"(x11_628), "f"(-1.4426950408889634f));
                float _exp2_119 = approx_exp2(_mul_351);
                float _add_119;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_119) : "f"(1.0f), "f"(_exp2_119));
                float _rcp_119 = approx_rcp(_add_119);
                float sig11_632 = _rcp_119;
                float _mul_352;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_352) : "f"(x00_625), "f"(sig00_629));
                float g00_633 = _mul_352;
                float _mul_353;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_353) : "f"(x01_626), "f"(sig01_630));
                float g01_634 = _mul_353;
                float _mul_354;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_354) : "f"(x10_627), "f"(sig10_631));
                float g10_635 = _mul_354;
                float _mul_355;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_355) : "f"(x11_628), "f"(sig11_632));
                float g11_636 = _mul_355;
                float lin00_637 = _tmem_load_14[4];
                float lin01_638 = _tmem_load_14[5];
                float lin10_639 = _tmem_load_15[4];
                float lin11_640 = _tmem_load_15[5];
                float _max_116;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_116) : "f"(lin00_637), "f"(neg_cl));
                float _min_232;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_232) : "f"(_max_116), "f"(cl));
                lin00_637 = _min_232;
                float _max_117;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_117) : "f"(lin01_638), "f"(neg_cl));
                float _min_233;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_233) : "f"(_max_117), "f"(cl));
                lin01_638 = _min_233;
                float _max_118;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_118) : "f"(lin10_639), "f"(neg_cl));
                float _min_234;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_234) : "f"(_max_118), "f"(cl));
                lin10_639 = _min_234;
                float _max_119;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_119) : "f"(lin11_640), "f"(neg_cl));
                float _min_235;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_235) : "f"(_max_119), "f"(cl));
                lin11_640 = _min_235;
                float _min_236;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_236) : "f"(g00_633), "f"(cl));
                g00_633 = _min_236;
                float _min_237;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_237) : "f"(g01_634), "f"(cl));
                g01_634 = _min_237;
                float _min_238;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_238) : "f"(g10_635), "f"(cl));
                g10_635 = _min_238;
                float _min_239;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_239) : "f"(g11_636), "f"(cl));
                g11_636 = _min_239;
                float _mul_356;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_356) : "f"(lin00_637), "f"(g00_633));
                float value00_641 = _mul_356;
                float _mul_357;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_357) : "f"(lin01_638), "f"(g01_634));
                float value01_642 = _mul_357;
                float _mul_358;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_358) : "f"(lin10_639), "f"(g10_635));
                float value10_643 = _mul_358;
                float _mul_359;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_359) : "f"(lin11_640), "f"(g11_636));
                float value11_644 = _mul_359;
                pair[0] = value00_641;
                pair[1] = value10_643;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token0_623 * 32 + (feature_chunk ^ local_token0_623 % 8) * 4 + feature_word] = word[0];
                pair[0] = value01_642;
                pair[1] = value11_644;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token1_624 * 32 + (feature_chunk ^ local_token1_624 % 8) * 4 + feature_word] = word[0];
                int local_token0_645 = lane_1 % 4 * 2 + 16;
                int local_token1_646 = local_token0_645 + 1;
                float x00_647 = _tmem_load_14[10];
                float x01_648 = _tmem_load_14[11];
                float x10_649 = _tmem_load_15[10];
                float x11_650 = _tmem_load_15[11];
                float _mul_360;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_360) : "f"(x00_647), "f"(-1.4426950408889634f));
                float _exp2_120 = approx_exp2(_mul_360);
                float _add_120;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_120) : "f"(1.0f), "f"(_exp2_120));
                float _rcp_120 = approx_rcp(_add_120);
                float sig00_651 = _rcp_120;
                float _mul_361;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_361) : "f"(x01_648), "f"(-1.4426950408889634f));
                float _exp2_121 = approx_exp2(_mul_361);
                float _add_121;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_121) : "f"(1.0f), "f"(_exp2_121));
                float _rcp_121 = approx_rcp(_add_121);
                float sig01_652 = _rcp_121;
                float _mul_362;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_362) : "f"(x10_649), "f"(-1.4426950408889634f));
                float _exp2_122 = approx_exp2(_mul_362);
                float _add_122;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_122) : "f"(1.0f), "f"(_exp2_122));
                float _rcp_122 = approx_rcp(_add_122);
                float sig10_653 = _rcp_122;
                float _mul_363;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_363) : "f"(x11_650), "f"(-1.4426950408889634f));
                float _exp2_123 = approx_exp2(_mul_363);
                float _add_123;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_123) : "f"(1.0f), "f"(_exp2_123));
                float _rcp_123 = approx_rcp(_add_123);
                float sig11_654 = _rcp_123;
                float _mul_364;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_364) : "f"(x00_647), "f"(sig00_651));
                float g00_655 = _mul_364;
                float _mul_365;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_365) : "f"(x01_648), "f"(sig01_652));
                float g01_656 = _mul_365;
                float _mul_366;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_366) : "f"(x10_649), "f"(sig10_653));
                float g10_657 = _mul_366;
                float _mul_367;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_367) : "f"(x11_650), "f"(sig11_654));
                float g11_658 = _mul_367;
                float lin00_659 = _tmem_load_14[8];
                float lin01_660 = _tmem_load_14[9];
                float lin10_661 = _tmem_load_15[8];
                float lin11_662 = _tmem_load_15[9];
                float _max_120;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_120) : "f"(lin00_659), "f"(neg_cl));
                float _min_240;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_240) : "f"(_max_120), "f"(cl));
                lin00_659 = _min_240;
                float _max_121;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_121) : "f"(lin01_660), "f"(neg_cl));
                float _min_241;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_241) : "f"(_max_121), "f"(cl));
                lin01_660 = _min_241;
                float _max_122;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_122) : "f"(lin10_661), "f"(neg_cl));
                float _min_242;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_242) : "f"(_max_122), "f"(cl));
                lin10_661 = _min_242;
                float _max_123;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_123) : "f"(lin11_662), "f"(neg_cl));
                float _min_243;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_243) : "f"(_max_123), "f"(cl));
                lin11_662 = _min_243;
                float _min_244;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_244) : "f"(g00_655), "f"(cl));
                g00_655 = _min_244;
                float _min_245;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_245) : "f"(g01_656), "f"(cl));
                g01_656 = _min_245;
                float _min_246;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_246) : "f"(g10_657), "f"(cl));
                g10_657 = _min_246;
                float _min_247;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_247) : "f"(g11_658), "f"(cl));
                g11_658 = _min_247;
                float _mul_368;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_368) : "f"(lin00_659), "f"(g00_655));
                float value00_663 = _mul_368;
                float _mul_369;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_369) : "f"(lin01_660), "f"(g01_656));
                float value01_664 = _mul_369;
                float _mul_370;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_370) : "f"(lin10_661), "f"(g10_657));
                float value10_665 = _mul_370;
                float _mul_371;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_371) : "f"(lin11_662), "f"(g11_658));
                float value11_666 = _mul_371;
                pair[0] = value00_663;
                pair[1] = value10_665;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token0_645 * 32 + (feature_chunk ^ local_token0_645 % 8) * 4 + feature_word] = word[0];
                pair[0] = value01_664;
                pair[1] = value11_666;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token1_646 * 32 + (feature_chunk ^ local_token1_646 % 8) * 4 + feature_word] = word[0];
                int local_token0_667 = lane_1 % 4 * 2 + 24;
                int local_token1_668 = local_token0_667 + 1;
                float x00_669 = _tmem_load_14[14];
                float x01_670 = _tmem_load_14[15];
                float x10_671 = _tmem_load_15[14];
                float x11_672 = _tmem_load_15[15];
                float _mul_372;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_372) : "f"(x00_669), "f"(-1.4426950408889634f));
                float _exp2_124 = approx_exp2(_mul_372);
                float _add_124;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_124) : "f"(1.0f), "f"(_exp2_124));
                float _rcp_124 = approx_rcp(_add_124);
                float sig00_673 = _rcp_124;
                float _mul_373;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_373) : "f"(x01_670), "f"(-1.4426950408889634f));
                float _exp2_125 = approx_exp2(_mul_373);
                float _add_125;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_125) : "f"(1.0f), "f"(_exp2_125));
                float _rcp_125 = approx_rcp(_add_125);
                float sig01_674 = _rcp_125;
                float _mul_374;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_374) : "f"(x10_671), "f"(-1.4426950408889634f));
                float _exp2_126 = approx_exp2(_mul_374);
                float _add_126;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_126) : "f"(1.0f), "f"(_exp2_126));
                float _rcp_126 = approx_rcp(_add_126);
                float sig10_675 = _rcp_126;
                float _mul_375;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_375) : "f"(x11_672), "f"(-1.4426950408889634f));
                float _exp2_127 = approx_exp2(_mul_375);
                float _add_127;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_127) : "f"(1.0f), "f"(_exp2_127));
                float _rcp_127 = approx_rcp(_add_127);
                float sig11_676 = _rcp_127;
                float _mul_376;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_376) : "f"(x00_669), "f"(sig00_673));
                float g00_677 = _mul_376;
                float _mul_377;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_377) : "f"(x01_670), "f"(sig01_674));
                float g01_678 = _mul_377;
                float _mul_378;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_378) : "f"(x10_671), "f"(sig10_675));
                float g10_679 = _mul_378;
                float _mul_379;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_379) : "f"(x11_672), "f"(sig11_676));
                float g11_680 = _mul_379;
                float lin00_681 = _tmem_load_14[12];
                float lin01_682 = _tmem_load_14[13];
                float lin10_683 = _tmem_load_15[12];
                float lin11_684 = _tmem_load_15[13];
                float _max_124;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_124) : "f"(lin00_681), "f"(neg_cl));
                float _min_248;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_248) : "f"(_max_124), "f"(cl));
                lin00_681 = _min_248;
                float _max_125;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_125) : "f"(lin01_682), "f"(neg_cl));
                float _min_249;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_249) : "f"(_max_125), "f"(cl));
                lin01_682 = _min_249;
                float _max_126;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_126) : "f"(lin10_683), "f"(neg_cl));
                float _min_250;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_250) : "f"(_max_126), "f"(cl));
                lin10_683 = _min_250;
                float _max_127;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_127) : "f"(lin11_684), "f"(neg_cl));
                float _min_251;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_251) : "f"(_max_127), "f"(cl));
                lin11_684 = _min_251;
                float _min_252;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_252) : "f"(g00_677), "f"(cl));
                g00_677 = _min_252;
                float _min_253;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_253) : "f"(g01_678), "f"(cl));
                g01_678 = _min_253;
                float _min_254;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_254) : "f"(g10_679), "f"(cl));
                g10_679 = _min_254;
                float _min_255;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_255) : "f"(g11_680), "f"(cl));
                g11_680 = _min_255;
                float _mul_380;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_380) : "f"(lin00_681), "f"(g00_677));
                float value00_685 = _mul_380;
                float _mul_381;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_381) : "f"(lin01_682), "f"(g01_678));
                float value01_686 = _mul_381;
                float _mul_382;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_382) : "f"(lin10_683), "f"(g10_679));
                float value10_687 = _mul_382;
                float _mul_383;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_383) : "f"(lin11_684), "f"(g11_680));
                float value11_688 = _mul_383;
                pair[0] = value00_685;
                pair[1] = value10_687;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token0_667 * 32 + (feature_chunk ^ local_token0_667 % 8) * 4 + feature_word] = word[0];
                pair[0] = value01_686;
                pair[1] = value11_688;
                #pragma unroll
                for (int _lp = 0; _lp < 1; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                    word[_lp] = *(uint32_t*)&_bf2;
                }
                epi_staging_u32[local_token1_668 * 32 + (feature_chunk ^ local_token1_668 % 8) * 4 + feature_word] = word[0];
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 7, 128;" ::: "memory");
                if (warp == 0) {
                    if (elect_sync()) {
                        int padding_rows_7 = (256 - valid_rows % 256) % 256;
                        tma_store_4d((&C), m_tile * 64, padding_rows_7 + 224, 1073741824, n_tile * 256 - (unsigned int)padding_rows_7 + 1073741824, epi_staging_addr);
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
        asm volatile("setmaxnreg.dec.sync.aligned.u32 80;");
        { // load_b_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int stage = 0;
            unsigned int work_stage_1 = 0;
            unsigned int m_tile_1 = blockIdx.x;
            unsigned int n_tile_1 = blockIdx.y;
            int warp_local = warp - 4;
            int route_base = 0;
            int routed[32];
            unsigned int cta_mask = 1 << cta_rank;
            unsigned int _phase_k_done = 1;
            unsigned int _phase_work_full_1 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_1 = 0; _tile_iter_1 < grid_m / 2 * grid_n; _tile_iter_1++) {
                if (m_tile_1 >= (unsigned int)grid_m || n_tile_1 >= (unsigned int)num_non_exiting_ctas[0]) {
                    break;
                }
                route_base = n_tile_1 * 256 + (unsigned int)(cta_rank * 128) + (unsigned int)(warp_local * 4);
                for (int row = 0; row < 4; row++) {
                    routed[row] = route_map[route_base + row];
                }
                route_base = n_tile_1 * 256 + (unsigned int)(cta_rank * 128) + (unsigned int)((4 + warp_local) * 4);
                for (int row_1 = 0; row_1 < 4; row_1++) {
                    routed[4 + row_1] = route_map[route_base + row_1];
                }
                route_base = n_tile_1 * 256 + (unsigned int)(cta_rank * 128) + (unsigned int)((8 + warp_local) * 4);
                for (int row_2 = 0; row_2 < 4; row_2++) {
                    routed[8 + row_2] = route_map[route_base + row_2];
                }
                route_base = n_tile_1 * 256 + (unsigned int)(cta_rank * 128) + (unsigned int)((12 + warp_local) * 4);
                for (int row_3 = 0; row_3 < 4; row_3++) {
                    routed[12 + row_3] = route_map[route_base + row_3];
                }
                route_base = n_tile_1 * 256 + (unsigned int)(cta_rank * 128) + (unsigned int)((16 + warp_local) * 4);
                for (int row_4 = 0; row_4 < 4; row_4++) {
                    routed[16 + row_4] = route_map[route_base + row_4];
                }
                route_base = n_tile_1 * 256 + (unsigned int)(cta_rank * 128) + (unsigned int)((20 + warp_local) * 4);
                for (int row_5 = 0; row_5 < 4; row_5++) {
                    routed[20 + row_5] = route_map[route_base + row_5];
                }
                route_base = n_tile_1 * 256 + (unsigned int)(cta_rank * 128) + (unsigned int)((24 + warp_local) * 4);
                for (int row_6 = 0; row_6 < 4; row_6++) {
                    routed[24 + row_6] = route_map[route_base + row_6];
                }
                route_base = n_tile_1 * 256 + (unsigned int)(cta_rank * 128) + (unsigned int)((28 + warp_local) * 4);
                for (int row_7 = 0; row_7 < 4; row_7++) {
                    routed[28 + row_7] = route_map[route_base + row_7];
                }
                #pragma unroll 1
                for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                    mbarrier_wait(k_done_addr + (stage) * 8, _phase_k_done);
                    if (elect_sync()) {
                        tma_gather4_gmem2smem_mc_cta2(smem_b_addr + stage * 16384 + (unsigned int)(warp_local * 512), (&B), iter_k * 64, routed[0], routed[1], routed[2], routed[3], ((b_full_addr + (stage) * 8) & 0xFEFFFFFF), cta_mask);
                        tma_gather4_gmem2smem_mc_cta2(smem_b_addr + stage * 16384 + (unsigned int)((4 + warp_local) * 512), (&B), iter_k * 64, routed[4], routed[5], routed[6], routed[7], ((b_full_addr + (stage) * 8) & 0xFEFFFFFF), cta_mask);
                        tma_gather4_gmem2smem_mc_cta2(smem_b_addr + stage * 16384 + (unsigned int)((8 + warp_local) * 512), (&B), iter_k * 64, routed[8], routed[9], routed[10], routed[11], ((b_full_addr + (stage) * 8) & 0xFEFFFFFF), cta_mask);
                        tma_gather4_gmem2smem_mc_cta2(smem_b_addr + stage * 16384 + (unsigned int)((12 + warp_local) * 512), (&B), iter_k * 64, routed[12], routed[13], routed[14], routed[15], ((b_full_addr + (stage) * 8) & 0xFEFFFFFF), cta_mask);
                        tma_gather4_gmem2smem_mc_cta2(smem_b_addr + stage * 16384 + (unsigned int)((16 + warp_local) * 512), (&B), iter_k * 64, routed[16], routed[17], routed[18], routed[19], ((b_full_addr + (stage) * 8) & 0xFEFFFFFF), cta_mask);
                        tma_gather4_gmem2smem_mc_cta2(smem_b_addr + stage * 16384 + (unsigned int)((20 + warp_local) * 512), (&B), iter_k * 64, routed[20], routed[21], routed[22], routed[23], ((b_full_addr + (stage) * 8) & 0xFEFFFFFF), cta_mask);
                        tma_gather4_gmem2smem_mc_cta2(smem_b_addr + stage * 16384 + (unsigned int)((24 + warp_local) * 512), (&B), iter_k * 64, routed[24], routed[25], routed[26], routed[27], ((b_full_addr + (stage) * 8) & 0xFEFFFFFF), cta_mask);
                        tma_gather4_gmem2smem_mc_cta2(smem_b_addr + stage * 16384 + (unsigned int)((28 + warp_local) * 512), (&B), iter_k * 64, routed[28], routed[29], routed[30], routed[31], ((b_full_addr + (stage) * 8) & 0xFEFFFFFF), cta_mask);
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
                        int _mma_b_lo_0 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_2) * 1024;
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
                    "mov.b32 id, 272630928;\n\t"
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
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_accum + (acc_stage_1 * 256))), "r"(((((iter_k_2 == 0) ? 1 : 0)) ? 0 : 1)));
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
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
