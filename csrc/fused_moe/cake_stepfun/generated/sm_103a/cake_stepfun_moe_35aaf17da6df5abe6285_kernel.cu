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

__global__ __launch_bounds__(384, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_cake_stepfun_moe_35aaf17da6df5abe6285(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap C, int* __restrict__ route_map, int* __restrict__ tile_expert, int* __restrict__ tile_mn_limit, int* __restrict__ num_non_exiting_ctas, int M_out, int K, int grid_m, int grid_n, int K_tiles, float* __restrict__ clamp_limit)
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
                float neg_cl = -cl;
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
                float _exp2_0 = approx_exp2((-x00) * 1.4426950408889634f);
                float _rcp_0 = approx_rcp(1.0f + _exp2_0);
                float sig00 = _rcp_0;
                float _exp2_1 = approx_exp2((-x01) * 1.4426950408889634f);
                float _rcp_1 = approx_rcp(1.0f + _exp2_1);
                float sig01 = _rcp_1;
                float _exp2_2 = approx_exp2((-x10) * 1.4426950408889634f);
                float _rcp_2 = approx_rcp(1.0f + _exp2_2);
                float sig10 = _rcp_2;
                float _exp2_3 = approx_exp2((-x11) * 1.4426950408889634f);
                float _rcp_3 = approx_rcp(1.0f + _exp2_3);
                float sig11 = _rcp_3;
                float g00 = x00 * sig00;
                float g01 = x01 * sig01;
                float g10 = x10 * sig10;
                float g11 = x11 * sig11;
                float lin00 = _tmem_load_0[0];
                float lin01 = _tmem_load_0[1];
                float lin10 = _tmem_load_1[0];
                float lin11 = _tmem_load_1[1];
                float _max_0 = max_noftz(lin00, neg_cl);
                float _min_0 = fminf(_max_0, cl);
                lin00 = _min_0;
                float _max_1 = max_noftz(lin01, neg_cl);
                float _min_1 = fminf(_max_1, cl);
                lin01 = _min_1;
                float _max_2 = max_noftz(lin10, neg_cl);
                float _min_2 = fminf(_max_2, cl);
                lin10 = _min_2;
                float _max_3 = max_noftz(lin11, neg_cl);
                float _min_3 = fminf(_max_3, cl);
                lin11 = _min_3;
                float _min_4 = fminf(g00, cl);
                g00 = _min_4;
                float _min_5 = fminf(g01, cl);
                g01 = _min_5;
                float _min_6 = fminf(g10, cl);
                g10 = _min_6;
                float _min_7 = fminf(g11, cl);
                g11 = _min_7;
                float value00 = lin00 * g00;
                float value01 = lin01 * g01;
                float value10 = lin10 * g10;
                float value11 = lin11 * g11;
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
                float _exp2_4 = approx_exp2((-x00_2) * 1.4426950408889634f);
                float _rcp_4 = approx_rcp(1.0f + _exp2_4);
                float sig00_6 = _rcp_4;
                float _exp2_5 = approx_exp2((-x01_3) * 1.4426950408889634f);
                float _rcp_5 = approx_rcp(1.0f + _exp2_5);
                float sig01_7 = _rcp_5;
                float _exp2_6 = approx_exp2((-x10_4) * 1.4426950408889634f);
                float _rcp_6 = approx_rcp(1.0f + _exp2_6);
                float sig10_8 = _rcp_6;
                float _exp2_7 = approx_exp2((-x11_5) * 1.4426950408889634f);
                float _rcp_7 = approx_rcp(1.0f + _exp2_7);
                float sig11_9 = _rcp_7;
                float g00_10 = x00_2 * sig00_6;
                float g01_11 = x01_3 * sig01_7;
                float g10_12 = x10_4 * sig10_8;
                float g11_13 = x11_5 * sig11_9;
                float lin00_14 = _tmem_load_0[4];
                float lin01_15 = _tmem_load_0[5];
                float lin10_16 = _tmem_load_1[4];
                float lin11_17 = _tmem_load_1[5];
                float _max_4 = max_noftz(lin00_14, neg_cl);
                float _min_8 = fminf(_max_4, cl);
                lin00_14 = _min_8;
                float _max_5 = max_noftz(lin01_15, neg_cl);
                float _min_9 = fminf(_max_5, cl);
                lin01_15 = _min_9;
                float _max_6 = max_noftz(lin10_16, neg_cl);
                float _min_10 = fminf(_max_6, cl);
                lin10_16 = _min_10;
                float _max_7 = max_noftz(lin11_17, neg_cl);
                float _min_11 = fminf(_max_7, cl);
                lin11_17 = _min_11;
                float _min_12 = fminf(g00_10, cl);
                g00_10 = _min_12;
                float _min_13 = fminf(g01_11, cl);
                g01_11 = _min_13;
                float _min_14 = fminf(g10_12, cl);
                g10_12 = _min_14;
                float _min_15 = fminf(g11_13, cl);
                g11_13 = _min_15;
                float value00_18 = lin00_14 * g00_10;
                float value01_19 = lin01_15 * g01_11;
                float value10_20 = lin10_16 * g10_12;
                float value11_21 = lin11_17 * g11_13;
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
                float _exp2_8 = approx_exp2((-x00_24) * 1.4426950408889634f);
                float _rcp_8 = approx_rcp(1.0f + _exp2_8);
                float sig00_28 = _rcp_8;
                float _exp2_9 = approx_exp2((-x01_25) * 1.4426950408889634f);
                float _rcp_9 = approx_rcp(1.0f + _exp2_9);
                float sig01_29 = _rcp_9;
                float _exp2_10 = approx_exp2((-x10_26) * 1.4426950408889634f);
                float _rcp_10 = approx_rcp(1.0f + _exp2_10);
                float sig10_30 = _rcp_10;
                float _exp2_11 = approx_exp2((-x11_27) * 1.4426950408889634f);
                float _rcp_11 = approx_rcp(1.0f + _exp2_11);
                float sig11_31 = _rcp_11;
                float g00_32 = x00_24 * sig00_28;
                float g01_33 = x01_25 * sig01_29;
                float g10_34 = x10_26 * sig10_30;
                float g11_35 = x11_27 * sig11_31;
                float lin00_36 = _tmem_load_0[8];
                float lin01_37 = _tmem_load_0[9];
                float lin10_38 = _tmem_load_1[8];
                float lin11_39 = _tmem_load_1[9];
                float _max_8 = max_noftz(lin00_36, neg_cl);
                float _min_16 = fminf(_max_8, cl);
                lin00_36 = _min_16;
                float _max_9 = max_noftz(lin01_37, neg_cl);
                float _min_17 = fminf(_max_9, cl);
                lin01_37 = _min_17;
                float _max_10 = max_noftz(lin10_38, neg_cl);
                float _min_18 = fminf(_max_10, cl);
                lin10_38 = _min_18;
                float _max_11 = max_noftz(lin11_39, neg_cl);
                float _min_19 = fminf(_max_11, cl);
                lin11_39 = _min_19;
                float _min_20 = fminf(g00_32, cl);
                g00_32 = _min_20;
                float _min_21 = fminf(g01_33, cl);
                g01_33 = _min_21;
                float _min_22 = fminf(g10_34, cl);
                g10_34 = _min_22;
                float _min_23 = fminf(g11_35, cl);
                g11_35 = _min_23;
                float value00_40 = lin00_36 * g00_32;
                float value01_41 = lin01_37 * g01_33;
                float value10_42 = lin10_38 * g10_34;
                float value11_43 = lin11_39 * g11_35;
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
                float _exp2_12 = approx_exp2((-x00_46) * 1.4426950408889634f);
                float _rcp_12 = approx_rcp(1.0f + _exp2_12);
                float sig00_50 = _rcp_12;
                float _exp2_13 = approx_exp2((-x01_47) * 1.4426950408889634f);
                float _rcp_13 = approx_rcp(1.0f + _exp2_13);
                float sig01_51 = _rcp_13;
                float _exp2_14 = approx_exp2((-x10_48) * 1.4426950408889634f);
                float _rcp_14 = approx_rcp(1.0f + _exp2_14);
                float sig10_52 = _rcp_14;
                float _exp2_15 = approx_exp2((-x11_49) * 1.4426950408889634f);
                float _rcp_15 = approx_rcp(1.0f + _exp2_15);
                float sig11_53 = _rcp_15;
                float g00_54 = x00_46 * sig00_50;
                float g01_55 = x01_47 * sig01_51;
                float g10_56 = x10_48 * sig10_52;
                float g11_57 = x11_49 * sig11_53;
                float lin00_58 = _tmem_load_0[12];
                float lin01_59 = _tmem_load_0[13];
                float lin10_60 = _tmem_load_1[12];
                float lin11_61 = _tmem_load_1[13];
                float _max_12 = max_noftz(lin00_58, neg_cl);
                float _min_24 = fminf(_max_12, cl);
                lin00_58 = _min_24;
                float _max_13 = max_noftz(lin01_59, neg_cl);
                float _min_25 = fminf(_max_13, cl);
                lin01_59 = _min_25;
                float _max_14 = max_noftz(lin10_60, neg_cl);
                float _min_26 = fminf(_max_14, cl);
                lin10_60 = _min_26;
                float _max_15 = max_noftz(lin11_61, neg_cl);
                float _min_27 = fminf(_max_15, cl);
                lin11_61 = _min_27;
                float _min_28 = fminf(g00_54, cl);
                g00_54 = _min_28;
                float _min_29 = fminf(g01_55, cl);
                g01_55 = _min_29;
                float _min_30 = fminf(g10_56, cl);
                g10_56 = _min_30;
                float _min_31 = fminf(g11_57, cl);
                g11_57 = _min_31;
                float value00_62 = lin00_58 * g00_54;
                float value01_63 = lin01_59 * g01_55;
                float value10_64 = lin10_60 * g10_56;
                float value11_65 = lin11_61 * g11_57;
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
                float _exp2_16 = approx_exp2((-x00_69) * 1.4426950408889634f);
                float _rcp_16 = approx_rcp(1.0f + _exp2_16);
                float sig00_73 = _rcp_16;
                float _exp2_17 = approx_exp2((-x01_70) * 1.4426950408889634f);
                float _rcp_17 = approx_rcp(1.0f + _exp2_17);
                float sig01_74 = _rcp_17;
                float _exp2_18 = approx_exp2((-x10_71) * 1.4426950408889634f);
                float _rcp_18 = approx_rcp(1.0f + _exp2_18);
                float sig10_75 = _rcp_18;
                float _exp2_19 = approx_exp2((-x11_72) * 1.4426950408889634f);
                float _rcp_19 = approx_rcp(1.0f + _exp2_19);
                float sig11_76 = _rcp_19;
                float g00_77 = x00_69 * sig00_73;
                float g01_78 = x01_70 * sig01_74;
                float g10_79 = x10_71 * sig10_75;
                float g11_80 = x11_72 * sig11_76;
                float lin00_81 = _tmem_load_2[0];
                float lin01_82 = _tmem_load_2[1];
                float lin10_83 = _tmem_load_3[0];
                float lin11_84 = _tmem_load_3[1];
                float _max_16 = max_noftz(lin00_81, neg_cl);
                float _min_32 = fminf(_max_16, cl);
                lin00_81 = _min_32;
                float _max_17 = max_noftz(lin01_82, neg_cl);
                float _min_33 = fminf(_max_17, cl);
                lin01_82 = _min_33;
                float _max_18 = max_noftz(lin10_83, neg_cl);
                float _min_34 = fminf(_max_18, cl);
                lin10_83 = _min_34;
                float _max_19 = max_noftz(lin11_84, neg_cl);
                float _min_35 = fminf(_max_19, cl);
                lin11_84 = _min_35;
                float _min_36 = fminf(g00_77, cl);
                g00_77 = _min_36;
                float _min_37 = fminf(g01_78, cl);
                g01_78 = _min_37;
                float _min_38 = fminf(g10_79, cl);
                g10_79 = _min_38;
                float _min_39 = fminf(g11_80, cl);
                g11_80 = _min_39;
                float value00_85 = lin00_81 * g00_77;
                float value01_86 = lin01_82 * g01_78;
                float value10_87 = lin10_83 * g10_79;
                float value11_88 = lin11_84 * g11_80;
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
                float _exp2_20 = approx_exp2((-x00_91) * 1.4426950408889634f);
                float _rcp_20 = approx_rcp(1.0f + _exp2_20);
                float sig00_95 = _rcp_20;
                float _exp2_21 = approx_exp2((-x01_92) * 1.4426950408889634f);
                float _rcp_21 = approx_rcp(1.0f + _exp2_21);
                float sig01_96 = _rcp_21;
                float _exp2_22 = approx_exp2((-x10_93) * 1.4426950408889634f);
                float _rcp_22 = approx_rcp(1.0f + _exp2_22);
                float sig10_97 = _rcp_22;
                float _exp2_23 = approx_exp2((-x11_94) * 1.4426950408889634f);
                float _rcp_23 = approx_rcp(1.0f + _exp2_23);
                float sig11_98 = _rcp_23;
                float g00_99 = x00_91 * sig00_95;
                float g01_100 = x01_92 * sig01_96;
                float g10_101 = x10_93 * sig10_97;
                float g11_102 = x11_94 * sig11_98;
                float lin00_103 = _tmem_load_2[4];
                float lin01_104 = _tmem_load_2[5];
                float lin10_105 = _tmem_load_3[4];
                float lin11_106 = _tmem_load_3[5];
                float _max_20 = max_noftz(lin00_103, neg_cl);
                float _min_40 = fminf(_max_20, cl);
                lin00_103 = _min_40;
                float _max_21 = max_noftz(lin01_104, neg_cl);
                float _min_41 = fminf(_max_21, cl);
                lin01_104 = _min_41;
                float _max_22 = max_noftz(lin10_105, neg_cl);
                float _min_42 = fminf(_max_22, cl);
                lin10_105 = _min_42;
                float _max_23 = max_noftz(lin11_106, neg_cl);
                float _min_43 = fminf(_max_23, cl);
                lin11_106 = _min_43;
                float _min_44 = fminf(g00_99, cl);
                g00_99 = _min_44;
                float _min_45 = fminf(g01_100, cl);
                g01_100 = _min_45;
                float _min_46 = fminf(g10_101, cl);
                g10_101 = _min_46;
                float _min_47 = fminf(g11_102, cl);
                g11_102 = _min_47;
                float value00_107 = lin00_103 * g00_99;
                float value01_108 = lin01_104 * g01_100;
                float value10_109 = lin10_105 * g10_101;
                float value11_110 = lin11_106 * g11_102;
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
                float _exp2_24 = approx_exp2((-x00_113) * 1.4426950408889634f);
                float _rcp_24 = approx_rcp(1.0f + _exp2_24);
                float sig00_117 = _rcp_24;
                float _exp2_25 = approx_exp2((-x01_114) * 1.4426950408889634f);
                float _rcp_25 = approx_rcp(1.0f + _exp2_25);
                float sig01_118 = _rcp_25;
                float _exp2_26 = approx_exp2((-x10_115) * 1.4426950408889634f);
                float _rcp_26 = approx_rcp(1.0f + _exp2_26);
                float sig10_119 = _rcp_26;
                float _exp2_27 = approx_exp2((-x11_116) * 1.4426950408889634f);
                float _rcp_27 = approx_rcp(1.0f + _exp2_27);
                float sig11_120 = _rcp_27;
                float g00_121 = x00_113 * sig00_117;
                float g01_122 = x01_114 * sig01_118;
                float g10_123 = x10_115 * sig10_119;
                float g11_124 = x11_116 * sig11_120;
                float lin00_125 = _tmem_load_2[8];
                float lin01_126 = _tmem_load_2[9];
                float lin10_127 = _tmem_load_3[8];
                float lin11_128 = _tmem_load_3[9];
                float _max_24 = max_noftz(lin00_125, neg_cl);
                float _min_48 = fminf(_max_24, cl);
                lin00_125 = _min_48;
                float _max_25 = max_noftz(lin01_126, neg_cl);
                float _min_49 = fminf(_max_25, cl);
                lin01_126 = _min_49;
                float _max_26 = max_noftz(lin10_127, neg_cl);
                float _min_50 = fminf(_max_26, cl);
                lin10_127 = _min_50;
                float _max_27 = max_noftz(lin11_128, neg_cl);
                float _min_51 = fminf(_max_27, cl);
                lin11_128 = _min_51;
                float _min_52 = fminf(g00_121, cl);
                g00_121 = _min_52;
                float _min_53 = fminf(g01_122, cl);
                g01_122 = _min_53;
                float _min_54 = fminf(g10_123, cl);
                g10_123 = _min_54;
                float _min_55 = fminf(g11_124, cl);
                g11_124 = _min_55;
                float value00_129 = lin00_125 * g00_121;
                float value01_130 = lin01_126 * g01_122;
                float value10_131 = lin10_127 * g10_123;
                float value11_132 = lin11_128 * g11_124;
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
                float _exp2_28 = approx_exp2((-x00_135) * 1.4426950408889634f);
                float _rcp_28 = approx_rcp(1.0f + _exp2_28);
                float sig00_139 = _rcp_28;
                float _exp2_29 = approx_exp2((-x01_136) * 1.4426950408889634f);
                float _rcp_29 = approx_rcp(1.0f + _exp2_29);
                float sig01_140 = _rcp_29;
                float _exp2_30 = approx_exp2((-x10_137) * 1.4426950408889634f);
                float _rcp_30 = approx_rcp(1.0f + _exp2_30);
                float sig10_141 = _rcp_30;
                float _exp2_31 = approx_exp2((-x11_138) * 1.4426950408889634f);
                float _rcp_31 = approx_rcp(1.0f + _exp2_31);
                float sig11_142 = _rcp_31;
                float g00_143 = x00_135 * sig00_139;
                float g01_144 = x01_136 * sig01_140;
                float g10_145 = x10_137 * sig10_141;
                float g11_146 = x11_138 * sig11_142;
                float lin00_147 = _tmem_load_2[12];
                float lin01_148 = _tmem_load_2[13];
                float lin10_149 = _tmem_load_3[12];
                float lin11_150 = _tmem_load_3[13];
                float _max_28 = max_noftz(lin00_147, neg_cl);
                float _min_56 = fminf(_max_28, cl);
                lin00_147 = _min_56;
                float _max_29 = max_noftz(lin01_148, neg_cl);
                float _min_57 = fminf(_max_29, cl);
                lin01_148 = _min_57;
                float _max_30 = max_noftz(lin10_149, neg_cl);
                float _min_58 = fminf(_max_30, cl);
                lin10_149 = _min_58;
                float _max_31 = max_noftz(lin11_150, neg_cl);
                float _min_59 = fminf(_max_31, cl);
                lin11_150 = _min_59;
                float _min_60 = fminf(g00_143, cl);
                g00_143 = _min_60;
                float _min_61 = fminf(g01_144, cl);
                g01_144 = _min_61;
                float _min_62 = fminf(g10_145, cl);
                g10_145 = _min_62;
                float _min_63 = fminf(g11_146, cl);
                g11_146 = _min_63;
                float value00_151 = lin00_147 * g00_143;
                float value01_152 = lin01_148 * g01_144;
                float value10_153 = lin10_149 * g10_145;
                float value11_154 = lin11_150 * g11_146;
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
                float _exp2_32 = approx_exp2((-x00_158) * 1.4426950408889634f);
                float _rcp_32 = approx_rcp(1.0f + _exp2_32);
                float sig00_162 = _rcp_32;
                float _exp2_33 = approx_exp2((-x01_159) * 1.4426950408889634f);
                float _rcp_33 = approx_rcp(1.0f + _exp2_33);
                float sig01_163 = _rcp_33;
                float _exp2_34 = approx_exp2((-x10_160) * 1.4426950408889634f);
                float _rcp_34 = approx_rcp(1.0f + _exp2_34);
                float sig10_164 = _rcp_34;
                float _exp2_35 = approx_exp2((-x11_161) * 1.4426950408889634f);
                float _rcp_35 = approx_rcp(1.0f + _exp2_35);
                float sig11_165 = _rcp_35;
                float g00_166 = x00_158 * sig00_162;
                float g01_167 = x01_159 * sig01_163;
                float g10_168 = x10_160 * sig10_164;
                float g11_169 = x11_161 * sig11_165;
                float lin00_170 = _tmem_load_4[0];
                float lin01_171 = _tmem_load_4[1];
                float lin10_172 = _tmem_load_5[0];
                float lin11_173 = _tmem_load_5[1];
                float _max_32 = max_noftz(lin00_170, neg_cl);
                float _min_64 = fminf(_max_32, cl);
                lin00_170 = _min_64;
                float _max_33 = max_noftz(lin01_171, neg_cl);
                float _min_65 = fminf(_max_33, cl);
                lin01_171 = _min_65;
                float _max_34 = max_noftz(lin10_172, neg_cl);
                float _min_66 = fminf(_max_34, cl);
                lin10_172 = _min_66;
                float _max_35 = max_noftz(lin11_173, neg_cl);
                float _min_67 = fminf(_max_35, cl);
                lin11_173 = _min_67;
                float _min_68 = fminf(g00_166, cl);
                g00_166 = _min_68;
                float _min_69 = fminf(g01_167, cl);
                g01_167 = _min_69;
                float _min_70 = fminf(g10_168, cl);
                g10_168 = _min_70;
                float _min_71 = fminf(g11_169, cl);
                g11_169 = _min_71;
                float value00_174 = lin00_170 * g00_166;
                float value01_175 = lin01_171 * g01_167;
                float value10_176 = lin10_172 * g10_168;
                float value11_177 = lin11_173 * g11_169;
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
                float _exp2_36 = approx_exp2((-x00_180) * 1.4426950408889634f);
                float _rcp_36 = approx_rcp(1.0f + _exp2_36);
                float sig00_184 = _rcp_36;
                float _exp2_37 = approx_exp2((-x01_181) * 1.4426950408889634f);
                float _rcp_37 = approx_rcp(1.0f + _exp2_37);
                float sig01_185 = _rcp_37;
                float _exp2_38 = approx_exp2((-x10_182) * 1.4426950408889634f);
                float _rcp_38 = approx_rcp(1.0f + _exp2_38);
                float sig10_186 = _rcp_38;
                float _exp2_39 = approx_exp2((-x11_183) * 1.4426950408889634f);
                float _rcp_39 = approx_rcp(1.0f + _exp2_39);
                float sig11_187 = _rcp_39;
                float g00_188 = x00_180 * sig00_184;
                float g01_189 = x01_181 * sig01_185;
                float g10_190 = x10_182 * sig10_186;
                float g11_191 = x11_183 * sig11_187;
                float lin00_192 = _tmem_load_4[4];
                float lin01_193 = _tmem_load_4[5];
                float lin10_194 = _tmem_load_5[4];
                float lin11_195 = _tmem_load_5[5];
                float _max_36 = max_noftz(lin00_192, neg_cl);
                float _min_72 = fminf(_max_36, cl);
                lin00_192 = _min_72;
                float _max_37 = max_noftz(lin01_193, neg_cl);
                float _min_73 = fminf(_max_37, cl);
                lin01_193 = _min_73;
                float _max_38 = max_noftz(lin10_194, neg_cl);
                float _min_74 = fminf(_max_38, cl);
                lin10_194 = _min_74;
                float _max_39 = max_noftz(lin11_195, neg_cl);
                float _min_75 = fminf(_max_39, cl);
                lin11_195 = _min_75;
                float _min_76 = fminf(g00_188, cl);
                g00_188 = _min_76;
                float _min_77 = fminf(g01_189, cl);
                g01_189 = _min_77;
                float _min_78 = fminf(g10_190, cl);
                g10_190 = _min_78;
                float _min_79 = fminf(g11_191, cl);
                g11_191 = _min_79;
                float value00_196 = lin00_192 * g00_188;
                float value01_197 = lin01_193 * g01_189;
                float value10_198 = lin10_194 * g10_190;
                float value11_199 = lin11_195 * g11_191;
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
                float _exp2_40 = approx_exp2((-x00_202) * 1.4426950408889634f);
                float _rcp_40 = approx_rcp(1.0f + _exp2_40);
                float sig00_206 = _rcp_40;
                float _exp2_41 = approx_exp2((-x01_203) * 1.4426950408889634f);
                float _rcp_41 = approx_rcp(1.0f + _exp2_41);
                float sig01_207 = _rcp_41;
                float _exp2_42 = approx_exp2((-x10_204) * 1.4426950408889634f);
                float _rcp_42 = approx_rcp(1.0f + _exp2_42);
                float sig10_208 = _rcp_42;
                float _exp2_43 = approx_exp2((-x11_205) * 1.4426950408889634f);
                float _rcp_43 = approx_rcp(1.0f + _exp2_43);
                float sig11_209 = _rcp_43;
                float g00_210 = x00_202 * sig00_206;
                float g01_211 = x01_203 * sig01_207;
                float g10_212 = x10_204 * sig10_208;
                float g11_213 = x11_205 * sig11_209;
                float lin00_214 = _tmem_load_4[8];
                float lin01_215 = _tmem_load_4[9];
                float lin10_216 = _tmem_load_5[8];
                float lin11_217 = _tmem_load_5[9];
                float _max_40 = max_noftz(lin00_214, neg_cl);
                float _min_80 = fminf(_max_40, cl);
                lin00_214 = _min_80;
                float _max_41 = max_noftz(lin01_215, neg_cl);
                float _min_81 = fminf(_max_41, cl);
                lin01_215 = _min_81;
                float _max_42 = max_noftz(lin10_216, neg_cl);
                float _min_82 = fminf(_max_42, cl);
                lin10_216 = _min_82;
                float _max_43 = max_noftz(lin11_217, neg_cl);
                float _min_83 = fminf(_max_43, cl);
                lin11_217 = _min_83;
                float _min_84 = fminf(g00_210, cl);
                g00_210 = _min_84;
                float _min_85 = fminf(g01_211, cl);
                g01_211 = _min_85;
                float _min_86 = fminf(g10_212, cl);
                g10_212 = _min_86;
                float _min_87 = fminf(g11_213, cl);
                g11_213 = _min_87;
                float value00_218 = lin00_214 * g00_210;
                float value01_219 = lin01_215 * g01_211;
                float value10_220 = lin10_216 * g10_212;
                float value11_221 = lin11_217 * g11_213;
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
                float _exp2_44 = approx_exp2((-x00_224) * 1.4426950408889634f);
                float _rcp_44 = approx_rcp(1.0f + _exp2_44);
                float sig00_228 = _rcp_44;
                float _exp2_45 = approx_exp2((-x01_225) * 1.4426950408889634f);
                float _rcp_45 = approx_rcp(1.0f + _exp2_45);
                float sig01_229 = _rcp_45;
                float _exp2_46 = approx_exp2((-x10_226) * 1.4426950408889634f);
                float _rcp_46 = approx_rcp(1.0f + _exp2_46);
                float sig10_230 = _rcp_46;
                float _exp2_47 = approx_exp2((-x11_227) * 1.4426950408889634f);
                float _rcp_47 = approx_rcp(1.0f + _exp2_47);
                float sig11_231 = _rcp_47;
                float g00_232 = x00_224 * sig00_228;
                float g01_233 = x01_225 * sig01_229;
                float g10_234 = x10_226 * sig10_230;
                float g11_235 = x11_227 * sig11_231;
                float lin00_236 = _tmem_load_4[12];
                float lin01_237 = _tmem_load_4[13];
                float lin10_238 = _tmem_load_5[12];
                float lin11_239 = _tmem_load_5[13];
                float _max_44 = max_noftz(lin00_236, neg_cl);
                float _min_88 = fminf(_max_44, cl);
                lin00_236 = _min_88;
                float _max_45 = max_noftz(lin01_237, neg_cl);
                float _min_89 = fminf(_max_45, cl);
                lin01_237 = _min_89;
                float _max_46 = max_noftz(lin10_238, neg_cl);
                float _min_90 = fminf(_max_46, cl);
                lin10_238 = _min_90;
                float _max_47 = max_noftz(lin11_239, neg_cl);
                float _min_91 = fminf(_max_47, cl);
                lin11_239 = _min_91;
                float _min_92 = fminf(g00_232, cl);
                g00_232 = _min_92;
                float _min_93 = fminf(g01_233, cl);
                g01_233 = _min_93;
                float _min_94 = fminf(g10_234, cl);
                g10_234 = _min_94;
                float _min_95 = fminf(g11_235, cl);
                g11_235 = _min_95;
                float value00_240 = lin00_236 * g00_232;
                float value01_241 = lin01_237 * g01_233;
                float value10_242 = lin10_238 * g10_234;
                float value11_243 = lin11_239 * g11_235;
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
                float _exp2_48 = approx_exp2((-x00_247) * 1.4426950408889634f);
                float _rcp_48 = approx_rcp(1.0f + _exp2_48);
                float sig00_251 = _rcp_48;
                float _exp2_49 = approx_exp2((-x01_248) * 1.4426950408889634f);
                float _rcp_49 = approx_rcp(1.0f + _exp2_49);
                float sig01_252 = _rcp_49;
                float _exp2_50 = approx_exp2((-x10_249) * 1.4426950408889634f);
                float _rcp_50 = approx_rcp(1.0f + _exp2_50);
                float sig10_253 = _rcp_50;
                float _exp2_51 = approx_exp2((-x11_250) * 1.4426950408889634f);
                float _rcp_51 = approx_rcp(1.0f + _exp2_51);
                float sig11_254 = _rcp_51;
                float g00_255 = x00_247 * sig00_251;
                float g01_256 = x01_248 * sig01_252;
                float g10_257 = x10_249 * sig10_253;
                float g11_258 = x11_250 * sig11_254;
                float lin00_259 = _tmem_load_6[0];
                float lin01_260 = _tmem_load_6[1];
                float lin10_261 = _tmem_load_7[0];
                float lin11_262 = _tmem_load_7[1];
                float _max_48 = max_noftz(lin00_259, neg_cl);
                float _min_96 = fminf(_max_48, cl);
                lin00_259 = _min_96;
                float _max_49 = max_noftz(lin01_260, neg_cl);
                float _min_97 = fminf(_max_49, cl);
                lin01_260 = _min_97;
                float _max_50 = max_noftz(lin10_261, neg_cl);
                float _min_98 = fminf(_max_50, cl);
                lin10_261 = _min_98;
                float _max_51 = max_noftz(lin11_262, neg_cl);
                float _min_99 = fminf(_max_51, cl);
                lin11_262 = _min_99;
                float _min_100 = fminf(g00_255, cl);
                g00_255 = _min_100;
                float _min_101 = fminf(g01_256, cl);
                g01_256 = _min_101;
                float _min_102 = fminf(g10_257, cl);
                g10_257 = _min_102;
                float _min_103 = fminf(g11_258, cl);
                g11_258 = _min_103;
                float value00_263 = lin00_259 * g00_255;
                float value01_264 = lin01_260 * g01_256;
                float value10_265 = lin10_261 * g10_257;
                float value11_266 = lin11_262 * g11_258;
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
                float _exp2_52 = approx_exp2((-x00_269) * 1.4426950408889634f);
                float _rcp_52 = approx_rcp(1.0f + _exp2_52);
                float sig00_273 = _rcp_52;
                float _exp2_53 = approx_exp2((-x01_270) * 1.4426950408889634f);
                float _rcp_53 = approx_rcp(1.0f + _exp2_53);
                float sig01_274 = _rcp_53;
                float _exp2_54 = approx_exp2((-x10_271) * 1.4426950408889634f);
                float _rcp_54 = approx_rcp(1.0f + _exp2_54);
                float sig10_275 = _rcp_54;
                float _exp2_55 = approx_exp2((-x11_272) * 1.4426950408889634f);
                float _rcp_55 = approx_rcp(1.0f + _exp2_55);
                float sig11_276 = _rcp_55;
                float g00_277 = x00_269 * sig00_273;
                float g01_278 = x01_270 * sig01_274;
                float g10_279 = x10_271 * sig10_275;
                float g11_280 = x11_272 * sig11_276;
                float lin00_281 = _tmem_load_6[4];
                float lin01_282 = _tmem_load_6[5];
                float lin10_283 = _tmem_load_7[4];
                float lin11_284 = _tmem_load_7[5];
                float _max_52 = max_noftz(lin00_281, neg_cl);
                float _min_104 = fminf(_max_52, cl);
                lin00_281 = _min_104;
                float _max_53 = max_noftz(lin01_282, neg_cl);
                float _min_105 = fminf(_max_53, cl);
                lin01_282 = _min_105;
                float _max_54 = max_noftz(lin10_283, neg_cl);
                float _min_106 = fminf(_max_54, cl);
                lin10_283 = _min_106;
                float _max_55 = max_noftz(lin11_284, neg_cl);
                float _min_107 = fminf(_max_55, cl);
                lin11_284 = _min_107;
                float _min_108 = fminf(g00_277, cl);
                g00_277 = _min_108;
                float _min_109 = fminf(g01_278, cl);
                g01_278 = _min_109;
                float _min_110 = fminf(g10_279, cl);
                g10_279 = _min_110;
                float _min_111 = fminf(g11_280, cl);
                g11_280 = _min_111;
                float value00_285 = lin00_281 * g00_277;
                float value01_286 = lin01_282 * g01_278;
                float value10_287 = lin10_283 * g10_279;
                float value11_288 = lin11_284 * g11_280;
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
                float _exp2_56 = approx_exp2((-x00_291) * 1.4426950408889634f);
                float _rcp_56 = approx_rcp(1.0f + _exp2_56);
                float sig00_295 = _rcp_56;
                float _exp2_57 = approx_exp2((-x01_292) * 1.4426950408889634f);
                float _rcp_57 = approx_rcp(1.0f + _exp2_57);
                float sig01_296 = _rcp_57;
                float _exp2_58 = approx_exp2((-x10_293) * 1.4426950408889634f);
                float _rcp_58 = approx_rcp(1.0f + _exp2_58);
                float sig10_297 = _rcp_58;
                float _exp2_59 = approx_exp2((-x11_294) * 1.4426950408889634f);
                float _rcp_59 = approx_rcp(1.0f + _exp2_59);
                float sig11_298 = _rcp_59;
                float g00_299 = x00_291 * sig00_295;
                float g01_300 = x01_292 * sig01_296;
                float g10_301 = x10_293 * sig10_297;
                float g11_302 = x11_294 * sig11_298;
                float lin00_303 = _tmem_load_6[8];
                float lin01_304 = _tmem_load_6[9];
                float lin10_305 = _tmem_load_7[8];
                float lin11_306 = _tmem_load_7[9];
                float _max_56 = max_noftz(lin00_303, neg_cl);
                float _min_112 = fminf(_max_56, cl);
                lin00_303 = _min_112;
                float _max_57 = max_noftz(lin01_304, neg_cl);
                float _min_113 = fminf(_max_57, cl);
                lin01_304 = _min_113;
                float _max_58 = max_noftz(lin10_305, neg_cl);
                float _min_114 = fminf(_max_58, cl);
                lin10_305 = _min_114;
                float _max_59 = max_noftz(lin11_306, neg_cl);
                float _min_115 = fminf(_max_59, cl);
                lin11_306 = _min_115;
                float _min_116 = fminf(g00_299, cl);
                g00_299 = _min_116;
                float _min_117 = fminf(g01_300, cl);
                g01_300 = _min_117;
                float _min_118 = fminf(g10_301, cl);
                g10_301 = _min_118;
                float _min_119 = fminf(g11_302, cl);
                g11_302 = _min_119;
                float value00_307 = lin00_303 * g00_299;
                float value01_308 = lin01_304 * g01_300;
                float value10_309 = lin10_305 * g10_301;
                float value11_310 = lin11_306 * g11_302;
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
                float _exp2_60 = approx_exp2((-x00_313) * 1.4426950408889634f);
                float _rcp_60 = approx_rcp(1.0f + _exp2_60);
                float sig00_317 = _rcp_60;
                float _exp2_61 = approx_exp2((-x01_314) * 1.4426950408889634f);
                float _rcp_61 = approx_rcp(1.0f + _exp2_61);
                float sig01_318 = _rcp_61;
                float _exp2_62 = approx_exp2((-x10_315) * 1.4426950408889634f);
                float _rcp_62 = approx_rcp(1.0f + _exp2_62);
                float sig10_319 = _rcp_62;
                float _exp2_63 = approx_exp2((-x11_316) * 1.4426950408889634f);
                float _rcp_63 = approx_rcp(1.0f + _exp2_63);
                float sig11_320 = _rcp_63;
                float g00_321 = x00_313 * sig00_317;
                float g01_322 = x01_314 * sig01_318;
                float g10_323 = x10_315 * sig10_319;
                float g11_324 = x11_316 * sig11_320;
                float lin00_325 = _tmem_load_6[12];
                float lin01_326 = _tmem_load_6[13];
                float lin10_327 = _tmem_load_7[12];
                float lin11_328 = _tmem_load_7[13];
                float _max_60 = max_noftz(lin00_325, neg_cl);
                float _min_120 = fminf(_max_60, cl);
                lin00_325 = _min_120;
                float _max_61 = max_noftz(lin01_326, neg_cl);
                float _min_121 = fminf(_max_61, cl);
                lin01_326 = _min_121;
                float _max_62 = max_noftz(lin10_327, neg_cl);
                float _min_122 = fminf(_max_62, cl);
                lin10_327 = _min_122;
                float _max_63 = max_noftz(lin11_328, neg_cl);
                float _min_123 = fminf(_max_63, cl);
                lin11_328 = _min_123;
                float _min_124 = fminf(g00_321, cl);
                g00_321 = _min_124;
                float _min_125 = fminf(g01_322, cl);
                g01_322 = _min_125;
                float _min_126 = fminf(g10_323, cl);
                g10_323 = _min_126;
                float _min_127 = fminf(g11_324, cl);
                g11_324 = _min_127;
                float value00_329 = lin00_325 * g00_321;
                float value01_330 = lin01_326 * g01_322;
                float value10_331 = lin10_327 * g10_323;
                float value11_332 = lin11_328 * g11_324;
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
                float _exp2_64 = approx_exp2((-x00_336) * 1.4426950408889634f);
                float _rcp_64 = approx_rcp(1.0f + _exp2_64);
                float sig00_340 = _rcp_64;
                float _exp2_65 = approx_exp2((-x01_337) * 1.4426950408889634f);
                float _rcp_65 = approx_rcp(1.0f + _exp2_65);
                float sig01_341 = _rcp_65;
                float _exp2_66 = approx_exp2((-x10_338) * 1.4426950408889634f);
                float _rcp_66 = approx_rcp(1.0f + _exp2_66);
                float sig10_342 = _rcp_66;
                float _exp2_67 = approx_exp2((-x11_339) * 1.4426950408889634f);
                float _rcp_67 = approx_rcp(1.0f + _exp2_67);
                float sig11_343 = _rcp_67;
                float g00_344 = x00_336 * sig00_340;
                float g01_345 = x01_337 * sig01_341;
                float g10_346 = x10_338 * sig10_342;
                float g11_347 = x11_339 * sig11_343;
                float lin00_348 = _tmem_load_8[0];
                float lin01_349 = _tmem_load_8[1];
                float lin10_350 = _tmem_load_9[0];
                float lin11_351 = _tmem_load_9[1];
                float _max_64 = max_noftz(lin00_348, neg_cl);
                float _min_128 = fminf(_max_64, cl);
                lin00_348 = _min_128;
                float _max_65 = max_noftz(lin01_349, neg_cl);
                float _min_129 = fminf(_max_65, cl);
                lin01_349 = _min_129;
                float _max_66 = max_noftz(lin10_350, neg_cl);
                float _min_130 = fminf(_max_66, cl);
                lin10_350 = _min_130;
                float _max_67 = max_noftz(lin11_351, neg_cl);
                float _min_131 = fminf(_max_67, cl);
                lin11_351 = _min_131;
                float _min_132 = fminf(g00_344, cl);
                g00_344 = _min_132;
                float _min_133 = fminf(g01_345, cl);
                g01_345 = _min_133;
                float _min_134 = fminf(g10_346, cl);
                g10_346 = _min_134;
                float _min_135 = fminf(g11_347, cl);
                g11_347 = _min_135;
                float value00_352 = lin00_348 * g00_344;
                float value01_353 = lin01_349 * g01_345;
                float value10_354 = lin10_350 * g10_346;
                float value11_355 = lin11_351 * g11_347;
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
                float _exp2_68 = approx_exp2((-x00_358) * 1.4426950408889634f);
                float _rcp_68 = approx_rcp(1.0f + _exp2_68);
                float sig00_362 = _rcp_68;
                float _exp2_69 = approx_exp2((-x01_359) * 1.4426950408889634f);
                float _rcp_69 = approx_rcp(1.0f + _exp2_69);
                float sig01_363 = _rcp_69;
                float _exp2_70 = approx_exp2((-x10_360) * 1.4426950408889634f);
                float _rcp_70 = approx_rcp(1.0f + _exp2_70);
                float sig10_364 = _rcp_70;
                float _exp2_71 = approx_exp2((-x11_361) * 1.4426950408889634f);
                float _rcp_71 = approx_rcp(1.0f + _exp2_71);
                float sig11_365 = _rcp_71;
                float g00_366 = x00_358 * sig00_362;
                float g01_367 = x01_359 * sig01_363;
                float g10_368 = x10_360 * sig10_364;
                float g11_369 = x11_361 * sig11_365;
                float lin00_370 = _tmem_load_8[4];
                float lin01_371 = _tmem_load_8[5];
                float lin10_372 = _tmem_load_9[4];
                float lin11_373 = _tmem_load_9[5];
                float _max_68 = max_noftz(lin00_370, neg_cl);
                float _min_136 = fminf(_max_68, cl);
                lin00_370 = _min_136;
                float _max_69 = max_noftz(lin01_371, neg_cl);
                float _min_137 = fminf(_max_69, cl);
                lin01_371 = _min_137;
                float _max_70 = max_noftz(lin10_372, neg_cl);
                float _min_138 = fminf(_max_70, cl);
                lin10_372 = _min_138;
                float _max_71 = max_noftz(lin11_373, neg_cl);
                float _min_139 = fminf(_max_71, cl);
                lin11_373 = _min_139;
                float _min_140 = fminf(g00_366, cl);
                g00_366 = _min_140;
                float _min_141 = fminf(g01_367, cl);
                g01_367 = _min_141;
                float _min_142 = fminf(g10_368, cl);
                g10_368 = _min_142;
                float _min_143 = fminf(g11_369, cl);
                g11_369 = _min_143;
                float value00_374 = lin00_370 * g00_366;
                float value01_375 = lin01_371 * g01_367;
                float value10_376 = lin10_372 * g10_368;
                float value11_377 = lin11_373 * g11_369;
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
                float _exp2_72 = approx_exp2((-x00_380) * 1.4426950408889634f);
                float _rcp_72 = approx_rcp(1.0f + _exp2_72);
                float sig00_384 = _rcp_72;
                float _exp2_73 = approx_exp2((-x01_381) * 1.4426950408889634f);
                float _rcp_73 = approx_rcp(1.0f + _exp2_73);
                float sig01_385 = _rcp_73;
                float _exp2_74 = approx_exp2((-x10_382) * 1.4426950408889634f);
                float _rcp_74 = approx_rcp(1.0f + _exp2_74);
                float sig10_386 = _rcp_74;
                float _exp2_75 = approx_exp2((-x11_383) * 1.4426950408889634f);
                float _rcp_75 = approx_rcp(1.0f + _exp2_75);
                float sig11_387 = _rcp_75;
                float g00_388 = x00_380 * sig00_384;
                float g01_389 = x01_381 * sig01_385;
                float g10_390 = x10_382 * sig10_386;
                float g11_391 = x11_383 * sig11_387;
                float lin00_392 = _tmem_load_8[8];
                float lin01_393 = _tmem_load_8[9];
                float lin10_394 = _tmem_load_9[8];
                float lin11_395 = _tmem_load_9[9];
                float _max_72 = max_noftz(lin00_392, neg_cl);
                float _min_144 = fminf(_max_72, cl);
                lin00_392 = _min_144;
                float _max_73 = max_noftz(lin01_393, neg_cl);
                float _min_145 = fminf(_max_73, cl);
                lin01_393 = _min_145;
                float _max_74 = max_noftz(lin10_394, neg_cl);
                float _min_146 = fminf(_max_74, cl);
                lin10_394 = _min_146;
                float _max_75 = max_noftz(lin11_395, neg_cl);
                float _min_147 = fminf(_max_75, cl);
                lin11_395 = _min_147;
                float _min_148 = fminf(g00_388, cl);
                g00_388 = _min_148;
                float _min_149 = fminf(g01_389, cl);
                g01_389 = _min_149;
                float _min_150 = fminf(g10_390, cl);
                g10_390 = _min_150;
                float _min_151 = fminf(g11_391, cl);
                g11_391 = _min_151;
                float value00_396 = lin00_392 * g00_388;
                float value01_397 = lin01_393 * g01_389;
                float value10_398 = lin10_394 * g10_390;
                float value11_399 = lin11_395 * g11_391;
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
                float _exp2_76 = approx_exp2((-x00_402) * 1.4426950408889634f);
                float _rcp_76 = approx_rcp(1.0f + _exp2_76);
                float sig00_406 = _rcp_76;
                float _exp2_77 = approx_exp2((-x01_403) * 1.4426950408889634f);
                float _rcp_77 = approx_rcp(1.0f + _exp2_77);
                float sig01_407 = _rcp_77;
                float _exp2_78 = approx_exp2((-x10_404) * 1.4426950408889634f);
                float _rcp_78 = approx_rcp(1.0f + _exp2_78);
                float sig10_408 = _rcp_78;
                float _exp2_79 = approx_exp2((-x11_405) * 1.4426950408889634f);
                float _rcp_79 = approx_rcp(1.0f + _exp2_79);
                float sig11_409 = _rcp_79;
                float g00_410 = x00_402 * sig00_406;
                float g01_411 = x01_403 * sig01_407;
                float g10_412 = x10_404 * sig10_408;
                float g11_413 = x11_405 * sig11_409;
                float lin00_414 = _tmem_load_8[12];
                float lin01_415 = _tmem_load_8[13];
                float lin10_416 = _tmem_load_9[12];
                float lin11_417 = _tmem_load_9[13];
                float _max_76 = max_noftz(lin00_414, neg_cl);
                float _min_152 = fminf(_max_76, cl);
                lin00_414 = _min_152;
                float _max_77 = max_noftz(lin01_415, neg_cl);
                float _min_153 = fminf(_max_77, cl);
                lin01_415 = _min_153;
                float _max_78 = max_noftz(lin10_416, neg_cl);
                float _min_154 = fminf(_max_78, cl);
                lin10_416 = _min_154;
                float _max_79 = max_noftz(lin11_417, neg_cl);
                float _min_155 = fminf(_max_79, cl);
                lin11_417 = _min_155;
                float _min_156 = fminf(g00_410, cl);
                g00_410 = _min_156;
                float _min_157 = fminf(g01_411, cl);
                g01_411 = _min_157;
                float _min_158 = fminf(g10_412, cl);
                g10_412 = _min_158;
                float _min_159 = fminf(g11_413, cl);
                g11_413 = _min_159;
                float value00_418 = lin00_414 * g00_410;
                float value01_419 = lin01_415 * g01_411;
                float value10_420 = lin10_416 * g10_412;
                float value11_421 = lin11_417 * g11_413;
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
                float _exp2_80 = approx_exp2((-x00_425) * 1.4426950408889634f);
                float _rcp_80 = approx_rcp(1.0f + _exp2_80);
                float sig00_429 = _rcp_80;
                float _exp2_81 = approx_exp2((-x01_426) * 1.4426950408889634f);
                float _rcp_81 = approx_rcp(1.0f + _exp2_81);
                float sig01_430 = _rcp_81;
                float _exp2_82 = approx_exp2((-x10_427) * 1.4426950408889634f);
                float _rcp_82 = approx_rcp(1.0f + _exp2_82);
                float sig10_431 = _rcp_82;
                float _exp2_83 = approx_exp2((-x11_428) * 1.4426950408889634f);
                float _rcp_83 = approx_rcp(1.0f + _exp2_83);
                float sig11_432 = _rcp_83;
                float g00_433 = x00_425 * sig00_429;
                float g01_434 = x01_426 * sig01_430;
                float g10_435 = x10_427 * sig10_431;
                float g11_436 = x11_428 * sig11_432;
                float lin00_437 = _tmem_load_10[0];
                float lin01_438 = _tmem_load_10[1];
                float lin10_439 = _tmem_load_11[0];
                float lin11_440 = _tmem_load_11[1];
                float _max_80 = max_noftz(lin00_437, neg_cl);
                float _min_160 = fminf(_max_80, cl);
                lin00_437 = _min_160;
                float _max_81 = max_noftz(lin01_438, neg_cl);
                float _min_161 = fminf(_max_81, cl);
                lin01_438 = _min_161;
                float _max_82 = max_noftz(lin10_439, neg_cl);
                float _min_162 = fminf(_max_82, cl);
                lin10_439 = _min_162;
                float _max_83 = max_noftz(lin11_440, neg_cl);
                float _min_163 = fminf(_max_83, cl);
                lin11_440 = _min_163;
                float _min_164 = fminf(g00_433, cl);
                g00_433 = _min_164;
                float _min_165 = fminf(g01_434, cl);
                g01_434 = _min_165;
                float _min_166 = fminf(g10_435, cl);
                g10_435 = _min_166;
                float _min_167 = fminf(g11_436, cl);
                g11_436 = _min_167;
                float value00_441 = lin00_437 * g00_433;
                float value01_442 = lin01_438 * g01_434;
                float value10_443 = lin10_439 * g10_435;
                float value11_444 = lin11_440 * g11_436;
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
                float _exp2_84 = approx_exp2((-x00_447) * 1.4426950408889634f);
                float _rcp_84 = approx_rcp(1.0f + _exp2_84);
                float sig00_451 = _rcp_84;
                float _exp2_85 = approx_exp2((-x01_448) * 1.4426950408889634f);
                float _rcp_85 = approx_rcp(1.0f + _exp2_85);
                float sig01_452 = _rcp_85;
                float _exp2_86 = approx_exp2((-x10_449) * 1.4426950408889634f);
                float _rcp_86 = approx_rcp(1.0f + _exp2_86);
                float sig10_453 = _rcp_86;
                float _exp2_87 = approx_exp2((-x11_450) * 1.4426950408889634f);
                float _rcp_87 = approx_rcp(1.0f + _exp2_87);
                float sig11_454 = _rcp_87;
                float g00_455 = x00_447 * sig00_451;
                float g01_456 = x01_448 * sig01_452;
                float g10_457 = x10_449 * sig10_453;
                float g11_458 = x11_450 * sig11_454;
                float lin00_459 = _tmem_load_10[4];
                float lin01_460 = _tmem_load_10[5];
                float lin10_461 = _tmem_load_11[4];
                float lin11_462 = _tmem_load_11[5];
                float _max_84 = max_noftz(lin00_459, neg_cl);
                float _min_168 = fminf(_max_84, cl);
                lin00_459 = _min_168;
                float _max_85 = max_noftz(lin01_460, neg_cl);
                float _min_169 = fminf(_max_85, cl);
                lin01_460 = _min_169;
                float _max_86 = max_noftz(lin10_461, neg_cl);
                float _min_170 = fminf(_max_86, cl);
                lin10_461 = _min_170;
                float _max_87 = max_noftz(lin11_462, neg_cl);
                float _min_171 = fminf(_max_87, cl);
                lin11_462 = _min_171;
                float _min_172 = fminf(g00_455, cl);
                g00_455 = _min_172;
                float _min_173 = fminf(g01_456, cl);
                g01_456 = _min_173;
                float _min_174 = fminf(g10_457, cl);
                g10_457 = _min_174;
                float _min_175 = fminf(g11_458, cl);
                g11_458 = _min_175;
                float value00_463 = lin00_459 * g00_455;
                float value01_464 = lin01_460 * g01_456;
                float value10_465 = lin10_461 * g10_457;
                float value11_466 = lin11_462 * g11_458;
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
                float _exp2_88 = approx_exp2((-x00_469) * 1.4426950408889634f);
                float _rcp_88 = approx_rcp(1.0f + _exp2_88);
                float sig00_473 = _rcp_88;
                float _exp2_89 = approx_exp2((-x01_470) * 1.4426950408889634f);
                float _rcp_89 = approx_rcp(1.0f + _exp2_89);
                float sig01_474 = _rcp_89;
                float _exp2_90 = approx_exp2((-x10_471) * 1.4426950408889634f);
                float _rcp_90 = approx_rcp(1.0f + _exp2_90);
                float sig10_475 = _rcp_90;
                float _exp2_91 = approx_exp2((-x11_472) * 1.4426950408889634f);
                float _rcp_91 = approx_rcp(1.0f + _exp2_91);
                float sig11_476 = _rcp_91;
                float g00_477 = x00_469 * sig00_473;
                float g01_478 = x01_470 * sig01_474;
                float g10_479 = x10_471 * sig10_475;
                float g11_480 = x11_472 * sig11_476;
                float lin00_481 = _tmem_load_10[8];
                float lin01_482 = _tmem_load_10[9];
                float lin10_483 = _tmem_load_11[8];
                float lin11_484 = _tmem_load_11[9];
                float _max_88 = max_noftz(lin00_481, neg_cl);
                float _min_176 = fminf(_max_88, cl);
                lin00_481 = _min_176;
                float _max_89 = max_noftz(lin01_482, neg_cl);
                float _min_177 = fminf(_max_89, cl);
                lin01_482 = _min_177;
                float _max_90 = max_noftz(lin10_483, neg_cl);
                float _min_178 = fminf(_max_90, cl);
                lin10_483 = _min_178;
                float _max_91 = max_noftz(lin11_484, neg_cl);
                float _min_179 = fminf(_max_91, cl);
                lin11_484 = _min_179;
                float _min_180 = fminf(g00_477, cl);
                g00_477 = _min_180;
                float _min_181 = fminf(g01_478, cl);
                g01_478 = _min_181;
                float _min_182 = fminf(g10_479, cl);
                g10_479 = _min_182;
                float _min_183 = fminf(g11_480, cl);
                g11_480 = _min_183;
                float value00_485 = lin00_481 * g00_477;
                float value01_486 = lin01_482 * g01_478;
                float value10_487 = lin10_483 * g10_479;
                float value11_488 = lin11_484 * g11_480;
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
                float _exp2_92 = approx_exp2((-x00_491) * 1.4426950408889634f);
                float _rcp_92 = approx_rcp(1.0f + _exp2_92);
                float sig00_495 = _rcp_92;
                float _exp2_93 = approx_exp2((-x01_492) * 1.4426950408889634f);
                float _rcp_93 = approx_rcp(1.0f + _exp2_93);
                float sig01_496 = _rcp_93;
                float _exp2_94 = approx_exp2((-x10_493) * 1.4426950408889634f);
                float _rcp_94 = approx_rcp(1.0f + _exp2_94);
                float sig10_497 = _rcp_94;
                float _exp2_95 = approx_exp2((-x11_494) * 1.4426950408889634f);
                float _rcp_95 = approx_rcp(1.0f + _exp2_95);
                float sig11_498 = _rcp_95;
                float g00_499 = x00_491 * sig00_495;
                float g01_500 = x01_492 * sig01_496;
                float g10_501 = x10_493 * sig10_497;
                float g11_502 = x11_494 * sig11_498;
                float lin00_503 = _tmem_load_10[12];
                float lin01_504 = _tmem_load_10[13];
                float lin10_505 = _tmem_load_11[12];
                float lin11_506 = _tmem_load_11[13];
                float _max_92 = max_noftz(lin00_503, neg_cl);
                float _min_184 = fminf(_max_92, cl);
                lin00_503 = _min_184;
                float _max_93 = max_noftz(lin01_504, neg_cl);
                float _min_185 = fminf(_max_93, cl);
                lin01_504 = _min_185;
                float _max_94 = max_noftz(lin10_505, neg_cl);
                float _min_186 = fminf(_max_94, cl);
                lin10_505 = _min_186;
                float _max_95 = max_noftz(lin11_506, neg_cl);
                float _min_187 = fminf(_max_95, cl);
                lin11_506 = _min_187;
                float _min_188 = fminf(g00_499, cl);
                g00_499 = _min_188;
                float _min_189 = fminf(g01_500, cl);
                g01_500 = _min_189;
                float _min_190 = fminf(g10_501, cl);
                g10_501 = _min_190;
                float _min_191 = fminf(g11_502, cl);
                g11_502 = _min_191;
                float value00_507 = lin00_503 * g00_499;
                float value01_508 = lin01_504 * g01_500;
                float value10_509 = lin10_505 * g10_501;
                float value11_510 = lin11_506 * g11_502;
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
                float _exp2_96 = approx_exp2((-x00_514) * 1.4426950408889634f);
                float _rcp_96 = approx_rcp(1.0f + _exp2_96);
                float sig00_518 = _rcp_96;
                float _exp2_97 = approx_exp2((-x01_515) * 1.4426950408889634f);
                float _rcp_97 = approx_rcp(1.0f + _exp2_97);
                float sig01_519 = _rcp_97;
                float _exp2_98 = approx_exp2((-x10_516) * 1.4426950408889634f);
                float _rcp_98 = approx_rcp(1.0f + _exp2_98);
                float sig10_520 = _rcp_98;
                float _exp2_99 = approx_exp2((-x11_517) * 1.4426950408889634f);
                float _rcp_99 = approx_rcp(1.0f + _exp2_99);
                float sig11_521 = _rcp_99;
                float g00_522 = x00_514 * sig00_518;
                float g01_523 = x01_515 * sig01_519;
                float g10_524 = x10_516 * sig10_520;
                float g11_525 = x11_517 * sig11_521;
                float lin00_526 = _tmem_load_12[0];
                float lin01_527 = _tmem_load_12[1];
                float lin10_528 = _tmem_load_13[0];
                float lin11_529 = _tmem_load_13[1];
                float _max_96 = max_noftz(lin00_526, neg_cl);
                float _min_192 = fminf(_max_96, cl);
                lin00_526 = _min_192;
                float _max_97 = max_noftz(lin01_527, neg_cl);
                float _min_193 = fminf(_max_97, cl);
                lin01_527 = _min_193;
                float _max_98 = max_noftz(lin10_528, neg_cl);
                float _min_194 = fminf(_max_98, cl);
                lin10_528 = _min_194;
                float _max_99 = max_noftz(lin11_529, neg_cl);
                float _min_195 = fminf(_max_99, cl);
                lin11_529 = _min_195;
                float _min_196 = fminf(g00_522, cl);
                g00_522 = _min_196;
                float _min_197 = fminf(g01_523, cl);
                g01_523 = _min_197;
                float _min_198 = fminf(g10_524, cl);
                g10_524 = _min_198;
                float _min_199 = fminf(g11_525, cl);
                g11_525 = _min_199;
                float value00_530 = lin00_526 * g00_522;
                float value01_531 = lin01_527 * g01_523;
                float value10_532 = lin10_528 * g10_524;
                float value11_533 = lin11_529 * g11_525;
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
                float _exp2_100 = approx_exp2((-x00_536) * 1.4426950408889634f);
                float _rcp_100 = approx_rcp(1.0f + _exp2_100);
                float sig00_540 = _rcp_100;
                float _exp2_101 = approx_exp2((-x01_537) * 1.4426950408889634f);
                float _rcp_101 = approx_rcp(1.0f + _exp2_101);
                float sig01_541 = _rcp_101;
                float _exp2_102 = approx_exp2((-x10_538) * 1.4426950408889634f);
                float _rcp_102 = approx_rcp(1.0f + _exp2_102);
                float sig10_542 = _rcp_102;
                float _exp2_103 = approx_exp2((-x11_539) * 1.4426950408889634f);
                float _rcp_103 = approx_rcp(1.0f + _exp2_103);
                float sig11_543 = _rcp_103;
                float g00_544 = x00_536 * sig00_540;
                float g01_545 = x01_537 * sig01_541;
                float g10_546 = x10_538 * sig10_542;
                float g11_547 = x11_539 * sig11_543;
                float lin00_548 = _tmem_load_12[4];
                float lin01_549 = _tmem_load_12[5];
                float lin10_550 = _tmem_load_13[4];
                float lin11_551 = _tmem_load_13[5];
                float _max_100 = max_noftz(lin00_548, neg_cl);
                float _min_200 = fminf(_max_100, cl);
                lin00_548 = _min_200;
                float _max_101 = max_noftz(lin01_549, neg_cl);
                float _min_201 = fminf(_max_101, cl);
                lin01_549 = _min_201;
                float _max_102 = max_noftz(lin10_550, neg_cl);
                float _min_202 = fminf(_max_102, cl);
                lin10_550 = _min_202;
                float _max_103 = max_noftz(lin11_551, neg_cl);
                float _min_203 = fminf(_max_103, cl);
                lin11_551 = _min_203;
                float _min_204 = fminf(g00_544, cl);
                g00_544 = _min_204;
                float _min_205 = fminf(g01_545, cl);
                g01_545 = _min_205;
                float _min_206 = fminf(g10_546, cl);
                g10_546 = _min_206;
                float _min_207 = fminf(g11_547, cl);
                g11_547 = _min_207;
                float value00_552 = lin00_548 * g00_544;
                float value01_553 = lin01_549 * g01_545;
                float value10_554 = lin10_550 * g10_546;
                float value11_555 = lin11_551 * g11_547;
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
                float _exp2_104 = approx_exp2((-x00_558) * 1.4426950408889634f);
                float _rcp_104 = approx_rcp(1.0f + _exp2_104);
                float sig00_562 = _rcp_104;
                float _exp2_105 = approx_exp2((-x01_559) * 1.4426950408889634f);
                float _rcp_105 = approx_rcp(1.0f + _exp2_105);
                float sig01_563 = _rcp_105;
                float _exp2_106 = approx_exp2((-x10_560) * 1.4426950408889634f);
                float _rcp_106 = approx_rcp(1.0f + _exp2_106);
                float sig10_564 = _rcp_106;
                float _exp2_107 = approx_exp2((-x11_561) * 1.4426950408889634f);
                float _rcp_107 = approx_rcp(1.0f + _exp2_107);
                float sig11_565 = _rcp_107;
                float g00_566 = x00_558 * sig00_562;
                float g01_567 = x01_559 * sig01_563;
                float g10_568 = x10_560 * sig10_564;
                float g11_569 = x11_561 * sig11_565;
                float lin00_570 = _tmem_load_12[8];
                float lin01_571 = _tmem_load_12[9];
                float lin10_572 = _tmem_load_13[8];
                float lin11_573 = _tmem_load_13[9];
                float _max_104 = max_noftz(lin00_570, neg_cl);
                float _min_208 = fminf(_max_104, cl);
                lin00_570 = _min_208;
                float _max_105 = max_noftz(lin01_571, neg_cl);
                float _min_209 = fminf(_max_105, cl);
                lin01_571 = _min_209;
                float _max_106 = max_noftz(lin10_572, neg_cl);
                float _min_210 = fminf(_max_106, cl);
                lin10_572 = _min_210;
                float _max_107 = max_noftz(lin11_573, neg_cl);
                float _min_211 = fminf(_max_107, cl);
                lin11_573 = _min_211;
                float _min_212 = fminf(g00_566, cl);
                g00_566 = _min_212;
                float _min_213 = fminf(g01_567, cl);
                g01_567 = _min_213;
                float _min_214 = fminf(g10_568, cl);
                g10_568 = _min_214;
                float _min_215 = fminf(g11_569, cl);
                g11_569 = _min_215;
                float value00_574 = lin00_570 * g00_566;
                float value01_575 = lin01_571 * g01_567;
                float value10_576 = lin10_572 * g10_568;
                float value11_577 = lin11_573 * g11_569;
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
                float _exp2_108 = approx_exp2((-x00_580) * 1.4426950408889634f);
                float _rcp_108 = approx_rcp(1.0f + _exp2_108);
                float sig00_584 = _rcp_108;
                float _exp2_109 = approx_exp2((-x01_581) * 1.4426950408889634f);
                float _rcp_109 = approx_rcp(1.0f + _exp2_109);
                float sig01_585 = _rcp_109;
                float _exp2_110 = approx_exp2((-x10_582) * 1.4426950408889634f);
                float _rcp_110 = approx_rcp(1.0f + _exp2_110);
                float sig10_586 = _rcp_110;
                float _exp2_111 = approx_exp2((-x11_583) * 1.4426950408889634f);
                float _rcp_111 = approx_rcp(1.0f + _exp2_111);
                float sig11_587 = _rcp_111;
                float g00_588 = x00_580 * sig00_584;
                float g01_589 = x01_581 * sig01_585;
                float g10_590 = x10_582 * sig10_586;
                float g11_591 = x11_583 * sig11_587;
                float lin00_592 = _tmem_load_12[12];
                float lin01_593 = _tmem_load_12[13];
                float lin10_594 = _tmem_load_13[12];
                float lin11_595 = _tmem_load_13[13];
                float _max_108 = max_noftz(lin00_592, neg_cl);
                float _min_216 = fminf(_max_108, cl);
                lin00_592 = _min_216;
                float _max_109 = max_noftz(lin01_593, neg_cl);
                float _min_217 = fminf(_max_109, cl);
                lin01_593 = _min_217;
                float _max_110 = max_noftz(lin10_594, neg_cl);
                float _min_218 = fminf(_max_110, cl);
                lin10_594 = _min_218;
                float _max_111 = max_noftz(lin11_595, neg_cl);
                float _min_219 = fminf(_max_111, cl);
                lin11_595 = _min_219;
                float _min_220 = fminf(g00_588, cl);
                g00_588 = _min_220;
                float _min_221 = fminf(g01_589, cl);
                g01_589 = _min_221;
                float _min_222 = fminf(g10_590, cl);
                g10_590 = _min_222;
                float _min_223 = fminf(g11_591, cl);
                g11_591 = _min_223;
                float value00_596 = lin00_592 * g00_588;
                float value01_597 = lin01_593 * g01_589;
                float value10_598 = lin10_594 * g10_590;
                float value11_599 = lin11_595 * g11_591;
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
                float _exp2_112 = approx_exp2((-x00_603) * 1.4426950408889634f);
                float _rcp_112 = approx_rcp(1.0f + _exp2_112);
                float sig00_607 = _rcp_112;
                float _exp2_113 = approx_exp2((-x01_604) * 1.4426950408889634f);
                float _rcp_113 = approx_rcp(1.0f + _exp2_113);
                float sig01_608 = _rcp_113;
                float _exp2_114 = approx_exp2((-x10_605) * 1.4426950408889634f);
                float _rcp_114 = approx_rcp(1.0f + _exp2_114);
                float sig10_609 = _rcp_114;
                float _exp2_115 = approx_exp2((-x11_606) * 1.4426950408889634f);
                float _rcp_115 = approx_rcp(1.0f + _exp2_115);
                float sig11_610 = _rcp_115;
                float g00_611 = x00_603 * sig00_607;
                float g01_612 = x01_604 * sig01_608;
                float g10_613 = x10_605 * sig10_609;
                float g11_614 = x11_606 * sig11_610;
                float lin00_615 = _tmem_load_14[0];
                float lin01_616 = _tmem_load_14[1];
                float lin10_617 = _tmem_load_15[0];
                float lin11_618 = _tmem_load_15[1];
                float _max_112 = max_noftz(lin00_615, neg_cl);
                float _min_224 = fminf(_max_112, cl);
                lin00_615 = _min_224;
                float _max_113 = max_noftz(lin01_616, neg_cl);
                float _min_225 = fminf(_max_113, cl);
                lin01_616 = _min_225;
                float _max_114 = max_noftz(lin10_617, neg_cl);
                float _min_226 = fminf(_max_114, cl);
                lin10_617 = _min_226;
                float _max_115 = max_noftz(lin11_618, neg_cl);
                float _min_227 = fminf(_max_115, cl);
                lin11_618 = _min_227;
                float _min_228 = fminf(g00_611, cl);
                g00_611 = _min_228;
                float _min_229 = fminf(g01_612, cl);
                g01_612 = _min_229;
                float _min_230 = fminf(g10_613, cl);
                g10_613 = _min_230;
                float _min_231 = fminf(g11_614, cl);
                g11_614 = _min_231;
                float value00_619 = lin00_615 * g00_611;
                float value01_620 = lin01_616 * g01_612;
                float value10_621 = lin10_617 * g10_613;
                float value11_622 = lin11_618 * g11_614;
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
                float _exp2_116 = approx_exp2((-x00_625) * 1.4426950408889634f);
                float _rcp_116 = approx_rcp(1.0f + _exp2_116);
                float sig00_629 = _rcp_116;
                float _exp2_117 = approx_exp2((-x01_626) * 1.4426950408889634f);
                float _rcp_117 = approx_rcp(1.0f + _exp2_117);
                float sig01_630 = _rcp_117;
                float _exp2_118 = approx_exp2((-x10_627) * 1.4426950408889634f);
                float _rcp_118 = approx_rcp(1.0f + _exp2_118);
                float sig10_631 = _rcp_118;
                float _exp2_119 = approx_exp2((-x11_628) * 1.4426950408889634f);
                float _rcp_119 = approx_rcp(1.0f + _exp2_119);
                float sig11_632 = _rcp_119;
                float g00_633 = x00_625 * sig00_629;
                float g01_634 = x01_626 * sig01_630;
                float g10_635 = x10_627 * sig10_631;
                float g11_636 = x11_628 * sig11_632;
                float lin00_637 = _tmem_load_14[4];
                float lin01_638 = _tmem_load_14[5];
                float lin10_639 = _tmem_load_15[4];
                float lin11_640 = _tmem_load_15[5];
                float _max_116 = max_noftz(lin00_637, neg_cl);
                float _min_232 = fminf(_max_116, cl);
                lin00_637 = _min_232;
                float _max_117 = max_noftz(lin01_638, neg_cl);
                float _min_233 = fminf(_max_117, cl);
                lin01_638 = _min_233;
                float _max_118 = max_noftz(lin10_639, neg_cl);
                float _min_234 = fminf(_max_118, cl);
                lin10_639 = _min_234;
                float _max_119 = max_noftz(lin11_640, neg_cl);
                float _min_235 = fminf(_max_119, cl);
                lin11_640 = _min_235;
                float _min_236 = fminf(g00_633, cl);
                g00_633 = _min_236;
                float _min_237 = fminf(g01_634, cl);
                g01_634 = _min_237;
                float _min_238 = fminf(g10_635, cl);
                g10_635 = _min_238;
                float _min_239 = fminf(g11_636, cl);
                g11_636 = _min_239;
                float value00_641 = lin00_637 * g00_633;
                float value01_642 = lin01_638 * g01_634;
                float value10_643 = lin10_639 * g10_635;
                float value11_644 = lin11_640 * g11_636;
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
                float _exp2_120 = approx_exp2((-x00_647) * 1.4426950408889634f);
                float _rcp_120 = approx_rcp(1.0f + _exp2_120);
                float sig00_651 = _rcp_120;
                float _exp2_121 = approx_exp2((-x01_648) * 1.4426950408889634f);
                float _rcp_121 = approx_rcp(1.0f + _exp2_121);
                float sig01_652 = _rcp_121;
                float _exp2_122 = approx_exp2((-x10_649) * 1.4426950408889634f);
                float _rcp_122 = approx_rcp(1.0f + _exp2_122);
                float sig10_653 = _rcp_122;
                float _exp2_123 = approx_exp2((-x11_650) * 1.4426950408889634f);
                float _rcp_123 = approx_rcp(1.0f + _exp2_123);
                float sig11_654 = _rcp_123;
                float g00_655 = x00_647 * sig00_651;
                float g01_656 = x01_648 * sig01_652;
                float g10_657 = x10_649 * sig10_653;
                float g11_658 = x11_650 * sig11_654;
                float lin00_659 = _tmem_load_14[8];
                float lin01_660 = _tmem_load_14[9];
                float lin10_661 = _tmem_load_15[8];
                float lin11_662 = _tmem_load_15[9];
                float _max_120 = max_noftz(lin00_659, neg_cl);
                float _min_240 = fminf(_max_120, cl);
                lin00_659 = _min_240;
                float _max_121 = max_noftz(lin01_660, neg_cl);
                float _min_241 = fminf(_max_121, cl);
                lin01_660 = _min_241;
                float _max_122 = max_noftz(lin10_661, neg_cl);
                float _min_242 = fminf(_max_122, cl);
                lin10_661 = _min_242;
                float _max_123 = max_noftz(lin11_662, neg_cl);
                float _min_243 = fminf(_max_123, cl);
                lin11_662 = _min_243;
                float _min_244 = fminf(g00_655, cl);
                g00_655 = _min_244;
                float _min_245 = fminf(g01_656, cl);
                g01_656 = _min_245;
                float _min_246 = fminf(g10_657, cl);
                g10_657 = _min_246;
                float _min_247 = fminf(g11_658, cl);
                g11_658 = _min_247;
                float value00_663 = lin00_659 * g00_655;
                float value01_664 = lin01_660 * g01_656;
                float value10_665 = lin10_661 * g10_657;
                float value11_666 = lin11_662 * g11_658;
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
                float _exp2_124 = approx_exp2((-x00_669) * 1.4426950408889634f);
                float _rcp_124 = approx_rcp(1.0f + _exp2_124);
                float sig00_673 = _rcp_124;
                float _exp2_125 = approx_exp2((-x01_670) * 1.4426950408889634f);
                float _rcp_125 = approx_rcp(1.0f + _exp2_125);
                float sig01_674 = _rcp_125;
                float _exp2_126 = approx_exp2((-x10_671) * 1.4426950408889634f);
                float _rcp_126 = approx_rcp(1.0f + _exp2_126);
                float sig10_675 = _rcp_126;
                float _exp2_127 = approx_exp2((-x11_672) * 1.4426950408889634f);
                float _rcp_127 = approx_rcp(1.0f + _exp2_127);
                float sig11_676 = _rcp_127;
                float g00_677 = x00_669 * sig00_673;
                float g01_678 = x01_670 * sig01_674;
                float g10_679 = x10_671 * sig10_675;
                float g11_680 = x11_672 * sig11_676;
                float lin00_681 = _tmem_load_14[12];
                float lin01_682 = _tmem_load_14[13];
                float lin10_683 = _tmem_load_15[12];
                float lin11_684 = _tmem_load_15[13];
                float _max_124 = max_noftz(lin00_681, neg_cl);
                float _min_248 = fminf(_max_124, cl);
                lin00_681 = _min_248;
                float _max_125 = max_noftz(lin01_682, neg_cl);
                float _min_249 = fminf(_max_125, cl);
                lin01_682 = _min_249;
                float _max_126 = max_noftz(lin10_683, neg_cl);
                float _min_250 = fminf(_max_126, cl);
                lin10_683 = _min_250;
                float _max_127 = max_noftz(lin11_684, neg_cl);
                float _min_251 = fminf(_max_127, cl);
                lin11_684 = _min_251;
                float _min_252 = fminf(g00_677, cl);
                g00_677 = _min_252;
                float _min_253 = fminf(g01_678, cl);
                g01_678 = _min_253;
                float _min_254 = fminf(g10_679, cl);
                g10_679 = _min_254;
                float _min_255 = fminf(g11_680, cl);
                g11_680 = _min_255;
                float value00_685 = lin00_681 * g00_677;
                float value01_686 = lin01_682 * g01_678;
                float value10_687 = lin10_683 * g10_679;
                float value11_688 = lin11_684 * g11_680;
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
                    : "=r"(_clc_ctaid_6)
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
                    : "=r"(_clc_ctaid_7)
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
                    : "=r"(_clc_ctaid_2)
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
                    : "=r"(_clc_ctaid_3)
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
                    : "=r"(_clc_ctaid_0)
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
                    : "=r"(_clc_ctaid_1)
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
                        : "=r"(_clc_ctaid_4)
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
                        : "=r"(_clc_ctaid_5)
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
                        : "=r"(_clc_ctaid_8)
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
                        : "=r"(_clc_ctaid_9)
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
                    : "=r"(_clc_ctaid_10)
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
                    : "=r"(_clc_ctaid_11)
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
