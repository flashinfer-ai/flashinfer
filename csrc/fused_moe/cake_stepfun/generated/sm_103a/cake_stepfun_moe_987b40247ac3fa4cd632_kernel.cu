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
#define TMEM_NCOLS 176
#define TMEM_ACCUM_OFFSET 0
#define TMEM_SFA_OFFSET 128
#define TMEM_SFB_OFFSET 160
#define NUM_K_PIPE_STAGES 4
#define NUM_MMA_PIPE_STAGES 2
#define NUM_WORK_PIPE_STAGES 3
#define NUM_THROTTLE_PIPE_STAGES 3
#define NUM_FAST_PIPE_STAGES 1
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 32768
#define SMEM_SMEM_A_STRIDE 32768
#define SMEM_SMEM_B_OFF 132096
#define SMEM_SMEM_B_STAGE_BYTES 8192
#define SMEM_SMEM_B_STRIDE 8192
#define SMEM_EPI_STAGING_OFF 164864
#define SMEM_EPI_STAGING_STAGE_BYTES 4096
#define SMEM_EPI_STAGING_STRIDE 4096
#define SMEM_EPI_PAIRS_OFF 164864
#define SMEM_EPI_PAIRS_STAGE_BYTES 4096
#define SMEM_EPI_PAIRS_STRIDE 4096
#define SMEM_AMAX_SMEM_OFF 168960
#define SMEM_AMAX_SMEM_STAGE_BYTES 1024
#define SMEM_AMAX_SMEM_STRIDE 1024
#define SMEM_SMEM_SFA_OFF 169984
#define SMEM_SMEM_SFA_STAGE_BYTES 1024
#define SMEM_SMEM_SFA_STRIDE 1024
#define SMEM_SMEM_SFB_OFF 174080
#define SMEM_SMEM_SFB_STAGE_BYTES 512
#define SMEM_SMEM_SFB_STRIDE 512
#define SMEM_WORK_RESPONSE_OFF 176128
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_FAST_RESPONSE_OFF 176176
#define SMEM_FAST_RESPONSE_STAGE_BYTES 64
#define SMEM_FAST_RESPONSE_STRIDE 64
#define SMEM_TOTAL 176256
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


__device__ __forceinline__ void tcgen05_mma_mxf8_bs_cta2(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
    int sfa_taddr, int sfb_taddr, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale"
        " [%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(sfa_taddr), "r"(sfb_taddr),
           "r"(enable_input_d));
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
kernel_cake_stepfun_moe_987b40247ac3fa4cd632(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, uint8_t* __restrict__ SFB, const __grid_constant__ CUtensorMap C, uint8_t* __restrict__ SFC, int* __restrict__ route_map, int* __restrict__ tile_expert, int* __restrict__ tile_mn_limit, int* __restrict__ num_non_exiting_ctas, float* __restrict__ clamp_limit, int M_out, int K, int grid_m, int grid_n, int K_tiles)
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
    #define b_full_addr (mbar_base + 32)
    #define sfa_full_addr (mbar_base + 64)
    #define sfb_full_addr (mbar_base + 96)
    #define sfa_free_addr (mbar_base + 128)
    #define sfb_free_addr (mbar_base + 160)
    #define tmem_sfa_full_addr (mbar_base + 192)
    #define tmem_sfb_full_addr (mbar_base + 224)
    #define k_done_addr (mbar_base + 256)
    #define mma_full_addr (mbar_base + 288)
    #define mma_free_addr (mbar_base + 304)
    #define work_full_addr (mbar_base + 320)
    #define work_empty_addr (mbar_base + 344)
    #define throttle_full_addr (mbar_base + 368)
    #define throttle_empty_addr (mbar_base + 392)
    #define fast_ready_addr (mbar_base + 416)

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
    uint8_t* epi_staging = reinterpret_cast<uint8_t*>(smem_raw + 164864);
    const int epi_staging_addr = smem + 164864;
    uint16_t* epi_pairs = reinterpret_cast<uint16_t*>(smem_raw + 164864);
    const int epi_pairs_addr = smem + 164864;
    float* amax_smem = reinterpret_cast<float*>(smem_raw + 168960);
    const int amax_smem_addr = smem + 168960;
    uint8_t* smem_sfa = reinterpret_cast<uint8_t*>(smem_raw + 169984);
    const int smem_sfa_addr = smem + 169984;
    uint8_t* smem_sfb = reinterpret_cast<uint8_t*>(smem_raw + 174080);
    const int smem_sfb_addr = smem + 174080;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 176128);
    const int work_response_addr = smem + 176128;
    unsigned int* fast_response = reinterpret_cast<unsigned int*>(smem_raw + 176176);
    const int fast_response_addr = smem + 176176;
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if ((int)blockIdx.y >= num_non_exiting_ctas[0]) return;

    // Mbarrier init (16 pipeline groups, 0 ordered-sequence groups, 53 barriers)
    // Mbarriers at smem_raw[0..424)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'k_pipe' ---
            // a_full: 4 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            mbarrier_init(smem + 8, 2);
            mbarrier_init(smem + 16, 2);
            mbarrier_init(smem + 24, 2);
            // b_full: 4 barriers, init_count=2
            mbarrier_init(smem + 32, 2);
            mbarrier_init(smem + 40, 2);
            mbarrier_init(smem + 48, 2);
            mbarrier_init(smem + 56, 2);
            // sfa_full: 4 barriers, init_count=2
            mbarrier_init(smem + 64, 2);
            mbarrier_init(smem + 72, 2);
            mbarrier_init(smem + 80, 2);
            mbarrier_init(smem + 88, 2);
            // sfb_full: 4 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // sfa_free: 4 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            // sfb_free: 4 barriers, init_count=128
            mbarrier_init(smem + 160, 128);
            mbarrier_init(smem + 168, 128);
            mbarrier_init(smem + 176, 128);
            mbarrier_init(smem + 184, 128);
            // tmem_sfa_full: 4 barriers, init_count=32
            mbarrier_init(smem + 192, 32);
            mbarrier_init(smem + 200, 32);
            mbarrier_init(smem + 208, 32);
            mbarrier_init(smem + 216, 32);
            // tmem_sfb_full: 4 barriers, init_count=256
            mbarrier_init(smem + 224, 256);
            mbarrier_init(smem + 232, 256);
            mbarrier_init(smem + 240, 256);
            mbarrier_init(smem + 248, 256);
            // k_done: 4 barriers, init_count=1
            mbarrier_init(smem + 256, 1);
            mbarrier_init(smem + 264, 1);
            mbarrier_init(smem + 272, 1);
            mbarrier_init(smem + 280, 1);
            // --- pipeline 'mma_pipe' ---
            // mma_full: 2 barriers, init_count=1
            mbarrier_init(smem + 288, 1);
            mbarrier_init(smem + 296, 1);
            // mma_free: 2 barriers, init_count=256
            mbarrier_init(smem + 304, 256);
            mbarrier_init(smem + 312, 256);
            // --- pipeline 'work_pipe' ---
            // work_full: 3 barriers, init_count=1
            mbarrier_init(smem + 320, 1);
            mbarrier_init(smem + 328, 1);
            mbarrier_init(smem + 336, 1);
            // work_empty: 3 barriers, init_count=928
            mbarrier_init(smem + 344, 928);
            mbarrier_init(smem + 352, 928);
            mbarrier_init(smem + 360, 928);
            // --- pipeline 'throttle_pipe' ---
            // throttle_full: 3 barriers, init_count=32
            mbarrier_init(smem + 368, 32);
            mbarrier_init(smem + 376, 32);
            mbarrier_init(smem + 384, 32);
            // throttle_empty: 3 barriers, init_count=32
            mbarrier_init(smem + 392, 32);
            mbarrier_init(smem + 400, 32);
            mbarrier_init(smem + 408, 32);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (256 columns, 176 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 424);
    if (warp == 0) {
        int _tmem_hold = smem + 424;
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
    const int tmem_sfa = taddr + 128;
    const int tmem_sfb = taddr + 160;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 8 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
    }

    // ---- Role: epilogue ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 168;");
        { // epilogue_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            const int warp_0 = warp;
            const int lane_1 = lane;
            unsigned int acc_stage = 0;
            unsigned int work_stage = 0;
            unsigned int m_tile = blockIdx.x;
            unsigned int n_tile = blockIdx.y;
            int base_row = warp_0 * 16 + lane_1 / 4 * 2;
            int sf_block = warp_0 / 2;
            int sfc_row_group_stride = (M_out + 127) / 128 * 32;
            float values[32] = {0};
            float block_max[16] = {0};
            unsigned int _phase_mma_full = 0;
            unsigned int _phase_work_full = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter = 0; _tile_iter < grid_m / 2 * grid_n; _tile_iter++) {
                if (m_tile >= (unsigned int)grid_m || n_tile >= (unsigned int)num_non_exiting_ctas[0]) {
                    break;
                }
                int expert = tile_expert[n_tile];
                int valid_rows = (unsigned int)tile_mn_limit[n_tile] - n_tile * 64;
                float cl = clamp_limit[expert];
                float neg_cl = -cl;
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
                asm volatile("cp.async.bulk.wait_group.read 0;");
                asm volatile("barrier.sync 7, 128;" ::: "memory");
                for (int token_group = 0; token_group < 8; token_group++) {
                    float x0 = 0.0f;
                    float x1 = 0.0f;
                    float sig = 0.0f;
                    float swish = 0.0f;
                    float value = 0.0f;
                    x0 = _tmem_load_0[token_group * 4];
                    x1 = _tmem_load_0[token_group * 4 + 2];
                    float _exp2_0 = approx_exp2(-(x1 * 1.4426950408889634f));
                    float _rcp_0 = approx_rcp(1.0f + _exp2_0);
                    sig = _rcp_0;
                    swish = x1 * sig;
                    float _max_0 = max_noftz(x0, neg_cl);
                    float _min_0 = fminf(_max_0, cl);
                    float _min_1 = fminf(swish, cl);
                    value = _min_0 * _min_1;
                    values[token_group * 4] = value;
                    x0 = _tmem_load_0[token_group * 4 + 1];
                    x1 = _tmem_load_0[token_group * 4 + 2 + 1];
                    float _exp2_1 = approx_exp2(-(x1 * 1.4426950408889634f));
                    float _rcp_1 = approx_rcp(1.0f + _exp2_1);
                    sig = _rcp_1;
                    swish = x1 * sig;
                    float _max_1 = max_noftz(x0, neg_cl);
                    float _min_2 = fminf(_max_1, cl);
                    float _min_3 = fminf(swish, cl);
                    value = _min_2 * _min_3;
                    values[token_group * 4 + 1] = value;
                    x0 = _tmem_load_1[token_group * 4];
                    x1 = _tmem_load_1[token_group * 4 + 2];
                    float _exp2_2 = approx_exp2(-(x1 * 1.4426950408889634f));
                    float _rcp_2 = approx_rcp(1.0f + _exp2_2);
                    sig = _rcp_2;
                    swish = x1 * sig;
                    float _max_2 = max_noftz(x0, neg_cl);
                    float _min_4 = fminf(_max_2, cl);
                    float _min_5 = fminf(swish, cl);
                    value = _min_4 * _min_5;
                    values[token_group * 4 + 2] = value;
                    x0 = _tmem_load_1[token_group * 4 + 1];
                    x1 = _tmem_load_1[token_group * 4 + 2 + 1];
                    float _exp2_3 = approx_exp2(-(x1 * 1.4426950408889634f));
                    float _rcp_3 = approx_rcp(1.0f + _exp2_3);
                    sig = _rcp_3;
                    swish = x1 * sig;
                    float _max_3 = max_noftz(x0, neg_cl);
                    float _min_6 = fminf(_max_3, cl);
                    float _min_7 = fminf(swish, cl);
                    value = _min_6 * _min_7;
                    values[token_group * 4 + 2 + 1] = value;
                    float _fabs_0 = fabsf(values[token_group * 4]);
                    float _fabs_1 = fabsf(values[token_group * 4 + 2]);
                    float _max_4 = max_noftz(_fabs_0, _fabs_1);
                    float bm0 = _max_4;
                    float _fabs_2 = fabsf(values[token_group * 4 + 1]);
                    float _fabs_3 = fabsf(values[token_group * 4 + 3]);
                    float _max_5 = max_noftz(_fabs_2, _fabs_3);
                    float bm1 = _max_5;
                    float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, bm0, 4);
                    float _max_6 = max_noftz(bm0, _shfl_xor_0);
                    bm0 = _max_6;
                    float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, bm1, 4);
                    float _max_7 = max_noftz(bm1, _shfl_xor_1);
                    bm1 = _max_7;
                    float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, bm0, 8);
                    float _max_8 = max_noftz(bm0, _shfl_xor_2);
                    bm0 = _max_8;
                    float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, bm1, 8);
                    float _max_9 = max_noftz(bm1, _shfl_xor_3);
                    bm1 = _max_9;
                    float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, bm0, 16);
                    float _max_10 = max_noftz(bm0, _shfl_xor_4);
                    bm0 = _max_10;
                    float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, bm1, 16);
                    float _max_11 = max_noftz(bm1, _shfl_xor_5);
                    bm1 = _max_11;
                    block_max[token_group * 2] = bm0;
                    block_max[token_group * 2 + 1] = bm1;
                    int token0 = lane_1 % 4 * 2 + token_group * 8;
                    if (lane_1 < 4) {
                        amax_smem[warp_0 * 64 + token0] = bm0;
                        amax_smem[warp_0 * 64 + token0 + 1] = bm1;
                    }
                }
                asm volatile("barrier.sync 7, 128;" ::: "memory");
                for (int token_group_1 = 0; token_group_1 < 8; token_group_1++) {
                    int token0_1 = lane_1 % 4 * 2 + token_group_1 * 8;
                    int token1 = token0_1 + 1;
                    float peer0 = amax_smem[(warp_0 ^ 1) * 64 + token0_1];
                    float peer1 = amax_smem[(warp_0 ^ 1) * 64 + token1];
                    float _max_12 = max_noftz(block_max[token_group_1 * 2], peer0);
                    float amax0 = _max_12;
                    float _max_13 = max_noftz(block_max[token_group_1 * 2 + 1], peer1);
                    float amax1 = _max_13;
                    amax0 = ((amax0 >= 1.1754943508222875e-38f) ? amax0 : 0.0f);
                    amax1 = ((amax1 >= 1.1754943508222875e-38f) ? amax1 : 0.0f);
                    float sf0 = ((amax0 != 0.0f) ? amax0 : 448.0f) * 0.0022321429569274187f;
                    float sf1 = ((amax1 != 0.0f) ? amax1 : 448.0f) * 0.0022321429569274187f;
                    uint16_t _ue8m0x2_f32_0;
                    asm("cvt.rp.satfinite.ue8m0x2.f32 %0, %1, %2;" : "=h"(_ue8m0x2_f32_0) : "f"(sf1), "f"(sf0));
                    unsigned int code0 = (unsigned int)_ue8m0x2_f32_0 & 255;
                    unsigned int code1 = (unsigned int)_ue8m0x2_f32_0 >> 8 & 255;
                    float inv0 = __uint_as_float(254 - code0 << 23);
                    float inv1 = __uint_as_float(254 - code1 << 23);
                    float2 _f2_0 = make_float2(values[token_group_1 * 4], values[token_group_1 * 4 + 2]);
                    float2 _f2_1 = make_float2(inv0, inv0);
                    float2 _mul_f32x2_0;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_0) : "l"(*(const unsigned long long*)&_f2_0), "l"(*(const unsigned long long*)&_f2_1));
                    float2 _f2_2 = make_float2(values[token_group_1 * 4 + 1], values[token_group_1 * 4 + 3]);
                    float2 _f2_3 = make_float2(inv1, inv1);
                    float2 _mul_f32x2_1;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_1) : "l"(*(const unsigned long long*)&_f2_2), "l"(*(const unsigned long long*)&_f2_3));
                    uint16_t _e4m3x2_f32_0;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_0) : "f"(_mul_f32x2_0.y), "f"(_mul_f32x2_0.x));
                    uint16_t _e4m3x2_f32_1;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_1) : "f"(_mul_f32x2_1.y), "f"(_mul_f32x2_1.x));
                    if (warp_0 % 2 == 0 && lane_1 < 8) {
                        int token_sel = token0_1 + lane_1 / 4;
                        unsigned int code_sel = ((lane_1 / 4 == 0) ? code0 : code1);
                        int elt_n = n_tile * 64 + (unsigned int)token_sel;
                        int elt_m_block = m_tile * 2 + (unsigned int)sf_block;
                        int sf_index = elt_n / 8 * sfc_row_group_stride + elt_m_block / 4 * 32 + elt_n % 8 * 4 + elt_m_block % 4;
                        if (token_sel < valid_rows) {
                            SFC[sf_index] = code_sel;
                        }
                    }
                    int flat0 = token0_1 * 64 + base_row;
                    int flat1 = token1 * 64 + base_row;
                    epi_pairs[flat0 / 2 ^ flat0 / 128 % 4 * 8] = _e4m3x2_f32_0;
                    epi_pairs[flat1 / 2 ^ flat1 / 128 % 4 * 8] = _e4m3x2_f32_1;
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 7, 128;" ::: "memory");
                if (warp == 0) {
                    if (elect_sync()) {
                        int padding_rows = (64 - valid_rows % 64) % 64;
                        tma_store_4d((&C), m_tile * 64, padding_rows, 1073741824, n_tile * 64 - (unsigned int)padding_rows + 1073741824, epi_staging_addr);
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
                unsigned int raw_valid = 0;
                unsigned int valid = 0;
                unsigned int next_x = 0;
                unsigned int next_y = 0;
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
                    : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                    : "memory");
                raw_valid = _clc_valid_7;
                valid = raw_valid;
                uint32_t _clc_ctaid_14 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(_clc_ctaid_14)
                    : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                    : "memory");
                next_x = _clc_ctaid_14;
                uint32_t _clc_ctaid_15 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(_clc_ctaid_15)
                    : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                    : "memory");
                next_y = _clc_ctaid_15;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                if (next_y >= (unsigned int)num_non_exiting_ctas[0]) {
                    valid = 0;
                }
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + work_stage * 8), "r"(0) : "memory");
                work_stage += 1;
                if (work_stage == 3) { work_stage = 0; _phase_work_full ^= 1; }
                if (valid == 0) {
                    break;
                }
                m_tile = next_x + (unsigned int)cta_rank;
                n_tile = next_y;
            }
        }
    }
    // ---- Role: copy_sfb ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 72;");
        { // copy_sfb_main
            unsigned int stage = 0;
            unsigned int work_stage_1 = 0;
            unsigned int m_tile_1 = blockIdx.x;
            unsigned int n_tile_1 = blockIdx.y;
            const int lane_0 = lane;
            unsigned int word[2];
            unsigned int _phase_sfb_full = 0;
            unsigned int _phase_k_done = 1;
            unsigned int _phase_work_full_1 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_1 = 0; _tile_iter_1 < grid_m / 2 * grid_n; _tile_iter_1++) {
                if (m_tile_1 >= (unsigned int)grid_m || n_tile_1 >= (unsigned int)num_non_exiting_ctas[0]) {
                    break;
                }
                #pragma unroll 1
                for (int _iter_k = 0; _iter_k < K_tiles; _iter_k++) {
                    mbarrier_wait(sfb_full_addr + (stage) * 8, _phase_sfb_full);
                    mbarrier_wait(k_done_addr + (stage) * 8, _phase_k_done);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    const int logical_lane = lane_0;
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&word[0])) : "r"(smem_sfb_addr + stage * 512 + (unsigned int)(logical_lane / 8 * 2 * 32) + (unsigned int)(logical_lane % 8 * 4)));
                    const int logical_lane_0 = lane_0 + 32;
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&word[1])) : "r"(smem_sfb_addr + stage * 512 + (unsigned int)(logical_lane_0 / 8 * 2 * 32) + (unsigned int)(logical_lane_0 % 8 * 4)));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x2.b32"
                        " [%0], {%1, %2};"
                        :: "r"(taddr + 160 + stage * 4), "r"(word[0]), "r"(word[1]));
                    const int logical_lane_1 = lane_0;
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&word[0])) : "r"(smem_sfb_addr + stage * 512 + (unsigned int)((logical_lane_1 / 8 * 2 + 1) * 32) + (unsigned int)(logical_lane_1 % 8 * 4)));
                    const int logical_lane_2 = lane_0 + 32;
                    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&word[1])) : "r"(smem_sfb_addr + stage * 512 + (unsigned int)((logical_lane_2 / 8 * 2 + 1) * 32) + (unsigned int)(logical_lane_2 % 8 * 4)));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x2.b32"
                        " [%0], {%1, %2};"
                        :: "r"(taddr + 160 + stage * 4 + 2), "r"(word[0]), "r"(word[1]));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("barrier.sync 4, 128;" ::: "memory");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((tmem_sfb_full_addr + (stage) * 8) & 0xFEFFFFFF) : "memory");
                    mbarrier_arrive(sfb_free_addr + (stage) * 8);
                    stage += 1;
                    if (stage == 4) { stage = 0; _phase_sfb_full ^= 1; _phase_k_done ^= 1; }
                }
                mbarrier_wait(work_full_addr + (work_stage_1) * 8, _phase_work_full_1);
                unsigned int raw_valid_1 = 0;
                unsigned int valid_1 = 0;
                unsigned int next_x_1 = 0;
                unsigned int next_y_1 = 0;
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
                    : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                    : "memory");
                raw_valid_1 = _clc_valid_5;
                valid_1 = raw_valid_1;
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
                    : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                    : "memory");
                next_x_1 = _clc_ctaid_10;
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
                    : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                    : "memory");
                next_y_1 = _clc_ctaid_11;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                if (next_y_1 >= (unsigned int)num_non_exiting_ctas[0]) {
                    valid_1 = 0;
                }
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + work_stage_1 * 8), "r"(0) : "memory");
                work_stage_1 += 1;
                if (work_stage_1 == 3) { work_stage_1 = 0; _phase_work_full_1 ^= 1; }
                if (valid_1 == 0) {
                    break;
                }
                m_tile_1 = next_x_1 + (unsigned int)cta_rank;
                n_tile_1 = next_y_1;
            }
        }
    }
    // ---- Role: load_b ----
    if (warp == 8) {
        { // load_b_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int stage_1 = 0;
            unsigned int work_stage_2 = 0;
            unsigned int m_tile_2 = blockIdx.x;
            unsigned int n_tile_2 = blockIdx.y;
            unsigned int cta_mask = 1 << cta_rank;
            int routed[32];
            unsigned int _phase_k_done_1 = 1;
            unsigned int _phase_work_full_2 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_2 = 0; _tile_iter_2 < grid_m / 2 * grid_n; _tile_iter_2++) {
                if (m_tile_2 >= (unsigned int)grid_m || n_tile_2 >= (unsigned int)num_non_exiting_ctas[0]) {
                    break;
                }
                int route_base = n_tile_2 * 64 + (unsigned int)(cta_rank * 32);
                for (int row = 0; row < 32; row++) {
                    routed[row] = route_map[route_base + row];
                }
                #pragma unroll 1
                for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                    mbarrier_wait(k_done_addr + (stage_1) * 8, _phase_k_done_1);
                    if (elect_sync()) {
                        tma_gather4_gmem2smem_mc_cta2(smem_b_addr + stage_1 * 8192, (&B), iter_k * 256, routed[0], routed[1], routed[2], routed[3], ((b_full_addr + (stage_1) * 8) & 0xFEFFFFFF), cta_mask);
                        tma_gather4_gmem2smem_mc_cta2(smem_b_addr + stage_1 * 8192 + 4096, (&B), iter_k * 256 + 128, routed[0], routed[1], routed[2], routed[3], ((b_full_addr + (stage_1) * 8) & 0xFEFFFFFF), cta_mask);
                        tma_gather4_gmem2smem_mc_cta2(smem_b_addr + stage_1 * 8192 + 512, (&B), iter_k * 256, routed[4], routed[5], routed[6], routed[7], ((b_full_addr + (stage_1) * 8) & 0xFEFFFFFF), cta_mask);
                        tma_gather4_gmem2smem_mc_cta2(smem_b_addr + stage_1 * 8192 + 4096 + 512, (&B), iter_k * 256 + 128, routed[4], routed[5], routed[6], routed[7], ((b_full_addr + (stage_1) * 8) & 0xFEFFFFFF), cta_mask);
                        tma_gather4_gmem2smem_mc_cta2(smem_b_addr + stage_1 * 8192 + 1024, (&B), iter_k * 256, routed[8], routed[9], routed[10], routed[11], ((b_full_addr + (stage_1) * 8) & 0xFEFFFFFF), cta_mask);
                        tma_gather4_gmem2smem_mc_cta2(smem_b_addr + stage_1 * 8192 + 4096 + 1024, (&B), iter_k * 256 + 128, routed[8], routed[9], routed[10], routed[11], ((b_full_addr + (stage_1) * 8) & 0xFEFFFFFF), cta_mask);
                        tma_gather4_gmem2smem_mc_cta2(smem_b_addr + stage_1 * 8192 + 1536, (&B), iter_k * 256, routed[12], routed[13], routed[14], routed[15], ((b_full_addr + (stage_1) * 8) & 0xFEFFFFFF), cta_mask);
                        tma_gather4_gmem2smem_mc_cta2(smem_b_addr + stage_1 * 8192 + 4096 + 1536, (&B), iter_k * 256 + 128, routed[12], routed[13], routed[14], routed[15], ((b_full_addr + (stage_1) * 8) & 0xFEFFFFFF), cta_mask);
                        tma_gather4_gmem2smem_mc_cta2(smem_b_addr + stage_1 * 8192 + 2048, (&B), iter_k * 256, routed[16], routed[17], routed[18], routed[19], ((b_full_addr + (stage_1) * 8) & 0xFEFFFFFF), cta_mask);
                        tma_gather4_gmem2smem_mc_cta2(smem_b_addr + stage_1 * 8192 + 4096 + 2048, (&B), iter_k * 256 + 128, routed[16], routed[17], routed[18], routed[19], ((b_full_addr + (stage_1) * 8) & 0xFEFFFFFF), cta_mask);
                        tma_gather4_gmem2smem_mc_cta2(smem_b_addr + stage_1 * 8192 + 2560, (&B), iter_k * 256, routed[20], routed[21], routed[22], routed[23], ((b_full_addr + (stage_1) * 8) & 0xFEFFFFFF), cta_mask);
                        tma_gather4_gmem2smem_mc_cta2(smem_b_addr + stage_1 * 8192 + 4096 + 2560, (&B), iter_k * 256 + 128, routed[20], routed[21], routed[22], routed[23], ((b_full_addr + (stage_1) * 8) & 0xFEFFFFFF), cta_mask);
                        tma_gather4_gmem2smem_mc_cta2(smem_b_addr + stage_1 * 8192 + 3072, (&B), iter_k * 256, routed[24], routed[25], routed[26], routed[27], ((b_full_addr + (stage_1) * 8) & 0xFEFFFFFF), cta_mask);
                        tma_gather4_gmem2smem_mc_cta2(smem_b_addr + stage_1 * 8192 + 4096 + 3072, (&B), iter_k * 256 + 128, routed[24], routed[25], routed[26], routed[27], ((b_full_addr + (stage_1) * 8) & 0xFEFFFFFF), cta_mask);
                        tma_gather4_gmem2smem_mc_cta2(smem_b_addr + stage_1 * 8192 + 3584, (&B), iter_k * 256, routed[28], routed[29], routed[30], routed[31], ((b_full_addr + (stage_1) * 8) & 0xFEFFFFFF), cta_mask);
                        tma_gather4_gmem2smem_mc_cta2(smem_b_addr + stage_1 * 8192 + 4096 + 3584, (&B), iter_k * 256 + 128, routed[28], routed[29], routed[30], routed[31], ((b_full_addr + (stage_1) * 8) & 0xFEFFFFFF), cta_mask);
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((b_full_addr + (stage_1) * 8) & 0xFEFFFFFF), "r"((uint32_t)(8192)) : "memory");
                    }
                    stage_1 += 1;
                    if (stage_1 == 4) { stage_1 = 0; _phase_k_done_1 ^= 1; }
                }
                mbarrier_wait(work_full_addr + (work_stage_2) * 8, _phase_work_full_2);
                unsigned int raw_valid_2 = 0;
                unsigned int valid_2 = 0;
                unsigned int next_x_2 = 0;
                unsigned int next_y_2 = 0;
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
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                    : "memory");
                raw_valid_2 = _clc_valid_1;
                valid_2 = raw_valid_2;
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
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                    : "memory");
                next_x_2 = _clc_ctaid_2;
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
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                    : "memory");
                next_y_2 = _clc_ctaid_3;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                if (next_y_2 >= (unsigned int)num_non_exiting_ctas[0]) {
                    valid_2 = 0;
                }
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + work_stage_2 * 8), "r"(0) : "memory");
                work_stage_2 += 1;
                if (work_stage_2 == 3) { work_stage_2 = 0; _phase_work_full_2 ^= 1; }
                if (valid_2 == 0) {
                    break;
                }
                m_tile_2 = next_x_2 + (unsigned int)cta_rank;
                n_tile_2 = next_y_2;
            }
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
    }
    // ---- Role: load_sfb ----
    if (warp == 9) {
        { // load_sfb_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int stage_2 = 0;
            unsigned int work_stage_3 = 0;
            unsigned int m_tile_3 = blockIdx.x;
            unsigned int n_tile_3 = blockIdx.y;
            const int lane_0_1 = lane;
            int k_group = lane_0_1 % 2;
            int sf_stride = K / 32;
            int routed_1[4];
            unsigned int _phase_sfb_free = 1;
            unsigned int _phase_work_full_3 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_3 = 0; _tile_iter_3 < grid_m / 2 * grid_n; _tile_iter_3++) {
                if (m_tile_3 >= (unsigned int)grid_m || n_tile_3 >= (unsigned int)num_non_exiting_ctas[0]) {
                    break;
                }
                int valid_rows_1 = (unsigned int)tile_mn_limit[n_tile_3] - n_tile_3 * 64;
                routed_1[0] = route_map[n_tile_3 * 64 + (unsigned int)(lane_0_1 / 2)];
                routed_1[1] = route_map[n_tile_3 * 64 + (unsigned int)(lane_0_1 / 2) + 16];
                routed_1[2] = route_map[n_tile_3 * 64 + (unsigned int)(lane_0_1 / 2) + 32];
                routed_1[3] = route_map[n_tile_3 * 64 + (unsigned int)(lane_0_1 / 2) + 48];
                #pragma unroll 1
                for (int iter_k_1 = 0; iter_k_1 < K_tiles; iter_k_1++) {
                    mbarrier_wait(sfb_free_addr + (stage_2) * 8, _phase_sfb_free);
                    int row_1 = lane_0_1 / 2;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p;\n\t"
                        "setp.ne.b32 p, %0, 0;\n\t"
                        "@p cp.async.ca.shared::cta.global [%1], [%2], 4;\n\t"
                        "}"
                        :: "r"((row_1 < valid_rows_1) ? 1 : 0), "r"(smem_sfb_addr + stage_2 * 512 + (unsigned int)((row_1 / 8 * 2 + k_group) * 32) + (unsigned int)(row_1 % 8 * 4)), "l"(SFB + (routed_1[0] * sf_stride + iter_k_1 * 8 + k_group * 4)));
                    int row_0 = lane_0_1 / 2 + 16;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p;\n\t"
                        "setp.ne.b32 p, %0, 0;\n\t"
                        "@p cp.async.ca.shared::cta.global [%1], [%2], 4;\n\t"
                        "}"
                        :: "r"((row_0 < valid_rows_1) ? 1 : 0), "r"(smem_sfb_addr + stage_2 * 512 + (unsigned int)((row_0 / 8 * 2 + k_group) * 32) + (unsigned int)(row_0 % 8 * 4)), "l"(SFB + (routed_1[1] * sf_stride + iter_k_1 * 8 + k_group * 4)));
                    int row_1_1 = lane_0_1 / 2 + 32;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p;\n\t"
                        "setp.ne.b32 p, %0, 0;\n\t"
                        "@p cp.async.ca.shared::cta.global [%1], [%2], 4;\n\t"
                        "}"
                        :: "r"((row_1_1 < valid_rows_1) ? 1 : 0), "r"(smem_sfb_addr + stage_2 * 512 + (unsigned int)((row_1_1 / 8 * 2 + k_group) * 32) + (unsigned int)(row_1_1 % 8 * 4)), "l"(SFB + (routed_1[2] * sf_stride + iter_k_1 * 8 + k_group * 4)));
                    int row_2 = lane_0_1 / 2 + 48;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p;\n\t"
                        "setp.ne.b32 p, %0, 0;\n\t"
                        "@p cp.async.ca.shared::cta.global [%1], [%2], 4;\n\t"
                        "}"
                        :: "r"((row_2 < valid_rows_1) ? 1 : 0), "r"(smem_sfb_addr + stage_2 * 512 + (unsigned int)((row_2 / 8 * 2 + k_group) * 32) + (unsigned int)(row_2 % 8 * 4)), "l"(SFB + (routed_1[3] * sf_stride + iter_k_1 * 8 + k_group * 4)));
                    asm volatile(
                        "{\n\t"
                        "cp.async.mbarrier.arrive.shared::cta.b64 [%0];\n\t"
                        "}"
                        :: "r"(sfb_full_addr + (stage_2) * 8) : "memory");
                    asm volatile("barrier.sync 8, 32;" ::: "memory");
                    if (elect_sync()) {
                        mbarrier_arrive(sfb_full_addr + (stage_2) * 8);
                    }
                    stage_2 += 1;
                    if (stage_2 == 4) { stage_2 = 0; _phase_sfb_free ^= 1; }
                }
                mbarrier_wait(work_full_addr + (work_stage_3) * 8, _phase_work_full_3);
                unsigned int raw_valid_3 = 0;
                unsigned int valid_3 = 0;
                unsigned int next_x_3 = 0;
                unsigned int next_y_3 = 0;
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
                    : "r"(work_response_addr + work_stage_3 * 16 + 0 * 16)
                    : "memory");
                raw_valid_3 = _clc_valid_3;
                valid_3 = raw_valid_3;
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
                    : "r"(work_response_addr + work_stage_3 * 16 + 0 * 16)
                    : "memory");
                next_x_3 = _clc_ctaid_6;
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
                    : "r"(work_response_addr + work_stage_3 * 16 + 0 * 16)
                    : "memory");
                next_y_3 = _clc_ctaid_7;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                if (next_y_3 >= (unsigned int)num_non_exiting_ctas[0]) {
                    valid_3 = 0;
                }
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + work_stage_3 * 8), "r"(0) : "memory");
                work_stage_3 += 1;
                if (work_stage_3 == 3) { work_stage_3 = 0; _phase_work_full_3 ^= 1; }
                if (valid_3 == 0) {
                    break;
                }
                m_tile_3 = next_x_3 + (unsigned int)cta_rank;
                n_tile_3 = next_y_3;
            }
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
    }
    // ---- Role: load_a ----
    if (warp == 10) {
        { // load_a_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int stage_3 = 0;
            unsigned int work_stage_4 = 0;
            unsigned int throttle_stage = 0;
            unsigned int m_tile_4 = blockIdx.x;
            unsigned int n_tile_4 = blockIdx.y;
            unsigned int cta_mask_1 = 1 << cta_rank;
            unsigned int _phase_throttle_empty = 1;
            unsigned int _phase_k_done_2 = 1;
            unsigned int _phase_work_full_4 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_4 = 0; _tile_iter_4 < grid_m / 2 * grid_n; _tile_iter_4++) {
                if (m_tile_4 >= (unsigned int)grid_m || n_tile_4 >= (unsigned int)num_non_exiting_ctas[0]) {
                    break;
                }
                int expert_1 = tile_expert[n_tile_4];
                if (cta_rank == 0) {
                    mbarrier_wait(throttle_empty_addr + (throttle_stage) * 8, _phase_throttle_empty);
                    mbarrier_arrive(throttle_full_addr + (throttle_stage) * 8);
                    throttle_stage += 1;
                    if (throttle_stage == 3) { throttle_stage = 0; _phase_throttle_empty ^= 1; }
                }
                #pragma unroll 1
                for (int iter_k_2 = 0; iter_k_2 < K_tiles; iter_k_2++) {
                    mbarrier_wait(k_done_addr + (stage_3) * 8, _phase_k_done_2);
                    if (elect_sync()) {
                        asm volatile(
                            "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5}], [%6], %7, %8;"
                            :: "r"(smem_a_addr + stage_3 * 32768), "l"((&A)), "r"(0), "r"(m_tile_4 * 128), "r"(iter_k_2 * 2), "r"(expert_1),
                               "r"(((a_full_addr + (stage_3) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(cta_mask_1)), "l"(0x12F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5}], [%6], %7, %8;"
                            :: "r"(smem_a_addr + stage_3 * 32768 + 16384), "l"((&A)), "r"(0), "r"(m_tile_4 * 128), "r"(iter_k_2 * 2 + 1), "r"(expert_1),
                               "r"(((a_full_addr + (stage_3) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(cta_mask_1)), "l"(0x12F0000000000000ULL) : "memory");
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((a_full_addr + (stage_3) * 8) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                    }
                    stage_3 += 1;
                    if (stage_3 == 4) { stage_3 = 0; _phase_k_done_2 ^= 1; }
                }
                mbarrier_wait(work_full_addr + (work_stage_4) * 8, _phase_work_full_4);
                unsigned int raw_valid_4 = 0;
                unsigned int valid_4 = 0;
                unsigned int next_x_4 = 0;
                unsigned int next_y_4 = 0;
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
                    : "r"(work_response_addr + work_stage_4 * 16 + 0 * 16)
                    : "memory");
                raw_valid_4 = _clc_valid_0;
                valid_4 = raw_valid_4;
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
                    : "r"(work_response_addr + work_stage_4 * 16 + 0 * 16)
                    : "memory");
                next_x_4 = _clc_ctaid_0;
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
                    : "r"(work_response_addr + work_stage_4 * 16 + 0 * 16)
                    : "memory");
                next_y_4 = _clc_ctaid_1;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                if (next_y_4 >= (unsigned int)num_non_exiting_ctas[0]) {
                    valid_4 = 0;
                }
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + work_stage_4 * 8), "r"(0) : "memory");
                work_stage_4 += 1;
                if (work_stage_4 == 3) { work_stage_4 = 0; _phase_work_full_4 ^= 1; }
                if (valid_4 == 0) {
                    break;
                }
                m_tile_4 = next_x_4 + (unsigned int)cta_rank;
                n_tile_4 = next_y_4;
            }
        }
    }
    // ---- Role: load_sfa ----
    if (warp == 11) {
        { // load_sfa_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int stage_4 = 0;
            unsigned int work_stage_5 = 0;
            unsigned int m_tile_5 = blockIdx.x;
            unsigned int n_tile_5 = blockIdx.y;
            unsigned int cta_mask_2 = 1 << cta_rank;
            unsigned int _phase_sfa_free = 1;
            unsigned int _phase_work_full_5 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_5 = 0; _tile_iter_5 < grid_m / 2 * grid_n; _tile_iter_5++) {
                if (m_tile_5 >= (unsigned int)grid_m || n_tile_5 >= (unsigned int)num_non_exiting_ctas[0]) {
                    break;
                }
                int expert_2 = tile_expert[n_tile_5];
                #pragma unroll 1
                for (int iter_k_3 = 0; iter_k_3 < K_tiles; iter_k_3++) {
                    mbarrier_wait(sfa_free_addr + (stage_4) * 8, _phase_sfa_free);
                    if (elect_sync()) {
                        asm volatile(
                            "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5}], [%6], %7, %8;"
                            :: "r"(smem_sfa_addr + stage_4 * 1024), "l"((&SFA)), "r"(0), "r"(0), "r"(iter_k_3 * 2), "r"((unsigned int)(expert_2 * grid_m) + m_tile_5),
                               "r"(((sfa_full_addr + (stage_4) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(cta_mask_2)), "l"(0x12F0000000000000ULL) : "memory");
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((sfa_full_addr + (stage_4) * 8) & 0xFEFFFFFF), "r"((uint32_t)(1024)) : "memory");
                    }
                    stage_4 += 1;
                    if (stage_4 == 4) { stage_4 = 0; _phase_sfa_free ^= 1; }
                }
                mbarrier_wait(work_full_addr + (work_stage_5) * 8, _phase_work_full_5);
                unsigned int raw_valid_5 = 0;
                unsigned int valid_5 = 0;
                unsigned int next_x_5 = 0;
                unsigned int next_y_5 = 0;
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
                    : "r"(work_response_addr + work_stage_5 * 16 + 0 * 16)
                    : "memory");
                raw_valid_5 = _clc_valid_2;
                valid_5 = raw_valid_5;
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
                    : "r"(work_response_addr + work_stage_5 * 16 + 0 * 16)
                    : "memory");
                next_x_5 = _clc_ctaid_4;
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
                    : "r"(work_response_addr + work_stage_5 * 16 + 0 * 16)
                    : "memory");
                next_y_5 = _clc_ctaid_5;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                if (next_y_5 >= (unsigned int)num_non_exiting_ctas[0]) {
                    valid_5 = 0;
                }
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + work_stage_5 * 8), "r"(0) : "memory");
                work_stage_5 += 1;
                if (work_stage_5 == 3) { work_stage_5 = 0; _phase_work_full_5 ^= 1; }
                if (valid_5 == 0) {
                    break;
                }
                m_tile_5 = next_x_5 + (unsigned int)cta_rank;
                n_tile_5 = next_y_5;
            }
        }
    }
    // ---- Role: copy_sfa ----
    if (warp == 12) {
        { // copy_sfa_main
            unsigned int _phase_sfa_full = 0;
            unsigned int _phase_k_done_3 = 1;
            unsigned int _phase_work_full_6 = 0;
            if (cta_rank == 0) {
                unsigned int stage_5 = 0;
                unsigned int work_stage_6 = 0;
                unsigned int m_tile_6 = blockIdx.x;
                unsigned int n_tile_6 = blockIdx.y;
                #pragma unroll 1
                for (unsigned int _tile_iter_6 = 0; _tile_iter_6 < grid_m / 2 * grid_n; _tile_iter_6++) {
                    if (m_tile_6 >= (unsigned int)grid_m || n_tile_6 >= (unsigned int)num_non_exiting_ctas[0]) {
                        break;
                    }
                    #pragma unroll 1
                    for (int _iter_k_1 = 0; _iter_k_1 < K_tiles; _iter_k_1++) {
                        mbarrier_wait(sfa_full_addr + (stage_5) * 8, _phase_sfa_full);
                        mbarrier_wait(k_done_addr + (stage_5) * 8, _phase_k_done_3);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        if (elect_sync()) {
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                            #endif
                            {
                                uint64_t _tcgen05_cp_desc_0 = ((((uint64_t)(smem_sfa_addr + stage_5 * 1024)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                asm volatile(
                                    "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                    :: "r"((uint32_t)((unsigned int)tmem_sfa + stage_5 * 8)), "l"(_tcgen05_cp_desc_0)
                                    : "memory");
                            }
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                            #endif
                            {
                                uint64_t _tcgen05_cp_desc_1 = ((((uint64_t)(smem_sfa_addr + stage_5 * 1024 + 512)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                asm volatile(
                                    "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                    :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_5 * 8 + 4))), "l"(_tcgen05_cp_desc_1)
                                    : "memory");
                            }
                        }
                        mbarrier_arrive(tmem_sfa_full_addr + (stage_5) * 8);
                        elect_commit_cg2_multicast(sfa_free_addr + (stage_5) * 8, (uint16_t)(3));
                        stage_5 += 1;
                        if (stage_5 == 4) { stage_5 = 0; _phase_sfa_full ^= 1; _phase_k_done_3 ^= 1; }
                    }
                    mbarrier_wait(work_full_addr + (work_stage_6) * 8, _phase_work_full_6);
                    unsigned int raw_valid_6 = 0;
                    unsigned int valid_6 = 0;
                    unsigned int next_x_6 = 0;
                    unsigned int next_y_6 = 0;
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
                        : "r"(work_response_addr + work_stage_6 * 16 + 0 * 16)
                        : "memory");
                    raw_valid_6 = _clc_valid_4;
                    valid_6 = raw_valid_6;
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
                        : "r"(work_response_addr + work_stage_6 * 16 + 0 * 16)
                        : "memory");
                    next_x_6 = _clc_ctaid_8;
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
                        : "r"(work_response_addr + work_stage_6 * 16 + 0 * 16)
                        : "memory");
                    next_y_6 = _clc_ctaid_9;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    if (next_y_6 >= (unsigned int)num_non_exiting_ctas[0]) {
                        valid_6 = 0;
                    }
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(work_empty_addr + work_stage_6 * 8), "r"(0) : "memory");
                    work_stage_6 += 1;
                    if (work_stage_6 == 3) { work_stage_6 = 0; _phase_work_full_6 ^= 1; }
                    if (valid_6 == 0) {
                        break;
                    }
                    m_tile_6 = next_x_6 + (unsigned int)cta_rank;
                    n_tile_6 = next_y_6;
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 13) {
        { // mma_main
            unsigned int _phase_mma_free = 1;
            unsigned int _phase_a_full = 0;
            unsigned int _phase_b_full = 0;
            unsigned int _phase_tmem_sfa_full = 0;
            unsigned int _phase_tmem_sfb_full = 0;
            unsigned int _phase_work_full_7 = 0;
            if (cta_rank == 0) {
                unsigned int stage_6 = 0;
                unsigned int acc_stage_1 = 0;
                unsigned int work_stage_7 = 0;
                unsigned int m_tile_7 = blockIdx.x;
                unsigned int n_tile_7 = blockIdx.y;
                #pragma unroll 1
                for (unsigned int _tile_iter_7 = 0; _tile_iter_7 < grid_m / 2 * grid_n; _tile_iter_7++) {
                    if (m_tile_7 >= (unsigned int)grid_m || n_tile_7 >= (unsigned int)num_non_exiting_ctas[0]) {
                        break;
                    }
                    mbarrier_wait(mma_free_addr + (acc_stage_1) * 8, _phase_mma_free);
                    #pragma unroll 2
                    for (int iter_k_4 = 0; iter_k_4 < K_tiles; iter_k_4++) {
                        mbarrier_wait(a_full_addr + (stage_6) * 8, _phase_a_full);
                        mbarrier_wait(b_full_addr + (stage_6) * 8, _phase_b_full);
                        mbarrier_wait(tmem_sfa_full_addr + (stage_6) * 8, _phase_tmem_sfa_full);
                        mbarrier_wait(tmem_sfb_full_addr + (stage_6) * 8, _phase_tmem_sfb_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_0 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_6) * 2048;
                        int _mma_b_lo_0 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_6) * 512;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf8_bs_cta2((tmem_accum + (acc_stage_1 * 64)), a_desc + 0, b_desc + 0,
                                    0x10900000U, (unsigned int)tmem_sfa + stage_6 * 8, (unsigned int)tmem_sfb + stage_6 * 4, ((((1) ? ((iter_k_4 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                                tcgen05_mma_mxf8_bs_cta2((tmem_accum + (acc_stage_1 * 64)), a_desc + 2, b_desc + 2,
                                    0x30900010U, (unsigned int)tmem_sfa + stage_6 * 8, (unsigned int)tmem_sfb + stage_6 * 4, 1);
                                tcgen05_mma_mxf8_bs_cta2((tmem_accum + (acc_stage_1 * 64)), a_desc + 4, b_desc + 4,
                                    0x50900020U, (unsigned int)tmem_sfa + stage_6 * 8, (unsigned int)tmem_sfb + stage_6 * 4, 1);
                                tcgen05_mma_mxf8_bs_cta2((tmem_accum + (acc_stage_1 * 64)), a_desc + 6, b_desc + 6,
                                    0x70900030U, (unsigned int)tmem_sfa + stage_6 * 8, (unsigned int)tmem_sfb + stage_6 * 4, 1);
                            }
                        }
                        int _mma_a_lo_1 = (((smem_a_addr + 16384) >> 4) & 0x3FFF) + (stage_6) * 2048;
                        int _mma_b_lo_1 = (((smem_b_addr + 4096) >> 4) & 0x3FFF) + (stage_6) * 512;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf8_bs_cta2((tmem_accum + (acc_stage_1 * 64)), a_desc + 0, b_desc + 0,
                                    0x10900000U, (unsigned int)tmem_sfa + (stage_6 * 8 + 4), (unsigned int)tmem_sfb + (stage_6 * 4 + 2), ((((0) ? ((iter_k_4 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                                tcgen05_mma_mxf8_bs_cta2((tmem_accum + (acc_stage_1 * 64)), a_desc + 2, b_desc + 2,
                                    0x30900010U, (unsigned int)tmem_sfa + (stage_6 * 8 + 4), (unsigned int)tmem_sfb + (stage_6 * 4 + 2), 1);
                                tcgen05_mma_mxf8_bs_cta2((tmem_accum + (acc_stage_1 * 64)), a_desc + 4, b_desc + 4,
                                    0x50900020U, (unsigned int)tmem_sfa + (stage_6 * 8 + 4), (unsigned int)tmem_sfb + (stage_6 * 4 + 2), 1);
                                tcgen05_mma_mxf8_bs_cta2((tmem_accum + (acc_stage_1 * 64)), a_desc + 6, b_desc + 6,
                                    0x70900030U, (unsigned int)tmem_sfa + (stage_6 * 8 + 4), (unsigned int)tmem_sfb + (stage_6 * 4 + 2), 1);
                            }
                        }
                        elect_commit_cg2_multicast(k_done_addr + (stage_6) * 8, (uint16_t)(3));
                        if (iter_k_4 + 1 == K_tiles) {
                            elect_commit_cg2_multicast(mma_full_addr + (acc_stage_1) * 8, (uint16_t)(3));
                        }
                        stage_6 += 1;
                        if (stage_6 == 4) { stage_6 = 0; _phase_a_full ^= 1; _phase_b_full ^= 1; _phase_tmem_sfa_full ^= 1; _phase_tmem_sfb_full ^= 1; }
                    }
                    acc_stage_1 += 1;
                    if (acc_stage_1 == 2) { acc_stage_1 = 0; _phase_mma_free ^= 1; }
                    mbarrier_wait(work_full_addr + (work_stage_7) * 8, _phase_work_full_7);
                    unsigned int raw_valid_7 = 0;
                    unsigned int valid_7 = 0;
                    unsigned int next_x_7 = 0;
                    unsigned int next_y_7 = 0;
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
                        : "r"(work_response_addr + work_stage_7 * 16 + 0 * 16)
                        : "memory");
                    raw_valid_7 = _clc_valid_6;
                    valid_7 = raw_valid_7;
                    uint32_t _clc_ctaid_12 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "+r"(_clc_ctaid_12)
                        : "r"(work_response_addr + work_stage_7 * 16 + 0 * 16)
                        : "memory");
                    next_x_7 = _clc_ctaid_12;
                    uint32_t _clc_ctaid_13 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "+r"(_clc_ctaid_13)
                        : "r"(work_response_addr + work_stage_7 * 16 + 0 * 16)
                        : "memory");
                    next_y_7 = _clc_ctaid_13;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    if (next_y_7 >= (unsigned int)num_non_exiting_ctas[0]) {
                        valid_7 = 0;
                    }
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(work_empty_addr + work_stage_7 * 8), "r"(0) : "memory");
                    work_stage_7 += 1;
                    if (work_stage_7 == 3) { work_stage_7 = 0; _phase_work_full_7 ^= 1; }
                    if (valid_7 == 0) {
                        break;
                    }
                    m_tile_7 = next_x_7 + (unsigned int)cta_rank;
                    n_tile_7 = next_y_7;
                }
            }
        }
    }
    // ---- Role: work_id ----
    if (warp == 14) {
        { // work_id_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int work_stage_8 = 0;
            unsigned int throttle_stage_1 = 0;
            unsigned int fast_stage = 0;
            unsigned int _phase_throttle_full = 0;
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_work_full_8 = 0;
            unsigned int _phase_fast_ready = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (unsigned int _tile_iter_8 = 0; _tile_iter_8 < grid_m / 2 * grid_n; _tile_iter_8++) {
                    mbarrier_wait(throttle_full_addr + (throttle_stage_1) * 8, _phase_throttle_full);
                    mbarrier_arrive(throttle_empty_addr + (throttle_stage_1) * 8);
                    throttle_stage_1 += 1;
                    if (throttle_stage_1 == 3) { throttle_stage_1 = 0; _phase_throttle_full ^= 1; }
                    mbarrier_wait_cluster_hint(work_empty_addr + (work_stage_8) * 8, _phase_work_empty, 10000000);
                    if (lane < 2) {
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                            "}"
                            :: "r"(work_full_addr + work_stage_8 * 8), "r"(lane), "r"((uint32_t)(16)) : "memory");
                    }
                    if (elect_sync()) {
                        asm volatile(
                            "fence.proxy.async.shared::cta;\n\t"
                            "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                                " [%0], [%1];"
                            :: "r"(work_response_addr + work_stage_8 * 16 + 0 * 16), "r"(work_full_addr + work_stage_8 * 8)
                            : "memory");
                    }
                    mbarrier_wait(work_full_addr + (work_stage_8) * 8, _phase_work_full_8);
                    unsigned int raw_valid_8 = 0;
                    unsigned int valid_8 = 0;
                    unsigned int next_x_8 = 0;
                    unsigned int next_y_8 = 0;
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
                        : "r"(work_response_addr + work_stage_8 * 16 + 0 * 16)
                        : "memory");
                    raw_valid_8 = _clc_valid_8;
                    valid_8 = raw_valid_8;
                    uint32_t _clc_ctaid_16 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "+r"(_clc_ctaid_16)
                        : "r"(work_response_addr + work_stage_8 * 16 + 0 * 16)
                        : "memory");
                    next_x_8 = _clc_ctaid_16;
                    uint32_t _clc_ctaid_17 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "+r"(_clc_ctaid_17)
                        : "r"(work_response_addr + work_stage_8 * 16 + 0 * 16)
                        : "memory");
                    next_y_8 = _clc_ctaid_17;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    if (next_y_8 >= (unsigned int)num_non_exiting_ctas[0]) {
                        valid_8 = 0;
                    }
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(work_empty_addr + work_stage_8 * 8), "r"(0) : "memory");
                    work_stage_8 += 1;
                    if (work_stage_8 == 3) { work_stage_8 = 0; _phase_work_empty ^= 1; _phase_work_full_8 ^= 1; }
                    if (valid_8 == 0) {
                        if (raw_valid_8 == 1) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            mbarrier_init(fast_ready_addr, 1);
                            #pragma unroll 1
                            for (unsigned int _drain_iter = 0; _drain_iter < grid_m / 2 * grid_n; _drain_iter++) {
                                if (elect_sync()) {
                                    mbarrier_arrive_expect_tx(fast_ready_addr + (fast_stage) * 8, 64);
                                    for (int slot_idx = 0; slot_idx < 4; slot_idx++) {
                                        asm volatile(
                                            "fence.proxy.async.shared::cta;\n\t"
                                            "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                                ".mbarrier::complete_tx::bytes.b128"
                                                " [%0], [%1];"
                                            :: "r"(fast_response_addr + fast_stage * 64 + slot_idx * 16), "r"(fast_ready_addr + fast_stage * 8)
                                            : "memory");
                                    }
                                }
                                mbarrier_wait(fast_ready_addr + (fast_stage) * 8, _phase_fast_ready);
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
                                    : "r"(fast_response_addr + fast_stage * 64 + 0 * 16)
                                    : "memory");
                                uint32_t _clc_valid_10 = 0;
                                asm volatile(
                                    "{\n\t"
                                    ".reg .pred p1;\n\t"
                                    ".reg .b128 clc_r;\n\t"
                                    "ld.shared.b128 clc_r, [%1];\n\t"
                                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                                    "selp.u32 %0, 1, 0, p1;\n\t"
                                    "}\n"
                                    : "=r"(_clc_valid_10)
                                    : "r"(fast_response_addr + fast_stage * 64 + 1 * 16)
                                    : "memory");
                                uint32_t _clc_valid_11 = 0;
                                asm volatile(
                                    "{\n\t"
                                    ".reg .pred p1;\n\t"
                                    ".reg .b128 clc_r;\n\t"
                                    "ld.shared.b128 clc_r, [%1];\n\t"
                                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                                    "selp.u32 %0, 1, 0, p1;\n\t"
                                    "}\n"
                                    : "=r"(_clc_valid_11)
                                    : "r"(fast_response_addr + fast_stage * 64 + 2 * 16)
                                    : "memory");
                                uint32_t _clc_valid_12 = 0;
                                asm volatile(
                                    "{\n\t"
                                    ".reg .pred p1;\n\t"
                                    ".reg .b128 clc_r;\n\t"
                                    "ld.shared.b128 clc_r, [%1];\n\t"
                                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                                    "selp.u32 %0, 1, 0, p1;\n\t"
                                    "}\n"
                                    : "=r"(_clc_valid_12)
                                    : "r"(fast_response_addr + fast_stage * 64 + 3 * 16)
                                    : "memory");
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                _phase_fast_ready ^= 1;
                                if (_clc_valid_9 + _clc_valid_10 + _clc_valid_11 + _clc_valid_12 == 0) {
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
            unsigned int work_stage_9 = 0;
            unsigned int m_tile_8 = blockIdx.x;
            unsigned int n_tile_8 = blockIdx.y;
            unsigned int _phase_work_full_9 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_9 = 0; _tile_iter_9 < grid_m / 2 * grid_n; _tile_iter_9++) {
                if (m_tile_8 >= (unsigned int)grid_m || n_tile_8 >= (unsigned int)num_non_exiting_ctas[0]) {
                    break;
                }
                mbarrier_wait(work_full_addr + (work_stage_9) * 8, _phase_work_full_9);
                unsigned int raw_valid_9 = 0;
                unsigned int valid_9 = 0;
                unsigned int next_x_9 = 0;
                unsigned int next_y_9 = 0;
                uint32_t _clc_valid_13 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_13)
                    : "r"(work_response_addr + work_stage_9 * 16 + 0 * 16)
                    : "memory");
                raw_valid_9 = _clc_valid_13;
                valid_9 = raw_valid_9;
                uint32_t _clc_ctaid_18 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(_clc_ctaid_18)
                    : "r"(work_response_addr + work_stage_9 * 16 + 0 * 16)
                    : "memory");
                next_x_9 = _clc_ctaid_18;
                uint32_t _clc_ctaid_19 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(_clc_ctaid_19)
                    : "r"(work_response_addr + work_stage_9 * 16 + 0 * 16)
                    : "memory");
                next_y_9 = _clc_ctaid_19;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                if (next_y_9 >= (unsigned int)num_non_exiting_ctas[0]) {
                    valid_9 = 0;
                }
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + work_stage_9 * 8), "r"(0) : "memory");
                work_stage_9 += 1;
                if (work_stage_9 == 3) { work_stage_9 = 0; _phase_work_full_9 ^= 1; }
                if (valid_9 == 0) {
                    break;
                }
                m_tile_8 = next_x_9 + (unsigned int)cta_rank;
                n_tile_8 = next_y_9;
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
