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
#define TMEM_NCOLS 496
#define TMEM_ACCUM_OFFSET 0
#define TMEM_SFA_OFFSET 448
#define TMEM_SFB_OFFSET 464
#define NUM_K_PIPE_STAGES 4
#define NUM_MMA_PIPE_STAGES 1
#define NUM_WORK_PIPE_STAGES 3
#define NUM_THROTTLE_PIPE_STAGES 3
#define NUM_FAST_PIPE_STAGES 1
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 16384
#define SMEM_SMEM_B_OFF 66560
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 16384
#define SMEM_EPI_STAGING_OFF 132096
#define SMEM_EPI_STAGING_STAGE_BYTES 4096
#define SMEM_EPI_STAGING_STRIDE 4096
#define SMEM_SMEM_SFA_OFF 145408
#define SMEM_SMEM_SFA_STAGE_BYTES 2048
#define SMEM_SMEM_SFA_STRIDE 2048
#define SMEM_SMEM_SFB_OFF 153600
#define SMEM_SMEM_SFB_STAGE_BYTES 4096
#define SMEM_SMEM_SFB_STRIDE 4096
#define SMEM_WORK_RESPONSE_OFF 169984
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_FAST_RESPONSE_OFF 170032
#define SMEM_FAST_RESPONSE_STAGE_BYTES 64
#define SMEM_FAST_RESPONSE_STRIDE 64
#define SMEM_TOTAL 170112
#define THREADS 640
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


__device__ __forceinline__ void tcgen05_mma_mxf4nvf4_bs_cta2(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
    int sfa_taddr, int sfb_taddr, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X"
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

__global__ __launch_bounds__(640, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_cake_stepfun_moe_6f24da6f906ef838a51e(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, int* __restrict__ SFB, const __grid_constant__ CUtensorMap C, uint8_t* __restrict__ SFC, int* __restrict__ route_map, int* __restrict__ tile_expert, int* __restrict__ tile_mn_limit, int* __restrict__ num_non_exiting_ctas, float* __restrict__ scale_c, float* __restrict__ scale_gate, float* __restrict__ clamp_limit, float* __restrict__ act_alpha, float* __restrict__ act_beta, int M_out, int K, int grid_m, int grid_n, int K_tiles)
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
    #define k_done_addr (mbar_base + 128)
    #define mma_full_addr (mbar_base + 160)
    #define mma_free_addr (mbar_base + 168)
    #define work_full_addr (mbar_base + 176)
    #define work_empty_addr (mbar_base + 200)
    #define throttle_full_addr (mbar_base + 224)
    #define throttle_empty_addr (mbar_base + 248)
    #define fast_ready_addr (mbar_base + 272)

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
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + 66560);
    const int smem_b_addr = smem + 66560;
    uint8_t* epi_staging = reinterpret_cast<uint8_t*>(smem_raw + 132096);
    const int epi_staging_addr = smem + 132096;
    uint8_t* smem_sfa = reinterpret_cast<uint8_t*>(smem_raw + 145408);
    const int smem_sfa_addr = smem + 145408;
    unsigned int* smem_sfb = reinterpret_cast<unsigned int*>(smem_raw + 153600);
    const int smem_sfb_addr = smem + 153600;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 169984);
    const int work_response_addr = smem + 169984;
    unsigned int* fast_response = reinterpret_cast<unsigned int*>(smem_raw + 170032);
    const int fast_response_addr = smem + 170032;
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if ((int)blockIdx.y >= num_non_exiting_ctas[0]) return;

    // Mbarrier init (12 pipeline groups, 0 ordered-sequence groups, 35 barriers)
    // Mbarriers at smem_raw[0..280)

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
            // sfb_full: 4 barriers, init_count=512
            mbarrier_init(smem + 96, 512);
            mbarrier_init(smem + 104, 512);
            mbarrier_init(smem + 112, 512);
            mbarrier_init(smem + 120, 512);
            // k_done: 4 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            // --- pipeline 'mma_pipe' ---
            // mma_full: 1 barriers, init_count=1
            mbarrier_init(smem + 160, 1);
            // mma_free: 1 barriers, init_count=256
            mbarrier_init(smem + 168, 256);
            // --- pipeline 'work_pipe' ---
            // work_full: 3 barriers, init_count=1
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            mbarrier_init(smem + 192, 1);
            // work_empty: 3 barriers, init_count=1216
            mbarrier_init(smem + 200, 1216);
            mbarrier_init(smem + 208, 1216);
            mbarrier_init(smem + 216, 1216);
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

    // TMEM alloc (512 columns, 496 used)
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
    const int tmem_sfa = taddr + 448;
    const int tmem_sfb = taddr + 464;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 16 && warp <= 19) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 88;");
    }

    // ---- Role: epilogue ----
    if (warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 144;");
        { // epilogue_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            const int warp_0 = warp;
            const int warp_group = warp_0 / 4;
            const int warp_local = warp_0 % 4;
            const int lane_1 = lane;
            unsigned int work_stage = 0;
            int epilogue_local_idx = 0;
            unsigned int m_tile = blockIdx.x;
            unsigned int n_tile = blockIdx.y;
            float quant_pair[8] = {0};
            unsigned int _phase_mma_full_0 = 0;
            unsigned int _phase_work_full = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter = 0; _tile_iter < grid_m / 2 * grid_n; _tile_iter++) {
                if (m_tile >= (unsigned int)grid_m || n_tile >= (unsigned int)grid_n) {
                    break;
                }
                int expert = tile_expert[n_tile];
                int valid_rows = (unsigned int)tile_mn_limit[n_tile] - n_tile * 256;
                float sc = scale_c[expert];
                float sg = scale_gate[expert];
                float cl = clamp_limit[expert];
                float al = act_alpha[expert];
                float be = act_beta[expert];
                float neg_cl = -cl;
                float beta_sg = be * sg;
                float alpha_sg = al * sg;
                mbarrier_wait(mma_full_addr, _phase_mma_full_0);
                _phase_mma_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int token_block = warp_group;
                if (epilogue_local_idx == 0) {
                    token_block = (token_block + 3) % 4;
                }
                int accum_col = epilogue_local_idx * 192 + token_block * 64;
                int row_addr = warp_local * 32 << 16;
                float _tmem_load_0[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[31]))
                    : "r"(taddr + (unsigned int)row_addr + (unsigned int)accum_col));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float _tmem_load_1[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[31]))
                    : "r"(taddr + (unsigned int)row_addr + 1048576 + (unsigned int)accum_col));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int base_row = warp_local * 16 + lane_1 / 4 * 2;
                if (warp_group == 0) {
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((mma_free_addr) & 0xFEFFFFFF) : "memory");
                }
                asm volatile("cp.async.bulk.wait_group.read 0;");
                if (warp_group == 0) {
                    asm volatile("barrier.sync 7, 128;" ::: "memory");
                }
                if (warp_group == 1) {
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                }
                int local_token0 = lane_1 % 4 * 2;
                int local_token1 = local_token0 + 1;
                int token0 = token_block * 64 + local_token0;
                int token1 = token0 + 1;
                float _max_0 = max_noftz(_tmem_load_0[0], neg_cl);
                float _min_0 = fminf(_max_0, cl);
                float step_lin00 = _min_0;
                float _max_1 = max_noftz(_tmem_load_0[1], neg_cl);
                float _min_1 = fminf(_max_1, cl);
                float step_lin01 = _min_1;
                float _max_2 = max_noftz(_tmem_load_1[0], neg_cl);
                float _min_2 = fminf(_max_2, cl);
                float step_lin10 = _min_2;
                float _max_3 = max_noftz(_tmem_load_1[1], neg_cl);
                float _min_3 = fminf(_max_3, cl);
                float step_lin11 = _min_3;
                float step_x00 = _tmem_load_0[2];
                float step_x01 = _tmem_load_0[3];
                float step_x10 = _tmem_load_1[2];
                float step_x11 = _tmem_load_1[3];
                float _exp2_0 = approx_exp2((-(step_x00 * sg)) * 1.4426950408889634f);
                float _rcp_0 = approx_rcp(1.0f + _exp2_0);
                float step_sig00 = _rcp_0;
                float _exp2_1 = approx_exp2((-(step_x01 * sg)) * 1.4426950408889634f);
                float _rcp_1 = approx_rcp(1.0f + _exp2_1);
                float step_sig01 = _rcp_1;
                float _exp2_2 = approx_exp2((-(step_x10 * sg)) * 1.4426950408889634f);
                float _rcp_2 = approx_rcp(1.0f + _exp2_2);
                float step_sig10 = _rcp_2;
                float _exp2_3 = approx_exp2((-(step_x11 * sg)) * 1.4426950408889634f);
                float _rcp_3 = approx_rcp(1.0f + _exp2_3);
                float step_sig11 = _rcp_3;
                float _min_4 = fminf(step_x00 * step_sig00, cl);
                float step_g00 = _min_4;
                float _min_5 = fminf(step_x01 * step_sig01, cl);
                float step_g01 = _min_5;
                float _min_6 = fminf(step_x10 * step_sig10, cl);
                float step_g10 = _min_6;
                float _min_7 = fminf(step_x11 * step_sig11, cl);
                float step_g11 = _min_7;
                float value00 = step_lin00 * sc * sg * step_g00;
                float value01 = step_lin01 * sc * sg * step_g01;
                float value10 = step_lin10 * sc * sg * step_g10;
                float value11 = step_lin11 * sc * sg * step_g11;
                float _fabs_0 = fabsf(value00);
                float _fabs_1 = fabsf(value10);
                float _max_4 = max_noftz(_fabs_0, _fabs_1);
                float block_max0 = _max_4;
                float _fabs_2 = fabsf(value01);
                float _fabs_3 = fabsf(value11);
                float _max_5 = max_noftz(_fabs_2, _fabs_3);
                float block_max1 = _max_5;
                float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, block_max0, 4);
                float _max_6 = max_noftz(block_max0, _shfl_xor_0);
                block_max0 = _max_6;
                float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, block_max1, 4);
                float _max_7 = max_noftz(block_max1, _shfl_xor_1);
                block_max1 = _max_7;
                float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, block_max0, 8);
                float _max_8 = max_noftz(block_max0, _shfl_xor_2);
                block_max0 = _max_8;
                float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, block_max1, 8);
                float _max_9 = max_noftz(block_max1, _shfl_xor_3);
                block_max1 = _max_9;
                float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, block_max0, 16);
                float _max_10 = max_noftz(block_max0, _shfl_xor_4);
                block_max0 = _max_10;
                float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, block_max1, 16);
                float _max_11 = max_noftz(block_max1, _shfl_xor_5);
                block_max1 = _max_11;
                float _fp8_rt_0;
                uint16_t _e4m3x2_0;
                uint32_t _f16x2_0;
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_0) : "f"(0.0f), "f"(block_max0 * 0.16666666666666666f));
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_0) : "h"(_e4m3x2_0));
                uint16_t _fp8_h0_0 = (uint16_t)(_f16x2_0 & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_0) : "h"(_fp8_h0_0));
                float scale0 = _fp8_rt_0;
                float _fp8_rt_1;
                uint16_t _e4m3x2_1;
                uint32_t _f16x2_1;
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_1) : "f"(0.0f), "f"(block_max1 * 0.16666666666666666f));
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1) : "h"(_e4m3x2_1));
                uint16_t _fp8_h0_1 = (uint16_t)(_f16x2_1 & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_1) : "h"(_fp8_h0_1));
                float scale1 = _fp8_rt_1;
                float inv_scale0 = 0.0f;
                float inv_scale1 = 0.0f;
                if (scale0 != 0.0f) {
                    inv_scale0 = 1.0f / scale0;
                }
                if (scale1 != 0.0f) {
                    inv_scale1 = 1.0f / scale1;
                }
                quant_pair[0] = value00 * inv_scale0;
                quant_pair[1] = value10 * inv_scale0;
                uint32_t _slice_lo_mask_0;
                {
                    int _lim_2 = 2;
                    if (_lim_2 <= 0) { _slice_lo_mask_0 = 0u; }
                    else if (_lim_2 >= 8) { _slice_lo_mask_0 = ((1u << 8) - 1u); }
                    else {
                        asm volatile("{"
                            ".reg .u32 t;\n\t"
                            "shl.b32 t, 1, %1;\n\t"
                            "add.u32 %0, t, -1;\n\t"
                            "}" : "=r"(_slice_lo_mask_0) : "r"(_lim_2));
                    }
                }
                uint32_t _slice_hi_mask_0;
                {
                    int _lim_3 = 8;
                    if (_lim_3 <= 0) { _slice_hi_mask_0 = 0u; }
                    else if (_lim_3 >= 8) { _slice_hi_mask_0 = ((1u << 8) - 1u); }
                    else {
                        asm volatile("{"
                            ".reg .u32 t;\n\t"
                            "shl.b32 t, 1, %1;\n\t"
                            "add.u32 %0, t, -1;\n\t"
                            "}" : "=r"(_slice_hi_mask_0) : "r"(_lim_3));
                    }
                }
                if (!(_slice_lo_mask_0 | ~_slice_hi_mask_0 & (1u << 0))) quant_pair[0] = 0.0f;
                if (!(_slice_lo_mask_0 | ~_slice_hi_mask_0 & (1u << 1))) quant_pair[1] = 0.0f;
                if (!(_slice_lo_mask_0 | ~_slice_hi_mask_0 & (1u << 2))) quant_pair[2] = 0.0f;
                if (!(_slice_lo_mask_0 | ~_slice_hi_mask_0 & (1u << 3))) quant_pair[3] = 0.0f;
                if (!(_slice_lo_mask_0 | ~_slice_hi_mask_0 & (1u << 4))) quant_pair[4] = 0.0f;
                if (!(_slice_lo_mask_0 | ~_slice_hi_mask_0 & (1u << 5))) quant_pair[5] = 0.0f;
                if (!(_slice_lo_mask_0 | ~_slice_hi_mask_0 & (1u << 6))) quant_pair[6] = 0.0f;
                if (!(_slice_lo_mask_0 | ~_slice_hi_mask_0 & (1u << 7))) quant_pair[7] = 0.0f;
                uint32_t _fp4_0[1];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_0[0]) : "f"(quant_pair[0]), "f"(quant_pair[1]), "f"(quant_pair[2]), "f"(quant_pair[3]), "f"(quant_pair[4]), "f"(quant_pair[5]), "f"(quant_pair[6]), "f"(quant_pair[7]));
                quant_pair[0] = value01 * inv_scale1;
                quant_pair[1] = value11 * inv_scale1;
                uint32_t _fp4_1[1];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_1[0]) : "f"(quant_pair[0]), "f"(quant_pair[1]), "f"(quant_pair[2]), "f"(quant_pair[3]), "f"(quant_pair[4]), "f"(quant_pair[5]), "f"(quant_pair[6]), "f"(quant_pair[7]));
                if (lane_1 < 4) {
                    int sf_feature = m_tile * 4 + (unsigned int)warp_local;
                    int sf_tile_stride = 2 * (M_out / 64) * 512;
                    int sf_base = n_tile * (unsigned int)sf_tile_stride + (unsigned int)(token0 / 128 * (M_out / 64) * 512) + (unsigned int)(sf_feature / 4 * 512) + (unsigned int)(token0 % 32 * 16) + (unsigned int)(token0 % 128 / 32 * 4) + (unsigned int)(sf_feature % 4);
                    if (token0 < valid_rows) {
                        {
                            unsigned short _fp8_pair;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(scale0));
                            *(reinterpret_cast<unsigned char*>(SFC + sf_base) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                        }
                    }
                    if (token1 < valid_rows) {
                        {
                            unsigned short _fp8_pair;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(scale1));
                            *(reinterpret_cast<unsigned char*>(SFC + (sf_base + 16)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                        }
                    }
                }
                int smem_flat0 = local_token0 * 64 + base_row;
                int smem_flat1 = local_token1 * 64 + base_row;
                int smem_index0 = smem_flat0 / 2 ^ smem_flat0 / 256 % 2 * 16;
                int smem_index1 = smem_flat1 / 2 ^ smem_flat1 / 256 % 2 * 16;
                epi_staging[warp_group * 2048 + smem_index0] = _fp4_0[0];
                epi_staging[warp_group * 2048 + smem_index1] = _fp4_1[0];
                int local_token0_0 = lane_1 % 4 * 2 + 8;
                int local_token1_1 = local_token0_0 + 1;
                int token0_2 = token_block * 64 + local_token0_0;
                int token1_3 = token0_2 + 1;
                float _max_12 = max_noftz(_tmem_load_0[4], neg_cl);
                float _min_8 = fminf(_max_12, cl);
                float step_lin00_4 = _min_8;
                float _max_13 = max_noftz(_tmem_load_0[5], neg_cl);
                float _min_9 = fminf(_max_13, cl);
                float step_lin01_5 = _min_9;
                float _max_14 = max_noftz(_tmem_load_1[4], neg_cl);
                float _min_10 = fminf(_max_14, cl);
                float step_lin10_6 = _min_10;
                float _max_15 = max_noftz(_tmem_load_1[5], neg_cl);
                float _min_11 = fminf(_max_15, cl);
                float step_lin11_7 = _min_11;
                float step_x00_8 = _tmem_load_0[6];
                float step_x01_9 = _tmem_load_0[7];
                float step_x10_10 = _tmem_load_1[6];
                float step_x11_11 = _tmem_load_1[7];
                float _exp2_4 = approx_exp2((-(step_x00_8 * sg)) * 1.4426950408889634f);
                float _rcp_4 = approx_rcp(1.0f + _exp2_4);
                float step_sig00_12 = _rcp_4;
                float _exp2_5 = approx_exp2((-(step_x01_9 * sg)) * 1.4426950408889634f);
                float _rcp_5 = approx_rcp(1.0f + _exp2_5);
                float step_sig01_13 = _rcp_5;
                float _exp2_6 = approx_exp2((-(step_x10_10 * sg)) * 1.4426950408889634f);
                float _rcp_6 = approx_rcp(1.0f + _exp2_6);
                float step_sig10_14 = _rcp_6;
                float _exp2_7 = approx_exp2((-(step_x11_11 * sg)) * 1.4426950408889634f);
                float _rcp_7 = approx_rcp(1.0f + _exp2_7);
                float step_sig11_15 = _rcp_7;
                float _min_12 = fminf(step_x00_8 * step_sig00_12, cl);
                float step_g00_16 = _min_12;
                float _min_13 = fminf(step_x01_9 * step_sig01_13, cl);
                float step_g01_17 = _min_13;
                float _min_14 = fminf(step_x10_10 * step_sig10_14, cl);
                float step_g10_18 = _min_14;
                float _min_15 = fminf(step_x11_11 * step_sig11_15, cl);
                float step_g11_19 = _min_15;
                float value00_20 = step_lin00_4 * sc * sg * step_g00_16;
                float value01_21 = step_lin01_5 * sc * sg * step_g01_17;
                float value10_22 = step_lin10_6 * sc * sg * step_g10_18;
                float value11_23 = step_lin11_7 * sc * sg * step_g11_19;
                float _fabs_4 = fabsf(value00_20);
                float _fabs_5 = fabsf(value10_22);
                float _max_16 = max_noftz(_fabs_4, _fabs_5);
                float block_max0_24 = _max_16;
                float _fabs_6 = fabsf(value01_21);
                float _fabs_7 = fabsf(value11_23);
                float _max_17 = max_noftz(_fabs_6, _fabs_7);
                float block_max1_25 = _max_17;
                float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, block_max0_24, 4);
                float _max_18 = max_noftz(block_max0_24, _shfl_xor_6);
                block_max0_24 = _max_18;
                float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, block_max1_25, 4);
                float _max_19 = max_noftz(block_max1_25, _shfl_xor_7);
                block_max1_25 = _max_19;
                float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, block_max0_24, 8);
                float _max_20 = max_noftz(block_max0_24, _shfl_xor_8);
                block_max0_24 = _max_20;
                float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, block_max1_25, 8);
                float _max_21 = max_noftz(block_max1_25, _shfl_xor_9);
                block_max1_25 = _max_21;
                float _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, block_max0_24, 16);
                float _max_22 = max_noftz(block_max0_24, _shfl_xor_10);
                block_max0_24 = _max_22;
                float _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, block_max1_25, 16);
                float _max_23 = max_noftz(block_max1_25, _shfl_xor_11);
                block_max1_25 = _max_23;
                float _fp8_rt_2;
                uint16_t _e4m3x2_4;
                uint32_t _f16x2_4;
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_4) : "f"(0.0f), "f"(block_max0_24 * 0.16666666666666666f));
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_4) : "h"(_e4m3x2_4));
                uint16_t _fp8_h0_4 = (uint16_t)(_f16x2_4 & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_2) : "h"(_fp8_h0_4));
                float scale0_26 = _fp8_rt_2;
                float _fp8_rt_3;
                uint16_t _e4m3x2_5;
                uint32_t _f16x2_5;
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_5) : "f"(0.0f), "f"(block_max1_25 * 0.16666666666666666f));
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_5) : "h"(_e4m3x2_5));
                uint16_t _fp8_h0_5 = (uint16_t)(_f16x2_5 & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_3) : "h"(_fp8_h0_5));
                float scale1_27 = _fp8_rt_3;
                float inv_scale0_28 = 0.0f;
                float inv_scale1_29 = 0.0f;
                if (scale0_26 != 0.0f) {
                    inv_scale0_28 = 1.0f / scale0_26;
                }
                if (scale1_27 != 0.0f) {
                    inv_scale1_29 = 1.0f / scale1_27;
                }
                quant_pair[0] = value00_20 * inv_scale0_28;
                quant_pair[1] = value10_22 * inv_scale0_28;
                uint32_t _slice_lo_mask_1;
                {
                    int _lim_6 = 2;
                    if (_lim_6 <= 0) { _slice_lo_mask_1 = 0u; }
                    else if (_lim_6 >= 8) { _slice_lo_mask_1 = ((1u << 8) - 1u); }
                    else {
                        asm volatile("{"
                            ".reg .u32 t;\n\t"
                            "shl.b32 t, 1, %1;\n\t"
                            "add.u32 %0, t, -1;\n\t"
                            "}" : "=r"(_slice_lo_mask_1) : "r"(_lim_6));
                    }
                }
                uint32_t _slice_hi_mask_1;
                {
                    int _lim_7 = 8;
                    if (_lim_7 <= 0) { _slice_hi_mask_1 = 0u; }
                    else if (_lim_7 >= 8) { _slice_hi_mask_1 = ((1u << 8) - 1u); }
                    else {
                        asm volatile("{"
                            ".reg .u32 t;\n\t"
                            "shl.b32 t, 1, %1;\n\t"
                            "add.u32 %0, t, -1;\n\t"
                            "}" : "=r"(_slice_hi_mask_1) : "r"(_lim_7));
                    }
                }
                if (!(_slice_lo_mask_1 | ~_slice_hi_mask_1 & (1u << 0))) quant_pair[0] = 0.0f;
                if (!(_slice_lo_mask_1 | ~_slice_hi_mask_1 & (1u << 1))) quant_pair[1] = 0.0f;
                if (!(_slice_lo_mask_1 | ~_slice_hi_mask_1 & (1u << 2))) quant_pair[2] = 0.0f;
                if (!(_slice_lo_mask_1 | ~_slice_hi_mask_1 & (1u << 3))) quant_pair[3] = 0.0f;
                if (!(_slice_lo_mask_1 | ~_slice_hi_mask_1 & (1u << 4))) quant_pair[4] = 0.0f;
                if (!(_slice_lo_mask_1 | ~_slice_hi_mask_1 & (1u << 5))) quant_pair[5] = 0.0f;
                if (!(_slice_lo_mask_1 | ~_slice_hi_mask_1 & (1u << 6))) quant_pair[6] = 0.0f;
                if (!(_slice_lo_mask_1 | ~_slice_hi_mask_1 & (1u << 7))) quant_pair[7] = 0.0f;
                uint32_t _fp4_2[1];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_2[0]) : "f"(quant_pair[0]), "f"(quant_pair[1]), "f"(quant_pair[2]), "f"(quant_pair[3]), "f"(quant_pair[4]), "f"(quant_pair[5]), "f"(quant_pair[6]), "f"(quant_pair[7]));
                quant_pair[0] = value01_21 * inv_scale1_29;
                quant_pair[1] = value11_23 * inv_scale1_29;
                uint32_t _fp4_3[1];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_3[0]) : "f"(quant_pair[0]), "f"(quant_pair[1]), "f"(quant_pair[2]), "f"(quant_pair[3]), "f"(quant_pair[4]), "f"(quant_pair[5]), "f"(quant_pair[6]), "f"(quant_pair[7]));
                if (lane_1 < 4) {
                    int sf_feature_1 = m_tile * 4 + (unsigned int)warp_local;
                    int sf_tile_stride_1 = 2 * (M_out / 64) * 512;
                    int sf_base_1 = n_tile * (unsigned int)sf_tile_stride_1 + (unsigned int)(token0_2 / 128 * (M_out / 64) * 512) + (unsigned int)(sf_feature_1 / 4 * 512) + (unsigned int)(token0_2 % 32 * 16) + (unsigned int)(token0_2 % 128 / 32 * 4) + (unsigned int)(sf_feature_1 % 4);
                    if (token0_2 < valid_rows) {
                        {
                            unsigned short _fp8_pair;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(scale0_26));
                            *(reinterpret_cast<unsigned char*>(SFC + sf_base_1) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                        }
                    }
                    if (token1_3 < valid_rows) {
                        {
                            unsigned short _fp8_pair;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(scale1_27));
                            *(reinterpret_cast<unsigned char*>(SFC + (sf_base_1 + 16)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                        }
                    }
                }
                int smem_flat0_30 = local_token0_0 * 64 + base_row;
                int smem_flat1_31 = local_token1_1 * 64 + base_row;
                int smem_index0_32 = smem_flat0_30 / 2 ^ smem_flat0_30 / 256 % 2 * 16;
                int smem_index1_33 = smem_flat1_31 / 2 ^ smem_flat1_31 / 256 % 2 * 16;
                epi_staging[warp_group * 2048 + smem_index0_32] = _fp4_2[0];
                epi_staging[warp_group * 2048 + smem_index1_33] = _fp4_3[0];
                int local_token0_34 = lane_1 % 4 * 2 + 16;
                int local_token1_35 = local_token0_34 + 1;
                int token0_36 = token_block * 64 + local_token0_34;
                int token1_37 = token0_36 + 1;
                float _max_24 = max_noftz(_tmem_load_0[8], neg_cl);
                float _min_16 = fminf(_max_24, cl);
                float step_lin00_38 = _min_16;
                float _max_25 = max_noftz(_tmem_load_0[9], neg_cl);
                float _min_17 = fminf(_max_25, cl);
                float step_lin01_39 = _min_17;
                float _max_26 = max_noftz(_tmem_load_1[8], neg_cl);
                float _min_18 = fminf(_max_26, cl);
                float step_lin10_40 = _min_18;
                float _max_27 = max_noftz(_tmem_load_1[9], neg_cl);
                float _min_19 = fminf(_max_27, cl);
                float step_lin11_41 = _min_19;
                float step_x00_42 = _tmem_load_0[10];
                float step_x01_43 = _tmem_load_0[11];
                float step_x10_44 = _tmem_load_1[10];
                float step_x11_45 = _tmem_load_1[11];
                float _exp2_8 = approx_exp2((-(step_x00_42 * sg)) * 1.4426950408889634f);
                float _rcp_8 = approx_rcp(1.0f + _exp2_8);
                float step_sig00_46 = _rcp_8;
                float _exp2_9 = approx_exp2((-(step_x01_43 * sg)) * 1.4426950408889634f);
                float _rcp_9 = approx_rcp(1.0f + _exp2_9);
                float step_sig01_47 = _rcp_9;
                float _exp2_10 = approx_exp2((-(step_x10_44 * sg)) * 1.4426950408889634f);
                float _rcp_10 = approx_rcp(1.0f + _exp2_10);
                float step_sig10_48 = _rcp_10;
                float _exp2_11 = approx_exp2((-(step_x11_45 * sg)) * 1.4426950408889634f);
                float _rcp_11 = approx_rcp(1.0f + _exp2_11);
                float step_sig11_49 = _rcp_11;
                float _min_20 = fminf(step_x00_42 * step_sig00_46, cl);
                float step_g00_50 = _min_20;
                float _min_21 = fminf(step_x01_43 * step_sig01_47, cl);
                float step_g01_51 = _min_21;
                float _min_22 = fminf(step_x10_44 * step_sig10_48, cl);
                float step_g10_52 = _min_22;
                float _min_23 = fminf(step_x11_45 * step_sig11_49, cl);
                float step_g11_53 = _min_23;
                float value00_54 = step_lin00_38 * sc * sg * step_g00_50;
                float value01_55 = step_lin01_39 * sc * sg * step_g01_51;
                float value10_56 = step_lin10_40 * sc * sg * step_g10_52;
                float value11_57 = step_lin11_41 * sc * sg * step_g11_53;
                float _fabs_8 = fabsf(value00_54);
                float _fabs_9 = fabsf(value10_56);
                float _max_28 = max_noftz(_fabs_8, _fabs_9);
                float block_max0_58 = _max_28;
                float _fabs_10 = fabsf(value01_55);
                float _fabs_11 = fabsf(value11_57);
                float _max_29 = max_noftz(_fabs_10, _fabs_11);
                float block_max1_59 = _max_29;
                float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, block_max0_58, 4);
                float _max_30 = max_noftz(block_max0_58, _shfl_xor_12);
                block_max0_58 = _max_30;
                float _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, block_max1_59, 4);
                float _max_31 = max_noftz(block_max1_59, _shfl_xor_13);
                block_max1_59 = _max_31;
                float _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, block_max0_58, 8);
                float _max_32 = max_noftz(block_max0_58, _shfl_xor_14);
                block_max0_58 = _max_32;
                float _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, block_max1_59, 8);
                float _max_33 = max_noftz(block_max1_59, _shfl_xor_15);
                block_max1_59 = _max_33;
                float _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, block_max0_58, 16);
                float _max_34 = max_noftz(block_max0_58, _shfl_xor_16);
                block_max0_58 = _max_34;
                float _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, block_max1_59, 16);
                float _max_35 = max_noftz(block_max1_59, _shfl_xor_17);
                block_max1_59 = _max_35;
                float _fp8_rt_4;
                uint16_t _e4m3x2_8;
                uint32_t _f16x2_8;
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_8) : "f"(0.0f), "f"(block_max0_58 * 0.16666666666666666f));
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_8) : "h"(_e4m3x2_8));
                uint16_t _fp8_h0_8 = (uint16_t)(_f16x2_8 & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_4) : "h"(_fp8_h0_8));
                float scale0_60 = _fp8_rt_4;
                float _fp8_rt_5;
                uint16_t _e4m3x2_9;
                uint32_t _f16x2_9;
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_9) : "f"(0.0f), "f"(block_max1_59 * 0.16666666666666666f));
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_9) : "h"(_e4m3x2_9));
                uint16_t _fp8_h0_9 = (uint16_t)(_f16x2_9 & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_5) : "h"(_fp8_h0_9));
                float scale1_61 = _fp8_rt_5;
                float inv_scale0_62 = 0.0f;
                float inv_scale1_63 = 0.0f;
                if (scale0_60 != 0.0f) {
                    inv_scale0_62 = 1.0f / scale0_60;
                }
                if (scale1_61 != 0.0f) {
                    inv_scale1_63 = 1.0f / scale1_61;
                }
                quant_pair[0] = value00_54 * inv_scale0_62;
                quant_pair[1] = value10_56 * inv_scale0_62;
                uint32_t _slice_lo_mask_2;
                {
                    int _lim_10 = 2;
                    if (_lim_10 <= 0) { _slice_lo_mask_2 = 0u; }
                    else if (_lim_10 >= 8) { _slice_lo_mask_2 = ((1u << 8) - 1u); }
                    else {
                        asm volatile("{"
                            ".reg .u32 t;\n\t"
                            "shl.b32 t, 1, %1;\n\t"
                            "add.u32 %0, t, -1;\n\t"
                            "}" : "=r"(_slice_lo_mask_2) : "r"(_lim_10));
                    }
                }
                uint32_t _slice_hi_mask_2;
                {
                    int _lim_11 = 8;
                    if (_lim_11 <= 0) { _slice_hi_mask_2 = 0u; }
                    else if (_lim_11 >= 8) { _slice_hi_mask_2 = ((1u << 8) - 1u); }
                    else {
                        asm volatile("{"
                            ".reg .u32 t;\n\t"
                            "shl.b32 t, 1, %1;\n\t"
                            "add.u32 %0, t, -1;\n\t"
                            "}" : "=r"(_slice_hi_mask_2) : "r"(_lim_11));
                    }
                }
                if (!(_slice_lo_mask_2 | ~_slice_hi_mask_2 & (1u << 0))) quant_pair[0] = 0.0f;
                if (!(_slice_lo_mask_2 | ~_slice_hi_mask_2 & (1u << 1))) quant_pair[1] = 0.0f;
                if (!(_slice_lo_mask_2 | ~_slice_hi_mask_2 & (1u << 2))) quant_pair[2] = 0.0f;
                if (!(_slice_lo_mask_2 | ~_slice_hi_mask_2 & (1u << 3))) quant_pair[3] = 0.0f;
                if (!(_slice_lo_mask_2 | ~_slice_hi_mask_2 & (1u << 4))) quant_pair[4] = 0.0f;
                if (!(_slice_lo_mask_2 | ~_slice_hi_mask_2 & (1u << 5))) quant_pair[5] = 0.0f;
                if (!(_slice_lo_mask_2 | ~_slice_hi_mask_2 & (1u << 6))) quant_pair[6] = 0.0f;
                if (!(_slice_lo_mask_2 | ~_slice_hi_mask_2 & (1u << 7))) quant_pair[7] = 0.0f;
                uint32_t _fp4_4[1];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_4[0]) : "f"(quant_pair[0]), "f"(quant_pair[1]), "f"(quant_pair[2]), "f"(quant_pair[3]), "f"(quant_pair[4]), "f"(quant_pair[5]), "f"(quant_pair[6]), "f"(quant_pair[7]));
                quant_pair[0] = value01_55 * inv_scale1_63;
                quant_pair[1] = value11_57 * inv_scale1_63;
                uint32_t _fp4_5[1];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_5[0]) : "f"(quant_pair[0]), "f"(quant_pair[1]), "f"(quant_pair[2]), "f"(quant_pair[3]), "f"(quant_pair[4]), "f"(quant_pair[5]), "f"(quant_pair[6]), "f"(quant_pair[7]));
                if (lane_1 < 4) {
                    int sf_feature_2 = m_tile * 4 + (unsigned int)warp_local;
                    int sf_tile_stride_2 = 2 * (M_out / 64) * 512;
                    int sf_base_2 = n_tile * (unsigned int)sf_tile_stride_2 + (unsigned int)(token0_36 / 128 * (M_out / 64) * 512) + (unsigned int)(sf_feature_2 / 4 * 512) + (unsigned int)(token0_36 % 32 * 16) + (unsigned int)(token0_36 % 128 / 32 * 4) + (unsigned int)(sf_feature_2 % 4);
                    if (token0_36 < valid_rows) {
                        {
                            unsigned short _fp8_pair;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(scale0_60));
                            *(reinterpret_cast<unsigned char*>(SFC + sf_base_2) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                        }
                    }
                    if (token1_37 < valid_rows) {
                        {
                            unsigned short _fp8_pair;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(scale1_61));
                            *(reinterpret_cast<unsigned char*>(SFC + (sf_base_2 + 16)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                        }
                    }
                }
                int smem_flat0_64 = local_token0_34 * 64 + base_row;
                int smem_flat1_65 = local_token1_35 * 64 + base_row;
                int smem_index0_66 = smem_flat0_64 / 2 ^ smem_flat0_64 / 256 % 2 * 16;
                int smem_index1_67 = smem_flat1_65 / 2 ^ smem_flat1_65 / 256 % 2 * 16;
                epi_staging[warp_group * 2048 + smem_index0_66] = _fp4_4[0];
                epi_staging[warp_group * 2048 + smem_index1_67] = _fp4_5[0];
                int local_token0_68 = lane_1 % 4 * 2 + 24;
                int local_token1_69 = local_token0_68 + 1;
                int token0_70 = token_block * 64 + local_token0_68;
                int token1_71 = token0_70 + 1;
                float _max_36 = max_noftz(_tmem_load_0[12], neg_cl);
                float _min_24 = fminf(_max_36, cl);
                float step_lin00_72 = _min_24;
                float _max_37 = max_noftz(_tmem_load_0[13], neg_cl);
                float _min_25 = fminf(_max_37, cl);
                float step_lin01_73 = _min_25;
                float _max_38 = max_noftz(_tmem_load_1[12], neg_cl);
                float _min_26 = fminf(_max_38, cl);
                float step_lin10_74 = _min_26;
                float _max_39 = max_noftz(_tmem_load_1[13], neg_cl);
                float _min_27 = fminf(_max_39, cl);
                float step_lin11_75 = _min_27;
                float step_x00_76 = _tmem_load_0[14];
                float step_x01_77 = _tmem_load_0[15];
                float step_x10_78 = _tmem_load_1[14];
                float step_x11_79 = _tmem_load_1[15];
                float _exp2_12 = approx_exp2((-(step_x00_76 * sg)) * 1.4426950408889634f);
                float _rcp_12 = approx_rcp(1.0f + _exp2_12);
                float step_sig00_80 = _rcp_12;
                float _exp2_13 = approx_exp2((-(step_x01_77 * sg)) * 1.4426950408889634f);
                float _rcp_13 = approx_rcp(1.0f + _exp2_13);
                float step_sig01_81 = _rcp_13;
                float _exp2_14 = approx_exp2((-(step_x10_78 * sg)) * 1.4426950408889634f);
                float _rcp_14 = approx_rcp(1.0f + _exp2_14);
                float step_sig10_82 = _rcp_14;
                float _exp2_15 = approx_exp2((-(step_x11_79 * sg)) * 1.4426950408889634f);
                float _rcp_15 = approx_rcp(1.0f + _exp2_15);
                float step_sig11_83 = _rcp_15;
                float _min_28 = fminf(step_x00_76 * step_sig00_80, cl);
                float step_g00_84 = _min_28;
                float _min_29 = fminf(step_x01_77 * step_sig01_81, cl);
                float step_g01_85 = _min_29;
                float _min_30 = fminf(step_x10_78 * step_sig10_82, cl);
                float step_g10_86 = _min_30;
                float _min_31 = fminf(step_x11_79 * step_sig11_83, cl);
                float step_g11_87 = _min_31;
                float value00_88 = step_lin00_72 * sc * sg * step_g00_84;
                float value01_89 = step_lin01_73 * sc * sg * step_g01_85;
                float value10_90 = step_lin10_74 * sc * sg * step_g10_86;
                float value11_91 = step_lin11_75 * sc * sg * step_g11_87;
                float _fabs_12 = fabsf(value00_88);
                float _fabs_13 = fabsf(value10_90);
                float _max_40 = max_noftz(_fabs_12, _fabs_13);
                float block_max0_92 = _max_40;
                float _fabs_14 = fabsf(value01_89);
                float _fabs_15 = fabsf(value11_91);
                float _max_41 = max_noftz(_fabs_14, _fabs_15);
                float block_max1_93 = _max_41;
                float _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, block_max0_92, 4);
                float _max_42 = max_noftz(block_max0_92, _shfl_xor_18);
                block_max0_92 = _max_42;
                float _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, block_max1_93, 4);
                float _max_43 = max_noftz(block_max1_93, _shfl_xor_19);
                block_max1_93 = _max_43;
                float _shfl_xor_20 = __shfl_xor_sync(0xFFFFFFFF, block_max0_92, 8);
                float _max_44 = max_noftz(block_max0_92, _shfl_xor_20);
                block_max0_92 = _max_44;
                float _shfl_xor_21 = __shfl_xor_sync(0xFFFFFFFF, block_max1_93, 8);
                float _max_45 = max_noftz(block_max1_93, _shfl_xor_21);
                block_max1_93 = _max_45;
                float _shfl_xor_22 = __shfl_xor_sync(0xFFFFFFFF, block_max0_92, 16);
                float _max_46 = max_noftz(block_max0_92, _shfl_xor_22);
                block_max0_92 = _max_46;
                float _shfl_xor_23 = __shfl_xor_sync(0xFFFFFFFF, block_max1_93, 16);
                float _max_47 = max_noftz(block_max1_93, _shfl_xor_23);
                block_max1_93 = _max_47;
                float _fp8_rt_6;
                uint16_t _e4m3x2_12;
                uint32_t _f16x2_12;
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_12) : "f"(0.0f), "f"(block_max0_92 * 0.16666666666666666f));
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_12) : "h"(_e4m3x2_12));
                uint16_t _fp8_h0_12 = (uint16_t)(_f16x2_12 & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_6) : "h"(_fp8_h0_12));
                float scale0_94 = _fp8_rt_6;
                float _fp8_rt_7;
                uint16_t _e4m3x2_13;
                uint32_t _f16x2_13;
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_13) : "f"(0.0f), "f"(block_max1_93 * 0.16666666666666666f));
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_13) : "h"(_e4m3x2_13));
                uint16_t _fp8_h0_13 = (uint16_t)(_f16x2_13 & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_7) : "h"(_fp8_h0_13));
                float scale1_95 = _fp8_rt_7;
                float inv_scale0_96 = 0.0f;
                float inv_scale1_97 = 0.0f;
                if (scale0_94 != 0.0f) {
                    inv_scale0_96 = 1.0f / scale0_94;
                }
                if (scale1_95 != 0.0f) {
                    inv_scale1_97 = 1.0f / scale1_95;
                }
                quant_pair[0] = value00_88 * inv_scale0_96;
                quant_pair[1] = value10_90 * inv_scale0_96;
                uint32_t _slice_lo_mask_3;
                {
                    int _lim_14 = 2;
                    if (_lim_14 <= 0) { _slice_lo_mask_3 = 0u; }
                    else if (_lim_14 >= 8) { _slice_lo_mask_3 = ((1u << 8) - 1u); }
                    else {
                        asm volatile("{"
                            ".reg .u32 t;\n\t"
                            "shl.b32 t, 1, %1;\n\t"
                            "add.u32 %0, t, -1;\n\t"
                            "}" : "=r"(_slice_lo_mask_3) : "r"(_lim_14));
                    }
                }
                uint32_t _slice_hi_mask_3;
                {
                    int _lim_15 = 8;
                    if (_lim_15 <= 0) { _slice_hi_mask_3 = 0u; }
                    else if (_lim_15 >= 8) { _slice_hi_mask_3 = ((1u << 8) - 1u); }
                    else {
                        asm volatile("{"
                            ".reg .u32 t;\n\t"
                            "shl.b32 t, 1, %1;\n\t"
                            "add.u32 %0, t, -1;\n\t"
                            "}" : "=r"(_slice_hi_mask_3) : "r"(_lim_15));
                    }
                }
                if (!(_slice_lo_mask_3 | ~_slice_hi_mask_3 & (1u << 0))) quant_pair[0] = 0.0f;
                if (!(_slice_lo_mask_3 | ~_slice_hi_mask_3 & (1u << 1))) quant_pair[1] = 0.0f;
                if (!(_slice_lo_mask_3 | ~_slice_hi_mask_3 & (1u << 2))) quant_pair[2] = 0.0f;
                if (!(_slice_lo_mask_3 | ~_slice_hi_mask_3 & (1u << 3))) quant_pair[3] = 0.0f;
                if (!(_slice_lo_mask_3 | ~_slice_hi_mask_3 & (1u << 4))) quant_pair[4] = 0.0f;
                if (!(_slice_lo_mask_3 | ~_slice_hi_mask_3 & (1u << 5))) quant_pair[5] = 0.0f;
                if (!(_slice_lo_mask_3 | ~_slice_hi_mask_3 & (1u << 6))) quant_pair[6] = 0.0f;
                if (!(_slice_lo_mask_3 | ~_slice_hi_mask_3 & (1u << 7))) quant_pair[7] = 0.0f;
                uint32_t _fp4_6[1];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_6[0]) : "f"(quant_pair[0]), "f"(quant_pair[1]), "f"(quant_pair[2]), "f"(quant_pair[3]), "f"(quant_pair[4]), "f"(quant_pair[5]), "f"(quant_pair[6]), "f"(quant_pair[7]));
                quant_pair[0] = value01_89 * inv_scale1_97;
                quant_pair[1] = value11_91 * inv_scale1_97;
                uint32_t _fp4_7[1];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_7[0]) : "f"(quant_pair[0]), "f"(quant_pair[1]), "f"(quant_pair[2]), "f"(quant_pair[3]), "f"(quant_pair[4]), "f"(quant_pair[5]), "f"(quant_pair[6]), "f"(quant_pair[7]));
                if (lane_1 < 4) {
                    int sf_feature_3 = m_tile * 4 + (unsigned int)warp_local;
                    int sf_tile_stride_3 = 2 * (M_out / 64) * 512;
                    int sf_base_3 = n_tile * (unsigned int)sf_tile_stride_3 + (unsigned int)(token0_70 / 128 * (M_out / 64) * 512) + (unsigned int)(sf_feature_3 / 4 * 512) + (unsigned int)(token0_70 % 32 * 16) + (unsigned int)(token0_70 % 128 / 32 * 4) + (unsigned int)(sf_feature_3 % 4);
                    if (token0_70 < valid_rows) {
                        {
                            unsigned short _fp8_pair;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(scale0_94));
                            *(reinterpret_cast<unsigned char*>(SFC + sf_base_3) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                        }
                    }
                    if (token1_71 < valid_rows) {
                        {
                            unsigned short _fp8_pair;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(scale1_95));
                            *(reinterpret_cast<unsigned char*>(SFC + (sf_base_3 + 16)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                        }
                    }
                }
                int smem_flat0_98 = local_token0_68 * 64 + base_row;
                int smem_flat1_99 = local_token1_69 * 64 + base_row;
                int smem_index0_100 = smem_flat0_98 / 2 ^ smem_flat0_98 / 256 % 2 * 16;
                int smem_index1_101 = smem_flat1_99 / 2 ^ smem_flat1_99 / 256 % 2 * 16;
                epi_staging[warp_group * 2048 + smem_index0_100] = _fp4_6[0];
                epi_staging[warp_group * 2048 + smem_index1_101] = _fp4_7[0];
                int local_token0_102 = lane_1 % 4 * 2 + 32;
                int local_token1_103 = local_token0_102 + 1;
                int token0_104 = token_block * 64 + local_token0_102;
                int token1_105 = token0_104 + 1;
                float _max_48 = max_noftz(_tmem_load_0[16], neg_cl);
                float _min_32 = fminf(_max_48, cl);
                float step_lin00_106 = _min_32;
                float _max_49 = max_noftz(_tmem_load_0[17], neg_cl);
                float _min_33 = fminf(_max_49, cl);
                float step_lin01_107 = _min_33;
                float _max_50 = max_noftz(_tmem_load_1[16], neg_cl);
                float _min_34 = fminf(_max_50, cl);
                float step_lin10_108 = _min_34;
                float _max_51 = max_noftz(_tmem_load_1[17], neg_cl);
                float _min_35 = fminf(_max_51, cl);
                float step_lin11_109 = _min_35;
                float step_x00_110 = _tmem_load_0[18];
                float step_x01_111 = _tmem_load_0[19];
                float step_x10_112 = _tmem_load_1[18];
                float step_x11_113 = _tmem_load_1[19];
                float _exp2_16 = approx_exp2((-(step_x00_110 * sg)) * 1.4426950408889634f);
                float _rcp_16 = approx_rcp(1.0f + _exp2_16);
                float step_sig00_114 = _rcp_16;
                float _exp2_17 = approx_exp2((-(step_x01_111 * sg)) * 1.4426950408889634f);
                float _rcp_17 = approx_rcp(1.0f + _exp2_17);
                float step_sig01_115 = _rcp_17;
                float _exp2_18 = approx_exp2((-(step_x10_112 * sg)) * 1.4426950408889634f);
                float _rcp_18 = approx_rcp(1.0f + _exp2_18);
                float step_sig10_116 = _rcp_18;
                float _exp2_19 = approx_exp2((-(step_x11_113 * sg)) * 1.4426950408889634f);
                float _rcp_19 = approx_rcp(1.0f + _exp2_19);
                float step_sig11_117 = _rcp_19;
                float _min_36 = fminf(step_x00_110 * step_sig00_114, cl);
                float step_g00_118 = _min_36;
                float _min_37 = fminf(step_x01_111 * step_sig01_115, cl);
                float step_g01_119 = _min_37;
                float _min_38 = fminf(step_x10_112 * step_sig10_116, cl);
                float step_g10_120 = _min_38;
                float _min_39 = fminf(step_x11_113 * step_sig11_117, cl);
                float step_g11_121 = _min_39;
                float value00_122 = step_lin00_106 * sc * sg * step_g00_118;
                float value01_123 = step_lin01_107 * sc * sg * step_g01_119;
                float value10_124 = step_lin10_108 * sc * sg * step_g10_120;
                float value11_125 = step_lin11_109 * sc * sg * step_g11_121;
                float _fabs_16 = fabsf(value00_122);
                float _fabs_17 = fabsf(value10_124);
                float _max_52 = max_noftz(_fabs_16, _fabs_17);
                float block_max0_126 = _max_52;
                float _fabs_18 = fabsf(value01_123);
                float _fabs_19 = fabsf(value11_125);
                float _max_53 = max_noftz(_fabs_18, _fabs_19);
                float block_max1_127 = _max_53;
                float _shfl_xor_24 = __shfl_xor_sync(0xFFFFFFFF, block_max0_126, 4);
                float _max_54 = max_noftz(block_max0_126, _shfl_xor_24);
                block_max0_126 = _max_54;
                float _shfl_xor_25 = __shfl_xor_sync(0xFFFFFFFF, block_max1_127, 4);
                float _max_55 = max_noftz(block_max1_127, _shfl_xor_25);
                block_max1_127 = _max_55;
                float _shfl_xor_26 = __shfl_xor_sync(0xFFFFFFFF, block_max0_126, 8);
                float _max_56 = max_noftz(block_max0_126, _shfl_xor_26);
                block_max0_126 = _max_56;
                float _shfl_xor_27 = __shfl_xor_sync(0xFFFFFFFF, block_max1_127, 8);
                float _max_57 = max_noftz(block_max1_127, _shfl_xor_27);
                block_max1_127 = _max_57;
                float _shfl_xor_28 = __shfl_xor_sync(0xFFFFFFFF, block_max0_126, 16);
                float _max_58 = max_noftz(block_max0_126, _shfl_xor_28);
                block_max0_126 = _max_58;
                float _shfl_xor_29 = __shfl_xor_sync(0xFFFFFFFF, block_max1_127, 16);
                float _max_59 = max_noftz(block_max1_127, _shfl_xor_29);
                block_max1_127 = _max_59;
                float _fp8_rt_8;
                uint16_t _e4m3x2_16;
                uint32_t _f16x2_16;
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_16) : "f"(0.0f), "f"(block_max0_126 * 0.16666666666666666f));
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_16) : "h"(_e4m3x2_16));
                uint16_t _fp8_h0_16 = (uint16_t)(_f16x2_16 & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_8) : "h"(_fp8_h0_16));
                float scale0_128 = _fp8_rt_8;
                float _fp8_rt_9;
                uint16_t _e4m3x2_17;
                uint32_t _f16x2_17;
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_17) : "f"(0.0f), "f"(block_max1_127 * 0.16666666666666666f));
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_17) : "h"(_e4m3x2_17));
                uint16_t _fp8_h0_17 = (uint16_t)(_f16x2_17 & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_9) : "h"(_fp8_h0_17));
                float scale1_129 = _fp8_rt_9;
                float inv_scale0_130 = 0.0f;
                float inv_scale1_131 = 0.0f;
                if (scale0_128 != 0.0f) {
                    inv_scale0_130 = 1.0f / scale0_128;
                }
                if (scale1_129 != 0.0f) {
                    inv_scale1_131 = 1.0f / scale1_129;
                }
                quant_pair[0] = value00_122 * inv_scale0_130;
                quant_pair[1] = value10_124 * inv_scale0_130;
                uint32_t _slice_lo_mask_4;
                {
                    int _lim_18 = 2;
                    if (_lim_18 <= 0) { _slice_lo_mask_4 = 0u; }
                    else if (_lim_18 >= 8) { _slice_lo_mask_4 = ((1u << 8) - 1u); }
                    else {
                        asm volatile("{"
                            ".reg .u32 t;\n\t"
                            "shl.b32 t, 1, %1;\n\t"
                            "add.u32 %0, t, -1;\n\t"
                            "}" : "=r"(_slice_lo_mask_4) : "r"(_lim_18));
                    }
                }
                uint32_t _slice_hi_mask_4;
                {
                    int _lim_19 = 8;
                    if (_lim_19 <= 0) { _slice_hi_mask_4 = 0u; }
                    else if (_lim_19 >= 8) { _slice_hi_mask_4 = ((1u << 8) - 1u); }
                    else {
                        asm volatile("{"
                            ".reg .u32 t;\n\t"
                            "shl.b32 t, 1, %1;\n\t"
                            "add.u32 %0, t, -1;\n\t"
                            "}" : "=r"(_slice_hi_mask_4) : "r"(_lim_19));
                    }
                }
                if (!(_slice_lo_mask_4 | ~_slice_hi_mask_4 & (1u << 0))) quant_pair[0] = 0.0f;
                if (!(_slice_lo_mask_4 | ~_slice_hi_mask_4 & (1u << 1))) quant_pair[1] = 0.0f;
                if (!(_slice_lo_mask_4 | ~_slice_hi_mask_4 & (1u << 2))) quant_pair[2] = 0.0f;
                if (!(_slice_lo_mask_4 | ~_slice_hi_mask_4 & (1u << 3))) quant_pair[3] = 0.0f;
                if (!(_slice_lo_mask_4 | ~_slice_hi_mask_4 & (1u << 4))) quant_pair[4] = 0.0f;
                if (!(_slice_lo_mask_4 | ~_slice_hi_mask_4 & (1u << 5))) quant_pair[5] = 0.0f;
                if (!(_slice_lo_mask_4 | ~_slice_hi_mask_4 & (1u << 6))) quant_pair[6] = 0.0f;
                if (!(_slice_lo_mask_4 | ~_slice_hi_mask_4 & (1u << 7))) quant_pair[7] = 0.0f;
                uint32_t _fp4_8[1];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_8[0]) : "f"(quant_pair[0]), "f"(quant_pair[1]), "f"(quant_pair[2]), "f"(quant_pair[3]), "f"(quant_pair[4]), "f"(quant_pair[5]), "f"(quant_pair[6]), "f"(quant_pair[7]));
                quant_pair[0] = value01_123 * inv_scale1_131;
                quant_pair[1] = value11_125 * inv_scale1_131;
                uint32_t _fp4_9[1];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_9[0]) : "f"(quant_pair[0]), "f"(quant_pair[1]), "f"(quant_pair[2]), "f"(quant_pair[3]), "f"(quant_pair[4]), "f"(quant_pair[5]), "f"(quant_pair[6]), "f"(quant_pair[7]));
                if (lane_1 < 4) {
                    int sf_feature_4 = m_tile * 4 + (unsigned int)warp_local;
                    int sf_tile_stride_4 = 2 * (M_out / 64) * 512;
                    int sf_base_4 = n_tile * (unsigned int)sf_tile_stride_4 + (unsigned int)(token0_104 / 128 * (M_out / 64) * 512) + (unsigned int)(sf_feature_4 / 4 * 512) + (unsigned int)(token0_104 % 32 * 16) + (unsigned int)(token0_104 % 128 / 32 * 4) + (unsigned int)(sf_feature_4 % 4);
                    if (token0_104 < valid_rows) {
                        {
                            unsigned short _fp8_pair;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(scale0_128));
                            *(reinterpret_cast<unsigned char*>(SFC + sf_base_4) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                        }
                    }
                    if (token1_105 < valid_rows) {
                        {
                            unsigned short _fp8_pair;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(scale1_129));
                            *(reinterpret_cast<unsigned char*>(SFC + (sf_base_4 + 16)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                        }
                    }
                }
                int smem_flat0_132 = local_token0_102 * 64 + base_row;
                int smem_flat1_133 = local_token1_103 * 64 + base_row;
                int smem_index0_134 = smem_flat0_132 / 2 ^ smem_flat0_132 / 256 % 2 * 16;
                int smem_index1_135 = smem_flat1_133 / 2 ^ smem_flat1_133 / 256 % 2 * 16;
                epi_staging[warp_group * 2048 + smem_index0_134] = _fp4_8[0];
                epi_staging[warp_group * 2048 + smem_index1_135] = _fp4_9[0];
                int local_token0_136 = lane_1 % 4 * 2 + 40;
                int local_token1_137 = local_token0_136 + 1;
                int token0_138 = token_block * 64 + local_token0_136;
                int token1_139 = token0_138 + 1;
                float _max_60 = max_noftz(_tmem_load_0[20], neg_cl);
                float _min_40 = fminf(_max_60, cl);
                float step_lin00_140 = _min_40;
                float _max_61 = max_noftz(_tmem_load_0[21], neg_cl);
                float _min_41 = fminf(_max_61, cl);
                float step_lin01_141 = _min_41;
                float _max_62 = max_noftz(_tmem_load_1[20], neg_cl);
                float _min_42 = fminf(_max_62, cl);
                float step_lin10_142 = _min_42;
                float _max_63 = max_noftz(_tmem_load_1[21], neg_cl);
                float _min_43 = fminf(_max_63, cl);
                float step_lin11_143 = _min_43;
                float step_x00_144 = _tmem_load_0[22];
                float step_x01_145 = _tmem_load_0[23];
                float step_x10_146 = _tmem_load_1[22];
                float step_x11_147 = _tmem_load_1[23];
                float _exp2_20 = approx_exp2((-(step_x00_144 * sg)) * 1.4426950408889634f);
                float _rcp_20 = approx_rcp(1.0f + _exp2_20);
                float step_sig00_148 = _rcp_20;
                float _exp2_21 = approx_exp2((-(step_x01_145 * sg)) * 1.4426950408889634f);
                float _rcp_21 = approx_rcp(1.0f + _exp2_21);
                float step_sig01_149 = _rcp_21;
                float _exp2_22 = approx_exp2((-(step_x10_146 * sg)) * 1.4426950408889634f);
                float _rcp_22 = approx_rcp(1.0f + _exp2_22);
                float step_sig10_150 = _rcp_22;
                float _exp2_23 = approx_exp2((-(step_x11_147 * sg)) * 1.4426950408889634f);
                float _rcp_23 = approx_rcp(1.0f + _exp2_23);
                float step_sig11_151 = _rcp_23;
                float _min_44 = fminf(step_x00_144 * step_sig00_148, cl);
                float step_g00_152 = _min_44;
                float _min_45 = fminf(step_x01_145 * step_sig01_149, cl);
                float step_g01_153 = _min_45;
                float _min_46 = fminf(step_x10_146 * step_sig10_150, cl);
                float step_g10_154 = _min_46;
                float _min_47 = fminf(step_x11_147 * step_sig11_151, cl);
                float step_g11_155 = _min_47;
                float value00_156 = step_lin00_140 * sc * sg * step_g00_152;
                float value01_157 = step_lin01_141 * sc * sg * step_g01_153;
                float value10_158 = step_lin10_142 * sc * sg * step_g10_154;
                float value11_159 = step_lin11_143 * sc * sg * step_g11_155;
                float _fabs_20 = fabsf(value00_156);
                float _fabs_21 = fabsf(value10_158);
                float _max_64 = max_noftz(_fabs_20, _fabs_21);
                float block_max0_160 = _max_64;
                float _fabs_22 = fabsf(value01_157);
                float _fabs_23 = fabsf(value11_159);
                float _max_65 = max_noftz(_fabs_22, _fabs_23);
                float block_max1_161 = _max_65;
                float _shfl_xor_30 = __shfl_xor_sync(0xFFFFFFFF, block_max0_160, 4);
                float _max_66 = max_noftz(block_max0_160, _shfl_xor_30);
                block_max0_160 = _max_66;
                float _shfl_xor_31 = __shfl_xor_sync(0xFFFFFFFF, block_max1_161, 4);
                float _max_67 = max_noftz(block_max1_161, _shfl_xor_31);
                block_max1_161 = _max_67;
                float _shfl_xor_32 = __shfl_xor_sync(0xFFFFFFFF, block_max0_160, 8);
                float _max_68 = max_noftz(block_max0_160, _shfl_xor_32);
                block_max0_160 = _max_68;
                float _shfl_xor_33 = __shfl_xor_sync(0xFFFFFFFF, block_max1_161, 8);
                float _max_69 = max_noftz(block_max1_161, _shfl_xor_33);
                block_max1_161 = _max_69;
                float _shfl_xor_34 = __shfl_xor_sync(0xFFFFFFFF, block_max0_160, 16);
                float _max_70 = max_noftz(block_max0_160, _shfl_xor_34);
                block_max0_160 = _max_70;
                float _shfl_xor_35 = __shfl_xor_sync(0xFFFFFFFF, block_max1_161, 16);
                float _max_71 = max_noftz(block_max1_161, _shfl_xor_35);
                block_max1_161 = _max_71;
                float _fp8_rt_10;
                uint16_t _e4m3x2_20;
                uint32_t _f16x2_20;
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_20) : "f"(0.0f), "f"(block_max0_160 * 0.16666666666666666f));
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_20) : "h"(_e4m3x2_20));
                uint16_t _fp8_h0_20 = (uint16_t)(_f16x2_20 & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_10) : "h"(_fp8_h0_20));
                float scale0_162 = _fp8_rt_10;
                float _fp8_rt_11;
                uint16_t _e4m3x2_21;
                uint32_t _f16x2_21;
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_21) : "f"(0.0f), "f"(block_max1_161 * 0.16666666666666666f));
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_21) : "h"(_e4m3x2_21));
                uint16_t _fp8_h0_21 = (uint16_t)(_f16x2_21 & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_11) : "h"(_fp8_h0_21));
                float scale1_163 = _fp8_rt_11;
                float inv_scale0_164 = 0.0f;
                float inv_scale1_165 = 0.0f;
                if (scale0_162 != 0.0f) {
                    inv_scale0_164 = 1.0f / scale0_162;
                }
                if (scale1_163 != 0.0f) {
                    inv_scale1_165 = 1.0f / scale1_163;
                }
                quant_pair[0] = value00_156 * inv_scale0_164;
                quant_pair[1] = value10_158 * inv_scale0_164;
                uint32_t _slice_lo_mask_5;
                {
                    int _lim_22 = 2;
                    if (_lim_22 <= 0) { _slice_lo_mask_5 = 0u; }
                    else if (_lim_22 >= 8) { _slice_lo_mask_5 = ((1u << 8) - 1u); }
                    else {
                        asm volatile("{"
                            ".reg .u32 t;\n\t"
                            "shl.b32 t, 1, %1;\n\t"
                            "add.u32 %0, t, -1;\n\t"
                            "}" : "=r"(_slice_lo_mask_5) : "r"(_lim_22));
                    }
                }
                uint32_t _slice_hi_mask_5;
                {
                    int _lim_23 = 8;
                    if (_lim_23 <= 0) { _slice_hi_mask_5 = 0u; }
                    else if (_lim_23 >= 8) { _slice_hi_mask_5 = ((1u << 8) - 1u); }
                    else {
                        asm volatile("{"
                            ".reg .u32 t;\n\t"
                            "shl.b32 t, 1, %1;\n\t"
                            "add.u32 %0, t, -1;\n\t"
                            "}" : "=r"(_slice_hi_mask_5) : "r"(_lim_23));
                    }
                }
                if (!(_slice_lo_mask_5 | ~_slice_hi_mask_5 & (1u << 0))) quant_pair[0] = 0.0f;
                if (!(_slice_lo_mask_5 | ~_slice_hi_mask_5 & (1u << 1))) quant_pair[1] = 0.0f;
                if (!(_slice_lo_mask_5 | ~_slice_hi_mask_5 & (1u << 2))) quant_pair[2] = 0.0f;
                if (!(_slice_lo_mask_5 | ~_slice_hi_mask_5 & (1u << 3))) quant_pair[3] = 0.0f;
                if (!(_slice_lo_mask_5 | ~_slice_hi_mask_5 & (1u << 4))) quant_pair[4] = 0.0f;
                if (!(_slice_lo_mask_5 | ~_slice_hi_mask_5 & (1u << 5))) quant_pair[5] = 0.0f;
                if (!(_slice_lo_mask_5 | ~_slice_hi_mask_5 & (1u << 6))) quant_pair[6] = 0.0f;
                if (!(_slice_lo_mask_5 | ~_slice_hi_mask_5 & (1u << 7))) quant_pair[7] = 0.0f;
                uint32_t _fp4_10[1];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_10[0]) : "f"(quant_pair[0]), "f"(quant_pair[1]), "f"(quant_pair[2]), "f"(quant_pair[3]), "f"(quant_pair[4]), "f"(quant_pair[5]), "f"(quant_pair[6]), "f"(quant_pair[7]));
                quant_pair[0] = value01_157 * inv_scale1_165;
                quant_pair[1] = value11_159 * inv_scale1_165;
                uint32_t _fp4_11[1];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_11[0]) : "f"(quant_pair[0]), "f"(quant_pair[1]), "f"(quant_pair[2]), "f"(quant_pair[3]), "f"(quant_pair[4]), "f"(quant_pair[5]), "f"(quant_pair[6]), "f"(quant_pair[7]));
                if (lane_1 < 4) {
                    int sf_feature_5 = m_tile * 4 + (unsigned int)warp_local;
                    int sf_tile_stride_5 = 2 * (M_out / 64) * 512;
                    int sf_base_5 = n_tile * (unsigned int)sf_tile_stride_5 + (unsigned int)(token0_138 / 128 * (M_out / 64) * 512) + (unsigned int)(sf_feature_5 / 4 * 512) + (unsigned int)(token0_138 % 32 * 16) + (unsigned int)(token0_138 % 128 / 32 * 4) + (unsigned int)(sf_feature_5 % 4);
                    if (token0_138 < valid_rows) {
                        {
                            unsigned short _fp8_pair;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(scale0_162));
                            *(reinterpret_cast<unsigned char*>(SFC + sf_base_5) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                        }
                    }
                    if (token1_139 < valid_rows) {
                        {
                            unsigned short _fp8_pair;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(scale1_163));
                            *(reinterpret_cast<unsigned char*>(SFC + (sf_base_5 + 16)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                        }
                    }
                }
                int smem_flat0_166 = local_token0_136 * 64 + base_row;
                int smem_flat1_167 = local_token1_137 * 64 + base_row;
                int smem_index0_168 = smem_flat0_166 / 2 ^ smem_flat0_166 / 256 % 2 * 16;
                int smem_index1_169 = smem_flat1_167 / 2 ^ smem_flat1_167 / 256 % 2 * 16;
                epi_staging[warp_group * 2048 + smem_index0_168] = _fp4_10[0];
                epi_staging[warp_group * 2048 + smem_index1_169] = _fp4_11[0];
                int local_token0_170 = lane_1 % 4 * 2 + 48;
                int local_token1_171 = local_token0_170 + 1;
                int token0_172 = token_block * 64 + local_token0_170;
                int token1_173 = token0_172 + 1;
                float _max_72 = max_noftz(_tmem_load_0[24], neg_cl);
                float _min_48 = fminf(_max_72, cl);
                float step_lin00_174 = _min_48;
                float _max_73 = max_noftz(_tmem_load_0[25], neg_cl);
                float _min_49 = fminf(_max_73, cl);
                float step_lin01_175 = _min_49;
                float _max_74 = max_noftz(_tmem_load_1[24], neg_cl);
                float _min_50 = fminf(_max_74, cl);
                float step_lin10_176 = _min_50;
                float _max_75 = max_noftz(_tmem_load_1[25], neg_cl);
                float _min_51 = fminf(_max_75, cl);
                float step_lin11_177 = _min_51;
                float step_x00_178 = _tmem_load_0[26];
                float step_x01_179 = _tmem_load_0[27];
                float step_x10_180 = _tmem_load_1[26];
                float step_x11_181 = _tmem_load_1[27];
                float _exp2_24 = approx_exp2((-(step_x00_178 * sg)) * 1.4426950408889634f);
                float _rcp_24 = approx_rcp(1.0f + _exp2_24);
                float step_sig00_182 = _rcp_24;
                float _exp2_25 = approx_exp2((-(step_x01_179 * sg)) * 1.4426950408889634f);
                float _rcp_25 = approx_rcp(1.0f + _exp2_25);
                float step_sig01_183 = _rcp_25;
                float _exp2_26 = approx_exp2((-(step_x10_180 * sg)) * 1.4426950408889634f);
                float _rcp_26 = approx_rcp(1.0f + _exp2_26);
                float step_sig10_184 = _rcp_26;
                float _exp2_27 = approx_exp2((-(step_x11_181 * sg)) * 1.4426950408889634f);
                float _rcp_27 = approx_rcp(1.0f + _exp2_27);
                float step_sig11_185 = _rcp_27;
                float _min_52 = fminf(step_x00_178 * step_sig00_182, cl);
                float step_g00_186 = _min_52;
                float _min_53 = fminf(step_x01_179 * step_sig01_183, cl);
                float step_g01_187 = _min_53;
                float _min_54 = fminf(step_x10_180 * step_sig10_184, cl);
                float step_g10_188 = _min_54;
                float _min_55 = fminf(step_x11_181 * step_sig11_185, cl);
                float step_g11_189 = _min_55;
                float value00_190 = step_lin00_174 * sc * sg * step_g00_186;
                float value01_191 = step_lin01_175 * sc * sg * step_g01_187;
                float value10_192 = step_lin10_176 * sc * sg * step_g10_188;
                float value11_193 = step_lin11_177 * sc * sg * step_g11_189;
                float _fabs_24 = fabsf(value00_190);
                float _fabs_25 = fabsf(value10_192);
                float _max_76 = max_noftz(_fabs_24, _fabs_25);
                float block_max0_194 = _max_76;
                float _fabs_26 = fabsf(value01_191);
                float _fabs_27 = fabsf(value11_193);
                float _max_77 = max_noftz(_fabs_26, _fabs_27);
                float block_max1_195 = _max_77;
                float _shfl_xor_36 = __shfl_xor_sync(0xFFFFFFFF, block_max0_194, 4);
                float _max_78 = max_noftz(block_max0_194, _shfl_xor_36);
                block_max0_194 = _max_78;
                float _shfl_xor_37 = __shfl_xor_sync(0xFFFFFFFF, block_max1_195, 4);
                float _max_79 = max_noftz(block_max1_195, _shfl_xor_37);
                block_max1_195 = _max_79;
                float _shfl_xor_38 = __shfl_xor_sync(0xFFFFFFFF, block_max0_194, 8);
                float _max_80 = max_noftz(block_max0_194, _shfl_xor_38);
                block_max0_194 = _max_80;
                float _shfl_xor_39 = __shfl_xor_sync(0xFFFFFFFF, block_max1_195, 8);
                float _max_81 = max_noftz(block_max1_195, _shfl_xor_39);
                block_max1_195 = _max_81;
                float _shfl_xor_40 = __shfl_xor_sync(0xFFFFFFFF, block_max0_194, 16);
                float _max_82 = max_noftz(block_max0_194, _shfl_xor_40);
                block_max0_194 = _max_82;
                float _shfl_xor_41 = __shfl_xor_sync(0xFFFFFFFF, block_max1_195, 16);
                float _max_83 = max_noftz(block_max1_195, _shfl_xor_41);
                block_max1_195 = _max_83;
                float _fp8_rt_12;
                uint16_t _e4m3x2_24;
                uint32_t _f16x2_24;
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_24) : "f"(0.0f), "f"(block_max0_194 * 0.16666666666666666f));
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_24) : "h"(_e4m3x2_24));
                uint16_t _fp8_h0_24 = (uint16_t)(_f16x2_24 & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_12) : "h"(_fp8_h0_24));
                float scale0_196 = _fp8_rt_12;
                float _fp8_rt_13;
                uint16_t _e4m3x2_25;
                uint32_t _f16x2_25;
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_25) : "f"(0.0f), "f"(block_max1_195 * 0.16666666666666666f));
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_25) : "h"(_e4m3x2_25));
                uint16_t _fp8_h0_25 = (uint16_t)(_f16x2_25 & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_13) : "h"(_fp8_h0_25));
                float scale1_197 = _fp8_rt_13;
                float inv_scale0_198 = 0.0f;
                float inv_scale1_199 = 0.0f;
                if (scale0_196 != 0.0f) {
                    inv_scale0_198 = 1.0f / scale0_196;
                }
                if (scale1_197 != 0.0f) {
                    inv_scale1_199 = 1.0f / scale1_197;
                }
                quant_pair[0] = value00_190 * inv_scale0_198;
                quant_pair[1] = value10_192 * inv_scale0_198;
                uint32_t _slice_lo_mask_6;
                {
                    int _lim_26 = 2;
                    if (_lim_26 <= 0) { _slice_lo_mask_6 = 0u; }
                    else if (_lim_26 >= 8) { _slice_lo_mask_6 = ((1u << 8) - 1u); }
                    else {
                        asm volatile("{"
                            ".reg .u32 t;\n\t"
                            "shl.b32 t, 1, %1;\n\t"
                            "add.u32 %0, t, -1;\n\t"
                            "}" : "=r"(_slice_lo_mask_6) : "r"(_lim_26));
                    }
                }
                uint32_t _slice_hi_mask_6;
                {
                    int _lim_27 = 8;
                    if (_lim_27 <= 0) { _slice_hi_mask_6 = 0u; }
                    else if (_lim_27 >= 8) { _slice_hi_mask_6 = ((1u << 8) - 1u); }
                    else {
                        asm volatile("{"
                            ".reg .u32 t;\n\t"
                            "shl.b32 t, 1, %1;\n\t"
                            "add.u32 %0, t, -1;\n\t"
                            "}" : "=r"(_slice_hi_mask_6) : "r"(_lim_27));
                    }
                }
                if (!(_slice_lo_mask_6 | ~_slice_hi_mask_6 & (1u << 0))) quant_pair[0] = 0.0f;
                if (!(_slice_lo_mask_6 | ~_slice_hi_mask_6 & (1u << 1))) quant_pair[1] = 0.0f;
                if (!(_slice_lo_mask_6 | ~_slice_hi_mask_6 & (1u << 2))) quant_pair[2] = 0.0f;
                if (!(_slice_lo_mask_6 | ~_slice_hi_mask_6 & (1u << 3))) quant_pair[3] = 0.0f;
                if (!(_slice_lo_mask_6 | ~_slice_hi_mask_6 & (1u << 4))) quant_pair[4] = 0.0f;
                if (!(_slice_lo_mask_6 | ~_slice_hi_mask_6 & (1u << 5))) quant_pair[5] = 0.0f;
                if (!(_slice_lo_mask_6 | ~_slice_hi_mask_6 & (1u << 6))) quant_pair[6] = 0.0f;
                if (!(_slice_lo_mask_6 | ~_slice_hi_mask_6 & (1u << 7))) quant_pair[7] = 0.0f;
                uint32_t _fp4_12[1];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_12[0]) : "f"(quant_pair[0]), "f"(quant_pair[1]), "f"(quant_pair[2]), "f"(quant_pair[3]), "f"(quant_pair[4]), "f"(quant_pair[5]), "f"(quant_pair[6]), "f"(quant_pair[7]));
                quant_pair[0] = value01_191 * inv_scale1_199;
                quant_pair[1] = value11_193 * inv_scale1_199;
                uint32_t _fp4_13[1];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_13[0]) : "f"(quant_pair[0]), "f"(quant_pair[1]), "f"(quant_pair[2]), "f"(quant_pair[3]), "f"(quant_pair[4]), "f"(quant_pair[5]), "f"(quant_pair[6]), "f"(quant_pair[7]));
                if (lane_1 < 4) {
                    int sf_feature_6 = m_tile * 4 + (unsigned int)warp_local;
                    int sf_tile_stride_6 = 2 * (M_out / 64) * 512;
                    int sf_base_6 = n_tile * (unsigned int)sf_tile_stride_6 + (unsigned int)(token0_172 / 128 * (M_out / 64) * 512) + (unsigned int)(sf_feature_6 / 4 * 512) + (unsigned int)(token0_172 % 32 * 16) + (unsigned int)(token0_172 % 128 / 32 * 4) + (unsigned int)(sf_feature_6 % 4);
                    if (token0_172 < valid_rows) {
                        {
                            unsigned short _fp8_pair;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(scale0_196));
                            *(reinterpret_cast<unsigned char*>(SFC + sf_base_6) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                        }
                    }
                    if (token1_173 < valid_rows) {
                        {
                            unsigned short _fp8_pair;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(scale1_197));
                            *(reinterpret_cast<unsigned char*>(SFC + (sf_base_6 + 16)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                        }
                    }
                }
                int smem_flat0_200 = local_token0_170 * 64 + base_row;
                int smem_flat1_201 = local_token1_171 * 64 + base_row;
                int smem_index0_202 = smem_flat0_200 / 2 ^ smem_flat0_200 / 256 % 2 * 16;
                int smem_index1_203 = smem_flat1_201 / 2 ^ smem_flat1_201 / 256 % 2 * 16;
                epi_staging[warp_group * 2048 + smem_index0_202] = _fp4_12[0];
                epi_staging[warp_group * 2048 + smem_index1_203] = _fp4_13[0];
                int local_token0_204 = lane_1 % 4 * 2 + 56;
                int local_token1_205 = local_token0_204 + 1;
                int token0_206 = token_block * 64 + local_token0_204;
                int token1_207 = token0_206 + 1;
                float _max_84 = max_noftz(_tmem_load_0[28], neg_cl);
                float _min_56 = fminf(_max_84, cl);
                float step_lin00_208 = _min_56;
                float _max_85 = max_noftz(_tmem_load_0[29], neg_cl);
                float _min_57 = fminf(_max_85, cl);
                float step_lin01_209 = _min_57;
                float _max_86 = max_noftz(_tmem_load_1[28], neg_cl);
                float _min_58 = fminf(_max_86, cl);
                float step_lin10_210 = _min_58;
                float _max_87 = max_noftz(_tmem_load_1[29], neg_cl);
                float _min_59 = fminf(_max_87, cl);
                float step_lin11_211 = _min_59;
                float step_x00_212 = _tmem_load_0[30];
                float step_x01_213 = _tmem_load_0[31];
                float step_x10_214 = _tmem_load_1[30];
                float step_x11_215 = _tmem_load_1[31];
                float _exp2_28 = approx_exp2((-(step_x00_212 * sg)) * 1.4426950408889634f);
                float _rcp_28 = approx_rcp(1.0f + _exp2_28);
                float step_sig00_216 = _rcp_28;
                float _exp2_29 = approx_exp2((-(step_x01_213 * sg)) * 1.4426950408889634f);
                float _rcp_29 = approx_rcp(1.0f + _exp2_29);
                float step_sig01_217 = _rcp_29;
                float _exp2_30 = approx_exp2((-(step_x10_214 * sg)) * 1.4426950408889634f);
                float _rcp_30 = approx_rcp(1.0f + _exp2_30);
                float step_sig10_218 = _rcp_30;
                float _exp2_31 = approx_exp2((-(step_x11_215 * sg)) * 1.4426950408889634f);
                float _rcp_31 = approx_rcp(1.0f + _exp2_31);
                float step_sig11_219 = _rcp_31;
                float _min_60 = fminf(step_x00_212 * step_sig00_216, cl);
                float step_g00_220 = _min_60;
                float _min_61 = fminf(step_x01_213 * step_sig01_217, cl);
                float step_g01_221 = _min_61;
                float _min_62 = fminf(step_x10_214 * step_sig10_218, cl);
                float step_g10_222 = _min_62;
                float _min_63 = fminf(step_x11_215 * step_sig11_219, cl);
                float step_g11_223 = _min_63;
                float value00_224 = step_lin00_208 * sc * sg * step_g00_220;
                float value01_225 = step_lin01_209 * sc * sg * step_g01_221;
                float value10_226 = step_lin10_210 * sc * sg * step_g10_222;
                float value11_227 = step_lin11_211 * sc * sg * step_g11_223;
                float _fabs_28 = fabsf(value00_224);
                float _fabs_29 = fabsf(value10_226);
                float _max_88 = max_noftz(_fabs_28, _fabs_29);
                float block_max0_228 = _max_88;
                float _fabs_30 = fabsf(value01_225);
                float _fabs_31 = fabsf(value11_227);
                float _max_89 = max_noftz(_fabs_30, _fabs_31);
                float block_max1_229 = _max_89;
                float _shfl_xor_42 = __shfl_xor_sync(0xFFFFFFFF, block_max0_228, 4);
                float _max_90 = max_noftz(block_max0_228, _shfl_xor_42);
                block_max0_228 = _max_90;
                float _shfl_xor_43 = __shfl_xor_sync(0xFFFFFFFF, block_max1_229, 4);
                float _max_91 = max_noftz(block_max1_229, _shfl_xor_43);
                block_max1_229 = _max_91;
                float _shfl_xor_44 = __shfl_xor_sync(0xFFFFFFFF, block_max0_228, 8);
                float _max_92 = max_noftz(block_max0_228, _shfl_xor_44);
                block_max0_228 = _max_92;
                float _shfl_xor_45 = __shfl_xor_sync(0xFFFFFFFF, block_max1_229, 8);
                float _max_93 = max_noftz(block_max1_229, _shfl_xor_45);
                block_max1_229 = _max_93;
                float _shfl_xor_46 = __shfl_xor_sync(0xFFFFFFFF, block_max0_228, 16);
                float _max_94 = max_noftz(block_max0_228, _shfl_xor_46);
                block_max0_228 = _max_94;
                float _shfl_xor_47 = __shfl_xor_sync(0xFFFFFFFF, block_max1_229, 16);
                float _max_95 = max_noftz(block_max1_229, _shfl_xor_47);
                block_max1_229 = _max_95;
                float _fp8_rt_14;
                uint16_t _e4m3x2_28;
                uint32_t _f16x2_28;
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_28) : "f"(0.0f), "f"(block_max0_228 * 0.16666666666666666f));
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_28) : "h"(_e4m3x2_28));
                uint16_t _fp8_h0_28 = (uint16_t)(_f16x2_28 & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_14) : "h"(_fp8_h0_28));
                float scale0_230 = _fp8_rt_14;
                float _fp8_rt_15;
                uint16_t _e4m3x2_29;
                uint32_t _f16x2_29;
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_29) : "f"(0.0f), "f"(block_max1_229 * 0.16666666666666666f));
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_29) : "h"(_e4m3x2_29));
                uint16_t _fp8_h0_29 = (uint16_t)(_f16x2_29 & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_15) : "h"(_fp8_h0_29));
                float scale1_231 = _fp8_rt_15;
                float inv_scale0_232 = 0.0f;
                float inv_scale1_233 = 0.0f;
                if (scale0_230 != 0.0f) {
                    inv_scale0_232 = 1.0f / scale0_230;
                }
                if (scale1_231 != 0.0f) {
                    inv_scale1_233 = 1.0f / scale1_231;
                }
                quant_pair[0] = value00_224 * inv_scale0_232;
                quant_pair[1] = value10_226 * inv_scale0_232;
                uint32_t _slice_lo_mask_7;
                {
                    int _lim_30 = 2;
                    if (_lim_30 <= 0) { _slice_lo_mask_7 = 0u; }
                    else if (_lim_30 >= 8) { _slice_lo_mask_7 = ((1u << 8) - 1u); }
                    else {
                        asm volatile("{"
                            ".reg .u32 t;\n\t"
                            "shl.b32 t, 1, %1;\n\t"
                            "add.u32 %0, t, -1;\n\t"
                            "}" : "=r"(_slice_lo_mask_7) : "r"(_lim_30));
                    }
                }
                uint32_t _slice_hi_mask_7;
                {
                    int _lim_31 = 8;
                    if (_lim_31 <= 0) { _slice_hi_mask_7 = 0u; }
                    else if (_lim_31 >= 8) { _slice_hi_mask_7 = ((1u << 8) - 1u); }
                    else {
                        asm volatile("{"
                            ".reg .u32 t;\n\t"
                            "shl.b32 t, 1, %1;\n\t"
                            "add.u32 %0, t, -1;\n\t"
                            "}" : "=r"(_slice_hi_mask_7) : "r"(_lim_31));
                    }
                }
                if (!(_slice_lo_mask_7 | ~_slice_hi_mask_7 & (1u << 0))) quant_pair[0] = 0.0f;
                if (!(_slice_lo_mask_7 | ~_slice_hi_mask_7 & (1u << 1))) quant_pair[1] = 0.0f;
                if (!(_slice_lo_mask_7 | ~_slice_hi_mask_7 & (1u << 2))) quant_pair[2] = 0.0f;
                if (!(_slice_lo_mask_7 | ~_slice_hi_mask_7 & (1u << 3))) quant_pair[3] = 0.0f;
                if (!(_slice_lo_mask_7 | ~_slice_hi_mask_7 & (1u << 4))) quant_pair[4] = 0.0f;
                if (!(_slice_lo_mask_7 | ~_slice_hi_mask_7 & (1u << 5))) quant_pair[5] = 0.0f;
                if (!(_slice_lo_mask_7 | ~_slice_hi_mask_7 & (1u << 6))) quant_pair[6] = 0.0f;
                if (!(_slice_lo_mask_7 | ~_slice_hi_mask_7 & (1u << 7))) quant_pair[7] = 0.0f;
                uint32_t _fp4_14[1];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_14[0]) : "f"(quant_pair[0]), "f"(quant_pair[1]), "f"(quant_pair[2]), "f"(quant_pair[3]), "f"(quant_pair[4]), "f"(quant_pair[5]), "f"(quant_pair[6]), "f"(quant_pair[7]));
                quant_pair[0] = value01_225 * inv_scale1_233;
                quant_pair[1] = value11_227 * inv_scale1_233;
                uint32_t _fp4_15[1];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_15[0]) : "f"(quant_pair[0]), "f"(quant_pair[1]), "f"(quant_pair[2]), "f"(quant_pair[3]), "f"(quant_pair[4]), "f"(quant_pair[5]), "f"(quant_pair[6]), "f"(quant_pair[7]));
                if (lane_1 < 4) {
                    int sf_feature_7 = m_tile * 4 + (unsigned int)warp_local;
                    int sf_tile_stride_7 = 2 * (M_out / 64) * 512;
                    int sf_base_7 = n_tile * (unsigned int)sf_tile_stride_7 + (unsigned int)(token0_206 / 128 * (M_out / 64) * 512) + (unsigned int)(sf_feature_7 / 4 * 512) + (unsigned int)(token0_206 % 32 * 16) + (unsigned int)(token0_206 % 128 / 32 * 4) + (unsigned int)(sf_feature_7 % 4);
                    if (token0_206 < valid_rows) {
                        {
                            unsigned short _fp8_pair;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(scale0_230));
                            *(reinterpret_cast<unsigned char*>(SFC + sf_base_7) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                        }
                    }
                    if (token1_207 < valid_rows) {
                        {
                            unsigned short _fp8_pair;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(scale1_231));
                            *(reinterpret_cast<unsigned char*>(SFC + (sf_base_7 + 16)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                        }
                    }
                }
                int smem_flat0_234 = local_token0_204 * 64 + base_row;
                int smem_flat1_235 = local_token1_205 * 64 + base_row;
                int smem_index0_236 = smem_flat0_234 / 2 ^ smem_flat0_234 / 256 % 2 * 16;
                int smem_index1_237 = smem_flat1_235 / 2 ^ smem_flat1_235 / 256 % 2 * 16;
                epi_staging[warp_group * 2048 + smem_index0_236] = _fp4_14[0];
                epi_staging[warp_group * 2048 + smem_index1_237] = _fp4_15[0];
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                if (warp_group == 0) {
                    asm volatile("barrier.sync 7, 128;" ::: "memory");
                    if (warp == 0) {
                        if (elect_sync()) {
                            int padding_rows = (256 - valid_rows % 256) % 256;
                            tma_store_4d((&C), m_tile * 64, padding_rows + token_block * 64, 1073741824, n_tile * 256 - (unsigned int)padding_rows + 1073741824, epi_staging_addr);
                        }
                    }
                }
                if (warp_group == 1) {
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                    if (warp == 4) {
                        if (elect_sync()) {
                            int padding_rows_1 = (256 - valid_rows % 256) % 256;
                            tma_store_4d((&C), m_tile * 64, padding_rows_1 + token_block * 64, 1073741824, n_tile * 256 - (unsigned int)padding_rows_1 + 1073741824, epi_staging_addr + 2048);
                        }
                    }
                }
                asm volatile("cp.async.bulk.commit_group;");
                if (warp_group == 0) {
                    asm volatile("barrier.sync 7, 128;" ::: "memory");
                }
                if (warp_group == 1) {
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                }
                int token_block_238 = 2 + warp_group;
                if (epilogue_local_idx == 0) {
                    token_block_238 = (token_block_238 + 3) % 4;
                }
                int accum_col_239 = epilogue_local_idx * 192 + token_block_238 * 64;
                int row_addr_240 = warp_local * 32 << 16;
                float _tmem_load_2[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[31]))
                    : "r"(taddr + (unsigned int)row_addr_240 + (unsigned int)accum_col_239));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float _tmem_load_3[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[31]))
                    : "r"(taddr + (unsigned int)row_addr_240 + 1048576 + (unsigned int)accum_col_239));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int base_row_241 = warp_local * 16 + lane_1 / 4 * 2;
                asm volatile("cp.async.bulk.wait_group.read 0;");
                if (warp_group == 0) {
                    asm volatile("barrier.sync 7, 128;" ::: "memory");
                }
                if (warp_group == 1) {
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                }
                int local_token0_242 = lane_1 % 4 * 2;
                int local_token1_243 = local_token0_242 + 1;
                int token0_244 = token_block_238 * 64 + local_token0_242;
                int token1_245 = token0_244 + 1;
                float _max_96 = max_noftz(_tmem_load_2[0], neg_cl);
                float _min_64 = fminf(_max_96, cl);
                float step_lin00_246 = _min_64;
                float _max_97 = max_noftz(_tmem_load_2[1], neg_cl);
                float _min_65 = fminf(_max_97, cl);
                float step_lin01_247 = _min_65;
                float _max_98 = max_noftz(_tmem_load_3[0], neg_cl);
                float _min_66 = fminf(_max_98, cl);
                float step_lin10_248 = _min_66;
                float _max_99 = max_noftz(_tmem_load_3[1], neg_cl);
                float _min_67 = fminf(_max_99, cl);
                float step_lin11_249 = _min_67;
                float step_x00_250 = _tmem_load_2[2];
                float step_x01_251 = _tmem_load_2[3];
                float step_x10_252 = _tmem_load_3[2];
                float step_x11_253 = _tmem_load_3[3];
                float _exp2_32 = approx_exp2((-(step_x00_250 * sg)) * 1.4426950408889634f);
                float _rcp_32 = approx_rcp(1.0f + _exp2_32);
                float step_sig00_254 = _rcp_32;
                float _exp2_33 = approx_exp2((-(step_x01_251 * sg)) * 1.4426950408889634f);
                float _rcp_33 = approx_rcp(1.0f + _exp2_33);
                float step_sig01_255 = _rcp_33;
                float _exp2_34 = approx_exp2((-(step_x10_252 * sg)) * 1.4426950408889634f);
                float _rcp_34 = approx_rcp(1.0f + _exp2_34);
                float step_sig10_256 = _rcp_34;
                float _exp2_35 = approx_exp2((-(step_x11_253 * sg)) * 1.4426950408889634f);
                float _rcp_35 = approx_rcp(1.0f + _exp2_35);
                float step_sig11_257 = _rcp_35;
                float _min_68 = fminf(step_x00_250 * step_sig00_254, cl);
                float step_g00_258 = _min_68;
                float _min_69 = fminf(step_x01_251 * step_sig01_255, cl);
                float step_g01_259 = _min_69;
                float _min_70 = fminf(step_x10_252 * step_sig10_256, cl);
                float step_g10_260 = _min_70;
                float _min_71 = fminf(step_x11_253 * step_sig11_257, cl);
                float step_g11_261 = _min_71;
                float value00_262 = step_lin00_246 * sc * sg * step_g00_258;
                float value01_263 = step_lin01_247 * sc * sg * step_g01_259;
                float value10_264 = step_lin10_248 * sc * sg * step_g10_260;
                float value11_265 = step_lin11_249 * sc * sg * step_g11_261;
                float _fabs_32 = fabsf(value00_262);
                float _fabs_33 = fabsf(value10_264);
                float _max_100 = max_noftz(_fabs_32, _fabs_33);
                float block_max0_266 = _max_100;
                float _fabs_34 = fabsf(value01_263);
                float _fabs_35 = fabsf(value11_265);
                float _max_101 = max_noftz(_fabs_34, _fabs_35);
                float block_max1_267 = _max_101;
                float _shfl_xor_48 = __shfl_xor_sync(0xFFFFFFFF, block_max0_266, 4);
                float _max_102 = max_noftz(block_max0_266, _shfl_xor_48);
                block_max0_266 = _max_102;
                float _shfl_xor_49 = __shfl_xor_sync(0xFFFFFFFF, block_max1_267, 4);
                float _max_103 = max_noftz(block_max1_267, _shfl_xor_49);
                block_max1_267 = _max_103;
                float _shfl_xor_50 = __shfl_xor_sync(0xFFFFFFFF, block_max0_266, 8);
                float _max_104 = max_noftz(block_max0_266, _shfl_xor_50);
                block_max0_266 = _max_104;
                float _shfl_xor_51 = __shfl_xor_sync(0xFFFFFFFF, block_max1_267, 8);
                float _max_105 = max_noftz(block_max1_267, _shfl_xor_51);
                block_max1_267 = _max_105;
                float _shfl_xor_52 = __shfl_xor_sync(0xFFFFFFFF, block_max0_266, 16);
                float _max_106 = max_noftz(block_max0_266, _shfl_xor_52);
                block_max0_266 = _max_106;
                float _shfl_xor_53 = __shfl_xor_sync(0xFFFFFFFF, block_max1_267, 16);
                float _max_107 = max_noftz(block_max1_267, _shfl_xor_53);
                block_max1_267 = _max_107;
                float _fp8_rt_16;
                uint16_t _e4m3x2_32;
                uint32_t _f16x2_32;
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_32) : "f"(0.0f), "f"(block_max0_266 * 0.16666666666666666f));
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_32) : "h"(_e4m3x2_32));
                uint16_t _fp8_h0_32 = (uint16_t)(_f16x2_32 & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_16) : "h"(_fp8_h0_32));
                float scale0_268 = _fp8_rt_16;
                float _fp8_rt_17;
                uint16_t _e4m3x2_33;
                uint32_t _f16x2_33;
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_33) : "f"(0.0f), "f"(block_max1_267 * 0.16666666666666666f));
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_33) : "h"(_e4m3x2_33));
                uint16_t _fp8_h0_33 = (uint16_t)(_f16x2_33 & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_17) : "h"(_fp8_h0_33));
                float scale1_269 = _fp8_rt_17;
                float inv_scale0_270 = 0.0f;
                float inv_scale1_271 = 0.0f;
                if (scale0_268 != 0.0f) {
                    inv_scale0_270 = 1.0f / scale0_268;
                }
                if (scale1_269 != 0.0f) {
                    inv_scale1_271 = 1.0f / scale1_269;
                }
                quant_pair[0] = value00_262 * inv_scale0_270;
                quant_pair[1] = value10_264 * inv_scale0_270;
                uint32_t _slice_lo_mask_8;
                {
                    int _lim_34 = 2;
                    if (_lim_34 <= 0) { _slice_lo_mask_8 = 0u; }
                    else if (_lim_34 >= 8) { _slice_lo_mask_8 = ((1u << 8) - 1u); }
                    else {
                        asm volatile("{"
                            ".reg .u32 t;\n\t"
                            "shl.b32 t, 1, %1;\n\t"
                            "add.u32 %0, t, -1;\n\t"
                            "}" : "=r"(_slice_lo_mask_8) : "r"(_lim_34));
                    }
                }
                uint32_t _slice_hi_mask_8;
                {
                    int _lim_35 = 8;
                    if (_lim_35 <= 0) { _slice_hi_mask_8 = 0u; }
                    else if (_lim_35 >= 8) { _slice_hi_mask_8 = ((1u << 8) - 1u); }
                    else {
                        asm volatile("{"
                            ".reg .u32 t;\n\t"
                            "shl.b32 t, 1, %1;\n\t"
                            "add.u32 %0, t, -1;\n\t"
                            "}" : "=r"(_slice_hi_mask_8) : "r"(_lim_35));
                    }
                }
                if (!(_slice_lo_mask_8 | ~_slice_hi_mask_8 & (1u << 0))) quant_pair[0] = 0.0f;
                if (!(_slice_lo_mask_8 | ~_slice_hi_mask_8 & (1u << 1))) quant_pair[1] = 0.0f;
                if (!(_slice_lo_mask_8 | ~_slice_hi_mask_8 & (1u << 2))) quant_pair[2] = 0.0f;
                if (!(_slice_lo_mask_8 | ~_slice_hi_mask_8 & (1u << 3))) quant_pair[3] = 0.0f;
                if (!(_slice_lo_mask_8 | ~_slice_hi_mask_8 & (1u << 4))) quant_pair[4] = 0.0f;
                if (!(_slice_lo_mask_8 | ~_slice_hi_mask_8 & (1u << 5))) quant_pair[5] = 0.0f;
                if (!(_slice_lo_mask_8 | ~_slice_hi_mask_8 & (1u << 6))) quant_pair[6] = 0.0f;
                if (!(_slice_lo_mask_8 | ~_slice_hi_mask_8 & (1u << 7))) quant_pair[7] = 0.0f;
                uint32_t _fp4_16[1];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_16[0]) : "f"(quant_pair[0]), "f"(quant_pair[1]), "f"(quant_pair[2]), "f"(quant_pair[3]), "f"(quant_pair[4]), "f"(quant_pair[5]), "f"(quant_pair[6]), "f"(quant_pair[7]));
                quant_pair[0] = value01_263 * inv_scale1_271;
                quant_pair[1] = value11_265 * inv_scale1_271;
                uint32_t _fp4_17[1];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_17[0]) : "f"(quant_pair[0]), "f"(quant_pair[1]), "f"(quant_pair[2]), "f"(quant_pair[3]), "f"(quant_pair[4]), "f"(quant_pair[5]), "f"(quant_pair[6]), "f"(quant_pair[7]));
                if (lane_1 < 4) {
                    int sf_feature_8 = m_tile * 4 + (unsigned int)warp_local;
                    int sf_tile_stride_8 = 2 * (M_out / 64) * 512;
                    int sf_base_8 = n_tile * (unsigned int)sf_tile_stride_8 + (unsigned int)(token0_244 / 128 * (M_out / 64) * 512) + (unsigned int)(sf_feature_8 / 4 * 512) + (unsigned int)(token0_244 % 32 * 16) + (unsigned int)(token0_244 % 128 / 32 * 4) + (unsigned int)(sf_feature_8 % 4);
                    if (token0_244 < valid_rows) {
                        {
                            unsigned short _fp8_pair;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(scale0_268));
                            *(reinterpret_cast<unsigned char*>(SFC + sf_base_8) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                        }
                    }
                    if (token1_245 < valid_rows) {
                        {
                            unsigned short _fp8_pair;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(scale1_269));
                            *(reinterpret_cast<unsigned char*>(SFC + (sf_base_8 + 16)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                        }
                    }
                }
                int smem_flat0_272 = local_token0_242 * 64 + base_row_241;
                int smem_flat1_273 = local_token1_243 * 64 + base_row_241;
                int smem_index0_274 = smem_flat0_272 / 2 ^ smem_flat0_272 / 256 % 2 * 16;
                int smem_index1_275 = smem_flat1_273 / 2 ^ smem_flat1_273 / 256 % 2 * 16;
                epi_staging[warp_group * 2048 + smem_index0_274] = _fp4_16[0];
                epi_staging[warp_group * 2048 + smem_index1_275] = _fp4_17[0];
                int local_token0_276 = lane_1 % 4 * 2 + 8;
                int local_token1_277 = local_token0_276 + 1;
                int token0_278 = token_block_238 * 64 + local_token0_276;
                int token1_279 = token0_278 + 1;
                float _max_108 = max_noftz(_tmem_load_2[4], neg_cl);
                float _min_72 = fminf(_max_108, cl);
                float step_lin00_280 = _min_72;
                float _max_109 = max_noftz(_tmem_load_2[5], neg_cl);
                float _min_73 = fminf(_max_109, cl);
                float step_lin01_281 = _min_73;
                float _max_110 = max_noftz(_tmem_load_3[4], neg_cl);
                float _min_74 = fminf(_max_110, cl);
                float step_lin10_282 = _min_74;
                float _max_111 = max_noftz(_tmem_load_3[5], neg_cl);
                float _min_75 = fminf(_max_111, cl);
                float step_lin11_283 = _min_75;
                float step_x00_284 = _tmem_load_2[6];
                float step_x01_285 = _tmem_load_2[7];
                float step_x10_286 = _tmem_load_3[6];
                float step_x11_287 = _tmem_load_3[7];
                float _exp2_36 = approx_exp2((-(step_x00_284 * sg)) * 1.4426950408889634f);
                float _rcp_36 = approx_rcp(1.0f + _exp2_36);
                float step_sig00_288 = _rcp_36;
                float _exp2_37 = approx_exp2((-(step_x01_285 * sg)) * 1.4426950408889634f);
                float _rcp_37 = approx_rcp(1.0f + _exp2_37);
                float step_sig01_289 = _rcp_37;
                float _exp2_38 = approx_exp2((-(step_x10_286 * sg)) * 1.4426950408889634f);
                float _rcp_38 = approx_rcp(1.0f + _exp2_38);
                float step_sig10_290 = _rcp_38;
                float _exp2_39 = approx_exp2((-(step_x11_287 * sg)) * 1.4426950408889634f);
                float _rcp_39 = approx_rcp(1.0f + _exp2_39);
                float step_sig11_291 = _rcp_39;
                float _min_76 = fminf(step_x00_284 * step_sig00_288, cl);
                float step_g00_292 = _min_76;
                float _min_77 = fminf(step_x01_285 * step_sig01_289, cl);
                float step_g01_293 = _min_77;
                float _min_78 = fminf(step_x10_286 * step_sig10_290, cl);
                float step_g10_294 = _min_78;
                float _min_79 = fminf(step_x11_287 * step_sig11_291, cl);
                float step_g11_295 = _min_79;
                float value00_296 = step_lin00_280 * sc * sg * step_g00_292;
                float value01_297 = step_lin01_281 * sc * sg * step_g01_293;
                float value10_298 = step_lin10_282 * sc * sg * step_g10_294;
                float value11_299 = step_lin11_283 * sc * sg * step_g11_295;
                float _fabs_36 = fabsf(value00_296);
                float _fabs_37 = fabsf(value10_298);
                float _max_112 = max_noftz(_fabs_36, _fabs_37);
                float block_max0_300 = _max_112;
                float _fabs_38 = fabsf(value01_297);
                float _fabs_39 = fabsf(value11_299);
                float _max_113 = max_noftz(_fabs_38, _fabs_39);
                float block_max1_301 = _max_113;
                float _shfl_xor_54 = __shfl_xor_sync(0xFFFFFFFF, block_max0_300, 4);
                float _max_114 = max_noftz(block_max0_300, _shfl_xor_54);
                block_max0_300 = _max_114;
                float _shfl_xor_55 = __shfl_xor_sync(0xFFFFFFFF, block_max1_301, 4);
                float _max_115 = max_noftz(block_max1_301, _shfl_xor_55);
                block_max1_301 = _max_115;
                float _shfl_xor_56 = __shfl_xor_sync(0xFFFFFFFF, block_max0_300, 8);
                float _max_116 = max_noftz(block_max0_300, _shfl_xor_56);
                block_max0_300 = _max_116;
                float _shfl_xor_57 = __shfl_xor_sync(0xFFFFFFFF, block_max1_301, 8);
                float _max_117 = max_noftz(block_max1_301, _shfl_xor_57);
                block_max1_301 = _max_117;
                float _shfl_xor_58 = __shfl_xor_sync(0xFFFFFFFF, block_max0_300, 16);
                float _max_118 = max_noftz(block_max0_300, _shfl_xor_58);
                block_max0_300 = _max_118;
                float _shfl_xor_59 = __shfl_xor_sync(0xFFFFFFFF, block_max1_301, 16);
                float _max_119 = max_noftz(block_max1_301, _shfl_xor_59);
                block_max1_301 = _max_119;
                float _fp8_rt_18;
                uint16_t _e4m3x2_36;
                uint32_t _f16x2_36;
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_36) : "f"(0.0f), "f"(block_max0_300 * 0.16666666666666666f));
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_36) : "h"(_e4m3x2_36));
                uint16_t _fp8_h0_36 = (uint16_t)(_f16x2_36 & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_18) : "h"(_fp8_h0_36));
                float scale0_302 = _fp8_rt_18;
                float _fp8_rt_19;
                uint16_t _e4m3x2_37;
                uint32_t _f16x2_37;
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_37) : "f"(0.0f), "f"(block_max1_301 * 0.16666666666666666f));
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_37) : "h"(_e4m3x2_37));
                uint16_t _fp8_h0_37 = (uint16_t)(_f16x2_37 & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_19) : "h"(_fp8_h0_37));
                float scale1_303 = _fp8_rt_19;
                float inv_scale0_304 = 0.0f;
                float inv_scale1_305 = 0.0f;
                if (scale0_302 != 0.0f) {
                    inv_scale0_304 = 1.0f / scale0_302;
                }
                if (scale1_303 != 0.0f) {
                    inv_scale1_305 = 1.0f / scale1_303;
                }
                quant_pair[0] = value00_296 * inv_scale0_304;
                quant_pair[1] = value10_298 * inv_scale0_304;
                uint32_t _slice_lo_mask_9;
                {
                    int _lim_38 = 2;
                    if (_lim_38 <= 0) { _slice_lo_mask_9 = 0u; }
                    else if (_lim_38 >= 8) { _slice_lo_mask_9 = ((1u << 8) - 1u); }
                    else {
                        asm volatile("{"
                            ".reg .u32 t;\n\t"
                            "shl.b32 t, 1, %1;\n\t"
                            "add.u32 %0, t, -1;\n\t"
                            "}" : "=r"(_slice_lo_mask_9) : "r"(_lim_38));
                    }
                }
                uint32_t _slice_hi_mask_9;
                {
                    int _lim_39 = 8;
                    if (_lim_39 <= 0) { _slice_hi_mask_9 = 0u; }
                    else if (_lim_39 >= 8) { _slice_hi_mask_9 = ((1u << 8) - 1u); }
                    else {
                        asm volatile("{"
                            ".reg .u32 t;\n\t"
                            "shl.b32 t, 1, %1;\n\t"
                            "add.u32 %0, t, -1;\n\t"
                            "}" : "=r"(_slice_hi_mask_9) : "r"(_lim_39));
                    }
                }
                if (!(_slice_lo_mask_9 | ~_slice_hi_mask_9 & (1u << 0))) quant_pair[0] = 0.0f;
                if (!(_slice_lo_mask_9 | ~_slice_hi_mask_9 & (1u << 1))) quant_pair[1] = 0.0f;
                if (!(_slice_lo_mask_9 | ~_slice_hi_mask_9 & (1u << 2))) quant_pair[2] = 0.0f;
                if (!(_slice_lo_mask_9 | ~_slice_hi_mask_9 & (1u << 3))) quant_pair[3] = 0.0f;
                if (!(_slice_lo_mask_9 | ~_slice_hi_mask_9 & (1u << 4))) quant_pair[4] = 0.0f;
                if (!(_slice_lo_mask_9 | ~_slice_hi_mask_9 & (1u << 5))) quant_pair[5] = 0.0f;
                if (!(_slice_lo_mask_9 | ~_slice_hi_mask_9 & (1u << 6))) quant_pair[6] = 0.0f;
                if (!(_slice_lo_mask_9 | ~_slice_hi_mask_9 & (1u << 7))) quant_pair[7] = 0.0f;
                uint32_t _fp4_18[1];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_18[0]) : "f"(quant_pair[0]), "f"(quant_pair[1]), "f"(quant_pair[2]), "f"(quant_pair[3]), "f"(quant_pair[4]), "f"(quant_pair[5]), "f"(quant_pair[6]), "f"(quant_pair[7]));
                quant_pair[0] = value01_297 * inv_scale1_305;
                quant_pair[1] = value11_299 * inv_scale1_305;
                uint32_t _fp4_19[1];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_19[0]) : "f"(quant_pair[0]), "f"(quant_pair[1]), "f"(quant_pair[2]), "f"(quant_pair[3]), "f"(quant_pair[4]), "f"(quant_pair[5]), "f"(quant_pair[6]), "f"(quant_pair[7]));
                if (lane_1 < 4) {
                    int sf_feature_9 = m_tile * 4 + (unsigned int)warp_local;
                    int sf_tile_stride_9 = 2 * (M_out / 64) * 512;
                    int sf_base_9 = n_tile * (unsigned int)sf_tile_stride_9 + (unsigned int)(token0_278 / 128 * (M_out / 64) * 512) + (unsigned int)(sf_feature_9 / 4 * 512) + (unsigned int)(token0_278 % 32 * 16) + (unsigned int)(token0_278 % 128 / 32 * 4) + (unsigned int)(sf_feature_9 % 4);
                    if (token0_278 < valid_rows) {
                        {
                            unsigned short _fp8_pair;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(scale0_302));
                            *(reinterpret_cast<unsigned char*>(SFC + sf_base_9) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                        }
                    }
                    if (token1_279 < valid_rows) {
                        {
                            unsigned short _fp8_pair;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(scale1_303));
                            *(reinterpret_cast<unsigned char*>(SFC + (sf_base_9 + 16)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                        }
                    }
                }
                int smem_flat0_306 = local_token0_276 * 64 + base_row_241;
                int smem_flat1_307 = local_token1_277 * 64 + base_row_241;
                int smem_index0_308 = smem_flat0_306 / 2 ^ smem_flat0_306 / 256 % 2 * 16;
                int smem_index1_309 = smem_flat1_307 / 2 ^ smem_flat1_307 / 256 % 2 * 16;
                epi_staging[warp_group * 2048 + smem_index0_308] = _fp4_18[0];
                epi_staging[warp_group * 2048 + smem_index1_309] = _fp4_19[0];
                int local_token0_310 = lane_1 % 4 * 2 + 16;
                int local_token1_311 = local_token0_310 + 1;
                int token0_312 = token_block_238 * 64 + local_token0_310;
                int token1_313 = token0_312 + 1;
                float _max_120 = max_noftz(_tmem_load_2[8], neg_cl);
                float _min_80 = fminf(_max_120, cl);
                float step_lin00_314 = _min_80;
                float _max_121 = max_noftz(_tmem_load_2[9], neg_cl);
                float _min_81 = fminf(_max_121, cl);
                float step_lin01_315 = _min_81;
                float _max_122 = max_noftz(_tmem_load_3[8], neg_cl);
                float _min_82 = fminf(_max_122, cl);
                float step_lin10_316 = _min_82;
                float _max_123 = max_noftz(_tmem_load_3[9], neg_cl);
                float _min_83 = fminf(_max_123, cl);
                float step_lin11_317 = _min_83;
                float step_x00_318 = _tmem_load_2[10];
                float step_x01_319 = _tmem_load_2[11];
                float step_x10_320 = _tmem_load_3[10];
                float step_x11_321 = _tmem_load_3[11];
                float _exp2_40 = approx_exp2((-(step_x00_318 * sg)) * 1.4426950408889634f);
                float _rcp_40 = approx_rcp(1.0f + _exp2_40);
                float step_sig00_322 = _rcp_40;
                float _exp2_41 = approx_exp2((-(step_x01_319 * sg)) * 1.4426950408889634f);
                float _rcp_41 = approx_rcp(1.0f + _exp2_41);
                float step_sig01_323 = _rcp_41;
                float _exp2_42 = approx_exp2((-(step_x10_320 * sg)) * 1.4426950408889634f);
                float _rcp_42 = approx_rcp(1.0f + _exp2_42);
                float step_sig10_324 = _rcp_42;
                float _exp2_43 = approx_exp2((-(step_x11_321 * sg)) * 1.4426950408889634f);
                float _rcp_43 = approx_rcp(1.0f + _exp2_43);
                float step_sig11_325 = _rcp_43;
                float _min_84 = fminf(step_x00_318 * step_sig00_322, cl);
                float step_g00_326 = _min_84;
                float _min_85 = fminf(step_x01_319 * step_sig01_323, cl);
                float step_g01_327 = _min_85;
                float _min_86 = fminf(step_x10_320 * step_sig10_324, cl);
                float step_g10_328 = _min_86;
                float _min_87 = fminf(step_x11_321 * step_sig11_325, cl);
                float step_g11_329 = _min_87;
                float value00_330 = step_lin00_314 * sc * sg * step_g00_326;
                float value01_331 = step_lin01_315 * sc * sg * step_g01_327;
                float value10_332 = step_lin10_316 * sc * sg * step_g10_328;
                float value11_333 = step_lin11_317 * sc * sg * step_g11_329;
                float _fabs_40 = fabsf(value00_330);
                float _fabs_41 = fabsf(value10_332);
                float _max_124 = max_noftz(_fabs_40, _fabs_41);
                float block_max0_334 = _max_124;
                float _fabs_42 = fabsf(value01_331);
                float _fabs_43 = fabsf(value11_333);
                float _max_125 = max_noftz(_fabs_42, _fabs_43);
                float block_max1_335 = _max_125;
                float _shfl_xor_60 = __shfl_xor_sync(0xFFFFFFFF, block_max0_334, 4);
                float _max_126 = max_noftz(block_max0_334, _shfl_xor_60);
                block_max0_334 = _max_126;
                float _shfl_xor_61 = __shfl_xor_sync(0xFFFFFFFF, block_max1_335, 4);
                float _max_127 = max_noftz(block_max1_335, _shfl_xor_61);
                block_max1_335 = _max_127;
                float _shfl_xor_62 = __shfl_xor_sync(0xFFFFFFFF, block_max0_334, 8);
                float _max_128 = max_noftz(block_max0_334, _shfl_xor_62);
                block_max0_334 = _max_128;
                float _shfl_xor_63 = __shfl_xor_sync(0xFFFFFFFF, block_max1_335, 8);
                float _max_129 = max_noftz(block_max1_335, _shfl_xor_63);
                block_max1_335 = _max_129;
                float _shfl_xor_64 = __shfl_xor_sync(0xFFFFFFFF, block_max0_334, 16);
                float _max_130 = max_noftz(block_max0_334, _shfl_xor_64);
                block_max0_334 = _max_130;
                float _shfl_xor_65 = __shfl_xor_sync(0xFFFFFFFF, block_max1_335, 16);
                float _max_131 = max_noftz(block_max1_335, _shfl_xor_65);
                block_max1_335 = _max_131;
                float _fp8_rt_20;
                uint16_t _e4m3x2_40;
                uint32_t _f16x2_40;
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_40) : "f"(0.0f), "f"(block_max0_334 * 0.16666666666666666f));
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_40) : "h"(_e4m3x2_40));
                uint16_t _fp8_h0_40 = (uint16_t)(_f16x2_40 & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_20) : "h"(_fp8_h0_40));
                float scale0_336 = _fp8_rt_20;
                float _fp8_rt_21;
                uint16_t _e4m3x2_41;
                uint32_t _f16x2_41;
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_41) : "f"(0.0f), "f"(block_max1_335 * 0.16666666666666666f));
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_41) : "h"(_e4m3x2_41));
                uint16_t _fp8_h0_41 = (uint16_t)(_f16x2_41 & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_21) : "h"(_fp8_h0_41));
                float scale1_337 = _fp8_rt_21;
                float inv_scale0_338 = 0.0f;
                float inv_scale1_339 = 0.0f;
                if (scale0_336 != 0.0f) {
                    inv_scale0_338 = 1.0f / scale0_336;
                }
                if (scale1_337 != 0.0f) {
                    inv_scale1_339 = 1.0f / scale1_337;
                }
                quant_pair[0] = value00_330 * inv_scale0_338;
                quant_pair[1] = value10_332 * inv_scale0_338;
                uint32_t _slice_lo_mask_10;
                {
                    int _lim_42 = 2;
                    if (_lim_42 <= 0) { _slice_lo_mask_10 = 0u; }
                    else if (_lim_42 >= 8) { _slice_lo_mask_10 = ((1u << 8) - 1u); }
                    else {
                        asm volatile("{"
                            ".reg .u32 t;\n\t"
                            "shl.b32 t, 1, %1;\n\t"
                            "add.u32 %0, t, -1;\n\t"
                            "}" : "=r"(_slice_lo_mask_10) : "r"(_lim_42));
                    }
                }
                uint32_t _slice_hi_mask_10;
                {
                    int _lim_43 = 8;
                    if (_lim_43 <= 0) { _slice_hi_mask_10 = 0u; }
                    else if (_lim_43 >= 8) { _slice_hi_mask_10 = ((1u << 8) - 1u); }
                    else {
                        asm volatile("{"
                            ".reg .u32 t;\n\t"
                            "shl.b32 t, 1, %1;\n\t"
                            "add.u32 %0, t, -1;\n\t"
                            "}" : "=r"(_slice_hi_mask_10) : "r"(_lim_43));
                    }
                }
                if (!(_slice_lo_mask_10 | ~_slice_hi_mask_10 & (1u << 0))) quant_pair[0] = 0.0f;
                if (!(_slice_lo_mask_10 | ~_slice_hi_mask_10 & (1u << 1))) quant_pair[1] = 0.0f;
                if (!(_slice_lo_mask_10 | ~_slice_hi_mask_10 & (1u << 2))) quant_pair[2] = 0.0f;
                if (!(_slice_lo_mask_10 | ~_slice_hi_mask_10 & (1u << 3))) quant_pair[3] = 0.0f;
                if (!(_slice_lo_mask_10 | ~_slice_hi_mask_10 & (1u << 4))) quant_pair[4] = 0.0f;
                if (!(_slice_lo_mask_10 | ~_slice_hi_mask_10 & (1u << 5))) quant_pair[5] = 0.0f;
                if (!(_slice_lo_mask_10 | ~_slice_hi_mask_10 & (1u << 6))) quant_pair[6] = 0.0f;
                if (!(_slice_lo_mask_10 | ~_slice_hi_mask_10 & (1u << 7))) quant_pair[7] = 0.0f;
                uint32_t _fp4_20[1];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_20[0]) : "f"(quant_pair[0]), "f"(quant_pair[1]), "f"(quant_pair[2]), "f"(quant_pair[3]), "f"(quant_pair[4]), "f"(quant_pair[5]), "f"(quant_pair[6]), "f"(quant_pair[7]));
                quant_pair[0] = value01_331 * inv_scale1_339;
                quant_pair[1] = value11_333 * inv_scale1_339;
                uint32_t _fp4_21[1];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_21[0]) : "f"(quant_pair[0]), "f"(quant_pair[1]), "f"(quant_pair[2]), "f"(quant_pair[3]), "f"(quant_pair[4]), "f"(quant_pair[5]), "f"(quant_pair[6]), "f"(quant_pair[7]));
                if (lane_1 < 4) {
                    int sf_feature_10 = m_tile * 4 + (unsigned int)warp_local;
                    int sf_tile_stride_10 = 2 * (M_out / 64) * 512;
                    int sf_base_10 = n_tile * (unsigned int)sf_tile_stride_10 + (unsigned int)(token0_312 / 128 * (M_out / 64) * 512) + (unsigned int)(sf_feature_10 / 4 * 512) + (unsigned int)(token0_312 % 32 * 16) + (unsigned int)(token0_312 % 128 / 32 * 4) + (unsigned int)(sf_feature_10 % 4);
                    if (token0_312 < valid_rows) {
                        {
                            unsigned short _fp8_pair;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(scale0_336));
                            *(reinterpret_cast<unsigned char*>(SFC + sf_base_10) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                        }
                    }
                    if (token1_313 < valid_rows) {
                        {
                            unsigned short _fp8_pair;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(scale1_337));
                            *(reinterpret_cast<unsigned char*>(SFC + (sf_base_10 + 16)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                        }
                    }
                }
                int smem_flat0_340 = local_token0_310 * 64 + base_row_241;
                int smem_flat1_341 = local_token1_311 * 64 + base_row_241;
                int smem_index0_342 = smem_flat0_340 / 2 ^ smem_flat0_340 / 256 % 2 * 16;
                int smem_index1_343 = smem_flat1_341 / 2 ^ smem_flat1_341 / 256 % 2 * 16;
                epi_staging[warp_group * 2048 + smem_index0_342] = _fp4_20[0];
                epi_staging[warp_group * 2048 + smem_index1_343] = _fp4_21[0];
                int local_token0_344 = lane_1 % 4 * 2 + 24;
                int local_token1_345 = local_token0_344 + 1;
                int token0_346 = token_block_238 * 64 + local_token0_344;
                int token1_347 = token0_346 + 1;
                float _max_132 = max_noftz(_tmem_load_2[12], neg_cl);
                float _min_88 = fminf(_max_132, cl);
                float step_lin00_348 = _min_88;
                float _max_133 = max_noftz(_tmem_load_2[13], neg_cl);
                float _min_89 = fminf(_max_133, cl);
                float step_lin01_349 = _min_89;
                float _max_134 = max_noftz(_tmem_load_3[12], neg_cl);
                float _min_90 = fminf(_max_134, cl);
                float step_lin10_350 = _min_90;
                float _max_135 = max_noftz(_tmem_load_3[13], neg_cl);
                float _min_91 = fminf(_max_135, cl);
                float step_lin11_351 = _min_91;
                float step_x00_352 = _tmem_load_2[14];
                float step_x01_353 = _tmem_load_2[15];
                float step_x10_354 = _tmem_load_3[14];
                float step_x11_355 = _tmem_load_3[15];
                float _exp2_44 = approx_exp2((-(step_x00_352 * sg)) * 1.4426950408889634f);
                float _rcp_44 = approx_rcp(1.0f + _exp2_44);
                float step_sig00_356 = _rcp_44;
                float _exp2_45 = approx_exp2((-(step_x01_353 * sg)) * 1.4426950408889634f);
                float _rcp_45 = approx_rcp(1.0f + _exp2_45);
                float step_sig01_357 = _rcp_45;
                float _exp2_46 = approx_exp2((-(step_x10_354 * sg)) * 1.4426950408889634f);
                float _rcp_46 = approx_rcp(1.0f + _exp2_46);
                float step_sig10_358 = _rcp_46;
                float _exp2_47 = approx_exp2((-(step_x11_355 * sg)) * 1.4426950408889634f);
                float _rcp_47 = approx_rcp(1.0f + _exp2_47);
                float step_sig11_359 = _rcp_47;
                float _min_92 = fminf(step_x00_352 * step_sig00_356, cl);
                float step_g00_360 = _min_92;
                float _min_93 = fminf(step_x01_353 * step_sig01_357, cl);
                float step_g01_361 = _min_93;
                float _min_94 = fminf(step_x10_354 * step_sig10_358, cl);
                float step_g10_362 = _min_94;
                float _min_95 = fminf(step_x11_355 * step_sig11_359, cl);
                float step_g11_363 = _min_95;
                float value00_364 = step_lin00_348 * sc * sg * step_g00_360;
                float value01_365 = step_lin01_349 * sc * sg * step_g01_361;
                float value10_366 = step_lin10_350 * sc * sg * step_g10_362;
                float value11_367 = step_lin11_351 * sc * sg * step_g11_363;
                float _fabs_44 = fabsf(value00_364);
                float _fabs_45 = fabsf(value10_366);
                float _max_136 = max_noftz(_fabs_44, _fabs_45);
                float block_max0_368 = _max_136;
                float _fabs_46 = fabsf(value01_365);
                float _fabs_47 = fabsf(value11_367);
                float _max_137 = max_noftz(_fabs_46, _fabs_47);
                float block_max1_369 = _max_137;
                float _shfl_xor_66 = __shfl_xor_sync(0xFFFFFFFF, block_max0_368, 4);
                float _max_138 = max_noftz(block_max0_368, _shfl_xor_66);
                block_max0_368 = _max_138;
                float _shfl_xor_67 = __shfl_xor_sync(0xFFFFFFFF, block_max1_369, 4);
                float _max_139 = max_noftz(block_max1_369, _shfl_xor_67);
                block_max1_369 = _max_139;
                float _shfl_xor_68 = __shfl_xor_sync(0xFFFFFFFF, block_max0_368, 8);
                float _max_140 = max_noftz(block_max0_368, _shfl_xor_68);
                block_max0_368 = _max_140;
                float _shfl_xor_69 = __shfl_xor_sync(0xFFFFFFFF, block_max1_369, 8);
                float _max_141 = max_noftz(block_max1_369, _shfl_xor_69);
                block_max1_369 = _max_141;
                float _shfl_xor_70 = __shfl_xor_sync(0xFFFFFFFF, block_max0_368, 16);
                float _max_142 = max_noftz(block_max0_368, _shfl_xor_70);
                block_max0_368 = _max_142;
                float _shfl_xor_71 = __shfl_xor_sync(0xFFFFFFFF, block_max1_369, 16);
                float _max_143 = max_noftz(block_max1_369, _shfl_xor_71);
                block_max1_369 = _max_143;
                float _fp8_rt_22;
                uint16_t _e4m3x2_44;
                uint32_t _f16x2_44;
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_44) : "f"(0.0f), "f"(block_max0_368 * 0.16666666666666666f));
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_44) : "h"(_e4m3x2_44));
                uint16_t _fp8_h0_44 = (uint16_t)(_f16x2_44 & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_22) : "h"(_fp8_h0_44));
                float scale0_370 = _fp8_rt_22;
                float _fp8_rt_23;
                uint16_t _e4m3x2_45;
                uint32_t _f16x2_45;
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_45) : "f"(0.0f), "f"(block_max1_369 * 0.16666666666666666f));
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_45) : "h"(_e4m3x2_45));
                uint16_t _fp8_h0_45 = (uint16_t)(_f16x2_45 & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_23) : "h"(_fp8_h0_45));
                float scale1_371 = _fp8_rt_23;
                float inv_scale0_372 = 0.0f;
                float inv_scale1_373 = 0.0f;
                if (scale0_370 != 0.0f) {
                    inv_scale0_372 = 1.0f / scale0_370;
                }
                if (scale1_371 != 0.0f) {
                    inv_scale1_373 = 1.0f / scale1_371;
                }
                quant_pair[0] = value00_364 * inv_scale0_372;
                quant_pair[1] = value10_366 * inv_scale0_372;
                uint32_t _slice_lo_mask_11;
                {
                    int _lim_46 = 2;
                    if (_lim_46 <= 0) { _slice_lo_mask_11 = 0u; }
                    else if (_lim_46 >= 8) { _slice_lo_mask_11 = ((1u << 8) - 1u); }
                    else {
                        asm volatile("{"
                            ".reg .u32 t;\n\t"
                            "shl.b32 t, 1, %1;\n\t"
                            "add.u32 %0, t, -1;\n\t"
                            "}" : "=r"(_slice_lo_mask_11) : "r"(_lim_46));
                    }
                }
                uint32_t _slice_hi_mask_11;
                {
                    int _lim_47 = 8;
                    if (_lim_47 <= 0) { _slice_hi_mask_11 = 0u; }
                    else if (_lim_47 >= 8) { _slice_hi_mask_11 = ((1u << 8) - 1u); }
                    else {
                        asm volatile("{"
                            ".reg .u32 t;\n\t"
                            "shl.b32 t, 1, %1;\n\t"
                            "add.u32 %0, t, -1;\n\t"
                            "}" : "=r"(_slice_hi_mask_11) : "r"(_lim_47));
                    }
                }
                if (!(_slice_lo_mask_11 | ~_slice_hi_mask_11 & (1u << 0))) quant_pair[0] = 0.0f;
                if (!(_slice_lo_mask_11 | ~_slice_hi_mask_11 & (1u << 1))) quant_pair[1] = 0.0f;
                if (!(_slice_lo_mask_11 | ~_slice_hi_mask_11 & (1u << 2))) quant_pair[2] = 0.0f;
                if (!(_slice_lo_mask_11 | ~_slice_hi_mask_11 & (1u << 3))) quant_pair[3] = 0.0f;
                if (!(_slice_lo_mask_11 | ~_slice_hi_mask_11 & (1u << 4))) quant_pair[4] = 0.0f;
                if (!(_slice_lo_mask_11 | ~_slice_hi_mask_11 & (1u << 5))) quant_pair[5] = 0.0f;
                if (!(_slice_lo_mask_11 | ~_slice_hi_mask_11 & (1u << 6))) quant_pair[6] = 0.0f;
                if (!(_slice_lo_mask_11 | ~_slice_hi_mask_11 & (1u << 7))) quant_pair[7] = 0.0f;
                uint32_t _fp4_22[1];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_22[0]) : "f"(quant_pair[0]), "f"(quant_pair[1]), "f"(quant_pair[2]), "f"(quant_pair[3]), "f"(quant_pair[4]), "f"(quant_pair[5]), "f"(quant_pair[6]), "f"(quant_pair[7]));
                quant_pair[0] = value01_365 * inv_scale1_373;
                quant_pair[1] = value11_367 * inv_scale1_373;
                uint32_t _fp4_23[1];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_23[0]) : "f"(quant_pair[0]), "f"(quant_pair[1]), "f"(quant_pair[2]), "f"(quant_pair[3]), "f"(quant_pair[4]), "f"(quant_pair[5]), "f"(quant_pair[6]), "f"(quant_pair[7]));
                if (lane_1 < 4) {
                    int sf_feature_11 = m_tile * 4 + (unsigned int)warp_local;
                    int sf_tile_stride_11 = 2 * (M_out / 64) * 512;
                    int sf_base_11 = n_tile * (unsigned int)sf_tile_stride_11 + (unsigned int)(token0_346 / 128 * (M_out / 64) * 512) + (unsigned int)(sf_feature_11 / 4 * 512) + (unsigned int)(token0_346 % 32 * 16) + (unsigned int)(token0_346 % 128 / 32 * 4) + (unsigned int)(sf_feature_11 % 4);
                    if (token0_346 < valid_rows) {
                        {
                            unsigned short _fp8_pair;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(scale0_370));
                            *(reinterpret_cast<unsigned char*>(SFC + sf_base_11) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                        }
                    }
                    if (token1_347 < valid_rows) {
                        {
                            unsigned short _fp8_pair;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(scale1_371));
                            *(reinterpret_cast<unsigned char*>(SFC + (sf_base_11 + 16)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                        }
                    }
                }
                int smem_flat0_374 = local_token0_344 * 64 + base_row_241;
                int smem_flat1_375 = local_token1_345 * 64 + base_row_241;
                int smem_index0_376 = smem_flat0_374 / 2 ^ smem_flat0_374 / 256 % 2 * 16;
                int smem_index1_377 = smem_flat1_375 / 2 ^ smem_flat1_375 / 256 % 2 * 16;
                epi_staging[warp_group * 2048 + smem_index0_376] = _fp4_22[0];
                epi_staging[warp_group * 2048 + smem_index1_377] = _fp4_23[0];
                int local_token0_378 = lane_1 % 4 * 2 + 32;
                int local_token1_379 = local_token0_378 + 1;
                int token0_380 = token_block_238 * 64 + local_token0_378;
                int token1_381 = token0_380 + 1;
                float _max_144 = max_noftz(_tmem_load_2[16], neg_cl);
                float _min_96 = fminf(_max_144, cl);
                float step_lin00_382 = _min_96;
                float _max_145 = max_noftz(_tmem_load_2[17], neg_cl);
                float _min_97 = fminf(_max_145, cl);
                float step_lin01_383 = _min_97;
                float _max_146 = max_noftz(_tmem_load_3[16], neg_cl);
                float _min_98 = fminf(_max_146, cl);
                float step_lin10_384 = _min_98;
                float _max_147 = max_noftz(_tmem_load_3[17], neg_cl);
                float _min_99 = fminf(_max_147, cl);
                float step_lin11_385 = _min_99;
                float step_x00_386 = _tmem_load_2[18];
                float step_x01_387 = _tmem_load_2[19];
                float step_x10_388 = _tmem_load_3[18];
                float step_x11_389 = _tmem_load_3[19];
                float _exp2_48 = approx_exp2((-(step_x00_386 * sg)) * 1.4426950408889634f);
                float _rcp_48 = approx_rcp(1.0f + _exp2_48);
                float step_sig00_390 = _rcp_48;
                float _exp2_49 = approx_exp2((-(step_x01_387 * sg)) * 1.4426950408889634f);
                float _rcp_49 = approx_rcp(1.0f + _exp2_49);
                float step_sig01_391 = _rcp_49;
                float _exp2_50 = approx_exp2((-(step_x10_388 * sg)) * 1.4426950408889634f);
                float _rcp_50 = approx_rcp(1.0f + _exp2_50);
                float step_sig10_392 = _rcp_50;
                float _exp2_51 = approx_exp2((-(step_x11_389 * sg)) * 1.4426950408889634f);
                float _rcp_51 = approx_rcp(1.0f + _exp2_51);
                float step_sig11_393 = _rcp_51;
                float _min_100 = fminf(step_x00_386 * step_sig00_390, cl);
                float step_g00_394 = _min_100;
                float _min_101 = fminf(step_x01_387 * step_sig01_391, cl);
                float step_g01_395 = _min_101;
                float _min_102 = fminf(step_x10_388 * step_sig10_392, cl);
                float step_g10_396 = _min_102;
                float _min_103 = fminf(step_x11_389 * step_sig11_393, cl);
                float step_g11_397 = _min_103;
                float value00_398 = step_lin00_382 * sc * sg * step_g00_394;
                float value01_399 = step_lin01_383 * sc * sg * step_g01_395;
                float value10_400 = step_lin10_384 * sc * sg * step_g10_396;
                float value11_401 = step_lin11_385 * sc * sg * step_g11_397;
                float _fabs_48 = fabsf(value00_398);
                float _fabs_49 = fabsf(value10_400);
                float _max_148 = max_noftz(_fabs_48, _fabs_49);
                float block_max0_402 = _max_148;
                float _fabs_50 = fabsf(value01_399);
                float _fabs_51 = fabsf(value11_401);
                float _max_149 = max_noftz(_fabs_50, _fabs_51);
                float block_max1_403 = _max_149;
                float _shfl_xor_72 = __shfl_xor_sync(0xFFFFFFFF, block_max0_402, 4);
                float _max_150 = max_noftz(block_max0_402, _shfl_xor_72);
                block_max0_402 = _max_150;
                float _shfl_xor_73 = __shfl_xor_sync(0xFFFFFFFF, block_max1_403, 4);
                float _max_151 = max_noftz(block_max1_403, _shfl_xor_73);
                block_max1_403 = _max_151;
                float _shfl_xor_74 = __shfl_xor_sync(0xFFFFFFFF, block_max0_402, 8);
                float _max_152 = max_noftz(block_max0_402, _shfl_xor_74);
                block_max0_402 = _max_152;
                float _shfl_xor_75 = __shfl_xor_sync(0xFFFFFFFF, block_max1_403, 8);
                float _max_153 = max_noftz(block_max1_403, _shfl_xor_75);
                block_max1_403 = _max_153;
                float _shfl_xor_76 = __shfl_xor_sync(0xFFFFFFFF, block_max0_402, 16);
                float _max_154 = max_noftz(block_max0_402, _shfl_xor_76);
                block_max0_402 = _max_154;
                float _shfl_xor_77 = __shfl_xor_sync(0xFFFFFFFF, block_max1_403, 16);
                float _max_155 = max_noftz(block_max1_403, _shfl_xor_77);
                block_max1_403 = _max_155;
                float _fp8_rt_24;
                uint16_t _e4m3x2_48;
                uint32_t _f16x2_48;
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_48) : "f"(0.0f), "f"(block_max0_402 * 0.16666666666666666f));
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_48) : "h"(_e4m3x2_48));
                uint16_t _fp8_h0_48 = (uint16_t)(_f16x2_48 & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_24) : "h"(_fp8_h0_48));
                float scale0_404 = _fp8_rt_24;
                float _fp8_rt_25;
                uint16_t _e4m3x2_49;
                uint32_t _f16x2_49;
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_49) : "f"(0.0f), "f"(block_max1_403 * 0.16666666666666666f));
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_49) : "h"(_e4m3x2_49));
                uint16_t _fp8_h0_49 = (uint16_t)(_f16x2_49 & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_25) : "h"(_fp8_h0_49));
                float scale1_405 = _fp8_rt_25;
                float inv_scale0_406 = 0.0f;
                float inv_scale1_407 = 0.0f;
                if (scale0_404 != 0.0f) {
                    inv_scale0_406 = 1.0f / scale0_404;
                }
                if (scale1_405 != 0.0f) {
                    inv_scale1_407 = 1.0f / scale1_405;
                }
                quant_pair[0] = value00_398 * inv_scale0_406;
                quant_pair[1] = value10_400 * inv_scale0_406;
                uint32_t _slice_lo_mask_12;
                {
                    int _lim_50 = 2;
                    if (_lim_50 <= 0) { _slice_lo_mask_12 = 0u; }
                    else if (_lim_50 >= 8) { _slice_lo_mask_12 = ((1u << 8) - 1u); }
                    else {
                        asm volatile("{"
                            ".reg .u32 t;\n\t"
                            "shl.b32 t, 1, %1;\n\t"
                            "add.u32 %0, t, -1;\n\t"
                            "}" : "=r"(_slice_lo_mask_12) : "r"(_lim_50));
                    }
                }
                uint32_t _slice_hi_mask_12;
                {
                    int _lim_51 = 8;
                    if (_lim_51 <= 0) { _slice_hi_mask_12 = 0u; }
                    else if (_lim_51 >= 8) { _slice_hi_mask_12 = ((1u << 8) - 1u); }
                    else {
                        asm volatile("{"
                            ".reg .u32 t;\n\t"
                            "shl.b32 t, 1, %1;\n\t"
                            "add.u32 %0, t, -1;\n\t"
                            "}" : "=r"(_slice_hi_mask_12) : "r"(_lim_51));
                    }
                }
                if (!(_slice_lo_mask_12 | ~_slice_hi_mask_12 & (1u << 0))) quant_pair[0] = 0.0f;
                if (!(_slice_lo_mask_12 | ~_slice_hi_mask_12 & (1u << 1))) quant_pair[1] = 0.0f;
                if (!(_slice_lo_mask_12 | ~_slice_hi_mask_12 & (1u << 2))) quant_pair[2] = 0.0f;
                if (!(_slice_lo_mask_12 | ~_slice_hi_mask_12 & (1u << 3))) quant_pair[3] = 0.0f;
                if (!(_slice_lo_mask_12 | ~_slice_hi_mask_12 & (1u << 4))) quant_pair[4] = 0.0f;
                if (!(_slice_lo_mask_12 | ~_slice_hi_mask_12 & (1u << 5))) quant_pair[5] = 0.0f;
                if (!(_slice_lo_mask_12 | ~_slice_hi_mask_12 & (1u << 6))) quant_pair[6] = 0.0f;
                if (!(_slice_lo_mask_12 | ~_slice_hi_mask_12 & (1u << 7))) quant_pair[7] = 0.0f;
                uint32_t _fp4_24[1];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_24[0]) : "f"(quant_pair[0]), "f"(quant_pair[1]), "f"(quant_pair[2]), "f"(quant_pair[3]), "f"(quant_pair[4]), "f"(quant_pair[5]), "f"(quant_pair[6]), "f"(quant_pair[7]));
                quant_pair[0] = value01_399 * inv_scale1_407;
                quant_pair[1] = value11_401 * inv_scale1_407;
                uint32_t _fp4_25[1];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_25[0]) : "f"(quant_pair[0]), "f"(quant_pair[1]), "f"(quant_pair[2]), "f"(quant_pair[3]), "f"(quant_pair[4]), "f"(quant_pair[5]), "f"(quant_pair[6]), "f"(quant_pair[7]));
                if (lane_1 < 4) {
                    int sf_feature_12 = m_tile * 4 + (unsigned int)warp_local;
                    int sf_tile_stride_12 = 2 * (M_out / 64) * 512;
                    int sf_base_12 = n_tile * (unsigned int)sf_tile_stride_12 + (unsigned int)(token0_380 / 128 * (M_out / 64) * 512) + (unsigned int)(sf_feature_12 / 4 * 512) + (unsigned int)(token0_380 % 32 * 16) + (unsigned int)(token0_380 % 128 / 32 * 4) + (unsigned int)(sf_feature_12 % 4);
                    if (token0_380 < valid_rows) {
                        {
                            unsigned short _fp8_pair;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(scale0_404));
                            *(reinterpret_cast<unsigned char*>(SFC + sf_base_12) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                        }
                    }
                    if (token1_381 < valid_rows) {
                        {
                            unsigned short _fp8_pair;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(scale1_405));
                            *(reinterpret_cast<unsigned char*>(SFC + (sf_base_12 + 16)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                        }
                    }
                }
                int smem_flat0_408 = local_token0_378 * 64 + base_row_241;
                int smem_flat1_409 = local_token1_379 * 64 + base_row_241;
                int smem_index0_410 = smem_flat0_408 / 2 ^ smem_flat0_408 / 256 % 2 * 16;
                int smem_index1_411 = smem_flat1_409 / 2 ^ smem_flat1_409 / 256 % 2 * 16;
                epi_staging[warp_group * 2048 + smem_index0_410] = _fp4_24[0];
                epi_staging[warp_group * 2048 + smem_index1_411] = _fp4_25[0];
                int local_token0_412 = lane_1 % 4 * 2 + 40;
                int local_token1_413 = local_token0_412 + 1;
                int token0_414 = token_block_238 * 64 + local_token0_412;
                int token1_415 = token0_414 + 1;
                float _max_156 = max_noftz(_tmem_load_2[20], neg_cl);
                float _min_104 = fminf(_max_156, cl);
                float step_lin00_416 = _min_104;
                float _max_157 = max_noftz(_tmem_load_2[21], neg_cl);
                float _min_105 = fminf(_max_157, cl);
                float step_lin01_417 = _min_105;
                float _max_158 = max_noftz(_tmem_load_3[20], neg_cl);
                float _min_106 = fminf(_max_158, cl);
                float step_lin10_418 = _min_106;
                float _max_159 = max_noftz(_tmem_load_3[21], neg_cl);
                float _min_107 = fminf(_max_159, cl);
                float step_lin11_419 = _min_107;
                float step_x00_420 = _tmem_load_2[22];
                float step_x01_421 = _tmem_load_2[23];
                float step_x10_422 = _tmem_load_3[22];
                float step_x11_423 = _tmem_load_3[23];
                float _exp2_52 = approx_exp2((-(step_x00_420 * sg)) * 1.4426950408889634f);
                float _rcp_52 = approx_rcp(1.0f + _exp2_52);
                float step_sig00_424 = _rcp_52;
                float _exp2_53 = approx_exp2((-(step_x01_421 * sg)) * 1.4426950408889634f);
                float _rcp_53 = approx_rcp(1.0f + _exp2_53);
                float step_sig01_425 = _rcp_53;
                float _exp2_54 = approx_exp2((-(step_x10_422 * sg)) * 1.4426950408889634f);
                float _rcp_54 = approx_rcp(1.0f + _exp2_54);
                float step_sig10_426 = _rcp_54;
                float _exp2_55 = approx_exp2((-(step_x11_423 * sg)) * 1.4426950408889634f);
                float _rcp_55 = approx_rcp(1.0f + _exp2_55);
                float step_sig11_427 = _rcp_55;
                float _min_108 = fminf(step_x00_420 * step_sig00_424, cl);
                float step_g00_428 = _min_108;
                float _min_109 = fminf(step_x01_421 * step_sig01_425, cl);
                float step_g01_429 = _min_109;
                float _min_110 = fminf(step_x10_422 * step_sig10_426, cl);
                float step_g10_430 = _min_110;
                float _min_111 = fminf(step_x11_423 * step_sig11_427, cl);
                float step_g11_431 = _min_111;
                float value00_432 = step_lin00_416 * sc * sg * step_g00_428;
                float value01_433 = step_lin01_417 * sc * sg * step_g01_429;
                float value10_434 = step_lin10_418 * sc * sg * step_g10_430;
                float value11_435 = step_lin11_419 * sc * sg * step_g11_431;
                float _fabs_52 = fabsf(value00_432);
                float _fabs_53 = fabsf(value10_434);
                float _max_160 = max_noftz(_fabs_52, _fabs_53);
                float block_max0_436 = _max_160;
                float _fabs_54 = fabsf(value01_433);
                float _fabs_55 = fabsf(value11_435);
                float _max_161 = max_noftz(_fabs_54, _fabs_55);
                float block_max1_437 = _max_161;
                float _shfl_xor_78 = __shfl_xor_sync(0xFFFFFFFF, block_max0_436, 4);
                float _max_162 = max_noftz(block_max0_436, _shfl_xor_78);
                block_max0_436 = _max_162;
                float _shfl_xor_79 = __shfl_xor_sync(0xFFFFFFFF, block_max1_437, 4);
                float _max_163 = max_noftz(block_max1_437, _shfl_xor_79);
                block_max1_437 = _max_163;
                float _shfl_xor_80 = __shfl_xor_sync(0xFFFFFFFF, block_max0_436, 8);
                float _max_164 = max_noftz(block_max0_436, _shfl_xor_80);
                block_max0_436 = _max_164;
                float _shfl_xor_81 = __shfl_xor_sync(0xFFFFFFFF, block_max1_437, 8);
                float _max_165 = max_noftz(block_max1_437, _shfl_xor_81);
                block_max1_437 = _max_165;
                float _shfl_xor_82 = __shfl_xor_sync(0xFFFFFFFF, block_max0_436, 16);
                float _max_166 = max_noftz(block_max0_436, _shfl_xor_82);
                block_max0_436 = _max_166;
                float _shfl_xor_83 = __shfl_xor_sync(0xFFFFFFFF, block_max1_437, 16);
                float _max_167 = max_noftz(block_max1_437, _shfl_xor_83);
                block_max1_437 = _max_167;
                float _fp8_rt_26;
                uint16_t _e4m3x2_52;
                uint32_t _f16x2_52;
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_52) : "f"(0.0f), "f"(block_max0_436 * 0.16666666666666666f));
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_52) : "h"(_e4m3x2_52));
                uint16_t _fp8_h0_52 = (uint16_t)(_f16x2_52 & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_26) : "h"(_fp8_h0_52));
                float scale0_438 = _fp8_rt_26;
                float _fp8_rt_27;
                uint16_t _e4m3x2_53;
                uint32_t _f16x2_53;
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_53) : "f"(0.0f), "f"(block_max1_437 * 0.16666666666666666f));
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_53) : "h"(_e4m3x2_53));
                uint16_t _fp8_h0_53 = (uint16_t)(_f16x2_53 & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_27) : "h"(_fp8_h0_53));
                float scale1_439 = _fp8_rt_27;
                float inv_scale0_440 = 0.0f;
                float inv_scale1_441 = 0.0f;
                if (scale0_438 != 0.0f) {
                    inv_scale0_440 = 1.0f / scale0_438;
                }
                if (scale1_439 != 0.0f) {
                    inv_scale1_441 = 1.0f / scale1_439;
                }
                quant_pair[0] = value00_432 * inv_scale0_440;
                quant_pair[1] = value10_434 * inv_scale0_440;
                uint32_t _slice_lo_mask_13;
                {
                    int _lim_54 = 2;
                    if (_lim_54 <= 0) { _slice_lo_mask_13 = 0u; }
                    else if (_lim_54 >= 8) { _slice_lo_mask_13 = ((1u << 8) - 1u); }
                    else {
                        asm volatile("{"
                            ".reg .u32 t;\n\t"
                            "shl.b32 t, 1, %1;\n\t"
                            "add.u32 %0, t, -1;\n\t"
                            "}" : "=r"(_slice_lo_mask_13) : "r"(_lim_54));
                    }
                }
                uint32_t _slice_hi_mask_13;
                {
                    int _lim_55 = 8;
                    if (_lim_55 <= 0) { _slice_hi_mask_13 = 0u; }
                    else if (_lim_55 >= 8) { _slice_hi_mask_13 = ((1u << 8) - 1u); }
                    else {
                        asm volatile("{"
                            ".reg .u32 t;\n\t"
                            "shl.b32 t, 1, %1;\n\t"
                            "add.u32 %0, t, -1;\n\t"
                            "}" : "=r"(_slice_hi_mask_13) : "r"(_lim_55));
                    }
                }
                if (!(_slice_lo_mask_13 | ~_slice_hi_mask_13 & (1u << 0))) quant_pair[0] = 0.0f;
                if (!(_slice_lo_mask_13 | ~_slice_hi_mask_13 & (1u << 1))) quant_pair[1] = 0.0f;
                if (!(_slice_lo_mask_13 | ~_slice_hi_mask_13 & (1u << 2))) quant_pair[2] = 0.0f;
                if (!(_slice_lo_mask_13 | ~_slice_hi_mask_13 & (1u << 3))) quant_pair[3] = 0.0f;
                if (!(_slice_lo_mask_13 | ~_slice_hi_mask_13 & (1u << 4))) quant_pair[4] = 0.0f;
                if (!(_slice_lo_mask_13 | ~_slice_hi_mask_13 & (1u << 5))) quant_pair[5] = 0.0f;
                if (!(_slice_lo_mask_13 | ~_slice_hi_mask_13 & (1u << 6))) quant_pair[6] = 0.0f;
                if (!(_slice_lo_mask_13 | ~_slice_hi_mask_13 & (1u << 7))) quant_pair[7] = 0.0f;
                uint32_t _fp4_26[1];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_26[0]) : "f"(quant_pair[0]), "f"(quant_pair[1]), "f"(quant_pair[2]), "f"(quant_pair[3]), "f"(quant_pair[4]), "f"(quant_pair[5]), "f"(quant_pair[6]), "f"(quant_pair[7]));
                quant_pair[0] = value01_433 * inv_scale1_441;
                quant_pair[1] = value11_435 * inv_scale1_441;
                uint32_t _fp4_27[1];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_27[0]) : "f"(quant_pair[0]), "f"(quant_pair[1]), "f"(quant_pair[2]), "f"(quant_pair[3]), "f"(quant_pair[4]), "f"(quant_pair[5]), "f"(quant_pair[6]), "f"(quant_pair[7]));
                if (lane_1 < 4) {
                    int sf_feature_13 = m_tile * 4 + (unsigned int)warp_local;
                    int sf_tile_stride_13 = 2 * (M_out / 64) * 512;
                    int sf_base_13 = n_tile * (unsigned int)sf_tile_stride_13 + (unsigned int)(token0_414 / 128 * (M_out / 64) * 512) + (unsigned int)(sf_feature_13 / 4 * 512) + (unsigned int)(token0_414 % 32 * 16) + (unsigned int)(token0_414 % 128 / 32 * 4) + (unsigned int)(sf_feature_13 % 4);
                    if (token0_414 < valid_rows) {
                        {
                            unsigned short _fp8_pair;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(scale0_438));
                            *(reinterpret_cast<unsigned char*>(SFC + sf_base_13) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                        }
                    }
                    if (token1_415 < valid_rows) {
                        {
                            unsigned short _fp8_pair;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(scale1_439));
                            *(reinterpret_cast<unsigned char*>(SFC + (sf_base_13 + 16)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                        }
                    }
                }
                int smem_flat0_442 = local_token0_412 * 64 + base_row_241;
                int smem_flat1_443 = local_token1_413 * 64 + base_row_241;
                int smem_index0_444 = smem_flat0_442 / 2 ^ smem_flat0_442 / 256 % 2 * 16;
                int smem_index1_445 = smem_flat1_443 / 2 ^ smem_flat1_443 / 256 % 2 * 16;
                epi_staging[warp_group * 2048 + smem_index0_444] = _fp4_26[0];
                epi_staging[warp_group * 2048 + smem_index1_445] = _fp4_27[0];
                int local_token0_446 = lane_1 % 4 * 2 + 48;
                int local_token1_447 = local_token0_446 + 1;
                int token0_448 = token_block_238 * 64 + local_token0_446;
                int token1_449 = token0_448 + 1;
                float _max_168 = max_noftz(_tmem_load_2[24], neg_cl);
                float _min_112 = fminf(_max_168, cl);
                float step_lin00_450 = _min_112;
                float _max_169 = max_noftz(_tmem_load_2[25], neg_cl);
                float _min_113 = fminf(_max_169, cl);
                float step_lin01_451 = _min_113;
                float _max_170 = max_noftz(_tmem_load_3[24], neg_cl);
                float _min_114 = fminf(_max_170, cl);
                float step_lin10_452 = _min_114;
                float _max_171 = max_noftz(_tmem_load_3[25], neg_cl);
                float _min_115 = fminf(_max_171, cl);
                float step_lin11_453 = _min_115;
                float step_x00_454 = _tmem_load_2[26];
                float step_x01_455 = _tmem_load_2[27];
                float step_x10_456 = _tmem_load_3[26];
                float step_x11_457 = _tmem_load_3[27];
                float _exp2_56 = approx_exp2((-(step_x00_454 * sg)) * 1.4426950408889634f);
                float _rcp_56 = approx_rcp(1.0f + _exp2_56);
                float step_sig00_458 = _rcp_56;
                float _exp2_57 = approx_exp2((-(step_x01_455 * sg)) * 1.4426950408889634f);
                float _rcp_57 = approx_rcp(1.0f + _exp2_57);
                float step_sig01_459 = _rcp_57;
                float _exp2_58 = approx_exp2((-(step_x10_456 * sg)) * 1.4426950408889634f);
                float _rcp_58 = approx_rcp(1.0f + _exp2_58);
                float step_sig10_460 = _rcp_58;
                float _exp2_59 = approx_exp2((-(step_x11_457 * sg)) * 1.4426950408889634f);
                float _rcp_59 = approx_rcp(1.0f + _exp2_59);
                float step_sig11_461 = _rcp_59;
                float _min_116 = fminf(step_x00_454 * step_sig00_458, cl);
                float step_g00_462 = _min_116;
                float _min_117 = fminf(step_x01_455 * step_sig01_459, cl);
                float step_g01_463 = _min_117;
                float _min_118 = fminf(step_x10_456 * step_sig10_460, cl);
                float step_g10_464 = _min_118;
                float _min_119 = fminf(step_x11_457 * step_sig11_461, cl);
                float step_g11_465 = _min_119;
                float value00_466 = step_lin00_450 * sc * sg * step_g00_462;
                float value01_467 = step_lin01_451 * sc * sg * step_g01_463;
                float value10_468 = step_lin10_452 * sc * sg * step_g10_464;
                float value11_469 = step_lin11_453 * sc * sg * step_g11_465;
                float _fabs_56 = fabsf(value00_466);
                float _fabs_57 = fabsf(value10_468);
                float _max_172 = max_noftz(_fabs_56, _fabs_57);
                float block_max0_470 = _max_172;
                float _fabs_58 = fabsf(value01_467);
                float _fabs_59 = fabsf(value11_469);
                float _max_173 = max_noftz(_fabs_58, _fabs_59);
                float block_max1_471 = _max_173;
                float _shfl_xor_84 = __shfl_xor_sync(0xFFFFFFFF, block_max0_470, 4);
                float _max_174 = max_noftz(block_max0_470, _shfl_xor_84);
                block_max0_470 = _max_174;
                float _shfl_xor_85 = __shfl_xor_sync(0xFFFFFFFF, block_max1_471, 4);
                float _max_175 = max_noftz(block_max1_471, _shfl_xor_85);
                block_max1_471 = _max_175;
                float _shfl_xor_86 = __shfl_xor_sync(0xFFFFFFFF, block_max0_470, 8);
                float _max_176 = max_noftz(block_max0_470, _shfl_xor_86);
                block_max0_470 = _max_176;
                float _shfl_xor_87 = __shfl_xor_sync(0xFFFFFFFF, block_max1_471, 8);
                float _max_177 = max_noftz(block_max1_471, _shfl_xor_87);
                block_max1_471 = _max_177;
                float _shfl_xor_88 = __shfl_xor_sync(0xFFFFFFFF, block_max0_470, 16);
                float _max_178 = max_noftz(block_max0_470, _shfl_xor_88);
                block_max0_470 = _max_178;
                float _shfl_xor_89 = __shfl_xor_sync(0xFFFFFFFF, block_max1_471, 16);
                float _max_179 = max_noftz(block_max1_471, _shfl_xor_89);
                block_max1_471 = _max_179;
                float _fp8_rt_28;
                uint16_t _e4m3x2_56;
                uint32_t _f16x2_56;
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_56) : "f"(0.0f), "f"(block_max0_470 * 0.16666666666666666f));
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_56) : "h"(_e4m3x2_56));
                uint16_t _fp8_h0_56 = (uint16_t)(_f16x2_56 & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_28) : "h"(_fp8_h0_56));
                float scale0_472 = _fp8_rt_28;
                float _fp8_rt_29;
                uint16_t _e4m3x2_57;
                uint32_t _f16x2_57;
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_57) : "f"(0.0f), "f"(block_max1_471 * 0.16666666666666666f));
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_57) : "h"(_e4m3x2_57));
                uint16_t _fp8_h0_57 = (uint16_t)(_f16x2_57 & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_29) : "h"(_fp8_h0_57));
                float scale1_473 = _fp8_rt_29;
                float inv_scale0_474 = 0.0f;
                float inv_scale1_475 = 0.0f;
                if (scale0_472 != 0.0f) {
                    inv_scale0_474 = 1.0f / scale0_472;
                }
                if (scale1_473 != 0.0f) {
                    inv_scale1_475 = 1.0f / scale1_473;
                }
                quant_pair[0] = value00_466 * inv_scale0_474;
                quant_pair[1] = value10_468 * inv_scale0_474;
                uint32_t _slice_lo_mask_14;
                {
                    int _lim_58 = 2;
                    if (_lim_58 <= 0) { _slice_lo_mask_14 = 0u; }
                    else if (_lim_58 >= 8) { _slice_lo_mask_14 = ((1u << 8) - 1u); }
                    else {
                        asm volatile("{"
                            ".reg .u32 t;\n\t"
                            "shl.b32 t, 1, %1;\n\t"
                            "add.u32 %0, t, -1;\n\t"
                            "}" : "=r"(_slice_lo_mask_14) : "r"(_lim_58));
                    }
                }
                uint32_t _slice_hi_mask_14;
                {
                    int _lim_59 = 8;
                    if (_lim_59 <= 0) { _slice_hi_mask_14 = 0u; }
                    else if (_lim_59 >= 8) { _slice_hi_mask_14 = ((1u << 8) - 1u); }
                    else {
                        asm volatile("{"
                            ".reg .u32 t;\n\t"
                            "shl.b32 t, 1, %1;\n\t"
                            "add.u32 %0, t, -1;\n\t"
                            "}" : "=r"(_slice_hi_mask_14) : "r"(_lim_59));
                    }
                }
                if (!(_slice_lo_mask_14 | ~_slice_hi_mask_14 & (1u << 0))) quant_pair[0] = 0.0f;
                if (!(_slice_lo_mask_14 | ~_slice_hi_mask_14 & (1u << 1))) quant_pair[1] = 0.0f;
                if (!(_slice_lo_mask_14 | ~_slice_hi_mask_14 & (1u << 2))) quant_pair[2] = 0.0f;
                if (!(_slice_lo_mask_14 | ~_slice_hi_mask_14 & (1u << 3))) quant_pair[3] = 0.0f;
                if (!(_slice_lo_mask_14 | ~_slice_hi_mask_14 & (1u << 4))) quant_pair[4] = 0.0f;
                if (!(_slice_lo_mask_14 | ~_slice_hi_mask_14 & (1u << 5))) quant_pair[5] = 0.0f;
                if (!(_slice_lo_mask_14 | ~_slice_hi_mask_14 & (1u << 6))) quant_pair[6] = 0.0f;
                if (!(_slice_lo_mask_14 | ~_slice_hi_mask_14 & (1u << 7))) quant_pair[7] = 0.0f;
                uint32_t _fp4_28[1];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_28[0]) : "f"(quant_pair[0]), "f"(quant_pair[1]), "f"(quant_pair[2]), "f"(quant_pair[3]), "f"(quant_pair[4]), "f"(quant_pair[5]), "f"(quant_pair[6]), "f"(quant_pair[7]));
                quant_pair[0] = value01_467 * inv_scale1_475;
                quant_pair[1] = value11_469 * inv_scale1_475;
                uint32_t _fp4_29[1];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_29[0]) : "f"(quant_pair[0]), "f"(quant_pair[1]), "f"(quant_pair[2]), "f"(quant_pair[3]), "f"(quant_pair[4]), "f"(quant_pair[5]), "f"(quant_pair[6]), "f"(quant_pair[7]));
                if (lane_1 < 4) {
                    int sf_feature_14 = m_tile * 4 + (unsigned int)warp_local;
                    int sf_tile_stride_14 = 2 * (M_out / 64) * 512;
                    int sf_base_14 = n_tile * (unsigned int)sf_tile_stride_14 + (unsigned int)(token0_448 / 128 * (M_out / 64) * 512) + (unsigned int)(sf_feature_14 / 4 * 512) + (unsigned int)(token0_448 % 32 * 16) + (unsigned int)(token0_448 % 128 / 32 * 4) + (unsigned int)(sf_feature_14 % 4);
                    if (token0_448 < valid_rows) {
                        {
                            unsigned short _fp8_pair;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(scale0_472));
                            *(reinterpret_cast<unsigned char*>(SFC + sf_base_14) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                        }
                    }
                    if (token1_449 < valid_rows) {
                        {
                            unsigned short _fp8_pair;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(scale1_473));
                            *(reinterpret_cast<unsigned char*>(SFC + (sf_base_14 + 16)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                        }
                    }
                }
                int smem_flat0_476 = local_token0_446 * 64 + base_row_241;
                int smem_flat1_477 = local_token1_447 * 64 + base_row_241;
                int smem_index0_478 = smem_flat0_476 / 2 ^ smem_flat0_476 / 256 % 2 * 16;
                int smem_index1_479 = smem_flat1_477 / 2 ^ smem_flat1_477 / 256 % 2 * 16;
                epi_staging[warp_group * 2048 + smem_index0_478] = _fp4_28[0];
                epi_staging[warp_group * 2048 + smem_index1_479] = _fp4_29[0];
                int local_token0_480 = lane_1 % 4 * 2 + 56;
                int local_token1_481 = local_token0_480 + 1;
                int token0_482 = token_block_238 * 64 + local_token0_480;
                int token1_483 = token0_482 + 1;
                float _max_180 = max_noftz(_tmem_load_2[28], neg_cl);
                float _min_120 = fminf(_max_180, cl);
                float step_lin00_484 = _min_120;
                float _max_181 = max_noftz(_tmem_load_2[29], neg_cl);
                float _min_121 = fminf(_max_181, cl);
                float step_lin01_485 = _min_121;
                float _max_182 = max_noftz(_tmem_load_3[28], neg_cl);
                float _min_122 = fminf(_max_182, cl);
                float step_lin10_486 = _min_122;
                float _max_183 = max_noftz(_tmem_load_3[29], neg_cl);
                float _min_123 = fminf(_max_183, cl);
                float step_lin11_487 = _min_123;
                float step_x00_488 = _tmem_load_2[30];
                float step_x01_489 = _tmem_load_2[31];
                float step_x10_490 = _tmem_load_3[30];
                float step_x11_491 = _tmem_load_3[31];
                float _exp2_60 = approx_exp2((-(step_x00_488 * sg)) * 1.4426950408889634f);
                float _rcp_60 = approx_rcp(1.0f + _exp2_60);
                float step_sig00_492 = _rcp_60;
                float _exp2_61 = approx_exp2((-(step_x01_489 * sg)) * 1.4426950408889634f);
                float _rcp_61 = approx_rcp(1.0f + _exp2_61);
                float step_sig01_493 = _rcp_61;
                float _exp2_62 = approx_exp2((-(step_x10_490 * sg)) * 1.4426950408889634f);
                float _rcp_62 = approx_rcp(1.0f + _exp2_62);
                float step_sig10_494 = _rcp_62;
                float _exp2_63 = approx_exp2((-(step_x11_491 * sg)) * 1.4426950408889634f);
                float _rcp_63 = approx_rcp(1.0f + _exp2_63);
                float step_sig11_495 = _rcp_63;
                float _min_124 = fminf(step_x00_488 * step_sig00_492, cl);
                float step_g00_496 = _min_124;
                float _min_125 = fminf(step_x01_489 * step_sig01_493, cl);
                float step_g01_497 = _min_125;
                float _min_126 = fminf(step_x10_490 * step_sig10_494, cl);
                float step_g10_498 = _min_126;
                float _min_127 = fminf(step_x11_491 * step_sig11_495, cl);
                float step_g11_499 = _min_127;
                float value00_500 = step_lin00_484 * sc * sg * step_g00_496;
                float value01_501 = step_lin01_485 * sc * sg * step_g01_497;
                float value10_502 = step_lin10_486 * sc * sg * step_g10_498;
                float value11_503 = step_lin11_487 * sc * sg * step_g11_499;
                float _fabs_60 = fabsf(value00_500);
                float _fabs_61 = fabsf(value10_502);
                float _max_184 = max_noftz(_fabs_60, _fabs_61);
                float block_max0_504 = _max_184;
                float _fabs_62 = fabsf(value01_501);
                float _fabs_63 = fabsf(value11_503);
                float _max_185 = max_noftz(_fabs_62, _fabs_63);
                float block_max1_505 = _max_185;
                float _shfl_xor_90 = __shfl_xor_sync(0xFFFFFFFF, block_max0_504, 4);
                float _max_186 = max_noftz(block_max0_504, _shfl_xor_90);
                block_max0_504 = _max_186;
                float _shfl_xor_91 = __shfl_xor_sync(0xFFFFFFFF, block_max1_505, 4);
                float _max_187 = max_noftz(block_max1_505, _shfl_xor_91);
                block_max1_505 = _max_187;
                float _shfl_xor_92 = __shfl_xor_sync(0xFFFFFFFF, block_max0_504, 8);
                float _max_188 = max_noftz(block_max0_504, _shfl_xor_92);
                block_max0_504 = _max_188;
                float _shfl_xor_93 = __shfl_xor_sync(0xFFFFFFFF, block_max1_505, 8);
                float _max_189 = max_noftz(block_max1_505, _shfl_xor_93);
                block_max1_505 = _max_189;
                float _shfl_xor_94 = __shfl_xor_sync(0xFFFFFFFF, block_max0_504, 16);
                float _max_190 = max_noftz(block_max0_504, _shfl_xor_94);
                block_max0_504 = _max_190;
                float _shfl_xor_95 = __shfl_xor_sync(0xFFFFFFFF, block_max1_505, 16);
                float _max_191 = max_noftz(block_max1_505, _shfl_xor_95);
                block_max1_505 = _max_191;
                float _fp8_rt_30;
                uint16_t _e4m3x2_60;
                uint32_t _f16x2_60;
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_60) : "f"(0.0f), "f"(block_max0_504 * 0.16666666666666666f));
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_60) : "h"(_e4m3x2_60));
                uint16_t _fp8_h0_60 = (uint16_t)(_f16x2_60 & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_30) : "h"(_fp8_h0_60));
                float scale0_506 = _fp8_rt_30;
                float _fp8_rt_31;
                uint16_t _e4m3x2_61;
                uint32_t _f16x2_61;
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_61) : "f"(0.0f), "f"(block_max1_505 * 0.16666666666666666f));
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_61) : "h"(_e4m3x2_61));
                uint16_t _fp8_h0_61 = (uint16_t)(_f16x2_61 & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_31) : "h"(_fp8_h0_61));
                float scale1_507 = _fp8_rt_31;
                float inv_scale0_508 = 0.0f;
                float inv_scale1_509 = 0.0f;
                if (scale0_506 != 0.0f) {
                    inv_scale0_508 = 1.0f / scale0_506;
                }
                if (scale1_507 != 0.0f) {
                    inv_scale1_509 = 1.0f / scale1_507;
                }
                quant_pair[0] = value00_500 * inv_scale0_508;
                quant_pair[1] = value10_502 * inv_scale0_508;
                uint32_t _slice_lo_mask_15;
                {
                    int _lim_62 = 2;
                    if (_lim_62 <= 0) { _slice_lo_mask_15 = 0u; }
                    else if (_lim_62 >= 8) { _slice_lo_mask_15 = ((1u << 8) - 1u); }
                    else {
                        asm volatile("{"
                            ".reg .u32 t;\n\t"
                            "shl.b32 t, 1, %1;\n\t"
                            "add.u32 %0, t, -1;\n\t"
                            "}" : "=r"(_slice_lo_mask_15) : "r"(_lim_62));
                    }
                }
                uint32_t _slice_hi_mask_15;
                {
                    int _lim_63 = 8;
                    if (_lim_63 <= 0) { _slice_hi_mask_15 = 0u; }
                    else if (_lim_63 >= 8) { _slice_hi_mask_15 = ((1u << 8) - 1u); }
                    else {
                        asm volatile("{"
                            ".reg .u32 t;\n\t"
                            "shl.b32 t, 1, %1;\n\t"
                            "add.u32 %0, t, -1;\n\t"
                            "}" : "=r"(_slice_hi_mask_15) : "r"(_lim_63));
                    }
                }
                if (!(_slice_lo_mask_15 | ~_slice_hi_mask_15 & (1u << 0))) quant_pair[0] = 0.0f;
                if (!(_slice_lo_mask_15 | ~_slice_hi_mask_15 & (1u << 1))) quant_pair[1] = 0.0f;
                if (!(_slice_lo_mask_15 | ~_slice_hi_mask_15 & (1u << 2))) quant_pair[2] = 0.0f;
                if (!(_slice_lo_mask_15 | ~_slice_hi_mask_15 & (1u << 3))) quant_pair[3] = 0.0f;
                if (!(_slice_lo_mask_15 | ~_slice_hi_mask_15 & (1u << 4))) quant_pair[4] = 0.0f;
                if (!(_slice_lo_mask_15 | ~_slice_hi_mask_15 & (1u << 5))) quant_pair[5] = 0.0f;
                if (!(_slice_lo_mask_15 | ~_slice_hi_mask_15 & (1u << 6))) quant_pair[6] = 0.0f;
                if (!(_slice_lo_mask_15 | ~_slice_hi_mask_15 & (1u << 7))) quant_pair[7] = 0.0f;
                uint32_t _fp4_30[1];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_30[0]) : "f"(quant_pair[0]), "f"(quant_pair[1]), "f"(quant_pair[2]), "f"(quant_pair[3]), "f"(quant_pair[4]), "f"(quant_pair[5]), "f"(quant_pair[6]), "f"(quant_pair[7]));
                quant_pair[0] = value01_501 * inv_scale1_509;
                quant_pair[1] = value11_503 * inv_scale1_509;
                uint32_t _fp4_31[1];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_31[0]) : "f"(quant_pair[0]), "f"(quant_pair[1]), "f"(quant_pair[2]), "f"(quant_pair[3]), "f"(quant_pair[4]), "f"(quant_pair[5]), "f"(quant_pair[6]), "f"(quant_pair[7]));
                if (lane_1 < 4) {
                    int sf_feature_15 = m_tile * 4 + (unsigned int)warp_local;
                    int sf_tile_stride_15 = 2 * (M_out / 64) * 512;
                    int sf_base_15 = n_tile * (unsigned int)sf_tile_stride_15 + (unsigned int)(token0_482 / 128 * (M_out / 64) * 512) + (unsigned int)(sf_feature_15 / 4 * 512) + (unsigned int)(token0_482 % 32 * 16) + (unsigned int)(token0_482 % 128 / 32 * 4) + (unsigned int)(sf_feature_15 % 4);
                    if (token0_482 < valid_rows) {
                        {
                            unsigned short _fp8_pair;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(scale0_506));
                            *(reinterpret_cast<unsigned char*>(SFC + sf_base_15) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                        }
                    }
                    if (token1_483 < valid_rows) {
                        {
                            unsigned short _fp8_pair;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(scale1_507));
                            *(reinterpret_cast<unsigned char*>(SFC + (sf_base_15 + 16)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                        }
                    }
                }
                int smem_flat0_510 = local_token0_480 * 64 + base_row_241;
                int smem_flat1_511 = local_token1_481 * 64 + base_row_241;
                int smem_index0_512 = smem_flat0_510 / 2 ^ smem_flat0_510 / 256 % 2 * 16;
                int smem_index1_513 = smem_flat1_511 / 2 ^ smem_flat1_511 / 256 % 2 * 16;
                epi_staging[warp_group * 2048 + smem_index0_512] = _fp4_30[0];
                epi_staging[warp_group * 2048 + smem_index1_513] = _fp4_31[0];
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                if (warp_group == 0) {
                    asm volatile("barrier.sync 7, 128;" ::: "memory");
                    if (warp == 0) {
                        if (elect_sync()) {
                            int padding_rows_2 = (256 - valid_rows % 256) % 256;
                            tma_store_4d((&C), m_tile * 64, padding_rows_2 + token_block_238 * 64, 1073741824, n_tile * 256 - (unsigned int)padding_rows_2 + 1073741824, epi_staging_addr);
                        }
                    }
                }
                if (warp_group == 1) {
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                    if (warp == 4) {
                        if (elect_sync()) {
                            int padding_rows_3 = (256 - valid_rows % 256) % 256;
                            tma_store_4d((&C), m_tile * 64, padding_rows_3 + token_block_238 * 64, 1073741824, n_tile * 256 - (unsigned int)padding_rows_3 + 1073741824, epi_staging_addr + 2048);
                        }
                    }
                }
                asm volatile("cp.async.bulk.commit_group;");
                if (warp_group == 0) {
                    asm volatile("barrier.sync 7, 128;" ::: "memory");
                }
                if (warp_group == 1) {
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                }
                epilogue_local_idx = epilogue_local_idx ^ 1;
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
                if (((_clc_ctaid_7 < (unsigned int)num_non_exiting_ctas[0]) ? _clc_valid_3 : (unsigned int)0) == 0) {
                    break;
                }
                m_tile = _clc_ctaid_6 + (unsigned int)cta_rank;
                n_tile = _clc_ctaid_7;
            }
        }
    }
    // ---- Role: load_b_sfb ----
    if (warp >= 8 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 32;");
        { // load_b_sfb_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int stage = 0;
            unsigned int work_stage_1 = 0;
            unsigned int m_tile_1 = blockIdx.x;
            unsigned int n_tile_1 = blockIdx.y;
            int warp_local_1 = warp - 8;
            int loader_thread = warp_local_1 * 32 + lane;
            int routed[16];
            unsigned int cta_mask = 1 << cta_rank;
            unsigned int _phase_k_done = 1;
            unsigned int _phase_work_full_1 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_1 = 0; _tile_iter_1 < grid_m / 2 * grid_n; _tile_iter_1++) {
                if (m_tile_1 >= (unsigned int)grid_m || n_tile_1 >= (unsigned int)grid_n) {
                    break;
                }
                int valid_rows_1 = (unsigned int)tile_mn_limit[n_tile_1] - n_tile_1 * 256;
                int route_base = n_tile_1 * 256 + (unsigned int)(cta_rank * 128) + (unsigned int)(warp_local_1 * 4);
                for (int row = 0; row < 4; row++) {
                    routed[row] = route_map[route_base + row];
                }
                int route_base_0 = n_tile_1 * 256 + (unsigned int)(cta_rank * 128) + (unsigned int)((8 + warp_local_1) * 4);
                for (int row_1 = 0; row_1 < 4; row_1++) {
                    routed[4 + row_1] = route_map[route_base_0 + row_1];
                }
                int route_base_1 = n_tile_1 * 256 + (unsigned int)(cta_rank * 128) + (unsigned int)((16 + warp_local_1) * 4);
                for (int row_2 = 0; row_2 < 4; row_2++) {
                    routed[8 + row_2] = route_map[route_base_1 + row_2];
                }
                int route_base_2 = n_tile_1 * 256 + (unsigned int)(cta_rank * 128) + (unsigned int)((24 + warp_local_1) * 4);
                for (int row_3 = 0; row_3 < 4; row_3++) {
                    routed[12 + row_3] = route_map[route_base_2 + row_3];
                }
                int shuffled_row = loader_thread / 128 * 128 + loader_thread % 4 * 32 + loader_thread % 128 / 4;
                int routed_sf = 0;
                if (shuffled_row < valid_rows_1) {
                    routed_sf = route_map[n_tile_1 * 256 + (unsigned int)shuffled_row];
                }
                #pragma unroll 1
                for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                    mbarrier_wait(k_done_addr + (stage) * 8, _phase_k_done);
                    if (elect_sync()) {
                        tma_gather4_gmem2smem_mc_cta2(smem_b_addr + stage * 16384 + (unsigned int)(warp_local_1 * 512), (&B), iter_k * 128, routed[0], routed[1], routed[2], routed[3], ((b_full_addr + (stage) * 8) & 0xFEFFFFFF), cta_mask);
                        tma_gather4_gmem2smem_mc_cta2(smem_b_addr + stage * 16384 + (unsigned int)((8 + warp_local_1) * 512), (&B), iter_k * 128, routed[4], routed[5], routed[6], routed[7], ((b_full_addr + (stage) * 8) & 0xFEFFFFFF), cta_mask);
                        tma_gather4_gmem2smem_mc_cta2(smem_b_addr + stage * 16384 + (unsigned int)((16 + warp_local_1) * 512), (&B), iter_k * 128, routed[8], routed[9], routed[10], routed[11], ((b_full_addr + (stage) * 8) & 0xFEFFFFFF), cta_mask);
                        tma_gather4_gmem2smem_mc_cta2(smem_b_addr + stage * 16384 + (unsigned int)((24 + warp_local_1) * 512), (&B), iter_k * 128, routed[12], routed[13], routed[14], routed[15], ((b_full_addr + (stage) * 8) & 0xFEFFFFFF), cta_mask);
                    }
                    if (warp == 8) {
                        if (elect_sync()) {
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((b_full_addr + (stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                        }
                    }
                    if (shuffled_row < valid_rows_1) {
                        int _vec_load_0[4];
                        {
                            const int4* _ivptr_0 = reinterpret_cast<const int4*>(SFB + routed_sf * (K / 64) + iter_k * 4);
                            int4 _ivld_0;
                            _ivld_0 = *_ivptr_0;
                            _vec_load_0[0 + 0] = _ivld_0.x;
                            _vec_load_0[0 + 1] = _ivld_0.y;
                            _vec_load_0[0 + 2] = _ivld_0.z;
                            _vec_load_0[0 + 3] = _ivld_0.w;
                        }
                        int data_block_row = loader_thread / 128;
                        int row_in_block = loader_thread % 128;
                        {
                            uint32_t _ival_1 = static_cast<uint32_t>(_vec_load_0[0]);
                            uint32_t _addr_1 = static_cast<uint32_t>((smem_sfb_addr + stage * 4096 + (unsigned int)(data_block_row * 2048 + row_in_block * 4)));
                            asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_1), "r"(_ival_1) : "memory");
                        }
                        {
                            uint32_t _ival_2 = static_cast<uint32_t>(_vec_load_0[1]);
                            uint32_t _addr_2 = static_cast<uint32_t>((smem_sfb_addr + stage * 4096 + (unsigned int)(data_block_row * 2048 + 512 + row_in_block * 4)));
                            asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_2), "r"(_ival_2) : "memory");
                        }
                        {
                            uint32_t _ival_3 = static_cast<uint32_t>(_vec_load_0[2]);
                            uint32_t _addr_3 = static_cast<uint32_t>((smem_sfb_addr + stage * 4096 + (unsigned int)(data_block_row * 2048 + 1024 + row_in_block * 4)));
                            asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_3), "r"(_ival_3) : "memory");
                        }
                        {
                            uint32_t _ival_4 = static_cast<uint32_t>(_vec_load_0[3]);
                            uint32_t _addr_4 = static_cast<uint32_t>((smem_sfb_addr + stage * 4096 + (unsigned int)(data_block_row * 2048 + 1536 + row_in_block * 4)));
                            asm volatile("st.shared.u32 [%0], %1;" :: "r"(_addr_4), "r"(_ival_4) : "memory");
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((sfb_full_addr + (stage) * 8) & 0xFEFFFFFF) : "memory");
                    stage += 1;
                    if (stage == 4) { stage = 0; _phase_k_done ^= 1; }
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
                if (((_clc_ctaid_3 < (unsigned int)num_non_exiting_ctas[0]) ? _clc_valid_1 : (unsigned int)0) == 0) {
                    break;
                }
                m_tile_1 = _clc_ctaid_2 + (unsigned int)cta_rank;
                n_tile_1 = _clc_ctaid_3;
            }
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
    }
    // ---- Role: load_a_sfa ----
    if (warp == 16) {
        { // load_a_sfa_main
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
                if (m_tile_2 >= (unsigned int)grid_m || n_tile_2 >= (unsigned int)grid_n) {
                    break;
                }
                int expert_1 = tile_expert[n_tile_2];
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
                        asm volatile(
                            "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5}], [%6], %7, %8;"
                            :: "r"(smem_a_addr + stage_1 * 16384), "l"((&A)), "r"(0), "r"(m_tile_2 * 128), "r"(iter_k_1), "r"(expert_1),
                               "r"(((a_full_addr + (stage_1) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(cta_mask_1)), "l"(0x12F0000000000000ULL) : "memory");
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((a_full_addr + (stage_1) * 8) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5}], [%6], %7, %8;"
                            :: "r"(smem_sfa_addr + stage_1 * 2048), "l"((&SFA)), "r"(0), "r"(0), "r"(iter_k_1 * 4), "r"((unsigned int)(expert_1 * grid_m) + m_tile_2),
                               "r"(((sfa_full_addr + (stage_1) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(cta_mask_1)), "l"(0x12F0000000000000ULL) : "memory");
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((sfa_full_addr + (stage_1) * 8) & 0xFEFFFFFF), "r"((uint32_t)(2048)) : "memory");
                    }
                    stage_1 += 1;
                    if (stage_1 == 4) { stage_1 = 0; _phase_k_done_1 ^= 1; }
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
                if (((_clc_ctaid_1 < (unsigned int)num_non_exiting_ctas[0]) ? _clc_valid_0 : (unsigned int)0) == 0) {
                    break;
                }
                m_tile_2 = _clc_ctaid_0 + (unsigned int)cta_rank;
                n_tile_2 = _clc_ctaid_1;
            }
        }
    }
    // ---- Role: copy_sfab_mma ----
    if (warp == 17) {
        { // copy_sfab_mma_main
            unsigned int _phase_mma_free_0 = 1;
            unsigned int _phase_a_full = 0;
            unsigned int _phase_b_full = 0;
            unsigned int _phase_sfa_full = 0;
            unsigned int _phase_sfb_full = 0;
            unsigned int _phase_work_full_3 = 0;
            if (cta_rank == 0) {
                unsigned int stage_2 = 0;
                unsigned int work_stage_3 = 0;
                int mma_local_idx = 0;
                unsigned int m_tile_3 = blockIdx.x;
                unsigned int n_tile_3 = blockIdx.y;
                #pragma unroll 1
                for (unsigned int _tile_iter_3 = 0; _tile_iter_3 < grid_m / 2 * grid_n; _tile_iter_3++) {
                    if (m_tile_3 >= (unsigned int)grid_m || n_tile_3 >= (unsigned int)grid_n) {
                        break;
                    }
                    mbarrier_wait(mma_free_addr, _phase_mma_free_0);
                    _phase_mma_free_0 ^= 1;
                    #pragma unroll 1
                    for (int iter_k_2 = 0; iter_k_2 < K_tiles; iter_k_2++) {
                        mbarrier_wait(a_full_addr + (stage_2) * 8, _phase_a_full);
                        mbarrier_wait(b_full_addr + (stage_2) * 8, _phase_b_full);
                        mbarrier_wait(sfa_full_addr + (stage_2) * 8, _phase_sfa_full);
                        mbarrier_wait(sfb_full_addr + (stage_2) * 8, _phase_sfb_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        if (elect_sync()) {
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                            #endif
                            {
                                uint64_t _tcgen05_cp_desc_0 = ((((uint64_t)(smem_sfa_addr + stage_2 * 2048)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                asm volatile(
                                    "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                    :: "r"((uint32_t)(tmem_sfa)), "l"(_tcgen05_cp_desc_0)
                                    : "memory");
                            }
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                            #endif
                            {
                                uint64_t _tcgen05_cp_desc_1 = ((((uint64_t)(smem_sfa_addr + stage_2 * 2048 + 512)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                asm volatile(
                                    "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                    :: "r"((uint32_t)(tmem_sfa + 4)), "l"(_tcgen05_cp_desc_1)
                                    : "memory");
                            }
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                            #endif
                            {
                                uint64_t _tcgen05_cp_desc_2 = ((((uint64_t)(smem_sfa_addr + stage_2 * 2048 + 1024)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                asm volatile(
                                    "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                    :: "r"((uint32_t)(tmem_sfa + 8)), "l"(_tcgen05_cp_desc_2)
                                    : "memory");
                            }
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                            #endif
                            {
                                uint64_t _tcgen05_cp_desc_3 = ((((uint64_t)(smem_sfa_addr + stage_2 * 2048 + 1536)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                asm volatile(
                                    "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                    :: "r"((uint32_t)(tmem_sfa + 12)), "l"(_tcgen05_cp_desc_3)
                                    : "memory");
                            }
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                            #endif
                            {
                                uint64_t _tcgen05_cp_desc_4 = ((((uint64_t)(smem_sfb_addr + stage_2 * 4096)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                asm volatile(
                                    "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                    :: "r"((uint32_t)(tmem_sfb)), "l"(_tcgen05_cp_desc_4)
                                    : "memory");
                            }
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                            #endif
                            {
                                uint64_t _tcgen05_cp_desc_5 = ((((uint64_t)(smem_sfb_addr + stage_2 * 4096 + 512)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                asm volatile(
                                    "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                    :: "r"((uint32_t)(tmem_sfb + 8)), "l"(_tcgen05_cp_desc_5)
                                    : "memory");
                            }
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                            #endif
                            {
                                uint64_t _tcgen05_cp_desc_6 = ((((uint64_t)(smem_sfb_addr + stage_2 * 4096 + 1024)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                asm volatile(
                                    "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                    :: "r"((uint32_t)(tmem_sfb + 16)), "l"(_tcgen05_cp_desc_6)
                                    : "memory");
                            }
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                            #endif
                            {
                                uint64_t _tcgen05_cp_desc_7 = ((((uint64_t)(smem_sfb_addr + stage_2 * 4096 + 1536)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                asm volatile(
                                    "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                    :: "r"((uint32_t)(tmem_sfb + 24)), "l"(_tcgen05_cp_desc_7)
                                    : "memory");
                            }
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                            #endif
                            {
                                uint64_t _tcgen05_cp_desc_8 = ((((uint64_t)(smem_sfb_addr + stage_2 * 4096 + 2048)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                asm volatile(
                                    "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                    :: "r"((uint32_t)(tmem_sfb + 4)), "l"(_tcgen05_cp_desc_8)
                                    : "memory");
                            }
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                            #endif
                            {
                                uint64_t _tcgen05_cp_desc_9 = ((((uint64_t)(smem_sfb_addr + stage_2 * 4096 + 2048 + 512)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                asm volatile(
                                    "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                    :: "r"((uint32_t)(tmem_sfb + 12)), "l"(_tcgen05_cp_desc_9)
                                    : "memory");
                            }
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                            #endif
                            {
                                uint64_t _tcgen05_cp_desc_10 = ((((uint64_t)(smem_sfb_addr + stage_2 * 4096 + 2048 + 1024)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                asm volatile(
                                    "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                    :: "r"((uint32_t)(tmem_sfb + 20)), "l"(_tcgen05_cp_desc_10)
                                    : "memory");
                            }
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                            #endif
                            {
                                uint64_t _tcgen05_cp_desc_11 = ((((uint64_t)(smem_sfb_addr + stage_2 * 4096 + 2048 + 1536)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                asm volatile(
                                    "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                    :: "r"((uint32_t)(tmem_sfb + 28)), "l"(_tcgen05_cp_desc_11)
                                    : "memory");
                            }
                        }
                        int _mma_a_lo_0 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_2) * 1024;
                        int _mma_b_lo_0 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_2) * 1024;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_cta2((tmem_accum + (mma_local_idx * 192)), a_desc + 0, b_desc + 0,
                                    0x10400480U, tmem_sfa + 0, tmem_sfb + 0, ((((1) ? ((iter_k_2 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                            }
                        }
                        int _mma_a_lo_1 = (((smem_a_addr + 32) >> 4) & 0x3FFF) + (stage_2) * 1024;
                        int _mma_b_lo_1 = (((smem_b_addr + 32) >> 4) & 0x3FFF) + (stage_2) * 1024;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_cta2((tmem_accum + (mma_local_idx * 192)), a_desc + 0, b_desc + 0,
                                    0x10400480U, tmem_sfa + 4 + 0, tmem_sfb + 8 + 0, ((((0) ? ((iter_k_2 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                            }
                        }
                        int _mma_a_lo_2 = (((smem_a_addr + 64) >> 4) & 0x3FFF) + (stage_2) * 1024;
                        int _mma_b_lo_2 = (((smem_b_addr + 64) >> 4) & 0x3FFF) + (stage_2) * 1024;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_2) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_2) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_cta2((tmem_accum + (mma_local_idx * 192)), a_desc + 0, b_desc + 0,
                                    0x10400480U, tmem_sfa + 8 + 0, tmem_sfb + 16 + 0, ((((0) ? ((iter_k_2 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                            }
                        }
                        int _mma_a_lo_3 = (((smem_a_addr + 96) >> 4) & 0x3FFF) + (stage_2) * 1024;
                        int _mma_b_lo_3 = (((smem_b_addr + 96) >> 4) & 0x3FFF) + (stage_2) * 1024;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_3) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_3) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_cta2((tmem_accum + (mma_local_idx * 192)), a_desc + 0, b_desc + 0,
                                    0x10400480U, tmem_sfa + 12 + 0, tmem_sfb + 24 + 0, ((((0) ? ((iter_k_2 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                            }
                        }
                        elect_commit_cg2_multicast(k_done_addr + (stage_2) * 8, (uint16_t)(3));
                        stage_2 += 1;
                        if (stage_2 == 4) { stage_2 = 0; _phase_a_full ^= 1; _phase_b_full ^= 1; _phase_sfa_full ^= 1; _phase_sfb_full ^= 1; }
                    }
                    elect_commit_cg2_multicast(mma_full_addr, (uint16_t)(3));
                    mma_local_idx = mma_local_idx ^ 1;
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
                    if (((_clc_ctaid_5 < (unsigned int)num_non_exiting_ctas[0]) ? _clc_valid_2 : (unsigned int)0) == 0) {
                        break;
                    }
                    m_tile_3 = _clc_ctaid_4 + (unsigned int)cta_rank;
                    n_tile_3 = _clc_ctaid_5;
                }
            }
        }
    }
    // ---- Role: work_id ----
    if (warp == 18) {
        { // work_id_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int work_stage_4 = 0;
            unsigned int throttle_stage_1 = 0;
            unsigned int _phase_throttle_full = 0;
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_work_full_4 = 0;
            unsigned int _phase_fast_ready = 0;
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
                        : "r"(work_response_addr + work_stage_4 * 16 + 0 * 16)
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
                        : "r"(work_response_addr + work_stage_4 * 16 + 0 * 16)
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
                    if (((_clc_ctaid_11 < (unsigned int)num_non_exiting_ctas[0]) ? _clc_valid_5 : (unsigned int)0) == 0) {
                        if (_clc_valid_5 != 0) {
                            if (_clc_ctaid_11 >= (unsigned int)num_non_exiting_ctas[0]) {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                if (elect_sync()) {
                                    mbarrier_init(fast_ready_addr, 1);
                                }
                                __syncwarp();
                                unsigned int fast_stage = 0;
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
                                        : "r"(fast_response_addr + fast_stage * 64 + 0 * 16)
                                        : "memory");
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
                                        : "r"(fast_response_addr + fast_stage * 64 + 1 * 16)
                                        : "memory");
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
                                        : "r"(fast_response_addr + fast_stage * 64 + 2 * 16)
                                        : "memory");
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
                                        : "r"(fast_response_addr + fast_stage * 64 + 3 * 16)
                                        : "memory");
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    _phase_fast_ready ^= 1;
                                    if (_clc_valid_6 + _clc_valid_7 + _clc_valid_8 + _clc_valid_9 == 0) {
                                        break;
                                    }
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
    if (warp == 19) {
        { // padding_main
            unsigned int work_stage_5 = 0;
            unsigned int m_tile_4 = blockIdx.x;
            unsigned int n_tile_4 = blockIdx.y;
            unsigned int _phase_work_full_5 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_5 = 0; _tile_iter_5 < grid_m / 2 * grid_n; _tile_iter_5++) {
                if (m_tile_4 >= (unsigned int)grid_m || n_tile_4 >= (unsigned int)grid_n) {
                    break;
                }
                mbarrier_wait(work_full_addr + (work_stage_5) * 8, _phase_work_full_5);
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
                    : "r"(work_response_addr + work_stage_5 * 16 + 0 * 16)
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
                    : "r"(work_response_addr + work_stage_5 * 16 + 0 * 16)
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
                if (((_clc_ctaid_9 < (unsigned int)num_non_exiting_ctas[0]) ? _clc_valid_4 : (unsigned int)0) == 0) {
                    break;
                }
                m_tile_4 = _clc_ctaid_8 + (unsigned int)cta_rank;
                n_tile_4 = _clc_ctaid_9;
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
