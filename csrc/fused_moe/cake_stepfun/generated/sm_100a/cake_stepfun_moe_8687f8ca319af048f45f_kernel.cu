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


extern "C" {

__global__ __launch_bounds__(640, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_cake_stepfun_moe_8687f8ca319af048f45f(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, int* __restrict__ SFB, __nv_bfloat16* __restrict__ C, float* __restrict__ SFC, int* __restrict__ route_map, int* __restrict__ tile_expert, int* __restrict__ tile_mn_limit, int* __restrict__ num_non_exiting_ctas, float* __restrict__ scale_c, float* __restrict__ scale_gate, float* __restrict__ clamp_limit, float* __restrict__ act_alpha, float* __restrict__ act_beta, int M_out, int K, int grid_m, int grid_n, int K_tiles)
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
            float out_pair[2] = {0};
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
                int tok_slot_base = (int)n_tile * 256;
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
                int tok_feature = (int)m_tile * 64 + base_row;
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
                float tok_sf0 = 0.0f;
                float tok_sf1 = 0.0f;
                if (token0 < valid_rows) {
                    int tok_route0 = route_map[tok_slot_base + token0];
                    tok_sf0 = SFC[tok_route0];
                }
                if (token1 < valid_rows) {
                    int tok_route1 = route_map[tok_slot_base + token1];
                    tok_sf1 = SFC[tok_route1];
                }
                float _max_0 = max_noftz(_tmem_load_0[0] * tok_sf0, neg_cl);
                float _min_0 = fminf(_max_0, cl);
                float tok_lin00 = _min_0;
                float _max_1 = max_noftz(_tmem_load_0[1] * tok_sf1, neg_cl);
                float _min_1 = fminf(_max_1, cl);
                float tok_lin01 = _min_1;
                float _max_2 = max_noftz(_tmem_load_1[0] * tok_sf0, neg_cl);
                float _min_2 = fminf(_max_2, cl);
                float tok_lin10 = _min_2;
                float _max_3 = max_noftz(_tmem_load_1[1] * tok_sf1, neg_cl);
                float _min_3 = fminf(_max_3, cl);
                float tok_lin11 = _min_3;
                float tok_x00 = _tmem_load_0[2] * tok_sf0;
                float tok_x01 = _tmem_load_0[3] * tok_sf1;
                float tok_x10 = _tmem_load_1[2] * tok_sf0;
                float tok_x11 = _tmem_load_1[3] * tok_sf1;
                float _exp2_0 = approx_exp2((-(tok_x00 * sg)) * 1.4426950408889634f);
                float _rcp_0 = approx_rcp(1.0f + _exp2_0);
                float tok_sig00 = _rcp_0;
                float _exp2_1 = approx_exp2((-(tok_x01 * sg)) * 1.4426950408889634f);
                float _rcp_1 = approx_rcp(1.0f + _exp2_1);
                float tok_sig01 = _rcp_1;
                float _exp2_2 = approx_exp2((-(tok_x10 * sg)) * 1.4426950408889634f);
                float _rcp_2 = approx_rcp(1.0f + _exp2_2);
                float tok_sig10 = _rcp_2;
                float _exp2_3 = approx_exp2((-(tok_x11 * sg)) * 1.4426950408889634f);
                float _rcp_3 = approx_rcp(1.0f + _exp2_3);
                float tok_sig11 = _rcp_3;
                float _min_4 = fminf(tok_x00 * tok_sig00, cl);
                float tok_g00 = _min_4;
                float _min_5 = fminf(tok_x01 * tok_sig01, cl);
                float tok_g01 = _min_5;
                float _min_6 = fminf(tok_x10 * tok_sig10, cl);
                float tok_g10 = _min_6;
                float _min_7 = fminf(tok_x11 * tok_sig11, cl);
                float tok_g11 = _min_7;
                float value00 = tok_lin00 * sc * sg * tok_g00;
                float value01 = tok_lin01 * sc * sg * tok_g01;
                float value10 = tok_lin10 * sc * sg * tok_g10;
                float value11 = tok_lin11 * sc * sg * tok_g11;
                out_pair[0] = value00;
                out_pair[1] = value10;
                if (token0 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token0) * M_out + tok_feature)))[0]) = _pk;
                    }
                }
                out_pair[0] = value01;
                out_pair[1] = value11;
                if (token1 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token1) * M_out + tok_feature)))[0]) = _pk;
                    }
                }
                int local_token0_0 = lane_1 % 4 * 2 + 8;
                int local_token1_1 = local_token0_0 + 1;
                int token0_2 = token_block * 64 + local_token0_0;
                int token1_3 = token0_2 + 1;
                float tok_sf0_4 = 0.0f;
                float tok_sf1_5 = 0.0f;
                if (token0_2 < valid_rows) {
                    int tok_route0_1 = route_map[tok_slot_base + token0_2];
                    tok_sf0_4 = SFC[tok_route0_1];
                }
                if (token1_3 < valid_rows) {
                    int tok_route1_1 = route_map[tok_slot_base + token1_3];
                    tok_sf1_5 = SFC[tok_route1_1];
                }
                float _max_4 = max_noftz(_tmem_load_0[4] * tok_sf0_4, neg_cl);
                float _min_8 = fminf(_max_4, cl);
                float tok_lin00_6 = _min_8;
                float _max_5 = max_noftz(_tmem_load_0[5] * tok_sf1_5, neg_cl);
                float _min_9 = fminf(_max_5, cl);
                float tok_lin01_7 = _min_9;
                float _max_6 = max_noftz(_tmem_load_1[4] * tok_sf0_4, neg_cl);
                float _min_10 = fminf(_max_6, cl);
                float tok_lin10_8 = _min_10;
                float _max_7 = max_noftz(_tmem_load_1[5] * tok_sf1_5, neg_cl);
                float _min_11 = fminf(_max_7, cl);
                float tok_lin11_9 = _min_11;
                float tok_x00_10 = _tmem_load_0[6] * tok_sf0_4;
                float tok_x01_11 = _tmem_load_0[7] * tok_sf1_5;
                float tok_x10_12 = _tmem_load_1[6] * tok_sf0_4;
                float tok_x11_13 = _tmem_load_1[7] * tok_sf1_5;
                float _exp2_4 = approx_exp2((-(tok_x00_10 * sg)) * 1.4426950408889634f);
                float _rcp_4 = approx_rcp(1.0f + _exp2_4);
                float tok_sig00_14 = _rcp_4;
                float _exp2_5 = approx_exp2((-(tok_x01_11 * sg)) * 1.4426950408889634f);
                float _rcp_5 = approx_rcp(1.0f + _exp2_5);
                float tok_sig01_15 = _rcp_5;
                float _exp2_6 = approx_exp2((-(tok_x10_12 * sg)) * 1.4426950408889634f);
                float _rcp_6 = approx_rcp(1.0f + _exp2_6);
                float tok_sig10_16 = _rcp_6;
                float _exp2_7 = approx_exp2((-(tok_x11_13 * sg)) * 1.4426950408889634f);
                float _rcp_7 = approx_rcp(1.0f + _exp2_7);
                float tok_sig11_17 = _rcp_7;
                float _min_12 = fminf(tok_x00_10 * tok_sig00_14, cl);
                float tok_g00_18 = _min_12;
                float _min_13 = fminf(tok_x01_11 * tok_sig01_15, cl);
                float tok_g01_19 = _min_13;
                float _min_14 = fminf(tok_x10_12 * tok_sig10_16, cl);
                float tok_g10_20 = _min_14;
                float _min_15 = fminf(tok_x11_13 * tok_sig11_17, cl);
                float tok_g11_21 = _min_15;
                float value00_22 = tok_lin00_6 * sc * sg * tok_g00_18;
                float value01_23 = tok_lin01_7 * sc * sg * tok_g01_19;
                float value10_24 = tok_lin10_8 * sc * sg * tok_g10_20;
                float value11_25 = tok_lin11_9 * sc * sg * tok_g11_21;
                out_pair[0] = value00_22;
                out_pair[1] = value10_24;
                if (token0_2 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token0_2) * M_out + tok_feature)))[0]) = _pk;
                    }
                }
                out_pair[0] = value01_23;
                out_pair[1] = value11_25;
                if (token1_3 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token1_3) * M_out + tok_feature)))[0]) = _pk;
                    }
                }
                int local_token0_26 = lane_1 % 4 * 2 + 16;
                int local_token1_27 = local_token0_26 + 1;
                int token0_28 = token_block * 64 + local_token0_26;
                int token1_29 = token0_28 + 1;
                float tok_sf0_30 = 0.0f;
                float tok_sf1_31 = 0.0f;
                if (token0_28 < valid_rows) {
                    int tok_route0_2 = route_map[tok_slot_base + token0_28];
                    tok_sf0_30 = SFC[tok_route0_2];
                }
                if (token1_29 < valid_rows) {
                    int tok_route1_2 = route_map[tok_slot_base + token1_29];
                    tok_sf1_31 = SFC[tok_route1_2];
                }
                float _max_8 = max_noftz(_tmem_load_0[8] * tok_sf0_30, neg_cl);
                float _min_16 = fminf(_max_8, cl);
                float tok_lin00_32 = _min_16;
                float _max_9 = max_noftz(_tmem_load_0[9] * tok_sf1_31, neg_cl);
                float _min_17 = fminf(_max_9, cl);
                float tok_lin01_33 = _min_17;
                float _max_10 = max_noftz(_tmem_load_1[8] * tok_sf0_30, neg_cl);
                float _min_18 = fminf(_max_10, cl);
                float tok_lin10_34 = _min_18;
                float _max_11 = max_noftz(_tmem_load_1[9] * tok_sf1_31, neg_cl);
                float _min_19 = fminf(_max_11, cl);
                float tok_lin11_35 = _min_19;
                float tok_x00_36 = _tmem_load_0[10] * tok_sf0_30;
                float tok_x01_37 = _tmem_load_0[11] * tok_sf1_31;
                float tok_x10_38 = _tmem_load_1[10] * tok_sf0_30;
                float tok_x11_39 = _tmem_load_1[11] * tok_sf1_31;
                float _exp2_8 = approx_exp2((-(tok_x00_36 * sg)) * 1.4426950408889634f);
                float _rcp_8 = approx_rcp(1.0f + _exp2_8);
                float tok_sig00_40 = _rcp_8;
                float _exp2_9 = approx_exp2((-(tok_x01_37 * sg)) * 1.4426950408889634f);
                float _rcp_9 = approx_rcp(1.0f + _exp2_9);
                float tok_sig01_41 = _rcp_9;
                float _exp2_10 = approx_exp2((-(tok_x10_38 * sg)) * 1.4426950408889634f);
                float _rcp_10 = approx_rcp(1.0f + _exp2_10);
                float tok_sig10_42 = _rcp_10;
                float _exp2_11 = approx_exp2((-(tok_x11_39 * sg)) * 1.4426950408889634f);
                float _rcp_11 = approx_rcp(1.0f + _exp2_11);
                float tok_sig11_43 = _rcp_11;
                float _min_20 = fminf(tok_x00_36 * tok_sig00_40, cl);
                float tok_g00_44 = _min_20;
                float _min_21 = fminf(tok_x01_37 * tok_sig01_41, cl);
                float tok_g01_45 = _min_21;
                float _min_22 = fminf(tok_x10_38 * tok_sig10_42, cl);
                float tok_g10_46 = _min_22;
                float _min_23 = fminf(tok_x11_39 * tok_sig11_43, cl);
                float tok_g11_47 = _min_23;
                float value00_48 = tok_lin00_32 * sc * sg * tok_g00_44;
                float value01_49 = tok_lin01_33 * sc * sg * tok_g01_45;
                float value10_50 = tok_lin10_34 * sc * sg * tok_g10_46;
                float value11_51 = tok_lin11_35 * sc * sg * tok_g11_47;
                out_pair[0] = value00_48;
                out_pair[1] = value10_50;
                if (token0_28 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token0_28) * M_out + tok_feature)))[0]) = _pk;
                    }
                }
                out_pair[0] = value01_49;
                out_pair[1] = value11_51;
                if (token1_29 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token1_29) * M_out + tok_feature)))[0]) = _pk;
                    }
                }
                int local_token0_52 = lane_1 % 4 * 2 + 24;
                int local_token1_53 = local_token0_52 + 1;
                int token0_54 = token_block * 64 + local_token0_52;
                int token1_55 = token0_54 + 1;
                float tok_sf0_56 = 0.0f;
                float tok_sf1_57 = 0.0f;
                if (token0_54 < valid_rows) {
                    int tok_route0_3 = route_map[tok_slot_base + token0_54];
                    tok_sf0_56 = SFC[tok_route0_3];
                }
                if (token1_55 < valid_rows) {
                    int tok_route1_3 = route_map[tok_slot_base + token1_55];
                    tok_sf1_57 = SFC[tok_route1_3];
                }
                float _max_12 = max_noftz(_tmem_load_0[12] * tok_sf0_56, neg_cl);
                float _min_24 = fminf(_max_12, cl);
                float tok_lin00_58 = _min_24;
                float _max_13 = max_noftz(_tmem_load_0[13] * tok_sf1_57, neg_cl);
                float _min_25 = fminf(_max_13, cl);
                float tok_lin01_59 = _min_25;
                float _max_14 = max_noftz(_tmem_load_1[12] * tok_sf0_56, neg_cl);
                float _min_26 = fminf(_max_14, cl);
                float tok_lin10_60 = _min_26;
                float _max_15 = max_noftz(_tmem_load_1[13] * tok_sf1_57, neg_cl);
                float _min_27 = fminf(_max_15, cl);
                float tok_lin11_61 = _min_27;
                float tok_x00_62 = _tmem_load_0[14] * tok_sf0_56;
                float tok_x01_63 = _tmem_load_0[15] * tok_sf1_57;
                float tok_x10_64 = _tmem_load_1[14] * tok_sf0_56;
                float tok_x11_65 = _tmem_load_1[15] * tok_sf1_57;
                float _exp2_12 = approx_exp2((-(tok_x00_62 * sg)) * 1.4426950408889634f);
                float _rcp_12 = approx_rcp(1.0f + _exp2_12);
                float tok_sig00_66 = _rcp_12;
                float _exp2_13 = approx_exp2((-(tok_x01_63 * sg)) * 1.4426950408889634f);
                float _rcp_13 = approx_rcp(1.0f + _exp2_13);
                float tok_sig01_67 = _rcp_13;
                float _exp2_14 = approx_exp2((-(tok_x10_64 * sg)) * 1.4426950408889634f);
                float _rcp_14 = approx_rcp(1.0f + _exp2_14);
                float tok_sig10_68 = _rcp_14;
                float _exp2_15 = approx_exp2((-(tok_x11_65 * sg)) * 1.4426950408889634f);
                float _rcp_15 = approx_rcp(1.0f + _exp2_15);
                float tok_sig11_69 = _rcp_15;
                float _min_28 = fminf(tok_x00_62 * tok_sig00_66, cl);
                float tok_g00_70 = _min_28;
                float _min_29 = fminf(tok_x01_63 * tok_sig01_67, cl);
                float tok_g01_71 = _min_29;
                float _min_30 = fminf(tok_x10_64 * tok_sig10_68, cl);
                float tok_g10_72 = _min_30;
                float _min_31 = fminf(tok_x11_65 * tok_sig11_69, cl);
                float tok_g11_73 = _min_31;
                float value00_74 = tok_lin00_58 * sc * sg * tok_g00_70;
                float value01_75 = tok_lin01_59 * sc * sg * tok_g01_71;
                float value10_76 = tok_lin10_60 * sc * sg * tok_g10_72;
                float value11_77 = tok_lin11_61 * sc * sg * tok_g11_73;
                out_pair[0] = value00_74;
                out_pair[1] = value10_76;
                if (token0_54 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token0_54) * M_out + tok_feature)))[0]) = _pk;
                    }
                }
                out_pair[0] = value01_75;
                out_pair[1] = value11_77;
                if (token1_55 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token1_55) * M_out + tok_feature)))[0]) = _pk;
                    }
                }
                int local_token0_78 = lane_1 % 4 * 2 + 32;
                int local_token1_79 = local_token0_78 + 1;
                int token0_80 = token_block * 64 + local_token0_78;
                int token1_81 = token0_80 + 1;
                float tok_sf0_82 = 0.0f;
                float tok_sf1_83 = 0.0f;
                if (token0_80 < valid_rows) {
                    int tok_route0_4 = route_map[tok_slot_base + token0_80];
                    tok_sf0_82 = SFC[tok_route0_4];
                }
                if (token1_81 < valid_rows) {
                    int tok_route1_4 = route_map[tok_slot_base + token1_81];
                    tok_sf1_83 = SFC[tok_route1_4];
                }
                float _max_16 = max_noftz(_tmem_load_0[16] * tok_sf0_82, neg_cl);
                float _min_32 = fminf(_max_16, cl);
                float tok_lin00_84 = _min_32;
                float _max_17 = max_noftz(_tmem_load_0[17] * tok_sf1_83, neg_cl);
                float _min_33 = fminf(_max_17, cl);
                float tok_lin01_85 = _min_33;
                float _max_18 = max_noftz(_tmem_load_1[16] * tok_sf0_82, neg_cl);
                float _min_34 = fminf(_max_18, cl);
                float tok_lin10_86 = _min_34;
                float _max_19 = max_noftz(_tmem_load_1[17] * tok_sf1_83, neg_cl);
                float _min_35 = fminf(_max_19, cl);
                float tok_lin11_87 = _min_35;
                float tok_x00_88 = _tmem_load_0[18] * tok_sf0_82;
                float tok_x01_89 = _tmem_load_0[19] * tok_sf1_83;
                float tok_x10_90 = _tmem_load_1[18] * tok_sf0_82;
                float tok_x11_91 = _tmem_load_1[19] * tok_sf1_83;
                float _exp2_16 = approx_exp2((-(tok_x00_88 * sg)) * 1.4426950408889634f);
                float _rcp_16 = approx_rcp(1.0f + _exp2_16);
                float tok_sig00_92 = _rcp_16;
                float _exp2_17 = approx_exp2((-(tok_x01_89 * sg)) * 1.4426950408889634f);
                float _rcp_17 = approx_rcp(1.0f + _exp2_17);
                float tok_sig01_93 = _rcp_17;
                float _exp2_18 = approx_exp2((-(tok_x10_90 * sg)) * 1.4426950408889634f);
                float _rcp_18 = approx_rcp(1.0f + _exp2_18);
                float tok_sig10_94 = _rcp_18;
                float _exp2_19 = approx_exp2((-(tok_x11_91 * sg)) * 1.4426950408889634f);
                float _rcp_19 = approx_rcp(1.0f + _exp2_19);
                float tok_sig11_95 = _rcp_19;
                float _min_36 = fminf(tok_x00_88 * tok_sig00_92, cl);
                float tok_g00_96 = _min_36;
                float _min_37 = fminf(tok_x01_89 * tok_sig01_93, cl);
                float tok_g01_97 = _min_37;
                float _min_38 = fminf(tok_x10_90 * tok_sig10_94, cl);
                float tok_g10_98 = _min_38;
                float _min_39 = fminf(tok_x11_91 * tok_sig11_95, cl);
                float tok_g11_99 = _min_39;
                float value00_100 = tok_lin00_84 * sc * sg * tok_g00_96;
                float value01_101 = tok_lin01_85 * sc * sg * tok_g01_97;
                float value10_102 = tok_lin10_86 * sc * sg * tok_g10_98;
                float value11_103 = tok_lin11_87 * sc * sg * tok_g11_99;
                out_pair[0] = value00_100;
                out_pair[1] = value10_102;
                if (token0_80 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token0_80) * M_out + tok_feature)))[0]) = _pk;
                    }
                }
                out_pair[0] = value01_101;
                out_pair[1] = value11_103;
                if (token1_81 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token1_81) * M_out + tok_feature)))[0]) = _pk;
                    }
                }
                int local_token0_104 = lane_1 % 4 * 2 + 40;
                int local_token1_105 = local_token0_104 + 1;
                int token0_106 = token_block * 64 + local_token0_104;
                int token1_107 = token0_106 + 1;
                float tok_sf0_108 = 0.0f;
                float tok_sf1_109 = 0.0f;
                if (token0_106 < valid_rows) {
                    int tok_route0_5 = route_map[tok_slot_base + token0_106];
                    tok_sf0_108 = SFC[tok_route0_5];
                }
                if (token1_107 < valid_rows) {
                    int tok_route1_5 = route_map[tok_slot_base + token1_107];
                    tok_sf1_109 = SFC[tok_route1_5];
                }
                float _max_20 = max_noftz(_tmem_load_0[20] * tok_sf0_108, neg_cl);
                float _min_40 = fminf(_max_20, cl);
                float tok_lin00_110 = _min_40;
                float _max_21 = max_noftz(_tmem_load_0[21] * tok_sf1_109, neg_cl);
                float _min_41 = fminf(_max_21, cl);
                float tok_lin01_111 = _min_41;
                float _max_22 = max_noftz(_tmem_load_1[20] * tok_sf0_108, neg_cl);
                float _min_42 = fminf(_max_22, cl);
                float tok_lin10_112 = _min_42;
                float _max_23 = max_noftz(_tmem_load_1[21] * tok_sf1_109, neg_cl);
                float _min_43 = fminf(_max_23, cl);
                float tok_lin11_113 = _min_43;
                float tok_x00_114 = _tmem_load_0[22] * tok_sf0_108;
                float tok_x01_115 = _tmem_load_0[23] * tok_sf1_109;
                float tok_x10_116 = _tmem_load_1[22] * tok_sf0_108;
                float tok_x11_117 = _tmem_load_1[23] * tok_sf1_109;
                float _exp2_20 = approx_exp2((-(tok_x00_114 * sg)) * 1.4426950408889634f);
                float _rcp_20 = approx_rcp(1.0f + _exp2_20);
                float tok_sig00_118 = _rcp_20;
                float _exp2_21 = approx_exp2((-(tok_x01_115 * sg)) * 1.4426950408889634f);
                float _rcp_21 = approx_rcp(1.0f + _exp2_21);
                float tok_sig01_119 = _rcp_21;
                float _exp2_22 = approx_exp2((-(tok_x10_116 * sg)) * 1.4426950408889634f);
                float _rcp_22 = approx_rcp(1.0f + _exp2_22);
                float tok_sig10_120 = _rcp_22;
                float _exp2_23 = approx_exp2((-(tok_x11_117 * sg)) * 1.4426950408889634f);
                float _rcp_23 = approx_rcp(1.0f + _exp2_23);
                float tok_sig11_121 = _rcp_23;
                float _min_44 = fminf(tok_x00_114 * tok_sig00_118, cl);
                float tok_g00_122 = _min_44;
                float _min_45 = fminf(tok_x01_115 * tok_sig01_119, cl);
                float tok_g01_123 = _min_45;
                float _min_46 = fminf(tok_x10_116 * tok_sig10_120, cl);
                float tok_g10_124 = _min_46;
                float _min_47 = fminf(tok_x11_117 * tok_sig11_121, cl);
                float tok_g11_125 = _min_47;
                float value00_126 = tok_lin00_110 * sc * sg * tok_g00_122;
                float value01_127 = tok_lin01_111 * sc * sg * tok_g01_123;
                float value10_128 = tok_lin10_112 * sc * sg * tok_g10_124;
                float value11_129 = tok_lin11_113 * sc * sg * tok_g11_125;
                out_pair[0] = value00_126;
                out_pair[1] = value10_128;
                if (token0_106 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token0_106) * M_out + tok_feature)))[0]) = _pk;
                    }
                }
                out_pair[0] = value01_127;
                out_pair[1] = value11_129;
                if (token1_107 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token1_107) * M_out + tok_feature)))[0]) = _pk;
                    }
                }
                int local_token0_130 = lane_1 % 4 * 2 + 48;
                int local_token1_131 = local_token0_130 + 1;
                int token0_132 = token_block * 64 + local_token0_130;
                int token1_133 = token0_132 + 1;
                float tok_sf0_134 = 0.0f;
                float tok_sf1_135 = 0.0f;
                if (token0_132 < valid_rows) {
                    int tok_route0_6 = route_map[tok_slot_base + token0_132];
                    tok_sf0_134 = SFC[tok_route0_6];
                }
                if (token1_133 < valid_rows) {
                    int tok_route1_6 = route_map[tok_slot_base + token1_133];
                    tok_sf1_135 = SFC[tok_route1_6];
                }
                float _max_24 = max_noftz(_tmem_load_0[24] * tok_sf0_134, neg_cl);
                float _min_48 = fminf(_max_24, cl);
                float tok_lin00_136 = _min_48;
                float _max_25 = max_noftz(_tmem_load_0[25] * tok_sf1_135, neg_cl);
                float _min_49 = fminf(_max_25, cl);
                float tok_lin01_137 = _min_49;
                float _max_26 = max_noftz(_tmem_load_1[24] * tok_sf0_134, neg_cl);
                float _min_50 = fminf(_max_26, cl);
                float tok_lin10_138 = _min_50;
                float _max_27 = max_noftz(_tmem_load_1[25] * tok_sf1_135, neg_cl);
                float _min_51 = fminf(_max_27, cl);
                float tok_lin11_139 = _min_51;
                float tok_x00_140 = _tmem_load_0[26] * tok_sf0_134;
                float tok_x01_141 = _tmem_load_0[27] * tok_sf1_135;
                float tok_x10_142 = _tmem_load_1[26] * tok_sf0_134;
                float tok_x11_143 = _tmem_load_1[27] * tok_sf1_135;
                float _exp2_24 = approx_exp2((-(tok_x00_140 * sg)) * 1.4426950408889634f);
                float _rcp_24 = approx_rcp(1.0f + _exp2_24);
                float tok_sig00_144 = _rcp_24;
                float _exp2_25 = approx_exp2((-(tok_x01_141 * sg)) * 1.4426950408889634f);
                float _rcp_25 = approx_rcp(1.0f + _exp2_25);
                float tok_sig01_145 = _rcp_25;
                float _exp2_26 = approx_exp2((-(tok_x10_142 * sg)) * 1.4426950408889634f);
                float _rcp_26 = approx_rcp(1.0f + _exp2_26);
                float tok_sig10_146 = _rcp_26;
                float _exp2_27 = approx_exp2((-(tok_x11_143 * sg)) * 1.4426950408889634f);
                float _rcp_27 = approx_rcp(1.0f + _exp2_27);
                float tok_sig11_147 = _rcp_27;
                float _min_52 = fminf(tok_x00_140 * tok_sig00_144, cl);
                float tok_g00_148 = _min_52;
                float _min_53 = fminf(tok_x01_141 * tok_sig01_145, cl);
                float tok_g01_149 = _min_53;
                float _min_54 = fminf(tok_x10_142 * tok_sig10_146, cl);
                float tok_g10_150 = _min_54;
                float _min_55 = fminf(tok_x11_143 * tok_sig11_147, cl);
                float tok_g11_151 = _min_55;
                float value00_152 = tok_lin00_136 * sc * sg * tok_g00_148;
                float value01_153 = tok_lin01_137 * sc * sg * tok_g01_149;
                float value10_154 = tok_lin10_138 * sc * sg * tok_g10_150;
                float value11_155 = tok_lin11_139 * sc * sg * tok_g11_151;
                out_pair[0] = value00_152;
                out_pair[1] = value10_154;
                if (token0_132 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token0_132) * M_out + tok_feature)))[0]) = _pk;
                    }
                }
                out_pair[0] = value01_153;
                out_pair[1] = value11_155;
                if (token1_133 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token1_133) * M_out + tok_feature)))[0]) = _pk;
                    }
                }
                int local_token0_156 = lane_1 % 4 * 2 + 56;
                int local_token1_157 = local_token0_156 + 1;
                int token0_158 = token_block * 64 + local_token0_156;
                int token1_159 = token0_158 + 1;
                float tok_sf0_160 = 0.0f;
                float tok_sf1_161 = 0.0f;
                if (token0_158 < valid_rows) {
                    int tok_route0_7 = route_map[tok_slot_base + token0_158];
                    tok_sf0_160 = SFC[tok_route0_7];
                }
                if (token1_159 < valid_rows) {
                    int tok_route1_7 = route_map[tok_slot_base + token1_159];
                    tok_sf1_161 = SFC[tok_route1_7];
                }
                float _max_28 = max_noftz(_tmem_load_0[28] * tok_sf0_160, neg_cl);
                float _min_56 = fminf(_max_28, cl);
                float tok_lin00_162 = _min_56;
                float _max_29 = max_noftz(_tmem_load_0[29] * tok_sf1_161, neg_cl);
                float _min_57 = fminf(_max_29, cl);
                float tok_lin01_163 = _min_57;
                float _max_30 = max_noftz(_tmem_load_1[28] * tok_sf0_160, neg_cl);
                float _min_58 = fminf(_max_30, cl);
                float tok_lin10_164 = _min_58;
                float _max_31 = max_noftz(_tmem_load_1[29] * tok_sf1_161, neg_cl);
                float _min_59 = fminf(_max_31, cl);
                float tok_lin11_165 = _min_59;
                float tok_x00_166 = _tmem_load_0[30] * tok_sf0_160;
                float tok_x01_167 = _tmem_load_0[31] * tok_sf1_161;
                float tok_x10_168 = _tmem_load_1[30] * tok_sf0_160;
                float tok_x11_169 = _tmem_load_1[31] * tok_sf1_161;
                float _exp2_28 = approx_exp2((-(tok_x00_166 * sg)) * 1.4426950408889634f);
                float _rcp_28 = approx_rcp(1.0f + _exp2_28);
                float tok_sig00_170 = _rcp_28;
                float _exp2_29 = approx_exp2((-(tok_x01_167 * sg)) * 1.4426950408889634f);
                float _rcp_29 = approx_rcp(1.0f + _exp2_29);
                float tok_sig01_171 = _rcp_29;
                float _exp2_30 = approx_exp2((-(tok_x10_168 * sg)) * 1.4426950408889634f);
                float _rcp_30 = approx_rcp(1.0f + _exp2_30);
                float tok_sig10_172 = _rcp_30;
                float _exp2_31 = approx_exp2((-(tok_x11_169 * sg)) * 1.4426950408889634f);
                float _rcp_31 = approx_rcp(1.0f + _exp2_31);
                float tok_sig11_173 = _rcp_31;
                float _min_60 = fminf(tok_x00_166 * tok_sig00_170, cl);
                float tok_g00_174 = _min_60;
                float _min_61 = fminf(tok_x01_167 * tok_sig01_171, cl);
                float tok_g01_175 = _min_61;
                float _min_62 = fminf(tok_x10_168 * tok_sig10_172, cl);
                float tok_g10_176 = _min_62;
                float _min_63 = fminf(tok_x11_169 * tok_sig11_173, cl);
                float tok_g11_177 = _min_63;
                float value00_178 = tok_lin00_162 * sc * sg * tok_g00_174;
                float value01_179 = tok_lin01_163 * sc * sg * tok_g01_175;
                float value10_180 = tok_lin10_164 * sc * sg * tok_g10_176;
                float value11_181 = tok_lin11_165 * sc * sg * tok_g11_177;
                out_pair[0] = value00_178;
                out_pair[1] = value10_180;
                if (token0_158 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token0_158) * M_out + tok_feature)))[0]) = _pk;
                    }
                }
                out_pair[0] = value01_179;
                out_pair[1] = value11_181;
                if (token1_159 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token1_159) * M_out + tok_feature)))[0]) = _pk;
                    }
                }
                int token_block_182 = 2 + warp_group;
                if (epilogue_local_idx == 0) {
                    token_block_182 = (token_block_182 + 3) % 4;
                }
                int accum_col_183 = epilogue_local_idx * 192 + token_block_182 * 64;
                int row_addr_184 = warp_local * 32 << 16;
                float _tmem_load_2[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[31]))
                    : "r"(taddr + (unsigned int)row_addr_184 + (unsigned int)accum_col_183));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float _tmem_load_3[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[31]))
                    : "r"(taddr + (unsigned int)row_addr_184 + 1048576 + (unsigned int)accum_col_183));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int base_row_185 = warp_local * 16 + lane_1 / 4 * 2;
                int tok_feature_186 = (int)m_tile * 64 + base_row_185;
                asm volatile("cp.async.bulk.wait_group.read 0;");
                if (warp_group == 0) {
                    asm volatile("barrier.sync 7, 128;" ::: "memory");
                }
                if (warp_group == 1) {
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                }
                int local_token0_187 = lane_1 % 4 * 2;
                int local_token1_188 = local_token0_187 + 1;
                int token0_189 = token_block_182 * 64 + local_token0_187;
                int token1_190 = token0_189 + 1;
                float tok_sf0_191 = 0.0f;
                float tok_sf1_192 = 0.0f;
                if (token0_189 < valid_rows) {
                    int tok_route0_8 = route_map[tok_slot_base + token0_189];
                    tok_sf0_191 = SFC[tok_route0_8];
                }
                if (token1_190 < valid_rows) {
                    int tok_route1_8 = route_map[tok_slot_base + token1_190];
                    tok_sf1_192 = SFC[tok_route1_8];
                }
                float _max_32 = max_noftz(_tmem_load_2[0] * tok_sf0_191, neg_cl);
                float _min_64 = fminf(_max_32, cl);
                float tok_lin00_193 = _min_64;
                float _max_33 = max_noftz(_tmem_load_2[1] * tok_sf1_192, neg_cl);
                float _min_65 = fminf(_max_33, cl);
                float tok_lin01_194 = _min_65;
                float _max_34 = max_noftz(_tmem_load_3[0] * tok_sf0_191, neg_cl);
                float _min_66 = fminf(_max_34, cl);
                float tok_lin10_195 = _min_66;
                float _max_35 = max_noftz(_tmem_load_3[1] * tok_sf1_192, neg_cl);
                float _min_67 = fminf(_max_35, cl);
                float tok_lin11_196 = _min_67;
                float tok_x00_197 = _tmem_load_2[2] * tok_sf0_191;
                float tok_x01_198 = _tmem_load_2[3] * tok_sf1_192;
                float tok_x10_199 = _tmem_load_3[2] * tok_sf0_191;
                float tok_x11_200 = _tmem_load_3[3] * tok_sf1_192;
                float _exp2_32 = approx_exp2((-(tok_x00_197 * sg)) * 1.4426950408889634f);
                float _rcp_32 = approx_rcp(1.0f + _exp2_32);
                float tok_sig00_201 = _rcp_32;
                float _exp2_33 = approx_exp2((-(tok_x01_198 * sg)) * 1.4426950408889634f);
                float _rcp_33 = approx_rcp(1.0f + _exp2_33);
                float tok_sig01_202 = _rcp_33;
                float _exp2_34 = approx_exp2((-(tok_x10_199 * sg)) * 1.4426950408889634f);
                float _rcp_34 = approx_rcp(1.0f + _exp2_34);
                float tok_sig10_203 = _rcp_34;
                float _exp2_35 = approx_exp2((-(tok_x11_200 * sg)) * 1.4426950408889634f);
                float _rcp_35 = approx_rcp(1.0f + _exp2_35);
                float tok_sig11_204 = _rcp_35;
                float _min_68 = fminf(tok_x00_197 * tok_sig00_201, cl);
                float tok_g00_205 = _min_68;
                float _min_69 = fminf(tok_x01_198 * tok_sig01_202, cl);
                float tok_g01_206 = _min_69;
                float _min_70 = fminf(tok_x10_199 * tok_sig10_203, cl);
                float tok_g10_207 = _min_70;
                float _min_71 = fminf(tok_x11_200 * tok_sig11_204, cl);
                float tok_g11_208 = _min_71;
                float value00_209 = tok_lin00_193 * sc * sg * tok_g00_205;
                float value01_210 = tok_lin01_194 * sc * sg * tok_g01_206;
                float value10_211 = tok_lin10_195 * sc * sg * tok_g10_207;
                float value11_212 = tok_lin11_196 * sc * sg * tok_g11_208;
                out_pair[0] = value00_209;
                out_pair[1] = value10_211;
                if (token0_189 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token0_189) * M_out + tok_feature_186)))[0]) = _pk;
                    }
                }
                out_pair[0] = value01_210;
                out_pair[1] = value11_212;
                if (token1_190 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token1_190) * M_out + tok_feature_186)))[0]) = _pk;
                    }
                }
                int local_token0_213 = lane_1 % 4 * 2 + 8;
                int local_token1_214 = local_token0_213 + 1;
                int token0_215 = token_block_182 * 64 + local_token0_213;
                int token1_216 = token0_215 + 1;
                float tok_sf0_217 = 0.0f;
                float tok_sf1_218 = 0.0f;
                if (token0_215 < valid_rows) {
                    int tok_route0_9 = route_map[tok_slot_base + token0_215];
                    tok_sf0_217 = SFC[tok_route0_9];
                }
                if (token1_216 < valid_rows) {
                    int tok_route1_9 = route_map[tok_slot_base + token1_216];
                    tok_sf1_218 = SFC[tok_route1_9];
                }
                float _max_36 = max_noftz(_tmem_load_2[4] * tok_sf0_217, neg_cl);
                float _min_72 = fminf(_max_36, cl);
                float tok_lin00_219 = _min_72;
                float _max_37 = max_noftz(_tmem_load_2[5] * tok_sf1_218, neg_cl);
                float _min_73 = fminf(_max_37, cl);
                float tok_lin01_220 = _min_73;
                float _max_38 = max_noftz(_tmem_load_3[4] * tok_sf0_217, neg_cl);
                float _min_74 = fminf(_max_38, cl);
                float tok_lin10_221 = _min_74;
                float _max_39 = max_noftz(_tmem_load_3[5] * tok_sf1_218, neg_cl);
                float _min_75 = fminf(_max_39, cl);
                float tok_lin11_222 = _min_75;
                float tok_x00_223 = _tmem_load_2[6] * tok_sf0_217;
                float tok_x01_224 = _tmem_load_2[7] * tok_sf1_218;
                float tok_x10_225 = _tmem_load_3[6] * tok_sf0_217;
                float tok_x11_226 = _tmem_load_3[7] * tok_sf1_218;
                float _exp2_36 = approx_exp2((-(tok_x00_223 * sg)) * 1.4426950408889634f);
                float _rcp_36 = approx_rcp(1.0f + _exp2_36);
                float tok_sig00_227 = _rcp_36;
                float _exp2_37 = approx_exp2((-(tok_x01_224 * sg)) * 1.4426950408889634f);
                float _rcp_37 = approx_rcp(1.0f + _exp2_37);
                float tok_sig01_228 = _rcp_37;
                float _exp2_38 = approx_exp2((-(tok_x10_225 * sg)) * 1.4426950408889634f);
                float _rcp_38 = approx_rcp(1.0f + _exp2_38);
                float tok_sig10_229 = _rcp_38;
                float _exp2_39 = approx_exp2((-(tok_x11_226 * sg)) * 1.4426950408889634f);
                float _rcp_39 = approx_rcp(1.0f + _exp2_39);
                float tok_sig11_230 = _rcp_39;
                float _min_76 = fminf(tok_x00_223 * tok_sig00_227, cl);
                float tok_g00_231 = _min_76;
                float _min_77 = fminf(tok_x01_224 * tok_sig01_228, cl);
                float tok_g01_232 = _min_77;
                float _min_78 = fminf(tok_x10_225 * tok_sig10_229, cl);
                float tok_g10_233 = _min_78;
                float _min_79 = fminf(tok_x11_226 * tok_sig11_230, cl);
                float tok_g11_234 = _min_79;
                float value00_235 = tok_lin00_219 * sc * sg * tok_g00_231;
                float value01_236 = tok_lin01_220 * sc * sg * tok_g01_232;
                float value10_237 = tok_lin10_221 * sc * sg * tok_g10_233;
                float value11_238 = tok_lin11_222 * sc * sg * tok_g11_234;
                out_pair[0] = value00_235;
                out_pair[1] = value10_237;
                if (token0_215 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token0_215) * M_out + tok_feature_186)))[0]) = _pk;
                    }
                }
                out_pair[0] = value01_236;
                out_pair[1] = value11_238;
                if (token1_216 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token1_216) * M_out + tok_feature_186)))[0]) = _pk;
                    }
                }
                int local_token0_239 = lane_1 % 4 * 2 + 16;
                int local_token1_240 = local_token0_239 + 1;
                int token0_241 = token_block_182 * 64 + local_token0_239;
                int token1_242 = token0_241 + 1;
                float tok_sf0_243 = 0.0f;
                float tok_sf1_244 = 0.0f;
                if (token0_241 < valid_rows) {
                    int tok_route0_10 = route_map[tok_slot_base + token0_241];
                    tok_sf0_243 = SFC[tok_route0_10];
                }
                if (token1_242 < valid_rows) {
                    int tok_route1_10 = route_map[tok_slot_base + token1_242];
                    tok_sf1_244 = SFC[tok_route1_10];
                }
                float _max_40 = max_noftz(_tmem_load_2[8] * tok_sf0_243, neg_cl);
                float _min_80 = fminf(_max_40, cl);
                float tok_lin00_245 = _min_80;
                float _max_41 = max_noftz(_tmem_load_2[9] * tok_sf1_244, neg_cl);
                float _min_81 = fminf(_max_41, cl);
                float tok_lin01_246 = _min_81;
                float _max_42 = max_noftz(_tmem_load_3[8] * tok_sf0_243, neg_cl);
                float _min_82 = fminf(_max_42, cl);
                float tok_lin10_247 = _min_82;
                float _max_43 = max_noftz(_tmem_load_3[9] * tok_sf1_244, neg_cl);
                float _min_83 = fminf(_max_43, cl);
                float tok_lin11_248 = _min_83;
                float tok_x00_249 = _tmem_load_2[10] * tok_sf0_243;
                float tok_x01_250 = _tmem_load_2[11] * tok_sf1_244;
                float tok_x10_251 = _tmem_load_3[10] * tok_sf0_243;
                float tok_x11_252 = _tmem_load_3[11] * tok_sf1_244;
                float _exp2_40 = approx_exp2((-(tok_x00_249 * sg)) * 1.4426950408889634f);
                float _rcp_40 = approx_rcp(1.0f + _exp2_40);
                float tok_sig00_253 = _rcp_40;
                float _exp2_41 = approx_exp2((-(tok_x01_250 * sg)) * 1.4426950408889634f);
                float _rcp_41 = approx_rcp(1.0f + _exp2_41);
                float tok_sig01_254 = _rcp_41;
                float _exp2_42 = approx_exp2((-(tok_x10_251 * sg)) * 1.4426950408889634f);
                float _rcp_42 = approx_rcp(1.0f + _exp2_42);
                float tok_sig10_255 = _rcp_42;
                float _exp2_43 = approx_exp2((-(tok_x11_252 * sg)) * 1.4426950408889634f);
                float _rcp_43 = approx_rcp(1.0f + _exp2_43);
                float tok_sig11_256 = _rcp_43;
                float _min_84 = fminf(tok_x00_249 * tok_sig00_253, cl);
                float tok_g00_257 = _min_84;
                float _min_85 = fminf(tok_x01_250 * tok_sig01_254, cl);
                float tok_g01_258 = _min_85;
                float _min_86 = fminf(tok_x10_251 * tok_sig10_255, cl);
                float tok_g10_259 = _min_86;
                float _min_87 = fminf(tok_x11_252 * tok_sig11_256, cl);
                float tok_g11_260 = _min_87;
                float value00_261 = tok_lin00_245 * sc * sg * tok_g00_257;
                float value01_262 = tok_lin01_246 * sc * sg * tok_g01_258;
                float value10_263 = tok_lin10_247 * sc * sg * tok_g10_259;
                float value11_264 = tok_lin11_248 * sc * sg * tok_g11_260;
                out_pair[0] = value00_261;
                out_pair[1] = value10_263;
                if (token0_241 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token0_241) * M_out + tok_feature_186)))[0]) = _pk;
                    }
                }
                out_pair[0] = value01_262;
                out_pair[1] = value11_264;
                if (token1_242 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token1_242) * M_out + tok_feature_186)))[0]) = _pk;
                    }
                }
                int local_token0_265 = lane_1 % 4 * 2 + 24;
                int local_token1_266 = local_token0_265 + 1;
                int token0_267 = token_block_182 * 64 + local_token0_265;
                int token1_268 = token0_267 + 1;
                float tok_sf0_269 = 0.0f;
                float tok_sf1_270 = 0.0f;
                if (token0_267 < valid_rows) {
                    int tok_route0_11 = route_map[tok_slot_base + token0_267];
                    tok_sf0_269 = SFC[tok_route0_11];
                }
                if (token1_268 < valid_rows) {
                    int tok_route1_11 = route_map[tok_slot_base + token1_268];
                    tok_sf1_270 = SFC[tok_route1_11];
                }
                float _max_44 = max_noftz(_tmem_load_2[12] * tok_sf0_269, neg_cl);
                float _min_88 = fminf(_max_44, cl);
                float tok_lin00_271 = _min_88;
                float _max_45 = max_noftz(_tmem_load_2[13] * tok_sf1_270, neg_cl);
                float _min_89 = fminf(_max_45, cl);
                float tok_lin01_272 = _min_89;
                float _max_46 = max_noftz(_tmem_load_3[12] * tok_sf0_269, neg_cl);
                float _min_90 = fminf(_max_46, cl);
                float tok_lin10_273 = _min_90;
                float _max_47 = max_noftz(_tmem_load_3[13] * tok_sf1_270, neg_cl);
                float _min_91 = fminf(_max_47, cl);
                float tok_lin11_274 = _min_91;
                float tok_x00_275 = _tmem_load_2[14] * tok_sf0_269;
                float tok_x01_276 = _tmem_load_2[15] * tok_sf1_270;
                float tok_x10_277 = _tmem_load_3[14] * tok_sf0_269;
                float tok_x11_278 = _tmem_load_3[15] * tok_sf1_270;
                float _exp2_44 = approx_exp2((-(tok_x00_275 * sg)) * 1.4426950408889634f);
                float _rcp_44 = approx_rcp(1.0f + _exp2_44);
                float tok_sig00_279 = _rcp_44;
                float _exp2_45 = approx_exp2((-(tok_x01_276 * sg)) * 1.4426950408889634f);
                float _rcp_45 = approx_rcp(1.0f + _exp2_45);
                float tok_sig01_280 = _rcp_45;
                float _exp2_46 = approx_exp2((-(tok_x10_277 * sg)) * 1.4426950408889634f);
                float _rcp_46 = approx_rcp(1.0f + _exp2_46);
                float tok_sig10_281 = _rcp_46;
                float _exp2_47 = approx_exp2((-(tok_x11_278 * sg)) * 1.4426950408889634f);
                float _rcp_47 = approx_rcp(1.0f + _exp2_47);
                float tok_sig11_282 = _rcp_47;
                float _min_92 = fminf(tok_x00_275 * tok_sig00_279, cl);
                float tok_g00_283 = _min_92;
                float _min_93 = fminf(tok_x01_276 * tok_sig01_280, cl);
                float tok_g01_284 = _min_93;
                float _min_94 = fminf(tok_x10_277 * tok_sig10_281, cl);
                float tok_g10_285 = _min_94;
                float _min_95 = fminf(tok_x11_278 * tok_sig11_282, cl);
                float tok_g11_286 = _min_95;
                float value00_287 = tok_lin00_271 * sc * sg * tok_g00_283;
                float value01_288 = tok_lin01_272 * sc * sg * tok_g01_284;
                float value10_289 = tok_lin10_273 * sc * sg * tok_g10_285;
                float value11_290 = tok_lin11_274 * sc * sg * tok_g11_286;
                out_pair[0] = value00_287;
                out_pair[1] = value10_289;
                if (token0_267 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token0_267) * M_out + tok_feature_186)))[0]) = _pk;
                    }
                }
                out_pair[0] = value01_288;
                out_pair[1] = value11_290;
                if (token1_268 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token1_268) * M_out + tok_feature_186)))[0]) = _pk;
                    }
                }
                int local_token0_291 = lane_1 % 4 * 2 + 32;
                int local_token1_292 = local_token0_291 + 1;
                int token0_293 = token_block_182 * 64 + local_token0_291;
                int token1_294 = token0_293 + 1;
                float tok_sf0_295 = 0.0f;
                float tok_sf1_296 = 0.0f;
                if (token0_293 < valid_rows) {
                    int tok_route0_12 = route_map[tok_slot_base + token0_293];
                    tok_sf0_295 = SFC[tok_route0_12];
                }
                if (token1_294 < valid_rows) {
                    int tok_route1_12 = route_map[tok_slot_base + token1_294];
                    tok_sf1_296 = SFC[tok_route1_12];
                }
                float _max_48 = max_noftz(_tmem_load_2[16] * tok_sf0_295, neg_cl);
                float _min_96 = fminf(_max_48, cl);
                float tok_lin00_297 = _min_96;
                float _max_49 = max_noftz(_tmem_load_2[17] * tok_sf1_296, neg_cl);
                float _min_97 = fminf(_max_49, cl);
                float tok_lin01_298 = _min_97;
                float _max_50 = max_noftz(_tmem_load_3[16] * tok_sf0_295, neg_cl);
                float _min_98 = fminf(_max_50, cl);
                float tok_lin10_299 = _min_98;
                float _max_51 = max_noftz(_tmem_load_3[17] * tok_sf1_296, neg_cl);
                float _min_99 = fminf(_max_51, cl);
                float tok_lin11_300 = _min_99;
                float tok_x00_301 = _tmem_load_2[18] * tok_sf0_295;
                float tok_x01_302 = _tmem_load_2[19] * tok_sf1_296;
                float tok_x10_303 = _tmem_load_3[18] * tok_sf0_295;
                float tok_x11_304 = _tmem_load_3[19] * tok_sf1_296;
                float _exp2_48 = approx_exp2((-(tok_x00_301 * sg)) * 1.4426950408889634f);
                float _rcp_48 = approx_rcp(1.0f + _exp2_48);
                float tok_sig00_305 = _rcp_48;
                float _exp2_49 = approx_exp2((-(tok_x01_302 * sg)) * 1.4426950408889634f);
                float _rcp_49 = approx_rcp(1.0f + _exp2_49);
                float tok_sig01_306 = _rcp_49;
                float _exp2_50 = approx_exp2((-(tok_x10_303 * sg)) * 1.4426950408889634f);
                float _rcp_50 = approx_rcp(1.0f + _exp2_50);
                float tok_sig10_307 = _rcp_50;
                float _exp2_51 = approx_exp2((-(tok_x11_304 * sg)) * 1.4426950408889634f);
                float _rcp_51 = approx_rcp(1.0f + _exp2_51);
                float tok_sig11_308 = _rcp_51;
                float _min_100 = fminf(tok_x00_301 * tok_sig00_305, cl);
                float tok_g00_309 = _min_100;
                float _min_101 = fminf(tok_x01_302 * tok_sig01_306, cl);
                float tok_g01_310 = _min_101;
                float _min_102 = fminf(tok_x10_303 * tok_sig10_307, cl);
                float tok_g10_311 = _min_102;
                float _min_103 = fminf(tok_x11_304 * tok_sig11_308, cl);
                float tok_g11_312 = _min_103;
                float value00_313 = tok_lin00_297 * sc * sg * tok_g00_309;
                float value01_314 = tok_lin01_298 * sc * sg * tok_g01_310;
                float value10_315 = tok_lin10_299 * sc * sg * tok_g10_311;
                float value11_316 = tok_lin11_300 * sc * sg * tok_g11_312;
                out_pair[0] = value00_313;
                out_pair[1] = value10_315;
                if (token0_293 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token0_293) * M_out + tok_feature_186)))[0]) = _pk;
                    }
                }
                out_pair[0] = value01_314;
                out_pair[1] = value11_316;
                if (token1_294 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token1_294) * M_out + tok_feature_186)))[0]) = _pk;
                    }
                }
                int local_token0_317 = lane_1 % 4 * 2 + 40;
                int local_token1_318 = local_token0_317 + 1;
                int token0_319 = token_block_182 * 64 + local_token0_317;
                int token1_320 = token0_319 + 1;
                float tok_sf0_321 = 0.0f;
                float tok_sf1_322 = 0.0f;
                if (token0_319 < valid_rows) {
                    int tok_route0_13 = route_map[tok_slot_base + token0_319];
                    tok_sf0_321 = SFC[tok_route0_13];
                }
                if (token1_320 < valid_rows) {
                    int tok_route1_13 = route_map[tok_slot_base + token1_320];
                    tok_sf1_322 = SFC[tok_route1_13];
                }
                float _max_52 = max_noftz(_tmem_load_2[20] * tok_sf0_321, neg_cl);
                float _min_104 = fminf(_max_52, cl);
                float tok_lin00_323 = _min_104;
                float _max_53 = max_noftz(_tmem_load_2[21] * tok_sf1_322, neg_cl);
                float _min_105 = fminf(_max_53, cl);
                float tok_lin01_324 = _min_105;
                float _max_54 = max_noftz(_tmem_load_3[20] * tok_sf0_321, neg_cl);
                float _min_106 = fminf(_max_54, cl);
                float tok_lin10_325 = _min_106;
                float _max_55 = max_noftz(_tmem_load_3[21] * tok_sf1_322, neg_cl);
                float _min_107 = fminf(_max_55, cl);
                float tok_lin11_326 = _min_107;
                float tok_x00_327 = _tmem_load_2[22] * tok_sf0_321;
                float tok_x01_328 = _tmem_load_2[23] * tok_sf1_322;
                float tok_x10_329 = _tmem_load_3[22] * tok_sf0_321;
                float tok_x11_330 = _tmem_load_3[23] * tok_sf1_322;
                float _exp2_52 = approx_exp2((-(tok_x00_327 * sg)) * 1.4426950408889634f);
                float _rcp_52 = approx_rcp(1.0f + _exp2_52);
                float tok_sig00_331 = _rcp_52;
                float _exp2_53 = approx_exp2((-(tok_x01_328 * sg)) * 1.4426950408889634f);
                float _rcp_53 = approx_rcp(1.0f + _exp2_53);
                float tok_sig01_332 = _rcp_53;
                float _exp2_54 = approx_exp2((-(tok_x10_329 * sg)) * 1.4426950408889634f);
                float _rcp_54 = approx_rcp(1.0f + _exp2_54);
                float tok_sig10_333 = _rcp_54;
                float _exp2_55 = approx_exp2((-(tok_x11_330 * sg)) * 1.4426950408889634f);
                float _rcp_55 = approx_rcp(1.0f + _exp2_55);
                float tok_sig11_334 = _rcp_55;
                float _min_108 = fminf(tok_x00_327 * tok_sig00_331, cl);
                float tok_g00_335 = _min_108;
                float _min_109 = fminf(tok_x01_328 * tok_sig01_332, cl);
                float tok_g01_336 = _min_109;
                float _min_110 = fminf(tok_x10_329 * tok_sig10_333, cl);
                float tok_g10_337 = _min_110;
                float _min_111 = fminf(tok_x11_330 * tok_sig11_334, cl);
                float tok_g11_338 = _min_111;
                float value00_339 = tok_lin00_323 * sc * sg * tok_g00_335;
                float value01_340 = tok_lin01_324 * sc * sg * tok_g01_336;
                float value10_341 = tok_lin10_325 * sc * sg * tok_g10_337;
                float value11_342 = tok_lin11_326 * sc * sg * tok_g11_338;
                out_pair[0] = value00_339;
                out_pair[1] = value10_341;
                if (token0_319 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token0_319) * M_out + tok_feature_186)))[0]) = _pk;
                    }
                }
                out_pair[0] = value01_340;
                out_pair[1] = value11_342;
                if (token1_320 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token1_320) * M_out + tok_feature_186)))[0]) = _pk;
                    }
                }
                int local_token0_343 = lane_1 % 4 * 2 + 48;
                int local_token1_344 = local_token0_343 + 1;
                int token0_345 = token_block_182 * 64 + local_token0_343;
                int token1_346 = token0_345 + 1;
                float tok_sf0_347 = 0.0f;
                float tok_sf1_348 = 0.0f;
                if (token0_345 < valid_rows) {
                    int tok_route0_14 = route_map[tok_slot_base + token0_345];
                    tok_sf0_347 = SFC[tok_route0_14];
                }
                if (token1_346 < valid_rows) {
                    int tok_route1_14 = route_map[tok_slot_base + token1_346];
                    tok_sf1_348 = SFC[tok_route1_14];
                }
                float _max_56 = max_noftz(_tmem_load_2[24] * tok_sf0_347, neg_cl);
                float _min_112 = fminf(_max_56, cl);
                float tok_lin00_349 = _min_112;
                float _max_57 = max_noftz(_tmem_load_2[25] * tok_sf1_348, neg_cl);
                float _min_113 = fminf(_max_57, cl);
                float tok_lin01_350 = _min_113;
                float _max_58 = max_noftz(_tmem_load_3[24] * tok_sf0_347, neg_cl);
                float _min_114 = fminf(_max_58, cl);
                float tok_lin10_351 = _min_114;
                float _max_59 = max_noftz(_tmem_load_3[25] * tok_sf1_348, neg_cl);
                float _min_115 = fminf(_max_59, cl);
                float tok_lin11_352 = _min_115;
                float tok_x00_353 = _tmem_load_2[26] * tok_sf0_347;
                float tok_x01_354 = _tmem_load_2[27] * tok_sf1_348;
                float tok_x10_355 = _tmem_load_3[26] * tok_sf0_347;
                float tok_x11_356 = _tmem_load_3[27] * tok_sf1_348;
                float _exp2_56 = approx_exp2((-(tok_x00_353 * sg)) * 1.4426950408889634f);
                float _rcp_56 = approx_rcp(1.0f + _exp2_56);
                float tok_sig00_357 = _rcp_56;
                float _exp2_57 = approx_exp2((-(tok_x01_354 * sg)) * 1.4426950408889634f);
                float _rcp_57 = approx_rcp(1.0f + _exp2_57);
                float tok_sig01_358 = _rcp_57;
                float _exp2_58 = approx_exp2((-(tok_x10_355 * sg)) * 1.4426950408889634f);
                float _rcp_58 = approx_rcp(1.0f + _exp2_58);
                float tok_sig10_359 = _rcp_58;
                float _exp2_59 = approx_exp2((-(tok_x11_356 * sg)) * 1.4426950408889634f);
                float _rcp_59 = approx_rcp(1.0f + _exp2_59);
                float tok_sig11_360 = _rcp_59;
                float _min_116 = fminf(tok_x00_353 * tok_sig00_357, cl);
                float tok_g00_361 = _min_116;
                float _min_117 = fminf(tok_x01_354 * tok_sig01_358, cl);
                float tok_g01_362 = _min_117;
                float _min_118 = fminf(tok_x10_355 * tok_sig10_359, cl);
                float tok_g10_363 = _min_118;
                float _min_119 = fminf(tok_x11_356 * tok_sig11_360, cl);
                float tok_g11_364 = _min_119;
                float value00_365 = tok_lin00_349 * sc * sg * tok_g00_361;
                float value01_366 = tok_lin01_350 * sc * sg * tok_g01_362;
                float value10_367 = tok_lin10_351 * sc * sg * tok_g10_363;
                float value11_368 = tok_lin11_352 * sc * sg * tok_g11_364;
                out_pair[0] = value00_365;
                out_pair[1] = value10_367;
                if (token0_345 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token0_345) * M_out + tok_feature_186)))[0]) = _pk;
                    }
                }
                out_pair[0] = value01_366;
                out_pair[1] = value11_368;
                if (token1_346 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token1_346) * M_out + tok_feature_186)))[0]) = _pk;
                    }
                }
                int local_token0_369 = lane_1 % 4 * 2 + 56;
                int local_token1_370 = local_token0_369 + 1;
                int token0_371 = token_block_182 * 64 + local_token0_369;
                int token1_372 = token0_371 + 1;
                float tok_sf0_373 = 0.0f;
                float tok_sf1_374 = 0.0f;
                if (token0_371 < valid_rows) {
                    int tok_route0_15 = route_map[tok_slot_base + token0_371];
                    tok_sf0_373 = SFC[tok_route0_15];
                }
                if (token1_372 < valid_rows) {
                    int tok_route1_15 = route_map[tok_slot_base + token1_372];
                    tok_sf1_374 = SFC[tok_route1_15];
                }
                float _max_60 = max_noftz(_tmem_load_2[28] * tok_sf0_373, neg_cl);
                float _min_120 = fminf(_max_60, cl);
                float tok_lin00_375 = _min_120;
                float _max_61 = max_noftz(_tmem_load_2[29] * tok_sf1_374, neg_cl);
                float _min_121 = fminf(_max_61, cl);
                float tok_lin01_376 = _min_121;
                float _max_62 = max_noftz(_tmem_load_3[28] * tok_sf0_373, neg_cl);
                float _min_122 = fminf(_max_62, cl);
                float tok_lin10_377 = _min_122;
                float _max_63 = max_noftz(_tmem_load_3[29] * tok_sf1_374, neg_cl);
                float _min_123 = fminf(_max_63, cl);
                float tok_lin11_378 = _min_123;
                float tok_x00_379 = _tmem_load_2[30] * tok_sf0_373;
                float tok_x01_380 = _tmem_load_2[31] * tok_sf1_374;
                float tok_x10_381 = _tmem_load_3[30] * tok_sf0_373;
                float tok_x11_382 = _tmem_load_3[31] * tok_sf1_374;
                float _exp2_60 = approx_exp2((-(tok_x00_379 * sg)) * 1.4426950408889634f);
                float _rcp_60 = approx_rcp(1.0f + _exp2_60);
                float tok_sig00_383 = _rcp_60;
                float _exp2_61 = approx_exp2((-(tok_x01_380 * sg)) * 1.4426950408889634f);
                float _rcp_61 = approx_rcp(1.0f + _exp2_61);
                float tok_sig01_384 = _rcp_61;
                float _exp2_62 = approx_exp2((-(tok_x10_381 * sg)) * 1.4426950408889634f);
                float _rcp_62 = approx_rcp(1.0f + _exp2_62);
                float tok_sig10_385 = _rcp_62;
                float _exp2_63 = approx_exp2((-(tok_x11_382 * sg)) * 1.4426950408889634f);
                float _rcp_63 = approx_rcp(1.0f + _exp2_63);
                float tok_sig11_386 = _rcp_63;
                float _min_124 = fminf(tok_x00_379 * tok_sig00_383, cl);
                float tok_g00_387 = _min_124;
                float _min_125 = fminf(tok_x01_380 * tok_sig01_384, cl);
                float tok_g01_388 = _min_125;
                float _min_126 = fminf(tok_x10_381 * tok_sig10_385, cl);
                float tok_g10_389 = _min_126;
                float _min_127 = fminf(tok_x11_382 * tok_sig11_386, cl);
                float tok_g11_390 = _min_127;
                float value00_391 = tok_lin00_375 * sc * sg * tok_g00_387;
                float value01_392 = tok_lin01_376 * sc * sg * tok_g01_388;
                float value10_393 = tok_lin10_377 * sc * sg * tok_g10_389;
                float value11_394 = tok_lin11_378 * sc * sg * tok_g11_390;
                out_pair[0] = value00_391;
                out_pair[1] = value10_393;
                if (token0_371 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token0_371) * M_out + tok_feature_186)))[0]) = _pk;
                    }
                }
                out_pair[0] = value01_392;
                out_pair[1] = value11_394;
                if (token1_372 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token1_372) * M_out + tok_feature_186)))[0]) = _pk;
                    }
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
