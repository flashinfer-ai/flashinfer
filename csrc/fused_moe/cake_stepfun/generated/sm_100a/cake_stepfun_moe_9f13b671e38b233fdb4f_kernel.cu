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
kernel_cake_stepfun_moe_9f13b671e38b233fdb4f(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, int* __restrict__ SFB, __nv_bfloat16* __restrict__ C, float* __restrict__ SFC, int* __restrict__ route_map, int* __restrict__ tile_expert, int* __restrict__ tile_mn_limit, int* __restrict__ num_non_exiting_ctas, float* __restrict__ scale_c, float* __restrict__ scale_gate, float* __restrict__ clamp_limit, float* __restrict__ act_alpha, float* __restrict__ act_beta, int M_out, int K, int grid_m, int grid_n, int K_tiles)
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
            // mma_free: 1 barriers, init_count=512
            mbarrier_init(smem + 168, 512);
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
                float _neg_0;
                asm("neg.ftz.f32 %0, %1;" : "=f"(_neg_0) : "f"(cl));
                float neg_cl = _neg_0;
                float beta_sg = be * sg;
                float alpha_sg = al * sg;
                float _mul_0;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_0) : "f"(-1.4426950216293335f), "f"(sg));
                float neg_fused = _mul_0;
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
                {
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
                float _mul_1;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_1) : "f"(_tmem_load_0[0]), "f"(tok_sf0));
                float _max_0;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_0) : "f"(_mul_1), "f"(neg_cl));
                float _min_0;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_0) : "f"(_max_0), "f"(cl));
                float tok_lin00 = _min_0;
                float _mul_2;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_2) : "f"(_tmem_load_0[1]), "f"(tok_sf1));
                float _max_1;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_1) : "f"(_mul_2), "f"(neg_cl));
                float _min_1;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_1) : "f"(_max_1), "f"(cl));
                float tok_lin01 = _min_1;
                float _mul_3;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_3) : "f"(_tmem_load_1[0]), "f"(tok_sf0));
                float _max_2;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_2) : "f"(_mul_3), "f"(neg_cl));
                float _min_2;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_2) : "f"(_max_2), "f"(cl));
                float tok_lin10 = _min_2;
                float _mul_4;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_4) : "f"(_tmem_load_1[1]), "f"(tok_sf1));
                float _max_3;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_3) : "f"(_mul_4), "f"(neg_cl));
                float _min_3;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_3) : "f"(_max_3), "f"(cl));
                float tok_lin11 = _min_3;
                float _mul_5;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_5) : "f"(_tmem_load_0[2]), "f"(tok_sf0));
                float tok_x00 = _mul_5;
                float _mul_6;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_6) : "f"(_tmem_load_0[3]), "f"(tok_sf1));
                float tok_x01 = _mul_6;
                float _mul_7;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_7) : "f"(_tmem_load_1[2]), "f"(tok_sf0));
                float tok_x10 = _mul_7;
                float _mul_8;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_8) : "f"(_tmem_load_1[3]), "f"(tok_sf1));
                float tok_x11 = _mul_8;
                float _mul_9;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_9) : "f"(tok_x00), "f"(neg_fused));
                float _exp2_0 = approx_exp2(_mul_9);
                float _add_0;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_0) : "f"(1.0f), "f"(_exp2_0));
                float _rcp_0 = approx_rcp(_add_0);
                float tok_sig00 = _rcp_0;
                float _mul_10;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_10) : "f"(tok_x01), "f"(neg_fused));
                float _exp2_1 = approx_exp2(_mul_10);
                float _add_1;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_1) : "f"(1.0f), "f"(_exp2_1));
                float _rcp_1 = approx_rcp(_add_1);
                float tok_sig01 = _rcp_1;
                float _mul_11;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_11) : "f"(tok_x10), "f"(neg_fused));
                float _exp2_2 = approx_exp2(_mul_11);
                float _add_2;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_2) : "f"(1.0f), "f"(_exp2_2));
                float _rcp_2 = approx_rcp(_add_2);
                float tok_sig10 = _rcp_2;
                float _mul_12;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_12) : "f"(tok_x11), "f"(neg_fused));
                float _exp2_3 = approx_exp2(_mul_12);
                float _add_3;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_3) : "f"(1.0f), "f"(_exp2_3));
                float _rcp_3 = approx_rcp(_add_3);
                float tok_sig11 = _rcp_3;
                float _mul_13;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_13) : "f"(tok_x00), "f"(tok_sig00));
                float _min_4;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_4) : "f"(_mul_13), "f"(cl));
                float tok_g00 = _min_4;
                float _mul_14;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_14) : "f"(tok_x01), "f"(tok_sig01));
                float _min_5;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_5) : "f"(_mul_14), "f"(cl));
                float tok_g01 = _min_5;
                float _mul_15;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_15) : "f"(tok_x10), "f"(tok_sig10));
                float _min_6;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_6) : "f"(_mul_15), "f"(cl));
                float tok_g10 = _min_6;
                float _mul_16;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_16) : "f"(tok_x11), "f"(tok_sig11));
                float _min_7;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_7) : "f"(_mul_16), "f"(cl));
                float tok_g11 = _min_7;
                float _fma_0;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_0) : "f"(tok_lin00), "f"(sg), "f"(0.0f));
                float _mul_17;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_17) : "f"(_fma_0), "f"(tok_g00));
                float tok_acc00 = _mul_17;
                float _fma_1;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_1) : "f"(tok_lin01), "f"(sg), "f"(0.0f));
                float _mul_18;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_18) : "f"(_fma_1), "f"(tok_g01));
                float tok_acc01 = _mul_18;
                float _fma_2;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_2) : "f"(tok_lin10), "f"(sg), "f"(0.0f));
                float _mul_19;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_19) : "f"(_fma_2), "f"(tok_g10));
                float tok_acc10 = _mul_19;
                float _fma_3;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_3) : "f"(tok_lin11), "f"(sg), "f"(0.0f));
                float _mul_20;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_20) : "f"(_fma_3), "f"(tok_g11));
                float tok_acc11 = _mul_20;
                float2 _f2_0 = make_float2(sc, sc);
                float2 tok_sc = _f2_0;
                float2 _f2_1 = make_float2(tok_acc00, tok_acc10);
                float2 _mul_f32x2_0;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_0) : "l"(*(const unsigned long long*)&_f2_1), "l"(*(const unsigned long long*)&tok_sc));
                float2 tok_out0 = _mul_f32x2_0;
                float2 _f2_2 = make_float2(tok_acc01, tok_acc11);
                float2 _mul_f32x2_1;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_1) : "l"(*(const unsigned long long*)&_f2_2), "l"(*(const unsigned long long*)&tok_sc));
                float2 tok_out1 = _mul_f32x2_1;
                float value00 = tok_out0.x;
                float value10 = tok_out0.y;
                float value01 = tok_out1.x;
                float value11 = tok_out1.y;
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
                float _mul_21;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_21) : "f"(_tmem_load_0[4]), "f"(tok_sf0_4));
                float _max_4;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_4) : "f"(_mul_21), "f"(neg_cl));
                float _min_8;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_8) : "f"(_max_4), "f"(cl));
                float tok_lin00_6 = _min_8;
                float _mul_22;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_22) : "f"(_tmem_load_0[5]), "f"(tok_sf1_5));
                float _max_5;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_5) : "f"(_mul_22), "f"(neg_cl));
                float _min_9;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_9) : "f"(_max_5), "f"(cl));
                float tok_lin01_7 = _min_9;
                float _mul_23;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_23) : "f"(_tmem_load_1[4]), "f"(tok_sf0_4));
                float _max_6;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_6) : "f"(_mul_23), "f"(neg_cl));
                float _min_10;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_10) : "f"(_max_6), "f"(cl));
                float tok_lin10_8 = _min_10;
                float _mul_24;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_24) : "f"(_tmem_load_1[5]), "f"(tok_sf1_5));
                float _max_7;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_7) : "f"(_mul_24), "f"(neg_cl));
                float _min_11;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_11) : "f"(_max_7), "f"(cl));
                float tok_lin11_9 = _min_11;
                float _mul_25;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_25) : "f"(_tmem_load_0[6]), "f"(tok_sf0_4));
                float tok_x00_10 = _mul_25;
                float _mul_26;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_26) : "f"(_tmem_load_0[7]), "f"(tok_sf1_5));
                float tok_x01_11 = _mul_26;
                float _mul_27;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_27) : "f"(_tmem_load_1[6]), "f"(tok_sf0_4));
                float tok_x10_12 = _mul_27;
                float _mul_28;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_28) : "f"(_tmem_load_1[7]), "f"(tok_sf1_5));
                float tok_x11_13 = _mul_28;
                float _mul_29;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_29) : "f"(tok_x00_10), "f"(neg_fused));
                float _exp2_4 = approx_exp2(_mul_29);
                float _add_4;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_4) : "f"(1.0f), "f"(_exp2_4));
                float _rcp_4 = approx_rcp(_add_4);
                float tok_sig00_14 = _rcp_4;
                float _mul_30;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_30) : "f"(tok_x01_11), "f"(neg_fused));
                float _exp2_5 = approx_exp2(_mul_30);
                float _add_5;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_5) : "f"(1.0f), "f"(_exp2_5));
                float _rcp_5 = approx_rcp(_add_5);
                float tok_sig01_15 = _rcp_5;
                float _mul_31;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_31) : "f"(tok_x10_12), "f"(neg_fused));
                float _exp2_6 = approx_exp2(_mul_31);
                float _add_6;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_6) : "f"(1.0f), "f"(_exp2_6));
                float _rcp_6 = approx_rcp(_add_6);
                float tok_sig10_16 = _rcp_6;
                float _mul_32;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_32) : "f"(tok_x11_13), "f"(neg_fused));
                float _exp2_7 = approx_exp2(_mul_32);
                float _add_7;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_7) : "f"(1.0f), "f"(_exp2_7));
                float _rcp_7 = approx_rcp(_add_7);
                float tok_sig11_17 = _rcp_7;
                float _mul_33;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_33) : "f"(tok_x00_10), "f"(tok_sig00_14));
                float _min_12;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_12) : "f"(_mul_33), "f"(cl));
                float tok_g00_18 = _min_12;
                float _mul_34;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_34) : "f"(tok_x01_11), "f"(tok_sig01_15));
                float _min_13;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_13) : "f"(_mul_34), "f"(cl));
                float tok_g01_19 = _min_13;
                float _mul_35;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_35) : "f"(tok_x10_12), "f"(tok_sig10_16));
                float _min_14;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_14) : "f"(_mul_35), "f"(cl));
                float tok_g10_20 = _min_14;
                float _mul_36;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_36) : "f"(tok_x11_13), "f"(tok_sig11_17));
                float _min_15;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_15) : "f"(_mul_36), "f"(cl));
                float tok_g11_21 = _min_15;
                float _fma_4;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_4) : "f"(tok_lin00_6), "f"(sg), "f"(0.0f));
                float _mul_37;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_37) : "f"(_fma_4), "f"(tok_g00_18));
                float tok_acc00_22 = _mul_37;
                float _fma_5;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_5) : "f"(tok_lin01_7), "f"(sg), "f"(0.0f));
                float _mul_38;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_38) : "f"(_fma_5), "f"(tok_g01_19));
                float tok_acc01_23 = _mul_38;
                float _fma_6;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_6) : "f"(tok_lin10_8), "f"(sg), "f"(0.0f));
                float _mul_39;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_39) : "f"(_fma_6), "f"(tok_g10_20));
                float tok_acc10_24 = _mul_39;
                float _fma_7;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_7) : "f"(tok_lin11_9), "f"(sg), "f"(0.0f));
                float _mul_40;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_40) : "f"(_fma_7), "f"(tok_g11_21));
                float tok_acc11_25 = _mul_40;
                float2 _f2_3 = make_float2(sc, sc);
                float2 tok_sc_26 = _f2_3;
                float2 _f2_4 = make_float2(tok_acc00_22, tok_acc10_24);
                float2 _mul_f32x2_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_2) : "l"(*(const unsigned long long*)&_f2_4), "l"(*(const unsigned long long*)&tok_sc_26));
                float2 tok_out0_27 = _mul_f32x2_2;
                float2 _f2_5 = make_float2(tok_acc01_23, tok_acc11_25);
                float2 _mul_f32x2_3;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_3) : "l"(*(const unsigned long long*)&_f2_5), "l"(*(const unsigned long long*)&tok_sc_26));
                float2 tok_out1_28 = _mul_f32x2_3;
                float value00_29 = tok_out0_27.x;
                float value10_30 = tok_out0_27.y;
                float value01_31 = tok_out1_28.x;
                float value11_32 = tok_out1_28.y;
                out_pair[0] = value00_29;
                out_pair[1] = value10_30;
                if (token0_2 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token0_2) * M_out + tok_feature)))[0]) = _pk;
                    }
                }
                out_pair[0] = value01_31;
                out_pair[1] = value11_32;
                if (token1_3 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token1_3) * M_out + tok_feature)))[0]) = _pk;
                    }
                }
                int local_token0_33 = lane_1 % 4 * 2 + 16;
                int local_token1_34 = local_token0_33 + 1;
                int token0_35 = token_block * 64 + local_token0_33;
                int token1_36 = token0_35 + 1;
                float tok_sf0_37 = 0.0f;
                float tok_sf1_38 = 0.0f;
                if (token0_35 < valid_rows) {
                    int tok_route0_2 = route_map[tok_slot_base + token0_35];
                    tok_sf0_37 = SFC[tok_route0_2];
                }
                if (token1_36 < valid_rows) {
                    int tok_route1_2 = route_map[tok_slot_base + token1_36];
                    tok_sf1_38 = SFC[tok_route1_2];
                }
                float _mul_41;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_41) : "f"(_tmem_load_0[8]), "f"(tok_sf0_37));
                float _max_8;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_8) : "f"(_mul_41), "f"(neg_cl));
                float _min_16;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_16) : "f"(_max_8), "f"(cl));
                float tok_lin00_39 = _min_16;
                float _mul_42;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_42) : "f"(_tmem_load_0[9]), "f"(tok_sf1_38));
                float _max_9;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_9) : "f"(_mul_42), "f"(neg_cl));
                float _min_17;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_17) : "f"(_max_9), "f"(cl));
                float tok_lin01_40 = _min_17;
                float _mul_43;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_43) : "f"(_tmem_load_1[8]), "f"(tok_sf0_37));
                float _max_10;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_10) : "f"(_mul_43), "f"(neg_cl));
                float _min_18;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_18) : "f"(_max_10), "f"(cl));
                float tok_lin10_41 = _min_18;
                float _mul_44;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_44) : "f"(_tmem_load_1[9]), "f"(tok_sf1_38));
                float _max_11;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_11) : "f"(_mul_44), "f"(neg_cl));
                float _min_19;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_19) : "f"(_max_11), "f"(cl));
                float tok_lin11_42 = _min_19;
                float _mul_45;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_45) : "f"(_tmem_load_0[10]), "f"(tok_sf0_37));
                float tok_x00_43 = _mul_45;
                float _mul_46;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_46) : "f"(_tmem_load_0[11]), "f"(tok_sf1_38));
                float tok_x01_44 = _mul_46;
                float _mul_47;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_47) : "f"(_tmem_load_1[10]), "f"(tok_sf0_37));
                float tok_x10_45 = _mul_47;
                float _mul_48;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_48) : "f"(_tmem_load_1[11]), "f"(tok_sf1_38));
                float tok_x11_46 = _mul_48;
                float _mul_49;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_49) : "f"(tok_x00_43), "f"(neg_fused));
                float _exp2_8 = approx_exp2(_mul_49);
                float _add_8;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_8) : "f"(1.0f), "f"(_exp2_8));
                float _rcp_8 = approx_rcp(_add_8);
                float tok_sig00_47 = _rcp_8;
                float _mul_50;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_50) : "f"(tok_x01_44), "f"(neg_fused));
                float _exp2_9 = approx_exp2(_mul_50);
                float _add_9;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_9) : "f"(1.0f), "f"(_exp2_9));
                float _rcp_9 = approx_rcp(_add_9);
                float tok_sig01_48 = _rcp_9;
                float _mul_51;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_51) : "f"(tok_x10_45), "f"(neg_fused));
                float _exp2_10 = approx_exp2(_mul_51);
                float _add_10;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_10) : "f"(1.0f), "f"(_exp2_10));
                float _rcp_10 = approx_rcp(_add_10);
                float tok_sig10_49 = _rcp_10;
                float _mul_52;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_52) : "f"(tok_x11_46), "f"(neg_fused));
                float _exp2_11 = approx_exp2(_mul_52);
                float _add_11;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_11) : "f"(1.0f), "f"(_exp2_11));
                float _rcp_11 = approx_rcp(_add_11);
                float tok_sig11_50 = _rcp_11;
                float _mul_53;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_53) : "f"(tok_x00_43), "f"(tok_sig00_47));
                float _min_20;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_20) : "f"(_mul_53), "f"(cl));
                float tok_g00_51 = _min_20;
                float _mul_54;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_54) : "f"(tok_x01_44), "f"(tok_sig01_48));
                float _min_21;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_21) : "f"(_mul_54), "f"(cl));
                float tok_g01_52 = _min_21;
                float _mul_55;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_55) : "f"(tok_x10_45), "f"(tok_sig10_49));
                float _min_22;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_22) : "f"(_mul_55), "f"(cl));
                float tok_g10_53 = _min_22;
                float _mul_56;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_56) : "f"(tok_x11_46), "f"(tok_sig11_50));
                float _min_23;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_23) : "f"(_mul_56), "f"(cl));
                float tok_g11_54 = _min_23;
                float _fma_8;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_8) : "f"(tok_lin00_39), "f"(sg), "f"(0.0f));
                float _mul_57;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_57) : "f"(_fma_8), "f"(tok_g00_51));
                float tok_acc00_55 = _mul_57;
                float _fma_9;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_9) : "f"(tok_lin01_40), "f"(sg), "f"(0.0f));
                float _mul_58;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_58) : "f"(_fma_9), "f"(tok_g01_52));
                float tok_acc01_56 = _mul_58;
                float _fma_10;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_10) : "f"(tok_lin10_41), "f"(sg), "f"(0.0f));
                float _mul_59;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_59) : "f"(_fma_10), "f"(tok_g10_53));
                float tok_acc10_57 = _mul_59;
                float _fma_11;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_11) : "f"(tok_lin11_42), "f"(sg), "f"(0.0f));
                float _mul_60;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_60) : "f"(_fma_11), "f"(tok_g11_54));
                float tok_acc11_58 = _mul_60;
                float2 _f2_6 = make_float2(sc, sc);
                float2 tok_sc_59 = _f2_6;
                float2 _f2_7 = make_float2(tok_acc00_55, tok_acc10_57);
                float2 _mul_f32x2_4;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_4) : "l"(*(const unsigned long long*)&_f2_7), "l"(*(const unsigned long long*)&tok_sc_59));
                float2 tok_out0_60 = _mul_f32x2_4;
                float2 _f2_8 = make_float2(tok_acc01_56, tok_acc11_58);
                float2 _mul_f32x2_5;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_5) : "l"(*(const unsigned long long*)&_f2_8), "l"(*(const unsigned long long*)&tok_sc_59));
                float2 tok_out1_61 = _mul_f32x2_5;
                float value00_62 = tok_out0_60.x;
                float value10_63 = tok_out0_60.y;
                float value01_64 = tok_out1_61.x;
                float value11_65 = tok_out1_61.y;
                out_pair[0] = value00_62;
                out_pair[1] = value10_63;
                if (token0_35 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token0_35) * M_out + tok_feature)))[0]) = _pk;
                    }
                }
                out_pair[0] = value01_64;
                out_pair[1] = value11_65;
                if (token1_36 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token1_36) * M_out + tok_feature)))[0]) = _pk;
                    }
                }
                int local_token0_66 = lane_1 % 4 * 2 + 24;
                int local_token1_67 = local_token0_66 + 1;
                int token0_68 = token_block * 64 + local_token0_66;
                int token1_69 = token0_68 + 1;
                float tok_sf0_70 = 0.0f;
                float tok_sf1_71 = 0.0f;
                if (token0_68 < valid_rows) {
                    int tok_route0_3 = route_map[tok_slot_base + token0_68];
                    tok_sf0_70 = SFC[tok_route0_3];
                }
                if (token1_69 < valid_rows) {
                    int tok_route1_3 = route_map[tok_slot_base + token1_69];
                    tok_sf1_71 = SFC[tok_route1_3];
                }
                float _mul_61;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_61) : "f"(_tmem_load_0[12]), "f"(tok_sf0_70));
                float _max_12;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_12) : "f"(_mul_61), "f"(neg_cl));
                float _min_24;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_24) : "f"(_max_12), "f"(cl));
                float tok_lin00_72 = _min_24;
                float _mul_62;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_62) : "f"(_tmem_load_0[13]), "f"(tok_sf1_71));
                float _max_13;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_13) : "f"(_mul_62), "f"(neg_cl));
                float _min_25;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_25) : "f"(_max_13), "f"(cl));
                float tok_lin01_73 = _min_25;
                float _mul_63;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_63) : "f"(_tmem_load_1[12]), "f"(tok_sf0_70));
                float _max_14;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_14) : "f"(_mul_63), "f"(neg_cl));
                float _min_26;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_26) : "f"(_max_14), "f"(cl));
                float tok_lin10_74 = _min_26;
                float _mul_64;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_64) : "f"(_tmem_load_1[13]), "f"(tok_sf1_71));
                float _max_15;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_15) : "f"(_mul_64), "f"(neg_cl));
                float _min_27;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_27) : "f"(_max_15), "f"(cl));
                float tok_lin11_75 = _min_27;
                float _mul_65;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_65) : "f"(_tmem_load_0[14]), "f"(tok_sf0_70));
                float tok_x00_76 = _mul_65;
                float _mul_66;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_66) : "f"(_tmem_load_0[15]), "f"(tok_sf1_71));
                float tok_x01_77 = _mul_66;
                float _mul_67;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_67) : "f"(_tmem_load_1[14]), "f"(tok_sf0_70));
                float tok_x10_78 = _mul_67;
                float _mul_68;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_68) : "f"(_tmem_load_1[15]), "f"(tok_sf1_71));
                float tok_x11_79 = _mul_68;
                float _mul_69;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_69) : "f"(tok_x00_76), "f"(neg_fused));
                float _exp2_12 = approx_exp2(_mul_69);
                float _add_12;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_12) : "f"(1.0f), "f"(_exp2_12));
                float _rcp_12 = approx_rcp(_add_12);
                float tok_sig00_80 = _rcp_12;
                float _mul_70;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_70) : "f"(tok_x01_77), "f"(neg_fused));
                float _exp2_13 = approx_exp2(_mul_70);
                float _add_13;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_13) : "f"(1.0f), "f"(_exp2_13));
                float _rcp_13 = approx_rcp(_add_13);
                float tok_sig01_81 = _rcp_13;
                float _mul_71;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_71) : "f"(tok_x10_78), "f"(neg_fused));
                float _exp2_14 = approx_exp2(_mul_71);
                float _add_14;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_14) : "f"(1.0f), "f"(_exp2_14));
                float _rcp_14 = approx_rcp(_add_14);
                float tok_sig10_82 = _rcp_14;
                float _mul_72;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_72) : "f"(tok_x11_79), "f"(neg_fused));
                float _exp2_15 = approx_exp2(_mul_72);
                float _add_15;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_15) : "f"(1.0f), "f"(_exp2_15));
                float _rcp_15 = approx_rcp(_add_15);
                float tok_sig11_83 = _rcp_15;
                float _mul_73;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_73) : "f"(tok_x00_76), "f"(tok_sig00_80));
                float _min_28;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_28) : "f"(_mul_73), "f"(cl));
                float tok_g00_84 = _min_28;
                float _mul_74;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_74) : "f"(tok_x01_77), "f"(tok_sig01_81));
                float _min_29;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_29) : "f"(_mul_74), "f"(cl));
                float tok_g01_85 = _min_29;
                float _mul_75;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_75) : "f"(tok_x10_78), "f"(tok_sig10_82));
                float _min_30;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_30) : "f"(_mul_75), "f"(cl));
                float tok_g10_86 = _min_30;
                float _mul_76;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_76) : "f"(tok_x11_79), "f"(tok_sig11_83));
                float _min_31;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_31) : "f"(_mul_76), "f"(cl));
                float tok_g11_87 = _min_31;
                float _fma_12;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_12) : "f"(tok_lin00_72), "f"(sg), "f"(0.0f));
                float _mul_77;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_77) : "f"(_fma_12), "f"(tok_g00_84));
                float tok_acc00_88 = _mul_77;
                float _fma_13;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_13) : "f"(tok_lin01_73), "f"(sg), "f"(0.0f));
                float _mul_78;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_78) : "f"(_fma_13), "f"(tok_g01_85));
                float tok_acc01_89 = _mul_78;
                float _fma_14;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_14) : "f"(tok_lin10_74), "f"(sg), "f"(0.0f));
                float _mul_79;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_79) : "f"(_fma_14), "f"(tok_g10_86));
                float tok_acc10_90 = _mul_79;
                float _fma_15;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_15) : "f"(tok_lin11_75), "f"(sg), "f"(0.0f));
                float _mul_80;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_80) : "f"(_fma_15), "f"(tok_g11_87));
                float tok_acc11_91 = _mul_80;
                float2 _f2_9 = make_float2(sc, sc);
                float2 tok_sc_92 = _f2_9;
                float2 _f2_10 = make_float2(tok_acc00_88, tok_acc10_90);
                float2 _mul_f32x2_6;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_6) : "l"(*(const unsigned long long*)&_f2_10), "l"(*(const unsigned long long*)&tok_sc_92));
                float2 tok_out0_93 = _mul_f32x2_6;
                float2 _f2_11 = make_float2(tok_acc01_89, tok_acc11_91);
                float2 _mul_f32x2_7;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_7) : "l"(*(const unsigned long long*)&_f2_11), "l"(*(const unsigned long long*)&tok_sc_92));
                float2 tok_out1_94 = _mul_f32x2_7;
                float value00_95 = tok_out0_93.x;
                float value10_96 = tok_out0_93.y;
                float value01_97 = tok_out1_94.x;
                float value11_98 = tok_out1_94.y;
                out_pair[0] = value00_95;
                out_pair[1] = value10_96;
                if (token0_68 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token0_68) * M_out + tok_feature)))[0]) = _pk;
                    }
                }
                out_pair[0] = value01_97;
                out_pair[1] = value11_98;
                if (token1_69 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token1_69) * M_out + tok_feature)))[0]) = _pk;
                    }
                }
                int local_token0_99 = lane_1 % 4 * 2 + 32;
                int local_token1_100 = local_token0_99 + 1;
                int token0_101 = token_block * 64 + local_token0_99;
                int token1_102 = token0_101 + 1;
                float tok_sf0_103 = 0.0f;
                float tok_sf1_104 = 0.0f;
                if (token0_101 < valid_rows) {
                    int tok_route0_4 = route_map[tok_slot_base + token0_101];
                    tok_sf0_103 = SFC[tok_route0_4];
                }
                if (token1_102 < valid_rows) {
                    int tok_route1_4 = route_map[tok_slot_base + token1_102];
                    tok_sf1_104 = SFC[tok_route1_4];
                }
                float _mul_81;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_81) : "f"(_tmem_load_0[16]), "f"(tok_sf0_103));
                float _max_16;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_16) : "f"(_mul_81), "f"(neg_cl));
                float _min_32;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_32) : "f"(_max_16), "f"(cl));
                float tok_lin00_105 = _min_32;
                float _mul_82;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_82) : "f"(_tmem_load_0[17]), "f"(tok_sf1_104));
                float _max_17;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_17) : "f"(_mul_82), "f"(neg_cl));
                float _min_33;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_33) : "f"(_max_17), "f"(cl));
                float tok_lin01_106 = _min_33;
                float _mul_83;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_83) : "f"(_tmem_load_1[16]), "f"(tok_sf0_103));
                float _max_18;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_18) : "f"(_mul_83), "f"(neg_cl));
                float _min_34;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_34) : "f"(_max_18), "f"(cl));
                float tok_lin10_107 = _min_34;
                float _mul_84;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_84) : "f"(_tmem_load_1[17]), "f"(tok_sf1_104));
                float _max_19;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_19) : "f"(_mul_84), "f"(neg_cl));
                float _min_35;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_35) : "f"(_max_19), "f"(cl));
                float tok_lin11_108 = _min_35;
                float _mul_85;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_85) : "f"(_tmem_load_0[18]), "f"(tok_sf0_103));
                float tok_x00_109 = _mul_85;
                float _mul_86;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_86) : "f"(_tmem_load_0[19]), "f"(tok_sf1_104));
                float tok_x01_110 = _mul_86;
                float _mul_87;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_87) : "f"(_tmem_load_1[18]), "f"(tok_sf0_103));
                float tok_x10_111 = _mul_87;
                float _mul_88;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_88) : "f"(_tmem_load_1[19]), "f"(tok_sf1_104));
                float tok_x11_112 = _mul_88;
                float _mul_89;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_89) : "f"(tok_x00_109), "f"(neg_fused));
                float _exp2_16 = approx_exp2(_mul_89);
                float _add_16;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_16) : "f"(1.0f), "f"(_exp2_16));
                float _rcp_16 = approx_rcp(_add_16);
                float tok_sig00_113 = _rcp_16;
                float _mul_90;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_90) : "f"(tok_x01_110), "f"(neg_fused));
                float _exp2_17 = approx_exp2(_mul_90);
                float _add_17;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_17) : "f"(1.0f), "f"(_exp2_17));
                float _rcp_17 = approx_rcp(_add_17);
                float tok_sig01_114 = _rcp_17;
                float _mul_91;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_91) : "f"(tok_x10_111), "f"(neg_fused));
                float _exp2_18 = approx_exp2(_mul_91);
                float _add_18;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_18) : "f"(1.0f), "f"(_exp2_18));
                float _rcp_18 = approx_rcp(_add_18);
                float tok_sig10_115 = _rcp_18;
                float _mul_92;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_92) : "f"(tok_x11_112), "f"(neg_fused));
                float _exp2_19 = approx_exp2(_mul_92);
                float _add_19;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_19) : "f"(1.0f), "f"(_exp2_19));
                float _rcp_19 = approx_rcp(_add_19);
                float tok_sig11_116 = _rcp_19;
                float _mul_93;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_93) : "f"(tok_x00_109), "f"(tok_sig00_113));
                float _min_36;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_36) : "f"(_mul_93), "f"(cl));
                float tok_g00_117 = _min_36;
                float _mul_94;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_94) : "f"(tok_x01_110), "f"(tok_sig01_114));
                float _min_37;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_37) : "f"(_mul_94), "f"(cl));
                float tok_g01_118 = _min_37;
                float _mul_95;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_95) : "f"(tok_x10_111), "f"(tok_sig10_115));
                float _min_38;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_38) : "f"(_mul_95), "f"(cl));
                float tok_g10_119 = _min_38;
                float _mul_96;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_96) : "f"(tok_x11_112), "f"(tok_sig11_116));
                float _min_39;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_39) : "f"(_mul_96), "f"(cl));
                float tok_g11_120 = _min_39;
                float _fma_16;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_16) : "f"(tok_lin00_105), "f"(sg), "f"(0.0f));
                float _mul_97;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_97) : "f"(_fma_16), "f"(tok_g00_117));
                float tok_acc00_121 = _mul_97;
                float _fma_17;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_17) : "f"(tok_lin01_106), "f"(sg), "f"(0.0f));
                float _mul_98;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_98) : "f"(_fma_17), "f"(tok_g01_118));
                float tok_acc01_122 = _mul_98;
                float _fma_18;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_18) : "f"(tok_lin10_107), "f"(sg), "f"(0.0f));
                float _mul_99;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_99) : "f"(_fma_18), "f"(tok_g10_119));
                float tok_acc10_123 = _mul_99;
                float _fma_19;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_19) : "f"(tok_lin11_108), "f"(sg), "f"(0.0f));
                float _mul_100;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_100) : "f"(_fma_19), "f"(tok_g11_120));
                float tok_acc11_124 = _mul_100;
                float2 _f2_12 = make_float2(sc, sc);
                float2 tok_sc_125 = _f2_12;
                float2 _f2_13 = make_float2(tok_acc00_121, tok_acc10_123);
                float2 _mul_f32x2_8;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_8) : "l"(*(const unsigned long long*)&_f2_13), "l"(*(const unsigned long long*)&tok_sc_125));
                float2 tok_out0_126 = _mul_f32x2_8;
                float2 _f2_14 = make_float2(tok_acc01_122, tok_acc11_124);
                float2 _mul_f32x2_9;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_9) : "l"(*(const unsigned long long*)&_f2_14), "l"(*(const unsigned long long*)&tok_sc_125));
                float2 tok_out1_127 = _mul_f32x2_9;
                float value00_128 = tok_out0_126.x;
                float value10_129 = tok_out0_126.y;
                float value01_130 = tok_out1_127.x;
                float value11_131 = tok_out1_127.y;
                out_pair[0] = value00_128;
                out_pair[1] = value10_129;
                if (token0_101 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token0_101) * M_out + tok_feature)))[0]) = _pk;
                    }
                }
                out_pair[0] = value01_130;
                out_pair[1] = value11_131;
                if (token1_102 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token1_102) * M_out + tok_feature)))[0]) = _pk;
                    }
                }
                int local_token0_132 = lane_1 % 4 * 2 + 40;
                int local_token1_133 = local_token0_132 + 1;
                int token0_134 = token_block * 64 + local_token0_132;
                int token1_135 = token0_134 + 1;
                float tok_sf0_136 = 0.0f;
                float tok_sf1_137 = 0.0f;
                if (token0_134 < valid_rows) {
                    int tok_route0_5 = route_map[tok_slot_base + token0_134];
                    tok_sf0_136 = SFC[tok_route0_5];
                }
                if (token1_135 < valid_rows) {
                    int tok_route1_5 = route_map[tok_slot_base + token1_135];
                    tok_sf1_137 = SFC[tok_route1_5];
                }
                float _mul_101;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_101) : "f"(_tmem_load_0[20]), "f"(tok_sf0_136));
                float _max_20;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_20) : "f"(_mul_101), "f"(neg_cl));
                float _min_40;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_40) : "f"(_max_20), "f"(cl));
                float tok_lin00_138 = _min_40;
                float _mul_102;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_102) : "f"(_tmem_load_0[21]), "f"(tok_sf1_137));
                float _max_21;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_21) : "f"(_mul_102), "f"(neg_cl));
                float _min_41;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_41) : "f"(_max_21), "f"(cl));
                float tok_lin01_139 = _min_41;
                float _mul_103;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_103) : "f"(_tmem_load_1[20]), "f"(tok_sf0_136));
                float _max_22;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_22) : "f"(_mul_103), "f"(neg_cl));
                float _min_42;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_42) : "f"(_max_22), "f"(cl));
                float tok_lin10_140 = _min_42;
                float _mul_104;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_104) : "f"(_tmem_load_1[21]), "f"(tok_sf1_137));
                float _max_23;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_23) : "f"(_mul_104), "f"(neg_cl));
                float _min_43;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_43) : "f"(_max_23), "f"(cl));
                float tok_lin11_141 = _min_43;
                float _mul_105;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_105) : "f"(_tmem_load_0[22]), "f"(tok_sf0_136));
                float tok_x00_142 = _mul_105;
                float _mul_106;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_106) : "f"(_tmem_load_0[23]), "f"(tok_sf1_137));
                float tok_x01_143 = _mul_106;
                float _mul_107;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_107) : "f"(_tmem_load_1[22]), "f"(tok_sf0_136));
                float tok_x10_144 = _mul_107;
                float _mul_108;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_108) : "f"(_tmem_load_1[23]), "f"(tok_sf1_137));
                float tok_x11_145 = _mul_108;
                float _mul_109;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_109) : "f"(tok_x00_142), "f"(neg_fused));
                float _exp2_20 = approx_exp2(_mul_109);
                float _add_20;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_20) : "f"(1.0f), "f"(_exp2_20));
                float _rcp_20 = approx_rcp(_add_20);
                float tok_sig00_146 = _rcp_20;
                float _mul_110;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_110) : "f"(tok_x01_143), "f"(neg_fused));
                float _exp2_21 = approx_exp2(_mul_110);
                float _add_21;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_21) : "f"(1.0f), "f"(_exp2_21));
                float _rcp_21 = approx_rcp(_add_21);
                float tok_sig01_147 = _rcp_21;
                float _mul_111;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_111) : "f"(tok_x10_144), "f"(neg_fused));
                float _exp2_22 = approx_exp2(_mul_111);
                float _add_22;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_22) : "f"(1.0f), "f"(_exp2_22));
                float _rcp_22 = approx_rcp(_add_22);
                float tok_sig10_148 = _rcp_22;
                float _mul_112;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_112) : "f"(tok_x11_145), "f"(neg_fused));
                float _exp2_23 = approx_exp2(_mul_112);
                float _add_23;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_23) : "f"(1.0f), "f"(_exp2_23));
                float _rcp_23 = approx_rcp(_add_23);
                float tok_sig11_149 = _rcp_23;
                float _mul_113;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_113) : "f"(tok_x00_142), "f"(tok_sig00_146));
                float _min_44;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_44) : "f"(_mul_113), "f"(cl));
                float tok_g00_150 = _min_44;
                float _mul_114;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_114) : "f"(tok_x01_143), "f"(tok_sig01_147));
                float _min_45;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_45) : "f"(_mul_114), "f"(cl));
                float tok_g01_151 = _min_45;
                float _mul_115;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_115) : "f"(tok_x10_144), "f"(tok_sig10_148));
                float _min_46;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_46) : "f"(_mul_115), "f"(cl));
                float tok_g10_152 = _min_46;
                float _mul_116;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_116) : "f"(tok_x11_145), "f"(tok_sig11_149));
                float _min_47;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_47) : "f"(_mul_116), "f"(cl));
                float tok_g11_153 = _min_47;
                float _fma_20;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_20) : "f"(tok_lin00_138), "f"(sg), "f"(0.0f));
                float _mul_117;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_117) : "f"(_fma_20), "f"(tok_g00_150));
                float tok_acc00_154 = _mul_117;
                float _fma_21;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_21) : "f"(tok_lin01_139), "f"(sg), "f"(0.0f));
                float _mul_118;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_118) : "f"(_fma_21), "f"(tok_g01_151));
                float tok_acc01_155 = _mul_118;
                float _fma_22;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_22) : "f"(tok_lin10_140), "f"(sg), "f"(0.0f));
                float _mul_119;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_119) : "f"(_fma_22), "f"(tok_g10_152));
                float tok_acc10_156 = _mul_119;
                float _fma_23;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_23) : "f"(tok_lin11_141), "f"(sg), "f"(0.0f));
                float _mul_120;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_120) : "f"(_fma_23), "f"(tok_g11_153));
                float tok_acc11_157 = _mul_120;
                float2 _f2_15 = make_float2(sc, sc);
                float2 tok_sc_158 = _f2_15;
                float2 _f2_16 = make_float2(tok_acc00_154, tok_acc10_156);
                float2 _mul_f32x2_10;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_10) : "l"(*(const unsigned long long*)&_f2_16), "l"(*(const unsigned long long*)&tok_sc_158));
                float2 tok_out0_159 = _mul_f32x2_10;
                float2 _f2_17 = make_float2(tok_acc01_155, tok_acc11_157);
                float2 _mul_f32x2_11;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_11) : "l"(*(const unsigned long long*)&_f2_17), "l"(*(const unsigned long long*)&tok_sc_158));
                float2 tok_out1_160 = _mul_f32x2_11;
                float value00_161 = tok_out0_159.x;
                float value10_162 = tok_out0_159.y;
                float value01_163 = tok_out1_160.x;
                float value11_164 = tok_out1_160.y;
                out_pair[0] = value00_161;
                out_pair[1] = value10_162;
                if (token0_134 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token0_134) * M_out + tok_feature)))[0]) = _pk;
                    }
                }
                out_pair[0] = value01_163;
                out_pair[1] = value11_164;
                if (token1_135 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token1_135) * M_out + tok_feature)))[0]) = _pk;
                    }
                }
                int local_token0_165 = lane_1 % 4 * 2 + 48;
                int local_token1_166 = local_token0_165 + 1;
                int token0_167 = token_block * 64 + local_token0_165;
                int token1_168 = token0_167 + 1;
                float tok_sf0_169 = 0.0f;
                float tok_sf1_170 = 0.0f;
                if (token0_167 < valid_rows) {
                    int tok_route0_6 = route_map[tok_slot_base + token0_167];
                    tok_sf0_169 = SFC[tok_route0_6];
                }
                if (token1_168 < valid_rows) {
                    int tok_route1_6 = route_map[tok_slot_base + token1_168];
                    tok_sf1_170 = SFC[tok_route1_6];
                }
                float _mul_121;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_121) : "f"(_tmem_load_0[24]), "f"(tok_sf0_169));
                float _max_24;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_24) : "f"(_mul_121), "f"(neg_cl));
                float _min_48;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_48) : "f"(_max_24), "f"(cl));
                float tok_lin00_171 = _min_48;
                float _mul_122;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_122) : "f"(_tmem_load_0[25]), "f"(tok_sf1_170));
                float _max_25;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_25) : "f"(_mul_122), "f"(neg_cl));
                float _min_49;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_49) : "f"(_max_25), "f"(cl));
                float tok_lin01_172 = _min_49;
                float _mul_123;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_123) : "f"(_tmem_load_1[24]), "f"(tok_sf0_169));
                float _max_26;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_26) : "f"(_mul_123), "f"(neg_cl));
                float _min_50;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_50) : "f"(_max_26), "f"(cl));
                float tok_lin10_173 = _min_50;
                float _mul_124;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_124) : "f"(_tmem_load_1[25]), "f"(tok_sf1_170));
                float _max_27;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_27) : "f"(_mul_124), "f"(neg_cl));
                float _min_51;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_51) : "f"(_max_27), "f"(cl));
                float tok_lin11_174 = _min_51;
                float _mul_125;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_125) : "f"(_tmem_load_0[26]), "f"(tok_sf0_169));
                float tok_x00_175 = _mul_125;
                float _mul_126;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_126) : "f"(_tmem_load_0[27]), "f"(tok_sf1_170));
                float tok_x01_176 = _mul_126;
                float _mul_127;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_127) : "f"(_tmem_load_1[26]), "f"(tok_sf0_169));
                float tok_x10_177 = _mul_127;
                float _mul_128;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_128) : "f"(_tmem_load_1[27]), "f"(tok_sf1_170));
                float tok_x11_178 = _mul_128;
                float _mul_129;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_129) : "f"(tok_x00_175), "f"(neg_fused));
                float _exp2_24 = approx_exp2(_mul_129);
                float _add_24;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_24) : "f"(1.0f), "f"(_exp2_24));
                float _rcp_24 = approx_rcp(_add_24);
                float tok_sig00_179 = _rcp_24;
                float _mul_130;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_130) : "f"(tok_x01_176), "f"(neg_fused));
                float _exp2_25 = approx_exp2(_mul_130);
                float _add_25;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_25) : "f"(1.0f), "f"(_exp2_25));
                float _rcp_25 = approx_rcp(_add_25);
                float tok_sig01_180 = _rcp_25;
                float _mul_131;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_131) : "f"(tok_x10_177), "f"(neg_fused));
                float _exp2_26 = approx_exp2(_mul_131);
                float _add_26;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_26) : "f"(1.0f), "f"(_exp2_26));
                float _rcp_26 = approx_rcp(_add_26);
                float tok_sig10_181 = _rcp_26;
                float _mul_132;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_132) : "f"(tok_x11_178), "f"(neg_fused));
                float _exp2_27 = approx_exp2(_mul_132);
                float _add_27;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_27) : "f"(1.0f), "f"(_exp2_27));
                float _rcp_27 = approx_rcp(_add_27);
                float tok_sig11_182 = _rcp_27;
                float _mul_133;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_133) : "f"(tok_x00_175), "f"(tok_sig00_179));
                float _min_52;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_52) : "f"(_mul_133), "f"(cl));
                float tok_g00_183 = _min_52;
                float _mul_134;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_134) : "f"(tok_x01_176), "f"(tok_sig01_180));
                float _min_53;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_53) : "f"(_mul_134), "f"(cl));
                float tok_g01_184 = _min_53;
                float _mul_135;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_135) : "f"(tok_x10_177), "f"(tok_sig10_181));
                float _min_54;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_54) : "f"(_mul_135), "f"(cl));
                float tok_g10_185 = _min_54;
                float _mul_136;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_136) : "f"(tok_x11_178), "f"(tok_sig11_182));
                float _min_55;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_55) : "f"(_mul_136), "f"(cl));
                float tok_g11_186 = _min_55;
                float _fma_24;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_24) : "f"(tok_lin00_171), "f"(sg), "f"(0.0f));
                float _mul_137;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_137) : "f"(_fma_24), "f"(tok_g00_183));
                float tok_acc00_187 = _mul_137;
                float _fma_25;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_25) : "f"(tok_lin01_172), "f"(sg), "f"(0.0f));
                float _mul_138;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_138) : "f"(_fma_25), "f"(tok_g01_184));
                float tok_acc01_188 = _mul_138;
                float _fma_26;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_26) : "f"(tok_lin10_173), "f"(sg), "f"(0.0f));
                float _mul_139;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_139) : "f"(_fma_26), "f"(tok_g10_185));
                float tok_acc10_189 = _mul_139;
                float _fma_27;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_27) : "f"(tok_lin11_174), "f"(sg), "f"(0.0f));
                float _mul_140;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_140) : "f"(_fma_27), "f"(tok_g11_186));
                float tok_acc11_190 = _mul_140;
                float2 _f2_18 = make_float2(sc, sc);
                float2 tok_sc_191 = _f2_18;
                float2 _f2_19 = make_float2(tok_acc00_187, tok_acc10_189);
                float2 _mul_f32x2_12;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_12) : "l"(*(const unsigned long long*)&_f2_19), "l"(*(const unsigned long long*)&tok_sc_191));
                float2 tok_out0_192 = _mul_f32x2_12;
                float2 _f2_20 = make_float2(tok_acc01_188, tok_acc11_190);
                float2 _mul_f32x2_13;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_13) : "l"(*(const unsigned long long*)&_f2_20), "l"(*(const unsigned long long*)&tok_sc_191));
                float2 tok_out1_193 = _mul_f32x2_13;
                float value00_194 = tok_out0_192.x;
                float value10_195 = tok_out0_192.y;
                float value01_196 = tok_out1_193.x;
                float value11_197 = tok_out1_193.y;
                out_pair[0] = value00_194;
                out_pair[1] = value10_195;
                if (token0_167 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token0_167) * M_out + tok_feature)))[0]) = _pk;
                    }
                }
                out_pair[0] = value01_196;
                out_pair[1] = value11_197;
                if (token1_168 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token1_168) * M_out + tok_feature)))[0]) = _pk;
                    }
                }
                int local_token0_198 = lane_1 % 4 * 2 + 56;
                int local_token1_199 = local_token0_198 + 1;
                int token0_200 = token_block * 64 + local_token0_198;
                int token1_201 = token0_200 + 1;
                float tok_sf0_202 = 0.0f;
                float tok_sf1_203 = 0.0f;
                if (token0_200 < valid_rows) {
                    int tok_route0_7 = route_map[tok_slot_base + token0_200];
                    tok_sf0_202 = SFC[tok_route0_7];
                }
                if (token1_201 < valid_rows) {
                    int tok_route1_7 = route_map[tok_slot_base + token1_201];
                    tok_sf1_203 = SFC[tok_route1_7];
                }
                float _mul_141;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_141) : "f"(_tmem_load_0[28]), "f"(tok_sf0_202));
                float _max_28;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_28) : "f"(_mul_141), "f"(neg_cl));
                float _min_56;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_56) : "f"(_max_28), "f"(cl));
                float tok_lin00_204 = _min_56;
                float _mul_142;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_142) : "f"(_tmem_load_0[29]), "f"(tok_sf1_203));
                float _max_29;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_29) : "f"(_mul_142), "f"(neg_cl));
                float _min_57;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_57) : "f"(_max_29), "f"(cl));
                float tok_lin01_205 = _min_57;
                float _mul_143;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_143) : "f"(_tmem_load_1[28]), "f"(tok_sf0_202));
                float _max_30;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_30) : "f"(_mul_143), "f"(neg_cl));
                float _min_58;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_58) : "f"(_max_30), "f"(cl));
                float tok_lin10_206 = _min_58;
                float _mul_144;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_144) : "f"(_tmem_load_1[29]), "f"(tok_sf1_203));
                float _max_31;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_31) : "f"(_mul_144), "f"(neg_cl));
                float _min_59;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_59) : "f"(_max_31), "f"(cl));
                float tok_lin11_207 = _min_59;
                float _mul_145;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_145) : "f"(_tmem_load_0[30]), "f"(tok_sf0_202));
                float tok_x00_208 = _mul_145;
                float _mul_146;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_146) : "f"(_tmem_load_0[31]), "f"(tok_sf1_203));
                float tok_x01_209 = _mul_146;
                float _mul_147;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_147) : "f"(_tmem_load_1[30]), "f"(tok_sf0_202));
                float tok_x10_210 = _mul_147;
                float _mul_148;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_148) : "f"(_tmem_load_1[31]), "f"(tok_sf1_203));
                float tok_x11_211 = _mul_148;
                float _mul_149;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_149) : "f"(tok_x00_208), "f"(neg_fused));
                float _exp2_28 = approx_exp2(_mul_149);
                float _add_28;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_28) : "f"(1.0f), "f"(_exp2_28));
                float _rcp_28 = approx_rcp(_add_28);
                float tok_sig00_212 = _rcp_28;
                float _mul_150;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_150) : "f"(tok_x01_209), "f"(neg_fused));
                float _exp2_29 = approx_exp2(_mul_150);
                float _add_29;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_29) : "f"(1.0f), "f"(_exp2_29));
                float _rcp_29 = approx_rcp(_add_29);
                float tok_sig01_213 = _rcp_29;
                float _mul_151;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_151) : "f"(tok_x10_210), "f"(neg_fused));
                float _exp2_30 = approx_exp2(_mul_151);
                float _add_30;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_30) : "f"(1.0f), "f"(_exp2_30));
                float _rcp_30 = approx_rcp(_add_30);
                float tok_sig10_214 = _rcp_30;
                float _mul_152;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_152) : "f"(tok_x11_211), "f"(neg_fused));
                float _exp2_31 = approx_exp2(_mul_152);
                float _add_31;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_31) : "f"(1.0f), "f"(_exp2_31));
                float _rcp_31 = approx_rcp(_add_31);
                float tok_sig11_215 = _rcp_31;
                float _mul_153;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_153) : "f"(tok_x00_208), "f"(tok_sig00_212));
                float _min_60;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_60) : "f"(_mul_153), "f"(cl));
                float tok_g00_216 = _min_60;
                float _mul_154;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_154) : "f"(tok_x01_209), "f"(tok_sig01_213));
                float _min_61;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_61) : "f"(_mul_154), "f"(cl));
                float tok_g01_217 = _min_61;
                float _mul_155;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_155) : "f"(tok_x10_210), "f"(tok_sig10_214));
                float _min_62;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_62) : "f"(_mul_155), "f"(cl));
                float tok_g10_218 = _min_62;
                float _mul_156;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_156) : "f"(tok_x11_211), "f"(tok_sig11_215));
                float _min_63;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_63) : "f"(_mul_156), "f"(cl));
                float tok_g11_219 = _min_63;
                float _fma_28;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_28) : "f"(tok_lin00_204), "f"(sg), "f"(0.0f));
                float _mul_157;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_157) : "f"(_fma_28), "f"(tok_g00_216));
                float tok_acc00_220 = _mul_157;
                float _fma_29;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_29) : "f"(tok_lin01_205), "f"(sg), "f"(0.0f));
                float _mul_158;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_158) : "f"(_fma_29), "f"(tok_g01_217));
                float tok_acc01_221 = _mul_158;
                float _fma_30;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_30) : "f"(tok_lin10_206), "f"(sg), "f"(0.0f));
                float _mul_159;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_159) : "f"(_fma_30), "f"(tok_g10_218));
                float tok_acc10_222 = _mul_159;
                float _fma_31;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_31) : "f"(tok_lin11_207), "f"(sg), "f"(0.0f));
                float _mul_160;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_160) : "f"(_fma_31), "f"(tok_g11_219));
                float tok_acc11_223 = _mul_160;
                float2 _f2_21 = make_float2(sc, sc);
                float2 tok_sc_224 = _f2_21;
                float2 _f2_22 = make_float2(tok_acc00_220, tok_acc10_222);
                float2 _mul_f32x2_14;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_14) : "l"(*(const unsigned long long*)&_f2_22), "l"(*(const unsigned long long*)&tok_sc_224));
                float2 tok_out0_225 = _mul_f32x2_14;
                float2 _f2_23 = make_float2(tok_acc01_221, tok_acc11_223);
                float2 _mul_f32x2_15;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_15) : "l"(*(const unsigned long long*)&_f2_23), "l"(*(const unsigned long long*)&tok_sc_224));
                float2 tok_out1_226 = _mul_f32x2_15;
                float value00_227 = tok_out0_225.x;
                float value10_228 = tok_out0_225.y;
                float value01_229 = tok_out1_226.x;
                float value11_230 = tok_out1_226.y;
                out_pair[0] = value00_227;
                out_pair[1] = value10_228;
                if (token0_200 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token0_200) * M_out + tok_feature)))[0]) = _pk;
                    }
                }
                out_pair[0] = value01_229;
                out_pair[1] = value11_230;
                if (token1_201 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token1_201) * M_out + tok_feature)))[0]) = _pk;
                    }
                }
                int token_block_231 = 2 + warp_group;
                if (epilogue_local_idx == 0) {
                    token_block_231 = (token_block_231 + 3) % 4;
                }
                int accum_col_232 = epilogue_local_idx * 192 + token_block_231 * 64;
                int row_addr_233 = warp_local * 32 << 16;
                float _tmem_load_2[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[31]))
                    : "r"(taddr + (unsigned int)row_addr_233 + (unsigned int)accum_col_232));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float _tmem_load_3[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[31]))
                    : "r"(taddr + (unsigned int)row_addr_233 + 1048576 + (unsigned int)accum_col_232));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int base_row_234 = warp_local * 16 + lane_1 / 4 * 2;
                int tok_feature_235 = (int)m_tile * 64 + base_row_234;
                asm volatile("cp.async.bulk.wait_group.read 0;");
                if (warp_group == 0) {
                    asm volatile("barrier.sync 7, 128;" ::: "memory");
                }
                if (warp_group == 1) {
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                }
                int local_token0_236 = lane_1 % 4 * 2;
                int local_token1_237 = local_token0_236 + 1;
                int token0_238 = token_block_231 * 64 + local_token0_236;
                int token1_239 = token0_238 + 1;
                float tok_sf0_240 = 0.0f;
                float tok_sf1_241 = 0.0f;
                if (token0_238 < valid_rows) {
                    int tok_route0_8 = route_map[tok_slot_base + token0_238];
                    tok_sf0_240 = SFC[tok_route0_8];
                }
                if (token1_239 < valid_rows) {
                    int tok_route1_8 = route_map[tok_slot_base + token1_239];
                    tok_sf1_241 = SFC[tok_route1_8];
                }
                float _mul_161;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_161) : "f"(_tmem_load_2[0]), "f"(tok_sf0_240));
                float _max_32;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_32) : "f"(_mul_161), "f"(neg_cl));
                float _min_64;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_64) : "f"(_max_32), "f"(cl));
                float tok_lin00_242 = _min_64;
                float _mul_162;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_162) : "f"(_tmem_load_2[1]), "f"(tok_sf1_241));
                float _max_33;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_33) : "f"(_mul_162), "f"(neg_cl));
                float _min_65;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_65) : "f"(_max_33), "f"(cl));
                float tok_lin01_243 = _min_65;
                float _mul_163;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_163) : "f"(_tmem_load_3[0]), "f"(tok_sf0_240));
                float _max_34;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_34) : "f"(_mul_163), "f"(neg_cl));
                float _min_66;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_66) : "f"(_max_34), "f"(cl));
                float tok_lin10_244 = _min_66;
                float _mul_164;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_164) : "f"(_tmem_load_3[1]), "f"(tok_sf1_241));
                float _max_35;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_35) : "f"(_mul_164), "f"(neg_cl));
                float _min_67;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_67) : "f"(_max_35), "f"(cl));
                float tok_lin11_245 = _min_67;
                float _mul_165;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_165) : "f"(_tmem_load_2[2]), "f"(tok_sf0_240));
                float tok_x00_246 = _mul_165;
                float _mul_166;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_166) : "f"(_tmem_load_2[3]), "f"(tok_sf1_241));
                float tok_x01_247 = _mul_166;
                float _mul_167;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_167) : "f"(_tmem_load_3[2]), "f"(tok_sf0_240));
                float tok_x10_248 = _mul_167;
                float _mul_168;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_168) : "f"(_tmem_load_3[3]), "f"(tok_sf1_241));
                float tok_x11_249 = _mul_168;
                float _mul_169;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_169) : "f"(tok_x00_246), "f"(neg_fused));
                float _exp2_32 = approx_exp2(_mul_169);
                float _add_32;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_32) : "f"(1.0f), "f"(_exp2_32));
                float _rcp_32 = approx_rcp(_add_32);
                float tok_sig00_250 = _rcp_32;
                float _mul_170;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_170) : "f"(tok_x01_247), "f"(neg_fused));
                float _exp2_33 = approx_exp2(_mul_170);
                float _add_33;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_33) : "f"(1.0f), "f"(_exp2_33));
                float _rcp_33 = approx_rcp(_add_33);
                float tok_sig01_251 = _rcp_33;
                float _mul_171;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_171) : "f"(tok_x10_248), "f"(neg_fused));
                float _exp2_34 = approx_exp2(_mul_171);
                float _add_34;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_34) : "f"(1.0f), "f"(_exp2_34));
                float _rcp_34 = approx_rcp(_add_34);
                float tok_sig10_252 = _rcp_34;
                float _mul_172;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_172) : "f"(tok_x11_249), "f"(neg_fused));
                float _exp2_35 = approx_exp2(_mul_172);
                float _add_35;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_35) : "f"(1.0f), "f"(_exp2_35));
                float _rcp_35 = approx_rcp(_add_35);
                float tok_sig11_253 = _rcp_35;
                float _mul_173;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_173) : "f"(tok_x00_246), "f"(tok_sig00_250));
                float _min_68;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_68) : "f"(_mul_173), "f"(cl));
                float tok_g00_254 = _min_68;
                float _mul_174;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_174) : "f"(tok_x01_247), "f"(tok_sig01_251));
                float _min_69;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_69) : "f"(_mul_174), "f"(cl));
                float tok_g01_255 = _min_69;
                float _mul_175;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_175) : "f"(tok_x10_248), "f"(tok_sig10_252));
                float _min_70;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_70) : "f"(_mul_175), "f"(cl));
                float tok_g10_256 = _min_70;
                float _mul_176;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_176) : "f"(tok_x11_249), "f"(tok_sig11_253));
                float _min_71;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_71) : "f"(_mul_176), "f"(cl));
                float tok_g11_257 = _min_71;
                float _fma_32;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_32) : "f"(tok_lin00_242), "f"(sg), "f"(0.0f));
                float _mul_177;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_177) : "f"(_fma_32), "f"(tok_g00_254));
                float tok_acc00_258 = _mul_177;
                float _fma_33;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_33) : "f"(tok_lin01_243), "f"(sg), "f"(0.0f));
                float _mul_178;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_178) : "f"(_fma_33), "f"(tok_g01_255));
                float tok_acc01_259 = _mul_178;
                float _fma_34;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_34) : "f"(tok_lin10_244), "f"(sg), "f"(0.0f));
                float _mul_179;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_179) : "f"(_fma_34), "f"(tok_g10_256));
                float tok_acc10_260 = _mul_179;
                float _fma_35;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_35) : "f"(tok_lin11_245), "f"(sg), "f"(0.0f));
                float _mul_180;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_180) : "f"(_fma_35), "f"(tok_g11_257));
                float tok_acc11_261 = _mul_180;
                float2 _f2_24 = make_float2(sc, sc);
                float2 tok_sc_262 = _f2_24;
                float2 _f2_25 = make_float2(tok_acc00_258, tok_acc10_260);
                float2 _mul_f32x2_16;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_16) : "l"(*(const unsigned long long*)&_f2_25), "l"(*(const unsigned long long*)&tok_sc_262));
                float2 tok_out0_263 = _mul_f32x2_16;
                float2 _f2_26 = make_float2(tok_acc01_259, tok_acc11_261);
                float2 _mul_f32x2_17;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_17) : "l"(*(const unsigned long long*)&_f2_26), "l"(*(const unsigned long long*)&tok_sc_262));
                float2 tok_out1_264 = _mul_f32x2_17;
                float value00_265 = tok_out0_263.x;
                float value10_266 = tok_out0_263.y;
                float value01_267 = tok_out1_264.x;
                float value11_268 = tok_out1_264.y;
                out_pair[0] = value00_265;
                out_pair[1] = value10_266;
                if (token0_238 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token0_238) * M_out + tok_feature_235)))[0]) = _pk;
                    }
                }
                out_pair[0] = value01_267;
                out_pair[1] = value11_268;
                if (token1_239 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token1_239) * M_out + tok_feature_235)))[0]) = _pk;
                    }
                }
                int local_token0_269 = lane_1 % 4 * 2 + 8;
                int local_token1_270 = local_token0_269 + 1;
                int token0_271 = token_block_231 * 64 + local_token0_269;
                int token1_272 = token0_271 + 1;
                float tok_sf0_273 = 0.0f;
                float tok_sf1_274 = 0.0f;
                if (token0_271 < valid_rows) {
                    int tok_route0_9 = route_map[tok_slot_base + token0_271];
                    tok_sf0_273 = SFC[tok_route0_9];
                }
                if (token1_272 < valid_rows) {
                    int tok_route1_9 = route_map[tok_slot_base + token1_272];
                    tok_sf1_274 = SFC[tok_route1_9];
                }
                float _mul_181;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_181) : "f"(_tmem_load_2[4]), "f"(tok_sf0_273));
                float _max_36;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_36) : "f"(_mul_181), "f"(neg_cl));
                float _min_72;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_72) : "f"(_max_36), "f"(cl));
                float tok_lin00_275 = _min_72;
                float _mul_182;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_182) : "f"(_tmem_load_2[5]), "f"(tok_sf1_274));
                float _max_37;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_37) : "f"(_mul_182), "f"(neg_cl));
                float _min_73;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_73) : "f"(_max_37), "f"(cl));
                float tok_lin01_276 = _min_73;
                float _mul_183;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_183) : "f"(_tmem_load_3[4]), "f"(tok_sf0_273));
                float _max_38;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_38) : "f"(_mul_183), "f"(neg_cl));
                float _min_74;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_74) : "f"(_max_38), "f"(cl));
                float tok_lin10_277 = _min_74;
                float _mul_184;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_184) : "f"(_tmem_load_3[5]), "f"(tok_sf1_274));
                float _max_39;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_39) : "f"(_mul_184), "f"(neg_cl));
                float _min_75;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_75) : "f"(_max_39), "f"(cl));
                float tok_lin11_278 = _min_75;
                float _mul_185;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_185) : "f"(_tmem_load_2[6]), "f"(tok_sf0_273));
                float tok_x00_279 = _mul_185;
                float _mul_186;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_186) : "f"(_tmem_load_2[7]), "f"(tok_sf1_274));
                float tok_x01_280 = _mul_186;
                float _mul_187;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_187) : "f"(_tmem_load_3[6]), "f"(tok_sf0_273));
                float tok_x10_281 = _mul_187;
                float _mul_188;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_188) : "f"(_tmem_load_3[7]), "f"(tok_sf1_274));
                float tok_x11_282 = _mul_188;
                float _mul_189;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_189) : "f"(tok_x00_279), "f"(neg_fused));
                float _exp2_36 = approx_exp2(_mul_189);
                float _add_36;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_36) : "f"(1.0f), "f"(_exp2_36));
                float _rcp_36 = approx_rcp(_add_36);
                float tok_sig00_283 = _rcp_36;
                float _mul_190;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_190) : "f"(tok_x01_280), "f"(neg_fused));
                float _exp2_37 = approx_exp2(_mul_190);
                float _add_37;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_37) : "f"(1.0f), "f"(_exp2_37));
                float _rcp_37 = approx_rcp(_add_37);
                float tok_sig01_284 = _rcp_37;
                float _mul_191;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_191) : "f"(tok_x10_281), "f"(neg_fused));
                float _exp2_38 = approx_exp2(_mul_191);
                float _add_38;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_38) : "f"(1.0f), "f"(_exp2_38));
                float _rcp_38 = approx_rcp(_add_38);
                float tok_sig10_285 = _rcp_38;
                float _mul_192;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_192) : "f"(tok_x11_282), "f"(neg_fused));
                float _exp2_39 = approx_exp2(_mul_192);
                float _add_39;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_39) : "f"(1.0f), "f"(_exp2_39));
                float _rcp_39 = approx_rcp(_add_39);
                float tok_sig11_286 = _rcp_39;
                float _mul_193;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_193) : "f"(tok_x00_279), "f"(tok_sig00_283));
                float _min_76;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_76) : "f"(_mul_193), "f"(cl));
                float tok_g00_287 = _min_76;
                float _mul_194;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_194) : "f"(tok_x01_280), "f"(tok_sig01_284));
                float _min_77;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_77) : "f"(_mul_194), "f"(cl));
                float tok_g01_288 = _min_77;
                float _mul_195;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_195) : "f"(tok_x10_281), "f"(tok_sig10_285));
                float _min_78;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_78) : "f"(_mul_195), "f"(cl));
                float tok_g10_289 = _min_78;
                float _mul_196;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_196) : "f"(tok_x11_282), "f"(tok_sig11_286));
                float _min_79;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_79) : "f"(_mul_196), "f"(cl));
                float tok_g11_290 = _min_79;
                float _fma_36;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_36) : "f"(tok_lin00_275), "f"(sg), "f"(0.0f));
                float _mul_197;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_197) : "f"(_fma_36), "f"(tok_g00_287));
                float tok_acc00_291 = _mul_197;
                float _fma_37;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_37) : "f"(tok_lin01_276), "f"(sg), "f"(0.0f));
                float _mul_198;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_198) : "f"(_fma_37), "f"(tok_g01_288));
                float tok_acc01_292 = _mul_198;
                float _fma_38;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_38) : "f"(tok_lin10_277), "f"(sg), "f"(0.0f));
                float _mul_199;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_199) : "f"(_fma_38), "f"(tok_g10_289));
                float tok_acc10_293 = _mul_199;
                float _fma_39;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_39) : "f"(tok_lin11_278), "f"(sg), "f"(0.0f));
                float _mul_200;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_200) : "f"(_fma_39), "f"(tok_g11_290));
                float tok_acc11_294 = _mul_200;
                float2 _f2_27 = make_float2(sc, sc);
                float2 tok_sc_295 = _f2_27;
                float2 _f2_28 = make_float2(tok_acc00_291, tok_acc10_293);
                float2 _mul_f32x2_18;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_18) : "l"(*(const unsigned long long*)&_f2_28), "l"(*(const unsigned long long*)&tok_sc_295));
                float2 tok_out0_296 = _mul_f32x2_18;
                float2 _f2_29 = make_float2(tok_acc01_292, tok_acc11_294);
                float2 _mul_f32x2_19;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_19) : "l"(*(const unsigned long long*)&_f2_29), "l"(*(const unsigned long long*)&tok_sc_295));
                float2 tok_out1_297 = _mul_f32x2_19;
                float value00_298 = tok_out0_296.x;
                float value10_299 = tok_out0_296.y;
                float value01_300 = tok_out1_297.x;
                float value11_301 = tok_out1_297.y;
                out_pair[0] = value00_298;
                out_pair[1] = value10_299;
                if (token0_271 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token0_271) * M_out + tok_feature_235)))[0]) = _pk;
                    }
                }
                out_pair[0] = value01_300;
                out_pair[1] = value11_301;
                if (token1_272 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token1_272) * M_out + tok_feature_235)))[0]) = _pk;
                    }
                }
                int local_token0_302 = lane_1 % 4 * 2 + 16;
                int local_token1_303 = local_token0_302 + 1;
                int token0_304 = token_block_231 * 64 + local_token0_302;
                int token1_305 = token0_304 + 1;
                float tok_sf0_306 = 0.0f;
                float tok_sf1_307 = 0.0f;
                if (token0_304 < valid_rows) {
                    int tok_route0_10 = route_map[tok_slot_base + token0_304];
                    tok_sf0_306 = SFC[tok_route0_10];
                }
                if (token1_305 < valid_rows) {
                    int tok_route1_10 = route_map[tok_slot_base + token1_305];
                    tok_sf1_307 = SFC[tok_route1_10];
                }
                float _mul_201;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_201) : "f"(_tmem_load_2[8]), "f"(tok_sf0_306));
                float _max_40;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_40) : "f"(_mul_201), "f"(neg_cl));
                float _min_80;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_80) : "f"(_max_40), "f"(cl));
                float tok_lin00_308 = _min_80;
                float _mul_202;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_202) : "f"(_tmem_load_2[9]), "f"(tok_sf1_307));
                float _max_41;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_41) : "f"(_mul_202), "f"(neg_cl));
                float _min_81;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_81) : "f"(_max_41), "f"(cl));
                float tok_lin01_309 = _min_81;
                float _mul_203;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_203) : "f"(_tmem_load_3[8]), "f"(tok_sf0_306));
                float _max_42;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_42) : "f"(_mul_203), "f"(neg_cl));
                float _min_82;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_82) : "f"(_max_42), "f"(cl));
                float tok_lin10_310 = _min_82;
                float _mul_204;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_204) : "f"(_tmem_load_3[9]), "f"(tok_sf1_307));
                float _max_43;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_43) : "f"(_mul_204), "f"(neg_cl));
                float _min_83;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_83) : "f"(_max_43), "f"(cl));
                float tok_lin11_311 = _min_83;
                float _mul_205;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_205) : "f"(_tmem_load_2[10]), "f"(tok_sf0_306));
                float tok_x00_312 = _mul_205;
                float _mul_206;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_206) : "f"(_tmem_load_2[11]), "f"(tok_sf1_307));
                float tok_x01_313 = _mul_206;
                float _mul_207;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_207) : "f"(_tmem_load_3[10]), "f"(tok_sf0_306));
                float tok_x10_314 = _mul_207;
                float _mul_208;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_208) : "f"(_tmem_load_3[11]), "f"(tok_sf1_307));
                float tok_x11_315 = _mul_208;
                float _mul_209;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_209) : "f"(tok_x00_312), "f"(neg_fused));
                float _exp2_40 = approx_exp2(_mul_209);
                float _add_40;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_40) : "f"(1.0f), "f"(_exp2_40));
                float _rcp_40 = approx_rcp(_add_40);
                float tok_sig00_316 = _rcp_40;
                float _mul_210;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_210) : "f"(tok_x01_313), "f"(neg_fused));
                float _exp2_41 = approx_exp2(_mul_210);
                float _add_41;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_41) : "f"(1.0f), "f"(_exp2_41));
                float _rcp_41 = approx_rcp(_add_41);
                float tok_sig01_317 = _rcp_41;
                float _mul_211;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_211) : "f"(tok_x10_314), "f"(neg_fused));
                float _exp2_42 = approx_exp2(_mul_211);
                float _add_42;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_42) : "f"(1.0f), "f"(_exp2_42));
                float _rcp_42 = approx_rcp(_add_42);
                float tok_sig10_318 = _rcp_42;
                float _mul_212;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_212) : "f"(tok_x11_315), "f"(neg_fused));
                float _exp2_43 = approx_exp2(_mul_212);
                float _add_43;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_43) : "f"(1.0f), "f"(_exp2_43));
                float _rcp_43 = approx_rcp(_add_43);
                float tok_sig11_319 = _rcp_43;
                float _mul_213;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_213) : "f"(tok_x00_312), "f"(tok_sig00_316));
                float _min_84;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_84) : "f"(_mul_213), "f"(cl));
                float tok_g00_320 = _min_84;
                float _mul_214;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_214) : "f"(tok_x01_313), "f"(tok_sig01_317));
                float _min_85;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_85) : "f"(_mul_214), "f"(cl));
                float tok_g01_321 = _min_85;
                float _mul_215;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_215) : "f"(tok_x10_314), "f"(tok_sig10_318));
                float _min_86;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_86) : "f"(_mul_215), "f"(cl));
                float tok_g10_322 = _min_86;
                float _mul_216;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_216) : "f"(tok_x11_315), "f"(tok_sig11_319));
                float _min_87;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_87) : "f"(_mul_216), "f"(cl));
                float tok_g11_323 = _min_87;
                float _fma_40;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_40) : "f"(tok_lin00_308), "f"(sg), "f"(0.0f));
                float _mul_217;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_217) : "f"(_fma_40), "f"(tok_g00_320));
                float tok_acc00_324 = _mul_217;
                float _fma_41;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_41) : "f"(tok_lin01_309), "f"(sg), "f"(0.0f));
                float _mul_218;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_218) : "f"(_fma_41), "f"(tok_g01_321));
                float tok_acc01_325 = _mul_218;
                float _fma_42;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_42) : "f"(tok_lin10_310), "f"(sg), "f"(0.0f));
                float _mul_219;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_219) : "f"(_fma_42), "f"(tok_g10_322));
                float tok_acc10_326 = _mul_219;
                float _fma_43;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_43) : "f"(tok_lin11_311), "f"(sg), "f"(0.0f));
                float _mul_220;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_220) : "f"(_fma_43), "f"(tok_g11_323));
                float tok_acc11_327 = _mul_220;
                float2 _f2_30 = make_float2(sc, sc);
                float2 tok_sc_328 = _f2_30;
                float2 _f2_31 = make_float2(tok_acc00_324, tok_acc10_326);
                float2 _mul_f32x2_20;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_20) : "l"(*(const unsigned long long*)&_f2_31), "l"(*(const unsigned long long*)&tok_sc_328));
                float2 tok_out0_329 = _mul_f32x2_20;
                float2 _f2_32 = make_float2(tok_acc01_325, tok_acc11_327);
                float2 _mul_f32x2_21;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_21) : "l"(*(const unsigned long long*)&_f2_32), "l"(*(const unsigned long long*)&tok_sc_328));
                float2 tok_out1_330 = _mul_f32x2_21;
                float value00_331 = tok_out0_329.x;
                float value10_332 = tok_out0_329.y;
                float value01_333 = tok_out1_330.x;
                float value11_334 = tok_out1_330.y;
                out_pair[0] = value00_331;
                out_pair[1] = value10_332;
                if (token0_304 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token0_304) * M_out + tok_feature_235)))[0]) = _pk;
                    }
                }
                out_pair[0] = value01_333;
                out_pair[1] = value11_334;
                if (token1_305 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token1_305) * M_out + tok_feature_235)))[0]) = _pk;
                    }
                }
                int local_token0_335 = lane_1 % 4 * 2 + 24;
                int local_token1_336 = local_token0_335 + 1;
                int token0_337 = token_block_231 * 64 + local_token0_335;
                int token1_338 = token0_337 + 1;
                float tok_sf0_339 = 0.0f;
                float tok_sf1_340 = 0.0f;
                if (token0_337 < valid_rows) {
                    int tok_route0_11 = route_map[tok_slot_base + token0_337];
                    tok_sf0_339 = SFC[tok_route0_11];
                }
                if (token1_338 < valid_rows) {
                    int tok_route1_11 = route_map[tok_slot_base + token1_338];
                    tok_sf1_340 = SFC[tok_route1_11];
                }
                float _mul_221;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_221) : "f"(_tmem_load_2[12]), "f"(tok_sf0_339));
                float _max_44;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_44) : "f"(_mul_221), "f"(neg_cl));
                float _min_88;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_88) : "f"(_max_44), "f"(cl));
                float tok_lin00_341 = _min_88;
                float _mul_222;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_222) : "f"(_tmem_load_2[13]), "f"(tok_sf1_340));
                float _max_45;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_45) : "f"(_mul_222), "f"(neg_cl));
                float _min_89;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_89) : "f"(_max_45), "f"(cl));
                float tok_lin01_342 = _min_89;
                float _mul_223;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_223) : "f"(_tmem_load_3[12]), "f"(tok_sf0_339));
                float _max_46;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_46) : "f"(_mul_223), "f"(neg_cl));
                float _min_90;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_90) : "f"(_max_46), "f"(cl));
                float tok_lin10_343 = _min_90;
                float _mul_224;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_224) : "f"(_tmem_load_3[13]), "f"(tok_sf1_340));
                float _max_47;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_47) : "f"(_mul_224), "f"(neg_cl));
                float _min_91;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_91) : "f"(_max_47), "f"(cl));
                float tok_lin11_344 = _min_91;
                float _mul_225;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_225) : "f"(_tmem_load_2[14]), "f"(tok_sf0_339));
                float tok_x00_345 = _mul_225;
                float _mul_226;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_226) : "f"(_tmem_load_2[15]), "f"(tok_sf1_340));
                float tok_x01_346 = _mul_226;
                float _mul_227;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_227) : "f"(_tmem_load_3[14]), "f"(tok_sf0_339));
                float tok_x10_347 = _mul_227;
                float _mul_228;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_228) : "f"(_tmem_load_3[15]), "f"(tok_sf1_340));
                float tok_x11_348 = _mul_228;
                float _mul_229;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_229) : "f"(tok_x00_345), "f"(neg_fused));
                float _exp2_44 = approx_exp2(_mul_229);
                float _add_44;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_44) : "f"(1.0f), "f"(_exp2_44));
                float _rcp_44 = approx_rcp(_add_44);
                float tok_sig00_349 = _rcp_44;
                float _mul_230;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_230) : "f"(tok_x01_346), "f"(neg_fused));
                float _exp2_45 = approx_exp2(_mul_230);
                float _add_45;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_45) : "f"(1.0f), "f"(_exp2_45));
                float _rcp_45 = approx_rcp(_add_45);
                float tok_sig01_350 = _rcp_45;
                float _mul_231;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_231) : "f"(tok_x10_347), "f"(neg_fused));
                float _exp2_46 = approx_exp2(_mul_231);
                float _add_46;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_46) : "f"(1.0f), "f"(_exp2_46));
                float _rcp_46 = approx_rcp(_add_46);
                float tok_sig10_351 = _rcp_46;
                float _mul_232;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_232) : "f"(tok_x11_348), "f"(neg_fused));
                float _exp2_47 = approx_exp2(_mul_232);
                float _add_47;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_47) : "f"(1.0f), "f"(_exp2_47));
                float _rcp_47 = approx_rcp(_add_47);
                float tok_sig11_352 = _rcp_47;
                float _mul_233;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_233) : "f"(tok_x00_345), "f"(tok_sig00_349));
                float _min_92;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_92) : "f"(_mul_233), "f"(cl));
                float tok_g00_353 = _min_92;
                float _mul_234;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_234) : "f"(tok_x01_346), "f"(tok_sig01_350));
                float _min_93;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_93) : "f"(_mul_234), "f"(cl));
                float tok_g01_354 = _min_93;
                float _mul_235;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_235) : "f"(tok_x10_347), "f"(tok_sig10_351));
                float _min_94;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_94) : "f"(_mul_235), "f"(cl));
                float tok_g10_355 = _min_94;
                float _mul_236;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_236) : "f"(tok_x11_348), "f"(tok_sig11_352));
                float _min_95;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_95) : "f"(_mul_236), "f"(cl));
                float tok_g11_356 = _min_95;
                float _fma_44;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_44) : "f"(tok_lin00_341), "f"(sg), "f"(0.0f));
                float _mul_237;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_237) : "f"(_fma_44), "f"(tok_g00_353));
                float tok_acc00_357 = _mul_237;
                float _fma_45;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_45) : "f"(tok_lin01_342), "f"(sg), "f"(0.0f));
                float _mul_238;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_238) : "f"(_fma_45), "f"(tok_g01_354));
                float tok_acc01_358 = _mul_238;
                float _fma_46;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_46) : "f"(tok_lin10_343), "f"(sg), "f"(0.0f));
                float _mul_239;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_239) : "f"(_fma_46), "f"(tok_g10_355));
                float tok_acc10_359 = _mul_239;
                float _fma_47;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_47) : "f"(tok_lin11_344), "f"(sg), "f"(0.0f));
                float _mul_240;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_240) : "f"(_fma_47), "f"(tok_g11_356));
                float tok_acc11_360 = _mul_240;
                float2 _f2_33 = make_float2(sc, sc);
                float2 tok_sc_361 = _f2_33;
                float2 _f2_34 = make_float2(tok_acc00_357, tok_acc10_359);
                float2 _mul_f32x2_22;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_22) : "l"(*(const unsigned long long*)&_f2_34), "l"(*(const unsigned long long*)&tok_sc_361));
                float2 tok_out0_362 = _mul_f32x2_22;
                float2 _f2_35 = make_float2(tok_acc01_358, tok_acc11_360);
                float2 _mul_f32x2_23;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_23) : "l"(*(const unsigned long long*)&_f2_35), "l"(*(const unsigned long long*)&tok_sc_361));
                float2 tok_out1_363 = _mul_f32x2_23;
                float value00_364 = tok_out0_362.x;
                float value10_365 = tok_out0_362.y;
                float value01_366 = tok_out1_363.x;
                float value11_367 = tok_out1_363.y;
                out_pair[0] = value00_364;
                out_pair[1] = value10_365;
                if (token0_337 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token0_337) * M_out + tok_feature_235)))[0]) = _pk;
                    }
                }
                out_pair[0] = value01_366;
                out_pair[1] = value11_367;
                if (token1_338 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token1_338) * M_out + tok_feature_235)))[0]) = _pk;
                    }
                }
                int local_token0_368 = lane_1 % 4 * 2 + 32;
                int local_token1_369 = local_token0_368 + 1;
                int token0_370 = token_block_231 * 64 + local_token0_368;
                int token1_371 = token0_370 + 1;
                float tok_sf0_372 = 0.0f;
                float tok_sf1_373 = 0.0f;
                if (token0_370 < valid_rows) {
                    int tok_route0_12 = route_map[tok_slot_base + token0_370];
                    tok_sf0_372 = SFC[tok_route0_12];
                }
                if (token1_371 < valid_rows) {
                    int tok_route1_12 = route_map[tok_slot_base + token1_371];
                    tok_sf1_373 = SFC[tok_route1_12];
                }
                float _mul_241;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_241) : "f"(_tmem_load_2[16]), "f"(tok_sf0_372));
                float _max_48;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_48) : "f"(_mul_241), "f"(neg_cl));
                float _min_96;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_96) : "f"(_max_48), "f"(cl));
                float tok_lin00_374 = _min_96;
                float _mul_242;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_242) : "f"(_tmem_load_2[17]), "f"(tok_sf1_373));
                float _max_49;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_49) : "f"(_mul_242), "f"(neg_cl));
                float _min_97;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_97) : "f"(_max_49), "f"(cl));
                float tok_lin01_375 = _min_97;
                float _mul_243;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_243) : "f"(_tmem_load_3[16]), "f"(tok_sf0_372));
                float _max_50;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_50) : "f"(_mul_243), "f"(neg_cl));
                float _min_98;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_98) : "f"(_max_50), "f"(cl));
                float tok_lin10_376 = _min_98;
                float _mul_244;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_244) : "f"(_tmem_load_3[17]), "f"(tok_sf1_373));
                float _max_51;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_51) : "f"(_mul_244), "f"(neg_cl));
                float _min_99;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_99) : "f"(_max_51), "f"(cl));
                float tok_lin11_377 = _min_99;
                float _mul_245;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_245) : "f"(_tmem_load_2[18]), "f"(tok_sf0_372));
                float tok_x00_378 = _mul_245;
                float _mul_246;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_246) : "f"(_tmem_load_2[19]), "f"(tok_sf1_373));
                float tok_x01_379 = _mul_246;
                float _mul_247;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_247) : "f"(_tmem_load_3[18]), "f"(tok_sf0_372));
                float tok_x10_380 = _mul_247;
                float _mul_248;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_248) : "f"(_tmem_load_3[19]), "f"(tok_sf1_373));
                float tok_x11_381 = _mul_248;
                float _mul_249;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_249) : "f"(tok_x00_378), "f"(neg_fused));
                float _exp2_48 = approx_exp2(_mul_249);
                float _add_48;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_48) : "f"(1.0f), "f"(_exp2_48));
                float _rcp_48 = approx_rcp(_add_48);
                float tok_sig00_382 = _rcp_48;
                float _mul_250;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_250) : "f"(tok_x01_379), "f"(neg_fused));
                float _exp2_49 = approx_exp2(_mul_250);
                float _add_49;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_49) : "f"(1.0f), "f"(_exp2_49));
                float _rcp_49 = approx_rcp(_add_49);
                float tok_sig01_383 = _rcp_49;
                float _mul_251;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_251) : "f"(tok_x10_380), "f"(neg_fused));
                float _exp2_50 = approx_exp2(_mul_251);
                float _add_50;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_50) : "f"(1.0f), "f"(_exp2_50));
                float _rcp_50 = approx_rcp(_add_50);
                float tok_sig10_384 = _rcp_50;
                float _mul_252;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_252) : "f"(tok_x11_381), "f"(neg_fused));
                float _exp2_51 = approx_exp2(_mul_252);
                float _add_51;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_51) : "f"(1.0f), "f"(_exp2_51));
                float _rcp_51 = approx_rcp(_add_51);
                float tok_sig11_385 = _rcp_51;
                float _mul_253;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_253) : "f"(tok_x00_378), "f"(tok_sig00_382));
                float _min_100;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_100) : "f"(_mul_253), "f"(cl));
                float tok_g00_386 = _min_100;
                float _mul_254;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_254) : "f"(tok_x01_379), "f"(tok_sig01_383));
                float _min_101;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_101) : "f"(_mul_254), "f"(cl));
                float tok_g01_387 = _min_101;
                float _mul_255;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_255) : "f"(tok_x10_380), "f"(tok_sig10_384));
                float _min_102;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_102) : "f"(_mul_255), "f"(cl));
                float tok_g10_388 = _min_102;
                float _mul_256;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_256) : "f"(tok_x11_381), "f"(tok_sig11_385));
                float _min_103;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_103) : "f"(_mul_256), "f"(cl));
                float tok_g11_389 = _min_103;
                float _fma_48;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_48) : "f"(tok_lin00_374), "f"(sg), "f"(0.0f));
                float _mul_257;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_257) : "f"(_fma_48), "f"(tok_g00_386));
                float tok_acc00_390 = _mul_257;
                float _fma_49;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_49) : "f"(tok_lin01_375), "f"(sg), "f"(0.0f));
                float _mul_258;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_258) : "f"(_fma_49), "f"(tok_g01_387));
                float tok_acc01_391 = _mul_258;
                float _fma_50;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_50) : "f"(tok_lin10_376), "f"(sg), "f"(0.0f));
                float _mul_259;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_259) : "f"(_fma_50), "f"(tok_g10_388));
                float tok_acc10_392 = _mul_259;
                float _fma_51;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_51) : "f"(tok_lin11_377), "f"(sg), "f"(0.0f));
                float _mul_260;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_260) : "f"(_fma_51), "f"(tok_g11_389));
                float tok_acc11_393 = _mul_260;
                float2 _f2_36 = make_float2(sc, sc);
                float2 tok_sc_394 = _f2_36;
                float2 _f2_37 = make_float2(tok_acc00_390, tok_acc10_392);
                float2 _mul_f32x2_24;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_24) : "l"(*(const unsigned long long*)&_f2_37), "l"(*(const unsigned long long*)&tok_sc_394));
                float2 tok_out0_395 = _mul_f32x2_24;
                float2 _f2_38 = make_float2(tok_acc01_391, tok_acc11_393);
                float2 _mul_f32x2_25;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_25) : "l"(*(const unsigned long long*)&_f2_38), "l"(*(const unsigned long long*)&tok_sc_394));
                float2 tok_out1_396 = _mul_f32x2_25;
                float value00_397 = tok_out0_395.x;
                float value10_398 = tok_out0_395.y;
                float value01_399 = tok_out1_396.x;
                float value11_400 = tok_out1_396.y;
                out_pair[0] = value00_397;
                out_pair[1] = value10_398;
                if (token0_370 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token0_370) * M_out + tok_feature_235)))[0]) = _pk;
                    }
                }
                out_pair[0] = value01_399;
                out_pair[1] = value11_400;
                if (token1_371 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token1_371) * M_out + tok_feature_235)))[0]) = _pk;
                    }
                }
                int local_token0_401 = lane_1 % 4 * 2 + 40;
                int local_token1_402 = local_token0_401 + 1;
                int token0_403 = token_block_231 * 64 + local_token0_401;
                int token1_404 = token0_403 + 1;
                float tok_sf0_405 = 0.0f;
                float tok_sf1_406 = 0.0f;
                if (token0_403 < valid_rows) {
                    int tok_route0_13 = route_map[tok_slot_base + token0_403];
                    tok_sf0_405 = SFC[tok_route0_13];
                }
                if (token1_404 < valid_rows) {
                    int tok_route1_13 = route_map[tok_slot_base + token1_404];
                    tok_sf1_406 = SFC[tok_route1_13];
                }
                float _mul_261;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_261) : "f"(_tmem_load_2[20]), "f"(tok_sf0_405));
                float _max_52;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_52) : "f"(_mul_261), "f"(neg_cl));
                float _min_104;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_104) : "f"(_max_52), "f"(cl));
                float tok_lin00_407 = _min_104;
                float _mul_262;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_262) : "f"(_tmem_load_2[21]), "f"(tok_sf1_406));
                float _max_53;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_53) : "f"(_mul_262), "f"(neg_cl));
                float _min_105;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_105) : "f"(_max_53), "f"(cl));
                float tok_lin01_408 = _min_105;
                float _mul_263;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_263) : "f"(_tmem_load_3[20]), "f"(tok_sf0_405));
                float _max_54;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_54) : "f"(_mul_263), "f"(neg_cl));
                float _min_106;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_106) : "f"(_max_54), "f"(cl));
                float tok_lin10_409 = _min_106;
                float _mul_264;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_264) : "f"(_tmem_load_3[21]), "f"(tok_sf1_406));
                float _max_55;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_55) : "f"(_mul_264), "f"(neg_cl));
                float _min_107;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_107) : "f"(_max_55), "f"(cl));
                float tok_lin11_410 = _min_107;
                float _mul_265;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_265) : "f"(_tmem_load_2[22]), "f"(tok_sf0_405));
                float tok_x00_411 = _mul_265;
                float _mul_266;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_266) : "f"(_tmem_load_2[23]), "f"(tok_sf1_406));
                float tok_x01_412 = _mul_266;
                float _mul_267;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_267) : "f"(_tmem_load_3[22]), "f"(tok_sf0_405));
                float tok_x10_413 = _mul_267;
                float _mul_268;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_268) : "f"(_tmem_load_3[23]), "f"(tok_sf1_406));
                float tok_x11_414 = _mul_268;
                float _mul_269;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_269) : "f"(tok_x00_411), "f"(neg_fused));
                float _exp2_52 = approx_exp2(_mul_269);
                float _add_52;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_52) : "f"(1.0f), "f"(_exp2_52));
                float _rcp_52 = approx_rcp(_add_52);
                float tok_sig00_415 = _rcp_52;
                float _mul_270;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_270) : "f"(tok_x01_412), "f"(neg_fused));
                float _exp2_53 = approx_exp2(_mul_270);
                float _add_53;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_53) : "f"(1.0f), "f"(_exp2_53));
                float _rcp_53 = approx_rcp(_add_53);
                float tok_sig01_416 = _rcp_53;
                float _mul_271;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_271) : "f"(tok_x10_413), "f"(neg_fused));
                float _exp2_54 = approx_exp2(_mul_271);
                float _add_54;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_54) : "f"(1.0f), "f"(_exp2_54));
                float _rcp_54 = approx_rcp(_add_54);
                float tok_sig10_417 = _rcp_54;
                float _mul_272;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_272) : "f"(tok_x11_414), "f"(neg_fused));
                float _exp2_55 = approx_exp2(_mul_272);
                float _add_55;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_55) : "f"(1.0f), "f"(_exp2_55));
                float _rcp_55 = approx_rcp(_add_55);
                float tok_sig11_418 = _rcp_55;
                float _mul_273;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_273) : "f"(tok_x00_411), "f"(tok_sig00_415));
                float _min_108;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_108) : "f"(_mul_273), "f"(cl));
                float tok_g00_419 = _min_108;
                float _mul_274;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_274) : "f"(tok_x01_412), "f"(tok_sig01_416));
                float _min_109;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_109) : "f"(_mul_274), "f"(cl));
                float tok_g01_420 = _min_109;
                float _mul_275;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_275) : "f"(tok_x10_413), "f"(tok_sig10_417));
                float _min_110;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_110) : "f"(_mul_275), "f"(cl));
                float tok_g10_421 = _min_110;
                float _mul_276;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_276) : "f"(tok_x11_414), "f"(tok_sig11_418));
                float _min_111;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_111) : "f"(_mul_276), "f"(cl));
                float tok_g11_422 = _min_111;
                float _fma_52;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_52) : "f"(tok_lin00_407), "f"(sg), "f"(0.0f));
                float _mul_277;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_277) : "f"(_fma_52), "f"(tok_g00_419));
                float tok_acc00_423 = _mul_277;
                float _fma_53;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_53) : "f"(tok_lin01_408), "f"(sg), "f"(0.0f));
                float _mul_278;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_278) : "f"(_fma_53), "f"(tok_g01_420));
                float tok_acc01_424 = _mul_278;
                float _fma_54;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_54) : "f"(tok_lin10_409), "f"(sg), "f"(0.0f));
                float _mul_279;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_279) : "f"(_fma_54), "f"(tok_g10_421));
                float tok_acc10_425 = _mul_279;
                float _fma_55;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_55) : "f"(tok_lin11_410), "f"(sg), "f"(0.0f));
                float _mul_280;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_280) : "f"(_fma_55), "f"(tok_g11_422));
                float tok_acc11_426 = _mul_280;
                float2 _f2_39 = make_float2(sc, sc);
                float2 tok_sc_427 = _f2_39;
                float2 _f2_40 = make_float2(tok_acc00_423, tok_acc10_425);
                float2 _mul_f32x2_26;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_26) : "l"(*(const unsigned long long*)&_f2_40), "l"(*(const unsigned long long*)&tok_sc_427));
                float2 tok_out0_428 = _mul_f32x2_26;
                float2 _f2_41 = make_float2(tok_acc01_424, tok_acc11_426);
                float2 _mul_f32x2_27;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_27) : "l"(*(const unsigned long long*)&_f2_41), "l"(*(const unsigned long long*)&tok_sc_427));
                float2 tok_out1_429 = _mul_f32x2_27;
                float value00_430 = tok_out0_428.x;
                float value10_431 = tok_out0_428.y;
                float value01_432 = tok_out1_429.x;
                float value11_433 = tok_out1_429.y;
                out_pair[0] = value00_430;
                out_pair[1] = value10_431;
                if (token0_403 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token0_403) * M_out + tok_feature_235)))[0]) = _pk;
                    }
                }
                out_pair[0] = value01_432;
                out_pair[1] = value11_433;
                if (token1_404 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token1_404) * M_out + tok_feature_235)))[0]) = _pk;
                    }
                }
                int local_token0_434 = lane_1 % 4 * 2 + 48;
                int local_token1_435 = local_token0_434 + 1;
                int token0_436 = token_block_231 * 64 + local_token0_434;
                int token1_437 = token0_436 + 1;
                float tok_sf0_438 = 0.0f;
                float tok_sf1_439 = 0.0f;
                if (token0_436 < valid_rows) {
                    int tok_route0_14 = route_map[tok_slot_base + token0_436];
                    tok_sf0_438 = SFC[tok_route0_14];
                }
                if (token1_437 < valid_rows) {
                    int tok_route1_14 = route_map[tok_slot_base + token1_437];
                    tok_sf1_439 = SFC[tok_route1_14];
                }
                float _mul_281;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_281) : "f"(_tmem_load_2[24]), "f"(tok_sf0_438));
                float _max_56;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_56) : "f"(_mul_281), "f"(neg_cl));
                float _min_112;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_112) : "f"(_max_56), "f"(cl));
                float tok_lin00_440 = _min_112;
                float _mul_282;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_282) : "f"(_tmem_load_2[25]), "f"(tok_sf1_439));
                float _max_57;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_57) : "f"(_mul_282), "f"(neg_cl));
                float _min_113;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_113) : "f"(_max_57), "f"(cl));
                float tok_lin01_441 = _min_113;
                float _mul_283;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_283) : "f"(_tmem_load_3[24]), "f"(tok_sf0_438));
                float _max_58;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_58) : "f"(_mul_283), "f"(neg_cl));
                float _min_114;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_114) : "f"(_max_58), "f"(cl));
                float tok_lin10_442 = _min_114;
                float _mul_284;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_284) : "f"(_tmem_load_3[25]), "f"(tok_sf1_439));
                float _max_59;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_59) : "f"(_mul_284), "f"(neg_cl));
                float _min_115;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_115) : "f"(_max_59), "f"(cl));
                float tok_lin11_443 = _min_115;
                float _mul_285;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_285) : "f"(_tmem_load_2[26]), "f"(tok_sf0_438));
                float tok_x00_444 = _mul_285;
                float _mul_286;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_286) : "f"(_tmem_load_2[27]), "f"(tok_sf1_439));
                float tok_x01_445 = _mul_286;
                float _mul_287;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_287) : "f"(_tmem_load_3[26]), "f"(tok_sf0_438));
                float tok_x10_446 = _mul_287;
                float _mul_288;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_288) : "f"(_tmem_load_3[27]), "f"(tok_sf1_439));
                float tok_x11_447 = _mul_288;
                float _mul_289;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_289) : "f"(tok_x00_444), "f"(neg_fused));
                float _exp2_56 = approx_exp2(_mul_289);
                float _add_56;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_56) : "f"(1.0f), "f"(_exp2_56));
                float _rcp_56 = approx_rcp(_add_56);
                float tok_sig00_448 = _rcp_56;
                float _mul_290;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_290) : "f"(tok_x01_445), "f"(neg_fused));
                float _exp2_57 = approx_exp2(_mul_290);
                float _add_57;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_57) : "f"(1.0f), "f"(_exp2_57));
                float _rcp_57 = approx_rcp(_add_57);
                float tok_sig01_449 = _rcp_57;
                float _mul_291;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_291) : "f"(tok_x10_446), "f"(neg_fused));
                float _exp2_58 = approx_exp2(_mul_291);
                float _add_58;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_58) : "f"(1.0f), "f"(_exp2_58));
                float _rcp_58 = approx_rcp(_add_58);
                float tok_sig10_450 = _rcp_58;
                float _mul_292;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_292) : "f"(tok_x11_447), "f"(neg_fused));
                float _exp2_59 = approx_exp2(_mul_292);
                float _add_59;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_59) : "f"(1.0f), "f"(_exp2_59));
                float _rcp_59 = approx_rcp(_add_59);
                float tok_sig11_451 = _rcp_59;
                float _mul_293;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_293) : "f"(tok_x00_444), "f"(tok_sig00_448));
                float _min_116;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_116) : "f"(_mul_293), "f"(cl));
                float tok_g00_452 = _min_116;
                float _mul_294;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_294) : "f"(tok_x01_445), "f"(tok_sig01_449));
                float _min_117;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_117) : "f"(_mul_294), "f"(cl));
                float tok_g01_453 = _min_117;
                float _mul_295;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_295) : "f"(tok_x10_446), "f"(tok_sig10_450));
                float _min_118;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_118) : "f"(_mul_295), "f"(cl));
                float tok_g10_454 = _min_118;
                float _mul_296;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_296) : "f"(tok_x11_447), "f"(tok_sig11_451));
                float _min_119;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_119) : "f"(_mul_296), "f"(cl));
                float tok_g11_455 = _min_119;
                float _fma_56;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_56) : "f"(tok_lin00_440), "f"(sg), "f"(0.0f));
                float _mul_297;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_297) : "f"(_fma_56), "f"(tok_g00_452));
                float tok_acc00_456 = _mul_297;
                float _fma_57;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_57) : "f"(tok_lin01_441), "f"(sg), "f"(0.0f));
                float _mul_298;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_298) : "f"(_fma_57), "f"(tok_g01_453));
                float tok_acc01_457 = _mul_298;
                float _fma_58;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_58) : "f"(tok_lin10_442), "f"(sg), "f"(0.0f));
                float _mul_299;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_299) : "f"(_fma_58), "f"(tok_g10_454));
                float tok_acc10_458 = _mul_299;
                float _fma_59;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_59) : "f"(tok_lin11_443), "f"(sg), "f"(0.0f));
                float _mul_300;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_300) : "f"(_fma_59), "f"(tok_g11_455));
                float tok_acc11_459 = _mul_300;
                float2 _f2_42 = make_float2(sc, sc);
                float2 tok_sc_460 = _f2_42;
                float2 _f2_43 = make_float2(tok_acc00_456, tok_acc10_458);
                float2 _mul_f32x2_28;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_28) : "l"(*(const unsigned long long*)&_f2_43), "l"(*(const unsigned long long*)&tok_sc_460));
                float2 tok_out0_461 = _mul_f32x2_28;
                float2 _f2_44 = make_float2(tok_acc01_457, tok_acc11_459);
                float2 _mul_f32x2_29;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_29) : "l"(*(const unsigned long long*)&_f2_44), "l"(*(const unsigned long long*)&tok_sc_460));
                float2 tok_out1_462 = _mul_f32x2_29;
                float value00_463 = tok_out0_461.x;
                float value10_464 = tok_out0_461.y;
                float value01_465 = tok_out1_462.x;
                float value11_466 = tok_out1_462.y;
                out_pair[0] = value00_463;
                out_pair[1] = value10_464;
                if (token0_436 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token0_436) * M_out + tok_feature_235)))[0]) = _pk;
                    }
                }
                out_pair[0] = value01_465;
                out_pair[1] = value11_466;
                if (token1_437 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token1_437) * M_out + tok_feature_235)))[0]) = _pk;
                    }
                }
                int local_token0_467 = lane_1 % 4 * 2 + 56;
                int local_token1_468 = local_token0_467 + 1;
                int token0_469 = token_block_231 * 64 + local_token0_467;
                int token1_470 = token0_469 + 1;
                float tok_sf0_471 = 0.0f;
                float tok_sf1_472 = 0.0f;
                if (token0_469 < valid_rows) {
                    int tok_route0_15 = route_map[tok_slot_base + token0_469];
                    tok_sf0_471 = SFC[tok_route0_15];
                }
                if (token1_470 < valid_rows) {
                    int tok_route1_15 = route_map[tok_slot_base + token1_470];
                    tok_sf1_472 = SFC[tok_route1_15];
                }
                float _mul_301;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_301) : "f"(_tmem_load_2[28]), "f"(tok_sf0_471));
                float _max_60;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_60) : "f"(_mul_301), "f"(neg_cl));
                float _min_120;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_120) : "f"(_max_60), "f"(cl));
                float tok_lin00_473 = _min_120;
                float _mul_302;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_302) : "f"(_tmem_load_2[29]), "f"(tok_sf1_472));
                float _max_61;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_61) : "f"(_mul_302), "f"(neg_cl));
                float _min_121;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_121) : "f"(_max_61), "f"(cl));
                float tok_lin01_474 = _min_121;
                float _mul_303;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_303) : "f"(_tmem_load_3[28]), "f"(tok_sf0_471));
                float _max_62;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_62) : "f"(_mul_303), "f"(neg_cl));
                float _min_122;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_122) : "f"(_max_62), "f"(cl));
                float tok_lin10_475 = _min_122;
                float _mul_304;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_304) : "f"(_tmem_load_3[29]), "f"(tok_sf1_472));
                float _max_63;
                asm("max.ftz.f32 %0, %1, %2;" : "=f"(_max_63) : "f"(_mul_304), "f"(neg_cl));
                float _min_123;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_123) : "f"(_max_63), "f"(cl));
                float tok_lin11_476 = _min_123;
                float _mul_305;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_305) : "f"(_tmem_load_2[30]), "f"(tok_sf0_471));
                float tok_x00_477 = _mul_305;
                float _mul_306;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_306) : "f"(_tmem_load_2[31]), "f"(tok_sf1_472));
                float tok_x01_478 = _mul_306;
                float _mul_307;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_307) : "f"(_tmem_load_3[30]), "f"(tok_sf0_471));
                float tok_x10_479 = _mul_307;
                float _mul_308;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_308) : "f"(_tmem_load_3[31]), "f"(tok_sf1_472));
                float tok_x11_480 = _mul_308;
                float _mul_309;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_309) : "f"(tok_x00_477), "f"(neg_fused));
                float _exp2_60 = approx_exp2(_mul_309);
                float _add_60;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_60) : "f"(1.0f), "f"(_exp2_60));
                float _rcp_60 = approx_rcp(_add_60);
                float tok_sig00_481 = _rcp_60;
                float _mul_310;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_310) : "f"(tok_x01_478), "f"(neg_fused));
                float _exp2_61 = approx_exp2(_mul_310);
                float _add_61;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_61) : "f"(1.0f), "f"(_exp2_61));
                float _rcp_61 = approx_rcp(_add_61);
                float tok_sig01_482 = _rcp_61;
                float _mul_311;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_311) : "f"(tok_x10_479), "f"(neg_fused));
                float _exp2_62 = approx_exp2(_mul_311);
                float _add_62;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_62) : "f"(1.0f), "f"(_exp2_62));
                float _rcp_62 = approx_rcp(_add_62);
                float tok_sig10_483 = _rcp_62;
                float _mul_312;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_312) : "f"(tok_x11_480), "f"(neg_fused));
                float _exp2_63 = approx_exp2(_mul_312);
                float _add_63;
                asm("add.rn.ftz.f32 %0, %1, %2;" : "=f"(_add_63) : "f"(1.0f), "f"(_exp2_63));
                float _rcp_63 = approx_rcp(_add_63);
                float tok_sig11_484 = _rcp_63;
                float _mul_313;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_313) : "f"(tok_x00_477), "f"(tok_sig00_481));
                float _min_124;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_124) : "f"(_mul_313), "f"(cl));
                float tok_g00_485 = _min_124;
                float _mul_314;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_314) : "f"(tok_x01_478), "f"(tok_sig01_482));
                float _min_125;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_125) : "f"(_mul_314), "f"(cl));
                float tok_g01_486 = _min_125;
                float _mul_315;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_315) : "f"(tok_x10_479), "f"(tok_sig10_483));
                float _min_126;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_126) : "f"(_mul_315), "f"(cl));
                float tok_g10_487 = _min_126;
                float _mul_316;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_316) : "f"(tok_x11_480), "f"(tok_sig11_484));
                float _min_127;
                asm("min.ftz.f32 %0, %1, %2;" : "=f"(_min_127) : "f"(_mul_316), "f"(cl));
                float tok_g11_488 = _min_127;
                float _fma_60;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_60) : "f"(tok_lin00_473), "f"(sg), "f"(0.0f));
                float _mul_317;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_317) : "f"(_fma_60), "f"(tok_g00_485));
                float tok_acc00_489 = _mul_317;
                float _fma_61;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_61) : "f"(tok_lin01_474), "f"(sg), "f"(0.0f));
                float _mul_318;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_318) : "f"(_fma_61), "f"(tok_g01_486));
                float tok_acc01_490 = _mul_318;
                float _fma_62;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_62) : "f"(tok_lin10_475), "f"(sg), "f"(0.0f));
                float _mul_319;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_319) : "f"(_fma_62), "f"(tok_g10_487));
                float tok_acc10_491 = _mul_319;
                float _fma_63;
                asm("fma.rn.ftz.f32 %0, %1, %2, %3;" : "=f"(_fma_63) : "f"(tok_lin11_476), "f"(sg), "f"(0.0f));
                float _mul_320;
                asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(_mul_320) : "f"(_fma_63), "f"(tok_g11_488));
                float tok_acc11_492 = _mul_320;
                float2 _f2_45 = make_float2(sc, sc);
                float2 tok_sc_493 = _f2_45;
                float2 _f2_46 = make_float2(tok_acc00_489, tok_acc10_491);
                float2 _mul_f32x2_30;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_30) : "l"(*(const unsigned long long*)&_f2_46), "l"(*(const unsigned long long*)&tok_sc_493));
                float2 tok_out0_494 = _mul_f32x2_30;
                float2 _f2_47 = make_float2(tok_acc01_490, tok_acc11_492);
                float2 _mul_f32x2_31;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_31) : "l"(*(const unsigned long long*)&_f2_47), "l"(*(const unsigned long long*)&tok_sc_493));
                float2 tok_out1_495 = _mul_f32x2_31;
                float value00_496 = tok_out0_494.x;
                float value10_497 = tok_out0_494.y;
                float value01_498 = tok_out1_495.x;
                float value11_499 = tok_out1_495.y;
                out_pair[0] = value00_496;
                out_pair[1] = value10_497;
                if (token0_469 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token0_469) * M_out + tok_feature_235)))[0]) = _pk;
                    }
                }
                out_pair[0] = value01_498;
                out_pair[1] = value11_499;
                if (token1_470 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token1_470) * M_out + tok_feature_235)))[0]) = _pk;
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
