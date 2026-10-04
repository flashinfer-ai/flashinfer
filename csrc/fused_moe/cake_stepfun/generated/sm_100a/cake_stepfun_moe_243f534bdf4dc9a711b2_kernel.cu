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
#define TMEM_NCOLS 320
#define TMEM_ACCUM_OFFSET 0
#define TMEM_SFA_OFFSET 128
#define TMEM_SFB_OFFSET 256
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
#define SMEM_EPI_STAGING_STAGE_BYTES 1024
#define SMEM_EPI_STAGING_STRIDE 1024
#define SMEM_SMEM_SFA_OFF 168448
#define SMEM_SMEM_SFA_STAGE_BYTES 4096
#define SMEM_SMEM_SFA_STRIDE 4096
#define SMEM_SMEM_SFB_OFF 184832
#define SMEM_SMEM_SFB_STAGE_BYTES 2048
#define SMEM_SMEM_SFB_STRIDE 2048
#define SMEM_WORK_RESPONSE_OFF 193024
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_FAST_RESPONSE_OFF 193072
#define SMEM_FAST_RESPONSE_STAGE_BYTES 64
#define SMEM_FAST_RESPONSE_STRIDE 64
#define SMEM_TOTAL 193152
#define THREADS 896
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






__device__ __forceinline__ void tma_gather4_gmem2smem(
    int dst, const void *tmap_ptr,
    int col_idx, int row0, int row1, int row2, int row3,
    int mbar_addr) {
    // Canonical .shared::cta form for non-multicast gather4, matching
    // trtllm-gen / cuda_ptx and the PTX ISA qualifier order
    // (dim.dst.src.load_mode.completion_mechanism). Per the PTX grammar,
    // .shared::cluster is reserved for the multicast variant (ctaMask).
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
        :: "r"(dst), "l"(tmap_ptr), "r"(col_idx),
           "r"(row0), "r"(row1), "r"(row2), "r"(row3),
           "r"(mbar_addr) : "memory");
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

__global__ __launch_bounds__(896, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_cake_stepfun_moe_243f534bdf4dc9a711b2(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, const __grid_constant__ CUtensorMap C, uint8_t* __restrict__ SFC, int* __restrict__ route_map, int* __restrict__ tile_expert, int* __restrict__ tile_mn_limit, int* __restrict__ num_non_exiting_ctas, float* __restrict__ scale_c, float* __restrict__ scale_gate, float* __restrict__ clamp_limit, float* __restrict__ act_alpha, float* __restrict__ act_beta, int M_out, int K, int grid_m, int grid_n, int K_tiles)
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
    uint8_t* smem_sfa = reinterpret_cast<uint8_t*>(smem_raw + 168448);
    const int smem_sfa_addr = smem + 168448;
    uint8_t* smem_sfb = reinterpret_cast<uint8_t*>(smem_raw + 184832);
    const int smem_sfb_addr = smem + 184832;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 193024);
    const int work_response_addr = smem + 193024;
    unsigned int* fast_response = reinterpret_cast<unsigned int*>(smem_raw + 193072);
    const int fast_response_addr = smem + 193072;
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
            // work_empty: 3 barriers, init_count=1696
            mbarrier_init(smem + 344, 1696);
            mbarrier_init(smem + 352, 1696);
            mbarrier_init(smem + 360, 1696);
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

    // TMEM alloc (512 columns, 320 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 424);
    if (warp == 0) {
        int _tmem_hold = smem + 424;
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
    const int tmem_sfa = taddr + 128;
    const int tmem_sfb = taddr + 256;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 20 && warp <= 27) {
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
            float quant_pair[8] = {0};
            unsigned int _phase_mma_full = 0;
            unsigned int _phase_work_full = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter = 0; _tile_iter < grid_m / 2 * grid_n; _tile_iter++) {
                if (m_tile >= (unsigned int)grid_m || n_tile >= (unsigned int)grid_n) {
                    break;
                }
                int expert = tile_expert[n_tile];
                int valid_rows = (unsigned int)tile_mn_limit[n_tile] - n_tile * 64;
                float sc = scale_c[expert];
                float sg = scale_gate[expert];
                float cl = clamp_limit[expert];
                float al = act_alpha[expert];
                float be = act_beta[expert];
                float neg_cl = -cl;
                float beta_sg = be * sg;
                float alpha_sg = al * sg;
                mbarrier_wait(mma_full_addr + (acc_stage) * 8, _phase_mma_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int acc_offset = acc_stage * 64;
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
                int base_row = warp_0 * 16 + lane_1 / 4 * 2;
                asm volatile("cp.async.bulk.wait_group.read 0;");
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                int token0 = lane_1 % 4 * 2;
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
                    int sf_feature = m_tile * 4 + (unsigned int)warp_0;
                    int sf_token_group = token0 / 8;
                    int sf_tile_stride = 8 * (M_out / 64) * 32;
                    int sf_base = n_tile * (unsigned int)sf_tile_stride + (unsigned int)(sf_token_group * (M_out / 64) * 32) + (unsigned int)(sf_feature / 4 * 32) + (unsigned int)(token0 % 8 * 4) + (unsigned int)(sf_feature % 4);
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
                            *(reinterpret_cast<unsigned char*>(SFC + (sf_base + 4)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                        }
                    }
                }
                int local_token0 = token0;
                int local_token1 = local_token0 + 1;
                int smem_flat0 = local_token0 * 64 + base_row;
                int smem_flat1 = local_token1 * 64 + base_row;
                int smem_index0 = smem_flat0 / 2 ^ smem_flat0 / 256 % 2 * 16;
                int smem_index1 = smem_flat1 / 2 ^ smem_flat1 / 256 % 2 * 16;
                epi_staging[smem_index0] = _fp4_0[0];
                epi_staging[smem_index1] = _fp4_1[0];
                int token0_0 = lane_1 % 4 * 2 + 8;
                int token1_1 = token0_0 + 1;
                float _max_12 = max_noftz(_tmem_load_0[4], neg_cl);
                float _min_8 = fminf(_max_12, cl);
                float step_lin00_2 = _min_8;
                float _max_13 = max_noftz(_tmem_load_0[5], neg_cl);
                float _min_9 = fminf(_max_13, cl);
                float step_lin01_3 = _min_9;
                float _max_14 = max_noftz(_tmem_load_1[4], neg_cl);
                float _min_10 = fminf(_max_14, cl);
                float step_lin10_4 = _min_10;
                float _max_15 = max_noftz(_tmem_load_1[5], neg_cl);
                float _min_11 = fminf(_max_15, cl);
                float step_lin11_5 = _min_11;
                float step_x00_6 = _tmem_load_0[6];
                float step_x01_7 = _tmem_load_0[7];
                float step_x10_8 = _tmem_load_1[6];
                float step_x11_9 = _tmem_load_1[7];
                float _exp2_4 = approx_exp2((-(step_x00_6 * sg)) * 1.4426950408889634f);
                float _rcp_4 = approx_rcp(1.0f + _exp2_4);
                float step_sig00_10 = _rcp_4;
                float _exp2_5 = approx_exp2((-(step_x01_7 * sg)) * 1.4426950408889634f);
                float _rcp_5 = approx_rcp(1.0f + _exp2_5);
                float step_sig01_11 = _rcp_5;
                float _exp2_6 = approx_exp2((-(step_x10_8 * sg)) * 1.4426950408889634f);
                float _rcp_6 = approx_rcp(1.0f + _exp2_6);
                float step_sig10_12 = _rcp_6;
                float _exp2_7 = approx_exp2((-(step_x11_9 * sg)) * 1.4426950408889634f);
                float _rcp_7 = approx_rcp(1.0f + _exp2_7);
                float step_sig11_13 = _rcp_7;
                float _min_12 = fminf(step_x00_6 * step_sig00_10, cl);
                float step_g00_14 = _min_12;
                float _min_13 = fminf(step_x01_7 * step_sig01_11, cl);
                float step_g01_15 = _min_13;
                float _min_14 = fminf(step_x10_8 * step_sig10_12, cl);
                float step_g10_16 = _min_14;
                float _min_15 = fminf(step_x11_9 * step_sig11_13, cl);
                float step_g11_17 = _min_15;
                float value00_18 = step_lin00_2 * sc * sg * step_g00_14;
                float value01_19 = step_lin01_3 * sc * sg * step_g01_15;
                float value10_20 = step_lin10_4 * sc * sg * step_g10_16;
                float value11_21 = step_lin11_5 * sc * sg * step_g11_17;
                float _fabs_4 = fabsf(value00_18);
                float _fabs_5 = fabsf(value10_20);
                float _max_16 = max_noftz(_fabs_4, _fabs_5);
                float block_max0_22 = _max_16;
                float _fabs_6 = fabsf(value01_19);
                float _fabs_7 = fabsf(value11_21);
                float _max_17 = max_noftz(_fabs_6, _fabs_7);
                float block_max1_23 = _max_17;
                float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, block_max0_22, 4);
                float _max_18 = max_noftz(block_max0_22, _shfl_xor_6);
                block_max0_22 = _max_18;
                float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, block_max1_23, 4);
                float _max_19 = max_noftz(block_max1_23, _shfl_xor_7);
                block_max1_23 = _max_19;
                float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, block_max0_22, 8);
                float _max_20 = max_noftz(block_max0_22, _shfl_xor_8);
                block_max0_22 = _max_20;
                float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, block_max1_23, 8);
                float _max_21 = max_noftz(block_max1_23, _shfl_xor_9);
                block_max1_23 = _max_21;
                float _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, block_max0_22, 16);
                float _max_22 = max_noftz(block_max0_22, _shfl_xor_10);
                block_max0_22 = _max_22;
                float _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, block_max1_23, 16);
                float _max_23 = max_noftz(block_max1_23, _shfl_xor_11);
                block_max1_23 = _max_23;
                float _fp8_rt_2;
                uint16_t _e4m3x2_4;
                uint32_t _f16x2_4;
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_4) : "f"(0.0f), "f"(block_max0_22 * 0.16666666666666666f));
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_4) : "h"(_e4m3x2_4));
                uint16_t _fp8_h0_4 = (uint16_t)(_f16x2_4 & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_2) : "h"(_fp8_h0_4));
                float scale0_24 = _fp8_rt_2;
                float _fp8_rt_3;
                uint16_t _e4m3x2_5;
                uint32_t _f16x2_5;
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_5) : "f"(0.0f), "f"(block_max1_23 * 0.16666666666666666f));
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_5) : "h"(_e4m3x2_5));
                uint16_t _fp8_h0_5 = (uint16_t)(_f16x2_5 & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_3) : "h"(_fp8_h0_5));
                float scale1_25 = _fp8_rt_3;
                float inv_scale0_26 = 0.0f;
                float inv_scale1_27 = 0.0f;
                if (scale0_24 != 0.0f) {
                    inv_scale0_26 = 1.0f / scale0_24;
                }
                if (scale1_25 != 0.0f) {
                    inv_scale1_27 = 1.0f / scale1_25;
                }
                quant_pair[0] = value00_18 * inv_scale0_26;
                quant_pair[1] = value10_20 * inv_scale0_26;
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
                quant_pair[0] = value01_19 * inv_scale1_27;
                quant_pair[1] = value11_21 * inv_scale1_27;
                uint32_t _fp4_3[1];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_3[0]) : "f"(quant_pair[0]), "f"(quant_pair[1]), "f"(quant_pair[2]), "f"(quant_pair[3]), "f"(quant_pair[4]), "f"(quant_pair[5]), "f"(quant_pair[6]), "f"(quant_pair[7]));
                if (lane_1 < 4) {
                    int sf_feature_1 = m_tile * 4 + (unsigned int)warp_0;
                    int sf_token_group_1 = token0_0 / 8;
                    int sf_tile_stride_1 = 8 * (M_out / 64) * 32;
                    int sf_base_1 = n_tile * (unsigned int)sf_tile_stride_1 + (unsigned int)(sf_token_group_1 * (M_out / 64) * 32) + (unsigned int)(sf_feature_1 / 4 * 32) + (unsigned int)(token0_0 % 8 * 4) + (unsigned int)(sf_feature_1 % 4);
                    if (token0_0 < valid_rows) {
                        {
                            unsigned short _fp8_pair;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(scale0_24));
                            *(reinterpret_cast<unsigned char*>(SFC + sf_base_1) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                        }
                    }
                    if (token1_1 < valid_rows) {
                        {
                            unsigned short _fp8_pair;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(scale1_25));
                            *(reinterpret_cast<unsigned char*>(SFC + (sf_base_1 + 4)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                        }
                    }
                }
                int local_token0_28 = token0_0;
                int local_token1_29 = local_token0_28 + 1;
                int smem_flat0_30 = local_token0_28 * 64 + base_row;
                int smem_flat1_31 = local_token1_29 * 64 + base_row;
                int smem_index0_32 = smem_flat0_30 / 2 ^ smem_flat0_30 / 256 % 2 * 16;
                int smem_index1_33 = smem_flat1_31 / 2 ^ smem_flat1_31 / 256 % 2 * 16;
                epi_staging[smem_index0_32] = _fp4_2[0];
                epi_staging[smem_index1_33] = _fp4_3[0];
                int token0_34 = lane_1 % 4 * 2 + 16;
                int token1_35 = token0_34 + 1;
                float _max_24 = max_noftz(_tmem_load_0[8], neg_cl);
                float _min_16 = fminf(_max_24, cl);
                float step_lin00_36 = _min_16;
                float _max_25 = max_noftz(_tmem_load_0[9], neg_cl);
                float _min_17 = fminf(_max_25, cl);
                float step_lin01_37 = _min_17;
                float _max_26 = max_noftz(_tmem_load_1[8], neg_cl);
                float _min_18 = fminf(_max_26, cl);
                float step_lin10_38 = _min_18;
                float _max_27 = max_noftz(_tmem_load_1[9], neg_cl);
                float _min_19 = fminf(_max_27, cl);
                float step_lin11_39 = _min_19;
                float step_x00_40 = _tmem_load_0[10];
                float step_x01_41 = _tmem_load_0[11];
                float step_x10_42 = _tmem_load_1[10];
                float step_x11_43 = _tmem_load_1[11];
                float _exp2_8 = approx_exp2((-(step_x00_40 * sg)) * 1.4426950408889634f);
                float _rcp_8 = approx_rcp(1.0f + _exp2_8);
                float step_sig00_44 = _rcp_8;
                float _exp2_9 = approx_exp2((-(step_x01_41 * sg)) * 1.4426950408889634f);
                float _rcp_9 = approx_rcp(1.0f + _exp2_9);
                float step_sig01_45 = _rcp_9;
                float _exp2_10 = approx_exp2((-(step_x10_42 * sg)) * 1.4426950408889634f);
                float _rcp_10 = approx_rcp(1.0f + _exp2_10);
                float step_sig10_46 = _rcp_10;
                float _exp2_11 = approx_exp2((-(step_x11_43 * sg)) * 1.4426950408889634f);
                float _rcp_11 = approx_rcp(1.0f + _exp2_11);
                float step_sig11_47 = _rcp_11;
                float _min_20 = fminf(step_x00_40 * step_sig00_44, cl);
                float step_g00_48 = _min_20;
                float _min_21 = fminf(step_x01_41 * step_sig01_45, cl);
                float step_g01_49 = _min_21;
                float _min_22 = fminf(step_x10_42 * step_sig10_46, cl);
                float step_g10_50 = _min_22;
                float _min_23 = fminf(step_x11_43 * step_sig11_47, cl);
                float step_g11_51 = _min_23;
                float value00_52 = step_lin00_36 * sc * sg * step_g00_48;
                float value01_53 = step_lin01_37 * sc * sg * step_g01_49;
                float value10_54 = step_lin10_38 * sc * sg * step_g10_50;
                float value11_55 = step_lin11_39 * sc * sg * step_g11_51;
                float _fabs_8 = fabsf(value00_52);
                float _fabs_9 = fabsf(value10_54);
                float _max_28 = max_noftz(_fabs_8, _fabs_9);
                float block_max0_56 = _max_28;
                float _fabs_10 = fabsf(value01_53);
                float _fabs_11 = fabsf(value11_55);
                float _max_29 = max_noftz(_fabs_10, _fabs_11);
                float block_max1_57 = _max_29;
                float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, block_max0_56, 4);
                float _max_30 = max_noftz(block_max0_56, _shfl_xor_12);
                block_max0_56 = _max_30;
                float _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, block_max1_57, 4);
                float _max_31 = max_noftz(block_max1_57, _shfl_xor_13);
                block_max1_57 = _max_31;
                float _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, block_max0_56, 8);
                float _max_32 = max_noftz(block_max0_56, _shfl_xor_14);
                block_max0_56 = _max_32;
                float _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, block_max1_57, 8);
                float _max_33 = max_noftz(block_max1_57, _shfl_xor_15);
                block_max1_57 = _max_33;
                float _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, block_max0_56, 16);
                float _max_34 = max_noftz(block_max0_56, _shfl_xor_16);
                block_max0_56 = _max_34;
                float _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, block_max1_57, 16);
                float _max_35 = max_noftz(block_max1_57, _shfl_xor_17);
                block_max1_57 = _max_35;
                float _fp8_rt_4;
                uint16_t _e4m3x2_8;
                uint32_t _f16x2_8;
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_8) : "f"(0.0f), "f"(block_max0_56 * 0.16666666666666666f));
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_8) : "h"(_e4m3x2_8));
                uint16_t _fp8_h0_8 = (uint16_t)(_f16x2_8 & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_4) : "h"(_fp8_h0_8));
                float scale0_58 = _fp8_rt_4;
                float _fp8_rt_5;
                uint16_t _e4m3x2_9;
                uint32_t _f16x2_9;
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_9) : "f"(0.0f), "f"(block_max1_57 * 0.16666666666666666f));
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_9) : "h"(_e4m3x2_9));
                uint16_t _fp8_h0_9 = (uint16_t)(_f16x2_9 & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_5) : "h"(_fp8_h0_9));
                float scale1_59 = _fp8_rt_5;
                float inv_scale0_60 = 0.0f;
                float inv_scale1_61 = 0.0f;
                if (scale0_58 != 0.0f) {
                    inv_scale0_60 = 1.0f / scale0_58;
                }
                if (scale1_59 != 0.0f) {
                    inv_scale1_61 = 1.0f / scale1_59;
                }
                quant_pair[0] = value00_52 * inv_scale0_60;
                quant_pair[1] = value10_54 * inv_scale0_60;
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
                quant_pair[0] = value01_53 * inv_scale1_61;
                quant_pair[1] = value11_55 * inv_scale1_61;
                uint32_t _fp4_5[1];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_5[0]) : "f"(quant_pair[0]), "f"(quant_pair[1]), "f"(quant_pair[2]), "f"(quant_pair[3]), "f"(quant_pair[4]), "f"(quant_pair[5]), "f"(quant_pair[6]), "f"(quant_pair[7]));
                if (lane_1 < 4) {
                    int sf_feature_2 = m_tile * 4 + (unsigned int)warp_0;
                    int sf_token_group_2 = token0_34 / 8;
                    int sf_tile_stride_2 = 8 * (M_out / 64) * 32;
                    int sf_base_2 = n_tile * (unsigned int)sf_tile_stride_2 + (unsigned int)(sf_token_group_2 * (M_out / 64) * 32) + (unsigned int)(sf_feature_2 / 4 * 32) + (unsigned int)(token0_34 % 8 * 4) + (unsigned int)(sf_feature_2 % 4);
                    if (token0_34 < valid_rows) {
                        {
                            unsigned short _fp8_pair;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(scale0_58));
                            *(reinterpret_cast<unsigned char*>(SFC + sf_base_2) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                        }
                    }
                    if (token1_35 < valid_rows) {
                        {
                            unsigned short _fp8_pair;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(scale1_59));
                            *(reinterpret_cast<unsigned char*>(SFC + (sf_base_2 + 4)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                        }
                    }
                }
                int local_token0_62 = token0_34;
                int local_token1_63 = local_token0_62 + 1;
                int smem_flat0_64 = local_token0_62 * 64 + base_row;
                int smem_flat1_65 = local_token1_63 * 64 + base_row;
                int smem_index0_66 = smem_flat0_64 / 2 ^ smem_flat0_64 / 256 % 2 * 16;
                int smem_index1_67 = smem_flat1_65 / 2 ^ smem_flat1_65 / 256 % 2 * 16;
                epi_staging[smem_index0_66] = _fp4_4[0];
                epi_staging[smem_index1_67] = _fp4_5[0];
                int token0_68 = lane_1 % 4 * 2 + 24;
                int token1_69 = token0_68 + 1;
                float _max_36 = max_noftz(_tmem_load_0[12], neg_cl);
                float _min_24 = fminf(_max_36, cl);
                float step_lin00_70 = _min_24;
                float _max_37 = max_noftz(_tmem_load_0[13], neg_cl);
                float _min_25 = fminf(_max_37, cl);
                float step_lin01_71 = _min_25;
                float _max_38 = max_noftz(_tmem_load_1[12], neg_cl);
                float _min_26 = fminf(_max_38, cl);
                float step_lin10_72 = _min_26;
                float _max_39 = max_noftz(_tmem_load_1[13], neg_cl);
                float _min_27 = fminf(_max_39, cl);
                float step_lin11_73 = _min_27;
                float step_x00_74 = _tmem_load_0[14];
                float step_x01_75 = _tmem_load_0[15];
                float step_x10_76 = _tmem_load_1[14];
                float step_x11_77 = _tmem_load_1[15];
                float _exp2_12 = approx_exp2((-(step_x00_74 * sg)) * 1.4426950408889634f);
                float _rcp_12 = approx_rcp(1.0f + _exp2_12);
                float step_sig00_78 = _rcp_12;
                float _exp2_13 = approx_exp2((-(step_x01_75 * sg)) * 1.4426950408889634f);
                float _rcp_13 = approx_rcp(1.0f + _exp2_13);
                float step_sig01_79 = _rcp_13;
                float _exp2_14 = approx_exp2((-(step_x10_76 * sg)) * 1.4426950408889634f);
                float _rcp_14 = approx_rcp(1.0f + _exp2_14);
                float step_sig10_80 = _rcp_14;
                float _exp2_15 = approx_exp2((-(step_x11_77 * sg)) * 1.4426950408889634f);
                float _rcp_15 = approx_rcp(1.0f + _exp2_15);
                float step_sig11_81 = _rcp_15;
                float _min_28 = fminf(step_x00_74 * step_sig00_78, cl);
                float step_g00_82 = _min_28;
                float _min_29 = fminf(step_x01_75 * step_sig01_79, cl);
                float step_g01_83 = _min_29;
                float _min_30 = fminf(step_x10_76 * step_sig10_80, cl);
                float step_g10_84 = _min_30;
                float _min_31 = fminf(step_x11_77 * step_sig11_81, cl);
                float step_g11_85 = _min_31;
                float value00_86 = step_lin00_70 * sc * sg * step_g00_82;
                float value01_87 = step_lin01_71 * sc * sg * step_g01_83;
                float value10_88 = step_lin10_72 * sc * sg * step_g10_84;
                float value11_89 = step_lin11_73 * sc * sg * step_g11_85;
                float _fabs_12 = fabsf(value00_86);
                float _fabs_13 = fabsf(value10_88);
                float _max_40 = max_noftz(_fabs_12, _fabs_13);
                float block_max0_90 = _max_40;
                float _fabs_14 = fabsf(value01_87);
                float _fabs_15 = fabsf(value11_89);
                float _max_41 = max_noftz(_fabs_14, _fabs_15);
                float block_max1_91 = _max_41;
                float _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, block_max0_90, 4);
                float _max_42 = max_noftz(block_max0_90, _shfl_xor_18);
                block_max0_90 = _max_42;
                float _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, block_max1_91, 4);
                float _max_43 = max_noftz(block_max1_91, _shfl_xor_19);
                block_max1_91 = _max_43;
                float _shfl_xor_20 = __shfl_xor_sync(0xFFFFFFFF, block_max0_90, 8);
                float _max_44 = max_noftz(block_max0_90, _shfl_xor_20);
                block_max0_90 = _max_44;
                float _shfl_xor_21 = __shfl_xor_sync(0xFFFFFFFF, block_max1_91, 8);
                float _max_45 = max_noftz(block_max1_91, _shfl_xor_21);
                block_max1_91 = _max_45;
                float _shfl_xor_22 = __shfl_xor_sync(0xFFFFFFFF, block_max0_90, 16);
                float _max_46 = max_noftz(block_max0_90, _shfl_xor_22);
                block_max0_90 = _max_46;
                float _shfl_xor_23 = __shfl_xor_sync(0xFFFFFFFF, block_max1_91, 16);
                float _max_47 = max_noftz(block_max1_91, _shfl_xor_23);
                block_max1_91 = _max_47;
                float _fp8_rt_6;
                uint16_t _e4m3x2_12;
                uint32_t _f16x2_12;
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_12) : "f"(0.0f), "f"(block_max0_90 * 0.16666666666666666f));
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_12) : "h"(_e4m3x2_12));
                uint16_t _fp8_h0_12 = (uint16_t)(_f16x2_12 & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_6) : "h"(_fp8_h0_12));
                float scale0_92 = _fp8_rt_6;
                float _fp8_rt_7;
                uint16_t _e4m3x2_13;
                uint32_t _f16x2_13;
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_13) : "f"(0.0f), "f"(block_max1_91 * 0.16666666666666666f));
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_13) : "h"(_e4m3x2_13));
                uint16_t _fp8_h0_13 = (uint16_t)(_f16x2_13 & 0xFFFFu);
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_7) : "h"(_fp8_h0_13));
                float scale1_93 = _fp8_rt_7;
                float inv_scale0_94 = 0.0f;
                float inv_scale1_95 = 0.0f;
                if (scale0_92 != 0.0f) {
                    inv_scale0_94 = 1.0f / scale0_92;
                }
                if (scale1_93 != 0.0f) {
                    inv_scale1_95 = 1.0f / scale1_93;
                }
                quant_pair[0] = value00_86 * inv_scale0_94;
                quant_pair[1] = value10_88 * inv_scale0_94;
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
                quant_pair[0] = value01_87 * inv_scale1_95;
                quant_pair[1] = value11_89 * inv_scale1_95;
                uint32_t _fp4_7[1];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_7[0]) : "f"(quant_pair[0]), "f"(quant_pair[1]), "f"(quant_pair[2]), "f"(quant_pair[3]), "f"(quant_pair[4]), "f"(quant_pair[5]), "f"(quant_pair[6]), "f"(quant_pair[7]));
                if (lane_1 < 4) {
                    int sf_feature_3 = m_tile * 4 + (unsigned int)warp_0;
                    int sf_token_group_3 = token0_68 / 8;
                    int sf_tile_stride_3 = 8 * (M_out / 64) * 32;
                    int sf_base_3 = n_tile * (unsigned int)sf_tile_stride_3 + (unsigned int)(sf_token_group_3 * (M_out / 64) * 32) + (unsigned int)(sf_feature_3 / 4 * 32) + (unsigned int)(token0_68 % 8 * 4) + (unsigned int)(sf_feature_3 % 4);
                    if (token0_68 < valid_rows) {
                        {
                            unsigned short _fp8_pair;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(scale0_92));
                            *(reinterpret_cast<unsigned char*>(SFC + sf_base_3) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                        }
                    }
                    if (token1_69 < valid_rows) {
                        {
                            unsigned short _fp8_pair;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(scale1_93));
                            *(reinterpret_cast<unsigned char*>(SFC + (sf_base_3 + 4)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                        }
                    }
                }
                int local_token0_96 = token0_68;
                int local_token1_97 = local_token0_96 + 1;
                int smem_flat0_98 = local_token0_96 * 64 + base_row;
                int smem_flat1_99 = local_token1_97 * 64 + base_row;
                int smem_index0_100 = smem_flat0_98 / 2 ^ smem_flat0_98 / 256 % 2 * 16;
                int smem_index1_101 = smem_flat1_99 / 2 ^ smem_flat1_99 / 256 % 2 * 16;
                epi_staging[smem_index0_100] = _fp4_6[0];
                epi_staging[smem_index1_101] = _fp4_7[0];
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
                int acc_offset_102 = acc_stage * 64 + 32;
                float _tmem_load_2[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[15]))
                    : "r"(taddr + (unsigned int)acc_offset_102));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float _tmem_load_3[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[15]))
                    : "r"(taddr + 1048576 + (unsigned int)acc_offset_102));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int base_row_103 = warp_0 * 16 + lane_1 / 4 * 2;
                asm volatile("cp.async.bulk.wait_group.read 0;");
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                int token0_104 = lane_1 % 4 * 2 + 32;
                int token1_105 = token0_104 + 1;
                float _max_48 = max_noftz(_tmem_load_2[0], neg_cl);
                float _min_32 = fminf(_max_48, cl);
                float step_lin00_106 = _min_32;
                float _max_49 = max_noftz(_tmem_load_2[1], neg_cl);
                float _min_33 = fminf(_max_49, cl);
                float step_lin01_107 = _min_33;
                float _max_50 = max_noftz(_tmem_load_3[0], neg_cl);
                float _min_34 = fminf(_max_50, cl);
                float step_lin10_108 = _min_34;
                float _max_51 = max_noftz(_tmem_load_3[1], neg_cl);
                float _min_35 = fminf(_max_51, cl);
                float step_lin11_109 = _min_35;
                float step_x00_110 = _tmem_load_2[2];
                float step_x01_111 = _tmem_load_2[3];
                float step_x10_112 = _tmem_load_3[2];
                float step_x11_113 = _tmem_load_3[3];
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
                    int sf_feature_4 = m_tile * 4 + (unsigned int)warp_0;
                    int sf_token_group_4 = token0_104 / 8;
                    int sf_tile_stride_4 = 8 * (M_out / 64) * 32;
                    int sf_base_4 = n_tile * (unsigned int)sf_tile_stride_4 + (unsigned int)(sf_token_group_4 * (M_out / 64) * 32) + (unsigned int)(sf_feature_4 / 4 * 32) + (unsigned int)(token0_104 % 8 * 4) + (unsigned int)(sf_feature_4 % 4);
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
                            *(reinterpret_cast<unsigned char*>(SFC + (sf_base_4 + 4)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                        }
                    }
                }
                int local_token0_132 = token0_104 - 32;
                int local_token1_133 = local_token0_132 + 1;
                int smem_flat0_134 = local_token0_132 * 64 + base_row_103;
                int smem_flat1_135 = local_token1_133 * 64 + base_row_103;
                int smem_index0_136 = smem_flat0_134 / 2 ^ smem_flat0_134 / 256 % 2 * 16;
                int smem_index1_137 = smem_flat1_135 / 2 ^ smem_flat1_135 / 256 % 2 * 16;
                epi_staging[smem_index0_136] = _fp4_8[0];
                epi_staging[smem_index1_137] = _fp4_9[0];
                int token0_138 = lane_1 % 4 * 2 + 8 + 32;
                int token1_139 = token0_138 + 1;
                float _max_60 = max_noftz(_tmem_load_2[4], neg_cl);
                float _min_40 = fminf(_max_60, cl);
                float step_lin00_140 = _min_40;
                float _max_61 = max_noftz(_tmem_load_2[5], neg_cl);
                float _min_41 = fminf(_max_61, cl);
                float step_lin01_141 = _min_41;
                float _max_62 = max_noftz(_tmem_load_3[4], neg_cl);
                float _min_42 = fminf(_max_62, cl);
                float step_lin10_142 = _min_42;
                float _max_63 = max_noftz(_tmem_load_3[5], neg_cl);
                float _min_43 = fminf(_max_63, cl);
                float step_lin11_143 = _min_43;
                float step_x00_144 = _tmem_load_2[6];
                float step_x01_145 = _tmem_load_2[7];
                float step_x10_146 = _tmem_load_3[6];
                float step_x11_147 = _tmem_load_3[7];
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
                    int sf_feature_5 = m_tile * 4 + (unsigned int)warp_0;
                    int sf_token_group_5 = token0_138 / 8;
                    int sf_tile_stride_5 = 8 * (M_out / 64) * 32;
                    int sf_base_5 = n_tile * (unsigned int)sf_tile_stride_5 + (unsigned int)(sf_token_group_5 * (M_out / 64) * 32) + (unsigned int)(sf_feature_5 / 4 * 32) + (unsigned int)(token0_138 % 8 * 4) + (unsigned int)(sf_feature_5 % 4);
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
                            *(reinterpret_cast<unsigned char*>(SFC + (sf_base_5 + 4)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                        }
                    }
                }
                int local_token0_166 = token0_138 - 32;
                int local_token1_167 = local_token0_166 + 1;
                int smem_flat0_168 = local_token0_166 * 64 + base_row_103;
                int smem_flat1_169 = local_token1_167 * 64 + base_row_103;
                int smem_index0_170 = smem_flat0_168 / 2 ^ smem_flat0_168 / 256 % 2 * 16;
                int smem_index1_171 = smem_flat1_169 / 2 ^ smem_flat1_169 / 256 % 2 * 16;
                epi_staging[smem_index0_170] = _fp4_10[0];
                epi_staging[smem_index1_171] = _fp4_11[0];
                int token0_172 = lane_1 % 4 * 2 + 16 + 32;
                int token1_173 = token0_172 + 1;
                float _max_72 = max_noftz(_tmem_load_2[8], neg_cl);
                float _min_48 = fminf(_max_72, cl);
                float step_lin00_174 = _min_48;
                float _max_73 = max_noftz(_tmem_load_2[9], neg_cl);
                float _min_49 = fminf(_max_73, cl);
                float step_lin01_175 = _min_49;
                float _max_74 = max_noftz(_tmem_load_3[8], neg_cl);
                float _min_50 = fminf(_max_74, cl);
                float step_lin10_176 = _min_50;
                float _max_75 = max_noftz(_tmem_load_3[9], neg_cl);
                float _min_51 = fminf(_max_75, cl);
                float step_lin11_177 = _min_51;
                float step_x00_178 = _tmem_load_2[10];
                float step_x01_179 = _tmem_load_2[11];
                float step_x10_180 = _tmem_load_3[10];
                float step_x11_181 = _tmem_load_3[11];
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
                    int sf_feature_6 = m_tile * 4 + (unsigned int)warp_0;
                    int sf_token_group_6 = token0_172 / 8;
                    int sf_tile_stride_6 = 8 * (M_out / 64) * 32;
                    int sf_base_6 = n_tile * (unsigned int)sf_tile_stride_6 + (unsigned int)(sf_token_group_6 * (M_out / 64) * 32) + (unsigned int)(sf_feature_6 / 4 * 32) + (unsigned int)(token0_172 % 8 * 4) + (unsigned int)(sf_feature_6 % 4);
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
                            *(reinterpret_cast<unsigned char*>(SFC + (sf_base_6 + 4)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                        }
                    }
                }
                int local_token0_200 = token0_172 - 32;
                int local_token1_201 = local_token0_200 + 1;
                int smem_flat0_202 = local_token0_200 * 64 + base_row_103;
                int smem_flat1_203 = local_token1_201 * 64 + base_row_103;
                int smem_index0_204 = smem_flat0_202 / 2 ^ smem_flat0_202 / 256 % 2 * 16;
                int smem_index1_205 = smem_flat1_203 / 2 ^ smem_flat1_203 / 256 % 2 * 16;
                epi_staging[smem_index0_204] = _fp4_12[0];
                epi_staging[smem_index1_205] = _fp4_13[0];
                int token0_206 = lane_1 % 4 * 2 + 24 + 32;
                int token1_207 = token0_206 + 1;
                float _max_84 = max_noftz(_tmem_load_2[12], neg_cl);
                float _min_56 = fminf(_max_84, cl);
                float step_lin00_208 = _min_56;
                float _max_85 = max_noftz(_tmem_load_2[13], neg_cl);
                float _min_57 = fminf(_max_85, cl);
                float step_lin01_209 = _min_57;
                float _max_86 = max_noftz(_tmem_load_3[12], neg_cl);
                float _min_58 = fminf(_max_86, cl);
                float step_lin10_210 = _min_58;
                float _max_87 = max_noftz(_tmem_load_3[13], neg_cl);
                float _min_59 = fminf(_max_87, cl);
                float step_lin11_211 = _min_59;
                float step_x00_212 = _tmem_load_2[14];
                float step_x01_213 = _tmem_load_2[15];
                float step_x10_214 = _tmem_load_3[14];
                float step_x11_215 = _tmem_load_3[15];
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
                    int sf_feature_7 = m_tile * 4 + (unsigned int)warp_0;
                    int sf_token_group_7 = token0_206 / 8;
                    int sf_tile_stride_7 = 8 * (M_out / 64) * 32;
                    int sf_base_7 = n_tile * (unsigned int)sf_tile_stride_7 + (unsigned int)(sf_token_group_7 * (M_out / 64) * 32) + (unsigned int)(sf_feature_7 / 4 * 32) + (unsigned int)(token0_206 % 8 * 4) + (unsigned int)(sf_feature_7 % 4);
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
                            *(reinterpret_cast<unsigned char*>(SFC + (sf_base_7 + 4)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                        }
                    }
                }
                int local_token0_234 = token0_206 - 32;
                int local_token1_235 = local_token0_234 + 1;
                int smem_flat0_236 = local_token0_234 * 64 + base_row_103;
                int smem_flat1_237 = local_token1_235 * 64 + base_row_103;
                int smem_index0_238 = smem_flat0_236 / 2 ^ smem_flat0_236 / 256 % 2 * 16;
                int smem_index1_239 = smem_flat1_237 / 2 ^ smem_flat1_237 / 256 % 2 * 16;
                epi_staging[smem_index0_238] = _fp4_14[0];
                epi_staging[smem_index1_239] = _fp4_15[0];
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                if (warp == 0) {
                    if (elect_sync()) {
                        int padding_rows_1 = (64 - valid_rows % 64) % 64;
                        tma_store_4d((&C), m_tile * 64, padding_rows_1 + 32, 1073741824, n_tile * 64 - (unsigned int)padding_rows_1 + 1073741824, epi_staging_addr);
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
                uint32_t _clc_ctaid_14 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_14)
                    : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_15 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_15)
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
                if (((_clc_ctaid_15 < (unsigned int)num_non_exiting_ctas[0]) ? _clc_valid_7 : (unsigned int)0) == 0) {
                    break;
                }
                m_tile = _clc_ctaid_14 + (unsigned int)cta_rank;
                n_tile = _clc_ctaid_15;
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
                if (m_tile_1 >= (unsigned int)grid_m || n_tile_1 >= (unsigned int)grid_n) {
                    break;
                }
                #pragma unroll 1
                for (int _iter_k = 0; _iter_k < K_tiles; _iter_k++) {
                    mbarrier_wait(sfb_full_addr + (stage) * 8, _phase_sfb_full);
                    mbarrier_wait(k_done_addr + (stage) * 8, _phase_k_done);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    for (int reg = 0; reg < 2; reg++) {
                        const int logical_lane = lane_0 + reg * 32;
                        {
                            const int vec_half = 0;
                            const int vec_word = 0;
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&word[reg])) : "r"(smem_sfb_addr + stage * 2048 + (unsigned int)((logical_lane * 2 + vec_half ^ logical_lane / 4 % 2) * 16) + (unsigned int)(vec_word * 4)));
                        }
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x2.b32"
                        " [%0], {%1, %2};"
                        :: "r"(taddr + 256 + stage * 16), "r"(word[0]), "r"(word[1]));
                    for (int reg_1 = 0; reg_1 < 2; reg_1++) {
                        const int logical_lane_1 = lane_0 + reg_1 * 32;
                        {
                            const int vec_half_1 = 0;
                            const int vec_word_1 = 1;
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&word[reg_1])) : "r"(smem_sfb_addr + stage * 2048 + (unsigned int)((logical_lane_1 * 2 + vec_half_1 ^ logical_lane_1 / 4 % 2) * 16) + (unsigned int)(vec_word_1 * 4)));
                        }
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x2.b32"
                        " [%0], {%1, %2};"
                        :: "r"(taddr + 256 + stage * 16 + 2), "r"(word[0]), "r"(word[1]));
                    for (int reg_2 = 0; reg_2 < 2; reg_2++) {
                        const int logical_lane_2 = lane_0 + reg_2 * 32;
                        {
                            const int vec_half_2 = 0;
                            const int vec_word_2 = 2;
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&word[reg_2])) : "r"(smem_sfb_addr + stage * 2048 + (unsigned int)((logical_lane_2 * 2 + vec_half_2 ^ logical_lane_2 / 4 % 2) * 16) + (unsigned int)(vec_word_2 * 4)));
                        }
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x2.b32"
                        " [%0], {%1, %2};"
                        :: "r"(taddr + 256 + stage * 16 + 4), "r"(word[0]), "r"(word[1]));
                    for (int reg_3 = 0; reg_3 < 2; reg_3++) {
                        const int logical_lane_3 = lane_0 + reg_3 * 32;
                        {
                            const int vec_half_3 = 0;
                            const int vec_word_3 = 3;
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&word[reg_3])) : "r"(smem_sfb_addr + stage * 2048 + (unsigned int)((logical_lane_3 * 2 + vec_half_3 ^ logical_lane_3 / 4 % 2) * 16) + (unsigned int)(vec_word_3 * 4)));
                        }
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x2.b32"
                        " [%0], {%1, %2};"
                        :: "r"(taddr + 256 + stage * 16 + 6), "r"(word[0]), "r"(word[1]));
                    for (int reg_4 = 0; reg_4 < 2; reg_4++) {
                        const int logical_lane_4 = lane_0 + reg_4 * 32;
                        {
                            const int vec_half_4 = 1;
                            const int vec_word_4 = 0;
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&word[reg_4])) : "r"(smem_sfb_addr + stage * 2048 + (unsigned int)((logical_lane_4 * 2 + vec_half_4 ^ logical_lane_4 / 4 % 2) * 16) + (unsigned int)(vec_word_4 * 4)));
                        }
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x2.b32"
                        " [%0], {%1, %2};"
                        :: "r"(taddr + 256 + stage * 16 + 8), "r"(word[0]), "r"(word[1]));
                    for (int reg_5 = 0; reg_5 < 2; reg_5++) {
                        const int logical_lane_5 = lane_0 + reg_5 * 32;
                        {
                            const int vec_half_5 = 1;
                            const int vec_word_5 = 1;
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&word[reg_5])) : "r"(smem_sfb_addr + stage * 2048 + (unsigned int)((logical_lane_5 * 2 + vec_half_5 ^ logical_lane_5 / 4 % 2) * 16) + (unsigned int)(vec_word_5 * 4)));
                        }
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x2.b32"
                        " [%0], {%1, %2};"
                        :: "r"(taddr + 256 + stage * 16 + 10), "r"(word[0]), "r"(word[1]));
                    for (int reg_6 = 0; reg_6 < 2; reg_6++) {
                        const int logical_lane_6 = lane_0 + reg_6 * 32;
                        {
                            const int vec_half_6 = 1;
                            const int vec_word_6 = 2;
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&word[reg_6])) : "r"(smem_sfb_addr + stage * 2048 + (unsigned int)((logical_lane_6 * 2 + vec_half_6 ^ logical_lane_6 / 4 % 2) * 16) + (unsigned int)(vec_word_6 * 4)));
                        }
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x2.b32"
                        " [%0], {%1, %2};"
                        :: "r"(taddr + 256 + stage * 16 + 12), "r"(word[0]), "r"(word[1]));
                    for (int reg_7 = 0; reg_7 < 2; reg_7++) {
                        const int logical_lane_7 = lane_0 + reg_7 * 32;
                        {
                            const int vec_half_7 = 1;
                            const int vec_word_7 = 3;
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&word[reg_7])) : "r"(smem_sfb_addr + stage * 2048 + (unsigned int)((logical_lane_7 * 2 + vec_half_7 ^ logical_lane_7 / 4 % 2) * 16) + (unsigned int)(vec_word_7 * 4)));
                        }
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x2.b32"
                        " [%0], {%1, %2};"
                        :: "r"(taddr + 256 + stage * 16 + 14), "r"(word[0]), "r"(word[1]));
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
                    : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
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
                if (((_clc_ctaid_11 < (unsigned int)num_non_exiting_ctas[0]) ? _clc_valid_5 : (unsigned int)0) == 0) {
                    break;
                }
                m_tile_1 = _clc_ctaid_10 + (unsigned int)cta_rank;
                n_tile_1 = _clc_ctaid_11;
            }
        }
    }
    // ---- Role: load_b ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
        { // load_b_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int stage_1 = 0;
            unsigned int work_stage_2 = 0;
            unsigned int m_tile_2 = blockIdx.x;
            unsigned int n_tile_2 = blockIdx.y;
            int warp_local = warp - 8;
            int route_base = 0;
            int routed[8];
            unsigned int cta_mask = 1 << cta_rank;
            unsigned int _phase_k_done_1 = 1;
            unsigned int _phase_work_full_2 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_2 = 0; _tile_iter_2 < grid_m / 2 * grid_n; _tile_iter_2++) {
                if (m_tile_2 >= (unsigned int)grid_m || n_tile_2 >= (unsigned int)grid_n) {
                    break;
                }
                route_base = n_tile_2 * 64 + (unsigned int)(cta_rank * 32) + (unsigned int)(warp_local * 4);
                for (int row = 0; row < 4; row++) {
                    routed[row] = route_map[route_base + row];
                }
                route_base = n_tile_2 * 64 + (unsigned int)(cta_rank * 32) + (unsigned int)((4 + warp_local) * 4);
                for (int row_1 = 0; row_1 < 4; row_1++) {
                    routed[4 + row_1] = route_map[route_base + row_1];
                }
                #pragma unroll 1
                for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                    mbarrier_wait(k_done_addr + (stage_1) * 8, _phase_k_done_1);
                    if (elect_sync()) {
                        tma_gather4_gmem2smem_mc_cta2(smem_b_addr + stage_1 * 8192 + (unsigned int)(warp_local * 512), (&B), iter_k * 256, routed[0], routed[1], routed[2], routed[3], ((b_full_addr + (stage_1) * 8) & 0xFEFFFFFF), cta_mask);
                        tma_gather4_gmem2smem_mc_cta2(smem_b_addr + stage_1 * 8192 + 4096 + (unsigned int)(warp_local * 512), (&B), iter_k * 256 + 128, routed[0], routed[1], routed[2], routed[3], ((b_full_addr + (stage_1) * 8) & 0xFEFFFFFF), cta_mask);
                        tma_gather4_gmem2smem_mc_cta2(smem_b_addr + stage_1 * 8192 + (unsigned int)((4 + warp_local) * 512), (&B), iter_k * 256, routed[4], routed[5], routed[6], routed[7], ((b_full_addr + (stage_1) * 8) & 0xFEFFFFFF), cta_mask);
                        tma_gather4_gmem2smem_mc_cta2(smem_b_addr + stage_1 * 8192 + 4096 + (unsigned int)((4 + warp_local) * 512), (&B), iter_k * 256 + 128, routed[4], routed[5], routed[6], routed[7], ((b_full_addr + (stage_1) * 8) & 0xFEFFFFFF), cta_mask);
                    }
                    if (warp == 8) {
                        if (elect_sync()) {
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((b_full_addr + (stage_1) * 8) & 0xFEFFFFFF), "r"((uint32_t)(8192)) : "memory");
                        }
                    }
                    stage_1 += 1;
                    if (stage_1 == 4) { stage_1 = 0; _phase_k_done_1 ^= 1; }
                }
                mbarrier_wait(work_full_addr + (work_stage_2) * 8, _phase_work_full_2);
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
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
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
                if (((_clc_ctaid_3 < (unsigned int)num_non_exiting_ctas[0]) ? _clc_valid_1 : (unsigned int)0) == 0) {
                    break;
                }
                m_tile_2 = _clc_ctaid_2 + (unsigned int)cta_rank;
                n_tile_2 = _clc_ctaid_3;
            }
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
    }
    // ---- Role: load_sfb ----
    if (warp >= 12 && warp <= 19) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
        { // load_sfb_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int stage_2 = 0;
            unsigned int work_stage_3 = 0;
            unsigned int m_tile_3 = blockIdx.x;
            unsigned int n_tile_3 = blockIdx.y;
            int warp_local_1 = warp - 12;
            int routed_1[8];
            unsigned int _phase_sfb_free = 1;
            unsigned int _phase_work_full_3 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_3 = 0; _tile_iter_3 < grid_m / 2 * grid_n; _tile_iter_3++) {
                if (m_tile_3 >= (unsigned int)grid_m || n_tile_3 >= (unsigned int)grid_n) {
                    break;
                }
                int route_base_1 = 0;
                route_base_1 = n_tile_3 * 64 + (unsigned int)(warp_local_1 * 4);
                for (int row_2 = 0; row_2 < 4; row_2++) {
                    routed_1[row_2] = route_map[route_base_1 + row_2];
                }
                route_base_1 = n_tile_3 * 64 + (unsigned int)((8 + warp_local_1) * 4);
                for (int row_3 = 0; row_3 < 4; row_3++) {
                    routed_1[4 + row_3] = route_map[route_base_1 + row_3];
                }
                #pragma unroll 1
                for (int iter_k_1 = 0; iter_k_1 < K_tiles; iter_k_1++) {
                    mbarrier_wait(sfb_free_addr + (stage_2) * 8, _phase_sfb_free);
                    if (elect_sync()) {
                        tma_gather4_gmem2smem(smem_sfb_addr + stage_2 * 2048 + (unsigned int)(warp_local_1 * 128), (&SFB), iter_k_1 * 32, routed_1[0], routed_1[1], routed_1[2], routed_1[3], sfb_full_addr + (stage_2) * 8);
                        tma_gather4_gmem2smem(smem_sfb_addr + stage_2 * 2048 + (unsigned int)((8 + warp_local_1) * 128), (&SFB), iter_k_1 * 32, routed_1[4], routed_1[5], routed_1[6], routed_1[7], sfb_full_addr + (stage_2) * 8);
                    }
                    if (warp == 12) {
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(sfb_full_addr + (stage_2) * 8, 2048);
                        }
                    }
                    stage_2 += 1;
                    if (stage_2 == 4) { stage_2 = 0; _phase_sfb_free ^= 1; }
                }
                mbarrier_wait(work_full_addr + (work_stage_3) * 8, _phase_work_full_3);
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
                    : "r"(work_response_addr + work_stage_3 * 16 + 0 * 16)
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
                if (((_clc_ctaid_7 < (unsigned int)num_non_exiting_ctas[0]) ? _clc_valid_3 : (unsigned int)0) == 0) {
                    break;
                }
                m_tile_3 = _clc_ctaid_6 + (unsigned int)cta_rank;
                n_tile_3 = _clc_ctaid_7;
            }
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
    }
    // ---- Role: load_a ----
    if (warp == 20) {
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
                if (m_tile_4 >= (unsigned int)grid_m || n_tile_4 >= (unsigned int)grid_n) {
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
                    : "r"(work_response_addr + work_stage_4 * 16 + 0 * 16)
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
                if (work_stage_4 == 3) { work_stage_4 = 0; _phase_work_full_4 ^= 1; }
                if (((_clc_ctaid_1 < (unsigned int)num_non_exiting_ctas[0]) ? _clc_valid_0 : (unsigned int)0) == 0) {
                    break;
                }
                m_tile_4 = _clc_ctaid_0 + (unsigned int)cta_rank;
                n_tile_4 = _clc_ctaid_1;
            }
        }
    }
    // ---- Role: load_sfa ----
    if (warp == 21) {
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
                if (m_tile_5 >= (unsigned int)grid_m || n_tile_5 >= (unsigned int)grid_n) {
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
                            :: "r"(smem_sfa_addr + stage_4 * 4096), "l"((&SFA)), "r"(0), "r"(0), "r"(iter_k_3 * 8), "r"((unsigned int)(expert_2 * grid_m) + m_tile_5),
                               "r"(((sfa_full_addr + (stage_4) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(cta_mask_2)), "l"(0x12F0000000000000ULL) : "memory");
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((sfa_full_addr + (stage_4) * 8) & 0xFEFFFFFF), "r"((uint32_t)(4096)) : "memory");
                    }
                    stage_4 += 1;
                    if (stage_4 == 4) { stage_4 = 0; _phase_sfa_free ^= 1; }
                }
                mbarrier_wait(work_full_addr + (work_stage_5) * 8, _phase_work_full_5);
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
                    : "r"(work_response_addr + work_stage_5 * 16 + 0 * 16)
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
                if (((_clc_ctaid_5 < (unsigned int)num_non_exiting_ctas[0]) ? _clc_valid_2 : (unsigned int)0) == 0) {
                    break;
                }
                m_tile_5 = _clc_ctaid_4 + (unsigned int)cta_rank;
                n_tile_5 = _clc_ctaid_5;
            }
        }
    }
    // ---- Role: copy_sfa ----
    if (warp == 22) {
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
                    if (m_tile_6 >= (unsigned int)grid_m || n_tile_6 >= (unsigned int)grid_n) {
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
                                uint64_t _tcgen05_cp_desc_0 = ((((uint64_t)(smem_sfa_addr + stage_5 * 4096)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                asm volatile(
                                    "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                    :: "r"((uint32_t)((unsigned int)tmem_sfa + stage_5 * 32)), "l"(_tcgen05_cp_desc_0)
                                    : "memory");
                            }
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                            #endif
                            {
                                uint64_t _tcgen05_cp_desc_1 = ((((uint64_t)(smem_sfa_addr + stage_5 * 4096 + 512)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                asm volatile(
                                    "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                    :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_5 * 32 + 4))), "l"(_tcgen05_cp_desc_1)
                                    : "memory");
                            }
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                            #endif
                            {
                                uint64_t _tcgen05_cp_desc_2 = ((((uint64_t)(smem_sfa_addr + stage_5 * 4096 + 1024)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                asm volatile(
                                    "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                    :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_5 * 32 + 8))), "l"(_tcgen05_cp_desc_2)
                                    : "memory");
                            }
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                            #endif
                            {
                                uint64_t _tcgen05_cp_desc_3 = ((((uint64_t)(smem_sfa_addr + stage_5 * 4096 + 1536)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                asm volatile(
                                    "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                    :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_5 * 32 + 12))), "l"(_tcgen05_cp_desc_3)
                                    : "memory");
                            }
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                            #endif
                            {
                                uint64_t _tcgen05_cp_desc_4 = ((((uint64_t)(smem_sfa_addr + stage_5 * 4096 + 2048)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                asm volatile(
                                    "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                    :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_5 * 32 + 16))), "l"(_tcgen05_cp_desc_4)
                                    : "memory");
                            }
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                            #endif
                            {
                                uint64_t _tcgen05_cp_desc_5 = ((((uint64_t)(smem_sfa_addr + stage_5 * 4096 + 2560)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                asm volatile(
                                    "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                    :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_5 * 32 + 20))), "l"(_tcgen05_cp_desc_5)
                                    : "memory");
                            }
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                            #endif
                            {
                                uint64_t _tcgen05_cp_desc_6 = ((((uint64_t)(smem_sfa_addr + stage_5 * 4096 + 3072)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                asm volatile(
                                    "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                    :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_5 * 32 + 24))), "l"(_tcgen05_cp_desc_6)
                                    : "memory");
                            }
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                            #endif
                            {
                                uint64_t _tcgen05_cp_desc_7 = ((((uint64_t)(smem_sfa_addr + stage_5 * 4096 + 3584)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                asm volatile(
                                    "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                    :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_5 * 32 + 28))), "l"(_tcgen05_cp_desc_7)
                                    : "memory");
                            }
                        }
                        mbarrier_arrive(tmem_sfa_full_addr + (stage_5) * 8);
                        elect_commit_cg2_multicast(sfa_free_addr + (stage_5) * 8, (uint16_t)(3));
                        stage_5 += 1;
                        if (stage_5 == 4) { stage_5 = 0; _phase_sfa_full ^= 1; _phase_k_done_3 ^= 1; }
                    }
                    mbarrier_wait(work_full_addr + (work_stage_6) * 8, _phase_work_full_6);
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
                        : "r"(work_response_addr + work_stage_6 * 16 + 0 * 16)
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
                        : "r"(work_response_addr + work_stage_6 * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(work_empty_addr + work_stage_6 * 8), "r"(0) : "memory");
                    work_stage_6 += 1;
                    if (work_stage_6 == 3) { work_stage_6 = 0; _phase_work_full_6 ^= 1; }
                    if (((_clc_ctaid_9 < (unsigned int)num_non_exiting_ctas[0]) ? _clc_valid_4 : (unsigned int)0) == 0) {
                        break;
                    }
                    m_tile_6 = _clc_ctaid_8 + (unsigned int)cta_rank;
                    n_tile_6 = _clc_ctaid_9;
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 23) {
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
                    if (m_tile_7 >= (unsigned int)grid_m || n_tile_7 >= (unsigned int)grid_n) {
                        break;
                    }
                    mbarrier_wait(mma_free_addr + (acc_stage_1) * 8, _phase_mma_free);
                    #pragma unroll 1
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

                                tcgen05_mma_mxf4nvf4_bs_cta2((tmem_accum + (acc_stage_1 * 64)), a_desc + 0, b_desc + 0,
                                    0x10100480U, (unsigned int)tmem_sfa + stage_6 * 32 + 0, (unsigned int)tmem_sfb + stage_6 * 16 + 0, ((((1) ? ((iter_k_4 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                            }
                        }
                        int _mma_a_lo_1 = (((smem_a_addr + 32) >> 4) & 0x3FFF) + (stage_6) * 2048;
                        int _mma_b_lo_1 = (((smem_b_addr + 32) >> 4) & 0x3FFF) + (stage_6) * 512;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_cta2((tmem_accum + (acc_stage_1 * 64)), a_desc + 0, b_desc + 0,
                                    0x10100480U, (unsigned int)tmem_sfa + (stage_6 * 32 + 4) + 0, (unsigned int)tmem_sfb + (stage_6 * 16 + 2) + 0, ((((0) ? ((iter_k_4 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                            }
                        }
                        int _mma_a_lo_2 = (((smem_a_addr + 64) >> 4) & 0x3FFF) + (stage_6) * 2048;
                        int _mma_b_lo_2 = (((smem_b_addr + 64) >> 4) & 0x3FFF) + (stage_6) * 512;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_2) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_2) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_cta2((tmem_accum + (acc_stage_1 * 64)), a_desc + 0, b_desc + 0,
                                    0x10100480U, (unsigned int)tmem_sfa + (stage_6 * 32 + 8) + 0, (unsigned int)tmem_sfb + (stage_6 * 16 + 4) + 0, ((((0) ? ((iter_k_4 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                            }
                        }
                        int _mma_a_lo_3 = (((smem_a_addr + 96) >> 4) & 0x3FFF) + (stage_6) * 2048;
                        int _mma_b_lo_3 = (((smem_b_addr + 96) >> 4) & 0x3FFF) + (stage_6) * 512;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_3) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_3) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_cta2((tmem_accum + (acc_stage_1 * 64)), a_desc + 0, b_desc + 0,
                                    0x10100480U, (unsigned int)tmem_sfa + (stage_6 * 32 + 12) + 0, (unsigned int)tmem_sfb + (stage_6 * 16 + 6) + 0, ((((0) ? ((iter_k_4 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                            }
                        }
                        int _mma_a_lo_4 = (((smem_a_addr + 16384) >> 4) & 0x3FFF) + (stage_6) * 2048;
                        int _mma_b_lo_4 = (((smem_b_addr + 4096) >> 4) & 0x3FFF) + (stage_6) * 512;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_4) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_4) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_cta2((tmem_accum + (acc_stage_1 * 64)), a_desc + 0, b_desc + 0,
                                    0x10100480U, (unsigned int)tmem_sfa + (stage_6 * 32 + 16) + 0, (unsigned int)tmem_sfb + (stage_6 * 16 + 8) + 0, ((((0) ? ((iter_k_4 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                            }
                        }
                        int _mma_a_lo_5 = (((smem_a_addr + 16416) >> 4) & 0x3FFF) + (stage_6) * 2048;
                        int _mma_b_lo_5 = (((smem_b_addr + 4128) >> 4) & 0x3FFF) + (stage_6) * 512;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_5) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_5) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_cta2((tmem_accum + (acc_stage_1 * 64)), a_desc + 0, b_desc + 0,
                                    0x10100480U, (unsigned int)tmem_sfa + (stage_6 * 32 + 20) + 0, (unsigned int)tmem_sfb + (stage_6 * 16 + 10) + 0, ((((0) ? ((iter_k_4 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                            }
                        }
                        int _mma_a_lo_6 = (((smem_a_addr + 16448) >> 4) & 0x3FFF) + (stage_6) * 2048;
                        int _mma_b_lo_6 = (((smem_b_addr + 4160) >> 4) & 0x3FFF) + (stage_6) * 512;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_6) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_6) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_cta2((tmem_accum + (acc_stage_1 * 64)), a_desc + 0, b_desc + 0,
                                    0x10100480U, (unsigned int)tmem_sfa + (stage_6 * 32 + 24) + 0, (unsigned int)tmem_sfb + (stage_6 * 16 + 12) + 0, ((((0) ? ((iter_k_4 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                            }
                        }
                        int _mma_a_lo_7 = (((smem_a_addr + 16480) >> 4) & 0x3FFF) + (stage_6) * 2048;
                        int _mma_b_lo_7 = (((smem_b_addr + 4192) >> 4) & 0x3FFF) + (stage_6) * 512;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_7) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_7) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_cta2((tmem_accum + (acc_stage_1 * 64)), a_desc + 0, b_desc + 0,
                                    0x10100480U, (unsigned int)tmem_sfa + (stage_6 * 32 + 28) + 0, (unsigned int)tmem_sfb + (stage_6 * 16 + 14) + 0, ((((0) ? ((iter_k_4 == 0) ? 1 : 0) : 0)) ? 0 : 1));
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
                    uint32_t _clc_ctaid_12 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_12)
                        : "r"(work_response_addr + work_stage_7 * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_13 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_13)
                        : "r"(work_response_addr + work_stage_7 * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(work_empty_addr + work_stage_7 * 8), "r"(0) : "memory");
                    work_stage_7 += 1;
                    if (work_stage_7 == 3) { work_stage_7 = 0; _phase_work_full_7 ^= 1; }
                    if (((_clc_ctaid_13 < (unsigned int)num_non_exiting_ctas[0]) ? _clc_valid_6 : (unsigned int)0) == 0) {
                        break;
                    }
                    m_tile_7 = _clc_ctaid_12 + (unsigned int)cta_rank;
                    n_tile_7 = _clc_ctaid_13;
                }
            }
        }
    }
    // ---- Role: work_id ----
    if (warp == 24) {
        { // work_id_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int work_stage_8 = 0;
            unsigned int throttle_stage_1 = 0;
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
                    uint32_t _clc_ctaid_16 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_16)
                        : "r"(work_response_addr + work_stage_8 * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_17 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_17)
                        : "r"(work_response_addr + work_stage_8 * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(work_empty_addr + work_stage_8 * 8), "r"(0) : "memory");
                    work_stage_8 += 1;
                    if (work_stage_8 == 3) { work_stage_8 = 0; _phase_work_empty ^= 1; _phase_work_full_8 ^= 1; }
                    if (((_clc_ctaid_17 < (unsigned int)num_non_exiting_ctas[0]) ? _clc_valid_8 : (unsigned int)0) == 0) {
                        if (_clc_ctaid_17 >= (unsigned int)num_non_exiting_ctas[0]) {
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
    if (warp >= 25 && warp <= 27) {
        { // padding_main
            unsigned int work_stage_9 = 0;
            unsigned int m_tile_8 = blockIdx.x;
            unsigned int n_tile_8 = blockIdx.y;
            unsigned int _phase_work_full_9 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_9 = 0; _tile_iter_9 < grid_m / 2 * grid_n; _tile_iter_9++) {
                if (m_tile_8 >= (unsigned int)grid_m || n_tile_8 >= (unsigned int)grid_n) {
                    break;
                }
                mbarrier_wait(work_full_addr + (work_stage_9) * 8, _phase_work_full_9);
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
                uint32_t _clc_ctaid_18 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_18)
                    : "r"(work_response_addr + work_stage_9 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_19 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_19)
                    : "r"(work_response_addr + work_stage_9 * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + work_stage_9 * 8), "r"(0) : "memory");
                work_stage_9 += 1;
                if (work_stage_9 == 3) { work_stage_9 = 0; _phase_work_full_9 ^= 1; }
                if (((_clc_ctaid_19 < (unsigned int)num_non_exiting_ctas[0]) ? _clc_valid_13 : (unsigned int)0) == 0) {
                    break;
                }
                m_tile_8 = _clc_ctaid_18 + (unsigned int)cta_rank;
                n_tile_8 = _clc_ctaid_19;
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
