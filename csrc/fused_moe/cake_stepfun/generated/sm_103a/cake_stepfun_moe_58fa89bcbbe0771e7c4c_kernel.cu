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




extern "C" {

__global__ __launch_bounds__(896, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_cake_stepfun_moe_58fa89bcbbe0771e7c4c(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, __nv_bfloat16* __restrict__ C, float* __restrict__ SFC, int* __restrict__ route_map, int* __restrict__ tile_expert, int* __restrict__ tile_mn_limit, int* __restrict__ num_non_exiting_ctas, float* __restrict__ scale_c, float* __restrict__ scale_gate, float* __restrict__ clamp_limit, float* __restrict__ act_alpha, float* __restrict__ act_beta, int M_out, int K, int grid_m, int grid_n, int K_tiles)
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
            float out_pair[2] = {0};
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
                int tok_slot_base = (int)n_tile * 64;
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
                int tok_feature = (int)m_tile * 64 + base_row;
                asm volatile("cp.async.bulk.wait_group.read 0;");
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                int token0 = lane_1 % 4 * 2;
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
                int token0_0 = lane_1 % 4 * 2 + 8;
                int token1_1 = token0_0 + 1;
                float tok_sf0_2 = 0.0f;
                float tok_sf1_3 = 0.0f;
                if (token0_0 < valid_rows) {
                    int tok_route0_1 = route_map[tok_slot_base + token0_0];
                    tok_sf0_2 = SFC[tok_route0_1];
                }
                if (token1_1 < valid_rows) {
                    int tok_route1_1 = route_map[tok_slot_base + token1_1];
                    tok_sf1_3 = SFC[tok_route1_1];
                }
                float _max_4 = max_noftz(_tmem_load_0[4] * tok_sf0_2, neg_cl);
                float _min_8 = fminf(_max_4, cl);
                float tok_lin00_4 = _min_8;
                float _max_5 = max_noftz(_tmem_load_0[5] * tok_sf1_3, neg_cl);
                float _min_9 = fminf(_max_5, cl);
                float tok_lin01_5 = _min_9;
                float _max_6 = max_noftz(_tmem_load_1[4] * tok_sf0_2, neg_cl);
                float _min_10 = fminf(_max_6, cl);
                float tok_lin10_6 = _min_10;
                float _max_7 = max_noftz(_tmem_load_1[5] * tok_sf1_3, neg_cl);
                float _min_11 = fminf(_max_7, cl);
                float tok_lin11_7 = _min_11;
                float tok_x00_8 = _tmem_load_0[6] * tok_sf0_2;
                float tok_x01_9 = _tmem_load_0[7] * tok_sf1_3;
                float tok_x10_10 = _tmem_load_1[6] * tok_sf0_2;
                float tok_x11_11 = _tmem_load_1[7] * tok_sf1_3;
                float _exp2_4 = approx_exp2((-(tok_x00_8 * sg)) * 1.4426950408889634f);
                float _rcp_4 = approx_rcp(1.0f + _exp2_4);
                float tok_sig00_12 = _rcp_4;
                float _exp2_5 = approx_exp2((-(tok_x01_9 * sg)) * 1.4426950408889634f);
                float _rcp_5 = approx_rcp(1.0f + _exp2_5);
                float tok_sig01_13 = _rcp_5;
                float _exp2_6 = approx_exp2((-(tok_x10_10 * sg)) * 1.4426950408889634f);
                float _rcp_6 = approx_rcp(1.0f + _exp2_6);
                float tok_sig10_14 = _rcp_6;
                float _exp2_7 = approx_exp2((-(tok_x11_11 * sg)) * 1.4426950408889634f);
                float _rcp_7 = approx_rcp(1.0f + _exp2_7);
                float tok_sig11_15 = _rcp_7;
                float _min_12 = fminf(tok_x00_8 * tok_sig00_12, cl);
                float tok_g00_16 = _min_12;
                float _min_13 = fminf(tok_x01_9 * tok_sig01_13, cl);
                float tok_g01_17 = _min_13;
                float _min_14 = fminf(tok_x10_10 * tok_sig10_14, cl);
                float tok_g10_18 = _min_14;
                float _min_15 = fminf(tok_x11_11 * tok_sig11_15, cl);
                float tok_g11_19 = _min_15;
                float value00_20 = tok_lin00_4 * sc * sg * tok_g00_16;
                float value01_21 = tok_lin01_5 * sc * sg * tok_g01_17;
                float value10_22 = tok_lin10_6 * sc * sg * tok_g10_18;
                float value11_23 = tok_lin11_7 * sc * sg * tok_g11_19;
                out_pair[0] = value00_20;
                out_pair[1] = value10_22;
                if (token0_0 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token0_0) * M_out + tok_feature)))[0]) = _pk;
                    }
                }
                out_pair[0] = value01_21;
                out_pair[1] = value11_23;
                if (token1_1 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token1_1) * M_out + tok_feature)))[0]) = _pk;
                    }
                }
                int token0_24 = lane_1 % 4 * 2 + 16;
                int token1_25 = token0_24 + 1;
                float tok_sf0_26 = 0.0f;
                float tok_sf1_27 = 0.0f;
                if (token0_24 < valid_rows) {
                    int tok_route0_2 = route_map[tok_slot_base + token0_24];
                    tok_sf0_26 = SFC[tok_route0_2];
                }
                if (token1_25 < valid_rows) {
                    int tok_route1_2 = route_map[tok_slot_base + token1_25];
                    tok_sf1_27 = SFC[tok_route1_2];
                }
                float _max_8 = max_noftz(_tmem_load_0[8] * tok_sf0_26, neg_cl);
                float _min_16 = fminf(_max_8, cl);
                float tok_lin00_28 = _min_16;
                float _max_9 = max_noftz(_tmem_load_0[9] * tok_sf1_27, neg_cl);
                float _min_17 = fminf(_max_9, cl);
                float tok_lin01_29 = _min_17;
                float _max_10 = max_noftz(_tmem_load_1[8] * tok_sf0_26, neg_cl);
                float _min_18 = fminf(_max_10, cl);
                float tok_lin10_30 = _min_18;
                float _max_11 = max_noftz(_tmem_load_1[9] * tok_sf1_27, neg_cl);
                float _min_19 = fminf(_max_11, cl);
                float tok_lin11_31 = _min_19;
                float tok_x00_32 = _tmem_load_0[10] * tok_sf0_26;
                float tok_x01_33 = _tmem_load_0[11] * tok_sf1_27;
                float tok_x10_34 = _tmem_load_1[10] * tok_sf0_26;
                float tok_x11_35 = _tmem_load_1[11] * tok_sf1_27;
                float _exp2_8 = approx_exp2((-(tok_x00_32 * sg)) * 1.4426950408889634f);
                float _rcp_8 = approx_rcp(1.0f + _exp2_8);
                float tok_sig00_36 = _rcp_8;
                float _exp2_9 = approx_exp2((-(tok_x01_33 * sg)) * 1.4426950408889634f);
                float _rcp_9 = approx_rcp(1.0f + _exp2_9);
                float tok_sig01_37 = _rcp_9;
                float _exp2_10 = approx_exp2((-(tok_x10_34 * sg)) * 1.4426950408889634f);
                float _rcp_10 = approx_rcp(1.0f + _exp2_10);
                float tok_sig10_38 = _rcp_10;
                float _exp2_11 = approx_exp2((-(tok_x11_35 * sg)) * 1.4426950408889634f);
                float _rcp_11 = approx_rcp(1.0f + _exp2_11);
                float tok_sig11_39 = _rcp_11;
                float _min_20 = fminf(tok_x00_32 * tok_sig00_36, cl);
                float tok_g00_40 = _min_20;
                float _min_21 = fminf(tok_x01_33 * tok_sig01_37, cl);
                float tok_g01_41 = _min_21;
                float _min_22 = fminf(tok_x10_34 * tok_sig10_38, cl);
                float tok_g10_42 = _min_22;
                float _min_23 = fminf(tok_x11_35 * tok_sig11_39, cl);
                float tok_g11_43 = _min_23;
                float value00_44 = tok_lin00_28 * sc * sg * tok_g00_40;
                float value01_45 = tok_lin01_29 * sc * sg * tok_g01_41;
                float value10_46 = tok_lin10_30 * sc * sg * tok_g10_42;
                float value11_47 = tok_lin11_31 * sc * sg * tok_g11_43;
                out_pair[0] = value00_44;
                out_pair[1] = value10_46;
                if (token0_24 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token0_24) * M_out + tok_feature)))[0]) = _pk;
                    }
                }
                out_pair[0] = value01_45;
                out_pair[1] = value11_47;
                if (token1_25 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token1_25) * M_out + tok_feature)))[0]) = _pk;
                    }
                }
                int token0_48 = lane_1 % 4 * 2 + 24;
                int token1_49 = token0_48 + 1;
                float tok_sf0_50 = 0.0f;
                float tok_sf1_51 = 0.0f;
                if (token0_48 < valid_rows) {
                    int tok_route0_3 = route_map[tok_slot_base + token0_48];
                    tok_sf0_50 = SFC[tok_route0_3];
                }
                if (token1_49 < valid_rows) {
                    int tok_route1_3 = route_map[tok_slot_base + token1_49];
                    tok_sf1_51 = SFC[tok_route1_3];
                }
                float _max_12 = max_noftz(_tmem_load_0[12] * tok_sf0_50, neg_cl);
                float _min_24 = fminf(_max_12, cl);
                float tok_lin00_52 = _min_24;
                float _max_13 = max_noftz(_tmem_load_0[13] * tok_sf1_51, neg_cl);
                float _min_25 = fminf(_max_13, cl);
                float tok_lin01_53 = _min_25;
                float _max_14 = max_noftz(_tmem_load_1[12] * tok_sf0_50, neg_cl);
                float _min_26 = fminf(_max_14, cl);
                float tok_lin10_54 = _min_26;
                float _max_15 = max_noftz(_tmem_load_1[13] * tok_sf1_51, neg_cl);
                float _min_27 = fminf(_max_15, cl);
                float tok_lin11_55 = _min_27;
                float tok_x00_56 = _tmem_load_0[14] * tok_sf0_50;
                float tok_x01_57 = _tmem_load_0[15] * tok_sf1_51;
                float tok_x10_58 = _tmem_load_1[14] * tok_sf0_50;
                float tok_x11_59 = _tmem_load_1[15] * tok_sf1_51;
                float _exp2_12 = approx_exp2((-(tok_x00_56 * sg)) * 1.4426950408889634f);
                float _rcp_12 = approx_rcp(1.0f + _exp2_12);
                float tok_sig00_60 = _rcp_12;
                float _exp2_13 = approx_exp2((-(tok_x01_57 * sg)) * 1.4426950408889634f);
                float _rcp_13 = approx_rcp(1.0f + _exp2_13);
                float tok_sig01_61 = _rcp_13;
                float _exp2_14 = approx_exp2((-(tok_x10_58 * sg)) * 1.4426950408889634f);
                float _rcp_14 = approx_rcp(1.0f + _exp2_14);
                float tok_sig10_62 = _rcp_14;
                float _exp2_15 = approx_exp2((-(tok_x11_59 * sg)) * 1.4426950408889634f);
                float _rcp_15 = approx_rcp(1.0f + _exp2_15);
                float tok_sig11_63 = _rcp_15;
                float _min_28 = fminf(tok_x00_56 * tok_sig00_60, cl);
                float tok_g00_64 = _min_28;
                float _min_29 = fminf(tok_x01_57 * tok_sig01_61, cl);
                float tok_g01_65 = _min_29;
                float _min_30 = fminf(tok_x10_58 * tok_sig10_62, cl);
                float tok_g10_66 = _min_30;
                float _min_31 = fminf(tok_x11_59 * tok_sig11_63, cl);
                float tok_g11_67 = _min_31;
                float value00_68 = tok_lin00_52 * sc * sg * tok_g00_64;
                float value01_69 = tok_lin01_53 * sc * sg * tok_g01_65;
                float value10_70 = tok_lin10_54 * sc * sg * tok_g10_66;
                float value11_71 = tok_lin11_55 * sc * sg * tok_g11_67;
                out_pair[0] = value00_68;
                out_pair[1] = value10_70;
                if (token0_48 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token0_48) * M_out + tok_feature)))[0]) = _pk;
                    }
                }
                out_pair[0] = value01_69;
                out_pair[1] = value11_71;
                if (token1_49 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token1_49) * M_out + tok_feature)))[0]) = _pk;
                    }
                }
                int acc_offset_72 = acc_stage * 64 + 32;
                float _tmem_load_2[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[15]))
                    : "r"(taddr + (unsigned int)acc_offset_72));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float _tmem_load_3[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[15]))
                    : "r"(taddr + 1048576 + (unsigned int)acc_offset_72));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int base_row_73 = warp_0 * 16 + lane_1 / 4 * 2;
                int tok_feature_74 = (int)m_tile * 64 + base_row_73;
                asm volatile("cp.async.bulk.wait_group.read 0;");
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                int token0_75 = lane_1 % 4 * 2 + 32;
                int token1_76 = token0_75 + 1;
                float tok_sf0_77 = 0.0f;
                float tok_sf1_78 = 0.0f;
                if (token0_75 < valid_rows) {
                    int tok_route0_4 = route_map[tok_slot_base + token0_75];
                    tok_sf0_77 = SFC[tok_route0_4];
                }
                if (token1_76 < valid_rows) {
                    int tok_route1_4 = route_map[tok_slot_base + token1_76];
                    tok_sf1_78 = SFC[tok_route1_4];
                }
                float _max_16 = max_noftz(_tmem_load_2[0] * tok_sf0_77, neg_cl);
                float _min_32 = fminf(_max_16, cl);
                float tok_lin00_79 = _min_32;
                float _max_17 = max_noftz(_tmem_load_2[1] * tok_sf1_78, neg_cl);
                float _min_33 = fminf(_max_17, cl);
                float tok_lin01_80 = _min_33;
                float _max_18 = max_noftz(_tmem_load_3[0] * tok_sf0_77, neg_cl);
                float _min_34 = fminf(_max_18, cl);
                float tok_lin10_81 = _min_34;
                float _max_19 = max_noftz(_tmem_load_3[1] * tok_sf1_78, neg_cl);
                float _min_35 = fminf(_max_19, cl);
                float tok_lin11_82 = _min_35;
                float tok_x00_83 = _tmem_load_2[2] * tok_sf0_77;
                float tok_x01_84 = _tmem_load_2[3] * tok_sf1_78;
                float tok_x10_85 = _tmem_load_3[2] * tok_sf0_77;
                float tok_x11_86 = _tmem_load_3[3] * tok_sf1_78;
                float _exp2_16 = approx_exp2((-(tok_x00_83 * sg)) * 1.4426950408889634f);
                float _rcp_16 = approx_rcp(1.0f + _exp2_16);
                float tok_sig00_87 = _rcp_16;
                float _exp2_17 = approx_exp2((-(tok_x01_84 * sg)) * 1.4426950408889634f);
                float _rcp_17 = approx_rcp(1.0f + _exp2_17);
                float tok_sig01_88 = _rcp_17;
                float _exp2_18 = approx_exp2((-(tok_x10_85 * sg)) * 1.4426950408889634f);
                float _rcp_18 = approx_rcp(1.0f + _exp2_18);
                float tok_sig10_89 = _rcp_18;
                float _exp2_19 = approx_exp2((-(tok_x11_86 * sg)) * 1.4426950408889634f);
                float _rcp_19 = approx_rcp(1.0f + _exp2_19);
                float tok_sig11_90 = _rcp_19;
                float _min_36 = fminf(tok_x00_83 * tok_sig00_87, cl);
                float tok_g00_91 = _min_36;
                float _min_37 = fminf(tok_x01_84 * tok_sig01_88, cl);
                float tok_g01_92 = _min_37;
                float _min_38 = fminf(tok_x10_85 * tok_sig10_89, cl);
                float tok_g10_93 = _min_38;
                float _min_39 = fminf(tok_x11_86 * tok_sig11_90, cl);
                float tok_g11_94 = _min_39;
                float value00_95 = tok_lin00_79 * sc * sg * tok_g00_91;
                float value01_96 = tok_lin01_80 * sc * sg * tok_g01_92;
                float value10_97 = tok_lin10_81 * sc * sg * tok_g10_93;
                float value11_98 = tok_lin11_82 * sc * sg * tok_g11_94;
                out_pair[0] = value00_95;
                out_pair[1] = value10_97;
                if (token0_75 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token0_75) * M_out + tok_feature_74)))[0]) = _pk;
                    }
                }
                out_pair[0] = value01_96;
                out_pair[1] = value11_98;
                if (token1_76 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token1_76) * M_out + tok_feature_74)))[0]) = _pk;
                    }
                }
                int token0_99 = lane_1 % 4 * 2 + 8 + 32;
                int token1_100 = token0_99 + 1;
                float tok_sf0_101 = 0.0f;
                float tok_sf1_102 = 0.0f;
                if (token0_99 < valid_rows) {
                    int tok_route0_5 = route_map[tok_slot_base + token0_99];
                    tok_sf0_101 = SFC[tok_route0_5];
                }
                if (token1_100 < valid_rows) {
                    int tok_route1_5 = route_map[tok_slot_base + token1_100];
                    tok_sf1_102 = SFC[tok_route1_5];
                }
                float _max_20 = max_noftz(_tmem_load_2[4] * tok_sf0_101, neg_cl);
                float _min_40 = fminf(_max_20, cl);
                float tok_lin00_103 = _min_40;
                float _max_21 = max_noftz(_tmem_load_2[5] * tok_sf1_102, neg_cl);
                float _min_41 = fminf(_max_21, cl);
                float tok_lin01_104 = _min_41;
                float _max_22 = max_noftz(_tmem_load_3[4] * tok_sf0_101, neg_cl);
                float _min_42 = fminf(_max_22, cl);
                float tok_lin10_105 = _min_42;
                float _max_23 = max_noftz(_tmem_load_3[5] * tok_sf1_102, neg_cl);
                float _min_43 = fminf(_max_23, cl);
                float tok_lin11_106 = _min_43;
                float tok_x00_107 = _tmem_load_2[6] * tok_sf0_101;
                float tok_x01_108 = _tmem_load_2[7] * tok_sf1_102;
                float tok_x10_109 = _tmem_load_3[6] * tok_sf0_101;
                float tok_x11_110 = _tmem_load_3[7] * tok_sf1_102;
                float _exp2_20 = approx_exp2((-(tok_x00_107 * sg)) * 1.4426950408889634f);
                float _rcp_20 = approx_rcp(1.0f + _exp2_20);
                float tok_sig00_111 = _rcp_20;
                float _exp2_21 = approx_exp2((-(tok_x01_108 * sg)) * 1.4426950408889634f);
                float _rcp_21 = approx_rcp(1.0f + _exp2_21);
                float tok_sig01_112 = _rcp_21;
                float _exp2_22 = approx_exp2((-(tok_x10_109 * sg)) * 1.4426950408889634f);
                float _rcp_22 = approx_rcp(1.0f + _exp2_22);
                float tok_sig10_113 = _rcp_22;
                float _exp2_23 = approx_exp2((-(tok_x11_110 * sg)) * 1.4426950408889634f);
                float _rcp_23 = approx_rcp(1.0f + _exp2_23);
                float tok_sig11_114 = _rcp_23;
                float _min_44 = fminf(tok_x00_107 * tok_sig00_111, cl);
                float tok_g00_115 = _min_44;
                float _min_45 = fminf(tok_x01_108 * tok_sig01_112, cl);
                float tok_g01_116 = _min_45;
                float _min_46 = fminf(tok_x10_109 * tok_sig10_113, cl);
                float tok_g10_117 = _min_46;
                float _min_47 = fminf(tok_x11_110 * tok_sig11_114, cl);
                float tok_g11_118 = _min_47;
                float value00_119 = tok_lin00_103 * sc * sg * tok_g00_115;
                float value01_120 = tok_lin01_104 * sc * sg * tok_g01_116;
                float value10_121 = tok_lin10_105 * sc * sg * tok_g10_117;
                float value11_122 = tok_lin11_106 * sc * sg * tok_g11_118;
                out_pair[0] = value00_119;
                out_pair[1] = value10_121;
                if (token0_99 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token0_99) * M_out + tok_feature_74)))[0]) = _pk;
                    }
                }
                out_pair[0] = value01_120;
                out_pair[1] = value11_122;
                if (token1_100 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token1_100) * M_out + tok_feature_74)))[0]) = _pk;
                    }
                }
                int token0_123 = lane_1 % 4 * 2 + 16 + 32;
                int token1_124 = token0_123 + 1;
                float tok_sf0_125 = 0.0f;
                float tok_sf1_126 = 0.0f;
                if (token0_123 < valid_rows) {
                    int tok_route0_6 = route_map[tok_slot_base + token0_123];
                    tok_sf0_125 = SFC[tok_route0_6];
                }
                if (token1_124 < valid_rows) {
                    int tok_route1_6 = route_map[tok_slot_base + token1_124];
                    tok_sf1_126 = SFC[tok_route1_6];
                }
                float _max_24 = max_noftz(_tmem_load_2[8] * tok_sf0_125, neg_cl);
                float _min_48 = fminf(_max_24, cl);
                float tok_lin00_127 = _min_48;
                float _max_25 = max_noftz(_tmem_load_2[9] * tok_sf1_126, neg_cl);
                float _min_49 = fminf(_max_25, cl);
                float tok_lin01_128 = _min_49;
                float _max_26 = max_noftz(_tmem_load_3[8] * tok_sf0_125, neg_cl);
                float _min_50 = fminf(_max_26, cl);
                float tok_lin10_129 = _min_50;
                float _max_27 = max_noftz(_tmem_load_3[9] * tok_sf1_126, neg_cl);
                float _min_51 = fminf(_max_27, cl);
                float tok_lin11_130 = _min_51;
                float tok_x00_131 = _tmem_load_2[10] * tok_sf0_125;
                float tok_x01_132 = _tmem_load_2[11] * tok_sf1_126;
                float tok_x10_133 = _tmem_load_3[10] * tok_sf0_125;
                float tok_x11_134 = _tmem_load_3[11] * tok_sf1_126;
                float _exp2_24 = approx_exp2((-(tok_x00_131 * sg)) * 1.4426950408889634f);
                float _rcp_24 = approx_rcp(1.0f + _exp2_24);
                float tok_sig00_135 = _rcp_24;
                float _exp2_25 = approx_exp2((-(tok_x01_132 * sg)) * 1.4426950408889634f);
                float _rcp_25 = approx_rcp(1.0f + _exp2_25);
                float tok_sig01_136 = _rcp_25;
                float _exp2_26 = approx_exp2((-(tok_x10_133 * sg)) * 1.4426950408889634f);
                float _rcp_26 = approx_rcp(1.0f + _exp2_26);
                float tok_sig10_137 = _rcp_26;
                float _exp2_27 = approx_exp2((-(tok_x11_134 * sg)) * 1.4426950408889634f);
                float _rcp_27 = approx_rcp(1.0f + _exp2_27);
                float tok_sig11_138 = _rcp_27;
                float _min_52 = fminf(tok_x00_131 * tok_sig00_135, cl);
                float tok_g00_139 = _min_52;
                float _min_53 = fminf(tok_x01_132 * tok_sig01_136, cl);
                float tok_g01_140 = _min_53;
                float _min_54 = fminf(tok_x10_133 * tok_sig10_137, cl);
                float tok_g10_141 = _min_54;
                float _min_55 = fminf(tok_x11_134 * tok_sig11_138, cl);
                float tok_g11_142 = _min_55;
                float value00_143 = tok_lin00_127 * sc * sg * tok_g00_139;
                float value01_144 = tok_lin01_128 * sc * sg * tok_g01_140;
                float value10_145 = tok_lin10_129 * sc * sg * tok_g10_141;
                float value11_146 = tok_lin11_130 * sc * sg * tok_g11_142;
                out_pair[0] = value00_143;
                out_pair[1] = value10_145;
                if (token0_123 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token0_123) * M_out + tok_feature_74)))[0]) = _pk;
                    }
                }
                out_pair[0] = value01_144;
                out_pair[1] = value11_146;
                if (token1_124 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token1_124) * M_out + tok_feature_74)))[0]) = _pk;
                    }
                }
                int token0_147 = lane_1 % 4 * 2 + 24 + 32;
                int token1_148 = token0_147 + 1;
                float tok_sf0_149 = 0.0f;
                float tok_sf1_150 = 0.0f;
                if (token0_147 < valid_rows) {
                    int tok_route0_7 = route_map[tok_slot_base + token0_147];
                    tok_sf0_149 = SFC[tok_route0_7];
                }
                if (token1_148 < valid_rows) {
                    int tok_route1_7 = route_map[tok_slot_base + token1_148];
                    tok_sf1_150 = SFC[tok_route1_7];
                }
                float _max_28 = max_noftz(_tmem_load_2[12] * tok_sf0_149, neg_cl);
                float _min_56 = fminf(_max_28, cl);
                float tok_lin00_151 = _min_56;
                float _max_29 = max_noftz(_tmem_load_2[13] * tok_sf1_150, neg_cl);
                float _min_57 = fminf(_max_29, cl);
                float tok_lin01_152 = _min_57;
                float _max_30 = max_noftz(_tmem_load_3[12] * tok_sf0_149, neg_cl);
                float _min_58 = fminf(_max_30, cl);
                float tok_lin10_153 = _min_58;
                float _max_31 = max_noftz(_tmem_load_3[13] * tok_sf1_150, neg_cl);
                float _min_59 = fminf(_max_31, cl);
                float tok_lin11_154 = _min_59;
                float tok_x00_155 = _tmem_load_2[14] * tok_sf0_149;
                float tok_x01_156 = _tmem_load_2[15] * tok_sf1_150;
                float tok_x10_157 = _tmem_load_3[14] * tok_sf0_149;
                float tok_x11_158 = _tmem_load_3[15] * tok_sf1_150;
                float _exp2_28 = approx_exp2((-(tok_x00_155 * sg)) * 1.4426950408889634f);
                float _rcp_28 = approx_rcp(1.0f + _exp2_28);
                float tok_sig00_159 = _rcp_28;
                float _exp2_29 = approx_exp2((-(tok_x01_156 * sg)) * 1.4426950408889634f);
                float _rcp_29 = approx_rcp(1.0f + _exp2_29);
                float tok_sig01_160 = _rcp_29;
                float _exp2_30 = approx_exp2((-(tok_x10_157 * sg)) * 1.4426950408889634f);
                float _rcp_30 = approx_rcp(1.0f + _exp2_30);
                float tok_sig10_161 = _rcp_30;
                float _exp2_31 = approx_exp2((-(tok_x11_158 * sg)) * 1.4426950408889634f);
                float _rcp_31 = approx_rcp(1.0f + _exp2_31);
                float tok_sig11_162 = _rcp_31;
                float _min_60 = fminf(tok_x00_155 * tok_sig00_159, cl);
                float tok_g00_163 = _min_60;
                float _min_61 = fminf(tok_x01_156 * tok_sig01_160, cl);
                float tok_g01_164 = _min_61;
                float _min_62 = fminf(tok_x10_157 * tok_sig10_161, cl);
                float tok_g10_165 = _min_62;
                float _min_63 = fminf(tok_x11_158 * tok_sig11_162, cl);
                float tok_g11_166 = _min_63;
                float value00_167 = tok_lin00_151 * sc * sg * tok_g00_163;
                float value01_168 = tok_lin01_152 * sc * sg * tok_g01_164;
                float value10_169 = tok_lin10_153 * sc * sg * tok_g10_165;
                float value11_170 = tok_lin11_154 * sc * sg * tok_g11_166;
                out_pair[0] = value00_167;
                out_pair[1] = value10_169;
                if (token0_147 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token0_147) * M_out + tok_feature_74)))[0]) = _pk;
                    }
                }
                out_pair[0] = value01_168;
                out_pair[1] = value11_170;
                if (token1_148 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token1_148) * M_out + tok_feature_74)))[0]) = _pk;
                    }
                }
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
                    : "+r"(_clc_ctaid_14)
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
                    : "+r"(_clc_ctaid_15)
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
                    : "+r"(_clc_ctaid_10)
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
                    : "+r"(_clc_ctaid_11)
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
                    : "+r"(_clc_ctaid_2)
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
                    : "+r"(_clc_ctaid_3)
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
                    : "+r"(_clc_ctaid_6)
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
                    : "+r"(_clc_ctaid_7)
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
                    : "+r"(_clc_ctaid_0)
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
                    : "+r"(_clc_ctaid_1)
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
                    : "+r"(_clc_ctaid_4)
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
                    : "+r"(_clc_ctaid_5)
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
                        : "+r"(_clc_ctaid_8)
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
                        : "+r"(_clc_ctaid_9)
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
                        : "+r"(_clc_ctaid_12)
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
                        : "+r"(_clc_ctaid_13)
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
                        : "+r"(_clc_ctaid_16)
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
                        : "+r"(_clc_ctaid_17)
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
                        if (_clc_valid_8 != 0) {
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
                    : "+r"(_clc_ctaid_18)
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
                    : "+r"(_clc_ctaid_19)
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
