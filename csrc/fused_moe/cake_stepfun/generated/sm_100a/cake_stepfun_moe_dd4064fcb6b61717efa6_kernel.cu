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
kernel_cake_stepfun_moe_dd4064fcb6b61717efa6(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, __nv_bfloat16* __restrict__ C, float* __restrict__ SFC, int* __restrict__ route_map, int* __restrict__ tile_expert, int* __restrict__ tile_mn_limit, int* __restrict__ num_non_exiting_ctas, float* __restrict__ scale_c, float* __restrict__ scale_gate, float* __restrict__ clamp_limit, float* __restrict__ act_alpha, float* __restrict__ act_beta, int M_out, int K, int grid_m, int grid_n, int K_tiles)
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
                float fused = 1.4426950216293335f * sg;
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
                float _exp2_0 = approx_exp2(-(tok_x00 * fused));
                float _rcp_0 = approx_rcp(1.0f + _exp2_0);
                float tok_sig00 = _rcp_0;
                float _exp2_1 = approx_exp2(-(tok_x01 * fused));
                float _rcp_1 = approx_rcp(1.0f + _exp2_1);
                float tok_sig01 = _rcp_1;
                float _exp2_2 = approx_exp2(-(tok_x10 * fused));
                float _rcp_2 = approx_rcp(1.0f + _exp2_2);
                float tok_sig10 = _rcp_2;
                float _exp2_3 = approx_exp2(-(tok_x11 * fused));
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
                float _fma_0 = __fmaf_rn(tok_lin00, sg, 0.0f);
                float tok_acc00 = _fma_0 * tok_g00;
                float _fma_1 = __fmaf_rn(tok_lin01, sg, 0.0f);
                float tok_acc01 = _fma_1 * tok_g01;
                float _fma_2 = __fmaf_rn(tok_lin10, sg, 0.0f);
                float tok_acc10 = _fma_2 * tok_g10;
                float _fma_3 = __fmaf_rn(tok_lin11, sg, 0.0f);
                float tok_acc11 = _fma_3 * tok_g11;
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
                float _exp2_4 = approx_exp2(-(tok_x00_8 * fused));
                float _rcp_4 = approx_rcp(1.0f + _exp2_4);
                float tok_sig00_12 = _rcp_4;
                float _exp2_5 = approx_exp2(-(tok_x01_9 * fused));
                float _rcp_5 = approx_rcp(1.0f + _exp2_5);
                float tok_sig01_13 = _rcp_5;
                float _exp2_6 = approx_exp2(-(tok_x10_10 * fused));
                float _rcp_6 = approx_rcp(1.0f + _exp2_6);
                float tok_sig10_14 = _rcp_6;
                float _exp2_7 = approx_exp2(-(tok_x11_11 * fused));
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
                float _fma_4 = __fmaf_rn(tok_lin00_4, sg, 0.0f);
                float tok_acc00_20 = _fma_4 * tok_g00_16;
                float _fma_5 = __fmaf_rn(tok_lin01_5, sg, 0.0f);
                float tok_acc01_21 = _fma_5 * tok_g01_17;
                float _fma_6 = __fmaf_rn(tok_lin10_6, sg, 0.0f);
                float tok_acc10_22 = _fma_6 * tok_g10_18;
                float _fma_7 = __fmaf_rn(tok_lin11_7, sg, 0.0f);
                float tok_acc11_23 = _fma_7 * tok_g11_19;
                float2 _f2_3 = make_float2(sc, sc);
                float2 tok_sc_24 = _f2_3;
                float2 _f2_4 = make_float2(tok_acc00_20, tok_acc10_22);
                float2 _mul_f32x2_2;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_2) : "l"(*(const unsigned long long*)&_f2_4), "l"(*(const unsigned long long*)&tok_sc_24));
                float2 tok_out0_25 = _mul_f32x2_2;
                float2 _f2_5 = make_float2(tok_acc01_21, tok_acc11_23);
                float2 _mul_f32x2_3;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_3) : "l"(*(const unsigned long long*)&_f2_5), "l"(*(const unsigned long long*)&tok_sc_24));
                float2 tok_out1_26 = _mul_f32x2_3;
                float value00_27 = tok_out0_25.x;
                float value10_28 = tok_out0_25.y;
                float value01_29 = tok_out1_26.x;
                float value11_30 = tok_out1_26.y;
                out_pair[0] = value00_27;
                out_pair[1] = value10_28;
                if (token0_0 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token0_0) * M_out + tok_feature)))[0]) = _pk;
                    }
                }
                out_pair[0] = value01_29;
                out_pair[1] = value11_30;
                if (token1_1 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token1_1) * M_out + tok_feature)))[0]) = _pk;
                    }
                }
                int token0_31 = lane_1 % 4 * 2 + 16;
                int token1_32 = token0_31 + 1;
                float tok_sf0_33 = 0.0f;
                float tok_sf1_34 = 0.0f;
                if (token0_31 < valid_rows) {
                    int tok_route0_2 = route_map[tok_slot_base + token0_31];
                    tok_sf0_33 = SFC[tok_route0_2];
                }
                if (token1_32 < valid_rows) {
                    int tok_route1_2 = route_map[tok_slot_base + token1_32];
                    tok_sf1_34 = SFC[tok_route1_2];
                }
                float _max_8 = max_noftz(_tmem_load_0[8] * tok_sf0_33, neg_cl);
                float _min_16 = fminf(_max_8, cl);
                float tok_lin00_35 = _min_16;
                float _max_9 = max_noftz(_tmem_load_0[9] * tok_sf1_34, neg_cl);
                float _min_17 = fminf(_max_9, cl);
                float tok_lin01_36 = _min_17;
                float _max_10 = max_noftz(_tmem_load_1[8] * tok_sf0_33, neg_cl);
                float _min_18 = fminf(_max_10, cl);
                float tok_lin10_37 = _min_18;
                float _max_11 = max_noftz(_tmem_load_1[9] * tok_sf1_34, neg_cl);
                float _min_19 = fminf(_max_11, cl);
                float tok_lin11_38 = _min_19;
                float tok_x00_39 = _tmem_load_0[10] * tok_sf0_33;
                float tok_x01_40 = _tmem_load_0[11] * tok_sf1_34;
                float tok_x10_41 = _tmem_load_1[10] * tok_sf0_33;
                float tok_x11_42 = _tmem_load_1[11] * tok_sf1_34;
                float _exp2_8 = approx_exp2(-(tok_x00_39 * fused));
                float _rcp_8 = approx_rcp(1.0f + _exp2_8);
                float tok_sig00_43 = _rcp_8;
                float _exp2_9 = approx_exp2(-(tok_x01_40 * fused));
                float _rcp_9 = approx_rcp(1.0f + _exp2_9);
                float tok_sig01_44 = _rcp_9;
                float _exp2_10 = approx_exp2(-(tok_x10_41 * fused));
                float _rcp_10 = approx_rcp(1.0f + _exp2_10);
                float tok_sig10_45 = _rcp_10;
                float _exp2_11 = approx_exp2(-(tok_x11_42 * fused));
                float _rcp_11 = approx_rcp(1.0f + _exp2_11);
                float tok_sig11_46 = _rcp_11;
                float _min_20 = fminf(tok_x00_39 * tok_sig00_43, cl);
                float tok_g00_47 = _min_20;
                float _min_21 = fminf(tok_x01_40 * tok_sig01_44, cl);
                float tok_g01_48 = _min_21;
                float _min_22 = fminf(tok_x10_41 * tok_sig10_45, cl);
                float tok_g10_49 = _min_22;
                float _min_23 = fminf(tok_x11_42 * tok_sig11_46, cl);
                float tok_g11_50 = _min_23;
                float _fma_8 = __fmaf_rn(tok_lin00_35, sg, 0.0f);
                float tok_acc00_51 = _fma_8 * tok_g00_47;
                float _fma_9 = __fmaf_rn(tok_lin01_36, sg, 0.0f);
                float tok_acc01_52 = _fma_9 * tok_g01_48;
                float _fma_10 = __fmaf_rn(tok_lin10_37, sg, 0.0f);
                float tok_acc10_53 = _fma_10 * tok_g10_49;
                float _fma_11 = __fmaf_rn(tok_lin11_38, sg, 0.0f);
                float tok_acc11_54 = _fma_11 * tok_g11_50;
                float2 _f2_6 = make_float2(sc, sc);
                float2 tok_sc_55 = _f2_6;
                float2 _f2_7 = make_float2(tok_acc00_51, tok_acc10_53);
                float2 _mul_f32x2_4;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_4) : "l"(*(const unsigned long long*)&_f2_7), "l"(*(const unsigned long long*)&tok_sc_55));
                float2 tok_out0_56 = _mul_f32x2_4;
                float2 _f2_8 = make_float2(tok_acc01_52, tok_acc11_54);
                float2 _mul_f32x2_5;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_5) : "l"(*(const unsigned long long*)&_f2_8), "l"(*(const unsigned long long*)&tok_sc_55));
                float2 tok_out1_57 = _mul_f32x2_5;
                float value00_58 = tok_out0_56.x;
                float value10_59 = tok_out0_56.y;
                float value01_60 = tok_out1_57.x;
                float value11_61 = tok_out1_57.y;
                out_pair[0] = value00_58;
                out_pair[1] = value10_59;
                if (token0_31 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token0_31) * M_out + tok_feature)))[0]) = _pk;
                    }
                }
                out_pair[0] = value01_60;
                out_pair[1] = value11_61;
                if (token1_32 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token1_32) * M_out + tok_feature)))[0]) = _pk;
                    }
                }
                int token0_62 = lane_1 % 4 * 2 + 24;
                int token1_63 = token0_62 + 1;
                float tok_sf0_64 = 0.0f;
                float tok_sf1_65 = 0.0f;
                if (token0_62 < valid_rows) {
                    int tok_route0_3 = route_map[tok_slot_base + token0_62];
                    tok_sf0_64 = SFC[tok_route0_3];
                }
                if (token1_63 < valid_rows) {
                    int tok_route1_3 = route_map[tok_slot_base + token1_63];
                    tok_sf1_65 = SFC[tok_route1_3];
                }
                float _max_12 = max_noftz(_tmem_load_0[12] * tok_sf0_64, neg_cl);
                float _min_24 = fminf(_max_12, cl);
                float tok_lin00_66 = _min_24;
                float _max_13 = max_noftz(_tmem_load_0[13] * tok_sf1_65, neg_cl);
                float _min_25 = fminf(_max_13, cl);
                float tok_lin01_67 = _min_25;
                float _max_14 = max_noftz(_tmem_load_1[12] * tok_sf0_64, neg_cl);
                float _min_26 = fminf(_max_14, cl);
                float tok_lin10_68 = _min_26;
                float _max_15 = max_noftz(_tmem_load_1[13] * tok_sf1_65, neg_cl);
                float _min_27 = fminf(_max_15, cl);
                float tok_lin11_69 = _min_27;
                float tok_x00_70 = _tmem_load_0[14] * tok_sf0_64;
                float tok_x01_71 = _tmem_load_0[15] * tok_sf1_65;
                float tok_x10_72 = _tmem_load_1[14] * tok_sf0_64;
                float tok_x11_73 = _tmem_load_1[15] * tok_sf1_65;
                float _exp2_12 = approx_exp2(-(tok_x00_70 * fused));
                float _rcp_12 = approx_rcp(1.0f + _exp2_12);
                float tok_sig00_74 = _rcp_12;
                float _exp2_13 = approx_exp2(-(tok_x01_71 * fused));
                float _rcp_13 = approx_rcp(1.0f + _exp2_13);
                float tok_sig01_75 = _rcp_13;
                float _exp2_14 = approx_exp2(-(tok_x10_72 * fused));
                float _rcp_14 = approx_rcp(1.0f + _exp2_14);
                float tok_sig10_76 = _rcp_14;
                float _exp2_15 = approx_exp2(-(tok_x11_73 * fused));
                float _rcp_15 = approx_rcp(1.0f + _exp2_15);
                float tok_sig11_77 = _rcp_15;
                float _min_28 = fminf(tok_x00_70 * tok_sig00_74, cl);
                float tok_g00_78 = _min_28;
                float _min_29 = fminf(tok_x01_71 * tok_sig01_75, cl);
                float tok_g01_79 = _min_29;
                float _min_30 = fminf(tok_x10_72 * tok_sig10_76, cl);
                float tok_g10_80 = _min_30;
                float _min_31 = fminf(tok_x11_73 * tok_sig11_77, cl);
                float tok_g11_81 = _min_31;
                float _fma_12 = __fmaf_rn(tok_lin00_66, sg, 0.0f);
                float tok_acc00_82 = _fma_12 * tok_g00_78;
                float _fma_13 = __fmaf_rn(tok_lin01_67, sg, 0.0f);
                float tok_acc01_83 = _fma_13 * tok_g01_79;
                float _fma_14 = __fmaf_rn(tok_lin10_68, sg, 0.0f);
                float tok_acc10_84 = _fma_14 * tok_g10_80;
                float _fma_15 = __fmaf_rn(tok_lin11_69, sg, 0.0f);
                float tok_acc11_85 = _fma_15 * tok_g11_81;
                float2 _f2_9 = make_float2(sc, sc);
                float2 tok_sc_86 = _f2_9;
                float2 _f2_10 = make_float2(tok_acc00_82, tok_acc10_84);
                float2 _mul_f32x2_6;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_6) : "l"(*(const unsigned long long*)&_f2_10), "l"(*(const unsigned long long*)&tok_sc_86));
                float2 tok_out0_87 = _mul_f32x2_6;
                float2 _f2_11 = make_float2(tok_acc01_83, tok_acc11_85);
                float2 _mul_f32x2_7;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_7) : "l"(*(const unsigned long long*)&_f2_11), "l"(*(const unsigned long long*)&tok_sc_86));
                float2 tok_out1_88 = _mul_f32x2_7;
                float value00_89 = tok_out0_87.x;
                float value10_90 = tok_out0_87.y;
                float value01_91 = tok_out1_88.x;
                float value11_92 = tok_out1_88.y;
                out_pair[0] = value00_89;
                out_pair[1] = value10_90;
                if (token0_62 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token0_62) * M_out + tok_feature)))[0]) = _pk;
                    }
                }
                out_pair[0] = value01_91;
                out_pair[1] = value11_92;
                if (token1_63 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token1_63) * M_out + tok_feature)))[0]) = _pk;
                    }
                }
                int acc_offset_93 = acc_stage * 64 + 32;
                float _tmem_load_2[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[15]))
                    : "r"(taddr + (unsigned int)acc_offset_93));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float _tmem_load_3[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[15]))
                    : "r"(taddr + 1048576 + (unsigned int)acc_offset_93));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                int base_row_94 = warp_0 * 16 + lane_1 / 4 * 2;
                int tok_feature_95 = (int)m_tile * 64 + base_row_94;
                asm volatile("cp.async.bulk.wait_group.read 0;");
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                int token0_96 = lane_1 % 4 * 2 + 32;
                int token1_97 = token0_96 + 1;
                float tok_sf0_98 = 0.0f;
                float tok_sf1_99 = 0.0f;
                if (token0_96 < valid_rows) {
                    int tok_route0_4 = route_map[tok_slot_base + token0_96];
                    tok_sf0_98 = SFC[tok_route0_4];
                }
                if (token1_97 < valid_rows) {
                    int tok_route1_4 = route_map[tok_slot_base + token1_97];
                    tok_sf1_99 = SFC[tok_route1_4];
                }
                float _max_16 = max_noftz(_tmem_load_2[0] * tok_sf0_98, neg_cl);
                float _min_32 = fminf(_max_16, cl);
                float tok_lin00_100 = _min_32;
                float _max_17 = max_noftz(_tmem_load_2[1] * tok_sf1_99, neg_cl);
                float _min_33 = fminf(_max_17, cl);
                float tok_lin01_101 = _min_33;
                float _max_18 = max_noftz(_tmem_load_3[0] * tok_sf0_98, neg_cl);
                float _min_34 = fminf(_max_18, cl);
                float tok_lin10_102 = _min_34;
                float _max_19 = max_noftz(_tmem_load_3[1] * tok_sf1_99, neg_cl);
                float _min_35 = fminf(_max_19, cl);
                float tok_lin11_103 = _min_35;
                float tok_x00_104 = _tmem_load_2[2] * tok_sf0_98;
                float tok_x01_105 = _tmem_load_2[3] * tok_sf1_99;
                float tok_x10_106 = _tmem_load_3[2] * tok_sf0_98;
                float tok_x11_107 = _tmem_load_3[3] * tok_sf1_99;
                float _exp2_16 = approx_exp2(-(tok_x00_104 * fused));
                float _rcp_16 = approx_rcp(1.0f + _exp2_16);
                float tok_sig00_108 = _rcp_16;
                float _exp2_17 = approx_exp2(-(tok_x01_105 * fused));
                float _rcp_17 = approx_rcp(1.0f + _exp2_17);
                float tok_sig01_109 = _rcp_17;
                float _exp2_18 = approx_exp2(-(tok_x10_106 * fused));
                float _rcp_18 = approx_rcp(1.0f + _exp2_18);
                float tok_sig10_110 = _rcp_18;
                float _exp2_19 = approx_exp2(-(tok_x11_107 * fused));
                float _rcp_19 = approx_rcp(1.0f + _exp2_19);
                float tok_sig11_111 = _rcp_19;
                float _min_36 = fminf(tok_x00_104 * tok_sig00_108, cl);
                float tok_g00_112 = _min_36;
                float _min_37 = fminf(tok_x01_105 * tok_sig01_109, cl);
                float tok_g01_113 = _min_37;
                float _min_38 = fminf(tok_x10_106 * tok_sig10_110, cl);
                float tok_g10_114 = _min_38;
                float _min_39 = fminf(tok_x11_107 * tok_sig11_111, cl);
                float tok_g11_115 = _min_39;
                float _fma_16 = __fmaf_rn(tok_lin00_100, sg, 0.0f);
                float tok_acc00_116 = _fma_16 * tok_g00_112;
                float _fma_17 = __fmaf_rn(tok_lin01_101, sg, 0.0f);
                float tok_acc01_117 = _fma_17 * tok_g01_113;
                float _fma_18 = __fmaf_rn(tok_lin10_102, sg, 0.0f);
                float tok_acc10_118 = _fma_18 * tok_g10_114;
                float _fma_19 = __fmaf_rn(tok_lin11_103, sg, 0.0f);
                float tok_acc11_119 = _fma_19 * tok_g11_115;
                float2 _f2_12 = make_float2(sc, sc);
                float2 tok_sc_120 = _f2_12;
                float2 _f2_13 = make_float2(tok_acc00_116, tok_acc10_118);
                float2 _mul_f32x2_8;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_8) : "l"(*(const unsigned long long*)&_f2_13), "l"(*(const unsigned long long*)&tok_sc_120));
                float2 tok_out0_121 = _mul_f32x2_8;
                float2 _f2_14 = make_float2(tok_acc01_117, tok_acc11_119);
                float2 _mul_f32x2_9;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_9) : "l"(*(const unsigned long long*)&_f2_14), "l"(*(const unsigned long long*)&tok_sc_120));
                float2 tok_out1_122 = _mul_f32x2_9;
                float value00_123 = tok_out0_121.x;
                float value10_124 = tok_out0_121.y;
                float value01_125 = tok_out1_122.x;
                float value11_126 = tok_out1_122.y;
                out_pair[0] = value00_123;
                out_pair[1] = value10_124;
                if (token0_96 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token0_96) * M_out + tok_feature_95)))[0]) = _pk;
                    }
                }
                out_pair[0] = value01_125;
                out_pair[1] = value11_126;
                if (token1_97 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token1_97) * M_out + tok_feature_95)))[0]) = _pk;
                    }
                }
                int token0_127 = lane_1 % 4 * 2 + 8 + 32;
                int token1_128 = token0_127 + 1;
                float tok_sf0_129 = 0.0f;
                float tok_sf1_130 = 0.0f;
                if (token0_127 < valid_rows) {
                    int tok_route0_5 = route_map[tok_slot_base + token0_127];
                    tok_sf0_129 = SFC[tok_route0_5];
                }
                if (token1_128 < valid_rows) {
                    int tok_route1_5 = route_map[tok_slot_base + token1_128];
                    tok_sf1_130 = SFC[tok_route1_5];
                }
                float _max_20 = max_noftz(_tmem_load_2[4] * tok_sf0_129, neg_cl);
                float _min_40 = fminf(_max_20, cl);
                float tok_lin00_131 = _min_40;
                float _max_21 = max_noftz(_tmem_load_2[5] * tok_sf1_130, neg_cl);
                float _min_41 = fminf(_max_21, cl);
                float tok_lin01_132 = _min_41;
                float _max_22 = max_noftz(_tmem_load_3[4] * tok_sf0_129, neg_cl);
                float _min_42 = fminf(_max_22, cl);
                float tok_lin10_133 = _min_42;
                float _max_23 = max_noftz(_tmem_load_3[5] * tok_sf1_130, neg_cl);
                float _min_43 = fminf(_max_23, cl);
                float tok_lin11_134 = _min_43;
                float tok_x00_135 = _tmem_load_2[6] * tok_sf0_129;
                float tok_x01_136 = _tmem_load_2[7] * tok_sf1_130;
                float tok_x10_137 = _tmem_load_3[6] * tok_sf0_129;
                float tok_x11_138 = _tmem_load_3[7] * tok_sf1_130;
                float _exp2_20 = approx_exp2(-(tok_x00_135 * fused));
                float _rcp_20 = approx_rcp(1.0f + _exp2_20);
                float tok_sig00_139 = _rcp_20;
                float _exp2_21 = approx_exp2(-(tok_x01_136 * fused));
                float _rcp_21 = approx_rcp(1.0f + _exp2_21);
                float tok_sig01_140 = _rcp_21;
                float _exp2_22 = approx_exp2(-(tok_x10_137 * fused));
                float _rcp_22 = approx_rcp(1.0f + _exp2_22);
                float tok_sig10_141 = _rcp_22;
                float _exp2_23 = approx_exp2(-(tok_x11_138 * fused));
                float _rcp_23 = approx_rcp(1.0f + _exp2_23);
                float tok_sig11_142 = _rcp_23;
                float _min_44 = fminf(tok_x00_135 * tok_sig00_139, cl);
                float tok_g00_143 = _min_44;
                float _min_45 = fminf(tok_x01_136 * tok_sig01_140, cl);
                float tok_g01_144 = _min_45;
                float _min_46 = fminf(tok_x10_137 * tok_sig10_141, cl);
                float tok_g10_145 = _min_46;
                float _min_47 = fminf(tok_x11_138 * tok_sig11_142, cl);
                float tok_g11_146 = _min_47;
                float _fma_20 = __fmaf_rn(tok_lin00_131, sg, 0.0f);
                float tok_acc00_147 = _fma_20 * tok_g00_143;
                float _fma_21 = __fmaf_rn(tok_lin01_132, sg, 0.0f);
                float tok_acc01_148 = _fma_21 * tok_g01_144;
                float _fma_22 = __fmaf_rn(tok_lin10_133, sg, 0.0f);
                float tok_acc10_149 = _fma_22 * tok_g10_145;
                float _fma_23 = __fmaf_rn(tok_lin11_134, sg, 0.0f);
                float tok_acc11_150 = _fma_23 * tok_g11_146;
                float2 _f2_15 = make_float2(sc, sc);
                float2 tok_sc_151 = _f2_15;
                float2 _f2_16 = make_float2(tok_acc00_147, tok_acc10_149);
                float2 _mul_f32x2_10;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_10) : "l"(*(const unsigned long long*)&_f2_16), "l"(*(const unsigned long long*)&tok_sc_151));
                float2 tok_out0_152 = _mul_f32x2_10;
                float2 _f2_17 = make_float2(tok_acc01_148, tok_acc11_150);
                float2 _mul_f32x2_11;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_11) : "l"(*(const unsigned long long*)&_f2_17), "l"(*(const unsigned long long*)&tok_sc_151));
                float2 tok_out1_153 = _mul_f32x2_11;
                float value00_154 = tok_out0_152.x;
                float value10_155 = tok_out0_152.y;
                float value01_156 = tok_out1_153.x;
                float value11_157 = tok_out1_153.y;
                out_pair[0] = value00_154;
                out_pair[1] = value10_155;
                if (token0_127 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token0_127) * M_out + tok_feature_95)))[0]) = _pk;
                    }
                }
                out_pair[0] = value01_156;
                out_pair[1] = value11_157;
                if (token1_128 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token1_128) * M_out + tok_feature_95)))[0]) = _pk;
                    }
                }
                int token0_158 = lane_1 % 4 * 2 + 16 + 32;
                int token1_159 = token0_158 + 1;
                float tok_sf0_160 = 0.0f;
                float tok_sf1_161 = 0.0f;
                if (token0_158 < valid_rows) {
                    int tok_route0_6 = route_map[tok_slot_base + token0_158];
                    tok_sf0_160 = SFC[tok_route0_6];
                }
                if (token1_159 < valid_rows) {
                    int tok_route1_6 = route_map[tok_slot_base + token1_159];
                    tok_sf1_161 = SFC[tok_route1_6];
                }
                float _max_24 = max_noftz(_tmem_load_2[8] * tok_sf0_160, neg_cl);
                float _min_48 = fminf(_max_24, cl);
                float tok_lin00_162 = _min_48;
                float _max_25 = max_noftz(_tmem_load_2[9] * tok_sf1_161, neg_cl);
                float _min_49 = fminf(_max_25, cl);
                float tok_lin01_163 = _min_49;
                float _max_26 = max_noftz(_tmem_load_3[8] * tok_sf0_160, neg_cl);
                float _min_50 = fminf(_max_26, cl);
                float tok_lin10_164 = _min_50;
                float _max_27 = max_noftz(_tmem_load_3[9] * tok_sf1_161, neg_cl);
                float _min_51 = fminf(_max_27, cl);
                float tok_lin11_165 = _min_51;
                float tok_x00_166 = _tmem_load_2[10] * tok_sf0_160;
                float tok_x01_167 = _tmem_load_2[11] * tok_sf1_161;
                float tok_x10_168 = _tmem_load_3[10] * tok_sf0_160;
                float tok_x11_169 = _tmem_load_3[11] * tok_sf1_161;
                float _exp2_24 = approx_exp2(-(tok_x00_166 * fused));
                float _rcp_24 = approx_rcp(1.0f + _exp2_24);
                float tok_sig00_170 = _rcp_24;
                float _exp2_25 = approx_exp2(-(tok_x01_167 * fused));
                float _rcp_25 = approx_rcp(1.0f + _exp2_25);
                float tok_sig01_171 = _rcp_25;
                float _exp2_26 = approx_exp2(-(tok_x10_168 * fused));
                float _rcp_26 = approx_rcp(1.0f + _exp2_26);
                float tok_sig10_172 = _rcp_26;
                float _exp2_27 = approx_exp2(-(tok_x11_169 * fused));
                float _rcp_27 = approx_rcp(1.0f + _exp2_27);
                float tok_sig11_173 = _rcp_27;
                float _min_52 = fminf(tok_x00_166 * tok_sig00_170, cl);
                float tok_g00_174 = _min_52;
                float _min_53 = fminf(tok_x01_167 * tok_sig01_171, cl);
                float tok_g01_175 = _min_53;
                float _min_54 = fminf(tok_x10_168 * tok_sig10_172, cl);
                float tok_g10_176 = _min_54;
                float _min_55 = fminf(tok_x11_169 * tok_sig11_173, cl);
                float tok_g11_177 = _min_55;
                float _fma_24 = __fmaf_rn(tok_lin00_162, sg, 0.0f);
                float tok_acc00_178 = _fma_24 * tok_g00_174;
                float _fma_25 = __fmaf_rn(tok_lin01_163, sg, 0.0f);
                float tok_acc01_179 = _fma_25 * tok_g01_175;
                float _fma_26 = __fmaf_rn(tok_lin10_164, sg, 0.0f);
                float tok_acc10_180 = _fma_26 * tok_g10_176;
                float _fma_27 = __fmaf_rn(tok_lin11_165, sg, 0.0f);
                float tok_acc11_181 = _fma_27 * tok_g11_177;
                float2 _f2_18 = make_float2(sc, sc);
                float2 tok_sc_182 = _f2_18;
                float2 _f2_19 = make_float2(tok_acc00_178, tok_acc10_180);
                float2 _mul_f32x2_12;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_12) : "l"(*(const unsigned long long*)&_f2_19), "l"(*(const unsigned long long*)&tok_sc_182));
                float2 tok_out0_183 = _mul_f32x2_12;
                float2 _f2_20 = make_float2(tok_acc01_179, tok_acc11_181);
                float2 _mul_f32x2_13;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_13) : "l"(*(const unsigned long long*)&_f2_20), "l"(*(const unsigned long long*)&tok_sc_182));
                float2 tok_out1_184 = _mul_f32x2_13;
                float value00_185 = tok_out0_183.x;
                float value10_186 = tok_out0_183.y;
                float value01_187 = tok_out1_184.x;
                float value11_188 = tok_out1_184.y;
                out_pair[0] = value00_185;
                out_pair[1] = value10_186;
                if (token0_158 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token0_158) * M_out + tok_feature_95)))[0]) = _pk;
                    }
                }
                out_pair[0] = value01_187;
                out_pair[1] = value11_188;
                if (token1_159 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token1_159) * M_out + tok_feature_95)))[0]) = _pk;
                    }
                }
                int token0_189 = lane_1 % 4 * 2 + 24 + 32;
                int token1_190 = token0_189 + 1;
                float tok_sf0_191 = 0.0f;
                float tok_sf1_192 = 0.0f;
                if (token0_189 < valid_rows) {
                    int tok_route0_7 = route_map[tok_slot_base + token0_189];
                    tok_sf0_191 = SFC[tok_route0_7];
                }
                if (token1_190 < valid_rows) {
                    int tok_route1_7 = route_map[tok_slot_base + token1_190];
                    tok_sf1_192 = SFC[tok_route1_7];
                }
                float _max_28 = max_noftz(_tmem_load_2[12] * tok_sf0_191, neg_cl);
                float _min_56 = fminf(_max_28, cl);
                float tok_lin00_193 = _min_56;
                float _max_29 = max_noftz(_tmem_load_2[13] * tok_sf1_192, neg_cl);
                float _min_57 = fminf(_max_29, cl);
                float tok_lin01_194 = _min_57;
                float _max_30 = max_noftz(_tmem_load_3[12] * tok_sf0_191, neg_cl);
                float _min_58 = fminf(_max_30, cl);
                float tok_lin10_195 = _min_58;
                float _max_31 = max_noftz(_tmem_load_3[13] * tok_sf1_192, neg_cl);
                float _min_59 = fminf(_max_31, cl);
                float tok_lin11_196 = _min_59;
                float tok_x00_197 = _tmem_load_2[14] * tok_sf0_191;
                float tok_x01_198 = _tmem_load_2[15] * tok_sf1_192;
                float tok_x10_199 = _tmem_load_3[14] * tok_sf0_191;
                float tok_x11_200 = _tmem_load_3[15] * tok_sf1_192;
                float _exp2_28 = approx_exp2(-(tok_x00_197 * fused));
                float _rcp_28 = approx_rcp(1.0f + _exp2_28);
                float tok_sig00_201 = _rcp_28;
                float _exp2_29 = approx_exp2(-(tok_x01_198 * fused));
                float _rcp_29 = approx_rcp(1.0f + _exp2_29);
                float tok_sig01_202 = _rcp_29;
                float _exp2_30 = approx_exp2(-(tok_x10_199 * fused));
                float _rcp_30 = approx_rcp(1.0f + _exp2_30);
                float tok_sig10_203 = _rcp_30;
                float _exp2_31 = approx_exp2(-(tok_x11_200 * fused));
                float _rcp_31 = approx_rcp(1.0f + _exp2_31);
                float tok_sig11_204 = _rcp_31;
                float _min_60 = fminf(tok_x00_197 * tok_sig00_201, cl);
                float tok_g00_205 = _min_60;
                float _min_61 = fminf(tok_x01_198 * tok_sig01_202, cl);
                float tok_g01_206 = _min_61;
                float _min_62 = fminf(tok_x10_199 * tok_sig10_203, cl);
                float tok_g10_207 = _min_62;
                float _min_63 = fminf(tok_x11_200 * tok_sig11_204, cl);
                float tok_g11_208 = _min_63;
                float _fma_28 = __fmaf_rn(tok_lin00_193, sg, 0.0f);
                float tok_acc00_209 = _fma_28 * tok_g00_205;
                float _fma_29 = __fmaf_rn(tok_lin01_194, sg, 0.0f);
                float tok_acc01_210 = _fma_29 * tok_g01_206;
                float _fma_30 = __fmaf_rn(tok_lin10_195, sg, 0.0f);
                float tok_acc10_211 = _fma_30 * tok_g10_207;
                float _fma_31 = __fmaf_rn(tok_lin11_196, sg, 0.0f);
                float tok_acc11_212 = _fma_31 * tok_g11_208;
                float2 _f2_21 = make_float2(sc, sc);
                float2 tok_sc_213 = _f2_21;
                float2 _f2_22 = make_float2(tok_acc00_209, tok_acc10_211);
                float2 _mul_f32x2_14;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_14) : "l"(*(const unsigned long long*)&_f2_22), "l"(*(const unsigned long long*)&tok_sc_213));
                float2 tok_out0_214 = _mul_f32x2_14;
                float2 _f2_23 = make_float2(tok_acc01_210, tok_acc11_212);
                float2 _mul_f32x2_15;
                asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_15) : "l"(*(const unsigned long long*)&_f2_23), "l"(*(const unsigned long long*)&tok_sc_213));
                float2 tok_out1_215 = _mul_f32x2_15;
                float value00_216 = tok_out0_214.x;
                float value10_217 = tok_out0_214.y;
                float value01_218 = tok_out1_215.x;
                float value11_219 = tok_out1_215.y;
                out_pair[0] = value00_216;
                out_pair[1] = value10_217;
                if (token0_189 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token0_189) * M_out + tok_feature_95)))[0]) = _pk;
                    }
                }
                out_pair[0] = value01_218;
                out_pair[1] = value11_219;
                if (token1_190 < valid_rows) {
                    {
                        __nv_bfloat162 _pk = __floats2bfloat162_rn(out_pair[0 + 0], out_pair[0 + 1]);
                        *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(C + ((tok_slot_base + token1_190) * M_out + tok_feature_95)))[0]) = _pk;
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
