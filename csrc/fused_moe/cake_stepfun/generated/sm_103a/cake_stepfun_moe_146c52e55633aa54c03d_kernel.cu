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
#define NUM_K_PIPE_STAGES 5
#define NUM_MMA_PIPE_STAGES 1
#define NUM_WORK_PIPE_STAGES 3
#define NUM_THROTTLE_PIPE_STAGES 3
#define NUM_FAST_PIPE_STAGES 1
#define NUM_EXIT_PIPE_STAGES 1
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 16384
#define SMEM_SMEM_B_OFF 82944
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 16384
#define SMEM_EPI_STAGING_OFF 164864
#define SMEM_EPI_STAGING_STAGE_BYTES 16384
#define SMEM_EPI_STAGING_STRIDE 16384
#define SMEM_EPI_STAGING_U64_OFF 164864
#define SMEM_EPI_STAGING_U64_STAGE_BYTES 16384
#define SMEM_EPI_STAGING_U64_STRIDE 16384
#define SMEM_SMEM_SFA_OFF 188416
#define SMEM_SMEM_SFA_STAGE_BYTES 2048
#define SMEM_SMEM_SFA_STRIDE 2048
#define SMEM_SMEM_SFB_OFF 198656
#define SMEM_SMEM_SFB_STAGE_BYTES 4096
#define SMEM_SMEM_SFB_STRIDE 4096
#define SMEM_WORK_RESPONSE_OFF 220160
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_FAST_RESPONSE_OFF 220208
#define SMEM_FAST_RESPONSE_STAGE_BYTES 64
#define SMEM_FAST_RESPONSE_STRIDE 64
#define SMEM_TOTAL 220288
#define THREADS 384
#define BLOCK_N 256
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






__device__ __forceinline__ void tma_store_4d(
    const void *tmap, int x, int y, int z, int w, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3, %4}], [%5];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(w), "r"(smem_addr) : "memory");
}


extern "C" {

__global__ __launch_bounds__(384, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_cake_stepfun_moe_146c52e55633aa54c03d(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, const __grid_constant__ CUtensorMap C, int* __restrict__ tile_expert, int* __restrict__ tile_mn_limit, int* __restrict__ num_non_exiting_ctas, float* __restrict__ scale_c, int M_out, int K, int grid_m, int grid_n, int K_tiles)
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
    #define b_full_addr (mbar_base + 40)
    #define sfa_full_addr (mbar_base + 80)
    #define sfb_full_addr (mbar_base + 120)
    #define a_empty_addr (mbar_base + 160)
    #define b_empty_addr (mbar_base + 200)
    #define sfa_empty_addr (mbar_base + 240)
    #define sfb_empty_addr (mbar_base + 280)
    #define mma_full_addr (mbar_base + 320)
    #define mma_free_addr (mbar_base + 328)
    #define work_full_addr (mbar_base + 336)
    #define work_empty_addr (mbar_base + 360)
    #define throttle_full_addr (mbar_base + 384)
    #define throttle_empty_addr (mbar_base + 408)
    #define fast_ready_addr (mbar_base + 432)
    #define exit_bar_addr (mbar_base + 440)

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
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + 82944);
    const int smem_b_addr = smem + 82944;
    __nv_bfloat16* epi_staging = reinterpret_cast<__nv_bfloat16*>(smem_raw + 164864);
    const int epi_staging_addr = smem + 164864;
    unsigned long long* epi_staging_u64 = reinterpret_cast<unsigned long long*>(smem_raw + 164864);
    const int epi_staging_u64_addr = smem + 164864;
    uint8_t* smem_sfa = reinterpret_cast<uint8_t*>(smem_raw + 188416);
    const int smem_sfa_addr = smem + 188416;
    uint8_t* smem_sfb = reinterpret_cast<uint8_t*>(smem_raw + 198656);
    const int smem_sfb_addr = smem + 198656;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 220160);
    const int work_response_addr = smem + 220160;
    unsigned int* fast_response = reinterpret_cast<unsigned int*>(smem_raw + 220208);
    const int fast_response_addr = smem + 220208;

    // Mbarrier init (16 pipeline groups, 0 ordered-sequence groups, 56 barriers)
    // Mbarriers at smem_raw[0..448)

    if (warp == 0) {
        // --- pipeline 'k_pipe' ---
        // a_full: 5 barriers, init_count=1
        // b_full: 5 barriers, init_count=1
        // sfa_full: 5 barriers, init_count=1
        // sfb_full: 5 barriers, init_count=1
        // a_empty: 5 barriers, init_count=1
        // b_empty: 5 barriers, init_count=1
        // sfa_empty: 5 barriers, init_count=1
        // sfb_empty: 5 barriers, init_count=1
        // --- pipeline 'mma_pipe' ---
        // mma_full: 1 barriers, init_count=1
        // mma_free: 1 barriers, init_count=256
        // --- pipeline 'work_pipe' ---
        // work_full: 3 barriers, init_count=1
        // work_empty: 3 barriers, init_count=672
        // --- pipeline 'throttle_pipe' ---
        // throttle_full: 3 barriers, init_count=32
        // throttle_empty: 3 barriers, init_count=32
        // --- pipeline 'fast_pipe' ---
        // fast_ready: 1 barriers, init_count=1
        // --- pipeline 'exit_pipe' ---
        // exit_bar: 1 barriers, init_count=32
        // Warp-cooperative initialization in physical record order.
        mbarrier_init(smem + 0 + lane * 8, 1);
        uint32_t _mbarrier_init_count_0_32 = 32;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(23), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(22), "r"((uint32_t)(32)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(16), "r"((uint32_t)(672)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(13), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(10), "r"((uint32_t)(256)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(9), "r"((uint32_t)(1)));
        if (lane < 24) {
            mbarrier_init(smem + 256 + lane * 8, _mbarrier_init_count_0_32);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    __syncwarp();

    // TMEM alloc (512 columns, 496 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 448);
    if (warp == 0) {
        int _tmem_hold = smem + 448;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;
    const int tmem_sfa = taddr + 448;
    const int tmem_sfb = taddr + 464;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 4 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
    }

    // ---- Role: epilogue ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 160;");
        { // epilogue_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            int tile_count = num_non_exiting_ctas[0];
            const int warp_0 = warp;
            const int lane_1 = lane;
            unsigned int work_stage = 0;
            int epilogue_local_idx = 0;
            unsigned int m_tile = blockIdx.x;
            unsigned int n_tile = blockIdx.y;
            int base_feature = warp_0 * 32 + lane_1 / 4 * 4;
            int base_token = lane_1 % 4 * 2;
            int row_addr = warp_0 * 32 << 16;
            float converted[4];
            unsigned int packed[2];
            unsigned long long packed_word = 0;
            float frag[64];
            unsigned int _phase_mma_full_0 = 0;
            unsigned int _phase_work_full = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter = 0; _tile_iter < grid_m / 2 * grid_n; _tile_iter++) {
                int valid_rows = (unsigned int)tile_mn_limit[n_tile] - n_tile * 256;
                if (tile_count > (int)n_tile) {
                    if (valid_rows > 0) {
                        int expert_e = tile_expert[n_tile];
                        float sc = scale_c[expert_e];
                        mbarrier_wait(mma_full_addr, _phase_mma_full_0);
                        _phase_mma_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int token_block = 0;
                        if (epilogue_local_idx == 0) {
                            token_block = 3;
                        }
                        int acc_col = epilogue_local_idx * 192 + token_block * 64;
                        float _tmem_load_0[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[31]))
                            : "r"(taddr + (unsigned int)row_addr + (unsigned int)acc_col));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        float _tmem_load_1[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[31]))
                            : "r"(taddr + (unsigned int)row_addr + 1048576 + (unsigned int)acc_col));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        for (int i = 0; i < 32; i++) {
                            frag[i] = _tmem_load_0[i];
                            frag[32 + i] = _tmem_load_1[i];
                        }
                        {
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                :: "r"((mma_free_addr) & 0xFEFFFFFF) : "memory");
                        }
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                        asm volatile("barrier.sync 7, 128;" ::: "memory");
                        int token = base_token;
                        float _mul_0;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_0) : "f"(frag[0]), "f"(sc));
                        converted[0] = _mul_0;
                        float _mul_1;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_1) : "f"(frag[2]), "f"(sc));
                        converted[1] = _mul_1;
                        float _mul_2;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_2) : "f"(frag[32]), "f"(sc));
                        converted[2] = _mul_2;
                        float _mul_3;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_3) : "f"(frag[34]), "f"(sc));
                        converted[3] = _mul_3;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature = token % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token * 64 + (base_feature ^ swizzle_feature)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token * 64 + (base_feature - 64 ^ swizzle_feature)) / 4] = packed_word;
                        }
                        int token_0 = base_token + 1;
                        float _mul_4;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_4) : "f"(frag[1]), "f"(sc));
                        converted[0] = _mul_4;
                        float _mul_5;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_5) : "f"(frag[3]), "f"(sc));
                        converted[1] = _mul_5;
                        float _mul_6;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_6) : "f"(frag[33]), "f"(sc));
                        converted[2] = _mul_6;
                        float _mul_7;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_7) : "f"(frag[35]), "f"(sc));
                        converted[3] = _mul_7;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_1 = token_0 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_0 * 64 + (base_feature ^ swizzle_feature_1)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_0 * 64 + (base_feature - 64 ^ swizzle_feature_1)) / 4] = packed_word;
                        }
                        int token_2 = base_token + 8;
                        float _mul_8;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_8) : "f"(frag[4]), "f"(sc));
                        converted[0] = _mul_8;
                        float _mul_9;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_9) : "f"(frag[6]), "f"(sc));
                        converted[1] = _mul_9;
                        float _mul_10;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_10) : "f"(frag[36]), "f"(sc));
                        converted[2] = _mul_10;
                        float _mul_11;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_11) : "f"(frag[38]), "f"(sc));
                        converted[3] = _mul_11;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_3 = token_2 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_2 * 64 + (base_feature ^ swizzle_feature_3)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_2 * 64 + (base_feature - 64 ^ swizzle_feature_3)) / 4] = packed_word;
                        }
                        int token_4 = base_token + 8 + 1;
                        float _mul_12;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_12) : "f"(frag[5]), "f"(sc));
                        converted[0] = _mul_12;
                        float _mul_13;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_13) : "f"(frag[7]), "f"(sc));
                        converted[1] = _mul_13;
                        float _mul_14;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_14) : "f"(frag[37]), "f"(sc));
                        converted[2] = _mul_14;
                        float _mul_15;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_15) : "f"(frag[39]), "f"(sc));
                        converted[3] = _mul_15;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_5 = token_4 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_4 * 64 + (base_feature ^ swizzle_feature_5)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_4 * 64 + (base_feature - 64 ^ swizzle_feature_5)) / 4] = packed_word;
                        }
                        int token_6 = base_token + 16;
                        float _mul_16;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_16) : "f"(frag[8]), "f"(sc));
                        converted[0] = _mul_16;
                        float _mul_17;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_17) : "f"(frag[10]), "f"(sc));
                        converted[1] = _mul_17;
                        float _mul_18;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_18) : "f"(frag[40]), "f"(sc));
                        converted[2] = _mul_18;
                        float _mul_19;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_19) : "f"(frag[42]), "f"(sc));
                        converted[3] = _mul_19;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_7 = token_6 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_6 * 64 + (base_feature ^ swizzle_feature_7)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_6 * 64 + (base_feature - 64 ^ swizzle_feature_7)) / 4] = packed_word;
                        }
                        int token_8 = base_token + 16 + 1;
                        float _mul_20;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_20) : "f"(frag[9]), "f"(sc));
                        converted[0] = _mul_20;
                        float _mul_21;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_21) : "f"(frag[11]), "f"(sc));
                        converted[1] = _mul_21;
                        float _mul_22;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_22) : "f"(frag[41]), "f"(sc));
                        converted[2] = _mul_22;
                        float _mul_23;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_23) : "f"(frag[43]), "f"(sc));
                        converted[3] = _mul_23;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_9 = token_8 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_8 * 64 + (base_feature ^ swizzle_feature_9)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_8 * 64 + (base_feature - 64 ^ swizzle_feature_9)) / 4] = packed_word;
                        }
                        int token_10 = base_token + 24;
                        float _mul_24;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_24) : "f"(frag[12]), "f"(sc));
                        converted[0] = _mul_24;
                        float _mul_25;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_25) : "f"(frag[14]), "f"(sc));
                        converted[1] = _mul_25;
                        float _mul_26;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_26) : "f"(frag[44]), "f"(sc));
                        converted[2] = _mul_26;
                        float _mul_27;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_27) : "f"(frag[46]), "f"(sc));
                        converted[3] = _mul_27;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_11 = token_10 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_10 * 64 + (base_feature ^ swizzle_feature_11)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_10 * 64 + (base_feature - 64 ^ swizzle_feature_11)) / 4] = packed_word;
                        }
                        int token_12 = base_token + 24 + 1;
                        float _mul_28;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_28) : "f"(frag[13]), "f"(sc));
                        converted[0] = _mul_28;
                        float _mul_29;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_29) : "f"(frag[15]), "f"(sc));
                        converted[1] = _mul_29;
                        float _mul_30;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_30) : "f"(frag[45]), "f"(sc));
                        converted[2] = _mul_30;
                        float _mul_31;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_31) : "f"(frag[47]), "f"(sc));
                        converted[3] = _mul_31;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_13 = token_12 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_12 * 64 + (base_feature ^ swizzle_feature_13)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_12 * 64 + (base_feature - 64 ^ swizzle_feature_13)) / 4] = packed_word;
                        }
                        int token_14 = base_token + 32;
                        float _mul_32;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_32) : "f"(frag[16]), "f"(sc));
                        converted[0] = _mul_32;
                        float _mul_33;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_33) : "f"(frag[18]), "f"(sc));
                        converted[1] = _mul_33;
                        float _mul_34;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_34) : "f"(frag[48]), "f"(sc));
                        converted[2] = _mul_34;
                        float _mul_35;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_35) : "f"(frag[50]), "f"(sc));
                        converted[3] = _mul_35;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_15 = token_14 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_14 * 64 + (base_feature ^ swizzle_feature_15)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_14 * 64 + (base_feature - 64 ^ swizzle_feature_15)) / 4] = packed_word;
                        }
                        int token_16 = base_token + 32 + 1;
                        float _mul_36;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_36) : "f"(frag[17]), "f"(sc));
                        converted[0] = _mul_36;
                        float _mul_37;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_37) : "f"(frag[19]), "f"(sc));
                        converted[1] = _mul_37;
                        float _mul_38;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_38) : "f"(frag[49]), "f"(sc));
                        converted[2] = _mul_38;
                        float _mul_39;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_39) : "f"(frag[51]), "f"(sc));
                        converted[3] = _mul_39;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_17 = token_16 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_16 * 64 + (base_feature ^ swizzle_feature_17)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_16 * 64 + (base_feature - 64 ^ swizzle_feature_17)) / 4] = packed_word;
                        }
                        int token_18 = base_token + 40;
                        float _mul_40;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_40) : "f"(frag[20]), "f"(sc));
                        converted[0] = _mul_40;
                        float _mul_41;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_41) : "f"(frag[22]), "f"(sc));
                        converted[1] = _mul_41;
                        float _mul_42;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_42) : "f"(frag[52]), "f"(sc));
                        converted[2] = _mul_42;
                        float _mul_43;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_43) : "f"(frag[54]), "f"(sc));
                        converted[3] = _mul_43;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_19 = token_18 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_18 * 64 + (base_feature ^ swizzle_feature_19)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_18 * 64 + (base_feature - 64 ^ swizzle_feature_19)) / 4] = packed_word;
                        }
                        int token_20 = base_token + 40 + 1;
                        float _mul_44;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_44) : "f"(frag[21]), "f"(sc));
                        converted[0] = _mul_44;
                        float _mul_45;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_45) : "f"(frag[23]), "f"(sc));
                        converted[1] = _mul_45;
                        float _mul_46;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_46) : "f"(frag[53]), "f"(sc));
                        converted[2] = _mul_46;
                        float _mul_47;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_47) : "f"(frag[55]), "f"(sc));
                        converted[3] = _mul_47;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_21 = token_20 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_20 * 64 + (base_feature ^ swizzle_feature_21)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_20 * 64 + (base_feature - 64 ^ swizzle_feature_21)) / 4] = packed_word;
                        }
                        int token_22 = base_token + 48;
                        float _mul_48;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_48) : "f"(frag[24]), "f"(sc));
                        converted[0] = _mul_48;
                        float _mul_49;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_49) : "f"(frag[26]), "f"(sc));
                        converted[1] = _mul_49;
                        float _mul_50;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_50) : "f"(frag[56]), "f"(sc));
                        converted[2] = _mul_50;
                        float _mul_51;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_51) : "f"(frag[58]), "f"(sc));
                        converted[3] = _mul_51;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_23 = token_22 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_22 * 64 + (base_feature ^ swizzle_feature_23)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_22 * 64 + (base_feature - 64 ^ swizzle_feature_23)) / 4] = packed_word;
                        }
                        int token_24 = base_token + 48 + 1;
                        float _mul_52;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_52) : "f"(frag[25]), "f"(sc));
                        converted[0] = _mul_52;
                        float _mul_53;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_53) : "f"(frag[27]), "f"(sc));
                        converted[1] = _mul_53;
                        float _mul_54;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_54) : "f"(frag[57]), "f"(sc));
                        converted[2] = _mul_54;
                        float _mul_55;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_55) : "f"(frag[59]), "f"(sc));
                        converted[3] = _mul_55;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_25 = token_24 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_24 * 64 + (base_feature ^ swizzle_feature_25)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_24 * 64 + (base_feature - 64 ^ swizzle_feature_25)) / 4] = packed_word;
                        }
                        int token_26 = base_token + 56;
                        float _mul_56;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_56) : "f"(frag[28]), "f"(sc));
                        converted[0] = _mul_56;
                        float _mul_57;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_57) : "f"(frag[30]), "f"(sc));
                        converted[1] = _mul_57;
                        float _mul_58;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_58) : "f"(frag[60]), "f"(sc));
                        converted[2] = _mul_58;
                        float _mul_59;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_59) : "f"(frag[62]), "f"(sc));
                        converted[3] = _mul_59;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_27 = token_26 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_26 * 64 + (base_feature ^ swizzle_feature_27)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_26 * 64 + (base_feature - 64 ^ swizzle_feature_27)) / 4] = packed_word;
                        }
                        int token_28 = base_token + 56 + 1;
                        float _mul_60;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_60) : "f"(frag[29]), "f"(sc));
                        converted[0] = _mul_60;
                        float _mul_61;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_61) : "f"(frag[31]), "f"(sc));
                        converted[1] = _mul_61;
                        float _mul_62;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_62) : "f"(frag[61]), "f"(sc));
                        converted[2] = _mul_62;
                        float _mul_63;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_63) : "f"(frag[63]), "f"(sc));
                        converted[3] = _mul_63;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_29 = token_28 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_28 * 64 + (base_feature ^ swizzle_feature_29)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_28 * 64 + (base_feature - 64 ^ swizzle_feature_29)) / 4] = packed_word;
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 7, 128;" ::: "memory");
                        if (warp == 0) {
                            if (elect_sync()) {
                                int padding_rows = (256 - valid_rows % 256) % 256;
                                tma_store_4d((&C), m_tile * 128, padding_rows + token_block * 64, 1073741824, n_tile * 256 - (unsigned int)padding_rows + 1073741824, epi_staging_addr);
                                tma_store_4d((&C), m_tile * 128 + 64, padding_rows + token_block * 64, 1073741824, n_tile * 256 - (unsigned int)padding_rows + 1073741824, epi_staging_addr + 8192);
                            }
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                        asm volatile("barrier.sync 7, 128;" ::: "memory");
                        int token_block_30 = 1;
                        if (epilogue_local_idx == 0) {
                            token_block_30 = 0;
                        }
                        int acc_col_31 = epilogue_local_idx * 192 + token_block_30 * 64;
                        float _tmem_load_2[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[31]))
                            : "r"(taddr + (unsigned int)row_addr + (unsigned int)acc_col_31));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        float _tmem_load_3[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[31]))
                            : "r"(taddr + (unsigned int)row_addr + 1048576 + (unsigned int)acc_col_31));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        for (int i_1 = 0; i_1 < 32; i_1++) {
                            frag[i_1] = _tmem_load_2[i_1];
                            frag[32 + i_1] = _tmem_load_3[i_1];
                        }
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                        asm volatile("barrier.sync 7, 128;" ::: "memory");
                        int token_32 = base_token;
                        float _mul_64;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_64) : "f"(frag[0]), "f"(sc));
                        converted[0] = _mul_64;
                        float _mul_65;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_65) : "f"(frag[2]), "f"(sc));
                        converted[1] = _mul_65;
                        float _mul_66;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_66) : "f"(frag[32]), "f"(sc));
                        converted[2] = _mul_66;
                        float _mul_67;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_67) : "f"(frag[34]), "f"(sc));
                        converted[3] = _mul_67;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_33 = token_32 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_32 * 64 + (base_feature ^ swizzle_feature_33)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_32 * 64 + (base_feature - 64 ^ swizzle_feature_33)) / 4] = packed_word;
                        }
                        int token_34 = base_token + 1;
                        float _mul_68;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_68) : "f"(frag[1]), "f"(sc));
                        converted[0] = _mul_68;
                        float _mul_69;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_69) : "f"(frag[3]), "f"(sc));
                        converted[1] = _mul_69;
                        float _mul_70;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_70) : "f"(frag[33]), "f"(sc));
                        converted[2] = _mul_70;
                        float _mul_71;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_71) : "f"(frag[35]), "f"(sc));
                        converted[3] = _mul_71;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_35 = token_34 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_34 * 64 + (base_feature ^ swizzle_feature_35)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_34 * 64 + (base_feature - 64 ^ swizzle_feature_35)) / 4] = packed_word;
                        }
                        int token_36 = base_token + 8;
                        float _mul_72;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_72) : "f"(frag[4]), "f"(sc));
                        converted[0] = _mul_72;
                        float _mul_73;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_73) : "f"(frag[6]), "f"(sc));
                        converted[1] = _mul_73;
                        float _mul_74;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_74) : "f"(frag[36]), "f"(sc));
                        converted[2] = _mul_74;
                        float _mul_75;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_75) : "f"(frag[38]), "f"(sc));
                        converted[3] = _mul_75;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_37 = token_36 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_36 * 64 + (base_feature ^ swizzle_feature_37)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_36 * 64 + (base_feature - 64 ^ swizzle_feature_37)) / 4] = packed_word;
                        }
                        int token_38 = base_token + 8 + 1;
                        float _mul_76;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_76) : "f"(frag[5]), "f"(sc));
                        converted[0] = _mul_76;
                        float _mul_77;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_77) : "f"(frag[7]), "f"(sc));
                        converted[1] = _mul_77;
                        float _mul_78;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_78) : "f"(frag[37]), "f"(sc));
                        converted[2] = _mul_78;
                        float _mul_79;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_79) : "f"(frag[39]), "f"(sc));
                        converted[3] = _mul_79;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_39 = token_38 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_38 * 64 + (base_feature ^ swizzle_feature_39)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_38 * 64 + (base_feature - 64 ^ swizzle_feature_39)) / 4] = packed_word;
                        }
                        int token_40 = base_token + 16;
                        float _mul_80;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_80) : "f"(frag[8]), "f"(sc));
                        converted[0] = _mul_80;
                        float _mul_81;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_81) : "f"(frag[10]), "f"(sc));
                        converted[1] = _mul_81;
                        float _mul_82;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_82) : "f"(frag[40]), "f"(sc));
                        converted[2] = _mul_82;
                        float _mul_83;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_83) : "f"(frag[42]), "f"(sc));
                        converted[3] = _mul_83;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_41 = token_40 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_40 * 64 + (base_feature ^ swizzle_feature_41)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_40 * 64 + (base_feature - 64 ^ swizzle_feature_41)) / 4] = packed_word;
                        }
                        int token_42 = base_token + 16 + 1;
                        float _mul_84;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_84) : "f"(frag[9]), "f"(sc));
                        converted[0] = _mul_84;
                        float _mul_85;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_85) : "f"(frag[11]), "f"(sc));
                        converted[1] = _mul_85;
                        float _mul_86;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_86) : "f"(frag[41]), "f"(sc));
                        converted[2] = _mul_86;
                        float _mul_87;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_87) : "f"(frag[43]), "f"(sc));
                        converted[3] = _mul_87;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_43 = token_42 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_42 * 64 + (base_feature ^ swizzle_feature_43)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_42 * 64 + (base_feature - 64 ^ swizzle_feature_43)) / 4] = packed_word;
                        }
                        int token_44 = base_token + 24;
                        float _mul_88;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_88) : "f"(frag[12]), "f"(sc));
                        converted[0] = _mul_88;
                        float _mul_89;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_89) : "f"(frag[14]), "f"(sc));
                        converted[1] = _mul_89;
                        float _mul_90;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_90) : "f"(frag[44]), "f"(sc));
                        converted[2] = _mul_90;
                        float _mul_91;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_91) : "f"(frag[46]), "f"(sc));
                        converted[3] = _mul_91;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_45 = token_44 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_44 * 64 + (base_feature ^ swizzle_feature_45)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_44 * 64 + (base_feature - 64 ^ swizzle_feature_45)) / 4] = packed_word;
                        }
                        int token_46 = base_token + 24 + 1;
                        float _mul_92;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_92) : "f"(frag[13]), "f"(sc));
                        converted[0] = _mul_92;
                        float _mul_93;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_93) : "f"(frag[15]), "f"(sc));
                        converted[1] = _mul_93;
                        float _mul_94;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_94) : "f"(frag[45]), "f"(sc));
                        converted[2] = _mul_94;
                        float _mul_95;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_95) : "f"(frag[47]), "f"(sc));
                        converted[3] = _mul_95;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_47 = token_46 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_46 * 64 + (base_feature ^ swizzle_feature_47)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_46 * 64 + (base_feature - 64 ^ swizzle_feature_47)) / 4] = packed_word;
                        }
                        int token_48 = base_token + 32;
                        float _mul_96;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_96) : "f"(frag[16]), "f"(sc));
                        converted[0] = _mul_96;
                        float _mul_97;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_97) : "f"(frag[18]), "f"(sc));
                        converted[1] = _mul_97;
                        float _mul_98;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_98) : "f"(frag[48]), "f"(sc));
                        converted[2] = _mul_98;
                        float _mul_99;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_99) : "f"(frag[50]), "f"(sc));
                        converted[3] = _mul_99;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_49 = token_48 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_48 * 64 + (base_feature ^ swizzle_feature_49)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_48 * 64 + (base_feature - 64 ^ swizzle_feature_49)) / 4] = packed_word;
                        }
                        int token_50 = base_token + 32 + 1;
                        float _mul_100;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_100) : "f"(frag[17]), "f"(sc));
                        converted[0] = _mul_100;
                        float _mul_101;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_101) : "f"(frag[19]), "f"(sc));
                        converted[1] = _mul_101;
                        float _mul_102;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_102) : "f"(frag[49]), "f"(sc));
                        converted[2] = _mul_102;
                        float _mul_103;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_103) : "f"(frag[51]), "f"(sc));
                        converted[3] = _mul_103;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_51 = token_50 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_50 * 64 + (base_feature ^ swizzle_feature_51)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_50 * 64 + (base_feature - 64 ^ swizzle_feature_51)) / 4] = packed_word;
                        }
                        int token_52 = base_token + 40;
                        float _mul_104;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_104) : "f"(frag[20]), "f"(sc));
                        converted[0] = _mul_104;
                        float _mul_105;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_105) : "f"(frag[22]), "f"(sc));
                        converted[1] = _mul_105;
                        float _mul_106;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_106) : "f"(frag[52]), "f"(sc));
                        converted[2] = _mul_106;
                        float _mul_107;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_107) : "f"(frag[54]), "f"(sc));
                        converted[3] = _mul_107;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_53 = token_52 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_52 * 64 + (base_feature ^ swizzle_feature_53)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_52 * 64 + (base_feature - 64 ^ swizzle_feature_53)) / 4] = packed_word;
                        }
                        int token_54 = base_token + 40 + 1;
                        float _mul_108;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_108) : "f"(frag[21]), "f"(sc));
                        converted[0] = _mul_108;
                        float _mul_109;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_109) : "f"(frag[23]), "f"(sc));
                        converted[1] = _mul_109;
                        float _mul_110;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_110) : "f"(frag[53]), "f"(sc));
                        converted[2] = _mul_110;
                        float _mul_111;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_111) : "f"(frag[55]), "f"(sc));
                        converted[3] = _mul_111;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_55 = token_54 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_54 * 64 + (base_feature ^ swizzle_feature_55)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_54 * 64 + (base_feature - 64 ^ swizzle_feature_55)) / 4] = packed_word;
                        }
                        int token_56 = base_token + 48;
                        float _mul_112;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_112) : "f"(frag[24]), "f"(sc));
                        converted[0] = _mul_112;
                        float _mul_113;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_113) : "f"(frag[26]), "f"(sc));
                        converted[1] = _mul_113;
                        float _mul_114;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_114) : "f"(frag[56]), "f"(sc));
                        converted[2] = _mul_114;
                        float _mul_115;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_115) : "f"(frag[58]), "f"(sc));
                        converted[3] = _mul_115;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_57 = token_56 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_56 * 64 + (base_feature ^ swizzle_feature_57)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_56 * 64 + (base_feature - 64 ^ swizzle_feature_57)) / 4] = packed_word;
                        }
                        int token_58 = base_token + 48 + 1;
                        float _mul_116;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_116) : "f"(frag[25]), "f"(sc));
                        converted[0] = _mul_116;
                        float _mul_117;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_117) : "f"(frag[27]), "f"(sc));
                        converted[1] = _mul_117;
                        float _mul_118;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_118) : "f"(frag[57]), "f"(sc));
                        converted[2] = _mul_118;
                        float _mul_119;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_119) : "f"(frag[59]), "f"(sc));
                        converted[3] = _mul_119;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_59 = token_58 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_58 * 64 + (base_feature ^ swizzle_feature_59)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_58 * 64 + (base_feature - 64 ^ swizzle_feature_59)) / 4] = packed_word;
                        }
                        int token_60 = base_token + 56;
                        float _mul_120;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_120) : "f"(frag[28]), "f"(sc));
                        converted[0] = _mul_120;
                        float _mul_121;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_121) : "f"(frag[30]), "f"(sc));
                        converted[1] = _mul_121;
                        float _mul_122;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_122) : "f"(frag[60]), "f"(sc));
                        converted[2] = _mul_122;
                        float _mul_123;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_123) : "f"(frag[62]), "f"(sc));
                        converted[3] = _mul_123;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_61 = token_60 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_60 * 64 + (base_feature ^ swizzle_feature_61)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_60 * 64 + (base_feature - 64 ^ swizzle_feature_61)) / 4] = packed_word;
                        }
                        int token_62 = base_token + 56 + 1;
                        float _mul_124;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_124) : "f"(frag[29]), "f"(sc));
                        converted[0] = _mul_124;
                        float _mul_125;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_125) : "f"(frag[31]), "f"(sc));
                        converted[1] = _mul_125;
                        float _mul_126;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_126) : "f"(frag[61]), "f"(sc));
                        converted[2] = _mul_126;
                        float _mul_127;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_127) : "f"(frag[63]), "f"(sc));
                        converted[3] = _mul_127;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_63 = token_62 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_62 * 64 + (base_feature ^ swizzle_feature_63)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_62 * 64 + (base_feature - 64 ^ swizzle_feature_63)) / 4] = packed_word;
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 7, 128;" ::: "memory");
                        if (warp == 0) {
                            if (elect_sync()) {
                                int padding_rows_1 = (256 - valid_rows % 256) % 256;
                                tma_store_4d((&C), m_tile * 128, padding_rows_1 + token_block_30 * 64, 1073741824, n_tile * 256 - (unsigned int)padding_rows_1 + 1073741824, epi_staging_addr);
                                tma_store_4d((&C), m_tile * 128 + 64, padding_rows_1 + token_block_30 * 64, 1073741824, n_tile * 256 - (unsigned int)padding_rows_1 + 1073741824, epi_staging_addr + 8192);
                            }
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                        asm volatile("barrier.sync 7, 128;" ::: "memory");
                        int token_block_64 = 2;
                        if (epilogue_local_idx == 0) {
                            token_block_64 = 1;
                        }
                        int acc_col_65 = epilogue_local_idx * 192 + token_block_64 * 64;
                        float _tmem_load_4[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[31]))
                            : "r"(taddr + (unsigned int)row_addr + (unsigned int)acc_col_65));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        float _tmem_load_5[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[31]))
                            : "r"(taddr + (unsigned int)row_addr + 1048576 + (unsigned int)acc_col_65));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        for (int i_2 = 0; i_2 < 32; i_2++) {
                            frag[i_2] = _tmem_load_4[i_2];
                            frag[32 + i_2] = _tmem_load_5[i_2];
                        }
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                        asm volatile("barrier.sync 7, 128;" ::: "memory");
                        int token_66 = base_token;
                        float _mul_128;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_128) : "f"(frag[0]), "f"(sc));
                        converted[0] = _mul_128;
                        float _mul_129;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_129) : "f"(frag[2]), "f"(sc));
                        converted[1] = _mul_129;
                        float _mul_130;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_130) : "f"(frag[32]), "f"(sc));
                        converted[2] = _mul_130;
                        float _mul_131;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_131) : "f"(frag[34]), "f"(sc));
                        converted[3] = _mul_131;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_67 = token_66 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_66 * 64 + (base_feature ^ swizzle_feature_67)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_66 * 64 + (base_feature - 64 ^ swizzle_feature_67)) / 4] = packed_word;
                        }
                        int token_68 = base_token + 1;
                        float _mul_132;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_132) : "f"(frag[1]), "f"(sc));
                        converted[0] = _mul_132;
                        float _mul_133;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_133) : "f"(frag[3]), "f"(sc));
                        converted[1] = _mul_133;
                        float _mul_134;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_134) : "f"(frag[33]), "f"(sc));
                        converted[2] = _mul_134;
                        float _mul_135;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_135) : "f"(frag[35]), "f"(sc));
                        converted[3] = _mul_135;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_69 = token_68 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_68 * 64 + (base_feature ^ swizzle_feature_69)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_68 * 64 + (base_feature - 64 ^ swizzle_feature_69)) / 4] = packed_word;
                        }
                        int token_70 = base_token + 8;
                        float _mul_136;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_136) : "f"(frag[4]), "f"(sc));
                        converted[0] = _mul_136;
                        float _mul_137;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_137) : "f"(frag[6]), "f"(sc));
                        converted[1] = _mul_137;
                        float _mul_138;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_138) : "f"(frag[36]), "f"(sc));
                        converted[2] = _mul_138;
                        float _mul_139;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_139) : "f"(frag[38]), "f"(sc));
                        converted[3] = _mul_139;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_71 = token_70 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_70 * 64 + (base_feature ^ swizzle_feature_71)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_70 * 64 + (base_feature - 64 ^ swizzle_feature_71)) / 4] = packed_word;
                        }
                        int token_72 = base_token + 8 + 1;
                        float _mul_140;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_140) : "f"(frag[5]), "f"(sc));
                        converted[0] = _mul_140;
                        float _mul_141;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_141) : "f"(frag[7]), "f"(sc));
                        converted[1] = _mul_141;
                        float _mul_142;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_142) : "f"(frag[37]), "f"(sc));
                        converted[2] = _mul_142;
                        float _mul_143;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_143) : "f"(frag[39]), "f"(sc));
                        converted[3] = _mul_143;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_73 = token_72 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_72 * 64 + (base_feature ^ swizzle_feature_73)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_72 * 64 + (base_feature - 64 ^ swizzle_feature_73)) / 4] = packed_word;
                        }
                        int token_74 = base_token + 16;
                        float _mul_144;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_144) : "f"(frag[8]), "f"(sc));
                        converted[0] = _mul_144;
                        float _mul_145;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_145) : "f"(frag[10]), "f"(sc));
                        converted[1] = _mul_145;
                        float _mul_146;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_146) : "f"(frag[40]), "f"(sc));
                        converted[2] = _mul_146;
                        float _mul_147;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_147) : "f"(frag[42]), "f"(sc));
                        converted[3] = _mul_147;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_75 = token_74 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_74 * 64 + (base_feature ^ swizzle_feature_75)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_74 * 64 + (base_feature - 64 ^ swizzle_feature_75)) / 4] = packed_word;
                        }
                        int token_76 = base_token + 16 + 1;
                        float _mul_148;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_148) : "f"(frag[9]), "f"(sc));
                        converted[0] = _mul_148;
                        float _mul_149;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_149) : "f"(frag[11]), "f"(sc));
                        converted[1] = _mul_149;
                        float _mul_150;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_150) : "f"(frag[41]), "f"(sc));
                        converted[2] = _mul_150;
                        float _mul_151;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_151) : "f"(frag[43]), "f"(sc));
                        converted[3] = _mul_151;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_77 = token_76 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_76 * 64 + (base_feature ^ swizzle_feature_77)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_76 * 64 + (base_feature - 64 ^ swizzle_feature_77)) / 4] = packed_word;
                        }
                        int token_78 = base_token + 24;
                        float _mul_152;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_152) : "f"(frag[12]), "f"(sc));
                        converted[0] = _mul_152;
                        float _mul_153;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_153) : "f"(frag[14]), "f"(sc));
                        converted[1] = _mul_153;
                        float _mul_154;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_154) : "f"(frag[44]), "f"(sc));
                        converted[2] = _mul_154;
                        float _mul_155;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_155) : "f"(frag[46]), "f"(sc));
                        converted[3] = _mul_155;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_79 = token_78 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_78 * 64 + (base_feature ^ swizzle_feature_79)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_78 * 64 + (base_feature - 64 ^ swizzle_feature_79)) / 4] = packed_word;
                        }
                        int token_80 = base_token + 24 + 1;
                        float _mul_156;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_156) : "f"(frag[13]), "f"(sc));
                        converted[0] = _mul_156;
                        float _mul_157;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_157) : "f"(frag[15]), "f"(sc));
                        converted[1] = _mul_157;
                        float _mul_158;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_158) : "f"(frag[45]), "f"(sc));
                        converted[2] = _mul_158;
                        float _mul_159;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_159) : "f"(frag[47]), "f"(sc));
                        converted[3] = _mul_159;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_81 = token_80 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_80 * 64 + (base_feature ^ swizzle_feature_81)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_80 * 64 + (base_feature - 64 ^ swizzle_feature_81)) / 4] = packed_word;
                        }
                        int token_82 = base_token + 32;
                        float _mul_160;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_160) : "f"(frag[16]), "f"(sc));
                        converted[0] = _mul_160;
                        float _mul_161;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_161) : "f"(frag[18]), "f"(sc));
                        converted[1] = _mul_161;
                        float _mul_162;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_162) : "f"(frag[48]), "f"(sc));
                        converted[2] = _mul_162;
                        float _mul_163;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_163) : "f"(frag[50]), "f"(sc));
                        converted[3] = _mul_163;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_83 = token_82 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_82 * 64 + (base_feature ^ swizzle_feature_83)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_82 * 64 + (base_feature - 64 ^ swizzle_feature_83)) / 4] = packed_word;
                        }
                        int token_84 = base_token + 32 + 1;
                        float _mul_164;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_164) : "f"(frag[17]), "f"(sc));
                        converted[0] = _mul_164;
                        float _mul_165;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_165) : "f"(frag[19]), "f"(sc));
                        converted[1] = _mul_165;
                        float _mul_166;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_166) : "f"(frag[49]), "f"(sc));
                        converted[2] = _mul_166;
                        float _mul_167;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_167) : "f"(frag[51]), "f"(sc));
                        converted[3] = _mul_167;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_85 = token_84 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_84 * 64 + (base_feature ^ swizzle_feature_85)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_84 * 64 + (base_feature - 64 ^ swizzle_feature_85)) / 4] = packed_word;
                        }
                        int token_86 = base_token + 40;
                        float _mul_168;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_168) : "f"(frag[20]), "f"(sc));
                        converted[0] = _mul_168;
                        float _mul_169;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_169) : "f"(frag[22]), "f"(sc));
                        converted[1] = _mul_169;
                        float _mul_170;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_170) : "f"(frag[52]), "f"(sc));
                        converted[2] = _mul_170;
                        float _mul_171;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_171) : "f"(frag[54]), "f"(sc));
                        converted[3] = _mul_171;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_87 = token_86 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_86 * 64 + (base_feature ^ swizzle_feature_87)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_86 * 64 + (base_feature - 64 ^ swizzle_feature_87)) / 4] = packed_word;
                        }
                        int token_88 = base_token + 40 + 1;
                        float _mul_172;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_172) : "f"(frag[21]), "f"(sc));
                        converted[0] = _mul_172;
                        float _mul_173;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_173) : "f"(frag[23]), "f"(sc));
                        converted[1] = _mul_173;
                        float _mul_174;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_174) : "f"(frag[53]), "f"(sc));
                        converted[2] = _mul_174;
                        float _mul_175;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_175) : "f"(frag[55]), "f"(sc));
                        converted[3] = _mul_175;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_89 = token_88 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_88 * 64 + (base_feature ^ swizzle_feature_89)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_88 * 64 + (base_feature - 64 ^ swizzle_feature_89)) / 4] = packed_word;
                        }
                        int token_90 = base_token + 48;
                        float _mul_176;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_176) : "f"(frag[24]), "f"(sc));
                        converted[0] = _mul_176;
                        float _mul_177;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_177) : "f"(frag[26]), "f"(sc));
                        converted[1] = _mul_177;
                        float _mul_178;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_178) : "f"(frag[56]), "f"(sc));
                        converted[2] = _mul_178;
                        float _mul_179;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_179) : "f"(frag[58]), "f"(sc));
                        converted[3] = _mul_179;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_91 = token_90 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_90 * 64 + (base_feature ^ swizzle_feature_91)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_90 * 64 + (base_feature - 64 ^ swizzle_feature_91)) / 4] = packed_word;
                        }
                        int token_92 = base_token + 48 + 1;
                        float _mul_180;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_180) : "f"(frag[25]), "f"(sc));
                        converted[0] = _mul_180;
                        float _mul_181;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_181) : "f"(frag[27]), "f"(sc));
                        converted[1] = _mul_181;
                        float _mul_182;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_182) : "f"(frag[57]), "f"(sc));
                        converted[2] = _mul_182;
                        float _mul_183;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_183) : "f"(frag[59]), "f"(sc));
                        converted[3] = _mul_183;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_93 = token_92 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_92 * 64 + (base_feature ^ swizzle_feature_93)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_92 * 64 + (base_feature - 64 ^ swizzle_feature_93)) / 4] = packed_word;
                        }
                        int token_94 = base_token + 56;
                        float _mul_184;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_184) : "f"(frag[28]), "f"(sc));
                        converted[0] = _mul_184;
                        float _mul_185;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_185) : "f"(frag[30]), "f"(sc));
                        converted[1] = _mul_185;
                        float _mul_186;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_186) : "f"(frag[60]), "f"(sc));
                        converted[2] = _mul_186;
                        float _mul_187;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_187) : "f"(frag[62]), "f"(sc));
                        converted[3] = _mul_187;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_95 = token_94 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_94 * 64 + (base_feature ^ swizzle_feature_95)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_94 * 64 + (base_feature - 64 ^ swizzle_feature_95)) / 4] = packed_word;
                        }
                        int token_96 = base_token + 56 + 1;
                        float _mul_188;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_188) : "f"(frag[29]), "f"(sc));
                        converted[0] = _mul_188;
                        float _mul_189;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_189) : "f"(frag[31]), "f"(sc));
                        converted[1] = _mul_189;
                        float _mul_190;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_190) : "f"(frag[61]), "f"(sc));
                        converted[2] = _mul_190;
                        float _mul_191;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_191) : "f"(frag[63]), "f"(sc));
                        converted[3] = _mul_191;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_97 = token_96 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_96 * 64 + (base_feature ^ swizzle_feature_97)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_96 * 64 + (base_feature - 64 ^ swizzle_feature_97)) / 4] = packed_word;
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 7, 128;" ::: "memory");
                        if (warp == 0) {
                            if (elect_sync()) {
                                int padding_rows_2 = (256 - valid_rows % 256) % 256;
                                tma_store_4d((&C), m_tile * 128, padding_rows_2 + token_block_64 * 64, 1073741824, n_tile * 256 - (unsigned int)padding_rows_2 + 1073741824, epi_staging_addr);
                                tma_store_4d((&C), m_tile * 128 + 64, padding_rows_2 + token_block_64 * 64, 1073741824, n_tile * 256 - (unsigned int)padding_rows_2 + 1073741824, epi_staging_addr + 8192);
                            }
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                        asm volatile("barrier.sync 7, 128;" ::: "memory");
                        int token_block_98 = 3;
                        if (epilogue_local_idx == 0) {
                            token_block_98 = 2;
                        }
                        int acc_col_99 = epilogue_local_idx * 192 + token_block_98 * 64;
                        float _tmem_load_6[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[31]))
                            : "r"(taddr + (unsigned int)row_addr + (unsigned int)acc_col_99));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        float _tmem_load_7[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[31]))
                            : "r"(taddr + (unsigned int)row_addr + 1048576 + (unsigned int)acc_col_99));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        for (int i_3 = 0; i_3 < 32; i_3++) {
                            frag[i_3] = _tmem_load_6[i_3];
                            frag[32 + i_3] = _tmem_load_7[i_3];
                        }
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                        asm volatile("barrier.sync 7, 128;" ::: "memory");
                        int token_100 = base_token;
                        float _mul_192;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_192) : "f"(frag[0]), "f"(sc));
                        converted[0] = _mul_192;
                        float _mul_193;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_193) : "f"(frag[2]), "f"(sc));
                        converted[1] = _mul_193;
                        float _mul_194;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_194) : "f"(frag[32]), "f"(sc));
                        converted[2] = _mul_194;
                        float _mul_195;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_195) : "f"(frag[34]), "f"(sc));
                        converted[3] = _mul_195;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_101 = token_100 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_100 * 64 + (base_feature ^ swizzle_feature_101)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_100 * 64 + (base_feature - 64 ^ swizzle_feature_101)) / 4] = packed_word;
                        }
                        int token_102 = base_token + 1;
                        float _mul_196;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_196) : "f"(frag[1]), "f"(sc));
                        converted[0] = _mul_196;
                        float _mul_197;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_197) : "f"(frag[3]), "f"(sc));
                        converted[1] = _mul_197;
                        float _mul_198;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_198) : "f"(frag[33]), "f"(sc));
                        converted[2] = _mul_198;
                        float _mul_199;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_199) : "f"(frag[35]), "f"(sc));
                        converted[3] = _mul_199;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_103 = token_102 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_102 * 64 + (base_feature ^ swizzle_feature_103)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_102 * 64 + (base_feature - 64 ^ swizzle_feature_103)) / 4] = packed_word;
                        }
                        int token_104 = base_token + 8;
                        float _mul_200;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_200) : "f"(frag[4]), "f"(sc));
                        converted[0] = _mul_200;
                        float _mul_201;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_201) : "f"(frag[6]), "f"(sc));
                        converted[1] = _mul_201;
                        float _mul_202;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_202) : "f"(frag[36]), "f"(sc));
                        converted[2] = _mul_202;
                        float _mul_203;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_203) : "f"(frag[38]), "f"(sc));
                        converted[3] = _mul_203;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_105 = token_104 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_104 * 64 + (base_feature ^ swizzle_feature_105)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_104 * 64 + (base_feature - 64 ^ swizzle_feature_105)) / 4] = packed_word;
                        }
                        int token_106 = base_token + 8 + 1;
                        float _mul_204;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_204) : "f"(frag[5]), "f"(sc));
                        converted[0] = _mul_204;
                        float _mul_205;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_205) : "f"(frag[7]), "f"(sc));
                        converted[1] = _mul_205;
                        float _mul_206;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_206) : "f"(frag[37]), "f"(sc));
                        converted[2] = _mul_206;
                        float _mul_207;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_207) : "f"(frag[39]), "f"(sc));
                        converted[3] = _mul_207;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_107 = token_106 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_106 * 64 + (base_feature ^ swizzle_feature_107)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_106 * 64 + (base_feature - 64 ^ swizzle_feature_107)) / 4] = packed_word;
                        }
                        int token_108 = base_token + 16;
                        float _mul_208;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_208) : "f"(frag[8]), "f"(sc));
                        converted[0] = _mul_208;
                        float _mul_209;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_209) : "f"(frag[10]), "f"(sc));
                        converted[1] = _mul_209;
                        float _mul_210;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_210) : "f"(frag[40]), "f"(sc));
                        converted[2] = _mul_210;
                        float _mul_211;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_211) : "f"(frag[42]), "f"(sc));
                        converted[3] = _mul_211;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_109 = token_108 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_108 * 64 + (base_feature ^ swizzle_feature_109)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_108 * 64 + (base_feature - 64 ^ swizzle_feature_109)) / 4] = packed_word;
                        }
                        int token_110 = base_token + 16 + 1;
                        float _mul_212;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_212) : "f"(frag[9]), "f"(sc));
                        converted[0] = _mul_212;
                        float _mul_213;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_213) : "f"(frag[11]), "f"(sc));
                        converted[1] = _mul_213;
                        float _mul_214;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_214) : "f"(frag[41]), "f"(sc));
                        converted[2] = _mul_214;
                        float _mul_215;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_215) : "f"(frag[43]), "f"(sc));
                        converted[3] = _mul_215;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_111 = token_110 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_110 * 64 + (base_feature ^ swizzle_feature_111)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_110 * 64 + (base_feature - 64 ^ swizzle_feature_111)) / 4] = packed_word;
                        }
                        int token_112 = base_token + 24;
                        float _mul_216;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_216) : "f"(frag[12]), "f"(sc));
                        converted[0] = _mul_216;
                        float _mul_217;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_217) : "f"(frag[14]), "f"(sc));
                        converted[1] = _mul_217;
                        float _mul_218;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_218) : "f"(frag[44]), "f"(sc));
                        converted[2] = _mul_218;
                        float _mul_219;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_219) : "f"(frag[46]), "f"(sc));
                        converted[3] = _mul_219;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_113 = token_112 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_112 * 64 + (base_feature ^ swizzle_feature_113)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_112 * 64 + (base_feature - 64 ^ swizzle_feature_113)) / 4] = packed_word;
                        }
                        int token_114 = base_token + 24 + 1;
                        float _mul_220;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_220) : "f"(frag[13]), "f"(sc));
                        converted[0] = _mul_220;
                        float _mul_221;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_221) : "f"(frag[15]), "f"(sc));
                        converted[1] = _mul_221;
                        float _mul_222;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_222) : "f"(frag[45]), "f"(sc));
                        converted[2] = _mul_222;
                        float _mul_223;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_223) : "f"(frag[47]), "f"(sc));
                        converted[3] = _mul_223;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_115 = token_114 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_114 * 64 + (base_feature ^ swizzle_feature_115)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_114 * 64 + (base_feature - 64 ^ swizzle_feature_115)) / 4] = packed_word;
                        }
                        int token_116 = base_token + 32;
                        float _mul_224;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_224) : "f"(frag[16]), "f"(sc));
                        converted[0] = _mul_224;
                        float _mul_225;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_225) : "f"(frag[18]), "f"(sc));
                        converted[1] = _mul_225;
                        float _mul_226;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_226) : "f"(frag[48]), "f"(sc));
                        converted[2] = _mul_226;
                        float _mul_227;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_227) : "f"(frag[50]), "f"(sc));
                        converted[3] = _mul_227;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_117 = token_116 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_116 * 64 + (base_feature ^ swizzle_feature_117)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_116 * 64 + (base_feature - 64 ^ swizzle_feature_117)) / 4] = packed_word;
                        }
                        int token_118 = base_token + 32 + 1;
                        float _mul_228;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_228) : "f"(frag[17]), "f"(sc));
                        converted[0] = _mul_228;
                        float _mul_229;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_229) : "f"(frag[19]), "f"(sc));
                        converted[1] = _mul_229;
                        float _mul_230;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_230) : "f"(frag[49]), "f"(sc));
                        converted[2] = _mul_230;
                        float _mul_231;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_231) : "f"(frag[51]), "f"(sc));
                        converted[3] = _mul_231;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_119 = token_118 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_118 * 64 + (base_feature ^ swizzle_feature_119)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_118 * 64 + (base_feature - 64 ^ swizzle_feature_119)) / 4] = packed_word;
                        }
                        int token_120 = base_token + 40;
                        float _mul_232;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_232) : "f"(frag[20]), "f"(sc));
                        converted[0] = _mul_232;
                        float _mul_233;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_233) : "f"(frag[22]), "f"(sc));
                        converted[1] = _mul_233;
                        float _mul_234;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_234) : "f"(frag[52]), "f"(sc));
                        converted[2] = _mul_234;
                        float _mul_235;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_235) : "f"(frag[54]), "f"(sc));
                        converted[3] = _mul_235;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_121 = token_120 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_120 * 64 + (base_feature ^ swizzle_feature_121)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_120 * 64 + (base_feature - 64 ^ swizzle_feature_121)) / 4] = packed_word;
                        }
                        int token_122 = base_token + 40 + 1;
                        float _mul_236;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_236) : "f"(frag[21]), "f"(sc));
                        converted[0] = _mul_236;
                        float _mul_237;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_237) : "f"(frag[23]), "f"(sc));
                        converted[1] = _mul_237;
                        float _mul_238;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_238) : "f"(frag[53]), "f"(sc));
                        converted[2] = _mul_238;
                        float _mul_239;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_239) : "f"(frag[55]), "f"(sc));
                        converted[3] = _mul_239;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_123 = token_122 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_122 * 64 + (base_feature ^ swizzle_feature_123)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_122 * 64 + (base_feature - 64 ^ swizzle_feature_123)) / 4] = packed_word;
                        }
                        int token_124 = base_token + 48;
                        float _mul_240;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_240) : "f"(frag[24]), "f"(sc));
                        converted[0] = _mul_240;
                        float _mul_241;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_241) : "f"(frag[26]), "f"(sc));
                        converted[1] = _mul_241;
                        float _mul_242;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_242) : "f"(frag[56]), "f"(sc));
                        converted[2] = _mul_242;
                        float _mul_243;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_243) : "f"(frag[58]), "f"(sc));
                        converted[3] = _mul_243;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_125 = token_124 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_124 * 64 + (base_feature ^ swizzle_feature_125)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_124 * 64 + (base_feature - 64 ^ swizzle_feature_125)) / 4] = packed_word;
                        }
                        int token_126 = base_token + 48 + 1;
                        float _mul_244;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_244) : "f"(frag[25]), "f"(sc));
                        converted[0] = _mul_244;
                        float _mul_245;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_245) : "f"(frag[27]), "f"(sc));
                        converted[1] = _mul_245;
                        float _mul_246;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_246) : "f"(frag[57]), "f"(sc));
                        converted[2] = _mul_246;
                        float _mul_247;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_247) : "f"(frag[59]), "f"(sc));
                        converted[3] = _mul_247;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_127 = token_126 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_126 * 64 + (base_feature ^ swizzle_feature_127)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_126 * 64 + (base_feature - 64 ^ swizzle_feature_127)) / 4] = packed_word;
                        }
                        int token_128 = base_token + 56;
                        float _mul_248;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_248) : "f"(frag[28]), "f"(sc));
                        converted[0] = _mul_248;
                        float _mul_249;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_249) : "f"(frag[30]), "f"(sc));
                        converted[1] = _mul_249;
                        float _mul_250;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_250) : "f"(frag[60]), "f"(sc));
                        converted[2] = _mul_250;
                        float _mul_251;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_251) : "f"(frag[62]), "f"(sc));
                        converted[3] = _mul_251;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_129 = token_128 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_128 * 64 + (base_feature ^ swizzle_feature_129)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_128 * 64 + (base_feature - 64 ^ swizzle_feature_129)) / 4] = packed_word;
                        }
                        int token_130 = base_token + 56 + 1;
                        float _mul_252;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_252) : "f"(frag[29]), "f"(sc));
                        converted[0] = _mul_252;
                        float _mul_253;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_253) : "f"(frag[31]), "f"(sc));
                        converted[1] = _mul_253;
                        float _mul_254;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_254) : "f"(frag[61]), "f"(sc));
                        converted[2] = _mul_254;
                        float _mul_255;
                        asm("mul.rn.f32 %0, %1, %2;" : "=f"(_mul_255) : "f"(frag[63]), "f"(sc));
                        converted[3] = _mul_255;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_131 = token_130 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_130 * 64 + (base_feature ^ swizzle_feature_131)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_130 * 64 + (base_feature - 64 ^ swizzle_feature_131)) / 4] = packed_word;
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 7, 128;" ::: "memory");
                        if (warp == 0) {
                            if (elect_sync()) {
                                int padding_rows_3 = (256 - valid_rows % 256) % 256;
                                tma_store_4d((&C), m_tile * 128, padding_rows_3 + token_block_98 * 64, 1073741824, n_tile * 256 - (unsigned int)padding_rows_3 + 1073741824, epi_staging_addr);
                                tma_store_4d((&C), m_tile * 128 + 64, padding_rows_3 + token_block_98 * 64, 1073741824, n_tile * 256 - (unsigned int)padding_rows_3 + 1073741824, epi_staging_addr + 8192);
                            }
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                        asm volatile("barrier.sync 7, 128;" ::: "memory");
                        epilogue_local_idx = epilogue_local_idx ^ 1;
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage) * 8, _phase_work_full);
                unsigned int valid = 0;
                unsigned int next_x = 0;
                unsigned int next_y = 0;
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
                    : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                    : "memory");
                valid = _clc_valid_5;
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
                    : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                    : "memory");
                next_x = _clc_ctaid_10;
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
                    : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                    : "memory");
                next_y = _clc_ctaid_11;
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
                unsigned int valid_0 = valid;
                m_tile = next_x + (unsigned int)cta_rank;
                n_tile = next_y;
                if (valid_0 == 0) {
                    break;
                }
            }
            asm volatile("barrier.sync 10, 128;" ::: "memory");
            if (warp_0 == 0) {
                if (cta_rank == 0) {
                    mbarrier_wait(exit_bar_addr, 0);
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(exit_bar_addr + 0 * 8), "r"(1) : "memory");
                } else {
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(exit_bar_addr + 0 * 8), "r"(0) : "memory");
                    mbarrier_wait(exit_bar_addr, 0);
                }
                int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
                asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(512));
            }
        }
    }
    // ---- Role: load_b ----
    if (warp == 4) {
        { // load_b_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            int tile_count_1 = num_non_exiting_ctas[0];
            unsigned int stage = 0;
            unsigned int work_stage_1 = 0;
            unsigned int m_tile_1 = blockIdx.x;
            unsigned int n_tile_1 = blockIdx.y;
            unsigned int cta_mask = 1 << cta_rank;
            unsigned int _phase_b_empty = 1;
            unsigned int _phase_work_full_1 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_1 = 0; _tile_iter_1 < grid_m / 2 * grid_n; _tile_iter_1++) {
                int valid_rows_1 = (unsigned int)tile_mn_limit[n_tile_1] - n_tile_1 * 256;
                if (tile_count_1 > (int)n_tile_1) {
                    if (valid_rows_1 > 0) {
                        int padding_rows_4 = (256 - valid_rows_1 % 256) % 256;
                        int box_row0 = cta_rank * 128 + padding_rows_4;
                        #pragma unroll 1
                        for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                            mbarrier_wait(b_empty_addr + (stage) * 8, _phase_b_empty);
                            if (elect_sync()) {
                                if (cta_rank == 0) {
                                    asm volatile(
                                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                        :: "r"((b_full_addr + (stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                                }
                                asm volatile(
                                    "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                    " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                                    :: "r"(smem_b_addr + stage * 16384), "l"((&B)), "r"(iter_k * 128), "r"(box_row0), "r"(1073741824), "r"(n_tile_1 * 256 - (unsigned int)padding_rows_4 + 1073741824),
                                       "r"(((b_full_addr + (stage) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(cta_mask)) : "memory");
                            }
                            stage += 1;
                            if (stage == 5) { stage = 0; _phase_b_empty ^= 1; }
                        }
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_1) * 8, _phase_work_full_1);
                unsigned int valid_1 = 0;
                unsigned int next_x_1 = 0;
                unsigned int next_y_1 = 0;
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
                valid_1 = _clc_valid_1;
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
                next_x_1 = _clc_ctaid_2;
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
                next_y_1 = _clc_ctaid_3;
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
                unsigned int valid_0_1 = valid_1;
                m_tile_1 = next_x_1 + (unsigned int)cta_rank;
                n_tile_1 = next_y_1;
                if (valid_0_1 == 0) {
                    break;
                }
            }
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
    }
    // ---- Role: load_sfb ----
    if (warp == 5) {
        { // load_sfb_main
            unsigned int _phase_sfb_empty = 1;
            unsigned int _phase_work_full_2 = 0;
            if (cta_rank == 0) {
                asm volatile("griddepcontrol.wait;" ::: "memory");
                int tile_count_2 = num_non_exiting_ctas[0];
                unsigned int stage_1 = 0;
                unsigned int work_stage_2 = 0;
                unsigned int m_tile_2 = blockIdx.x;
                unsigned int n_tile_2 = blockIdx.y;
                #pragma unroll 1
                for (unsigned int _tile_iter_2 = 0; _tile_iter_2 < grid_m / 2 * grid_n; _tile_iter_2++) {
                    if (tile_count_2 > (int)n_tile_2) {
                        int valid_rows_sfb = (unsigned int)tile_mn_limit[n_tile_2] - n_tile_2 * 256;
                        if (valid_rows_sfb > 0) {
                            #pragma unroll 1
                            for (int iter_k_1 = 0; iter_k_1 < K_tiles; iter_k_1++) {
                                mbarrier_wait(sfb_empty_addr + (stage_1) * 8, _phase_sfb_empty);
                                if (elect_sync()) {
                                    asm volatile(
                                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                        :: "r"((sfb_full_addr + (stage_1) * 8) & 0xFEFFFFFF), "r"((uint32_t)(8192)) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                                        :: "r"(smem_sfb_addr + stage_1 * 4096), "l"((&SFB)), "r"(0), "r"(0), "r"(iter_k_1 * 4), "r"(n_tile_2 * 2),
                                           "r"(((sfb_full_addr + (stage_1) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)) : "memory");
                                }
                                stage_1 += 1;
                                if (stage_1 == 5) { stage_1 = 0; _phase_sfb_empty ^= 1; }
                            }
                        }
                    }
                    mbarrier_wait(work_full_addr + (work_stage_2) * 8, _phase_work_full_2);
                    unsigned int valid_2 = 0;
                    unsigned int next_x_2 = 0;
                    unsigned int next_y_2 = 0;
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
                        : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                        : "memory");
                    valid_2 = _clc_valid_2;
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
                        : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                        : "memory");
                    next_x_2 = _clc_ctaid_4;
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
                        : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                        : "memory");
                    next_y_2 = _clc_ctaid_5;
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
                    unsigned int valid_0_2 = valid_2;
                    m_tile_2 = next_x_2 + (unsigned int)cta_rank;
                    n_tile_2 = next_y_2;
                    if (valid_0_2 == 0) {
                        break;
                    }
                }
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
        }
    }
    // ---- Role: load_a ----
    if (warp == 6) {
        { // load_a_main
            int tile_count_3 = num_non_exiting_ctas[0];
            unsigned int stage_2 = 0;
            unsigned int work_stage_3 = 0;
            unsigned int throttle_stage = 0;
            unsigned int m_tile_3 = blockIdx.x;
            unsigned int n_tile_3 = blockIdx.y;
            unsigned int cta_mask_1 = 1 << cta_rank;
            unsigned int _phase_throttle_empty = 1;
            unsigned int _phase_a_empty = 1;
            unsigned int _phase_work_full_3 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_3 = 0; _tile_iter_3 < grid_m / 2 * grid_n; _tile_iter_3++) {
                if (tile_count_3 > (int)n_tile_3) {
                    int valid_rows_a = (unsigned int)tile_mn_limit[n_tile_3] - n_tile_3 * 256;
                    if (valid_rows_a > 0) {
                        int expert = tile_expert[n_tile_3];
                        if (cta_rank == 0) {
                            mbarrier_wait(throttle_empty_addr + (throttle_stage) * 8, _phase_throttle_empty);
                            mbarrier_arrive(throttle_full_addr + (throttle_stage) * 8);
                            throttle_stage += 1;
                            if (throttle_stage == 3) { throttle_stage = 0; _phase_throttle_empty ^= 1; }
                        }
                        #pragma unroll 1
                        for (int iter_k_2 = 0; iter_k_2 < K_tiles; iter_k_2++) {
                            mbarrier_wait(a_empty_addr + (stage_2) * 8, _phase_a_empty);
                            if (elect_sync()) {
                                if (cta_rank == 0) {
                                    asm volatile(
                                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                        :: "r"((a_full_addr + (stage_2) * 8) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                                }
                                asm volatile(
                                    "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                    " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                                    :: "r"(smem_a_addr + stage_2 * 16384), "l"((&A)), "r"(0), "r"(m_tile_3 * 128), "r"(iter_k_2), "r"(expert),
                                       "r"(((a_full_addr + (stage_2) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(cta_mask_1)) : "memory");
                            }
                            stage_2 += 1;
                            if (stage_2 == 5) { stage_2 = 0; _phase_a_empty ^= 1; }
                        }
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_3) * 8, _phase_work_full_3);
                unsigned int valid_3 = 0;
                unsigned int next_x_3 = 0;
                unsigned int next_y_3 = 0;
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
                    : "r"(work_response_addr + work_stage_3 * 16 + 0 * 16)
                    : "memory");
                valid_3 = _clc_valid_0;
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
                    : "r"(work_response_addr + work_stage_3 * 16 + 0 * 16)
                    : "memory");
                next_x_3 = _clc_ctaid_0;
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
                    : "r"(work_response_addr + work_stage_3 * 16 + 0 * 16)
                    : "memory");
                next_y_3 = _clc_ctaid_1;
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
                unsigned int valid_0_3 = valid_3;
                m_tile_3 = next_x_3 + (unsigned int)cta_rank;
                n_tile_3 = next_y_3;
                if (valid_0_3 == 0) {
                    break;
                }
            }
        }
    }
    // ---- Role: load_sfa ----
    if (warp == 7) {
        { // load_sfa_main
            int tile_count_4 = num_non_exiting_ctas[0];
            unsigned int stage_3 = 0;
            unsigned int work_stage_4 = 0;
            unsigned int m_tile_4 = blockIdx.x;
            unsigned int n_tile_4 = blockIdx.y;
            unsigned int cta_mask_2 = 1 << cta_rank;
            unsigned int _phase_sfa_empty = 1;
            unsigned int _phase_work_full_4 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_4 = 0; _tile_iter_4 < grid_m / 2 * grid_n; _tile_iter_4++) {
                if (tile_count_4 > (int)n_tile_4) {
                    int valid_rows_sfa = (unsigned int)tile_mn_limit[n_tile_4] - n_tile_4 * 256;
                    if (valid_rows_sfa > 0) {
                        int expert_sfa = tile_expert[n_tile_4];
                        #pragma unroll 1
                        for (int iter_k_3 = 0; iter_k_3 < K_tiles; iter_k_3++) {
                            mbarrier_wait(sfa_empty_addr + (stage_3) * 8, _phase_sfa_empty);
                            if (elect_sync()) {
                                if (cta_rank == 0) {
                                    asm volatile(
                                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                        :: "r"((sfa_full_addr + (stage_3) * 8) & 0xFEFFFFFF), "r"((uint32_t)(4096)) : "memory");
                                }
                                asm volatile(
                                    "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                    " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                                    :: "r"(smem_sfa_addr + stage_3 * 2048), "l"((&SFA)), "r"(0), "r"(0), "r"(iter_k_3 * 4), "r"((unsigned int)(expert_sfa * grid_m) + m_tile_4),
                                       "r"(((sfa_full_addr + (stage_3) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(cta_mask_2)) : "memory");
                            }
                            stage_3 += 1;
                            if (stage_3 == 5) { stage_3 = 0; _phase_sfa_empty ^= 1; }
                        }
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_4) * 8, _phase_work_full_4);
                unsigned int valid_4 = 0;
                unsigned int next_x_4 = 0;
                unsigned int next_y_4 = 0;
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
                    : "r"(work_response_addr + work_stage_4 * 16 + 0 * 16)
                    : "memory");
                valid_4 = _clc_valid_3;
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
                    : "r"(work_response_addr + work_stage_4 * 16 + 0 * 16)
                    : "memory");
                next_x_4 = _clc_ctaid_6;
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
                    : "r"(work_response_addr + work_stage_4 * 16 + 0 * 16)
                    : "memory");
                next_y_4 = _clc_ctaid_7;
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
                unsigned int valid_0_4 = valid_4;
                m_tile_4 = next_x_4 + (unsigned int)cta_rank;
                n_tile_4 = next_y_4;
                if (valid_0_4 == 0) {
                    break;
                }
            }
        }
    }
    // ---- Role: copy_sfab_mma ----
    if (warp == 8) {
        { // copy_sfab_mma_main
            int tile_count_5 = num_non_exiting_ctas[0];
            unsigned int _phase_mma_free_0 = 1;
            unsigned int _phase_a_full = 0;
            unsigned int _phase_b_full = 0;
            unsigned int _phase_sfa_full = 0;
            unsigned int _phase_sfb_full = 0;
            unsigned int _phase_work_full_5 = 0;
            if (cta_rank == 0) {
                unsigned int stage_4 = 0;
                unsigned int work_stage_5 = 0;
                int mma_local_idx = 0;
                unsigned int m_tile_5 = blockIdx.x;
                unsigned int n_tile_5 = blockIdx.y;
                #pragma unroll 1
                for (unsigned int _tile_iter_5 = 0; _tile_iter_5 < grid_m / 2 * grid_n; _tile_iter_5++) {
                    if (tile_count_5 > (int)n_tile_5) {
                        int valid_rows_m = (unsigned int)tile_mn_limit[n_tile_5] - n_tile_5 * 256;
                        if (valid_rows_m > 0) {
                            mbarrier_wait(mma_free_addr, _phase_mma_free_0);
                            _phase_mma_free_0 ^= 1;
                            #pragma unroll 1
                            for (int iter_k_4 = 0; iter_k_4 < K_tiles; iter_k_4++) {
                                mbarrier_wait(a_full_addr + (stage_4) * 8, _phase_a_full);
                                mbarrier_wait(b_full_addr + (stage_4) * 8, _phase_b_full);
                                mbarrier_wait(sfa_full_addr + (stage_4) * 8, _phase_sfa_full);
                                mbarrier_wait(sfb_full_addr + (stage_4) * 8, _phase_sfb_full);
                                asm volatile("tcgen05.fence::after_thread_sync;");
                                if (elect_sync()) {
                                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                    #endif
                                    {
                                        uint64_t _tcgen05_cp_desc_0 = ((((uint64_t)(smem_sfa_addr + stage_4 * 2048)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                        asm volatile(
                                            "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                            :: "r"((uint32_t)(tmem_sfa)), "l"(_tcgen05_cp_desc_0)
                                            : "memory");
                                    }
                                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                    #endif
                                    {
                                        uint64_t _tcgen05_cp_desc_1 = ((((uint64_t)(smem_sfa_addr + stage_4 * 2048 + 512)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                        asm volatile(
                                            "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                            :: "r"((uint32_t)(tmem_sfa + 4)), "l"(_tcgen05_cp_desc_1)
                                            : "memory");
                                    }
                                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                    #endif
                                    {
                                        uint64_t _tcgen05_cp_desc_2 = ((((uint64_t)(smem_sfa_addr + stage_4 * 2048 + 1024)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                        asm volatile(
                                            "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                            :: "r"((uint32_t)(tmem_sfa + 8)), "l"(_tcgen05_cp_desc_2)
                                            : "memory");
                                    }
                                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                    #endif
                                    {
                                        uint64_t _tcgen05_cp_desc_3 = ((((uint64_t)(smem_sfa_addr + stage_4 * 2048 + 1536)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                        asm volatile(
                                            "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                            :: "r"((uint32_t)(tmem_sfa + 12)), "l"(_tcgen05_cp_desc_3)
                                            : "memory");
                                    }
                                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                    #endif
                                    {
                                        uint64_t _tcgen05_cp_desc_4 = ((((uint64_t)(smem_sfb_addr + stage_4 * 4096)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                        asm volatile(
                                            "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                            :: "r"((uint32_t)(tmem_sfb)), "l"(_tcgen05_cp_desc_4)
                                            : "memory");
                                    }
                                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                    #endif
                                    {
                                        uint64_t _tcgen05_cp_desc_5 = ((((uint64_t)(smem_sfb_addr + stage_4 * 4096 + 512)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                        asm volatile(
                                            "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                            :: "r"((uint32_t)(tmem_sfb + 8)), "l"(_tcgen05_cp_desc_5)
                                            : "memory");
                                    }
                                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                    #endif
                                    {
                                        uint64_t _tcgen05_cp_desc_6 = ((((uint64_t)(smem_sfb_addr + stage_4 * 4096 + 1024)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                        asm volatile(
                                            "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                            :: "r"((uint32_t)(tmem_sfb + 16)), "l"(_tcgen05_cp_desc_6)
                                            : "memory");
                                    }
                                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                    #endif
                                    {
                                        uint64_t _tcgen05_cp_desc_7 = ((((uint64_t)(smem_sfb_addr + stage_4 * 4096 + 1536)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                        asm volatile(
                                            "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                            :: "r"((uint32_t)(tmem_sfb + 24)), "l"(_tcgen05_cp_desc_7)
                                            : "memory");
                                    }
                                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                    #endif
                                    {
                                        uint64_t _tcgen05_cp_desc_8 = ((((uint64_t)(smem_sfb_addr + stage_4 * 4096 + 2048)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                        asm volatile(
                                            "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                            :: "r"((uint32_t)(tmem_sfb + 4)), "l"(_tcgen05_cp_desc_8)
                                            : "memory");
                                    }
                                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                    #endif
                                    {
                                        uint64_t _tcgen05_cp_desc_9 = ((((uint64_t)(smem_sfb_addr + stage_4 * 4096 + 2048 + 512)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                        asm volatile(
                                            "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                            :: "r"((uint32_t)(tmem_sfb + 12)), "l"(_tcgen05_cp_desc_9)
                                            : "memory");
                                    }
                                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                    #endif
                                    {
                                        uint64_t _tcgen05_cp_desc_10 = ((((uint64_t)(smem_sfb_addr + stage_4 * 4096 + 2048 + 1024)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                        asm volatile(
                                            "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                            :: "r"((uint32_t)(tmem_sfb + 20)), "l"(_tcgen05_cp_desc_10)
                                            : "memory");
                                    }
                                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                    #endif
                                    {
                                        uint64_t _tcgen05_cp_desc_11 = ((((uint64_t)(smem_sfb_addr + stage_4 * 4096 + 2048 + 1536)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                        asm volatile(
                                            "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
                                            :: "r"((uint32_t)(tmem_sfb + 28)), "l"(_tcgen05_cp_desc_11)
                                            : "memory");
                                    }
                                }
                                int _mma_a_lo_0 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage_4) * 1024;
                                int _mma_b_lo_0 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage_4) * 1024;
                                if (elect_sync()) {
                                    {
                                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                                        uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                                        tcgen05_mma_mxf4nvf4_bs_cta2((tmem_accum + (mma_local_idx * 192)), a_desc + 0, b_desc + 0,
                                            0x10400480U, tmem_sfa + 0, tmem_sfb + 0, ((((1) ? ((iter_k_4 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                                    }
                                }
                                int _mma_a_lo_1 = (((smem_a_addr + 32) >> 4) & 0x3FFF) + (stage_4) * 1024;
                                int _mma_b_lo_1 = (((smem_b_addr + 32) >> 4) & 0x3FFF) + (stage_4) * 1024;
                                if (elect_sync()) {
                                    {
                                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                                        uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);

                                        tcgen05_mma_mxf4nvf4_bs_cta2((tmem_accum + (mma_local_idx * 192)), a_desc + 0, b_desc + 0,
                                            0x10400480U, tmem_sfa + 4 + 0, tmem_sfb + 8 + 0, ((((0) ? ((iter_k_4 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                                    }
                                }
                                int _mma_a_lo_2 = (((smem_a_addr + 64) >> 4) & 0x3FFF) + (stage_4) * 1024;
                                int _mma_b_lo_2 = (((smem_b_addr + 64) >> 4) & 0x3FFF) + (stage_4) * 1024;
                                if (elect_sync()) {
                                    {
                                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_2) | ((uint64_t)0x40004040 << 32);
                                        uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_2) | ((uint64_t)0x40004040 << 32);

                                        tcgen05_mma_mxf4nvf4_bs_cta2((tmem_accum + (mma_local_idx * 192)), a_desc + 0, b_desc + 0,
                                            0x10400480U, tmem_sfa + 8 + 0, tmem_sfb + 16 + 0, ((((0) ? ((iter_k_4 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                                    }
                                }
                                int _mma_a_lo_3 = (((smem_a_addr + 96) >> 4) & 0x3FFF) + (stage_4) * 1024;
                                int _mma_b_lo_3 = (((smem_b_addr + 96) >> 4) & 0x3FFF) + (stage_4) * 1024;
                                if (elect_sync()) {
                                    {
                                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_3) | ((uint64_t)0x40004040 << 32);
                                        uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_3) | ((uint64_t)0x40004040 << 32);

                                        tcgen05_mma_mxf4nvf4_bs_cta2((tmem_accum + (mma_local_idx * 192)), a_desc + 0, b_desc + 0,
                                            0x10400480U, tmem_sfa + 12 + 0, tmem_sfb + 24 + 0, ((((0) ? ((iter_k_4 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                                    }
                                }
                                elect_commit_cg2_multicast(a_empty_addr + (stage_4) * 8, (uint16_t)(3));
                                elect_commit_cg2_multicast(b_empty_addr + (stage_4) * 8, (uint16_t)(3));
                                elect_commit_cg2_multicast(sfa_empty_addr + (stage_4) * 8, (uint16_t)(3));
                                elect_commit_cg2_multicast(sfb_empty_addr + (stage_4) * 8, (uint16_t)(1));
                                stage_4 += 1;
                                if (stage_4 == 5) { stage_4 = 0; _phase_a_full ^= 1; _phase_b_full ^= 1; _phase_sfa_full ^= 1; _phase_sfb_full ^= 1; }
                            }
                            elect_commit_cg2_multicast(mma_full_addr, (uint16_t)(3));
                            mma_local_idx = mma_local_idx ^ 1;
                        }
                    }
                    mbarrier_wait(work_full_addr + (work_stage_5) * 8, _phase_work_full_5);
                    unsigned int valid_5 = 0;
                    unsigned int next_x_5 = 0;
                    unsigned int next_y_5 = 0;
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
                    valid_5 = _clc_valid_4;
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
                    next_x_5 = _clc_ctaid_8;
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
                    next_y_5 = _clc_ctaid_9;
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
                    unsigned int valid_0_5 = valid_5;
                    m_tile_5 = next_x_5 + (unsigned int)cta_rank;
                    n_tile_5 = next_y_5;
                    if (valid_0_5 == 0) {
                        break;
                    }
                }
            }
        }
    }
    // ---- Role: work_id ----
    if (warp == 9) {
        { // work_id_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            int tile_count_6 = num_non_exiting_ctas[0];
            unsigned int work_stage_6 = 0;
            unsigned int throttle_stage_1 = 0;
            unsigned int fast_stage = 0;
            unsigned int m_tile_6 = blockIdx.x;
            unsigned int n_tile_6 = blockIdx.y;
            unsigned int _phase_throttle_full = 0;
            unsigned int _phase_fast_ready = 0;
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_work_full_6 = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (unsigned int _tile_iter_6 = 0; _tile_iter_6 < grid_m / 2 * grid_n; _tile_iter_6++) {
                    if (tile_count_6 > (int)n_tile_6) {
                        mbarrier_wait(throttle_full_addr + (throttle_stage_1) * 8, _phase_throttle_full);
                        mbarrier_arrive(throttle_empty_addr + (throttle_stage_1) * 8);
                        throttle_stage_1 += 1;
                        if (throttle_stage_1 == 3) { throttle_stage_1 = 0; _phase_throttle_full ^= 1; }
                    } else {
                        #pragma unroll 1
                        for (unsigned int _drain_iter = 0; _drain_iter < grid_m / 2 * grid_n; _drain_iter++) {
                            if (elect_sync()) {
                                mbarrier_arrive_expect_tx(fast_ready_addr + (fast_stage) * 8, 64);
                                asm volatile(
                                    "fence.proxy.async.shared::cta;\n\t"
                                    "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                        ".mbarrier::complete_tx::bytes.b128"
                                        " [%0], [%1];"
                                    :: "r"(fast_response_addr + fast_stage * 64 + 0 * 16), "r"(fast_ready_addr + fast_stage * 8)
                                    : "memory");
                                asm volatile(
                                    "fence.proxy.async.shared::cta;\n\t"
                                    "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                        ".mbarrier::complete_tx::bytes.b128"
                                        " [%0], [%1];"
                                    :: "r"(fast_response_addr + fast_stage * 64 + 1 * 16), "r"(fast_ready_addr + fast_stage * 8)
                                    : "memory");
                                asm volatile(
                                    "fence.proxy.async.shared::cta;\n\t"
                                    "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                        ".mbarrier::complete_tx::bytes.b128"
                                        " [%0], [%1];"
                                    :: "r"(fast_response_addr + fast_stage * 64 + 2 * 16), "r"(fast_ready_addr + fast_stage * 8)
                                    : "memory");
                                asm volatile(
                                    "fence.proxy.async.shared::cta;\n\t"
                                    "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                        ".mbarrier::complete_tx::bytes.b128"
                                        " [%0], [%1];"
                                    :: "r"(fast_response_addr + fast_stage * 64 + 3 * 16), "r"(fast_ready_addr + fast_stage * 8)
                                    : "memory");
                            }
                            mbarrier_wait(fast_ready_addr + (fast_stage) * 8, _phase_fast_ready);
                            unsigned int canceled = 0;
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
                            canceled += _clc_valid_6;
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
                            canceled += _clc_valid_7;
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
                            canceled += _clc_valid_8;
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
                            canceled += _clc_valid_9;
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            _phase_fast_ready ^= 1;
                            if (canceled == 0) {
                                break;
                            }
                        }
                    }
                    mbarrier_wait(work_empty_addr + (work_stage_6) * 8, _phase_work_empty);
                    if (lane < 2) {
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                            "}"
                            :: "r"(work_full_addr + work_stage_6 * 8), "r"(lane), "r"((uint32_t)(16)) : "memory");
                    }
                    if (elect_sync()) {
                        asm volatile(
                            "fence.proxy.async.shared::cta;\n\t"
                            "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                                " [%0], [%1];"
                            :: "r"(work_response_addr + work_stage_6 * 16 + 0 * 16), "r"(work_full_addr + work_stage_6 * 8)
                            : "memory");
                    }
                    mbarrier_wait(work_full_addr + (work_stage_6) * 8, _phase_work_full_6);
                    unsigned int valid_6 = 0;
                    unsigned int next_x_6 = 0;
                    unsigned int next_y_6 = 0;
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
                        : "r"(work_response_addr + work_stage_6 * 16 + 0 * 16)
                        : "memory");
                    valid_6 = _clc_valid_10;
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
                        : "r"(work_response_addr + work_stage_6 * 16 + 0 * 16)
                        : "memory");
                    next_x_6 = _clc_ctaid_12;
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
                        : "r"(work_response_addr + work_stage_6 * 16 + 0 * 16)
                        : "memory");
                    next_y_6 = _clc_ctaid_13;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(work_empty_addr + work_stage_6 * 8), "r"(0) : "memory");
                    work_stage_6 += 1;
                    if (work_stage_6 == 3) { work_stage_6 = 0; _phase_work_empty ^= 1; _phase_work_full_6 ^= 1; }
                    unsigned int valid_0_6 = valid_6;
                    m_tile_6 = next_x_6 + (unsigned int)cta_rank;
                    n_tile_6 = next_y_6;
                    if (valid_0_6 == 0) {
                        break;
                    }
                }
                #pragma unroll 1
                for (unsigned int _tail_iter = 0; _tail_iter < 3; _tail_iter++) {
                    mbarrier_wait(work_empty_addr + (work_stage_6) * 8, _phase_work_empty);
                    work_stage_6 += 1;
                    if (work_stage_6 == 3) { work_stage_6 = 0; _phase_work_empty ^= 1; _phase_work_full_6 ^= 1; }
                }
            }
        }
    }
    // ---- Role: padding ----
    if (warp >= 10 && warp <= 11) {
        { // padding_main
            unsigned int work_stage_7 = 0;
            unsigned int m_tile_7 = blockIdx.x;
            unsigned int n_tile_7 = blockIdx.y;
            unsigned int _phase_work_full_7 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_7 = 0; _tile_iter_7 < grid_m / 2 * grid_n; _tile_iter_7++) {
                mbarrier_wait(work_full_addr + (work_stage_7) * 8, _phase_work_full_7);
                unsigned int valid_7 = 0;
                unsigned int next_x_7 = 0;
                unsigned int next_y_7 = 0;
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
                    : "r"(work_response_addr + work_stage_7 * 16 + 0 * 16)
                    : "memory");
                valid_7 = _clc_valid_11;
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
                    : "r"(work_response_addr + work_stage_7 * 16 + 0 * 16)
                    : "memory");
                next_x_7 = _clc_ctaid_14;
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
                    : "r"(work_response_addr + work_stage_7 * 16 + 0 * 16)
                    : "memory");
                next_y_7 = _clc_ctaid_15;
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
                unsigned int valid_0_7 = valid_7;
                m_tile_7 = next_x_7 + (unsigned int)cta_rank;
                n_tile_7 = next_y_7;
                if (valid_0_7 == 0) {
                    break;
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
