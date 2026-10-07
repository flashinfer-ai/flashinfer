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
#define TMEM_NCOLS 128
#define TMEM_ACCUM_OFFSET 0
#define NUM_K_PIPE_STAGES 5
#define NUM_MMA_PIPE_STAGES 2
#define NUM_WORK_PIPE_STAGES 3
#define NUM_THROTTLE_PIPE_STAGES 3
#define NUM_FAST_PIPE_STAGES 1
#define NUM_EXIT_PIPE_STAGES 1
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 32768
#define SMEM_SMEM_A_STRIDE 32768
#define SMEM_SMEM_B_OFF 164864
#define SMEM_SMEM_B_STAGE_BYTES 8192
#define SMEM_SMEM_B_STRIDE 8192
#define SMEM_EPI_STAGING_OFF 205824
#define SMEM_EPI_STAGING_STAGE_BYTES 16384
#define SMEM_EPI_STAGING_STRIDE 16384
#define SMEM_EPI_STAGING_U64_OFF 205824
#define SMEM_EPI_STAGING_U64_STAGE_BYTES 16384
#define SMEM_EPI_STAGING_U64_STRIDE 16384
#define SMEM_WORK_RESPONSE_OFF 222208
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_FAST_RESPONSE_OFF 222256
#define SMEM_FAST_RESPONSE_STAGE_BYTES 64
#define SMEM_FAST_RESPONSE_STRIDE 64
#define SMEM_TOTAL 222336
#define THREADS 256
#define BLOCK_N 64
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

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_cake_stepfun_moe_b392eb18f57ca62e718d(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap C, int* __restrict__ tile_expert, int* __restrict__ tile_mn_limit, int* __restrict__ num_non_exiting_ctas, int M_out, int K, int grid_m, int grid_n, int K_tiles)
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
    #define k_done_addr (mbar_base + 80)
    #define mma_full_addr (mbar_base + 120)
    #define mma_free_addr (mbar_base + 136)
    #define work_full_addr (mbar_base + 152)
    #define work_empty_addr (mbar_base + 176)
    #define throttle_full_addr (mbar_base + 200)
    #define throttle_empty_addr (mbar_base + 224)
    #define fast_ready_addr (mbar_base + 248)
    #define exit_bar_addr (mbar_base + 256)

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
    __nv_bfloat16* smem_b = reinterpret_cast<__nv_bfloat16*>(smem_raw + 164864);
    const int smem_b_addr = smem + 164864;
    __nv_bfloat16* epi_staging = reinterpret_cast<__nv_bfloat16*>(smem_raw + 205824);
    const int epi_staging_addr = smem + 205824;
    unsigned long long* epi_staging_u64 = reinterpret_cast<unsigned long long*>(smem_raw + 205824);
    const int epi_staging_u64_addr = smem + 205824;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 222208);
    const int work_response_addr = smem + 222208;
    unsigned int* fast_response = reinterpret_cast<unsigned int*>(smem_raw + 222256);
    const int fast_response_addr = smem + 222256;

    // Mbarrier init (11 pipeline groups, 0 ordered-sequence groups, 33 barriers)
    // Mbarriers at smem_raw[0..264)

    if (warp == 0) {
        // --- pipeline 'k_pipe' ---
        // a_full: 5 barriers, init_count=2
        // b_full: 5 barriers, init_count=2
        // k_done: 5 barriers, init_count=1
        // --- pipeline 'mma_pipe' ---
        // mma_full: 2 barriers, init_count=1
        // mma_free: 2 barriers, init_count=256
        // --- pipeline 'work_pipe' ---
        // work_full: 3 barriers, init_count=1
        // work_empty: 3 barriers, init_count=448
        // --- pipeline 'throttle_pipe' ---
        // throttle_full: 3 barriers, init_count=32
        // throttle_empty: 3 barriers, init_count=32
        // --- pipeline 'fast_pipe' ---
        // fast_ready: 1 barriers, init_count=1
        // --- pipeline 'exit_pipe' ---
        // exit_bar: 1 barriers, init_count=32
        // Warp-cooperative initialization in physical record order.
        uint32_t _mbarrier_init_count_0_0 = 1;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(31), "r"((uint32_t)(32)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(25), "r"((uint32_t)(448)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(22), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(19), "r"((uint32_t)(256)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(17), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(10), "r"((uint32_t)(2)));
        mbarrier_init(smem + 0 + lane * 8, _mbarrier_init_count_0_0);
        if (lane < 1) {
            mbarrier_init(smem + 256 + lane * 8, 32);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    __syncwarp();

    // TMEM alloc (128 columns, 128 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 264);
    if (warp == 0) {
        int _tmem_hold = smem + 264;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
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

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 96;");
    }

    // ---- Role: epilogue ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 168;");
        { // epilogue_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            int tile_count = num_non_exiting_ctas[0];
            const int warp_0 = warp;
            const int lane_1 = lane;
            unsigned int acc_stage = 0;
            unsigned int work_stage = 0;
            unsigned int m_tile = blockIdx.x;
            unsigned int n_tile = blockIdx.y;
            int base_feature = warp_0 * 32 + lane_1 / 4 * 4;
            int base_token = lane_1 % 4 * 2;
            float converted[4];
            unsigned int packed[2];
            unsigned long long packed_word = 0;
            unsigned int _phase_mma_full = 0;
            unsigned int _phase_work_full = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter = 0; _tile_iter < grid_m / 2 * grid_n; _tile_iter++) {
                int valid_rows = (unsigned int)tile_mn_limit[n_tile] - n_tile * 64;
                if (tile_count > (int)n_tile) {
                    if (valid_rows > 0) {
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
                        asm volatile("barrier.sync 8, 128;" ::: "memory");
                        int token = base_token;
                        converted[0] = _tmem_load_0[0];
                        converted[1] = _tmem_load_0[2];
                        converted[2] = _tmem_load_1[0];
                        converted[3] = _tmem_load_1[2];
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
                        converted[0] = _tmem_load_0[1];
                        converted[1] = _tmem_load_0[3];
                        converted[2] = _tmem_load_1[1];
                        converted[3] = _tmem_load_1[3];
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
                        converted[0] = _tmem_load_0[4];
                        converted[1] = _tmem_load_0[6];
                        converted[2] = _tmem_load_1[4];
                        converted[3] = _tmem_load_1[6];
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
                        converted[0] = _tmem_load_0[5];
                        converted[1] = _tmem_load_0[7];
                        converted[2] = _tmem_load_1[5];
                        converted[3] = _tmem_load_1[7];
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
                        converted[0] = _tmem_load_0[8];
                        converted[1] = _tmem_load_0[10];
                        converted[2] = _tmem_load_1[8];
                        converted[3] = _tmem_load_1[10];
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
                        converted[0] = _tmem_load_0[9];
                        converted[1] = _tmem_load_0[11];
                        converted[2] = _tmem_load_1[9];
                        converted[3] = _tmem_load_1[11];
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
                        converted[0] = _tmem_load_0[12];
                        converted[1] = _tmem_load_0[14];
                        converted[2] = _tmem_load_1[12];
                        converted[3] = _tmem_load_1[14];
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
                        converted[0] = _tmem_load_0[13];
                        converted[1] = _tmem_load_0[15];
                        converted[2] = _tmem_load_1[13];
                        converted[3] = _tmem_load_1[15];
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
                        converted[0] = _tmem_load_0[16];
                        converted[1] = _tmem_load_0[18];
                        converted[2] = _tmem_load_1[16];
                        converted[3] = _tmem_load_1[18];
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
                        converted[0] = _tmem_load_0[17];
                        converted[1] = _tmem_load_0[19];
                        converted[2] = _tmem_load_1[17];
                        converted[3] = _tmem_load_1[19];
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
                        converted[0] = _tmem_load_0[20];
                        converted[1] = _tmem_load_0[22];
                        converted[2] = _tmem_load_1[20];
                        converted[3] = _tmem_load_1[22];
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
                        converted[0] = _tmem_load_0[21];
                        converted[1] = _tmem_load_0[23];
                        converted[2] = _tmem_load_1[21];
                        converted[3] = _tmem_load_1[23];
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
                        converted[0] = _tmem_load_0[24];
                        converted[1] = _tmem_load_0[26];
                        converted[2] = _tmem_load_1[24];
                        converted[3] = _tmem_load_1[26];
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
                        converted[0] = _tmem_load_0[25];
                        converted[1] = _tmem_load_0[27];
                        converted[2] = _tmem_load_1[25];
                        converted[3] = _tmem_load_1[27];
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
                        converted[0] = _tmem_load_0[28];
                        converted[1] = _tmem_load_0[30];
                        converted[2] = _tmem_load_1[28];
                        converted[3] = _tmem_load_1[30];
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
                        converted[0] = _tmem_load_0[29];
                        converted[1] = _tmem_load_0[31];
                        converted[2] = _tmem_load_1[29];
                        converted[3] = _tmem_load_1[31];
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
                        asm volatile("barrier.sync 8, 128;" ::: "memory");
                        if (warp == 0) {
                            if (elect_sync()) {
                                int padding_rows = (64 - valid_rows % 64) % 64;
                                tma_store_4d((&C), m_tile * 128, padding_rows, 1073741824, n_tile * 64 - (unsigned int)padding_rows + 1073741824, epi_staging_addr);
                                tma_store_4d((&C), m_tile * 128 + 64, padding_rows, 1073741824, n_tile * 64 - (unsigned int)padding_rows + 1073741824, epi_staging_addr + 8192);
                            }
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                        asm volatile("barrier.sync 8, 128;" ::: "memory");
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((mma_free_addr + (acc_stage) * 8) & 0xFEFFFFFF) : "memory");
                        acc_stage += 1;
                        if (acc_stage == 2) { acc_stage = 0; _phase_mma_full ^= 1; }
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage) * 8, _phase_work_full);
                unsigned int valid = 0;
                unsigned int next_x = 0;
                unsigned int next_y = 0;
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
                valid = _clc_valid_3;
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
                next_x = _clc_ctaid_6;
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
                next_y = _clc_ctaid_7;
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
            asm volatile("barrier.sync 7, 128;" ::: "memory");
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
                asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(128));
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 4) {
        { // mma_main
            int tile_count_1 = num_non_exiting_ctas[0];
            unsigned int _phase_mma_free = 1;
            unsigned int _phase_a_full = 0;
            unsigned int _phase_b_full = 0;
            unsigned int _phase_work_full_1 = 0;
            if (cta_rank == 0) {
                unsigned int stage = 0;
                unsigned int acc_stage_1 = 0;
                unsigned int work_stage_1 = 0;
                unsigned int m_tile_1 = blockIdx.x;
                unsigned int n_tile_1 = blockIdx.y;
                #pragma unroll 1
                for (unsigned int _tile_iter_1 = 0; _tile_iter_1 < grid_m / 2 * grid_n; _tile_iter_1++) {
                    if (tile_count_1 > (int)n_tile_1) {
                        int valid_rows_m = (unsigned int)tile_mn_limit[n_tile_1] - n_tile_1 * 64;
                        if (valid_rows_m > 0) {
                            mbarrier_wait(mma_free_addr + (acc_stage_1) * 8, _phase_mma_free);
                            #pragma unroll 2
                            for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                                mbarrier_wait(a_full_addr + (stage) * 8, _phase_a_full);
                                mbarrier_wait(b_full_addr + (stage) * 8, _phase_b_full);
                                asm volatile("tcgen05.fence::after_thread_sync;");
                                int _mma_a_lo_0 = (((smem_a_addr) >> 4) & 0x3FFF) + (stage) * 2048;
                                int _mma_b_lo_0 = (((smem_b_addr) >> 4) & 0x3FFF) + (stage) * 512;
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
                    "mov.b32 id, 269485200;\n\t"
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
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 250;\n\t"
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
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_accum + (acc_stage_1 * 64))), "r"(((((iter_k == 0) ? 1 : 0)) ? 0 : 1)));
                                elect_commit_cg2_multicast(k_done_addr + (stage) * 8, (uint16_t)(3));
                                if (iter_k + 1 == K_tiles) {
                                    elect_commit_cg2_multicast(mma_full_addr + (acc_stage_1) * 8, (uint16_t)(3));
                                }
                                stage += 1;
                                if (stage == 5) { stage = 0; _phase_a_full ^= 1; _phase_b_full ^= 1; }
                            }
                            acc_stage_1 += 1;
                            if (acc_stage_1 == 2) { acc_stage_1 = 0; _phase_mma_free ^= 1; }
                        }
                    }
                    mbarrier_wait(work_full_addr + (work_stage_1) * 8, _phase_work_full_1);
                    unsigned int valid_1 = 0;
                    unsigned int next_x_1 = 0;
                    unsigned int next_y_1 = 0;
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
                        : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    valid_1 = _clc_valid_2;
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
                        : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    next_x_1 = _clc_ctaid_4;
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
                        : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    next_y_1 = _clc_ctaid_5;
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
            }
        }
    }
    // ---- Role: load_a ----
    if (warp == 5) {
        { // load_a_main
            int tile_count_2 = num_non_exiting_ctas[0];
            unsigned int stage_1 = 0;
            unsigned int work_stage_2 = 0;
            unsigned int throttle_stage = 0;
            unsigned int m_tile_2 = blockIdx.x;
            unsigned int n_tile_2 = blockIdx.y;
            unsigned int cta_mask = 1 << cta_rank;
            unsigned int _phase_throttle_empty = 1;
            unsigned int _phase_k_done = 1;
            unsigned int _phase_work_full_2 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_2 = 0; _tile_iter_2 < grid_m / 2 * grid_n; _tile_iter_2++) {
                if (tile_count_2 > (int)n_tile_2) {
                    int valid_rows_a = (unsigned int)tile_mn_limit[n_tile_2] - n_tile_2 * 64;
                    if (valid_rows_a > 0) {
                        int expert = tile_expert[n_tile_2];
                        if (cta_rank == 0) {
                            mbarrier_wait(throttle_empty_addr + (throttle_stage) * 8, _phase_throttle_empty);
                            mbarrier_arrive(throttle_full_addr + (throttle_stage) * 8);
                            throttle_stage += 1;
                            if (throttle_stage == 3) { throttle_stage = 0; _phase_throttle_empty ^= 1; }
                        }
                        #pragma unroll 1
                        for (int iter_k_1 = 0; iter_k_1 < K_tiles; iter_k_1++) {
                            mbarrier_wait(k_done_addr + (stage_1) * 8, _phase_k_done);
                            if (elect_sync()) {
                                asm volatile(
                                    "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                    " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                                    :: "r"(smem_a_addr + stage_1 * 32768), "l"((&A)), "r"(0), "r"(m_tile_2 * 128), "r"(iter_k_1 * 2), "r"(expert),
                                       "r"(((a_full_addr + (stage_1) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(cta_mask)) : "memory");
                                asm volatile(
                                    "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                    :: "r"((a_full_addr + (stage_1) * 8) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                            }
                            stage_1 += 1;
                            if (stage_1 == 5) { stage_1 = 0; _phase_k_done ^= 1; }
                        }
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_2) * 8, _phase_work_full_2);
                unsigned int valid_2 = 0;
                unsigned int next_x_2 = 0;
                unsigned int next_y_2 = 0;
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
                valid_2 = _clc_valid_0;
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
                next_x_2 = _clc_ctaid_0;
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
                next_y_2 = _clc_ctaid_1;
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
        }
    }
    // ---- Role: load_b ----
    if (warp == 6) {
        { // load_b_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            int tile_count_3 = num_non_exiting_ctas[0];
            unsigned int stage_2 = 0;
            unsigned int work_stage_3 = 0;
            unsigned int m_tile_3 = blockIdx.x;
            unsigned int n_tile_3 = blockIdx.y;
            unsigned int cta_mask_1 = 1 << cta_rank;
            unsigned int _phase_k_done_1 = 1;
            unsigned int _phase_work_full_3 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_3 = 0; _tile_iter_3 < grid_m / 2 * grid_n; _tile_iter_3++) {
                int valid_rows_1 = (unsigned int)tile_mn_limit[n_tile_3] - n_tile_3 * 64;
                if (tile_count_3 > (int)n_tile_3) {
                    if (valid_rows_1 > 0) {
                        int padding_rows_1 = (64 - valid_rows_1 % 64) % 64;
                        int box_row0 = cta_rank * 32 + padding_rows_1;
                        #pragma unroll 1
                        for (int iter_k_2 = 0; iter_k_2 < K_tiles; iter_k_2++) {
                            mbarrier_wait(k_done_addr + (stage_2) * 8, _phase_k_done_1);
                            if (elect_sync()) {
                                asm volatile(
                                    "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                    " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                                    :: "r"(smem_b_addr + stage_2 * 8192), "l"((&B)), "r"(iter_k_2 * 128), "r"(box_row0), "r"(1073741824), "r"(n_tile_3 * 64 - (unsigned int)padding_rows_1 + 1073741824),
                                       "r"(((b_full_addr + (stage_2) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(cta_mask_1)) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                    " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                                    :: "r"(smem_b_addr + stage_2 * 8192 + 4096), "l"((&B)), "r"(iter_k_2 * 128 + 64), "r"(box_row0), "r"(1073741824), "r"(n_tile_3 * 64 - (unsigned int)padding_rows_1 + 1073741824),
                                       "r"(((b_full_addr + (stage_2) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(cta_mask_1)) : "memory");
                                asm volatile(
                                    "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                    :: "r"((b_full_addr + (stage_2) * 8) & 0xFEFFFFFF), "r"((uint32_t)(8192)) : "memory");
                            }
                            stage_2 += 1;
                            if (stage_2 == 5) { stage_2 = 0; _phase_k_done_1 ^= 1; }
                        }
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_3) * 8, _phase_work_full_3);
                unsigned int valid_3 = 0;
                unsigned int next_x_3 = 0;
                unsigned int next_y_3 = 0;
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
                    : "r"(work_response_addr + work_stage_3 * 16 + 0 * 16)
                    : "memory");
                valid_3 = _clc_valid_1;
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
                    : "r"(work_response_addr + work_stage_3 * 16 + 0 * 16)
                    : "memory");
                next_x_3 = _clc_ctaid_2;
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
                    : "r"(work_response_addr + work_stage_3 * 16 + 0 * 16)
                    : "memory");
                next_y_3 = _clc_ctaid_3;
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
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
    }
    // ---- Role: work_id ----
    if (warp == 7) {
        { // work_id_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            int tile_count_4 = num_non_exiting_ctas[0];
            unsigned int work_stage_4 = 0;
            unsigned int throttle_stage_1 = 0;
            unsigned int fast_stage = 0;
            unsigned int m_tile_4 = blockIdx.x;
            unsigned int n_tile_4 = blockIdx.y;
            unsigned int _phase_throttle_full = 0;
            unsigned int _phase_fast_ready = 0;
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_work_full_4 = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (unsigned int _tile_iter_4 = 0; _tile_iter_4 < grid_m / 2 * grid_n; _tile_iter_4++) {
                    if (tile_count_4 > (int)n_tile_4) {
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
                                : "r"(fast_response_addr + fast_stage * 64 + 0 * 16)
                                : "memory");
                            canceled += _clc_valid_4;
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
                                : "r"(fast_response_addr + fast_stage * 64 + 1 * 16)
                                : "memory");
                            canceled += _clc_valid_5;
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
                                : "r"(fast_response_addr + fast_stage * 64 + 2 * 16)
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
                                : "r"(fast_response_addr + fast_stage * 64 + 3 * 16)
                                : "memory");
                            canceled += _clc_valid_7;
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            _phase_fast_ready ^= 1;
                            if (canceled == 0) {
                                break;
                            }
                        }
                    }
                    mbarrier_wait(work_empty_addr + (work_stage_4) * 8, _phase_work_empty);
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
                    unsigned int valid_4 = 0;
                    unsigned int next_x_4 = 0;
                    unsigned int next_y_4 = 0;
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
                        : "r"(work_response_addr + work_stage_4 * 16 + 0 * 16)
                        : "memory");
                    valid_4 = _clc_valid_8;
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
                    next_x_4 = _clc_ctaid_8;
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
                    next_y_4 = _clc_ctaid_9;
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
                    unsigned int valid_0_4 = valid_4;
                    m_tile_4 = next_x_4 + (unsigned int)cta_rank;
                    n_tile_4 = next_y_4;
                    if (valid_0_4 == 0) {
                        break;
                    }
                }
                #pragma unroll 1
                for (unsigned int _tail_iter = 0; _tail_iter < 3; _tail_iter++) {
                    mbarrier_wait(work_empty_addr + (work_stage_4) * 8, _phase_work_empty);
                    work_stage_4 += 1;
                    if (work_stage_4 == 3) { work_stage_4 = 0; _phase_work_empty ^= 1; _phase_work_full_4 ^= 1; }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
