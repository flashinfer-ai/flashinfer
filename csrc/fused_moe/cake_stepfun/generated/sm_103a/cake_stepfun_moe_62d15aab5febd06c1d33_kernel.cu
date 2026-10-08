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

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 272
#define TMEM_ACCUM_OFFSET 0
#define TMEM_SFA_OFFSET 128
#define TMEM_SFB_OFFSET 224
#define NUM_K_PIPE_STAGES 3
#define NUM_MMA_PIPE_STAGES 2
#define NUM_WORK_PIPE_STAGES 3
#define NUM_THROTTLE_PIPE_STAGES 3
#define NUM_FAST_PIPE_STAGES 1
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 32768
#define SMEM_SMEM_A_STRIDE 32768
#define SMEM_SMEM_B_OFF 99328
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 16384
#define SMEM_EPI_STAGING_OFF 148480
#define SMEM_EPI_STAGING_STAGE_BYTES 16384
#define SMEM_EPI_STAGING_STRIDE 16384
#define SMEM_EPI_STAGING_U64_OFF 148480
#define SMEM_EPI_STAGING_U64_STAGE_BYTES 16384
#define SMEM_EPI_STAGING_U64_STRIDE 16384
#define SMEM_SMEM_SFA_OFF 164864
#define SMEM_SMEM_SFA_STAGE_BYTES 4096
#define SMEM_SMEM_SFA_STRIDE 4096
#define SMEM_SMEM_SFB_OFF 177152
#define SMEM_SMEM_SFB_STAGE_BYTES 2048
#define SMEM_SMEM_SFB_STRIDE 2048
#define SMEM_WORK_RESPONSE_OFF 183296
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_FAST_RESPONSE_OFF 183344
#define SMEM_FAST_RESPONSE_STAGE_BYTES 64
#define SMEM_FAST_RESPONSE_STRIDE 64
#define SMEM_TOTAL 183424
#define THREADS 512
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


__device__ __forceinline__ void tcgen05_mma_mxf4nvf4_bs(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
    int sfa_taddr, int sfb_taddr, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X"
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


__device__ __forceinline__ void elect_commit(int mbar_addr) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];\n\t"
        "}\n"
        :: "r"(mbar_addr));
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



__device__ __forceinline__ void elect_commit2(int mbar_addr0, int mbar_addr1) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];\n\t"
        "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%1];\n\t"
        "}\n"
        :: "r"(mbar_addr0), "r"(mbar_addr1) : "memory");
}





__device__ __forceinline__ void tma_3d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_4d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_store_4d(
    const void *tmap, int x, int y, int z, int w, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3, %4}], [%5];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(w), "r"(smem_addr) : "memory");
}



__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) void
kernel_cake_stepfun_moe_62d15aab5febd06c1d33(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, const __grid_constant__ CUtensorMap C, int* __restrict__ tile_expert, int* __restrict__ tile_mn_limit, int* __restrict__ num_non_exiting_ctas, float* __restrict__ scale_c, int M_out, int K, int grid_m, int grid_n, int K_tiles)
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
    #define b_full_addr (mbar_base + 24)
    #define sfa_full_addr (mbar_base + 48)
    #define sfb_full_addr (mbar_base + 72)
    #define sfa_free_addr (mbar_base + 96)
    #define sfb_free_addr (mbar_base + 120)
    #define tmem_sfa_full_addr (mbar_base + 144)
    #define tmem_sfb_full_addr (mbar_base + 168)
    #define k_done_addr (mbar_base + 192)
    #define mma_full_addr (mbar_base + 216)
    #define mma_free_addr (mbar_base + 232)
    #define work_full_addr (mbar_base + 248)
    #define work_empty_addr (mbar_base + 272)
    #define throttle_full_addr (mbar_base + 296)
    #define throttle_empty_addr (mbar_base + 320)
    #define fast_ready_addr (mbar_base + 344)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + 99328);
    const int smem_b_addr = smem + 99328;
    __nv_bfloat16* epi_staging = reinterpret_cast<__nv_bfloat16*>(smem_raw + 148480);
    const int epi_staging_addr = smem + 148480;
    unsigned long long* epi_staging_u64 = reinterpret_cast<unsigned long long*>(smem_raw + 148480);
    const int epi_staging_u64_addr = smem + 148480;
    uint8_t* smem_sfa = reinterpret_cast<uint8_t*>(smem_raw + 164864);
    const int smem_sfa_addr = smem + 164864;
    uint8_t* smem_sfb = reinterpret_cast<uint8_t*>(smem_raw + 177152);
    const int smem_sfb_addr = smem + 177152;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 183296);
    const int work_response_addr = smem + 183296;
    unsigned int* fast_response = reinterpret_cast<unsigned int*>(smem_raw + 183344);
    const int fast_response_addr = smem + 183344;

    // Mbarrier init (16 pipeline groups, 0 ordered-sequence groups, 44 barriers)
    // Mbarriers at smem_raw[0..352)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'k_pipe' ---
            // a_full: 3 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            // b_full: 3 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // sfa_full: 3 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            // sfb_full: 3 barriers, init_count=1
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // sfa_free: 3 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            // sfb_free: 3 barriers, init_count=4
            mbarrier_init(smem + 120, 4);
            mbarrier_init(smem + 128, 4);
            mbarrier_init(smem + 136, 4);
            // tmem_sfa_full: 3 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            // tmem_sfb_full: 3 barriers, init_count=1
            mbarrier_init(smem + 168, 1);
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            // k_done: 3 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            // --- pipeline 'mma_pipe' ---
            // mma_full: 2 barriers, init_count=1
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            // mma_free: 2 barriers, init_count=4
            mbarrier_init(smem + 232, 4);
            mbarrier_init(smem + 240, 4);
            // --- pipeline 'work_pipe' ---
            // work_full: 3 barriers, init_count=1
            mbarrier_init(smem + 248, 1);
            mbarrier_init(smem + 256, 1);
            mbarrier_init(smem + 264, 1);
            // work_empty: 3 barriers, init_count=512
            mbarrier_init(smem + 272, 512);
            mbarrier_init(smem + 280, 512);
            mbarrier_init(smem + 288, 512);
            // --- pipeline 'throttle_pipe' ---
            // throttle_full: 3 barriers, init_count=32
            mbarrier_init(smem + 296, 32);
            mbarrier_init(smem + 304, 32);
            mbarrier_init(smem + 312, 32);
            // throttle_empty: 3 barriers, init_count=32
            mbarrier_init(smem + 320, 32);
            mbarrier_init(smem + 328, 32);
            mbarrier_init(smem + 336, 32);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 272 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 352);
    if (warp == 0) {
        int _tmem_hold = smem + 352;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;
    const int tmem_sfa = taddr + 128;
    const int tmem_sfb = taddr + 224;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 8 && warp <= 15) {
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
            unsigned int acc_stage = 0;
            unsigned int work_stage = 0;
            unsigned int m_tile = blockIdx.x;
            unsigned int n_tile = blockIdx.y;
            int base_feature = warp_0 * 32 + lane_1 / 4 * 4;
            int base_token = lane_1 % 4 * 2;
            float converted[4];
            unsigned int packed[2];
            unsigned long long packed_word = 0;
            float frag[64];
            unsigned int _phase_mma_full = 0;
            unsigned int _phase_work_full = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter = 0; _tile_iter < grid_m * grid_n; _tile_iter++) {
                int valid_rows = (unsigned int)tile_mn_limit[n_tile] - n_tile * 64;
                if (tile_count > (int)n_tile) {
                    if (valid_rows > 0) {
                        int expert_e = tile_expert[n_tile];
                        float sc = scale_c[expert_e];
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
                        for (int i = 0; i < 32; i++) {
                            frag[i] = _tmem_load_0[i];
                            frag[32 + i] = _tmem_load_1[i];
                        }
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                        asm volatile("barrier.sync 7, 128;" ::: "memory");
                        for (int token_group = 0; token_group < 8; token_group++) {
                            int token = base_token + token_group * 8;
                            converted[0] = frag[token_group * 4] * sc;
                            converted[1] = frag[token_group * 4 + 2] * sc;
                            converted[2] = frag[32 + token_group * 4] * sc;
                            converted[3] = frag[32 + token_group * 4 + 2] * sc;
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
                            int token_0 = base_token + token_group * 8 + 1;
                            converted[0] = frag[token_group * 4 + 1] * sc;
                            converted[1] = frag[token_group * 4 + 1 + 2] * sc;
                            converted[2] = frag[32 + (token_group * 4 + 1)] * sc;
                            converted[3] = frag[32 + (token_group * 4 + 1) + 2] * sc;
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
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 7, 128;" ::: "memory");
                        if (warp == 0) {
                            if (elect_sync()) {
                                int padding_rows = (64 - valid_rows % 64) % 64;
                                tma_store_4d((&C), m_tile * 128, padding_rows, 1073741824, n_tile * 64 - (unsigned int)padding_rows + 1073741824, epi_staging_addr);
                                tma_store_4d((&C), m_tile * 128 + 64, padding_rows, 1073741824, n_tile * 64 - (unsigned int)padding_rows + 1073741824, epi_staging_addr + 8192);
                            }
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                        asm volatile("barrier.sync 7, 128;" ::: "memory");
                        if (elect_sync()) {
                            mbarrier_arrive(mma_free_addr + (acc_stage) * 8);
                        }
                        acc_stage += 1;
                        if (acc_stage == 2) { acc_stage = 0; _phase_mma_full ^= 1; }
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage) * 8, _phase_work_full);
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
                valid = _clc_valid_7;
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
                mbarrier_arrive(work_empty_addr + (work_stage) * 8);
                work_stage += 1;
                if (work_stage == 3) { work_stage = 0; _phase_work_full ^= 1; }
                unsigned int valid_0 = valid;
                m_tile = next_x;
                n_tile = next_y;
                if (valid_0 == 0) {
                    break;
                }
            }
        }
    }
    // ---- Role: copy_sfb ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 72;");
        { // copy_sfb_main
            int tile_count_1 = num_non_exiting_ctas[0];
            unsigned int stage = 0;
            unsigned int work_stage_1 = 0;
            unsigned int m_tile_1 = blockIdx.x;
            unsigned int n_tile_1 = blockIdx.y;
            const int lane_0 = lane;
            unsigned int words[2];
            unsigned int _phase_sfb_full = 0;
            unsigned int _phase_k_done = 1;
            unsigned int _phase_work_full_1 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_1 = 0; _tile_iter_1 < grid_m * grid_n; _tile_iter_1++) {
                if (tile_count_1 > (int)n_tile_1) {
                    int valid_rows_csfb = (unsigned int)tile_mn_limit[n_tile_1] - n_tile_1 * 64;
                    if (valid_rows_csfb > 0) {
                        #pragma unroll 1
                        for (int _iter_k = 0; _iter_k < K_tiles; _iter_k++) {
                            mbarrier_wait(sfb_full_addr + (stage) * 8, _phase_sfb_full);
                            mbarrier_wait(k_done_addr + (stage) * 8, _phase_k_done);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            words[0] = 0;
                            int row_w = lane_0;
                            if (row_w < 64) {
                                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&words[0])) : "r"(smem_sfb_addr + stage * 2048 + (unsigned int)(row_w / 8 * 256) + (unsigned int)(row_w % 8 * 4)));
                            }
                            words[1] = 0;
                            int row_w_0 = lane_0 + 32;
                            if (row_w_0 < 64) {
                                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&words[1])) : "r"(smem_sfb_addr + stage * 2048 + (unsigned int)(row_w_0 / 8 * 256) + (unsigned int)(row_w_0 % 8 * 4)));
                            }
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x2.b32"
                                " [%0], {%1, %2};"
                                :: "r"(taddr + 224 + stage * 16), "r"(words[0]), "r"(words[1]));
                            words[0] = 0;
                            int row_w_1 = lane_0;
                            if (row_w_1 < 64) {
                                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&words[0])) : "r"(smem_sfb_addr + stage * 2048 + (unsigned int)(row_w_1 / 8 * 256) + 32 + (unsigned int)(row_w_1 % 8 * 4)));
                            }
                            words[1] = 0;
                            int row_w_2 = lane_0 + 32;
                            if (row_w_2 < 64) {
                                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&words[1])) : "r"(smem_sfb_addr + stage * 2048 + (unsigned int)(row_w_2 / 8 * 256) + 32 + (unsigned int)(row_w_2 % 8 * 4)));
                            }
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x2.b32"
                                " [%0], {%1, %2};"
                                :: "r"(taddr + 224 + stage * 16 + 2), "r"(words[0]), "r"(words[1]));
                            words[0] = 0;
                            int row_w_3 = lane_0;
                            if (row_w_3 < 64) {
                                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&words[0])) : "r"(smem_sfb_addr + stage * 2048 + (unsigned int)(row_w_3 / 8 * 256) + 64 + (unsigned int)(row_w_3 % 8 * 4)));
                            }
                            words[1] = 0;
                            int row_w_4 = lane_0 + 32;
                            if (row_w_4 < 64) {
                                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&words[1])) : "r"(smem_sfb_addr + stage * 2048 + (unsigned int)(row_w_4 / 8 * 256) + 64 + (unsigned int)(row_w_4 % 8 * 4)));
                            }
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x2.b32"
                                " [%0], {%1, %2};"
                                :: "r"(taddr + 224 + stage * 16 + 4), "r"(words[0]), "r"(words[1]));
                            words[0] = 0;
                            int row_w_5 = lane_0;
                            if (row_w_5 < 64) {
                                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&words[0])) : "r"(smem_sfb_addr + stage * 2048 + (unsigned int)(row_w_5 / 8 * 256) + 96 + (unsigned int)(row_w_5 % 8 * 4)));
                            }
                            words[1] = 0;
                            int row_w_6 = lane_0 + 32;
                            if (row_w_6 < 64) {
                                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&words[1])) : "r"(smem_sfb_addr + stage * 2048 + (unsigned int)(row_w_6 / 8 * 256) + 96 + (unsigned int)(row_w_6 % 8 * 4)));
                            }
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x2.b32"
                                " [%0], {%1, %2};"
                                :: "r"(taddr + 224 + stage * 16 + 6), "r"(words[0]), "r"(words[1]));
                            words[0] = 0;
                            int row_w_7 = lane_0;
                            if (row_w_7 < 64) {
                                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&words[0])) : "r"(smem_sfb_addr + stage * 2048 + (unsigned int)(row_w_7 / 8 * 256) + 128 + (unsigned int)(row_w_7 % 8 * 4)));
                            }
                            words[1] = 0;
                            int row_w_8 = lane_0 + 32;
                            if (row_w_8 < 64) {
                                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&words[1])) : "r"(smem_sfb_addr + stage * 2048 + (unsigned int)(row_w_8 / 8 * 256) + 128 + (unsigned int)(row_w_8 % 8 * 4)));
                            }
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x2.b32"
                                " [%0], {%1, %2};"
                                :: "r"(taddr + 224 + stage * 16 + 8), "r"(words[0]), "r"(words[1]));
                            words[0] = 0;
                            int row_w_9 = lane_0;
                            if (row_w_9 < 64) {
                                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&words[0])) : "r"(smem_sfb_addr + stage * 2048 + (unsigned int)(row_w_9 / 8 * 256) + 160 + (unsigned int)(row_w_9 % 8 * 4)));
                            }
                            words[1] = 0;
                            int row_w_10 = lane_0 + 32;
                            if (row_w_10 < 64) {
                                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&words[1])) : "r"(smem_sfb_addr + stage * 2048 + (unsigned int)(row_w_10 / 8 * 256) + 160 + (unsigned int)(row_w_10 % 8 * 4)));
                            }
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x2.b32"
                                " [%0], {%1, %2};"
                                :: "r"(taddr + 224 + stage * 16 + 10), "r"(words[0]), "r"(words[1]));
                            words[0] = 0;
                            int row_w_11 = lane_0;
                            if (row_w_11 < 64) {
                                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&words[0])) : "r"(smem_sfb_addr + stage * 2048 + (unsigned int)(row_w_11 / 8 * 256) + 192 + (unsigned int)(row_w_11 % 8 * 4)));
                            }
                            words[1] = 0;
                            int row_w_12 = lane_0 + 32;
                            if (row_w_12 < 64) {
                                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&words[1])) : "r"(smem_sfb_addr + stage * 2048 + (unsigned int)(row_w_12 / 8 * 256) + 192 + (unsigned int)(row_w_12 % 8 * 4)));
                            }
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x2.b32"
                                " [%0], {%1, %2};"
                                :: "r"(taddr + 224 + stage * 16 + 12), "r"(words[0]), "r"(words[1]));
                            words[0] = 0;
                            int row_w_13 = lane_0;
                            if (row_w_13 < 64) {
                                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&words[0])) : "r"(smem_sfb_addr + stage * 2048 + (unsigned int)(row_w_13 / 8 * 256) + 224 + (unsigned int)(row_w_13 % 8 * 4)));
                            }
                            words[1] = 0;
                            int row_w_14 = lane_0 + 32;
                            if (row_w_14 < 64) {
                                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&words[1])) : "r"(smem_sfb_addr + stage * 2048 + (unsigned int)(row_w_14 / 8 * 256) + 224 + (unsigned int)(row_w_14 % 8 * 4)));
                            }
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x2.b32"
                                " [%0], {%1, %2};"
                                :: "r"(taddr + 224 + stage * 16 + 14), "r"(words[0]), "r"(words[1]));
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            asm volatile("barrier.sync 4, 128;" ::: "memory");
                            if (warp == 4) {
                                if (elect_sync()) {
                                    mbarrier_arrive(tmem_sfb_full_addr + (stage) * 8);
                                }
                            }
                            if (elect_sync()) {
                                mbarrier_arrive(sfb_free_addr + (stage) * 8);
                            }
                            stage += 1;
                            if (stage == 3) { stage = 0; _phase_sfb_full ^= 1; _phase_k_done ^= 1; }
                        }
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_1) * 8, _phase_work_full_1);
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
                valid_1 = _clc_valid_5;
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
                mbarrier_arrive(work_empty_addr + (work_stage_1) * 8);
                work_stage_1 += 1;
                if (work_stage_1 == 3) { work_stage_1 = 0; _phase_work_full_1 ^= 1; }
                unsigned int valid_0_1 = valid_1;
                m_tile_1 = next_x_1;
                n_tile_1 = next_y_1;
                if (valid_0_1 == 0) {
                    break;
                }
            }
        }
    }
    // ---- Role: load_b ----
    if (warp == 8) {
        { // load_b_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            int tile_count_2 = num_non_exiting_ctas[0];
            unsigned int stage_1 = 0;
            unsigned int work_stage_2 = 0;
            unsigned int m_tile_2 = blockIdx.x;
            unsigned int n_tile_2 = blockIdx.y;
            unsigned int _phase_k_done_1 = 1;
            unsigned int _phase_work_full_2 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_2 = 0; _tile_iter_2 < grid_m * grid_n; _tile_iter_2++) {
                int valid_rows_1 = (unsigned int)tile_mn_limit[n_tile_2] - n_tile_2 * 64;
                if (tile_count_2 > (int)n_tile_2) {
                    if (valid_rows_1 > 0) {
                        int padding_rows_1 = (64 - valid_rows_1 % 64) % 64;
                        #pragma unroll 1
                        for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                            mbarrier_wait(k_done_addr + (stage_1) * 8, _phase_k_done_1);
                            if (elect_sync()) {
                                tma_4d_gmem2smem(smem_b_addr + stage_1 * 16384, (&B), iter_k * 256, padding_rows_1, 1073741824, n_tile_2 * 64 - (unsigned int)padding_rows_1 + 1073741824, b_full_addr + (stage_1) * 8);
                                tma_4d_gmem2smem(smem_b_addr + stage_1 * 16384 + 8192, (&B), iter_k * 256 + 128, padding_rows_1, 1073741824, n_tile_2 * 64 - (unsigned int)padding_rows_1 + 1073741824, b_full_addr + (stage_1) * 8);
                                mbarrier_arrive_expect_tx(b_full_addr + (stage_1) * 8, 16384);
                            }
                            stage_1 += 1;
                            if (stage_1 == 3) { stage_1 = 0; _phase_k_done_1 ^= 1; }
                        }
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_2) * 8, _phase_work_full_2);
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
                valid_2 = _clc_valid_1;
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
                mbarrier_arrive(work_empty_addr + (work_stage_2) * 8);
                work_stage_2 += 1;
                if (work_stage_2 == 3) { work_stage_2 = 0; _phase_work_full_2 ^= 1; }
                unsigned int valid_0_2 = valid_2;
                m_tile_2 = next_x_2;
                n_tile_2 = next_y_2;
                if (valid_0_2 == 0) {
                    break;
                }
            }
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
    }
    // ---- Role: load_sfb ----
    if (warp == 9) {
        { // load_sfb_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            int tile_count_3 = num_non_exiting_ctas[0];
            unsigned int stage_2 = 0;
            unsigned int work_stage_3 = 0;
            unsigned int m_tile_3 = blockIdx.x;
            unsigned int n_tile_3 = blockIdx.y;
            unsigned int _phase_sfb_free = 1;
            unsigned int _phase_work_full_3 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_3 = 0; _tile_iter_3 < grid_m * grid_n; _tile_iter_3++) {
                if (tile_count_3 > (int)n_tile_3) {
                    int valid_rows_sfb = (unsigned int)tile_mn_limit[n_tile_3] - n_tile_3 * 64;
                    if (valid_rows_sfb > 0) {
                        #pragma unroll 1
                        for (int iter_k_1 = 0; iter_k_1 < K_tiles; iter_k_1++) {
                            mbarrier_wait(sfb_free_addr + (stage_2) * 8, _phase_sfb_free);
                            if (elect_sync()) {
                                tma_3d_gmem2smem(smem_sfb_addr + stage_2 * 2048, (&SFB), 0, iter_k_1 * 8, n_tile_3 * 8, sfb_full_addr + (stage_2) * 8);
                                mbarrier_arrive_expect_tx(sfb_full_addr + (stage_2) * 8, 2048);
                            }
                            stage_2 += 1;
                            if (stage_2 == 3) { stage_2 = 0; _phase_sfb_free ^= 1; }
                        }
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_3) * 8, _phase_work_full_3);
                unsigned int valid_3 = 0;
                unsigned int next_x_3 = 0;
                unsigned int next_y_3 = 0;
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
                valid_3 = _clc_valid_2;
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
                next_x_3 = _clc_ctaid_4;
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
                next_y_3 = _clc_ctaid_5;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage_3) * 8);
                work_stage_3 += 1;
                if (work_stage_3 == 3) { work_stage_3 = 0; _phase_work_full_3 ^= 1; }
                unsigned int valid_0_3 = valid_3;
                m_tile_3 = next_x_3;
                n_tile_3 = next_y_3;
                if (valid_0_3 == 0) {
                    break;
                }
            }
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
    }
    // ---- Role: load_a ----
    if (warp == 10) {
        { // load_a_main
            int tile_count_4 = num_non_exiting_ctas[0];
            unsigned int stage_3 = 0;
            unsigned int work_stage_4 = 0;
            unsigned int throttle_stage = 0;
            unsigned int m_tile_4 = blockIdx.x;
            unsigned int n_tile_4 = blockIdx.y;
            unsigned int _phase_throttle_empty = 1;
            unsigned int _phase_k_done_2 = 1;
            unsigned int _phase_work_full_4 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_4 = 0; _tile_iter_4 < grid_m * grid_n; _tile_iter_4++) {
                if (tile_count_4 > (int)n_tile_4) {
                    int expert = tile_expert[n_tile_4];
                    mbarrier_wait(throttle_empty_addr + (throttle_stage) * 8, _phase_throttle_empty);
                    mbarrier_arrive(throttle_full_addr + (throttle_stage) * 8);
                    throttle_stage += 1;
                    if (throttle_stage == 3) { throttle_stage = 0; _phase_throttle_empty ^= 1; }
                    int valid_rows_a = (unsigned int)tile_mn_limit[n_tile_4] - n_tile_4 * 64;
                    if (valid_rows_a > 0) {
                        #pragma unroll 1
                        for (int iter_k_2 = 0; iter_k_2 < K_tiles; iter_k_2++) {
                            mbarrier_wait(k_done_addr + (stage_3) * 8, _phase_k_done_2);
                            if (elect_sync()) {
                                tma_4d_gmem2smem(smem_a_addr + stage_3 * 32768, (&A), 0, m_tile_4 * 128, iter_k_2 * 2, expert, a_full_addr + (stage_3) * 8);
                                tma_4d_gmem2smem(smem_a_addr + stage_3 * 32768 + 16384, (&A), 0, m_tile_4 * 128, iter_k_2 * 2 + 1, expert, a_full_addr + (stage_3) * 8);
                                mbarrier_arrive_expect_tx(a_full_addr + (stage_3) * 8, 32768);
                            }
                            stage_3 += 1;
                            if (stage_3 == 3) { stage_3 = 0; _phase_k_done_2 ^= 1; }
                        }
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_4) * 8, _phase_work_full_4);
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
                valid_4 = _clc_valid_0;
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
                mbarrier_arrive(work_empty_addr + (work_stage_4) * 8);
                work_stage_4 += 1;
                if (work_stage_4 == 3) { work_stage_4 = 0; _phase_work_full_4 ^= 1; }
                unsigned int valid_0_4 = valid_4;
                m_tile_4 = next_x_4;
                n_tile_4 = next_y_4;
                if (valid_0_4 == 0) {
                    break;
                }
            }
        }
    }
    // ---- Role: load_sfa ----
    if (warp == 11) {
        { // load_sfa_main
            int tile_count_5 = num_non_exiting_ctas[0];
            unsigned int stage_4 = 0;
            unsigned int work_stage_5 = 0;
            unsigned int m_tile_5 = blockIdx.x;
            unsigned int n_tile_5 = blockIdx.y;
            unsigned int _phase_sfa_free = 1;
            unsigned int _phase_work_full_5 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_5 = 0; _tile_iter_5 < grid_m * grid_n; _tile_iter_5++) {
                int expert_sfa = tile_expert[n_tile_5];
                if (tile_count_5 > (int)n_tile_5) {
                    int valid_rows_sfa = (unsigned int)tile_mn_limit[n_tile_5] - n_tile_5 * 64;
                    if (valid_rows_sfa > 0) {
                        #pragma unroll 1
                        for (int iter_k_3 = 0; iter_k_3 < K_tiles; iter_k_3++) {
                            mbarrier_wait(sfa_free_addr + (stage_4) * 8, _phase_sfa_free);
                            if (elect_sync()) {
                                tma_4d_gmem2smem(smem_sfa_addr + stage_4 * 4096, (&SFA), 0, 0, iter_k_3 * 8, (unsigned int)(expert_sfa * grid_m) + m_tile_5, sfa_full_addr + (stage_4) * 8);
                                mbarrier_arrive_expect_tx(sfa_full_addr + (stage_4) * 8, 4096);
                            }
                            stage_4 += 1;
                            if (stage_4 == 3) { stage_4 = 0; _phase_sfa_free ^= 1; }
                        }
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_5) * 8, _phase_work_full_5);
                unsigned int valid_5 = 0;
                unsigned int next_x_5 = 0;
                unsigned int next_y_5 = 0;
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
                    : "r"(work_response_addr + work_stage_5 * 16 + 0 * 16)
                    : "memory");
                valid_5 = _clc_valid_3;
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
                    : "r"(work_response_addr + work_stage_5 * 16 + 0 * 16)
                    : "memory");
                next_x_5 = _clc_ctaid_6;
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
                    : "r"(work_response_addr + work_stage_5 * 16 + 0 * 16)
                    : "memory");
                next_y_5 = _clc_ctaid_7;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage_5) * 8);
                work_stage_5 += 1;
                if (work_stage_5 == 3) { work_stage_5 = 0; _phase_work_full_5 ^= 1; }
                unsigned int valid_0_5 = valid_5;
                m_tile_5 = next_x_5;
                n_tile_5 = next_y_5;
                if (valid_0_5 == 0) {
                    break;
                }
            }
        }
    }
    // ---- Role: copy_sfa ----
    if (warp == 12) {
        { // copy_sfa_main
            int tile_count_6 = num_non_exiting_ctas[0];
            unsigned int stage_5 = 0;
            unsigned int work_stage_6 = 0;
            unsigned int m_tile_6 = blockIdx.x;
            unsigned int n_tile_6 = blockIdx.y;
            unsigned int _phase_sfa_full = 0;
            unsigned int _phase_k_done_3 = 1;
            unsigned int _phase_work_full_6 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_6 = 0; _tile_iter_6 < grid_m * grid_n; _tile_iter_6++) {
                if (tile_count_6 > (int)n_tile_6) {
                    int valid_rows_csfa = (unsigned int)tile_mn_limit[n_tile_6] - n_tile_6 * 64;
                    if (valid_rows_csfa > 0) {
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
                                        "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                        :: "r"((uint32_t)((unsigned int)tmem_sfa + stage_5 * 32)), "l"(_tcgen05_cp_desc_0)
                                        : "memory");
                                }
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                #endif
                                {
                                    uint64_t _tcgen05_cp_desc_1 = ((((uint64_t)(smem_sfa_addr + stage_5 * 4096 + 512)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                    asm volatile(
                                        "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                        :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_5 * 32 + 4))), "l"(_tcgen05_cp_desc_1)
                                        : "memory");
                                }
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                #endif
                                {
                                    uint64_t _tcgen05_cp_desc_2 = ((((uint64_t)(smem_sfa_addr + stage_5 * 4096 + 1024)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                    asm volatile(
                                        "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                        :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_5 * 32 + 8))), "l"(_tcgen05_cp_desc_2)
                                        : "memory");
                                }
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                #endif
                                {
                                    uint64_t _tcgen05_cp_desc_3 = ((((uint64_t)(smem_sfa_addr + stage_5 * 4096 + 1536)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                    asm volatile(
                                        "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                        :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_5 * 32 + 12))), "l"(_tcgen05_cp_desc_3)
                                        : "memory");
                                }
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                #endif
                                {
                                    uint64_t _tcgen05_cp_desc_4 = ((((uint64_t)(smem_sfa_addr + stage_5 * 4096 + 2048)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                    asm volatile(
                                        "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                        :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_5 * 32 + 16))), "l"(_tcgen05_cp_desc_4)
                                        : "memory");
                                }
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                #endif
                                {
                                    uint64_t _tcgen05_cp_desc_5 = ((((uint64_t)(smem_sfa_addr + stage_5 * 4096 + 2560)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                    asm volatile(
                                        "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                        :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_5 * 32 + 20))), "l"(_tcgen05_cp_desc_5)
                                        : "memory");
                                }
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                #endif
                                {
                                    uint64_t _tcgen05_cp_desc_6 = ((((uint64_t)(smem_sfa_addr + stage_5 * 4096 + 3072)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                    asm volatile(
                                        "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                        :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_5 * 32 + 24))), "l"(_tcgen05_cp_desc_6)
                                        : "memory");
                                }
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                #endif
                                {
                                    uint64_t _tcgen05_cp_desc_7 = ((((uint64_t)(smem_sfa_addr + stage_5 * 4096 + 3584)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                    asm volatile(
                                        "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                        :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_5 * 32 + 28))), "l"(_tcgen05_cp_desc_7)
                                        : "memory");
                                }
                            }
                            elect_commit2(tmem_sfa_full_addr + (stage_5) * 8, sfa_free_addr + (stage_5) * 8);
                            stage_5 += 1;
                            if (stage_5 == 3) { stage_5 = 0; _phase_sfa_full ^= 1; _phase_k_done_3 ^= 1; }
                        }
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_6) * 8, _phase_work_full_6);
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
                valid_6 = _clc_valid_4;
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
                mbarrier_arrive(work_empty_addr + (work_stage_6) * 8);
                work_stage_6 += 1;
                if (work_stage_6 == 3) { work_stage_6 = 0; _phase_work_full_6 ^= 1; }
                unsigned int valid_0_6 = valid_6;
                m_tile_6 = next_x_6;
                n_tile_6 = next_y_6;
                if (valid_0_6 == 0) {
                    break;
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 13) {
        { // mma_main
            int tile_count_7 = num_non_exiting_ctas[0];
            unsigned int stage_6 = 0;
            unsigned int acc_stage_1 = 0;
            unsigned int work_stage_7 = 0;
            unsigned int m_tile_7 = blockIdx.x;
            unsigned int n_tile_7 = blockIdx.y;
            unsigned int _phase_mma_free = 1;
            unsigned int _phase_a_full = 0;
            unsigned int _phase_b_full = 0;
            unsigned int _phase_tmem_sfa_full = 0;
            unsigned int _phase_tmem_sfb_full = 0;
            unsigned int _phase_work_full_7 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_7 = 0; _tile_iter_7 < grid_m * grid_n; _tile_iter_7++) {
                if (tile_count_7 > (int)n_tile_7) {
                    int valid_rows_m = (unsigned int)tile_mn_limit[n_tile_7] - n_tile_7 * 64;
                    if (valid_rows_m > 0) {
                        mbarrier_wait(mma_free_addr + (acc_stage_1) * 8, _phase_mma_free);
                        #pragma unroll 1
                        for (int iter_k_4 = 0; iter_k_4 < K_tiles; iter_k_4++) {
                            mbarrier_wait(a_full_addr + (stage_6) * 8, _phase_a_full);
                            mbarrier_wait(b_full_addr + (stage_6) * 8, _phase_b_full);
                            mbarrier_wait(tmem_sfa_full_addr + (stage_6) * 8, _phase_tmem_sfa_full);
                            mbarrier_wait(tmem_sfb_full_addr + (stage_6) * 8, _phase_tmem_sfb_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_a_lo_0 = make_warp_uniform((((smem_a_addr) >> 4) & 0x3FFF) + (stage_6) * 2048);
                            int _mma_b_lo_0 = make_warp_uniform((((smem_b_addr) >> 4) & 0x3FFF) + (stage_6) * 1024);
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs((tmem_accum + (acc_stage_1 * 64)), a_desc + 0, b_desc + 0,
                                        0x8100480U, (unsigned int)tmem_sfa + stage_6 * 32 + 0, (unsigned int)tmem_sfb + stage_6 * 16 + 0, ((((1) ? ((iter_k_4 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                                }
                            }
                            int _mma_a_lo_1 = make_warp_uniform((((smem_a_addr + 32) >> 4) & 0x3FFF) + (stage_6) * 2048);
                            int _mma_b_lo_1 = make_warp_uniform((((smem_b_addr + 32) >> 4) & 0x3FFF) + (stage_6) * 1024);
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs((tmem_accum + (acc_stage_1 * 64)), a_desc + 0, b_desc + 0,
                                        0x8100480U, (unsigned int)tmem_sfa + (stage_6 * 32 + 4) + 0, (unsigned int)tmem_sfb + (stage_6 * 16 + 2) + 0, ((((0) ? ((iter_k_4 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                                }
                            }
                            int _mma_a_lo_2 = make_warp_uniform((((smem_a_addr + 64) >> 4) & 0x3FFF) + (stage_6) * 2048);
                            int _mma_b_lo_2 = make_warp_uniform((((smem_b_addr + 64) >> 4) & 0x3FFF) + (stage_6) * 1024);
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_2) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_2) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs((tmem_accum + (acc_stage_1 * 64)), a_desc + 0, b_desc + 0,
                                        0x8100480U, (unsigned int)tmem_sfa + (stage_6 * 32 + 8) + 0, (unsigned int)tmem_sfb + (stage_6 * 16 + 4) + 0, ((((0) ? ((iter_k_4 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                                }
                            }
                            int _mma_a_lo_3 = make_warp_uniform((((smem_a_addr + 96) >> 4) & 0x3FFF) + (stage_6) * 2048);
                            int _mma_b_lo_3 = make_warp_uniform((((smem_b_addr + 96) >> 4) & 0x3FFF) + (stage_6) * 1024);
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_3) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_3) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs((tmem_accum + (acc_stage_1 * 64)), a_desc + 0, b_desc + 0,
                                        0x8100480U, (unsigned int)tmem_sfa + (stage_6 * 32 + 12) + 0, (unsigned int)tmem_sfb + (stage_6 * 16 + 6) + 0, ((((0) ? ((iter_k_4 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                                }
                            }
                            int _mma_a_lo_4 = make_warp_uniform((((smem_a_addr + 16384) >> 4) & 0x3FFF) + (stage_6) * 2048);
                            int _mma_b_lo_4 = make_warp_uniform((((smem_b_addr + 8192) >> 4) & 0x3FFF) + (stage_6) * 1024);
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_4) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_4) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs((tmem_accum + (acc_stage_1 * 64)), a_desc + 0, b_desc + 0,
                                        0x8100480U, (unsigned int)tmem_sfa + (stage_6 * 32 + 16) + 0, (unsigned int)tmem_sfb + (stage_6 * 16 + 8) + 0, ((((0) ? ((iter_k_4 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                                }
                            }
                            int _mma_a_lo_5 = make_warp_uniform((((smem_a_addr + 16416) >> 4) & 0x3FFF) + (stage_6) * 2048);
                            int _mma_b_lo_5 = make_warp_uniform((((smem_b_addr + 8224) >> 4) & 0x3FFF) + (stage_6) * 1024);
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_5) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_5) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs((tmem_accum + (acc_stage_1 * 64)), a_desc + 0, b_desc + 0,
                                        0x8100480U, (unsigned int)tmem_sfa + (stage_6 * 32 + 20) + 0, (unsigned int)tmem_sfb + (stage_6 * 16 + 10) + 0, ((((0) ? ((iter_k_4 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                                }
                            }
                            int _mma_a_lo_6 = make_warp_uniform((((smem_a_addr + 16448) >> 4) & 0x3FFF) + (stage_6) * 2048);
                            int _mma_b_lo_6 = make_warp_uniform((((smem_b_addr + 8256) >> 4) & 0x3FFF) + (stage_6) * 1024);
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_6) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_6) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs((tmem_accum + (acc_stage_1 * 64)), a_desc + 0, b_desc + 0,
                                        0x8100480U, (unsigned int)tmem_sfa + (stage_6 * 32 + 24) + 0, (unsigned int)tmem_sfb + (stage_6 * 16 + 12) + 0, ((((0) ? ((iter_k_4 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                                }
                            }
                            int _mma_a_lo_7 = make_warp_uniform((((smem_a_addr + 16480) >> 4) & 0x3FFF) + (stage_6) * 2048);
                            int _mma_b_lo_7 = make_warp_uniform((((smem_b_addr + 8288) >> 4) & 0x3FFF) + (stage_6) * 1024);
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_7) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_7) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs((tmem_accum + (acc_stage_1 * 64)), a_desc + 0, b_desc + 0,
                                        0x8100480U, (unsigned int)tmem_sfa + (stage_6 * 32 + 28) + 0, (unsigned int)tmem_sfb + (stage_6 * 16 + 14) + 0, ((((0) ? ((iter_k_4 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                                }
                            }
                            elect_commit(k_done_addr + (stage_6) * 8);
                            if (iter_k_4 + 1 == K_tiles) {
                                elect_commit(mma_full_addr + (acc_stage_1) * 8);
                            }
                            stage_6 += 1;
                            if (stage_6 == 3) { stage_6 = 0; _phase_a_full ^= 1; _phase_b_full ^= 1; _phase_tmem_sfa_full ^= 1; _phase_tmem_sfb_full ^= 1; }
                        }
                        acc_stage_1 += 1;
                        if (acc_stage_1 == 2) { acc_stage_1 = 0; _phase_mma_free ^= 1; }
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_7) * 8, _phase_work_full_7);
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
                valid_7 = _clc_valid_6;
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
                mbarrier_arrive(work_empty_addr + (work_stage_7) * 8);
                work_stage_7 += 1;
                if (work_stage_7 == 3) { work_stage_7 = 0; _phase_work_full_7 ^= 1; }
                unsigned int valid_0_7 = valid_7;
                m_tile_7 = next_x_7;
                n_tile_7 = next_y_7;
                if (valid_0_7 == 0) {
                    break;
                }
            }
        }
    }
    // ---- Role: work_id ----
    if (warp == 14) {
        { // work_id_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            int tile_count_8 = num_non_exiting_ctas[0];
            unsigned int work_stage_8 = 0;
            unsigned int throttle_stage_1 = 0;
            unsigned int m_tile_8 = blockIdx.x;
            unsigned int n_tile_8 = blockIdx.y;
            unsigned int _phase_throttle_full = 0;
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_work_full_8 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_8 = 0; _tile_iter_8 < grid_m * grid_n; _tile_iter_8++) {
                if (tile_count_8 > (int)n_tile_8) {
                    mbarrier_wait(throttle_full_addr + (throttle_stage_1) * 8, _phase_throttle_full);
                    mbarrier_arrive(throttle_empty_addr + (throttle_stage_1) * 8);
                    throttle_stage_1 += 1;
                    if (throttle_stage_1 == 3) { throttle_stage_1 = 0; _phase_throttle_full ^= 1; }
                } else {
                    if (elect_sync()) {
                        mbarrier_init(fast_ready_addr, 1);
                        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
                    }
                    __syncwarp();
                    unsigned int fast_stage = 0;
                    unsigned int fast_phase = 0;
                    #pragma unroll 1
                    for (unsigned int _drain_iter = 0; _drain_iter < grid_m * grid_n; _drain_iter++) {
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
                        mbarrier_wait(fast_ready_addr + (fast_stage) * 8, fast_phase);
                        unsigned int canceled = 0;
                        for (int slot_idx_1 = 0; slot_idx_1 < 4; slot_idx_1++) {
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
                                : "r"(fast_response_addr + fast_stage * 64 + slot_idx_1 * 16)
                                : "memory");
                            canceled += _clc_valid_8;
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        fast_phase ^= 1;
                        if (canceled == 0) {
                            break;
                        }
                    }
                }
                if (elect_sync()) {
                    mbarrier_wait(work_empty_addr + (work_stage_8) * 8, _phase_work_empty);
                    mbarrier_arrive_expect_tx(work_full_addr + (work_stage_8) * 8, 16);
                    asm volatile(
                        "fence.proxy.async.shared::cta;\n\t"
                        "clusterlaunchcontrol.try_cancel.async.shared::cta"
                            ".mbarrier::complete_tx::bytes.b128"
                            " [%0], [%1];"
                        :: "r"(work_response_addr + work_stage_8 * 16 + 0 * 16), "r"(work_full_addr + work_stage_8 * 8)
                        : "memory");
                }
                __syncwarp();
                mbarrier_wait(work_full_addr + (work_stage_8) * 8, _phase_work_full_8);
                unsigned int valid_8 = 0;
                unsigned int next_x_8 = 0;
                unsigned int next_y_8 = 0;
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
                    : "r"(work_response_addr + work_stage_8 * 16 + 0 * 16)
                    : "memory");
                valid_8 = _clc_valid_9;
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
                mbarrier_arrive(work_empty_addr + (work_stage_8) * 8);
                work_stage_8 += 1;
                if (work_stage_8 == 3) { work_stage_8 = 0; _phase_work_empty ^= 1; _phase_work_full_8 ^= 1; }
                unsigned int valid_0_8 = valid_8;
                m_tile_8 = next_x_8;
                n_tile_8 = next_y_8;
                if (valid_0_8 == 0) {
                    break;
                }
            }
            #pragma unroll 1
            for (unsigned int _tail_iter = 0; _tail_iter < 3; _tail_iter++) {
                mbarrier_wait(work_empty_addr + (work_stage_8) * 8, _phase_work_empty);
                work_stage_8 += 1;
                if (work_stage_8 == 3) { work_stage_8 = 0; _phase_work_empty ^= 1; _phase_work_full_8 ^= 1; }
            }
        }
    }
    // ---- Role: padding ----
    if (warp == 15) {
        { // padding_main
            unsigned int work_stage_9 = 0;
            unsigned int m_tile_9 = blockIdx.x;
            unsigned int n_tile_9 = blockIdx.y;
            unsigned int _phase_work_full_9 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_9 = 0; _tile_iter_9 < grid_m * grid_n; _tile_iter_9++) {
                mbarrier_wait(work_full_addr + (work_stage_9) * 8, _phase_work_full_9);
                unsigned int valid_9 = 0;
                unsigned int next_x_9 = 0;
                unsigned int next_y_9 = 0;
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
                    : "r"(work_response_addr + work_stage_9 * 16 + 0 * 16)
                    : "memory");
                valid_9 = _clc_valid_10;
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
                mbarrier_arrive(work_empty_addr + (work_stage_9) * 8);
                work_stage_9 += 1;
                if (work_stage_9 == 3) { work_stage_9 = 0; _phase_work_full_9 ^= 1; }
                unsigned int valid_0_9 = valid_9;
                m_tile_9 = next_x_9;
                n_tile_9 = next_y_9;
                if (valid_0_9 == 0) {
                    break;
                }
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
