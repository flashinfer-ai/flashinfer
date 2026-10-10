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
#define TMEM_NCOLS 496
#define TMEM_ACCUM_OFFSET 0
#define TMEM_SFA_OFFSET 448
#define TMEM_SFB_OFFSET 464
#define NUM_K_PIPE_STAGES 3
#define NUM_MMA_PIPE_STAGES 1
#define NUM_WORK_PIPE_STAGES 3
#define NUM_THROTTLE_PIPE_STAGES 3
#define NUM_FAST_PIPE_STAGES 1
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 16384
#define SMEM_SMEM_B_OFF 50176
#define SMEM_SMEM_B_STAGE_BYTES 32768
#define SMEM_SMEM_B_STRIDE 32768
#define SMEM_EPI_STAGING_OFF 148480
#define SMEM_EPI_STAGING_STAGE_BYTES 16384
#define SMEM_EPI_STAGING_STRIDE 16384
#define SMEM_EPI_STAGING_U64_OFF 148480
#define SMEM_EPI_STAGING_U64_STAGE_BYTES 16384
#define SMEM_EPI_STAGING_U64_STRIDE 16384
#define SMEM_TOK_STAGE_OFF 164864
#define SMEM_TOK_STAGE_STAGE_BYTES 1024
#define SMEM_TOK_STAGE_STRIDE 1024
#define SMEM_SMEM_SFA_OFF 165888
#define SMEM_SMEM_SFA_STAGE_BYTES 2048
#define SMEM_SMEM_SFA_STRIDE 2048
#define SMEM_SMEM_SFB_OFF 172032
#define SMEM_SMEM_SFB_STAGE_BYTES 4096
#define SMEM_SMEM_SFB_STRIDE 4096
#define SMEM_WORK_RESPONSE_OFF 184320
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_FAST_RESPONSE_OFF 184368
#define SMEM_FAST_RESPONSE_STAGE_BYTES 64
#define SMEM_FAST_RESPONSE_STRIDE 64
#define SMEM_TOTAL 184448
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

__global__ __launch_bounds__(384, LAUNCH_MIN_BLOCKS) void
kernel_cake_stepfun_moe_598eb684b8fafcde47f4(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, const __grid_constant__ CUtensorMap C, int* __restrict__ tile_expert, int* __restrict__ tile_mn_limit, int* __restrict__ num_non_exiting_ctas, float* __restrict__ scale_c, int M_out, int K, int grid_m, int grid_n, int K_tiles, float* __restrict__ token_sf)
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
    #define a_empty_addr (mbar_base + 96)
    #define b_empty_addr (mbar_base + 120)
    #define sfa_empty_addr (mbar_base + 144)
    #define sfb_empty_addr (mbar_base + 168)
    #define mma_full_addr (mbar_base + 192)
    #define mma_free_addr (mbar_base + 200)
    #define work_full_addr (mbar_base + 208)
    #define work_empty_addr (mbar_base + 232)
    #define throttle_full_addr (mbar_base + 256)
    #define throttle_empty_addr (mbar_base + 280)
    #define fast_ready_addr (mbar_base + 304)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + 50176);
    const int smem_b_addr = smem + 50176;
    __nv_bfloat16* epi_staging = reinterpret_cast<__nv_bfloat16*>(smem_raw + 148480);
    const int epi_staging_addr = smem + 148480;
    unsigned long long* epi_staging_u64 = reinterpret_cast<unsigned long long*>(smem_raw + 148480);
    const int epi_staging_u64_addr = smem + 148480;
    float* tok_stage = reinterpret_cast<float*>(smem_raw + 164864);
    const int tok_stage_addr = smem + 164864;
    uint8_t* smem_sfa = reinterpret_cast<uint8_t*>(smem_raw + 165888);
    const int smem_sfa_addr = smem + 165888;
    uint8_t* smem_sfb = reinterpret_cast<uint8_t*>(smem_raw + 172032);
    const int smem_sfb_addr = smem + 172032;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 184320);
    const int work_response_addr = smem + 184320;
    unsigned int* fast_response = reinterpret_cast<unsigned int*>(smem_raw + 184368);
    const int fast_response_addr = smem + 184368;

    // Mbarrier init (15 pipeline groups, 0 ordered-sequence groups, 39 barriers)
    // Mbarriers at smem_raw[0..312)

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
            // a_empty: 3 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            // b_empty: 3 barriers, init_count=1
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            // sfa_empty: 3 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            // sfb_empty: 3 barriers, init_count=1
            mbarrier_init(smem + 168, 1);
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            // --- pipeline 'mma_pipe' ---
            // mma_full: 1 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            // mma_free: 1 barriers, init_count=128
            mbarrier_init(smem + 200, 128);
            // --- pipeline 'work_pipe' ---
            // work_full: 3 barriers, init_count=1
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            // work_empty: 3 barriers, init_count=384
            mbarrier_init(smem + 232, 384);
            mbarrier_init(smem + 240, 384);
            mbarrier_init(smem + 248, 384);
            // --- pipeline 'throttle_pipe' ---
            // throttle_full: 3 barriers, init_count=32
            mbarrier_init(smem + 256, 32);
            mbarrier_init(smem + 264, 32);
            mbarrier_init(smem + 272, 32);
            // throttle_empty: 3 barriers, init_count=32
            mbarrier_init(smem + 280, 32);
            mbarrier_init(smem + 288, 32);
            mbarrier_init(smem + 296, 32);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 496 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 312);
    if (warp == 0) {
        int _tmem_hold = smem + 312;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
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
            int epi_thread = warp_0 * 32 + lane_1;
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
            for (unsigned int _tile_iter = 0; _tile_iter < grid_m * grid_n; _tile_iter++) {
                int valid_rows = (unsigned int)tile_mn_limit[n_tile] - n_tile * 256;
                if (tile_count > (int)n_tile) {
                    if (valid_rows > 0) {
                        int expert_e = tile_expert[n_tile];
                        float sc = scale_c[expert_e];
                        int tok_base = (int)n_tile * 256;
                        if (epi_thread + tok_base < tile_mn_limit[n_tile]) {
                            tok_stage[epi_thread] = token_sf[tok_base + epi_thread];
                        }
                        if (epi_thread + 128 + tok_base < tile_mn_limit[n_tile]) {
                            tok_stage[epi_thread + 128] = token_sf[tok_base + 128 + epi_thread];
                        }
                        asm volatile("barrier.sync 8, 128;" ::: "memory");
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
                            mbarrier_arrive(mma_free_addr);
                        }
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                        asm volatile("barrier.sync 7, 128;" ::: "memory");
                        int token = base_token;
                        float tok = tok_stage[token_block * 64 + token];
                        converted[0] = frag[0] * tok * sc;
                        converted[1] = frag[2] * tok * sc;
                        converted[2] = frag[32] * tok * sc;
                        converted[3] = frag[34] * tok * sc;
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
                        float tok_1 = tok_stage[token_block * 64 + token_0];
                        converted[0] = frag[1] * tok_1 * sc;
                        converted[1] = frag[3] * tok_1 * sc;
                        converted[2] = frag[33] * tok_1 * sc;
                        converted[3] = frag[35] * tok_1 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_2 = token_0 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_0 * 64 + (base_feature ^ swizzle_feature_2)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_0 * 64 + (base_feature - 64 ^ swizzle_feature_2)) / 4] = packed_word;
                        }
                        int token_3 = base_token + 8;
                        float tok_4 = tok_stage[token_block * 64 + token_3];
                        converted[0] = frag[4] * tok_4 * sc;
                        converted[1] = frag[6] * tok_4 * sc;
                        converted[2] = frag[36] * tok_4 * sc;
                        converted[3] = frag[38] * tok_4 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_5 = token_3 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_3 * 64 + (base_feature ^ swizzle_feature_5)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_3 * 64 + (base_feature - 64 ^ swizzle_feature_5)) / 4] = packed_word;
                        }
                        int token_6 = base_token + 8 + 1;
                        float tok_7 = tok_stage[token_block * 64 + token_6];
                        converted[0] = frag[5] * tok_7 * sc;
                        converted[1] = frag[7] * tok_7 * sc;
                        converted[2] = frag[37] * tok_7 * sc;
                        converted[3] = frag[39] * tok_7 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_8 = token_6 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_6 * 64 + (base_feature ^ swizzle_feature_8)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_6 * 64 + (base_feature - 64 ^ swizzle_feature_8)) / 4] = packed_word;
                        }
                        int token_9 = base_token + 16;
                        float tok_10 = tok_stage[token_block * 64 + token_9];
                        converted[0] = frag[8] * tok_10 * sc;
                        converted[1] = frag[10] * tok_10 * sc;
                        converted[2] = frag[40] * tok_10 * sc;
                        converted[3] = frag[42] * tok_10 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_11 = token_9 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_9 * 64 + (base_feature ^ swizzle_feature_11)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_9 * 64 + (base_feature - 64 ^ swizzle_feature_11)) / 4] = packed_word;
                        }
                        int token_12 = base_token + 16 + 1;
                        float tok_13 = tok_stage[token_block * 64 + token_12];
                        converted[0] = frag[9] * tok_13 * sc;
                        converted[1] = frag[11] * tok_13 * sc;
                        converted[2] = frag[41] * tok_13 * sc;
                        converted[3] = frag[43] * tok_13 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_14 = token_12 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_12 * 64 + (base_feature ^ swizzle_feature_14)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_12 * 64 + (base_feature - 64 ^ swizzle_feature_14)) / 4] = packed_word;
                        }
                        int token_15 = base_token + 24;
                        float tok_16 = tok_stage[token_block * 64 + token_15];
                        converted[0] = frag[12] * tok_16 * sc;
                        converted[1] = frag[14] * tok_16 * sc;
                        converted[2] = frag[44] * tok_16 * sc;
                        converted[3] = frag[46] * tok_16 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_17 = token_15 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_15 * 64 + (base_feature ^ swizzle_feature_17)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_15 * 64 + (base_feature - 64 ^ swizzle_feature_17)) / 4] = packed_word;
                        }
                        int token_18 = base_token + 24 + 1;
                        float tok_19 = tok_stage[token_block * 64 + token_18];
                        converted[0] = frag[13] * tok_19 * sc;
                        converted[1] = frag[15] * tok_19 * sc;
                        converted[2] = frag[45] * tok_19 * sc;
                        converted[3] = frag[47] * tok_19 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_20 = token_18 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_18 * 64 + (base_feature ^ swizzle_feature_20)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_18 * 64 + (base_feature - 64 ^ swizzle_feature_20)) / 4] = packed_word;
                        }
                        int token_21 = base_token + 32;
                        float tok_22 = tok_stage[token_block * 64 + token_21];
                        converted[0] = frag[16] * tok_22 * sc;
                        converted[1] = frag[18] * tok_22 * sc;
                        converted[2] = frag[48] * tok_22 * sc;
                        converted[3] = frag[50] * tok_22 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_23 = token_21 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_21 * 64 + (base_feature ^ swizzle_feature_23)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_21 * 64 + (base_feature - 64 ^ swizzle_feature_23)) / 4] = packed_word;
                        }
                        int token_24 = base_token + 32 + 1;
                        float tok_25 = tok_stage[token_block * 64 + token_24];
                        converted[0] = frag[17] * tok_25 * sc;
                        converted[1] = frag[19] * tok_25 * sc;
                        converted[2] = frag[49] * tok_25 * sc;
                        converted[3] = frag[51] * tok_25 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_26 = token_24 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_24 * 64 + (base_feature ^ swizzle_feature_26)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_24 * 64 + (base_feature - 64 ^ swizzle_feature_26)) / 4] = packed_word;
                        }
                        int token_27 = base_token + 40;
                        float tok_28 = tok_stage[token_block * 64 + token_27];
                        converted[0] = frag[20] * tok_28 * sc;
                        converted[1] = frag[22] * tok_28 * sc;
                        converted[2] = frag[52] * tok_28 * sc;
                        converted[3] = frag[54] * tok_28 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_29 = token_27 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_27 * 64 + (base_feature ^ swizzle_feature_29)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_27 * 64 + (base_feature - 64 ^ swizzle_feature_29)) / 4] = packed_word;
                        }
                        int token_30 = base_token + 40 + 1;
                        float tok_31 = tok_stage[token_block * 64 + token_30];
                        converted[0] = frag[21] * tok_31 * sc;
                        converted[1] = frag[23] * tok_31 * sc;
                        converted[2] = frag[53] * tok_31 * sc;
                        converted[3] = frag[55] * tok_31 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_32 = token_30 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_30 * 64 + (base_feature ^ swizzle_feature_32)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_30 * 64 + (base_feature - 64 ^ swizzle_feature_32)) / 4] = packed_word;
                        }
                        int token_33 = base_token + 48;
                        float tok_34 = tok_stage[token_block * 64 + token_33];
                        converted[0] = frag[24] * tok_34 * sc;
                        converted[1] = frag[26] * tok_34 * sc;
                        converted[2] = frag[56] * tok_34 * sc;
                        converted[3] = frag[58] * tok_34 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_35 = token_33 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_33 * 64 + (base_feature ^ swizzle_feature_35)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_33 * 64 + (base_feature - 64 ^ swizzle_feature_35)) / 4] = packed_word;
                        }
                        int token_36 = base_token + 48 + 1;
                        float tok_37 = tok_stage[token_block * 64 + token_36];
                        converted[0] = frag[25] * tok_37 * sc;
                        converted[1] = frag[27] * tok_37 * sc;
                        converted[2] = frag[57] * tok_37 * sc;
                        converted[3] = frag[59] * tok_37 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_38 = token_36 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_36 * 64 + (base_feature ^ swizzle_feature_38)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_36 * 64 + (base_feature - 64 ^ swizzle_feature_38)) / 4] = packed_word;
                        }
                        int token_39 = base_token + 56;
                        float tok_40 = tok_stage[token_block * 64 + token_39];
                        converted[0] = frag[28] * tok_40 * sc;
                        converted[1] = frag[30] * tok_40 * sc;
                        converted[2] = frag[60] * tok_40 * sc;
                        converted[3] = frag[62] * tok_40 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_41 = token_39 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_39 * 64 + (base_feature ^ swizzle_feature_41)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_39 * 64 + (base_feature - 64 ^ swizzle_feature_41)) / 4] = packed_word;
                        }
                        int token_42 = base_token + 56 + 1;
                        float tok_43 = tok_stage[token_block * 64 + token_42];
                        converted[0] = frag[29] * tok_43 * sc;
                        converted[1] = frag[31] * tok_43 * sc;
                        converted[2] = frag[61] * tok_43 * sc;
                        converted[3] = frag[63] * tok_43 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_44 = token_42 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_42 * 64 + (base_feature ^ swizzle_feature_44)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_42 * 64 + (base_feature - 64 ^ swizzle_feature_44)) / 4] = packed_word;
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
                        int token_block_45 = 1;
                        if (epilogue_local_idx == 0) {
                            token_block_45 = 0;
                        }
                        int acc_col_46 = epilogue_local_idx * 192 + token_block_45 * 64;
                        float _tmem_load_2[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[31]))
                            : "r"(taddr + (unsigned int)row_addr + (unsigned int)acc_col_46));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        float _tmem_load_3[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[31]))
                            : "r"(taddr + (unsigned int)row_addr + 1048576 + (unsigned int)acc_col_46));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        for (int i_1 = 0; i_1 < 32; i_1++) {
                            frag[i_1] = _tmem_load_2[i_1];
                            frag[32 + i_1] = _tmem_load_3[i_1];
                        }
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                        asm volatile("barrier.sync 7, 128;" ::: "memory");
                        int token_47 = base_token;
                        float tok_48 = tok_stage[token_block_45 * 64 + token_47];
                        converted[0] = frag[0] * tok_48 * sc;
                        converted[1] = frag[2] * tok_48 * sc;
                        converted[2] = frag[32] * tok_48 * sc;
                        converted[3] = frag[34] * tok_48 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_49 = token_47 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_47 * 64 + (base_feature ^ swizzle_feature_49)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_47 * 64 + (base_feature - 64 ^ swizzle_feature_49)) / 4] = packed_word;
                        }
                        int token_50 = base_token + 1;
                        float tok_51 = tok_stage[token_block_45 * 64 + token_50];
                        converted[0] = frag[1] * tok_51 * sc;
                        converted[1] = frag[3] * tok_51 * sc;
                        converted[2] = frag[33] * tok_51 * sc;
                        converted[3] = frag[35] * tok_51 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_52 = token_50 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_50 * 64 + (base_feature ^ swizzle_feature_52)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_50 * 64 + (base_feature - 64 ^ swizzle_feature_52)) / 4] = packed_word;
                        }
                        int token_53 = base_token + 8;
                        float tok_54 = tok_stage[token_block_45 * 64 + token_53];
                        converted[0] = frag[4] * tok_54 * sc;
                        converted[1] = frag[6] * tok_54 * sc;
                        converted[2] = frag[36] * tok_54 * sc;
                        converted[3] = frag[38] * tok_54 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_55 = token_53 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_53 * 64 + (base_feature ^ swizzle_feature_55)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_53 * 64 + (base_feature - 64 ^ swizzle_feature_55)) / 4] = packed_word;
                        }
                        int token_56 = base_token + 8 + 1;
                        float tok_57 = tok_stage[token_block_45 * 64 + token_56];
                        converted[0] = frag[5] * tok_57 * sc;
                        converted[1] = frag[7] * tok_57 * sc;
                        converted[2] = frag[37] * tok_57 * sc;
                        converted[3] = frag[39] * tok_57 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_58 = token_56 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_56 * 64 + (base_feature ^ swizzle_feature_58)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_56 * 64 + (base_feature - 64 ^ swizzle_feature_58)) / 4] = packed_word;
                        }
                        int token_59 = base_token + 16;
                        float tok_60 = tok_stage[token_block_45 * 64 + token_59];
                        converted[0] = frag[8] * tok_60 * sc;
                        converted[1] = frag[10] * tok_60 * sc;
                        converted[2] = frag[40] * tok_60 * sc;
                        converted[3] = frag[42] * tok_60 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_61 = token_59 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_59 * 64 + (base_feature ^ swizzle_feature_61)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_59 * 64 + (base_feature - 64 ^ swizzle_feature_61)) / 4] = packed_word;
                        }
                        int token_62 = base_token + 16 + 1;
                        float tok_63 = tok_stage[token_block_45 * 64 + token_62];
                        converted[0] = frag[9] * tok_63 * sc;
                        converted[1] = frag[11] * tok_63 * sc;
                        converted[2] = frag[41] * tok_63 * sc;
                        converted[3] = frag[43] * tok_63 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_64 = token_62 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_62 * 64 + (base_feature ^ swizzle_feature_64)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_62 * 64 + (base_feature - 64 ^ swizzle_feature_64)) / 4] = packed_word;
                        }
                        int token_65 = base_token + 24;
                        float tok_66 = tok_stage[token_block_45 * 64 + token_65];
                        converted[0] = frag[12] * tok_66 * sc;
                        converted[1] = frag[14] * tok_66 * sc;
                        converted[2] = frag[44] * tok_66 * sc;
                        converted[3] = frag[46] * tok_66 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_67 = token_65 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_65 * 64 + (base_feature ^ swizzle_feature_67)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_65 * 64 + (base_feature - 64 ^ swizzle_feature_67)) / 4] = packed_word;
                        }
                        int token_68 = base_token + 24 + 1;
                        float tok_69 = tok_stage[token_block_45 * 64 + token_68];
                        converted[0] = frag[13] * tok_69 * sc;
                        converted[1] = frag[15] * tok_69 * sc;
                        converted[2] = frag[45] * tok_69 * sc;
                        converted[3] = frag[47] * tok_69 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_70 = token_68 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_68 * 64 + (base_feature ^ swizzle_feature_70)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_68 * 64 + (base_feature - 64 ^ swizzle_feature_70)) / 4] = packed_word;
                        }
                        int token_71 = base_token + 32;
                        float tok_72 = tok_stage[token_block_45 * 64 + token_71];
                        converted[0] = frag[16] * tok_72 * sc;
                        converted[1] = frag[18] * tok_72 * sc;
                        converted[2] = frag[48] * tok_72 * sc;
                        converted[3] = frag[50] * tok_72 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_73 = token_71 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_71 * 64 + (base_feature ^ swizzle_feature_73)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_71 * 64 + (base_feature - 64 ^ swizzle_feature_73)) / 4] = packed_word;
                        }
                        int token_74 = base_token + 32 + 1;
                        float tok_75 = tok_stage[token_block_45 * 64 + token_74];
                        converted[0] = frag[17] * tok_75 * sc;
                        converted[1] = frag[19] * tok_75 * sc;
                        converted[2] = frag[49] * tok_75 * sc;
                        converted[3] = frag[51] * tok_75 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_76 = token_74 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_74 * 64 + (base_feature ^ swizzle_feature_76)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_74 * 64 + (base_feature - 64 ^ swizzle_feature_76)) / 4] = packed_word;
                        }
                        int token_77 = base_token + 40;
                        float tok_78 = tok_stage[token_block_45 * 64 + token_77];
                        converted[0] = frag[20] * tok_78 * sc;
                        converted[1] = frag[22] * tok_78 * sc;
                        converted[2] = frag[52] * tok_78 * sc;
                        converted[3] = frag[54] * tok_78 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_79 = token_77 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_77 * 64 + (base_feature ^ swizzle_feature_79)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_77 * 64 + (base_feature - 64 ^ swizzle_feature_79)) / 4] = packed_word;
                        }
                        int token_80 = base_token + 40 + 1;
                        float tok_81 = tok_stage[token_block_45 * 64 + token_80];
                        converted[0] = frag[21] * tok_81 * sc;
                        converted[1] = frag[23] * tok_81 * sc;
                        converted[2] = frag[53] * tok_81 * sc;
                        converted[3] = frag[55] * tok_81 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_82 = token_80 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_80 * 64 + (base_feature ^ swizzle_feature_82)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_80 * 64 + (base_feature - 64 ^ swizzle_feature_82)) / 4] = packed_word;
                        }
                        int token_83 = base_token + 48;
                        float tok_84 = tok_stage[token_block_45 * 64 + token_83];
                        converted[0] = frag[24] * tok_84 * sc;
                        converted[1] = frag[26] * tok_84 * sc;
                        converted[2] = frag[56] * tok_84 * sc;
                        converted[3] = frag[58] * tok_84 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_85 = token_83 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_83 * 64 + (base_feature ^ swizzle_feature_85)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_83 * 64 + (base_feature - 64 ^ swizzle_feature_85)) / 4] = packed_word;
                        }
                        int token_86 = base_token + 48 + 1;
                        float tok_87 = tok_stage[token_block_45 * 64 + token_86];
                        converted[0] = frag[25] * tok_87 * sc;
                        converted[1] = frag[27] * tok_87 * sc;
                        converted[2] = frag[57] * tok_87 * sc;
                        converted[3] = frag[59] * tok_87 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_88 = token_86 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_86 * 64 + (base_feature ^ swizzle_feature_88)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_86 * 64 + (base_feature - 64 ^ swizzle_feature_88)) / 4] = packed_word;
                        }
                        int token_89 = base_token + 56;
                        float tok_90 = tok_stage[token_block_45 * 64 + token_89];
                        converted[0] = frag[28] * tok_90 * sc;
                        converted[1] = frag[30] * tok_90 * sc;
                        converted[2] = frag[60] * tok_90 * sc;
                        converted[3] = frag[62] * tok_90 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_91 = token_89 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_89 * 64 + (base_feature ^ swizzle_feature_91)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_89 * 64 + (base_feature - 64 ^ swizzle_feature_91)) / 4] = packed_word;
                        }
                        int token_92 = base_token + 56 + 1;
                        float tok_93 = tok_stage[token_block_45 * 64 + token_92];
                        converted[0] = frag[29] * tok_93 * sc;
                        converted[1] = frag[31] * tok_93 * sc;
                        converted[2] = frag[61] * tok_93 * sc;
                        converted[3] = frag[63] * tok_93 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_94 = token_92 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_92 * 64 + (base_feature ^ swizzle_feature_94)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_92 * 64 + (base_feature - 64 ^ swizzle_feature_94)) / 4] = packed_word;
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 7, 128;" ::: "memory");
                        if (warp == 0) {
                            if (elect_sync()) {
                                int padding_rows_1 = (256 - valid_rows % 256) % 256;
                                tma_store_4d((&C), m_tile * 128, padding_rows_1 + token_block_45 * 64, 1073741824, n_tile * 256 - (unsigned int)padding_rows_1 + 1073741824, epi_staging_addr);
                                tma_store_4d((&C), m_tile * 128 + 64, padding_rows_1 + token_block_45 * 64, 1073741824, n_tile * 256 - (unsigned int)padding_rows_1 + 1073741824, epi_staging_addr + 8192);
                            }
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                        asm volatile("barrier.sync 7, 128;" ::: "memory");
                        int token_block_95 = 2;
                        if (epilogue_local_idx == 0) {
                            token_block_95 = 1;
                        }
                        int acc_col_96 = epilogue_local_idx * 192 + token_block_95 * 64;
                        float _tmem_load_4[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[31]))
                            : "r"(taddr + (unsigned int)row_addr + (unsigned int)acc_col_96));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        float _tmem_load_5[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[31]))
                            : "r"(taddr + (unsigned int)row_addr + 1048576 + (unsigned int)acc_col_96));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        for (int i_2 = 0; i_2 < 32; i_2++) {
                            frag[i_2] = _tmem_load_4[i_2];
                            frag[32 + i_2] = _tmem_load_5[i_2];
                        }
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                        asm volatile("barrier.sync 7, 128;" ::: "memory");
                        int token_97 = base_token;
                        float tok_98 = tok_stage[token_block_95 * 64 + token_97];
                        converted[0] = frag[0] * tok_98 * sc;
                        converted[1] = frag[2] * tok_98 * sc;
                        converted[2] = frag[32] * tok_98 * sc;
                        converted[3] = frag[34] * tok_98 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_99 = token_97 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_97 * 64 + (base_feature ^ swizzle_feature_99)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_97 * 64 + (base_feature - 64 ^ swizzle_feature_99)) / 4] = packed_word;
                        }
                        int token_100 = base_token + 1;
                        float tok_101 = tok_stage[token_block_95 * 64 + token_100];
                        converted[0] = frag[1] * tok_101 * sc;
                        converted[1] = frag[3] * tok_101 * sc;
                        converted[2] = frag[33] * tok_101 * sc;
                        converted[3] = frag[35] * tok_101 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_102 = token_100 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_100 * 64 + (base_feature ^ swizzle_feature_102)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_100 * 64 + (base_feature - 64 ^ swizzle_feature_102)) / 4] = packed_word;
                        }
                        int token_103 = base_token + 8;
                        float tok_104 = tok_stage[token_block_95 * 64 + token_103];
                        converted[0] = frag[4] * tok_104 * sc;
                        converted[1] = frag[6] * tok_104 * sc;
                        converted[2] = frag[36] * tok_104 * sc;
                        converted[3] = frag[38] * tok_104 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_105 = token_103 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_103 * 64 + (base_feature ^ swizzle_feature_105)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_103 * 64 + (base_feature - 64 ^ swizzle_feature_105)) / 4] = packed_word;
                        }
                        int token_106 = base_token + 8 + 1;
                        float tok_107 = tok_stage[token_block_95 * 64 + token_106];
                        converted[0] = frag[5] * tok_107 * sc;
                        converted[1] = frag[7] * tok_107 * sc;
                        converted[2] = frag[37] * tok_107 * sc;
                        converted[3] = frag[39] * tok_107 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_108 = token_106 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_106 * 64 + (base_feature ^ swizzle_feature_108)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_106 * 64 + (base_feature - 64 ^ swizzle_feature_108)) / 4] = packed_word;
                        }
                        int token_109 = base_token + 16;
                        float tok_110 = tok_stage[token_block_95 * 64 + token_109];
                        converted[0] = frag[8] * tok_110 * sc;
                        converted[1] = frag[10] * tok_110 * sc;
                        converted[2] = frag[40] * tok_110 * sc;
                        converted[3] = frag[42] * tok_110 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_111 = token_109 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_109 * 64 + (base_feature ^ swizzle_feature_111)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_109 * 64 + (base_feature - 64 ^ swizzle_feature_111)) / 4] = packed_word;
                        }
                        int token_112 = base_token + 16 + 1;
                        float tok_113 = tok_stage[token_block_95 * 64 + token_112];
                        converted[0] = frag[9] * tok_113 * sc;
                        converted[1] = frag[11] * tok_113 * sc;
                        converted[2] = frag[41] * tok_113 * sc;
                        converted[3] = frag[43] * tok_113 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_114 = token_112 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_112 * 64 + (base_feature ^ swizzle_feature_114)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_112 * 64 + (base_feature - 64 ^ swizzle_feature_114)) / 4] = packed_word;
                        }
                        int token_115 = base_token + 24;
                        float tok_116 = tok_stage[token_block_95 * 64 + token_115];
                        converted[0] = frag[12] * tok_116 * sc;
                        converted[1] = frag[14] * tok_116 * sc;
                        converted[2] = frag[44] * tok_116 * sc;
                        converted[3] = frag[46] * tok_116 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_117 = token_115 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_115 * 64 + (base_feature ^ swizzle_feature_117)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_115 * 64 + (base_feature - 64 ^ swizzle_feature_117)) / 4] = packed_word;
                        }
                        int token_118 = base_token + 24 + 1;
                        float tok_119 = tok_stage[token_block_95 * 64 + token_118];
                        converted[0] = frag[13] * tok_119 * sc;
                        converted[1] = frag[15] * tok_119 * sc;
                        converted[2] = frag[45] * tok_119 * sc;
                        converted[3] = frag[47] * tok_119 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_120 = token_118 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_118 * 64 + (base_feature ^ swizzle_feature_120)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_118 * 64 + (base_feature - 64 ^ swizzle_feature_120)) / 4] = packed_word;
                        }
                        int token_121 = base_token + 32;
                        float tok_122 = tok_stage[token_block_95 * 64 + token_121];
                        converted[0] = frag[16] * tok_122 * sc;
                        converted[1] = frag[18] * tok_122 * sc;
                        converted[2] = frag[48] * tok_122 * sc;
                        converted[3] = frag[50] * tok_122 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_123 = token_121 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_121 * 64 + (base_feature ^ swizzle_feature_123)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_121 * 64 + (base_feature - 64 ^ swizzle_feature_123)) / 4] = packed_word;
                        }
                        int token_124 = base_token + 32 + 1;
                        float tok_125 = tok_stage[token_block_95 * 64 + token_124];
                        converted[0] = frag[17] * tok_125 * sc;
                        converted[1] = frag[19] * tok_125 * sc;
                        converted[2] = frag[49] * tok_125 * sc;
                        converted[3] = frag[51] * tok_125 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_126 = token_124 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_124 * 64 + (base_feature ^ swizzle_feature_126)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_124 * 64 + (base_feature - 64 ^ swizzle_feature_126)) / 4] = packed_word;
                        }
                        int token_127 = base_token + 40;
                        float tok_128 = tok_stage[token_block_95 * 64 + token_127];
                        converted[0] = frag[20] * tok_128 * sc;
                        converted[1] = frag[22] * tok_128 * sc;
                        converted[2] = frag[52] * tok_128 * sc;
                        converted[3] = frag[54] * tok_128 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_129 = token_127 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_127 * 64 + (base_feature ^ swizzle_feature_129)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_127 * 64 + (base_feature - 64 ^ swizzle_feature_129)) / 4] = packed_word;
                        }
                        int token_130 = base_token + 40 + 1;
                        float tok_131 = tok_stage[token_block_95 * 64 + token_130];
                        converted[0] = frag[21] * tok_131 * sc;
                        converted[1] = frag[23] * tok_131 * sc;
                        converted[2] = frag[53] * tok_131 * sc;
                        converted[3] = frag[55] * tok_131 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_132 = token_130 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_130 * 64 + (base_feature ^ swizzle_feature_132)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_130 * 64 + (base_feature - 64 ^ swizzle_feature_132)) / 4] = packed_word;
                        }
                        int token_133 = base_token + 48;
                        float tok_134 = tok_stage[token_block_95 * 64 + token_133];
                        converted[0] = frag[24] * tok_134 * sc;
                        converted[1] = frag[26] * tok_134 * sc;
                        converted[2] = frag[56] * tok_134 * sc;
                        converted[3] = frag[58] * tok_134 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_135 = token_133 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_133 * 64 + (base_feature ^ swizzle_feature_135)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_133 * 64 + (base_feature - 64 ^ swizzle_feature_135)) / 4] = packed_word;
                        }
                        int token_136 = base_token + 48 + 1;
                        float tok_137 = tok_stage[token_block_95 * 64 + token_136];
                        converted[0] = frag[25] * tok_137 * sc;
                        converted[1] = frag[27] * tok_137 * sc;
                        converted[2] = frag[57] * tok_137 * sc;
                        converted[3] = frag[59] * tok_137 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_138 = token_136 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_136 * 64 + (base_feature ^ swizzle_feature_138)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_136 * 64 + (base_feature - 64 ^ swizzle_feature_138)) / 4] = packed_word;
                        }
                        int token_139 = base_token + 56;
                        float tok_140 = tok_stage[token_block_95 * 64 + token_139];
                        converted[0] = frag[28] * tok_140 * sc;
                        converted[1] = frag[30] * tok_140 * sc;
                        converted[2] = frag[60] * tok_140 * sc;
                        converted[3] = frag[62] * tok_140 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_141 = token_139 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_139 * 64 + (base_feature ^ swizzle_feature_141)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_139 * 64 + (base_feature - 64 ^ swizzle_feature_141)) / 4] = packed_word;
                        }
                        int token_142 = base_token + 56 + 1;
                        float tok_143 = tok_stage[token_block_95 * 64 + token_142];
                        converted[0] = frag[29] * tok_143 * sc;
                        converted[1] = frag[31] * tok_143 * sc;
                        converted[2] = frag[61] * tok_143 * sc;
                        converted[3] = frag[63] * tok_143 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_144 = token_142 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_142 * 64 + (base_feature ^ swizzle_feature_144)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_142 * 64 + (base_feature - 64 ^ swizzle_feature_144)) / 4] = packed_word;
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 7, 128;" ::: "memory");
                        if (warp == 0) {
                            if (elect_sync()) {
                                int padding_rows_2 = (256 - valid_rows % 256) % 256;
                                tma_store_4d((&C), m_tile * 128, padding_rows_2 + token_block_95 * 64, 1073741824, n_tile * 256 - (unsigned int)padding_rows_2 + 1073741824, epi_staging_addr);
                                tma_store_4d((&C), m_tile * 128 + 64, padding_rows_2 + token_block_95 * 64, 1073741824, n_tile * 256 - (unsigned int)padding_rows_2 + 1073741824, epi_staging_addr + 8192);
                            }
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                        asm volatile("barrier.sync 7, 128;" ::: "memory");
                        int token_block_145 = 3;
                        if (epilogue_local_idx == 0) {
                            token_block_145 = 2;
                        }
                        int acc_col_146 = epilogue_local_idx * 192 + token_block_145 * 64;
                        float _tmem_load_6[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[31]))
                            : "r"(taddr + (unsigned int)row_addr + (unsigned int)acc_col_146));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        float _tmem_load_7[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_7[31]))
                            : "r"(taddr + (unsigned int)row_addr + 1048576 + (unsigned int)acc_col_146));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        for (int i_3 = 0; i_3 < 32; i_3++) {
                            frag[i_3] = _tmem_load_6[i_3];
                            frag[32 + i_3] = _tmem_load_7[i_3];
                        }
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                        asm volatile("barrier.sync 7, 128;" ::: "memory");
                        int token_147 = base_token;
                        float tok_148 = tok_stage[token_block_145 * 64 + token_147];
                        converted[0] = frag[0] * tok_148 * sc;
                        converted[1] = frag[2] * tok_148 * sc;
                        converted[2] = frag[32] * tok_148 * sc;
                        converted[3] = frag[34] * tok_148 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_149 = token_147 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_147 * 64 + (base_feature ^ swizzle_feature_149)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_147 * 64 + (base_feature - 64 ^ swizzle_feature_149)) / 4] = packed_word;
                        }
                        int token_150 = base_token + 1;
                        float tok_151 = tok_stage[token_block_145 * 64 + token_150];
                        converted[0] = frag[1] * tok_151 * sc;
                        converted[1] = frag[3] * tok_151 * sc;
                        converted[2] = frag[33] * tok_151 * sc;
                        converted[3] = frag[35] * tok_151 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_152 = token_150 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_150 * 64 + (base_feature ^ swizzle_feature_152)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_150 * 64 + (base_feature - 64 ^ swizzle_feature_152)) / 4] = packed_word;
                        }
                        int token_153 = base_token + 8;
                        float tok_154 = tok_stage[token_block_145 * 64 + token_153];
                        converted[0] = frag[4] * tok_154 * sc;
                        converted[1] = frag[6] * tok_154 * sc;
                        converted[2] = frag[36] * tok_154 * sc;
                        converted[3] = frag[38] * tok_154 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_155 = token_153 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_153 * 64 + (base_feature ^ swizzle_feature_155)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_153 * 64 + (base_feature - 64 ^ swizzle_feature_155)) / 4] = packed_word;
                        }
                        int token_156 = base_token + 8 + 1;
                        float tok_157 = tok_stage[token_block_145 * 64 + token_156];
                        converted[0] = frag[5] * tok_157 * sc;
                        converted[1] = frag[7] * tok_157 * sc;
                        converted[2] = frag[37] * tok_157 * sc;
                        converted[3] = frag[39] * tok_157 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_158 = token_156 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_156 * 64 + (base_feature ^ swizzle_feature_158)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_156 * 64 + (base_feature - 64 ^ swizzle_feature_158)) / 4] = packed_word;
                        }
                        int token_159 = base_token + 16;
                        float tok_160 = tok_stage[token_block_145 * 64 + token_159];
                        converted[0] = frag[8] * tok_160 * sc;
                        converted[1] = frag[10] * tok_160 * sc;
                        converted[2] = frag[40] * tok_160 * sc;
                        converted[3] = frag[42] * tok_160 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_161 = token_159 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_159 * 64 + (base_feature ^ swizzle_feature_161)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_159 * 64 + (base_feature - 64 ^ swizzle_feature_161)) / 4] = packed_word;
                        }
                        int token_162 = base_token + 16 + 1;
                        float tok_163 = tok_stage[token_block_145 * 64 + token_162];
                        converted[0] = frag[9] * tok_163 * sc;
                        converted[1] = frag[11] * tok_163 * sc;
                        converted[2] = frag[41] * tok_163 * sc;
                        converted[3] = frag[43] * tok_163 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_164 = token_162 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_162 * 64 + (base_feature ^ swizzle_feature_164)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_162 * 64 + (base_feature - 64 ^ swizzle_feature_164)) / 4] = packed_word;
                        }
                        int token_165 = base_token + 24;
                        float tok_166 = tok_stage[token_block_145 * 64 + token_165];
                        converted[0] = frag[12] * tok_166 * sc;
                        converted[1] = frag[14] * tok_166 * sc;
                        converted[2] = frag[44] * tok_166 * sc;
                        converted[3] = frag[46] * tok_166 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_167 = token_165 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_165 * 64 + (base_feature ^ swizzle_feature_167)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_165 * 64 + (base_feature - 64 ^ swizzle_feature_167)) / 4] = packed_word;
                        }
                        int token_168 = base_token + 24 + 1;
                        float tok_169 = tok_stage[token_block_145 * 64 + token_168];
                        converted[0] = frag[13] * tok_169 * sc;
                        converted[1] = frag[15] * tok_169 * sc;
                        converted[2] = frag[45] * tok_169 * sc;
                        converted[3] = frag[47] * tok_169 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_170 = token_168 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_168 * 64 + (base_feature ^ swizzle_feature_170)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_168 * 64 + (base_feature - 64 ^ swizzle_feature_170)) / 4] = packed_word;
                        }
                        int token_171 = base_token + 32;
                        float tok_172 = tok_stage[token_block_145 * 64 + token_171];
                        converted[0] = frag[16] * tok_172 * sc;
                        converted[1] = frag[18] * tok_172 * sc;
                        converted[2] = frag[48] * tok_172 * sc;
                        converted[3] = frag[50] * tok_172 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_173 = token_171 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_171 * 64 + (base_feature ^ swizzle_feature_173)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_171 * 64 + (base_feature - 64 ^ swizzle_feature_173)) / 4] = packed_word;
                        }
                        int token_174 = base_token + 32 + 1;
                        float tok_175 = tok_stage[token_block_145 * 64 + token_174];
                        converted[0] = frag[17] * tok_175 * sc;
                        converted[1] = frag[19] * tok_175 * sc;
                        converted[2] = frag[49] * tok_175 * sc;
                        converted[3] = frag[51] * tok_175 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_176 = token_174 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_174 * 64 + (base_feature ^ swizzle_feature_176)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_174 * 64 + (base_feature - 64 ^ swizzle_feature_176)) / 4] = packed_word;
                        }
                        int token_177 = base_token + 40;
                        float tok_178 = tok_stage[token_block_145 * 64 + token_177];
                        converted[0] = frag[20] * tok_178 * sc;
                        converted[1] = frag[22] * tok_178 * sc;
                        converted[2] = frag[52] * tok_178 * sc;
                        converted[3] = frag[54] * tok_178 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_179 = token_177 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_177 * 64 + (base_feature ^ swizzle_feature_179)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_177 * 64 + (base_feature - 64 ^ swizzle_feature_179)) / 4] = packed_word;
                        }
                        int token_180 = base_token + 40 + 1;
                        float tok_181 = tok_stage[token_block_145 * 64 + token_180];
                        converted[0] = frag[21] * tok_181 * sc;
                        converted[1] = frag[23] * tok_181 * sc;
                        converted[2] = frag[53] * tok_181 * sc;
                        converted[3] = frag[55] * tok_181 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_182 = token_180 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_180 * 64 + (base_feature ^ swizzle_feature_182)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_180 * 64 + (base_feature - 64 ^ swizzle_feature_182)) / 4] = packed_word;
                        }
                        int token_183 = base_token + 48;
                        float tok_184 = tok_stage[token_block_145 * 64 + token_183];
                        converted[0] = frag[24] * tok_184 * sc;
                        converted[1] = frag[26] * tok_184 * sc;
                        converted[2] = frag[56] * tok_184 * sc;
                        converted[3] = frag[58] * tok_184 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_185 = token_183 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_183 * 64 + (base_feature ^ swizzle_feature_185)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_183 * 64 + (base_feature - 64 ^ swizzle_feature_185)) / 4] = packed_word;
                        }
                        int token_186 = base_token + 48 + 1;
                        float tok_187 = tok_stage[token_block_145 * 64 + token_186];
                        converted[0] = frag[25] * tok_187 * sc;
                        converted[1] = frag[27] * tok_187 * sc;
                        converted[2] = frag[57] * tok_187 * sc;
                        converted[3] = frag[59] * tok_187 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_188 = token_186 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_186 * 64 + (base_feature ^ swizzle_feature_188)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_186 * 64 + (base_feature - 64 ^ swizzle_feature_188)) / 4] = packed_word;
                        }
                        int token_189 = base_token + 56;
                        float tok_190 = tok_stage[token_block_145 * 64 + token_189];
                        converted[0] = frag[28] * tok_190 * sc;
                        converted[1] = frag[30] * tok_190 * sc;
                        converted[2] = frag[60] * tok_190 * sc;
                        converted[3] = frag[62] * tok_190 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_191 = token_189 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_189 * 64 + (base_feature ^ swizzle_feature_191)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_189 * 64 + (base_feature - 64 ^ swizzle_feature_191)) / 4] = packed_word;
                        }
                        int token_192 = base_token + 56 + 1;
                        float tok_193 = tok_stage[token_block_145 * 64 + token_192];
                        converted[0] = frag[29] * tok_193 * sc;
                        converted[1] = frag[31] * tok_193 * sc;
                        converted[2] = frag[61] * tok_193 * sc;
                        converted[3] = frag[63] * tok_193 * sc;
                        #pragma unroll
                        for (int _lp = 0; _lp < 2; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(converted[_lp*2 + 0], converted[_lp*2+1 + 0]));
                            packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        packed_word = (unsigned long long)packed[0] | (unsigned long long)packed[1] << 32;
                        int swizzle_feature_194 = token_192 % 8 * 8;
                        if (base_feature < 64) {
                            epi_staging_u64[(token_192 * 64 + (base_feature ^ swizzle_feature_194)) / 4] = packed_word;
                        } else {
                            epi_staging_u64[(4096 + token_192 * 64 + (base_feature - 64 ^ swizzle_feature_194)) / 4] = packed_word;
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 7, 128;" ::: "memory");
                        if (warp == 0) {
                            if (elect_sync()) {
                                int padding_rows_3 = (256 - valid_rows % 256) % 256;
                                tma_store_4d((&C), m_tile * 128, padding_rows_3 + token_block_145 * 64, 1073741824, n_tile * 256 - (unsigned int)padding_rows_3 + 1073741824, epi_staging_addr);
                                tma_store_4d((&C), m_tile * 128 + 64, padding_rows_3 + token_block_145 * 64, 1073741824, n_tile * 256 - (unsigned int)padding_rows_3 + 1073741824, epi_staging_addr + 8192);
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
    // ---- Role: load_b ----
    if (warp == 4) {
        { // load_b_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            int tile_count_1 = num_non_exiting_ctas[0];
            unsigned int stage = 0;
            unsigned int work_stage_1 = 0;
            unsigned int m_tile_1 = blockIdx.x;
            unsigned int n_tile_1 = blockIdx.y;
            unsigned int _phase_b_empty = 1;
            unsigned int _phase_work_full_1 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_1 = 0; _tile_iter_1 < grid_m * grid_n; _tile_iter_1++) {
                int valid_rows_1 = (unsigned int)tile_mn_limit[n_tile_1] - n_tile_1 * 256;
                if (tile_count_1 > (int)n_tile_1) {
                    if (valid_rows_1 > 0) {
                        int padding_rows_4 = (256 - valid_rows_1 % 256) % 256;
                        #pragma unroll 1
                        for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                            mbarrier_wait(b_empty_addr + (stage) * 8, _phase_b_empty);
                            if (elect_sync()) {
                                tma_4d_gmem2smem(smem_b_addr + stage * 32768, (&B), iter_k * 128, padding_rows_4, 1073741824, n_tile_1 * 256 - (unsigned int)padding_rows_4 + 1073741824, b_full_addr + (stage) * 8);
                                mbarrier_arrive_expect_tx(b_full_addr + (stage) * 8, 32768);
                            }
                            stage += 1;
                            if (stage == 3) { stage = 0; _phase_b_empty ^= 1; }
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
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
    }
    // ---- Role: load_sfb ----
    if (warp == 5) {
        { // load_sfb_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            int tile_count_2 = num_non_exiting_ctas[0];
            unsigned int stage_1 = 0;
            unsigned int work_stage_2 = 0;
            unsigned int m_tile_2 = blockIdx.x;
            unsigned int n_tile_2 = blockIdx.y;
            unsigned int _phase_sfb_empty = 1;
            unsigned int _phase_work_full_2 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_2 = 0; _tile_iter_2 < grid_m * grid_n; _tile_iter_2++) {
                if (tile_count_2 > (int)n_tile_2) {
                    int valid_rows_sfb = (unsigned int)tile_mn_limit[n_tile_2] - n_tile_2 * 256;
                    if (valid_rows_sfb > 0) {
                        #pragma unroll 1
                        for (int iter_k_1 = 0; iter_k_1 < K_tiles; iter_k_1++) {
                            mbarrier_wait(sfb_empty_addr + (stage_1) * 8, _phase_sfb_empty);
                            if (elect_sync()) {
                                tma_4d_gmem2smem(smem_sfb_addr + stage_1 * 4096, (&SFB), 0, 0, iter_k_1 * 4, n_tile_2 * 2, sfb_full_addr + (stage_1) * 8);
                                mbarrier_arrive_expect_tx(sfb_full_addr + (stage_1) * 8, 4096);
                            }
                            stage_1 += 1;
                            if (stage_1 == 3) { stage_1 = 0; _phase_sfb_empty ^= 1; }
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
    // ---- Role: load_a ----
    if (warp == 6) {
        { // load_a_main
            int tile_count_3 = num_non_exiting_ctas[0];
            unsigned int stage_2 = 0;
            unsigned int work_stage_3 = 0;
            unsigned int throttle_stage = 0;
            unsigned int m_tile_3 = blockIdx.x;
            unsigned int n_tile_3 = blockIdx.y;
            unsigned int _phase_throttle_empty = 1;
            unsigned int _phase_a_empty = 1;
            unsigned int _phase_work_full_3 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_3 = 0; _tile_iter_3 < grid_m * grid_n; _tile_iter_3++) {
                if (tile_count_3 > (int)n_tile_3) {
                    int expert = tile_expert[n_tile_3];
                    mbarrier_wait(throttle_empty_addr + (throttle_stage) * 8, _phase_throttle_empty);
                    mbarrier_arrive(throttle_full_addr + (throttle_stage) * 8);
                    throttle_stage += 1;
                    if (throttle_stage == 3) { throttle_stage = 0; _phase_throttle_empty ^= 1; }
                    int valid_rows_a = (unsigned int)tile_mn_limit[n_tile_3] - n_tile_3 * 256;
                    if (valid_rows_a > 0) {
                        #pragma unroll 1
                        for (int iter_k_2 = 0; iter_k_2 < K_tiles; iter_k_2++) {
                            mbarrier_wait(a_empty_addr + (stage_2) * 8, _phase_a_empty);
                            if (elect_sync()) {
                                tma_4d_gmem2smem(smem_a_addr + stage_2 * 16384, (&A), 0, m_tile_3 * 128, iter_k_2, expert, a_full_addr + (stage_2) * 8);
                                mbarrier_arrive_expect_tx(a_full_addr + (stage_2) * 8, 16384);
                            }
                            stage_2 += 1;
                            if (stage_2 == 3) { stage_2 = 0; _phase_a_empty ^= 1; }
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
            unsigned int _phase_sfa_empty = 1;
            unsigned int _phase_work_full_4 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_4 = 0; _tile_iter_4 < grid_m * grid_n; _tile_iter_4++) {
                int expert_sfa = tile_expert[n_tile_4];
                if (tile_count_4 > (int)n_tile_4) {
                    int valid_rows_sfa = (unsigned int)tile_mn_limit[n_tile_4] - n_tile_4 * 256;
                    if (valid_rows_sfa > 0) {
                        #pragma unroll 1
                        for (int iter_k_3 = 0; iter_k_3 < K_tiles; iter_k_3++) {
                            mbarrier_wait(sfa_empty_addr + (stage_3) * 8, _phase_sfa_empty);
                            if (elect_sync()) {
                                tma_4d_gmem2smem(smem_sfa_addr + stage_3 * 2048, (&SFA), 0, 0, iter_k_3 * 4, (unsigned int)(expert_sfa * grid_m) + m_tile_4, sfa_full_addr + (stage_3) * 8);
                                mbarrier_arrive_expect_tx(sfa_full_addr + (stage_3) * 8, 2048);
                            }
                            stage_3 += 1;
                            if (stage_3 == 3) { stage_3 = 0; _phase_sfa_empty ^= 1; }
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
    // ---- Role: copy_sfab_mma ----
    if (warp == 8) {
        { // copy_sfab_mma_main
            int tile_count_5 = num_non_exiting_ctas[0];
            unsigned int stage_4 = 0;
            unsigned int work_stage_5 = 0;
            int mma_local_idx = 0;
            unsigned int m_tile_5 = blockIdx.x;
            unsigned int n_tile_5 = blockIdx.y;
            unsigned int _phase_mma_free_0 = 1;
            unsigned int _phase_a_full = 0;
            unsigned int _phase_b_full = 0;
            unsigned int _phase_sfa_full = 0;
            unsigned int _phase_sfb_full = 0;
            unsigned int _phase_work_full_5 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_5 = 0; _tile_iter_5 < grid_m * grid_n; _tile_iter_5++) {
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
                                        "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                        :: "r"((uint32_t)(tmem_sfa)), "l"(_tcgen05_cp_desc_0)
                                        : "memory");
                                }
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                #endif
                                {
                                    uint64_t _tcgen05_cp_desc_1 = ((((uint64_t)(smem_sfa_addr + stage_4 * 2048 + 512)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                    asm volatile(
                                        "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                        :: "r"((uint32_t)(tmem_sfa + 4)), "l"(_tcgen05_cp_desc_1)
                                        : "memory");
                                }
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                #endif
                                {
                                    uint64_t _tcgen05_cp_desc_2 = ((((uint64_t)(smem_sfa_addr + stage_4 * 2048 + 1024)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                    asm volatile(
                                        "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                        :: "r"((uint32_t)(tmem_sfa + 8)), "l"(_tcgen05_cp_desc_2)
                                        : "memory");
                                }
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                #endif
                                {
                                    uint64_t _tcgen05_cp_desc_3 = ((((uint64_t)(smem_sfa_addr + stage_4 * 2048 + 1536)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                    asm volatile(
                                        "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                        :: "r"((uint32_t)(tmem_sfa + 12)), "l"(_tcgen05_cp_desc_3)
                                        : "memory");
                                }
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                #endif
                                {
                                    uint64_t _tcgen05_cp_desc_4 = ((((uint64_t)(smem_sfb_addr + stage_4 * 4096)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                    asm volatile(
                                        "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                        :: "r"((uint32_t)(tmem_sfb)), "l"(_tcgen05_cp_desc_4)
                                        : "memory");
                                }
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                #endif
                                {
                                    uint64_t _tcgen05_cp_desc_5 = ((((uint64_t)(smem_sfb_addr + stage_4 * 4096 + 512)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                    asm volatile(
                                        "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                        :: "r"((uint32_t)(tmem_sfb + 8)), "l"(_tcgen05_cp_desc_5)
                                        : "memory");
                                }
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                #endif
                                {
                                    uint64_t _tcgen05_cp_desc_6 = ((((uint64_t)(smem_sfb_addr + stage_4 * 4096 + 1024)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                    asm volatile(
                                        "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                        :: "r"((uint32_t)(tmem_sfb + 16)), "l"(_tcgen05_cp_desc_6)
                                        : "memory");
                                }
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                #endif
                                {
                                    uint64_t _tcgen05_cp_desc_7 = ((((uint64_t)(smem_sfb_addr + stage_4 * 4096 + 1536)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                    asm volatile(
                                        "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                        :: "r"((uint32_t)(tmem_sfb + 24)), "l"(_tcgen05_cp_desc_7)
                                        : "memory");
                                }
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                #endif
                                {
                                    uint64_t _tcgen05_cp_desc_8 = ((((uint64_t)(smem_sfb_addr + stage_4 * 4096 + 2048)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                    asm volatile(
                                        "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                        :: "r"((uint32_t)(tmem_sfb + 4)), "l"(_tcgen05_cp_desc_8)
                                        : "memory");
                                }
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                #endif
                                {
                                    uint64_t _tcgen05_cp_desc_9 = ((((uint64_t)(smem_sfb_addr + stage_4 * 4096 + 2048 + 512)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                    asm volatile(
                                        "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                        :: "r"((uint32_t)(tmem_sfb + 12)), "l"(_tcgen05_cp_desc_9)
                                        : "memory");
                                }
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                #endif
                                {
                                    uint64_t _tcgen05_cp_desc_10 = ((((uint64_t)(smem_sfb_addr + stage_4 * 4096 + 2048 + 1024)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                    asm volatile(
                                        "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                        :: "r"((uint32_t)(tmem_sfb + 20)), "l"(_tcgen05_cp_desc_10)
                                        : "memory");
                                }
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                                #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                                #endif
                                {
                                    uint64_t _tcgen05_cp_desc_11 = ((((uint64_t)(smem_sfb_addr + stage_4 * 4096 + 2048 + 1536)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                                    asm volatile(
                                        "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                        :: "r"((uint32_t)(tmem_sfb + 28)), "l"(_tcgen05_cp_desc_11)
                                        : "memory");
                                }
                            }
                            int _mma_a_lo_0 = make_warp_uniform((((smem_a_addr) >> 4) & 0x3FFF) + (stage_4) * 1024);
                            int _mma_b_lo_0 = make_warp_uniform((((smem_b_addr) >> 4) & 0x3FFF) + (stage_4) * 2048);
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs((tmem_accum + (mma_local_idx * 192)), a_desc + 0, b_desc + 0,
                                        0x8400480U, tmem_sfa + 0, tmem_sfb + 0, ((((1) ? ((iter_k_4 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                                }
                            }
                            int _mma_a_lo_1 = make_warp_uniform((((smem_a_addr + 32) >> 4) & 0x3FFF) + (stage_4) * 1024);
                            int _mma_b_lo_1 = make_warp_uniform((((smem_b_addr + 32) >> 4) & 0x3FFF) + (stage_4) * 2048);
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs((tmem_accum + (mma_local_idx * 192)), a_desc + 0, b_desc + 0,
                                        0x8400480U, tmem_sfa + 4 + 0, tmem_sfb + 8 + 0, ((((0) ? ((iter_k_4 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                                }
                            }
                            int _mma_a_lo_2 = make_warp_uniform((((smem_a_addr + 64) >> 4) & 0x3FFF) + (stage_4) * 1024);
                            int _mma_b_lo_2 = make_warp_uniform((((smem_b_addr + 64) >> 4) & 0x3FFF) + (stage_4) * 2048);
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_2) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_2) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs((tmem_accum + (mma_local_idx * 192)), a_desc + 0, b_desc + 0,
                                        0x8400480U, tmem_sfa + 8 + 0, tmem_sfb + 16 + 0, ((((0) ? ((iter_k_4 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                                }
                            }
                            int _mma_a_lo_3 = make_warp_uniform((((smem_a_addr + 96) >> 4) & 0x3FFF) + (stage_4) * 1024);
                            int _mma_b_lo_3 = make_warp_uniform((((smem_b_addr + 96) >> 4) & 0x3FFF) + (stage_4) * 2048);
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_3) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_3) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs((tmem_accum + (mma_local_idx * 192)), a_desc + 0, b_desc + 0,
                                        0x8400480U, tmem_sfa + 12 + 0, tmem_sfb + 24 + 0, ((((0) ? ((iter_k_4 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                                }
                            }
                            elect_commit(a_empty_addr + (stage_4) * 8);
                            elect_commit(b_empty_addr + (stage_4) * 8);
                            elect_commit(sfa_empty_addr + (stage_4) * 8);
                            elect_commit(sfb_empty_addr + (stage_4) * 8);
                            if (iter_k_4 + 1 == K_tiles) {
                                elect_commit(mma_full_addr);
                            }
                            stage_4 += 1;
                            if (stage_4 == 3) { stage_4 = 0; _phase_a_full ^= 1; _phase_b_full ^= 1; _phase_sfa_full ^= 1; _phase_sfb_full ^= 1; }
                        }
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
    // ---- Role: work_id ----
    if (warp == 9) {
        { // work_id_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            int tile_count_6 = num_non_exiting_ctas[0];
            unsigned int work_stage_6 = 0;
            unsigned int throttle_stage_1 = 0;
            unsigned int m_tile_6 = blockIdx.x;
            unsigned int n_tile_6 = blockIdx.y;
            unsigned int _phase_throttle_full = 0;
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_work_full_6 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_6 = 0; _tile_iter_6 < grid_m * grid_n; _tile_iter_6++) {
                if (tile_count_6 > (int)n_tile_6) {
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
                                : "r"(fast_response_addr + fast_stage * 64 + slot_idx_1 * 16)
                                : "memory");
                            canceled += _clc_valid_6;
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        fast_phase ^= 1;
                        if (canceled == 0) {
                            break;
                        }
                    }
                }
                if (elect_sync()) {
                    mbarrier_wait(work_empty_addr + (work_stage_6) * 8, _phase_work_empty);
                    mbarrier_arrive_expect_tx(work_full_addr + (work_stage_6) * 8, 16);
                    asm volatile(
                        "fence.proxy.async.shared::cta;\n\t"
                        "clusterlaunchcontrol.try_cancel.async.shared::cta"
                            ".mbarrier::complete_tx::bytes.b128"
                            " [%0], [%1];"
                        :: "r"(work_response_addr + work_stage_6 * 16 + 0 * 16), "r"(work_full_addr + work_stage_6 * 8)
                        : "memory");
                }
                __syncwarp();
                mbarrier_wait(work_full_addr + (work_stage_6) * 8, _phase_work_full_6);
                unsigned int valid_6 = 0;
                unsigned int next_x_6 = 0;
                unsigned int next_y_6 = 0;
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
                    : "r"(work_response_addr + work_stage_6 * 16 + 0 * 16)
                    : "memory");
                valid_6 = _clc_valid_7;
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
                mbarrier_arrive(work_empty_addr + (work_stage_6) * 8);
                work_stage_6 += 1;
                if (work_stage_6 == 3) { work_stage_6 = 0; _phase_work_empty ^= 1; _phase_work_full_6 ^= 1; }
                unsigned int valid_0_6 = valid_6;
                m_tile_6 = next_x_6;
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
    // ---- Role: padding ----
    if (warp >= 10 && warp <= 11) {
        { // padding_main
            unsigned int work_stage_7 = 0;
            unsigned int m_tile_7 = blockIdx.x;
            unsigned int n_tile_7 = blockIdx.y;
            unsigned int _phase_work_full_7 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_7 = 0; _tile_iter_7 < grid_m * grid_n; _tile_iter_7++) {
                mbarrier_wait(work_full_addr + (work_stage_7) * 8, _phase_work_full_7);
                unsigned int valid_7 = 0;
                unsigned int next_x_7 = 0;
                unsigned int next_y_7 = 0;
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
                    : "r"(work_response_addr + work_stage_7 * 16 + 0 * 16)
                    : "memory");
                valid_7 = _clc_valid_8;
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

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
