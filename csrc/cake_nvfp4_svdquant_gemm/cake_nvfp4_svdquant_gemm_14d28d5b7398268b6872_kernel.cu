/*
 * Copyright (c) 2023 by FlashInfer team.
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
#define TMEM_NCOLS 288
#define TMEM_ACCUM_OFFSET 0
#define TMEM_TMEM_SFA_OFFSET 256
#define TMEM_TMEM_SFB_OFFSET 272
#define NUM_TMA_PIPE_STAGES 7
#define NUM_ACC_PIPE_STAGES 2
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 28672
#define SMEM_SMEM_B_OFF 17408
#define SMEM_SMEM_B_STAGE_BYTES 8192
#define SMEM_SMEM_B_STRIDE 28672
#define SMEM_SMEM_SFA_OFF 25600
#define SMEM_SMEM_SFA_STAGE_BYTES 2048
#define SMEM_SMEM_SFA_STRIDE 28672
#define SMEM_SMEM_SFB_OFF 27648
#define SMEM_SMEM_SFB_STAGE_BYTES 2048
#define SMEM_SMEM_SFB_STRIDE 28672
#define SMEM_SMEM_D_OFF 1024
#define SMEM_SMEM_D_STAGE_BYTES 16384
#define SMEM_SMEM_D_STRIDE 28672
#define SMEM_SMEM_L1_OFF 17408
#define SMEM_SMEM_L1_STAGE_BYTES 8192
#define SMEM_SMEM_L1_STRIDE 28672
#define SMEM_SMEM_OUT_OFF 201728
#define SMEM_SMEM_OUT_STAGE_BYTES 8192
#define SMEM_SMEM_OUT_STRIDE 8192
#define SMEM_TOTAL 218112
#define THREADS 256

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


__device__ __forceinline__ uint32_t mbarrier_try_wait(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
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

__device__ __forceinline__ void mbarrier_wait_token(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait(mbar_addr, phase);
    }
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



__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
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







__device__ __forceinline__ uint64_t make_sf_cp_desc_lo_sbo512(int lo) {
    const int SBO = 512;
    return (uint64_t)(uint32_t)lo
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL);
}


__device__ __forceinline__ void tcgen05_cp_32x128b_warpx4_cta2(
    int taddr, uint64_t s_desc) {
    asm volatile(
        "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
        :: "r"(taddr), "l"(s_desc));
}



__device__ __forceinline__ void tma_3d_gmem2smem_cta2(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_2d_gmem2smem_cta2(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_4d_gmem2smem_cta2(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_store_2d(
    const void *tmap, int x, int y, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2}], [%3];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(smem_addr) : "memory");
}


extern "C" {

__global__ __launch_bounds__(256, 1) __cluster_dims__(2,1,1) void
kernel_cake_nvfp4_svdquant_gemm_14d28d5b7398268b6872(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, const __grid_constant__ CUtensorMap D, const __grid_constant__ CUtensorMap L1, float* __restrict__ alpha, const __grid_constant__ CUtensorMap out, int K_tiles)
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
    #define tma_full_addr (mbar_base + 0)
    #define tma_empty_addr (mbar_base + 56)
    #define acc_full_addr (mbar_base + 112)
    #define acc_empty_addr (mbar_base + 128)
    #define tmem_dealloc_bar_addr (mbar_base + 144)

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
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int smem_b_addr = smem + 17408;
    uint8_t* smem_sfa = reinterpret_cast<uint8_t*>(smem_raw + 25600);
    const int smem_sfa_addr = smem + 25600;
    uint8_t* smem_sfb = reinterpret_cast<uint8_t*>(smem_raw + 27648);
    const int smem_sfb_addr = smem + 27648;
    __nv_bfloat16* smem_d = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_d_addr = smem + 1024;
    __nv_bfloat16* smem_l1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int smem_l1_addr = smem + 17408;
    __nv_bfloat16* smem_out = reinterpret_cast<__nv_bfloat16*>(smem_raw + 201728);
    const int smem_out_addr = smem + 201728;

    // Mbarrier init (5 pipeline groups, 0 ordered-sequence groups, 19 barriers)
    // Mbarriers at smem_raw[0..152)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full: 7 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            // tma_empty: 7 barriers, init_count=1
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            // --- pipeline 'acc_pipe' ---
            // acc_full: 2 barriers, init_count=1
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // acc_empty: 2 barriers, init_count=8
            mbarrier_init(smem + 128, 8);
            mbarrier_init(smem + 136, 8);
            // tmem_dealloc_bar: 1 barriers, init_count=32
            mbarrier_init(smem + 144, 32);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 288 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 152);
    if (warp == 0) {
        int _tmem_hold = smem + 152;
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
    const int tmem_tmem_sfa = taddr + 256;
    const int tmem_tmem_sfb = taddr + 272;

    // ---- Role: mma ----
    if (warp == 0) {
        { // mma_main
            unsigned int mma_stage = 0;
            unsigned int mma_phase = 0;
            unsigned int acc_stage = 0;
            unsigned int _phase_acc_empty = 1;
            if (cta_rank == 0) {
                uint32_t _mbar_token_2 = mbarrier_try_wait(tma_full_addr + (mma_stage) * 8, mma_phase);
                unsigned int peek_full = _mbar_token_2;
                mbarrier_wait(acc_empty_addr + (acc_stage) * 8, _phase_acc_empty);
                #pragma unroll 1
                for (unsigned int k_tile_mma = 0; k_tile_mma < K_tiles; k_tile_mma++) {
                    mbarrier_wait_token(tma_full_addr + (mma_stage) * 8, mma_phase, peek_full);
                    unsigned int read_stage = mma_stage;
                    mma_stage += 1;
                    if (mma_stage == 7) { mma_stage = 0; mma_phase ^= 1; }
                    uint32_t _mbar_token_3 = mbarrier_try_wait(tma_full_addr + (mma_stage) * 8, mma_phase);
                    peek_full = _mbar_token_3;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa, make_sf_cp_desc_lo_sbo512((((smem_sfa_addr) >> 4) + (read_stage) * 1792)));
                        tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfa + 4), make_sf_cp_desc_lo_sbo512((((smem_sfa_addr) >> 4) + (read_stage) * 1792 + 8)));
                        tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfa + 8), make_sf_cp_desc_lo_sbo512((((smem_sfa_addr) >> 4) + (read_stage) * 1792 + 16)));
                        tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfa + 12), make_sf_cp_desc_lo_sbo512((((smem_sfa_addr) >> 4) + (read_stage) * 1792 + 24)));
                    }
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb, make_sf_cp_desc_lo_sbo512((((smem_sfb_addr) >> 4) + (read_stage) * 1792)));
                        tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 4), make_sf_cp_desc_lo_sbo512((((smem_sfb_addr) >> 4) + (read_stage) * 1792 + 8)));
                        tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 8), make_sf_cp_desc_lo_sbo512((((smem_sfb_addr) >> 4) + (read_stage) * 1792 + 16)));
                        tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 12), make_sf_cp_desc_lo_sbo512((((smem_sfb_addr) >> 4) + (read_stage) * 1792 + 24)));
                    }
                    int init_flag = ((k_tile_mma == 0) ? 1 : 0);
                    int _mma_a_lo_0 = (((smem_a_addr) >> 4) & 0x3FFF) + (read_stage) * 1792;
                    int _mma_b_lo_0 = (((smem_b_addr) >> 4) & 0x3FFF) + (read_stage) * 1792;
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2((tmem_accum + (acc_stage * 128)), a_desc + 0, b_desc + 0,
                                0x10200480U, tmem_tmem_sfa + 0, tmem_tmem_sfb + 0, ((((1) ? init_flag : 0)) ? 0 : 1));
                        }
                    }
                    int _mma_a_lo_1 = (((smem_a_addr + 32) >> 4) & 0x3FFF) + (read_stage) * 1792;
                    int _mma_b_lo_1 = (((smem_b_addr + 32) >> 4) & 0x3FFF) + (read_stage) * 1792;
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2((tmem_accum + (acc_stage * 128)), a_desc + 0, b_desc + 0,
                                0x10200480U, tmem_tmem_sfa + 4 + 0, tmem_tmem_sfb + 4 + 0, ((((0) ? init_flag : 0)) ? 0 : 1));
                        }
                    }
                    int _mma_a_lo_2 = (((smem_a_addr + 64) >> 4) & 0x3FFF) + (read_stage) * 1792;
                    int _mma_b_lo_2 = (((smem_b_addr + 64) >> 4) & 0x3FFF) + (read_stage) * 1792;
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_2) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_2) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2((tmem_accum + (acc_stage * 128)), a_desc + 0, b_desc + 0,
                                0x10200480U, tmem_tmem_sfa + 8 + 0, tmem_tmem_sfb + 8 + 0, ((((0) ? init_flag : 0)) ? 0 : 1));
                        }
                    }
                    int _mma_a_lo_3 = (((smem_a_addr + 96) >> 4) & 0x3FFF) + (read_stage) * 1792;
                    int _mma_b_lo_3 = (((smem_b_addr + 96) >> 4) & 0x3FFF) + (read_stage) * 1792;
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_3) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_3) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2((tmem_accum + (acc_stage * 128)), a_desc + 0, b_desc + 0,
                                0x10200480U, tmem_tmem_sfa + 12 + 0, tmem_tmem_sfb + 12 + 0, ((((0) ? init_flag : 0)) ? 0 : 1));
                        }
                    }
                    elect_commit_cg2_multicast(tma_empty_addr + (read_stage) * 8, (uint16_t)(3));
                }
                mbarrier_wait_token(tma_full_addr + (mma_stage) * 8, mma_phase, peek_full);
                unsigned int read_stage_1 = mma_stage;
                mma_stage += 1;
                if (mma_stage == 7) { mma_stage = 0; mma_phase ^= 1; }
                asm volatile("tcgen05.fence::after_thread_sync;");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                int _mma_a_lo_4 = (((smem_d_addr) >> 4) & 0x3FFF) + (read_stage_1) * 1792;
                int _mma_b_lo_4 = (((smem_l1_addr) >> 4) & 0x3FFF) + (read_stage_1) * 1792;
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
                    "mov.b32 id, 270533776;\n\t"
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
                    "}\n"
                    :: "r"(_mma_a_lo_4), "r"(_mma_b_lo_4), "r"((tmem_accum + (acc_stage * 128))), "r"(1));
                elect_commit_cg2_multicast(tma_empty_addr + (read_stage_1) * 8, (uint16_t)(3));
                elect_commit_cg2_multicast(acc_full_addr + (acc_stage) * 8, (uint16_t)(3));
            }
            asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
            if (cta_rank == 0) {
                mbarrier_wait(acc_empty_addr + (acc_stage) * 8, 0);
            }
            int dealloc_peer_rank = cta_rank ^ 1;
            if (cta_rank != 0) {
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(tmem_dealloc_bar_addr), "r"(dealloc_peer_rank) : "memory");
            }
            mbarrier_wait(tmem_dealloc_bar_addr, 0);
            if (cta_rank == 0) {
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(tmem_dealloc_bar_addr), "r"(dealloc_peer_rank) : "memory");
            }
            int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
            asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(512));
        }
    }
    // ---- Role: mainloop_prefetch ----
    if (warp == 1) {
        { // mainloop_prefetch_main
            if (elect_sync()) {
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A))) : "memory");
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B))) : "memory");
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SFA))) : "memory");
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SFB))) : "memory");
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&D))) : "memory");
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&L1))) : "memory");
            }
        }
    }
    // ---- Role: load ----
    if (warp == 2) {
        { // load_main
            unsigned int load_stage = 0;
            unsigned int load_phase = 1;
            uint32_t _mbar_token_0 = mbarrier_try_wait(tma_empty_addr + (load_stage) * 8, load_phase);
            unsigned int peek_empty = _mbar_token_0;
            int bid_m = blockIdx.x;
            int bid_n = blockIdx.y;
            int off_m = bid_m * 128;
            int off_n = bid_n * 128;
            int local_off_n = off_n + cta_rank * 64;
            #pragma unroll 1
            for (unsigned int k_tile = 0; k_tile < K_tiles; k_tile++) {
                mbarrier_wait_token(tma_empty_addr + (load_stage) * 8, load_phase, peek_empty);
                unsigned int write_stage = load_stage;
                if (cta_rank == 0) {
                    if (elect_sync()) {
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((tma_full_addr + (write_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(57344)) : "memory");
                    }
                }
                load_stage += 1;
                if (load_stage == 7) { load_stage = 0; load_phase ^= 1; }
                uint32_t _mbar_token_1 = mbarrier_try_wait(tma_empty_addr + (load_stage) * 8, load_phase);
                peek_empty = _mbar_token_1;
                if (elect_sync()) {
                    tma_3d_gmem2smem_cta2(smem_a_addr + write_stage * 28672, (&A), 0, off_m, k_tile, ((tma_full_addr + (write_stage) * 8) & 0xFEFFFFFF));
                }
                if (elect_sync()) {
                    tma_3d_gmem2smem_cta2(smem_b_addr + write_stage * 28672, (&B), 0, local_off_n, k_tile, ((tma_full_addr + (write_stage) * 8) & 0xFEFFFFFF));
                }
                if (elect_sync()) {
                    tma_4d_gmem2smem_cta2(smem_sfa_addr + write_stage * 28672, (&SFA), 0, 4 * k_tile, 0, bid_m, ((tma_full_addr + (write_stage) * 8) & 0xFEFFFFFF));
                }
                if (elect_sync()) {
                    tma_4d_gmem2smem_cta2(smem_sfb_addr + write_stage * 28672, (&SFB), 0, 4 * k_tile, 0, bid_n, ((tma_full_addr + (write_stage) * 8) & 0xFEFFFFFF));
                }
            }
            mbarrier_wait_token(tma_empty_addr + (load_stage) * 8, load_phase, peek_empty);
            unsigned int write_stage_1 = load_stage;
            if (cta_rank == 0) {
                if (elect_sync()) {
                    asm volatile(
                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                        :: "r"((tma_full_addr + (write_stage_1) * 8) & 0xFEFFFFFF), "r"((uint32_t)(57344)) : "memory");
                }
            }
            load_stage += 1;
            if (load_stage == 7) { load_stage = 0; load_phase ^= 1; }
            if (elect_sync()) {
                tma_2d_gmem2smem_cta2(smem_d_addr + write_stage_1 * 28672, (&D), 0, off_m, ((tma_full_addr + (write_stage_1) * 8) & 0xFEFFFFFF));
            }
            if (elect_sync()) {
                tma_2d_gmem2smem_cta2(smem_l1_addr + write_stage_1 * 28672, (&L1), 0, local_off_n, ((tma_full_addr + (write_stage_1) * 8) & 0xFEFFFFFF));
            }
            if (elect_sync()) {
                tma_4d_gmem2smem_cta2(smem_sfa_addr + write_stage_1 * 28672, (&SFA), 0, 0, 0, bid_m, ((tma_full_addr + (write_stage_1) * 8) & 0xFEFFFFFF));
            }
            if (elect_sync()) {
                tma_4d_gmem2smem_cta2(smem_sfb_addr + write_stage_1 * 28672, (&SFB), 0, 0, 0, bid_n, ((tma_full_addr + (write_stage_1) * 8) & 0xFEFFFFFF));
            }
            #pragma unroll
            for (int _tail = 0; _tail < 7; _tail++) {
                mbarrier_wait(tma_empty_addr + (load_stage) * 8, load_phase);
                load_stage += 1;
                if (load_stage == 7) { load_stage = 0; load_phase ^= 1; }
            }
        }
    }
    // ---- Role: output_prefetch ----
    if (warp == 3) {
        { // output_prefetch_main
            if (elect_sync()) {
                asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&out))) : "memory");
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 4 && warp <= 7) {
        { // epilogue_main
            const int epi_warp = warp - 4;
            unsigned int acc_stage_1 = 0;
            unsigned int store_stage = 0;
            int bid_m_1 = blockIdx.x;
            int bid_n_1 = blockIdx.y;
            int off_m_1 = bid_m_1 * 128;
            int off_n_1 = bid_n_1 * 128;
            int local_off_n_1 = off_n_1 + cta_rank * 64;
            unsigned int _phase_acc_full = 0;
            mbarrier_wait(acc_full_addr + (acc_stage_1) * 8, _phase_acc_full);
            asm volatile("tcgen05.fence::after_thread_sync;");
            float alpha_value = alpha[0];
            #pragma unroll
            for (int subtile = 0; subtile < 4; subtile++) {
                int tmem_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + acc_stage_1 * 128 + (unsigned int)(subtile * 32);
                float _tmem_load_0[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                    : "r"(tmem_addr));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float result[32];
                #pragma unroll
                for (int j = 0; j < 32; j++) {
                    result[j] = _tmem_load_0[j] * alpha_value;
                }
                uint32_t result_bf16[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(result[_lp*2 + 0], result[_lp*2+1 + 0]));
                    result_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                if (subtile > 0) {
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                    if (warp == 4) {
                        if (elect_sync()) {
                            tma_store_2d((&out), off_n_1 + (subtile - 1) * 32, off_m_1, smem_out_addr + store_stage * 8192);
                        }
                    }
                    if (warp == 4) {
                        asm volatile("cp.async.bulk.commit_group;");
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                    }
                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                    store_stage = store_stage ^ 1;
                }
                int out_stage_row = store_stage * 128 + (unsigned int)(epi_warp * 32) + (unsigned int)lane;
                unsigned int out_abs = smem_out_addr + (unsigned int)(out_stage_row * 64);
                unsigned int out_swz = out_abs / 8 & 48;
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_out_addr + ((unsigned int)(out_stage_row * 64) + (0 ^ out_swz)))), "r"(result_bf16[0]), "r"(result_bf16[1]), "r"(result_bf16[2]), "r"(result_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_out_addr + ((unsigned int)(out_stage_row * 64) + (16 ^ out_swz)))), "r"(result_bf16[4]), "r"(result_bf16[5]), "r"(result_bf16[6]), "r"(result_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_out_addr + ((unsigned int)(out_stage_row * 64) + (32 ^ out_swz)))), "r"(result_bf16[8]), "r"(result_bf16[9]), "r"(result_bf16[10]), "r"(result_bf16[11]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_out_addr + ((unsigned int)(out_stage_row * 64) + (48 ^ out_swz)))), "r"(result_bf16[12]), "r"(result_bf16[13]), "r"(result_bf16[14]), "r"(result_bf16[15]) : "memory");
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("barrier.sync 1, 128;" ::: "memory");
            if (warp == 4) {
                if (elect_sync()) {
                    tma_store_2d((&out), off_n_1 + 96, off_m_1, smem_out_addr + store_stage * 8192);
                }
            }
            if (warp == 4) {
                asm volatile("cp.async.bulk.commit_group;");
                asm volatile("cp.async.bulk.wait_group.read 1;");
            }
            asm volatile("barrier.sync 1, 128;" ::: "memory");
            if (warp == 4) {
                asm volatile("cp.async.bulk.wait_group.read 0;");
            }
            if (elect_sync()) {
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((acc_empty_addr + (acc_stage_1) * 8) & 0xFEFFFFFF) : "memory");
            }
        }
    }

    // Cleanup
}

} // extern "C"
