/*
 * Copyright (c) 2026 by FlashInfer team.
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
#define TMEM_NCOLS 512
#define TMEM_ACC_OFFSET 0
#define TMEM_TMEM_SFA_OFFSET 256
#define TMEM_TMEM_SFB_OFFSET 304
#define NUM_TMA_PIPE_STAGES 6
#define NUM_SF_PIPE_STAGES 5
#define NUM_ACC_PIPE_STAGES 1
#define NUM_CLC_PIPE_STAGES 2
#define NUM_THR_PIPE_STAGES 2
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 32768
#define SMEM_SMEM_B_OFF 17408
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 32768
#define SMEM_SMEM_V2_OFF 1024
#define SMEM_SMEM_V2_STAGE_BYTES 6144
#define SMEM_SMEM_V2_STRIDE 32768
#define SMEM_SMEM_V2_0_OFF 1072
#define SMEM_SMEM_V2_0_STAGE_BYTES 6144
#define SMEM_SMEM_V2_0_STRIDE 32768
#define SMEM_SMEM_V2_1_OFF 1120
#define SMEM_SMEM_V2_1_STAGE_BYTES 6144
#define SMEM_SMEM_V2_1_STRIDE 32768
#define SMEM_SMEM_V2_2_OFF 1040
#define SMEM_SMEM_V2_2_STAGE_BYTES 6144
#define SMEM_SMEM_V2_2_STRIDE 32768
#define SMEM_SMEM_V2_3_OFF 1088
#define SMEM_SMEM_V2_3_STAGE_BYTES 6144
#define SMEM_SMEM_V2_3_STRIDE 32768
#define SMEM_SMEM_V2_4_OFF 1136
#define SMEM_SMEM_V2_4_STAGE_BYTES 6144
#define SMEM_SMEM_V2_4_STRIDE 32768
#define SMEM_SMEM_V2_5_OFF 1056
#define SMEM_SMEM_V2_5_STAGE_BYTES 6144
#define SMEM_SMEM_V2_5_STRIDE 32768
#define SMEM_SMEM_V2_6_OFF 1104
#define SMEM_SMEM_V2_6_STAGE_BYTES 6144
#define SMEM_SMEM_V2_6_STRIDE 32768
#define SMEM_SMEM_V2_7_OFF 1024
#define SMEM_SMEM_V2_7_STAGE_BYTES 6144
#define SMEM_SMEM_V2_7_STRIDE 32768
#define SMEM_SMEM_V2_8_OFF 17408
#define SMEM_SMEM_V2_8_STAGE_BYTES 6144
#define SMEM_SMEM_V2_8_STRIDE 32768
#define SMEM_SMEM_V2_9_OFF 17456
#define SMEM_SMEM_V2_9_STAGE_BYTES 6144
#define SMEM_SMEM_V2_9_STRIDE 32768
#define SMEM_SMEM_V2_10_OFF 17504
#define SMEM_SMEM_V2_10_STAGE_BYTES 6144
#define SMEM_SMEM_V2_10_STRIDE 32768
#define SMEM_SMEM_V2_11_OFF 17424
#define SMEM_SMEM_V2_11_STAGE_BYTES 6144
#define SMEM_SMEM_V2_11_STRIDE 32768
#define SMEM_SMEM_V2_12_OFF 17472
#define SMEM_SMEM_V2_12_STAGE_BYTES 6144
#define SMEM_SMEM_V2_12_STRIDE 32768
#define SMEM_SMEM_V2_13_OFF 17520
#define SMEM_SMEM_V2_13_STAGE_BYTES 6144
#define SMEM_SMEM_V2_13_STRIDE 32768
#define SMEM_SMEM_V2_14_OFF 17440
#define SMEM_SMEM_V2_14_STAGE_BYTES 6144
#define SMEM_SMEM_V2_14_STRIDE 32768
#define SMEM_SMEM_V2_15_OFF 17488
#define SMEM_SMEM_V2_15_STAGE_BYTES 6144
#define SMEM_SMEM_V2_15_STRIDE 32768
#define SMEM_SMEM_V2_16_OFF 17408
#define SMEM_SMEM_V2_16_STAGE_BYTES 6144
#define SMEM_SMEM_V2_16_STRIDE 32768
#define SMEM_SMEM_SFA_OFF 197632
#define SMEM_SMEM_SFA_STAGE_BYTES 2048
#define SMEM_SMEM_SFA_STRIDE 6144
#define SMEM_SMEM_V21_OFF 199680
#define SMEM_SMEM_V21_STAGE_BYTES 1024
#define SMEM_SMEM_V21_STRIDE 6144
#define SMEM_SMEM_V22_OFF 200704
#define SMEM_SMEM_V22_STAGE_BYTES 1024
#define SMEM_SMEM_V22_STRIDE 6144
#define SMEM_SMEM_V23_OFF 201728
#define SMEM_SMEM_V23_STAGE_BYTES 1024
#define SMEM_SMEM_V23_STRIDE 6144
#define SMEM_SMEM_V24_OFF 202752
#define SMEM_SMEM_V24_STAGE_BYTES 1024
#define SMEM_SMEM_V24_STRIDE 6144
#define SMEM_SMEM_V25_OFF 199680
#define SMEM_SMEM_V25_STAGE_BYTES 2048
#define SMEM_SMEM_V25_STRIDE 6144
#define SMEM_SMEM_V26_OFF 201728
#define SMEM_SMEM_V26_STAGE_BYTES 2048
#define SMEM_SMEM_V26_STRIDE 6144
#define SMEM_CLC_RESP_OFF 228352
#define SMEM_CLC_RESP_STAGE_BYTES 16
#define SMEM_CLC_RESP_STRIDE 16
#define SMEM_TOTAL 228480
#define THREADS 256
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


__device__ __forceinline__ void tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
    int sfa_taddr, int sfb_taddr, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16"
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


__device__ __forceinline__ void mbarrier_arrive(int mbar_addr) {
    asm volatile(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0];"
        :: "r"(mbar_addr) : "memory");
}





// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)




__device__ __forceinline__ uint64_t make_sf_cp_desc_lo_sbo128(int lo) {
    const int SBO = 128;
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




extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) void
kernel_cake_nvfp4_per_token_7e68f46d2c17ead412fc(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, const __grid_constant__ CUtensorMap A_h, const __grid_constant__ CUtensorMap B_h, const __grid_constant__ CUtensorMap SFA_h, uint8_t* __restrict__ SFA_RAW, uint8_t* __restrict__ SFB_RAW, float* __restrict__ alpha, __half* __restrict__ out, int M, int N, int K_tiles, int tok_tiles, int num_tiles, long long out_stride_m2)
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
    #define tma_empty_addr (mbar_base + 48)
    #define sf_full_addr (mbar_base + 96)
    #define sf_empty_addr (mbar_base + 136)
    #define acc_full_addr (mbar_base + 176)
    #define acc_empty_addr (mbar_base + 184)
    #define tmem_dealloc_bar_addr (mbar_base + 192)
    #define clc_full_addr (mbar_base + 200)
    #define clc_empty_addr (mbar_base + 216)
    #define thr_full_addr (mbar_base + 232)
    #define thr_empty_addr (mbar_base + 248)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));
    unsigned int cluster_size;
    asm volatile("mov.u32 %0, %%cluster_nctarank;" : "=r"(cluster_size));

    // Kernel setup ops
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int smem_b_addr = smem + 17408;
    uint8_t* smem_v2 = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_v2_addr = smem + 1024;
    uint8_t* smem_v2_0 = reinterpret_cast<uint8_t*>(smem_raw + 1072);
    const int smem_v2_addr_0 = smem + 1072;
    uint8_t* smem_v2_1 = reinterpret_cast<uint8_t*>(smem_raw + 1120);
    const int smem_v2_addr_1 = smem + 1120;
    uint8_t* smem_v2_2 = reinterpret_cast<uint8_t*>(smem_raw + 1040);
    const int smem_v2_addr_2 = smem + 1040;
    uint8_t* smem_v2_3 = reinterpret_cast<uint8_t*>(smem_raw + 1088);
    const int smem_v2_addr_3 = smem + 1088;
    uint8_t* smem_v2_4 = reinterpret_cast<uint8_t*>(smem_raw + 1136);
    const int smem_v2_addr_4 = smem + 1136;
    uint8_t* smem_v2_5 = reinterpret_cast<uint8_t*>(smem_raw + 1056);
    const int smem_v2_addr_5 = smem + 1056;
    uint8_t* smem_v2_6 = reinterpret_cast<uint8_t*>(smem_raw + 1104);
    const int smem_v2_addr_6 = smem + 1104;
    uint8_t* smem_v2_7 = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_v2_addr_7 = smem + 1024;
    uint8_t* smem_v2_8 = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int smem_v2_addr_8 = smem + 17408;
    uint8_t* smem_v2_9 = reinterpret_cast<uint8_t*>(smem_raw + 17456);
    const int smem_v2_addr_9 = smem + 17456;
    uint8_t* smem_v2_10 = reinterpret_cast<uint8_t*>(smem_raw + 17504);
    const int smem_v2_addr_10 = smem + 17504;
    uint8_t* smem_v2_11 = reinterpret_cast<uint8_t*>(smem_raw + 17424);
    const int smem_v2_addr_11 = smem + 17424;
    uint8_t* smem_v2_12 = reinterpret_cast<uint8_t*>(smem_raw + 17472);
    const int smem_v2_addr_12 = smem + 17472;
    uint8_t* smem_v2_13 = reinterpret_cast<uint8_t*>(smem_raw + 17520);
    const int smem_v2_addr_13 = smem + 17520;
    uint8_t* smem_v2_14 = reinterpret_cast<uint8_t*>(smem_raw + 17440);
    const int smem_v2_addr_14 = smem + 17440;
    uint8_t* smem_v2_15 = reinterpret_cast<uint8_t*>(smem_raw + 17488);
    const int smem_v2_addr_15 = smem + 17488;
    uint8_t* smem_v2_16 = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int smem_v2_addr_16 = smem + 17408;
    uint8_t* smem_sfa = reinterpret_cast<uint8_t*>(smem_raw + 197632);
    const int smem_sfa_addr = smem + 197632;
    uint8_t* smem_v21 = reinterpret_cast<uint8_t*>(smem_raw + 199680);
    const int smem_v21_addr = smem + 199680;
    uint8_t* smem_v22 = reinterpret_cast<uint8_t*>(smem_raw + 200704);
    const int smem_v22_addr = smem + 200704;
    uint8_t* smem_v23 = reinterpret_cast<uint8_t*>(smem_raw + 201728);
    const int smem_v23_addr = smem + 201728;
    uint8_t* smem_v24 = reinterpret_cast<uint8_t*>(smem_raw + 202752);
    const int smem_v24_addr = smem + 202752;
    uint8_t* smem_v25 = reinterpret_cast<uint8_t*>(smem_raw + 199680);
    const int smem_v25_addr = smem + 199680;
    uint8_t* smem_v26 = reinterpret_cast<uint8_t*>(smem_raw + 201728);
    const int smem_v26_addr = smem + 201728;
    unsigned int* clc_resp = reinterpret_cast<unsigned int*>(smem_raw + 228352);
    const int clc_resp_addr = smem + 228352;
    if (warp == 2) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A))) : "memory"); }
    if (warp == 2) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B))) : "memory"); }
    if (warp == 2) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SFA))) : "memory"); }
    if (warp == 2) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SFB))) : "memory"); }
    if (warp == 2) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A_h))) : "memory"); }
    if (warp == 2) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B_h))) : "memory"); }
    if (warp == 2) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SFA_h))) : "memory"); }
    if (tid == 0) {
        #pragma unroll
        for (int st = 0; st < 6; st++) {
            mbarrier_init(tma_empty_addr + (st) * 8, (((int)cluster_size == 8) ? 3 : 1));
        }
        #pragma unroll
        for (int st_1 = 0; st_1 < 5; st_1++) {
            mbarrier_init(sf_empty_addr + (st_1) * 8, (((int)cluster_size == 8) ? 3 : 1));
        }
        #pragma unroll
        for (int st_2 = 0; st_2 < 2; st_2++) {
            mbarrier_init(clc_empty_addr + (st_2) * 8, 32 + 224 * (int)cluster_size);
        }
    }

    // Mbarrier init (11 pipeline groups, 0 ordered-sequence groups, 33 barriers)
    // Mbarriers at smem_raw[0..264)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full: 6 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // --- pipeline 'sf_pipe' ---
            // sf_full: 5 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            // --- pipeline 'acc_pipe' ---
            // acc_full: 1 barriers, init_count=1
            mbarrier_init(smem + 176, 1);
            // acc_empty: 1 barriers, init_count=256
            mbarrier_init(smem + 184, 256);
            // tmem_dealloc_bar: 1 barriers, init_count=32
            mbarrier_init(smem + 192, 32);
            // --- pipeline 'clc_pipe' ---
            // clc_full: 2 barriers, init_count=1
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            // --- pipeline 'thr_pipe' ---
            // thr_full: 2 barriers, init_count=32
            mbarrier_init(smem + 232, 32);
            mbarrier_init(smem + 240, 32);
            // thr_empty: 2 barriers, init_count=32
            mbarrier_init(smem + 248, 32);
            mbarrier_init(smem + 256, 32);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    // Publish explicit kernel-setup mbarrier initialization.
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");

    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 400 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 264);
    if (warp == 2) {
        int _tmem_hold = smem + 264;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    if (warp == 1 || warp == 2 || warp == 4 || warp == 5 || warp == 6 || warp == 7) {
        asm volatile("barrier.sync.aligned %0, %1;" :: "r"(2), "r"(192) : "memory");
    }
    const int taddr = (warp == 1 || warp == 2 || warp == 4 || warp == 5 || warp == 6 || warp == 7) ? tmem_addr_storage[0] : 0;

    // Kernel post-init ops
    const int tmem_acc = taddr;
    const int tmem_tmem_sfa = taddr + 256;
    const int tmem_tmem_sfb = taddr + 304;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            unsigned int load_stage = 0;
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int load_clc_stage = 0;
            unsigned int load_thr_stage = 0;
            unsigned int load_rank = cta_rank;
            int load_cs = (int)cluster_size;
            unsigned int tile_l = (2 * ((cluster_id >> 2) / (unsigned int)(tok_tiles / 2)) + (cluster_id >> 1 & 1)) * (unsigned int)tok_tiles + (2 * ((cluster_id >> 2) % (unsigned int)(tok_tiles / 2)) + (cluster_id & 1));
            unsigned int _phase_thr_empty = 1;
            unsigned int _phase_tma_empty = 1;
            unsigned int _phase_clc_full = 0;
            #pragma unroll 1
            for (unsigned int _p = 0; _p < num_tiles; _p++) {
                if (cta_rank == 0) {
                    mbarrier_wait(thr_empty_addr + (load_thr_stage) * 8, _phase_thr_empty);
                    mbarrier_arrive(thr_full_addr + (load_thr_stage) * 8);
                    load_thr_stage += 1;
                    if (load_thr_stage == 2) { load_thr_stage = 0; _phase_thr_empty ^= 1; }
                }
                if (load_cs == 8) {
                    int w_idx = tile_l / (unsigned int)tok_tiles;
                    int tok_idx = tile_l - (unsigned int)(w_idx * tok_tiles);
                    int a_rows = tok_idx * 256 + (cta_rank & 1) * 128 + (cta_rank >> 2) * 64;
                    int b_rows = w_idx * 256 + (cta_rank & 1) * 128 + (cta_rank >> 1 & 1) * 64;
                    unsigned int a_half = (unsigned int)((cta_rank >> 2) * 8192);
                    unsigned int b_half = (unsigned int)((cta_rank >> 1 & 1) * 8192);
                    int a_mask = 17 << (cta_rank & 3);
                    int b_mask = 5 << (cta_rank & 5);
                    #pragma unroll 1
                    for (unsigned int k_tile = 0; k_tile < K_tiles; k_tile++) {
                        mbarrier_wait(tma_empty_addr + (load_stage) * 8, _phase_tma_empty);
                        if ((cta_rank & 1) == 0) {
                            if (elect_sync()) {
                                asm volatile(
                                    "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                    :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(65536)) : "memory");
                            }
                        }
                        if (elect_sync()) {
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6, %7;"
                                :: "r"(smem_a_addr + load_stage * 32768 + a_half), "l"((&A_h)), "r"(0), "r"(a_rows), "r"(k_tile),
                                   "r"(((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(a_mask)), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6, %7;"
                                :: "r"(smem_b_addr + load_stage * 32768 + b_half), "l"((&B_h)), "r"(0), "r"(b_rows), "r"(k_tile),
                                   "r"(((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(b_mask)), "l"(0x14F0000000000000ULL) : "memory");
                        }
                        if ((int)k_tile + 6 < K_tiles) {
                            if (elect_sync()) {
                                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&A_h))), "r"((int)(0)), "r"((int)(a_rows)), "r"((int)(k_tile + 6)) : "memory");
                                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&B_h))), "r"((int)(0)), "r"((int)(b_rows)), "r"((int)(k_tile + 6)) : "memory");
                            }
                        }
                        load_stage += 1;
                        if (load_stage == 6) { load_stage = 0; _phase_tma_empty ^= 1; }
                    }
                } else {
                    int w_idx_1 = tile_l / (unsigned int)tok_tiles;
                    int tok_idx_1 = tile_l - (unsigned int)(w_idx_1 * tok_tiles);
                    int a_rows_1 = tok_idx_1 * 256 + cta_rank * 128;
                    int b_rows_1 = w_idx_1 * 256 + cta_rank * 128;
                    #pragma unroll 1
                    for (unsigned int k_tile_1 = 0; k_tile_1 < K_tiles; k_tile_1++) {
                        mbarrier_wait(tma_empty_addr + (load_stage) * 8, _phase_tma_empty);
                        if (cta_rank == 0) {
                            if (elect_sync()) {
                                asm volatile(
                                    "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                    :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(65536)) : "memory");
                            }
                        }
                        if (elect_sync()) {
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(smem_a_addr + load_stage * 32768), "l"((&A)), "r"(0), "r"(a_rows_1), "r"(k_tile_1),
                                   "r"(((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(smem_b_addr + load_stage * 32768), "l"((&B)), "r"(0), "r"(b_rows_1), "r"(k_tile_1),
                                   "r"(((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        }
                        load_stage += 1;
                        if (load_stage == 6) { load_stage = 0; _phase_tma_empty ^= 1; }
                    }
                }
                mbarrier_wait(clc_full_addr + (load_clc_stage) * 8, _phase_clc_full);
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
                    : "r"(clc_resp_addr + load_clc_stage * 16 + 0 * 16)
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
                    : "r"(clc_resp_addr + load_clc_stage * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(clc_empty_addr + (load_clc_stage) * 8), "r"(0) : "memory");
                load_clc_stage += 1;
                if (load_clc_stage == 2) { load_clc_stage = 0; _phase_clc_full ^= 1; }
                if (_clc_valid_1 == 0) {
                    break;
                }
                tile_l = (2 * (((_clc_ctaid_0 + load_rank) / 2 >> 2) / (unsigned int)(tok_tiles / 2)) + ((_clc_ctaid_0 + load_rank) / 2 >> 1 & 1)) * (unsigned int)tok_tiles + (2 * (((_clc_ctaid_0 + load_rank) / 2 >> 2) % (unsigned int)(tok_tiles / 2)) + ((_clc_ctaid_0 + load_rank) / 2 & 1));
            }
            #pragma unroll 1
            for (int _ = 0; _ < 6; _++) {
                mbarrier_wait(tma_empty_addr + (load_stage) * 8, _phase_tma_empty);
                load_stage += 1;
                if (load_stage == 6) { load_stage = 0; _phase_tma_empty ^= 1; }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            unsigned int mma_stage = 0;
            unsigned int mma_sf_stage = 0;
            unsigned int completed = 0;
            unsigned int mma_clc_stage = 0;
            unsigned int mma_rank = cta_rank;
            int mma_cs = (int)cluster_size;
            int rel_mask = 51 << (cta_rank & 2) | 15 << (cta_rank & 4);
            int acc_mask = 3 << (cta_rank & 6);
            unsigned int tile_m = (2 * ((cluster_id >> 2) / (unsigned int)(tok_tiles / 2)) + (cluster_id >> 1 & 1)) * (unsigned int)tok_tiles + (2 * ((cluster_id >> 2) % (unsigned int)(tok_tiles / 2)) + (cluster_id & 1));
            unsigned int _phase_acc_empty = 1;
            unsigned int _phase_sf_full = 0;
            unsigned int _phase_tma_full = 0;
            unsigned int _phase_clc_full_1 = 0;
            #pragma unroll 1
            for (unsigned int _p_1 = 0; _p_1 < num_tiles; _p_1++) {
                if ((cta_rank & 1) == 0) {
                    if (mma_cs == 8) {
                        int acc_base = (int)0 * 256;
                        int w_idx_mma = tile_m / (unsigned int)tok_tiles;
                        int sfb_shift = w_idx_mma * 256 % 128 / 32;
                        mbarrier_wait(acc_empty_addr, _phase_acc_empty);
                        int k96_supers = K_tiles / 3;
                        int k96_tail = K_tiles - k96_supers * 3;
                        #pragma unroll 1
                        for (unsigned int k_super = 0; k_super < k96_supers; k_super++) {
                            int init_super = ((k_super == 0) ? 1 : 0);
                            mbarrier_wait(sf_full_addr + (mma_sf_stage) * 8, _phase_sf_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            if (elect_sync()) {
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa, make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_sf_stage) * 384)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfa + 4), make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_sf_stage) * 384 + 32)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfa + 8), make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_sf_stage) * 384 + 64)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfa + 12), make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_sf_stage) * 384 + 96)));
                            }
                            if (elect_sync()) {
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb, make_sf_cp_desc_lo_sbo128((((smem_v25_addr) >> 4) + (mma_sf_stage) * 384)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 4), make_sf_cp_desc_lo_sbo128((((smem_v25_addr) >> 4) + (mma_sf_stage) * 384 + 32)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 8), make_sf_cp_desc_lo_sbo128((((smem_v25_addr) >> 4) + (mma_sf_stage) * 384 + 64)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 12), make_sf_cp_desc_lo_sbo128((((smem_v25_addr) >> 4) + (mma_sf_stage) * 384 + 96)));
                            }
                            if (elect_sync()) {
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + 16, make_sf_cp_desc_lo_sbo128((((smem_v26_addr) >> 4) + (mma_sf_stage) * 384)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 16 + 4), make_sf_cp_desc_lo_sbo128((((smem_v26_addr) >> 4) + (mma_sf_stage) * 384 + 32)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 16 + 8), make_sf_cp_desc_lo_sbo128((((smem_v26_addr) >> 4) + (mma_sf_stage) * 384 + 64)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 16 + 12), make_sf_cp_desc_lo_sbo128((((smem_v26_addr) >> 4) + (mma_sf_stage) * 384 + 96)));
                            }
                            elect_commit_cg2_multicast(sf_empty_addr + (mma_sf_stage) * 8, (uint16_t)(rel_mask));
                            mma_sf_stage += 1;
                            if (mma_sf_stage == 5) { mma_sf_stage = 0; _phase_sf_full ^= 1; }
                            mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            unsigned int st_p0 = mma_stage;
                            int mma_a_lo = (((smem_v2_addr) >> 4) & 0x3FFF) + (st_p0) * 2048;
                            int mma_b_lo = (((smem_v2_addr_8) >> 4) & 0x3FFF) + (st_p0) * 2048;
                            int mma_a_next_lo = (((smem_v2_addr_7) >> 4) & 0x3FFF) + (st_p0) * 2048;
                            int mma_b_next_lo = (((smem_v2_addr_16) >> 4) & 0x3FFF) + (st_p0) * 2048;
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)mma_a_lo | ((uint64_t)((uint32_t)mma_a_next_lo & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)mma_b_lo | ((uint64_t)((uint32_t)mma_b_next_lo & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2((tmem_acc + (acc_base)), a_desc + 0, b_desc + 0,
                                        0x90400480U, tmem_tmem_sfa + 0, tmem_tmem_sfb + sfb_shift + 0, ((init_super) ? 0 : 1));
                                }
                            }
                            int mma_a_lo_0 = (((smem_v2_addr_0) >> 4) & 0x3FFF) + (st_p0) * 2048;
                            int mma_b_lo_0 = (((smem_v2_addr_9) >> 4) & 0x3FFF) + (st_p0) * 2048;
                            int mma_a_next_lo_0 = (((smem_v2_addr_7) >> 4) & 0x3FFF) + (st_p0) * 2048;
                            int mma_b_next_lo_0 = (((smem_v2_addr_16) >> 4) & 0x3FFF) + (st_p0) * 2048;
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)mma_a_lo_0 | ((uint64_t)((uint32_t)mma_a_next_lo_0 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)mma_b_lo_0 | ((uint64_t)((uint32_t)mma_b_next_lo_0 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2((tmem_acc + (acc_base)), a_desc + 0, b_desc + 0,
                                        0xd04004a0U, tmem_tmem_sfa + 4 + 0, tmem_tmem_sfb + (8 + sfb_shift) + 0, 1);
                                }
                            }
                            mma_stage += 1;
                            if (mma_stage == 6) { mma_stage = 0; _phase_tma_full ^= 1; }
                            mbarrier_wait(sf_full_addr + (mma_sf_stage) * 8, _phase_sf_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            if (elect_sync()) {
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + 16, make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_sf_stage) * 384)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfa + 16 + 4), make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_sf_stage) * 384 + 32)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfa + 16 + 8), make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_sf_stage) * 384 + 64)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfa + 16 + 12), make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_sf_stage) * 384 + 96)));
                            }
                            if (elect_sync()) {
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + 32, make_sf_cp_desc_lo_sbo128((((smem_v25_addr) >> 4) + (mma_sf_stage) * 384)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 32 + 4), make_sf_cp_desc_lo_sbo128((((smem_v25_addr) >> 4) + (mma_sf_stage) * 384 + 32)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 32 + 8), make_sf_cp_desc_lo_sbo128((((smem_v25_addr) >> 4) + (mma_sf_stage) * 384 + 64)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 32 + 12), make_sf_cp_desc_lo_sbo128((((smem_v25_addr) >> 4) + (mma_sf_stage) * 384 + 96)));
                            }
                            if (elect_sync()) {
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + 48, make_sf_cp_desc_lo_sbo128((((smem_v26_addr) >> 4) + (mma_sf_stage) * 384)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 48 + 4), make_sf_cp_desc_lo_sbo128((((smem_v26_addr) >> 4) + (mma_sf_stage) * 384 + 32)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 48 + 8), make_sf_cp_desc_lo_sbo128((((smem_v26_addr) >> 4) + (mma_sf_stage) * 384 + 64)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 48 + 12), make_sf_cp_desc_lo_sbo128((((smem_v26_addr) >> 4) + (mma_sf_stage) * 384 + 96)));
                            }
                            elect_commit_cg2_multicast(sf_empty_addr + (mma_sf_stage) * 8, (uint16_t)(rel_mask));
                            mma_sf_stage += 1;
                            if (mma_sf_stage == 5) { mma_sf_stage = 0; _phase_sf_full ^= 1; }
                            mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            unsigned int st_p1 = mma_stage;
                            int mma_a_lo_1 = (((smem_v2_addr_1) >> 4) & 0x3FFF) + (st_p0) * 2048;
                            int mma_b_lo_1 = (((smem_v2_addr_10) >> 4) & 0x3FFF) + (st_p0) * 2048;
                            int mma_a_next_lo_1 = (((smem_v2_addr_7) >> 4) & 0x3FFF) + (st_p1) * 2048;
                            int mma_b_next_lo_1 = (((smem_v2_addr_16) >> 4) & 0x3FFF) + (st_p1) * 2048;
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)mma_a_lo_1 | ((uint64_t)((uint32_t)mma_a_next_lo_1 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)mma_b_lo_1 | ((uint64_t)((uint32_t)mma_b_next_lo_1 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2((tmem_acc + (acc_base)), a_desc + 0, b_desc + 0,
                                        0x90400480U, tmem_tmem_sfa + 12 + 0, tmem_tmem_sfb + (24 + sfb_shift) + 0, 1);
                                }
                            }
                            elect_commit_cg2_multicast(tma_empty_addr + (((mma_stage + 5 >= 6) ? mma_stage + 5 - 6 : mma_stage + 5)) * 8, (uint16_t)(rel_mask));
                            int mma_a_lo_2 = (((smem_v2_addr_2) >> 4) & 0x3FFF) + (st_p1) * 2048;
                            int mma_b_lo_2 = (((smem_v2_addr_11) >> 4) & 0x3FFF) + (st_p1) * 2048;
                            int mma_a_next_lo_2 = (((smem_v2_addr_7) >> 4) & 0x3FFF) + (st_p1) * 2048;
                            int mma_b_next_lo_2 = (((smem_v2_addr_16) >> 4) & 0x3FFF) + (st_p1) * 2048;
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)mma_a_lo_2 | ((uint64_t)((uint32_t)mma_a_next_lo_2 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)mma_b_lo_2 | ((uint64_t)((uint32_t)mma_b_next_lo_2 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2((tmem_acc + (acc_base)), a_desc + 0, b_desc + 0,
                                        0xd04004a0U, tmem_tmem_sfa + 16 + 0, tmem_tmem_sfb + (32 + sfb_shift) + 0, 1);
                                }
                            }
                            int mma_a_lo_3 = (((smem_v2_addr_3) >> 4) & 0x3FFF) + (st_p1) * 2048;
                            int mma_b_lo_3 = (((smem_v2_addr_12) >> 4) & 0x3FFF) + (st_p1) * 2048;
                            int mma_a_next_lo_3 = (((smem_v2_addr_7) >> 4) & 0x3FFF) + (st_p1) * 2048;
                            int mma_b_next_lo_3 = (((smem_v2_addr_16) >> 4) & 0x3FFF) + (st_p1) * 2048;
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)mma_a_lo_3 | ((uint64_t)((uint32_t)mma_a_next_lo_3 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)mma_b_lo_3 | ((uint64_t)((uint32_t)mma_b_next_lo_3 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2((tmem_acc + (acc_base)), a_desc + 0, b_desc + 0,
                                        0x90400480U, tmem_tmem_sfa + 24 + 0, tmem_tmem_sfb + (48 + sfb_shift) + 0, 1);
                                }
                            }
                            mma_stage += 1;
                            if (mma_stage == 6) { mma_stage = 0; _phase_tma_full ^= 1; }
                            mbarrier_wait(sf_full_addr + (mma_sf_stage) * 8, _phase_sf_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            if (elect_sync()) {
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + 32, make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_sf_stage) * 384)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfa + 32 + 4), make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_sf_stage) * 384 + 32)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfa + 32 + 8), make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_sf_stage) * 384 + 64)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfa + 32 + 12), make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_sf_stage) * 384 + 96)));
                            }
                            if (elect_sync()) {
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + 64, make_sf_cp_desc_lo_sbo128((((smem_v25_addr) >> 4) + (mma_sf_stage) * 384)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 64 + 4), make_sf_cp_desc_lo_sbo128((((smem_v25_addr) >> 4) + (mma_sf_stage) * 384 + 32)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 64 + 8), make_sf_cp_desc_lo_sbo128((((smem_v25_addr) >> 4) + (mma_sf_stage) * 384 + 64)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 64 + 12), make_sf_cp_desc_lo_sbo128((((smem_v25_addr) >> 4) + (mma_sf_stage) * 384 + 96)));
                            }
                            if (elect_sync()) {
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + 80, make_sf_cp_desc_lo_sbo128((((smem_v26_addr) >> 4) + (mma_sf_stage) * 384)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 80 + 4), make_sf_cp_desc_lo_sbo128((((smem_v26_addr) >> 4) + (mma_sf_stage) * 384 + 32)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 80 + 8), make_sf_cp_desc_lo_sbo128((((smem_v26_addr) >> 4) + (mma_sf_stage) * 384 + 64)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 80 + 12), make_sf_cp_desc_lo_sbo128((((smem_v26_addr) >> 4) + (mma_sf_stage) * 384 + 96)));
                            }
                            elect_commit_cg2_multicast(sf_empty_addr + (mma_sf_stage) * 8, (uint16_t)(rel_mask));
                            mma_sf_stage += 1;
                            if (mma_sf_stage == 5) { mma_sf_stage = 0; _phase_sf_full ^= 1; }
                            mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            unsigned int st_p2 = mma_stage;
                            int mma_a_lo_4 = (((smem_v2_addr_4) >> 4) & 0x3FFF) + (st_p1) * 2048;
                            int mma_b_lo_4 = (((smem_v2_addr_13) >> 4) & 0x3FFF) + (st_p1) * 2048;
                            int mma_a_next_lo_4 = (((smem_v2_addr_7) >> 4) & 0x3FFF) + (st_p2) * 2048;
                            int mma_b_next_lo_4 = (((smem_v2_addr_16) >> 4) & 0x3FFF) + (st_p2) * 2048;
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)mma_a_lo_4 | ((uint64_t)((uint32_t)mma_a_next_lo_4 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)mma_b_lo_4 | ((uint64_t)((uint32_t)mma_b_next_lo_4 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2((tmem_acc + (acc_base)), a_desc + 0, b_desc + 0,
                                        0xd04004a0U, tmem_tmem_sfa + 28 + 0, tmem_tmem_sfb + (56 + sfb_shift) + 0, 1);
                                }
                            }
                            elect_commit_cg2_multicast(tma_empty_addr + (((mma_stage + 5 >= 6) ? mma_stage + 5 - 6 : mma_stage + 5)) * 8, (uint16_t)(rel_mask));
                            int mma_a_lo_5 = (((smem_v2_addr_5) >> 4) & 0x3FFF) + (st_p2) * 2048;
                            int mma_b_lo_5 = (((smem_v2_addr_14) >> 4) & 0x3FFF) + (st_p2) * 2048;
                            int mma_a_next_lo_5 = (((smem_v2_addr_7) >> 4) & 0x3FFF) + (st_p2) * 2048;
                            int mma_b_next_lo_5 = (((smem_v2_addr_16) >> 4) & 0x3FFF) + (st_p2) * 2048;
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)mma_a_lo_5 | ((uint64_t)((uint32_t)mma_a_next_lo_5 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)mma_b_lo_5 | ((uint64_t)((uint32_t)mma_b_next_lo_5 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2((tmem_acc + (acc_base)), a_desc + 0, b_desc + 0,
                                        0x90400480U, tmem_tmem_sfa + 36 + 0, tmem_tmem_sfb + (72 + sfb_shift) + 0, 1);
                                }
                            }
                            int mma_a_lo_6 = (((smem_v2_addr_6) >> 4) & 0x3FFF) + (st_p2) * 2048;
                            int mma_b_lo_6 = (((smem_v2_addr_15) >> 4) & 0x3FFF) + (st_p2) * 2048;
                            int mma_a_next_lo_6 = (((smem_v2_addr_7) >> 4) & 0x3FFF) + (st_p2) * 2048;
                            int mma_b_next_lo_6 = (((smem_v2_addr_16) >> 4) & 0x3FFF) + (st_p2) * 2048;
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)mma_a_lo_6 | ((uint64_t)((uint32_t)mma_a_next_lo_6 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)mma_b_lo_6 | ((uint64_t)((uint32_t)mma_b_next_lo_6 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2((tmem_acc + (acc_base)), a_desc + 0, b_desc + 0,
                                        0xd04004a0U, tmem_tmem_sfa + 40 + 0, tmem_tmem_sfb + (80 + sfb_shift) + 0, 1);
                                }
                            }
                            elect_commit_cg2_multicast(tma_empty_addr + (mma_stage) * 8, (uint16_t)(rel_mask));
                            mma_stage += 1;
                            if (mma_stage == 6) { mma_stage = 0; _phase_tma_full ^= 1; }
                        }
                        #pragma unroll 1
                        for (unsigned int k_slot = 0; k_slot < k96_tail; k_slot++) {
                            mbarrier_wait(sf_full_addr + (mma_sf_stage) * 8, _phase_sf_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            if (elect_sync()) {
                                tcgen05_cp_32x128b_warpx4_cta2((unsigned int)tmem_tmem_sfa + k_slot * 16, make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_sf_stage) * 384)));
                                tcgen05_cp_32x128b_warpx4_cta2(((unsigned int)tmem_tmem_sfa + k_slot * 16 + 4), make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_sf_stage) * 384 + 32)));
                                tcgen05_cp_32x128b_warpx4_cta2(((unsigned int)tmem_tmem_sfa + k_slot * 16 + 8), make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_sf_stage) * 384 + 64)));
                                tcgen05_cp_32x128b_warpx4_cta2(((unsigned int)tmem_tmem_sfa + k_slot * 16 + 12), make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_sf_stage) * 384 + 96)));
                            }
                            if (elect_sync()) {
                                tcgen05_cp_32x128b_warpx4_cta2((unsigned int)tmem_tmem_sfb + k_slot * 2 * 16, make_sf_cp_desc_lo_sbo128((((smem_v25_addr) >> 4) + (mma_sf_stage) * 384)));
                                tcgen05_cp_32x128b_warpx4_cta2(((unsigned int)tmem_tmem_sfb + k_slot * 2 * 16 + 4), make_sf_cp_desc_lo_sbo128((((smem_v25_addr) >> 4) + (mma_sf_stage) * 384 + 32)));
                                tcgen05_cp_32x128b_warpx4_cta2(((unsigned int)tmem_tmem_sfb + k_slot * 2 * 16 + 8), make_sf_cp_desc_lo_sbo128((((smem_v25_addr) >> 4) + (mma_sf_stage) * 384 + 64)));
                                tcgen05_cp_32x128b_warpx4_cta2(((unsigned int)tmem_tmem_sfb + k_slot * 2 * 16 + 12), make_sf_cp_desc_lo_sbo128((((smem_v25_addr) >> 4) + (mma_sf_stage) * 384 + 96)));
                            }
                            if (elect_sync()) {
                                tcgen05_cp_32x128b_warpx4_cta2((unsigned int)tmem_tmem_sfb + (k_slot * 2 + 1) * 16, make_sf_cp_desc_lo_sbo128((((smem_v26_addr) >> 4) + (mma_sf_stage) * 384)));
                                tcgen05_cp_32x128b_warpx4_cta2(((unsigned int)tmem_tmem_sfb + (k_slot * 2 + 1) * 16 + 4), make_sf_cp_desc_lo_sbo128((((smem_v26_addr) >> 4) + (mma_sf_stage) * 384 + 32)));
                                tcgen05_cp_32x128b_warpx4_cta2(((unsigned int)tmem_tmem_sfb + (k_slot * 2 + 1) * 16 + 8), make_sf_cp_desc_lo_sbo128((((smem_v26_addr) >> 4) + (mma_sf_stage) * 384 + 64)));
                                tcgen05_cp_32x128b_warpx4_cta2(((unsigned int)tmem_tmem_sfb + (k_slot * 2 + 1) * 16 + 12), make_sf_cp_desc_lo_sbo128((((smem_v26_addr) >> 4) + (mma_sf_stage) * 384 + 96)));
                            }
                            elect_commit_cg2_multicast(sf_empty_addr + (mma_sf_stage) * 8, (uint16_t)(rel_mask));
                            mma_sf_stage += 1;
                            if (mma_sf_stage == 5) { mma_sf_stage = 0; _phase_sf_full ^= 1; }
                            mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int tail_sfa_col = (int)k_slot * 16;
                            int tail_sfb_col = (int)k_slot * 32;
                            int init_tail = (((int)k_slot + k96_supers == 0) ? 1 : 0);
                            int _mma_a_lo_0 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage) * 2048;
                            int _mma_b_lo_0 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_stage) * 2048;
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_cta2((tmem_acc + (acc_base)), a_desc + 0, b_desc + 0,
                                        0x10400480U, tmem_tmem_sfa + tail_sfa_col + 0, tmem_tmem_sfb + (tail_sfb_col + sfb_shift) + 0, ((init_tail) ? 0 : 1));
                                }
                            }
                            int _mma_a_lo_1 = (((smem_a_addr + 32) >> 4) & 0x3FFF) + (mma_stage) * 2048;
                            int _mma_b_lo_1 = (((smem_b_addr + 32) >> 4) & 0x3FFF) + (mma_stage) * 2048;
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_cta2((tmem_acc + (acc_base)), a_desc + 0, b_desc + 0,
                                        0x10400480U, tmem_tmem_sfa + (tail_sfa_col + 4) + 0, tmem_tmem_sfb + (tail_sfb_col + 8 + sfb_shift) + 0, 1);
                                }
                            }
                            int _mma_a_lo_2 = (((smem_a_addr + 64) >> 4) & 0x3FFF) + (mma_stage) * 2048;
                            int _mma_b_lo_2 = (((smem_b_addr + 64) >> 4) & 0x3FFF) + (mma_stage) * 2048;
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_2) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_2) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_cta2((tmem_acc + (acc_base)), a_desc + 0, b_desc + 0,
                                        0x10400480U, tmem_tmem_sfa + (tail_sfa_col + 8) + 0, tmem_tmem_sfb + (tail_sfb_col + 16 + sfb_shift) + 0, 1);
                                }
                            }
                            int _mma_a_lo_3 = (((smem_a_addr + 96) >> 4) & 0x3FFF) + (mma_stage) * 2048;
                            int _mma_b_lo_3 = (((smem_b_addr + 96) >> 4) & 0x3FFF) + (mma_stage) * 2048;
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_3) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_3) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_cta2((tmem_acc + (acc_base)), a_desc + 0, b_desc + 0,
                                        0x10400480U, tmem_tmem_sfa + (tail_sfa_col + 12) + 0, tmem_tmem_sfb + (tail_sfb_col + 24 + sfb_shift) + 0, 1);
                                }
                            }
                            elect_commit_cg2_multicast(tma_empty_addr + (mma_stage) * 8, (uint16_t)(rel_mask));
                            mma_stage += 1;
                            if (mma_stage == 6) { mma_stage = 0; _phase_tma_full ^= 1; }
                        }
                        elect_commit_cg2_multicast(acc_full_addr, (uint16_t)(acc_mask));
                    } else {
                        int acc_base_1 = (int)0 * 256;
                        int w_idx_mma_1 = tile_m / (unsigned int)tok_tiles;
                        int sfb_shift_1 = w_idx_mma_1 * 256 % 128 / 32;
                        mbarrier_wait(acc_empty_addr, _phase_acc_empty);
                        int k96_supers_1 = K_tiles / 3;
                        int k96_tail_1 = K_tiles - k96_supers_1 * 3;
                        #pragma unroll 1
                        for (unsigned int k_super_1 = 0; k_super_1 < k96_supers_1; k_super_1++) {
                            int init_super_1 = ((k_super_1 == 0) ? 1 : 0);
                            mbarrier_wait(sf_full_addr + (mma_sf_stage) * 8, _phase_sf_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            if (elect_sync()) {
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa, make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_sf_stage) * 384)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfa + 4), make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_sf_stage) * 384 + 32)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfa + 8), make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_sf_stage) * 384 + 64)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfa + 12), make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_sf_stage) * 384 + 96)));
                            }
                            if (elect_sync()) {
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb, make_sf_cp_desc_lo_sbo128((((smem_v25_addr) >> 4) + (mma_sf_stage) * 384)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 4), make_sf_cp_desc_lo_sbo128((((smem_v25_addr) >> 4) + (mma_sf_stage) * 384 + 32)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 8), make_sf_cp_desc_lo_sbo128((((smem_v25_addr) >> 4) + (mma_sf_stage) * 384 + 64)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 12), make_sf_cp_desc_lo_sbo128((((smem_v25_addr) >> 4) + (mma_sf_stage) * 384 + 96)));
                            }
                            if (elect_sync()) {
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + 16, make_sf_cp_desc_lo_sbo128((((smem_v26_addr) >> 4) + (mma_sf_stage) * 384)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 16 + 4), make_sf_cp_desc_lo_sbo128((((smem_v26_addr) >> 4) + (mma_sf_stage) * 384 + 32)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 16 + 8), make_sf_cp_desc_lo_sbo128((((smem_v26_addr) >> 4) + (mma_sf_stage) * 384 + 64)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 16 + 12), make_sf_cp_desc_lo_sbo128((((smem_v26_addr) >> 4) + (mma_sf_stage) * 384 + 96)));
                            }
                            elect_commit_cg2_multicast(sf_empty_addr + (mma_sf_stage) * 8, (uint16_t)(3));
                            mma_sf_stage += 1;
                            if (mma_sf_stage == 5) { mma_sf_stage = 0; _phase_sf_full ^= 1; }
                            mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            unsigned int st_p0_1 = mma_stage;
                            int mma_a_lo_7 = (((smem_v2_addr) >> 4) & 0x3FFF) + (st_p0_1) * 2048;
                            int mma_b_lo_7 = (((smem_v2_addr_8) >> 4) & 0x3FFF) + (st_p0_1) * 2048;
                            int mma_a_next_lo_7 = (((smem_v2_addr_7) >> 4) & 0x3FFF) + (st_p0_1) * 2048;
                            int mma_b_next_lo_7 = (((smem_v2_addr_16) >> 4) & 0x3FFF) + (st_p0_1) * 2048;
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)mma_a_lo_7 | ((uint64_t)((uint32_t)mma_a_next_lo_7 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)mma_b_lo_7 | ((uint64_t)((uint32_t)mma_b_next_lo_7 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2((tmem_acc + (acc_base_1)), a_desc + 0, b_desc + 0,
                                        0x90400480U, tmem_tmem_sfa + 0, tmem_tmem_sfb + sfb_shift_1 + 0, ((init_super_1) ? 0 : 1));
                                }
                            }
                            int mma_a_lo_7_1 = (((smem_v2_addr_0) >> 4) & 0x3FFF) + (st_p0_1) * 2048;
                            int mma_b_lo_7_1 = (((smem_v2_addr_9) >> 4) & 0x3FFF) + (st_p0_1) * 2048;
                            int mma_a_next_lo_7_1 = (((smem_v2_addr_7) >> 4) & 0x3FFF) + (st_p0_1) * 2048;
                            int mma_b_next_lo_7_1 = (((smem_v2_addr_16) >> 4) & 0x3FFF) + (st_p0_1) * 2048;
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)mma_a_lo_7_1 | ((uint64_t)((uint32_t)mma_a_next_lo_7_1 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)mma_b_lo_7_1 | ((uint64_t)((uint32_t)mma_b_next_lo_7_1 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2((tmem_acc + (acc_base_1)), a_desc + 0, b_desc + 0,
                                        0xd04004a0U, tmem_tmem_sfa + 4 + 0, tmem_tmem_sfb + (8 + sfb_shift_1) + 0, 1);
                                }
                            }
                            mma_stage += 1;
                            if (mma_stage == 6) { mma_stage = 0; _phase_tma_full ^= 1; }
                            mbarrier_wait(sf_full_addr + (mma_sf_stage) * 8, _phase_sf_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            if (elect_sync()) {
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + 16, make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_sf_stage) * 384)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfa + 16 + 4), make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_sf_stage) * 384 + 32)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfa + 16 + 8), make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_sf_stage) * 384 + 64)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfa + 16 + 12), make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_sf_stage) * 384 + 96)));
                            }
                            if (elect_sync()) {
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + 32, make_sf_cp_desc_lo_sbo128((((smem_v25_addr) >> 4) + (mma_sf_stage) * 384)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 32 + 4), make_sf_cp_desc_lo_sbo128((((smem_v25_addr) >> 4) + (mma_sf_stage) * 384 + 32)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 32 + 8), make_sf_cp_desc_lo_sbo128((((smem_v25_addr) >> 4) + (mma_sf_stage) * 384 + 64)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 32 + 12), make_sf_cp_desc_lo_sbo128((((smem_v25_addr) >> 4) + (mma_sf_stage) * 384 + 96)));
                            }
                            if (elect_sync()) {
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + 48, make_sf_cp_desc_lo_sbo128((((smem_v26_addr) >> 4) + (mma_sf_stage) * 384)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 48 + 4), make_sf_cp_desc_lo_sbo128((((smem_v26_addr) >> 4) + (mma_sf_stage) * 384 + 32)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 48 + 8), make_sf_cp_desc_lo_sbo128((((smem_v26_addr) >> 4) + (mma_sf_stage) * 384 + 64)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 48 + 12), make_sf_cp_desc_lo_sbo128((((smem_v26_addr) >> 4) + (mma_sf_stage) * 384 + 96)));
                            }
                            elect_commit_cg2_multicast(sf_empty_addr + (mma_sf_stage) * 8, (uint16_t)(3));
                            mma_sf_stage += 1;
                            if (mma_sf_stage == 5) { mma_sf_stage = 0; _phase_sf_full ^= 1; }
                            mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            unsigned int st_p1_1 = mma_stage;
                            int mma_a_lo_8 = (((smem_v2_addr_1) >> 4) & 0x3FFF) + (st_p0_1) * 2048;
                            int mma_b_lo_8 = (((smem_v2_addr_10) >> 4) & 0x3FFF) + (st_p0_1) * 2048;
                            int mma_a_next_lo_8 = (((smem_v2_addr_7) >> 4) & 0x3FFF) + (st_p1_1) * 2048;
                            int mma_b_next_lo_8 = (((smem_v2_addr_16) >> 4) & 0x3FFF) + (st_p1_1) * 2048;
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)mma_a_lo_8 | ((uint64_t)((uint32_t)mma_a_next_lo_8 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)mma_b_lo_8 | ((uint64_t)((uint32_t)mma_b_next_lo_8 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2((tmem_acc + (acc_base_1)), a_desc + 0, b_desc + 0,
                                        0x90400480U, tmem_tmem_sfa + 12 + 0, tmem_tmem_sfb + (24 + sfb_shift_1) + 0, 1);
                                }
                            }
                            elect_commit_cg2_multicast(tma_empty_addr + (((mma_stage + 5 >= 6) ? mma_stage + 5 - 6 : mma_stage + 5)) * 8, (uint16_t)(3));
                            int mma_a_lo_9 = (((smem_v2_addr_2) >> 4) & 0x3FFF) + (st_p1_1) * 2048;
                            int mma_b_lo_9 = (((smem_v2_addr_11) >> 4) & 0x3FFF) + (st_p1_1) * 2048;
                            int mma_a_next_lo_9 = (((smem_v2_addr_7) >> 4) & 0x3FFF) + (st_p1_1) * 2048;
                            int mma_b_next_lo_9 = (((smem_v2_addr_16) >> 4) & 0x3FFF) + (st_p1_1) * 2048;
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)mma_a_lo_9 | ((uint64_t)((uint32_t)mma_a_next_lo_9 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)mma_b_lo_9 | ((uint64_t)((uint32_t)mma_b_next_lo_9 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2((tmem_acc + (acc_base_1)), a_desc + 0, b_desc + 0,
                                        0xd04004a0U, tmem_tmem_sfa + 16 + 0, tmem_tmem_sfb + (32 + sfb_shift_1) + 0, 1);
                                }
                            }
                            int mma_a_lo_10 = (((smem_v2_addr_3) >> 4) & 0x3FFF) + (st_p1_1) * 2048;
                            int mma_b_lo_10 = (((smem_v2_addr_12) >> 4) & 0x3FFF) + (st_p1_1) * 2048;
                            int mma_a_next_lo_10 = (((smem_v2_addr_7) >> 4) & 0x3FFF) + (st_p1_1) * 2048;
                            int mma_b_next_lo_10 = (((smem_v2_addr_16) >> 4) & 0x3FFF) + (st_p1_1) * 2048;
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)mma_a_lo_10 | ((uint64_t)((uint32_t)mma_a_next_lo_10 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)mma_b_lo_10 | ((uint64_t)((uint32_t)mma_b_next_lo_10 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2((tmem_acc + (acc_base_1)), a_desc + 0, b_desc + 0,
                                        0x90400480U, tmem_tmem_sfa + 24 + 0, tmem_tmem_sfb + (48 + sfb_shift_1) + 0, 1);
                                }
                            }
                            mma_stage += 1;
                            if (mma_stage == 6) { mma_stage = 0; _phase_tma_full ^= 1; }
                            mbarrier_wait(sf_full_addr + (mma_sf_stage) * 8, _phase_sf_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            if (elect_sync()) {
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + 32, make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_sf_stage) * 384)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfa + 32 + 4), make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_sf_stage) * 384 + 32)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfa + 32 + 8), make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_sf_stage) * 384 + 64)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfa + 32 + 12), make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_sf_stage) * 384 + 96)));
                            }
                            if (elect_sync()) {
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + 64, make_sf_cp_desc_lo_sbo128((((smem_v25_addr) >> 4) + (mma_sf_stage) * 384)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 64 + 4), make_sf_cp_desc_lo_sbo128((((smem_v25_addr) >> 4) + (mma_sf_stage) * 384 + 32)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 64 + 8), make_sf_cp_desc_lo_sbo128((((smem_v25_addr) >> 4) + (mma_sf_stage) * 384 + 64)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 64 + 12), make_sf_cp_desc_lo_sbo128((((smem_v25_addr) >> 4) + (mma_sf_stage) * 384 + 96)));
                            }
                            if (elect_sync()) {
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + 80, make_sf_cp_desc_lo_sbo128((((smem_v26_addr) >> 4) + (mma_sf_stage) * 384)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 80 + 4), make_sf_cp_desc_lo_sbo128((((smem_v26_addr) >> 4) + (mma_sf_stage) * 384 + 32)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 80 + 8), make_sf_cp_desc_lo_sbo128((((smem_v26_addr) >> 4) + (mma_sf_stage) * 384 + 64)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 80 + 12), make_sf_cp_desc_lo_sbo128((((smem_v26_addr) >> 4) + (mma_sf_stage) * 384 + 96)));
                            }
                            elect_commit_cg2_multicast(sf_empty_addr + (mma_sf_stage) * 8, (uint16_t)(3));
                            mma_sf_stage += 1;
                            if (mma_sf_stage == 5) { mma_sf_stage = 0; _phase_sf_full ^= 1; }
                            mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            unsigned int st_p2_1 = mma_stage;
                            int mma_a_lo_11 = (((smem_v2_addr_4) >> 4) & 0x3FFF) + (st_p1_1) * 2048;
                            int mma_b_lo_11 = (((smem_v2_addr_13) >> 4) & 0x3FFF) + (st_p1_1) * 2048;
                            int mma_a_next_lo_11 = (((smem_v2_addr_7) >> 4) & 0x3FFF) + (st_p2_1) * 2048;
                            int mma_b_next_lo_11 = (((smem_v2_addr_16) >> 4) & 0x3FFF) + (st_p2_1) * 2048;
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)mma_a_lo_11 | ((uint64_t)((uint32_t)mma_a_next_lo_11 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)mma_b_lo_11 | ((uint64_t)((uint32_t)mma_b_next_lo_11 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2((tmem_acc + (acc_base_1)), a_desc + 0, b_desc + 0,
                                        0xd04004a0U, tmem_tmem_sfa + 28 + 0, tmem_tmem_sfb + (56 + sfb_shift_1) + 0, 1);
                                }
                            }
                            elect_commit_cg2_multicast(tma_empty_addr + (((mma_stage + 5 >= 6) ? mma_stage + 5 - 6 : mma_stage + 5)) * 8, (uint16_t)(3));
                            int mma_a_lo_12 = (((smem_v2_addr_5) >> 4) & 0x3FFF) + (st_p2_1) * 2048;
                            int mma_b_lo_12 = (((smem_v2_addr_14) >> 4) & 0x3FFF) + (st_p2_1) * 2048;
                            int mma_a_next_lo_12 = (((smem_v2_addr_7) >> 4) & 0x3FFF) + (st_p2_1) * 2048;
                            int mma_b_next_lo_12 = (((smem_v2_addr_16) >> 4) & 0x3FFF) + (st_p2_1) * 2048;
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)mma_a_lo_12 | ((uint64_t)((uint32_t)mma_a_next_lo_12 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)mma_b_lo_12 | ((uint64_t)((uint32_t)mma_b_next_lo_12 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2((tmem_acc + (acc_base_1)), a_desc + 0, b_desc + 0,
                                        0x90400480U, tmem_tmem_sfa + 36 + 0, tmem_tmem_sfb + (72 + sfb_shift_1) + 0, 1);
                                }
                            }
                            int mma_a_lo_13 = (((smem_v2_addr_6) >> 4) & 0x3FFF) + (st_p2_1) * 2048;
                            int mma_b_lo_13 = (((smem_v2_addr_15) >> 4) & 0x3FFF) + (st_p2_1) * 2048;
                            int mma_a_next_lo_13 = (((smem_v2_addr_7) >> 4) & 0x3FFF) + (st_p2_1) * 2048;
                            int mma_b_next_lo_13 = (((smem_v2_addr_16) >> 4) & 0x3FFF) + (st_p2_1) * 2048;
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)mma_a_lo_13 | ((uint64_t)((uint32_t)mma_a_next_lo_13 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)mma_b_lo_13 | ((uint64_t)((uint32_t)mma_b_next_lo_13 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2((tmem_acc + (acc_base_1)), a_desc + 0, b_desc + 0,
                                        0xd04004a0U, tmem_tmem_sfa + 40 + 0, tmem_tmem_sfb + (80 + sfb_shift_1) + 0, 1);
                                }
                            }
                            elect_commit_cg2_multicast(tma_empty_addr + (mma_stage) * 8, (uint16_t)(3));
                            mma_stage += 1;
                            if (mma_stage == 6) { mma_stage = 0; _phase_tma_full ^= 1; }
                        }
                        #pragma unroll 1
                        for (unsigned int k_slot_1 = 0; k_slot_1 < k96_tail_1; k_slot_1++) {
                            mbarrier_wait(sf_full_addr + (mma_sf_stage) * 8, _phase_sf_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            if (elect_sync()) {
                                tcgen05_cp_32x128b_warpx4_cta2((unsigned int)tmem_tmem_sfa + k_slot_1 * 16, make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_sf_stage) * 384)));
                                tcgen05_cp_32x128b_warpx4_cta2(((unsigned int)tmem_tmem_sfa + k_slot_1 * 16 + 4), make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_sf_stage) * 384 + 32)));
                                tcgen05_cp_32x128b_warpx4_cta2(((unsigned int)tmem_tmem_sfa + k_slot_1 * 16 + 8), make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_sf_stage) * 384 + 64)));
                                tcgen05_cp_32x128b_warpx4_cta2(((unsigned int)tmem_tmem_sfa + k_slot_1 * 16 + 12), make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_sf_stage) * 384 + 96)));
                            }
                            if (elect_sync()) {
                                tcgen05_cp_32x128b_warpx4_cta2((unsigned int)tmem_tmem_sfb + k_slot_1 * 2 * 16, make_sf_cp_desc_lo_sbo128((((smem_v25_addr) >> 4) + (mma_sf_stage) * 384)));
                                tcgen05_cp_32x128b_warpx4_cta2(((unsigned int)tmem_tmem_sfb + k_slot_1 * 2 * 16 + 4), make_sf_cp_desc_lo_sbo128((((smem_v25_addr) >> 4) + (mma_sf_stage) * 384 + 32)));
                                tcgen05_cp_32x128b_warpx4_cta2(((unsigned int)tmem_tmem_sfb + k_slot_1 * 2 * 16 + 8), make_sf_cp_desc_lo_sbo128((((smem_v25_addr) >> 4) + (mma_sf_stage) * 384 + 64)));
                                tcgen05_cp_32x128b_warpx4_cta2(((unsigned int)tmem_tmem_sfb + k_slot_1 * 2 * 16 + 12), make_sf_cp_desc_lo_sbo128((((smem_v25_addr) >> 4) + (mma_sf_stage) * 384 + 96)));
                            }
                            if (elect_sync()) {
                                tcgen05_cp_32x128b_warpx4_cta2((unsigned int)tmem_tmem_sfb + (k_slot_1 * 2 + 1) * 16, make_sf_cp_desc_lo_sbo128((((smem_v26_addr) >> 4) + (mma_sf_stage) * 384)));
                                tcgen05_cp_32x128b_warpx4_cta2(((unsigned int)tmem_tmem_sfb + (k_slot_1 * 2 + 1) * 16 + 4), make_sf_cp_desc_lo_sbo128((((smem_v26_addr) >> 4) + (mma_sf_stage) * 384 + 32)));
                                tcgen05_cp_32x128b_warpx4_cta2(((unsigned int)tmem_tmem_sfb + (k_slot_1 * 2 + 1) * 16 + 8), make_sf_cp_desc_lo_sbo128((((smem_v26_addr) >> 4) + (mma_sf_stage) * 384 + 64)));
                                tcgen05_cp_32x128b_warpx4_cta2(((unsigned int)tmem_tmem_sfb + (k_slot_1 * 2 + 1) * 16 + 12), make_sf_cp_desc_lo_sbo128((((smem_v26_addr) >> 4) + (mma_sf_stage) * 384 + 96)));
                            }
                            elect_commit_cg2_multicast(sf_empty_addr + (mma_sf_stage) * 8, (uint16_t)(3));
                            mma_sf_stage += 1;
                            if (mma_sf_stage == 5) { mma_sf_stage = 0; _phase_sf_full ^= 1; }
                            mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int tail_sfa_col_1 = (int)k_slot_1 * 16;
                            int tail_sfb_col_1 = (int)k_slot_1 * 32;
                            int init_tail_1 = (((int)k_slot_1 + k96_supers_1 == 0) ? 1 : 0);
                            int _mma_a_lo_4 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage) * 2048;
                            int _mma_b_lo_4 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_stage) * 2048;
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_4) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_4) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_cta2((tmem_acc + (acc_base_1)), a_desc + 0, b_desc + 0,
                                        0x10400480U, tmem_tmem_sfa + tail_sfa_col_1 + 0, tmem_tmem_sfb + (tail_sfb_col_1 + sfb_shift_1) + 0, ((init_tail_1) ? 0 : 1));
                                }
                            }
                            int _mma_a_lo_5 = (((smem_a_addr + 32) >> 4) & 0x3FFF) + (mma_stage) * 2048;
                            int _mma_b_lo_5 = (((smem_b_addr + 32) >> 4) & 0x3FFF) + (mma_stage) * 2048;
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_5) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_5) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_cta2((tmem_acc + (acc_base_1)), a_desc + 0, b_desc + 0,
                                        0x10400480U, tmem_tmem_sfa + (tail_sfa_col_1 + 4) + 0, tmem_tmem_sfb + (tail_sfb_col_1 + 8 + sfb_shift_1) + 0, 1);
                                }
                            }
                            int _mma_a_lo_6 = (((smem_a_addr + 64) >> 4) & 0x3FFF) + (mma_stage) * 2048;
                            int _mma_b_lo_6 = (((smem_b_addr + 64) >> 4) & 0x3FFF) + (mma_stage) * 2048;
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_6) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_6) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_cta2((tmem_acc + (acc_base_1)), a_desc + 0, b_desc + 0,
                                        0x10400480U, tmem_tmem_sfa + (tail_sfa_col_1 + 8) + 0, tmem_tmem_sfb + (tail_sfb_col_1 + 16 + sfb_shift_1) + 0, 1);
                                }
                            }
                            int _mma_a_lo_7 = (((smem_a_addr + 96) >> 4) & 0x3FFF) + (mma_stage) * 2048;
                            int _mma_b_lo_7 = (((smem_b_addr + 96) >> 4) & 0x3FFF) + (mma_stage) * 2048;
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_7) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_7) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_cta2((tmem_acc + (acc_base_1)), a_desc + 0, b_desc + 0,
                                        0x10400480U, tmem_tmem_sfa + (tail_sfa_col_1 + 12) + 0, tmem_tmem_sfb + (tail_sfb_col_1 + 24 + sfb_shift_1) + 0, 1);
                                }
                            }
                            elect_commit_cg2_multicast(tma_empty_addr + (mma_stage) * 8, (uint16_t)(3));
                            mma_stage += 1;
                            if (mma_stage == 6) { mma_stage = 0; _phase_tma_full ^= 1; }
                        }
                        elect_commit_cg2_multicast(acc_full_addr, (uint16_t)(3));
                    }
                    _phase_acc_empty ^= 1;
                    completed += 1;
                }
                mbarrier_wait(clc_full_addr + (mma_clc_stage) * 8, _phase_clc_full_1);
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
                    : "r"(clc_resp_addr + mma_clc_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_1 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(_clc_ctaid_1)
                    : "r"(clc_resp_addr + mma_clc_stage * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(clc_empty_addr + (mma_clc_stage) * 8), "r"(0) : "memory");
                mma_clc_stage += 1;
                if (mma_clc_stage == 2) { mma_clc_stage = 0; _phase_clc_full_1 ^= 1; }
                if (_clc_valid_2 == 0) {
                    break;
                }
                tile_m = (2 * (((_clc_ctaid_1 + mma_rank) / 2 >> 2) / (unsigned int)(tok_tiles / 2)) + ((_clc_ctaid_1 + mma_rank) / 2 >> 1 & 1)) * (unsigned int)tok_tiles + (2 * (((_clc_ctaid_1 + mma_rank) / 2 >> 2) % (unsigned int)(tok_tiles / 2)) + ((_clc_ctaid_1 + mma_rank) / 2 & 1));
            }
            if ((cta_rank & 1) == 0) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
                if (completed > 0) {
                    int _parity_acc_empty = (int)((unsigned int)(int)completed % 2) ^ 1;
                    mbarrier_wait(acc_empty_addr, _parity_acc_empty);
                }
            }
            int dealloc_peer_rank = cta_rank ^ 1;
            if ((cta_rank & 1) != 0) {
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(tmem_dealloc_bar_addr), "r"(dealloc_peer_rank) : "memory");
            }
            mbarrier_wait(tmem_dealloc_bar_addr, 0);
            if ((cta_rank & 1) == 0) {
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(tmem_dealloc_bar_addr), "r"(dealloc_peer_rank) : "memory");
            }
            asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
        }
    }
    // ---- Role: sf_load ----
    if (warp == 2) {
        { // sf_load_main
            unsigned int sf_stage = 0;
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int sf_clc_stage = 0;
            unsigned int sf_rank = cta_rank;
            int sf_cs = (int)cluster_size;
            unsigned int tile_s = (2 * ((cluster_id >> 2) / (unsigned int)(tok_tiles / 2)) + (cluster_id >> 1 & 1)) * (unsigned int)tok_tiles + (2 * ((cluster_id >> 2) % (unsigned int)(tok_tiles / 2)) + (cluster_id & 1));
            unsigned int _phase_sf_empty = 1;
            unsigned int _phase_clc_full_2 = 0;
            #pragma unroll 1
            for (unsigned int _p_2 = 0; _p_2 < num_tiles; _p_2++) {
                if (sf_cs == 8) {
                    int w_idx_2 = tile_s / (unsigned int)tok_tiles;
                    int tok_idx_2 = tile_s - (unsigned int)(w_idx_2 * tok_tiles);
                    int a_atom = tok_idx_2 * 2 + (cta_rank & 1);
                    int b_atom = w_idx_2 * 2;
                    unsigned int a_set = (unsigned int)((cta_rank >> 2) * 2);
                    unsigned int b_set = (unsigned int)(cta_rank & 3);
                    unsigned int a_half_1 = (unsigned int)((cta_rank >> 2) * 1024);
                    unsigned int b_part = (unsigned int)((cta_rank & 3) * 1024);
                    int a_mask_1 = 17 << (cta_rank & 3);
                    int b_mask_1 = 15 << (cta_rank & 4);
                    #pragma unroll 1
                    for (unsigned int k_tile_2 = 0; k_tile_2 < K_tiles; k_tile_2++) {
                        mbarrier_wait(sf_empty_addr + (sf_stage) * 8, _phase_sf_empty);
                        if ((cta_rank & 1) == 0) {
                            if (elect_sync()) {
                                asm volatile(
                                    "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                    :: "r"((sf_full_addr + (sf_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(12288)) : "memory");
                            }
                        }
                        if (elect_sync()) {
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6, %7;"
                                :: "r"(smem_sfa_addr + sf_stage * 6144 + a_half_1), "l"((&SFA_h)), "r"(0), "r"(4 * k_tile_2 + a_set), "r"(a_atom),
                                   "r"(((sf_full_addr + (sf_stage) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(a_mask_1)), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6, %7;"
                                :: "r"(smem_v21_addr + sf_stage * 6144 + b_part), "l"((&SFB)), "r"(0), "r"(4 * k_tile_2 + b_set), "r"(b_atom),
                                   "r"(((sf_full_addr + (sf_stage) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(b_mask_1)), "l"(0x14F0000000000000ULL) : "memory");
                        }
                        if ((int)k_tile_2 + 5 < K_tiles) {
                            if (elect_sync()) {
                                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&SFA_h))), "r"((int)(0)), "r"((int)(4 * (k_tile_2 + 5) + a_set)), "r"((int)(a_atom)) : "memory");
                                asm volatile("cp.async.bulk.prefetch.tensor.3d.L2.global.tile [%0, {%1, %2, %3}];" :: "l"((uint64_t)((&SFB))), "r"((int)(0)), "r"((int)(4 * (k_tile_2 + 5) + b_set)), "r"((int)(b_atom)) : "memory");
                            }
                        }
                        sf_stage += 1;
                        if (sf_stage == 5) { sf_stage = 0; _phase_sf_empty ^= 1; }
                    }
                } else {
                    int w_idx_3 = tile_s / (unsigned int)tok_tiles;
                    int tok_idx_3 = tile_s - (unsigned int)(w_idx_3 * tok_tiles);
                    int a_rows_2 = tok_idx_3 * 256 + cta_rank * 128;
                    int a_atom_1 = a_rows_2 / 128;
                    int b_atom_1 = w_idx_3 * 256 / 128;
                    #pragma unroll 1
                    for (unsigned int k_tile_3 = 0; k_tile_3 < K_tiles; k_tile_3++) {
                        mbarrier_wait(sf_empty_addr + (sf_stage) * 8, _phase_sf_empty);
                        if (cta_rank == 0) {
                            if (elect_sync()) {
                                asm volatile(
                                    "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                    :: "r"((sf_full_addr + (sf_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(12288)) : "memory");
                            }
                        }
                        if (elect_sync()) {
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(smem_sfa_addr + sf_stage * 6144), "l"((&SFA)), "r"(0), "r"(4 * k_tile_3), "r"(a_atom_1),
                                   "r"(((sf_full_addr + (sf_stage) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        }
                        if (cta_rank == 0) {
                            if (elect_sync()) {
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6, %7;"
                                    :: "r"(smem_v21_addr + sf_stage * 6144), "l"((&SFB)), "r"(0), "r"(4 * k_tile_3), "r"(b_atom_1),
                                       "r"(((sf_full_addr + (sf_stage) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)), "l"(0x14F0000000000000ULL) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6, %7;"
                                    :: "r"(smem_v22_addr + sf_stage * 6144), "l"((&SFB)), "r"(0), "r"(4 * k_tile_3 + 1), "r"(b_atom_1),
                                       "r"(((sf_full_addr + (sf_stage) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)), "l"(0x14F0000000000000ULL) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6, %7;"
                                    :: "r"(smem_v23_addr + sf_stage * 6144), "l"((&SFB)), "r"(0), "r"(4 * k_tile_3 + 2), "r"(b_atom_1),
                                       "r"(((sf_full_addr + (sf_stage) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)), "l"(0x14F0000000000000ULL) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6, %7;"
                                    :: "r"(smem_v24_addr + sf_stage * 6144), "l"((&SFB)), "r"(0), "r"(4 * k_tile_3 + 3), "r"(b_atom_1),
                                       "r"(((sf_full_addr + (sf_stage) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)), "l"(0x14F0000000000000ULL) : "memory");
                            }
                        }
                        sf_stage += 1;
                        if (sf_stage == 5) { sf_stage = 0; _phase_sf_empty ^= 1; }
                    }
                }
                mbarrier_wait(clc_full_addr + (sf_clc_stage) * 8, _phase_clc_full_2);
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
                    : "r"(clc_resp_addr + sf_clc_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_3 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(_clc_ctaid_3)
                    : "r"(clc_resp_addr + sf_clc_stage * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(clc_empty_addr + (sf_clc_stage) * 8), "r"(0) : "memory");
                sf_clc_stage += 1;
                if (sf_clc_stage == 2) { sf_clc_stage = 0; _phase_clc_full_2 ^= 1; }
                if (_clc_valid_4 == 0) {
                    break;
                }
                tile_s = (2 * (((_clc_ctaid_3 + sf_rank) / 2 >> 2) / (unsigned int)(tok_tiles / 2)) + ((_clc_ctaid_3 + sf_rank) / 2 >> 1 & 1)) * (unsigned int)tok_tiles + (2 * (((_clc_ctaid_3 + sf_rank) / 2 >> 2) % (unsigned int)(tok_tiles / 2)) + ((_clc_ctaid_3 + sf_rank) / 2 & 1));
            }
            #pragma unroll 1
            for (int __1 = 0; __1 < 5; __1++) {
                mbarrier_wait(sf_empty_addr + (sf_stage) * 8, _phase_sf_empty);
                sf_stage += 1;
                if (sf_stage == 5) { sf_stage = 0; _phase_sf_empty ^= 1; }
            }
        }
    }
    // ---- Role: sched ----
    if (warp == 3) {
        { // sched_main
            unsigned int _phase_thr_full = 0;
            unsigned int _phase_clc_empty = 1;
            unsigned int _phase_clc_full_3 = 0;
            if (cta_rank == 0) {
                unsigned int sched_clc_stage = 0;
                unsigned int sched_thr_stage = 0;
                int sched_ctas = (int)cluster_size;
                #pragma unroll 1
                for (unsigned int _p_3 = 0; _p_3 < num_tiles; _p_3++) {
                    mbarrier_wait(thr_full_addr + (sched_thr_stage) * 8, _phase_thr_full);
                    mbarrier_arrive(thr_empty_addr + (sched_thr_stage) * 8);
                    sched_thr_stage += 1;
                    if (sched_thr_stage == 2) { sched_thr_stage = 0; _phase_thr_full ^= 1; }
                    mbarrier_wait(clc_empty_addr + (sched_clc_stage) * 8, _phase_clc_empty);
                    if (lane < sched_ctas) {
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                            "}"
                            :: "r"(clc_full_addr + (sched_clc_stage) * 8), "r"(lane), "r"((uint32_t)(16)) : "memory");
                    }
                    if (elect_sync()) {
                        asm volatile(
                            "fence.proxy.async.shared::cta;\n\t"
                            "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                                " [%0], [%1];"
                            :: "r"(clc_resp_addr + sched_clc_stage * 16 + 0 * 16), "r"(clc_full_addr + sched_clc_stage * 8)
                            : "memory");
                    }
                    mbarrier_wait(clc_full_addr + (sched_clc_stage) * 8, _phase_clc_full_3);
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
                        : "r"(clc_resp_addr + sched_clc_stage * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(clc_empty_addr + (sched_clc_stage) * 8), "r"(0) : "memory");
                    sched_clc_stage += 1;
                    if (sched_clc_stage == 2) { sched_clc_stage = 0; _phase_clc_empty ^= 1; _phase_clc_full_3 ^= 1; }
                    if (_clc_valid_0 == 0) {
                        break;
                    }
                }
                #pragma unroll
                for (int __2 = 0; __2 < 2; __2++) {
                    mbarrier_wait(clc_empty_addr + (sched_clc_stage) * 8, _phase_clc_empty);
                    sched_clc_stage += 1;
                    if (sched_clc_stage == 2) { sched_clc_stage = 0; _phase_clc_empty ^= 1; _phase_clc_full_3 ^= 1; }
                }
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 4 && warp <= 7) {
        { // epilogue_main
            const int epi_warp = warp - 4;
            int epi_row = epi_warp % 4 * 32 + lane;
            int stg_vec = (int)(out_stride_m2 % 16 == 0 && (unsigned long long)out % 32 == 0);
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int epi_clc_stage = 0;
            unsigned int epi_rank = cta_rank;
            int epi_cs = (int)cluster_size;
            unsigned int tile_e = (2 * ((cluster_id >> 2) / (unsigned int)(tok_tiles / 2)) + (cluster_id >> 1 & 1)) * (unsigned int)tok_tiles + (2 * ((cluster_id >> 2) % (unsigned int)(tok_tiles / 2)) + (cluster_id & 1));
            unsigned int _phase_clc_full_4 = 0;
            unsigned int _phase_acc_full = 0;
            #pragma unroll 1
            for (unsigned int _p_4 = 0; _p_4 < num_tiles; _p_4++) {
                mbarrier_wait(clc_full_addr + (epi_clc_stage) * 8, _phase_clc_full_4);
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
                    : "r"(clc_resp_addr + epi_clc_stage * 16 + 0 * 16)
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
                    : "r"(clc_resp_addr + epi_clc_stage * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(clc_empty_addr + (epi_clc_stage) * 8), "r"(0) : "memory");
                epi_clc_stage += 1;
                if (epi_clc_stage == 2) { epi_clc_stage = 0; _phase_clc_full_4 ^= 1; }
                if (epi_cs == 8) {
                    int w_idx_4 = tile_e / (unsigned int)tok_tiles;
                    int tok_idx_4 = tile_e - (unsigned int)(w_idx_4 * tok_tiles);
                    int off_tok = tok_idx_4 * 256 + (cta_rank & 1) * 128;
                    int off_w = w_idx_4 * 256;
                    int acc_base_2 = (int)0 * 256;
                    int lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)acc_base_2;
                    int tok_row = ((off_tok + epi_row) < (M - 1) ? (off_tok + epi_row) : (M - 1));
                    float alpha_row = alpha[tok_row];
                    int stg_tok = off_tok + epi_row;
                    mbarrier_wait(acc_full_addr, _phase_acc_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    #pragma unroll
                    for (int subtile = 0; subtile < 8; subtile++) {
                        int tmem_addr = lane_addr + subtile * 32;
                        float _tmem_load_0[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                            : "r"(tmem_addr));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        if (subtile == 7) {
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                :: "r"((acc_empty_addr) & 0xFEFFFFFF) : "memory");
                        }
                        {
                            float2 _pair_scale_even2_0 = make_float2(alpha_row, alpha_row);
                            float2 _pair_scale_odd2_0 = make_float2(alpha_row, alpha_row);
                            float2* _pair_scale_src2_0 = reinterpret_cast<float2*>(&_tmem_load_0[0]);
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[0]) : "l"(*(unsigned long long*)&_pair_scale_even2_0));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[1]) : "l"(*(unsigned long long*)&_pair_scale_odd2_0));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[2]) : "l"(*(unsigned long long*)&_pair_scale_even2_0));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[3]) : "l"(*(unsigned long long*)&_pair_scale_odd2_0));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[4]) : "l"(*(unsigned long long*)&_pair_scale_even2_0));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[5]) : "l"(*(unsigned long long*)&_pair_scale_odd2_0));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[6]) : "l"(*(unsigned long long*)&_pair_scale_even2_0));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[7]) : "l"(*(unsigned long long*)&_pair_scale_odd2_0));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[8]) : "l"(*(unsigned long long*)&_pair_scale_even2_0));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[9]) : "l"(*(unsigned long long*)&_pair_scale_odd2_0));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[10]) : "l"(*(unsigned long long*)&_pair_scale_even2_0));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[11]) : "l"(*(unsigned long long*)&_pair_scale_odd2_0));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[12]) : "l"(*(unsigned long long*)&_pair_scale_even2_0));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[13]) : "l"(*(unsigned long long*)&_pair_scale_odd2_0));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[14]) : "l"(*(unsigned long long*)&_pair_scale_even2_0));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_0[15]) : "l"(*(unsigned long long*)&_pair_scale_odd2_0));
                        }
                        unsigned int tmem_load_0_bf16[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __half2 _h2 = __float22half2_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                            tmem_load_0_bf16[_lp] = *(uint32_t*)&_h2;
                        }
                        int stg_col = off_w + subtile * 32;
                        if (stg_tok < M) {
                            if (stg_vec != 0 && stg_col + 32 <= N) {
                                asm volatile("st.global.L1::no_allocate.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(reinterpret_cast<unsigned int*>(out + ((long long)stg_tok * out_stride_m2 + (long long)(stg_col + 0)))), "r"(tmem_load_0_bf16[(0) + 0]), "r"(tmem_load_0_bf16[(0) + 1]), "r"(tmem_load_0_bf16[(0) + 2]), "r"(tmem_load_0_bf16[(0) + 3]), "r"(tmem_load_0_bf16[(0) + 4]), "r"(tmem_load_0_bf16[(0) + 5]), "r"(tmem_load_0_bf16[(0) + 6]), "r"(tmem_load_0_bf16[(0) + 7]) : "memory");
                                asm volatile("st.global.L1::no_allocate.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(reinterpret_cast<unsigned int*>(out + ((long long)stg_tok * out_stride_m2 + (long long)(stg_col + 16)))), "r"(tmem_load_0_bf16[(8) + 0]), "r"(tmem_load_0_bf16[(8) + 1]), "r"(tmem_load_0_bf16[(8) + 2]), "r"(tmem_load_0_bf16[(8) + 3]), "r"(tmem_load_0_bf16[(8) + 4]), "r"(tmem_load_0_bf16[(8) + 5]), "r"(tmem_load_0_bf16[(8) + 6]), "r"(tmem_load_0_bf16[(8) + 7]) : "memory");
                            } else {
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok * out_stride_m2 + (long long)(stg_col + 0)))), "h"((unsigned short)(tmem_load_0_bf16[0])), "r"((unsigned int)(stg_col + 0 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok * out_stride_m2 + (long long)(stg_col + 1)))), "h"((unsigned short)(tmem_load_0_bf16[0] >> 16)), "r"((unsigned int)(stg_col + 1 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok * out_stride_m2 + (long long)(stg_col + 2)))), "h"((unsigned short)(tmem_load_0_bf16[1])), "r"((unsigned int)(stg_col + 2 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok * out_stride_m2 + (long long)(stg_col + 3)))), "h"((unsigned short)(tmem_load_0_bf16[1] >> 16)), "r"((unsigned int)(stg_col + 3 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok * out_stride_m2 + (long long)(stg_col + 4)))), "h"((unsigned short)(tmem_load_0_bf16[2])), "r"((unsigned int)(stg_col + 4 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok * out_stride_m2 + (long long)(stg_col + 5)))), "h"((unsigned short)(tmem_load_0_bf16[2] >> 16)), "r"((unsigned int)(stg_col + 5 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok * out_stride_m2 + (long long)(stg_col + 6)))), "h"((unsigned short)(tmem_load_0_bf16[3])), "r"((unsigned int)(stg_col + 6 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok * out_stride_m2 + (long long)(stg_col + 7)))), "h"((unsigned short)(tmem_load_0_bf16[3] >> 16)), "r"((unsigned int)(stg_col + 7 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok * out_stride_m2 + (long long)(stg_col + 8)))), "h"((unsigned short)(tmem_load_0_bf16[4])), "r"((unsigned int)(stg_col + 8 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok * out_stride_m2 + (long long)(stg_col + 9)))), "h"((unsigned short)(tmem_load_0_bf16[4] >> 16)), "r"((unsigned int)(stg_col + 9 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok * out_stride_m2 + (long long)(stg_col + 10)))), "h"((unsigned short)(tmem_load_0_bf16[5])), "r"((unsigned int)(stg_col + 10 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok * out_stride_m2 + (long long)(stg_col + 11)))), "h"((unsigned short)(tmem_load_0_bf16[5] >> 16)), "r"((unsigned int)(stg_col + 11 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok * out_stride_m2 + (long long)(stg_col + 12)))), "h"((unsigned short)(tmem_load_0_bf16[6])), "r"((unsigned int)(stg_col + 12 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok * out_stride_m2 + (long long)(stg_col + 13)))), "h"((unsigned short)(tmem_load_0_bf16[6] >> 16)), "r"((unsigned int)(stg_col + 13 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok * out_stride_m2 + (long long)(stg_col + 14)))), "h"((unsigned short)(tmem_load_0_bf16[7])), "r"((unsigned int)(stg_col + 14 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok * out_stride_m2 + (long long)(stg_col + 15)))), "h"((unsigned short)(tmem_load_0_bf16[7] >> 16)), "r"((unsigned int)(stg_col + 15 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok * out_stride_m2 + (long long)(stg_col + 16)))), "h"((unsigned short)(tmem_load_0_bf16[8])), "r"((unsigned int)(stg_col + 16 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok * out_stride_m2 + (long long)(stg_col + 17)))), "h"((unsigned short)(tmem_load_0_bf16[8] >> 16)), "r"((unsigned int)(stg_col + 17 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok * out_stride_m2 + (long long)(stg_col + 18)))), "h"((unsigned short)(tmem_load_0_bf16[9])), "r"((unsigned int)(stg_col + 18 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok * out_stride_m2 + (long long)(stg_col + 19)))), "h"((unsigned short)(tmem_load_0_bf16[9] >> 16)), "r"((unsigned int)(stg_col + 19 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok * out_stride_m2 + (long long)(stg_col + 20)))), "h"((unsigned short)(tmem_load_0_bf16[10])), "r"((unsigned int)(stg_col + 20 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok * out_stride_m2 + (long long)(stg_col + 21)))), "h"((unsigned short)(tmem_load_0_bf16[10] >> 16)), "r"((unsigned int)(stg_col + 21 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok * out_stride_m2 + (long long)(stg_col + 22)))), "h"((unsigned short)(tmem_load_0_bf16[11])), "r"((unsigned int)(stg_col + 22 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok * out_stride_m2 + (long long)(stg_col + 23)))), "h"((unsigned short)(tmem_load_0_bf16[11] >> 16)), "r"((unsigned int)(stg_col + 23 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok * out_stride_m2 + (long long)(stg_col + 24)))), "h"((unsigned short)(tmem_load_0_bf16[12])), "r"((unsigned int)(stg_col + 24 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok * out_stride_m2 + (long long)(stg_col + 25)))), "h"((unsigned short)(tmem_load_0_bf16[12] >> 16)), "r"((unsigned int)(stg_col + 25 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok * out_stride_m2 + (long long)(stg_col + 26)))), "h"((unsigned short)(tmem_load_0_bf16[13])), "r"((unsigned int)(stg_col + 26 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok * out_stride_m2 + (long long)(stg_col + 27)))), "h"((unsigned short)(tmem_load_0_bf16[13] >> 16)), "r"((unsigned int)(stg_col + 27 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok * out_stride_m2 + (long long)(stg_col + 28)))), "h"((unsigned short)(tmem_load_0_bf16[14])), "r"((unsigned int)(stg_col + 28 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok * out_stride_m2 + (long long)(stg_col + 29)))), "h"((unsigned short)(tmem_load_0_bf16[14] >> 16)), "r"((unsigned int)(stg_col + 29 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok * out_stride_m2 + (long long)(stg_col + 30)))), "h"((unsigned short)(tmem_load_0_bf16[15])), "r"((unsigned int)(stg_col + 30 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok * out_stride_m2 + (long long)(stg_col + 31)))), "h"((unsigned short)(tmem_load_0_bf16[15] >> 16)), "r"((unsigned int)(stg_col + 31 < N)) : "memory");
                            }
                        }
                    }
                } else {
                    int w_idx_5 = tile_e / (unsigned int)tok_tiles;
                    int tok_idx_5 = tile_e - (unsigned int)(w_idx_5 * tok_tiles);
                    int off_tok_1 = tok_idx_5 * 256 + cta_rank * 128;
                    int off_w_1 = w_idx_5 * 256;
                    int acc_base_3 = (int)0 * 256;
                    int lane_addr_1 = taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)acc_base_3;
                    int tok_row_1 = ((off_tok_1 + epi_row) < (M - 1) ? (off_tok_1 + epi_row) : (M - 1));
                    float alpha_row_1 = alpha[tok_row_1];
                    int stg_tok_1 = off_tok_1 + epi_row;
                    mbarrier_wait(acc_full_addr, _phase_acc_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    #pragma unroll
                    for (int subtile_1 = 0; subtile_1 < 8; subtile_1++) {
                        int tmem_addr_1 = lane_addr_1 + subtile_1 * 32;
                        float _tmem_load_0_1[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_0_1[0]), "=f"(_tmem_load_0_1[1]), "=f"(_tmem_load_0_1[2]), "=f"(_tmem_load_0_1[3]), "=f"(_tmem_load_0_1[4]), "=f"(_tmem_load_0_1[5]), "=f"(_tmem_load_0_1[6]), "=f"(_tmem_load_0_1[7]), "=f"(_tmem_load_0_1[8]), "=f"(_tmem_load_0_1[9]), "=f"(_tmem_load_0_1[10]), "=f"(_tmem_load_0_1[11]), "=f"(_tmem_load_0_1[12]), "=f"(_tmem_load_0_1[13]), "=f"(_tmem_load_0_1[14]), "=f"(_tmem_load_0_1[15]), "=f"(_tmem_load_0_1[16]), "=f"(_tmem_load_0_1[17]), "=f"(_tmem_load_0_1[18]), "=f"(_tmem_load_0_1[19]), "=f"(_tmem_load_0_1[20]), "=f"(_tmem_load_0_1[21]), "=f"(_tmem_load_0_1[22]), "=f"(_tmem_load_0_1[23]), "=f"(_tmem_load_0_1[24]), "=f"(_tmem_load_0_1[25]), "=f"(_tmem_load_0_1[26]), "=f"(_tmem_load_0_1[27]), "=f"(_tmem_load_0_1[28]), "=f"(_tmem_load_0_1[29]), "=f"(_tmem_load_0_1[30]), "=f"(_tmem_load_0_1[31])
                            : "r"(tmem_addr_1));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        if (subtile_1 == 7) {
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                :: "r"((acc_empty_addr) & 0xFEFFFFFF) : "memory");
                        }
                        {
                            float2 _pair_scale_even2_1 = make_float2(alpha_row_1, alpha_row_1);
                            float2 _pair_scale_odd2_1 = make_float2(alpha_row_1, alpha_row_1);
                            float2* _pair_scale_src2_1 = reinterpret_cast<float2*>(&_tmem_load_0_1[0]);
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_1[0]) : "l"(*(unsigned long long*)&_pair_scale_even2_1));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_1[1]) : "l"(*(unsigned long long*)&_pair_scale_odd2_1));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_1[2]) : "l"(*(unsigned long long*)&_pair_scale_even2_1));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_1[3]) : "l"(*(unsigned long long*)&_pair_scale_odd2_1));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_1[4]) : "l"(*(unsigned long long*)&_pair_scale_even2_1));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_1[5]) : "l"(*(unsigned long long*)&_pair_scale_odd2_1));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_1[6]) : "l"(*(unsigned long long*)&_pair_scale_even2_1));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_1[7]) : "l"(*(unsigned long long*)&_pair_scale_odd2_1));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_1[8]) : "l"(*(unsigned long long*)&_pair_scale_even2_1));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_1[9]) : "l"(*(unsigned long long*)&_pair_scale_odd2_1));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_1[10]) : "l"(*(unsigned long long*)&_pair_scale_even2_1));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_1[11]) : "l"(*(unsigned long long*)&_pair_scale_odd2_1));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_1[12]) : "l"(*(unsigned long long*)&_pair_scale_even2_1));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_1[13]) : "l"(*(unsigned long long*)&_pair_scale_odd2_1));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_1[14]) : "l"(*(unsigned long long*)&_pair_scale_even2_1));
                            asm("mul.rn.f32x2 %0, %0, %1;" : "+l"(*(unsigned long long*)&_pair_scale_src2_1[15]) : "l"(*(unsigned long long*)&_pair_scale_odd2_1));
                        }
                        unsigned int tmem_load_0_bf16_1[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __half2 _h2 = __float22half2_rn(make_float2(_tmem_load_0_1[_lp*2 + 0], _tmem_load_0_1[_lp*2+1 + 0]));
                            tmem_load_0_bf16_1[_lp] = *(uint32_t*)&_h2;
                        }
                        int stg_col_1 = off_w_1 + subtile_1 * 32;
                        if (stg_tok_1 < M) {
                            if (stg_vec != 0 && stg_col_1 + 32 <= N) {
                                asm volatile("st.global.L1::no_allocate.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(reinterpret_cast<unsigned int*>(out + ((long long)stg_tok_1 * out_stride_m2 + (long long)(stg_col_1 + 0)))), "r"(tmem_load_0_bf16_1[(0) + 0]), "r"(tmem_load_0_bf16_1[(0) + 1]), "r"(tmem_load_0_bf16_1[(0) + 2]), "r"(tmem_load_0_bf16_1[(0) + 3]), "r"(tmem_load_0_bf16_1[(0) + 4]), "r"(tmem_load_0_bf16_1[(0) + 5]), "r"(tmem_load_0_bf16_1[(0) + 6]), "r"(tmem_load_0_bf16_1[(0) + 7]) : "memory");
                                asm volatile("st.global.L1::no_allocate.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};" :: "l"(reinterpret_cast<unsigned int*>(out + ((long long)stg_tok_1 * out_stride_m2 + (long long)(stg_col_1 + 16)))), "r"(tmem_load_0_bf16_1[(8) + 0]), "r"(tmem_load_0_bf16_1[(8) + 1]), "r"(tmem_load_0_bf16_1[(8) + 2]), "r"(tmem_load_0_bf16_1[(8) + 3]), "r"(tmem_load_0_bf16_1[(8) + 4]), "r"(tmem_load_0_bf16_1[(8) + 5]), "r"(tmem_load_0_bf16_1[(8) + 6]), "r"(tmem_load_0_bf16_1[(8) + 7]) : "memory");
                            } else {
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok_1 * out_stride_m2 + (long long)(stg_col_1 + 0)))), "h"((unsigned short)(tmem_load_0_bf16_1[0])), "r"((unsigned int)(stg_col_1 + 0 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok_1 * out_stride_m2 + (long long)(stg_col_1 + 1)))), "h"((unsigned short)(tmem_load_0_bf16_1[0] >> 16)), "r"((unsigned int)(stg_col_1 + 1 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok_1 * out_stride_m2 + (long long)(stg_col_1 + 2)))), "h"((unsigned short)(tmem_load_0_bf16_1[1])), "r"((unsigned int)(stg_col_1 + 2 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok_1 * out_stride_m2 + (long long)(stg_col_1 + 3)))), "h"((unsigned short)(tmem_load_0_bf16_1[1] >> 16)), "r"((unsigned int)(stg_col_1 + 3 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok_1 * out_stride_m2 + (long long)(stg_col_1 + 4)))), "h"((unsigned short)(tmem_load_0_bf16_1[2])), "r"((unsigned int)(stg_col_1 + 4 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok_1 * out_stride_m2 + (long long)(stg_col_1 + 5)))), "h"((unsigned short)(tmem_load_0_bf16_1[2] >> 16)), "r"((unsigned int)(stg_col_1 + 5 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok_1 * out_stride_m2 + (long long)(stg_col_1 + 6)))), "h"((unsigned short)(tmem_load_0_bf16_1[3])), "r"((unsigned int)(stg_col_1 + 6 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok_1 * out_stride_m2 + (long long)(stg_col_1 + 7)))), "h"((unsigned short)(tmem_load_0_bf16_1[3] >> 16)), "r"((unsigned int)(stg_col_1 + 7 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok_1 * out_stride_m2 + (long long)(stg_col_1 + 8)))), "h"((unsigned short)(tmem_load_0_bf16_1[4])), "r"((unsigned int)(stg_col_1 + 8 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok_1 * out_stride_m2 + (long long)(stg_col_1 + 9)))), "h"((unsigned short)(tmem_load_0_bf16_1[4] >> 16)), "r"((unsigned int)(stg_col_1 + 9 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok_1 * out_stride_m2 + (long long)(stg_col_1 + 10)))), "h"((unsigned short)(tmem_load_0_bf16_1[5])), "r"((unsigned int)(stg_col_1 + 10 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok_1 * out_stride_m2 + (long long)(stg_col_1 + 11)))), "h"((unsigned short)(tmem_load_0_bf16_1[5] >> 16)), "r"((unsigned int)(stg_col_1 + 11 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok_1 * out_stride_m2 + (long long)(stg_col_1 + 12)))), "h"((unsigned short)(tmem_load_0_bf16_1[6])), "r"((unsigned int)(stg_col_1 + 12 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok_1 * out_stride_m2 + (long long)(stg_col_1 + 13)))), "h"((unsigned short)(tmem_load_0_bf16_1[6] >> 16)), "r"((unsigned int)(stg_col_1 + 13 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok_1 * out_stride_m2 + (long long)(stg_col_1 + 14)))), "h"((unsigned short)(tmem_load_0_bf16_1[7])), "r"((unsigned int)(stg_col_1 + 14 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok_1 * out_stride_m2 + (long long)(stg_col_1 + 15)))), "h"((unsigned short)(tmem_load_0_bf16_1[7] >> 16)), "r"((unsigned int)(stg_col_1 + 15 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok_1 * out_stride_m2 + (long long)(stg_col_1 + 16)))), "h"((unsigned short)(tmem_load_0_bf16_1[8])), "r"((unsigned int)(stg_col_1 + 16 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok_1 * out_stride_m2 + (long long)(stg_col_1 + 17)))), "h"((unsigned short)(tmem_load_0_bf16_1[8] >> 16)), "r"((unsigned int)(stg_col_1 + 17 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok_1 * out_stride_m2 + (long long)(stg_col_1 + 18)))), "h"((unsigned short)(tmem_load_0_bf16_1[9])), "r"((unsigned int)(stg_col_1 + 18 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok_1 * out_stride_m2 + (long long)(stg_col_1 + 19)))), "h"((unsigned short)(tmem_load_0_bf16_1[9] >> 16)), "r"((unsigned int)(stg_col_1 + 19 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok_1 * out_stride_m2 + (long long)(stg_col_1 + 20)))), "h"((unsigned short)(tmem_load_0_bf16_1[10])), "r"((unsigned int)(stg_col_1 + 20 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok_1 * out_stride_m2 + (long long)(stg_col_1 + 21)))), "h"((unsigned short)(tmem_load_0_bf16_1[10] >> 16)), "r"((unsigned int)(stg_col_1 + 21 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok_1 * out_stride_m2 + (long long)(stg_col_1 + 22)))), "h"((unsigned short)(tmem_load_0_bf16_1[11])), "r"((unsigned int)(stg_col_1 + 22 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok_1 * out_stride_m2 + (long long)(stg_col_1 + 23)))), "h"((unsigned short)(tmem_load_0_bf16_1[11] >> 16)), "r"((unsigned int)(stg_col_1 + 23 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok_1 * out_stride_m2 + (long long)(stg_col_1 + 24)))), "h"((unsigned short)(tmem_load_0_bf16_1[12])), "r"((unsigned int)(stg_col_1 + 24 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok_1 * out_stride_m2 + (long long)(stg_col_1 + 25)))), "h"((unsigned short)(tmem_load_0_bf16_1[12] >> 16)), "r"((unsigned int)(stg_col_1 + 25 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok_1 * out_stride_m2 + (long long)(stg_col_1 + 26)))), "h"((unsigned short)(tmem_load_0_bf16_1[13])), "r"((unsigned int)(stg_col_1 + 26 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok_1 * out_stride_m2 + (long long)(stg_col_1 + 27)))), "h"((unsigned short)(tmem_load_0_bf16_1[13] >> 16)), "r"((unsigned int)(stg_col_1 + 27 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok_1 * out_stride_m2 + (long long)(stg_col_1 + 28)))), "h"((unsigned short)(tmem_load_0_bf16_1[14])), "r"((unsigned int)(stg_col_1 + 28 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok_1 * out_stride_m2 + (long long)(stg_col_1 + 29)))), "h"((unsigned short)(tmem_load_0_bf16_1[14] >> 16)), "r"((unsigned int)(stg_col_1 + 29 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok_1 * out_stride_m2 + (long long)(stg_col_1 + 30)))), "h"((unsigned short)(tmem_load_0_bf16_1[15])), "r"((unsigned int)(stg_col_1 + 30 < N)) : "memory");
                                asm volatile("{ .reg .pred p_; setp.ne.b32 p_, %2, 0; @p_ st.global.b16 [%0], %1; }" :: "l"(reinterpret_cast<uint64_t>(out + ((long long)stg_tok_1 * out_stride_m2 + (long long)(stg_col_1 + 31)))), "h"((unsigned short)(tmem_load_0_bf16_1[15] >> 16)), "r"((unsigned int)(stg_col_1 + 31 < N)) : "memory");
                            }
                        }
                    }
                }
                _phase_acc_full ^= 1;
                if (_clc_valid_3 == 0) {
                    break;
                }
                tile_e = (2 * (((_clc_ctaid_2 + epi_rank) / 2 >> 2) / (unsigned int)(tok_tiles / 2)) + ((_clc_ctaid_2 + epi_rank) / 2 >> 1 & 1)) * (unsigned int)tok_tiles + (2 * (((_clc_ctaid_2 + epi_rank) / 2 >> 2) % (unsigned int)(tok_tiles / 2)) + ((_clc_ctaid_2 + epi_rank) / 2 & 1));
            }
        }
    }

    // Cleanup
}

} // extern "C"
