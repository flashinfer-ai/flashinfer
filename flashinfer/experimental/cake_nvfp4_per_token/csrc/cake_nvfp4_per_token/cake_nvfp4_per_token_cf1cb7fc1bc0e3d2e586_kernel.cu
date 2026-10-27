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
#define NUM_TMA_PIPE_STAGES 5
#define NUM_ACC_PIPE_STAGES 1
#define NUM_CLC_PIPE_STAGES 2
#define NUM_THR_PIPE_STAGES 2
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 38912
#define SMEM_SMEM_B_OFF 17408
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 38912
#define SMEM_SMEM_V2_OFF 1024
#define SMEM_SMEM_V2_STAGE_BYTES 6144
#define SMEM_SMEM_V2_STRIDE 38912
#define SMEM_SMEM_V3_OFF 1072
#define SMEM_SMEM_V3_STAGE_BYTES 6144
#define SMEM_SMEM_V3_STRIDE 38912
#define SMEM_SMEM_V4_OFF 1120
#define SMEM_SMEM_V4_STAGE_BYTES 6144
#define SMEM_SMEM_V4_STRIDE 38912
#define SMEM_SMEM_V5_OFF 1040
#define SMEM_SMEM_V5_STAGE_BYTES 6144
#define SMEM_SMEM_V5_STRIDE 38912
#define SMEM_SMEM_V6_OFF 1088
#define SMEM_SMEM_V6_STAGE_BYTES 6144
#define SMEM_SMEM_V6_STRIDE 38912
#define SMEM_SMEM_V7_OFF 1136
#define SMEM_SMEM_V7_STAGE_BYTES 6144
#define SMEM_SMEM_V7_STRIDE 38912
#define SMEM_SMEM_V8_OFF 1056
#define SMEM_SMEM_V8_STAGE_BYTES 6144
#define SMEM_SMEM_V8_STRIDE 38912
#define SMEM_SMEM_V9_OFF 1104
#define SMEM_SMEM_V9_STAGE_BYTES 6144
#define SMEM_SMEM_V9_STRIDE 38912
#define SMEM_SMEM_V10_OFF 1024
#define SMEM_SMEM_V10_STAGE_BYTES 6144
#define SMEM_SMEM_V10_STRIDE 38912
#define SMEM_SMEM_V11_OFF 17408
#define SMEM_SMEM_V11_STAGE_BYTES 6144
#define SMEM_SMEM_V11_STRIDE 38912
#define SMEM_SMEM_V12_OFF 17456
#define SMEM_SMEM_V12_STAGE_BYTES 6144
#define SMEM_SMEM_V12_STRIDE 38912
#define SMEM_SMEM_V13_OFF 17504
#define SMEM_SMEM_V13_STAGE_BYTES 6144
#define SMEM_SMEM_V13_STRIDE 38912
#define SMEM_SMEM_V14_OFF 17424
#define SMEM_SMEM_V14_STAGE_BYTES 6144
#define SMEM_SMEM_V14_STRIDE 38912
#define SMEM_SMEM_V15_OFF 17472
#define SMEM_SMEM_V15_STAGE_BYTES 6144
#define SMEM_SMEM_V15_STRIDE 38912
#define SMEM_SMEM_V16_OFF 17520
#define SMEM_SMEM_V16_STAGE_BYTES 6144
#define SMEM_SMEM_V16_STRIDE 38912
#define SMEM_SMEM_V17_OFF 17440
#define SMEM_SMEM_V17_STAGE_BYTES 6144
#define SMEM_SMEM_V17_STRIDE 38912
#define SMEM_SMEM_V18_OFF 17488
#define SMEM_SMEM_V18_STAGE_BYTES 6144
#define SMEM_SMEM_V18_STRIDE 38912
#define SMEM_SMEM_V19_OFF 17408
#define SMEM_SMEM_V19_STAGE_BYTES 6144
#define SMEM_SMEM_V19_STRIDE 38912
#define SMEM_SMEM_SFA_OFF 33792
#define SMEM_SMEM_SFA_STAGE_BYTES 2048
#define SMEM_SMEM_SFA_STRIDE 38912
#define SMEM_SMEM_V21_OFF 35840
#define SMEM_SMEM_V21_STAGE_BYTES 1024
#define SMEM_SMEM_V21_STRIDE 38912
#define SMEM_SMEM_V22_OFF 36864
#define SMEM_SMEM_V22_STAGE_BYTES 1024
#define SMEM_SMEM_V22_STRIDE 38912
#define SMEM_SMEM_V23_OFF 37888
#define SMEM_SMEM_V23_STAGE_BYTES 1024
#define SMEM_SMEM_V23_STRIDE 38912
#define SMEM_SMEM_V24_OFF 38912
#define SMEM_SMEM_V24_STAGE_BYTES 1024
#define SMEM_SMEM_V24_STRIDE 38912
#define SMEM_SMEM_V25_OFF 35840
#define SMEM_SMEM_V25_STAGE_BYTES 2048
#define SMEM_SMEM_V25_STRIDE 38912
#define SMEM_SMEM_V26_OFF 37888
#define SMEM_SMEM_V26_STAGE_BYTES 2048
#define SMEM_SMEM_V26_STRIDE 38912
#define SMEM_SMEM_OUT_OFF 195584
#define SMEM_SMEM_OUT_STAGE_BYTES 8192
#define SMEM_SMEM_OUT_STRIDE 8192
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





__device__ __forceinline__ void tma_store_2d(
    const void *tmap, int x, int y, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2}], [%3];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(smem_addr) : "memory");
}


extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_cake_nvfp4_per_token_cf1cb7fc1bc0e3d2e586(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, uint8_t* __restrict__ SFA_RAW, uint8_t* __restrict__ SFB_RAW, float* __restrict__ alpha, const __grid_constant__ CUtensorMap out, int M, int N, int K_tiles, int tok_tiles, int num_tiles)
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
    #define tma_empty_addr (mbar_base + 40)
    #define acc_full_addr (mbar_base + 80)
    #define acc_empty_addr (mbar_base + 88)
    #define tmem_dealloc_bar_addr (mbar_base + 96)
    #define clc_full_addr (mbar_base + 104)
    #define clc_empty_addr (mbar_base + 120)
    #define thr_full_addr (mbar_base + 136)
    #define thr_empty_addr (mbar_base + 152)

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
    uint8_t* smem_v2 = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_v2_addr = smem + 1024;
    uint8_t* smem_v3 = reinterpret_cast<uint8_t*>(smem_raw + 1072);
    const int smem_v3_addr = smem + 1072;
    uint8_t* smem_v4 = reinterpret_cast<uint8_t*>(smem_raw + 1120);
    const int smem_v4_addr = smem + 1120;
    uint8_t* smem_v5 = reinterpret_cast<uint8_t*>(smem_raw + 1040);
    const int smem_v5_addr = smem + 1040;
    uint8_t* smem_v6 = reinterpret_cast<uint8_t*>(smem_raw + 1088);
    const int smem_v6_addr = smem + 1088;
    uint8_t* smem_v7 = reinterpret_cast<uint8_t*>(smem_raw + 1136);
    const int smem_v7_addr = smem + 1136;
    uint8_t* smem_v8 = reinterpret_cast<uint8_t*>(smem_raw + 1056);
    const int smem_v8_addr = smem + 1056;
    uint8_t* smem_v9 = reinterpret_cast<uint8_t*>(smem_raw + 1104);
    const int smem_v9_addr = smem + 1104;
    uint8_t* smem_v10 = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_v10_addr = smem + 1024;
    uint8_t* smem_v11 = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int smem_v11_addr = smem + 17408;
    uint8_t* smem_v12 = reinterpret_cast<uint8_t*>(smem_raw + 17456);
    const int smem_v12_addr = smem + 17456;
    uint8_t* smem_v13 = reinterpret_cast<uint8_t*>(smem_raw + 17504);
    const int smem_v13_addr = smem + 17504;
    uint8_t* smem_v14 = reinterpret_cast<uint8_t*>(smem_raw + 17424);
    const int smem_v14_addr = smem + 17424;
    uint8_t* smem_v15 = reinterpret_cast<uint8_t*>(smem_raw + 17472);
    const int smem_v15_addr = smem + 17472;
    uint8_t* smem_v16 = reinterpret_cast<uint8_t*>(smem_raw + 17520);
    const int smem_v16_addr = smem + 17520;
    uint8_t* smem_v17 = reinterpret_cast<uint8_t*>(smem_raw + 17440);
    const int smem_v17_addr = smem + 17440;
    uint8_t* smem_v18 = reinterpret_cast<uint8_t*>(smem_raw + 17488);
    const int smem_v18_addr = smem + 17488;
    uint8_t* smem_v19 = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int smem_v19_addr = smem + 17408;
    uint8_t* smem_sfa = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int smem_sfa_addr = smem + 33792;
    uint8_t* smem_v21 = reinterpret_cast<uint8_t*>(smem_raw + 35840);
    const int smem_v21_addr = smem + 35840;
    uint8_t* smem_v22 = reinterpret_cast<uint8_t*>(smem_raw + 36864);
    const int smem_v22_addr = smem + 36864;
    uint8_t* smem_v23 = reinterpret_cast<uint8_t*>(smem_raw + 37888);
    const int smem_v23_addr = smem + 37888;
    uint8_t* smem_v24 = reinterpret_cast<uint8_t*>(smem_raw + 38912);
    const int smem_v24_addr = smem + 38912;
    uint8_t* smem_v25 = reinterpret_cast<uint8_t*>(smem_raw + 35840);
    const int smem_v25_addr = smem + 35840;
    uint8_t* smem_v26 = reinterpret_cast<uint8_t*>(smem_raw + 37888);
    const int smem_v26_addr = smem + 37888;
    __half* smem_out = reinterpret_cast<__half*>(smem_raw + 195584);
    const int smem_out_addr = smem + 195584;
    unsigned int* clc_resp = reinterpret_cast<unsigned int*>(smem_raw + 228352);
    const int clc_resp_addr = smem + 228352;
    if (warp == 2) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A))) : "memory"); }
    if (warp == 2) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B))) : "memory"); }
    if (warp == 2) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SFA))) : "memory"); }
    if (warp == 2) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SFB))) : "memory"); }
    if (warp == 2) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&out))) : "memory"); }

    // Mbarrier init (9 pipeline groups, 0 ordered-sequence groups, 21 barriers)
    // Mbarriers at smem_raw[0..168)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full: 5 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            // tma_empty: 5 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // --- pipeline 'acc_pipe' ---
            // acc_full: 1 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            // acc_empty: 1 barriers, init_count=256
            mbarrier_init(smem + 88, 256);
            // tmem_dealloc_bar: 1 barriers, init_count=32
            mbarrier_init(smem + 96, 32);
            // --- pipeline 'clc_pipe' ---
            // clc_full: 2 barriers, init_count=1
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            // clc_empty: 2 barriers, init_count=416
            mbarrier_init(smem + 120, 416);
            mbarrier_init(smem + 128, 416);
            // --- pipeline 'thr_pipe' ---
            // thr_full: 2 barriers, init_count=32
            mbarrier_init(smem + 136, 32);
            mbarrier_init(smem + 144, 32);
            // thr_empty: 2 barriers, init_count=32
            mbarrier_init(smem + 152, 32);
            mbarrier_init(smem + 160, 32);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 400 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 168);
    if (warp == 2) {
        int _tmem_hold = smem + 168;
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
            unsigned int tile_l = cluster_id;
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
                int w_idx = tile_l / (unsigned int)tok_tiles;
                int tok_idx = tile_l - (unsigned int)(w_idx * tok_tiles);
                unsigned int _min_0 = ((8) < ((unsigned int)tok_tiles - tile_l / (unsigned int)(8 * (num_tiles / tok_tiles)) * 8) ? (8) : ((unsigned int)tok_tiles - tile_l / (unsigned int)(8 * (num_tiles / tok_tiles)) * 8));
                w_idx = (tile_l - tile_l / (unsigned int)(8 * (num_tiles / tok_tiles)) * (unsigned int)(8 * (num_tiles / tok_tiles))) / (unsigned int)_min_0;
                unsigned int _min_0_1 = ((8) < ((unsigned int)tok_tiles - tile_l / (unsigned int)(8 * (num_tiles / tok_tiles)) * 8) ? (8) : ((unsigned int)tok_tiles - tile_l / (unsigned int)(8 * (num_tiles / tok_tiles)) * 8));
                tok_idx = tile_l / (unsigned int)(8 * (num_tiles / tok_tiles)) * 8 + (tile_l - tile_l / (unsigned int)(8 * (num_tiles / tok_tiles)) * (unsigned int)(8 * (num_tiles / tok_tiles))) - (tile_l - tile_l / (unsigned int)(8 * (num_tiles / tok_tiles)) * (unsigned int)(8 * (num_tiles / tok_tiles))) / (unsigned int)_min_0_1 * (unsigned int)_min_0_1;
                int a_rows = tok_idx * 256 + cta_rank * 128;
                int b_rows = w_idx * 256 + cta_rank * 128;
                int a_atom = a_rows / 128;
                int b_atom = w_idx * 256 / 128;
                #pragma unroll 1
                for (unsigned int k_tile = 0; k_tile < K_tiles; k_tile++) {
                    mbarrier_wait(tma_empty_addr + (load_stage) * 8, _phase_tma_empty);
                    if (cta_rank == 0) {
                        if (elect_sync()) {
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(77824)) : "memory");
                        }
                    }
                    if (elect_sync()) {
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(smem_a_addr + load_stage * 38912), "l"((&A)), "r"(0), "r"(a_rows), "r"(k_tile),
                               "r"(((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(smem_sfa_addr + load_stage * 38912), "l"((&SFA)), "r"(0), "r"(4 * k_tile), "r"(a_atom),
                               "r"(((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.cta_group::2.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                            :: "r"(smem_b_addr + load_stage * 38912), "l"((&B)), "r"(0), "r"(b_rows), "r"(k_tile),
                               "r"(((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF)), "l"(0x14F0000000000000ULL) : "memory");
                    }
                    if (cta_rank == 0) {
                        if (elect_sync()) {
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6, %7;"
                                :: "r"(smem_v21_addr + load_stage * 38912), "l"((&SFB)), "r"(0), "r"(4 * k_tile), "r"(b_atom),
                                   "r"(((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6, %7;"
                                :: "r"(smem_v22_addr + load_stage * 38912), "l"((&SFB)), "r"(0), "r"(4 * k_tile + 1), "r"(b_atom),
                                   "r"(((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6, %7;"
                                :: "r"(smem_v23_addr + load_stage * 38912), "l"((&SFB)), "r"(0), "r"(4 * k_tile + 2), "r"(b_atom),
                                   "r"(((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)), "l"(0x14F0000000000000ULL) : "memory");
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2.L2::cache_hint"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6, %7;"
                                :: "r"(smem_v24_addr + load_stage * 38912), "l"((&SFB)), "r"(0), "r"(4 * k_tile + 3), "r"(b_atom),
                                   "r"(((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                    load_stage += 1;
                    if (load_stage == 5) { load_stage = 0; _phase_tma_empty ^= 1; }
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
                    :: "r"(clc_empty_addr + load_clc_stage * 8), "r"(0) : "memory");
                load_clc_stage += 1;
                if (load_clc_stage == 2) { load_clc_stage = 0; _phase_clc_full ^= 1; }
                if (_clc_valid_1 == 0) {
                    break;
                }
                tile_l = _clc_ctaid_0 / 2;
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            unsigned int mma_stage = 0;
            unsigned int completed = 0;
            unsigned int mma_clc_stage = 0;
            unsigned int tile_m = cluster_id;
            unsigned int _phase_acc_empty = 1;
            unsigned int _phase_tma_full = 0;
            unsigned int _phase_clc_full_1 = 0;
            #pragma unroll 1
            for (unsigned int _p_1 = 0; _p_1 < num_tiles; _p_1++) {
                if (cta_rank == 0) {
                    int acc_base = (int)0 * 256;
                    int w_idx_mma = tile_m / (unsigned int)tok_tiles;
                    unsigned int _min_0_2 = ((8) < ((unsigned int)tok_tiles - tile_m / (unsigned int)(8 * (num_tiles / tok_tiles)) * 8) ? (8) : ((unsigned int)tok_tiles - tile_m / (unsigned int)(8 * (num_tiles / tok_tiles)) * 8));
                    w_idx_mma = (tile_m - tile_m / (unsigned int)(8 * (num_tiles / tok_tiles)) * (unsigned int)(8 * (num_tiles / tok_tiles))) / (unsigned int)_min_0_2;
                    int sfb_shift = w_idx_mma * 256 % 128 / 32;
                    mbarrier_wait(acc_empty_addr, _phase_acc_empty);
                    int k96_supers = K_tiles / 3;
                    int k96_tail = K_tiles - k96_supers * 3;
                    #pragma unroll 1
                    for (unsigned int k_super = 0; k_super < k96_supers; k_super++) {
                        int init_super = ((k_super == 0) ? 1 : 0);
                        mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        unsigned int st_p0 = mma_stage;
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa, make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (st_p0) * 2432)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfa + 4), make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (st_p0) * 2432 + 32)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfa + 8), make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (st_p0) * 2432 + 64)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfa + 12), make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (st_p0) * 2432 + 96)));
                        }
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb, make_sf_cp_desc_lo_sbo128((((smem_v25_addr) >> 4) + (st_p0) * 2432)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 4), make_sf_cp_desc_lo_sbo128((((smem_v25_addr) >> 4) + (st_p0) * 2432 + 32)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 8), make_sf_cp_desc_lo_sbo128((((smem_v25_addr) >> 4) + (st_p0) * 2432 + 64)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 12), make_sf_cp_desc_lo_sbo128((((smem_v25_addr) >> 4) + (st_p0) * 2432 + 96)));
                        }
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + 16, make_sf_cp_desc_lo_sbo128((((smem_v26_addr) >> 4) + (st_p0) * 2432)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 16 + 4), make_sf_cp_desc_lo_sbo128((((smem_v26_addr) >> 4) + (st_p0) * 2432 + 32)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 16 + 8), make_sf_cp_desc_lo_sbo128((((smem_v26_addr) >> 4) + (st_p0) * 2432 + 64)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 16 + 12), make_sf_cp_desc_lo_sbo128((((smem_v26_addr) >> 4) + (st_p0) * 2432 + 96)));
                        }
                        int _mma_a_lo_0 = (((smem_v2_addr) >> 4) & 0x3FFF) + (st_p0) * 2432;
                        int _mma_b_lo_0 = (((smem_v11_addr) >> 4) & 0x3FFF) + (st_p0) * 2432;
                        int _mma_a_next_lo_0 = (((smem_v10_addr) >> 4) & 0x3FFF) + (st_p0) * 2432;
                        int _mma_b_next_lo_0 = (((smem_v19_addr) >> 4) & 0x3FFF) + (st_p0) * 2432;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0 | ((uint64_t)((uint32_t)_mma_a_next_lo_0 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0 | ((uint64_t)((uint32_t)_mma_b_next_lo_0 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2((tmem_acc + (acc_base)), a_desc + 0, b_desc + 0,
                                    0x90400480U, tmem_tmem_sfa + 0, tmem_tmem_sfb + sfb_shift + 0, ((init_super) ? 0 : 1));
                            }
                        }
                        int _mma_a_lo_1 = (((smem_v3_addr) >> 4) & 0x3FFF) + (st_p0) * 2432;
                        int _mma_b_lo_1 = (((smem_v12_addr) >> 4) & 0x3FFF) + (st_p0) * 2432;
                        int _mma_a_next_lo_1 = (((smem_v10_addr) >> 4) & 0x3FFF) + (st_p0) * 2432;
                        int _mma_b_next_lo_1 = (((smem_v19_addr) >> 4) & 0x3FFF) + (st_p0) * 2432;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1 | ((uint64_t)((uint32_t)_mma_a_next_lo_1 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1 | ((uint64_t)((uint32_t)_mma_b_next_lo_1 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2((tmem_acc + (acc_base)), a_desc + 0, b_desc + 0,
                                    0xd04004a0U, tmem_tmem_sfa + 4 + 0, tmem_tmem_sfb + (8 + sfb_shift) + 0, 1);
                            }
                        }
                        mma_stage += 1;
                        if (mma_stage == 5) { mma_stage = 0; _phase_tma_full ^= 1; }
                        mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        unsigned int st_p1 = mma_stage;
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + 16, make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (st_p1) * 2432)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfa + 16 + 4), make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (st_p1) * 2432 + 32)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfa + 16 + 8), make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (st_p1) * 2432 + 64)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfa + 16 + 12), make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (st_p1) * 2432 + 96)));
                        }
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + 32, make_sf_cp_desc_lo_sbo128((((smem_v25_addr) >> 4) + (st_p1) * 2432)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 32 + 4), make_sf_cp_desc_lo_sbo128((((smem_v25_addr) >> 4) + (st_p1) * 2432 + 32)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 32 + 8), make_sf_cp_desc_lo_sbo128((((smem_v25_addr) >> 4) + (st_p1) * 2432 + 64)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 32 + 12), make_sf_cp_desc_lo_sbo128((((smem_v25_addr) >> 4) + (st_p1) * 2432 + 96)));
                        }
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + 48, make_sf_cp_desc_lo_sbo128((((smem_v26_addr) >> 4) + (st_p1) * 2432)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 48 + 4), make_sf_cp_desc_lo_sbo128((((smem_v26_addr) >> 4) + (st_p1) * 2432 + 32)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 48 + 8), make_sf_cp_desc_lo_sbo128((((smem_v26_addr) >> 4) + (st_p1) * 2432 + 64)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 48 + 12), make_sf_cp_desc_lo_sbo128((((smem_v26_addr) >> 4) + (st_p1) * 2432 + 96)));
                        }
                        int _mma_a_lo_2 = (((smem_v4_addr) >> 4) & 0x3FFF) + (st_p0) * 2432;
                        int _mma_b_lo_2 = (((smem_v13_addr) >> 4) & 0x3FFF) + (st_p0) * 2432;
                        int _mma_a_next_lo_2 = (((smem_v10_addr) >> 4) & 0x3FFF) + (st_p1) * 2432;
                        int _mma_b_next_lo_2 = (((smem_v19_addr) >> 4) & 0x3FFF) + (st_p1) * 2432;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_2 | ((uint64_t)((uint32_t)_mma_a_next_lo_2 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_2 | ((uint64_t)((uint32_t)_mma_b_next_lo_2 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2((tmem_acc + (acc_base)), a_desc + 0, b_desc + 0,
                                    0x90400480U, tmem_tmem_sfa + 12 + 0, tmem_tmem_sfb + (24 + sfb_shift) + 0, 1);
                            }
                        }
                        elect_commit_cg2_multicast(tma_empty_addr + (st_p0) * 8, (uint16_t)(3));
                        int _mma_a_lo_3 = (((smem_v5_addr) >> 4) & 0x3FFF) + (st_p1) * 2432;
                        int _mma_b_lo_3 = (((smem_v14_addr) >> 4) & 0x3FFF) + (st_p1) * 2432;
                        int _mma_a_next_lo_3 = (((smem_v10_addr) >> 4) & 0x3FFF) + (st_p1) * 2432;
                        int _mma_b_next_lo_3 = (((smem_v19_addr) >> 4) & 0x3FFF) + (st_p1) * 2432;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_3 | ((uint64_t)((uint32_t)_mma_a_next_lo_3 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_3 | ((uint64_t)((uint32_t)_mma_b_next_lo_3 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2((tmem_acc + (acc_base)), a_desc + 0, b_desc + 0,
                                    0xd04004a0U, tmem_tmem_sfa + 16 + 0, tmem_tmem_sfb + (32 + sfb_shift) + 0, 1);
                            }
                        }
                        int _mma_a_lo_4 = (((smem_v6_addr) >> 4) & 0x3FFF) + (st_p1) * 2432;
                        int _mma_b_lo_4 = (((smem_v15_addr) >> 4) & 0x3FFF) + (st_p1) * 2432;
                        int _mma_a_next_lo_4 = (((smem_v10_addr) >> 4) & 0x3FFF) + (st_p1) * 2432;
                        int _mma_b_next_lo_4 = (((smem_v19_addr) >> 4) & 0x3FFF) + (st_p1) * 2432;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_4 | ((uint64_t)((uint32_t)_mma_a_next_lo_4 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_4 | ((uint64_t)((uint32_t)_mma_b_next_lo_4 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2((tmem_acc + (acc_base)), a_desc + 0, b_desc + 0,
                                    0x90400480U, tmem_tmem_sfa + 24 + 0, tmem_tmem_sfb + (48 + sfb_shift) + 0, 1);
                            }
                        }
                        mma_stage += 1;
                        if (mma_stage == 5) { mma_stage = 0; _phase_tma_full ^= 1; }
                        mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        unsigned int st_p2 = mma_stage;
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + 32, make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (st_p2) * 2432)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfa + 32 + 4), make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (st_p2) * 2432 + 32)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfa + 32 + 8), make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (st_p2) * 2432 + 64)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfa + 32 + 12), make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (st_p2) * 2432 + 96)));
                        }
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + 64, make_sf_cp_desc_lo_sbo128((((smem_v25_addr) >> 4) + (st_p2) * 2432)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 64 + 4), make_sf_cp_desc_lo_sbo128((((smem_v25_addr) >> 4) + (st_p2) * 2432 + 32)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 64 + 8), make_sf_cp_desc_lo_sbo128((((smem_v25_addr) >> 4) + (st_p2) * 2432 + 64)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 64 + 12), make_sf_cp_desc_lo_sbo128((((smem_v25_addr) >> 4) + (st_p2) * 2432 + 96)));
                        }
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + 80, make_sf_cp_desc_lo_sbo128((((smem_v26_addr) >> 4) + (st_p2) * 2432)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 80 + 4), make_sf_cp_desc_lo_sbo128((((smem_v26_addr) >> 4) + (st_p2) * 2432 + 32)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 80 + 8), make_sf_cp_desc_lo_sbo128((((smem_v26_addr) >> 4) + (st_p2) * 2432 + 64)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 80 + 12), make_sf_cp_desc_lo_sbo128((((smem_v26_addr) >> 4) + (st_p2) * 2432 + 96)));
                        }
                        int _mma_a_lo_5 = (((smem_v7_addr) >> 4) & 0x3FFF) + (st_p1) * 2432;
                        int _mma_b_lo_5 = (((smem_v16_addr) >> 4) & 0x3FFF) + (st_p1) * 2432;
                        int _mma_a_next_lo_5 = (((smem_v10_addr) >> 4) & 0x3FFF) + (st_p2) * 2432;
                        int _mma_b_next_lo_5 = (((smem_v19_addr) >> 4) & 0x3FFF) + (st_p2) * 2432;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_5 | ((uint64_t)((uint32_t)_mma_a_next_lo_5 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_5 | ((uint64_t)((uint32_t)_mma_b_next_lo_5 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2((tmem_acc + (acc_base)), a_desc + 0, b_desc + 0,
                                    0xd04004a0U, tmem_tmem_sfa + 28 + 0, tmem_tmem_sfb + (56 + sfb_shift) + 0, 1);
                            }
                        }
                        elect_commit_cg2_multicast(tma_empty_addr + (st_p1) * 8, (uint16_t)(3));
                        int _mma_a_lo_6 = (((smem_v8_addr) >> 4) & 0x3FFF) + (st_p2) * 2432;
                        int _mma_b_lo_6 = (((smem_v17_addr) >> 4) & 0x3FFF) + (st_p2) * 2432;
                        int _mma_a_next_lo_6 = (((smem_v10_addr) >> 4) & 0x3FFF) + (st_p2) * 2432;
                        int _mma_b_next_lo_6 = (((smem_v19_addr) >> 4) & 0x3FFF) + (st_p2) * 2432;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_6 | ((uint64_t)((uint32_t)_mma_a_next_lo_6 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_6 | ((uint64_t)((uint32_t)_mma_b_next_lo_6 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2((tmem_acc + (acc_base)), a_desc + 0, b_desc + 0,
                                    0x90400480U, tmem_tmem_sfa + 36 + 0, tmem_tmem_sfb + (72 + sfb_shift) + 0, 1);
                            }
                        }
                        int _mma_a_lo_7 = (((smem_v9_addr) >> 4) & 0x3FFF) + (st_p2) * 2432;
                        int _mma_b_lo_7 = (((smem_v18_addr) >> 4) & 0x3FFF) + (st_p2) * 2432;
                        int _mma_a_next_lo_7 = (((smem_v10_addr) >> 4) & 0x3FFF) + (st_p2) * 2432;
                        int _mma_b_next_lo_7 = (((smem_v19_addr) >> 4) & 0x3FFF) + (st_p2) * 2432;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_7 | ((uint64_t)((uint32_t)_mma_a_next_lo_7 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_7 | ((uint64_t)((uint32_t)_mma_b_next_lo_7 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2((tmem_acc + (acc_base)), a_desc + 0, b_desc + 0,
                                    0xd04004a0U, tmem_tmem_sfa + 40 + 0, tmem_tmem_sfb + (80 + sfb_shift) + 0, 1);
                            }
                        }
                        elect_commit_cg2_multicast(tma_empty_addr + (st_p2) * 8, (uint16_t)(3));
                        mma_stage += 1;
                        if (mma_stage == 5) { mma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    #pragma unroll 1
                    for (unsigned int k_slot = 0; k_slot < k96_tail; k_slot++) {
                        mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2((unsigned int)tmem_tmem_sfa + k_slot * 16, make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_stage) * 2432)));
                            tcgen05_cp_32x128b_warpx4_cta2(((unsigned int)tmem_tmem_sfa + k_slot * 16 + 4), make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_stage) * 2432 + 32)));
                            tcgen05_cp_32x128b_warpx4_cta2(((unsigned int)tmem_tmem_sfa + k_slot * 16 + 8), make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_stage) * 2432 + 64)));
                            tcgen05_cp_32x128b_warpx4_cta2(((unsigned int)tmem_tmem_sfa + k_slot * 16 + 12), make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_stage) * 2432 + 96)));
                        }
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2((unsigned int)tmem_tmem_sfb + k_slot * 2 * 16, make_sf_cp_desc_lo_sbo128((((smem_v25_addr) >> 4) + (mma_stage) * 2432)));
                            tcgen05_cp_32x128b_warpx4_cta2(((unsigned int)tmem_tmem_sfb + k_slot * 2 * 16 + 4), make_sf_cp_desc_lo_sbo128((((smem_v25_addr) >> 4) + (mma_stage) * 2432 + 32)));
                            tcgen05_cp_32x128b_warpx4_cta2(((unsigned int)tmem_tmem_sfb + k_slot * 2 * 16 + 8), make_sf_cp_desc_lo_sbo128((((smem_v25_addr) >> 4) + (mma_stage) * 2432 + 64)));
                            tcgen05_cp_32x128b_warpx4_cta2(((unsigned int)tmem_tmem_sfb + k_slot * 2 * 16 + 12), make_sf_cp_desc_lo_sbo128((((smem_v25_addr) >> 4) + (mma_stage) * 2432 + 96)));
                        }
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2((unsigned int)tmem_tmem_sfb + (k_slot * 2 + 1) * 16, make_sf_cp_desc_lo_sbo128((((smem_v26_addr) >> 4) + (mma_stage) * 2432)));
                            tcgen05_cp_32x128b_warpx4_cta2(((unsigned int)tmem_tmem_sfb + (k_slot * 2 + 1) * 16 + 4), make_sf_cp_desc_lo_sbo128((((smem_v26_addr) >> 4) + (mma_stage) * 2432 + 32)));
                            tcgen05_cp_32x128b_warpx4_cta2(((unsigned int)tmem_tmem_sfb + (k_slot * 2 + 1) * 16 + 8), make_sf_cp_desc_lo_sbo128((((smem_v26_addr) >> 4) + (mma_stage) * 2432 + 64)));
                            tcgen05_cp_32x128b_warpx4_cta2(((unsigned int)tmem_tmem_sfb + (k_slot * 2 + 1) * 16 + 12), make_sf_cp_desc_lo_sbo128((((smem_v26_addr) >> 4) + (mma_stage) * 2432 + 96)));
                        }
                        int tail_sfa_col = (int)k_slot * 16;
                        int tail_sfb_col = (int)k_slot * 32;
                        int init_tail = (((int)k_slot + k96_supers == 0) ? 1 : 0);
                        int _mma_a_lo_8 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage) * 2432;
                        int _mma_b_lo_8 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_stage) * 2432;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_8) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_8) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_cta2((tmem_acc + (acc_base)), a_desc + 0, b_desc + 0,
                                    0x10400480U, tmem_tmem_sfa + tail_sfa_col + 0, tmem_tmem_sfb + (tail_sfb_col + sfb_shift) + 0, ((init_tail) ? 0 : 1));
                            }
                        }
                        int _mma_a_lo_9 = (((smem_a_addr + 32) >> 4) & 0x3FFF) + (mma_stage) * 2432;
                        int _mma_b_lo_9 = (((smem_b_addr + 32) >> 4) & 0x3FFF) + (mma_stage) * 2432;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_9) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_9) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_cta2((tmem_acc + (acc_base)), a_desc + 0, b_desc + 0,
                                    0x10400480U, tmem_tmem_sfa + (tail_sfa_col + 4) + 0, tmem_tmem_sfb + (tail_sfb_col + 8 + sfb_shift) + 0, 1);
                            }
                        }
                        int _mma_a_lo_10 = (((smem_a_addr + 64) >> 4) & 0x3FFF) + (mma_stage) * 2432;
                        int _mma_b_lo_10 = (((smem_b_addr + 64) >> 4) & 0x3FFF) + (mma_stage) * 2432;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_10) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_10) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_cta2((tmem_acc + (acc_base)), a_desc + 0, b_desc + 0,
                                    0x10400480U, tmem_tmem_sfa + (tail_sfa_col + 8) + 0, tmem_tmem_sfb + (tail_sfb_col + 16 + sfb_shift) + 0, 1);
                            }
                        }
                        int _mma_a_lo_11 = (((smem_a_addr + 96) >> 4) & 0x3FFF) + (mma_stage) * 2432;
                        int _mma_b_lo_11 = (((smem_b_addr + 96) >> 4) & 0x3FFF) + (mma_stage) * 2432;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_11) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_11) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_cta2((tmem_acc + (acc_base)), a_desc + 0, b_desc + 0,
                                    0x10400480U, tmem_tmem_sfa + (tail_sfa_col + 12) + 0, tmem_tmem_sfb + (tail_sfb_col + 24 + sfb_shift) + 0, 1);
                            }
                        }
                        elect_commit_cg2_multicast(tma_empty_addr + (mma_stage) * 8, (uint16_t)(3));
                        mma_stage += 1;
                        if (mma_stage == 5) { mma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    elect_commit_cg2_multicast(acc_full_addr, (uint16_t)(3));
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
                    :: "r"(clc_empty_addr + mma_clc_stage * 8), "r"(0) : "memory");
                mma_clc_stage += 1;
                if (mma_clc_stage == 2) { mma_clc_stage = 0; _phase_clc_full_1 ^= 1; }
                if (_clc_valid_2 == 0) {
                    break;
                }
                tile_m = _clc_ctaid_1 / 2;
            }
            if (cta_rank == 0) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
                if (completed > 0) {
                    unsigned int _parity_acc_empty = completed % 2 ^ 1;
                    mbarrier_wait(acc_empty_addr, _parity_acc_empty);
                }
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
            asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
        }
    }
    // ---- Role: prefetch ----
    if (warp == 2) {
        // idle — no tasks assigned
    }
    // ---- Role: sched ----
    if (warp == 3) {
        { // sched_main
            unsigned int _phase_thr_full = 0;
            unsigned int _phase_clc_empty = 1;
            unsigned int _phase_clc_full_2 = 0;
            if (cta_rank == 0) {
                unsigned int sched_clc_stage = 0;
                unsigned int sched_thr_stage = 0;
                #pragma unroll 1
                for (unsigned int _p_2 = 0; _p_2 < num_tiles; _p_2++) {
                    mbarrier_wait(thr_full_addr + (sched_thr_stage) * 8, _phase_thr_full);
                    mbarrier_arrive(thr_empty_addr + (sched_thr_stage) * 8);
                    sched_thr_stage += 1;
                    if (sched_thr_stage == 2) { sched_thr_stage = 0; _phase_thr_full ^= 1; }
                    mbarrier_wait(clc_empty_addr + (sched_clc_stage) * 8, _phase_clc_empty);
                    if (lane < 2) {
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                            "}"
                            :: "r"(clc_full_addr + sched_clc_stage * 8), "r"(lane), "r"((uint32_t)(16)) : "memory");
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
                    mbarrier_wait(clc_full_addr + (sched_clc_stage) * 8, _phase_clc_full_2);
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
                        :: "r"(clc_empty_addr + sched_clc_stage * 8), "r"(0) : "memory");
                    sched_clc_stage += 1;
                    if (sched_clc_stage == 2) { sched_clc_stage = 0; _phase_clc_empty ^= 1; _phase_clc_full_2 ^= 1; }
                    if (_clc_valid_0 == 0) {
                        break;
                    }
                }
                #pragma unroll
                for (int _ = 0; _ < 2; _++) {
                    mbarrier_wait(clc_empty_addr + (sched_clc_stage) * 8, _phase_clc_empty);
                    sched_clc_stage += 1;
                    if (sched_clc_stage == 2) { sched_clc_stage = 0; _phase_clc_empty ^= 1; _phase_clc_full_2 ^= 1; }
                }
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 4 && warp <= 7) {
        { // epilogue_main
            const int epi_warp = warp - 4;
            int epi_row = epi_warp % 4 * 32 + lane;
            int epi_srow = epi_warp * 32 + lane;
            unsigned int store_stage = 0;
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int epi_clc_stage = 0;
            unsigned int tile_e = cluster_id;
            unsigned int _phase_clc_full_3 = 0;
            unsigned int _phase_acc_full = 0;
            #pragma unroll 1
            for (unsigned int _p_3 = 0; _p_3 < num_tiles; _p_3++) {
                mbarrier_wait(clc_full_addr + (epi_clc_stage) * 8, _phase_clc_full_3);
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
                    :: "r"(clc_empty_addr + epi_clc_stage * 8), "r"(0) : "memory");
                epi_clc_stage += 1;
                if (epi_clc_stage == 2) { epi_clc_stage = 0; _phase_clc_full_3 ^= 1; }
                int w_idx_1 = tile_e / (unsigned int)tok_tiles;
                int tok_idx_1 = tile_e - (unsigned int)(w_idx_1 * tok_tiles);
                unsigned int _min_0_3 = ((8) < ((unsigned int)tok_tiles - tile_e / (unsigned int)(8 * (num_tiles / tok_tiles)) * 8) ? (8) : ((unsigned int)tok_tiles - tile_e / (unsigned int)(8 * (num_tiles / tok_tiles)) * 8));
                w_idx_1 = (tile_e - tile_e / (unsigned int)(8 * (num_tiles / tok_tiles)) * (unsigned int)(8 * (num_tiles / tok_tiles))) / (unsigned int)_min_0_3;
                unsigned int _min_1 = ((8) < ((unsigned int)tok_tiles - tile_e / (unsigned int)(8 * (num_tiles / tok_tiles)) * 8) ? (8) : ((unsigned int)tok_tiles - tile_e / (unsigned int)(8 * (num_tiles / tok_tiles)) * 8));
                tok_idx_1 = tile_e / (unsigned int)(8 * (num_tiles / tok_tiles)) * 8 + (tile_e - tile_e / (unsigned int)(8 * (num_tiles / tok_tiles)) * (unsigned int)(8 * (num_tiles / tok_tiles))) - (tile_e - tile_e / (unsigned int)(8 * (num_tiles / tok_tiles)) * (unsigned int)(8 * (num_tiles / tok_tiles))) / (unsigned int)_min_1 * (unsigned int)_min_1;
                int off_tok = tok_idx_1 * 256 + cta_rank * 128;
                int off_w = w_idx_1 * 256;
                int acc_base_1 = (int)0 * 256;
                int lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)acc_base_1;
                int tok_row = ((off_tok + epi_row) < (M - 1) ? (off_tok + epi_row) : (M - 1));
                float alpha_row = alpha[tok_row];
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
                        float2 _pair_scale_even2_1 = make_float2(alpha_row, alpha_row);
                        float2 _pair_scale_odd2_1 = make_float2(alpha_row, alpha_row);
                        float2* _pair_scale_src2_1 = reinterpret_cast<float2*>(&_tmem_load_0[0]);
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
                    unsigned int _tmem_load_0_f16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __half2 _h2 = __float22half2_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                        _tmem_load_0_f16[_lp] = *(uint32_t*)&_h2;
                    }
                    if (subtile > 0) {
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (warp == 4) {
                            if (elect_sync()) {
                                tma_store_2d((&out), off_w + (subtile - 1) * 32, off_tok, smem_out_addr + store_stage * 8192);
                            }
                        }
                        if (warp == 4) {
                            asm volatile("cp.async.bulk.commit_group;");
                            asm volatile("cp.async.bulk.wait_group.read 3;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        store_stage = store_stage + 1;
                        if (store_stage == 4) {
                            store_stage = 0;
                        }
                    }
                    int out_stage_row = store_stage * 128 + (unsigned int)epi_srow;
                    unsigned int out_abs = smem_out_addr + (unsigned int)(out_stage_row * 64);
                    unsigned int out_swz = out_abs / 8 & 48;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_out_addr + ((unsigned int)(out_stage_row * 64) + (0 ^ out_swz)))), "r"(_tmem_load_0_f16[0]), "r"(_tmem_load_0_f16[1]), "r"(_tmem_load_0_f16[2]), "r"(_tmem_load_0_f16[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_out_addr + ((unsigned int)(out_stage_row * 64) + (16 ^ out_swz)))), "r"(_tmem_load_0_f16[4]), "r"(_tmem_load_0_f16[5]), "r"(_tmem_load_0_f16[6]), "r"(_tmem_load_0_f16[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_out_addr + ((unsigned int)(out_stage_row * 64) + (32 ^ out_swz)))), "r"(_tmem_load_0_f16[8]), "r"(_tmem_load_0_f16[9]), "r"(_tmem_load_0_f16[10]), "r"(_tmem_load_0_f16[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_out_addr + ((unsigned int)(out_stage_row * 64) + (48 ^ out_swz)))), "r"(_tmem_load_0_f16[12]), "r"(_tmem_load_0_f16[13]), "r"(_tmem_load_0_f16[14]), "r"(_tmem_load_0_f16[15]) : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                if (warp == 4) {
                    if (elect_sync()) {
                        tma_store_2d((&out), off_w + 224, off_tok, smem_out_addr + store_stage * 8192);
                    }
                }
                if (warp == 4) {
                    asm volatile("cp.async.bulk.commit_group;");
                    asm volatile("cp.async.bulk.wait_group.read 3;");
                }
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                store_stage = store_stage + 1;
                if (store_stage == 4) {
                    store_stage = 0;
                }
                _phase_acc_full ^= 1;
                if (_clc_valid_3 == 0) {
                    break;
                }
                tile_e = _clc_ctaid_2 / 2;
            }
            if (warp == 4) {
                asm volatile("cp.async.bulk.wait_group 0;");
            }
        }
    }

    // Cleanup
}

} // extern "C"
