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
#define TMEM_ACCUM_OFFSET 0
#define TMEM_TMEM_SFA_OFFSET 448
#define TMEM_TMEM_SFB_OFFSET 464
#define NUM_TMA_PIPE_STAGES 5
#define NUM_ACC_PIPE_STAGES 1
#define NUM_ACCF_PIPE_STAGES 2
#define NUM_CLC_PIPE_STAGES 2
#define NUM_THR_PIPE_STAGES 2
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 38912
#define SMEM_SMEM_B_OFF 17408
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 38912
#define SMEM_SMEM_SFA_OFF 33792
#define SMEM_SMEM_SFA_STAGE_BYTES 2048
#define SMEM_SMEM_SFA_STRIDE 38912
#define SMEM_SMEM_V3_OFF 35840
#define SMEM_SMEM_V3_STAGE_BYTES 1024
#define SMEM_SMEM_V3_STRIDE 38912
#define SMEM_SMEM_V4_OFF 36864
#define SMEM_SMEM_V4_STAGE_BYTES 1024
#define SMEM_SMEM_V4_STRIDE 38912
#define SMEM_SMEM_V5_OFF 37888
#define SMEM_SMEM_V5_STAGE_BYTES 1024
#define SMEM_SMEM_V5_STRIDE 38912
#define SMEM_SMEM_V6_OFF 38912
#define SMEM_SMEM_V6_STAGE_BYTES 1024
#define SMEM_SMEM_V6_STRIDE 38912
#define SMEM_SMEM_V7_OFF 35840
#define SMEM_SMEM_V7_STAGE_BYTES 2048
#define SMEM_SMEM_V7_STRIDE 38912
#define SMEM_SMEM_V8_OFF 37888
#define SMEM_SMEM_V8_STAGE_BYTES 2048
#define SMEM_SMEM_V8_STRIDE 38912
#define SMEM_SMEM_V9_OFF 35840
#define SMEM_SMEM_V9_STAGE_BYTES 512
#define SMEM_SMEM_V9_STRIDE 38912
#define SMEM_SMEM_V10_OFF 36352
#define SMEM_SMEM_V10_STAGE_BYTES 512
#define SMEM_SMEM_V10_STRIDE 38912
#define SMEM_SMEM_V11_OFF 36864
#define SMEM_SMEM_V11_STAGE_BYTES 512
#define SMEM_SMEM_V11_STRIDE 38912
#define SMEM_SMEM_V12_OFF 37376
#define SMEM_SMEM_V12_STAGE_BYTES 512
#define SMEM_SMEM_V12_STRIDE 38912
#define SMEM_SMEM_V13_OFF 37888
#define SMEM_SMEM_V13_STAGE_BYTES 512
#define SMEM_SMEM_V13_STRIDE 38912
#define SMEM_SMEM_V14_OFF 38400
#define SMEM_SMEM_V14_STAGE_BYTES 512
#define SMEM_SMEM_V14_STRIDE 38912
#define SMEM_SMEM_V15_OFF 38912
#define SMEM_SMEM_V15_STAGE_BYTES 512
#define SMEM_SMEM_V15_STRIDE 38912
#define SMEM_SMEM_V16_OFF 39424
#define SMEM_SMEM_V16_STAGE_BYTES 512
#define SMEM_SMEM_V16_STRIDE 38912
#define SMEM_SMEM_D_OFF 1024
#define SMEM_SMEM_D_STAGE_BYTES 16384
#define SMEM_SMEM_D_STRIDE 38912
#define SMEM_SMEM_L1_PRE_OFF 17408
#define SMEM_SMEM_L1_PRE_STAGE_BYTES 12288
#define SMEM_SMEM_L1_PRE_STRIDE 38912
#define SMEM_SMEM_L1_TAIL_OFF 29696
#define SMEM_SMEM_L1_TAIL_STAGE_BYTES 4096
#define SMEM_SMEM_L1_TAIL_STRIDE 38912
#define SMEM_SMEM_OUT_OFF 195584
#define SMEM_SMEM_OUT_STAGE_BYTES 16384
#define SMEM_SMEM_OUT_STRIDE 16384
#define SMEM_SMEM_BIAS_PAIRS_OFF 228352
#define SMEM_SMEM_BIAS_PAIRS_STAGE_BYTES 512
#define SMEM_SMEM_BIAS_PAIRS_STRIDE 512
#define SMEM_CLC_RESP_OFF 228864
#define SMEM_CLC_RESP_STAGE_BYTES 16
#define SMEM_CLC_RESP_STRIDE 16
#define SMEM_TOTAL 228992
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


__device__ __forceinline__ void elect_commit_cg2_local(int mbar_addr) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "@leader tcgen05.commit.cta_group::2.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];\n\t"
        "}\n"
        :: "r"(mbar_addr));
}


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

__device__ __forceinline__ float2 fma_f32x2_rn_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}




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


__device__ __forceinline__ void tma_store_2d(
    const void *tmap, int x, int y, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2}], [%3];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(smem_addr) : "memory");
}



extern "C" {

__global__ __launch_bounds__(256, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_cake_nvfp4_svdquant_gemm_a234cd99c8b3ef6af762(const __grid_constant__ CUtensorMap D, float* __restrict__ alpha, __nv_bfloat16* __restrict__ bias, const __grid_constant__ CUtensorMap out, int M, int N, int K_tiles, int tok_tiles, int w_tiles, int num_tiles, int raster_group, int has_bias, int rank, const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, const __grid_constant__ CUtensorMap L1, const __grid_constant__ CUtensorMap L1_tail, __nv_bfloat16* __restrict__ out_ptr)
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
    #define acc_ovl_full_addr (mbar_base + 80)
    #define acc_full_addr (mbar_base + 88)
    #define acc_empty_addr (mbar_base + 104)
    #define klast_done_addr (mbar_base + 112)
    #define tmem_drained_addr (mbar_base + 120)
    #define tmem_dealloc_bar_addr (mbar_base + 128)
    #define clc_full_addr (mbar_base + 136)
    #define clc_empty_addr (mbar_base + 152)
    #define thr_full_addr (mbar_base + 168)
    #define thr_empty_addr (mbar_base + 184)

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
    uint8_t* smem_sfa = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int smem_sfa_addr = smem + 33792;
    uint8_t* smem_v3 = reinterpret_cast<uint8_t*>(smem_raw + 35840);
    const int smem_v3_addr = smem + 35840;
    uint8_t* smem_v4 = reinterpret_cast<uint8_t*>(smem_raw + 36864);
    const int smem_v4_addr = smem + 36864;
    uint8_t* smem_v5 = reinterpret_cast<uint8_t*>(smem_raw + 37888);
    const int smem_v5_addr = smem + 37888;
    uint8_t* smem_v6 = reinterpret_cast<uint8_t*>(smem_raw + 38912);
    const int smem_v6_addr = smem + 38912;
    uint8_t* smem_v7 = reinterpret_cast<uint8_t*>(smem_raw + 35840);
    const int smem_v7_addr = smem + 35840;
    uint8_t* smem_v8 = reinterpret_cast<uint8_t*>(smem_raw + 37888);
    const int smem_v8_addr = smem + 37888;
    uint8_t* smem_v9 = reinterpret_cast<uint8_t*>(smem_raw + 35840);
    const int smem_v9_addr = smem + 35840;
    uint8_t* smem_v10 = reinterpret_cast<uint8_t*>(smem_raw + 36352);
    const int smem_v10_addr = smem + 36352;
    uint8_t* smem_v11 = reinterpret_cast<uint8_t*>(smem_raw + 36864);
    const int smem_v11_addr = smem + 36864;
    uint8_t* smem_v12 = reinterpret_cast<uint8_t*>(smem_raw + 37376);
    const int smem_v12_addr = smem + 37376;
    uint8_t* smem_v13 = reinterpret_cast<uint8_t*>(smem_raw + 37888);
    const int smem_v13_addr = smem + 37888;
    uint8_t* smem_v14 = reinterpret_cast<uint8_t*>(smem_raw + 38400);
    const int smem_v14_addr = smem + 38400;
    uint8_t* smem_v15 = reinterpret_cast<uint8_t*>(smem_raw + 38912);
    const int smem_v15_addr = smem + 38912;
    uint8_t* smem_v16 = reinterpret_cast<uint8_t*>(smem_raw + 39424);
    const int smem_v16_addr = smem + 39424;
    __nv_bfloat16* smem_d = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_d_addr = smem + 1024;
    __nv_bfloat16* smem_l1_pre = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int smem_l1_pre_addr = smem + 17408;
    __nv_bfloat16* smem_l1_tail = reinterpret_cast<__nv_bfloat16*>(smem_raw + 29696);
    const int smem_l1_tail_addr = smem + 29696;
    __nv_bfloat16* smem_out = reinterpret_cast<__nv_bfloat16*>(smem_raw + 195584);
    const int smem_out_addr = smem + 195584;
    unsigned int* smem_bias_pairs = reinterpret_cast<unsigned int*>(smem_raw + 228352);
    const int smem_bias_pairs_addr = smem + 228352;
    unsigned int* clc_resp = reinterpret_cast<unsigned int*>(smem_raw + 228864);
    const int clc_resp_addr = smem + 228864;
    if (warp == 2) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&A))) : "memory"); }
    if (warp == 2) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&B))) : "memory"); }
    if (warp == 2) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SFA))) : "memory"); }
    if (warp == 2) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&SFB))) : "memory"); }
    if (warp == 2) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&D))) : "memory"); }
    if (warp == 2) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&L1))) : "memory"); }
    if (warp == 2) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&L1_tail))) : "memory"); }
    if (warp == 2) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&out))) : "memory"); }

    // Mbarrier init (12 pipeline groups, 0 ordered-sequence groups, 25 barriers)
    // Mbarriers at smem_raw[0..200)

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
            // acc_ovl_full: 1 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            // --- pipeline 'accf_pipe' ---
            // acc_full: 2 barriers, init_count=1
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            // --- pipeline 'acc_pipe' ---
            // acc_empty: 1 barriers, init_count=256
            mbarrier_init(smem + 104, 256);
            // klast_done: 1 barriers, init_count=1
            mbarrier_init(smem + 112, 1);
            // tmem_drained: 1 barriers, init_count=256
            mbarrier_init(smem + 120, 256);
            // tmem_dealloc_bar: 1 barriers, init_count=32
            mbarrier_init(smem + 128, 32);
            // --- pipeline 'clc_pipe' ---
            // clc_full: 2 barriers, init_count=1
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            // clc_empty: 2 barriers, init_count=416
            mbarrier_init(smem + 152, 416);
            mbarrier_init(smem + 160, 416);
            // --- pipeline 'thr_pipe' ---
            // thr_full: 2 barriers, init_count=32
            mbarrier_init(smem + 168, 32);
            mbarrier_init(smem + 176, 32);
            // thr_empty: 2 barriers, init_count=32
            mbarrier_init(smem + 184, 32);
            mbarrier_init(smem + 192, 32);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 496 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 200);
    if (warp == 2) {
        int _tmem_hold = smem + 200;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    if (warp == 1 || warp == 2 || warp == 4 || warp == 5 || warp == 6 || warp == 7) {
        asm volatile("barrier.sync.aligned %0, %1;" :: "r"(2), "r"(192) : "memory");
    }
    const int taddr = (warp == 1 || warp == 2 || warp == 4 || warp == 5 || warp == 6 || warp == 7) ? tmem_addr_storage[0] : 0;

    // Kernel post-init ops
    const int tmem_accum = taddr;
    const int tmem_tmem_sfa = taddr + 448;
    const int tmem_tmem_sfb = taddr + 464;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            unsigned int load_stage = 0;
            unsigned int load_clc_stage = 0;
            unsigned int load_thr_stage = 0;
            unsigned int load_valid = 1;
            unsigned int load_next = 0;
            asm volatile("griddepcontrol.wait;" ::: "memory");
            int rank_chunks = 1;
            unsigned int load_lead = 0;
            if (K_tiles > 5) {
                load_lead = (unsigned int)(K_tiles - 4 - 1);
            }
            unsigned int tile_l = cluster_id;
            unsigned int _phase_thr_empty = 1;
            unsigned int _phase_tma_empty = 1;
            unsigned int _phase_clc_full = 0;
            #pragma unroll 1
            for (unsigned int load_tiles = 0; load_tiles < num_tiles; load_tiles++) {
                if (cta_rank == 0) {
                    mbarrier_wait(thr_empty_addr + (load_thr_stage) * 8, _phase_thr_empty);
                    mbarrier_arrive(thr_full_addr + (load_thr_stage) * 8);
                    load_thr_stage += 1;
                    if (load_thr_stage == 2) { load_thr_stage = 0; _phase_thr_empty ^= 1; }
                }
                int w_idx = tile_l / (unsigned int)tok_tiles;
                int tok_idx = tile_l - (unsigned int)(w_idx * tok_tiles);
                if (raster_group != 0) {
                    int _min_0 = ((raster_group) < (tok_tiles - (int)(tile_l / (unsigned int)(raster_group * w_tiles)) * raster_group) ? (raster_group) : (tok_tiles - (int)(tile_l / (unsigned int)(raster_group * w_tiles)) * raster_group));
                    w_idx = (tile_l - tile_l / (unsigned int)(raster_group * w_tiles) * (unsigned int)(raster_group * w_tiles)) / (unsigned int)_min_0;
                    int _min_0_1 = ((raster_group) < (tok_tiles - (int)(tile_l / (unsigned int)(raster_group * w_tiles)) * raster_group) ? (raster_group) : (tok_tiles - (int)(tile_l / (unsigned int)(raster_group * w_tiles)) * raster_group));
                    tok_idx = tile_l / (unsigned int)(raster_group * w_tiles) * (unsigned int)raster_group + (tile_l - tile_l / (unsigned int)(raster_group * w_tiles) * (unsigned int)(raster_group * w_tiles)) - (tile_l - tile_l / (unsigned int)(raster_group * w_tiles) * (unsigned int)(raster_group * w_tiles)) / (unsigned int)_min_0_1 * (unsigned int)_min_0_1;
                }
                int a_rows = tok_idx * 256 + cta_rank * 128;
                int a_atom = a_rows / 128;
                int b_atom = w_idx * 256 / 128;
                int b_rows = w_idx * 256 + cta_rank * 128;
                int buf_l = (int)(load_tiles & 1);
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
                        tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 38912, (&A), 0, a_rows, k_tile, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_sfa_addr + load_stage * 38912, (&SFA), 0, 4 * k_tile, a_atom, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 38912, (&B), 0, b_rows, k_tile, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                    }
                    if (elect_sync()) {
                        if (cta_rank == 0) {
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(smem_v7_addr + load_stage * 38912), "l"((&SFB)), "r"(0), "r"(4 * k_tile), "r"(b_atom),
                                   "r"(((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)) : "memory");
                        } else {
                            asm volatile(
                                "cp.async.bulk.tensor.3d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                :: "r"(smem_v8_addr + load_stage * 38912), "l"((&SFB)), "r"(0), "r"(4 * k_tile), "r"(b_atom + 1),
                                   "r"(((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF)), "h"((uint16_t)(3)) : "memory");
                        }
                    }
                    load_stage += 1;
                    if (load_stage == 5) { load_stage = 0; _phase_tma_empty ^= 1; }
                    if (k_tile == load_lead) {
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
                        load_valid = _clc_valid_1;
                        load_next = _clc_ctaid_0;
                    }
                }
                int w_idx_0 = tile_l / (unsigned int)tok_tiles;
                int tok_idx_1 = tile_l - (unsigned int)(w_idx_0 * tok_tiles);
                if (raster_group != 0) {
                    int _min_1 = ((raster_group) < (tok_tiles - (int)(tile_l / (unsigned int)(raster_group * w_tiles)) * raster_group) ? (raster_group) : (tok_tiles - (int)(tile_l / (unsigned int)(raster_group * w_tiles)) * raster_group));
                    w_idx_0 = (tile_l - tile_l / (unsigned int)(raster_group * w_tiles) * (unsigned int)(raster_group * w_tiles)) / (unsigned int)_min_1;
                    int _min_1_1 = ((raster_group) < (tok_tiles - (int)(tile_l / (unsigned int)(raster_group * w_tiles)) * raster_group) ? (raster_group) : (tok_tiles - (int)(tile_l / (unsigned int)(raster_group * w_tiles)) * raster_group));
                    tok_idx_1 = tile_l / (unsigned int)(raster_group * w_tiles) * (unsigned int)raster_group + (tile_l - tile_l / (unsigned int)(raster_group * w_tiles) * (unsigned int)(raster_group * w_tiles)) - (tile_l - tile_l / (unsigned int)(raster_group * w_tiles) * (unsigned int)(raster_group * w_tiles)) / (unsigned int)_min_1_1 * (unsigned int)_min_1_1;
                }
                int d_rows = tok_idx_1 * 256 + cta_rank * 128;
                int lb_pre = w_idx_0 * 256 + buf_l * 64 + cta_rank * 96;
                int lb_tail = w_idx_0 * 256 + (1 - buf_l) * 192 + cta_rank * 32;
                #pragma unroll 1
                for (unsigned int rank_chunk = 0; rank_chunk < rank_chunks; rank_chunk++) {
                    int rank_offset = (int)rank_chunk * 64;
                    mbarrier_wait(tma_empty_addr + (load_stage) * 8, _phase_tma_empty);
                    if (cta_rank == 0) {
                        if (elect_sync()) {
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(65536)) : "memory");
                        }
                    }
                    if (elect_sync()) {
                        tma_2d_gmem2smem_cta2(smem_d_addr + load_stage * 38912, (&D), rank_offset, d_rows, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_2d_gmem2smem_cta2(smem_l1_pre_addr + load_stage * 38912, (&L1), rank_offset, lb_pre, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_2d_gmem2smem_cta2(smem_l1_tail_addr + load_stage * 38912, (&L1_tail), rank_offset, lb_tail, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                    }
                    load_stage += 1;
                    if (load_stage == 5) { load_stage = 0; _phase_tma_empty ^= 1; }
                }
                if (load_valid == 0) {
                    break;
                }
                tile_l = load_next / 2;
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            unsigned int mma_stage = 0;
            unsigned int accf_stage = 0;
            unsigned int completed = 0;
            unsigned int mma_clc_stage = 0;
            unsigned int mma_valid = 1;
            unsigned int mma_next = 0;
            unsigned int held_stage = 0;
            unsigned int last_stage = 0;
            asm volatile("griddepcontrol.wait;" ::: "memory");
            int rank_chunks_1 = 1;
            unsigned int last_kb = (unsigned int)(K_tiles - 1);
            unsigned int mma_lead = 0;
            if (K_tiles > 5) {
                mma_lead = (unsigned int)(K_tiles - 4 - 1);
            }
            unsigned int tile_m = cluster_id;
            unsigned int _phase_tma_full = 0;
            unsigned int _phase_acc_empty = 1;
            unsigned int _phase_clc_full_1 = 0;
            #pragma unroll 1
            for (unsigned int _p = 0; _p < num_tiles; _p++) {
                int buf = (int)(completed & 1);
                int acc_base = buf * 192;
                #pragma unroll 1
                for (unsigned int k_tile_1 = 0; k_tile_1 < K_tiles; k_tile_1++) {
                    int init_flag = ((k_tile_1 == 0) ? 1 : 0);
                    if (cta_rank == 0) {
                        mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa, make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_stage) * 2432)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfa + 4), make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_stage) * 2432 + 32)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfa + 8), make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_stage) * 2432 + 64)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfa + 12), make_sf_cp_desc_lo_sbo128((((smem_sfa_addr) >> 4) + (mma_stage) * 2432 + 96)));
                        }
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb, make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (mma_stage) * 2432)));
                        }
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + 4, make_sf_cp_desc_lo_sbo128((((smem_v13_addr) >> 4) + (mma_stage) * 2432)));
                        }
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + 8, make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (mma_stage) * 2432)));
                        }
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + 12, make_sf_cp_desc_lo_sbo128((((smem_v14_addr) >> 4) + (mma_stage) * 2432)));
                        }
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + 16, make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (mma_stage) * 2432)));
                        }
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + 20, make_sf_cp_desc_lo_sbo128((((smem_v15_addr) >> 4) + (mma_stage) * 2432)));
                        }
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + 24, make_sf_cp_desc_lo_sbo128((((smem_v12_addr) >> 4) + (mma_stage) * 2432)));
                        }
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + 28, make_sf_cp_desc_lo_sbo128((((smem_v16_addr) >> 4) + (mma_stage) * 2432)));
                        }
                        if (k_tile_1 == 0) {
                            mbarrier_wait(acc_empty_addr, _phase_acc_empty);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                        }
                        int _smem_desc_lo_0 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage) * 2432;
                        int _smem_desc_lo_1 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_stage) * 2432;
                        if (elect_sync()) {
                            {
                                int _smem_desc_lo_2 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage) * 2432;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_0 | ((uint64_t)((uint32_t)_smem_desc_lo_2 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_3 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_stage) * 2432;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_1 | ((uint64_t)((uint32_t)_smem_desc_lo_3 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2((tmem_accum + (acc_base)), a_desc + 0, b_desc + 0,
                                    0x90400480U, tmem_tmem_sfa + 0, tmem_tmem_sfb + 0, ((init_flag) ? 0 : 1));
                            }
                        }
                        int _smem_desc_lo_4 = (((smem_a_addr + 48) >> 4) & 0x3FFF) + (mma_stage) * 2432;
                        int _smem_desc_lo_5 = (((smem_b_addr + 48) >> 4) & 0x3FFF) + (mma_stage) * 2432;
                        if (elect_sync()) {
                            {
                                int _smem_desc_lo_6 = (((smem_a_addr) >> 4) & 0x3FFF) + (mma_stage) * 2432;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_4 | ((uint64_t)((uint32_t)_smem_desc_lo_6 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_7 = (((smem_b_addr) >> 4) & 0x3FFF) + (mma_stage) * 2432;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_5 | ((uint64_t)((uint32_t)_smem_desc_lo_7 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2((tmem_accum + (acc_base)), a_desc + 0, b_desc + 0,
                                    0xd04004a0U, tmem_tmem_sfa + 4 + 0, tmem_tmem_sfb + 8 + 0, 1);
                            }
                        }
                        int _panel_a_lo = (((smem_a_addr + 96) >> 4) & 0x3FFF) + (mma_stage) * 2432;
                        int _panel_b_lo = (((smem_b_addr + 96) >> 4) & 0x3FFF) + (mma_stage) * 2432;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_panel_a_lo) | ((uint64_t)0x40004040 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_panel_b_lo) | ((uint64_t)0x40004040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_cta2((tmem_accum + (acc_base)), a_desc + 0, b_desc + 0,
                                    0x10400480U, tmem_tmem_sfa + 12 + 0, tmem_tmem_sfb + 24 + 0, 1);
                            }
                        }
                        elect_commit_cg2_multicast(tma_empty_addr + (mma_stage) * 8, (uint16_t)(3));
                    }
                    mma_stage += 1;
                    if (mma_stage == 5) { mma_stage = 0; _phase_tma_full ^= 1; }
                    if (k_tile_1 == mma_lead) {
                        mbarrier_wait(clc_full_addr + (mma_clc_stage) * 8, _phase_clc_full_1);
                        asm volatile("tcgen05.fence::after_thread_sync;");
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
                        mma_valid = _clc_valid_2;
                        mma_next = _clc_ctaid_1;
                    }
                }
                last_stage = mma_stage;
                if (cta_rank == 0) {
                    elect_commit_cg2_local(klast_done_addr);
                    int _parity_klast_done = (int)((unsigned int)(int)completed % 2);
                    mbarrier_wait(klast_done_addr, _parity_klast_done);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                }
                #pragma unroll 1
                for (unsigned int rank_chunk_1 = 0; rank_chunk_1 < rank_chunks_1; rank_chunk_1++) {
                    if (cta_rank == 0) {
                        mbarrier_wait(tma_full_addr + (mma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int init_c = 0;
                        int rank_cols_left = 32 - (int)rank_chunk_1 * 64;
                        if (rank_cols_left == 32) {
                            int _mma_a_lo_4 = (((smem_d_addr) >> 4) & 0x3FFF) + (mma_stage) * 2432;
                            int _mma_b_lo_4 = (((smem_l1_tail_addr) >> 4) & 0x3FFF) + (mma_stage) * 2432;
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
                    "}\n"
                    :: "r"(_mma_a_lo_4), "r"(_mma_b_lo_4), "r"((tmem_accum + (192))), "r"(((init_c) ? 0 : 1)) : "memory");
                        } else {
                            int _mma_a_lo_5 = (((smem_d_addr) >> 4) & 0x3FFF) + (mma_stage) * 2432;
                            int _mma_b_lo_5 = (((smem_l1_tail_addr) >> 4) & 0x3FFF) + (mma_stage) * 2432;
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
                    "}\n"
                    :: "r"(_mma_a_lo_5), "r"(_mma_b_lo_5), "r"((tmem_accum + (192))), "r"(((init_c) ? 0 : 1)) : "memory");
                        }
                    }
                    mma_stage += 1;
                    if (mma_stage == 5) { mma_stage = 0; _phase_tma_full ^= 1; }
                }
                if (cta_rank == 0) {
                    elect_commit_cg2_multicast(acc_ovl_full_addr, (uint16_t)(3));
                    #pragma unroll 1
                    for (unsigned int rank_chunk_2 = 0; rank_chunk_2 < rank_chunks_1; rank_chunk_2++) {
                        unsigned int hs = last_stage + rank_chunk_2;
                        if (hs >= 5) {
                            hs = hs - 5;
                        }
                        int init_c_1 = 0;
                        int rank_cols_left_1 = 32 - (int)rank_chunk_2 * 64;
                        if (rank_cols_left_1 == 32) {
                            int _mma_a_lo_6 = (((smem_d_addr) >> 4) & 0x3FFF) + (hs) * 2432;
                            int _mma_b_lo_6 = (((smem_l1_pre_addr) >> 4) & 0x3FFF) + (hs) * 2432;
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
                    "mov.b32 id, 271582352;\n\t"
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
                    :: "r"(_mma_a_lo_6), "r"(_mma_b_lo_6), "r"((tmem_accum + (buf * 256))), "r"(((init_c_1) ? 0 : 1)) : "memory");
                        } else {
                            int _mma_a_lo_7 = (((smem_d_addr) >> 4) & 0x3FFF) + (hs) * 2432;
                            int _mma_b_lo_7 = (((smem_l1_pre_addr) >> 4) & 0x3FFF) + (hs) * 2432;
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
                    "mov.b32 id, 271582352;\n\t"
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
                    "}\n"
                    :: "r"(_mma_a_lo_7), "r"(_mma_b_lo_7), "r"((tmem_accum + (buf * 256))), "r"(((init_c_1) ? 0 : 1)) : "memory");
                        }
                    }
                    elect_commit_cg2_multicast(acc_full_addr + (accf_stage) * 8, (uint16_t)(3));
                    #pragma unroll 1
                    for (unsigned int rank_chunk_3 = 0; rank_chunk_3 < rank_chunks_1; rank_chunk_3++) {
                        unsigned int hs_1 = last_stage + rank_chunk_3;
                        if (hs_1 >= 5) {
                            hs_1 = hs_1 - 5;
                        }
                        elect_commit_cg2_multicast(tma_empty_addr + (((mma_stage + (4 + rank_chunk_3) >= 5) ? mma_stage + (4 + rank_chunk_3) - 5 : mma_stage + (4 + rank_chunk_3))) * 8, (uint16_t)(3));
                    }
                }
                _phase_acc_empty ^= 1;
                accf_stage += 1;
                if (accf_stage == 2) { accf_stage = 0; }
                completed += 1;
                if (mma_valid == 0) {
                    break;
                }
                tile_m = mma_next / 2;
            }
            if (cta_rank == 0) {
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
                mbarrier_wait(tmem_drained_addr, 0);
            }
            int dealloc_peer_rank = cta_rank ^ 1;
            if (cta_rank != 0) {
                asm volatile("tcgen05.fence::before_thread_sync;");
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
                asm volatile("tcgen05.fence::before_thread_sync;");
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
                for (unsigned int _p_1 = 0; _p_1 < num_tiles; _p_1++) {
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
                        :: "r"(clc_empty_addr + (sched_clc_stage) * 8), "r"(0) : "memory");
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
            int epi_tid = epi_warp * 32 + lane;
            unsigned int accf_stage_1 = 0;
            unsigned int store_stage = 0;
            asm volatile("griddepcontrol.wait;" ::: "memory");
            float alpha_value = alpha[0];
            int out_hinted = 0;
            if (K_tiles >= 24) {
                out_hinted = 1;
            }
            if (w_tiles >= 24) {
                out_hinted = 1;
            }
            float2 _f2_0 = make_float2(alpha_value, alpha_value);
            int lane_row_addr = taddr + (unsigned int)(epi_warp * 32 << 16);
            unsigned int epi_clc_stage = 0;
            unsigned int tile_e = cluster_id;
            unsigned int _phase_clc_full_3 = 0;
            unsigned int _phase_acc_ovl_full = 0;
            unsigned int _phase_acc_full = 0;
            #pragma unroll 1
            for (unsigned int tiles_done = 0; tiles_done < num_tiles; tiles_done++) {
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
                    :: "r"(clc_empty_addr + (epi_clc_stage) * 8), "r"(0) : "memory");
                epi_clc_stage += 1;
                if (epi_clc_stage == 2) { epi_clc_stage = 0; _phase_clc_full_3 ^= 1; }
                int w_idx_1 = tile_e / (unsigned int)tok_tiles;
                int tok_idx_2 = tile_e - (unsigned int)(w_idx_1 * tok_tiles);
                if (raster_group != 0) {
                    int _min_0_2 = ((raster_group) < (tok_tiles - (int)(tile_e / (unsigned int)(raster_group * w_tiles)) * raster_group) ? (raster_group) : (tok_tiles - (int)(tile_e / (unsigned int)(raster_group * w_tiles)) * raster_group));
                    w_idx_1 = (tile_e - tile_e / (unsigned int)(raster_group * w_tiles) * (unsigned int)(raster_group * w_tiles)) / (unsigned int)_min_0_2;
                    int _min_2 = ((raster_group) < (tok_tiles - (int)(tile_e / (unsigned int)(raster_group * w_tiles)) * raster_group) ? (raster_group) : (tok_tiles - (int)(tile_e / (unsigned int)(raster_group * w_tiles)) * raster_group));
                    tok_idx_2 = tile_e / (unsigned int)(raster_group * w_tiles) * (unsigned int)raster_group + (tile_e - tile_e / (unsigned int)(raster_group * w_tiles) * (unsigned int)(raster_group * w_tiles)) - (tile_e - tile_e / (unsigned int)(raster_group * w_tiles) * (unsigned int)(raster_group * w_tiles)) / (unsigned int)_min_2 * (unsigned int)_min_2;
                }
                int off_tok = tok_idx_2 * 256 + cta_rank * 128;
                int off_w = w_idx_1 * 256;
                int buf_1 = (int)(tiles_done & 1);
                int acc_base_1 = buf_1 * 192;
                if (epi_tid < 128) {
                    unsigned int staged_bias_pair = 0;
                    int bias_col = off_w + epi_tid * 2;
                    if (has_bias != 0) {
                        if (bias_col < N) {
                            unsigned int _vec_load_0[2];
                            {
                                uint32_t _bf16x2_bits_1;
                                _bf16x2_bits_1 = *reinterpret_cast<const uint32_t*>(bias + bias_col);
                                _vec_load_0[0] = _bf16x2_bits_1;
                            }
                            staged_bias_pair = _vec_load_0[0];
                        }
                    }
                    smem_bias_pairs[epi_tid] = staged_bias_pair;
                }
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                asm volatile("tcgen05.fence::after_thread_sync;");
                int prev_sub = 0;
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    int sub = 3 - j;
                    if (buf_1 != 0) {
                        sub = j;
                    }
                    if (j == 0) {
                        mbarrier_wait(acc_ovl_full_addr, _phase_acc_ovl_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                    }
                    if (j == 1) {
                        mbarrier_wait(acc_full_addr + (accf_stage_1) * 8, _phase_acc_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                    }
                    int tmem_addr = lane_row_addr + acc_base_1 + sub * 64;
                    float _tmem_load_0[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x64.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, [%64];"
                        : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31]), "=f"(_tmem_load_0[32]), "=f"(_tmem_load_0[33]), "=f"(_tmem_load_0[34]), "=f"(_tmem_load_0[35]), "=f"(_tmem_load_0[36]), "=f"(_tmem_load_0[37]), "=f"(_tmem_load_0[38]), "=f"(_tmem_load_0[39]), "=f"(_tmem_load_0[40]), "=f"(_tmem_load_0[41]), "=f"(_tmem_load_0[42]), "=f"(_tmem_load_0[43]), "=f"(_tmem_load_0[44]), "=f"(_tmem_load_0[45]), "=f"(_tmem_load_0[46]), "=f"(_tmem_load_0[47]), "=f"(_tmem_load_0[48]), "=f"(_tmem_load_0[49]), "=f"(_tmem_load_0[50]), "=f"(_tmem_load_0[51]), "=f"(_tmem_load_0[52]), "=f"(_tmem_load_0[53]), "=f"(_tmem_load_0[54]), "=f"(_tmem_load_0[55]), "=f"(_tmem_load_0[56]), "=f"(_tmem_load_0[57]), "=f"(_tmem_load_0[58]), "=f"(_tmem_load_0[59]), "=f"(_tmem_load_0[60]), "=f"(_tmem_load_0[61]), "=f"(_tmem_load_0[62]), "=f"(_tmem_load_0[63])
                        : "r"(tmem_addr));
                    unsigned int bias_pairs[32];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[0])), "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[(0) + 3]))
                        : "r"(smem_bias_pairs_addr + (unsigned int)(sub * 32 * 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[4])), "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[(4) + 3]))
                        : "r"(smem_bias_pairs_addr + (unsigned int)((sub * 32 + 4) * 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[8])), "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[(8) + 3]))
                        : "r"(smem_bias_pairs_addr + (unsigned int)((sub * 32 + 8) * 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[12])), "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[(12) + 3]))
                        : "r"(smem_bias_pairs_addr + (unsigned int)((sub * 32 + 12) * 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[16])), "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[(16) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[(16) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[(16) + 3]))
                        : "r"(smem_bias_pairs_addr + (unsigned int)((sub * 32 + 16) * 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[20])), "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[(20) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[(20) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[(20) + 3]))
                        : "r"(smem_bias_pairs_addr + (unsigned int)((sub * 32 + 20) * 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[24])), "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[(24) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[(24) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[(24) + 3]))
                        : "r"(smem_bias_pairs_addr + (unsigned int)((sub * 32 + 24) * 4)));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[28])), "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[(28) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[(28) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&bias_pairs[(28) + 3]))
                        : "r"(smem_bias_pairs_addr + (unsigned int)((sub * 32 + 28) * 4)));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    if (j == 0) {
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((acc_empty_addr) & 0xFEFFFFFF) : "memory");
                    }
                    float result[64];
                    unsigned int bias_pair = 0;
                    unsigned int bias_lo_bits = 0;
                    float bias_lo = 0.0f;
                    unsigned int bias_hi_bits = 0;
                    float bias_hi = 0.0f;
                    #pragma unroll
                    for (int jj = 0; jj < 32; jj++) {
                        bias_pair = bias_pairs[jj];
                        bias_lo_bits = bias_pair << 16;
                        bias_lo = reinterpret_cast<float*>(&bias_lo_bits)[0];
                        bias_hi_bits = bias_pair & 4294901760u;
                        bias_hi = reinterpret_cast<float*>(&bias_hi_bits)[0];
                        float2 result_pair = fma_f32x2_rn_ftz(make_float2(_tmem_load_0[2 * jj], _tmem_load_0[2 * jj + 1]), _f2_0, make_float2(bias_lo, bias_hi));
                        result[2 * jj] = result_pair.x;
                        result[2 * jj + 1] = result_pair.y;
                    }
                    unsigned int result_bf16[32];
                    #pragma unroll
                    for (int _lp = 0; _lp < 32; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(result[_lp*2 + 0], result[_lp*2+1 + 0]));
                        result_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    int v8_row = off_tok + epi_tid;
                    int v8_col = off_w + sub * 64;
                    if (v8_row < M) {
                        if (v8_col < N) {
                            int v8_idx = v8_row * N + v8_col;
                            {
                                const unsigned* _raw_stv8_2 = reinterpret_cast<const unsigned*>(result_bf16 + 0);
                                asm volatile(
                                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                    :: "l"((void*)(out_ptr + v8_idx)), "r"(_raw_stv8_2[0]), "r"(_raw_stv8_2[1]), "r"(_raw_stv8_2[2]), "r"(_raw_stv8_2[3]), "r"(_raw_stv8_2[4]), "r"(_raw_stv8_2[5]), "r"(_raw_stv8_2[6]), "r"(_raw_stv8_2[7]) : "memory");
                            }
                            {
                                const unsigned* _raw_stv8_3 = reinterpret_cast<const unsigned*>(result_bf16 + 8);
                                asm volatile(
                                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                    :: "l"((void*)(out_ptr + (v8_idx + 16))), "r"(_raw_stv8_3[0]), "r"(_raw_stv8_3[1]), "r"(_raw_stv8_3[2]), "r"(_raw_stv8_3[3]), "r"(_raw_stv8_3[4]), "r"(_raw_stv8_3[5]), "r"(_raw_stv8_3[6]), "r"(_raw_stv8_3[7]) : "memory");
                            }
                            {
                                const unsigned* _raw_stv8_4 = reinterpret_cast<const unsigned*>(result_bf16 + 16);
                                asm volatile(
                                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                    :: "l"((void*)(out_ptr + (v8_idx + 32))), "r"(_raw_stv8_4[0]), "r"(_raw_stv8_4[1]), "r"(_raw_stv8_4[2]), "r"(_raw_stv8_4[3]), "r"(_raw_stv8_4[4]), "r"(_raw_stv8_4[5]), "r"(_raw_stv8_4[6]), "r"(_raw_stv8_4[7]) : "memory");
                            }
                            {
                                const unsigned* _raw_stv8_5 = reinterpret_cast<const unsigned*>(result_bf16 + 24);
                                asm volatile(
                                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                    :: "l"((void*)(out_ptr + (v8_idx + 48))), "r"(_raw_stv8_5[0]), "r"(_raw_stv8_5[1]), "r"(_raw_stv8_5[2]), "r"(_raw_stv8_5[3]), "r"(_raw_stv8_5[4]), "r"(_raw_stv8_5[5]), "r"(_raw_stv8_5[6]), "r"(_raw_stv8_5[7]) : "memory");
                            }
                        }
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    prev_sub = sub;
                    continue;
                    if (j > 0) {
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (warp == 4) {
                            if (elect_sync()) {
                                if (out_hinted != 0) {
                                    asm volatile(
                                        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group.L2::cache_hint"
                                        " [%0, {%1, %2}], [%3], %4;"
                                        :: "l"((&out)), "r"(off_w + prev_sub * 64), "r"(off_tok), "r"(smem_out_addr + store_stage * 16384), "l"(0x12F0000000000000ULL) : "memory");
                                } else {
                                    tma_store_2d((&out), off_w + prev_sub * 64, off_tok, smem_out_addr + store_stage * 16384);
                                }
                            }
                        }
                        if (warp == 4) {
                            asm volatile("cp.async.bulk.commit_group;");
                            asm volatile("cp.async.bulk.wait_group.read 1;");
                        }
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        store_stage = store_stage + 1;
                        if (store_stage == 2) {
                            store_stage = 0;
                        }
                    }
                    int out_stage_row = store_stage * 128 + (unsigned int)epi_tid;
                    unsigned int out_abs = smem_out_addr + (unsigned int)(out_stage_row * 128);
                    unsigned int out_swz = out_abs / 8 & 112;
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_out_addr + ((unsigned int)(out_stage_row * 128) + (0 ^ out_swz)))), "r"(result_bf16[0]), "r"(result_bf16[1]), "r"(result_bf16[2]), "r"(result_bf16[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_out_addr + ((unsigned int)(out_stage_row * 128) + (16 ^ out_swz)))), "r"(result_bf16[4]), "r"(result_bf16[5]), "r"(result_bf16[6]), "r"(result_bf16[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_out_addr + ((unsigned int)(out_stage_row * 128) + (32 ^ out_swz)))), "r"(result_bf16[8]), "r"(result_bf16[9]), "r"(result_bf16[10]), "r"(result_bf16[11]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_out_addr + ((unsigned int)(out_stage_row * 128) + (48 ^ out_swz)))), "r"(result_bf16[12]), "r"(result_bf16[13]), "r"(result_bf16[14]), "r"(result_bf16[15]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_out_addr + ((unsigned int)(out_stage_row * 128) + (64 ^ out_swz)))), "r"(result_bf16[16]), "r"(result_bf16[17]), "r"(result_bf16[18]), "r"(result_bf16[19]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_out_addr + ((unsigned int)(out_stage_row * 128) + (80 ^ out_swz)))), "r"(result_bf16[20]), "r"(result_bf16[21]), "r"(result_bf16[22]), "r"(result_bf16[23]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_out_addr + ((unsigned int)(out_stage_row * 128) + (96 ^ out_swz)))), "r"(result_bf16[24]), "r"(result_bf16[25]), "r"(result_bf16[26]), "r"(result_bf16[27]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_out_addr + ((unsigned int)(out_stage_row * 128) + (112 ^ out_swz)))), "r"(result_bf16[28]), "r"(result_bf16[29]), "r"(result_bf16[30]), "r"(result_bf16[31]) : "memory");
                    prev_sub = sub;
                }
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                _phase_acc_ovl_full ^= 1;
                accf_stage_1 += 1;
                if (accf_stage_1 == 2) { accf_stage_1 = 0; _phase_acc_full ^= 1; }
                if (_clc_valid_3 == 0) {
                    break;
                }
                tile_e = _clc_ctaid_2 / 2;
            }
            asm volatile(
                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                :: "r"((tmem_drained_addr) & 0xFEFFFFFF) : "memory");
            if (warp == 4) {
                asm volatile("cp.async.bulk.wait_group 0;");
            }
        }
    }

    // Cleanup
}

} // extern "C"
