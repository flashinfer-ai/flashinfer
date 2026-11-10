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
#define TMEM_NCOLS 400
#define TMEM_ACCUM_OFFSET 0
#define TMEM_TMEM_SFA_OFFSET 256
#define TMEM_TMEM_SFB_OFFSET 304
#define NUM_TMA_PIPE_STAGES 5
#define NUM_MAINLOOP_PIPE_STAGES 1
#define NUM_WORK_PIPE_STAGES 4
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 38912
#define SMEM_SMEM_B_OFF 17408
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 38912
#define SMEM_SMEM_SFA_ALL_OFF 33792
#define SMEM_SMEM_SFA_ALL_STAGE_BYTES 2048
#define SMEM_SMEM_SFA_ALL_STRIDE 38912
#define SMEM_SMEM_SFB_ALL_OFF 35840
#define SMEM_SMEM_SFB_ALL_STAGE_BYTES 4096
#define SMEM_SMEM_SFB_ALL_STRIDE 38912
#define SMEM_SMEM_V4_OFF 33792
#define SMEM_SMEM_V4_STAGE_BYTES 512
#define SMEM_SMEM_V4_STRIDE 38912
#define SMEM_SMEM_V5_OFF 34304
#define SMEM_SMEM_V5_STAGE_BYTES 512
#define SMEM_SMEM_V5_STRIDE 38912
#define SMEM_SMEM_V6_OFF 34816
#define SMEM_SMEM_V6_STAGE_BYTES 512
#define SMEM_SMEM_V6_STRIDE 38912
#define SMEM_SMEM_V7_OFF 35328
#define SMEM_SMEM_V7_STAGE_BYTES 512
#define SMEM_SMEM_V7_STRIDE 38912
#define SMEM_SMEM_V8_OFF 35840
#define SMEM_SMEM_V8_STAGE_BYTES 1024
#define SMEM_SMEM_V8_STRIDE 38912
#define SMEM_SMEM_V9_OFF 36864
#define SMEM_SMEM_V9_STAGE_BYTES 1024
#define SMEM_SMEM_V9_STRIDE 38912
#define SMEM_SMEM_V10_OFF 37888
#define SMEM_SMEM_V10_STAGE_BYTES 1024
#define SMEM_SMEM_V10_STRIDE 38912
#define SMEM_SMEM_V11_OFF 38912
#define SMEM_SMEM_V11_STAGE_BYTES 1024
#define SMEM_SMEM_V11_STRIDE 38912
#define SMEM_EPI_STAGING_OFF 195584
#define SMEM_EPI_STAGING_STAGE_BYTES 16384
#define SMEM_EPI_STAGING_STRIDE 16384
#define SMEM_WORK_RESPONSE_OFF 228352
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_EXCH_OFF 228416
#define SMEM_EXCH_STAGE_BYTES 3072
#define SMEM_EXCH_STRIDE 3072
#define SMEM_SMEM_V15_OFF 1024
#define SMEM_SMEM_V15_STAGE_BYTES 6144
#define SMEM_SMEM_V15_STRIDE 38912
#define SMEM_SMEM_V16_OFF 1072
#define SMEM_SMEM_V16_STAGE_BYTES 6144
#define SMEM_SMEM_V16_STRIDE 38912
#define SMEM_SMEM_V17_OFF 1120
#define SMEM_SMEM_V17_STAGE_BYTES 6144
#define SMEM_SMEM_V17_STRIDE 38912
#define SMEM_SMEM_V18_OFF 1040
#define SMEM_SMEM_V18_STAGE_BYTES 6144
#define SMEM_SMEM_V18_STRIDE 38912
#define SMEM_SMEM_V19_OFF 1088
#define SMEM_SMEM_V19_STAGE_BYTES 6144
#define SMEM_SMEM_V19_STRIDE 38912
#define SMEM_SMEM_V20_OFF 1136
#define SMEM_SMEM_V20_STAGE_BYTES 6144
#define SMEM_SMEM_V20_STRIDE 38912
#define SMEM_SMEM_V21_OFF 1056
#define SMEM_SMEM_V21_STAGE_BYTES 6144
#define SMEM_SMEM_V21_STRIDE 38912
#define SMEM_SMEM_V22_OFF 1104
#define SMEM_SMEM_V22_STAGE_BYTES 6144
#define SMEM_SMEM_V22_STRIDE 38912
#define SMEM_SMEM_V23_OFF 1024
#define SMEM_SMEM_V23_STAGE_BYTES 6144
#define SMEM_SMEM_V23_STRIDE 38912
#define SMEM_SMEM_V24_OFF 17408
#define SMEM_SMEM_V24_STAGE_BYTES 6144
#define SMEM_SMEM_V24_STRIDE 38912
#define SMEM_SMEM_V25_OFF 17456
#define SMEM_SMEM_V25_STAGE_BYTES 6144
#define SMEM_SMEM_V25_STRIDE 38912
#define SMEM_SMEM_V26_OFF 17504
#define SMEM_SMEM_V26_STAGE_BYTES 6144
#define SMEM_SMEM_V26_STRIDE 38912
#define SMEM_SMEM_V27_OFF 17424
#define SMEM_SMEM_V27_STAGE_BYTES 6144
#define SMEM_SMEM_V27_STRIDE 38912
#define SMEM_SMEM_V28_OFF 17472
#define SMEM_SMEM_V28_STAGE_BYTES 6144
#define SMEM_SMEM_V28_STRIDE 38912
#define SMEM_SMEM_V29_OFF 17520
#define SMEM_SMEM_V29_STAGE_BYTES 6144
#define SMEM_SMEM_V29_STRIDE 38912
#define SMEM_SMEM_V30_OFF 17440
#define SMEM_SMEM_V30_STAGE_BYTES 6144
#define SMEM_SMEM_V30_STRIDE 38912
#define SMEM_SMEM_V31_OFF 17488
#define SMEM_SMEM_V31_STAGE_BYTES 6144
#define SMEM_SMEM_V31_STRIDE 38912
#define SMEM_SMEM_V32_OFF 17408
#define SMEM_SMEM_V32_STAGE_BYTES 6144
#define SMEM_SMEM_V32_STRIDE 38912
#define SMEM_TOTAL 231552
#define BLOCK_M 128
#define BLOCK_N 256
#define B_HALF_N 128
#define BLOCK_K 256
#define CTA_GROUP 2
#define NUM_STAGES 5
#define GROUP_M 64
#define WORK_STAGES 4
#define WORK_CONSUMERS 1058
#define NUM_K_ITERS 21
#define N_TILES 84
#define SF_K_TILES 84
#define tiles_per_group (GROUP_M * N_TILES)
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


__device__ __forceinline__ void tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
    int sfa_taddr, int sfb_taddr, int enable_input_d, uint32_t leader) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader, p;\n\t"
        "setp.ne.b32 leader, %7, 0;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "@leader tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc), "r"(i_desc), "r"(sfa_taddr), "r"(sfb_taddr), "r"(enable_input_d), "r"(leader) : "memory");
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


__device__ __forceinline__ void tcgen05_cp_32x128b_warpx4_cta2_leader(
    int taddr, uint64_t s_desc, uint32_t leader) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "setp.ne.b32 leader, %2, 0;\n\t"
        "@leader tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;\n\t"
        "}\n"
        :: "r"(taddr), "l"(s_desc), "r"(leader) : "memory");
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



__device__ __forceinline__ void tma_store_4d(
    const void *tmap, int x, int y, int z, int w, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3, %4}], [%5];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(w), "r"(smem_addr) : "memory");
}




__device__ __forceinline__ void tmem_ld_x16(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x16.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7,"
        "  %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
        : "=f"(dst[0]),  "=f"(dst[1]),  "=f"(dst[2]),  "=f"(dst[3]),
          "=f"(dst[4]),  "=f"(dst[5]),  "=f"(dst[6]),  "=f"(dst[7]),
          "=f"(dst[8]),  "=f"(dst[9]),  "=f"(dst[10]), "=f"(dst[11]),
          "=f"(dst[12]), "=f"(dst[13]), "=f"(dst[14]), "=f"(dst[15])
        : "r"(tmem_addr));
}



__device__ __forceinline__ unsigned int __as_u32(float v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "f"(v));
    return u;
}
__device__ __forceinline__ unsigned int __as_u32(__nv_bfloat162 v) {
    return *reinterpret_cast<const unsigned int*>(&v);
}
__device__ __forceinline__ unsigned int __as_u32(unsigned int v) { return v; }
__device__ __forceinline__ unsigned int __as_u32(int v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "r"(v));
    return u;
}

extern "C" {

__global__ __launch_bounds__(640, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_cake_minimax_h3_nvfp4_pre_attention_9cc4be9080a461a19d27(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, float* __restrict__ alpha, __nv_bfloat16* __restrict__ q_norm_weight, __nv_bfloat16* __restrict__ k_norm_weight, __nv_bfloat16* __restrict__ rope_cos_sin, float* __restrict__ out_global_scale, uint8_t* __restrict__ out_q, uint8_t* __restrict__ out_sf, const __grid_constant__ CUtensorMap OUTQ, unsigned int* __restrict__ qkv_words, unsigned int* __restrict__ debug_q_words, unsigned int* __restrict__ debug_k_words, int write_debug, float eps, int M, int m_tiles, int HEADS_PER_DESTINATION, int ROWS_PER_DESTINATION, int SCALE_STRIDE)
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
    #define mma_done_addr (mbar_base + 40)
    #define mainloop_done_addr (mbar_base + 80)
    #define epilogue_done_addr (mbar_base + 88)
    #define work_full_addr (mbar_base + 96)
    #define work_empty_addr (mbar_base + 128)

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
    uint8_t* smem_sfa_all = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int smem_sfa_all_addr = smem + 33792;
    uint8_t* smem_sfb_all = reinterpret_cast<uint8_t*>(smem_raw + 35840);
    const int smem_sfb_all_addr = smem + 35840;
    uint8_t* smem_v4 = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int smem_v4_addr = smem + 33792;
    uint8_t* smem_v5 = reinterpret_cast<uint8_t*>(smem_raw + 34304);
    const int smem_v5_addr = smem + 34304;
    uint8_t* smem_v6 = reinterpret_cast<uint8_t*>(smem_raw + 34816);
    const int smem_v6_addr = smem + 34816;
    uint8_t* smem_v7 = reinterpret_cast<uint8_t*>(smem_raw + 35328);
    const int smem_v7_addr = smem + 35328;
    uint8_t* smem_v8 = reinterpret_cast<uint8_t*>(smem_raw + 35840);
    const int smem_v8_addr = smem + 35840;
    uint8_t* smem_v9 = reinterpret_cast<uint8_t*>(smem_raw + 36864);
    const int smem_v9_addr = smem + 36864;
    uint8_t* smem_v10 = reinterpret_cast<uint8_t*>(smem_raw + 37888);
    const int smem_v10_addr = smem + 37888;
    uint8_t* smem_v11 = reinterpret_cast<uint8_t*>(smem_raw + 38912);
    const int smem_v11_addr = smem + 38912;
    uint8_t* epi_staging = reinterpret_cast<uint8_t*>(smem_raw + 195584);
    const int epi_staging_addr = smem + 195584;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 228352);
    const int work_response_addr = smem + 228352;
    float* exch = reinterpret_cast<float*>(smem_raw + 228416);
    const int exch_addr = smem + 228416;
    uint8_t* smem_v15 = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_v15_addr = smem + 1024;
    uint8_t* smem_v16 = reinterpret_cast<uint8_t*>(smem_raw + 1072);
    const int smem_v16_addr = smem + 1072;
    uint8_t* smem_v17 = reinterpret_cast<uint8_t*>(smem_raw + 1120);
    const int smem_v17_addr = smem + 1120;
    uint8_t* smem_v18 = reinterpret_cast<uint8_t*>(smem_raw + 1040);
    const int smem_v18_addr = smem + 1040;
    uint8_t* smem_v19 = reinterpret_cast<uint8_t*>(smem_raw + 1088);
    const int smem_v19_addr = smem + 1088;
    uint8_t* smem_v20 = reinterpret_cast<uint8_t*>(smem_raw + 1136);
    const int smem_v20_addr = smem + 1136;
    uint8_t* smem_v21 = reinterpret_cast<uint8_t*>(smem_raw + 1056);
    const int smem_v21_addr = smem + 1056;
    uint8_t* smem_v22 = reinterpret_cast<uint8_t*>(smem_raw + 1104);
    const int smem_v22_addr = smem + 1104;
    uint8_t* smem_v23 = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_v23_addr = smem + 1024;
    uint8_t* smem_v24 = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int smem_v24_addr = smem + 17408;
    uint8_t* smem_v25 = reinterpret_cast<uint8_t*>(smem_raw + 17456);
    const int smem_v25_addr = smem + 17456;
    uint8_t* smem_v26 = reinterpret_cast<uint8_t*>(smem_raw + 17504);
    const int smem_v26_addr = smem + 17504;
    uint8_t* smem_v27 = reinterpret_cast<uint8_t*>(smem_raw + 17424);
    const int smem_v27_addr = smem + 17424;
    uint8_t* smem_v28 = reinterpret_cast<uint8_t*>(smem_raw + 17472);
    const int smem_v28_addr = smem + 17472;
    uint8_t* smem_v29 = reinterpret_cast<uint8_t*>(smem_raw + 17520);
    const int smem_v29_addr = smem + 17520;
    uint8_t* smem_v30 = reinterpret_cast<uint8_t*>(smem_raw + 17440);
    const int smem_v30_addr = smem + 17440;
    uint8_t* smem_v31 = reinterpret_cast<uint8_t*>(smem_raw + 17488);
    const int smem_v31_addr = smem + 17488;
    uint8_t* smem_v32 = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int smem_v32_addr = smem + 17408;

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 20 barriers)
    // Mbarriers at smem_raw[0..160)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full: 5 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            mbarrier_init(smem + 8, 2);
            mbarrier_init(smem + 16, 2);
            mbarrier_init(smem + 24, 2);
            mbarrier_init(smem + 32, 2);
            // mma_done: 5 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // --- pipeline 'mainloop_pipe' ---
            // mainloop_done: 1 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            // epilogue_done: 1 barriers, init_count=32
            mbarrier_init(smem + 88, 32);
            // --- pipeline 'work_pipe' ---
            // work_full: 4 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // work_empty: 4 barriers, init_count=1058
            mbarrier_init(smem + 128, 1058);
            mbarrier_init(smem + 136, 1058);
            mbarrier_init(smem + 144, 1058);
            mbarrier_init(smem + 152, 1058);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 400 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 160);
    if (warp == 0) {
        int _tmem_hold = smem + 160;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;
    const int tmem_tmem_sfa = taddr + 256;
    const int tmem_tmem_sfb = taddr + 304;

    // ---- Warpgroup: 0 ----
    if (warp >= 0 && warp <= 3) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 32;");
        // ---- Role: load ----
        if (warp == 0) {
            { // load_main
                unsigned int load_stage = 0;
                unsigned int work_stage = 0;
                int weight_row_base = cta_rank * B_HALF_N;
                unsigned int _phase_work_empty = 1;
                unsigned int _phase_mma_done = 1;
                unsigned int _phase_work_full = 0;
                if (elect_sync()) {
                    unsigned int this_bid = bid;
                    #pragma unroll 1
                    for (unsigned int _tile_iter = 0; _tile_iter < m_tiles / CTA_GROUP * N_TILES; _tile_iter++) {
                        if (cta_rank == 0) {
                            mbarrier_wait_cluster_hint(work_empty_addr + (work_stage) * 8, _phase_work_empty, 10000000);
                            asm volatile(
                                "{\n\t"
                                ".reg .b32 remAddr32;\n\t"
                                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                                "}"
                                :: "r"(work_full_addr + work_stage * 8), "r"(0), "r"((uint32_t)(16)) : "memory");
                            asm volatile(
                                "{\n\t"
                                ".reg .b32 remAddr32;\n\t"
                                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                                "}"
                                :: "r"(work_full_addr + work_stage * 8), "r"(1), "r"((uint32_t)(16)) : "memory");
                            asm volatile(
                                "fence.proxy.async.shared::cta;\n\t"
                                "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                    ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                                    " [%0], [%1];"
                                :: "r"(work_response_addr + work_stage * 16 + 0 * 16), "r"(work_full_addr + work_stage * 8)
                                : "memory");
                        }
                        int group = this_bid / (unsigned int)tiles_per_group;
                        int first_m = group * GROUP_M;
                        int remaining = m_tiles - first_m;
                        int group_size = ((remaining >= GROUP_M) ? GROUP_M : remaining);
                        int local = this_bid % (unsigned int)tiles_per_group;
                        int bid_m = first_m + local % group_size;
                        int bid_n = local / group_size;
                        int off_m = bid_m * BLOCK_M;
                        int off_n = bid_n * BLOCK_N;
                        int sfa_tile_row = bid_m * SF_K_TILES;
                        int sfb_tile_row = bid_n * SF_K_TILES;
                        #pragma unroll 1
                        for (int iter_k = 0; iter_k < NUM_K_ITERS; iter_k++) {
                            mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                            tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 38912, (&A), 0, off_m, iter_k, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 38912, (&B), 0, weight_row_base + off_n, iter_k, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            int k_set_base = iter_k * 4;
                            tma_3d_gmem2smem_cta2(smem_sfa_all_addr + load_stage * 38912, (&SFA), 0, 0, sfa_tile_row + k_set_base, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            tma_3d_gmem2smem_cta2(smem_sfb_all_addr + load_stage * 38912, (&SFB), 0, 0, sfb_tile_row + k_set_base, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(38912)) : "memory");
                            load_stage += 1;
                            if (load_stage == 5) { load_stage = 0; _phase_mma_done ^= 1; }
                        }
                        mbarrier_wait(work_full_addr + (work_stage) * 8, _phase_work_full);
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
                            : "r"(work_response_addr + work_stage * 16 + 0 * 16)
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
                        if (work_stage == 4) { work_stage = 0; _phase_work_empty ^= 1; _phase_work_full ^= 1; }
                        if (_clc_valid_0 == 0) {
                            break;
                        }
                        this_bid = _clc_ctaid_0 + (unsigned int)cta_rank;
                    }
                    #pragma unroll 1
                    for (int _ = 0; _ < 5; _++) {
                        mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                        load_stage += 1;
                        if (load_stage == 5) { load_stage = 0; _phase_mma_done ^= 1; }
                    }
                }
            }
        // ---- Role: mma ----
        } else if (warp == 1) {
            { // mma_main
                unsigned int mma_tma_stage = 0;
                unsigned int work_stage_1 = 0;
                unsigned int _phase_tma_full = 0;
                unsigned int _phase_epilogue_done = 1;
                unsigned int _phase_work_full_1 = 0;
                if (cta_rank == 0) {
                    #pragma unroll 1
                    for (unsigned int _tile_iter_1 = 0; _tile_iter_1 < m_tiles / CTA_GROUP * N_TILES; _tile_iter_1++) {
                        mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        uint32_t _elected_0 = elect_sync();
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa, make_sf_cp_desc_lo_sbo128((((smem_v4_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb, make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 4), make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_tma_stage) * 2432 + 32)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 4, make_sf_cp_desc_lo_sbo128((((smem_v5_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 8, make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 8 + 4), make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (mma_tma_stage) * 2432 + 32)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 8, make_sf_cp_desc_lo_sbo128((((smem_v6_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 16, make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 16 + 4), make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (mma_tma_stage) * 2432 + 32)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 12, make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 24, make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (mma_tma_stage) * 2432)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 24 + 4), make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (mma_tma_stage) * 2432 + 32)), _elected_0);
                        mbarrier_wait(epilogue_done_addr, _phase_epilogue_done);
                        #pragma unroll 1
                        for (int super_k = 0; super_k < 7; super_k++) {
                            mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            unsigned int stage_p0 = mma_tma_stage;
                            int init_flag = ((super_k == 0) ? 1 : 0);
                            if (elect_sync()) {
                                if (super_k != 0) {
                                    tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa, make_sf_cp_desc_lo_sbo128((((smem_v4_addr) >> 4) + (stage_p0) * 2432)));
                                    tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb, make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (stage_p0) * 2432)));
                                    tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 4), make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (stage_p0) * 2432 + 32)));
                                    tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + 4, make_sf_cp_desc_lo_sbo128((((smem_v5_addr) >> 4) + (stage_p0) * 2432)));
                                    tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + 8, make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (stage_p0) * 2432)));
                                    tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 8 + 4), make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (stage_p0) * 2432 + 32)));
                                    tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + 8, make_sf_cp_desc_lo_sbo128((((smem_v6_addr) >> 4) + (stage_p0) * 2432)));
                                    tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + 16, make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (stage_p0) * 2432)));
                                    tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 16 + 4), make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (stage_p0) * 2432 + 32)));
                                    tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + 12, make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (stage_p0) * 2432)));
                                    tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + 24, make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (stage_p0) * 2432)));
                                    tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 24 + 4), make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (stage_p0) * 2432 + 32)));
                                }
                                int _smem_desc_lo_1 = (((smem_v15_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                                int _smem_desc_lo_2 = (((smem_v24_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                                {
                                    int _smem_desc_lo_3 = (((smem_v23_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_1 | ((uint64_t)((uint32_t)_smem_desc_lo_3 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                    int _smem_desc_lo_4 = (((smem_v32_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_2 | ((uint64_t)((uint32_t)_smem_desc_lo_4 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2(tmem_accum, a_desc + 0, b_desc + 0,
                                        0x90400480U, tmem_tmem_sfa + 0, tmem_tmem_sfb + 0, ((init_flag) ? 0 : 1));
                                }
                                int _smem_desc_lo_5 = (((smem_v16_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                                int _smem_desc_lo_6 = (((smem_v25_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                                {
                                    int _smem_desc_lo_7 = (((smem_v23_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_5 | ((uint64_t)((uint32_t)_smem_desc_lo_7 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                    int _smem_desc_lo_8 = (((smem_v32_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_6 | ((uint64_t)((uint32_t)_smem_desc_lo_8 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2(tmem_accum, a_desc + 0, b_desc + 0,
                                        0xd04004a0U, tmem_tmem_sfa + 4 + 0, tmem_tmem_sfb + 8 + 0, 1);
                                }
                            }
                            mma_tma_stage += 1;
                            if (mma_tma_stage == 5) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                            mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            unsigned int stage_p1 = mma_tma_stage;
                            uint32_t _elected_9 = elect_sync();
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 16, make_sf_cp_desc_lo_sbo128((((smem_v4_addr) >> 4) + (stage_p1) * 2432)), _elected_9);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 32, make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (stage_p1) * 2432)), _elected_9);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 32 + 4), make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (stage_p1) * 2432 + 32)), _elected_9);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 20, make_sf_cp_desc_lo_sbo128((((smem_v5_addr) >> 4) + (stage_p1) * 2432)), _elected_9);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 40, make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (stage_p1) * 2432)), _elected_9);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 40 + 4), make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (stage_p1) * 2432 + 32)), _elected_9);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 24, make_sf_cp_desc_lo_sbo128((((smem_v6_addr) >> 4) + (stage_p1) * 2432)), _elected_9);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 48, make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (stage_p1) * 2432)), _elected_9);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 48 + 4), make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (stage_p1) * 2432 + 32)), _elected_9);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 28, make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (stage_p1) * 2432)), _elected_9);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 56, make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (stage_p1) * 2432)), _elected_9);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 56 + 4), make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (stage_p1) * 2432 + 32)), _elected_9);
                            int _smem_desc_lo_10 = (((smem_v17_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                            int _smem_desc_lo_11 = (((smem_v26_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                            {
                                int _smem_desc_lo_12 = (((smem_v23_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_10 | ((uint64_t)((uint32_t)_smem_desc_lo_12 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_13 = (((smem_v32_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_11 | ((uint64_t)((uint32_t)_smem_desc_lo_13 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    0x90400480U, tmem_tmem_sfa + 12 + 0, tmem_tmem_sfb + 24 + 0, 1, _elected_9);
                            }
                            elect_commit_cg2_multicast(mma_done_addr + (stage_p0) * 8, (uint16_t)(3));
                            uint32_t _elected_14 = elect_sync();
                            int _smem_desc_lo_15 = (((smem_v18_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                            int _smem_desc_lo_16 = (((smem_v27_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                            {
                                int _smem_desc_lo_17 = (((smem_v23_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_15 | ((uint64_t)((uint32_t)_smem_desc_lo_17 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_18 = (((smem_v32_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_16 | ((uint64_t)((uint32_t)_smem_desc_lo_18 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    0xd04004a0U, tmem_tmem_sfa + 16 + 0, tmem_tmem_sfb + 32 + 0, 1, _elected_14);
                            }
                            int _smem_desc_lo_19 = (((smem_v19_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                            int _smem_desc_lo_20 = (((smem_v28_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                            {
                                int _smem_desc_lo_21 = (((smem_v23_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_19 | ((uint64_t)((uint32_t)_smem_desc_lo_21 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_22 = (((smem_v32_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_20 | ((uint64_t)((uint32_t)_smem_desc_lo_22 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    0x90400480U, tmem_tmem_sfa + 24 + 0, tmem_tmem_sfb + 48 + 0, 1, _elected_14);
                            }
                            mma_tma_stage += 1;
                            if (mma_tma_stage == 5) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                            mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            unsigned int stage_p2 = mma_tma_stage;
                            uint32_t _elected_23 = elect_sync();
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 32, make_sf_cp_desc_lo_sbo128((((smem_v4_addr) >> 4) + (stage_p2) * 2432)), _elected_23);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 64, make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (stage_p2) * 2432)), _elected_23);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 64 + 4), make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (stage_p2) * 2432 + 32)), _elected_23);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 36, make_sf_cp_desc_lo_sbo128((((smem_v5_addr) >> 4) + (stage_p2) * 2432)), _elected_23);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 72, make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (stage_p2) * 2432)), _elected_23);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 72 + 4), make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (stage_p2) * 2432 + 32)), _elected_23);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 40, make_sf_cp_desc_lo_sbo128((((smem_v6_addr) >> 4) + (stage_p2) * 2432)), _elected_23);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 80, make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (stage_p2) * 2432)), _elected_23);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 80 + 4), make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (stage_p2) * 2432 + 32)), _elected_23);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 44, make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (stage_p2) * 2432)), _elected_23);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 88, make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (stage_p2) * 2432)), _elected_23);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 88 + 4), make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (stage_p2) * 2432 + 32)), _elected_23);
                            int _smem_desc_lo_24 = (((smem_v20_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                            int _smem_desc_lo_25 = (((smem_v29_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                            {
                                int _smem_desc_lo_26 = (((smem_v23_addr) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_24 | ((uint64_t)((uint32_t)_smem_desc_lo_26 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_27 = (((smem_v32_addr) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_25 | ((uint64_t)((uint32_t)_smem_desc_lo_27 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    0xd04004a0U, tmem_tmem_sfa + 28 + 0, tmem_tmem_sfb + 56 + 0, 1, _elected_23);
                            }
                            elect_commit_cg2_multicast(mma_done_addr + (stage_p1) * 8, (uint16_t)(3));
                            uint32_t _elected_28 = elect_sync();
                            int _smem_desc_lo_29 = (((smem_v21_addr) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                            int _smem_desc_lo_30 = (((smem_v30_addr) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                            {
                                int _smem_desc_lo_31 = (((smem_v23_addr) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_29 | ((uint64_t)((uint32_t)_smem_desc_lo_31 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_32 = (((smem_v32_addr) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_30 | ((uint64_t)((uint32_t)_smem_desc_lo_32 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    0x90400480U, tmem_tmem_sfa + 36 + 0, tmem_tmem_sfb + 72 + 0, 1, _elected_28);
                            }
                            int _smem_desc_lo_33 = (((smem_v22_addr) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                            int _smem_desc_lo_34 = (((smem_v31_addr) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                            {
                                int _smem_desc_lo_35 = (((smem_v23_addr) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_33 | ((uint64_t)((uint32_t)_smem_desc_lo_35 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_36 = (((smem_v32_addr) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_34 | ((uint64_t)((uint32_t)_smem_desc_lo_36 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    0xd04004a0U, tmem_tmem_sfa + 40 + 0, tmem_tmem_sfb + 80 + 0, 1, _elected_28);
                            }
                            elect_commit_cg2_multicast(mma_done_addr + (stage_p2) * 8, (uint16_t)(3));
                            mma_tma_stage += 1;
                            if (mma_tma_stage == 5) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                        }
                        elect_commit_cg2_multicast(mainloop_done_addr, (uint16_t)(3));
                        _phase_epilogue_done ^= 1;
                        mbarrier_wait(work_full_addr + (work_stage_1) * 8, _phase_work_full_1);
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
                            : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                            : "memory");
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                            "}"
                            :: "r"(work_empty_addr + work_stage_1 * 8), "r"(0) : "memory");
                        work_stage_1 += 1;
                        if (work_stage_1 == 4) { work_stage_1 = 0; _phase_work_full_1 ^= 1; }
                        if (_clc_valid_1 == 0) {
                            break;
                        }
                    }
                }
            }
        // ---- Role: padding ----
        } else if (warp >= 2 && warp <= 3) {
            // idle — no tasks assigned
        }
    }
    // ---- Role: epilogue_u ----
    if (warp >= 4 && warp <= 19) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 112;");
        { // epilogue_u_main
            const int col_set_rt = (warp - 4) / 8;
            unsigned int work_stage_2 = 0;
            const int epi_index = warp - 4 - col_set_rt * 8;
            const int epi_warp = epi_index % 4;
            const int head_half = epi_index / 4;
            const int local_row = epi_warp * 32 + lane;
            int exch_slot = (head_half * 128 + local_row) * 3;
            int pair_bar = 1 + head_half * 4 + epi_warp;
            float alpha_v = alpha[0];
            int rows_per_token = HEADS_PER_DESTINATION * 3;
            unsigned int this_bid_1 = bid;
            unsigned int tile_parity = 0;
            int store_bar = 14 + head_half;
            unsigned int _phase_mainloop_done = 0;
            unsigned int _phase_work_full_2 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_2 = 0; _tile_iter_2 < m_tiles / CTA_GROUP * N_TILES; _tile_iter_2++) {
                mbarrier_wait(mainloop_done_addr, _phase_mainloop_done);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int group_1 = this_bid_1 / (unsigned int)tiles_per_group;
                int first_m_1 = group_1 * GROUP_M;
                int remaining_1 = m_tiles - first_m_1;
                int group_size_1 = ((remaining_1 >= GROUP_M) ? GROUP_M : remaining_1);
                int local_1 = this_bid_1 % (unsigned int)tiles_per_group;
                int bid_m_1 = first_m_1 + local_1 % group_size_1;
                int bid_n_1 = local_1 / group_size_1;
                int off_m_1 = bid_m_1 * BLOCK_M;
                int off_n_1 = bid_n_1 * BLOCK_N;
                int token = off_m_1 + local_row;
                int head_kind = bid_n_1 * 2 + head_half;
                int head = head_kind / 3;
                int kind = head_kind % 3;
                int lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)(head_half * B_HALF_N);
                unsigned int words[32];
                float _tmem_load_0[16];
                tmem_ld_x16(&_tmem_load_0[0], lane_addr + ((col_set_rt == 0) ? 0 : 2) * 16);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    words[j] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_0[2 * j] * alpha_v, _tmem_load_0[2 * j + 1] * alpha_v)));
                }
                float _tmem_load_1[16];
                tmem_ld_x16(&_tmem_load_1[0], lane_addr + ((col_set_rt == 0) ? 1 : 5) * 16);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_1 = 0; j_1 < 8; j_1++) {
                    words[8 + j_1] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_1[2 * j_1] * alpha_v, _tmem_load_1[2 * j_1 + 1] * alpha_v)));
                }
                float _tmem_load_2[16];
                tmem_ld_x16(&_tmem_load_2[0], lane_addr + ((col_set_rt == 0) ? 3 : 6) * 16);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_2 = 0; j_2 < 8; j_2++) {
                    words[16 + j_2] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_2[2 * j_2] * alpha_v, _tmem_load_2[2 * j_2 + 1] * alpha_v)));
                }
                float _tmem_load_3[16];
                tmem_ld_x16(&_tmem_load_3[0], lane_addr + ((col_set_rt == 0) ? 4 : 7) * 16);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_3 = 0; j_3 < 8; j_3++) {
                    words[24 + j_3] = __as_u32(__float22bfloat162_rn(make_float2(_tmem_load_3[2 * j_3] * alpha_v, _tmem_load_3[2 * j_3 + 1] * alpha_v)));
                }
                if (elect_sync()) {
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((epilogue_done_addr) & 0xFEFFFFFF) : "memory");
                }
                _phase_mainloop_done ^= 1;
                int destination = head / HEADS_PER_DESTINATION;
                int local_head = head % HEADS_PER_DESTINATION;
                int head_kind_local = local_head * 3 + kind;
                int row_in_destination = token * rows_per_token + head_kind_local;
                unsigned int sr = (unsigned int)row_in_destination;
                unsigned int scale_swizzle = sr >> 7 << 10 | (sr & 31) << 4 | (sr >> 5 & 3) << 2;
                unsigned long long scale_row_base = (unsigned long long)destination * (unsigned long long)SCALE_STRIDE + (unsigned long long)scale_swizzle;
                asm volatile("tcgen05.fence::before_thread_sync;");
                unsigned int staging_buf = (unsigned int)head_half;
                if (kind < 2) {
                    float partial[4];
                    #pragma unroll
                    for (int i = 0; i < 4; i++) {
                        float sum_lo = 0.0f;
                        #pragma unroll
                        for (int j_4 = 0; j_4 < 4; j_4++) {
                            float v_lo = __uint_as_float(words[i * 8 + j_4] << 16);
                            float v_hi = __uint_as_float(words[i * 8 + j_4] & 4294901760u);
                            sum_lo = __fmaf_rn(v_lo, v_lo, sum_lo);
                            sum_lo = __fmaf_rn(v_hi, v_hi, sum_lo);
                        }
                        float sum_hi = 0.0f;
                        #pragma unroll
                        for (int j_5 = 4; j_5 < 8; j_5++) {
                            float v_lo2 = __uint_as_float(words[i * 8 + j_5] << 16);
                            float v_hi2 = __uint_as_float(words[i * 8 + j_5] & 4294901760u);
                            sum_hi = __fmaf_rn(v_lo2, v_lo2, sum_hi);
                            sum_hi = __fmaf_rn(v_hi2, v_hi2, sum_hi);
                        }
                        partial[i] = sum_lo + sum_hi;
                    }
                    if (col_set_rt == 0) {
                        exch[exch_slot] = ((col_set_rt == 0) ? partial[0] + partial[1] : partial[0]);
                        exch[exch_slot + 1] = ((col_set_rt == 0) ? partial[2] : partial[1]);
                        exch[exch_slot + 2] = ((col_set_rt == 0) ? partial[3] : partial[2] + partial[3]);
                    }
                    asm volatile("barrier.sync %0, 64;" :: "r"(pair_bar) : "memory");
                    float ra0 = exch[exch_slot];
                    float ra1 = exch[exch_slot + 1];
                    float ra2 = exch[exch_slot + 2];
                    if (col_set_rt != 0) {
                        exch[exch_slot] = ((col_set_rt == 0) ? partial[0] + partial[1] : partial[0]);
                        exch[exch_slot + 1] = ((col_set_rt == 0) ? partial[2] : partial[1]);
                        exch[exch_slot + 2] = ((col_set_rt == 0) ? partial[3] : partial[2] + partial[3]);
                    }
                    asm volatile("barrier.sync %0, 64;" :: "r"(pair_bar) : "memory");
                    float rb0 = exch[exch_slot];
                    float rb1 = exch[exch_slot + 1];
                    float rb2 = exch[exch_slot + 2];
                    float s01 = ((col_set_rt == 0) ? partial[0] + partial[1] : ((col_set_rt == 0) ? rb0 : ra0));
                    float s23 = ((col_set_rt == 0) ? ((col_set_rt == 0) ? rb0 : ra0) : partial[0]) + ((col_set_rt == 0) ? partial[2] : ((col_set_rt == 0) ? rb1 : ra1));
                    float s45 = ((col_set_rt == 0) ? partial[3] : ((col_set_rt == 0) ? rb2 : ra2)) + ((col_set_rt == 0) ? ((col_set_rt == 0) ? rb1 : ra1) : partial[1]);
                    float s67 = ((col_set_rt == 0) ? ((col_set_rt == 0) ? rb2 : ra2) : partial[2] + partial[3]);
                    float sum_sq = s01 + s23 + (s45 + s67);
                    float mean_sq = __fdiv_rn(sum_sq, 128.0f);
                    float rstd = rsqrtf(mean_sq + eps);
                    float2 _f2_0 = make_float2(rstd, rstd);
                    unsigned int normalized[32];
                    unsigned int loaded_w[8];
                    #pragma unroll
                    for (int h = 0; h < 2; h++) {
                        {
                            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(((kind == 0) ? q_norm_weight : k_norm_weight) + ((col_set_rt == 0) ? 0 : 2) * 16 + h * 8);
                            uint4* _vdst_0 = reinterpret_cast<uint4*>(&loaded_w[4 * h]);
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vdst_0[_blk] = _vptr_0[_blk];
                            }
                        }
                    }
                    #pragma unroll
                    for (int j_6 = 0; j_6 < 8; j_6++) {
                        float2 _f32x2_mul_a_1 = make_float2(__uint_as_float(words[j_6] << 16), __uint_as_float(words[j_6] & 4294901760u));
                        float2 scaled;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1), "l"(*(const unsigned long long*)&_f2_0));
                        float2 _f32x2_mul_b_2 = make_float2(__uint_as_float(loaded_w[j_6] << 16), __uint_as_float(loaded_w[j_6] & 4294901760u));
                        float2 weighted;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&weighted) : "l"(*(const unsigned long long*)&scaled), "l"(*(const unsigned long long*)&_f32x2_mul_b_2));
                        normalized[j_6] = __as_u32(__float22bfloat162_rn(make_float2(weighted.x, weighted.y)));
                    }
                    unsigned int loaded_w_0[8];
                    #pragma unroll
                    for (int h_1 = 0; h_1 < 2; h_1++) {
                        {
                            const uint4* _vptr_3 = reinterpret_cast<const uint4*>(((kind == 0) ? q_norm_weight : k_norm_weight) + ((col_set_rt == 0) ? 1 : 5) * 16 + h_1 * 8);
                            uint4* _vdst_3 = reinterpret_cast<uint4*>(&loaded_w_0[4 * h_1]);
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vdst_3[_blk] = _vptr_3[_blk];
                            }
                        }
                    }
                    #pragma unroll
                    for (int j_7 = 0; j_7 < 8; j_7++) {
                        float2 _f32x2_mul_a_4 = make_float2(__uint_as_float(words[8 + j_7] << 16), __uint_as_float(words[8 + j_7] & 4294901760u));
                        float2 scaled_1;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_4), "l"(*(const unsigned long long*)&_f2_0));
                        float2 _f32x2_mul_b_5 = make_float2(__uint_as_float(loaded_w_0[j_7] << 16), __uint_as_float(loaded_w_0[j_7] & 4294901760u));
                        float2 weighted_1;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&weighted_1) : "l"(*(const unsigned long long*)&scaled_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_5));
                        normalized[8 + j_7] = __as_u32(__float22bfloat162_rn(make_float2(weighted_1.x, weighted_1.y)));
                    }
                    unsigned int loaded_w_1[8];
                    #pragma unroll
                    for (int h_2 = 0; h_2 < 2; h_2++) {
                        {
                            const uint4* _vptr_6 = reinterpret_cast<const uint4*>(((kind == 0) ? q_norm_weight : k_norm_weight) + ((col_set_rt == 0) ? 3 : 6) * 16 + h_2 * 8);
                            uint4* _vdst_6 = reinterpret_cast<uint4*>(&loaded_w_1[4 * h_2]);
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vdst_6[_blk] = _vptr_6[_blk];
                            }
                        }
                    }
                    #pragma unroll
                    for (int j_8 = 0; j_8 < 8; j_8++) {
                        float2 _f32x2_mul_a_7 = make_float2(__uint_as_float(words[16 + j_8] << 16), __uint_as_float(words[16 + j_8] & 4294901760u));
                        float2 scaled_2;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_7), "l"(*(const unsigned long long*)&_f2_0));
                        float2 _f32x2_mul_b_8 = make_float2(__uint_as_float(loaded_w_1[j_8] << 16), __uint_as_float(loaded_w_1[j_8] & 4294901760u));
                        float2 weighted_2;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&weighted_2) : "l"(*(const unsigned long long*)&scaled_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_8));
                        normalized[16 + j_8] = __as_u32(__float22bfloat162_rn(make_float2(weighted_2.x, weighted_2.y)));
                    }
                    unsigned int loaded_w_2[8];
                    #pragma unroll
                    for (int h_3 = 0; h_3 < 2; h_3++) {
                        {
                            const uint4* _vptr_9 = reinterpret_cast<const uint4*>(((kind == 0) ? q_norm_weight : k_norm_weight) + ((col_set_rt == 0) ? 4 : 7) * 16 + h_3 * 8);
                            uint4* _vdst_9 = reinterpret_cast<uint4*>(&loaded_w_2[4 * h_3]);
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vdst_9[_blk] = _vptr_9[_blk];
                            }
                        }
                    }
                    #pragma unroll
                    for (int j_9 = 0; j_9 < 8; j_9++) {
                        float2 _f32x2_mul_a_10 = make_float2(__uint_as_float(words[24 + j_9] << 16), __uint_as_float(words[24 + j_9] & 4294901760u));
                        float2 scaled_3;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_10), "l"(*(const unsigned long long*)&_f2_0));
                        float2 _f32x2_mul_b_11 = make_float2(__uint_as_float(loaded_w_2[j_9] << 16), __uint_as_float(loaded_w_2[j_9] & 4294901760u));
                        float2 weighted_3;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&weighted_3) : "l"(*(const unsigned long long*)&scaled_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_11));
                        normalized[24 + j_9] = __as_u32(__float22bfloat162_rn(make_float2(weighted_3.x, weighted_3.y)));
                    }
                    int rope_token = ((token) < (M - 1) ? (token) : (M - 1));
                    unsigned int loaded_cos[8];
                    unsigned int loaded_sin[8];
                    #pragma unroll
                    for (int h_4 = 0; h_4 < 2; h_4++) {
                        {
                            const uint4* _vptr_12 = reinterpret_cast<const uint4*>(rope_cos_sin + rope_token * 96 + ((col_set_rt == 0) ? 0 : 2) * 16 + h_4 * 8);
                            uint4* _vdst_12 = reinterpret_cast<uint4*>(&loaded_cos[4 * h_4]);
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vdst_12[_blk] = _vptr_12[_blk];
                            }
                        }
                        {
                            const uint4* _vptr_13 = reinterpret_cast<const uint4*>(rope_cos_sin + rope_token * 96 + 48 + ((col_set_rt == 0) ? 0 : 2) * 16 + h_4 * 8);
                            uint4* _vdst_13 = reinterpret_cast<uint4*>(&loaded_sin[4 * h_4]);
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vdst_13[_blk] = _vptr_13[_blk];
                            }
                        }
                    }
                    #pragma unroll
                    for (int j_10 = 0; j_10 < 8; j_10++) {
                        unsigned int n_lo_word = normalized[j_10];
                        unsigned int n_hi_word = ((col_set_rt == 0) ? normalized[16 + j_10] : normalized[8 + j_10]);
                        float2 _f2_1 = make_float2(__uint_as_float(loaded_cos[j_10] << 16), __uint_as_float(loaded_cos[j_10] & 4294901760u));
                        float2 _f2_2 = make_float2(__uint_as_float(loaded_sin[j_10] << 16 ^ 2147483648u), __uint_as_float(loaded_sin[j_10] & 4294901760u ^ 2147483648u));
                        float2 _f2_3 = make_float2(__uint_as_float(loaded_sin[j_10] << 16), __uint_as_float(loaded_sin[j_10] & 4294901760u));
                        float2 _f32x2_mul_a_14 = make_float2(__uint_as_float(n_hi_word << 16), __uint_as_float(n_hi_word & 4294901760u));
                        float2 term_lo;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&term_lo) : "l"(*(const unsigned long long*)&_f32x2_mul_a_14), "l"(*(const unsigned long long*)&_f2_2));
                        float2 rotated = fma_f32x2_rn_ftz(_f2_1, make_float2(__uint_as_float(n_lo_word << 16), __uint_as_float(n_lo_word & 4294901760u)), term_lo);
                        float2 _f32x2_mul_a_15 = make_float2(__uint_as_float(n_lo_word << 16), __uint_as_float(n_lo_word & 4294901760u));
                        float2 term_hi;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&term_hi) : "l"(*(const unsigned long long*)&_f32x2_mul_a_15), "l"(*(const unsigned long long*)&_f2_3));
                        float2 rotated_hi = fma_f32x2_rn_ftz(_f2_1, make_float2(__uint_as_float(n_hi_word << 16), __uint_as_float(n_hi_word & 4294901760u)), term_hi);
                        normalized[j_10] = __as_u32(__float22bfloat162_rn(make_float2(rotated.x, rotated.y)));
                        unsigned int rot_hi_word = __as_u32(__float22bfloat162_rn(make_float2(rotated_hi.x, rotated_hi.y)));
                        normalized[16 + j_10] = ((col_set_rt == 0) ? rot_hi_word : normalized[16 + j_10]);
                        normalized[8 + j_10] = ((col_set_rt == 0) ? normalized[8 + j_10] : rot_hi_word);
                    }
                    unsigned int loaded_cos_3[8];
                    unsigned int loaded_sin_4[8];
                    #pragma unroll
                    for (int h_5 = 0; h_5 < 2; h_5++) {
                        {
                            const uint4* _vptr_16 = reinterpret_cast<const uint4*>(rope_cos_sin + rope_token * 96 + 16 + h_5 * 8);
                            uint4* _vdst_16 = reinterpret_cast<uint4*>(&loaded_cos_3[4 * h_5]);
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vdst_16[_blk] = _vptr_16[_blk];
                            }
                        }
                        {
                            const uint4* _vptr_17 = reinterpret_cast<const uint4*>(rope_cos_sin + rope_token * 96 + 48 + 16 + h_5 * 8);
                            uint4* _vdst_17 = reinterpret_cast<uint4*>(&loaded_sin_4[4 * h_5]);
                            #pragma unroll
                            for (int _blk = 0; _blk < 1; _blk++) {
                                _vdst_17[_blk] = _vptr_17[_blk];
                            }
                        }
                    }
                    if (col_set_rt == 0) {
                        #pragma unroll
                        for (int j_11 = 0; j_11 < 8; j_11++) {
                            unsigned int n_lo_word_1 = normalized[8 + j_11];
                            unsigned int n_hi_word_1 = normalized[24 + j_11];
                            float2 _f2_4 = make_float2(__uint_as_float(loaded_cos_3[j_11] << 16), __uint_as_float(loaded_cos_3[j_11] & 4294901760u));
                            float2 _f2_5 = make_float2(__uint_as_float(loaded_sin_4[j_11] << 16 ^ 2147483648u), __uint_as_float(loaded_sin_4[j_11] & 4294901760u ^ 2147483648u));
                            float2 _f2_6 = make_float2(__uint_as_float(loaded_sin_4[j_11] << 16), __uint_as_float(loaded_sin_4[j_11] & 4294901760u));
                            float2 _f32x2_mul_a_18 = make_float2(__uint_as_float(n_hi_word_1 << 16), __uint_as_float(n_hi_word_1 & 4294901760u));
                            float2 term_lo_1;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&term_lo_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_18), "l"(*(const unsigned long long*)&_f2_5));
                            float2 rotated_1 = fma_f32x2_rn_ftz(_f2_4, make_float2(__uint_as_float(n_lo_word_1 << 16), __uint_as_float(n_lo_word_1 & 4294901760u)), term_lo_1);
                            float2 _f32x2_mul_a_19 = make_float2(__uint_as_float(n_lo_word_1 << 16), __uint_as_float(n_lo_word_1 & 4294901760u));
                            float2 term_hi_1;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&term_hi_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_19), "l"(*(const unsigned long long*)&_f2_6));
                            float2 rotated_hi_1 = fma_f32x2_rn_ftz(_f2_4, make_float2(__uint_as_float(n_hi_word_1 << 16), __uint_as_float(n_hi_word_1 & 4294901760u)), term_hi_1);
                            normalized[8 + j_11] = __as_u32(__float22bfloat162_rn(make_float2(rotated_1.x, rotated_1.y)));
                            unsigned int rot_hi_word_1 = __as_u32(__float22bfloat162_rn(make_float2(rotated_hi_1.x, rotated_hi_1.y)));
                            normalized[24 + j_11] = rot_hi_word_1;
                        }
                    }
                    if (write_debug != 0) {
                        if (token < M) {
                            unsigned long long debug_row = ((unsigned long long)token * 56 + (unsigned long long)head) * 64;
                            #pragma unroll
                            for (int c = 0; c < 2; c++) {
                                unsigned int debug_chunk[4];
                                #pragma unroll
                                for (int j_12 = 0; j_12 < 4; j_12++) {
                                    debug_chunk[j_12] = normalized[4 * c + j_12];
                                }
                                {
                                    auto _gst_20 = ((kind == 0) ? debug_q_words : debug_k_words) + (debug_row + (unsigned long long)(((col_set_rt == 0) ? 0 : 2) * 8 + 4 * c));
                                    __builtin_assume(__isGlobal(_gst_20));
                                    {
                                        int4 _iv4 = make_int4(debug_chunk[0 + 0], debug_chunk[0 + 1], debug_chunk[0 + 2], debug_chunk[0 + 3]);
                                        *reinterpret_cast<int4*>(_gst_20 + 0) = _iv4;
                                    }
                                }
                            }
                            #pragma unroll
                            for (int c_1 = 0; c_1 < 2; c_1++) {
                                unsigned int debug_chunk_1[4];
                                #pragma unroll
                                for (int j_13 = 0; j_13 < 4; j_13++) {
                                    debug_chunk_1[j_13] = normalized[8 + 4 * c_1 + j_13];
                                }
                                {
                                    auto _gst_21 = ((kind == 0) ? debug_q_words : debug_k_words) + (debug_row + (unsigned long long)(((col_set_rt == 0) ? 1 : 5) * 8 + 4 * c_1));
                                    __builtin_assume(__isGlobal(_gst_21));
                                    {
                                        int4 _iv4 = make_int4(debug_chunk_1[0 + 0], debug_chunk_1[0 + 1], debug_chunk_1[0 + 2], debug_chunk_1[0 + 3]);
                                        *reinterpret_cast<int4*>(_gst_21 + 0) = _iv4;
                                    }
                                }
                            }
                            #pragma unroll
                            for (int c_2 = 0; c_2 < 2; c_2++) {
                                unsigned int debug_chunk_2[4];
                                #pragma unroll
                                for (int j_14 = 0; j_14 < 4; j_14++) {
                                    debug_chunk_2[j_14] = normalized[16 + 4 * c_2 + j_14];
                                }
                                {
                                    auto _gst_22 = ((kind == 0) ? debug_q_words : debug_k_words) + (debug_row + (unsigned long long)(((col_set_rt == 0) ? 3 : 6) * 8 + 4 * c_2));
                                    __builtin_assume(__isGlobal(_gst_22));
                                    {
                                        int4 _iv4 = make_int4(debug_chunk_2[0 + 0], debug_chunk_2[0 + 1], debug_chunk_2[0 + 2], debug_chunk_2[0 + 3]);
                                        *reinterpret_cast<int4*>(_gst_22 + 0) = _iv4;
                                    }
                                }
                            }
                            #pragma unroll
                            for (int c_3 = 0; c_3 < 2; c_3++) {
                                unsigned int debug_chunk_3[4];
                                #pragma unroll
                                for (int j_15 = 0; j_15 < 4; j_15++) {
                                    debug_chunk_3[j_15] = normalized[24 + 4 * c_3 + j_15];
                                }
                                {
                                    auto _gst_23 = ((kind == 0) ? debug_q_words : debug_k_words) + (debug_row + (unsigned long long)(((col_set_rt == 0) ? 4 : 7) * 8 + 4 * c_3));
                                    __builtin_assume(__isGlobal(_gst_23));
                                    {
                                        int4 _iv4 = make_int4(debug_chunk_3[0 + 0], debug_chunk_3[0 + 1], debug_chunk_3[0 + 2], debug_chunk_3[0 + 3]);
                                        *reinterpret_cast<int4*>(_gst_23 + 0) = _iv4;
                                    }
                                }
                            }
                        }
                    }
                    if (col_set_rt == 0) {
                        if (epi_warp == 0) {
                            if (elect_sync()) {
                                asm volatile("cp.async.bulk.wait_group.read 0;");
                            }
                        }
                    }
                    asm volatile("barrier.sync %0, 256;" :: "r"(store_bar) : "memory");
                    float out_scale_sel = ((kind == 0) ? out_global_scale[0] : ((kind == 1) ? out_global_scale[1] : out_global_scale[2]));
                    float2 _f2_7 = make_float2(out_scale_sel, out_scale_sel);
                    float cast_values[16];
                    unsigned int e4m3_words[4];
                    #pragma unroll
                    for (int j_16 = 0; j_16 < 8; j_16++) {
                        float2 _f32x2_mul_a_24 = make_float2(__uint_as_float(normalized[j_16] << 16), __uint_as_float(normalized[j_16] & 4294901760u));
                        float2 scaled_4;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_24), "l"(*(const unsigned long long*)&_f2_7));
                        cast_values[2 * j_16] = scaled_4.x;
                        cast_values[2 * j_16 + 1] = scaled_4.y;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(cast_values[0]), "f"(cast_values[1]),
                                               "f"(cast_values[2]), "f"(cast_values[3]));
                        e4m3_words[0] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(cast_values[4]), "f"(cast_values[5]),
                                               "f"(cast_values[6]), "f"(cast_values[7]));
                        e4m3_words[1] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(cast_values[8]), "f"(cast_values[9]),
                                               "f"(cast_values[10]), "f"(cast_values[11]));
                        e4m3_words[2] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(cast_values[12]), "f"(cast_values[13]),
                                               "f"(cast_values[14]), "f"(cast_values[15]));
                        e4m3_words[3] = _packed;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((epi_staging_addr + staging_buf * 16384 + (unsigned int)(local_row * 128 + ((col_set_rt == 0) ? 0 : 2) * 16 ^ (local_row * 128 + ((col_set_rt == 0) ? 0 : 2) * 16 >> 7 & 7) << 4))), "r"(e4m3_words[0]), "r"(e4m3_words[1]), "r"(e4m3_words[2]), "r"(e4m3_words[3]) : "memory");
                    float cast_values_5[16];
                    unsigned int e4m3_words_6[4];
                    #pragma unroll
                    for (int j_17 = 0; j_17 < 8; j_17++) {
                        float2 _f32x2_mul_a_25 = make_float2(__uint_as_float(normalized[8 + j_17] << 16), __uint_as_float(normalized[8 + j_17] & 4294901760u));
                        float2 scaled_5;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_25), "l"(*(const unsigned long long*)&_f2_7));
                        cast_values_5[2 * j_17] = scaled_5.x;
                        cast_values_5[2 * j_17 + 1] = scaled_5.y;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(cast_values_5[0]), "f"(cast_values_5[1]),
                                               "f"(cast_values_5[2]), "f"(cast_values_5[3]));
                        e4m3_words_6[0] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(cast_values_5[4]), "f"(cast_values_5[5]),
                                               "f"(cast_values_5[6]), "f"(cast_values_5[7]));
                        e4m3_words_6[1] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(cast_values_5[8]), "f"(cast_values_5[9]),
                                               "f"(cast_values_5[10]), "f"(cast_values_5[11]));
                        e4m3_words_6[2] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(cast_values_5[12]), "f"(cast_values_5[13]),
                                               "f"(cast_values_5[14]), "f"(cast_values_5[15]));
                        e4m3_words_6[3] = _packed;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((epi_staging_addr + staging_buf * 16384 + (unsigned int)(local_row * 128 + ((col_set_rt == 0) ? 1 : 5) * 16 ^ (local_row * 128 + ((col_set_rt == 0) ? 1 : 5) * 16 >> 7 & 7) << 4))), "r"(e4m3_words_6[0]), "r"(e4m3_words_6[1]), "r"(e4m3_words_6[2]), "r"(e4m3_words_6[3]) : "memory");
                    float cast_values_7[16];
                    unsigned int e4m3_words_8[4];
                    #pragma unroll
                    for (int j_18 = 0; j_18 < 8; j_18++) {
                        float2 _f32x2_mul_a_26 = make_float2(__uint_as_float(normalized[16 + j_18] << 16), __uint_as_float(normalized[16 + j_18] & 4294901760u));
                        float2 scaled_6;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_26), "l"(*(const unsigned long long*)&_f2_7));
                        cast_values_7[2 * j_18] = scaled_6.x;
                        cast_values_7[2 * j_18 + 1] = scaled_6.y;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(cast_values_7[0]), "f"(cast_values_7[1]),
                                               "f"(cast_values_7[2]), "f"(cast_values_7[3]));
                        e4m3_words_8[0] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(cast_values_7[4]), "f"(cast_values_7[5]),
                                               "f"(cast_values_7[6]), "f"(cast_values_7[7]));
                        e4m3_words_8[1] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(cast_values_7[8]), "f"(cast_values_7[9]),
                                               "f"(cast_values_7[10]), "f"(cast_values_7[11]));
                        e4m3_words_8[2] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(cast_values_7[12]), "f"(cast_values_7[13]),
                                               "f"(cast_values_7[14]), "f"(cast_values_7[15]));
                        e4m3_words_8[3] = _packed;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((epi_staging_addr + staging_buf * 16384 + (unsigned int)(local_row * 128 + ((col_set_rt == 0) ? 3 : 6) * 16 ^ (local_row * 128 + ((col_set_rt == 0) ? 3 : 6) * 16 >> 7 & 7) << 4))), "r"(e4m3_words_8[0]), "r"(e4m3_words_8[1]), "r"(e4m3_words_8[2]), "r"(e4m3_words_8[3]) : "memory");
                    float cast_values_9[16];
                    unsigned int e4m3_words_10[4];
                    #pragma unroll
                    for (int j_19 = 0; j_19 < 8; j_19++) {
                        float2 _f32x2_mul_a_27 = make_float2(__uint_as_float(normalized[24 + j_19] << 16), __uint_as_float(normalized[24 + j_19] & 4294901760u));
                        float2 scaled_7;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_27), "l"(*(const unsigned long long*)&_f2_7));
                        cast_values_9[2 * j_19] = scaled_7.x;
                        cast_values_9[2 * j_19 + 1] = scaled_7.y;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(cast_values_9[0]), "f"(cast_values_9[1]),
                                               "f"(cast_values_9[2]), "f"(cast_values_9[3]));
                        e4m3_words_10[0] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(cast_values_9[4]), "f"(cast_values_9[5]),
                                               "f"(cast_values_9[6]), "f"(cast_values_9[7]));
                        e4m3_words_10[1] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(cast_values_9[8]), "f"(cast_values_9[9]),
                                               "f"(cast_values_9[10]), "f"(cast_values_9[11]));
                        e4m3_words_10[2] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(cast_values_9[12]), "f"(cast_values_9[13]),
                                               "f"(cast_values_9[14]), "f"(cast_values_9[15]));
                        e4m3_words_10[3] = _packed;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((epi_staging_addr + staging_buf * 16384 + (unsigned int)(local_row * 128 + ((col_set_rt == 0) ? 4 : 7) * 16 ^ (local_row * 128 + ((col_set_rt == 0) ? 4 : 7) * 16 >> 7 & 7) << 4))), "r"(e4m3_words_10[0]), "r"(e4m3_words_10[1]), "r"(e4m3_words_10[2]), "r"(e4m3_words_10[3]) : "memory");
                } else {
                    if (col_set_rt == 0) {
                        if (epi_warp == 0) {
                            if (elect_sync()) {
                                asm volatile("cp.async.bulk.wait_group.read 0;");
                            }
                        }
                    }
                    asm volatile("barrier.sync %0, 256;" :: "r"(store_bar) : "memory");
                    float out_scale_sel_1 = ((kind == 0) ? out_global_scale[0] : ((kind == 1) ? out_global_scale[1] : out_global_scale[2]));
                    float2 _f2_8 = make_float2(out_scale_sel_1, out_scale_sel_1);
                    float cast_values_1[16];
                    unsigned int e4m3_words_1[4];
                    #pragma unroll
                    for (int j_20 = 0; j_20 < 8; j_20++) {
                        float2 _f32x2_mul_a_28 = make_float2(__uint_as_float(words[j_20] << 16), __uint_as_float(words[j_20] & 4294901760u));
                        float2 scaled_8;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_28), "l"(*(const unsigned long long*)&_f2_8));
                        cast_values_1[2 * j_20] = scaled_8.x;
                        cast_values_1[2 * j_20 + 1] = scaled_8.y;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(cast_values_1[0]), "f"(cast_values_1[1]),
                                               "f"(cast_values_1[2]), "f"(cast_values_1[3]));
                        e4m3_words_1[0] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(cast_values_1[4]), "f"(cast_values_1[5]),
                                               "f"(cast_values_1[6]), "f"(cast_values_1[7]));
                        e4m3_words_1[1] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(cast_values_1[8]), "f"(cast_values_1[9]),
                                               "f"(cast_values_1[10]), "f"(cast_values_1[11]));
                        e4m3_words_1[2] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(cast_values_1[12]), "f"(cast_values_1[13]),
                                               "f"(cast_values_1[14]), "f"(cast_values_1[15]));
                        e4m3_words_1[3] = _packed;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((epi_staging_addr + staging_buf * 16384 + (unsigned int)(local_row * 128 + ((col_set_rt == 0) ? 0 : 2) * 16 ^ (local_row * 128 + ((col_set_rt == 0) ? 0 : 2) * 16 >> 7 & 7) << 4))), "r"(e4m3_words_1[0]), "r"(e4m3_words_1[1]), "r"(e4m3_words_1[2]), "r"(e4m3_words_1[3]) : "memory");
                    float cast_values_0[16];
                    unsigned int e4m3_words_1_1[4];
                    #pragma unroll
                    for (int j_21 = 0; j_21 < 8; j_21++) {
                        float2 _f32x2_mul_a_29 = make_float2(__uint_as_float(words[8 + j_21] << 16), __uint_as_float(words[8 + j_21] & 4294901760u));
                        float2 scaled_9;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_29), "l"(*(const unsigned long long*)&_f2_8));
                        cast_values_0[2 * j_21] = scaled_9.x;
                        cast_values_0[2 * j_21 + 1] = scaled_9.y;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(cast_values_0[0]), "f"(cast_values_0[1]),
                                               "f"(cast_values_0[2]), "f"(cast_values_0[3]));
                        e4m3_words_1_1[0] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(cast_values_0[4]), "f"(cast_values_0[5]),
                                               "f"(cast_values_0[6]), "f"(cast_values_0[7]));
                        e4m3_words_1_1[1] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(cast_values_0[8]), "f"(cast_values_0[9]),
                                               "f"(cast_values_0[10]), "f"(cast_values_0[11]));
                        e4m3_words_1_1[2] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(cast_values_0[12]), "f"(cast_values_0[13]),
                                               "f"(cast_values_0[14]), "f"(cast_values_0[15]));
                        e4m3_words_1_1[3] = _packed;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((epi_staging_addr + staging_buf * 16384 + (unsigned int)(local_row * 128 + ((col_set_rt == 0) ? 1 : 5) * 16 ^ (local_row * 128 + ((col_set_rt == 0) ? 1 : 5) * 16 >> 7 & 7) << 4))), "r"(e4m3_words_1_1[0]), "r"(e4m3_words_1_1[1]), "r"(e4m3_words_1_1[2]), "r"(e4m3_words_1_1[3]) : "memory");
                    float cast_values_2[16];
                    unsigned int e4m3_words_3[4];
                    #pragma unroll
                    for (int j_22 = 0; j_22 < 8; j_22++) {
                        float2 _f32x2_mul_a_30 = make_float2(__uint_as_float(words[16 + j_22] << 16), __uint_as_float(words[16 + j_22] & 4294901760u));
                        float2 scaled_10;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_30), "l"(*(const unsigned long long*)&_f2_8));
                        cast_values_2[2 * j_22] = scaled_10.x;
                        cast_values_2[2 * j_22 + 1] = scaled_10.y;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(cast_values_2[0]), "f"(cast_values_2[1]),
                                               "f"(cast_values_2[2]), "f"(cast_values_2[3]));
                        e4m3_words_3[0] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(cast_values_2[4]), "f"(cast_values_2[5]),
                                               "f"(cast_values_2[6]), "f"(cast_values_2[7]));
                        e4m3_words_3[1] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(cast_values_2[8]), "f"(cast_values_2[9]),
                                               "f"(cast_values_2[10]), "f"(cast_values_2[11]));
                        e4m3_words_3[2] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(cast_values_2[12]), "f"(cast_values_2[13]),
                                               "f"(cast_values_2[14]), "f"(cast_values_2[15]));
                        e4m3_words_3[3] = _packed;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((epi_staging_addr + staging_buf * 16384 + (unsigned int)(local_row * 128 + ((col_set_rt == 0) ? 3 : 6) * 16 ^ (local_row * 128 + ((col_set_rt == 0) ? 3 : 6) * 16 >> 7 & 7) << 4))), "r"(e4m3_words_3[0]), "r"(e4m3_words_3[1]), "r"(e4m3_words_3[2]), "r"(e4m3_words_3[3]) : "memory");
                    float cast_values_4[16];
                    unsigned int e4m3_words_5[4];
                    #pragma unroll
                    for (int j_23 = 0; j_23 < 8; j_23++) {
                        float2 _f32x2_mul_a_31 = make_float2(__uint_as_float(words[24 + j_23] << 16), __uint_as_float(words[24 + j_23] & 4294901760u));
                        float2 scaled_11;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_31), "l"(*(const unsigned long long*)&_f2_8));
                        cast_values_4[2 * j_23] = scaled_11.x;
                        cast_values_4[2 * j_23 + 1] = scaled_11.y;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(cast_values_4[0]), "f"(cast_values_4[1]),
                                               "f"(cast_values_4[2]), "f"(cast_values_4[3]));
                        e4m3_words_5[0] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(cast_values_4[4]), "f"(cast_values_4[5]),
                                               "f"(cast_values_4[6]), "f"(cast_values_4[7]));
                        e4m3_words_5[1] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(cast_values_4[8]), "f"(cast_values_4[9]),
                                               "f"(cast_values_4[10]), "f"(cast_values_4[11]));
                        e4m3_words_5[2] = _packed;
                    }
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(cast_values_4[12]), "f"(cast_values_4[13]),
                                               "f"(cast_values_4[14]), "f"(cast_values_4[15]));
                        e4m3_words_5[3] = _packed;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((epi_staging_addr + staging_buf * 16384 + (unsigned int)(local_row * 128 + ((col_set_rt == 0) ? 4 : 7) * 16 ^ (local_row * 128 + ((col_set_rt == 0) ? 4 : 7) * 16 >> 7 & 7) << 4))), "r"(e4m3_words_5[0]), "r"(e4m3_words_5[1]), "r"(e4m3_words_5[2]), "r"(e4m3_words_5[3]) : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync %0, 256;" :: "r"(store_bar) : "memory");
                if (col_set_rt == 0) {
                    if (epi_warp == 0) {
                        if (elect_sync()) {
                            tma_store_4d((&OUTQ), 0, head_kind_local, off_m_1, destination, epi_staging_addr + staging_buf * 16384);
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    }
                }
                tile_parity = tile_parity ^ 1;
                mbarrier_wait(work_full_addr + (work_stage_2) * 8, _phase_work_full_2);
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
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + work_stage_2 * 8), "r"(0) : "memory");
                work_stage_2 += 1;
                if (work_stage_2 == 4) { work_stage_2 = 0; _phase_work_full_2 ^= 1; }
                if (_clc_valid_2 == 0) {
                    break;
                }
                this_bid_1 = _clc_ctaid_2 + (unsigned int)cta_rank;
            }
            if (col_set_rt == 0) {
                if (epi_warp == 0) {
                    if (elect_sync()) {
                        asm volatile("cp.async.bulk.wait_group 0;");
                    }
                }
            }
        }
    }

    // Cleanup
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
