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
#define NUM_SF_PIPE_STAGES 5
#define NUM_MAINLOOP_PIPE_STAGES 1
#define NUM_WORK_PIPE_STAGES 4
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 32768
#define SMEM_SMEM_B_OFF 17408
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 32768
#define SMEM_SMEM_SFA_ALL_OFF 164864
#define SMEM_SMEM_SFA_ALL_STAGE_BYTES 2048
#define SMEM_SMEM_SFA_ALL_STRIDE 6144
#define SMEM_SMEM_SFB_ALL_OFF 166912
#define SMEM_SMEM_SFB_ALL_STAGE_BYTES 4096
#define SMEM_SMEM_SFB_ALL_STRIDE 6144
#define SMEM_SMEM_V4_OFF 164864
#define SMEM_SMEM_V4_STAGE_BYTES 512
#define SMEM_SMEM_V4_STRIDE 6144
#define SMEM_SMEM_V5_OFF 165376
#define SMEM_SMEM_V5_STAGE_BYTES 512
#define SMEM_SMEM_V5_STRIDE 6144
#define SMEM_SMEM_V6_OFF 165888
#define SMEM_SMEM_V6_STAGE_BYTES 512
#define SMEM_SMEM_V6_STRIDE 6144
#define SMEM_SMEM_V7_OFF 166400
#define SMEM_SMEM_V7_STAGE_BYTES 512
#define SMEM_SMEM_V7_STRIDE 6144
#define SMEM_SMEM_V8_OFF 166912
#define SMEM_SMEM_V8_STAGE_BYTES 1024
#define SMEM_SMEM_V8_STRIDE 6144
#define SMEM_SMEM_V9_OFF 167936
#define SMEM_SMEM_V9_STAGE_BYTES 1024
#define SMEM_SMEM_V9_STRIDE 6144
#define SMEM_SMEM_V10_OFF 168960
#define SMEM_SMEM_V10_STAGE_BYTES 1024
#define SMEM_SMEM_V10_STRIDE 6144
#define SMEM_SMEM_V11_OFF 169984
#define SMEM_SMEM_V11_STAGE_BYTES 1024
#define SMEM_SMEM_V11_STRIDE 6144
#define SMEM_EPI_STAGING_OFF 195584
#define SMEM_EPI_STAGING_STAGE_BYTES 8192
#define SMEM_EPI_STAGING_STRIDE 8192
#define SMEM_WORK_RESPONSE_OFF 228352
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_EXCH_OFF 228416
#define SMEM_EXCH_STAGE_BYTES 3072
#define SMEM_EXCH_STRIDE 3072
#define SMEM_SMEM_V15_OFF 1024
#define SMEM_SMEM_V15_STAGE_BYTES 6144
#define SMEM_SMEM_V15_STRIDE 32768
#define SMEM_SMEM_V16_OFF 1072
#define SMEM_SMEM_V16_STAGE_BYTES 6144
#define SMEM_SMEM_V16_STRIDE 32768
#define SMEM_SMEM_V17_OFF 1120
#define SMEM_SMEM_V17_STAGE_BYTES 6144
#define SMEM_SMEM_V17_STRIDE 32768
#define SMEM_SMEM_V18_OFF 1040
#define SMEM_SMEM_V18_STAGE_BYTES 6144
#define SMEM_SMEM_V18_STRIDE 32768
#define SMEM_SMEM_V19_OFF 1088
#define SMEM_SMEM_V19_STAGE_BYTES 6144
#define SMEM_SMEM_V19_STRIDE 32768
#define SMEM_SMEM_V20_OFF 1136
#define SMEM_SMEM_V20_STAGE_BYTES 6144
#define SMEM_SMEM_V20_STRIDE 32768
#define SMEM_SMEM_V21_OFF 1056
#define SMEM_SMEM_V21_STAGE_BYTES 6144
#define SMEM_SMEM_V21_STRIDE 32768
#define SMEM_SMEM_V22_OFF 1104
#define SMEM_SMEM_V22_STAGE_BYTES 6144
#define SMEM_SMEM_V22_STRIDE 32768
#define SMEM_SMEM_V23_OFF 1024
#define SMEM_SMEM_V23_STAGE_BYTES 6144
#define SMEM_SMEM_V23_STRIDE 32768
#define SMEM_SMEM_V15_0_OFF 17408
#define SMEM_SMEM_V15_0_STAGE_BYTES 6144
#define SMEM_SMEM_V15_0_STRIDE 32768
#define SMEM_SMEM_V16_0_OFF 17456
#define SMEM_SMEM_V16_0_STAGE_BYTES 6144
#define SMEM_SMEM_V16_0_STRIDE 32768
#define SMEM_SMEM_V17_0_OFF 17504
#define SMEM_SMEM_V17_0_STAGE_BYTES 6144
#define SMEM_SMEM_V17_0_STRIDE 32768
#define SMEM_SMEM_V18_0_OFF 17424
#define SMEM_SMEM_V18_0_STAGE_BYTES 6144
#define SMEM_SMEM_V18_0_STRIDE 32768
#define SMEM_SMEM_V19_0_OFF 17472
#define SMEM_SMEM_V19_0_STAGE_BYTES 6144
#define SMEM_SMEM_V19_0_STRIDE 32768
#define SMEM_SMEM_V20_0_OFF 17520
#define SMEM_SMEM_V20_0_STAGE_BYTES 6144
#define SMEM_SMEM_V20_0_STRIDE 32768
#define SMEM_SMEM_V21_0_OFF 17440
#define SMEM_SMEM_V21_0_STAGE_BYTES 6144
#define SMEM_SMEM_V21_0_STRIDE 32768
#define SMEM_SMEM_V22_0_OFF 17488
#define SMEM_SMEM_V22_0_STAGE_BYTES 6144
#define SMEM_SMEM_V22_0_STRIDE 32768
#define SMEM_SMEM_V23_0_OFF 17408
#define SMEM_SMEM_V23_0_STAGE_BYTES 6144
#define SMEM_SMEM_V23_0_STRIDE 32768
#define SMEM_TOTAL 231552
#define BLOCK_M 128
#define BLOCK_N 256
#define B_HALF_N 128
#define BLOCK_K 256
#define CTA_GROUP 2
#define NUM_STAGES 5
#define GROUP_M 64
#define WORK_STAGES 4
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



__device__ __forceinline__ void tcgen05_commit_cg2_multicast_leader(
    int mbar_addr, uint16_t cta_mask, uint32_t leader) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "setp.ne.b32 leader, %2, 0;\n\t"
        "@leader tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [%0], %1;\n\t"
        "}\n"
        :: "r"(mbar_addr), "h"(cta_mask), "r"(leader) : "memory");
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
kernel_cake_minimax_h3_nvfp4_pre_attention_81f02e8b61f9f5b9d75c(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, float* __restrict__ alpha, __nv_bfloat16* __restrict__ q_norm_weight, __nv_bfloat16* __restrict__ k_norm_weight, __nv_bfloat16* __restrict__ rope_cos_sin, float* __restrict__ out_global_scale, uint8_t* __restrict__ out_q, uint8_t* __restrict__ out_sf, const __grid_constant__ CUtensorMap OUTQ, unsigned int* __restrict__ qkv_words, unsigned int* __restrict__ debug_q_words, unsigned int* __restrict__ debug_k_words, int write_debug, float eps, int M, int m_tiles, int HEADS_PER_DESTINATION, int ROWS_PER_DESTINATION, int SCALE_STRIDE)
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
    #define sf_full_addr (mbar_base + 80)
    #define sf_empty_addr (mbar_base + 120)
    #define mainloop_done_addr (mbar_base + 160)
    #define epilogue_done_addr (mbar_base + 168)
    #define work_full_addr (mbar_base + 176)
    #define work_empty_addr (mbar_base + 208)

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
    uint8_t* smem_sfa_all = reinterpret_cast<uint8_t*>(smem_raw + 164864);
    const int smem_sfa_all_addr = smem + 164864;
    uint8_t* smem_sfb_all = reinterpret_cast<uint8_t*>(smem_raw + 166912);
    const int smem_sfb_all_addr = smem + 166912;
    uint8_t* smem_v4 = reinterpret_cast<uint8_t*>(smem_raw + 164864);
    const int smem_v4_addr = smem + 164864;
    uint8_t* smem_v5 = reinterpret_cast<uint8_t*>(smem_raw + 165376);
    const int smem_v5_addr = smem + 165376;
    uint8_t* smem_v6 = reinterpret_cast<uint8_t*>(smem_raw + 165888);
    const int smem_v6_addr = smem + 165888;
    uint8_t* smem_v7 = reinterpret_cast<uint8_t*>(smem_raw + 166400);
    const int smem_v7_addr = smem + 166400;
    uint8_t* smem_v8 = reinterpret_cast<uint8_t*>(smem_raw + 166912);
    const int smem_v8_addr = smem + 166912;
    uint8_t* smem_v9 = reinterpret_cast<uint8_t*>(smem_raw + 167936);
    const int smem_v9_addr = smem + 167936;
    uint8_t* smem_v10 = reinterpret_cast<uint8_t*>(smem_raw + 168960);
    const int smem_v10_addr = smem + 168960;
    uint8_t* smem_v11 = reinterpret_cast<uint8_t*>(smem_raw + 169984);
    const int smem_v11_addr = smem + 169984;
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
    uint8_t* smem_v15_0 = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int smem_v15_addr_0 = smem + 17408;
    uint8_t* smem_v16_0 = reinterpret_cast<uint8_t*>(smem_raw + 17456);
    const int smem_v16_addr_0 = smem + 17456;
    uint8_t* smem_v17_0 = reinterpret_cast<uint8_t*>(smem_raw + 17504);
    const int smem_v17_addr_0 = smem + 17504;
    uint8_t* smem_v18_0 = reinterpret_cast<uint8_t*>(smem_raw + 17424);
    const int smem_v18_addr_0 = smem + 17424;
    uint8_t* smem_v19_0 = reinterpret_cast<uint8_t*>(smem_raw + 17472);
    const int smem_v19_addr_0 = smem + 17472;
    uint8_t* smem_v20_0 = reinterpret_cast<uint8_t*>(smem_raw + 17520);
    const int smem_v20_addr_0 = smem + 17520;
    uint8_t* smem_v21_0 = reinterpret_cast<uint8_t*>(smem_raw + 17440);
    const int smem_v21_addr_0 = smem + 17440;
    uint8_t* smem_v22_0 = reinterpret_cast<uint8_t*>(smem_raw + 17488);
    const int smem_v22_addr_0 = smem + 17488;
    uint8_t* smem_v23_0 = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int smem_v23_addr_0 = smem + 17408;

    // Mbarrier init (8 pipeline groups, 0 ordered-sequence groups, 30 barriers)
    // Mbarriers at smem_raw[0..240)

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
            // --- pipeline 'sf_pipe' ---
            // sf_full: 5 barriers, init_count=2
            mbarrier_init(smem + 80, 2);
            mbarrier_init(smem + 88, 2);
            mbarrier_init(smem + 96, 2);
            mbarrier_init(smem + 104, 2);
            mbarrier_init(smem + 112, 2);
            // sf_empty: 5 barriers, init_count=1
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            // --- pipeline 'mainloop_pipe' ---
            // mainloop_done: 1 barriers, init_count=1
            mbarrier_init(smem + 160, 1);
            // epilogue_done: 1 barriers, init_count=32
            mbarrier_init(smem + 168, 32);
            // --- pipeline 'work_pipe' ---
            // work_full: 4 barriers, init_count=1
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            // work_empty: 4 barriers, init_count=1060
            mbarrier_init(smem + 208, 1060);
            mbarrier_init(smem + 216, 1060);
            mbarrier_init(smem + 224, 1060);
            mbarrier_init(smem + 232, 1060);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 400 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 240);
    if (warp == 0) {
        int _tmem_hold = smem + 240;
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
                                :: "r"(work_full_addr + (work_stage) * 8), "r"(0), "r"((uint32_t)(16)) : "memory");
                            asm volatile(
                                "{\n\t"
                                ".reg .b32 remAddr32;\n\t"
                                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                                "}"
                                :: "r"(work_full_addr + (work_stage) * 8), "r"(1), "r"((uint32_t)(16)) : "memory");
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
                        #pragma unroll 1
                        for (int iter_k = 0; iter_k < NUM_K_ITERS; iter_k++) {
                            mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                            tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 32768, (&A), 0, off_m, iter_k, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 32768, (&B), 0, weight_row_base + off_n, iter_k, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                            load_stage += 1;
                            if (load_stage == 5) { load_stage = 0; _phase_mma_done ^= 1; }
                        }
                        mbarrier_wait(work_full_addr + (work_stage) * 8, _phase_work_full);
                        uint32_t clc_valid = 0;
                        asm volatile(
                            "{\n\t"
                            ".reg .pred p1;\n\t"
                            ".reg .b128 clc_r;\n\t"
                            "ld.shared.b128 clc_r, [%1];\n\t"
                            "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                            "selp.u32 %0, 1, 0, p1;\n\t"
                            "}\n"
                            : "=r"(clc_valid)
                            : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                            : "memory");
                        uint32_t clc_ctaid = 0;
                        asm volatile(
                            "{\n\t"
                            ".reg .pred p1;\n\t"
                            ".reg .b128 clc_r;\n\t"
                            "ld.shared.b128 clc_r, [%1];\n\t"
                            "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                            "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                            "}\n"
                            : "+r"(clc_ctaid)
                            : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                            : "memory");
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                            "}"
                            :: "r"(work_empty_addr + (work_stage) * 8), "r"(0) : "memory");
                        work_stage += 1;
                        if (work_stage == 4) { work_stage = 0; _phase_work_empty ^= 1; _phase_work_full ^= 1; }
                        if (clc_valid == 0) {
                            break;
                        }
                        this_bid = clc_ctaid + (unsigned int)cta_rank;
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
                unsigned int mma_full_stage = 0;
                unsigned int mma_sf_stage = 0;
                unsigned int work_stage_1 = 0;
                unsigned int _phase_epilogue_done = 1;
                unsigned int _phase_sf_full = 0;
                unsigned int _phase_tma_full = 0;
                unsigned int _phase_work_full_1 = 0;
                if (cta_rank == 0) {
                    #pragma unroll 1
                    for (unsigned int _tile_iter_1 = 0; _tile_iter_1 < m_tiles / CTA_GROUP * N_TILES; _tile_iter_1++) {
                        mbarrier_wait(epilogue_done_addr, _phase_epilogue_done);
                        mbarrier_wait(sf_full_addr + (mma_sf_stage) * 8, _phase_sf_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        uint32_t _elected_0 = elect_sync();
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa, make_sf_cp_desc_lo_sbo128((((smem_v4_addr) >> 4) + (mma_sf_stage) * 384)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb, make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_sf_stage) * 384)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 4), make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_sf_stage) * 384 + 32)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 4, make_sf_cp_desc_lo_sbo128((((smem_v5_addr) >> 4) + (mma_sf_stage) * 384)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 8, make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (mma_sf_stage) * 384)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 8 + 4), make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (mma_sf_stage) * 384 + 32)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 8, make_sf_cp_desc_lo_sbo128((((smem_v6_addr) >> 4) + (mma_sf_stage) * 384)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 16, make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (mma_sf_stage) * 384)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 16 + 4), make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (mma_sf_stage) * 384 + 32)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 12, make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (mma_sf_stage) * 384)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 24, make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (mma_sf_stage) * 384)), _elected_0);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 24 + 4), make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (mma_sf_stage) * 384 + 32)), _elected_0);
                        tcgen05_commit_cg2_multicast_leader(sf_empty_addr + (mma_sf_stage) * 8, (uint16_t)(3), _elected_0);
                        mma_sf_stage += 1;
                        if (mma_sf_stage == 5) { mma_sf_stage = 0; _phase_sf_full ^= 1; }
                        mbarrier_wait(tma_full_addr + (mma_full_stage) * 8, _phase_tma_full);
                        mma_full_stage += 1;
                        if (mma_full_stage == 5) { mma_full_stage = 0; _phase_tma_full ^= 1; }
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        unsigned int stage_p0 = mma_tma_stage;
                        uint32_t _elected_1 = elect_sync();
                        int _smem_desc_lo_2 = (((smem_v15_addr) >> 4) & 0x3FFF) + (stage_p0) * 2048;
                        int _smem_desc_lo_3 = (((smem_v15_addr_0) >> 4) & 0x3FFF) + (stage_p0) * 2048;
                        {
                            int _smem_desc_lo_4 = (((smem_v23_addr) >> 4) & 0x3FFF) + (stage_p0) * 2048;
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_2 | ((uint64_t)((uint32_t)_smem_desc_lo_4 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                            int _smem_desc_lo_5 = (((smem_v23_addr_0) >> 4) & 0x3FFF) + (stage_p0) * 2048;
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_3 | ((uint64_t)((uint32_t)_smem_desc_lo_5 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                            tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                0x90400480U, tmem_tmem_sfa + 0, tmem_tmem_sfb + 0, 0, _elected_1);
                        }
                        int _smem_desc_lo_6 = (((smem_v16_addr) >> 4) & 0x3FFF) + (stage_p0) * 2048;
                        int _smem_desc_lo_7 = (((smem_v16_addr_0) >> 4) & 0x3FFF) + (stage_p0) * 2048;
                        {
                            int _smem_desc_lo_8 = (((smem_v23_addr) >> 4) & 0x3FFF) + (stage_p0) * 2048;
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_6 | ((uint64_t)((uint32_t)_smem_desc_lo_8 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                            int _smem_desc_lo_9 = (((smem_v23_addr_0) >> 4) & 0x3FFF) + (stage_p0) * 2048;
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_7 | ((uint64_t)((uint32_t)_smem_desc_lo_9 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                            tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                0xd04004a0U, tmem_tmem_sfa + 4 + 0, tmem_tmem_sfb + 8 + 0, 1, _elected_1);
                        }
                        mbarrier_wait(sf_full_addr + (mma_sf_stage) * 8, _phase_sf_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        uint32_t _elected_10 = elect_sync();
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 16, make_sf_cp_desc_lo_sbo128((((smem_v4_addr) >> 4) + (mma_sf_stage) * 384)), _elected_10);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 32, make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_sf_stage) * 384)), _elected_10);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 32 + 4), make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_sf_stage) * 384 + 32)), _elected_10);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 20, make_sf_cp_desc_lo_sbo128((((smem_v5_addr) >> 4) + (mma_sf_stage) * 384)), _elected_10);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 40, make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (mma_sf_stage) * 384)), _elected_10);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 40 + 4), make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (mma_sf_stage) * 384 + 32)), _elected_10);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 24, make_sf_cp_desc_lo_sbo128((((smem_v6_addr) >> 4) + (mma_sf_stage) * 384)), _elected_10);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 48, make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (mma_sf_stage) * 384)), _elected_10);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 48 + 4), make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (mma_sf_stage) * 384 + 32)), _elected_10);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 28, make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (mma_sf_stage) * 384)), _elected_10);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 56, make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (mma_sf_stage) * 384)), _elected_10);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 56 + 4), make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (mma_sf_stage) * 384 + 32)), _elected_10);
                        tcgen05_commit_cg2_multicast_leader(sf_empty_addr + (mma_sf_stage) * 8, (uint16_t)(3), _elected_10);
                        mma_sf_stage += 1;
                        if (mma_sf_stage == 5) { mma_sf_stage = 0; _phase_sf_full ^= 1; }
                        mbarrier_wait(tma_full_addr + (mma_full_stage) * 8, _phase_tma_full);
                        mma_full_stage += 1;
                        if (mma_full_stage == 5) { mma_full_stage = 0; _phase_tma_full ^= 1; }
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        unsigned int stage_p1 = ((mma_tma_stage + 1 < 5) ? mma_tma_stage + 1 : (unsigned int)0);
                        int _smem_desc_lo_11 = (((smem_v17_addr) >> 4) & 0x3FFF) + (stage_p0) * 2048;
                        int _smem_desc_lo_12 = (((smem_v17_addr_0) >> 4) & 0x3FFF) + (stage_p0) * 2048;
                        if (elect_sync()) {
                            {
                                int _smem_desc_lo_13 = (((smem_v23_addr) >> 4) & 0x3FFF) + (stage_p1) * 2048;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_11 | ((uint64_t)((uint32_t)_smem_desc_lo_13 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_14 = (((smem_v23_addr_0) >> 4) & 0x3FFF) + (stage_p1) * 2048;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_12 | ((uint64_t)((uint32_t)_smem_desc_lo_14 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2(tmem_accum, a_desc + 0, b_desc + 0,
                                    0x90400480U, tmem_tmem_sfa + 12 + 0, tmem_tmem_sfb + 24 + 0, 1);
                            }
                        }
                        elect_commit_cg2_multicast(mma_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                        mma_tma_stage += 1;
                        if (mma_tma_stage == 5) { mma_tma_stage = 0; }
                        uint32_t _elected_15 = elect_sync();
                        int _smem_desc_lo_16 = (((smem_v18_addr) >> 4) & 0x3FFF) + (stage_p1) * 2048;
                        int _smem_desc_lo_17 = (((smem_v18_addr_0) >> 4) & 0x3FFF) + (stage_p1) * 2048;
                        {
                            int _smem_desc_lo_18 = (((smem_v23_addr) >> 4) & 0x3FFF) + (stage_p1) * 2048;
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_16 | ((uint64_t)((uint32_t)_smem_desc_lo_18 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                            int _smem_desc_lo_19 = (((smem_v23_addr_0) >> 4) & 0x3FFF) + (stage_p1) * 2048;
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_17 | ((uint64_t)((uint32_t)_smem_desc_lo_19 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                            tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                0xd04004a0U, tmem_tmem_sfa + 16 + 0, tmem_tmem_sfb + 32 + 0, 1, _elected_15);
                        }
                        int _smem_desc_lo_20 = (((smem_v19_addr) >> 4) & 0x3FFF) + (stage_p1) * 2048;
                        int _smem_desc_lo_21 = (((smem_v19_addr_0) >> 4) & 0x3FFF) + (stage_p1) * 2048;
                        {
                            int _smem_desc_lo_22 = (((smem_v23_addr) >> 4) & 0x3FFF) + (stage_p1) * 2048;
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_20 | ((uint64_t)((uint32_t)_smem_desc_lo_22 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                            int _smem_desc_lo_23 = (((smem_v23_addr_0) >> 4) & 0x3FFF) + (stage_p1) * 2048;
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_21 | ((uint64_t)((uint32_t)_smem_desc_lo_23 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                            tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                0x90400480U, tmem_tmem_sfa + 24 + 0, tmem_tmem_sfb + 48 + 0, 1, _elected_15);
                        }
                        mbarrier_wait(sf_full_addr + (mma_sf_stage) * 8, _phase_sf_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        uint32_t _elected_24 = elect_sync();
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 32, make_sf_cp_desc_lo_sbo128((((smem_v4_addr) >> 4) + (mma_sf_stage) * 384)), _elected_24);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 64, make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_sf_stage) * 384)), _elected_24);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 64 + 4), make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_sf_stage) * 384 + 32)), _elected_24);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 36, make_sf_cp_desc_lo_sbo128((((smem_v5_addr) >> 4) + (mma_sf_stage) * 384)), _elected_24);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 72, make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (mma_sf_stage) * 384)), _elected_24);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 72 + 4), make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (mma_sf_stage) * 384 + 32)), _elected_24);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 40, make_sf_cp_desc_lo_sbo128((((smem_v6_addr) >> 4) + (mma_sf_stage) * 384)), _elected_24);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 80, make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (mma_sf_stage) * 384)), _elected_24);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 80 + 4), make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (mma_sf_stage) * 384 + 32)), _elected_24);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 44, make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (mma_sf_stage) * 384)), _elected_24);
                        tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 88, make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (mma_sf_stage) * 384)), _elected_24);
                        tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 88 + 4), make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (mma_sf_stage) * 384 + 32)), _elected_24);
                        tcgen05_commit_cg2_multicast_leader(sf_empty_addr + (mma_sf_stage) * 8, (uint16_t)(3), _elected_24);
                        mma_sf_stage += 1;
                        if (mma_sf_stage == 5) { mma_sf_stage = 0; _phase_sf_full ^= 1; }
                        mbarrier_wait(tma_full_addr + (mma_full_stage) * 8, _phase_tma_full);
                        mma_full_stage += 1;
                        if (mma_full_stage == 5) { mma_full_stage = 0; _phase_tma_full ^= 1; }
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        unsigned int stage_p2 = ((mma_tma_stage + 1 < 5) ? mma_tma_stage + 1 : (unsigned int)0);
                        int _smem_desc_lo_25 = (((smem_v20_addr) >> 4) & 0x3FFF) + (stage_p1) * 2048;
                        int _smem_desc_lo_26 = (((smem_v20_addr_0) >> 4) & 0x3FFF) + (stage_p1) * 2048;
                        if (elect_sync()) {
                            {
                                int _smem_desc_lo_27 = (((smem_v23_addr) >> 4) & 0x3FFF) + (stage_p2) * 2048;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_25 | ((uint64_t)((uint32_t)_smem_desc_lo_27 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_28 = (((smem_v23_addr_0) >> 4) & 0x3FFF) + (stage_p2) * 2048;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_26 | ((uint64_t)((uint32_t)_smem_desc_lo_28 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2(tmem_accum, a_desc + 0, b_desc + 0,
                                    0xd04004a0U, tmem_tmem_sfa + 28 + 0, tmem_tmem_sfb + 56 + 0, 1);
                            }
                        }
                        elect_commit_cg2_multicast(mma_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                        mma_tma_stage += 1;
                        if (mma_tma_stage == 5) { mma_tma_stage = 0; }
                        uint32_t _elected_29 = elect_sync();
                        int _smem_desc_lo_30 = (((smem_v21_addr) >> 4) & 0x3FFF) + (stage_p2) * 2048;
                        int _smem_desc_lo_31 = (((smem_v21_addr_0) >> 4) & 0x3FFF) + (stage_p2) * 2048;
                        {
                            int _smem_desc_lo_32 = (((smem_v23_addr) >> 4) & 0x3FFF) + (stage_p2) * 2048;
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_30 | ((uint64_t)((uint32_t)_smem_desc_lo_32 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                            int _smem_desc_lo_33 = (((smem_v23_addr_0) >> 4) & 0x3FFF) + (stage_p2) * 2048;
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_31 | ((uint64_t)((uint32_t)_smem_desc_lo_33 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                            tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                0x90400480U, tmem_tmem_sfa + 36 + 0, tmem_tmem_sfb + 72 + 0, 1, _elected_29);
                        }
                        int _smem_desc_lo_34 = (((smem_v22_addr) >> 4) & 0x3FFF) + (stage_p2) * 2048;
                        int _smem_desc_lo_35 = (((smem_v22_addr_0) >> 4) & 0x3FFF) + (stage_p2) * 2048;
                        {
                            int _smem_desc_lo_36 = (((smem_v23_addr) >> 4) & 0x3FFF) + (stage_p2) * 2048;
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_34 | ((uint64_t)((uint32_t)_smem_desc_lo_36 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                            int _smem_desc_lo_37 = (((smem_v23_addr_0) >> 4) & 0x3FFF) + (stage_p2) * 2048;
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_35 | ((uint64_t)((uint32_t)_smem_desc_lo_37 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                            tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                0xd04004a0U, tmem_tmem_sfa + 40 + 0, tmem_tmem_sfb + 80 + 0, 1, _elected_29);
                        }
                        elect_commit_cg2_multicast(mma_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                        mma_tma_stage += 1;
                        if (mma_tma_stage == 5) { mma_tma_stage = 0; }
                        #pragma unroll 1
                        for (int super_k = 1; super_k < 7; super_k++) {
                            mbarrier_wait(sf_full_addr + (mma_sf_stage) * 8, _phase_sf_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            uint32_t _elected_38 = elect_sync();
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa, make_sf_cp_desc_lo_sbo128((((smem_v4_addr) >> 4) + (mma_sf_stage) * 384)), _elected_38);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb, make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_sf_stage) * 384)), _elected_38);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 4), make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_sf_stage) * 384 + 32)), _elected_38);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 4, make_sf_cp_desc_lo_sbo128((((smem_v5_addr) >> 4) + (mma_sf_stage) * 384)), _elected_38);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 8, make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (mma_sf_stage) * 384)), _elected_38);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 8 + 4), make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (mma_sf_stage) * 384 + 32)), _elected_38);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 8, make_sf_cp_desc_lo_sbo128((((smem_v6_addr) >> 4) + (mma_sf_stage) * 384)), _elected_38);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 16, make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (mma_sf_stage) * 384)), _elected_38);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 16 + 4), make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (mma_sf_stage) * 384 + 32)), _elected_38);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 12, make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (mma_sf_stage) * 384)), _elected_38);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 24, make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (mma_sf_stage) * 384)), _elected_38);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 24 + 4), make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (mma_sf_stage) * 384 + 32)), _elected_38);
                            tcgen05_commit_cg2_multicast_leader(sf_empty_addr + (mma_sf_stage) * 8, (uint16_t)(3), _elected_38);
                            mma_sf_stage += 1;
                            if (mma_sf_stage == 5) { mma_sf_stage = 0; _phase_sf_full ^= 1; }
                            mbarrier_wait(tma_full_addr + (mma_full_stage) * 8, _phase_tma_full);
                            mma_full_stage += 1;
                            if (mma_full_stage == 5) { mma_full_stage = 0; _phase_tma_full ^= 1; }
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            unsigned int stage_p0_0 = mma_tma_stage;
                            uint32_t _elected_39 = elect_sync();
                            int _smem_desc_lo_40 = (((smem_v15_addr) >> 4) & 0x3FFF) + (stage_p0_0) * 2048;
                            int _smem_desc_lo_41 = (((smem_v15_addr_0) >> 4) & 0x3FFF) + (stage_p0_0) * 2048;
                            {
                                int _smem_desc_lo_42 = (((smem_v23_addr) >> 4) & 0x3FFF) + (stage_p0_0) * 2048;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_40 | ((uint64_t)((uint32_t)_smem_desc_lo_42 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_43 = (((smem_v23_addr_0) >> 4) & 0x3FFF) + (stage_p0_0) * 2048;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_41 | ((uint64_t)((uint32_t)_smem_desc_lo_43 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    0x90400480U, tmem_tmem_sfa + 0, tmem_tmem_sfb + 0, 1, _elected_39);
                            }
                            int _smem_desc_lo_44 = (((smem_v16_addr) >> 4) & 0x3FFF) + (stage_p0_0) * 2048;
                            int _smem_desc_lo_45 = (((smem_v16_addr_0) >> 4) & 0x3FFF) + (stage_p0_0) * 2048;
                            {
                                int _smem_desc_lo_46 = (((smem_v23_addr) >> 4) & 0x3FFF) + (stage_p0_0) * 2048;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_44 | ((uint64_t)((uint32_t)_smem_desc_lo_46 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_47 = (((smem_v23_addr_0) >> 4) & 0x3FFF) + (stage_p0_0) * 2048;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_45 | ((uint64_t)((uint32_t)_smem_desc_lo_47 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    0xd04004a0U, tmem_tmem_sfa + 4 + 0, tmem_tmem_sfb + 8 + 0, 1, _elected_39);
                            }
                            mbarrier_wait(sf_full_addr + (mma_sf_stage) * 8, _phase_sf_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            uint32_t _elected_48 = elect_sync();
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 16, make_sf_cp_desc_lo_sbo128((((smem_v4_addr) >> 4) + (mma_sf_stage) * 384)), _elected_48);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 32, make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_sf_stage) * 384)), _elected_48);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 32 + 4), make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_sf_stage) * 384 + 32)), _elected_48);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 20, make_sf_cp_desc_lo_sbo128((((smem_v5_addr) >> 4) + (mma_sf_stage) * 384)), _elected_48);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 40, make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (mma_sf_stage) * 384)), _elected_48);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 40 + 4), make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (mma_sf_stage) * 384 + 32)), _elected_48);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 24, make_sf_cp_desc_lo_sbo128((((smem_v6_addr) >> 4) + (mma_sf_stage) * 384)), _elected_48);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 48, make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (mma_sf_stage) * 384)), _elected_48);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 48 + 4), make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (mma_sf_stage) * 384 + 32)), _elected_48);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 28, make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (mma_sf_stage) * 384)), _elected_48);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 56, make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (mma_sf_stage) * 384)), _elected_48);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 56 + 4), make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (mma_sf_stage) * 384 + 32)), _elected_48);
                            tcgen05_commit_cg2_multicast_leader(sf_empty_addr + (mma_sf_stage) * 8, (uint16_t)(3), _elected_48);
                            mma_sf_stage += 1;
                            if (mma_sf_stage == 5) { mma_sf_stage = 0; _phase_sf_full ^= 1; }
                            mbarrier_wait(tma_full_addr + (mma_full_stage) * 8, _phase_tma_full);
                            mma_full_stage += 1;
                            if (mma_full_stage == 5) { mma_full_stage = 0; _phase_tma_full ^= 1; }
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            unsigned int stage_p1_0 = ((mma_tma_stage + 1 < 5) ? mma_tma_stage + 1 : (unsigned int)0);
                            int _smem_desc_lo_49 = (((smem_v17_addr) >> 4) & 0x3FFF) + (stage_p0_0) * 2048;
                            int _smem_desc_lo_50 = (((smem_v17_addr_0) >> 4) & 0x3FFF) + (stage_p0_0) * 2048;
                            if (elect_sync()) {
                                {
                                    int _smem_desc_lo_51 = (((smem_v23_addr) >> 4) & 0x3FFF) + (stage_p1_0) * 2048;
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_49 | ((uint64_t)((uint32_t)_smem_desc_lo_51 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                    int _smem_desc_lo_52 = (((smem_v23_addr_0) >> 4) & 0x3FFF) + (stage_p1_0) * 2048;
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_50 | ((uint64_t)((uint32_t)_smem_desc_lo_52 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2(tmem_accum, a_desc + 0, b_desc + 0,
                                        0x90400480U, tmem_tmem_sfa + 12 + 0, tmem_tmem_sfb + 24 + 0, 1);
                                }
                            }
                            elect_commit_cg2_multicast(mma_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                            mma_tma_stage += 1;
                            if (mma_tma_stage == 5) { mma_tma_stage = 0; }
                            uint32_t _elected_53 = elect_sync();
                            int _smem_desc_lo_54 = (((smem_v18_addr) >> 4) & 0x3FFF) + (stage_p1_0) * 2048;
                            int _smem_desc_lo_55 = (((smem_v18_addr_0) >> 4) & 0x3FFF) + (stage_p1_0) * 2048;
                            {
                                int _smem_desc_lo_56 = (((smem_v23_addr) >> 4) & 0x3FFF) + (stage_p1_0) * 2048;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_54 | ((uint64_t)((uint32_t)_smem_desc_lo_56 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_57 = (((smem_v23_addr_0) >> 4) & 0x3FFF) + (stage_p1_0) * 2048;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_55 | ((uint64_t)((uint32_t)_smem_desc_lo_57 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    0xd04004a0U, tmem_tmem_sfa + 16 + 0, tmem_tmem_sfb + 32 + 0, 1, _elected_53);
                            }
                            int _smem_desc_lo_58 = (((smem_v19_addr) >> 4) & 0x3FFF) + (stage_p1_0) * 2048;
                            int _smem_desc_lo_59 = (((smem_v19_addr_0) >> 4) & 0x3FFF) + (stage_p1_0) * 2048;
                            {
                                int _smem_desc_lo_60 = (((smem_v23_addr) >> 4) & 0x3FFF) + (stage_p1_0) * 2048;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_58 | ((uint64_t)((uint32_t)_smem_desc_lo_60 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_61 = (((smem_v23_addr_0) >> 4) & 0x3FFF) + (stage_p1_0) * 2048;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_59 | ((uint64_t)((uint32_t)_smem_desc_lo_61 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    0x90400480U, tmem_tmem_sfa + 24 + 0, tmem_tmem_sfb + 48 + 0, 1, _elected_53);
                            }
                            mbarrier_wait(sf_full_addr + (mma_sf_stage) * 8, _phase_sf_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            uint32_t _elected_62 = elect_sync();
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 32, make_sf_cp_desc_lo_sbo128((((smem_v4_addr) >> 4) + (mma_sf_stage) * 384)), _elected_62);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 64, make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_sf_stage) * 384)), _elected_62);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 64 + 4), make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (mma_sf_stage) * 384 + 32)), _elected_62);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 36, make_sf_cp_desc_lo_sbo128((((smem_v5_addr) >> 4) + (mma_sf_stage) * 384)), _elected_62);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 72, make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (mma_sf_stage) * 384)), _elected_62);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 72 + 4), make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (mma_sf_stage) * 384 + 32)), _elected_62);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 40, make_sf_cp_desc_lo_sbo128((((smem_v6_addr) >> 4) + (mma_sf_stage) * 384)), _elected_62);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 80, make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (mma_sf_stage) * 384)), _elected_62);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 80 + 4), make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (mma_sf_stage) * 384 + 32)), _elected_62);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfa + 44, make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (mma_sf_stage) * 384)), _elected_62);
                            tcgen05_cp_32x128b_warpx4_cta2_leader(tmem_tmem_sfb + 88, make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (mma_sf_stage) * 384)), _elected_62);
                            tcgen05_cp_32x128b_warpx4_cta2_leader((tmem_tmem_sfb + 88 + 4), make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (mma_sf_stage) * 384 + 32)), _elected_62);
                            tcgen05_commit_cg2_multicast_leader(sf_empty_addr + (mma_sf_stage) * 8, (uint16_t)(3), _elected_62);
                            mma_sf_stage += 1;
                            if (mma_sf_stage == 5) { mma_sf_stage = 0; _phase_sf_full ^= 1; }
                            mbarrier_wait(tma_full_addr + (mma_full_stage) * 8, _phase_tma_full);
                            mma_full_stage += 1;
                            if (mma_full_stage == 5) { mma_full_stage = 0; _phase_tma_full ^= 1; }
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            unsigned int stage_p2_0 = ((mma_tma_stage + 1 < 5) ? mma_tma_stage + 1 : (unsigned int)0);
                            int _smem_desc_lo_63 = (((smem_v20_addr) >> 4) & 0x3FFF) + (stage_p1_0) * 2048;
                            int _smem_desc_lo_64 = (((smem_v20_addr_0) >> 4) & 0x3FFF) + (stage_p1_0) * 2048;
                            if (elect_sync()) {
                                {
                                    int _smem_desc_lo_65 = (((smem_v23_addr) >> 4) & 0x3FFF) + (stage_p2_0) * 2048;
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_63 | ((uint64_t)((uint32_t)_smem_desc_lo_65 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                    int _smem_desc_lo_66 = (((smem_v23_addr_0) >> 4) & 0x3FFF) + (stage_p2_0) * 2048;
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_64 | ((uint64_t)((uint32_t)_smem_desc_lo_66 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2(tmem_accum, a_desc + 0, b_desc + 0,
                                        0xd04004a0U, tmem_tmem_sfa + 28 + 0, tmem_tmem_sfb + 56 + 0, 1);
                                }
                            }
                            elect_commit_cg2_multicast(mma_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                            mma_tma_stage += 1;
                            if (mma_tma_stage == 5) { mma_tma_stage = 0; }
                            uint32_t _elected_67 = elect_sync();
                            int _smem_desc_lo_68 = (((smem_v21_addr) >> 4) & 0x3FFF) + (stage_p2_0) * 2048;
                            int _smem_desc_lo_69 = (((smem_v21_addr_0) >> 4) & 0x3FFF) + (stage_p2_0) * 2048;
                            {
                                int _smem_desc_lo_70 = (((smem_v23_addr) >> 4) & 0x3FFF) + (stage_p2_0) * 2048;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_68 | ((uint64_t)((uint32_t)_smem_desc_lo_70 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_71 = (((smem_v23_addr_0) >> 4) & 0x3FFF) + (stage_p2_0) * 2048;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_69 | ((uint64_t)((uint32_t)_smem_desc_lo_71 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    0x90400480U, tmem_tmem_sfa + 36 + 0, tmem_tmem_sfb + 72 + 0, 1, _elected_67);
                            }
                            int _smem_desc_lo_72 = (((smem_v22_addr) >> 4) & 0x3FFF) + (stage_p2_0) * 2048;
                            int _smem_desc_lo_73 = (((smem_v22_addr_0) >> 4) & 0x3FFF) + (stage_p2_0) * 2048;
                            {
                                int _smem_desc_lo_74 = (((smem_v23_addr) >> 4) & 0x3FFF) + (stage_p2_0) * 2048;
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_72 | ((uint64_t)((uint32_t)_smem_desc_lo_74 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                int _smem_desc_lo_75 = (((smem_v23_addr_0) >> 4) & 0x3FFF) + (stage_p2_0) * 2048;
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_73 | ((uint64_t)((uint32_t)_smem_desc_lo_75 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2_leader(tmem_accum, a_desc + 0, b_desc + 0,
                                    0xd04004a0U, tmem_tmem_sfa + 40 + 0, tmem_tmem_sfb + 80 + 0, 1, _elected_67);
                            }
                            elect_commit_cg2_multicast(mma_done_addr + (mma_tma_stage) * 8, (uint16_t)(3));
                            mma_tma_stage += 1;
                            if (mma_tma_stage == 5) { mma_tma_stage = 0; }
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
                            :: "r"(work_empty_addr + (work_stage_1) * 8), "r"(0) : "memory");
                        work_stage_1 += 1;
                        if (work_stage_1 == 4) { work_stage_1 = 0; _phase_work_full_1 ^= 1; }
                        if (_clc_valid_1 == 0) {
                            break;
                        }
                    }
                }
            }
        // ---- Role: sf_load ----
        } else if (warp == 2) {
            { // sf_load_main
                unsigned int sf_stage = 0;
                unsigned int work_stage_2 = 0;
                unsigned int _phase_sf_empty = 1;
                unsigned int _phase_work_full_2 = 0;
                if (elect_sync()) {
                    unsigned int this_bid_1 = bid;
                    #pragma unroll 1
                    for (unsigned int _tile_iter_2 = 0; _tile_iter_2 < m_tiles / CTA_GROUP * N_TILES; _tile_iter_2++) {
                        int group_1 = this_bid_1 / (unsigned int)tiles_per_group;
                        int first_m_1 = group_1 * GROUP_M;
                        int remaining_1 = m_tiles - first_m_1;
                        int group_size_1 = ((remaining_1 >= GROUP_M) ? GROUP_M : remaining_1);
                        int local_1 = this_bid_1 % (unsigned int)tiles_per_group;
                        int bid_m_1 = first_m_1 + local_1 % group_size_1;
                        int bid_n_1 = local_1 / group_size_1;
                        int off_m_1 = bid_m_1 * BLOCK_M;
                        int off_n_1 = bid_n_1 * BLOCK_N;
                        int sfa_tile_row = bid_m_1 * SF_K_TILES;
                        int sfb_tile_row = bid_n_1 * SF_K_TILES;
                        #pragma unroll 1
                        for (int iter_k_1 = 0; iter_k_1 < NUM_K_ITERS; iter_k_1++) {
                            mbarrier_wait(sf_empty_addr + (sf_stage) * 8, _phase_sf_empty);
                            int k_set_base = iter_k_1 * 4;
                            tma_3d_gmem2smem_cta2(smem_sfa_all_addr + sf_stage * 6144, (&SFA), 0, 0, sfa_tile_row + k_set_base, ((sf_full_addr + (sf_stage) * 8) & 0xFEFFFFFF));
                            tma_3d_gmem2smem_cta2(smem_sfb_all_addr + sf_stage * 6144, (&SFB), 0, 0, sfb_tile_row + k_set_base, ((sf_full_addr + (sf_stage) * 8) & 0xFEFFFFFF));
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((sf_full_addr + (sf_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(6144)) : "memory");
                            sf_stage += 1;
                            if (sf_stage == 5) { sf_stage = 0; _phase_sf_empty ^= 1; }
                        }
                        mbarrier_wait(work_full_addr + (work_stage_2) * 8, _phase_work_full_2);
                        uint32_t clc_valid_1 = 0;
                        asm volatile(
                            "{\n\t"
                            ".reg .pred p1;\n\t"
                            ".reg .b128 clc_r;\n\t"
                            "ld.shared.b128 clc_r, [%1];\n\t"
                            "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                            "selp.u32 %0, 1, 0, p1;\n\t"
                            "}\n"
                            : "=r"(clc_valid_1)
                            : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                            : "memory");
                        uint32_t clc_ctaid_1 = 0;
                        asm volatile(
                            "{\n\t"
                            ".reg .pred p1;\n\t"
                            ".reg .b128 clc_r;\n\t"
                            "ld.shared.b128 clc_r, [%1];\n\t"
                            "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                            "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                            "}\n"
                            : "+r"(clc_ctaid_1)
                            : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                            : "memory");
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                            "}"
                            :: "r"(work_empty_addr + (work_stage_2) * 8), "r"(0) : "memory");
                        work_stage_2 += 1;
                        if (work_stage_2 == 4) { work_stage_2 = 0; _phase_work_full_2 ^= 1; }
                        if (clc_valid_1 == 0) {
                            break;
                        }
                        this_bid_1 = clc_ctaid_1 + (unsigned int)cta_rank;
                    }
                    #pragma unroll 1
                    for (int __1 = 0; __1 < 5; __1++) {
                        mbarrier_wait(sf_empty_addr + (sf_stage) * 8, _phase_sf_empty);
                        sf_stage += 1;
                        if (sf_stage == 5) { sf_stage = 0; _phase_sf_empty ^= 1; }
                    }
                }
            }
        // ---- Role: padding ----
        } else if (warp == 3) {
            // idle — no tasks assigned
        }
    }
    // ---- Role: epilogue_u ----
    if (warp >= 4 && warp <= 19) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 112;");
        { // epilogue_u_main
            const int col_set_rt = (warp - 4) / 8;
            unsigned int work_stage_3 = 0;
            const int epi_index = warp - 4 - col_set_rt * 8;
            const int epi_warp = epi_index % 4;
            const int head_half = epi_index / 4;
            const int local_row = epi_warp * 32 + lane;
            int exch_slot = (head_half * 128 + local_row) * 3;
            int pair_bar = 1 + head_half * 4 + epi_warp;
            float alpha_v = alpha[0];
            float global_scale = out_global_scale[0];
            float global_scale_rcp = approx_rcp(global_scale);
            int rows_per_token = HEADS_PER_DESTINATION * 3;
            unsigned int this_bid_2 = bid;
            unsigned int tile_parity = 0;
            int store_bar = 14 + head_half;
            unsigned int _phase_mainloop_done = 0;
            unsigned int _phase_work_full_3 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_3 = 0; _tile_iter_3 < m_tiles / CTA_GROUP * N_TILES; _tile_iter_3++) {
                mbarrier_wait(mainloop_done_addr, _phase_mainloop_done);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int group_2 = this_bid_2 / (unsigned int)tiles_per_group;
                int first_m_2 = group_2 * GROUP_M;
                int remaining_2 = m_tiles - first_m_2;
                int group_size_2 = ((remaining_2 >= GROUP_M) ? GROUP_M : remaining_2);
                int local_2 = this_bid_2 % (unsigned int)tiles_per_group;
                int bid_m_2 = first_m_2 + local_2 % group_size_2;
                int bid_n_2 = local_2 / group_size_2;
                int off_m_2 = bid_m_2 * BLOCK_M;
                int off_n_2 = bid_n_2 * BLOCK_N;
                int token = off_m_2 + local_row;
                int head_kind = bid_n_2 * 2 + head_half;
                int head = head_kind / 3;
                int kind = head_kind % 3;
                int lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)(head_half * B_HALF_N);
                unsigned int words[32];
                float tmem_load[16];
                tmem_ld_x16(&tmem_load[0], lane_addr + ((col_set_rt == 0) ? 0 : 2) * 16);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    words[j] = __as_u32(__float22bfloat162_rn(make_float2(tmem_load[2 * j] * alpha_v, tmem_load[2 * j + 1] * alpha_v)));
                }
                float tmem_load_1[16];
                tmem_ld_x16(&tmem_load_1[0], lane_addr + ((col_set_rt == 0) ? 1 : 5) * 16);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_1 = 0; j_1 < 8; j_1++) {
                    words[8 + j_1] = __as_u32(__float22bfloat162_rn(make_float2(tmem_load_1[2 * j_1] * alpha_v, tmem_load_1[2 * j_1 + 1] * alpha_v)));
                }
                float tmem_load_2[16];
                tmem_ld_x16(&tmem_load_2[0], lane_addr + ((col_set_rt == 0) ? 3 : 6) * 16);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_2 = 0; j_2 < 8; j_2++) {
                    words[16 + j_2] = __as_u32(__float22bfloat162_rn(make_float2(tmem_load_2[2 * j_2] * alpha_v, tmem_load_2[2 * j_2 + 1] * alpha_v)));
                }
                float tmem_load_3[16];
                tmem_ld_x16(&tmem_load_3[0], lane_addr + ((col_set_rt == 0) ? 4 : 7) * 16);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                #pragma unroll
                for (int j_3 = 0; j_3 < 8; j_3++) {
                    words[24 + j_3] = __as_u32(__float22bfloat162_rn(make_float2(tmem_load_3[2 * j_3] * alpha_v, tmem_load_3[2 * j_3 + 1] * alpha_v)));
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
                unsigned int staging_buf = tile_parity << 1 | (unsigned int)head_half;
                int staging_row = (int)staging_buf * 128 + local_row;
                if (col_set_rt == 0) {
                    if (epi_warp == 0) {
                        if (elect_sync()) {
                            asm volatile("cp.async.bulk.wait_group.read 1;");
                        }
                    }
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync %0, 256;" :: "r"(store_bar) : "memory");
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
                    float ra0 = 0.0f;
                    float ra1 = 0.0f;
                    float ra2 = 0.0f;
                    if (col_set_rt != 0) {
                        ra0 = exch[exch_slot];
                        ra1 = exch[exch_slot + 1];
                        ra2 = exch[exch_slot + 2];
                        exch[exch_slot] = ((col_set_rt == 0) ? partial[0] + partial[1] : partial[0]);
                        exch[exch_slot + 1] = ((col_set_rt == 0) ? partial[2] : partial[1]);
                        exch[exch_slot + 2] = ((col_set_rt == 0) ? partial[3] : partial[2] + partial[3]);
                    }
                    asm volatile("barrier.sync %0, 64;" :: "r"(pair_bar) : "memory");
                    float rb0 = 0.0f;
                    float rb1 = 0.0f;
                    float rb2 = 0.0f;
                    if (col_set_rt == 0) {
                        rb0 = exch[exch_slot];
                        rb1 = exch[exch_slot + 1];
                        rb2 = exch[exch_slot + 2];
                    }
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
                    for (int j_0 = 0; j_0 < 8; j_0++) {
                        float2 _f32x2_mul_a_1 = make_float2(__uint_as_float(words[j_0] << 16), __uint_as_float(words[j_0] & 4294901760u));
                        float2 scaled_0;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_0) : "l"(*(const unsigned long long*)&_f32x2_mul_a_1), "l"(*(const unsigned long long*)&_f2_0));
                        float2 _f32x2_mul_b_2 = make_float2(__uint_as_float(loaded_w[j_0] << 16), __uint_as_float(loaded_w[j_0] & 4294901760u));
                        float2 weighted_0;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&weighted_0) : "l"(*(const unsigned long long*)&scaled_0), "l"(*(const unsigned long long*)&_f32x2_mul_b_2));
                        normalized[j_0] = __as_u32(__float22bfloat162_rn(make_float2(weighted_0.x, weighted_0.y)));
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
                    for (int j_1_1 = 0; j_1_1 < 8; j_1_1++) {
                        float2 _f32x2_mul_a_4 = make_float2(__uint_as_float(words[8 + j_1_1] << 16), __uint_as_float(words[8 + j_1_1] & 4294901760u));
                        float2 scaled_1;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_4), "l"(*(const unsigned long long*)&_f2_0));
                        float2 _f32x2_mul_b_5 = make_float2(__uint_as_float(loaded_w_0[j_1_1] << 16), __uint_as_float(loaded_w_0[j_1_1] & 4294901760u));
                        float2 weighted_1;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&weighted_1) : "l"(*(const unsigned long long*)&scaled_1), "l"(*(const unsigned long long*)&_f32x2_mul_b_5));
                        normalized[8 + j_1_1] = __as_u32(__float22bfloat162_rn(make_float2(weighted_1.x, weighted_1.y)));
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
                    for (int j_2_1 = 0; j_2_1 < 8; j_2_1++) {
                        float2 _f32x2_mul_a_7 = make_float2(__uint_as_float(words[16 + j_2_1] << 16), __uint_as_float(words[16 + j_2_1] & 4294901760u));
                        float2 scaled_2;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_7), "l"(*(const unsigned long long*)&_f2_0));
                        float2 _f32x2_mul_b_8 = make_float2(__uint_as_float(loaded_w_1[j_2_1] << 16), __uint_as_float(loaded_w_1[j_2_1] & 4294901760u));
                        float2 weighted_2;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&weighted_2) : "l"(*(const unsigned long long*)&scaled_2), "l"(*(const unsigned long long*)&_f32x2_mul_b_8));
                        normalized[16 + j_2_1] = __as_u32(__float22bfloat162_rn(make_float2(weighted_2.x, weighted_2.y)));
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
                    for (int j_3_1 = 0; j_3_1 < 8; j_3_1++) {
                        float2 _f32x2_mul_a_10 = make_float2(__uint_as_float(words[24 + j_3_1] << 16), __uint_as_float(words[24 + j_3_1] & 4294901760u));
                        float2 scaled_3;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_3) : "l"(*(const unsigned long long*)&_f32x2_mul_a_10), "l"(*(const unsigned long long*)&_f2_0));
                        float2 _f32x2_mul_b_11 = make_float2(__uint_as_float(loaded_w_2[j_3_1] << 16), __uint_as_float(loaded_w_2[j_3_1] & 4294901760u));
                        float2 weighted_3;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&weighted_3) : "l"(*(const unsigned long long*)&scaled_3), "l"(*(const unsigned long long*)&_f32x2_mul_b_11));
                        normalized[24 + j_3_1] = __as_u32(__float22bfloat162_rn(make_float2(weighted_3.x, weighted_3.y)));
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
                    for (int j_6 = 0; j_6 < 8; j_6++) {
                        unsigned int n_lo_word = normalized[j_6];
                        unsigned int n_hi_word = ((col_set_rt == 0) ? normalized[16 + j_6] : normalized[8 + j_6]);
                        float2 f2 = make_float2(__uint_as_float(loaded_cos[j_6] << 16), __uint_as_float(loaded_cos[j_6] & 4294901760u));
                        float2 f2_0 = make_float2(__uint_as_float(loaded_sin[j_6] << 16 ^ 2147483648u), __uint_as_float(loaded_sin[j_6] & 4294901760u ^ 2147483648u));
                        float2 f2_1 = make_float2(__uint_as_float(loaded_sin[j_6] << 16), __uint_as_float(loaded_sin[j_6] & 4294901760u));
                        float2 _f32x2_mul_a_14 = make_float2(__uint_as_float(n_hi_word << 16), __uint_as_float(n_hi_word & 4294901760u));
                        float2 term_lo;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&term_lo) : "l"(*(const unsigned long long*)&_f32x2_mul_a_14), "l"(*(const unsigned long long*)&f2_0));
                        float2 rotated = fma_f32x2_rn_ftz(f2, make_float2(__uint_as_float(n_lo_word << 16), __uint_as_float(n_lo_word & 4294901760u)), term_lo);
                        float2 _f32x2_mul_a_15 = make_float2(__uint_as_float(n_lo_word << 16), __uint_as_float(n_lo_word & 4294901760u));
                        float2 term_hi;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&term_hi) : "l"(*(const unsigned long long*)&_f32x2_mul_a_15), "l"(*(const unsigned long long*)&f2_1));
                        float2 rotated_hi = fma_f32x2_rn_ftz(f2, make_float2(__uint_as_float(n_hi_word << 16), __uint_as_float(n_hi_word & 4294901760u)), term_hi);
                        normalized[j_6] = __as_u32(__float22bfloat162_rn(make_float2(rotated.x, rotated.y)));
                        unsigned int rot_hi_word = __as_u32(__float22bfloat162_rn(make_float2(rotated_hi.x, rotated_hi.y)));
                        normalized[16 + j_6] = ((col_set_rt == 0) ? rot_hi_word : normalized[16 + j_6]);
                        normalized[8 + j_6] = ((col_set_rt == 0) ? normalized[8 + j_6] : rot_hi_word);
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
                        for (int j_7 = 0; j_7 < 8; j_7++) {
                            unsigned int n_lo_word_1 = normalized[8 + j_7];
                            unsigned int n_hi_word_1 = normalized[24 + j_7];
                            float2 f2_2 = make_float2(__uint_as_float(loaded_cos_3[j_7] << 16), __uint_as_float(loaded_cos_3[j_7] & 4294901760u));
                            float2 f2_2_1 = make_float2(__uint_as_float(loaded_sin_4[j_7] << 16 ^ 2147483648u), __uint_as_float(loaded_sin_4[j_7] & 4294901760u ^ 2147483648u));
                            float2 f2_3 = make_float2(__uint_as_float(loaded_sin_4[j_7] << 16), __uint_as_float(loaded_sin_4[j_7] & 4294901760u));
                            float2 _f32x2_mul_a_18 = make_float2(__uint_as_float(n_hi_word_1 << 16), __uint_as_float(n_hi_word_1 & 4294901760u));
                            float2 term_lo_1;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&term_lo_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_18), "l"(*(const unsigned long long*)&f2_2_1));
                            float2 rotated_1 = fma_f32x2_rn_ftz(f2_2, make_float2(__uint_as_float(n_lo_word_1 << 16), __uint_as_float(n_lo_word_1 & 4294901760u)), term_lo_1);
                            float2 _f32x2_mul_a_19 = make_float2(__uint_as_float(n_lo_word_1 << 16), __uint_as_float(n_lo_word_1 & 4294901760u));
                            float2 term_hi_1;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&term_hi_1) : "l"(*(const unsigned long long*)&_f32x2_mul_a_19), "l"(*(const unsigned long long*)&f2_3));
                            float2 rotated_hi_1 = fma_f32x2_rn_ftz(f2_2, make_float2(__uint_as_float(n_hi_word_1 << 16), __uint_as_float(n_hi_word_1 & 4294901760u)), term_hi_1);
                            normalized[8 + j_7] = __as_u32(__float22bfloat162_rn(make_float2(rotated_1.x, rotated_1.y)));
                            unsigned int rot_hi_word_1 = __as_u32(__float22bfloat162_rn(make_float2(rotated_hi_1.x, rotated_hi_1.y)));
                            normalized[24 + j_7] = rot_hi_word_1;
                        }
                    }
                    if (write_debug != 0) {
                        if (token < M) {
                            unsigned long long debug_row = ((unsigned long long)token * 56 + (unsigned long long)head) * 64;
                            #pragma unroll
                            for (int c_0 = 0; c_0 < 2; c_0++) {
                                unsigned int debug_chunk_0[4];
                                #pragma unroll
                                for (int j_4_1 = 0; j_4_1 < 4; j_4_1++) {
                                    debug_chunk_0[j_4_1] = normalized[4 * c_0 + j_4_1];
                                }
                                {
                                    auto _gst_20 = ((kind == 0) ? debug_q_words : debug_k_words) + (debug_row + (unsigned long long)(((col_set_rt == 0) ? 0 : 2) * 8 + 4 * c_0));
                                    __builtin_assume(__isGlobal(_gst_20));
                                    {
                                        int4 _iv4 = make_int4(debug_chunk_0[0 + 0], debug_chunk_0[0 + 1], debug_chunk_0[0 + 2], debug_chunk_0[0 + 3]);
                                        *reinterpret_cast<int4*>(_gst_20 + 0) = _iv4;
                                    }
                                }
                            }
                            #pragma unroll
                            for (int c_1 = 0; c_1 < 2; c_1++) {
                                unsigned int debug_chunk_1[4];
                                #pragma unroll
                                for (int j_5_1 = 0; j_5_1 < 4; j_5_1++) {
                                    debug_chunk_1[j_5_1] = normalized[8 + 4 * c_1 + j_5_1];
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
                                for (int j_6_1 = 0; j_6_1 < 4; j_6_1++) {
                                    debug_chunk_2[j_6_1] = normalized[16 + 4 * c_2 + j_6_1];
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
                                for (int j_7_1 = 0; j_7_1 < 4; j_7_1++) {
                                    debug_chunk_3[j_7_1] = normalized[24 + 4 * c_3 + j_7_1];
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
                    unsigned int row_out[8];
                    float sf_values[4];
                    float values_0[16];
                    float absolute_0[16];
                    float quant_values_0[16];
                    unsigned int packed_0[2];
                    #pragma unroll
                    for (int j_8 = 0; j_8 < 8; j_8++) {
                        values_0[2 * j_8] = __uint_as_float(normalized[j_8] << 16);
                        values_0[2 * j_8 + 1] = __uint_as_float(normalized[j_8] & 4294901760u);
                    }
                    #pragma unroll
                    for (int j_9 = 0; j_9 < 16; j_9++) {
                        absolute_0[j_9] = values_0[j_9];
                    }
                    absolute_0[0] = fabsf(absolute_0[0]);
                    absolute_0[1] = fabsf(absolute_0[1]);
                    absolute_0[2] = fabsf(absolute_0[2]);
                    absolute_0[3] = fabsf(absolute_0[3]);
                    absolute_0[4] = fabsf(absolute_0[4]);
                    absolute_0[5] = fabsf(absolute_0[5]);
                    absolute_0[6] = fabsf(absolute_0[6]);
                    absolute_0[7] = fabsf(absolute_0[7]);
                    absolute_0[8] = fabsf(absolute_0[8]);
                    absolute_0[9] = fabsf(absolute_0[9]);
                    absolute_0[10] = fabsf(absolute_0[10]);
                    absolute_0[11] = fabsf(absolute_0[11]);
                    absolute_0[12] = fabsf(absolute_0[12]);
                    absolute_0[13] = fabsf(absolute_0[13]);
                    absolute_0[14] = fabsf(absolute_0[14]);
                    absolute_0[15] = fabsf(absolute_0[15]);
                    float amax_0 = absolute_0[0];
                    #pragma unroll
                    for (int _lr = 1; _lr < 16; _lr++) {
                        amax_0 = max_noftz(amax_0, absolute_0[_lr]);
                    }
                    float _rcp_24 = approx_rcp(6.0f);
                    float sf_value_0 = global_scale * (amax_0 * _rcp_24);
                    uint16_t _e4m3x2_26;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_26) : "f"(0.0f), "f"(sf_value_0));
                    uint8_t _cvt_satfinite_e4m3_f32_25 = (uint8_t)_e4m3x2_26;
                    uint32_t _f16x2_decode_27;
                    float sf_rounded_0;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_27) : "h"(_e4m3x2_26));
                    uint16_t _f16_decode_27 = (uint16_t)_f16x2_decode_27;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(sf_rounded_0) : "h"(_f16_decode_27));
                    float _rcp_28 = approx_rcp(sf_rounded_0 * global_scale_rcp);
                    float output_scale_0 = fminf(_rcp_28, 3.4028235e+38f);
                    float2 f2_4 = make_float2(output_scale_0, output_scale_0);
                    #pragma unroll
                    for (int j_10 = 0; j_10 < 8; j_10++) {
                        float2 _f32x2_mul_a_29 = make_float2(values_0[2 * j_10], values_0[2 * j_10 + 1]);
                        float2 scaled_4;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_4) : "l"(*(const unsigned long long*)&_f32x2_mul_a_29), "l"(*(const unsigned long long*)&f2_4));
                        quant_values_0[2 * j_10] = scaled_4.x;
                        quant_values_0[2 * j_10 + 1] = scaled_4.y;
                    }
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_0[0]) : "f"(quant_values_0[0]), "f"(quant_values_0[1]), "f"(quant_values_0[2]), "f"(quant_values_0[3]), "f"(quant_values_0[4]), "f"(quant_values_0[5]), "f"(quant_values_0[6]), "f"(quant_values_0[7]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_0[1]) : "f"(quant_values_0[8]), "f"(quant_values_0[9]), "f"(quant_values_0[10]), "f"(quant_values_0[11]), "f"(quant_values_0[12]), "f"(quant_values_0[13]), "f"(quant_values_0[14]), "f"(quant_values_0[15]));
                    row_out[0] = packed_0[0];
                    row_out[1] = packed_0[1];
                    sf_values[0] = sf_value_0;
                    float values_1[16];
                    float absolute_1[16];
                    float quant_values_1[16];
                    unsigned int packed_1[2];
                    #pragma unroll
                    for (int j_11 = 0; j_11 < 8; j_11++) {
                        values_1[2 * j_11] = __uint_as_float(normalized[8 + j_11] << 16);
                        values_1[2 * j_11 + 1] = __uint_as_float(normalized[8 + j_11] & 4294901760u);
                    }
                    #pragma unroll
                    for (int j_12 = 0; j_12 < 16; j_12++) {
                        absolute_1[j_12] = values_1[j_12];
                    }
                    absolute_1[0] = fabsf(absolute_1[0]);
                    absolute_1[1] = fabsf(absolute_1[1]);
                    absolute_1[2] = fabsf(absolute_1[2]);
                    absolute_1[3] = fabsf(absolute_1[3]);
                    absolute_1[4] = fabsf(absolute_1[4]);
                    absolute_1[5] = fabsf(absolute_1[5]);
                    absolute_1[6] = fabsf(absolute_1[6]);
                    absolute_1[7] = fabsf(absolute_1[7]);
                    absolute_1[8] = fabsf(absolute_1[8]);
                    absolute_1[9] = fabsf(absolute_1[9]);
                    absolute_1[10] = fabsf(absolute_1[10]);
                    absolute_1[11] = fabsf(absolute_1[11]);
                    absolute_1[12] = fabsf(absolute_1[12]);
                    absolute_1[13] = fabsf(absolute_1[13]);
                    absolute_1[14] = fabsf(absolute_1[14]);
                    absolute_1[15] = fabsf(absolute_1[15]);
                    float amax_1 = absolute_1[0];
                    #pragma unroll
                    for (int _lr = 1; _lr < 16; _lr++) {
                        amax_1 = max_noftz(amax_1, absolute_1[_lr]);
                    }
                    float _rcp_30 = approx_rcp(6.0f);
                    float sf_value_1 = global_scale * (amax_1 * _rcp_30);
                    uint16_t _e4m3x2_32;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_32) : "f"(0.0f), "f"(sf_value_1));
                    uint8_t _cvt_satfinite_e4m3_f32_31 = (uint8_t)_e4m3x2_32;
                    uint32_t _f16x2_decode_33;
                    float sf_rounded_1;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_33) : "h"(_e4m3x2_32));
                    uint16_t _f16_decode_33 = (uint16_t)_f16x2_decode_33;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(sf_rounded_1) : "h"(_f16_decode_33));
                    float _rcp_34 = approx_rcp(sf_rounded_1 * global_scale_rcp);
                    float output_scale_1 = fminf(_rcp_34, 3.4028235e+38f);
                    float2 f2_5 = make_float2(output_scale_1, output_scale_1);
                    #pragma unroll
                    for (int j_13 = 0; j_13 < 8; j_13++) {
                        float2 _f32x2_mul_a_35 = make_float2(values_1[2 * j_13], values_1[2 * j_13 + 1]);
                        float2 scaled_5;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_5) : "l"(*(const unsigned long long*)&_f32x2_mul_a_35), "l"(*(const unsigned long long*)&f2_5));
                        quant_values_1[2 * j_13] = scaled_5.x;
                        quant_values_1[2 * j_13 + 1] = scaled_5.y;
                    }
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_1[0]) : "f"(quant_values_1[0]), "f"(quant_values_1[1]), "f"(quant_values_1[2]), "f"(quant_values_1[3]), "f"(quant_values_1[4]), "f"(quant_values_1[5]), "f"(quant_values_1[6]), "f"(quant_values_1[7]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_1[1]) : "f"(quant_values_1[8]), "f"(quant_values_1[9]), "f"(quant_values_1[10]), "f"(quant_values_1[11]), "f"(quant_values_1[12]), "f"(quant_values_1[13]), "f"(quant_values_1[14]), "f"(quant_values_1[15]));
                    row_out[2] = packed_1[0];
                    row_out[3] = packed_1[1];
                    sf_values[1] = sf_value_1;
                    float values_2[16];
                    float absolute_2[16];
                    float quant_values_2[16];
                    unsigned int packed_2[2];
                    #pragma unroll
                    for (int j_14 = 0; j_14 < 8; j_14++) {
                        values_2[2 * j_14] = __uint_as_float(normalized[16 + j_14] << 16);
                        values_2[2 * j_14 + 1] = __uint_as_float(normalized[16 + j_14] & 4294901760u);
                    }
                    #pragma unroll
                    for (int j_15 = 0; j_15 < 16; j_15++) {
                        absolute_2[j_15] = values_2[j_15];
                    }
                    absolute_2[0] = fabsf(absolute_2[0]);
                    absolute_2[1] = fabsf(absolute_2[1]);
                    absolute_2[2] = fabsf(absolute_2[2]);
                    absolute_2[3] = fabsf(absolute_2[3]);
                    absolute_2[4] = fabsf(absolute_2[4]);
                    absolute_2[5] = fabsf(absolute_2[5]);
                    absolute_2[6] = fabsf(absolute_2[6]);
                    absolute_2[7] = fabsf(absolute_2[7]);
                    absolute_2[8] = fabsf(absolute_2[8]);
                    absolute_2[9] = fabsf(absolute_2[9]);
                    absolute_2[10] = fabsf(absolute_2[10]);
                    absolute_2[11] = fabsf(absolute_2[11]);
                    absolute_2[12] = fabsf(absolute_2[12]);
                    absolute_2[13] = fabsf(absolute_2[13]);
                    absolute_2[14] = fabsf(absolute_2[14]);
                    absolute_2[15] = fabsf(absolute_2[15]);
                    float amax_2 = absolute_2[0];
                    #pragma unroll
                    for (int _lr = 1; _lr < 16; _lr++) {
                        amax_2 = max_noftz(amax_2, absolute_2[_lr]);
                    }
                    float _rcp_36 = approx_rcp(6.0f);
                    float sf_value_2 = global_scale * (amax_2 * _rcp_36);
                    uint16_t _e4m3x2_38;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_38) : "f"(0.0f), "f"(sf_value_2));
                    uint8_t _cvt_satfinite_e4m3_f32_37 = (uint8_t)_e4m3x2_38;
                    uint32_t _f16x2_decode_39;
                    float sf_rounded_2;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_39) : "h"(_e4m3x2_38));
                    uint16_t _f16_decode_39 = (uint16_t)_f16x2_decode_39;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(sf_rounded_2) : "h"(_f16_decode_39));
                    float _rcp_40 = approx_rcp(sf_rounded_2 * global_scale_rcp);
                    float output_scale_2 = fminf(_rcp_40, 3.4028235e+38f);
                    float2 f2_6 = make_float2(output_scale_2, output_scale_2);
                    #pragma unroll
                    for (int j_16 = 0; j_16 < 8; j_16++) {
                        float2 _f32x2_mul_a_41 = make_float2(values_2[2 * j_16], values_2[2 * j_16 + 1]);
                        float2 scaled_6;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_6) : "l"(*(const unsigned long long*)&_f32x2_mul_a_41), "l"(*(const unsigned long long*)&f2_6));
                        quant_values_2[2 * j_16] = scaled_6.x;
                        quant_values_2[2 * j_16 + 1] = scaled_6.y;
                    }
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_2[0]) : "f"(quant_values_2[0]), "f"(quant_values_2[1]), "f"(quant_values_2[2]), "f"(quant_values_2[3]), "f"(quant_values_2[4]), "f"(quant_values_2[5]), "f"(quant_values_2[6]), "f"(quant_values_2[7]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_2[1]) : "f"(quant_values_2[8]), "f"(quant_values_2[9]), "f"(quant_values_2[10]), "f"(quant_values_2[11]), "f"(quant_values_2[12]), "f"(quant_values_2[13]), "f"(quant_values_2[14]), "f"(quant_values_2[15]));
                    row_out[4] = packed_2[0];
                    row_out[5] = packed_2[1];
                    sf_values[2] = sf_value_2;
                    float values_3[16];
                    float absolute_3[16];
                    float quant_values_3[16];
                    unsigned int packed_3[2];
                    #pragma unroll
                    for (int j_17 = 0; j_17 < 8; j_17++) {
                        values_3[2 * j_17] = __uint_as_float(normalized[24 + j_17] << 16);
                        values_3[2 * j_17 + 1] = __uint_as_float(normalized[24 + j_17] & 4294901760u);
                    }
                    #pragma unroll
                    for (int j_18 = 0; j_18 < 16; j_18++) {
                        absolute_3[j_18] = values_3[j_18];
                    }
                    absolute_3[0] = fabsf(absolute_3[0]);
                    absolute_3[1] = fabsf(absolute_3[1]);
                    absolute_3[2] = fabsf(absolute_3[2]);
                    absolute_3[3] = fabsf(absolute_3[3]);
                    absolute_3[4] = fabsf(absolute_3[4]);
                    absolute_3[5] = fabsf(absolute_3[5]);
                    absolute_3[6] = fabsf(absolute_3[6]);
                    absolute_3[7] = fabsf(absolute_3[7]);
                    absolute_3[8] = fabsf(absolute_3[8]);
                    absolute_3[9] = fabsf(absolute_3[9]);
                    absolute_3[10] = fabsf(absolute_3[10]);
                    absolute_3[11] = fabsf(absolute_3[11]);
                    absolute_3[12] = fabsf(absolute_3[12]);
                    absolute_3[13] = fabsf(absolute_3[13]);
                    absolute_3[14] = fabsf(absolute_3[14]);
                    absolute_3[15] = fabsf(absolute_3[15]);
                    float amax_3 = absolute_3[0];
                    #pragma unroll
                    for (int _lr = 1; _lr < 16; _lr++) {
                        amax_3 = max_noftz(amax_3, absolute_3[_lr]);
                    }
                    float _rcp_42 = approx_rcp(6.0f);
                    float sf_value_3 = global_scale * (amax_3 * _rcp_42);
                    uint16_t _e4m3x2_44;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_44) : "f"(0.0f), "f"(sf_value_3));
                    uint8_t _cvt_satfinite_e4m3_f32_43 = (uint8_t)_e4m3x2_44;
                    uint32_t _f16x2_decode_45;
                    float sf_rounded_3;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_45) : "h"(_e4m3x2_44));
                    uint16_t _f16_decode_45 = (uint16_t)_f16x2_decode_45;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(sf_rounded_3) : "h"(_f16_decode_45));
                    float _rcp_46 = approx_rcp(sf_rounded_3 * global_scale_rcp);
                    float output_scale_3 = fminf(_rcp_46, 3.4028235e+38f);
                    float2 f2_7 = make_float2(output_scale_3, output_scale_3);
                    #pragma unroll
                    for (int j_19 = 0; j_19 < 8; j_19++) {
                        float2 _f32x2_mul_a_47 = make_float2(values_3[2 * j_19], values_3[2 * j_19 + 1]);
                        float2 scaled_7;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_7) : "l"(*(const unsigned long long*)&_f32x2_mul_a_47), "l"(*(const unsigned long long*)&f2_7));
                        quant_values_3[2 * j_19] = scaled_7.x;
                        quant_values_3[2 * j_19 + 1] = scaled_7.y;
                    }
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_3[0]) : "f"(quant_values_3[0]), "f"(quant_values_3[1]), "f"(quant_values_3[2]), "f"(quant_values_3[3]), "f"(quant_values_3[4]), "f"(quant_values_3[5]), "f"(quant_values_3[6]), "f"(quant_values_3[7]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_3[1]) : "f"(quant_values_3[8]), "f"(quant_values_3[9]), "f"(quant_values_3[10]), "f"(quant_values_3[11]), "f"(quant_values_3[12]), "f"(quant_values_3[13]), "f"(quant_values_3[14]), "f"(quant_values_3[15]));
                    row_out[6] = packed_3[0];
                    row_out[7] = packed_3[1];
                    sf_values[3] = sf_value_3;
                    if (col_set_rt == 0) {
                        exch[exch_slot] = ((col_set_rt == 0) ? sf_values[3] : sf_values[0]);
                    }
                    asm volatile("barrier.sync %0, 64;" :: "r"(pair_bar) : "memory");
                    float recv_a = 0.0f;
                    if (col_set_rt != 0) {
                        recv_a = exch[exch_slot];
                        exch[exch_slot] = ((col_set_rt == 0) ? sf_values[3] : sf_values[0]);
                    }
                    asm volatile("barrier.sync %0, 64;" :: "r"(pair_bar) : "memory");
                    float recv_b = 0.0f;
                    if (col_set_rt == 0) {
                        recv_b = exch[exch_slot];
                    }
                    float sf_group_values[4];
                    sf_group_values[0] = ((col_set_rt == 0) ? sf_values[0] : ((col_set_rt == 0) ? recv_b : recv_a));
                    sf_group_values[1] = sf_values[1];
                    sf_group_values[2] = ((col_set_rt == 0) ? ((col_set_rt == 0) ? recv_b : recv_a) : sf_values[2]);
                    sf_group_values[3] = ((col_set_rt == 0) ? sf_values[2] : sf_values[3]);
                    unsigned int sf_packed[1];
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sf_group_values[0]), "f"(sf_group_values[1]),
                                               "f"(sf_group_values[2]), "f"(sf_group_values[3]));
                        sf_packed[0] = _packed;
                    }
                    if (token < M) {
                        *(reinterpret_cast<int*>(out_sf + (scale_row_base + (unsigned long long)(col_set_rt * 512))) + (0)) = sf_packed[0];
                    }
                    if (col_set_rt == 0) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((epi_staging_addr + (unsigned int)(staging_row * 64 ^ (staging_row * 64 >> 7 & 3) << 4))), "r"(row_out[0]), "r"(row_out[1]), "r"(row_out[2]), "r"(row_out[3]) : "memory");
                        asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"((epi_staging_addr + (unsigned int)(staging_row * 64 + 24 ^ (staging_row * 64 + 24 >> 7 & 3) << 4))), "r"(row_out[4]), "r"(row_out[5]) : "memory");
                        asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"((epi_staging_addr + (unsigned int)(staging_row * 64 + 32 ^ (staging_row * 64 + 32 >> 7 & 3) << 4))), "r"(row_out[6]), "r"(row_out[7]) : "memory");
                    } else {
                        asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"((epi_staging_addr + (unsigned int)(staging_row * 64 + 16 ^ (staging_row * 64 + 16 >> 7 & 3) << 4))), "r"(row_out[0]), "r"(row_out[1]) : "memory");
                        asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"((epi_staging_addr + (unsigned int)(staging_row * 64 + 40 ^ (staging_row * 64 + 40 >> 7 & 3) << 4))), "r"(row_out[2]), "r"(row_out[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((epi_staging_addr + (unsigned int)(staging_row * 64 + 48 ^ (staging_row * 64 + 48 >> 7 & 3) << 4))), "r"(row_out[4]), "r"(row_out[5]), "r"(row_out[6]), "r"(row_out[7]) : "memory");
                    }
                } else {
                    unsigned int row_out_1[8];
                    float sf_values_1[4];
                    float values_4[16];
                    float absolute_4[16];
                    float quant_values_4[16];
                    unsigned int packed_4[2];
                    #pragma unroll
                    for (int j_20 = 0; j_20 < 8; j_20++) {
                        values_4[2 * j_20] = __uint_as_float(words[j_20] << 16);
                        values_4[2 * j_20 + 1] = __uint_as_float(words[j_20] & 4294901760u);
                    }
                    #pragma unroll
                    for (int j_21 = 0; j_21 < 16; j_21++) {
                        absolute_4[j_21] = values_4[j_21];
                    }
                    absolute_4[0] = fabsf(absolute_4[0]);
                    absolute_4[1] = fabsf(absolute_4[1]);
                    absolute_4[2] = fabsf(absolute_4[2]);
                    absolute_4[3] = fabsf(absolute_4[3]);
                    absolute_4[4] = fabsf(absolute_4[4]);
                    absolute_4[5] = fabsf(absolute_4[5]);
                    absolute_4[6] = fabsf(absolute_4[6]);
                    absolute_4[7] = fabsf(absolute_4[7]);
                    absolute_4[8] = fabsf(absolute_4[8]);
                    absolute_4[9] = fabsf(absolute_4[9]);
                    absolute_4[10] = fabsf(absolute_4[10]);
                    absolute_4[11] = fabsf(absolute_4[11]);
                    absolute_4[12] = fabsf(absolute_4[12]);
                    absolute_4[13] = fabsf(absolute_4[13]);
                    absolute_4[14] = fabsf(absolute_4[14]);
                    absolute_4[15] = fabsf(absolute_4[15]);
                    float amax_4 = absolute_4[0];
                    #pragma unroll
                    for (int _lr = 1; _lr < 16; _lr++) {
                        amax_4 = max_noftz(amax_4, absolute_4[_lr]);
                    }
                    float _rcp_48 = approx_rcp(6.0f);
                    float sf_value_4 = global_scale * (amax_4 * _rcp_48);
                    uint16_t _e4m3x2_50;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_50) : "f"(0.0f), "f"(sf_value_4));
                    uint8_t _cvt_satfinite_e4m3_f32_49 = (uint8_t)_e4m3x2_50;
                    uint32_t _f16x2_decode_51;
                    float sf_rounded_4;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_51) : "h"(_e4m3x2_50));
                    uint16_t _f16_decode_51 = (uint16_t)_f16x2_decode_51;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(sf_rounded_4) : "h"(_f16_decode_51));
                    float _rcp_52 = approx_rcp(sf_rounded_4 * global_scale_rcp);
                    float output_scale_4 = fminf(_rcp_52, 3.4028235e+38f);
                    float2 f2_8 = make_float2(output_scale_4, output_scale_4);
                    #pragma unroll
                    for (int j_22 = 0; j_22 < 8; j_22++) {
                        float2 _f32x2_mul_a_53 = make_float2(values_4[2 * j_22], values_4[2 * j_22 + 1]);
                        float2 scaled_8;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_8) : "l"(*(const unsigned long long*)&_f32x2_mul_a_53), "l"(*(const unsigned long long*)&f2_8));
                        quant_values_4[2 * j_22] = scaled_8.x;
                        quant_values_4[2 * j_22 + 1] = scaled_8.y;
                    }
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_4[0]) : "f"(quant_values_4[0]), "f"(quant_values_4[1]), "f"(quant_values_4[2]), "f"(quant_values_4[3]), "f"(quant_values_4[4]), "f"(quant_values_4[5]), "f"(quant_values_4[6]), "f"(quant_values_4[7]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_4[1]) : "f"(quant_values_4[8]), "f"(quant_values_4[9]), "f"(quant_values_4[10]), "f"(quant_values_4[11]), "f"(quant_values_4[12]), "f"(quant_values_4[13]), "f"(quant_values_4[14]), "f"(quant_values_4[15]));
                    row_out_1[0] = packed_4[0];
                    row_out_1[1] = packed_4[1];
                    sf_values_1[0] = sf_value_4;
                    float values_5[16];
                    float absolute_5[16];
                    float quant_values_5[16];
                    unsigned int packed_5[2];
                    #pragma unroll
                    for (int j_23 = 0; j_23 < 8; j_23++) {
                        values_5[2 * j_23] = __uint_as_float(words[8 + j_23] << 16);
                        values_5[2 * j_23 + 1] = __uint_as_float(words[8 + j_23] & 4294901760u);
                    }
                    #pragma unroll
                    for (int j_24 = 0; j_24 < 16; j_24++) {
                        absolute_5[j_24] = values_5[j_24];
                    }
                    absolute_5[0] = fabsf(absolute_5[0]);
                    absolute_5[1] = fabsf(absolute_5[1]);
                    absolute_5[2] = fabsf(absolute_5[2]);
                    absolute_5[3] = fabsf(absolute_5[3]);
                    absolute_5[4] = fabsf(absolute_5[4]);
                    absolute_5[5] = fabsf(absolute_5[5]);
                    absolute_5[6] = fabsf(absolute_5[6]);
                    absolute_5[7] = fabsf(absolute_5[7]);
                    absolute_5[8] = fabsf(absolute_5[8]);
                    absolute_5[9] = fabsf(absolute_5[9]);
                    absolute_5[10] = fabsf(absolute_5[10]);
                    absolute_5[11] = fabsf(absolute_5[11]);
                    absolute_5[12] = fabsf(absolute_5[12]);
                    absolute_5[13] = fabsf(absolute_5[13]);
                    absolute_5[14] = fabsf(absolute_5[14]);
                    absolute_5[15] = fabsf(absolute_5[15]);
                    float amax_5 = absolute_5[0];
                    #pragma unroll
                    for (int _lr = 1; _lr < 16; _lr++) {
                        amax_5 = max_noftz(amax_5, absolute_5[_lr]);
                    }
                    float _rcp_54 = approx_rcp(6.0f);
                    float sf_value_5 = global_scale * (amax_5 * _rcp_54);
                    uint16_t _e4m3x2_56;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_56) : "f"(0.0f), "f"(sf_value_5));
                    uint8_t _cvt_satfinite_e4m3_f32_55 = (uint8_t)_e4m3x2_56;
                    uint32_t _f16x2_decode_57;
                    float sf_rounded_5;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_57) : "h"(_e4m3x2_56));
                    uint16_t _f16_decode_57 = (uint16_t)_f16x2_decode_57;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(sf_rounded_5) : "h"(_f16_decode_57));
                    float _rcp_58 = approx_rcp(sf_rounded_5 * global_scale_rcp);
                    float output_scale_5 = fminf(_rcp_58, 3.4028235e+38f);
                    float2 f2_9 = make_float2(output_scale_5, output_scale_5);
                    #pragma unroll
                    for (int j_25 = 0; j_25 < 8; j_25++) {
                        float2 _f32x2_mul_a_59 = make_float2(values_5[2 * j_25], values_5[2 * j_25 + 1]);
                        float2 scaled_9;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_9) : "l"(*(const unsigned long long*)&_f32x2_mul_a_59), "l"(*(const unsigned long long*)&f2_9));
                        quant_values_5[2 * j_25] = scaled_9.x;
                        quant_values_5[2 * j_25 + 1] = scaled_9.y;
                    }
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_5[0]) : "f"(quant_values_5[0]), "f"(quant_values_5[1]), "f"(quant_values_5[2]), "f"(quant_values_5[3]), "f"(quant_values_5[4]), "f"(quant_values_5[5]), "f"(quant_values_5[6]), "f"(quant_values_5[7]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_5[1]) : "f"(quant_values_5[8]), "f"(quant_values_5[9]), "f"(quant_values_5[10]), "f"(quant_values_5[11]), "f"(quant_values_5[12]), "f"(quant_values_5[13]), "f"(quant_values_5[14]), "f"(quant_values_5[15]));
                    row_out_1[2] = packed_5[0];
                    row_out_1[3] = packed_5[1];
                    sf_values_1[1] = sf_value_5;
                    float values_6[16];
                    float absolute_6[16];
                    float quant_values_6[16];
                    unsigned int packed_6[2];
                    #pragma unroll
                    for (int j_26 = 0; j_26 < 8; j_26++) {
                        values_6[2 * j_26] = __uint_as_float(words[16 + j_26] << 16);
                        values_6[2 * j_26 + 1] = __uint_as_float(words[16 + j_26] & 4294901760u);
                    }
                    #pragma unroll
                    for (int j_27 = 0; j_27 < 16; j_27++) {
                        absolute_6[j_27] = values_6[j_27];
                    }
                    absolute_6[0] = fabsf(absolute_6[0]);
                    absolute_6[1] = fabsf(absolute_6[1]);
                    absolute_6[2] = fabsf(absolute_6[2]);
                    absolute_6[3] = fabsf(absolute_6[3]);
                    absolute_6[4] = fabsf(absolute_6[4]);
                    absolute_6[5] = fabsf(absolute_6[5]);
                    absolute_6[6] = fabsf(absolute_6[6]);
                    absolute_6[7] = fabsf(absolute_6[7]);
                    absolute_6[8] = fabsf(absolute_6[8]);
                    absolute_6[9] = fabsf(absolute_6[9]);
                    absolute_6[10] = fabsf(absolute_6[10]);
                    absolute_6[11] = fabsf(absolute_6[11]);
                    absolute_6[12] = fabsf(absolute_6[12]);
                    absolute_6[13] = fabsf(absolute_6[13]);
                    absolute_6[14] = fabsf(absolute_6[14]);
                    absolute_6[15] = fabsf(absolute_6[15]);
                    float amax_6 = absolute_6[0];
                    #pragma unroll
                    for (int _lr = 1; _lr < 16; _lr++) {
                        amax_6 = max_noftz(amax_6, absolute_6[_lr]);
                    }
                    float _rcp_60 = approx_rcp(6.0f);
                    float sf_value_6 = global_scale * (amax_6 * _rcp_60);
                    uint16_t _e4m3x2_62;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_62) : "f"(0.0f), "f"(sf_value_6));
                    uint8_t _cvt_satfinite_e4m3_f32_61 = (uint8_t)_e4m3x2_62;
                    uint32_t _f16x2_decode_63;
                    float sf_rounded_6;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_63) : "h"(_e4m3x2_62));
                    uint16_t _f16_decode_63 = (uint16_t)_f16x2_decode_63;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(sf_rounded_6) : "h"(_f16_decode_63));
                    float _rcp_64 = approx_rcp(sf_rounded_6 * global_scale_rcp);
                    float output_scale_6 = fminf(_rcp_64, 3.4028235e+38f);
                    float2 f2_10 = make_float2(output_scale_6, output_scale_6);
                    #pragma unroll
                    for (int j_28 = 0; j_28 < 8; j_28++) {
                        float2 _f32x2_mul_a_65 = make_float2(values_6[2 * j_28], values_6[2 * j_28 + 1]);
                        float2 scaled_10;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_10) : "l"(*(const unsigned long long*)&_f32x2_mul_a_65), "l"(*(const unsigned long long*)&f2_10));
                        quant_values_6[2 * j_28] = scaled_10.x;
                        quant_values_6[2 * j_28 + 1] = scaled_10.y;
                    }
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_6[0]) : "f"(quant_values_6[0]), "f"(quant_values_6[1]), "f"(quant_values_6[2]), "f"(quant_values_6[3]), "f"(quant_values_6[4]), "f"(quant_values_6[5]), "f"(quant_values_6[6]), "f"(quant_values_6[7]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_6[1]) : "f"(quant_values_6[8]), "f"(quant_values_6[9]), "f"(quant_values_6[10]), "f"(quant_values_6[11]), "f"(quant_values_6[12]), "f"(quant_values_6[13]), "f"(quant_values_6[14]), "f"(quant_values_6[15]));
                    row_out_1[4] = packed_6[0];
                    row_out_1[5] = packed_6[1];
                    sf_values_1[2] = sf_value_6;
                    float values_7[16];
                    float absolute_7[16];
                    float quant_values_7[16];
                    unsigned int packed_7[2];
                    #pragma unroll
                    for (int j_29 = 0; j_29 < 8; j_29++) {
                        values_7[2 * j_29] = __uint_as_float(words[24 + j_29] << 16);
                        values_7[2 * j_29 + 1] = __uint_as_float(words[24 + j_29] & 4294901760u);
                    }
                    #pragma unroll
                    for (int j_30 = 0; j_30 < 16; j_30++) {
                        absolute_7[j_30] = values_7[j_30];
                    }
                    absolute_7[0] = fabsf(absolute_7[0]);
                    absolute_7[1] = fabsf(absolute_7[1]);
                    absolute_7[2] = fabsf(absolute_7[2]);
                    absolute_7[3] = fabsf(absolute_7[3]);
                    absolute_7[4] = fabsf(absolute_7[4]);
                    absolute_7[5] = fabsf(absolute_7[5]);
                    absolute_7[6] = fabsf(absolute_7[6]);
                    absolute_7[7] = fabsf(absolute_7[7]);
                    absolute_7[8] = fabsf(absolute_7[8]);
                    absolute_7[9] = fabsf(absolute_7[9]);
                    absolute_7[10] = fabsf(absolute_7[10]);
                    absolute_7[11] = fabsf(absolute_7[11]);
                    absolute_7[12] = fabsf(absolute_7[12]);
                    absolute_7[13] = fabsf(absolute_7[13]);
                    absolute_7[14] = fabsf(absolute_7[14]);
                    absolute_7[15] = fabsf(absolute_7[15]);
                    float amax_7 = absolute_7[0];
                    #pragma unroll
                    for (int _lr = 1; _lr < 16; _lr++) {
                        amax_7 = max_noftz(amax_7, absolute_7[_lr]);
                    }
                    float _rcp_66 = approx_rcp(6.0f);
                    float sf_value_7 = global_scale * (amax_7 * _rcp_66);
                    uint16_t _e4m3x2_68;
                    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_68) : "f"(0.0f), "f"(sf_value_7));
                    uint8_t _cvt_satfinite_e4m3_f32_67 = (uint8_t)_e4m3x2_68;
                    uint32_t _f16x2_decode_69;
                    float sf_rounded_7;
                    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_decode_69) : "h"(_e4m3x2_68));
                    uint16_t _f16_decode_69 = (uint16_t)_f16x2_decode_69;
                    asm("cvt.f32.f16 %0, %1;" : "=f"(sf_rounded_7) : "h"(_f16_decode_69));
                    float _rcp_70 = approx_rcp(sf_rounded_7 * global_scale_rcp);
                    float output_scale_7 = fminf(_rcp_70, 3.4028235e+38f);
                    float2 f2_11 = make_float2(output_scale_7, output_scale_7);
                    #pragma unroll
                    for (int j_31 = 0; j_31 < 8; j_31++) {
                        float2 _f32x2_mul_a_71 = make_float2(values_7[2 * j_31], values_7[2 * j_31 + 1]);
                        float2 scaled_11;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&scaled_11) : "l"(*(const unsigned long long*)&_f32x2_mul_a_71), "l"(*(const unsigned long long*)&f2_11));
                        quant_values_7[2 * j_31] = scaled_11.x;
                        quant_values_7[2 * j_31 + 1] = scaled_11.y;
                    }
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_7[0]) : "f"(quant_values_7[0]), "f"(quant_values_7[1]), "f"(quant_values_7[2]), "f"(quant_values_7[3]), "f"(quant_values_7[4]), "f"(quant_values_7[5]), "f"(quant_values_7[6]), "f"(quant_values_7[7]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_7[1]) : "f"(quant_values_7[8]), "f"(quant_values_7[9]), "f"(quant_values_7[10]), "f"(quant_values_7[11]), "f"(quant_values_7[12]), "f"(quant_values_7[13]), "f"(quant_values_7[14]), "f"(quant_values_7[15]));
                    row_out_1[6] = packed_7[0];
                    row_out_1[7] = packed_7[1];
                    sf_values_1[3] = sf_value_7;
                    if (col_set_rt == 0) {
                        exch[exch_slot] = ((col_set_rt == 0) ? sf_values_1[3] : sf_values_1[0]);
                    }
                    asm volatile("barrier.sync %0, 64;" :: "r"(pair_bar) : "memory");
                    float recv_a_1 = 0.0f;
                    if (col_set_rt != 0) {
                        recv_a_1 = exch[exch_slot];
                        exch[exch_slot] = ((col_set_rt == 0) ? sf_values_1[3] : sf_values_1[0]);
                    }
                    asm volatile("barrier.sync %0, 64;" :: "r"(pair_bar) : "memory");
                    float recv_b_1 = 0.0f;
                    if (col_set_rt == 0) {
                        recv_b_1 = exch[exch_slot];
                    }
                    float sf_group_values_1[4];
                    sf_group_values_1[0] = ((col_set_rt == 0) ? sf_values_1[0] : ((col_set_rt == 0) ? recv_b_1 : recv_a_1));
                    sf_group_values_1[1] = sf_values_1[1];
                    sf_group_values_1[2] = ((col_set_rt == 0) ? ((col_set_rt == 0) ? recv_b_1 : recv_a_1) : sf_values_1[2]);
                    sf_group_values_1[3] = ((col_set_rt == 0) ? sf_values_1[2] : sf_values_1[3]);
                    unsigned int sf_packed_1[1];
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sf_group_values_1[0]), "f"(sf_group_values_1[1]),
                                               "f"(sf_group_values_1[2]), "f"(sf_group_values_1[3]));
                        sf_packed_1[0] = _packed;
                    }
                    if (token < M) {
                        *(reinterpret_cast<int*>(out_sf + (scale_row_base + (unsigned long long)(col_set_rt * 512))) + (0)) = sf_packed_1[0];
                    }
                    if (col_set_rt == 0) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((epi_staging_addr + (unsigned int)(staging_row * 64 ^ (staging_row * 64 >> 7 & 3) << 4))), "r"(row_out_1[0]), "r"(row_out_1[1]), "r"(row_out_1[2]), "r"(row_out_1[3]) : "memory");
                        asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"((epi_staging_addr + (unsigned int)(staging_row * 64 + 24 ^ (staging_row * 64 + 24 >> 7 & 3) << 4))), "r"(row_out_1[4]), "r"(row_out_1[5]) : "memory");
                        asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"((epi_staging_addr + (unsigned int)(staging_row * 64 + 32 ^ (staging_row * 64 + 32 >> 7 & 3) << 4))), "r"(row_out_1[6]), "r"(row_out_1[7]) : "memory");
                    } else {
                        asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"((epi_staging_addr + (unsigned int)(staging_row * 64 + 16 ^ (staging_row * 64 + 16 >> 7 & 3) << 4))), "r"(row_out_1[0]), "r"(row_out_1[1]) : "memory");
                        asm volatile("st.shared.v2.b32 [%0], {%1,%2};" :: "r"((epi_staging_addr + (unsigned int)(staging_row * 64 + 40 ^ (staging_row * 64 + 40 >> 7 & 3) << 4))), "r"(row_out_1[2]), "r"(row_out_1[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((epi_staging_addr + (unsigned int)(staging_row * 64 + 48 ^ (staging_row * 64 + 48 >> 7 & 3) << 4))), "r"(row_out_1[4]), "r"(row_out_1[5]), "r"(row_out_1[6]), "r"(row_out_1[7]) : "memory");
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync %0, 256;" :: "r"(store_bar) : "memory");
                if (col_set_rt == 0) {
                    if (epi_warp == 0) {
                        if (elect_sync()) {
                            tma_store_4d((&OUTQ), 0, head_kind_local, off_m_2, destination, epi_staging_addr + staging_buf * 8192);
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    }
                }
                tile_parity = tile_parity ^ 1;
                mbarrier_wait(work_full_addr + (work_stage_3) * 8, _phase_work_full_3);
                uint32_t clc_valid_2 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(clc_valid_2)
                    : "r"(work_response_addr + work_stage_3 * 16 + 0 * 16)
                    : "memory");
                uint32_t clc_ctaid_2 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "+r"(clc_ctaid_2)
                    : "r"(work_response_addr + work_stage_3 * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + (work_stage_3) * 8), "r"(0) : "memory");
                work_stage_3 += 1;
                if (work_stage_3 == 4) { work_stage_3 = 0; _phase_work_full_3 ^= 1; }
                if (clc_valid_2 == 0) {
                    break;
                }
                this_bid_2 = clc_ctaid_2 + (unsigned int)cta_rank;
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
