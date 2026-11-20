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

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 472
#define TMEM_SCORES_OFFSET 0
#define TMEM_OUTPUT_0_OFFSET 128
#define TMEM_OUTPUT_1_OFFSET 256
#define TMEM_TMEM_SFA_QK0_OFFSET 384
#define TMEM_TMEM_SFB_QK0_OFFSET 392
#define TMEM_TMEM_SFA_QK1_OFFSET 400
#define TMEM_TMEM_SFA_PV0_LO_OFFSET 408
#define TMEM_TMEM_SFA_PV0_HI_OFFSET 412
#define TMEM_TMEM_SFA_PV1_LO_OFFSET 416
#define TMEM_TMEM_SFA_PV1_HI_OFFSET 420
#define TMEM_TMEM_SFB_PV_LO_OFFSET 424
#define TMEM_TMEM_SFB_PV_HI_OFFSET 428
#define TMEM_TMEM_SFA_QL0_OFFSET 448
#define TMEM_TMEM_SFA_QL1_OFFSET 456
#define TMEM_TMEM_SFB_QKL_OFFSET 464
#define NUM_K_PIPE_STAGES 4
#define NUM_V_PIPE_STAGES 5
#define SMEM_ROW_STATE_OFF 1024
#define SMEM_ROW_STATE_STAGE_BYTES 2048
#define SMEM_ROW_STATE_STRIDE 2048
#define SMEM_SPLIT_FLAG_OFF 206848
#define SMEM_SPLIT_FLAG_STAGE_BYTES 16
#define SMEM_SPLIT_FLAG_STRIDE 16
#define SMEM_SMEM_Q_OFF 3072
#define SMEM_SMEM_Q_STAGE_BYTES 8192
#define SMEM_SMEM_Q_STRIDE 8192
#define SMEM_SMEM_SFQ_OFF 19456
#define SMEM_SMEM_SFQ_STAGE_BYTES 1024
#define SMEM_SMEM_SFQ_STRIDE 1024
#define SMEM_SMEM_K_OFF 21504
#define SMEM_SMEM_K_STAGE_BYTES 4096
#define SMEM_SMEM_K_STRIDE 4096
#define SMEM_SMEM_VT_OFF 41984
#define SMEM_SMEM_VT_STAGE_BYTES 4096
#define SMEM_SMEM_VT_STRIDE 4096
#define SMEM_SMEM_SFK_OFF 37888
#define SMEM_SMEM_SFK_STAGE_BYTES 1024
#define SMEM_SMEM_SFK_STRIDE 1024
#define SMEM_SMEM_SFVT_LO_OFF 62464
#define SMEM_SMEM_SFVT_LO_STAGE_BYTES 512
#define SMEM_SMEM_SFVT_LO_STRIDE 1024
#define SMEM_SMEM_SFVT_HI_OFF 62976
#define SMEM_SMEM_SFVT_HI_STAGE_BYTES 512
#define SMEM_SMEM_SFVT_HI_STRIDE 1024
#define SMEM_SMEM_P_OFF 67584
#define SMEM_SMEM_P_STAGE_BYTES 8192
#define SMEM_SMEM_P_STRIDE 8192
#define SMEM_SMEM_O00_OFF 83968
#define SMEM_SMEM_O00_STAGE_BYTES 16384
#define SMEM_SMEM_O00_STRIDE 16384
#define SMEM_SMEM_O01_OFF 100352
#define SMEM_SMEM_O01_STAGE_BYTES 16384
#define SMEM_SMEM_O01_STRIDE 16384
#define SMEM_SMEM_O10_OFF 116736
#define SMEM_SMEM_O10_STAGE_BYTES 16384
#define SMEM_SMEM_O10_STRIDE 16384
#define SMEM_SMEM_O11_OFF 133120
#define SMEM_SMEM_O11_STAGE_BYTES 16384
#define SMEM_SMEM_O11_STRIDE 16384
#define SMEM_SMEM_QL_OFF 188416
#define SMEM_SMEM_QL_STAGE_BYTES 8192
#define SMEM_SMEM_QL_STRIDE 8192
#define SMEM_SMEM_SFQL_OFF 204800
#define SMEM_SMEM_SFQL_STAGE_BYTES 1024
#define SMEM_SMEM_SFQL_STRIDE 1024
#define SMEM_SMEM_KL_OFF 167936
#define SMEM_SMEM_KL_STAGE_BYTES 4096
#define SMEM_SMEM_KL_STRIDE 4096
#define SMEM_SMEM_SFKL_OFF 184320
#define SMEM_SMEM_SFKL_STAGE_BYTES 1024
#define SMEM_SMEM_SFKL_STRIDE 1024
#define SMEM_SMEM_TAB_OFF 207872
#define SMEM_SMEM_TAB_STAGE_BYTES 2048
#define SMEM_SMEM_TAB_STRIDE 2048
#define SMEM_SMEM_SEL_OFF 216064
#define SMEM_SMEM_SEL_STAGE_BYTES 4096
#define SMEM_SMEM_SEL_STRIDE 4096
#define SMEM_TOTAL 232448
#define THREADS 512
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

__device__ __forceinline__ void mbarrier_wait_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        ".reg .u32 WAIT_ADDR;\n\t"
        "mov.u32 WAIT_ADDR, %0;\n\t"
        "LAB_WAIT_HINT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [WAIT_ADDR], %1, %2;\n\t"
        "@P1 bra.uni DONE_HINT;\n\t"
        "bra.uni LAB_WAIT_HINT;\n\t"
        "DONE_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
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


__device__ __forceinline__ void mma_ss_step_cg2(
    int a_lo, int b_lo, int taddr, uint32_t i_desc, int enable_d,
    uint32_t a_dhi, uint32_t b_dhi) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader, p;\n\t"
        ".reg .b32 adhi, bdhi, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
        ".reg .b64 da, db;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\t"
        "mov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
        "mov.b32 adhi, %5;\n\t"
        "mov.b32 bdhi, %6;\n\t"
        "mov.b64 da, {%0, adhi};\n\t"
        "mov.b64 db, {%1, bdhi};\n\t"
        "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, %3, "
        "{m0, m1, m2, m3, m4, m5, m6, m7}, p;\n\t"
        "}\n"
        :: "r"(a_lo), "r"(b_lo), "r"(taddr), "r"(i_desc), "r"(enable_d), "r"(a_dhi), "r"(b_dhi));
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


__device__ __forceinline__ void tmem_st_x16_f32(int tmem_addr, const float* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x16.b32"
        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8,"
        "  %9, %10, %11, %12, %13, %14, %15, %16};"
        :: "r"(tmem_addr),
           "f"(src[0]),  "f"(src[1]),  "f"(src[2]),  "f"(src[3]),
           "f"(src[4]),  "f"(src[5]),  "f"(src[6]),  "f"(src[7]),
           "f"(src[8]),  "f"(src[9]),  "f"(src[10]), "f"(src[11]),
           "f"(src[12]), "f"(src[13]), "f"(src[14]), "f"(src[15]));
}


__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
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








__device__ __forceinline__ void fma_f32x2_inplace(float2* a, float2 b, float2 c) {
    unsigned long long r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(r)
        : "l"(*(unsigned long long*)a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    *(unsigned long long*)a = r;
}

__device__ __forceinline__ void mul_f32x2_noftz_inplace(float2* a, float2 b) {
    asm("mul.rn.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ float2 add_f32x2(float2 a, float2 b) {
    float2 r;
    asm("add.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
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



__device__ __forceinline__ uint64_t make_sf_cp_desc_lo_sbo256(int lo) {
    const int SBO = 256;
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



__device__ __forceinline__ void tma_2d_gmem2smem_cta2(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
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

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_cake_nvfp4_attention_4ba55dcb40d27804c77b(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap Vt, const __grid_constant__ CUtensorMap SFQ, const __grid_constant__ CUtensorMap SFK, const __grid_constant__ CUtensorMap SFVtLo, const __grid_constant__ CUtensorMap SFVtHi, const __grid_constant__ CUtensorMap QM, const __grid_constant__ CUtensorMap SFQM, const __grid_constant__ CUtensorMap QL, const __grid_constant__ CUtensorMap SFQL, const __grid_constant__ CUtensorMap KL, const __grid_constant__ CUtensorMap SFKL, const __grid_constant__ CUtensorMap O, int seqlen_q, int seqlen_kv, int q_stride, int kv_stride, float softmax_scale_log2, int total_bh, int heads, int n_full_tiles, int n_chunks, float* __restrict__ part_o, float* __restrict__ part_ml, unsigned int* __restrict__ split_counters, float* __restrict__ q_scale, float* __restrict__ k_scale)
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
    #define q_full_addr (mbar_base + 0)
    #define q_empty_addr (mbar_base + 16)
    #define k_full_addr (mbar_base + 24)
    #define k_empty_addr (mbar_base + 56)
    #define v_full_addr (mbar_base + 88)
    #define v_empty_addr (mbar_base + 128)
    #define s_full_addr (mbar_base + 168)
    #define s_empty_addr (mbar_base + 184)
    #define p_full_addr (mbar_base + 200)
    #define p_full_2_addr (mbar_base + 216)
    #define p_empty_addr (mbar_base + 232)
    #define corr_sig_lo_addr (mbar_base + 248)
    #define corr_sig_hi_addr (mbar_base + 264)
    #define corr_done_addr (mbar_base + 280)
    #define o_full_addr (mbar_base + 296)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    float* row_state = reinterpret_cast<float*>(smem_raw + 1024);
    const int row_state_addr = smem + 1024;
    int* split_flag = reinterpret_cast<int*>(smem_raw + 206848);
    const int split_flag_addr = smem + 206848;
    uint8_t* smem_q = reinterpret_cast<uint8_t*>(smem_raw + 3072);
    const int smem_q_addr = smem + 3072;
    uint8_t* smem_sfq = reinterpret_cast<uint8_t*>(smem_raw + 19456);
    const int smem_sfq_addr = smem + 19456;
    uint8_t* smem_k = reinterpret_cast<uint8_t*>(smem_raw + 21504);
    const int smem_k_addr = smem + 21504;
    uint8_t* smem_vt = reinterpret_cast<uint8_t*>(smem_raw + 41984);
    const int smem_vt_addr = smem + 41984;
    uint8_t* smem_sfk = reinterpret_cast<uint8_t*>(smem_raw + 37888);
    const int smem_sfk_addr = smem + 37888;
    uint8_t* smem_sfvt_lo = reinterpret_cast<uint8_t*>(smem_raw + 62464);
    const int smem_sfvt_lo_addr = smem + 62464;
    uint8_t* smem_sfvt_hi = reinterpret_cast<uint8_t*>(smem_raw + 62976);
    const int smem_sfvt_hi_addr = smem + 62976;
    uint8_t* smem_p = reinterpret_cast<uint8_t*>(smem_raw + 67584);
    const int smem_p_addr = smem + 67584;
    __nv_bfloat16* smem_o00 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 83968);
    const int smem_o00_addr = smem + 83968;
    __nv_bfloat16* smem_o01 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 100352);
    const int smem_o01_addr = smem + 100352;
    __nv_bfloat16* smem_o10 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 116736);
    const int smem_o10_addr = smem + 116736;
    __nv_bfloat16* smem_o11 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 133120);
    const int smem_o11_addr = smem + 133120;
    uint8_t* smem_ql = reinterpret_cast<uint8_t*>(smem_raw + 188416);
    const int smem_ql_addr = smem + 188416;
    uint8_t* smem_sfql = reinterpret_cast<uint8_t*>(smem_raw + 204800);
    const int smem_sfql_addr = smem + 204800;
    uint8_t* smem_kl = reinterpret_cast<uint8_t*>(smem_raw + 167936);
    const int smem_kl_addr = smem + 167936;
    uint8_t* smem_sfkl = reinterpret_cast<uint8_t*>(smem_raw + 184320);
    const int smem_sfkl_addr = smem + 184320;
    __nv_bfloat16* smem_tab = reinterpret_cast<__nv_bfloat16*>(smem_raw + 207872);
    const int smem_tab_addr = smem + 207872;
    __nv_bfloat16* smem_sel = reinterpret_cast<__nv_bfloat16*>(smem_raw + 216064);
    const int smem_sel_addr = smem + 216064;
    unsigned int sel_words[4];
    if (warp == 15) {
        sel_words[0] = 0;
        sel_words[1] = 0;
        sel_words[2] = 0;
        sel_words[3] = 0;
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_sel_addr + (unsigned int)(lane * 16)), "r"(sel_words[0]), "r"(sel_words[1]), "r"(sel_words[2]), "r"(sel_words[3]) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_sel_addr + (unsigned int)((32 + lane) * 16)), "r"(sel_words[0]), "r"(sel_words[1]), "r"(sel_words[2]), "r"(sel_words[3]) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_sel_addr + (unsigned int)((64 + lane) * 16)), "r"(sel_words[0]), "r"(sel_words[1]), "r"(sel_words[2]), "r"(sel_words[3]) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_sel_addr + (unsigned int)((96 + lane) * 16)), "r"(sel_words[0]), "r"(sel_words[1]), "r"(sel_words[2]), "r"(sel_words[3]) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_sel_addr + (unsigned int)((128 + lane) * 16)), "r"(sel_words[0]), "r"(sel_words[1]), "r"(sel_words[2]), "r"(sel_words[3]) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_sel_addr + (unsigned int)((160 + lane) * 16)), "r"(sel_words[0]), "r"(sel_words[1]), "r"(sel_words[2]), "r"(sel_words[3]) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_sel_addr + (unsigned int)((192 + lane) * 16)), "r"(sel_words[0]), "r"(sel_words[1]), "r"(sel_words[2]), "r"(sel_words[3]) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_sel_addr + (unsigned int)((224 + lane) * 16)), "r"(sel_words[0]), "r"(sel_words[1]), "r"(sel_words[2]), "r"(sel_words[3]) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_sel_addr + (unsigned int)((256 + lane) * 16)), "r"(sel_words[0]), "r"(sel_words[1]), "r"(sel_words[2]), "r"(sel_words[3]) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_sel_addr + (unsigned int)((288 + lane) * 16)), "r"(sel_words[0]), "r"(sel_words[1]), "r"(sel_words[2]), "r"(sel_words[3]) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_sel_addr + (unsigned int)((320 + lane) * 16)), "r"(sel_words[0]), "r"(sel_words[1]), "r"(sel_words[2]), "r"(sel_words[3]) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_sel_addr + (unsigned int)((352 + lane) * 16)), "r"(sel_words[0]), "r"(sel_words[1]), "r"(sel_words[2]), "r"(sel_words[3]) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_sel_addr + (unsigned int)((384 + lane) * 16)), "r"(sel_words[0]), "r"(sel_words[1]), "r"(sel_words[2]), "r"(sel_words[3]) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_sel_addr + (unsigned int)((416 + lane) * 16)), "r"(sel_words[0]), "r"(sel_words[1]), "r"(sel_words[2]), "r"(sel_words[3]) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_sel_addr + (unsigned int)((448 + lane) * 16)), "r"(sel_words[0]), "r"(sel_words[1]), "r"(sel_words[2]), "r"(sel_words[3]) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_sel_addr + (unsigned int)((480 + lane) * 16)), "r"(sel_words[0]), "r"(sel_words[1]), "r"(sel_words[2]), "r"(sel_words[3]) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_sel_addr + (unsigned int)((512 + lane) * 16)), "r"(sel_words[0]), "r"(sel_words[1]), "r"(sel_words[2]), "r"(sel_words[3]) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_sel_addr + (unsigned int)((544 + lane) * 16)), "r"(sel_words[0]), "r"(sel_words[1]), "r"(sel_words[2]), "r"(sel_words[3]) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_sel_addr + (unsigned int)((576 + lane) * 16)), "r"(sel_words[0]), "r"(sel_words[1]), "r"(sel_words[2]), "r"(sel_words[3]) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_sel_addr + (unsigned int)((608 + lane) * 16)), "r"(sel_words[0]), "r"(sel_words[1]), "r"(sel_words[2]), "r"(sel_words[3]) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_sel_addr + (unsigned int)((640 + lane) * 16)), "r"(sel_words[0]), "r"(sel_words[1]), "r"(sel_words[2]), "r"(sel_words[3]) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_sel_addr + (unsigned int)((672 + lane) * 16)), "r"(sel_words[0]), "r"(sel_words[1]), "r"(sel_words[2]), "r"(sel_words[3]) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_sel_addr + (unsigned int)((704 + lane) * 16)), "r"(sel_words[0]), "r"(sel_words[1]), "r"(sel_words[2]), "r"(sel_words[3]) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_sel_addr + (unsigned int)((736 + lane) * 16)), "r"(sel_words[0]), "r"(sel_words[1]), "r"(sel_words[2]), "r"(sel_words[3]) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_sel_addr + (unsigned int)((768 + lane) * 16)), "r"(sel_words[0]), "r"(sel_words[1]), "r"(sel_words[2]), "r"(sel_words[3]) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_sel_addr + (unsigned int)((800 + lane) * 16)), "r"(sel_words[0]), "r"(sel_words[1]), "r"(sel_words[2]), "r"(sel_words[3]) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_sel_addr + (unsigned int)((832 + lane) * 16)), "r"(sel_words[0]), "r"(sel_words[1]), "r"(sel_words[2]), "r"(sel_words[3]) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_sel_addr + (unsigned int)((864 + lane) * 16)), "r"(sel_words[0]), "r"(sel_words[1]), "r"(sel_words[2]), "r"(sel_words[3]) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_sel_addr + (unsigned int)((896 + lane) * 16)), "r"(sel_words[0]), "r"(sel_words[1]), "r"(sel_words[2]), "r"(sel_words[3]) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_sel_addr + (unsigned int)((928 + lane) * 16)), "r"(sel_words[0]), "r"(sel_words[1]), "r"(sel_words[2]), "r"(sel_words[3]) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_sel_addr + (unsigned int)((960 + lane) * 16)), "r"(sel_words[0]), "r"(sel_words[1]), "r"(sel_words[2]), "r"(sel_words[3]) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_sel_addr + (unsigned int)((992 + lane) * 16)), "r"(sel_words[0]), "r"(sel_words[1]), "r"(sel_words[2]), "r"(sel_words[3]) : "memory");
        __syncwarp();
        unsigned int sel_rank = (unsigned int)cta_rank;
        sel_words[0] = (sel_rank + 1 & 1) * 16256;
        sel_words[1] = sel_rank * 16256;
        sel_words[2] = (sel_rank + 1 & 1) * 16256;
        sel_words[3] = sel_rank * 16256;
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_sel_addr + (unsigned int)(lane * 4 * 32 ^ (lane * 4 * 32 >> 7 & 1) << 4))), "r"(sel_words[0]), "r"(sel_words[1]), "r"(sel_words[2]), "r"(sel_words[3]) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_sel_addr + (unsigned int)((lane * 4 + 1) * 32 ^ ((lane * 4 + 1) * 32 >> 7 & 1) << 4))), "r"(sel_words[0]), "r"(sel_words[1]), "r"(sel_words[2]), "r"(sel_words[3]) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_sel_addr + (unsigned int)((lane * 4 + 2) * 32 ^ ((lane * 4 + 2) * 32 >> 7 & 1) << 4))), "r"(sel_words[0]), "r"(sel_words[1]), "r"(sel_words[2]), "r"(sel_words[3]) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_sel_addr + (unsigned int)((lane * 4 + 3) * 32 ^ ((lane * 4 + 3) * 32 >> 7 & 1) << 4))), "r"(sel_words[0]), "r"(sel_words[1]), "r"(sel_words[2]), "r"(sel_words[3]) : "memory");
        sel_words[0] = (sel_rank + 1 & 1) * 1065353216;
        sel_words[1] = sel_rank * 1065353216;
        sel_words[2] = (sel_rank + 1 & 1) * 1065353216;
        sel_words[3] = sel_rank * 1065353216;
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_sel_addr + 4096 + (unsigned int)(lane * 4 * 32 ^ (lane * 4 * 32 >> 7 & 1) << 4))), "r"(sel_words[0]), "r"(sel_words[1]), "r"(sel_words[2]), "r"(sel_words[3]) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_sel_addr + 4096 + (unsigned int)((lane * 4 + 1) * 32 ^ ((lane * 4 + 1) * 32 >> 7 & 1) << 4))), "r"(sel_words[0]), "r"(sel_words[1]), "r"(sel_words[2]), "r"(sel_words[3]) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_sel_addr + 4096 + (unsigned int)((lane * 4 + 2) * 32 ^ ((lane * 4 + 2) * 32 >> 7 & 1) << 4))), "r"(sel_words[0]), "r"(sel_words[1]), "r"(sel_words[2]), "r"(sel_words[3]) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_sel_addr + 4096 + (unsigned int)((lane * 4 + 3) * 32 ^ ((lane * 4 + 3) * 32 >> 7 & 1) << 4))), "r"(sel_words[0]), "r"(sel_words[1]), "r"(sel_words[2]), "r"(sel_words[3]) : "memory");
        sel_words[0] = (sel_rank + 1 & 1) * 16256;
        sel_words[1] = sel_rank * 16256;
        sel_words[2] = (sel_rank + 1 & 1) * 16256;
        sel_words[3] = sel_rank * 16256;
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_sel_addr + 8192 + (unsigned int)(lane * 4 * 32 + 16 ^ (lane * 4 * 32 + 16 >> 7 & 1) << 4))), "r"(sel_words[0]), "r"(sel_words[1]), "r"(sel_words[2]), "r"(sel_words[3]) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_sel_addr + 8192 + (unsigned int)((lane * 4 + 1) * 32 + 16 ^ ((lane * 4 + 1) * 32 + 16 >> 7 & 1) << 4))), "r"(sel_words[0]), "r"(sel_words[1]), "r"(sel_words[2]), "r"(sel_words[3]) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_sel_addr + 8192 + (unsigned int)((lane * 4 + 2) * 32 + 16 ^ ((lane * 4 + 2) * 32 + 16 >> 7 & 1) << 4))), "r"(sel_words[0]), "r"(sel_words[1]), "r"(sel_words[2]), "r"(sel_words[3]) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_sel_addr + 8192 + (unsigned int)((lane * 4 + 3) * 32 + 16 ^ ((lane * 4 + 3) * 32 + 16 >> 7 & 1) << 4))), "r"(sel_words[0]), "r"(sel_words[1]), "r"(sel_words[2]), "r"(sel_words[3]) : "memory");
        sel_words[0] = (sel_rank + 1 & 1) * 1065353216;
        sel_words[1] = sel_rank * 1065353216;
        sel_words[2] = (sel_rank + 1 & 1) * 1065353216;
        sel_words[3] = sel_rank * 1065353216;
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_sel_addr + 12288 + (unsigned int)(lane * 4 * 32 + 16 ^ (lane * 4 * 32 + 16 >> 7 & 1) << 4))), "r"(sel_words[0]), "r"(sel_words[1]), "r"(sel_words[2]), "r"(sel_words[3]) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_sel_addr + 12288 + (unsigned int)((lane * 4 + 1) * 32 + 16 ^ ((lane * 4 + 1) * 32 + 16 >> 7 & 1) << 4))), "r"(sel_words[0]), "r"(sel_words[1]), "r"(sel_words[2]), "r"(sel_words[3]) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_sel_addr + 12288 + (unsigned int)((lane * 4 + 2) * 32 + 16 ^ ((lane * 4 + 2) * 32 + 16 >> 7 & 1) << 4))), "r"(sel_words[0]), "r"(sel_words[1]), "r"(sel_words[2]), "r"(sel_words[3]) : "memory");
        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_sel_addr + 12288 + (unsigned int)((lane * 4 + 3) * 32 + 16 ^ ((lane * 4 + 3) * 32 + 16 >> 7 & 1) << 4))), "r"(sel_words[0]), "r"(sel_words[1]), "r"(sel_words[2]), "r"(sel_words[3]) : "memory");
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        __syncwarp();
    }

    // Mbarrier init (15 pipeline groups, 0 ordered-sequence groups, 39 barriers)
    // Mbarriers at smem_raw[0..312)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 2 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            mbarrier_init(smem + 8, 2);
            // q_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            // --- pipeline 'k_pipe' ---
            // k_full: 4 barriers, init_count=2
            mbarrier_init(smem + 24, 2);
            mbarrier_init(smem + 32, 2);
            mbarrier_init(smem + 40, 2);
            mbarrier_init(smem + 48, 2);
            // k_empty: 4 barriers, init_count=1
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            // --- pipeline 'v_pipe' ---
            // v_full: 5 barriers, init_count=2
            mbarrier_init(smem + 88, 2);
            mbarrier_init(smem + 96, 2);
            mbarrier_init(smem + 104, 2);
            mbarrier_init(smem + 112, 2);
            mbarrier_init(smem + 120, 2);
            // v_empty: 5 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 168, 1);
            mbarrier_init(smem + 176, 1);
            // s_empty: 2 barriers, init_count=256
            mbarrier_init(smem + 184, 256);
            mbarrier_init(smem + 192, 256);
            // p_full: 2 barriers, init_count=512
            mbarrier_init(smem + 200, 512);
            mbarrier_init(smem + 208, 512);
            // p_full_2: 2 barriers, init_count=256
            mbarrier_init(smem + 216, 256);
            mbarrier_init(smem + 224, 256);
            // p_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 232, 1);
            mbarrier_init(smem + 240, 1);
            // corr_sig_lo: 2 barriers, init_count=64
            mbarrier_init(smem + 248, 64);
            mbarrier_init(smem + 256, 64);
            // corr_sig_hi: 2 barriers, init_count=64
            mbarrier_init(smem + 264, 64);
            mbarrier_init(smem + 272, 64);
            // corr_done: 2 barriers, init_count=128
            mbarrier_init(smem + 280, 128);
            mbarrier_init(smem + 288, 128);
            // o_full: 2 barriers, init_count=1
            mbarrier_init(smem + 296, 1);
            mbarrier_init(smem + 304, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 472 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 312);
    if (warp == 0) {
        int _tmem_hold = smem + 312;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_scores = taddr;
    const int tmem_output_0 = taddr + 128;
    const int tmem_output_1 = taddr + 256;
    const int tmem_tmem_sfa_qk0 = taddr + 384;
    const int tmem_tmem_sfb_qk0 = taddr + 392;
    const int tmem_tmem_sfa_qk1 = taddr + 400;
    const int tmem_tmem_sfa_pv0_lo = taddr + 408;
    const int tmem_tmem_sfa_pv0_hi = taddr + 412;
    const int tmem_tmem_sfa_pv1_lo = taddr + 416;
    const int tmem_tmem_sfa_pv1_hi = taddr + 420;
    const int tmem_tmem_sfb_pv_lo = taddr + 424;
    const int tmem_tmem_sfb_pv_hi = taddr + 428;
    const int tmem_tmem_sfa_ql0 = taddr + 448;
    const int tmem_tmem_sfa_ql1 = taddr + 456;
    const int tmem_tmem_sfb_qkl = taddr + 464;
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (warp >= 8 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
    }

    // ---- Role: softmax ----
    if (warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 208;");
        { // softmax_main
            unsigned int total_tiles = (seqlen_q + 512 - 1) / 512 * total_bh;
            unsigned int n_full_u = (unsigned int)n_full_tiles;
            unsigned int n_items = n_full_u + (total_tiles - n_full_u) * (unsigned int)n_chunks;
            unsigned int stage = make_warp_uniform(warp / 4);
            int scale_off = make_warp_uniform(stage * 128);
            float neg_c = -softmax_scale_log2;
            int pv_sfa_lo_col = (unsigned int)408 + stage * 8 + (unsigned int)(warp % 4);
            int pv_sfa_hi_col = (unsigned int)412 + stage * 8 + (unsigned int)(warp % 4);
            unsigned int _phase_s_full = 0;
            unsigned int _phase_p_empty = 1;
            unsigned int _phase_corr_done = 0;
            unsigned int _phase_o_full = 0;
            #pragma unroll 1
            for (unsigned int tile_idx = cluster_id; tile_idx < n_items; tile_idx += num_clusters) {
                unsigned int nkb_all = (seqlen_kv + 128 - 1) / 128;
                unsigned int n_full_u_0 = (unsigned int)n_full_tiles;
                unsigned int n_chunks_u = (unsigned int)n_chunks;
                unsigned int _max_0 = ((tile_idx) > (n_full_u_0) ? (tile_idx) : (n_full_u_0));
                unsigned int split_idx = _max_0 - n_full_u_0;
                unsigned int chunk = split_idx - split_idx / n_chunks_u * n_chunks_u;
                unsigned int _min_1 = ((tile_idx) < (n_full_u_0) ? (tile_idx) : (n_full_u_0));
                unsigned int tile_idx_1 = _min_1 + split_idx / n_chunks_u;
                unsigned int _min_2 = ((tile_idx) < (n_full_u_0) ? (tile_idx) : (n_full_u_0));
                unsigned int whole = ((n_full_u_0 - _min_2) < (1) ? (n_full_u_0 - _min_2) : (1));
                unsigned int n_lo = chunk * nkb_all / n_chunks_u;
                unsigned int n_hi = (((chunk + 1) * nkb_all / n_chunks_u) > (nkb_all * whole) ? ((chunk + 1) * nkb_all / n_chunks_u) : (nkb_all * whole));
                unsigned int cluster_m_block = tile_idx_1 % (unsigned int)((seqlen_q + 512 - 1) / 512);
                unsigned int bh = tile_idx_1 / (unsigned int)((seqlen_q + 512 - 1) / 512);
                unsigned int m_block = cluster_m_block * 4 + (unsigned int)(cta_rank * 2);
                unsigned int num_n_blocks = n_hi - n_lo;
                float row_max = -CUDART_INF_F;
                float row_max_scaled = 0.0f;
                float row_sum = 0.0f;
                int n_last = (int)n_hi - 1;
                int n_first = (int)n_lo;
                #pragma unroll 1
                for (int n_block = n_last; n_block >= n_first; n_block--) {
                    mbarrier_wait_hint(s_full_addr + (stage) * 8, _phase_s_full, 10000000);
                    _phase_s_full ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int s_addr = taddr + (unsigned int)0 + (unsigned int)(warp % 4 * 32 << 16);
                    float sv[128];
                    float tile_max = -CUDART_INF_F;
                    float lo_max = -CUDART_INF_F;
                    float hi_max = -CUDART_INF_F;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                    #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                    #endif
                    asm volatile(
                        "tcgen05.ld.red.sync.aligned.32x32b.x64.max.f32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, [%65];"
                        : "=f"(sv[0]), "=f"(sv[1]), "=f"(sv[2]), "=f"(sv[3]), "=f"(sv[4]), "=f"(sv[5]), "=f"(sv[6]), "=f"(sv[7]), "=f"(sv[8]), "=f"(sv[9]), "=f"(sv[10]), "=f"(sv[11]), "=f"(sv[12]), "=f"(sv[13]), "=f"(sv[14]), "=f"(sv[15]), "=f"(sv[16]), "=f"(sv[17]), "=f"(sv[18]), "=f"(sv[19]), "=f"(sv[20]), "=f"(sv[21]), "=f"(sv[22]), "=f"(sv[23]), "=f"(sv[24]), "=f"(sv[25]), "=f"(sv[26]), "=f"(sv[27]), "=f"(sv[28]), "=f"(sv[29]), "=f"(sv[30]), "=f"(sv[31]), "=f"(sv[32]), "=f"(sv[33]), "=f"(sv[34]), "=f"(sv[35]), "=f"(sv[36]), "=f"(sv[37]), "=f"(sv[38]), "=f"(sv[39]), "=f"(sv[40]), "=f"(sv[41]), "=f"(sv[42]), "=f"(sv[43]), "=f"(sv[44]), "=f"(sv[45]), "=f"(sv[46]), "=f"(sv[47]), "=f"(sv[48]), "=f"(sv[49]), "=f"(sv[50]), "=f"(sv[51]), "=f"(sv[52]), "=f"(sv[53]), "=f"(sv[54]), "=f"(sv[55]), "=f"(sv[56]), "=f"(sv[57]), "=f"(sv[58]), "=f"(sv[59]), "=f"(sv[60]), "=f"(sv[61]), "=f"(sv[62]), "=f"(sv[63]), "=f"(lo_max)
                        : "r"(tmem_scores + (warp % 4 * 32 << 16)));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                    #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                    #endif
                    asm volatile(
                        "tcgen05.ld.red.sync.aligned.32x32b.x64.max.f32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, [%65];"
                        : "=f"(sv[64]), "=f"(sv[65]), "=f"(sv[66]), "=f"(sv[67]), "=f"(sv[68]), "=f"(sv[69]), "=f"(sv[70]), "=f"(sv[71]), "=f"(sv[72]), "=f"(sv[73]), "=f"(sv[74]), "=f"(sv[75]), "=f"(sv[76]), "=f"(sv[77]), "=f"(sv[78]), "=f"(sv[79]), "=f"(sv[80]), "=f"(sv[81]), "=f"(sv[82]), "=f"(sv[83]), "=f"(sv[84]), "=f"(sv[85]), "=f"(sv[86]), "=f"(sv[87]), "=f"(sv[88]), "=f"(sv[89]), "=f"(sv[90]), "=f"(sv[91]), "=f"(sv[92]), "=f"(sv[93]), "=f"(sv[94]), "=f"(sv[95]), "=f"(sv[96]), "=f"(sv[97]), "=f"(sv[98]), "=f"(sv[99]), "=f"(sv[100]), "=f"(sv[101]), "=f"(sv[102]), "=f"(sv[103]), "=f"(sv[104]), "=f"(sv[105]), "=f"(sv[106]), "=f"(sv[107]), "=f"(sv[108]), "=f"(sv[109]), "=f"(sv[110]), "=f"(sv[111]), "=f"(sv[112]), "=f"(sv[113]), "=f"(sv[114]), "=f"(sv[115]), "=f"(sv[116]), "=f"(sv[117]), "=f"(sv[118]), "=f"(sv[119]), "=f"(sv[120]), "=f"(sv[121]), "=f"(sv[122]), "=f"(sv[123]), "=f"(sv[124]), "=f"(sv[125]), "=f"(sv[126]), "=f"(sv[127]), "=f"(hi_max)
                        : "r"(tmem_scores + 64 + (warp % 4 * 32 << 16)));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    tile_max = max_noftz(lo_max, hi_max);
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((s_empty_addr + (stage) * 8) & 0xFEFFFFFF) : "memory");
                    int tail_valid = seqlen_kv - n_block * 128;
                    if (tail_valid < 128) {
                        uint32_t _thresh_mask_3;
                        {
                            int _lim_4 = tail_valid;
                            if (_lim_4 <= 0) { _thresh_mask_3 = 0u; }
                            else if (_lim_4 >= 32) { _thresh_mask_3 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_thresh_mask_3) : "r"(_lim_4));
                            }
                        }
                        if (!(_thresh_mask_3 & (1u << 0))) sv[0] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 1))) sv[1] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 2))) sv[2] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 3))) sv[3] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 4))) sv[4] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 5))) sv[5] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 6))) sv[6] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 7))) sv[7] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 8))) sv[8] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 9))) sv[9] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 10))) sv[10] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 11))) sv[11] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 12))) sv[12] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 13))) sv[13] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 14))) sv[14] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 15))) sv[15] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 16))) sv[16] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 17))) sv[17] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 18))) sv[18] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 19))) sv[19] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 20))) sv[20] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 21))) sv[21] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 22))) sv[22] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 23))) sv[23] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 24))) sv[24] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 25))) sv[25] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 26))) sv[26] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 27))) sv[27] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 28))) sv[28] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 29))) sv[29] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 30))) sv[30] = -CUDART_INF_F;
                        if (!(_thresh_mask_3 & (1u << 31))) sv[31] = -CUDART_INF_F;
                        uint32_t _thresh_mask_5;
                        {
                            int _lim_6 = tail_valid - 32;
                            if (_lim_6 <= 0) { _thresh_mask_5 = 0u; }
                            else if (_lim_6 >= 32) { _thresh_mask_5 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_thresh_mask_5) : "r"(_lim_6));
                            }
                        }
                        if (!(_thresh_mask_5 & (1u << 0))) sv[32] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 1))) sv[33] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 2))) sv[34] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 3))) sv[35] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 4))) sv[36] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 5))) sv[37] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 6))) sv[38] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 7))) sv[39] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 8))) sv[40] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 9))) sv[41] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 10))) sv[42] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 11))) sv[43] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 12))) sv[44] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 13))) sv[45] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 14))) sv[46] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 15))) sv[47] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 16))) sv[48] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 17))) sv[49] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 18))) sv[50] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 19))) sv[51] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 20))) sv[52] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 21))) sv[53] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 22))) sv[54] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 23))) sv[55] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 24))) sv[56] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 25))) sv[57] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 26))) sv[58] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 27))) sv[59] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 28))) sv[60] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 29))) sv[61] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 30))) sv[62] = -CUDART_INF_F;
                        if (!(_thresh_mask_5 & (1u << 31))) sv[63] = -CUDART_INF_F;
                        uint32_t _thresh_mask_7;
                        {
                            int _lim_8 = tail_valid - 64;
                            if (_lim_8 <= 0) { _thresh_mask_7 = 0u; }
                            else if (_lim_8 >= 32) { _thresh_mask_7 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_thresh_mask_7) : "r"(_lim_8));
                            }
                        }
                        if (!(_thresh_mask_7 & (1u << 0))) sv[64] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 1))) sv[65] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 2))) sv[66] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 3))) sv[67] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 4))) sv[68] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 5))) sv[69] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 6))) sv[70] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 7))) sv[71] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 8))) sv[72] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 9))) sv[73] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 10))) sv[74] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 11))) sv[75] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 12))) sv[76] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 13))) sv[77] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 14))) sv[78] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 15))) sv[79] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 16))) sv[80] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 17))) sv[81] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 18))) sv[82] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 19))) sv[83] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 20))) sv[84] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 21))) sv[85] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 22))) sv[86] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 23))) sv[87] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 24))) sv[88] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 25))) sv[89] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 26))) sv[90] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 27))) sv[91] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 28))) sv[92] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 29))) sv[93] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 30))) sv[94] = -CUDART_INF_F;
                        if (!(_thresh_mask_7 & (1u << 31))) sv[95] = -CUDART_INF_F;
                        uint32_t _thresh_mask_9;
                        {
                            int _lim_10 = tail_valid - 96;
                            if (_lim_10 <= 0) { _thresh_mask_9 = 0u; }
                            else if (_lim_10 >= 32) { _thresh_mask_9 = 0xFFFFFFFFu; }
                            else {
                                asm volatile("{"
                                    ".reg .u32 t;\n\t"
                                    "shl.b32 t, 1, %1;\n\t"
                                    "add.u32 %0, t, -1;\n\t"
                                    "}" : "=r"(_thresh_mask_9) : "r"(_lim_10));
                            }
                        }
                        if (!(_thresh_mask_9 & (1u << 0))) sv[96] = -CUDART_INF_F;
                        if (!(_thresh_mask_9 & (1u << 1))) sv[97] = -CUDART_INF_F;
                        if (!(_thresh_mask_9 & (1u << 2))) sv[98] = -CUDART_INF_F;
                        if (!(_thresh_mask_9 & (1u << 3))) sv[99] = -CUDART_INF_F;
                        if (!(_thresh_mask_9 & (1u << 4))) sv[100] = -CUDART_INF_F;
                        if (!(_thresh_mask_9 & (1u << 5))) sv[101] = -CUDART_INF_F;
                        if (!(_thresh_mask_9 & (1u << 6))) sv[102] = -CUDART_INF_F;
                        if (!(_thresh_mask_9 & (1u << 7))) sv[103] = -CUDART_INF_F;
                        if (!(_thresh_mask_9 & (1u << 8))) sv[104] = -CUDART_INF_F;
                        if (!(_thresh_mask_9 & (1u << 9))) sv[105] = -CUDART_INF_F;
                        if (!(_thresh_mask_9 & (1u << 10))) sv[106] = -CUDART_INF_F;
                        if (!(_thresh_mask_9 & (1u << 11))) sv[107] = -CUDART_INF_F;
                        if (!(_thresh_mask_9 & (1u << 12))) sv[108] = -CUDART_INF_F;
                        if (!(_thresh_mask_9 & (1u << 13))) sv[109] = -CUDART_INF_F;
                        if (!(_thresh_mask_9 & (1u << 14))) sv[110] = -CUDART_INF_F;
                        if (!(_thresh_mask_9 & (1u << 15))) sv[111] = -CUDART_INF_F;
                        if (!(_thresh_mask_9 & (1u << 16))) sv[112] = -CUDART_INF_F;
                        if (!(_thresh_mask_9 & (1u << 17))) sv[113] = -CUDART_INF_F;
                        if (!(_thresh_mask_9 & (1u << 18))) sv[114] = -CUDART_INF_F;
                        if (!(_thresh_mask_9 & (1u << 19))) sv[115] = -CUDART_INF_F;
                        if (!(_thresh_mask_9 & (1u << 20))) sv[116] = -CUDART_INF_F;
                        if (!(_thresh_mask_9 & (1u << 21))) sv[117] = -CUDART_INF_F;
                        if (!(_thresh_mask_9 & (1u << 22))) sv[118] = -CUDART_INF_F;
                        if (!(_thresh_mask_9 & (1u << 23))) sv[119] = -CUDART_INF_F;
                        if (!(_thresh_mask_9 & (1u << 24))) sv[120] = -CUDART_INF_F;
                        if (!(_thresh_mask_9 & (1u << 25))) sv[121] = -CUDART_INF_F;
                        if (!(_thresh_mask_9 & (1u << 26))) sv[122] = -CUDART_INF_F;
                        if (!(_thresh_mask_9 & (1u << 27))) sv[123] = -CUDART_INF_F;
                        if (!(_thresh_mask_9 & (1u << 28))) sv[124] = -CUDART_INF_F;
                        if (!(_thresh_mask_9 & (1u << 29))) sv[125] = -CUDART_INF_F;
                        if (!(_thresh_mask_9 & (1u << 30))) sv[126] = -CUDART_INF_F;
                        if (!(_thresh_mask_9 & (1u << 31))) sv[127] = -CUDART_INF_F;
                    }
                    float bms[8];
                    float _reg_array_reduce_11 = sv[0];
                    #pragma unroll
                    for (int _lr = 1; _lr < 16; _lr++) {
                        _reg_array_reduce_11 = max_noftz(_reg_array_reduce_11, sv[_lr]);
                    }
                    bms[0] = _reg_array_reduce_11;
                    float _reg_array_reduce_12 = sv[(16) + 0];
                    #pragma unroll
                    for (int _lr = 1; _lr < 16; _lr++) {
                        _reg_array_reduce_12 = max_noftz(_reg_array_reduce_12, sv[(16) + _lr]);
                    }
                    bms[1] = _reg_array_reduce_12;
                    float _reg_array_reduce_13 = sv[(32) + 0];
                    #pragma unroll
                    for (int _lr = 1; _lr < 16; _lr++) {
                        _reg_array_reduce_13 = max_noftz(_reg_array_reduce_13, sv[(32) + _lr]);
                    }
                    bms[2] = _reg_array_reduce_13;
                    float _reg_array_reduce_14 = sv[(48) + 0];
                    #pragma unroll
                    for (int _lr = 1; _lr < 16; _lr++) {
                        _reg_array_reduce_14 = max_noftz(_reg_array_reduce_14, sv[(48) + _lr]);
                    }
                    bms[3] = _reg_array_reduce_14;
                    float _reg_array_reduce_15 = sv[(64) + 0];
                    #pragma unroll
                    for (int _lr = 1; _lr < 16; _lr++) {
                        _reg_array_reduce_15 = max_noftz(_reg_array_reduce_15, sv[(64) + _lr]);
                    }
                    bms[4] = _reg_array_reduce_15;
                    float _reg_array_reduce_16 = sv[(80) + 0];
                    #pragma unroll
                    for (int _lr = 1; _lr < 16; _lr++) {
                        _reg_array_reduce_16 = max_noftz(_reg_array_reduce_16, sv[(80) + _lr]);
                    }
                    bms[5] = _reg_array_reduce_16;
                    float _reg_array_reduce_17 = sv[(96) + 0];
                    #pragma unroll
                    for (int _lr = 1; _lr < 16; _lr++) {
                        _reg_array_reduce_17 = max_noftz(_reg_array_reduce_17, sv[(96) + _lr]);
                    }
                    bms[6] = _reg_array_reduce_17;
                    float _reg_array_reduce_18 = sv[(112) + 0];
                    #pragma unroll
                    for (int _lr = 1; _lr < 16; _lr++) {
                        _reg_array_reduce_18 = max_noftz(_reg_array_reduce_18, sv[(112) + _lr]);
                    }
                    bms[7] = _reg_array_reduce_18;
                    if (tail_valid < 128) {
                        float _reg_array_reduce_19 = bms[0];
                        #pragma unroll
                        for (int _lr = 1; _lr < 8; _lr++) {
                            _reg_array_reduce_19 = max_noftz(_reg_array_reduce_19, bms[_lr]);
                        }
                        tile_max = _reg_array_reduce_19;
                    }
                    float new_max;
                    asm("max.f32 %0, %1, %2;" : "=f"(new_max) : "f"(tile_max), "f"(row_max));
                    float new_max_scaled = new_max * softmax_scale_log2;
                    float acc_scale_log2 = __fmaf_rn(row_max, softmax_scale_log2, -new_max_scaled);
                    float acc_scale;
                    if (acc_scale_log2 >= 0.0f) {
                        acc_scale = 1.0f;
                        new_max_scaled = row_max_scaled;
                    } else {
                        float _exp2_20 = approx_exp2(acc_scale_log2);
                        acc_scale = ((row_max > -CUDART_INF_F) ? _exp2_20 : 1.0f);
                        row_max = new_max;
                        row_max_scaled = new_max_scaled;
                    }
                    row_state[warp % 4 * 32 + lane + scale_off] = acc_scale;
                    if (warp % 4 < 2) {
                        mbarrier_arrive(corr_sig_lo_addr + (stage) * 8);
                    } else {
                        mbarrier_arrive(corr_sig_hi_addr + (stage) * 8);
                    }
                    mbarrier_wait(p_empty_addr + (stage) * 8, _phase_p_empty);
                    _phase_p_empty ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    float bias = 11.392318f - new_max_scaled;
                    float2 tot2 = make_float2(0.0f, 0.0f);
                    float sf_values[4];
                    float bm0 = approx_exp2(__fmaf_rn(bms[0], softmax_scale_log2, bias));
                    float sf_f0 = bm0 * 0.16666667f;
                    sf_values[0] = sf_f0;
                    float bb0 = ((bms[0] > -CUDART_INF_F) ? __fmaf_rn(bms[0], neg_c, 2.5849626f) : 0.0f);
                    const float2 _fma_b2_21 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_22 = {bb0, bb0};
                    #pragma unroll
                    for (int _lf = 0; _lf < 8; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 0))[_lf], _fma_b2_21, _fma_c2_22);
                    #pragma unroll
                    for (int _le = 0; _le < 16; _le++) {
                        sv[_le] = approx_exp2(sv[_le]);
                    }
                    float2 partial = make_float2(sv[0], sv[1]);
                    #pragma unroll
                    for (int pair = 2; pair < 16; pair += 2) {
                        partial = add_f32x2(partial, make_float2(sv[pair], sv[pair + 1]));
                    }
                    float2 frag_sum2_0 = partial;
                    float2 scale2_0 = make_float2(sf_f0, sf_f0);
                    tot2 = fma_f32x2_rn_ftz(frag_sum2_0, scale2_0, tot2);
                    float bm1 = approx_exp2(__fmaf_rn(bms[1], softmax_scale_log2, bias));
                    float sf_f1 = bm1 * 0.16666667f;
                    sf_values[1] = sf_f1;
                    float bb1 = ((bms[1] > -CUDART_INF_F) ? __fmaf_rn(bms[1], neg_c, 2.5849626f) : 0.0f);
                    const float2 _fma_b2_23 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_24 = {bb1, bb1};
                    #pragma unroll
                    for (int _lf = 0; _lf < 8; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 16))[_lf], _fma_b2_23, _fma_c2_24);
                    #pragma unroll
                    for (int _le = 0; _le < 16; _le++) {
                        sv[_le + 16] = approx_exp2(sv[_le + 16]);
                    }
                    float2 partial_0 = make_float2(sv[16], sv[17]);
                    #pragma unroll
                    for (int pair_1 = 2; pair_1 < 16; pair_1 += 2) {
                        partial_0 = add_f32x2(partial_0, make_float2(sv[16 + pair_1], sv[16 + (pair_1 + 1)]));
                    }
                    float2 frag_sum2_1 = partial_0;
                    float2 scale2_1 = make_float2(sf_f1, sf_f1);
                    tot2 = fma_f32x2_rn_ftz(frag_sum2_1, scale2_1, tot2);
                    unsigned int _fp4_0[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_0[0]) : "f"(sv[0]), "f"(sv[1]), "f"(sv[2]), "f"(sv[3]), "f"(sv[4]), "f"(sv[5]), "f"(sv[6]), "f"(sv[7]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_0[1]) : "f"(sv[8]), "f"(sv[9]), "f"(sv[10]), "f"(sv[11]), "f"(sv[12]), "f"(sv[13]), "f"(sv[14]), "f"(sv[15]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_0[2]) : "f"(sv[16]), "f"(sv[17]), "f"(sv[18]), "f"(sv[19]), "f"(sv[20]), "f"(sv[21]), "f"(sv[22]), "f"(sv[23]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_0[3]) : "f"(sv[24]), "f"(sv[25]), "f"(sv[26]), "f"(sv[27]), "f"(sv[28]), "f"(sv[29]), "f"(sv[30]), "f"(sv[31]));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_p_addr + stage * 8192 + (unsigned int)((warp % 4 * 32 + lane) * 64 ^ ((warp % 4 * 32 + lane) * 64 >> 7 & 3) << 4))), "r"(_fp4_0[0]), "r"(_fp4_0[1]), "r"(_fp4_0[2]), "r"(_fp4_0[3]) : "memory");
                    float bm2 = approx_exp2(__fmaf_rn(bms[2], softmax_scale_log2, bias));
                    float sf_f2 = bm2 * 0.16666667f;
                    sf_values[2] = sf_f2;
                    float bb2 = ((bms[2] > -CUDART_INF_F) ? __fmaf_rn(bms[2], neg_c, 2.5849626f) : 0.0f);
                    const float2 _fma_b2_25 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_26 = {bb2, bb2};
                    #pragma unroll
                    for (int _lf = 0; _lf < 8; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 32))[_lf], _fma_b2_25, _fma_c2_26);
                    #pragma unroll
                    for (int _le = 0; _le < 16; _le++) {
                        sv[_le + 32] = approx_exp2(sv[_le + 32]);
                    }
                    float2 partial_1 = make_float2(sv[32], sv[33]);
                    #pragma unroll
                    for (int pair_2 = 2; pair_2 < 16; pair_2 += 2) {
                        partial_1 = add_f32x2(partial_1, make_float2(sv[32 + pair_2], sv[32 + (pair_2 + 1)]));
                    }
                    float2 frag_sum2_2 = partial_1;
                    float2 scale2_2 = make_float2(sf_f2, sf_f2);
                    tot2 = fma_f32x2_rn_ftz(frag_sum2_2, scale2_2, tot2);
                    float bm3 = approx_exp2(__fmaf_rn(bms[3], softmax_scale_log2, bias));
                    float sf_f3 = bm3 * 0.16666667f;
                    sf_values[3] = sf_f3;
                    float bb3 = ((bms[3] > -CUDART_INF_F) ? __fmaf_rn(bms[3], neg_c, 2.5849626f) : 0.0f);
                    const float2 _fma_b2_27 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_28 = {bb3, bb3};
                    #pragma unroll
                    for (int _lf = 0; _lf < 8; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 48))[_lf], _fma_b2_27, _fma_c2_28);
                    #pragma unroll
                    for (int _le = 0; _le < 16; _le++) {
                        sv[_le + 48] = approx_exp2(sv[_le + 48]);
                    }
                    float2 partial_2 = make_float2(sv[48], sv[49]);
                    #pragma unroll
                    for (int pair_3 = 2; pair_3 < 16; pair_3 += 2) {
                        partial_2 = add_f32x2(partial_2, make_float2(sv[48 + pair_3], sv[48 + (pair_3 + 1)]));
                    }
                    float2 frag_sum2_3 = partial_2;
                    float2 scale2_3 = make_float2(sf_f3, sf_f3);
                    tot2 = fma_f32x2_rn_ftz(frag_sum2_3, scale2_3, tot2);
                    unsigned int _fp4_1[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_1[0]) : "f"(sv[32]), "f"(sv[33]), "f"(sv[34]), "f"(sv[35]), "f"(sv[36]), "f"(sv[37]), "f"(sv[38]), "f"(sv[39]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_1[1]) : "f"(sv[40]), "f"(sv[41]), "f"(sv[42]), "f"(sv[43]), "f"(sv[44]), "f"(sv[45]), "f"(sv[46]), "f"(sv[47]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_1[2]) : "f"(sv[48]), "f"(sv[49]), "f"(sv[50]), "f"(sv[51]), "f"(sv[52]), "f"(sv[53]), "f"(sv[54]), "f"(sv[55]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_1[3]) : "f"(sv[56]), "f"(sv[57]), "f"(sv[58]), "f"(sv[59]), "f"(sv[60]), "f"(sv[61]), "f"(sv[62]), "f"(sv[63]));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_p_addr + stage * 8192 + (unsigned int)((warp % 4 * 32 + lane) * 64 + 16 ^ ((warp % 4 * 32 + lane) * 64 + 16 >> 7 & 3) << 4))), "r"(_fp4_1[0]), "r"(_fp4_1[1]), "r"(_fp4_1[2]), "r"(_fp4_1[3]) : "memory");
                    unsigned int _fp8_0[1];
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sf_values[0]), "f"(sf_values[1]),
                                               "f"(sf_values[2]), "f"(sf_values[3]));
                        _fp8_0[0] = _packed;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(taddr + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)pv_sfa_lo_col), "r"(_fp8_0[0]));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((p_full_addr + (stage) * 8) & 0xFEFFFFFF) : "memory");
                    float bm4 = approx_exp2(__fmaf_rn(bms[4], softmax_scale_log2, bias));
                    float sf_f4 = bm4 * 0.16666667f;
                    sf_values[0] = sf_f4;
                    float bb4 = ((bms[4] > -CUDART_INF_F) ? __fmaf_rn(bms[4], neg_c, 2.5849626f) : 0.0f);
                    const float2 _fma_b2_29 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_30 = {bb4, bb4};
                    #pragma unroll
                    for (int _lf = 0; _lf < 8; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 64))[_lf], _fma_b2_29, _fma_c2_30);
                    #pragma unroll
                    for (int _le = 0; _le < 16; _le++) {
                        sv[_le + 64] = approx_exp2(sv[_le + 64]);
                    }
                    float2 partial_3 = make_float2(sv[64], sv[65]);
                    #pragma unroll
                    for (int pair_4 = 2; pair_4 < 16; pair_4 += 2) {
                        partial_3 = add_f32x2(partial_3, make_float2(sv[64 + pair_4], sv[64 + (pair_4 + 1)]));
                    }
                    float2 frag_sum2_4 = partial_3;
                    float2 scale2_4 = make_float2(sf_f4, sf_f4);
                    tot2 = fma_f32x2_rn_ftz(frag_sum2_4, scale2_4, tot2);
                    float bm5 = approx_exp2(__fmaf_rn(bms[5], softmax_scale_log2, bias));
                    float sf_f5 = bm5 * 0.16666667f;
                    sf_values[1] = sf_f5;
                    float bb5 = ((bms[5] > -CUDART_INF_F) ? __fmaf_rn(bms[5], neg_c, 2.5849626f) : 0.0f);
                    const float2 _fma_b2_31 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_32 = {bb5, bb5};
                    #pragma unroll
                    for (int _lf = 0; _lf < 8; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 80))[_lf], _fma_b2_31, _fma_c2_32);
                    #pragma unroll
                    for (int _le = 0; _le < 16; _le++) {
                        sv[_le + 80] = approx_exp2(sv[_le + 80]);
                    }
                    float2 partial_4 = make_float2(sv[80], sv[81]);
                    #pragma unroll
                    for (int pair_5 = 2; pair_5 < 16; pair_5 += 2) {
                        partial_4 = add_f32x2(partial_4, make_float2(sv[80 + pair_5], sv[80 + (pair_5 + 1)]));
                    }
                    float2 frag_sum2_5 = partial_4;
                    float2 scale2_5 = make_float2(sf_f5, sf_f5);
                    tot2 = fma_f32x2_rn_ftz(frag_sum2_5, scale2_5, tot2);
                    unsigned int _fp4_2[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_2[0]) : "f"(sv[64]), "f"(sv[65]), "f"(sv[66]), "f"(sv[67]), "f"(sv[68]), "f"(sv[69]), "f"(sv[70]), "f"(sv[71]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_2[1]) : "f"(sv[72]), "f"(sv[73]), "f"(sv[74]), "f"(sv[75]), "f"(sv[76]), "f"(sv[77]), "f"(sv[78]), "f"(sv[79]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_2[2]) : "f"(sv[80]), "f"(sv[81]), "f"(sv[82]), "f"(sv[83]), "f"(sv[84]), "f"(sv[85]), "f"(sv[86]), "f"(sv[87]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_2[3]) : "f"(sv[88]), "f"(sv[89]), "f"(sv[90]), "f"(sv[91]), "f"(sv[92]), "f"(sv[93]), "f"(sv[94]), "f"(sv[95]));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_p_addr + stage * 8192 + (unsigned int)((warp % 4 * 32 + lane) * 64 + 32 ^ ((warp % 4 * 32 + lane) * 64 + 32 >> 7 & 3) << 4))), "r"(_fp4_2[0]), "r"(_fp4_2[1]), "r"(_fp4_2[2]), "r"(_fp4_2[3]) : "memory");
                    float bm6 = approx_exp2(__fmaf_rn(bms[6], softmax_scale_log2, bias));
                    float sf_f6 = bm6 * 0.16666667f;
                    sf_values[2] = sf_f6;
                    float bb6 = ((bms[6] > -CUDART_INF_F) ? __fmaf_rn(bms[6], neg_c, 2.5849626f) : 0.0f);
                    const float2 _fma_b2_33 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_34 = {bb6, bb6};
                    #pragma unroll
                    for (int _lf = 0; _lf < 8; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 96))[_lf], _fma_b2_33, _fma_c2_34);
                    #pragma unroll
                    for (int _le = 0; _le < 16; _le++) {
                        sv[_le + 96] = approx_exp2(sv[_le + 96]);
                    }
                    float2 partial_5 = make_float2(sv[96], sv[97]);
                    #pragma unroll
                    for (int pair_6 = 2; pair_6 < 16; pair_6 += 2) {
                        partial_5 = add_f32x2(partial_5, make_float2(sv[96 + pair_6], sv[96 + (pair_6 + 1)]));
                    }
                    float2 frag_sum2_6 = partial_5;
                    float2 scale2_6 = make_float2(sf_f6, sf_f6);
                    tot2 = fma_f32x2_rn_ftz(frag_sum2_6, scale2_6, tot2);
                    float bm7 = approx_exp2(__fmaf_rn(bms[7], softmax_scale_log2, bias));
                    float sf_f7 = bm7 * 0.16666667f;
                    sf_values[3] = sf_f7;
                    float bb7 = ((bms[7] > -CUDART_INF_F) ? __fmaf_rn(bms[7], neg_c, 2.5849626f) : 0.0f);
                    const float2 _fma_b2_35 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_36 = {bb7, bb7};
                    #pragma unroll
                    for (int _lf = 0; _lf < 8; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 112))[_lf], _fma_b2_35, _fma_c2_36);
                    #pragma unroll
                    for (int _le = 0; _le < 16; _le++) {
                        sv[_le + 112] = approx_exp2(sv[_le + 112]);
                    }
                    float2 partial_6 = make_float2(sv[112], sv[113]);
                    #pragma unroll
                    for (int pair_7 = 2; pair_7 < 16; pair_7 += 2) {
                        partial_6 = add_f32x2(partial_6, make_float2(sv[112 + pair_7], sv[112 + (pair_7 + 1)]));
                    }
                    float2 frag_sum2_7 = partial_6;
                    float2 scale2_7 = make_float2(sf_f7, sf_f7);
                    tot2 = fma_f32x2_rn_ftz(frag_sum2_7, scale2_7, tot2);
                    unsigned int _fp4_3[4];
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_3[0]) : "f"(sv[96]), "f"(sv[97]), "f"(sv[98]), "f"(sv[99]), "f"(sv[100]), "f"(sv[101]), "f"(sv[102]), "f"(sv[103]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_3[1]) : "f"(sv[104]), "f"(sv[105]), "f"(sv[106]), "f"(sv[107]), "f"(sv[108]), "f"(sv[109]), "f"(sv[110]), "f"(sv[111]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_3[2]) : "f"(sv[112]), "f"(sv[113]), "f"(sv[114]), "f"(sv[115]), "f"(sv[116]), "f"(sv[117]), "f"(sv[118]), "f"(sv[119]));
                    asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_3[3]) : "f"(sv[120]), "f"(sv[121]), "f"(sv[122]), "f"(sv[123]), "f"(sv[124]), "f"(sv[125]), "f"(sv[126]), "f"(sv[127]));
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_p_addr + stage * 8192 + (unsigned int)((warp % 4 * 32 + lane) * 64 + 48 ^ ((warp % 4 * 32 + lane) * 64 + 48 >> 7 & 3) << 4))), "r"(_fp4_3[0]), "r"(_fp4_3[1]), "r"(_fp4_3[2]), "r"(_fp4_3[3]) : "memory");
                    unsigned int _fp8_1[1];
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(sf_values[0]), "f"(sf_values[1]),
                                               "f"(sf_values[2]), "f"(sf_values[3]));
                        _fp8_1[0] = _packed;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(taddr + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)pv_sfa_hi_col), "r"(_fp8_1[0]));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((p_full_2_addr + (stage) * 8) & 0xFEFFFFFF) : "memory");
                    float block_sum = tot2.x + tot2.y;
                    row_sum = __fmaf_rn(row_sum, acc_scale, block_sum);
                    mbarrier_wait(corr_done_addr + (stage) * 8, _phase_corr_done);
                    _phase_corr_done ^= 1;
                }
                mbarrier_wait(o_full_addr + (stage) * 8, _phase_o_full);
                _phase_o_full ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float final_scale = 0.0f;
                int split_slot = (int)tile_idx - n_full_tiles;
                if (split_slot < 0) {
                    float _rcp_37 = approx_rcp(row_sum);
                    final_scale = ((row_sum != 0.0f && row_sum == row_sum) ? _rcp_37 : 0.0f);
                    int out_row = (int)((m_block + stage) * 128);
                    int out_head = (int)(bh % (unsigned int)heads);
                    int out_batch = (int)(bh / (unsigned int)heads);
                    if (stage == 0) {
                        if (warp == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                        }
                        asm volatile("barrier.sync 3, 128;" ::: "memory");
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int load_addr = taddr + (unsigned int)128 + (unsigned int)(warp % 4 * 32 << 16);
                        float _tmem_load_0[16];
                        tmem_ld_x16(&_tmem_load_0[0], load_addr);
                        float _tmem_load_1[16];
                        tmem_ld_x16(&_tmem_load_1[0], load_addr + 64);
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        const float2 _scale2_38 = {final_scale, final_scale};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_0)[_ls], _scale2_38);
                        unsigned int _tmem_load_0_bf16[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 8; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                            _tmem_load_0_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o00_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 ^ ((warp % 4 * 32 + lane) * 128 >> 7 & 7) << 4))), "r"(_tmem_load_0_bf16[0]), "r"(_tmem_load_0_bf16[1]), "r"(_tmem_load_0_bf16[2]), "r"(_tmem_load_0_bf16[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o00_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + 16 ^ ((warp % 4 * 32 + lane) * 128 + 16 >> 7 & 7) << 4))), "r"(_tmem_load_0_bf16[4]), "r"(_tmem_load_0_bf16[5]), "r"(_tmem_load_0_bf16[6]), "r"(_tmem_load_0_bf16[7]) : "memory");
                        const float2 _scale2_39 = {final_scale, final_scale};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_1)[_ls], _scale2_39);
                        unsigned int _tmem_load_1_bf16[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 8; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_1[_lp*2 + 0], _tmem_load_1[_lp*2+1 + 0]));
                            _tmem_load_1_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o01_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 ^ ((warp % 4 * 32 + lane) * 128 >> 7 & 7) << 4))), "r"(_tmem_load_1_bf16[0]), "r"(_tmem_load_1_bf16[1]), "r"(_tmem_load_1_bf16[2]), "r"(_tmem_load_1_bf16[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o01_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + 16 ^ ((warp % 4 * 32 + lane) * 128 + 16 >> 7 & 7) << 4))), "r"(_tmem_load_1_bf16[4]), "r"(_tmem_load_1_bf16[5]), "r"(_tmem_load_1_bf16[6]), "r"(_tmem_load_1_bf16[7]) : "memory");
                        int load_addr_0 = taddr + (unsigned int)128 + (unsigned int)(warp % 4 * 32 << 16) + 16;
                        float _tmem_load_2[16];
                        tmem_ld_x16(&_tmem_load_2[0], load_addr_0);
                        float _tmem_load_3[16];
                        tmem_ld_x16(&_tmem_load_3[0], load_addr_0 + 64);
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        const float2 _scale2_40 = {final_scale, final_scale};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_2)[_ls], _scale2_40);
                        unsigned int _tmem_load_2_bf16[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 8; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 0], _tmem_load_2[_lp*2+1 + 0]));
                            _tmem_load_2_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o00_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + 32 ^ ((warp % 4 * 32 + lane) * 128 + 32 >> 7 & 7) << 4))), "r"(_tmem_load_2_bf16[0]), "r"(_tmem_load_2_bf16[1]), "r"(_tmem_load_2_bf16[2]), "r"(_tmem_load_2_bf16[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o00_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + 48 ^ ((warp % 4 * 32 + lane) * 128 + 48 >> 7 & 7) << 4))), "r"(_tmem_load_2_bf16[4]), "r"(_tmem_load_2_bf16[5]), "r"(_tmem_load_2_bf16[6]), "r"(_tmem_load_2_bf16[7]) : "memory");
                        const float2 _scale2_41 = {final_scale, final_scale};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_3)[_ls], _scale2_41);
                        unsigned int _tmem_load_3_bf16[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 8; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 0], _tmem_load_3[_lp*2+1 + 0]));
                            _tmem_load_3_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o01_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + 32 ^ ((warp % 4 * 32 + lane) * 128 + 32 >> 7 & 7) << 4))), "r"(_tmem_load_3_bf16[0]), "r"(_tmem_load_3_bf16[1]), "r"(_tmem_load_3_bf16[2]), "r"(_tmem_load_3_bf16[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o01_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + 48 ^ ((warp % 4 * 32 + lane) * 128 + 48 >> 7 & 7) << 4))), "r"(_tmem_load_3_bf16[4]), "r"(_tmem_load_3_bf16[5]), "r"(_tmem_load_3_bf16[6]), "r"(_tmem_load_3_bf16[7]) : "memory");
                        int load_addr_1 = taddr + (unsigned int)128 + (unsigned int)(warp % 4 * 32 << 16) + 32;
                        float _tmem_load_4[16];
                        tmem_ld_x16(&_tmem_load_4[0], load_addr_1);
                        float _tmem_load_5[16];
                        tmem_ld_x16(&_tmem_load_5[0], load_addr_1 + 64);
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        const float2 _scale2_42 = {final_scale, final_scale};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_4)[_ls], _scale2_42);
                        unsigned int _tmem_load_4_bf16[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 8; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_4[_lp*2 + 0], _tmem_load_4[_lp*2+1 + 0]));
                            _tmem_load_4_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o00_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + 64 ^ ((warp % 4 * 32 + lane) * 128 + 64 >> 7 & 7) << 4))), "r"(_tmem_load_4_bf16[0]), "r"(_tmem_load_4_bf16[1]), "r"(_tmem_load_4_bf16[2]), "r"(_tmem_load_4_bf16[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o00_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + 80 ^ ((warp % 4 * 32 + lane) * 128 + 80 >> 7 & 7) << 4))), "r"(_tmem_load_4_bf16[4]), "r"(_tmem_load_4_bf16[5]), "r"(_tmem_load_4_bf16[6]), "r"(_tmem_load_4_bf16[7]) : "memory");
                        const float2 _scale2_43 = {final_scale, final_scale};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_5)[_ls], _scale2_43);
                        unsigned int _tmem_load_5_bf16[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 8; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_5[_lp*2 + 0], _tmem_load_5[_lp*2+1 + 0]));
                            _tmem_load_5_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o01_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + 64 ^ ((warp % 4 * 32 + lane) * 128 + 64 >> 7 & 7) << 4))), "r"(_tmem_load_5_bf16[0]), "r"(_tmem_load_5_bf16[1]), "r"(_tmem_load_5_bf16[2]), "r"(_tmem_load_5_bf16[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o01_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + 80 ^ ((warp % 4 * 32 + lane) * 128 + 80 >> 7 & 7) << 4))), "r"(_tmem_load_5_bf16[4]), "r"(_tmem_load_5_bf16[5]), "r"(_tmem_load_5_bf16[6]), "r"(_tmem_load_5_bf16[7]) : "memory");
                        int load_addr_2 = taddr + (unsigned int)128 + (unsigned int)(warp % 4 * 32 << 16) + 48;
                        float _tmem_load_6[16];
                        tmem_ld_x16(&_tmem_load_6[0], load_addr_2);
                        float _tmem_load_7[16];
                        tmem_ld_x16(&_tmem_load_7[0], load_addr_2 + 64);
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        const float2 _scale2_44 = {final_scale, final_scale};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_6)[_ls], _scale2_44);
                        unsigned int _tmem_load_6_bf16[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 8; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_6[_lp*2 + 0], _tmem_load_6[_lp*2+1 + 0]));
                            _tmem_load_6_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o00_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + 96 ^ ((warp % 4 * 32 + lane) * 128 + 96 >> 7 & 7) << 4))), "r"(_tmem_load_6_bf16[0]), "r"(_tmem_load_6_bf16[1]), "r"(_tmem_load_6_bf16[2]), "r"(_tmem_load_6_bf16[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o00_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + 112 ^ ((warp % 4 * 32 + lane) * 128 + 112 >> 7 & 7) << 4))), "r"(_tmem_load_6_bf16[4]), "r"(_tmem_load_6_bf16[5]), "r"(_tmem_load_6_bf16[6]), "r"(_tmem_load_6_bf16[7]) : "memory");
                        const float2 _scale2_45 = {final_scale, final_scale};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_7)[_ls], _scale2_45);
                        unsigned int _tmem_load_7_bf16[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 8; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_7[_lp*2 + 0], _tmem_load_7[_lp*2+1 + 0]));
                            _tmem_load_7_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o01_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + 96 ^ ((warp % 4 * 32 + lane) * 128 + 96 >> 7 & 7) << 4))), "r"(_tmem_load_7_bf16[0]), "r"(_tmem_load_7_bf16[1]), "r"(_tmem_load_7_bf16[2]), "r"(_tmem_load_7_bf16[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o01_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + 112 ^ ((warp % 4 * 32 + lane) * 128 + 112 >> 7 & 7) << 4))), "r"(_tmem_load_7_bf16[4]), "r"(_tmem_load_7_bf16[5]), "r"(_tmem_load_7_bf16[6]), "r"(_tmem_load_7_bf16[7]) : "memory");
                        asm volatile("fence.proxy.async;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile("barrier.sync 1, 128;" ::: "memory");
                        if (warp == 0) {
                            if (elect_sync()) {
                                tma_store_4d((&O), 0, out_row, out_head, out_batch, smem_o00_addr);
                                tma_store_4d((&O), 64, out_row, out_head, out_batch, smem_o01_addr);
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    } else {
                        if (warp == 4) {
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                        }
                        asm volatile("barrier.sync 4, 128;" ::: "memory");
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int load_addr_3 = taddr + (unsigned int)256 + (unsigned int)(warp % 4 * 32 << 16);
                        float _tmem_load_8[16];
                        tmem_ld_x16(&_tmem_load_8[0], load_addr_3);
                        float _tmem_load_9[16];
                        tmem_ld_x16(&_tmem_load_9[0], load_addr_3 + 64);
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        const float2 _scale2_46 = {final_scale, final_scale};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_8)[_ls], _scale2_46);
                        unsigned int _tmem_load_8_bf16[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 8; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_8[_lp*2 + 0], _tmem_load_8[_lp*2+1 + 0]));
                            _tmem_load_8_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o10_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 ^ ((warp % 4 * 32 + lane) * 128 >> 7 & 7) << 4))), "r"(_tmem_load_8_bf16[0]), "r"(_tmem_load_8_bf16[1]), "r"(_tmem_load_8_bf16[2]), "r"(_tmem_load_8_bf16[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o10_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + 16 ^ ((warp % 4 * 32 + lane) * 128 + 16 >> 7 & 7) << 4))), "r"(_tmem_load_8_bf16[4]), "r"(_tmem_load_8_bf16[5]), "r"(_tmem_load_8_bf16[6]), "r"(_tmem_load_8_bf16[7]) : "memory");
                        const float2 _scale2_47 = {final_scale, final_scale};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_9)[_ls], _scale2_47);
                        unsigned int _tmem_load_9_bf16[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 8; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_9[_lp*2 + 0], _tmem_load_9[_lp*2+1 + 0]));
                            _tmem_load_9_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o11_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 ^ ((warp % 4 * 32 + lane) * 128 >> 7 & 7) << 4))), "r"(_tmem_load_9_bf16[0]), "r"(_tmem_load_9_bf16[1]), "r"(_tmem_load_9_bf16[2]), "r"(_tmem_load_9_bf16[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o11_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + 16 ^ ((warp % 4 * 32 + lane) * 128 + 16 >> 7 & 7) << 4))), "r"(_tmem_load_9_bf16[4]), "r"(_tmem_load_9_bf16[5]), "r"(_tmem_load_9_bf16[6]), "r"(_tmem_load_9_bf16[7]) : "memory");
                        int load_addr_0_1 = taddr + (unsigned int)256 + (unsigned int)(warp % 4 * 32 << 16) + 16;
                        float _tmem_load_10[16];
                        tmem_ld_x16(&_tmem_load_10[0], load_addr_0_1);
                        float _tmem_load_11[16];
                        tmem_ld_x16(&_tmem_load_11[0], load_addr_0_1 + 64);
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        const float2 _scale2_48 = {final_scale, final_scale};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_10)[_ls], _scale2_48);
                        unsigned int _tmem_load_10_bf16[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 8; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_10[_lp*2 + 0], _tmem_load_10[_lp*2+1 + 0]));
                            _tmem_load_10_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o10_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + 32 ^ ((warp % 4 * 32 + lane) * 128 + 32 >> 7 & 7) << 4))), "r"(_tmem_load_10_bf16[0]), "r"(_tmem_load_10_bf16[1]), "r"(_tmem_load_10_bf16[2]), "r"(_tmem_load_10_bf16[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o10_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + 48 ^ ((warp % 4 * 32 + lane) * 128 + 48 >> 7 & 7) << 4))), "r"(_tmem_load_10_bf16[4]), "r"(_tmem_load_10_bf16[5]), "r"(_tmem_load_10_bf16[6]), "r"(_tmem_load_10_bf16[7]) : "memory");
                        const float2 _scale2_49 = {final_scale, final_scale};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_11)[_ls], _scale2_49);
                        unsigned int _tmem_load_11_bf16[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 8; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_11[_lp*2 + 0], _tmem_load_11[_lp*2+1 + 0]));
                            _tmem_load_11_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o11_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + 32 ^ ((warp % 4 * 32 + lane) * 128 + 32 >> 7 & 7) << 4))), "r"(_tmem_load_11_bf16[0]), "r"(_tmem_load_11_bf16[1]), "r"(_tmem_load_11_bf16[2]), "r"(_tmem_load_11_bf16[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o11_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + 48 ^ ((warp % 4 * 32 + lane) * 128 + 48 >> 7 & 7) << 4))), "r"(_tmem_load_11_bf16[4]), "r"(_tmem_load_11_bf16[5]), "r"(_tmem_load_11_bf16[6]), "r"(_tmem_load_11_bf16[7]) : "memory");
                        int load_addr_1_1 = taddr + (unsigned int)256 + (unsigned int)(warp % 4 * 32 << 16) + 32;
                        float _tmem_load_12[16];
                        tmem_ld_x16(&_tmem_load_12[0], load_addr_1_1);
                        float _tmem_load_13[16];
                        tmem_ld_x16(&_tmem_load_13[0], load_addr_1_1 + 64);
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        const float2 _scale2_50 = {final_scale, final_scale};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_12)[_ls], _scale2_50);
                        unsigned int _tmem_load_12_bf16[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 8; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_12[_lp*2 + 0], _tmem_load_12[_lp*2+1 + 0]));
                            _tmem_load_12_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o10_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + 64 ^ ((warp % 4 * 32 + lane) * 128 + 64 >> 7 & 7) << 4))), "r"(_tmem_load_12_bf16[0]), "r"(_tmem_load_12_bf16[1]), "r"(_tmem_load_12_bf16[2]), "r"(_tmem_load_12_bf16[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o10_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + 80 ^ ((warp % 4 * 32 + lane) * 128 + 80 >> 7 & 7) << 4))), "r"(_tmem_load_12_bf16[4]), "r"(_tmem_load_12_bf16[5]), "r"(_tmem_load_12_bf16[6]), "r"(_tmem_load_12_bf16[7]) : "memory");
                        const float2 _scale2_51 = {final_scale, final_scale};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_13)[_ls], _scale2_51);
                        unsigned int _tmem_load_13_bf16[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 8; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_13[_lp*2 + 0], _tmem_load_13[_lp*2+1 + 0]));
                            _tmem_load_13_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o11_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + 64 ^ ((warp % 4 * 32 + lane) * 128 + 64 >> 7 & 7) << 4))), "r"(_tmem_load_13_bf16[0]), "r"(_tmem_load_13_bf16[1]), "r"(_tmem_load_13_bf16[2]), "r"(_tmem_load_13_bf16[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o11_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + 80 ^ ((warp % 4 * 32 + lane) * 128 + 80 >> 7 & 7) << 4))), "r"(_tmem_load_13_bf16[4]), "r"(_tmem_load_13_bf16[5]), "r"(_tmem_load_13_bf16[6]), "r"(_tmem_load_13_bf16[7]) : "memory");
                        int load_addr_2_1 = taddr + (unsigned int)256 + (unsigned int)(warp % 4 * 32 << 16) + 48;
                        float _tmem_load_14[16];
                        tmem_ld_x16(&_tmem_load_14[0], load_addr_2_1);
                        float _tmem_load_15[16];
                        tmem_ld_x16(&_tmem_load_15[0], load_addr_2_1 + 64);
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        const float2 _scale2_52 = {final_scale, final_scale};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_14)[_ls], _scale2_52);
                        unsigned int _tmem_load_14_bf16[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 8; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_14[_lp*2 + 0], _tmem_load_14[_lp*2+1 + 0]));
                            _tmem_load_14_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o10_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + 96 ^ ((warp % 4 * 32 + lane) * 128 + 96 >> 7 & 7) << 4))), "r"(_tmem_load_14_bf16[0]), "r"(_tmem_load_14_bf16[1]), "r"(_tmem_load_14_bf16[2]), "r"(_tmem_load_14_bf16[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o10_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + 112 ^ ((warp % 4 * 32 + lane) * 128 + 112 >> 7 & 7) << 4))), "r"(_tmem_load_14_bf16[4]), "r"(_tmem_load_14_bf16[5]), "r"(_tmem_load_14_bf16[6]), "r"(_tmem_load_14_bf16[7]) : "memory");
                        const float2 _scale2_53 = {final_scale, final_scale};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_15)[_ls], _scale2_53);
                        unsigned int _tmem_load_15_bf16[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 8; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_15[_lp*2 + 0], _tmem_load_15[_lp*2+1 + 0]));
                            _tmem_load_15_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o11_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + 96 ^ ((warp % 4 * 32 + lane) * 128 + 96 >> 7 & 7) << 4))), "r"(_tmem_load_15_bf16[0]), "r"(_tmem_load_15_bf16[1]), "r"(_tmem_load_15_bf16[2]), "r"(_tmem_load_15_bf16[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o11_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + 112 ^ ((warp % 4 * 32 + lane) * 128 + 112 >> 7 & 7) << 4))), "r"(_tmem_load_15_bf16[4]), "r"(_tmem_load_15_bf16[5]), "r"(_tmem_load_15_bf16[6]), "r"(_tmem_load_15_bf16[7]) : "memory");
                        asm volatile("fence.proxy.async;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile("barrier.sync 2, 128;" ::: "memory");
                        if (warp == 4) {
                            if (elect_sync()) {
                                tma_store_4d((&O), 0, out_row, out_head, out_batch, smem_o10_addr);
                                tma_store_4d((&O), 64, out_row, out_head, out_batch, smem_o11_addr);
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    }
                } else {
                    unsigned int n_chunks_u_0 = (unsigned int)n_chunks;
                    unsigned int slot_u = (unsigned int)split_slot;
                    unsigned int split_tile = slot_u / n_chunks_u_0;
                    unsigned int tile_slot0 = split_tile * n_chunks_u_0;
                    unsigned int part_row = ((unsigned int)(cta_rank * 2) + stage) * 128 + (unsigned int)(warp % 4 * 32 + lane);
                    unsigned int own_row = slot_u * 512 + part_row;
                    float ml[2];
                    ml[0] = row_max_scaled;
                    ml[1] = row_sum;
                    {
                        float2 _v2 = make_float2(ml[0 + 0], ml[0 + 1]);
                        *reinterpret_cast<float2*>((part_ml + (own_row * 2)) + 0) = _v2;
                    }
                    unsigned int o_tmem_off = ((stage == 0) ? 128 : 256);
                    int load_addr_4 = taddr + (unsigned int)o_tmem_off + (unsigned int)(warp % 4 * 32 << 16);
                    float _tmem_load_16[16];
                    tmem_ld_x16(&_tmem_load_16[0], load_addr_4);
                    float _tmem_load_17[16];
                    tmem_ld_x16(&_tmem_load_17[0], load_addr_4 + 64);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    unsigned int off = own_row * 128;
                    {
                        float4 _v4 = make_float4(_tmem_load_16[0 + 0], _tmem_load_16[0 + 1], _tmem_load_16[0 + 2], _tmem_load_16[0 + 3]);
                        *reinterpret_cast<float4*>((part_o + off) + 0) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_17[0 + 0], _tmem_load_17[0 + 1], _tmem_load_17[0 + 2], _tmem_load_17[0 + 3]);
                        *reinterpret_cast<float4*>((part_o + (off + 64)) + 0) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_16[4 + 0], _tmem_load_16[4 + 1], _tmem_load_16[4 + 2], _tmem_load_16[4 + 3]);
                        *reinterpret_cast<float4*>((part_o + (off + 4)) + 0) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_17[4 + 0], _tmem_load_17[4 + 1], _tmem_load_17[4 + 2], _tmem_load_17[4 + 3]);
                        *reinterpret_cast<float4*>((part_o + (off + 64 + 4)) + 0) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_16[8 + 0], _tmem_load_16[8 + 1], _tmem_load_16[8 + 2], _tmem_load_16[8 + 3]);
                        *reinterpret_cast<float4*>((part_o + (off + 8)) + 0) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_17[8 + 0], _tmem_load_17[8 + 1], _tmem_load_17[8 + 2], _tmem_load_17[8 + 3]);
                        *reinterpret_cast<float4*>((part_o + (off + 64 + 8)) + 0) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_16[12 + 0], _tmem_load_16[12 + 1], _tmem_load_16[12 + 2], _tmem_load_16[12 + 3]);
                        *reinterpret_cast<float4*>((part_o + (off + 12)) + 0) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_17[12 + 0], _tmem_load_17[12 + 1], _tmem_load_17[12 + 2], _tmem_load_17[12 + 3]);
                        *reinterpret_cast<float4*>((part_o + (off + 64 + 12)) + 0) = _v4;
                    }
                    int load_addr_1_2 = taddr + (unsigned int)o_tmem_off + (unsigned int)(warp % 4 * 32 << 16) + 16;
                    float _tmem_load_18[16];
                    tmem_ld_x16(&_tmem_load_18[0], load_addr_1_2);
                    float _tmem_load_19[16];
                    tmem_ld_x16(&_tmem_load_19[0], load_addr_1_2 + 64);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    unsigned int off_2 = own_row * 128 + 16;
                    {
                        float4 _v4 = make_float4(_tmem_load_18[0 + 0], _tmem_load_18[0 + 1], _tmem_load_18[0 + 2], _tmem_load_18[0 + 3]);
                        *reinterpret_cast<float4*>((part_o + off_2) + 0) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_19[0 + 0], _tmem_load_19[0 + 1], _tmem_load_19[0 + 2], _tmem_load_19[0 + 3]);
                        *reinterpret_cast<float4*>((part_o + (off_2 + 64)) + 0) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_18[4 + 0], _tmem_load_18[4 + 1], _tmem_load_18[4 + 2], _tmem_load_18[4 + 3]);
                        *reinterpret_cast<float4*>((part_o + (off_2 + 4)) + 0) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_19[4 + 0], _tmem_load_19[4 + 1], _tmem_load_19[4 + 2], _tmem_load_19[4 + 3]);
                        *reinterpret_cast<float4*>((part_o + (off_2 + 64 + 4)) + 0) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_18[8 + 0], _tmem_load_18[8 + 1], _tmem_load_18[8 + 2], _tmem_load_18[8 + 3]);
                        *reinterpret_cast<float4*>((part_o + (off_2 + 8)) + 0) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_19[8 + 0], _tmem_load_19[8 + 1], _tmem_load_19[8 + 2], _tmem_load_19[8 + 3]);
                        *reinterpret_cast<float4*>((part_o + (off_2 + 64 + 8)) + 0) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_18[12 + 0], _tmem_load_18[12 + 1], _tmem_load_18[12 + 2], _tmem_load_18[12 + 3]);
                        *reinterpret_cast<float4*>((part_o + (off_2 + 12)) + 0) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_19[12 + 0], _tmem_load_19[12 + 1], _tmem_load_19[12 + 2], _tmem_load_19[12 + 3]);
                        *reinterpret_cast<float4*>((part_o + (off_2 + 64 + 12)) + 0) = _v4;
                    }
                    int load_addr_3_1 = taddr + (unsigned int)o_tmem_off + (unsigned int)(warp % 4 * 32 << 16) + 32;
                    float _tmem_load_20[16];
                    tmem_ld_x16(&_tmem_load_20[0], load_addr_3_1);
                    float _tmem_load_21[16];
                    tmem_ld_x16(&_tmem_load_21[0], load_addr_3_1 + 64);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    unsigned int off_4 = own_row * 128 + 32;
                    {
                        float4 _v4 = make_float4(_tmem_load_20[0 + 0], _tmem_load_20[0 + 1], _tmem_load_20[0 + 2], _tmem_load_20[0 + 3]);
                        *reinterpret_cast<float4*>((part_o + off_4) + 0) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_21[0 + 0], _tmem_load_21[0 + 1], _tmem_load_21[0 + 2], _tmem_load_21[0 + 3]);
                        *reinterpret_cast<float4*>((part_o + (off_4 + 64)) + 0) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_20[4 + 0], _tmem_load_20[4 + 1], _tmem_load_20[4 + 2], _tmem_load_20[4 + 3]);
                        *reinterpret_cast<float4*>((part_o + (off_4 + 4)) + 0) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_21[4 + 0], _tmem_load_21[4 + 1], _tmem_load_21[4 + 2], _tmem_load_21[4 + 3]);
                        *reinterpret_cast<float4*>((part_o + (off_4 + 64 + 4)) + 0) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_20[8 + 0], _tmem_load_20[8 + 1], _tmem_load_20[8 + 2], _tmem_load_20[8 + 3]);
                        *reinterpret_cast<float4*>((part_o + (off_4 + 8)) + 0) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_21[8 + 0], _tmem_load_21[8 + 1], _tmem_load_21[8 + 2], _tmem_load_21[8 + 3]);
                        *reinterpret_cast<float4*>((part_o + (off_4 + 64 + 8)) + 0) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_20[12 + 0], _tmem_load_20[12 + 1], _tmem_load_20[12 + 2], _tmem_load_20[12 + 3]);
                        *reinterpret_cast<float4*>((part_o + (off_4 + 12)) + 0) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_21[12 + 0], _tmem_load_21[12 + 1], _tmem_load_21[12 + 2], _tmem_load_21[12 + 3]);
                        *reinterpret_cast<float4*>((part_o + (off_4 + 64 + 12)) + 0) = _v4;
                    }
                    int load_addr_5 = taddr + (unsigned int)o_tmem_off + (unsigned int)(warp % 4 * 32 << 16) + 48;
                    float _tmem_load_22[16];
                    tmem_ld_x16(&_tmem_load_22[0], load_addr_5);
                    float _tmem_load_23[16];
                    tmem_ld_x16(&_tmem_load_23[0], load_addr_5 + 64);
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    unsigned int off_6 = own_row * 128 + 48;
                    {
                        float4 _v4 = make_float4(_tmem_load_22[0 + 0], _tmem_load_22[0 + 1], _tmem_load_22[0 + 2], _tmem_load_22[0 + 3]);
                        *reinterpret_cast<float4*>((part_o + off_6) + 0) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_23[0 + 0], _tmem_load_23[0 + 1], _tmem_load_23[0 + 2], _tmem_load_23[0 + 3]);
                        *reinterpret_cast<float4*>((part_o + (off_6 + 64)) + 0) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_22[4 + 0], _tmem_load_22[4 + 1], _tmem_load_22[4 + 2], _tmem_load_22[4 + 3]);
                        *reinterpret_cast<float4*>((part_o + (off_6 + 4)) + 0) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_23[4 + 0], _tmem_load_23[4 + 1], _tmem_load_23[4 + 2], _tmem_load_23[4 + 3]);
                        *reinterpret_cast<float4*>((part_o + (off_6 + 64 + 4)) + 0) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_22[8 + 0], _tmem_load_22[8 + 1], _tmem_load_22[8 + 2], _tmem_load_22[8 + 3]);
                        *reinterpret_cast<float4*>((part_o + (off_6 + 8)) + 0) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_23[8 + 0], _tmem_load_23[8 + 1], _tmem_load_23[8 + 2], _tmem_load_23[8 + 3]);
                        *reinterpret_cast<float4*>((part_o + (off_6 + 64 + 8)) + 0) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_22[12 + 0], _tmem_load_22[12 + 1], _tmem_load_22[12 + 2], _tmem_load_22[12 + 3]);
                        *reinterpret_cast<float4*>((part_o + (off_6 + 12)) + 0) = _v4;
                    }
                    {
                        float4 _v4 = make_float4(_tmem_load_23[12 + 0], _tmem_load_23[12 + 1], _tmem_load_23[12 + 2], _tmem_load_23[12 + 3]);
                        *reinterpret_cast<float4*>((part_o + (off_6 + 64 + 12)) + 0) = _v4;
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("barrier.sync 5, 256;" ::: "memory");
                    if (tid == 0) {
                        unsigned int counter_idx = split_tile * 2 + (unsigned int)cta_rank;
                        unsigned int old;
                        asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                            : "=r"(old) : "l"(&split_counters[counter_idx]), "r"(static_cast<uint32_t>(1)) : "memory");
                        int last = ((old == n_chunks_u_0 - 1) ? 1 : 0);
                        split_flag[0] = last;
                        if (last == 1) {
                            *(reinterpret_cast<unsigned int*>(split_counters + counter_idx) + (0)) = 0;
                        }
                    }
                    asm volatile("barrier.sync 5, 256;" ::: "memory");
                    if (split_flag[0] == 1) {
                        float row_max_all = -CUDART_INF_F;
                        if (n_chunks_u_0 > 0) {
                            row_max_all = max_noftz(row_max_all, part_ml[(tile_slot0 * 512 + part_row) * 2]);
                        }
                        if (n_chunks_u_0 > 1) {
                            row_max_all = max_noftz(row_max_all, part_ml[((tile_slot0 + 1) * 512 + part_row) * 2]);
                        }
                        if (n_chunks_u_0 > 2) {
                            row_max_all = max_noftz(row_max_all, part_ml[((tile_slot0 + 2) * 512 + part_row) * 2]);
                        }
                        if (n_chunks_u_0 > 3) {
                            row_max_all = max_noftz(row_max_all, part_ml[((tile_slot0 + 3) * 512 + part_row) * 2]);
                        }
                        float l_all = 0.0f;
                        if (n_chunks_u_0 > 0) {
                            unsigned int ml_idx = (tile_slot0 * 512 + part_row) * 2;
                            float w_c = approx_exp2(part_ml[ml_idx] - row_max_all);
                            l_all = __fmaf_rn(part_ml[ml_idx + 1], w_c, l_all);
                        }
                        if (n_chunks_u_0 > 1) {
                            unsigned int ml_idx_1 = ((tile_slot0 + 1) * 512 + part_row) * 2;
                            float w_c_1 = approx_exp2(part_ml[ml_idx_1] - row_max_all);
                            l_all = __fmaf_rn(part_ml[ml_idx_1 + 1], w_c_1, l_all);
                        }
                        if (n_chunks_u_0 > 2) {
                            unsigned int ml_idx_2 = ((tile_slot0 + 2) * 512 + part_row) * 2;
                            float w_c_2 = approx_exp2(part_ml[ml_idx_2] - row_max_all);
                            l_all = __fmaf_rn(part_ml[ml_idx_2 + 1], w_c_2, l_all);
                        }
                        if (n_chunks_u_0 > 3) {
                            unsigned int ml_idx_3 = ((tile_slot0 + 3) * 512 + part_row) * 2;
                            float w_c_3 = approx_exp2(part_ml[ml_idx_3] - row_max_all);
                            l_all = __fmaf_rn(part_ml[ml_idx_3 + 1], w_c_3, l_all);
                        }
                        float _rcp_54 = approx_rcp(l_all);
                        final_scale = ((l_all != 0.0f && l_all == l_all) ? _rcp_54 : 0.0f);
                        int out_row_1 = (int)((m_block + stage) * 128);
                        int out_head_1 = (int)(bh % (unsigned int)heads);
                        int out_batch_1 = (int)(bh / (unsigned int)heads);
                        if (stage == 0) {
                            if (warp == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 0;");
                            }
                            asm volatile("barrier.sync 3, 128;" ::: "memory");
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            float acc[16];
                            float acc_hi[16];
                            acc[0] = 0.0f;
                            acc_hi[0] = 0.0f;
                            acc[1] = 0.0f;
                            acc_hi[1] = 0.0f;
                            acc[2] = 0.0f;
                            acc_hi[2] = 0.0f;
                            acc[3] = 0.0f;
                            acc_hi[3] = 0.0f;
                            acc[4] = 0.0f;
                            acc_hi[4] = 0.0f;
                            acc[5] = 0.0f;
                            acc_hi[5] = 0.0f;
                            acc[6] = 0.0f;
                            acc_hi[6] = 0.0f;
                            acc[7] = 0.0f;
                            acc_hi[7] = 0.0f;
                            acc[8] = 0.0f;
                            acc_hi[8] = 0.0f;
                            acc[9] = 0.0f;
                            acc_hi[9] = 0.0f;
                            acc[10] = 0.0f;
                            acc_hi[10] = 0.0f;
                            acc[11] = 0.0f;
                            acc_hi[11] = 0.0f;
                            acc[12] = 0.0f;
                            acc_hi[12] = 0.0f;
                            acc[13] = 0.0f;
                            acc_hi[13] = 0.0f;
                            acc[14] = 0.0f;
                            acc_hi[14] = 0.0f;
                            acc[15] = 0.0f;
                            acc_hi[15] = 0.0f;
                            float part[16];
                            float part_hi[16];
                            if (n_chunks_u_0 > 0) {
                                unsigned int crow = tile_slot0 * 512 + part_row;
                                float w_c_4 = approx_exp2(part_ml[crow * 2] - row_max_all);
                                unsigned int off_0 = crow * 128;
                                {
                                    unsigned _ldv8_55_0;
                                    unsigned _ldv8_55_1;
                                    unsigned _ldv8_55_2;
                                    unsigned _ldv8_55_3;
                                    unsigned _ldv8_55_4;
                                    unsigned _ldv8_55_5;
                                    unsigned _ldv8_55_6;
                                    unsigned _ldv8_55_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_55_0), "=r"(_ldv8_55_1), "=r"(_ldv8_55_2), "=r"(_ldv8_55_3), "=r"(_ldv8_55_4), "=r"(_ldv8_55_5), "=r"(_ldv8_55_6), "=r"(_ldv8_55_7) : "l"((const void*)(part_o + (off_0))) : "memory");
                                    part[0 + 0] = __uint_as_float(_ldv8_55_0);
                                    part[0 + 1] = __uint_as_float(_ldv8_55_1);
                                    part[0 + 2] = __uint_as_float(_ldv8_55_2);
                                    part[0 + 3] = __uint_as_float(_ldv8_55_3);
                                    part[0 + 4] = __uint_as_float(_ldv8_55_4);
                                    part[0 + 5] = __uint_as_float(_ldv8_55_5);
                                    part[0 + 6] = __uint_as_float(_ldv8_55_6);
                                    part[0 + 7] = __uint_as_float(_ldv8_55_7);
                                }
                                {
                                    unsigned _ldv8_56_0;
                                    unsigned _ldv8_56_1;
                                    unsigned _ldv8_56_2;
                                    unsigned _ldv8_56_3;
                                    unsigned _ldv8_56_4;
                                    unsigned _ldv8_56_5;
                                    unsigned _ldv8_56_6;
                                    unsigned _ldv8_56_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_56_0), "=r"(_ldv8_56_1), "=r"(_ldv8_56_2), "=r"(_ldv8_56_3), "=r"(_ldv8_56_4), "=r"(_ldv8_56_5), "=r"(_ldv8_56_6), "=r"(_ldv8_56_7) : "l"((const void*)(part_o + (off_0 + 8))) : "memory");
                                    part[8 + 0] = __uint_as_float(_ldv8_56_0);
                                    part[8 + 1] = __uint_as_float(_ldv8_56_1);
                                    part[8 + 2] = __uint_as_float(_ldv8_56_2);
                                    part[8 + 3] = __uint_as_float(_ldv8_56_3);
                                    part[8 + 4] = __uint_as_float(_ldv8_56_4);
                                    part[8 + 5] = __uint_as_float(_ldv8_56_5);
                                    part[8 + 6] = __uint_as_float(_ldv8_56_6);
                                    part[8 + 7] = __uint_as_float(_ldv8_56_7);
                                }
                                {
                                    unsigned _ldv8_57_0;
                                    unsigned _ldv8_57_1;
                                    unsigned _ldv8_57_2;
                                    unsigned _ldv8_57_3;
                                    unsigned _ldv8_57_4;
                                    unsigned _ldv8_57_5;
                                    unsigned _ldv8_57_6;
                                    unsigned _ldv8_57_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_57_0), "=r"(_ldv8_57_1), "=r"(_ldv8_57_2), "=r"(_ldv8_57_3), "=r"(_ldv8_57_4), "=r"(_ldv8_57_5), "=r"(_ldv8_57_6), "=r"(_ldv8_57_7) : "l"((const void*)(part_o + (off_0 + 64))) : "memory");
                                    part_hi[0 + 0] = __uint_as_float(_ldv8_57_0);
                                    part_hi[0 + 1] = __uint_as_float(_ldv8_57_1);
                                    part_hi[0 + 2] = __uint_as_float(_ldv8_57_2);
                                    part_hi[0 + 3] = __uint_as_float(_ldv8_57_3);
                                    part_hi[0 + 4] = __uint_as_float(_ldv8_57_4);
                                    part_hi[0 + 5] = __uint_as_float(_ldv8_57_5);
                                    part_hi[0 + 6] = __uint_as_float(_ldv8_57_6);
                                    part_hi[0 + 7] = __uint_as_float(_ldv8_57_7);
                                }
                                {
                                    unsigned _ldv8_58_0;
                                    unsigned _ldv8_58_1;
                                    unsigned _ldv8_58_2;
                                    unsigned _ldv8_58_3;
                                    unsigned _ldv8_58_4;
                                    unsigned _ldv8_58_5;
                                    unsigned _ldv8_58_6;
                                    unsigned _ldv8_58_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_58_0), "=r"(_ldv8_58_1), "=r"(_ldv8_58_2), "=r"(_ldv8_58_3), "=r"(_ldv8_58_4), "=r"(_ldv8_58_5), "=r"(_ldv8_58_6), "=r"(_ldv8_58_7) : "l"((const void*)(part_o + (off_0 + 64 + 8))) : "memory");
                                    part_hi[8 + 0] = __uint_as_float(_ldv8_58_0);
                                    part_hi[8 + 1] = __uint_as_float(_ldv8_58_1);
                                    part_hi[8 + 2] = __uint_as_float(_ldv8_58_2);
                                    part_hi[8 + 3] = __uint_as_float(_ldv8_58_3);
                                    part_hi[8 + 4] = __uint_as_float(_ldv8_58_4);
                                    part_hi[8 + 5] = __uint_as_float(_ldv8_58_5);
                                    part_hi[8 + 6] = __uint_as_float(_ldv8_58_6);
                                    part_hi[8 + 7] = __uint_as_float(_ldv8_58_7);
                                }
                                acc[0] = __fmaf_rn(part[0], w_c_4, acc[0]);
                                acc_hi[0] = __fmaf_rn(part_hi[0], w_c_4, acc_hi[0]);
                                acc[1] = __fmaf_rn(part[1], w_c_4, acc[1]);
                                acc_hi[1] = __fmaf_rn(part_hi[1], w_c_4, acc_hi[1]);
                                acc[2] = __fmaf_rn(part[2], w_c_4, acc[2]);
                                acc_hi[2] = __fmaf_rn(part_hi[2], w_c_4, acc_hi[2]);
                                acc[3] = __fmaf_rn(part[3], w_c_4, acc[3]);
                                acc_hi[3] = __fmaf_rn(part_hi[3], w_c_4, acc_hi[3]);
                                acc[4] = __fmaf_rn(part[4], w_c_4, acc[4]);
                                acc_hi[4] = __fmaf_rn(part_hi[4], w_c_4, acc_hi[4]);
                                acc[5] = __fmaf_rn(part[5], w_c_4, acc[5]);
                                acc_hi[5] = __fmaf_rn(part_hi[5], w_c_4, acc_hi[5]);
                                acc[6] = __fmaf_rn(part[6], w_c_4, acc[6]);
                                acc_hi[6] = __fmaf_rn(part_hi[6], w_c_4, acc_hi[6]);
                                acc[7] = __fmaf_rn(part[7], w_c_4, acc[7]);
                                acc_hi[7] = __fmaf_rn(part_hi[7], w_c_4, acc_hi[7]);
                                acc[8] = __fmaf_rn(part[8], w_c_4, acc[8]);
                                acc_hi[8] = __fmaf_rn(part_hi[8], w_c_4, acc_hi[8]);
                                acc[9] = __fmaf_rn(part[9], w_c_4, acc[9]);
                                acc_hi[9] = __fmaf_rn(part_hi[9], w_c_4, acc_hi[9]);
                                acc[10] = __fmaf_rn(part[10], w_c_4, acc[10]);
                                acc_hi[10] = __fmaf_rn(part_hi[10], w_c_4, acc_hi[10]);
                                acc[11] = __fmaf_rn(part[11], w_c_4, acc[11]);
                                acc_hi[11] = __fmaf_rn(part_hi[11], w_c_4, acc_hi[11]);
                                acc[12] = __fmaf_rn(part[12], w_c_4, acc[12]);
                                acc_hi[12] = __fmaf_rn(part_hi[12], w_c_4, acc_hi[12]);
                                acc[13] = __fmaf_rn(part[13], w_c_4, acc[13]);
                                acc_hi[13] = __fmaf_rn(part_hi[13], w_c_4, acc_hi[13]);
                                acc[14] = __fmaf_rn(part[14], w_c_4, acc[14]);
                                acc_hi[14] = __fmaf_rn(part_hi[14], w_c_4, acc_hi[14]);
                                acc[15] = __fmaf_rn(part[15], w_c_4, acc[15]);
                                acc_hi[15] = __fmaf_rn(part_hi[15], w_c_4, acc_hi[15]);
                            }
                            if (n_chunks_u_0 > 1) {
                                unsigned int crow_1 = (tile_slot0 + 1) * 512 + part_row;
                                float w_c_5 = approx_exp2(part_ml[crow_1 * 2] - row_max_all);
                                unsigned int off_0_1 = crow_1 * 128;
                                {
                                    unsigned _ldv8_59_0;
                                    unsigned _ldv8_59_1;
                                    unsigned _ldv8_59_2;
                                    unsigned _ldv8_59_3;
                                    unsigned _ldv8_59_4;
                                    unsigned _ldv8_59_5;
                                    unsigned _ldv8_59_6;
                                    unsigned _ldv8_59_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_59_0), "=r"(_ldv8_59_1), "=r"(_ldv8_59_2), "=r"(_ldv8_59_3), "=r"(_ldv8_59_4), "=r"(_ldv8_59_5), "=r"(_ldv8_59_6), "=r"(_ldv8_59_7) : "l"((const void*)(part_o + (off_0_1))) : "memory");
                                    part[0 + 0] = __uint_as_float(_ldv8_59_0);
                                    part[0 + 1] = __uint_as_float(_ldv8_59_1);
                                    part[0 + 2] = __uint_as_float(_ldv8_59_2);
                                    part[0 + 3] = __uint_as_float(_ldv8_59_3);
                                    part[0 + 4] = __uint_as_float(_ldv8_59_4);
                                    part[0 + 5] = __uint_as_float(_ldv8_59_5);
                                    part[0 + 6] = __uint_as_float(_ldv8_59_6);
                                    part[0 + 7] = __uint_as_float(_ldv8_59_7);
                                }
                                {
                                    unsigned _ldv8_60_0;
                                    unsigned _ldv8_60_1;
                                    unsigned _ldv8_60_2;
                                    unsigned _ldv8_60_3;
                                    unsigned _ldv8_60_4;
                                    unsigned _ldv8_60_5;
                                    unsigned _ldv8_60_6;
                                    unsigned _ldv8_60_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_60_0), "=r"(_ldv8_60_1), "=r"(_ldv8_60_2), "=r"(_ldv8_60_3), "=r"(_ldv8_60_4), "=r"(_ldv8_60_5), "=r"(_ldv8_60_6), "=r"(_ldv8_60_7) : "l"((const void*)(part_o + (off_0_1 + 8))) : "memory");
                                    part[8 + 0] = __uint_as_float(_ldv8_60_0);
                                    part[8 + 1] = __uint_as_float(_ldv8_60_1);
                                    part[8 + 2] = __uint_as_float(_ldv8_60_2);
                                    part[8 + 3] = __uint_as_float(_ldv8_60_3);
                                    part[8 + 4] = __uint_as_float(_ldv8_60_4);
                                    part[8 + 5] = __uint_as_float(_ldv8_60_5);
                                    part[8 + 6] = __uint_as_float(_ldv8_60_6);
                                    part[8 + 7] = __uint_as_float(_ldv8_60_7);
                                }
                                {
                                    unsigned _ldv8_61_0;
                                    unsigned _ldv8_61_1;
                                    unsigned _ldv8_61_2;
                                    unsigned _ldv8_61_3;
                                    unsigned _ldv8_61_4;
                                    unsigned _ldv8_61_5;
                                    unsigned _ldv8_61_6;
                                    unsigned _ldv8_61_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_61_0), "=r"(_ldv8_61_1), "=r"(_ldv8_61_2), "=r"(_ldv8_61_3), "=r"(_ldv8_61_4), "=r"(_ldv8_61_5), "=r"(_ldv8_61_6), "=r"(_ldv8_61_7) : "l"((const void*)(part_o + (off_0_1 + 64))) : "memory");
                                    part_hi[0 + 0] = __uint_as_float(_ldv8_61_0);
                                    part_hi[0 + 1] = __uint_as_float(_ldv8_61_1);
                                    part_hi[0 + 2] = __uint_as_float(_ldv8_61_2);
                                    part_hi[0 + 3] = __uint_as_float(_ldv8_61_3);
                                    part_hi[0 + 4] = __uint_as_float(_ldv8_61_4);
                                    part_hi[0 + 5] = __uint_as_float(_ldv8_61_5);
                                    part_hi[0 + 6] = __uint_as_float(_ldv8_61_6);
                                    part_hi[0 + 7] = __uint_as_float(_ldv8_61_7);
                                }
                                {
                                    unsigned _ldv8_62_0;
                                    unsigned _ldv8_62_1;
                                    unsigned _ldv8_62_2;
                                    unsigned _ldv8_62_3;
                                    unsigned _ldv8_62_4;
                                    unsigned _ldv8_62_5;
                                    unsigned _ldv8_62_6;
                                    unsigned _ldv8_62_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_62_0), "=r"(_ldv8_62_1), "=r"(_ldv8_62_2), "=r"(_ldv8_62_3), "=r"(_ldv8_62_4), "=r"(_ldv8_62_5), "=r"(_ldv8_62_6), "=r"(_ldv8_62_7) : "l"((const void*)(part_o + (off_0_1 + 64 + 8))) : "memory");
                                    part_hi[8 + 0] = __uint_as_float(_ldv8_62_0);
                                    part_hi[8 + 1] = __uint_as_float(_ldv8_62_1);
                                    part_hi[8 + 2] = __uint_as_float(_ldv8_62_2);
                                    part_hi[8 + 3] = __uint_as_float(_ldv8_62_3);
                                    part_hi[8 + 4] = __uint_as_float(_ldv8_62_4);
                                    part_hi[8 + 5] = __uint_as_float(_ldv8_62_5);
                                    part_hi[8 + 6] = __uint_as_float(_ldv8_62_6);
                                    part_hi[8 + 7] = __uint_as_float(_ldv8_62_7);
                                }
                                acc[0] = __fmaf_rn(part[0], w_c_5, acc[0]);
                                acc_hi[0] = __fmaf_rn(part_hi[0], w_c_5, acc_hi[0]);
                                acc[1] = __fmaf_rn(part[1], w_c_5, acc[1]);
                                acc_hi[1] = __fmaf_rn(part_hi[1], w_c_5, acc_hi[1]);
                                acc[2] = __fmaf_rn(part[2], w_c_5, acc[2]);
                                acc_hi[2] = __fmaf_rn(part_hi[2], w_c_5, acc_hi[2]);
                                acc[3] = __fmaf_rn(part[3], w_c_5, acc[3]);
                                acc_hi[3] = __fmaf_rn(part_hi[3], w_c_5, acc_hi[3]);
                                acc[4] = __fmaf_rn(part[4], w_c_5, acc[4]);
                                acc_hi[4] = __fmaf_rn(part_hi[4], w_c_5, acc_hi[4]);
                                acc[5] = __fmaf_rn(part[5], w_c_5, acc[5]);
                                acc_hi[5] = __fmaf_rn(part_hi[5], w_c_5, acc_hi[5]);
                                acc[6] = __fmaf_rn(part[6], w_c_5, acc[6]);
                                acc_hi[6] = __fmaf_rn(part_hi[6], w_c_5, acc_hi[6]);
                                acc[7] = __fmaf_rn(part[7], w_c_5, acc[7]);
                                acc_hi[7] = __fmaf_rn(part_hi[7], w_c_5, acc_hi[7]);
                                acc[8] = __fmaf_rn(part[8], w_c_5, acc[8]);
                                acc_hi[8] = __fmaf_rn(part_hi[8], w_c_5, acc_hi[8]);
                                acc[9] = __fmaf_rn(part[9], w_c_5, acc[9]);
                                acc_hi[9] = __fmaf_rn(part_hi[9], w_c_5, acc_hi[9]);
                                acc[10] = __fmaf_rn(part[10], w_c_5, acc[10]);
                                acc_hi[10] = __fmaf_rn(part_hi[10], w_c_5, acc_hi[10]);
                                acc[11] = __fmaf_rn(part[11], w_c_5, acc[11]);
                                acc_hi[11] = __fmaf_rn(part_hi[11], w_c_5, acc_hi[11]);
                                acc[12] = __fmaf_rn(part[12], w_c_5, acc[12]);
                                acc_hi[12] = __fmaf_rn(part_hi[12], w_c_5, acc_hi[12]);
                                acc[13] = __fmaf_rn(part[13], w_c_5, acc[13]);
                                acc_hi[13] = __fmaf_rn(part_hi[13], w_c_5, acc_hi[13]);
                                acc[14] = __fmaf_rn(part[14], w_c_5, acc[14]);
                                acc_hi[14] = __fmaf_rn(part_hi[14], w_c_5, acc_hi[14]);
                                acc[15] = __fmaf_rn(part[15], w_c_5, acc[15]);
                                acc_hi[15] = __fmaf_rn(part_hi[15], w_c_5, acc_hi[15]);
                            }
                            if (n_chunks_u_0 > 2) {
                                unsigned int crow_2 = (tile_slot0 + 2) * 512 + part_row;
                                float w_c_6 = approx_exp2(part_ml[crow_2 * 2] - row_max_all);
                                unsigned int off_0_2 = crow_2 * 128;
                                {
                                    unsigned _ldv8_63_0;
                                    unsigned _ldv8_63_1;
                                    unsigned _ldv8_63_2;
                                    unsigned _ldv8_63_3;
                                    unsigned _ldv8_63_4;
                                    unsigned _ldv8_63_5;
                                    unsigned _ldv8_63_6;
                                    unsigned _ldv8_63_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_63_0), "=r"(_ldv8_63_1), "=r"(_ldv8_63_2), "=r"(_ldv8_63_3), "=r"(_ldv8_63_4), "=r"(_ldv8_63_5), "=r"(_ldv8_63_6), "=r"(_ldv8_63_7) : "l"((const void*)(part_o + (off_0_2))) : "memory");
                                    part[0 + 0] = __uint_as_float(_ldv8_63_0);
                                    part[0 + 1] = __uint_as_float(_ldv8_63_1);
                                    part[0 + 2] = __uint_as_float(_ldv8_63_2);
                                    part[0 + 3] = __uint_as_float(_ldv8_63_3);
                                    part[0 + 4] = __uint_as_float(_ldv8_63_4);
                                    part[0 + 5] = __uint_as_float(_ldv8_63_5);
                                    part[0 + 6] = __uint_as_float(_ldv8_63_6);
                                    part[0 + 7] = __uint_as_float(_ldv8_63_7);
                                }
                                {
                                    unsigned _ldv8_64_0;
                                    unsigned _ldv8_64_1;
                                    unsigned _ldv8_64_2;
                                    unsigned _ldv8_64_3;
                                    unsigned _ldv8_64_4;
                                    unsigned _ldv8_64_5;
                                    unsigned _ldv8_64_6;
                                    unsigned _ldv8_64_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_64_0), "=r"(_ldv8_64_1), "=r"(_ldv8_64_2), "=r"(_ldv8_64_3), "=r"(_ldv8_64_4), "=r"(_ldv8_64_5), "=r"(_ldv8_64_6), "=r"(_ldv8_64_7) : "l"((const void*)(part_o + (off_0_2 + 8))) : "memory");
                                    part[8 + 0] = __uint_as_float(_ldv8_64_0);
                                    part[8 + 1] = __uint_as_float(_ldv8_64_1);
                                    part[8 + 2] = __uint_as_float(_ldv8_64_2);
                                    part[8 + 3] = __uint_as_float(_ldv8_64_3);
                                    part[8 + 4] = __uint_as_float(_ldv8_64_4);
                                    part[8 + 5] = __uint_as_float(_ldv8_64_5);
                                    part[8 + 6] = __uint_as_float(_ldv8_64_6);
                                    part[8 + 7] = __uint_as_float(_ldv8_64_7);
                                }
                                {
                                    unsigned _ldv8_65_0;
                                    unsigned _ldv8_65_1;
                                    unsigned _ldv8_65_2;
                                    unsigned _ldv8_65_3;
                                    unsigned _ldv8_65_4;
                                    unsigned _ldv8_65_5;
                                    unsigned _ldv8_65_6;
                                    unsigned _ldv8_65_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_65_0), "=r"(_ldv8_65_1), "=r"(_ldv8_65_2), "=r"(_ldv8_65_3), "=r"(_ldv8_65_4), "=r"(_ldv8_65_5), "=r"(_ldv8_65_6), "=r"(_ldv8_65_7) : "l"((const void*)(part_o + (off_0_2 + 64))) : "memory");
                                    part_hi[0 + 0] = __uint_as_float(_ldv8_65_0);
                                    part_hi[0 + 1] = __uint_as_float(_ldv8_65_1);
                                    part_hi[0 + 2] = __uint_as_float(_ldv8_65_2);
                                    part_hi[0 + 3] = __uint_as_float(_ldv8_65_3);
                                    part_hi[0 + 4] = __uint_as_float(_ldv8_65_4);
                                    part_hi[0 + 5] = __uint_as_float(_ldv8_65_5);
                                    part_hi[0 + 6] = __uint_as_float(_ldv8_65_6);
                                    part_hi[0 + 7] = __uint_as_float(_ldv8_65_7);
                                }
                                {
                                    unsigned _ldv8_66_0;
                                    unsigned _ldv8_66_1;
                                    unsigned _ldv8_66_2;
                                    unsigned _ldv8_66_3;
                                    unsigned _ldv8_66_4;
                                    unsigned _ldv8_66_5;
                                    unsigned _ldv8_66_6;
                                    unsigned _ldv8_66_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_66_0), "=r"(_ldv8_66_1), "=r"(_ldv8_66_2), "=r"(_ldv8_66_3), "=r"(_ldv8_66_4), "=r"(_ldv8_66_5), "=r"(_ldv8_66_6), "=r"(_ldv8_66_7) : "l"((const void*)(part_o + (off_0_2 + 64 + 8))) : "memory");
                                    part_hi[8 + 0] = __uint_as_float(_ldv8_66_0);
                                    part_hi[8 + 1] = __uint_as_float(_ldv8_66_1);
                                    part_hi[8 + 2] = __uint_as_float(_ldv8_66_2);
                                    part_hi[8 + 3] = __uint_as_float(_ldv8_66_3);
                                    part_hi[8 + 4] = __uint_as_float(_ldv8_66_4);
                                    part_hi[8 + 5] = __uint_as_float(_ldv8_66_5);
                                    part_hi[8 + 6] = __uint_as_float(_ldv8_66_6);
                                    part_hi[8 + 7] = __uint_as_float(_ldv8_66_7);
                                }
                                acc[0] = __fmaf_rn(part[0], w_c_6, acc[0]);
                                acc_hi[0] = __fmaf_rn(part_hi[0], w_c_6, acc_hi[0]);
                                acc[1] = __fmaf_rn(part[1], w_c_6, acc[1]);
                                acc_hi[1] = __fmaf_rn(part_hi[1], w_c_6, acc_hi[1]);
                                acc[2] = __fmaf_rn(part[2], w_c_6, acc[2]);
                                acc_hi[2] = __fmaf_rn(part_hi[2], w_c_6, acc_hi[2]);
                                acc[3] = __fmaf_rn(part[3], w_c_6, acc[3]);
                                acc_hi[3] = __fmaf_rn(part_hi[3], w_c_6, acc_hi[3]);
                                acc[4] = __fmaf_rn(part[4], w_c_6, acc[4]);
                                acc_hi[4] = __fmaf_rn(part_hi[4], w_c_6, acc_hi[4]);
                                acc[5] = __fmaf_rn(part[5], w_c_6, acc[5]);
                                acc_hi[5] = __fmaf_rn(part_hi[5], w_c_6, acc_hi[5]);
                                acc[6] = __fmaf_rn(part[6], w_c_6, acc[6]);
                                acc_hi[6] = __fmaf_rn(part_hi[6], w_c_6, acc_hi[6]);
                                acc[7] = __fmaf_rn(part[7], w_c_6, acc[7]);
                                acc_hi[7] = __fmaf_rn(part_hi[7], w_c_6, acc_hi[7]);
                                acc[8] = __fmaf_rn(part[8], w_c_6, acc[8]);
                                acc_hi[8] = __fmaf_rn(part_hi[8], w_c_6, acc_hi[8]);
                                acc[9] = __fmaf_rn(part[9], w_c_6, acc[9]);
                                acc_hi[9] = __fmaf_rn(part_hi[9], w_c_6, acc_hi[9]);
                                acc[10] = __fmaf_rn(part[10], w_c_6, acc[10]);
                                acc_hi[10] = __fmaf_rn(part_hi[10], w_c_6, acc_hi[10]);
                                acc[11] = __fmaf_rn(part[11], w_c_6, acc[11]);
                                acc_hi[11] = __fmaf_rn(part_hi[11], w_c_6, acc_hi[11]);
                                acc[12] = __fmaf_rn(part[12], w_c_6, acc[12]);
                                acc_hi[12] = __fmaf_rn(part_hi[12], w_c_6, acc_hi[12]);
                                acc[13] = __fmaf_rn(part[13], w_c_6, acc[13]);
                                acc_hi[13] = __fmaf_rn(part_hi[13], w_c_6, acc_hi[13]);
                                acc[14] = __fmaf_rn(part[14], w_c_6, acc[14]);
                                acc_hi[14] = __fmaf_rn(part_hi[14], w_c_6, acc_hi[14]);
                                acc[15] = __fmaf_rn(part[15], w_c_6, acc[15]);
                                acc_hi[15] = __fmaf_rn(part_hi[15], w_c_6, acc_hi[15]);
                            }
                            if (n_chunks_u_0 > 3) {
                                unsigned int crow_3 = (tile_slot0 + 3) * 512 + part_row;
                                float w_c_7 = approx_exp2(part_ml[crow_3 * 2] - row_max_all);
                                unsigned int off_0_3 = crow_3 * 128;
                                {
                                    unsigned _ldv8_67_0;
                                    unsigned _ldv8_67_1;
                                    unsigned _ldv8_67_2;
                                    unsigned _ldv8_67_3;
                                    unsigned _ldv8_67_4;
                                    unsigned _ldv8_67_5;
                                    unsigned _ldv8_67_6;
                                    unsigned _ldv8_67_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_67_0), "=r"(_ldv8_67_1), "=r"(_ldv8_67_2), "=r"(_ldv8_67_3), "=r"(_ldv8_67_4), "=r"(_ldv8_67_5), "=r"(_ldv8_67_6), "=r"(_ldv8_67_7) : "l"((const void*)(part_o + (off_0_3))) : "memory");
                                    part[0 + 0] = __uint_as_float(_ldv8_67_0);
                                    part[0 + 1] = __uint_as_float(_ldv8_67_1);
                                    part[0 + 2] = __uint_as_float(_ldv8_67_2);
                                    part[0 + 3] = __uint_as_float(_ldv8_67_3);
                                    part[0 + 4] = __uint_as_float(_ldv8_67_4);
                                    part[0 + 5] = __uint_as_float(_ldv8_67_5);
                                    part[0 + 6] = __uint_as_float(_ldv8_67_6);
                                    part[0 + 7] = __uint_as_float(_ldv8_67_7);
                                }
                                {
                                    unsigned _ldv8_68_0;
                                    unsigned _ldv8_68_1;
                                    unsigned _ldv8_68_2;
                                    unsigned _ldv8_68_3;
                                    unsigned _ldv8_68_4;
                                    unsigned _ldv8_68_5;
                                    unsigned _ldv8_68_6;
                                    unsigned _ldv8_68_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_68_0), "=r"(_ldv8_68_1), "=r"(_ldv8_68_2), "=r"(_ldv8_68_3), "=r"(_ldv8_68_4), "=r"(_ldv8_68_5), "=r"(_ldv8_68_6), "=r"(_ldv8_68_7) : "l"((const void*)(part_o + (off_0_3 + 8))) : "memory");
                                    part[8 + 0] = __uint_as_float(_ldv8_68_0);
                                    part[8 + 1] = __uint_as_float(_ldv8_68_1);
                                    part[8 + 2] = __uint_as_float(_ldv8_68_2);
                                    part[8 + 3] = __uint_as_float(_ldv8_68_3);
                                    part[8 + 4] = __uint_as_float(_ldv8_68_4);
                                    part[8 + 5] = __uint_as_float(_ldv8_68_5);
                                    part[8 + 6] = __uint_as_float(_ldv8_68_6);
                                    part[8 + 7] = __uint_as_float(_ldv8_68_7);
                                }
                                {
                                    unsigned _ldv8_69_0;
                                    unsigned _ldv8_69_1;
                                    unsigned _ldv8_69_2;
                                    unsigned _ldv8_69_3;
                                    unsigned _ldv8_69_4;
                                    unsigned _ldv8_69_5;
                                    unsigned _ldv8_69_6;
                                    unsigned _ldv8_69_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_69_0), "=r"(_ldv8_69_1), "=r"(_ldv8_69_2), "=r"(_ldv8_69_3), "=r"(_ldv8_69_4), "=r"(_ldv8_69_5), "=r"(_ldv8_69_6), "=r"(_ldv8_69_7) : "l"((const void*)(part_o + (off_0_3 + 64))) : "memory");
                                    part_hi[0 + 0] = __uint_as_float(_ldv8_69_0);
                                    part_hi[0 + 1] = __uint_as_float(_ldv8_69_1);
                                    part_hi[0 + 2] = __uint_as_float(_ldv8_69_2);
                                    part_hi[0 + 3] = __uint_as_float(_ldv8_69_3);
                                    part_hi[0 + 4] = __uint_as_float(_ldv8_69_4);
                                    part_hi[0 + 5] = __uint_as_float(_ldv8_69_5);
                                    part_hi[0 + 6] = __uint_as_float(_ldv8_69_6);
                                    part_hi[0 + 7] = __uint_as_float(_ldv8_69_7);
                                }
                                {
                                    unsigned _ldv8_70_0;
                                    unsigned _ldv8_70_1;
                                    unsigned _ldv8_70_2;
                                    unsigned _ldv8_70_3;
                                    unsigned _ldv8_70_4;
                                    unsigned _ldv8_70_5;
                                    unsigned _ldv8_70_6;
                                    unsigned _ldv8_70_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_70_0), "=r"(_ldv8_70_1), "=r"(_ldv8_70_2), "=r"(_ldv8_70_3), "=r"(_ldv8_70_4), "=r"(_ldv8_70_5), "=r"(_ldv8_70_6), "=r"(_ldv8_70_7) : "l"((const void*)(part_o + (off_0_3 + 64 + 8))) : "memory");
                                    part_hi[8 + 0] = __uint_as_float(_ldv8_70_0);
                                    part_hi[8 + 1] = __uint_as_float(_ldv8_70_1);
                                    part_hi[8 + 2] = __uint_as_float(_ldv8_70_2);
                                    part_hi[8 + 3] = __uint_as_float(_ldv8_70_3);
                                    part_hi[8 + 4] = __uint_as_float(_ldv8_70_4);
                                    part_hi[8 + 5] = __uint_as_float(_ldv8_70_5);
                                    part_hi[8 + 6] = __uint_as_float(_ldv8_70_6);
                                    part_hi[8 + 7] = __uint_as_float(_ldv8_70_7);
                                }
                                acc[0] = __fmaf_rn(part[0], w_c_7, acc[0]);
                                acc_hi[0] = __fmaf_rn(part_hi[0], w_c_7, acc_hi[0]);
                                acc[1] = __fmaf_rn(part[1], w_c_7, acc[1]);
                                acc_hi[1] = __fmaf_rn(part_hi[1], w_c_7, acc_hi[1]);
                                acc[2] = __fmaf_rn(part[2], w_c_7, acc[2]);
                                acc_hi[2] = __fmaf_rn(part_hi[2], w_c_7, acc_hi[2]);
                                acc[3] = __fmaf_rn(part[3], w_c_7, acc[3]);
                                acc_hi[3] = __fmaf_rn(part_hi[3], w_c_7, acc_hi[3]);
                                acc[4] = __fmaf_rn(part[4], w_c_7, acc[4]);
                                acc_hi[4] = __fmaf_rn(part_hi[4], w_c_7, acc_hi[4]);
                                acc[5] = __fmaf_rn(part[5], w_c_7, acc[5]);
                                acc_hi[5] = __fmaf_rn(part_hi[5], w_c_7, acc_hi[5]);
                                acc[6] = __fmaf_rn(part[6], w_c_7, acc[6]);
                                acc_hi[6] = __fmaf_rn(part_hi[6], w_c_7, acc_hi[6]);
                                acc[7] = __fmaf_rn(part[7], w_c_7, acc[7]);
                                acc_hi[7] = __fmaf_rn(part_hi[7], w_c_7, acc_hi[7]);
                                acc[8] = __fmaf_rn(part[8], w_c_7, acc[8]);
                                acc_hi[8] = __fmaf_rn(part_hi[8], w_c_7, acc_hi[8]);
                                acc[9] = __fmaf_rn(part[9], w_c_7, acc[9]);
                                acc_hi[9] = __fmaf_rn(part_hi[9], w_c_7, acc_hi[9]);
                                acc[10] = __fmaf_rn(part[10], w_c_7, acc[10]);
                                acc_hi[10] = __fmaf_rn(part_hi[10], w_c_7, acc_hi[10]);
                                acc[11] = __fmaf_rn(part[11], w_c_7, acc[11]);
                                acc_hi[11] = __fmaf_rn(part_hi[11], w_c_7, acc_hi[11]);
                                acc[12] = __fmaf_rn(part[12], w_c_7, acc[12]);
                                acc_hi[12] = __fmaf_rn(part_hi[12], w_c_7, acc_hi[12]);
                                acc[13] = __fmaf_rn(part[13], w_c_7, acc[13]);
                                acc_hi[13] = __fmaf_rn(part_hi[13], w_c_7, acc_hi[13]);
                                acc[14] = __fmaf_rn(part[14], w_c_7, acc[14]);
                                acc_hi[14] = __fmaf_rn(part_hi[14], w_c_7, acc_hi[14]);
                                acc[15] = __fmaf_rn(part[15], w_c_7, acc[15]);
                                acc_hi[15] = __fmaf_rn(part_hi[15], w_c_7, acc_hi[15]);
                            }
                            const float2 _scale2_71 = {final_scale, final_scale};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(acc)[_ls], _scale2_71);
                            unsigned int acc_bf16[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc[_lp*2 + 0], acc[_lp*2+1 + 0]));
                                acc_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o00_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 ^ ((warp % 4 * 32 + lane) * 128 >> 7 & 7) << 4))), "r"(acc_bf16[0]), "r"(acc_bf16[1]), "r"(acc_bf16[2]), "r"(acc_bf16[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o00_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + 16 ^ ((warp % 4 * 32 + lane) * 128 + 16 >> 7 & 7) << 4))), "r"(acc_bf16[4]), "r"(acc_bf16[5]), "r"(acc_bf16[6]), "r"(acc_bf16[7]) : "memory");
                            const float2 _scale2_72 = {final_scale, final_scale};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(acc_hi)[_ls], _scale2_72);
                            unsigned int acc_hi_bf16[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_hi[_lp*2 + 0], acc_hi[_lp*2+1 + 0]));
                                acc_hi_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o01_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 ^ ((warp % 4 * 32 + lane) * 128 >> 7 & 7) << 4))), "r"(acc_hi_bf16[0]), "r"(acc_hi_bf16[1]), "r"(acc_hi_bf16[2]), "r"(acc_hi_bf16[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o01_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + 16 ^ ((warp % 4 * 32 + lane) * 128 + 16 >> 7 & 7) << 4))), "r"(acc_hi_bf16[4]), "r"(acc_hi_bf16[5]), "r"(acc_hi_bf16[6]), "r"(acc_hi_bf16[7]) : "memory");
                            float acc_0[16];
                            float acc_hi_1[16];
                            acc_0[0] = 0.0f;
                            acc_hi_1[0] = 0.0f;
                            acc_0[1] = 0.0f;
                            acc_hi_1[1] = 0.0f;
                            acc_0[2] = 0.0f;
                            acc_hi_1[2] = 0.0f;
                            acc_0[3] = 0.0f;
                            acc_hi_1[3] = 0.0f;
                            acc_0[4] = 0.0f;
                            acc_hi_1[4] = 0.0f;
                            acc_0[5] = 0.0f;
                            acc_hi_1[5] = 0.0f;
                            acc_0[6] = 0.0f;
                            acc_hi_1[6] = 0.0f;
                            acc_0[7] = 0.0f;
                            acc_hi_1[7] = 0.0f;
                            acc_0[8] = 0.0f;
                            acc_hi_1[8] = 0.0f;
                            acc_0[9] = 0.0f;
                            acc_hi_1[9] = 0.0f;
                            acc_0[10] = 0.0f;
                            acc_hi_1[10] = 0.0f;
                            acc_0[11] = 0.0f;
                            acc_hi_1[11] = 0.0f;
                            acc_0[12] = 0.0f;
                            acc_hi_1[12] = 0.0f;
                            acc_0[13] = 0.0f;
                            acc_hi_1[13] = 0.0f;
                            acc_0[14] = 0.0f;
                            acc_hi_1[14] = 0.0f;
                            acc_0[15] = 0.0f;
                            acc_hi_1[15] = 0.0f;
                            float part_2[16];
                            float part_hi_3[16];
                            if (n_chunks_u_0 > 0) {
                                unsigned int crow_4 = tile_slot0 * 512 + part_row;
                                float w_c_8 = approx_exp2(part_ml[crow_4 * 2] - row_max_all);
                                unsigned int off_0_4 = crow_4 * 128 + 16;
                                {
                                    unsigned _ldv8_73_0;
                                    unsigned _ldv8_73_1;
                                    unsigned _ldv8_73_2;
                                    unsigned _ldv8_73_3;
                                    unsigned _ldv8_73_4;
                                    unsigned _ldv8_73_5;
                                    unsigned _ldv8_73_6;
                                    unsigned _ldv8_73_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_73_0), "=r"(_ldv8_73_1), "=r"(_ldv8_73_2), "=r"(_ldv8_73_3), "=r"(_ldv8_73_4), "=r"(_ldv8_73_5), "=r"(_ldv8_73_6), "=r"(_ldv8_73_7) : "l"((const void*)(part_o + (off_0_4))) : "memory");
                                    part_2[0 + 0] = __uint_as_float(_ldv8_73_0);
                                    part_2[0 + 1] = __uint_as_float(_ldv8_73_1);
                                    part_2[0 + 2] = __uint_as_float(_ldv8_73_2);
                                    part_2[0 + 3] = __uint_as_float(_ldv8_73_3);
                                    part_2[0 + 4] = __uint_as_float(_ldv8_73_4);
                                    part_2[0 + 5] = __uint_as_float(_ldv8_73_5);
                                    part_2[0 + 6] = __uint_as_float(_ldv8_73_6);
                                    part_2[0 + 7] = __uint_as_float(_ldv8_73_7);
                                }
                                {
                                    unsigned _ldv8_74_0;
                                    unsigned _ldv8_74_1;
                                    unsigned _ldv8_74_2;
                                    unsigned _ldv8_74_3;
                                    unsigned _ldv8_74_4;
                                    unsigned _ldv8_74_5;
                                    unsigned _ldv8_74_6;
                                    unsigned _ldv8_74_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_74_0), "=r"(_ldv8_74_1), "=r"(_ldv8_74_2), "=r"(_ldv8_74_3), "=r"(_ldv8_74_4), "=r"(_ldv8_74_5), "=r"(_ldv8_74_6), "=r"(_ldv8_74_7) : "l"((const void*)(part_o + (off_0_4 + 8))) : "memory");
                                    part_2[8 + 0] = __uint_as_float(_ldv8_74_0);
                                    part_2[8 + 1] = __uint_as_float(_ldv8_74_1);
                                    part_2[8 + 2] = __uint_as_float(_ldv8_74_2);
                                    part_2[8 + 3] = __uint_as_float(_ldv8_74_3);
                                    part_2[8 + 4] = __uint_as_float(_ldv8_74_4);
                                    part_2[8 + 5] = __uint_as_float(_ldv8_74_5);
                                    part_2[8 + 6] = __uint_as_float(_ldv8_74_6);
                                    part_2[8 + 7] = __uint_as_float(_ldv8_74_7);
                                }
                                {
                                    unsigned _ldv8_75_0;
                                    unsigned _ldv8_75_1;
                                    unsigned _ldv8_75_2;
                                    unsigned _ldv8_75_3;
                                    unsigned _ldv8_75_4;
                                    unsigned _ldv8_75_5;
                                    unsigned _ldv8_75_6;
                                    unsigned _ldv8_75_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_75_0), "=r"(_ldv8_75_1), "=r"(_ldv8_75_2), "=r"(_ldv8_75_3), "=r"(_ldv8_75_4), "=r"(_ldv8_75_5), "=r"(_ldv8_75_6), "=r"(_ldv8_75_7) : "l"((const void*)(part_o + (off_0_4 + 64))) : "memory");
                                    part_hi_3[0 + 0] = __uint_as_float(_ldv8_75_0);
                                    part_hi_3[0 + 1] = __uint_as_float(_ldv8_75_1);
                                    part_hi_3[0 + 2] = __uint_as_float(_ldv8_75_2);
                                    part_hi_3[0 + 3] = __uint_as_float(_ldv8_75_3);
                                    part_hi_3[0 + 4] = __uint_as_float(_ldv8_75_4);
                                    part_hi_3[0 + 5] = __uint_as_float(_ldv8_75_5);
                                    part_hi_3[0 + 6] = __uint_as_float(_ldv8_75_6);
                                    part_hi_3[0 + 7] = __uint_as_float(_ldv8_75_7);
                                }
                                {
                                    unsigned _ldv8_76_0;
                                    unsigned _ldv8_76_1;
                                    unsigned _ldv8_76_2;
                                    unsigned _ldv8_76_3;
                                    unsigned _ldv8_76_4;
                                    unsigned _ldv8_76_5;
                                    unsigned _ldv8_76_6;
                                    unsigned _ldv8_76_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_76_0), "=r"(_ldv8_76_1), "=r"(_ldv8_76_2), "=r"(_ldv8_76_3), "=r"(_ldv8_76_4), "=r"(_ldv8_76_5), "=r"(_ldv8_76_6), "=r"(_ldv8_76_7) : "l"((const void*)(part_o + (off_0_4 + 64 + 8))) : "memory");
                                    part_hi_3[8 + 0] = __uint_as_float(_ldv8_76_0);
                                    part_hi_3[8 + 1] = __uint_as_float(_ldv8_76_1);
                                    part_hi_3[8 + 2] = __uint_as_float(_ldv8_76_2);
                                    part_hi_3[8 + 3] = __uint_as_float(_ldv8_76_3);
                                    part_hi_3[8 + 4] = __uint_as_float(_ldv8_76_4);
                                    part_hi_3[8 + 5] = __uint_as_float(_ldv8_76_5);
                                    part_hi_3[8 + 6] = __uint_as_float(_ldv8_76_6);
                                    part_hi_3[8 + 7] = __uint_as_float(_ldv8_76_7);
                                }
                                acc_0[0] = __fmaf_rn(part_2[0], w_c_8, acc_0[0]);
                                acc_hi_1[0] = __fmaf_rn(part_hi_3[0], w_c_8, acc_hi_1[0]);
                                acc_0[1] = __fmaf_rn(part_2[1], w_c_8, acc_0[1]);
                                acc_hi_1[1] = __fmaf_rn(part_hi_3[1], w_c_8, acc_hi_1[1]);
                                acc_0[2] = __fmaf_rn(part_2[2], w_c_8, acc_0[2]);
                                acc_hi_1[2] = __fmaf_rn(part_hi_3[2], w_c_8, acc_hi_1[2]);
                                acc_0[3] = __fmaf_rn(part_2[3], w_c_8, acc_0[3]);
                                acc_hi_1[3] = __fmaf_rn(part_hi_3[3], w_c_8, acc_hi_1[3]);
                                acc_0[4] = __fmaf_rn(part_2[4], w_c_8, acc_0[4]);
                                acc_hi_1[4] = __fmaf_rn(part_hi_3[4], w_c_8, acc_hi_1[4]);
                                acc_0[5] = __fmaf_rn(part_2[5], w_c_8, acc_0[5]);
                                acc_hi_1[5] = __fmaf_rn(part_hi_3[5], w_c_8, acc_hi_1[5]);
                                acc_0[6] = __fmaf_rn(part_2[6], w_c_8, acc_0[6]);
                                acc_hi_1[6] = __fmaf_rn(part_hi_3[6], w_c_8, acc_hi_1[6]);
                                acc_0[7] = __fmaf_rn(part_2[7], w_c_8, acc_0[7]);
                                acc_hi_1[7] = __fmaf_rn(part_hi_3[7], w_c_8, acc_hi_1[7]);
                                acc_0[8] = __fmaf_rn(part_2[8], w_c_8, acc_0[8]);
                                acc_hi_1[8] = __fmaf_rn(part_hi_3[8], w_c_8, acc_hi_1[8]);
                                acc_0[9] = __fmaf_rn(part_2[9], w_c_8, acc_0[9]);
                                acc_hi_1[9] = __fmaf_rn(part_hi_3[9], w_c_8, acc_hi_1[9]);
                                acc_0[10] = __fmaf_rn(part_2[10], w_c_8, acc_0[10]);
                                acc_hi_1[10] = __fmaf_rn(part_hi_3[10], w_c_8, acc_hi_1[10]);
                                acc_0[11] = __fmaf_rn(part_2[11], w_c_8, acc_0[11]);
                                acc_hi_1[11] = __fmaf_rn(part_hi_3[11], w_c_8, acc_hi_1[11]);
                                acc_0[12] = __fmaf_rn(part_2[12], w_c_8, acc_0[12]);
                                acc_hi_1[12] = __fmaf_rn(part_hi_3[12], w_c_8, acc_hi_1[12]);
                                acc_0[13] = __fmaf_rn(part_2[13], w_c_8, acc_0[13]);
                                acc_hi_1[13] = __fmaf_rn(part_hi_3[13], w_c_8, acc_hi_1[13]);
                                acc_0[14] = __fmaf_rn(part_2[14], w_c_8, acc_0[14]);
                                acc_hi_1[14] = __fmaf_rn(part_hi_3[14], w_c_8, acc_hi_1[14]);
                                acc_0[15] = __fmaf_rn(part_2[15], w_c_8, acc_0[15]);
                                acc_hi_1[15] = __fmaf_rn(part_hi_3[15], w_c_8, acc_hi_1[15]);
                            }
                            if (n_chunks_u_0 > 1) {
                                unsigned int crow_5 = (tile_slot0 + 1) * 512 + part_row;
                                float w_c_9 = approx_exp2(part_ml[crow_5 * 2] - row_max_all);
                                unsigned int off_0_5 = crow_5 * 128 + 16;
                                {
                                    unsigned _ldv8_77_0;
                                    unsigned _ldv8_77_1;
                                    unsigned _ldv8_77_2;
                                    unsigned _ldv8_77_3;
                                    unsigned _ldv8_77_4;
                                    unsigned _ldv8_77_5;
                                    unsigned _ldv8_77_6;
                                    unsigned _ldv8_77_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_77_0), "=r"(_ldv8_77_1), "=r"(_ldv8_77_2), "=r"(_ldv8_77_3), "=r"(_ldv8_77_4), "=r"(_ldv8_77_5), "=r"(_ldv8_77_6), "=r"(_ldv8_77_7) : "l"((const void*)(part_o + (off_0_5))) : "memory");
                                    part_2[0 + 0] = __uint_as_float(_ldv8_77_0);
                                    part_2[0 + 1] = __uint_as_float(_ldv8_77_1);
                                    part_2[0 + 2] = __uint_as_float(_ldv8_77_2);
                                    part_2[0 + 3] = __uint_as_float(_ldv8_77_3);
                                    part_2[0 + 4] = __uint_as_float(_ldv8_77_4);
                                    part_2[0 + 5] = __uint_as_float(_ldv8_77_5);
                                    part_2[0 + 6] = __uint_as_float(_ldv8_77_6);
                                    part_2[0 + 7] = __uint_as_float(_ldv8_77_7);
                                }
                                {
                                    unsigned _ldv8_78_0;
                                    unsigned _ldv8_78_1;
                                    unsigned _ldv8_78_2;
                                    unsigned _ldv8_78_3;
                                    unsigned _ldv8_78_4;
                                    unsigned _ldv8_78_5;
                                    unsigned _ldv8_78_6;
                                    unsigned _ldv8_78_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_78_0), "=r"(_ldv8_78_1), "=r"(_ldv8_78_2), "=r"(_ldv8_78_3), "=r"(_ldv8_78_4), "=r"(_ldv8_78_5), "=r"(_ldv8_78_6), "=r"(_ldv8_78_7) : "l"((const void*)(part_o + (off_0_5 + 8))) : "memory");
                                    part_2[8 + 0] = __uint_as_float(_ldv8_78_0);
                                    part_2[8 + 1] = __uint_as_float(_ldv8_78_1);
                                    part_2[8 + 2] = __uint_as_float(_ldv8_78_2);
                                    part_2[8 + 3] = __uint_as_float(_ldv8_78_3);
                                    part_2[8 + 4] = __uint_as_float(_ldv8_78_4);
                                    part_2[8 + 5] = __uint_as_float(_ldv8_78_5);
                                    part_2[8 + 6] = __uint_as_float(_ldv8_78_6);
                                    part_2[8 + 7] = __uint_as_float(_ldv8_78_7);
                                }
                                {
                                    unsigned _ldv8_79_0;
                                    unsigned _ldv8_79_1;
                                    unsigned _ldv8_79_2;
                                    unsigned _ldv8_79_3;
                                    unsigned _ldv8_79_4;
                                    unsigned _ldv8_79_5;
                                    unsigned _ldv8_79_6;
                                    unsigned _ldv8_79_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_79_0), "=r"(_ldv8_79_1), "=r"(_ldv8_79_2), "=r"(_ldv8_79_3), "=r"(_ldv8_79_4), "=r"(_ldv8_79_5), "=r"(_ldv8_79_6), "=r"(_ldv8_79_7) : "l"((const void*)(part_o + (off_0_5 + 64))) : "memory");
                                    part_hi_3[0 + 0] = __uint_as_float(_ldv8_79_0);
                                    part_hi_3[0 + 1] = __uint_as_float(_ldv8_79_1);
                                    part_hi_3[0 + 2] = __uint_as_float(_ldv8_79_2);
                                    part_hi_3[0 + 3] = __uint_as_float(_ldv8_79_3);
                                    part_hi_3[0 + 4] = __uint_as_float(_ldv8_79_4);
                                    part_hi_3[0 + 5] = __uint_as_float(_ldv8_79_5);
                                    part_hi_3[0 + 6] = __uint_as_float(_ldv8_79_6);
                                    part_hi_3[0 + 7] = __uint_as_float(_ldv8_79_7);
                                }
                                {
                                    unsigned _ldv8_80_0;
                                    unsigned _ldv8_80_1;
                                    unsigned _ldv8_80_2;
                                    unsigned _ldv8_80_3;
                                    unsigned _ldv8_80_4;
                                    unsigned _ldv8_80_5;
                                    unsigned _ldv8_80_6;
                                    unsigned _ldv8_80_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_80_0), "=r"(_ldv8_80_1), "=r"(_ldv8_80_2), "=r"(_ldv8_80_3), "=r"(_ldv8_80_4), "=r"(_ldv8_80_5), "=r"(_ldv8_80_6), "=r"(_ldv8_80_7) : "l"((const void*)(part_o + (off_0_5 + 64 + 8))) : "memory");
                                    part_hi_3[8 + 0] = __uint_as_float(_ldv8_80_0);
                                    part_hi_3[8 + 1] = __uint_as_float(_ldv8_80_1);
                                    part_hi_3[8 + 2] = __uint_as_float(_ldv8_80_2);
                                    part_hi_3[8 + 3] = __uint_as_float(_ldv8_80_3);
                                    part_hi_3[8 + 4] = __uint_as_float(_ldv8_80_4);
                                    part_hi_3[8 + 5] = __uint_as_float(_ldv8_80_5);
                                    part_hi_3[8 + 6] = __uint_as_float(_ldv8_80_6);
                                    part_hi_3[8 + 7] = __uint_as_float(_ldv8_80_7);
                                }
                                acc_0[0] = __fmaf_rn(part_2[0], w_c_9, acc_0[0]);
                                acc_hi_1[0] = __fmaf_rn(part_hi_3[0], w_c_9, acc_hi_1[0]);
                                acc_0[1] = __fmaf_rn(part_2[1], w_c_9, acc_0[1]);
                                acc_hi_1[1] = __fmaf_rn(part_hi_3[1], w_c_9, acc_hi_1[1]);
                                acc_0[2] = __fmaf_rn(part_2[2], w_c_9, acc_0[2]);
                                acc_hi_1[2] = __fmaf_rn(part_hi_3[2], w_c_9, acc_hi_1[2]);
                                acc_0[3] = __fmaf_rn(part_2[3], w_c_9, acc_0[3]);
                                acc_hi_1[3] = __fmaf_rn(part_hi_3[3], w_c_9, acc_hi_1[3]);
                                acc_0[4] = __fmaf_rn(part_2[4], w_c_9, acc_0[4]);
                                acc_hi_1[4] = __fmaf_rn(part_hi_3[4], w_c_9, acc_hi_1[4]);
                                acc_0[5] = __fmaf_rn(part_2[5], w_c_9, acc_0[5]);
                                acc_hi_1[5] = __fmaf_rn(part_hi_3[5], w_c_9, acc_hi_1[5]);
                                acc_0[6] = __fmaf_rn(part_2[6], w_c_9, acc_0[6]);
                                acc_hi_1[6] = __fmaf_rn(part_hi_3[6], w_c_9, acc_hi_1[6]);
                                acc_0[7] = __fmaf_rn(part_2[7], w_c_9, acc_0[7]);
                                acc_hi_1[7] = __fmaf_rn(part_hi_3[7], w_c_9, acc_hi_1[7]);
                                acc_0[8] = __fmaf_rn(part_2[8], w_c_9, acc_0[8]);
                                acc_hi_1[8] = __fmaf_rn(part_hi_3[8], w_c_9, acc_hi_1[8]);
                                acc_0[9] = __fmaf_rn(part_2[9], w_c_9, acc_0[9]);
                                acc_hi_1[9] = __fmaf_rn(part_hi_3[9], w_c_9, acc_hi_1[9]);
                                acc_0[10] = __fmaf_rn(part_2[10], w_c_9, acc_0[10]);
                                acc_hi_1[10] = __fmaf_rn(part_hi_3[10], w_c_9, acc_hi_1[10]);
                                acc_0[11] = __fmaf_rn(part_2[11], w_c_9, acc_0[11]);
                                acc_hi_1[11] = __fmaf_rn(part_hi_3[11], w_c_9, acc_hi_1[11]);
                                acc_0[12] = __fmaf_rn(part_2[12], w_c_9, acc_0[12]);
                                acc_hi_1[12] = __fmaf_rn(part_hi_3[12], w_c_9, acc_hi_1[12]);
                                acc_0[13] = __fmaf_rn(part_2[13], w_c_9, acc_0[13]);
                                acc_hi_1[13] = __fmaf_rn(part_hi_3[13], w_c_9, acc_hi_1[13]);
                                acc_0[14] = __fmaf_rn(part_2[14], w_c_9, acc_0[14]);
                                acc_hi_1[14] = __fmaf_rn(part_hi_3[14], w_c_9, acc_hi_1[14]);
                                acc_0[15] = __fmaf_rn(part_2[15], w_c_9, acc_0[15]);
                                acc_hi_1[15] = __fmaf_rn(part_hi_3[15], w_c_9, acc_hi_1[15]);
                            }
                            if (n_chunks_u_0 > 2) {
                                unsigned int crow_6 = (tile_slot0 + 2) * 512 + part_row;
                                float w_c_10 = approx_exp2(part_ml[crow_6 * 2] - row_max_all);
                                unsigned int off_0_6 = crow_6 * 128 + 16;
                                {
                                    unsigned _ldv8_81_0;
                                    unsigned _ldv8_81_1;
                                    unsigned _ldv8_81_2;
                                    unsigned _ldv8_81_3;
                                    unsigned _ldv8_81_4;
                                    unsigned _ldv8_81_5;
                                    unsigned _ldv8_81_6;
                                    unsigned _ldv8_81_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_81_0), "=r"(_ldv8_81_1), "=r"(_ldv8_81_2), "=r"(_ldv8_81_3), "=r"(_ldv8_81_4), "=r"(_ldv8_81_5), "=r"(_ldv8_81_6), "=r"(_ldv8_81_7) : "l"((const void*)(part_o + (off_0_6))) : "memory");
                                    part_2[0 + 0] = __uint_as_float(_ldv8_81_0);
                                    part_2[0 + 1] = __uint_as_float(_ldv8_81_1);
                                    part_2[0 + 2] = __uint_as_float(_ldv8_81_2);
                                    part_2[0 + 3] = __uint_as_float(_ldv8_81_3);
                                    part_2[0 + 4] = __uint_as_float(_ldv8_81_4);
                                    part_2[0 + 5] = __uint_as_float(_ldv8_81_5);
                                    part_2[0 + 6] = __uint_as_float(_ldv8_81_6);
                                    part_2[0 + 7] = __uint_as_float(_ldv8_81_7);
                                }
                                {
                                    unsigned _ldv8_82_0;
                                    unsigned _ldv8_82_1;
                                    unsigned _ldv8_82_2;
                                    unsigned _ldv8_82_3;
                                    unsigned _ldv8_82_4;
                                    unsigned _ldv8_82_5;
                                    unsigned _ldv8_82_6;
                                    unsigned _ldv8_82_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_82_0), "=r"(_ldv8_82_1), "=r"(_ldv8_82_2), "=r"(_ldv8_82_3), "=r"(_ldv8_82_4), "=r"(_ldv8_82_5), "=r"(_ldv8_82_6), "=r"(_ldv8_82_7) : "l"((const void*)(part_o + (off_0_6 + 8))) : "memory");
                                    part_2[8 + 0] = __uint_as_float(_ldv8_82_0);
                                    part_2[8 + 1] = __uint_as_float(_ldv8_82_1);
                                    part_2[8 + 2] = __uint_as_float(_ldv8_82_2);
                                    part_2[8 + 3] = __uint_as_float(_ldv8_82_3);
                                    part_2[8 + 4] = __uint_as_float(_ldv8_82_4);
                                    part_2[8 + 5] = __uint_as_float(_ldv8_82_5);
                                    part_2[8 + 6] = __uint_as_float(_ldv8_82_6);
                                    part_2[8 + 7] = __uint_as_float(_ldv8_82_7);
                                }
                                {
                                    unsigned _ldv8_83_0;
                                    unsigned _ldv8_83_1;
                                    unsigned _ldv8_83_2;
                                    unsigned _ldv8_83_3;
                                    unsigned _ldv8_83_4;
                                    unsigned _ldv8_83_5;
                                    unsigned _ldv8_83_6;
                                    unsigned _ldv8_83_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_83_0), "=r"(_ldv8_83_1), "=r"(_ldv8_83_2), "=r"(_ldv8_83_3), "=r"(_ldv8_83_4), "=r"(_ldv8_83_5), "=r"(_ldv8_83_6), "=r"(_ldv8_83_7) : "l"((const void*)(part_o + (off_0_6 + 64))) : "memory");
                                    part_hi_3[0 + 0] = __uint_as_float(_ldv8_83_0);
                                    part_hi_3[0 + 1] = __uint_as_float(_ldv8_83_1);
                                    part_hi_3[0 + 2] = __uint_as_float(_ldv8_83_2);
                                    part_hi_3[0 + 3] = __uint_as_float(_ldv8_83_3);
                                    part_hi_3[0 + 4] = __uint_as_float(_ldv8_83_4);
                                    part_hi_3[0 + 5] = __uint_as_float(_ldv8_83_5);
                                    part_hi_3[0 + 6] = __uint_as_float(_ldv8_83_6);
                                    part_hi_3[0 + 7] = __uint_as_float(_ldv8_83_7);
                                }
                                {
                                    unsigned _ldv8_84_0;
                                    unsigned _ldv8_84_1;
                                    unsigned _ldv8_84_2;
                                    unsigned _ldv8_84_3;
                                    unsigned _ldv8_84_4;
                                    unsigned _ldv8_84_5;
                                    unsigned _ldv8_84_6;
                                    unsigned _ldv8_84_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_84_0), "=r"(_ldv8_84_1), "=r"(_ldv8_84_2), "=r"(_ldv8_84_3), "=r"(_ldv8_84_4), "=r"(_ldv8_84_5), "=r"(_ldv8_84_6), "=r"(_ldv8_84_7) : "l"((const void*)(part_o + (off_0_6 + 64 + 8))) : "memory");
                                    part_hi_3[8 + 0] = __uint_as_float(_ldv8_84_0);
                                    part_hi_3[8 + 1] = __uint_as_float(_ldv8_84_1);
                                    part_hi_3[8 + 2] = __uint_as_float(_ldv8_84_2);
                                    part_hi_3[8 + 3] = __uint_as_float(_ldv8_84_3);
                                    part_hi_3[8 + 4] = __uint_as_float(_ldv8_84_4);
                                    part_hi_3[8 + 5] = __uint_as_float(_ldv8_84_5);
                                    part_hi_3[8 + 6] = __uint_as_float(_ldv8_84_6);
                                    part_hi_3[8 + 7] = __uint_as_float(_ldv8_84_7);
                                }
                                acc_0[0] = __fmaf_rn(part_2[0], w_c_10, acc_0[0]);
                                acc_hi_1[0] = __fmaf_rn(part_hi_3[0], w_c_10, acc_hi_1[0]);
                                acc_0[1] = __fmaf_rn(part_2[1], w_c_10, acc_0[1]);
                                acc_hi_1[1] = __fmaf_rn(part_hi_3[1], w_c_10, acc_hi_1[1]);
                                acc_0[2] = __fmaf_rn(part_2[2], w_c_10, acc_0[2]);
                                acc_hi_1[2] = __fmaf_rn(part_hi_3[2], w_c_10, acc_hi_1[2]);
                                acc_0[3] = __fmaf_rn(part_2[3], w_c_10, acc_0[3]);
                                acc_hi_1[3] = __fmaf_rn(part_hi_3[3], w_c_10, acc_hi_1[3]);
                                acc_0[4] = __fmaf_rn(part_2[4], w_c_10, acc_0[4]);
                                acc_hi_1[4] = __fmaf_rn(part_hi_3[4], w_c_10, acc_hi_1[4]);
                                acc_0[5] = __fmaf_rn(part_2[5], w_c_10, acc_0[5]);
                                acc_hi_1[5] = __fmaf_rn(part_hi_3[5], w_c_10, acc_hi_1[5]);
                                acc_0[6] = __fmaf_rn(part_2[6], w_c_10, acc_0[6]);
                                acc_hi_1[6] = __fmaf_rn(part_hi_3[6], w_c_10, acc_hi_1[6]);
                                acc_0[7] = __fmaf_rn(part_2[7], w_c_10, acc_0[7]);
                                acc_hi_1[7] = __fmaf_rn(part_hi_3[7], w_c_10, acc_hi_1[7]);
                                acc_0[8] = __fmaf_rn(part_2[8], w_c_10, acc_0[8]);
                                acc_hi_1[8] = __fmaf_rn(part_hi_3[8], w_c_10, acc_hi_1[8]);
                                acc_0[9] = __fmaf_rn(part_2[9], w_c_10, acc_0[9]);
                                acc_hi_1[9] = __fmaf_rn(part_hi_3[9], w_c_10, acc_hi_1[9]);
                                acc_0[10] = __fmaf_rn(part_2[10], w_c_10, acc_0[10]);
                                acc_hi_1[10] = __fmaf_rn(part_hi_3[10], w_c_10, acc_hi_1[10]);
                                acc_0[11] = __fmaf_rn(part_2[11], w_c_10, acc_0[11]);
                                acc_hi_1[11] = __fmaf_rn(part_hi_3[11], w_c_10, acc_hi_1[11]);
                                acc_0[12] = __fmaf_rn(part_2[12], w_c_10, acc_0[12]);
                                acc_hi_1[12] = __fmaf_rn(part_hi_3[12], w_c_10, acc_hi_1[12]);
                                acc_0[13] = __fmaf_rn(part_2[13], w_c_10, acc_0[13]);
                                acc_hi_1[13] = __fmaf_rn(part_hi_3[13], w_c_10, acc_hi_1[13]);
                                acc_0[14] = __fmaf_rn(part_2[14], w_c_10, acc_0[14]);
                                acc_hi_1[14] = __fmaf_rn(part_hi_3[14], w_c_10, acc_hi_1[14]);
                                acc_0[15] = __fmaf_rn(part_2[15], w_c_10, acc_0[15]);
                                acc_hi_1[15] = __fmaf_rn(part_hi_3[15], w_c_10, acc_hi_1[15]);
                            }
                            if (n_chunks_u_0 > 3) {
                                unsigned int crow_7 = (tile_slot0 + 3) * 512 + part_row;
                                float w_c_11 = approx_exp2(part_ml[crow_7 * 2] - row_max_all);
                                unsigned int off_0_7 = crow_7 * 128 + 16;
                                {
                                    unsigned _ldv8_85_0;
                                    unsigned _ldv8_85_1;
                                    unsigned _ldv8_85_2;
                                    unsigned _ldv8_85_3;
                                    unsigned _ldv8_85_4;
                                    unsigned _ldv8_85_5;
                                    unsigned _ldv8_85_6;
                                    unsigned _ldv8_85_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_85_0), "=r"(_ldv8_85_1), "=r"(_ldv8_85_2), "=r"(_ldv8_85_3), "=r"(_ldv8_85_4), "=r"(_ldv8_85_5), "=r"(_ldv8_85_6), "=r"(_ldv8_85_7) : "l"((const void*)(part_o + (off_0_7))) : "memory");
                                    part_2[0 + 0] = __uint_as_float(_ldv8_85_0);
                                    part_2[0 + 1] = __uint_as_float(_ldv8_85_1);
                                    part_2[0 + 2] = __uint_as_float(_ldv8_85_2);
                                    part_2[0 + 3] = __uint_as_float(_ldv8_85_3);
                                    part_2[0 + 4] = __uint_as_float(_ldv8_85_4);
                                    part_2[0 + 5] = __uint_as_float(_ldv8_85_5);
                                    part_2[0 + 6] = __uint_as_float(_ldv8_85_6);
                                    part_2[0 + 7] = __uint_as_float(_ldv8_85_7);
                                }
                                {
                                    unsigned _ldv8_86_0;
                                    unsigned _ldv8_86_1;
                                    unsigned _ldv8_86_2;
                                    unsigned _ldv8_86_3;
                                    unsigned _ldv8_86_4;
                                    unsigned _ldv8_86_5;
                                    unsigned _ldv8_86_6;
                                    unsigned _ldv8_86_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_86_0), "=r"(_ldv8_86_1), "=r"(_ldv8_86_2), "=r"(_ldv8_86_3), "=r"(_ldv8_86_4), "=r"(_ldv8_86_5), "=r"(_ldv8_86_6), "=r"(_ldv8_86_7) : "l"((const void*)(part_o + (off_0_7 + 8))) : "memory");
                                    part_2[8 + 0] = __uint_as_float(_ldv8_86_0);
                                    part_2[8 + 1] = __uint_as_float(_ldv8_86_1);
                                    part_2[8 + 2] = __uint_as_float(_ldv8_86_2);
                                    part_2[8 + 3] = __uint_as_float(_ldv8_86_3);
                                    part_2[8 + 4] = __uint_as_float(_ldv8_86_4);
                                    part_2[8 + 5] = __uint_as_float(_ldv8_86_5);
                                    part_2[8 + 6] = __uint_as_float(_ldv8_86_6);
                                    part_2[8 + 7] = __uint_as_float(_ldv8_86_7);
                                }
                                {
                                    unsigned _ldv8_87_0;
                                    unsigned _ldv8_87_1;
                                    unsigned _ldv8_87_2;
                                    unsigned _ldv8_87_3;
                                    unsigned _ldv8_87_4;
                                    unsigned _ldv8_87_5;
                                    unsigned _ldv8_87_6;
                                    unsigned _ldv8_87_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_87_0), "=r"(_ldv8_87_1), "=r"(_ldv8_87_2), "=r"(_ldv8_87_3), "=r"(_ldv8_87_4), "=r"(_ldv8_87_5), "=r"(_ldv8_87_6), "=r"(_ldv8_87_7) : "l"((const void*)(part_o + (off_0_7 + 64))) : "memory");
                                    part_hi_3[0 + 0] = __uint_as_float(_ldv8_87_0);
                                    part_hi_3[0 + 1] = __uint_as_float(_ldv8_87_1);
                                    part_hi_3[0 + 2] = __uint_as_float(_ldv8_87_2);
                                    part_hi_3[0 + 3] = __uint_as_float(_ldv8_87_3);
                                    part_hi_3[0 + 4] = __uint_as_float(_ldv8_87_4);
                                    part_hi_3[0 + 5] = __uint_as_float(_ldv8_87_5);
                                    part_hi_3[0 + 6] = __uint_as_float(_ldv8_87_6);
                                    part_hi_3[0 + 7] = __uint_as_float(_ldv8_87_7);
                                }
                                {
                                    unsigned _ldv8_88_0;
                                    unsigned _ldv8_88_1;
                                    unsigned _ldv8_88_2;
                                    unsigned _ldv8_88_3;
                                    unsigned _ldv8_88_4;
                                    unsigned _ldv8_88_5;
                                    unsigned _ldv8_88_6;
                                    unsigned _ldv8_88_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_88_0), "=r"(_ldv8_88_1), "=r"(_ldv8_88_2), "=r"(_ldv8_88_3), "=r"(_ldv8_88_4), "=r"(_ldv8_88_5), "=r"(_ldv8_88_6), "=r"(_ldv8_88_7) : "l"((const void*)(part_o + (off_0_7 + 64 + 8))) : "memory");
                                    part_hi_3[8 + 0] = __uint_as_float(_ldv8_88_0);
                                    part_hi_3[8 + 1] = __uint_as_float(_ldv8_88_1);
                                    part_hi_3[8 + 2] = __uint_as_float(_ldv8_88_2);
                                    part_hi_3[8 + 3] = __uint_as_float(_ldv8_88_3);
                                    part_hi_3[8 + 4] = __uint_as_float(_ldv8_88_4);
                                    part_hi_3[8 + 5] = __uint_as_float(_ldv8_88_5);
                                    part_hi_3[8 + 6] = __uint_as_float(_ldv8_88_6);
                                    part_hi_3[8 + 7] = __uint_as_float(_ldv8_88_7);
                                }
                                acc_0[0] = __fmaf_rn(part_2[0], w_c_11, acc_0[0]);
                                acc_hi_1[0] = __fmaf_rn(part_hi_3[0], w_c_11, acc_hi_1[0]);
                                acc_0[1] = __fmaf_rn(part_2[1], w_c_11, acc_0[1]);
                                acc_hi_1[1] = __fmaf_rn(part_hi_3[1], w_c_11, acc_hi_1[1]);
                                acc_0[2] = __fmaf_rn(part_2[2], w_c_11, acc_0[2]);
                                acc_hi_1[2] = __fmaf_rn(part_hi_3[2], w_c_11, acc_hi_1[2]);
                                acc_0[3] = __fmaf_rn(part_2[3], w_c_11, acc_0[3]);
                                acc_hi_1[3] = __fmaf_rn(part_hi_3[3], w_c_11, acc_hi_1[3]);
                                acc_0[4] = __fmaf_rn(part_2[4], w_c_11, acc_0[4]);
                                acc_hi_1[4] = __fmaf_rn(part_hi_3[4], w_c_11, acc_hi_1[4]);
                                acc_0[5] = __fmaf_rn(part_2[5], w_c_11, acc_0[5]);
                                acc_hi_1[5] = __fmaf_rn(part_hi_3[5], w_c_11, acc_hi_1[5]);
                                acc_0[6] = __fmaf_rn(part_2[6], w_c_11, acc_0[6]);
                                acc_hi_1[6] = __fmaf_rn(part_hi_3[6], w_c_11, acc_hi_1[6]);
                                acc_0[7] = __fmaf_rn(part_2[7], w_c_11, acc_0[7]);
                                acc_hi_1[7] = __fmaf_rn(part_hi_3[7], w_c_11, acc_hi_1[7]);
                                acc_0[8] = __fmaf_rn(part_2[8], w_c_11, acc_0[8]);
                                acc_hi_1[8] = __fmaf_rn(part_hi_3[8], w_c_11, acc_hi_1[8]);
                                acc_0[9] = __fmaf_rn(part_2[9], w_c_11, acc_0[9]);
                                acc_hi_1[9] = __fmaf_rn(part_hi_3[9], w_c_11, acc_hi_1[9]);
                                acc_0[10] = __fmaf_rn(part_2[10], w_c_11, acc_0[10]);
                                acc_hi_1[10] = __fmaf_rn(part_hi_3[10], w_c_11, acc_hi_1[10]);
                                acc_0[11] = __fmaf_rn(part_2[11], w_c_11, acc_0[11]);
                                acc_hi_1[11] = __fmaf_rn(part_hi_3[11], w_c_11, acc_hi_1[11]);
                                acc_0[12] = __fmaf_rn(part_2[12], w_c_11, acc_0[12]);
                                acc_hi_1[12] = __fmaf_rn(part_hi_3[12], w_c_11, acc_hi_1[12]);
                                acc_0[13] = __fmaf_rn(part_2[13], w_c_11, acc_0[13]);
                                acc_hi_1[13] = __fmaf_rn(part_hi_3[13], w_c_11, acc_hi_1[13]);
                                acc_0[14] = __fmaf_rn(part_2[14], w_c_11, acc_0[14]);
                                acc_hi_1[14] = __fmaf_rn(part_hi_3[14], w_c_11, acc_hi_1[14]);
                                acc_0[15] = __fmaf_rn(part_2[15], w_c_11, acc_0[15]);
                                acc_hi_1[15] = __fmaf_rn(part_hi_3[15], w_c_11, acc_hi_1[15]);
                            }
                            const float2 _scale2_89 = {final_scale, final_scale};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(acc_0)[_ls], _scale2_89);
                            unsigned int acc_0_bf16[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_0[_lp*2 + 0], acc_0[_lp*2+1 + 0]));
                                acc_0_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o00_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + 32 ^ ((warp % 4 * 32 + lane) * 128 + 32 >> 7 & 7) << 4))), "r"(acc_0_bf16[0]), "r"(acc_0_bf16[1]), "r"(acc_0_bf16[2]), "r"(acc_0_bf16[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o00_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + 48 ^ ((warp % 4 * 32 + lane) * 128 + 48 >> 7 & 7) << 4))), "r"(acc_0_bf16[4]), "r"(acc_0_bf16[5]), "r"(acc_0_bf16[6]), "r"(acc_0_bf16[7]) : "memory");
                            const float2 _scale2_90 = {final_scale, final_scale};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(acc_hi_1)[_ls], _scale2_90);
                            unsigned int acc_hi_1_bf16[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_hi_1[_lp*2 + 0], acc_hi_1[_lp*2+1 + 0]));
                                acc_hi_1_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o01_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + 32 ^ ((warp % 4 * 32 + lane) * 128 + 32 >> 7 & 7) << 4))), "r"(acc_hi_1_bf16[0]), "r"(acc_hi_1_bf16[1]), "r"(acc_hi_1_bf16[2]), "r"(acc_hi_1_bf16[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o01_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + 48 ^ ((warp % 4 * 32 + lane) * 128 + 48 >> 7 & 7) << 4))), "r"(acc_hi_1_bf16[4]), "r"(acc_hi_1_bf16[5]), "r"(acc_hi_1_bf16[6]), "r"(acc_hi_1_bf16[7]) : "memory");
                            float acc_4[16];
                            float acc_hi_5[16];
                            acc_4[0] = 0.0f;
                            acc_hi_5[0] = 0.0f;
                            acc_4[1] = 0.0f;
                            acc_hi_5[1] = 0.0f;
                            acc_4[2] = 0.0f;
                            acc_hi_5[2] = 0.0f;
                            acc_4[3] = 0.0f;
                            acc_hi_5[3] = 0.0f;
                            acc_4[4] = 0.0f;
                            acc_hi_5[4] = 0.0f;
                            acc_4[5] = 0.0f;
                            acc_hi_5[5] = 0.0f;
                            acc_4[6] = 0.0f;
                            acc_hi_5[6] = 0.0f;
                            acc_4[7] = 0.0f;
                            acc_hi_5[7] = 0.0f;
                            acc_4[8] = 0.0f;
                            acc_hi_5[8] = 0.0f;
                            acc_4[9] = 0.0f;
                            acc_hi_5[9] = 0.0f;
                            acc_4[10] = 0.0f;
                            acc_hi_5[10] = 0.0f;
                            acc_4[11] = 0.0f;
                            acc_hi_5[11] = 0.0f;
                            acc_4[12] = 0.0f;
                            acc_hi_5[12] = 0.0f;
                            acc_4[13] = 0.0f;
                            acc_hi_5[13] = 0.0f;
                            acc_4[14] = 0.0f;
                            acc_hi_5[14] = 0.0f;
                            acc_4[15] = 0.0f;
                            acc_hi_5[15] = 0.0f;
                            float part_6[16];
                            float part_hi_7[16];
                            if (n_chunks_u_0 > 0) {
                                unsigned int crow_8 = tile_slot0 * 512 + part_row;
                                float w_c_12 = approx_exp2(part_ml[crow_8 * 2] - row_max_all);
                                unsigned int off_0_8 = crow_8 * 128 + 32;
                                {
                                    unsigned _ldv8_91_0;
                                    unsigned _ldv8_91_1;
                                    unsigned _ldv8_91_2;
                                    unsigned _ldv8_91_3;
                                    unsigned _ldv8_91_4;
                                    unsigned _ldv8_91_5;
                                    unsigned _ldv8_91_6;
                                    unsigned _ldv8_91_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_91_0), "=r"(_ldv8_91_1), "=r"(_ldv8_91_2), "=r"(_ldv8_91_3), "=r"(_ldv8_91_4), "=r"(_ldv8_91_5), "=r"(_ldv8_91_6), "=r"(_ldv8_91_7) : "l"((const void*)(part_o + (off_0_8))) : "memory");
                                    part_6[0 + 0] = __uint_as_float(_ldv8_91_0);
                                    part_6[0 + 1] = __uint_as_float(_ldv8_91_1);
                                    part_6[0 + 2] = __uint_as_float(_ldv8_91_2);
                                    part_6[0 + 3] = __uint_as_float(_ldv8_91_3);
                                    part_6[0 + 4] = __uint_as_float(_ldv8_91_4);
                                    part_6[0 + 5] = __uint_as_float(_ldv8_91_5);
                                    part_6[0 + 6] = __uint_as_float(_ldv8_91_6);
                                    part_6[0 + 7] = __uint_as_float(_ldv8_91_7);
                                }
                                {
                                    unsigned _ldv8_92_0;
                                    unsigned _ldv8_92_1;
                                    unsigned _ldv8_92_2;
                                    unsigned _ldv8_92_3;
                                    unsigned _ldv8_92_4;
                                    unsigned _ldv8_92_5;
                                    unsigned _ldv8_92_6;
                                    unsigned _ldv8_92_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_92_0), "=r"(_ldv8_92_1), "=r"(_ldv8_92_2), "=r"(_ldv8_92_3), "=r"(_ldv8_92_4), "=r"(_ldv8_92_5), "=r"(_ldv8_92_6), "=r"(_ldv8_92_7) : "l"((const void*)(part_o + (off_0_8 + 8))) : "memory");
                                    part_6[8 + 0] = __uint_as_float(_ldv8_92_0);
                                    part_6[8 + 1] = __uint_as_float(_ldv8_92_1);
                                    part_6[8 + 2] = __uint_as_float(_ldv8_92_2);
                                    part_6[8 + 3] = __uint_as_float(_ldv8_92_3);
                                    part_6[8 + 4] = __uint_as_float(_ldv8_92_4);
                                    part_6[8 + 5] = __uint_as_float(_ldv8_92_5);
                                    part_6[8 + 6] = __uint_as_float(_ldv8_92_6);
                                    part_6[8 + 7] = __uint_as_float(_ldv8_92_7);
                                }
                                {
                                    unsigned _ldv8_93_0;
                                    unsigned _ldv8_93_1;
                                    unsigned _ldv8_93_2;
                                    unsigned _ldv8_93_3;
                                    unsigned _ldv8_93_4;
                                    unsigned _ldv8_93_5;
                                    unsigned _ldv8_93_6;
                                    unsigned _ldv8_93_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_93_0), "=r"(_ldv8_93_1), "=r"(_ldv8_93_2), "=r"(_ldv8_93_3), "=r"(_ldv8_93_4), "=r"(_ldv8_93_5), "=r"(_ldv8_93_6), "=r"(_ldv8_93_7) : "l"((const void*)(part_o + (off_0_8 + 64))) : "memory");
                                    part_hi_7[0 + 0] = __uint_as_float(_ldv8_93_0);
                                    part_hi_7[0 + 1] = __uint_as_float(_ldv8_93_1);
                                    part_hi_7[0 + 2] = __uint_as_float(_ldv8_93_2);
                                    part_hi_7[0 + 3] = __uint_as_float(_ldv8_93_3);
                                    part_hi_7[0 + 4] = __uint_as_float(_ldv8_93_4);
                                    part_hi_7[0 + 5] = __uint_as_float(_ldv8_93_5);
                                    part_hi_7[0 + 6] = __uint_as_float(_ldv8_93_6);
                                    part_hi_7[0 + 7] = __uint_as_float(_ldv8_93_7);
                                }
                                {
                                    unsigned _ldv8_94_0;
                                    unsigned _ldv8_94_1;
                                    unsigned _ldv8_94_2;
                                    unsigned _ldv8_94_3;
                                    unsigned _ldv8_94_4;
                                    unsigned _ldv8_94_5;
                                    unsigned _ldv8_94_6;
                                    unsigned _ldv8_94_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_94_0), "=r"(_ldv8_94_1), "=r"(_ldv8_94_2), "=r"(_ldv8_94_3), "=r"(_ldv8_94_4), "=r"(_ldv8_94_5), "=r"(_ldv8_94_6), "=r"(_ldv8_94_7) : "l"((const void*)(part_o + (off_0_8 + 64 + 8))) : "memory");
                                    part_hi_7[8 + 0] = __uint_as_float(_ldv8_94_0);
                                    part_hi_7[8 + 1] = __uint_as_float(_ldv8_94_1);
                                    part_hi_7[8 + 2] = __uint_as_float(_ldv8_94_2);
                                    part_hi_7[8 + 3] = __uint_as_float(_ldv8_94_3);
                                    part_hi_7[8 + 4] = __uint_as_float(_ldv8_94_4);
                                    part_hi_7[8 + 5] = __uint_as_float(_ldv8_94_5);
                                    part_hi_7[8 + 6] = __uint_as_float(_ldv8_94_6);
                                    part_hi_7[8 + 7] = __uint_as_float(_ldv8_94_7);
                                }
                                acc_4[0] = __fmaf_rn(part_6[0], w_c_12, acc_4[0]);
                                acc_hi_5[0] = __fmaf_rn(part_hi_7[0], w_c_12, acc_hi_5[0]);
                                acc_4[1] = __fmaf_rn(part_6[1], w_c_12, acc_4[1]);
                                acc_hi_5[1] = __fmaf_rn(part_hi_7[1], w_c_12, acc_hi_5[1]);
                                acc_4[2] = __fmaf_rn(part_6[2], w_c_12, acc_4[2]);
                                acc_hi_5[2] = __fmaf_rn(part_hi_7[2], w_c_12, acc_hi_5[2]);
                                acc_4[3] = __fmaf_rn(part_6[3], w_c_12, acc_4[3]);
                                acc_hi_5[3] = __fmaf_rn(part_hi_7[3], w_c_12, acc_hi_5[3]);
                                acc_4[4] = __fmaf_rn(part_6[4], w_c_12, acc_4[4]);
                                acc_hi_5[4] = __fmaf_rn(part_hi_7[4], w_c_12, acc_hi_5[4]);
                                acc_4[5] = __fmaf_rn(part_6[5], w_c_12, acc_4[5]);
                                acc_hi_5[5] = __fmaf_rn(part_hi_7[5], w_c_12, acc_hi_5[5]);
                                acc_4[6] = __fmaf_rn(part_6[6], w_c_12, acc_4[6]);
                                acc_hi_5[6] = __fmaf_rn(part_hi_7[6], w_c_12, acc_hi_5[6]);
                                acc_4[7] = __fmaf_rn(part_6[7], w_c_12, acc_4[7]);
                                acc_hi_5[7] = __fmaf_rn(part_hi_7[7], w_c_12, acc_hi_5[7]);
                                acc_4[8] = __fmaf_rn(part_6[8], w_c_12, acc_4[8]);
                                acc_hi_5[8] = __fmaf_rn(part_hi_7[8], w_c_12, acc_hi_5[8]);
                                acc_4[9] = __fmaf_rn(part_6[9], w_c_12, acc_4[9]);
                                acc_hi_5[9] = __fmaf_rn(part_hi_7[9], w_c_12, acc_hi_5[9]);
                                acc_4[10] = __fmaf_rn(part_6[10], w_c_12, acc_4[10]);
                                acc_hi_5[10] = __fmaf_rn(part_hi_7[10], w_c_12, acc_hi_5[10]);
                                acc_4[11] = __fmaf_rn(part_6[11], w_c_12, acc_4[11]);
                                acc_hi_5[11] = __fmaf_rn(part_hi_7[11], w_c_12, acc_hi_5[11]);
                                acc_4[12] = __fmaf_rn(part_6[12], w_c_12, acc_4[12]);
                                acc_hi_5[12] = __fmaf_rn(part_hi_7[12], w_c_12, acc_hi_5[12]);
                                acc_4[13] = __fmaf_rn(part_6[13], w_c_12, acc_4[13]);
                                acc_hi_5[13] = __fmaf_rn(part_hi_7[13], w_c_12, acc_hi_5[13]);
                                acc_4[14] = __fmaf_rn(part_6[14], w_c_12, acc_4[14]);
                                acc_hi_5[14] = __fmaf_rn(part_hi_7[14], w_c_12, acc_hi_5[14]);
                                acc_4[15] = __fmaf_rn(part_6[15], w_c_12, acc_4[15]);
                                acc_hi_5[15] = __fmaf_rn(part_hi_7[15], w_c_12, acc_hi_5[15]);
                            }
                            if (n_chunks_u_0 > 1) {
                                unsigned int crow_9 = (tile_slot0 + 1) * 512 + part_row;
                                float w_c_13 = approx_exp2(part_ml[crow_9 * 2] - row_max_all);
                                unsigned int off_0_9 = crow_9 * 128 + 32;
                                {
                                    unsigned _ldv8_95_0;
                                    unsigned _ldv8_95_1;
                                    unsigned _ldv8_95_2;
                                    unsigned _ldv8_95_3;
                                    unsigned _ldv8_95_4;
                                    unsigned _ldv8_95_5;
                                    unsigned _ldv8_95_6;
                                    unsigned _ldv8_95_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_95_0), "=r"(_ldv8_95_1), "=r"(_ldv8_95_2), "=r"(_ldv8_95_3), "=r"(_ldv8_95_4), "=r"(_ldv8_95_5), "=r"(_ldv8_95_6), "=r"(_ldv8_95_7) : "l"((const void*)(part_o + (off_0_9))) : "memory");
                                    part_6[0 + 0] = __uint_as_float(_ldv8_95_0);
                                    part_6[0 + 1] = __uint_as_float(_ldv8_95_1);
                                    part_6[0 + 2] = __uint_as_float(_ldv8_95_2);
                                    part_6[0 + 3] = __uint_as_float(_ldv8_95_3);
                                    part_6[0 + 4] = __uint_as_float(_ldv8_95_4);
                                    part_6[0 + 5] = __uint_as_float(_ldv8_95_5);
                                    part_6[0 + 6] = __uint_as_float(_ldv8_95_6);
                                    part_6[0 + 7] = __uint_as_float(_ldv8_95_7);
                                }
                                {
                                    unsigned _ldv8_96_0;
                                    unsigned _ldv8_96_1;
                                    unsigned _ldv8_96_2;
                                    unsigned _ldv8_96_3;
                                    unsigned _ldv8_96_4;
                                    unsigned _ldv8_96_5;
                                    unsigned _ldv8_96_6;
                                    unsigned _ldv8_96_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_96_0), "=r"(_ldv8_96_1), "=r"(_ldv8_96_2), "=r"(_ldv8_96_3), "=r"(_ldv8_96_4), "=r"(_ldv8_96_5), "=r"(_ldv8_96_6), "=r"(_ldv8_96_7) : "l"((const void*)(part_o + (off_0_9 + 8))) : "memory");
                                    part_6[8 + 0] = __uint_as_float(_ldv8_96_0);
                                    part_6[8 + 1] = __uint_as_float(_ldv8_96_1);
                                    part_6[8 + 2] = __uint_as_float(_ldv8_96_2);
                                    part_6[8 + 3] = __uint_as_float(_ldv8_96_3);
                                    part_6[8 + 4] = __uint_as_float(_ldv8_96_4);
                                    part_6[8 + 5] = __uint_as_float(_ldv8_96_5);
                                    part_6[8 + 6] = __uint_as_float(_ldv8_96_6);
                                    part_6[8 + 7] = __uint_as_float(_ldv8_96_7);
                                }
                                {
                                    unsigned _ldv8_97_0;
                                    unsigned _ldv8_97_1;
                                    unsigned _ldv8_97_2;
                                    unsigned _ldv8_97_3;
                                    unsigned _ldv8_97_4;
                                    unsigned _ldv8_97_5;
                                    unsigned _ldv8_97_6;
                                    unsigned _ldv8_97_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_97_0), "=r"(_ldv8_97_1), "=r"(_ldv8_97_2), "=r"(_ldv8_97_3), "=r"(_ldv8_97_4), "=r"(_ldv8_97_5), "=r"(_ldv8_97_6), "=r"(_ldv8_97_7) : "l"((const void*)(part_o + (off_0_9 + 64))) : "memory");
                                    part_hi_7[0 + 0] = __uint_as_float(_ldv8_97_0);
                                    part_hi_7[0 + 1] = __uint_as_float(_ldv8_97_1);
                                    part_hi_7[0 + 2] = __uint_as_float(_ldv8_97_2);
                                    part_hi_7[0 + 3] = __uint_as_float(_ldv8_97_3);
                                    part_hi_7[0 + 4] = __uint_as_float(_ldv8_97_4);
                                    part_hi_7[0 + 5] = __uint_as_float(_ldv8_97_5);
                                    part_hi_7[0 + 6] = __uint_as_float(_ldv8_97_6);
                                    part_hi_7[0 + 7] = __uint_as_float(_ldv8_97_7);
                                }
                                {
                                    unsigned _ldv8_98_0;
                                    unsigned _ldv8_98_1;
                                    unsigned _ldv8_98_2;
                                    unsigned _ldv8_98_3;
                                    unsigned _ldv8_98_4;
                                    unsigned _ldv8_98_5;
                                    unsigned _ldv8_98_6;
                                    unsigned _ldv8_98_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_98_0), "=r"(_ldv8_98_1), "=r"(_ldv8_98_2), "=r"(_ldv8_98_3), "=r"(_ldv8_98_4), "=r"(_ldv8_98_5), "=r"(_ldv8_98_6), "=r"(_ldv8_98_7) : "l"((const void*)(part_o + (off_0_9 + 64 + 8))) : "memory");
                                    part_hi_7[8 + 0] = __uint_as_float(_ldv8_98_0);
                                    part_hi_7[8 + 1] = __uint_as_float(_ldv8_98_1);
                                    part_hi_7[8 + 2] = __uint_as_float(_ldv8_98_2);
                                    part_hi_7[8 + 3] = __uint_as_float(_ldv8_98_3);
                                    part_hi_7[8 + 4] = __uint_as_float(_ldv8_98_4);
                                    part_hi_7[8 + 5] = __uint_as_float(_ldv8_98_5);
                                    part_hi_7[8 + 6] = __uint_as_float(_ldv8_98_6);
                                    part_hi_7[8 + 7] = __uint_as_float(_ldv8_98_7);
                                }
                                acc_4[0] = __fmaf_rn(part_6[0], w_c_13, acc_4[0]);
                                acc_hi_5[0] = __fmaf_rn(part_hi_7[0], w_c_13, acc_hi_5[0]);
                                acc_4[1] = __fmaf_rn(part_6[1], w_c_13, acc_4[1]);
                                acc_hi_5[1] = __fmaf_rn(part_hi_7[1], w_c_13, acc_hi_5[1]);
                                acc_4[2] = __fmaf_rn(part_6[2], w_c_13, acc_4[2]);
                                acc_hi_5[2] = __fmaf_rn(part_hi_7[2], w_c_13, acc_hi_5[2]);
                                acc_4[3] = __fmaf_rn(part_6[3], w_c_13, acc_4[3]);
                                acc_hi_5[3] = __fmaf_rn(part_hi_7[3], w_c_13, acc_hi_5[3]);
                                acc_4[4] = __fmaf_rn(part_6[4], w_c_13, acc_4[4]);
                                acc_hi_5[4] = __fmaf_rn(part_hi_7[4], w_c_13, acc_hi_5[4]);
                                acc_4[5] = __fmaf_rn(part_6[5], w_c_13, acc_4[5]);
                                acc_hi_5[5] = __fmaf_rn(part_hi_7[5], w_c_13, acc_hi_5[5]);
                                acc_4[6] = __fmaf_rn(part_6[6], w_c_13, acc_4[6]);
                                acc_hi_5[6] = __fmaf_rn(part_hi_7[6], w_c_13, acc_hi_5[6]);
                                acc_4[7] = __fmaf_rn(part_6[7], w_c_13, acc_4[7]);
                                acc_hi_5[7] = __fmaf_rn(part_hi_7[7], w_c_13, acc_hi_5[7]);
                                acc_4[8] = __fmaf_rn(part_6[8], w_c_13, acc_4[8]);
                                acc_hi_5[8] = __fmaf_rn(part_hi_7[8], w_c_13, acc_hi_5[8]);
                                acc_4[9] = __fmaf_rn(part_6[9], w_c_13, acc_4[9]);
                                acc_hi_5[9] = __fmaf_rn(part_hi_7[9], w_c_13, acc_hi_5[9]);
                                acc_4[10] = __fmaf_rn(part_6[10], w_c_13, acc_4[10]);
                                acc_hi_5[10] = __fmaf_rn(part_hi_7[10], w_c_13, acc_hi_5[10]);
                                acc_4[11] = __fmaf_rn(part_6[11], w_c_13, acc_4[11]);
                                acc_hi_5[11] = __fmaf_rn(part_hi_7[11], w_c_13, acc_hi_5[11]);
                                acc_4[12] = __fmaf_rn(part_6[12], w_c_13, acc_4[12]);
                                acc_hi_5[12] = __fmaf_rn(part_hi_7[12], w_c_13, acc_hi_5[12]);
                                acc_4[13] = __fmaf_rn(part_6[13], w_c_13, acc_4[13]);
                                acc_hi_5[13] = __fmaf_rn(part_hi_7[13], w_c_13, acc_hi_5[13]);
                                acc_4[14] = __fmaf_rn(part_6[14], w_c_13, acc_4[14]);
                                acc_hi_5[14] = __fmaf_rn(part_hi_7[14], w_c_13, acc_hi_5[14]);
                                acc_4[15] = __fmaf_rn(part_6[15], w_c_13, acc_4[15]);
                                acc_hi_5[15] = __fmaf_rn(part_hi_7[15], w_c_13, acc_hi_5[15]);
                            }
                            if (n_chunks_u_0 > 2) {
                                unsigned int crow_10 = (tile_slot0 + 2) * 512 + part_row;
                                float w_c_14 = approx_exp2(part_ml[crow_10 * 2] - row_max_all);
                                unsigned int off_0_10 = crow_10 * 128 + 32;
                                {
                                    unsigned _ldv8_99_0;
                                    unsigned _ldv8_99_1;
                                    unsigned _ldv8_99_2;
                                    unsigned _ldv8_99_3;
                                    unsigned _ldv8_99_4;
                                    unsigned _ldv8_99_5;
                                    unsigned _ldv8_99_6;
                                    unsigned _ldv8_99_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_99_0), "=r"(_ldv8_99_1), "=r"(_ldv8_99_2), "=r"(_ldv8_99_3), "=r"(_ldv8_99_4), "=r"(_ldv8_99_5), "=r"(_ldv8_99_6), "=r"(_ldv8_99_7) : "l"((const void*)(part_o + (off_0_10))) : "memory");
                                    part_6[0 + 0] = __uint_as_float(_ldv8_99_0);
                                    part_6[0 + 1] = __uint_as_float(_ldv8_99_1);
                                    part_6[0 + 2] = __uint_as_float(_ldv8_99_2);
                                    part_6[0 + 3] = __uint_as_float(_ldv8_99_3);
                                    part_6[0 + 4] = __uint_as_float(_ldv8_99_4);
                                    part_6[0 + 5] = __uint_as_float(_ldv8_99_5);
                                    part_6[0 + 6] = __uint_as_float(_ldv8_99_6);
                                    part_6[0 + 7] = __uint_as_float(_ldv8_99_7);
                                }
                                {
                                    unsigned _ldv8_100_0;
                                    unsigned _ldv8_100_1;
                                    unsigned _ldv8_100_2;
                                    unsigned _ldv8_100_3;
                                    unsigned _ldv8_100_4;
                                    unsigned _ldv8_100_5;
                                    unsigned _ldv8_100_6;
                                    unsigned _ldv8_100_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_100_0), "=r"(_ldv8_100_1), "=r"(_ldv8_100_2), "=r"(_ldv8_100_3), "=r"(_ldv8_100_4), "=r"(_ldv8_100_5), "=r"(_ldv8_100_6), "=r"(_ldv8_100_7) : "l"((const void*)(part_o + (off_0_10 + 8))) : "memory");
                                    part_6[8 + 0] = __uint_as_float(_ldv8_100_0);
                                    part_6[8 + 1] = __uint_as_float(_ldv8_100_1);
                                    part_6[8 + 2] = __uint_as_float(_ldv8_100_2);
                                    part_6[8 + 3] = __uint_as_float(_ldv8_100_3);
                                    part_6[8 + 4] = __uint_as_float(_ldv8_100_4);
                                    part_6[8 + 5] = __uint_as_float(_ldv8_100_5);
                                    part_6[8 + 6] = __uint_as_float(_ldv8_100_6);
                                    part_6[8 + 7] = __uint_as_float(_ldv8_100_7);
                                }
                                {
                                    unsigned _ldv8_101_0;
                                    unsigned _ldv8_101_1;
                                    unsigned _ldv8_101_2;
                                    unsigned _ldv8_101_3;
                                    unsigned _ldv8_101_4;
                                    unsigned _ldv8_101_5;
                                    unsigned _ldv8_101_6;
                                    unsigned _ldv8_101_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_101_0), "=r"(_ldv8_101_1), "=r"(_ldv8_101_2), "=r"(_ldv8_101_3), "=r"(_ldv8_101_4), "=r"(_ldv8_101_5), "=r"(_ldv8_101_6), "=r"(_ldv8_101_7) : "l"((const void*)(part_o + (off_0_10 + 64))) : "memory");
                                    part_hi_7[0 + 0] = __uint_as_float(_ldv8_101_0);
                                    part_hi_7[0 + 1] = __uint_as_float(_ldv8_101_1);
                                    part_hi_7[0 + 2] = __uint_as_float(_ldv8_101_2);
                                    part_hi_7[0 + 3] = __uint_as_float(_ldv8_101_3);
                                    part_hi_7[0 + 4] = __uint_as_float(_ldv8_101_4);
                                    part_hi_7[0 + 5] = __uint_as_float(_ldv8_101_5);
                                    part_hi_7[0 + 6] = __uint_as_float(_ldv8_101_6);
                                    part_hi_7[0 + 7] = __uint_as_float(_ldv8_101_7);
                                }
                                {
                                    unsigned _ldv8_102_0;
                                    unsigned _ldv8_102_1;
                                    unsigned _ldv8_102_2;
                                    unsigned _ldv8_102_3;
                                    unsigned _ldv8_102_4;
                                    unsigned _ldv8_102_5;
                                    unsigned _ldv8_102_6;
                                    unsigned _ldv8_102_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_102_0), "=r"(_ldv8_102_1), "=r"(_ldv8_102_2), "=r"(_ldv8_102_3), "=r"(_ldv8_102_4), "=r"(_ldv8_102_5), "=r"(_ldv8_102_6), "=r"(_ldv8_102_7) : "l"((const void*)(part_o + (off_0_10 + 64 + 8))) : "memory");
                                    part_hi_7[8 + 0] = __uint_as_float(_ldv8_102_0);
                                    part_hi_7[8 + 1] = __uint_as_float(_ldv8_102_1);
                                    part_hi_7[8 + 2] = __uint_as_float(_ldv8_102_2);
                                    part_hi_7[8 + 3] = __uint_as_float(_ldv8_102_3);
                                    part_hi_7[8 + 4] = __uint_as_float(_ldv8_102_4);
                                    part_hi_7[8 + 5] = __uint_as_float(_ldv8_102_5);
                                    part_hi_7[8 + 6] = __uint_as_float(_ldv8_102_6);
                                    part_hi_7[8 + 7] = __uint_as_float(_ldv8_102_7);
                                }
                                acc_4[0] = __fmaf_rn(part_6[0], w_c_14, acc_4[0]);
                                acc_hi_5[0] = __fmaf_rn(part_hi_7[0], w_c_14, acc_hi_5[0]);
                                acc_4[1] = __fmaf_rn(part_6[1], w_c_14, acc_4[1]);
                                acc_hi_5[1] = __fmaf_rn(part_hi_7[1], w_c_14, acc_hi_5[1]);
                                acc_4[2] = __fmaf_rn(part_6[2], w_c_14, acc_4[2]);
                                acc_hi_5[2] = __fmaf_rn(part_hi_7[2], w_c_14, acc_hi_5[2]);
                                acc_4[3] = __fmaf_rn(part_6[3], w_c_14, acc_4[3]);
                                acc_hi_5[3] = __fmaf_rn(part_hi_7[3], w_c_14, acc_hi_5[3]);
                                acc_4[4] = __fmaf_rn(part_6[4], w_c_14, acc_4[4]);
                                acc_hi_5[4] = __fmaf_rn(part_hi_7[4], w_c_14, acc_hi_5[4]);
                                acc_4[5] = __fmaf_rn(part_6[5], w_c_14, acc_4[5]);
                                acc_hi_5[5] = __fmaf_rn(part_hi_7[5], w_c_14, acc_hi_5[5]);
                                acc_4[6] = __fmaf_rn(part_6[6], w_c_14, acc_4[6]);
                                acc_hi_5[6] = __fmaf_rn(part_hi_7[6], w_c_14, acc_hi_5[6]);
                                acc_4[7] = __fmaf_rn(part_6[7], w_c_14, acc_4[7]);
                                acc_hi_5[7] = __fmaf_rn(part_hi_7[7], w_c_14, acc_hi_5[7]);
                                acc_4[8] = __fmaf_rn(part_6[8], w_c_14, acc_4[8]);
                                acc_hi_5[8] = __fmaf_rn(part_hi_7[8], w_c_14, acc_hi_5[8]);
                                acc_4[9] = __fmaf_rn(part_6[9], w_c_14, acc_4[9]);
                                acc_hi_5[9] = __fmaf_rn(part_hi_7[9], w_c_14, acc_hi_5[9]);
                                acc_4[10] = __fmaf_rn(part_6[10], w_c_14, acc_4[10]);
                                acc_hi_5[10] = __fmaf_rn(part_hi_7[10], w_c_14, acc_hi_5[10]);
                                acc_4[11] = __fmaf_rn(part_6[11], w_c_14, acc_4[11]);
                                acc_hi_5[11] = __fmaf_rn(part_hi_7[11], w_c_14, acc_hi_5[11]);
                                acc_4[12] = __fmaf_rn(part_6[12], w_c_14, acc_4[12]);
                                acc_hi_5[12] = __fmaf_rn(part_hi_7[12], w_c_14, acc_hi_5[12]);
                                acc_4[13] = __fmaf_rn(part_6[13], w_c_14, acc_4[13]);
                                acc_hi_5[13] = __fmaf_rn(part_hi_7[13], w_c_14, acc_hi_5[13]);
                                acc_4[14] = __fmaf_rn(part_6[14], w_c_14, acc_4[14]);
                                acc_hi_5[14] = __fmaf_rn(part_hi_7[14], w_c_14, acc_hi_5[14]);
                                acc_4[15] = __fmaf_rn(part_6[15], w_c_14, acc_4[15]);
                                acc_hi_5[15] = __fmaf_rn(part_hi_7[15], w_c_14, acc_hi_5[15]);
                            }
                            if (n_chunks_u_0 > 3) {
                                unsigned int crow_11 = (tile_slot0 + 3) * 512 + part_row;
                                float w_c_15 = approx_exp2(part_ml[crow_11 * 2] - row_max_all);
                                unsigned int off_0_11 = crow_11 * 128 + 32;
                                {
                                    unsigned _ldv8_103_0;
                                    unsigned _ldv8_103_1;
                                    unsigned _ldv8_103_2;
                                    unsigned _ldv8_103_3;
                                    unsigned _ldv8_103_4;
                                    unsigned _ldv8_103_5;
                                    unsigned _ldv8_103_6;
                                    unsigned _ldv8_103_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_103_0), "=r"(_ldv8_103_1), "=r"(_ldv8_103_2), "=r"(_ldv8_103_3), "=r"(_ldv8_103_4), "=r"(_ldv8_103_5), "=r"(_ldv8_103_6), "=r"(_ldv8_103_7) : "l"((const void*)(part_o + (off_0_11))) : "memory");
                                    part_6[0 + 0] = __uint_as_float(_ldv8_103_0);
                                    part_6[0 + 1] = __uint_as_float(_ldv8_103_1);
                                    part_6[0 + 2] = __uint_as_float(_ldv8_103_2);
                                    part_6[0 + 3] = __uint_as_float(_ldv8_103_3);
                                    part_6[0 + 4] = __uint_as_float(_ldv8_103_4);
                                    part_6[0 + 5] = __uint_as_float(_ldv8_103_5);
                                    part_6[0 + 6] = __uint_as_float(_ldv8_103_6);
                                    part_6[0 + 7] = __uint_as_float(_ldv8_103_7);
                                }
                                {
                                    unsigned _ldv8_104_0;
                                    unsigned _ldv8_104_1;
                                    unsigned _ldv8_104_2;
                                    unsigned _ldv8_104_3;
                                    unsigned _ldv8_104_4;
                                    unsigned _ldv8_104_5;
                                    unsigned _ldv8_104_6;
                                    unsigned _ldv8_104_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_104_0), "=r"(_ldv8_104_1), "=r"(_ldv8_104_2), "=r"(_ldv8_104_3), "=r"(_ldv8_104_4), "=r"(_ldv8_104_5), "=r"(_ldv8_104_6), "=r"(_ldv8_104_7) : "l"((const void*)(part_o + (off_0_11 + 8))) : "memory");
                                    part_6[8 + 0] = __uint_as_float(_ldv8_104_0);
                                    part_6[8 + 1] = __uint_as_float(_ldv8_104_1);
                                    part_6[8 + 2] = __uint_as_float(_ldv8_104_2);
                                    part_6[8 + 3] = __uint_as_float(_ldv8_104_3);
                                    part_6[8 + 4] = __uint_as_float(_ldv8_104_4);
                                    part_6[8 + 5] = __uint_as_float(_ldv8_104_5);
                                    part_6[8 + 6] = __uint_as_float(_ldv8_104_6);
                                    part_6[8 + 7] = __uint_as_float(_ldv8_104_7);
                                }
                                {
                                    unsigned _ldv8_105_0;
                                    unsigned _ldv8_105_1;
                                    unsigned _ldv8_105_2;
                                    unsigned _ldv8_105_3;
                                    unsigned _ldv8_105_4;
                                    unsigned _ldv8_105_5;
                                    unsigned _ldv8_105_6;
                                    unsigned _ldv8_105_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_105_0), "=r"(_ldv8_105_1), "=r"(_ldv8_105_2), "=r"(_ldv8_105_3), "=r"(_ldv8_105_4), "=r"(_ldv8_105_5), "=r"(_ldv8_105_6), "=r"(_ldv8_105_7) : "l"((const void*)(part_o + (off_0_11 + 64))) : "memory");
                                    part_hi_7[0 + 0] = __uint_as_float(_ldv8_105_0);
                                    part_hi_7[0 + 1] = __uint_as_float(_ldv8_105_1);
                                    part_hi_7[0 + 2] = __uint_as_float(_ldv8_105_2);
                                    part_hi_7[0 + 3] = __uint_as_float(_ldv8_105_3);
                                    part_hi_7[0 + 4] = __uint_as_float(_ldv8_105_4);
                                    part_hi_7[0 + 5] = __uint_as_float(_ldv8_105_5);
                                    part_hi_7[0 + 6] = __uint_as_float(_ldv8_105_6);
                                    part_hi_7[0 + 7] = __uint_as_float(_ldv8_105_7);
                                }
                                {
                                    unsigned _ldv8_106_0;
                                    unsigned _ldv8_106_1;
                                    unsigned _ldv8_106_2;
                                    unsigned _ldv8_106_3;
                                    unsigned _ldv8_106_4;
                                    unsigned _ldv8_106_5;
                                    unsigned _ldv8_106_6;
                                    unsigned _ldv8_106_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_106_0), "=r"(_ldv8_106_1), "=r"(_ldv8_106_2), "=r"(_ldv8_106_3), "=r"(_ldv8_106_4), "=r"(_ldv8_106_5), "=r"(_ldv8_106_6), "=r"(_ldv8_106_7) : "l"((const void*)(part_o + (off_0_11 + 64 + 8))) : "memory");
                                    part_hi_7[8 + 0] = __uint_as_float(_ldv8_106_0);
                                    part_hi_7[8 + 1] = __uint_as_float(_ldv8_106_1);
                                    part_hi_7[8 + 2] = __uint_as_float(_ldv8_106_2);
                                    part_hi_7[8 + 3] = __uint_as_float(_ldv8_106_3);
                                    part_hi_7[8 + 4] = __uint_as_float(_ldv8_106_4);
                                    part_hi_7[8 + 5] = __uint_as_float(_ldv8_106_5);
                                    part_hi_7[8 + 6] = __uint_as_float(_ldv8_106_6);
                                    part_hi_7[8 + 7] = __uint_as_float(_ldv8_106_7);
                                }
                                acc_4[0] = __fmaf_rn(part_6[0], w_c_15, acc_4[0]);
                                acc_hi_5[0] = __fmaf_rn(part_hi_7[0], w_c_15, acc_hi_5[0]);
                                acc_4[1] = __fmaf_rn(part_6[1], w_c_15, acc_4[1]);
                                acc_hi_5[1] = __fmaf_rn(part_hi_7[1], w_c_15, acc_hi_5[1]);
                                acc_4[2] = __fmaf_rn(part_6[2], w_c_15, acc_4[2]);
                                acc_hi_5[2] = __fmaf_rn(part_hi_7[2], w_c_15, acc_hi_5[2]);
                                acc_4[3] = __fmaf_rn(part_6[3], w_c_15, acc_4[3]);
                                acc_hi_5[3] = __fmaf_rn(part_hi_7[3], w_c_15, acc_hi_5[3]);
                                acc_4[4] = __fmaf_rn(part_6[4], w_c_15, acc_4[4]);
                                acc_hi_5[4] = __fmaf_rn(part_hi_7[4], w_c_15, acc_hi_5[4]);
                                acc_4[5] = __fmaf_rn(part_6[5], w_c_15, acc_4[5]);
                                acc_hi_5[5] = __fmaf_rn(part_hi_7[5], w_c_15, acc_hi_5[5]);
                                acc_4[6] = __fmaf_rn(part_6[6], w_c_15, acc_4[6]);
                                acc_hi_5[6] = __fmaf_rn(part_hi_7[6], w_c_15, acc_hi_5[6]);
                                acc_4[7] = __fmaf_rn(part_6[7], w_c_15, acc_4[7]);
                                acc_hi_5[7] = __fmaf_rn(part_hi_7[7], w_c_15, acc_hi_5[7]);
                                acc_4[8] = __fmaf_rn(part_6[8], w_c_15, acc_4[8]);
                                acc_hi_5[8] = __fmaf_rn(part_hi_7[8], w_c_15, acc_hi_5[8]);
                                acc_4[9] = __fmaf_rn(part_6[9], w_c_15, acc_4[9]);
                                acc_hi_5[9] = __fmaf_rn(part_hi_7[9], w_c_15, acc_hi_5[9]);
                                acc_4[10] = __fmaf_rn(part_6[10], w_c_15, acc_4[10]);
                                acc_hi_5[10] = __fmaf_rn(part_hi_7[10], w_c_15, acc_hi_5[10]);
                                acc_4[11] = __fmaf_rn(part_6[11], w_c_15, acc_4[11]);
                                acc_hi_5[11] = __fmaf_rn(part_hi_7[11], w_c_15, acc_hi_5[11]);
                                acc_4[12] = __fmaf_rn(part_6[12], w_c_15, acc_4[12]);
                                acc_hi_5[12] = __fmaf_rn(part_hi_7[12], w_c_15, acc_hi_5[12]);
                                acc_4[13] = __fmaf_rn(part_6[13], w_c_15, acc_4[13]);
                                acc_hi_5[13] = __fmaf_rn(part_hi_7[13], w_c_15, acc_hi_5[13]);
                                acc_4[14] = __fmaf_rn(part_6[14], w_c_15, acc_4[14]);
                                acc_hi_5[14] = __fmaf_rn(part_hi_7[14], w_c_15, acc_hi_5[14]);
                                acc_4[15] = __fmaf_rn(part_6[15], w_c_15, acc_4[15]);
                                acc_hi_5[15] = __fmaf_rn(part_hi_7[15], w_c_15, acc_hi_5[15]);
                            }
                            const float2 _scale2_107 = {final_scale, final_scale};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(acc_4)[_ls], _scale2_107);
                            unsigned int acc_4_bf16[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_4[_lp*2 + 0], acc_4[_lp*2+1 + 0]));
                                acc_4_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o00_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + 64 ^ ((warp % 4 * 32 + lane) * 128 + 64 >> 7 & 7) << 4))), "r"(acc_4_bf16[0]), "r"(acc_4_bf16[1]), "r"(acc_4_bf16[2]), "r"(acc_4_bf16[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o00_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + 80 ^ ((warp % 4 * 32 + lane) * 128 + 80 >> 7 & 7) << 4))), "r"(acc_4_bf16[4]), "r"(acc_4_bf16[5]), "r"(acc_4_bf16[6]), "r"(acc_4_bf16[7]) : "memory");
                            const float2 _scale2_108 = {final_scale, final_scale};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(acc_hi_5)[_ls], _scale2_108);
                            unsigned int acc_hi_5_bf16[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_hi_5[_lp*2 + 0], acc_hi_5[_lp*2+1 + 0]));
                                acc_hi_5_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o01_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + 64 ^ ((warp % 4 * 32 + lane) * 128 + 64 >> 7 & 7) << 4))), "r"(acc_hi_5_bf16[0]), "r"(acc_hi_5_bf16[1]), "r"(acc_hi_5_bf16[2]), "r"(acc_hi_5_bf16[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o01_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + 80 ^ ((warp % 4 * 32 + lane) * 128 + 80 >> 7 & 7) << 4))), "r"(acc_hi_5_bf16[4]), "r"(acc_hi_5_bf16[5]), "r"(acc_hi_5_bf16[6]), "r"(acc_hi_5_bf16[7]) : "memory");
                            float acc_8[16];
                            float acc_hi_9[16];
                            acc_8[0] = 0.0f;
                            acc_hi_9[0] = 0.0f;
                            acc_8[1] = 0.0f;
                            acc_hi_9[1] = 0.0f;
                            acc_8[2] = 0.0f;
                            acc_hi_9[2] = 0.0f;
                            acc_8[3] = 0.0f;
                            acc_hi_9[3] = 0.0f;
                            acc_8[4] = 0.0f;
                            acc_hi_9[4] = 0.0f;
                            acc_8[5] = 0.0f;
                            acc_hi_9[5] = 0.0f;
                            acc_8[6] = 0.0f;
                            acc_hi_9[6] = 0.0f;
                            acc_8[7] = 0.0f;
                            acc_hi_9[7] = 0.0f;
                            acc_8[8] = 0.0f;
                            acc_hi_9[8] = 0.0f;
                            acc_8[9] = 0.0f;
                            acc_hi_9[9] = 0.0f;
                            acc_8[10] = 0.0f;
                            acc_hi_9[10] = 0.0f;
                            acc_8[11] = 0.0f;
                            acc_hi_9[11] = 0.0f;
                            acc_8[12] = 0.0f;
                            acc_hi_9[12] = 0.0f;
                            acc_8[13] = 0.0f;
                            acc_hi_9[13] = 0.0f;
                            acc_8[14] = 0.0f;
                            acc_hi_9[14] = 0.0f;
                            acc_8[15] = 0.0f;
                            acc_hi_9[15] = 0.0f;
                            float part_10[16];
                            float part_hi_11[16];
                            if (n_chunks_u_0 > 0) {
                                unsigned int crow_12 = tile_slot0 * 512 + part_row;
                                float w_c_16 = approx_exp2(part_ml[crow_12 * 2] - row_max_all);
                                unsigned int off_0_12 = crow_12 * 128 + 48;
                                {
                                    unsigned _ldv8_109_0;
                                    unsigned _ldv8_109_1;
                                    unsigned _ldv8_109_2;
                                    unsigned _ldv8_109_3;
                                    unsigned _ldv8_109_4;
                                    unsigned _ldv8_109_5;
                                    unsigned _ldv8_109_6;
                                    unsigned _ldv8_109_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_109_0), "=r"(_ldv8_109_1), "=r"(_ldv8_109_2), "=r"(_ldv8_109_3), "=r"(_ldv8_109_4), "=r"(_ldv8_109_5), "=r"(_ldv8_109_6), "=r"(_ldv8_109_7) : "l"((const void*)(part_o + (off_0_12))) : "memory");
                                    part_10[0 + 0] = __uint_as_float(_ldv8_109_0);
                                    part_10[0 + 1] = __uint_as_float(_ldv8_109_1);
                                    part_10[0 + 2] = __uint_as_float(_ldv8_109_2);
                                    part_10[0 + 3] = __uint_as_float(_ldv8_109_3);
                                    part_10[0 + 4] = __uint_as_float(_ldv8_109_4);
                                    part_10[0 + 5] = __uint_as_float(_ldv8_109_5);
                                    part_10[0 + 6] = __uint_as_float(_ldv8_109_6);
                                    part_10[0 + 7] = __uint_as_float(_ldv8_109_7);
                                }
                                {
                                    unsigned _ldv8_110_0;
                                    unsigned _ldv8_110_1;
                                    unsigned _ldv8_110_2;
                                    unsigned _ldv8_110_3;
                                    unsigned _ldv8_110_4;
                                    unsigned _ldv8_110_5;
                                    unsigned _ldv8_110_6;
                                    unsigned _ldv8_110_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_110_0), "=r"(_ldv8_110_1), "=r"(_ldv8_110_2), "=r"(_ldv8_110_3), "=r"(_ldv8_110_4), "=r"(_ldv8_110_5), "=r"(_ldv8_110_6), "=r"(_ldv8_110_7) : "l"((const void*)(part_o + (off_0_12 + 8))) : "memory");
                                    part_10[8 + 0] = __uint_as_float(_ldv8_110_0);
                                    part_10[8 + 1] = __uint_as_float(_ldv8_110_1);
                                    part_10[8 + 2] = __uint_as_float(_ldv8_110_2);
                                    part_10[8 + 3] = __uint_as_float(_ldv8_110_3);
                                    part_10[8 + 4] = __uint_as_float(_ldv8_110_4);
                                    part_10[8 + 5] = __uint_as_float(_ldv8_110_5);
                                    part_10[8 + 6] = __uint_as_float(_ldv8_110_6);
                                    part_10[8 + 7] = __uint_as_float(_ldv8_110_7);
                                }
                                {
                                    unsigned _ldv8_111_0;
                                    unsigned _ldv8_111_1;
                                    unsigned _ldv8_111_2;
                                    unsigned _ldv8_111_3;
                                    unsigned _ldv8_111_4;
                                    unsigned _ldv8_111_5;
                                    unsigned _ldv8_111_6;
                                    unsigned _ldv8_111_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_111_0), "=r"(_ldv8_111_1), "=r"(_ldv8_111_2), "=r"(_ldv8_111_3), "=r"(_ldv8_111_4), "=r"(_ldv8_111_5), "=r"(_ldv8_111_6), "=r"(_ldv8_111_7) : "l"((const void*)(part_o + (off_0_12 + 64))) : "memory");
                                    part_hi_11[0 + 0] = __uint_as_float(_ldv8_111_0);
                                    part_hi_11[0 + 1] = __uint_as_float(_ldv8_111_1);
                                    part_hi_11[0 + 2] = __uint_as_float(_ldv8_111_2);
                                    part_hi_11[0 + 3] = __uint_as_float(_ldv8_111_3);
                                    part_hi_11[0 + 4] = __uint_as_float(_ldv8_111_4);
                                    part_hi_11[0 + 5] = __uint_as_float(_ldv8_111_5);
                                    part_hi_11[0 + 6] = __uint_as_float(_ldv8_111_6);
                                    part_hi_11[0 + 7] = __uint_as_float(_ldv8_111_7);
                                }
                                {
                                    unsigned _ldv8_112_0;
                                    unsigned _ldv8_112_1;
                                    unsigned _ldv8_112_2;
                                    unsigned _ldv8_112_3;
                                    unsigned _ldv8_112_4;
                                    unsigned _ldv8_112_5;
                                    unsigned _ldv8_112_6;
                                    unsigned _ldv8_112_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_112_0), "=r"(_ldv8_112_1), "=r"(_ldv8_112_2), "=r"(_ldv8_112_3), "=r"(_ldv8_112_4), "=r"(_ldv8_112_5), "=r"(_ldv8_112_6), "=r"(_ldv8_112_7) : "l"((const void*)(part_o + (off_0_12 + 64 + 8))) : "memory");
                                    part_hi_11[8 + 0] = __uint_as_float(_ldv8_112_0);
                                    part_hi_11[8 + 1] = __uint_as_float(_ldv8_112_1);
                                    part_hi_11[8 + 2] = __uint_as_float(_ldv8_112_2);
                                    part_hi_11[8 + 3] = __uint_as_float(_ldv8_112_3);
                                    part_hi_11[8 + 4] = __uint_as_float(_ldv8_112_4);
                                    part_hi_11[8 + 5] = __uint_as_float(_ldv8_112_5);
                                    part_hi_11[8 + 6] = __uint_as_float(_ldv8_112_6);
                                    part_hi_11[8 + 7] = __uint_as_float(_ldv8_112_7);
                                }
                                acc_8[0] = __fmaf_rn(part_10[0], w_c_16, acc_8[0]);
                                acc_hi_9[0] = __fmaf_rn(part_hi_11[0], w_c_16, acc_hi_9[0]);
                                acc_8[1] = __fmaf_rn(part_10[1], w_c_16, acc_8[1]);
                                acc_hi_9[1] = __fmaf_rn(part_hi_11[1], w_c_16, acc_hi_9[1]);
                                acc_8[2] = __fmaf_rn(part_10[2], w_c_16, acc_8[2]);
                                acc_hi_9[2] = __fmaf_rn(part_hi_11[2], w_c_16, acc_hi_9[2]);
                                acc_8[3] = __fmaf_rn(part_10[3], w_c_16, acc_8[3]);
                                acc_hi_9[3] = __fmaf_rn(part_hi_11[3], w_c_16, acc_hi_9[3]);
                                acc_8[4] = __fmaf_rn(part_10[4], w_c_16, acc_8[4]);
                                acc_hi_9[4] = __fmaf_rn(part_hi_11[4], w_c_16, acc_hi_9[4]);
                                acc_8[5] = __fmaf_rn(part_10[5], w_c_16, acc_8[5]);
                                acc_hi_9[5] = __fmaf_rn(part_hi_11[5], w_c_16, acc_hi_9[5]);
                                acc_8[6] = __fmaf_rn(part_10[6], w_c_16, acc_8[6]);
                                acc_hi_9[6] = __fmaf_rn(part_hi_11[6], w_c_16, acc_hi_9[6]);
                                acc_8[7] = __fmaf_rn(part_10[7], w_c_16, acc_8[7]);
                                acc_hi_9[7] = __fmaf_rn(part_hi_11[7], w_c_16, acc_hi_9[7]);
                                acc_8[8] = __fmaf_rn(part_10[8], w_c_16, acc_8[8]);
                                acc_hi_9[8] = __fmaf_rn(part_hi_11[8], w_c_16, acc_hi_9[8]);
                                acc_8[9] = __fmaf_rn(part_10[9], w_c_16, acc_8[9]);
                                acc_hi_9[9] = __fmaf_rn(part_hi_11[9], w_c_16, acc_hi_9[9]);
                                acc_8[10] = __fmaf_rn(part_10[10], w_c_16, acc_8[10]);
                                acc_hi_9[10] = __fmaf_rn(part_hi_11[10], w_c_16, acc_hi_9[10]);
                                acc_8[11] = __fmaf_rn(part_10[11], w_c_16, acc_8[11]);
                                acc_hi_9[11] = __fmaf_rn(part_hi_11[11], w_c_16, acc_hi_9[11]);
                                acc_8[12] = __fmaf_rn(part_10[12], w_c_16, acc_8[12]);
                                acc_hi_9[12] = __fmaf_rn(part_hi_11[12], w_c_16, acc_hi_9[12]);
                                acc_8[13] = __fmaf_rn(part_10[13], w_c_16, acc_8[13]);
                                acc_hi_9[13] = __fmaf_rn(part_hi_11[13], w_c_16, acc_hi_9[13]);
                                acc_8[14] = __fmaf_rn(part_10[14], w_c_16, acc_8[14]);
                                acc_hi_9[14] = __fmaf_rn(part_hi_11[14], w_c_16, acc_hi_9[14]);
                                acc_8[15] = __fmaf_rn(part_10[15], w_c_16, acc_8[15]);
                                acc_hi_9[15] = __fmaf_rn(part_hi_11[15], w_c_16, acc_hi_9[15]);
                            }
                            if (n_chunks_u_0 > 1) {
                                unsigned int crow_13 = (tile_slot0 + 1) * 512 + part_row;
                                float w_c_17 = approx_exp2(part_ml[crow_13 * 2] - row_max_all);
                                unsigned int off_0_13 = crow_13 * 128 + 48;
                                {
                                    unsigned _ldv8_113_0;
                                    unsigned _ldv8_113_1;
                                    unsigned _ldv8_113_2;
                                    unsigned _ldv8_113_3;
                                    unsigned _ldv8_113_4;
                                    unsigned _ldv8_113_5;
                                    unsigned _ldv8_113_6;
                                    unsigned _ldv8_113_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_113_0), "=r"(_ldv8_113_1), "=r"(_ldv8_113_2), "=r"(_ldv8_113_3), "=r"(_ldv8_113_4), "=r"(_ldv8_113_5), "=r"(_ldv8_113_6), "=r"(_ldv8_113_7) : "l"((const void*)(part_o + (off_0_13))) : "memory");
                                    part_10[0 + 0] = __uint_as_float(_ldv8_113_0);
                                    part_10[0 + 1] = __uint_as_float(_ldv8_113_1);
                                    part_10[0 + 2] = __uint_as_float(_ldv8_113_2);
                                    part_10[0 + 3] = __uint_as_float(_ldv8_113_3);
                                    part_10[0 + 4] = __uint_as_float(_ldv8_113_4);
                                    part_10[0 + 5] = __uint_as_float(_ldv8_113_5);
                                    part_10[0 + 6] = __uint_as_float(_ldv8_113_6);
                                    part_10[0 + 7] = __uint_as_float(_ldv8_113_7);
                                }
                                {
                                    unsigned _ldv8_114_0;
                                    unsigned _ldv8_114_1;
                                    unsigned _ldv8_114_2;
                                    unsigned _ldv8_114_3;
                                    unsigned _ldv8_114_4;
                                    unsigned _ldv8_114_5;
                                    unsigned _ldv8_114_6;
                                    unsigned _ldv8_114_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_114_0), "=r"(_ldv8_114_1), "=r"(_ldv8_114_2), "=r"(_ldv8_114_3), "=r"(_ldv8_114_4), "=r"(_ldv8_114_5), "=r"(_ldv8_114_6), "=r"(_ldv8_114_7) : "l"((const void*)(part_o + (off_0_13 + 8))) : "memory");
                                    part_10[8 + 0] = __uint_as_float(_ldv8_114_0);
                                    part_10[8 + 1] = __uint_as_float(_ldv8_114_1);
                                    part_10[8 + 2] = __uint_as_float(_ldv8_114_2);
                                    part_10[8 + 3] = __uint_as_float(_ldv8_114_3);
                                    part_10[8 + 4] = __uint_as_float(_ldv8_114_4);
                                    part_10[8 + 5] = __uint_as_float(_ldv8_114_5);
                                    part_10[8 + 6] = __uint_as_float(_ldv8_114_6);
                                    part_10[8 + 7] = __uint_as_float(_ldv8_114_7);
                                }
                                {
                                    unsigned _ldv8_115_0;
                                    unsigned _ldv8_115_1;
                                    unsigned _ldv8_115_2;
                                    unsigned _ldv8_115_3;
                                    unsigned _ldv8_115_4;
                                    unsigned _ldv8_115_5;
                                    unsigned _ldv8_115_6;
                                    unsigned _ldv8_115_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_115_0), "=r"(_ldv8_115_1), "=r"(_ldv8_115_2), "=r"(_ldv8_115_3), "=r"(_ldv8_115_4), "=r"(_ldv8_115_5), "=r"(_ldv8_115_6), "=r"(_ldv8_115_7) : "l"((const void*)(part_o + (off_0_13 + 64))) : "memory");
                                    part_hi_11[0 + 0] = __uint_as_float(_ldv8_115_0);
                                    part_hi_11[0 + 1] = __uint_as_float(_ldv8_115_1);
                                    part_hi_11[0 + 2] = __uint_as_float(_ldv8_115_2);
                                    part_hi_11[0 + 3] = __uint_as_float(_ldv8_115_3);
                                    part_hi_11[0 + 4] = __uint_as_float(_ldv8_115_4);
                                    part_hi_11[0 + 5] = __uint_as_float(_ldv8_115_5);
                                    part_hi_11[0 + 6] = __uint_as_float(_ldv8_115_6);
                                    part_hi_11[0 + 7] = __uint_as_float(_ldv8_115_7);
                                }
                                {
                                    unsigned _ldv8_116_0;
                                    unsigned _ldv8_116_1;
                                    unsigned _ldv8_116_2;
                                    unsigned _ldv8_116_3;
                                    unsigned _ldv8_116_4;
                                    unsigned _ldv8_116_5;
                                    unsigned _ldv8_116_6;
                                    unsigned _ldv8_116_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_116_0), "=r"(_ldv8_116_1), "=r"(_ldv8_116_2), "=r"(_ldv8_116_3), "=r"(_ldv8_116_4), "=r"(_ldv8_116_5), "=r"(_ldv8_116_6), "=r"(_ldv8_116_7) : "l"((const void*)(part_o + (off_0_13 + 64 + 8))) : "memory");
                                    part_hi_11[8 + 0] = __uint_as_float(_ldv8_116_0);
                                    part_hi_11[8 + 1] = __uint_as_float(_ldv8_116_1);
                                    part_hi_11[8 + 2] = __uint_as_float(_ldv8_116_2);
                                    part_hi_11[8 + 3] = __uint_as_float(_ldv8_116_3);
                                    part_hi_11[8 + 4] = __uint_as_float(_ldv8_116_4);
                                    part_hi_11[8 + 5] = __uint_as_float(_ldv8_116_5);
                                    part_hi_11[8 + 6] = __uint_as_float(_ldv8_116_6);
                                    part_hi_11[8 + 7] = __uint_as_float(_ldv8_116_7);
                                }
                                acc_8[0] = __fmaf_rn(part_10[0], w_c_17, acc_8[0]);
                                acc_hi_9[0] = __fmaf_rn(part_hi_11[0], w_c_17, acc_hi_9[0]);
                                acc_8[1] = __fmaf_rn(part_10[1], w_c_17, acc_8[1]);
                                acc_hi_9[1] = __fmaf_rn(part_hi_11[1], w_c_17, acc_hi_9[1]);
                                acc_8[2] = __fmaf_rn(part_10[2], w_c_17, acc_8[2]);
                                acc_hi_9[2] = __fmaf_rn(part_hi_11[2], w_c_17, acc_hi_9[2]);
                                acc_8[3] = __fmaf_rn(part_10[3], w_c_17, acc_8[3]);
                                acc_hi_9[3] = __fmaf_rn(part_hi_11[3], w_c_17, acc_hi_9[3]);
                                acc_8[4] = __fmaf_rn(part_10[4], w_c_17, acc_8[4]);
                                acc_hi_9[4] = __fmaf_rn(part_hi_11[4], w_c_17, acc_hi_9[4]);
                                acc_8[5] = __fmaf_rn(part_10[5], w_c_17, acc_8[5]);
                                acc_hi_9[5] = __fmaf_rn(part_hi_11[5], w_c_17, acc_hi_9[5]);
                                acc_8[6] = __fmaf_rn(part_10[6], w_c_17, acc_8[6]);
                                acc_hi_9[6] = __fmaf_rn(part_hi_11[6], w_c_17, acc_hi_9[6]);
                                acc_8[7] = __fmaf_rn(part_10[7], w_c_17, acc_8[7]);
                                acc_hi_9[7] = __fmaf_rn(part_hi_11[7], w_c_17, acc_hi_9[7]);
                                acc_8[8] = __fmaf_rn(part_10[8], w_c_17, acc_8[8]);
                                acc_hi_9[8] = __fmaf_rn(part_hi_11[8], w_c_17, acc_hi_9[8]);
                                acc_8[9] = __fmaf_rn(part_10[9], w_c_17, acc_8[9]);
                                acc_hi_9[9] = __fmaf_rn(part_hi_11[9], w_c_17, acc_hi_9[9]);
                                acc_8[10] = __fmaf_rn(part_10[10], w_c_17, acc_8[10]);
                                acc_hi_9[10] = __fmaf_rn(part_hi_11[10], w_c_17, acc_hi_9[10]);
                                acc_8[11] = __fmaf_rn(part_10[11], w_c_17, acc_8[11]);
                                acc_hi_9[11] = __fmaf_rn(part_hi_11[11], w_c_17, acc_hi_9[11]);
                                acc_8[12] = __fmaf_rn(part_10[12], w_c_17, acc_8[12]);
                                acc_hi_9[12] = __fmaf_rn(part_hi_11[12], w_c_17, acc_hi_9[12]);
                                acc_8[13] = __fmaf_rn(part_10[13], w_c_17, acc_8[13]);
                                acc_hi_9[13] = __fmaf_rn(part_hi_11[13], w_c_17, acc_hi_9[13]);
                                acc_8[14] = __fmaf_rn(part_10[14], w_c_17, acc_8[14]);
                                acc_hi_9[14] = __fmaf_rn(part_hi_11[14], w_c_17, acc_hi_9[14]);
                                acc_8[15] = __fmaf_rn(part_10[15], w_c_17, acc_8[15]);
                                acc_hi_9[15] = __fmaf_rn(part_hi_11[15], w_c_17, acc_hi_9[15]);
                            }
                            if (n_chunks_u_0 > 2) {
                                unsigned int crow_14 = (tile_slot0 + 2) * 512 + part_row;
                                float w_c_18 = approx_exp2(part_ml[crow_14 * 2] - row_max_all);
                                unsigned int off_0_14 = crow_14 * 128 + 48;
                                {
                                    unsigned _ldv8_117_0;
                                    unsigned _ldv8_117_1;
                                    unsigned _ldv8_117_2;
                                    unsigned _ldv8_117_3;
                                    unsigned _ldv8_117_4;
                                    unsigned _ldv8_117_5;
                                    unsigned _ldv8_117_6;
                                    unsigned _ldv8_117_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_117_0), "=r"(_ldv8_117_1), "=r"(_ldv8_117_2), "=r"(_ldv8_117_3), "=r"(_ldv8_117_4), "=r"(_ldv8_117_5), "=r"(_ldv8_117_6), "=r"(_ldv8_117_7) : "l"((const void*)(part_o + (off_0_14))) : "memory");
                                    part_10[0 + 0] = __uint_as_float(_ldv8_117_0);
                                    part_10[0 + 1] = __uint_as_float(_ldv8_117_1);
                                    part_10[0 + 2] = __uint_as_float(_ldv8_117_2);
                                    part_10[0 + 3] = __uint_as_float(_ldv8_117_3);
                                    part_10[0 + 4] = __uint_as_float(_ldv8_117_4);
                                    part_10[0 + 5] = __uint_as_float(_ldv8_117_5);
                                    part_10[0 + 6] = __uint_as_float(_ldv8_117_6);
                                    part_10[0 + 7] = __uint_as_float(_ldv8_117_7);
                                }
                                {
                                    unsigned _ldv8_118_0;
                                    unsigned _ldv8_118_1;
                                    unsigned _ldv8_118_2;
                                    unsigned _ldv8_118_3;
                                    unsigned _ldv8_118_4;
                                    unsigned _ldv8_118_5;
                                    unsigned _ldv8_118_6;
                                    unsigned _ldv8_118_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_118_0), "=r"(_ldv8_118_1), "=r"(_ldv8_118_2), "=r"(_ldv8_118_3), "=r"(_ldv8_118_4), "=r"(_ldv8_118_5), "=r"(_ldv8_118_6), "=r"(_ldv8_118_7) : "l"((const void*)(part_o + (off_0_14 + 8))) : "memory");
                                    part_10[8 + 0] = __uint_as_float(_ldv8_118_0);
                                    part_10[8 + 1] = __uint_as_float(_ldv8_118_1);
                                    part_10[8 + 2] = __uint_as_float(_ldv8_118_2);
                                    part_10[8 + 3] = __uint_as_float(_ldv8_118_3);
                                    part_10[8 + 4] = __uint_as_float(_ldv8_118_4);
                                    part_10[8 + 5] = __uint_as_float(_ldv8_118_5);
                                    part_10[8 + 6] = __uint_as_float(_ldv8_118_6);
                                    part_10[8 + 7] = __uint_as_float(_ldv8_118_7);
                                }
                                {
                                    unsigned _ldv8_119_0;
                                    unsigned _ldv8_119_1;
                                    unsigned _ldv8_119_2;
                                    unsigned _ldv8_119_3;
                                    unsigned _ldv8_119_4;
                                    unsigned _ldv8_119_5;
                                    unsigned _ldv8_119_6;
                                    unsigned _ldv8_119_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_119_0), "=r"(_ldv8_119_1), "=r"(_ldv8_119_2), "=r"(_ldv8_119_3), "=r"(_ldv8_119_4), "=r"(_ldv8_119_5), "=r"(_ldv8_119_6), "=r"(_ldv8_119_7) : "l"((const void*)(part_o + (off_0_14 + 64))) : "memory");
                                    part_hi_11[0 + 0] = __uint_as_float(_ldv8_119_0);
                                    part_hi_11[0 + 1] = __uint_as_float(_ldv8_119_1);
                                    part_hi_11[0 + 2] = __uint_as_float(_ldv8_119_2);
                                    part_hi_11[0 + 3] = __uint_as_float(_ldv8_119_3);
                                    part_hi_11[0 + 4] = __uint_as_float(_ldv8_119_4);
                                    part_hi_11[0 + 5] = __uint_as_float(_ldv8_119_5);
                                    part_hi_11[0 + 6] = __uint_as_float(_ldv8_119_6);
                                    part_hi_11[0 + 7] = __uint_as_float(_ldv8_119_7);
                                }
                                {
                                    unsigned _ldv8_120_0;
                                    unsigned _ldv8_120_1;
                                    unsigned _ldv8_120_2;
                                    unsigned _ldv8_120_3;
                                    unsigned _ldv8_120_4;
                                    unsigned _ldv8_120_5;
                                    unsigned _ldv8_120_6;
                                    unsigned _ldv8_120_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_120_0), "=r"(_ldv8_120_1), "=r"(_ldv8_120_2), "=r"(_ldv8_120_3), "=r"(_ldv8_120_4), "=r"(_ldv8_120_5), "=r"(_ldv8_120_6), "=r"(_ldv8_120_7) : "l"((const void*)(part_o + (off_0_14 + 64 + 8))) : "memory");
                                    part_hi_11[8 + 0] = __uint_as_float(_ldv8_120_0);
                                    part_hi_11[8 + 1] = __uint_as_float(_ldv8_120_1);
                                    part_hi_11[8 + 2] = __uint_as_float(_ldv8_120_2);
                                    part_hi_11[8 + 3] = __uint_as_float(_ldv8_120_3);
                                    part_hi_11[8 + 4] = __uint_as_float(_ldv8_120_4);
                                    part_hi_11[8 + 5] = __uint_as_float(_ldv8_120_5);
                                    part_hi_11[8 + 6] = __uint_as_float(_ldv8_120_6);
                                    part_hi_11[8 + 7] = __uint_as_float(_ldv8_120_7);
                                }
                                acc_8[0] = __fmaf_rn(part_10[0], w_c_18, acc_8[0]);
                                acc_hi_9[0] = __fmaf_rn(part_hi_11[0], w_c_18, acc_hi_9[0]);
                                acc_8[1] = __fmaf_rn(part_10[1], w_c_18, acc_8[1]);
                                acc_hi_9[1] = __fmaf_rn(part_hi_11[1], w_c_18, acc_hi_9[1]);
                                acc_8[2] = __fmaf_rn(part_10[2], w_c_18, acc_8[2]);
                                acc_hi_9[2] = __fmaf_rn(part_hi_11[2], w_c_18, acc_hi_9[2]);
                                acc_8[3] = __fmaf_rn(part_10[3], w_c_18, acc_8[3]);
                                acc_hi_9[3] = __fmaf_rn(part_hi_11[3], w_c_18, acc_hi_9[3]);
                                acc_8[4] = __fmaf_rn(part_10[4], w_c_18, acc_8[4]);
                                acc_hi_9[4] = __fmaf_rn(part_hi_11[4], w_c_18, acc_hi_9[4]);
                                acc_8[5] = __fmaf_rn(part_10[5], w_c_18, acc_8[5]);
                                acc_hi_9[5] = __fmaf_rn(part_hi_11[5], w_c_18, acc_hi_9[5]);
                                acc_8[6] = __fmaf_rn(part_10[6], w_c_18, acc_8[6]);
                                acc_hi_9[6] = __fmaf_rn(part_hi_11[6], w_c_18, acc_hi_9[6]);
                                acc_8[7] = __fmaf_rn(part_10[7], w_c_18, acc_8[7]);
                                acc_hi_9[7] = __fmaf_rn(part_hi_11[7], w_c_18, acc_hi_9[7]);
                                acc_8[8] = __fmaf_rn(part_10[8], w_c_18, acc_8[8]);
                                acc_hi_9[8] = __fmaf_rn(part_hi_11[8], w_c_18, acc_hi_9[8]);
                                acc_8[9] = __fmaf_rn(part_10[9], w_c_18, acc_8[9]);
                                acc_hi_9[9] = __fmaf_rn(part_hi_11[9], w_c_18, acc_hi_9[9]);
                                acc_8[10] = __fmaf_rn(part_10[10], w_c_18, acc_8[10]);
                                acc_hi_9[10] = __fmaf_rn(part_hi_11[10], w_c_18, acc_hi_9[10]);
                                acc_8[11] = __fmaf_rn(part_10[11], w_c_18, acc_8[11]);
                                acc_hi_9[11] = __fmaf_rn(part_hi_11[11], w_c_18, acc_hi_9[11]);
                                acc_8[12] = __fmaf_rn(part_10[12], w_c_18, acc_8[12]);
                                acc_hi_9[12] = __fmaf_rn(part_hi_11[12], w_c_18, acc_hi_9[12]);
                                acc_8[13] = __fmaf_rn(part_10[13], w_c_18, acc_8[13]);
                                acc_hi_9[13] = __fmaf_rn(part_hi_11[13], w_c_18, acc_hi_9[13]);
                                acc_8[14] = __fmaf_rn(part_10[14], w_c_18, acc_8[14]);
                                acc_hi_9[14] = __fmaf_rn(part_hi_11[14], w_c_18, acc_hi_9[14]);
                                acc_8[15] = __fmaf_rn(part_10[15], w_c_18, acc_8[15]);
                                acc_hi_9[15] = __fmaf_rn(part_hi_11[15], w_c_18, acc_hi_9[15]);
                            }
                            if (n_chunks_u_0 > 3) {
                                unsigned int crow_15 = (tile_slot0 + 3) * 512 + part_row;
                                float w_c_19 = approx_exp2(part_ml[crow_15 * 2] - row_max_all);
                                unsigned int off_0_15 = crow_15 * 128 + 48;
                                {
                                    unsigned _ldv8_121_0;
                                    unsigned _ldv8_121_1;
                                    unsigned _ldv8_121_2;
                                    unsigned _ldv8_121_3;
                                    unsigned _ldv8_121_4;
                                    unsigned _ldv8_121_5;
                                    unsigned _ldv8_121_6;
                                    unsigned _ldv8_121_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_121_0), "=r"(_ldv8_121_1), "=r"(_ldv8_121_2), "=r"(_ldv8_121_3), "=r"(_ldv8_121_4), "=r"(_ldv8_121_5), "=r"(_ldv8_121_6), "=r"(_ldv8_121_7) : "l"((const void*)(part_o + (off_0_15))) : "memory");
                                    part_10[0 + 0] = __uint_as_float(_ldv8_121_0);
                                    part_10[0 + 1] = __uint_as_float(_ldv8_121_1);
                                    part_10[0 + 2] = __uint_as_float(_ldv8_121_2);
                                    part_10[0 + 3] = __uint_as_float(_ldv8_121_3);
                                    part_10[0 + 4] = __uint_as_float(_ldv8_121_4);
                                    part_10[0 + 5] = __uint_as_float(_ldv8_121_5);
                                    part_10[0 + 6] = __uint_as_float(_ldv8_121_6);
                                    part_10[0 + 7] = __uint_as_float(_ldv8_121_7);
                                }
                                {
                                    unsigned _ldv8_122_0;
                                    unsigned _ldv8_122_1;
                                    unsigned _ldv8_122_2;
                                    unsigned _ldv8_122_3;
                                    unsigned _ldv8_122_4;
                                    unsigned _ldv8_122_5;
                                    unsigned _ldv8_122_6;
                                    unsigned _ldv8_122_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_122_0), "=r"(_ldv8_122_1), "=r"(_ldv8_122_2), "=r"(_ldv8_122_3), "=r"(_ldv8_122_4), "=r"(_ldv8_122_5), "=r"(_ldv8_122_6), "=r"(_ldv8_122_7) : "l"((const void*)(part_o + (off_0_15 + 8))) : "memory");
                                    part_10[8 + 0] = __uint_as_float(_ldv8_122_0);
                                    part_10[8 + 1] = __uint_as_float(_ldv8_122_1);
                                    part_10[8 + 2] = __uint_as_float(_ldv8_122_2);
                                    part_10[8 + 3] = __uint_as_float(_ldv8_122_3);
                                    part_10[8 + 4] = __uint_as_float(_ldv8_122_4);
                                    part_10[8 + 5] = __uint_as_float(_ldv8_122_5);
                                    part_10[8 + 6] = __uint_as_float(_ldv8_122_6);
                                    part_10[8 + 7] = __uint_as_float(_ldv8_122_7);
                                }
                                {
                                    unsigned _ldv8_123_0;
                                    unsigned _ldv8_123_1;
                                    unsigned _ldv8_123_2;
                                    unsigned _ldv8_123_3;
                                    unsigned _ldv8_123_4;
                                    unsigned _ldv8_123_5;
                                    unsigned _ldv8_123_6;
                                    unsigned _ldv8_123_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_123_0), "=r"(_ldv8_123_1), "=r"(_ldv8_123_2), "=r"(_ldv8_123_3), "=r"(_ldv8_123_4), "=r"(_ldv8_123_5), "=r"(_ldv8_123_6), "=r"(_ldv8_123_7) : "l"((const void*)(part_o + (off_0_15 + 64))) : "memory");
                                    part_hi_11[0 + 0] = __uint_as_float(_ldv8_123_0);
                                    part_hi_11[0 + 1] = __uint_as_float(_ldv8_123_1);
                                    part_hi_11[0 + 2] = __uint_as_float(_ldv8_123_2);
                                    part_hi_11[0 + 3] = __uint_as_float(_ldv8_123_3);
                                    part_hi_11[0 + 4] = __uint_as_float(_ldv8_123_4);
                                    part_hi_11[0 + 5] = __uint_as_float(_ldv8_123_5);
                                    part_hi_11[0 + 6] = __uint_as_float(_ldv8_123_6);
                                    part_hi_11[0 + 7] = __uint_as_float(_ldv8_123_7);
                                }
                                {
                                    unsigned _ldv8_124_0;
                                    unsigned _ldv8_124_1;
                                    unsigned _ldv8_124_2;
                                    unsigned _ldv8_124_3;
                                    unsigned _ldv8_124_4;
                                    unsigned _ldv8_124_5;
                                    unsigned _ldv8_124_6;
                                    unsigned _ldv8_124_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_124_0), "=r"(_ldv8_124_1), "=r"(_ldv8_124_2), "=r"(_ldv8_124_3), "=r"(_ldv8_124_4), "=r"(_ldv8_124_5), "=r"(_ldv8_124_6), "=r"(_ldv8_124_7) : "l"((const void*)(part_o + (off_0_15 + 64 + 8))) : "memory");
                                    part_hi_11[8 + 0] = __uint_as_float(_ldv8_124_0);
                                    part_hi_11[8 + 1] = __uint_as_float(_ldv8_124_1);
                                    part_hi_11[8 + 2] = __uint_as_float(_ldv8_124_2);
                                    part_hi_11[8 + 3] = __uint_as_float(_ldv8_124_3);
                                    part_hi_11[8 + 4] = __uint_as_float(_ldv8_124_4);
                                    part_hi_11[8 + 5] = __uint_as_float(_ldv8_124_5);
                                    part_hi_11[8 + 6] = __uint_as_float(_ldv8_124_6);
                                    part_hi_11[8 + 7] = __uint_as_float(_ldv8_124_7);
                                }
                                acc_8[0] = __fmaf_rn(part_10[0], w_c_19, acc_8[0]);
                                acc_hi_9[0] = __fmaf_rn(part_hi_11[0], w_c_19, acc_hi_9[0]);
                                acc_8[1] = __fmaf_rn(part_10[1], w_c_19, acc_8[1]);
                                acc_hi_9[1] = __fmaf_rn(part_hi_11[1], w_c_19, acc_hi_9[1]);
                                acc_8[2] = __fmaf_rn(part_10[2], w_c_19, acc_8[2]);
                                acc_hi_9[2] = __fmaf_rn(part_hi_11[2], w_c_19, acc_hi_9[2]);
                                acc_8[3] = __fmaf_rn(part_10[3], w_c_19, acc_8[3]);
                                acc_hi_9[3] = __fmaf_rn(part_hi_11[3], w_c_19, acc_hi_9[3]);
                                acc_8[4] = __fmaf_rn(part_10[4], w_c_19, acc_8[4]);
                                acc_hi_9[4] = __fmaf_rn(part_hi_11[4], w_c_19, acc_hi_9[4]);
                                acc_8[5] = __fmaf_rn(part_10[5], w_c_19, acc_8[5]);
                                acc_hi_9[5] = __fmaf_rn(part_hi_11[5], w_c_19, acc_hi_9[5]);
                                acc_8[6] = __fmaf_rn(part_10[6], w_c_19, acc_8[6]);
                                acc_hi_9[6] = __fmaf_rn(part_hi_11[6], w_c_19, acc_hi_9[6]);
                                acc_8[7] = __fmaf_rn(part_10[7], w_c_19, acc_8[7]);
                                acc_hi_9[7] = __fmaf_rn(part_hi_11[7], w_c_19, acc_hi_9[7]);
                                acc_8[8] = __fmaf_rn(part_10[8], w_c_19, acc_8[8]);
                                acc_hi_9[8] = __fmaf_rn(part_hi_11[8], w_c_19, acc_hi_9[8]);
                                acc_8[9] = __fmaf_rn(part_10[9], w_c_19, acc_8[9]);
                                acc_hi_9[9] = __fmaf_rn(part_hi_11[9], w_c_19, acc_hi_9[9]);
                                acc_8[10] = __fmaf_rn(part_10[10], w_c_19, acc_8[10]);
                                acc_hi_9[10] = __fmaf_rn(part_hi_11[10], w_c_19, acc_hi_9[10]);
                                acc_8[11] = __fmaf_rn(part_10[11], w_c_19, acc_8[11]);
                                acc_hi_9[11] = __fmaf_rn(part_hi_11[11], w_c_19, acc_hi_9[11]);
                                acc_8[12] = __fmaf_rn(part_10[12], w_c_19, acc_8[12]);
                                acc_hi_9[12] = __fmaf_rn(part_hi_11[12], w_c_19, acc_hi_9[12]);
                                acc_8[13] = __fmaf_rn(part_10[13], w_c_19, acc_8[13]);
                                acc_hi_9[13] = __fmaf_rn(part_hi_11[13], w_c_19, acc_hi_9[13]);
                                acc_8[14] = __fmaf_rn(part_10[14], w_c_19, acc_8[14]);
                                acc_hi_9[14] = __fmaf_rn(part_hi_11[14], w_c_19, acc_hi_9[14]);
                                acc_8[15] = __fmaf_rn(part_10[15], w_c_19, acc_8[15]);
                                acc_hi_9[15] = __fmaf_rn(part_hi_11[15], w_c_19, acc_hi_9[15]);
                            }
                            const float2 _scale2_125 = {final_scale, final_scale};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(acc_8)[_ls], _scale2_125);
                            unsigned int acc_8_bf16[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_8[_lp*2 + 0], acc_8[_lp*2+1 + 0]));
                                acc_8_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o00_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + 96 ^ ((warp % 4 * 32 + lane) * 128 + 96 >> 7 & 7) << 4))), "r"(acc_8_bf16[0]), "r"(acc_8_bf16[1]), "r"(acc_8_bf16[2]), "r"(acc_8_bf16[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o00_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + 112 ^ ((warp % 4 * 32 + lane) * 128 + 112 >> 7 & 7) << 4))), "r"(acc_8_bf16[4]), "r"(acc_8_bf16[5]), "r"(acc_8_bf16[6]), "r"(acc_8_bf16[7]) : "memory");
                            const float2 _scale2_126 = {final_scale, final_scale};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(acc_hi_9)[_ls], _scale2_126);
                            unsigned int acc_hi_9_bf16[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_hi_9[_lp*2 + 0], acc_hi_9[_lp*2+1 + 0]));
                                acc_hi_9_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o01_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + 96 ^ ((warp % 4 * 32 + lane) * 128 + 96 >> 7 & 7) << 4))), "r"(acc_hi_9_bf16[0]), "r"(acc_hi_9_bf16[1]), "r"(acc_hi_9_bf16[2]), "r"(acc_hi_9_bf16[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o01_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + 112 ^ ((warp % 4 * 32 + lane) * 128 + 112 >> 7 & 7) << 4))), "r"(acc_hi_9_bf16[4]), "r"(acc_hi_9_bf16[5]), "r"(acc_hi_9_bf16[6]), "r"(acc_hi_9_bf16[7]) : "memory");
                            asm volatile("fence.proxy.async;" ::: "memory");
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            asm volatile("barrier.sync 1, 128;" ::: "memory");
                            if (warp == 0) {
                                if (elect_sync()) {
                                    tma_store_4d((&O), 0, out_row_1, out_head_1, out_batch_1, smem_o00_addr);
                                    tma_store_4d((&O), 64, out_row_1, out_head_1, out_batch_1, smem_o01_addr);
                                }
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                        } else {
                            if (warp == 4) {
                                asm volatile("cp.async.bulk.wait_group.read 0;");
                            }
                            asm volatile("barrier.sync 4, 128;" ::: "memory");
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            float acc_1[16];
                            float acc_hi_2[16];
                            acc_1[0] = 0.0f;
                            acc_hi_2[0] = 0.0f;
                            acc_1[1] = 0.0f;
                            acc_hi_2[1] = 0.0f;
                            acc_1[2] = 0.0f;
                            acc_hi_2[2] = 0.0f;
                            acc_1[3] = 0.0f;
                            acc_hi_2[3] = 0.0f;
                            acc_1[4] = 0.0f;
                            acc_hi_2[4] = 0.0f;
                            acc_1[5] = 0.0f;
                            acc_hi_2[5] = 0.0f;
                            acc_1[6] = 0.0f;
                            acc_hi_2[6] = 0.0f;
                            acc_1[7] = 0.0f;
                            acc_hi_2[7] = 0.0f;
                            acc_1[8] = 0.0f;
                            acc_hi_2[8] = 0.0f;
                            acc_1[9] = 0.0f;
                            acc_hi_2[9] = 0.0f;
                            acc_1[10] = 0.0f;
                            acc_hi_2[10] = 0.0f;
                            acc_1[11] = 0.0f;
                            acc_hi_2[11] = 0.0f;
                            acc_1[12] = 0.0f;
                            acc_hi_2[12] = 0.0f;
                            acc_1[13] = 0.0f;
                            acc_hi_2[13] = 0.0f;
                            acc_1[14] = 0.0f;
                            acc_hi_2[14] = 0.0f;
                            acc_1[15] = 0.0f;
                            acc_hi_2[15] = 0.0f;
                            float part_1[16];
                            float part_hi_1[16];
                            if (n_chunks_u_0 > 0) {
                                unsigned int crow_16 = tile_slot0 * 512 + part_row;
                                float w_c_20 = approx_exp2(part_ml[crow_16 * 2] - row_max_all);
                                unsigned int off_0_16 = crow_16 * 128;
                                {
                                    unsigned _ldv8_127_0;
                                    unsigned _ldv8_127_1;
                                    unsigned _ldv8_127_2;
                                    unsigned _ldv8_127_3;
                                    unsigned _ldv8_127_4;
                                    unsigned _ldv8_127_5;
                                    unsigned _ldv8_127_6;
                                    unsigned _ldv8_127_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_127_0), "=r"(_ldv8_127_1), "=r"(_ldv8_127_2), "=r"(_ldv8_127_3), "=r"(_ldv8_127_4), "=r"(_ldv8_127_5), "=r"(_ldv8_127_6), "=r"(_ldv8_127_7) : "l"((const void*)(part_o + (off_0_16))) : "memory");
                                    part_1[0 + 0] = __uint_as_float(_ldv8_127_0);
                                    part_1[0 + 1] = __uint_as_float(_ldv8_127_1);
                                    part_1[0 + 2] = __uint_as_float(_ldv8_127_2);
                                    part_1[0 + 3] = __uint_as_float(_ldv8_127_3);
                                    part_1[0 + 4] = __uint_as_float(_ldv8_127_4);
                                    part_1[0 + 5] = __uint_as_float(_ldv8_127_5);
                                    part_1[0 + 6] = __uint_as_float(_ldv8_127_6);
                                    part_1[0 + 7] = __uint_as_float(_ldv8_127_7);
                                }
                                {
                                    unsigned _ldv8_128_0;
                                    unsigned _ldv8_128_1;
                                    unsigned _ldv8_128_2;
                                    unsigned _ldv8_128_3;
                                    unsigned _ldv8_128_4;
                                    unsigned _ldv8_128_5;
                                    unsigned _ldv8_128_6;
                                    unsigned _ldv8_128_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_128_0), "=r"(_ldv8_128_1), "=r"(_ldv8_128_2), "=r"(_ldv8_128_3), "=r"(_ldv8_128_4), "=r"(_ldv8_128_5), "=r"(_ldv8_128_6), "=r"(_ldv8_128_7) : "l"((const void*)(part_o + (off_0_16 + 8))) : "memory");
                                    part_1[8 + 0] = __uint_as_float(_ldv8_128_0);
                                    part_1[8 + 1] = __uint_as_float(_ldv8_128_1);
                                    part_1[8 + 2] = __uint_as_float(_ldv8_128_2);
                                    part_1[8 + 3] = __uint_as_float(_ldv8_128_3);
                                    part_1[8 + 4] = __uint_as_float(_ldv8_128_4);
                                    part_1[8 + 5] = __uint_as_float(_ldv8_128_5);
                                    part_1[8 + 6] = __uint_as_float(_ldv8_128_6);
                                    part_1[8 + 7] = __uint_as_float(_ldv8_128_7);
                                }
                                {
                                    unsigned _ldv8_129_0;
                                    unsigned _ldv8_129_1;
                                    unsigned _ldv8_129_2;
                                    unsigned _ldv8_129_3;
                                    unsigned _ldv8_129_4;
                                    unsigned _ldv8_129_5;
                                    unsigned _ldv8_129_6;
                                    unsigned _ldv8_129_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_129_0), "=r"(_ldv8_129_1), "=r"(_ldv8_129_2), "=r"(_ldv8_129_3), "=r"(_ldv8_129_4), "=r"(_ldv8_129_5), "=r"(_ldv8_129_6), "=r"(_ldv8_129_7) : "l"((const void*)(part_o + (off_0_16 + 64))) : "memory");
                                    part_hi_1[0 + 0] = __uint_as_float(_ldv8_129_0);
                                    part_hi_1[0 + 1] = __uint_as_float(_ldv8_129_1);
                                    part_hi_1[0 + 2] = __uint_as_float(_ldv8_129_2);
                                    part_hi_1[0 + 3] = __uint_as_float(_ldv8_129_3);
                                    part_hi_1[0 + 4] = __uint_as_float(_ldv8_129_4);
                                    part_hi_1[0 + 5] = __uint_as_float(_ldv8_129_5);
                                    part_hi_1[0 + 6] = __uint_as_float(_ldv8_129_6);
                                    part_hi_1[0 + 7] = __uint_as_float(_ldv8_129_7);
                                }
                                {
                                    unsigned _ldv8_130_0;
                                    unsigned _ldv8_130_1;
                                    unsigned _ldv8_130_2;
                                    unsigned _ldv8_130_3;
                                    unsigned _ldv8_130_4;
                                    unsigned _ldv8_130_5;
                                    unsigned _ldv8_130_6;
                                    unsigned _ldv8_130_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_130_0), "=r"(_ldv8_130_1), "=r"(_ldv8_130_2), "=r"(_ldv8_130_3), "=r"(_ldv8_130_4), "=r"(_ldv8_130_5), "=r"(_ldv8_130_6), "=r"(_ldv8_130_7) : "l"((const void*)(part_o + (off_0_16 + 64 + 8))) : "memory");
                                    part_hi_1[8 + 0] = __uint_as_float(_ldv8_130_0);
                                    part_hi_1[8 + 1] = __uint_as_float(_ldv8_130_1);
                                    part_hi_1[8 + 2] = __uint_as_float(_ldv8_130_2);
                                    part_hi_1[8 + 3] = __uint_as_float(_ldv8_130_3);
                                    part_hi_1[8 + 4] = __uint_as_float(_ldv8_130_4);
                                    part_hi_1[8 + 5] = __uint_as_float(_ldv8_130_5);
                                    part_hi_1[8 + 6] = __uint_as_float(_ldv8_130_6);
                                    part_hi_1[8 + 7] = __uint_as_float(_ldv8_130_7);
                                }
                                acc_1[0] = __fmaf_rn(part_1[0], w_c_20, acc_1[0]);
                                acc_hi_2[0] = __fmaf_rn(part_hi_1[0], w_c_20, acc_hi_2[0]);
                                acc_1[1] = __fmaf_rn(part_1[1], w_c_20, acc_1[1]);
                                acc_hi_2[1] = __fmaf_rn(part_hi_1[1], w_c_20, acc_hi_2[1]);
                                acc_1[2] = __fmaf_rn(part_1[2], w_c_20, acc_1[2]);
                                acc_hi_2[2] = __fmaf_rn(part_hi_1[2], w_c_20, acc_hi_2[2]);
                                acc_1[3] = __fmaf_rn(part_1[3], w_c_20, acc_1[3]);
                                acc_hi_2[3] = __fmaf_rn(part_hi_1[3], w_c_20, acc_hi_2[3]);
                                acc_1[4] = __fmaf_rn(part_1[4], w_c_20, acc_1[4]);
                                acc_hi_2[4] = __fmaf_rn(part_hi_1[4], w_c_20, acc_hi_2[4]);
                                acc_1[5] = __fmaf_rn(part_1[5], w_c_20, acc_1[5]);
                                acc_hi_2[5] = __fmaf_rn(part_hi_1[5], w_c_20, acc_hi_2[5]);
                                acc_1[6] = __fmaf_rn(part_1[6], w_c_20, acc_1[6]);
                                acc_hi_2[6] = __fmaf_rn(part_hi_1[6], w_c_20, acc_hi_2[6]);
                                acc_1[7] = __fmaf_rn(part_1[7], w_c_20, acc_1[7]);
                                acc_hi_2[7] = __fmaf_rn(part_hi_1[7], w_c_20, acc_hi_2[7]);
                                acc_1[8] = __fmaf_rn(part_1[8], w_c_20, acc_1[8]);
                                acc_hi_2[8] = __fmaf_rn(part_hi_1[8], w_c_20, acc_hi_2[8]);
                                acc_1[9] = __fmaf_rn(part_1[9], w_c_20, acc_1[9]);
                                acc_hi_2[9] = __fmaf_rn(part_hi_1[9], w_c_20, acc_hi_2[9]);
                                acc_1[10] = __fmaf_rn(part_1[10], w_c_20, acc_1[10]);
                                acc_hi_2[10] = __fmaf_rn(part_hi_1[10], w_c_20, acc_hi_2[10]);
                                acc_1[11] = __fmaf_rn(part_1[11], w_c_20, acc_1[11]);
                                acc_hi_2[11] = __fmaf_rn(part_hi_1[11], w_c_20, acc_hi_2[11]);
                                acc_1[12] = __fmaf_rn(part_1[12], w_c_20, acc_1[12]);
                                acc_hi_2[12] = __fmaf_rn(part_hi_1[12], w_c_20, acc_hi_2[12]);
                                acc_1[13] = __fmaf_rn(part_1[13], w_c_20, acc_1[13]);
                                acc_hi_2[13] = __fmaf_rn(part_hi_1[13], w_c_20, acc_hi_2[13]);
                                acc_1[14] = __fmaf_rn(part_1[14], w_c_20, acc_1[14]);
                                acc_hi_2[14] = __fmaf_rn(part_hi_1[14], w_c_20, acc_hi_2[14]);
                                acc_1[15] = __fmaf_rn(part_1[15], w_c_20, acc_1[15]);
                                acc_hi_2[15] = __fmaf_rn(part_hi_1[15], w_c_20, acc_hi_2[15]);
                            }
                            if (n_chunks_u_0 > 1) {
                                unsigned int crow_17 = (tile_slot0 + 1) * 512 + part_row;
                                float w_c_21 = approx_exp2(part_ml[crow_17 * 2] - row_max_all);
                                unsigned int off_0_17 = crow_17 * 128;
                                {
                                    unsigned _ldv8_131_0;
                                    unsigned _ldv8_131_1;
                                    unsigned _ldv8_131_2;
                                    unsigned _ldv8_131_3;
                                    unsigned _ldv8_131_4;
                                    unsigned _ldv8_131_5;
                                    unsigned _ldv8_131_6;
                                    unsigned _ldv8_131_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_131_0), "=r"(_ldv8_131_1), "=r"(_ldv8_131_2), "=r"(_ldv8_131_3), "=r"(_ldv8_131_4), "=r"(_ldv8_131_5), "=r"(_ldv8_131_6), "=r"(_ldv8_131_7) : "l"((const void*)(part_o + (off_0_17))) : "memory");
                                    part_1[0 + 0] = __uint_as_float(_ldv8_131_0);
                                    part_1[0 + 1] = __uint_as_float(_ldv8_131_1);
                                    part_1[0 + 2] = __uint_as_float(_ldv8_131_2);
                                    part_1[0 + 3] = __uint_as_float(_ldv8_131_3);
                                    part_1[0 + 4] = __uint_as_float(_ldv8_131_4);
                                    part_1[0 + 5] = __uint_as_float(_ldv8_131_5);
                                    part_1[0 + 6] = __uint_as_float(_ldv8_131_6);
                                    part_1[0 + 7] = __uint_as_float(_ldv8_131_7);
                                }
                                {
                                    unsigned _ldv8_132_0;
                                    unsigned _ldv8_132_1;
                                    unsigned _ldv8_132_2;
                                    unsigned _ldv8_132_3;
                                    unsigned _ldv8_132_4;
                                    unsigned _ldv8_132_5;
                                    unsigned _ldv8_132_6;
                                    unsigned _ldv8_132_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_132_0), "=r"(_ldv8_132_1), "=r"(_ldv8_132_2), "=r"(_ldv8_132_3), "=r"(_ldv8_132_4), "=r"(_ldv8_132_5), "=r"(_ldv8_132_6), "=r"(_ldv8_132_7) : "l"((const void*)(part_o + (off_0_17 + 8))) : "memory");
                                    part_1[8 + 0] = __uint_as_float(_ldv8_132_0);
                                    part_1[8 + 1] = __uint_as_float(_ldv8_132_1);
                                    part_1[8 + 2] = __uint_as_float(_ldv8_132_2);
                                    part_1[8 + 3] = __uint_as_float(_ldv8_132_3);
                                    part_1[8 + 4] = __uint_as_float(_ldv8_132_4);
                                    part_1[8 + 5] = __uint_as_float(_ldv8_132_5);
                                    part_1[8 + 6] = __uint_as_float(_ldv8_132_6);
                                    part_1[8 + 7] = __uint_as_float(_ldv8_132_7);
                                }
                                {
                                    unsigned _ldv8_133_0;
                                    unsigned _ldv8_133_1;
                                    unsigned _ldv8_133_2;
                                    unsigned _ldv8_133_3;
                                    unsigned _ldv8_133_4;
                                    unsigned _ldv8_133_5;
                                    unsigned _ldv8_133_6;
                                    unsigned _ldv8_133_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_133_0), "=r"(_ldv8_133_1), "=r"(_ldv8_133_2), "=r"(_ldv8_133_3), "=r"(_ldv8_133_4), "=r"(_ldv8_133_5), "=r"(_ldv8_133_6), "=r"(_ldv8_133_7) : "l"((const void*)(part_o + (off_0_17 + 64))) : "memory");
                                    part_hi_1[0 + 0] = __uint_as_float(_ldv8_133_0);
                                    part_hi_1[0 + 1] = __uint_as_float(_ldv8_133_1);
                                    part_hi_1[0 + 2] = __uint_as_float(_ldv8_133_2);
                                    part_hi_1[0 + 3] = __uint_as_float(_ldv8_133_3);
                                    part_hi_1[0 + 4] = __uint_as_float(_ldv8_133_4);
                                    part_hi_1[0 + 5] = __uint_as_float(_ldv8_133_5);
                                    part_hi_1[0 + 6] = __uint_as_float(_ldv8_133_6);
                                    part_hi_1[0 + 7] = __uint_as_float(_ldv8_133_7);
                                }
                                {
                                    unsigned _ldv8_134_0;
                                    unsigned _ldv8_134_1;
                                    unsigned _ldv8_134_2;
                                    unsigned _ldv8_134_3;
                                    unsigned _ldv8_134_4;
                                    unsigned _ldv8_134_5;
                                    unsigned _ldv8_134_6;
                                    unsigned _ldv8_134_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_134_0), "=r"(_ldv8_134_1), "=r"(_ldv8_134_2), "=r"(_ldv8_134_3), "=r"(_ldv8_134_4), "=r"(_ldv8_134_5), "=r"(_ldv8_134_6), "=r"(_ldv8_134_7) : "l"((const void*)(part_o + (off_0_17 + 64 + 8))) : "memory");
                                    part_hi_1[8 + 0] = __uint_as_float(_ldv8_134_0);
                                    part_hi_1[8 + 1] = __uint_as_float(_ldv8_134_1);
                                    part_hi_1[8 + 2] = __uint_as_float(_ldv8_134_2);
                                    part_hi_1[8 + 3] = __uint_as_float(_ldv8_134_3);
                                    part_hi_1[8 + 4] = __uint_as_float(_ldv8_134_4);
                                    part_hi_1[8 + 5] = __uint_as_float(_ldv8_134_5);
                                    part_hi_1[8 + 6] = __uint_as_float(_ldv8_134_6);
                                    part_hi_1[8 + 7] = __uint_as_float(_ldv8_134_7);
                                }
                                acc_1[0] = __fmaf_rn(part_1[0], w_c_21, acc_1[0]);
                                acc_hi_2[0] = __fmaf_rn(part_hi_1[0], w_c_21, acc_hi_2[0]);
                                acc_1[1] = __fmaf_rn(part_1[1], w_c_21, acc_1[1]);
                                acc_hi_2[1] = __fmaf_rn(part_hi_1[1], w_c_21, acc_hi_2[1]);
                                acc_1[2] = __fmaf_rn(part_1[2], w_c_21, acc_1[2]);
                                acc_hi_2[2] = __fmaf_rn(part_hi_1[2], w_c_21, acc_hi_2[2]);
                                acc_1[3] = __fmaf_rn(part_1[3], w_c_21, acc_1[3]);
                                acc_hi_2[3] = __fmaf_rn(part_hi_1[3], w_c_21, acc_hi_2[3]);
                                acc_1[4] = __fmaf_rn(part_1[4], w_c_21, acc_1[4]);
                                acc_hi_2[4] = __fmaf_rn(part_hi_1[4], w_c_21, acc_hi_2[4]);
                                acc_1[5] = __fmaf_rn(part_1[5], w_c_21, acc_1[5]);
                                acc_hi_2[5] = __fmaf_rn(part_hi_1[5], w_c_21, acc_hi_2[5]);
                                acc_1[6] = __fmaf_rn(part_1[6], w_c_21, acc_1[6]);
                                acc_hi_2[6] = __fmaf_rn(part_hi_1[6], w_c_21, acc_hi_2[6]);
                                acc_1[7] = __fmaf_rn(part_1[7], w_c_21, acc_1[7]);
                                acc_hi_2[7] = __fmaf_rn(part_hi_1[7], w_c_21, acc_hi_2[7]);
                                acc_1[8] = __fmaf_rn(part_1[8], w_c_21, acc_1[8]);
                                acc_hi_2[8] = __fmaf_rn(part_hi_1[8], w_c_21, acc_hi_2[8]);
                                acc_1[9] = __fmaf_rn(part_1[9], w_c_21, acc_1[9]);
                                acc_hi_2[9] = __fmaf_rn(part_hi_1[9], w_c_21, acc_hi_2[9]);
                                acc_1[10] = __fmaf_rn(part_1[10], w_c_21, acc_1[10]);
                                acc_hi_2[10] = __fmaf_rn(part_hi_1[10], w_c_21, acc_hi_2[10]);
                                acc_1[11] = __fmaf_rn(part_1[11], w_c_21, acc_1[11]);
                                acc_hi_2[11] = __fmaf_rn(part_hi_1[11], w_c_21, acc_hi_2[11]);
                                acc_1[12] = __fmaf_rn(part_1[12], w_c_21, acc_1[12]);
                                acc_hi_2[12] = __fmaf_rn(part_hi_1[12], w_c_21, acc_hi_2[12]);
                                acc_1[13] = __fmaf_rn(part_1[13], w_c_21, acc_1[13]);
                                acc_hi_2[13] = __fmaf_rn(part_hi_1[13], w_c_21, acc_hi_2[13]);
                                acc_1[14] = __fmaf_rn(part_1[14], w_c_21, acc_1[14]);
                                acc_hi_2[14] = __fmaf_rn(part_hi_1[14], w_c_21, acc_hi_2[14]);
                                acc_1[15] = __fmaf_rn(part_1[15], w_c_21, acc_1[15]);
                                acc_hi_2[15] = __fmaf_rn(part_hi_1[15], w_c_21, acc_hi_2[15]);
                            }
                            if (n_chunks_u_0 > 2) {
                                unsigned int crow_18 = (tile_slot0 + 2) * 512 + part_row;
                                float w_c_22 = approx_exp2(part_ml[crow_18 * 2] - row_max_all);
                                unsigned int off_0_18 = crow_18 * 128;
                                {
                                    unsigned _ldv8_135_0;
                                    unsigned _ldv8_135_1;
                                    unsigned _ldv8_135_2;
                                    unsigned _ldv8_135_3;
                                    unsigned _ldv8_135_4;
                                    unsigned _ldv8_135_5;
                                    unsigned _ldv8_135_6;
                                    unsigned _ldv8_135_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_135_0), "=r"(_ldv8_135_1), "=r"(_ldv8_135_2), "=r"(_ldv8_135_3), "=r"(_ldv8_135_4), "=r"(_ldv8_135_5), "=r"(_ldv8_135_6), "=r"(_ldv8_135_7) : "l"((const void*)(part_o + (off_0_18))) : "memory");
                                    part_1[0 + 0] = __uint_as_float(_ldv8_135_0);
                                    part_1[0 + 1] = __uint_as_float(_ldv8_135_1);
                                    part_1[0 + 2] = __uint_as_float(_ldv8_135_2);
                                    part_1[0 + 3] = __uint_as_float(_ldv8_135_3);
                                    part_1[0 + 4] = __uint_as_float(_ldv8_135_4);
                                    part_1[0 + 5] = __uint_as_float(_ldv8_135_5);
                                    part_1[0 + 6] = __uint_as_float(_ldv8_135_6);
                                    part_1[0 + 7] = __uint_as_float(_ldv8_135_7);
                                }
                                {
                                    unsigned _ldv8_136_0;
                                    unsigned _ldv8_136_1;
                                    unsigned _ldv8_136_2;
                                    unsigned _ldv8_136_3;
                                    unsigned _ldv8_136_4;
                                    unsigned _ldv8_136_5;
                                    unsigned _ldv8_136_6;
                                    unsigned _ldv8_136_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_136_0), "=r"(_ldv8_136_1), "=r"(_ldv8_136_2), "=r"(_ldv8_136_3), "=r"(_ldv8_136_4), "=r"(_ldv8_136_5), "=r"(_ldv8_136_6), "=r"(_ldv8_136_7) : "l"((const void*)(part_o + (off_0_18 + 8))) : "memory");
                                    part_1[8 + 0] = __uint_as_float(_ldv8_136_0);
                                    part_1[8 + 1] = __uint_as_float(_ldv8_136_1);
                                    part_1[8 + 2] = __uint_as_float(_ldv8_136_2);
                                    part_1[8 + 3] = __uint_as_float(_ldv8_136_3);
                                    part_1[8 + 4] = __uint_as_float(_ldv8_136_4);
                                    part_1[8 + 5] = __uint_as_float(_ldv8_136_5);
                                    part_1[8 + 6] = __uint_as_float(_ldv8_136_6);
                                    part_1[8 + 7] = __uint_as_float(_ldv8_136_7);
                                }
                                {
                                    unsigned _ldv8_137_0;
                                    unsigned _ldv8_137_1;
                                    unsigned _ldv8_137_2;
                                    unsigned _ldv8_137_3;
                                    unsigned _ldv8_137_4;
                                    unsigned _ldv8_137_5;
                                    unsigned _ldv8_137_6;
                                    unsigned _ldv8_137_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_137_0), "=r"(_ldv8_137_1), "=r"(_ldv8_137_2), "=r"(_ldv8_137_3), "=r"(_ldv8_137_4), "=r"(_ldv8_137_5), "=r"(_ldv8_137_6), "=r"(_ldv8_137_7) : "l"((const void*)(part_o + (off_0_18 + 64))) : "memory");
                                    part_hi_1[0 + 0] = __uint_as_float(_ldv8_137_0);
                                    part_hi_1[0 + 1] = __uint_as_float(_ldv8_137_1);
                                    part_hi_1[0 + 2] = __uint_as_float(_ldv8_137_2);
                                    part_hi_1[0 + 3] = __uint_as_float(_ldv8_137_3);
                                    part_hi_1[0 + 4] = __uint_as_float(_ldv8_137_4);
                                    part_hi_1[0 + 5] = __uint_as_float(_ldv8_137_5);
                                    part_hi_1[0 + 6] = __uint_as_float(_ldv8_137_6);
                                    part_hi_1[0 + 7] = __uint_as_float(_ldv8_137_7);
                                }
                                {
                                    unsigned _ldv8_138_0;
                                    unsigned _ldv8_138_1;
                                    unsigned _ldv8_138_2;
                                    unsigned _ldv8_138_3;
                                    unsigned _ldv8_138_4;
                                    unsigned _ldv8_138_5;
                                    unsigned _ldv8_138_6;
                                    unsigned _ldv8_138_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_138_0), "=r"(_ldv8_138_1), "=r"(_ldv8_138_2), "=r"(_ldv8_138_3), "=r"(_ldv8_138_4), "=r"(_ldv8_138_5), "=r"(_ldv8_138_6), "=r"(_ldv8_138_7) : "l"((const void*)(part_o + (off_0_18 + 64 + 8))) : "memory");
                                    part_hi_1[8 + 0] = __uint_as_float(_ldv8_138_0);
                                    part_hi_1[8 + 1] = __uint_as_float(_ldv8_138_1);
                                    part_hi_1[8 + 2] = __uint_as_float(_ldv8_138_2);
                                    part_hi_1[8 + 3] = __uint_as_float(_ldv8_138_3);
                                    part_hi_1[8 + 4] = __uint_as_float(_ldv8_138_4);
                                    part_hi_1[8 + 5] = __uint_as_float(_ldv8_138_5);
                                    part_hi_1[8 + 6] = __uint_as_float(_ldv8_138_6);
                                    part_hi_1[8 + 7] = __uint_as_float(_ldv8_138_7);
                                }
                                acc_1[0] = __fmaf_rn(part_1[0], w_c_22, acc_1[0]);
                                acc_hi_2[0] = __fmaf_rn(part_hi_1[0], w_c_22, acc_hi_2[0]);
                                acc_1[1] = __fmaf_rn(part_1[1], w_c_22, acc_1[1]);
                                acc_hi_2[1] = __fmaf_rn(part_hi_1[1], w_c_22, acc_hi_2[1]);
                                acc_1[2] = __fmaf_rn(part_1[2], w_c_22, acc_1[2]);
                                acc_hi_2[2] = __fmaf_rn(part_hi_1[2], w_c_22, acc_hi_2[2]);
                                acc_1[3] = __fmaf_rn(part_1[3], w_c_22, acc_1[3]);
                                acc_hi_2[3] = __fmaf_rn(part_hi_1[3], w_c_22, acc_hi_2[3]);
                                acc_1[4] = __fmaf_rn(part_1[4], w_c_22, acc_1[4]);
                                acc_hi_2[4] = __fmaf_rn(part_hi_1[4], w_c_22, acc_hi_2[4]);
                                acc_1[5] = __fmaf_rn(part_1[5], w_c_22, acc_1[5]);
                                acc_hi_2[5] = __fmaf_rn(part_hi_1[5], w_c_22, acc_hi_2[5]);
                                acc_1[6] = __fmaf_rn(part_1[6], w_c_22, acc_1[6]);
                                acc_hi_2[6] = __fmaf_rn(part_hi_1[6], w_c_22, acc_hi_2[6]);
                                acc_1[7] = __fmaf_rn(part_1[7], w_c_22, acc_1[7]);
                                acc_hi_2[7] = __fmaf_rn(part_hi_1[7], w_c_22, acc_hi_2[7]);
                                acc_1[8] = __fmaf_rn(part_1[8], w_c_22, acc_1[8]);
                                acc_hi_2[8] = __fmaf_rn(part_hi_1[8], w_c_22, acc_hi_2[8]);
                                acc_1[9] = __fmaf_rn(part_1[9], w_c_22, acc_1[9]);
                                acc_hi_2[9] = __fmaf_rn(part_hi_1[9], w_c_22, acc_hi_2[9]);
                                acc_1[10] = __fmaf_rn(part_1[10], w_c_22, acc_1[10]);
                                acc_hi_2[10] = __fmaf_rn(part_hi_1[10], w_c_22, acc_hi_2[10]);
                                acc_1[11] = __fmaf_rn(part_1[11], w_c_22, acc_1[11]);
                                acc_hi_2[11] = __fmaf_rn(part_hi_1[11], w_c_22, acc_hi_2[11]);
                                acc_1[12] = __fmaf_rn(part_1[12], w_c_22, acc_1[12]);
                                acc_hi_2[12] = __fmaf_rn(part_hi_1[12], w_c_22, acc_hi_2[12]);
                                acc_1[13] = __fmaf_rn(part_1[13], w_c_22, acc_1[13]);
                                acc_hi_2[13] = __fmaf_rn(part_hi_1[13], w_c_22, acc_hi_2[13]);
                                acc_1[14] = __fmaf_rn(part_1[14], w_c_22, acc_1[14]);
                                acc_hi_2[14] = __fmaf_rn(part_hi_1[14], w_c_22, acc_hi_2[14]);
                                acc_1[15] = __fmaf_rn(part_1[15], w_c_22, acc_1[15]);
                                acc_hi_2[15] = __fmaf_rn(part_hi_1[15], w_c_22, acc_hi_2[15]);
                            }
                            if (n_chunks_u_0 > 3) {
                                unsigned int crow_19 = (tile_slot0 + 3) * 512 + part_row;
                                float w_c_23 = approx_exp2(part_ml[crow_19 * 2] - row_max_all);
                                unsigned int off_0_19 = crow_19 * 128;
                                {
                                    unsigned _ldv8_139_0;
                                    unsigned _ldv8_139_1;
                                    unsigned _ldv8_139_2;
                                    unsigned _ldv8_139_3;
                                    unsigned _ldv8_139_4;
                                    unsigned _ldv8_139_5;
                                    unsigned _ldv8_139_6;
                                    unsigned _ldv8_139_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_139_0), "=r"(_ldv8_139_1), "=r"(_ldv8_139_2), "=r"(_ldv8_139_3), "=r"(_ldv8_139_4), "=r"(_ldv8_139_5), "=r"(_ldv8_139_6), "=r"(_ldv8_139_7) : "l"((const void*)(part_o + (off_0_19))) : "memory");
                                    part_1[0 + 0] = __uint_as_float(_ldv8_139_0);
                                    part_1[0 + 1] = __uint_as_float(_ldv8_139_1);
                                    part_1[0 + 2] = __uint_as_float(_ldv8_139_2);
                                    part_1[0 + 3] = __uint_as_float(_ldv8_139_3);
                                    part_1[0 + 4] = __uint_as_float(_ldv8_139_4);
                                    part_1[0 + 5] = __uint_as_float(_ldv8_139_5);
                                    part_1[0 + 6] = __uint_as_float(_ldv8_139_6);
                                    part_1[0 + 7] = __uint_as_float(_ldv8_139_7);
                                }
                                {
                                    unsigned _ldv8_140_0;
                                    unsigned _ldv8_140_1;
                                    unsigned _ldv8_140_2;
                                    unsigned _ldv8_140_3;
                                    unsigned _ldv8_140_4;
                                    unsigned _ldv8_140_5;
                                    unsigned _ldv8_140_6;
                                    unsigned _ldv8_140_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_140_0), "=r"(_ldv8_140_1), "=r"(_ldv8_140_2), "=r"(_ldv8_140_3), "=r"(_ldv8_140_4), "=r"(_ldv8_140_5), "=r"(_ldv8_140_6), "=r"(_ldv8_140_7) : "l"((const void*)(part_o + (off_0_19 + 8))) : "memory");
                                    part_1[8 + 0] = __uint_as_float(_ldv8_140_0);
                                    part_1[8 + 1] = __uint_as_float(_ldv8_140_1);
                                    part_1[8 + 2] = __uint_as_float(_ldv8_140_2);
                                    part_1[8 + 3] = __uint_as_float(_ldv8_140_3);
                                    part_1[8 + 4] = __uint_as_float(_ldv8_140_4);
                                    part_1[8 + 5] = __uint_as_float(_ldv8_140_5);
                                    part_1[8 + 6] = __uint_as_float(_ldv8_140_6);
                                    part_1[8 + 7] = __uint_as_float(_ldv8_140_7);
                                }
                                {
                                    unsigned _ldv8_141_0;
                                    unsigned _ldv8_141_1;
                                    unsigned _ldv8_141_2;
                                    unsigned _ldv8_141_3;
                                    unsigned _ldv8_141_4;
                                    unsigned _ldv8_141_5;
                                    unsigned _ldv8_141_6;
                                    unsigned _ldv8_141_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_141_0), "=r"(_ldv8_141_1), "=r"(_ldv8_141_2), "=r"(_ldv8_141_3), "=r"(_ldv8_141_4), "=r"(_ldv8_141_5), "=r"(_ldv8_141_6), "=r"(_ldv8_141_7) : "l"((const void*)(part_o + (off_0_19 + 64))) : "memory");
                                    part_hi_1[0 + 0] = __uint_as_float(_ldv8_141_0);
                                    part_hi_1[0 + 1] = __uint_as_float(_ldv8_141_1);
                                    part_hi_1[0 + 2] = __uint_as_float(_ldv8_141_2);
                                    part_hi_1[0 + 3] = __uint_as_float(_ldv8_141_3);
                                    part_hi_1[0 + 4] = __uint_as_float(_ldv8_141_4);
                                    part_hi_1[0 + 5] = __uint_as_float(_ldv8_141_5);
                                    part_hi_1[0 + 6] = __uint_as_float(_ldv8_141_6);
                                    part_hi_1[0 + 7] = __uint_as_float(_ldv8_141_7);
                                }
                                {
                                    unsigned _ldv8_142_0;
                                    unsigned _ldv8_142_1;
                                    unsigned _ldv8_142_2;
                                    unsigned _ldv8_142_3;
                                    unsigned _ldv8_142_4;
                                    unsigned _ldv8_142_5;
                                    unsigned _ldv8_142_6;
                                    unsigned _ldv8_142_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_142_0), "=r"(_ldv8_142_1), "=r"(_ldv8_142_2), "=r"(_ldv8_142_3), "=r"(_ldv8_142_4), "=r"(_ldv8_142_5), "=r"(_ldv8_142_6), "=r"(_ldv8_142_7) : "l"((const void*)(part_o + (off_0_19 + 64 + 8))) : "memory");
                                    part_hi_1[8 + 0] = __uint_as_float(_ldv8_142_0);
                                    part_hi_1[8 + 1] = __uint_as_float(_ldv8_142_1);
                                    part_hi_1[8 + 2] = __uint_as_float(_ldv8_142_2);
                                    part_hi_1[8 + 3] = __uint_as_float(_ldv8_142_3);
                                    part_hi_1[8 + 4] = __uint_as_float(_ldv8_142_4);
                                    part_hi_1[8 + 5] = __uint_as_float(_ldv8_142_5);
                                    part_hi_1[8 + 6] = __uint_as_float(_ldv8_142_6);
                                    part_hi_1[8 + 7] = __uint_as_float(_ldv8_142_7);
                                }
                                acc_1[0] = __fmaf_rn(part_1[0], w_c_23, acc_1[0]);
                                acc_hi_2[0] = __fmaf_rn(part_hi_1[0], w_c_23, acc_hi_2[0]);
                                acc_1[1] = __fmaf_rn(part_1[1], w_c_23, acc_1[1]);
                                acc_hi_2[1] = __fmaf_rn(part_hi_1[1], w_c_23, acc_hi_2[1]);
                                acc_1[2] = __fmaf_rn(part_1[2], w_c_23, acc_1[2]);
                                acc_hi_2[2] = __fmaf_rn(part_hi_1[2], w_c_23, acc_hi_2[2]);
                                acc_1[3] = __fmaf_rn(part_1[3], w_c_23, acc_1[3]);
                                acc_hi_2[3] = __fmaf_rn(part_hi_1[3], w_c_23, acc_hi_2[3]);
                                acc_1[4] = __fmaf_rn(part_1[4], w_c_23, acc_1[4]);
                                acc_hi_2[4] = __fmaf_rn(part_hi_1[4], w_c_23, acc_hi_2[4]);
                                acc_1[5] = __fmaf_rn(part_1[5], w_c_23, acc_1[5]);
                                acc_hi_2[5] = __fmaf_rn(part_hi_1[5], w_c_23, acc_hi_2[5]);
                                acc_1[6] = __fmaf_rn(part_1[6], w_c_23, acc_1[6]);
                                acc_hi_2[6] = __fmaf_rn(part_hi_1[6], w_c_23, acc_hi_2[6]);
                                acc_1[7] = __fmaf_rn(part_1[7], w_c_23, acc_1[7]);
                                acc_hi_2[7] = __fmaf_rn(part_hi_1[7], w_c_23, acc_hi_2[7]);
                                acc_1[8] = __fmaf_rn(part_1[8], w_c_23, acc_1[8]);
                                acc_hi_2[8] = __fmaf_rn(part_hi_1[8], w_c_23, acc_hi_2[8]);
                                acc_1[9] = __fmaf_rn(part_1[9], w_c_23, acc_1[9]);
                                acc_hi_2[9] = __fmaf_rn(part_hi_1[9], w_c_23, acc_hi_2[9]);
                                acc_1[10] = __fmaf_rn(part_1[10], w_c_23, acc_1[10]);
                                acc_hi_2[10] = __fmaf_rn(part_hi_1[10], w_c_23, acc_hi_2[10]);
                                acc_1[11] = __fmaf_rn(part_1[11], w_c_23, acc_1[11]);
                                acc_hi_2[11] = __fmaf_rn(part_hi_1[11], w_c_23, acc_hi_2[11]);
                                acc_1[12] = __fmaf_rn(part_1[12], w_c_23, acc_1[12]);
                                acc_hi_2[12] = __fmaf_rn(part_hi_1[12], w_c_23, acc_hi_2[12]);
                                acc_1[13] = __fmaf_rn(part_1[13], w_c_23, acc_1[13]);
                                acc_hi_2[13] = __fmaf_rn(part_hi_1[13], w_c_23, acc_hi_2[13]);
                                acc_1[14] = __fmaf_rn(part_1[14], w_c_23, acc_1[14]);
                                acc_hi_2[14] = __fmaf_rn(part_hi_1[14], w_c_23, acc_hi_2[14]);
                                acc_1[15] = __fmaf_rn(part_1[15], w_c_23, acc_1[15]);
                                acc_hi_2[15] = __fmaf_rn(part_hi_1[15], w_c_23, acc_hi_2[15]);
                            }
                            const float2 _scale2_143 = {final_scale, final_scale};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(acc_1)[_ls], _scale2_143);
                            unsigned int acc_bf16_1[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_1[_lp*2 + 0], acc_1[_lp*2+1 + 0]));
                                acc_bf16_1[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o10_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 ^ ((warp % 4 * 32 + lane) * 128 >> 7 & 7) << 4))), "r"(acc_bf16_1[0]), "r"(acc_bf16_1[1]), "r"(acc_bf16_1[2]), "r"(acc_bf16_1[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o10_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + 16 ^ ((warp % 4 * 32 + lane) * 128 + 16 >> 7 & 7) << 4))), "r"(acc_bf16_1[4]), "r"(acc_bf16_1[5]), "r"(acc_bf16_1[6]), "r"(acc_bf16_1[7]) : "memory");
                            const float2 _scale2_144 = {final_scale, final_scale};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(acc_hi_2)[_ls], _scale2_144);
                            unsigned int acc_hi_bf16_1[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_hi_2[_lp*2 + 0], acc_hi_2[_lp*2+1 + 0]));
                                acc_hi_bf16_1[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o11_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 ^ ((warp % 4 * 32 + lane) * 128 >> 7 & 7) << 4))), "r"(acc_hi_bf16_1[0]), "r"(acc_hi_bf16_1[1]), "r"(acc_hi_bf16_1[2]), "r"(acc_hi_bf16_1[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o11_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + 16 ^ ((warp % 4 * 32 + lane) * 128 + 16 >> 7 & 7) << 4))), "r"(acc_hi_bf16_1[4]), "r"(acc_hi_bf16_1[5]), "r"(acc_hi_bf16_1[6]), "r"(acc_hi_bf16_1[7]) : "memory");
                            float acc_0_1[16];
                            float acc_hi_1_1[16];
                            acc_0_1[0] = 0.0f;
                            acc_hi_1_1[0] = 0.0f;
                            acc_0_1[1] = 0.0f;
                            acc_hi_1_1[1] = 0.0f;
                            acc_0_1[2] = 0.0f;
                            acc_hi_1_1[2] = 0.0f;
                            acc_0_1[3] = 0.0f;
                            acc_hi_1_1[3] = 0.0f;
                            acc_0_1[4] = 0.0f;
                            acc_hi_1_1[4] = 0.0f;
                            acc_0_1[5] = 0.0f;
                            acc_hi_1_1[5] = 0.0f;
                            acc_0_1[6] = 0.0f;
                            acc_hi_1_1[6] = 0.0f;
                            acc_0_1[7] = 0.0f;
                            acc_hi_1_1[7] = 0.0f;
                            acc_0_1[8] = 0.0f;
                            acc_hi_1_1[8] = 0.0f;
                            acc_0_1[9] = 0.0f;
                            acc_hi_1_1[9] = 0.0f;
                            acc_0_1[10] = 0.0f;
                            acc_hi_1_1[10] = 0.0f;
                            acc_0_1[11] = 0.0f;
                            acc_hi_1_1[11] = 0.0f;
                            acc_0_1[12] = 0.0f;
                            acc_hi_1_1[12] = 0.0f;
                            acc_0_1[13] = 0.0f;
                            acc_hi_1_1[13] = 0.0f;
                            acc_0_1[14] = 0.0f;
                            acc_hi_1_1[14] = 0.0f;
                            acc_0_1[15] = 0.0f;
                            acc_hi_1_1[15] = 0.0f;
                            float part_2_1[16];
                            float part_hi_3_1[16];
                            if (n_chunks_u_0 > 0) {
                                unsigned int crow_20 = tile_slot0 * 512 + part_row;
                                float w_c_24 = approx_exp2(part_ml[crow_20 * 2] - row_max_all);
                                unsigned int off_0_20 = crow_20 * 128 + 16;
                                {
                                    unsigned _ldv8_145_0;
                                    unsigned _ldv8_145_1;
                                    unsigned _ldv8_145_2;
                                    unsigned _ldv8_145_3;
                                    unsigned _ldv8_145_4;
                                    unsigned _ldv8_145_5;
                                    unsigned _ldv8_145_6;
                                    unsigned _ldv8_145_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_145_0), "=r"(_ldv8_145_1), "=r"(_ldv8_145_2), "=r"(_ldv8_145_3), "=r"(_ldv8_145_4), "=r"(_ldv8_145_5), "=r"(_ldv8_145_6), "=r"(_ldv8_145_7) : "l"((const void*)(part_o + (off_0_20))) : "memory");
                                    part_2_1[0 + 0] = __uint_as_float(_ldv8_145_0);
                                    part_2_1[0 + 1] = __uint_as_float(_ldv8_145_1);
                                    part_2_1[0 + 2] = __uint_as_float(_ldv8_145_2);
                                    part_2_1[0 + 3] = __uint_as_float(_ldv8_145_3);
                                    part_2_1[0 + 4] = __uint_as_float(_ldv8_145_4);
                                    part_2_1[0 + 5] = __uint_as_float(_ldv8_145_5);
                                    part_2_1[0 + 6] = __uint_as_float(_ldv8_145_6);
                                    part_2_1[0 + 7] = __uint_as_float(_ldv8_145_7);
                                }
                                {
                                    unsigned _ldv8_146_0;
                                    unsigned _ldv8_146_1;
                                    unsigned _ldv8_146_2;
                                    unsigned _ldv8_146_3;
                                    unsigned _ldv8_146_4;
                                    unsigned _ldv8_146_5;
                                    unsigned _ldv8_146_6;
                                    unsigned _ldv8_146_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_146_0), "=r"(_ldv8_146_1), "=r"(_ldv8_146_2), "=r"(_ldv8_146_3), "=r"(_ldv8_146_4), "=r"(_ldv8_146_5), "=r"(_ldv8_146_6), "=r"(_ldv8_146_7) : "l"((const void*)(part_o + (off_0_20 + 8))) : "memory");
                                    part_2_1[8 + 0] = __uint_as_float(_ldv8_146_0);
                                    part_2_1[8 + 1] = __uint_as_float(_ldv8_146_1);
                                    part_2_1[8 + 2] = __uint_as_float(_ldv8_146_2);
                                    part_2_1[8 + 3] = __uint_as_float(_ldv8_146_3);
                                    part_2_1[8 + 4] = __uint_as_float(_ldv8_146_4);
                                    part_2_1[8 + 5] = __uint_as_float(_ldv8_146_5);
                                    part_2_1[8 + 6] = __uint_as_float(_ldv8_146_6);
                                    part_2_1[8 + 7] = __uint_as_float(_ldv8_146_7);
                                }
                                {
                                    unsigned _ldv8_147_0;
                                    unsigned _ldv8_147_1;
                                    unsigned _ldv8_147_2;
                                    unsigned _ldv8_147_3;
                                    unsigned _ldv8_147_4;
                                    unsigned _ldv8_147_5;
                                    unsigned _ldv8_147_6;
                                    unsigned _ldv8_147_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_147_0), "=r"(_ldv8_147_1), "=r"(_ldv8_147_2), "=r"(_ldv8_147_3), "=r"(_ldv8_147_4), "=r"(_ldv8_147_5), "=r"(_ldv8_147_6), "=r"(_ldv8_147_7) : "l"((const void*)(part_o + (off_0_20 + 64))) : "memory");
                                    part_hi_3_1[0 + 0] = __uint_as_float(_ldv8_147_0);
                                    part_hi_3_1[0 + 1] = __uint_as_float(_ldv8_147_1);
                                    part_hi_3_1[0 + 2] = __uint_as_float(_ldv8_147_2);
                                    part_hi_3_1[0 + 3] = __uint_as_float(_ldv8_147_3);
                                    part_hi_3_1[0 + 4] = __uint_as_float(_ldv8_147_4);
                                    part_hi_3_1[0 + 5] = __uint_as_float(_ldv8_147_5);
                                    part_hi_3_1[0 + 6] = __uint_as_float(_ldv8_147_6);
                                    part_hi_3_1[0 + 7] = __uint_as_float(_ldv8_147_7);
                                }
                                {
                                    unsigned _ldv8_148_0;
                                    unsigned _ldv8_148_1;
                                    unsigned _ldv8_148_2;
                                    unsigned _ldv8_148_3;
                                    unsigned _ldv8_148_4;
                                    unsigned _ldv8_148_5;
                                    unsigned _ldv8_148_6;
                                    unsigned _ldv8_148_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_148_0), "=r"(_ldv8_148_1), "=r"(_ldv8_148_2), "=r"(_ldv8_148_3), "=r"(_ldv8_148_4), "=r"(_ldv8_148_5), "=r"(_ldv8_148_6), "=r"(_ldv8_148_7) : "l"((const void*)(part_o + (off_0_20 + 64 + 8))) : "memory");
                                    part_hi_3_1[8 + 0] = __uint_as_float(_ldv8_148_0);
                                    part_hi_3_1[8 + 1] = __uint_as_float(_ldv8_148_1);
                                    part_hi_3_1[8 + 2] = __uint_as_float(_ldv8_148_2);
                                    part_hi_3_1[8 + 3] = __uint_as_float(_ldv8_148_3);
                                    part_hi_3_1[8 + 4] = __uint_as_float(_ldv8_148_4);
                                    part_hi_3_1[8 + 5] = __uint_as_float(_ldv8_148_5);
                                    part_hi_3_1[8 + 6] = __uint_as_float(_ldv8_148_6);
                                    part_hi_3_1[8 + 7] = __uint_as_float(_ldv8_148_7);
                                }
                                acc_0_1[0] = __fmaf_rn(part_2_1[0], w_c_24, acc_0_1[0]);
                                acc_hi_1_1[0] = __fmaf_rn(part_hi_3_1[0], w_c_24, acc_hi_1_1[0]);
                                acc_0_1[1] = __fmaf_rn(part_2_1[1], w_c_24, acc_0_1[1]);
                                acc_hi_1_1[1] = __fmaf_rn(part_hi_3_1[1], w_c_24, acc_hi_1_1[1]);
                                acc_0_1[2] = __fmaf_rn(part_2_1[2], w_c_24, acc_0_1[2]);
                                acc_hi_1_1[2] = __fmaf_rn(part_hi_3_1[2], w_c_24, acc_hi_1_1[2]);
                                acc_0_1[3] = __fmaf_rn(part_2_1[3], w_c_24, acc_0_1[3]);
                                acc_hi_1_1[3] = __fmaf_rn(part_hi_3_1[3], w_c_24, acc_hi_1_1[3]);
                                acc_0_1[4] = __fmaf_rn(part_2_1[4], w_c_24, acc_0_1[4]);
                                acc_hi_1_1[4] = __fmaf_rn(part_hi_3_1[4], w_c_24, acc_hi_1_1[4]);
                                acc_0_1[5] = __fmaf_rn(part_2_1[5], w_c_24, acc_0_1[5]);
                                acc_hi_1_1[5] = __fmaf_rn(part_hi_3_1[5], w_c_24, acc_hi_1_1[5]);
                                acc_0_1[6] = __fmaf_rn(part_2_1[6], w_c_24, acc_0_1[6]);
                                acc_hi_1_1[6] = __fmaf_rn(part_hi_3_1[6], w_c_24, acc_hi_1_1[6]);
                                acc_0_1[7] = __fmaf_rn(part_2_1[7], w_c_24, acc_0_1[7]);
                                acc_hi_1_1[7] = __fmaf_rn(part_hi_3_1[7], w_c_24, acc_hi_1_1[7]);
                                acc_0_1[8] = __fmaf_rn(part_2_1[8], w_c_24, acc_0_1[8]);
                                acc_hi_1_1[8] = __fmaf_rn(part_hi_3_1[8], w_c_24, acc_hi_1_1[8]);
                                acc_0_1[9] = __fmaf_rn(part_2_1[9], w_c_24, acc_0_1[9]);
                                acc_hi_1_1[9] = __fmaf_rn(part_hi_3_1[9], w_c_24, acc_hi_1_1[9]);
                                acc_0_1[10] = __fmaf_rn(part_2_1[10], w_c_24, acc_0_1[10]);
                                acc_hi_1_1[10] = __fmaf_rn(part_hi_3_1[10], w_c_24, acc_hi_1_1[10]);
                                acc_0_1[11] = __fmaf_rn(part_2_1[11], w_c_24, acc_0_1[11]);
                                acc_hi_1_1[11] = __fmaf_rn(part_hi_3_1[11], w_c_24, acc_hi_1_1[11]);
                                acc_0_1[12] = __fmaf_rn(part_2_1[12], w_c_24, acc_0_1[12]);
                                acc_hi_1_1[12] = __fmaf_rn(part_hi_3_1[12], w_c_24, acc_hi_1_1[12]);
                                acc_0_1[13] = __fmaf_rn(part_2_1[13], w_c_24, acc_0_1[13]);
                                acc_hi_1_1[13] = __fmaf_rn(part_hi_3_1[13], w_c_24, acc_hi_1_1[13]);
                                acc_0_1[14] = __fmaf_rn(part_2_1[14], w_c_24, acc_0_1[14]);
                                acc_hi_1_1[14] = __fmaf_rn(part_hi_3_1[14], w_c_24, acc_hi_1_1[14]);
                                acc_0_1[15] = __fmaf_rn(part_2_1[15], w_c_24, acc_0_1[15]);
                                acc_hi_1_1[15] = __fmaf_rn(part_hi_3_1[15], w_c_24, acc_hi_1_1[15]);
                            }
                            if (n_chunks_u_0 > 1) {
                                unsigned int crow_21 = (tile_slot0 + 1) * 512 + part_row;
                                float w_c_25 = approx_exp2(part_ml[crow_21 * 2] - row_max_all);
                                unsigned int off_0_21 = crow_21 * 128 + 16;
                                {
                                    unsigned _ldv8_149_0;
                                    unsigned _ldv8_149_1;
                                    unsigned _ldv8_149_2;
                                    unsigned _ldv8_149_3;
                                    unsigned _ldv8_149_4;
                                    unsigned _ldv8_149_5;
                                    unsigned _ldv8_149_6;
                                    unsigned _ldv8_149_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_149_0), "=r"(_ldv8_149_1), "=r"(_ldv8_149_2), "=r"(_ldv8_149_3), "=r"(_ldv8_149_4), "=r"(_ldv8_149_5), "=r"(_ldv8_149_6), "=r"(_ldv8_149_7) : "l"((const void*)(part_o + (off_0_21))) : "memory");
                                    part_2_1[0 + 0] = __uint_as_float(_ldv8_149_0);
                                    part_2_1[0 + 1] = __uint_as_float(_ldv8_149_1);
                                    part_2_1[0 + 2] = __uint_as_float(_ldv8_149_2);
                                    part_2_1[0 + 3] = __uint_as_float(_ldv8_149_3);
                                    part_2_1[0 + 4] = __uint_as_float(_ldv8_149_4);
                                    part_2_1[0 + 5] = __uint_as_float(_ldv8_149_5);
                                    part_2_1[0 + 6] = __uint_as_float(_ldv8_149_6);
                                    part_2_1[0 + 7] = __uint_as_float(_ldv8_149_7);
                                }
                                {
                                    unsigned _ldv8_150_0;
                                    unsigned _ldv8_150_1;
                                    unsigned _ldv8_150_2;
                                    unsigned _ldv8_150_3;
                                    unsigned _ldv8_150_4;
                                    unsigned _ldv8_150_5;
                                    unsigned _ldv8_150_6;
                                    unsigned _ldv8_150_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_150_0), "=r"(_ldv8_150_1), "=r"(_ldv8_150_2), "=r"(_ldv8_150_3), "=r"(_ldv8_150_4), "=r"(_ldv8_150_5), "=r"(_ldv8_150_6), "=r"(_ldv8_150_7) : "l"((const void*)(part_o + (off_0_21 + 8))) : "memory");
                                    part_2_1[8 + 0] = __uint_as_float(_ldv8_150_0);
                                    part_2_1[8 + 1] = __uint_as_float(_ldv8_150_1);
                                    part_2_1[8 + 2] = __uint_as_float(_ldv8_150_2);
                                    part_2_1[8 + 3] = __uint_as_float(_ldv8_150_3);
                                    part_2_1[8 + 4] = __uint_as_float(_ldv8_150_4);
                                    part_2_1[8 + 5] = __uint_as_float(_ldv8_150_5);
                                    part_2_1[8 + 6] = __uint_as_float(_ldv8_150_6);
                                    part_2_1[8 + 7] = __uint_as_float(_ldv8_150_7);
                                }
                                {
                                    unsigned _ldv8_151_0;
                                    unsigned _ldv8_151_1;
                                    unsigned _ldv8_151_2;
                                    unsigned _ldv8_151_3;
                                    unsigned _ldv8_151_4;
                                    unsigned _ldv8_151_5;
                                    unsigned _ldv8_151_6;
                                    unsigned _ldv8_151_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_151_0), "=r"(_ldv8_151_1), "=r"(_ldv8_151_2), "=r"(_ldv8_151_3), "=r"(_ldv8_151_4), "=r"(_ldv8_151_5), "=r"(_ldv8_151_6), "=r"(_ldv8_151_7) : "l"((const void*)(part_o + (off_0_21 + 64))) : "memory");
                                    part_hi_3_1[0 + 0] = __uint_as_float(_ldv8_151_0);
                                    part_hi_3_1[0 + 1] = __uint_as_float(_ldv8_151_1);
                                    part_hi_3_1[0 + 2] = __uint_as_float(_ldv8_151_2);
                                    part_hi_3_1[0 + 3] = __uint_as_float(_ldv8_151_3);
                                    part_hi_3_1[0 + 4] = __uint_as_float(_ldv8_151_4);
                                    part_hi_3_1[0 + 5] = __uint_as_float(_ldv8_151_5);
                                    part_hi_3_1[0 + 6] = __uint_as_float(_ldv8_151_6);
                                    part_hi_3_1[0 + 7] = __uint_as_float(_ldv8_151_7);
                                }
                                {
                                    unsigned _ldv8_152_0;
                                    unsigned _ldv8_152_1;
                                    unsigned _ldv8_152_2;
                                    unsigned _ldv8_152_3;
                                    unsigned _ldv8_152_4;
                                    unsigned _ldv8_152_5;
                                    unsigned _ldv8_152_6;
                                    unsigned _ldv8_152_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_152_0), "=r"(_ldv8_152_1), "=r"(_ldv8_152_2), "=r"(_ldv8_152_3), "=r"(_ldv8_152_4), "=r"(_ldv8_152_5), "=r"(_ldv8_152_6), "=r"(_ldv8_152_7) : "l"((const void*)(part_o + (off_0_21 + 64 + 8))) : "memory");
                                    part_hi_3_1[8 + 0] = __uint_as_float(_ldv8_152_0);
                                    part_hi_3_1[8 + 1] = __uint_as_float(_ldv8_152_1);
                                    part_hi_3_1[8 + 2] = __uint_as_float(_ldv8_152_2);
                                    part_hi_3_1[8 + 3] = __uint_as_float(_ldv8_152_3);
                                    part_hi_3_1[8 + 4] = __uint_as_float(_ldv8_152_4);
                                    part_hi_3_1[8 + 5] = __uint_as_float(_ldv8_152_5);
                                    part_hi_3_1[8 + 6] = __uint_as_float(_ldv8_152_6);
                                    part_hi_3_1[8 + 7] = __uint_as_float(_ldv8_152_7);
                                }
                                acc_0_1[0] = __fmaf_rn(part_2_1[0], w_c_25, acc_0_1[0]);
                                acc_hi_1_1[0] = __fmaf_rn(part_hi_3_1[0], w_c_25, acc_hi_1_1[0]);
                                acc_0_1[1] = __fmaf_rn(part_2_1[1], w_c_25, acc_0_1[1]);
                                acc_hi_1_1[1] = __fmaf_rn(part_hi_3_1[1], w_c_25, acc_hi_1_1[1]);
                                acc_0_1[2] = __fmaf_rn(part_2_1[2], w_c_25, acc_0_1[2]);
                                acc_hi_1_1[2] = __fmaf_rn(part_hi_3_1[2], w_c_25, acc_hi_1_1[2]);
                                acc_0_1[3] = __fmaf_rn(part_2_1[3], w_c_25, acc_0_1[3]);
                                acc_hi_1_1[3] = __fmaf_rn(part_hi_3_1[3], w_c_25, acc_hi_1_1[3]);
                                acc_0_1[4] = __fmaf_rn(part_2_1[4], w_c_25, acc_0_1[4]);
                                acc_hi_1_1[4] = __fmaf_rn(part_hi_3_1[4], w_c_25, acc_hi_1_1[4]);
                                acc_0_1[5] = __fmaf_rn(part_2_1[5], w_c_25, acc_0_1[5]);
                                acc_hi_1_1[5] = __fmaf_rn(part_hi_3_1[5], w_c_25, acc_hi_1_1[5]);
                                acc_0_1[6] = __fmaf_rn(part_2_1[6], w_c_25, acc_0_1[6]);
                                acc_hi_1_1[6] = __fmaf_rn(part_hi_3_1[6], w_c_25, acc_hi_1_1[6]);
                                acc_0_1[7] = __fmaf_rn(part_2_1[7], w_c_25, acc_0_1[7]);
                                acc_hi_1_1[7] = __fmaf_rn(part_hi_3_1[7], w_c_25, acc_hi_1_1[7]);
                                acc_0_1[8] = __fmaf_rn(part_2_1[8], w_c_25, acc_0_1[8]);
                                acc_hi_1_1[8] = __fmaf_rn(part_hi_3_1[8], w_c_25, acc_hi_1_1[8]);
                                acc_0_1[9] = __fmaf_rn(part_2_1[9], w_c_25, acc_0_1[9]);
                                acc_hi_1_1[9] = __fmaf_rn(part_hi_3_1[9], w_c_25, acc_hi_1_1[9]);
                                acc_0_1[10] = __fmaf_rn(part_2_1[10], w_c_25, acc_0_1[10]);
                                acc_hi_1_1[10] = __fmaf_rn(part_hi_3_1[10], w_c_25, acc_hi_1_1[10]);
                                acc_0_1[11] = __fmaf_rn(part_2_1[11], w_c_25, acc_0_1[11]);
                                acc_hi_1_1[11] = __fmaf_rn(part_hi_3_1[11], w_c_25, acc_hi_1_1[11]);
                                acc_0_1[12] = __fmaf_rn(part_2_1[12], w_c_25, acc_0_1[12]);
                                acc_hi_1_1[12] = __fmaf_rn(part_hi_3_1[12], w_c_25, acc_hi_1_1[12]);
                                acc_0_1[13] = __fmaf_rn(part_2_1[13], w_c_25, acc_0_1[13]);
                                acc_hi_1_1[13] = __fmaf_rn(part_hi_3_1[13], w_c_25, acc_hi_1_1[13]);
                                acc_0_1[14] = __fmaf_rn(part_2_1[14], w_c_25, acc_0_1[14]);
                                acc_hi_1_1[14] = __fmaf_rn(part_hi_3_1[14], w_c_25, acc_hi_1_1[14]);
                                acc_0_1[15] = __fmaf_rn(part_2_1[15], w_c_25, acc_0_1[15]);
                                acc_hi_1_1[15] = __fmaf_rn(part_hi_3_1[15], w_c_25, acc_hi_1_1[15]);
                            }
                            if (n_chunks_u_0 > 2) {
                                unsigned int crow_22 = (tile_slot0 + 2) * 512 + part_row;
                                float w_c_26 = approx_exp2(part_ml[crow_22 * 2] - row_max_all);
                                unsigned int off_0_22 = crow_22 * 128 + 16;
                                {
                                    unsigned _ldv8_153_0;
                                    unsigned _ldv8_153_1;
                                    unsigned _ldv8_153_2;
                                    unsigned _ldv8_153_3;
                                    unsigned _ldv8_153_4;
                                    unsigned _ldv8_153_5;
                                    unsigned _ldv8_153_6;
                                    unsigned _ldv8_153_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_153_0), "=r"(_ldv8_153_1), "=r"(_ldv8_153_2), "=r"(_ldv8_153_3), "=r"(_ldv8_153_4), "=r"(_ldv8_153_5), "=r"(_ldv8_153_6), "=r"(_ldv8_153_7) : "l"((const void*)(part_o + (off_0_22))) : "memory");
                                    part_2_1[0 + 0] = __uint_as_float(_ldv8_153_0);
                                    part_2_1[0 + 1] = __uint_as_float(_ldv8_153_1);
                                    part_2_1[0 + 2] = __uint_as_float(_ldv8_153_2);
                                    part_2_1[0 + 3] = __uint_as_float(_ldv8_153_3);
                                    part_2_1[0 + 4] = __uint_as_float(_ldv8_153_4);
                                    part_2_1[0 + 5] = __uint_as_float(_ldv8_153_5);
                                    part_2_1[0 + 6] = __uint_as_float(_ldv8_153_6);
                                    part_2_1[0 + 7] = __uint_as_float(_ldv8_153_7);
                                }
                                {
                                    unsigned _ldv8_154_0;
                                    unsigned _ldv8_154_1;
                                    unsigned _ldv8_154_2;
                                    unsigned _ldv8_154_3;
                                    unsigned _ldv8_154_4;
                                    unsigned _ldv8_154_5;
                                    unsigned _ldv8_154_6;
                                    unsigned _ldv8_154_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_154_0), "=r"(_ldv8_154_1), "=r"(_ldv8_154_2), "=r"(_ldv8_154_3), "=r"(_ldv8_154_4), "=r"(_ldv8_154_5), "=r"(_ldv8_154_6), "=r"(_ldv8_154_7) : "l"((const void*)(part_o + (off_0_22 + 8))) : "memory");
                                    part_2_1[8 + 0] = __uint_as_float(_ldv8_154_0);
                                    part_2_1[8 + 1] = __uint_as_float(_ldv8_154_1);
                                    part_2_1[8 + 2] = __uint_as_float(_ldv8_154_2);
                                    part_2_1[8 + 3] = __uint_as_float(_ldv8_154_3);
                                    part_2_1[8 + 4] = __uint_as_float(_ldv8_154_4);
                                    part_2_1[8 + 5] = __uint_as_float(_ldv8_154_5);
                                    part_2_1[8 + 6] = __uint_as_float(_ldv8_154_6);
                                    part_2_1[8 + 7] = __uint_as_float(_ldv8_154_7);
                                }
                                {
                                    unsigned _ldv8_155_0;
                                    unsigned _ldv8_155_1;
                                    unsigned _ldv8_155_2;
                                    unsigned _ldv8_155_3;
                                    unsigned _ldv8_155_4;
                                    unsigned _ldv8_155_5;
                                    unsigned _ldv8_155_6;
                                    unsigned _ldv8_155_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_155_0), "=r"(_ldv8_155_1), "=r"(_ldv8_155_2), "=r"(_ldv8_155_3), "=r"(_ldv8_155_4), "=r"(_ldv8_155_5), "=r"(_ldv8_155_6), "=r"(_ldv8_155_7) : "l"((const void*)(part_o + (off_0_22 + 64))) : "memory");
                                    part_hi_3_1[0 + 0] = __uint_as_float(_ldv8_155_0);
                                    part_hi_3_1[0 + 1] = __uint_as_float(_ldv8_155_1);
                                    part_hi_3_1[0 + 2] = __uint_as_float(_ldv8_155_2);
                                    part_hi_3_1[0 + 3] = __uint_as_float(_ldv8_155_3);
                                    part_hi_3_1[0 + 4] = __uint_as_float(_ldv8_155_4);
                                    part_hi_3_1[0 + 5] = __uint_as_float(_ldv8_155_5);
                                    part_hi_3_1[0 + 6] = __uint_as_float(_ldv8_155_6);
                                    part_hi_3_1[0 + 7] = __uint_as_float(_ldv8_155_7);
                                }
                                {
                                    unsigned _ldv8_156_0;
                                    unsigned _ldv8_156_1;
                                    unsigned _ldv8_156_2;
                                    unsigned _ldv8_156_3;
                                    unsigned _ldv8_156_4;
                                    unsigned _ldv8_156_5;
                                    unsigned _ldv8_156_6;
                                    unsigned _ldv8_156_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_156_0), "=r"(_ldv8_156_1), "=r"(_ldv8_156_2), "=r"(_ldv8_156_3), "=r"(_ldv8_156_4), "=r"(_ldv8_156_5), "=r"(_ldv8_156_6), "=r"(_ldv8_156_7) : "l"((const void*)(part_o + (off_0_22 + 64 + 8))) : "memory");
                                    part_hi_3_1[8 + 0] = __uint_as_float(_ldv8_156_0);
                                    part_hi_3_1[8 + 1] = __uint_as_float(_ldv8_156_1);
                                    part_hi_3_1[8 + 2] = __uint_as_float(_ldv8_156_2);
                                    part_hi_3_1[8 + 3] = __uint_as_float(_ldv8_156_3);
                                    part_hi_3_1[8 + 4] = __uint_as_float(_ldv8_156_4);
                                    part_hi_3_1[8 + 5] = __uint_as_float(_ldv8_156_5);
                                    part_hi_3_1[8 + 6] = __uint_as_float(_ldv8_156_6);
                                    part_hi_3_1[8 + 7] = __uint_as_float(_ldv8_156_7);
                                }
                                acc_0_1[0] = __fmaf_rn(part_2_1[0], w_c_26, acc_0_1[0]);
                                acc_hi_1_1[0] = __fmaf_rn(part_hi_3_1[0], w_c_26, acc_hi_1_1[0]);
                                acc_0_1[1] = __fmaf_rn(part_2_1[1], w_c_26, acc_0_1[1]);
                                acc_hi_1_1[1] = __fmaf_rn(part_hi_3_1[1], w_c_26, acc_hi_1_1[1]);
                                acc_0_1[2] = __fmaf_rn(part_2_1[2], w_c_26, acc_0_1[2]);
                                acc_hi_1_1[2] = __fmaf_rn(part_hi_3_1[2], w_c_26, acc_hi_1_1[2]);
                                acc_0_1[3] = __fmaf_rn(part_2_1[3], w_c_26, acc_0_1[3]);
                                acc_hi_1_1[3] = __fmaf_rn(part_hi_3_1[3], w_c_26, acc_hi_1_1[3]);
                                acc_0_1[4] = __fmaf_rn(part_2_1[4], w_c_26, acc_0_1[4]);
                                acc_hi_1_1[4] = __fmaf_rn(part_hi_3_1[4], w_c_26, acc_hi_1_1[4]);
                                acc_0_1[5] = __fmaf_rn(part_2_1[5], w_c_26, acc_0_1[5]);
                                acc_hi_1_1[5] = __fmaf_rn(part_hi_3_1[5], w_c_26, acc_hi_1_1[5]);
                                acc_0_1[6] = __fmaf_rn(part_2_1[6], w_c_26, acc_0_1[6]);
                                acc_hi_1_1[6] = __fmaf_rn(part_hi_3_1[6], w_c_26, acc_hi_1_1[6]);
                                acc_0_1[7] = __fmaf_rn(part_2_1[7], w_c_26, acc_0_1[7]);
                                acc_hi_1_1[7] = __fmaf_rn(part_hi_3_1[7], w_c_26, acc_hi_1_1[7]);
                                acc_0_1[8] = __fmaf_rn(part_2_1[8], w_c_26, acc_0_1[8]);
                                acc_hi_1_1[8] = __fmaf_rn(part_hi_3_1[8], w_c_26, acc_hi_1_1[8]);
                                acc_0_1[9] = __fmaf_rn(part_2_1[9], w_c_26, acc_0_1[9]);
                                acc_hi_1_1[9] = __fmaf_rn(part_hi_3_1[9], w_c_26, acc_hi_1_1[9]);
                                acc_0_1[10] = __fmaf_rn(part_2_1[10], w_c_26, acc_0_1[10]);
                                acc_hi_1_1[10] = __fmaf_rn(part_hi_3_1[10], w_c_26, acc_hi_1_1[10]);
                                acc_0_1[11] = __fmaf_rn(part_2_1[11], w_c_26, acc_0_1[11]);
                                acc_hi_1_1[11] = __fmaf_rn(part_hi_3_1[11], w_c_26, acc_hi_1_1[11]);
                                acc_0_1[12] = __fmaf_rn(part_2_1[12], w_c_26, acc_0_1[12]);
                                acc_hi_1_1[12] = __fmaf_rn(part_hi_3_1[12], w_c_26, acc_hi_1_1[12]);
                                acc_0_1[13] = __fmaf_rn(part_2_1[13], w_c_26, acc_0_1[13]);
                                acc_hi_1_1[13] = __fmaf_rn(part_hi_3_1[13], w_c_26, acc_hi_1_1[13]);
                                acc_0_1[14] = __fmaf_rn(part_2_1[14], w_c_26, acc_0_1[14]);
                                acc_hi_1_1[14] = __fmaf_rn(part_hi_3_1[14], w_c_26, acc_hi_1_1[14]);
                                acc_0_1[15] = __fmaf_rn(part_2_1[15], w_c_26, acc_0_1[15]);
                                acc_hi_1_1[15] = __fmaf_rn(part_hi_3_1[15], w_c_26, acc_hi_1_1[15]);
                            }
                            if (n_chunks_u_0 > 3) {
                                unsigned int crow_23 = (tile_slot0 + 3) * 512 + part_row;
                                float w_c_27 = approx_exp2(part_ml[crow_23 * 2] - row_max_all);
                                unsigned int off_0_23 = crow_23 * 128 + 16;
                                {
                                    unsigned _ldv8_157_0;
                                    unsigned _ldv8_157_1;
                                    unsigned _ldv8_157_2;
                                    unsigned _ldv8_157_3;
                                    unsigned _ldv8_157_4;
                                    unsigned _ldv8_157_5;
                                    unsigned _ldv8_157_6;
                                    unsigned _ldv8_157_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_157_0), "=r"(_ldv8_157_1), "=r"(_ldv8_157_2), "=r"(_ldv8_157_3), "=r"(_ldv8_157_4), "=r"(_ldv8_157_5), "=r"(_ldv8_157_6), "=r"(_ldv8_157_7) : "l"((const void*)(part_o + (off_0_23))) : "memory");
                                    part_2_1[0 + 0] = __uint_as_float(_ldv8_157_0);
                                    part_2_1[0 + 1] = __uint_as_float(_ldv8_157_1);
                                    part_2_1[0 + 2] = __uint_as_float(_ldv8_157_2);
                                    part_2_1[0 + 3] = __uint_as_float(_ldv8_157_3);
                                    part_2_1[0 + 4] = __uint_as_float(_ldv8_157_4);
                                    part_2_1[0 + 5] = __uint_as_float(_ldv8_157_5);
                                    part_2_1[0 + 6] = __uint_as_float(_ldv8_157_6);
                                    part_2_1[0 + 7] = __uint_as_float(_ldv8_157_7);
                                }
                                {
                                    unsigned _ldv8_158_0;
                                    unsigned _ldv8_158_1;
                                    unsigned _ldv8_158_2;
                                    unsigned _ldv8_158_3;
                                    unsigned _ldv8_158_4;
                                    unsigned _ldv8_158_5;
                                    unsigned _ldv8_158_6;
                                    unsigned _ldv8_158_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_158_0), "=r"(_ldv8_158_1), "=r"(_ldv8_158_2), "=r"(_ldv8_158_3), "=r"(_ldv8_158_4), "=r"(_ldv8_158_5), "=r"(_ldv8_158_6), "=r"(_ldv8_158_7) : "l"((const void*)(part_o + (off_0_23 + 8))) : "memory");
                                    part_2_1[8 + 0] = __uint_as_float(_ldv8_158_0);
                                    part_2_1[8 + 1] = __uint_as_float(_ldv8_158_1);
                                    part_2_1[8 + 2] = __uint_as_float(_ldv8_158_2);
                                    part_2_1[8 + 3] = __uint_as_float(_ldv8_158_3);
                                    part_2_1[8 + 4] = __uint_as_float(_ldv8_158_4);
                                    part_2_1[8 + 5] = __uint_as_float(_ldv8_158_5);
                                    part_2_1[8 + 6] = __uint_as_float(_ldv8_158_6);
                                    part_2_1[8 + 7] = __uint_as_float(_ldv8_158_7);
                                }
                                {
                                    unsigned _ldv8_159_0;
                                    unsigned _ldv8_159_1;
                                    unsigned _ldv8_159_2;
                                    unsigned _ldv8_159_3;
                                    unsigned _ldv8_159_4;
                                    unsigned _ldv8_159_5;
                                    unsigned _ldv8_159_6;
                                    unsigned _ldv8_159_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_159_0), "=r"(_ldv8_159_1), "=r"(_ldv8_159_2), "=r"(_ldv8_159_3), "=r"(_ldv8_159_4), "=r"(_ldv8_159_5), "=r"(_ldv8_159_6), "=r"(_ldv8_159_7) : "l"((const void*)(part_o + (off_0_23 + 64))) : "memory");
                                    part_hi_3_1[0 + 0] = __uint_as_float(_ldv8_159_0);
                                    part_hi_3_1[0 + 1] = __uint_as_float(_ldv8_159_1);
                                    part_hi_3_1[0 + 2] = __uint_as_float(_ldv8_159_2);
                                    part_hi_3_1[0 + 3] = __uint_as_float(_ldv8_159_3);
                                    part_hi_3_1[0 + 4] = __uint_as_float(_ldv8_159_4);
                                    part_hi_3_1[0 + 5] = __uint_as_float(_ldv8_159_5);
                                    part_hi_3_1[0 + 6] = __uint_as_float(_ldv8_159_6);
                                    part_hi_3_1[0 + 7] = __uint_as_float(_ldv8_159_7);
                                }
                                {
                                    unsigned _ldv8_160_0;
                                    unsigned _ldv8_160_1;
                                    unsigned _ldv8_160_2;
                                    unsigned _ldv8_160_3;
                                    unsigned _ldv8_160_4;
                                    unsigned _ldv8_160_5;
                                    unsigned _ldv8_160_6;
                                    unsigned _ldv8_160_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_160_0), "=r"(_ldv8_160_1), "=r"(_ldv8_160_2), "=r"(_ldv8_160_3), "=r"(_ldv8_160_4), "=r"(_ldv8_160_5), "=r"(_ldv8_160_6), "=r"(_ldv8_160_7) : "l"((const void*)(part_o + (off_0_23 + 64 + 8))) : "memory");
                                    part_hi_3_1[8 + 0] = __uint_as_float(_ldv8_160_0);
                                    part_hi_3_1[8 + 1] = __uint_as_float(_ldv8_160_1);
                                    part_hi_3_1[8 + 2] = __uint_as_float(_ldv8_160_2);
                                    part_hi_3_1[8 + 3] = __uint_as_float(_ldv8_160_3);
                                    part_hi_3_1[8 + 4] = __uint_as_float(_ldv8_160_4);
                                    part_hi_3_1[8 + 5] = __uint_as_float(_ldv8_160_5);
                                    part_hi_3_1[8 + 6] = __uint_as_float(_ldv8_160_6);
                                    part_hi_3_1[8 + 7] = __uint_as_float(_ldv8_160_7);
                                }
                                acc_0_1[0] = __fmaf_rn(part_2_1[0], w_c_27, acc_0_1[0]);
                                acc_hi_1_1[0] = __fmaf_rn(part_hi_3_1[0], w_c_27, acc_hi_1_1[0]);
                                acc_0_1[1] = __fmaf_rn(part_2_1[1], w_c_27, acc_0_1[1]);
                                acc_hi_1_1[1] = __fmaf_rn(part_hi_3_1[1], w_c_27, acc_hi_1_1[1]);
                                acc_0_1[2] = __fmaf_rn(part_2_1[2], w_c_27, acc_0_1[2]);
                                acc_hi_1_1[2] = __fmaf_rn(part_hi_3_1[2], w_c_27, acc_hi_1_1[2]);
                                acc_0_1[3] = __fmaf_rn(part_2_1[3], w_c_27, acc_0_1[3]);
                                acc_hi_1_1[3] = __fmaf_rn(part_hi_3_1[3], w_c_27, acc_hi_1_1[3]);
                                acc_0_1[4] = __fmaf_rn(part_2_1[4], w_c_27, acc_0_1[4]);
                                acc_hi_1_1[4] = __fmaf_rn(part_hi_3_1[4], w_c_27, acc_hi_1_1[4]);
                                acc_0_1[5] = __fmaf_rn(part_2_1[5], w_c_27, acc_0_1[5]);
                                acc_hi_1_1[5] = __fmaf_rn(part_hi_3_1[5], w_c_27, acc_hi_1_1[5]);
                                acc_0_1[6] = __fmaf_rn(part_2_1[6], w_c_27, acc_0_1[6]);
                                acc_hi_1_1[6] = __fmaf_rn(part_hi_3_1[6], w_c_27, acc_hi_1_1[6]);
                                acc_0_1[7] = __fmaf_rn(part_2_1[7], w_c_27, acc_0_1[7]);
                                acc_hi_1_1[7] = __fmaf_rn(part_hi_3_1[7], w_c_27, acc_hi_1_1[7]);
                                acc_0_1[8] = __fmaf_rn(part_2_1[8], w_c_27, acc_0_1[8]);
                                acc_hi_1_1[8] = __fmaf_rn(part_hi_3_1[8], w_c_27, acc_hi_1_1[8]);
                                acc_0_1[9] = __fmaf_rn(part_2_1[9], w_c_27, acc_0_1[9]);
                                acc_hi_1_1[9] = __fmaf_rn(part_hi_3_1[9], w_c_27, acc_hi_1_1[9]);
                                acc_0_1[10] = __fmaf_rn(part_2_1[10], w_c_27, acc_0_1[10]);
                                acc_hi_1_1[10] = __fmaf_rn(part_hi_3_1[10], w_c_27, acc_hi_1_1[10]);
                                acc_0_1[11] = __fmaf_rn(part_2_1[11], w_c_27, acc_0_1[11]);
                                acc_hi_1_1[11] = __fmaf_rn(part_hi_3_1[11], w_c_27, acc_hi_1_1[11]);
                                acc_0_1[12] = __fmaf_rn(part_2_1[12], w_c_27, acc_0_1[12]);
                                acc_hi_1_1[12] = __fmaf_rn(part_hi_3_1[12], w_c_27, acc_hi_1_1[12]);
                                acc_0_1[13] = __fmaf_rn(part_2_1[13], w_c_27, acc_0_1[13]);
                                acc_hi_1_1[13] = __fmaf_rn(part_hi_3_1[13], w_c_27, acc_hi_1_1[13]);
                                acc_0_1[14] = __fmaf_rn(part_2_1[14], w_c_27, acc_0_1[14]);
                                acc_hi_1_1[14] = __fmaf_rn(part_hi_3_1[14], w_c_27, acc_hi_1_1[14]);
                                acc_0_1[15] = __fmaf_rn(part_2_1[15], w_c_27, acc_0_1[15]);
                                acc_hi_1_1[15] = __fmaf_rn(part_hi_3_1[15], w_c_27, acc_hi_1_1[15]);
                            }
                            const float2 _scale2_161 = {final_scale, final_scale};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(acc_0_1)[_ls], _scale2_161);
                            unsigned int acc_0_bf16_1[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_0_1[_lp*2 + 0], acc_0_1[_lp*2+1 + 0]));
                                acc_0_bf16_1[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o10_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + 32 ^ ((warp % 4 * 32 + lane) * 128 + 32 >> 7 & 7) << 4))), "r"(acc_0_bf16_1[0]), "r"(acc_0_bf16_1[1]), "r"(acc_0_bf16_1[2]), "r"(acc_0_bf16_1[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o10_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + 48 ^ ((warp % 4 * 32 + lane) * 128 + 48 >> 7 & 7) << 4))), "r"(acc_0_bf16_1[4]), "r"(acc_0_bf16_1[5]), "r"(acc_0_bf16_1[6]), "r"(acc_0_bf16_1[7]) : "memory");
                            const float2 _scale2_162 = {final_scale, final_scale};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(acc_hi_1_1)[_ls], _scale2_162);
                            unsigned int acc_hi_1_bf16_1[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_hi_1_1[_lp*2 + 0], acc_hi_1_1[_lp*2+1 + 0]));
                                acc_hi_1_bf16_1[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o11_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + 32 ^ ((warp % 4 * 32 + lane) * 128 + 32 >> 7 & 7) << 4))), "r"(acc_hi_1_bf16_1[0]), "r"(acc_hi_1_bf16_1[1]), "r"(acc_hi_1_bf16_1[2]), "r"(acc_hi_1_bf16_1[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o11_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + 48 ^ ((warp % 4 * 32 + lane) * 128 + 48 >> 7 & 7) << 4))), "r"(acc_hi_1_bf16_1[4]), "r"(acc_hi_1_bf16_1[5]), "r"(acc_hi_1_bf16_1[6]), "r"(acc_hi_1_bf16_1[7]) : "memory");
                            float acc_4_1[16];
                            float acc_hi_5_1[16];
                            acc_4_1[0] = 0.0f;
                            acc_hi_5_1[0] = 0.0f;
                            acc_4_1[1] = 0.0f;
                            acc_hi_5_1[1] = 0.0f;
                            acc_4_1[2] = 0.0f;
                            acc_hi_5_1[2] = 0.0f;
                            acc_4_1[3] = 0.0f;
                            acc_hi_5_1[3] = 0.0f;
                            acc_4_1[4] = 0.0f;
                            acc_hi_5_1[4] = 0.0f;
                            acc_4_1[5] = 0.0f;
                            acc_hi_5_1[5] = 0.0f;
                            acc_4_1[6] = 0.0f;
                            acc_hi_5_1[6] = 0.0f;
                            acc_4_1[7] = 0.0f;
                            acc_hi_5_1[7] = 0.0f;
                            acc_4_1[8] = 0.0f;
                            acc_hi_5_1[8] = 0.0f;
                            acc_4_1[9] = 0.0f;
                            acc_hi_5_1[9] = 0.0f;
                            acc_4_1[10] = 0.0f;
                            acc_hi_5_1[10] = 0.0f;
                            acc_4_1[11] = 0.0f;
                            acc_hi_5_1[11] = 0.0f;
                            acc_4_1[12] = 0.0f;
                            acc_hi_5_1[12] = 0.0f;
                            acc_4_1[13] = 0.0f;
                            acc_hi_5_1[13] = 0.0f;
                            acc_4_1[14] = 0.0f;
                            acc_hi_5_1[14] = 0.0f;
                            acc_4_1[15] = 0.0f;
                            acc_hi_5_1[15] = 0.0f;
                            float part_6_1[16];
                            float part_hi_7_1[16];
                            if (n_chunks_u_0 > 0) {
                                unsigned int crow_24 = tile_slot0 * 512 + part_row;
                                float w_c_28 = approx_exp2(part_ml[crow_24 * 2] - row_max_all);
                                unsigned int off_0_24 = crow_24 * 128 + 32;
                                {
                                    unsigned _ldv8_163_0;
                                    unsigned _ldv8_163_1;
                                    unsigned _ldv8_163_2;
                                    unsigned _ldv8_163_3;
                                    unsigned _ldv8_163_4;
                                    unsigned _ldv8_163_5;
                                    unsigned _ldv8_163_6;
                                    unsigned _ldv8_163_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_163_0), "=r"(_ldv8_163_1), "=r"(_ldv8_163_2), "=r"(_ldv8_163_3), "=r"(_ldv8_163_4), "=r"(_ldv8_163_5), "=r"(_ldv8_163_6), "=r"(_ldv8_163_7) : "l"((const void*)(part_o + (off_0_24))) : "memory");
                                    part_6_1[0 + 0] = __uint_as_float(_ldv8_163_0);
                                    part_6_1[0 + 1] = __uint_as_float(_ldv8_163_1);
                                    part_6_1[0 + 2] = __uint_as_float(_ldv8_163_2);
                                    part_6_1[0 + 3] = __uint_as_float(_ldv8_163_3);
                                    part_6_1[0 + 4] = __uint_as_float(_ldv8_163_4);
                                    part_6_1[0 + 5] = __uint_as_float(_ldv8_163_5);
                                    part_6_1[0 + 6] = __uint_as_float(_ldv8_163_6);
                                    part_6_1[0 + 7] = __uint_as_float(_ldv8_163_7);
                                }
                                {
                                    unsigned _ldv8_164_0;
                                    unsigned _ldv8_164_1;
                                    unsigned _ldv8_164_2;
                                    unsigned _ldv8_164_3;
                                    unsigned _ldv8_164_4;
                                    unsigned _ldv8_164_5;
                                    unsigned _ldv8_164_6;
                                    unsigned _ldv8_164_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_164_0), "=r"(_ldv8_164_1), "=r"(_ldv8_164_2), "=r"(_ldv8_164_3), "=r"(_ldv8_164_4), "=r"(_ldv8_164_5), "=r"(_ldv8_164_6), "=r"(_ldv8_164_7) : "l"((const void*)(part_o + (off_0_24 + 8))) : "memory");
                                    part_6_1[8 + 0] = __uint_as_float(_ldv8_164_0);
                                    part_6_1[8 + 1] = __uint_as_float(_ldv8_164_1);
                                    part_6_1[8 + 2] = __uint_as_float(_ldv8_164_2);
                                    part_6_1[8 + 3] = __uint_as_float(_ldv8_164_3);
                                    part_6_1[8 + 4] = __uint_as_float(_ldv8_164_4);
                                    part_6_1[8 + 5] = __uint_as_float(_ldv8_164_5);
                                    part_6_1[8 + 6] = __uint_as_float(_ldv8_164_6);
                                    part_6_1[8 + 7] = __uint_as_float(_ldv8_164_7);
                                }
                                {
                                    unsigned _ldv8_165_0;
                                    unsigned _ldv8_165_1;
                                    unsigned _ldv8_165_2;
                                    unsigned _ldv8_165_3;
                                    unsigned _ldv8_165_4;
                                    unsigned _ldv8_165_5;
                                    unsigned _ldv8_165_6;
                                    unsigned _ldv8_165_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_165_0), "=r"(_ldv8_165_1), "=r"(_ldv8_165_2), "=r"(_ldv8_165_3), "=r"(_ldv8_165_4), "=r"(_ldv8_165_5), "=r"(_ldv8_165_6), "=r"(_ldv8_165_7) : "l"((const void*)(part_o + (off_0_24 + 64))) : "memory");
                                    part_hi_7_1[0 + 0] = __uint_as_float(_ldv8_165_0);
                                    part_hi_7_1[0 + 1] = __uint_as_float(_ldv8_165_1);
                                    part_hi_7_1[0 + 2] = __uint_as_float(_ldv8_165_2);
                                    part_hi_7_1[0 + 3] = __uint_as_float(_ldv8_165_3);
                                    part_hi_7_1[0 + 4] = __uint_as_float(_ldv8_165_4);
                                    part_hi_7_1[0 + 5] = __uint_as_float(_ldv8_165_5);
                                    part_hi_7_1[0 + 6] = __uint_as_float(_ldv8_165_6);
                                    part_hi_7_1[0 + 7] = __uint_as_float(_ldv8_165_7);
                                }
                                {
                                    unsigned _ldv8_166_0;
                                    unsigned _ldv8_166_1;
                                    unsigned _ldv8_166_2;
                                    unsigned _ldv8_166_3;
                                    unsigned _ldv8_166_4;
                                    unsigned _ldv8_166_5;
                                    unsigned _ldv8_166_6;
                                    unsigned _ldv8_166_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_166_0), "=r"(_ldv8_166_1), "=r"(_ldv8_166_2), "=r"(_ldv8_166_3), "=r"(_ldv8_166_4), "=r"(_ldv8_166_5), "=r"(_ldv8_166_6), "=r"(_ldv8_166_7) : "l"((const void*)(part_o + (off_0_24 + 64 + 8))) : "memory");
                                    part_hi_7_1[8 + 0] = __uint_as_float(_ldv8_166_0);
                                    part_hi_7_1[8 + 1] = __uint_as_float(_ldv8_166_1);
                                    part_hi_7_1[8 + 2] = __uint_as_float(_ldv8_166_2);
                                    part_hi_7_1[8 + 3] = __uint_as_float(_ldv8_166_3);
                                    part_hi_7_1[8 + 4] = __uint_as_float(_ldv8_166_4);
                                    part_hi_7_1[8 + 5] = __uint_as_float(_ldv8_166_5);
                                    part_hi_7_1[8 + 6] = __uint_as_float(_ldv8_166_6);
                                    part_hi_7_1[8 + 7] = __uint_as_float(_ldv8_166_7);
                                }
                                acc_4_1[0] = __fmaf_rn(part_6_1[0], w_c_28, acc_4_1[0]);
                                acc_hi_5_1[0] = __fmaf_rn(part_hi_7_1[0], w_c_28, acc_hi_5_1[0]);
                                acc_4_1[1] = __fmaf_rn(part_6_1[1], w_c_28, acc_4_1[1]);
                                acc_hi_5_1[1] = __fmaf_rn(part_hi_7_1[1], w_c_28, acc_hi_5_1[1]);
                                acc_4_1[2] = __fmaf_rn(part_6_1[2], w_c_28, acc_4_1[2]);
                                acc_hi_5_1[2] = __fmaf_rn(part_hi_7_1[2], w_c_28, acc_hi_5_1[2]);
                                acc_4_1[3] = __fmaf_rn(part_6_1[3], w_c_28, acc_4_1[3]);
                                acc_hi_5_1[3] = __fmaf_rn(part_hi_7_1[3], w_c_28, acc_hi_5_1[3]);
                                acc_4_1[4] = __fmaf_rn(part_6_1[4], w_c_28, acc_4_1[4]);
                                acc_hi_5_1[4] = __fmaf_rn(part_hi_7_1[4], w_c_28, acc_hi_5_1[4]);
                                acc_4_1[5] = __fmaf_rn(part_6_1[5], w_c_28, acc_4_1[5]);
                                acc_hi_5_1[5] = __fmaf_rn(part_hi_7_1[5], w_c_28, acc_hi_5_1[5]);
                                acc_4_1[6] = __fmaf_rn(part_6_1[6], w_c_28, acc_4_1[6]);
                                acc_hi_5_1[6] = __fmaf_rn(part_hi_7_1[6], w_c_28, acc_hi_5_1[6]);
                                acc_4_1[7] = __fmaf_rn(part_6_1[7], w_c_28, acc_4_1[7]);
                                acc_hi_5_1[7] = __fmaf_rn(part_hi_7_1[7], w_c_28, acc_hi_5_1[7]);
                                acc_4_1[8] = __fmaf_rn(part_6_1[8], w_c_28, acc_4_1[8]);
                                acc_hi_5_1[8] = __fmaf_rn(part_hi_7_1[8], w_c_28, acc_hi_5_1[8]);
                                acc_4_1[9] = __fmaf_rn(part_6_1[9], w_c_28, acc_4_1[9]);
                                acc_hi_5_1[9] = __fmaf_rn(part_hi_7_1[9], w_c_28, acc_hi_5_1[9]);
                                acc_4_1[10] = __fmaf_rn(part_6_1[10], w_c_28, acc_4_1[10]);
                                acc_hi_5_1[10] = __fmaf_rn(part_hi_7_1[10], w_c_28, acc_hi_5_1[10]);
                                acc_4_1[11] = __fmaf_rn(part_6_1[11], w_c_28, acc_4_1[11]);
                                acc_hi_5_1[11] = __fmaf_rn(part_hi_7_1[11], w_c_28, acc_hi_5_1[11]);
                                acc_4_1[12] = __fmaf_rn(part_6_1[12], w_c_28, acc_4_1[12]);
                                acc_hi_5_1[12] = __fmaf_rn(part_hi_7_1[12], w_c_28, acc_hi_5_1[12]);
                                acc_4_1[13] = __fmaf_rn(part_6_1[13], w_c_28, acc_4_1[13]);
                                acc_hi_5_1[13] = __fmaf_rn(part_hi_7_1[13], w_c_28, acc_hi_5_1[13]);
                                acc_4_1[14] = __fmaf_rn(part_6_1[14], w_c_28, acc_4_1[14]);
                                acc_hi_5_1[14] = __fmaf_rn(part_hi_7_1[14], w_c_28, acc_hi_5_1[14]);
                                acc_4_1[15] = __fmaf_rn(part_6_1[15], w_c_28, acc_4_1[15]);
                                acc_hi_5_1[15] = __fmaf_rn(part_hi_7_1[15], w_c_28, acc_hi_5_1[15]);
                            }
                            if (n_chunks_u_0 > 1) {
                                unsigned int crow_25 = (tile_slot0 + 1) * 512 + part_row;
                                float w_c_29 = approx_exp2(part_ml[crow_25 * 2] - row_max_all);
                                unsigned int off_0_25 = crow_25 * 128 + 32;
                                {
                                    unsigned _ldv8_167_0;
                                    unsigned _ldv8_167_1;
                                    unsigned _ldv8_167_2;
                                    unsigned _ldv8_167_3;
                                    unsigned _ldv8_167_4;
                                    unsigned _ldv8_167_5;
                                    unsigned _ldv8_167_6;
                                    unsigned _ldv8_167_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_167_0), "=r"(_ldv8_167_1), "=r"(_ldv8_167_2), "=r"(_ldv8_167_3), "=r"(_ldv8_167_4), "=r"(_ldv8_167_5), "=r"(_ldv8_167_6), "=r"(_ldv8_167_7) : "l"((const void*)(part_o + (off_0_25))) : "memory");
                                    part_6_1[0 + 0] = __uint_as_float(_ldv8_167_0);
                                    part_6_1[0 + 1] = __uint_as_float(_ldv8_167_1);
                                    part_6_1[0 + 2] = __uint_as_float(_ldv8_167_2);
                                    part_6_1[0 + 3] = __uint_as_float(_ldv8_167_3);
                                    part_6_1[0 + 4] = __uint_as_float(_ldv8_167_4);
                                    part_6_1[0 + 5] = __uint_as_float(_ldv8_167_5);
                                    part_6_1[0 + 6] = __uint_as_float(_ldv8_167_6);
                                    part_6_1[0 + 7] = __uint_as_float(_ldv8_167_7);
                                }
                                {
                                    unsigned _ldv8_168_0;
                                    unsigned _ldv8_168_1;
                                    unsigned _ldv8_168_2;
                                    unsigned _ldv8_168_3;
                                    unsigned _ldv8_168_4;
                                    unsigned _ldv8_168_5;
                                    unsigned _ldv8_168_6;
                                    unsigned _ldv8_168_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_168_0), "=r"(_ldv8_168_1), "=r"(_ldv8_168_2), "=r"(_ldv8_168_3), "=r"(_ldv8_168_4), "=r"(_ldv8_168_5), "=r"(_ldv8_168_6), "=r"(_ldv8_168_7) : "l"((const void*)(part_o + (off_0_25 + 8))) : "memory");
                                    part_6_1[8 + 0] = __uint_as_float(_ldv8_168_0);
                                    part_6_1[8 + 1] = __uint_as_float(_ldv8_168_1);
                                    part_6_1[8 + 2] = __uint_as_float(_ldv8_168_2);
                                    part_6_1[8 + 3] = __uint_as_float(_ldv8_168_3);
                                    part_6_1[8 + 4] = __uint_as_float(_ldv8_168_4);
                                    part_6_1[8 + 5] = __uint_as_float(_ldv8_168_5);
                                    part_6_1[8 + 6] = __uint_as_float(_ldv8_168_6);
                                    part_6_1[8 + 7] = __uint_as_float(_ldv8_168_7);
                                }
                                {
                                    unsigned _ldv8_169_0;
                                    unsigned _ldv8_169_1;
                                    unsigned _ldv8_169_2;
                                    unsigned _ldv8_169_3;
                                    unsigned _ldv8_169_4;
                                    unsigned _ldv8_169_5;
                                    unsigned _ldv8_169_6;
                                    unsigned _ldv8_169_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_169_0), "=r"(_ldv8_169_1), "=r"(_ldv8_169_2), "=r"(_ldv8_169_3), "=r"(_ldv8_169_4), "=r"(_ldv8_169_5), "=r"(_ldv8_169_6), "=r"(_ldv8_169_7) : "l"((const void*)(part_o + (off_0_25 + 64))) : "memory");
                                    part_hi_7_1[0 + 0] = __uint_as_float(_ldv8_169_0);
                                    part_hi_7_1[0 + 1] = __uint_as_float(_ldv8_169_1);
                                    part_hi_7_1[0 + 2] = __uint_as_float(_ldv8_169_2);
                                    part_hi_7_1[0 + 3] = __uint_as_float(_ldv8_169_3);
                                    part_hi_7_1[0 + 4] = __uint_as_float(_ldv8_169_4);
                                    part_hi_7_1[0 + 5] = __uint_as_float(_ldv8_169_5);
                                    part_hi_7_1[0 + 6] = __uint_as_float(_ldv8_169_6);
                                    part_hi_7_1[0 + 7] = __uint_as_float(_ldv8_169_7);
                                }
                                {
                                    unsigned _ldv8_170_0;
                                    unsigned _ldv8_170_1;
                                    unsigned _ldv8_170_2;
                                    unsigned _ldv8_170_3;
                                    unsigned _ldv8_170_4;
                                    unsigned _ldv8_170_5;
                                    unsigned _ldv8_170_6;
                                    unsigned _ldv8_170_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_170_0), "=r"(_ldv8_170_1), "=r"(_ldv8_170_2), "=r"(_ldv8_170_3), "=r"(_ldv8_170_4), "=r"(_ldv8_170_5), "=r"(_ldv8_170_6), "=r"(_ldv8_170_7) : "l"((const void*)(part_o + (off_0_25 + 64 + 8))) : "memory");
                                    part_hi_7_1[8 + 0] = __uint_as_float(_ldv8_170_0);
                                    part_hi_7_1[8 + 1] = __uint_as_float(_ldv8_170_1);
                                    part_hi_7_1[8 + 2] = __uint_as_float(_ldv8_170_2);
                                    part_hi_7_1[8 + 3] = __uint_as_float(_ldv8_170_3);
                                    part_hi_7_1[8 + 4] = __uint_as_float(_ldv8_170_4);
                                    part_hi_7_1[8 + 5] = __uint_as_float(_ldv8_170_5);
                                    part_hi_7_1[8 + 6] = __uint_as_float(_ldv8_170_6);
                                    part_hi_7_1[8 + 7] = __uint_as_float(_ldv8_170_7);
                                }
                                acc_4_1[0] = __fmaf_rn(part_6_1[0], w_c_29, acc_4_1[0]);
                                acc_hi_5_1[0] = __fmaf_rn(part_hi_7_1[0], w_c_29, acc_hi_5_1[0]);
                                acc_4_1[1] = __fmaf_rn(part_6_1[1], w_c_29, acc_4_1[1]);
                                acc_hi_5_1[1] = __fmaf_rn(part_hi_7_1[1], w_c_29, acc_hi_5_1[1]);
                                acc_4_1[2] = __fmaf_rn(part_6_1[2], w_c_29, acc_4_1[2]);
                                acc_hi_5_1[2] = __fmaf_rn(part_hi_7_1[2], w_c_29, acc_hi_5_1[2]);
                                acc_4_1[3] = __fmaf_rn(part_6_1[3], w_c_29, acc_4_1[3]);
                                acc_hi_5_1[3] = __fmaf_rn(part_hi_7_1[3], w_c_29, acc_hi_5_1[3]);
                                acc_4_1[4] = __fmaf_rn(part_6_1[4], w_c_29, acc_4_1[4]);
                                acc_hi_5_1[4] = __fmaf_rn(part_hi_7_1[4], w_c_29, acc_hi_5_1[4]);
                                acc_4_1[5] = __fmaf_rn(part_6_1[5], w_c_29, acc_4_1[5]);
                                acc_hi_5_1[5] = __fmaf_rn(part_hi_7_1[5], w_c_29, acc_hi_5_1[5]);
                                acc_4_1[6] = __fmaf_rn(part_6_1[6], w_c_29, acc_4_1[6]);
                                acc_hi_5_1[6] = __fmaf_rn(part_hi_7_1[6], w_c_29, acc_hi_5_1[6]);
                                acc_4_1[7] = __fmaf_rn(part_6_1[7], w_c_29, acc_4_1[7]);
                                acc_hi_5_1[7] = __fmaf_rn(part_hi_7_1[7], w_c_29, acc_hi_5_1[7]);
                                acc_4_1[8] = __fmaf_rn(part_6_1[8], w_c_29, acc_4_1[8]);
                                acc_hi_5_1[8] = __fmaf_rn(part_hi_7_1[8], w_c_29, acc_hi_5_1[8]);
                                acc_4_1[9] = __fmaf_rn(part_6_1[9], w_c_29, acc_4_1[9]);
                                acc_hi_5_1[9] = __fmaf_rn(part_hi_7_1[9], w_c_29, acc_hi_5_1[9]);
                                acc_4_1[10] = __fmaf_rn(part_6_1[10], w_c_29, acc_4_1[10]);
                                acc_hi_5_1[10] = __fmaf_rn(part_hi_7_1[10], w_c_29, acc_hi_5_1[10]);
                                acc_4_1[11] = __fmaf_rn(part_6_1[11], w_c_29, acc_4_1[11]);
                                acc_hi_5_1[11] = __fmaf_rn(part_hi_7_1[11], w_c_29, acc_hi_5_1[11]);
                                acc_4_1[12] = __fmaf_rn(part_6_1[12], w_c_29, acc_4_1[12]);
                                acc_hi_5_1[12] = __fmaf_rn(part_hi_7_1[12], w_c_29, acc_hi_5_1[12]);
                                acc_4_1[13] = __fmaf_rn(part_6_1[13], w_c_29, acc_4_1[13]);
                                acc_hi_5_1[13] = __fmaf_rn(part_hi_7_1[13], w_c_29, acc_hi_5_1[13]);
                                acc_4_1[14] = __fmaf_rn(part_6_1[14], w_c_29, acc_4_1[14]);
                                acc_hi_5_1[14] = __fmaf_rn(part_hi_7_1[14], w_c_29, acc_hi_5_1[14]);
                                acc_4_1[15] = __fmaf_rn(part_6_1[15], w_c_29, acc_4_1[15]);
                                acc_hi_5_1[15] = __fmaf_rn(part_hi_7_1[15], w_c_29, acc_hi_5_1[15]);
                            }
                            if (n_chunks_u_0 > 2) {
                                unsigned int crow_26 = (tile_slot0 + 2) * 512 + part_row;
                                float w_c_30 = approx_exp2(part_ml[crow_26 * 2] - row_max_all);
                                unsigned int off_0_26 = crow_26 * 128 + 32;
                                {
                                    unsigned _ldv8_171_0;
                                    unsigned _ldv8_171_1;
                                    unsigned _ldv8_171_2;
                                    unsigned _ldv8_171_3;
                                    unsigned _ldv8_171_4;
                                    unsigned _ldv8_171_5;
                                    unsigned _ldv8_171_6;
                                    unsigned _ldv8_171_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_171_0), "=r"(_ldv8_171_1), "=r"(_ldv8_171_2), "=r"(_ldv8_171_3), "=r"(_ldv8_171_4), "=r"(_ldv8_171_5), "=r"(_ldv8_171_6), "=r"(_ldv8_171_7) : "l"((const void*)(part_o + (off_0_26))) : "memory");
                                    part_6_1[0 + 0] = __uint_as_float(_ldv8_171_0);
                                    part_6_1[0 + 1] = __uint_as_float(_ldv8_171_1);
                                    part_6_1[0 + 2] = __uint_as_float(_ldv8_171_2);
                                    part_6_1[0 + 3] = __uint_as_float(_ldv8_171_3);
                                    part_6_1[0 + 4] = __uint_as_float(_ldv8_171_4);
                                    part_6_1[0 + 5] = __uint_as_float(_ldv8_171_5);
                                    part_6_1[0 + 6] = __uint_as_float(_ldv8_171_6);
                                    part_6_1[0 + 7] = __uint_as_float(_ldv8_171_7);
                                }
                                {
                                    unsigned _ldv8_172_0;
                                    unsigned _ldv8_172_1;
                                    unsigned _ldv8_172_2;
                                    unsigned _ldv8_172_3;
                                    unsigned _ldv8_172_4;
                                    unsigned _ldv8_172_5;
                                    unsigned _ldv8_172_6;
                                    unsigned _ldv8_172_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_172_0), "=r"(_ldv8_172_1), "=r"(_ldv8_172_2), "=r"(_ldv8_172_3), "=r"(_ldv8_172_4), "=r"(_ldv8_172_5), "=r"(_ldv8_172_6), "=r"(_ldv8_172_7) : "l"((const void*)(part_o + (off_0_26 + 8))) : "memory");
                                    part_6_1[8 + 0] = __uint_as_float(_ldv8_172_0);
                                    part_6_1[8 + 1] = __uint_as_float(_ldv8_172_1);
                                    part_6_1[8 + 2] = __uint_as_float(_ldv8_172_2);
                                    part_6_1[8 + 3] = __uint_as_float(_ldv8_172_3);
                                    part_6_1[8 + 4] = __uint_as_float(_ldv8_172_4);
                                    part_6_1[8 + 5] = __uint_as_float(_ldv8_172_5);
                                    part_6_1[8 + 6] = __uint_as_float(_ldv8_172_6);
                                    part_6_1[8 + 7] = __uint_as_float(_ldv8_172_7);
                                }
                                {
                                    unsigned _ldv8_173_0;
                                    unsigned _ldv8_173_1;
                                    unsigned _ldv8_173_2;
                                    unsigned _ldv8_173_3;
                                    unsigned _ldv8_173_4;
                                    unsigned _ldv8_173_5;
                                    unsigned _ldv8_173_6;
                                    unsigned _ldv8_173_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_173_0), "=r"(_ldv8_173_1), "=r"(_ldv8_173_2), "=r"(_ldv8_173_3), "=r"(_ldv8_173_4), "=r"(_ldv8_173_5), "=r"(_ldv8_173_6), "=r"(_ldv8_173_7) : "l"((const void*)(part_o + (off_0_26 + 64))) : "memory");
                                    part_hi_7_1[0 + 0] = __uint_as_float(_ldv8_173_0);
                                    part_hi_7_1[0 + 1] = __uint_as_float(_ldv8_173_1);
                                    part_hi_7_1[0 + 2] = __uint_as_float(_ldv8_173_2);
                                    part_hi_7_1[0 + 3] = __uint_as_float(_ldv8_173_3);
                                    part_hi_7_1[0 + 4] = __uint_as_float(_ldv8_173_4);
                                    part_hi_7_1[0 + 5] = __uint_as_float(_ldv8_173_5);
                                    part_hi_7_1[0 + 6] = __uint_as_float(_ldv8_173_6);
                                    part_hi_7_1[0 + 7] = __uint_as_float(_ldv8_173_7);
                                }
                                {
                                    unsigned _ldv8_174_0;
                                    unsigned _ldv8_174_1;
                                    unsigned _ldv8_174_2;
                                    unsigned _ldv8_174_3;
                                    unsigned _ldv8_174_4;
                                    unsigned _ldv8_174_5;
                                    unsigned _ldv8_174_6;
                                    unsigned _ldv8_174_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_174_0), "=r"(_ldv8_174_1), "=r"(_ldv8_174_2), "=r"(_ldv8_174_3), "=r"(_ldv8_174_4), "=r"(_ldv8_174_5), "=r"(_ldv8_174_6), "=r"(_ldv8_174_7) : "l"((const void*)(part_o + (off_0_26 + 64 + 8))) : "memory");
                                    part_hi_7_1[8 + 0] = __uint_as_float(_ldv8_174_0);
                                    part_hi_7_1[8 + 1] = __uint_as_float(_ldv8_174_1);
                                    part_hi_7_1[8 + 2] = __uint_as_float(_ldv8_174_2);
                                    part_hi_7_1[8 + 3] = __uint_as_float(_ldv8_174_3);
                                    part_hi_7_1[8 + 4] = __uint_as_float(_ldv8_174_4);
                                    part_hi_7_1[8 + 5] = __uint_as_float(_ldv8_174_5);
                                    part_hi_7_1[8 + 6] = __uint_as_float(_ldv8_174_6);
                                    part_hi_7_1[8 + 7] = __uint_as_float(_ldv8_174_7);
                                }
                                acc_4_1[0] = __fmaf_rn(part_6_1[0], w_c_30, acc_4_1[0]);
                                acc_hi_5_1[0] = __fmaf_rn(part_hi_7_1[0], w_c_30, acc_hi_5_1[0]);
                                acc_4_1[1] = __fmaf_rn(part_6_1[1], w_c_30, acc_4_1[1]);
                                acc_hi_5_1[1] = __fmaf_rn(part_hi_7_1[1], w_c_30, acc_hi_5_1[1]);
                                acc_4_1[2] = __fmaf_rn(part_6_1[2], w_c_30, acc_4_1[2]);
                                acc_hi_5_1[2] = __fmaf_rn(part_hi_7_1[2], w_c_30, acc_hi_5_1[2]);
                                acc_4_1[3] = __fmaf_rn(part_6_1[3], w_c_30, acc_4_1[3]);
                                acc_hi_5_1[3] = __fmaf_rn(part_hi_7_1[3], w_c_30, acc_hi_5_1[3]);
                                acc_4_1[4] = __fmaf_rn(part_6_1[4], w_c_30, acc_4_1[4]);
                                acc_hi_5_1[4] = __fmaf_rn(part_hi_7_1[4], w_c_30, acc_hi_5_1[4]);
                                acc_4_1[5] = __fmaf_rn(part_6_1[5], w_c_30, acc_4_1[5]);
                                acc_hi_5_1[5] = __fmaf_rn(part_hi_7_1[5], w_c_30, acc_hi_5_1[5]);
                                acc_4_1[6] = __fmaf_rn(part_6_1[6], w_c_30, acc_4_1[6]);
                                acc_hi_5_1[6] = __fmaf_rn(part_hi_7_1[6], w_c_30, acc_hi_5_1[6]);
                                acc_4_1[7] = __fmaf_rn(part_6_1[7], w_c_30, acc_4_1[7]);
                                acc_hi_5_1[7] = __fmaf_rn(part_hi_7_1[7], w_c_30, acc_hi_5_1[7]);
                                acc_4_1[8] = __fmaf_rn(part_6_1[8], w_c_30, acc_4_1[8]);
                                acc_hi_5_1[8] = __fmaf_rn(part_hi_7_1[8], w_c_30, acc_hi_5_1[8]);
                                acc_4_1[9] = __fmaf_rn(part_6_1[9], w_c_30, acc_4_1[9]);
                                acc_hi_5_1[9] = __fmaf_rn(part_hi_7_1[9], w_c_30, acc_hi_5_1[9]);
                                acc_4_1[10] = __fmaf_rn(part_6_1[10], w_c_30, acc_4_1[10]);
                                acc_hi_5_1[10] = __fmaf_rn(part_hi_7_1[10], w_c_30, acc_hi_5_1[10]);
                                acc_4_1[11] = __fmaf_rn(part_6_1[11], w_c_30, acc_4_1[11]);
                                acc_hi_5_1[11] = __fmaf_rn(part_hi_7_1[11], w_c_30, acc_hi_5_1[11]);
                                acc_4_1[12] = __fmaf_rn(part_6_1[12], w_c_30, acc_4_1[12]);
                                acc_hi_5_1[12] = __fmaf_rn(part_hi_7_1[12], w_c_30, acc_hi_5_1[12]);
                                acc_4_1[13] = __fmaf_rn(part_6_1[13], w_c_30, acc_4_1[13]);
                                acc_hi_5_1[13] = __fmaf_rn(part_hi_7_1[13], w_c_30, acc_hi_5_1[13]);
                                acc_4_1[14] = __fmaf_rn(part_6_1[14], w_c_30, acc_4_1[14]);
                                acc_hi_5_1[14] = __fmaf_rn(part_hi_7_1[14], w_c_30, acc_hi_5_1[14]);
                                acc_4_1[15] = __fmaf_rn(part_6_1[15], w_c_30, acc_4_1[15]);
                                acc_hi_5_1[15] = __fmaf_rn(part_hi_7_1[15], w_c_30, acc_hi_5_1[15]);
                            }
                            if (n_chunks_u_0 > 3) {
                                unsigned int crow_27 = (tile_slot0 + 3) * 512 + part_row;
                                float w_c_31 = approx_exp2(part_ml[crow_27 * 2] - row_max_all);
                                unsigned int off_0_27 = crow_27 * 128 + 32;
                                {
                                    unsigned _ldv8_175_0;
                                    unsigned _ldv8_175_1;
                                    unsigned _ldv8_175_2;
                                    unsigned _ldv8_175_3;
                                    unsigned _ldv8_175_4;
                                    unsigned _ldv8_175_5;
                                    unsigned _ldv8_175_6;
                                    unsigned _ldv8_175_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_175_0), "=r"(_ldv8_175_1), "=r"(_ldv8_175_2), "=r"(_ldv8_175_3), "=r"(_ldv8_175_4), "=r"(_ldv8_175_5), "=r"(_ldv8_175_6), "=r"(_ldv8_175_7) : "l"((const void*)(part_o + (off_0_27))) : "memory");
                                    part_6_1[0 + 0] = __uint_as_float(_ldv8_175_0);
                                    part_6_1[0 + 1] = __uint_as_float(_ldv8_175_1);
                                    part_6_1[0 + 2] = __uint_as_float(_ldv8_175_2);
                                    part_6_1[0 + 3] = __uint_as_float(_ldv8_175_3);
                                    part_6_1[0 + 4] = __uint_as_float(_ldv8_175_4);
                                    part_6_1[0 + 5] = __uint_as_float(_ldv8_175_5);
                                    part_6_1[0 + 6] = __uint_as_float(_ldv8_175_6);
                                    part_6_1[0 + 7] = __uint_as_float(_ldv8_175_7);
                                }
                                {
                                    unsigned _ldv8_176_0;
                                    unsigned _ldv8_176_1;
                                    unsigned _ldv8_176_2;
                                    unsigned _ldv8_176_3;
                                    unsigned _ldv8_176_4;
                                    unsigned _ldv8_176_5;
                                    unsigned _ldv8_176_6;
                                    unsigned _ldv8_176_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_176_0), "=r"(_ldv8_176_1), "=r"(_ldv8_176_2), "=r"(_ldv8_176_3), "=r"(_ldv8_176_4), "=r"(_ldv8_176_5), "=r"(_ldv8_176_6), "=r"(_ldv8_176_7) : "l"((const void*)(part_o + (off_0_27 + 8))) : "memory");
                                    part_6_1[8 + 0] = __uint_as_float(_ldv8_176_0);
                                    part_6_1[8 + 1] = __uint_as_float(_ldv8_176_1);
                                    part_6_1[8 + 2] = __uint_as_float(_ldv8_176_2);
                                    part_6_1[8 + 3] = __uint_as_float(_ldv8_176_3);
                                    part_6_1[8 + 4] = __uint_as_float(_ldv8_176_4);
                                    part_6_1[8 + 5] = __uint_as_float(_ldv8_176_5);
                                    part_6_1[8 + 6] = __uint_as_float(_ldv8_176_6);
                                    part_6_1[8 + 7] = __uint_as_float(_ldv8_176_7);
                                }
                                {
                                    unsigned _ldv8_177_0;
                                    unsigned _ldv8_177_1;
                                    unsigned _ldv8_177_2;
                                    unsigned _ldv8_177_3;
                                    unsigned _ldv8_177_4;
                                    unsigned _ldv8_177_5;
                                    unsigned _ldv8_177_6;
                                    unsigned _ldv8_177_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_177_0), "=r"(_ldv8_177_1), "=r"(_ldv8_177_2), "=r"(_ldv8_177_3), "=r"(_ldv8_177_4), "=r"(_ldv8_177_5), "=r"(_ldv8_177_6), "=r"(_ldv8_177_7) : "l"((const void*)(part_o + (off_0_27 + 64))) : "memory");
                                    part_hi_7_1[0 + 0] = __uint_as_float(_ldv8_177_0);
                                    part_hi_7_1[0 + 1] = __uint_as_float(_ldv8_177_1);
                                    part_hi_7_1[0 + 2] = __uint_as_float(_ldv8_177_2);
                                    part_hi_7_1[0 + 3] = __uint_as_float(_ldv8_177_3);
                                    part_hi_7_1[0 + 4] = __uint_as_float(_ldv8_177_4);
                                    part_hi_7_1[0 + 5] = __uint_as_float(_ldv8_177_5);
                                    part_hi_7_1[0 + 6] = __uint_as_float(_ldv8_177_6);
                                    part_hi_7_1[0 + 7] = __uint_as_float(_ldv8_177_7);
                                }
                                {
                                    unsigned _ldv8_178_0;
                                    unsigned _ldv8_178_1;
                                    unsigned _ldv8_178_2;
                                    unsigned _ldv8_178_3;
                                    unsigned _ldv8_178_4;
                                    unsigned _ldv8_178_5;
                                    unsigned _ldv8_178_6;
                                    unsigned _ldv8_178_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_178_0), "=r"(_ldv8_178_1), "=r"(_ldv8_178_2), "=r"(_ldv8_178_3), "=r"(_ldv8_178_4), "=r"(_ldv8_178_5), "=r"(_ldv8_178_6), "=r"(_ldv8_178_7) : "l"((const void*)(part_o + (off_0_27 + 64 + 8))) : "memory");
                                    part_hi_7_1[8 + 0] = __uint_as_float(_ldv8_178_0);
                                    part_hi_7_1[8 + 1] = __uint_as_float(_ldv8_178_1);
                                    part_hi_7_1[8 + 2] = __uint_as_float(_ldv8_178_2);
                                    part_hi_7_1[8 + 3] = __uint_as_float(_ldv8_178_3);
                                    part_hi_7_1[8 + 4] = __uint_as_float(_ldv8_178_4);
                                    part_hi_7_1[8 + 5] = __uint_as_float(_ldv8_178_5);
                                    part_hi_7_1[8 + 6] = __uint_as_float(_ldv8_178_6);
                                    part_hi_7_1[8 + 7] = __uint_as_float(_ldv8_178_7);
                                }
                                acc_4_1[0] = __fmaf_rn(part_6_1[0], w_c_31, acc_4_1[0]);
                                acc_hi_5_1[0] = __fmaf_rn(part_hi_7_1[0], w_c_31, acc_hi_5_1[0]);
                                acc_4_1[1] = __fmaf_rn(part_6_1[1], w_c_31, acc_4_1[1]);
                                acc_hi_5_1[1] = __fmaf_rn(part_hi_7_1[1], w_c_31, acc_hi_5_1[1]);
                                acc_4_1[2] = __fmaf_rn(part_6_1[2], w_c_31, acc_4_1[2]);
                                acc_hi_5_1[2] = __fmaf_rn(part_hi_7_1[2], w_c_31, acc_hi_5_1[2]);
                                acc_4_1[3] = __fmaf_rn(part_6_1[3], w_c_31, acc_4_1[3]);
                                acc_hi_5_1[3] = __fmaf_rn(part_hi_7_1[3], w_c_31, acc_hi_5_1[3]);
                                acc_4_1[4] = __fmaf_rn(part_6_1[4], w_c_31, acc_4_1[4]);
                                acc_hi_5_1[4] = __fmaf_rn(part_hi_7_1[4], w_c_31, acc_hi_5_1[4]);
                                acc_4_1[5] = __fmaf_rn(part_6_1[5], w_c_31, acc_4_1[5]);
                                acc_hi_5_1[5] = __fmaf_rn(part_hi_7_1[5], w_c_31, acc_hi_5_1[5]);
                                acc_4_1[6] = __fmaf_rn(part_6_1[6], w_c_31, acc_4_1[6]);
                                acc_hi_5_1[6] = __fmaf_rn(part_hi_7_1[6], w_c_31, acc_hi_5_1[6]);
                                acc_4_1[7] = __fmaf_rn(part_6_1[7], w_c_31, acc_4_1[7]);
                                acc_hi_5_1[7] = __fmaf_rn(part_hi_7_1[7], w_c_31, acc_hi_5_1[7]);
                                acc_4_1[8] = __fmaf_rn(part_6_1[8], w_c_31, acc_4_1[8]);
                                acc_hi_5_1[8] = __fmaf_rn(part_hi_7_1[8], w_c_31, acc_hi_5_1[8]);
                                acc_4_1[9] = __fmaf_rn(part_6_1[9], w_c_31, acc_4_1[9]);
                                acc_hi_5_1[9] = __fmaf_rn(part_hi_7_1[9], w_c_31, acc_hi_5_1[9]);
                                acc_4_1[10] = __fmaf_rn(part_6_1[10], w_c_31, acc_4_1[10]);
                                acc_hi_5_1[10] = __fmaf_rn(part_hi_7_1[10], w_c_31, acc_hi_5_1[10]);
                                acc_4_1[11] = __fmaf_rn(part_6_1[11], w_c_31, acc_4_1[11]);
                                acc_hi_5_1[11] = __fmaf_rn(part_hi_7_1[11], w_c_31, acc_hi_5_1[11]);
                                acc_4_1[12] = __fmaf_rn(part_6_1[12], w_c_31, acc_4_1[12]);
                                acc_hi_5_1[12] = __fmaf_rn(part_hi_7_1[12], w_c_31, acc_hi_5_1[12]);
                                acc_4_1[13] = __fmaf_rn(part_6_1[13], w_c_31, acc_4_1[13]);
                                acc_hi_5_1[13] = __fmaf_rn(part_hi_7_1[13], w_c_31, acc_hi_5_1[13]);
                                acc_4_1[14] = __fmaf_rn(part_6_1[14], w_c_31, acc_4_1[14]);
                                acc_hi_5_1[14] = __fmaf_rn(part_hi_7_1[14], w_c_31, acc_hi_5_1[14]);
                                acc_4_1[15] = __fmaf_rn(part_6_1[15], w_c_31, acc_4_1[15]);
                                acc_hi_5_1[15] = __fmaf_rn(part_hi_7_1[15], w_c_31, acc_hi_5_1[15]);
                            }
                            const float2 _scale2_179 = {final_scale, final_scale};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(acc_4_1)[_ls], _scale2_179);
                            unsigned int acc_4_bf16_1[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_4_1[_lp*2 + 0], acc_4_1[_lp*2+1 + 0]));
                                acc_4_bf16_1[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o10_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + 64 ^ ((warp % 4 * 32 + lane) * 128 + 64 >> 7 & 7) << 4))), "r"(acc_4_bf16_1[0]), "r"(acc_4_bf16_1[1]), "r"(acc_4_bf16_1[2]), "r"(acc_4_bf16_1[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o10_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + 80 ^ ((warp % 4 * 32 + lane) * 128 + 80 >> 7 & 7) << 4))), "r"(acc_4_bf16_1[4]), "r"(acc_4_bf16_1[5]), "r"(acc_4_bf16_1[6]), "r"(acc_4_bf16_1[7]) : "memory");
                            const float2 _scale2_180 = {final_scale, final_scale};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(acc_hi_5_1)[_ls], _scale2_180);
                            unsigned int acc_hi_5_bf16_1[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_hi_5_1[_lp*2 + 0], acc_hi_5_1[_lp*2+1 + 0]));
                                acc_hi_5_bf16_1[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o11_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + 64 ^ ((warp % 4 * 32 + lane) * 128 + 64 >> 7 & 7) << 4))), "r"(acc_hi_5_bf16_1[0]), "r"(acc_hi_5_bf16_1[1]), "r"(acc_hi_5_bf16_1[2]), "r"(acc_hi_5_bf16_1[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o11_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + 80 ^ ((warp % 4 * 32 + lane) * 128 + 80 >> 7 & 7) << 4))), "r"(acc_hi_5_bf16_1[4]), "r"(acc_hi_5_bf16_1[5]), "r"(acc_hi_5_bf16_1[6]), "r"(acc_hi_5_bf16_1[7]) : "memory");
                            float acc_8_1[16];
                            float acc_hi_9_1[16];
                            acc_8_1[0] = 0.0f;
                            acc_hi_9_1[0] = 0.0f;
                            acc_8_1[1] = 0.0f;
                            acc_hi_9_1[1] = 0.0f;
                            acc_8_1[2] = 0.0f;
                            acc_hi_9_1[2] = 0.0f;
                            acc_8_1[3] = 0.0f;
                            acc_hi_9_1[3] = 0.0f;
                            acc_8_1[4] = 0.0f;
                            acc_hi_9_1[4] = 0.0f;
                            acc_8_1[5] = 0.0f;
                            acc_hi_9_1[5] = 0.0f;
                            acc_8_1[6] = 0.0f;
                            acc_hi_9_1[6] = 0.0f;
                            acc_8_1[7] = 0.0f;
                            acc_hi_9_1[7] = 0.0f;
                            acc_8_1[8] = 0.0f;
                            acc_hi_9_1[8] = 0.0f;
                            acc_8_1[9] = 0.0f;
                            acc_hi_9_1[9] = 0.0f;
                            acc_8_1[10] = 0.0f;
                            acc_hi_9_1[10] = 0.0f;
                            acc_8_1[11] = 0.0f;
                            acc_hi_9_1[11] = 0.0f;
                            acc_8_1[12] = 0.0f;
                            acc_hi_9_1[12] = 0.0f;
                            acc_8_1[13] = 0.0f;
                            acc_hi_9_1[13] = 0.0f;
                            acc_8_1[14] = 0.0f;
                            acc_hi_9_1[14] = 0.0f;
                            acc_8_1[15] = 0.0f;
                            acc_hi_9_1[15] = 0.0f;
                            float part_10_1[16];
                            float part_hi_11_1[16];
                            if (n_chunks_u_0 > 0) {
                                unsigned int crow_28 = tile_slot0 * 512 + part_row;
                                float w_c_32 = approx_exp2(part_ml[crow_28 * 2] - row_max_all);
                                unsigned int off_0_28 = crow_28 * 128 + 48;
                                {
                                    unsigned _ldv8_181_0;
                                    unsigned _ldv8_181_1;
                                    unsigned _ldv8_181_2;
                                    unsigned _ldv8_181_3;
                                    unsigned _ldv8_181_4;
                                    unsigned _ldv8_181_5;
                                    unsigned _ldv8_181_6;
                                    unsigned _ldv8_181_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_181_0), "=r"(_ldv8_181_1), "=r"(_ldv8_181_2), "=r"(_ldv8_181_3), "=r"(_ldv8_181_4), "=r"(_ldv8_181_5), "=r"(_ldv8_181_6), "=r"(_ldv8_181_7) : "l"((const void*)(part_o + (off_0_28))) : "memory");
                                    part_10_1[0 + 0] = __uint_as_float(_ldv8_181_0);
                                    part_10_1[0 + 1] = __uint_as_float(_ldv8_181_1);
                                    part_10_1[0 + 2] = __uint_as_float(_ldv8_181_2);
                                    part_10_1[0 + 3] = __uint_as_float(_ldv8_181_3);
                                    part_10_1[0 + 4] = __uint_as_float(_ldv8_181_4);
                                    part_10_1[0 + 5] = __uint_as_float(_ldv8_181_5);
                                    part_10_1[0 + 6] = __uint_as_float(_ldv8_181_6);
                                    part_10_1[0 + 7] = __uint_as_float(_ldv8_181_7);
                                }
                                {
                                    unsigned _ldv8_182_0;
                                    unsigned _ldv8_182_1;
                                    unsigned _ldv8_182_2;
                                    unsigned _ldv8_182_3;
                                    unsigned _ldv8_182_4;
                                    unsigned _ldv8_182_5;
                                    unsigned _ldv8_182_6;
                                    unsigned _ldv8_182_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_182_0), "=r"(_ldv8_182_1), "=r"(_ldv8_182_2), "=r"(_ldv8_182_3), "=r"(_ldv8_182_4), "=r"(_ldv8_182_5), "=r"(_ldv8_182_6), "=r"(_ldv8_182_7) : "l"((const void*)(part_o + (off_0_28 + 8))) : "memory");
                                    part_10_1[8 + 0] = __uint_as_float(_ldv8_182_0);
                                    part_10_1[8 + 1] = __uint_as_float(_ldv8_182_1);
                                    part_10_1[8 + 2] = __uint_as_float(_ldv8_182_2);
                                    part_10_1[8 + 3] = __uint_as_float(_ldv8_182_3);
                                    part_10_1[8 + 4] = __uint_as_float(_ldv8_182_4);
                                    part_10_1[8 + 5] = __uint_as_float(_ldv8_182_5);
                                    part_10_1[8 + 6] = __uint_as_float(_ldv8_182_6);
                                    part_10_1[8 + 7] = __uint_as_float(_ldv8_182_7);
                                }
                                {
                                    unsigned _ldv8_183_0;
                                    unsigned _ldv8_183_1;
                                    unsigned _ldv8_183_2;
                                    unsigned _ldv8_183_3;
                                    unsigned _ldv8_183_4;
                                    unsigned _ldv8_183_5;
                                    unsigned _ldv8_183_6;
                                    unsigned _ldv8_183_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_183_0), "=r"(_ldv8_183_1), "=r"(_ldv8_183_2), "=r"(_ldv8_183_3), "=r"(_ldv8_183_4), "=r"(_ldv8_183_5), "=r"(_ldv8_183_6), "=r"(_ldv8_183_7) : "l"((const void*)(part_o + (off_0_28 + 64))) : "memory");
                                    part_hi_11_1[0 + 0] = __uint_as_float(_ldv8_183_0);
                                    part_hi_11_1[0 + 1] = __uint_as_float(_ldv8_183_1);
                                    part_hi_11_1[0 + 2] = __uint_as_float(_ldv8_183_2);
                                    part_hi_11_1[0 + 3] = __uint_as_float(_ldv8_183_3);
                                    part_hi_11_1[0 + 4] = __uint_as_float(_ldv8_183_4);
                                    part_hi_11_1[0 + 5] = __uint_as_float(_ldv8_183_5);
                                    part_hi_11_1[0 + 6] = __uint_as_float(_ldv8_183_6);
                                    part_hi_11_1[0 + 7] = __uint_as_float(_ldv8_183_7);
                                }
                                {
                                    unsigned _ldv8_184_0;
                                    unsigned _ldv8_184_1;
                                    unsigned _ldv8_184_2;
                                    unsigned _ldv8_184_3;
                                    unsigned _ldv8_184_4;
                                    unsigned _ldv8_184_5;
                                    unsigned _ldv8_184_6;
                                    unsigned _ldv8_184_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_184_0), "=r"(_ldv8_184_1), "=r"(_ldv8_184_2), "=r"(_ldv8_184_3), "=r"(_ldv8_184_4), "=r"(_ldv8_184_5), "=r"(_ldv8_184_6), "=r"(_ldv8_184_7) : "l"((const void*)(part_o + (off_0_28 + 64 + 8))) : "memory");
                                    part_hi_11_1[8 + 0] = __uint_as_float(_ldv8_184_0);
                                    part_hi_11_1[8 + 1] = __uint_as_float(_ldv8_184_1);
                                    part_hi_11_1[8 + 2] = __uint_as_float(_ldv8_184_2);
                                    part_hi_11_1[8 + 3] = __uint_as_float(_ldv8_184_3);
                                    part_hi_11_1[8 + 4] = __uint_as_float(_ldv8_184_4);
                                    part_hi_11_1[8 + 5] = __uint_as_float(_ldv8_184_5);
                                    part_hi_11_1[8 + 6] = __uint_as_float(_ldv8_184_6);
                                    part_hi_11_1[8 + 7] = __uint_as_float(_ldv8_184_7);
                                }
                                acc_8_1[0] = __fmaf_rn(part_10_1[0], w_c_32, acc_8_1[0]);
                                acc_hi_9_1[0] = __fmaf_rn(part_hi_11_1[0], w_c_32, acc_hi_9_1[0]);
                                acc_8_1[1] = __fmaf_rn(part_10_1[1], w_c_32, acc_8_1[1]);
                                acc_hi_9_1[1] = __fmaf_rn(part_hi_11_1[1], w_c_32, acc_hi_9_1[1]);
                                acc_8_1[2] = __fmaf_rn(part_10_1[2], w_c_32, acc_8_1[2]);
                                acc_hi_9_1[2] = __fmaf_rn(part_hi_11_1[2], w_c_32, acc_hi_9_1[2]);
                                acc_8_1[3] = __fmaf_rn(part_10_1[3], w_c_32, acc_8_1[3]);
                                acc_hi_9_1[3] = __fmaf_rn(part_hi_11_1[3], w_c_32, acc_hi_9_1[3]);
                                acc_8_1[4] = __fmaf_rn(part_10_1[4], w_c_32, acc_8_1[4]);
                                acc_hi_9_1[4] = __fmaf_rn(part_hi_11_1[4], w_c_32, acc_hi_9_1[4]);
                                acc_8_1[5] = __fmaf_rn(part_10_1[5], w_c_32, acc_8_1[5]);
                                acc_hi_9_1[5] = __fmaf_rn(part_hi_11_1[5], w_c_32, acc_hi_9_1[5]);
                                acc_8_1[6] = __fmaf_rn(part_10_1[6], w_c_32, acc_8_1[6]);
                                acc_hi_9_1[6] = __fmaf_rn(part_hi_11_1[6], w_c_32, acc_hi_9_1[6]);
                                acc_8_1[7] = __fmaf_rn(part_10_1[7], w_c_32, acc_8_1[7]);
                                acc_hi_9_1[7] = __fmaf_rn(part_hi_11_1[7], w_c_32, acc_hi_9_1[7]);
                                acc_8_1[8] = __fmaf_rn(part_10_1[8], w_c_32, acc_8_1[8]);
                                acc_hi_9_1[8] = __fmaf_rn(part_hi_11_1[8], w_c_32, acc_hi_9_1[8]);
                                acc_8_1[9] = __fmaf_rn(part_10_1[9], w_c_32, acc_8_1[9]);
                                acc_hi_9_1[9] = __fmaf_rn(part_hi_11_1[9], w_c_32, acc_hi_9_1[9]);
                                acc_8_1[10] = __fmaf_rn(part_10_1[10], w_c_32, acc_8_1[10]);
                                acc_hi_9_1[10] = __fmaf_rn(part_hi_11_1[10], w_c_32, acc_hi_9_1[10]);
                                acc_8_1[11] = __fmaf_rn(part_10_1[11], w_c_32, acc_8_1[11]);
                                acc_hi_9_1[11] = __fmaf_rn(part_hi_11_1[11], w_c_32, acc_hi_9_1[11]);
                                acc_8_1[12] = __fmaf_rn(part_10_1[12], w_c_32, acc_8_1[12]);
                                acc_hi_9_1[12] = __fmaf_rn(part_hi_11_1[12], w_c_32, acc_hi_9_1[12]);
                                acc_8_1[13] = __fmaf_rn(part_10_1[13], w_c_32, acc_8_1[13]);
                                acc_hi_9_1[13] = __fmaf_rn(part_hi_11_1[13], w_c_32, acc_hi_9_1[13]);
                                acc_8_1[14] = __fmaf_rn(part_10_1[14], w_c_32, acc_8_1[14]);
                                acc_hi_9_1[14] = __fmaf_rn(part_hi_11_1[14], w_c_32, acc_hi_9_1[14]);
                                acc_8_1[15] = __fmaf_rn(part_10_1[15], w_c_32, acc_8_1[15]);
                                acc_hi_9_1[15] = __fmaf_rn(part_hi_11_1[15], w_c_32, acc_hi_9_1[15]);
                            }
                            if (n_chunks_u_0 > 1) {
                                unsigned int crow_29 = (tile_slot0 + 1) * 512 + part_row;
                                float w_c_33 = approx_exp2(part_ml[crow_29 * 2] - row_max_all);
                                unsigned int off_0_29 = crow_29 * 128 + 48;
                                {
                                    unsigned _ldv8_185_0;
                                    unsigned _ldv8_185_1;
                                    unsigned _ldv8_185_2;
                                    unsigned _ldv8_185_3;
                                    unsigned _ldv8_185_4;
                                    unsigned _ldv8_185_5;
                                    unsigned _ldv8_185_6;
                                    unsigned _ldv8_185_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_185_0), "=r"(_ldv8_185_1), "=r"(_ldv8_185_2), "=r"(_ldv8_185_3), "=r"(_ldv8_185_4), "=r"(_ldv8_185_5), "=r"(_ldv8_185_6), "=r"(_ldv8_185_7) : "l"((const void*)(part_o + (off_0_29))) : "memory");
                                    part_10_1[0 + 0] = __uint_as_float(_ldv8_185_0);
                                    part_10_1[0 + 1] = __uint_as_float(_ldv8_185_1);
                                    part_10_1[0 + 2] = __uint_as_float(_ldv8_185_2);
                                    part_10_1[0 + 3] = __uint_as_float(_ldv8_185_3);
                                    part_10_1[0 + 4] = __uint_as_float(_ldv8_185_4);
                                    part_10_1[0 + 5] = __uint_as_float(_ldv8_185_5);
                                    part_10_1[0 + 6] = __uint_as_float(_ldv8_185_6);
                                    part_10_1[0 + 7] = __uint_as_float(_ldv8_185_7);
                                }
                                {
                                    unsigned _ldv8_186_0;
                                    unsigned _ldv8_186_1;
                                    unsigned _ldv8_186_2;
                                    unsigned _ldv8_186_3;
                                    unsigned _ldv8_186_4;
                                    unsigned _ldv8_186_5;
                                    unsigned _ldv8_186_6;
                                    unsigned _ldv8_186_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_186_0), "=r"(_ldv8_186_1), "=r"(_ldv8_186_2), "=r"(_ldv8_186_3), "=r"(_ldv8_186_4), "=r"(_ldv8_186_5), "=r"(_ldv8_186_6), "=r"(_ldv8_186_7) : "l"((const void*)(part_o + (off_0_29 + 8))) : "memory");
                                    part_10_1[8 + 0] = __uint_as_float(_ldv8_186_0);
                                    part_10_1[8 + 1] = __uint_as_float(_ldv8_186_1);
                                    part_10_1[8 + 2] = __uint_as_float(_ldv8_186_2);
                                    part_10_1[8 + 3] = __uint_as_float(_ldv8_186_3);
                                    part_10_1[8 + 4] = __uint_as_float(_ldv8_186_4);
                                    part_10_1[8 + 5] = __uint_as_float(_ldv8_186_5);
                                    part_10_1[8 + 6] = __uint_as_float(_ldv8_186_6);
                                    part_10_1[8 + 7] = __uint_as_float(_ldv8_186_7);
                                }
                                {
                                    unsigned _ldv8_187_0;
                                    unsigned _ldv8_187_1;
                                    unsigned _ldv8_187_2;
                                    unsigned _ldv8_187_3;
                                    unsigned _ldv8_187_4;
                                    unsigned _ldv8_187_5;
                                    unsigned _ldv8_187_6;
                                    unsigned _ldv8_187_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_187_0), "=r"(_ldv8_187_1), "=r"(_ldv8_187_2), "=r"(_ldv8_187_3), "=r"(_ldv8_187_4), "=r"(_ldv8_187_5), "=r"(_ldv8_187_6), "=r"(_ldv8_187_7) : "l"((const void*)(part_o + (off_0_29 + 64))) : "memory");
                                    part_hi_11_1[0 + 0] = __uint_as_float(_ldv8_187_0);
                                    part_hi_11_1[0 + 1] = __uint_as_float(_ldv8_187_1);
                                    part_hi_11_1[0 + 2] = __uint_as_float(_ldv8_187_2);
                                    part_hi_11_1[0 + 3] = __uint_as_float(_ldv8_187_3);
                                    part_hi_11_1[0 + 4] = __uint_as_float(_ldv8_187_4);
                                    part_hi_11_1[0 + 5] = __uint_as_float(_ldv8_187_5);
                                    part_hi_11_1[0 + 6] = __uint_as_float(_ldv8_187_6);
                                    part_hi_11_1[0 + 7] = __uint_as_float(_ldv8_187_7);
                                }
                                {
                                    unsigned _ldv8_188_0;
                                    unsigned _ldv8_188_1;
                                    unsigned _ldv8_188_2;
                                    unsigned _ldv8_188_3;
                                    unsigned _ldv8_188_4;
                                    unsigned _ldv8_188_5;
                                    unsigned _ldv8_188_6;
                                    unsigned _ldv8_188_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_188_0), "=r"(_ldv8_188_1), "=r"(_ldv8_188_2), "=r"(_ldv8_188_3), "=r"(_ldv8_188_4), "=r"(_ldv8_188_5), "=r"(_ldv8_188_6), "=r"(_ldv8_188_7) : "l"((const void*)(part_o + (off_0_29 + 64 + 8))) : "memory");
                                    part_hi_11_1[8 + 0] = __uint_as_float(_ldv8_188_0);
                                    part_hi_11_1[8 + 1] = __uint_as_float(_ldv8_188_1);
                                    part_hi_11_1[8 + 2] = __uint_as_float(_ldv8_188_2);
                                    part_hi_11_1[8 + 3] = __uint_as_float(_ldv8_188_3);
                                    part_hi_11_1[8 + 4] = __uint_as_float(_ldv8_188_4);
                                    part_hi_11_1[8 + 5] = __uint_as_float(_ldv8_188_5);
                                    part_hi_11_1[8 + 6] = __uint_as_float(_ldv8_188_6);
                                    part_hi_11_1[8 + 7] = __uint_as_float(_ldv8_188_7);
                                }
                                acc_8_1[0] = __fmaf_rn(part_10_1[0], w_c_33, acc_8_1[0]);
                                acc_hi_9_1[0] = __fmaf_rn(part_hi_11_1[0], w_c_33, acc_hi_9_1[0]);
                                acc_8_1[1] = __fmaf_rn(part_10_1[1], w_c_33, acc_8_1[1]);
                                acc_hi_9_1[1] = __fmaf_rn(part_hi_11_1[1], w_c_33, acc_hi_9_1[1]);
                                acc_8_1[2] = __fmaf_rn(part_10_1[2], w_c_33, acc_8_1[2]);
                                acc_hi_9_1[2] = __fmaf_rn(part_hi_11_1[2], w_c_33, acc_hi_9_1[2]);
                                acc_8_1[3] = __fmaf_rn(part_10_1[3], w_c_33, acc_8_1[3]);
                                acc_hi_9_1[3] = __fmaf_rn(part_hi_11_1[3], w_c_33, acc_hi_9_1[3]);
                                acc_8_1[4] = __fmaf_rn(part_10_1[4], w_c_33, acc_8_1[4]);
                                acc_hi_9_1[4] = __fmaf_rn(part_hi_11_1[4], w_c_33, acc_hi_9_1[4]);
                                acc_8_1[5] = __fmaf_rn(part_10_1[5], w_c_33, acc_8_1[5]);
                                acc_hi_9_1[5] = __fmaf_rn(part_hi_11_1[5], w_c_33, acc_hi_9_1[5]);
                                acc_8_1[6] = __fmaf_rn(part_10_1[6], w_c_33, acc_8_1[6]);
                                acc_hi_9_1[6] = __fmaf_rn(part_hi_11_1[6], w_c_33, acc_hi_9_1[6]);
                                acc_8_1[7] = __fmaf_rn(part_10_1[7], w_c_33, acc_8_1[7]);
                                acc_hi_9_1[7] = __fmaf_rn(part_hi_11_1[7], w_c_33, acc_hi_9_1[7]);
                                acc_8_1[8] = __fmaf_rn(part_10_1[8], w_c_33, acc_8_1[8]);
                                acc_hi_9_1[8] = __fmaf_rn(part_hi_11_1[8], w_c_33, acc_hi_9_1[8]);
                                acc_8_1[9] = __fmaf_rn(part_10_1[9], w_c_33, acc_8_1[9]);
                                acc_hi_9_1[9] = __fmaf_rn(part_hi_11_1[9], w_c_33, acc_hi_9_1[9]);
                                acc_8_1[10] = __fmaf_rn(part_10_1[10], w_c_33, acc_8_1[10]);
                                acc_hi_9_1[10] = __fmaf_rn(part_hi_11_1[10], w_c_33, acc_hi_9_1[10]);
                                acc_8_1[11] = __fmaf_rn(part_10_1[11], w_c_33, acc_8_1[11]);
                                acc_hi_9_1[11] = __fmaf_rn(part_hi_11_1[11], w_c_33, acc_hi_9_1[11]);
                                acc_8_1[12] = __fmaf_rn(part_10_1[12], w_c_33, acc_8_1[12]);
                                acc_hi_9_1[12] = __fmaf_rn(part_hi_11_1[12], w_c_33, acc_hi_9_1[12]);
                                acc_8_1[13] = __fmaf_rn(part_10_1[13], w_c_33, acc_8_1[13]);
                                acc_hi_9_1[13] = __fmaf_rn(part_hi_11_1[13], w_c_33, acc_hi_9_1[13]);
                                acc_8_1[14] = __fmaf_rn(part_10_1[14], w_c_33, acc_8_1[14]);
                                acc_hi_9_1[14] = __fmaf_rn(part_hi_11_1[14], w_c_33, acc_hi_9_1[14]);
                                acc_8_1[15] = __fmaf_rn(part_10_1[15], w_c_33, acc_8_1[15]);
                                acc_hi_9_1[15] = __fmaf_rn(part_hi_11_1[15], w_c_33, acc_hi_9_1[15]);
                            }
                            if (n_chunks_u_0 > 2) {
                                unsigned int crow_30 = (tile_slot0 + 2) * 512 + part_row;
                                float w_c_34 = approx_exp2(part_ml[crow_30 * 2] - row_max_all);
                                unsigned int off_0_30 = crow_30 * 128 + 48;
                                {
                                    unsigned _ldv8_189_0;
                                    unsigned _ldv8_189_1;
                                    unsigned _ldv8_189_2;
                                    unsigned _ldv8_189_3;
                                    unsigned _ldv8_189_4;
                                    unsigned _ldv8_189_5;
                                    unsigned _ldv8_189_6;
                                    unsigned _ldv8_189_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_189_0), "=r"(_ldv8_189_1), "=r"(_ldv8_189_2), "=r"(_ldv8_189_3), "=r"(_ldv8_189_4), "=r"(_ldv8_189_5), "=r"(_ldv8_189_6), "=r"(_ldv8_189_7) : "l"((const void*)(part_o + (off_0_30))) : "memory");
                                    part_10_1[0 + 0] = __uint_as_float(_ldv8_189_0);
                                    part_10_1[0 + 1] = __uint_as_float(_ldv8_189_1);
                                    part_10_1[0 + 2] = __uint_as_float(_ldv8_189_2);
                                    part_10_1[0 + 3] = __uint_as_float(_ldv8_189_3);
                                    part_10_1[0 + 4] = __uint_as_float(_ldv8_189_4);
                                    part_10_1[0 + 5] = __uint_as_float(_ldv8_189_5);
                                    part_10_1[0 + 6] = __uint_as_float(_ldv8_189_6);
                                    part_10_1[0 + 7] = __uint_as_float(_ldv8_189_7);
                                }
                                {
                                    unsigned _ldv8_190_0;
                                    unsigned _ldv8_190_1;
                                    unsigned _ldv8_190_2;
                                    unsigned _ldv8_190_3;
                                    unsigned _ldv8_190_4;
                                    unsigned _ldv8_190_5;
                                    unsigned _ldv8_190_6;
                                    unsigned _ldv8_190_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_190_0), "=r"(_ldv8_190_1), "=r"(_ldv8_190_2), "=r"(_ldv8_190_3), "=r"(_ldv8_190_4), "=r"(_ldv8_190_5), "=r"(_ldv8_190_6), "=r"(_ldv8_190_7) : "l"((const void*)(part_o + (off_0_30 + 8))) : "memory");
                                    part_10_1[8 + 0] = __uint_as_float(_ldv8_190_0);
                                    part_10_1[8 + 1] = __uint_as_float(_ldv8_190_1);
                                    part_10_1[8 + 2] = __uint_as_float(_ldv8_190_2);
                                    part_10_1[8 + 3] = __uint_as_float(_ldv8_190_3);
                                    part_10_1[8 + 4] = __uint_as_float(_ldv8_190_4);
                                    part_10_1[8 + 5] = __uint_as_float(_ldv8_190_5);
                                    part_10_1[8 + 6] = __uint_as_float(_ldv8_190_6);
                                    part_10_1[8 + 7] = __uint_as_float(_ldv8_190_7);
                                }
                                {
                                    unsigned _ldv8_191_0;
                                    unsigned _ldv8_191_1;
                                    unsigned _ldv8_191_2;
                                    unsigned _ldv8_191_3;
                                    unsigned _ldv8_191_4;
                                    unsigned _ldv8_191_5;
                                    unsigned _ldv8_191_6;
                                    unsigned _ldv8_191_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_191_0), "=r"(_ldv8_191_1), "=r"(_ldv8_191_2), "=r"(_ldv8_191_3), "=r"(_ldv8_191_4), "=r"(_ldv8_191_5), "=r"(_ldv8_191_6), "=r"(_ldv8_191_7) : "l"((const void*)(part_o + (off_0_30 + 64))) : "memory");
                                    part_hi_11_1[0 + 0] = __uint_as_float(_ldv8_191_0);
                                    part_hi_11_1[0 + 1] = __uint_as_float(_ldv8_191_1);
                                    part_hi_11_1[0 + 2] = __uint_as_float(_ldv8_191_2);
                                    part_hi_11_1[0 + 3] = __uint_as_float(_ldv8_191_3);
                                    part_hi_11_1[0 + 4] = __uint_as_float(_ldv8_191_4);
                                    part_hi_11_1[0 + 5] = __uint_as_float(_ldv8_191_5);
                                    part_hi_11_1[0 + 6] = __uint_as_float(_ldv8_191_6);
                                    part_hi_11_1[0 + 7] = __uint_as_float(_ldv8_191_7);
                                }
                                {
                                    unsigned _ldv8_192_0;
                                    unsigned _ldv8_192_1;
                                    unsigned _ldv8_192_2;
                                    unsigned _ldv8_192_3;
                                    unsigned _ldv8_192_4;
                                    unsigned _ldv8_192_5;
                                    unsigned _ldv8_192_6;
                                    unsigned _ldv8_192_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_192_0), "=r"(_ldv8_192_1), "=r"(_ldv8_192_2), "=r"(_ldv8_192_3), "=r"(_ldv8_192_4), "=r"(_ldv8_192_5), "=r"(_ldv8_192_6), "=r"(_ldv8_192_7) : "l"((const void*)(part_o + (off_0_30 + 64 + 8))) : "memory");
                                    part_hi_11_1[8 + 0] = __uint_as_float(_ldv8_192_0);
                                    part_hi_11_1[8 + 1] = __uint_as_float(_ldv8_192_1);
                                    part_hi_11_1[8 + 2] = __uint_as_float(_ldv8_192_2);
                                    part_hi_11_1[8 + 3] = __uint_as_float(_ldv8_192_3);
                                    part_hi_11_1[8 + 4] = __uint_as_float(_ldv8_192_4);
                                    part_hi_11_1[8 + 5] = __uint_as_float(_ldv8_192_5);
                                    part_hi_11_1[8 + 6] = __uint_as_float(_ldv8_192_6);
                                    part_hi_11_1[8 + 7] = __uint_as_float(_ldv8_192_7);
                                }
                                acc_8_1[0] = __fmaf_rn(part_10_1[0], w_c_34, acc_8_1[0]);
                                acc_hi_9_1[0] = __fmaf_rn(part_hi_11_1[0], w_c_34, acc_hi_9_1[0]);
                                acc_8_1[1] = __fmaf_rn(part_10_1[1], w_c_34, acc_8_1[1]);
                                acc_hi_9_1[1] = __fmaf_rn(part_hi_11_1[1], w_c_34, acc_hi_9_1[1]);
                                acc_8_1[2] = __fmaf_rn(part_10_1[2], w_c_34, acc_8_1[2]);
                                acc_hi_9_1[2] = __fmaf_rn(part_hi_11_1[2], w_c_34, acc_hi_9_1[2]);
                                acc_8_1[3] = __fmaf_rn(part_10_1[3], w_c_34, acc_8_1[3]);
                                acc_hi_9_1[3] = __fmaf_rn(part_hi_11_1[3], w_c_34, acc_hi_9_1[3]);
                                acc_8_1[4] = __fmaf_rn(part_10_1[4], w_c_34, acc_8_1[4]);
                                acc_hi_9_1[4] = __fmaf_rn(part_hi_11_1[4], w_c_34, acc_hi_9_1[4]);
                                acc_8_1[5] = __fmaf_rn(part_10_1[5], w_c_34, acc_8_1[5]);
                                acc_hi_9_1[5] = __fmaf_rn(part_hi_11_1[5], w_c_34, acc_hi_9_1[5]);
                                acc_8_1[6] = __fmaf_rn(part_10_1[6], w_c_34, acc_8_1[6]);
                                acc_hi_9_1[6] = __fmaf_rn(part_hi_11_1[6], w_c_34, acc_hi_9_1[6]);
                                acc_8_1[7] = __fmaf_rn(part_10_1[7], w_c_34, acc_8_1[7]);
                                acc_hi_9_1[7] = __fmaf_rn(part_hi_11_1[7], w_c_34, acc_hi_9_1[7]);
                                acc_8_1[8] = __fmaf_rn(part_10_1[8], w_c_34, acc_8_1[8]);
                                acc_hi_9_1[8] = __fmaf_rn(part_hi_11_1[8], w_c_34, acc_hi_9_1[8]);
                                acc_8_1[9] = __fmaf_rn(part_10_1[9], w_c_34, acc_8_1[9]);
                                acc_hi_9_1[9] = __fmaf_rn(part_hi_11_1[9], w_c_34, acc_hi_9_1[9]);
                                acc_8_1[10] = __fmaf_rn(part_10_1[10], w_c_34, acc_8_1[10]);
                                acc_hi_9_1[10] = __fmaf_rn(part_hi_11_1[10], w_c_34, acc_hi_9_1[10]);
                                acc_8_1[11] = __fmaf_rn(part_10_1[11], w_c_34, acc_8_1[11]);
                                acc_hi_9_1[11] = __fmaf_rn(part_hi_11_1[11], w_c_34, acc_hi_9_1[11]);
                                acc_8_1[12] = __fmaf_rn(part_10_1[12], w_c_34, acc_8_1[12]);
                                acc_hi_9_1[12] = __fmaf_rn(part_hi_11_1[12], w_c_34, acc_hi_9_1[12]);
                                acc_8_1[13] = __fmaf_rn(part_10_1[13], w_c_34, acc_8_1[13]);
                                acc_hi_9_1[13] = __fmaf_rn(part_hi_11_1[13], w_c_34, acc_hi_9_1[13]);
                                acc_8_1[14] = __fmaf_rn(part_10_1[14], w_c_34, acc_8_1[14]);
                                acc_hi_9_1[14] = __fmaf_rn(part_hi_11_1[14], w_c_34, acc_hi_9_1[14]);
                                acc_8_1[15] = __fmaf_rn(part_10_1[15], w_c_34, acc_8_1[15]);
                                acc_hi_9_1[15] = __fmaf_rn(part_hi_11_1[15], w_c_34, acc_hi_9_1[15]);
                            }
                            if (n_chunks_u_0 > 3) {
                                unsigned int crow_31 = (tile_slot0 + 3) * 512 + part_row;
                                float w_c_35 = approx_exp2(part_ml[crow_31 * 2] - row_max_all);
                                unsigned int off_0_31 = crow_31 * 128 + 48;
                                {
                                    unsigned _ldv8_193_0;
                                    unsigned _ldv8_193_1;
                                    unsigned _ldv8_193_2;
                                    unsigned _ldv8_193_3;
                                    unsigned _ldv8_193_4;
                                    unsigned _ldv8_193_5;
                                    unsigned _ldv8_193_6;
                                    unsigned _ldv8_193_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_193_0), "=r"(_ldv8_193_1), "=r"(_ldv8_193_2), "=r"(_ldv8_193_3), "=r"(_ldv8_193_4), "=r"(_ldv8_193_5), "=r"(_ldv8_193_6), "=r"(_ldv8_193_7) : "l"((const void*)(part_o + (off_0_31))) : "memory");
                                    part_10_1[0 + 0] = __uint_as_float(_ldv8_193_0);
                                    part_10_1[0 + 1] = __uint_as_float(_ldv8_193_1);
                                    part_10_1[0 + 2] = __uint_as_float(_ldv8_193_2);
                                    part_10_1[0 + 3] = __uint_as_float(_ldv8_193_3);
                                    part_10_1[0 + 4] = __uint_as_float(_ldv8_193_4);
                                    part_10_1[0 + 5] = __uint_as_float(_ldv8_193_5);
                                    part_10_1[0 + 6] = __uint_as_float(_ldv8_193_6);
                                    part_10_1[0 + 7] = __uint_as_float(_ldv8_193_7);
                                }
                                {
                                    unsigned _ldv8_194_0;
                                    unsigned _ldv8_194_1;
                                    unsigned _ldv8_194_2;
                                    unsigned _ldv8_194_3;
                                    unsigned _ldv8_194_4;
                                    unsigned _ldv8_194_5;
                                    unsigned _ldv8_194_6;
                                    unsigned _ldv8_194_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_194_0), "=r"(_ldv8_194_1), "=r"(_ldv8_194_2), "=r"(_ldv8_194_3), "=r"(_ldv8_194_4), "=r"(_ldv8_194_5), "=r"(_ldv8_194_6), "=r"(_ldv8_194_7) : "l"((const void*)(part_o + (off_0_31 + 8))) : "memory");
                                    part_10_1[8 + 0] = __uint_as_float(_ldv8_194_0);
                                    part_10_1[8 + 1] = __uint_as_float(_ldv8_194_1);
                                    part_10_1[8 + 2] = __uint_as_float(_ldv8_194_2);
                                    part_10_1[8 + 3] = __uint_as_float(_ldv8_194_3);
                                    part_10_1[8 + 4] = __uint_as_float(_ldv8_194_4);
                                    part_10_1[8 + 5] = __uint_as_float(_ldv8_194_5);
                                    part_10_1[8 + 6] = __uint_as_float(_ldv8_194_6);
                                    part_10_1[8 + 7] = __uint_as_float(_ldv8_194_7);
                                }
                                {
                                    unsigned _ldv8_195_0;
                                    unsigned _ldv8_195_1;
                                    unsigned _ldv8_195_2;
                                    unsigned _ldv8_195_3;
                                    unsigned _ldv8_195_4;
                                    unsigned _ldv8_195_5;
                                    unsigned _ldv8_195_6;
                                    unsigned _ldv8_195_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_195_0), "=r"(_ldv8_195_1), "=r"(_ldv8_195_2), "=r"(_ldv8_195_3), "=r"(_ldv8_195_4), "=r"(_ldv8_195_5), "=r"(_ldv8_195_6), "=r"(_ldv8_195_7) : "l"((const void*)(part_o + (off_0_31 + 64))) : "memory");
                                    part_hi_11_1[0 + 0] = __uint_as_float(_ldv8_195_0);
                                    part_hi_11_1[0 + 1] = __uint_as_float(_ldv8_195_1);
                                    part_hi_11_1[0 + 2] = __uint_as_float(_ldv8_195_2);
                                    part_hi_11_1[0 + 3] = __uint_as_float(_ldv8_195_3);
                                    part_hi_11_1[0 + 4] = __uint_as_float(_ldv8_195_4);
                                    part_hi_11_1[0 + 5] = __uint_as_float(_ldv8_195_5);
                                    part_hi_11_1[0 + 6] = __uint_as_float(_ldv8_195_6);
                                    part_hi_11_1[0 + 7] = __uint_as_float(_ldv8_195_7);
                                }
                                {
                                    unsigned _ldv8_196_0;
                                    unsigned _ldv8_196_1;
                                    unsigned _ldv8_196_2;
                                    unsigned _ldv8_196_3;
                                    unsigned _ldv8_196_4;
                                    unsigned _ldv8_196_5;
                                    unsigned _ldv8_196_6;
                                    unsigned _ldv8_196_7;
                                    asm volatile(
                                        "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                        : "=r"(_ldv8_196_0), "=r"(_ldv8_196_1), "=r"(_ldv8_196_2), "=r"(_ldv8_196_3), "=r"(_ldv8_196_4), "=r"(_ldv8_196_5), "=r"(_ldv8_196_6), "=r"(_ldv8_196_7) : "l"((const void*)(part_o + (off_0_31 + 64 + 8))) : "memory");
                                    part_hi_11_1[8 + 0] = __uint_as_float(_ldv8_196_0);
                                    part_hi_11_1[8 + 1] = __uint_as_float(_ldv8_196_1);
                                    part_hi_11_1[8 + 2] = __uint_as_float(_ldv8_196_2);
                                    part_hi_11_1[8 + 3] = __uint_as_float(_ldv8_196_3);
                                    part_hi_11_1[8 + 4] = __uint_as_float(_ldv8_196_4);
                                    part_hi_11_1[8 + 5] = __uint_as_float(_ldv8_196_5);
                                    part_hi_11_1[8 + 6] = __uint_as_float(_ldv8_196_6);
                                    part_hi_11_1[8 + 7] = __uint_as_float(_ldv8_196_7);
                                }
                                acc_8_1[0] = __fmaf_rn(part_10_1[0], w_c_35, acc_8_1[0]);
                                acc_hi_9_1[0] = __fmaf_rn(part_hi_11_1[0], w_c_35, acc_hi_9_1[0]);
                                acc_8_1[1] = __fmaf_rn(part_10_1[1], w_c_35, acc_8_1[1]);
                                acc_hi_9_1[1] = __fmaf_rn(part_hi_11_1[1], w_c_35, acc_hi_9_1[1]);
                                acc_8_1[2] = __fmaf_rn(part_10_1[2], w_c_35, acc_8_1[2]);
                                acc_hi_9_1[2] = __fmaf_rn(part_hi_11_1[2], w_c_35, acc_hi_9_1[2]);
                                acc_8_1[3] = __fmaf_rn(part_10_1[3], w_c_35, acc_8_1[3]);
                                acc_hi_9_1[3] = __fmaf_rn(part_hi_11_1[3], w_c_35, acc_hi_9_1[3]);
                                acc_8_1[4] = __fmaf_rn(part_10_1[4], w_c_35, acc_8_1[4]);
                                acc_hi_9_1[4] = __fmaf_rn(part_hi_11_1[4], w_c_35, acc_hi_9_1[4]);
                                acc_8_1[5] = __fmaf_rn(part_10_1[5], w_c_35, acc_8_1[5]);
                                acc_hi_9_1[5] = __fmaf_rn(part_hi_11_1[5], w_c_35, acc_hi_9_1[5]);
                                acc_8_1[6] = __fmaf_rn(part_10_1[6], w_c_35, acc_8_1[6]);
                                acc_hi_9_1[6] = __fmaf_rn(part_hi_11_1[6], w_c_35, acc_hi_9_1[6]);
                                acc_8_1[7] = __fmaf_rn(part_10_1[7], w_c_35, acc_8_1[7]);
                                acc_hi_9_1[7] = __fmaf_rn(part_hi_11_1[7], w_c_35, acc_hi_9_1[7]);
                                acc_8_1[8] = __fmaf_rn(part_10_1[8], w_c_35, acc_8_1[8]);
                                acc_hi_9_1[8] = __fmaf_rn(part_hi_11_1[8], w_c_35, acc_hi_9_1[8]);
                                acc_8_1[9] = __fmaf_rn(part_10_1[9], w_c_35, acc_8_1[9]);
                                acc_hi_9_1[9] = __fmaf_rn(part_hi_11_1[9], w_c_35, acc_hi_9_1[9]);
                                acc_8_1[10] = __fmaf_rn(part_10_1[10], w_c_35, acc_8_1[10]);
                                acc_hi_9_1[10] = __fmaf_rn(part_hi_11_1[10], w_c_35, acc_hi_9_1[10]);
                                acc_8_1[11] = __fmaf_rn(part_10_1[11], w_c_35, acc_8_1[11]);
                                acc_hi_9_1[11] = __fmaf_rn(part_hi_11_1[11], w_c_35, acc_hi_9_1[11]);
                                acc_8_1[12] = __fmaf_rn(part_10_1[12], w_c_35, acc_8_1[12]);
                                acc_hi_9_1[12] = __fmaf_rn(part_hi_11_1[12], w_c_35, acc_hi_9_1[12]);
                                acc_8_1[13] = __fmaf_rn(part_10_1[13], w_c_35, acc_8_1[13]);
                                acc_hi_9_1[13] = __fmaf_rn(part_hi_11_1[13], w_c_35, acc_hi_9_1[13]);
                                acc_8_1[14] = __fmaf_rn(part_10_1[14], w_c_35, acc_8_1[14]);
                                acc_hi_9_1[14] = __fmaf_rn(part_hi_11_1[14], w_c_35, acc_hi_9_1[14]);
                                acc_8_1[15] = __fmaf_rn(part_10_1[15], w_c_35, acc_8_1[15]);
                                acc_hi_9_1[15] = __fmaf_rn(part_hi_11_1[15], w_c_35, acc_hi_9_1[15]);
                            }
                            const float2 _scale2_197 = {final_scale, final_scale};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(acc_8_1)[_ls], _scale2_197);
                            unsigned int acc_8_bf16_1[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_8_1[_lp*2 + 0], acc_8_1[_lp*2+1 + 0]));
                                acc_8_bf16_1[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o10_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + 96 ^ ((warp % 4 * 32 + lane) * 128 + 96 >> 7 & 7) << 4))), "r"(acc_8_bf16_1[0]), "r"(acc_8_bf16_1[1]), "r"(acc_8_bf16_1[2]), "r"(acc_8_bf16_1[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o10_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + 112 ^ ((warp % 4 * 32 + lane) * 128 + 112 >> 7 & 7) << 4))), "r"(acc_8_bf16_1[4]), "r"(acc_8_bf16_1[5]), "r"(acc_8_bf16_1[6]), "r"(acc_8_bf16_1[7]) : "memory");
                            const float2 _scale2_198 = {final_scale, final_scale};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(acc_hi_9_1)[_ls], _scale2_198);
                            unsigned int acc_hi_9_bf16_1[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(acc_hi_9_1[_lp*2 + 0], acc_hi_9_1[_lp*2+1 + 0]));
                                acc_hi_9_bf16_1[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o11_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + 96 ^ ((warp % 4 * 32 + lane) * 128 + 96 >> 7 & 7) << 4))), "r"(acc_hi_9_bf16_1[0]), "r"(acc_hi_9_bf16_1[1]), "r"(acc_hi_9_bf16_1[2]), "r"(acc_hi_9_bf16_1[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o11_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + 112 ^ ((warp % 4 * 32 + lane) * 128 + 112 >> 7 & 7) << 4))), "r"(acc_hi_9_bf16_1[4]), "r"(acc_hi_9_bf16_1[5]), "r"(acc_hi_9_bf16_1[6]), "r"(acc_hi_9_bf16_1[7]) : "memory");
                            asm volatile("fence.proxy.async;" ::: "memory");
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            asm volatile("barrier.sync 2, 128;" ::: "memory");
                            if (warp == 4) {
                                if (elect_sync()) {
                                    tma_store_4d((&O), 0, out_row_1, out_head_1, out_batch_1, smem_o10_addr);
                                    tma_store_4d((&O), 64, out_row_1, out_head_1, out_batch_1, smem_o11_addr);
                                }
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                        }
                    }
                }
            }
            if (stage == 0) {
                if (warp == 0) {
                    asm volatile("cp.async.bulk.wait_group.read 0;");
                }
                asm volatile("barrier.sync 3, 128;" ::: "memory");
            } else {
                if (warp == 4) {
                    asm volatile("cp.async.bulk.wait_group.read 0;");
                }
                asm volatile("barrier.sync 4, 128;" ::: "memory");
            }
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
    }
    // ---- Role: correction_lo ----
    if (warp >= 8 && warp <= 9) {
        { // correction_lo_main
            unsigned int total_tiles_1 = (seqlen_q + 512 - 1) / 512 * total_bh;
            unsigned int n_full_u_1 = (unsigned int)n_full_tiles;
            unsigned int n_items_1 = n_full_u_1 + (total_tiles_1 - n_full_u_1) * (unsigned int)n_chunks;
            int owned_row_base = make_warp_uniform(warp % 2 * 32);
            unsigned int _phase_p_empty_0 = 1;
            unsigned int _phase_p_empty_1 = 1;
            unsigned int _phase_corr_sig_lo_0 = 0;
            unsigned int _phase_corr_sig_lo_1 = 0;
            unsigned int _phase_o_full_0 = 0;
            unsigned int _phase_o_full_1 = 0;
            #pragma unroll 1
            for (unsigned int tile_idx_2 = cluster_id; tile_idx_2 < n_items_1; tile_idx_2 += num_clusters) {
                unsigned int nkb_all_1 = (seqlen_kv + 128 - 1) / 128;
                unsigned int n_full_u_0_1 = (unsigned int)n_full_tiles;
                unsigned int n_chunks_u_1 = (unsigned int)n_chunks;
                unsigned int _max_0_1 = ((tile_idx_2) > (n_full_u_0_1) ? (tile_idx_2) : (n_full_u_0_1));
                unsigned int split_idx_1 = _max_0_1 - n_full_u_0_1;
                unsigned int chunk_1 = split_idx_1 - split_idx_1 / n_chunks_u_1 * n_chunks_u_1;
                unsigned int _min_1_1 = ((tile_idx_2) < (n_full_u_0_1) ? (tile_idx_2) : (n_full_u_0_1));
                unsigned int tile_idx_1_1 = _min_1_1 + split_idx_1 / n_chunks_u_1;
                unsigned int _min_2_1 = ((tile_idx_2) < (n_full_u_0_1) ? (tile_idx_2) : (n_full_u_0_1));
                unsigned int whole_1 = ((n_full_u_0_1 - _min_2_1) < (1) ? (n_full_u_0_1 - _min_2_1) : (1));
                unsigned int n_lo_1 = chunk_1 * nkb_all_1 / n_chunks_u_1;
                unsigned int n_hi_1 = (((chunk_1 + 1) * nkb_all_1 / n_chunks_u_1) > (nkb_all_1 * whole_1) ? ((chunk_1 + 1) * nkb_all_1 / n_chunks_u_1) : (nkb_all_1 * whole_1));
                unsigned int cluster_m_block_1 = tile_idx_1_1 % (unsigned int)((seqlen_q + 512 - 1) / 512);
                unsigned int bh_1 = tile_idx_1_1 / (unsigned int)((seqlen_q + 512 - 1) / 512);
                unsigned int m_block_1 = cluster_m_block_1 * 4 + (unsigned int)(cta_rank * 2);
                unsigned int num_n_blocks_1 = n_hi_1 - n_lo_1;
                mbarrier_wait(p_empty_addr, _phase_p_empty_0);
                _phase_p_empty_0 ^= 1;
                mbarrier_wait(p_empty_addr + 8, _phase_p_empty_1);
                _phase_p_empty_1 ^= 1;
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((p_full_addr) & 0xFEFFFFFF) : "memory");
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((p_full_addr + 8) & 0xFEFFFFFF) : "memory");
                mbarrier_wait(corr_sig_lo_addr, _phase_corr_sig_lo_0);
                _phase_corr_sig_lo_0 ^= 1;
                mbarrier_arrive(corr_done_addr);
                mbarrier_wait(corr_sig_lo_addr + 8, _phase_corr_sig_lo_1);
                _phase_corr_sig_lo_1 ^= 1;
                mbarrier_arrive(corr_done_addr + 8);
                #pragma unroll 1
                for (unsigned int n_iter = 1; n_iter < num_n_blocks_1; n_iter++) {
                    mbarrier_wait(corr_sig_lo_addr, _phase_corr_sig_lo_0);
                    _phase_corr_sig_lo_0 ^= 1;
                    mbarrier_wait(p_empty_addr, _phase_p_empty_0);
                    _phase_p_empty_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int logical_row = owned_row_base + lane;
                    int tmem_row_base = make_warp_uniform(owned_row_base << 16);
                    float scale = row_state[logical_row];
                    int skip_rescale = __all_sync(0xFFFFFFFF, scale == 1.0f);
                    if (skip_rescale == 0) {
                        #pragma unroll
                        for (int col = 0; col < 8; col++) {
                            int addr = make_warp_uniform(taddr + (unsigned int)128 + (unsigned int)tmem_row_base + (unsigned int)(col * 16));
                            float _tmem_load_24[16];
                            tmem_ld_x16(&_tmem_load_24[0], addr);
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            const float2 _scale2_3 = {scale, scale};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_24)[_ls], _scale2_3);
                            tmem_st_x16_f32(addr, _tmem_load_24);
                        }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((p_full_addr) & 0xFEFFFFFF) : "memory");
                    mbarrier_arrive(corr_done_addr);
                    mbarrier_wait(corr_sig_lo_addr + 8, _phase_corr_sig_lo_1);
                    _phase_corr_sig_lo_1 ^= 1;
                    mbarrier_wait(p_empty_addr + 8, _phase_p_empty_1);
                    _phase_p_empty_1 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int logical_row_0 = owned_row_base + lane;
                    int tmem_row_base_1 = make_warp_uniform(owned_row_base << 16);
                    float scale_2 = row_state[logical_row_0 + 128];
                    int skip_rescale_3 = __all_sync(0xFFFFFFFF, scale_2 == 1.0f);
                    if (skip_rescale_3 == 0) {
                        #pragma unroll
                        for (int col_1 = 0; col_1 < 8; col_1++) {
                            int addr_1 = make_warp_uniform(taddr + (unsigned int)256 + (unsigned int)tmem_row_base_1 + (unsigned int)(col_1 * 16));
                            float _tmem_load_25[16];
                            tmem_ld_x16(&_tmem_load_25[0], addr_1);
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            const float2 _scale2_4 = {scale_2, scale_2};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_25)[_ls], _scale2_4);
                            tmem_st_x16_f32(addr_1, _tmem_load_25);
                        }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((p_full_addr + 8) & 0xFEFFFFFF) : "memory");
                    mbarrier_arrive(corr_done_addr + 8);
                }
                mbarrier_wait(o_full_addr, _phase_o_full_0);
                _phase_o_full_0 ^= 1;
                mbarrier_wait(o_full_addr + 8, _phase_o_full_1);
                _phase_o_full_1 ^= 1;
            }
        }
    }
    // ---- Role: correction_hi ----
    if (warp >= 10 && warp <= 11) {
        { // correction_hi_main
            unsigned int total_tiles_2 = (seqlen_q + 512 - 1) / 512 * total_bh;
            unsigned int n_full_u_2 = (unsigned int)n_full_tiles;
            unsigned int n_items_2 = n_full_u_2 + (total_tiles_2 - n_full_u_2) * (unsigned int)n_chunks;
            int owned_row_base_1 = make_warp_uniform(64 + warp % 2 * 32);
            unsigned int _phase_p_empty_0_1 = 1;
            unsigned int _phase_p_empty_1_1 = 1;
            unsigned int _phase_corr_sig_hi_0 = 0;
            unsigned int _phase_corr_sig_hi_1 = 0;
            unsigned int _phase_o_full_0_1 = 0;
            unsigned int _phase_o_full_1_1 = 0;
            #pragma unroll 1
            for (unsigned int tile_idx_3 = cluster_id; tile_idx_3 < n_items_2; tile_idx_3 += num_clusters) {
                unsigned int nkb_all_2 = (seqlen_kv + 128 - 1) / 128;
                unsigned int n_full_u_0_2 = (unsigned int)n_full_tiles;
                unsigned int n_chunks_u_2 = (unsigned int)n_chunks;
                unsigned int _max_0_2 = ((tile_idx_3) > (n_full_u_0_2) ? (tile_idx_3) : (n_full_u_0_2));
                unsigned int split_idx_2 = _max_0_2 - n_full_u_0_2;
                unsigned int chunk_2 = split_idx_2 - split_idx_2 / n_chunks_u_2 * n_chunks_u_2;
                unsigned int _min_1_2 = ((tile_idx_3) < (n_full_u_0_2) ? (tile_idx_3) : (n_full_u_0_2));
                unsigned int tile_idx_1_2 = _min_1_2 + split_idx_2 / n_chunks_u_2;
                unsigned int _min_2_2 = ((tile_idx_3) < (n_full_u_0_2) ? (tile_idx_3) : (n_full_u_0_2));
                unsigned int whole_2 = ((n_full_u_0_2 - _min_2_2) < (1) ? (n_full_u_0_2 - _min_2_2) : (1));
                unsigned int n_lo_2 = chunk_2 * nkb_all_2 / n_chunks_u_2;
                unsigned int n_hi_2 = (((chunk_2 + 1) * nkb_all_2 / n_chunks_u_2) > (nkb_all_2 * whole_2) ? ((chunk_2 + 1) * nkb_all_2 / n_chunks_u_2) : (nkb_all_2 * whole_2));
                unsigned int cluster_m_block_2 = tile_idx_1_2 % (unsigned int)((seqlen_q + 512 - 1) / 512);
                unsigned int bh_2 = tile_idx_1_2 / (unsigned int)((seqlen_q + 512 - 1) / 512);
                unsigned int m_block_2 = cluster_m_block_2 * 4 + (unsigned int)(cta_rank * 2);
                unsigned int num_n_blocks_2 = n_hi_2 - n_lo_2;
                mbarrier_wait(p_empty_addr, _phase_p_empty_0_1);
                _phase_p_empty_0_1 ^= 1;
                mbarrier_wait(p_empty_addr + 8, _phase_p_empty_1_1);
                _phase_p_empty_1_1 ^= 1;
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((p_full_addr) & 0xFEFFFFFF) : "memory");
                asm volatile(
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                    :: "r"((p_full_addr + 8) & 0xFEFFFFFF) : "memory");
                mbarrier_wait(corr_sig_hi_addr, _phase_corr_sig_hi_0);
                _phase_corr_sig_hi_0 ^= 1;
                mbarrier_arrive(corr_done_addr);
                mbarrier_wait(corr_sig_hi_addr + 8, _phase_corr_sig_hi_1);
                _phase_corr_sig_hi_1 ^= 1;
                mbarrier_arrive(corr_done_addr + 8);
                #pragma unroll 1
                for (unsigned int n_iter_1 = 1; n_iter_1 < num_n_blocks_2; n_iter_1++) {
                    mbarrier_wait(corr_sig_hi_addr, _phase_corr_sig_hi_0);
                    _phase_corr_sig_hi_0 ^= 1;
                    mbarrier_wait(p_empty_addr, _phase_p_empty_0_1);
                    _phase_p_empty_0_1 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int logical_row_1 = owned_row_base_1 + lane;
                    int tmem_row_base_2 = make_warp_uniform(owned_row_base_1 << 16);
                    float scale_1 = row_state[logical_row_1];
                    int skip_rescale_1 = __all_sync(0xFFFFFFFF, scale_1 == 1.0f);
                    if (skip_rescale_1 == 0) {
                        #pragma unroll
                        for (int col_2 = 0; col_2 < 8; col_2++) {
                            int addr_2 = make_warp_uniform(taddr + (unsigned int)128 + (unsigned int)tmem_row_base_2 + (unsigned int)(col_2 * 16));
                            float _tmem_load_26[16];
                            tmem_ld_x16(&_tmem_load_26[0], addr_2);
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            const float2 _scale2_3 = {scale_1, scale_1};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_26)[_ls], _scale2_3);
                            tmem_st_x16_f32(addr_2, _tmem_load_26);
                        }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((p_full_addr) & 0xFEFFFFFF) : "memory");
                    mbarrier_arrive(corr_done_addr);
                    mbarrier_wait(corr_sig_hi_addr + 8, _phase_corr_sig_hi_1);
                    _phase_corr_sig_hi_1 ^= 1;
                    mbarrier_wait(p_empty_addr + 8, _phase_p_empty_1_1);
                    _phase_p_empty_1_1 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int logical_row_0_1 = owned_row_base_1 + lane;
                    int tmem_row_base_1_1 = make_warp_uniform(owned_row_base_1 << 16);
                    float scale_2_1 = row_state[logical_row_0_1 + 128];
                    int skip_rescale_3_1 = __all_sync(0xFFFFFFFF, scale_2_1 == 1.0f);
                    if (skip_rescale_3_1 == 0) {
                        #pragma unroll
                        for (int col_3 = 0; col_3 < 8; col_3++) {
                            int addr_3 = make_warp_uniform(taddr + (unsigned int)256 + (unsigned int)tmem_row_base_1_1 + (unsigned int)(col_3 * 16));
                            float _tmem_load_27[16];
                            tmem_ld_x16(&_tmem_load_27[0], addr_3);
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            const float2 _scale2_4 = {scale_2_1, scale_2_1};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_27)[_ls], _scale2_4);
                            tmem_st_x16_f32(addr_3, _tmem_load_27);
                        }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((p_full_addr + 8) & 0xFEFFFFFF) : "memory");
                    mbarrier_arrive(corr_done_addr + 8);
                }
                mbarrier_wait(o_full_addr, _phase_o_full_0_1);
                _phase_o_full_0_1 ^= 1;
                mbarrier_wait(o_full_addr + 8, _phase_o_full_1_1);
                _phase_o_full_1_1 ^= 1;
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 12) {
        { // mma_main
            unsigned int _phase_q_full_0 = 0;
            unsigned int _phase_q_full_1 = 0;
            unsigned int _phase_s_empty_0 = 0;
            unsigned int _phase_s_empty_1 = 0;
            unsigned int _phase_p_full_0 = 0;
            unsigned int _phase_p_full_1 = 0;
            unsigned int _phase_p_full_2_0 = 0;
            unsigned int _phase_p_full_2_1 = 0;
            if (cta_rank == 0) {
                unsigned int total_tiles_3 = (seqlen_q + 512 - 1) / 512 * total_bh;
                unsigned int n_full_u_3 = (unsigned int)n_full_tiles;
                unsigned int n_items_3 = n_full_u_3 + (total_tiles_3 - n_full_u_3) * (unsigned int)n_chunks;
                unsigned int k_stage = 0;
                unsigned int k_phase = 0;
                unsigned int v_stage = 0;
                unsigned int v_phase = 0;
                #pragma unroll 1
                for (unsigned int tile_idx_4 = cluster_id; tile_idx_4 < n_items_3; tile_idx_4 += num_clusters) {
                    unsigned int nkb_all_3 = (seqlen_kv + 128 - 1) / 128;
                    unsigned int n_full_u_0_3 = (unsigned int)n_full_tiles;
                    unsigned int n_chunks_u_3 = (unsigned int)n_chunks;
                    unsigned int _max_0_3 = ((tile_idx_4) > (n_full_u_0_3) ? (tile_idx_4) : (n_full_u_0_3));
                    unsigned int split_idx_3 = _max_0_3 - n_full_u_0_3;
                    unsigned int chunk_3 = split_idx_3 - split_idx_3 / n_chunks_u_3 * n_chunks_u_3;
                    unsigned int _min_1_3 = ((tile_idx_4) < (n_full_u_0_3) ? (tile_idx_4) : (n_full_u_0_3));
                    unsigned int tile_idx_1_3 = _min_1_3 + split_idx_3 / n_chunks_u_3;
                    unsigned int _min_2_3 = ((tile_idx_4) < (n_full_u_0_3) ? (tile_idx_4) : (n_full_u_0_3));
                    unsigned int whole_3 = ((n_full_u_0_3 - _min_2_3) < (1) ? (n_full_u_0_3 - _min_2_3) : (1));
                    unsigned int n_lo_3 = chunk_3 * nkb_all_3 / n_chunks_u_3;
                    unsigned int n_hi_3 = (((chunk_3 + 1) * nkb_all_3 / n_chunks_u_3) > (nkb_all_3 * whole_3) ? ((chunk_3 + 1) * nkb_all_3 / n_chunks_u_3) : (nkb_all_3 * whole_3));
                    unsigned int cluster_m_block_3 = tile_idx_1_3 % (unsigned int)((seqlen_q + 512 - 1) / 512);
                    unsigned int bh_3 = tile_idx_1_3 / (unsigned int)((seqlen_q + 512 - 1) / 512);
                    unsigned int m_block_3 = cluster_m_block_3 * 4 + (unsigned int)(cta_rank * 2);
                    unsigned int num_n_blocks_3 = n_hi_3 - n_lo_3;
                    unsigned int sel0 = (m_block_3 / 4 & 1) * 2;
                    unsigned int sel1 = sel0 + 1;
                    mbarrier_wait(q_full_addr, _phase_q_full_0);
                    _phase_q_full_0 ^= 1;
                    mbarrier_wait(q_full_addr + 8, _phase_q_full_1);
                    _phase_q_full_1 ^= 1;
                    mbarrier_wait(k_full_addr + (k_stage) * 8, k_phase);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa_qk0, make_sf_cp_desc_lo_sbo256((((smem_sfq_addr) >> 4))));
                        tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfa_qk0 + 4), make_sf_cp_desc_lo_sbo256((((smem_sfq_addr) >> 4) + 8)));
                    }
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa_ql0, make_sf_cp_desc_lo_sbo256((((smem_sfql_addr) >> 4))));
                        tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfa_ql0 + 4), make_sf_cp_desc_lo_sbo256((((smem_sfql_addr) >> 4) + 8)));
                    }
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb_qk0, make_sf_cp_desc_lo_sbo256((((smem_sfk_addr) >> 4) + (k_stage) * 64)));
                        tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb_qk0 + 4), make_sf_cp_desc_lo_sbo256((((smem_sfk_addr) >> 4) + (k_stage) * 64 + 8)));
                    }
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb_qkl, make_sf_cp_desc_lo_sbo256((((smem_sfkl_addr) >> 4) + (k_stage) * 64)));
                        tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb_qkl + 4), make_sf_cp_desc_lo_sbo256((((smem_sfkl_addr) >> 4) + (k_stage) * 64 + 8)));
                    }
                    int _mma_a_lo_0 = (((smem_sel_addr) >> 4) & 0x3FFF) + (sel0) * 256;
                    int _mma_b_lo_0 = (((smem_tab_addr) >> 4) & 0x3FFF) + (k_stage) * 128;
                    mma_ss_step_cg2(_mma_a_lo_0, _mma_b_lo_0, tmem_scores, 270533776, 0, 0xC0004010U, 0xC0004010U);
                    int _mma_a_lo_1 = (((smem_q_addr) >> 4) & 0x3FFF) + (0) * 512;
                    int _mma_b_lo_1 = (((smem_k_addr) >> 4) & 0x3FFF) + (k_stage) * 256;
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x80004020 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x80004020 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 0, b_desc + 0,
                                0x10200480U, tmem_tmem_sfa_qk0 + 0, tmem_tmem_sfb_qk0 + 0, 1);
                            tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 2, b_desc + 2,
                                0x10200480U, tmem_tmem_sfa_qk0 + 4, tmem_tmem_sfb_qk0 + 4, 1);
                        }
                    }
                    int _mma_a_lo_2 = (((smem_ql_addr) >> 4) & 0x3FFF) + (0) * 512;
                    int _mma_b_lo_2 = (((smem_k_addr) >> 4) & 0x3FFF) + (k_stage) * 256;
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_2) | ((uint64_t)0x80004020 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_2) | ((uint64_t)0x80004020 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 0, b_desc + 0,
                                0x10200480U, tmem_tmem_sfa_ql0 + 0, tmem_tmem_sfb_qk0 + 0, 1);
                            tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 2, b_desc + 2,
                                0x10200480U, tmem_tmem_sfa_ql0 + 4, tmem_tmem_sfb_qk0 + 4, 1);
                        }
                    }
                    int _mma_a_lo_3 = (((smem_q_addr) >> 4) & 0x3FFF) + (0) * 512;
                    int _mma_b_lo_3 = (((smem_kl_addr) >> 4) & 0x3FFF) + (k_stage) * 256;
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_3) | ((uint64_t)0x80004020 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_3) | ((uint64_t)0x80004020 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 0, b_desc + 0,
                                0x10200480U, tmem_tmem_sfa_qk0 + 0, tmem_tmem_sfb_qkl + 0, 1);
                            tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 2, b_desc + 2,
                                0x10200480U, tmem_tmem_sfa_qk0 + 4, tmem_tmem_sfb_qkl + 4, 1);
                        }
                    }
                    elect_commit_cg2_multicast(s_full_addr, (uint16_t)(3));
                    mbarrier_wait(s_empty_addr, _phase_s_empty_0);
                    _phase_s_empty_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa_qk1, make_sf_cp_desc_lo_sbo256((((smem_sfq_addr) >> 4) + 64)));
                        tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfa_qk1 + 4), make_sf_cp_desc_lo_sbo256((((smem_sfq_addr) >> 4) + 64 + 8)));
                    }
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa_ql1, make_sf_cp_desc_lo_sbo256((((smem_sfql_addr) >> 4) + 64)));
                        tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfa_ql1 + 4), make_sf_cp_desc_lo_sbo256((((smem_sfql_addr) >> 4) + 64 + 8)));
                    }
                    int _mma_a_lo_4 = (((smem_sel_addr) >> 4) & 0x3FFF) + (sel1) * 256;
                    int _mma_b_lo_4 = (((smem_tab_addr) >> 4) & 0x3FFF) + (k_stage) * 128;
                    mma_ss_step_cg2(_mma_a_lo_4, _mma_b_lo_4, tmem_scores, 270533776, 0, 0xC0004010U, 0xC0004010U);
                    int _mma_a_lo_5 = (((smem_q_addr) >> 4) & 0x3FFF) + (1) * 512;
                    int _mma_b_lo_5 = (((smem_k_addr) >> 4) & 0x3FFF) + (k_stage) * 256;
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_5) | ((uint64_t)0x80004020 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_5) | ((uint64_t)0x80004020 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 0, b_desc + 0,
                                0x10200480U, tmem_tmem_sfa_qk1 + 0, tmem_tmem_sfb_qk0 + 0, 1);
                            tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 2, b_desc + 2,
                                0x10200480U, tmem_tmem_sfa_qk1 + 4, tmem_tmem_sfb_qk0 + 4, 1);
                        }
                    }
                    int _mma_a_lo_6 = (((smem_ql_addr) >> 4) & 0x3FFF) + (1) * 512;
                    int _mma_b_lo_6 = (((smem_k_addr) >> 4) & 0x3FFF) + (k_stage) * 256;
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_6) | ((uint64_t)0x80004020 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_6) | ((uint64_t)0x80004020 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 0, b_desc + 0,
                                0x10200480U, tmem_tmem_sfa_ql1 + 0, tmem_tmem_sfb_qk0 + 0, 1);
                            tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 2, b_desc + 2,
                                0x10200480U, tmem_tmem_sfa_ql1 + 4, tmem_tmem_sfb_qk0 + 4, 1);
                        }
                    }
                    int _mma_a_lo_7 = (((smem_q_addr) >> 4) & 0x3FFF) + (1) * 512;
                    int _mma_b_lo_7 = (((smem_kl_addr) >> 4) & 0x3FFF) + (k_stage) * 256;
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_7) | ((uint64_t)0x80004020 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_7) | ((uint64_t)0x80004020 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 0, b_desc + 0,
                                0x10200480U, tmem_tmem_sfa_qk1 + 0, tmem_tmem_sfb_qkl + 0, 1);
                            tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 2, b_desc + 2,
                                0x10200480U, tmem_tmem_sfa_qk1 + 4, tmem_tmem_sfb_qkl + 4, 1);
                        }
                    }
                    elect_commit_cg2_multicast(s_full_addr + 8, (uint16_t)(3));
                    elect_commit_cg2_multicast(k_empty_addr + (k_stage) * 8, (uint16_t)(3));
                    k_stage += 1;
                    if (k_stage == 4) { k_stage = 0; k_phase ^= 1; }
                    unsigned int first_pv = 1;
                    #pragma unroll 1
                    for (unsigned int n_iter_2 = 0; n_iter_2 < num_n_blocks_3 - 1; n_iter_2++) {
                        int first_pv_flag = first_pv;
                        mbarrier_wait(v_full_addr + (v_stage) * 8, v_phase);
                        mbarrier_wait(k_full_addr + (k_stage) * 8, k_phase);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb_qk0, make_sf_cp_desc_lo_sbo256((((smem_sfk_addr) >> 4) + (k_stage) * 64)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb_qk0 + 4), make_sf_cp_desc_lo_sbo256((((smem_sfk_addr) >> 4) + (k_stage) * 64 + 8)));
                        }
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb_qkl, make_sf_cp_desc_lo_sbo256((((smem_sfkl_addr) >> 4) + (k_stage) * 64)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb_qkl + 4), make_sf_cp_desc_lo_sbo256((((smem_sfkl_addr) >> 4) + (k_stage) * 64 + 8)));
                        }
                        mbarrier_wait(s_empty_addr + 8, _phase_s_empty_1);
                        _phase_s_empty_1 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_8 = (((smem_sel_addr) >> 4) & 0x3FFF) + (sel0) * 256;
                        int _mma_b_lo_8 = (((smem_tab_addr) >> 4) & 0x3FFF) + (k_stage) * 128;
                        mma_ss_step_cg2(_mma_a_lo_8, _mma_b_lo_8, tmem_scores, 270533776, 0, 0xC0004010U, 0xC0004010U);
                        int _mma_a_lo_9 = (((smem_q_addr) >> 4) & 0x3FFF) + (0) * 512;
                        int _mma_b_lo_9 = (((smem_k_addr) >> 4) & 0x3FFF) + (k_stage) * 256;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_9) | ((uint64_t)0x80004020 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_9) | ((uint64_t)0x80004020 << 32);

                                tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 0, b_desc + 0,
                                    0x10200480U, tmem_tmem_sfa_qk0 + 0, tmem_tmem_sfb_qk0 + 0, 1);
                                tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 2, b_desc + 2,
                                    0x10200480U, tmem_tmem_sfa_qk0 + 4, tmem_tmem_sfb_qk0 + 4, 1);
                            }
                        }
                        int _mma_a_lo_10 = (((smem_ql_addr) >> 4) & 0x3FFF) + (0) * 512;
                        int _mma_b_lo_10 = (((smem_k_addr) >> 4) & 0x3FFF) + (k_stage) * 256;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_10) | ((uint64_t)0x80004020 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_10) | ((uint64_t)0x80004020 << 32);

                                tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 0, b_desc + 0,
                                    0x10200480U, tmem_tmem_sfa_ql0 + 0, tmem_tmem_sfb_qk0 + 0, 1);
                                tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 2, b_desc + 2,
                                    0x10200480U, tmem_tmem_sfa_ql0 + 4, tmem_tmem_sfb_qk0 + 4, 1);
                            }
                        }
                        int _mma_a_lo_11 = (((smem_q_addr) >> 4) & 0x3FFF) + (0) * 512;
                        int _mma_b_lo_11 = (((smem_kl_addr) >> 4) & 0x3FFF) + (k_stage) * 256;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_11) | ((uint64_t)0x80004020 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_11) | ((uint64_t)0x80004020 << 32);

                                tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 0, b_desc + 0,
                                    0x10200480U, tmem_tmem_sfa_qk0 + 0, tmem_tmem_sfb_qkl + 0, 1);
                                tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 2, b_desc + 2,
                                    0x10200480U, tmem_tmem_sfa_qk0 + 4, tmem_tmem_sfb_qkl + 4, 1);
                            }
                        }
                        elect_commit_cg2_multicast(s_full_addr, (uint16_t)(3));
                        mbarrier_wait(s_empty_addr, _phase_s_empty_0);
                        _phase_s_empty_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_12 = (((smem_sel_addr) >> 4) & 0x3FFF) + (sel1) * 256;
                        int _mma_b_lo_12 = (((smem_tab_addr) >> 4) & 0x3FFF) + (k_stage) * 128;
                        mma_ss_step_cg2(_mma_a_lo_12, _mma_b_lo_12, tmem_scores, 270533776, 0, 0xC0004010U, 0xC0004010U);
                        int _mma_a_lo_13 = (((smem_q_addr) >> 4) & 0x3FFF) + (1) * 512;
                        int _mma_b_lo_13 = (((smem_k_addr) >> 4) & 0x3FFF) + (k_stage) * 256;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_13) | ((uint64_t)0x80004020 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_13) | ((uint64_t)0x80004020 << 32);

                                tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 0, b_desc + 0,
                                    0x10200480U, tmem_tmem_sfa_qk1 + 0, tmem_tmem_sfb_qk0 + 0, 1);
                                tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 2, b_desc + 2,
                                    0x10200480U, tmem_tmem_sfa_qk1 + 4, tmem_tmem_sfb_qk0 + 4, 1);
                            }
                        }
                        int _mma_a_lo_14 = (((smem_ql_addr) >> 4) & 0x3FFF) + (1) * 512;
                        int _mma_b_lo_14 = (((smem_k_addr) >> 4) & 0x3FFF) + (k_stage) * 256;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_14) | ((uint64_t)0x80004020 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_14) | ((uint64_t)0x80004020 << 32);

                                tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 0, b_desc + 0,
                                    0x10200480U, tmem_tmem_sfa_ql1 + 0, tmem_tmem_sfb_qk0 + 0, 1);
                                tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 2, b_desc + 2,
                                    0x10200480U, tmem_tmem_sfa_ql1 + 4, tmem_tmem_sfb_qk0 + 4, 1);
                            }
                        }
                        int _mma_a_lo_15 = (((smem_q_addr) >> 4) & 0x3FFF) + (1) * 512;
                        int _mma_b_lo_15 = (((smem_kl_addr) >> 4) & 0x3FFF) + (k_stage) * 256;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_15) | ((uint64_t)0x80004020 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_15) | ((uint64_t)0x80004020 << 32);

                                tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 0, b_desc + 0,
                                    0x10200480U, tmem_tmem_sfa_qk1 + 0, tmem_tmem_sfb_qkl + 0, 1);
                                tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 2, b_desc + 2,
                                    0x10200480U, tmem_tmem_sfa_qk1 + 4, tmem_tmem_sfb_qkl + 4, 1);
                            }
                        }
                        elect_commit_cg2_multicast(s_full_addr + 8, (uint16_t)(3));
                        elect_commit_cg2_multicast(k_empty_addr + (k_stage) * 8, (uint16_t)(3));
                        k_stage += 1;
                        if (k_stage == 4) { k_stage = 0; k_phase ^= 1; }
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb_pv_lo, make_sf_cp_desc_lo_sbo128((((smem_sfvt_lo_addr) >> 4) + (v_stage) * 64)));
                        }
                        mbarrier_wait(p_full_addr, _phase_p_full_0);
                        _phase_p_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_16 = (((smem_p_addr) >> 4) & 0x3FFF) + (0) * 512;
                        int _mma_b_lo_16 = (((smem_vt_addr) >> 4) & 0x3FFF) + (v_stage) * 256;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_16) | ((uint64_t)0x80004020 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_16) | ((uint64_t)0x80004020 << 32);

                                tcgen05_mma_mxf4nvf4_bs_cta2(tmem_output_0, a_desc + 0, b_desc + 0,
                                    0x10200480U, tmem_tmem_sfa_pv0_lo + 0, tmem_tmem_sfb_pv_lo + 0, ((first_pv_flag) ? 0 : 1));
                            }
                        }
                        mbarrier_wait(p_full_addr + 8, _phase_p_full_1);
                        _phase_p_full_1 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_17 = (((smem_p_addr) >> 4) & 0x3FFF) + (1) * 512;
                        int _mma_b_lo_17 = (((smem_vt_addr) >> 4) & 0x3FFF) + (v_stage) * 256;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_17) | ((uint64_t)0x80004020 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_17) | ((uint64_t)0x80004020 << 32);

                                tcgen05_mma_mxf4nvf4_bs_cta2(tmem_output_1, a_desc + 0, b_desc + 0,
                                    0x10200480U, tmem_tmem_sfa_pv1_lo + 0, tmem_tmem_sfb_pv_lo + 0, ((first_pv_flag) ? 0 : 1));
                            }
                        }
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb_pv_hi, make_sf_cp_desc_lo_sbo128((((smem_sfvt_hi_addr) >> 4) + (v_stage) * 64)));
                        }
                        mbarrier_wait(p_full_2_addr, _phase_p_full_2_0);
                        _phase_p_full_2_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_18 = (((smem_p_addr + 32) >> 4) & 0x3FFF) + (0) * 512;
                        int _mma_b_lo_18 = (((smem_vt_addr + 32) >> 4) & 0x3FFF) + (v_stage) * 256;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_18) | ((uint64_t)0x80004020 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_18) | ((uint64_t)0x80004020 << 32);

                                tcgen05_mma_mxf4nvf4_bs_cta2(tmem_output_0, a_desc + 0, b_desc + 0,
                                    0x10200480U, tmem_tmem_sfa_pv0_hi + 0, tmem_tmem_sfb_pv_hi + 0, 1);
                            }
                        }
                        elect_commit_cg2_multicast(p_empty_addr, (uint16_t)(3));
                        mbarrier_wait(p_full_2_addr + 8, _phase_p_full_2_1);
                        _phase_p_full_2_1 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_19 = (((smem_p_addr + 32) >> 4) & 0x3FFF) + (1) * 512;
                        int _mma_b_lo_19 = (((smem_vt_addr + 32) >> 4) & 0x3FFF) + (v_stage) * 256;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_19) | ((uint64_t)0x80004020 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_19) | ((uint64_t)0x80004020 << 32);

                                tcgen05_mma_mxf4nvf4_bs_cta2(tmem_output_1, a_desc + 0, b_desc + 0,
                                    0x10200480U, tmem_tmem_sfa_pv1_hi + 0, tmem_tmem_sfb_pv_hi + 0, 1);
                            }
                        }
                        elect_commit_cg2_multicast(p_empty_addr + 8, (uint16_t)(3));
                        elect_commit_cg2_multicast(v_empty_addr + (v_stage) * 8, (uint16_t)(3));
                        v_stage += 1;
                        if (v_stage == 5) { v_stage = 0; v_phase ^= 1; }
                        first_pv = 0;
                    }
                    int first_pv_flag_1 = first_pv;
                    mbarrier_wait(v_full_addr + (v_stage) * 8, v_phase);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb_pv_lo, make_sf_cp_desc_lo_sbo128((((smem_sfvt_lo_addr) >> 4) + (v_stage) * 64)));
                    }
                    mbarrier_wait(p_full_addr, _phase_p_full_0);
                    _phase_p_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_20 = (((smem_p_addr) >> 4) & 0x3FFF) + (0) * 512;
                    int _mma_b_lo_20 = (((smem_vt_addr) >> 4) & 0x3FFF) + (v_stage) * 256;
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_20) | ((uint64_t)0x80004020 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_20) | ((uint64_t)0x80004020 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2(tmem_output_0, a_desc + 0, b_desc + 0,
                                0x10200480U, tmem_tmem_sfa_pv0_lo + 0, tmem_tmem_sfb_pv_lo + 0, ((first_pv_flag_1) ? 0 : 1));
                        }
                    }
                    if (elect_sync()) {
                        tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb_pv_hi, make_sf_cp_desc_lo_sbo128((((smem_sfvt_hi_addr) >> 4) + (v_stage) * 64)));
                    }
                    mbarrier_wait(p_full_2_addr, _phase_p_full_2_0);
                    _phase_p_full_2_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_21 = (((smem_p_addr + 32) >> 4) & 0x3FFF) + (0) * 512;
                    int _mma_b_lo_21 = (((smem_vt_addr + 32) >> 4) & 0x3FFF) + (v_stage) * 256;
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_21) | ((uint64_t)0x80004020 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_21) | ((uint64_t)0x80004020 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2(tmem_output_0, a_desc + 0, b_desc + 0,
                                0x10200480U, tmem_tmem_sfa_pv0_hi + 0, tmem_tmem_sfb_pv_hi + 0, 1);
                        }
                    }
                    elect_commit_cg2_multicast(p_empty_addr, (uint16_t)(3));
                    mbarrier_wait(p_full_addr + 8, _phase_p_full_1);
                    _phase_p_full_1 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_22 = (((smem_p_addr) >> 4) & 0x3FFF) + (1) * 512;
                    int _mma_b_lo_22 = (((smem_vt_addr) >> 4) & 0x3FFF) + (v_stage) * 256;
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_22) | ((uint64_t)0x80004020 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_22) | ((uint64_t)0x80004020 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2(tmem_output_1, a_desc + 0, b_desc + 0,
                                0x10200480U, tmem_tmem_sfa_pv1_lo + 0, tmem_tmem_sfb_pv_lo + 0, ((first_pv_flag_1) ? 0 : 1));
                        }
                    }
                    mbarrier_wait(p_full_2_addr + 8, _phase_p_full_2_1);
                    _phase_p_full_2_1 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_23 = (((smem_p_addr + 32) >> 4) & 0x3FFF) + (1) * 512;
                    int _mma_b_lo_23 = (((smem_vt_addr + 32) >> 4) & 0x3FFF) + (v_stage) * 256;
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_23) | ((uint64_t)0x80004020 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_23) | ((uint64_t)0x80004020 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2(tmem_output_1, a_desc + 0, b_desc + 0,
                                0x10200480U, tmem_tmem_sfa_pv1_hi + 0, tmem_tmem_sfb_pv_hi + 0, 1);
                        }
                    }
                    elect_commit_cg2_multicast(p_empty_addr + 8, (uint16_t)(3));
                    elect_commit_cg2_multicast(v_empty_addr + (v_stage) * 8, (uint16_t)(3));
                    v_stage += 1;
                    if (v_stage == 5) { v_stage = 0; v_phase ^= 1; }
                    mbarrier_wait(s_empty_addr + 8, _phase_s_empty_1);
                    _phase_s_empty_1 ^= 1;
                    elect_commit_cg2_multicast(q_empty_addr, (uint16_t)(3));
                    elect_commit_cg2_multicast(o_full_addr, (uint16_t)(3));
                    elect_commit_cg2_multicast(o_full_addr + 8, (uint16_t)(3));
                }
            }
        }
    }
    // ---- Role: load ----
    if (warp == 13) {
        { // load_main
            unsigned int total_tiles_4 = (seqlen_q + 512 - 1) / 512 * total_bh;
            unsigned int n_full_u_4 = (unsigned int)n_full_tiles;
            unsigned int n_items_4 = n_full_u_4 + (total_tiles_4 - n_full_u_4) * (unsigned int)n_chunks;
            unsigned int num_n_blocks_all = kv_stride / 128;
            unsigned int num_q_blocks_all = q_stride / 128;
            unsigned int k_load_stage = 0;
            unsigned int _phase_q_empty_0 = 1;
            unsigned int _phase_k_empty = 1;
            #pragma unroll 1
            for (unsigned int tile_idx_5 = cluster_id; tile_idx_5 < n_items_4; tile_idx_5 += num_clusters) {
                unsigned int nkb_all_4 = (seqlen_kv + 128 - 1) / 128;
                unsigned int n_full_u_0_4 = (unsigned int)n_full_tiles;
                unsigned int n_chunks_u_4 = (unsigned int)n_chunks;
                unsigned int _max_0_4 = ((tile_idx_5) > (n_full_u_0_4) ? (tile_idx_5) : (n_full_u_0_4));
                unsigned int split_idx_4 = _max_0_4 - n_full_u_0_4;
                unsigned int chunk_4 = split_idx_4 - split_idx_4 / n_chunks_u_4 * n_chunks_u_4;
                unsigned int _min_1_4 = ((tile_idx_5) < (n_full_u_0_4) ? (tile_idx_5) : (n_full_u_0_4));
                unsigned int tile_idx_1_4 = _min_1_4 + split_idx_4 / n_chunks_u_4;
                unsigned int _min_2_4 = ((tile_idx_5) < (n_full_u_0_4) ? (tile_idx_5) : (n_full_u_0_4));
                unsigned int whole_4 = ((n_full_u_0_4 - _min_2_4) < (1) ? (n_full_u_0_4 - _min_2_4) : (1));
                unsigned int n_lo_4 = chunk_4 * nkb_all_4 / n_chunks_u_4;
                unsigned int n_hi_4 = (((chunk_4 + 1) * nkb_all_4 / n_chunks_u_4) > (nkb_all_4 * whole_4) ? ((chunk_4 + 1) * nkb_all_4 / n_chunks_u_4) : (nkb_all_4 * whole_4));
                unsigned int cluster_m_block_4 = tile_idx_1_4 % (unsigned int)((seqlen_q + 512 - 1) / 512);
                unsigned int bh_4 = tile_idx_1_4 / (unsigned int)((seqlen_q + 512 - 1) / 512);
                unsigned int m_block_4 = cluster_m_block_4 * 4 + (unsigned int)(cta_rank * 2);
                unsigned int num_n_blocks_4 = n_hi_4 - n_lo_4;
                int q_row = bh_4 * (unsigned int)q_stride + m_block_4 * 128;
                int q_sf_tile = bh_4 * num_q_blocks_all + m_block_4;
                mbarrier_wait(q_empty_addr, _phase_q_empty_0);
                _phase_q_empty_0 ^= 1;
                if (elect_sync()) {
                    tma_2d_gmem2smem_cta2(smem_q_addr, (&Q), 0, q_row, ((q_full_addr) & 0xFEFFFFFF));
                    tma_2d_gmem2smem_cta2(smem_sfq_addr, (&SFQ), 0, q_sf_tile * 32, ((q_full_addr) & 0xFEFFFFFF));
                    tma_2d_gmem2smem_cta2(smem_ql_addr, (&QL), 0, q_row, ((q_full_addr) & 0xFEFFFFFF));
                    tma_2d_gmem2smem_cta2(smem_sfql_addr, (&SFQL), 0, q_sf_tile * 32, ((q_full_addr) & 0xFEFFFFFF));
                    tma_2d_gmem2smem_cta2(smem_q_addr + 8192, (&Q), 0, q_row + 128, ((q_full_addr + 8) & 0xFEFFFFFF));
                    tma_2d_gmem2smem_cta2(smem_sfq_addr + 1024, (&SFQ), 0, (q_sf_tile + 1) * 32, ((q_full_addr + 8) & 0xFEFFFFFF));
                    tma_2d_gmem2smem_cta2(smem_ql_addr + 8192, (&QL), 0, q_row + 128, ((q_full_addr + 8) & 0xFEFFFFFF));
                    tma_2d_gmem2smem_cta2(smem_sfql_addr + 1024, (&SFQL), 0, (q_sf_tile + 1) * 32, ((q_full_addr + 8) & 0xFEFFFFFF));
                    asm volatile(
                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                        :: "r"((q_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(18432)) : "memory");
                    asm volatile(
                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                        :: "r"((q_full_addr + 8) & 0xFEFFFFFF), "r"((uint32_t)(18432)) : "memory");
                }
                #pragma unroll 1
                for (unsigned int ni = 0; ni < num_n_blocks_4; ni++) {
                    unsigned int n = n_hi_4 - 1 - ni;
                    int kv_row = bh_4 * (unsigned int)kv_stride + n * 128;
                    int kv_sf_tile = bh_4 * num_n_blocks_all + n;
                    mbarrier_wait(k_empty_addr + (k_load_stage) * 8, _phase_k_empty);
                    if (elect_sync()) {
                        tma_2d_gmem2smem_cta2(smem_k_addr + k_load_stage * 4096, (&K), 0, kv_row + cta_rank * 64, ((k_full_addr + (k_load_stage) * 8) & 0xFEFFFFFF));
                        tma_2d_gmem2smem_cta2(smem_sfk_addr + k_load_stage * 1024, (&SFK), 0, kv_sf_tile * 32, ((k_full_addr + (k_load_stage) * 8) & 0xFEFFFFFF));
                        tma_2d_gmem2smem_cta2(smem_kl_addr + k_load_stage * 4096, (&KL), 0, kv_row + cta_rank * 64, ((k_full_addr + (k_load_stage) * 8) & 0xFEFFFFFF));
                        tma_2d_gmem2smem_cta2(smem_sfkl_addr + k_load_stage * 1024, (&SFKL), 0, kv_sf_tile * 32, ((k_full_addr + (k_load_stage) * 8) & 0xFEFFFFFF));
                        int tab_row = (int)(((bh_4 * ((num_q_blocks_all + 7) / 8) + m_block_4 / 8) * (num_n_blocks_all * 2) + n * 2 + (unsigned int)cta_rank) * 64);
                        tma_2d_gmem2smem_cta2(smem_tab_addr + k_load_stage * 2048, (&QM), 0, tab_row, ((k_full_addr + (k_load_stage) * 8) & 0xFEFFFFFF));
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((k_full_addr + (k_load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(12288)) : "memory");
                    }
                    k_load_stage += 1;
                    if (k_load_stage == 4) { k_load_stage = 0; _phase_k_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: v_load ----
    if (warp == 14) {
        { // v_load_main
            unsigned int total_tiles_5 = (seqlen_q + 512 - 1) / 512 * total_bh;
            unsigned int n_full_u_5 = (unsigned int)n_full_tiles;
            unsigned int n_items_5 = n_full_u_5 + (total_tiles_5 - n_full_u_5) * (unsigned int)n_chunks;
            unsigned int num_n_blocks_all_1 = kv_stride / 128;
            unsigned int v_load_stage = 0;
            unsigned int _phase_v_empty = 1;
            #pragma unroll 1
            for (unsigned int tile_idx_6 = cluster_id; tile_idx_6 < n_items_5; tile_idx_6 += num_clusters) {
                unsigned int nkb_all_5 = (seqlen_kv + 128 - 1) / 128;
                unsigned int n_full_u_0_5 = (unsigned int)n_full_tiles;
                unsigned int n_chunks_u_5 = (unsigned int)n_chunks;
                unsigned int _max_0_5 = ((tile_idx_6) > (n_full_u_0_5) ? (tile_idx_6) : (n_full_u_0_5));
                unsigned int split_idx_5 = _max_0_5 - n_full_u_0_5;
                unsigned int chunk_5 = split_idx_5 - split_idx_5 / n_chunks_u_5 * n_chunks_u_5;
                unsigned int _min_1_5 = ((tile_idx_6) < (n_full_u_0_5) ? (tile_idx_6) : (n_full_u_0_5));
                unsigned int tile_idx_1_5 = _min_1_5 + split_idx_5 / n_chunks_u_5;
                unsigned int _min_2_5 = ((tile_idx_6) < (n_full_u_0_5) ? (tile_idx_6) : (n_full_u_0_5));
                unsigned int whole_5 = ((n_full_u_0_5 - _min_2_5) < (1) ? (n_full_u_0_5 - _min_2_5) : (1));
                unsigned int n_lo_5 = chunk_5 * nkb_all_5 / n_chunks_u_5;
                unsigned int n_hi_5 = (((chunk_5 + 1) * nkb_all_5 / n_chunks_u_5) > (nkb_all_5 * whole_5) ? ((chunk_5 + 1) * nkb_all_5 / n_chunks_u_5) : (nkb_all_5 * whole_5));
                unsigned int cluster_m_block_5 = tile_idx_1_5 % (unsigned int)((seqlen_q + 512 - 1) / 512);
                unsigned int bh_5 = tile_idx_1_5 / (unsigned int)((seqlen_q + 512 - 1) / 512);
                unsigned int m_block_5 = cluster_m_block_5 * 4 + (unsigned int)(cta_rank * 2);
                unsigned int num_n_blocks_5 = n_hi_5 - n_lo_5;
                #pragma unroll 1
                for (unsigned int ni_1 = 0; ni_1 < num_n_blocks_5; ni_1++) {
                    unsigned int n_1 = n_hi_5 - 1 - ni_1;
                    int kv_sf_tile_1 = bh_5 * num_n_blocks_all_1 + n_1;
                    int vt_row = bh_5 * 128;
                    int vt_col = n_1 * 64;
                    mbarrier_wait(v_empty_addr + (v_load_stage) * 8, _phase_v_empty);
                    if (elect_sync()) {
                        tma_2d_gmem2smem_cta2(smem_vt_addr + v_load_stage * 4096, (&Vt), vt_col, vt_row + cta_rank * 64, ((v_full_addr + (v_load_stage) * 8) & 0xFEFFFFFF));
                        tma_2d_gmem2smem_cta2(smem_sfvt_lo_addr + v_load_stage * 1024, (&SFVtLo), 0, kv_sf_tile_1 * 16, ((v_full_addr + (v_load_stage) * 8) & 0xFEFFFFFF));
                        tma_2d_gmem2smem_cta2(smem_sfvt_hi_addr + v_load_stage * 1024, (&SFVtHi), 0, kv_sf_tile_1 * 16, ((v_full_addr + (v_load_stage) * 8) & 0xFEFFFFFF));
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((v_full_addr + (v_load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(5120)) : "memory");
                    }
                    v_load_stage += 1;
                    if (v_load_stage == 5) { v_load_stage = 0; _phase_v_empty ^= 1; }
                }
            }
        }
    }
    // ---- Role: idle ----
    if (warp == 15) {
        { // idle_main
            __syncwarp();
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
