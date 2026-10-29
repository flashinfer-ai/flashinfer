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
#define TMEM_NCOLS 448
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
#define TMEM_TMEM_SFA_PV0_LO_B1_OFFSET 432
#define TMEM_TMEM_SFA_PV0_HI_B1_OFFSET 436
#define TMEM_TMEM_SFA_PV1_LO_B1_OFFSET 440
#define TMEM_TMEM_SFA_PV1_HI_B1_OFFSET 444
#define NUM_V_PIPE_STAGES 5
#define NUM_K_PIPE_STAGES 4
#define SMEM_O_STAGE_OFF 100352
#define SMEM_O_STAGE_STAGE_BYTES 69632
#define SMEM_O_STAGE_STRIDE 69632
#define SMEM_ROW_STATE_OFF 1024
#define SMEM_ROW_STATE_STAGE_BYTES 2048
#define SMEM_ROW_STATE_STRIDE 2048
#define SMEM_SMEM_Q_OFF 3072
#define SMEM_SMEM_Q_STAGE_BYTES 8192
#define SMEM_SMEM_Q_STRIDE 8192
#define SMEM_SMEM_SFQ_OFF 19456
#define SMEM_SMEM_SFQ_STAGE_BYTES 1024
#define SMEM_SMEM_SFQ_STRIDE 1024
#define SMEM_SMEM_K_OFF 21504
#define SMEM_SMEM_K_STAGE_BYTES 4096
#define SMEM_SMEM_K_STRIDE 4096
#define SMEM_SMEM_SFK_OFF 37888
#define SMEM_SMEM_SFK_STAGE_BYTES 1024
#define SMEM_SMEM_SFK_STRIDE 1024
#define SMEM_SMEM_VT_OFF 41984
#define SMEM_SMEM_VT_STAGE_BYTES 4096
#define SMEM_SMEM_VT_STRIDE 4096
#define SMEM_SMEM_SFVT_LO_OFF 62464
#define SMEM_SMEM_SFVT_LO_STAGE_BYTES 512
#define SMEM_SMEM_SFVT_LO_STRIDE 1024
#define SMEM_SMEM_SFVT_HI_OFF 62976
#define SMEM_SMEM_SFVT_HI_STAGE_BYTES 512
#define SMEM_SMEM_SFVT_HI_STRIDE 1024
#define SMEM_SMEM_P_OFF 67584
#define SMEM_SMEM_P_STAGE_BYTES 8192
#define SMEM_SMEM_P_STRIDE 8192
#define SMEM_TOTAL 169984
#define THREADS 512
#if __CUDA_ARCH__ == 1030
#define USE_TMEM_LD_RED 1
#else
#define USE_TMEM_LD_RED 0
#endif
#define BLOCK_M 128
#define BLOCK_N 128
#define HEAD_DIM 128
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




__device__ __forceinline__ float row_max_reduce(float2 acc) {
    return max_noftz(acc.x, acc.y);
}


__device__ __forceinline__ void row_max_x32_accum(const float* sv, float2& acc) {
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        if (j % 2 == 0)
            acc.x = max_noftz(acc.x, max_noftz(sv[j*2], sv[j*2+1]));
        else
            acc.y = max_noftz(acc.y, max_noftz(sv[j*2], sv[j*2+1]));
    }
}

#if __CUDA_ARCH__ == 1030
#else

__device__ __forceinline__ float2 ex2_emulation_f32x2_value(float2 value) {
    const float c0 = 1.0f, c1 = 0.695146143436431884765625f;
    const float c2 = 0.227564394474029541015625f, c3 = 0.077119089663028717041015625f;
    const float magic = 12582912.0f;
    float x0 = max_noftz(value.x, -127.0f), x1 = max_noftz(value.y, -127.0f);
    float2 xc2 = make_float2(x0, x1), magic2 = make_float2(magic, magic);
    float2 xr2;
    asm("add.rm.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xr2)
        : "l"(*(unsigned long long*)&xc2), "l"(*(unsigned long long*)&magic2));
    float2 c3_2 = make_float2(c3, c3), c2_2 = make_float2(c2, c2);
    float2 c1_2 = make_float2(c1, c1), c0_2 = make_float2(c0, c0);
    float2 xrb2, xfrac2;
    asm("sub.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xrb2)
        : "l"(*(unsigned long long*)&xr2), "l"(*(unsigned long long*)&magic2));
    asm("sub.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xfrac2)
        : "l"(*(unsigned long long*)&xc2), "l"(*(unsigned long long*)&xrb2));
    float2 poly2;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;" : "=l"(*(unsigned long long*)&poly2)
        : "l"(*(unsigned long long*)&c3_2), "l"(*(unsigned long long*)&xfrac2), "l"(*(unsigned long long*)&c2_2));
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;" : "=l"(*(unsigned long long*)&poly2)
        : "l"(*(unsigned long long*)&poly2), "l"(*(unsigned long long*)&xfrac2), "l"(*(unsigned long long*)&c1_2));
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;" : "=l"(*(unsigned long long*)&poly2)
        : "l"(*(unsigned long long*)&poly2), "l"(*(unsigned long long*)&xfrac2), "l"(*(unsigned long long*)&c0_2));
    int x0r_i, x1r_i, p0_i, p1_i;
    asm("mov.b64 {%0, %1}, %2;" : "=r"(x0r_i), "=r"(x1r_i) : "l"(*(unsigned long long*)&xr2));
    asm("mov.b64 {%0, %1}, %2;" : "=r"(p0_i), "=r"(p1_i) : "l"(*(unsigned long long*)&poly2));
    float r0, r1;
    asm("mov.b32 %0, %1;" : "=f"(r0) : "r"((x0r_i << 23) + p0_i));
    asm("mov.b32 %0, %1;" : "=f"(r1) : "r"((x1r_i << 23) + p1_i));
    return make_float2(r0, r1);
}
#endif



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

__device__ __forceinline__ float2 fma_f32x2(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    return r;
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

__device__ __forceinline__ float2 mul_f32x2_rn_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

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



__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_cake_minimax_h3_varlen_attention_1f0285434cb51b016f42(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap Vt, const __grid_constant__ CUtensorMap SFQ, const __grid_constant__ CUtensorMap SFK, const __grid_constant__ CUtensorMap SFVtLo, const __grid_constant__ CUtensorMap SFVtHi, __nv_bfloat16* __restrict__ O, int* __restrict__ cl_head, int* __restrict__ cl_seg_begin, int* __restrict__ cl_seg_len, int* __restrict__ cl_kv_base, int* __restrict__ cl_q_block, unsigned int num_tiles, int heads, int PB, float softmax_scale_log2)
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
    unsigned int* o_stage = reinterpret_cast<unsigned int*>(smem_raw + 100352);
    const int o_stage_addr = smem + 100352;
    float* row_state = reinterpret_cast<float*>(smem_raw + 1024);
    const int row_state_addr = smem + 1024;
    uint8_t* smem_q = reinterpret_cast<uint8_t*>(smem_raw + 3072);
    const int smem_q_addr = smem + 3072;
    uint8_t* smem_sfq = reinterpret_cast<uint8_t*>(smem_raw + 19456);
    const int smem_sfq_addr = smem + 19456;
    uint8_t* smem_k = reinterpret_cast<uint8_t*>(smem_raw + 21504);
    const int smem_k_addr = smem + 21504;
    uint8_t* smem_sfk = reinterpret_cast<uint8_t*>(smem_raw + 37888);
    const int smem_sfk_addr = smem + 37888;
    uint8_t* smem_vt = reinterpret_cast<uint8_t*>(smem_raw + 41984);
    const int smem_vt_addr = smem + 41984;
    uint8_t* smem_sfvt_lo = reinterpret_cast<uint8_t*>(smem_raw + 62464);
    const int smem_sfvt_lo_addr = smem + 62464;
    uint8_t* smem_sfvt_hi = reinterpret_cast<uint8_t*>(smem_raw + 62976);
    const int smem_sfvt_hi_addr = smem + 62976;
    uint8_t* smem_p = reinterpret_cast<uint8_t*>(smem_raw + 67584);
    const int smem_p_addr = smem + 67584;

    // Mbarrier init (14 pipeline groups, 0 ordered-sequence groups, 39 barriers)
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
            // p_full: 4 barriers, init_count=512
            mbarrier_init(smem + 200, 512);
            mbarrier_init(smem + 208, 512);
            mbarrier_init(smem + 216, 512);
            mbarrier_init(smem + 224, 512);
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

    // TMEM alloc (512 columns, 448 used)
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
    const int tmem_tmem_sfa_pv0_lo_b1 = taddr + 432;
    const int tmem_tmem_sfa_pv0_hi_b1 = taddr + 436;
    const int tmem_tmem_sfa_pv1_lo_b1 = taddr + 440;
    const int tmem_tmem_sfa_pv1_hi_b1 = taddr + 444;
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if (warp >= 8 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
    }

    // ---- Role: softmax ----
    if (warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 208;");
        { // softmax_main
            unsigned int total_tiles = num_tiles;
            unsigned int stage = make_warp_uniform(warp / 4);
            int scale_off = make_warp_uniform(stage * 128);
            int p_col = make_warp_uniform(0);
            unsigned int p_slot = stage * 2;
            unsigned int _phase_s_full = 0;
            unsigned int _phase_corr_done = 0;
            unsigned int _phase_o_full = 0;
            #pragma unroll 1
            for (unsigned int tile_idx = cluster_id; tile_idx < total_tiles; tile_idx += num_clusters) {
                int head = cl_head[tile_idx];
                int seg_begin = cl_seg_begin[tile_idx];
                int seg_len = cl_seg_len[tile_idx];
                int kv_base = cl_kv_base[tile_idx];
                int q_word = cl_q_block[tile_idx];
                int q_base = q_word & 65535;
                int q_half = q_word >> 16;
                int two_stages = ((q_half == 0) ? 1 : 0);
                int m_block = q_base + (q_half >> 1) * 2 + cta_rank * (1 + two_stages);
                unsigned int num_n_blocks = (seg_len + 128 - 1) / 128;
                int kv_begin = 0;
                int ws_slot = -1;
                int active = ((stage == 0) ? 1 : two_stages);
                if (active == 1) {
                    int tail_base = seg_len;
                    unsigned int n_count = num_n_blocks;
                    float row_max = -CUDART_INF_F;
                    float row_max_scaled = 0.0f;
                    float row_sum = 0.0f;
                    #pragma unroll 1
#if __CUDA_ARCH__ == 1030
                    for (unsigned int n_iter = 0; n_iter < n_count; n_iter++) {
#else
                    for (unsigned int n_iter = 0; n_iter < 1; n_iter++) {
#endif
                        int n_block = n_count - 1 - n_iter;
                        mbarrier_wait_hint(s_full_addr + (stage) * 8, _phase_s_full, 10000000);
                        _phase_s_full ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int s_addr = taddr + (unsigned int)0 + (unsigned int)(warp % 4 * 32 << 16);
                        float sv[128];
                        float tile_max = -CUDART_INF_F;
                        {
#if __CUDA_ARCH__ == 1030
                            float lo_max = -CUDART_INF_F;
                            float hi_max = -CUDART_INF_F;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                            #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                            #endif
#else
#endif
                            asm volatile(
#if __CUDA_ARCH__ == 1030
                                "tcgen05.ld.red.sync.aligned.32x32b.x64.max.f32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, [%65];"
                                : "=f"(sv[0]), "=f"(sv[1]), "=f"(sv[2]), "=f"(sv[3]), "=f"(sv[4]), "=f"(sv[5]), "=f"(sv[6]), "=f"(sv[7]), "=f"(sv[8]), "=f"(sv[9]), "=f"(sv[10]), "=f"(sv[11]), "=f"(sv[12]), "=f"(sv[13]), "=f"(sv[14]), "=f"(sv[15]), "=f"(sv[16]), "=f"(sv[17]), "=f"(sv[18]), "=f"(sv[19]), "=f"(sv[20]), "=f"(sv[21]), "=f"(sv[22]), "=f"(sv[23]), "=f"(sv[24]), "=f"(sv[25]), "=f"(sv[26]), "=f"(sv[27]), "=f"(sv[28]), "=f"(sv[29]), "=f"(sv[30]), "=f"(sv[31]), "=f"(sv[32]), "=f"(sv[33]), "=f"(sv[34]), "=f"(sv[35]), "=f"(sv[36]), "=f"(sv[37]), "=f"(sv[38]), "=f"(sv[39]), "=f"(sv[40]), "=f"(sv[41]), "=f"(sv[42]), "=f"(sv[43]), "=f"(sv[44]), "=f"(sv[45]), "=f"(sv[46]), "=f"(sv[47]), "=f"(sv[48]), "=f"(sv[49]), "=f"(sv[50]), "=f"(sv[51]), "=f"(sv[52]), "=f"(sv[53]), "=f"(sv[54]), "=f"(sv[55]), "=f"(sv[56]), "=f"(sv[57]), "=f"(sv[58]), "=f"(sv[59]), "=f"(sv[60]), "=f"(sv[61]), "=f"(sv[62]), "=f"(sv[63]), "=f"(lo_max)
                                : "r"(tmem_scores + (warp % 4 * 32 << 16)));
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                            #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                            #endif
#else
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(sv[0]), "=f"(sv[1]), "=f"(sv[2]), "=f"(sv[3]), "=f"(sv[4]), "=f"(sv[5]), "=f"(sv[6]), "=f"(sv[7]), "=f"(sv[8]), "=f"(sv[9]), "=f"(sv[10]), "=f"(sv[11]), "=f"(sv[12]), "=f"(sv[13]), "=f"(sv[14]), "=f"(sv[15]), "=f"(sv[16]), "=f"(sv[17]), "=f"(sv[18]), "=f"(sv[19]), "=f"(sv[20]), "=f"(sv[21]), "=f"(sv[22]), "=f"(sv[23]), "=f"(sv[24]), "=f"(sv[25]), "=f"(sv[26]), "=f"(sv[27]), "=f"(sv[28]), "=f"(sv[29]), "=f"(sv[30]), "=f"(sv[31])
                                : "r"(s_addr));
#endif
                            asm volatile(
#if __CUDA_ARCH__ == 1030
                                "tcgen05.ld.red.sync.aligned.32x32b.x64.max.f32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, [%65];"
                                : "=f"(sv[64]), "=f"(sv[65]), "=f"(sv[66]), "=f"(sv[67]), "=f"(sv[68]), "=f"(sv[69]), "=f"(sv[70]), "=f"(sv[71]), "=f"(sv[72]), "=f"(sv[73]), "=f"(sv[74]), "=f"(sv[75]), "=f"(sv[76]), "=f"(sv[77]), "=f"(sv[78]), "=f"(sv[79]), "=f"(sv[80]), "=f"(sv[81]), "=f"(sv[82]), "=f"(sv[83]), "=f"(sv[84]), "=f"(sv[85]), "=f"(sv[86]), "=f"(sv[87]), "=f"(sv[88]), "=f"(sv[89]), "=f"(sv[90]), "=f"(sv[91]), "=f"(sv[92]), "=f"(sv[93]), "=f"(sv[94]), "=f"(sv[95]), "=f"(sv[96]), "=f"(sv[97]), "=f"(sv[98]), "=f"(sv[99]), "=f"(sv[100]), "=f"(sv[101]), "=f"(sv[102]), "=f"(sv[103]), "=f"(sv[104]), "=f"(sv[105]), "=f"(sv[106]), "=f"(sv[107]), "=f"(sv[108]), "=f"(sv[109]), "=f"(sv[110]), "=f"(sv[111]), "=f"(sv[112]), "=f"(sv[113]), "=f"(sv[114]), "=f"(sv[115]), "=f"(sv[116]), "=f"(sv[117]), "=f"(sv[118]), "=f"(sv[119]), "=f"(sv[120]), "=f"(sv[121]), "=f"(sv[122]), "=f"(sv[123]), "=f"(sv[124]), "=f"(sv[125]), "=f"(sv[126]), "=f"(sv[127]), "=f"(hi_max)
                                : "r"(tmem_scores + 64 + (warp % 4 * 32 << 16)));
#else
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(sv[32]), "=f"(sv[33]), "=f"(sv[34]), "=f"(sv[35]), "=f"(sv[36]), "=f"(sv[37]), "=f"(sv[38]), "=f"(sv[39]), "=f"(sv[40]), "=f"(sv[41]), "=f"(sv[42]), "=f"(sv[43]), "=f"(sv[44]), "=f"(sv[45]), "=f"(sv[46]), "=f"(sv[47]), "=f"(sv[48]), "=f"(sv[49]), "=f"(sv[50]), "=f"(sv[51]), "=f"(sv[52]), "=f"(sv[53]), "=f"(sv[54]), "=f"(sv[55]), "=f"(sv[56]), "=f"(sv[57]), "=f"(sv[58]), "=f"(sv[59]), "=f"(sv[60]), "=f"(sv[61]), "=f"(sv[62]), "=f"(sv[63])
                                : "r"(s_addr + 32));
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(sv[64]), "=f"(sv[65]), "=f"(sv[66]), "=f"(sv[67]), "=f"(sv[68]), "=f"(sv[69]), "=f"(sv[70]), "=f"(sv[71]), "=f"(sv[72]), "=f"(sv[73]), "=f"(sv[74]), "=f"(sv[75]), "=f"(sv[76]), "=f"(sv[77]), "=f"(sv[78]), "=f"(sv[79]), "=f"(sv[80]), "=f"(sv[81]), "=f"(sv[82]), "=f"(sv[83]), "=f"(sv[84]), "=f"(sv[85]), "=f"(sv[86]), "=f"(sv[87]), "=f"(sv[88]), "=f"(sv[89]), "=f"(sv[90]), "=f"(sv[91]), "=f"(sv[92]), "=f"(sv[93]), "=f"(sv[94]), "=f"(sv[95])
                                : "r"(s_addr + 64));
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(sv[96]), "=f"(sv[97]), "=f"(sv[98]), "=f"(sv[99]), "=f"(sv[100]), "=f"(sv[101]), "=f"(sv[102]), "=f"(sv[103]), "=f"(sv[104]), "=f"(sv[105]), "=f"(sv[106]), "=f"(sv[107]), "=f"(sv[108]), "=f"(sv[109]), "=f"(sv[110]), "=f"(sv[111]), "=f"(sv[112]), "=f"(sv[113]), "=f"(sv[114]), "=f"(sv[115]), "=f"(sv[116]), "=f"(sv[117]), "=f"(sv[118]), "=f"(sv[119]), "=f"(sv[120]), "=f"(sv[121]), "=f"(sv[122]), "=f"(sv[123]), "=f"(sv[124]), "=f"(sv[125]), "=f"(sv[126]), "=f"(sv[127])
                                : "r"(s_addr + 96));
#endif
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
#if __CUDA_ARCH__ == 1030
                            tile_max = max_noftz(lo_max, hi_max);
#else
                            float2 _reg_reduce_max2_1 = {-CAKE_INF, -CAKE_INF};
                            row_max_x32_accum(&sv[0], _reg_reduce_max2_1);
                            row_max_x32_accum(&sv[32], _reg_reduce_max2_1);
                            row_max_x32_accum(&sv[64], _reg_reduce_max2_1);
                            row_max_x32_accum(&sv[96], _reg_reduce_max2_1);
                            float _reg_array_reduce_0 = row_max_reduce(_reg_reduce_max2_1);
                            tile_max = _reg_array_reduce_0;
#endif
                        }
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((s_empty_addr + (stage) * 8) & 0xFEFFFFFF) : "memory");
                        int tail_valid = tail_base - n_block * 128;
                        if (tail_valid < 128) {
#if __CUDA_ARCH__ == 1030
                            uint32_t _thresh_mask_0;
                            {
                                int _lim_1 = tail_valid;
                                if (_lim_1 <= 0) { _thresh_mask_0 = 0u; }
                                else if (_lim_1 >= 32) { _thresh_mask_0 = 0xFFFFFFFFu; }
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
                                        "}" : "=r"(_thresh_mask_0) : "r"(_lim_1));
                                }
                            }
                            if (!(_thresh_mask_0 & (1u << 0))) sv[0] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 1))) sv[1] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 2))) sv[2] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 3))) sv[3] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 4))) sv[4] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 5))) sv[5] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 6))) sv[6] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 7))) sv[7] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 8))) sv[8] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 9))) sv[9] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 10))) sv[10] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 11))) sv[11] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 12))) sv[12] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 13))) sv[13] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 14))) sv[14] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 15))) sv[15] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 16))) sv[16] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 17))) sv[17] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 18))) sv[18] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 19))) sv[19] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 20))) sv[20] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 21))) sv[21] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 22))) sv[22] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 23))) sv[23] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 24))) sv[24] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 25))) sv[25] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 26))) sv[26] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 27))) sv[27] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 28))) sv[28] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 29))) sv[29] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 30))) sv[30] = -CUDART_INF_F;
                            if (!(_thresh_mask_0 & (1u << 31))) sv[31] = -CUDART_INF_F;
#else
#endif
                            uint32_t _thresh_mask_2;
                            {
#if __CUDA_ARCH__ == 1030
                                int _lim_3 = tail_valid - 32;
#else
                                int _lim_3 = tail_valid;
#endif
                                if (_lim_3 <= 0) { _thresh_mask_2 = 0u; }
                                else if (_lim_3 >= 32) { _thresh_mask_2 = 0xFFFFFFFFu; }
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
                                        "}" : "=r"(_thresh_mask_2) : "r"(_lim_3));
                                }
                            }
#if __CUDA_ARCH__ == 1030
                            if (!(_thresh_mask_2 & (1u << 0))) sv[32] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 1))) sv[33] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 2))) sv[34] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 3))) sv[35] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 4))) sv[36] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 5))) sv[37] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 6))) sv[38] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 7))) sv[39] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 8))) sv[40] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 9))) sv[41] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 10))) sv[42] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 11))) sv[43] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 12))) sv[44] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 13))) sv[45] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 14))) sv[46] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 15))) sv[47] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 16))) sv[48] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 17))) sv[49] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 18))) sv[50] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 19))) sv[51] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 20))) sv[52] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 21))) sv[53] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 22))) sv[54] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 23))) sv[55] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 24))) sv[56] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 25))) sv[57] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 26))) sv[58] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 27))) sv[59] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 28))) sv[60] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 29))) sv[61] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 30))) sv[62] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 31))) sv[63] = -CUDART_INF_F;
#else
                            if (!(_thresh_mask_2 & (1u << 0))) sv[0] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 1))) sv[1] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 2))) sv[2] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 3))) sv[3] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 4))) sv[4] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 5))) sv[5] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 6))) sv[6] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 7))) sv[7] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 8))) sv[8] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 9))) sv[9] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 10))) sv[10] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 11))) sv[11] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 12))) sv[12] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 13))) sv[13] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 14))) sv[14] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 15))) sv[15] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 16))) sv[16] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 17))) sv[17] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 18))) sv[18] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 19))) sv[19] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 20))) sv[20] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 21))) sv[21] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 22))) sv[22] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 23))) sv[23] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 24))) sv[24] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 25))) sv[25] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 26))) sv[26] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 27))) sv[27] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 28))) sv[28] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 29))) sv[29] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 30))) sv[30] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 31))) sv[31] = -CUDART_INF_F;
#endif
                            uint32_t _thresh_mask_4;
                            {
#if __CUDA_ARCH__ == 1030
                                int _lim_5 = tail_valid - 64;
#else
                                int _lim_5 = tail_valid - 32;
#endif
                                if (_lim_5 <= 0) { _thresh_mask_4 = 0u; }
                                else if (_lim_5 >= 32) { _thresh_mask_4 = 0xFFFFFFFFu; }
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
                                        "}" : "=r"(_thresh_mask_4) : "r"(_lim_5));
                                }
                            }
#if __CUDA_ARCH__ == 1030
                            if (!(_thresh_mask_4 & (1u << 0))) sv[64] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 1))) sv[65] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 2))) sv[66] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 3))) sv[67] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 4))) sv[68] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 5))) sv[69] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 6))) sv[70] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 7))) sv[71] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 8))) sv[72] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 9))) sv[73] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 10))) sv[74] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 11))) sv[75] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 12))) sv[76] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 13))) sv[77] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 14))) sv[78] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 15))) sv[79] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 16))) sv[80] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 17))) sv[81] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 18))) sv[82] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 19))) sv[83] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 20))) sv[84] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 21))) sv[85] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 22))) sv[86] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 23))) sv[87] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 24))) sv[88] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 25))) sv[89] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 26))) sv[90] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 27))) sv[91] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 28))) sv[92] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 29))) sv[93] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 30))) sv[94] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 31))) sv[95] = -CUDART_INF_F;
#else
                            if (!(_thresh_mask_4 & (1u << 0))) sv[32] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 1))) sv[33] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 2))) sv[34] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 3))) sv[35] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 4))) sv[36] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 5))) sv[37] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 6))) sv[38] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 7))) sv[39] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 8))) sv[40] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 9))) sv[41] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 10))) sv[42] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 11))) sv[43] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 12))) sv[44] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 13))) sv[45] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 14))) sv[46] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 15))) sv[47] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 16))) sv[48] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 17))) sv[49] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 18))) sv[50] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 19))) sv[51] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 20))) sv[52] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 21))) sv[53] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 22))) sv[54] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 23))) sv[55] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 24))) sv[56] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 25))) sv[57] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 26))) sv[58] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 27))) sv[59] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 28))) sv[60] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 29))) sv[61] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 30))) sv[62] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 31))) sv[63] = -CUDART_INF_F;
#endif
                            uint32_t _thresh_mask_6;
                            {
#if __CUDA_ARCH__ == 1030
                                int _lim_7 = tail_valid - 96;
#else
                                int _lim_7 = tail_valid - 64;
#endif
                                if (_lim_7 <= 0) { _thresh_mask_6 = 0u; }
                                else if (_lim_7 >= 32) { _thresh_mask_6 = 0xFFFFFFFFu; }
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
                                        "}" : "=r"(_thresh_mask_6) : "r"(_lim_7));
                                }
                            }
#if __CUDA_ARCH__ == 1030
                            if (!(_thresh_mask_6 & (1u << 0))) sv[96] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 1))) sv[97] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 2))) sv[98] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 3))) sv[99] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 4))) sv[100] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 5))) sv[101] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 6))) sv[102] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 7))) sv[103] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 8))) sv[104] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 9))) sv[105] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 10))) sv[106] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 11))) sv[107] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 12))) sv[108] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 13))) sv[109] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 14))) sv[110] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 15))) sv[111] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 16))) sv[112] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 17))) sv[113] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 18))) sv[114] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 19))) sv[115] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 20))) sv[116] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 21))) sv[117] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 22))) sv[118] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 23))) sv[119] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 24))) sv[120] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 25))) sv[121] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 26))) sv[122] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 27))) sv[123] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 28))) sv[124] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 29))) sv[125] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 30))) sv[126] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 31))) sv[127] = -CUDART_INF_F;
                            float2 _reg_reduce_max2_9 = {-CAKE_INF, -CAKE_INF};
                            row_max_x32_accum(&sv[0], _reg_reduce_max2_9);
                            row_max_x32_accum(&sv[32], _reg_reduce_max2_9);
                            row_max_x32_accum(&sv[64], _reg_reduce_max2_9);
                            row_max_x32_accum(&sv[96], _reg_reduce_max2_9);
                            float _reg_array_reduce_8 = row_max_reduce(_reg_reduce_max2_9);
                            tile_max = _reg_array_reduce_8;
#else
                            if (!(_thresh_mask_6 & (1u << 0))) sv[64] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 1))) sv[65] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 2))) sv[66] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 3))) sv[67] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 4))) sv[68] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 5))) sv[69] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 6))) sv[70] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 7))) sv[71] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 8))) sv[72] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 9))) sv[73] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 10))) sv[74] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 11))) sv[75] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 12))) sv[76] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 13))) sv[77] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 14))) sv[78] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 15))) sv[79] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 16))) sv[80] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 17))) sv[81] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 18))) sv[82] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 19))) sv[83] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 20))) sv[84] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 21))) sv[85] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 22))) sv[86] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 23))) sv[87] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 24))) sv[88] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 25))) sv[89] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 26))) sv[90] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 27))) sv[91] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 28))) sv[92] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 29))) sv[93] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 30))) sv[94] = -CUDART_INF_F;
                            if (!(_thresh_mask_6 & (1u << 31))) sv[95] = -CUDART_INF_F;
                            uint32_t _thresh_mask_8;
                            {
                                int _lim_9 = tail_valid - 96;
                                if (_lim_9 <= 0) { _thresh_mask_8 = 0u; }
                                else if (_lim_9 >= 32) { _thresh_mask_8 = 0xFFFFFFFFu; }
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
                                        "}" : "=r"(_thresh_mask_8) : "r"(_lim_9));
                                }
                            }
                            if (!(_thresh_mask_8 & (1u << 0))) sv[96] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 1))) sv[97] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 2))) sv[98] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 3))) sv[99] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 4))) sv[100] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 5))) sv[101] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 6))) sv[102] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 7))) sv[103] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 8))) sv[104] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 9))) sv[105] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 10))) sv[106] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 11))) sv[107] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 12))) sv[108] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 13))) sv[109] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 14))) sv[110] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 15))) sv[111] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 16))) sv[112] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 17))) sv[113] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 18))) sv[114] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 19))) sv[115] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 20))) sv[116] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 21))) sv[117] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 22))) sv[118] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 23))) sv[119] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 24))) sv[120] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 25))) sv[121] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 26))) sv[122] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 27))) sv[123] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 28))) sv[124] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 29))) sv[125] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 30))) sv[126] = -CUDART_INF_F;
                            if (!(_thresh_mask_8 & (1u << 31))) sv[127] = -CUDART_INF_F;
                            float2 _reg_reduce_max2_11 = {-CAKE_INF, -CAKE_INF};
                            row_max_x32_accum(&sv[0], _reg_reduce_max2_11);
                            row_max_x32_accum(&sv[32], _reg_reduce_max2_11);
                            row_max_x32_accum(&sv[64], _reg_reduce_max2_11);
                            row_max_x32_accum(&sv[96], _reg_reduce_max2_11);
                            float _reg_array_reduce_10 = row_max_reduce(_reg_reduce_max2_11);
                            tile_max = _reg_array_reduce_10;
#endif
                        }
                        float new_max;
                        asm("max.f32 %0, %1, %2;" : "=f"(new_max) : "f"(tile_max), "f"(row_max));
                        float safe_max = ((new_max == -CUDART_INF_F) ? 0.0f : new_max);
                        float new_max_scaled = safe_max * softmax_scale_log2;
                        float acc_scale_log2 = __fmaf_rn(row_max, softmax_scale_log2, -new_max_scaled);
                        float acc_scale;
                        if (acc_scale_log2 >= -8.0f) {
                            acc_scale = 1.0f;
                            new_max_scaled = row_max_scaled;
                        } else {
#if __CUDA_ARCH__ == 1030
                            float _exp2_10 = approx_exp2(acc_scale_log2);
                            acc_scale = ((row_max > -CUDART_INF_F) ? _exp2_10 : 1.0f);
#else
                            float _exp2_12 = approx_exp2(acc_scale_log2);
                            acc_scale = ((row_max > -CUDART_INF_F) ? _exp2_12 : 1.0f);
#endif
                            row_max = new_max;
                            row_max_scaled = new_max_scaled;
                        }
                        row_state[warp % 4 * 32 + lane + scale_off] = acc_scale;
                        if (warp % 4 < 2) {
                            mbarrier_arrive(corr_sig_lo_addr + (stage) * 8);
                        } else {
                            mbarrier_arrive(corr_sig_hi_addr + (stage) * 8);
                        }
                        float block_sum = 0.0f;
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        float2 block_sum2 = make_float2(0.0f, 0.0f);
                        float block_max = tile_max;
                        float block_max_scaled = ((block_max > -CUDART_INF_F) ? block_max * softmax_scale_log2 : 0.0f);
#if __CUDA_ARCH__ == 1030
                        float _exp2_11 = approx_exp2(block_max_scaled - new_max_scaled - 2.5849626f);
                        float p_scale = ((block_max > -CUDART_INF_F) ? _exp2_11 : 0.0f);
#else
                        float _exp2_13 = approx_exp2(block_max_scaled - new_max_scaled - 2.5849626f);
                        float p_scale = ((block_max > -CUDART_INF_F) ? _exp2_13 : 0.0f);
#endif
                        uint16_t _e4m3x2_f32_0;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_0) : "f"(p_scale), "f"(p_scale));
                        unsigned int sf_packed[1];
                        sf_packed[0] = (unsigned int)_e4m3x2_f32_0 * 65537;
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_12 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_13 = {2.5849626f - block_max_scaled, 2.5849626f - block_max_scaled};
                        float2 _fma_pair_14 = fma_f32x2(make_float2((sv + 0)[0], (sv + 0)[1]), _fma_b2_12, _fma_c2_13);
                        (sv + 0)[0] = _fma_pair_14.x;
                        (sv + 0)[1] = _fma_pair_14.y;
                        float2 _fma_pair_15 = fma_f32x2(make_float2((sv + 0)[2], (sv + 0)[3]), _fma_b2_12, _fma_c2_13);
                        (sv + 0)[2] = _fma_pair_15.x;
                        (sv + 0)[3] = _fma_pair_15.y;
                        float2 _fma_pair_16 = fma_f32x2(make_float2((sv + 0)[4], (sv + 0)[5]), _fma_b2_12, _fma_c2_13);
                        (sv + 0)[4] = _fma_pair_16.x;
                        (sv + 0)[5] = _fma_pair_16.y;
                        float2 _fma_pair_17 = fma_f32x2(make_float2((sv + 0)[6], (sv + 0)[7]), _fma_b2_12, _fma_c2_13);
                        (sv + 0)[6] = _fma_pair_17.x;
                        (sv + 0)[7] = _fma_pair_17.y;
                        float2 _fma_pair_18 = fma_f32x2(make_float2((sv + 0)[8], (sv + 0)[9]), _fma_b2_12, _fma_c2_13);
                        (sv + 0)[8] = _fma_pair_18.x;
                        (sv + 0)[9] = _fma_pair_18.y;
                        float2 _fma_pair_19 = fma_f32x2(make_float2((sv + 0)[10], (sv + 0)[11]), _fma_b2_12, _fma_c2_13);
                        (sv + 0)[10] = _fma_pair_19.x;
                        (sv + 0)[11] = _fma_pair_19.y;
                        float2 _fma_pair_20 = fma_f32x2(make_float2((sv + 0)[12], (sv + 0)[13]), _fma_b2_12, _fma_c2_13);
                        (sv + 0)[12] = _fma_pair_20.x;
                        (sv + 0)[13] = _fma_pair_20.y;
                        float2 _fma_pair_21 = fma_f32x2(make_float2((sv + 0)[14], (sv + 0)[15]), _fma_b2_12, _fma_c2_13);
                        (sv + 0)[14] = _fma_pair_21.x;
                        (sv + 0)[15] = _fma_pair_21.y;
                        float2 _fma_pair_22 = fma_f32x2(make_float2((sv + 0)[16], (sv + 0)[17]), _fma_b2_12, _fma_c2_13);
                        (sv + 0)[16] = _fma_pair_22.x;
                        (sv + 0)[17] = _fma_pair_22.y;
                        float2 _fma_pair_23 = fma_f32x2(make_float2((sv + 0)[18], (sv + 0)[19]), _fma_b2_12, _fma_c2_13);
                        (sv + 0)[18] = _fma_pair_23.x;
                        (sv + 0)[19] = _fma_pair_23.y;
                        float2 _fma_pair_24 = fma_f32x2(make_float2((sv + 0)[20], (sv + 0)[21]), _fma_b2_12, _fma_c2_13);
                        (sv + 0)[20] = _fma_pair_24.x;
                        (sv + 0)[21] = _fma_pair_24.y;
                        float2 _fma_pair_25 = fma_f32x2(make_float2((sv + 0)[22], (sv + 0)[23]), _fma_b2_12, _fma_c2_13);
                        (sv + 0)[22] = _fma_pair_25.x;
                        (sv + 0)[23] = _fma_pair_25.y;
                        float2 _fma_pair_26 = fma_f32x2(make_float2((sv + 0)[24], (sv + 0)[25]), _fma_b2_12, _fma_c2_13);
                        (sv + 0)[24] = _fma_pair_26.x;
                        (sv + 0)[25] = _fma_pair_26.y;
                        float2 _fma_pair_27 = fma_f32x2(make_float2((sv + 0)[26], (sv + 0)[27]), _fma_b2_12, _fma_c2_13);
                        (sv + 0)[26] = _fma_pair_27.x;
                        (sv + 0)[27] = _fma_pair_27.y;
                        float2 _fma_pair_28 = fma_f32x2(make_float2((sv + 0)[28], (sv + 0)[29]), _fma_b2_12, _fma_c2_13);
                        (sv + 0)[28] = _fma_pair_28.x;
                        (sv + 0)[29] = _fma_pair_28.y;
                        float2 _fma_pair_29 = fma_f32x2(make_float2((sv + 0)[30], (sv + 0)[31]), _fma_b2_12, _fma_c2_13);
                        (sv + 0)[30] = _fma_pair_29.x;
                        (sv + 0)[31] = _fma_pair_29.y;
#else
                        const float2 _fma_b2_14 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_15 = {2.5849626f - block_max_scaled, 2.5849626f - block_max_scaled};
                        float2 _fma_pair_16 = fma_f32x2(make_float2((sv + 0)[0], (sv + 0)[1]), _fma_b2_14, _fma_c2_15);
                        (sv + 0)[0] = _fma_pair_16.x;
                        (sv + 0)[1] = _fma_pair_16.y;
                        float2 _fma_pair_17 = fma_f32x2(make_float2((sv + 0)[2], (sv + 0)[3]), _fma_b2_14, _fma_c2_15);
                        (sv + 0)[2] = _fma_pair_17.x;
                        (sv + 0)[3] = _fma_pair_17.y;
                        float2 _fma_pair_18 = fma_f32x2(make_float2((sv + 0)[4], (sv + 0)[5]), _fma_b2_14, _fma_c2_15);
                        (sv + 0)[4] = _fma_pair_18.x;
                        (sv + 0)[5] = _fma_pair_18.y;
                        float2 _fma_pair_19 = fma_f32x2(make_float2((sv + 0)[6], (sv + 0)[7]), _fma_b2_14, _fma_c2_15);
                        (sv + 0)[6] = _fma_pair_19.x;
                        (sv + 0)[7] = _fma_pair_19.y;
                        float2 _fma_pair_20 = fma_f32x2(make_float2((sv + 0)[8], (sv + 0)[9]), _fma_b2_14, _fma_c2_15);
                        (sv + 0)[8] = _fma_pair_20.x;
                        (sv + 0)[9] = _fma_pair_20.y;
                        float2 _fma_pair_21 = fma_f32x2(make_float2((sv + 0)[10], (sv + 0)[11]), _fma_b2_14, _fma_c2_15);
                        (sv + 0)[10] = _fma_pair_21.x;
                        (sv + 0)[11] = _fma_pair_21.y;
                        float2 _fma_pair_22 = fma_f32x2(make_float2((sv + 0)[12], (sv + 0)[13]), _fma_b2_14, _fma_c2_15);
                        (sv + 0)[12] = _fma_pair_22.x;
                        (sv + 0)[13] = _fma_pair_22.y;
                        float2 _fma_pair_23 = fma_f32x2(make_float2((sv + 0)[14], (sv + 0)[15]), _fma_b2_14, _fma_c2_15);
                        (sv + 0)[14] = _fma_pair_23.x;
                        (sv + 0)[15] = _fma_pair_23.y;
                        float2 _fma_pair_24 = fma_f32x2(make_float2((sv + 0)[16], (sv + 0)[17]), _fma_b2_14, _fma_c2_15);
                        (sv + 0)[16] = _fma_pair_24.x;
                        (sv + 0)[17] = _fma_pair_24.y;
                        float2 _fma_pair_25 = fma_f32x2(make_float2((sv + 0)[18], (sv + 0)[19]), _fma_b2_14, _fma_c2_15);
                        (sv + 0)[18] = _fma_pair_25.x;
                        (sv + 0)[19] = _fma_pair_25.y;
                        float2 _fma_pair_26 = fma_f32x2(make_float2((sv + 0)[20], (sv + 0)[21]), _fma_b2_14, _fma_c2_15);
                        (sv + 0)[20] = _fma_pair_26.x;
                        (sv + 0)[21] = _fma_pair_26.y;
                        float2 _fma_pair_27 = fma_f32x2(make_float2((sv + 0)[22], (sv + 0)[23]), _fma_b2_14, _fma_c2_15);
                        (sv + 0)[22] = _fma_pair_27.x;
                        (sv + 0)[23] = _fma_pair_27.y;
                        float2 _fma_pair_28 = fma_f32x2(make_float2((sv + 0)[24], (sv + 0)[25]), _fma_b2_14, _fma_c2_15);
                        (sv + 0)[24] = _fma_pair_28.x;
                        (sv + 0)[25] = _fma_pair_28.y;
                        float2 _fma_pair_29 = fma_f32x2(make_float2((sv + 0)[26], (sv + 0)[27]), _fma_b2_14, _fma_c2_15);
                        (sv + 0)[26] = _fma_pair_29.x;
                        (sv + 0)[27] = _fma_pair_29.y;
                        float2 _fma_pair_30 = fma_f32x2(make_float2((sv + 0)[28], (sv + 0)[29]), _fma_b2_14, _fma_c2_15);
                        (sv + 0)[28] = _fma_pair_30.x;
                        (sv + 0)[29] = _fma_pair_30.y;
                        float2 _fma_pair_31 = fma_f32x2(make_float2((sv + 0)[30], (sv + 0)[31]), _fma_b2_14, _fma_c2_15);
                        (sv + 0)[30] = _fma_pair_31.x;
                        (sv + 0)[31] = _fma_pair_31.y;
#endif
                        {
                            #pragma unroll
#if __CUDA_ARCH__ == 1030
                            for (int _le = 0; _le < 32; _le++) {
#else
                            for (int _le = 0; _le < 24; _le++) {
#endif
                                sv[_le] = approx_exp2(sv[_le]);
                            }
#if __CUDA_ARCH__ == 1030
#else
                            float2 _exp2_pair_32 = ex2_emulation_f32x2_value(make_float2(sv[24], sv[25]));
                            sv[24] = _exp2_pair_32.x;
                            sv[25] = _exp2_pair_32.y;
                            float2 _exp2_pair_33 = ex2_emulation_f32x2_value(make_float2(sv[26], sv[27]));
                            sv[26] = _exp2_pair_33.x;
                            sv[27] = _exp2_pair_33.y;
                            float2 _exp2_pair_34 = ex2_emulation_f32x2_value(make_float2(sv[28], sv[29]));
                            sv[28] = _exp2_pair_34.x;
                            sv[29] = _exp2_pair_34.y;
                            float2 _exp2_pair_35 = ex2_emulation_f32x2_value(make_float2(sv[30], sv[31]));
                            sv[30] = _exp2_pair_35.x;
                            sv[31] = _exp2_pair_35.y;
#endif
                        }
                        float2 partial = make_float2(sv[0], sv[1]);
                        #pragma unroll
                        for (int pair = 2; pair < 16; pair += 2) {
                            partial = add_f32x2(partial, make_float2(sv[pair], sv[pair + 1]));
                        }
                        float2 frag_sum2 = partial;
                        float2 scale2 = make_float2(p_scale, p_scale);
                        block_sum2 = fma_f32x2_rn_ftz(frag_sum2, scale2, block_sum2);
                        float2 partial_0 = make_float2(sv[16], sv[17]);
                        #pragma unroll
                        for (int pair_1 = 2; pair_1 < 16; pair_1 += 2) {
                            partial_0 = add_f32x2(partial_0, make_float2(sv[16 + pair_1], sv[16 + (pair_1 + 1)]));
                        }
                        frag_sum2 = partial_0;
                        block_sum2 = fma_f32x2_rn_ftz(frag_sum2, scale2, block_sum2);
                        unsigned int _fp4_0[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_0[0]) : "f"(sv[0]), "f"(sv[1]), "f"(sv[2]), "f"(sv[3]), "f"(sv[4]), "f"(sv[5]), "f"(sv[6]), "f"(sv[7]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_0[1]) : "f"(sv[8]), "f"(sv[9]), "f"(sv[10]), "f"(sv[11]), "f"(sv[12]), "f"(sv[13]), "f"(sv[14]), "f"(sv[15]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_0[2]) : "f"(sv[16]), "f"(sv[17]), "f"(sv[18]), "f"(sv[19]), "f"(sv[20]), "f"(sv[21]), "f"(sv[22]), "f"(sv[23]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_0[3]) : "f"(sv[24]), "f"(sv[25]), "f"(sv[26]), "f"(sv[27]), "f"(sv[28]), "f"(sv[29]), "f"(sv[30]), "f"(sv[31]));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_p_addr + p_slot * 8192 + (unsigned int)((warp % 4 * 32 + lane) * 64 ^ ((warp % 4 * 32 + lane) * 64 >> 7 & 3) << 4))), "r"(_fp4_0[0]), "r"(_fp4_0[1]), "r"(_fp4_0[2]), "r"(_fp4_0[3]) : "memory");
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_30 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_31 = {2.5849626f - block_max_scaled, 2.5849626f - block_max_scaled};
                        float2 _fma_pair_32 = fma_f32x2(make_float2((sv + 32)[0], (sv + 32)[1]), _fma_b2_30, _fma_c2_31);
                        (sv + 32)[0] = _fma_pair_32.x;
                        (sv + 32)[1] = _fma_pair_32.y;
                        float2 _fma_pair_33 = fma_f32x2(make_float2((sv + 32)[2], (sv + 32)[3]), _fma_b2_30, _fma_c2_31);
                        (sv + 32)[2] = _fma_pair_33.x;
                        (sv + 32)[3] = _fma_pair_33.y;
                        float2 _fma_pair_34 = fma_f32x2(make_float2((sv + 32)[4], (sv + 32)[5]), _fma_b2_30, _fma_c2_31);
                        (sv + 32)[4] = _fma_pair_34.x;
                        (sv + 32)[5] = _fma_pair_34.y;
                        float2 _fma_pair_35 = fma_f32x2(make_float2((sv + 32)[6], (sv + 32)[7]), _fma_b2_30, _fma_c2_31);
                        (sv + 32)[6] = _fma_pair_35.x;
                        (sv + 32)[7] = _fma_pair_35.y;
                        float2 _fma_pair_36 = fma_f32x2(make_float2((sv + 32)[8], (sv + 32)[9]), _fma_b2_30, _fma_c2_31);
                        (sv + 32)[8] = _fma_pair_36.x;
                        (sv + 32)[9] = _fma_pair_36.y;
                        float2 _fma_pair_37 = fma_f32x2(make_float2((sv + 32)[10], (sv + 32)[11]), _fma_b2_30, _fma_c2_31);
                        (sv + 32)[10] = _fma_pair_37.x;
                        (sv + 32)[11] = _fma_pair_37.y;
                        float2 _fma_pair_38 = fma_f32x2(make_float2((sv + 32)[12], (sv + 32)[13]), _fma_b2_30, _fma_c2_31);
                        (sv + 32)[12] = _fma_pair_38.x;
                        (sv + 32)[13] = _fma_pair_38.y;
                        float2 _fma_pair_39 = fma_f32x2(make_float2((sv + 32)[14], (sv + 32)[15]), _fma_b2_30, _fma_c2_31);
                        (sv + 32)[14] = _fma_pair_39.x;
                        (sv + 32)[15] = _fma_pair_39.y;
                        float2 _fma_pair_40 = fma_f32x2(make_float2((sv + 32)[16], (sv + 32)[17]), _fma_b2_30, _fma_c2_31);
                        (sv + 32)[16] = _fma_pair_40.x;
                        (sv + 32)[17] = _fma_pair_40.y;
                        float2 _fma_pair_41 = fma_f32x2(make_float2((sv + 32)[18], (sv + 32)[19]), _fma_b2_30, _fma_c2_31);
                        (sv + 32)[18] = _fma_pair_41.x;
                        (sv + 32)[19] = _fma_pair_41.y;
                        float2 _fma_pair_42 = fma_f32x2(make_float2((sv + 32)[20], (sv + 32)[21]), _fma_b2_30, _fma_c2_31);
                        (sv + 32)[20] = _fma_pair_42.x;
                        (sv + 32)[21] = _fma_pair_42.y;
                        float2 _fma_pair_43 = fma_f32x2(make_float2((sv + 32)[22], (sv + 32)[23]), _fma_b2_30, _fma_c2_31);
                        (sv + 32)[22] = _fma_pair_43.x;
                        (sv + 32)[23] = _fma_pair_43.y;
                        float2 _fma_pair_44 = fma_f32x2(make_float2((sv + 32)[24], (sv + 32)[25]), _fma_b2_30, _fma_c2_31);
                        (sv + 32)[24] = _fma_pair_44.x;
                        (sv + 32)[25] = _fma_pair_44.y;
                        float2 _fma_pair_45 = fma_f32x2(make_float2((sv + 32)[26], (sv + 32)[27]), _fma_b2_30, _fma_c2_31);
                        (sv + 32)[26] = _fma_pair_45.x;
                        (sv + 32)[27] = _fma_pair_45.y;
                        float2 _fma_pair_46 = fma_f32x2(make_float2((sv + 32)[28], (sv + 32)[29]), _fma_b2_30, _fma_c2_31);
                        (sv + 32)[28] = _fma_pair_46.x;
                        (sv + 32)[29] = _fma_pair_46.y;
                        float2 _fma_pair_47 = fma_f32x2(make_float2((sv + 32)[30], (sv + 32)[31]), _fma_b2_30, _fma_c2_31);
                        (sv + 32)[30] = _fma_pair_47.x;
                        (sv + 32)[31] = _fma_pair_47.y;
#else
                        const float2 _fma_b2_36 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_37 = {2.5849626f - block_max_scaled, 2.5849626f - block_max_scaled};
                        float2 _fma_pair_38 = fma_f32x2(make_float2((sv + 32)[0], (sv + 32)[1]), _fma_b2_36, _fma_c2_37);
                        (sv + 32)[0] = _fma_pair_38.x;
                        (sv + 32)[1] = _fma_pair_38.y;
                        float2 _fma_pair_39 = fma_f32x2(make_float2((sv + 32)[2], (sv + 32)[3]), _fma_b2_36, _fma_c2_37);
                        (sv + 32)[2] = _fma_pair_39.x;
                        (sv + 32)[3] = _fma_pair_39.y;
                        float2 _fma_pair_40 = fma_f32x2(make_float2((sv + 32)[4], (sv + 32)[5]), _fma_b2_36, _fma_c2_37);
                        (sv + 32)[4] = _fma_pair_40.x;
                        (sv + 32)[5] = _fma_pair_40.y;
                        float2 _fma_pair_41 = fma_f32x2(make_float2((sv + 32)[6], (sv + 32)[7]), _fma_b2_36, _fma_c2_37);
                        (sv + 32)[6] = _fma_pair_41.x;
                        (sv + 32)[7] = _fma_pair_41.y;
                        float2 _fma_pair_42 = fma_f32x2(make_float2((sv + 32)[8], (sv + 32)[9]), _fma_b2_36, _fma_c2_37);
                        (sv + 32)[8] = _fma_pair_42.x;
                        (sv + 32)[9] = _fma_pair_42.y;
                        float2 _fma_pair_43 = fma_f32x2(make_float2((sv + 32)[10], (sv + 32)[11]), _fma_b2_36, _fma_c2_37);
                        (sv + 32)[10] = _fma_pair_43.x;
                        (sv + 32)[11] = _fma_pair_43.y;
                        float2 _fma_pair_44 = fma_f32x2(make_float2((sv + 32)[12], (sv + 32)[13]), _fma_b2_36, _fma_c2_37);
                        (sv + 32)[12] = _fma_pair_44.x;
                        (sv + 32)[13] = _fma_pair_44.y;
                        float2 _fma_pair_45 = fma_f32x2(make_float2((sv + 32)[14], (sv + 32)[15]), _fma_b2_36, _fma_c2_37);
                        (sv + 32)[14] = _fma_pair_45.x;
                        (sv + 32)[15] = _fma_pair_45.y;
                        float2 _fma_pair_46 = fma_f32x2(make_float2((sv + 32)[16], (sv + 32)[17]), _fma_b2_36, _fma_c2_37);
                        (sv + 32)[16] = _fma_pair_46.x;
                        (sv + 32)[17] = _fma_pair_46.y;
                        float2 _fma_pair_47 = fma_f32x2(make_float2((sv + 32)[18], (sv + 32)[19]), _fma_b2_36, _fma_c2_37);
                        (sv + 32)[18] = _fma_pair_47.x;
                        (sv + 32)[19] = _fma_pair_47.y;
                        float2 _fma_pair_48 = fma_f32x2(make_float2((sv + 32)[20], (sv + 32)[21]), _fma_b2_36, _fma_c2_37);
                        (sv + 32)[20] = _fma_pair_48.x;
                        (sv + 32)[21] = _fma_pair_48.y;
                        float2 _fma_pair_49 = fma_f32x2(make_float2((sv + 32)[22], (sv + 32)[23]), _fma_b2_36, _fma_c2_37);
                        (sv + 32)[22] = _fma_pair_49.x;
                        (sv + 32)[23] = _fma_pair_49.y;
                        float2 _fma_pair_50 = fma_f32x2(make_float2((sv + 32)[24], (sv + 32)[25]), _fma_b2_36, _fma_c2_37);
                        (sv + 32)[24] = _fma_pair_50.x;
                        (sv + 32)[25] = _fma_pair_50.y;
                        float2 _fma_pair_51 = fma_f32x2(make_float2((sv + 32)[26], (sv + 32)[27]), _fma_b2_36, _fma_c2_37);
                        (sv + 32)[26] = _fma_pair_51.x;
                        (sv + 32)[27] = _fma_pair_51.y;
                        float2 _fma_pair_52 = fma_f32x2(make_float2((sv + 32)[28], (sv + 32)[29]), _fma_b2_36, _fma_c2_37);
                        (sv + 32)[28] = _fma_pair_52.x;
                        (sv + 32)[29] = _fma_pair_52.y;
                        float2 _fma_pair_53 = fma_f32x2(make_float2((sv + 32)[30], (sv + 32)[31]), _fma_b2_36, _fma_c2_37);
                        (sv + 32)[30] = _fma_pair_53.x;
                        (sv + 32)[31] = _fma_pair_53.y;
#endif
                        {
                            #pragma unroll
#if __CUDA_ARCH__ == 1030
                            for (int _le = 0; _le < 32; _le++) {
#else
                            for (int _le = 0; _le < 24; _le++) {
#endif
                                sv[_le + 32] = approx_exp2(sv[_le + 32]);
                            }
#if __CUDA_ARCH__ == 1030
#else
                            float2 _exp2_pair_54 = ex2_emulation_f32x2_value(make_float2(sv[56], sv[57]));
                            sv[56] = _exp2_pair_54.x;
                            sv[57] = _exp2_pair_54.y;
                            float2 _exp2_pair_55 = ex2_emulation_f32x2_value(make_float2(sv[58], sv[59]));
                            sv[58] = _exp2_pair_55.x;
                            sv[59] = _exp2_pair_55.y;
                            float2 _exp2_pair_56 = ex2_emulation_f32x2_value(make_float2(sv[60], sv[61]));
                            sv[60] = _exp2_pair_56.x;
                            sv[61] = _exp2_pair_56.y;
                            float2 _exp2_pair_57 = ex2_emulation_f32x2_value(make_float2(sv[62], sv[63]));
                            sv[62] = _exp2_pair_57.x;
                            sv[63] = _exp2_pair_57.y;
#endif
                        }
                        float2 partial_1 = make_float2(sv[32], sv[33]);
                        #pragma unroll
                        for (int pair_2 = 2; pair_2 < 16; pair_2 += 2) {
                            partial_1 = add_f32x2(partial_1, make_float2(sv[32 + pair_2], sv[32 + (pair_2 + 1)]));
                        }
                        float2 frag_sum2_2 = partial_1;
                        float2 scale2_3 = make_float2(p_scale, p_scale);
                        block_sum2 = fma_f32x2_rn_ftz(frag_sum2_2, scale2_3, block_sum2);
                        float2 partial_4 = make_float2(sv[48], sv[49]);
                        #pragma unroll
                        for (int pair_3 = 2; pair_3 < 16; pair_3 += 2) {
                            partial_4 = add_f32x2(partial_4, make_float2(sv[48 + pair_3], sv[48 + (pair_3 + 1)]));
                        }
                        frag_sum2_2 = partial_4;
                        block_sum2 = fma_f32x2_rn_ftz(frag_sum2_2, scale2_3, block_sum2);
                        unsigned int _fp4_1[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_1[0]) : "f"(sv[32]), "f"(sv[33]), "f"(sv[34]), "f"(sv[35]), "f"(sv[36]), "f"(sv[37]), "f"(sv[38]), "f"(sv[39]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_1[1]) : "f"(sv[40]), "f"(sv[41]), "f"(sv[42]), "f"(sv[43]), "f"(sv[44]), "f"(sv[45]), "f"(sv[46]), "f"(sv[47]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_1[2]) : "f"(sv[48]), "f"(sv[49]), "f"(sv[50]), "f"(sv[51]), "f"(sv[52]), "f"(sv[53]), "f"(sv[54]), "f"(sv[55]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_1[3]) : "f"(sv[56]), "f"(sv[57]), "f"(sv[58]), "f"(sv[59]), "f"(sv[60]), "f"(sv[61]), "f"(sv[62]), "f"(sv[63]));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_p_addr + p_slot * 8192 + (unsigned int)((warp % 4 * 32 + lane) * 64 + 16 ^ ((warp % 4 * 32 + lane) * 64 + 16 >> 7 & 3) << 4))), "r"(_fp4_1[0]), "r"(_fp4_1[1]), "r"(_fp4_1[2]), "r"(_fp4_1[3]) : "memory");
                        int pv_sfa_lo_col = (unsigned int)408 + stage * 8 + (unsigned int)(warp % 4) + (p_slot & 1) * 24;
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(taddr + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)pv_sfa_lo_col), "r"(sf_packed[0]));
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_48 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_49 = {2.5849626f - block_max_scaled, 2.5849626f - block_max_scaled};
                        float2 _fma_pair_50 = fma_f32x2(make_float2((sv + 64)[0], (sv + 64)[1]), _fma_b2_48, _fma_c2_49);
                        (sv + 64)[0] = _fma_pair_50.x;
                        (sv + 64)[1] = _fma_pair_50.y;
                        float2 _fma_pair_51 = fma_f32x2(make_float2((sv + 64)[2], (sv + 64)[3]), _fma_b2_48, _fma_c2_49);
                        (sv + 64)[2] = _fma_pair_51.x;
                        (sv + 64)[3] = _fma_pair_51.y;
                        float2 _fma_pair_52 = fma_f32x2(make_float2((sv + 64)[4], (sv + 64)[5]), _fma_b2_48, _fma_c2_49);
                        (sv + 64)[4] = _fma_pair_52.x;
                        (sv + 64)[5] = _fma_pair_52.y;
                        float2 _fma_pair_53 = fma_f32x2(make_float2((sv + 64)[6], (sv + 64)[7]), _fma_b2_48, _fma_c2_49);
                        (sv + 64)[6] = _fma_pair_53.x;
                        (sv + 64)[7] = _fma_pair_53.y;
                        float2 _fma_pair_54 = fma_f32x2(make_float2((sv + 64)[8], (sv + 64)[9]), _fma_b2_48, _fma_c2_49);
                        (sv + 64)[8] = _fma_pair_54.x;
                        (sv + 64)[9] = _fma_pair_54.y;
                        float2 _fma_pair_55 = fma_f32x2(make_float2((sv + 64)[10], (sv + 64)[11]), _fma_b2_48, _fma_c2_49);
                        (sv + 64)[10] = _fma_pair_55.x;
                        (sv + 64)[11] = _fma_pair_55.y;
                        float2 _fma_pair_56 = fma_f32x2(make_float2((sv + 64)[12], (sv + 64)[13]), _fma_b2_48, _fma_c2_49);
                        (sv + 64)[12] = _fma_pair_56.x;
                        (sv + 64)[13] = _fma_pair_56.y;
                        float2 _fma_pair_57 = fma_f32x2(make_float2((sv + 64)[14], (sv + 64)[15]), _fma_b2_48, _fma_c2_49);
                        (sv + 64)[14] = _fma_pair_57.x;
                        (sv + 64)[15] = _fma_pair_57.y;
                        float2 _fma_pair_58 = fma_f32x2(make_float2((sv + 64)[16], (sv + 64)[17]), _fma_b2_48, _fma_c2_49);
                        (sv + 64)[16] = _fma_pair_58.x;
                        (sv + 64)[17] = _fma_pair_58.y;
                        float2 _fma_pair_59 = fma_f32x2(make_float2((sv + 64)[18], (sv + 64)[19]), _fma_b2_48, _fma_c2_49);
                        (sv + 64)[18] = _fma_pair_59.x;
                        (sv + 64)[19] = _fma_pair_59.y;
                        float2 _fma_pair_60 = fma_f32x2(make_float2((sv + 64)[20], (sv + 64)[21]), _fma_b2_48, _fma_c2_49);
                        (sv + 64)[20] = _fma_pair_60.x;
                        (sv + 64)[21] = _fma_pair_60.y;
                        float2 _fma_pair_61 = fma_f32x2(make_float2((sv + 64)[22], (sv + 64)[23]), _fma_b2_48, _fma_c2_49);
                        (sv + 64)[22] = _fma_pair_61.x;
                        (sv + 64)[23] = _fma_pair_61.y;
                        float2 _fma_pair_62 = fma_f32x2(make_float2((sv + 64)[24], (sv + 64)[25]), _fma_b2_48, _fma_c2_49);
                        (sv + 64)[24] = _fma_pair_62.x;
                        (sv + 64)[25] = _fma_pair_62.y;
                        float2 _fma_pair_63 = fma_f32x2(make_float2((sv + 64)[26], (sv + 64)[27]), _fma_b2_48, _fma_c2_49);
                        (sv + 64)[26] = _fma_pair_63.x;
                        (sv + 64)[27] = _fma_pair_63.y;
                        float2 _fma_pair_64 = fma_f32x2(make_float2((sv + 64)[28], (sv + 64)[29]), _fma_b2_48, _fma_c2_49);
                        (sv + 64)[28] = _fma_pair_64.x;
                        (sv + 64)[29] = _fma_pair_64.y;
                        float2 _fma_pair_65 = fma_f32x2(make_float2((sv + 64)[30], (sv + 64)[31]), _fma_b2_48, _fma_c2_49);
                        (sv + 64)[30] = _fma_pair_65.x;
                        (sv + 64)[31] = _fma_pair_65.y;
#else
                        const float2 _fma_b2_58 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_59 = {2.5849626f - block_max_scaled, 2.5849626f - block_max_scaled};
                        float2 _fma_pair_60 = fma_f32x2(make_float2((sv + 64)[0], (sv + 64)[1]), _fma_b2_58, _fma_c2_59);
                        (sv + 64)[0] = _fma_pair_60.x;
                        (sv + 64)[1] = _fma_pair_60.y;
                        float2 _fma_pair_61 = fma_f32x2(make_float2((sv + 64)[2], (sv + 64)[3]), _fma_b2_58, _fma_c2_59);
                        (sv + 64)[2] = _fma_pair_61.x;
                        (sv + 64)[3] = _fma_pair_61.y;
                        float2 _fma_pair_62 = fma_f32x2(make_float2((sv + 64)[4], (sv + 64)[5]), _fma_b2_58, _fma_c2_59);
                        (sv + 64)[4] = _fma_pair_62.x;
                        (sv + 64)[5] = _fma_pair_62.y;
                        float2 _fma_pair_63 = fma_f32x2(make_float2((sv + 64)[6], (sv + 64)[7]), _fma_b2_58, _fma_c2_59);
                        (sv + 64)[6] = _fma_pair_63.x;
                        (sv + 64)[7] = _fma_pair_63.y;
                        float2 _fma_pair_64 = fma_f32x2(make_float2((sv + 64)[8], (sv + 64)[9]), _fma_b2_58, _fma_c2_59);
                        (sv + 64)[8] = _fma_pair_64.x;
                        (sv + 64)[9] = _fma_pair_64.y;
                        float2 _fma_pair_65 = fma_f32x2(make_float2((sv + 64)[10], (sv + 64)[11]), _fma_b2_58, _fma_c2_59);
                        (sv + 64)[10] = _fma_pair_65.x;
                        (sv + 64)[11] = _fma_pair_65.y;
                        float2 _fma_pair_66 = fma_f32x2(make_float2((sv + 64)[12], (sv + 64)[13]), _fma_b2_58, _fma_c2_59);
                        (sv + 64)[12] = _fma_pair_66.x;
                        (sv + 64)[13] = _fma_pair_66.y;
                        float2 _fma_pair_67 = fma_f32x2(make_float2((sv + 64)[14], (sv + 64)[15]), _fma_b2_58, _fma_c2_59);
                        (sv + 64)[14] = _fma_pair_67.x;
                        (sv + 64)[15] = _fma_pair_67.y;
                        float2 _fma_pair_68 = fma_f32x2(make_float2((sv + 64)[16], (sv + 64)[17]), _fma_b2_58, _fma_c2_59);
                        (sv + 64)[16] = _fma_pair_68.x;
                        (sv + 64)[17] = _fma_pair_68.y;
                        float2 _fma_pair_69 = fma_f32x2(make_float2((sv + 64)[18], (sv + 64)[19]), _fma_b2_58, _fma_c2_59);
                        (sv + 64)[18] = _fma_pair_69.x;
                        (sv + 64)[19] = _fma_pair_69.y;
                        float2 _fma_pair_70 = fma_f32x2(make_float2((sv + 64)[20], (sv + 64)[21]), _fma_b2_58, _fma_c2_59);
                        (sv + 64)[20] = _fma_pair_70.x;
                        (sv + 64)[21] = _fma_pair_70.y;
                        float2 _fma_pair_71 = fma_f32x2(make_float2((sv + 64)[22], (sv + 64)[23]), _fma_b2_58, _fma_c2_59);
                        (sv + 64)[22] = _fma_pair_71.x;
                        (sv + 64)[23] = _fma_pair_71.y;
                        float2 _fma_pair_72 = fma_f32x2(make_float2((sv + 64)[24], (sv + 64)[25]), _fma_b2_58, _fma_c2_59);
                        (sv + 64)[24] = _fma_pair_72.x;
                        (sv + 64)[25] = _fma_pair_72.y;
                        float2 _fma_pair_73 = fma_f32x2(make_float2((sv + 64)[26], (sv + 64)[27]), _fma_b2_58, _fma_c2_59);
                        (sv + 64)[26] = _fma_pair_73.x;
                        (sv + 64)[27] = _fma_pair_73.y;
                        float2 _fma_pair_74 = fma_f32x2(make_float2((sv + 64)[28], (sv + 64)[29]), _fma_b2_58, _fma_c2_59);
                        (sv + 64)[28] = _fma_pair_74.x;
                        (sv + 64)[29] = _fma_pair_74.y;
                        float2 _fma_pair_75 = fma_f32x2(make_float2((sv + 64)[30], (sv + 64)[31]), _fma_b2_58, _fma_c2_59);
                        (sv + 64)[30] = _fma_pair_75.x;
                        (sv + 64)[31] = _fma_pair_75.y;
#endif
                        {
                            #pragma unroll
#if __CUDA_ARCH__ == 1030
                            for (int _le = 0; _le < 32; _le++) {
#else
                            for (int _le = 0; _le < 24; _le++) {
#endif
                                sv[_le + 64] = approx_exp2(sv[_le + 64]);
                            }
#if __CUDA_ARCH__ == 1030
#else
                            float2 _exp2_pair_76 = ex2_emulation_f32x2_value(make_float2(sv[88], sv[89]));
                            sv[88] = _exp2_pair_76.x;
                            sv[89] = _exp2_pair_76.y;
                            float2 _exp2_pair_77 = ex2_emulation_f32x2_value(make_float2(sv[90], sv[91]));
                            sv[90] = _exp2_pair_77.x;
                            sv[91] = _exp2_pair_77.y;
                            float2 _exp2_pair_78 = ex2_emulation_f32x2_value(make_float2(sv[92], sv[93]));
                            sv[92] = _exp2_pair_78.x;
                            sv[93] = _exp2_pair_78.y;
                            float2 _exp2_pair_79 = ex2_emulation_f32x2_value(make_float2(sv[94], sv[95]));
                            sv[94] = _exp2_pair_79.x;
                            sv[95] = _exp2_pair_79.y;
#endif
                        }
                        float2 partial_5 = make_float2(sv[64], sv[65]);
                        #pragma unroll
                        for (int pair_4 = 2; pair_4 < 16; pair_4 += 2) {
                            partial_5 = add_f32x2(partial_5, make_float2(sv[64 + pair_4], sv[64 + (pair_4 + 1)]));
                        }
                        float2 scale2_6 = make_float2(p_scale, p_scale);
                        block_sum2 = fma_f32x2_rn_ftz(partial_5, scale2_6, block_sum2);
                        float2 partial_7 = make_float2(sv[80], sv[81]);
                        #pragma unroll
                        for (int pair_5 = 2; pair_5 < 16; pair_5 += 2) {
                            partial_7 = add_f32x2(partial_7, make_float2(sv[80 + pair_5], sv[80 + (pair_5 + 1)]));
                        }
                        block_sum2 = fma_f32x2_rn_ftz(partial_7, scale2_6, block_sum2);
                        unsigned int _fp4_2[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_2[0]) : "f"(sv[64]), "f"(sv[65]), "f"(sv[66]), "f"(sv[67]), "f"(sv[68]), "f"(sv[69]), "f"(sv[70]), "f"(sv[71]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_2[1]) : "f"(sv[72]), "f"(sv[73]), "f"(sv[74]), "f"(sv[75]), "f"(sv[76]), "f"(sv[77]), "f"(sv[78]), "f"(sv[79]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_2[2]) : "f"(sv[80]), "f"(sv[81]), "f"(sv[82]), "f"(sv[83]), "f"(sv[84]), "f"(sv[85]), "f"(sv[86]), "f"(sv[87]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_2[3]) : "f"(sv[88]), "f"(sv[89]), "f"(sv[90]), "f"(sv[91]), "f"(sv[92]), "f"(sv[93]), "f"(sv[94]), "f"(sv[95]));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_p_addr + p_slot * 8192 + (unsigned int)((warp % 4 * 32 + lane) * 64 + 32 ^ ((warp % 4 * 32 + lane) * 64 + 32 >> 7 & 3) << 4))), "r"(_fp4_2[0]), "r"(_fp4_2[1]), "r"(_fp4_2[2]), "r"(_fp4_2[3]) : "memory");
#if __CUDA_ARCH__ == 1030
                        const float2 _fma_b2_66 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_67 = {2.5849626f - block_max_scaled, 2.5849626f - block_max_scaled};
                        float2 _fma_pair_68 = fma_f32x2(make_float2((sv + 96)[0], (sv + 96)[1]), _fma_b2_66, _fma_c2_67);
                        (sv + 96)[0] = _fma_pair_68.x;
                        (sv + 96)[1] = _fma_pair_68.y;
                        float2 _fma_pair_69 = fma_f32x2(make_float2((sv + 96)[2], (sv + 96)[3]), _fma_b2_66, _fma_c2_67);
                        (sv + 96)[2] = _fma_pair_69.x;
                        (sv + 96)[3] = _fma_pair_69.y;
                        float2 _fma_pair_70 = fma_f32x2(make_float2((sv + 96)[4], (sv + 96)[5]), _fma_b2_66, _fma_c2_67);
                        (sv + 96)[4] = _fma_pair_70.x;
                        (sv + 96)[5] = _fma_pair_70.y;
                        float2 _fma_pair_71 = fma_f32x2(make_float2((sv + 96)[6], (sv + 96)[7]), _fma_b2_66, _fma_c2_67);
                        (sv + 96)[6] = _fma_pair_71.x;
                        (sv + 96)[7] = _fma_pair_71.y;
                        float2 _fma_pair_72 = fma_f32x2(make_float2((sv + 96)[8], (sv + 96)[9]), _fma_b2_66, _fma_c2_67);
                        (sv + 96)[8] = _fma_pair_72.x;
                        (sv + 96)[9] = _fma_pair_72.y;
                        float2 _fma_pair_73 = fma_f32x2(make_float2((sv + 96)[10], (sv + 96)[11]), _fma_b2_66, _fma_c2_67);
                        (sv + 96)[10] = _fma_pair_73.x;
                        (sv + 96)[11] = _fma_pair_73.y;
                        float2 _fma_pair_74 = fma_f32x2(make_float2((sv + 96)[12], (sv + 96)[13]), _fma_b2_66, _fma_c2_67);
                        (sv + 96)[12] = _fma_pair_74.x;
                        (sv + 96)[13] = _fma_pair_74.y;
                        float2 _fma_pair_75 = fma_f32x2(make_float2((sv + 96)[14], (sv + 96)[15]), _fma_b2_66, _fma_c2_67);
                        (sv + 96)[14] = _fma_pair_75.x;
                        (sv + 96)[15] = _fma_pair_75.y;
                        float2 _fma_pair_76 = fma_f32x2(make_float2((sv + 96)[16], (sv + 96)[17]), _fma_b2_66, _fma_c2_67);
                        (sv + 96)[16] = _fma_pair_76.x;
                        (sv + 96)[17] = _fma_pair_76.y;
                        float2 _fma_pair_77 = fma_f32x2(make_float2((sv + 96)[18], (sv + 96)[19]), _fma_b2_66, _fma_c2_67);
                        (sv + 96)[18] = _fma_pair_77.x;
                        (sv + 96)[19] = _fma_pair_77.y;
                        float2 _fma_pair_78 = fma_f32x2(make_float2((sv + 96)[20], (sv + 96)[21]), _fma_b2_66, _fma_c2_67);
                        (sv + 96)[20] = _fma_pair_78.x;
                        (sv + 96)[21] = _fma_pair_78.y;
                        float2 _fma_pair_79 = fma_f32x2(make_float2((sv + 96)[22], (sv + 96)[23]), _fma_b2_66, _fma_c2_67);
                        (sv + 96)[22] = _fma_pair_79.x;
                        (sv + 96)[23] = _fma_pair_79.y;
                        float2 _fma_pair_80 = fma_f32x2(make_float2((sv + 96)[24], (sv + 96)[25]), _fma_b2_66, _fma_c2_67);
                        (sv + 96)[24] = _fma_pair_80.x;
                        (sv + 96)[25] = _fma_pair_80.y;
                        float2 _fma_pair_81 = fma_f32x2(make_float2((sv + 96)[26], (sv + 96)[27]), _fma_b2_66, _fma_c2_67);
                        (sv + 96)[26] = _fma_pair_81.x;
                        (sv + 96)[27] = _fma_pair_81.y;
                        float2 _fma_pair_82 = fma_f32x2(make_float2((sv + 96)[28], (sv + 96)[29]), _fma_b2_66, _fma_c2_67);
                        (sv + 96)[28] = _fma_pair_82.x;
                        (sv + 96)[29] = _fma_pair_82.y;
                        float2 _fma_pair_83 = fma_f32x2(make_float2((sv + 96)[30], (sv + 96)[31]), _fma_b2_66, _fma_c2_67);
                        (sv + 96)[30] = _fma_pair_83.x;
                        (sv + 96)[31] = _fma_pair_83.y;
#else
                        const float2 _fma_b2_80 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_81 = {2.5849626f - block_max_scaled, 2.5849626f - block_max_scaled};
                        float2 _fma_pair_82 = fma_f32x2(make_float2((sv + 96)[0], (sv + 96)[1]), _fma_b2_80, _fma_c2_81);
                        (sv + 96)[0] = _fma_pair_82.x;
                        (sv + 96)[1] = _fma_pair_82.y;
                        float2 _fma_pair_83 = fma_f32x2(make_float2((sv + 96)[2], (sv + 96)[3]), _fma_b2_80, _fma_c2_81);
                        (sv + 96)[2] = _fma_pair_83.x;
                        (sv + 96)[3] = _fma_pair_83.y;
                        float2 _fma_pair_84 = fma_f32x2(make_float2((sv + 96)[4], (sv + 96)[5]), _fma_b2_80, _fma_c2_81);
                        (sv + 96)[4] = _fma_pair_84.x;
                        (sv + 96)[5] = _fma_pair_84.y;
                        float2 _fma_pair_85 = fma_f32x2(make_float2((sv + 96)[6], (sv + 96)[7]), _fma_b2_80, _fma_c2_81);
                        (sv + 96)[6] = _fma_pair_85.x;
                        (sv + 96)[7] = _fma_pair_85.y;
                        float2 _fma_pair_86 = fma_f32x2(make_float2((sv + 96)[8], (sv + 96)[9]), _fma_b2_80, _fma_c2_81);
                        (sv + 96)[8] = _fma_pair_86.x;
                        (sv + 96)[9] = _fma_pair_86.y;
                        float2 _fma_pair_87 = fma_f32x2(make_float2((sv + 96)[10], (sv + 96)[11]), _fma_b2_80, _fma_c2_81);
                        (sv + 96)[10] = _fma_pair_87.x;
                        (sv + 96)[11] = _fma_pair_87.y;
                        float2 _fma_pair_88 = fma_f32x2(make_float2((sv + 96)[12], (sv + 96)[13]), _fma_b2_80, _fma_c2_81);
                        (sv + 96)[12] = _fma_pair_88.x;
                        (sv + 96)[13] = _fma_pair_88.y;
                        float2 _fma_pair_89 = fma_f32x2(make_float2((sv + 96)[14], (sv + 96)[15]), _fma_b2_80, _fma_c2_81);
                        (sv + 96)[14] = _fma_pair_89.x;
                        (sv + 96)[15] = _fma_pair_89.y;
                        float2 _fma_pair_90 = fma_f32x2(make_float2((sv + 96)[16], (sv + 96)[17]), _fma_b2_80, _fma_c2_81);
                        (sv + 96)[16] = _fma_pair_90.x;
                        (sv + 96)[17] = _fma_pair_90.y;
                        float2 _fma_pair_91 = fma_f32x2(make_float2((sv + 96)[18], (sv + 96)[19]), _fma_b2_80, _fma_c2_81);
                        (sv + 96)[18] = _fma_pair_91.x;
                        (sv + 96)[19] = _fma_pair_91.y;
                        float2 _fma_pair_92 = fma_f32x2(make_float2((sv + 96)[20], (sv + 96)[21]), _fma_b2_80, _fma_c2_81);
                        (sv + 96)[20] = _fma_pair_92.x;
                        (sv + 96)[21] = _fma_pair_92.y;
                        float2 _fma_pair_93 = fma_f32x2(make_float2((sv + 96)[22], (sv + 96)[23]), _fma_b2_80, _fma_c2_81);
                        (sv + 96)[22] = _fma_pair_93.x;
                        (sv + 96)[23] = _fma_pair_93.y;
                        float2 _fma_pair_94 = fma_f32x2(make_float2((sv + 96)[24], (sv + 96)[25]), _fma_b2_80, _fma_c2_81);
                        (sv + 96)[24] = _fma_pair_94.x;
                        (sv + 96)[25] = _fma_pair_94.y;
                        float2 _fma_pair_95 = fma_f32x2(make_float2((sv + 96)[26], (sv + 96)[27]), _fma_b2_80, _fma_c2_81);
                        (sv + 96)[26] = _fma_pair_95.x;
                        (sv + 96)[27] = _fma_pair_95.y;
                        float2 _fma_pair_96 = fma_f32x2(make_float2((sv + 96)[28], (sv + 96)[29]), _fma_b2_80, _fma_c2_81);
                        (sv + 96)[28] = _fma_pair_96.x;
                        (sv + 96)[29] = _fma_pair_96.y;
                        float2 _fma_pair_97 = fma_f32x2(make_float2((sv + 96)[30], (sv + 96)[31]), _fma_b2_80, _fma_c2_81);
                        (sv + 96)[30] = _fma_pair_97.x;
                        (sv + 96)[31] = _fma_pair_97.y;
#endif
                        {
                            #pragma unroll
#if __CUDA_ARCH__ == 1030
                            for (int _le = 0; _le < 32; _le++) {
#else
                            for (int _le = 0; _le < 24; _le++) {
#endif
                                sv[_le + 96] = approx_exp2(sv[_le + 96]);
                            }
#if __CUDA_ARCH__ == 1030
#else
                            float2 _exp2_pair_98 = ex2_emulation_f32x2_value(make_float2(sv[120], sv[121]));
                            sv[120] = _exp2_pair_98.x;
                            sv[121] = _exp2_pair_98.y;
                            float2 _exp2_pair_99 = ex2_emulation_f32x2_value(make_float2(sv[122], sv[123]));
                            sv[122] = _exp2_pair_99.x;
                            sv[123] = _exp2_pair_99.y;
                            float2 _exp2_pair_100 = ex2_emulation_f32x2_value(make_float2(sv[124], sv[125]));
                            sv[124] = _exp2_pair_100.x;
                            sv[125] = _exp2_pair_100.y;
                            float2 _exp2_pair_101 = ex2_emulation_f32x2_value(make_float2(sv[126], sv[127]));
                            sv[126] = _exp2_pair_101.x;
                            sv[127] = _exp2_pair_101.y;
#endif
                        }
                        float2 partial_8 = make_float2(sv[96], sv[97]);
                        #pragma unroll
                        for (int pair_6 = 2; pair_6 < 16; pair_6 += 2) {
                            partial_8 = add_f32x2(partial_8, make_float2(sv[96 + pair_6], sv[96 + (pair_6 + 1)]));
                        }
                        float2 scale2_9 = make_float2(p_scale, p_scale);
                        block_sum2 = fma_f32x2_rn_ftz(partial_8, scale2_9, block_sum2);
                        float2 partial_10 = make_float2(sv[112], sv[113]);
                        #pragma unroll
                        for (int pair_7 = 2; pair_7 < 16; pair_7 += 2) {
                            partial_10 = add_f32x2(partial_10, make_float2(sv[112 + pair_7], sv[112 + (pair_7 + 1)]));
                        }
                        block_sum2 = fma_f32x2_rn_ftz(partial_10, scale2_9, block_sum2);
                        unsigned int _fp4_3[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_3[0]) : "f"(sv[96]), "f"(sv[97]), "f"(sv[98]), "f"(sv[99]), "f"(sv[100]), "f"(sv[101]), "f"(sv[102]), "f"(sv[103]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_3[1]) : "f"(sv[104]), "f"(sv[105]), "f"(sv[106]), "f"(sv[107]), "f"(sv[108]), "f"(sv[109]), "f"(sv[110]), "f"(sv[111]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_3[2]) : "f"(sv[112]), "f"(sv[113]), "f"(sv[114]), "f"(sv[115]), "f"(sv[116]), "f"(sv[117]), "f"(sv[118]), "f"(sv[119]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_3[3]) : "f"(sv[120]), "f"(sv[121]), "f"(sv[122]), "f"(sv[123]), "f"(sv[124]), "f"(sv[125]), "f"(sv[126]), "f"(sv[127]));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_p_addr + p_slot * 8192 + (unsigned int)((warp % 4 * 32 + lane) * 64 + 48 ^ ((warp % 4 * 32 + lane) * 64 + 48 >> 7 & 3) << 4))), "r"(_fp4_3[0]), "r"(_fp4_3[1]), "r"(_fp4_3[2]), "r"(_fp4_3[3]) : "memory");
                        int pv_sfa_hi_col = (unsigned int)412 + stage * 8 + (unsigned int)(warp % 4) + (p_slot & 1) * 24;
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(taddr + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)pv_sfa_hi_col), "r"(sf_packed[0]));
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((p_full_addr + (p_slot) * 8) & 0xFEFFFFFF) : "memory");
                        p_slot += 1;
                        if (p_slot == (stage * 2 + 2)) { p_slot = stage * 2; }
                        block_sum = block_sum2.x + block_sum2.y;
                        row_sum = __fmaf_rn(row_sum, acc_scale, block_sum);
                        mbarrier_wait(corr_done_addr + (stage) * 8, _phase_corr_done);
                        _phase_corr_done ^= 1;
                    }
#if __CUDA_ARCH__ == 1030
#else
                    #pragma unroll 1
                    for (unsigned int n_iter_1 = 1; n_iter_1 < n_count; n_iter_1++) {
                        mbarrier_wait_hint(s_full_addr + (stage) * 8, _phase_s_full, 10000000);
                        _phase_s_full ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int s_addr_1 = taddr + (unsigned int)0 + (unsigned int)(warp % 4 * 32 << 16);
                        float sv_1[128];
                        float tile_max_1 = -CUDART_INF_F;
                        {
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(sv_1[0]), "=f"(sv_1[1]), "=f"(sv_1[2]), "=f"(sv_1[3]), "=f"(sv_1[4]), "=f"(sv_1[5]), "=f"(sv_1[6]), "=f"(sv_1[7]), "=f"(sv_1[8]), "=f"(sv_1[9]), "=f"(sv_1[10]), "=f"(sv_1[11]), "=f"(sv_1[12]), "=f"(sv_1[13]), "=f"(sv_1[14]), "=f"(sv_1[15]), "=f"(sv_1[16]), "=f"(sv_1[17]), "=f"(sv_1[18]), "=f"(sv_1[19]), "=f"(sv_1[20]), "=f"(sv_1[21]), "=f"(sv_1[22]), "=f"(sv_1[23]), "=f"(sv_1[24]), "=f"(sv_1[25]), "=f"(sv_1[26]), "=f"(sv_1[27]), "=f"(sv_1[28]), "=f"(sv_1[29]), "=f"(sv_1[30]), "=f"(sv_1[31])
                                : "r"(s_addr_1));
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(sv_1[32]), "=f"(sv_1[33]), "=f"(sv_1[34]), "=f"(sv_1[35]), "=f"(sv_1[36]), "=f"(sv_1[37]), "=f"(sv_1[38]), "=f"(sv_1[39]), "=f"(sv_1[40]), "=f"(sv_1[41]), "=f"(sv_1[42]), "=f"(sv_1[43]), "=f"(sv_1[44]), "=f"(sv_1[45]), "=f"(sv_1[46]), "=f"(sv_1[47]), "=f"(sv_1[48]), "=f"(sv_1[49]), "=f"(sv_1[50]), "=f"(sv_1[51]), "=f"(sv_1[52]), "=f"(sv_1[53]), "=f"(sv_1[54]), "=f"(sv_1[55]), "=f"(sv_1[56]), "=f"(sv_1[57]), "=f"(sv_1[58]), "=f"(sv_1[59]), "=f"(sv_1[60]), "=f"(sv_1[61]), "=f"(sv_1[62]), "=f"(sv_1[63])
                                : "r"(s_addr_1 + 32));
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(sv_1[64]), "=f"(sv_1[65]), "=f"(sv_1[66]), "=f"(sv_1[67]), "=f"(sv_1[68]), "=f"(sv_1[69]), "=f"(sv_1[70]), "=f"(sv_1[71]), "=f"(sv_1[72]), "=f"(sv_1[73]), "=f"(sv_1[74]), "=f"(sv_1[75]), "=f"(sv_1[76]), "=f"(sv_1[77]), "=f"(sv_1[78]), "=f"(sv_1[79]), "=f"(sv_1[80]), "=f"(sv_1[81]), "=f"(sv_1[82]), "=f"(sv_1[83]), "=f"(sv_1[84]), "=f"(sv_1[85]), "=f"(sv_1[86]), "=f"(sv_1[87]), "=f"(sv_1[88]), "=f"(sv_1[89]), "=f"(sv_1[90]), "=f"(sv_1[91]), "=f"(sv_1[92]), "=f"(sv_1[93]), "=f"(sv_1[94]), "=f"(sv_1[95])
                                : "r"(s_addr_1 + 64));
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(sv_1[96]), "=f"(sv_1[97]), "=f"(sv_1[98]), "=f"(sv_1[99]), "=f"(sv_1[100]), "=f"(sv_1[101]), "=f"(sv_1[102]), "=f"(sv_1[103]), "=f"(sv_1[104]), "=f"(sv_1[105]), "=f"(sv_1[106]), "=f"(sv_1[107]), "=f"(sv_1[108]), "=f"(sv_1[109]), "=f"(sv_1[110]), "=f"(sv_1[111]), "=f"(sv_1[112]), "=f"(sv_1[113]), "=f"(sv_1[114]), "=f"(sv_1[115]), "=f"(sv_1[116]), "=f"(sv_1[117]), "=f"(sv_1[118]), "=f"(sv_1[119]), "=f"(sv_1[120]), "=f"(sv_1[121]), "=f"(sv_1[122]), "=f"(sv_1[123]), "=f"(sv_1[124]), "=f"(sv_1[125]), "=f"(sv_1[126]), "=f"(sv_1[127])
                                : "r"(s_addr_1 + 96));
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            float2 _reg_reduce_max2_103 = {-CAKE_INF, -CAKE_INF};
                            row_max_x32_accum(&sv_1[0], _reg_reduce_max2_103);
                            row_max_x32_accum(&sv_1[32], _reg_reduce_max2_103);
                            row_max_x32_accum(&sv_1[64], _reg_reduce_max2_103);
                            row_max_x32_accum(&sv_1[96], _reg_reduce_max2_103);
                            float _reg_array_reduce_102 = row_max_reduce(_reg_reduce_max2_103);
                            tile_max_1 = _reg_array_reduce_102;
                        }
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((s_empty_addr + (stage) * 8) & 0xFEFFFFFF) : "memory");
                        float new_max_1;
                        asm("max.f32 %0, %1, %2;" : "=f"(new_max_1) : "f"(tile_max_1), "f"(row_max));
                        float safe_max_1 = new_max_1;
                        float new_max_scaled_1 = safe_max_1 * softmax_scale_log2;
                        float acc_scale_log2_1 = __fmaf_rn(row_max, softmax_scale_log2, -new_max_scaled_1);
                        float acc_scale_1;
                        if (acc_scale_log2_1 >= -8.0f) {
                            acc_scale_1 = 1.0f;
                            new_max_scaled_1 = row_max_scaled;
                        } else {
                            float _exp2_104 = approx_exp2(acc_scale_log2_1);
                            acc_scale_1 = ((row_max > -CUDART_INF_F) ? _exp2_104 : 1.0f);
                            row_max = new_max_1;
                            row_max_scaled = new_max_scaled_1;
                        }
                        row_state[warp % 4 * 32 + lane + scale_off] = acc_scale_1;
                        if (warp % 4 < 2) {
                            mbarrier_arrive(corr_sig_lo_addr + (stage) * 8);
                        } else {
                            mbarrier_arrive(corr_sig_hi_addr + (stage) * 8);
                        }
                        float block_sum_1 = 0.0f;
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        float2 block_sum2_1 = make_float2(0.0f, 0.0f);
                        float block_max_1 = tile_max_1;
                        float block_max_scaled_1 = ((block_max_1 > -CUDART_INF_F) ? block_max_1 * softmax_scale_log2 : 0.0f);
                        float _exp2_105 = approx_exp2(block_max_scaled_1 - new_max_scaled_1 - 2.5849626f);
                        float p_scale_1 = ((block_max_1 > -CUDART_INF_F) ? _exp2_105 : 0.0f);
                        uint16_t _e4m3x2_f32_1;
                        asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_1) : "f"(p_scale_1), "f"(p_scale_1));
                        unsigned int sf_packed_1[1];
                        sf_packed_1[0] = (unsigned int)_e4m3x2_f32_1 * 65537;
                        const float2 _fma_b2_106 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_107 = {2.5849626f - block_max_scaled_1, 2.5849626f - block_max_scaled_1};
                        float2 _fma_pair_108 = fma_f32x2(make_float2((sv_1 + 0)[0], (sv_1 + 0)[1]), _fma_b2_106, _fma_c2_107);
                        (sv_1 + 0)[0] = _fma_pair_108.x;
                        (sv_1 + 0)[1] = _fma_pair_108.y;
                        float2 _fma_pair_109 = fma_f32x2(make_float2((sv_1 + 0)[2], (sv_1 + 0)[3]), _fma_b2_106, _fma_c2_107);
                        (sv_1 + 0)[2] = _fma_pair_109.x;
                        (sv_1 + 0)[3] = _fma_pair_109.y;
                        float2 _fma_pair_110 = fma_f32x2(make_float2((sv_1 + 0)[4], (sv_1 + 0)[5]), _fma_b2_106, _fma_c2_107);
                        (sv_1 + 0)[4] = _fma_pair_110.x;
                        (sv_1 + 0)[5] = _fma_pair_110.y;
                        float2 _fma_pair_111 = fma_f32x2(make_float2((sv_1 + 0)[6], (sv_1 + 0)[7]), _fma_b2_106, _fma_c2_107);
                        (sv_1 + 0)[6] = _fma_pair_111.x;
                        (sv_1 + 0)[7] = _fma_pair_111.y;
                        float2 _fma_pair_112 = fma_f32x2(make_float2((sv_1 + 0)[8], (sv_1 + 0)[9]), _fma_b2_106, _fma_c2_107);
                        (sv_1 + 0)[8] = _fma_pair_112.x;
                        (sv_1 + 0)[9] = _fma_pair_112.y;
                        float2 _fma_pair_113 = fma_f32x2(make_float2((sv_1 + 0)[10], (sv_1 + 0)[11]), _fma_b2_106, _fma_c2_107);
                        (sv_1 + 0)[10] = _fma_pair_113.x;
                        (sv_1 + 0)[11] = _fma_pair_113.y;
                        float2 _fma_pair_114 = fma_f32x2(make_float2((sv_1 + 0)[12], (sv_1 + 0)[13]), _fma_b2_106, _fma_c2_107);
                        (sv_1 + 0)[12] = _fma_pair_114.x;
                        (sv_1 + 0)[13] = _fma_pair_114.y;
                        float2 _fma_pair_115 = fma_f32x2(make_float2((sv_1 + 0)[14], (sv_1 + 0)[15]), _fma_b2_106, _fma_c2_107);
                        (sv_1 + 0)[14] = _fma_pair_115.x;
                        (sv_1 + 0)[15] = _fma_pair_115.y;
                        float2 _fma_pair_116 = fma_f32x2(make_float2((sv_1 + 0)[16], (sv_1 + 0)[17]), _fma_b2_106, _fma_c2_107);
                        (sv_1 + 0)[16] = _fma_pair_116.x;
                        (sv_1 + 0)[17] = _fma_pair_116.y;
                        float2 _fma_pair_117 = fma_f32x2(make_float2((sv_1 + 0)[18], (sv_1 + 0)[19]), _fma_b2_106, _fma_c2_107);
                        (sv_1 + 0)[18] = _fma_pair_117.x;
                        (sv_1 + 0)[19] = _fma_pair_117.y;
                        float2 _fma_pair_118 = fma_f32x2(make_float2((sv_1 + 0)[20], (sv_1 + 0)[21]), _fma_b2_106, _fma_c2_107);
                        (sv_1 + 0)[20] = _fma_pair_118.x;
                        (sv_1 + 0)[21] = _fma_pair_118.y;
                        float2 _fma_pair_119 = fma_f32x2(make_float2((sv_1 + 0)[22], (sv_1 + 0)[23]), _fma_b2_106, _fma_c2_107);
                        (sv_1 + 0)[22] = _fma_pair_119.x;
                        (sv_1 + 0)[23] = _fma_pair_119.y;
                        float2 _fma_pair_120 = fma_f32x2(make_float2((sv_1 + 0)[24], (sv_1 + 0)[25]), _fma_b2_106, _fma_c2_107);
                        (sv_1 + 0)[24] = _fma_pair_120.x;
                        (sv_1 + 0)[25] = _fma_pair_120.y;
                        float2 _fma_pair_121 = fma_f32x2(make_float2((sv_1 + 0)[26], (sv_1 + 0)[27]), _fma_b2_106, _fma_c2_107);
                        (sv_1 + 0)[26] = _fma_pair_121.x;
                        (sv_1 + 0)[27] = _fma_pair_121.y;
                        float2 _fma_pair_122 = fma_f32x2(make_float2((sv_1 + 0)[28], (sv_1 + 0)[29]), _fma_b2_106, _fma_c2_107);
                        (sv_1 + 0)[28] = _fma_pair_122.x;
                        (sv_1 + 0)[29] = _fma_pair_122.y;
                        float2 _fma_pair_123 = fma_f32x2(make_float2((sv_1 + 0)[30], (sv_1 + 0)[31]), _fma_b2_106, _fma_c2_107);
                        (sv_1 + 0)[30] = _fma_pair_123.x;
                        (sv_1 + 0)[31] = _fma_pair_123.y;
                        {
                            #pragma unroll
                            for (int _le = 0; _le < 24; _le++) {
                                sv_1[_le] = approx_exp2(sv_1[_le]);
                            }
                            float2 _exp2_pair_124 = ex2_emulation_f32x2_value(make_float2(sv_1[24], sv_1[25]));
                            sv_1[24] = _exp2_pair_124.x;
                            sv_1[25] = _exp2_pair_124.y;
                            float2 _exp2_pair_125 = ex2_emulation_f32x2_value(make_float2(sv_1[26], sv_1[27]));
                            sv_1[26] = _exp2_pair_125.x;
                            sv_1[27] = _exp2_pair_125.y;
                            float2 _exp2_pair_126 = ex2_emulation_f32x2_value(make_float2(sv_1[28], sv_1[29]));
                            sv_1[28] = _exp2_pair_126.x;
                            sv_1[29] = _exp2_pair_126.y;
                            float2 _exp2_pair_127 = ex2_emulation_f32x2_value(make_float2(sv_1[30], sv_1[31]));
                            sv_1[30] = _exp2_pair_127.x;
                            sv_1[31] = _exp2_pair_127.y;
                        }
                        float2 partial_2 = make_float2(sv_1[0], sv_1[1]);
                        #pragma unroll
                        for (int pair_8 = 2; pair_8 < 16; pair_8 += 2) {
                            partial_2 = add_f32x2(partial_2, make_float2(sv_1[pair_8], sv_1[pair_8 + 1]));
                        }
                        float2 frag_sum2_1 = partial_2;
                        float2 scale2_1 = make_float2(p_scale_1, p_scale_1);
                        block_sum2_1 = fma_f32x2_rn_ftz(frag_sum2_1, scale2_1, block_sum2_1);
                        float2 partial_0_1 = make_float2(sv_1[16], sv_1[17]);
                        #pragma unroll
                        for (int pair_9 = 2; pair_9 < 16; pair_9 += 2) {
                            partial_0_1 = add_f32x2(partial_0_1, make_float2(sv_1[16 + pair_9], sv_1[16 + (pair_9 + 1)]));
                        }
                        frag_sum2_1 = partial_0_1;
                        block_sum2_1 = fma_f32x2_rn_ftz(frag_sum2_1, scale2_1, block_sum2_1);
                        unsigned int _fp4_4[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_4[0]) : "f"(sv_1[0]), "f"(sv_1[1]), "f"(sv_1[2]), "f"(sv_1[3]), "f"(sv_1[4]), "f"(sv_1[5]), "f"(sv_1[6]), "f"(sv_1[7]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_4[1]) : "f"(sv_1[8]), "f"(sv_1[9]), "f"(sv_1[10]), "f"(sv_1[11]), "f"(sv_1[12]), "f"(sv_1[13]), "f"(sv_1[14]), "f"(sv_1[15]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_4[2]) : "f"(sv_1[16]), "f"(sv_1[17]), "f"(sv_1[18]), "f"(sv_1[19]), "f"(sv_1[20]), "f"(sv_1[21]), "f"(sv_1[22]), "f"(sv_1[23]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_4[3]) : "f"(sv_1[24]), "f"(sv_1[25]), "f"(sv_1[26]), "f"(sv_1[27]), "f"(sv_1[28]), "f"(sv_1[29]), "f"(sv_1[30]), "f"(sv_1[31]));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_p_addr + p_slot * 8192 + (unsigned int)((warp % 4 * 32 + lane) * 64 ^ ((warp % 4 * 32 + lane) * 64 >> 7 & 3) << 4))), "r"(_fp4_4[0]), "r"(_fp4_4[1]), "r"(_fp4_4[2]), "r"(_fp4_4[3]) : "memory");
                        const float2 _fma_b2_128 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_129 = {2.5849626f - block_max_scaled_1, 2.5849626f - block_max_scaled_1};
                        float2 _fma_pair_130 = fma_f32x2(make_float2((sv_1 + 32)[0], (sv_1 + 32)[1]), _fma_b2_128, _fma_c2_129);
                        (sv_1 + 32)[0] = _fma_pair_130.x;
                        (sv_1 + 32)[1] = _fma_pair_130.y;
                        float2 _fma_pair_131 = fma_f32x2(make_float2((sv_1 + 32)[2], (sv_1 + 32)[3]), _fma_b2_128, _fma_c2_129);
                        (sv_1 + 32)[2] = _fma_pair_131.x;
                        (sv_1 + 32)[3] = _fma_pair_131.y;
                        float2 _fma_pair_132 = fma_f32x2(make_float2((sv_1 + 32)[4], (sv_1 + 32)[5]), _fma_b2_128, _fma_c2_129);
                        (sv_1 + 32)[4] = _fma_pair_132.x;
                        (sv_1 + 32)[5] = _fma_pair_132.y;
                        float2 _fma_pair_133 = fma_f32x2(make_float2((sv_1 + 32)[6], (sv_1 + 32)[7]), _fma_b2_128, _fma_c2_129);
                        (sv_1 + 32)[6] = _fma_pair_133.x;
                        (sv_1 + 32)[7] = _fma_pair_133.y;
                        float2 _fma_pair_134 = fma_f32x2(make_float2((sv_1 + 32)[8], (sv_1 + 32)[9]), _fma_b2_128, _fma_c2_129);
                        (sv_1 + 32)[8] = _fma_pair_134.x;
                        (sv_1 + 32)[9] = _fma_pair_134.y;
                        float2 _fma_pair_135 = fma_f32x2(make_float2((sv_1 + 32)[10], (sv_1 + 32)[11]), _fma_b2_128, _fma_c2_129);
                        (sv_1 + 32)[10] = _fma_pair_135.x;
                        (sv_1 + 32)[11] = _fma_pair_135.y;
                        float2 _fma_pair_136 = fma_f32x2(make_float2((sv_1 + 32)[12], (sv_1 + 32)[13]), _fma_b2_128, _fma_c2_129);
                        (sv_1 + 32)[12] = _fma_pair_136.x;
                        (sv_1 + 32)[13] = _fma_pair_136.y;
                        float2 _fma_pair_137 = fma_f32x2(make_float2((sv_1 + 32)[14], (sv_1 + 32)[15]), _fma_b2_128, _fma_c2_129);
                        (sv_1 + 32)[14] = _fma_pair_137.x;
                        (sv_1 + 32)[15] = _fma_pair_137.y;
                        float2 _fma_pair_138 = fma_f32x2(make_float2((sv_1 + 32)[16], (sv_1 + 32)[17]), _fma_b2_128, _fma_c2_129);
                        (sv_1 + 32)[16] = _fma_pair_138.x;
                        (sv_1 + 32)[17] = _fma_pair_138.y;
                        float2 _fma_pair_139 = fma_f32x2(make_float2((sv_1 + 32)[18], (sv_1 + 32)[19]), _fma_b2_128, _fma_c2_129);
                        (sv_1 + 32)[18] = _fma_pair_139.x;
                        (sv_1 + 32)[19] = _fma_pair_139.y;
                        float2 _fma_pair_140 = fma_f32x2(make_float2((sv_1 + 32)[20], (sv_1 + 32)[21]), _fma_b2_128, _fma_c2_129);
                        (sv_1 + 32)[20] = _fma_pair_140.x;
                        (sv_1 + 32)[21] = _fma_pair_140.y;
                        float2 _fma_pair_141 = fma_f32x2(make_float2((sv_1 + 32)[22], (sv_1 + 32)[23]), _fma_b2_128, _fma_c2_129);
                        (sv_1 + 32)[22] = _fma_pair_141.x;
                        (sv_1 + 32)[23] = _fma_pair_141.y;
                        float2 _fma_pair_142 = fma_f32x2(make_float2((sv_1 + 32)[24], (sv_1 + 32)[25]), _fma_b2_128, _fma_c2_129);
                        (sv_1 + 32)[24] = _fma_pair_142.x;
                        (sv_1 + 32)[25] = _fma_pair_142.y;
                        float2 _fma_pair_143 = fma_f32x2(make_float2((sv_1 + 32)[26], (sv_1 + 32)[27]), _fma_b2_128, _fma_c2_129);
                        (sv_1 + 32)[26] = _fma_pair_143.x;
                        (sv_1 + 32)[27] = _fma_pair_143.y;
                        float2 _fma_pair_144 = fma_f32x2(make_float2((sv_1 + 32)[28], (sv_1 + 32)[29]), _fma_b2_128, _fma_c2_129);
                        (sv_1 + 32)[28] = _fma_pair_144.x;
                        (sv_1 + 32)[29] = _fma_pair_144.y;
                        float2 _fma_pair_145 = fma_f32x2(make_float2((sv_1 + 32)[30], (sv_1 + 32)[31]), _fma_b2_128, _fma_c2_129);
                        (sv_1 + 32)[30] = _fma_pair_145.x;
                        (sv_1 + 32)[31] = _fma_pair_145.y;
                        {
                            #pragma unroll
                            for (int _le = 0; _le < 24; _le++) {
                                sv_1[_le + 32] = approx_exp2(sv_1[_le + 32]);
                            }
                            float2 _exp2_pair_146 = ex2_emulation_f32x2_value(make_float2(sv_1[56], sv_1[57]));
                            sv_1[56] = _exp2_pair_146.x;
                            sv_1[57] = _exp2_pair_146.y;
                            float2 _exp2_pair_147 = ex2_emulation_f32x2_value(make_float2(sv_1[58], sv_1[59]));
                            sv_1[58] = _exp2_pair_147.x;
                            sv_1[59] = _exp2_pair_147.y;
                            float2 _exp2_pair_148 = ex2_emulation_f32x2_value(make_float2(sv_1[60], sv_1[61]));
                            sv_1[60] = _exp2_pair_148.x;
                            sv_1[61] = _exp2_pair_148.y;
                            float2 _exp2_pair_149 = ex2_emulation_f32x2_value(make_float2(sv_1[62], sv_1[63]));
                            sv_1[62] = _exp2_pair_149.x;
                            sv_1[63] = _exp2_pair_149.y;
                        }
                        float2 partial_1_1 = make_float2(sv_1[32], sv_1[33]);
                        #pragma unroll
                        for (int pair_10 = 2; pair_10 < 16; pair_10 += 2) {
                            partial_1_1 = add_f32x2(partial_1_1, make_float2(sv_1[32 + pair_10], sv_1[32 + (pair_10 + 1)]));
                        }
                        float2 frag_sum2_2_1 = partial_1_1;
                        float2 scale2_3_1 = make_float2(p_scale_1, p_scale_1);
                        block_sum2_1 = fma_f32x2_rn_ftz(frag_sum2_2_1, scale2_3_1, block_sum2_1);
                        float2 partial_4_1 = make_float2(sv_1[48], sv_1[49]);
                        #pragma unroll
                        for (int pair_11 = 2; pair_11 < 16; pair_11 += 2) {
                            partial_4_1 = add_f32x2(partial_4_1, make_float2(sv_1[48 + pair_11], sv_1[48 + (pair_11 + 1)]));
                        }
                        frag_sum2_2_1 = partial_4_1;
                        block_sum2_1 = fma_f32x2_rn_ftz(frag_sum2_2_1, scale2_3_1, block_sum2_1);
                        unsigned int _fp4_5[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_5[0]) : "f"(sv_1[32]), "f"(sv_1[33]), "f"(sv_1[34]), "f"(sv_1[35]), "f"(sv_1[36]), "f"(sv_1[37]), "f"(sv_1[38]), "f"(sv_1[39]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_5[1]) : "f"(sv_1[40]), "f"(sv_1[41]), "f"(sv_1[42]), "f"(sv_1[43]), "f"(sv_1[44]), "f"(sv_1[45]), "f"(sv_1[46]), "f"(sv_1[47]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_5[2]) : "f"(sv_1[48]), "f"(sv_1[49]), "f"(sv_1[50]), "f"(sv_1[51]), "f"(sv_1[52]), "f"(sv_1[53]), "f"(sv_1[54]), "f"(sv_1[55]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_5[3]) : "f"(sv_1[56]), "f"(sv_1[57]), "f"(sv_1[58]), "f"(sv_1[59]), "f"(sv_1[60]), "f"(sv_1[61]), "f"(sv_1[62]), "f"(sv_1[63]));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_p_addr + p_slot * 8192 + (unsigned int)((warp % 4 * 32 + lane) * 64 + 16 ^ ((warp % 4 * 32 + lane) * 64 + 16 >> 7 & 3) << 4))), "r"(_fp4_5[0]), "r"(_fp4_5[1]), "r"(_fp4_5[2]), "r"(_fp4_5[3]) : "memory");
                        int pv_sfa_lo_col_1 = (unsigned int)408 + stage * 8 + (unsigned int)(warp % 4) + (p_slot & 1) * 24;
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(taddr + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)pv_sfa_lo_col_1), "r"(sf_packed_1[0]));
                        const float2 _fma_b2_150 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_151 = {2.5849626f - block_max_scaled_1, 2.5849626f - block_max_scaled_1};
                        float2 _fma_pair_152 = fma_f32x2(make_float2((sv_1 + 64)[0], (sv_1 + 64)[1]), _fma_b2_150, _fma_c2_151);
                        (sv_1 + 64)[0] = _fma_pair_152.x;
                        (sv_1 + 64)[1] = _fma_pair_152.y;
                        float2 _fma_pair_153 = fma_f32x2(make_float2((sv_1 + 64)[2], (sv_1 + 64)[3]), _fma_b2_150, _fma_c2_151);
                        (sv_1 + 64)[2] = _fma_pair_153.x;
                        (sv_1 + 64)[3] = _fma_pair_153.y;
                        float2 _fma_pair_154 = fma_f32x2(make_float2((sv_1 + 64)[4], (sv_1 + 64)[5]), _fma_b2_150, _fma_c2_151);
                        (sv_1 + 64)[4] = _fma_pair_154.x;
                        (sv_1 + 64)[5] = _fma_pair_154.y;
                        float2 _fma_pair_155 = fma_f32x2(make_float2((sv_1 + 64)[6], (sv_1 + 64)[7]), _fma_b2_150, _fma_c2_151);
                        (sv_1 + 64)[6] = _fma_pair_155.x;
                        (sv_1 + 64)[7] = _fma_pair_155.y;
                        float2 _fma_pair_156 = fma_f32x2(make_float2((sv_1 + 64)[8], (sv_1 + 64)[9]), _fma_b2_150, _fma_c2_151);
                        (sv_1 + 64)[8] = _fma_pair_156.x;
                        (sv_1 + 64)[9] = _fma_pair_156.y;
                        float2 _fma_pair_157 = fma_f32x2(make_float2((sv_1 + 64)[10], (sv_1 + 64)[11]), _fma_b2_150, _fma_c2_151);
                        (sv_1 + 64)[10] = _fma_pair_157.x;
                        (sv_1 + 64)[11] = _fma_pair_157.y;
                        float2 _fma_pair_158 = fma_f32x2(make_float2((sv_1 + 64)[12], (sv_1 + 64)[13]), _fma_b2_150, _fma_c2_151);
                        (sv_1 + 64)[12] = _fma_pair_158.x;
                        (sv_1 + 64)[13] = _fma_pair_158.y;
                        float2 _fma_pair_159 = fma_f32x2(make_float2((sv_1 + 64)[14], (sv_1 + 64)[15]), _fma_b2_150, _fma_c2_151);
                        (sv_1 + 64)[14] = _fma_pair_159.x;
                        (sv_1 + 64)[15] = _fma_pair_159.y;
                        float2 _fma_pair_160 = fma_f32x2(make_float2((sv_1 + 64)[16], (sv_1 + 64)[17]), _fma_b2_150, _fma_c2_151);
                        (sv_1 + 64)[16] = _fma_pair_160.x;
                        (sv_1 + 64)[17] = _fma_pair_160.y;
                        float2 _fma_pair_161 = fma_f32x2(make_float2((sv_1 + 64)[18], (sv_1 + 64)[19]), _fma_b2_150, _fma_c2_151);
                        (sv_1 + 64)[18] = _fma_pair_161.x;
                        (sv_1 + 64)[19] = _fma_pair_161.y;
                        float2 _fma_pair_162 = fma_f32x2(make_float2((sv_1 + 64)[20], (sv_1 + 64)[21]), _fma_b2_150, _fma_c2_151);
                        (sv_1 + 64)[20] = _fma_pair_162.x;
                        (sv_1 + 64)[21] = _fma_pair_162.y;
                        float2 _fma_pair_163 = fma_f32x2(make_float2((sv_1 + 64)[22], (sv_1 + 64)[23]), _fma_b2_150, _fma_c2_151);
                        (sv_1 + 64)[22] = _fma_pair_163.x;
                        (sv_1 + 64)[23] = _fma_pair_163.y;
                        float2 _fma_pair_164 = fma_f32x2(make_float2((sv_1 + 64)[24], (sv_1 + 64)[25]), _fma_b2_150, _fma_c2_151);
                        (sv_1 + 64)[24] = _fma_pair_164.x;
                        (sv_1 + 64)[25] = _fma_pair_164.y;
                        float2 _fma_pair_165 = fma_f32x2(make_float2((sv_1 + 64)[26], (sv_1 + 64)[27]), _fma_b2_150, _fma_c2_151);
                        (sv_1 + 64)[26] = _fma_pair_165.x;
                        (sv_1 + 64)[27] = _fma_pair_165.y;
                        float2 _fma_pair_166 = fma_f32x2(make_float2((sv_1 + 64)[28], (sv_1 + 64)[29]), _fma_b2_150, _fma_c2_151);
                        (sv_1 + 64)[28] = _fma_pair_166.x;
                        (sv_1 + 64)[29] = _fma_pair_166.y;
                        float2 _fma_pair_167 = fma_f32x2(make_float2((sv_1 + 64)[30], (sv_1 + 64)[31]), _fma_b2_150, _fma_c2_151);
                        (sv_1 + 64)[30] = _fma_pair_167.x;
                        (sv_1 + 64)[31] = _fma_pair_167.y;
                        {
                            #pragma unroll
                            for (int _le = 0; _le < 24; _le++) {
                                sv_1[_le + 64] = approx_exp2(sv_1[_le + 64]);
                            }
                            float2 _exp2_pair_168 = ex2_emulation_f32x2_value(make_float2(sv_1[88], sv_1[89]));
                            sv_1[88] = _exp2_pair_168.x;
                            sv_1[89] = _exp2_pair_168.y;
                            float2 _exp2_pair_169 = ex2_emulation_f32x2_value(make_float2(sv_1[90], sv_1[91]));
                            sv_1[90] = _exp2_pair_169.x;
                            sv_1[91] = _exp2_pair_169.y;
                            float2 _exp2_pair_170 = ex2_emulation_f32x2_value(make_float2(sv_1[92], sv_1[93]));
                            sv_1[92] = _exp2_pair_170.x;
                            sv_1[93] = _exp2_pair_170.y;
                            float2 _exp2_pair_171 = ex2_emulation_f32x2_value(make_float2(sv_1[94], sv_1[95]));
                            sv_1[94] = _exp2_pair_171.x;
                            sv_1[95] = _exp2_pair_171.y;
                        }
                        float2 partial_5_1 = make_float2(sv_1[64], sv_1[65]);
                        #pragma unroll
                        for (int pair_12 = 2; pair_12 < 16; pair_12 += 2) {
                            partial_5_1 = add_f32x2(partial_5_1, make_float2(sv_1[64 + pair_12], sv_1[64 + (pair_12 + 1)]));
                        }
                        float2 scale2_6_1 = make_float2(p_scale_1, p_scale_1);
                        block_sum2_1 = fma_f32x2_rn_ftz(partial_5_1, scale2_6_1, block_sum2_1);
                        float2 partial_7_1 = make_float2(sv_1[80], sv_1[81]);
                        #pragma unroll
                        for (int pair_13 = 2; pair_13 < 16; pair_13 += 2) {
                            partial_7_1 = add_f32x2(partial_7_1, make_float2(sv_1[80 + pair_13], sv_1[80 + (pair_13 + 1)]));
                        }
                        block_sum2_1 = fma_f32x2_rn_ftz(partial_7_1, scale2_6_1, block_sum2_1);
                        unsigned int _fp4_6[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_6[0]) : "f"(sv_1[64]), "f"(sv_1[65]), "f"(sv_1[66]), "f"(sv_1[67]), "f"(sv_1[68]), "f"(sv_1[69]), "f"(sv_1[70]), "f"(sv_1[71]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_6[1]) : "f"(sv_1[72]), "f"(sv_1[73]), "f"(sv_1[74]), "f"(sv_1[75]), "f"(sv_1[76]), "f"(sv_1[77]), "f"(sv_1[78]), "f"(sv_1[79]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_6[2]) : "f"(sv_1[80]), "f"(sv_1[81]), "f"(sv_1[82]), "f"(sv_1[83]), "f"(sv_1[84]), "f"(sv_1[85]), "f"(sv_1[86]), "f"(sv_1[87]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_6[3]) : "f"(sv_1[88]), "f"(sv_1[89]), "f"(sv_1[90]), "f"(sv_1[91]), "f"(sv_1[92]), "f"(sv_1[93]), "f"(sv_1[94]), "f"(sv_1[95]));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_p_addr + p_slot * 8192 + (unsigned int)((warp % 4 * 32 + lane) * 64 + 32 ^ ((warp % 4 * 32 + lane) * 64 + 32 >> 7 & 3) << 4))), "r"(_fp4_6[0]), "r"(_fp4_6[1]), "r"(_fp4_6[2]), "r"(_fp4_6[3]) : "memory");
                        const float2 _fma_b2_172 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_173 = {2.5849626f - block_max_scaled_1, 2.5849626f - block_max_scaled_1};
                        float2 _fma_pair_174 = fma_f32x2(make_float2((sv_1 + 96)[0], (sv_1 + 96)[1]), _fma_b2_172, _fma_c2_173);
                        (sv_1 + 96)[0] = _fma_pair_174.x;
                        (sv_1 + 96)[1] = _fma_pair_174.y;
                        float2 _fma_pair_175 = fma_f32x2(make_float2((sv_1 + 96)[2], (sv_1 + 96)[3]), _fma_b2_172, _fma_c2_173);
                        (sv_1 + 96)[2] = _fma_pair_175.x;
                        (sv_1 + 96)[3] = _fma_pair_175.y;
                        float2 _fma_pair_176 = fma_f32x2(make_float2((sv_1 + 96)[4], (sv_1 + 96)[5]), _fma_b2_172, _fma_c2_173);
                        (sv_1 + 96)[4] = _fma_pair_176.x;
                        (sv_1 + 96)[5] = _fma_pair_176.y;
                        float2 _fma_pair_177 = fma_f32x2(make_float2((sv_1 + 96)[6], (sv_1 + 96)[7]), _fma_b2_172, _fma_c2_173);
                        (sv_1 + 96)[6] = _fma_pair_177.x;
                        (sv_1 + 96)[7] = _fma_pair_177.y;
                        float2 _fma_pair_178 = fma_f32x2(make_float2((sv_1 + 96)[8], (sv_1 + 96)[9]), _fma_b2_172, _fma_c2_173);
                        (sv_1 + 96)[8] = _fma_pair_178.x;
                        (sv_1 + 96)[9] = _fma_pair_178.y;
                        float2 _fma_pair_179 = fma_f32x2(make_float2((sv_1 + 96)[10], (sv_1 + 96)[11]), _fma_b2_172, _fma_c2_173);
                        (sv_1 + 96)[10] = _fma_pair_179.x;
                        (sv_1 + 96)[11] = _fma_pair_179.y;
                        float2 _fma_pair_180 = fma_f32x2(make_float2((sv_1 + 96)[12], (sv_1 + 96)[13]), _fma_b2_172, _fma_c2_173);
                        (sv_1 + 96)[12] = _fma_pair_180.x;
                        (sv_1 + 96)[13] = _fma_pair_180.y;
                        float2 _fma_pair_181 = fma_f32x2(make_float2((sv_1 + 96)[14], (sv_1 + 96)[15]), _fma_b2_172, _fma_c2_173);
                        (sv_1 + 96)[14] = _fma_pair_181.x;
                        (sv_1 + 96)[15] = _fma_pair_181.y;
                        float2 _fma_pair_182 = fma_f32x2(make_float2((sv_1 + 96)[16], (sv_1 + 96)[17]), _fma_b2_172, _fma_c2_173);
                        (sv_1 + 96)[16] = _fma_pair_182.x;
                        (sv_1 + 96)[17] = _fma_pair_182.y;
                        float2 _fma_pair_183 = fma_f32x2(make_float2((sv_1 + 96)[18], (sv_1 + 96)[19]), _fma_b2_172, _fma_c2_173);
                        (sv_1 + 96)[18] = _fma_pair_183.x;
                        (sv_1 + 96)[19] = _fma_pair_183.y;
                        float2 _fma_pair_184 = fma_f32x2(make_float2((sv_1 + 96)[20], (sv_1 + 96)[21]), _fma_b2_172, _fma_c2_173);
                        (sv_1 + 96)[20] = _fma_pair_184.x;
                        (sv_1 + 96)[21] = _fma_pair_184.y;
                        float2 _fma_pair_185 = fma_f32x2(make_float2((sv_1 + 96)[22], (sv_1 + 96)[23]), _fma_b2_172, _fma_c2_173);
                        (sv_1 + 96)[22] = _fma_pair_185.x;
                        (sv_1 + 96)[23] = _fma_pair_185.y;
                        float2 _fma_pair_186 = fma_f32x2(make_float2((sv_1 + 96)[24], (sv_1 + 96)[25]), _fma_b2_172, _fma_c2_173);
                        (sv_1 + 96)[24] = _fma_pair_186.x;
                        (sv_1 + 96)[25] = _fma_pair_186.y;
                        float2 _fma_pair_187 = fma_f32x2(make_float2((sv_1 + 96)[26], (sv_1 + 96)[27]), _fma_b2_172, _fma_c2_173);
                        (sv_1 + 96)[26] = _fma_pair_187.x;
                        (sv_1 + 96)[27] = _fma_pair_187.y;
                        float2 _fma_pair_188 = fma_f32x2(make_float2((sv_1 + 96)[28], (sv_1 + 96)[29]), _fma_b2_172, _fma_c2_173);
                        (sv_1 + 96)[28] = _fma_pair_188.x;
                        (sv_1 + 96)[29] = _fma_pair_188.y;
                        float2 _fma_pair_189 = fma_f32x2(make_float2((sv_1 + 96)[30], (sv_1 + 96)[31]), _fma_b2_172, _fma_c2_173);
                        (sv_1 + 96)[30] = _fma_pair_189.x;
                        (sv_1 + 96)[31] = _fma_pair_189.y;
                        {
                            #pragma unroll
                            for (int _le = 0; _le < 24; _le++) {
                                sv_1[_le + 96] = approx_exp2(sv_1[_le + 96]);
                            }
                            float2 _exp2_pair_190 = ex2_emulation_f32x2_value(make_float2(sv_1[120], sv_1[121]));
                            sv_1[120] = _exp2_pair_190.x;
                            sv_1[121] = _exp2_pair_190.y;
                            float2 _exp2_pair_191 = ex2_emulation_f32x2_value(make_float2(sv_1[122], sv_1[123]));
                            sv_1[122] = _exp2_pair_191.x;
                            sv_1[123] = _exp2_pair_191.y;
                            float2 _exp2_pair_192 = ex2_emulation_f32x2_value(make_float2(sv_1[124], sv_1[125]));
                            sv_1[124] = _exp2_pair_192.x;
                            sv_1[125] = _exp2_pair_192.y;
                            float2 _exp2_pair_193 = ex2_emulation_f32x2_value(make_float2(sv_1[126], sv_1[127]));
                            sv_1[126] = _exp2_pair_193.x;
                            sv_1[127] = _exp2_pair_193.y;
                        }
                        float2 partial_8_1 = make_float2(sv_1[96], sv_1[97]);
                        #pragma unroll
                        for (int pair_14 = 2; pair_14 < 16; pair_14 += 2) {
                            partial_8_1 = add_f32x2(partial_8_1, make_float2(sv_1[96 + pair_14], sv_1[96 + (pair_14 + 1)]));
                        }
                        float2 scale2_9_1 = make_float2(p_scale_1, p_scale_1);
                        block_sum2_1 = fma_f32x2_rn_ftz(partial_8_1, scale2_9_1, block_sum2_1);
                        float2 partial_10_1 = make_float2(sv_1[112], sv_1[113]);
                        #pragma unroll
                        for (int pair_15 = 2; pair_15 < 16; pair_15 += 2) {
                            partial_10_1 = add_f32x2(partial_10_1, make_float2(sv_1[112 + pair_15], sv_1[112 + (pair_15 + 1)]));
                        }
                        block_sum2_1 = fma_f32x2_rn_ftz(partial_10_1, scale2_9_1, block_sum2_1);
                        unsigned int _fp4_7[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_7[0]) : "f"(sv_1[96]), "f"(sv_1[97]), "f"(sv_1[98]), "f"(sv_1[99]), "f"(sv_1[100]), "f"(sv_1[101]), "f"(sv_1[102]), "f"(sv_1[103]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_7[1]) : "f"(sv_1[104]), "f"(sv_1[105]), "f"(sv_1[106]), "f"(sv_1[107]), "f"(sv_1[108]), "f"(sv_1[109]), "f"(sv_1[110]), "f"(sv_1[111]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_7[2]) : "f"(sv_1[112]), "f"(sv_1[113]), "f"(sv_1[114]), "f"(sv_1[115]), "f"(sv_1[116]), "f"(sv_1[117]), "f"(sv_1[118]), "f"(sv_1[119]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_7[3]) : "f"(sv_1[120]), "f"(sv_1[121]), "f"(sv_1[122]), "f"(sv_1[123]), "f"(sv_1[124]), "f"(sv_1[125]), "f"(sv_1[126]), "f"(sv_1[127]));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_p_addr + p_slot * 8192 + (unsigned int)((warp % 4 * 32 + lane) * 64 + 48 ^ ((warp % 4 * 32 + lane) * 64 + 48 >> 7 & 3) << 4))), "r"(_fp4_7[0]), "r"(_fp4_7[1]), "r"(_fp4_7[2]), "r"(_fp4_7[3]) : "memory");
                        int pv_sfa_hi_col_1 = (unsigned int)412 + stage * 8 + (unsigned int)(warp % 4) + (p_slot & 1) * 24;
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(taddr + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)pv_sfa_hi_col_1), "r"(sf_packed_1[0]));
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((p_full_addr + (p_slot) * 8) & 0xFEFFFFFF) : "memory");
                        p_slot += 1;
                        if (p_slot == (stage * 2 + 2)) { p_slot = stage * 2; }
                        block_sum_1 = block_sum2_1.x + block_sum2_1.y;
                        row_sum = __fmaf_rn(row_sum, acc_scale_1, block_sum_1);
                        mbarrier_wait(corr_done_addr + (stage) * 8, _phase_corr_done);
                        _phase_corr_done ^= 1;
                    }
#endif
                    mbarrier_wait(o_full_addr + (stage) * 8, _phase_o_full);
                    _phase_o_full ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float final_scale = 0.0f;
#if __CUDA_ARCH__ == 1030
                    float _rcp_84 = approx_rcp(row_sum);
                    final_scale = ((row_sum != 0.0f && row_sum == row_sum) ? _rcp_84 : 0.0f);
#else
                    float _rcp_194 = approx_rcp(row_sum);
                    final_scale = ((row_sum != 0.0f && row_sum == row_sum) ? _rcp_194 : 0.0f);
#endif
                    int seg_len_out = seg_len;
                    int local_row = ((unsigned int)m_block + stage) * 128 + (unsigned int)(warp % 4 * 32 + lane);
                    int token = seg_begin + local_row;
                    long long out_off = ((long long)token * (long long)heads + (long long)head) * 128;
                    int tmem_o_off = make_warp_uniform((unsigned int)128 + stage * 128);
                    unsigned int row_addr = o_stage_addr + ((unsigned int)scale_off + (unsigned int)(warp % 4 * 32 + lane)) * 272;
                    asm volatile("cp.async.bulk.wait_group.read 0;");
                    #pragma unroll
                    for (int part = 0; part < 2; part++) {
                        int o_addr = taddr + (unsigned int)tmem_o_off + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(part * 64);
                        float ov[64];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(ov[0]), "=f"(ov[1]), "=f"(ov[2]), "=f"(ov[3]), "=f"(ov[4]), "=f"(ov[5]), "=f"(ov[6]), "=f"(ov[7]), "=f"(ov[8]), "=f"(ov[9]), "=f"(ov[10]), "=f"(ov[11]), "=f"(ov[12]), "=f"(ov[13]), "=f"(ov[14]), "=f"(ov[15]), "=f"(ov[16]), "=f"(ov[17]), "=f"(ov[18]), "=f"(ov[19]), "=f"(ov[20]), "=f"(ov[21]), "=f"(ov[22]), "=f"(ov[23]), "=f"(ov[24]), "=f"(ov[25]), "=f"(ov[26]), "=f"(ov[27]), "=f"(ov[28]), "=f"(ov[29]), "=f"(ov[30]), "=f"(ov[31])
                            : "r"(o_addr));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(ov[32]), "=f"(ov[33]), "=f"(ov[34]), "=f"(ov[35]), "=f"(ov[36]), "=f"(ov[37]), "=f"(ov[38]), "=f"(ov[39]), "=f"(ov[40]), "=f"(ov[41]), "=f"(ov[42]), "=f"(ov[43]), "=f"(ov[44]), "=f"(ov[45]), "=f"(ov[46]), "=f"(ov[47]), "=f"(ov[48]), "=f"(ov[49]), "=f"(ov[50]), "=f"(ov[51]), "=f"(ov[52]), "=f"(ov[53]), "=f"(ov[54]), "=f"(ov[55]), "=f"(ov[56]), "=f"(ov[57]), "=f"(ov[58]), "=f"(ov[59]), "=f"(ov[60]), "=f"(ov[61]), "=f"(ov[62]), "=f"(ov[63])
                            : "r"(o_addr + 32));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        if (local_row < seg_len_out) {
                            float _scaled_0[16];
#if __CUDA_ARCH__ == 1030
                            const float2 _scale2_85 = {final_scale, final_scale};
#else
                            const float2 _scale2_195 = {final_scale, final_scale};
#endif
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
#if __CUDA_ARCH__ == 1030
                                reinterpret_cast<float2*>(_scaled_0)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>(ov)[_ls], _scale2_85);
#else
                                reinterpret_cast<float2*>(_scaled_0)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>(ov)[_ls], _scale2_195);
#endif
                            unsigned int _packed_0[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_0[_lp*2 + 0], _scaled_0[_lp*2+1 + 0]));
                                _packed_0[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_0[_lp*2 + 2], _scaled_0[_lp*2+1 + 2]));
                                _packed_0[_lp + 1] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_0[_lp*2 + 4], _scaled_0[_lp*2+1 + 4]));
                                _packed_0[_lp + 2] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_0[_lp*2 + 6], _scaled_0[_lp*2+1 + 6]));
                                _packed_0[_lp + 3] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_0[_lp*2 + 8], _scaled_0[_lp*2+1 + 8]));
                                _packed_0[_lp + 4] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_0[_lp*2 + 10], _scaled_0[_lp*2+1 + 10]));
                                _packed_0[_lp + 5] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_0[_lp*2 + 12], _scaled_0[_lp*2+1 + 12]));
                                _packed_0[_lp + 6] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_0[_lp*2 + 14], _scaled_0[_lp*2+1 + 14]));
                                _packed_0[_lp + 7] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_addr + (unsigned int)(part * 4 * 32)), "r"(_packed_0[0]), "r"(_packed_0[1]), "r"(_packed_0[2]), "r"(_packed_0[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_addr + (unsigned int)(part * 4 * 32 + 16)), "r"(_packed_0[4]), "r"(_packed_0[5]), "r"(_packed_0[6]), "r"(_packed_0[7]) : "memory");
                            float _scaled_1[16];
#if __CUDA_ARCH__ == 1030
                            const float2 _scale2_86 = {final_scale, final_scale};
#else
                            const float2 _scale2_196 = {final_scale, final_scale};
#endif
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
#if __CUDA_ARCH__ == 1030
                                reinterpret_cast<float2*>(_scaled_1)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((ov + 16))[_ls], _scale2_86);
#else
                                reinterpret_cast<float2*>(_scaled_1)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((ov + 16))[_ls], _scale2_196);
#endif
                            unsigned int _packed_1[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_1[_lp*2 + 0], _scaled_1[_lp*2+1 + 0]));
                                _packed_1[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_1[_lp*2 + 2], _scaled_1[_lp*2+1 + 2]));
                                _packed_1[_lp + 1] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_1[_lp*2 + 4], _scaled_1[_lp*2+1 + 4]));
                                _packed_1[_lp + 2] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_1[_lp*2 + 6], _scaled_1[_lp*2+1 + 6]));
                                _packed_1[_lp + 3] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_1[_lp*2 + 8], _scaled_1[_lp*2+1 + 8]));
                                _packed_1[_lp + 4] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_1[_lp*2 + 10], _scaled_1[_lp*2+1 + 10]));
                                _packed_1[_lp + 5] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_1[_lp*2 + 12], _scaled_1[_lp*2+1 + 12]));
                                _packed_1[_lp + 6] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_1[_lp*2 + 14], _scaled_1[_lp*2+1 + 14]));
                                _packed_1[_lp + 7] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_addr + (unsigned int)((part * 4 + 1) * 32)), "r"(_packed_1[0]), "r"(_packed_1[1]), "r"(_packed_1[2]), "r"(_packed_1[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_addr + (unsigned int)((part * 4 + 1) * 32 + 16)), "r"(_packed_1[4]), "r"(_packed_1[5]), "r"(_packed_1[6]), "r"(_packed_1[7]) : "memory");
                            float _scaled_2[16];
#if __CUDA_ARCH__ == 1030
                            const float2 _scale2_87 = {final_scale, final_scale};
#else
                            const float2 _scale2_197 = {final_scale, final_scale};
#endif
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
#if __CUDA_ARCH__ == 1030
                                reinterpret_cast<float2*>(_scaled_2)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((ov + 32))[_ls], _scale2_87);
#else
                                reinterpret_cast<float2*>(_scaled_2)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((ov + 32))[_ls], _scale2_197);
#endif
                            unsigned int _packed_2[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_2[_lp*2 + 0], _scaled_2[_lp*2+1 + 0]));
                                _packed_2[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_2[_lp*2 + 2], _scaled_2[_lp*2+1 + 2]));
                                _packed_2[_lp + 1] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_2[_lp*2 + 4], _scaled_2[_lp*2+1 + 4]));
                                _packed_2[_lp + 2] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_2[_lp*2 + 6], _scaled_2[_lp*2+1 + 6]));
                                _packed_2[_lp + 3] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_2[_lp*2 + 8], _scaled_2[_lp*2+1 + 8]));
                                _packed_2[_lp + 4] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_2[_lp*2 + 10], _scaled_2[_lp*2+1 + 10]));
                                _packed_2[_lp + 5] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_2[_lp*2 + 12], _scaled_2[_lp*2+1 + 12]));
                                _packed_2[_lp + 6] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_2[_lp*2 + 14], _scaled_2[_lp*2+1 + 14]));
                                _packed_2[_lp + 7] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_addr + (unsigned int)((part * 4 + 2) * 32)), "r"(_packed_2[0]), "r"(_packed_2[1]), "r"(_packed_2[2]), "r"(_packed_2[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_addr + (unsigned int)((part * 4 + 2) * 32 + 16)), "r"(_packed_2[4]), "r"(_packed_2[5]), "r"(_packed_2[6]), "r"(_packed_2[7]) : "memory");
                            float _scaled_3[16];
#if __CUDA_ARCH__ == 1030
                            const float2 _scale2_88 = {final_scale, final_scale};
#else
                            const float2 _scale2_198 = {final_scale, final_scale};
#endif
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
#if __CUDA_ARCH__ == 1030
                                reinterpret_cast<float2*>(_scaled_3)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((ov + 48))[_ls], _scale2_88);
#else
                                reinterpret_cast<float2*>(_scaled_3)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((ov + 48))[_ls], _scale2_198);
#endif
                            unsigned int _packed_3[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_3[_lp*2 + 0], _scaled_3[_lp*2+1 + 0]));
                                _packed_3[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_3[_lp*2 + 2], _scaled_3[_lp*2+1 + 2]));
                                _packed_3[_lp + 1] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_3[_lp*2 + 4], _scaled_3[_lp*2+1 + 4]));
                                _packed_3[_lp + 2] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_3[_lp*2 + 6], _scaled_3[_lp*2+1 + 6]));
                                _packed_3[_lp + 3] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_3[_lp*2 + 8], _scaled_3[_lp*2+1 + 8]));
                                _packed_3[_lp + 4] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_3[_lp*2 + 10], _scaled_3[_lp*2+1 + 10]));
                                _packed_3[_lp + 5] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_3[_lp*2 + 12], _scaled_3[_lp*2+1 + 12]));
                                _packed_3[_lp + 6] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_3[_lp*2 + 14], _scaled_3[_lp*2+1 + 14]));
                                _packed_3[_lp + 7] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_addr + (unsigned int)((part * 4 + 3) * 32)), "r"(_packed_3[0]), "r"(_packed_3[1]), "r"(_packed_3[2]), "r"(_packed_3[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(row_addr + (unsigned int)((part * 4 + 3) * 32 + 16)), "r"(_packed_3[4]), "r"(_packed_3[5]), "r"(_packed_3[6]), "r"(_packed_3[7]) : "memory");
                        }
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    if (local_row < seg_len_out) {
                        {
#if __CUDA_ARCH__ == 1030
                            void* _cpbulk_dst_89 = reinterpret_cast<void*>(O + out_off);
#else
                            void* _cpbulk_dst_199 = reinterpret_cast<void*>(O + out_off);
#endif
                            asm volatile(
                                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
#if __CUDA_ARCH__ == 1030
                                :: "l"(_cpbulk_dst_89), "r"(row_addr), "r"((uint32_t)(256))
#else
                                :: "l"(_cpbulk_dst_199), "r"(row_addr), "r"((uint32_t)(256))
#endif
                                : "memory");
                        }
                    }
                    asm volatile("cp.async.bulk.commit_group;");
                }
            }
            asm volatile("cp.async.bulk.wait_group.read 0;");
        }
    }
    // ---- Role: correction_lo ----
    if (warp >= 8 && warp <= 9) {
        { // correction_lo_main
            unsigned int total_tiles_1 = num_tiles;
            int owned_row_base = make_warp_uniform(warp % 2 * 32);
            unsigned int p_pub0 = 0;
            unsigned int p_pub1 = 2;
            unsigned int _phase_p_empty_0 = 1;
            unsigned int _phase_p_empty_1 = 1;
            unsigned int _phase_corr_sig_lo_0 = 0;
            unsigned int _phase_corr_sig_lo_1 = 0;
            unsigned int _phase_o_full_0 = 0;
            unsigned int _phase_o_full_1 = 0;
            #pragma unroll 1
            for (unsigned int tile_idx_1 = cluster_id; tile_idx_1 < total_tiles_1; tile_idx_1 += num_clusters) {
                int head_1 = cl_head[tile_idx_1];
                int seg_begin_1 = cl_seg_begin[tile_idx_1];
                int seg_len_1 = cl_seg_len[tile_idx_1];
                int kv_base_1 = cl_kv_base[tile_idx_1];
                int q_word_1 = cl_q_block[tile_idx_1];
                int q_base_1 = q_word_1 & 65535;
                int q_half_1 = q_word_1 >> 16;
                int two_stages_1 = ((q_half_1 == 0) ? 1 : 0);
                int m_block_1 = q_base_1 + (q_half_1 >> 1) * 2 + cta_rank * (1 + two_stages_1);
                unsigned int num_n_blocks_1 = (seg_len_1 + 128 - 1) / 128;
                int kv_begin_1 = 0;
                int ws_slot_1 = -1;
                if (two_stages_1 == 1) {
                    mbarrier_wait(p_empty_addr, _phase_p_empty_0);
                    _phase_p_empty_0 ^= 1;
                    mbarrier_wait(p_empty_addr + 8, _phase_p_empty_1);
                    _phase_p_empty_1 ^= 1;
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((p_full_addr + (p_pub0) * 8) & 0xFEFFFFFF) : "memory");
                    p_pub0 += 1;
                    if (p_pub0 == 2) { p_pub0 = 0; }
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((p_full_addr + (p_pub1) * 8) & 0xFEFFFFFF) : "memory");
                    p_pub1 += 1;
                    if (p_pub1 == 4) { p_pub1 = 2; }
                    mbarrier_wait(corr_sig_lo_addr, _phase_corr_sig_lo_0);
                    _phase_corr_sig_lo_0 ^= 1;
                    mbarrier_arrive(corr_done_addr);
                    mbarrier_wait(corr_sig_lo_addr + 8, _phase_corr_sig_lo_1);
                    _phase_corr_sig_lo_1 ^= 1;
                    mbarrier_arrive(corr_done_addr + 8);
                    #pragma unroll 1
#if __CUDA_ARCH__ == 1030
                    for (unsigned int n_iter_1 = 1; n_iter_1 < num_n_blocks_1; n_iter_1++) {
#else
                    for (unsigned int n_iter_2 = 1; n_iter_2 < num_n_blocks_1; n_iter_2++) {
#endif
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
                                float _tmem_load_0[16];
                                tmem_ld_x16(&_tmem_load_0[0], addr);
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                const float2 _scale2_0 = {scale, scale};
                                #pragma unroll
                                for (int _ls = 0; _ls < 8; _ls++)
                                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_0)[_ls], _scale2_0);
                                tmem_st_x16_f32(addr, _tmem_load_0);
                            }
                        }
                        int skip_rescale_0 = skip_rescale;
                        if (skip_rescale_0 == 0) {
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        }
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((p_full_addr + (p_pub0) * 8) & 0xFEFFFFFF) : "memory");
                        p_pub0 += 1;
                        if (p_pub0 == 2) { p_pub0 = 0; }
                        mbarrier_arrive(corr_done_addr);
                        mbarrier_wait(corr_sig_lo_addr + 8, _phase_corr_sig_lo_1);
                        _phase_corr_sig_lo_1 ^= 1;
                        mbarrier_wait(p_empty_addr + 8, _phase_p_empty_1);
                        _phase_p_empty_1 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int logical_row_1 = owned_row_base + lane;
                        int tmem_row_base_2 = make_warp_uniform(owned_row_base << 16);
                        float scale_3 = row_state[logical_row_1 + 128];
                        int skip_rescale_4 = __all_sync(0xFFFFFFFF, scale_3 == 1.0f);
                        if (skip_rescale_4 == 0) {
                            #pragma unroll
                            for (int col_1 = 0; col_1 < 8; col_1++) {
                                int addr_1 = make_warp_uniform(taddr + (unsigned int)256 + (unsigned int)tmem_row_base_2 + (unsigned int)(col_1 * 16));
                                float _tmem_load_1[16];
                                tmem_ld_x16(&_tmem_load_1[0], addr_1);
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                const float2 _scale2_1 = {scale_3, scale_3};
                                #pragma unroll
                                for (int _ls = 0; _ls < 8; _ls++)
                                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_1)[_ls], _scale2_1);
                                tmem_st_x16_f32(addr_1, _tmem_load_1);
                            }
                        }
                        skip_rescale_0 = skip_rescale_4;
                        if (skip_rescale_0 == 0) {
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        }
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((p_full_addr + (p_pub1) * 8) & 0xFEFFFFFF) : "memory");
                        p_pub1 += 1;
                        if (p_pub1 == 4) { p_pub1 = 2; }
                        mbarrier_arrive(corr_done_addr + 8);
                    }
                    mbarrier_wait(o_full_addr, _phase_o_full_0);
                    _phase_o_full_0 ^= 1;
                    mbarrier_wait(o_full_addr + 8, _phase_o_full_1);
                    _phase_o_full_1 ^= 1;
                } else {
                    mbarrier_wait(p_empty_addr, _phase_p_empty_0);
                    _phase_p_empty_0 ^= 1;
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((p_full_addr + (p_pub0) * 8) & 0xFEFFFFFF) : "memory");
                    p_pub0 += 1;
                    if (p_pub0 == 2) { p_pub0 = 0; }
                    mbarrier_wait(corr_sig_lo_addr, _phase_corr_sig_lo_0);
                    _phase_corr_sig_lo_0 ^= 1;
                    mbarrier_arrive(corr_done_addr);
                    #pragma unroll 1
#if __CUDA_ARCH__ == 1030
                    for (unsigned int n_iter_2 = 1; n_iter_2 < num_n_blocks_1; n_iter_2++) {
#else
                    for (unsigned int n_iter_3 = 1; n_iter_3 < num_n_blocks_1; n_iter_3++) {
#endif
                        mbarrier_wait(corr_sig_lo_addr, _phase_corr_sig_lo_0);
                        _phase_corr_sig_lo_0 ^= 1;
                        mbarrier_wait(p_empty_addr, _phase_p_empty_0);
                        _phase_p_empty_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int logical_row_2 = owned_row_base + lane;
                        int tmem_row_base_1 = make_warp_uniform(owned_row_base << 16);
                        float scale_1 = row_state[logical_row_2];
                        int skip_rescale_1 = __all_sync(0xFFFFFFFF, scale_1 == 1.0f);
                        if (skip_rescale_1 == 0) {
                            #pragma unroll
                            for (int col_2 = 0; col_2 < 8; col_2++) {
                                int addr_2 = make_warp_uniform(taddr + (unsigned int)128 + (unsigned int)tmem_row_base_1 + (unsigned int)(col_2 * 16));
                                float _tmem_load_2[16];
                                tmem_ld_x16(&_tmem_load_2[0], addr_2);
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                const float2 _scale2_2 = {scale_1, scale_1};
                                #pragma unroll
                                for (int _ls = 0; _ls < 8; _ls++)
                                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_2)[_ls], _scale2_2);
                                tmem_st_x16_f32(addr_2, _tmem_load_2);
                            }
                        }
                        int skip_rescale_0_1 = skip_rescale_1;
                        if (skip_rescale_0_1 == 0) {
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        }
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((p_full_addr + (p_pub0) * 8) & 0xFEFFFFFF) : "memory");
                        p_pub0 += 1;
                        if (p_pub0 == 2) { p_pub0 = 0; }
                        mbarrier_arrive(corr_done_addr);
                    }
                    mbarrier_wait(o_full_addr, _phase_o_full_0);
                    _phase_o_full_0 ^= 1;
                }
            }
            mbarrier_wait(p_empty_addr, _phase_p_empty_0);
            _phase_p_empty_0 ^= 1;
            mbarrier_wait(p_empty_addr + 8, _phase_p_empty_1);
            _phase_p_empty_1 ^= 1;
        }
    }
    // ---- Role: correction_hi ----
    if (warp >= 10 && warp <= 11) {
        { // correction_hi_main
            unsigned int total_tiles_2 = num_tiles;
            int owned_row_base_1 = make_warp_uniform(64 + warp % 2 * 32);
            unsigned int p_pub0_1 = 0;
            unsigned int p_pub1_1 = 2;
            unsigned int _phase_p_empty_0_1 = 1;
            unsigned int _phase_p_empty_1_1 = 1;
            unsigned int _phase_corr_sig_hi_0 = 0;
            unsigned int _phase_corr_sig_hi_1 = 0;
            unsigned int _phase_o_full_0_1 = 0;
            unsigned int _phase_o_full_1_1 = 0;
            #pragma unroll 1
            for (unsigned int tile_idx_2 = cluster_id; tile_idx_2 < total_tiles_2; tile_idx_2 += num_clusters) {
                int head_2 = cl_head[tile_idx_2];
                int seg_begin_2 = cl_seg_begin[tile_idx_2];
                int seg_len_2 = cl_seg_len[tile_idx_2];
                int kv_base_2 = cl_kv_base[tile_idx_2];
                int q_word_2 = cl_q_block[tile_idx_2];
                int q_base_2 = q_word_2 & 65535;
                int q_half_2 = q_word_2 >> 16;
                int two_stages_2 = ((q_half_2 == 0) ? 1 : 0);
                int m_block_2 = q_base_2 + (q_half_2 >> 1) * 2 + cta_rank * (1 + two_stages_2);
                unsigned int num_n_blocks_2 = (seg_len_2 + 128 - 1) / 128;
                int kv_begin_2 = 0;
                int ws_slot_2 = -1;
                if (two_stages_2 == 1) {
                    mbarrier_wait(p_empty_addr, _phase_p_empty_0_1);
                    _phase_p_empty_0_1 ^= 1;
                    mbarrier_wait(p_empty_addr + 8, _phase_p_empty_1_1);
                    _phase_p_empty_1_1 ^= 1;
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((p_full_addr + (p_pub0_1) * 8) & 0xFEFFFFFF) : "memory");
                    p_pub0_1 += 1;
                    if (p_pub0_1 == 2) { p_pub0_1 = 0; }
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((p_full_addr + (p_pub1_1) * 8) & 0xFEFFFFFF) : "memory");
                    p_pub1_1 += 1;
                    if (p_pub1_1 == 4) { p_pub1_1 = 2; }
                    mbarrier_wait(corr_sig_hi_addr, _phase_corr_sig_hi_0);
                    _phase_corr_sig_hi_0 ^= 1;
                    mbarrier_arrive(corr_done_addr);
                    mbarrier_wait(corr_sig_hi_addr + 8, _phase_corr_sig_hi_1);
                    _phase_corr_sig_hi_1 ^= 1;
                    mbarrier_arrive(corr_done_addr + 8);
                    #pragma unroll 1
#if __CUDA_ARCH__ == 1030
                    for (unsigned int n_iter_3 = 1; n_iter_3 < num_n_blocks_2; n_iter_3++) {
#else
                    for (unsigned int n_iter_4 = 1; n_iter_4 < num_n_blocks_2; n_iter_4++) {
#endif
                        mbarrier_wait(corr_sig_hi_addr, _phase_corr_sig_hi_0);
                        _phase_corr_sig_hi_0 ^= 1;
                        mbarrier_wait(p_empty_addr, _phase_p_empty_0_1);
                        _phase_p_empty_0_1 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int logical_row_3 = owned_row_base_1 + lane;
                        int tmem_row_base_3 = make_warp_uniform(owned_row_base_1 << 16);
                        float scale_2 = row_state[logical_row_3];
                        int skip_rescale_2 = __all_sync(0xFFFFFFFF, scale_2 == 1.0f);
                        if (skip_rescale_2 == 0) {
                            #pragma unroll
                            for (int col_3 = 0; col_3 < 8; col_3++) {
                                int addr_3 = make_warp_uniform(taddr + (unsigned int)128 + (unsigned int)tmem_row_base_3 + (unsigned int)(col_3 * 16));
                                float _tmem_load_3[16];
                                tmem_ld_x16(&_tmem_load_3[0], addr_3);
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                const float2 _scale2_0 = {scale_2, scale_2};
                                #pragma unroll
                                for (int _ls = 0; _ls < 8; _ls++)
                                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_3)[_ls], _scale2_0);
                                tmem_st_x16_f32(addr_3, _tmem_load_3);
                            }
                        }
                        int skip_rescale_0_2 = skip_rescale_2;
                        if (skip_rescale_0_2 == 0) {
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        }
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((p_full_addr + (p_pub0_1) * 8) & 0xFEFFFFFF) : "memory");
                        p_pub0_1 += 1;
                        if (p_pub0_1 == 2) { p_pub0_1 = 0; }
                        mbarrier_arrive(corr_done_addr);
                        mbarrier_wait(corr_sig_hi_addr + 8, _phase_corr_sig_hi_1);
                        _phase_corr_sig_hi_1 ^= 1;
                        mbarrier_wait(p_empty_addr + 8, _phase_p_empty_1_1);
                        _phase_p_empty_1_1 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int logical_row_1_1 = owned_row_base_1 + lane;
                        int tmem_row_base_2_1 = make_warp_uniform(owned_row_base_1 << 16);
                        float scale_3_1 = row_state[logical_row_1_1 + 128];
                        int skip_rescale_4_1 = __all_sync(0xFFFFFFFF, scale_3_1 == 1.0f);
                        if (skip_rescale_4_1 == 0) {
                            #pragma unroll
                            for (int col_4 = 0; col_4 < 8; col_4++) {
                                int addr_4 = make_warp_uniform(taddr + (unsigned int)256 + (unsigned int)tmem_row_base_2_1 + (unsigned int)(col_4 * 16));
                                float _tmem_load_4[16];
                                tmem_ld_x16(&_tmem_load_4[0], addr_4);
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                const float2 _scale2_1 = {scale_3_1, scale_3_1};
                                #pragma unroll
                                for (int _ls = 0; _ls < 8; _ls++)
                                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_4)[_ls], _scale2_1);
                                tmem_st_x16_f32(addr_4, _tmem_load_4);
                            }
                        }
                        skip_rescale_0_2 = skip_rescale_4_1;
                        if (skip_rescale_0_2 == 0) {
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        }
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((p_full_addr + (p_pub1_1) * 8) & 0xFEFFFFFF) : "memory");
                        p_pub1_1 += 1;
                        if (p_pub1_1 == 4) { p_pub1_1 = 2; }
                        mbarrier_arrive(corr_done_addr + 8);
                    }
                    mbarrier_wait(o_full_addr, _phase_o_full_0_1);
                    _phase_o_full_0_1 ^= 1;
                    mbarrier_wait(o_full_addr + 8, _phase_o_full_1_1);
                    _phase_o_full_1_1 ^= 1;
                } else {
                    mbarrier_wait(p_empty_addr, _phase_p_empty_0_1);
                    _phase_p_empty_0_1 ^= 1;
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((p_full_addr + (p_pub0_1) * 8) & 0xFEFFFFFF) : "memory");
                    p_pub0_1 += 1;
                    if (p_pub0_1 == 2) { p_pub0_1 = 0; }
                    mbarrier_wait(corr_sig_hi_addr, _phase_corr_sig_hi_0);
                    _phase_corr_sig_hi_0 ^= 1;
                    mbarrier_arrive(corr_done_addr);
                    #pragma unroll 1
#if __CUDA_ARCH__ == 1030
                    for (unsigned int n_iter_4 = 1; n_iter_4 < num_n_blocks_2; n_iter_4++) {
#else
                    for (unsigned int n_iter_5 = 1; n_iter_5 < num_n_blocks_2; n_iter_5++) {
#endif
                        mbarrier_wait(corr_sig_hi_addr, _phase_corr_sig_hi_0);
                        _phase_corr_sig_hi_0 ^= 1;
                        mbarrier_wait(p_empty_addr, _phase_p_empty_0_1);
                        _phase_p_empty_0_1 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int logical_row_4 = owned_row_base_1 + lane;
                        int tmem_row_base_4 = make_warp_uniform(owned_row_base_1 << 16);
                        float scale_4 = row_state[logical_row_4];
                        int skip_rescale_3 = __all_sync(0xFFFFFFFF, scale_4 == 1.0f);
                        if (skip_rescale_3 == 0) {
                            #pragma unroll
                            for (int col_5 = 0; col_5 < 8; col_5++) {
                                int addr_5 = make_warp_uniform(taddr + (unsigned int)128 + (unsigned int)tmem_row_base_4 + (unsigned int)(col_5 * 16));
                                float _tmem_load_5[16];
                                tmem_ld_x16(&_tmem_load_5[0], addr_5);
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                const float2 _scale2_2 = {scale_4, scale_4};
                                #pragma unroll
                                for (int _ls = 0; _ls < 8; _ls++)
                                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_5)[_ls], _scale2_2);
                                tmem_st_x16_f32(addr_5, _tmem_load_5);
                            }
                        }
                        int skip_rescale_0_3 = skip_rescale_3;
                        if (skip_rescale_0_3 == 0) {
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        }
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((p_full_addr + (p_pub0_1) * 8) & 0xFEFFFFFF) : "memory");
                        p_pub0_1 += 1;
                        if (p_pub0_1 == 2) { p_pub0_1 = 0; }
                        mbarrier_arrive(corr_done_addr);
                    }
                    mbarrier_wait(o_full_addr, _phase_o_full_0_1);
                    _phase_o_full_0_1 ^= 1;
                }
            }
            mbarrier_wait(p_empty_addr, _phase_p_empty_0_1);
            _phase_p_empty_0_1 ^= 1;
            mbarrier_wait(p_empty_addr + 8, _phase_p_empty_1_1);
            _phase_p_empty_1_1 ^= 1;
        }
    }
    // ---- Role: mma ----
    if (warp == 12) {
        { // mma_main
            unsigned int p_full_cursor0 = 0;
            unsigned int p_full_cursor1 = 2;
            unsigned int _phase_q_full_0 = 0;
            unsigned int _phase_q_full_1 = 0;
            unsigned int _phase_k_full = 0;
            unsigned int _phase_s_empty_0 = 0;
            unsigned int _phase_v_full = 0;
            unsigned int _phase_s_empty_1 = 0;
            unsigned int _phase_p_full = 0;
            unsigned int _phase_p_full_1 = 0;
            if (cta_rank == 0) {
                unsigned int total_tiles_3 = num_tiles;
                unsigned int k_stage = 0;
                unsigned int v_stage = 0;
                #pragma unroll 1
                for (unsigned int tile_idx_3 = cluster_id; tile_idx_3 < total_tiles_3; tile_idx_3 += num_clusters) {
                    int head_3 = cl_head[tile_idx_3];
                    int seg_begin_3 = cl_seg_begin[tile_idx_3];
                    int seg_len_3 = cl_seg_len[tile_idx_3];
                    int kv_base_3 = cl_kv_base[tile_idx_3];
                    int q_word_3 = cl_q_block[tile_idx_3];
                    int q_base_3 = q_word_3 & 65535;
                    int q_half_3 = q_word_3 >> 16;
                    int two_stages_3 = ((q_half_3 == 0) ? 1 : 0);
                    int m_block_3 = q_base_3 + (q_half_3 >> 1) * 2 + cta_rank * (1 + two_stages_3);
                    unsigned int num_n_blocks_3 = (seg_len_3 + 128 - 1) / 128;
                    int kv_begin_3 = 0;
                    int ws_slot_3 = -1;
                    if (two_stages_3 == 1) {
                        mbarrier_wait(q_full_addr, _phase_q_full_0);
                        _phase_q_full_0 ^= 1;
                        mbarrier_wait(q_full_addr + 8, _phase_q_full_1);
                        _phase_q_full_1 ^= 1;
                        mbarrier_wait(k_full_addr + (k_stage) * 8, _phase_k_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa_qk0, make_sf_cp_desc_lo_sbo256((((smem_sfq_addr) >> 4))));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfa_qk0 + 4), make_sf_cp_desc_lo_sbo256((((smem_sfq_addr) >> 4) + 8)));
                        }
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb_qk0, make_sf_cp_desc_lo_sbo256((((smem_sfk_addr) >> 4) + (k_stage) * 64)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb_qk0 + 4), make_sf_cp_desc_lo_sbo256((((smem_sfk_addr) >> 4) + (k_stage) * 64 + 8)));
                        }
                        int _mma_a_lo_0 = (((smem_q_addr) >> 4) & 0x3FFF) + (0) * 512;
                        int _mma_b_lo_0 = (((smem_k_addr) >> 4) & 0x3FFF) + (k_stage) * 256;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x80004020 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x80004020 << 32);

                                tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 0, b_desc + 0,
                                    0x10200480U, tmem_tmem_sfa_qk0 + 0, tmem_tmem_sfb_qk0 + 0, 0);
                                tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 2, b_desc + 2,
                                    0x10200480U, tmem_tmem_sfa_qk0 + 4, tmem_tmem_sfb_qk0 + 4, 1);
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
                        int _mma_a_lo_1 = (((smem_q_addr) >> 4) & 0x3FFF) + (1) * 512;
                        int _mma_b_lo_1 = (((smem_k_addr) >> 4) & 0x3FFF) + (k_stage) * 256;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x80004020 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x80004020 << 32);

                                tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 0, b_desc + 0,
                                    0x10200480U, tmem_tmem_sfa_qk1 + 0, tmem_tmem_sfb_qk0 + 0, 0);
                                tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 2, b_desc + 2,
                                    0x10200480U, tmem_tmem_sfa_qk1 + 4, tmem_tmem_sfb_qk0 + 4, 1);
                            }
                        }
                        elect_commit_cg2_multicast(s_full_addr + 8, (uint16_t)(3));
                        elect_commit_cg2_multicast(k_empty_addr + (k_stage) * 8, (uint16_t)(3));
                        k_stage += 1;
                        if (k_stage == 4) { k_stage = 0; _phase_k_full ^= 1; }
                        unsigned int first_pv = 1;
                        #pragma unroll 1
#if __CUDA_ARCH__ == 1030
                        for (unsigned int n_iter_5 = 0; n_iter_5 < num_n_blocks_3 - 1; n_iter_5++) {
#else
                        for (unsigned int n_iter_6 = 0; n_iter_6 < num_n_blocks_3 - 1; n_iter_6++) {
#endif
                            int first_pv_flag = first_pv;
                            mbarrier_wait(v_full_addr + (v_stage) * 8, _phase_v_full);
                            mbarrier_wait(k_full_addr + (k_stage) * 8, _phase_k_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            if (elect_sync()) {
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb_qk0, make_sf_cp_desc_lo_sbo256((((smem_sfk_addr) >> 4) + (k_stage) * 64)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb_qk0 + 4), make_sf_cp_desc_lo_sbo256((((smem_sfk_addr) >> 4) + (k_stage) * 64 + 8)));
                            }
                            mbarrier_wait(s_empty_addr + 8, _phase_s_empty_1);
                            _phase_s_empty_1 ^= 1;
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_a_lo_2 = (((smem_q_addr) >> 4) & 0x3FFF) + (0) * 512;
                            int _mma_b_lo_2 = (((smem_k_addr) >> 4) & 0x3FFF) + (k_stage) * 256;
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_2) | ((uint64_t)0x80004020 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_2) | ((uint64_t)0x80004020 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 0, b_desc + 0,
                                        0x10200480U, tmem_tmem_sfa_qk0 + 0, tmem_tmem_sfb_qk0 + 0, 0);
                                    tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 2, b_desc + 2,
                                        0x10200480U, tmem_tmem_sfa_qk0 + 4, tmem_tmem_sfb_qk0 + 4, 1);
                                }
                            }
                            elect_commit_cg2_multicast(s_full_addr, (uint16_t)(3));
                            mbarrier_wait(s_empty_addr, _phase_s_empty_0);
                            _phase_s_empty_0 ^= 1;
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_a_lo_3 = (((smem_q_addr) >> 4) & 0x3FFF) + (1) * 512;
                            int _mma_b_lo_3 = (((smem_k_addr) >> 4) & 0x3FFF) + (k_stage) * 256;
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_3) | ((uint64_t)0x80004020 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_3) | ((uint64_t)0x80004020 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 0, b_desc + 0,
                                        0x10200480U, tmem_tmem_sfa_qk1 + 0, tmem_tmem_sfb_qk0 + 0, 0);
                                    tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 2, b_desc + 2,
                                        0x10200480U, tmem_tmem_sfa_qk1 + 4, tmem_tmem_sfb_qk0 + 4, 1);
                                }
                            }
                            elect_commit_cg2_multicast(s_full_addr + 8, (uint16_t)(3));
                            elect_commit_cg2_multicast(k_empty_addr + (k_stage) * 8, (uint16_t)(3));
                            k_stage += 1;
                            if (k_stage == 4) { k_stage = 0; _phase_k_full ^= 1; }
                            if (elect_sync()) {
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb_pv_lo, make_sf_cp_desc_lo_sbo128((((smem_sfvt_lo_addr) >> 4) + (v_stage) * 64)));
                            }
                            if (elect_sync()) {
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb_pv_hi, make_sf_cp_desc_lo_sbo128((((smem_sfvt_hi_addr) >> 4) + (v_stage) * 64)));
                            }
                            mbarrier_wait(p_full_addr + (p_full_cursor0) * 8, _phase_p_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            if ((p_full_cursor0 & 1) == 0) {
                                int _mma_a_lo_4 = (((smem_p_addr) >> 4) & 0x3FFF) + (p_full_cursor0) * 512;
                                int _mma_b_lo_4 = (((smem_vt_addr) >> 4) & 0x3FFF) + (v_stage) * 256;
                                if (elect_sync()) {
                                    {
                                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_4) | ((uint64_t)0x80004020 << 32);
                                        uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_4) | ((uint64_t)0x80004020 << 32);

                                        tcgen05_mma_mxf4nvf4_bs_cta2(tmem_output_0, a_desc + 0, b_desc + 0,
                                            0x10200480U, tmem_tmem_sfa_pv0_lo + 0, tmem_tmem_sfb_pv_lo + 0, ((first_pv_flag) ? 0 : 1));
                                    }
                                }
                                int _mma_a_lo_5 = (((smem_p_addr + 32) >> 4) & 0x3FFF) + (p_full_cursor0) * 512;
                                int _mma_b_lo_5 = (((smem_vt_addr + 32) >> 4) & 0x3FFF) + (v_stage) * 256;
                                if (elect_sync()) {
                                    {
                                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_5) | ((uint64_t)0x80004020 << 32);
                                        uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_5) | ((uint64_t)0x80004020 << 32);

                                        tcgen05_mma_mxf4nvf4_bs_cta2(tmem_output_0, a_desc + 0, b_desc + 0,
                                            0x10200480U, tmem_tmem_sfa_pv0_hi + 0, tmem_tmem_sfb_pv_hi + 0, 1);
                                    }
                                }
                            } else {
                                int _mma_a_lo_6 = (((smem_p_addr) >> 4) & 0x3FFF) + (p_full_cursor0) * 512;
                                int _mma_b_lo_6 = (((smem_vt_addr) >> 4) & 0x3FFF) + (v_stage) * 256;
                                if (elect_sync()) {
                                    {
                                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_6) | ((uint64_t)0x80004020 << 32);
                                        uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_6) | ((uint64_t)0x80004020 << 32);

                                        tcgen05_mma_mxf4nvf4_bs_cta2(tmem_output_0, a_desc + 0, b_desc + 0,
                                            0x10200480U, tmem_tmem_sfa_pv0_lo_b1 + 0, tmem_tmem_sfb_pv_lo + 0, ((first_pv_flag) ? 0 : 1));
                                    }
                                }
                                int _mma_a_lo_7 = (((smem_p_addr + 32) >> 4) & 0x3FFF) + (p_full_cursor0) * 512;
                                int _mma_b_lo_7 = (((smem_vt_addr + 32) >> 4) & 0x3FFF) + (v_stage) * 256;
                                if (elect_sync()) {
                                    {
                                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_7) | ((uint64_t)0x80004020 << 32);
                                        uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_7) | ((uint64_t)0x80004020 << 32);

                                        tcgen05_mma_mxf4nvf4_bs_cta2(tmem_output_0, a_desc + 0, b_desc + 0,
                                            0x10200480U, tmem_tmem_sfa_pv0_hi_b1 + 0, tmem_tmem_sfb_pv_hi + 0, 1);
                                    }
                                }
                            }
                            elect_commit_cg2_multicast(p_empty_addr, (uint16_t)(3));
                            p_full_cursor0 += 1;
                            if (p_full_cursor0 == 2) { p_full_cursor0 = 0; _phase_p_full ^= 1; }
                            mbarrier_wait(p_full_addr + (p_full_cursor1) * 8, _phase_p_full_1);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            if ((p_full_cursor1 & 1) == 0) {
                                int _mma_a_lo_8 = (((smem_p_addr) >> 4) & 0x3FFF) + (p_full_cursor1) * 512;
                                int _mma_b_lo_8 = (((smem_vt_addr) >> 4) & 0x3FFF) + (v_stage) * 256;
                                if (elect_sync()) {
                                    {
                                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_8) | ((uint64_t)0x80004020 << 32);
                                        uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_8) | ((uint64_t)0x80004020 << 32);

                                        tcgen05_mma_mxf4nvf4_bs_cta2(tmem_output_1, a_desc + 0, b_desc + 0,
                                            0x10200480U, tmem_tmem_sfa_pv1_lo + 0, tmem_tmem_sfb_pv_lo + 0, ((first_pv_flag) ? 0 : 1));
                                    }
                                }
                                int _mma_a_lo_9 = (((smem_p_addr + 32) >> 4) & 0x3FFF) + (p_full_cursor1) * 512;
                                int _mma_b_lo_9 = (((smem_vt_addr + 32) >> 4) & 0x3FFF) + (v_stage) * 256;
                                if (elect_sync()) {
                                    {
                                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_9) | ((uint64_t)0x80004020 << 32);
                                        uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_9) | ((uint64_t)0x80004020 << 32);

                                        tcgen05_mma_mxf4nvf4_bs_cta2(tmem_output_1, a_desc + 0, b_desc + 0,
                                            0x10200480U, tmem_tmem_sfa_pv1_hi + 0, tmem_tmem_sfb_pv_hi + 0, 1);
                                    }
                                }
                            } else {
                                int _mma_a_lo_10 = (((smem_p_addr) >> 4) & 0x3FFF) + (p_full_cursor1) * 512;
                                int _mma_b_lo_10 = (((smem_vt_addr) >> 4) & 0x3FFF) + (v_stage) * 256;
                                if (elect_sync()) {
                                    {
                                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_10) | ((uint64_t)0x80004020 << 32);
                                        uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_10) | ((uint64_t)0x80004020 << 32);

                                        tcgen05_mma_mxf4nvf4_bs_cta2(tmem_output_1, a_desc + 0, b_desc + 0,
                                            0x10200480U, tmem_tmem_sfa_pv1_lo_b1 + 0, tmem_tmem_sfb_pv_lo + 0, ((first_pv_flag) ? 0 : 1));
                                    }
                                }
                                int _mma_a_lo_11 = (((smem_p_addr + 32) >> 4) & 0x3FFF) + (p_full_cursor1) * 512;
                                int _mma_b_lo_11 = (((smem_vt_addr + 32) >> 4) & 0x3FFF) + (v_stage) * 256;
                                if (elect_sync()) {
                                    {
                                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_11) | ((uint64_t)0x80004020 << 32);
                                        uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_11) | ((uint64_t)0x80004020 << 32);

                                        tcgen05_mma_mxf4nvf4_bs_cta2(tmem_output_1, a_desc + 0, b_desc + 0,
                                            0x10200480U, tmem_tmem_sfa_pv1_hi_b1 + 0, tmem_tmem_sfb_pv_hi + 0, 1);
                                    }
                                }
                            }
                            elect_commit_cg2_multicast(p_empty_addr + 8, (uint16_t)(3));
                            p_full_cursor1 += 1;
                            if (p_full_cursor1 == 4) { p_full_cursor1 = 2; _phase_p_full_1 ^= 1; }
                            elect_commit_cg2_multicast(v_empty_addr + (v_stage) * 8, (uint16_t)(3));
                            v_stage += 1;
                            if (v_stage == 5) { v_stage = 0; _phase_v_full ^= 1; }
                            first_pv = 0;
                        }
                        elect_commit_cg2_multicast(q_empty_addr, (uint16_t)(3));
                        int first_pv_flag_1 = first_pv;
                        mbarrier_wait(v_full_addr + (v_stage) * 8, _phase_v_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb_pv_lo, make_sf_cp_desc_lo_sbo128((((smem_sfvt_lo_addr) >> 4) + (v_stage) * 64)));
                        }
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb_pv_hi, make_sf_cp_desc_lo_sbo128((((smem_sfvt_hi_addr) >> 4) + (v_stage) * 64)));
                        }
                        mbarrier_wait(p_full_addr + (p_full_cursor0) * 8, _phase_p_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        if ((p_full_cursor0 & 1) == 0) {
                            int _mma_a_lo_12 = (((smem_p_addr) >> 4) & 0x3FFF) + (p_full_cursor0) * 512;
                            int _mma_b_lo_12 = (((smem_vt_addr) >> 4) & 0x3FFF) + (v_stage) * 256;
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_12) | ((uint64_t)0x80004020 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_12) | ((uint64_t)0x80004020 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_cta2(tmem_output_0, a_desc + 0, b_desc + 0,
                                        0x10200480U, tmem_tmem_sfa_pv0_lo + 0, tmem_tmem_sfb_pv_lo + 0, ((first_pv_flag_1) ? 0 : 1));
                                }
                            }
                            int _mma_a_lo_13 = (((smem_p_addr + 32) >> 4) & 0x3FFF) + (p_full_cursor0) * 512;
                            int _mma_b_lo_13 = (((smem_vt_addr + 32) >> 4) & 0x3FFF) + (v_stage) * 256;
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_13) | ((uint64_t)0x80004020 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_13) | ((uint64_t)0x80004020 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_cta2(tmem_output_0, a_desc + 0, b_desc + 0,
                                        0x10200480U, tmem_tmem_sfa_pv0_hi + 0, tmem_tmem_sfb_pv_hi + 0, 1);
                                }
                            }
                        } else {
                            int _mma_a_lo_14 = (((smem_p_addr) >> 4) & 0x3FFF) + (p_full_cursor0) * 512;
                            int _mma_b_lo_14 = (((smem_vt_addr) >> 4) & 0x3FFF) + (v_stage) * 256;
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_14) | ((uint64_t)0x80004020 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_14) | ((uint64_t)0x80004020 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_cta2(tmem_output_0, a_desc + 0, b_desc + 0,
                                        0x10200480U, tmem_tmem_sfa_pv0_lo_b1 + 0, tmem_tmem_sfb_pv_lo + 0, ((first_pv_flag_1) ? 0 : 1));
                                }
                            }
                            int _mma_a_lo_15 = (((smem_p_addr + 32) >> 4) & 0x3FFF) + (p_full_cursor0) * 512;
                            int _mma_b_lo_15 = (((smem_vt_addr + 32) >> 4) & 0x3FFF) + (v_stage) * 256;
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_15) | ((uint64_t)0x80004020 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_15) | ((uint64_t)0x80004020 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_cta2(tmem_output_0, a_desc + 0, b_desc + 0,
                                        0x10200480U, tmem_tmem_sfa_pv0_hi_b1 + 0, tmem_tmem_sfb_pv_hi + 0, 1);
                                }
                            }
                        }
                        elect_commit_cg2_multicast(p_empty_addr, (uint16_t)(3));
                        p_full_cursor0 += 1;
                        if (p_full_cursor0 == 2) { p_full_cursor0 = 0; _phase_p_full ^= 1; }
                        mbarrier_wait(p_full_addr + (p_full_cursor1) * 8, _phase_p_full_1);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        if ((p_full_cursor1 & 1) == 0) {
                            int _mma_a_lo_16 = (((smem_p_addr) >> 4) & 0x3FFF) + (p_full_cursor1) * 512;
                            int _mma_b_lo_16 = (((smem_vt_addr) >> 4) & 0x3FFF) + (v_stage) * 256;
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_16) | ((uint64_t)0x80004020 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_16) | ((uint64_t)0x80004020 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_cta2(tmem_output_1, a_desc + 0, b_desc + 0,
                                        0x10200480U, tmem_tmem_sfa_pv1_lo + 0, tmem_tmem_sfb_pv_lo + 0, ((first_pv_flag_1) ? 0 : 1));
                                }
                            }
                            int _mma_a_lo_17 = (((smem_p_addr + 32) >> 4) & 0x3FFF) + (p_full_cursor1) * 512;
                            int _mma_b_lo_17 = (((smem_vt_addr + 32) >> 4) & 0x3FFF) + (v_stage) * 256;
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_17) | ((uint64_t)0x80004020 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_17) | ((uint64_t)0x80004020 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_cta2(tmem_output_1, a_desc + 0, b_desc + 0,
                                        0x10200480U, tmem_tmem_sfa_pv1_hi + 0, tmem_tmem_sfb_pv_hi + 0, 1);
                                }
                            }
                        } else {
                            int _mma_a_lo_18 = (((smem_p_addr) >> 4) & 0x3FFF) + (p_full_cursor1) * 512;
                            int _mma_b_lo_18 = (((smem_vt_addr) >> 4) & 0x3FFF) + (v_stage) * 256;
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_18) | ((uint64_t)0x80004020 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_18) | ((uint64_t)0x80004020 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_cta2(tmem_output_1, a_desc + 0, b_desc + 0,
                                        0x10200480U, tmem_tmem_sfa_pv1_lo_b1 + 0, tmem_tmem_sfb_pv_lo + 0, ((first_pv_flag_1) ? 0 : 1));
                                }
                            }
                            int _mma_a_lo_19 = (((smem_p_addr + 32) >> 4) & 0x3FFF) + (p_full_cursor1) * 512;
                            int _mma_b_lo_19 = (((smem_vt_addr + 32) >> 4) & 0x3FFF) + (v_stage) * 256;
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_19) | ((uint64_t)0x80004020 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_19) | ((uint64_t)0x80004020 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_cta2(tmem_output_1, a_desc + 0, b_desc + 0,
                                        0x10200480U, tmem_tmem_sfa_pv1_hi_b1 + 0, tmem_tmem_sfb_pv_hi + 0, 1);
                                }
                            }
                        }
                        elect_commit_cg2_multicast(p_empty_addr + 8, (uint16_t)(3));
                        p_full_cursor1 += 1;
                        if (p_full_cursor1 == 4) { p_full_cursor1 = 2; _phase_p_full_1 ^= 1; }
                        elect_commit_cg2_multicast(v_empty_addr + (v_stage) * 8, (uint16_t)(3));
                        v_stage += 1;
                        if (v_stage == 5) { v_stage = 0; _phase_v_full ^= 1; }
                        mbarrier_wait(s_empty_addr + 8, _phase_s_empty_1);
                        _phase_s_empty_1 ^= 1;
                        elect_commit_cg2_multicast(o_full_addr, (uint16_t)(3));
                        elect_commit_cg2_multicast(o_full_addr + 8, (uint16_t)(3));
                    } else {
                        mbarrier_wait(q_full_addr, _phase_q_full_0);
                        _phase_q_full_0 ^= 1;
                        mbarrier_wait(k_full_addr + (k_stage) * 8, _phase_k_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa_qk0, make_sf_cp_desc_lo_sbo256((((smem_sfq_addr) >> 4))));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfa_qk0 + 4), make_sf_cp_desc_lo_sbo256((((smem_sfq_addr) >> 4) + 8)));
                        }
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb_qk0, make_sf_cp_desc_lo_sbo256((((smem_sfk_addr) >> 4) + (k_stage) * 64)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb_qk0 + 4), make_sf_cp_desc_lo_sbo256((((smem_sfk_addr) >> 4) + (k_stage) * 64 + 8)));
                        }
                        int _mma_a_lo_20 = (((smem_q_addr) >> 4) & 0x3FFF) + (0) * 512;
                        int _mma_b_lo_20 = (((smem_k_addr) >> 4) & 0x3FFF) + (k_stage) * 256;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_20) | ((uint64_t)0x80004020 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_20) | ((uint64_t)0x80004020 << 32);

                                tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 0, b_desc + 0,
                                    0x10200480U, tmem_tmem_sfa_qk0 + 0, tmem_tmem_sfb_qk0 + 0, 0);
                                tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 2, b_desc + 2,
                                    0x10200480U, tmem_tmem_sfa_qk0 + 4, tmem_tmem_sfb_qk0 + 4, 1);
                            }
                        }
                        elect_commit_cg2_multicast(s_full_addr, (uint16_t)(3));
                        elect_commit_cg2_multicast(k_empty_addr + (k_stage) * 8, (uint16_t)(3));
                        k_stage += 1;
                        if (k_stage == 4) { k_stage = 0; _phase_k_full ^= 1; }
                        unsigned int first_pv_1 = 1;
                        #pragma unroll 1
#if __CUDA_ARCH__ == 1030
                        for (unsigned int n_iter_6 = 0; n_iter_6 < num_n_blocks_3 - 1; n_iter_6++) {
#else
                        for (unsigned int n_iter_7 = 0; n_iter_7 < num_n_blocks_3 - 1; n_iter_7++) {
#endif
                            int first_pv_flag_2 = first_pv_1;
                            mbarrier_wait(v_full_addr + (v_stage) * 8, _phase_v_full);
                            mbarrier_wait(k_full_addr + (k_stage) * 8, _phase_k_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            if (elect_sync()) {
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb_qk0, make_sf_cp_desc_lo_sbo256((((smem_sfk_addr) >> 4) + (k_stage) * 64)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb_qk0 + 4), make_sf_cp_desc_lo_sbo256((((smem_sfk_addr) >> 4) + (k_stage) * 64 + 8)));
                            }
                            mbarrier_wait(s_empty_addr, _phase_s_empty_0);
                            _phase_s_empty_0 ^= 1;
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_a_lo_21 = (((smem_q_addr) >> 4) & 0x3FFF) + (0) * 512;
                            int _mma_b_lo_21 = (((smem_k_addr) >> 4) & 0x3FFF) + (k_stage) * 256;
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_21) | ((uint64_t)0x80004020 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_21) | ((uint64_t)0x80004020 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 0, b_desc + 0,
                                        0x10200480U, tmem_tmem_sfa_qk0 + 0, tmem_tmem_sfb_qk0 + 0, 0);
                                    tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 2, b_desc + 2,
                                        0x10200480U, tmem_tmem_sfa_qk0 + 4, tmem_tmem_sfb_qk0 + 4, 1);
                                }
                            }
                            elect_commit_cg2_multicast(s_full_addr, (uint16_t)(3));
                            elect_commit_cg2_multicast(k_empty_addr + (k_stage) * 8, (uint16_t)(3));
                            k_stage += 1;
                            if (k_stage == 4) { k_stage = 0; _phase_k_full ^= 1; }
                            if (elect_sync()) {
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb_pv_lo, make_sf_cp_desc_lo_sbo128((((smem_sfvt_lo_addr) >> 4) + (v_stage) * 64)));
                            }
                            if (elect_sync()) {
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb_pv_hi, make_sf_cp_desc_lo_sbo128((((smem_sfvt_hi_addr) >> 4) + (v_stage) * 64)));
                            }
                            mbarrier_wait(p_full_addr + (p_full_cursor0) * 8, _phase_p_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            if ((p_full_cursor0 & 1) == 0) {
                                int _mma_a_lo_22 = (((smem_p_addr) >> 4) & 0x3FFF) + (p_full_cursor0) * 512;
                                int _mma_b_lo_22 = (((smem_vt_addr) >> 4) & 0x3FFF) + (v_stage) * 256;
                                if (elect_sync()) {
                                    {
                                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_22) | ((uint64_t)0x80004020 << 32);
                                        uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_22) | ((uint64_t)0x80004020 << 32);

                                        tcgen05_mma_mxf4nvf4_bs_cta2(tmem_output_0, a_desc + 0, b_desc + 0,
                                            0x10200480U, tmem_tmem_sfa_pv0_lo + 0, tmem_tmem_sfb_pv_lo + 0, ((first_pv_flag_2) ? 0 : 1));
                                    }
                                }
                                int _mma_a_lo_23 = (((smem_p_addr + 32) >> 4) & 0x3FFF) + (p_full_cursor0) * 512;
                                int _mma_b_lo_23 = (((smem_vt_addr + 32) >> 4) & 0x3FFF) + (v_stage) * 256;
                                if (elect_sync()) {
                                    {
                                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_23) | ((uint64_t)0x80004020 << 32);
                                        uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_23) | ((uint64_t)0x80004020 << 32);

                                        tcgen05_mma_mxf4nvf4_bs_cta2(tmem_output_0, a_desc + 0, b_desc + 0,
                                            0x10200480U, tmem_tmem_sfa_pv0_hi + 0, tmem_tmem_sfb_pv_hi + 0, 1);
                                    }
                                }
                            } else {
                                int _mma_a_lo_24 = (((smem_p_addr) >> 4) & 0x3FFF) + (p_full_cursor0) * 512;
                                int _mma_b_lo_24 = (((smem_vt_addr) >> 4) & 0x3FFF) + (v_stage) * 256;
                                if (elect_sync()) {
                                    {
                                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_24) | ((uint64_t)0x80004020 << 32);
                                        uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_24) | ((uint64_t)0x80004020 << 32);

                                        tcgen05_mma_mxf4nvf4_bs_cta2(tmem_output_0, a_desc + 0, b_desc + 0,
                                            0x10200480U, tmem_tmem_sfa_pv0_lo_b1 + 0, tmem_tmem_sfb_pv_lo + 0, ((first_pv_flag_2) ? 0 : 1));
                                    }
                                }
                                int _mma_a_lo_25 = (((smem_p_addr + 32) >> 4) & 0x3FFF) + (p_full_cursor0) * 512;
                                int _mma_b_lo_25 = (((smem_vt_addr + 32) >> 4) & 0x3FFF) + (v_stage) * 256;
                                if (elect_sync()) {
                                    {
                                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_25) | ((uint64_t)0x80004020 << 32);
                                        uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_25) | ((uint64_t)0x80004020 << 32);

                                        tcgen05_mma_mxf4nvf4_bs_cta2(tmem_output_0, a_desc + 0, b_desc + 0,
                                            0x10200480U, tmem_tmem_sfa_pv0_hi_b1 + 0, tmem_tmem_sfb_pv_hi + 0, 1);
                                    }
                                }
                            }
                            elect_commit_cg2_multicast(p_empty_addr, (uint16_t)(3));
                            p_full_cursor0 += 1;
                            if (p_full_cursor0 == 2) { p_full_cursor0 = 0; _phase_p_full ^= 1; }
                            elect_commit_cg2_multicast(v_empty_addr + (v_stage) * 8, (uint16_t)(3));
                            v_stage += 1;
                            if (v_stage == 5) { v_stage = 0; _phase_v_full ^= 1; }
                            first_pv_1 = 0;
                        }
                        elect_commit_cg2_multicast(q_empty_addr, (uint16_t)(3));
                        int first_pv_flag_3 = first_pv_1;
                        mbarrier_wait(v_full_addr + (v_stage) * 8, _phase_v_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb_pv_lo, make_sf_cp_desc_lo_sbo128((((smem_sfvt_lo_addr) >> 4) + (v_stage) * 64)));
                        }
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb_pv_hi, make_sf_cp_desc_lo_sbo128((((smem_sfvt_hi_addr) >> 4) + (v_stage) * 64)));
                        }
                        mbarrier_wait(p_full_addr + (p_full_cursor0) * 8, _phase_p_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        if ((p_full_cursor0 & 1) == 0) {
                            int _mma_a_lo_26 = (((smem_p_addr) >> 4) & 0x3FFF) + (p_full_cursor0) * 512;
                            int _mma_b_lo_26 = (((smem_vt_addr) >> 4) & 0x3FFF) + (v_stage) * 256;
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_26) | ((uint64_t)0x80004020 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_26) | ((uint64_t)0x80004020 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_cta2(tmem_output_0, a_desc + 0, b_desc + 0,
                                        0x10200480U, tmem_tmem_sfa_pv0_lo + 0, tmem_tmem_sfb_pv_lo + 0, ((first_pv_flag_3) ? 0 : 1));
                                }
                            }
                            int _mma_a_lo_27 = (((smem_p_addr + 32) >> 4) & 0x3FFF) + (p_full_cursor0) * 512;
                            int _mma_b_lo_27 = (((smem_vt_addr + 32) >> 4) & 0x3FFF) + (v_stage) * 256;
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_27) | ((uint64_t)0x80004020 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_27) | ((uint64_t)0x80004020 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_cta2(tmem_output_0, a_desc + 0, b_desc + 0,
                                        0x10200480U, tmem_tmem_sfa_pv0_hi + 0, tmem_tmem_sfb_pv_hi + 0, 1);
                                }
                            }
                        } else {
                            int _mma_a_lo_28 = (((smem_p_addr) >> 4) & 0x3FFF) + (p_full_cursor0) * 512;
                            int _mma_b_lo_28 = (((smem_vt_addr) >> 4) & 0x3FFF) + (v_stage) * 256;
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_28) | ((uint64_t)0x80004020 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_28) | ((uint64_t)0x80004020 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_cta2(tmem_output_0, a_desc + 0, b_desc + 0,
                                        0x10200480U, tmem_tmem_sfa_pv0_lo_b1 + 0, tmem_tmem_sfb_pv_lo + 0, ((first_pv_flag_3) ? 0 : 1));
                                }
                            }
                            int _mma_a_lo_29 = (((smem_p_addr + 32) >> 4) & 0x3FFF) + (p_full_cursor0) * 512;
                            int _mma_b_lo_29 = (((smem_vt_addr + 32) >> 4) & 0x3FFF) + (v_stage) * 256;
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_29) | ((uint64_t)0x80004020 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_29) | ((uint64_t)0x80004020 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_cta2(tmem_output_0, a_desc + 0, b_desc + 0,
                                        0x10200480U, tmem_tmem_sfa_pv0_hi_b1 + 0, tmem_tmem_sfb_pv_hi + 0, 1);
                                }
                            }
                        }
                        elect_commit_cg2_multicast(p_empty_addr, (uint16_t)(3));
                        p_full_cursor0 += 1;
                        if (p_full_cursor0 == 2) { p_full_cursor0 = 0; _phase_p_full ^= 1; }
                        elect_commit_cg2_multicast(v_empty_addr + (v_stage) * 8, (uint16_t)(3));
                        v_stage += 1;
                        if (v_stage == 5) { v_stage = 0; _phase_v_full ^= 1; }
                        mbarrier_wait(s_empty_addr, _phase_s_empty_0);
                        _phase_s_empty_0 ^= 1;
                        elect_commit_cg2_multicast(o_full_addr, (uint16_t)(3));
                    }
                }
            }
        }
    }
    // ---- Role: load ----
    if (warp == 13) {
        { // load_main
            unsigned int total_tiles_4 = num_tiles;
            unsigned int k_load_stage = 0;
            unsigned int _phase_q_empty_0 = 1;
            unsigned int _phase_k_empty = 1;
            #pragma unroll 1
            for (unsigned int tile_idx_4 = cluster_id; tile_idx_4 < total_tiles_4; tile_idx_4 += num_clusters) {
                int head_4 = cl_head[tile_idx_4];
                int seg_begin_4 = cl_seg_begin[tile_idx_4];
                int seg_len_4 = cl_seg_len[tile_idx_4];
                int kv_base_4 = cl_kv_base[tile_idx_4];
                int q_word_4 = cl_q_block[tile_idx_4];
                int q_base_4 = q_word_4 & 65535;
                int q_half_4 = q_word_4 >> 16;
                int two_stages_4 = ((q_half_4 == 0) ? 1 : 0);
                int m_block_4 = q_base_4 + (q_half_4 >> 1) * 2 + cta_rank * (1 + two_stages_4);
                unsigned int num_n_blocks_4 = (seg_len_4 + 128 - 1) / 128;
                int kv_begin_4 = 0;
                int ws_slot_4 = -1;
                int q_tile = kv_base_4 + m_block_4;
                int q_sf_tile = head_4 * PB + q_tile;
                int q_row = q_sf_tile * 128;
                mbarrier_wait(q_empty_addr, _phase_q_empty_0);
                _phase_q_empty_0 ^= 1;
                if (elect_sync()) {
                    tma_2d_gmem2smem_cta2(smem_q_addr, (&Q), 0, q_row, ((q_full_addr) & 0xFEFFFFFF));
                    tma_2d_gmem2smem_cta2(smem_sfq_addr, (&SFQ), 0, q_sf_tile * 32, ((q_full_addr) & 0xFEFFFFFF));
                    if (two_stages_4 == 1) {
                        tma_2d_gmem2smem_cta2(smem_q_addr + 8192, (&Q), 0, q_row + 128, ((q_full_addr + 8) & 0xFEFFFFFF));
                        tma_2d_gmem2smem_cta2(smem_sfq_addr + 1024, (&SFQ), 0, (q_sf_tile + 1) * 32, ((q_full_addr + 8) & 0xFEFFFFFF));
                    }
                    asm volatile(
                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                        :: "r"((q_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(9216)) : "memory");
                    if (two_stages_4 == 1) {
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((q_full_addr + 8) & 0xFEFFFFFF), "r"((uint32_t)(9216)) : "memory");
                    }
                }
                #pragma unroll 1
                for (unsigned int ni = 0; ni < num_n_blocks_4; ni++) {
                    unsigned int n = num_n_blocks_4 - 1 - ni;
                    int kv_sf_tile = (unsigned int)(head_4 * PB + kv_base_4) + n;
                    int kv_row = kv_sf_tile * 128;
                    mbarrier_wait(k_empty_addr + (k_load_stage) * 8, _phase_k_empty);
                    if (elect_sync()) {
                        tma_2d_gmem2smem_cta2(smem_k_addr + k_load_stage * 4096, (&K), 0, kv_row + cta_rank * 64, ((k_full_addr + (k_load_stage) * 8) & 0xFEFFFFFF));
                        tma_2d_gmem2smem_cta2(smem_sfk_addr + k_load_stage * 1024, (&SFK), 0, kv_sf_tile * 32, ((k_full_addr + (k_load_stage) * 8) & 0xFEFFFFFF));
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((k_full_addr + (k_load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(5120)) : "memory");
                    }
                    k_load_stage += 1;
                    if (k_load_stage == 4) { k_load_stage = 0; _phase_k_empty ^= 1; }
                }
            }
            #pragma unroll
            for (int _ = 0; _ < 4; _++) {
                mbarrier_wait(k_empty_addr + (k_load_stage) * 8, _phase_k_empty);
                k_load_stage += 1;
                if (k_load_stage == 4) { k_load_stage = 0; _phase_k_empty ^= 1; }
            }
            mbarrier_wait(q_empty_addr, _phase_q_empty_0);
        }
    }
    // ---- Role: v_load ----
    if (warp == 14) {
        { // v_load_main
            unsigned int total_tiles_5 = num_tiles;
            unsigned int v_load_stage = 0;
            unsigned int _phase_v_empty = 1;
            #pragma unroll 1
            for (unsigned int tile_idx_5 = cluster_id; tile_idx_5 < total_tiles_5; tile_idx_5 += num_clusters) {
                int head_5 = cl_head[tile_idx_5];
                int seg_begin_5 = cl_seg_begin[tile_idx_5];
                int seg_len_5 = cl_seg_len[tile_idx_5];
                int kv_base_5 = cl_kv_base[tile_idx_5];
                int q_word_5 = cl_q_block[tile_idx_5];
                int q_base_5 = q_word_5 & 65535;
                int q_half_5 = q_word_5 >> 16;
                int two_stages_5 = ((q_half_5 == 0) ? 1 : 0);
                int m_block_5 = q_base_5 + (q_half_5 >> 1) * 2 + cta_rank * (1 + two_stages_5);
                unsigned int num_n_blocks_5 = (seg_len_5 + 128 - 1) / 128;
                int kv_begin_5 = 0;
                int ws_slot_5 = -1;
                #pragma unroll 1
                for (unsigned int ni_1 = 0; ni_1 < num_n_blocks_5; ni_1++) {
                    unsigned int n_1 = num_n_blocks_5 - 1 - ni_1;
                    int kv_tile = (unsigned int)kv_base_5 + n_1;
                    int kv_sf_tile_1 = head_5 * PB + kv_tile;
                    mbarrier_wait(v_empty_addr + (v_load_stage) * 8, _phase_v_empty);
                    if (elect_sync()) {
                        tma_2d_gmem2smem_cta2(smem_vt_addr + v_load_stage * 4096, (&Vt), kv_tile * 64, head_5 * 128 + cta_rank * 64, ((v_full_addr + (v_load_stage) * 8) & 0xFEFFFFFF));
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
            #pragma unroll
            for (int __1 = 0; __1 < 5; __1++) {
                mbarrier_wait(v_empty_addr + (v_load_stage) * 8, _phase_v_empty);
                v_load_stage += 1;
                if (v_load_stage == 5) { v_load_stage = 0; _phase_v_empty ^= 1; }
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
