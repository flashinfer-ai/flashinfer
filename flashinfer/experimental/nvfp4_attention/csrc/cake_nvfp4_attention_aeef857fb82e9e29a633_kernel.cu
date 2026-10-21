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
#define TMEM_NCOLS 432
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
#define NUM_K_PIPE_STAGES 4
#define NUM_V_PIPE_STAGES 5
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
#define SMEM_SMEM_SFP_CP_LO_OFF 83968
#define SMEM_SMEM_SFP_CP_LO_STAGE_BYTES 512
#define SMEM_SMEM_SFP_CP_LO_STRIDE 1024
#define SMEM_SMEM_SFP_CP_HI_OFF 84480
#define SMEM_SMEM_SFP_CP_HI_STAGE_BYTES 512
#define SMEM_SMEM_SFP_CP_HI_STRIDE 1024
#define SMEM_SMEM_O00_OFF 86016
#define SMEM_SMEM_O00_STAGE_BYTES 16384
#define SMEM_SMEM_O00_STRIDE 16384
#define SMEM_SMEM_O01_OFF 102400
#define SMEM_SMEM_O01_STAGE_BYTES 16384
#define SMEM_SMEM_O01_STRIDE 16384
#define SMEM_SMEM_O10_OFF 118784
#define SMEM_SMEM_O10_STAGE_BYTES 16384
#define SMEM_SMEM_O10_STRIDE 16384
#define SMEM_SMEM_O11_OFF 135168
#define SMEM_SMEM_O11_STAGE_BYTES 16384
#define SMEM_SMEM_O11_STRIDE 16384
#define SMEM_TOTAL 151552
#define THREADS 512
#define IS_CAUSAL 0
#define USE_TMEM_LD_RED 1
#define HAS_KV_TAIL 0
#define GROUPWISE_P_SCALE 0
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


__device__ __forceinline__ void tma_store_2d(
    const void *tmap, int x, int y, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2}], [%3];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(smem_addr) : "memory");
}



__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_cake_nvfp4_attention_aeef857fb82e9e29a633(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap Vt, const __grid_constant__ CUtensorMap SFQ, const __grid_constant__ CUtensorMap SFK, const __grid_constant__ CUtensorMap SFVtLo, const __grid_constant__ CUtensorMap SFVtHi, const __grid_constant__ CUtensorMap O, int seqlen_q, int seqlen_kv, int q_stride, int kv_stride, float softmax_scale_log2, int total_bh)
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
    uint8_t* smem_sfp_cp_lo = reinterpret_cast<uint8_t*>(smem_raw + 83968);
    const int smem_sfp_cp_lo_addr = smem + 83968;
    uint8_t* smem_sfp_cp_hi = reinterpret_cast<uint8_t*>(smem_raw + 84480);
    const int smem_sfp_cp_hi_addr = smem + 84480;
    __nv_bfloat16* smem_o00 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 86016);
    const int smem_o00_addr = smem + 86016;
    __nv_bfloat16* smem_o01 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 102400);
    const int smem_o01_addr = smem + 102400;
    __nv_bfloat16* smem_o10 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 118784);
    const int smem_o10_addr = smem + 118784;
    __nv_bfloat16* smem_o11 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 135168);
    const int smem_o11_addr = smem + 135168;

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

    // TMEM alloc (512 columns, 432 used)
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
    asm volatile("griddepcontrol.wait;" ::: "memory");

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 8 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
    }

    // ---- Role: softmax ----
    if (warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 208;");
        { // softmax_main
            unsigned int total_tiles = (seqlen_q + 4 * BLOCK_M - 1) / (4 * BLOCK_M) * total_bh;
            unsigned int stage = make_warp_uniform(warp / 4);
            int scale_off = make_warp_uniform(stage * (unsigned int)BLOCK_M);
            unsigned int _phase_s_full = 0;
            unsigned int _phase_p_empty = 1;
            unsigned int _phase_corr_done = 0;
            unsigned int _phase_o_full = 0;
            #pragma unroll 1
            for (unsigned int tile_idx = cluster_id; tile_idx < total_tiles; tile_idx += num_clusters) {
                unsigned int cluster_m_block = tile_idx % (unsigned int)((seqlen_q + 4 * BLOCK_M - 1) / (4 * BLOCK_M));
                unsigned int bh = tile_idx / (unsigned int)((seqlen_q + 4 * BLOCK_M - 1) / (4 * BLOCK_M));
                unsigned int m_block = cluster_m_block * 4 + (unsigned int)(cta_rank * 2);
                unsigned int num_n_blocks = (seqlen_kv + BLOCK_N - 1) / BLOCK_N;
                int causal_row = (m_block + stage) * (unsigned int)BLOCK_M + (unsigned int)(warp % 4 * 32 + lane);
                float row_max = -CUDART_INF_F;
                float row_max_scaled = 0.0f;
                float row_sum = 0.0f;
                #pragma unroll 1
                for (unsigned int n_iter = 0; n_iter < num_n_blocks; n_iter++) {
                    int n_block = num_n_blocks - 1 - n_iter;
                    mbarrier_wait_hint(s_full_addr + (stage) * 8, _phase_s_full, 10000000);
                    _phase_s_full ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int s_addr = taddr + (unsigned int)0 + (unsigned int)(warp % 4 * 32 << 16);
                    float sv[128];
                    float tile_max = -CUDART_INF_F;
                    {
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
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((s_empty_addr + (stage) * 8) & 0xFEFFFFFF) : "memory");
                    int tail_valid = seqlen_kv - n_block * BLOCK_N;
                    int valid_count = causal_row - n_block * BLOCK_N + 1;
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
                        float _exp2_0 = approx_exp2(acc_scale_log2);
                        acc_scale = ((row_max > -CUDART_INF_F) ? _exp2_0 : 1.0f);
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
                    float2 block_sum2 = make_float2(0.0f, 0.0f);
                    float sf_values[4];
                    {
                        float block_max = tile_max;
                        float block_max_scaled = ((block_max > -CUDART_INF_F) ? block_max * softmax_scale_log2 : 0.0f);
                        float _exp2_1 = approx_exp2(block_max_scaled - new_max_scaled - 2.584962500721156f);
                        float p_scale = ((block_max > -CUDART_INF_F) ? _exp2_1 : 0.0f);
                        sf_values[0] = p_scale;
                        sf_values[1] = p_scale;
                        sf_values[2] = p_scale;
                        sf_values[3] = p_scale;
                        const float2 _fma_b2_2 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_3 = {2.584962500721156f - block_max_scaled, 2.584962500721156f - block_max_scaled};
                        float2 _fma_pair_4 = fma_f32x2(make_float2(((sv + 0))[0], ((sv + 0))[1]), _fma_b2_2, _fma_c2_3);
                        (sv + 0)[0] = _fma_pair_4.x;
                        (sv + 0)[1] = _fma_pair_4.y;
                        float2 _fma_pair_5 = fma_f32x2(make_float2(((sv + 0))[2], ((sv + 0))[3]), _fma_b2_2, _fma_c2_3);
                        (sv + 0)[2] = _fma_pair_5.x;
                        (sv + 0)[3] = _fma_pair_5.y;
                        float2 _fma_pair_6 = fma_f32x2(make_float2(((sv + 0))[4], ((sv + 0))[5]), _fma_b2_2, _fma_c2_3);
                        (sv + 0)[4] = _fma_pair_6.x;
                        (sv + 0)[5] = _fma_pair_6.y;
                        float2 _fma_pair_7 = fma_f32x2(make_float2(((sv + 0))[6], ((sv + 0))[7]), _fma_b2_2, _fma_c2_3);
                        (sv + 0)[6] = _fma_pair_7.x;
                        (sv + 0)[7] = _fma_pair_7.y;
                        float2 _fma_pair_8 = fma_f32x2(make_float2(((sv + 0))[8], ((sv + 0))[9]), _fma_b2_2, _fma_c2_3);
                        (sv + 0)[8] = _fma_pair_8.x;
                        (sv + 0)[9] = _fma_pair_8.y;
                        float2 _fma_pair_9 = fma_f32x2(make_float2(((sv + 0))[10], ((sv + 0))[11]), _fma_b2_2, _fma_c2_3);
                        (sv + 0)[10] = _fma_pair_9.x;
                        (sv + 0)[11] = _fma_pair_9.y;
                        float2 _fma_pair_10 = fma_f32x2(make_float2(((sv + 0))[12], ((sv + 0))[13]), _fma_b2_2, _fma_c2_3);
                        (sv + 0)[12] = _fma_pair_10.x;
                        (sv + 0)[13] = _fma_pair_10.y;
                        float2 _fma_pair_11 = fma_f32x2(make_float2(((sv + 0))[14], ((sv + 0))[15]), _fma_b2_2, _fma_c2_3);
                        (sv + 0)[14] = _fma_pair_11.x;
                        (sv + 0)[15] = _fma_pair_11.y;
                        float2 _fma_pair_12 = fma_f32x2(make_float2(((sv + 0))[16], ((sv + 0))[17]), _fma_b2_2, _fma_c2_3);
                        (sv + 0)[16] = _fma_pair_12.x;
                        (sv + 0)[17] = _fma_pair_12.y;
                        float2 _fma_pair_13 = fma_f32x2(make_float2(((sv + 0))[18], ((sv + 0))[19]), _fma_b2_2, _fma_c2_3);
                        (sv + 0)[18] = _fma_pair_13.x;
                        (sv + 0)[19] = _fma_pair_13.y;
                        float2 _fma_pair_14 = fma_f32x2(make_float2(((sv + 0))[20], ((sv + 0))[21]), _fma_b2_2, _fma_c2_3);
                        (sv + 0)[20] = _fma_pair_14.x;
                        (sv + 0)[21] = _fma_pair_14.y;
                        float2 _fma_pair_15 = fma_f32x2(make_float2(((sv + 0))[22], ((sv + 0))[23]), _fma_b2_2, _fma_c2_3);
                        (sv + 0)[22] = _fma_pair_15.x;
                        (sv + 0)[23] = _fma_pair_15.y;
                        float2 _fma_pair_16 = fma_f32x2(make_float2(((sv + 0))[24], ((sv + 0))[25]), _fma_b2_2, _fma_c2_3);
                        (sv + 0)[24] = _fma_pair_16.x;
                        (sv + 0)[25] = _fma_pair_16.y;
                        float2 _fma_pair_17 = fma_f32x2(make_float2(((sv + 0))[26], ((sv + 0))[27]), _fma_b2_2, _fma_c2_3);
                        (sv + 0)[26] = _fma_pair_17.x;
                        (sv + 0)[27] = _fma_pair_17.y;
                        float2 _fma_pair_18 = fma_f32x2(make_float2(((sv + 0))[28], ((sv + 0))[29]), _fma_b2_2, _fma_c2_3);
                        (sv + 0)[28] = _fma_pair_18.x;
                        (sv + 0)[29] = _fma_pair_18.y;
                        float2 _fma_pair_19 = fma_f32x2(make_float2(((sv + 0))[30], ((sv + 0))[31]), _fma_b2_2, _fma_c2_3);
                        (sv + 0)[30] = _fma_pair_19.x;
                        (sv + 0)[31] = _fma_pair_19.y;
                        #pragma unroll
                        for (int _le = 0; _le < 32; _le++) {
                            sv[_le] = approx_exp2(sv[_le]);
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
                        unsigned int _fp4_12[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_12[0]) : "f"(sv[0]), "f"(sv[1]), "f"(sv[2]), "f"(sv[3]), "f"(sv[4]), "f"(sv[5]), "f"(sv[6]), "f"(sv[7]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_12[1]) : "f"(sv[8]), "f"(sv[9]), "f"(sv[10]), "f"(sv[11]), "f"(sv[12]), "f"(sv[13]), "f"(sv[14]), "f"(sv[15]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_12[2]) : "f"(sv[16]), "f"(sv[17]), "f"(sv[18]), "f"(sv[19]), "f"(sv[20]), "f"(sv[21]), "f"(sv[22]), "f"(sv[23]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_12[3]) : "f"(sv[24]), "f"(sv[25]), "f"(sv[26]), "f"(sv[27]), "f"(sv[28]), "f"(sv[29]), "f"(sv[30]), "f"(sv[31]));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_p_addr + stage * 8192 + (unsigned int)((warp % 4 * 32 + lane) * 64 ^ ((warp % 4 * 32 + lane) * 64 >> 7 & 3) << 4))), "r"(_fp4_12[0]), "r"(_fp4_12[1]), "r"(_fp4_12[2]), "r"(_fp4_12[3]) : "memory");
                        const float2 _fma_b2_20 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_21 = {2.584962500721156f - block_max_scaled, 2.584962500721156f - block_max_scaled};
                        float2 _fma_pair_22 = fma_f32x2(make_float2(((sv + 32))[0], ((sv + 32))[1]), _fma_b2_20, _fma_c2_21);
                        (sv + 32)[0] = _fma_pair_22.x;
                        (sv + 32)[1] = _fma_pair_22.y;
                        float2 _fma_pair_23 = fma_f32x2(make_float2(((sv + 32))[2], ((sv + 32))[3]), _fma_b2_20, _fma_c2_21);
                        (sv + 32)[2] = _fma_pair_23.x;
                        (sv + 32)[3] = _fma_pair_23.y;
                        float2 _fma_pair_24 = fma_f32x2(make_float2(((sv + 32))[4], ((sv + 32))[5]), _fma_b2_20, _fma_c2_21);
                        (sv + 32)[4] = _fma_pair_24.x;
                        (sv + 32)[5] = _fma_pair_24.y;
                        float2 _fma_pair_25 = fma_f32x2(make_float2(((sv + 32))[6], ((sv + 32))[7]), _fma_b2_20, _fma_c2_21);
                        (sv + 32)[6] = _fma_pair_25.x;
                        (sv + 32)[7] = _fma_pair_25.y;
                        float2 _fma_pair_26 = fma_f32x2(make_float2(((sv + 32))[8], ((sv + 32))[9]), _fma_b2_20, _fma_c2_21);
                        (sv + 32)[8] = _fma_pair_26.x;
                        (sv + 32)[9] = _fma_pair_26.y;
                        float2 _fma_pair_27 = fma_f32x2(make_float2(((sv + 32))[10], ((sv + 32))[11]), _fma_b2_20, _fma_c2_21);
                        (sv + 32)[10] = _fma_pair_27.x;
                        (sv + 32)[11] = _fma_pair_27.y;
                        float2 _fma_pair_28 = fma_f32x2(make_float2(((sv + 32))[12], ((sv + 32))[13]), _fma_b2_20, _fma_c2_21);
                        (sv + 32)[12] = _fma_pair_28.x;
                        (sv + 32)[13] = _fma_pair_28.y;
                        float2 _fma_pair_29 = fma_f32x2(make_float2(((sv + 32))[14], ((sv + 32))[15]), _fma_b2_20, _fma_c2_21);
                        (sv + 32)[14] = _fma_pair_29.x;
                        (sv + 32)[15] = _fma_pair_29.y;
                        float2 _fma_pair_30 = fma_f32x2(make_float2(((sv + 32))[16], ((sv + 32))[17]), _fma_b2_20, _fma_c2_21);
                        (sv + 32)[16] = _fma_pair_30.x;
                        (sv + 32)[17] = _fma_pair_30.y;
                        float2 _fma_pair_31 = fma_f32x2(make_float2(((sv + 32))[18], ((sv + 32))[19]), _fma_b2_20, _fma_c2_21);
                        (sv + 32)[18] = _fma_pair_31.x;
                        (sv + 32)[19] = _fma_pair_31.y;
                        float2 _fma_pair_32 = fma_f32x2(make_float2(((sv + 32))[20], ((sv + 32))[21]), _fma_b2_20, _fma_c2_21);
                        (sv + 32)[20] = _fma_pair_32.x;
                        (sv + 32)[21] = _fma_pair_32.y;
                        float2 _fma_pair_33 = fma_f32x2(make_float2(((sv + 32))[22], ((sv + 32))[23]), _fma_b2_20, _fma_c2_21);
                        (sv + 32)[22] = _fma_pair_33.x;
                        (sv + 32)[23] = _fma_pair_33.y;
                        float2 _fma_pair_34 = fma_f32x2(make_float2(((sv + 32))[24], ((sv + 32))[25]), _fma_b2_20, _fma_c2_21);
                        (sv + 32)[24] = _fma_pair_34.x;
                        (sv + 32)[25] = _fma_pair_34.y;
                        float2 _fma_pair_35 = fma_f32x2(make_float2(((sv + 32))[26], ((sv + 32))[27]), _fma_b2_20, _fma_c2_21);
                        (sv + 32)[26] = _fma_pair_35.x;
                        (sv + 32)[27] = _fma_pair_35.y;
                        float2 _fma_pair_36 = fma_f32x2(make_float2(((sv + 32))[28], ((sv + 32))[29]), _fma_b2_20, _fma_c2_21);
                        (sv + 32)[28] = _fma_pair_36.x;
                        (sv + 32)[29] = _fma_pair_36.y;
                        float2 _fma_pair_37 = fma_f32x2(make_float2(((sv + 32))[30], ((sv + 32))[31]), _fma_b2_20, _fma_c2_21);
                        (sv + 32)[30] = _fma_pair_37.x;
                        (sv + 32)[31] = _fma_pair_37.y;
                        #pragma unroll
                        for (int _le = 0; _le < 32; _le++) {
                            sv[_le + 32] = approx_exp2(sv[_le + 32]);
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
                        unsigned int _fp4_13[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_13[0]) : "f"(sv[32]), "f"(sv[33]), "f"(sv[34]), "f"(sv[35]), "f"(sv[36]), "f"(sv[37]), "f"(sv[38]), "f"(sv[39]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_13[1]) : "f"(sv[40]), "f"(sv[41]), "f"(sv[42]), "f"(sv[43]), "f"(sv[44]), "f"(sv[45]), "f"(sv[46]), "f"(sv[47]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_13[2]) : "f"(sv[48]), "f"(sv[49]), "f"(sv[50]), "f"(sv[51]), "f"(sv[52]), "f"(sv[53]), "f"(sv[54]), "f"(sv[55]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_13[3]) : "f"(sv[56]), "f"(sv[57]), "f"(sv[58]), "f"(sv[59]), "f"(sv[60]), "f"(sv[61]), "f"(sv[62]), "f"(sv[63]));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_p_addr + stage * 8192 + (unsigned int)((warp % 4 * 32 + lane) * 64 + 16 ^ ((warp % 4 * 32 + lane) * 64 + 16 >> 7 & 3) << 4))), "r"(_fp4_13[0]), "r"(_fp4_13[1]), "r"(_fp4_13[2]), "r"(_fp4_13[3]) : "memory");
                        unsigned int _fp8_2[1];
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
                            _fp8_2[0] = _packed;
                        }
                        int pv_sfa_lo_col = (unsigned int)408 + stage * 8 + (unsigned int)(warp % 4);
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(taddr + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)pv_sfa_lo_col), "r"(_fp8_2[0]));
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((p_full_addr + (stage) * 8) & 0xFEFFFFFF) : "memory");
                        const float2 _fma_b2_38 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_39 = {2.584962500721156f - block_max_scaled, 2.584962500721156f - block_max_scaled};
                        float2 _fma_pair_40 = fma_f32x2(make_float2(((sv + 64))[0], ((sv + 64))[1]), _fma_b2_38, _fma_c2_39);
                        (sv + 64)[0] = _fma_pair_40.x;
                        (sv + 64)[1] = _fma_pair_40.y;
                        float2 _fma_pair_41 = fma_f32x2(make_float2(((sv + 64))[2], ((sv + 64))[3]), _fma_b2_38, _fma_c2_39);
                        (sv + 64)[2] = _fma_pair_41.x;
                        (sv + 64)[3] = _fma_pair_41.y;
                        float2 _fma_pair_42 = fma_f32x2(make_float2(((sv + 64))[4], ((sv + 64))[5]), _fma_b2_38, _fma_c2_39);
                        (sv + 64)[4] = _fma_pair_42.x;
                        (sv + 64)[5] = _fma_pair_42.y;
                        float2 _fma_pair_43 = fma_f32x2(make_float2(((sv + 64))[6], ((sv + 64))[7]), _fma_b2_38, _fma_c2_39);
                        (sv + 64)[6] = _fma_pair_43.x;
                        (sv + 64)[7] = _fma_pair_43.y;
                        float2 _fma_pair_44 = fma_f32x2(make_float2(((sv + 64))[8], ((sv + 64))[9]), _fma_b2_38, _fma_c2_39);
                        (sv + 64)[8] = _fma_pair_44.x;
                        (sv + 64)[9] = _fma_pair_44.y;
                        float2 _fma_pair_45 = fma_f32x2(make_float2(((sv + 64))[10], ((sv + 64))[11]), _fma_b2_38, _fma_c2_39);
                        (sv + 64)[10] = _fma_pair_45.x;
                        (sv + 64)[11] = _fma_pair_45.y;
                        float2 _fma_pair_46 = fma_f32x2(make_float2(((sv + 64))[12], ((sv + 64))[13]), _fma_b2_38, _fma_c2_39);
                        (sv + 64)[12] = _fma_pair_46.x;
                        (sv + 64)[13] = _fma_pair_46.y;
                        float2 _fma_pair_47 = fma_f32x2(make_float2(((sv + 64))[14], ((sv + 64))[15]), _fma_b2_38, _fma_c2_39);
                        (sv + 64)[14] = _fma_pair_47.x;
                        (sv + 64)[15] = _fma_pair_47.y;
                        float2 _fma_pair_48 = fma_f32x2(make_float2(((sv + 64))[16], ((sv + 64))[17]), _fma_b2_38, _fma_c2_39);
                        (sv + 64)[16] = _fma_pair_48.x;
                        (sv + 64)[17] = _fma_pair_48.y;
                        float2 _fma_pair_49 = fma_f32x2(make_float2(((sv + 64))[18], ((sv + 64))[19]), _fma_b2_38, _fma_c2_39);
                        (sv + 64)[18] = _fma_pair_49.x;
                        (sv + 64)[19] = _fma_pair_49.y;
                        float2 _fma_pair_50 = fma_f32x2(make_float2(((sv + 64))[20], ((sv + 64))[21]), _fma_b2_38, _fma_c2_39);
                        (sv + 64)[20] = _fma_pair_50.x;
                        (sv + 64)[21] = _fma_pair_50.y;
                        float2 _fma_pair_51 = fma_f32x2(make_float2(((sv + 64))[22], ((sv + 64))[23]), _fma_b2_38, _fma_c2_39);
                        (sv + 64)[22] = _fma_pair_51.x;
                        (sv + 64)[23] = _fma_pair_51.y;
                        float2 _fma_pair_52 = fma_f32x2(make_float2(((sv + 64))[24], ((sv + 64))[25]), _fma_b2_38, _fma_c2_39);
                        (sv + 64)[24] = _fma_pair_52.x;
                        (sv + 64)[25] = _fma_pair_52.y;
                        float2 _fma_pair_53 = fma_f32x2(make_float2(((sv + 64))[26], ((sv + 64))[27]), _fma_b2_38, _fma_c2_39);
                        (sv + 64)[26] = _fma_pair_53.x;
                        (sv + 64)[27] = _fma_pair_53.y;
                        float2 _fma_pair_54 = fma_f32x2(make_float2(((sv + 64))[28], ((sv + 64))[29]), _fma_b2_38, _fma_c2_39);
                        (sv + 64)[28] = _fma_pair_54.x;
                        (sv + 64)[29] = _fma_pair_54.y;
                        float2 _fma_pair_55 = fma_f32x2(make_float2(((sv + 64))[30], ((sv + 64))[31]), _fma_b2_38, _fma_c2_39);
                        (sv + 64)[30] = _fma_pair_55.x;
                        (sv + 64)[31] = _fma_pair_55.y;
                        #pragma unroll
                        for (int _le = 0; _le < 32; _le++) {
                            sv[_le + 64] = approx_exp2(sv[_le + 64]);
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
                        unsigned int _fp4_14[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_14[0]) : "f"(sv[64]), "f"(sv[65]), "f"(sv[66]), "f"(sv[67]), "f"(sv[68]), "f"(sv[69]), "f"(sv[70]), "f"(sv[71]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_14[1]) : "f"(sv[72]), "f"(sv[73]), "f"(sv[74]), "f"(sv[75]), "f"(sv[76]), "f"(sv[77]), "f"(sv[78]), "f"(sv[79]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_14[2]) : "f"(sv[80]), "f"(sv[81]), "f"(sv[82]), "f"(sv[83]), "f"(sv[84]), "f"(sv[85]), "f"(sv[86]), "f"(sv[87]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_14[3]) : "f"(sv[88]), "f"(sv[89]), "f"(sv[90]), "f"(sv[91]), "f"(sv[92]), "f"(sv[93]), "f"(sv[94]), "f"(sv[95]));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_p_addr + stage * 8192 + (unsigned int)((warp % 4 * 32 + lane) * 64 + 32 ^ ((warp % 4 * 32 + lane) * 64 + 32 >> 7 & 3) << 4))), "r"(_fp4_14[0]), "r"(_fp4_14[1]), "r"(_fp4_14[2]), "r"(_fp4_14[3]) : "memory");
                        const float2 _fma_b2_56 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_57 = {2.584962500721156f - block_max_scaled, 2.584962500721156f - block_max_scaled};
                        float2 _fma_pair_58 = fma_f32x2(make_float2(((sv + 96))[0], ((sv + 96))[1]), _fma_b2_56, _fma_c2_57);
                        (sv + 96)[0] = _fma_pair_58.x;
                        (sv + 96)[1] = _fma_pair_58.y;
                        float2 _fma_pair_59 = fma_f32x2(make_float2(((sv + 96))[2], ((sv + 96))[3]), _fma_b2_56, _fma_c2_57);
                        (sv + 96)[2] = _fma_pair_59.x;
                        (sv + 96)[3] = _fma_pair_59.y;
                        float2 _fma_pair_60 = fma_f32x2(make_float2(((sv + 96))[4], ((sv + 96))[5]), _fma_b2_56, _fma_c2_57);
                        (sv + 96)[4] = _fma_pair_60.x;
                        (sv + 96)[5] = _fma_pair_60.y;
                        float2 _fma_pair_61 = fma_f32x2(make_float2(((sv + 96))[6], ((sv + 96))[7]), _fma_b2_56, _fma_c2_57);
                        (sv + 96)[6] = _fma_pair_61.x;
                        (sv + 96)[7] = _fma_pair_61.y;
                        float2 _fma_pair_62 = fma_f32x2(make_float2(((sv + 96))[8], ((sv + 96))[9]), _fma_b2_56, _fma_c2_57);
                        (sv + 96)[8] = _fma_pair_62.x;
                        (sv + 96)[9] = _fma_pair_62.y;
                        float2 _fma_pair_63 = fma_f32x2(make_float2(((sv + 96))[10], ((sv + 96))[11]), _fma_b2_56, _fma_c2_57);
                        (sv + 96)[10] = _fma_pair_63.x;
                        (sv + 96)[11] = _fma_pair_63.y;
                        float2 _fma_pair_64 = fma_f32x2(make_float2(((sv + 96))[12], ((sv + 96))[13]), _fma_b2_56, _fma_c2_57);
                        (sv + 96)[12] = _fma_pair_64.x;
                        (sv + 96)[13] = _fma_pair_64.y;
                        float2 _fma_pair_65 = fma_f32x2(make_float2(((sv + 96))[14], ((sv + 96))[15]), _fma_b2_56, _fma_c2_57);
                        (sv + 96)[14] = _fma_pair_65.x;
                        (sv + 96)[15] = _fma_pair_65.y;
                        float2 _fma_pair_66 = fma_f32x2(make_float2(((sv + 96))[16], ((sv + 96))[17]), _fma_b2_56, _fma_c2_57);
                        (sv + 96)[16] = _fma_pair_66.x;
                        (sv + 96)[17] = _fma_pair_66.y;
                        float2 _fma_pair_67 = fma_f32x2(make_float2(((sv + 96))[18], ((sv + 96))[19]), _fma_b2_56, _fma_c2_57);
                        (sv + 96)[18] = _fma_pair_67.x;
                        (sv + 96)[19] = _fma_pair_67.y;
                        float2 _fma_pair_68 = fma_f32x2(make_float2(((sv + 96))[20], ((sv + 96))[21]), _fma_b2_56, _fma_c2_57);
                        (sv + 96)[20] = _fma_pair_68.x;
                        (sv + 96)[21] = _fma_pair_68.y;
                        float2 _fma_pair_69 = fma_f32x2(make_float2(((sv + 96))[22], ((sv + 96))[23]), _fma_b2_56, _fma_c2_57);
                        (sv + 96)[22] = _fma_pair_69.x;
                        (sv + 96)[23] = _fma_pair_69.y;
                        float2 _fma_pair_70 = fma_f32x2(make_float2(((sv + 96))[24], ((sv + 96))[25]), _fma_b2_56, _fma_c2_57);
                        (sv + 96)[24] = _fma_pair_70.x;
                        (sv + 96)[25] = _fma_pair_70.y;
                        float2 _fma_pair_71 = fma_f32x2(make_float2(((sv + 96))[26], ((sv + 96))[27]), _fma_b2_56, _fma_c2_57);
                        (sv + 96)[26] = _fma_pair_71.x;
                        (sv + 96)[27] = _fma_pair_71.y;
                        float2 _fma_pair_72 = fma_f32x2(make_float2(((sv + 96))[28], ((sv + 96))[29]), _fma_b2_56, _fma_c2_57);
                        (sv + 96)[28] = _fma_pair_72.x;
                        (sv + 96)[29] = _fma_pair_72.y;
                        float2 _fma_pair_73 = fma_f32x2(make_float2(((sv + 96))[30], ((sv + 96))[31]), _fma_b2_56, _fma_c2_57);
                        (sv + 96)[30] = _fma_pair_73.x;
                        (sv + 96)[31] = _fma_pair_73.y;
                        #pragma unroll
                        for (int _le = 0; _le < 32; _le++) {
                            sv[_le + 96] = approx_exp2(sv[_le + 96]);
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
                        unsigned int _fp4_15[4];
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_15[0]) : "f"(sv[96]), "f"(sv[97]), "f"(sv[98]), "f"(sv[99]), "f"(sv[100]), "f"(sv[101]), "f"(sv[102]), "f"(sv[103]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_15[1]) : "f"(sv[104]), "f"(sv[105]), "f"(sv[106]), "f"(sv[107]), "f"(sv[108]), "f"(sv[109]), "f"(sv[110]), "f"(sv[111]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_15[2]) : "f"(sv[112]), "f"(sv[113]), "f"(sv[114]), "f"(sv[115]), "f"(sv[116]), "f"(sv[117]), "f"(sv[118]), "f"(sv[119]));
                        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_15[3]) : "f"(sv[120]), "f"(sv[121]), "f"(sv[122]), "f"(sv[123]), "f"(sv[124]), "f"(sv[125]), "f"(sv[126]), "f"(sv[127]));
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_p_addr + stage * 8192 + (unsigned int)((warp % 4 * 32 + lane) * 64 + 48 ^ ((warp % 4 * 32 + lane) * 64 + 48 >> 7 & 3) << 4))), "r"(_fp4_15[0]), "r"(_fp4_15[1]), "r"(_fp4_15[2]), "r"(_fp4_15[3]) : "memory");
                        int pv_sfa_hi_col = (unsigned int)412 + stage * 8 + (unsigned int)(warp % 4);
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(taddr + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)pv_sfa_hi_col), "r"(_fp8_2[0]));
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                            :: "r"((p_full_2_addr + (stage) * 8) & 0xFEFFFFFF) : "memory");
                    }
                    float block_sum = block_sum2.x + block_sum2.y;
                    row_sum = __fmaf_rn(row_sum, acc_scale, block_sum);
                    mbarrier_wait(corr_done_addr + (stage) * 8, _phase_corr_done);
                    _phase_corr_done ^= 1;
                }
                mbarrier_wait(o_full_addr + (stage) * 8, _phase_o_full);
                _phase_o_full ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float _rcp_74 = approx_rcp(row_sum);
                float final_scale = ((row_sum != 0.0f && row_sum == row_sum) ? _rcp_74 : 0.0f);
                if (stage == 0) {
                    {
                        if (warp == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                        }
                        asm volatile("barrier.sync 3, 128;" ::: "memory");
                        asm volatile("tcgen05.fence::after_thread_sync;");
                    }
                    {
                        #pragma unroll
                        for (int col = 0; col < HEAD_DIM / 128; col++) {
                            int addr = taddr + (unsigned int)128 + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(col * 64);
                            float _tmem_load_8[64];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(_tmem_load_8[0]), "=f"(_tmem_load_8[1]), "=f"(_tmem_load_8[2]), "=f"(_tmem_load_8[3]), "=f"(_tmem_load_8[4]), "=f"(_tmem_load_8[5]), "=f"(_tmem_load_8[6]), "=f"(_tmem_load_8[7]), "=f"(_tmem_load_8[8]), "=f"(_tmem_load_8[9]), "=f"(_tmem_load_8[10]), "=f"(_tmem_load_8[11]), "=f"(_tmem_load_8[12]), "=f"(_tmem_load_8[13]), "=f"(_tmem_load_8[14]), "=f"(_tmem_load_8[15]), "=f"(_tmem_load_8[16]), "=f"(_tmem_load_8[17]), "=f"(_tmem_load_8[18]), "=f"(_tmem_load_8[19]), "=f"(_tmem_load_8[20]), "=f"(_tmem_load_8[21]), "=f"(_tmem_load_8[22]), "=f"(_tmem_load_8[23]), "=f"(_tmem_load_8[24]), "=f"(_tmem_load_8[25]), "=f"(_tmem_load_8[26]), "=f"(_tmem_load_8[27]), "=f"(_tmem_load_8[28]), "=f"(_tmem_load_8[29]), "=f"(_tmem_load_8[30]), "=f"(_tmem_load_8[31])
                                : "r"(addr));
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(_tmem_load_8[32]), "=f"(_tmem_load_8[33]), "=f"(_tmem_load_8[34]), "=f"(_tmem_load_8[35]), "=f"(_tmem_load_8[36]), "=f"(_tmem_load_8[37]), "=f"(_tmem_load_8[38]), "=f"(_tmem_load_8[39]), "=f"(_tmem_load_8[40]), "=f"(_tmem_load_8[41]), "=f"(_tmem_load_8[42]), "=f"(_tmem_load_8[43]), "=f"(_tmem_load_8[44]), "=f"(_tmem_load_8[45]), "=f"(_tmem_load_8[46]), "=f"(_tmem_load_8[47]), "=f"(_tmem_load_8[48]), "=f"(_tmem_load_8[49]), "=f"(_tmem_load_8[50]), "=f"(_tmem_load_8[51]), "=f"(_tmem_load_8[52]), "=f"(_tmem_load_8[53]), "=f"(_tmem_load_8[54]), "=f"(_tmem_load_8[55]), "=f"(_tmem_load_8[56]), "=f"(_tmem_load_8[57]), "=f"(_tmem_load_8[58]), "=f"(_tmem_load_8[59]), "=f"(_tmem_load_8[60]), "=f"(_tmem_load_8[61]), "=f"(_tmem_load_8[62]), "=f"(_tmem_load_8[63])
                                : "r"(addr + 32));
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            #if __CUDA_ARCH__ >= 1000
                            const float2 _scale2_75 = {final_scale, final_scale};
                            #pragma unroll
                            for (int _ls = 0; _ls < 32; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_8)[_ls], _scale2_75);
                            #else
                            #pragma unroll
                            for (int _ls = 0; _ls < 64; _ls++) {
                                _tmem_load_8[_ls] = _tmem_load_8[_ls] * final_scale;
                            }
                            #endif
                            unsigned int _tmem_load_8_bf16[32];
                            #pragma unroll
                            for (int _lp = 0; _lp < 32; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_8[_lp*2 + 0], _tmem_load_8[_lp*2+1 + 0]));
                                _tmem_load_8_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o00_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + col * 128 ^ ((warp % 4 * 32 + lane) * 128 + col * 128 >> 7 & 7) << 4))), "r"(_tmem_load_8_bf16[0]), "r"(_tmem_load_8_bf16[1]), "r"(_tmem_load_8_bf16[2]), "r"(_tmem_load_8_bf16[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o00_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + (col * 128 + 16) ^ ((warp % 4 * 32 + lane) * 128 + (col * 128 + 16) >> 7 & 7) << 4))), "r"(_tmem_load_8_bf16[4]), "r"(_tmem_load_8_bf16[5]), "r"(_tmem_load_8_bf16[6]), "r"(_tmem_load_8_bf16[7]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o00_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + (col * 128 + 32) ^ ((warp % 4 * 32 + lane) * 128 + (col * 128 + 32) >> 7 & 7) << 4))), "r"(_tmem_load_8_bf16[8]), "r"(_tmem_load_8_bf16[9]), "r"(_tmem_load_8_bf16[10]), "r"(_tmem_load_8_bf16[11]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o00_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + (col * 128 + 48) ^ ((warp % 4 * 32 + lane) * 128 + (col * 128 + 48) >> 7 & 7) << 4))), "r"(_tmem_load_8_bf16[12]), "r"(_tmem_load_8_bf16[13]), "r"(_tmem_load_8_bf16[14]), "r"(_tmem_load_8_bf16[15]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o00_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + (col * 128 + 64) ^ ((warp % 4 * 32 + lane) * 128 + (col * 128 + 64) >> 7 & 7) << 4))), "r"(_tmem_load_8_bf16[16]), "r"(_tmem_load_8_bf16[17]), "r"(_tmem_load_8_bf16[18]), "r"(_tmem_load_8_bf16[19]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o00_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + (col * 128 + 80) ^ ((warp % 4 * 32 + lane) * 128 + (col * 128 + 80) >> 7 & 7) << 4))), "r"(_tmem_load_8_bf16[20]), "r"(_tmem_load_8_bf16[21]), "r"(_tmem_load_8_bf16[22]), "r"(_tmem_load_8_bf16[23]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o00_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + (col * 128 + 96) ^ ((warp % 4 * 32 + lane) * 128 + (col * 128 + 96) >> 7 & 7) << 4))), "r"(_tmem_load_8_bf16[24]), "r"(_tmem_load_8_bf16[25]), "r"(_tmem_load_8_bf16[26]), "r"(_tmem_load_8_bf16[27]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o00_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + (col * 128 + 112) ^ ((warp % 4 * 32 + lane) * 128 + (col * 128 + 112) >> 7 & 7) << 4))), "r"(_tmem_load_8_bf16[28]), "r"(_tmem_load_8_bf16[29]), "r"(_tmem_load_8_bf16[30]), "r"(_tmem_load_8_bf16[31]) : "memory");
                        }
                        #pragma unroll
                        for (int col_1 = 0; col_1 < HEAD_DIM / 128; col_1++) {
                            int addr_1 = taddr + (unsigned int)128 + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(HEAD_DIM / 2) + (unsigned int)(col_1 * 64);
                            float _tmem_load_9[64];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(_tmem_load_9[0]), "=f"(_tmem_load_9[1]), "=f"(_tmem_load_9[2]), "=f"(_tmem_load_9[3]), "=f"(_tmem_load_9[4]), "=f"(_tmem_load_9[5]), "=f"(_tmem_load_9[6]), "=f"(_tmem_load_9[7]), "=f"(_tmem_load_9[8]), "=f"(_tmem_load_9[9]), "=f"(_tmem_load_9[10]), "=f"(_tmem_load_9[11]), "=f"(_tmem_load_9[12]), "=f"(_tmem_load_9[13]), "=f"(_tmem_load_9[14]), "=f"(_tmem_load_9[15]), "=f"(_tmem_load_9[16]), "=f"(_tmem_load_9[17]), "=f"(_tmem_load_9[18]), "=f"(_tmem_load_9[19]), "=f"(_tmem_load_9[20]), "=f"(_tmem_load_9[21]), "=f"(_tmem_load_9[22]), "=f"(_tmem_load_9[23]), "=f"(_tmem_load_9[24]), "=f"(_tmem_load_9[25]), "=f"(_tmem_load_9[26]), "=f"(_tmem_load_9[27]), "=f"(_tmem_load_9[28]), "=f"(_tmem_load_9[29]), "=f"(_tmem_load_9[30]), "=f"(_tmem_load_9[31])
                                : "r"(addr_1));
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(_tmem_load_9[32]), "=f"(_tmem_load_9[33]), "=f"(_tmem_load_9[34]), "=f"(_tmem_load_9[35]), "=f"(_tmem_load_9[36]), "=f"(_tmem_load_9[37]), "=f"(_tmem_load_9[38]), "=f"(_tmem_load_9[39]), "=f"(_tmem_load_9[40]), "=f"(_tmem_load_9[41]), "=f"(_tmem_load_9[42]), "=f"(_tmem_load_9[43]), "=f"(_tmem_load_9[44]), "=f"(_tmem_load_9[45]), "=f"(_tmem_load_9[46]), "=f"(_tmem_load_9[47]), "=f"(_tmem_load_9[48]), "=f"(_tmem_load_9[49]), "=f"(_tmem_load_9[50]), "=f"(_tmem_load_9[51]), "=f"(_tmem_load_9[52]), "=f"(_tmem_load_9[53]), "=f"(_tmem_load_9[54]), "=f"(_tmem_load_9[55]), "=f"(_tmem_load_9[56]), "=f"(_tmem_load_9[57]), "=f"(_tmem_load_9[58]), "=f"(_tmem_load_9[59]), "=f"(_tmem_load_9[60]), "=f"(_tmem_load_9[61]), "=f"(_tmem_load_9[62]), "=f"(_tmem_load_9[63])
                                : "r"(addr_1 + 32));
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            #if __CUDA_ARCH__ >= 1000
                            const float2 _scale2_76 = {final_scale, final_scale};
                            #pragma unroll
                            for (int _ls = 0; _ls < 32; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_9)[_ls], _scale2_76);
                            #else
                            #pragma unroll
                            for (int _ls = 0; _ls < 64; _ls++) {
                                _tmem_load_9[_ls] = _tmem_load_9[_ls] * final_scale;
                            }
                            #endif
                            unsigned int _tmem_load_9_bf16[32];
                            #pragma unroll
                            for (int _lp = 0; _lp < 32; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_9[_lp*2 + 0], _tmem_load_9[_lp*2+1 + 0]));
                                _tmem_load_9_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o01_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + col_1 * 128 ^ ((warp % 4 * 32 + lane) * 128 + col_1 * 128 >> 7 & 7) << 4))), "r"(_tmem_load_9_bf16[0]), "r"(_tmem_load_9_bf16[1]), "r"(_tmem_load_9_bf16[2]), "r"(_tmem_load_9_bf16[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o01_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + (col_1 * 128 + 16) ^ ((warp % 4 * 32 + lane) * 128 + (col_1 * 128 + 16) >> 7 & 7) << 4))), "r"(_tmem_load_9_bf16[4]), "r"(_tmem_load_9_bf16[5]), "r"(_tmem_load_9_bf16[6]), "r"(_tmem_load_9_bf16[7]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o01_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + (col_1 * 128 + 32) ^ ((warp % 4 * 32 + lane) * 128 + (col_1 * 128 + 32) >> 7 & 7) << 4))), "r"(_tmem_load_9_bf16[8]), "r"(_tmem_load_9_bf16[9]), "r"(_tmem_load_9_bf16[10]), "r"(_tmem_load_9_bf16[11]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o01_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + (col_1 * 128 + 48) ^ ((warp % 4 * 32 + lane) * 128 + (col_1 * 128 + 48) >> 7 & 7) << 4))), "r"(_tmem_load_9_bf16[12]), "r"(_tmem_load_9_bf16[13]), "r"(_tmem_load_9_bf16[14]), "r"(_tmem_load_9_bf16[15]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o01_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + (col_1 * 128 + 64) ^ ((warp % 4 * 32 + lane) * 128 + (col_1 * 128 + 64) >> 7 & 7) << 4))), "r"(_tmem_load_9_bf16[16]), "r"(_tmem_load_9_bf16[17]), "r"(_tmem_load_9_bf16[18]), "r"(_tmem_load_9_bf16[19]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o01_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + (col_1 * 128 + 80) ^ ((warp % 4 * 32 + lane) * 128 + (col_1 * 128 + 80) >> 7 & 7) << 4))), "r"(_tmem_load_9_bf16[20]), "r"(_tmem_load_9_bf16[21]), "r"(_tmem_load_9_bf16[22]), "r"(_tmem_load_9_bf16[23]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o01_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + (col_1 * 128 + 96) ^ ((warp % 4 * 32 + lane) * 128 + (col_1 * 128 + 96) >> 7 & 7) << 4))), "r"(_tmem_load_9_bf16[24]), "r"(_tmem_load_9_bf16[25]), "r"(_tmem_load_9_bf16[26]), "r"(_tmem_load_9_bf16[27]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o01_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + (col_1 * 128 + 112) ^ ((warp % 4 * 32 + lane) * 128 + (col_1 * 128 + 112) >> 7 & 7) << 4))), "r"(_tmem_load_9_bf16[28]), "r"(_tmem_load_9_bf16[29]), "r"(_tmem_load_9_bf16[30]), "r"(_tmem_load_9_bf16[31]) : "memory");
                        }
                    }
                    asm volatile("fence.proxy.async;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                    int out_row0 = bh * (unsigned int)q_stride + m_block * (unsigned int)BLOCK_M;
                    if (warp == 0) {
                        if (elect_sync()) {
                            tma_store_2d((&O), 0, out_row0, smem_o00_addr);
                            tma_store_2d((&O), 64, out_row0, smem_o01_addr);
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                } else {
                    {
                        if (warp == 4) {
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                        }
                        asm volatile("barrier.sync 4, 128;" ::: "memory");
                        asm volatile("tcgen05.fence::after_thread_sync;");
                    }
                    {
                        #pragma unroll
                        for (int col_2 = 0; col_2 < HEAD_DIM / 128; col_2++) {
                            int addr_2 = taddr + (unsigned int)256 + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(col_2 * 64);
                            float _tmem_load_18[64];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(_tmem_load_18[0]), "=f"(_tmem_load_18[1]), "=f"(_tmem_load_18[2]), "=f"(_tmem_load_18[3]), "=f"(_tmem_load_18[4]), "=f"(_tmem_load_18[5]), "=f"(_tmem_load_18[6]), "=f"(_tmem_load_18[7]), "=f"(_tmem_load_18[8]), "=f"(_tmem_load_18[9]), "=f"(_tmem_load_18[10]), "=f"(_tmem_load_18[11]), "=f"(_tmem_load_18[12]), "=f"(_tmem_load_18[13]), "=f"(_tmem_load_18[14]), "=f"(_tmem_load_18[15]), "=f"(_tmem_load_18[16]), "=f"(_tmem_load_18[17]), "=f"(_tmem_load_18[18]), "=f"(_tmem_load_18[19]), "=f"(_tmem_load_18[20]), "=f"(_tmem_load_18[21]), "=f"(_tmem_load_18[22]), "=f"(_tmem_load_18[23]), "=f"(_tmem_load_18[24]), "=f"(_tmem_load_18[25]), "=f"(_tmem_load_18[26]), "=f"(_tmem_load_18[27]), "=f"(_tmem_load_18[28]), "=f"(_tmem_load_18[29]), "=f"(_tmem_load_18[30]), "=f"(_tmem_load_18[31])
                                : "r"(addr_2));
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(_tmem_load_18[32]), "=f"(_tmem_load_18[33]), "=f"(_tmem_load_18[34]), "=f"(_tmem_load_18[35]), "=f"(_tmem_load_18[36]), "=f"(_tmem_load_18[37]), "=f"(_tmem_load_18[38]), "=f"(_tmem_load_18[39]), "=f"(_tmem_load_18[40]), "=f"(_tmem_load_18[41]), "=f"(_tmem_load_18[42]), "=f"(_tmem_load_18[43]), "=f"(_tmem_load_18[44]), "=f"(_tmem_load_18[45]), "=f"(_tmem_load_18[46]), "=f"(_tmem_load_18[47]), "=f"(_tmem_load_18[48]), "=f"(_tmem_load_18[49]), "=f"(_tmem_load_18[50]), "=f"(_tmem_load_18[51]), "=f"(_tmem_load_18[52]), "=f"(_tmem_load_18[53]), "=f"(_tmem_load_18[54]), "=f"(_tmem_load_18[55]), "=f"(_tmem_load_18[56]), "=f"(_tmem_load_18[57]), "=f"(_tmem_load_18[58]), "=f"(_tmem_load_18[59]), "=f"(_tmem_load_18[60]), "=f"(_tmem_load_18[61]), "=f"(_tmem_load_18[62]), "=f"(_tmem_load_18[63])
                                : "r"(addr_2 + 32));
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            #if __CUDA_ARCH__ >= 1000
                            const float2 _scale2_77 = {final_scale, final_scale};
                            #pragma unroll
                            for (int _ls = 0; _ls < 32; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_18)[_ls], _scale2_77);
                            #else
                            #pragma unroll
                            for (int _ls = 0; _ls < 64; _ls++) {
                                _tmem_load_18[_ls] = _tmem_load_18[_ls] * final_scale;
                            }
                            #endif
                            unsigned int _tmem_load_18_bf16[32];
                            #pragma unroll
                            for (int _lp = 0; _lp < 32; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_18[_lp*2 + 0], _tmem_load_18[_lp*2+1 + 0]));
                                _tmem_load_18_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o10_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + col_2 * 128 ^ ((warp % 4 * 32 + lane) * 128 + col_2 * 128 >> 7 & 7) << 4))), "r"(_tmem_load_18_bf16[0]), "r"(_tmem_load_18_bf16[1]), "r"(_tmem_load_18_bf16[2]), "r"(_tmem_load_18_bf16[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o10_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + (col_2 * 128 + 16) ^ ((warp % 4 * 32 + lane) * 128 + (col_2 * 128 + 16) >> 7 & 7) << 4))), "r"(_tmem_load_18_bf16[4]), "r"(_tmem_load_18_bf16[5]), "r"(_tmem_load_18_bf16[6]), "r"(_tmem_load_18_bf16[7]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o10_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + (col_2 * 128 + 32) ^ ((warp % 4 * 32 + lane) * 128 + (col_2 * 128 + 32) >> 7 & 7) << 4))), "r"(_tmem_load_18_bf16[8]), "r"(_tmem_load_18_bf16[9]), "r"(_tmem_load_18_bf16[10]), "r"(_tmem_load_18_bf16[11]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o10_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + (col_2 * 128 + 48) ^ ((warp % 4 * 32 + lane) * 128 + (col_2 * 128 + 48) >> 7 & 7) << 4))), "r"(_tmem_load_18_bf16[12]), "r"(_tmem_load_18_bf16[13]), "r"(_tmem_load_18_bf16[14]), "r"(_tmem_load_18_bf16[15]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o10_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + (col_2 * 128 + 64) ^ ((warp % 4 * 32 + lane) * 128 + (col_2 * 128 + 64) >> 7 & 7) << 4))), "r"(_tmem_load_18_bf16[16]), "r"(_tmem_load_18_bf16[17]), "r"(_tmem_load_18_bf16[18]), "r"(_tmem_load_18_bf16[19]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o10_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + (col_2 * 128 + 80) ^ ((warp % 4 * 32 + lane) * 128 + (col_2 * 128 + 80) >> 7 & 7) << 4))), "r"(_tmem_load_18_bf16[20]), "r"(_tmem_load_18_bf16[21]), "r"(_tmem_load_18_bf16[22]), "r"(_tmem_load_18_bf16[23]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o10_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + (col_2 * 128 + 96) ^ ((warp % 4 * 32 + lane) * 128 + (col_2 * 128 + 96) >> 7 & 7) << 4))), "r"(_tmem_load_18_bf16[24]), "r"(_tmem_load_18_bf16[25]), "r"(_tmem_load_18_bf16[26]), "r"(_tmem_load_18_bf16[27]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o10_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + (col_2 * 128 + 112) ^ ((warp % 4 * 32 + lane) * 128 + (col_2 * 128 + 112) >> 7 & 7) << 4))), "r"(_tmem_load_18_bf16[28]), "r"(_tmem_load_18_bf16[29]), "r"(_tmem_load_18_bf16[30]), "r"(_tmem_load_18_bf16[31]) : "memory");
                        }
                        #pragma unroll
                        for (int col_3 = 0; col_3 < HEAD_DIM / 128; col_3++) {
                            int addr_3 = taddr + (unsigned int)256 + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(HEAD_DIM / 2) + (unsigned int)(col_3 * 64);
                            float _tmem_load_19[64];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(_tmem_load_19[0]), "=f"(_tmem_load_19[1]), "=f"(_tmem_load_19[2]), "=f"(_tmem_load_19[3]), "=f"(_tmem_load_19[4]), "=f"(_tmem_load_19[5]), "=f"(_tmem_load_19[6]), "=f"(_tmem_load_19[7]), "=f"(_tmem_load_19[8]), "=f"(_tmem_load_19[9]), "=f"(_tmem_load_19[10]), "=f"(_tmem_load_19[11]), "=f"(_tmem_load_19[12]), "=f"(_tmem_load_19[13]), "=f"(_tmem_load_19[14]), "=f"(_tmem_load_19[15]), "=f"(_tmem_load_19[16]), "=f"(_tmem_load_19[17]), "=f"(_tmem_load_19[18]), "=f"(_tmem_load_19[19]), "=f"(_tmem_load_19[20]), "=f"(_tmem_load_19[21]), "=f"(_tmem_load_19[22]), "=f"(_tmem_load_19[23]), "=f"(_tmem_load_19[24]), "=f"(_tmem_load_19[25]), "=f"(_tmem_load_19[26]), "=f"(_tmem_load_19[27]), "=f"(_tmem_load_19[28]), "=f"(_tmem_load_19[29]), "=f"(_tmem_load_19[30]), "=f"(_tmem_load_19[31])
                                : "r"(addr_3));
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(_tmem_load_19[32]), "=f"(_tmem_load_19[33]), "=f"(_tmem_load_19[34]), "=f"(_tmem_load_19[35]), "=f"(_tmem_load_19[36]), "=f"(_tmem_load_19[37]), "=f"(_tmem_load_19[38]), "=f"(_tmem_load_19[39]), "=f"(_tmem_load_19[40]), "=f"(_tmem_load_19[41]), "=f"(_tmem_load_19[42]), "=f"(_tmem_load_19[43]), "=f"(_tmem_load_19[44]), "=f"(_tmem_load_19[45]), "=f"(_tmem_load_19[46]), "=f"(_tmem_load_19[47]), "=f"(_tmem_load_19[48]), "=f"(_tmem_load_19[49]), "=f"(_tmem_load_19[50]), "=f"(_tmem_load_19[51]), "=f"(_tmem_load_19[52]), "=f"(_tmem_load_19[53]), "=f"(_tmem_load_19[54]), "=f"(_tmem_load_19[55]), "=f"(_tmem_load_19[56]), "=f"(_tmem_load_19[57]), "=f"(_tmem_load_19[58]), "=f"(_tmem_load_19[59]), "=f"(_tmem_load_19[60]), "=f"(_tmem_load_19[61]), "=f"(_tmem_load_19[62]), "=f"(_tmem_load_19[63])
                                : "r"(addr_3 + 32));
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            #if __CUDA_ARCH__ >= 1000
                            const float2 _scale2_78 = {final_scale, final_scale};
                            #pragma unroll
                            for (int _ls = 0; _ls < 32; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_19)[_ls], _scale2_78);
                            #else
                            #pragma unroll
                            for (int _ls = 0; _ls < 64; _ls++) {
                                _tmem_load_19[_ls] = _tmem_load_19[_ls] * final_scale;
                            }
                            #endif
                            unsigned int _tmem_load_19_bf16[32];
                            #pragma unroll
                            for (int _lp = 0; _lp < 32; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_19[_lp*2 + 0], _tmem_load_19[_lp*2+1 + 0]));
                                _tmem_load_19_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o11_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + col_3 * 128 ^ ((warp % 4 * 32 + lane) * 128 + col_3 * 128 >> 7 & 7) << 4))), "r"(_tmem_load_19_bf16[0]), "r"(_tmem_load_19_bf16[1]), "r"(_tmem_load_19_bf16[2]), "r"(_tmem_load_19_bf16[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o11_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + (col_3 * 128 + 16) ^ ((warp % 4 * 32 + lane) * 128 + (col_3 * 128 + 16) >> 7 & 7) << 4))), "r"(_tmem_load_19_bf16[4]), "r"(_tmem_load_19_bf16[5]), "r"(_tmem_load_19_bf16[6]), "r"(_tmem_load_19_bf16[7]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o11_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + (col_3 * 128 + 32) ^ ((warp % 4 * 32 + lane) * 128 + (col_3 * 128 + 32) >> 7 & 7) << 4))), "r"(_tmem_load_19_bf16[8]), "r"(_tmem_load_19_bf16[9]), "r"(_tmem_load_19_bf16[10]), "r"(_tmem_load_19_bf16[11]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o11_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + (col_3 * 128 + 48) ^ ((warp % 4 * 32 + lane) * 128 + (col_3 * 128 + 48) >> 7 & 7) << 4))), "r"(_tmem_load_19_bf16[12]), "r"(_tmem_load_19_bf16[13]), "r"(_tmem_load_19_bf16[14]), "r"(_tmem_load_19_bf16[15]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o11_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + (col_3 * 128 + 64) ^ ((warp % 4 * 32 + lane) * 128 + (col_3 * 128 + 64) >> 7 & 7) << 4))), "r"(_tmem_load_19_bf16[16]), "r"(_tmem_load_19_bf16[17]), "r"(_tmem_load_19_bf16[18]), "r"(_tmem_load_19_bf16[19]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o11_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + (col_3 * 128 + 80) ^ ((warp % 4 * 32 + lane) * 128 + (col_3 * 128 + 80) >> 7 & 7) << 4))), "r"(_tmem_load_19_bf16[20]), "r"(_tmem_load_19_bf16[21]), "r"(_tmem_load_19_bf16[22]), "r"(_tmem_load_19_bf16[23]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o11_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + (col_3 * 128 + 96) ^ ((warp % 4 * 32 + lane) * 128 + (col_3 * 128 + 96) >> 7 & 7) << 4))), "r"(_tmem_load_19_bf16[24]), "r"(_tmem_load_19_bf16[25]), "r"(_tmem_load_19_bf16[26]), "r"(_tmem_load_19_bf16[27]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o11_addr + (unsigned int)((warp % 4 * 32 + lane) * 128 + (col_3 * 128 + 112) ^ ((warp % 4 * 32 + lane) * 128 + (col_3 * 128 + 112) >> 7 & 7) << 4))), "r"(_tmem_load_19_bf16[28]), "r"(_tmem_load_19_bf16[29]), "r"(_tmem_load_19_bf16[30]), "r"(_tmem_load_19_bf16[31]) : "memory");
                        }
                    }
                    asm volatile("fence.proxy.async;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("barrier.sync 2, 128;" ::: "memory");
                    int out_row1 = bh * (unsigned int)q_stride + (m_block + 1) * (unsigned int)BLOCK_M;
                    if (warp == 4) {
                        if (elect_sync()) {
                            tma_store_2d((&O), 0, out_row1, smem_o10_addr);
                            tma_store_2d((&O), 64, out_row1, smem_o11_addr);
                        }
                        asm volatile("cp.async.bulk.commit_group;");
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
        }
    }
    // ---- Role: correction_lo ----
    if (warp >= 8 && warp <= 9) {
        { // correction_lo_main
            unsigned int total_tiles_1 = (seqlen_q + 4 * BLOCK_M - 1) / (4 * BLOCK_M) * total_bh;
            int owned_row_base = make_warp_uniform(warp % 2 * 32);
            unsigned int _phase_p_empty_0 = 1;
            unsigned int _phase_p_empty_1 = 1;
            unsigned int _phase_corr_sig_lo_0 = 0;
            unsigned int _phase_corr_sig_lo_1 = 0;
            unsigned int _phase_o_full_0 = 0;
            unsigned int _phase_o_full_1 = 0;
            #pragma unroll 1
            for (unsigned int tile_idx_1 = cluster_id; tile_idx_1 < total_tiles_1; tile_idx_1 += num_clusters) {
                unsigned int cluster_m_block_1 = tile_idx_1 % (unsigned int)((seqlen_q + 4 * BLOCK_M - 1) / (4 * BLOCK_M));
                unsigned int bh_1 = tile_idx_1 / (unsigned int)((seqlen_q + 4 * BLOCK_M - 1) / (4 * BLOCK_M));
                unsigned int m_block_1 = cluster_m_block_1 * 4 + (unsigned int)(cta_rank * 2);
                unsigned int num_n_blocks_1 = (seqlen_kv + BLOCK_N - 1) / BLOCK_N;
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
                for (unsigned int n_iter_1 = 1; n_iter_1 < num_n_blocks_1; n_iter_1++) {
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
                        for (int col_4 = 0; col_4 < HEAD_DIM / 16; col_4++) {
                            int addr_4 = make_warp_uniform(taddr + (unsigned int)128 + (unsigned int)tmem_row_base + (unsigned int)(col_4 * 16));
                            float _tmem_load_20[16];
                            tmem_ld_x16(&_tmem_load_20[0], addr_4);
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            #if __CUDA_ARCH__ >= 1000
                            const float2 _scale2_0 = {scale, scale};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_20)[_ls], _scale2_0);
                            #else
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++) {
                                _tmem_load_20[_ls] = _tmem_load_20[_ls] * scale;
                            }
                            #endif
                            tmem_st_x16_f32(addr_4, _tmem_load_20);
                        }
                    }
                    int skip_rescale_0 = skip_rescale;
                    if (skip_rescale_0 == 0) {
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
                    int logical_row_1 = owned_row_base + lane;
                    int tmem_row_base_2 = make_warp_uniform(owned_row_base << 16);
                    float scale_3 = row_state[logical_row_1 + BLOCK_M];
                    int skip_rescale_4 = __all_sync(0xFFFFFFFF, scale_3 == 1.0f);
                    if (skip_rescale_4 == 0) {
                        #pragma unroll
                        for (int col_5 = 0; col_5 < HEAD_DIM / 16; col_5++) {
                            int addr_5 = make_warp_uniform(taddr + (unsigned int)256 + (unsigned int)tmem_row_base_2 + (unsigned int)(col_5 * 16));
                            float _tmem_load_21[16];
                            tmem_ld_x16(&_tmem_load_21[0], addr_5);
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            #if __CUDA_ARCH__ >= 1000
                            const float2 _scale2_1 = {scale_3, scale_3};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_21)[_ls], _scale2_1);
                            #else
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++) {
                                _tmem_load_21[_ls] = _tmem_load_21[_ls] * scale_3;
                            }
                            #endif
                            tmem_st_x16_f32(addr_5, _tmem_load_21);
                        }
                    }
                    skip_rescale_0 = skip_rescale_4;
                    if (skip_rescale_0 == 0) {
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
            unsigned int total_tiles_2 = (seqlen_q + 4 * BLOCK_M - 1) / (4 * BLOCK_M) * total_bh;
            int owned_row_base_1 = make_warp_uniform(64 + warp % 2 * 32);
            unsigned int _phase_p_empty_0_1 = 1;
            unsigned int _phase_p_empty_1_1 = 1;
            unsigned int _phase_corr_sig_hi_0 = 0;
            unsigned int _phase_corr_sig_hi_1 = 0;
            unsigned int _phase_o_full_0_1 = 0;
            unsigned int _phase_o_full_1_1 = 0;
            #pragma unroll 1
            for (unsigned int tile_idx_2 = cluster_id; tile_idx_2 < total_tiles_2; tile_idx_2 += num_clusters) {
                unsigned int cluster_m_block_2 = tile_idx_2 % (unsigned int)((seqlen_q + 4 * BLOCK_M - 1) / (4 * BLOCK_M));
                unsigned int bh_2 = tile_idx_2 / (unsigned int)((seqlen_q + 4 * BLOCK_M - 1) / (4 * BLOCK_M));
                unsigned int m_block_2 = cluster_m_block_2 * 4 + (unsigned int)(cta_rank * 2);
                unsigned int num_n_blocks_2 = (seqlen_kv + BLOCK_N - 1) / BLOCK_N;
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
                for (unsigned int n_iter_2 = 1; n_iter_2 < num_n_blocks_2; n_iter_2++) {
                    mbarrier_wait(corr_sig_hi_addr, _phase_corr_sig_hi_0);
                    _phase_corr_sig_hi_0 ^= 1;
                    mbarrier_wait(p_empty_addr, _phase_p_empty_0_1);
                    _phase_p_empty_0_1 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int logical_row_2 = owned_row_base_1 + lane;
                    int tmem_row_base_1 = make_warp_uniform(owned_row_base_1 << 16);
                    float scale_1 = row_state[logical_row_2];
                    int skip_rescale_1 = __all_sync(0xFFFFFFFF, scale_1 == 1.0f);
                    if (skip_rescale_1 == 0) {
                        #pragma unroll
                        for (int col_6 = 0; col_6 < HEAD_DIM / 16; col_6++) {
                            int addr_6 = make_warp_uniform(taddr + (unsigned int)128 + (unsigned int)tmem_row_base_1 + (unsigned int)(col_6 * 16));
                            float _tmem_load_22[16];
                            tmem_ld_x16(&_tmem_load_22[0], addr_6);
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            #if __CUDA_ARCH__ >= 1000
                            const float2 _scale2_0 = {scale_1, scale_1};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_22)[_ls], _scale2_0);
                            #else
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++) {
                                _tmem_load_22[_ls] = _tmem_load_22[_ls] * scale_1;
                            }
                            #endif
                            tmem_st_x16_f32(addr_6, _tmem_load_22);
                        }
                    }
                    int skip_rescale_0_1 = skip_rescale_1;
                    if (skip_rescale_0_1 == 0) {
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
                    int logical_row_1_1 = owned_row_base_1 + lane;
                    int tmem_row_base_2_1 = make_warp_uniform(owned_row_base_1 << 16);
                    float scale_3_1 = row_state[logical_row_1_1 + BLOCK_M];
                    int skip_rescale_4_1 = __all_sync(0xFFFFFFFF, scale_3_1 == 1.0f);
                    if (skip_rescale_4_1 == 0) {
                        #pragma unroll
                        for (int col_7 = 0; col_7 < HEAD_DIM / 16; col_7++) {
                            int addr_7 = make_warp_uniform(taddr + (unsigned int)256 + (unsigned int)tmem_row_base_2_1 + (unsigned int)(col_7 * 16));
                            float _tmem_load_23[16];
                            tmem_ld_x16(&_tmem_load_23[0], addr_7);
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            #if __CUDA_ARCH__ >= 1000
                            const float2 _scale2_1 = {scale_3_1, scale_3_1};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_23)[_ls], _scale2_1);
                            #else
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++) {
                                _tmem_load_23[_ls] = _tmem_load_23[_ls] * scale_3_1;
                            }
                            #endif
                            tmem_st_x16_f32(addr_7, _tmem_load_23);
                        }
                    }
                    skip_rescale_0_1 = skip_rescale_4_1;
                    if (skip_rescale_0_1 == 0) {
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
                unsigned int total_tiles_3 = (seqlen_q + 4 * BLOCK_M - 1) / (4 * BLOCK_M) * total_bh;
                unsigned int k_stage = 0;
                unsigned int k_phase = 0;
                unsigned int v_stage = 0;
                unsigned int v_phase = 0;
                #pragma unroll 1
                for (unsigned int tile_idx_3 = cluster_id; tile_idx_3 < total_tiles_3; tile_idx_3 += num_clusters) {
                    unsigned int cluster_m_block_3 = tile_idx_3 % (unsigned int)((seqlen_q + 4 * BLOCK_M - 1) / (4 * BLOCK_M));
                    unsigned int bh_3 = tile_idx_3 / (unsigned int)((seqlen_q + 4 * BLOCK_M - 1) / (4 * BLOCK_M));
                    unsigned int m_block_3 = cluster_m_block_3 * 4 + (unsigned int)(cta_rank * 2);
                    unsigned int num_n_blocks_3 = (seqlen_kv + BLOCK_N - 1) / BLOCK_N;
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
                        tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb_qk0, make_sf_cp_desc_lo_sbo256((((smem_sfk_addr) >> 4) + (k_stage) * 64)));
                        tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb_qk0 + 4), make_sf_cp_desc_lo_sbo256((((smem_sfk_addr) >> 4) + (k_stage) * 64 + 8)));
                    }
                    int _smem_desc_lo_0 = (((smem_q_addr) >> 4) & 0x3FFF) + (0) * 512;
                    int _smem_desc_lo_1 = (((smem_k_addr) >> 4) & 0x3FFF) + (k_stage) * 256;
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_0) | ((uint64_t)0x80004020 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_1) | ((uint64_t)0x80004020 << 32);

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
                    int _smem_desc_lo_2 = (((smem_q_addr) >> 4) & 0x3FFF) + (1) * 512;
                    int _smem_desc_lo_3 = (((smem_k_addr) >> 4) & 0x3FFF) + (k_stage) * 256;
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_2) | ((uint64_t)0x80004020 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_3) | ((uint64_t)0x80004020 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 0, b_desc + 0,
                                0x10200480U, tmem_tmem_sfa_qk1 + 0, tmem_tmem_sfb_qk0 + 0, 0);
                            tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 2, b_desc + 2,
                                0x10200480U, tmem_tmem_sfa_qk1 + 4, tmem_tmem_sfb_qk0 + 4, 1);
                        }
                    }
                    elect_commit_cg2_multicast(s_full_addr + 8, (uint16_t)(3));
                    elect_commit_cg2_multicast(k_empty_addr + (k_stage) * 8, (uint16_t)(3));
                    k_stage += 1;
                    if (k_stage == 4) { k_stage = 0; k_phase ^= 1; }
                    unsigned int first_pv = 1;
                    #pragma unroll 1
                    for (unsigned int n_iter_3 = 0; n_iter_3 < num_n_blocks_3 - 1; n_iter_3++) {
                        int first_pv_flag = first_pv;
                        mbarrier_wait(v_full_addr + (v_stage) * 8, v_phase);
                        mbarrier_wait(k_full_addr + (k_stage) * 8, k_phase);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        if (elect_sync()) {
                            tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb_qk0, make_sf_cp_desc_lo_sbo256((((smem_sfk_addr) >> 4) + (k_stage) * 64)));
                            tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb_qk0 + 4), make_sf_cp_desc_lo_sbo256((((smem_sfk_addr) >> 4) + (k_stage) * 64 + 8)));
                        }
                        mbarrier_wait(s_empty_addr + 8, _phase_s_empty_1);
                        _phase_s_empty_1 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _smem_desc_lo_4 = (((smem_q_addr) >> 4) & 0x3FFF) + (0) * 512;
                        int _smem_desc_lo_5 = (((smem_k_addr) >> 4) & 0x3FFF) + (k_stage) * 256;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_4) | ((uint64_t)0x80004020 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_5) | ((uint64_t)0x80004020 << 32);

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
                        int _smem_desc_lo_6 = (((smem_q_addr) >> 4) & 0x3FFF) + (1) * 512;
                        int _smem_desc_lo_7 = (((smem_k_addr) >> 4) & 0x3FFF) + (k_stage) * 256;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_6) | ((uint64_t)0x80004020 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_7) | ((uint64_t)0x80004020 << 32);

                                tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 0, b_desc + 0,
                                    0x10200480U, tmem_tmem_sfa_qk1 + 0, tmem_tmem_sfb_qk0 + 0, 0);
                                tcgen05_mma_mxf4nvf4_bs_cta2(tmem_scores, a_desc + 2, b_desc + 2,
                                    0x10200480U, tmem_tmem_sfa_qk1 + 4, tmem_tmem_sfb_qk0 + 4, 1);
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
                        int _smem_desc_lo_8 = (((smem_p_addr) >> 4) & 0x3FFF) + (0) * 512;
                        int _smem_desc_lo_9 = (((smem_vt_addr) >> 4) & 0x3FFF) + (v_stage) * 256;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_8) | ((uint64_t)0x80004020 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_9) | ((uint64_t)0x80004020 << 32);

                                tcgen05_mma_mxf4nvf4_bs_cta2(tmem_output_0, a_desc + 0, b_desc + 0,
                                    0x10200480U, tmem_tmem_sfa_pv0_lo + 0, tmem_tmem_sfb_pv_lo + 0, ((first_pv_flag) ? 0 : 1));
                            }
                        }
                        mbarrier_wait(p_full_addr + 8, _phase_p_full_1);
                        _phase_p_full_1 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _smem_desc_lo_10 = (((smem_p_addr) >> 4) & 0x3FFF) + (1) * 512;
                        int _smem_desc_lo_11 = (((smem_vt_addr) >> 4) & 0x3FFF) + (v_stage) * 256;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_10) | ((uint64_t)0x80004020 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_11) | ((uint64_t)0x80004020 << 32);

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
                        int _smem_desc_lo_12 = (((smem_p_addr + 32) >> 4) & 0x3FFF) + (0) * 512;
                        int _smem_desc_lo_13 = (((smem_vt_addr + 32) >> 4) & 0x3FFF) + (v_stage) * 256;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_12) | ((uint64_t)0x80004020 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_13) | ((uint64_t)0x80004020 << 32);

                                tcgen05_mma_mxf4nvf4_bs_cta2(tmem_output_0, a_desc + 0, b_desc + 0,
                                    0x10200480U, tmem_tmem_sfa_pv0_hi + 0, tmem_tmem_sfb_pv_hi + 0, 1);
                            }
                        }
                        elect_commit_cg2_multicast(p_empty_addr, (uint16_t)(3));
                        mbarrier_wait(p_full_2_addr + 8, _phase_p_full_2_1);
                        _phase_p_full_2_1 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _smem_desc_lo_14 = (((smem_p_addr + 32) >> 4) & 0x3FFF) + (1) * 512;
                        int _smem_desc_lo_15 = (((smem_vt_addr + 32) >> 4) & 0x3FFF) + (v_stage) * 256;
                        if (elect_sync()) {
                            {
                                uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_14) | ((uint64_t)0x80004020 << 32);
                                uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_15) | ((uint64_t)0x80004020 << 32);

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
                    int _smem_desc_lo_16 = (((smem_p_addr) >> 4) & 0x3FFF) + (0) * 512;
                    int _smem_desc_lo_17 = (((smem_vt_addr) >> 4) & 0x3FFF) + (v_stage) * 256;
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_16) | ((uint64_t)0x80004020 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_17) | ((uint64_t)0x80004020 << 32);

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
                    int _smem_desc_lo_18 = (((smem_p_addr + 32) >> 4) & 0x3FFF) + (0) * 512;
                    int _smem_desc_lo_19 = (((smem_vt_addr + 32) >> 4) & 0x3FFF) + (v_stage) * 256;
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_18) | ((uint64_t)0x80004020 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_19) | ((uint64_t)0x80004020 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2(tmem_output_0, a_desc + 0, b_desc + 0,
                                0x10200480U, tmem_tmem_sfa_pv0_hi + 0, tmem_tmem_sfb_pv_hi + 0, 1);
                        }
                    }
                    elect_commit_cg2_multicast(p_empty_addr, (uint16_t)(3));
                    mbarrier_wait(p_full_addr + 8, _phase_p_full_1);
                    _phase_p_full_1 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _smem_desc_lo_20 = (((smem_p_addr) >> 4) & 0x3FFF) + (1) * 512;
                    int _smem_desc_lo_21 = (((smem_vt_addr) >> 4) & 0x3FFF) + (v_stage) * 256;
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_20) | ((uint64_t)0x80004020 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_21) | ((uint64_t)0x80004020 << 32);

                            tcgen05_mma_mxf4nvf4_bs_cta2(tmem_output_1, a_desc + 0, b_desc + 0,
                                0x10200480U, tmem_tmem_sfa_pv1_lo + 0, tmem_tmem_sfb_pv_lo + 0, ((first_pv_flag_1) ? 0 : 1));
                        }
                    }
                    mbarrier_wait(p_full_2_addr + 8, _phase_p_full_2_1);
                    _phase_p_full_2_1 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _smem_desc_lo_22 = (((smem_p_addr + 32) >> 4) & 0x3FFF) + (1) * 512;
                    int _smem_desc_lo_23 = (((smem_vt_addr + 32) >> 4) & 0x3FFF) + (v_stage) * 256;
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_smem_desc_lo_22) | ((uint64_t)0x80004020 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_smem_desc_lo_23) | ((uint64_t)0x80004020 << 32);

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
            unsigned int total_tiles_4 = (seqlen_q + 4 * BLOCK_M - 1) / (4 * BLOCK_M) * total_bh;
            unsigned int num_n_blocks_all = kv_stride / BLOCK_N;
            unsigned int num_q_blocks_all = q_stride / BLOCK_M;
            unsigned int k_load_stage = 0;
            unsigned int _phase_q_empty_0 = 1;
            unsigned int _phase_k_empty = 1;
            #pragma unroll 1
            for (unsigned int tile_idx_4 = cluster_id; tile_idx_4 < total_tiles_4; tile_idx_4 += num_clusters) {
                unsigned int cluster_m_block_4 = tile_idx_4 % (unsigned int)((seqlen_q + 4 * BLOCK_M - 1) / (4 * BLOCK_M));
                unsigned int bh_4 = tile_idx_4 / (unsigned int)((seqlen_q + 4 * BLOCK_M - 1) / (4 * BLOCK_M));
                unsigned int m_block_4 = cluster_m_block_4 * 4 + (unsigned int)(cta_rank * 2);
                unsigned int num_n_blocks_4 = (seqlen_kv + BLOCK_N - 1) / BLOCK_N;
                int q_row = bh_4 * (unsigned int)q_stride + m_block_4 * (unsigned int)BLOCK_M;
                int q_sf_tile = bh_4 * num_q_blocks_all + m_block_4;
                mbarrier_wait(q_empty_addr, _phase_q_empty_0);
                _phase_q_empty_0 ^= 1;
                if (elect_sync()) {
                    tma_2d_gmem2smem_cta2(smem_q_addr, (&Q), 0, q_row, ((q_full_addr) & 0xFEFFFFFF));
                    tma_2d_gmem2smem_cta2(smem_sfq_addr, (&SFQ), 0, q_sf_tile * 32, ((q_full_addr) & 0xFEFFFFFF));
                    tma_2d_gmem2smem_cta2(smem_q_addr + 8192, (&Q), 0, q_row + BLOCK_M, ((q_full_addr + 8) & 0xFEFFFFFF));
                    tma_2d_gmem2smem_cta2(smem_sfq_addr + 1024, (&SFQ), 0, (q_sf_tile + 1) * 32, ((q_full_addr + 8) & 0xFEFFFFFF));
                    asm volatile(
                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                        :: "r"((q_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(9216)) : "memory");
                    asm volatile(
                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                        :: "r"((q_full_addr + 8) & 0xFEFFFFFF), "r"((uint32_t)(9216)) : "memory");
                }
                #pragma unroll 1
                for (unsigned int ni = 0; ni < num_n_blocks_4; ni++) {
                    unsigned int n = num_n_blocks_4 - 1 - ni;
                    int kv_row = bh_4 * (unsigned int)kv_stride + n * (unsigned int)BLOCK_N;
                    int kv_sf_tile = bh_4 * num_n_blocks_all + n;
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
        }
    }
    // ---- Role: v_load ----
    if (warp == 14) {
        { // v_load_main
            unsigned int total_tiles_5 = (seqlen_q + 4 * BLOCK_M - 1) / (4 * BLOCK_M) * total_bh;
            unsigned int num_n_blocks_all_1 = kv_stride / BLOCK_N;
            unsigned int v_load_stage = 0;
            unsigned int _phase_v_empty = 1;
            #pragma unroll 1
            for (unsigned int tile_idx_5 = cluster_id; tile_idx_5 < total_tiles_5; tile_idx_5 += num_clusters) {
                unsigned int cluster_m_block_5 = tile_idx_5 % (unsigned int)((seqlen_q + 4 * BLOCK_M - 1) / (4 * BLOCK_M));
                unsigned int bh_5 = tile_idx_5 / (unsigned int)((seqlen_q + 4 * BLOCK_M - 1) / (4 * BLOCK_M));
                unsigned int m_block_5 = cluster_m_block_5 * 4 + (unsigned int)(cta_rank * 2);
                unsigned int num_n_blocks_5 = (seqlen_kv + BLOCK_N - 1) / BLOCK_N;
                #pragma unroll 1
                for (unsigned int ni_1 = 0; ni_1 < num_n_blocks_5; ni_1++) {
                    unsigned int n_1 = num_n_blocks_5 - 1 - ni_1;
                    int kv_sf_tile_1 = bh_5 * num_n_blocks_all_1 + n_1;
                    int vt_row = bh_5 * (unsigned int)HEAD_DIM;
                    int vt_col = n_1 * (unsigned int)(BLOCK_N / 2);
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
