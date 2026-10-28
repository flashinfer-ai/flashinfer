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
#define TMEM_NCOLS 480
#define TMEM_TMEM_ACC_OFFSET 0
#define TMEM_TMEM_SFA0_OFFSET 320
#define TMEM_TMEM_SFA1_OFFSET 328
#define TMEM_TMEM_SFA2_OFFSET 336
#define TMEM_TMEM_SFA3_OFFSET 344
#define TMEM_TMEM_SFB0_OFFSET 352
#define TMEM_TMEM_SFB1_OFFSET 360
#define TMEM_TMEM_SFB2_OFFSET 368
#define TMEM_TMEM_SFB3_OFFSET 376
#define TMEM_TMEM_SFB4_OFFSET 384
#define TMEM_TMEM_SFB5_OFFSET 392
#define TMEM_TMEM_SFB6_OFFSET 400
#define TMEM_TMEM_SFB7_OFFSET 408
#define TMEM_TMEM_Q_OFFSET 416
#define NUM_RAW_PIPE_STAGES 3
#define NUM_V_PIPE_STAGES 2
#define NUM_P_PIPE_STAGES 2
#define NUM_S_PIPE_STAGES 2
#define NUM_SF_PIPE_STAGES 2
#define SMEM_SMEM_Q0_OFF 1024
#define SMEM_SMEM_Q0_STAGE_BYTES 4096
#define SMEM_SMEM_Q0_STRIDE 4096
#define SMEM_SMEM_Q1_OFF 5120
#define SMEM_SMEM_Q1_STAGE_BYTES 4096
#define SMEM_SMEM_Q1_STRIDE 4096
#define SMEM_SMEM_Q2_OFF 9216
#define SMEM_SMEM_Q2_STAGE_BYTES 4096
#define SMEM_SMEM_Q2_STRIDE 4096
#define SMEM_SMEM_Q3_OFF 13312
#define SMEM_SMEM_Q3_STAGE_BYTES 4096
#define SMEM_SMEM_Q3_STRIDE 4096
#define SMEM_SMEM_QS_OFF 17408
#define SMEM_SMEM_QS_STAGE_BYTES 2048
#define SMEM_SMEM_QS_STRIDE 2048
#define SMEM_SMEM_K0H0_OFF 19456
#define SMEM_SMEM_K0H0_STAGE_BYTES 2048
#define SMEM_SMEM_K0H0_STRIDE 28672
#define SMEM_SMEM_K1H0_OFF 23552
#define SMEM_SMEM_K1H0_STAGE_BYTES 2048
#define SMEM_SMEM_K1H0_STRIDE 28672
#define SMEM_SMEM_K2H0_OFF 27648
#define SMEM_SMEM_K2H0_STAGE_BYTES 2048
#define SMEM_SMEM_K2H0_STRIDE 28672
#define SMEM_SMEM_K3H0_OFF 31744
#define SMEM_SMEM_K3H0_STAGE_BYTES 2048
#define SMEM_SMEM_K3H0_STRIDE 28672
#define SMEM_SMEM_K0H1_OFF 21504
#define SMEM_SMEM_K0H1_STAGE_BYTES 2048
#define SMEM_SMEM_K0H1_STRIDE 28672
#define SMEM_SMEM_K1H1_OFF 25600
#define SMEM_SMEM_K1H1_STAGE_BYTES 2048
#define SMEM_SMEM_K1H1_STRIDE 28672
#define SMEM_SMEM_K2H1_OFF 29696
#define SMEM_SMEM_K2H1_STAGE_BYTES 2048
#define SMEM_SMEM_K2H1_STRIDE 28672
#define SMEM_SMEM_K3H1_OFF 33792
#define SMEM_SMEM_K3H1_STAGE_BYTES 2048
#define SMEM_SMEM_K3H1_STRIDE 28672
#define SMEM_SMEM_KQ0H0_OFF 19456
#define SMEM_SMEM_KQ0H0_STAGE_BYTES 4096
#define SMEM_SMEM_KQ0H0_STRIDE 28672
#define SMEM_SMEM_KQ1H0_OFF 23552
#define SMEM_SMEM_KQ1H0_STAGE_BYTES 4096
#define SMEM_SMEM_KQ1H0_STRIDE 28672
#define SMEM_SMEM_KQ2H0_OFF 27648
#define SMEM_SMEM_KQ2H0_STAGE_BYTES 4096
#define SMEM_SMEM_KQ2H0_STRIDE 28672
#define SMEM_SMEM_KQ3H0_OFF 31744
#define SMEM_SMEM_KQ3H0_STAGE_BYTES 4096
#define SMEM_SMEM_KQ3H0_STRIDE 28672
#define SMEM_SMEM_KQ0H1_OFF 21504
#define SMEM_SMEM_KQ0H1_STAGE_BYTES 4096
#define SMEM_SMEM_KQ0H1_STRIDE 28672
#define SMEM_SMEM_KQ1H1_OFF 25600
#define SMEM_SMEM_KQ1H1_STAGE_BYTES 4096
#define SMEM_SMEM_KQ1H1_STRIDE 28672
#define SMEM_SMEM_KQ2H1_OFF 29696
#define SMEM_SMEM_KQ2H1_STAGE_BYTES 4096
#define SMEM_SMEM_KQ2H1_STRIDE 28672
#define SMEM_SMEM_KQ3H1_OFF 33792
#define SMEM_SMEM_KQ3H1_STAGE_BYTES 4096
#define SMEM_SMEM_KQ3H1_STRIDE 28672
#define SMEM_SMEM_KO0H0_OFF 35840
#define SMEM_SMEM_KO0H0_STAGE_BYTES 2048
#define SMEM_SMEM_KO0H0_STRIDE 28672
#define SMEM_SMEM_KO1H0_OFF 39936
#define SMEM_SMEM_KO1H0_STAGE_BYTES 2048
#define SMEM_SMEM_KO1H0_STRIDE 28672
#define SMEM_SMEM_KO0H1_OFF 37888
#define SMEM_SMEM_KO0H1_STAGE_BYTES 2048
#define SMEM_SMEM_KO0H1_STRIDE 28672
#define SMEM_SMEM_KO1H1_OFF 41984
#define SMEM_SMEM_KO1H1_STAGE_BYTES 2048
#define SMEM_SMEM_KO1H1_STRIDE 28672
#define SMEM_SMEM_KS_OFF 44032
#define SMEM_SMEM_KS_STAGE_BYTES 4096
#define SMEM_SMEM_KS_STRIDE 28672
#define SMEM_SMEM_RAW_FLAT_OFF 19456
#define SMEM_SMEM_RAW_FLAT_STAGE_BYTES 86016
#define SMEM_SMEM_RAW_FLAT_STRIDE 86016
#define SMEM_SMEM_V0_OFF 105472
#define SMEM_SMEM_V0_STAGE_BYTES 16384
#define SMEM_SMEM_V0_STRIDE 32768
#define SMEM_SMEM_V1_OFF 121856
#define SMEM_SMEM_V1_STAGE_BYTES 16384
#define SMEM_SMEM_V1_STRIDE 32768
#define SMEM_SMEM_P_OFF 171008
#define SMEM_SMEM_P_STAGE_BYTES 8192
#define SMEM_SMEM_P_STRIDE 8192
#define SMEM_ROW_STATE_OFF 187392
#define SMEM_ROW_STATE_STAGE_BYTES 512
#define SMEM_ROW_STATE_STRIDE 512
#define SMEM_SMEM_XCHG_MAX_OFF 187904
#define SMEM_SMEM_XCHG_MAX_STAGE_BYTES 512
#define SMEM_SMEM_XCHG_MAX_STRIDE 512
#define SMEM_SMEM_XCHG_SUM_OFF 188416
#define SMEM_SMEM_XCHG_SUM_STAGE_BYTES 512
#define SMEM_SMEM_XCHG_SUM_STRIDE 512
#define SMEM_SMEM_PAGES_OFF 188928
#define SMEM_SMEM_PAGES_STAGE_BYTES 128
#define SMEM_SMEM_PAGES_STRIDE 128
#define SMEM_SMEM_EPI_OFF 189440
#define SMEM_SMEM_EPI_STAGE_BYTES 32768
#define SMEM_SMEM_EPI_STRIDE 32768
#define SMEM_TOTAL 222208
#define THREADS 512
#define SKIP_XFORM 0
#define LAZY_RESCALE 1
#define IDENTITY_PAGES 0
#define SKIP_EXP 0
#define SKIP_KPUB 0
#define SKIP_PSTORE 0
#define SKIP_QK 0
#define SKIP_PV 0
#define SKIP_SMATH 0
#define SKIP_EPISTORE 0
#define SKIP_EPISTS 0
#define EPI_DIRECT 0
#define TRACE 0
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

__device__ __forceinline__ void mbarrier_wait_token(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait(mbar_addr, phase);
    }
}


__device__ __forceinline__ uint32_t mbarrier_test_wait(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.test_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
}






union MmaSmemDesc {
    uint64_t u64;
    uint32_t u32[2];
};



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



__device__ __forceinline__ void tmem_st_x32_f32(int tmem_addr, const float* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x32.b32"
        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8,"
        "  %9, %10, %11, %12, %13, %14, %15, %16,"
        "  %17, %18, %19, %20, %21, %22, %23, %24,"
        "  %25, %26, %27, %28, %29, %30, %31, %32};"
        :: "r"(tmem_addr),
           "f"(src[0]),  "f"(src[1]),  "f"(src[2]),  "f"(src[3]),
           "f"(src[4]),  "f"(src[5]),  "f"(src[6]),  "f"(src[7]),
           "f"(src[8]),  "f"(src[9]),  "f"(src[10]), "f"(src[11]),
           "f"(src[12]), "f"(src[13]), "f"(src[14]), "f"(src[15]),
           "f"(src[16]), "f"(src[17]), "f"(src[18]), "f"(src[19]),
           "f"(src[20]), "f"(src[21]), "f"(src[22]), "f"(src[23]),
           "f"(src[24]), "f"(src[25]), "f"(src[26]), "f"(src[27]),
           "f"(src[28]), "f"(src[29]), "f"(src[30]), "f"(src[31]));
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




__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
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


__device__ __forceinline__ void tcgen05_commit_cg2_multicast(int mbar_addr, uint16_t cta_mask) {
    asm volatile(
        "{\n\t"
        ".reg .b16 lo, hi;\n\t"
        "mov.b32 {lo, hi}, %1;\n\t"
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one"
        ".shared::cluster.multicast::cluster.b64 [%0], lo;\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"((uint32_t)cta_mask) : "memory");
}


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}


// Qmul4 (QMUL4.E4M3x4.E2M1x4.E4M3x4.SATFINITE) as the instruction: byte k of the result is the E2M1 value in one
// nibble of src times the E4M3 value in one byte of scale, rounded once to E4M3 (to nearest even, saturating to
// +-448). mul .e2m1p4x4 reads nibble 2k of a word, mul .e2m1x4 nibble k of a 16-bit half; both times byte k of
// their scale word.
//   qmul4_direct   nibble 2k      byte k
//   qmul4_b<j>     nibble 2k      byte j
//   qmul4_lower4   nibble k       byte 0
//   qmul4_higher4  nibble k + 4   byte 0

// Byte kByte of scale in all four bytes.
template <int kByte>
__device__ __forceinline__ uint32_t qmul4_broadcast(uint32_t scale) {
  uint32_t y;
  asm("prmt.b32 %0, %1, 0, %2;" : "=r"(y) : "r"(scale), "n"(kByte * 0x1111));
  return y;
}

__device__ __forceinline__ uint32_t qmul4_lower4(uint32_t src, uint32_t scale) {
  uint32_t y;
  asm("{\n.reg .b16 h0, h1;\nmov.b32 {h0, h1}, %1;\nmul.rn.satfinite.e4m3x4.e2m1x4.e4m3x4 %0, h0, %2;\n}\n"
      : "=r"(y)
      : "r"(src), "r"(qmul4_broadcast<0>(scale)));
  return y;
}

__device__ __forceinline__ uint32_t qmul4_higher4(uint32_t src, uint32_t scale) {
  uint32_t y;
  asm("{\n.reg .b16 h0, h1;\nmov.b32 {h0, h1}, %1;\nmul.rn.satfinite.e4m3x4.e2m1x4.e4m3x4 %0, h1, %2;\n}\n"
      : "=r"(y)
      : "r"(src), "r"(qmul4_broadcast<0>(scale)));
  return y;
}

extern "C" {

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) __cluster_dims__(2,1,1) void
kernel_cake_nvfp4_mla_decode_05927afc40e38d96365b(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap QS, const __grid_constant__ CUtensorMap KV, const __grid_constant__ CUtensorMap KVS, const __grid_constant__ CUtensorMap OT, const __grid_constant__ CUtensorMap PO, __nv_bfloat16* __restrict__ O, float* __restrict__ LSE, __nv_bfloat16* __restrict__ partial_o, float* __restrict__ partial_lse, int* __restrict__ work_table, int* __restrict__ unit_first, int* __restrict__ page_table, int* __restrict__ seq_lens, int* __restrict__ q_indptr, float* __restrict__ sinks, int rows_total, int num_heads, int q_len, int max_pages, int max_splits, float scale_log2)
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
    #define q_empty_addr (mbar_base + 8)
    #define raw_full_addr (mbar_base + 16)
    #define raw_empty_addr (mbar_base + 40)
    #define sfa_full_addr (mbar_base + 64)
    #define sf_full_addr (mbar_base + 72)
    #define sf_empty_addr (mbar_base + 88)
    #define s_full_addr (mbar_base + 104)
    #define s_empty_addr (mbar_base + 120)
    #define v_full_addr (mbar_base + 136)
    #define v_empty_addr (mbar_base + 152)
    #define corr_sig_addr (mbar_base + 168)
    #define corr_empty_addr (mbar_base + 176)
    #define p_full_addr (mbar_base + 184)
    #define p_empty_addr (mbar_base + 200)
    #define pv_done_addr (mbar_base + 216)
    #define o_full_addr (mbar_base + 224)
    #define o_empty_addr (mbar_base + 232)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    uint8_t* smem_q0 = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_q0_addr = smem + 1024;
    uint8_t* smem_q1 = reinterpret_cast<uint8_t*>(smem_raw + 5120);
    const int smem_q1_addr = smem + 5120;
    uint8_t* smem_q2 = reinterpret_cast<uint8_t*>(smem_raw + 9216);
    const int smem_q2_addr = smem + 9216;
    uint8_t* smem_q3 = reinterpret_cast<uint8_t*>(smem_raw + 13312);
    const int smem_q3_addr = smem + 13312;
    uint8_t* smem_qs = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int smem_qs_addr = smem + 17408;
    uint8_t* smem_k0h0 = reinterpret_cast<uint8_t*>(smem_raw + 19456);
    const int smem_k0h0_addr = smem + 19456;
    uint8_t* smem_k1h0 = reinterpret_cast<uint8_t*>(smem_raw + 23552);
    const int smem_k1h0_addr = smem + 23552;
    uint8_t* smem_k2h0 = reinterpret_cast<uint8_t*>(smem_raw + 27648);
    const int smem_k2h0_addr = smem + 27648;
    uint8_t* smem_k3h0 = reinterpret_cast<uint8_t*>(smem_raw + 31744);
    const int smem_k3h0_addr = smem + 31744;
    uint8_t* smem_k0h1 = reinterpret_cast<uint8_t*>(smem_raw + 21504);
    const int smem_k0h1_addr = smem + 21504;
    uint8_t* smem_k1h1 = reinterpret_cast<uint8_t*>(smem_raw + 25600);
    const int smem_k1h1_addr = smem + 25600;
    uint8_t* smem_k2h1 = reinterpret_cast<uint8_t*>(smem_raw + 29696);
    const int smem_k2h1_addr = smem + 29696;
    uint8_t* smem_k3h1 = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int smem_k3h1_addr = smem + 33792;
    uint8_t* smem_kq0h0 = reinterpret_cast<uint8_t*>(smem_raw + 19456);
    const int smem_kq0h0_addr = smem + 19456;
    uint8_t* smem_kq1h0 = reinterpret_cast<uint8_t*>(smem_raw + 23552);
    const int smem_kq1h0_addr = smem + 23552;
    uint8_t* smem_kq2h0 = reinterpret_cast<uint8_t*>(smem_raw + 27648);
    const int smem_kq2h0_addr = smem + 27648;
    uint8_t* smem_kq3h0 = reinterpret_cast<uint8_t*>(smem_raw + 31744);
    const int smem_kq3h0_addr = smem + 31744;
    uint8_t* smem_kq0h1 = reinterpret_cast<uint8_t*>(smem_raw + 21504);
    const int smem_kq0h1_addr = smem + 21504;
    uint8_t* smem_kq1h1 = reinterpret_cast<uint8_t*>(smem_raw + 25600);
    const int smem_kq1h1_addr = smem + 25600;
    uint8_t* smem_kq2h1 = reinterpret_cast<uint8_t*>(smem_raw + 29696);
    const int smem_kq2h1_addr = smem + 29696;
    uint8_t* smem_kq3h1 = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int smem_kq3h1_addr = smem + 33792;
    uint8_t* smem_ko0h0 = reinterpret_cast<uint8_t*>(smem_raw + 35840);
    const int smem_ko0h0_addr = smem + 35840;
    uint8_t* smem_ko1h0 = reinterpret_cast<uint8_t*>(smem_raw + 39936);
    const int smem_ko1h0_addr = smem + 39936;
    uint8_t* smem_ko0h1 = reinterpret_cast<uint8_t*>(smem_raw + 37888);
    const int smem_ko0h1_addr = smem + 37888;
    uint8_t* smem_ko1h1 = reinterpret_cast<uint8_t*>(smem_raw + 41984);
    const int smem_ko1h1_addr = smem + 41984;
    uint8_t* smem_ks = reinterpret_cast<uint8_t*>(smem_raw + 44032);
    const int smem_ks_addr = smem + 44032;
    uint8_t* smem_raw_flat = reinterpret_cast<uint8_t*>(smem_raw + 19456);
    const int smem_raw_flat_addr = smem + 19456;
    uint8_t* smem_v0 = reinterpret_cast<uint8_t*>(smem_raw + 105472);
    const int smem_v0_addr = smem + 105472;
    uint8_t* smem_v1 = reinterpret_cast<uint8_t*>(smem_raw + 121856);
    const int smem_v1_addr = smem + 121856;
    uint8_t* smem_p = reinterpret_cast<uint8_t*>(smem_raw + 171008);
    const int smem_p_addr = smem + 171008;
    float* row_state = reinterpret_cast<float*>(smem_raw + 187392);
    const int row_state_addr = smem + 187392;
    float* smem_xchg_max = reinterpret_cast<float*>(smem_raw + 187904);
    const int smem_xchg_max_addr = smem + 187904;
    float* smem_xchg_sum = reinterpret_cast<float*>(smem_raw + 188416);
    const int smem_xchg_sum_addr = smem + 188416;
    int* smem_pages = reinterpret_cast<int*>(smem_raw + 188928);
    const int smem_pages_addr = smem + 188928;
    uint8_t* smem_epi = reinterpret_cast<uint8_t*>(smem_raw + 189440);
    const int smem_epi_addr = smem + 189440;

    // Mbarrier init (18 pipeline groups, 0 ordered-sequence groups, 30 barriers)
    // Mbarriers at smem_raw[0..240)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // q_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // --- pipeline 'raw_pipe' ---
            // raw_full: 3 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            // raw_empty: 3 barriers, init_count=321
            mbarrier_init(smem + 40, 321);
            mbarrier_init(smem + 48, 321);
            mbarrier_init(smem + 56, 321);
            // sfa_full: 1 barriers, init_count=256
            mbarrier_init(smem + 64, 256);
            // --- pipeline 'sf_pipe' ---
            // sf_full: 2 barriers, init_count=256
            mbarrier_init(smem + 72, 256);
            mbarrier_init(smem + 80, 256);
            // sf_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            // --- pipeline 's_pipe' ---
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            // s_empty: 2 barriers, init_count=256
            mbarrier_init(smem + 120, 256);
            mbarrier_init(smem + 128, 256);
            // --- pipeline 'v_pipe' ---
            // v_full: 2 barriers, init_count=384
            mbarrier_init(smem + 136, 384);
            mbarrier_init(smem + 144, 384);
            // v_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            // corr_sig: 1 barriers, init_count=128
            mbarrier_init(smem + 168, 128);
            // corr_empty: 1 barriers, init_count=128
            mbarrier_init(smem + 176, 128);
            // --- pipeline 'p_pipe' ---
            // p_full: 2 barriers, init_count=512
            mbarrier_init(smem + 184, 512);
            mbarrier_init(smem + 192, 512);
            // p_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            // pv_done: 1 barriers, init_count=1
            mbarrier_init(smem + 216, 1);
            // o_full: 1 barriers, init_count=1
            mbarrier_init(smem + 224, 1);
            // o_empty: 1 barriers, init_count=256
            mbarrier_init(smem + 232, 256);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 480 used)
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
    const int tmem_tmem_acc = taddr;
    const int tmem_tmem_sfa0 = taddr + 320;
    const int tmem_tmem_sfa1 = taddr + 328;
    const int tmem_tmem_sfa2 = taddr + 336;
    const int tmem_tmem_sfa3 = taddr + 344;
    const int tmem_tmem_sfb0 = taddr + 352;
    const int tmem_tmem_sfb1 = taddr + 360;
    const int tmem_tmem_sfb2 = taddr + 368;
    const int tmem_tmem_sfb3 = taddr + 376;
    const int tmem_tmem_sfb4 = taddr + 384;
    const int tmem_tmem_sfb5 = taddr + 392;
    const int tmem_tmem_sfb6 = taddr + 400;
    const int tmem_tmem_sfb7 = taddr + 408;
    const int tmem_tmem_q = taddr + 416;
    if (warp >= 8 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 96;");
    }

    // ---- Role: softmax ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 208;");
        { // softmax_main
            int quadrant = make_warp_uniform(warp % 4);
            int n_half = make_warp_uniform(quadrant / 2);
            int row_c = quadrant % 2 * 32 + lane;
            int lane_base = quadrant * 32 << 16;
            int xchg_idx = n_half * 64 + row_c;
            int xchg_bar = make_warp_uniform(2 + quadrant % 2);
            int rank_s = cta_rank;
            int p_stage_s = 1;
            unsigned int _phase_p_empty = 0;
            p_stage_s += 1;
            if (p_stage_s == 2) { p_stage_s = 0; _phase_p_empty ^= 1; }
            int s_stage_s = 1;
            unsigned int _phase_s_full = 1;
            s_stage_s += 1;
            if (s_stage_s == 2) { s_stage_s = 0; _phase_s_full ^= 1; }
            int unit_s = cluster_id;
            int item_lo_s = unit_first[unit_s];
            int item_hi_s = unit_first[unit_s + 1];
            int tr_s = blockIdx.x == 0 && warp == 0 && lane == 0;
            int gt_s = 0;
            unsigned int _phase_corr_empty_0 = 1;
            unsigned int _phase_o_full_0 = 0;
            #pragma unroll 1
            for (int item_s = item_lo_s; item_s < item_hi_s; item_s += 2) {
                int b_s = work_table[item_s * 8];
                int m_tile_s = work_table[item_s * 8 + 1];
                int tile_start_s = work_table[item_s * 8 + 3];
                int tile_end_s = work_table[item_s * 8 + 4];
                int split_s = work_table[item_s * 8 + 5];
                int num_splits_s = work_table[item_s * 8 + 6];
                int flags_s = work_table[item_s * 8 + 7];
                int direct_out = (flags_s & 2) != 0;
                int num_tiles_s = tile_end_s - tile_start_s;
                int kv_len_s = seq_lens[b_s];
                int q_start_s = q_indptr[b_s];
                int row_in_req = m_tile_s * 128 + rank_s * 64 + row_c;
                int token_s = row_in_req / num_heads;
                int head_s = row_in_req - token_s * num_heads;
                int row_valid = row_in_req < q_len * num_heads;
                int row_limit = ((kv_len_s - q_len + token_s + 1) < (kv_len_s) ? (kv_len_s - q_len + token_s + 1) : (kv_len_s));
                float row_max = -CUDART_INF_F;
                float row_sum = 0.0f;
                if ((flags_s & 1) != 0) {
                    int sink_head = ((head_s) < (num_heads - 1) ? (head_s) : (num_heads - 1));
                    row_max = sinks[sink_head] * 1.442695f;
                    if (n_half == 0) {
                        row_sum = 64.0f;
                    }
                }
                #pragma unroll 1
                for (int it_s = 0; it_s < num_tiles_s; it_s++) {
                    float sv0[32];
                    float sv1[32];
                    mbarrier_wait(s_full_addr + (s_stage_s) * 8, _phase_s_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    {
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(sv0[0]), "=f"(sv0[1]), "=f"(sv0[2]), "=f"(sv0[3]), "=f"(sv0[4]), "=f"(sv0[5]), "=f"(sv0[6]), "=f"(sv0[7]), "=f"(sv0[8]), "=f"(sv0[9]), "=f"(sv0[10]), "=f"(sv0[11]), "=f"(sv0[12]), "=f"(sv0[13]), "=f"(sv0[14]), "=f"(sv0[15]), "=f"(sv0[16]), "=f"(sv0[17]), "=f"(sv0[18]), "=f"(sv0[19]), "=f"(sv0[20]), "=f"(sv0[21]), "=f"(sv0[22]), "=f"(sv0[23]), "=f"(sv0[24]), "=f"(sv0[25]), "=f"(sv0[26]), "=f"(sv0[27]), "=f"(sv0[28]), "=f"(sv0[29]), "=f"(sv0[30]), "=f"(sv0[31])
                            : "r"(taddr + (unsigned int)lane_base + (unsigned int)(s_stage_s * 32)));
                    }
                    s_stage_s += 1;
                    if (s_stage_s == 2) { s_stage_s = 0; _phase_s_full ^= 1; }
                    mbarrier_wait(s_full_addr + (s_stage_s) * 8, _phase_s_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    {
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(s_empty_addr + (((s_stage_s + 1 < 2) ? s_stage_s + 1 : (int)0)) * 8), "r"(0) : "memory");
                    {
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(sv1[0]), "=f"(sv1[1]), "=f"(sv1[2]), "=f"(sv1[3]), "=f"(sv1[4]), "=f"(sv1[5]), "=f"(sv1[6]), "=f"(sv1[7]), "=f"(sv1[8]), "=f"(sv1[9]), "=f"(sv1[10]), "=f"(sv1[11]), "=f"(sv1[12]), "=f"(sv1[13]), "=f"(sv1[14]), "=f"(sv1[15]), "=f"(sv1[16]), "=f"(sv1[17]), "=f"(sv1[18]), "=f"(sv1[19]), "=f"(sv1[20]), "=f"(sv1[21]), "=f"(sv1[22]), "=f"(sv1[23]), "=f"(sv1[24]), "=f"(sv1[25]), "=f"(sv1[26]), "=f"(sv1[27]), "=f"(sv1[28]), "=f"(sv1[29]), "=f"(sv1[30]), "=f"(sv1[31])
                            : "r"(taddr + (unsigned int)lane_base + (unsigned int)(s_stage_s * 32)));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(s_empty_addr + (s_stage_s) * 8), "r"(0) : "memory");
                    s_stage_s += 1;
                    if (s_stage_s == 2) { s_stage_s = 0; _phase_s_full ^= 1; }
                    int tile_base_s = (tile_start_s + it_s) * 128 + n_half * 32;
                    int _max_0 = ((row_limit - tile_base_s) > (0) ? (row_limit - tile_base_s) : (0));
                    int valid_0 = ((_max_0) < (32) ? (_max_0) : (32));
                    int _max_1 = ((row_limit - tile_base_s - 64) > (0) ? (row_limit - tile_base_s - 64) : (0));
                    int valid_1 = ((_max_1) < (32) ? (_max_1) : (32));
                    uint32_t _mbar_token_0 = mbarrier_test_wait(p_empty_addr + (p_stage_s) * 8, _phase_p_empty);
                    float tile_max = 0.0f;
                    {
                        if (valid_0 < 32) {
                            uint32_t _thresh_mask_2;
                            {
                                int _lim_3 = valid_0;
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
                            if (!(_thresh_mask_2 & (1u << 0))) sv0[0] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 1))) sv0[1] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 2))) sv0[2] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 3))) sv0[3] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 4))) sv0[4] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 5))) sv0[5] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 6))) sv0[6] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 7))) sv0[7] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 8))) sv0[8] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 9))) sv0[9] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 10))) sv0[10] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 11))) sv0[11] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 12))) sv0[12] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 13))) sv0[13] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 14))) sv0[14] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 15))) sv0[15] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 16))) sv0[16] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 17))) sv0[17] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 18))) sv0[18] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 19))) sv0[19] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 20))) sv0[20] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 21))) sv0[21] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 22))) sv0[22] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 23))) sv0[23] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 24))) sv0[24] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 25))) sv0[25] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 26))) sv0[26] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 27))) sv0[27] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 28))) sv0[28] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 29))) sv0[29] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 30))) sv0[30] = -CUDART_INF_F;
                            if (!(_thresh_mask_2 & (1u << 31))) sv0[31] = -CUDART_INF_F;
                        }
                        if (valid_1 < 32) {
                            uint32_t _thresh_mask_4;
                            {
                                int _lim_5 = valid_1;
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
                            if (!(_thresh_mask_4 & (1u << 0))) sv1[0] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 1))) sv1[1] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 2))) sv1[2] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 3))) sv1[3] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 4))) sv1[4] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 5))) sv1[5] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 6))) sv1[6] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 7))) sv1[7] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 8))) sv1[8] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 9))) sv1[9] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 10))) sv1[10] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 11))) sv1[11] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 12))) sv1[12] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 13))) sv1[13] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 14))) sv1[14] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 15))) sv1[15] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 16))) sv1[16] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 17))) sv1[17] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 18))) sv1[18] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 19))) sv1[19] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 20))) sv1[20] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 21))) sv1[21] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 22))) sv1[22] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 23))) sv1[23] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 24))) sv1[24] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 25))) sv1[25] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 26))) sv1[26] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 27))) sv1[27] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 28))) sv1[28] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 29))) sv1[29] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 30))) sv1[30] = -CUDART_INF_F;
                            if (!(_thresh_mask_4 & (1u << 31))) sv1[31] = -CUDART_INF_F;
                        }
                        float2 _reg_reduce_max2_7 = {-CAKE_INF, -CAKE_INF};
                        _reg_reduce_max2_7.x = max_noftz(_reg_reduce_max2_7.x, max_noftz(sv0[0], sv0[1]));
                        _reg_reduce_max2_7.y = max_noftz(_reg_reduce_max2_7.y, max_noftz(sv0[2], sv0[3]));
                        _reg_reduce_max2_7.x = max_noftz(_reg_reduce_max2_7.x, max_noftz(sv0[4], sv0[5]));
                        _reg_reduce_max2_7.y = max_noftz(_reg_reduce_max2_7.y, max_noftz(sv0[6], sv0[7]));
                        _reg_reduce_max2_7.x = max_noftz(_reg_reduce_max2_7.x, max_noftz(sv0[8], sv0[9]));
                        _reg_reduce_max2_7.y = max_noftz(_reg_reduce_max2_7.y, max_noftz(sv0[10], sv0[11]));
                        _reg_reduce_max2_7.x = max_noftz(_reg_reduce_max2_7.x, max_noftz(sv0[12], sv0[13]));
                        _reg_reduce_max2_7.y = max_noftz(_reg_reduce_max2_7.y, max_noftz(sv0[14], sv0[15]));
                        _reg_reduce_max2_7.x = max_noftz(_reg_reduce_max2_7.x, max_noftz(sv0[16], sv0[17]));
                        _reg_reduce_max2_7.y = max_noftz(_reg_reduce_max2_7.y, max_noftz(sv0[18], sv0[19]));
                        _reg_reduce_max2_7.x = max_noftz(_reg_reduce_max2_7.x, max_noftz(sv0[20], sv0[21]));
                        _reg_reduce_max2_7.y = max_noftz(_reg_reduce_max2_7.y, max_noftz(sv0[22], sv0[23]));
                        _reg_reduce_max2_7.x = max_noftz(_reg_reduce_max2_7.x, max_noftz(sv0[24], sv0[25]));
                        _reg_reduce_max2_7.y = max_noftz(_reg_reduce_max2_7.y, max_noftz(sv0[26], sv0[27]));
                        _reg_reduce_max2_7.x = max_noftz(_reg_reduce_max2_7.x, max_noftz(sv0[28], sv0[29]));
                        _reg_reduce_max2_7.y = max_noftz(_reg_reduce_max2_7.y, max_noftz(sv0[30], sv0[31]));
                        float _reg_array_reduce_6 = row_max_reduce(_reg_reduce_max2_7);
                        float2 _reg_reduce_max2_9 = {-CAKE_INF, -CAKE_INF};
                        _reg_reduce_max2_9.x = max_noftz(_reg_reduce_max2_9.x, max_noftz(sv1[0], sv1[1]));
                        _reg_reduce_max2_9.y = max_noftz(_reg_reduce_max2_9.y, max_noftz(sv1[2], sv1[3]));
                        _reg_reduce_max2_9.x = max_noftz(_reg_reduce_max2_9.x, max_noftz(sv1[4], sv1[5]));
                        _reg_reduce_max2_9.y = max_noftz(_reg_reduce_max2_9.y, max_noftz(sv1[6], sv1[7]));
                        _reg_reduce_max2_9.x = max_noftz(_reg_reduce_max2_9.x, max_noftz(sv1[8], sv1[9]));
                        _reg_reduce_max2_9.y = max_noftz(_reg_reduce_max2_9.y, max_noftz(sv1[10], sv1[11]));
                        _reg_reduce_max2_9.x = max_noftz(_reg_reduce_max2_9.x, max_noftz(sv1[12], sv1[13]));
                        _reg_reduce_max2_9.y = max_noftz(_reg_reduce_max2_9.y, max_noftz(sv1[14], sv1[15]));
                        _reg_reduce_max2_9.x = max_noftz(_reg_reduce_max2_9.x, max_noftz(sv1[16], sv1[17]));
                        _reg_reduce_max2_9.y = max_noftz(_reg_reduce_max2_9.y, max_noftz(sv1[18], sv1[19]));
                        _reg_reduce_max2_9.x = max_noftz(_reg_reduce_max2_9.x, max_noftz(sv1[20], sv1[21]));
                        _reg_reduce_max2_9.y = max_noftz(_reg_reduce_max2_9.y, max_noftz(sv1[22], sv1[23]));
                        _reg_reduce_max2_9.x = max_noftz(_reg_reduce_max2_9.x, max_noftz(sv1[24], sv1[25]));
                        _reg_reduce_max2_9.y = max_noftz(_reg_reduce_max2_9.y, max_noftz(sv1[26], sv1[27]));
                        _reg_reduce_max2_9.x = max_noftz(_reg_reduce_max2_9.x, max_noftz(sv1[28], sv1[29]));
                        _reg_reduce_max2_9.y = max_noftz(_reg_reduce_max2_9.y, max_noftz(sv1[30], sv1[31]));
                        float _reg_array_reduce_8 = row_max_reduce(_reg_reduce_max2_9);
                        tile_max = max_noftz(_reg_array_reduce_6, _reg_array_reduce_8);
                    }
                    float tile_max_scaled = ((tile_max > -CUDART_INF_F) ? tile_max * scale_log2 : -CUDART_INF_F);
                    float new_max;
                    asm("max.f32 %0, %1, %2;" : "=f"(new_max) : "f"(row_max), "f"(tile_max_scaled));
                    int xv_keep = row_max > -CUDART_INF_F && new_max - row_max <= 2.0f;
                    uint32_t xv_need;
                    asm volatile("{\n\t.reg .pred p, q;\n\tsetp.ne.u32 p, %1, 0;\n\tbarrier.red.or.pred q, %2, 64, p;\n\tselp.u32 %0, 1, 0, q;\n\t}" : "=r"(xv_need) : "r"((uint32_t)(xv_keep == 0)), "r"(xchg_bar) : "memory");
                    int _warp_vote_10 = __any_sync(0xFFFFFFFF, xv_need != 0);
                    if (_warp_vote_10 != 0) {
                        smem_xchg_max[xchg_idx] = new_max;
                        asm volatile("barrier.sync %0, 64;" :: "r"(xchg_bar) : "memory");
                        new_max = max_noftz(new_max, smem_xchg_max[xchg_idx ^ 64]);
                        asm volatile("barrier.sync %0, 64;" :: "r"(xchg_bar) : "memory");
                        if (row_max > -CUDART_INF_F && new_max - row_max <= 2.0f) {
                            new_max = row_max;
                        }
                    } else {
                        new_max = row_max;
                    }
                    float safe_max = ((new_max == -CUDART_INF_F) ? 0.0f : new_max);
                    float _exp2_11 = approx_exp2(row_max - safe_max);
                    float alpha = ((row_max > -CUDART_INF_F) ? _exp2_11 : 1.0f);
                    row_max = new_max;
                    mbarrier_wait(corr_empty_addr, _phase_corr_empty_0);
                    _phase_corr_empty_0 ^= 1;
                    row_state[row_c] = alpha;
                    mbarrier_arrive(corr_sig_addr);
                    mbarrier_wait_token(p_empty_addr + (p_stage_s) * 8, _phase_p_empty, _mbar_token_0);
                    float block_sum = 0.0f;
                    {
                        const float2 _fma_b2_12 = {scale_log2, scale_log2};
                        const float2 _fma_c2_13 = {6.0f - safe_max, 6.0f - safe_max};
                        float2 _fma_pair_14 = fma_f32x2(make_float2(sv0[0], sv0[1]), _fma_b2_12, _fma_c2_13);
                        sv0[0] = _fma_pair_14.x;
                        sv0[1] = _fma_pair_14.y;
                        float2 _fma_pair_15 = fma_f32x2(make_float2(sv0[2], sv0[3]), _fma_b2_12, _fma_c2_13);
                        sv0[2] = _fma_pair_15.x;
                        sv0[3] = _fma_pair_15.y;
                        float2 _fma_pair_16 = fma_f32x2(make_float2(sv0[4], sv0[5]), _fma_b2_12, _fma_c2_13);
                        sv0[4] = _fma_pair_16.x;
                        sv0[5] = _fma_pair_16.y;
                        float2 _fma_pair_17 = fma_f32x2(make_float2(sv0[6], sv0[7]), _fma_b2_12, _fma_c2_13);
                        sv0[6] = _fma_pair_17.x;
                        sv0[7] = _fma_pair_17.y;
                        float2 _fma_pair_18 = fma_f32x2(make_float2(sv0[8], sv0[9]), _fma_b2_12, _fma_c2_13);
                        sv0[8] = _fma_pair_18.x;
                        sv0[9] = _fma_pair_18.y;
                        float2 _fma_pair_19 = fma_f32x2(make_float2(sv0[10], sv0[11]), _fma_b2_12, _fma_c2_13);
                        sv0[10] = _fma_pair_19.x;
                        sv0[11] = _fma_pair_19.y;
                        float2 _fma_pair_20 = fma_f32x2(make_float2(sv0[12], sv0[13]), _fma_b2_12, _fma_c2_13);
                        sv0[12] = _fma_pair_20.x;
                        sv0[13] = _fma_pair_20.y;
                        float2 _fma_pair_21 = fma_f32x2(make_float2(sv0[14], sv0[15]), _fma_b2_12, _fma_c2_13);
                        sv0[14] = _fma_pair_21.x;
                        sv0[15] = _fma_pair_21.y;
                        float2 _fma_pair_22 = fma_f32x2(make_float2(sv0[16], sv0[17]), _fma_b2_12, _fma_c2_13);
                        sv0[16] = _fma_pair_22.x;
                        sv0[17] = _fma_pair_22.y;
                        float2 _fma_pair_23 = fma_f32x2(make_float2(sv0[18], sv0[19]), _fma_b2_12, _fma_c2_13);
                        sv0[18] = _fma_pair_23.x;
                        sv0[19] = _fma_pair_23.y;
                        float2 _fma_pair_24 = fma_f32x2(make_float2(sv0[20], sv0[21]), _fma_b2_12, _fma_c2_13);
                        sv0[20] = _fma_pair_24.x;
                        sv0[21] = _fma_pair_24.y;
                        float2 _fma_pair_25 = fma_f32x2(make_float2(sv0[22], sv0[23]), _fma_b2_12, _fma_c2_13);
                        sv0[22] = _fma_pair_25.x;
                        sv0[23] = _fma_pair_25.y;
                        float2 _fma_pair_26 = fma_f32x2(make_float2(sv0[24], sv0[25]), _fma_b2_12, _fma_c2_13);
                        sv0[24] = _fma_pair_26.x;
                        sv0[25] = _fma_pair_26.y;
                        float2 _fma_pair_27 = fma_f32x2(make_float2(sv0[26], sv0[27]), _fma_b2_12, _fma_c2_13);
                        sv0[26] = _fma_pair_27.x;
                        sv0[27] = _fma_pair_27.y;
                        float2 _fma_pair_28 = fma_f32x2(make_float2(sv0[28], sv0[29]), _fma_b2_12, _fma_c2_13);
                        sv0[28] = _fma_pair_28.x;
                        sv0[29] = _fma_pair_28.y;
                        float2 _fma_pair_29 = fma_f32x2(make_float2(sv0[30], sv0[31]), _fma_b2_12, _fma_c2_13);
                        sv0[30] = _fma_pair_29.x;
                        sv0[31] = _fma_pair_29.y;
                        const float2 _fma_b2_30 = {scale_log2, scale_log2};
                        const float2 _fma_c2_31 = {6.0f - safe_max, 6.0f - safe_max};
                        float2 _fma_pair_32 = fma_f32x2(make_float2(sv1[0], sv1[1]), _fma_b2_30, _fma_c2_31);
                        sv1[0] = _fma_pair_32.x;
                        sv1[1] = _fma_pair_32.y;
                        float2 _fma_pair_33 = fma_f32x2(make_float2(sv1[2], sv1[3]), _fma_b2_30, _fma_c2_31);
                        sv1[2] = _fma_pair_33.x;
                        sv1[3] = _fma_pair_33.y;
                        float2 _fma_pair_34 = fma_f32x2(make_float2(sv1[4], sv1[5]), _fma_b2_30, _fma_c2_31);
                        sv1[4] = _fma_pair_34.x;
                        sv1[5] = _fma_pair_34.y;
                        float2 _fma_pair_35 = fma_f32x2(make_float2(sv1[6], sv1[7]), _fma_b2_30, _fma_c2_31);
                        sv1[6] = _fma_pair_35.x;
                        sv1[7] = _fma_pair_35.y;
                        float2 _fma_pair_36 = fma_f32x2(make_float2(sv1[8], sv1[9]), _fma_b2_30, _fma_c2_31);
                        sv1[8] = _fma_pair_36.x;
                        sv1[9] = _fma_pair_36.y;
                        float2 _fma_pair_37 = fma_f32x2(make_float2(sv1[10], sv1[11]), _fma_b2_30, _fma_c2_31);
                        sv1[10] = _fma_pair_37.x;
                        sv1[11] = _fma_pair_37.y;
                        float2 _fma_pair_38 = fma_f32x2(make_float2(sv1[12], sv1[13]), _fma_b2_30, _fma_c2_31);
                        sv1[12] = _fma_pair_38.x;
                        sv1[13] = _fma_pair_38.y;
                        float2 _fma_pair_39 = fma_f32x2(make_float2(sv1[14], sv1[15]), _fma_b2_30, _fma_c2_31);
                        sv1[14] = _fma_pair_39.x;
                        sv1[15] = _fma_pair_39.y;
                        float2 _fma_pair_40 = fma_f32x2(make_float2(sv1[16], sv1[17]), _fma_b2_30, _fma_c2_31);
                        sv1[16] = _fma_pair_40.x;
                        sv1[17] = _fma_pair_40.y;
                        float2 _fma_pair_41 = fma_f32x2(make_float2(sv1[18], sv1[19]), _fma_b2_30, _fma_c2_31);
                        sv1[18] = _fma_pair_41.x;
                        sv1[19] = _fma_pair_41.y;
                        float2 _fma_pair_42 = fma_f32x2(make_float2(sv1[20], sv1[21]), _fma_b2_30, _fma_c2_31);
                        sv1[20] = _fma_pair_42.x;
                        sv1[21] = _fma_pair_42.y;
                        float2 _fma_pair_43 = fma_f32x2(make_float2(sv1[22], sv1[23]), _fma_b2_30, _fma_c2_31);
                        sv1[22] = _fma_pair_43.x;
                        sv1[23] = _fma_pair_43.y;
                        float2 _fma_pair_44 = fma_f32x2(make_float2(sv1[24], sv1[25]), _fma_b2_30, _fma_c2_31);
                        sv1[24] = _fma_pair_44.x;
                        sv1[25] = _fma_pair_44.y;
                        float2 _fma_pair_45 = fma_f32x2(make_float2(sv1[26], sv1[27]), _fma_b2_30, _fma_c2_31);
                        sv1[26] = _fma_pair_45.x;
                        sv1[27] = _fma_pair_45.y;
                        float2 _fma_pair_46 = fma_f32x2(make_float2(sv1[28], sv1[29]), _fma_b2_30, _fma_c2_31);
                        sv1[28] = _fma_pair_46.x;
                        sv1[29] = _fma_pair_46.y;
                        float2 _fma_pair_47 = fma_f32x2(make_float2(sv1[30], sv1[31]), _fma_b2_30, _fma_c2_31);
                        sv1[30] = _fma_pair_47.x;
                        sv1[31] = _fma_pair_47.y;
                        #pragma unroll
                        for (int _le = 0; _le < 32; _le++) {
                            sv0[_le] = approx_exp2(sv0[_le]);
                        }
                        #pragma unroll
                        for (int _le = 0; _le < 32; _le++) {
                            sv1[_le] = approx_exp2(sv1[_le]);
                        }
                        float2 _reg_reduce_sum2_49 = make_float2(0.0f, 0.0f);
                        _reg_reduce_sum2_49 = add_f32x2(_reg_reduce_sum2_49, make_float2(sv0[0], sv0[1]));
                        _reg_reduce_sum2_49 = add_f32x2(_reg_reduce_sum2_49, make_float2(sv0[2], sv0[3]));
                        _reg_reduce_sum2_49 = add_f32x2(_reg_reduce_sum2_49, make_float2(sv0[4], sv0[5]));
                        _reg_reduce_sum2_49 = add_f32x2(_reg_reduce_sum2_49, make_float2(sv0[6], sv0[7]));
                        _reg_reduce_sum2_49 = add_f32x2(_reg_reduce_sum2_49, make_float2(sv0[8], sv0[9]));
                        _reg_reduce_sum2_49 = add_f32x2(_reg_reduce_sum2_49, make_float2(sv0[10], sv0[11]));
                        _reg_reduce_sum2_49 = add_f32x2(_reg_reduce_sum2_49, make_float2(sv0[12], sv0[13]));
                        _reg_reduce_sum2_49 = add_f32x2(_reg_reduce_sum2_49, make_float2(sv0[14], sv0[15]));
                        _reg_reduce_sum2_49 = add_f32x2(_reg_reduce_sum2_49, make_float2(sv0[16], sv0[17]));
                        _reg_reduce_sum2_49 = add_f32x2(_reg_reduce_sum2_49, make_float2(sv0[18], sv0[19]));
                        _reg_reduce_sum2_49 = add_f32x2(_reg_reduce_sum2_49, make_float2(sv0[20], sv0[21]));
                        _reg_reduce_sum2_49 = add_f32x2(_reg_reduce_sum2_49, make_float2(sv0[22], sv0[23]));
                        _reg_reduce_sum2_49 = add_f32x2(_reg_reduce_sum2_49, make_float2(sv0[24], sv0[25]));
                        _reg_reduce_sum2_49 = add_f32x2(_reg_reduce_sum2_49, make_float2(sv0[26], sv0[27]));
                        _reg_reduce_sum2_49 = add_f32x2(_reg_reduce_sum2_49, make_float2(sv0[28], sv0[29]));
                        _reg_reduce_sum2_49 = add_f32x2(_reg_reduce_sum2_49, make_float2(sv0[30], sv0[31]));
                        float _reg_array_reduce_48 = _reg_reduce_sum2_49.x + _reg_reduce_sum2_49.y;
                        float2 _reg_reduce_sum2_51 = make_float2(0.0f, 0.0f);
                        _reg_reduce_sum2_51 = add_f32x2(_reg_reduce_sum2_51, make_float2(sv1[0], sv1[1]));
                        _reg_reduce_sum2_51 = add_f32x2(_reg_reduce_sum2_51, make_float2(sv1[2], sv1[3]));
                        _reg_reduce_sum2_51 = add_f32x2(_reg_reduce_sum2_51, make_float2(sv1[4], sv1[5]));
                        _reg_reduce_sum2_51 = add_f32x2(_reg_reduce_sum2_51, make_float2(sv1[6], sv1[7]));
                        _reg_reduce_sum2_51 = add_f32x2(_reg_reduce_sum2_51, make_float2(sv1[8], sv1[9]));
                        _reg_reduce_sum2_51 = add_f32x2(_reg_reduce_sum2_51, make_float2(sv1[10], sv1[11]));
                        _reg_reduce_sum2_51 = add_f32x2(_reg_reduce_sum2_51, make_float2(sv1[12], sv1[13]));
                        _reg_reduce_sum2_51 = add_f32x2(_reg_reduce_sum2_51, make_float2(sv1[14], sv1[15]));
                        _reg_reduce_sum2_51 = add_f32x2(_reg_reduce_sum2_51, make_float2(sv1[16], sv1[17]));
                        _reg_reduce_sum2_51 = add_f32x2(_reg_reduce_sum2_51, make_float2(sv1[18], sv1[19]));
                        _reg_reduce_sum2_51 = add_f32x2(_reg_reduce_sum2_51, make_float2(sv1[20], sv1[21]));
                        _reg_reduce_sum2_51 = add_f32x2(_reg_reduce_sum2_51, make_float2(sv1[22], sv1[23]));
                        _reg_reduce_sum2_51 = add_f32x2(_reg_reduce_sum2_51, make_float2(sv1[24], sv1[25]));
                        _reg_reduce_sum2_51 = add_f32x2(_reg_reduce_sum2_51, make_float2(sv1[26], sv1[27]));
                        _reg_reduce_sum2_51 = add_f32x2(_reg_reduce_sum2_51, make_float2(sv1[28], sv1[29]));
                        _reg_reduce_sum2_51 = add_f32x2(_reg_reduce_sum2_51, make_float2(sv1[30], sv1[31]));
                        float _reg_array_reduce_50 = _reg_reduce_sum2_51.x + _reg_reduce_sum2_51.y;
                        block_sum = _reg_array_reduce_48 + _reg_array_reduce_50;
                    }
                    row_sum = __fmaf_rn(row_sum, alpha, block_sum);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    {
                        int p_base = smem_p_addr + (unsigned int)(p_stage_s * 8192);
                        unsigned int _fp8_0[8];
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(sv0[0]), "f"(sv0[1]),
                                                   "f"(sv0[2]), "f"(sv0[3]));
                            _fp8_0[0] = _packed;
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
                                : "=r"(_packed) : "f"(sv0[4]), "f"(sv0[5]),
                                                   "f"(sv0[6]), "f"(sv0[7]));
                            _fp8_0[1] = _packed;
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
                                : "=r"(_packed) : "f"(sv0[8]), "f"(sv0[9]),
                                                   "f"(sv0[10]), "f"(sv0[11]));
                            _fp8_0[2] = _packed;
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
                                : "=r"(_packed) : "f"(sv0[12]), "f"(sv0[13]),
                                                   "f"(sv0[14]), "f"(sv0[15]));
                            _fp8_0[3] = _packed;
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
                                : "=r"(_packed) : "f"(sv0[16]), "f"(sv0[17]),
                                                   "f"(sv0[18]), "f"(sv0[19]));
                            _fp8_0[4] = _packed;
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
                                : "=r"(_packed) : "f"(sv0[20]), "f"(sv0[21]),
                                                   "f"(sv0[22]), "f"(sv0[23]));
                            _fp8_0[5] = _packed;
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
                                : "=r"(_packed) : "f"(sv0[24]), "f"(sv0[25]),
                                                   "f"(sv0[26]), "f"(sv0[27]));
                            _fp8_0[6] = _packed;
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
                                : "=r"(_packed) : "f"(sv0[28]), "f"(sv0[29]),
                                                   "f"(sv0[30]), "f"(sv0[31]));
                            _fp8_0[7] = _packed;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p_base + (row_c * 128 + n_half * 32 ^ (row_c * 128 + n_half * 32 >> 7 & 7) << 4))), "r"(_fp8_0[0]), "r"(_fp8_0[1]), "r"(_fp8_0[2]), "r"(_fp8_0[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p_base + (row_c * 128 + (n_half * 32 + 16) ^ (row_c * 128 + (n_half * 32 + 16) >> 7 & 7) << 4))), "r"(_fp8_0[4]), "r"(_fp8_0[5]), "r"(_fp8_0[6]), "r"(_fp8_0[7]) : "memory");
                        unsigned int _fp8_1[8];
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(sv1[0]), "f"(sv1[1]),
                                                   "f"(sv1[2]), "f"(sv1[3]));
                            _fp8_1[0] = _packed;
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
                                : "=r"(_packed) : "f"(sv1[4]), "f"(sv1[5]),
                                                   "f"(sv1[6]), "f"(sv1[7]));
                            _fp8_1[1] = _packed;
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
                                : "=r"(_packed) : "f"(sv1[8]), "f"(sv1[9]),
                                                   "f"(sv1[10]), "f"(sv1[11]));
                            _fp8_1[2] = _packed;
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
                                : "=r"(_packed) : "f"(sv1[12]), "f"(sv1[13]),
                                                   "f"(sv1[14]), "f"(sv1[15]));
                            _fp8_1[3] = _packed;
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
                                : "=r"(_packed) : "f"(sv1[16]), "f"(sv1[17]),
                                                   "f"(sv1[18]), "f"(sv1[19]));
                            _fp8_1[4] = _packed;
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
                                : "=r"(_packed) : "f"(sv1[20]), "f"(sv1[21]),
                                                   "f"(sv1[22]), "f"(sv1[23]));
                            _fp8_1[5] = _packed;
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
                                : "=r"(_packed) : "f"(sv1[24]), "f"(sv1[25]),
                                                   "f"(sv1[26]), "f"(sv1[27]));
                            _fp8_1[6] = _packed;
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
                                : "=r"(_packed) : "f"(sv1[28]), "f"(sv1[29]),
                                                   "f"(sv1[30]), "f"(sv1[31]));
                            _fp8_1[7] = _packed;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p_base + (row_c * 128 + (64 + n_half * 32) ^ (row_c * 128 + (64 + n_half * 32) >> 7 & 7) << 4))), "r"(_fp8_1[0]), "r"(_fp8_1[1]), "r"(_fp8_1[2]), "r"(_fp8_1[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p_base + (row_c * 128 + (64 + n_half * 32 + 16) ^ (row_c * 128 + (64 + n_half * 32 + 16) >> 7 & 7) << 4))), "r"(_fp8_1[4]), "r"(_fp8_1[5]), "r"(_fp8_1[6]), "r"(_fp8_1[7]) : "memory");
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(p_full_addr + (p_stage_s) * 8), "r"(0) : "memory");
                    p_stage_s += 1;
                    if (p_stage_s == 2) { p_stage_s = 0; _phase_p_empty ^= 1; }
                }
                mbarrier_wait(o_full_addr, _phase_o_full_0);
                _phase_o_full_0 ^= 1;
                smem_xchg_sum[xchg_idx] = row_sum;
                asm volatile("barrier.sync %0, 64;" :: "r"(xchg_bar) : "memory");
                row_sum = row_sum + smem_xchg_sum[xchg_idx ^ 64];
                asm volatile("barrier.sync %0, 64;" :: "r"(xchg_bar) : "memory");
                asm volatile("tcgen05.fence::after_thread_sync;");
                float _rcp_52 = approx_rcp(row_sum);
                float inv_sum = ((row_sum > 0.0f) ? _rcp_52 : 0.0f);
                int out_row = (q_start_s + token_s) * num_heads + head_s;
                int o_base = out_row * 512 + n_half * 128;
                int part_base = (split_s * rows_total + out_row) * 512 + n_half * 128;
                float _log2_53;
                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_53) : "f"(row_sum));
                float lse2_s = row_max + _log2_53 - 6.0f;
                int _max_54 = ((q_len * num_heads - (row_in_req - row_c)) > (0) ? (q_len * num_heads - (row_in_req - row_c)) : (0));
                int valid_rows_cta = ((_max_54) < (64) ? (_max_54) : (64));
                int out_row_base = out_row - row_c;
                int epi_bulk = valid_rows_cta == 64;
                int warp_tile0 = smem_p_addr + (unsigned int)(quadrant * 4096);
                int warp_epi_base = smem_epi_addr + (unsigned int)(quadrant * 8192);
                int warp_tile1 = warp_epi_base;
                int warp_tile2 = warp_epi_base + 4096;
                int warp_row0 = out_row_base + quadrant % 2 * 32;
                if (epi_bulk != 0) {
                    if (elect_sync()) {
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                    }
                    __syncwarp();
                    asm volatile("tcgen05.fence::after_thread_sync;");
                }
                float o_ra[32];
                float o_rb[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(o_ra[0]), "=f"(o_ra[1]), "=f"(o_ra[2]), "=f"(o_ra[3]), "=f"(o_ra[4]), "=f"(o_ra[5]), "=f"(o_ra[6]), "=f"(o_ra[7]), "=f"(o_ra[8]), "=f"(o_ra[9]), "=f"(o_ra[10]), "=f"(o_ra[11]), "=f"(o_ra[12]), "=f"(o_ra[13]), "=f"(o_ra[14]), "=f"(o_ra[15]), "=f"(o_ra[16]), "=f"(o_ra[17]), "=f"(o_ra[18]), "=f"(o_ra[19]), "=f"(o_ra[20]), "=f"(o_ra[21]), "=f"(o_ra[22]), "=f"(o_ra[23]), "=f"(o_ra[24]), "=f"(o_ra[25]), "=f"(o_ra[26]), "=f"(o_ra[27]), "=f"(o_ra[28]), "=f"(o_ra[29]), "=f"(o_ra[30]), "=f"(o_ra[31])
                    : "r"(taddr + (unsigned int)lane_base + 64));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(o_rb[0]), "=f"(o_rb[1]), "=f"(o_rb[2]), "=f"(o_rb[3]), "=f"(o_rb[4]), "=f"(o_rb[5]), "=f"(o_rb[6]), "=f"(o_rb[7]), "=f"(o_rb[8]), "=f"(o_rb[9]), "=f"(o_rb[10]), "=f"(o_rb[11]), "=f"(o_rb[12]), "=f"(o_rb[13]), "=f"(o_rb[14]), "=f"(o_rb[15]), "=f"(o_rb[16]), "=f"(o_rb[17]), "=f"(o_rb[18]), "=f"(o_rb[19]), "=f"(o_rb[20]), "=f"(o_rb[21]), "=f"(o_rb[22]), "=f"(o_rb[23]), "=f"(o_rb[24]), "=f"(o_rb[25]), "=f"(o_rb[26]), "=f"(o_rb[27]), "=f"(o_rb[28]), "=f"(o_rb[29]), "=f"(o_rb[30]), "=f"(o_rb[31])
                    : "r"(taddr + (unsigned int)lane_base + 64 + 32));
                if (epi_bulk != 0) {
                    {
                        const float2 _scale2_55 = {inv_sum, inv_sum};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_ra)[_ls], _scale2_55);
                        unsigned int o_ra_bf16[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_ra[_lp*2 + 0], o_ra[_lp*2+1 + 0]));
                            o_ra_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile0 + (lane * 128 ^ (lane * 128 >> 7 & 7) << 4))), "r"(o_ra_bf16[0]), "r"(o_ra_bf16[1]), "r"(o_ra_bf16[2]), "r"(o_ra_bf16[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile0 + (lane * 128 + 16 ^ (lane * 128 + 16 >> 7 & 7) << 4))), "r"(o_ra_bf16[4]), "r"(o_ra_bf16[5]), "r"(o_ra_bf16[6]), "r"(o_ra_bf16[7]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile0 + (lane * 128 + 32 ^ (lane * 128 + 32 >> 7 & 7) << 4))), "r"(o_ra_bf16[8]), "r"(o_ra_bf16[9]), "r"(o_ra_bf16[10]), "r"(o_ra_bf16[11]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile0 + (lane * 128 + 48 ^ (lane * 128 + 48 >> 7 & 7) << 4))), "r"(o_ra_bf16[12]), "r"(o_ra_bf16[13]), "r"(o_ra_bf16[14]), "r"(o_ra_bf16[15]) : "memory");
                    }
                } else if (row_valid != 0) {
                    if (direct_out != 0) {
                        float _scaled_0[16];
                        const float2 _scale2_56 = {inv_sum, inv_sum};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            reinterpret_cast<float2*>(_scaled_0)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>(o_ra)[_ls], _scale2_56);
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
                        reinterpret_cast<int4*>(O + o_base)[0] = reinterpret_cast<int4*>(_packed_0)[0];
                        reinterpret_cast<int4*>(O + (o_base + 8))[0] = reinterpret_cast<int4*>(_packed_0 + (4))[0];
                        float _scaled_1[16];
                        const float2 _scale2_57 = {inv_sum, inv_sum};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            reinterpret_cast<float2*>(_scaled_1)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((o_ra + 16))[_ls], _scale2_57);
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
                        reinterpret_cast<int4*>(O + (o_base + 16))[0] = reinterpret_cast<int4*>(_packed_1)[0];
                        reinterpret_cast<int4*>(O + (o_base + 16 + 8))[0] = reinterpret_cast<int4*>(_packed_1 + (4))[0];
                    } else {
                        float _scaled_2[16];
                        const float2 _scale2_58 = {inv_sum, inv_sum};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            reinterpret_cast<float2*>(_scaled_2)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>(o_ra)[_ls], _scale2_58);
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
                        reinterpret_cast<int4*>(partial_o + part_base)[0] = reinterpret_cast<int4*>(_packed_2)[0];
                        reinterpret_cast<int4*>(partial_o + (part_base + 8))[0] = reinterpret_cast<int4*>(_packed_2 + (4))[0];
                        float _scaled_3[16];
                        const float2 _scale2_59 = {inv_sum, inv_sum};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            reinterpret_cast<float2*>(_scaled_3)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((o_ra + 16))[_ls], _scale2_59);
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
                        reinterpret_cast<int4*>(partial_o + (part_base + 16))[0] = reinterpret_cast<int4*>(_packed_3)[0];
                        reinterpret_cast<int4*>(partial_o + (part_base + 16 + 8))[0] = reinterpret_cast<int4*>(_packed_3 + (4))[0];
                    }
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(o_ra[0]), "=f"(o_ra[1]), "=f"(o_ra[2]), "=f"(o_ra[3]), "=f"(o_ra[4]), "=f"(o_ra[5]), "=f"(o_ra[6]), "=f"(o_ra[7]), "=f"(o_ra[8]), "=f"(o_ra[9]), "=f"(o_ra[10]), "=f"(o_ra[11]), "=f"(o_ra[12]), "=f"(o_ra[13]), "=f"(o_ra[14]), "=f"(o_ra[15]), "=f"(o_ra[16]), "=f"(o_ra[17]), "=f"(o_ra[18]), "=f"(o_ra[19]), "=f"(o_ra[20]), "=f"(o_ra[21]), "=f"(o_ra[22]), "=f"(o_ra[23]), "=f"(o_ra[24]), "=f"(o_ra[25]), "=f"(o_ra[26]), "=f"(o_ra[27]), "=f"(o_ra[28]), "=f"(o_ra[29]), "=f"(o_ra[30]), "=f"(o_ra[31])
                    : "r"(taddr + (unsigned int)lane_base + 64 + 64));
                if (epi_bulk != 0) {
                    {
                        const float2 _scale2_60 = {inv_sum, inv_sum};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_rb)[_ls], _scale2_60);
                        unsigned int o_rb_bf16[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_rb[_lp*2 + 0], o_rb[_lp*2+1 + 0]));
                            o_rb_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile0 + (lane * 128 + 64 ^ (lane * 128 + 64 >> 7 & 7) << 4))), "r"(o_rb_bf16[0]), "r"(o_rb_bf16[1]), "r"(o_rb_bf16[2]), "r"(o_rb_bf16[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile0 + (lane * 128 + 80 ^ (lane * 128 + 80 >> 7 & 7) << 4))), "r"(o_rb_bf16[4]), "r"(o_rb_bf16[5]), "r"(o_rb_bf16[6]), "r"(o_rb_bf16[7]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile0 + (lane * 128 + 96 ^ (lane * 128 + 96 >> 7 & 7) << 4))), "r"(o_rb_bf16[8]), "r"(o_rb_bf16[9]), "r"(o_rb_bf16[10]), "r"(o_rb_bf16[11]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile0 + (lane * 128 + 112 ^ (lane * 128 + 112 >> 7 & 7) << 4))), "r"(o_rb_bf16[12]), "r"(o_rb_bf16[13]), "r"(o_rb_bf16[14]), "r"(o_rb_bf16[15]) : "memory");
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        __syncwarp();
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        if (elect_sync()) {
                            if (direct_out != 0) {
                                tma_store_2d((&OT), n_half * 128 + 32 - 32, warp_row0, warp_tile0);
                            } else {
                                tma_store_2d((&PO), n_half * 128 + 32 - 32, split_s * rows_total + warp_row0, warp_tile0);
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    }
                } else if (row_valid != 0) {
                    if (direct_out != 0) {
                        float _scaled_4[16];
                        const float2 _scale2_61 = {inv_sum, inv_sum};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            reinterpret_cast<float2*>(_scaled_4)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>(o_rb)[_ls], _scale2_61);
                        unsigned int _packed_4[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_4[_lp*2 + 0], _scaled_4[_lp*2+1 + 0]));
                            _packed_4[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_4[_lp*2 + 2], _scaled_4[_lp*2+1 + 2]));
                            _packed_4[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_4[_lp*2 + 4], _scaled_4[_lp*2+1 + 4]));
                            _packed_4[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_4[_lp*2 + 6], _scaled_4[_lp*2+1 + 6]));
                            _packed_4[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_4[_lp*2 + 8], _scaled_4[_lp*2+1 + 8]));
                            _packed_4[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_4[_lp*2 + 10], _scaled_4[_lp*2+1 + 10]));
                            _packed_4[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_4[_lp*2 + 12], _scaled_4[_lp*2+1 + 12]));
                            _packed_4[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_4[_lp*2 + 14], _scaled_4[_lp*2+1 + 14]));
                            _packed_4[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(O + (o_base + 32))[0] = reinterpret_cast<int4*>(_packed_4)[0];
                        reinterpret_cast<int4*>(O + (o_base + 32 + 8))[0] = reinterpret_cast<int4*>(_packed_4 + (4))[0];
                        float _scaled_5[16];
                        const float2 _scale2_62 = {inv_sum, inv_sum};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            reinterpret_cast<float2*>(_scaled_5)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((o_rb + 16))[_ls], _scale2_62);
                        unsigned int _packed_5[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_5[_lp*2 + 0], _scaled_5[_lp*2+1 + 0]));
                            _packed_5[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_5[_lp*2 + 2], _scaled_5[_lp*2+1 + 2]));
                            _packed_5[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_5[_lp*2 + 4], _scaled_5[_lp*2+1 + 4]));
                            _packed_5[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_5[_lp*2 + 6], _scaled_5[_lp*2+1 + 6]));
                            _packed_5[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_5[_lp*2 + 8], _scaled_5[_lp*2+1 + 8]));
                            _packed_5[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_5[_lp*2 + 10], _scaled_5[_lp*2+1 + 10]));
                            _packed_5[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_5[_lp*2 + 12], _scaled_5[_lp*2+1 + 12]));
                            _packed_5[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_5[_lp*2 + 14], _scaled_5[_lp*2+1 + 14]));
                            _packed_5[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(O + (o_base + 32 + 16))[0] = reinterpret_cast<int4*>(_packed_5)[0];
                        reinterpret_cast<int4*>(O + (o_base + 32 + 16 + 8))[0] = reinterpret_cast<int4*>(_packed_5 + (4))[0];
                    } else {
                        float _scaled_6[16];
                        const float2 _scale2_63 = {inv_sum, inv_sum};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            reinterpret_cast<float2*>(_scaled_6)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>(o_rb)[_ls], _scale2_63);
                        unsigned int _packed_6[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_6[_lp*2 + 0], _scaled_6[_lp*2+1 + 0]));
                            _packed_6[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_6[_lp*2 + 2], _scaled_6[_lp*2+1 + 2]));
                            _packed_6[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_6[_lp*2 + 4], _scaled_6[_lp*2+1 + 4]));
                            _packed_6[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_6[_lp*2 + 6], _scaled_6[_lp*2+1 + 6]));
                            _packed_6[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_6[_lp*2 + 8], _scaled_6[_lp*2+1 + 8]));
                            _packed_6[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_6[_lp*2 + 10], _scaled_6[_lp*2+1 + 10]));
                            _packed_6[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_6[_lp*2 + 12], _scaled_6[_lp*2+1 + 12]));
                            _packed_6[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_6[_lp*2 + 14], _scaled_6[_lp*2+1 + 14]));
                            _packed_6[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(partial_o + (part_base + 32))[0] = reinterpret_cast<int4*>(_packed_6)[0];
                        reinterpret_cast<int4*>(partial_o + (part_base + 32 + 8))[0] = reinterpret_cast<int4*>(_packed_6 + (4))[0];
                        float _scaled_7[16];
                        const float2 _scale2_64 = {inv_sum, inv_sum};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            reinterpret_cast<float2*>(_scaled_7)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((o_rb + 16))[_ls], _scale2_64);
                        unsigned int _packed_7[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_7[_lp*2 + 0], _scaled_7[_lp*2+1 + 0]));
                            _packed_7[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_7[_lp*2 + 2], _scaled_7[_lp*2+1 + 2]));
                            _packed_7[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_7[_lp*2 + 4], _scaled_7[_lp*2+1 + 4]));
                            _packed_7[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_7[_lp*2 + 6], _scaled_7[_lp*2+1 + 6]));
                            _packed_7[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_7[_lp*2 + 8], _scaled_7[_lp*2+1 + 8]));
                            _packed_7[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_7[_lp*2 + 10], _scaled_7[_lp*2+1 + 10]));
                            _packed_7[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_7[_lp*2 + 12], _scaled_7[_lp*2+1 + 12]));
                            _packed_7[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_7[_lp*2 + 14], _scaled_7[_lp*2+1 + 14]));
                            _packed_7[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(partial_o + (part_base + 32 + 16))[0] = reinterpret_cast<int4*>(_packed_7)[0];
                        reinterpret_cast<int4*>(partial_o + (part_base + 32 + 16 + 8))[0] = reinterpret_cast<int4*>(_packed_7 + (4))[0];
                    }
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(o_rb[0]), "=f"(o_rb[1]), "=f"(o_rb[2]), "=f"(o_rb[3]), "=f"(o_rb[4]), "=f"(o_rb[5]), "=f"(o_rb[6]), "=f"(o_rb[7]), "=f"(o_rb[8]), "=f"(o_rb[9]), "=f"(o_rb[10]), "=f"(o_rb[11]), "=f"(o_rb[12]), "=f"(o_rb[13]), "=f"(o_rb[14]), "=f"(o_rb[15]), "=f"(o_rb[16]), "=f"(o_rb[17]), "=f"(o_rb[18]), "=f"(o_rb[19]), "=f"(o_rb[20]), "=f"(o_rb[21]), "=f"(o_rb[22]), "=f"(o_rb[23]), "=f"(o_rb[24]), "=f"(o_rb[25]), "=f"(o_rb[26]), "=f"(o_rb[27]), "=f"(o_rb[28]), "=f"(o_rb[29]), "=f"(o_rb[30]), "=f"(o_rb[31])
                    : "r"(taddr + (unsigned int)lane_base + 64 + 96));
                if (epi_bulk != 0) {
                    {
                        const float2 _scale2_65 = {inv_sum, inv_sum};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_ra)[_ls], _scale2_65);
                        unsigned int o_ra_bf16_1[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_ra[_lp*2 + 0], o_ra[_lp*2+1 + 0]));
                            o_ra_bf16_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile1 + (lane * 128 ^ (lane * 128 >> 7 & 7) << 4))), "r"(o_ra_bf16_1[0]), "r"(o_ra_bf16_1[1]), "r"(o_ra_bf16_1[2]), "r"(o_ra_bf16_1[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile1 + (lane * 128 + 16 ^ (lane * 128 + 16 >> 7 & 7) << 4))), "r"(o_ra_bf16_1[4]), "r"(o_ra_bf16_1[5]), "r"(o_ra_bf16_1[6]), "r"(o_ra_bf16_1[7]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile1 + (lane * 128 + 32 ^ (lane * 128 + 32 >> 7 & 7) << 4))), "r"(o_ra_bf16_1[8]), "r"(o_ra_bf16_1[9]), "r"(o_ra_bf16_1[10]), "r"(o_ra_bf16_1[11]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile1 + (lane * 128 + 48 ^ (lane * 128 + 48 >> 7 & 7) << 4))), "r"(o_ra_bf16_1[12]), "r"(o_ra_bf16_1[13]), "r"(o_ra_bf16_1[14]), "r"(o_ra_bf16_1[15]) : "memory");
                    }
                } else if (row_valid != 0) {
                    if (direct_out != 0) {
                        float _scaled_8[16];
                        const float2 _scale2_66 = {inv_sum, inv_sum};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            reinterpret_cast<float2*>(_scaled_8)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>(o_ra)[_ls], _scale2_66);
                        unsigned int _packed_8[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_8[_lp*2 + 0], _scaled_8[_lp*2+1 + 0]));
                            _packed_8[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_8[_lp*2 + 2], _scaled_8[_lp*2+1 + 2]));
                            _packed_8[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_8[_lp*2 + 4], _scaled_8[_lp*2+1 + 4]));
                            _packed_8[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_8[_lp*2 + 6], _scaled_8[_lp*2+1 + 6]));
                            _packed_8[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_8[_lp*2 + 8], _scaled_8[_lp*2+1 + 8]));
                            _packed_8[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_8[_lp*2 + 10], _scaled_8[_lp*2+1 + 10]));
                            _packed_8[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_8[_lp*2 + 12], _scaled_8[_lp*2+1 + 12]));
                            _packed_8[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_8[_lp*2 + 14], _scaled_8[_lp*2+1 + 14]));
                            _packed_8[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(O + (o_base + 64))[0] = reinterpret_cast<int4*>(_packed_8)[0];
                        reinterpret_cast<int4*>(O + (o_base + 64 + 8))[0] = reinterpret_cast<int4*>(_packed_8 + (4))[0];
                        float _scaled_9[16];
                        const float2 _scale2_67 = {inv_sum, inv_sum};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            reinterpret_cast<float2*>(_scaled_9)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((o_ra + 16))[_ls], _scale2_67);
                        unsigned int _packed_9[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_9[_lp*2 + 0], _scaled_9[_lp*2+1 + 0]));
                            _packed_9[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_9[_lp*2 + 2], _scaled_9[_lp*2+1 + 2]));
                            _packed_9[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_9[_lp*2 + 4], _scaled_9[_lp*2+1 + 4]));
                            _packed_9[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_9[_lp*2 + 6], _scaled_9[_lp*2+1 + 6]));
                            _packed_9[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_9[_lp*2 + 8], _scaled_9[_lp*2+1 + 8]));
                            _packed_9[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_9[_lp*2 + 10], _scaled_9[_lp*2+1 + 10]));
                            _packed_9[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_9[_lp*2 + 12], _scaled_9[_lp*2+1 + 12]));
                            _packed_9[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_9[_lp*2 + 14], _scaled_9[_lp*2+1 + 14]));
                            _packed_9[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(O + (o_base + 64 + 16))[0] = reinterpret_cast<int4*>(_packed_9)[0];
                        reinterpret_cast<int4*>(O + (o_base + 64 + 16 + 8))[0] = reinterpret_cast<int4*>(_packed_9 + (4))[0];
                    } else {
                        float _scaled_10[16];
                        const float2 _scale2_68 = {inv_sum, inv_sum};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            reinterpret_cast<float2*>(_scaled_10)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>(o_ra)[_ls], _scale2_68);
                        unsigned int _packed_10[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_10[_lp*2 + 0], _scaled_10[_lp*2+1 + 0]));
                            _packed_10[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_10[_lp*2 + 2], _scaled_10[_lp*2+1 + 2]));
                            _packed_10[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_10[_lp*2 + 4], _scaled_10[_lp*2+1 + 4]));
                            _packed_10[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_10[_lp*2 + 6], _scaled_10[_lp*2+1 + 6]));
                            _packed_10[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_10[_lp*2 + 8], _scaled_10[_lp*2+1 + 8]));
                            _packed_10[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_10[_lp*2 + 10], _scaled_10[_lp*2+1 + 10]));
                            _packed_10[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_10[_lp*2 + 12], _scaled_10[_lp*2+1 + 12]));
                            _packed_10[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_10[_lp*2 + 14], _scaled_10[_lp*2+1 + 14]));
                            _packed_10[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(partial_o + (part_base + 64))[0] = reinterpret_cast<int4*>(_packed_10)[0];
                        reinterpret_cast<int4*>(partial_o + (part_base + 64 + 8))[0] = reinterpret_cast<int4*>(_packed_10 + (4))[0];
                        float _scaled_11[16];
                        const float2 _scale2_69 = {inv_sum, inv_sum};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            reinterpret_cast<float2*>(_scaled_11)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((o_ra + 16))[_ls], _scale2_69);
                        unsigned int _packed_11[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_11[_lp*2 + 0], _scaled_11[_lp*2+1 + 0]));
                            _packed_11[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_11[_lp*2 + 2], _scaled_11[_lp*2+1 + 2]));
                            _packed_11[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_11[_lp*2 + 4], _scaled_11[_lp*2+1 + 4]));
                            _packed_11[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_11[_lp*2 + 6], _scaled_11[_lp*2+1 + 6]));
                            _packed_11[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_11[_lp*2 + 8], _scaled_11[_lp*2+1 + 8]));
                            _packed_11[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_11[_lp*2 + 10], _scaled_11[_lp*2+1 + 10]));
                            _packed_11[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_11[_lp*2 + 12], _scaled_11[_lp*2+1 + 12]));
                            _packed_11[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_11[_lp*2 + 14], _scaled_11[_lp*2+1 + 14]));
                            _packed_11[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(partial_o + (part_base + 64 + 16))[0] = reinterpret_cast<int4*>(_packed_11)[0];
                        reinterpret_cast<int4*>(partial_o + (part_base + 64 + 16 + 8))[0] = reinterpret_cast<int4*>(_packed_11 + (4))[0];
                    }
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(o_ra[0]), "=f"(o_ra[1]), "=f"(o_ra[2]), "=f"(o_ra[3]), "=f"(o_ra[4]), "=f"(o_ra[5]), "=f"(o_ra[6]), "=f"(o_ra[7]), "=f"(o_ra[8]), "=f"(o_ra[9]), "=f"(o_ra[10]), "=f"(o_ra[11]), "=f"(o_ra[12]), "=f"(o_ra[13]), "=f"(o_ra[14]), "=f"(o_ra[15]), "=f"(o_ra[16]), "=f"(o_ra[17]), "=f"(o_ra[18]), "=f"(o_ra[19]), "=f"(o_ra[20]), "=f"(o_ra[21]), "=f"(o_ra[22]), "=f"(o_ra[23]), "=f"(o_ra[24]), "=f"(o_ra[25]), "=f"(o_ra[26]), "=f"(o_ra[27]), "=f"(o_ra[28]), "=f"(o_ra[29]), "=f"(o_ra[30]), "=f"(o_ra[31])
                    : "r"(taddr + (unsigned int)lane_base + 64 + 128));
                if (epi_bulk != 0) {
                    {
                        const float2 _scale2_70 = {inv_sum, inv_sum};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_rb)[_ls], _scale2_70);
                        unsigned int o_rb_bf16_1[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_rb[_lp*2 + 0], o_rb[_lp*2+1 + 0]));
                            o_rb_bf16_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile1 + (lane * 128 + 64 ^ (lane * 128 + 64 >> 7 & 7) << 4))), "r"(o_rb_bf16_1[0]), "r"(o_rb_bf16_1[1]), "r"(o_rb_bf16_1[2]), "r"(o_rb_bf16_1[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile1 + (lane * 128 + 80 ^ (lane * 128 + 80 >> 7 & 7) << 4))), "r"(o_rb_bf16_1[4]), "r"(o_rb_bf16_1[5]), "r"(o_rb_bf16_1[6]), "r"(o_rb_bf16_1[7]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile1 + (lane * 128 + 96 ^ (lane * 128 + 96 >> 7 & 7) << 4))), "r"(o_rb_bf16_1[8]), "r"(o_rb_bf16_1[9]), "r"(o_rb_bf16_1[10]), "r"(o_rb_bf16_1[11]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile1 + (lane * 128 + 112 ^ (lane * 128 + 112 >> 7 & 7) << 4))), "r"(o_rb_bf16_1[12]), "r"(o_rb_bf16_1[13]), "r"(o_rb_bf16_1[14]), "r"(o_rb_bf16_1[15]) : "memory");
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        __syncwarp();
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        if (elect_sync()) {
                            if (direct_out != 0) {
                                tma_store_2d((&OT), n_half * 128 + 96 - 32, warp_row0, warp_tile1);
                            } else {
                                tma_store_2d((&PO), n_half * 128 + 96 - 32, split_s * rows_total + warp_row0, warp_tile1);
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    }
                } else if (row_valid != 0) {
                    if (direct_out != 0) {
                        float _scaled_12[16];
                        const float2 _scale2_71 = {inv_sum, inv_sum};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            reinterpret_cast<float2*>(_scaled_12)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>(o_rb)[_ls], _scale2_71);
                        unsigned int _packed_12[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_12[_lp*2 + 0], _scaled_12[_lp*2+1 + 0]));
                            _packed_12[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_12[_lp*2 + 2], _scaled_12[_lp*2+1 + 2]));
                            _packed_12[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_12[_lp*2 + 4], _scaled_12[_lp*2+1 + 4]));
                            _packed_12[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_12[_lp*2 + 6], _scaled_12[_lp*2+1 + 6]));
                            _packed_12[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_12[_lp*2 + 8], _scaled_12[_lp*2+1 + 8]));
                            _packed_12[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_12[_lp*2 + 10], _scaled_12[_lp*2+1 + 10]));
                            _packed_12[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_12[_lp*2 + 12], _scaled_12[_lp*2+1 + 12]));
                            _packed_12[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_12[_lp*2 + 14], _scaled_12[_lp*2+1 + 14]));
                            _packed_12[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(O + (o_base + 96))[0] = reinterpret_cast<int4*>(_packed_12)[0];
                        reinterpret_cast<int4*>(O + (o_base + 96 + 8))[0] = reinterpret_cast<int4*>(_packed_12 + (4))[0];
                        float _scaled_13[16];
                        const float2 _scale2_72 = {inv_sum, inv_sum};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            reinterpret_cast<float2*>(_scaled_13)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((o_rb + 16))[_ls], _scale2_72);
                        unsigned int _packed_13[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_13[_lp*2 + 0], _scaled_13[_lp*2+1 + 0]));
                            _packed_13[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_13[_lp*2 + 2], _scaled_13[_lp*2+1 + 2]));
                            _packed_13[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_13[_lp*2 + 4], _scaled_13[_lp*2+1 + 4]));
                            _packed_13[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_13[_lp*2 + 6], _scaled_13[_lp*2+1 + 6]));
                            _packed_13[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_13[_lp*2 + 8], _scaled_13[_lp*2+1 + 8]));
                            _packed_13[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_13[_lp*2 + 10], _scaled_13[_lp*2+1 + 10]));
                            _packed_13[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_13[_lp*2 + 12], _scaled_13[_lp*2+1 + 12]));
                            _packed_13[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_13[_lp*2 + 14], _scaled_13[_lp*2+1 + 14]));
                            _packed_13[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(O + (o_base + 96 + 16))[0] = reinterpret_cast<int4*>(_packed_13)[0];
                        reinterpret_cast<int4*>(O + (o_base + 96 + 16 + 8))[0] = reinterpret_cast<int4*>(_packed_13 + (4))[0];
                    } else {
                        float _scaled_14[16];
                        const float2 _scale2_73 = {inv_sum, inv_sum};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            reinterpret_cast<float2*>(_scaled_14)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>(o_rb)[_ls], _scale2_73);
                        unsigned int _packed_14[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_14[_lp*2 + 0], _scaled_14[_lp*2+1 + 0]));
                            _packed_14[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_14[_lp*2 + 2], _scaled_14[_lp*2+1 + 2]));
                            _packed_14[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_14[_lp*2 + 4], _scaled_14[_lp*2+1 + 4]));
                            _packed_14[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_14[_lp*2 + 6], _scaled_14[_lp*2+1 + 6]));
                            _packed_14[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_14[_lp*2 + 8], _scaled_14[_lp*2+1 + 8]));
                            _packed_14[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_14[_lp*2 + 10], _scaled_14[_lp*2+1 + 10]));
                            _packed_14[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_14[_lp*2 + 12], _scaled_14[_lp*2+1 + 12]));
                            _packed_14[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_14[_lp*2 + 14], _scaled_14[_lp*2+1 + 14]));
                            _packed_14[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(partial_o + (part_base + 96))[0] = reinterpret_cast<int4*>(_packed_14)[0];
                        reinterpret_cast<int4*>(partial_o + (part_base + 96 + 8))[0] = reinterpret_cast<int4*>(_packed_14 + (4))[0];
                        float _scaled_15[16];
                        const float2 _scale2_74 = {inv_sum, inv_sum};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            reinterpret_cast<float2*>(_scaled_15)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((o_rb + 16))[_ls], _scale2_74);
                        unsigned int _packed_15[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_15[_lp*2 + 0], _scaled_15[_lp*2+1 + 0]));
                            _packed_15[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_15[_lp*2 + 2], _scaled_15[_lp*2+1 + 2]));
                            _packed_15[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_15[_lp*2 + 4], _scaled_15[_lp*2+1 + 4]));
                            _packed_15[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_15[_lp*2 + 6], _scaled_15[_lp*2+1 + 6]));
                            _packed_15[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_15[_lp*2 + 8], _scaled_15[_lp*2+1 + 8]));
                            _packed_15[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_15[_lp*2 + 10], _scaled_15[_lp*2+1 + 10]));
                            _packed_15[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_15[_lp*2 + 12], _scaled_15[_lp*2+1 + 12]));
                            _packed_15[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_15[_lp*2 + 14], _scaled_15[_lp*2+1 + 14]));
                            _packed_15[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(partial_o + (part_base + 96 + 16))[0] = reinterpret_cast<int4*>(_packed_15)[0];
                        reinterpret_cast<int4*>(partial_o + (part_base + 96 + 16 + 8))[0] = reinterpret_cast<int4*>(_packed_15 + (4))[0];
                    }
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(o_rb[0]), "=f"(o_rb[1]), "=f"(o_rb[2]), "=f"(o_rb[3]), "=f"(o_rb[4]), "=f"(o_rb[5]), "=f"(o_rb[6]), "=f"(o_rb[7]), "=f"(o_rb[8]), "=f"(o_rb[9]), "=f"(o_rb[10]), "=f"(o_rb[11]), "=f"(o_rb[12]), "=f"(o_rb[13]), "=f"(o_rb[14]), "=f"(o_rb[15]), "=f"(o_rb[16]), "=f"(o_rb[17]), "=f"(o_rb[18]), "=f"(o_rb[19]), "=f"(o_rb[20]), "=f"(o_rb[21]), "=f"(o_rb[22]), "=f"(o_rb[23]), "=f"(o_rb[24]), "=f"(o_rb[25]), "=f"(o_rb[26]), "=f"(o_rb[27]), "=f"(o_rb[28]), "=f"(o_rb[29]), "=f"(o_rb[30]), "=f"(o_rb[31])
                    : "r"(taddr + (unsigned int)lane_base + 64 + 128 + 32));
                if (epi_bulk != 0) {
                    {
                        const float2 _scale2_75 = {inv_sum, inv_sum};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_ra)[_ls], _scale2_75);
                        unsigned int o_ra_bf16_2[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_ra[_lp*2 + 0], o_ra[_lp*2+1 + 0]));
                            o_ra_bf16_2[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile2 + (lane * 128 ^ (lane * 128 >> 7 & 7) << 4))), "r"(o_ra_bf16_2[0]), "r"(o_ra_bf16_2[1]), "r"(o_ra_bf16_2[2]), "r"(o_ra_bf16_2[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile2 + (lane * 128 + 16 ^ (lane * 128 + 16 >> 7 & 7) << 4))), "r"(o_ra_bf16_2[4]), "r"(o_ra_bf16_2[5]), "r"(o_ra_bf16_2[6]), "r"(o_ra_bf16_2[7]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile2 + (lane * 128 + 32 ^ (lane * 128 + 32 >> 7 & 7) << 4))), "r"(o_ra_bf16_2[8]), "r"(o_ra_bf16_2[9]), "r"(o_ra_bf16_2[10]), "r"(o_ra_bf16_2[11]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile2 + (lane * 128 + 48 ^ (lane * 128 + 48 >> 7 & 7) << 4))), "r"(o_ra_bf16_2[12]), "r"(o_ra_bf16_2[13]), "r"(o_ra_bf16_2[14]), "r"(o_ra_bf16_2[15]) : "memory");
                    }
                } else if (row_valid != 0) {
                    if (direct_out != 0) {
                        float _scaled_16[16];
                        const float2 _scale2_76 = {inv_sum, inv_sum};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            reinterpret_cast<float2*>(_scaled_16)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>(o_ra)[_ls], _scale2_76);
                        unsigned int _packed_16[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_16[_lp*2 + 0], _scaled_16[_lp*2+1 + 0]));
                            _packed_16[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_16[_lp*2 + 2], _scaled_16[_lp*2+1 + 2]));
                            _packed_16[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_16[_lp*2 + 4], _scaled_16[_lp*2+1 + 4]));
                            _packed_16[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_16[_lp*2 + 6], _scaled_16[_lp*2+1 + 6]));
                            _packed_16[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_16[_lp*2 + 8], _scaled_16[_lp*2+1 + 8]));
                            _packed_16[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_16[_lp*2 + 10], _scaled_16[_lp*2+1 + 10]));
                            _packed_16[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_16[_lp*2 + 12], _scaled_16[_lp*2+1 + 12]));
                            _packed_16[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_16[_lp*2 + 14], _scaled_16[_lp*2+1 + 14]));
                            _packed_16[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(O + (o_base + 256))[0] = reinterpret_cast<int4*>(_packed_16)[0];
                        reinterpret_cast<int4*>(O + (o_base + 256 + 8))[0] = reinterpret_cast<int4*>(_packed_16 + (4))[0];
                        float _scaled_17[16];
                        const float2 _scale2_77 = {inv_sum, inv_sum};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            reinterpret_cast<float2*>(_scaled_17)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((o_ra + 16))[_ls], _scale2_77);
                        unsigned int _packed_17[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_17[_lp*2 + 0], _scaled_17[_lp*2+1 + 0]));
                            _packed_17[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_17[_lp*2 + 2], _scaled_17[_lp*2+1 + 2]));
                            _packed_17[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_17[_lp*2 + 4], _scaled_17[_lp*2+1 + 4]));
                            _packed_17[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_17[_lp*2 + 6], _scaled_17[_lp*2+1 + 6]));
                            _packed_17[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_17[_lp*2 + 8], _scaled_17[_lp*2+1 + 8]));
                            _packed_17[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_17[_lp*2 + 10], _scaled_17[_lp*2+1 + 10]));
                            _packed_17[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_17[_lp*2 + 12], _scaled_17[_lp*2+1 + 12]));
                            _packed_17[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_17[_lp*2 + 14], _scaled_17[_lp*2+1 + 14]));
                            _packed_17[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(O + (o_base + 256 + 16))[0] = reinterpret_cast<int4*>(_packed_17)[0];
                        reinterpret_cast<int4*>(O + (o_base + 256 + 16 + 8))[0] = reinterpret_cast<int4*>(_packed_17 + (4))[0];
                    } else {
                        float _scaled_18[16];
                        const float2 _scale2_78 = {inv_sum, inv_sum};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            reinterpret_cast<float2*>(_scaled_18)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>(o_ra)[_ls], _scale2_78);
                        unsigned int _packed_18[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_18[_lp*2 + 0], _scaled_18[_lp*2+1 + 0]));
                            _packed_18[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_18[_lp*2 + 2], _scaled_18[_lp*2+1 + 2]));
                            _packed_18[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_18[_lp*2 + 4], _scaled_18[_lp*2+1 + 4]));
                            _packed_18[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_18[_lp*2 + 6], _scaled_18[_lp*2+1 + 6]));
                            _packed_18[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_18[_lp*2 + 8], _scaled_18[_lp*2+1 + 8]));
                            _packed_18[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_18[_lp*2 + 10], _scaled_18[_lp*2+1 + 10]));
                            _packed_18[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_18[_lp*2 + 12], _scaled_18[_lp*2+1 + 12]));
                            _packed_18[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_18[_lp*2 + 14], _scaled_18[_lp*2+1 + 14]));
                            _packed_18[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(partial_o + (part_base + 256))[0] = reinterpret_cast<int4*>(_packed_18)[0];
                        reinterpret_cast<int4*>(partial_o + (part_base + 256 + 8))[0] = reinterpret_cast<int4*>(_packed_18 + (4))[0];
                        float _scaled_19[16];
                        const float2 _scale2_79 = {inv_sum, inv_sum};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            reinterpret_cast<float2*>(_scaled_19)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((o_ra + 16))[_ls], _scale2_79);
                        unsigned int _packed_19[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_19[_lp*2 + 0], _scaled_19[_lp*2+1 + 0]));
                            _packed_19[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_19[_lp*2 + 2], _scaled_19[_lp*2+1 + 2]));
                            _packed_19[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_19[_lp*2 + 4], _scaled_19[_lp*2+1 + 4]));
                            _packed_19[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_19[_lp*2 + 6], _scaled_19[_lp*2+1 + 6]));
                            _packed_19[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_19[_lp*2 + 8], _scaled_19[_lp*2+1 + 8]));
                            _packed_19[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_19[_lp*2 + 10], _scaled_19[_lp*2+1 + 10]));
                            _packed_19[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_19[_lp*2 + 12], _scaled_19[_lp*2+1 + 12]));
                            _packed_19[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_19[_lp*2 + 14], _scaled_19[_lp*2+1 + 14]));
                            _packed_19[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(partial_o + (part_base + 256 + 16))[0] = reinterpret_cast<int4*>(_packed_19)[0];
                        reinterpret_cast<int4*>(partial_o + (part_base + 256 + 16 + 8))[0] = reinterpret_cast<int4*>(_packed_19 + (4))[0];
                    }
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(o_ra[0]), "=f"(o_ra[1]), "=f"(o_ra[2]), "=f"(o_ra[3]), "=f"(o_ra[4]), "=f"(o_ra[5]), "=f"(o_ra[6]), "=f"(o_ra[7]), "=f"(o_ra[8]), "=f"(o_ra[9]), "=f"(o_ra[10]), "=f"(o_ra[11]), "=f"(o_ra[12]), "=f"(o_ra[13]), "=f"(o_ra[14]), "=f"(o_ra[15]), "=f"(o_ra[16]), "=f"(o_ra[17]), "=f"(o_ra[18]), "=f"(o_ra[19]), "=f"(o_ra[20]), "=f"(o_ra[21]), "=f"(o_ra[22]), "=f"(o_ra[23]), "=f"(o_ra[24]), "=f"(o_ra[25]), "=f"(o_ra[26]), "=f"(o_ra[27]), "=f"(o_ra[28]), "=f"(o_ra[29]), "=f"(o_ra[30]), "=f"(o_ra[31])
                    : "r"(taddr + (unsigned int)lane_base + 64 + 128 + 64));
                if (epi_bulk != 0) {
                    {
                        const float2 _scale2_80 = {inv_sum, inv_sum};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_rb)[_ls], _scale2_80);
                        unsigned int o_rb_bf16_2[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_rb[_lp*2 + 0], o_rb[_lp*2+1 + 0]));
                            o_rb_bf16_2[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile2 + (lane * 128 + 64 ^ (lane * 128 + 64 >> 7 & 7) << 4))), "r"(o_rb_bf16_2[0]), "r"(o_rb_bf16_2[1]), "r"(o_rb_bf16_2[2]), "r"(o_rb_bf16_2[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile2 + (lane * 128 + 80 ^ (lane * 128 + 80 >> 7 & 7) << 4))), "r"(o_rb_bf16_2[4]), "r"(o_rb_bf16_2[5]), "r"(o_rb_bf16_2[6]), "r"(o_rb_bf16_2[7]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile2 + (lane * 128 + 96 ^ (lane * 128 + 96 >> 7 & 7) << 4))), "r"(o_rb_bf16_2[8]), "r"(o_rb_bf16_2[9]), "r"(o_rb_bf16_2[10]), "r"(o_rb_bf16_2[11]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile2 + (lane * 128 + 112 ^ (lane * 128 + 112 >> 7 & 7) << 4))), "r"(o_rb_bf16_2[12]), "r"(o_rb_bf16_2[13]), "r"(o_rb_bf16_2[14]), "r"(o_rb_bf16_2[15]) : "memory");
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        __syncwarp();
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        if (elect_sync()) {
                            if (direct_out != 0) {
                                tma_store_2d((&OT), n_half * 128 + 256 + 32 - 32, warp_row0, warp_tile2);
                            } else {
                                tma_store_2d((&PO), n_half * 128 + 256 + 32 - 32, split_s * rows_total + warp_row0, warp_tile2);
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    }
                } else if (row_valid != 0) {
                    if (direct_out != 0) {
                        float _scaled_20[16];
                        const float2 _scale2_81 = {inv_sum, inv_sum};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            reinterpret_cast<float2*>(_scaled_20)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>(o_rb)[_ls], _scale2_81);
                        unsigned int _packed_20[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_20[_lp*2 + 0], _scaled_20[_lp*2+1 + 0]));
                            _packed_20[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_20[_lp*2 + 2], _scaled_20[_lp*2+1 + 2]));
                            _packed_20[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_20[_lp*2 + 4], _scaled_20[_lp*2+1 + 4]));
                            _packed_20[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_20[_lp*2 + 6], _scaled_20[_lp*2+1 + 6]));
                            _packed_20[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_20[_lp*2 + 8], _scaled_20[_lp*2+1 + 8]));
                            _packed_20[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_20[_lp*2 + 10], _scaled_20[_lp*2+1 + 10]));
                            _packed_20[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_20[_lp*2 + 12], _scaled_20[_lp*2+1 + 12]));
                            _packed_20[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_20[_lp*2 + 14], _scaled_20[_lp*2+1 + 14]));
                            _packed_20[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(O + (o_base + 256 + 32))[0] = reinterpret_cast<int4*>(_packed_20)[0];
                        reinterpret_cast<int4*>(O + (o_base + 256 + 32 + 8))[0] = reinterpret_cast<int4*>(_packed_20 + (4))[0];
                        float _scaled_21[16];
                        const float2 _scale2_82 = {inv_sum, inv_sum};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            reinterpret_cast<float2*>(_scaled_21)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((o_rb + 16))[_ls], _scale2_82);
                        unsigned int _packed_21[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_21[_lp*2 + 0], _scaled_21[_lp*2+1 + 0]));
                            _packed_21[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_21[_lp*2 + 2], _scaled_21[_lp*2+1 + 2]));
                            _packed_21[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_21[_lp*2 + 4], _scaled_21[_lp*2+1 + 4]));
                            _packed_21[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_21[_lp*2 + 6], _scaled_21[_lp*2+1 + 6]));
                            _packed_21[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_21[_lp*2 + 8], _scaled_21[_lp*2+1 + 8]));
                            _packed_21[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_21[_lp*2 + 10], _scaled_21[_lp*2+1 + 10]));
                            _packed_21[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_21[_lp*2 + 12], _scaled_21[_lp*2+1 + 12]));
                            _packed_21[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_21[_lp*2 + 14], _scaled_21[_lp*2+1 + 14]));
                            _packed_21[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(O + (o_base + 256 + 32 + 16))[0] = reinterpret_cast<int4*>(_packed_21)[0];
                        reinterpret_cast<int4*>(O + (o_base + 256 + 32 + 16 + 8))[0] = reinterpret_cast<int4*>(_packed_21 + (4))[0];
                    } else {
                        float _scaled_22[16];
                        const float2 _scale2_83 = {inv_sum, inv_sum};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            reinterpret_cast<float2*>(_scaled_22)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>(o_rb)[_ls], _scale2_83);
                        unsigned int _packed_22[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_22[_lp*2 + 0], _scaled_22[_lp*2+1 + 0]));
                            _packed_22[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_22[_lp*2 + 2], _scaled_22[_lp*2+1 + 2]));
                            _packed_22[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_22[_lp*2 + 4], _scaled_22[_lp*2+1 + 4]));
                            _packed_22[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_22[_lp*2 + 6], _scaled_22[_lp*2+1 + 6]));
                            _packed_22[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_22[_lp*2 + 8], _scaled_22[_lp*2+1 + 8]));
                            _packed_22[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_22[_lp*2 + 10], _scaled_22[_lp*2+1 + 10]));
                            _packed_22[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_22[_lp*2 + 12], _scaled_22[_lp*2+1 + 12]));
                            _packed_22[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_22[_lp*2 + 14], _scaled_22[_lp*2+1 + 14]));
                            _packed_22[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(partial_o + (part_base + 256 + 32))[0] = reinterpret_cast<int4*>(_packed_22)[0];
                        reinterpret_cast<int4*>(partial_o + (part_base + 256 + 32 + 8))[0] = reinterpret_cast<int4*>(_packed_22 + (4))[0];
                        float _scaled_23[16];
                        const float2 _scale2_84 = {inv_sum, inv_sum};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            reinterpret_cast<float2*>(_scaled_23)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((o_rb + 16))[_ls], _scale2_84);
                        unsigned int _packed_23[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_23[_lp*2 + 0], _scaled_23[_lp*2+1 + 0]));
                            _packed_23[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_23[_lp*2 + 2], _scaled_23[_lp*2+1 + 2]));
                            _packed_23[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_23[_lp*2 + 4], _scaled_23[_lp*2+1 + 4]));
                            _packed_23[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_23[_lp*2 + 6], _scaled_23[_lp*2+1 + 6]));
                            _packed_23[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_23[_lp*2 + 8], _scaled_23[_lp*2+1 + 8]));
                            _packed_23[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_23[_lp*2 + 10], _scaled_23[_lp*2+1 + 10]));
                            _packed_23[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_23[_lp*2 + 12], _scaled_23[_lp*2+1 + 12]));
                            _packed_23[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_23[_lp*2 + 14], _scaled_23[_lp*2+1 + 14]));
                            _packed_23[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(partial_o + (part_base + 256 + 32 + 16))[0] = reinterpret_cast<int4*>(_packed_23)[0];
                        reinterpret_cast<int4*>(partial_o + (part_base + 256 + 32 + 16 + 8))[0] = reinterpret_cast<int4*>(_packed_23 + (4))[0];
                    }
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(o_rb[0]), "=f"(o_rb[1]), "=f"(o_rb[2]), "=f"(o_rb[3]), "=f"(o_rb[4]), "=f"(o_rb[5]), "=f"(o_rb[6]), "=f"(o_rb[7]), "=f"(o_rb[8]), "=f"(o_rb[9]), "=f"(o_rb[10]), "=f"(o_rb[11]), "=f"(o_rb[12]), "=f"(o_rb[13]), "=f"(o_rb[14]), "=f"(o_rb[15]), "=f"(o_rb[16]), "=f"(o_rb[17]), "=f"(o_rb[18]), "=f"(o_rb[19]), "=f"(o_rb[20]), "=f"(o_rb[21]), "=f"(o_rb[22]), "=f"(o_rb[23]), "=f"(o_rb[24]), "=f"(o_rb[25]), "=f"(o_rb[26]), "=f"(o_rb[27]), "=f"(o_rb[28]), "=f"(o_rb[29]), "=f"(o_rb[30]), "=f"(o_rb[31])
                    : "r"(taddr + (unsigned int)lane_base + 64 + 128 + 96));
                if (epi_bulk != 0) {
                    {
                        const float2 _scale2_85 = {inv_sum, inv_sum};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_ra)[_ls], _scale2_85);
                        unsigned int o_ra_bf16_3[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_ra[_lp*2 + 0], o_ra[_lp*2+1 + 0]));
                            o_ra_bf16_3[_lp] = *(uint32_t*)&_bf2;
                        }
                        if (elect_sync()) {
                            asm volatile("cp.async.bulk.wait_group.read 1;");
                        }
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        __syncwarp();
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile1 + (lane * 128 ^ (lane * 128 >> 7 & 7) << 4))), "r"(o_ra_bf16_3[0]), "r"(o_ra_bf16_3[1]), "r"(o_ra_bf16_3[2]), "r"(o_ra_bf16_3[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile1 + (lane * 128 + 16 ^ (lane * 128 + 16 >> 7 & 7) << 4))), "r"(o_ra_bf16_3[4]), "r"(o_ra_bf16_3[5]), "r"(o_ra_bf16_3[6]), "r"(o_ra_bf16_3[7]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile1 + (lane * 128 + 32 ^ (lane * 128 + 32 >> 7 & 7) << 4))), "r"(o_ra_bf16_3[8]), "r"(o_ra_bf16_3[9]), "r"(o_ra_bf16_3[10]), "r"(o_ra_bf16_3[11]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile1 + (lane * 128 + 48 ^ (lane * 128 + 48 >> 7 & 7) << 4))), "r"(o_ra_bf16_3[12]), "r"(o_ra_bf16_3[13]), "r"(o_ra_bf16_3[14]), "r"(o_ra_bf16_3[15]) : "memory");
                    }
                } else if (row_valid != 0) {
                    if (direct_out != 0) {
                        float _scaled_24[16];
                        const float2 _scale2_86 = {inv_sum, inv_sum};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            reinterpret_cast<float2*>(_scaled_24)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>(o_ra)[_ls], _scale2_86);
                        unsigned int _packed_24[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_24[_lp*2 + 0], _scaled_24[_lp*2+1 + 0]));
                            _packed_24[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_24[_lp*2 + 2], _scaled_24[_lp*2+1 + 2]));
                            _packed_24[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_24[_lp*2 + 4], _scaled_24[_lp*2+1 + 4]));
                            _packed_24[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_24[_lp*2 + 6], _scaled_24[_lp*2+1 + 6]));
                            _packed_24[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_24[_lp*2 + 8], _scaled_24[_lp*2+1 + 8]));
                            _packed_24[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_24[_lp*2 + 10], _scaled_24[_lp*2+1 + 10]));
                            _packed_24[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_24[_lp*2 + 12], _scaled_24[_lp*2+1 + 12]));
                            _packed_24[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_24[_lp*2 + 14], _scaled_24[_lp*2+1 + 14]));
                            _packed_24[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(O + (o_base + 256 + 64))[0] = reinterpret_cast<int4*>(_packed_24)[0];
                        reinterpret_cast<int4*>(O + (o_base + 256 + 64 + 8))[0] = reinterpret_cast<int4*>(_packed_24 + (4))[0];
                        float _scaled_25[16];
                        const float2 _scale2_87 = {inv_sum, inv_sum};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            reinterpret_cast<float2*>(_scaled_25)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((o_ra + 16))[_ls], _scale2_87);
                        unsigned int _packed_25[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_25[_lp*2 + 0], _scaled_25[_lp*2+1 + 0]));
                            _packed_25[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_25[_lp*2 + 2], _scaled_25[_lp*2+1 + 2]));
                            _packed_25[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_25[_lp*2 + 4], _scaled_25[_lp*2+1 + 4]));
                            _packed_25[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_25[_lp*2 + 6], _scaled_25[_lp*2+1 + 6]));
                            _packed_25[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_25[_lp*2 + 8], _scaled_25[_lp*2+1 + 8]));
                            _packed_25[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_25[_lp*2 + 10], _scaled_25[_lp*2+1 + 10]));
                            _packed_25[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_25[_lp*2 + 12], _scaled_25[_lp*2+1 + 12]));
                            _packed_25[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_25[_lp*2 + 14], _scaled_25[_lp*2+1 + 14]));
                            _packed_25[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(O + (o_base + 256 + 64 + 16))[0] = reinterpret_cast<int4*>(_packed_25)[0];
                        reinterpret_cast<int4*>(O + (o_base + 256 + 64 + 16 + 8))[0] = reinterpret_cast<int4*>(_packed_25 + (4))[0];
                    } else {
                        float _scaled_26[16];
                        const float2 _scale2_88 = {inv_sum, inv_sum};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            reinterpret_cast<float2*>(_scaled_26)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>(o_ra)[_ls], _scale2_88);
                        unsigned int _packed_26[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_26[_lp*2 + 0], _scaled_26[_lp*2+1 + 0]));
                            _packed_26[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_26[_lp*2 + 2], _scaled_26[_lp*2+1 + 2]));
                            _packed_26[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_26[_lp*2 + 4], _scaled_26[_lp*2+1 + 4]));
                            _packed_26[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_26[_lp*2 + 6], _scaled_26[_lp*2+1 + 6]));
                            _packed_26[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_26[_lp*2 + 8], _scaled_26[_lp*2+1 + 8]));
                            _packed_26[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_26[_lp*2 + 10], _scaled_26[_lp*2+1 + 10]));
                            _packed_26[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_26[_lp*2 + 12], _scaled_26[_lp*2+1 + 12]));
                            _packed_26[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_26[_lp*2 + 14], _scaled_26[_lp*2+1 + 14]));
                            _packed_26[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(partial_o + (part_base + 256 + 64))[0] = reinterpret_cast<int4*>(_packed_26)[0];
                        reinterpret_cast<int4*>(partial_o + (part_base + 256 + 64 + 8))[0] = reinterpret_cast<int4*>(_packed_26 + (4))[0];
                        float _scaled_27[16];
                        const float2 _scale2_89 = {inv_sum, inv_sum};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            reinterpret_cast<float2*>(_scaled_27)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((o_ra + 16))[_ls], _scale2_89);
                        unsigned int _packed_27[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_27[_lp*2 + 0], _scaled_27[_lp*2+1 + 0]));
                            _packed_27[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_27[_lp*2 + 2], _scaled_27[_lp*2+1 + 2]));
                            _packed_27[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_27[_lp*2 + 4], _scaled_27[_lp*2+1 + 4]));
                            _packed_27[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_27[_lp*2 + 6], _scaled_27[_lp*2+1 + 6]));
                            _packed_27[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_27[_lp*2 + 8], _scaled_27[_lp*2+1 + 8]));
                            _packed_27[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_27[_lp*2 + 10], _scaled_27[_lp*2+1 + 10]));
                            _packed_27[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_27[_lp*2 + 12], _scaled_27[_lp*2+1 + 12]));
                            _packed_27[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_27[_lp*2 + 14], _scaled_27[_lp*2+1 + 14]));
                            _packed_27[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(partial_o + (part_base + 256 + 64 + 16))[0] = reinterpret_cast<int4*>(_packed_27)[0];
                        reinterpret_cast<int4*>(partial_o + (part_base + 256 + 64 + 16 + 8))[0] = reinterpret_cast<int4*>(_packed_27 + (4))[0];
                    }
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(o_empty_addr), "r"(0) : "memory");
                if (epi_bulk != 0) {
                    {
                        const float2 _scale2_90 = {inv_sum, inv_sum};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_rb)[_ls], _scale2_90);
                        unsigned int o_rb_bf16_3[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_rb[_lp*2 + 0], o_rb[_lp*2+1 + 0]));
                            o_rb_bf16_3[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile1 + (lane * 128 + 64 ^ (lane * 128 + 64 >> 7 & 7) << 4))), "r"(o_rb_bf16_3[0]), "r"(o_rb_bf16_3[1]), "r"(o_rb_bf16_3[2]), "r"(o_rb_bf16_3[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile1 + (lane * 128 + 80 ^ (lane * 128 + 80 >> 7 & 7) << 4))), "r"(o_rb_bf16_3[4]), "r"(o_rb_bf16_3[5]), "r"(o_rb_bf16_3[6]), "r"(o_rb_bf16_3[7]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile1 + (lane * 128 + 96 ^ (lane * 128 + 96 >> 7 & 7) << 4))), "r"(o_rb_bf16_3[8]), "r"(o_rb_bf16_3[9]), "r"(o_rb_bf16_3[10]), "r"(o_rb_bf16_3[11]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((warp_tile1 + (lane * 128 + 112 ^ (lane * 128 + 112 >> 7 & 7) << 4))), "r"(o_rb_bf16_3[12]), "r"(o_rb_bf16_3[13]), "r"(o_rb_bf16_3[14]), "r"(o_rb_bf16_3[15]) : "memory");
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        __syncwarp();
                        if (elect_sync()) {
                            if (direct_out != 0) {
                                tma_store_2d((&OT), n_half * 128 + 256 + 96 - 32, warp_row0, warp_tile1);
                            } else {
                                tma_store_2d((&PO), n_half * 128 + 256 + 96 - 32, split_s * rows_total + warp_row0, warp_tile1);
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    }
                } else if (row_valid != 0) {
                    if (direct_out != 0) {
                        float _scaled_28[16];
                        const float2 _scale2_91 = {inv_sum, inv_sum};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            reinterpret_cast<float2*>(_scaled_28)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>(o_rb)[_ls], _scale2_91);
                        unsigned int _packed_28[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_28[_lp*2 + 0], _scaled_28[_lp*2+1 + 0]));
                            _packed_28[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_28[_lp*2 + 2], _scaled_28[_lp*2+1 + 2]));
                            _packed_28[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_28[_lp*2 + 4], _scaled_28[_lp*2+1 + 4]));
                            _packed_28[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_28[_lp*2 + 6], _scaled_28[_lp*2+1 + 6]));
                            _packed_28[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_28[_lp*2 + 8], _scaled_28[_lp*2+1 + 8]));
                            _packed_28[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_28[_lp*2 + 10], _scaled_28[_lp*2+1 + 10]));
                            _packed_28[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_28[_lp*2 + 12], _scaled_28[_lp*2+1 + 12]));
                            _packed_28[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_28[_lp*2 + 14], _scaled_28[_lp*2+1 + 14]));
                            _packed_28[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(O + (o_base + 256 + 96))[0] = reinterpret_cast<int4*>(_packed_28)[0];
                        reinterpret_cast<int4*>(O + (o_base + 256 + 96 + 8))[0] = reinterpret_cast<int4*>(_packed_28 + (4))[0];
                        float _scaled_29[16];
                        const float2 _scale2_92 = {inv_sum, inv_sum};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            reinterpret_cast<float2*>(_scaled_29)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((o_rb + 16))[_ls], _scale2_92);
                        unsigned int _packed_29[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_29[_lp*2 + 0], _scaled_29[_lp*2+1 + 0]));
                            _packed_29[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_29[_lp*2 + 2], _scaled_29[_lp*2+1 + 2]));
                            _packed_29[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_29[_lp*2 + 4], _scaled_29[_lp*2+1 + 4]));
                            _packed_29[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_29[_lp*2 + 6], _scaled_29[_lp*2+1 + 6]));
                            _packed_29[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_29[_lp*2 + 8], _scaled_29[_lp*2+1 + 8]));
                            _packed_29[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_29[_lp*2 + 10], _scaled_29[_lp*2+1 + 10]));
                            _packed_29[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_29[_lp*2 + 12], _scaled_29[_lp*2+1 + 12]));
                            _packed_29[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_29[_lp*2 + 14], _scaled_29[_lp*2+1 + 14]));
                            _packed_29[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(O + (o_base + 256 + 96 + 16))[0] = reinterpret_cast<int4*>(_packed_29)[0];
                        reinterpret_cast<int4*>(O + (o_base + 256 + 96 + 16 + 8))[0] = reinterpret_cast<int4*>(_packed_29 + (4))[0];
                    } else {
                        float _scaled_30[16];
                        const float2 _scale2_93 = {inv_sum, inv_sum};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            reinterpret_cast<float2*>(_scaled_30)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>(o_rb)[_ls], _scale2_93);
                        unsigned int _packed_30[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_30[_lp*2 + 0], _scaled_30[_lp*2+1 + 0]));
                            _packed_30[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_30[_lp*2 + 2], _scaled_30[_lp*2+1 + 2]));
                            _packed_30[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_30[_lp*2 + 4], _scaled_30[_lp*2+1 + 4]));
                            _packed_30[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_30[_lp*2 + 6], _scaled_30[_lp*2+1 + 6]));
                            _packed_30[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_30[_lp*2 + 8], _scaled_30[_lp*2+1 + 8]));
                            _packed_30[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_30[_lp*2 + 10], _scaled_30[_lp*2+1 + 10]));
                            _packed_30[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_30[_lp*2 + 12], _scaled_30[_lp*2+1 + 12]));
                            _packed_30[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_30[_lp*2 + 14], _scaled_30[_lp*2+1 + 14]));
                            _packed_30[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(partial_o + (part_base + 256 + 96))[0] = reinterpret_cast<int4*>(_packed_30)[0];
                        reinterpret_cast<int4*>(partial_o + (part_base + 256 + 96 + 8))[0] = reinterpret_cast<int4*>(_packed_30 + (4))[0];
                        float _scaled_31[16];
                        const float2 _scale2_94 = {inv_sum, inv_sum};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            reinterpret_cast<float2*>(_scaled_31)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>((o_rb + 16))[_ls], _scale2_94);
                        unsigned int _packed_31[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_31[_lp*2 + 0], _scaled_31[_lp*2+1 + 0]));
                            _packed_31[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_31[_lp*2 + 2], _scaled_31[_lp*2+1 + 2]));
                            _packed_31[_lp + 1] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_31[_lp*2 + 4], _scaled_31[_lp*2+1 + 4]));
                            _packed_31[_lp + 2] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_31[_lp*2 + 6], _scaled_31[_lp*2+1 + 6]));
                            _packed_31[_lp + 3] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_31[_lp*2 + 8], _scaled_31[_lp*2+1 + 8]));
                            _packed_31[_lp + 4] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_31[_lp*2 + 10], _scaled_31[_lp*2+1 + 10]));
                            _packed_31[_lp + 5] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_31[_lp*2 + 12], _scaled_31[_lp*2+1 + 12]));
                            _packed_31[_lp + 6] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_scaled_31[_lp*2 + 14], _scaled_31[_lp*2+1 + 14]));
                            _packed_31[_lp + 7] = *(uint32_t*)&_bf2;
                        }
                        reinterpret_cast<int4*>(partial_o + (part_base + 256 + 96 + 16))[0] = reinterpret_cast<int4*>(_packed_31)[0];
                        reinterpret_cast<int4*>(partial_o + (part_base + 256 + 96 + 16 + 8))[0] = reinterpret_cast<int4*>(_packed_31 + (4))[0];
                    }
                }
                if (epi_bulk != 0) {
                    if (elect_sync()) {
                        asm volatile("cp.async.bulk.wait_group.read 3;");
                    }
                    __syncwarp();
                }
                if ((row_valid & (int)(n_half == 0)) != 0) {
                    if (direct_out != 0) {
                        LSE[out_row] = lse2_s * 0.6931472f;
                    } else {
                        partial_lse[out_row * max_splits + split_s] = lse2_s;
                        #pragma unroll 1
                        for (int extra = num_splits_s; extra < max_splits; extra++) {
                            partial_lse[out_row * max_splits + extra] = -CUDART_INF_F;
                        }
                    }
                }
                gt_s = gt_s + num_tiles_s;
            }
        }
    }
    // ---- Role: correction ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 112;");
        { // correction_main
            int quadrant_c = make_warp_uniform(warp % 4);
            int row_cc = quadrant_c % 2 * 32 + lane;
            int lane_base_c = quadrant_c * 32 << 16;
            int p_stage_c = 0;
            int raw_stage_c = 2;
            unsigned int _phase_raw_full = 1;
            raw_stage_c += 1;
            if (raw_stage_c == 3) { raw_stage_c = 0; _phase_raw_full ^= 1; }
            int sf_stage_c = 1;
            unsigned int _phase_sf_empty = 0;
            sf_stage_c += 1;
            if (sf_stage_c == 2) { sf_stage_c = 0; _phase_sf_empty ^= 1; }
            int unit_c = cluster_id;
            int item_lo_c = unit_first[unit_c];
            int item_hi_c = unit_first[unit_c + 1];
            int tr_c = blockIdx.x == 0 && warp == 4 && lane == 0;
            int gt_c = 0;
            unsigned int _phase_q_full_0 = 0;
            unsigned int _phase_corr_sig_0 = 0;
            unsigned int _phase_pv_done_0 = 0;
            #pragma unroll 1
            for (int item_c = item_lo_c; item_c < item_hi_c; item_c += 2) {
                int tile_start_c = work_table[item_c * 8 + 3];
                int tile_end_c = work_table[item_c * 8 + 4];
                int num_tiles_c = tile_end_c - tile_start_c;
                mbarrier_wait(q_full_addr, _phase_q_full_0);
                _phase_q_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                unsigned int q_words_c[8];
                int qs_addr_c = smem_qs_addr + (unsigned int)(row_cc * 32);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_words_c[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_c[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_c[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_c[(0) + 3]))
                    : "r"(qs_addr_c));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_words_c[4])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_c[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_c[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_c[(4) + 3]))
                    : "r"(qs_addr_c + 16));
                unsigned int q_row_words_c[16];
                int q_swz_c = row_cc >> 1 & 3;
                int q_atom_addr_c_3_0 = smem_q0_addr + (unsigned int)(row_cc * 64);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(0) + 3]))
                    : "r"(q_atom_addr_c_3_0 + (q_swz_c ^ 0) * 16));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[4])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(4) + 3]))
                    : "r"(q_atom_addr_c_3_0 + (q_swz_c ^ 1) * 16));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[8])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(8) + 3]))
                    : "r"(q_atom_addr_c_3_0 + (q_swz_c ^ 2) * 16));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[12])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(12) + 3]))
                    : "r"(q_atom_addr_c_3_0 + (q_swz_c ^ 3) * 16));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x16.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 416), "r"(q_row_words_c[0]), "r"(q_row_words_c[1]), "r"(q_row_words_c[2]), "r"(q_row_words_c[3]), "r"(q_row_words_c[4]), "r"(q_row_words_c[5]), "r"(q_row_words_c[6]), "r"(q_row_words_c[7]), "r"(q_row_words_c[8]), "r"(q_row_words_c[9]), "r"(q_row_words_c[10]), "r"(q_row_words_c[11]), "r"(q_row_words_c[12]), "r"(q_row_words_c[13]), "r"(q_row_words_c[14]), "r"(q_row_words_c[15]));
                int q_atom_addr_c_3_1 = smem_q1_addr + (unsigned int)(row_cc * 64);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(0) + 3]))
                    : "r"(q_atom_addr_c_3_1 + (q_swz_c ^ 0) * 16));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[4])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(4) + 3]))
                    : "r"(q_atom_addr_c_3_1 + (q_swz_c ^ 1) * 16));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[8])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(8) + 3]))
                    : "r"(q_atom_addr_c_3_1 + (q_swz_c ^ 2) * 16));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[12])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(12) + 3]))
                    : "r"(q_atom_addr_c_3_1 + (q_swz_c ^ 3) * 16));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x16.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 416 + 16), "r"(q_row_words_c[0]), "r"(q_row_words_c[1]), "r"(q_row_words_c[2]), "r"(q_row_words_c[3]), "r"(q_row_words_c[4]), "r"(q_row_words_c[5]), "r"(q_row_words_c[6]), "r"(q_row_words_c[7]), "r"(q_row_words_c[8]), "r"(q_row_words_c[9]), "r"(q_row_words_c[10]), "r"(q_row_words_c[11]), "r"(q_row_words_c[12]), "r"(q_row_words_c[13]), "r"(q_row_words_c[14]), "r"(q_row_words_c[15]));
                int q_atom_addr_c_3_2 = smem_q2_addr + (unsigned int)(row_cc * 64);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(0) + 3]))
                    : "r"(q_atom_addr_c_3_2 + (q_swz_c ^ 0) * 16));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[4])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(4) + 3]))
                    : "r"(q_atom_addr_c_3_2 + (q_swz_c ^ 1) * 16));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[8])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(8) + 3]))
                    : "r"(q_atom_addr_c_3_2 + (q_swz_c ^ 2) * 16));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[12])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(12) + 3]))
                    : "r"(q_atom_addr_c_3_2 + (q_swz_c ^ 3) * 16));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x16.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 416 + 32), "r"(q_row_words_c[0]), "r"(q_row_words_c[1]), "r"(q_row_words_c[2]), "r"(q_row_words_c[3]), "r"(q_row_words_c[4]), "r"(q_row_words_c[5]), "r"(q_row_words_c[6]), "r"(q_row_words_c[7]), "r"(q_row_words_c[8]), "r"(q_row_words_c[9]), "r"(q_row_words_c[10]), "r"(q_row_words_c[11]), "r"(q_row_words_c[12]), "r"(q_row_words_c[13]), "r"(q_row_words_c[14]), "r"(q_row_words_c[15]));
                int q_atom_addr_c_3_3 = smem_q3_addr + (unsigned int)(row_cc * 64);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(0) + 3]))
                    : "r"(q_atom_addr_c_3_3 + (q_swz_c ^ 0) * 16));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[4])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(4) + 3]))
                    : "r"(q_atom_addr_c_3_3 + (q_swz_c ^ 1) * 16));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[8])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(8) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(8) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(8) + 3]))
                    : "r"(q_atom_addr_c_3_3 + (q_swz_c ^ 2) * 16));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[12])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(12) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(12) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_row_words_c[(12) + 3]))
                    : "r"(q_atom_addr_c_3_3 + (q_swz_c ^ 3) * 16));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x16.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 416 + 48), "r"(q_row_words_c[0]), "r"(q_row_words_c[1]), "r"(q_row_words_c[2]), "r"(q_row_words_c[3]), "r"(q_row_words_c[4]), "r"(q_row_words_c[5]), "r"(q_row_words_c[6]), "r"(q_row_words_c[7]), "r"(q_row_words_c[8]), "r"(q_row_words_c[9]), "r"(q_row_words_c[10]), "r"(q_row_words_c[11]), "r"(q_row_words_c[12]), "r"(q_row_words_c[13]), "r"(q_row_words_c[14]), "r"(q_row_words_c[15]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320), "r"(q_words_c[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 1), "r"(q_words_c[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 2), "r"(q_words_c[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 3), "r"(q_words_c[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 4), "r"((q_words_c + 1)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 4 + 1), "r"((q_words_c + 1)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 4 + 2), "r"((q_words_c + 1)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 4 + 3), "r"((q_words_c + 1)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 8), "r"((q_words_c + 2)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 8 + 1), "r"((q_words_c + 2)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 8 + 2), "r"((q_words_c + 2)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 8 + 3), "r"((q_words_c + 2)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 8 + 4), "r"((q_words_c + 3)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 8 + 4 + 1), "r"((q_words_c + 3)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 8 + 4 + 2), "r"((q_words_c + 3)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 8 + 4 + 3), "r"((q_words_c + 3)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 16), "r"((q_words_c + 4)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 16 + 1), "r"((q_words_c + 4)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 16 + 2), "r"((q_words_c + 4)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 16 + 3), "r"((q_words_c + 4)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 16 + 4), "r"((q_words_c + 5)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 16 + 4 + 1), "r"((q_words_c + 5)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 16 + 4 + 2), "r"((q_words_c + 5)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 16 + 4 + 3), "r"((q_words_c + 5)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 24), "r"((q_words_c + 6)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 24 + 1), "r"((q_words_c + 6)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 24 + 2), "r"((q_words_c + 6)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 24 + 3), "r"((q_words_c + 6)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 24 + 4), "r"((q_words_c + 7)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 24 + 4 + 1), "r"((q_words_c + 7)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 24 + 4 + 2), "r"((q_words_c + 7)[0]));
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x1.b32"
                    " [%0], {%1};"
                    :: "r"(taddr + (unsigned int)lane_base_c + 320 + 24 + 4 + 3), "r"((q_words_c + 7)[0]));
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(sfa_full_addr), "r"(0) : "memory");
                mbarrier_wait(raw_full_addr + (raw_stage_c) * 8, _phase_raw_full);
                mbarrier_wait(sf_empty_addr + (sf_stage_c) * 8, _phase_sf_empty);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int sfb_base_c = taddr + (unsigned int)lane_base_c + 352 + (unsigned int)(sf_stage_c * 32);
                unsigned int k_words_c[8];
                int tok_c = quadrant_c / 2 * 32 + lane;
                int ks_addr_c = smem_ks_addr + (unsigned int)(raw_stage_c * 28672) + (unsigned int)(tok_c * 32);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_words_c[0])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c[(0) + 3]))
                    : "r"(ks_addr_c));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_words_c[4])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c[(4) + 3]))
                    : "r"(ks_addr_c + 16));
                {
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(sfb_base_c), "r"(k_words_c[0]));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(sfb_base_c + 4), "r"((k_words_c + 1)[0]));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(sfb_base_c + 8), "r"((k_words_c + 2)[0]));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(sfb_base_c + 8 + 4), "r"((k_words_c + 3)[0]));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(sfb_base_c + 16), "r"((k_words_c + 4)[0]));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(sfb_base_c + 16 + 4), "r"((k_words_c + 5)[0]));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(sfb_base_c + 24), "r"((k_words_c + 6)[0]));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(sfb_base_c + 24 + 4), "r"((k_words_c + 7)[0]));
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(sf_full_addr + (sf_stage_c) * 8), "r"(0) : "memory");
                sf_stage_c += 1;
                if (sf_stage_c == 2) { sf_stage_c = 0; _phase_sf_empty ^= 1; }
                mbarrier_wait(raw_full_addr + (raw_stage_c) * 8, _phase_raw_full);
                mbarrier_wait(sf_empty_addr + (sf_stage_c) * 8, _phase_sf_empty);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int sfb_base_c_0 = taddr + (unsigned int)lane_base_c + 352 + (unsigned int)(sf_stage_c * 32);
                unsigned int k_words_c_0[8];
                int tok_c_0 = 64 + quadrant_c / 2 * 32 + lane;
                int ks_addr_c_0 = smem_ks_addr + (unsigned int)(raw_stage_c * 28672) + (unsigned int)(tok_c_0 * 32);
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_0[(0) + 3]))
                    : "r"(ks_addr_c_0));
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_0[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_0[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_0[(4) + 3]))
                    : "r"(ks_addr_c_0 + 16));
                {
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(sfb_base_c_0), "r"(k_words_c_0[0]));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(sfb_base_c_0 + 4), "r"((k_words_c_0 + 1)[0]));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(sfb_base_c_0 + 8), "r"((k_words_c_0 + 2)[0]));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(sfb_base_c_0 + 8 + 4), "r"((k_words_c_0 + 3)[0]));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(sfb_base_c_0 + 16), "r"((k_words_c_0 + 4)[0]));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(sfb_base_c_0 + 16 + 4), "r"((k_words_c_0 + 5)[0]));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(sfb_base_c_0 + 24), "r"((k_words_c_0 + 6)[0]));
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(sfb_base_c_0 + 24 + 4), "r"((k_words_c_0 + 7)[0]));
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(raw_empty_addr + (raw_stage_c) * 8);
                raw_stage_c += 1;
                if (raw_stage_c == 3) { raw_stage_c = 0; _phase_raw_full ^= 1; }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(sf_full_addr + (sf_stage_c) * 8), "r"(0) : "memory");
                sf_stage_c += 1;
                if (sf_stage_c == 2) { sf_stage_c = 0; _phase_sf_empty ^= 1; }
                if (num_tiles_c > 1) {
                    mbarrier_wait(raw_full_addr + (raw_stage_c) * 8, _phase_raw_full);
                    mbarrier_wait(sf_empty_addr + (sf_stage_c) * 8, _phase_sf_empty);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int sfb_base_c_0_0 = taddr + (unsigned int)lane_base_c + 352 + (unsigned int)(sf_stage_c * 32);
                    unsigned int k_words_c_1[8];
                    int tok_c_2 = quadrant_c / 2 * 32 + lane;
                    int ks_addr_c_3 = smem_ks_addr + (unsigned int)(raw_stage_c * 28672) + (unsigned int)(tok_c_2 * 32);
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_1[(0) + 3]))
                        : "r"(ks_addr_c_3));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_1[(4) + 3]))
                        : "r"(ks_addr_c_3 + 16));
                    {
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(sfb_base_c_0_0), "r"(k_words_c_1[0]));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(sfb_base_c_0_0 + 4), "r"((k_words_c_1 + 1)[0]));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(sfb_base_c_0_0 + 8), "r"((k_words_c_1 + 2)[0]));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(sfb_base_c_0_0 + 8 + 4), "r"((k_words_c_1 + 3)[0]));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(sfb_base_c_0_0 + 16), "r"((k_words_c_1 + 4)[0]));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(sfb_base_c_0_0 + 16 + 4), "r"((k_words_c_1 + 5)[0]));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(sfb_base_c_0_0 + 24), "r"((k_words_c_1 + 6)[0]));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(sfb_base_c_0_0 + 24 + 4), "r"((k_words_c_1 + 7)[0]));
                    }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(sf_full_addr + (sf_stage_c) * 8), "r"(0) : "memory");
                    sf_stage_c += 1;
                    if (sf_stage_c == 2) { sf_stage_c = 0; _phase_sf_empty ^= 1; }
                    mbarrier_wait(raw_full_addr + (raw_stage_c) * 8, _phase_raw_full);
                    mbarrier_wait(sf_empty_addr + (sf_stage_c) * 8, _phase_sf_empty);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int sfb_base_c_1 = taddr + (unsigned int)lane_base_c + 352 + (unsigned int)(sf_stage_c * 32);
                    unsigned int k_words_c_2[8];
                    int tok_c_1 = 64 + quadrant_c / 2 * 32 + lane;
                    int ks_addr_c_1 = smem_ks_addr + (unsigned int)(raw_stage_c * 28672) + (unsigned int)(tok_c_1 * 32);
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_2[(0) + 3]))
                        : "r"(ks_addr_c_1));
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_2[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_2[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_2[(4) + 3]))
                        : "r"(ks_addr_c_1 + 16));
                    {
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(sfb_base_c_1), "r"(k_words_c_2[0]));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(sfb_base_c_1 + 4), "r"((k_words_c_2 + 1)[0]));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(sfb_base_c_1 + 8), "r"((k_words_c_2 + 2)[0]));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(sfb_base_c_1 + 8 + 4), "r"((k_words_c_2 + 3)[0]));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(sfb_base_c_1 + 16), "r"((k_words_c_2 + 4)[0]));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(sfb_base_c_1 + 16 + 4), "r"((k_words_c_2 + 5)[0]));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(sfb_base_c_1 + 24), "r"((k_words_c_2 + 6)[0]));
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x1.b32"
                            " [%0], {%1};"
                            :: "r"(sfb_base_c_1 + 24 + 4), "r"((k_words_c_2 + 7)[0]));
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(raw_empty_addr + (raw_stage_c) * 8);
                    raw_stage_c += 1;
                    if (raw_stage_c == 3) { raw_stage_c = 0; _phase_raw_full ^= 1; }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(sf_full_addr + (sf_stage_c) * 8), "r"(0) : "memory");
                    sf_stage_c += 1;
                    if (sf_stage_c == 2) { sf_stage_c = 0; _phase_sf_empty ^= 1; }
                }
                #pragma unroll 1
                for (int it_c = 0; it_c < num_tiles_c; it_c++) {
                    mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
                    _phase_corr_sig_0 ^= 1;
                    float alpha_c = row_state[row_cc];
                    mbarrier_arrive(corr_empty_addr);
                    if (num_tiles_c > it_c + 2) {
                        mbarrier_wait(raw_full_addr + (raw_stage_c) * 8, _phase_raw_full);
                        mbarrier_wait(sf_empty_addr + (sf_stage_c) * 8, _phase_sf_empty);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int sfb_base_c_0_1 = taddr + (unsigned int)lane_base_c + 352 + (unsigned int)(sf_stage_c * 32);
                        unsigned int k_words_c_1_1[8];
                        int tok_c_2_1 = quadrant_c / 2 * 32 + lane;
                        int ks_addr_c_3_1 = smem_ks_addr + (unsigned int)(raw_stage_c * 28672) + (unsigned int)(tok_c_2_1 * 32);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_1_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_1_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_1_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_1_1[(0) + 3]))
                            : "r"(ks_addr_c_3_1));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_1_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_1_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_1_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_1_1[(4) + 3]))
                            : "r"(ks_addr_c_3_1 + 16));
                        {
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                                " [%0], {%1};"
                                :: "r"(sfb_base_c_0_1), "r"(k_words_c_1_1[0]));
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                                " [%0], {%1};"
                                :: "r"(sfb_base_c_0_1 + 4), "r"((k_words_c_1_1 + 1)[0]));
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                                " [%0], {%1};"
                                :: "r"(sfb_base_c_0_1 + 8), "r"((k_words_c_1_1 + 2)[0]));
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                                " [%0], {%1};"
                                :: "r"(sfb_base_c_0_1 + 8 + 4), "r"((k_words_c_1_1 + 3)[0]));
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                                " [%0], {%1};"
                                :: "r"(sfb_base_c_0_1 + 16), "r"((k_words_c_1_1 + 4)[0]));
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                                " [%0], {%1};"
                                :: "r"(sfb_base_c_0_1 + 16 + 4), "r"((k_words_c_1_1 + 5)[0]));
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                                " [%0], {%1};"
                                :: "r"(sfb_base_c_0_1 + 24), "r"((k_words_c_1_1 + 6)[0]));
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                                " [%0], {%1};"
                                :: "r"(sfb_base_c_0_1 + 24 + 4), "r"((k_words_c_1_1 + 7)[0]));
                        }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                            "}"
                            :: "r"(sf_full_addr + (sf_stage_c) * 8), "r"(0) : "memory");
                        sf_stage_c += 1;
                        if (sf_stage_c == 2) { sf_stage_c = 0; _phase_sf_empty ^= 1; }
                        mbarrier_wait(raw_full_addr + (raw_stage_c) * 8, _phase_raw_full);
                        mbarrier_wait(sf_empty_addr + (sf_stage_c) * 8, _phase_sf_empty);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int sfb_base_c_2 = taddr + (unsigned int)lane_base_c + 352 + (unsigned int)(sf_stage_c * 32);
                        unsigned int k_words_c_3[8];
                        int tok_c_3 = 64 + quadrant_c / 2 * 32 + lane;
                        int ks_addr_c_2 = smem_ks_addr + (unsigned int)(raw_stage_c * 28672) + (unsigned int)(tok_c_3 * 32);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_3[(0) + 3]))
                            : "r"(ks_addr_c_2));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_3[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_3[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_words_c_3[(4) + 3]))
                            : "r"(ks_addr_c_2 + 16));
                        {
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                                " [%0], {%1};"
                                :: "r"(sfb_base_c_2), "r"(k_words_c_3[0]));
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                                " [%0], {%1};"
                                :: "r"(sfb_base_c_2 + 4), "r"((k_words_c_3 + 1)[0]));
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                                " [%0], {%1};"
                                :: "r"(sfb_base_c_2 + 8), "r"((k_words_c_3 + 2)[0]));
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                                " [%0], {%1};"
                                :: "r"(sfb_base_c_2 + 8 + 4), "r"((k_words_c_3 + 3)[0]));
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                                " [%0], {%1};"
                                :: "r"(sfb_base_c_2 + 16), "r"((k_words_c_3 + 4)[0]));
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                                " [%0], {%1};"
                                :: "r"(sfb_base_c_2 + 16 + 4), "r"((k_words_c_3 + 5)[0]));
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                                " [%0], {%1};"
                                :: "r"(sfb_base_c_2 + 24), "r"((k_words_c_3 + 6)[0]));
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x1.b32"
                                " [%0], {%1};"
                                :: "r"(sfb_base_c_2 + 24 + 4), "r"((k_words_c_3 + 7)[0]));
                        }
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive(raw_empty_addr + (raw_stage_c) * 8);
                        raw_stage_c += 1;
                        if (raw_stage_c == 3) { raw_stage_c = 0; _phase_raw_full ^= 1; }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                            "}"
                            :: "r"(sf_full_addr + (sf_stage_c) * 8), "r"(0) : "memory");
                        sf_stage_c += 1;
                        if (sf_stage_c == 2) { sf_stage_c = 0; _phase_sf_empty ^= 1; }
                    }
                    int need_rescale = 0;
                    if (it_c > 0) {
                        int _warp_vote_0 = __all_sync(0xFFFFFFFF, alpha_c == 1.0f);
                        need_rescale = _warp_vote_0 == 0;
                    }
                    if (need_rescale != 0) {
                        mbarrier_wait(pv_done_addr, _phase_pv_done_0);
                        _phase_pv_done_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        #pragma unroll
                        for (int col = 0; col < 256; col += 32) {
                            int o_addr = taddr + (unsigned int)lane_base_c + 64 + (unsigned int)col;
                            float _tmem_load_0[32];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                                : "r"(o_addr));
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            const float2 _scale2_1 = {alpha_c, alpha_c};
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++)
                                mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(_tmem_load_0)[_ls], _scale2_1);
                            tmem_st_x32_f32(o_addr, _tmem_load_0);
                        }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(p_full_addr + (p_stage_c) * 8), "r"(0) : "memory");
                    if (it_c > 0) {
                        if (need_rescale == 0) {
                            mbarrier_wait(pv_done_addr, _phase_pv_done_0);
                            _phase_pv_done_0 ^= 1;
                        }
                    }
                    p_stage_c += 1;
                    if (p_stage_c == 2) { p_stage_c = 0; }
                }
                mbarrier_wait(pv_done_addr, _phase_pv_done_0);
                _phase_pv_done_0 ^= 1;
                gt_c = gt_c + num_tiles_c;
            }
        }
    }
    // ---- Role: transform ----
    if (warp >= 8 && warp <= 13) {
        { // transform_main
            int raw_stage_t = 2;
            unsigned int _phase_raw_full_1 = 1;
            raw_stage_t += 1;
            if (raw_stage_t == 3) { raw_stage_t = 0; _phase_raw_full_1 ^= 1; }
            int v_stage_t = 1;
            unsigned int _phase_v_empty = 0;
            v_stage_t += 1;
            if (v_stage_t == 2) { v_stage_t = 0; _phase_v_empty ^= 1; }
            int role_tid = (warp - 8) * 32 + lane;
            int rank_t = cta_rank;
            int unit_t = cluster_id;
            int pair_r0 = role_tid;
            int src_rel_r0 = (1 - ((pair_r0 / 8 / 2 * 2 + pair_r0 % 8 / 4 - (pair_r0 / 8 / 2 * 2 + pair_r0 % 8 / 4) / 64 * 64) / 32 ^ rank_t)) * ((pair_r0 / 8 % 2 * 2 + rank_t) * 4096) + ((pair_r0 / 8 / 2 * 2 + pair_r0 % 8 / 4 - (pair_r0 / 8 / 2 * 2 + pair_r0 % 8 / 4) / 64 * 64) / 32 ^ rank_t) * (16384 + pair_r0 / 8 % 2 * 4096) + (pair_r0 / 8 / 2 * 2 + pair_r0 % 8 / 4) / 64 * 2048 + (pair_r0 / 8 / 2 * 2 + pair_r0 % 8 / 4 - (pair_r0 / 8 / 2 * 2 + pair_r0 % 8 / 4) / 64 * 64) % 32 * 64 + (pair_r0 % 4 ^ (pair_r0 / 8 / 2 * 2 + pair_r0 % 8 / 4 - (pair_r0 / 8 / 2 * 2 + pair_r0 % 8 / 4) / 64 * 64) % 32 >> 1 & 3) * 16;
            int scale_rel_r0 = (pair_r0 / 8 / 2 * 2 + pair_r0 % 8 / 4) * 32 + (pair_r0 / 8 % 2 * 2 + rank_t) * 8 + pair_r0 % 4 * 2;
            int v_rel_r0 = pair_r0 / 8 % 2 * 16384 + ((pair_r0 / 8 / 2 * 2 + pair_r0 % 8 / 4) * 128 + pair_r0 % 4 * 32 ^ ((pair_r0 / 8 / 2 * 2 + pair_r0 % 8 / 4) * 128 + pair_r0 % 4 * 32 >> 7 & 7) << 4);
            int pair_r1 = role_tid + 192;
            int src_rel_r1 = (1 - ((pair_r1 / 8 / 2 * 2 + pair_r1 % 8 / 4 - (pair_r1 / 8 / 2 * 2 + pair_r1 % 8 / 4) / 64 * 64) / 32 ^ rank_t)) * ((pair_r1 / 8 % 2 * 2 + rank_t) * 4096) + ((pair_r1 / 8 / 2 * 2 + pair_r1 % 8 / 4 - (pair_r1 / 8 / 2 * 2 + pair_r1 % 8 / 4) / 64 * 64) / 32 ^ rank_t) * (16384 + pair_r1 / 8 % 2 * 4096) + (pair_r1 / 8 / 2 * 2 + pair_r1 % 8 / 4) / 64 * 2048 + (pair_r1 / 8 / 2 * 2 + pair_r1 % 8 / 4 - (pair_r1 / 8 / 2 * 2 + pair_r1 % 8 / 4) / 64 * 64) % 32 * 64 + (pair_r1 % 4 ^ (pair_r1 / 8 / 2 * 2 + pair_r1 % 8 / 4 - (pair_r1 / 8 / 2 * 2 + pair_r1 % 8 / 4) / 64 * 64) % 32 >> 1 & 3) * 16;
            int scale_rel_r1 = (pair_r1 / 8 / 2 * 2 + pair_r1 % 8 / 4) * 32 + (pair_r1 / 8 % 2 * 2 + rank_t) * 8 + pair_r1 % 4 * 2;
            int v_rel_r1 = pair_r1 / 8 % 2 * 16384 + ((pair_r1 / 8 / 2 * 2 + pair_r1 % 8 / 4) * 128 + pair_r1 % 4 * 32 ^ ((pair_r1 / 8 / 2 * 2 + pair_r1 % 8 / 4) * 128 + pair_r1 % 4 * 32 >> 7 & 7) << 4);
            int pair_r2 = role_tid + 384;
            int src_rel_r2 = (1 - ((pair_r2 / 8 / 2 * 2 + pair_r2 % 8 / 4 - (pair_r2 / 8 / 2 * 2 + pair_r2 % 8 / 4) / 64 * 64) / 32 ^ rank_t)) * ((pair_r2 / 8 % 2 * 2 + rank_t) * 4096) + ((pair_r2 / 8 / 2 * 2 + pair_r2 % 8 / 4 - (pair_r2 / 8 / 2 * 2 + pair_r2 % 8 / 4) / 64 * 64) / 32 ^ rank_t) * (16384 + pair_r2 / 8 % 2 * 4096) + (pair_r2 / 8 / 2 * 2 + pair_r2 % 8 / 4) / 64 * 2048 + (pair_r2 / 8 / 2 * 2 + pair_r2 % 8 / 4 - (pair_r2 / 8 / 2 * 2 + pair_r2 % 8 / 4) / 64 * 64) % 32 * 64 + (pair_r2 % 4 ^ (pair_r2 / 8 / 2 * 2 + pair_r2 % 8 / 4 - (pair_r2 / 8 / 2 * 2 + pair_r2 % 8 / 4) / 64 * 64) % 32 >> 1 & 3) * 16;
            int scale_rel_r2 = (pair_r2 / 8 / 2 * 2 + pair_r2 % 8 / 4) * 32 + (pair_r2 / 8 % 2 * 2 + rank_t) * 8 + pair_r2 % 4 * 2;
            int v_rel_r2 = pair_r2 / 8 % 2 * 16384 + ((pair_r2 / 8 / 2 * 2 + pair_r2 % 8 / 4) * 128 + pair_r2 % 4 * 32 ^ ((pair_r2 / 8 / 2 * 2 + pair_r2 % 8 / 4) * 128 + pair_r2 % 4 * 32 >> 7 & 7) << 4);
            int pair_r3 = role_tid + 576;
            int src_rel_r3 = (1 - ((pair_r3 / 8 / 2 * 2 + pair_r3 % 8 / 4 - (pair_r3 / 8 / 2 * 2 + pair_r3 % 8 / 4) / 64 * 64) / 32 ^ rank_t)) * ((pair_r3 / 8 % 2 * 2 + rank_t) * 4096) + ((pair_r3 / 8 / 2 * 2 + pair_r3 % 8 / 4 - (pair_r3 / 8 / 2 * 2 + pair_r3 % 8 / 4) / 64 * 64) / 32 ^ rank_t) * (16384 + pair_r3 / 8 % 2 * 4096) + (pair_r3 / 8 / 2 * 2 + pair_r3 % 8 / 4) / 64 * 2048 + (pair_r3 / 8 / 2 * 2 + pair_r3 % 8 / 4 - (pair_r3 / 8 / 2 * 2 + pair_r3 % 8 / 4) / 64 * 64) % 32 * 64 + (pair_r3 % 4 ^ (pair_r3 / 8 / 2 * 2 + pair_r3 % 8 / 4 - (pair_r3 / 8 / 2 * 2 + pair_r3 % 8 / 4) / 64 * 64) % 32 >> 1 & 3) * 16;
            int scale_rel_r3 = (pair_r3 / 8 / 2 * 2 + pair_r3 % 8 / 4) * 32 + (pair_r3 / 8 % 2 * 2 + rank_t) * 8 + pair_r3 % 4 * 2;
            int v_rel_r3 = pair_r3 / 8 % 2 * 16384 + ((pair_r3 / 8 / 2 * 2 + pair_r3 % 8 / 4) * 128 + pair_r3 % 4 * 32 ^ ((pair_r3 / 8 / 2 * 2 + pair_r3 % 8 / 4) * 128 + pair_r3 % 4 * 32 >> 7 & 7) << 4);
            int pair_r4 = role_tid + 768;
            int src_rel_r4 = (1 - ((pair_r4 / 8 / 2 * 2 + pair_r4 % 8 / 4 - (pair_r4 / 8 / 2 * 2 + pair_r4 % 8 / 4) / 64 * 64) / 32 ^ rank_t)) * ((pair_r4 / 8 % 2 * 2 + rank_t) * 4096) + ((pair_r4 / 8 / 2 * 2 + pair_r4 % 8 / 4 - (pair_r4 / 8 / 2 * 2 + pair_r4 % 8 / 4) / 64 * 64) / 32 ^ rank_t) * (16384 + pair_r4 / 8 % 2 * 4096) + (pair_r4 / 8 / 2 * 2 + pair_r4 % 8 / 4) / 64 * 2048 + (pair_r4 / 8 / 2 * 2 + pair_r4 % 8 / 4 - (pair_r4 / 8 / 2 * 2 + pair_r4 % 8 / 4) / 64 * 64) % 32 * 64 + (pair_r4 % 4 ^ (pair_r4 / 8 / 2 * 2 + pair_r4 % 8 / 4 - (pair_r4 / 8 / 2 * 2 + pair_r4 % 8 / 4) / 64 * 64) % 32 >> 1 & 3) * 16;
            int scale_rel_r4 = (pair_r4 / 8 / 2 * 2 + pair_r4 % 8 / 4) * 32 + (pair_r4 / 8 % 2 * 2 + rank_t) * 8 + pair_r4 % 4 * 2;
            int v_rel_r4 = pair_r4 / 8 % 2 * 16384 + ((pair_r4 / 8 / 2 * 2 + pair_r4 % 8 / 4) * 128 + pair_r4 % 4 * 32 ^ ((pair_r4 / 8 / 2 * 2 + pair_r4 % 8 / 4) * 128 + pair_r4 % 4 * 32 >> 7 & 7) << 4);
            int pair_tail = role_tid - 64 + 960;
            int src_rel_tail = (1 - ((pair_tail / 8 / 2 * 2 + pair_tail % 8 / 4 - (pair_tail / 8 / 2 * 2 + pair_tail % 8 / 4) / 64 * 64) / 32 ^ rank_t)) * ((pair_tail / 8 % 2 * 2 + rank_t) * 4096) + ((pair_tail / 8 / 2 * 2 + pair_tail % 8 / 4 - (pair_tail / 8 / 2 * 2 + pair_tail % 8 / 4) / 64 * 64) / 32 ^ rank_t) * (16384 + pair_tail / 8 % 2 * 4096) + (pair_tail / 8 / 2 * 2 + pair_tail % 8 / 4) / 64 * 2048 + (pair_tail / 8 / 2 * 2 + pair_tail % 8 / 4 - (pair_tail / 8 / 2 * 2 + pair_tail % 8 / 4) / 64 * 64) % 32 * 64 + (pair_tail % 4 ^ (pair_tail / 8 / 2 * 2 + pair_tail % 8 / 4 - (pair_tail / 8 / 2 * 2 + pair_tail % 8 / 4) / 64 * 64) % 32 >> 1 & 3) * 16;
            int scale_rel_tail = (pair_tail / 8 / 2 * 2 + pair_tail % 8 / 4) * 32 + (pair_tail / 8 % 2 * 2 + rank_t) * 8 + pair_tail % 4 * 2;
            int v_rel_tail = pair_tail / 8 % 2 * 16384 + ((pair_tail / 8 / 2 * 2 + pair_tail % 8 / 4) * 128 + pair_tail % 4 * 32 ^ ((pair_tail / 8 / 2 * 2 + pair_tail % 8 / 4) * 128 + pair_tail % 4 * 32 >> 7 & 7) << 4);
            int item_lo_t = unit_first[unit_t];
            int item_hi_t = unit_first[unit_t + 1];
            int tr_t = blockIdx.x == 0 && warp == 8 && lane == 0;
            int gt_t = 0;
            #pragma unroll 1
            for (int item_t = item_lo_t; item_t < item_hi_t; item_t += 2) {
                int tile_start_t = work_table[item_t * 8 + 3];
                int tile_end_t = work_table[item_t * 8 + 4];
                int num_tiles_t = tile_end_t - tile_start_t;
                #pragma unroll 1
                for (int it_t = 0; it_t < num_tiles_t; it_t++) {
                    mbarrier_wait(raw_full_addr + (raw_stage_t) * 8, _phase_raw_full_1);
                    mbarrier_wait(v_empty_addr + (v_stage_t) * 8, _phase_v_empty);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    int raw_base_t = smem_k0h0_addr + (unsigned int)(raw_stage_t * 28672);
                    int scale_off_t = raw_stage_t * 28672 + 24576;
                    int v_off_t = v_stage_t * 32768;
                    unsigned int packed[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&packed[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 3]))
                        : "r"(raw_base_t + src_rel_r0));
                    unsigned int scale0 = smem_raw_flat[scale_off_t + scale_rel_r0];
                    unsigned int scale1 = smem_raw_flat[scale_off_t + scale_rel_r0 + 1];
                    unsigned int packed_1[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&packed_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_1[(0) + 3]))
                        : "r"(raw_base_t + src_rel_r1));
                    unsigned int scale0_0 = smem_raw_flat[scale_off_t + scale_rel_r1];
                    unsigned int scale1_0 = smem_raw_flat[scale_off_t + scale_rel_r1 + 1];
                    unsigned int packed_2[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&packed_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_2[(0) + 3]))
                        : "r"(raw_base_t + src_rel_r2));
                    unsigned int scale0_2 = smem_raw_flat[scale_off_t + scale_rel_r2];
                    unsigned int scale1_1 = smem_raw_flat[scale_off_t + scale_rel_r2 + 1];
                    unsigned int converted[8];
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4)) && defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000 || __CUDA_ARCH_SPECIFIC__ == 1030)
                    converted[0] = qmul4_lower4(packed[0], scale0);
                    converted[1] = qmul4_higher4(packed[0], scale0);
                    converted[2] = qmul4_lower4(packed[1], scale0);
                    converted[3] = qmul4_higher4(packed[1], scale0);
                    converted[4] = qmul4_lower4(packed[2], scale1);
                    converted[5] = qmul4_higher4(packed[2], scale1);
                    converted[6] = qmul4_lower4(packed[3], scale1);
                    converted[7] = qmul4_higher4(packed[3], scale1);
                    #else
                    {
                        uint16_t _qmul4_pair_0_sb0 = (uint16_t)((scale0) & 0xFFu);
                        uint16_t _qmul4_pair_0_s0 = (uint16_t)(_qmul4_pair_0_sb0 | (_qmul4_pair_0_sb0 << 8));
                        uint16_t _qmul4_pair_0_sb1 = (uint16_t)((scale1) & 0xFFu);
                        uint16_t _qmul4_pair_0_s1 = (uint16_t)(_qmul4_pair_0_sb1 | (_qmul4_pair_0_sb1 << 8));
                        asm volatile("{\n"
                            ".reg .b8 b0, b1, b2, b3;\n"
                            ".reg .b32 ah0, ah1, ah2, ah3, sh0, sh1;\n"
                            ".reg .b16 e0, e1, e2, e3;\n"
                            "cvt.rn.f16x2.e4m3x2 sh0, %8;\n"
                            "cvt.rn.f16x2.e4m3x2 sh1, %9;\n"
                            "mov.b32 {b0, b1, b2, b3}, %10;\n"
                            "cvt.rn.f16x2.e2m1x2 ah0, b0;\n"
                            "mul.rn.f16x2 ah0, ah0, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
                            "cvt.rn.f16x2.e2m1x2 ah1, b1;\n"
                            "mul.rn.f16x2 ah1, ah1, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
                            "cvt.rn.f16x2.e2m1x2 ah2, b2;\n"
                            "mul.rn.f16x2 ah2, ah2, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e2, ah2;\n"
                            "cvt.rn.f16x2.e2m1x2 ah3, b3;\n"
                            "mul.rn.f16x2 ah3, ah3, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e3, ah3;\n"
                            "mov.b32 %0, {e0, e1};\n"
                            "mov.b32 %1, {e2, e3};\n"
                            "mov.b32 {b0, b1, b2, b3}, %11;\n"
                            "cvt.rn.f16x2.e2m1x2 ah0, b0;\n"
                            "mul.rn.f16x2 ah0, ah0, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
                            "cvt.rn.f16x2.e2m1x2 ah1, b1;\n"
                            "mul.rn.f16x2 ah1, ah1, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
                            "cvt.rn.f16x2.e2m1x2 ah2, b2;\n"
                            "mul.rn.f16x2 ah2, ah2, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e2, ah2;\n"
                            "cvt.rn.f16x2.e2m1x2 ah3, b3;\n"
                            "mul.rn.f16x2 ah3, ah3, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e3, ah3;\n"
                            "mov.b32 %2, {e0, e1};\n"
                            "mov.b32 %3, {e2, e3};\n"
                            "mov.b32 {b0, b1, b2, b3}, %12;\n"
                            "cvt.rn.f16x2.e2m1x2 ah0, b0;\n"
                            "mul.rn.f16x2 ah0, ah0, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
                            "cvt.rn.f16x2.e2m1x2 ah1, b1;\n"
                            "mul.rn.f16x2 ah1, ah1, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
                            "cvt.rn.f16x2.e2m1x2 ah2, b2;\n"
                            "mul.rn.f16x2 ah2, ah2, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e2, ah2;\n"
                            "cvt.rn.f16x2.e2m1x2 ah3, b3;\n"
                            "mul.rn.f16x2 ah3, ah3, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e3, ah3;\n"
                            "mov.b32 %4, {e0, e1};\n"
                            "mov.b32 %5, {e2, e3};\n"
                            "mov.b32 {b0, b1, b2, b3}, %13;\n"
                            "cvt.rn.f16x2.e2m1x2 ah0, b0;\n"
                            "mul.rn.f16x2 ah0, ah0, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
                            "cvt.rn.f16x2.e2m1x2 ah1, b1;\n"
                            "mul.rn.f16x2 ah1, ah1, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
                            "cvt.rn.f16x2.e2m1x2 ah2, b2;\n"
                            "mul.rn.f16x2 ah2, ah2, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e2, ah2;\n"
                            "cvt.rn.f16x2.e2m1x2 ah3, b3;\n"
                            "mul.rn.f16x2 ah3, ah3, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e3, ah3;\n"
                            "mov.b32 %6, {e0, e1};\n"
                            "mov.b32 %7, {e2, e3};\n"
                            "}\n"
                            : "=&r"(converted[0]), "=&r"(converted[1]), "=&r"(converted[2]), "=&r"(converted[3]), "=&r"(converted[4]), "=&r"(converted[5]), "=&r"(converted[6]), "=&r"(converted[7])
                            : "h"(_qmul4_pair_0_s0), "h"(_qmul4_pair_0_s1), "r"(packed[0]), "r"(packed[1]), "r"(packed[2]), "r"(packed[3]));
                    }
                    #endif
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v0_addr + (unsigned int)(v_off_t + v_rel_r0)), "r"(converted[0]), "r"(converted[1]), "r"(converted[2]), "r"(converted[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v0_addr + (unsigned int)(v_off_t + (v_rel_r0 ^ 16))), "r"(converted[4]), "r"(converted[5]), "r"(converted[6]), "r"(converted[7]) : "memory");
                    unsigned int packed_4[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&packed_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_4[(0) + 3]))
                        : "r"(raw_base_t + src_rel_r3));
                    unsigned int scale0_3 = smem_raw_flat[scale_off_t + scale_rel_r3];
                    unsigned int scale1_3 = smem_raw_flat[scale_off_t + scale_rel_r3 + 1];
                    unsigned int converted_0[8];
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4)) && defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000 || __CUDA_ARCH_SPECIFIC__ == 1030)
                    converted_0[0] = qmul4_lower4(packed_1[0], scale0_0);
                    converted_0[1] = qmul4_higher4(packed_1[0], scale0_0);
                    converted_0[2] = qmul4_lower4(packed_1[1], scale0_0);
                    converted_0[3] = qmul4_higher4(packed_1[1], scale0_0);
                    converted_0[4] = qmul4_lower4(packed_1[2], scale1_0);
                    converted_0[5] = qmul4_higher4(packed_1[2], scale1_0);
                    converted_0[6] = qmul4_lower4(packed_1[3], scale1_0);
                    converted_0[7] = qmul4_higher4(packed_1[3], scale1_0);
                    #else
                    {
                        uint16_t _qmul4_pair_1_sb0 = (uint16_t)((scale0_0) & 0xFFu);
                        uint16_t _qmul4_pair_1_s0 = (uint16_t)(_qmul4_pair_1_sb0 | (_qmul4_pair_1_sb0 << 8));
                        uint16_t _qmul4_pair_1_sb1 = (uint16_t)((scale1_0) & 0xFFu);
                        uint16_t _qmul4_pair_1_s1 = (uint16_t)(_qmul4_pair_1_sb1 | (_qmul4_pair_1_sb1 << 8));
                        asm volatile("{\n"
                            ".reg .b8 b0, b1, b2, b3;\n"
                            ".reg .b32 ah0, ah1, ah2, ah3, sh0, sh1;\n"
                            ".reg .b16 e0, e1, e2, e3;\n"
                            "cvt.rn.f16x2.e4m3x2 sh0, %8;\n"
                            "cvt.rn.f16x2.e4m3x2 sh1, %9;\n"
                            "mov.b32 {b0, b1, b2, b3}, %10;\n"
                            "cvt.rn.f16x2.e2m1x2 ah0, b0;\n"
                            "mul.rn.f16x2 ah0, ah0, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
                            "cvt.rn.f16x2.e2m1x2 ah1, b1;\n"
                            "mul.rn.f16x2 ah1, ah1, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
                            "cvt.rn.f16x2.e2m1x2 ah2, b2;\n"
                            "mul.rn.f16x2 ah2, ah2, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e2, ah2;\n"
                            "cvt.rn.f16x2.e2m1x2 ah3, b3;\n"
                            "mul.rn.f16x2 ah3, ah3, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e3, ah3;\n"
                            "mov.b32 %0, {e0, e1};\n"
                            "mov.b32 %1, {e2, e3};\n"
                            "mov.b32 {b0, b1, b2, b3}, %11;\n"
                            "cvt.rn.f16x2.e2m1x2 ah0, b0;\n"
                            "mul.rn.f16x2 ah0, ah0, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
                            "cvt.rn.f16x2.e2m1x2 ah1, b1;\n"
                            "mul.rn.f16x2 ah1, ah1, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
                            "cvt.rn.f16x2.e2m1x2 ah2, b2;\n"
                            "mul.rn.f16x2 ah2, ah2, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e2, ah2;\n"
                            "cvt.rn.f16x2.e2m1x2 ah3, b3;\n"
                            "mul.rn.f16x2 ah3, ah3, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e3, ah3;\n"
                            "mov.b32 %2, {e0, e1};\n"
                            "mov.b32 %3, {e2, e3};\n"
                            "mov.b32 {b0, b1, b2, b3}, %12;\n"
                            "cvt.rn.f16x2.e2m1x2 ah0, b0;\n"
                            "mul.rn.f16x2 ah0, ah0, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
                            "cvt.rn.f16x2.e2m1x2 ah1, b1;\n"
                            "mul.rn.f16x2 ah1, ah1, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
                            "cvt.rn.f16x2.e2m1x2 ah2, b2;\n"
                            "mul.rn.f16x2 ah2, ah2, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e2, ah2;\n"
                            "cvt.rn.f16x2.e2m1x2 ah3, b3;\n"
                            "mul.rn.f16x2 ah3, ah3, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e3, ah3;\n"
                            "mov.b32 %4, {e0, e1};\n"
                            "mov.b32 %5, {e2, e3};\n"
                            "mov.b32 {b0, b1, b2, b3}, %13;\n"
                            "cvt.rn.f16x2.e2m1x2 ah0, b0;\n"
                            "mul.rn.f16x2 ah0, ah0, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
                            "cvt.rn.f16x2.e2m1x2 ah1, b1;\n"
                            "mul.rn.f16x2 ah1, ah1, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
                            "cvt.rn.f16x2.e2m1x2 ah2, b2;\n"
                            "mul.rn.f16x2 ah2, ah2, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e2, ah2;\n"
                            "cvt.rn.f16x2.e2m1x2 ah3, b3;\n"
                            "mul.rn.f16x2 ah3, ah3, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e3, ah3;\n"
                            "mov.b32 %6, {e0, e1};\n"
                            "mov.b32 %7, {e2, e3};\n"
                            "}\n"
                            : "=&r"(converted_0[0]), "=&r"(converted_0[1]), "=&r"(converted_0[2]), "=&r"(converted_0[3]), "=&r"(converted_0[4]), "=&r"(converted_0[5]), "=&r"(converted_0[6]), "=&r"(converted_0[7])
                            : "h"(_qmul4_pair_1_s0), "h"(_qmul4_pair_1_s1), "r"(packed_1[0]), "r"(packed_1[1]), "r"(packed_1[2]), "r"(packed_1[3]));
                    }
                    #endif
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v0_addr + (unsigned int)(v_off_t + v_rel_r1)), "r"(converted_0[0]), "r"(converted_0[1]), "r"(converted_0[2]), "r"(converted_0[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v0_addr + (unsigned int)(v_off_t + (v_rel_r1 ^ 16))), "r"(converted_0[4]), "r"(converted_0[5]), "r"(converted_0[6]), "r"(converted_0[7]) : "memory");
                    unsigned int packed_5[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&packed_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_5[(0) + 3]))
                        : "r"(raw_base_t + src_rel_r4));
                    unsigned int scale0_5 = smem_raw_flat[scale_off_t + scale_rel_r4];
                    unsigned int scale1_4 = smem_raw_flat[scale_off_t + scale_rel_r4 + 1];
                    unsigned int converted_1[8];
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4)) && defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000 || __CUDA_ARCH_SPECIFIC__ == 1030)
                    converted_1[0] = qmul4_lower4(packed_2[0], scale0_2);
                    converted_1[1] = qmul4_higher4(packed_2[0], scale0_2);
                    converted_1[2] = qmul4_lower4(packed_2[1], scale0_2);
                    converted_1[3] = qmul4_higher4(packed_2[1], scale0_2);
                    converted_1[4] = qmul4_lower4(packed_2[2], scale1_1);
                    converted_1[5] = qmul4_higher4(packed_2[2], scale1_1);
                    converted_1[6] = qmul4_lower4(packed_2[3], scale1_1);
                    converted_1[7] = qmul4_higher4(packed_2[3], scale1_1);
                    #else
                    {
                        uint16_t _qmul4_pair_2_sb0 = (uint16_t)((scale0_2) & 0xFFu);
                        uint16_t _qmul4_pair_2_s0 = (uint16_t)(_qmul4_pair_2_sb0 | (_qmul4_pair_2_sb0 << 8));
                        uint16_t _qmul4_pair_2_sb1 = (uint16_t)((scale1_1) & 0xFFu);
                        uint16_t _qmul4_pair_2_s1 = (uint16_t)(_qmul4_pair_2_sb1 | (_qmul4_pair_2_sb1 << 8));
                        asm volatile("{\n"
                            ".reg .b8 b0, b1, b2, b3;\n"
                            ".reg .b32 ah0, ah1, ah2, ah3, sh0, sh1;\n"
                            ".reg .b16 e0, e1, e2, e3;\n"
                            "cvt.rn.f16x2.e4m3x2 sh0, %8;\n"
                            "cvt.rn.f16x2.e4m3x2 sh1, %9;\n"
                            "mov.b32 {b0, b1, b2, b3}, %10;\n"
                            "cvt.rn.f16x2.e2m1x2 ah0, b0;\n"
                            "mul.rn.f16x2 ah0, ah0, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
                            "cvt.rn.f16x2.e2m1x2 ah1, b1;\n"
                            "mul.rn.f16x2 ah1, ah1, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
                            "cvt.rn.f16x2.e2m1x2 ah2, b2;\n"
                            "mul.rn.f16x2 ah2, ah2, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e2, ah2;\n"
                            "cvt.rn.f16x2.e2m1x2 ah3, b3;\n"
                            "mul.rn.f16x2 ah3, ah3, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e3, ah3;\n"
                            "mov.b32 %0, {e0, e1};\n"
                            "mov.b32 %1, {e2, e3};\n"
                            "mov.b32 {b0, b1, b2, b3}, %11;\n"
                            "cvt.rn.f16x2.e2m1x2 ah0, b0;\n"
                            "mul.rn.f16x2 ah0, ah0, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
                            "cvt.rn.f16x2.e2m1x2 ah1, b1;\n"
                            "mul.rn.f16x2 ah1, ah1, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
                            "cvt.rn.f16x2.e2m1x2 ah2, b2;\n"
                            "mul.rn.f16x2 ah2, ah2, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e2, ah2;\n"
                            "cvt.rn.f16x2.e2m1x2 ah3, b3;\n"
                            "mul.rn.f16x2 ah3, ah3, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e3, ah3;\n"
                            "mov.b32 %2, {e0, e1};\n"
                            "mov.b32 %3, {e2, e3};\n"
                            "mov.b32 {b0, b1, b2, b3}, %12;\n"
                            "cvt.rn.f16x2.e2m1x2 ah0, b0;\n"
                            "mul.rn.f16x2 ah0, ah0, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
                            "cvt.rn.f16x2.e2m1x2 ah1, b1;\n"
                            "mul.rn.f16x2 ah1, ah1, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
                            "cvt.rn.f16x2.e2m1x2 ah2, b2;\n"
                            "mul.rn.f16x2 ah2, ah2, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e2, ah2;\n"
                            "cvt.rn.f16x2.e2m1x2 ah3, b3;\n"
                            "mul.rn.f16x2 ah3, ah3, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e3, ah3;\n"
                            "mov.b32 %4, {e0, e1};\n"
                            "mov.b32 %5, {e2, e3};\n"
                            "mov.b32 {b0, b1, b2, b3}, %13;\n"
                            "cvt.rn.f16x2.e2m1x2 ah0, b0;\n"
                            "mul.rn.f16x2 ah0, ah0, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
                            "cvt.rn.f16x2.e2m1x2 ah1, b1;\n"
                            "mul.rn.f16x2 ah1, ah1, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
                            "cvt.rn.f16x2.e2m1x2 ah2, b2;\n"
                            "mul.rn.f16x2 ah2, ah2, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e2, ah2;\n"
                            "cvt.rn.f16x2.e2m1x2 ah3, b3;\n"
                            "mul.rn.f16x2 ah3, ah3, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e3, ah3;\n"
                            "mov.b32 %6, {e0, e1};\n"
                            "mov.b32 %7, {e2, e3};\n"
                            "}\n"
                            : "=&r"(converted_1[0]), "=&r"(converted_1[1]), "=&r"(converted_1[2]), "=&r"(converted_1[3]), "=&r"(converted_1[4]), "=&r"(converted_1[5]), "=&r"(converted_1[6]), "=&r"(converted_1[7])
                            : "h"(_qmul4_pair_2_s0), "h"(_qmul4_pair_2_s1), "r"(packed_2[0]), "r"(packed_2[1]), "r"(packed_2[2]), "r"(packed_2[3]));
                    }
                    #endif
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v0_addr + (unsigned int)(v_off_t + v_rel_r2)), "r"(converted_1[0]), "r"(converted_1[1]), "r"(converted_1[2]), "r"(converted_1[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v0_addr + (unsigned int)(v_off_t + (v_rel_r2 ^ 16))), "r"(converted_1[4]), "r"(converted_1[5]), "r"(converted_1[6]), "r"(converted_1[7]) : "memory");
                    unsigned int converted_2[8];
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4)) && defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000 || __CUDA_ARCH_SPECIFIC__ == 1030)
                    converted_2[0] = qmul4_lower4(packed_4[0], scale0_3);
                    converted_2[1] = qmul4_higher4(packed_4[0], scale0_3);
                    converted_2[2] = qmul4_lower4(packed_4[1], scale0_3);
                    converted_2[3] = qmul4_higher4(packed_4[1], scale0_3);
                    converted_2[4] = qmul4_lower4(packed_4[2], scale1_3);
                    converted_2[5] = qmul4_higher4(packed_4[2], scale1_3);
                    converted_2[6] = qmul4_lower4(packed_4[3], scale1_3);
                    converted_2[7] = qmul4_higher4(packed_4[3], scale1_3);
                    #else
                    {
                        uint16_t _qmul4_pair_3_sb0 = (uint16_t)((scale0_3) & 0xFFu);
                        uint16_t _qmul4_pair_3_s0 = (uint16_t)(_qmul4_pair_3_sb0 | (_qmul4_pair_3_sb0 << 8));
                        uint16_t _qmul4_pair_3_sb1 = (uint16_t)((scale1_3) & 0xFFu);
                        uint16_t _qmul4_pair_3_s1 = (uint16_t)(_qmul4_pair_3_sb1 | (_qmul4_pair_3_sb1 << 8));
                        asm volatile("{\n"
                            ".reg .b8 b0, b1, b2, b3;\n"
                            ".reg .b32 ah0, ah1, ah2, ah3, sh0, sh1;\n"
                            ".reg .b16 e0, e1, e2, e3;\n"
                            "cvt.rn.f16x2.e4m3x2 sh0, %8;\n"
                            "cvt.rn.f16x2.e4m3x2 sh1, %9;\n"
                            "mov.b32 {b0, b1, b2, b3}, %10;\n"
                            "cvt.rn.f16x2.e2m1x2 ah0, b0;\n"
                            "mul.rn.f16x2 ah0, ah0, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
                            "cvt.rn.f16x2.e2m1x2 ah1, b1;\n"
                            "mul.rn.f16x2 ah1, ah1, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
                            "cvt.rn.f16x2.e2m1x2 ah2, b2;\n"
                            "mul.rn.f16x2 ah2, ah2, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e2, ah2;\n"
                            "cvt.rn.f16x2.e2m1x2 ah3, b3;\n"
                            "mul.rn.f16x2 ah3, ah3, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e3, ah3;\n"
                            "mov.b32 %0, {e0, e1};\n"
                            "mov.b32 %1, {e2, e3};\n"
                            "mov.b32 {b0, b1, b2, b3}, %11;\n"
                            "cvt.rn.f16x2.e2m1x2 ah0, b0;\n"
                            "mul.rn.f16x2 ah0, ah0, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
                            "cvt.rn.f16x2.e2m1x2 ah1, b1;\n"
                            "mul.rn.f16x2 ah1, ah1, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
                            "cvt.rn.f16x2.e2m1x2 ah2, b2;\n"
                            "mul.rn.f16x2 ah2, ah2, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e2, ah2;\n"
                            "cvt.rn.f16x2.e2m1x2 ah3, b3;\n"
                            "mul.rn.f16x2 ah3, ah3, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e3, ah3;\n"
                            "mov.b32 %2, {e0, e1};\n"
                            "mov.b32 %3, {e2, e3};\n"
                            "mov.b32 {b0, b1, b2, b3}, %12;\n"
                            "cvt.rn.f16x2.e2m1x2 ah0, b0;\n"
                            "mul.rn.f16x2 ah0, ah0, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
                            "cvt.rn.f16x2.e2m1x2 ah1, b1;\n"
                            "mul.rn.f16x2 ah1, ah1, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
                            "cvt.rn.f16x2.e2m1x2 ah2, b2;\n"
                            "mul.rn.f16x2 ah2, ah2, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e2, ah2;\n"
                            "cvt.rn.f16x2.e2m1x2 ah3, b3;\n"
                            "mul.rn.f16x2 ah3, ah3, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e3, ah3;\n"
                            "mov.b32 %4, {e0, e1};\n"
                            "mov.b32 %5, {e2, e3};\n"
                            "mov.b32 {b0, b1, b2, b3}, %13;\n"
                            "cvt.rn.f16x2.e2m1x2 ah0, b0;\n"
                            "mul.rn.f16x2 ah0, ah0, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
                            "cvt.rn.f16x2.e2m1x2 ah1, b1;\n"
                            "mul.rn.f16x2 ah1, ah1, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
                            "cvt.rn.f16x2.e2m1x2 ah2, b2;\n"
                            "mul.rn.f16x2 ah2, ah2, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e2, ah2;\n"
                            "cvt.rn.f16x2.e2m1x2 ah3, b3;\n"
                            "mul.rn.f16x2 ah3, ah3, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e3, ah3;\n"
                            "mov.b32 %6, {e0, e1};\n"
                            "mov.b32 %7, {e2, e3};\n"
                            "}\n"
                            : "=&r"(converted_2[0]), "=&r"(converted_2[1]), "=&r"(converted_2[2]), "=&r"(converted_2[3]), "=&r"(converted_2[4]), "=&r"(converted_2[5]), "=&r"(converted_2[6]), "=&r"(converted_2[7])
                            : "h"(_qmul4_pair_3_s0), "h"(_qmul4_pair_3_s1), "r"(packed_4[0]), "r"(packed_4[1]), "r"(packed_4[2]), "r"(packed_4[3]));
                    }
                    #endif
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v0_addr + (unsigned int)(v_off_t + v_rel_r3)), "r"(converted_2[0]), "r"(converted_2[1]), "r"(converted_2[2]), "r"(converted_2[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v0_addr + (unsigned int)(v_off_t + (v_rel_r3 ^ 16))), "r"(converted_2[4]), "r"(converted_2[5]), "r"(converted_2[6]), "r"(converted_2[7]) : "memory");
                    if (role_tid >= 64 && role_tid < 128) {
                        unsigned int packed_1_0[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&packed_1_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_1_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_1_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_1_0[(0) + 3]))
                            : "r"(raw_base_t + src_rel_tail));
                        unsigned int scale0_2_0 = smem_raw_flat[scale_off_t + scale_rel_tail];
                        unsigned int scale1_3_0 = smem_raw_flat[scale_off_t + scale_rel_tail + 1];
                        unsigned int converted_4[8];
                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4)) && defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000 || __CUDA_ARCH_SPECIFIC__ == 1030)
                        converted_4[0] = qmul4_lower4(packed_1_0[0], scale0_2_0);
                        converted_4[1] = qmul4_higher4(packed_1_0[0], scale0_2_0);
                        converted_4[2] = qmul4_lower4(packed_1_0[1], scale0_2_0);
                        converted_4[3] = qmul4_higher4(packed_1_0[1], scale0_2_0);
                        converted_4[4] = qmul4_lower4(packed_1_0[2], scale1_3_0);
                        converted_4[5] = qmul4_higher4(packed_1_0[2], scale1_3_0);
                        converted_4[6] = qmul4_lower4(packed_1_0[3], scale1_3_0);
                        converted_4[7] = qmul4_higher4(packed_1_0[3], scale1_3_0);
                        #else
                        {
                            uint16_t _qmul4_pair_4_sb0 = (uint16_t)((scale0_2_0) & 0xFFu);
                            uint16_t _qmul4_pair_4_s0 = (uint16_t)(_qmul4_pair_4_sb0 | (_qmul4_pair_4_sb0 << 8));
                            uint16_t _qmul4_pair_4_sb1 = (uint16_t)((scale1_3_0) & 0xFFu);
                            uint16_t _qmul4_pair_4_s1 = (uint16_t)(_qmul4_pair_4_sb1 | (_qmul4_pair_4_sb1 << 8));
                            asm volatile("{\n"
                                ".reg .b8 b0, b1, b2, b3;\n"
                                ".reg .b32 ah0, ah1, ah2, ah3, sh0, sh1;\n"
                                ".reg .b16 e0, e1, e2, e3;\n"
                                "cvt.rn.f16x2.e4m3x2 sh0, %8;\n"
                                "cvt.rn.f16x2.e4m3x2 sh1, %9;\n"
                                "mov.b32 {b0, b1, b2, b3}, %10;\n"
                                "cvt.rn.f16x2.e2m1x2 ah0, b0;\n"
                                "mul.rn.f16x2 ah0, ah0, sh0;\n"
                                "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
                                "cvt.rn.f16x2.e2m1x2 ah1, b1;\n"
                                "mul.rn.f16x2 ah1, ah1, sh0;\n"
                                "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
                                "cvt.rn.f16x2.e2m1x2 ah2, b2;\n"
                                "mul.rn.f16x2 ah2, ah2, sh0;\n"
                                "cvt.rn.satfinite.e4m3x2.f16x2 e2, ah2;\n"
                                "cvt.rn.f16x2.e2m1x2 ah3, b3;\n"
                                "mul.rn.f16x2 ah3, ah3, sh0;\n"
                                "cvt.rn.satfinite.e4m3x2.f16x2 e3, ah3;\n"
                                "mov.b32 %0, {e0, e1};\n"
                                "mov.b32 %1, {e2, e3};\n"
                                "mov.b32 {b0, b1, b2, b3}, %11;\n"
                                "cvt.rn.f16x2.e2m1x2 ah0, b0;\n"
                                "mul.rn.f16x2 ah0, ah0, sh0;\n"
                                "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
                                "cvt.rn.f16x2.e2m1x2 ah1, b1;\n"
                                "mul.rn.f16x2 ah1, ah1, sh0;\n"
                                "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
                                "cvt.rn.f16x2.e2m1x2 ah2, b2;\n"
                                "mul.rn.f16x2 ah2, ah2, sh0;\n"
                                "cvt.rn.satfinite.e4m3x2.f16x2 e2, ah2;\n"
                                "cvt.rn.f16x2.e2m1x2 ah3, b3;\n"
                                "mul.rn.f16x2 ah3, ah3, sh0;\n"
                                "cvt.rn.satfinite.e4m3x2.f16x2 e3, ah3;\n"
                                "mov.b32 %2, {e0, e1};\n"
                                "mov.b32 %3, {e2, e3};\n"
                                "mov.b32 {b0, b1, b2, b3}, %12;\n"
                                "cvt.rn.f16x2.e2m1x2 ah0, b0;\n"
                                "mul.rn.f16x2 ah0, ah0, sh1;\n"
                                "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
                                "cvt.rn.f16x2.e2m1x2 ah1, b1;\n"
                                "mul.rn.f16x2 ah1, ah1, sh1;\n"
                                "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
                                "cvt.rn.f16x2.e2m1x2 ah2, b2;\n"
                                "mul.rn.f16x2 ah2, ah2, sh1;\n"
                                "cvt.rn.satfinite.e4m3x2.f16x2 e2, ah2;\n"
                                "cvt.rn.f16x2.e2m1x2 ah3, b3;\n"
                                "mul.rn.f16x2 ah3, ah3, sh1;\n"
                                "cvt.rn.satfinite.e4m3x2.f16x2 e3, ah3;\n"
                                "mov.b32 %4, {e0, e1};\n"
                                "mov.b32 %5, {e2, e3};\n"
                                "mov.b32 {b0, b1, b2, b3}, %13;\n"
                                "cvt.rn.f16x2.e2m1x2 ah0, b0;\n"
                                "mul.rn.f16x2 ah0, ah0, sh1;\n"
                                "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
                                "cvt.rn.f16x2.e2m1x2 ah1, b1;\n"
                                "mul.rn.f16x2 ah1, ah1, sh1;\n"
                                "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
                                "cvt.rn.f16x2.e2m1x2 ah2, b2;\n"
                                "mul.rn.f16x2 ah2, ah2, sh1;\n"
                                "cvt.rn.satfinite.e4m3x2.f16x2 e2, ah2;\n"
                                "cvt.rn.f16x2.e2m1x2 ah3, b3;\n"
                                "mul.rn.f16x2 ah3, ah3, sh1;\n"
                                "cvt.rn.satfinite.e4m3x2.f16x2 e3, ah3;\n"
                                "mov.b32 %6, {e0, e1};\n"
                                "mov.b32 %7, {e2, e3};\n"
                                "}\n"
                                : "=&r"(converted_4[0]), "=&r"(converted_4[1]), "=&r"(converted_4[2]), "=&r"(converted_4[3]), "=&r"(converted_4[4]), "=&r"(converted_4[5]), "=&r"(converted_4[6]), "=&r"(converted_4[7])
                                : "h"(_qmul4_pair_4_s0), "h"(_qmul4_pair_4_s1), "r"(packed_1_0[0]), "r"(packed_1_0[1]), "r"(packed_1_0[2]), "r"(packed_1_0[3]));
                        }
                        #endif
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v0_addr + (unsigned int)(v_off_t + v_rel_tail)), "r"(converted_4[0]), "r"(converted_4[1]), "r"(converted_4[2]), "r"(converted_4[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v0_addr + (unsigned int)(v_off_t + (v_rel_tail ^ 16))), "r"(converted_4[4]), "r"(converted_4[5]), "r"(converted_4[6]), "r"(converted_4[7]) : "memory");
                    }
                    unsigned int converted_3[8];
                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 4)) && defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1000 || __CUDA_ARCH_SPECIFIC__ == 1030)
                    converted_3[0] = qmul4_lower4(packed_5[0], scale0_5);
                    converted_3[1] = qmul4_higher4(packed_5[0], scale0_5);
                    converted_3[2] = qmul4_lower4(packed_5[1], scale0_5);
                    converted_3[3] = qmul4_higher4(packed_5[1], scale0_5);
                    converted_3[4] = qmul4_lower4(packed_5[2], scale1_4);
                    converted_3[5] = qmul4_higher4(packed_5[2], scale1_4);
                    converted_3[6] = qmul4_lower4(packed_5[3], scale1_4);
                    converted_3[7] = qmul4_higher4(packed_5[3], scale1_4);
                    #else
                    {
                        uint16_t _qmul4_pair_5_sb0 = (uint16_t)((scale0_5) & 0xFFu);
                        uint16_t _qmul4_pair_5_s0 = (uint16_t)(_qmul4_pair_5_sb0 | (_qmul4_pair_5_sb0 << 8));
                        uint16_t _qmul4_pair_5_sb1 = (uint16_t)((scale1_4) & 0xFFu);
                        uint16_t _qmul4_pair_5_s1 = (uint16_t)(_qmul4_pair_5_sb1 | (_qmul4_pair_5_sb1 << 8));
                        asm volatile("{\n"
                            ".reg .b8 b0, b1, b2, b3;\n"
                            ".reg .b32 ah0, ah1, ah2, ah3, sh0, sh1;\n"
                            ".reg .b16 e0, e1, e2, e3;\n"
                            "cvt.rn.f16x2.e4m3x2 sh0, %8;\n"
                            "cvt.rn.f16x2.e4m3x2 sh1, %9;\n"
                            "mov.b32 {b0, b1, b2, b3}, %10;\n"
                            "cvt.rn.f16x2.e2m1x2 ah0, b0;\n"
                            "mul.rn.f16x2 ah0, ah0, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
                            "cvt.rn.f16x2.e2m1x2 ah1, b1;\n"
                            "mul.rn.f16x2 ah1, ah1, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
                            "cvt.rn.f16x2.e2m1x2 ah2, b2;\n"
                            "mul.rn.f16x2 ah2, ah2, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e2, ah2;\n"
                            "cvt.rn.f16x2.e2m1x2 ah3, b3;\n"
                            "mul.rn.f16x2 ah3, ah3, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e3, ah3;\n"
                            "mov.b32 %0, {e0, e1};\n"
                            "mov.b32 %1, {e2, e3};\n"
                            "mov.b32 {b0, b1, b2, b3}, %11;\n"
                            "cvt.rn.f16x2.e2m1x2 ah0, b0;\n"
                            "mul.rn.f16x2 ah0, ah0, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
                            "cvt.rn.f16x2.e2m1x2 ah1, b1;\n"
                            "mul.rn.f16x2 ah1, ah1, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
                            "cvt.rn.f16x2.e2m1x2 ah2, b2;\n"
                            "mul.rn.f16x2 ah2, ah2, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e2, ah2;\n"
                            "cvt.rn.f16x2.e2m1x2 ah3, b3;\n"
                            "mul.rn.f16x2 ah3, ah3, sh0;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e3, ah3;\n"
                            "mov.b32 %2, {e0, e1};\n"
                            "mov.b32 %3, {e2, e3};\n"
                            "mov.b32 {b0, b1, b2, b3}, %12;\n"
                            "cvt.rn.f16x2.e2m1x2 ah0, b0;\n"
                            "mul.rn.f16x2 ah0, ah0, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
                            "cvt.rn.f16x2.e2m1x2 ah1, b1;\n"
                            "mul.rn.f16x2 ah1, ah1, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
                            "cvt.rn.f16x2.e2m1x2 ah2, b2;\n"
                            "mul.rn.f16x2 ah2, ah2, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e2, ah2;\n"
                            "cvt.rn.f16x2.e2m1x2 ah3, b3;\n"
                            "mul.rn.f16x2 ah3, ah3, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e3, ah3;\n"
                            "mov.b32 %4, {e0, e1};\n"
                            "mov.b32 %5, {e2, e3};\n"
                            "mov.b32 {b0, b1, b2, b3}, %13;\n"
                            "cvt.rn.f16x2.e2m1x2 ah0, b0;\n"
                            "mul.rn.f16x2 ah0, ah0, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e0, ah0;\n"
                            "cvt.rn.f16x2.e2m1x2 ah1, b1;\n"
                            "mul.rn.f16x2 ah1, ah1, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e1, ah1;\n"
                            "cvt.rn.f16x2.e2m1x2 ah2, b2;\n"
                            "mul.rn.f16x2 ah2, ah2, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e2, ah2;\n"
                            "cvt.rn.f16x2.e2m1x2 ah3, b3;\n"
                            "mul.rn.f16x2 ah3, ah3, sh1;\n"
                            "cvt.rn.satfinite.e4m3x2.f16x2 e3, ah3;\n"
                            "mov.b32 %6, {e0, e1};\n"
                            "mov.b32 %7, {e2, e3};\n"
                            "}\n"
                            : "=&r"(converted_3[0]), "=&r"(converted_3[1]), "=&r"(converted_3[2]), "=&r"(converted_3[3]), "=&r"(converted_3[4]), "=&r"(converted_3[5]), "=&r"(converted_3[6]), "=&r"(converted_3[7])
                            : "h"(_qmul4_pair_5_s0), "h"(_qmul4_pair_5_s1), "r"(packed_5[0]), "r"(packed_5[1]), "r"(packed_5[2]), "r"(packed_5[3]));
                    }
                    #endif
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v0_addr + (unsigned int)(v_off_t + v_rel_r4)), "r"(converted_3[0]), "r"(converted_3[1]), "r"(converted_3[2]), "r"(converted_3[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_v0_addr + (unsigned int)(v_off_t + (v_rel_r4 ^ 16))), "r"(converted_3[4]), "r"(converted_3[5]), "r"(converted_3[6]), "r"(converted_3[7]) : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(v_full_addr + (v_stage_t) * 8), "r"(0) : "memory");
                    mbarrier_arrive(raw_empty_addr + (raw_stage_t) * 8);
                    raw_stage_t += 1;
                    if (raw_stage_t == 3) { raw_stage_t = 0; _phase_raw_full_1 ^= 1; }
                    v_stage_t += 1;
                    if (v_stage_t == 2) { v_stage_t = 0; _phase_v_empty ^= 1; }
                }
                gt_t = gt_t + num_tiles_t;
            }
        }
    }
    // ---- Role: load ----
    if (warp == 14) {
        { // load_main
            int raw_stage_l = 2;
            unsigned int _phase_raw_empty = 0;
            raw_stage_l += 1;
            if (raw_stage_l == 3) { raw_stage_l = 0; _phase_raw_empty ^= 1; }
            int unit_l = cluster_id;
            int rank_l = cta_rank;
            int item_lo_l = unit_first[unit_l];
            int item_hi_l = unit_first[unit_l + 1];
            int tr_l = blockIdx.x == 0 && lane == 0;
            int gt_l = 0;
            unsigned int _phase_q_empty_0 = 1;
            #pragma unroll 1
            for (int item_l = item_lo_l; item_l < item_hi_l; item_l += 2) {
                int b_l = work_table[item_l * 8];
                int m_tile_l = work_table[item_l * 8 + 1];
                int tile_start_l = work_table[item_l * 8 + 3];
                int tile_end_l = work_table[item_l * 8 + 4];
                int num_tiles_l = tile_end_l - tile_start_l;
                int q_row0 = q_indptr[b_l] * num_heads + m_tile_l * 128 + rank_l * 64;
                int kv_len_l = seq_lens[b_l];
                int num_pages_l = (kv_len_l + 64 - 1) / 64;
                int first_fetch_l = 2 * tile_start_l + lane;
                int _min_0 = ((first_fetch_l) < (max_pages - 1) ? (first_fetch_l) : (max_pages - 1));
                int first_batch_l = page_table[b_l * max_pages + _min_0];
                mbarrier_wait(q_empty_addr, _phase_q_empty_0);
                _phase_q_empty_0 ^= 1;
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(q_full_addr, 18432);
                    tma_2d_gmem2smem(smem_q0_addr, (&Q), 0, q_row0, q_full_addr);
                    tma_2d_gmem2smem(smem_q1_addr, (&Q), 64, q_row0, q_full_addr);
                    tma_2d_gmem2smem(smem_q2_addr, (&Q), 128, q_row0, q_full_addr);
                    tma_2d_gmem2smem(smem_q3_addr, (&Q), 192, q_row0, q_full_addr);
                    tma_2d_gmem2smem(smem_qs_addr, (&QS), 0, q_row0, q_full_addr);
                }
                smem_pages[lane] = ((first_fetch_l < num_pages_l) ? first_batch_l : 0);
                __syncwarp();
                #pragma unroll 1
                for (int it_l = 0; it_l < num_tiles_l; it_l++) {
                    int batch_pos = it_l % 16;
                    if (batch_pos == 0 && it_l > 0) {
                        int fetch_page = 2 * (tile_start_l + it_l) + lane;
                        int fetched = ((fetch_page < num_pages_l) ? page_table[b_l * max_pages + fetch_page] : 0);
                        smem_pages[lane] = fetched;
                        __syncwarp();
                    }
                    int page_a = smem_pages[batch_pos * 2];
                    int page_b = smem_pages[batch_pos * 2 + 1];
                    if (num_pages_l <= 2 * (tile_start_l + it_l) + 1) {
                        page_b = page_a;
                    }
                    mbarrier_wait(raw_empty_addr + (raw_stage_l) * 8, _phase_raw_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(raw_full_addr + (raw_stage_l) * 8, 28672);
                        int kv_row_own_h = page_a * 64 + rank_l * 32;
                        int kv_row_oth_h = page_a * 64 + (1 - rank_l) * 32;
                        tma_2d_gmem2smem(smem_k0h0_addr + (unsigned int)(raw_stage_l * 28672), (&KV), 0, kv_row_own_h, raw_full_addr + (raw_stage_l) * 8);
                        tma_2d_gmem2smem(smem_k1h0_addr + (unsigned int)(raw_stage_l * 28672), (&KV), 64, kv_row_own_h, raw_full_addr + (raw_stage_l) * 8);
                        tma_2d_gmem2smem(smem_k2h0_addr + (unsigned int)(raw_stage_l * 28672), (&KV), 128, kv_row_own_h, raw_full_addr + (raw_stage_l) * 8);
                        tma_2d_gmem2smem(smem_k3h0_addr + (unsigned int)(raw_stage_l * 28672), (&KV), 192, kv_row_own_h, raw_full_addr + (raw_stage_l) * 8);
                        if (rank_l == 0) {
                            tma_2d_gmem2smem(smem_ko0h0_addr + (unsigned int)(raw_stage_l * 28672), (&KV), 0, kv_row_oth_h, raw_full_addr + (raw_stage_l) * 8);
                            tma_2d_gmem2smem(smem_ko1h0_addr + (unsigned int)(raw_stage_l * 28672), (&KV), 128, kv_row_oth_h, raw_full_addr + (raw_stage_l) * 8);
                        } else {
                            tma_2d_gmem2smem(smem_ko0h0_addr + (unsigned int)(raw_stage_l * 28672), (&KV), 64, kv_row_oth_h, raw_full_addr + (raw_stage_l) * 8);
                            tma_2d_gmem2smem(smem_ko1h0_addr + (unsigned int)(raw_stage_l * 28672), (&KV), 192, kv_row_oth_h, raw_full_addr + (raw_stage_l) * 8);
                        }
                        tma_2d_gmem2smem(smem_ks_addr + (unsigned int)(raw_stage_l * 28672), (&KVS), 0, page_a * 64, raw_full_addr + (raw_stage_l) * 8);
                        int kv_row_own_h_0 = page_b * 64 + rank_l * 32;
                        int kv_row_oth_h_0 = page_b * 64 + (1 - rank_l) * 32;
                        tma_2d_gmem2smem(smem_k0h1_addr + (unsigned int)(raw_stage_l * 28672), (&KV), 0, kv_row_own_h_0, raw_full_addr + (raw_stage_l) * 8);
                        tma_2d_gmem2smem(smem_k1h1_addr + (unsigned int)(raw_stage_l * 28672), (&KV), 64, kv_row_own_h_0, raw_full_addr + (raw_stage_l) * 8);
                        tma_2d_gmem2smem(smem_k2h1_addr + (unsigned int)(raw_stage_l * 28672), (&KV), 128, kv_row_own_h_0, raw_full_addr + (raw_stage_l) * 8);
                        tma_2d_gmem2smem(smem_k3h1_addr + (unsigned int)(raw_stage_l * 28672), (&KV), 192, kv_row_own_h_0, raw_full_addr + (raw_stage_l) * 8);
                        if (rank_l == 0) {
                            tma_2d_gmem2smem(smem_ko0h1_addr + (unsigned int)(raw_stage_l * 28672), (&KV), 0, kv_row_oth_h_0, raw_full_addr + (raw_stage_l) * 8);
                            tma_2d_gmem2smem(smem_ko1h1_addr + (unsigned int)(raw_stage_l * 28672), (&KV), 128, kv_row_oth_h_0, raw_full_addr + (raw_stage_l) * 8);
                        } else {
                            tma_2d_gmem2smem(smem_ko0h1_addr + (unsigned int)(raw_stage_l * 28672), (&KV), 64, kv_row_oth_h_0, raw_full_addr + (raw_stage_l) * 8);
                            tma_2d_gmem2smem(smem_ko1h1_addr + (unsigned int)(raw_stage_l * 28672), (&KV), 192, kv_row_oth_h_0, raw_full_addr + (raw_stage_l) * 8);
                        }
                        tma_2d_gmem2smem(smem_ks_addr + (unsigned int)(raw_stage_l * 28672) + 2048, (&KVS), 0, page_b * 64, raw_full_addr + (raw_stage_l) * 8);
                    }
                    raw_stage_l += 1;
                    if (raw_stage_l == 3) { raw_stage_l = 0; _phase_raw_empty ^= 1; }
                }
                gt_l = gt_l + num_tiles_l;
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 15) {
        { // mma_main
            int raw_stage_m = 2;
            unsigned int _phase_raw_full_2 = 1;
            raw_stage_m += 1;
            if (raw_stage_m == 3) { raw_stage_m = 0; _phase_raw_full_2 ^= 1; }
            int v_stage_m = 1;
            unsigned int _phase_v_full = 1;
            v_stage_m += 1;
            if (v_stage_m == 2) { v_stage_m = 0; _phase_v_full ^= 1; }
            int p_stage_m = 1;
            unsigned int _phase_p_full = 1;
            p_stage_m += 1;
            if (p_stage_m == 2) { p_stage_m = 0; _phase_p_full ^= 1; }
            int s_stage_m = 1;
            unsigned int _phase_s_empty = 0;
            s_stage_m += 1;
            if (s_stage_m == 2) { s_stage_m = 0; _phase_s_empty ^= 1; }
            int sf_stage_m = 1;
            unsigned int _phase_sf_full = 1;
            sf_stage_m += 1;
            if (sf_stage_m == 2) { sf_stage_m = 0; _phase_sf_full ^= 1; }
            int unit_m = cluster_id;
            int item_lo_m = unit_first[unit_m];
            int item_hi_m = unit_first[unit_m + 1];
            int tr_m = blockIdx.x == 0 && lane == 0;
            int gt_m = 0;
            unsigned int _phase_q_full_0_1 = 0;
            unsigned int _phase_sfa_full_0 = 0;
            unsigned int _phase_o_empty_0 = 1;
            #pragma unroll 1
            for (int item_m = item_lo_m; item_m < item_hi_m; item_m += 2) {
                int tile_start_m = work_table[item_m * 8 + 3];
                int tile_end_m = work_table[item_m * 8 + 4];
                int num_tiles_m = tile_end_m - tile_start_m;
                if (cta_rank == 0) {
                    if (elect_sync()) {
                        mbarrier_wait(q_full_addr, _phase_q_full_0_1);
                        _phase_q_full_0_1 ^= 1;
                        mbarrier_wait(sfa_full_addr, _phase_sfa_full_0);
                        _phase_sfa_full_0 ^= 1;
                        #pragma unroll 1
                        for (int it_m = 0; it_m < num_tiles_m; it_m++) {
                            mbarrier_wait(raw_full_addr + (raw_stage_m) * 8, _phase_raw_full_2);
                            mbarrier_wait(sf_full_addr + (sf_stage_m) * 8, _phase_sf_full);
                            mbarrier_wait(s_empty_addr + (s_stage_m) * 8, _phase_s_empty);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            if (s_stage_m == 0) {
                                int _smem_desc_lo_0 = (((smem_kq0h0_addr) >> 4) & 0x3FFF) + (raw_stage_m) * 1792;
                                {
                                    const uint32_t b_desc_lo = (uint32_t)(_smem_desc_lo_0);
                                    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"(tmem_tmem_acc), "r"(tmem_tmem_q), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x80004020 << 32))), "r"(0x8100480U), "r"(tmem_tmem_sfa0), "r"(tmem_tmem_sfb0), "r"(0) : "memory");
                                    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"(tmem_tmem_acc), "r"((tmem_tmem_q + 8)), "l"(((uint64_t)(b_desc_lo + 2u) | ((uint64_t)0x80004020 << 32))), "r"(0x8100480U), "r"((tmem_tmem_sfa0 + 4)), "r"((tmem_tmem_sfb0 + 4)), "r"(1) : "memory");
                                }
                                int _smem_desc_lo_1 = (((smem_kq1h0_addr) >> 4) & 0x3FFF) + (raw_stage_m) * 1792;
                                {
                                    const uint32_t b_desc_lo = (uint32_t)(_smem_desc_lo_1);
                                    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"(tmem_tmem_acc), "r"(tmem_tmem_q + 16), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x80004020 << 32))), "r"(0x8100480U), "r"(tmem_tmem_sfa1), "r"(tmem_tmem_sfb1), "r"(1) : "memory");
                                    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"(tmem_tmem_acc), "r"((tmem_tmem_q + 16 + 8)), "l"(((uint64_t)(b_desc_lo + 2u) | ((uint64_t)0x80004020 << 32))), "r"(0x8100480U), "r"((tmem_tmem_sfa1 + 4)), "r"((tmem_tmem_sfb1 + 4)), "r"(1) : "memory");
                                }
                                int _smem_desc_lo_2 = (((smem_kq2h0_addr) >> 4) & 0x3FFF) + (raw_stage_m) * 1792;
                                {
                                    const uint32_t b_desc_lo = (uint32_t)(_smem_desc_lo_2);
                                    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"(tmem_tmem_acc), "r"(tmem_tmem_q + 32), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x80004020 << 32))), "r"(0x8100480U), "r"(tmem_tmem_sfa2), "r"(tmem_tmem_sfb2), "r"(1) : "memory");
                                    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"(tmem_tmem_acc), "r"((tmem_tmem_q + 32 + 8)), "l"(((uint64_t)(b_desc_lo + 2u) | ((uint64_t)0x80004020 << 32))), "r"(0x8100480U), "r"((tmem_tmem_sfa2 + 4)), "r"((tmem_tmem_sfb2 + 4)), "r"(1) : "memory");
                                }
                                int _smem_desc_lo_3 = (((smem_kq3h0_addr) >> 4) & 0x3FFF) + (raw_stage_m) * 1792;
                                {
                                    const uint32_t b_desc_lo = (uint32_t)(_smem_desc_lo_3);
                                    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"(tmem_tmem_acc), "r"(tmem_tmem_q + 48), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x80004020 << 32))), "r"(0x8100480U), "r"(tmem_tmem_sfa3), "r"(tmem_tmem_sfb3), "r"(1) : "memory");
                                    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"(tmem_tmem_acc), "r"((tmem_tmem_q + 48 + 8)), "l"(((uint64_t)(b_desc_lo + 2u) | ((uint64_t)0x80004020 << 32))), "r"(0x8100480U), "r"((tmem_tmem_sfa3 + 4)), "r"((tmem_tmem_sfb3 + 4)), "r"(1) : "memory");
                                }
                            } else {
                                int _smem_desc_lo_4 = (((smem_kq0h0_addr) >> 4) & 0x3FFF) + (raw_stage_m) * 1792;
                                {
                                    const uint32_t b_desc_lo = (uint32_t)(_smem_desc_lo_4);
                                    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_tmem_acc + (32))), "r"(tmem_tmem_q), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x80004020 << 32))), "r"(0x8100480U), "r"(tmem_tmem_sfa0), "r"(tmem_tmem_sfb4), "r"(0) : "memory");
                                    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_tmem_acc + (32))), "r"((tmem_tmem_q + 8)), "l"(((uint64_t)(b_desc_lo + 2u) | ((uint64_t)0x80004020 << 32))), "r"(0x8100480U), "r"((tmem_tmem_sfa0 + 4)), "r"((tmem_tmem_sfb4 + 4)), "r"(1) : "memory");
                                }
                                int _smem_desc_lo_5 = (((smem_kq1h0_addr) >> 4) & 0x3FFF) + (raw_stage_m) * 1792;
                                {
                                    const uint32_t b_desc_lo = (uint32_t)(_smem_desc_lo_5);
                                    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_tmem_acc + (32))), "r"(tmem_tmem_q + 16), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x80004020 << 32))), "r"(0x8100480U), "r"(tmem_tmem_sfa1), "r"(tmem_tmem_sfb5), "r"(1) : "memory");
                                    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_tmem_acc + (32))), "r"((tmem_tmem_q + 16 + 8)), "l"(((uint64_t)(b_desc_lo + 2u) | ((uint64_t)0x80004020 << 32))), "r"(0x8100480U), "r"((tmem_tmem_sfa1 + 4)), "r"((tmem_tmem_sfb5 + 4)), "r"(1) : "memory");
                                }
                                int _smem_desc_lo_6 = (((smem_kq2h0_addr) >> 4) & 0x3FFF) + (raw_stage_m) * 1792;
                                {
                                    const uint32_t b_desc_lo = (uint32_t)(_smem_desc_lo_6);
                                    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_tmem_acc + (32))), "r"(tmem_tmem_q + 32), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x80004020 << 32))), "r"(0x8100480U), "r"(tmem_tmem_sfa2), "r"(tmem_tmem_sfb6), "r"(1) : "memory");
                                    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_tmem_acc + (32))), "r"((tmem_tmem_q + 32 + 8)), "l"(((uint64_t)(b_desc_lo + 2u) | ((uint64_t)0x80004020 << 32))), "r"(0x8100480U), "r"((tmem_tmem_sfa2 + 4)), "r"((tmem_tmem_sfb6 + 4)), "r"(1) : "memory");
                                }
                                int _smem_desc_lo_7 = (((smem_kq3h0_addr) >> 4) & 0x3FFF) + (raw_stage_m) * 1792;
                                {
                                    const uint32_t b_desc_lo = (uint32_t)(_smem_desc_lo_7);
                                    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_tmem_acc + (32))), "r"(tmem_tmem_q + 48), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x80004020 << 32))), "r"(0x8100480U), "r"(tmem_tmem_sfa3), "r"(tmem_tmem_sfb7), "r"(1) : "memory");
                                    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_tmem_acc + (32))), "r"((tmem_tmem_q + 48 + 8)), "l"(((uint64_t)(b_desc_lo + 2u) | ((uint64_t)0x80004020 << 32))), "r"(0x8100480U), "r"((tmem_tmem_sfa3 + 4)), "r"((tmem_tmem_sfb7 + 4)), "r"(1) : "memory");
                                }
                            }
                            tcgen05_commit_cg2_multicast(s_full_addr + (s_stage_m) * 8, (uint16_t)(3));
                            tcgen05_commit_cg2_multicast(sf_empty_addr + (sf_stage_m) * 8, (uint16_t)(3));
                            sf_stage_m += 1;
                            if (sf_stage_m == 2) { sf_stage_m = 0; _phase_sf_full ^= 1; }
                            s_stage_m += 1;
                            if (s_stage_m == 2) { s_stage_m = 0; _phase_s_empty ^= 1; }
                            mbarrier_wait(raw_full_addr + (raw_stage_m) * 8, _phase_raw_full_2);
                            mbarrier_wait(sf_full_addr + (sf_stage_m) * 8, _phase_sf_full);
                            mbarrier_wait(s_empty_addr + (s_stage_m) * 8, _phase_s_empty);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            if (s_stage_m == 0) {
                                int _smem_desc_lo_8 = (((smem_kq0h1_addr) >> 4) & 0x3FFF) + (raw_stage_m) * 1792;
                                {
                                    const uint32_t b_desc_lo = (uint32_t)(_smem_desc_lo_8);
                                    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"(tmem_tmem_acc), "r"(tmem_tmem_q), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x80004020 << 32))), "r"(0x8100480U), "r"(tmem_tmem_sfa0), "r"(tmem_tmem_sfb0), "r"(0) : "memory");
                                    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"(tmem_tmem_acc), "r"((tmem_tmem_q + 8)), "l"(((uint64_t)(b_desc_lo + 2u) | ((uint64_t)0x80004020 << 32))), "r"(0x8100480U), "r"((tmem_tmem_sfa0 + 4)), "r"((tmem_tmem_sfb0 + 4)), "r"(1) : "memory");
                                }
                                int _smem_desc_lo_9 = (((smem_kq1h1_addr) >> 4) & 0x3FFF) + (raw_stage_m) * 1792;
                                {
                                    const uint32_t b_desc_lo = (uint32_t)(_smem_desc_lo_9);
                                    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"(tmem_tmem_acc), "r"(tmem_tmem_q + 16), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x80004020 << 32))), "r"(0x8100480U), "r"(tmem_tmem_sfa1), "r"(tmem_tmem_sfb1), "r"(1) : "memory");
                                    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"(tmem_tmem_acc), "r"((tmem_tmem_q + 16 + 8)), "l"(((uint64_t)(b_desc_lo + 2u) | ((uint64_t)0x80004020 << 32))), "r"(0x8100480U), "r"((tmem_tmem_sfa1 + 4)), "r"((tmem_tmem_sfb1 + 4)), "r"(1) : "memory");
                                }
                                int _smem_desc_lo_10 = (((smem_kq2h1_addr) >> 4) & 0x3FFF) + (raw_stage_m) * 1792;
                                {
                                    const uint32_t b_desc_lo = (uint32_t)(_smem_desc_lo_10);
                                    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"(tmem_tmem_acc), "r"(tmem_tmem_q + 32), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x80004020 << 32))), "r"(0x8100480U), "r"(tmem_tmem_sfa2), "r"(tmem_tmem_sfb2), "r"(1) : "memory");
                                    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"(tmem_tmem_acc), "r"((tmem_tmem_q + 32 + 8)), "l"(((uint64_t)(b_desc_lo + 2u) | ((uint64_t)0x80004020 << 32))), "r"(0x8100480U), "r"((tmem_tmem_sfa2 + 4)), "r"((tmem_tmem_sfb2 + 4)), "r"(1) : "memory");
                                }
                                int _smem_desc_lo_11 = (((smem_kq3h1_addr) >> 4) & 0x3FFF) + (raw_stage_m) * 1792;
                                {
                                    const uint32_t b_desc_lo = (uint32_t)(_smem_desc_lo_11);
                                    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"(tmem_tmem_acc), "r"(tmem_tmem_q + 48), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x80004020 << 32))), "r"(0x8100480U), "r"(tmem_tmem_sfa3), "r"(tmem_tmem_sfb3), "r"(1) : "memory");
                                    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"(tmem_tmem_acc), "r"((tmem_tmem_q + 48 + 8)), "l"(((uint64_t)(b_desc_lo + 2u) | ((uint64_t)0x80004020 << 32))), "r"(0x8100480U), "r"((tmem_tmem_sfa3 + 4)), "r"((tmem_tmem_sfb3 + 4)), "r"(1) : "memory");
                                }
                            } else {
                                int _smem_desc_lo_12 = (((smem_kq0h1_addr) >> 4) & 0x3FFF) + (raw_stage_m) * 1792;
                                {
                                    const uint32_t b_desc_lo = (uint32_t)(_smem_desc_lo_12);
                                    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_tmem_acc + (32))), "r"(tmem_tmem_q), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x80004020 << 32))), "r"(0x8100480U), "r"(tmem_tmem_sfa0), "r"(tmem_tmem_sfb4), "r"(0) : "memory");
                                    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_tmem_acc + (32))), "r"((tmem_tmem_q + 8)), "l"(((uint64_t)(b_desc_lo + 2u) | ((uint64_t)0x80004020 << 32))), "r"(0x8100480U), "r"((tmem_tmem_sfa0 + 4)), "r"((tmem_tmem_sfb4 + 4)), "r"(1) : "memory");
                                }
                                int _smem_desc_lo_13 = (((smem_kq1h1_addr) >> 4) & 0x3FFF) + (raw_stage_m) * 1792;
                                {
                                    const uint32_t b_desc_lo = (uint32_t)(_smem_desc_lo_13);
                                    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_tmem_acc + (32))), "r"(tmem_tmem_q + 16), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x80004020 << 32))), "r"(0x8100480U), "r"(tmem_tmem_sfa1), "r"(tmem_tmem_sfb5), "r"(1) : "memory");
                                    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_tmem_acc + (32))), "r"((tmem_tmem_q + 16 + 8)), "l"(((uint64_t)(b_desc_lo + 2u) | ((uint64_t)0x80004020 << 32))), "r"(0x8100480U), "r"((tmem_tmem_sfa1 + 4)), "r"((tmem_tmem_sfb5 + 4)), "r"(1) : "memory");
                                }
                                int _smem_desc_lo_14 = (((smem_kq2h1_addr) >> 4) & 0x3FFF) + (raw_stage_m) * 1792;
                                {
                                    const uint32_t b_desc_lo = (uint32_t)(_smem_desc_lo_14);
                                    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_tmem_acc + (32))), "r"(tmem_tmem_q + 32), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x80004020 << 32))), "r"(0x8100480U), "r"(tmem_tmem_sfa2), "r"(tmem_tmem_sfb6), "r"(1) : "memory");
                                    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_tmem_acc + (32))), "r"((tmem_tmem_q + 32 + 8)), "l"(((uint64_t)(b_desc_lo + 2u) | ((uint64_t)0x80004020 << 32))), "r"(0x8100480U), "r"((tmem_tmem_sfa2 + 4)), "r"((tmem_tmem_sfb6 + 4)), "r"(1) : "memory");
                                }
                                int _smem_desc_lo_15 = (((smem_kq3h1_addr) >> 4) & 0x3FFF) + (raw_stage_m) * 1792;
                                {
                                    const uint32_t b_desc_lo = (uint32_t)(_smem_desc_lo_15);
                                    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_tmem_acc + (32))), "r"(tmem_tmem_q + 48), "l"(((uint64_t)b_desc_lo | ((uint64_t)0x80004020 << 32))), "r"(0x8100480U), "r"(tmem_tmem_sfa3), "r"(tmem_tmem_sfb7), "r"(1) : "memory");
                                    asm volatile("{\n\t.reg .pred p;\n\tsetp.ne.b32 p, %6, 0;\n\ttcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X [%0], [%1], %2, %3, [%4], [%5], p;\n\t}\n" :: "r"((tmem_tmem_acc + (32))), "r"((tmem_tmem_q + 48 + 8)), "l"(((uint64_t)(b_desc_lo + 2u) | ((uint64_t)0x80004020 << 32))), "r"(0x8100480U), "r"((tmem_tmem_sfa3 + 4)), "r"((tmem_tmem_sfb7 + 4)), "r"(1) : "memory");
                                }
                            }
                            tcgen05_commit_cg2_multicast(s_full_addr + (s_stage_m) * 8, (uint16_t)(3));
                            tcgen05_commit_cg2_multicast(sf_empty_addr + (sf_stage_m) * 8, (uint16_t)(3));
                            tcgen05_commit_cg2_multicast(raw_empty_addr + (raw_stage_m) * 8, (uint16_t)(3));
                            raw_stage_m += 1;
                            if (raw_stage_m == 3) { raw_stage_m = 0; _phase_raw_full_2 ^= 1; }
                            sf_stage_m += 1;
                            if (sf_stage_m == 2) { sf_stage_m = 0; _phase_sf_full ^= 1; }
                            s_stage_m += 1;
                            if (s_stage_m == 2) { s_stage_m = 0; _phase_s_empty ^= 1; }
                            if (it_m >= 1) {
                                if (it_m - 1 == 0) {
                                    mbarrier_wait(o_empty_addr, _phase_o_empty_0);
                                    _phase_o_empty_0 ^= 1;
                                }
                            }
                            if (it_m >= 2) {
                                int first_pv_flag = ((it_m - 2 == 0) ? 1 : 0);
                                mbarrier_wait(v_full_addr + (v_stage_m) * 8, _phase_v_full);
                                mbarrier_wait(p_full_addr + (p_stage_m) * 8, _phase_p_full);
                                asm volatile("tcgen05.fence::after_thread_sync;");
                                {
                                    int _smem_desc_lo_16 = (((smem_p_addr) >> 4) & 0x3FFF) + (p_stage_m) * 512;
                                    int _smem_desc_lo_17 = ((((smem_v0_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage_m) * 2048;
                                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 138477584;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_16), "r"(_smem_desc_lo_17), "r"((tmem_tmem_acc + (64))), "r"(((first_pv_flag) ? 0 : 1)) : "memory");
                                    int _smem_desc_lo_18 = (((smem_p_addr) >> 4) & 0x3FFF) + (p_stage_m) * 512;
                                    int _smem_desc_lo_19 = ((((smem_v1_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage_m) * 2048;
                                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 138477584;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_18), "r"(_smem_desc_lo_19), "r"((tmem_tmem_acc + (192))), "r"(((first_pv_flag) ? 0 : 1)) : "memory");
                                }
                                tcgen05_commit_cg2_multicast(pv_done_addr, (uint16_t)(3));
                                tcgen05_commit_cg2_multicast(p_empty_addr + (p_stage_m) * 8, (uint16_t)(3));
                                tcgen05_commit_cg2_multicast(v_empty_addr + (v_stage_m) * 8, (uint16_t)(3));
                                p_stage_m += 1;
                                if (p_stage_m == 2) { p_stage_m = 0; _phase_p_full ^= 1; }
                                v_stage_m += 1;
                                if (v_stage_m == 2) { v_stage_m = 0; _phase_v_full ^= 1; }
                                if (it_m - 2 + 1 == num_tiles_m) {
                                    tcgen05_commit_cg2_multicast(o_full_addr, (uint16_t)(3));
                                }
                            }
                        }
                        tcgen05_commit_cg2_multicast(q_empty_addr, (uint16_t)(3));
                        if (num_tiles_m - 1 == 0) {
                            mbarrier_wait(o_empty_addr, _phase_o_empty_0);
                            _phase_o_empty_0 ^= 1;
                        }
                        int _max_0_1 = ((num_tiles_m - 2) > (0) ? (num_tiles_m - 2) : (0));
                        #pragma unroll 1
                        for (int j_m = _max_0_1; j_m < num_tiles_m; j_m++) {
                            int first_pv_flag_1 = ((j_m == 0) ? 1 : 0);
                            mbarrier_wait(v_full_addr + (v_stage_m) * 8, _phase_v_full);
                            mbarrier_wait(p_full_addr + (p_stage_m) * 8, _phase_p_full);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            {
                                int _smem_desc_lo_20 = (((smem_p_addr) >> 4) & 0x3FFF) + (p_stage_m) * 512;
                                int _smem_desc_lo_21 = ((((smem_v0_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage_m) * 2048;
                                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 138477584;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_20), "r"(_smem_desc_lo_21), "r"((tmem_tmem_acc + (64))), "r"(((first_pv_flag_1) ? 0 : 1)) : "memory");
                                int _smem_desc_lo_22 = (((smem_p_addr) >> 4) & 0x3FFF) + (p_stage_m) * 512;
                                int _smem_desc_lo_23 = ((((smem_v1_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage_m) * 2048;
                                asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 138477584;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_smem_desc_lo_22), "r"(_smem_desc_lo_23), "r"((tmem_tmem_acc + (192))), "r"(((first_pv_flag_1) ? 0 : 1)) : "memory");
                            }
                            tcgen05_commit_cg2_multicast(pv_done_addr, (uint16_t)(3));
                            tcgen05_commit_cg2_multicast(p_empty_addr + (p_stage_m) * 8, (uint16_t)(3));
                            tcgen05_commit_cg2_multicast(v_empty_addr + (v_stage_m) * 8, (uint16_t)(3));
                            p_stage_m += 1;
                            if (p_stage_m == 2) { p_stage_m = 0; _phase_p_full ^= 1; }
                            v_stage_m += 1;
                            if (v_stage_m == 2) { v_stage_m = 0; _phase_v_full ^= 1; }
                            if (j_m + 1 == num_tiles_m) {
                                tcgen05_commit_cg2_multicast(o_full_addr, (uint16_t)(3));
                            }
                        }
                        gt_m += num_tiles_m;
                    }
                }
            }
        }
    }

    // Kernel teardown ops
    asm volatile("tcgen05.fence::before_thread_sync;");
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // Cleanup
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
