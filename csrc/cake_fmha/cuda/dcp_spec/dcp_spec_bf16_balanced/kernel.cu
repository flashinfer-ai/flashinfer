/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#define CAKE_FMHA_INF CUDART_INF_F
#define TMEM_NCOLS 512
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#define TMEM_TMEM_ST_OFFSET 0
#define TMEM_TMEM_OT_OFFSET 128
#else
#define TMEM_TMEM_S_OFFSET 0
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
#define TMEM_TMEM_S_OFFSET 0
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#define TMEM_TMEM_O_OFFSET 256
#define NUM_Q_PIPE_STAGES 1
#define NUM_K_PIPE_STAGES 3
#define NUM_V_PIPE_STAGES 3
#define NUM_SM_PIPE_STAGES 2
#define NUM_STATS_PIPE_STAGES 2
#define NUM_PAGE_PIPE_STAGES 6
#define NUM_WORK_PIPE_STAGES 4
#define SMEM_SMEM_XMAX_OFF 1024
#define SMEM_SMEM_XMAX_STAGE_BYTES 1024
#define SMEM_SMEM_XMAX_STRIDE 1024
#define SMEM_SMEM_SUM_OFF 3072
#define SMEM_SMEM_SUM_STAGE_BYTES 512
#define SMEM_SMEM_SUM_STRIDE 512
#define SMEM_SMEM_CORR_FLAG_OFF 4608
#define SMEM_SMEM_CORR_FLAG_STAGE_BYTES 16
#define SMEM_SMEM_CORR_FLAG_STRIDE 16
#define SMEM_SMEM_MAX_OFF 3584
#define SMEM_SMEM_MAX_STAGE_BYTES 512
#define SMEM_SMEM_MAX_STRIDE 512
#define SMEM_SMEM_PAGE_OFFSETS_OFF 4096
#define SMEM_SMEM_PAGE_OFFSETS_STAGE_BYTES 192
#define SMEM_SMEM_PAGE_OFFSETS_STRIDE 192
#define SMEM_WORK_TOKEN_WORDS_OFF 4352
#define SMEM_WORK_TOKEN_WORDS_STAGE_BYTES 256
#define SMEM_WORK_TOKEN_WORDS_STRIDE 256
#define SMEM_SCHED_SEQ_LENS_OFF 5120
#define SMEM_SCHED_SEQ_LENS_STAGE_BYTES 4096
#define SMEM_SCHED_SEQ_LENS_STRIDE 4096
#define SMEM_SCHED_PHASE_OFF 9216
#define SMEM_SCHED_PHASE_STAGE_BYTES 4096
#define SMEM_SCHED_PHASE_STRIDE 4096
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#define SMEM_SWAP_FLAGS_OFF 4640
#define SMEM_SWAP_FLAGS_STAGE_BYTES 64
#define SMEM_SWAP_FLAGS_STRIDE 64
#define SMEM_SWAP_ACC_OFF 4704
#define SMEM_SWAP_ACC_STAGE_BYTES 256
#define SMEM_SWAP_ACC_STRIDE 256
#define SMEM_SWAP_PART_OFF 1024
#define SMEM_SWAP_PART_STAGE_BYTES 1024
#define SMEM_SWAP_PART_STRIDE 1024
#define SMEM_SWAP_INV_OFF 2048
#define SMEM_SWAP_INV_STAGE_BYTES 512
#define SMEM_SWAP_INV_STRIDE 512
#define SMEM_SWAP_PT_OFF 5120
#define SMEM_SWAP_PT_STAGE_BYTES 8192
#define SMEM_SWAP_PT_STRIDE 8192
#define SMEM_SWAP_PT1_OFF 21504
#define SMEM_SWAP_PT1_STAGE_BYTES 8192
#define SMEM_SWAP_PT1_STRIDE 8192
#define SMEM_SMEM_Q32_OFF 13312
#define SMEM_SMEM_Q32_STAGE_BYTES 8192
#define SMEM_SMEM_Q32_STRIDE 8192
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#define SMEM_RSTAGE_DATA_OFF 13312
#define SMEM_RSTAGE_DATA_STAGE_BYTES 131072
#define SMEM_RSTAGE_DATA_STRIDE 131072
#define SMEM_RSTAGE_STATS_OFF 144384
#define SMEM_RSTAGE_STATS_STAGE_BYTES 16384
#define SMEM_RSTAGE_STATS_STRIDE 16384
#define SMEM_SMEM_QT_OFF 13312
#define SMEM_SMEM_QT_STAGE_BYTES 16384
#define SMEM_SMEM_QT_STRIDE 16384
#define SMEM_SMEM_K_OFF 29696
#define SMEM_SMEM_K_STAGE_BYTES 32768
#define SMEM_SMEM_K_STRIDE 32768
#define SMEM_SMEM_V_OFF 128000
#define SMEM_SMEM_V_STAGE_BYTES 32768
#define SMEM_SMEM_V_STRIDE 32768
#define SMEM_TOTAL 226304
#define THREADS 384
#ifndef N_ROWS
#define N_ROWS 64
#endif
#define BLOCK_N 128
#define HEAD_DIM 128
#define PAGE_SIZE 16
#define NUM_K_STAGES 3
#define NUM_V_STAGES 3

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

#if CAKE_FMHA_DCP_PROGRAM == 1
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

#endif  // CAKE_FMHA_DCP_PROGRAM == 1

__device__ __forceinline__ void tcgen05_mma_f16(
    int taddr, uint64_t a_desc, uint64_t b_desc,
    uint32_t i_desc, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(enable_input_d)
         : "memory");
}



union MmaSmemDesc {
    uint64_t u64;
    uint32_t u32[2];
};

__device__ __forceinline__ void incr_smem_desc_lo(uint64_t& smem_desc, uint32_t offset) {
    MmaSmemDesc tmp;
    tmp.u64 = smem_desc;
    tmp.u32[0] += offset;
    smem_desc = tmp.u64;
}


__device__ __forceinline__ void elect_commit(int mbar_addr) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];\n\t"
        "}\n"
        :: "r"(mbar_addr));
}


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



#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
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


#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
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




#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 4
__device__ __forceinline__ float row_max_reduce(float2 acc) {
    return max_noftz(acc.x, acc.y);
}






__device__ __forceinline__ void mul_f32x2_inplace(float2* a, float2 b) {
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
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 4
#else
__device__ __forceinline__ float row_max_reduce(float2 acc) {
    return max_noftz(acc.x, acc.y);
}






__device__ __forceinline__ void mul_f32x2_inplace(float2* a, float2 b) {
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
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5




__device__ __forceinline__ void tma_4d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_5d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int v, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.5d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w), "r"(v),
           "r"(mbar_addr) : "memory");
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
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
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
}



__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(THREADS) void
kernel_cake_fmha_dcp_spec_bf16_balanced(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap V, __nv_bfloat16* __restrict__ O_ptr, float* __restrict__ LSE_ptr, int* __restrict__ page_table, int* __restrict__ causal_seqlens_kv_global, float* __restrict__ partial_o, float* __restrict__ partial_stats, unsigned int* __restrict__ tile_counters, unsigned int* __restrict__ queue_counters, int max_pages_per_seq, float softmax_scale_log2, int num_q_heads, int num_kv_heads, int batch_size, int q_len, int cp_rank, int cp_world_log2, unsigned int max_items)
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
    #define k_full_addr (mbar_base + 16)
    #define k_empty_addr (mbar_base + 40)
    #define v_full_addr (mbar_base + 64)
    #define v_empty_addr (mbar_base + 88)
    #define s_full_addr (mbar_base + 112)
    #define p_full_addr (mbar_base + 128)
    #define o_ready_addr (mbar_base + 144)
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
    #define p_full_b_addr (mbar_base + 160)
    #define pa_done_addr (mbar_base + 176)
    #define o_empty_addr (mbar_base + 192)
    #define stats_full_addr (mbar_base + 200)
    #define stats_empty_addr (mbar_base + 216)
    #define tmem_dealloc_addr (mbar_base + 232)
    #define page_offsets_full_addr (mbar_base + 240)
    #define page_offsets_empty_addr (mbar_base + 288)
    #define work_full_addr (mbar_base + 336)
    #define work_empty_addr (mbar_base + 368)
    #define claim_gate_addr (mbar_base + 400)
#else
    #define o_empty_addr (mbar_base + 160)
    #define stats_full_addr (mbar_base + 168)
    #define stats_empty_addr (mbar_base + 184)
    #define tmem_dealloc_addr (mbar_base + 200)
    #define page_offsets_full_addr (mbar_base + 208)
    #define page_offsets_empty_addr (mbar_base + 256)
    #define work_full_addr (mbar_base + 304)
    #define work_empty_addr (mbar_base + 336)
    #define claim_gate_addr (mbar_base + 368)
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
    #define o_empty_addr (mbar_base + 160)
    #define stats_full_addr (mbar_base + 168)
    #define stats_empty_addr (mbar_base + 184)
    #define tmem_dealloc_addr (mbar_base + 200)
    #define page_offsets_full_addr (mbar_base + 208)
    #define page_offsets_empty_addr (mbar_base + 256)
    #define work_full_addr (mbar_base + 304)
    #define work_empty_addr (mbar_base + 336)
    #define claim_gate_addr (mbar_base + 368)
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    float* smem_xmax = reinterpret_cast<float*>(smem_raw + SMEM_SMEM_XMAX_OFF);
    const int smem_xmax_addr = smem + SMEM_SMEM_XMAX_OFF;
    float* smem_sum = reinterpret_cast<float*>(smem_raw + SMEM_SMEM_SUM_OFF);
    const int smem_sum_addr = smem + SMEM_SMEM_SUM_OFF;
    unsigned int* smem_corr_flag = reinterpret_cast<unsigned int*>(smem_raw + SMEM_SMEM_CORR_FLAG_OFF);
    const int smem_corr_flag_addr = smem + SMEM_SMEM_CORR_FLAG_OFF;
    float* smem_max = reinterpret_cast<float*>(smem_raw + SMEM_SMEM_MAX_OFF);
    const int smem_max_addr = smem + SMEM_SMEM_MAX_OFF;
    int* smem_page_offsets = reinterpret_cast<int*>(smem_raw + SMEM_SMEM_PAGE_OFFSETS_OFF);
    const int smem_page_offsets_addr = smem + SMEM_SMEM_PAGE_OFFSETS_OFF;
    unsigned int* work_token_words = reinterpret_cast<unsigned int*>(smem_raw + SMEM_WORK_TOKEN_WORDS_OFF);
    const int work_token_words_addr = smem + SMEM_WORK_TOKEN_WORDS_OFF;
    int* sched_seq_lens = reinterpret_cast<int*>(smem_raw + SMEM_SCHED_SEQ_LENS_OFF);
    const int sched_seq_lens_addr = smem + SMEM_SCHED_SEQ_LENS_OFF;
    int* sched_phase = reinterpret_cast<int*>(smem_raw + SMEM_SCHED_PHASE_OFF);
    const int sched_phase_addr = smem + SMEM_SCHED_PHASE_OFF;
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
    unsigned int* swap_flags = reinterpret_cast<unsigned int*>(smem_raw + SMEM_SWAP_FLAGS_OFF);
    const int swap_flags_addr = smem + SMEM_SWAP_FLAGS_OFF;
    float* swap_acc = reinterpret_cast<float*>(smem_raw + SMEM_SWAP_ACC_OFF);
    const int swap_acc_addr = smem + SMEM_SWAP_ACC_OFF;
    float* swap_part = reinterpret_cast<float*>(smem_raw + SMEM_SWAP_PART_OFF);
    const int swap_part_addr = smem + SMEM_SWAP_PART_OFF;
    float* swap_inv = reinterpret_cast<float*>(smem_raw + SMEM_SWAP_INV_OFF);
    const int swap_inv_addr = smem + SMEM_SWAP_INV_OFF;
    __nv_bfloat16* swap_pT = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SWAP_PT_OFF);
    const int swap_pT_addr = smem + SMEM_SWAP_PT_OFF;
    __nv_bfloat16* swap_pT1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SWAP_PT1_OFF);
    const int swap_pT1_addr = smem + SMEM_SWAP_PT1_OFF;
    __nv_bfloat16* smem_q32 = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_Q32_OFF);
    const int smem_q32_addr = smem + SMEM_SMEM_Q32_OFF;
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
    float* rstage_data = reinterpret_cast<float*>(smem_raw + SMEM_RSTAGE_DATA_OFF);
    const int rstage_data_addr = smem + SMEM_RSTAGE_DATA_OFF;
    float* rstage_stats = reinterpret_cast<float*>(smem_raw + SMEM_RSTAGE_STATS_OFF);
    const int rstage_stats_addr = smem + SMEM_RSTAGE_STATS_OFF;
    __nv_bfloat16* smem_qt = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_QT_OFF);
    const int smem_qt_addr = smem + SMEM_SMEM_QT_OFF;
    __nv_bfloat16* smem_k = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_K_OFF);
    const int smem_k_addr = smem + SMEM_SMEM_K_OFF;
    __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_V_OFF);
    const int smem_v_addr = smem + SMEM_SMEM_V_OFF;
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Q))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&K))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&V))) : "memory");
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 5
    unsigned int ei_valid = 0;
    int ei_batch = 0;
    int ei_head = 0;
    int ei_chunk = 0;
    int ei_len = 0;
    int ei_cnt = 0;
    int ei_last = 0;
    unsigned int ei_row_ok = 0;
    int ei_row[4];
    #pragma unroll
    for (int sl_e = 0; sl_e < 4; sl_e++) {
        ei_row[sl_e] = 0;
    }
    if (warp == 11) {
        unsigned int ei_ticket = blockIdx.x;
        unsigned int ei_items = (unsigned int)(batch_size * num_kv_heads);
        unsigned int ei_tile = ei_ticket;
        if (ei_ticket < ei_items) {
            ei_valid = 1;
            float _rcp_0 = approx_rcp((float)num_kv_heads);
            unsigned int q = (unsigned int)((float)ei_tile * _rcp_0);
            if (ei_tile < q * (unsigned int)num_kv_heads) {
                q = q - 1;
            }
            if (ei_tile >= (q + 1) * (unsigned int)num_kv_heads) {
                q = q + 1;
            }
            ei_batch = (int)q;
            ei_head = (int)(ei_tile - (unsigned int)ei_batch * (unsigned int)num_kv_heads);
            int ei_lastpos_x = causal_seqlens_kv_global[ei_batch] + (q_len - 1) - cp_rank;
            int ei_len_x = 0;
            if (ei_lastpos_x >= 0) {
                ei_len_x = (ei_lastpos_x >> cp_world_log2) + 1;
            }
            int _max_0 = (((ei_len_x + BLOCK_N - 1) / BLOCK_N) > (1) ? ((ei_len_x + BLOCK_N - 1) / BLOCK_N) : (1));
            int ei_nblk_x = _max_0;
            int ei_bbeg_x = 0;
            int ei_bend_x = ei_nblk_x;
            int ei_cnt_x = ei_bend_x - ei_bbeg_x;
            int ei_last_x = ei_bend_x - 1;
            ei_len = ei_len_x;
            ei_cnt = ei_cnt_x;
            ei_last = ei_last_x;
            unsigned int ei_row_ok_x = 0;
            if (max_pages_per_seq <= 128) {
                ei_row_ok_x = 1;
                int ei_pt_base = ei_batch * max_pages_per_seq;
                #pragma unroll
                for (int sl_e_1 = 0; sl_e_1 < 4; sl_e_1++) {
                    int ei_ridx = sl_e_1 * 32 + lane;
                    if (ei_ridx < max_pages_per_seq) {
                        ei_row[sl_e_1] = page_table[ei_pt_base + ei_ridx];
                    }
                }
            }
            ei_row_ok = ei_row_ok_x;
        }
    }
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
#else
#if CAKE_FMHA_DCP_PROGRAM == 4
    unsigned int ei_valid = 0;
    int ei_batch = 0;
    int ei_head = 0;
    int ei_chunk = 0;
    int ei_len = 0;
    int ei_cnt = 0;
    int ei_last = 0;
    unsigned int ei_row_ok = 0;
    int ei_row[4];
    #pragma unroll
    for (int sl_e = 0; sl_e < 4; sl_e++) {
        ei_row[sl_e] = 0;
    }
    if (warp == 11) {
        unsigned int ei_ticket = blockIdx.x;
        unsigned int ei_items = (unsigned int)(batch_size * num_kv_heads);
        unsigned int ei_tile = ei_ticket;
        if (ei_ticket < ei_items) {
            ei_valid = 1;
            float _rcp_0 = approx_rcp((float)num_kv_heads);
            unsigned int q = (unsigned int)((float)ei_tile * _rcp_0);
            if (ei_tile < q * (unsigned int)num_kv_heads) {
                q = q - 1;
            }
            if (ei_tile >= (q + 1) * (unsigned int)num_kv_heads) {
                q = q + 1;
            }
            ei_batch = (int)q;
            ei_head = (int)(ei_tile - (unsigned int)ei_batch * (unsigned int)num_kv_heads);
            int ei_lastpos_x = causal_seqlens_kv_global[ei_batch] + (q_len - 1) - cp_rank;
            int ei_len_x = 0;
            if (ei_lastpos_x >= 0) {
                ei_len_x = (ei_lastpos_x >> cp_world_log2) + 1;
            }
            int _max_0 = (((ei_len_x + BLOCK_N - 1) / BLOCK_N) > (1) ? ((ei_len_x + BLOCK_N - 1) / BLOCK_N) : (1));
            int ei_nblk_x = _max_0;
            int ei_bbeg_x = 0;
            int ei_bend_x = ei_nblk_x;
            int ei_cnt_x = ei_bend_x - ei_bbeg_x;
            int ei_last_x = ei_bend_x - 1;
            ei_len = ei_len_x;
            ei_cnt = ei_cnt_x;
            ei_last = ei_last_x;
            unsigned int ei_row_ok_x = 0;
            if (max_pages_per_seq <= 128) {
                ei_row_ok_x = 1;
                int ei_pt_base = ei_batch * max_pages_per_seq;
                #pragma unroll
                for (int sl_e_1 = 0; sl_e_1 < 4; sl_e_1++) {
                    int ei_ridx = sl_e_1 * 32 + lane;
                    if (ei_ridx < max_pages_per_seq) {
                        ei_row[sl_e_1] = page_table[ei_pt_base + ei_ridx];
                    }
                }
            }
            ei_row_ok = ei_row_ok_x;
        }
    }
#endif  // CAKE_FMHA_DCP_PROGRAM == 4
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5

#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
    // Mbarrier init (20 pipeline groups, 0 ordered-sequence groups, 51 barriers)
    // Mbarriers at smem_raw[0..408)
#else
    // Mbarrier init (18 pipeline groups, 0 ordered-sequence groups, 47 barriers)
    // Mbarriers at smem_raw[0..376)
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
    // Mbarrier init (18 pipeline groups, 0 ordered-sequence groups, 47 barriers)
    // Mbarriers at smem_raw[0..376)
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'q_pipe' ---
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // q_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // --- pipeline 'k_pipe' ---
            // k_full: 3 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            // k_empty: 3 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // --- pipeline 'v_pipe' ---
            // v_full: 3 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            // v_empty: 3 barriers, init_count=1
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            // --- pipeline 'sm_pipe' ---
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // p_full: 2 barriers, init_count=256
            mbarrier_init(smem + 128, 256);
            mbarrier_init(smem + 136, 256);
            // o_ready: 2 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
            // p_full_b: 2 barriers, init_count=256
            mbarrier_init(smem + 160, 256);
            mbarrier_init(smem + 168, 256);
            // pa_done: 2 barriers, init_count=1
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
            // o_empty: 1 barriers, init_count=128
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
            mbarrier_init(smem + 192, 128);
#else
            mbarrier_init(smem + 160, 128);
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
            mbarrier_init(smem + 160, 128);
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
            // --- pipeline 'stats_pipe' ---
            // stats_full: 2 barriers, init_count=128
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
            mbarrier_init(smem + 200, 128);
            mbarrier_init(smem + 208, 128);
#else
            mbarrier_init(smem + 168, 128);
            mbarrier_init(smem + 176, 128);
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
            mbarrier_init(smem + 168, 128);
            mbarrier_init(smem + 176, 128);
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
            // stats_empty: 2 barriers, init_count=4
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
            mbarrier_init(smem + 216, 4);
            mbarrier_init(smem + 224, 4);
#else
            mbarrier_init(smem + 184, 4);
            mbarrier_init(smem + 192, 4);
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
            mbarrier_init(smem + 184, 4);
            mbarrier_init(smem + 192, 4);
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
            // tmem_dealloc: 1 barriers, init_count=128
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
            mbarrier_init(smem + 232, 128);
#else
            mbarrier_init(smem + 200, 128);
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
            mbarrier_init(smem + 200, 128);
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
            // --- pipeline 'page_pipe' ---
            // page_offsets_full: 6 barriers, init_count=1
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 4
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            mbarrier_init(smem + 232, 1);
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 4
#else
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            mbarrier_init(smem + 232, 1);
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
            mbarrier_init(smem + 240, 1);
            mbarrier_init(smem + 248, 1);
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 4
            // page_offsets_empty: 6 barriers, init_count=1
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 4
#else
            // page_offsets_empty: 6 barriers, init_count=1
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
            mbarrier_init(smem + 256, 1);
            mbarrier_init(smem + 264, 1);
            mbarrier_init(smem + 272, 1);
            mbarrier_init(smem + 280, 1);
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
            // page_offsets_empty: 6 barriers, init_count=1
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
            mbarrier_init(smem + 288, 1);
            mbarrier_init(smem + 296, 1);
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 4
            // --- pipeline 'work_pipe' ---
            // work_full: 4 barriers, init_count=1
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 4
#else
            // --- pipeline 'work_pipe' ---
            // work_full: 4 barriers, init_count=1
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
            mbarrier_init(smem + 304, 1);
            mbarrier_init(smem + 312, 1);
            mbarrier_init(smem + 320, 1);
            mbarrier_init(smem + 328, 1);
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
            // --- pipeline 'work_pipe' ---
            // work_full: 4 barriers, init_count=1
            mbarrier_init(smem + 336, 1);
            mbarrier_init(smem + 344, 1);
            mbarrier_init(smem + 352, 1);
            mbarrier_init(smem + 360, 1);
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
            // work_empty: 4 barriers, init_count=352
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
            mbarrier_init(smem + 368, 352);
            mbarrier_init(smem + 376, 352);
            mbarrier_init(smem + 384, 352);
            mbarrier_init(smem + 392, 352);
#else
            mbarrier_init(smem + 336, 352);
            mbarrier_init(smem + 344, 352);
            mbarrier_init(smem + 352, 352);
            mbarrier_init(smem + 360, 352);
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
            mbarrier_init(smem + 336, 352);
            mbarrier_init(smem + 344, 352);
            mbarrier_init(smem + 352, 352);
            mbarrier_init(smem + 360, 352);
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
            // claim_gate: 1 barriers, init_count=1
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
            mbarrier_init(smem + 400, 1);
#else
            mbarrier_init(smem + 368, 1);
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
            mbarrier_init(smem + 368, 1);
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 384 used)
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 408);
#else
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 376);
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 376);
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
    if (warp == 0) {
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
        int _tmem_hold = smem + 408;
#else
        int _tmem_hold = smem + 376;
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
        int _tmem_hold = smem + 376;
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
    const int tmem_tmem_sT = taddr;
    const int tmem_tmem_oT = taddr + 128;
#else
    const int tmem_tmem_s = taddr;
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
    const int tmem_tmem_s = taddr;
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
    const int tmem_tmem_o = taddr + 256;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
    }

    // ---- Role: softmax ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
        { // softmax_main
            const int s_warp = warp;
            int sm_tid = s_warp * 32 + lane;
            int my_row = s_warp * 16 + lane % 16;
            int half = lane / 16;
            int tok_base = half * 64;
            int my_s_base = taddr + (unsigned int)(s_warp * 32 << 16);
            int rows_live = ((s_warp * 16 < N_ROWS) ? 1 : 0);
            int row_j = my_row / 8;
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 5
            int _min_1 = ((row_j) < (q_len - 1) ? (row_j) : (q_len - 1));
            int vis_j = _min_1;
#else
            int _min_0 = ((row_j) < (q_len - 1) ? (row_j) : (q_len - 1));
            int vis_j = _min_0;
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
#else
#if CAKE_FMHA_DCP_PROGRAM == 4
            int _min_1 = ((row_j) < (q_len - 1) ? (row_j) : (q_len - 1));
            int vis_j = _min_1;
#else
            int _min_0 = ((row_j) < (q_len - 1) ? (row_j) : (q_len - 1));
            int vis_j = _min_0;
#endif  // CAKE_FMHA_DCP_PROGRAM == 4
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
            int _min_1 = ((row_j) < (q_len - 1) ? (row_j) : (q_len - 1));
            int vis_j = _min_1;
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
            unsigned int sm_stage = 0;
            unsigned int sm_phase = 0;
            unsigned int xm_slot_s = 0;
            unsigned int st_stage_s = 0;
            unsigned int st_phase_s = 1;
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 5
            float _rcp_3 = approx_rcp(softmax_scale_log2);
            float thr_raw = 8.0f * _rcp_3;
#else
            float _rcp_2 = approx_rcp(softmax_scale_log2);
            float thr_raw = 8.0f * _rcp_2;
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
#else
#if CAKE_FMHA_DCP_PROGRAM == 4
            float _rcp_3 = approx_rcp(softmax_scale_log2);
            float thr_raw = 8.0f * _rcp_3;
#else
            float _rcp_2 = approx_rcp(softmax_scale_log2);
            float thr_raw = 8.0f * _rcp_2;
#endif  // CAKE_FMHA_DCP_PROGRAM == 4
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
            float _rcp_14 = approx_rcp(softmax_scale_log2);
            float thr_raw = 8.0f * _rcp_14;
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
            unsigned int work_stage_s = 0;
            unsigned int _phase_work_full = 0;
            mbarrier_wait(work_full_addr + (work_stage_s) * 8, _phase_work_full);
            unsigned int base = work_stage_s * 16;
            unsigned int valid = work_token_words[base];
            unsigned int kind = work_token_words[base + 1];
            unsigned int batch = work_token_words[base + 2];
            unsigned int kv_head = work_token_words[base + 3];
            unsigned int block_begin = work_token_words[base + 4];
            unsigned int block_end = work_token_words[base + 5];
            unsigned int seqlen = work_token_words[base + 6];
            unsigned int n_chunks = work_token_words[base + 7];
            unsigned int slot_tile_base = work_token_words[base + 8];
            unsigned int counter_idx = work_token_words[base + 9];
            unsigned int chunk = work_token_words[base + 10];
            unsigned int phase = work_token_words[base + 11];
            mbarrier_arrive(work_empty_addr + (work_stage_s) * 8);
            work_stage_s += 1;
            if (work_stage_s == 4) { work_stage_s = 0; _phase_work_full ^= 1; }
            unsigned int valid_s = valid;
            int kind_s = (int)kind;
            int block_begin_s = (int)block_begin;
            int block_end_s = (int)block_end;
            int seqlen_s = (int)seqlen;
            int batch_s = (int)batch;
            int kv_head_s = (int)kv_head;
            int n_chunks_s = (int)n_chunks;
            int slot_tile_base_s = (int)slot_tile_base;
            int phase_s = (int)phase;
            int live_rows_s = q_len * 8;
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
            unsigned int pv_base_s = 0;
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
            #pragma unroll 1
            for (unsigned int _tile_iter_s = 0; _tile_iter_s < max_items; _tile_iter_s++) {
                if (valid_s == 0) {
                    break;
                }
                if (kind_s == 0) {
                    int cnt_s = block_end_s - block_begin_s;
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                    int k_w = s_warp * 32 + lane;
                    int rows_hi = ((N_ROWS > 32) ? 1 : 0);
                    int sT_base = taddr + (unsigned int)(s_warp * 32 << 16);
                    int oT_base_w = taddr + 128 + (unsigned int)(s_warp * 32 << 16);
                    int vis_a[16];
                    int vis_b[16];
                    #pragma unroll
                    for (int j = 0; j < 16; j++) {
                        int row_j_v = j / 8;
#if CAKE_FMHA_DCP_PROGRAM == 5
                        int _min_2 = ((row_j_v) < (q_len - 1) ? (row_j_v) : (q_len - 1));
                        int vis_j_v = _min_2;
#else
                        int _min_1 = ((row_j_v) < (q_len - 1) ? (row_j_v) : (q_len - 1));
                        int vis_j_v = _min_1;
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
                        int back_v = q_len - 1 - vis_j_v - phase_s;
                        int vis_v = seqlen_s;
                        if (back_v > 0) {
                            vis_v = seqlen_s - (back_v + (1 << cp_world_log2) - 1 >> cp_world_log2);
                        }
                        vis_a[j] = vis_v;
                        int row_j_v_0 = (32 + j) / 8;
#if CAKE_FMHA_DCP_PROGRAM == 5
                        int _min_3 = ((row_j_v_0) < (q_len - 1) ? (row_j_v_0) : (q_len - 1));
                        int vis_j_v_1 = _min_3;
#else
                        int _min_2 = ((row_j_v_0) < (q_len - 1) ? (row_j_v_0) : (q_len - 1));
                        int vis_j_v_1 = _min_2;
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
                        int back_v_2 = q_len - 1 - vis_j_v_1 - phase_s;
                        int vis_v_3 = seqlen_s;
                        if (back_v_2 > 0) {
                            vis_v_3 = seqlen_s - (back_v_2 + (1 << cp_world_log2) - 1 >> cp_world_log2);
                        }
                        vis_b[j] = vis_v_3;
#else
                    int back_s = q_len - 1 - vis_j - phase_s;
                    int vis_col = seqlen_s;
                    if (back_s > 0) {
                        vis_col = seqlen_s - (back_s + (1 << cp_world_log2) - 1 >> cp_world_log2);
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                    int back_s = q_len - 1 - vis_j - phase_s;
                    int vis_col = seqlen_s;
                    if (back_s > 0) {
                        vis_col = seqlen_s - (back_s + (1 << cp_world_log2) - 1 >> cp_world_log2);
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                    }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                    float ref_a[16];
                    float ref_b[16];
                    float psum_a[16];
                    float psum_b[16];
                    ref_a[0] = -CAKE_FMHA_INF;
                    ref_a[1] = -CAKE_FMHA_INF;
                    ref_a[2] = -CAKE_FMHA_INF;
                    ref_a[3] = -CAKE_FMHA_INF;
                    ref_a[4] = -CAKE_FMHA_INF;
                    ref_a[5] = -CAKE_FMHA_INF;
                    ref_a[6] = -CAKE_FMHA_INF;
                    ref_a[7] = -CAKE_FMHA_INF;
                    ref_a[8] = -CAKE_FMHA_INF;
                    ref_a[9] = -CAKE_FMHA_INF;
                    ref_a[10] = -CAKE_FMHA_INF;
                    ref_a[11] = -CAKE_FMHA_INF;
                    ref_a[12] = -CAKE_FMHA_INF;
                    ref_a[13] = -CAKE_FMHA_INF;
                    ref_a[14] = -CAKE_FMHA_INF;
                    ref_a[15] = -CAKE_FMHA_INF;
                    ref_b[0] = -CAKE_FMHA_INF;
                    ref_b[1] = -CAKE_FMHA_INF;
                    ref_b[2] = -CAKE_FMHA_INF;
                    ref_b[3] = -CAKE_FMHA_INF;
                    ref_b[4] = -CAKE_FMHA_INF;
                    ref_b[5] = -CAKE_FMHA_INF;
                    ref_b[6] = -CAKE_FMHA_INF;
                    ref_b[7] = -CAKE_FMHA_INF;
                    ref_b[8] = -CAKE_FMHA_INF;
                    ref_b[9] = -CAKE_FMHA_INF;
                    ref_b[10] = -CAKE_FMHA_INF;
                    ref_b[11] = -CAKE_FMHA_INF;
                    ref_b[12] = -CAKE_FMHA_INF;
                    ref_b[13] = -CAKE_FMHA_INF;
                    ref_b[14] = -CAKE_FMHA_INF;
                    ref_b[15] = -CAKE_FMHA_INF;
                    psum_a[0] = 0.0f;
                    psum_a[1] = 0.0f;
                    psum_a[2] = 0.0f;
                    psum_a[3] = 0.0f;
                    psum_a[4] = 0.0f;
                    psum_a[5] = 0.0f;
                    psum_a[6] = 0.0f;
                    psum_a[7] = 0.0f;
                    psum_a[8] = 0.0f;
                    psum_a[9] = 0.0f;
                    psum_a[10] = 0.0f;
                    psum_a[11] = 0.0f;
                    psum_a[12] = 0.0f;
                    psum_a[13] = 0.0f;
                    psum_a[14] = 0.0f;
                    psum_a[15] = 0.0f;
                    psum_b[0] = 0.0f;
                    psum_b[1] = 0.0f;
                    psum_b[2] = 0.0f;
                    psum_b[3] = 0.0f;
                    psum_b[4] = 0.0f;
                    psum_b[5] = 0.0f;
                    psum_b[6] = 0.0f;
                    psum_b[7] = 0.0f;
                    psum_b[8] = 0.0f;
                    psum_b[9] = 0.0f;
                    psum_b[10] = 0.0f;
                    psum_b[11] = 0.0f;
                    psum_b[12] = 0.0f;
                    psum_b[13] = 0.0f;
                    psum_b[14] = 0.0f;
                    psum_b[15] = 0.0f;
#else
                    float row_max = -CAKE_FMHA_INF;
                    float psum = 0.0f;
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                    float row_max = -CAKE_FMHA_INF;
                    float psum = 0.0f;
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                    #pragma unroll 1
                    for (int n = 0; n < cnt_s; n++) {
                        if (sm_tid == 0) {
                        }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                        {
                            mbarrier_wait(s_full_addr + (sm_stage) * 8, sm_phase);
                        }
#else
                        mbarrier_wait(s_full_addr + (sm_stage) * 8, sm_phase);
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                        mbarrier_wait(s_full_addr + (sm_stage) * 8, sm_phase);
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                        if (sm_tid == 0) {
                        }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                        int my_block_w = block_begin_s + cnt_s - 1 - n;
                        int kpos = my_block_w * BLOCK_N + k_w;
                        int par_w = n & 1;
                        float sv_a[16];
                        float sv_b[16];
                        tmem_ld_x16(&sv_a[0], (unsigned int)sT_base + sm_stage * 64);
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        if (rows_hi != 0) {
                            tmem_ld_x16(&sv_b[0], (unsigned int)sT_base + sm_stage * 64 + 32);
#else
                        int my_block = block_begin_s + cnt_s - 1 - n;
                        int blk_pos = my_block * BLOCK_N + tok_base;
                        float sv[32];
                        float lmax = -CAKE_FMHA_INF;
                        float new_max = row_max;
                        float acc_scale = 1.0f;
                        float lsum = 0.0f;
                        int xm_off_s = (int)xm_slot_s * 128;
                        if (rows_live != 0) {
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                                : "=r"(*reinterpret_cast<uint32_t*>(&sv[0])), "=r"(*reinterpret_cast<uint32_t*>(&sv[1])), "=r"(*reinterpret_cast<uint32_t*>(&sv[2])), "=r"(*reinterpret_cast<uint32_t*>(&sv[3])), "=r"(*reinterpret_cast<uint32_t*>(&sv[4])), "=r"(*reinterpret_cast<uint32_t*>(&sv[5])), "=r"(*reinterpret_cast<uint32_t*>(&sv[6])), "=r"(*reinterpret_cast<uint32_t*>(&sv[7])), "=r"(*reinterpret_cast<uint32_t*>(&sv[8])), "=r"(*reinterpret_cast<uint32_t*>(&sv[9])), "=r"(*reinterpret_cast<uint32_t*>(&sv[10])), "=r"(*reinterpret_cast<uint32_t*>(&sv[11])), "=r"(*reinterpret_cast<uint32_t*>(&sv[12])), "=r"(*reinterpret_cast<uint32_t*>(&sv[13])), "=r"(*reinterpret_cast<uint32_t*>(&sv[14])), "=r"(*reinterpret_cast<uint32_t*>(&sv[15])), "=r"(*reinterpret_cast<uint32_t*>(&sv[16])), "=r"(*reinterpret_cast<uint32_t*>(&sv[17])), "=r"(*reinterpret_cast<uint32_t*>(&sv[18])), "=r"(*reinterpret_cast<uint32_t*>(&sv[19])), "=r"(*reinterpret_cast<uint32_t*>(&sv[20])), "=r"(*reinterpret_cast<uint32_t*>(&sv[21])), "=r"(*reinterpret_cast<uint32_t*>(&sv[22])), "=r"(*reinterpret_cast<uint32_t*>(&sv[23])), "=r"(*reinterpret_cast<uint32_t*>(&sv[24])), "=r"(*reinterpret_cast<uint32_t*>(&sv[25])), "=r"(*reinterpret_cast<uint32_t*>(&sv[26])), "=r"(*reinterpret_cast<uint32_t*>(&sv[27])), "=r"(*reinterpret_cast<uint32_t*>(&sv[28])), "=r"(*reinterpret_cast<uint32_t*>(&sv[29])), "=r"(*reinterpret_cast<uint32_t*>(&sv[30])), "=r"(*reinterpret_cast<uint32_t*>(&sv[31]))
                                : "r"((unsigned int)my_s_base + sm_stage * (unsigned int)BLOCK_N));
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                        int my_block = block_begin_s + cnt_s - 1 - n;
                        int blk_pos = my_block * BLOCK_N + tok_base;
                        float sv[32];
                        float lmax = -CAKE_FMHA_INF;
                        float new_max = row_max;
                        float acc_scale = 1.0f;
                        float lsum = 0.0f;
                        int xm_off_s = (int)xm_slot_s * 128;
                        if (rows_live != 0) {
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                                : "=r"(*reinterpret_cast<uint32_t*>(&sv[0])), "=r"(*reinterpret_cast<uint32_t*>(&sv[1])), "=r"(*reinterpret_cast<uint32_t*>(&sv[2])), "=r"(*reinterpret_cast<uint32_t*>(&sv[3])), "=r"(*reinterpret_cast<uint32_t*>(&sv[4])), "=r"(*reinterpret_cast<uint32_t*>(&sv[5])), "=r"(*reinterpret_cast<uint32_t*>(&sv[6])), "=r"(*reinterpret_cast<uint32_t*>(&sv[7])), "=r"(*reinterpret_cast<uint32_t*>(&sv[8])), "=r"(*reinterpret_cast<uint32_t*>(&sv[9])), "=r"(*reinterpret_cast<uint32_t*>(&sv[10])), "=r"(*reinterpret_cast<uint32_t*>(&sv[11])), "=r"(*reinterpret_cast<uint32_t*>(&sv[12])), "=r"(*reinterpret_cast<uint32_t*>(&sv[13])), "=r"(*reinterpret_cast<uint32_t*>(&sv[14])), "=r"(*reinterpret_cast<uint32_t*>(&sv[15])), "=r"(*reinterpret_cast<uint32_t*>(&sv[16])), "=r"(*reinterpret_cast<uint32_t*>(&sv[17])), "=r"(*reinterpret_cast<uint32_t*>(&sv[18])), "=r"(*reinterpret_cast<uint32_t*>(&sv[19])), "=r"(*reinterpret_cast<uint32_t*>(&sv[20])), "=r"(*reinterpret_cast<uint32_t*>(&sv[21])), "=r"(*reinterpret_cast<uint32_t*>(&sv[22])), "=r"(*reinterpret_cast<uint32_t*>(&sv[23])), "=r"(*reinterpret_cast<uint32_t*>(&sv[24])), "=r"(*reinterpret_cast<uint32_t*>(&sv[25])), "=r"(*reinterpret_cast<uint32_t*>(&sv[26])), "=r"(*reinterpret_cast<uint32_t*>(&sv[27])), "=r"(*reinterpret_cast<uint32_t*>(&sv[28])), "=r"(*reinterpret_cast<uint32_t*>(&sv[29])), "=r"(*reinterpret_cast<uint32_t*>(&sv[30])), "=r"(*reinterpret_cast<uint32_t*>(&sv[31]))
                                : "r"((unsigned int)my_s_base + sm_stage * (unsigned int)BLOCK_N));
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                        }
                        int exc_w = 0;
                        #pragma unroll
                        for (int j_1 = 0; j_1 < 16; j_1++) {
                            if (kpos >= vis_a[j_1]) {
                                sv_a[j_1] = -CAKE_FMHA_INF;
#else
                            int n_vis = vis_col - blk_pos;
                            if (my_row >= N_ROWS) {
                                n_vis = 0;
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                            int n_vis = vis_col - blk_pos;
                            if (my_row >= N_ROWS) {
                                n_vis = 0;
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                            }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                            if (sv_a[j_1] > ref_a[j_1] + thr_raw) {
                                exc_w = 1;
                            }
                        }
                        if (rows_hi != 0) {
                            #pragma unroll
                            for (int j_2 = 0; j_2 < 16; j_2++) {
                                if (kpos >= vis_b[j_2]) {
                                    sv_b[j_2] = -CAKE_FMHA_INF;
                                }
                                if (sv_b[j_2] > ref_b[j_2] + thr_raw) {
                                    exc_w = 1;
                                }
                            }
                        }
                        unsigned int flag_w = 0;
                        int _vote_0 = __any_sync(0xFFFFFFFF, exc_w != 0);
                        if (_vote_0 != 0) {
                            #pragma unroll
                            for (int j_3 = 0; j_3 < 16; j_3++) {
                                int _vote_1 = __any_sync(0xFFFFFFFF, sv_a[j_3] > ref_a[j_3] + thr_raw);
                                if (_vote_1 != 0) {
                                    flag_w = flag_w | (unsigned int)(1 << j_3);
                                }
                            }
                            if (rows_hi != 0) {
                                #pragma unroll
                                for (int j_4 = 0; j_4 < 16; j_4++) {
                                    int _vote_2 = __any_sync(0xFFFFFFFF, sv_b[j_4] > ref_b[j_4] + thr_raw);
                                    if (_vote_2 != 0) {
                                        flag_w = flag_w | (unsigned int)(1 << 16 + j_4);
#else
                            if (n_vis < 32) {
#if CAKE_FMHA_DCP_PROGRAM == 4
                                int _max_5 = ((n_vis) > (0) ? (n_vis) : (0));
                                int n_lo = _max_5;
#else
                                int _max_3 = ((n_vis) > (0) ? (n_vis) : (0));
                                int n_lo = _max_3;
#endif  // CAKE_FMHA_DCP_PROGRAM == 4
                                uint32_t _slice_lo_mask_0;
                                {
                                    int _lim_0 = n_lo;
                                    if (_lim_0 <= 0) { _slice_lo_mask_0 = 0u; }
                                    else if (_lim_0 >= 32) { _slice_lo_mask_0 = 0xFFFFFFFFu; }
                                    else {
                                        asm volatile("{"
                                            ".reg .u32 t;\n\t"
                                            "shl.b32 t, 1, %1;\n\t"
                                            "add.u32 %0, t, -1;\n\t"
                                            "}" : "=r"(_slice_lo_mask_0) : "r"(_lim_0));
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                            if (n_vis < 32) {
                                int _max_15 = ((n_vis) > (0) ? (n_vis) : (0));
                                int n_lo = _max_15;
                                uint32_t _slice_lo_mask_0;
                                {
                                    int _lim_0 = n_lo;
                                    if (_lim_0 <= 0) { _slice_lo_mask_0 = 0u; }
                                    else if (_lim_0 >= 32) { _slice_lo_mask_0 = 0xFFFFFFFFu; }
                                    else {
                                        asm volatile("{"
                                            ".reg .u32 t;\n\t"
                                            "shl.b32 t, 1, %1;\n\t"
                                            "add.u32 %0, t, -1;\n\t"
                                            "}" : "=r"(_slice_lo_mask_0) : "r"(_lim_0));
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                                    }
                                }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 4
                                if (!(_slice_lo_mask_0 & (1u << 0))) sv[0] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 1))) sv[1] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 2))) sv[2] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 3))) sv[3] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 4))) sv[4] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 5))) sv[5] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 6))) sv[6] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 7))) sv[7] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 8))) sv[8] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 9))) sv[9] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 10))) sv[10] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 11))) sv[11] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 12))) sv[12] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 13))) sv[13] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 14))) sv[14] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 15))) sv[15] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 16))) sv[16] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 17))) sv[17] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 18))) sv[18] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 19))) sv[19] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 20))) sv[20] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 21))) sv[21] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 22))) sv[22] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 23))) sv[23] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 24))) sv[24] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 25))) sv[25] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 26))) sv[26] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 27))) sv[27] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 28))) sv[28] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 29))) sv[29] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 30))) sv[30] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 31))) sv[31] = -CAKE_FMHA_INF;
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 4
#else
                                if (!(_slice_lo_mask_0 & (1u << 0))) sv[0] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 1))) sv[1] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 2))) sv[2] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 3))) sv[3] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 4))) sv[4] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 5))) sv[5] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 6))) sv[6] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 7))) sv[7] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 8))) sv[8] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 9))) sv[9] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 10))) sv[10] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 11))) sv[11] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 12))) sv[12] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 13))) sv[13] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 14))) sv[14] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 15))) sv[15] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 16))) sv[16] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 17))) sv[17] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 18))) sv[18] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 19))) sv[19] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 20))) sv[20] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 21))) sv[21] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 22))) sv[22] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 23))) sv[23] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 24))) sv[24] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 25))) sv[25] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 26))) sv[26] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 27))) sv[27] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 28))) sv[28] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 29))) sv[29] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 30))) sv[30] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 31))) sv[31] = -CAKE_FMHA_INF;
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                            }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                        }
                        if (lane == 0) {
                            swap_flags[par_w * 4 + s_warp] = flag_w;
                        }
                        asm volatile("barrier.sync 8, 128;" ::: "memory");
                        unsigned int fl_w = swap_flags[par_w * 4] | swap_flags[par_w * 4 + 1] | swap_flags[par_w * 4 + 2] | swap_flags[par_w * 4 + 3];
                        if (sm_tid == 0) {
                        }
                        if (fl_w != 0) {
                            int b4 = lane >> 4 & 1;
                            int b3 = lane >> 3 & 1;
                            int b2 = lane >> 2 & 1;
                            int b1 = lane >> 1 & 1;
                            float w8[8];
                            #pragma unroll
                            for (int i = 0; i < 8; i++) {
                                float send8 = ((b4 != 0) ? sv_a[i] : sv_a[i + 8]);
                                float keep8 = ((b4 != 0) ? sv_a[i + 8] : sv_a[i]);
                                float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, send8, 16);
#if CAKE_FMHA_DCP_PROGRAM == 5
                                float _max_5 = max_noftz(keep8, _shfl_xor_0);
                                w8[i] = _max_5;
#else
                                float _max_3 = max_noftz(keep8, _shfl_xor_0);
                                w8[i] = _max_3;
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
                            }
                            float x4[4];
                            #pragma unroll
                            for (int i_1 = 0; i_1 < 4; i_1++) {
                                float send4 = ((b3 != 0) ? w8[i_1] : w8[i_1 + 4]);
                                float keep4 = ((b3 != 0) ? w8[i_1 + 4] : w8[i_1]);
                                float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, send4, 8);
#if CAKE_FMHA_DCP_PROGRAM == 5
                                float _max_6 = max_noftz(keep4, _shfl_xor_1);
                                x4[i_1] = _max_6;
#else
                                float _max_4 = max_noftz(keep4, _shfl_xor_1);
                                x4[i_1] = _max_4;
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
                            }
                            float y2[2];
                            #pragma unroll
                            for (int i_2 = 0; i_2 < 2; i_2++) {
                                float send2 = ((b2 != 0) ? x4[i_2] : x4[i_2 + 2]);
                                float keep2 = ((b2 != 0) ? x4[i_2 + 2] : x4[i_2]);
                                float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, send2, 4);
#if CAKE_FMHA_DCP_PROGRAM == 5
                                float _max_7 = max_noftz(keep2, _shfl_xor_2);
                                y2[i_2] = _max_7;
#else
                                float _max_5 = max_noftz(keep2, _shfl_xor_2);
                                y2[i_2] = _max_5;
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
                            }
                            float send1 = ((b1 != 0) ? y2[0] : y2[1]);
                            float keep1 = ((b1 != 0) ? y2[1] : y2[0]);
                            float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, send1, 2);
#if CAKE_FMHA_DCP_PROGRAM == 5
                            float _max_8 = max_noftz(keep1, _shfl_xor_3);
                            float z = _max_8;
#else
                            float _max_6 = max_noftz(keep1, _shfl_xor_3);
                            float z = _max_6;
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
                            float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, z, 1);
#if CAKE_FMHA_DCP_PROGRAM == 5
                            float _max_9 = max_noftz(z, _shfl_xor_4);
                            z = _max_9;
#else
                            float _max_7 = max_noftz(z, _shfl_xor_4);
                            z = _max_7;
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
                            float zmax = z;
                            if ((lane & 1) == 0) {
                                swap_part[(lane >> 1 & 15) * 4 + s_warp] = zmax;
                            }
                            if (rows_hi != 0) {
                                int b4_0 = lane >> 4 & 1;
                                int b3_1 = lane >> 3 & 1;
                                int b2_2 = lane >> 2 & 1;
                                int b1_3 = lane >> 1 & 1;
                                float w8_4[8];
                                #pragma unroll
                                for (int i_3 = 0; i_3 < 8; i_3++) {
                                    float send8_1 = ((b4_0 != 0) ? sv_b[i_3] : sv_b[i_3 + 8]);
                                    float keep8_1 = ((b4_0 != 0) ? sv_b[i_3 + 8] : sv_b[i_3]);
                                    float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, send8_1, 16);
#if CAKE_FMHA_DCP_PROGRAM == 5
                                    float _max_10 = max_noftz(keep8_1, _shfl_xor_5);
                                    w8_4[i_3] = _max_10;
#else
                                    float _max_8 = max_noftz(keep8_1, _shfl_xor_5);
                                    w8_4[i_3] = _max_8;
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
                                }
                                float x4_5[4];
                                #pragma unroll
                                for (int i_4 = 0; i_4 < 4; i_4++) {
                                    float send4_1 = ((b3_1 != 0) ? w8_4[i_4] : w8_4[i_4 + 4]);
                                    float keep4_1 = ((b3_1 != 0) ? w8_4[i_4 + 4] : w8_4[i_4]);
                                    float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, send4_1, 8);
#if CAKE_FMHA_DCP_PROGRAM == 5
                                    float _max_11 = max_noftz(keep4_1, _shfl_xor_6);
                                    x4_5[i_4] = _max_11;
#else
                                    float _max_9 = max_noftz(keep4_1, _shfl_xor_6);
                                    x4_5[i_4] = _max_9;
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
                                }
                                float y2_6[2];
                                #pragma unroll
                                for (int i_5 = 0; i_5 < 2; i_5++) {
                                    float send2_1 = ((b2_2 != 0) ? x4_5[i_5] : x4_5[i_5 + 2]);
                                    float keep2_1 = ((b2_2 != 0) ? x4_5[i_5 + 2] : x4_5[i_5]);
                                    float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, send2_1, 4);
#if CAKE_FMHA_DCP_PROGRAM == 5
                                    float _max_12 = max_noftz(keep2_1, _shfl_xor_7);
                                    y2_6[i_5] = _max_12;
#else
                                    float _max_10 = max_noftz(keep2_1, _shfl_xor_7);
                                    y2_6[i_5] = _max_10;
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
                                }
                                float send1_7 = ((b1_3 != 0) ? y2_6[0] : y2_6[1]);
                                float keep1_8 = ((b1_3 != 0) ? y2_6[1] : y2_6[0]);
                                float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, send1_7, 2);
#if CAKE_FMHA_DCP_PROGRAM == 5
                                float _max_13 = max_noftz(keep1_8, _shfl_xor_8);
                                float z_9 = _max_13;
#else
                                float _max_11 = max_noftz(keep1_8, _shfl_xor_8);
                                float z_9 = _max_11;
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
                                float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, z_9, 1);
#if CAKE_FMHA_DCP_PROGRAM == 5
                                float _max_14 = max_noftz(z_9, _shfl_xor_9);
                                z_9 = _max_14;
#else
                                float _max_12 = max_noftz(z_9, _shfl_xor_9);
                                z_9 = _max_12;
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
                                float zmax_10 = z_9;
                                if ((lane & 1) == 0) {
                                    swap_part[(16 + (lane >> 1 & 15)) * 4 + s_warp] = zmax_10;
                                }
                            }
                            if (sm_tid == 0) {
                            }
                            asm volatile("barrier.sync 8, 128;" ::: "memory");
                            if (sm_tid == 0) {
                            }
                            if (n == 0) {
                                float p64s[64];
                                #pragma unroll
                                for (int j_5 = 0; j_5 < 16; j_5++) {
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&p64s[4 * j_5])), "=r"(*reinterpret_cast<uint32_t*>(&p64s[(4 * j_5) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&p64s[(4 * j_5) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&p64s[(4 * j_5) + 3]))
                                        : "r"(swap_part_addr + (unsigned int)(j_5 * 16)));
                                }
                                #pragma unroll
                                for (int j_6 = 0; j_6 < 16; j_6++) {
#if CAKE_FMHA_DCP_PROGRAM == 5
                                    float _max_15 = max_noftz(p64s[4 * j_6], p64s[4 * j_6 + 1]);
                                    float _max_16 = max_noftz(p64s[4 * j_6 + 2], p64s[4 * j_6 + 3]);
                                    float _max_17 = max_noftz(_max_15, _max_16);
                                    float bm_sj = _max_17;
#else
                                    float _max_13 = max_noftz(p64s[4 * j_6], p64s[4 * j_6 + 1]);
                                    float _max_14 = max_noftz(p64s[4 * j_6 + 2], p64s[4 * j_6 + 3]);
                                    float _max_15 = max_noftz(_max_13, _max_14);
                                    float bm_sj = _max_15;
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
                                    int flagged_sj = fl_w >> (unsigned int)j_6 & 1;
                                    ref_a[j_6] = ((flagged_sj != 0) ? bm_sj : ref_a[j_6]);
                                }
                                if (rows_hi != 0) {
                                    float p64s_0[64];
                                    #pragma unroll
                                    for (int j_7 = 0; j_7 < 16; j_7++) {
                                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                            : "=r"(*reinterpret_cast<uint32_t*>(&p64s_0[4 * j_7])), "=r"(*reinterpret_cast<uint32_t*>(&p64s_0[(4 * j_7) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&p64s_0[(4 * j_7) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&p64s_0[(4 * j_7) + 3]))
                                            : "r"(swap_part_addr + (unsigned int)((16 + j_7) * 16)));
                                    }
                                    #pragma unroll
                                    for (int j_8 = 0; j_8 < 16; j_8++) {
#if CAKE_FMHA_DCP_PROGRAM == 5
                                        float _max_18 = max_noftz(p64s_0[4 * j_8], p64s_0[4 * j_8 + 1]);
                                        float _max_19 = max_noftz(p64s_0[4 * j_8 + 2], p64s_0[4 * j_8 + 3]);
                                        float _max_20 = max_noftz(_max_18, _max_19);
                                        float bm_sj_1 = _max_20;
#else
                                        float _max_16 = max_noftz(p64s_0[4 * j_8], p64s_0[4 * j_8 + 1]);
                                        float _max_17 = max_noftz(p64s_0[4 * j_8 + 2], p64s_0[4 * j_8 + 3]);
                                        float _max_18 = max_noftz(_max_16, _max_17);
                                        float bm_sj_1 = _max_18;
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
                                        int flagged_sj_1 = fl_w >> (unsigned int)(16 + j_8) & 1;
                                        ref_b[j_8] = ((flagged_sj_1 != 0) ? bm_sj_1 : ref_b[j_8]);
                                    }
                                }
                            } else {
                                float p64[64];
                                #pragma unroll
                                for (int j_9 = 0; j_9 < 16; j_9++) {
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&p64[4 * j_9])), "=r"(*reinterpret_cast<uint32_t*>(&p64[(4 * j_9) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&p64[(4 * j_9) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&p64[(4 * j_9) + 3]))
                                        : "r"(swap_part_addr + (unsigned int)(j_9 * 16)));
                                }
                                float my_acc = 1.0f;
                                #pragma unroll
                                for (int j_10 = 0; j_10 < 16; j_10++) {
#if CAKE_FMHA_DCP_PROGRAM == 5
                                    float _max_21 = max_noftz(p64[4 * j_10], p64[4 * j_10 + 1]);
                                    float _max_22 = max_noftz(p64[4 * j_10 + 2], p64[4 * j_10 + 3]);
                                    float _max_23 = max_noftz(_max_21, _max_22);
                                    float bm_j = _max_23;
#else
                                    float _max_19 = max_noftz(p64[4 * j_10], p64[4 * j_10 + 1]);
                                    float _max_20 = max_noftz(p64[4 * j_10 + 2], p64[4 * j_10 + 3]);
                                    float _max_21 = max_noftz(_max_19, _max_20);
                                    float bm_j = _max_21;
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
                                    int flagged_j = fl_w >> (unsigned int)j_10 & 1;
                                    float _exp2_0 = approx_exp2(softmax_scale_log2 * (ref_a[j_10] - bm_j));
                                    float e_j = _exp2_0;
                                    float acc_j = ((flagged_j != 0 && ref_a[j_10] > -CAKE_FMHA_INF) ? e_j : 1.0f);
                                    psum_a[j_10] = psum_a[j_10] * acc_j;
                                    ref_a[j_10] = ((flagged_j != 0) ? bm_j : ref_a[j_10]);
                                    my_acc = ((lane == j_10) ? acc_j : my_acc);
                                }
                                if (s_warp == 0) {
                                    if (lane < 16) {
                                        swap_acc[lane] = my_acc;
                                    }
                                }
                                if (rows_hi != 0) {
                                    float p64_0[64];
                                    #pragma unroll
                                    for (int j_11 = 0; j_11 < 16; j_11++) {
                                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                            : "=r"(*reinterpret_cast<uint32_t*>(&p64_0[4 * j_11])), "=r"(*reinterpret_cast<uint32_t*>(&p64_0[(4 * j_11) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&p64_0[(4 * j_11) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&p64_0[(4 * j_11) + 3]))
                                            : "r"(swap_part_addr + (unsigned int)((16 + j_11) * 16)));
                                    }
                                    float my_acc_1 = 1.0f;
                                    #pragma unroll
                                    for (int j_12 = 0; j_12 < 16; j_12++) {
#if CAKE_FMHA_DCP_PROGRAM == 5
                                        float _max_24 = max_noftz(p64_0[4 * j_12], p64_0[4 * j_12 + 1]);
                                        float _max_25 = max_noftz(p64_0[4 * j_12 + 2], p64_0[4 * j_12 + 3]);
                                        float _max_26 = max_noftz(_max_24, _max_25);
                                        float bm_j_1 = _max_26;
#else
                                        float _max_22 = max_noftz(p64_0[4 * j_12], p64_0[4 * j_12 + 1]);
                                        float _max_23 = max_noftz(p64_0[4 * j_12 + 2], p64_0[4 * j_12 + 3]);
                                        float _max_24 = max_noftz(_max_22, _max_23);
                                        float bm_j_1 = _max_24;
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
                                        int flagged_j_1 = fl_w >> (unsigned int)(16 + j_12) & 1;
                                        float _exp2_1 = approx_exp2(softmax_scale_log2 * (ref_b[j_12] - bm_j_1));
                                        float e_j_1 = _exp2_1;
                                        float acc_j_1 = ((flagged_j_1 != 0 && ref_b[j_12] > -CAKE_FMHA_INF) ? e_j_1 : 1.0f);
                                        psum_b[j_12] = psum_b[j_12] * acc_j_1;
                                        ref_b[j_12] = ((flagged_j_1 != 0) ? bm_j_1 : ref_b[j_12]);
                                        my_acc_1 = ((lane == j_12) ? acc_j_1 : my_acc_1);
                                    }
                                    if (s_warp == 0) {
                                        if (lane < 16) {
                                            swap_acc[32 + lane] = my_acc_1;
                                        }
                                    }
                                }
                            }
                            if (sm_tid == 0) {
                            }
                            asm volatile("barrier.sync 8, 128;" ::: "memory");
                        }
                        if (sm_tid == 0) {
                        }
                        #pragma unroll
                        for (int j_13 = 0; j_13 < 16; j_13++) {
                            float safe_j = ((ref_a[j_13] == -CAKE_FMHA_INF) ? 0.0f : ref_a[j_13]);
                            float _fma_0 = __fmaf_rn(sv_a[j_13], softmax_scale_log2, (-safe_j) * softmax_scale_log2);
                            sv_a[j_13] = _fma_0;
                        }
                        #pragma unroll
                        for (int _le = 0; _le < 16; _le++) {
                            sv_a[_le] = approx_exp2(sv_a[_le]);
                        }
                        #pragma unroll
                        for (int j_14 = 0; j_14 < 16; j_14++) {
                            psum_a[j_14] = psum_a[j_14] + sv_a[j_14];
                        }
                        if (rows_hi != 0) {
                            #pragma unroll
                            for (int j_15 = 0; j_15 < 16; j_15++) {
                                float safe_j_1 = ((ref_b[j_15] == -CAKE_FMHA_INF) ? 0.0f : ref_b[j_15]);
                                float _fma_1 = __fmaf_rn(sv_b[j_15], softmax_scale_log2, (-safe_j_1) * softmax_scale_log2);
                                sv_b[j_15] = _fma_1;
                            }
                            #pragma unroll
                            for (int _le = 0; _le < 16; _le++) {
                                sv_b[_le] = approx_exp2(sv_b[_le]);
                            }
                            #pragma unroll
                            for (int j_16 = 0; j_16 < 16; j_16++) {
                                psum_b[j_16] = psum_b[j_16] + sv_b[j_16];
#else
                            float2 _reg_reduce_max2_1 = {-CAKE_FMHA_INF, -CAKE_FMHA_INF};
                            _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(sv[0], sv[1]));
                            _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(sv[2], sv[3]));
                            _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(sv[4], sv[5]));
                            _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(sv[6], sv[7]));
                            _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(sv[8], sv[9]));
                            _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(sv[10], sv[11]));
                            _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(sv[12], sv[13]));
                            _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(sv[14], sv[15]));
                            _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(sv[16], sv[17]));
                            _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(sv[18], sv[19]));
                            _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(sv[20], sv[21]));
                            _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(sv[22], sv[23]));
                            _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(sv[24], sv[25]));
                            _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(sv[26], sv[27]));
                            _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(sv[28], sv[29]));
                            _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(sv[30], sv[31]));
                            float sv_max = row_max_reduce(_reg_reduce_max2_1);
                            lmax = sv_max;
                            float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, lmax, 16);
#if CAKE_FMHA_DCP_PROGRAM == 4
                            float _max_6 = max_noftz(lmax, _shfl_xor_0);
                            lmax = _max_6;
#else
                            float _max_4 = max_noftz(lmax, _shfl_xor_0);
                            lmax = _max_4;
#endif  // CAKE_FMHA_DCP_PROGRAM == 4
                            if (half == 0) {
                                smem_xmax[xm_off_s + my_row] = lmax;
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                            float2 _reg_reduce_max2_1 = {-CAKE_FMHA_INF, -CAKE_FMHA_INF};
                            _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(sv[0], sv[1]));
                            _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(sv[2], sv[3]));
                            _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(sv[4], sv[5]));
                            _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(sv[6], sv[7]));
                            _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(sv[8], sv[9]));
                            _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(sv[10], sv[11]));
                            _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(sv[12], sv[13]));
                            _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(sv[14], sv[15]));
                            _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(sv[16], sv[17]));
                            _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(sv[18], sv[19]));
                            _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(sv[20], sv[21]));
                            _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(sv[22], sv[23]));
                            _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(sv[24], sv[25]));
                            _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(sv[26], sv[27]));
                            _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(sv[28], sv[29]));
                            _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(sv[30], sv[31]));
                            float sv_max = row_max_reduce(_reg_reduce_max2_1);
                            lmax = sv_max;
                            float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, lmax, 16);
                            float _max_16 = max_noftz(lmax, _shfl_xor_3);
                            lmax = _max_16;
                            if (half == 0) {
                                smem_xmax[xm_off_s + my_row] = lmax;
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                            }
                        }
                        if (sm_tid == 0) {
                        }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                        if (sm_tid == 0) {
                        }
                        unsigned int k_blk_o = pv_base_s + (unsigned int)n;
                        if (rows_hi != 0) {
                            if (k_blk_o > 0) {
                                unsigned int k_pv1f = k_blk_o - 1;
                                {
                                    mbarrier_wait(o_ready_addr + ((int)(k_pv1f & 1)) * 8, (int)(k_pv1f >> 1 & 1));
                                }
                            }
                        } else if (k_blk_o >= 2) {
                            unsigned int k_pv2f = k_blk_o - 2;
                            {
                                mbarrier_wait(o_ready_addr + ((int)(k_pv2f & 1)) * 8, (int)(k_pv2f >> 1 & 1));
                            }
                        }
                        if (sm_tid == 0) {
                        }
                        if (fl_w != 0) {
                            if (n != 0) {
                                unsigned int k_pv1r = k_blk_o - 1;
                                {
                                    mbarrier_wait(o_ready_addr + ((int)(k_pv1r & 1)) * 8, (int)(k_pv1r >> 1 & 1));
                                }
                                asm volatile("tcgen05.fence::after_thread_sync;");
                                float acc16[16];
                                #pragma unroll
#if CAKE_FMHA_DCP_PROGRAM == 5
                                for (int q_1 = 0; q_1 < 4; q_1++) {
#else
                                for (int q = 0; q < 4; q++) {
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
#if CAKE_FMHA_DCP_PROGRAM == 5
                                        : "=r"(*reinterpret_cast<uint32_t*>(&acc16[4 * q_1])), "=r"(*reinterpret_cast<uint32_t*>(&acc16[(4 * q_1) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&acc16[(4 * q_1) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&acc16[(4 * q_1) + 3]))
                                        : "r"(swap_acc_addr + (unsigned int)(4 * q_1 * 4)));
#else
                                        : "=r"(*reinterpret_cast<uint32_t*>(&acc16[4 * q])), "=r"(*reinterpret_cast<uint32_t*>(&acc16[(4 * q) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&acc16[(4 * q) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&acc16[(4 * q) + 3]))
                                        : "r"(swap_acc_addr + (unsigned int)(4 * q * 4)));
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
                                }
                                float _tmem_load_0[16];
                                tmem_ld_x16(&_tmem_load_0[0], oT_base_w);
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                #pragma unroll
                                for (int j_17 = 0; j_17 < 16; j_17++) {
                                    _tmem_load_0[j_17] = _tmem_load_0[j_17] * acc16[j_17];
                                }
                                tmem_st_x16_f32(oT_base_w, _tmem_load_0);
                                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                                if (rows_hi != 0) {
                                    float acc16_0[16];
                                    #pragma unroll
#if CAKE_FMHA_DCP_PROGRAM == 5
                                    for (int q_2 = 0; q_2 < 4; q_2++) {
#else
                                    for (int q_1 = 0; q_1 < 4; q_1++) {
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
                                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
#if CAKE_FMHA_DCP_PROGRAM == 5
                                            : "=r"(*reinterpret_cast<uint32_t*>(&acc16_0[4 * q_2])), "=r"(*reinterpret_cast<uint32_t*>(&acc16_0[(4 * q_2) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&acc16_0[(4 * q_2) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&acc16_0[(4 * q_2) + 3]))
                                            : "r"(swap_acc_addr + (unsigned int)((32 + 4 * q_2) * 4)));
#else
                                            : "=r"(*reinterpret_cast<uint32_t*>(&acc16_0[4 * q_1])), "=r"(*reinterpret_cast<uint32_t*>(&acc16_0[(4 * q_1) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&acc16_0[(4 * q_1) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&acc16_0[(4 * q_1) + 3]))
                                            : "r"(swap_acc_addr + (unsigned int)((32 + 4 * q_1) * 4)));
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
                                    }
                                    float _tmem_load_1[16];
                                    tmem_ld_x16(&_tmem_load_1[0], oT_base_w + 32);
                                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                                    #pragma unroll
                                    for (int j_18 = 0; j_18 < 16; j_18++) {
                                        _tmem_load_1[j_18] = _tmem_load_1[j_18] * acc16_0[j_18];
                                    }
                                    tmem_st_x16_f32(oT_base_w + 32, _tmem_load_1);
                                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
#else
                        asm volatile("barrier.sync 8, 256;" ::: "memory");
                        if (rows_live != 0) {
#if CAKE_FMHA_DCP_PROGRAM == 4
                            float _max_7 = max_noftz(lmax, smem_xmax[xm_off_s + 64 + my_row]);
                            lmax = _max_7;
#else
                            float _max_5 = max_noftz(lmax, smem_xmax[xm_off_s + 64 + my_row]);
                            lmax = _max_5;
#endif  // CAKE_FMHA_DCP_PROGRAM == 4
                            if (lmax > row_max + thr_raw) {
                                new_max = lmax;
                                if (row_max > -CAKE_FMHA_INF) {
                                    float _exp2_0 = approx_exp2(softmax_scale_log2 * (row_max - new_max));
                                    acc_scale = _exp2_0;
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                        asm volatile("barrier.sync 8, 256;" ::: "memory");
                        if (rows_live != 0) {
                            float _max_17 = max_noftz(lmax, smem_xmax[xm_off_s + 64 + my_row]);
                            lmax = _max_17;
                            if (lmax > row_max + thr_raw) {
                                new_max = lmax;
                                if (row_max > -CAKE_FMHA_INF) {
                                    float _exp2_0 = approx_exp2(softmax_scale_log2 * (row_max - new_max));
                                    acc_scale = _exp2_0;
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                                }
                            }
                        }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                        int pb_w = 0;
                        if (rows_hi == 0) {
                            pb_w = (int)(k_blk_o & 1);
#else
                        xm_slot_s = xm_slot_s ^ 1;
                        if (sm_tid == 0) {
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                        xm_slot_s = xm_slot_s ^ 1;
                        if (sm_tid == 0) {
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                        }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                        if (pb_w == 0) {
                            unsigned int words8[8];
#else
                        if (rows_live != 0) {
                            float safe_max = ((new_max == -CAKE_FMHA_INF) ? 0.0f : new_max);
                            const float2 _fma_b2_2 = {softmax_scale_log2, softmax_scale_log2};
                            const float2 _fma_c2_3 = {(-safe_max) * softmax_scale_log2, (-safe_max) * softmax_scale_log2};
                            float2 _fma_pair_4 = fma_f32x2(make_float2(sv[0], sv[1]), _fma_b2_2, _fma_c2_3);
                            sv[0] = _fma_pair_4.x;
                            sv[1] = _fma_pair_4.y;
                            float2 _fma_pair_5 = fma_f32x2(make_float2(sv[2], sv[3]), _fma_b2_2, _fma_c2_3);
                            sv[2] = _fma_pair_5.x;
                            sv[3] = _fma_pair_5.y;
                            float2 _fma_pair_6 = fma_f32x2(make_float2(sv[4], sv[5]), _fma_b2_2, _fma_c2_3);
                            sv[4] = _fma_pair_6.x;
                            sv[5] = _fma_pair_6.y;
                            float2 _fma_pair_7 = fma_f32x2(make_float2(sv[6], sv[7]), _fma_b2_2, _fma_c2_3);
                            sv[6] = _fma_pair_7.x;
                            sv[7] = _fma_pair_7.y;
                            float2 _fma_pair_8 = fma_f32x2(make_float2(sv[8], sv[9]), _fma_b2_2, _fma_c2_3);
                            sv[8] = _fma_pair_8.x;
                            sv[9] = _fma_pair_8.y;
                            float2 _fma_pair_9 = fma_f32x2(make_float2(sv[10], sv[11]), _fma_b2_2, _fma_c2_3);
                            sv[10] = _fma_pair_9.x;
                            sv[11] = _fma_pair_9.y;
                            float2 _fma_pair_10 = fma_f32x2(make_float2(sv[12], sv[13]), _fma_b2_2, _fma_c2_3);
                            sv[12] = _fma_pair_10.x;
                            sv[13] = _fma_pair_10.y;
                            float2 _fma_pair_11 = fma_f32x2(make_float2(sv[14], sv[15]), _fma_b2_2, _fma_c2_3);
                            sv[14] = _fma_pair_11.x;
                            sv[15] = _fma_pair_11.y;
                            float2 _fma_pair_12 = fma_f32x2(make_float2(sv[16], sv[17]), _fma_b2_2, _fma_c2_3);
                            sv[16] = _fma_pair_12.x;
                            sv[17] = _fma_pair_12.y;
                            float2 _fma_pair_13 = fma_f32x2(make_float2(sv[18], sv[19]), _fma_b2_2, _fma_c2_3);
                            sv[18] = _fma_pair_13.x;
                            sv[19] = _fma_pair_13.y;
                            float2 _fma_pair_14 = fma_f32x2(make_float2(sv[20], sv[21]), _fma_b2_2, _fma_c2_3);
                            sv[20] = _fma_pair_14.x;
                            sv[21] = _fma_pair_14.y;
                            float2 _fma_pair_15 = fma_f32x2(make_float2(sv[22], sv[23]), _fma_b2_2, _fma_c2_3);
                            sv[22] = _fma_pair_15.x;
                            sv[23] = _fma_pair_15.y;
                            float2 _fma_pair_16 = fma_f32x2(make_float2(sv[24], sv[25]), _fma_b2_2, _fma_c2_3);
                            sv[24] = _fma_pair_16.x;
                            sv[25] = _fma_pair_16.y;
                            float2 _fma_pair_17 = fma_f32x2(make_float2(sv[26], sv[27]), _fma_b2_2, _fma_c2_3);
                            sv[26] = _fma_pair_17.x;
                            sv[27] = _fma_pair_17.y;
                            float2 _fma_pair_18 = fma_f32x2(make_float2(sv[28], sv[29]), _fma_b2_2, _fma_c2_3);
                            sv[28] = _fma_pair_18.x;
                            sv[29] = _fma_pair_18.y;
                            float2 _fma_pair_19 = fma_f32x2(make_float2(sv[30], sv[31]), _fma_b2_2, _fma_c2_3);
                            sv[30] = _fma_pair_19.x;
                            sv[31] = _fma_pair_19.y;
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                        if (rows_live != 0) {
                            float safe_max = ((new_max == -CAKE_FMHA_INF) ? 0.0f : new_max);
                            const float2 _fma_b2_2 = {softmax_scale_log2, softmax_scale_log2};
                            const float2 _fma_c2_3 = {(-safe_max) * softmax_scale_log2, (-safe_max) * softmax_scale_log2};
                            float2 _fma_pair_4 = fma_f32x2(make_float2(sv[0], sv[1]), _fma_b2_2, _fma_c2_3);
                            sv[0] = _fma_pair_4.x;
                            sv[1] = _fma_pair_4.y;
                            float2 _fma_pair_5 = fma_f32x2(make_float2(sv[2], sv[3]), _fma_b2_2, _fma_c2_3);
                            sv[2] = _fma_pair_5.x;
                            sv[3] = _fma_pair_5.y;
                            float2 _fma_pair_6 = fma_f32x2(make_float2(sv[4], sv[5]), _fma_b2_2, _fma_c2_3);
                            sv[4] = _fma_pair_6.x;
                            sv[5] = _fma_pair_6.y;
                            float2 _fma_pair_7 = fma_f32x2(make_float2(sv[6], sv[7]), _fma_b2_2, _fma_c2_3);
                            sv[6] = _fma_pair_7.x;
                            sv[7] = _fma_pair_7.y;
                            float2 _fma_pair_8 = fma_f32x2(make_float2(sv[8], sv[9]), _fma_b2_2, _fma_c2_3);
                            sv[8] = _fma_pair_8.x;
                            sv[9] = _fma_pair_8.y;
                            float2 _fma_pair_9 = fma_f32x2(make_float2(sv[10], sv[11]), _fma_b2_2, _fma_c2_3);
                            sv[10] = _fma_pair_9.x;
                            sv[11] = _fma_pair_9.y;
                            float2 _fma_pair_10 = fma_f32x2(make_float2(sv[12], sv[13]), _fma_b2_2, _fma_c2_3);
                            sv[12] = _fma_pair_10.x;
                            sv[13] = _fma_pair_10.y;
                            float2 _fma_pair_11 = fma_f32x2(make_float2(sv[14], sv[15]), _fma_b2_2, _fma_c2_3);
                            sv[14] = _fma_pair_11.x;
                            sv[15] = _fma_pair_11.y;
                            float2 _fma_pair_12 = fma_f32x2(make_float2(sv[16], sv[17]), _fma_b2_2, _fma_c2_3);
                            sv[16] = _fma_pair_12.x;
                            sv[17] = _fma_pair_12.y;
                            float2 _fma_pair_13 = fma_f32x2(make_float2(sv[18], sv[19]), _fma_b2_2, _fma_c2_3);
                            sv[18] = _fma_pair_13.x;
                            sv[19] = _fma_pair_13.y;
                            float2 _fma_pair_14 = fma_f32x2(make_float2(sv[20], sv[21]), _fma_b2_2, _fma_c2_3);
                            sv[20] = _fma_pair_14.x;
                            sv[21] = _fma_pair_14.y;
                            float2 _fma_pair_15 = fma_f32x2(make_float2(sv[22], sv[23]), _fma_b2_2, _fma_c2_3);
                            sv[22] = _fma_pair_15.x;
                            sv[23] = _fma_pair_15.y;
                            float2 _fma_pair_16 = fma_f32x2(make_float2(sv[24], sv[25]), _fma_b2_2, _fma_c2_3);
                            sv[24] = _fma_pair_16.x;
                            sv[25] = _fma_pair_16.y;
                            float2 _fma_pair_17 = fma_f32x2(make_float2(sv[26], sv[27]), _fma_b2_2, _fma_c2_3);
                            sv[26] = _fma_pair_17.x;
                            sv[27] = _fma_pair_17.y;
                            float2 _fma_pair_18 = fma_f32x2(make_float2(sv[28], sv[29]), _fma_b2_2, _fma_c2_3);
                            sv[28] = _fma_pair_18.x;
                            sv[29] = _fma_pair_18.y;
                            float2 _fma_pair_19 = fma_f32x2(make_float2(sv[30], sv[31]), _fma_b2_2, _fma_c2_3);
                            sv[30] = _fma_pair_19.x;
                            sv[31] = _fma_pair_19.y;
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                            #pragma unroll
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sv_a[_lp*2 + 0], sv_a[_lp*2+1 + 0]));
                                words8[_lp] = *(uint32_t*)&_bf2;
#else
                            for (int _le = 0; _le < 32; _le++) {
                                sv[_le] = approx_exp2(sv[_le]);
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                            for (int _le = 0; _le < 32; _le++) {
                                sv[_le] = approx_exp2(sv[_le]);
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                            }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                            int row_b = k_w * 64;
                            int swz = (k_w >> 1 & 3) << 4;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(swap_pT_addr + (unsigned int)(row_b + (swz ^ 0))), "r"(words8[0]), "r"(words8[1]), "r"(words8[2]), "r"(words8[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(swap_pT_addr + (unsigned int)(row_b + (swz ^ 16))), "r"(words8[4]), "r"(words8[5]), "r"(words8[6]), "r"(words8[7]) : "memory");
                        } else {
                            unsigned int words8_1[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sv_a[_lp*2 + 0], sv_a[_lp*2+1 + 0]));
                                words8_1[_lp] = *(uint32_t*)&_bf2;
                            }
                            int row_b_1 = k_w * 64;
                            int swz_1 = (k_w >> 1 & 3) << 4;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(swap_pT1_addr + (unsigned int)(row_b_1 + (swz_1 ^ 0))), "r"(words8_1[0]), "r"(words8_1[1]), "r"(words8_1[2]), "r"(words8_1[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(swap_pT1_addr + (unsigned int)(row_b_1 + (swz_1 ^ 16))), "r"(words8_1[4]), "r"(words8_1[5]), "r"(words8_1[6]), "r"(words8_1[7]) : "memory");
#else
                            float2 _reg_reduce_sum2_20 = make_float2(0.0f, 0.0f);
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[0], sv[1]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[2], sv[3]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[4], sv[5]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[6], sv[7]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[8], sv[9]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[10], sv[11]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[12], sv[13]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[14], sv[15]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[16], sv[17]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[18], sv[19]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[20], sv[21]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[22], sv[23]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[24], sv[25]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[26], sv[27]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[28], sv[29]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[30], sv[31]));
                            float sv_sum = _reg_reduce_sum2_20.x + _reg_reduce_sum2_20.y;
                            lsum = sv_sum;
                            psum = psum * acc_scale + lsum;
                            row_max = new_max;
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                            float2 _reg_reduce_sum2_20 = make_float2(0.0f, 0.0f);
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[0], sv[1]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[2], sv[3]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[4], sv[5]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[6], sv[7]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[8], sv[9]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[10], sv[11]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[12], sv[13]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[14], sv[15]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[16], sv[17]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[18], sv[19]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[20], sv[21]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[22], sv[23]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[24], sv[25]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[26], sv[27]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[28], sv[29]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv[30], sv[31]));
                            float sv_sum = _reg_reduce_sum2_20.x + _reg_reduce_sum2_20.y;
                            lsum = sv_sum;
                            psum = psum * acc_scale + lsum;
                            row_max = new_max;
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                        }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#else
                        if (sm_tid == 0) {
                        }
                        if (rows_live != 0) {
                            unsigned int regs_p[16];
                            #pragma unroll
                            for (int _lp = 0; _lp < 16; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sv[_lp*2 + 0], sv[_lp*2+1 + 0]));
                                regs_p[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x32bx2.x16.b32"
                                " [%0], 32, {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                                :: "r"((unsigned int)my_s_base + sm_stage * (unsigned int)BLOCK_N), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[0])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[1])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[2])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[3])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[4])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[5])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[6])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[7])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[8])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[9])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[10])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[11])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[12])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[13])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[14])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[15])));
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        }
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                        if (sm_tid == 0) {
                        }
                        if (rows_live != 0) {
                            unsigned int regs_p[16];
                            #pragma unroll
                            for (int _lp = 0; _lp < 16; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sv[_lp*2 + 0], sv[_lp*2+1 + 0]));
                                regs_p[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x32bx2.x16.b32"
                                " [%0], 32, {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                                :: "r"((unsigned int)my_s_base + sm_stage * (unsigned int)BLOCK_N), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[0])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[1])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[2])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[3])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[4])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[5])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[6])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[7])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[8])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[9])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[10])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[11])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[12])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[13])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[14])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[15])));
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        }
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                        __syncwarp();
                        if (lane == 0) {
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;"
                                :: "r"(p_full_addr + (sm_stage) * 8), "r"((uint32_t)(32)) : "memory");
                        }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                        if (sm_tid == 0) {
                        }
                        if (rows_hi != 0) {
                            unsigned int k_pa = pv_base_s + (unsigned int)n;
                            mbarrier_wait(pa_done_addr + ((int)(k_pa & 1)) * 8, (int)(k_pa >> 1 & 1));
                            unsigned int words8_2[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sv_b[_lp*2 + 0], sv_b[_lp*2+1 + 0]));
                                words8_2[_lp] = *(uint32_t*)&_bf2;
                            }
                            int row_b_2 = k_w * 64;
                            int swz_2 = (k_w >> 1 & 3) << 4;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(swap_pT_addr + (unsigned int)(row_b_2 + (swz_2 ^ 0))), "r"(words8_2[0]), "r"(words8_2[1]), "r"(words8_2[2]), "r"(words8_2[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(swap_pT_addr + (unsigned int)(row_b_2 + (swz_2 ^ 16))), "r"(words8_2[4]), "r"(words8_2[5]), "r"(words8_2[6]), "r"(words8_2[7]) : "memory");
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            __syncwarp();
                            if (lane == 0) {
                                asm volatile(
                                    "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;"
                                    :: "r"(p_full_b_addr + (sm_stage) * 8), "r"((uint32_t)(32)) : "memory");
                            }
                        }
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                        sm_stage += 1;
                        if (sm_stage == 2) { sm_stage = 0; sm_phase ^= 1; }
                        if (sm_tid == 0) {
                        }
                    }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                    pv_base_s = pv_base_s + (unsigned int)cnt_s;
                    if (sm_tid == 0) {
                    }
#else
                    float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, psum, 16);
                    float total = psum + _shfl_xor_1;
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                    float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, psum, 16);
                    float total = psum + _shfl_xor_4;
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                    if (sm_tid == 0) {
                    }
                    mbarrier_wait(stats_empty_addr + (st_stage_s) * 8, st_phase_s);
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                    if (sm_tid == 0) {
                    }
                    int rows_hi_p = ((N_ROWS > 32) ? 1 : 0);
                    int b4_1 = lane >> 4 & 1;
                    int b3_2 = lane >> 3 & 1;
                    int b2_1 = lane >> 2 & 1;
                    int b1_1 = lane >> 1 & 1;
                    float w8_1[8];
                    #pragma unroll
                    for (int i_6 = 0; i_6 < 8; i_6++) {
                        float send8_2 = ((b4_1 != 0) ? psum_a[i_6] : psum_a[i_6 + 8]);
                        float keep8_2 = ((b4_1 != 0) ? psum_a[i_6 + 8] : psum_a[i_6]);
                        float _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, send8_2, 16);
                        w8_1[i_6] = keep8_2 + _shfl_xor_10;
                    }
                    float x4_1[4];
                    #pragma unroll
                    for (int i_7 = 0; i_7 < 4; i_7++) {
                        float send4_2 = ((b3_2 != 0) ? w8_1[i_7] : w8_1[i_7 + 4]);
                        float keep4_2 = ((b3_2 != 0) ? w8_1[i_7 + 4] : w8_1[i_7]);
                        float _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, send4_2, 8);
                        x4_1[i_7] = keep4_2 + _shfl_xor_11;
                    }
                    float y2_1[2];
                    #pragma unroll
                    for (int i_8 = 0; i_8 < 2; i_8++) {
                        float send2_2 = ((b2_1 != 0) ? x4_1[i_8] : x4_1[i_8 + 2]);
                        float keep2_2 = ((b2_1 != 0) ? x4_1[i_8 + 2] : x4_1[i_8]);
                        float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, send2_2, 4);
                        y2_1[i_8] = keep2_2 + _shfl_xor_12;
                    }
                    float send1_1 = ((b1_1 != 0) ? y2_1[0] : y2_1[1]);
                    float keep1_1 = ((b1_1 != 0) ? y2_1[1] : y2_1[0]);
                    float _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, send1_1, 2);
                    float z_1 = keep1_1 + _shfl_xor_13;
                    float _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, z_1, 1);
                    z_1 = z_1 + _shfl_xor_14;
                    float zs_a = z_1;
                    if ((lane & 1) == 0) {
                        swap_part[(lane >> 1 & 15) * 4 + s_warp] = zs_a;
                    }
                    if (rows_hi_p != 0) {
                        int b4_0_1 = lane >> 4 & 1;
                        int b3_1_1 = lane >> 3 & 1;
                        int b2_2_1 = lane >> 2 & 1;
                        int b1_3_1 = lane >> 1 & 1;
                        float w8_4_1[8];
                        #pragma unroll
                        for (int i_9 = 0; i_9 < 8; i_9++) {
                            float send8_3 = ((b4_0_1 != 0) ? psum_b[i_9] : psum_b[i_9 + 8]);
                            float keep8_3 = ((b4_0_1 != 0) ? psum_b[i_9 + 8] : psum_b[i_9]);
                            float _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, send8_3, 16);
                            w8_4_1[i_9] = keep8_3 + _shfl_xor_15;
                        }
                        float x4_5_1[4];
                        #pragma unroll
                        for (int i_10 = 0; i_10 < 4; i_10++) {
                            float send4_3 = ((b3_1_1 != 0) ? w8_4_1[i_10] : w8_4_1[i_10 + 4]);
                            float keep4_3 = ((b3_1_1 != 0) ? w8_4_1[i_10 + 4] : w8_4_1[i_10]);
                            float _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, send4_3, 8);
                            x4_5_1[i_10] = keep4_3 + _shfl_xor_16;
                        }
                        float y2_6_1[2];
                        #pragma unroll
                        for (int i_11 = 0; i_11 < 2; i_11++) {
                            float send2_3 = ((b2_2_1 != 0) ? x4_5_1[i_11] : x4_5_1[i_11 + 2]);
                            float keep2_3 = ((b2_2_1 != 0) ? x4_5_1[i_11 + 2] : x4_5_1[i_11]);
                            float _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, send2_3, 4);
                            y2_6_1[i_11] = keep2_3 + _shfl_xor_17;
                        }
                        float send1_7_1 = ((b1_3_1 != 0) ? y2_6_1[0] : y2_6_1[1]);
                        float keep1_8_1 = ((b1_3_1 != 0) ? y2_6_1[1] : y2_6_1[0]);
                        float _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, send1_7_1, 2);
                        float z_9_1 = keep1_8_1 + _shfl_xor_18;
                        float _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, z_9_1, 1);
                        z_9_1 = z_9_1 + _shfl_xor_19;
                        float zs_b = z_9_1;
                        if ((lane & 1) == 0) {
                            swap_part[(16 + (lane >> 1 & 15)) * 4 + s_warp] = zs_b;
                        }
                    }
                    if (s_warp == 0) {
                        if (lane == 0) {
                        }
                    }
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                    if (s_warp == 0) {
                        if (lane == 0) {
                        }
                        if (lane < 16) {
                            float q4[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q4[0])), "=r"(*reinterpret_cast<uint32_t*>(&q4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q4[(0) + 3]))
                                : "r"(swap_part_addr + (unsigned int)(lane * 16)));
                            float sum_pa = q4[0] + q4[1] + (q4[2] + q4[3]);
                            smem_sum[(int)st_stage_s * 64 + lane] = sum_pa;
#if CAKE_FMHA_DCP_PROGRAM == 5
                            float _rcp_4 = approx_rcp(sum_pa);
                            swap_inv[(int)st_stage_s * 64 + lane] = ((sum_pa > 0.0f) ? _rcp_4 : 0.0f);
#else
                            float _rcp_3 = approx_rcp(sum_pa);
                            swap_inv[(int)st_stage_s * 64 + lane] = ((sum_pa > 0.0f) ? _rcp_3 : 0.0f);
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
                            if (rows_hi_p != 0) {
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&q4[0])), "=r"(*reinterpret_cast<uint32_t*>(&q4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q4[(0) + 3]))
                                    : "r"(swap_part_addr + (unsigned int)((16 + lane) * 16)));
                                float sum_pb = q4[0] + q4[1] + (q4[2] + q4[3]);
                                smem_sum[(int)st_stage_s * 64 + 32 + lane] = sum_pb;
#if CAKE_FMHA_DCP_PROGRAM == 5
                                float _rcp_5 = approx_rcp(sum_pb);
                                swap_inv[(int)st_stage_s * 64 + 32 + lane] = ((sum_pb > 0.0f) ? _rcp_5 : 0.0f);
#else
                                float _rcp_4 = approx_rcp(sum_pb);
                                swap_inv[(int)st_stage_s * 64 + 32 + lane] = ((sum_pb > 0.0f) ? _rcp_4 : 0.0f);
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
                            }
                        }
                        if (lane == 0) {
                        }
                        float my_ref_a = ref_a[0];
                        float my_ref_b = ref_b[0];
                        #pragma unroll
                        for (int j_19 = 1; j_19 < 16; j_19++) {
                            my_ref_a = ((lane == j_19) ? ref_a[j_19] : my_ref_a);
                            if (rows_hi_p != 0) {
                                my_ref_b = ((lane == j_19) ? ref_b[j_19] : my_ref_b);
                            }
                        }
                        if (lane < 16) {
                            smem_max[(int)st_stage_s * 64 + lane] = my_ref_a;
                            if (rows_hi_p != 0) {
                                smem_max[(int)st_stage_s * 64 + 32 + lane] = my_ref_b;
                            }
                        }
                        if (lane == 0) {
#else
                    if (rows_live != 0) {
                        if (half == 0) {
                            smem_sum[st_stage_s * 64 + (unsigned int)my_row] = total;
                            smem_max[st_stage_s * 64 + (unsigned int)my_row] = row_max;
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                    if (rows_live != 0) {
                        if (half == 0) {
                            smem_sum[st_stage_s * 64 + (unsigned int)my_row] = total;
                            smem_max[st_stage_s * 64 + (unsigned int)my_row] = row_max;
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                        }
                    }
                    __syncwarp();
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                    if (sm_tid == 0) {
                    }
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                    if (lane == 0) {
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;"
                            :: "r"(stats_full_addr + (st_stage_s) * 8), "r"((uint32_t)(32)) : "memory");
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                    }
                    if (sm_tid == 0) {
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                    }
                    st_stage_s += 1;
                    if (st_stage_s == 2) { st_stage_s = 0; st_phase_s ^= 1; }
                    if (sm_tid == 0) {
                    }
                } else {
                    {
                        asm volatile("fence.acquire.gpu;" ::: "memory");
                    }
                    int r_row = (128 + sm_tid) / 4;
                    if (r_row < live_rows_s) {
                        int stats_stride_r = num_kv_heads * 128;
                        int o_stride_r = num_kv_heads * 8192;
                        int stats_row_r = slot_tile_base_s * 128 + r_row;
                        int f_r = 32 >> block_end_s;
                        int d0_r = block_begin_s * (4 * f_r) + (128 + sm_tid) % 4 * f_r;
                        int o_row_r = slot_tile_base_s * 8192 + r_row * 128 + d0_r;
                        int j_r = r_row / 8;
                        int h_r = r_row % 8;
                        int q_head_r = kv_head_s * 8 + h_r;
                        int o_idx_r = ((batch_s * q_len + j_r) * num_q_heads + q_head_r) * HEAD_DIM + d0_r;
                        int store_lse_r = 0;
                        if ((128 + sm_tid) % 4 == 0 && block_begin_s == 0) {
                            store_lse_r = 1;
                        }
                        int lse_idx_r = (batch_s * q_len + j_r) * num_q_heads + q_head_r;
                        {
                            int n_groups_r = 8 >> block_end_s;
                            float acc_f[4];
                            float out4[4];
                            int n_pad_r = (n_chunks_s + 7) / 8 * 8;
                            #pragma unroll 1
                            for (int g_r = 0; g_r < n_groups_r; g_r++) {
                                acc_f[0] = 0.0f;
                                acc_f[1] = 0.0f;
                                acc_f[2] = 0.0f;
                                acc_f[3] = 0.0f;
                                float m_f = -1e+30f;
                                float l_f = 0.0f;
                                int o_col_r = o_row_r + g_r * 4;
                                #pragma unroll 8
                                for (int c_m = 0; c_m < n_pad_r; c_m++) {
                                    int c_c = c_m;
                                    if (n_chunks_s <= c_m) {
                                        c_c = n_chunks_s - 1;
                                    }
                                    float m_k = partial_stats[stats_row_r + c_c * stats_stride_r];
                                    float l_k = partial_stats[stats_row_r + 64 + c_c * stats_stride_r];
                                    if (n_chunks_s <= c_m) {
                                        m_k = -1e+30f;
                                        l_k = 0.0f;
                                    }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 5
                                    float _max_31 = max_noftz(m_f, m_k);
                                    float m_new = _max_31;
#else
                                    float _max_29 = max_noftz(m_f, m_k);
                                    float m_new = _max_29;
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
                                    float _exp2_6 = approx_exp2((m_f - m_new) * softmax_scale_log2);
                                    float a_k = _exp2_6;
                                    float _exp2_7 = approx_exp2((m_k - m_new) * softmax_scale_log2);
                                    float b_k = _exp2_7;
                                    float _fma_6 = __fmaf_rn(l_k, b_k, l_f * a_k);
                                    l_f = _fma_6;
#else
#if CAKE_FMHA_DCP_PROGRAM == 4
                                    float _max_12 = max_noftz(m_f, m_k);
                                    float m_new = _max_12;
#else
                                    float _max_10 = max_noftz(m_f, m_k);
                                    float m_new = _max_10;
#endif  // CAKE_FMHA_DCP_PROGRAM == 4
                                    float _exp2_5 = approx_exp2((m_f - m_new) * softmax_scale_log2);
                                    float a_k = _exp2_5;
                                    float _exp2_6 = approx_exp2((m_k - m_new) * softmax_scale_log2);
                                    float b_k = _exp2_6;
                                    float _fma_4 = __fmaf_rn(l_k, b_k, l_f * a_k);
                                    l_f = _fma_4;
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                                    float _max_22 = max_noftz(m_f, m_k);
                                    float m_new = _max_22;
                                    float _exp2_5 = approx_exp2((m_f - m_new) * softmax_scale_log2);
                                    float a_k = _exp2_5;
                                    float _exp2_6 = approx_exp2((m_k - m_new) * softmax_scale_log2);
                                    float b_k = _exp2_6;
                                    float _fma_4 = __fmaf_rn(l_k, b_k, l_f * a_k);
                                    l_f = _fma_4;
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                                    float _vec_load_0[4];
                                    {
                                        float4 _v4 = *reinterpret_cast<const float4*>(partial_o + (o_col_r + c_c * o_stride_r) + 0);
                                        _vec_load_0[0 + 0] = _v4.x;
                                        _vec_load_0[0 + 1] = _v4.y;
                                        _vec_load_0[0 + 2] = _v4.z;
                                        _vec_load_0[0 + 3] = _v4.w;
                                    }
                                    #pragma unroll
                                    for (int k = 0; k < 4; k++) {
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                                        float _fma_7 = __fmaf_rn(_vec_load_0[k], b_k, acc_f[k] * a_k);
                                        acc_f[k] = _fma_7;
#else
                                        float _fma_5 = __fmaf_rn(_vec_load_0[k], b_k, acc_f[k] * a_k);
                                        acc_f[k] = _fma_5;
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                                        float _fma_5 = __fmaf_rn(_vec_load_0[k], b_k, acc_f[k] * a_k);
                                        acc_f[k] = _fma_5;
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                                    }
                                    m_f = m_new;
                                }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 5
                                float _rcp_8 = approx_rcp(l_f);
                                float inv_f = ((l_f > 0.0f) ? _rcp_8 : 0.0f);
#else
                                float _rcp_7 = approx_rcp(l_f);
                                float inv_f = ((l_f > 0.0f) ? _rcp_7 : 0.0f);
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
#else
#if CAKE_FMHA_DCP_PROGRAM == 4
                                float _rcp_6 = approx_rcp(l_f);
                                float inv_f = ((l_f > 0.0f) ? _rcp_6 : 0.0f);
#else
                                float _rcp_5 = approx_rcp(l_f);
                                float inv_f = ((l_f > 0.0f) ? _rcp_5 : 0.0f);
#endif  // CAKE_FMHA_DCP_PROGRAM == 4
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                                float _rcp_17 = approx_rcp(l_f);
                                float inv_f = ((l_f > 0.0f) ? _rcp_17 : 0.0f);
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                                #pragma unroll
                                for (int k4 = 0; k4 < 4; k4++) {
                                    out4[k4] = acc_f[k4] * inv_f;
                                }
                                {
                                    uint2 _pk2;
                                    __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
                                    _pk[0] = __floats2bfloat162_rn(out4[0 + 0], out4[0 + 1]);
                                    _pk[1] = __floats2bfloat162_rn(out4[0 + 2], out4[0 + 3]);
                                    *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(O_ptr + (o_idx_r + g_r * 4)))[0]) = _pk2;
                                }
                                if (store_lse_r != 0) {
                                    if (g_r == 0) {
                                        float lse_r = -CAKE_FMHA_INF;
                                        if (l_f > 0.0f) {
                                            float _log2_2;
                                            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_2) : "f"(l_f));
                                            lse_r = m_f * softmax_scale_log2 + _log2_2;
                                        }
                                        *(reinterpret_cast<float*>(LSE_ptr + lse_idx_r) + (0)) = lse_r;
                                    }
                                }
                            }
                        }
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_s) * 8, _phase_work_full);
                unsigned int base_0 = work_stage_s * 16;
                unsigned int valid_1 = work_token_words[base_0];
                unsigned int kind_2 = work_token_words[base_0 + 1];
                unsigned int batch_3 = work_token_words[base_0 + 2];
                unsigned int kv_head_4 = work_token_words[base_0 + 3];
                unsigned int block_begin_5 = work_token_words[base_0 + 4];
                unsigned int block_end_6 = work_token_words[base_0 + 5];
                unsigned int seqlen_7 = work_token_words[base_0 + 6];
                unsigned int n_chunks_8 = work_token_words[base_0 + 7];
                unsigned int slot_tile_base_9 = work_token_words[base_0 + 8];
                unsigned int counter_idx_10 = work_token_words[base_0 + 9];
                unsigned int chunk_11 = work_token_words[base_0 + 10];
                unsigned int phase_12 = work_token_words[base_0 + 11];
                mbarrier_arrive(work_empty_addr + (work_stage_s) * 8);
                work_stage_s += 1;
                if (work_stage_s == 4) { work_stage_s = 0; _phase_work_full ^= 1; }
                valid_s = valid_1;
                kind_s = (int)kind_2;
                block_begin_s = (int)block_begin_5;
                block_end_s = (int)block_end_6;
                seqlen_s = (int)seqlen_7;
                batch_s = (int)batch_3;
                kv_head_s = (int)kv_head_4;
                n_chunks_s = (int)n_chunks_8;
                slot_tile_base_s = (int)slot_tile_base_9;
                phase_s = (int)phase_12;
            }
            if (sm_tid == 0) {
            }
        }
    // ---- Role: correction ----
    } else if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 168;");
        { // correction_main
            const int warp_in_wg_c = warp % 4;
            const int corr_row = warp_in_wg_c * 32 << 16;
            int wg_tid_c = warp_in_wg_c * 32 + lane;
            int my_row_c = warp_in_wg_c * 16 + lane % 16;
            int half_c = lane / 16;
            int o_row_base = taddr + 256 + (unsigned int)corr_row;
            int my_s_base_c = taddr + (unsigned int)corr_row;
            int tok_base_c = half_c * 64 + 32;
            int rows_live_c = ((warp_in_wg_c * 16 < N_ROWS) ? 1 : 0);
            int row_j_c = my_row_c / 8;
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 5
            int _min_8 = ((row_j_c) < (q_len - 1) ? (row_j_c) : (q_len - 1));
            int vis_j_c = _min_8;
            float _rcp_9 = approx_rcp(softmax_scale_log2);
            float thr_raw_c = 8.0f * _rcp_9;
#else
            int _min_7 = ((row_j_c) < (q_len - 1) ? (row_j_c) : (q_len - 1));
            int vis_j_c = _min_7;
            float _rcp_8 = approx_rcp(softmax_scale_log2);
            float thr_raw_c = 8.0f * _rcp_8;
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
#else
#if CAKE_FMHA_DCP_PROGRAM == 4
            int _min_6 = ((row_j_c) < (q_len - 1) ? (row_j_c) : (q_len - 1));
            int vis_j_c = _min_6;
            float _rcp_7 = approx_rcp(softmax_scale_log2);
            float thr_raw_c = 8.0f * _rcp_7;
#else
            int _min_5 = ((row_j_c) < (q_len - 1) ? (row_j_c) : (q_len - 1));
            int vis_j_c = _min_5;
            float _rcp_6 = approx_rcp(softmax_scale_log2);
            float thr_raw_c = 8.0f * _rcp_6;
#endif  // CAKE_FMHA_DCP_PROGRAM == 4
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
            int _min_6 = ((row_j_c) < (q_len - 1) ? (row_j_c) : (q_len - 1));
            int vis_j_c = _min_6;
            float _rcp_18 = approx_rcp(softmax_scale_log2);
            float thr_raw_c = 8.0f * _rcp_18;
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
            int live_rows = q_len * 8;
            unsigned int sm_stage_c = 0;
            unsigned int sm_phase_c = 0;
            unsigned int xm_slot_c = 0;
            unsigned int st_stage_c = 0;
            unsigned int st_phase_c = 0;
            unsigned int pv_base = 0;
            int n_reduces_seen = 0;
            {
                int spec_tiles_c = batch_size * num_kv_heads;
                int spec_id_c = blockIdx.x;
                if (warp_in_wg_c == 0) {
                    if (spec_id_c < spec_tiles_c) {
                        int spec_batch_c = spec_id_c / num_kv_heads;
                        int spec_head_c = spec_id_c - spec_batch_c * num_kv_heads;
                        int spec_last_c = causal_seqlens_kv_global[spec_batch_c] + (q_len - 1) - cp_rank;
                        int spec_seqlen_c = 0;
                        if (spec_last_c >= 0) {
                            spec_seqlen_c = (spec_last_c >> cp_world_log2) + 1;
                        }
                        int spec_blocks_c = (spec_seqlen_c + BLOCK_N - 1) / BLOCK_N;
                        int spec_max_pg_c = (spec_seqlen_c + PAGE_SIZE - 1) / PAGE_SIZE - 1;
                        int spec_nb_c = lane >> 4;
                        int spec_pg_c = lane >> 1 & 7;
                        int spec_hg_c = lane & 1;
                        int spec_block_c = spec_blocks_c - 1 - spec_nb_c;
                        if (spec_block_c >= 0) {
                            int spec_page_idx_c = spec_block_c * 8 + spec_pg_c;
                            if (spec_page_idx_c > spec_max_pg_c) {
                                spec_page_idx_c = spec_max_pg_c;
                            }
                            int spec_page_c = page_table[spec_batch_c * max_pages_per_seq + spec_page_idx_c];
                            asm volatile("cp.async.bulk.prefetch.tensor.5d.L2.global.tile [%0, {%1, %2, %3, %4, %5}];" :: "l"((uint64_t)((&K))), "r"((int)(0)), "r"((int)(0)), "r"((int)(spec_hg_c)), "r"((int)(spec_head_c)), "r"((int)(spec_page_c)) : "memory");
                            if (spec_nb_c == 0) {
                                asm volatile("cp.async.bulk.prefetch.tensor.5d.L2.global.tile [%0, {%1, %2, %3, %4, %5}];" :: "l"((uint64_t)((&V))), "r"((int)(0)), "r"((int)(0)), "r"((int)(spec_hg_c)), "r"((int)(spec_head_c)), "r"((int)(spec_page_c)) : "memory");
                            }
                        }
                        if (lane == 0) {
                            asm volatile("cp.async.bulk.prefetch.tensor.4d.L2.global.tile [%0, {%1, %2, %3, %4}];" :: "l"((uint64_t)((&Q))), "r"((int)(0)), "r"((int)(spec_head_c * 8)), "r"((int)(spec_batch_c * q_len)), "r"((int)(0)) : "memory");
                        }
                    }
                }
            }
            unsigned int work_stage_c = 0;
            unsigned int _phase_work_full_1 = 0;
            mbarrier_wait(work_full_addr + (work_stage_c) * 8, _phase_work_full_1);
            unsigned int base_1 = work_stage_c * 16;
            unsigned int valid_2 = work_token_words[base_1];
            unsigned int kind_1 = work_token_words[base_1 + 1];
            unsigned int batch_1 = work_token_words[base_1 + 2];
            unsigned int kv_head_1 = work_token_words[base_1 + 3];
            unsigned int block_begin_1 = work_token_words[base_1 + 4];
            unsigned int block_end_1 = work_token_words[base_1 + 5];
            unsigned int seqlen_1 = work_token_words[base_1 + 6];
            unsigned int n_chunks_1 = work_token_words[base_1 + 7];
            unsigned int slot_tile_base_1 = work_token_words[base_1 + 8];
            unsigned int counter_idx_1 = work_token_words[base_1 + 9];
            unsigned int chunk_1 = work_token_words[base_1 + 10];
            unsigned int phase_1 = work_token_words[base_1 + 11];
            mbarrier_arrive(work_empty_addr + (work_stage_c) * 8);
            work_stage_c += 1;
            if (work_stage_c == 4) { work_stage_c = 0; _phase_work_full_1 ^= 1; }
            unsigned int valid_c = valid_2;
            int kind_c = (int)kind_1;
            int batch_c = (int)batch_1;
            int kv_head_c = (int)kv_head_1;
            int block_begin_c = (int)block_begin_1;
            int block_end_c = (int)block_end_1;
            int seqlen_c = (int)seqlen_1;
            int n_chunks_c = (int)n_chunks_1;
            int slot_tile_base_c = (int)slot_tile_base_1;
            int counter_idx_c = (int)counter_idx_1;
            int chunk_c = (int)chunk_1;
            int phase_c = (int)phase_1;
            #pragma unroll 1
            for (unsigned int _tile_iter_c = 0; _tile_iter_c < max_items; _tile_iter_c++) {
                if (valid_c == 0) {
                    break;
                }
                if (kind_c == 0) {
                    int cnt_c = block_end_c - block_begin_c;
                    int my_slot = slot_tile_base_c + chunk_c * num_kv_heads;
                    int back_c = q_len - 1 - vis_j_c - phase_c;
                    int vis_col_c = seqlen_c;
                    if (back_c > 0) {
                        vis_col_c = seqlen_c - (back_c + (1 << cp_world_log2) - 1 >> cp_world_log2);
                    }
                    float row_max_c = -CAKE_FMHA_INF;
                    float psum_c = 0.0f;
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                    int k_w_1 = warp_in_wg_c * 32 + lane;
                    int rows_hi_1 = ((N_ROWS > 32) ? 1 : 0);
                    int sT_base_1 = taddr + (unsigned int)(warp_in_wg_c * 32 << 16);
                    int oT_base_w_1 = taddr + 128 + (unsigned int)(warp_in_wg_c * 32 << 16);
                    int vis_a_1[16];
                    int vis_b_1[16];
                    #pragma unroll
                    for (int j_20 = 0; j_20 < 16; j_20++) {
                        int row_j_v_1 = (16 + j_20) / 8;
#if CAKE_FMHA_DCP_PROGRAM == 5
                        int _min_9 = ((row_j_v_1) < (q_len - 1) ? (row_j_v_1) : (q_len - 1));
                        int vis_j_v_2 = _min_9;
#else
                        int _min_8 = ((row_j_v_1) < (q_len - 1) ? (row_j_v_1) : (q_len - 1));
                        int vis_j_v_2 = _min_8;
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
                        int back_v_1 = q_len - 1 - vis_j_v_2 - phase_c;
                        int vis_v_1 = seqlen_c;
                        if (back_v_1 > 0) {
                            vis_v_1 = seqlen_c - (back_v_1 + (1 << cp_world_log2) - 1 >> cp_world_log2);
                        }
                        vis_a_1[j_20] = vis_v_1;
                        int row_j_v_0_1 = (48 + j_20) / 8;
#if CAKE_FMHA_DCP_PROGRAM == 5
                        int _min_10 = ((row_j_v_0_1) < (q_len - 1) ? (row_j_v_0_1) : (q_len - 1));
                        int vis_j_v_1_1 = _min_10;
#else
                        int _min_9 = ((row_j_v_0_1) < (q_len - 1) ? (row_j_v_0_1) : (q_len - 1));
                        int vis_j_v_1_1 = _min_9;
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
                        int back_v_2_1 = q_len - 1 - vis_j_v_1_1 - phase_c;
                        int vis_v_3_1 = seqlen_c;
                        if (back_v_2_1 > 0) {
                            vis_v_3_1 = seqlen_c - (back_v_2_1 + (1 << cp_world_log2) - 1 >> cp_world_log2);
                        }
                        vis_b_1[j_20] = vis_v_3_1;
                    }
                    float ref_a_1[16];
                    float ref_b_1[16];
                    float psum_a_1[16];
                    float psum_b_1[16];
                    ref_a_1[0] = -CAKE_FMHA_INF;
                    ref_a_1[1] = -CAKE_FMHA_INF;
                    ref_a_1[2] = -CAKE_FMHA_INF;
                    ref_a_1[3] = -CAKE_FMHA_INF;
                    ref_a_1[4] = -CAKE_FMHA_INF;
                    ref_a_1[5] = -CAKE_FMHA_INF;
                    ref_a_1[6] = -CAKE_FMHA_INF;
                    ref_a_1[7] = -CAKE_FMHA_INF;
                    ref_a_1[8] = -CAKE_FMHA_INF;
                    ref_a_1[9] = -CAKE_FMHA_INF;
                    ref_a_1[10] = -CAKE_FMHA_INF;
                    ref_a_1[11] = -CAKE_FMHA_INF;
                    ref_a_1[12] = -CAKE_FMHA_INF;
                    ref_a_1[13] = -CAKE_FMHA_INF;
                    ref_a_1[14] = -CAKE_FMHA_INF;
                    ref_a_1[15] = -CAKE_FMHA_INF;
                    ref_b_1[0] = -CAKE_FMHA_INF;
                    ref_b_1[1] = -CAKE_FMHA_INF;
                    ref_b_1[2] = -CAKE_FMHA_INF;
                    ref_b_1[3] = -CAKE_FMHA_INF;
                    ref_b_1[4] = -CAKE_FMHA_INF;
                    ref_b_1[5] = -CAKE_FMHA_INF;
                    ref_b_1[6] = -CAKE_FMHA_INF;
                    ref_b_1[7] = -CAKE_FMHA_INF;
                    ref_b_1[8] = -CAKE_FMHA_INF;
                    ref_b_1[9] = -CAKE_FMHA_INF;
                    ref_b_1[10] = -CAKE_FMHA_INF;
                    ref_b_1[11] = -CAKE_FMHA_INF;
                    ref_b_1[12] = -CAKE_FMHA_INF;
                    ref_b_1[13] = -CAKE_FMHA_INF;
                    ref_b_1[14] = -CAKE_FMHA_INF;
                    ref_b_1[15] = -CAKE_FMHA_INF;
                    psum_a_1[0] = 0.0f;
                    psum_a_1[1] = 0.0f;
                    psum_a_1[2] = 0.0f;
                    psum_a_1[3] = 0.0f;
                    psum_a_1[4] = 0.0f;
                    psum_a_1[5] = 0.0f;
                    psum_a_1[6] = 0.0f;
                    psum_a_1[7] = 0.0f;
                    psum_a_1[8] = 0.0f;
                    psum_a_1[9] = 0.0f;
                    psum_a_1[10] = 0.0f;
                    psum_a_1[11] = 0.0f;
                    psum_a_1[12] = 0.0f;
                    psum_a_1[13] = 0.0f;
                    psum_a_1[14] = 0.0f;
                    psum_a_1[15] = 0.0f;
                    psum_b_1[0] = 0.0f;
                    psum_b_1[1] = 0.0f;
                    psum_b_1[2] = 0.0f;
                    psum_b_1[3] = 0.0f;
                    psum_b_1[4] = 0.0f;
                    psum_b_1[5] = 0.0f;
                    psum_b_1[6] = 0.0f;
                    psum_b_1[7] = 0.0f;
                    psum_b_1[8] = 0.0f;
                    psum_b_1[9] = 0.0f;
                    psum_b_1[10] = 0.0f;
                    psum_b_1[11] = 0.0f;
                    psum_b_1[12] = 0.0f;
                    psum_b_1[13] = 0.0f;
                    psum_b_1[14] = 0.0f;
                    psum_b_1[15] = 0.0f;
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                    #pragma unroll 1
                    for (int n_1 = 0; n_1 < cnt_c; n_1++) {
                        if (wg_tid_c == 0) {
                        }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                        {
                            mbarrier_wait(s_full_addr + (sm_stage_c) * 8, sm_phase_c);
                        }
#else
                        mbarrier_wait(s_full_addr + (sm_stage_c) * 8, sm_phase_c);
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                        mbarrier_wait(s_full_addr + (sm_stage_c) * 8, sm_phase_c);
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                        if (wg_tid_c == 0) {
                        }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                        int my_block_w_1 = block_begin_c + cnt_c - 1 - n_1;
                        int kpos_1 = my_block_w_1 * BLOCK_N + k_w_1;
                        int par_w_1 = n_1 & 1;
                        float sv_a_1[16];
                        float sv_b_1[16];
                        tmem_ld_x16(&sv_a_1[0], (unsigned int)sT_base_1 + sm_stage_c * 64 + 16);
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        if (rows_hi_1 != 0) {
                            tmem_ld_x16(&sv_b_1[0], (unsigned int)sT_base_1 + sm_stage_c * 64 + 32 + 16);
#else
                        int my_block_c = block_begin_c + cnt_c - 1 - n_1;
                        int blk_pos_c = my_block_c * BLOCK_N + tok_base_c;
                        float sv_c[32];
                        float lmax_c = -CAKE_FMHA_INF;
                        float new_max_c = row_max_c;
                        float acc_scale_c = 1.0f;
                        int xm_off_c = (int)xm_slot_c * 128;
                        if (rows_live_c != 0) {
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                                : "=r"(*reinterpret_cast<uint32_t*>(&sv_c[0])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[1])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[2])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[3])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[4])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[5])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[6])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[7])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[8])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[9])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[10])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[11])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[12])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[13])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[14])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[15])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[16])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[17])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[18])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[19])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[20])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[21])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[22])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[23])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[24])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[25])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[26])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[27])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[28])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[29])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[30])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[31]))
                                : "r"((unsigned int)my_s_base_c + sm_stage_c * (unsigned int)BLOCK_N + 32));
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                        int my_block_c = block_begin_c + cnt_c - 1 - n_1;
                        int blk_pos_c = my_block_c * BLOCK_N + tok_base_c;
                        float sv_c[32];
                        float lmax_c = -CAKE_FMHA_INF;
                        float new_max_c = row_max_c;
                        float acc_scale_c = 1.0f;
                        int xm_off_c = (int)xm_slot_c * 128;
                        if (rows_live_c != 0) {
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                                : "=r"(*reinterpret_cast<uint32_t*>(&sv_c[0])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[1])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[2])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[3])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[4])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[5])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[6])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[7])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[8])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[9])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[10])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[11])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[12])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[13])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[14])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[15])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[16])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[17])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[18])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[19])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[20])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[21])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[22])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[23])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[24])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[25])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[26])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[27])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[28])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[29])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[30])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[31]))
                                : "r"((unsigned int)my_s_base_c + sm_stage_c * (unsigned int)BLOCK_N + 32));
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                        }
                        int exc_w_1 = 0;
                        #pragma unroll
                        for (int j_21 = 0; j_21 < 16; j_21++) {
                            if (kpos_1 >= vis_a_1[j_21]) {
                                sv_a_1[j_21] = -CAKE_FMHA_INF;
#else
                            int n_vis_c = vis_col_c - blk_pos_c;
                            if (my_row_c >= N_ROWS) {
                                n_vis_c = 0;
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                            int n_vis_c = vis_col_c - blk_pos_c;
                            if (my_row_c >= N_ROWS) {
                                n_vis_c = 0;
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                            }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                            if (sv_a_1[j_21] > ref_a_1[j_21] + thr_raw_c) {
                                exc_w_1 = 1;
#else
                            if (n_vis_c < 32) {
#if CAKE_FMHA_DCP_PROGRAM == 4
                                int _max_13 = ((n_vis_c) > (0) ? (n_vis_c) : (0));
                                int n_lo_c = _max_13;
#else
                                int _max_11 = ((n_vis_c) > (0) ? (n_vis_c) : (0));
                                int n_lo_c = _max_11;
#endif  // CAKE_FMHA_DCP_PROGRAM == 4
                                uint32_t _slice_lo_mask_1;
                                {
                                    int _lim_0 = n_lo_c;
                                    if (_lim_0 <= 0) { _slice_lo_mask_1 = 0u; }
                                    else if (_lim_0 >= 32) { _slice_lo_mask_1 = 0xFFFFFFFFu; }
                                    else {
                                        asm volatile("{"
                                            ".reg .u32 t;\n\t"
                                            "shl.b32 t, 1, %1;\n\t"
                                            "add.u32 %0, t, -1;\n\t"
                                            "}" : "=r"(_slice_lo_mask_1) : "r"(_lim_0));
                                    }
                                }
                                if (!(_slice_lo_mask_1 & (1u << 0))) sv_c[0] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 1))) sv_c[1] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 2))) sv_c[2] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 3))) sv_c[3] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 4))) sv_c[4] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 5))) sv_c[5] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 6))) sv_c[6] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 7))) sv_c[7] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 8))) sv_c[8] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 9))) sv_c[9] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 10))) sv_c[10] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 11))) sv_c[11] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 12))) sv_c[12] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 13))) sv_c[13] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 14))) sv_c[14] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 15))) sv_c[15] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 16))) sv_c[16] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 17))) sv_c[17] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 18))) sv_c[18] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 19))) sv_c[19] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 20))) sv_c[20] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 21))) sv_c[21] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 22))) sv_c[22] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 23))) sv_c[23] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 24))) sv_c[24] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 25))) sv_c[25] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 26))) sv_c[26] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 27))) sv_c[27] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 28))) sv_c[28] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 29))) sv_c[29] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 30))) sv_c[30] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 31))) sv_c[31] = -CAKE_FMHA_INF;
                            }
                            float2 _reg_reduce_max2_1 = {-CAKE_FMHA_INF, -CAKE_FMHA_INF};
                            _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(sv_c[0], sv_c[1]));
                            _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(sv_c[2], sv_c[3]));
                            _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(sv_c[4], sv_c[5]));
                            _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(sv_c[6], sv_c[7]));
                            _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(sv_c[8], sv_c[9]));
                            _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(sv_c[10], sv_c[11]));
                            _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(sv_c[12], sv_c[13]));
                            _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(sv_c[14], sv_c[15]));
                            _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(sv_c[16], sv_c[17]));
                            _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(sv_c[18], sv_c[19]));
                            _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(sv_c[20], sv_c[21]));
                            _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(sv_c[22], sv_c[23]));
                            _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(sv_c[24], sv_c[25]));
                            _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(sv_c[26], sv_c[27]));
                            _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(sv_c[28], sv_c[29]));
                            _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(sv_c[30], sv_c[31]));
                            float sv_c_max = row_max_reduce(_reg_reduce_max2_1);
                            lmax_c = sv_c_max;
                            float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, lmax_c, 16);
#if CAKE_FMHA_DCP_PROGRAM == 4
                            float _max_14 = max_noftz(lmax_c, _shfl_xor_2);
                            lmax_c = _max_14;
#else
                            float _max_12 = max_noftz(lmax_c, _shfl_xor_2);
                            lmax_c = _max_12;
#endif  // CAKE_FMHA_DCP_PROGRAM == 4
                            if (half_c == 0) {
                                smem_xmax[xm_off_c + 64 + my_row_c] = lmax_c;
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                            if (n_vis_c < 32) {
                                int _max_23 = ((n_vis_c) > (0) ? (n_vis_c) : (0));
                                int n_lo_c = _max_23;
                                uint32_t _slice_lo_mask_1;
                                {
                                    int _lim_0 = n_lo_c;
                                    if (_lim_0 <= 0) { _slice_lo_mask_1 = 0u; }
                                    else if (_lim_0 >= 32) { _slice_lo_mask_1 = 0xFFFFFFFFu; }
                                    else {
                                        asm volatile("{"
                                            ".reg .u32 t;\n\t"
                                            "shl.b32 t, 1, %1;\n\t"
                                            "add.u32 %0, t, -1;\n\t"
                                            "}" : "=r"(_slice_lo_mask_1) : "r"(_lim_0));
                                    }
                                }
                                if (!(_slice_lo_mask_1 & (1u << 0))) sv_c[0] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 1))) sv_c[1] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 2))) sv_c[2] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 3))) sv_c[3] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 4))) sv_c[4] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 5))) sv_c[5] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 6))) sv_c[6] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 7))) sv_c[7] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 8))) sv_c[8] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 9))) sv_c[9] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 10))) sv_c[10] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 11))) sv_c[11] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 12))) sv_c[12] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 13))) sv_c[13] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 14))) sv_c[14] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 15))) sv_c[15] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 16))) sv_c[16] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 17))) sv_c[17] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 18))) sv_c[18] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 19))) sv_c[19] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 20))) sv_c[20] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 21))) sv_c[21] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 22))) sv_c[22] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 23))) sv_c[23] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 24))) sv_c[24] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 25))) sv_c[25] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 26))) sv_c[26] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 27))) sv_c[27] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 28))) sv_c[28] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 29))) sv_c[29] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 30))) sv_c[30] = -CAKE_FMHA_INF;
                                if (!(_slice_lo_mask_1 & (1u << 31))) sv_c[31] = -CAKE_FMHA_INF;
                            }
                            float2 _reg_reduce_max2_1 = {-CAKE_FMHA_INF, -CAKE_FMHA_INF};
                            _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(sv_c[0], sv_c[1]));
                            _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(sv_c[2], sv_c[3]));
                            _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(sv_c[4], sv_c[5]));
                            _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(sv_c[6], sv_c[7]));
                            _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(sv_c[8], sv_c[9]));
                            _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(sv_c[10], sv_c[11]));
                            _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(sv_c[12], sv_c[13]));
                            _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(sv_c[14], sv_c[15]));
                            _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(sv_c[16], sv_c[17]));
                            _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(sv_c[18], sv_c[19]));
                            _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(sv_c[20], sv_c[21]));
                            _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(sv_c[22], sv_c[23]));
                            _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(sv_c[24], sv_c[25]));
                            _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(sv_c[26], sv_c[27]));
                            _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(sv_c[28], sv_c[29]));
                            _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(sv_c[30], sv_c[31]));
                            float sv_c_max = row_max_reduce(_reg_reduce_max2_1);
                            lmax_c = sv_c_max;
                            float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, lmax_c, 16);
                            float _max_24 = max_noftz(lmax_c, _shfl_xor_5);
                            lmax_c = _max_24;
                            if (half_c == 0) {
                                smem_xmax[xm_off_c + 64 + my_row_c] = lmax_c;
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                            }
                        }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                        if (rows_hi_1 != 0) {
                            #pragma unroll
                            for (int j_22 = 0; j_22 < 16; j_22++) {
                                if (kpos_1 >= vis_b_1[j_22]) {
                                    sv_b_1[j_22] = -CAKE_FMHA_INF;
                                }
                                if (sv_b_1[j_22] > ref_b_1[j_22] + thr_raw_c) {
                                    exc_w_1 = 1;
#else
                        asm volatile("barrier.sync 8, 256;" ::: "memory");
                        if (rows_live_c != 0) {
#if CAKE_FMHA_DCP_PROGRAM == 4
                            float _max_15 = max_noftz(lmax_c, smem_xmax[xm_off_c + my_row_c]);
                            lmax_c = _max_15;
#else
                            float _max_13 = max_noftz(lmax_c, smem_xmax[xm_off_c + my_row_c]);
                            lmax_c = _max_13;
#endif  // CAKE_FMHA_DCP_PROGRAM == 4
                            if (lmax_c > row_max_c + thr_raw_c) {
                                new_max_c = lmax_c;
                                if (row_max_c > -CAKE_FMHA_INF) {
                                    float _exp2_7 = approx_exp2(softmax_scale_log2 * (row_max_c - new_max_c));
                                    acc_scale_c = _exp2_7;
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                        asm volatile("barrier.sync 8, 256;" ::: "memory");
                        if (rows_live_c != 0) {
                            float _max_25 = max_noftz(lmax_c, smem_xmax[xm_off_c + my_row_c]);
                            lmax_c = _max_25;
                            if (lmax_c > row_max_c + thr_raw_c) {
                                new_max_c = lmax_c;
                                if (row_max_c > -CAKE_FMHA_INF) {
                                    float _exp2_7 = approx_exp2(softmax_scale_log2 * (row_max_c - new_max_c));
                                    acc_scale_c = _exp2_7;
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                                }
                            }
                        }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                        unsigned int flag_w_1 = 0;
                        int _vote_3 = __any_sync(0xFFFFFFFF, exc_w_1 != 0);
                        if (_vote_3 != 0) {
                            #pragma unroll
                            for (int j_23 = 0; j_23 < 16; j_23++) {
                                int _vote_4 = __any_sync(0xFFFFFFFF, sv_a_1[j_23] > ref_a_1[j_23] + thr_raw_c);
                                if (_vote_4 != 0) {
                                    flag_w_1 = flag_w_1 | (unsigned int)(1 << j_23);
                                }
                            }
                            if (rows_hi_1 != 0) {
                                #pragma unroll
                                for (int j_24 = 0; j_24 < 16; j_24++) {
                                    int _vote_5 = __any_sync(0xFFFFFFFF, sv_b_1[j_24] > ref_b_1[j_24] + thr_raw_c);
                                    if (_vote_5 != 0) {
                                        flag_w_1 = flag_w_1 | (unsigned int)(1 << 16 + j_24);
                                    }
                                }
                            }
                        }
                        if (lane == 0) {
                            swap_flags[8 + par_w_1 * 4 + warp_in_wg_c] = flag_w_1;
                        }
                        asm volatile("barrier.sync 9, 128;" ::: "memory");
                        unsigned int fl_w_1 = swap_flags[8 + par_w_1 * 4] | swap_flags[8 + par_w_1 * 4 + 1] | swap_flags[8 + par_w_1 * 4 + 2] | swap_flags[8 + par_w_1 * 4 + 3];
#else
                        xm_slot_c = xm_slot_c ^ 1;
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                        xm_slot_c = xm_slot_c ^ 1;
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                        if (wg_tid_c == 0) {
                        }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                        if (fl_w_1 != 0) {
                            int b4_2 = lane >> 4 & 1;
                            int b3_3 = lane >> 3 & 1;
                            int b2_3 = lane >> 2 & 1;
                            int b1_2 = lane >> 1 & 1;
                            float w8_2[8];
#else
                        if (rows_live_c != 0) {
                            float safe_max_c = ((new_max_c == -CAKE_FMHA_INF) ? 0.0f : new_max_c);
                            const float2 _fma_b2_2 = {softmax_scale_log2, softmax_scale_log2};
                            const float2 _fma_c2_3 = {(-safe_max_c) * softmax_scale_log2, (-safe_max_c) * softmax_scale_log2};
                            float2 _fma_pair_4 = fma_f32x2(make_float2(sv_c[0], sv_c[1]), _fma_b2_2, _fma_c2_3);
                            sv_c[0] = _fma_pair_4.x;
                            sv_c[1] = _fma_pair_4.y;
                            float2 _fma_pair_5 = fma_f32x2(make_float2(sv_c[2], sv_c[3]), _fma_b2_2, _fma_c2_3);
                            sv_c[2] = _fma_pair_5.x;
                            sv_c[3] = _fma_pair_5.y;
                            float2 _fma_pair_6 = fma_f32x2(make_float2(sv_c[4], sv_c[5]), _fma_b2_2, _fma_c2_3);
                            sv_c[4] = _fma_pair_6.x;
                            sv_c[5] = _fma_pair_6.y;
                            float2 _fma_pair_7 = fma_f32x2(make_float2(sv_c[6], sv_c[7]), _fma_b2_2, _fma_c2_3);
                            sv_c[6] = _fma_pair_7.x;
                            sv_c[7] = _fma_pair_7.y;
                            float2 _fma_pair_8 = fma_f32x2(make_float2(sv_c[8], sv_c[9]), _fma_b2_2, _fma_c2_3);
                            sv_c[8] = _fma_pair_8.x;
                            sv_c[9] = _fma_pair_8.y;
                            float2 _fma_pair_9 = fma_f32x2(make_float2(sv_c[10], sv_c[11]), _fma_b2_2, _fma_c2_3);
                            sv_c[10] = _fma_pair_9.x;
                            sv_c[11] = _fma_pair_9.y;
                            float2 _fma_pair_10 = fma_f32x2(make_float2(sv_c[12], sv_c[13]), _fma_b2_2, _fma_c2_3);
                            sv_c[12] = _fma_pair_10.x;
                            sv_c[13] = _fma_pair_10.y;
                            float2 _fma_pair_11 = fma_f32x2(make_float2(sv_c[14], sv_c[15]), _fma_b2_2, _fma_c2_3);
                            sv_c[14] = _fma_pair_11.x;
                            sv_c[15] = _fma_pair_11.y;
                            float2 _fma_pair_12 = fma_f32x2(make_float2(sv_c[16], sv_c[17]), _fma_b2_2, _fma_c2_3);
                            sv_c[16] = _fma_pair_12.x;
                            sv_c[17] = _fma_pair_12.y;
                            float2 _fma_pair_13 = fma_f32x2(make_float2(sv_c[18], sv_c[19]), _fma_b2_2, _fma_c2_3);
                            sv_c[18] = _fma_pair_13.x;
                            sv_c[19] = _fma_pair_13.y;
                            float2 _fma_pair_14 = fma_f32x2(make_float2(sv_c[20], sv_c[21]), _fma_b2_2, _fma_c2_3);
                            sv_c[20] = _fma_pair_14.x;
                            sv_c[21] = _fma_pair_14.y;
                            float2 _fma_pair_15 = fma_f32x2(make_float2(sv_c[22], sv_c[23]), _fma_b2_2, _fma_c2_3);
                            sv_c[22] = _fma_pair_15.x;
                            sv_c[23] = _fma_pair_15.y;
                            float2 _fma_pair_16 = fma_f32x2(make_float2(sv_c[24], sv_c[25]), _fma_b2_2, _fma_c2_3);
                            sv_c[24] = _fma_pair_16.x;
                            sv_c[25] = _fma_pair_16.y;
                            float2 _fma_pair_17 = fma_f32x2(make_float2(sv_c[26], sv_c[27]), _fma_b2_2, _fma_c2_3);
                            sv_c[26] = _fma_pair_17.x;
                            sv_c[27] = _fma_pair_17.y;
                            float2 _fma_pair_18 = fma_f32x2(make_float2(sv_c[28], sv_c[29]), _fma_b2_2, _fma_c2_3);
                            sv_c[28] = _fma_pair_18.x;
                            sv_c[29] = _fma_pair_18.y;
                            float2 _fma_pair_19 = fma_f32x2(make_float2(sv_c[30], sv_c[31]), _fma_b2_2, _fma_c2_3);
                            sv_c[30] = _fma_pair_19.x;
                            sv_c[31] = _fma_pair_19.y;
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                        if (rows_live_c != 0) {
                            float safe_max_c = ((new_max_c == -CAKE_FMHA_INF) ? 0.0f : new_max_c);
                            const float2 _fma_b2_2 = {softmax_scale_log2, softmax_scale_log2};
                            const float2 _fma_c2_3 = {(-safe_max_c) * softmax_scale_log2, (-safe_max_c) * softmax_scale_log2};
                            float2 _fma_pair_4 = fma_f32x2(make_float2(sv_c[0], sv_c[1]), _fma_b2_2, _fma_c2_3);
                            sv_c[0] = _fma_pair_4.x;
                            sv_c[1] = _fma_pair_4.y;
                            float2 _fma_pair_5 = fma_f32x2(make_float2(sv_c[2], sv_c[3]), _fma_b2_2, _fma_c2_3);
                            sv_c[2] = _fma_pair_5.x;
                            sv_c[3] = _fma_pair_5.y;
                            float2 _fma_pair_6 = fma_f32x2(make_float2(sv_c[4], sv_c[5]), _fma_b2_2, _fma_c2_3);
                            sv_c[4] = _fma_pair_6.x;
                            sv_c[5] = _fma_pair_6.y;
                            float2 _fma_pair_7 = fma_f32x2(make_float2(sv_c[6], sv_c[7]), _fma_b2_2, _fma_c2_3);
                            sv_c[6] = _fma_pair_7.x;
                            sv_c[7] = _fma_pair_7.y;
                            float2 _fma_pair_8 = fma_f32x2(make_float2(sv_c[8], sv_c[9]), _fma_b2_2, _fma_c2_3);
                            sv_c[8] = _fma_pair_8.x;
                            sv_c[9] = _fma_pair_8.y;
                            float2 _fma_pair_9 = fma_f32x2(make_float2(sv_c[10], sv_c[11]), _fma_b2_2, _fma_c2_3);
                            sv_c[10] = _fma_pair_9.x;
                            sv_c[11] = _fma_pair_9.y;
                            float2 _fma_pair_10 = fma_f32x2(make_float2(sv_c[12], sv_c[13]), _fma_b2_2, _fma_c2_3);
                            sv_c[12] = _fma_pair_10.x;
                            sv_c[13] = _fma_pair_10.y;
                            float2 _fma_pair_11 = fma_f32x2(make_float2(sv_c[14], sv_c[15]), _fma_b2_2, _fma_c2_3);
                            sv_c[14] = _fma_pair_11.x;
                            sv_c[15] = _fma_pair_11.y;
                            float2 _fma_pair_12 = fma_f32x2(make_float2(sv_c[16], sv_c[17]), _fma_b2_2, _fma_c2_3);
                            sv_c[16] = _fma_pair_12.x;
                            sv_c[17] = _fma_pair_12.y;
                            float2 _fma_pair_13 = fma_f32x2(make_float2(sv_c[18], sv_c[19]), _fma_b2_2, _fma_c2_3);
                            sv_c[18] = _fma_pair_13.x;
                            sv_c[19] = _fma_pair_13.y;
                            float2 _fma_pair_14 = fma_f32x2(make_float2(sv_c[20], sv_c[21]), _fma_b2_2, _fma_c2_3);
                            sv_c[20] = _fma_pair_14.x;
                            sv_c[21] = _fma_pair_14.y;
                            float2 _fma_pair_15 = fma_f32x2(make_float2(sv_c[22], sv_c[23]), _fma_b2_2, _fma_c2_3);
                            sv_c[22] = _fma_pair_15.x;
                            sv_c[23] = _fma_pair_15.y;
                            float2 _fma_pair_16 = fma_f32x2(make_float2(sv_c[24], sv_c[25]), _fma_b2_2, _fma_c2_3);
                            sv_c[24] = _fma_pair_16.x;
                            sv_c[25] = _fma_pair_16.y;
                            float2 _fma_pair_17 = fma_f32x2(make_float2(sv_c[26], sv_c[27]), _fma_b2_2, _fma_c2_3);
                            sv_c[26] = _fma_pair_17.x;
                            sv_c[27] = _fma_pair_17.y;
                            float2 _fma_pair_18 = fma_f32x2(make_float2(sv_c[28], sv_c[29]), _fma_b2_2, _fma_c2_3);
                            sv_c[28] = _fma_pair_18.x;
                            sv_c[29] = _fma_pair_18.y;
                            float2 _fma_pair_19 = fma_f32x2(make_float2(sv_c[30], sv_c[31]), _fma_b2_2, _fma_c2_3);
                            sv_c[30] = _fma_pair_19.x;
                            sv_c[31] = _fma_pair_19.y;
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                            #pragma unroll
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                            for (int i_12 = 0; i_12 < 8; i_12++) {
                                float send8_4 = ((b4_2 != 0) ? sv_a_1[i_12] : sv_a_1[i_12 + 8]);
                                float keep8_4 = ((b4_2 != 0) ? sv_a_1[i_12 + 8] : sv_a_1[i_12]);
                                float _shfl_xor_20 = __shfl_xor_sync(0xFFFFFFFF, send8_4, 16);
#if CAKE_FMHA_DCP_PROGRAM == 5
                                float _max_32 = max_noftz(keep8_4, _shfl_xor_20);
                                w8_2[i_12] = _max_32;
#else
                                float _max_30 = max_noftz(keep8_4, _shfl_xor_20);
                                w8_2[i_12] = _max_30;
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
#else
                            for (int _le = 0; _le < 32; _le++) {
                                sv_c[_le] = approx_exp2(sv_c[_le]);
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                            for (int _le = 0; _le < 32; _le++) {
                                sv_c[_le] = approx_exp2(sv_c[_le]);
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                            }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                            float x4_2[4];
#else
                            float2 _reg_reduce_sum2_20 = make_float2(0.0f, 0.0f);
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv_c[0], sv_c[1]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv_c[2], sv_c[3]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv_c[4], sv_c[5]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv_c[6], sv_c[7]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv_c[8], sv_c[9]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv_c[10], sv_c[11]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv_c[12], sv_c[13]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv_c[14], sv_c[15]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv_c[16], sv_c[17]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv_c[18], sv_c[19]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv_c[20], sv_c[21]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv_c[22], sv_c[23]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv_c[24], sv_c[25]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv_c[26], sv_c[27]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv_c[28], sv_c[29]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv_c[30], sv_c[31]));
                            float sv_c_sum = _reg_reduce_sum2_20.x + _reg_reduce_sum2_20.y;
                            float lsum_c = sv_c_sum;
                            psum_c = psum_c * acc_scale_c + lsum_c;
                            row_max_c = new_max_c;
                            unsigned int regs_pc[16];
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                            float2 _reg_reduce_sum2_20 = make_float2(0.0f, 0.0f);
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv_c[0], sv_c[1]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv_c[2], sv_c[3]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv_c[4], sv_c[5]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv_c[6], sv_c[7]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv_c[8], sv_c[9]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv_c[10], sv_c[11]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv_c[12], sv_c[13]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv_c[14], sv_c[15]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv_c[16], sv_c[17]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv_c[18], sv_c[19]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv_c[20], sv_c[21]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv_c[22], sv_c[23]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv_c[24], sv_c[25]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv_c[26], sv_c[27]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv_c[28], sv_c[29]));
                            _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(sv_c[30], sv_c[31]));
                            float sv_c_sum = _reg_reduce_sum2_20.x + _reg_reduce_sum2_20.y;
                            float lsum_c = sv_c_sum;
                            psum_c = psum_c * acc_scale_c + lsum_c;
                            row_max_c = new_max_c;
                            unsigned int regs_pc[16];
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                            #pragma unroll
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                            for (int i_13 = 0; i_13 < 4; i_13++) {
                                float send4_4 = ((b3_3 != 0) ? w8_2[i_13] : w8_2[i_13 + 4]);
                                float keep4_4 = ((b3_3 != 0) ? w8_2[i_13 + 4] : w8_2[i_13]);
                                float _shfl_xor_21 = __shfl_xor_sync(0xFFFFFFFF, send4_4, 8);
#if CAKE_FMHA_DCP_PROGRAM == 5
                                float _max_33 = max_noftz(keep4_4, _shfl_xor_21);
                                x4_2[i_13] = _max_33;
#else
                                float _max_31 = max_noftz(keep4_4, _shfl_xor_21);
                                x4_2[i_13] = _max_31;
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
#else
                            for (int _lp = 0; _lp < 16; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sv_c[_lp*2 + 0], sv_c[_lp*2+1 + 0]));
                                regs_pc[_lp] = *(uint32_t*)&_bf2;
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                            for (int _lp = 0; _lp < 16; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sv_c[_lp*2 + 0], sv_c[_lp*2+1 + 0]));
                                regs_pc[_lp] = *(uint32_t*)&_bf2;
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                            }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                            float y2_2[2];
                            #pragma unroll
                            for (int i_14 = 0; i_14 < 2; i_14++) {
                                float send2_4 = ((b2_3 != 0) ? x4_2[i_14] : x4_2[i_14 + 2]);
                                float keep2_4 = ((b2_3 != 0) ? x4_2[i_14 + 2] : x4_2[i_14]);
                                float _shfl_xor_22 = __shfl_xor_sync(0xFFFFFFFF, send2_4, 4);
#if CAKE_FMHA_DCP_PROGRAM == 5
                                float _max_34 = max_noftz(keep2_4, _shfl_xor_22);
                                y2_2[i_14] = _max_34;
#else
                                float _max_32 = max_noftz(keep2_4, _shfl_xor_22);
                                y2_2[i_14] = _max_32;
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
                            }
                            float send1_2 = ((b1_2 != 0) ? y2_2[0] : y2_2[1]);
                            float keep1_2 = ((b1_2 != 0) ? y2_2[1] : y2_2[0]);
                            float _shfl_xor_23 = __shfl_xor_sync(0xFFFFFFFF, send1_2, 2);
#if CAKE_FMHA_DCP_PROGRAM == 5
                            float _max_35 = max_noftz(keep1_2, _shfl_xor_23);
                            float z_2 = _max_35;
#else
                            float _max_33 = max_noftz(keep1_2, _shfl_xor_23);
                            float z_2 = _max_33;
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
                            float _shfl_xor_24 = __shfl_xor_sync(0xFFFFFFFF, z_2, 1);
#if CAKE_FMHA_DCP_PROGRAM == 5
                            float _max_36 = max_noftz(z_2, _shfl_xor_24);
                            z_2 = _max_36;
#else
                            float _max_34 = max_noftz(z_2, _shfl_xor_24);
                            z_2 = _max_34;
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
                            float zmax_1 = z_2;
                            if ((lane & 1) == 0) {
                                swap_part[(32 + (lane >> 1 & 15)) * 4 + warp_in_wg_c] = zmax_1;
                            }
                            if (rows_hi_1 != 0) {
                                int b4_0_2 = lane >> 4 & 1;
                                int b3_1_2 = lane >> 3 & 1;
                                int b2_2_2 = lane >> 2 & 1;
                                int b1_3_2 = lane >> 1 & 1;
                                float w8_4_2[8];
                                #pragma unroll
                                for (int i_15 = 0; i_15 < 8; i_15++) {
                                    float send8_5 = ((b4_0_2 != 0) ? sv_b_1[i_15] : sv_b_1[i_15 + 8]);
                                    float keep8_5 = ((b4_0_2 != 0) ? sv_b_1[i_15 + 8] : sv_b_1[i_15]);
                                    float _shfl_xor_25 = __shfl_xor_sync(0xFFFFFFFF, send8_5, 16);
#if CAKE_FMHA_DCP_PROGRAM == 5
                                    float _max_37 = max_noftz(keep8_5, _shfl_xor_25);
                                    w8_4_2[i_15] = _max_37;
#else
                                    float _max_35 = max_noftz(keep8_5, _shfl_xor_25);
                                    w8_4_2[i_15] = _max_35;
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
                                }
                                float x4_5_2[4];
                                #pragma unroll
                                for (int i_16 = 0; i_16 < 4; i_16++) {
                                    float send4_5 = ((b3_1_2 != 0) ? w8_4_2[i_16] : w8_4_2[i_16 + 4]);
                                    float keep4_5 = ((b3_1_2 != 0) ? w8_4_2[i_16 + 4] : w8_4_2[i_16]);
                                    float _shfl_xor_26 = __shfl_xor_sync(0xFFFFFFFF, send4_5, 8);
#if CAKE_FMHA_DCP_PROGRAM == 5
                                    float _max_38 = max_noftz(keep4_5, _shfl_xor_26);
                                    x4_5_2[i_16] = _max_38;
#else
                                    float _max_36 = max_noftz(keep4_5, _shfl_xor_26);
                                    x4_5_2[i_16] = _max_36;
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
                                }
                                float y2_6_2[2];
                                #pragma unroll
                                for (int i_17 = 0; i_17 < 2; i_17++) {
                                    float send2_5 = ((b2_2_2 != 0) ? x4_5_2[i_17] : x4_5_2[i_17 + 2]);
                                    float keep2_5 = ((b2_2_2 != 0) ? x4_5_2[i_17 + 2] : x4_5_2[i_17]);
                                    float _shfl_xor_27 = __shfl_xor_sync(0xFFFFFFFF, send2_5, 4);
#if CAKE_FMHA_DCP_PROGRAM == 5
                                    float _max_39 = max_noftz(keep2_5, _shfl_xor_27);
                                    y2_6_2[i_17] = _max_39;
#else
                                    float _max_37 = max_noftz(keep2_5, _shfl_xor_27);
                                    y2_6_2[i_17] = _max_37;
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
                                }
                                float send1_7_2 = ((b1_3_2 != 0) ? y2_6_2[0] : y2_6_2[1]);
                                float keep1_8_2 = ((b1_3_2 != 0) ? y2_6_2[1] : y2_6_2[0]);
                                float _shfl_xor_28 = __shfl_xor_sync(0xFFFFFFFF, send1_7_2, 2);
#if CAKE_FMHA_DCP_PROGRAM == 5
                                float _max_40 = max_noftz(keep1_8_2, _shfl_xor_28);
                                float z_9_2 = _max_40;
#else
                                float _max_38 = max_noftz(keep1_8_2, _shfl_xor_28);
                                float z_9_2 = _max_38;
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
                                float _shfl_xor_29 = __shfl_xor_sync(0xFFFFFFFF, z_9_2, 1);
#if CAKE_FMHA_DCP_PROGRAM == 5
                                float _max_41 = max_noftz(z_9_2, _shfl_xor_29);
                                z_9_2 = _max_41;
#else
                                float _max_39 = max_noftz(z_9_2, _shfl_xor_29);
                                z_9_2 = _max_39;
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
                                float zmax_10_1 = z_9_2;
                                if ((lane & 1) == 0) {
                                    swap_part[(48 + (lane >> 1 & 15)) * 4 + warp_in_wg_c] = zmax_10_1;
                                }
                            }
                            if (wg_tid_c == 0) {
                            }
                            asm volatile("barrier.sync 9, 128;" ::: "memory");
                            if (wg_tid_c == 0) {
                            }
                            if (n_1 == 0) {
                                float p64s_1[64];
                                #pragma unroll
                                for (int j_25 = 0; j_25 < 16; j_25++) {
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&p64s_1[4 * j_25])), "=r"(*reinterpret_cast<uint32_t*>(&p64s_1[(4 * j_25) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&p64s_1[(4 * j_25) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&p64s_1[(4 * j_25) + 3]))
                                        : "r"(swap_part_addr + (unsigned int)((32 + j_25) * 16)));
                                }
                                #pragma unroll
                                for (int j_26 = 0; j_26 < 16; j_26++) {
#if CAKE_FMHA_DCP_PROGRAM == 5
                                    float _max_42 = max_noftz(p64s_1[4 * j_26], p64s_1[4 * j_26 + 1]);
                                    float _max_43 = max_noftz(p64s_1[4 * j_26 + 2], p64s_1[4 * j_26 + 3]);
                                    float _max_44 = max_noftz(_max_42, _max_43);
                                    float bm_sj_2 = _max_44;
#else
                                    float _max_40 = max_noftz(p64s_1[4 * j_26], p64s_1[4 * j_26 + 1]);
                                    float _max_41 = max_noftz(p64s_1[4 * j_26 + 2], p64s_1[4 * j_26 + 3]);
                                    float _max_42 = max_noftz(_max_40, _max_41);
                                    float bm_sj_2 = _max_42;
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
                                    int flagged_sj_2 = fl_w_1 >> (unsigned int)j_26 & 1;
                                    ref_a_1[j_26] = ((flagged_sj_2 != 0) ? bm_sj_2 : ref_a_1[j_26]);
                                }
                                if (rows_hi_1 != 0) {
                                    float p64s_0_1[64];
                                    #pragma unroll
                                    for (int j_27 = 0; j_27 < 16; j_27++) {
                                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                            : "=r"(*reinterpret_cast<uint32_t*>(&p64s_0_1[4 * j_27])), "=r"(*reinterpret_cast<uint32_t*>(&p64s_0_1[(4 * j_27) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&p64s_0_1[(4 * j_27) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&p64s_0_1[(4 * j_27) + 3]))
                                            : "r"(swap_part_addr + (unsigned int)((48 + j_27) * 16)));
                                    }
                                    #pragma unroll
                                    for (int j_28 = 0; j_28 < 16; j_28++) {
#if CAKE_FMHA_DCP_PROGRAM == 5
                                        float _max_45 = max_noftz(p64s_0_1[4 * j_28], p64s_0_1[4 * j_28 + 1]);
                                        float _max_46 = max_noftz(p64s_0_1[4 * j_28 + 2], p64s_0_1[4 * j_28 + 3]);
                                        float _max_47 = max_noftz(_max_45, _max_46);
                                        float bm_sj_3 = _max_47;
#else
                                        float _max_43 = max_noftz(p64s_0_1[4 * j_28], p64s_0_1[4 * j_28 + 1]);
                                        float _max_44 = max_noftz(p64s_0_1[4 * j_28 + 2], p64s_0_1[4 * j_28 + 3]);
                                        float _max_45 = max_noftz(_max_43, _max_44);
                                        float bm_sj_3 = _max_45;
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
                                        int flagged_sj_3 = fl_w_1 >> (unsigned int)(16 + j_28) & 1;
                                        ref_b_1[j_28] = ((flagged_sj_3 != 0) ? bm_sj_3 : ref_b_1[j_28]);
                                    }
                                }
                            } else {
                                float p64_1[64];
                                #pragma unroll
                                for (int j_29 = 0; j_29 < 16; j_29++) {
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&p64_1[4 * j_29])), "=r"(*reinterpret_cast<uint32_t*>(&p64_1[(4 * j_29) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&p64_1[(4 * j_29) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&p64_1[(4 * j_29) + 3]))
                                        : "r"(swap_part_addr + (unsigned int)((32 + j_29) * 16)));
                                }
                                float my_acc_2 = 1.0f;
                                #pragma unroll
                                for (int j_30 = 0; j_30 < 16; j_30++) {
#if CAKE_FMHA_DCP_PROGRAM == 5
                                    float _max_48 = max_noftz(p64_1[4 * j_30], p64_1[4 * j_30 + 1]);
                                    float _max_49 = max_noftz(p64_1[4 * j_30 + 2], p64_1[4 * j_30 + 3]);
                                    float _max_50 = max_noftz(_max_48, _max_49);
                                    float bm_j_2 = _max_50;
#else
                                    float _max_46 = max_noftz(p64_1[4 * j_30], p64_1[4 * j_30 + 1]);
                                    float _max_47 = max_noftz(p64_1[4 * j_30 + 2], p64_1[4 * j_30 + 3]);
                                    float _max_48 = max_noftz(_max_46, _max_47);
                                    float bm_j_2 = _max_48;
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
                                    int flagged_j_2 = fl_w_1 >> (unsigned int)j_30 & 1;
                                    float _exp2_8 = approx_exp2(softmax_scale_log2 * (ref_a_1[j_30] - bm_j_2));
                                    float e_j_2 = _exp2_8;
                                    float acc_j_2 = ((flagged_j_2 != 0 && ref_a_1[j_30] > -CAKE_FMHA_INF) ? e_j_2 : 1.0f);
                                    psum_a_1[j_30] = psum_a_1[j_30] * acc_j_2;
                                    ref_a_1[j_30] = ((flagged_j_2 != 0) ? bm_j_2 : ref_a_1[j_30]);
                                    my_acc_2 = ((lane == j_30) ? acc_j_2 : my_acc_2);
                                }
                                if (warp_in_wg_c == 0) {
                                    if (lane < 16) {
                                        swap_acc[16 + lane] = my_acc_2;
                                    }
                                }
                                if (rows_hi_1 != 0) {
                                    float p64_0_1[64];
                                    #pragma unroll
                                    for (int j_31 = 0; j_31 < 16; j_31++) {
                                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                            : "=r"(*reinterpret_cast<uint32_t*>(&p64_0_1[4 * j_31])), "=r"(*reinterpret_cast<uint32_t*>(&p64_0_1[(4 * j_31) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&p64_0_1[(4 * j_31) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&p64_0_1[(4 * j_31) + 3]))
                                            : "r"(swap_part_addr + (unsigned int)((48 + j_31) * 16)));
                                    }
                                    float my_acc_1_1 = 1.0f;
                                    #pragma unroll
                                    for (int j_32 = 0; j_32 < 16; j_32++) {
#if CAKE_FMHA_DCP_PROGRAM == 5
                                        float _max_51 = max_noftz(p64_0_1[4 * j_32], p64_0_1[4 * j_32 + 1]);
                                        float _max_52 = max_noftz(p64_0_1[4 * j_32 + 2], p64_0_1[4 * j_32 + 3]);
                                        float _max_53 = max_noftz(_max_51, _max_52);
                                        float bm_j_3 = _max_53;
#else
                                        float _max_49 = max_noftz(p64_0_1[4 * j_32], p64_0_1[4 * j_32 + 1]);
                                        float _max_50 = max_noftz(p64_0_1[4 * j_32 + 2], p64_0_1[4 * j_32 + 3]);
                                        float _max_51 = max_noftz(_max_49, _max_50);
                                        float bm_j_3 = _max_51;
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
                                        int flagged_j_3 = fl_w_1 >> (unsigned int)(16 + j_32) & 1;
                                        float _exp2_9 = approx_exp2(softmax_scale_log2 * (ref_b_1[j_32] - bm_j_3));
                                        float e_j_3 = _exp2_9;
                                        float acc_j_3 = ((flagged_j_3 != 0 && ref_b_1[j_32] > -CAKE_FMHA_INF) ? e_j_3 : 1.0f);
                                        psum_b_1[j_32] = psum_b_1[j_32] * acc_j_3;
                                        ref_b_1[j_32] = ((flagged_j_3 != 0) ? bm_j_3 : ref_b_1[j_32]);
                                        my_acc_1_1 = ((lane == j_32) ? acc_j_3 : my_acc_1_1);
                                    }
                                    if (warp_in_wg_c == 0) {
                                        if (lane < 16) {
                                            swap_acc[48 + lane] = my_acc_1_1;
                                        }
                                    }
                                }
                            }
                            if (wg_tid_c == 0) {
                            }
                            asm volatile("barrier.sync 9, 128;" ::: "memory");
#else
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x32bx2.x16.b32"
                                " [%0], 32, {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                                :: "r"((unsigned int)my_s_base_c + sm_stage_c * (unsigned int)BLOCK_N + 16), "r"(*reinterpret_cast<const uint32_t*>(&regs_pc[0])), "r"(*reinterpret_cast<const uint32_t*>(&regs_pc[1])), "r"(*reinterpret_cast<const uint32_t*>(&regs_pc[2])), "r"(*reinterpret_cast<const uint32_t*>(&regs_pc[3])), "r"(*reinterpret_cast<const uint32_t*>(&regs_pc[4])), "r"(*reinterpret_cast<const uint32_t*>(&regs_pc[5])), "r"(*reinterpret_cast<const uint32_t*>(&regs_pc[6])), "r"(*reinterpret_cast<const uint32_t*>(&regs_pc[7])), "r"(*reinterpret_cast<const uint32_t*>(&regs_pc[8])), "r"(*reinterpret_cast<const uint32_t*>(&regs_pc[9])), "r"(*reinterpret_cast<const uint32_t*>(&regs_pc[10])), "r"(*reinterpret_cast<const uint32_t*>(&regs_pc[11])), "r"(*reinterpret_cast<const uint32_t*>(&regs_pc[12])), "r"(*reinterpret_cast<const uint32_t*>(&regs_pc[13])), "r"(*reinterpret_cast<const uint32_t*>(&regs_pc[14])), "r"(*reinterpret_cast<const uint32_t*>(&regs_pc[15])));
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x32bx2.x16.b32"
                                " [%0], 32, {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                                :: "r"((unsigned int)my_s_base_c + sm_stage_c * (unsigned int)BLOCK_N + 16), "r"(*reinterpret_cast<const uint32_t*>(&regs_pc[0])), "r"(*reinterpret_cast<const uint32_t*>(&regs_pc[1])), "r"(*reinterpret_cast<const uint32_t*>(&regs_pc[2])), "r"(*reinterpret_cast<const uint32_t*>(&regs_pc[3])), "r"(*reinterpret_cast<const uint32_t*>(&regs_pc[4])), "r"(*reinterpret_cast<const uint32_t*>(&regs_pc[5])), "r"(*reinterpret_cast<const uint32_t*>(&regs_pc[6])), "r"(*reinterpret_cast<const uint32_t*>(&regs_pc[7])), "r"(*reinterpret_cast<const uint32_t*>(&regs_pc[8])), "r"(*reinterpret_cast<const uint32_t*>(&regs_pc[9])), "r"(*reinterpret_cast<const uint32_t*>(&regs_pc[10])), "r"(*reinterpret_cast<const uint32_t*>(&regs_pc[11])), "r"(*reinterpret_cast<const uint32_t*>(&regs_pc[12])), "r"(*reinterpret_cast<const uint32_t*>(&regs_pc[13])), "r"(*reinterpret_cast<const uint32_t*>(&regs_pc[14])), "r"(*reinterpret_cast<const uint32_t*>(&regs_pc[15])));
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                        }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                        if (wg_tid_c == 0) {
                        }
                        #pragma unroll
                        for (int j_33 = 0; j_33 < 16; j_33++) {
                            float safe_j_2 = ((ref_a_1[j_33] == -CAKE_FMHA_INF) ? 0.0f : ref_a_1[j_33]);
                            float _fma_8 = __fmaf_rn(sv_a_1[j_33], softmax_scale_log2, (-safe_j_2) * softmax_scale_log2);
                            sv_a_1[j_33] = _fma_8;
                        }
                        #pragma unroll
                        for (int _le = 0; _le < 16; _le++) {
                            sv_a_1[_le] = approx_exp2(sv_a_1[_le]);
                        }
                        #pragma unroll
                        for (int j_34 = 0; j_34 < 16; j_34++) {
                            psum_a_1[j_34] = psum_a_1[j_34] + sv_a_1[j_34];
                        }
                        if (rows_hi_1 != 0) {
                            #pragma unroll
                            for (int j_35 = 0; j_35 < 16; j_35++) {
                                float safe_j_3 = ((ref_b_1[j_35] == -CAKE_FMHA_INF) ? 0.0f : ref_b_1[j_35]);
                                float _fma_9 = __fmaf_rn(sv_b_1[j_35], softmax_scale_log2, (-safe_j_3) * softmax_scale_log2);
                                sv_b_1[j_35] = _fma_9;
                            }
                            #pragma unroll
                            for (int _le = 0; _le < 16; _le++) {
                                sv_b_1[_le] = approx_exp2(sv_b_1[_le]);
                            }
                            #pragma unroll
                            for (int j_36 = 0; j_36 < 16; j_36++) {
                                psum_b_1[j_36] = psum_b_1[j_36] + sv_b_1[j_36];
                            }
                        }
                        if (wg_tid_c == 0) {
                        }
                        if (wg_tid_c == 0) {
                        }
                        unsigned int k_blk_o_1 = pv_base + (unsigned int)n_1;
                        if (rows_hi_1 != 0) {
                            if (k_blk_o_1 > 0) {
                                unsigned int k_pv1f_1 = k_blk_o_1 - 1;
                                {
                                    mbarrier_wait(o_ready_addr + ((int)(k_pv1f_1 & 1)) * 8, (int)(k_pv1f_1 >> 1 & 1));
                                }
                            }
                        } else if (k_blk_o_1 >= 2) {
                            unsigned int k_pv2f_1 = k_blk_o_1 - 2;
                            {
                                mbarrier_wait(o_ready_addr + ((int)(k_pv2f_1 & 1)) * 8, (int)(k_pv2f_1 >> 1 & 1));
                            }
                        }
                        if (wg_tid_c == 0) {
                        }
                        if (fl_w_1 != 0) {
                            if (n_1 != 0) {
                                unsigned int k_pv1r_1 = k_blk_o_1 - 1;
                                {
                                    mbarrier_wait(o_ready_addr + ((int)(k_pv1r_1 & 1)) * 8, (int)(k_pv1r_1 >> 1 & 1));
#else
                        int _vote_0 = __any_sync(0xFFFFFFFF, acc_scale_c != 1.0f);
                        if (_vote_0 != 0) {
                            if (n_1 > 0) {
                                unsigned int k_c = pv_base + (unsigned int)n_1 - 1;
                                int o_st_c = (int)(k_c & 1);
                                int o_ph_c = (int)(k_c >> 1 & 1);
                                mbarrier_wait(o_ready_addr + (o_st_c) * 8, o_ph_c);
                                if (wg_tid_c == 0) {
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                        int _vote_2 = __any_sync(0xFFFFFFFF, acc_scale_c != 1.0f);
                        if (_vote_2 != 0) {
                            if (n_1 > 0) {
                                unsigned int k_c = pv_base + (unsigned int)n_1 - 1;
                                int o_st_c = (int)(k_c & 1);
                                int o_ph_c = (int)(k_c >> 1 & 1);
                                mbarrier_wait(o_ready_addr + (o_st_c) * 8, o_ph_c);
                                if (wg_tid_c == 0) {
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                                }
                                asm volatile("tcgen05.fence::after_thread_sync;");
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                                float acc16_1[16];
                                #pragma unroll
#if CAKE_FMHA_DCP_PROGRAM == 5
                                for (int q_3 = 0; q_3 < 4; q_3++) {
#else
                                for (int q_2 = 0; q_2 < 4; q_2++) {
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
#if CAKE_FMHA_DCP_PROGRAM == 5
                                        : "=r"(*reinterpret_cast<uint32_t*>(&acc16_1[4 * q_3])), "=r"(*reinterpret_cast<uint32_t*>(&acc16_1[(4 * q_3) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&acc16_1[(4 * q_3) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&acc16_1[(4 * q_3) + 3]))
                                        : "r"(swap_acc_addr + (unsigned int)((16 + 4 * q_3) * 4)));
#else
                                        : "=r"(*reinterpret_cast<uint32_t*>(&acc16_1[4 * q_2])), "=r"(*reinterpret_cast<uint32_t*>(&acc16_1[(4 * q_2) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&acc16_1[(4 * q_2) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&acc16_1[(4 * q_2) + 3]))
                                        : "r"(swap_acc_addr + (unsigned int)((16 + 4 * q_2) * 4)));
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
                                }
                                float _tmem_load_2[16];
                                tmem_ld_x16(&_tmem_load_2[0], oT_base_w_1 + 16);
#else
                                float _tmem_load_0[64];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[31]))
                                    : "r"(o_row_base));
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                                float _tmem_load_0[64];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[31]))
                                    : "r"(o_row_base));
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                                #pragma unroll
                                for (int j_37 = 0; j_37 < 16; j_37++) {
                                    _tmem_load_2[j_37] = _tmem_load_2[j_37] * acc16_1[j_37];
                                }
                                tmem_st_x16_f32(oT_base_w_1 + 16, _tmem_load_2);
#else
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[32])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[33])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[34])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[35])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[36])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[37])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[38])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[39])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[40])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[41])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[42])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[43])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[44])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[45])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[46])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[47])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[48])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[49])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[50])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[51])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[52])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[53])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[54])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[55])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[56])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[57])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[58])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[59])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[60])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[61])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[62])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[63]))
                                    : "r"(o_row_base + 32));
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                #if __CUDA_ARCH__ >= 1000
                                const float2 _scale2_21 = {acc_scale_c, acc_scale_c};
                                #pragma unroll
                                for (int _ls = 0; _ls < 32; _ls++)
                                    mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_0)[_ls], _scale2_21);
                                #else
                                #pragma unroll
                                for (int _ls = 0; _ls < 64; _ls++) {
                                    _tmem_load_0[_ls] = _tmem_load_0[_ls] * acc_scale_c;
                                }
                                #endif
                                asm volatile(
                                    "tcgen05.st.sync.aligned.16x32bx2.x64.b32"
                                    " [%0], 64, {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64};"
                                    :: "r"(o_row_base), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[15])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[16])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[17])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[18])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[19])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[20])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[21])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[22])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[23])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[24])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[25])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[26])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[27])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[28])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[29])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[30])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[31])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[32])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[33])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[34])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[35])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[36])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[37])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[38])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[39])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[40])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[41])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[42])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[43])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[44])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[45])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[46])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[47])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[48])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[49])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[50])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[51])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[52])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[53])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[54])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[55])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[56])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[57])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[58])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[59])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[60])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[61])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[62])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[63])));
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[32])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[33])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[34])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[35])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[36])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[37])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[38])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[39])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[40])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[41])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[42])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[43])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[44])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[45])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[46])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[47])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[48])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[49])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[50])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[51])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[52])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[53])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[54])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[55])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[56])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[57])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[58])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[59])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[60])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[61])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[62])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[63]))
                                    : "r"(o_row_base + 32));
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                #if __CUDA_ARCH__ >= 1000
                                const float2 _scale2_21 = {acc_scale_c, acc_scale_c};
                                #pragma unroll
                                for (int _ls = 0; _ls < 32; _ls++)
                                    mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_0)[_ls], _scale2_21);
                                #else
                                #pragma unroll
                                for (int _ls = 0; _ls < 64; _ls++) {
                                    _tmem_load_0[_ls] = _tmem_load_0[_ls] * acc_scale_c;
                                }
                                #endif
                                asm volatile(
                                    "tcgen05.st.sync.aligned.16x32bx2.x64.b32"
                                    " [%0], 64, {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64};"
                                    :: "r"(o_row_base), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[15])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[16])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[17])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[18])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[19])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[20])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[21])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[22])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[23])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[24])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[25])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[26])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[27])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[28])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[29])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[30])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[31])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[32])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[33])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[34])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[35])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[36])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[37])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[38])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[39])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[40])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[41])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[42])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[43])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[44])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[45])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[46])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[47])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[48])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[49])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[50])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[51])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[52])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[53])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[54])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[55])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[56])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[57])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[58])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[59])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[60])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[61])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[62])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0[63])));
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                                if (rows_hi_1 != 0) {
                                    float acc16_0_1[16];
                                    #pragma unroll
#if CAKE_FMHA_DCP_PROGRAM == 5
                                    for (int q_4 = 0; q_4 < 4; q_4++) {
#else
                                    for (int q_3 = 0; q_3 < 4; q_3++) {
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
                                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
#if CAKE_FMHA_DCP_PROGRAM == 5
                                            : "=r"(*reinterpret_cast<uint32_t*>(&acc16_0_1[4 * q_4])), "=r"(*reinterpret_cast<uint32_t*>(&acc16_0_1[(4 * q_4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&acc16_0_1[(4 * q_4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&acc16_0_1[(4 * q_4) + 3]))
                                            : "r"(swap_acc_addr + (unsigned int)((48 + 4 * q_4) * 4)));
#else
                                            : "=r"(*reinterpret_cast<uint32_t*>(&acc16_0_1[4 * q_3])), "=r"(*reinterpret_cast<uint32_t*>(&acc16_0_1[(4 * q_3) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&acc16_0_1[(4 * q_3) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&acc16_0_1[(4 * q_3) + 3]))
                                            : "r"(swap_acc_addr + (unsigned int)((48 + 4 * q_3) * 4)));
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
                                    }
                                    float _tmem_load_3[16];
                                    tmem_ld_x16(&_tmem_load_3[0], oT_base_w_1 + 48);
                                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                                    #pragma unroll
                                    for (int j_38 = 0; j_38 < 16; j_38++) {
                                        _tmem_load_3[j_38] = _tmem_load_3[j_38] * acc16_0_1[j_38];
                                    }
                                    tmem_st_x16_f32(oT_base_w_1 + 48, _tmem_load_3);
                                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                                }
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                            }
                        }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                        int pb_w_1 = 0;
                        if (rows_hi_1 == 0) {
                            pb_w_1 = (int)(k_blk_o_1 & 1);
#else
                        if (n_1 >= 2) {
                            unsigned int k_o = pv_base + (unsigned int)n_1 - 2;
                            int o_st_o = (int)(k_o & 1);
                            int o_ph_o = (int)(k_o >> 1 & 1);
                            mbarrier_wait(o_ready_addr + (o_st_o) * 8, o_ph_o);
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                        if (n_1 >= 2) {
                            unsigned int k_o = pv_base + (unsigned int)n_1 - 2;
                            int o_st_o = (int)(k_o & 1);
                            int o_ph_o = (int)(k_o >> 1 & 1);
                            mbarrier_wait(o_ready_addr + (o_st_o) * 8, o_ph_o);
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                        }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                        if (pb_w_1 == 0) {
                            unsigned int words8_3[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sv_a_1[_lp*2 + 0], sv_a_1[_lp*2+1 + 0]));
                                words8_3[_lp] = *(uint32_t*)&_bf2;
                            }
                            int row_b_3 = k_w_1 * 64;
                            int swz_3 = (k_w_1 >> 1 & 3) << 4;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(swap_pT_addr + (unsigned int)(row_b_3 + (swz_3 ^ 32))), "r"(words8_3[0]), "r"(words8_3[1]), "r"(words8_3[2]), "r"(words8_3[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(swap_pT_addr + (unsigned int)(row_b_3 + (swz_3 ^ 48))), "r"(words8_3[4]), "r"(words8_3[5]), "r"(words8_3[6]), "r"(words8_3[7]) : "memory");
                        } else {
                            unsigned int words8_4[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sv_a_1[_lp*2 + 0], sv_a_1[_lp*2+1 + 0]));
                                words8_4[_lp] = *(uint32_t*)&_bf2;
                            }
                            int row_b_4 = k_w_1 * 64;
                            int swz_4 = (k_w_1 >> 1 & 3) << 4;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(swap_pT1_addr + (unsigned int)(row_b_4 + (swz_4 ^ 32))), "r"(words8_4[0]), "r"(words8_4[1]), "r"(words8_4[2]), "r"(words8_4[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(swap_pT1_addr + (unsigned int)(row_b_4 + (swz_4 ^ 48))), "r"(words8_4[4]), "r"(words8_4[5]), "r"(words8_4[6]), "r"(words8_4[7]) : "memory");
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                        __syncwarp();
                        if (lane == 0) {
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;"
                                :: "r"(p_full_addr + (sm_stage_c) * 8), "r"((uint32_t)(32)) : "memory");
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                        }
                        if (wg_tid_c == 0) {
                        }
                        if (rows_hi_1 != 0) {
                            unsigned int k_pa_1 = pv_base + (unsigned int)n_1;
                            mbarrier_wait(pa_done_addr + ((int)(k_pa_1 & 1)) * 8, (int)(k_pa_1 >> 1 & 1));
                            unsigned int words8_5[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sv_b_1[_lp*2 + 0], sv_b_1[_lp*2+1 + 0]));
                                words8_5[_lp] = *(uint32_t*)&_bf2;
                            }
                            int row_b_5 = k_w_1 * 64;
                            int swz_5 = (k_w_1 >> 1 & 3) << 4;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(swap_pT_addr + (unsigned int)(row_b_5 + (swz_5 ^ 32))), "r"(words8_5[0]), "r"(words8_5[1]), "r"(words8_5[2]), "r"(words8_5[3]) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(swap_pT_addr + (unsigned int)(row_b_5 + (swz_5 ^ 48))), "r"(words8_5[4]), "r"(words8_5[5]), "r"(words8_5[6]), "r"(words8_5[7]) : "memory");
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            __syncwarp();
                            if (lane == 0) {
                                asm volatile(
                                    "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;"
                                    :: "r"(p_full_b_addr + (sm_stage_c) * 8), "r"((uint32_t)(32)) : "memory");
                            }
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                        }
                        sm_stage_c += 1;
                        if (sm_stage_c == 2) { sm_stage_c = 0; sm_phase_c ^= 1; }
                        if (wg_tid_c == 0) {
                        }
                    }
                    if (cnt_c >= 2) {
                        unsigned int k_e2 = pv_base + (unsigned int)cnt_c - 2;
                        int o_st_e2 = (int)(k_e2 & 1);
                        int o_ph_e2 = (int)(k_e2 >> 1 & 1);
                        mbarrier_wait(o_ready_addr + (o_st_e2) * 8, o_ph_e2);
                    }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                    int st_off_pub = (int)st_stage_c * 64;
                    int rows_hi_p_1 = ((N_ROWS > 32) ? 1 : 0);
                    int b4_3 = lane >> 4 & 1;
                    int b3_4 = lane >> 3 & 1;
                    int b2_4 = lane >> 2 & 1;
                    int b1_4 = lane >> 1 & 1;
                    float w8_3[8];
                    #pragma unroll
                    for (int i_18 = 0; i_18 < 8; i_18++) {
                        float send8_6 = ((b4_3 != 0) ? psum_a_1[i_18] : psum_a_1[i_18 + 8]);
                        float keep8_6 = ((b4_3 != 0) ? psum_a_1[i_18 + 8] : psum_a_1[i_18]);
                        float _shfl_xor_30 = __shfl_xor_sync(0xFFFFFFFF, send8_6, 16);
                        w8_3[i_18] = keep8_6 + _shfl_xor_30;
                    }
                    float x4_3[4];
                    #pragma unroll
                    for (int i_19 = 0; i_19 < 4; i_19++) {
                        float send4_6 = ((b3_4 != 0) ? w8_3[i_19] : w8_3[i_19 + 4]);
                        float keep4_6 = ((b3_4 != 0) ? w8_3[i_19 + 4] : w8_3[i_19]);
                        float _shfl_xor_31 = __shfl_xor_sync(0xFFFFFFFF, send4_6, 8);
                        x4_3[i_19] = keep4_6 + _shfl_xor_31;
                    }
                    float y2_3[2];
                    #pragma unroll
                    for (int i_20 = 0; i_20 < 2; i_20++) {
                        float send2_6 = ((b2_4 != 0) ? x4_3[i_20] : x4_3[i_20 + 2]);
                        float keep2_6 = ((b2_4 != 0) ? x4_3[i_20 + 2] : x4_3[i_20]);
                        float _shfl_xor_32 = __shfl_xor_sync(0xFFFFFFFF, send2_6, 4);
                        y2_3[i_20] = keep2_6 + _shfl_xor_32;
                    }
                    float send1_3 = ((b1_4 != 0) ? y2_3[0] : y2_3[1]);
                    float keep1_3 = ((b1_4 != 0) ? y2_3[1] : y2_3[0]);
                    float _shfl_xor_33 = __shfl_xor_sync(0xFFFFFFFF, send1_3, 2);
                    float z_3 = keep1_3 + _shfl_xor_33;
                    float _shfl_xor_34 = __shfl_xor_sync(0xFFFFFFFF, z_3, 1);
                    z_3 = z_3 + _shfl_xor_34;
                    float zs_a_1 = z_3;
                    if ((lane & 1) == 0) {
                        swap_part[(32 + (lane >> 1 & 15)) * 4 + warp_in_wg_c] = zs_a_1;
                    }
                    if (rows_hi_p_1 != 0) {
                        int b4_0_3 = lane >> 4 & 1;
                        int b3_1_3 = lane >> 3 & 1;
                        int b2_2_3 = lane >> 2 & 1;
                        int b1_3_3 = lane >> 1 & 1;
                        float w8_4_3[8];
                        #pragma unroll
                        for (int i_21 = 0; i_21 < 8; i_21++) {
                            float send8_7 = ((b4_0_3 != 0) ? psum_b_1[i_21] : psum_b_1[i_21 + 8]);
                            float keep8_7 = ((b4_0_3 != 0) ? psum_b_1[i_21 + 8] : psum_b_1[i_21]);
                            float _shfl_xor_35 = __shfl_xor_sync(0xFFFFFFFF, send8_7, 16);
                            w8_4_3[i_21] = keep8_7 + _shfl_xor_35;
                        }
                        float x4_5_3[4];
                        #pragma unroll
                        for (int i_22 = 0; i_22 < 4; i_22++) {
                            float send4_7 = ((b3_1_3 != 0) ? w8_4_3[i_22] : w8_4_3[i_22 + 4]);
                            float keep4_7 = ((b3_1_3 != 0) ? w8_4_3[i_22 + 4] : w8_4_3[i_22]);
                            float _shfl_xor_36 = __shfl_xor_sync(0xFFFFFFFF, send4_7, 8);
                            x4_5_3[i_22] = keep4_7 + _shfl_xor_36;
                        }
                        float y2_6_3[2];
                        #pragma unroll
                        for (int i_23 = 0; i_23 < 2; i_23++) {
                            float send2_7 = ((b2_2_3 != 0) ? x4_5_3[i_23] : x4_5_3[i_23 + 2]);
                            float keep2_7 = ((b2_2_3 != 0) ? x4_5_3[i_23 + 2] : x4_5_3[i_23]);
                            float _shfl_xor_37 = __shfl_xor_sync(0xFFFFFFFF, send2_7, 4);
                            y2_6_3[i_23] = keep2_7 + _shfl_xor_37;
                        }
                        float send1_7_3 = ((b1_3_3 != 0) ? y2_6_3[0] : y2_6_3[1]);
                        float keep1_8_3 = ((b1_3_3 != 0) ? y2_6_3[1] : y2_6_3[0]);
                        float _shfl_xor_38 = __shfl_xor_sync(0xFFFFFFFF, send1_7_3, 2);
                        float z_9_3 = keep1_8_3 + _shfl_xor_38;
                        float _shfl_xor_39 = __shfl_xor_sync(0xFFFFFFFF, z_9_3, 1);
                        z_9_3 = z_9_3 + _shfl_xor_39;
                        float zs_b_1 = z_9_3;
                        if ((lane & 1) == 0) {
                            swap_part[(48 + (lane >> 1 & 15)) * 4 + warp_in_wg_c] = zs_b_1;
                        }
                    }
                    if (warp_in_wg_c == 0) {
                        if (lane == 0) {
                        }
                    }
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                    if (warp_in_wg_c == 0) {
                        if (lane == 0) {
                        }
                        if (lane < 16) {
                            float q4_1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&q4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q4_1[(0) + 3]))
                                : "r"(swap_part_addr + (unsigned int)((32 + lane) * 16)));
                            float sum_pa_1 = q4_1[0] + q4_1[1] + (q4_1[2] + q4_1[3]);
                            smem_sum[st_off_pub + 16 + lane] = sum_pa_1;
#if CAKE_FMHA_DCP_PROGRAM == 5
                            float _rcp_10 = approx_rcp(sum_pa_1);
                            swap_inv[st_off_pub + 16 + lane] = ((sum_pa_1 > 0.0f) ? _rcp_10 : 0.0f);
#else
                            float _rcp_9 = approx_rcp(sum_pa_1);
                            swap_inv[st_off_pub + 16 + lane] = ((sum_pa_1 > 0.0f) ? _rcp_9 : 0.0f);
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
                            if (rows_hi_p_1 != 0) {
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&q4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&q4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q4_1[(0) + 3]))
                                    : "r"(swap_part_addr + (unsigned int)((48 + lane) * 16)));
                                float sum_pb_1 = q4_1[0] + q4_1[1] + (q4_1[2] + q4_1[3]);
                                smem_sum[st_off_pub + 32 + 16 + lane] = sum_pb_1;
#if CAKE_FMHA_DCP_PROGRAM == 5
                                float _rcp_11 = approx_rcp(sum_pb_1);
                                swap_inv[st_off_pub + 32 + 16 + lane] = ((sum_pb_1 > 0.0f) ? _rcp_11 : 0.0f);
#else
                                float _rcp_10 = approx_rcp(sum_pb_1);
                                swap_inv[st_off_pub + 32 + 16 + lane] = ((sum_pb_1 > 0.0f) ? _rcp_10 : 0.0f);
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
                            }
                        }
                        if (lane == 0) {
                        }
                        float my_ref_a_1 = ref_a_1[0];
                        float my_ref_b_1 = ref_b_1[0];
                        #pragma unroll
                        for (int j_39 = 1; j_39 < 16; j_39++) {
                            my_ref_a_1 = ((lane == j_39) ? ref_a_1[j_39] : my_ref_a_1);
                            if (rows_hi_p_1 != 0) {
                                my_ref_b_1 = ((lane == j_39) ? ref_b_1[j_39] : my_ref_b_1);
                            }
                        }
                        if (lane < 16) {
                            smem_max[st_off_pub + 16 + lane] = my_ref_a_1;
                            if (rows_hi_p_1 != 0) {
                                smem_max[st_off_pub + 32 + 16 + lane] = my_ref_b_1;
                            }
                        }
                        if (lane == 0) {
                        }
                    }
                    if (wg_tid_c == 0) {
                    }
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                    unsigned int k_e = pv_base + (unsigned int)cnt_c - 1;
                    int o_st_e = (int)(k_e & 1);
                    int o_ph_e = (int)(k_e >> 1 & 1);
                    mbarrier_wait(o_ready_addr + (o_st_e) * 8, o_ph_e);
                    pv_base = pv_base + (unsigned int)cnt_c;
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                    if (wg_tid_c == 0) {
                    }
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                    asm volatile("tcgen05.fence::after_thread_sync;");
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                    int oT_base_c = taddr + 128 + (unsigned int)corr_row;
                    float _tmem_load_4[32];
#else
                    float _tmem_load_1[64];
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                    float _tmem_load_1[64];
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                    asm volatile(
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_4[0]), "=f"(_tmem_load_4[1]), "=f"(_tmem_load_4[2]), "=f"(_tmem_load_4[3]), "=f"(_tmem_load_4[4]), "=f"(_tmem_load_4[5]), "=f"(_tmem_load_4[6]), "=f"(_tmem_load_4[7]), "=f"(_tmem_load_4[8]), "=f"(_tmem_load_4[9]), "=f"(_tmem_load_4[10]), "=f"(_tmem_load_4[11]), "=f"(_tmem_load_4[12]), "=f"(_tmem_load_4[13]), "=f"(_tmem_load_4[14]), "=f"(_tmem_load_4[15]), "=f"(_tmem_load_4[16]), "=f"(_tmem_load_4[17]), "=f"(_tmem_load_4[18]), "=f"(_tmem_load_4[19]), "=f"(_tmem_load_4[20]), "=f"(_tmem_load_4[21]), "=f"(_tmem_load_4[22]), "=f"(_tmem_load_4[23]), "=f"(_tmem_load_4[24]), "=f"(_tmem_load_4[25]), "=f"(_tmem_load_4[26]), "=f"(_tmem_load_4[27]), "=f"(_tmem_load_4[28]), "=f"(_tmem_load_4[29]), "=f"(_tmem_load_4[30]), "=f"(_tmem_load_4[31])
                        : "r"(oT_base_c));
#else
                        "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[31]))
                        : "r"(o_row_base));
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                        "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[31]))
                        : "r"(o_row_base));
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                    int rows_hi_c = ((N_ROWS > 32) ? 1 : 0);
                    float oT_b[32];
                    if (rows_hi_c != 0) {
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(oT_b[0]), "=f"(oT_b[1]), "=f"(oT_b[2]), "=f"(oT_b[3]), "=f"(oT_b[4]), "=f"(oT_b[5]), "=f"(oT_b[6]), "=f"(oT_b[7]), "=f"(oT_b[8]), "=f"(oT_b[9]), "=f"(oT_b[10]), "=f"(oT_b[11]), "=f"(oT_b[12]), "=f"(oT_b[13]), "=f"(oT_b[14]), "=f"(oT_b[15]), "=f"(oT_b[16]), "=f"(oT_b[17]), "=f"(oT_b[18]), "=f"(oT_b[19]), "=f"(oT_b[20]), "=f"(oT_b[21]), "=f"(oT_b[22]), "=f"(oT_b[23]), "=f"(oT_b[24]), "=f"(oT_b[25]), "=f"(oT_b[26]), "=f"(oT_b[27]), "=f"(oT_b[28]), "=f"(oT_b[29]), "=f"(oT_b[30]), "=f"(oT_b[31])
                            : "r"(oT_base_c + 32));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                    }
                    float o_epi[64];
                    o_epi[0] = 0.0f;
                    o_epi[1] = 0.0f;
                    o_epi[2] = 0.0f;
                    o_epi[3] = 0.0f;
                    o_epi[4] = 0.0f;
                    o_epi[5] = 0.0f;
                    o_epi[6] = 0.0f;
                    o_epi[7] = 0.0f;
                    o_epi[8] = 0.0f;
                    o_epi[9] = 0.0f;
                    o_epi[10] = 0.0f;
                    o_epi[11] = 0.0f;
                    o_epi[12] = 0.0f;
                    o_epi[13] = 0.0f;
                    o_epi[14] = 0.0f;
                    o_epi[15] = 0.0f;
                    o_epi[16] = 0.0f;
                    o_epi[17] = 0.0f;
                    o_epi[18] = 0.0f;
                    o_epi[19] = 0.0f;
                    o_epi[20] = 0.0f;
                    o_epi[21] = 0.0f;
                    o_epi[22] = 0.0f;
                    o_epi[23] = 0.0f;
                    o_epi[24] = 0.0f;
                    o_epi[25] = 0.0f;
                    o_epi[26] = 0.0f;
                    o_epi[27] = 0.0f;
                    o_epi[28] = 0.0f;
                    o_epi[29] = 0.0f;
                    o_epi[30] = 0.0f;
                    o_epi[31] = 0.0f;
                    o_epi[32] = 0.0f;
                    o_epi[33] = 0.0f;
                    o_epi[34] = 0.0f;
                    o_epi[35] = 0.0f;
                    o_epi[36] = 0.0f;
                    o_epi[37] = 0.0f;
                    o_epi[38] = 0.0f;
                    o_epi[39] = 0.0f;
                    o_epi[40] = 0.0f;
                    o_epi[41] = 0.0f;
                    o_epi[42] = 0.0f;
                    o_epi[43] = 0.0f;
                    o_epi[44] = 0.0f;
                    o_epi[45] = 0.0f;
                    o_epi[46] = 0.0f;
                    o_epi[47] = 0.0f;
                    o_epi[48] = 0.0f;
                    o_epi[49] = 0.0f;
                    o_epi[50] = 0.0f;
                    o_epi[51] = 0.0f;
                    o_epi[52] = 0.0f;
                    o_epi[53] = 0.0f;
                    o_epi[54] = 0.0f;
                    o_epi[55] = 0.0f;
                    o_epi[56] = 0.0f;
                    o_epi[57] = 0.0f;
                    o_epi[58] = 0.0f;
                    o_epi[59] = 0.0f;
                    o_epi[60] = 0.0f;
                    o_epi[61] = 0.0f;
                    o_epi[62] = 0.0f;
                    o_epi[63] = 0.0f;
#else
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[32])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[33])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[34])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[35])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[36])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[37])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[38])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[39])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[40])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[41])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[42])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[43])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[44])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[45])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[46])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[47])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[48])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[49])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[50])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[51])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[52])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[53])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[54])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[55])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[56])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[57])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[58])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[59])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[60])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[61])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[62])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[63]))
                        : "r"(o_row_base + 32));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[32])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[33])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[34])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[35])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[36])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[37])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[38])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[39])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[40])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[41])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[42])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[43])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[44])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[45])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[46])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[47])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[48])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[49])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[50])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[51])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[52])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[53])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[54])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[55])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[56])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[57])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[58])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[59])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[60])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[61])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[62])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[63]))
                        : "r"(o_row_base + 32));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                    __syncwarp();
                    if (lane == 0) {
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;"
                            :: "r"(o_empty_addr), "r"((uint32_t)(32)) : "memory");
                    }
                    if (wg_tid_c == 0) {
                    }
                    mbarrier_wait(stats_full_addr + (st_stage_c) * 8, st_phase_c);
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                    if (wg_tid_c == 0) {
                    }
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                    int st_off = (int)st_stage_c * 64;
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                    asm volatile("barrier.sync 12, 128;" ::: "memory");
#else
                    float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, psum_c, 16);
                    float total_c = psum_c + _shfl_xor_3;
                    if (rows_live_c != 0) {
                        if (half_c == 0) {
                            smem_sum[st_off + my_row_c] = smem_sum[st_off + my_row_c] + total_c;
                        }
                    }
                    asm volatile("barrier.sync 10, 128;" ::: "memory");
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                    float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, psum_c, 16);
                    float total_c = psum_c + _shfl_xor_6;
                    if (rows_live_c != 0) {
                        if (half_c == 0) {
                            smem_sum[st_off + my_row_c] = smem_sum[st_off + my_row_c] + total_c;
                        }
                    }
                    asm volatile("barrier.sync 10, 128;" ::: "memory");
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                    if (wg_tid_c == 0) {
                    }
                    int publish_split = ((n_chunks_c > 1) ? 1 : 0);
                    if (publish_split == 0) {
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                        int d_c = wg_tid_c;
                        float inv32[32];
                        #pragma unroll
#if CAKE_FMHA_DCP_PROGRAM == 5
                        for (int q_5 = 0; q_5 < 8; q_5++) {
#else
                        for (int q_4 = 0; q_4 < 8; q_4++) {
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
#if CAKE_FMHA_DCP_PROGRAM == 5
                                : "=r"(*reinterpret_cast<uint32_t*>(&inv32[4 * q_5])), "=r"(*reinterpret_cast<uint32_t*>(&inv32[(4 * q_5) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inv32[(4 * q_5) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inv32[(4 * q_5) + 3]))
                                : "r"(swap_inv_addr + (unsigned int)((st_off + 4 * q_5) * 4)));
#else
                                : "=r"(*reinterpret_cast<uint32_t*>(&inv32[4 * q_4])), "=r"(*reinterpret_cast<uint32_t*>(&inv32[(4 * q_4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inv32[(4 * q_4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inv32[(4 * q_4) + 3]))
                                : "r"(swap_inv_addr + (unsigned int)((st_off + 4 * q_4) * 4)));
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
#else
                        float row_sum_c = 1.0f;
                        if (my_row_c < N_ROWS) {
                            row_sum_c = smem_sum[st_off + my_row_c];
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                        float row_sum_c = 1.0f;
                        if (my_row_c < N_ROWS) {
                            row_sum_c = smem_sum[st_off + my_row_c];
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                        }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                        int row_stride_e = num_q_heads * HEAD_DIM;
                        #pragma unroll
                        for (int r = 0; r < 32; r++) {
                            float inv_r = inv32[r];
                            float val_r = _tmem_load_4[r] * inv_r;
                            if (inv_r == 0.0f) {
                                val_r = 0.0f;
                            }
                            if (live_rows > r) {
                                *(reinterpret_cast<__nv_bfloat16*>(O_ptr + ((batch_c * q_len * num_q_heads + kv_head_c * 8) * HEAD_DIM + d_c + r / 8 * row_stride_e + r % 8 * HEAD_DIM)) + (0)) = __float2bfloat16_rn(val_r);
#else
#if CAKE_FMHA_DCP_PROGRAM == 4
                        float _rcp_8 = approx_rcp(row_sum_c);
                        float inv_c = ((row_sum_c > 0.0f) ? _rcp_8 : 0.0f);
#else
                        float _rcp_7 = approx_rcp(row_sum_c);
                        float inv_c = ((row_sum_c > 0.0f) ? _rcp_7 : 0.0f);
#endif  // CAKE_FMHA_DCP_PROGRAM == 4
                        if (row_sum_c <= 0.0f) {
                            #pragma unroll
                            for (int z_c = 0; z_c < 64; z_c++) {
                                _tmem_load_1[z_c] = 0.0f;
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                        float _rcp_19 = approx_rcp(row_sum_c);
                        float inv_c = ((row_sum_c > 0.0f) ? _rcp_19 : 0.0f);
                        if (row_sum_c <= 0.0f) {
                            #pragma unroll
                            for (int z_c = 0; z_c < 64; z_c++) {
                                _tmem_load_1[z_c] = 0.0f;
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                            }
                        }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                        if (rows_hi_c != 0) {
                            float inv32_0[32];
#else
                        if (my_row_c < live_rows) {
                            int j_c = my_row_c / 8;
                            int h_c = my_row_c % 8;
                            int q_head_c = kv_head_c * 8 + h_c;
                            int o_idx = ((batch_c * q_len + j_c) * num_q_heads + q_head_c) * HEAD_DIM + half_c * 64;
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                        if (my_row_c < live_rows) {
                            int j_c = my_row_c / 8;
                            int h_c = my_row_c % 8;
                            int q_head_c = kv_head_c * 8 + h_c;
                            int o_idx = ((batch_c * q_len + j_c) * num_q_heads + q_head_c) * HEAD_DIM + half_c * 64;
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                            #pragma unroll
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 5
                            for (int q_6 = 0; q_6 < 8; q_6++) {
#else
                            for (int q_5 = 0; q_5 < 8; q_5++) {
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
#if CAKE_FMHA_DCP_PROGRAM == 5
                                    : "=r"(*reinterpret_cast<uint32_t*>(&inv32_0[4 * q_6])), "=r"(*reinterpret_cast<uint32_t*>(&inv32_0[(4 * q_6) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inv32_0[(4 * q_6) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inv32_0[(4 * q_6) + 3]))
                                    : "r"(swap_inv_addr + (unsigned int)((st_off + 32 + 4 * q_6) * 4)));
#else
                                    : "=r"(*reinterpret_cast<uint32_t*>(&inv32_0[4 * q_5])), "=r"(*reinterpret_cast<uint32_t*>(&inv32_0[(4 * q_5) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inv32_0[(4 * q_5) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inv32_0[(4 * q_5) + 3]))
                                    : "r"(swap_inv_addr + (unsigned int)((st_off + 32 + 4 * q_5) * 4)));
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
                            }
                            int row_stride_e_1 = num_q_heads * HEAD_DIM;
                            #pragma unroll
                            for (int r_1 = 0; r_1 < 32; r_1++) {
                                float inv_r_1 = inv32_0[r_1];
                                float val_r_1 = oT_b[r_1] * inv_r_1;
                                if (inv_r_1 == 0.0f) {
                                    val_r_1 = 0.0f;
                                }
                                if (live_rows > 32 + r_1) {
                                    *(reinterpret_cast<__nv_bfloat16*>(O_ptr + ((batch_c * q_len * num_q_heads + kv_head_c * 8) * HEAD_DIM + d_c + (32 + r_1) / 8 * row_stride_e_1 + (32 + r_1) % 8 * HEAD_DIM)) + (0)) = __float2bfloat16_rn(val_r_1);
#else
                            for (int off = 0; off < 64; off += 8) {
                                {
                                    const float2 _prescale2_22 = {inv_c, inv_c};
                                    #if __CUDA_ARCH__ >= 1000
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 4; _ps++)
                                        mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_1[off])[_ps], _prescale2_22);
                                    #else
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 8; _ps++)
                                        _tmem_load_1[off + _ps] *= inv_c;
                                    #endif
                                    __nv_bfloat162 _pk[4];
                                    _pk[0] = __floats2bfloat162_rn(_tmem_load_1[off + 0], _tmem_load_1[off + 1]);
                                    _pk[1] = __floats2bfloat162_rn(_tmem_load_1[off + 2], _tmem_load_1[off + 3]);
                                    _pk[2] = __floats2bfloat162_rn(_tmem_load_1[off + 4], _tmem_load_1[off + 5]);
                                    _pk[3] = __floats2bfloat162_rn(_tmem_load_1[off + 6], _tmem_load_1[off + 7]);
                                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O_ptr + (o_idx + off)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                            for (int off = 0; off < 64; off += 8) {
                                {
                                    const float2 _prescale2_22 = {inv_c, inv_c};
                                    #if __CUDA_ARCH__ >= 1000
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 4; _ps++)
                                        mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_1[off])[_ps], _prescale2_22);
                                    #else
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 8; _ps++)
                                        _tmem_load_1[off + _ps] *= inv_c;
                                    #endif
                                    __nv_bfloat162 _pk[4];
                                    _pk[0] = __floats2bfloat162_rn(_tmem_load_1[off + 0], _tmem_load_1[off + 1]);
                                    _pk[1] = __floats2bfloat162_rn(_tmem_load_1[off + 2], _tmem_load_1[off + 3]);
                                    _pk[2] = __floats2bfloat162_rn(_tmem_load_1[off + 4], _tmem_load_1[off + 5]);
                                    _pk[3] = __floats2bfloat162_rn(_tmem_load_1[off + 6], _tmem_load_1[off + 7]);
                                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O_ptr + (o_idx + off)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                                }
                            }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                        }
                        int row_l = wg_tid_c;
                        int n_rows_c = 32 + 32 * rows_hi_c;
                        if (row_l < n_rows_c) {
                            if (row_l < live_rows) {
                                float sum_l = smem_sum[st_off + row_l];
                                float lse_l = -CAKE_FMHA_INF;
                                if (sum_l > 0.0f) {
#else
                            if (half_c == 0) {
                                float lse_c = -CAKE_FMHA_INF;
                                if (row_sum_c > 0.0f) {
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                            if (half_c == 0) {
                                float lse_c = -CAKE_FMHA_INF;
                                if (row_sum_c > 0.0f) {
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                                    float _log2_3;
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_3) : "f"(sum_l));
                                    lse_l = smem_max[st_off + row_l] * softmax_scale_log2 + _log2_3;
#else
                                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_3) : "f"(row_sum_c));
                                    lse_c = smem_max[st_off + my_row_c] * softmax_scale_log2 + _log2_3;
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_3) : "f"(row_sum_c));
                                    lse_c = smem_max[st_off + my_row_c] * softmax_scale_log2 + _log2_3;
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                                }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                                *(reinterpret_cast<float*>(LSE_ptr + ((batch_c * q_len + row_l / 8) * num_q_heads + kv_head_c * 8 + row_l % 8)) + (0)) = lse_l;
#else
                                *(reinterpret_cast<float*>(LSE_ptr + ((batch_c * q_len + j_c) * num_q_heads + q_head_c)) + (0)) = lse_c;
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                                *(reinterpret_cast<float*>(LSE_ptr + ((batch_c * q_len + j_c) * num_q_heads + q_head_c)) + (0)) = lse_c;
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                            }
                        }
                        __syncwarp();
                        if (elect_sync()) {
                            mbarrier_arrive(stats_empty_addr + (st_stage_c) * 8);
                        }
                    } else if (n_chunks_c == 2) {
                        if (my_row_c < N_ROWS) {
                            int p_base = my_slot * 8192 + my_row_c * 128 + half_c * 64;
                            #pragma unroll
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                            for (int off = 0; off < 64; off += 4) {
#else
                            for (int off_1 = 0; off_1 < 64; off_1 += 4) {
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                            for (int off_1 = 0; off_1 < 64; off_1 += 4) {
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                                {
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                                    float4 _v4 = make_float4(o_epi[off + 0], o_epi[off + 1], o_epi[off + 2], o_epi[off + 3]);
                                    *reinterpret_cast<float4*>((partial_o + (p_base + off)) + 0) = _v4;
#else
                                    float4 _v4 = make_float4(_tmem_load_1[off_1 + 0], _tmem_load_1[off_1 + 1], _tmem_load_1[off_1 + 2], _tmem_load_1[off_1 + 3]);
                                    *reinterpret_cast<float4*>((partial_o + (p_base + off_1)) + 0) = _v4;
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                                    float4 _v4 = make_float4(_tmem_load_1[off_1 + 0], _tmem_load_1[off_1 + 1], _tmem_load_1[off_1 + 2], _tmem_load_1[off_1 + 3]);
                                    *reinterpret_cast<float4*>((partial_o + (p_base + off_1)) + 0) = _v4;
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                                }
                            }
                        }
                        if (wg_tid_c < N_ROWS) {
                            *(reinterpret_cast<float*>(partial_stats + (my_slot * 128 + wg_tid_c)) + (0)) = smem_max[st_off + wg_tid_c];
                            *(reinterpret_cast<float*>(partial_stats + (my_slot * 128 + 64 + wg_tid_c)) + (0)) = smem_sum[st_off + wg_tid_c];
                        }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                        asm volatile("barrier.sync 12, 128;" ::: "memory");
#else
                        asm volatile("barrier.sync 10, 128;" ::: "memory");
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                        asm volatile("barrier.sync 10, 128;" ::: "memory");
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                        if (wg_tid_c == 0) {
                            {
                                asm volatile("fence.release.gpu;" ::: "memory");
                            }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                            unsigned int _atomic_old_2;
#else
                            unsigned int _atomic_old_4;
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                            asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                                : "=r"(_atomic_old_2) : "l"(&tile_counters[counter_idx_c * 4]), "r"(static_cast<uint32_t>(1)) : "memory");
                            unsigned int arrived_old = _atomic_old_2;
#else
                                : "=r"(_atomic_old_4) : "l"(&tile_counters[counter_idx_c * 4]), "r"(static_cast<uint32_t>(1)) : "memory");
                            unsigned int arrived_old = _atomic_old_4;
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                            smem_corr_flag[0] = arrived_old + 1;
                        }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                        asm volatile("barrier.sync 12, 128;" ::: "memory");
#else
                        asm volatile("barrier.sync 10, 128;" ::: "memory");
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                        asm volatile("barrier.sync 10, 128;" ::: "memory");
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                        unsigned int arrived_c = smem_corr_flag[0];
                        if ((int)arrived_c == n_chunks_c) {
                            if (wg_tid_c == 0) {
                            }
                            {
                                asm volatile("fence.acquire.gpu;" ::: "memory");
                            }
                            int other_slot = slot_tile_base_c + (1 - chunk_c) * num_kv_heads;
                            float w_s_c = 0.0f;
                            float w_o_c = 0.0f;
                            float lse_i = -CAKE_FMHA_INF;
                            if (my_row_c < N_ROWS) {
                                float m_o = partial_stats[other_slot * 128 + my_row_c];
                                float l_o = partial_stats[other_slot * 128 + 64 + my_row_c];
                                float m_s = smem_max[st_off + my_row_c];
                                float l_s = smem_sum[st_off + my_row_c];
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 5
                                float _max_54 = max_noftz(m_s, m_o);
                                float m_row_i = _max_54;
#else
                                float _max_52 = max_noftz(m_s, m_o);
                                float m_row_i = _max_52;
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
                                float _exp2_10 = approx_exp2((m_s - m_row_i) * softmax_scale_log2);
                                float w_s = _exp2_10;
                                float _exp2_11 = approx_exp2((m_o - m_row_i) * softmax_scale_log2);
                                float w_o = _exp2_11;
                                float _fma_10 = __fmaf_rn(w_s, l_s, w_o * l_o);
                                float den_i = _fma_10;
#if CAKE_FMHA_DCP_PROGRAM == 5
                                float _rcp_12 = approx_rcp(den_i);
                                float inv_i = ((den_i > 0.0f) ? _rcp_12 : 0.0f);
#else
                                float _rcp_11 = approx_rcp(den_i);
                                float inv_i = ((den_i > 0.0f) ? _rcp_11 : 0.0f);
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
#else
#if CAKE_FMHA_DCP_PROGRAM == 4
                                float _max_16 = max_noftz(m_s, m_o);
                                float m_row_i = _max_16;
#else
                                float _max_14 = max_noftz(m_s, m_o);
                                float m_row_i = _max_14;
#endif  // CAKE_FMHA_DCP_PROGRAM == 4
                                float _exp2_8 = approx_exp2((m_s - m_row_i) * softmax_scale_log2);
                                float w_s = _exp2_8;
                                float _exp2_9 = approx_exp2((m_o - m_row_i) * softmax_scale_log2);
                                float w_o = _exp2_9;
                                float _fma_6 = __fmaf_rn(w_s, l_s, w_o * l_o);
                                float den_i = _fma_6;
#if CAKE_FMHA_DCP_PROGRAM == 4
                                float _rcp_9 = approx_rcp(den_i);
                                float inv_i = ((den_i > 0.0f) ? _rcp_9 : 0.0f);
#else
                                float _rcp_8 = approx_rcp(den_i);
                                float inv_i = ((den_i > 0.0f) ? _rcp_8 : 0.0f);
#endif  // CAKE_FMHA_DCP_PROGRAM == 4
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                                float _max_26 = max_noftz(m_s, m_o);
                                float m_row_i = _max_26;
                                float _exp2_8 = approx_exp2((m_s - m_row_i) * softmax_scale_log2);
                                float w_s = _exp2_8;
                                float _exp2_9 = approx_exp2((m_o - m_row_i) * softmax_scale_log2);
                                float w_o = _exp2_9;
                                float _fma_6 = __fmaf_rn(w_s, l_s, w_o * l_o);
                                float den_i = _fma_6;
                                float _rcp_20 = approx_rcp(den_i);
                                float inv_i = ((den_i > 0.0f) ? _rcp_20 : 0.0f);
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                                w_s_c = w_s * inv_i;
                                w_o_c = w_o * inv_i;
                                if (den_i > 0.0f) {
                                    float _log2_4;
                                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_4) : "f"(den_i));
                                    lse_i = m_row_i * softmax_scale_log2 + _log2_4;
                                }
                            }
                            int oth_base = other_slot * 8192 + my_row_c * 128 + half_c * 64;
                            if (my_row_c < live_rows) {
                                #pragma unroll
                                for (int c0 = 0; c0 < 64; c0 += 4) {
                                    float _vec_load_1[4];
                                    {
                                        float4 _v4 = *reinterpret_cast<const float4*>(partial_o + (oth_base + c0) + 0);
                                        _vec_load_1[0 + 0] = _v4.x;
                                        _vec_load_1[0 + 1] = _v4.y;
                                        _vec_load_1[0 + 2] = _v4.z;
                                        _vec_load_1[0 + 3] = _v4.w;
                                    }
                                    #pragma unroll
                                    for (int c = 0; c < 4; c++) {
                                        {
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                                            float _fma_14 = __fmaf_rn(o_epi[c0 + c], w_s_c, _vec_load_1[c] * w_o_c);
                                            o_epi[c0 + c] = _fma_14;
#else
                                            float _fma_10 = __fmaf_rn(_tmem_load_1[c0 + c], w_s_c, _vec_load_1[c] * w_o_c);
                                            _tmem_load_1[c0 + c] = _fma_10;
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                                            float _fma_10 = __fmaf_rn(_tmem_load_1[c0 + c], w_s_c, _vec_load_1[c] * w_o_c);
                                            _tmem_load_1[c0 + c] = _fma_10;
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                                        }
                                    }
                                }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                                int j_c = my_row_c / 8;
                                int h_c = my_row_c % 8;
                                int q_head_c = kv_head_c * 8 + h_c;
                                int o_idx = ((batch_c * q_len + j_c) * num_q_heads + q_head_c) * HEAD_DIM + half_c * 64;
#else
                                int j_c_1 = my_row_c / 8;
                                int h_c_1 = my_row_c % 8;
                                int q_head_c_1 = kv_head_c * 8 + h_c_1;
                                int o_idx_1 = ((batch_c * q_len + j_c_1) * num_q_heads + q_head_c_1) * HEAD_DIM + half_c * 64;
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                                int j_c_1 = my_row_c / 8;
                                int h_c_1 = my_row_c % 8;
                                int q_head_c_1 = kv_head_c * 8 + h_c_1;
                                int o_idx_1 = ((batch_c * q_len + j_c_1) * num_q_heads + q_head_c_1) * HEAD_DIM + half_c * 64;
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                                #pragma unroll
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                                for (int off_1 = 0; off_1 < 64; off_1 += 8) {
#else
                                for (int off_2 = 0; off_2 < 64; off_2 += 8) {
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                                for (int off_2 = 0; off_2 < 64; off_2 += 8) {
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                                    {
                                        __nv_bfloat162 _pk[4];
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                                        _pk[0] = __floats2bfloat162_rn(o_epi[off_1 + 0], o_epi[off_1 + 1]);
                                        _pk[1] = __floats2bfloat162_rn(o_epi[off_1 + 2], o_epi[off_1 + 3]);
                                        _pk[2] = __floats2bfloat162_rn(o_epi[off_1 + 4], o_epi[off_1 + 5]);
                                        _pk[3] = __floats2bfloat162_rn(o_epi[off_1 + 6], o_epi[off_1 + 7]);
                                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O_ptr + (o_idx + off_1)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
#else
                                        _pk[0] = __floats2bfloat162_rn(_tmem_load_1[off_2 + 0], _tmem_load_1[off_2 + 1]);
                                        _pk[1] = __floats2bfloat162_rn(_tmem_load_1[off_2 + 2], _tmem_load_1[off_2 + 3]);
                                        _pk[2] = __floats2bfloat162_rn(_tmem_load_1[off_2 + 4], _tmem_load_1[off_2 + 5]);
                                        _pk[3] = __floats2bfloat162_rn(_tmem_load_1[off_2 + 6], _tmem_load_1[off_2 + 7]);
                                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O_ptr + (o_idx_1 + off_2)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                                        _pk[0] = __floats2bfloat162_rn(_tmem_load_1[off_2 + 0], _tmem_load_1[off_2 + 1]);
                                        _pk[1] = __floats2bfloat162_rn(_tmem_load_1[off_2 + 2], _tmem_load_1[off_2 + 3]);
                                        _pk[2] = __floats2bfloat162_rn(_tmem_load_1[off_2 + 4], _tmem_load_1[off_2 + 5]);
                                        _pk[3] = __floats2bfloat162_rn(_tmem_load_1[off_2 + 6], _tmem_load_1[off_2 + 7]);
                                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O_ptr + (o_idx_1 + off_2)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                                    }
                                }
                                if (half_c == 0) {
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                                    *(reinterpret_cast<float*>(LSE_ptr + ((batch_c * q_len + j_c) * num_q_heads + q_head_c)) + (0)) = lse_i;
#else
                                    *(reinterpret_cast<float*>(LSE_ptr + ((batch_c * q_len + j_c_1) * num_q_heads + q_head_c_1)) + (0)) = lse_i;
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                                    *(reinterpret_cast<float*>(LSE_ptr + ((batch_c * q_len + j_c_1) * num_q_heads + q_head_c_1)) + (0)) = lse_i;
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                                }
                            }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                            asm volatile("barrier.sync 12, 128;" ::: "memory");
#else
                            asm volatile("barrier.sync 10, 128;" ::: "memory");
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                            asm volatile("barrier.sync 10, 128;" ::: "memory");
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                            if (elect_sync()) {
                                mbarrier_arrive(stats_empty_addr + (st_stage_c) * 8);
                            }
                            n_reduces_seen = n_reduces_seen + 1;
                            if (wg_tid_c == 0) {
                                *(reinterpret_cast<unsigned int*>(tile_counters + (counter_idx_c * 4)) + (0)) = 0;
                            }
                        } else {
                            __syncwarp();
                            if (elect_sync()) {
                                mbarrier_arrive(stats_empty_addr + (st_stage_c) * 8);
                            }
                        }
                    } else {
                        if (my_row_c < N_ROWS) {
                            int p_base_1 = my_slot * 8192 + my_row_c * 128 + half_c * 64;
                            #pragma unroll
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                            for (int off_2 = 0; off_2 < 64; off_2 += 4) {
#else
                            for (int off_3 = 0; off_3 < 64; off_3 += 4) {
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                            for (int off_3 = 0; off_3 < 64; off_3 += 4) {
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                                {
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                                    float4 _v4 = make_float4(o_epi[off_2 + 0], o_epi[off_2 + 1], o_epi[off_2 + 2], o_epi[off_2 + 3]);
                                    *reinterpret_cast<float4*>((partial_o + (p_base_1 + off_2)) + 0) = _v4;
#else
                                    float4 _v4 = make_float4(_tmem_load_1[off_3 + 0], _tmem_load_1[off_3 + 1], _tmem_load_1[off_3 + 2], _tmem_load_1[off_3 + 3]);
                                    *reinterpret_cast<float4*>((partial_o + (p_base_1 + off_3)) + 0) = _v4;
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                                    float4 _v4 = make_float4(_tmem_load_1[off_3 + 0], _tmem_load_1[off_3 + 1], _tmem_load_1[off_3 + 2], _tmem_load_1[off_3 + 3]);
                                    *reinterpret_cast<float4*>((partial_o + (p_base_1 + off_3)) + 0) = _v4;
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                                }
                            }
                        }
                        if (wg_tid_c < N_ROWS) {
                            *(reinterpret_cast<float*>(partial_stats + (my_slot * 128 + wg_tid_c)) + (0)) = smem_max[st_off + wg_tid_c];
                            *(reinterpret_cast<float*>(partial_stats + (my_slot * 128 + 64 + wg_tid_c)) + (0)) = smem_sum[st_off + wg_tid_c];
                        }
                        __syncwarp();
                        if (elect_sync()) {
                            mbarrier_arrive(stats_empty_addr + (st_stage_c) * 8);
                        }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                        asm volatile("barrier.sync 12, 128;" ::: "memory");
#else
                        asm volatile("barrier.sync 10, 128;" ::: "memory");
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                        asm volatile("barrier.sync 10, 128;" ::: "memory");
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                        if (wg_tid_c == 0) {
                            {
                                asm volatile("fence.release.gpu;" ::: "memory");
                            }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                            unsigned int _atomic_old_3;
#else
                            unsigned int _atomic_old_5;
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                            asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                                : "=r"(_atomic_old_3) : "l"(&tile_counters[counter_idx_c * 4]), "r"(static_cast<uint32_t>(1)) : "memory");
                            unsigned int arrived_old_1 = _atomic_old_3;
#else
                                : "=r"(_atomic_old_5) : "l"(&tile_counters[counter_idx_c * 4]), "r"(static_cast<uint32_t>(1)) : "memory");
                            unsigned int arrived_old_1 = _atomic_old_5;
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                            if ((int)arrived_old_1 + 1 == n_chunks_c) {
                            }
                        }
                    }
                    st_stage_c += 1;
                    if (st_stage_c == 2) { st_stage_c = 0; st_phase_c ^= 1; }
                    if (wg_tid_c == 0) {
                        if (_tile_iter_c == 0) {
                        }
                    }
                } else {
                    if (wg_tid_c == 0) {
                    }
                    {
                        asm volatile("fence.acquire.gpu;" ::: "memory");
                    }
                    int r_row_1 = wg_tid_c / 4;
                    if (r_row_1 < live_rows) {
                        int stats_stride_r_1 = num_kv_heads * 128;
                        int o_stride_r_1 = num_kv_heads * 8192;
                        int stats_row_r_1 = slot_tile_base_c * 128 + r_row_1;
                        int f_r_1 = 32 >> block_end_c;
                        int d0_r_1 = block_begin_c * (4 * f_r_1) + wg_tid_c % 4 * f_r_1;
                        int o_row_r_1 = slot_tile_base_c * 8192 + r_row_1 * 128 + d0_r_1;
                        int j_r_1 = r_row_1 / 8;
                        int h_r_1 = r_row_1 % 8;
                        int q_head_r_1 = kv_head_c * 8 + h_r_1;
                        int o_idx_r_1 = ((batch_c * q_len + j_r_1) * num_q_heads + q_head_r_1) * HEAD_DIM + d0_r_1;
                        int store_lse_r_1 = 0;
                        if (wg_tid_c % 4 == 0 && block_begin_c == 0) {
                            store_lse_r_1 = 1;
                        }
                        int lse_idx_r_1 = (batch_c * q_len + j_r_1) * num_q_heads + q_head_r_1;
                        {
                            int n_groups_r_1 = 8 >> block_end_c;
                            float acc_f_1[4];
                            float out4_1[4];
                            int n_pad_r_1 = (n_chunks_c + 7) / 8 * 8;
                            #pragma unroll 1
                            for (int g_r_1 = 0; g_r_1 < n_groups_r_1; g_r_1++) {
                                acc_f_1[0] = 0.0f;
                                acc_f_1[1] = 0.0f;
                                acc_f_1[2] = 0.0f;
                                acc_f_1[3] = 0.0f;
                                float m_f_1 = -1e+30f;
                                float l_f_1 = 0.0f;
                                int o_col_r_1 = o_row_r_1 + g_r_1 * 4;
                                #pragma unroll 8
                                for (int c_m_1 = 0; c_m_1 < n_pad_r_1; c_m_1++) {
                                    int c_c_1 = c_m_1;
                                    if (n_chunks_c <= c_m_1) {
                                        c_c_1 = n_chunks_c - 1;
                                    }
                                    float m_k_1 = partial_stats[stats_row_r_1 + c_c_1 * stats_stride_r_1];
                                    float l_k_1 = partial_stats[stats_row_r_1 + 64 + c_c_1 * stats_stride_r_1];
                                    if (n_chunks_c <= c_m_1) {
                                        m_k_1 = -1e+30f;
                                        l_k_1 = 0.0f;
                                    }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 5
                                    float _max_59 = max_noftz(m_f_1, m_k_1);
                                    float m_new_1 = _max_59;
#else
                                    float _max_57 = max_noftz(m_f_1, m_k_1);
                                    float m_new_1 = _max_57;
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
                                    float _exp2_16 = approx_exp2((m_f_1 - m_new_1) * softmax_scale_log2);
                                    float a_k_1 = _exp2_16;
                                    float _exp2_17 = approx_exp2((m_k_1 - m_new_1) * softmax_scale_log2);
                                    float b_k_1 = _exp2_17;
                                    float _fma_19 = __fmaf_rn(l_k_1, b_k_1, l_f_1 * a_k_1);
                                    l_f_1 = _fma_19;
#else
#if CAKE_FMHA_DCP_PROGRAM == 4
                                    float _max_21 = max_noftz(m_f_1, m_k_1);
                                    float m_new_1 = _max_21;
#else
                                    float _max_19 = max_noftz(m_f_1, m_k_1);
                                    float m_new_1 = _max_19;
#endif  // CAKE_FMHA_DCP_PROGRAM == 4
                                    float _exp2_14 = approx_exp2((m_f_1 - m_new_1) * softmax_scale_log2);
                                    float a_k_1 = _exp2_14;
                                    float _exp2_15 = approx_exp2((m_k_1 - m_new_1) * softmax_scale_log2);
                                    float b_k_1 = _exp2_15;
                                    float _fma_15 = __fmaf_rn(l_k_1, b_k_1, l_f_1 * a_k_1);
                                    l_f_1 = _fma_15;
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                                    float _max_31 = max_noftz(m_f_1, m_k_1);
                                    float m_new_1 = _max_31;
                                    float _exp2_14 = approx_exp2((m_f_1 - m_new_1) * softmax_scale_log2);
                                    float a_k_1 = _exp2_14;
                                    float _exp2_15 = approx_exp2((m_k_1 - m_new_1) * softmax_scale_log2);
                                    float b_k_1 = _exp2_15;
                                    float _fma_15 = __fmaf_rn(l_k_1, b_k_1, l_f_1 * a_k_1);
                                    l_f_1 = _fma_15;
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                                    float _vec_load_2[4];
                                    {
                                        float4 _v4 = *reinterpret_cast<const float4*>(partial_o + (o_col_r_1 + c_c_1 * o_stride_r_1) + 0);
                                        _vec_load_2[0 + 0] = _v4.x;
                                        _vec_load_2[0 + 1] = _v4.y;
                                        _vec_load_2[0 + 2] = _v4.z;
                                        _vec_load_2[0 + 3] = _v4.w;
                                    }
                                    #pragma unroll
                                    for (int k_1 = 0; k_1 < 4; k_1++) {
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                                        float _fma_20 = __fmaf_rn(_vec_load_2[k_1], b_k_1, acc_f_1[k_1] * a_k_1);
                                        acc_f_1[k_1] = _fma_20;
#else
                                        float _fma_16 = __fmaf_rn(_vec_load_2[k_1], b_k_1, acc_f_1[k_1] * a_k_1);
                                        acc_f_1[k_1] = _fma_16;
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                                        float _fma_16 = __fmaf_rn(_vec_load_2[k_1], b_k_1, acc_f_1[k_1] * a_k_1);
                                        acc_f_1[k_1] = _fma_16;
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                                    }
                                    m_f_1 = m_new_1;
                                }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 5
                                float _rcp_15 = approx_rcp(l_f_1);
                                float inv_f_1 = ((l_f_1 > 0.0f) ? _rcp_15 : 0.0f);
#else
                                float _rcp_14 = approx_rcp(l_f_1);
                                float inv_f_1 = ((l_f_1 > 0.0f) ? _rcp_14 : 0.0f);
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
#else
#if CAKE_FMHA_DCP_PROGRAM == 4
                                float _rcp_12 = approx_rcp(l_f_1);
                                float inv_f_1 = ((l_f_1 > 0.0f) ? _rcp_12 : 0.0f);
#else
                                float _rcp_11 = approx_rcp(l_f_1);
                                float inv_f_1 = ((l_f_1 > 0.0f) ? _rcp_11 : 0.0f);
#endif  // CAKE_FMHA_DCP_PROGRAM == 4
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                                float _rcp_23 = approx_rcp(l_f_1);
                                float inv_f_1 = ((l_f_1 > 0.0f) ? _rcp_23 : 0.0f);
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                                #pragma unroll
                                for (int k4_1 = 0; k4_1 < 4; k4_1++) {
                                    out4_1[k4_1] = acc_f_1[k4_1] * inv_f_1;
                                }
                                {
                                    uint2 _pk2;
                                    __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
                                    _pk[0] = __floats2bfloat162_rn(out4_1[0 + 0], out4_1[0 + 1]);
                                    _pk[1] = __floats2bfloat162_rn(out4_1[0 + 2], out4_1[0 + 3]);
                                    *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(O_ptr + (o_idx_r_1 + g_r_1 * 4)))[0]) = _pk2;
                                }
                                if (store_lse_r_1 != 0) {
                                    if (g_r_1 == 0) {
                                        float lse_r_1 = -CAKE_FMHA_INF;
                                        if (l_f_1 > 0.0f) {
                                            float _log2_7;
                                            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_7) : "f"(l_f_1));
                                            lse_r_1 = m_f_1 * softmax_scale_log2 + _log2_7;
                                        }
                                        *(reinterpret_cast<float*>(LSE_ptr + lse_idx_r_1) + (0)) = lse_r_1;
                                    }
                                }
                            }
                        }
                    }
                    n_reduces_seen = n_reduces_seen + 1;
                    if (wg_tid_c == 0) {
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_c) * 8, _phase_work_full_1);
                unsigned int base_0_1 = work_stage_c * 16;
                unsigned int valid_1_1 = work_token_words[base_0_1];
                unsigned int kind_2_1 = work_token_words[base_0_1 + 1];
                unsigned int batch_3_1 = work_token_words[base_0_1 + 2];
                unsigned int kv_head_4_1 = work_token_words[base_0_1 + 3];
                unsigned int block_begin_5_1 = work_token_words[base_0_1 + 4];
                unsigned int block_end_6_1 = work_token_words[base_0_1 + 5];
                unsigned int seqlen_7_1 = work_token_words[base_0_1 + 6];
                unsigned int n_chunks_8_1 = work_token_words[base_0_1 + 7];
                unsigned int slot_tile_base_9_1 = work_token_words[base_0_1 + 8];
                unsigned int counter_idx_10_1 = work_token_words[base_0_1 + 9];
                unsigned int chunk_11_1 = work_token_words[base_0_1 + 10];
                unsigned int phase_12_1 = work_token_words[base_0_1 + 11];
                mbarrier_arrive(work_empty_addr + (work_stage_c) * 8);
                work_stage_c += 1;
                if (work_stage_c == 4) { work_stage_c = 0; _phase_work_full_1 ^= 1; }
                valid_c = valid_1_1;
                kind_c = (int)kind_2_1;
                batch_c = (int)batch_3_1;
                kv_head_c = (int)kv_head_4_1;
                block_begin_c = (int)block_begin_5_1;
                block_end_c = (int)block_end_6_1;
                seqlen_c = (int)seqlen_7_1;
                n_chunks_c = (int)n_chunks_8_1;
                slot_tile_base_c = (int)slot_tile_base_9_1;
                counter_idx_c = (int)counter_idx_10_1;
                chunk_c = (int)chunk_11_1;
                phase_c = (int)phase_12_1;
            }
            if (wg_tid_c == 0) {
            }
            mbarrier_arrive(tmem_dealloc_addr);
        }
    // ---- Role: mma_warp ----
    } else if (warp == 8) {
        { // mma_warp_main
            if (lane == 0) {
            }
            unsigned int work_stage_m = 0;
            unsigned int q_cons_stage = 0;
            unsigned int q_cons_phase = 0;
            unsigned int k_cons_stage = 0;
            unsigned int k_cons_phase = 0;
            unsigned int v_cons_stage = 0;
            unsigned int v_cons_phase = 0;
            int s_buf = 0;
            unsigned int pf_stage = 0;
            unsigned int pf_phase = 0;
            unsigned int pv_idx_m = 0;
            unsigned int _phase_work_full_2 = 0;
            mbarrier_wait(work_full_addr + (work_stage_m) * 8, _phase_work_full_2);
            unsigned int base_2 = work_stage_m * 16;
            unsigned int valid_3 = work_token_words[base_2];
            unsigned int kind_3 = work_token_words[base_2 + 1];
            unsigned int batch_2 = work_token_words[base_2 + 2];
            unsigned int kv_head_2 = work_token_words[base_2 + 3];
            unsigned int block_begin_2 = work_token_words[base_2 + 4];
            unsigned int block_end_2 = work_token_words[base_2 + 5];
            unsigned int seqlen_2 = work_token_words[base_2 + 6];
            unsigned int n_chunks_2 = work_token_words[base_2 + 7];
            unsigned int slot_tile_base_2 = work_token_words[base_2 + 8];
            unsigned int counter_idx_2 = work_token_words[base_2 + 9];
            unsigned int chunk_2 = work_token_words[base_2 + 10];
            unsigned int phase_2 = work_token_words[base_2 + 11];
            mbarrier_arrive(work_empty_addr + (work_stage_m) * 8);
            work_stage_m += 1;
            if (work_stage_m == 4) { work_stage_m = 0; _phase_work_full_2 ^= 1; }
            unsigned int valid_m = valid_3;
            int kind_m = (int)kind_3;
            int block_begin_m = (int)block_begin_2;
            int block_end_m = (int)block_end_2;
            unsigned int _phase_o_empty_0 = 1;
            #pragma unroll 1
            for (unsigned int _tile_iter_m = 0; _tile_iter_m < max_items; _tile_iter_m++) {
                if (valid_m == 0) {
                    break;
                }
                if (kind_m == 0) {
                    int cnt_m = block_end_m - block_begin_m;
                    mbarrier_wait(q_full_addr + (q_cons_stage) * 8, q_cons_phase);
                    mbarrier_wait(k_full_addr + (k_cons_stage) * 8, k_cons_phase);
                    if (_tile_iter_m == 0) {
                        if (lane == 0) {
                        }
                    }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                    int _mma_a_lo_0 = make_warp_uniform((((smem_k_addr) >> 4) & 0x3FFF) + (k_cons_stage) * 2048);
                    int _mma_b_lo_0 = make_warp_uniform((((smem_q32_addr) >> 4) & 0x3FFF) + (0) * 512);
#else
                    int _mma_a_lo_0 = make_warp_uniform((((smem_qt_addr) >> 4) & 0x3FFF) + (q_cons_stage) * 1024);
                    int _mma_b_lo_0 = make_warp_uniform((((smem_k_addr) >> 4) & 0x3FFF) + (k_cons_stage) * 2048);
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                    int _mma_a_lo_0 = make_warp_uniform((((smem_qt_addr) >> 4) & 0x3FFF) + (q_cons_stage) * 1024);
                    int _mma_b_lo_0 = make_warp_uniform((((smem_k_addr) >> 4) & 0x3FFF) + (k_cons_stage) * 2048);
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                    {
                        uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
                        uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                        if (elect_sync()) {
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                            tcgen05_mma_f16((tmem_tmem_sT + (s_buf * 64)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134743184, 0);
#else
                            tcgen05_mma_f16((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 69207184, 0);
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                            tcgen05_mma_f16((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 69207184, 0);
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                            tcgen05_mma_f16((tmem_tmem_sT + (s_buf * 64)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134743184, 1);
#else
                            tcgen05_mma_f16((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 69207184, 1);
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                            tcgen05_mma_f16((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 69207184, 1);
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                            tcgen05_mma_f16((tmem_tmem_sT + (s_buf * 64)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134743184, 1);
#else
                            tcgen05_mma_f16((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 69207184, 1);
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                            tcgen05_mma_f16((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 69207184, 1);
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                            tcgen05_mma_f16((tmem_tmem_sT + (s_buf * 64)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134743184, 1);
#else
                            tcgen05_mma_f16((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 69207184, 1);
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                            tcgen05_mma_f16((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 69207184, 1);
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                        }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 1018U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 250U);
#else
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 506U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 1018U);
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 506U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 1018U);
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                        if (elect_sync()) {
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                            tcgen05_mma_f16((tmem_tmem_sT + (s_buf * 64)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134743184, 1);
#else
                            tcgen05_mma_f16((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 69207184, 1);
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                            tcgen05_mma_f16((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 69207184, 1);
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                            tcgen05_mma_f16((tmem_tmem_sT + (s_buf * 64)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134743184, 1);
#else
                            tcgen05_mma_f16((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 69207184, 1);
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                            tcgen05_mma_f16((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 69207184, 1);
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                            tcgen05_mma_f16((tmem_tmem_sT + (s_buf * 64)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134743184, 1);
#else
                            tcgen05_mma_f16((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 69207184, 1);
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                            tcgen05_mma_f16((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 69207184, 1);
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                            tcgen05_mma_f16((tmem_tmem_sT + (s_buf * 64)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134743184, 1);
                        }
                    }
                    int rows_hi_qk = ((N_ROWS > 32) ? 1 : 0);
                    if (rows_hi_qk != 0) {
                        int _mma_a_lo_1 = make_warp_uniform((((smem_k_addr) >> 4) & 0x3FFF) + (k_cons_stage) * 2048);
                        int _mma_b_lo_1 = make_warp_uniform((((smem_q32_addr) >> 4) & 0x3FFF) + (1) * 512);
                        {
                            uint64_t _mma_ss_a_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_1);
                            uint64_t _mma_ss_b_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_1);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_sT + (s_buf * 64 + 32)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134743184, 0);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_sT + (s_buf * 64 + 32)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134743184, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_sT + (s_buf * 64 + 32)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134743184, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_sT + (s_buf * 64 + 32)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134743184, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_1, 1018U);
                            incr_smem_desc_lo(_mma_ss_b_desc_1, 250U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_sT + (s_buf * 64 + 32)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134743184, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_sT + (s_buf * 64 + 32)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134743184, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_sT + (s_buf * 64 + 32)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134743184, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_tmem_sT + (s_buf * 64 + 32)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134743184, 1);
                            }
#else
                            tcgen05_mma_f16((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 69207184, 1);
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                            tcgen05_mma_f16((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 69207184, 1);
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                        }
                    }
                    elect_commit(s_full_addr + (s_buf) * 8);
                    elect_commit(k_empty_addr + (k_cons_stage) * 8);
                    k_cons_stage += 1;
                    if (k_cons_stage == 3) { k_cons_stage = 0; k_cons_phase ^= 1; }
                    s_buf = s_buf ^ 1;
                    if (cnt_m == 1) {
                        elect_commit(q_empty_addr + (q_cons_stage) * 8);
                    }
                    if (lane == 0) {
                    }
                    mbarrier_wait(o_empty_addr, _phase_o_empty_0);
                    _phase_o_empty_0 ^= 1;
                    if (lane == 0) {
                    }
                    int first_pv = 1;
                    #pragma unroll 1
                    for (int n_2 = 0; n_2 < cnt_m; n_2++) {
                        int next_n = n_2 + 1;
                        if (lane == 0) {
                        }
                        if (next_n < cnt_m) {
                            if (lane == 0) {
                            }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                            {
                                mbarrier_wait(k_full_addr + (k_cons_stage) * 8, k_cons_phase);
                            }
#else
                            mbarrier_wait(k_full_addr + (k_cons_stage) * 8, k_cons_phase);
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                            mbarrier_wait(k_full_addr + (k_cons_stage) * 8, k_cons_phase);
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                            if (lane == 0) {
                            }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                            int _mma_a_lo_2 = make_warp_uniform((((smem_k_addr) >> 4) & 0x3FFF) + (k_cons_stage) * 2048);
                            int _mma_b_lo_2 = make_warp_uniform((((smem_q32_addr) >> 4) & 0x3FFF) + (0) * 512);
#else
                            int _mma_a_lo_1 = make_warp_uniform((((smem_qt_addr) >> 4) & 0x3FFF) + (q_cons_stage) * 1024);
                            int _mma_b_lo_1 = make_warp_uniform((((smem_k_addr) >> 4) & 0x3FFF) + (k_cons_stage) * 2048);
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                            int _mma_a_lo_1 = make_warp_uniform((((smem_qt_addr) >> 4) & 0x3FFF) + (q_cons_stage) * 1024);
                            int _mma_b_lo_1 = make_warp_uniform((((smem_k_addr) >> 4) & 0x3FFF) + (k_cons_stage) * 2048);
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                            {
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                                uint64_t _mma_ss_a_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_2);
                                uint64_t _mma_ss_b_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_2);
#else
                                uint64_t _mma_ss_a_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_1);
                                uint64_t _mma_ss_b_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_1);
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                                uint64_t _mma_ss_a_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_1);
                                uint64_t _mma_ss_b_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_1);
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                                if (elect_sync()) {
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                                    tcgen05_mma_f16((tmem_tmem_sT + (s_buf * 64)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134743184, 0);
#else
                                    tcgen05_mma_f16((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 69207184, 0);
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                                    tcgen05_mma_f16((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 69207184, 0);
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                                }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                                incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
#else
                                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                                if (elect_sync()) {
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                                    tcgen05_mma_f16((tmem_tmem_sT + (s_buf * 64)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134743184, 1);
#else
                                    tcgen05_mma_f16((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 69207184, 1);
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                                    tcgen05_mma_f16((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 69207184, 1);
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                                }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                                incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
#else
                                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                                if (elect_sync()) {
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                                    tcgen05_mma_f16((tmem_tmem_sT + (s_buf * 64)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134743184, 1);
#else
                                    tcgen05_mma_f16((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 69207184, 1);
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                                    tcgen05_mma_f16((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 69207184, 1);
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                                }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                                incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
#else
                                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                                if (elect_sync()) {
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                                    tcgen05_mma_f16((tmem_tmem_sT + (s_buf * 64)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134743184, 1);
#else
                                    tcgen05_mma_f16((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 69207184, 1);
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                                    tcgen05_mma_f16((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 69207184, 1);
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                                }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                                incr_smem_desc_lo(_mma_ss_a_desc_2, 1018U);
                                incr_smem_desc_lo(_mma_ss_b_desc_2, 250U);
#else
                                incr_smem_desc_lo(_mma_ss_a_desc_1, 506U);
                                incr_smem_desc_lo(_mma_ss_b_desc_1, 1018U);
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                                incr_smem_desc_lo(_mma_ss_a_desc_1, 506U);
                                incr_smem_desc_lo(_mma_ss_b_desc_1, 1018U);
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                                if (elect_sync()) {
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                                    tcgen05_mma_f16((tmem_tmem_sT + (s_buf * 64)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134743184, 1);
#else
                                    tcgen05_mma_f16((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 69207184, 1);
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                                    tcgen05_mma_f16((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 69207184, 1);
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                                }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                                incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
#else
                                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                                if (elect_sync()) {
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                                    tcgen05_mma_f16((tmem_tmem_sT + (s_buf * 64)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134743184, 1);
#else
                                    tcgen05_mma_f16((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 69207184, 1);
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                                    tcgen05_mma_f16((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 69207184, 1);
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                                }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                                incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
#else
                                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                                if (elect_sync()) {
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                                    tcgen05_mma_f16((tmem_tmem_sT + (s_buf * 64)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134743184, 1);
#else
                                    tcgen05_mma_f16((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 69207184, 1);
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                                    tcgen05_mma_f16((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 69207184, 1);
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                                }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                                incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
#else
                                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                                if (elect_sync()) {
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                                    tcgen05_mma_f16((tmem_tmem_sT + (s_buf * 64)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134743184, 1);
                                }
                            }
                            int rows_hi_qk_0 = ((N_ROWS > 32) ? 1 : 0);
                            if (rows_hi_qk_0 != 0) {
                                int _mma_a_lo_3 = make_warp_uniform((((smem_k_addr) >> 4) & 0x3FFF) + (k_cons_stage) * 2048);
                                int _mma_b_lo_3 = make_warp_uniform((((smem_q32_addr) >> 4) & 0x3FFF) + (1) * 512);
                                {
                                    uint64_t _mma_ss_a_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_3);
                                    uint64_t _mma_ss_b_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_3);
                                    if (elect_sync()) {
                                        tcgen05_mma_f16((tmem_tmem_sT + (s_buf * 64 + 32)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134743184, 0);
                                    }
                                    incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                                    incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                                    if (elect_sync()) {
                                        tcgen05_mma_f16((tmem_tmem_sT + (s_buf * 64 + 32)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134743184, 1);
                                    }
                                    incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                                    incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                                    if (elect_sync()) {
                                        tcgen05_mma_f16((tmem_tmem_sT + (s_buf * 64 + 32)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134743184, 1);
                                    }
                                    incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                                    incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                                    if (elect_sync()) {
                                        tcgen05_mma_f16((tmem_tmem_sT + (s_buf * 64 + 32)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134743184, 1);
                                    }
                                    incr_smem_desc_lo(_mma_ss_a_desc_3, 1018U);
                                    incr_smem_desc_lo(_mma_ss_b_desc_3, 250U);
                                    if (elect_sync()) {
                                        tcgen05_mma_f16((tmem_tmem_sT + (s_buf * 64 + 32)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134743184, 1);
                                    }
                                    incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                                    incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                                    if (elect_sync()) {
                                        tcgen05_mma_f16((tmem_tmem_sT + (s_buf * 64 + 32)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134743184, 1);
                                    }
                                    incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                                    incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                                    if (elect_sync()) {
                                        tcgen05_mma_f16((tmem_tmem_sT + (s_buf * 64 + 32)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134743184, 1);
                                    }
                                    incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                                    incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                                    if (elect_sync()) {
                                        tcgen05_mma_f16((tmem_tmem_sT + (s_buf * 64 + 32)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134743184, 1);
                                    }
#else
                                    tcgen05_mma_f16((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 69207184, 1);
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                                    tcgen05_mma_f16((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 69207184, 1);
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                                }
                            }
                            elect_commit(s_full_addr + (s_buf) * 8);
                            elect_commit(k_empty_addr + (k_cons_stage) * 8);
                            k_cons_stage += 1;
                            if (k_cons_stage == 3) { k_cons_stage = 0; k_cons_phase ^= 1; }
                            s_buf = s_buf ^ 1;
                            if (next_n + 1 == cnt_m) {
                                elect_commit(q_empty_addr + (q_cons_stage) * 8);
                            }
                        }
                        if (lane == 0) {
                        }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                        {
                            mbarrier_wait(v_full_addr + (v_cons_stage) * 8, v_cons_phase);
                        }
#else
                        mbarrier_wait(v_full_addr + (v_cons_stage) * 8, v_cons_phase);
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                        mbarrier_wait(v_full_addr + (v_cons_stage) * 8, v_cons_phase);
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                        if (lane == 0) {
                        }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                        if (pv_idx_m >= 2) {
                            unsigned int k_obs_m = pv_idx_m - 2;
                            {
                                mbarrier_wait(o_ready_addr + ((int)(k_obs_m & 1)) * 8, (int)(k_obs_m >> 1 & 1));
                            }
                        }
                        {
                            mbarrier_wait(p_full_addr + (pf_stage) * 8, pf_phase);
                        }
#else
                        mbarrier_wait(p_full_addr + (pf_stage) * 8, pf_phase);
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                        mbarrier_wait(p_full_addr + (pf_stage) * 8, pf_phase);
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                        if (lane == 0) {
                        }
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int first_pv_flag = first_pv;
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                        int rows_hi_m = ((N_ROWS > 32) ? 1 : 0);
                        int pb_m = 0;
                        if (rows_hi_m == 0) {
                            pb_m = (int)(pv_idx_m & 1);
                        }
                        if (pb_m == 0) {
                            int _mma_a_lo_4 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_cons_stage) * 2048);
                            int _mma_b_lo_4 = make_warp_uniform((((swap_pT_addr) >> 4) & 0x3FFF) | 0x2000000);
                            asm volatile(
#else
                        int _mma_b_lo_2 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_cons_stage) * 2048);
                        asm volatile(
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                        int _mma_b_lo_2 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_cons_stage) * 2048);
                        asm volatile(
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
#else
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x80004020;\n\t"
                    "mov.b32 id, 134841488;\n\t"
                    "mov.b32 alo, %0;\n\t"
#else
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 69272720;\n\t"
                    "mov.b32 ta, %2;\n\t"
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 69272720;\n\t"
                    "mov.b32 ta, %2;\n\t"
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                    "mov.b32 blo, %1;\n\t"
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
#else
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                    "}\n"
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                    :: "r"(_mma_a_lo_4), "r"(_mma_b_lo_4), "r"(tmem_tmem_oT), "r"(((first_pv_flag) ? 0 : 1)));
                        } else {
                            int _mma_a_lo_5 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_cons_stage) * 2048);
                            int _mma_b_lo_5 = make_warp_uniform((((swap_pT1_addr) >> 4) & 0x3FFF) | 0x2000000);
                            asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x80004020;\n\t"
                    "mov.b32 id, 134841488;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_5), "r"(_mma_b_lo_5), "r"(tmem_tmem_oT), "r"(((first_pv_flag) ? 0 : 1)));
                        }
                        if (rows_hi_m != 0) {
                            int o_st_pa = (int)(pv_idx_m & 1);
                            elect_commit(pa_done_addr + (o_st_pa) * 8);
                            mbarrier_wait(p_full_b_addr + (pf_stage) * 8, pf_phase);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_a_lo_6 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_cons_stage) * 2048);
                            int _mma_b_lo_6 = make_warp_uniform((((swap_pT_addr) >> 4) & 0x3FFF) | 0x2000000);
                            asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x80004020;\n\t"
                    "mov.b32 id, 134841488;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_6), "r"(_mma_b_lo_6), "r"((tmem_tmem_oT + (32))), "r"(((first_pv_flag) ? 0 : 1)));
                        }
#else
                    :: "r"(tmem_tmem_o), "r"(_mma_b_lo_2), "r"(tmem_tmem_s + (int)pf_stage * 128), "r"(((first_pv_flag) ? 0 : 1)));
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                    :: "r"(tmem_tmem_o), "r"(_mma_b_lo_2), "r"(tmem_tmem_s + (int)pf_stage * 128), "r"(((first_pv_flag) ? 0 : 1)));
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                        int o_st_m = (int)(pv_idx_m & 1);
                        elect_commit(v_empty_addr + (v_cons_stage) * 8);
                        elect_commit(o_ready_addr + (o_st_m) * 8);
                        pv_idx_m = pv_idx_m + 1;
                        v_cons_stage += 1;
                        if (v_cons_stage == 3) { v_cons_stage = 0; v_cons_phase ^= 1; }
                        pf_stage += 1;
                        if (pf_stage == 2) { pf_stage = 0; pf_phase ^= 1; }
                        first_pv = 0;
                        if (lane == 0) {
                        }
                    }
                    q_cons_phase ^= 1;
                    if (lane == 0) {
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_m) * 8, _phase_work_full_2);
                unsigned int base_0_2 = work_stage_m * 16;
                unsigned int valid_1_2 = work_token_words[base_0_2];
                unsigned int kind_2_2 = work_token_words[base_0_2 + 1];
                unsigned int batch_3_2 = work_token_words[base_0_2 + 2];
                unsigned int kv_head_4_2 = work_token_words[base_0_2 + 3];
                unsigned int block_begin_5_2 = work_token_words[base_0_2 + 4];
                unsigned int block_end_6_2 = work_token_words[base_0_2 + 5];
                unsigned int seqlen_7_2 = work_token_words[base_0_2 + 6];
                unsigned int n_chunks_8_2 = work_token_words[base_0_2 + 7];
                unsigned int slot_tile_base_9_2 = work_token_words[base_0_2 + 8];
                unsigned int counter_idx_10_2 = work_token_words[base_0_2 + 9];
                unsigned int chunk_11_2 = work_token_words[base_0_2 + 10];
                unsigned int phase_12_2 = work_token_words[base_0_2 + 11];
                mbarrier_arrive(work_empty_addr + (work_stage_m) * 8);
                work_stage_m += 1;
                if (work_stage_m == 4) { work_stage_m = 0; _phase_work_full_2 ^= 1; }
                valid_m = valid_1_2;
                kind_m = (int)kind_2_2;
                block_begin_m = (int)block_begin_5_2;
                block_end_m = (int)block_end_6_2;
            }
            unsigned int _phase_tmem_dealloc_0 = 0;
            mbarrier_wait(tmem_dealloc_addr, _phase_tmem_dealloc_0);
            _phase_tmem_dealloc_0 ^= 1;
            if (lane == 0) {
            }
        }
    // ---- Role: load_pgoff ----
    } else if (warp == 9) {
        { // load_pgoff_main
            int pg_blk_p = lane >> 3;
            int pg_lane_p = lane & 7;
            unsigned int page_prod_stage = 0;
            unsigned int page_prod_phase = 1;
            unsigned int work_stage_p = 0;
            unsigned int _phase_work_full_3 = 0;
            mbarrier_wait(work_full_addr + (work_stage_p) * 8, _phase_work_full_3);
            unsigned int base_3 = work_stage_p * 16;
            unsigned int valid_4 = work_token_words[base_3];
            unsigned int kind_4 = work_token_words[base_3 + 1];
            unsigned int batch_4 = work_token_words[base_3 + 2];
            unsigned int kv_head_3 = work_token_words[base_3 + 3];
            unsigned int block_begin_3 = work_token_words[base_3 + 4];
            unsigned int block_end_3 = work_token_words[base_3 + 5];
            unsigned int seqlen_3 = work_token_words[base_3 + 6];
            unsigned int n_chunks_3 = work_token_words[base_3 + 7];
            unsigned int slot_tile_base_3 = work_token_words[base_3 + 8];
            unsigned int counter_idx_3 = work_token_words[base_3 + 9];
            unsigned int chunk_3 = work_token_words[base_3 + 10];
            unsigned int phase_3 = work_token_words[base_3 + 11];
            mbarrier_arrive(work_empty_addr + (work_stage_p) * 8);
            work_stage_p += 1;
            if (work_stage_p == 4) { work_stage_p = 0; _phase_work_full_3 ^= 1; }
            unsigned int valid_p = valid_4;
            int kind_p = (int)kind_4;
            int batch_idx_p = (int)batch_4;
            int block_begin_p = (int)block_begin_3;
            int block_end_p = (int)block_end_3;
            int seqlen_kv_p = (int)seqlen_3;
            #pragma unroll 1
            for (unsigned int _tile_iter_p = 0; _tile_iter_p < max_items; _tile_iter_p++) {
                if (valid_p == 0) {
                    break;
                }
                if (kind_p == 0) {
                    int cta_n_blocks_p = block_end_p - block_begin_p;
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 5
                    int _max_3 = (((seqlen_kv_p + PAGE_SIZE - 1) / PAGE_SIZE - 1) > (0) ? ((seqlen_kv_p + PAGE_SIZE - 1) / PAGE_SIZE - 1) : (0));
                    int max_pg_p = _max_3;
#else
                    int _max_2 = (((seqlen_kv_p + PAGE_SIZE - 1) / PAGE_SIZE - 1) > (0) ? ((seqlen_kv_p + PAGE_SIZE - 1) / PAGE_SIZE - 1) : (0));
                    int max_pg_p = _max_2;
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
#else
#if CAKE_FMHA_DCP_PROGRAM == 4
                    int _max_3 = (((seqlen_kv_p + PAGE_SIZE - 1) / PAGE_SIZE - 1) > (0) ? ((seqlen_kv_p + PAGE_SIZE - 1) / PAGE_SIZE - 1) : (0));
                    int max_pg_p = _max_3;
#else
                    int _max_2 = (((seqlen_kv_p + PAGE_SIZE - 1) / PAGE_SIZE - 1) > (0) ? ((seqlen_kv_p + PAGE_SIZE - 1) / PAGE_SIZE - 1) : (0));
                    int max_pg_p = _max_2;
#endif  // CAKE_FMHA_DCP_PROGRAM == 4
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                    int _max_14 = (((seqlen_kv_p + PAGE_SIZE - 1) / PAGE_SIZE - 1) > (0) ? ((seqlen_kv_p + PAGE_SIZE - 1) / PAGE_SIZE - 1) : (0));
                    int max_pg_p = _max_14;
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                    int pt_base_p = batch_idx_p * max_pages_per_seq;
                    #pragma unroll 1
                    for (int ni0_p = 0; ni0_p < cta_n_blocks_p; ni0_p += 4) {
                        int g_cnt_p = cta_n_blocks_p - ni0_p;
                        if (g_cnt_p > 4) {
                            g_cnt_p = 4;
                        }
                        int n_block_p = block_begin_p + cta_n_blocks_p - 1 - (ni0_p + pg_blk_p);
                        int page_idx_p = n_block_p * 8 + pg_lane_p;
                        if (page_idx_p > max_pg_p) {
                            page_idx_p = max_pg_p;
                        }
                        int page_id_p = 0;
                        if (pg_blk_p < g_cnt_p) {
                            page_id_p = page_table[pt_base_p + page_idx_p];
                        }
                        #pragma unroll
                        for (int gc_p = 0; gc_p < 4; gc_p++) {
                            if (g_cnt_p > gc_p) {
                                int st_u_p = page_prod_stage + (unsigned int)gc_p;
                                int st_p = ((st_u_p >= 6) ? st_u_p - 6 : st_u_p);
                                int ph_p = ((st_u_p >= 6) ? page_prod_phase ^ 1 : page_prod_phase);
                                mbarrier_wait(page_offsets_empty_addr + (st_p) * 8, ph_p);
                                if (pg_blk_p == gc_p) {
                                    smem_page_offsets[st_p * 8 + pg_lane_p] = page_id_p;
                                }
                                __syncwarp();
                                if (elect_sync()) {
                                    mbarrier_arrive(page_offsets_full_addr + (st_p) * 8);
                                }
                            }
                        }
                        #pragma unroll
                        for (int gv_p = 0; gv_p < 4; gv_p++) {
                            if (g_cnt_p > gv_p) {
                                page_prod_stage += 1;
                                if (page_prod_stage == 6) { page_prod_stage = 0; page_prod_phase ^= 1; }
                            }
                        }
                        if (lane == 0) {
                        }
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_p) * 8, _phase_work_full_3);
                unsigned int base_0_3 = work_stage_p * 16;
                unsigned int valid_1_3 = work_token_words[base_0_3];
                unsigned int kind_2_3 = work_token_words[base_0_3 + 1];
                unsigned int batch_3_3 = work_token_words[base_0_3 + 2];
                unsigned int kv_head_4_3 = work_token_words[base_0_3 + 3];
                unsigned int block_begin_5_3 = work_token_words[base_0_3 + 4];
                unsigned int block_end_6_3 = work_token_words[base_0_3 + 5];
                unsigned int seqlen_7_3 = work_token_words[base_0_3 + 6];
                unsigned int n_chunks_8_3 = work_token_words[base_0_3 + 7];
                unsigned int slot_tile_base_9_3 = work_token_words[base_0_3 + 8];
                unsigned int counter_idx_10_3 = work_token_words[base_0_3 + 9];
                unsigned int chunk_11_3 = work_token_words[base_0_3 + 10];
                unsigned int phase_12_3 = work_token_words[base_0_3 + 11];
                mbarrier_arrive(work_empty_addr + (work_stage_p) * 8);
                work_stage_p += 1;
                if (work_stage_p == 4) { work_stage_p = 0; _phase_work_full_3 ^= 1; }
                valid_p = valid_1_3;
                kind_p = (int)kind_2_3;
                batch_idx_p = (int)batch_3_3;
                block_begin_p = (int)block_begin_5_3;
                block_end_p = (int)block_end_6_3;
                seqlen_kv_p = (int)seqlen_7_3;
            }
            if (lane == 0) {
            }
        }
    // ---- Role: scheduler ----
    } else if (warp == 10) {
        { // scheduler_main
            int lane_0 = lane;
            int num_ctas = gridDim.x;
            if (lane_0 == 0) {
            }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
            int sow_pairs_bound = (max_pages_per_seq * PAGE_SIZE + 255) / 256;
            unsigned int sow_n_max = (unsigned int)(sow_pairs_bound + 1 - 1);
            sow_n_max = 1;
            unsigned int sow_tiles = (unsigned int)(batch_size * num_kv_heads);
            unsigned int sow_chunk_items = sow_tiles * sow_n_max;
            unsigned int sow_shift = 0;
            if (sow_n_max > 4) {
                sow_shift = 1;
            }
            if (sow_n_max > 8) {
                sow_shift = 2;
            }
            if (sow_n_max > 16) {
                sow_shift = 3;
            }
            if (sow_chunk_items + (sow_tiles << 3) <= (unsigned int)num_ctas) {
                sow_shift = 3;
            }
            unsigned int sow_one = 1;
            unsigned int sow_total = sow_chunk_items + (sow_tiles << sow_shift);
            unsigned int sow_pairs_u = 1;
            sow_shift = 0;
            sow_total = sow_chunk_items;
            sow_pairs_u = (unsigned int)sow_pairs_bound;
            if (blockIdx.x == 0) {
                if (lane_0 == 0) {
                    int sow_plan_off = num_ctas * 2048;
                    *(reinterpret_cast<float*>(partial_stats + sow_plan_off) + (0)) = (float)sow_pairs_u;
                    *(reinterpret_cast<float*>(partial_stats + (sow_plan_off + 1)) + (0)) = (float)sow_total;
#else
            int items_per_chunk = num_kv_heads;
            int num_groups = (batch_size + 32 - 1) / 32;
            int cp_mask = (1 << cp_world_log2) - 1;
            #pragma unroll 1
            for (int gs = 0; gs < num_groups; gs++) {
                int bs = gs * 32 + lane_0;
                if (bs < batch_size) {
                    int last_pos = causal_seqlens_kv_global[bs] + (q_len - 1) - cp_rank;
                    int len_bs = 0;
                    int phase_bs = 0;
                    if (last_pos >= 0) {
                        len_bs = (last_pos >> cp_world_log2) + 1;
                        phase_bs = last_pos & cp_mask;
                    }
                    sched_seq_lens[bs] = len_bs;
                    sched_phase[bs] = phase_bs;
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                }
            }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
            unsigned int sow_ticket = blockIdx.x;
            unsigned int sow_valid = 0;
            unsigned int sow_kind = 0;
            int sow_batch = 0;
            int sow_head = 0;
            int sow_bbeg = 0;
            int sow_bend = 0;
            int sow_len = 0;
            int sow_len_tok = 0;
            int sow_phase = 0;
            int sow_phase_tok = 0;
            int sow_n = 0;
            int sow_slot = 0;
            int sow_ctr = 0;
            int sow_chunk = 0;
            unsigned int sow_slice = 0;
            unsigned int sow_tile = 0;
            unsigned int sow_r = 0;
            if (sow_ticket < sow_total) {
                if (sow_ticket < sow_chunk_items) {
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 5
                    float _rcp_1 = approx_rcp((float)sow_n_max);
                    unsigned int q_7 = (unsigned int)((float)sow_ticket * _rcp_1);
                    if (sow_ticket < q_7 * sow_n_max) {
                        q_7 = q_7 - 1;
#else
                    float _rcp_0 = approx_rcp((float)sow_n_max);
                    unsigned int q_6 = (unsigned int)((float)sow_ticket * _rcp_0);
                    if (sow_ticket < q_6 * sow_n_max) {
                        q_6 = q_6 - 1;
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
#else
#if CAKE_FMHA_DCP_PROGRAM == 4
                    float _rcp_1 = approx_rcp((float)sow_n_max);
                    unsigned int q_1 = (unsigned int)((float)sow_ticket * _rcp_1);
                    if (sow_ticket < q_1 * sow_n_max) {
                        q_1 = q_1 - 1;
#else
                    float _rcp_0 = approx_rcp((float)sow_n_max);
                    unsigned int q = (unsigned int)((float)sow_ticket * _rcp_0);
                    if (sow_ticket < q * sow_n_max) {
                        q = q - 1;
#endif  // CAKE_FMHA_DCP_PROGRAM == 4
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                    }
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 5
                    if (sow_ticket >= (q_7 + 1) * sow_n_max) {
                        q_7 = q_7 + 1;
#else
                    if (sow_ticket >= (q_6 + 1) * sow_n_max) {
                        q_6 = q_6 + 1;
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
#else
#if CAKE_FMHA_DCP_PROGRAM == 4
                    if (sow_ticket >= (q_1 + 1) * sow_n_max) {
                        q_1 = q_1 + 1;
#else
                    if (sow_ticket >= (q + 1) * sow_n_max) {
                        q = q + 1;
#endif  // CAKE_FMHA_DCP_PROGRAM == 4
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                    }
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 5
                    sow_tile = q_7;
#else
                    sow_tile = q_6;
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
#else
#if CAKE_FMHA_DCP_PROGRAM == 4
                    sow_tile = q_1;
#else
                    sow_tile = q;
#endif  // CAKE_FMHA_DCP_PROGRAM == 4
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                    sow_chunk = (int)(sow_ticket - sow_tile * sow_n_max);
                } else {
                    sow_kind = 1;
                    sow_r = sow_ticket - sow_chunk_items;
                    sow_tile = sow_r >> sow_shift;
                    sow_slice = sow_r - (sow_tile << sow_shift);
#else
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncwarp();
            unsigned int total_pairs = 0;
            unsigned int p_max = 0;
            unsigned int p_min = 4294967295;
            #pragma unroll 1
            for (int g1 = 0; g1 < num_groups; g1++) {
                int b1 = g1 * 32 + lane_0;
                unsigned int pairs1 = 0;
                unsigned int pairs1_min = 4294967295;
                if (b1 < batch_size) {
                    int s1 = sched_seq_lens[b1];
                    int _max_0 = ((s1) > (1) ? (s1) : (1));
                    pairs1 = (unsigned int)((_max_0 + 255) / 256);
                    pairs1_min = pairs1;
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 5
                float _rcp_2 = approx_rcp((float)num_kv_heads);
                unsigned int q_8 = (unsigned int)((float)sow_tile * _rcp_2);
                if (sow_tile < q_8 * (unsigned int)num_kv_heads) {
                    q_8 = q_8 - 1;
#else
                float _rcp_1 = approx_rcp((float)num_kv_heads);
                unsigned int q_7 = (unsigned int)((float)sow_tile * _rcp_1);
                if (sow_tile < q_7 * (unsigned int)num_kv_heads) {
                    q_7 = q_7 - 1;
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
#else
#if CAKE_FMHA_DCP_PROGRAM == 4
                float _rcp_2 = approx_rcp((float)num_kv_heads);
                unsigned int q_2 = (unsigned int)((float)sow_tile * _rcp_2);
                if (sow_tile < q_2 * (unsigned int)num_kv_heads) {
                    q_2 = q_2 - 1;
#else
                float _rcp_1 = approx_rcp((float)num_kv_heads);
                unsigned int q_1 = (unsigned int)((float)sow_tile * _rcp_1);
                if (sow_tile < q_1 * (unsigned int)num_kv_heads) {
                    q_1 = q_1 - 1;
#endif  // CAKE_FMHA_DCP_PROGRAM == 4
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                unsigned int _warp_redux_u32_0;
                asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_0) : "r"(pairs1));
                total_pairs += _warp_redux_u32_0;
                unsigned int _warp_redux_u32_1;
                asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_1) : "r"(pairs1));
                unsigned int _max_1 = ((p_max) > (_warp_redux_u32_1) ? (p_max) : (_warp_redux_u32_1));
                p_max = _max_1;
                unsigned int _warp_redux_u32_2;
                asm volatile("redux.sync.min.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_2) : "r"(pairs1_min));
                unsigned int _min_0 = ((p_min) < (_warp_redux_u32_2) ? (p_min) : (_warp_redux_u32_2));
                p_min = _min_0;
            }
            unsigned int total_work = total_pairs * (unsigned int)items_per_chunk;
            unsigned int uniform_u = 0;
            if (p_max == p_min) {
                uniform_u = 1;
            }
            float _rcp_0 = approx_rcp((float)num_ctas);
            float ctas_rcp = _rcp_0;
            float _rcp_1 = approx_rcp((float)items_per_chunk);
            float items_rcp = _rcp_1;
            unsigned int q = (unsigned int)((float)total_work * ctas_rcp);
            if (total_work < q * (unsigned int)num_ctas) {
                q = q - 1;
            }
            if (total_work >= (q + 1) * (unsigned int)num_ctas) {
                q = q + 1;
            }
            if (total_work > q * (unsigned int)num_ctas) {
                q = q + 1;
            }
            unsigned int ideal_pairs = q;
            unsigned int balance_k = (ideal_pairs + 64 - 1) / 64;
            if (balance_k < 1) {
                balance_k = 1;
            }
            if (balance_k > 8) {
                balance_k = 8;
            }
            unsigned int chunk_divisor = balance_k * (unsigned int)num_ctas;
            float _rcp_2 = approx_rcp((float)chunk_divisor);
            unsigned int q_1 = (unsigned int)((float)total_work * _rcp_2);
            if (total_work < q_1 * chunk_divisor) {
                q_1 = q_1 - 1;
            }
            if (total_work >= (q_1 + 1) * chunk_divisor) {
                q_1 = q_1 + 1;
            }
            if (total_work > q_1 * chunk_divisor) {
                q_1 = q_1 + 1;
            }
            unsigned int chunk_pairs_u = q_1;
            if (chunk_pairs_u < 2) {
                chunk_pairs_u = 2;
            }
            if (chunk_pairs_u < p_max) {
                unsigned int split_ok = 0;
                if (p_max > ideal_pairs + chunk_pairs_u) {
                    split_ok = 1;
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 5
                if (sow_tile >= (q_8 + 1) * (unsigned int)num_kv_heads) {
                    q_8 = q_8 + 1;
#else
                if (sow_tile >= (q_7 + 1) * (unsigned int)num_kv_heads) {
                    q_7 = q_7 + 1;
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
#else
#if CAKE_FMHA_DCP_PROGRAM == 4
                if (sow_tile >= (q_2 + 1) * (unsigned int)num_kv_heads) {
                    q_2 = q_2 + 1;
#else
                if (sow_tile >= (q_1 + 1) * (unsigned int)num_kv_heads) {
                    q_1 = q_1 + 1;
#endif  // CAKE_FMHA_DCP_PROGRAM == 4
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                if (chunk_pairs_u < 2 * (p_max - p_min)) {
                    split_ok = 1;
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 5
                sow_batch = (int)q_8;
#else
                sow_batch = (int)q_7;
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
#else
#if CAKE_FMHA_DCP_PROGRAM == 4
                sow_batch = (int)q_2;
#else
                sow_batch = (int)q_1;
#endif  // CAKE_FMHA_DCP_PROGRAM == 4
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                sow_head = (int)(sow_tile - (unsigned int)sow_batch * (unsigned int)num_kv_heads);
                int sow_last = causal_seqlens_kv_global[sow_batch] + (q_len - 1) - cp_rank;
                int sow_cp_mask = (1 << cp_world_log2) - 1;
                if (sow_last >= 0) {
                    sow_len = (sow_last >> cp_world_log2) + 1;
                    sow_phase = sow_last & sow_cp_mask;
                }
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 5
                int _max_1 = ((sow_len) > (1) ? (sow_len) : (1));
                int sow_pairs = (_max_1 + 255) / 256;
#else
                int _max_0 = ((sow_len) > (1) ? (sow_len) : (1));
                int sow_pairs = (_max_0 + 255) / 256;
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
#else
#if CAKE_FMHA_DCP_PROGRAM == 4
                int _max_1 = ((sow_len) > (1) ? (sow_len) : (1));
                int sow_pairs = (_max_1 + 255) / 256;
#else
                int _max_0 = ((sow_len) > (1) ? (sow_len) : (1));
                int sow_pairs = (_max_0 + 255) / 256;
#endif  // CAKE_FMHA_DCP_PROGRAM == 4
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                sow_n = sow_pairs + 1 - 1;
                sow_n = 1;
                if (sow_n > 1) {
                    sow_slot = sow_batch * (int)sow_n_max * num_kv_heads + sow_head;
                    sow_ctr = sow_batch * num_kv_heads + sow_head;
                }
                if (sow_kind == 0) {
                    if (sow_chunk < sow_n) {
                        sow_valid = 1;
                        sow_len_tok = sow_len;
                        sow_phase_tok = sow_phase;
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 5
                        int _max_2 = (((sow_len + BLOCK_N - 1) / BLOCK_N) > (1) ? ((sow_len + BLOCK_N - 1) / BLOCK_N) : (1));
                        int sow_nblk = _max_2;
#else
                        int _max_1 = (((sow_len + BLOCK_N - 1) / BLOCK_N) > (1) ? ((sow_len + BLOCK_N - 1) / BLOCK_N) : (1));
                        int sow_nblk = _max_1;
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
#else
#if CAKE_FMHA_DCP_PROGRAM == 4
                        int _max_2 = (((sow_len + BLOCK_N - 1) / BLOCK_N) > (1) ? ((sow_len + BLOCK_N - 1) / BLOCK_N) : (1));
                        int sow_nblk = _max_2;
#else
                        int _max_1 = (((sow_len + BLOCK_N - 1) / BLOCK_N) > (1) ? ((sow_len + BLOCK_N - 1) / BLOCK_N) : (1));
                        int sow_nblk = _max_1;
#endif  // CAKE_FMHA_DCP_PROGRAM == 4
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
                        sow_bbeg = 2 * sow_chunk;
                        sow_bend = 2 * (sow_chunk + 1);
                        if (sow_chunk + 1 == sow_n) {
                            sow_bend = sow_nblk;
                        }
                    }
                } else if (sow_n > 2) {
                    sow_valid = 1;
                    sow_chunk = 0;
                    sow_bbeg = (int)sow_slice;
                    sow_bend = (int)sow_shift;
#else
                if (split_ok == 0) {
                    chunk_pairs_u = p_max;
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                }
            }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
            if (lane_0 == 0) {
            }
            unsigned int work_stage_sched = 0;
            unsigned int _phase_work_empty = 1;
            mbarrier_wait(work_empty_addr + (work_stage_sched) * 8, _phase_work_empty);
            unsigned int sow_base = work_stage_sched * 16;
            if (lane_0 == 0) {
                if (sow_valid != 0) {
                    if (sow_kind == 1) {
                        unsigned int sow_arrived = 0;
                        #pragma unroll 1
                        for (int _poll_w = 0; _poll_w < 1073741824; _poll_w++) {
                            unsigned int _atomic_old_0;
                            asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                                : "=r"(_atomic_old_0) : "l"(&tile_counters[sow_ctr * 4]), "r"(static_cast<uint32_t>(0)) : "memory");
                            sow_arrived = _atomic_old_0;
                            if ((int)sow_arrived == sow_n) {
                                break;
                            }
                        }
                    }
#else
            int whole_items = batch_size * items_per_chunk;
            if (whole_items <= num_ctas) {
                float _rcp_3 = approx_rcp((float)whole_items);
                unsigned int q_0 = (unsigned int)((float)(unsigned int)num_ctas * _rcp_3);
                if (q_0 * (unsigned int)whole_items > (unsigned int)num_ctas) {
                    q_0 = q_0 - 1;
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                work_token_words[sow_base + 1] = sow_kind;
                work_token_words[sow_base + 2] = (unsigned int)sow_batch;
                work_token_words[sow_base + 3] = (unsigned int)sow_head;
                work_token_words[sow_base + 4] = (unsigned int)sow_bbeg;
                work_token_words[sow_base + 5] = (unsigned int)sow_bend;
                work_token_words[sow_base + 6] = (unsigned int)sow_len_tok;
                work_token_words[sow_base + 7] = (unsigned int)sow_n;
                work_token_words[sow_base + 8] = (unsigned int)sow_slot;
                work_token_words[sow_base + 9] = (unsigned int)sow_ctr;
                work_token_words[sow_base + 10] = (unsigned int)sow_chunk;
                work_token_words[sow_base + 11] = (unsigned int)sow_phase_tok;
                work_token_words[sow_base] = sow_valid;
                mbarrier_arrive(work_full_addr + (work_stage_sched) * 8);
                if (sow_valid != 0) {
                    if (sow_kind == 1) {
                        unsigned int sow_slices = sow_one << sow_shift;
                        unsigned int _atomic_old_1;
                        asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                            : "=r"(_atomic_old_1) : "l"(&tile_counters[sow_ctr * 4 + 1]), "r"(static_cast<uint32_t>(1)) : "memory");
                        unsigned int sow_pops = _atomic_old_1;
                        if (sow_pops + 1 == sow_slices) {
                            *(reinterpret_cast<unsigned int*>(tile_counters + (sow_ctr * 4)) + (0)) = 0;
                            *(reinterpret_cast<unsigned int*>(tile_counters + (sow_ctr * 4 + 1)) + (0)) = 0;
#else
                if ((q_0 + 1) * (unsigned int)whole_items <= (unsigned int)num_ctas) {
                    q_0 = q_0 + 1;
                }
                int n_even = (int)q_0;
                if (n_even > 1) {
                    if (p_max >= 8 * (p_max - p_min)) {
                        float _rcp_4 = approx_rcp((float)n_even);
                        unsigned int q_2 = (unsigned int)((float)p_max * _rcp_4);
                        if (p_max < q_2 * (unsigned int)n_even) {
                            q_2 = q_2 - 1;
                        }
                        if (p_max >= (q_2 + 1) * (unsigned int)n_even) {
                            q_2 = q_2 + 1;
                        }
                        if (p_max > q_2 * (unsigned int)n_even) {
                            q_2 = q_2 + 1;
                        }
                        unsigned int l_even = q_2;
                        if (l_even < 2) {
                            l_even = 2;
                        }
                        if (l_even < p_max) {
                            chunk_pairs_u = l_even;
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                        }
                    }
                }
            }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
            work_stage_sched += 1;
            if (work_stage_sched == 4) { work_stage_sched = 0; _phase_work_empty ^= 1; }
            if (sow_valid != 0) {
                mbarrier_wait(work_empty_addr + (work_stage_sched) * 8, _phase_work_empty);
#else
            unsigned int q_2_1 = (unsigned int)((float)total_work * ctas_rcp);
            if (total_work < q_2_1 * (unsigned int)num_ctas) {
                q_2_1 = q_2_1 - 1;
            }
            if (total_work >= (q_2_1 + 1) * (unsigned int)num_ctas) {
                q_2_1 = q_2_1 + 1;
            }
            if (total_work > q_2_1 * (unsigned int)num_ctas) {
                q_2_1 = q_2_1 + 1;
            }
            unsigned int l_one = q_2_1;
            unsigned int sm_floor = ((unsigned int)num_ctas * 55 + 99) / 100;
            float _rcp_5 = approx_rcp((float)(100 * num_ctas));
            float ctas100_rcp = _rcp_5;
            unsigned int sw_den = 145;
            float _rcp_6 = approx_rcp((float)sw_den);
            float sw_rcp = _rcp_6;
            if (l_one < 2) {
                l_one = 2;
            }
            unsigned int fit_valid = 0;
            unsigned int l_fit = p_max;
            unsigned int whole_items_u = (unsigned int)(batch_size * items_per_chunk);
            unsigned int dense_u = 0;
            unsigned int fit_search = 0;
            if (whole_items_u <= (unsigned int)num_ctas) {
                if (p_max < 8 * (p_max - p_min)) {
                    fit_search = 1;
                }
            }
            if (fit_search != 0) {
                unsigned int l_hi = p_max;
                if (whole_items_u < (unsigned int)num_ctas) {
                    unsigned int denom_f = (unsigned int)num_ctas - whole_items_u;
                    float _rcp_7 = approx_rcp((float)denom_f);
                    unsigned int q_0_1 = (unsigned int)((float)total_work * _rcp_7);
                    if (total_work < q_0_1 * denom_f) {
                        q_0_1 = q_0_1 - 1;
                    }
                    if (total_work >= (q_0_1 + 1) * denom_f) {
                        q_0_1 = q_0_1 + 1;
                    }
                    if (total_work > q_0_1 * denom_f) {
                        q_0_1 = q_0_1 + 1;
                    }
                    unsigned int l_hi_f = q_0_1;
                    if (l_hi_f < p_max) {
                        l_hi = l_hi_f;
                    }
                }
                if (l_hi < l_one) {
                    l_hi = l_one;
                }
                unsigned int step_f = (l_hi - l_one + 31 - 1) / 31;
                if (step_f < 1) {
                    step_f = 1;
                }
                unsigned int l_lane = l_one + (unsigned int)lane_0 * step_f;
                if (l_lane > l_hi) {
                    l_lane = l_hi;
                }
                float _rcp_8 = approx_rcp((float)l_lane);
                float lane_rcp = _rcp_8;
                unsigned int t_lane = 0;
                unsigned int sp_lane = 0;
                unsigned int nm_lane = 0;
                #pragma unroll 1
                for (int bf = 0; bf < batch_size; bf++) {
                    int sf = sched_seq_lens[bf];
                    int _max_2 = ((sf) > (1) ? (sf) : (1));
                    unsigned int pairs_f = (unsigned int)((_max_2 + 255) / 256);
                    unsigned int q_0_2 = (unsigned int)((float)pairs_f * lane_rcp);
                    if (pairs_f < q_0_2 * l_lane) {
                        q_0_2 = q_0_2 - 1;
                    }
                    if (pairs_f >= (q_0_2 + 1) * l_lane) {
                        q_0_2 = q_0_2 + 1;
                    }
                    if (pairs_f > q_0_2 * l_lane) {
                        q_0_2 = q_0_2 + 1;
                    }
                    unsigned int nb_f = q_0_2;
                    t_lane += nb_f;
                }
                t_lane = t_lane * (unsigned int)items_per_chunk;
                unsigned int ok_key = 32;
                if (t_lane <= (unsigned int)num_ctas) {
                    ok_key = (unsigned int)lane_0;
                }
                unsigned int _warp_redux_u32_3;
                asm volatile("redux.sync.min.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_3) : "r"(ok_key));
                unsigned int first_ok = _warp_redux_u32_3;
                if (first_ok < 32) {
                    unsigned int _shfl_0 = __shfl_sync(0xFFFFFFFF, l_lane, (int)first_ok);
                    l_fit = _shfl_0;
                    fit_valid = 1;
                }
            }
            int cand_idx = lane_0 & 15;
            int req_parity = lane_0 >> 4;
            unsigned int cand = chunk_pairs_u;
            unsigned int cand_valid = 0;
            if (cand_idx < 13) {
                cand_valid = 1;
            }
            if (cand_idx == 9) {
                cand = p_max;
            }
            if (cand_idx > 9) {
                if (cand_idx < 13) {
                    cand = l_one * (unsigned int)(cand_idx - 8);
                    if (cand >= p_max) {
                        cand_valid = 0;
                    }
                }
            }
            if (cand_idx < 8) {
                unsigned int div_c = (unsigned int)(cand_idx + 1) * (unsigned int)num_ctas;
                float _rcp_9 = approx_rcp((float)div_c);
                unsigned int q_0_3 = (unsigned int)((float)total_work * _rcp_9);
                if (total_work < q_0_3 * div_c) {
                    q_0_3 = q_0_3 - 1;
                }
                if (total_work >= (q_0_3 + 1) * div_c) {
                    q_0_3 = q_0_3 + 1;
                }
                if (total_work > q_0_3 * div_c) {
                    q_0_3 = q_0_3 + 1;
                }
                cand = q_0_3;
                if (cand < 2) {
                    cand = 2;
                }
            }
            if (cand_idx == 13) {
                cand = l_fit;
                cand_valid = fit_valid;
            }
            float _rcp_10 = approx_rcp((float)cand);
            float cand_rcp = _rcp_10;
            unsigned int tickets_c = 0;
            unsigned int nmax_c = 0;
            unsigned int splits_c = 0;
            int half_batch = (batch_size + 1) / 2;
            #pragma unroll 1
            for (int hc = 0; hc < half_batch; hc++) {
                int bc = 2 * hc + req_parity;
                unsigned int pairs_c = 0;
                if (bc < batch_size) {
                    int sc = sched_seq_lens[bc];
                    int _max_4 = ((sc) > (1) ? (sc) : (1));
                    pairs_c = (unsigned int)((_max_4 + 255) / 256);
                }
                unsigned int q_0_4 = (unsigned int)((float)pairs_c * cand_rcp);
                if (pairs_c < q_0_4 * cand) {
                    q_0_4 = q_0_4 - 1;
                }
                if (pairs_c >= (q_0_4 + 1) * cand) {
                    q_0_4 = q_0_4 + 1;
                }
                if (pairs_c > q_0_4 * cand) {
                    q_0_4 = q_0_4 + 1;
                }
                unsigned int nb_c = q_0_4;
                tickets_c += nb_c;
                unsigned int _max_5 = ((nmax_c) > (nb_c) ? (nmax_c) : (nb_c));
                nmax_c = _max_5;
                if (nb_c > 2) {
                    splits_c += 1;
                }
            }
            unsigned int _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, tickets_c, 16);
            tickets_c += _shfl_xor_0;
            unsigned int _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, nmax_c, 16);
            unsigned int _max_6 = ((nmax_c) > (_shfl_xor_1) ? (nmax_c) : (_shfl_xor_1));
            nmax_c = _max_6;
            unsigned int _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, splits_c, 16);
            splits_c += _shfl_xor_2;
            tickets_c = tickets_c * (unsigned int)items_per_chunk;
            unsigned int q_3 = (unsigned int)((float)tickets_c * ctas_rcp);
            if (tickets_c < q_3 * (unsigned int)num_ctas) {
                q_3 = q_3 - 1;
            }
            if (tickets_c >= (q_3 + 1) * (unsigned int)num_ctas) {
                q_3 = q_3 + 1;
            }
            if (tickets_c > q_3 * (unsigned int)num_ctas) {
                q_3 = q_3 + 1;
            }
            unsigned int waves_c = q_3;
            unsigned int tail_c = 0;
            unsigned int red_c = 0;
            if (nmax_c == 2) {
                tail_c = 8;
            }
            if (nmax_c > 2) {
                unsigned int sh_c = 0;
                if (nmax_c > 4) {
                    sh_c = 1;
                }
                if (nmax_c > 8) {
                    sh_c = 2;
                }
                if (nmax_c > 16) {
                    sh_c = 3;
                }
                unsigned int one_c = 1;
                tail_c = 11 + (nmax_c + (one_c << sh_c) - 1 >> sh_c);
                red_c = splits_c * (unsigned int)items_per_chunk << sh_c;
            }
            unsigned int a_last_c = tickets_c - (waves_c - 1) * (unsigned int)num_ctas;
            unsigned int _max_7 = ((a_last_c) > (sm_floor) ? (a_last_c) : (sm_floor));
            unsigned int eff_c = _max_7;
            unsigned int late_c = 0;
            unsigned int cost_c = 4 * (waves_c - 1) * (cand + 20) + tail_c + late_c;
            unsigned int q_4 = (unsigned int)((float)(4 * cand * eff_c) * ctas_rcp);
            if (q_4 * (unsigned int)num_ctas > 4 * cand * eff_c) {
                q_4 = q_4 - 1;
            }
            if ((q_4 + 1) * (unsigned int)num_ctas <= 4 * cand * eff_c) {
                q_4 = q_4 + 1;
            }
            cost_c += q_4 + 80;
            unsigned int q_5 = (unsigned int)((float)(4 * cand * 5 * a_last_c) * ctas100_rcp);
            if (q_5 * (100 * (unsigned int)num_ctas) > 4 * cand * 5 * a_last_c) {
                q_5 = q_5 - 1;
            }
            if ((q_5 + 1) * (100 * (unsigned int)num_ctas) <= 4 * cand * 5 * a_last_c) {
                q_5 = q_5 + 1;
            }
            cost_c += q_5;
            unsigned int cost_key = 4294967295;
            if (cand_valid == 1) {
                if (req_parity == 0) {
                    cost_key = cost_c * 32 + (unsigned int)lane_0;
                }
            }
            unsigned int _warp_redux_u32_4;
            asm volatile("redux.sync.min.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_4) : "r"(cost_key));
            unsigned int best_key = _warp_redux_u32_4;
            int best_lane = (int)(best_key & 31);
            unsigned int _shfl_1 = __shfl_sync(0xFFFFFFFF, cand, best_lane);
            chunk_pairs_u = _shfl_1;
            int chunk_pairs = (int)chunk_pairs_u;
            float _rcp_11 = approx_rcp((float)chunk_pairs_u);
            float chunk_rcp = _rcp_11;
            unsigned int work_stage_sched = 0;
            unsigned int fast_tok = 0;
            unsigned int _phase_work_empty = 1;
            if (uniform_u == 1) {
                if (whole_items <= num_ctas) {
                    unsigned int q_0_5 = (unsigned int)((float)p_max * chunk_rcp);
                    if (p_max < q_0_5 * chunk_pairs_u) {
                        q_0_5 = q_0_5 - 1;
                    }
                    if (p_max >= (q_0_5 + 1) * chunk_pairs_u) {
                        q_0_5 = q_0_5 + 1;
                    }
                    if (p_max > q_0_5 * chunk_pairs_u) {
                        q_0_5 = q_0_5 + 1;
                    }
                    unsigned int ft_n_u = q_0_5;
                    unsigned int ft_full_u = ft_n_u;
                    if (p_max < ft_n_u * chunk_pairs_u) {
                        ft_full_u = ft_n_u - 1;
                    }
                    unsigned int ft_chunk_items = (unsigned int)batch_size * ft_n_u * (unsigned int)items_per_chunk;
                    unsigned int ft_total = ft_chunk_items;
                    if (ft_n_u > 2) {
                        unsigned int ft_shift = 0;
                        if (ft_n_u > 4) {
                            ft_shift = 1;
                        }
                        if (ft_n_u > 8) {
                            ft_shift = 2;
                        }
                        if (ft_n_u > 16) {
                            ft_shift = 3;
                        }
                        ft_total = ft_chunk_items + ((unsigned int)batch_size * (unsigned int)items_per_chunk << ft_shift);
                    }
                    if (ft_total <= (unsigned int)num_ctas) {
                        unsigned int ft_ticket = blockIdx.x;
                        if (ft_ticket < ft_chunk_items) {
                            fast_tok = 1;
                            mbarrier_wait(work_empty_addr + (work_stage_sched) * 8, _phase_work_empty);
                            unsigned int ft_base = work_stage_sched * 16;
                            unsigned int ft_b0_end = (unsigned int)batch_size * ft_full_u * (unsigned int)items_per_chunk;
                            unsigned int ft_den = ft_full_u;
                            if (ft_den < 1) {
                                ft_den = 1;
                            }
                            float _rcp_12 = approx_rcp((float)ft_den);
                            float ft_den_rcp = _rcp_12;
                            unsigned int ft_local = ft_ticket;
                            unsigned int ft_lchunk = 0;
                            int ft_batch = 0;
                            int ft_cidx = 0;
                            if (ft_ticket < ft_b0_end) {
                                unsigned int q_6 = (unsigned int)((float)ft_local * items_rcp);
                                if (ft_local < q_6 * (unsigned int)items_per_chunk) {
                                    q_6 = q_6 - 1;
                                }
                                if (ft_local >= (q_6 + 1) * (unsigned int)items_per_chunk) {
                                    q_6 = q_6 + 1;
                                }
                                ft_lchunk = q_6;
                                unsigned int q_7 = (unsigned int)((float)ft_lchunk * ft_den_rcp);
                                if (ft_lchunk < q_7 * ft_den) {
                                    q_7 = q_7 - 1;
                                }
                                if (ft_lchunk >= (q_7 + 1) * ft_den) {
                                    q_7 = q_7 + 1;
                                }
                                ft_batch = (int)q_7;
                                ft_cidx = (int)(ft_lchunk - (unsigned int)ft_batch * ft_den);
                            } else {
                                ft_local = ft_ticket - ft_b0_end;
                                unsigned int q_6_1 = (unsigned int)((float)ft_local * items_rcp);
                                if (ft_local < q_6_1 * (unsigned int)items_per_chunk) {
                                    q_6_1 = q_6_1 - 1;
                                }
                                if (ft_local >= (q_6_1 + 1) * (unsigned int)items_per_chunk) {
                                    q_6_1 = q_6_1 + 1;
                                }
                                ft_lchunk = q_6_1;
                                ft_batch = (int)ft_lchunk;
                                ft_cidx = (int)ft_full_u;
                            }
                            int ft_head = (int)(ft_local - ft_lchunk * (unsigned int)items_per_chunk);
                            int ft_seqlen = sched_seq_lens[ft_batch];
                            int ft_phase = sched_phase[ft_batch];
                            int _max_8 = (((ft_seqlen + BLOCK_N - 1) / BLOCK_N) > (1) ? ((ft_seqlen + BLOCK_N - 1) / BLOCK_N) : (1));
                            int ft_nblk = _max_8;
                            int ft_n = (int)ft_n_u;
                            int ft_bbeg = 2 * ft_cidx * chunk_pairs;
                            int ft_bend = 2 * (ft_cidx + 1) * chunk_pairs;
                            if (ft_cidx + 1 == ft_n) {
                                ft_bend = ft_nblk;
                            }
                            int ft_slot = 0;
                            int ft_ctr = 0;
                            if (ft_n > 1) {
                                ft_slot = ft_batch * ft_n * items_per_chunk + ft_head;
                                ft_ctr = ft_batch * items_per_chunk + ft_head;
                            }
                            if (lane_0 == 0) {
                                work_token_words[ft_base + 1] = 0;
                                work_token_words[ft_base + 2] = (unsigned int)ft_batch;
                                work_token_words[ft_base + 3] = (unsigned int)ft_head;
                                work_token_words[ft_base + 4] = (unsigned int)ft_bbeg;
                                work_token_words[ft_base + 5] = (unsigned int)ft_bend;
                                work_token_words[ft_base + 6] = (unsigned int)ft_seqlen;
                                work_token_words[ft_base + 7] = (unsigned int)ft_n;
                                work_token_words[ft_base + 8] = (unsigned int)ft_slot;
                                work_token_words[ft_base + 9] = (unsigned int)ft_ctr;
                                work_token_words[ft_base + 10] = (unsigned int)ft_cidx;
                                work_token_words[ft_base + 11] = (unsigned int)ft_phase;
                                work_token_words[ft_base] = 1;
                                mbarrier_arrive(work_full_addr + (work_stage_sched) * 8);
                            }
                            work_stage_sched += 1;
                            if (work_stage_sched == 4) { work_stage_sched = 0; _phase_work_empty ^= 1; }
                        }
                    }
                }
            }
            unsigned int n_full_chunks = 0;
            unsigned int n_chunks_total = 0;
            unsigned int n_split_requests = 0;
            unsigned int nmax_split = 0;
            unsigned int rem_bucket_total[4];
            #pragma unroll
            for (int bi = 0; bi < 4; bi++) {
                rem_bucket_total[bi] = 0;
            }
            unsigned int n_u = 0;
            unsigned int full_u = 0;
            int rb_u = 0;
            if (uniform_u == 1) {
                unsigned int q_0_6 = (unsigned int)((float)p_max * chunk_rcp);
                if (p_max < q_0_6 * chunk_pairs_u) {
                    q_0_6 = q_0_6 - 1;
                }
                if (p_max >= (q_0_6 + 1) * chunk_pairs_u) {
                    q_0_6 = q_0_6 + 1;
                }
                if (p_max > q_0_6 * chunk_pairs_u) {
                    q_0_6 = q_0_6 + 1;
                }
                n_u = q_0_6;
                full_u = n_u;
                if (p_max < n_u * chunk_pairs_u) {
                    full_u = n_u - 1;
                    unsigned int rem_u = p_max - full_u * chunk_pairs_u;
                    rb_u = 1;
                    #pragma unroll
                    for (int _rbu = 0; _rbu < 2; _rbu++) {
                        if (chunk_pairs_u > rem_u << (unsigned int)rb_u) {
                            rb_u = rb_u + 1;
                        }
                    }
                }
                n_full_chunks = (unsigned int)batch_size * full_u;
                n_chunks_total = (unsigned int)batch_size * n_u;
                if (n_u > 2) {
                    n_split_requests = (unsigned int)batch_size;
                    nmax_split = n_u;
                }
                #pragma unroll
                for (int bi_1 = 1; bi_1 < 4; bi_1++) {
                    if (rb_u == bi_1) {
                        rem_bucket_total[bi_1] = (unsigned int)batch_size;
                    }
                }
            } else {
                #pragma unroll 1
                for (int g2 = 0; g2 < num_groups; g2++) {
                    int b2 = g2 * 32 + lane_0;
                    unsigned int full2 = 0;
                    unsigned int n2 = 0;
                    unsigned int pack2 = 0;
                    unsigned int split2 = 0;
                    unsigned int nsplit2 = 0;
                    if (b2 < batch_size) {
                        int s2 = sched_seq_lens[b2];
                        int _max_9 = ((s2) > (1) ? (s2) : (1));
                        unsigned int pairs2 = (unsigned int)((_max_9 + 255) / 256);
                        unsigned int q_0_7 = (unsigned int)((float)pairs2 * chunk_rcp);
                        if (pairs2 < q_0_7 * chunk_pairs_u) {
                            q_0_7 = q_0_7 - 1;
                        }
                        if (pairs2 >= (q_0_7 + 1) * chunk_pairs_u) {
                            q_0_7 = q_0_7 + 1;
                        }
                        if (pairs2 > q_0_7 * chunk_pairs_u) {
                            q_0_7 = q_0_7 + 1;
                        }
                        n2 = q_0_7;
                        full2 = n2;
                        if (pairs2 < n2 * chunk_pairs_u) {
                            full2 = n2 - 1;
                            unsigned int rem2 = pairs2 - full2 * chunk_pairs_u;
                            int rb2 = 1;
                            #pragma unroll
                            for (int _rb = 0; _rb < 2; _rb++) {
                                if (chunk_pairs_u > rem2 << (unsigned int)rb2) {
                                    rb2 = rb2 + 1;
                                }
                            }
                            pack2 = (unsigned int)(1 << 8 * (rb2 - 1));
                        }
                        if (n2 > 2) {
                            split2 = 1;
                            nsplit2 = n2;
                        }
                    }
                    unsigned int _warp_redux_u32_5;
                    asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_5) : "r"(full2));
                    n_full_chunks += _warp_redux_u32_5;
                    unsigned int _warp_redux_u32_6;
                    asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_6) : "r"(n2));
                    n_chunks_total += _warp_redux_u32_6;
                    unsigned int _warp_redux_u32_7;
                    asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_7) : "r"(split2));
                    n_split_requests += _warp_redux_u32_7;
                    unsigned int _warp_redux_u32_8;
                    asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_8) : "r"(nsplit2));
                    unsigned int _max_10 = ((nmax_split) > (_warp_redux_u32_8) ? (nmax_split) : (_warp_redux_u32_8));
                    nmax_split = _max_10;
                    unsigned int _warp_redux_u32_9;
                    asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_9) : "r"(pack2));
                    unsigned int pack_group = _warp_redux_u32_9;
                    #pragma unroll
                    for (int bi_2 = 1; bi_2 < 4; bi_2++) {
                        rem_bucket_total[bi_2] = rem_bucket_total[bi_2] + (pack_group >> (unsigned int)(8 * (bi_2 - 1)) & 255);
                    }
                }
            }
            unsigned int full_den_u = full_u;
            if (full_den_u < 1) {
                full_den_u = 1;
            }
            float _rcp_13 = approx_rcp((float)full_den_u);
            float full_rcp_u = _rcp_13;
            unsigned int bucket_end[4];
            bucket_end[0] = n_full_chunks * (unsigned int)items_per_chunk;
            #pragma unroll
            for (int bi_3 = 1; bi_3 < 4; bi_3++) {
                bucket_end[bi_3] = bucket_end[bi_3 - 1] + rem_bucket_total[bi_3] * (unsigned int)items_per_chunk;
            }
            unsigned int chunk_items = n_chunks_total * (unsigned int)items_per_chunk;
            unsigned int reduce_shift = 0;
            if (nmax_split > 4) {
                reduce_shift = 1;
            }
            if (nmax_split > 8) {
                reduce_shift = 2;
            }
            if (nmax_split > 16) {
                reduce_shift = 3;
            }
            unsigned int reduce_items = n_split_requests * (unsigned int)items_per_chunk << reduce_shift;
            unsigned int total_items = chunk_items + reduce_items;
            if (blockIdx.x == 0) {
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                if (lane_0 == 0) {
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                    work_token_words[work_stage_sched * 16] = 0;
#else
                    int plan_off = num_ctas * 2048;
                    *(reinterpret_cast<float*>(partial_stats + plan_off) + (0)) = (float)chunk_pairs_u;
                    *(reinterpret_cast<float*>(partial_stats + (plan_off + 1)) + (0)) = (float)total_items;
                }
            }
            if (lane_0 == 0) {
            }
            int cur_group_b[4];
            unsigned int before_b[4];
            unsigned int si_before_b[4];
            unsigned int st_before_b[4];
            #pragma unroll
            for (int bi_4 = 0; bi_4 < 4; bi_4++) {
                cur_group_b[bi_4] = 0;
                before_b[bi_4] = 0;
                si_before_b[bi_4] = 0;
                st_before_b[bi_4] = 0;
            }
            unsigned int first_claim = 1;
            if (fast_tok == 1) {
                first_claim = 0;
            }
            unsigned int gate_phase = 0;
            #pragma unroll 1
            for (unsigned int _claim = 0; _claim < max_items + 1; _claim++) {
                mbarrier_wait(work_empty_addr + (work_stage_sched) * 8, _phase_work_empty);
                unsigned int ticket_lane0 = blockIdx.x;
                if (first_claim == 0) {
                    if (total_items <= (unsigned int)num_ctas) {
                        ticket_lane0 = (unsigned int)num_ctas;
                    } else {
                        mbarrier_wait_hint(claim_gate_addr, gate_phase, 1000);
                        gate_phase = gate_phase ^ 1;
                        if (lane_0 == 0) {
                            unsigned int _atomic_old_0 = atomicAdd(&queue_counters[0], 1);
                            ticket_lane0 = _atomic_old_0 + (unsigned int)num_ctas;
                        }
                    }
                }
                first_claim = 0;
                unsigned int _shfl_2 = __shfl_sync(0xFFFFFFFF, ticket_lane0, 0);
                unsigned int ticket = _shfl_2;
                unsigned int token_base = work_stage_sched * 16;
                unsigned int valid_tok = ((ticket < total_items) ? 1 : 0);
                int counter_idx_r = 0;
                unsigned int dec_tok = valid_tok;
                unsigned int dec_ticket = ticket;
                if (dec_tok != 0) {
                    if (dec_ticket >= chunk_items) {
                        unsigned int r_red = dec_ticket - chunk_items;
                        unsigned int rt_idx = r_red >> reduce_shift;
                        unsigned int rq_slice = r_red - (rt_idx << reduce_shift);
                        int sel_batch_r = 0;
                        int sel_n_r = 0;
                        int kv_head_r = 0;
                        int slot_tile_base_r = 0;
                        if (uniform_u == 1) {
                            unsigned int q_0_8 = (unsigned int)((float)rt_idx * items_rcp);
                            if (rt_idx < q_0_8 * (unsigned int)items_per_chunk) {
                                q_0_8 = q_0_8 - 1;
                            }
                            if (rt_idx >= (q_0_8 + 1) * (unsigned int)items_per_chunk) {
                                q_0_8 = q_0_8 + 1;
                            }
                            sel_batch_r = (int)q_0_8;
                            kv_head_r = (int)(rt_idx - (unsigned int)sel_batch_r * (unsigned int)items_per_chunk);
                            sel_n_r = (int)n_u;
                            slot_tile_base_r = sel_batch_r * (int)n_u * items_per_chunk + kv_head_r;
                            counter_idx_r = sel_batch_r * items_per_chunk + kv_head_r;
                        } else {
                            unsigned int rt_before = 0;
                            unsigned int si_before_r = 0;
                            unsigned int st_before_r = 0;
                            int b_r = 0;
                            int n_r = 0;
                            unsigned int mine_r = 0;
                            unsigned int si_r = 0;
                            unsigned int st_r = 0;
                            unsigned int incl_r = 0;
                            unsigned int incl_si_r = 0;
                            unsigned int incl_st_r = 0;
                            #pragma unroll 1
                            for (int _adv_r = 0; _adv_r < 32; _adv_r++) {
                                b_r = _adv_r * 32 + lane_0;
                                n_r = 0;
                                mine_r = 0;
                                si_r = 0;
                                st_r = 0;
                                if (b_r < batch_size) {
                                    int s_r = sched_seq_lens[b_r];
                                    int _max_11 = ((s_r) > (1) ? (s_r) : (1));
                                    int pairs_r = (_max_11 + 255) / 256;
                                    unsigned int q_0_9 = (unsigned int)((float)(unsigned int)pairs_r * chunk_rcp);
                                    if (q_0_9 * chunk_pairs_u > (unsigned int)pairs_r) {
                                        q_0_9 = q_0_9 - 1;
                                    }
                                    if ((q_0_9 + 1) * chunk_pairs_u <= (unsigned int)pairs_r) {
                                        q_0_9 = q_0_9 + 1;
                                    }
                                    if (q_0_9 * chunk_pairs_u < (unsigned int)pairs_r) {
                                        q_0_9 = q_0_9 + 1;
                                    }
                                    n_r = (int)q_0_9;
                                    if (n_r > 1) {
                                        si_r = (unsigned int)n_r;
                                        st_r = 1;
                                    }
                                    if (n_r > 2) {
                                        mine_r = (unsigned int)items_per_chunk;
                                    }
                                }
                                uint32_t _warp_scan_sum_u32_0 = mine_r;
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_0) : "r"(1));
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_0) : "r"(2));
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_0) : "r"(4));
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_0) : "r"(8));
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_0) : "r"(16));
                                incl_r = _warp_scan_sum_u32_0;
                                uint32_t _warp_scan_sum_u32_1 = si_r;
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_1) : "r"(1));
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_1) : "r"(2));
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_1) : "r"(4));
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_1) : "r"(8));
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_1) : "r"(16));
                                incl_si_r = _warp_scan_sum_u32_1;
                                uint32_t _warp_scan_sum_u32_2 = st_r;
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_2) : "r"(1));
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_2) : "r"(2));
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_2) : "r"(4));
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_2) : "r"(8));
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_2) : "r"(16));
                                incl_st_r = _warp_scan_sum_u32_2;
                                unsigned int _shfl_3 = __shfl_sync(0xFFFFFFFF, incl_r, 31);
                                unsigned int group_total_r = _shfl_3;
                                if (rt_idx < rt_before + group_total_r) {
                                    break;
                                }
                                rt_before = rt_before + group_total_r;
                                unsigned int _shfl_4 = __shfl_sync(0xFFFFFFFF, incl_si_r, 31);
                                si_before_r = si_before_r + _shfl_4;
                                unsigned int _shfl_5 = __shfl_sync(0xFFFFFFFF, incl_st_r, 31);
                                st_before_r = st_before_r + _shfl_5;
                            }
                            unsigned int in_group_r = rt_idx - rt_before;
                            unsigned int excl_r = incl_r - mine_r;
                            unsigned int excl_si_r = incl_si_r - si_r;
                            unsigned int excl_st_r = incl_st_r - st_r;
                            int hit_r = 0;
                            if (mine_r > 0) {
                                if (excl_r <= in_group_r) {
                                    if (in_group_r < incl_r) {
                                        hit_r = 1;
                                    }
                                }
                            }
                            unsigned int _vote_0 = __ballot_sync(0xFFFFFFFF, hit_r != 0);
                            unsigned int hit_mask_r = _vote_0;
                            int _ffs_0 = __ffs(hit_mask_r);
                            int hit_lane_r = _ffs_0 - 1;
                            int _shfl_6 = __shfl_sync(0xFFFFFFFF, b_r, hit_lane_r);
                            sel_batch_r = _shfl_6;
                            int _shfl_7 = __shfl_sync(0xFFFFFFFF, n_r, hit_lane_r);
                            sel_n_r = _shfl_7;
                            unsigned int _shfl_8 = __shfl_sync(0xFFFFFFFF, excl_r, hit_lane_r);
                            int sel_excl_r = (int)_shfl_8;
                            unsigned int _shfl_9 = __shfl_sync(0xFFFFFFFF, excl_si_r, hit_lane_r);
                            int sel_excl_si_r = (int)_shfl_9;
                            unsigned int _shfl_10 = __shfl_sync(0xFFFFFFFF, excl_st_r, hit_lane_r);
                            int sel_excl_st_r = (int)_shfl_10;
                            kv_head_r = (int)in_group_r - sel_excl_r;
                            slot_tile_base_r = ((int)si_before_r + sel_excl_si_r) * items_per_chunk + kv_head_r;
                            counter_idx_r = ((int)st_before_r + sel_excl_st_r) * items_per_chunk + kv_head_r;
                        }
                        if (lane_0 == 0) {
                            unsigned int arrived_r = 0;
                            #pragma unroll 1
                            for (int _poll_r = 0; _poll_r < 1073741824; _poll_r++) {
                                {
                                    unsigned int _atomic_old_2;
                                    asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                                        : "=r"(_atomic_old_2) : "l"(&tile_counters[counter_idx_r * 4]), "r"(static_cast<uint32_t>(0)) : "memory");
                                    arrived_r = _atomic_old_2;
                                }
                                if ((int)arrived_r == sel_n_r) {
                                    break;
                                }
                            }
                            work_token_words[token_base + 1] = 1;
                            work_token_words[token_base + 2] = (unsigned int)sel_batch_r;
                            work_token_words[token_base + 3] = (unsigned int)kv_head_r;
                            work_token_words[token_base + 7] = (unsigned int)sel_n_r;
                            work_token_words[token_base + 8] = (unsigned int)slot_tile_base_r;
                            work_token_words[token_base + 4] = rq_slice;
                            work_token_words[token_base + 5] = reduce_shift;
                            work_token_words[token_base + 6] = 0;
                            work_token_words[token_base + 9] = (unsigned int)counter_idx_r;
                            work_token_words[token_base + 10] = 0;
                            work_token_words[token_base + 11] = 0;
                        }
                    } else {
                        int bucket = 0;
                        unsigned int bucket_start = 0;
                        #pragma unroll
                        for (int bi_5 = 0; bi_5 < 4; bi_5++) {
                            if (bucket_end[bi_5] <= dec_ticket) {
                                bucket = bi_5 + 1;
                                bucket_start = bucket_end[bi_5];
                            }
                        }
                        unsigned int local_items = dec_ticket - bucket_start;
                        unsigned int q_0_10 = (unsigned int)((float)local_items * items_rcp);
                        if (local_items < q_0_10 * (unsigned int)items_per_chunk) {
                            q_0_10 = q_0_10 - 1;
                        }
                        if (local_items >= (q_0_10 + 1) * (unsigned int)items_per_chunk) {
                            q_0_10 = q_0_10 + 1;
                        }
                        unsigned int local_chunk = q_0_10;
                        int in_chunk = (int)(local_items - local_chunk * (unsigned int)items_per_chunk);
                        int sel_batch = 0;
                        int sel_seqlen = 0;
                        int sel_phase = 0;
                        int sel_n = 0;
                        int sel_full = 0;
                        int chunk_idx_s = 0;
                        int si_base_s = 0;
                        int st_base_s = 0;
                        if (uniform_u == 1) {
                            if (bucket == 0) {
                                unsigned int q_6_2 = (unsigned int)((float)local_chunk * full_rcp_u);
                                if (local_chunk < q_6_2 * full_den_u) {
                                    q_6_2 = q_6_2 - 1;
                                }
                                if (local_chunk >= (q_6_2 + 1) * full_den_u) {
                                    q_6_2 = q_6_2 + 1;
                                }
                                sel_batch = (int)q_6_2;
                                chunk_idx_s = (int)(local_chunk - (unsigned int)sel_batch * full_den_u);
                            } else {
                                sel_batch = (int)local_chunk;
                                chunk_idx_s = (int)full_u;
                            }
                            sel_seqlen = sched_seq_lens[sel_batch];
                            sel_phase = sched_phase[sel_batch];
                            sel_n = (int)n_u;
                            if (n_u > 1) {
                                si_base_s = sel_batch * (int)n_u;
                                st_base_s = sel_batch;
                            }
                        } else {
                            int cursor_group = 0;
                            unsigned int before = 0;
                            unsigned int si_before = 0;
                            unsigned int st_before = 0;
                            #pragma unroll
                            for (int bi_6 = 0; bi_6 < 4; bi_6++) {
                                if (bucket == bi_6) {
                                    cursor_group = cur_group_b[bi_6];
                                    before = before_b[bi_6];
                                    si_before = si_before_b[bi_6];
                                    st_before = st_before_b[bi_6];
                                }
                            }
                            int s3 = 0;
                            int ph3 = 0;
                            int pairs3 = 0;
                            int n3 = 0;
                            int fullc3 = 0;
                            unsigned int mine3 = 0;
                            unsigned int split_items3 = 0;
                            unsigned int split_tiles3 = 0;
                            unsigned int incl3 = 0;
                            unsigned int incl_si3 = 0;
                            unsigned int incl_st3 = 0;
                            int b3 = 0;
                            #pragma unroll 1
                            for (int _adv = 0; _adv < 32; _adv++) {
                                b3 = cursor_group * 32 + lane_0;
                                s3 = 0;
                                ph3 = 0;
                                pairs3 = 0;
                                n3 = 0;
                                fullc3 = 0;
                                mine3 = 0;
                                split_items3 = 0;
                                split_tiles3 = 0;
                                if (b3 < batch_size) {
                                    s3 = sched_seq_lens[b3];
                                    ph3 = sched_phase[b3];
                                    int _max_12 = ((s3) > (1) ? (s3) : (1));
                                    pairs3 = (_max_12 + 255) / 256;
                                    unsigned int q_6_3 = (unsigned int)((float)(unsigned int)pairs3 * chunk_rcp);
                                    if (q_6_3 * chunk_pairs_u > (unsigned int)pairs3) {
                                        q_6_3 = q_6_3 - 1;
                                    }
                                    if ((q_6_3 + 1) * chunk_pairs_u <= (unsigned int)pairs3) {
                                        q_6_3 = q_6_3 + 1;
                                    }
                                    if (q_6_3 * chunk_pairs_u < (unsigned int)pairs3) {
                                        q_6_3 = q_6_3 + 1;
                                    }
                                    n3 = (int)q_6_3;
                                    fullc3 = n3;
                                    int rem3 = 0;
                                    if (pairs3 < n3 * chunk_pairs) {
                                        fullc3 = n3 - 1;
                                        rem3 = pairs3 - fullc3 * chunk_pairs;
                                    }
                                    if (n3 > 1) {
                                        split_items3 = (unsigned int)n3;
                                        split_tiles3 = 1;
                                    }
                                    if (bucket == 0) {
                                        mine3 = (unsigned int)fullc3;
                                    } else if (rem3 > 0) {
                                        int rb3 = 1;
                                        #pragma unroll
                                        for (int _rb3 = 0; _rb3 < 2; _rb3++) {
                                            if (chunk_pairs > rem3 << rb3) {
                                                rb3 = rb3 + 1;
                                            }
                                        }
                                        if (rb3 == bucket) {
                                            mine3 = 1;
                                        }
                                    }
                                }
                                uint32_t _warp_scan_sum_u32_3 = mine3;
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_3) : "r"(1));
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_3) : "r"(2));
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_3) : "r"(4));
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_3) : "r"(8));
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_3) : "r"(16));
                                incl3 = _warp_scan_sum_u32_3;
                                uint32_t _warp_scan_sum_u32_4 = split_items3;
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_4) : "r"(1));
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_4) : "r"(2));
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_4) : "r"(4));
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_4) : "r"(8));
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_4) : "r"(16));
                                incl_si3 = _warp_scan_sum_u32_4;
                                uint32_t _warp_scan_sum_u32_5 = split_tiles3;
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_5) : "r"(1));
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_5) : "r"(2));
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_5) : "r"(4));
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_5) : "r"(8));
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_5) : "r"(16));
                                incl_st3 = _warp_scan_sum_u32_5;
                                unsigned int _shfl_11 = __shfl_sync(0xFFFFFFFF, incl3, 31);
                                unsigned int group_total = _shfl_11;
                                if (local_chunk < before + group_total) {
                                    break;
                                }
                                before = before + group_total;
                                unsigned int _shfl_12 = __shfl_sync(0xFFFFFFFF, incl_si3, 31);
                                si_before = si_before + _shfl_12;
                                unsigned int _shfl_13 = __shfl_sync(0xFFFFFFFF, incl_st3, 31);
                                st_before = st_before + _shfl_13;
                                cursor_group = cursor_group + 1;
                            }
                            #pragma unroll
                            for (int bi_7 = 0; bi_7 < 4; bi_7++) {
                                if (bucket == bi_7) {
                                    cur_group_b[bi_7] = cursor_group;
                                    before_b[bi_7] = before;
                                    si_before_b[bi_7] = si_before;
                                    st_before_b[bi_7] = st_before;
                                }
                            }
                            unsigned int in_group = local_chunk - before;
                            unsigned int excl3 = incl3 - mine3;
                            unsigned int excl_si3 = incl_si3 - split_items3;
                            unsigned int excl_st3 = incl_st3 - split_tiles3;
                            int hit3 = 0;
                            if (mine3 > 0) {
                                if (excl3 <= in_group) {
                                    if (in_group < incl3) {
                                        hit3 = 1;
                                    }
                                }
                            }
                            unsigned int _vote_1 = __ballot_sync(0xFFFFFFFF, hit3 != 0);
                            unsigned int hit_mask = _vote_1;
                            int _ffs_1 = __ffs(hit_mask);
                            int hit_lane = _ffs_1 - 1;
                            int _shfl_14 = __shfl_sync(0xFFFFFFFF, b3, hit_lane);
                            sel_batch = _shfl_14;
                            int _shfl_15 = __shfl_sync(0xFFFFFFFF, s3, hit_lane);
                            sel_seqlen = _shfl_15;
                            int _shfl_16 = __shfl_sync(0xFFFFFFFF, ph3, hit_lane);
                            sel_phase = _shfl_16;
                            int _shfl_17 = __shfl_sync(0xFFFFFFFF, n3, hit_lane);
                            sel_n = _shfl_17;
                            int _shfl_18 = __shfl_sync(0xFFFFFFFF, fullc3, hit_lane);
                            sel_full = _shfl_18;
                            unsigned int _shfl_19 = __shfl_sync(0xFFFFFFFF, excl3, hit_lane);
                            int sel_excl = (int)_shfl_19;
                            unsigned int _shfl_20 = __shfl_sync(0xFFFFFFFF, excl_si3, hit_lane);
                            int sel_excl_si = (int)_shfl_20;
                            unsigned int _shfl_21 = __shfl_sync(0xFFFFFFFF, excl_st3, hit_lane);
                            int sel_excl_st = (int)_shfl_21;
                            chunk_idx_s = sel_full;
                            if (bucket == 0) {
                                chunk_idx_s = (int)in_group - sel_excl;
                            }
                            if (sel_n > 1) {
                                si_base_s = (int)si_before + sel_excl_si;
                                st_base_s = (int)st_before + sel_excl_st;
                            }
                        }
                        int chunk_idx = chunk_idx_s;
                        int kv_head_sel = in_chunk;
                        int _max_13 = (((sel_seqlen + BLOCK_N - 1) / BLOCK_N) > (1) ? ((sel_seqlen + BLOCK_N - 1) / BLOCK_N) : (1));
                        int n_blocks_tile = _max_13;
                        int block_begin_4 = 2 * chunk_idx * chunk_pairs;
                        int block_end_4 = 2 * (chunk_idx + 1) * chunk_pairs;
                        if (chunk_idx + 1 == sel_n) {
                            block_end_4 = n_blocks_tile;
                        }
                        int slot_tile_base_4 = 0;
                        int counter_idx_4 = 0;
                        if (sel_n > 1) {
                            slot_tile_base_4 = si_base_s * items_per_chunk + kv_head_sel;
                            counter_idx_4 = st_base_s * items_per_chunk + kv_head_sel;
                        }
                        if (valid_tok != 0) {
                            if (lane_0 == 0) {
                                work_token_words[token_base + 1] = 0;
                                work_token_words[token_base + 2] = (unsigned int)sel_batch;
                                work_token_words[token_base + 3] = (unsigned int)kv_head_sel;
                                work_token_words[token_base + 4] = (unsigned int)block_begin_4;
                                work_token_words[token_base + 5] = (unsigned int)block_end_4;
                                work_token_words[token_base + 6] = (unsigned int)sel_seqlen;
                                work_token_words[token_base + 7] = (unsigned int)sel_n;
                                work_token_words[token_base + 8] = (unsigned int)slot_tile_base_4;
                                work_token_words[token_base + 9] = (unsigned int)counter_idx_4;
                                work_token_words[token_base + 10] = (unsigned int)chunk_idx;
                                work_token_words[token_base + 11] = (unsigned int)sel_phase;
                            }
                        }
                    }
                }
                if (lane_0 == 0) {
                    work_token_words[token_base] = valid_tok;
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                    mbarrier_arrive(work_full_addr + (work_stage_sched) * 8);
#if CAKE_FMHA_DCP_PROGRAM == 1
                    if (valid_tok != 0) {
                        if (ticket >= chunk_items) {
                            unsigned int one_r = 1;
                            unsigned int slices_r = one_r << reduce_shift;
                            unsigned int _atomic_old_3;
                            asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                                : "=r"(_atomic_old_3) : "l"(&tile_counters[counter_idx_r * 4 + 1]), "r"(static_cast<uint32_t>(1)) : "memory");
                            unsigned int pops_old = _atomic_old_3;
                            if (pops_old + 1 == slices_r) {
                                *(reinterpret_cast<unsigned int*>(tile_counters + (counter_idx_r * 4)) + (0)) = 0;
                                *(reinterpret_cast<unsigned int*>(tile_counters + (counter_idx_r * 4 + 1)) + (0)) = 0;
                            }
                        }
                    }
#endif  // CAKE_FMHA_DCP_PROGRAM == 1
                }
                work_stage_sched += 1;
                if (work_stage_sched == 4) { work_stage_sched = 0; _phase_work_empty ^= 1; }
#if CAKE_FMHA_DCP_PROGRAM == 1
                if (valid_tok == 0) {
                    break;
                }
#endif  // CAKE_FMHA_DCP_PROGRAM == 1
            }
            unsigned int done_old = 0;
            if (lane_0 == 0) {
                uint32_t _atomic_inc_old_0;
                asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                    : "=r"(_atomic_inc_old_0) : "l"(&queue_counters[1]), "r"(static_cast<uint32_t>(num_ctas - 1)) : "memory");
                done_old = _atomic_inc_old_0;
            }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
            unsigned int _shfl_0 = __shfl_sync(0xFFFFFFFF, done_old, 0);
            done_old = _shfl_0;
#else
            unsigned int _shfl_22 = __shfl_sync(0xFFFFFFFF, done_old, 0);
            done_old = _shfl_22;
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
            if ((int)done_old == num_ctas - 1) {
                if (lane_0 == 0) {
                    *(reinterpret_cast<unsigned int*>(queue_counters) + (0)) = 0;
                }
            }
        }
    // ---- Role: load_warp ----
    } else if (warp == 11) {
        { // load_warp_main
            if (lane == 0) {
            }
            unsigned int q_prod_stage = 0;
            unsigned int q_prod_phase = 1;
            unsigned int page_cons_stage = 0;
            unsigned int page_cons_phase = 0;
            unsigned int k_prod_stage = 0;
            unsigned int k_prod_phase = 1;
            unsigned int v_prod_stage = 0;
            unsigned int v_prod_phase = 1;
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 5
            if (ei_valid != 0) {
                int _min_0 = ((ei_cnt) < (3) ? (ei_cnt) : (3));
                int n_pre_e = _min_0;
                int _max_4 = (((ei_len + PAGE_SIZE - 1) / PAGE_SIZE - 1) > (0) ? ((ei_len + PAGE_SIZE - 1) / PAGE_SIZE - 1) : (0));
                int ei_max_pg = _max_4;
                int ei_page = 0;
                if (ei_row_ok == 0) {
                    int ei_blk = lane >> 3;
                    int ei_idx = (ei_last - ei_blk) * 8 + (lane & 7);
                    if (ei_idx > ei_max_pg) {
                        ei_idx = ei_max_pg;
                    }
                    if (ei_blk < n_pre_e) {
                        ei_page = page_table[ei_batch * max_pages_per_seq + ei_idx];
                    }
                }
                if (elect_sync()) {
                    mbarrier_wait(q_empty_addr + (q_prod_stage) * 8, q_prod_phase);
                    int rows_hi_e = ((N_ROWS > 32) ? 1 : 0);
                    if (rows_hi_e != 0) {
                        mbarrier_arrive_expect_tx(q_full_addr + (q_prod_stage) * 8, 64 * HEAD_DIM * 2);
                    } else {
                        mbarrier_arrive_expect_tx(q_full_addr + (q_prod_stage) * 8, 32 * HEAD_DIM * 2);
                    }
                    tma_4d_gmem2smem(smem_qt_addr + q_prod_stage * 16384, (&Q), 0, ei_head * 8, ei_batch * q_len, 0, q_full_addr + (q_prod_stage) * 8);
                    if (rows_hi_e != 0) {
                        tma_4d_gmem2smem(smem_qt_addr + q_prod_stage * 16384 + (unsigned int)(32 * HEAD_DIM * 2), (&Q), 0, ei_head * 8, ei_batch * q_len + 4, 0, q_full_addr + (q_prod_stage) * 8);
                    }
                }
                #pragma unroll 1
                for (int eb = 0; eb < n_pre_e; eb++) {
                    int pg_e[8];
                    if (ei_row_ok != 0) {
                        #pragma unroll
                        for (int pg_i = 0; pg_i < 8; pg_i++) {
                            int ep_e = (ei_last - eb) * 8 + pg_i;
                            if (ep_e > ei_max_pg) {
                                ep_e = ei_max_pg;
                            }
                            int ep_slot_e = ep_e >> 5;
                            int ep_val_e = ei_row[0];
                            #pragma unroll
                            for (int sl_e_2 = 1; sl_e_2 < 4; sl_e_2++) {
                                if (ep_slot_e == sl_e_2) {
                                    ep_val_e = ei_row[sl_e_2];
                                }
                            }
                            int _shfl_1 = __shfl_sync(0xFFFFFFFF, ep_val_e, ep_e & 31);
                            pg_e[pg_i] = _shfl_1;
                        }
                    } else {
                        #pragma unroll
                        for (int pg_i_1 = 0; pg_i_1 < 8; pg_i_1++) {
                            int _shfl_2 = __shfl_sync(0xFFFFFFFF, ei_page, eb * 8 + pg_i_1);
                            pg_e[pg_i_1] = _shfl_2;
                        }
                    }
                    if (elect_sync()) {
                        mbarrier_wait(k_empty_addr + (k_prod_stage) * 8, k_prod_phase);
                        mbarrier_arrive_expect_tx(k_full_addr + (k_prod_stage) * 8, 32768);
                        int kdst_e = smem_k_addr + k_prod_stage * 32768;
                        #pragma unroll
                        for (int pg_i_2 = 0; pg_i_2 < 8; pg_i_2++) {
                            int kpg_e = pg_e[pg_i_2];
                            #pragma unroll
                            for (int hg = 0; hg < 2; hg++) {
                                int ktoff_e = hg * 16384 + pg_i_2 * 2048;
                                tma_5d_gmem2smem(kdst_e + ktoff_e, (&K), 0, 0, hg, ei_head, kpg_e, k_full_addr + (k_prod_stage) * 8);
                            }
                        }
                        k_prod_stage += 1;
                        if (k_prod_stage == 3) { k_prod_stage = 0; k_prod_phase ^= 1; }
                        if (eb < 2) {
                            mbarrier_wait(v_empty_addr + (v_prod_stage) * 8, v_prod_phase);
                            mbarrier_arrive_expect_tx(v_full_addr + (v_prod_stage) * 8, 32768);
                            int vdst_e = smem_v_addr + v_prod_stage * 32768;
                            #pragma unroll
                            for (int pg_i_3 = 0; pg_i_3 < 8; pg_i_3++) {
                                int vpg_e = pg_e[pg_i_3];
                                #pragma unroll
                                for (int hg_1 = 0; hg_1 < 2; hg_1++) {
                                    int vtoff_e = hg_1 * 16384 + pg_i_3 * 2048;
                                    tma_5d_gmem2smem(vdst_e + vtoff_e, (&V), 0, 0, hg_1, ei_head, vpg_e, v_full_addr + (v_prod_stage) * 8);
                                }
                            }
                            v_prod_stage += 1;
                            if (v_prod_stage == 3) { v_prod_stage = 0; v_prod_phase ^= 1; }
                        }
                    }
                }
            }
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
#else
#if CAKE_FMHA_DCP_PROGRAM == 4
            if (ei_valid != 0) {
                int _min_0 = ((ei_cnt) < (3) ? (ei_cnt) : (3));
                int n_pre_e = _min_0;
                int _max_4 = (((ei_len + PAGE_SIZE - 1) / PAGE_SIZE - 1) > (0) ? ((ei_len + PAGE_SIZE - 1) / PAGE_SIZE - 1) : (0));
                int ei_max_pg = _max_4;
                int ei_page = 0;
                if (ei_row_ok == 0) {
                    int ei_blk = lane >> 3;
                    int ei_idx = (ei_last - ei_blk) * 8 + (lane & 7);
                    if (ei_idx > ei_max_pg) {
                        ei_idx = ei_max_pg;
                    }
                    if (ei_blk < n_pre_e) {
                        ei_page = page_table[ei_batch * max_pages_per_seq + ei_idx];
                    }
                }
                if (elect_sync()) {
                    mbarrier_wait(q_empty_addr + (q_prod_stage) * 8, q_prod_phase);
                    mbarrier_arrive_expect_tx(q_full_addr + (q_prod_stage) * 8, 64 * HEAD_DIM * 2);
                    tma_4d_gmem2smem(smem_qt_addr + q_prod_stage * 16384, (&Q), 0, ei_head * 8, ei_batch * q_len, 0, q_full_addr + (q_prod_stage) * 8);
                }
                #pragma unroll 1
                for (int eb = 0; eb < n_pre_e; eb++) {
                    int pg_e[8];
                    if (ei_row_ok != 0) {
                        #pragma unroll
                        for (int pg_i = 0; pg_i < 8; pg_i++) {
                            int ep_e = (ei_last - eb) * 8 + pg_i;
                            if (ep_e > ei_max_pg) {
                                ep_e = ei_max_pg;
                            }
                            int ep_slot_e = ep_e >> 5;
                            int ep_val_e = ei_row[0];
                            #pragma unroll
                            for (int sl_e_2 = 1; sl_e_2 < 4; sl_e_2++) {
                                if (ep_slot_e == sl_e_2) {
                                    ep_val_e = ei_row[sl_e_2];
                                }
                            }
                            int _shfl_1 = __shfl_sync(0xFFFFFFFF, ep_val_e, ep_e & 31);
                            pg_e[pg_i] = _shfl_1;
                        }
                    } else {
                        #pragma unroll
                        for (int pg_i_1 = 0; pg_i_1 < 8; pg_i_1++) {
                            int _shfl_2 = __shfl_sync(0xFFFFFFFF, ei_page, eb * 8 + pg_i_1);
                            pg_e[pg_i_1] = _shfl_2;
                        }
                    }
                    if (elect_sync()) {
                        mbarrier_wait(k_empty_addr + (k_prod_stage) * 8, k_prod_phase);
                        mbarrier_arrive_expect_tx(k_full_addr + (k_prod_stage) * 8, 32768);
                        int kdst_e = smem_k_addr + k_prod_stage * 32768;
                        #pragma unroll
                        for (int pg_i_2 = 0; pg_i_2 < 8; pg_i_2++) {
                            int kpg_e = pg_e[pg_i_2];
                            #pragma unroll
                            for (int hg = 0; hg < 2; hg++) {
                                int ktoff_e = hg * 16384 + pg_i_2 * 2048;
                                tma_5d_gmem2smem(kdst_e + ktoff_e, (&K), 0, 0, hg, ei_head, kpg_e, k_full_addr + (k_prod_stage) * 8);
                            }
                        }
                        k_prod_stage += 1;
                        if (k_prod_stage == 3) { k_prod_stage = 0; k_prod_phase ^= 1; }
                        if (eb < 2) {
                            mbarrier_wait(v_empty_addr + (v_prod_stage) * 8, v_prod_phase);
                            mbarrier_arrive_expect_tx(v_full_addr + (v_prod_stage) * 8, 32768);
                            int vdst_e = smem_v_addr + v_prod_stage * 32768;
                            #pragma unroll
                            for (int pg_i_3 = 0; pg_i_3 < 8; pg_i_3++) {
                                int vpg_e = pg_e[pg_i_3];
                                #pragma unroll
                                for (int hg_1 = 0; hg_1 < 2; hg_1++) {
                                    int vtoff_e = hg_1 * 16384 + pg_i_3 * 2048;
                                    tma_5d_gmem2smem(vdst_e + vtoff_e, (&V), 0, 0, hg_1, ei_head, vpg_e, v_full_addr + (v_prod_stage) * 8);
                                }
                            }
                            v_prod_stage += 1;
                            if (v_prod_stage == 3) { v_prod_stage = 0; v_prod_phase ^= 1; }
                        }
                    }
                }
            }
#endif  // CAKE_FMHA_DCP_PROGRAM == 4
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
            unsigned int work_stage_l = 0;
            unsigned int _phase_work_full_4 = 0;
            mbarrier_wait(work_full_addr + (work_stage_l) * 8, _phase_work_full_4);
            unsigned int base_4 = work_stage_l * 16;
            unsigned int valid_5 = work_token_words[base_4];
            unsigned int kind_5 = work_token_words[base_4 + 1];
            unsigned int batch_5 = work_token_words[base_4 + 2];
            unsigned int kv_head_5 = work_token_words[base_4 + 3];
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
            unsigned int block_begin_4 = work_token_words[base_4 + 4];
            unsigned int block_end_4 = work_token_words[base_4 + 5];
#else
            unsigned int block_begin_6 = work_token_words[base_4 + 4];
            unsigned int block_end_5 = work_token_words[base_4 + 5];
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
            unsigned int seqlen_4 = work_token_words[base_4 + 6];
            unsigned int n_chunks_4 = work_token_words[base_4 + 7];
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
            unsigned int slot_tile_base_4 = work_token_words[base_4 + 8];
            unsigned int counter_idx_4 = work_token_words[base_4 + 9];
#else
            unsigned int slot_tile_base_5 = work_token_words[base_4 + 8];
            unsigned int counter_idx_5 = work_token_words[base_4 + 9];
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
            unsigned int chunk_4 = work_token_words[base_4 + 10];
            unsigned int phase_4 = work_token_words[base_4 + 11];
            mbarrier_arrive(work_empty_addr + (work_stage_l) * 8);
            work_stage_l += 1;
            if (work_stage_l == 4) { work_stage_l = 0; _phase_work_full_4 ^= 1; }
            unsigned int valid_l = valid_5;
            int kind_l = (int)kind_5;
            int batch_idx_l = (int)batch_5;
            int kv_head_idx = (int)kv_head_5;
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
            int block_begin_l = (int)block_begin_4;
            int block_end_l = (int)block_end_4;
#else
            int block_begin_l = (int)block_begin_6;
            int block_end_l = (int)block_end_5;
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
            #pragma unroll 1
            for (unsigned int _tile_iter_l = 0; _tile_iter_l < max_items; _tile_iter_l++) {
                if (valid_l == 0) {
                    break;
                }
                if (kind_l != 0) {
                    if (lane == 0) {
                        mbarrier_arrive(claim_gate_addr);
                    }
                }
                if (kind_l == 0) {
                    int cta_n_blocks = block_end_l - block_begin_l;
                    int n_pre = ((cta_n_blocks < 3) ? cta_n_blocks : 3);
                    int gate_block = cta_n_blocks - 1 - 2;
                    if (gate_block < 0) {
                        gate_block = 0;
                    }
                    if (elect_sync()) {
                        #pragma unroll 1
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 5
                        for (int ni = 0; ni < cta_n_blocks; ni++) {
                            int nk = ni + 3;
#else
                        for (int ni = 0; ni < n_pre; ni++) {
                            int pre_stage_u = page_cons_stage + (unsigned int)ni;
                            int pre_stage = ((pre_stage_u >= 6) ? pre_stage_u - 6 : pre_stage_u);
                            int pre_phase = ((pre_stage_u >= 6) ? page_cons_phase ^ 1 : page_cons_phase);
                            int pre_pg_base = pre_stage * 8;
                            mbarrier_wait(page_offsets_full_addr + (pre_stage) * 8, pre_phase);
                            int pg_pre[8];
                            #pragma unroll
                            for (int pg_i = 0; pg_i < 8; pg_i++) {
                                pg_pre[pg_i] = smem_page_offsets[pre_pg_base + pg_i];
                            }
                            mbarrier_wait(k_empty_addr + (k_prod_stage) * 8, k_prod_phase);
                            mbarrier_arrive_expect_tx(k_full_addr + (k_prod_stage) * 8, 32768);
                            int kdst0 = smem_k_addr + k_prod_stage * 32768;
                            #pragma unroll
                            for (int pg_i_1 = 0; pg_i_1 < 8; pg_i_1++) {
                                int kpg0 = pg_pre[pg_i_1];
                                #pragma unroll
                                for (int hg = 0; hg < 2; hg++) {
                                    int ktoff0 = hg * 16384 + pg_i_1 * 2048;
                                    tma_5d_gmem2smem(kdst0 + ktoff0, (&K), 0, 0, hg, kv_head_idx, kpg0, k_full_addr + (k_prod_stage) * 8);
                                }
                            }
                            k_prod_stage += 1;
                            if (k_prod_stage == 3) { k_prod_stage = 0; k_prod_phase ^= 1; }
                            if (ni == 0) {
                                mbarrier_wait(q_empty_addr + (q_prod_stage) * 8, q_prod_phase);
                                if (_tile_iter_l == 0) {
                                }
                                int rows_hi_l = ((N_ROWS > 32) ? 1 : 0);
                                if (rows_hi_l != 0) {
                                    mbarrier_arrive_expect_tx(q_full_addr + (q_prod_stage) * 8, 64 * HEAD_DIM * 2);
                                } else {
                                    mbarrier_arrive_expect_tx(q_full_addr + (q_prod_stage) * 8, 32 * HEAD_DIM * 2);
                                }
                                tma_4d_gmem2smem(smem_qt_addr + q_prod_stage * 16384, (&Q), 0, kv_head_idx * 8, batch_idx_l * q_len, 0, q_full_addr + (q_prod_stage) * 8);
                                if (rows_hi_l != 0) {
                                    tma_4d_gmem2smem(smem_qt_addr + q_prod_stage * 16384 + (unsigned int)(32 * HEAD_DIM * 2), (&Q), 0, kv_head_idx * 8, batch_idx_l * q_len + 4, 0, q_full_addr + (q_prod_stage) * 8);
                                }
                            }
                            if (ni < 2) {
                                mbarrier_wait(v_empty_addr + (v_prod_stage) * 8, v_prod_phase);
                                mbarrier_arrive_expect_tx(v_full_addr + (v_prod_stage) * 8, 32768);
                                int vdst0 = smem_v_addr + v_prod_stage * 32768;
                                #pragma unroll
                                for (int pg_i_2 = 0; pg_i_2 < 8; pg_i_2++) {
                                    int vpg0 = pg_pre[pg_i_2];
                                    #pragma unroll
                                    for (int hg_1 = 0; hg_1 < 2; hg_1++) {
                                        int vtoff0 = hg_1 * 16384 + pg_i_2 * 2048;
                                        tma_5d_gmem2smem(vdst0 + vtoff0, (&V), 0, 0, hg_1, kv_head_idx, vpg0, v_full_addr + (v_prod_stage) * 8);
                                    }
                                }
                                v_prod_stage += 1;
                                if (v_prod_stage == 3) { v_prod_stage = 0; v_prod_phase ^= 1; }
                            }
                        }
                        #pragma unroll 1
                        for (int ni_1 = 0; ni_1 < cta_n_blocks; ni_1++) {
                            int nk = ni_1 + 3;
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
#else
#if CAKE_FMHA_DCP_PROGRAM == 4
                        for (int ni = 0; ni < cta_n_blocks; ni++) {
                            int nk = ni + 3;
#else
                        for (int ni = 0; ni < n_pre; ni++) {
                            int pre_stage_u = page_cons_stage + (unsigned int)ni;
                            int pre_stage = ((pre_stage_u >= 6) ? pre_stage_u - 6 : pre_stage_u);
                            int pre_phase = ((pre_stage_u >= 6) ? page_cons_phase ^ 1 : page_cons_phase);
                            int pre_pg_base = pre_stage * 8;
                            mbarrier_wait(page_offsets_full_addr + (pre_stage) * 8, pre_phase);
                            int pg_pre[8];
                            #pragma unroll
                            for (int pg_i = 0; pg_i < 8; pg_i++) {
                                pg_pre[pg_i] = smem_page_offsets[pre_pg_base + pg_i];
                            }
                            mbarrier_wait(k_empty_addr + (k_prod_stage) * 8, k_prod_phase);
                            mbarrier_arrive_expect_tx(k_full_addr + (k_prod_stage) * 8, 32768);
                            int kdst0 = smem_k_addr + k_prod_stage * 32768;
                            #pragma unroll
                            for (int pg_i_1 = 0; pg_i_1 < 8; pg_i_1++) {
                                int kpg0 = pg_pre[pg_i_1];
                                #pragma unroll
                                for (int hg = 0; hg < 2; hg++) {
                                    int ktoff0 = hg * 16384 + pg_i_1 * 2048;
                                    tma_5d_gmem2smem(kdst0 + ktoff0, (&K), 0, 0, hg, kv_head_idx, kpg0, k_full_addr + (k_prod_stage) * 8);
                                }
                            }
                            k_prod_stage += 1;
                            if (k_prod_stage == 3) { k_prod_stage = 0; k_prod_phase ^= 1; }
                            if (ni == 0) {
                                mbarrier_wait(q_empty_addr + (q_prod_stage) * 8, q_prod_phase);
                                if (_tile_iter_l == 0) {
                                }
                                mbarrier_arrive_expect_tx(q_full_addr + (q_prod_stage) * 8, 64 * HEAD_DIM * 2);
                                tma_4d_gmem2smem(smem_qt_addr + q_prod_stage * 16384, (&Q), 0, kv_head_idx * 8, batch_idx_l * q_len, 0, q_full_addr + (q_prod_stage) * 8);
                            }
                            if (ni < 2) {
                                mbarrier_wait(v_empty_addr + (v_prod_stage) * 8, v_prod_phase);
                                mbarrier_arrive_expect_tx(v_full_addr + (v_prod_stage) * 8, 32768);
                                int vdst0 = smem_v_addr + v_prod_stage * 32768;
                                #pragma unroll
                                for (int pg_i_2 = 0; pg_i_2 < 8; pg_i_2++) {
                                    int vpg0 = pg_pre[pg_i_2];
                                    #pragma unroll
                                    for (int hg_1 = 0; hg_1 < 2; hg_1++) {
                                        int vtoff0 = hg_1 * 16384 + pg_i_2 * 2048;
                                        tma_5d_gmem2smem(vdst0 + vtoff0, (&V), 0, 0, hg_1, kv_head_idx, vpg0, v_full_addr + (v_prod_stage) * 8);
                                    }
                                }
                                v_prod_stage += 1;
                                if (v_prod_stage == 3) { v_prod_stage = 0; v_prod_phase ^= 1; }
                            }
                        }
                        #pragma unroll 1
                        for (int ni_1 = 0; ni_1 < cta_n_blocks; ni_1++) {
                            int nk = ni_1 + 3;
#endif  // CAKE_FMHA_DCP_PROGRAM == 4
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                        for (int ni = 0; ni < n_pre; ni++) {
                            int pre_stage_u = page_cons_stage + (unsigned int)ni;
                            int pre_stage = ((pre_stage_u >= 6) ? pre_stage_u - 6 : pre_stage_u);
                            int pre_phase = ((pre_stage_u >= 6) ? page_cons_phase ^ 1 : page_cons_phase);
                            int pre_pg_base = pre_stage * 8;
                            mbarrier_wait(page_offsets_full_addr + (pre_stage) * 8, pre_phase);
                            int pg_pre[8];
                            #pragma unroll
                            for (int pg_i = 0; pg_i < 8; pg_i++) {
                                pg_pre[pg_i] = smem_page_offsets[pre_pg_base + pg_i];
                            }
                            mbarrier_wait(k_empty_addr + (k_prod_stage) * 8, k_prod_phase);
                            mbarrier_arrive_expect_tx(k_full_addr + (k_prod_stage) * 8, 32768);
                            int kdst0 = smem_k_addr + k_prod_stage * 32768;
                            #pragma unroll
                            for (int pg_i_1 = 0; pg_i_1 < 8; pg_i_1++) {
                                int kpg0 = pg_pre[pg_i_1];
                                #pragma unroll
                                for (int hg = 0; hg < 2; hg++) {
                                    int ktoff0 = hg * 16384 + pg_i_1 * 2048;
                                    tma_5d_gmem2smem(kdst0 + ktoff0, (&K), 0, 0, hg, kv_head_idx, kpg0, k_full_addr + (k_prod_stage) * 8);
                                }
                            }
                            k_prod_stage += 1;
                            if (k_prod_stage == 3) { k_prod_stage = 0; k_prod_phase ^= 1; }
                            if (ni == 0) {
                                mbarrier_wait(q_empty_addr + (q_prod_stage) * 8, q_prod_phase);
                                if (_tile_iter_l == 0) {
                                }
                                mbarrier_arrive_expect_tx(q_full_addr + (q_prod_stage) * 8, 64 * HEAD_DIM * 2);
                                tma_4d_gmem2smem(smem_qt_addr + q_prod_stage * 16384, (&Q), 0, kv_head_idx * 8, batch_idx_l * q_len, 0, q_full_addr + (q_prod_stage) * 8);
                            }
                            if (ni < 3) {
                                mbarrier_wait(v_empty_addr + (v_prod_stage) * 8, v_prod_phase);
                                mbarrier_arrive_expect_tx(v_full_addr + (v_prod_stage) * 8, 32768);
                                int vdst0 = smem_v_addr + v_prod_stage * 32768;
                                #pragma unroll
                                for (int pg_i_2 = 0; pg_i_2 < 8; pg_i_2++) {
                                    int vpg0 = pg_pre[pg_i_2];
                                    #pragma unroll
                                    for (int hg_1 = 0; hg_1 < 2; hg_1++) {
                                        int vtoff0 = hg_1 * 16384 + pg_i_2 * 2048;
                                        tma_5d_gmem2smem(vdst0 + vtoff0, (&V), 0, 0, hg_1, kv_head_idx, vpg0, v_full_addr + (v_prod_stage) * 8);
                                    }
                                }
                                v_prod_stage += 1;
                                if (v_prod_stage == 3) { v_prod_stage = 0; v_prod_phase ^= 1; }
                            }
                        }
                        #pragma unroll 1
                        for (int ni_1 = 0; ni_1 < cta_n_blocks; ni_1++) {
                            int nk = ni_1 + 3;
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                            if (nk < cta_n_blocks) {
                                int k_page_u = page_cons_stage + 3;
                                int k_page = ((k_page_u >= 6) ? k_page_u - 6 : k_page_u);
                                int k_page_phase = ((k_page_u >= 6) ? page_cons_phase ^ 1 : page_cons_phase);
                                int kpg_base = k_page * 8;
                                mbarrier_wait(page_offsets_full_addr + (k_page) * 8, k_page_phase);
                                int pg_nk[8];
                                #pragma unroll
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 5
                                for (int pg_i_4 = 0; pg_i_4 < 8; pg_i_4++) {
                                    pg_nk[pg_i_4] = smem_page_offsets[kpg_base + pg_i_4];
#else
                                for (int pg_i_3 = 0; pg_i_3 < 8; pg_i_3++) {
                                    pg_nk[pg_i_3] = smem_page_offsets[kpg_base + pg_i_3];
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
#else
#if CAKE_FMHA_DCP_PROGRAM == 4
                                for (int pg_i_4 = 0; pg_i_4 < 8; pg_i_4++) {
                                    pg_nk[pg_i_4] = smem_page_offsets[kpg_base + pg_i_4];
#else
                                for (int pg_i_3 = 0; pg_i_3 < 8; pg_i_3++) {
                                    pg_nk[pg_i_3] = smem_page_offsets[kpg_base + pg_i_3];
#endif  // CAKE_FMHA_DCP_PROGRAM == 4
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                                for (int pg_i_3 = 0; pg_i_3 < 8; pg_i_3++) {
                                    pg_nk[pg_i_3] = smem_page_offsets[kpg_base + pg_i_3];
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                                }
                                mbarrier_wait(k_empty_addr + (k_prod_stage) * 8, k_prod_phase);
                                mbarrier_arrive_expect_tx(k_full_addr + (k_prod_stage) * 8, 32768);
                                int kdst = smem_k_addr + k_prod_stage * 32768;
                                #pragma unroll
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 5
                                for (int pg_i_5 = 0; pg_i_5 < 8; pg_i_5++) {
                                    int npg0 = pg_nk[pg_i_5];
#else
                                for (int pg_i_4 = 0; pg_i_4 < 8; pg_i_4++) {
                                    int npg0 = pg_nk[pg_i_4];
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
#else
#if CAKE_FMHA_DCP_PROGRAM == 4
                                for (int pg_i_5 = 0; pg_i_5 < 8; pg_i_5++) {
                                    int npg0 = pg_nk[pg_i_5];
#else
                                for (int pg_i_4 = 0; pg_i_4 < 8; pg_i_4++) {
                                    int npg0 = pg_nk[pg_i_4];
#endif  // CAKE_FMHA_DCP_PROGRAM == 4
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                                for (int pg_i_4 = 0; pg_i_4 < 8; pg_i_4++) {
                                    int npg0 = pg_nk[pg_i_4];
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                                    #pragma unroll
                                    for (int hg_2 = 0; hg_2 < 2; hg_2++) {
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 5
                                        int ntoff = hg_2 * 16384 + pg_i_5 * 2048;
#else
                                        int ntoff = hg_2 * 16384 + pg_i_4 * 2048;
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
#else
#if CAKE_FMHA_DCP_PROGRAM == 4
                                        int ntoff = hg_2 * 16384 + pg_i_5 * 2048;
#else
                                        int ntoff = hg_2 * 16384 + pg_i_4 * 2048;
#endif  // CAKE_FMHA_DCP_PROGRAM == 4
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                                        int ntoff = hg_2 * 16384 + pg_i_4 * 2048;
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                                        tma_5d_gmem2smem(kdst + ntoff, (&K), 0, 0, hg_2, kv_head_idx, npg0, k_full_addr + (k_prod_stage) * 8);
                                    }
                                }
                                k_prod_stage += 1;
                                if (k_prod_stage == 3) { k_prod_stage = 0; k_prod_phase ^= 1; }
                            }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 5
                            int nv = ni + 2;
#else
                            int nv = ni_1 + 2;
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
#else
#if CAKE_FMHA_DCP_PROGRAM == 4
                            int nv = ni + 2;
#else
                            int nv = ni_1 + 2;
#endif  // CAKE_FMHA_DCP_PROGRAM == 4
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                            int nv = ni_1 + 3;
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                            if (nv < cta_n_blocks) {
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                                int v_page_u = page_cons_stage + 2;
#else
                                int v_page_u = page_cons_stage + 3;
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                                int v_page = ((v_page_u >= 6) ? v_page_u - 6 : v_page_u);
                                int v_page_phase = ((v_page_u >= 6) ? page_cons_phase ^ 1 : page_cons_phase);
                                int vpg_base = v_page * 8;
                                mbarrier_wait(page_offsets_full_addr + (v_page) * 8, v_page_phase);
                                int pg_nv[8];
                                #pragma unroll
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 5
                                for (int pg_i_6 = 0; pg_i_6 < 8; pg_i_6++) {
                                    pg_nv[pg_i_6] = smem_page_offsets[vpg_base + pg_i_6];
#else
                                for (int pg_i_5 = 0; pg_i_5 < 8; pg_i_5++) {
                                    pg_nv[pg_i_5] = smem_page_offsets[vpg_base + pg_i_5];
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
#else
#if CAKE_FMHA_DCP_PROGRAM == 4
                                for (int pg_i_6 = 0; pg_i_6 < 8; pg_i_6++) {
                                    pg_nv[pg_i_6] = smem_page_offsets[vpg_base + pg_i_6];
#else
                                for (int pg_i_5 = 0; pg_i_5 < 8; pg_i_5++) {
                                    pg_nv[pg_i_5] = smem_page_offsets[vpg_base + pg_i_5];
#endif  // CAKE_FMHA_DCP_PROGRAM == 4
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                                for (int pg_i_5 = 0; pg_i_5 < 8; pg_i_5++) {
                                    pg_nv[pg_i_5] = smem_page_offsets[vpg_base + pg_i_5];
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                                }
                                mbarrier_wait(v_empty_addr + (v_prod_stage) * 8, v_prod_phase);
                                mbarrier_arrive_expect_tx(v_full_addr + (v_prod_stage) * 8, 32768);
                                int vdst = smem_v_addr + v_prod_stage * 32768;
                                #pragma unroll
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 5
                                for (int pg_i_7 = 0; pg_i_7 < 8; pg_i_7++) {
                                    int vpg1 = pg_nv[pg_i_7];
#else
                                for (int pg_i_6 = 0; pg_i_6 < 8; pg_i_6++) {
                                    int vpg1 = pg_nv[pg_i_6];
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
#else
#if CAKE_FMHA_DCP_PROGRAM == 4
                                for (int pg_i_7 = 0; pg_i_7 < 8; pg_i_7++) {
                                    int vpg1 = pg_nv[pg_i_7];
#else
                                for (int pg_i_6 = 0; pg_i_6 < 8; pg_i_6++) {
                                    int vpg1 = pg_nv[pg_i_6];
#endif  // CAKE_FMHA_DCP_PROGRAM == 4
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                                for (int pg_i_6 = 0; pg_i_6 < 8; pg_i_6++) {
                                    int vpg1 = pg_nv[pg_i_6];
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                                    #pragma unroll
                                    for (int hg_3 = 0; hg_3 < 2; hg_3++) {
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 5
                                        int vtoff = hg_3 * 16384 + pg_i_7 * 2048;
#else
                                        int vtoff = hg_3 * 16384 + pg_i_6 * 2048;
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
#else
#if CAKE_FMHA_DCP_PROGRAM == 4
                                        int vtoff = hg_3 * 16384 + pg_i_7 * 2048;
#else
                                        int vtoff = hg_3 * 16384 + pg_i_6 * 2048;
#endif  // CAKE_FMHA_DCP_PROGRAM == 4
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                                        int vtoff = hg_3 * 16384 + pg_i_6 * 2048;
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                                        tma_5d_gmem2smem(vdst + vtoff, (&V), 0, 0, hg_3, kv_head_idx, vpg1, v_full_addr + (v_prod_stage) * 8);
                                    }
                                }
                                v_prod_stage += 1;
                                if (v_prod_stage == 3) { v_prod_stage = 0; v_prod_phase ^= 1; }
                            }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 5
                            if (n_pre > ni) {
                                mbarrier_wait(page_offsets_full_addr + (page_cons_stage) * 8, page_cons_phase);
                            }
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
#else
#if CAKE_FMHA_DCP_PROGRAM == 4
                            if (n_pre > ni) {
                                mbarrier_wait(page_offsets_full_addr + (page_cons_stage) * 8, page_cons_phase);
                            }
#endif  // CAKE_FMHA_DCP_PROGRAM == 4
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                            mbarrier_arrive(page_offsets_empty_addr + (page_cons_stage) * 8);
                            page_cons_stage += 1;
                            if (page_cons_stage == 6) { page_cons_stage = 0; page_cons_phase ^= 1; }
#if CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#if CAKE_FMHA_DCP_PROGRAM == 5
                            if (ni == gate_block) {
#else
                            if (ni_1 == gate_block) {
#endif  // CAKE_FMHA_DCP_PROGRAM == 5
#else
#if CAKE_FMHA_DCP_PROGRAM == 4
                            if (ni == gate_block) {
#else
                            if (ni_1 == gate_block) {
#endif  // CAKE_FMHA_DCP_PROGRAM == 4
#endif  // CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 5
#else
                            if (ni_1 == gate_block) {
#endif  // CAKE_FMHA_DCP_PROGRAM == 2 || CAKE_FMHA_DCP_PROGRAM == 3 || CAKE_FMHA_DCP_PROGRAM == 4 || CAKE_FMHA_DCP_PROGRAM == 5
                                mbarrier_arrive(claim_gate_addr);
                            }
                        }
                    }
                    q_prod_phase ^= 1;
                    if (lane == 0) {
                        if (_tile_iter_l == 0) {
                        }
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_l) * 8, _phase_work_full_4);
                unsigned int base_0_4 = work_stage_l * 16;
                unsigned int valid_1_4 = work_token_words[base_0_4];
                unsigned int kind_2_4 = work_token_words[base_0_4 + 1];
                unsigned int batch_3_4 = work_token_words[base_0_4 + 2];
                unsigned int kv_head_4_4 = work_token_words[base_0_4 + 3];
                unsigned int block_begin_5_4 = work_token_words[base_0_4 + 4];
                unsigned int block_end_6_4 = work_token_words[base_0_4 + 5];
                unsigned int seqlen_7_4 = work_token_words[base_0_4 + 6];
                unsigned int n_chunks_8_4 = work_token_words[base_0_4 + 7];
                unsigned int slot_tile_base_9_4 = work_token_words[base_0_4 + 8];
                unsigned int counter_idx_10_4 = work_token_words[base_0_4 + 9];
                unsigned int chunk_11_4 = work_token_words[base_0_4 + 10];
                unsigned int phase_12_4 = work_token_words[base_0_4 + 11];
                mbarrier_arrive(work_empty_addr + (work_stage_l) * 8);
                work_stage_l += 1;
                if (work_stage_l == 4) { work_stage_l = 0; _phase_work_full_4 ^= 1; }
                valid_l = valid_1_4;
                kind_l = (int)kind_2_4;
                batch_idx_l = (int)batch_3_4;
                kv_head_idx = (int)kv_head_4_4;
                block_begin_l = (int)block_begin_5_4;
                block_end_l = (int)block_end_6_4;
            }
            if (lane == 0) {
            }
        }
    }

    // Cleanup

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
