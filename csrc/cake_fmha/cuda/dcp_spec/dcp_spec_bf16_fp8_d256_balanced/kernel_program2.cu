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
#define TMEM_TMEM_S_OFFSET 0
#define TMEM_TMEM_O_G0_OFFSET 256
#define TMEM_TMEM_O_G1_OFFSET 384
#define NUM_Q_PIPE_STAGES 1
#define NUM_Q_RAW_PIPE_STAGES 1
#define NUM_K_PIPE_STAGES 3
#define NUM_V_PIPE_STAGES 2
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
#define SMEM_RSTAGE_DATA_OFF 13312
#define SMEM_RSTAGE_DATA_STAGE_BYTES 196608
#define SMEM_RSTAGE_DATA_STRIDE 196608
#define SMEM_RSTAGE_STATS_OFF 209920
#define SMEM_RSTAGE_STATS_STAGE_BYTES 16384
#define SMEM_RSTAGE_STATS_STRIDE 16384
#define SMEM_SMEM_QBF_OFF 46080
#define SMEM_SMEM_QBF_STAGE_BYTES 16384
#define SMEM_SMEM_QBF_STRIDE 16384
#define SMEM_SMEM_Q_HI_G0_OFF 13312
#define SMEM_SMEM_Q_HI_G0_STAGE_BYTES 8192
#define SMEM_SMEM_Q_HI_G0_STRIDE 8192
#define SMEM_SMEM_Q_HI_G1_OFF 21504
#define SMEM_SMEM_Q_HI_G1_STAGE_BYTES 8192
#define SMEM_SMEM_Q_HI_G1_STRIDE 8192
#define SMEM_SMEM_Q_LO_G0_OFF 29696
#define SMEM_SMEM_Q_LO_G0_STAGE_BYTES 8192
#define SMEM_SMEM_Q_LO_G0_STRIDE 8192
#define SMEM_SMEM_Q_LO_G1_OFF 37888
#define SMEM_SMEM_Q_LO_G1_STAGE_BYTES 8192
#define SMEM_SMEM_Q_LO_G1_STRIDE 8192
#define SMEM_SMEM_K_G0_OFF 62464
#define SMEM_SMEM_K_G0_STAGE_BYTES 16384
#define SMEM_SMEM_K_G0_STRIDE 16384
#define SMEM_SMEM_K_G1_OFF 111616
#define SMEM_SMEM_K_G1_STAGE_BYTES 16384
#define SMEM_SMEM_K_G1_STRIDE 16384
#define SMEM_SMEM_V_G0_OFF 160768
#define SMEM_SMEM_V_G0_STAGE_BYTES 16384
#define SMEM_SMEM_V_G0_STRIDE 16384
#define SMEM_SMEM_V_G1_OFF 193536
#define SMEM_SMEM_V_G1_STAGE_BYTES 16384
#define SMEM_SMEM_V_G1_STRIDE 16384
#define SMEM_TOTAL 226304
#define THREADS 384
#ifndef N_ROWS
#define N_ROWS 64
#endif
#define BLOCK_N 128
#define HEAD_DIM 256
#define PAGE_SIZE 64
#define NUM_K_STAGES 3
#define NUM_V_STAGES 2

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


__device__ __forceinline__ void tcgen05_mma_f8f6f4(
    int taddr, uint64_t a_desc, uint64_t b_desc,
    uint32_t i_desc, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t"
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
}




__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(THREADS) void
kernel_cake_fmha_dcp_spec_bf16_fp8_d256_balanced(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap V, __nv_bfloat16* __restrict__ O_ptr, float* __restrict__ LSE_ptr, int* __restrict__ page_table, int* __restrict__ causal_seqlens_kv_global, float* __restrict__ partial_o, float* __restrict__ partial_stats, unsigned int* __restrict__ tile_counters, unsigned int* __restrict__ queue_counters, int max_pages_per_seq, float softmax_scale_log2, float output_scale, int num_q_heads, int num_kv_heads, int batch_size, int q_len, int cp_rank, int cp_world_log2, unsigned int max_items)
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
    #define q_raw_full_addr (mbar_base + 0)
    #define q_raw_empty_addr (mbar_base + 8)
    #define q_full_addr (mbar_base + 16)
    #define q_empty_addr (mbar_base + 24)
    #define k_full_addr (mbar_base + 32)
    #define k_empty_addr (mbar_base + 56)
    #define v_full_addr (mbar_base + 80)
    #define v_empty_addr (mbar_base + 96)
    #define s_full_addr (mbar_base + 112)
    #define p_full_addr (mbar_base + 128)
    #define o_ready_addr (mbar_base + 144)
    #define o_empty_addr (mbar_base + 160)
    #define stats_full_addr (mbar_base + 168)
    #define stats_empty_addr (mbar_base + 184)
    #define tmem_dealloc_addr (mbar_base + 200)
    #define page_offsets_full_addr (mbar_base + 208)
    #define page_offsets_empty_addr (mbar_base + 256)
    #define work_full_addr (mbar_base + 304)
    #define work_empty_addr (mbar_base + 336)
    #define claim_gate_addr (mbar_base + 368)

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
    float* rstage_data = reinterpret_cast<float*>(smem_raw + SMEM_RSTAGE_DATA_OFF);
    const int rstage_data_addr = smem + SMEM_RSTAGE_DATA_OFF;
    float* rstage_stats = reinterpret_cast<float*>(smem_raw + SMEM_RSTAGE_STATS_OFF);
    const int rstage_stats_addr = smem + SMEM_RSTAGE_STATS_OFF;
    __nv_bfloat16* smem_qbf = reinterpret_cast<__nv_bfloat16*>(smem_raw + SMEM_SMEM_QBF_OFF);
    const int smem_qbf_addr = smem + SMEM_SMEM_QBF_OFF;
    uint8_t* smem_q_hi_g0 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_Q_HI_G0_OFF);
    const int smem_q_hi_g0_addr = smem + SMEM_SMEM_Q_HI_G0_OFF;
    uint8_t* smem_q_hi_g1 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_Q_HI_G1_OFF);
    const int smem_q_hi_g1_addr = smem + SMEM_SMEM_Q_HI_G1_OFF;
    uint8_t* smem_q_lo_g0 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_Q_LO_G0_OFF);
    const int smem_q_lo_g0_addr = smem + SMEM_SMEM_Q_LO_G0_OFF;
    uint8_t* smem_q_lo_g1 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_Q_LO_G1_OFF);
    const int smem_q_lo_g1_addr = smem + SMEM_SMEM_Q_LO_G1_OFF;
    uint8_t* smem_k_g0 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_K_G0_OFF);
    const int smem_k_g0_addr = smem + SMEM_SMEM_K_G0_OFF;
    uint8_t* smem_k_g1 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_K_G1_OFF);
    const int smem_k_g1_addr = smem + SMEM_SMEM_K_G1_OFF;
    uint8_t* smem_v_g0 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_V_G0_OFF);
    const int smem_v_g0_addr = smem + SMEM_SMEM_V_G0_OFF;
    uint8_t* smem_v_g1 = reinterpret_cast<uint8_t*>(smem_raw + SMEM_SMEM_V_G1_OFF);
    const int smem_v_g1_addr = smem + SMEM_SMEM_V_G1_OFF;
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Q))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&K))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&V))) : "memory");

    // Mbarrier init (20 pipeline groups, 0 ordered-sequence groups, 47 barriers)
    // Mbarriers at smem_raw[0..376)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'q_raw_pipe' ---
            // q_raw_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // q_raw_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // --- pipeline 'q_pipe' ---
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            // q_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 24, 1);
            // --- pipeline 'k_pipe' ---
            // k_full: 3 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            // k_empty: 3 barriers, init_count=1
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // --- pipeline 'v_pipe' ---
            // v_full: 2 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // v_empty: 2 barriers, init_count=1
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
            // o_empty: 1 barriers, init_count=128
            mbarrier_init(smem + 160, 128);
            // --- pipeline 'stats_pipe' ---
            // stats_full: 2 barriers, init_count=128
            mbarrier_init(smem + 168, 128);
            mbarrier_init(smem + 176, 128);
            // stats_empty: 2 barriers, init_count=4
            mbarrier_init(smem + 184, 4);
            mbarrier_init(smem + 192, 4);
            // tmem_dealloc: 1 barriers, init_count=128
            mbarrier_init(smem + 200, 128);
            // --- pipeline 'page_pipe' ---
            // page_offsets_full: 6 barriers, init_count=1
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            mbarrier_init(smem + 232, 1);
            mbarrier_init(smem + 240, 1);
            mbarrier_init(smem + 248, 1);
            // page_offsets_empty: 6 barriers, init_count=1
            mbarrier_init(smem + 256, 1);
            mbarrier_init(smem + 264, 1);
            mbarrier_init(smem + 272, 1);
            mbarrier_init(smem + 280, 1);
            mbarrier_init(smem + 288, 1);
            mbarrier_init(smem + 296, 1);
            // --- pipeline 'work_pipe' ---
            // work_full: 4 barriers, init_count=1
            mbarrier_init(smem + 304, 1);
            mbarrier_init(smem + 312, 1);
            mbarrier_init(smem + 320, 1);
            mbarrier_init(smem + 328, 1);
            // work_empty: 4 barriers, init_count=352
            mbarrier_init(smem + 336, 352);
            mbarrier_init(smem + 344, 352);
            mbarrier_init(smem + 352, 352);
            mbarrier_init(smem + 360, 352);
            // claim_gate: 1 barriers, init_count=1
            mbarrier_init(smem + 368, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 376);
    if (warp == 0) {
        int _tmem_hold = smem + 376;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_s = taddr;
    const int tmem_tmem_o_g0 = taddr + 256;
    const int tmem_tmem_o_g1 = taddr + 384;

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
            int row_j = my_row / 16;
            unsigned int sm_stage = 0;
            unsigned int sm_phase = 0;
            unsigned int xm_slot_s = 0;
            unsigned int st_stage_s = 0;
            unsigned int st_phase_s = 1;
            float _rcp_2 = approx_rcp(softmax_scale_log2);
            float thr_raw = 8.0f * _rcp_2;
            float p_scale_log2 = 0.8073549f;
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
            unsigned int q_tile = work_token_words[base + 12];
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
            int q_tile_s = (int)q_tile;
            if (valid_s != 0) {
                if (kind_s == 0) {
                    int _min_1 = ((q_len - q_tile_s * 4) < (4) ? (q_len - q_tile_s * 4) : (4));
                    int live_rows_w = _min_1 * 16;
                    int k2x_s = 0;
                    if (live_rows_w > 32) {
                        if ((int)block_end - (int)block_begin >= 1) {
                            k2x_s = 1;
                        }
                    }
                    mbarrier_wait(q_raw_full_addr, 0);
                    int q_hi_base_s = smem_q_hi_g0_addr;
                    int q_lo_base_s = smem_q_lo_g0_addr;
                    int q_key_bf_s = smem_qbf_addr / 128 % 8;
                    int q_key_hi_s = q_hi_base_s / 128 % 8;
                    int q_key_lo_s = q_lo_base_s / 128 % 8;
                    unsigned int q_words_s[4];
                    float q_f32_s[8];
                    float q_res_s[8];
                    unsigned int q_packed_s[2];
                    unsigned int q_packed_lo_s[2];
                    #pragma unroll 1
                    for (int qc_i = 0; qc_i < 4; qc_i++) {
                        int q_chunk_s = qc_i * 256 + (128 + sm_tid);
                        int q_row_s = q_chunk_s / 32;
                        int q_drow_s = q_row_s;
                        int q_col8_s = q_chunk_s % 32;
                        int q_kg_s = q_col8_s / 8;
                        int q_c16a_s = q_col8_s % 8;
                        int q_grp_s = q_col8_s / 16;
                        int q_c16_s = q_col8_s % 16 / 2;
                        int q_half_s = q_col8_s % 2 * 8;
                        if (q_drow_s < live_rows_w) {
                            int q_key_row_s = (q_key_bf_s + q_row_s) % 8;
                            int q_src_s = smem_qbf_addr + (unsigned int)(q_kg_s * 4096) + (unsigned int)(q_row_s * 128) + (unsigned int)((q_c16a_s ^ q_key_row_s) * 16);
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_words_s[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_s[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_s[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_s[(0) + 3]))
                                : "r"(q_src_s));
                            #pragma unroll
                            for (int qw_s = 0; qw_s < 4; qw_s++) {
                                unsigned int q_w_s = q_words_s[qw_s];
                                unsigned int q_wbits_d = q_w_s << 16;
                                float q_lo_d = 0.0f;
                                float q_lo_t_d = 0.0f;
                                float q_hi_d = 0.0f;
                                float q_hi_t_d = 0.0f;
                                q_lo_d = reinterpret_cast<float*>(&q_wbits_d)[0];
                                q_wbits_d = (q_w_s & 65520) << 16;
                                q_lo_t_d = reinterpret_cast<float*>(&q_wbits_d)[0];
                                q_wbits_d = q_w_s >> 16 << 16;
                                q_hi_d = reinterpret_cast<float*>(&q_wbits_d)[0];
                                q_wbits_d = (q_w_s >> 16 & 65520) << 16;
                                q_hi_t_d = reinterpret_cast<float*>(&q_wbits_d)[0];
                                q_f32_s[2 * qw_s] = q_lo_t_d;
                                q_res_s[2 * qw_s] = q_lo_d - q_lo_t_d;
                                q_f32_s[2 * qw_s + 1] = q_hi_t_d;
                                q_res_s[2 * qw_s + 1] = q_hi_d - q_hi_t_d;
                            }
                        } else {
                            #pragma unroll
                            for (int qz_s = 0; qz_s < 8; qz_s++) {
                                q_f32_s[qz_s] = 0.0f;
                                q_res_s[qz_s] = 0.0f;
                            }
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
                                : "=r"(_packed) : "f"(q_f32_s[0]), "f"(q_f32_s[1]),
                                                   "f"(q_f32_s[2]), "f"(q_f32_s[3]));
                            q_packed_s[0] = _packed;
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
                                : "=r"(_packed) : "f"(q_f32_s[4]), "f"(q_f32_s[5]),
                                                   "f"(q_f32_s[6]), "f"(q_f32_s[7]));
                            q_packed_s[1] = _packed;
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
                                : "=r"(_packed) : "f"(q_res_s[0]), "f"(q_res_s[1]),
                                                   "f"(q_res_s[2]), "f"(q_res_s[3]));
                            q_packed_lo_s[0] = _packed;
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
                                : "=r"(_packed) : "f"(q_res_s[4]), "f"(q_res_s[5]),
                                                   "f"(q_res_s[6]), "f"(q_res_s[7]));
                            q_packed_lo_s[1] = _packed;
                        }
                        int q_row_off_s = q_grp_s * 8192 + q_drow_s * 128 + q_half_s;
                        int q_hi_addr_s = q_hi_base_s + q_row_off_s + (q_c16_s ^ (q_key_hi_s + q_drow_s) % 8) * 16;
                        int q_lo_addr_s = q_lo_base_s + q_row_off_s + (q_c16_s ^ (q_key_lo_s + q_drow_s) % 8) * 16;
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_hi_addr_s), "r"((q_packed_s[0])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_hi_addr_s + 4), "r"((q_packed_s[1])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_lo_addr_s), "r"((q_packed_lo_s[0])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_lo_addr_s + 4), "r"((q_packed_lo_s[1])));
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    if (k2x_s != 0) {
                        int q_hi_base_s_0 = smem_q_hi_g0_addr;
                        int q_lo_base_s_1 = smem_q_lo_g0_addr;
                        int q_key_bf_s_2 = (smem_k_g0_addr + 32768) / 128 % 8;
                        int q_key_hi_s_3 = q_hi_base_s_0 / 128 % 8;
                        int q_key_lo_s_4 = q_lo_base_s_1 / 128 % 8;
                        unsigned int q_words_s_5[4];
                        float q_f32_s_6[8];
                        float q_res_s_7[8];
                        unsigned int q_packed_s_8[2];
                        unsigned int q_packed_lo_s_9[2];
                        #pragma unroll 1
                        for (int qc_i_1 = 0; qc_i_1 < 4; qc_i_1++) {
                            int q_chunk_s_1 = qc_i_1 * 256 + (128 + sm_tid);
                            int q_row_s_1 = q_chunk_s_1 / 32;
                            int q_drow_s_1 = 32 + q_row_s_1;
                            int q_col8_s_1 = q_chunk_s_1 % 32;
                            int q_kg_s_1 = q_col8_s_1 / 8;
                            int q_c16a_s_1 = q_col8_s_1 % 8;
                            int q_grp_s_1 = q_col8_s_1 / 16;
                            int q_c16_s_1 = q_col8_s_1 % 16 / 2;
                            int q_half_s_1 = q_col8_s_1 % 2 * 8;
                            if (q_drow_s_1 < live_rows_w) {
                                int q_key_row_s_1 = (q_key_bf_s_2 + q_row_s_1) % 8;
                                int q_src_s_1 = smem_k_g0_addr + 32768 + (unsigned int)(q_kg_s_1 * 4096) + (unsigned int)(q_row_s_1 * 128) + (unsigned int)((q_c16a_s_1 ^ q_key_row_s_1) * 16);
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_5[(0) + 3]))
                                    : "r"(q_src_s_1));
                                #pragma unroll
                                for (int qw_s_1 = 0; qw_s_1 < 4; qw_s_1++) {
                                    unsigned int q_w_s_1 = q_words_s_5[qw_s_1];
                                    unsigned int q_wbits_d_1 = q_w_s_1 << 16;
                                    float q_lo_d_1 = 0.0f;
                                    float q_lo_t_d_1 = 0.0f;
                                    float q_hi_d_1 = 0.0f;
                                    float q_hi_t_d_1 = 0.0f;
                                    q_lo_d_1 = reinterpret_cast<float*>(&q_wbits_d_1)[0];
                                    q_wbits_d_1 = (q_w_s_1 & 65520) << 16;
                                    q_lo_t_d_1 = reinterpret_cast<float*>(&q_wbits_d_1)[0];
                                    q_wbits_d_1 = q_w_s_1 >> 16 << 16;
                                    q_hi_d_1 = reinterpret_cast<float*>(&q_wbits_d_1)[0];
                                    q_wbits_d_1 = (q_w_s_1 >> 16 & 65520) << 16;
                                    q_hi_t_d_1 = reinterpret_cast<float*>(&q_wbits_d_1)[0];
                                    q_f32_s_6[2 * qw_s_1] = q_lo_t_d_1;
                                    q_res_s_7[2 * qw_s_1] = q_lo_d_1 - q_lo_t_d_1;
                                    q_f32_s_6[2 * qw_s_1 + 1] = q_hi_t_d_1;
                                    q_res_s_7[2 * qw_s_1 + 1] = q_hi_d_1 - q_hi_t_d_1;
                                }
                            } else {
                                #pragma unroll
                                for (int qz_s_1 = 0; qz_s_1 < 8; qz_s_1++) {
                                    q_f32_s_6[qz_s_1] = 0.0f;
                                    q_res_s_7[qz_s_1] = 0.0f;
                                }
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
                                    : "=r"(_packed) : "f"(q_f32_s_6[0]), "f"(q_f32_s_6[1]),
                                                       "f"(q_f32_s_6[2]), "f"(q_f32_s_6[3]));
                                q_packed_s_8[0] = _packed;
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
                                    : "=r"(_packed) : "f"(q_f32_s_6[4]), "f"(q_f32_s_6[5]),
                                                       "f"(q_f32_s_6[6]), "f"(q_f32_s_6[7]));
                                q_packed_s_8[1] = _packed;
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
                                    : "=r"(_packed) : "f"(q_res_s_7[0]), "f"(q_res_s_7[1]),
                                                       "f"(q_res_s_7[2]), "f"(q_res_s_7[3]));
                                q_packed_lo_s_9[0] = _packed;
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
                                    : "=r"(_packed) : "f"(q_res_s_7[4]), "f"(q_res_s_7[5]),
                                                       "f"(q_res_s_7[6]), "f"(q_res_s_7[7]));
                                q_packed_lo_s_9[1] = _packed;
                            }
                            int q_row_off_s_1 = q_grp_s_1 * 8192 + q_drow_s_1 * 128 + q_half_s_1;
                            int q_hi_addr_s_1 = q_hi_base_s_0 + q_row_off_s_1 + (q_c16_s_1 ^ (q_key_hi_s_3 + q_drow_s_1) % 8) * 16;
                            int q_lo_addr_s_1 = q_lo_base_s_1 + q_row_off_s_1 + (q_c16_s_1 ^ (q_key_lo_s_4 + q_drow_s_1) % 8) * 16;
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_hi_addr_s_1), "r"((q_packed_s_8[0])));
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_hi_addr_s_1 + 4), "r"((q_packed_s_8[1])));
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_lo_addr_s_1), "r"((q_packed_lo_s_9[0])));
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_lo_addr_s_1 + 4), "r"((q_packed_lo_s_9[1])));
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 9, 256;" ::: "memory");
                        if (sm_tid == 0) {
                            mbarrier_arrive(q_raw_empty_addr);
                            mbarrier_arrive(q_full_addr);
                        }
                    } else {
                        asm volatile("barrier.sync 9, 256;" ::: "memory");
                        if (sm_tid == 0) {
                            mbarrier_arrive(q_raw_empty_addr);
                        }
                        if (live_rows_w > 32) {
                            mbarrier_wait(q_raw_full_addr, 1);
                        }
                        int q_hi_base_s_0_1 = smem_q_hi_g0_addr;
                        int q_lo_base_s_1_1 = smem_q_lo_g0_addr;
                        int q_key_bf_s_2_1 = smem_qbf_addr / 128 % 8;
                        int q_key_hi_s_3_1 = q_hi_base_s_0_1 / 128 % 8;
                        int q_key_lo_s_4_1 = q_lo_base_s_1_1 / 128 % 8;
                        unsigned int q_words_s_5_1[4];
                        float q_f32_s_6_1[8];
                        float q_res_s_7_1[8];
                        unsigned int q_packed_s_8_1[2];
                        unsigned int q_packed_lo_s_9_1[2];
                        #pragma unroll 1
                        for (int qc_i_2 = 0; qc_i_2 < 4; qc_i_2++) {
                            int q_chunk_s_2 = qc_i_2 * 256 + (128 + sm_tid);
                            int q_row_s_2 = q_chunk_s_2 / 32;
                            int q_drow_s_2 = 32 + q_row_s_2;
                            int q_col8_s_2 = q_chunk_s_2 % 32;
                            int q_kg_s_2 = q_col8_s_2 / 8;
                            int q_c16a_s_2 = q_col8_s_2 % 8;
                            int q_grp_s_2 = q_col8_s_2 / 16;
                            int q_c16_s_2 = q_col8_s_2 % 16 / 2;
                            int q_half_s_2 = q_col8_s_2 % 2 * 8;
                            if (q_drow_s_2 < live_rows_w) {
                                int q_key_row_s_2 = (q_key_bf_s_2_1 + q_row_s_2) % 8;
                                int q_src_s_2 = smem_qbf_addr + (unsigned int)(q_kg_s_2 * 4096) + (unsigned int)(q_row_s_2 * 128) + (unsigned int)((q_c16a_s_2 ^ q_key_row_s_2) * 16);
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_5_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_5_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_5_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_5_1[(0) + 3]))
                                    : "r"(q_src_s_2));
                                #pragma unroll
                                for (int qw_s_2 = 0; qw_s_2 < 4; qw_s_2++) {
                                    unsigned int q_w_s_2 = q_words_s_5_1[qw_s_2];
                                    unsigned int q_wbits_d_2 = q_w_s_2 << 16;
                                    float q_lo_d_2 = 0.0f;
                                    float q_lo_t_d_2 = 0.0f;
                                    float q_hi_d_2 = 0.0f;
                                    float q_hi_t_d_2 = 0.0f;
                                    q_lo_d_2 = reinterpret_cast<float*>(&q_wbits_d_2)[0];
                                    q_wbits_d_2 = (q_w_s_2 & 65520) << 16;
                                    q_lo_t_d_2 = reinterpret_cast<float*>(&q_wbits_d_2)[0];
                                    q_wbits_d_2 = q_w_s_2 >> 16 << 16;
                                    q_hi_d_2 = reinterpret_cast<float*>(&q_wbits_d_2)[0];
                                    q_wbits_d_2 = (q_w_s_2 >> 16 & 65520) << 16;
                                    q_hi_t_d_2 = reinterpret_cast<float*>(&q_wbits_d_2)[0];
                                    q_f32_s_6_1[2 * qw_s_2] = q_lo_t_d_2;
                                    q_res_s_7_1[2 * qw_s_2] = q_lo_d_2 - q_lo_t_d_2;
                                    q_f32_s_6_1[2 * qw_s_2 + 1] = q_hi_t_d_2;
                                    q_res_s_7_1[2 * qw_s_2 + 1] = q_hi_d_2 - q_hi_t_d_2;
                                }
                            } else {
                                #pragma unroll
                                for (int qz_s_2 = 0; qz_s_2 < 8; qz_s_2++) {
                                    q_f32_s_6_1[qz_s_2] = 0.0f;
                                    q_res_s_7_1[qz_s_2] = 0.0f;
                                }
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
                                    : "=r"(_packed) : "f"(q_f32_s_6_1[0]), "f"(q_f32_s_6_1[1]),
                                                       "f"(q_f32_s_6_1[2]), "f"(q_f32_s_6_1[3]));
                                q_packed_s_8_1[0] = _packed;
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
                                    : "=r"(_packed) : "f"(q_f32_s_6_1[4]), "f"(q_f32_s_6_1[5]),
                                                       "f"(q_f32_s_6_1[6]), "f"(q_f32_s_6_1[7]));
                                q_packed_s_8_1[1] = _packed;
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
                                    : "=r"(_packed) : "f"(q_res_s_7_1[0]), "f"(q_res_s_7_1[1]),
                                                       "f"(q_res_s_7_1[2]), "f"(q_res_s_7_1[3]));
                                q_packed_lo_s_9_1[0] = _packed;
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
                                    : "=r"(_packed) : "f"(q_res_s_7_1[4]), "f"(q_res_s_7_1[5]),
                                                       "f"(q_res_s_7_1[6]), "f"(q_res_s_7_1[7]));
                                q_packed_lo_s_9_1[1] = _packed;
                            }
                            int q_row_off_s_2 = q_grp_s_2 * 8192 + q_drow_s_2 * 128 + q_half_s_2;
                            int q_hi_addr_s_2 = q_hi_base_s_0_1 + q_row_off_s_2 + (q_c16_s_2 ^ (q_key_hi_s_3_1 + q_drow_s_2) % 8) * 16;
                            int q_lo_addr_s_2 = q_lo_base_s_1_1 + q_row_off_s_2 + (q_c16_s_2 ^ (q_key_lo_s_4_1 + q_drow_s_2) % 8) * 16;
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_hi_addr_s_2), "r"((q_packed_s_8_1[0])));
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_hi_addr_s_2 + 4), "r"((q_packed_s_8_1[1])));
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_lo_addr_s_2), "r"((q_packed_lo_s_9_1[0])));
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_lo_addr_s_2 + 4), "r"((q_packed_lo_s_9_1[1])));
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 9, 256;" ::: "memory");
                        if (sm_tid == 0) {
                            if (live_rows_w > 32) {
                                mbarrier_arrive(q_raw_empty_addr);
                            }
                            mbarrier_arrive(q_full_addr);
                        }
                    }
                }
            }
            #pragma unroll 1
            for (unsigned int _tile_iter_s = 0; _tile_iter_s < max_items; _tile_iter_s++) {
                if (valid_s == 0) {
                    break;
                }
                if (kind_s == 0) {
                    int cnt_s = block_end_s - block_begin_s;
                    int _min_2 = ((q_tile_s * 4 + row_j) < (q_len - 1) ? (q_tile_s * 4 + row_j) : (q_len - 1));
                    int vis_j = _min_2;
                    int back_s = q_len - 1 - vis_j - phase_s;
                    int vis_col = seqlen_s;
                    if (back_s > 0) {
                        vis_col = seqlen_s - (back_s + (1 << cp_world_log2) - 1 >> cp_world_log2);
                    }
                    float row_max = -CAKE_FMHA_INF;
                    float psum = 0.0f;
                    #pragma unroll 1
                    for (int n = 0; n < cnt_s; n++) {
                        if (sm_tid == 0) {
                        }
                        mbarrier_wait(s_full_addr + (sm_stage) * 8, sm_phase);
                        if (sm_tid == 0) {
                        }
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
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            int n_vis = vis_col - blk_pos;
                            if (my_row >= N_ROWS) {
                                n_vis = 0;
                            }
                            if (n_vis < 32) {
                                int _max_3 = ((n_vis) > (0) ? (n_vis) : (0));
                                int n_lo = _max_3;
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
                                    }
                                }
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
                            }
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
                            float _max_4 = max_noftz(lmax, _shfl_xor_0);
                            lmax = _max_4;
                            if (half == 0) {
                                smem_xmax[xm_off_s + my_row] = lmax;
                            }
                        }
                        if (sm_tid == 0) {
                        }
                        asm volatile("barrier.sync 8, 256;" ::: "memory");
                        if (rows_live != 0) {
                            float _max_5 = max_noftz(lmax, smem_xmax[xm_off_s + 64 + my_row]);
                            lmax = _max_5;
                            if (lmax > row_max + thr_raw) {
                                new_max = lmax;
                                if (row_max > -CAKE_FMHA_INF) {
                                    float _exp2_0 = approx_exp2(softmax_scale_log2 * (row_max - new_max));
                                    acc_scale = _exp2_0;
                                }
                            }
                        }
                        xm_slot_s = xm_slot_s ^ 1;
                        if (sm_tid == 0) {
                        }
                        if (rows_live != 0) {
                            float safe_max = ((new_max == -CAKE_FMHA_INF) ? 0.0f : new_max);
                            const float2 _fma_b2_2 = {softmax_scale_log2, softmax_scale_log2};
                            const float2 _fma_c2_3 = {p_scale_log2 - safe_max * softmax_scale_log2, p_scale_log2 - safe_max * softmax_scale_log2};
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
                            #pragma unroll
                            for (int _le = 0; _le < 32; _le++) {
                                sv[_le] = approx_exp2(sv[_le]);
                            }
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
                        }
                        if (sm_tid == 0) {
                        }
                        if (rows_live != 0) {
                            unsigned int regs_p[8];
                            {
                                uint32_t _packed;
                                asm volatile("{\n\t"
                                    ".reg .b16 _lo;\n\t"
                                    ".reg .b16 _hi;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                    "}"
                                    : "=r"(_packed) : "f"(sv[0]), "f"(sv[1]),
                                                       "f"(sv[2]), "f"(sv[3]));
                                regs_p[0] = _packed;
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
                                    : "=r"(_packed) : "f"(sv[4]), "f"(sv[5]),
                                                       "f"(sv[6]), "f"(sv[7]));
                                regs_p[1] = _packed;
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
                                    : "=r"(_packed) : "f"(sv[8]), "f"(sv[9]),
                                                       "f"(sv[10]), "f"(sv[11]));
                                regs_p[2] = _packed;
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
                                    : "=r"(_packed) : "f"(sv[12]), "f"(sv[13]),
                                                       "f"(sv[14]), "f"(sv[15]));
                                regs_p[3] = _packed;
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
                                    : "=r"(_packed) : "f"(sv[16]), "f"(sv[17]),
                                                       "f"(sv[18]), "f"(sv[19]));
                                regs_p[4] = _packed;
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
                                    : "=r"(_packed) : "f"(sv[20]), "f"(sv[21]),
                                                       "f"(sv[22]), "f"(sv[23]));
                                regs_p[5] = _packed;
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
                                    : "=r"(_packed) : "f"(sv[24]), "f"(sv[25]),
                                                       "f"(sv[26]), "f"(sv[27]));
                                regs_p[6] = _packed;
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
                                    : "=r"(_packed) : "f"(sv[28]), "f"(sv[29]),
                                                       "f"(sv[30]), "f"(sv[31]));
                                regs_p[7] = _packed;
                            }
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x32bx2.x8.b32"
                                " [%0], 16, {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "r"((unsigned int)my_s_base + sm_stage * (unsigned int)BLOCK_N), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[0])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[1])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[2])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[3])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[4])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[5])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[6])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[7])));
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        }
                        __syncwarp();
                        if (lane == 0) {
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;"
                                :: "r"(p_full_addr + (sm_stage) * 8), "r"((uint32_t)(32)) : "memory");
                        }
                        sm_stage += 1;
                        if (sm_stage == 2) { sm_stage = 0; sm_phase ^= 1; }
                        if (sm_tid == 0) {
                        }
                    }
                    float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, psum, 16);
                    float total = psum + _shfl_xor_1;
                    if (sm_tid == 0) {
                    }
                    mbarrier_wait(stats_empty_addr + (st_stage_s) * 8, st_phase_s);
                    if (rows_live != 0) {
                        if (half == 0) {
                            smem_sum[st_stage_s * 64 + (unsigned int)my_row] = total;
                            smem_max[st_stage_s * 64 + (unsigned int)my_row] = row_max;
                        }
                    }
                    __syncwarp();
                    if (lane == 0) {
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;"
                            :: "r"(stats_full_addr + (st_stage_s) * 8), "r"((uint32_t)(32)) : "memory");
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
                    int _min_7 = ((q_len - q_tile_s * 4) < (4) ? (q_len - q_tile_s * 4) : (4));
                    int live_rows_r = _min_7 * 16;
                    if (r_row < live_rows_r) {
                        int items_r = num_kv_heads * ((q_len + 4 - 1) / 4);
                        int stats_stride_r = items_r * 128;
                        int o_stride_r = items_r * 16384;
                        int stats_row_r = slot_tile_base_s * 128 + r_row;
                        int f_r = 64 >> block_end_s;
                        int d0_r = block_begin_s * (4 * f_r) + (128 + sm_tid) % 4 * f_r;
                        {
                            d0_r = block_begin_s * (4 * f_r) + (128 + sm_tid) % 4 * 16;
                        }
                        int o_row_r = slot_tile_base_s * 16384 + r_row * 4 + d0_r * 64;
                        int j_r = q_tile_s * 4 + r_row / 16;
                        int h_r = r_row % 16;
                        int q_head_r = kv_head_s * 16 + h_r;
                        int o_idx_r = ((batch_s * q_len + j_r) * num_q_heads + q_head_r) * HEAD_DIM + d0_r;
                        int store_lse_r = 0;
                        if ((128 + sm_tid) % 4 == 0 && block_begin_s == 0) {
                            store_lse_r = 1;
                        }
                        int lse_idx_r = (batch_s * q_len + j_r) * num_q_heads + q_head_r;
                        int narrow_r = 0;
                        if (narrow_r != 0) {
                            int d0_n = block_begin_s * (4 * f_r) + (128 + sm_tid) % 4 * f_r;
                            int o_row_n = slot_tile_base_s * 16384 + r_row * 4 + d0_n * 64;
                            int o_idx_n = ((batch_s * q_len + j_r) * num_q_heads + q_head_r) * HEAD_DIM + d0_n;
                            float acc_f[4];
                            float out4[4];
                            int n_pad_r = (n_chunks_s + 15) / 16 * 16;
                            #pragma unroll 1
                            for (int g_r = 0; g_r < 16 >> block_end_s; g_r++) {
                                acc_f[0] = 0.0f;
                                acc_f[1] = 0.0f;
                                acc_f[2] = 0.0f;
                                acc_f[3] = 0.0f;
                                float m_f = -1e+30f;
                                float l_f = 0.0f;
                                int o_col_r = o_row_n + g_r * 4 * 64;
                                #pragma unroll 16
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
                                    float _max_8 = max_noftz(m_f, m_k);
                                    float m_new = _max_8;
                                    float _exp2_3 = approx_exp2((m_f - m_new) * softmax_scale_log2);
                                    float a_k = _exp2_3;
                                    float _exp2_4 = approx_exp2((m_k - m_new) * softmax_scale_log2);
                                    float b_k = _exp2_4;
                                    float _fma_2 = __fmaf_rn(l_k, b_k, l_f * a_k);
                                    l_f = _fma_2;
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
                                        float _fma_3 = __fmaf_rn(_vec_load_0[k], b_k, acc_f[k] * a_k);
                                        acc_f[k] = _fma_3;
                                    }
                                    m_f = m_new;
                                }
                                float _rcp_4 = approx_rcp(l_f);
                                float inv_f = ((l_f > 0.0f) ? _rcp_4 * output_scale : 0.0f);
                                #pragma unroll
                                for (int k4 = 0; k4 < 4; k4++) {
                                    out4[k4] = acc_f[k4] * inv_f;
                                }
                                {
                                    uint2 _pk2;
                                    __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
                                    _pk[0] = __floats2bfloat162_rn(out4[0 + 0], out4[0 + 1]);
                                    _pk[1] = __floats2bfloat162_rn(out4[0 + 2], out4[0 + 3]);
                                    *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(O_ptr + (o_idx_n + g_r * 4)))[0]) = _pk2;
                                }
                                if (store_lse_r != 0) {
                                    if (g_r == 0) {
                                        float lse_r = -CAKE_FMHA_INF;
                                        if (l_f > 0.0f) {
                                            float _log2_1;
                                            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_1) : "f"(l_f));
                                            lse_r = m_f * softmax_scale_log2 + _log2_1 - 0.8073549f;
                                        }
                                        *(reinterpret_cast<float*>(LSE_ptr + lse_idx_r) + (0)) = lse_r;
                                    }
                                }
                            }
                        } else {
                            int _max_9 = ((f_r / 16) > (1) ? (f_r / 16) : (1));
                            int n_span_r = _max_9;
                            int _min_8 = ((f_r / 4) < (4) ? (f_r / 4) : (4));
                            int active_r = _min_8;
                            if (active_r > (128 + sm_tid) % 4) {
                                #pragma unroll 1
                                for (int w_r = 0; w_r < n_span_r; w_r++) {
                                    int store_lse_w = ((w_r == 0) ? store_lse_r : 0);
                                    float acc_w[16];
                                    acc_w[0] = 0.0f;
                                    acc_w[1] = 0.0f;
                                    acc_w[2] = 0.0f;
                                    acc_w[3] = 0.0f;
                                    acc_w[4] = 0.0f;
                                    acc_w[5] = 0.0f;
                                    acc_w[6] = 0.0f;
                                    acc_w[7] = 0.0f;
                                    acc_w[8] = 0.0f;
                                    acc_w[9] = 0.0f;
                                    acc_w[10] = 0.0f;
                                    acc_w[11] = 0.0f;
                                    acc_w[12] = 0.0f;
                                    acc_w[13] = 0.0f;
                                    acc_w[14] = 0.0f;
                                    acc_w[15] = 0.0f;
                                    float m_w = -1e+30f;
                                    float l_w = 0.0f;
                                    int n_pad_w = (n_chunks_s + 3) / 4 * 4;
                                    #pragma unroll 1
                                    for (int c_b = 0; c_b < n_pad_w; c_b += 4) {
                                        float m_b[4];
                                        float l_b[4];
                                        float o_b[64];
                                        #pragma unroll
                                        for (int cb = 0; cb < 4; cb++) {
                                            int c_c_1 = c_b + cb;
                                            if (c_c_1 >= n_chunks_s) {
                                                c_c_1 = n_chunks_s - 1;
                                            }
                                            m_b[cb] = partial_stats[stats_row_r + c_c_1 * stats_stride_r];
                                            l_b[cb] = partial_stats[stats_row_r + 64 + c_c_1 * stats_stride_r];
                                            #pragma unroll
                                            for (int g = 0; g < 4; g++) {
                                                {
                                                    float4 _v4 = *reinterpret_cast<const float4*>(partial_o + o_row_r + w_r * 64 * 64 + g * 4 * 64 + c_c_1 * o_stride_r);
                                                    o_b[cb * 16 + g * 4 + 0] = _v4.x;
                                                    o_b[cb * 16 + g * 4 + 1] = _v4.y;
                                                    o_b[cb * 16 + g * 4 + 2] = _v4.z;
                                                    o_b[cb * 16 + g * 4 + 3] = _v4.w;
                                                }
                                            }
                                        }
                                        #pragma unroll
                                        for (int cb2 = 0; cb2 < 4; cb2++) {
                                            float m_k_1 = m_b[cb2];
                                            float l_k_1 = l_b[cb2];
                                            if (n_chunks_s <= c_b + cb2) {
                                                m_k_1 = -1e+30f;
                                                l_k_1 = 0.0f;
                                            }
                                            float _max_10 = max_noftz(m_w, m_k_1);
                                            float m_new_1 = _max_10;
                                            float _exp2_5 = approx_exp2((m_w - m_new_1) * softmax_scale_log2);
                                            float a_k_1 = _exp2_5;
                                            float _exp2_6 = approx_exp2((m_k_1 - m_new_1) * softmax_scale_log2);
                                            float b_k_1 = _exp2_6;
                                            float _fma_4 = __fmaf_rn(l_k_1, b_k_1, l_w * a_k_1);
                                            l_w = _fma_4;
                                            #pragma unroll
                                            for (int k2 = 0; k2 < 16; k2++) {
                                                float _fma_5 = __fmaf_rn(o_b[cb2 * 16 + k2], b_k_1, acc_w[k2] * a_k_1);
                                                acc_w[k2] = _fma_5;
                                            }
                                            m_w = m_new_1;
                                        }
                                    }
                                    float _rcp_5 = approx_rcp(l_w);
                                    float inv_w = ((l_w > 0.0f) ? _rcp_5 * output_scale : 0.0f);
                                    float out_w[4];
                                    #pragma unroll
                                    for (int g3 = 0; g3 < 4; g3++) {
                                        #pragma unroll
                                        for (int k3 = 0; k3 < 4; k3++) {
                                            out_w[k3] = acc_w[g3 * 4 + k3] * inv_w;
                                        }
                                        {
                                            uint2 _pk2;
                                            __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
                                            _pk[0] = __floats2bfloat162_rn(out_w[0 + 0], out_w[0 + 1]);
                                            _pk[1] = __floats2bfloat162_rn(out_w[0 + 2], out_w[0 + 3]);
                                            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(O_ptr + (o_idx_r + w_r * 64 + g3 * 4)))[0]) = _pk2;
                                        }
                                    }
                                    if (store_lse_w != 0) {
                                        float lse_w = -CAKE_FMHA_INF;
                                        if (l_w > 0.0f) {
                                            float _log2_2;
                                            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_2) : "f"(l_w));
                                            lse_w = m_w * softmax_scale_log2 + _log2_2 - 0.8073549f;
                                        }
                                        *(reinterpret_cast<float*>(LSE_ptr + lse_idx_r) + (0)) = lse_w;
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
                unsigned int q_tile_13 = work_token_words[base_0 + 12];
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
                q_tile_s = (int)q_tile_13;
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
            int o_row_base_g1 = taddr + 384 + (unsigned int)corr_row;
            int my_s_base_c = taddr + (unsigned int)corr_row;
            int tok_base_c = half_c * 64 + 32;
            int rows_live_c = ((warp_in_wg_c * 16 < N_ROWS) ? 1 : 0);
            int row_j_c = my_row_c / 16;
            float _rcp_7 = approx_rcp(softmax_scale_log2);
            float thr_raw_c = 8.0f * _rcp_7;
            float p_scale_log2_c = 0.8073549f;
            int items_per_chunk_c = num_kv_heads * ((q_len + 4 - 1) / 4);
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
                        int spec_pg_c = lane & 1;
                        int spec_block_c = spec_blocks_c - 1 - spec_nb_c;
                        if (spec_block_c >= 0) {
                            if ((lane & 15) < 2) {
                                int spec_page_idx_c = spec_block_c * 2 + spec_pg_c;
                                if (spec_page_idx_c > spec_max_pg_c) {
                                    spec_page_idx_c = spec_max_pg_c;
                                }
                                int spec_page_c = page_table[spec_batch_c * max_pages_per_seq + spec_page_idx_c];
                                asm volatile("cp.async.bulk.prefetch.tensor.5d.L2.global.tile [%0, {%1, %2, %3, %4, %5}];" :: "l"((uint64_t)((&K))), "r"((int)(0)), "r"((int)(0)), "r"((int)(0)), "r"((int)(spec_head_c)), "r"((int)(spec_page_c)) : "memory");
                                asm volatile("cp.async.bulk.prefetch.tensor.5d.L2.global.tile [%0, {%1, %2, %3, %4, %5}];" :: "l"((uint64_t)((&K))), "r"((int)(0)), "r"((int)(0)), "r"((int)(1)), "r"((int)(spec_head_c)), "r"((int)(spec_page_c)) : "memory");
                                if (spec_nb_c == 0) {
                                    asm volatile("cp.async.bulk.prefetch.tensor.5d.L2.global.tile [%0, {%1, %2, %3, %4, %5}];" :: "l"((uint64_t)((&V))), "r"((int)(0)), "r"((int)(0)), "r"((int)(0)), "r"((int)(spec_head_c)), "r"((int)(spec_page_c)) : "memory");
                                    asm volatile("cp.async.bulk.prefetch.tensor.5d.L2.global.tile [%0, {%1, %2, %3, %4, %5}];" :: "l"((uint64_t)((&V))), "r"((int)(0)), "r"((int)(0)), "r"((int)(1)), "r"((int)(spec_head_c)), "r"((int)(spec_page_c)) : "memory");
                                }
                            }
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
            unsigned int q_tile_1 = work_token_words[base_1 + 12];
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
            int q_tile_c = (int)q_tile_1;
            if (valid_c != 0) {
                if (kind_c == 0) {
                    int _min_9 = ((q_len - q_tile_c * 4) < (4) ? (q_len - q_tile_c * 4) : (4));
                    int live_rows_wc = _min_9 * 16;
                    int k2x_c = 0;
                    if (live_rows_wc > 32) {
                        if ((int)block_end_1 - (int)block_begin_1 >= 1) {
                            k2x_c = 1;
                        }
                    }
                    mbarrier_wait(q_raw_full_addr, 0);
                    int q_hi_base_s_1 = smem_q_hi_g0_addr;
                    int q_lo_base_s_2 = smem_q_lo_g0_addr;
                    int q_key_bf_s_1 = smem_qbf_addr / 128 % 8;
                    int q_key_hi_s_1 = q_hi_base_s_1 / 128 % 8;
                    int q_key_lo_s_1 = q_lo_base_s_2 / 128 % 8;
                    unsigned int q_words_s_1[4];
                    float q_f32_s_1[8];
                    float q_res_s_1[8];
                    unsigned int q_packed_s_1[2];
                    unsigned int q_packed_lo_s_1[2];
                    #pragma unroll 1
                    for (int qc_i_3 = 0; qc_i_3 < 4; qc_i_3++) {
                        int q_chunk_s_3 = qc_i_3 * 256 + wg_tid_c;
                        int q_row_s_3 = q_chunk_s_3 / 32;
                        int q_drow_s_3 = q_row_s_3;
                        int q_col8_s_3 = q_chunk_s_3 % 32;
                        int q_kg_s_3 = q_col8_s_3 / 8;
                        int q_c16a_s_3 = q_col8_s_3 % 8;
                        int q_grp_s_3 = q_col8_s_3 / 16;
                        int q_c16_s_3 = q_col8_s_3 % 16 / 2;
                        int q_half_s_3 = q_col8_s_3 % 2 * 8;
                        if (q_drow_s_3 < live_rows_wc) {
                            int q_key_row_s_3 = (q_key_bf_s_1 + q_row_s_3) % 8;
                            int q_src_s_3 = smem_qbf_addr + (unsigned int)(q_kg_s_3 * 4096) + (unsigned int)(q_row_s_3 * 128) + (unsigned int)((q_c16a_s_3 ^ q_key_row_s_3) * 16);
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_1[(0) + 3]))
                                : "r"(q_src_s_3));
                            #pragma unroll
                            for (int qw_s_3 = 0; qw_s_3 < 4; qw_s_3++) {
                                unsigned int q_w_s_3 = q_words_s_1[qw_s_3];
                                unsigned int q_wbits_d_3 = q_w_s_3 << 16;
                                float q_lo_d_3 = 0.0f;
                                float q_lo_t_d_3 = 0.0f;
                                float q_hi_d_3 = 0.0f;
                                float q_hi_t_d_3 = 0.0f;
                                q_lo_d_3 = reinterpret_cast<float*>(&q_wbits_d_3)[0];
                                q_wbits_d_3 = (q_w_s_3 & 65520) << 16;
                                q_lo_t_d_3 = reinterpret_cast<float*>(&q_wbits_d_3)[0];
                                q_wbits_d_3 = q_w_s_3 >> 16 << 16;
                                q_hi_d_3 = reinterpret_cast<float*>(&q_wbits_d_3)[0];
                                q_wbits_d_3 = (q_w_s_3 >> 16 & 65520) << 16;
                                q_hi_t_d_3 = reinterpret_cast<float*>(&q_wbits_d_3)[0];
                                q_f32_s_1[2 * qw_s_3] = q_lo_t_d_3;
                                q_res_s_1[2 * qw_s_3] = q_lo_d_3 - q_lo_t_d_3;
                                q_f32_s_1[2 * qw_s_3 + 1] = q_hi_t_d_3;
                                q_res_s_1[2 * qw_s_3 + 1] = q_hi_d_3 - q_hi_t_d_3;
                            }
                        } else {
                            #pragma unroll
                            for (int qz_s_3 = 0; qz_s_3 < 8; qz_s_3++) {
                                q_f32_s_1[qz_s_3] = 0.0f;
                                q_res_s_1[qz_s_3] = 0.0f;
                            }
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
                                : "=r"(_packed) : "f"(q_f32_s_1[0]), "f"(q_f32_s_1[1]),
                                                   "f"(q_f32_s_1[2]), "f"(q_f32_s_1[3]));
                            q_packed_s_1[0] = _packed;
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
                                : "=r"(_packed) : "f"(q_f32_s_1[4]), "f"(q_f32_s_1[5]),
                                                   "f"(q_f32_s_1[6]), "f"(q_f32_s_1[7]));
                            q_packed_s_1[1] = _packed;
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
                                : "=r"(_packed) : "f"(q_res_s_1[0]), "f"(q_res_s_1[1]),
                                                   "f"(q_res_s_1[2]), "f"(q_res_s_1[3]));
                            q_packed_lo_s_1[0] = _packed;
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
                                : "=r"(_packed) : "f"(q_res_s_1[4]), "f"(q_res_s_1[5]),
                                                   "f"(q_res_s_1[6]), "f"(q_res_s_1[7]));
                            q_packed_lo_s_1[1] = _packed;
                        }
                        int q_row_off_s_3 = q_grp_s_3 * 8192 + q_drow_s_3 * 128 + q_half_s_3;
                        int q_hi_addr_s_3 = q_hi_base_s_1 + q_row_off_s_3 + (q_c16_s_3 ^ (q_key_hi_s_1 + q_drow_s_3) % 8) * 16;
                        int q_lo_addr_s_3 = q_lo_base_s_2 + q_row_off_s_3 + (q_c16_s_3 ^ (q_key_lo_s_1 + q_drow_s_3) % 8) * 16;
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_hi_addr_s_3), "r"((q_packed_s_1[0])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_hi_addr_s_3 + 4), "r"((q_packed_s_1[1])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_lo_addr_s_3), "r"((q_packed_lo_s_1[0])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_lo_addr_s_3 + 4), "r"((q_packed_lo_s_1[1])));
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    if (k2x_c != 0) {
                        int q_hi_base_s_0_2 = smem_q_hi_g0_addr;
                        int q_lo_base_s_1_2 = smem_q_lo_g0_addr;
                        int q_key_bf_s_2_2 = (smem_k_g0_addr + 32768) / 128 % 8;
                        int q_key_hi_s_3_2 = q_hi_base_s_0_2 / 128 % 8;
                        int q_key_lo_s_4_2 = q_lo_base_s_1_2 / 128 % 8;
                        unsigned int q_words_s_5_2[4];
                        float q_f32_s_6_2[8];
                        float q_res_s_7_2[8];
                        unsigned int q_packed_s_8_2[2];
                        unsigned int q_packed_lo_s_9_2[2];
                        #pragma unroll 1
                        for (int qc_i_4 = 0; qc_i_4 < 4; qc_i_4++) {
                            int q_chunk_s_4 = qc_i_4 * 256 + wg_tid_c;
                            int q_row_s_4 = q_chunk_s_4 / 32;
                            int q_drow_s_4 = 32 + q_row_s_4;
                            int q_col8_s_4 = q_chunk_s_4 % 32;
                            int q_kg_s_4 = q_col8_s_4 / 8;
                            int q_c16a_s_4 = q_col8_s_4 % 8;
                            int q_grp_s_4 = q_col8_s_4 / 16;
                            int q_c16_s_4 = q_col8_s_4 % 16 / 2;
                            int q_half_s_4 = q_col8_s_4 % 2 * 8;
                            if (q_drow_s_4 < live_rows_wc) {
                                int q_key_row_s_4 = (q_key_bf_s_2_2 + q_row_s_4) % 8;
                                int q_src_s_4 = smem_k_g0_addr + 32768 + (unsigned int)(q_kg_s_4 * 4096) + (unsigned int)(q_row_s_4 * 128) + (unsigned int)((q_c16a_s_4 ^ q_key_row_s_4) * 16);
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_5_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_5_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_5_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_5_2[(0) + 3]))
                                    : "r"(q_src_s_4));
                                #pragma unroll
                                for (int qw_s_4 = 0; qw_s_4 < 4; qw_s_4++) {
                                    unsigned int q_w_s_4 = q_words_s_5_2[qw_s_4];
                                    unsigned int q_wbits_d_4 = q_w_s_4 << 16;
                                    float q_lo_d_4 = 0.0f;
                                    float q_lo_t_d_4 = 0.0f;
                                    float q_hi_d_4 = 0.0f;
                                    float q_hi_t_d_4 = 0.0f;
                                    q_lo_d_4 = reinterpret_cast<float*>(&q_wbits_d_4)[0];
                                    q_wbits_d_4 = (q_w_s_4 & 65520) << 16;
                                    q_lo_t_d_4 = reinterpret_cast<float*>(&q_wbits_d_4)[0];
                                    q_wbits_d_4 = q_w_s_4 >> 16 << 16;
                                    q_hi_d_4 = reinterpret_cast<float*>(&q_wbits_d_4)[0];
                                    q_wbits_d_4 = (q_w_s_4 >> 16 & 65520) << 16;
                                    q_hi_t_d_4 = reinterpret_cast<float*>(&q_wbits_d_4)[0];
                                    q_f32_s_6_2[2 * qw_s_4] = q_lo_t_d_4;
                                    q_res_s_7_2[2 * qw_s_4] = q_lo_d_4 - q_lo_t_d_4;
                                    q_f32_s_6_2[2 * qw_s_4 + 1] = q_hi_t_d_4;
                                    q_res_s_7_2[2 * qw_s_4 + 1] = q_hi_d_4 - q_hi_t_d_4;
                                }
                            } else {
                                #pragma unroll
                                for (int qz_s_4 = 0; qz_s_4 < 8; qz_s_4++) {
                                    q_f32_s_6_2[qz_s_4] = 0.0f;
                                    q_res_s_7_2[qz_s_4] = 0.0f;
                                }
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
                                    : "=r"(_packed) : "f"(q_f32_s_6_2[0]), "f"(q_f32_s_6_2[1]),
                                                       "f"(q_f32_s_6_2[2]), "f"(q_f32_s_6_2[3]));
                                q_packed_s_8_2[0] = _packed;
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
                                    : "=r"(_packed) : "f"(q_f32_s_6_2[4]), "f"(q_f32_s_6_2[5]),
                                                       "f"(q_f32_s_6_2[6]), "f"(q_f32_s_6_2[7]));
                                q_packed_s_8_2[1] = _packed;
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
                                    : "=r"(_packed) : "f"(q_res_s_7_2[0]), "f"(q_res_s_7_2[1]),
                                                       "f"(q_res_s_7_2[2]), "f"(q_res_s_7_2[3]));
                                q_packed_lo_s_9_2[0] = _packed;
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
                                    : "=r"(_packed) : "f"(q_res_s_7_2[4]), "f"(q_res_s_7_2[5]),
                                                       "f"(q_res_s_7_2[6]), "f"(q_res_s_7_2[7]));
                                q_packed_lo_s_9_2[1] = _packed;
                            }
                            int q_row_off_s_4 = q_grp_s_4 * 8192 + q_drow_s_4 * 128 + q_half_s_4;
                            int q_hi_addr_s_4 = q_hi_base_s_0_2 + q_row_off_s_4 + (q_c16_s_4 ^ (q_key_hi_s_3_2 + q_drow_s_4) % 8) * 16;
                            int q_lo_addr_s_4 = q_lo_base_s_1_2 + q_row_off_s_4 + (q_c16_s_4 ^ (q_key_lo_s_4_2 + q_drow_s_4) % 8) * 16;
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_hi_addr_s_4), "r"((q_packed_s_8_2[0])));
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_hi_addr_s_4 + 4), "r"((q_packed_s_8_2[1])));
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_lo_addr_s_4), "r"((q_packed_lo_s_9_2[0])));
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_lo_addr_s_4 + 4), "r"((q_packed_lo_s_9_2[1])));
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 9, 256;" ::: "memory");
                    } else {
                        asm volatile("barrier.sync 9, 256;" ::: "memory");
                        if (live_rows_wc > 32) {
                            mbarrier_wait(q_raw_full_addr, 1);
                        }
                        int q_hi_base_s_0_3 = smem_q_hi_g0_addr;
                        int q_lo_base_s_1_3 = smem_q_lo_g0_addr;
                        int q_key_bf_s_2_3 = smem_qbf_addr / 128 % 8;
                        int q_key_hi_s_3_3 = q_hi_base_s_0_3 / 128 % 8;
                        int q_key_lo_s_4_3 = q_lo_base_s_1_3 / 128 % 8;
                        unsigned int q_words_s_5_3[4];
                        float q_f32_s_6_3[8];
                        float q_res_s_7_3[8];
                        unsigned int q_packed_s_8_3[2];
                        unsigned int q_packed_lo_s_9_3[2];
                        #pragma unroll 1
                        for (int qc_i_5 = 0; qc_i_5 < 4; qc_i_5++) {
                            int q_chunk_s_5 = qc_i_5 * 256 + wg_tid_c;
                            int q_row_s_5 = q_chunk_s_5 / 32;
                            int q_drow_s_5 = 32 + q_row_s_5;
                            int q_col8_s_5 = q_chunk_s_5 % 32;
                            int q_kg_s_5 = q_col8_s_5 / 8;
                            int q_c16a_s_5 = q_col8_s_5 % 8;
                            int q_grp_s_5 = q_col8_s_5 / 16;
                            int q_c16_s_5 = q_col8_s_5 % 16 / 2;
                            int q_half_s_5 = q_col8_s_5 % 2 * 8;
                            if (q_drow_s_5 < live_rows_wc) {
                                int q_key_row_s_5 = (q_key_bf_s_2_3 + q_row_s_5) % 8;
                                int q_src_s_5 = smem_qbf_addr + (unsigned int)(q_kg_s_5 * 4096) + (unsigned int)(q_row_s_5 * 128) + (unsigned int)((q_c16a_s_5 ^ q_key_row_s_5) * 16);
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_5_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_5_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_5_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_5_3[(0) + 3]))
                                    : "r"(q_src_s_5));
                                #pragma unroll
                                for (int qw_s_5 = 0; qw_s_5 < 4; qw_s_5++) {
                                    unsigned int q_w_s_5 = q_words_s_5_3[qw_s_5];
                                    unsigned int q_wbits_d_5 = q_w_s_5 << 16;
                                    float q_lo_d_5 = 0.0f;
                                    float q_lo_t_d_5 = 0.0f;
                                    float q_hi_d_5 = 0.0f;
                                    float q_hi_t_d_5 = 0.0f;
                                    q_lo_d_5 = reinterpret_cast<float*>(&q_wbits_d_5)[0];
                                    q_wbits_d_5 = (q_w_s_5 & 65520) << 16;
                                    q_lo_t_d_5 = reinterpret_cast<float*>(&q_wbits_d_5)[0];
                                    q_wbits_d_5 = q_w_s_5 >> 16 << 16;
                                    q_hi_d_5 = reinterpret_cast<float*>(&q_wbits_d_5)[0];
                                    q_wbits_d_5 = (q_w_s_5 >> 16 & 65520) << 16;
                                    q_hi_t_d_5 = reinterpret_cast<float*>(&q_wbits_d_5)[0];
                                    q_f32_s_6_3[2 * qw_s_5] = q_lo_t_d_5;
                                    q_res_s_7_3[2 * qw_s_5] = q_lo_d_5 - q_lo_t_d_5;
                                    q_f32_s_6_3[2 * qw_s_5 + 1] = q_hi_t_d_5;
                                    q_res_s_7_3[2 * qw_s_5 + 1] = q_hi_d_5 - q_hi_t_d_5;
                                }
                            } else {
                                #pragma unroll
                                for (int qz_s_5 = 0; qz_s_5 < 8; qz_s_5++) {
                                    q_f32_s_6_3[qz_s_5] = 0.0f;
                                    q_res_s_7_3[qz_s_5] = 0.0f;
                                }
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
                                    : "=r"(_packed) : "f"(q_f32_s_6_3[0]), "f"(q_f32_s_6_3[1]),
                                                       "f"(q_f32_s_6_3[2]), "f"(q_f32_s_6_3[3]));
                                q_packed_s_8_3[0] = _packed;
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
                                    : "=r"(_packed) : "f"(q_f32_s_6_3[4]), "f"(q_f32_s_6_3[5]),
                                                       "f"(q_f32_s_6_3[6]), "f"(q_f32_s_6_3[7]));
                                q_packed_s_8_3[1] = _packed;
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
                                    : "=r"(_packed) : "f"(q_res_s_7_3[0]), "f"(q_res_s_7_3[1]),
                                                       "f"(q_res_s_7_3[2]), "f"(q_res_s_7_3[3]));
                                q_packed_lo_s_9_3[0] = _packed;
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
                                    : "=r"(_packed) : "f"(q_res_s_7_3[4]), "f"(q_res_s_7_3[5]),
                                                       "f"(q_res_s_7_3[6]), "f"(q_res_s_7_3[7]));
                                q_packed_lo_s_9_3[1] = _packed;
                            }
                            int q_row_off_s_5 = q_grp_s_5 * 8192 + q_drow_s_5 * 128 + q_half_s_5;
                            int q_hi_addr_s_5 = q_hi_base_s_0_3 + q_row_off_s_5 + (q_c16_s_5 ^ (q_key_hi_s_3_3 + q_drow_s_5) % 8) * 16;
                            int q_lo_addr_s_5 = q_lo_base_s_1_3 + q_row_off_s_5 + (q_c16_s_5 ^ (q_key_lo_s_4_3 + q_drow_s_5) % 8) * 16;
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_hi_addr_s_5), "r"((q_packed_s_8_3[0])));
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_hi_addr_s_5 + 4), "r"((q_packed_s_8_3[1])));
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_lo_addr_s_5), "r"((q_packed_lo_s_9_3[0])));
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_lo_addr_s_5 + 4), "r"((q_packed_lo_s_9_3[1])));
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 9, 256;" ::: "memory");
                    }
                }
            }
            #pragma unroll 1
            for (unsigned int _tile_iter_c = 0; _tile_iter_c < max_items; _tile_iter_c++) {
                if (valid_c == 0) {
                    break;
                }
                if (kind_c == 0) {
                    int cnt_c = block_end_c - block_begin_c;
                    int my_slot = slot_tile_base_c + chunk_c * items_per_chunk_c;
                    int _min_10 = ((q_len - q_tile_c * 4) < (4) ? (q_len - q_tile_c * 4) : (4));
                    int live_rows = _min_10 * 16;
                    int _min_11 = ((q_tile_c * 4 + row_j_c) < (q_len - 1) ? (q_tile_c * 4 + row_j_c) : (q_len - 1));
                    int vis_j_c = _min_11;
                    int back_c = q_len - 1 - vis_j_c - phase_c;
                    int vis_col_c = seqlen_c;
                    if (back_c > 0) {
                        vis_col_c = seqlen_c - (back_c + (1 << cp_world_log2) - 1 >> cp_world_log2);
                    }
                    float row_max_c = -CAKE_FMHA_INF;
                    float psum_c = 0.0f;
                    #pragma unroll 1
                    for (int n_1 = 0; n_1 < cnt_c; n_1++) {
                        if (wg_tid_c == 0) {
                        }
                        mbarrier_wait(s_full_addr + (sm_stage_c) * 8, sm_phase_c);
                        if (wg_tid_c == 0) {
                        }
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
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            int n_vis_c = vis_col_c - blk_pos_c;
                            if (my_row_c >= N_ROWS) {
                                n_vis_c = 0;
                            }
                            if (n_vis_c < 32) {
                                int _max_12 = ((n_vis_c) > (0) ? (n_vis_c) : (0));
                                int n_lo_c = _max_12;
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
                            float _max_13 = max_noftz(lmax_c, _shfl_xor_2);
                            lmax_c = _max_13;
                            if (half_c == 0) {
                                smem_xmax[xm_off_c + 64 + my_row_c] = lmax_c;
                            }
                        }
                        asm volatile("barrier.sync 8, 256;" ::: "memory");
                        if (rows_live_c != 0) {
                            float _max_14 = max_noftz(lmax_c, smem_xmax[xm_off_c + my_row_c]);
                            lmax_c = _max_14;
                            if (lmax_c > row_max_c + thr_raw_c) {
                                new_max_c = lmax_c;
                                if (row_max_c > -CAKE_FMHA_INF) {
                                    float _exp2_9 = approx_exp2(softmax_scale_log2 * (row_max_c - new_max_c));
                                    acc_scale_c = _exp2_9;
                                }
                            }
                        }
                        xm_slot_c = xm_slot_c ^ 1;
                        if (wg_tid_c == 0) {
                        }
                        if (rows_live_c != 0) {
                            float safe_max_c = ((new_max_c == -CAKE_FMHA_INF) ? 0.0f : new_max_c);
                            const float2 _fma_b2_2 = {softmax_scale_log2, softmax_scale_log2};
                            const float2 _fma_c2_3 = {p_scale_log2_c - safe_max_c * softmax_scale_log2, p_scale_log2_c - safe_max_c * softmax_scale_log2};
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
                            #pragma unroll
                            for (int _le = 0; _le < 32; _le++) {
                                sv_c[_le] = approx_exp2(sv_c[_le]);
                            }
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
                            unsigned int regs_pc[8];
                            {
                                uint32_t _packed;
                                asm volatile("{\n\t"
                                    ".reg .b16 _lo;\n\t"
                                    ".reg .b16 _hi;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                    "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                    "mov.b32 %0, {_lo, _hi};\n\t"
                                    "}"
                                    : "=r"(_packed) : "f"(sv_c[0]), "f"(sv_c[1]),
                                                       "f"(sv_c[2]), "f"(sv_c[3]));
                                regs_pc[0] = _packed;
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
                                    : "=r"(_packed) : "f"(sv_c[4]), "f"(sv_c[5]),
                                                       "f"(sv_c[6]), "f"(sv_c[7]));
                                regs_pc[1] = _packed;
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
                                    : "=r"(_packed) : "f"(sv_c[8]), "f"(sv_c[9]),
                                                       "f"(sv_c[10]), "f"(sv_c[11]));
                                regs_pc[2] = _packed;
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
                                    : "=r"(_packed) : "f"(sv_c[12]), "f"(sv_c[13]),
                                                       "f"(sv_c[14]), "f"(sv_c[15]));
                                regs_pc[3] = _packed;
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
                                    : "=r"(_packed) : "f"(sv_c[16]), "f"(sv_c[17]),
                                                       "f"(sv_c[18]), "f"(sv_c[19]));
                                regs_pc[4] = _packed;
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
                                    : "=r"(_packed) : "f"(sv_c[20]), "f"(sv_c[21]),
                                                       "f"(sv_c[22]), "f"(sv_c[23]));
                                regs_pc[5] = _packed;
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
                                    : "=r"(_packed) : "f"(sv_c[24]), "f"(sv_c[25]),
                                                       "f"(sv_c[26]), "f"(sv_c[27]));
                                regs_pc[6] = _packed;
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
                                    : "=r"(_packed) : "f"(sv_c[28]), "f"(sv_c[29]),
                                                       "f"(sv_c[30]), "f"(sv_c[31]));
                                regs_pc[7] = _packed;
                            }
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x32bx2.x8.b32"
                                " [%0], 16, {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "r"((unsigned int)my_s_base_c + sm_stage_c * (unsigned int)BLOCK_N + 8), "r"(*reinterpret_cast<const uint32_t*>(&regs_pc[0])), "r"(*reinterpret_cast<const uint32_t*>(&regs_pc[1])), "r"(*reinterpret_cast<const uint32_t*>(&regs_pc[2])), "r"(*reinterpret_cast<const uint32_t*>(&regs_pc[3])), "r"(*reinterpret_cast<const uint32_t*>(&regs_pc[4])), "r"(*reinterpret_cast<const uint32_t*>(&regs_pc[5])), "r"(*reinterpret_cast<const uint32_t*>(&regs_pc[6])), "r"(*reinterpret_cast<const uint32_t*>(&regs_pc[7])));
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        }
                        int _vote_0 = __any_sync(0xFFFFFFFF, acc_scale_c != 1.0f);
                        if (_vote_0 != 0) {
                            if (n_1 > 0) {
                                unsigned int k_c = pv_base + (unsigned int)n_1 - 1;
                                int o_st_c = (int)(k_c & 1);
                                int o_ph_c = (int)(k_c >> 1 & 1);
                                mbarrier_wait(o_ready_addr + (o_st_c) * 8, o_ph_c);
                                if (wg_tid_c == 0) {
                                }
                                asm volatile("tcgen05.fence::after_thread_sync;");
                                float _tmem_load_0[64];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[31]))
                                    : "r"(o_row_base));
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
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
                                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                                float _tmem_load_1[64];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[31]))
                                    : "r"(o_row_base_g1));
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[32])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[33])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[34])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[35])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[36])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[37])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[38])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[39])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[40])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[41])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[42])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[43])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[44])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[45])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[46])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[47])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[48])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[49])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[50])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[51])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[52])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[53])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[54])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[55])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[56])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[57])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[58])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[59])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[60])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[61])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[62])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[63]))
                                    : "r"(o_row_base_g1 + 32));
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                #if __CUDA_ARCH__ >= 1000
                                const float2 _scale2_22 = {acc_scale_c, acc_scale_c};
                                #pragma unroll
                                for (int _ls = 0; _ls < 32; _ls++)
                                    mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_1)[_ls], _scale2_22);
                                #else
                                #pragma unroll
                                for (int _ls = 0; _ls < 64; _ls++) {
                                    _tmem_load_1[_ls] = _tmem_load_1[_ls] * acc_scale_c;
                                }
                                #endif
                                asm volatile(
                                    "tcgen05.st.sync.aligned.16x32bx2.x64.b32"
                                    " [%0], 64, {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64};"
                                    :: "r"(o_row_base_g1), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[15])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[16])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[17])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[18])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[19])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[20])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[21])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[22])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[23])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[24])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[25])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[26])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[27])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[28])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[29])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[30])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[31])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[32])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[33])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[34])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[35])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[36])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[37])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[38])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[39])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[40])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[41])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[42])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[43])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[44])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[45])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[46])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[47])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[48])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[49])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[50])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[51])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[52])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[53])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[54])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[55])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[56])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[57])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[58])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[59])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[60])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[61])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[62])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[63])));
                                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            }
                        }
                        if (n_1 >= 2) {
                            unsigned int k_o = pv_base + (unsigned int)n_1 - 2;
                            int o_st_o = (int)(k_o & 1);
                            int o_ph_o = (int)(k_o >> 1 & 1);
                            mbarrier_wait(o_ready_addr + (o_st_o) * 8, o_ph_o);
                        }
                        __syncwarp();
                        if (lane == 0) {
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;"
                                :: "r"(p_full_addr + (sm_stage_c) * 8), "r"((uint32_t)(32)) : "memory");
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
                    unsigned int k_e = pv_base + (unsigned int)cnt_c - 1;
                    int o_st_e = (int)(k_e & 1);
                    int o_ph_e = (int)(k_e >> 1 & 1);
                    mbarrier_wait(o_ready_addr + (o_st_e) * 8, o_ph_e);
                    pv_base = pv_base + (unsigned int)cnt_c;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (wg_tid_c == 0) {
                    }
                    mbarrier_wait(stats_full_addr + (st_stage_c) * 8, st_phase_c);
                    int st_off = (int)st_stage_c * 64;
                    float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, psum_c, 16);
                    float total_c = psum_c + _shfl_xor_3;
                    if (rows_live_c != 0) {
                        if (half_c == 0) {
                            smem_sum[st_off + my_row_c] = smem_sum[st_off + my_row_c] + total_c;
                        }
                    }
                    asm volatile("barrier.sync 11, 128;" ::: "memory");
                    if (wg_tid_c == 0) {
                    }
                    int publish_split = ((n_chunks_c > 1) ? 1 : 0);
                    int d_g0 = half_c * 64;
                    if (publish_split == 0) {
                        float row_sum_c = 1.0f;
                        if (my_row_c < N_ROWS) {
                            row_sum_c = smem_sum[st_off + my_row_c];
                        }
                        float _rcp_8 = approx_rcp(row_sum_c);
                        float inv_c = ((row_sum_c > 0.0f) ? _rcp_8 : 0.0f);
                        int store_row_c = ((my_row_c < live_rows) ? 1 : 0);
                        int j_c = q_tile_c * 4 + my_row_c / 16;
                        int h_c = my_row_c % 16;
                        int q_head_c = kv_head_c * 16 + h_c;
                        int o_idx = ((batch_c * q_len + j_c) * num_q_heads + q_head_c) * HEAD_DIM + d_g0;
                        float _tmem_load_2[64];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[31]))
                            : "r"(o_row_base));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[32])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[33])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[34])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[35])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[36])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[37])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[38])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[39])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[40])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[41])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[42])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[43])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[44])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[45])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[46])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[47])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[48])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[49])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[50])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[51])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[52])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[53])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[54])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[55])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[56])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[57])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[58])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[59])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[60])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[61])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[62])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[63]))
                            : "r"(o_row_base + 32));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        if (row_sum_c <= 0.0f) {
                            #pragma unroll
                            for (int z_c = 0; z_c < 64; z_c++) {
                                _tmem_load_2[z_c] = 0.0f;
                            }
                        }
                        if (store_row_c != 0) {
                            #pragma unroll
                            for (int off = 0; off < 64; off += 8) {
                                {
                                    const float2 _prescale2_23 = {inv_c * output_scale, inv_c * output_scale};
                                    #if __CUDA_ARCH__ >= 1000
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 4; _ps++)
                                        mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_2[off])[_ps], _prescale2_23);
                                    #else
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 8; _ps++)
                                        _tmem_load_2[off + _ps] *= inv_c * output_scale;
                                    #endif
                                    __nv_bfloat162 _pk[4];
                                    _pk[0] = __floats2bfloat162_rn(_tmem_load_2[off + 0], _tmem_load_2[off + 1]);
                                    _pk[1] = __floats2bfloat162_rn(_tmem_load_2[off + 2], _tmem_load_2[off + 3]);
                                    _pk[2] = __floats2bfloat162_rn(_tmem_load_2[off + 4], _tmem_load_2[off + 5]);
                                    _pk[3] = __floats2bfloat162_rn(_tmem_load_2[off + 6], _tmem_load_2[off + 7]);
                                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O_ptr + (o_idx + off)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                }
                            }
                        }
                        float _tmem_load_3[64];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[31]))
                            : "r"(o_row_base_g1));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[32])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[33])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[34])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[35])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[36])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[37])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[38])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[39])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[40])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[41])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[42])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[43])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[44])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[45])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[46])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[47])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[48])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[49])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[50])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[51])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[52])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[53])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[54])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[55])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[56])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[57])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[58])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[59])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[60])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[61])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[62])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[63]))
                            : "r"(o_row_base_g1 + 32));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        __syncwarp();
                        if (lane == 0) {
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;"
                                :: "r"(o_empty_addr), "r"((uint32_t)(32)) : "memory");
                        }
                        if (row_sum_c <= 0.0f) {
                            #pragma unroll
                            for (int z1_c = 0; z1_c < 64; z1_c++) {
                                _tmem_load_3[z1_c] = 0.0f;
                            }
                        }
                        if (store_row_c != 0) {
                            #pragma unroll
                            for (int off1 = 0; off1 < 64; off1 += 8) {
                                {
                                    const float2 _prescale2_24 = {inv_c * output_scale, inv_c * output_scale};
                                    #if __CUDA_ARCH__ >= 1000
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 4; _ps++)
                                        mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_3[off1])[_ps], _prescale2_24);
                                    #else
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 8; _ps++)
                                        _tmem_load_3[off1 + _ps] *= inv_c * output_scale;
                                    #endif
                                    __nv_bfloat162 _pk[4];
                                    _pk[0] = __floats2bfloat162_rn(_tmem_load_3[off1 + 0], _tmem_load_3[off1 + 1]);
                                    _pk[1] = __floats2bfloat162_rn(_tmem_load_3[off1 + 2], _tmem_load_3[off1 + 3]);
                                    _pk[2] = __floats2bfloat162_rn(_tmem_load_3[off1 + 4], _tmem_load_3[off1 + 5]);
                                    _pk[3] = __floats2bfloat162_rn(_tmem_load_3[off1 + 6], _tmem_load_3[off1 + 7]);
                                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O_ptr + (o_idx + 128 + off1)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                }
                            }
                            if (half_c == 0) {
                                float lse_c = -CAKE_FMHA_INF;
                                if (row_sum_c > 0.0f) {
                                    float _log2_4;
                                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_4) : "f"(row_sum_c));
                                    lse_c = smem_max[st_off + my_row_c] * softmax_scale_log2 + _log2_4 - p_scale_log2_c;
                                }
                                *(reinterpret_cast<float*>(LSE_ptr + ((batch_c * q_len + j_c) * num_q_heads + q_head_c)) + (0)) = lse_c;
                            }
                        }
                        __syncwarp();
                        if (elect_sync()) {
                            mbarrier_arrive(stats_empty_addr + (st_stage_c) * 8);
                        }
                    } else {
                        int p_base = my_slot * 16384 + my_row_c * 4 + d_g0 * 64;
                        float _tmem_load_4[64];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[31]))
                            : "r"(o_row_base));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[32])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[33])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[34])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[35])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[36])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[37])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[38])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[39])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[40])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[41])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[42])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[43])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[44])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[45])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[46])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[47])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[48])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[49])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[50])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[51])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[52])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[53])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[54])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[55])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[56])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[57])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[58])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[59])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[60])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[61])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[62])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[63]))
                            : "r"(o_row_base + 32));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        if (my_row_c < N_ROWS) {
                            #pragma unroll
                            for (int off_1 = 0; off_1 < 64; off_1 += 4) {
                                {
                                    float4 _v4 = make_float4(_tmem_load_4[off_1 + 0], _tmem_load_4[off_1 + 1], _tmem_load_4[off_1 + 2], _tmem_load_4[off_1 + 3]);
                                    *reinterpret_cast<float4*>((partial_o + (p_base + off_1 * 64)) + 0) = _v4;
                                }
                            }
                        }
                        float _tmem_load_5[64];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[31]))
                            : "r"(o_row_base_g1));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[32])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[33])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[34])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[35])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[36])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[37])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[38])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[39])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[40])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[41])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[42])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[43])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[44])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[45])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[46])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[47])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[48])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[49])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[50])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[51])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[52])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[53])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[54])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[55])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[56])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[57])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[58])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[59])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[60])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[61])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[62])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[63]))
                            : "r"(o_row_base_g1 + 32));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        __syncwarp();
                        if (lane == 0) {
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;"
                                :: "r"(o_empty_addr), "r"((uint32_t)(32)) : "memory");
                        }
                        if (my_row_c < N_ROWS) {
                            #pragma unroll
                            for (int off1_1 = 0; off1_1 < 64; off1_1 += 4) {
                                {
                                    float4 _v4 = make_float4(_tmem_load_5[off1_1 + 0], _tmem_load_5[off1_1 + 1], _tmem_load_5[off1_1 + 2], _tmem_load_5[off1_1 + 3]);
                                    *reinterpret_cast<float4*>((partial_o + (p_base + (128 + off1_1) * 64)) + 0) = _v4;
                                }
                            }
                        }
                        if (wg_tid_c < N_ROWS) {
                            *(reinterpret_cast<float*>(partial_stats + (my_slot * 128 + wg_tid_c)) + (0)) = smem_max[st_off + wg_tid_c];
                            *(reinterpret_cast<float*>(partial_stats + (my_slot * 128 + 64 + wg_tid_c)) + (0)) = smem_sum[st_off + wg_tid_c];
                        }
                        if (n_chunks_c == 2) {
                            asm volatile("barrier.sync 11, 128;" ::: "memory");
                            if (wg_tid_c == 0) {
                                {
                                    asm volatile("fence.release.gpu;" ::: "memory");
                                }
                                unsigned int _atomic_old_2;
                                asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                                    : "=r"(_atomic_old_2) : "l"(&tile_counters[counter_idx_c * 4]), "r"(static_cast<uint32_t>(1)) : "memory");
                                unsigned int arrived_old = _atomic_old_2;
                                smem_corr_flag[0] = arrived_old + 1;
                            }
                            asm volatile("barrier.sync 11, 128;" ::: "memory");
                            unsigned int arrived_c = smem_corr_flag[0];
                            if ((int)arrived_c == n_chunks_c) {
                                if (wg_tid_c == 0) {
                                }
                                {
                                    asm volatile("fence.acquire.gpu;" ::: "memory");
                                }
                                int other_slot = slot_tile_base_c + (1 - chunk_c) * items_per_chunk_c;
                                float w_s_c = 0.0f;
                                float w_o_c = 0.0f;
                                float lse_i = -CAKE_FMHA_INF;
                                if (my_row_c < N_ROWS) {
                                    float m_o = partial_stats[other_slot * 128 + my_row_c];
                                    float l_o = partial_stats[other_slot * 128 + 64 + my_row_c];
                                    float m_s = smem_max[st_off + my_row_c];
                                    float l_s = smem_sum[st_off + my_row_c];
                                    float _max_15 = max_noftz(m_s, m_o);
                                    float m_row_i = _max_15;
                                    float _exp2_10 = approx_exp2((m_s - m_row_i) * softmax_scale_log2);
                                    float w_s = _exp2_10;
                                    float _exp2_11 = approx_exp2((m_o - m_row_i) * softmax_scale_log2);
                                    float w_o = _exp2_11;
                                    float _fma_8 = __fmaf_rn(w_s, l_s, w_o * l_o);
                                    float den_i = _fma_8;
                                    if (chunk_c != 0) {
                                        float _fma_9 = __fmaf_rn(w_o, l_o, w_s * l_s);
                                        den_i = _fma_9;
                                    }
                                    float _rcp_9 = approx_rcp(den_i);
                                    float inv_i = ((den_i > 0.0f) ? _rcp_9 * output_scale : 0.0f);
                                    w_s_c = w_s * inv_i;
                                    w_o_c = w_o * inv_i;
                                    if (den_i > 0.0f) {
                                        float _log2_5;
                                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_5) : "f"(den_i));
                                        lse_i = m_row_i * softmax_scale_log2 + _log2_5 - p_scale_log2_c;
                                    }
                                }
                                int oth_base = other_slot * 16384 + my_row_c * 4 + d_g0 * 64;
                                if (my_row_c < live_rows) {
                                    int j_c_1 = q_tile_c * 4 + my_row_c / 16;
                                    int h_c_1 = my_row_c % 16;
                                    int q_head_c_1 = kv_head_c * 16 + h_c_1;
                                    int o_idx_1 = ((batch_c * q_len + j_c_1) * num_q_heads + q_head_c_1) * HEAD_DIM + d_g0;
                                    float o_fold[64];
                                    #pragma unroll
                                    for (int c0 = 0; c0 < 64; c0 += 4) {
                                        float _vec_load_2[4];
                                        {
                                            float4 _v4 = *reinterpret_cast<const float4*>(partial_o + (p_base + c0 * 64) + 0);
                                            _vec_load_2[0 + 0] = _v4.x;
                                            _vec_load_2[0 + 1] = _v4.y;
                                            _vec_load_2[0 + 2] = _v4.z;
                                            _vec_load_2[0 + 3] = _v4.w;
                                        }
                                        float _vec_load_3[4];
                                        {
                                            float4 _v4 = *reinterpret_cast<const float4*>(partial_o + (oth_base + c0 * 64) + 0);
                                            _vec_load_3[0 + 0] = _v4.x;
                                            _vec_load_3[0 + 1] = _v4.y;
                                            _vec_load_3[0 + 2] = _v4.z;
                                            _vec_load_3[0 + 3] = _v4.w;
                                        }
                                        #pragma unroll
                                        for (int c = 0; c < 4; c++) {
                                            {
                                                float _fma_10 = __fmaf_rn(_vec_load_2[c], w_s_c, _vec_load_3[c] * w_o_c);
                                                float _fma_11 = __fmaf_rn(_vec_load_3[c], w_o_c, _vec_load_2[c] * w_s_c);
                                                float o_det_d1 = ((chunk_c == 0) ? _fma_10 : _fma_11);
                                                o_fold[c0 + c] = o_det_d1;
                                            }
                                        }
                                    }
                                    #pragma unroll
                                    for (int off_2 = 0; off_2 < 64; off_2 += 8) {
                                        {
                                            __nv_bfloat162 _pk[4];
                                            _pk[0] = __floats2bfloat162_rn(o_fold[off_2 + 0], o_fold[off_2 + 1]);
                                            _pk[1] = __floats2bfloat162_rn(o_fold[off_2 + 2], o_fold[off_2 + 3]);
                                            _pk[2] = __floats2bfloat162_rn(o_fold[off_2 + 4], o_fold[off_2 + 5]);
                                            _pk[3] = __floats2bfloat162_rn(o_fold[off_2 + 6], o_fold[off_2 + 7]);
                                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O_ptr + (o_idx_1 + off_2)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                        }
                                    }
                                    #pragma unroll
                                    for (int c1 = 0; c1 < 64; c1 += 4) {
                                        float _vec_load_4[4];
                                        {
                                            float4 _v4 = *reinterpret_cast<const float4*>(partial_o + (p_base + (128 + c1) * 64) + 0);
                                            _vec_load_4[0 + 0] = _v4.x;
                                            _vec_load_4[0 + 1] = _v4.y;
                                            _vec_load_4[0 + 2] = _v4.z;
                                            _vec_load_4[0 + 3] = _v4.w;
                                        }
                                        float _vec_load_5[4];
                                        {
                                            float4 _v4 = *reinterpret_cast<const float4*>(partial_o + (oth_base + (128 + c1) * 64) + 0);
                                            _vec_load_5[0 + 0] = _v4.x;
                                            _vec_load_5[0 + 1] = _v4.y;
                                            _vec_load_5[0 + 2] = _v4.z;
                                            _vec_load_5[0 + 3] = _v4.w;
                                        }
                                        #pragma unroll
                                        for (int c2 = 0; c2 < 4; c2++) {
                                            {
                                                float _fma_13 = __fmaf_rn(_vec_load_4[c2], w_s_c, _vec_load_5[c2] * w_o_c);
                                                float _fma_14 = __fmaf_rn(_vec_load_5[c2], w_o_c, _vec_load_4[c2] * w_s_c);
                                                float o_det_d2 = ((chunk_c == 0) ? _fma_13 : _fma_14);
                                                o_fold[c1 + c2] = o_det_d2;
                                            }
                                        }
                                    }
                                    #pragma unroll
                                    for (int off1_2 = 0; off1_2 < 64; off1_2 += 8) {
                                        {
                                            __nv_bfloat162 _pk[4];
                                            _pk[0] = __floats2bfloat162_rn(o_fold[off1_2 + 0], o_fold[off1_2 + 1]);
                                            _pk[1] = __floats2bfloat162_rn(o_fold[off1_2 + 2], o_fold[off1_2 + 3]);
                                            _pk[2] = __floats2bfloat162_rn(o_fold[off1_2 + 4], o_fold[off1_2 + 5]);
                                            _pk[3] = __floats2bfloat162_rn(o_fold[off1_2 + 6], o_fold[off1_2 + 7]);
                                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O_ptr + (o_idx_1 + 128 + off1_2)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                        }
                                    }
                                    if (half_c == 0) {
                                        *(reinterpret_cast<float*>(LSE_ptr + ((batch_c * q_len + j_c_1) * num_q_heads + q_head_c_1)) + (0)) = lse_i;
                                    }
                                }
                                asm volatile("barrier.sync 11, 128;" ::: "memory");
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
                            __syncwarp();
                            if (elect_sync()) {
                                mbarrier_arrive(stats_empty_addr + (st_stage_c) * 8);
                            }
                            asm volatile("barrier.sync 11, 128;" ::: "memory");
                            if (wg_tid_c == 0) {
                                {
                                    asm volatile("fence.release.gpu;" ::: "memory");
                                }
                                unsigned int _atomic_old_3;
                                asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                                    : "=r"(_atomic_old_3) : "l"(&tile_counters[counter_idx_c * 4]), "r"(static_cast<uint32_t>(1)) : "memory");
                                unsigned int arrived_old_1 = _atomic_old_3;
                                if ((int)arrived_old_1 + 1 == n_chunks_c) {
                                }
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
                    int _min_16 = ((q_len - q_tile_c * 4) < (4) ? (q_len - q_tile_c * 4) : (4));
                    int live_rows_r_1 = _min_16 * 16;
                    if (r_row_1 < live_rows_r_1) {
                        int items_r_1 = num_kv_heads * ((q_len + 4 - 1) / 4);
                        int stats_stride_r_1 = items_r_1 * 128;
                        int o_stride_r_1 = items_r_1 * 16384;
                        int stats_row_r_1 = slot_tile_base_c * 128 + r_row_1;
                        int f_r_1 = 64 >> block_end_c;
                        int d0_r_1 = block_begin_c * (4 * f_r_1) + wg_tid_c % 4 * f_r_1;
                        {
                            d0_r_1 = block_begin_c * (4 * f_r_1) + wg_tid_c % 4 * 16;
                        }
                        int o_row_r_1 = slot_tile_base_c * 16384 + r_row_1 * 4 + d0_r_1 * 64;
                        int j_r_1 = q_tile_c * 4 + r_row_1 / 16;
                        int h_r_1 = r_row_1 % 16;
                        int q_head_r_1 = kv_head_c * 16 + h_r_1;
                        int o_idx_r_1 = ((batch_c * q_len + j_r_1) * num_q_heads + q_head_r_1) * HEAD_DIM + d0_r_1;
                        int store_lse_r_1 = 0;
                        if (wg_tid_c % 4 == 0 && block_begin_c == 0) {
                            store_lse_r_1 = 1;
                        }
                        int lse_idx_r_1 = (batch_c * q_len + j_r_1) * num_q_heads + q_head_r_1;
                        int narrow_r_1 = 0;
                        if (narrow_r_1 != 0) {
                            int d0_n_1 = block_begin_c * (4 * f_r_1) + wg_tid_c % 4 * f_r_1;
                            int o_row_n_1 = slot_tile_base_c * 16384 + r_row_1 * 4 + d0_n_1 * 64;
                            int o_idx_n_1 = ((batch_c * q_len + j_r_1) * num_q_heads + q_head_r_1) * HEAD_DIM + d0_n_1;
                            float acc_f_1[4];
                            float out4_1[4];
                            int n_pad_r_1 = (n_chunks_c + 15) / 16 * 16;
                            #pragma unroll 1
                            for (int g_r_1 = 0; g_r_1 < 16 >> block_end_c; g_r_1++) {
                                acc_f_1[0] = 0.0f;
                                acc_f_1[1] = 0.0f;
                                acc_f_1[2] = 0.0f;
                                acc_f_1[3] = 0.0f;
                                float m_f_1 = -1e+30f;
                                float l_f_1 = 0.0f;
                                int o_col_r_1 = o_row_n_1 + g_r_1 * 4 * 64;
                                #pragma unroll 16
                                for (int c_m_1 = 0; c_m_1 < n_pad_r_1; c_m_1++) {
                                    int c_c_2 = c_m_1;
                                    if (n_chunks_c <= c_m_1) {
                                        c_c_2 = n_chunks_c - 1;
                                    }
                                    float m_k_2 = partial_stats[stats_row_r_1 + c_c_2 * stats_stride_r_1];
                                    float l_k_2 = partial_stats[stats_row_r_1 + 64 + c_c_2 * stats_stride_r_1];
                                    if (n_chunks_c <= c_m_1) {
                                        m_k_2 = -1e+30f;
                                        l_k_2 = 0.0f;
                                    }
                                    float _max_18 = max_noftz(m_f_1, m_k_2);
                                    float m_new_2 = _max_18;
                                    float _exp2_14 = approx_exp2((m_f_1 - m_new_2) * softmax_scale_log2);
                                    float a_k_2 = _exp2_14;
                                    float _exp2_15 = approx_exp2((m_k_2 - m_new_2) * softmax_scale_log2);
                                    float b_k_2 = _exp2_15;
                                    float _fma_18 = __fmaf_rn(l_k_2, b_k_2, l_f_1 * a_k_2);
                                    l_f_1 = _fma_18;
                                    float _vec_load_6[4];
                                    {
                                        float4 _v4 = *reinterpret_cast<const float4*>(partial_o + (o_col_r_1 + c_c_2 * o_stride_r_1) + 0);
                                        _vec_load_6[0 + 0] = _v4.x;
                                        _vec_load_6[0 + 1] = _v4.y;
                                        _vec_load_6[0 + 2] = _v4.z;
                                        _vec_load_6[0 + 3] = _v4.w;
                                    }
                                    #pragma unroll
                                    for (int k_1 = 0; k_1 < 4; k_1++) {
                                        float _fma_19 = __fmaf_rn(_vec_load_6[k_1], b_k_2, acc_f_1[k_1] * a_k_2);
                                        acc_f_1[k_1] = _fma_19;
                                    }
                                    m_f_1 = m_new_2;
                                }
                                float _rcp_11 = approx_rcp(l_f_1);
                                float inv_f_1 = ((l_f_1 > 0.0f) ? _rcp_11 * output_scale : 0.0f);
                                #pragma unroll
                                for (int k4_1 = 0; k4_1 < 4; k4_1++) {
                                    out4_1[k4_1] = acc_f_1[k4_1] * inv_f_1;
                                }
                                {
                                    uint2 _pk2;
                                    __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
                                    _pk[0] = __floats2bfloat162_rn(out4_1[0 + 0], out4_1[0 + 1]);
                                    _pk[1] = __floats2bfloat162_rn(out4_1[0 + 2], out4_1[0 + 3]);
                                    *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(O_ptr + (o_idx_n_1 + g_r_1 * 4)))[0]) = _pk2;
                                }
                                if (store_lse_r_1 != 0) {
                                    if (g_r_1 == 0) {
                                        float lse_r_1 = -CAKE_FMHA_INF;
                                        if (l_f_1 > 0.0f) {
                                            float _log2_7;
                                            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_7) : "f"(l_f_1));
                                            lse_r_1 = m_f_1 * softmax_scale_log2 + _log2_7 - 0.8073549f;
                                        }
                                        *(reinterpret_cast<float*>(LSE_ptr + lse_idx_r_1) + (0)) = lse_r_1;
                                    }
                                }
                            }
                        } else {
                            int _max_19 = ((f_r_1 / 16) > (1) ? (f_r_1 / 16) : (1));
                            int n_span_r_1 = _max_19;
                            int _min_17 = ((f_r_1 / 4) < (4) ? (f_r_1 / 4) : (4));
                            int active_r_1 = _min_17;
                            if (active_r_1 > wg_tid_c % 4) {
                                #pragma unroll 1
                                for (int w_r_1 = 0; w_r_1 < n_span_r_1; w_r_1++) {
                                    int store_lse_w_1 = ((w_r_1 == 0) ? store_lse_r_1 : 0);
                                    float acc_w_1[16];
                                    acc_w_1[0] = 0.0f;
                                    acc_w_1[1] = 0.0f;
                                    acc_w_1[2] = 0.0f;
                                    acc_w_1[3] = 0.0f;
                                    acc_w_1[4] = 0.0f;
                                    acc_w_1[5] = 0.0f;
                                    acc_w_1[6] = 0.0f;
                                    acc_w_1[7] = 0.0f;
                                    acc_w_1[8] = 0.0f;
                                    acc_w_1[9] = 0.0f;
                                    acc_w_1[10] = 0.0f;
                                    acc_w_1[11] = 0.0f;
                                    acc_w_1[12] = 0.0f;
                                    acc_w_1[13] = 0.0f;
                                    acc_w_1[14] = 0.0f;
                                    acc_w_1[15] = 0.0f;
                                    float m_w_1 = -1e+30f;
                                    float l_w_1 = 0.0f;
                                    int n_pad_w_1 = (n_chunks_c + 3) / 4 * 4;
                                    #pragma unroll 1
                                    for (int c_b_1 = 0; c_b_1 < n_pad_w_1; c_b_1 += 4) {
                                        float m_b_1[4];
                                        float l_b_1[4];
                                        float o_b_1[64];
                                        #pragma unroll
                                        for (int cb_1 = 0; cb_1 < 4; cb_1++) {
                                            int c_c_3 = c_b_1 + cb_1;
                                            if (c_c_3 >= n_chunks_c) {
                                                c_c_3 = n_chunks_c - 1;
                                            }
                                            m_b_1[cb_1] = partial_stats[stats_row_r_1 + c_c_3 * stats_stride_r_1];
                                            l_b_1[cb_1] = partial_stats[stats_row_r_1 + 64 + c_c_3 * stats_stride_r_1];
                                            #pragma unroll
                                            for (int g_1 = 0; g_1 < 4; g_1++) {
                                                {
                                                    float4 _v4 = *reinterpret_cast<const float4*>(partial_o + o_row_r_1 + w_r_1 * 64 * 64 + g_1 * 4 * 64 + c_c_3 * o_stride_r_1);
                                                    o_b_1[cb_1 * 16 + g_1 * 4 + 0] = _v4.x;
                                                    o_b_1[cb_1 * 16 + g_1 * 4 + 1] = _v4.y;
                                                    o_b_1[cb_1 * 16 + g_1 * 4 + 2] = _v4.z;
                                                    o_b_1[cb_1 * 16 + g_1 * 4 + 3] = _v4.w;
                                                }
                                            }
                                        }
                                        #pragma unroll
                                        for (int cb2_1 = 0; cb2_1 < 4; cb2_1++) {
                                            float m_k_3 = m_b_1[cb2_1];
                                            float l_k_3 = l_b_1[cb2_1];
                                            if (n_chunks_c <= c_b_1 + cb2_1) {
                                                m_k_3 = -1e+30f;
                                                l_k_3 = 0.0f;
                                            }
                                            float _max_20 = max_noftz(m_w_1, m_k_3);
                                            float m_new_3 = _max_20;
                                            float _exp2_16 = approx_exp2((m_w_1 - m_new_3) * softmax_scale_log2);
                                            float a_k_3 = _exp2_16;
                                            float _exp2_17 = approx_exp2((m_k_3 - m_new_3) * softmax_scale_log2);
                                            float b_k_3 = _exp2_17;
                                            float _fma_20 = __fmaf_rn(l_k_3, b_k_3, l_w_1 * a_k_3);
                                            l_w_1 = _fma_20;
                                            #pragma unroll
                                            for (int k2_1 = 0; k2_1 < 16; k2_1++) {
                                                float _fma_21 = __fmaf_rn(o_b_1[cb2_1 * 16 + k2_1], b_k_3, acc_w_1[k2_1] * a_k_3);
                                                acc_w_1[k2_1] = _fma_21;
                                            }
                                            m_w_1 = m_new_3;
                                        }
                                    }
                                    float _rcp_12 = approx_rcp(l_w_1);
                                    float inv_w_1 = ((l_w_1 > 0.0f) ? _rcp_12 * output_scale : 0.0f);
                                    float out_w_1[4];
                                    #pragma unroll
                                    for (int g3_1 = 0; g3_1 < 4; g3_1++) {
                                        #pragma unroll
                                        for (int k3_1 = 0; k3_1 < 4; k3_1++) {
                                            out_w_1[k3_1] = acc_w_1[g3_1 * 4 + k3_1] * inv_w_1;
                                        }
                                        {
                                            uint2 _pk2;
                                            __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
                                            _pk[0] = __floats2bfloat162_rn(out_w_1[0 + 0], out_w_1[0 + 1]);
                                            _pk[1] = __floats2bfloat162_rn(out_w_1[0 + 2], out_w_1[0 + 3]);
                                            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(O_ptr + (o_idx_r_1 + w_r_1 * 64 + g3_1 * 4)))[0]) = _pk2;
                                        }
                                    }
                                    if (store_lse_w_1 != 0) {
                                        float lse_w_1 = -CAKE_FMHA_INF;
                                        if (l_w_1 > 0.0f) {
                                            float _log2_8;
                                            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_8) : "f"(l_w_1));
                                            lse_w_1 = m_w_1 * softmax_scale_log2 + _log2_8 - 0.8073549f;
                                        }
                                        *(reinterpret_cast<float*>(LSE_ptr + lse_idx_r_1) + (0)) = lse_w_1;
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
                unsigned int q_tile_13_1 = work_token_words[base_0_1 + 12];
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
                q_tile_c = (int)q_tile_13_1;
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
            unsigned int q_tile_2 = work_token_words[base_2 + 12];
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
                    int _mma_a_lo_0 = make_warp_uniform((((smem_q_hi_g0_addr) >> 4) & 0x3FFF) + (q_cons_stage) * 512);
                    int _mma_b_lo_0 = make_warp_uniform((((smem_k_g0_addr) >> 4) & 0x3FFF) + (k_cons_stage) * 1024);
                    {
                        uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
                        uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 69206032, 0);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 69206032, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 69206032, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 69206032, 1);
                        }
                    }
                    int _mma_a_lo_1 = make_warp_uniform((((smem_q_lo_g0_addr) >> 4) & 0x3FFF) + (q_cons_stage) * 512);
                    int _mma_b_lo_1 = make_warp_uniform((((smem_k_g0_addr) >> 4) & 0x3FFF) + (k_cons_stage) * 1024);
                    {
                        uint64_t _mma_ss_a_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_1);
                        uint64_t _mma_ss_b_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_1);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 69206032, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 69206032, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 69206032, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 69206032, 1);
                        }
                    }
                    int _mma_a_lo_2 = make_warp_uniform((((smem_q_hi_g1_addr) >> 4) & 0x3FFF) + (q_cons_stage) * 512);
                    int _mma_b_lo_2 = make_warp_uniform((((smem_k_g1_addr) >> 4) & 0x3FFF) + (k_cons_stage) * 1024);
                    {
                        uint64_t _mma_ss_a_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_2);
                        uint64_t _mma_ss_b_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_2);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 69206032, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 69206032, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 69206032, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 69206032, 1);
                        }
                    }
                    int _mma_a_lo_3 = make_warp_uniform((((smem_q_lo_g1_addr) >> 4) & 0x3FFF) + (q_cons_stage) * 512);
                    int _mma_b_lo_3 = make_warp_uniform((((smem_k_g1_addr) >> 4) & 0x3FFF) + (k_cons_stage) * 1024);
                    {
                        uint64_t _mma_ss_a_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_3);
                        uint64_t _mma_ss_b_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_3);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 69206032, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 69206032, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 69206032, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 69206032, 1);
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
                            mbarrier_wait(k_full_addr + (k_cons_stage) * 8, k_cons_phase);
                            if (lane == 0) {
                            }
                            int _mma_a_lo_4 = make_warp_uniform((((smem_q_hi_g0_addr) >> 4) & 0x3FFF) + (q_cons_stage) * 512);
                            int _mma_b_lo_4 = make_warp_uniform((((smem_k_g0_addr) >> 4) & 0x3FFF) + (k_cons_stage) * 1024);
                            {
                                uint64_t _mma_ss_a_desc_4 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_4);
                                uint64_t _mma_ss_b_desc_4 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_4);
                                if (elect_sync()) {
                                    tcgen05_mma_f8f6f4((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_4, _mma_ss_b_desc_4, 69206032, 0);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_4, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f8f6f4((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_4, _mma_ss_b_desc_4, 69206032, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_4, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f8f6f4((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_4, _mma_ss_b_desc_4, 69206032, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_4, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f8f6f4((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_4, _mma_ss_b_desc_4, 69206032, 1);
                                }
                            }
                            int _mma_a_lo_5 = make_warp_uniform((((smem_q_lo_g0_addr) >> 4) & 0x3FFF) + (q_cons_stage) * 512);
                            int _mma_b_lo_5 = make_warp_uniform((((smem_k_g0_addr) >> 4) & 0x3FFF) + (k_cons_stage) * 1024);
                            {
                                uint64_t _mma_ss_a_desc_5 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_5);
                                uint64_t _mma_ss_b_desc_5 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_5);
                                if (elect_sync()) {
                                    tcgen05_mma_f8f6f4((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_5, _mma_ss_b_desc_5, 69206032, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_5, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f8f6f4((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_5, _mma_ss_b_desc_5, 69206032, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_5, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f8f6f4((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_5, _mma_ss_b_desc_5, 69206032, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_5, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f8f6f4((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_5, _mma_ss_b_desc_5, 69206032, 1);
                                }
                            }
                            int _mma_a_lo_6 = make_warp_uniform((((smem_q_hi_g1_addr) >> 4) & 0x3FFF) + (q_cons_stage) * 512);
                            int _mma_b_lo_6 = make_warp_uniform((((smem_k_g1_addr) >> 4) & 0x3FFF) + (k_cons_stage) * 1024);
                            {
                                uint64_t _mma_ss_a_desc_6 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_6);
                                uint64_t _mma_ss_b_desc_6 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_6);
                                if (elect_sync()) {
                                    tcgen05_mma_f8f6f4((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_6, _mma_ss_b_desc_6, 69206032, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_6, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f8f6f4((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_6, _mma_ss_b_desc_6, 69206032, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_6, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f8f6f4((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_6, _mma_ss_b_desc_6, 69206032, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_6, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f8f6f4((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_6, _mma_ss_b_desc_6, 69206032, 1);
                                }
                            }
                            int _mma_a_lo_7 = make_warp_uniform((((smem_q_lo_g1_addr) >> 4) & 0x3FFF) + (q_cons_stage) * 512);
                            int _mma_b_lo_7 = make_warp_uniform((((smem_k_g1_addr) >> 4) & 0x3FFF) + (k_cons_stage) * 1024);
                            {
                                uint64_t _mma_ss_a_desc_7 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_7);
                                uint64_t _mma_ss_b_desc_7 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_7);
                                if (elect_sync()) {
                                    tcgen05_mma_f8f6f4((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_7, _mma_ss_b_desc_7, 69206032, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_7, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f8f6f4((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_7, _mma_ss_b_desc_7, 69206032, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_7, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f8f6f4((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_7, _mma_ss_b_desc_7, 69206032, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_7, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f8f6f4((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_7, _mma_ss_b_desc_7, 69206032, 1);
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
                        mbarrier_wait(v_full_addr + (v_cons_stage) * 8, v_cons_phase);
                        if (lane == 0) {
                        }
                        mbarrier_wait(p_full_addr + (pf_stage) * 8, pf_phase);
                        if (lane == 0) {
                        }
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int first_pv_flag = first_pv;
                        int _mma_b_lo_8 = make_warp_uniform(((((smem_v_g0_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_cons_stage) * 1024);
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 69271568;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_o_g0), "r"(_mma_b_lo_8), "r"(tmem_tmem_s + (int)pf_stage * 128), "r"(((first_pv_flag) ? 0 : 1)));
                        int _mma_b_lo_9 = make_warp_uniform(((((smem_v_g1_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_cons_stage) * 1024);
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 69271568;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_o_g1), "r"(_mma_b_lo_9), "r"(tmem_tmem_s + (int)pf_stage * 128), "r"(((first_pv_flag) ? 0 : 1)));
                        int o_st_m = (int)(pv_idx_m & 1);
                        elect_commit(v_empty_addr + (v_cons_stage) * 8);
                        elect_commit(o_ready_addr + (o_st_m) * 8);
                        pv_idx_m = pv_idx_m + 1;
                        v_cons_stage += 1;
                        if (v_cons_stage == 2) { v_cons_stage = 0; v_cons_phase ^= 1; }
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
                unsigned int q_tile_13_2 = work_token_words[base_0_2 + 12];
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
            unsigned int q_prod_stage = 0;
            unsigned int q_prod_phase = 1;
            unsigned int qr_cons_stage = 0;
            unsigned int qr_cons_phase = 0;
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
            unsigned int q_tile_3 = work_token_words[base_3 + 12];
            mbarrier_arrive(work_empty_addr + (work_stage_p) * 8);
            work_stage_p += 1;
            if (work_stage_p == 4) { work_stage_p = 0; _phase_work_full_3 ^= 1; }
            unsigned int valid_p = valid_4;
            int kind_p = (int)kind_4;
            int batch_idx_p = (int)batch_4;
            int kv_head_p = (int)kv_head_3;
            int block_begin_p = (int)block_begin_3;
            int block_end_p = (int)block_end_3;
            int seqlen_kv_p = (int)seqlen_3;
            int q_tile_p = (int)q_tile_3;
            #pragma unroll 1
            for (unsigned int _tile_iter_p = 0; _tile_iter_p < max_items; _tile_iter_p++) {
                if (valid_p == 0) {
                    break;
                }
                if (kind_p == 0) {
                    int cta_n_blocks_p = block_end_p - block_begin_p;
                    int _max_2 = (((seqlen_kv_p + PAGE_SIZE - 1) / PAGE_SIZE - 1) > (0) ? ((seqlen_kv_p + PAGE_SIZE - 1) / PAGE_SIZE - 1) : (0));
                    int max_pg_p = _max_2;
                    int pt_base_p = batch_idx_p * max_pages_per_seq;
                    #pragma unroll 1
                    for (int ni0_p = 0; ni0_p < cta_n_blocks_p; ni0_p += 4) {
                        int g_cnt_p = cta_n_blocks_p - ni0_p;
                        if (g_cnt_p > 4) {
                            g_cnt_p = 4;
                        }
                        int n_block_p = block_begin_p + cta_n_blocks_p - 1 - (ni0_p + pg_blk_p);
                        int page_idx_p = n_block_p * 2 + pg_lane_p;
                        if (page_idx_p > max_pg_p) {
                            page_idx_p = max_pg_p;
                        }
                        int page_id_p = 0;
                        if (pg_blk_p < g_cnt_p) {
                            if (pg_lane_p < 2) {
                                page_id_p = page_table[pt_base_p + page_idx_p];
                            }
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
                        if (ni0_p == 0) {
                            int wide_done_p = 0;
                            if (_tile_iter_p == 0) {
                                wide_done_p = 1;
                            }
                            int _min_0 = ((q_len - q_tile_p * 4) < (4) ? (q_len - q_tile_p * 4) : (4));
                            int live_rows_p = _min_0 * 16;
                            if (wide_done_p == 0) {
                                mbarrier_wait(q_empty_addr + (q_prod_stage) * 8, q_prod_phase);
                                #pragma unroll
                                for (int qh_p = 0; qh_p < 2; qh_p++) {
                                    if (qh_p == 0) {
                                        mbarrier_wait(q_raw_full_addr + (qr_cons_stage) * 8, qr_cons_phase);
                                    }
                                    if (qh_p == 1) {
                                        if (live_rows_p > 32) {
                                            mbarrier_wait(q_raw_full_addr + (qr_cons_stage) * 8, qr_cons_phase);
                                        }
                                    }
                                    int q_hi_base_s_2 = smem_q_hi_g0_addr + q_prod_stage * 8192;
                                    int q_lo_base_s_3 = smem_q_lo_g0_addr + q_prod_stage * 8192;
                                    int q_key_bf_s_3 = (smem_qbf_addr + qr_cons_stage * 16384) / 128 % 8;
                                    int q_key_hi_s_2 = q_hi_base_s_2 / 128 % 8;
                                    int q_key_lo_s_2 = q_lo_base_s_3 / 128 % 8;
                                    unsigned int q_words_s_2[4];
                                    float q_f32_s_2[8];
                                    float q_res_s_2[8];
                                    unsigned int q_packed_s_2[2];
                                    unsigned int q_packed_lo_s_2[2];
                                    #pragma unroll 1
                                    for (int qc_i_6 = 0; qc_i_6 < 32; qc_i_6++) {
                                        int q_chunk_s_6 = qc_i_6 * 32 + lane;
                                        int q_row_s_6 = q_chunk_s_6 / 32;
                                        int q_drow_s_6 = qh_p * 32 + q_row_s_6;
                                        int q_col8_s_6 = q_chunk_s_6 % 32;
                                        int q_kg_s_6 = q_col8_s_6 / 8;
                                        int q_c16a_s_6 = q_col8_s_6 % 8;
                                        int q_grp_s_6 = q_col8_s_6 / 16;
                                        int q_c16_s_6 = q_col8_s_6 % 16 / 2;
                                        int q_half_s_6 = q_col8_s_6 % 2 * 8;
                                        if (q_drow_s_6 < live_rows_p) {
                                            int q_key_row_s_6 = (q_key_bf_s_3 + q_row_s_6) % 8;
                                            int q_src_s_6 = smem_qbf_addr + qr_cons_stage * 16384 + (unsigned int)(q_kg_s_6 * 4096) + (unsigned int)(q_row_s_6 * 128) + (unsigned int)((q_c16a_s_6 ^ q_key_row_s_6) * 16);
                                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                                : "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_2[(0) + 3]))
                                                : "r"(q_src_s_6));
                                            #pragma unroll
                                            for (int qw_s_6 = 0; qw_s_6 < 4; qw_s_6++) {
                                                unsigned int q_w_s_6 = q_words_s_2[qw_s_6];
                                                unsigned int q_wbits_d_6 = q_w_s_6 << 16;
                                                float q_lo_d_6 = 0.0f;
                                                float q_lo_t_d_6 = 0.0f;
                                                float q_hi_d_6 = 0.0f;
                                                float q_hi_t_d_6 = 0.0f;
                                                q_lo_d_6 = reinterpret_cast<float*>(&q_wbits_d_6)[0];
                                                q_wbits_d_6 = (q_w_s_6 & 65520) << 16;
                                                q_lo_t_d_6 = reinterpret_cast<float*>(&q_wbits_d_6)[0];
                                                q_wbits_d_6 = q_w_s_6 >> 16 << 16;
                                                q_hi_d_6 = reinterpret_cast<float*>(&q_wbits_d_6)[0];
                                                q_wbits_d_6 = (q_w_s_6 >> 16 & 65520) << 16;
                                                q_hi_t_d_6 = reinterpret_cast<float*>(&q_wbits_d_6)[0];
                                                q_f32_s_2[2 * qw_s_6] = q_lo_t_d_6;
                                                q_res_s_2[2 * qw_s_6] = q_lo_d_6 - q_lo_t_d_6;
                                                q_f32_s_2[2 * qw_s_6 + 1] = q_hi_t_d_6;
                                                q_res_s_2[2 * qw_s_6 + 1] = q_hi_d_6 - q_hi_t_d_6;
                                            }
                                        } else {
                                            #pragma unroll
                                            for (int qz_s_6 = 0; qz_s_6 < 8; qz_s_6++) {
                                                q_f32_s_2[qz_s_6] = 0.0f;
                                                q_res_s_2[qz_s_6] = 0.0f;
                                            }
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
                                                : "=r"(_packed) : "f"(q_f32_s_2[0]), "f"(q_f32_s_2[1]),
                                                                   "f"(q_f32_s_2[2]), "f"(q_f32_s_2[3]));
                                            q_packed_s_2[0] = _packed;
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
                                                : "=r"(_packed) : "f"(q_f32_s_2[4]), "f"(q_f32_s_2[5]),
                                                                   "f"(q_f32_s_2[6]), "f"(q_f32_s_2[7]));
                                            q_packed_s_2[1] = _packed;
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
                                                : "=r"(_packed) : "f"(q_res_s_2[0]), "f"(q_res_s_2[1]),
                                                                   "f"(q_res_s_2[2]), "f"(q_res_s_2[3]));
                                            q_packed_lo_s_2[0] = _packed;
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
                                                : "=r"(_packed) : "f"(q_res_s_2[4]), "f"(q_res_s_2[5]),
                                                                   "f"(q_res_s_2[6]), "f"(q_res_s_2[7]));
                                            q_packed_lo_s_2[1] = _packed;
                                        }
                                        int q_row_off_s_6 = q_grp_s_6 * 8192 + q_drow_s_6 * 128 + q_half_s_6;
                                        int q_hi_addr_s_6 = q_hi_base_s_2 + q_row_off_s_6 + (q_c16_s_6 ^ (q_key_hi_s_2 + q_drow_s_6) % 8) * 16;
                                        int q_lo_addr_s_6 = q_lo_base_s_3 + q_row_off_s_6 + (q_c16_s_6 ^ (q_key_lo_s_2 + q_drow_s_6) % 8) * 16;
                                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_hi_addr_s_6), "r"((q_packed_s_2[0])));
                                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_hi_addr_s_6 + 4), "r"((q_packed_s_2[1])));
                                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_lo_addr_s_6), "r"((q_packed_lo_s_2[0])));
                                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_lo_addr_s_6 + 4), "r"((q_packed_lo_s_2[1])));
                                    }
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    __syncwarp();
                                    if (qh_p == 0) {
                                        if (lane == 0) {
                                            mbarrier_arrive(q_raw_empty_addr + (qr_cons_stage) * 8);
                                        }
                                        qr_cons_phase ^= 1;
                                    }
                                    if (qh_p == 1) {
                                        if (live_rows_p > 32) {
                                            if (lane == 0) {
                                                mbarrier_arrive(q_raw_empty_addr + (qr_cons_stage) * 8);
                                            }
                                            qr_cons_phase ^= 1;
                                        }
                                    }
                                }
                                if (lane == 0) {
                                    mbarrier_arrive(q_full_addr + (q_prod_stage) * 8);
                                }
                            }
                            if (wide_done_p != 0) {
                                int k2x_p = 0;
                                if (live_rows_p > 32) {
                                    if (block_end_p - block_begin_p >= 1) {
                                        k2x_p = 1;
                                    }
                                }
                                qr_cons_phase ^= 1;
                                if (live_rows_p > 32) {
                                    if (k2x_p == 0) {
                                        qr_cons_phase ^= 1;
                                    }
                                }
                            }
                            q_prod_phase ^= 1;
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
                unsigned int q_tile_13_3 = work_token_words[base_0_3 + 12];
                mbarrier_arrive(work_empty_addr + (work_stage_p) * 8);
                work_stage_p += 1;
                if (work_stage_p == 4) { work_stage_p = 0; _phase_work_full_3 ^= 1; }
                valid_p = valid_1_3;
                kind_p = (int)kind_2_3;
                batch_idx_p = (int)batch_3_3;
                kv_head_p = (int)kv_head_4_3;
                block_begin_p = (int)block_begin_5_3;
                block_end_p = (int)block_end_6_3;
                seqlen_kv_p = (int)seqlen_7_3;
                q_tile_p = (int)q_tile_13_3;
            }
            if (lane == 0) {
            }
        }
    // ---- Role: scheduler ----
    } else if (warp == 10) {
        { // scheduler_main
            {
            }
            int lane_0 = lane;
            int num_ctas = gridDim.x;
            if (lane_0 == 0) {
            }
            int n_q_tiles = (q_len + 4 - 1) / 4;
            int items_per_chunk = num_kv_heads * n_q_tiles;
            int sow_pairs_bound = (max_pages_per_seq * PAGE_SIZE + 255) / 256;
            unsigned int sow_n_max = (unsigned int)(sow_pairs_bound + 1 - 1);
            unsigned int sow_items = (unsigned int)items_per_chunk;
            float _rcp_0 = approx_rcp((float)items_per_chunk);
            float sow_items_rcp = _rcp_0;
            unsigned int sow_tiles = (unsigned int)(batch_size * items_per_chunk);
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
            if (sow_n_max > 2) {
                sow_shift = sow_shift + 1;
            }
            unsigned int sow_one = 1;
            unsigned int sow_total = sow_chunk_items + (sow_tiles << sow_shift);
            unsigned int sow_pairs_u = 1;
            if (blockIdx.x == 0) {
                if (lane_0 == 0) {
                    int sow_plan_off = num_ctas * 2048;
                    *(reinterpret_cast<float*>(partial_stats + sow_plan_off) + (0)) = (float)sow_pairs_u;
                    *(reinterpret_cast<float*>(partial_stats + (sow_plan_off + 1)) + (0)) = (float)sow_total;
                }
            }
            unsigned int sow_ticket = blockIdx.x;
            unsigned int sow_valid = 0;
            unsigned int sow_kind = 0;
            int sow_batch = 0;
            int sow_item = 0;
            int sow_q_tile = 0;
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
            unsigned int sow_local_chunk = 0;
            unsigned int sow_rt = 0;
            unsigned int sow_r = 0;
            if (sow_ticket < sow_total) {
                if (sow_ticket < sow_chunk_items) {
                    unsigned int q = (unsigned int)((float)sow_ticket * sow_items_rcp);
                    if (sow_ticket < q * sow_items) {
                        q = q - 1;
                    }
                    if (sow_ticket >= (q + 1) * sow_items) {
                        q = q + 1;
                    }
                    sow_local_chunk = q;
                    sow_item = (int)(sow_ticket - sow_local_chunk * sow_items);
                    float _rcp_1 = approx_rcp((float)sow_n_max);
                    unsigned int q_0 = (unsigned int)((float)sow_local_chunk * _rcp_1);
                    if (sow_local_chunk < q_0 * sow_n_max) {
                        q_0 = q_0 - 1;
                    }
                    if (sow_local_chunk >= (q_0 + 1) * sow_n_max) {
                        q_0 = q_0 + 1;
                    }
                    sow_batch = (int)q_0;
                    sow_chunk = (int)(sow_local_chunk - (unsigned int)sow_batch * sow_n_max);
                } else {
                    sow_kind = 1;
                    sow_r = sow_ticket - sow_chunk_items;
                    sow_rt = sow_r >> sow_shift;
                    sow_slice = sow_r - (sow_rt << sow_shift);
                    unsigned int q_1 = (unsigned int)((float)sow_rt * sow_items_rcp);
                    if (sow_rt < q_1 * sow_items) {
                        q_1 = q_1 - 1;
                    }
                    if (sow_rt >= (q_1 + 1) * sow_items) {
                        q_1 = q_1 + 1;
                    }
                    sow_batch = (int)q_1;
                    sow_item = (int)(sow_rt - (unsigned int)sow_batch * sow_items);
                }
                sow_q_tile = sow_item / num_kv_heads;
                sow_head = sow_item - sow_q_tile * num_kv_heads;
                int sow_last = causal_seqlens_kv_global[sow_batch] + (q_len - 1) - cp_rank;
                int sow_cp_mask = (1 << cp_world_log2) - 1;
                if (sow_last >= 0) {
                    sow_len = (sow_last >> cp_world_log2) + 1;
                    sow_phase = sow_last & sow_cp_mask;
                }
                int _max_0 = ((sow_len) > (1) ? (sow_len) : (1));
                int sow_pairs = (_max_0 + 255) / 256;
                sow_n = sow_pairs + 1 - 1;
                if (sow_n > 1) {
                    sow_slot = sow_batch * (int)sow_n_max * items_per_chunk + sow_item;
                    sow_ctr = sow_batch * items_per_chunk + sow_item;
                }
                if (sow_kind == 0) {
                    if (sow_chunk < sow_n) {
                        sow_valid = 1;
                        sow_len_tok = sow_len;
                        sow_phase_tok = sow_phase;
                        int _max_1 = (((sow_len + BLOCK_N - 1) / BLOCK_N) > (1) ? ((sow_len + BLOCK_N - 1) / BLOCK_N) : (1));
                        int sow_nblk = _max_1;
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
                }
            }
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
                        for (int _sow_poll = 0; _sow_poll < 1073741824; _sow_poll++) {
                            {
                                unsigned int _atomic_old_0;
                                asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                                    : "=r"(_atomic_old_0) : "l"(&tile_counters[sow_ctr * 4]), "r"(static_cast<uint32_t>(0)) : "memory");
                                sow_arrived = _atomic_old_0;
                            }
                            if ((int)sow_arrived == sow_n) {
                                break;
                            }
                        }
                    }
                }
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
                work_token_words[sow_base + 12] = (unsigned int)sow_q_tile;
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
                        }
                    }
                }
            }
            work_stage_sched += 1;
            if (work_stage_sched == 4) { work_stage_sched = 0; _phase_work_empty ^= 1; }
            if (sow_valid != 0) {
                mbarrier_wait(work_empty_addr + (work_stage_sched) * 8, _phase_work_empty);
                if (lane_0 == 0) {
                    work_token_words[work_stage_sched * 16] = 0;
                    mbarrier_arrive(work_full_addr + (work_stage_sched) * 8);
                }
                work_stage_sched += 1;
                if (work_stage_sched == 4) { work_stage_sched = 0; _phase_work_empty ^= 1; }
            }
            unsigned int done_old = 0;
            if (lane_0 == 0) {
                uint32_t _atomic_inc_old_0;
                asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                    : "=r"(_atomic_inc_old_0) : "l"(&queue_counters[1]), "r"(static_cast<uint32_t>(num_ctas - 1)) : "memory");
                done_old = _atomic_inc_old_0;
            }
            unsigned int _shfl_0 = __shfl_sync(0xFFFFFFFF, done_old, 0);
            done_old = _shfl_0;
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
            unsigned int qr_prod_stage = 0;
            unsigned int qr_prod_phase = 1;
            unsigned int page_cons_stage = 0;
            unsigned int page_cons_phase = 0;
            unsigned int k_prod_stage = 0;
            unsigned int k_prod_phase = 1;
            unsigned int v_prod_stage = 0;
            unsigned int v_prod_phase = 1;
            unsigned int work_stage_l = 0;
            unsigned int _phase_work_full_4 = 0;
            mbarrier_wait(work_full_addr + (work_stage_l) * 8, _phase_work_full_4);
            unsigned int base_4 = work_stage_l * 16;
            unsigned int valid_5 = work_token_words[base_4];
            unsigned int kind_5 = work_token_words[base_4 + 1];
            unsigned int batch_5 = work_token_words[base_4 + 2];
            unsigned int kv_head_5 = work_token_words[base_4 + 3];
            unsigned int block_begin_4 = work_token_words[base_4 + 4];
            unsigned int block_end_4 = work_token_words[base_4 + 5];
            unsigned int seqlen_4 = work_token_words[base_4 + 6];
            unsigned int n_chunks_4 = work_token_words[base_4 + 7];
            unsigned int slot_tile_base_4 = work_token_words[base_4 + 8];
            unsigned int counter_idx_4 = work_token_words[base_4 + 9];
            unsigned int chunk_4 = work_token_words[base_4 + 10];
            unsigned int phase_4 = work_token_words[base_4 + 11];
            unsigned int q_tile_4 = work_token_words[base_4 + 12];
            mbarrier_arrive(work_empty_addr + (work_stage_l) * 8);
            work_stage_l += 1;
            if (work_stage_l == 4) { work_stage_l = 0; _phase_work_full_4 ^= 1; }
            unsigned int valid_l = valid_5;
            int kind_l = (int)kind_5;
            int batch_idx_l = (int)batch_5;
            int kv_head_idx = (int)kv_head_5;
            int pf_v_l = ((q_len <= 4) ? 1 : 0);
            int block_begin_l = (int)block_begin_4;
            int block_end_l = (int)block_end_4;
            int q_tile_l = (int)q_tile_4;
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
                        mbarrier_wait(q_raw_empty_addr + (qr_prod_stage) * 8, qr_prod_phase);
                        if (_tile_iter_l == 0) {
                        }
                        int k2x_l = 0;
                        if (_tile_iter_l == 0) {
                            if (q_len - q_tile_l * 4 > 2) {
                                if (cta_n_blocks >= 1) {
                                    k2x_l = 1;
                                }
                            }
                        }
                        if (k2x_l != 0) {
                            mbarrier_arrive_expect_tx(q_raw_full_addr + (qr_prod_stage) * 8, 64 * HEAD_DIM * 2);
                        } else {
                            mbarrier_arrive_expect_tx(q_raw_full_addr + (qr_prod_stage) * 8, 32 * HEAD_DIM * 2);
                        }
                        tma_4d_gmem2smem(smem_qbf_addr + qr_prod_stage * 16384, (&Q), 0, kv_head_idx * 16, batch_idx_l * q_len + q_tile_l * 4, 0, q_raw_full_addr + (qr_prod_stage) * 8);
                        if (k2x_l != 0) {
                            tma_4d_gmem2smem(smem_k_g0_addr + 32768, (&Q), 0, kv_head_idx * 16, batch_idx_l * q_len + q_tile_l * 4 + 2, 0, q_raw_full_addr + (qr_prod_stage) * 8);
                        }
                        qr_prod_phase ^= 1;
                        int q2_early_l = 0;
                        #pragma unroll 1
                        for (int ni = 0; ni < n_pre; ni++) {
                            int pre_stage_u = page_cons_stage + (unsigned int)ni;
                            int pre_stage = ((pre_stage_u >= 6) ? pre_stage_u - 6 : pre_stage_u);
                            int pre_phase = ((pre_stage_u >= 6) ? page_cons_phase ^ 1 : page_cons_phase);
                            int pre_pg_base = pre_stage * 8;
                            mbarrier_wait(page_offsets_full_addr + (pre_stage) * 8, pre_phase);
                            int pg_pre[2];
                            #pragma unroll
                            for (int pg_i = 0; pg_i < 2; pg_i++) {
                                pg_pre[pg_i] = smem_page_offsets[pre_pg_base + pg_i];
                            }
                            if (k2x_l != 0) {
                                if (ni == 2) {
                                    mbarrier_wait(q_raw_empty_addr + (qr_prod_stage) * 8, qr_prod_phase);
                                }
                            }
                            mbarrier_wait(k_empty_addr + (k_prod_stage) * 8, k_prod_phase);
                            mbarrier_arrive_expect_tx(k_full_addr + (k_prod_stage) * 8, 32768);
                            int kdst0 = smem_k_g0_addr + k_prod_stage * 16384;
                            int kdst0_g1 = smem_k_g1_addr + k_prod_stage * 16384;
                            #pragma unroll
                            for (int pg_i_1 = 0; pg_i_1 < 2; pg_i_1++) {
                                int kpg0 = pg_pre[pg_i_1];
                                int ktoff0 = pg_i_1 * 8192;
                                tma_5d_gmem2smem(kdst0 + ktoff0, (&K), 0, 0, 0, kv_head_idx, kpg0, k_full_addr + (k_prod_stage) * 8);
                                tma_5d_gmem2smem(kdst0_g1 + ktoff0, (&K), 0, 0, 1, kv_head_idx, kpg0, k_full_addr + (k_prod_stage) * 8);
                            }
                            k_prod_stage += 1;
                            if (k_prod_stage == 3) { k_prod_stage = 0; k_prod_phase ^= 1; }
                            if (ni < 2) {
                                mbarrier_wait(v_empty_addr + (v_prod_stage) * 8, v_prod_phase);
                                mbarrier_arrive_expect_tx(v_full_addr + (v_prod_stage) * 8, 32768);
                                int vdst0 = smem_v_g0_addr + v_prod_stage * 16384;
                                int vdst0_g1 = smem_v_g1_addr + v_prod_stage * 16384;
                                #pragma unroll
                                for (int pg_i_2 = 0; pg_i_2 < 2; pg_i_2++) {
                                    int vpg0 = pg_pre[pg_i_2];
                                    int vtoff0 = pg_i_2 * 8192;
                                    tma_5d_gmem2smem(vdst0 + vtoff0, (&V), 0, 0, 0, kv_head_idx, vpg0, v_full_addr + (v_prod_stage) * 8);
                                    tma_5d_gmem2smem(vdst0_g1 + vtoff0, (&V), 0, 0, 1, kv_head_idx, vpg0, v_full_addr + (v_prod_stage) * 8);
                                }
                                v_prod_stage += 1;
                                if (v_prod_stage == 2) { v_prod_stage = 0; v_prod_phase ^= 1; }
                            }
                        }
                        if (k2x_l != 0) {
                            if (cta_n_blocks < 3) {
                                mbarrier_wait(q_raw_empty_addr + (qr_prod_stage) * 8, qr_prod_phase);
                            }
                        }
                        int half2_late_l = 0;
                        if (k2x_l == 0) {
                            if (q_len - q_tile_l * 4 > 2) {
                                half2_late_l = 1;
                            }
                        }
                        if (half2_late_l != 0) {
                            mbarrier_wait(q_raw_empty_addr + (qr_prod_stage) * 8, qr_prod_phase);
                            mbarrier_arrive_expect_tx(q_raw_full_addr + (qr_prod_stage) * 8, 32 * HEAD_DIM * 2);
                            tma_4d_gmem2smem(smem_qbf_addr + qr_prod_stage * 16384, (&Q), 0, kv_head_idx * 16, batch_idx_l * q_len + q_tile_l * 4 + 2, 0, q_raw_full_addr + (qr_prod_stage) * 8);
                            qr_prod_phase ^= 1;
                        }
                        #pragma unroll 1
                        for (int ni_1 = 0; ni_1 < cta_n_blocks; ni_1++) {
                            int nk = ni_1 + 3;
                            if (nk < cta_n_blocks) {
                                int k_page_u = page_cons_stage + 3;
                                int k_page = ((k_page_u >= 6) ? k_page_u - 6 : k_page_u);
                                int k_page_phase = ((k_page_u >= 6) ? page_cons_phase ^ 1 : page_cons_phase);
                                int kpg_base = k_page * 8;
                                mbarrier_wait(page_offsets_full_addr + (k_page) * 8, k_page_phase);
                                int pg_nk[2];
                                #pragma unroll
                                for (int pg_i_3 = 0; pg_i_3 < 2; pg_i_3++) {
                                    pg_nk[pg_i_3] = smem_page_offsets[kpg_base + pg_i_3];
                                }
                                mbarrier_wait(k_empty_addr + (k_prod_stage) * 8, k_prod_phase);
                                mbarrier_arrive_expect_tx(k_full_addr + (k_prod_stage) * 8, 32768);
                                int kdst = smem_k_g0_addr + k_prod_stage * 16384;
                                int kdst_g1 = smem_k_g1_addr + k_prod_stage * 16384;
                                #pragma unroll
                                for (int pg_i_4 = 0; pg_i_4 < 2; pg_i_4++) {
                                    int npg0 = pg_nk[pg_i_4];
                                    int ntoff = pg_i_4 * 8192;
                                    tma_5d_gmem2smem(kdst + ntoff, (&K), 0, 0, 0, kv_head_idx, npg0, k_full_addr + (k_prod_stage) * 8);
                                    tma_5d_gmem2smem(kdst_g1 + ntoff, (&K), 0, 0, 1, kv_head_idx, npg0, k_full_addr + (k_prod_stage) * 8);
                                }
                                k_prod_stage += 1;
                                if (k_prod_stage == 3) { k_prod_stage = 0; k_prod_phase ^= 1; }
                            }
                            int nv = ni_1 + 2;
                            if (nv < cta_n_blocks) {
                                int v_page_u = page_cons_stage + 2;
                                int v_page = ((v_page_u >= 6) ? v_page_u - 6 : v_page_u);
                                int v_page_phase = ((v_page_u >= 6) ? page_cons_phase ^ 1 : page_cons_phase);
                                int vpg_base = v_page * 8;
                                mbarrier_wait(page_offsets_full_addr + (v_page) * 8, v_page_phase);
                                int pg_nv[2];
                                #pragma unroll
                                for (int pg_i_5 = 0; pg_i_5 < 2; pg_i_5++) {
                                    pg_nv[pg_i_5] = smem_page_offsets[vpg_base + pg_i_5];
                                }
                                mbarrier_wait(v_empty_addr + (v_prod_stage) * 8, v_prod_phase);
                                mbarrier_arrive_expect_tx(v_full_addr + (v_prod_stage) * 8, 32768);
                                int vdst = smem_v_g0_addr + v_prod_stage * 16384;
                                int vdst_g1 = smem_v_g1_addr + v_prod_stage * 16384;
                                #pragma unroll
                                for (int pg_i_6 = 0; pg_i_6 < 2; pg_i_6++) {
                                    int vpg1 = pg_nv[pg_i_6];
                                    int vtoff = pg_i_6 * 8192;
                                    tma_5d_gmem2smem(vdst + vtoff, (&V), 0, 0, 0, kv_head_idx, vpg1, v_full_addr + (v_prod_stage) * 8);
                                    tma_5d_gmem2smem(vdst_g1 + vtoff, (&V), 0, 0, 1, kv_head_idx, vpg1, v_full_addr + (v_prod_stage) * 8);
                                }
                                v_prod_stage += 1;
                                if (v_prod_stage == 2) { v_prod_stage = 0; v_prod_phase ^= 1; }
                            }
                            mbarrier_arrive(page_offsets_empty_addr + (page_cons_stage) * 8);
                            page_cons_stage += 1;
                            if (page_cons_stage == 6) { page_cons_stage = 0; page_cons_phase ^= 1; }
                            if (ni_1 == gate_block) {
                                mbarrier_arrive(claim_gate_addr);
                            }
                        }
                    }
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
                unsigned int q_tile_13_4 = work_token_words[base_0_4 + 12];
                mbarrier_arrive(work_empty_addr + (work_stage_l) * 8);
                work_stage_l += 1;
                if (work_stage_l == 4) { work_stage_l = 0; _phase_work_full_4 ^= 1; }
                valid_l = valid_1_4;
                kind_l = (int)kind_2_4;
                batch_idx_l = (int)batch_3_4;
                kv_head_idx = (int)kv_head_4_4;
                block_begin_l = (int)block_begin_5_4;
                block_end_l = (int)block_end_6_4;
                q_tile_l = (int)q_tile_13_4;
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
