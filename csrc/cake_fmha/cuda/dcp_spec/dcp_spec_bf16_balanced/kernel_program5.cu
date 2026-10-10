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
#define TMEM_TMEM_ST_OFFSET 0
#define TMEM_TMEM_OT_OFFSET 128
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

    // Mbarrier init (20 pipeline groups, 0 ordered-sequence groups, 51 barriers)
    // Mbarriers at smem_raw[0..408)

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
            // p_full_b: 2 barriers, init_count=256
            mbarrier_init(smem + 160, 256);
            mbarrier_init(smem + 168, 256);
            // pa_done: 2 barriers, init_count=1
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            // o_empty: 1 barriers, init_count=128
            mbarrier_init(smem + 192, 128);
            // --- pipeline 'stats_pipe' ---
            // stats_full: 2 barriers, init_count=128
            mbarrier_init(smem + 200, 128);
            mbarrier_init(smem + 208, 128);
            // stats_empty: 2 barriers, init_count=4
            mbarrier_init(smem + 216, 4);
            mbarrier_init(smem + 224, 4);
            // tmem_dealloc: 1 barriers, init_count=128
            mbarrier_init(smem + 232, 128);
            // --- pipeline 'page_pipe' ---
            // page_offsets_full: 6 barriers, init_count=1
            mbarrier_init(smem + 240, 1);
            mbarrier_init(smem + 248, 1);
            mbarrier_init(smem + 256, 1);
            mbarrier_init(smem + 264, 1);
            mbarrier_init(smem + 272, 1);
            mbarrier_init(smem + 280, 1);
            // page_offsets_empty: 6 barriers, init_count=1
            mbarrier_init(smem + 288, 1);
            mbarrier_init(smem + 296, 1);
            mbarrier_init(smem + 304, 1);
            mbarrier_init(smem + 312, 1);
            mbarrier_init(smem + 320, 1);
            mbarrier_init(smem + 328, 1);
            // --- pipeline 'work_pipe' ---
            // work_full: 4 barriers, init_count=1
            mbarrier_init(smem + 336, 1);
            mbarrier_init(smem + 344, 1);
            mbarrier_init(smem + 352, 1);
            mbarrier_init(smem + 360, 1);
            // work_empty: 4 barriers, init_count=352
            mbarrier_init(smem + 368, 352);
            mbarrier_init(smem + 376, 352);
            mbarrier_init(smem + 384, 352);
            mbarrier_init(smem + 392, 352);
            // claim_gate: 1 barriers, init_count=1
            mbarrier_init(smem + 400, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 384 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 408);
    if (warp == 0) {
        int _tmem_hold = smem + 408;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_sT = taddr;
    const int tmem_tmem_oT = taddr + 128;
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
            int _min_1 = ((row_j) < (q_len - 1) ? (row_j) : (q_len - 1));
            int vis_j = _min_1;
            unsigned int sm_stage = 0;
            unsigned int sm_phase = 0;
            unsigned int xm_slot_s = 0;
            unsigned int st_stage_s = 0;
            unsigned int st_phase_s = 1;
            float _rcp_3 = approx_rcp(softmax_scale_log2);
            float thr_raw = 8.0f * _rcp_3;
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
            unsigned int pv_base_s = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_s = 0; _tile_iter_s < max_items; _tile_iter_s++) {
                if (valid_s == 0) {
                    break;
                }
                if (kind_s == 0) {
                    int cnt_s = block_end_s - block_begin_s;
                    int k_w = s_warp * 32 + lane;
                    int rows_hi = ((N_ROWS > 32) ? 1 : 0);
                    int sT_base = taddr + (unsigned int)(s_warp * 32 << 16);
                    int oT_base_w = taddr + 128 + (unsigned int)(s_warp * 32 << 16);
                    int vis_a[16];
                    int vis_b[16];
                    #pragma unroll
                    for (int j = 0; j < 16; j++) {
                        int row_j_v = j / 8;
                        int _min_2 = ((row_j_v) < (q_len - 1) ? (row_j_v) : (q_len - 1));
                        int vis_j_v = _min_2;
                        int back_v = q_len - 1 - vis_j_v - phase_s;
                        int vis_v = seqlen_s;
                        if (back_v > 0) {
                            vis_v = seqlen_s - (back_v + (1 << cp_world_log2) - 1 >> cp_world_log2);
                        }
                        vis_a[j] = vis_v;
                        int row_j_v_0 = (32 + j) / 8;
                        int _min_3 = ((row_j_v_0) < (q_len - 1) ? (row_j_v_0) : (q_len - 1));
                        int vis_j_v_1 = _min_3;
                        int back_v_2 = q_len - 1 - vis_j_v_1 - phase_s;
                        int vis_v_3 = seqlen_s;
                        if (back_v_2 > 0) {
                            vis_v_3 = seqlen_s - (back_v_2 + (1 << cp_world_log2) - 1 >> cp_world_log2);
                        }
                        vis_b[j] = vis_v_3;
                    }
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
                    #pragma unroll 1
                    for (int n = 0; n < cnt_s; n++) {
                        if (sm_tid == 0) {
                        }
                        {
                            mbarrier_wait(s_full_addr + (sm_stage) * 8, sm_phase);
                        }
                        if (sm_tid == 0) {
                        }
                        int my_block_w = block_begin_s + cnt_s - 1 - n;
                        int kpos = my_block_w * BLOCK_N + k_w;
                        int par_w = n & 1;
                        float sv_a[16];
                        float sv_b[16];
                        tmem_ld_x16(&sv_a[0], (unsigned int)sT_base + sm_stage * 64);
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        if (rows_hi != 0) {
                            tmem_ld_x16(&sv_b[0], (unsigned int)sT_base + sm_stage * 64 + 32);
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                        }
                        int exc_w = 0;
                        #pragma unroll
                        for (int j_1 = 0; j_1 < 16; j_1++) {
                            if (kpos >= vis_a[j_1]) {
                                sv_a[j_1] = -CAKE_FMHA_INF;
                            }
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
                                    }
                                }
                            }
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
                                float _max_5 = max_noftz(keep8, _shfl_xor_0);
                                w8[i] = _max_5;
                            }
                            float x4[4];
                            #pragma unroll
                            for (int i_1 = 0; i_1 < 4; i_1++) {
                                float send4 = ((b3 != 0) ? w8[i_1] : w8[i_1 + 4]);
                                float keep4 = ((b3 != 0) ? w8[i_1 + 4] : w8[i_1]);
                                float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, send4, 8);
                                float _max_6 = max_noftz(keep4, _shfl_xor_1);
                                x4[i_1] = _max_6;
                            }
                            float y2[2];
                            #pragma unroll
                            for (int i_2 = 0; i_2 < 2; i_2++) {
                                float send2 = ((b2 != 0) ? x4[i_2] : x4[i_2 + 2]);
                                float keep2 = ((b2 != 0) ? x4[i_2 + 2] : x4[i_2]);
                                float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, send2, 4);
                                float _max_7 = max_noftz(keep2, _shfl_xor_2);
                                y2[i_2] = _max_7;
                            }
                            float send1 = ((b1 != 0) ? y2[0] : y2[1]);
                            float keep1 = ((b1 != 0) ? y2[1] : y2[0]);
                            float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, send1, 2);
                            float _max_8 = max_noftz(keep1, _shfl_xor_3);
                            float z = _max_8;
                            float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, z, 1);
                            float _max_9 = max_noftz(z, _shfl_xor_4);
                            z = _max_9;
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
                                    float _max_10 = max_noftz(keep8_1, _shfl_xor_5);
                                    w8_4[i_3] = _max_10;
                                }
                                float x4_5[4];
                                #pragma unroll
                                for (int i_4 = 0; i_4 < 4; i_4++) {
                                    float send4_1 = ((b3_1 != 0) ? w8_4[i_4] : w8_4[i_4 + 4]);
                                    float keep4_1 = ((b3_1 != 0) ? w8_4[i_4 + 4] : w8_4[i_4]);
                                    float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, send4_1, 8);
                                    float _max_11 = max_noftz(keep4_1, _shfl_xor_6);
                                    x4_5[i_4] = _max_11;
                                }
                                float y2_6[2];
                                #pragma unroll
                                for (int i_5 = 0; i_5 < 2; i_5++) {
                                    float send2_1 = ((b2_2 != 0) ? x4_5[i_5] : x4_5[i_5 + 2]);
                                    float keep2_1 = ((b2_2 != 0) ? x4_5[i_5 + 2] : x4_5[i_5]);
                                    float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, send2_1, 4);
                                    float _max_12 = max_noftz(keep2_1, _shfl_xor_7);
                                    y2_6[i_5] = _max_12;
                                }
                                float send1_7 = ((b1_3 != 0) ? y2_6[0] : y2_6[1]);
                                float keep1_8 = ((b1_3 != 0) ? y2_6[1] : y2_6[0]);
                                float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, send1_7, 2);
                                float _max_13 = max_noftz(keep1_8, _shfl_xor_8);
                                float z_9 = _max_13;
                                float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, z_9, 1);
                                float _max_14 = max_noftz(z_9, _shfl_xor_9);
                                z_9 = _max_14;
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
                                    float _max_15 = max_noftz(p64s[4 * j_6], p64s[4 * j_6 + 1]);
                                    float _max_16 = max_noftz(p64s[4 * j_6 + 2], p64s[4 * j_6 + 3]);
                                    float _max_17 = max_noftz(_max_15, _max_16);
                                    float bm_sj = _max_17;
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
                                        float _max_18 = max_noftz(p64s_0[4 * j_8], p64s_0[4 * j_8 + 1]);
                                        float _max_19 = max_noftz(p64s_0[4 * j_8 + 2], p64s_0[4 * j_8 + 3]);
                                        float _max_20 = max_noftz(_max_18, _max_19);
                                        float bm_sj_1 = _max_20;
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
                                    float _max_21 = max_noftz(p64[4 * j_10], p64[4 * j_10 + 1]);
                                    float _max_22 = max_noftz(p64[4 * j_10 + 2], p64[4 * j_10 + 3]);
                                    float _max_23 = max_noftz(_max_21, _max_22);
                                    float bm_j = _max_23;
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
                                        float _max_24 = max_noftz(p64_0[4 * j_12], p64_0[4 * j_12 + 1]);
                                        float _max_25 = max_noftz(p64_0[4 * j_12 + 2], p64_0[4 * j_12 + 3]);
                                        float _max_26 = max_noftz(_max_24, _max_25);
                                        float bm_j_1 = _max_26;
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
                            }
                        }
                        if (sm_tid == 0) {
                        }
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
                                for (int q_1 = 0; q_1 < 4; q_1++) {
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&acc16[4 * q_1])), "=r"(*reinterpret_cast<uint32_t*>(&acc16[(4 * q_1) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&acc16[(4 * q_1) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&acc16[(4 * q_1) + 3]))
                                        : "r"(swap_acc_addr + (unsigned int)(4 * q_1 * 4)));
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
                                    for (int q_2 = 0; q_2 < 4; q_2++) {
                                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                            : "=r"(*reinterpret_cast<uint32_t*>(&acc16_0[4 * q_2])), "=r"(*reinterpret_cast<uint32_t*>(&acc16_0[(4 * q_2) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&acc16_0[(4 * q_2) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&acc16_0[(4 * q_2) + 3]))
                                            : "r"(swap_acc_addr + (unsigned int)((32 + 4 * q_2) * 4)));
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
                                }
                            }
                        }
                        int pb_w = 0;
                        if (rows_hi == 0) {
                            pb_w = (int)(k_blk_o & 1);
                        }
                        if (pb_w == 0) {
                            unsigned int words8[8];
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sv_a[_lp*2 + 0], sv_a[_lp*2+1 + 0]));
                                words8[_lp] = *(uint32_t*)&_bf2;
                            }
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
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        __syncwarp();
                        if (lane == 0) {
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;"
                                :: "r"(p_full_addr + (sm_stage) * 8), "r"((uint32_t)(32)) : "memory");
                        }
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
                        sm_stage += 1;
                        if (sm_stage == 2) { sm_stage = 0; sm_phase ^= 1; }
                        if (sm_tid == 0) {
                        }
                    }
                    pv_base_s = pv_base_s + (unsigned int)cnt_s;
                    if (sm_tid == 0) {
                    }
                    if (sm_tid == 0) {
                    }
                    mbarrier_wait(stats_empty_addr + (st_stage_s) * 8, st_phase_s);
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
                            float _rcp_4 = approx_rcp(sum_pa);
                            swap_inv[(int)st_stage_s * 64 + lane] = ((sum_pa > 0.0f) ? _rcp_4 : 0.0f);
                            if (rows_hi_p != 0) {
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&q4[0])), "=r"(*reinterpret_cast<uint32_t*>(&q4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q4[(0) + 3]))
                                    : "r"(swap_part_addr + (unsigned int)((16 + lane) * 16)));
                                float sum_pb = q4[0] + q4[1] + (q4[2] + q4[3]);
                                smem_sum[(int)st_stage_s * 64 + 32 + lane] = sum_pb;
                                float _rcp_5 = approx_rcp(sum_pb);
                                swap_inv[(int)st_stage_s * 64 + 32 + lane] = ((sum_pb > 0.0f) ? _rcp_5 : 0.0f);
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
                        }
                    }
                    __syncwarp();
                    if (sm_tid == 0) {
                    }
                    if (lane == 0) {
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;"
                            :: "r"(stats_full_addr + (st_stage_s) * 8), "r"((uint32_t)(32)) : "memory");
                    }
                    if (sm_tid == 0) {
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
                                    float _max_31 = max_noftz(m_f, m_k);
                                    float m_new = _max_31;
                                    float _exp2_6 = approx_exp2((m_f - m_new) * softmax_scale_log2);
                                    float a_k = _exp2_6;
                                    float _exp2_7 = approx_exp2((m_k - m_new) * softmax_scale_log2);
                                    float b_k = _exp2_7;
                                    float _fma_6 = __fmaf_rn(l_k, b_k, l_f * a_k);
                                    l_f = _fma_6;
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
                                        float _fma_7 = __fmaf_rn(_vec_load_0[k], b_k, acc_f[k] * a_k);
                                        acc_f[k] = _fma_7;
                                    }
                                    m_f = m_new;
                                }
                                float _rcp_8 = approx_rcp(l_f);
                                float inv_f = ((l_f > 0.0f) ? _rcp_8 : 0.0f);
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
            int _min_8 = ((row_j_c) < (q_len - 1) ? (row_j_c) : (q_len - 1));
            int vis_j_c = _min_8;
            float _rcp_9 = approx_rcp(softmax_scale_log2);
            float thr_raw_c = 8.0f * _rcp_9;
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
                    int k_w_1 = warp_in_wg_c * 32 + lane;
                    int rows_hi_1 = ((N_ROWS > 32) ? 1 : 0);
                    int sT_base_1 = taddr + (unsigned int)(warp_in_wg_c * 32 << 16);
                    int oT_base_w_1 = taddr + 128 + (unsigned int)(warp_in_wg_c * 32 << 16);
                    int vis_a_1[16];
                    int vis_b_1[16];
                    #pragma unroll
                    for (int j_20 = 0; j_20 < 16; j_20++) {
                        int row_j_v_1 = (16 + j_20) / 8;
                        int _min_9 = ((row_j_v_1) < (q_len - 1) ? (row_j_v_1) : (q_len - 1));
                        int vis_j_v_2 = _min_9;
                        int back_v_1 = q_len - 1 - vis_j_v_2 - phase_c;
                        int vis_v_1 = seqlen_c;
                        if (back_v_1 > 0) {
                            vis_v_1 = seqlen_c - (back_v_1 + (1 << cp_world_log2) - 1 >> cp_world_log2);
                        }
                        vis_a_1[j_20] = vis_v_1;
                        int row_j_v_0_1 = (48 + j_20) / 8;
                        int _min_10 = ((row_j_v_0_1) < (q_len - 1) ? (row_j_v_0_1) : (q_len - 1));
                        int vis_j_v_1_1 = _min_10;
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
                    #pragma unroll 1
                    for (int n_1 = 0; n_1 < cnt_c; n_1++) {
                        if (wg_tid_c == 0) {
                        }
                        {
                            mbarrier_wait(s_full_addr + (sm_stage_c) * 8, sm_phase_c);
                        }
                        if (wg_tid_c == 0) {
                        }
                        int my_block_w_1 = block_begin_c + cnt_c - 1 - n_1;
                        int kpos_1 = my_block_w_1 * BLOCK_N + k_w_1;
                        int par_w_1 = n_1 & 1;
                        float sv_a_1[16];
                        float sv_b_1[16];
                        tmem_ld_x16(&sv_a_1[0], (unsigned int)sT_base_1 + sm_stage_c * 64 + 16);
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        if (rows_hi_1 != 0) {
                            tmem_ld_x16(&sv_b_1[0], (unsigned int)sT_base_1 + sm_stage_c * 64 + 32 + 16);
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                        }
                        int exc_w_1 = 0;
                        #pragma unroll
                        for (int j_21 = 0; j_21 < 16; j_21++) {
                            if (kpos_1 >= vis_a_1[j_21]) {
                                sv_a_1[j_21] = -CAKE_FMHA_INF;
                            }
                            if (sv_a_1[j_21] > ref_a_1[j_21] + thr_raw_c) {
                                exc_w_1 = 1;
                            }
                        }
                        if (rows_hi_1 != 0) {
                            #pragma unroll
                            for (int j_22 = 0; j_22 < 16; j_22++) {
                                if (kpos_1 >= vis_b_1[j_22]) {
                                    sv_b_1[j_22] = -CAKE_FMHA_INF;
                                }
                                if (sv_b_1[j_22] > ref_b_1[j_22] + thr_raw_c) {
                                    exc_w_1 = 1;
                                }
                            }
                        }
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
                        if (wg_tid_c == 0) {
                        }
                        if (fl_w_1 != 0) {
                            int b4_2 = lane >> 4 & 1;
                            int b3_3 = lane >> 3 & 1;
                            int b2_3 = lane >> 2 & 1;
                            int b1_2 = lane >> 1 & 1;
                            float w8_2[8];
                            #pragma unroll
                            for (int i_12 = 0; i_12 < 8; i_12++) {
                                float send8_4 = ((b4_2 != 0) ? sv_a_1[i_12] : sv_a_1[i_12 + 8]);
                                float keep8_4 = ((b4_2 != 0) ? sv_a_1[i_12 + 8] : sv_a_1[i_12]);
                                float _shfl_xor_20 = __shfl_xor_sync(0xFFFFFFFF, send8_4, 16);
                                float _max_32 = max_noftz(keep8_4, _shfl_xor_20);
                                w8_2[i_12] = _max_32;
                            }
                            float x4_2[4];
                            #pragma unroll
                            for (int i_13 = 0; i_13 < 4; i_13++) {
                                float send4_4 = ((b3_3 != 0) ? w8_2[i_13] : w8_2[i_13 + 4]);
                                float keep4_4 = ((b3_3 != 0) ? w8_2[i_13 + 4] : w8_2[i_13]);
                                float _shfl_xor_21 = __shfl_xor_sync(0xFFFFFFFF, send4_4, 8);
                                float _max_33 = max_noftz(keep4_4, _shfl_xor_21);
                                x4_2[i_13] = _max_33;
                            }
                            float y2_2[2];
                            #pragma unroll
                            for (int i_14 = 0; i_14 < 2; i_14++) {
                                float send2_4 = ((b2_3 != 0) ? x4_2[i_14] : x4_2[i_14 + 2]);
                                float keep2_4 = ((b2_3 != 0) ? x4_2[i_14 + 2] : x4_2[i_14]);
                                float _shfl_xor_22 = __shfl_xor_sync(0xFFFFFFFF, send2_4, 4);
                                float _max_34 = max_noftz(keep2_4, _shfl_xor_22);
                                y2_2[i_14] = _max_34;
                            }
                            float send1_2 = ((b1_2 != 0) ? y2_2[0] : y2_2[1]);
                            float keep1_2 = ((b1_2 != 0) ? y2_2[1] : y2_2[0]);
                            float _shfl_xor_23 = __shfl_xor_sync(0xFFFFFFFF, send1_2, 2);
                            float _max_35 = max_noftz(keep1_2, _shfl_xor_23);
                            float z_2 = _max_35;
                            float _shfl_xor_24 = __shfl_xor_sync(0xFFFFFFFF, z_2, 1);
                            float _max_36 = max_noftz(z_2, _shfl_xor_24);
                            z_2 = _max_36;
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
                                    float _max_37 = max_noftz(keep8_5, _shfl_xor_25);
                                    w8_4_2[i_15] = _max_37;
                                }
                                float x4_5_2[4];
                                #pragma unroll
                                for (int i_16 = 0; i_16 < 4; i_16++) {
                                    float send4_5 = ((b3_1_2 != 0) ? w8_4_2[i_16] : w8_4_2[i_16 + 4]);
                                    float keep4_5 = ((b3_1_2 != 0) ? w8_4_2[i_16 + 4] : w8_4_2[i_16]);
                                    float _shfl_xor_26 = __shfl_xor_sync(0xFFFFFFFF, send4_5, 8);
                                    float _max_38 = max_noftz(keep4_5, _shfl_xor_26);
                                    x4_5_2[i_16] = _max_38;
                                }
                                float y2_6_2[2];
                                #pragma unroll
                                for (int i_17 = 0; i_17 < 2; i_17++) {
                                    float send2_5 = ((b2_2_2 != 0) ? x4_5_2[i_17] : x4_5_2[i_17 + 2]);
                                    float keep2_5 = ((b2_2_2 != 0) ? x4_5_2[i_17 + 2] : x4_5_2[i_17]);
                                    float _shfl_xor_27 = __shfl_xor_sync(0xFFFFFFFF, send2_5, 4);
                                    float _max_39 = max_noftz(keep2_5, _shfl_xor_27);
                                    y2_6_2[i_17] = _max_39;
                                }
                                float send1_7_2 = ((b1_3_2 != 0) ? y2_6_2[0] : y2_6_2[1]);
                                float keep1_8_2 = ((b1_3_2 != 0) ? y2_6_2[1] : y2_6_2[0]);
                                float _shfl_xor_28 = __shfl_xor_sync(0xFFFFFFFF, send1_7_2, 2);
                                float _max_40 = max_noftz(keep1_8_2, _shfl_xor_28);
                                float z_9_2 = _max_40;
                                float _shfl_xor_29 = __shfl_xor_sync(0xFFFFFFFF, z_9_2, 1);
                                float _max_41 = max_noftz(z_9_2, _shfl_xor_29);
                                z_9_2 = _max_41;
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
                                    float _max_42 = max_noftz(p64s_1[4 * j_26], p64s_1[4 * j_26 + 1]);
                                    float _max_43 = max_noftz(p64s_1[4 * j_26 + 2], p64s_1[4 * j_26 + 3]);
                                    float _max_44 = max_noftz(_max_42, _max_43);
                                    float bm_sj_2 = _max_44;
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
                                        float _max_45 = max_noftz(p64s_0_1[4 * j_28], p64s_0_1[4 * j_28 + 1]);
                                        float _max_46 = max_noftz(p64s_0_1[4 * j_28 + 2], p64s_0_1[4 * j_28 + 3]);
                                        float _max_47 = max_noftz(_max_45, _max_46);
                                        float bm_sj_3 = _max_47;
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
                                    float _max_48 = max_noftz(p64_1[4 * j_30], p64_1[4 * j_30 + 1]);
                                    float _max_49 = max_noftz(p64_1[4 * j_30 + 2], p64_1[4 * j_30 + 3]);
                                    float _max_50 = max_noftz(_max_48, _max_49);
                                    float bm_j_2 = _max_50;
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
                                        float _max_51 = max_noftz(p64_0_1[4 * j_32], p64_0_1[4 * j_32 + 1]);
                                        float _max_52 = max_noftz(p64_0_1[4 * j_32 + 2], p64_0_1[4 * j_32 + 3]);
                                        float _max_53 = max_noftz(_max_51, _max_52);
                                        float bm_j_3 = _max_53;
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
                        }
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
                                }
                                asm volatile("tcgen05.fence::after_thread_sync;");
                                float acc16_1[16];
                                #pragma unroll
                                for (int q_3 = 0; q_3 < 4; q_3++) {
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&acc16_1[4 * q_3])), "=r"(*reinterpret_cast<uint32_t*>(&acc16_1[(4 * q_3) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&acc16_1[(4 * q_3) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&acc16_1[(4 * q_3) + 3]))
                                        : "r"(swap_acc_addr + (unsigned int)((16 + 4 * q_3) * 4)));
                                }
                                float _tmem_load_2[16];
                                tmem_ld_x16(&_tmem_load_2[0], oT_base_w_1 + 16);
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                #pragma unroll
                                for (int j_37 = 0; j_37 < 16; j_37++) {
                                    _tmem_load_2[j_37] = _tmem_load_2[j_37] * acc16_1[j_37];
                                }
                                tmem_st_x16_f32(oT_base_w_1 + 16, _tmem_load_2);
                                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                                if (rows_hi_1 != 0) {
                                    float acc16_0_1[16];
                                    #pragma unroll
                                    for (int q_4 = 0; q_4 < 4; q_4++) {
                                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                            : "=r"(*reinterpret_cast<uint32_t*>(&acc16_0_1[4 * q_4])), "=r"(*reinterpret_cast<uint32_t*>(&acc16_0_1[(4 * q_4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&acc16_0_1[(4 * q_4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&acc16_0_1[(4 * q_4) + 3]))
                                            : "r"(swap_acc_addr + (unsigned int)((48 + 4 * q_4) * 4)));
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
                            }
                        }
                        int pb_w_1 = 0;
                        if (rows_hi_1 == 0) {
                            pb_w_1 = (int)(k_blk_o_1 & 1);
                        }
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
                        __syncwarp();
                        if (lane == 0) {
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;"
                                :: "r"(p_full_addr + (sm_stage_c) * 8), "r"((uint32_t)(32)) : "memory");
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
                            float _rcp_10 = approx_rcp(sum_pa_1);
                            swap_inv[st_off_pub + 16 + lane] = ((sum_pa_1 > 0.0f) ? _rcp_10 : 0.0f);
                            if (rows_hi_p_1 != 0) {
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&q4_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&q4_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q4_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q4_1[(0) + 3]))
                                    : "r"(swap_part_addr + (unsigned int)((48 + lane) * 16)));
                                float sum_pb_1 = q4_1[0] + q4_1[1] + (q4_1[2] + q4_1[3]);
                                smem_sum[st_off_pub + 32 + 16 + lane] = sum_pb_1;
                                float _rcp_11 = approx_rcp(sum_pb_1);
                                swap_inv[st_off_pub + 32 + 16 + lane] = ((sum_pb_1 > 0.0f) ? _rcp_11 : 0.0f);
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
                    unsigned int k_e = pv_base + (unsigned int)cnt_c - 1;
                    int o_st_e = (int)(k_e & 1);
                    int o_ph_e = (int)(k_e >> 1 & 1);
                    mbarrier_wait(o_ready_addr + (o_st_e) * 8, o_ph_e);
                    pv_base = pv_base + (unsigned int)cnt_c;
                    if (wg_tid_c == 0) {
                    }
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int oT_base_c = taddr + 128 + (unsigned int)corr_row;
                    float _tmem_load_4[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_4[0]), "=f"(_tmem_load_4[1]), "=f"(_tmem_load_4[2]), "=f"(_tmem_load_4[3]), "=f"(_tmem_load_4[4]), "=f"(_tmem_load_4[5]), "=f"(_tmem_load_4[6]), "=f"(_tmem_load_4[7]), "=f"(_tmem_load_4[8]), "=f"(_tmem_load_4[9]), "=f"(_tmem_load_4[10]), "=f"(_tmem_load_4[11]), "=f"(_tmem_load_4[12]), "=f"(_tmem_load_4[13]), "=f"(_tmem_load_4[14]), "=f"(_tmem_load_4[15]), "=f"(_tmem_load_4[16]), "=f"(_tmem_load_4[17]), "=f"(_tmem_load_4[18]), "=f"(_tmem_load_4[19]), "=f"(_tmem_load_4[20]), "=f"(_tmem_load_4[21]), "=f"(_tmem_load_4[22]), "=f"(_tmem_load_4[23]), "=f"(_tmem_load_4[24]), "=f"(_tmem_load_4[25]), "=f"(_tmem_load_4[26]), "=f"(_tmem_load_4[27]), "=f"(_tmem_load_4[28]), "=f"(_tmem_load_4[29]), "=f"(_tmem_load_4[30]), "=f"(_tmem_load_4[31])
                        : "r"(oT_base_c));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
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
                    __syncwarp();
                    if (lane == 0) {
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;"
                            :: "r"(o_empty_addr), "r"((uint32_t)(32)) : "memory");
                    }
                    if (wg_tid_c == 0) {
                    }
                    mbarrier_wait(stats_full_addr + (st_stage_c) * 8, st_phase_c);
                    if (wg_tid_c == 0) {
                    }
                    int st_off = (int)st_stage_c * 64;
                    asm volatile("barrier.sync 12, 128;" ::: "memory");
                    if (wg_tid_c == 0) {
                    }
                    int publish_split = ((n_chunks_c > 1) ? 1 : 0);
                    if (publish_split == 0) {
                        int d_c = wg_tid_c;
                        float inv32[32];
                        #pragma unroll
                        for (int q_5 = 0; q_5 < 8; q_5++) {
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&inv32[4 * q_5])), "=r"(*reinterpret_cast<uint32_t*>(&inv32[(4 * q_5) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inv32[(4 * q_5) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inv32[(4 * q_5) + 3]))
                                : "r"(swap_inv_addr + (unsigned int)((st_off + 4 * q_5) * 4)));
                        }
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
                            }
                        }
                        if (rows_hi_c != 0) {
                            float inv32_0[32];
                            #pragma unroll
                            for (int q_6 = 0; q_6 < 8; q_6++) {
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&inv32_0[4 * q_6])), "=r"(*reinterpret_cast<uint32_t*>(&inv32_0[(4 * q_6) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&inv32_0[(4 * q_6) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&inv32_0[(4 * q_6) + 3]))
                                    : "r"(swap_inv_addr + (unsigned int)((st_off + 32 + 4 * q_6) * 4)));
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
                                }
                            }
                        }
                        int row_l = wg_tid_c;
                        int n_rows_c = 32 + 32 * rows_hi_c;
                        if (row_l < n_rows_c) {
                            if (row_l < live_rows) {
                                float sum_l = smem_sum[st_off + row_l];
                                float lse_l = -CAKE_FMHA_INF;
                                if (sum_l > 0.0f) {
                                    float _log2_3;
                                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_3) : "f"(sum_l));
                                    lse_l = smem_max[st_off + row_l] * softmax_scale_log2 + _log2_3;
                                }
                                *(reinterpret_cast<float*>(LSE_ptr + ((batch_c * q_len + row_l / 8) * num_q_heads + kv_head_c * 8 + row_l % 8)) + (0)) = lse_l;
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
                            for (int off = 0; off < 64; off += 4) {
                                {
                                    float4 _v4 = make_float4(o_epi[off + 0], o_epi[off + 1], o_epi[off + 2], o_epi[off + 3]);
                                    *reinterpret_cast<float4*>((partial_o + (p_base + off)) + 0) = _v4;
                                }
                            }
                        }
                        if (wg_tid_c < N_ROWS) {
                            *(reinterpret_cast<float*>(partial_stats + (my_slot * 128 + wg_tid_c)) + (0)) = smem_max[st_off + wg_tid_c];
                            *(reinterpret_cast<float*>(partial_stats + (my_slot * 128 + 64 + wg_tid_c)) + (0)) = smem_sum[st_off + wg_tid_c];
                        }
                        asm volatile("barrier.sync 12, 128;" ::: "memory");
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
                        asm volatile("barrier.sync 12, 128;" ::: "memory");
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
                                float _max_54 = max_noftz(m_s, m_o);
                                float m_row_i = _max_54;
                                float _exp2_10 = approx_exp2((m_s - m_row_i) * softmax_scale_log2);
                                float w_s = _exp2_10;
                                float _exp2_11 = approx_exp2((m_o - m_row_i) * softmax_scale_log2);
                                float w_o = _exp2_11;
                                float _fma_10 = __fmaf_rn(w_s, l_s, w_o * l_o);
                                float den_i = _fma_10;
                                float _rcp_12 = approx_rcp(den_i);
                                float inv_i = ((den_i > 0.0f) ? _rcp_12 : 0.0f);
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
                                            float _fma_14 = __fmaf_rn(o_epi[c0 + c], w_s_c, _vec_load_1[c] * w_o_c);
                                            o_epi[c0 + c] = _fma_14;
                                        }
                                    }
                                }
                                int j_c = my_row_c / 8;
                                int h_c = my_row_c % 8;
                                int q_head_c = kv_head_c * 8 + h_c;
                                int o_idx = ((batch_c * q_len + j_c) * num_q_heads + q_head_c) * HEAD_DIM + half_c * 64;
                                #pragma unroll
                                for (int off_1 = 0; off_1 < 64; off_1 += 8) {
                                    {
                                        __nv_bfloat162 _pk[4];
                                        _pk[0] = __floats2bfloat162_rn(o_epi[off_1 + 0], o_epi[off_1 + 1]);
                                        _pk[1] = __floats2bfloat162_rn(o_epi[off_1 + 2], o_epi[off_1 + 3]);
                                        _pk[2] = __floats2bfloat162_rn(o_epi[off_1 + 4], o_epi[off_1 + 5]);
                                        _pk[3] = __floats2bfloat162_rn(o_epi[off_1 + 6], o_epi[off_1 + 7]);
                                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O_ptr + (o_idx + off_1)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                    }
                                }
                                if (half_c == 0) {
                                    *(reinterpret_cast<float*>(LSE_ptr + ((batch_c * q_len + j_c) * num_q_heads + q_head_c)) + (0)) = lse_i;
                                }
                            }
                            asm volatile("barrier.sync 12, 128;" ::: "memory");
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
                            for (int off_2 = 0; off_2 < 64; off_2 += 4) {
                                {
                                    float4 _v4 = make_float4(o_epi[off_2 + 0], o_epi[off_2 + 1], o_epi[off_2 + 2], o_epi[off_2 + 3]);
                                    *reinterpret_cast<float4*>((partial_o + (p_base_1 + off_2)) + 0) = _v4;
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
                        asm volatile("barrier.sync 12, 128;" ::: "memory");
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
                                    float _max_59 = max_noftz(m_f_1, m_k_1);
                                    float m_new_1 = _max_59;
                                    float _exp2_16 = approx_exp2((m_f_1 - m_new_1) * softmax_scale_log2);
                                    float a_k_1 = _exp2_16;
                                    float _exp2_17 = approx_exp2((m_k_1 - m_new_1) * softmax_scale_log2);
                                    float b_k_1 = _exp2_17;
                                    float _fma_19 = __fmaf_rn(l_k_1, b_k_1, l_f_1 * a_k_1);
                                    l_f_1 = _fma_19;
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
                                        float _fma_20 = __fmaf_rn(_vec_load_2[k_1], b_k_1, acc_f_1[k_1] * a_k_1);
                                        acc_f_1[k_1] = _fma_20;
                                    }
                                    m_f_1 = m_new_1;
                                }
                                float _rcp_15 = approx_rcp(l_f_1);
                                float inv_f_1 = ((l_f_1 > 0.0f) ? _rcp_15 : 0.0f);
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
                    int _mma_a_lo_0 = make_warp_uniform((((smem_k_addr) >> 4) & 0x3FFF) + (k_cons_stage) * 2048);
                    int _mma_b_lo_0 = make_warp_uniform((((smem_q32_addr) >> 4) & 0x3FFF) + (0) * 512);
                    {
                        uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
                        uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_tmem_sT + (s_buf * 64)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134743184, 0);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_tmem_sT + (s_buf * 64)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134743184, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_tmem_sT + (s_buf * 64)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134743184, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_tmem_sT + (s_buf * 64)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134743184, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 1018U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 250U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_tmem_sT + (s_buf * 64)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134743184, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_tmem_sT + (s_buf * 64)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134743184, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_tmem_sT + (s_buf * 64)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134743184, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
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
                            {
                                mbarrier_wait(k_full_addr + (k_cons_stage) * 8, k_cons_phase);
                            }
                            if (lane == 0) {
                            }
                            int _mma_a_lo_2 = make_warp_uniform((((smem_k_addr) >> 4) & 0x3FFF) + (k_cons_stage) * 2048);
                            int _mma_b_lo_2 = make_warp_uniform((((smem_q32_addr) >> 4) & 0x3FFF) + (0) * 512);
                            {
                                uint64_t _mma_ss_a_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_2);
                                uint64_t _mma_ss_b_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_2);
                                if (elect_sync()) {
                                    tcgen05_mma_f16((tmem_tmem_sT + (s_buf * 64)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134743184, 0);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f16((tmem_tmem_sT + (s_buf * 64)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134743184, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f16((tmem_tmem_sT + (s_buf * 64)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134743184, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f16((tmem_tmem_sT + (s_buf * 64)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134743184, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_2, 1018U);
                                incr_smem_desc_lo(_mma_ss_b_desc_2, 250U);
                                if (elect_sync()) {
                                    tcgen05_mma_f16((tmem_tmem_sT + (s_buf * 64)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134743184, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f16((tmem_tmem_sT + (s_buf * 64)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134743184, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f16((tmem_tmem_sT + (s_buf * 64)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134743184, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                                if (elect_sync()) {
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
                        {
                            mbarrier_wait(v_full_addr + (v_cons_stage) * 8, v_cons_phase);
                        }
                        if (lane == 0) {
                        }
                        if (pv_idx_m >= 2) {
                            unsigned int k_obs_m = pv_idx_m - 2;
                            {
                                mbarrier_wait(o_ready_addr + ((int)(k_obs_m & 1)) * 8, (int)(k_obs_m >> 1 & 1));
                            }
                        }
                        {
                            mbarrier_wait(p_full_addr + (pf_stage) * 8, pf_phase);
                        }
                        if (lane == 0) {
                        }
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int first_pv_flag = first_pv;
                        int rows_hi_m = ((N_ROWS > 32) ? 1 : 0);
                        int pb_m = 0;
                        if (rows_hi_m == 0) {
                            pb_m = (int)(pv_idx_m & 1);
                        }
                        if (pb_m == 0) {
                            int _mma_a_lo_4 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_cons_stage) * 2048);
                            int _mma_b_lo_4 = make_warp_uniform((((swap_pT_addr) >> 4) & 0x3FFF) | 0x2000000);
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
                    int _max_3 = (((seqlen_kv_p + PAGE_SIZE - 1) / PAGE_SIZE - 1) > (0) ? ((seqlen_kv_p + PAGE_SIZE - 1) / PAGE_SIZE - 1) : (0));
                    int max_pg_p = _max_3;
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
                }
            }
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
                    float _rcp_1 = approx_rcp((float)sow_n_max);
                    unsigned int q_7 = (unsigned int)((float)sow_ticket * _rcp_1);
                    if (sow_ticket < q_7 * sow_n_max) {
                        q_7 = q_7 - 1;
                    }
                    if (sow_ticket >= (q_7 + 1) * sow_n_max) {
                        q_7 = q_7 + 1;
                    }
                    sow_tile = q_7;
                    sow_chunk = (int)(sow_ticket - sow_tile * sow_n_max);
                } else {
                    sow_kind = 1;
                    sow_r = sow_ticket - sow_chunk_items;
                    sow_tile = sow_r >> sow_shift;
                    sow_slice = sow_r - (sow_tile << sow_shift);
                }
                float _rcp_2 = approx_rcp((float)num_kv_heads);
                unsigned int q_8 = (unsigned int)((float)sow_tile * _rcp_2);
                if (sow_tile < q_8 * (unsigned int)num_kv_heads) {
                    q_8 = q_8 - 1;
                }
                if (sow_tile >= (q_8 + 1) * (unsigned int)num_kv_heads) {
                    q_8 = q_8 + 1;
                }
                sow_batch = (int)q_8;
                sow_head = (int)(sow_tile - (unsigned int)sow_batch * (unsigned int)num_kv_heads);
                int sow_last = causal_seqlens_kv_global[sow_batch] + (q_len - 1) - cp_rank;
                int sow_cp_mask = (1 << cp_world_log2) - 1;
                if (sow_last >= 0) {
                    sow_len = (sow_last >> cp_world_log2) + 1;
                    sow_phase = sow_last & sow_cp_mask;
                }
                int _max_1 = ((sow_len) > (1) ? (sow_len) : (1));
                int sow_pairs = (_max_1 + 255) / 256;
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
                        int _max_2 = (((sow_len + BLOCK_N - 1) / BLOCK_N) > (1) ? ((sow_len + BLOCK_N - 1) / BLOCK_N) : (1));
                        int sow_nblk = _max_2;
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
            unsigned int q_prod_stage = 0;
            unsigned int q_prod_phase = 1;
            unsigned int page_cons_stage = 0;
            unsigned int page_cons_phase = 0;
            unsigned int k_prod_stage = 0;
            unsigned int k_prod_phase = 1;
            unsigned int v_prod_stage = 0;
            unsigned int v_prod_phase = 1;
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
            mbarrier_arrive(work_empty_addr + (work_stage_l) * 8);
            work_stage_l += 1;
            if (work_stage_l == 4) { work_stage_l = 0; _phase_work_full_4 ^= 1; }
            unsigned int valid_l = valid_5;
            int kind_l = (int)kind_5;
            int batch_idx_l = (int)batch_5;
            int kv_head_idx = (int)kv_head_5;
            int block_begin_l = (int)block_begin_4;
            int block_end_l = (int)block_end_4;
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
                        for (int ni = 0; ni < cta_n_blocks; ni++) {
                            int nk = ni + 3;
                            if (nk < cta_n_blocks) {
                                int k_page_u = page_cons_stage + 3;
                                int k_page = ((k_page_u >= 6) ? k_page_u - 6 : k_page_u);
                                int k_page_phase = ((k_page_u >= 6) ? page_cons_phase ^ 1 : page_cons_phase);
                                int kpg_base = k_page * 8;
                                mbarrier_wait(page_offsets_full_addr + (k_page) * 8, k_page_phase);
                                int pg_nk[8];
                                #pragma unroll
                                for (int pg_i_4 = 0; pg_i_4 < 8; pg_i_4++) {
                                    pg_nk[pg_i_4] = smem_page_offsets[kpg_base + pg_i_4];
                                }
                                mbarrier_wait(k_empty_addr + (k_prod_stage) * 8, k_prod_phase);
                                mbarrier_arrive_expect_tx(k_full_addr + (k_prod_stage) * 8, 32768);
                                int kdst = smem_k_addr + k_prod_stage * 32768;
                                #pragma unroll
                                for (int pg_i_5 = 0; pg_i_5 < 8; pg_i_5++) {
                                    int npg0 = pg_nk[pg_i_5];
                                    #pragma unroll
                                    for (int hg_2 = 0; hg_2 < 2; hg_2++) {
                                        int ntoff = hg_2 * 16384 + pg_i_5 * 2048;
                                        tma_5d_gmem2smem(kdst + ntoff, (&K), 0, 0, hg_2, kv_head_idx, npg0, k_full_addr + (k_prod_stage) * 8);
                                    }
                                }
                                k_prod_stage += 1;
                                if (k_prod_stage == 3) { k_prod_stage = 0; k_prod_phase ^= 1; }
                            }
                            int nv = ni + 2;
                            if (nv < cta_n_blocks) {
                                int v_page_u = page_cons_stage + 2;
                                int v_page = ((v_page_u >= 6) ? v_page_u - 6 : v_page_u);
                                int v_page_phase = ((v_page_u >= 6) ? page_cons_phase ^ 1 : page_cons_phase);
                                int vpg_base = v_page * 8;
                                mbarrier_wait(page_offsets_full_addr + (v_page) * 8, v_page_phase);
                                int pg_nv[8];
                                #pragma unroll
                                for (int pg_i_6 = 0; pg_i_6 < 8; pg_i_6++) {
                                    pg_nv[pg_i_6] = smem_page_offsets[vpg_base + pg_i_6];
                                }
                                mbarrier_wait(v_empty_addr + (v_prod_stage) * 8, v_prod_phase);
                                mbarrier_arrive_expect_tx(v_full_addr + (v_prod_stage) * 8, 32768);
                                int vdst = smem_v_addr + v_prod_stage * 32768;
                                #pragma unroll
                                for (int pg_i_7 = 0; pg_i_7 < 8; pg_i_7++) {
                                    int vpg1 = pg_nv[pg_i_7];
                                    #pragma unroll
                                    for (int hg_3 = 0; hg_3 < 2; hg_3++) {
                                        int vtoff = hg_3 * 16384 + pg_i_7 * 2048;
                                        tma_5d_gmem2smem(vdst + vtoff, (&V), 0, 0, hg_3, kv_head_idx, vpg1, v_full_addr + (v_prod_stage) * 8);
                                    }
                                }
                                v_prod_stage += 1;
                                if (v_prod_stage == 3) { v_prod_stage = 0; v_prod_phase ^= 1; }
                            }
                            if (n_pre > ni) {
                                mbarrier_wait(page_offsets_full_addr + (page_cons_stage) * 8, page_cons_phase);
                            }
                            mbarrier_arrive(page_offsets_empty_addr + (page_cons_stage) * 8);
                            page_cons_stage += 1;
                            if (page_cons_stage == 6) { page_cons_stage = 0; page_cons_phase ^= 1; }
                            if (ni == gate_block) {
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
