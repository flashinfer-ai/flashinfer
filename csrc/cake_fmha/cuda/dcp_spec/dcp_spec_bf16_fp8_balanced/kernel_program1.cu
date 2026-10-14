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
#define TMEM_TMEM_O_OFFSET 256
#define NUM_Q_PIPE_STAGES 2
#define NUM_Q_RAW_PIPE_STAGES 1
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
#define SMEM_RSTAGE_DATA_OFF 13312
#define SMEM_RSTAGE_DATA_STAGE_BYTES 131072
#define SMEM_RSTAGE_DATA_STRIDE 131072
#define SMEM_RSTAGE_STATS_OFF 144384
#define SMEM_RSTAGE_STATS_STAGE_BYTES 16384
#define SMEM_RSTAGE_STATS_STRIDE 16384
#define SMEM_SMEM_QBF_OFF 46080
#define SMEM_SMEM_QBF_STAGE_BYTES 16384
#define SMEM_SMEM_QBF_STRIDE 16384
#define SMEM_SMEM_Q_HI_OFF 13312
#define SMEM_SMEM_Q_HI_STAGE_BYTES 8192
#define SMEM_SMEM_Q_HI_STRIDE 8192
#define SMEM_SMEM_Q_LO_OFF 29696
#define SMEM_SMEM_Q_LO_STAGE_BYTES 8192
#define SMEM_SMEM_Q_LO_STRIDE 8192
#define SMEM_SMEM_K_OFF 62464
#define SMEM_SMEM_K_STAGE_BYTES 16384
#define SMEM_SMEM_K_STRIDE 16384
#define SMEM_SMEM_V_OFF 111616
#define SMEM_SMEM_V_STAGE_BYTES 16384
#define SMEM_SMEM_V_STRIDE 16384
#define SMEM_TOTAL 160768
#define THREADS 384
#ifndef N_ROWS
#define N_ROWS 64
#endif
#define BLOCK_N 128
#define HEAD_DIM 128
#define PAGE_SIZE 64
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

__global__ __launch_bounds__(384) void
kernel_cake_fmha_dcp_spec_bf16_fp8_balanced(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap V, __nv_bfloat16* __restrict__ O_ptr, float* __restrict__ LSE_ptr, int* __restrict__ page_table, int* __restrict__ causal_seqlens_kv_global, float* __restrict__ partial_o, float* __restrict__ partial_stats, unsigned int* __restrict__ tile_counters, unsigned int* __restrict__ queue_counters, int max_pages_per_seq, float softmax_scale_log2, float output_scale, int num_q_heads, int num_kv_heads, int batch_size, int q_len, int cp_rank, int cp_world_log2, unsigned int max_items)
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
    #define q_empty_addr (mbar_base + 32)
    #define k_full_addr (mbar_base + 48)
    #define k_empty_addr (mbar_base + 72)
    #define v_full_addr (mbar_base + 96)
    #define v_empty_addr (mbar_base + 120)
    #define s_full_addr (mbar_base + 144)
    #define p_full_addr (mbar_base + 160)
    #define o_ready_addr (mbar_base + 176)
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
    float* smem_xmax = reinterpret_cast<float*>(smem_raw + 1024);
    const int smem_xmax_addr = smem + 1024;
    float* smem_sum = reinterpret_cast<float*>(smem_raw + 3072);
    const int smem_sum_addr = smem + 3072;
    unsigned int* smem_corr_flag = reinterpret_cast<unsigned int*>(smem_raw + 4608);
    const int smem_corr_flag_addr = smem + 4608;
    float* smem_max = reinterpret_cast<float*>(smem_raw + 3584);
    const int smem_max_addr = smem + 3584;
    int* smem_page_offsets = reinterpret_cast<int*>(smem_raw + 4096);
    const int smem_page_offsets_addr = smem + 4096;
    unsigned int* work_token_words = reinterpret_cast<unsigned int*>(smem_raw + 4352);
    const int work_token_words_addr = smem + 4352;
    int* sched_seq_lens = reinterpret_cast<int*>(smem_raw + 5120);
    const int sched_seq_lens_addr = smem + 5120;
    int* sched_phase = reinterpret_cast<int*>(smem_raw + 9216);
    const int sched_phase_addr = smem + 9216;
    float* rstage_data = reinterpret_cast<float*>(smem_raw + 13312);
    const int rstage_data_addr = smem + 13312;
    float* rstage_stats = reinterpret_cast<float*>(smem_raw + 144384);
    const int rstage_stats_addr = smem + 144384;
    __nv_bfloat16* smem_qbf = reinterpret_cast<__nv_bfloat16*>(smem_raw + 46080);
    const int smem_qbf_addr = smem + 46080;
    uint8_t* smem_q_hi = reinterpret_cast<uint8_t*>(smem_raw + 13312);
    const int smem_q_hi_addr = smem + 13312;
    uint8_t* smem_q_lo = reinterpret_cast<uint8_t*>(smem_raw + 29696);
    const int smem_q_lo_addr = smem + 29696;
    uint8_t* smem_k = reinterpret_cast<uint8_t*>(smem_raw + 62464);
    const int smem_k_addr = smem + 62464;
    uint8_t* smem_v = reinterpret_cast<uint8_t*>(smem_raw + 111616);
    const int smem_v_addr = smem + 111616;
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Q))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&K))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&V))) : "memory");

    // Mbarrier init (20 pipeline groups, 0 ordered-sequence groups, 51 barriers)
    // Mbarriers at smem_raw[0..408)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'q_raw_pipe' ---
            // q_raw_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // q_raw_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // --- pipeline 'q_pipe' ---
            // q_full: 2 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // q_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // --- pipeline 'k_pipe' ---
            // k_full: 3 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            // k_empty: 3 barriers, init_count=1
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // --- pipeline 'v_pipe' ---
            // v_full: 3 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            // v_empty: 3 barriers, init_count=1
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            // --- pipeline 'sm_pipe' ---
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            // p_full: 2 barriers, init_count=256
            mbarrier_init(smem + 160, 256);
            mbarrier_init(smem + 168, 256);
            // o_ready: 2 barriers, init_count=1
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
    const int tmem_tmem_s = taddr;
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
            int _min_7 = ((row_j) < (q_len - 1) ? (row_j) : (q_len - 1));
            int vis_j = _min_7;
            unsigned int sm_stage = 0;
            unsigned int sm_phase = 0;
            unsigned int xm_slot_s = 0;
            unsigned int st_stage_s = 0;
            unsigned int st_phase_s = 1;
            float _rcp_13 = approx_rcp(softmax_scale_log2);
            float thr_raw = 7.8073549f * _rcp_13;
            float p_scale_log2 = 1.0f;
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
            unsigned int spec_hit = work_token_words[base + 12];
            unsigned int spec_nblocks = work_token_words[base + 13];
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
            if (valid_s != 0) {
                if (kind_s == 0) {
                    mbarrier_wait(q_raw_full_addr, 0);
                    int q_hi_base_s = smem_q_hi_addr;
                    int q_lo_base_s = smem_q_lo_addr;
                    int live_rows_q = q_len * 8;
                    unsigned int q_words_s[8];
                    float q_f32_s[16];
                    float q_res_s[16];
                    unsigned int q_packed_s[4];
                    unsigned int q_packed_lo_s[4];
                    #pragma unroll
                    for (int qc_i = 0; qc_i < 2; qc_i++) {
                        int q_chunk_s = qc_i * 256 + sm_tid;
                        int q_row_s = q_chunk_s / 8;
                        int q_col16_s = q_chunk_s % 8;
                        if (q_row_s < live_rows_q) {
                            int q_kg_s = q_col16_s / 4;
                            int q_c16a_s = q_col16_s % 4 * 2;
                            int q_key_bf_s = (smem_qbf_addr / 128 + (unsigned int)q_row_s) % 8;
                            int q_src_row_s = smem_qbf_addr + (unsigned int)(q_kg_s * 8192) + (unsigned int)(q_row_s * 128);
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_words_s[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_s[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_s[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_s[(0) + 3]))
                                : "r"(q_src_row_s + (q_c16a_s ^ q_key_bf_s) * 16));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_words_s[4])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_s[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_s[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_s[(4) + 3]))
                                : "r"(q_src_row_s + (q_c16a_s + 1 ^ q_key_bf_s) * 16));
                            #pragma unroll
                            for (int qw_s = 0; qw_s < 8; qw_s++) {
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
                            for (int qz_s = 0; qz_s < 16; qz_s++) {
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
                                : "=r"(_packed) : "f"(q_f32_s[8]), "f"(q_f32_s[9]),
                                                   "f"(q_f32_s[10]), "f"(q_f32_s[11]));
                            q_packed_s[2] = _packed;
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
                                : "=r"(_packed) : "f"(q_f32_s[12]), "f"(q_f32_s[13]),
                                                   "f"(q_f32_s[14]), "f"(q_f32_s[15]));
                            q_packed_s[3] = _packed;
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
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(q_res_s[8]), "f"(q_res_s[9]),
                                                   "f"(q_res_s[10]), "f"(q_res_s[11]));
                            q_packed_lo_s[2] = _packed;
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
                                : "=r"(_packed) : "f"(q_res_s[12]), "f"(q_res_s[13]),
                                                   "f"(q_res_s[14]), "f"(q_res_s[15]));
                            q_packed_lo_s[3] = _packed;
                        }
                        int q_key_hi_s = (q_hi_base_s / 128 + q_row_s) % 8;
                        int q_key_lo_s = (q_lo_base_s / 128 + q_row_s) % 8;
                        int q_hi_addr_s = q_hi_base_s + q_row_s * 128 + (q_col16_s ^ q_key_hi_s) * 16;
                        int q_lo_addr_s = q_lo_base_s + q_row_s * 128 + (q_col16_s ^ q_key_lo_s) * 16;
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_hi_addr_s), "r"((q_packed_s[0])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_hi_addr_s + 4), "r"((q_packed_s[1])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_hi_addr_s + 8), "r"((q_packed_s[2])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_hi_addr_s + 12), "r"((q_packed_s[3])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_lo_addr_s), "r"((q_packed_lo_s[0])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_lo_addr_s + 4), "r"((q_packed_lo_s[1])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_lo_addr_s + 8), "r"((q_packed_lo_s[2])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_lo_addr_s + 12), "r"((q_packed_lo_s[3])));
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync 9, 256;" ::: "memory");
                    if (sm_tid == 0) {
                        mbarrier_arrive(q_raw_empty_addr);
                        mbarrier_arrive(q_full_addr);
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
                    int tail_last_s = 0;
                    {
                        int nbt_s = (seqlen_s + BLOCK_N - 1) / BLOCK_N;
                        if (block_end_s == nbt_s) {
                            if (seqlen_s - (nbt_s - 1) * BLOCK_N < 64) {
                                if (cnt_s >= 2) {
                                    tail_last_s = 1;
                                }
                            }
                        }
                    }
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
                        if (tail_last_s != 0) {
                            my_block = block_begin_s + cnt_s - 2 - n;
                            if (n == cnt_s - 1) {
                                my_block = block_begin_s + cnt_s - 1;
                            }
                        }
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
                                int _max_18 = ((n_vis) > (0) ? (n_vis) : (0));
                                int n_lo = _max_18;
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
                            float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, lmax, 16);
                            float _max_19 = max_noftz(lmax, _shfl_xor_3);
                            lmax = _max_19;
                            if (half == 0) {
                                smem_xmax[xm_off_s + my_row] = lmax;
                            }
                        }
                        if (sm_tid == 0) {
                        }
                        asm volatile("barrier.sync 8, 256;" ::: "memory");
                        if (rows_live != 0) {
                            float _max_20 = max_noftz(lmax, smem_xmax[xm_off_s + 64 + my_row]);
                            lmax = _max_20;
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
                    float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, psum, 16);
                    float total = psum + _shfl_xor_4;
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
                    if (r_row < live_rows_s) {
                        int stats_stride_r = num_kv_heads * 128;
                        int o_stride_r = num_kv_heads * 8192;
                        int stats_row_r = slot_tile_base_s * 128 + r_row;
                        int f_r = 32 >> block_end_s;
                        int d0_r = block_begin_s * (4 * f_r) + (128 + sm_tid) % 4 * f_r;
                        {
                            d0_r = block_begin_s * (4 * f_r) + (128 + sm_tid) % 4 * 16;
                        }
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
                        int narrow_r = 0;
                        if (block_end_s == 3) {
                            narrow_r = 1;
                        }
                        {
                            if (narrow_r == 0) {
                                int _max_23 = ((f_r / 16) > (1) ? (f_r / 16) : (1));
                                int n_span_r = _max_23;
                                int _min_10 = ((f_r / 4) < (4) ? (f_r / 4) : (4));
                                int active_r = _min_10;
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
                                                int c_c = c_b + cb;
                                                if (c_c >= n_chunks_s) {
                                                    c_c = n_chunks_s - 1;
                                                }
                                                m_b[cb] = partial_stats[stats_row_r + c_c * stats_stride_r];
                                                l_b[cb] = partial_stats[stats_row_r + 64 + c_c * stats_stride_r];
                                                #pragma unroll
                                                for (int g = 0; g < 4; g++) {
                                                    {
                                                        float4 _v4 = *reinterpret_cast<const float4*>(partial_o + o_row_r + w_r * 64 + g * 4 + c_c * o_stride_r);
                                                        o_b[cb * 16 + g * 4 + 0] = _v4.x;
                                                        o_b[cb * 16 + g * 4 + 1] = _v4.y;
                                                        o_b[cb * 16 + g * 4 + 2] = _v4.z;
                                                        o_b[cb * 16 + g * 4 + 3] = _v4.w;
                                                    }
                                                }
                                            }
                                            #pragma unroll
                                            for (int cb2 = 0; cb2 < 4; cb2++) {
                                                float m_k = m_b[cb2];
                                                float l_k = l_b[cb2];
                                                if (n_chunks_s <= c_b + cb2) {
                                                    m_k = -1e+30f;
                                                    l_k = 0.0f;
                                                }
                                                float _max_24 = max_noftz(m_w, m_k);
                                                float m_new = _max_24;
                                                float _exp2_3 = approx_exp2((m_w - m_new) * softmax_scale_log2);
                                                float a_k = _exp2_3;
                                                float _exp2_4 = approx_exp2((m_k - m_new) * softmax_scale_log2);
                                                float b_k = _exp2_4;
                                                float _fma_2 = __fmaf_rn(l_k, b_k, l_w * a_k);
                                                l_w = _fma_2;
                                                #pragma unroll
                                                for (int k2 = 0; k2 < 16; k2++) {
                                                    float _fma_3 = __fmaf_rn(o_b[cb2 * 16 + k2], b_k, acc_w[k2] * a_k);
                                                    acc_w[k2] = _fma_3;
                                                }
                                                m_w = m_new;
                                            }
                                        }
                                        float _rcp_15 = approx_rcp(l_w);
                                        float inv_w = ((l_w > 0.0f) ? _rcp_15 * output_scale : 0.0f);
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
                                                float _log2_1;
                                                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_1) : "f"(l_w));
                                                lse_w = m_w * softmax_scale_log2 + _log2_1 - 1.0f;
                                            }
                                            *(reinterpret_cast<float*>(LSE_ptr + lse_idx_r) + (0)) = lse_w;
                                        }
                                    }
                                }
                            }
                            if (narrow_r != 0) {
                                int o_row_n = slot_tile_base_s * 8192 + r_row * 128 + (block_begin_s * (4 * f_r) + (128 + sm_tid) % 4 * f_r);
                                int o_idx_n = ((batch_s * q_len + j_r) * num_q_heads + q_head_r) * HEAD_DIM + block_begin_s * (4 * f_r) + (128 + sm_tid) % 4 * f_r;
                                float acc_f[4];
                                float out4[4];
                                int n_pad_r = (n_chunks_s + 7) / 8 * 8;
                                #pragma unroll 1
                                for (int g_r = 0; g_r < 1; g_r++) {
                                    acc_f[0] = 0.0f;
                                    acc_f[1] = 0.0f;
                                    acc_f[2] = 0.0f;
                                    acc_f[3] = 0.0f;
                                    float m_f = -1e+30f;
                                    float l_f = 0.0f;
                                    int o_col_r = o_row_n + g_r * 4;
                                    #pragma unroll 8
                                    for (int c_m = 0; c_m < n_pad_r; c_m++) {
                                        int c_c_1 = c_m;
                                        if (n_chunks_s <= c_m) {
                                            c_c_1 = n_chunks_s - 1;
                                        }
                                        float m_k_1 = partial_stats[stats_row_r + c_c_1 * stats_stride_r];
                                        float l_k_1 = partial_stats[stats_row_r + 64 + c_c_1 * stats_stride_r];
                                        if (n_chunks_s <= c_m) {
                                            m_k_1 = -1e+30f;
                                            l_k_1 = 0.0f;
                                        }
                                        float _max_25 = max_noftz(m_f, m_k_1);
                                        float m_new_1 = _max_25;
                                        float _exp2_5 = approx_exp2((m_f - m_new_1) * softmax_scale_log2);
                                        float a_k_1 = _exp2_5;
                                        float _exp2_6 = approx_exp2((m_k_1 - m_new_1) * softmax_scale_log2);
                                        float b_k_1 = _exp2_6;
                                        float _fma_4 = __fmaf_rn(l_k_1, b_k_1, l_f * a_k_1);
                                        l_f = _fma_4;
                                        float _vec_load_0[4];
                                        {
                                            float4 _v4 = *reinterpret_cast<const float4*>(partial_o + (o_col_r + c_c_1 * o_stride_r) + 0);
                                            _vec_load_0[0 + 0] = _v4.x;
                                            _vec_load_0[0 + 1] = _v4.y;
                                            _vec_load_0[0 + 2] = _v4.z;
                                            _vec_load_0[0 + 3] = _v4.w;
                                        }
                                        #pragma unroll
                                        for (int k = 0; k < 4; k++) {
                                            float _fma_5 = __fmaf_rn(_vec_load_0[k], b_k_1, acc_f[k] * a_k_1);
                                            acc_f[k] = _fma_5;
                                        }
                                        m_f = m_new_1;
                                    }
                                    float _rcp_16 = approx_rcp(l_f);
                                    float inv_f = ((l_f > 0.0f) ? _rcp_16 * output_scale : 0.0f);
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
                                                float _log2_2;
                                                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_2) : "f"(l_f));
                                                lse_r = m_f * softmax_scale_log2 + _log2_2 - 1.0f;
                                            }
                                            *(reinterpret_cast<float*>(LSE_ptr + lse_idx_r) + (0)) = lse_r;
                                        }
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
                unsigned int spec_hit_13 = work_token_words[base_0 + 12];
                unsigned int spec_nblocks_14 = work_token_words[base_0 + 13];
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
            int _min_11 = ((row_j_c) < (q_len - 1) ? (row_j_c) : (q_len - 1));
            int vis_j_c = _min_11;
            float _rcp_18 = approx_rcp(softmax_scale_log2);
            float thr_raw_c = 7.8073549f * _rcp_18;
            float p_scale_log2_c = 1.0f;
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
                                if (spec_nb_c == 0) {
                                    asm volatile("cp.async.bulk.prefetch.tensor.5d.L2.global.tile [%0, {%1, %2, %3, %4, %5}];" :: "l"((uint64_t)((&V))), "r"((int)(0)), "r"((int)(0)), "r"((int)(0)), "r"((int)(spec_head_c)), "r"((int)(spec_page_c)) : "memory");
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
            unsigned int spec_hit_1 = work_token_words[base_1 + 12];
            unsigned int spec_nblocks_1 = work_token_words[base_1 + 13];
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
            if (valid_c != 0) {
                if (kind_c == 0) {
                    mbarrier_wait(q_raw_full_addr, 0);
                    int q_hi_base_s_1 = smem_q_hi_addr;
                    int q_lo_base_s_1 = smem_q_lo_addr;
                    int live_rows_q_1 = q_len * 8;
                    unsigned int q_words_s_1[8];
                    float q_f32_s_1[16];
                    float q_res_s_1[16];
                    unsigned int q_packed_s_1[4];
                    unsigned int q_packed_lo_s_1[4];
                    #pragma unroll
                    for (int qc_i_1 = 0; qc_i_1 < 2; qc_i_1++) {
                        int q_chunk_s_1 = qc_i_1 * 256 + (128 + wg_tid_c);
                        int q_row_s_1 = q_chunk_s_1 / 8;
                        int q_col16_s_1 = q_chunk_s_1 % 8;
                        if (q_row_s_1 < live_rows_q_1) {
                            int q_kg_s_1 = q_col16_s_1 / 4;
                            int q_c16a_s_1 = q_col16_s_1 % 4 * 2;
                            int q_key_bf_s_1 = (smem_qbf_addr / 128 + (unsigned int)q_row_s_1) % 8;
                            int q_src_row_s_1 = smem_qbf_addr + (unsigned int)(q_kg_s_1 * 8192) + (unsigned int)(q_row_s_1 * 128);
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_1[(0) + 3]))
                                : "r"(q_src_row_s_1 + (q_c16a_s_1 ^ q_key_bf_s_1) * 16));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_1[(4) + 3]))
                                : "r"(q_src_row_s_1 + (q_c16a_s_1 + 1 ^ q_key_bf_s_1) * 16));
                            #pragma unroll
                            for (int qw_s_1 = 0; qw_s_1 < 8; qw_s_1++) {
                                unsigned int q_w_s_1 = q_words_s_1[qw_s_1];
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
                                q_f32_s_1[2 * qw_s_1] = q_lo_t_d_1;
                                q_res_s_1[2 * qw_s_1] = q_lo_d_1 - q_lo_t_d_1;
                                q_f32_s_1[2 * qw_s_1 + 1] = q_hi_t_d_1;
                                q_res_s_1[2 * qw_s_1 + 1] = q_hi_d_1 - q_hi_t_d_1;
                            }
                        } else {
                            #pragma unroll
                            for (int qz_s_1 = 0; qz_s_1 < 16; qz_s_1++) {
                                q_f32_s_1[qz_s_1] = 0.0f;
                                q_res_s_1[qz_s_1] = 0.0f;
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
                                : "=r"(_packed) : "f"(q_f32_s_1[8]), "f"(q_f32_s_1[9]),
                                                   "f"(q_f32_s_1[10]), "f"(q_f32_s_1[11]));
                            q_packed_s_1[2] = _packed;
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
                                : "=r"(_packed) : "f"(q_f32_s_1[12]), "f"(q_f32_s_1[13]),
                                                   "f"(q_f32_s_1[14]), "f"(q_f32_s_1[15]));
                            q_packed_s_1[3] = _packed;
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
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(q_res_s_1[8]), "f"(q_res_s_1[9]),
                                                   "f"(q_res_s_1[10]), "f"(q_res_s_1[11]));
                            q_packed_lo_s_1[2] = _packed;
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
                                : "=r"(_packed) : "f"(q_res_s_1[12]), "f"(q_res_s_1[13]),
                                                   "f"(q_res_s_1[14]), "f"(q_res_s_1[15]));
                            q_packed_lo_s_1[3] = _packed;
                        }
                        int q_key_hi_s_1 = (q_hi_base_s_1 / 128 + q_row_s_1) % 8;
                        int q_key_lo_s_1 = (q_lo_base_s_1 / 128 + q_row_s_1) % 8;
                        int q_hi_addr_s_1 = q_hi_base_s_1 + q_row_s_1 * 128 + (q_col16_s_1 ^ q_key_hi_s_1) * 16;
                        int q_lo_addr_s_1 = q_lo_base_s_1 + q_row_s_1 * 128 + (q_col16_s_1 ^ q_key_lo_s_1) * 16;
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_hi_addr_s_1), "r"((q_packed_s_1[0])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_hi_addr_s_1 + 4), "r"((q_packed_s_1[1])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_hi_addr_s_1 + 8), "r"((q_packed_s_1[2])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_hi_addr_s_1 + 12), "r"((q_packed_s_1[3])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_lo_addr_s_1), "r"((q_packed_lo_s_1[0])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_lo_addr_s_1 + 4), "r"((q_packed_lo_s_1[1])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_lo_addr_s_1 + 8), "r"((q_packed_lo_s_1[2])));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_lo_addr_s_1 + 12), "r"((q_packed_lo_s_1[3])));
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync 9, 256;" ::: "memory");
                }
            }
            #pragma unroll 1
            for (unsigned int _tile_iter_c = 0; _tile_iter_c < max_items; _tile_iter_c++) {
                if (valid_c == 0) {
                    break;
                }
                if (kind_c == 0) {
                    int cnt_c = block_end_c - block_begin_c;
                    int tail_last_c = 0;
                    {
                        int nbt_c = (seqlen_c + BLOCK_N - 1) / BLOCK_N;
                        if (block_end_c == nbt_c) {
                            if (seqlen_c - (nbt_c - 1) * BLOCK_N < 64) {
                                if (cnt_c >= 2) {
                                    tail_last_c = 1;
                                }
                            }
                        }
                    }
                    int my_slot = slot_tile_base_c + chunk_c * num_kv_heads;
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
                        if (tail_last_c != 0) {
                            my_block_c = block_begin_c + cnt_c - 2 - n_1;
                            if (n_1 == cnt_c - 1) {
                                my_block_c = block_begin_c + cnt_c - 1;
                            }
                        }
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
                                int _max_27 = ((n_vis_c) > (0) ? (n_vis_c) : (0));
                                int n_lo_c = _max_27;
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
                            float _max_28 = max_noftz(lmax_c, _shfl_xor_5);
                            lmax_c = _max_28;
                            if (half_c == 0) {
                                smem_xmax[xm_off_c + 64 + my_row_c] = lmax_c;
                            }
                        }
                        asm volatile("barrier.sync 8, 256;" ::: "memory");
                        if (rows_live_c != 0) {
                            float _max_29 = max_noftz(lmax_c, smem_xmax[xm_off_c + my_row_c]);
                            lmax_c = _max_29;
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
                        int _vote_2 = __any_sync(0xFFFFFFFF, acc_scale_c != 1.0f);
                        if (_vote_2 != 0) {
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
                    float _tmem_load_1[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[31]))
                        : "r"(o_row_base));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[32])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[33])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[34])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[35])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[36])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[37])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[38])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[39])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[40])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[41])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[42])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[43])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[44])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[45])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[46])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[47])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[48])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[49])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[50])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[51])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[52])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[53])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[54])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[55])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[56])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[57])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[58])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[59])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[60])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[61])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[62])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[63]))
                        : "r"(o_row_base + 32));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    __syncwarp();
                    if (lane == 0) {
                        asm volatile(
                            "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;"
                            :: "r"(o_empty_addr), "r"((uint32_t)(32)) : "memory");
                    }
                    if (wg_tid_c == 0) {
                    }
                    mbarrier_wait(stats_full_addr + (st_stage_c) * 8, st_phase_c);
                    int st_off = (int)st_stage_c * 64;
                    float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, psum_c, 16);
                    float total_c = psum_c + _shfl_xor_6;
                    if (rows_live_c != 0) {
                        if (half_c == 0) {
                            smem_sum[st_off + my_row_c] = smem_sum[st_off + my_row_c] + total_c;
                        }
                    }
                    asm volatile("barrier.sync 11, 128;" ::: "memory");
                    if (wg_tid_c == 0) {
                    }
                    int publish_split = ((n_chunks_c > 1) ? 1 : 0);
                    if (publish_split == 0) {
                        float row_sum_c = 1.0f;
                        if (my_row_c < N_ROWS) {
                            row_sum_c = smem_sum[st_off + my_row_c];
                        }
                        float _rcp_19 = approx_rcp(row_sum_c);
                        float inv_c = ((row_sum_c > 0.0f) ? _rcp_19 : 0.0f);
                        if (row_sum_c <= 0.0f) {
                            #pragma unroll
                            for (int z_c = 0; z_c < 64; z_c++) {
                                _tmem_load_1[z_c] = 0.0f;
                            }
                        }
                        if (my_row_c < live_rows) {
                            int j_c = my_row_c / 8;
                            int h_c = my_row_c % 8;
                            int q_head_c = kv_head_c * 8 + h_c;
                            int o_idx = ((batch_c * q_len + j_c) * num_q_heads + q_head_c) * HEAD_DIM + half_c * 64;
                            #pragma unroll
                            for (int off = 0; off < 64; off += 8) {
                                {
                                    const float2 _prescale2_22 = {inv_c * output_scale, inv_c * output_scale};
                                    #if __CUDA_ARCH__ >= 1000
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 4; _ps++)
                                        mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_1[off])[_ps], _prescale2_22);
                                    #else
                                    #pragma unroll
                                    for (int _ps = 0; _ps < 8; _ps++)
                                        _tmem_load_1[off + _ps] *= inv_c * output_scale;
                                    #endif
                                    __nv_bfloat162 _pk[4];
                                    _pk[0] = __floats2bfloat162_rn(_tmem_load_1[off + 0], _tmem_load_1[off + 1]);
                                    _pk[1] = __floats2bfloat162_rn(_tmem_load_1[off + 2], _tmem_load_1[off + 3]);
                                    _pk[2] = __floats2bfloat162_rn(_tmem_load_1[off + 4], _tmem_load_1[off + 5]);
                                    _pk[3] = __floats2bfloat162_rn(_tmem_load_1[off + 6], _tmem_load_1[off + 7]);
                                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O_ptr + (o_idx + off)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
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
                    } else if (n_chunks_c == 2) {
                        if (my_row_c < N_ROWS) {
                            int p_base = my_slot * 8192 + my_row_c * 128 + half_c * 64;
                            #pragma unroll
                            for (int off_1 = 0; off_1 < 64; off_1 += 4) {
                                {
                                    float4 _v4 = make_float4(_tmem_load_1[off_1 + 0], _tmem_load_1[off_1 + 1], _tmem_load_1[off_1 + 2], _tmem_load_1[off_1 + 3]);
                                    *reinterpret_cast<float4*>((partial_o + (p_base + off_1)) + 0) = _v4;
                                }
                            }
                        }
                        if (wg_tid_c < N_ROWS) {
                            *(reinterpret_cast<float*>(partial_stats + (my_slot * 128 + wg_tid_c)) + (0)) = smem_max[st_off + wg_tid_c];
                            *(reinterpret_cast<float*>(partial_stats + (my_slot * 128 + 64 + wg_tid_c)) + (0)) = smem_sum[st_off + wg_tid_c];
                        }
                        asm volatile("barrier.sync 11, 128;" ::: "memory");
                        if (wg_tid_c == 0) {
                            {
                                asm volatile("fence.release.gpu;" ::: "memory");
                            }
                            unsigned int _atomic_old_4;
                            asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                                : "=r"(_atomic_old_4) : "l"(&tile_counters[counter_idx_c * 4]), "r"(static_cast<uint32_t>(1)) : "memory");
                            unsigned int arrived_old = _atomic_old_4;
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
                            int other_slot = slot_tile_base_c + (1 - chunk_c) * num_kv_heads;
                            float w_s_c = 0.0f;
                            float w_o_c = 0.0f;
                            float lse_i = -CAKE_FMHA_INF;
                            if (my_row_c < N_ROWS) {
                                float m_o = partial_stats[other_slot * 128 + my_row_c];
                                float l_o = partial_stats[other_slot * 128 + 64 + my_row_c];
                                float m_s = smem_max[st_off + my_row_c];
                                float l_s = smem_sum[st_off + my_row_c];
                                float _max_30 = max_noftz(m_s, m_o);
                                float m_row_i = _max_30;
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
                                float _rcp_20 = approx_rcp(den_i);
                                float inv_i = ((den_i > 0.0f) ? _rcp_20 * output_scale : 0.0f);
                                w_s_c = w_s * inv_i;
                                w_o_c = w_o * inv_i;
                                if (den_i > 0.0f) {
                                    float _log2_5;
                                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_5) : "f"(den_i));
                                    lse_i = m_row_i * softmax_scale_log2 + _log2_5 - p_scale_log2_c;
                                }
                            }
                            int oth_base = other_slot * 8192 + my_row_c * 128 + half_c * 64;
                            if (my_row_c < live_rows) {
                                #pragma unroll
                                for (int c0 = 0; c0 < 64; c0 += 4) {
                                    float _vec_load_2[4];
                                    {
                                        float4 _v4 = *reinterpret_cast<const float4*>(partial_o + (oth_base + c0) + 0);
                                        _vec_load_2[0 + 0] = _v4.x;
                                        _vec_load_2[0 + 1] = _v4.y;
                                        _vec_load_2[0 + 2] = _v4.z;
                                        _vec_load_2[0 + 3] = _v4.w;
                                    }
                                    #pragma unroll
                                    for (int c = 0; c < 4; c++) {
                                        {
                                            float _fma_10 = __fmaf_rn(_tmem_load_1[c0 + c], w_s_c, _vec_load_2[c] * w_o_c);
                                            float _fma_11 = __fmaf_rn(_vec_load_2[c], w_o_c, _tmem_load_1[c0 + c] * w_s_c);
                                            float o_det_d1 = ((chunk_c == 0) ? _fma_10 : _fma_11);
                                            _tmem_load_1[c0 + c] = o_det_d1;
                                        }
                                    }
                                }
                                int j_c_1 = my_row_c / 8;
                                int h_c_1 = my_row_c % 8;
                                int q_head_c_1 = kv_head_c * 8 + h_c_1;
                                int o_idx_1 = ((batch_c * q_len + j_c_1) * num_q_heads + q_head_c_1) * HEAD_DIM + half_c * 64;
                                #pragma unroll
                                for (int off_2 = 0; off_2 < 64; off_2 += 8) {
                                    {
                                        __nv_bfloat162 _pk[4];
                                        _pk[0] = __floats2bfloat162_rn(_tmem_load_1[off_2 + 0], _tmem_load_1[off_2 + 1]);
                                        _pk[1] = __floats2bfloat162_rn(_tmem_load_1[off_2 + 2], _tmem_load_1[off_2 + 3]);
                                        _pk[2] = __floats2bfloat162_rn(_tmem_load_1[off_2 + 4], _tmem_load_1[off_2 + 5]);
                                        _pk[3] = __floats2bfloat162_rn(_tmem_load_1[off_2 + 6], _tmem_load_1[off_2 + 7]);
                                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O_ptr + (o_idx_1 + off_2)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
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
                        if (my_row_c < N_ROWS) {
                            int p_base_1 = my_slot * 8192 + my_row_c * 128 + half_c * 64;
                            #pragma unroll
                            for (int off_3 = 0; off_3 < 64; off_3 += 4) {
                                {
                                    float4 _v4 = make_float4(_tmem_load_1[off_3 + 0], _tmem_load_1[off_3 + 1], _tmem_load_1[off_3 + 2], _tmem_load_1[off_3 + 3]);
                                    *reinterpret_cast<float4*>((partial_o + (p_base_1 + off_3)) + 0) = _v4;
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
                        asm volatile("barrier.sync 11, 128;" ::: "memory");
                        if (wg_tid_c == 0) {
                            {
                                asm volatile("fence.release.gpu;" ::: "memory");
                            }
                            unsigned int _atomic_old_5;
                            asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                                : "=r"(_atomic_old_5) : "l"(&tile_counters[counter_idx_c * 4]), "r"(static_cast<uint32_t>(1)) : "memory");
                            unsigned int arrived_old_1 = _atomic_old_5;
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
                        {
                            d0_r_1 = block_begin_c * (4 * f_r_1) + wg_tid_c % 4 * 16;
                        }
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
                        int narrow_r_1 = 0;
                        if (block_end_c == 3) {
                            narrow_r_1 = 1;
                        }
                        {
                            if (narrow_r_1 == 0) {
                                int _max_33 = ((f_r_1 / 16) > (1) ? (f_r_1 / 16) : (1));
                                int n_span_r_1 = _max_33;
                                int _min_14 = ((f_r_1 / 4) < (4) ? (f_r_1 / 4) : (4));
                                int active_r_1 = _min_14;
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
                                                int c_c_2 = c_b_1 + cb_1;
                                                if (c_c_2 >= n_chunks_c) {
                                                    c_c_2 = n_chunks_c - 1;
                                                }
                                                m_b_1[cb_1] = partial_stats[stats_row_r_1 + c_c_2 * stats_stride_r_1];
                                                l_b_1[cb_1] = partial_stats[stats_row_r_1 + 64 + c_c_2 * stats_stride_r_1];
                                                #pragma unroll
                                                for (int g_1 = 0; g_1 < 4; g_1++) {
                                                    {
                                                        float4 _v4 = *reinterpret_cast<const float4*>(partial_o + o_row_r_1 + w_r_1 * 64 + g_1 * 4 + c_c_2 * o_stride_r_1);
                                                        o_b_1[cb_1 * 16 + g_1 * 4 + 0] = _v4.x;
                                                        o_b_1[cb_1 * 16 + g_1 * 4 + 1] = _v4.y;
                                                        o_b_1[cb_1 * 16 + g_1 * 4 + 2] = _v4.z;
                                                        o_b_1[cb_1 * 16 + g_1 * 4 + 3] = _v4.w;
                                                    }
                                                }
                                            }
                                            #pragma unroll
                                            for (int cb2_1 = 0; cb2_1 < 4; cb2_1++) {
                                                float m_k_2 = m_b_1[cb2_1];
                                                float l_k_2 = l_b_1[cb2_1];
                                                if (n_chunks_c <= c_b_1 + cb2_1) {
                                                    m_k_2 = -1e+30f;
                                                    l_k_2 = 0.0f;
                                                }
                                                float _max_34 = max_noftz(m_w_1, m_k_2);
                                                float m_new_2 = _max_34;
                                                float _exp2_14 = approx_exp2((m_w_1 - m_new_2) * softmax_scale_log2);
                                                float a_k_2 = _exp2_14;
                                                float _exp2_15 = approx_exp2((m_k_2 - m_new_2) * softmax_scale_log2);
                                                float b_k_2 = _exp2_15;
                                                float _fma_15 = __fmaf_rn(l_k_2, b_k_2, l_w_1 * a_k_2);
                                                l_w_1 = _fma_15;
                                                #pragma unroll
                                                for (int k2_1 = 0; k2_1 < 16; k2_1++) {
                                                    float _fma_16 = __fmaf_rn(o_b_1[cb2_1 * 16 + k2_1], b_k_2, acc_w_1[k2_1] * a_k_2);
                                                    acc_w_1[k2_1] = _fma_16;
                                                }
                                                m_w_1 = m_new_2;
                                            }
                                        }
                                        float _rcp_22 = approx_rcp(l_w_1);
                                        float inv_w_1 = ((l_w_1 > 0.0f) ? _rcp_22 * output_scale : 0.0f);
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
                                                float _log2_7;
                                                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_7) : "f"(l_w_1));
                                                lse_w_1 = m_w_1 * softmax_scale_log2 + _log2_7 - 1.0f;
                                            }
                                            *(reinterpret_cast<float*>(LSE_ptr + lse_idx_r_1) + (0)) = lse_w_1;
                                        }
                                    }
                                }
                            }
                            if (narrow_r_1 != 0) {
                                int o_row_n_1 = slot_tile_base_c * 8192 + r_row_1 * 128 + (block_begin_c * (4 * f_r_1) + wg_tid_c % 4 * f_r_1);
                                int o_idx_n_1 = ((batch_c * q_len + j_r_1) * num_q_heads + q_head_r_1) * HEAD_DIM + block_begin_c * (4 * f_r_1) + wg_tid_c % 4 * f_r_1;
                                float acc_f_1[4];
                                float out4_1[4];
                                int n_pad_r_1 = (n_chunks_c + 7) / 8 * 8;
                                #pragma unroll 1
                                for (int g_r_1 = 0; g_r_1 < 1; g_r_1++) {
                                    acc_f_1[0] = 0.0f;
                                    acc_f_1[1] = 0.0f;
                                    acc_f_1[2] = 0.0f;
                                    acc_f_1[3] = 0.0f;
                                    float m_f_1 = -1e+30f;
                                    float l_f_1 = 0.0f;
                                    int o_col_r_1 = o_row_n_1 + g_r_1 * 4;
                                    #pragma unroll 8
                                    for (int c_m_1 = 0; c_m_1 < n_pad_r_1; c_m_1++) {
                                        int c_c_3 = c_m_1;
                                        if (n_chunks_c <= c_m_1) {
                                            c_c_3 = n_chunks_c - 1;
                                        }
                                        float m_k_3 = partial_stats[stats_row_r_1 + c_c_3 * stats_stride_r_1];
                                        float l_k_3 = partial_stats[stats_row_r_1 + 64 + c_c_3 * stats_stride_r_1];
                                        if (n_chunks_c <= c_m_1) {
                                            m_k_3 = -1e+30f;
                                            l_k_3 = 0.0f;
                                        }
                                        float _max_35 = max_noftz(m_f_1, m_k_3);
                                        float m_new_3 = _max_35;
                                        float _exp2_16 = approx_exp2((m_f_1 - m_new_3) * softmax_scale_log2);
                                        float a_k_3 = _exp2_16;
                                        float _exp2_17 = approx_exp2((m_k_3 - m_new_3) * softmax_scale_log2);
                                        float b_k_3 = _exp2_17;
                                        float _fma_17 = __fmaf_rn(l_k_3, b_k_3, l_f_1 * a_k_3);
                                        l_f_1 = _fma_17;
                                        float _vec_load_3[4];
                                        {
                                            float4 _v4 = *reinterpret_cast<const float4*>(partial_o + (o_col_r_1 + c_c_3 * o_stride_r_1) + 0);
                                            _vec_load_3[0 + 0] = _v4.x;
                                            _vec_load_3[0 + 1] = _v4.y;
                                            _vec_load_3[0 + 2] = _v4.z;
                                            _vec_load_3[0 + 3] = _v4.w;
                                        }
                                        #pragma unroll
                                        for (int k_1 = 0; k_1 < 4; k_1++) {
                                            float _fma_18 = __fmaf_rn(_vec_load_3[k_1], b_k_3, acc_f_1[k_1] * a_k_3);
                                            acc_f_1[k_1] = _fma_18;
                                        }
                                        m_f_1 = m_new_3;
                                    }
                                    float _rcp_23 = approx_rcp(l_f_1);
                                    float inv_f_1 = ((l_f_1 > 0.0f) ? _rcp_23 * output_scale : 0.0f);
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
                                                float _log2_8;
                                                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_8) : "f"(l_f_1));
                                                lse_r_1 = m_f_1 * softmax_scale_log2 + _log2_8 - 1.0f;
                                            }
                                            *(reinterpret_cast<float*>(LSE_ptr + lse_idx_r_1) + (0)) = lse_r_1;
                                        }
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
                unsigned int spec_hit_13_1 = work_token_words[base_0_1 + 12];
                unsigned int spec_nblocks_14_1 = work_token_words[base_0_1 + 13];
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
            unsigned int spec_hit_2 = work_token_words[base_2 + 12];
            unsigned int spec_nblocks_2 = work_token_words[base_2 + 13];
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
                    int _mma_a_lo_0 = make_warp_uniform((((smem_q_hi_addr) >> 4) & 0x3FFF) + (q_cons_stage) * 512);
                    int _mma_b_lo_0 = make_warp_uniform((((smem_k_addr) >> 4) & 0x3FFF) + (k_cons_stage) * 1024);
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
                    int _mma_a_lo_1 = make_warp_uniform((((smem_q_lo_addr) >> 4) & 0x3FFF) + (q_cons_stage) * 512);
                    int _mma_b_lo_1 = make_warp_uniform((((smem_k_addr) >> 4) & 0x3FFF) + (k_cons_stage) * 1024);
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
                            int _mma_a_lo_2 = make_warp_uniform((((smem_q_hi_addr) >> 4) & 0x3FFF) + (q_cons_stage) * 512);
                            int _mma_b_lo_2 = make_warp_uniform((((smem_k_addr) >> 4) & 0x3FFF) + (k_cons_stage) * 1024);
                            {
                                uint64_t _mma_ss_a_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_2);
                                uint64_t _mma_ss_b_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_2);
                                if (elect_sync()) {
                                    tcgen05_mma_f8f6f4((tmem_tmem_s + (s_buf * 128)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 69206032, 0);
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
                            int _mma_a_lo_3 = make_warp_uniform((((smem_q_lo_addr) >> 4) & 0x3FFF) + (q_cons_stage) * 512);
                            int _mma_b_lo_3 = make_warp_uniform((((smem_k_addr) >> 4) & 0x3FFF) + (k_cons_stage) * 1024);
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
                        int _mma_b_lo_4 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_cons_stage) * 1024);
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
                    :: "r"(tmem_tmem_o), "r"(_mma_b_lo_4), "r"(tmem_tmem_s + (int)pf_stage * 128), "r"(((first_pv_flag) ? 0 : 1)));
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
                    q_cons_stage += 1;
                    if (q_cons_stage == 2) { q_cons_stage = 0; q_cons_phase ^= 1; }
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
                unsigned int spec_hit_13_2 = work_token_words[base_0_2 + 12];
                unsigned int spec_nblocks_14_2 = work_token_words[base_0_2 + 13];
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
            unsigned int spec_hit_3 = work_token_words[base_3 + 12];
            unsigned int spec_nblocks_3 = work_token_words[base_3 + 13];
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
            int spec_hit_p = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_p = 0; _tile_iter_p < max_items; _tile_iter_p++) {
                if (valid_p == 0) {
                    break;
                }
                if (kind_p == 0) {
                    int cta_n_blocks_p = block_end_p - block_begin_p;
                    int tail_last_p = 0;
                    {
                        int nbt_p = (seqlen_kv_p + BLOCK_N - 1) / BLOCK_N;
                        if (block_end_p == nbt_p) {
                            if (seqlen_kv_p - (nbt_p - 1) * BLOCK_N < 64) {
                                if (cta_n_blocks_p >= 2) {
                                    tail_last_p = 1;
                                }
                            }
                        }
                    }
                    int _max_16 = (((seqlen_kv_p + PAGE_SIZE - 1) / PAGE_SIZE - 1) > (0) ? ((seqlen_kv_p + PAGE_SIZE - 1) / PAGE_SIZE - 1) : (0));
                    int max_pg_p = _max_16;
                    int pt_base_p = batch_idx_p * max_pages_per_seq;
                    #pragma unroll 1
                    for (int ni0_p = 0; ni0_p < cta_n_blocks_p; ni0_p += 4) {
                        int skip_g_p = 0;
                        if (skip_g_p == 0) {
                            int g_cnt_p = cta_n_blocks_p - ni0_p;
                            if (g_cnt_p > 4) {
                                g_cnt_p = 4;
                            }
                            int k_p = ni0_p + pg_blk_p;
                            int n_block_p = block_begin_p + cta_n_blocks_p - 1 - k_p;
                            if (tail_last_p != 0) {
                                n_block_p = block_begin_p + cta_n_blocks_p - 2 - k_p;
                                if (k_p == cta_n_blocks_p - 1) {
                                    n_block_p = block_begin_p + cta_n_blocks_p - 1;
                                }
                            }
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
                            if (lane == 0) {
                            }
                        }
                        if (ni0_p == 0) {
                            int wide_done_p = 0;
                            if (_tile_iter_p == 0) {
                                wide_done_p = 1;
                            }
                            if (wide_done_p == 0) {
                                mbarrier_wait(q_raw_full_addr + (qr_cons_stage) * 8, qr_cons_phase);
                                mbarrier_wait(q_empty_addr + (q_prod_stage) * 8, q_prod_phase);
                                int q_hi_base_s_2 = smem_q_hi_addr + q_prod_stage * 8192;
                                int q_lo_base_s_2 = smem_q_lo_addr + q_prod_stage * 8192;
                                int live_rows_q_2 = q_len * 8;
                                unsigned int q_words_s_2[8];
                                float q_f32_s_2[16];
                                float q_res_s_2[16];
                                unsigned int q_packed_s_2[4];
                                unsigned int q_packed_lo_s_2[4];
                                #pragma unroll
                                for (int qc_i_2 = 0; qc_i_2 < 16; qc_i_2++) {
                                    int q_chunk_s_2 = qc_i_2 * 32 + lane;
                                    int q_row_s_2 = q_chunk_s_2 / 8;
                                    int q_col16_s_2 = q_chunk_s_2 % 8;
                                    if (q_row_s_2 < live_rows_q_2) {
                                        int q_kg_s_2 = q_col16_s_2 / 4;
                                        int q_c16a_s_2 = q_col16_s_2 % 4 * 2;
                                        int q_key_bf_s_2 = ((smem_qbf_addr + qr_cons_stage * 16384) / 128 + (unsigned int)q_row_s_2) % 8;
                                        int q_src_row_s_2 = smem_qbf_addr + qr_cons_stage * 16384 + (unsigned int)(q_kg_s_2 * 8192) + (unsigned int)(q_row_s_2 * 128);
                                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                            : "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_2[(0) + 3]))
                                            : "r"(q_src_row_s_2 + (q_c16a_s_2 ^ q_key_bf_s_2) * 16));
                                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                            : "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_2[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_2[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_2[(4) + 3]))
                                            : "r"(q_src_row_s_2 + (q_c16a_s_2 + 1 ^ q_key_bf_s_2) * 16));
                                        #pragma unroll
                                        for (int qw_s_2 = 0; qw_s_2 < 8; qw_s_2++) {
                                            unsigned int q_w_s_2 = q_words_s_2[qw_s_2];
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
                                            q_f32_s_2[2 * qw_s_2] = q_lo_t_d_2;
                                            q_res_s_2[2 * qw_s_2] = q_lo_d_2 - q_lo_t_d_2;
                                            q_f32_s_2[2 * qw_s_2 + 1] = q_hi_t_d_2;
                                            q_res_s_2[2 * qw_s_2 + 1] = q_hi_d_2 - q_hi_t_d_2;
                                        }
                                    } else {
                                        #pragma unroll
                                        for (int qz_s_2 = 0; qz_s_2 < 16; qz_s_2++) {
                                            q_f32_s_2[qz_s_2] = 0.0f;
                                            q_res_s_2[qz_s_2] = 0.0f;
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
                                            : "=r"(_packed) : "f"(q_f32_s_2[8]), "f"(q_f32_s_2[9]),
                                                               "f"(q_f32_s_2[10]), "f"(q_f32_s_2[11]));
                                        q_packed_s_2[2] = _packed;
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
                                            : "=r"(_packed) : "f"(q_f32_s_2[12]), "f"(q_f32_s_2[13]),
                                                               "f"(q_f32_s_2[14]), "f"(q_f32_s_2[15]));
                                        q_packed_s_2[3] = _packed;
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
                                    {
                                        uint32_t _packed;
                                        asm volatile("{\n\t"
                                            ".reg .b16 _lo;\n\t"
                                            ".reg .b16 _hi;\n\t"
                                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                            "mov.b32 %0, {_lo, _hi};\n\t"
                                            "}"
                                            : "=r"(_packed) : "f"(q_res_s_2[8]), "f"(q_res_s_2[9]),
                                                               "f"(q_res_s_2[10]), "f"(q_res_s_2[11]));
                                        q_packed_lo_s_2[2] = _packed;
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
                                            : "=r"(_packed) : "f"(q_res_s_2[12]), "f"(q_res_s_2[13]),
                                                               "f"(q_res_s_2[14]), "f"(q_res_s_2[15]));
                                        q_packed_lo_s_2[3] = _packed;
                                    }
                                    int q_key_hi_s_2 = (q_hi_base_s_2 / 128 + q_row_s_2) % 8;
                                    int q_key_lo_s_2 = (q_lo_base_s_2 / 128 + q_row_s_2) % 8;
                                    int q_hi_addr_s_2 = q_hi_base_s_2 + q_row_s_2 * 128 + (q_col16_s_2 ^ q_key_hi_s_2) * 16;
                                    int q_lo_addr_s_2 = q_lo_base_s_2 + q_row_s_2 * 128 + (q_col16_s_2 ^ q_key_lo_s_2) * 16;
                                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_hi_addr_s_2), "r"((q_packed_s_2[0])));
                                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_hi_addr_s_2 + 4), "r"((q_packed_s_2[1])));
                                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_hi_addr_s_2 + 8), "r"((q_packed_s_2[2])));
                                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_hi_addr_s_2 + 12), "r"((q_packed_s_2[3])));
                                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_lo_addr_s_2), "r"((q_packed_lo_s_2[0])));
                                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_lo_addr_s_2 + 4), "r"((q_packed_lo_s_2[1])));
                                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_lo_addr_s_2 + 8), "r"((q_packed_lo_s_2[2])));
                                    asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_lo_addr_s_2 + 12), "r"((q_packed_lo_s_2[3])));
                                }
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                __syncwarp();
                                if (lane == 0) {
                                    mbarrier_arrive(q_raw_empty_addr + (qr_cons_stage) * 8);
                                    mbarrier_arrive(q_full_addr + (q_prod_stage) * 8);
                                }
                            }
                            qr_cons_phase ^= 1;
                            q_prod_stage += 1;
                            if (q_prod_stage == 2) { q_prod_stage = 0; q_prod_phase ^= 1; }
                        }
                    }
                }
                spec_hit_p = 0;
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
                unsigned int spec_hit_13_3 = work_token_words[base_0_3 + 12];
                unsigned int spec_nblocks_14_3 = work_token_words[base_0_3 + 13];
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
                }
            }
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
                }
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
            if (lane_0 == 0) {
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
                }
                if (chunk_pairs_u < 2 * (p_max - p_min)) {
                    split_ok = 1;
                }
                if (split_ok == 0) {
                    chunk_pairs_u = p_max;
                }
            }
            int whole_items = batch_size * items_per_chunk;
            if (whole_items <= num_ctas) {
                float _rcp_3 = approx_rcp((float)whole_items);
                unsigned int q_0 = (unsigned int)((float)(unsigned int)num_ctas * _rcp_3);
                if (q_0 * (unsigned int)whole_items > (unsigned int)num_ctas) {
                    q_0 = q_0 - 1;
                }
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
                        }
                    }
                }
            }
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
            if (whole_items_u <= (unsigned int)num_ctas) {
                if (l_one <= 8) {
                    dense_u = 1;
                }
            }
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
                    if (dense_u == 1) {
                        if (nb_f > 2) {
                            sp_lane += 1;
                        }
                        unsigned int _max_3 = ((nm_lane) > (nb_f) ? (nm_lane) : (nb_f));
                        nm_lane = _max_3;
                    }
                }
                t_lane = t_lane * (unsigned int)items_per_chunk;
                {
                    unsigned int sh_f = 0;
                    if (nm_lane > 4) {
                        sh_f = 1;
                    }
                    if (nm_lane > 8) {
                        sh_f = 2;
                    }
                    if (nm_lane > 16) {
                        sh_f = 3;
                    }
                    if (t_lane + (sp_lane * (unsigned int)items_per_chunk << 3) <= (unsigned int)num_ctas) {
                        sh_f = 3;
                    }
                    if (dense_u == 1) {
                        if (nm_lane > 2) {
                            t_lane += sp_lane * (unsigned int)items_per_chunk << sh_f;
                        }
                    }
                }
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
            if (lane_0 == 0) {
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
                if (dense_u == 1) {
                    cand = (unsigned int)(cand_idx + 1);
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
                if (dense_u == 1) {
                    tail_c = 12;
                }
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
                if (tickets_c + (splits_c * (unsigned int)items_per_chunk << 3) <= (unsigned int)num_ctas) {
                    sh_c = 3;
                }
                unsigned int one_c = 1;
                tail_c = 11 + (nmax_c + (one_c << sh_c) - 1 >> sh_c);
                if (dense_u == 1) {
                    tail_c = 6 + (4 * (nmax_c + (one_c << sh_c) - 1 >> sh_c) >> 1);
                }
                red_c = splits_c * (unsigned int)items_per_chunk << sh_c;
            }
            unsigned int a_last_c = tickets_c - (waves_c - 1) * (unsigned int)num_ctas;
            unsigned int _max_7 = ((a_last_c) > (sm_floor) ? (a_last_c) : (sm_floor));
            unsigned int eff_c = _max_7;
            unsigned int late_c = 0;
            if (dense_u == 1) {
                if (waves_c == 1) {
                    unsigned int q_0_5 = (unsigned int)((float)(100 * (unsigned int)num_ctas + 45 * tickets_c) * sw_rcp);
                    if (q_0_5 * sw_den > 100 * (unsigned int)num_ctas + 45 * tickets_c) {
                        q_0_5 = q_0_5 - 1;
                    }
                    if ((q_0_5 + 1) * sw_den <= 100 * (unsigned int)num_ctas + 45 * tickets_c) {
                        q_0_5 = q_0_5 + 1;
                    }
                    eff_c = q_0_5;
                }
                if (tickets_c + red_c > (unsigned int)num_ctas) {
                    late_c = 2;
                }
            }
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
            if (lane_0 == 0) {
            }
            int chunk_pairs = (int)chunk_pairs_u;
            float _rcp_11 = approx_rcp((float)chunk_pairs_u);
            float chunk_rcp = _rcp_11;
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
                        int _max_8 = ((s2) > (1) ? (s2) : (1));
                        unsigned int pairs2 = (unsigned int)((_max_8 + 255) / 256);
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
                    unsigned int _max_9 = ((nmax_split) > (_warp_redux_u32_8) ? (nmax_split) : (_warp_redux_u32_8));
                    nmax_split = _max_9;
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
            float _rcp_12 = approx_rcp((float)full_den_u);
            float full_rcp_u = _rcp_12;
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
            if (chunk_items + (n_split_requests * (unsigned int)items_per_chunk << 3) <= (unsigned int)num_ctas) {
                if (n_split_requests > 0) {
                    reduce_shift = 3;
                }
            }
            unsigned int reduce_items = n_split_requests * (unsigned int)items_per_chunk << reduce_shift;
            unsigned int total_items = chunk_items + reduce_items;
            if (blockIdx.x == 0) {
                if (lane_0 == 0) {
                    int plan_off = num_ctas * 2048;
                    *(reinterpret_cast<float*>(partial_stats + plan_off) + (0)) = (float)chunk_pairs_u;
                    *(reinterpret_cast<float*>(partial_stats + (plan_off + 1)) + (0)) = (float)total_items;
                }
            }
            if (lane_0 == 0) {
            }
            unsigned int work_stage_sched = 0;
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
            unsigned int gate_phase = 0;
            unsigned int _phase_work_empty = 1;
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
                int tok_batch_s = -1;
                int tok_head_s = -1;
                int tok_bb_s = -1;
                int tok_be_s = -1;
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
                                    int _max_10 = ((s_r) > (1) ? (s_r) : (1));
                                    int pairs_r = (_max_10 + 255) / 256;
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
                                unsigned int q_6 = (unsigned int)((float)local_chunk * full_rcp_u);
                                if (local_chunk < q_6 * full_den_u) {
                                    q_6 = q_6 - 1;
                                }
                                if (local_chunk >= (q_6 + 1) * full_den_u) {
                                    q_6 = q_6 + 1;
                                }
                                sel_batch = (int)q_6;
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
                                    int _max_11 = ((s3) > (1) ? (s3) : (1));
                                    pairs3 = (_max_11 + 255) / 256;
                                    unsigned int q_6_1 = (unsigned int)((float)(unsigned int)pairs3 * chunk_rcp);
                                    if (q_6_1 * chunk_pairs_u > (unsigned int)pairs3) {
                                        q_6_1 = q_6_1 - 1;
                                    }
                                    if ((q_6_1 + 1) * chunk_pairs_u <= (unsigned int)pairs3) {
                                        q_6_1 = q_6_1 + 1;
                                    }
                                    if (q_6_1 * chunk_pairs_u < (unsigned int)pairs3) {
                                        q_6_1 = q_6_1 + 1;
                                    }
                                    n3 = (int)q_6_1;
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
                        int _max_12 = (((sel_seqlen + BLOCK_N - 1) / BLOCK_N) > (1) ? ((sel_seqlen + BLOCK_N - 1) / BLOCK_N) : (1));
                        int n_blocks_tile = _max_12;
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
                        tok_batch_s = sel_batch;
                        tok_head_s = kv_head_sel;
                        tok_bb_s = block_begin_4;
                        tok_be_s = block_end_4;
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
                int spec_hit_s = 0;
                int spec_nb_s = 0;
                if (lane_0 == 0) {
                    work_token_words[token_base + 12] = (unsigned int)spec_hit_s;
                    work_token_words[token_base + 13] = (unsigned int)spec_nb_s;
                    work_token_words[token_base] = valid_tok;
                    mbarrier_arrive(work_full_addr + (work_stage_sched) * 8);
                    if (_claim == 0) {
                    }
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
                }
                work_stage_sched += 1;
                if (work_stage_sched == 4) { work_stage_sched = 0; _phase_work_empty ^= 1; }
                if (valid_tok == 0) {
                    break;
                }
            }
            unsigned int done_old = 0;
            if (lane_0 == 0) {
                uint32_t _atomic_inc_old_0;
                asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                    : "=r"(_atomic_inc_old_0) : "l"(&queue_counters[1]), "r"(static_cast<uint32_t>(num_ctas - 1)) : "memory");
                done_old = _atomic_inc_old_0;
            }
            unsigned int _shfl_22 = __shfl_sync(0xFFFFFFFF, done_old, 0);
            done_old = _shfl_22;
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
            int spec_nb_l = 0;
            int spec_npre_l = 0;
            unsigned int _phase_work_full_4 = 0;
            mbarrier_wait(work_full_addr + (work_stage_l) * 8, _phase_work_full_4);
            unsigned int base_4 = work_stage_l * 16;
            unsigned int valid_5 = work_token_words[base_4];
            unsigned int kind_5 = work_token_words[base_4 + 1];
            unsigned int batch_5 = work_token_words[base_4 + 2];
            unsigned int kv_head_5 = work_token_words[base_4 + 3];
            unsigned int block_begin_6 = work_token_words[base_4 + 4];
            unsigned int block_end_5 = work_token_words[base_4 + 5];
            unsigned int seqlen_4 = work_token_words[base_4 + 6];
            unsigned int n_chunks_4 = work_token_words[base_4 + 7];
            unsigned int slot_tile_base_5 = work_token_words[base_4 + 8];
            unsigned int counter_idx_5 = work_token_words[base_4 + 9];
            unsigned int chunk_4 = work_token_words[base_4 + 10];
            unsigned int phase_4 = work_token_words[base_4 + 11];
            unsigned int spec_hit_4 = work_token_words[base_4 + 12];
            unsigned int spec_nblocks_4 = work_token_words[base_4 + 13];
            mbarrier_arrive(work_empty_addr + (work_stage_l) * 8);
            work_stage_l += 1;
            if (work_stage_l == 4) { work_stage_l = 0; _phase_work_full_4 ^= 1; }
            unsigned int valid_l = valid_5;
            int kind_l = (int)kind_5;
            int batch_idx_l = (int)batch_5;
            int kv_head_idx = (int)kv_head_5;
            int block_begin_l = (int)block_begin_6;
            int block_end_l = (int)block_end_5;
            int skip_q_l = 0;
            int ni_start_l = 0;
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
                    int gate_block = cta_n_blocks - 1 - 8;
                    if (gate_block < 0) {
                        gate_block = 0;
                    }
                    if (elect_sync()) {
                        if (skip_q_l == 0) {
                            mbarrier_wait(q_raw_empty_addr + (qr_prod_stage) * 8, qr_prod_phase);
                            if (_tile_iter_l == 0) {
                            }
                            mbarrier_arrive_expect_tx(q_raw_full_addr + (qr_prod_stage) * 8, 64 * HEAD_DIM * 2);
                            tma_4d_gmem2smem(smem_qbf_addr + qr_prod_stage * 16384, (&Q), 0, kv_head_idx * 8, batch_idx_l * q_len, 0, q_raw_full_addr + (qr_prod_stage) * 8);
                        }
                        #pragma unroll 1
                        for (int ni = ni_start_l; ni < n_pre; ni++) {
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
                            mbarrier_wait(k_empty_addr + (k_prod_stage) * 8, k_prod_phase);
                            mbarrier_arrive_expect_tx(k_full_addr + (k_prod_stage) * 8, 16384);
                            int kdst0 = smem_k_addr + k_prod_stage * 16384;
                            #pragma unroll
                            for (int pg_i_1 = 0; pg_i_1 < 2; pg_i_1++) {
                                int kpg0 = pg_pre[pg_i_1];
                                int ktoff0 = pg_i_1 * 8192;
                                tma_5d_gmem2smem(kdst0 + ktoff0, (&K), 0, 0, 0, kv_head_idx, kpg0, k_full_addr + (k_prod_stage) * 8);
                            }
                            k_prod_stage += 1;
                            if (k_prod_stage == 3) { k_prod_stage = 0; k_prod_phase ^= 1; }
                            if (ni < 3) {
                                mbarrier_wait(v_empty_addr + (v_prod_stage) * 8, v_prod_phase);
                                mbarrier_arrive_expect_tx(v_full_addr + (v_prod_stage) * 8, 16384);
                                int vdst0 = smem_v_addr + v_prod_stage * 16384;
                                #pragma unroll
                                for (int pg_i_2 = 0; pg_i_2 < 2; pg_i_2++) {
                                    int vpg0 = pg_pre[pg_i_2];
                                    int vtoff0 = pg_i_2 * 8192;
                                    tma_5d_gmem2smem(vdst0 + vtoff0, (&V), 0, 0, 0, kv_head_idx, vpg0, v_full_addr + (v_prod_stage) * 8);
                                }
                                v_prod_stage += 1;
                                if (v_prod_stage == 3) { v_prod_stage = 0; v_prod_phase ^= 1; }
                            }
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
                                mbarrier_arrive_expect_tx(k_full_addr + (k_prod_stage) * 8, 16384);
                                int kdst = smem_k_addr + k_prod_stage * 16384;
                                #pragma unroll
                                for (int pg_i_4 = 0; pg_i_4 < 2; pg_i_4++) {
                                    int npg0 = pg_nk[pg_i_4];
                                    int ntoff = pg_i_4 * 8192;
                                    tma_5d_gmem2smem(kdst + ntoff, (&K), 0, 0, 0, kv_head_idx, npg0, k_full_addr + (k_prod_stage) * 8);
                                }
                                k_prod_stage += 1;
                                if (k_prod_stage == 3) { k_prod_stage = 0; k_prod_phase ^= 1; }
                            }
                            int nv = ni_1 + 3;
                            if (nv < cta_n_blocks) {
                                int v_page_u = page_cons_stage + 3;
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
                                mbarrier_arrive_expect_tx(v_full_addr + (v_prod_stage) * 8, 16384);
                                int vdst = smem_v_addr + v_prod_stage * 16384;
                                #pragma unroll
                                for (int pg_i_6 = 0; pg_i_6 < 2; pg_i_6++) {
                                    int vpg1 = pg_nv[pg_i_6];
                                    int vtoff = pg_i_6 * 8192;
                                    tma_5d_gmem2smem(vdst + vtoff, (&V), 0, 0, 0, kv_head_idx, vpg1, v_full_addr + (v_prod_stage) * 8);
                                }
                                v_prod_stage += 1;
                                if (v_prod_stage == 3) { v_prod_stage = 0; v_prod_phase ^= 1; }
                            }
                            mbarrier_arrive(page_offsets_empty_addr + (page_cons_stage) * 8);
                            page_cons_stage += 1;
                            if (page_cons_stage == 6) { page_cons_stage = 0; page_cons_phase ^= 1; }
                            if (ni_1 == gate_block) {
                                mbarrier_arrive(claim_gate_addr);
                            }
                        }
                    }
                    if (skip_q_l == 0) {
                        qr_prod_phase ^= 1;
                    }
                    skip_q_l = 0;
                    ni_start_l = 0;
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
                unsigned int spec_hit_13_4 = work_token_words[base_0_4 + 12];
                unsigned int spec_nblocks_14_4 = work_token_words[base_0_4 + 13];
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
