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




__device__ __forceinline__ void tma_4d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w),
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
    __nv_bfloat16* smem_qbf = reinterpret_cast<__nv_bfloat16*>(smem_raw + 46080);
    const int smem_qbf_addr = smem + 46080;
    uint8_t* smem_q_hi_g0 = reinterpret_cast<uint8_t*>(smem_raw + 13312);
    const int smem_q_hi_g0_addr = smem + 13312;
    uint8_t* smem_q_hi_g1 = reinterpret_cast<uint8_t*>(smem_raw + 21504);
    const int smem_q_hi_g1_addr = smem + 21504;
    uint8_t* smem_q_lo_g0 = reinterpret_cast<uint8_t*>(smem_raw + 29696);
    const int smem_q_lo_g0_addr = smem + 29696;
    uint8_t* smem_q_lo_g1 = reinterpret_cast<uint8_t*>(smem_raw + 37888);
    const int smem_q_lo_g1_addr = smem + 37888;
    uint8_t* smem_q_hi_img = reinterpret_cast<uint8_t*>(smem_raw + 13312);
    const int smem_q_hi_img_addr = smem + 13312;
    uint8_t* smem_q_lo_img = reinterpret_cast<uint8_t*>(smem_raw + 29696);
    const int smem_q_lo_img_addr = smem + 29696;
    uint8_t* smem_k_g0 = reinterpret_cast<uint8_t*>(smem_raw + 62464);
    const int smem_k_g0_addr = smem + 62464;
    uint8_t* smem_k_g1 = reinterpret_cast<uint8_t*>(smem_raw + 111616);
    const int smem_k_g1_addr = smem + 111616;
    uint8_t* smem_v_g0 = reinterpret_cast<uint8_t*>(smem_raw + 160768);
    const int smem_v_g0_addr = smem + 160768;
    uint8_t* smem_v_g1 = reinterpret_cast<uint8_t*>(smem_raw + 193536);
    const int smem_v_g1_addr = smem + 193536;
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
    const int tmem_s = taddr;
    const int tmem_o_g0 = taddr + 256;
    const int tmem_o_g1 = taddr + 384;
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
            int rows_live = ((s_warp * 16 < 64) ? 1 : 0);
            int row_j = my_row / 16;
            unsigned int sm_stage = 0;
            unsigned int xm_slot_s = 0;
            unsigned int st_stage_s = 0;
            float _rcp_0 = approx_rcp(softmax_scale_log2);
            float thr_raw = 8.0f * _rcp_0;
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
            unsigned int plan_flags = work_token_words[base + 13];
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
                    int q_hi_base_s = smem_q_hi_img_addr;
                    int q_lo_base_s = smem_q_lo_img_addr;
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
                                q_lo_d = __uint_as_float(q_wbits_d);
                                q_wbits_d = (q_w_s & 65520) << 16;
                                q_lo_t_d = __uint_as_float(q_wbits_d);
                                q_wbits_d = q_w_s >> 16 << 16;
                                q_hi_d = __uint_as_float(q_wbits_d);
                                q_wbits_d = (q_w_s >> 16 & 65520) << 16;
                                q_hi_t_d = __uint_as_float(q_wbits_d);
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
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_hi_addr_s), "r"(q_packed_s[0]));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_hi_addr_s + 4), "r"(q_packed_s[1]));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_lo_addr_s), "r"(q_packed_lo_s[0]));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_lo_addr_s + 4), "r"(q_packed_lo_s[1]));
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    if (k2x_s != 0) {
                        int q_hi_base_s_0 = smem_q_hi_img_addr;
                        int q_lo_base_s_0 = smem_q_lo_img_addr;
                        int q_key_bf_s_0 = (smem_k_g0_addr + 32768) / 128 % 8;
                        int q_key_hi_s_0 = q_hi_base_s_0 / 128 % 8;
                        int q_key_lo_s_0 = q_lo_base_s_0 / 128 % 8;
                        unsigned int q_words_s_0[4];
                        float q_f32_s_0[8];
                        float q_res_s_0[8];
                        unsigned int q_packed_s_0[2];
                        unsigned int q_packed_lo_s_0[2];
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
                                int q_key_row_s_1 = (q_key_bf_s_0 + q_row_s_1) % 8;
                                int q_src_s_1 = smem_k_g0_addr + 32768 + (unsigned int)(q_kg_s_1 * 4096) + (unsigned int)(q_row_s_1 * 128) + (unsigned int)((q_c16a_s_1 ^ q_key_row_s_1) * 16);
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_0[(0) + 3]))
                                    : "r"(q_src_s_1));
                                #pragma unroll
                                for (int qw_s_1 = 0; qw_s_1 < 4; qw_s_1++) {
                                    unsigned int q_w_s_1 = q_words_s_0[qw_s_1];
                                    unsigned int q_wbits_d_1 = q_w_s_1 << 16;
                                    float q_lo_d_1 = 0.0f;
                                    float q_lo_t_d_1 = 0.0f;
                                    float q_hi_d_1 = 0.0f;
                                    float q_hi_t_d_1 = 0.0f;
                                    q_lo_d_1 = __uint_as_float(q_wbits_d_1);
                                    q_wbits_d_1 = (q_w_s_1 & 65520) << 16;
                                    q_lo_t_d_1 = __uint_as_float(q_wbits_d_1);
                                    q_wbits_d_1 = q_w_s_1 >> 16 << 16;
                                    q_hi_d_1 = __uint_as_float(q_wbits_d_1);
                                    q_wbits_d_1 = (q_w_s_1 >> 16 & 65520) << 16;
                                    q_hi_t_d_1 = __uint_as_float(q_wbits_d_1);
                                    q_f32_s_0[2 * qw_s_1] = q_lo_t_d_1;
                                    q_res_s_0[2 * qw_s_1] = q_lo_d_1 - q_lo_t_d_1;
                                    q_f32_s_0[2 * qw_s_1 + 1] = q_hi_t_d_1;
                                    q_res_s_0[2 * qw_s_1 + 1] = q_hi_d_1 - q_hi_t_d_1;
                                }
                            } else {
                                #pragma unroll
                                for (int qz_s_1 = 0; qz_s_1 < 8; qz_s_1++) {
                                    q_f32_s_0[qz_s_1] = 0.0f;
                                    q_res_s_0[qz_s_1] = 0.0f;
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
                                    : "=r"(_packed) : "f"(q_f32_s_0[0]), "f"(q_f32_s_0[1]),
                                                       "f"(q_f32_s_0[2]), "f"(q_f32_s_0[3]));
                                q_packed_s_0[0] = _packed;
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
                                    : "=r"(_packed) : "f"(q_f32_s_0[4]), "f"(q_f32_s_0[5]),
                                                       "f"(q_f32_s_0[6]), "f"(q_f32_s_0[7]));
                                q_packed_s_0[1] = _packed;
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
                                    : "=r"(_packed) : "f"(q_res_s_0[0]), "f"(q_res_s_0[1]),
                                                       "f"(q_res_s_0[2]), "f"(q_res_s_0[3]));
                                q_packed_lo_s_0[0] = _packed;
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
                                    : "=r"(_packed) : "f"(q_res_s_0[4]), "f"(q_res_s_0[5]),
                                                       "f"(q_res_s_0[6]), "f"(q_res_s_0[7]));
                                q_packed_lo_s_0[1] = _packed;
                            }
                            int q_row_off_s_1 = q_grp_s_1 * 8192 + q_drow_s_1 * 128 + q_half_s_1;
                            int q_hi_addr_s_1 = q_hi_base_s_0 + q_row_off_s_1 + (q_c16_s_1 ^ (q_key_hi_s_0 + q_drow_s_1) % 8) * 16;
                            int q_lo_addr_s_1 = q_lo_base_s_0 + q_row_off_s_1 + (q_c16_s_1 ^ (q_key_lo_s_0 + q_drow_s_1) % 8) * 16;
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_hi_addr_s_1), "r"(q_packed_s_0[0]));
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_hi_addr_s_1 + 4), "r"(q_packed_s_0[1]));
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_lo_addr_s_1), "r"(q_packed_lo_s_0[0]));
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_lo_addr_s_1 + 4), "r"(q_packed_lo_s_0[1]));
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
                        int q_hi_base_s_1 = smem_q_hi_img_addr;
                        int q_lo_base_s_1 = smem_q_lo_img_addr;
                        int q_key_bf_s_1 = smem_qbf_addr / 128 % 8;
                        int q_key_hi_s_1 = q_hi_base_s_1 / 128 % 8;
                        int q_key_lo_s_1 = q_lo_base_s_1 / 128 % 8;
                        unsigned int q_words_s_1[4];
                        float q_f32_s_1[8];
                        float q_res_s_1[8];
                        unsigned int q_packed_s_1[2];
                        unsigned int q_packed_lo_s_1[2];
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
                                int q_key_row_s_2 = (q_key_bf_s_1 + q_row_s_2) % 8;
                                int q_src_s_2 = smem_qbf_addr + (unsigned int)(q_kg_s_2 * 4096) + (unsigned int)(q_row_s_2 * 128) + (unsigned int)((q_c16a_s_2 ^ q_key_row_s_2) * 16);
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_1[(0) + 3]))
                                    : "r"(q_src_s_2));
                                #pragma unroll
                                for (int qw_s_2 = 0; qw_s_2 < 4; qw_s_2++) {
                                    unsigned int q_w_s_2 = q_words_s_1[qw_s_2];
                                    unsigned int q_wbits_d_2 = q_w_s_2 << 16;
                                    float q_lo_d_2 = 0.0f;
                                    float q_lo_t_d_2 = 0.0f;
                                    float q_hi_d_2 = 0.0f;
                                    float q_hi_t_d_2 = 0.0f;
                                    q_lo_d_2 = __uint_as_float(q_wbits_d_2);
                                    q_wbits_d_2 = (q_w_s_2 & 65520) << 16;
                                    q_lo_t_d_2 = __uint_as_float(q_wbits_d_2);
                                    q_wbits_d_2 = q_w_s_2 >> 16 << 16;
                                    q_hi_d_2 = __uint_as_float(q_wbits_d_2);
                                    q_wbits_d_2 = (q_w_s_2 >> 16 & 65520) << 16;
                                    q_hi_t_d_2 = __uint_as_float(q_wbits_d_2);
                                    q_f32_s_1[2 * qw_s_2] = q_lo_t_d_2;
                                    q_res_s_1[2 * qw_s_2] = q_lo_d_2 - q_lo_t_d_2;
                                    q_f32_s_1[2 * qw_s_2 + 1] = q_hi_t_d_2;
                                    q_res_s_1[2 * qw_s_2 + 1] = q_hi_d_2 - q_hi_t_d_2;
                                }
                            } else {
                                #pragma unroll
                                for (int qz_s_2 = 0; qz_s_2 < 8; qz_s_2++) {
                                    q_f32_s_1[qz_s_2] = 0.0f;
                                    q_res_s_1[qz_s_2] = 0.0f;
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
                            int q_row_off_s_2 = q_grp_s_2 * 8192 + q_drow_s_2 * 128 + q_half_s_2;
                            int q_hi_addr_s_2 = q_hi_base_s_1 + q_row_off_s_2 + (q_c16_s_2 ^ (q_key_hi_s_1 + q_drow_s_2) % 8) * 16;
                            int q_lo_addr_s_2 = q_lo_base_s_1 + q_row_off_s_2 + (q_c16_s_2 ^ (q_key_lo_s_1 + q_drow_s_2) % 8) * 16;
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_hi_addr_s_2), "r"(q_packed_s_1[0]));
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_hi_addr_s_2 + 4), "r"(q_packed_s_1[1]));
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_lo_addr_s_2), "r"(q_packed_lo_s_1[0]));
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_lo_addr_s_2 + 4), "r"(q_packed_lo_s_1[1]));
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
            unsigned int _phase_s_full = 0;
            unsigned int _phase_stats_empty = 1;
            #pragma unroll 1
            for (unsigned int _tile_iter_s = 0; _tile_iter_s < max_items; _tile_iter_s++) {
                if (valid_s == 0) {
                    break;
                }
                if (kind_s == 0) {
                    int cnt_s = block_end_s - block_begin_s;
                    int vis_j = ((q_tile_s * 4 + row_j) < (q_len - 1) ? (q_tile_s * 4 + row_j) : (q_len - 1));
                    int back_s = q_len - 1 - vis_j - phase_s;
                    int vis_col = seqlen_s;
                    if (back_s > 0) {
                        vis_col = seqlen_s - (back_s + (1 << cp_world_log2) - 1 >> cp_world_log2);
                    }
                    float row_max = -CUDART_INF_F;
                    float psum = 0.0f;
                    #pragma unroll 1
                    for (int n = 0; n < cnt_s; n++) {
                        mbarrier_wait(s_full_addr + (sm_stage) * 8, _phase_s_full);
                        int my_block = block_begin_s + cnt_s - 1 - n;
                        int blk_pos = my_block * 128 + tok_base;
                        float sv[32];
                        float lmax = -CUDART_INF_F;
                        float new_max = row_max;
                        float acc_scale = 1.0f;
                        float lsum = 0.0f;
                        int xm_off_s = (int)xm_slot_s * 128;
                        if (rows_live != 0) {
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                                : "=r"(*reinterpret_cast<uint32_t*>(&sv[0])), "=r"(*reinterpret_cast<uint32_t*>(&sv[1])), "=r"(*reinterpret_cast<uint32_t*>(&sv[2])), "=r"(*reinterpret_cast<uint32_t*>(&sv[3])), "=r"(*reinterpret_cast<uint32_t*>(&sv[4])), "=r"(*reinterpret_cast<uint32_t*>(&sv[5])), "=r"(*reinterpret_cast<uint32_t*>(&sv[6])), "=r"(*reinterpret_cast<uint32_t*>(&sv[7])), "=r"(*reinterpret_cast<uint32_t*>(&sv[8])), "=r"(*reinterpret_cast<uint32_t*>(&sv[9])), "=r"(*reinterpret_cast<uint32_t*>(&sv[10])), "=r"(*reinterpret_cast<uint32_t*>(&sv[11])), "=r"(*reinterpret_cast<uint32_t*>(&sv[12])), "=r"(*reinterpret_cast<uint32_t*>(&sv[13])), "=r"(*reinterpret_cast<uint32_t*>(&sv[14])), "=r"(*reinterpret_cast<uint32_t*>(&sv[15])), "=r"(*reinterpret_cast<uint32_t*>(&sv[16])), "=r"(*reinterpret_cast<uint32_t*>(&sv[17])), "=r"(*reinterpret_cast<uint32_t*>(&sv[18])), "=r"(*reinterpret_cast<uint32_t*>(&sv[19])), "=r"(*reinterpret_cast<uint32_t*>(&sv[20])), "=r"(*reinterpret_cast<uint32_t*>(&sv[21])), "=r"(*reinterpret_cast<uint32_t*>(&sv[22])), "=r"(*reinterpret_cast<uint32_t*>(&sv[23])), "=r"(*reinterpret_cast<uint32_t*>(&sv[24])), "=r"(*reinterpret_cast<uint32_t*>(&sv[25])), "=r"(*reinterpret_cast<uint32_t*>(&sv[26])), "=r"(*reinterpret_cast<uint32_t*>(&sv[27])), "=r"(*reinterpret_cast<uint32_t*>(&sv[28])), "=r"(*reinterpret_cast<uint32_t*>(&sv[29])), "=r"(*reinterpret_cast<uint32_t*>(&sv[30])), "=r"(*reinterpret_cast<uint32_t*>(&sv[31]))
                                : "r"((unsigned int)my_s_base + sm_stage * 128));
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            int n_vis = vis_col - blk_pos;
                            if (my_row >= 64) {
                                n_vis = 0;
                            }
                            if (n_vis < 32) {
                                int n_lo = ((n_vis) > (0) ? (n_vis) : (0));
                                uint32_t _thresh_mask_2;
                                {
                                    int _lim_3 = n_lo;
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
                            }
                            float2 _reg_reduce_max2_4 = {-CUDART_INF_F, -CUDART_INF_F};
                            _reg_reduce_max2_4.x = max_noftz(_reg_reduce_max2_4.x, max_noftz(sv[0], sv[1]));
                            _reg_reduce_max2_4.y = max_noftz(_reg_reduce_max2_4.y, max_noftz(sv[2], sv[3]));
                            _reg_reduce_max2_4.x = max_noftz(_reg_reduce_max2_4.x, max_noftz(sv[4], sv[5]));
                            _reg_reduce_max2_4.y = max_noftz(_reg_reduce_max2_4.y, max_noftz(sv[6], sv[7]));
                            _reg_reduce_max2_4.x = max_noftz(_reg_reduce_max2_4.x, max_noftz(sv[8], sv[9]));
                            _reg_reduce_max2_4.y = max_noftz(_reg_reduce_max2_4.y, max_noftz(sv[10], sv[11]));
                            _reg_reduce_max2_4.x = max_noftz(_reg_reduce_max2_4.x, max_noftz(sv[12], sv[13]));
                            _reg_reduce_max2_4.y = max_noftz(_reg_reduce_max2_4.y, max_noftz(sv[14], sv[15]));
                            _reg_reduce_max2_4.x = max_noftz(_reg_reduce_max2_4.x, max_noftz(sv[16], sv[17]));
                            _reg_reduce_max2_4.y = max_noftz(_reg_reduce_max2_4.y, max_noftz(sv[18], sv[19]));
                            _reg_reduce_max2_4.x = max_noftz(_reg_reduce_max2_4.x, max_noftz(sv[20], sv[21]));
                            _reg_reduce_max2_4.y = max_noftz(_reg_reduce_max2_4.y, max_noftz(sv[22], sv[23]));
                            _reg_reduce_max2_4.x = max_noftz(_reg_reduce_max2_4.x, max_noftz(sv[24], sv[25]));
                            _reg_reduce_max2_4.y = max_noftz(_reg_reduce_max2_4.y, max_noftz(sv[26], sv[27]));
                            _reg_reduce_max2_4.x = max_noftz(_reg_reduce_max2_4.x, max_noftz(sv[28], sv[29]));
                            _reg_reduce_max2_4.y = max_noftz(_reg_reduce_max2_4.y, max_noftz(sv[30], sv[31]));
                            float sv_max = row_max_reduce(_reg_reduce_max2_4);
                            lmax = sv_max;
                            float _shfl_xor_sync_5 = __shfl_xor_sync(0xFFFFFFFF, lmax, 16);
                            lmax = max_noftz(lmax, _shfl_xor_sync_5);
                            if (half == 0) {
                                smem_xmax[xm_off_s + my_row] = lmax;
                            }
                        }
                        asm volatile("barrier.sync 8, 256;" ::: "memory");
                        if (rows_live != 0) {
                            lmax = max_noftz(lmax, smem_xmax[xm_off_s + 64 + my_row]);
                            if (lmax > row_max + thr_raw) {
                                new_max = lmax;
                                if (row_max > -CUDART_INF_F) {
                                    float _exp2_6 = approx_exp2(softmax_scale_log2 * (row_max - new_max));
                                    acc_scale = _exp2_6;
                                }
                            }
                        }
                        xm_slot_s = xm_slot_s ^ 1;
                        if (rows_live != 0) {
                            float safe_max = ((new_max == -CUDART_INF_F) ? 0.0f : new_max);
                            const float2 _fma_b2_7 = {softmax_scale_log2, softmax_scale_log2};
                            const float2 _fma_c2_8 = {p_scale_log2 - safe_max * softmax_scale_log2, p_scale_log2 - safe_max * softmax_scale_log2};
                            float2 _fma_pair_9 = fma_f32x2(make_float2(sv[0], sv[1]), _fma_b2_7, _fma_c2_8);
                            sv[0] = _fma_pair_9.x;
                            sv[1] = _fma_pair_9.y;
                            float2 _fma_pair_10 = fma_f32x2(make_float2(sv[2], sv[3]), _fma_b2_7, _fma_c2_8);
                            sv[2] = _fma_pair_10.x;
                            sv[3] = _fma_pair_10.y;
                            float2 _fma_pair_11 = fma_f32x2(make_float2(sv[4], sv[5]), _fma_b2_7, _fma_c2_8);
                            sv[4] = _fma_pair_11.x;
                            sv[5] = _fma_pair_11.y;
                            float2 _fma_pair_12 = fma_f32x2(make_float2(sv[6], sv[7]), _fma_b2_7, _fma_c2_8);
                            sv[6] = _fma_pair_12.x;
                            sv[7] = _fma_pair_12.y;
                            float2 _fma_pair_13 = fma_f32x2(make_float2(sv[8], sv[9]), _fma_b2_7, _fma_c2_8);
                            sv[8] = _fma_pair_13.x;
                            sv[9] = _fma_pair_13.y;
                            float2 _fma_pair_14 = fma_f32x2(make_float2(sv[10], sv[11]), _fma_b2_7, _fma_c2_8);
                            sv[10] = _fma_pair_14.x;
                            sv[11] = _fma_pair_14.y;
                            float2 _fma_pair_15 = fma_f32x2(make_float2(sv[12], sv[13]), _fma_b2_7, _fma_c2_8);
                            sv[12] = _fma_pair_15.x;
                            sv[13] = _fma_pair_15.y;
                            float2 _fma_pair_16 = fma_f32x2(make_float2(sv[14], sv[15]), _fma_b2_7, _fma_c2_8);
                            sv[14] = _fma_pair_16.x;
                            sv[15] = _fma_pair_16.y;
                            float2 _fma_pair_17 = fma_f32x2(make_float2(sv[16], sv[17]), _fma_b2_7, _fma_c2_8);
                            sv[16] = _fma_pair_17.x;
                            sv[17] = _fma_pair_17.y;
                            float2 _fma_pair_18 = fma_f32x2(make_float2(sv[18], sv[19]), _fma_b2_7, _fma_c2_8);
                            sv[18] = _fma_pair_18.x;
                            sv[19] = _fma_pair_18.y;
                            float2 _fma_pair_19 = fma_f32x2(make_float2(sv[20], sv[21]), _fma_b2_7, _fma_c2_8);
                            sv[20] = _fma_pair_19.x;
                            sv[21] = _fma_pair_19.y;
                            float2 _fma_pair_20 = fma_f32x2(make_float2(sv[22], sv[23]), _fma_b2_7, _fma_c2_8);
                            sv[22] = _fma_pair_20.x;
                            sv[23] = _fma_pair_20.y;
                            float2 _fma_pair_21 = fma_f32x2(make_float2(sv[24], sv[25]), _fma_b2_7, _fma_c2_8);
                            sv[24] = _fma_pair_21.x;
                            sv[25] = _fma_pair_21.y;
                            float2 _fma_pair_22 = fma_f32x2(make_float2(sv[26], sv[27]), _fma_b2_7, _fma_c2_8);
                            sv[26] = _fma_pair_22.x;
                            sv[27] = _fma_pair_22.y;
                            float2 _fma_pair_23 = fma_f32x2(make_float2(sv[28], sv[29]), _fma_b2_7, _fma_c2_8);
                            sv[28] = _fma_pair_23.x;
                            sv[29] = _fma_pair_23.y;
                            float2 _fma_pair_24 = fma_f32x2(make_float2(sv[30], sv[31]), _fma_b2_7, _fma_c2_8);
                            sv[30] = _fma_pair_24.x;
                            sv[31] = _fma_pair_24.y;
                            #pragma unroll
                            for (int _le = 0; _le < 32; _le++) {
                                sv[_le] = approx_exp2(sv[_le]);
                            }
                            float2 _reg_reduce_sum2_25 = make_float2(0.0f, 0.0f);
                            _reg_reduce_sum2_25 = add_f32x2(_reg_reduce_sum2_25, make_float2(sv[0], sv[1]));
                            _reg_reduce_sum2_25 = add_f32x2(_reg_reduce_sum2_25, make_float2(sv[2], sv[3]));
                            _reg_reduce_sum2_25 = add_f32x2(_reg_reduce_sum2_25, make_float2(sv[4], sv[5]));
                            _reg_reduce_sum2_25 = add_f32x2(_reg_reduce_sum2_25, make_float2(sv[6], sv[7]));
                            _reg_reduce_sum2_25 = add_f32x2(_reg_reduce_sum2_25, make_float2(sv[8], sv[9]));
                            _reg_reduce_sum2_25 = add_f32x2(_reg_reduce_sum2_25, make_float2(sv[10], sv[11]));
                            _reg_reduce_sum2_25 = add_f32x2(_reg_reduce_sum2_25, make_float2(sv[12], sv[13]));
                            _reg_reduce_sum2_25 = add_f32x2(_reg_reduce_sum2_25, make_float2(sv[14], sv[15]));
                            _reg_reduce_sum2_25 = add_f32x2(_reg_reduce_sum2_25, make_float2(sv[16], sv[17]));
                            _reg_reduce_sum2_25 = add_f32x2(_reg_reduce_sum2_25, make_float2(sv[18], sv[19]));
                            _reg_reduce_sum2_25 = add_f32x2(_reg_reduce_sum2_25, make_float2(sv[20], sv[21]));
                            _reg_reduce_sum2_25 = add_f32x2(_reg_reduce_sum2_25, make_float2(sv[22], sv[23]));
                            _reg_reduce_sum2_25 = add_f32x2(_reg_reduce_sum2_25, make_float2(sv[24], sv[25]));
                            _reg_reduce_sum2_25 = add_f32x2(_reg_reduce_sum2_25, make_float2(sv[26], sv[27]));
                            _reg_reduce_sum2_25 = add_f32x2(_reg_reduce_sum2_25, make_float2(sv[28], sv[29]));
                            _reg_reduce_sum2_25 = add_f32x2(_reg_reduce_sum2_25, make_float2(sv[30], sv[31]));
                            float sv_sum = _reg_reduce_sum2_25.x + _reg_reduce_sum2_25.y;
                            lsum = sv_sum;
                            psum = psum * acc_scale + lsum;
                            row_max = new_max;
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
                                :: "r"((unsigned int)my_s_base + sm_stage * 128), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[0])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[1])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[2])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[3])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[4])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[5])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[6])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[7])));
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        }
                        __syncwarp();
                        if (lane == 0) {
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;"
                                :: "r"(p_full_addr + (sm_stage) * 8), "r"((uint32_t)(32)) : "memory");
                        }
                        sm_stage += 1;
                        if (sm_stage == 2) { sm_stage = 0; _phase_s_full ^= 1; }
                    }
                    float _shfl_xor_sync_26 = __shfl_xor_sync(0xFFFFFFFF, psum, 16);
                    float total = psum + _shfl_xor_sync_26;
                    mbarrier_wait(stats_empty_addr + (st_stage_s) * 8, _phase_stats_empty);
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
                    if (st_stage_s == 2) { st_stage_s = 0; _phase_stats_empty ^= 1; }
                } else {
                    asm volatile("fence.acquire.gpu;" ::: "memory");
                    int r_row = (128 + sm_tid) / 4;
                    int _min_27 = ((q_len - q_tile_s * 4) < (4) ? (q_len - q_tile_s * 4) : (4));
                    int live_rows_r = _min_27 * 16;
                    if (r_row < live_rows_r) {
                        int items_r = num_kv_heads * ((q_len + 4 - 1) / 4);
                        int stats_stride_r = items_r * 128;
                        int o_stride_r = items_r * 16384;
                        int stats_row_r = slot_tile_base_s * 128 + r_row;
                        int f_r = 64 >> block_end_s;
                        int d0_r = block_begin_s * (4 * f_r) + (128 + sm_tid) % 4 * f_r;
                        d0_r = block_begin_s * (4 * f_r) + (128 + sm_tid) % 4 * 16;
                        int o_row_r = slot_tile_base_s * 16384 + r_row * 4 + d0_r * 64;
                        int j_r = q_tile_s * 4 + r_row / 16;
                        int h_r = r_row % 16;
                        int q_head_r = kv_head_s * 16 + h_r;
                        int o_idx_r = ((batch_s * q_len + j_r) * num_q_heads + q_head_r) * 256 + d0_r;
                        int store_lse_r = 0;
                        if ((128 + sm_tid) % 4 == 0 && block_begin_s == 0) {
                            store_lse_r = 1;
                        }
                        int lse_idx_r = (batch_s * q_len + j_r) * num_q_heads + q_head_r;
                        int narrow_r = 0;
                        if (narrow_r != 0) {
                            int d0_n = block_begin_s * (4 * f_r) + (128 + sm_tid) % 4 * f_r;
                            int o_row_n = slot_tile_base_s * 16384 + r_row * 4 + d0_n * 64;
                            int o_idx_n = ((batch_s * q_len + j_r) * num_q_heads + q_head_r) * 256 + d0_n;
                            float acc_f[4];
                            float out4[4];
                            int n_pad_r = (n_chunks_s + 15) / 16 * 16;
                            #pragma unroll 1
                            for (int g_r = 0; g_r < 16 >> block_end_s; g_r++) {
                                #pragma unroll
                                for (int k0 = 0; k0 < 4; k0++) {
                                    acc_f[k0] = 0.0f;
                                }
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
                                    float m_new;
                                    asm("max.f32 %0, %1, %2;" : "=f"(m_new) : "f"(m_f), "f"(m_k));
                                    float a_k = approx_exp2((m_f - m_new) * softmax_scale_log2);
                                    float b_k = approx_exp2((m_k - m_new) * softmax_scale_log2);
                                    l_f = __fmaf_rn(l_k, b_k, l_f * a_k);
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
                                        acc_f[k] = __fmaf_rn(_vec_load_0[k], b_k, acc_f[k] * a_k);
                                    }
                                    m_f = m_new;
                                }
                                float _rcp_29 = approx_rcp(l_f);
                                float inv_f = ((l_f > 0.0f) ? _rcp_29 * output_scale : 0.0f);
                                #pragma unroll
                                for (int k4 = 0; k4 < 4; k4++) {
                                    out4[k4] = acc_f[k4] * inv_f;
                                }
                                unsigned int _pk[2];
                                #pragma unroll
                                for (int _lp = 0; _lp < 2; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out4[_lp*2 + 0], out4[_lp*2+1 + 0]));
                                    _pk[_lp] = *(uint32_t*)&_bf2;
                                }
                                reinterpret_cast<int2*>(O_ptr + (o_idx_n + g_r * 4))[0] = reinterpret_cast<int2*>(_pk)[0];
                                if (store_lse_r != 0) {
                                    if (g_r == 0) {
                                        float lse_r = -CUDART_INF_F;
                                        if (l_f > 0.0f) {
                                            float _log2_30;
                                            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_30) : "f"(l_f));
                                            lse_r = m_f * softmax_scale_log2 + _log2_30 - 0.8073549f;
                                        }
                                        *(reinterpret_cast<float*>(LSE_ptr + lse_idx_r) + (0)) = lse_r;
                                    }
                                }
                            }
                        } else {
                            int n_span_r = ((f_r / 16) > (1) ? (f_r / 16) : (1));
                            int active_r = ((f_r / 4) < (4) ? (f_r / 4) : (4));
                            if (active_r > (128 + sm_tid) % 4) {
                                #pragma unroll 1
                                for (int w_r = 0; w_r < n_span_r; w_r++) {
                                    int store_lse_w = ((w_r == 0) ? store_lse_r : 0);
                                    float acc_w[16];
                                    #pragma unroll
                                    for (int k0_1 = 0; k0_1 < 16; k0_1++) {
                                        acc_w[k0_1] = 0.0f;
                                    }
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
                                                    float4 _v4 = *reinterpret_cast<const float4*>(partial_o + o_row_r + (w_r * 64 * 64) + (g * 4 * 64) + (c_c_1 * o_stride_r) + 0);
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
                                            float m_new_1;
                                            asm("max.f32 %0, %1, %2;" : "=f"(m_new_1) : "f"(m_w), "f"(m_k_1));
                                            float a_k_1 = approx_exp2((m_w - m_new_1) * softmax_scale_log2);
                                            float b_k_1 = approx_exp2((m_k_1 - m_new_1) * softmax_scale_log2);
                                            l_w = __fmaf_rn(l_k_1, b_k_1, l_w * a_k_1);
                                            #pragma unroll
                                            for (int k2 = 0; k2 < 16; k2++) {
                                                acc_w[k2] = __fmaf_rn(o_b[cb2 * 16 + k2], b_k_1, acc_w[k2] * a_k_1);
                                            }
                                            m_w = m_new_1;
                                        }
                                    }
                                    float _rcp_32 = approx_rcp(l_w);
                                    float inv_w = ((l_w > 0.0f) ? _rcp_32 * output_scale : 0.0f);
                                    float out_w[4];
                                    #pragma unroll
                                    for (int g3 = 0; g3 < 4; g3++) {
                                        #pragma unroll
                                        for (int k3 = 0; k3 < 4; k3++) {
                                            out_w[k3] = acc_w[g3 * 4 + k3] * inv_w;
                                        }
                                        unsigned int _pk_1[2];
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 2; _lp++) {
                                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_w[_lp*2 + 0], out_w[_lp*2+1 + 0]));
                                            _pk_1[_lp] = *(uint32_t*)&_bf2;
                                        }
                                        reinterpret_cast<int2*>(O_ptr + (o_idx_r + w_r * 64 + g3 * 4))[0] = reinterpret_cast<int2*>(_pk_1)[0];
                                    }
                                    if (store_lse_w != 0) {
                                        float lse_w = -CUDART_INF_F;
                                        if (l_w > 0.0f) {
                                            float _log2_33;
                                            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_33) : "f"(l_w));
                                            lse_w = m_w * softmax_scale_log2 + _log2_33 - 0.8073549f;
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
                unsigned int valid_0 = work_token_words[base_0];
                unsigned int kind_0 = work_token_words[base_0 + 1];
                unsigned int batch_0 = work_token_words[base_0 + 2];
                unsigned int kv_head_0 = work_token_words[base_0 + 3];
                unsigned int block_begin_0 = work_token_words[base_0 + 4];
                unsigned int block_end_0 = work_token_words[base_0 + 5];
                unsigned int seqlen_0 = work_token_words[base_0 + 6];
                unsigned int n_chunks_0 = work_token_words[base_0 + 7];
                unsigned int slot_tile_base_0 = work_token_words[base_0 + 8];
                unsigned int counter_idx_0 = work_token_words[base_0 + 9];
                unsigned int chunk_0 = work_token_words[base_0 + 10];
                unsigned int phase_0 = work_token_words[base_0 + 11];
                unsigned int q_tile_0 = work_token_words[base_0 + 12];
                unsigned int plan_flags_0 = work_token_words[base_0 + 13];
                mbarrier_arrive(work_empty_addr + (work_stage_s) * 8);
                work_stage_s += 1;
                if (work_stage_s == 4) { work_stage_s = 0; _phase_work_full ^= 1; }
                valid_s = valid_0;
                kind_s = (int)kind_0;
                block_begin_s = (int)block_begin_0;
                block_end_s = (int)block_end_0;
                seqlen_s = (int)seqlen_0;
                batch_s = (int)batch_0;
                kv_head_s = (int)kv_head_0;
                n_chunks_s = (int)n_chunks_0;
                slot_tile_base_s = (int)slot_tile_base_0;
                phase_s = (int)phase_0;
                q_tile_s = (int)q_tile_0;
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
            int rows_live_c = ((warp_in_wg_c * 16 < 64) ? 1 : 0);
            int row_j_c = my_row_c / 16;
            float _rcp_0_1 = approx_rcp(softmax_scale_log2);
            float thr_raw_c = 8.0f * _rcp_0_1;
            float p_scale_log2_c = 0.8073549f;
            int items_per_chunk_c = num_kv_heads * ((q_len + 4 - 1) / 4);
            unsigned int sm_stage_c = 0;
            unsigned int xm_slot_c = 0;
            unsigned int st_stage_c = 0;
            unsigned int pv_base = 0;
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
                    int spec_blocks_c = (spec_seqlen_c + 128 - 1) / 128;
                    int spec_max_pg_c = (spec_seqlen_c + 64 - 1) / 64 - 1;
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
            unsigned int work_stage_c = 0;
            unsigned int _phase_work_full_1 = 0;
            mbarrier_wait(work_full_addr + (work_stage_c) * 8, _phase_work_full_1);
            unsigned int base_1 = work_stage_c * 16;
            unsigned int valid_1 = work_token_words[base_1];
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
            unsigned int plan_flags_1 = work_token_words[base_1 + 13];
            mbarrier_arrive(work_empty_addr + (work_stage_c) * 8);
            work_stage_c += 1;
            if (work_stage_c == 4) { work_stage_c = 0; _phase_work_full_1 ^= 1; }
            unsigned int valid_c = valid_1;
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
                    int _min_1_1 = ((q_len - q_tile_c * 4) < (4) ? (q_len - q_tile_c * 4) : (4));
                    int live_rows_wc = _min_1_1 * 16;
                    int k2x_c = 0;
                    if (live_rows_wc > 32) {
                        if ((int)block_end_1 - (int)block_begin_1 >= 1) {
                            k2x_c = 1;
                        }
                    }
                    mbarrier_wait(q_raw_full_addr, 0);
                    int q_hi_base_s_2 = smem_q_hi_img_addr;
                    int q_lo_base_s_2 = smem_q_lo_img_addr;
                    int q_key_bf_s_2 = smem_qbf_addr / 128 % 8;
                    int q_key_hi_s_2 = q_hi_base_s_2 / 128 % 8;
                    int q_key_lo_s_2 = q_lo_base_s_2 / 128 % 8;
                    unsigned int q_words_s_2[4];
                    float q_f32_s_2[8];
                    float q_res_s_2[8];
                    unsigned int q_packed_s_2[2];
                    unsigned int q_packed_lo_s_2[2];
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
                            int q_key_row_s_3 = (q_key_bf_s_2 + q_row_s_3) % 8;
                            int q_src_s_3 = smem_qbf_addr + (unsigned int)(q_kg_s_3 * 4096) + (unsigned int)(q_row_s_3 * 128) + (unsigned int)((q_c16a_s_3 ^ q_key_row_s_3) * 16);
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_2[(0) + 3]))
                                : "r"(q_src_s_3));
                            #pragma unroll
                            for (int qw_s_3 = 0; qw_s_3 < 4; qw_s_3++) {
                                unsigned int q_w_s_3 = q_words_s_2[qw_s_3];
                                unsigned int q_wbits_d_3 = q_w_s_3 << 16;
                                float q_lo_d_3 = 0.0f;
                                float q_lo_t_d_3 = 0.0f;
                                float q_hi_d_3 = 0.0f;
                                float q_hi_t_d_3 = 0.0f;
                                q_lo_d_3 = __uint_as_float(q_wbits_d_3);
                                q_wbits_d_3 = (q_w_s_3 & 65520) << 16;
                                q_lo_t_d_3 = __uint_as_float(q_wbits_d_3);
                                q_wbits_d_3 = q_w_s_3 >> 16 << 16;
                                q_hi_d_3 = __uint_as_float(q_wbits_d_3);
                                q_wbits_d_3 = (q_w_s_3 >> 16 & 65520) << 16;
                                q_hi_t_d_3 = __uint_as_float(q_wbits_d_3);
                                q_f32_s_2[2 * qw_s_3] = q_lo_t_d_3;
                                q_res_s_2[2 * qw_s_3] = q_lo_d_3 - q_lo_t_d_3;
                                q_f32_s_2[2 * qw_s_3 + 1] = q_hi_t_d_3;
                                q_res_s_2[2 * qw_s_3 + 1] = q_hi_d_3 - q_hi_t_d_3;
                            }
                        } else {
                            #pragma unroll
                            for (int qz_s_3 = 0; qz_s_3 < 8; qz_s_3++) {
                                q_f32_s_2[qz_s_3] = 0.0f;
                                q_res_s_2[qz_s_3] = 0.0f;
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
                        int q_row_off_s_3 = q_grp_s_3 * 8192 + q_drow_s_3 * 128 + q_half_s_3;
                        int q_hi_addr_s_3 = q_hi_base_s_2 + q_row_off_s_3 + (q_c16_s_3 ^ (q_key_hi_s_2 + q_drow_s_3) % 8) * 16;
                        int q_lo_addr_s_3 = q_lo_base_s_2 + q_row_off_s_3 + (q_c16_s_3 ^ (q_key_lo_s_2 + q_drow_s_3) % 8) * 16;
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_hi_addr_s_3), "r"(q_packed_s_2[0]));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_hi_addr_s_3 + 4), "r"(q_packed_s_2[1]));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_lo_addr_s_3), "r"(q_packed_lo_s_2[0]));
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_lo_addr_s_3 + 4), "r"(q_packed_lo_s_2[1]));
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    if (k2x_c != 0) {
                        int q_hi_base_s_2_1 = smem_q_hi_img_addr;
                        int q_lo_base_s_2_1 = smem_q_lo_img_addr;
                        int q_key_bf_s_2_1 = (smem_k_g0_addr + 32768) / 128 % 8;
                        int q_key_hi_s_2_1 = q_hi_base_s_2_1 / 128 % 8;
                        int q_key_lo_s_2_1 = q_lo_base_s_2_1 / 128 % 8;
                        unsigned int q_words_s_2_1[4];
                        float q_f32_s_2_1[8];
                        float q_res_s_2_1[8];
                        unsigned int q_packed_s_2_1[2];
                        unsigned int q_packed_lo_s_2_1[2];
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
                                int q_key_row_s_4 = (q_key_bf_s_2_1 + q_row_s_4) % 8;
                                int q_src_s_4 = smem_k_g0_addr + 32768 + (unsigned int)(q_kg_s_4 * 4096) + (unsigned int)(q_row_s_4 * 128) + (unsigned int)((q_c16a_s_4 ^ q_key_row_s_4) * 16);
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_2_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_2_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_2_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_2_1[(0) + 3]))
                                    : "r"(q_src_s_4));
                                #pragma unroll
                                for (int qw_s_4 = 0; qw_s_4 < 4; qw_s_4++) {
                                    unsigned int q_w_s_4 = q_words_s_2_1[qw_s_4];
                                    unsigned int q_wbits_d_4 = q_w_s_4 << 16;
                                    float q_lo_d_4 = 0.0f;
                                    float q_lo_t_d_4 = 0.0f;
                                    float q_hi_d_4 = 0.0f;
                                    float q_hi_t_d_4 = 0.0f;
                                    q_lo_d_4 = __uint_as_float(q_wbits_d_4);
                                    q_wbits_d_4 = (q_w_s_4 & 65520) << 16;
                                    q_lo_t_d_4 = __uint_as_float(q_wbits_d_4);
                                    q_wbits_d_4 = q_w_s_4 >> 16 << 16;
                                    q_hi_d_4 = __uint_as_float(q_wbits_d_4);
                                    q_wbits_d_4 = (q_w_s_4 >> 16 & 65520) << 16;
                                    q_hi_t_d_4 = __uint_as_float(q_wbits_d_4);
                                    q_f32_s_2_1[2 * qw_s_4] = q_lo_t_d_4;
                                    q_res_s_2_1[2 * qw_s_4] = q_lo_d_4 - q_lo_t_d_4;
                                    q_f32_s_2_1[2 * qw_s_4 + 1] = q_hi_t_d_4;
                                    q_res_s_2_1[2 * qw_s_4 + 1] = q_hi_d_4 - q_hi_t_d_4;
                                }
                            } else {
                                #pragma unroll
                                for (int qz_s_4 = 0; qz_s_4 < 8; qz_s_4++) {
                                    q_f32_s_2_1[qz_s_4] = 0.0f;
                                    q_res_s_2_1[qz_s_4] = 0.0f;
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
                                    : "=r"(_packed) : "f"(q_f32_s_2_1[0]), "f"(q_f32_s_2_1[1]),
                                                       "f"(q_f32_s_2_1[2]), "f"(q_f32_s_2_1[3]));
                                q_packed_s_2_1[0] = _packed;
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
                                    : "=r"(_packed) : "f"(q_f32_s_2_1[4]), "f"(q_f32_s_2_1[5]),
                                                       "f"(q_f32_s_2_1[6]), "f"(q_f32_s_2_1[7]));
                                q_packed_s_2_1[1] = _packed;
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
                                    : "=r"(_packed) : "f"(q_res_s_2_1[0]), "f"(q_res_s_2_1[1]),
                                                       "f"(q_res_s_2_1[2]), "f"(q_res_s_2_1[3]));
                                q_packed_lo_s_2_1[0] = _packed;
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
                                    : "=r"(_packed) : "f"(q_res_s_2_1[4]), "f"(q_res_s_2_1[5]),
                                                       "f"(q_res_s_2_1[6]), "f"(q_res_s_2_1[7]));
                                q_packed_lo_s_2_1[1] = _packed;
                            }
                            int q_row_off_s_4 = q_grp_s_4 * 8192 + q_drow_s_4 * 128 + q_half_s_4;
                            int q_hi_addr_s_4 = q_hi_base_s_2_1 + q_row_off_s_4 + (q_c16_s_4 ^ (q_key_hi_s_2_1 + q_drow_s_4) % 8) * 16;
                            int q_lo_addr_s_4 = q_lo_base_s_2_1 + q_row_off_s_4 + (q_c16_s_4 ^ (q_key_lo_s_2_1 + q_drow_s_4) % 8) * 16;
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_hi_addr_s_4), "r"(q_packed_s_2_1[0]));
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_hi_addr_s_4 + 4), "r"(q_packed_s_2_1[1]));
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_lo_addr_s_4), "r"(q_packed_lo_s_2_1[0]));
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_lo_addr_s_4 + 4), "r"(q_packed_lo_s_2_1[1]));
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 9, 256;" ::: "memory");
                    } else {
                        asm volatile("barrier.sync 9, 256;" ::: "memory");
                        if (live_rows_wc > 32) {
                            mbarrier_wait(q_raw_full_addr, 1);
                        }
                        int q_hi_base_s_3 = smem_q_hi_img_addr;
                        int q_lo_base_s_3 = smem_q_lo_img_addr;
                        int q_key_bf_s_3 = smem_qbf_addr / 128 % 8;
                        int q_key_hi_s_3 = q_hi_base_s_3 / 128 % 8;
                        int q_key_lo_s_3 = q_lo_base_s_3 / 128 % 8;
                        unsigned int q_words_s_3[4];
                        float q_f32_s_3[8];
                        float q_res_s_3[8];
                        unsigned int q_packed_s_3[2];
                        unsigned int q_packed_lo_s_3[2];
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
                                int q_key_row_s_5 = (q_key_bf_s_3 + q_row_s_5) % 8;
                                int q_src_s_5 = smem_qbf_addr + (unsigned int)(q_kg_s_5 * 4096) + (unsigned int)(q_row_s_5 * 128) + (unsigned int)((q_c16a_s_5 ^ q_key_row_s_5) * 16);
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_3[(0) + 3]))
                                    : "r"(q_src_s_5));
                                #pragma unroll
                                for (int qw_s_5 = 0; qw_s_5 < 4; qw_s_5++) {
                                    unsigned int q_w_s_5 = q_words_s_3[qw_s_5];
                                    unsigned int q_wbits_d_5 = q_w_s_5 << 16;
                                    float q_lo_d_5 = 0.0f;
                                    float q_lo_t_d_5 = 0.0f;
                                    float q_hi_d_5 = 0.0f;
                                    float q_hi_t_d_5 = 0.0f;
                                    q_lo_d_5 = __uint_as_float(q_wbits_d_5);
                                    q_wbits_d_5 = (q_w_s_5 & 65520) << 16;
                                    q_lo_t_d_5 = __uint_as_float(q_wbits_d_5);
                                    q_wbits_d_5 = q_w_s_5 >> 16 << 16;
                                    q_hi_d_5 = __uint_as_float(q_wbits_d_5);
                                    q_wbits_d_5 = (q_w_s_5 >> 16 & 65520) << 16;
                                    q_hi_t_d_5 = __uint_as_float(q_wbits_d_5);
                                    q_f32_s_3[2 * qw_s_5] = q_lo_t_d_5;
                                    q_res_s_3[2 * qw_s_5] = q_lo_d_5 - q_lo_t_d_5;
                                    q_f32_s_3[2 * qw_s_5 + 1] = q_hi_t_d_5;
                                    q_res_s_3[2 * qw_s_5 + 1] = q_hi_d_5 - q_hi_t_d_5;
                                }
                            } else {
                                #pragma unroll
                                for (int qz_s_5 = 0; qz_s_5 < 8; qz_s_5++) {
                                    q_f32_s_3[qz_s_5] = 0.0f;
                                    q_res_s_3[qz_s_5] = 0.0f;
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
                                    : "=r"(_packed) : "f"(q_f32_s_3[0]), "f"(q_f32_s_3[1]),
                                                       "f"(q_f32_s_3[2]), "f"(q_f32_s_3[3]));
                                q_packed_s_3[0] = _packed;
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
                                    : "=r"(_packed) : "f"(q_f32_s_3[4]), "f"(q_f32_s_3[5]),
                                                       "f"(q_f32_s_3[6]), "f"(q_f32_s_3[7]));
                                q_packed_s_3[1] = _packed;
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
                                    : "=r"(_packed) : "f"(q_res_s_3[0]), "f"(q_res_s_3[1]),
                                                       "f"(q_res_s_3[2]), "f"(q_res_s_3[3]));
                                q_packed_lo_s_3[0] = _packed;
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
                                    : "=r"(_packed) : "f"(q_res_s_3[4]), "f"(q_res_s_3[5]),
                                                       "f"(q_res_s_3[6]), "f"(q_res_s_3[7]));
                                q_packed_lo_s_3[1] = _packed;
                            }
                            int q_row_off_s_5 = q_grp_s_5 * 8192 + q_drow_s_5 * 128 + q_half_s_5;
                            int q_hi_addr_s_5 = q_hi_base_s_3 + q_row_off_s_5 + (q_c16_s_5 ^ (q_key_hi_s_3 + q_drow_s_5) % 8) * 16;
                            int q_lo_addr_s_5 = q_lo_base_s_3 + q_row_off_s_5 + (q_c16_s_5 ^ (q_key_lo_s_3 + q_drow_s_5) % 8) * 16;
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_hi_addr_s_5), "r"(q_packed_s_3[0]));
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_hi_addr_s_5 + 4), "r"(q_packed_s_3[1]));
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_lo_addr_s_5), "r"(q_packed_lo_s_3[0]));
                            asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_lo_addr_s_5 + 4), "r"(q_packed_lo_s_3[1]));
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 9, 256;" ::: "memory");
                    }
                }
            }
            unsigned int _phase_s_full_1 = 0;
            unsigned int _phase_stats_full = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_c = 0; _tile_iter_c < max_items; _tile_iter_c++) {
                if (valid_c == 0) {
                    break;
                }
                if (kind_c == 0) {
                    int cnt_c = block_end_c - block_begin_c;
                    int my_slot = slot_tile_base_c + chunk_c * items_per_chunk_c;
                    int _min_2 = ((q_len - q_tile_c * 4) < (4) ? (q_len - q_tile_c * 4) : (4));
                    int live_rows = _min_2 * 16;
                    int vis_j_c = ((q_tile_c * 4 + row_j_c) < (q_len - 1) ? (q_tile_c * 4 + row_j_c) : (q_len - 1));
                    int back_c = q_len - 1 - vis_j_c - phase_c;
                    int vis_col_c = seqlen_c;
                    if (back_c > 0) {
                        vis_col_c = seqlen_c - (back_c + (1 << cp_world_log2) - 1 >> cp_world_log2);
                    }
                    float row_max_c = -CUDART_INF_F;
                    float psum_c = 0.0f;
                    #pragma unroll 1
                    for (int n_1 = 0; n_1 < cnt_c; n_1++) {
                        mbarrier_wait(s_full_addr + (sm_stage_c) * 8, _phase_s_full_1);
                        int my_block_c = block_begin_c + cnt_c - 1 - n_1;
                        int blk_pos_c = my_block_c * 128 + tok_base_c;
                        float sv_c[32];
                        float lmax_c = -CUDART_INF_F;
                        float new_max_c = row_max_c;
                        float acc_scale_c = 1.0f;
                        int xm_off_c = (int)xm_slot_c * 128;
                        if (rows_live_c != 0) {
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                                : "=r"(*reinterpret_cast<uint32_t*>(&sv_c[0])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[1])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[2])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[3])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[4])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[5])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[6])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[7])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[8])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[9])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[10])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[11])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[12])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[13])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[14])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[15])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[16])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[17])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[18])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[19])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[20])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[21])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[22])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[23])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[24])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[25])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[26])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[27])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[28])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[29])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[30])), "=r"(*reinterpret_cast<uint32_t*>(&sv_c[31]))
                                : "r"((unsigned int)my_s_base_c + sm_stage_c * 128 + 32));
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            int n_vis_c = vis_col_c - blk_pos_c;
                            if (my_row_c >= 64) {
                                n_vis_c = 0;
                            }
                            if (n_vis_c < 32) {
                                int n_lo_c = ((n_vis_c) > (0) ? (n_vis_c) : (0));
                                uint32_t _thresh_mask_3;
                                {
                                    int _lim_4 = n_lo_c;
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
                                if (!(_thresh_mask_3 & (1u << 0))) sv_c[0] = -CUDART_INF_F;
                                if (!(_thresh_mask_3 & (1u << 1))) sv_c[1] = -CUDART_INF_F;
                                if (!(_thresh_mask_3 & (1u << 2))) sv_c[2] = -CUDART_INF_F;
                                if (!(_thresh_mask_3 & (1u << 3))) sv_c[3] = -CUDART_INF_F;
                                if (!(_thresh_mask_3 & (1u << 4))) sv_c[4] = -CUDART_INF_F;
                                if (!(_thresh_mask_3 & (1u << 5))) sv_c[5] = -CUDART_INF_F;
                                if (!(_thresh_mask_3 & (1u << 6))) sv_c[6] = -CUDART_INF_F;
                                if (!(_thresh_mask_3 & (1u << 7))) sv_c[7] = -CUDART_INF_F;
                                if (!(_thresh_mask_3 & (1u << 8))) sv_c[8] = -CUDART_INF_F;
                                if (!(_thresh_mask_3 & (1u << 9))) sv_c[9] = -CUDART_INF_F;
                                if (!(_thresh_mask_3 & (1u << 10))) sv_c[10] = -CUDART_INF_F;
                                if (!(_thresh_mask_3 & (1u << 11))) sv_c[11] = -CUDART_INF_F;
                                if (!(_thresh_mask_3 & (1u << 12))) sv_c[12] = -CUDART_INF_F;
                                if (!(_thresh_mask_3 & (1u << 13))) sv_c[13] = -CUDART_INF_F;
                                if (!(_thresh_mask_3 & (1u << 14))) sv_c[14] = -CUDART_INF_F;
                                if (!(_thresh_mask_3 & (1u << 15))) sv_c[15] = -CUDART_INF_F;
                                if (!(_thresh_mask_3 & (1u << 16))) sv_c[16] = -CUDART_INF_F;
                                if (!(_thresh_mask_3 & (1u << 17))) sv_c[17] = -CUDART_INF_F;
                                if (!(_thresh_mask_3 & (1u << 18))) sv_c[18] = -CUDART_INF_F;
                                if (!(_thresh_mask_3 & (1u << 19))) sv_c[19] = -CUDART_INF_F;
                                if (!(_thresh_mask_3 & (1u << 20))) sv_c[20] = -CUDART_INF_F;
                                if (!(_thresh_mask_3 & (1u << 21))) sv_c[21] = -CUDART_INF_F;
                                if (!(_thresh_mask_3 & (1u << 22))) sv_c[22] = -CUDART_INF_F;
                                if (!(_thresh_mask_3 & (1u << 23))) sv_c[23] = -CUDART_INF_F;
                                if (!(_thresh_mask_3 & (1u << 24))) sv_c[24] = -CUDART_INF_F;
                                if (!(_thresh_mask_3 & (1u << 25))) sv_c[25] = -CUDART_INF_F;
                                if (!(_thresh_mask_3 & (1u << 26))) sv_c[26] = -CUDART_INF_F;
                                if (!(_thresh_mask_3 & (1u << 27))) sv_c[27] = -CUDART_INF_F;
                                if (!(_thresh_mask_3 & (1u << 28))) sv_c[28] = -CUDART_INF_F;
                                if (!(_thresh_mask_3 & (1u << 29))) sv_c[29] = -CUDART_INF_F;
                                if (!(_thresh_mask_3 & (1u << 30))) sv_c[30] = -CUDART_INF_F;
                                if (!(_thresh_mask_3 & (1u << 31))) sv_c[31] = -CUDART_INF_F;
                            }
                            float2 _reg_reduce_max2_5 = {-CUDART_INF_F, -CUDART_INF_F};
                            _reg_reduce_max2_5.x = max_noftz(_reg_reduce_max2_5.x, max_noftz(sv_c[0], sv_c[1]));
                            _reg_reduce_max2_5.y = max_noftz(_reg_reduce_max2_5.y, max_noftz(sv_c[2], sv_c[3]));
                            _reg_reduce_max2_5.x = max_noftz(_reg_reduce_max2_5.x, max_noftz(sv_c[4], sv_c[5]));
                            _reg_reduce_max2_5.y = max_noftz(_reg_reduce_max2_5.y, max_noftz(sv_c[6], sv_c[7]));
                            _reg_reduce_max2_5.x = max_noftz(_reg_reduce_max2_5.x, max_noftz(sv_c[8], sv_c[9]));
                            _reg_reduce_max2_5.y = max_noftz(_reg_reduce_max2_5.y, max_noftz(sv_c[10], sv_c[11]));
                            _reg_reduce_max2_5.x = max_noftz(_reg_reduce_max2_5.x, max_noftz(sv_c[12], sv_c[13]));
                            _reg_reduce_max2_5.y = max_noftz(_reg_reduce_max2_5.y, max_noftz(sv_c[14], sv_c[15]));
                            _reg_reduce_max2_5.x = max_noftz(_reg_reduce_max2_5.x, max_noftz(sv_c[16], sv_c[17]));
                            _reg_reduce_max2_5.y = max_noftz(_reg_reduce_max2_5.y, max_noftz(sv_c[18], sv_c[19]));
                            _reg_reduce_max2_5.x = max_noftz(_reg_reduce_max2_5.x, max_noftz(sv_c[20], sv_c[21]));
                            _reg_reduce_max2_5.y = max_noftz(_reg_reduce_max2_5.y, max_noftz(sv_c[22], sv_c[23]));
                            _reg_reduce_max2_5.x = max_noftz(_reg_reduce_max2_5.x, max_noftz(sv_c[24], sv_c[25]));
                            _reg_reduce_max2_5.y = max_noftz(_reg_reduce_max2_5.y, max_noftz(sv_c[26], sv_c[27]));
                            _reg_reduce_max2_5.x = max_noftz(_reg_reduce_max2_5.x, max_noftz(sv_c[28], sv_c[29]));
                            _reg_reduce_max2_5.y = max_noftz(_reg_reduce_max2_5.y, max_noftz(sv_c[30], sv_c[31]));
                            float sv_c_max = row_max_reduce(_reg_reduce_max2_5);
                            lmax_c = sv_c_max;
                            float _shfl_xor_sync_6 = __shfl_xor_sync(0xFFFFFFFF, lmax_c, 16);
                            lmax_c = max_noftz(lmax_c, _shfl_xor_sync_6);
                            if (half_c == 0) {
                                smem_xmax[xm_off_c + 64 + my_row_c] = lmax_c;
                            }
                        }
                        asm volatile("barrier.sync 8, 256;" ::: "memory");
                        if (rows_live_c != 0) {
                            lmax_c = max_noftz(lmax_c, smem_xmax[xm_off_c + my_row_c]);
                            if (lmax_c > row_max_c + thr_raw_c) {
                                new_max_c = lmax_c;
                                if (row_max_c > -CUDART_INF_F) {
                                    float _exp2_7 = approx_exp2(softmax_scale_log2 * (row_max_c - new_max_c));
                                    acc_scale_c = _exp2_7;
                                }
                            }
                        }
                        xm_slot_c = xm_slot_c ^ 1;
                        if (rows_live_c != 0) {
                            float safe_max_c = ((new_max_c == -CUDART_INF_F) ? 0.0f : new_max_c);
                            const float2 _fma_b2_8 = {softmax_scale_log2, softmax_scale_log2};
                            const float2 _fma_c2_9 = {p_scale_log2_c - safe_max_c * softmax_scale_log2, p_scale_log2_c - safe_max_c * softmax_scale_log2};
                            float2 _fma_pair_10 = fma_f32x2(make_float2(sv_c[0], sv_c[1]), _fma_b2_8, _fma_c2_9);
                            sv_c[0] = _fma_pair_10.x;
                            sv_c[1] = _fma_pair_10.y;
                            float2 _fma_pair_11 = fma_f32x2(make_float2(sv_c[2], sv_c[3]), _fma_b2_8, _fma_c2_9);
                            sv_c[2] = _fma_pair_11.x;
                            sv_c[3] = _fma_pair_11.y;
                            float2 _fma_pair_12 = fma_f32x2(make_float2(sv_c[4], sv_c[5]), _fma_b2_8, _fma_c2_9);
                            sv_c[4] = _fma_pair_12.x;
                            sv_c[5] = _fma_pair_12.y;
                            float2 _fma_pair_13 = fma_f32x2(make_float2(sv_c[6], sv_c[7]), _fma_b2_8, _fma_c2_9);
                            sv_c[6] = _fma_pair_13.x;
                            sv_c[7] = _fma_pair_13.y;
                            float2 _fma_pair_14 = fma_f32x2(make_float2(sv_c[8], sv_c[9]), _fma_b2_8, _fma_c2_9);
                            sv_c[8] = _fma_pair_14.x;
                            sv_c[9] = _fma_pair_14.y;
                            float2 _fma_pair_15 = fma_f32x2(make_float2(sv_c[10], sv_c[11]), _fma_b2_8, _fma_c2_9);
                            sv_c[10] = _fma_pair_15.x;
                            sv_c[11] = _fma_pair_15.y;
                            float2 _fma_pair_16 = fma_f32x2(make_float2(sv_c[12], sv_c[13]), _fma_b2_8, _fma_c2_9);
                            sv_c[12] = _fma_pair_16.x;
                            sv_c[13] = _fma_pair_16.y;
                            float2 _fma_pair_17 = fma_f32x2(make_float2(sv_c[14], sv_c[15]), _fma_b2_8, _fma_c2_9);
                            sv_c[14] = _fma_pair_17.x;
                            sv_c[15] = _fma_pair_17.y;
                            float2 _fma_pair_18 = fma_f32x2(make_float2(sv_c[16], sv_c[17]), _fma_b2_8, _fma_c2_9);
                            sv_c[16] = _fma_pair_18.x;
                            sv_c[17] = _fma_pair_18.y;
                            float2 _fma_pair_19 = fma_f32x2(make_float2(sv_c[18], sv_c[19]), _fma_b2_8, _fma_c2_9);
                            sv_c[18] = _fma_pair_19.x;
                            sv_c[19] = _fma_pair_19.y;
                            float2 _fma_pair_20 = fma_f32x2(make_float2(sv_c[20], sv_c[21]), _fma_b2_8, _fma_c2_9);
                            sv_c[20] = _fma_pair_20.x;
                            sv_c[21] = _fma_pair_20.y;
                            float2 _fma_pair_21 = fma_f32x2(make_float2(sv_c[22], sv_c[23]), _fma_b2_8, _fma_c2_9);
                            sv_c[22] = _fma_pair_21.x;
                            sv_c[23] = _fma_pair_21.y;
                            float2 _fma_pair_22 = fma_f32x2(make_float2(sv_c[24], sv_c[25]), _fma_b2_8, _fma_c2_9);
                            sv_c[24] = _fma_pair_22.x;
                            sv_c[25] = _fma_pair_22.y;
                            float2 _fma_pair_23 = fma_f32x2(make_float2(sv_c[26], sv_c[27]), _fma_b2_8, _fma_c2_9);
                            sv_c[26] = _fma_pair_23.x;
                            sv_c[27] = _fma_pair_23.y;
                            float2 _fma_pair_24 = fma_f32x2(make_float2(sv_c[28], sv_c[29]), _fma_b2_8, _fma_c2_9);
                            sv_c[28] = _fma_pair_24.x;
                            sv_c[29] = _fma_pair_24.y;
                            float2 _fma_pair_25 = fma_f32x2(make_float2(sv_c[30], sv_c[31]), _fma_b2_8, _fma_c2_9);
                            sv_c[30] = _fma_pair_25.x;
                            sv_c[31] = _fma_pair_25.y;
                            #pragma unroll
                            for (int _le = 0; _le < 32; _le++) {
                                sv_c[_le] = approx_exp2(sv_c[_le]);
                            }
                            float2 _reg_reduce_sum2_26 = make_float2(0.0f, 0.0f);
                            _reg_reduce_sum2_26 = add_f32x2(_reg_reduce_sum2_26, make_float2(sv_c[0], sv_c[1]));
                            _reg_reduce_sum2_26 = add_f32x2(_reg_reduce_sum2_26, make_float2(sv_c[2], sv_c[3]));
                            _reg_reduce_sum2_26 = add_f32x2(_reg_reduce_sum2_26, make_float2(sv_c[4], sv_c[5]));
                            _reg_reduce_sum2_26 = add_f32x2(_reg_reduce_sum2_26, make_float2(sv_c[6], sv_c[7]));
                            _reg_reduce_sum2_26 = add_f32x2(_reg_reduce_sum2_26, make_float2(sv_c[8], sv_c[9]));
                            _reg_reduce_sum2_26 = add_f32x2(_reg_reduce_sum2_26, make_float2(sv_c[10], sv_c[11]));
                            _reg_reduce_sum2_26 = add_f32x2(_reg_reduce_sum2_26, make_float2(sv_c[12], sv_c[13]));
                            _reg_reduce_sum2_26 = add_f32x2(_reg_reduce_sum2_26, make_float2(sv_c[14], sv_c[15]));
                            _reg_reduce_sum2_26 = add_f32x2(_reg_reduce_sum2_26, make_float2(sv_c[16], sv_c[17]));
                            _reg_reduce_sum2_26 = add_f32x2(_reg_reduce_sum2_26, make_float2(sv_c[18], sv_c[19]));
                            _reg_reduce_sum2_26 = add_f32x2(_reg_reduce_sum2_26, make_float2(sv_c[20], sv_c[21]));
                            _reg_reduce_sum2_26 = add_f32x2(_reg_reduce_sum2_26, make_float2(sv_c[22], sv_c[23]));
                            _reg_reduce_sum2_26 = add_f32x2(_reg_reduce_sum2_26, make_float2(sv_c[24], sv_c[25]));
                            _reg_reduce_sum2_26 = add_f32x2(_reg_reduce_sum2_26, make_float2(sv_c[26], sv_c[27]));
                            _reg_reduce_sum2_26 = add_f32x2(_reg_reduce_sum2_26, make_float2(sv_c[28], sv_c[29]));
                            _reg_reduce_sum2_26 = add_f32x2(_reg_reduce_sum2_26, make_float2(sv_c[30], sv_c[31]));
                            float lsum_c = _reg_reduce_sum2_26.x + _reg_reduce_sum2_26.y;
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
                                :: "r"((unsigned int)my_s_base_c + sm_stage_c * 128 + 8), "r"(*reinterpret_cast<const uint32_t*>(&regs_pc[0])), "r"(*reinterpret_cast<const uint32_t*>(&regs_pc[1])), "r"(*reinterpret_cast<const uint32_t*>(&regs_pc[2])), "r"(*reinterpret_cast<const uint32_t*>(&regs_pc[3])), "r"(*reinterpret_cast<const uint32_t*>(&regs_pc[4])), "r"(*reinterpret_cast<const uint32_t*>(&regs_pc[5])), "r"(*reinterpret_cast<const uint32_t*>(&regs_pc[6])), "r"(*reinterpret_cast<const uint32_t*>(&regs_pc[7])));
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        }
                        int _warp_vote_27 = __any_sync(0xFFFFFFFF, acc_scale_c != 1.0f);
                        if (_warp_vote_27 != 0) {
                            if (n_1 > 0) {
                                unsigned int _stage_o_ready = (pv_base + (unsigned int)n_1 - 1) % 2;
                                unsigned int _parity_o_ready = (pv_base + (unsigned int)n_1 - 1) / 2 % 2;
                                mbarrier_wait(o_ready_addr + (_stage_o_ready) * 8, _parity_o_ready);
                                asm volatile("tcgen05.fence::after_thread_sync;");
                                float o_frag[64];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&o_frag[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[1])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[2])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[3])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[4])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[5])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[6])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[7])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[8])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[9])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[10])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[11])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[12])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[13])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[14])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[15])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[16])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[17])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[18])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[19])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[20])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[21])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[22])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[23])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[24])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[25])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[26])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[27])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[28])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[29])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[30])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[31]))
                                    : "r"(o_row_base));
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&o_frag[32])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[33])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[34])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[35])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[36])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[37])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[38])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[39])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[40])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[41])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[42])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[43])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[44])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[45])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[46])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[47])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[48])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[49])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[50])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[51])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[52])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[53])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[54])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[55])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[56])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[57])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[58])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[59])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[60])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[61])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[62])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag[63]))
                                    : "r"(o_row_base + 32));
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                const float2 _scale2_28 = {acc_scale_c, acc_scale_c};
                                #pragma unroll
                                for (int _ls = 0; _ls < 32; _ls++)
                                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_frag)[_ls], _scale2_28);
                                asm volatile(
                                    "tcgen05.st.sync.aligned.16x32bx2.x64.b32"
                                    " [%0], 64, {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64};"
                                    :: "r"(o_row_base), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[0])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[1])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[2])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[3])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[4])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[5])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[6])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[7])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[8])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[9])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[10])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[11])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[12])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[13])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[14])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[15])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[16])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[17])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[18])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[19])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[20])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[21])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[22])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[23])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[24])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[25])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[26])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[27])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[28])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[29])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[30])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[31])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[32])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[33])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[34])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[35])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[36])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[37])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[38])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[39])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[40])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[41])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[42])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[43])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[44])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[45])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[46])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[47])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[48])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[49])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[50])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[51])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[52])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[53])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[54])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[55])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[56])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[57])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[58])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[59])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[60])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[61])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[62])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag[63])));
                                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                                float o_frag_0[64];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[15])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[16])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[17])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[18])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[19])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[20])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[21])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[22])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[23])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[24])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[25])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[26])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[27])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[28])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[29])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[30])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[31]))
                                    : "r"(o_row_base_g1));
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[32])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[33])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[34])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[35])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[36])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[37])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[38])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[39])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[40])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[41])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[42])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[43])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[44])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[45])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[46])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[47])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[48])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[49])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[50])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[51])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[52])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[53])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[54])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[55])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[56])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[57])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[58])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[59])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[60])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[61])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[62])), "=r"(*reinterpret_cast<uint32_t*>(&o_frag_0[63]))
                                    : "r"(o_row_base_g1 + 32));
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                const float2 _scale2_29 = {acc_scale_c, acc_scale_c};
                                #pragma unroll
                                for (int _ls = 0; _ls < 32; _ls++)
                                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>(o_frag_0)[_ls], _scale2_29);
                                asm volatile(
                                    "tcgen05.st.sync.aligned.16x32bx2.x64.b32"
                                    " [%0], 64, {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64};"
                                    :: "r"(o_row_base_g1), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[0])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[1])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[2])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[3])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[4])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[5])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[6])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[7])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[8])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[9])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[10])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[11])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[12])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[13])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[14])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[15])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[16])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[17])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[18])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[19])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[20])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[21])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[22])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[23])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[24])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[25])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[26])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[27])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[28])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[29])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[30])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[31])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[32])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[33])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[34])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[35])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[36])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[37])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[38])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[39])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[40])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[41])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[42])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[43])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[44])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[45])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[46])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[47])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[48])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[49])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[50])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[51])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[52])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[53])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[54])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[55])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[56])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[57])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[58])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[59])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[60])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[61])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[62])), "r"(*reinterpret_cast<const uint32_t*>(&o_frag_0[63])));
                                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            }
                        }
                        if (n_1 >= 2) {
                            unsigned int _stage_o_ready_1 = (pv_base + (unsigned int)n_1 - 2) % 2;
                            unsigned int _parity_o_ready_1 = (pv_base + (unsigned int)n_1 - 2) / 2 % 2;
                            mbarrier_wait(o_ready_addr + (_stage_o_ready_1) * 8, _parity_o_ready_1);
                        }
                        __syncwarp();
                        if (lane == 0) {
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;"
                                :: "r"(p_full_addr + (sm_stage_c) * 8), "r"((uint32_t)(32)) : "memory");
                        }
                        sm_stage_c += 1;
                        if (sm_stage_c == 2) { sm_stage_c = 0; _phase_s_full_1 ^= 1; }
                    }
                    pv_base += (unsigned int)cnt_c;
                    if (cnt_c >= 2) {
                        unsigned int _stage_o_ready_2 = (pv_base - 2) % 2;
                        unsigned int _parity_o_ready_2 = (pv_base - 2) / 2 % 2;
                        mbarrier_wait(o_ready_addr + (_stage_o_ready_2) * 8, _parity_o_ready_2);
                    }
                    unsigned int _stage_o_ready_3 = (pv_base - 1) % 2;
                    unsigned int _parity_o_ready_3 = (pv_base - 1) / 2 % 2;
                    mbarrier_wait(o_ready_addr + (_stage_o_ready_3) * 8, _parity_o_ready_3);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    mbarrier_wait(stats_full_addr + (st_stage_c) * 8, _phase_stats_full);
                    int st_off = (int)st_stage_c * 64;
                    float _shfl_xor_sync_30 = __shfl_xor_sync(0xFFFFFFFF, psum_c, 16);
                    float total_c = psum_c + _shfl_xor_sync_30;
                    if (rows_live_c != 0) {
                        if (half_c == 0) {
                            smem_sum[st_off + my_row_c] = smem_sum[st_off + my_row_c] + total_c;
                        }
                    }
                    asm volatile("barrier.sync 11, 128;" ::: "memory");
                    int publish_split = ((n_chunks_c > 1) ? 1 : 0);
                    int d_g0 = half_c * 64;
                    if (publish_split == 0) {
                        float row_sum_c = 1.0f;
                        if (my_row_c < 64) {
                            row_sum_c = smem_sum[st_off + my_row_c];
                        }
                        float _rcp_31 = approx_rcp(row_sum_c);
                        float inv_c = ((row_sum_c > 0.0f) ? _rcp_31 : 0.0f);
                        int store_row_c = ((my_row_c < live_rows) ? 1 : 0);
                        int j_c = q_tile_c * 4 + my_row_c / 16;
                        int h_c = my_row_c % 16;
                        int q_head_c = kv_head_c * 16 + h_c;
                        int o_idx = ((batch_c * q_len + j_c) * num_q_heads + q_head_c) * 256 + d_g0;
                        float o_epi[64];
                        float o_epi_g1[64];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&o_epi[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[1])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[2])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[3])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[4])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[5])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[6])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[7])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[8])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[9])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[10])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[11])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[12])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[13])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[14])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[15])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[16])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[17])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[18])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[19])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[20])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[21])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[22])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[23])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[24])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[25])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[26])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[27])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[28])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[29])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[30])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[31]))
                            : "r"(o_row_base));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&o_epi[32])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[33])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[34])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[35])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[36])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[37])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[38])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[39])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[40])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[41])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[42])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[43])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[44])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[45])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[46])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[47])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[48])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[49])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[50])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[51])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[52])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[53])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[54])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[55])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[56])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[57])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[58])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[59])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[60])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[61])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[62])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi[63]))
                            : "r"(o_row_base + 32));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        if (row_sum_c <= 0.0f) {
                            #pragma unroll
                            for (int z_c = 0; z_c < 64; z_c++) {
                                o_epi[z_c] = 0.0f;
                            }
                        }
                        if (store_row_c != 0) {
                            #pragma unroll
                            for (int off = 0; off < 64; off += 8) {
                                const float2 _scale2_32 = {inv_c * output_scale, inv_c * output_scale};
                                #pragma unroll
                                for (int _ls = 0; _ls < 4; _ls++)
                                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>((o_epi + off))[_ls], _scale2_32);
                                unsigned int _pk_2[4];
                                #pragma unroll
                                for (int _lp = 0; _lp < 4; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((o_epi + off)[_lp*2 + 0], (o_epi + off)[_lp*2+1 + 0]));
                                    _pk_2[_lp] = *(uint32_t*)&_bf2;
                                }
                                reinterpret_cast<int4*>(O_ptr + (o_idx + off))[0] = reinterpret_cast<int4*>(_pk_2)[0];
                            }
                        }
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[1])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[2])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[3])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[4])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[5])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[6])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[7])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[8])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[9])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[10])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[11])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[12])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[13])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[14])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[15])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[16])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[17])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[18])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[19])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[20])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[21])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[22])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[23])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[24])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[25])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[26])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[27])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[28])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[29])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[30])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[31]))
                            : "r"(o_row_base_g1));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[32])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[33])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[34])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[35])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[36])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[37])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[38])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[39])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[40])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[41])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[42])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[43])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[44])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[45])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[46])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[47])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[48])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[49])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[50])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[51])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[52])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[53])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[54])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[55])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[56])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[57])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[58])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[59])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[60])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[61])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[62])), "=r"(*reinterpret_cast<uint32_t*>(&o_epi_g1[63]))
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
                            for (int z_c_1 = 0; z_c_1 < 64; z_c_1++) {
                                o_epi_g1[z_c_1] = 0.0f;
                            }
                        }
                        if (store_row_c != 0) {
                            #pragma unroll
                            for (int off_1 = 0; off_1 < 64; off_1 += 8) {
                                const float2 _scale2_33 = {inv_c * output_scale, inv_c * output_scale};
                                #pragma unroll
                                for (int _ls = 0; _ls < 4; _ls++)
                                    mul_f32x2_noftz_inplace(&reinterpret_cast<float2*>((o_epi_g1 + off_1))[_ls], _scale2_33);
                                unsigned int _pk_3[4];
                                #pragma unroll
                                for (int _lp = 0; _lp < 4; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((o_epi_g1 + off_1)[_lp*2 + 0], (o_epi_g1 + off_1)[_lp*2+1 + 0]));
                                    _pk_3[_lp] = *(uint32_t*)&_bf2;
                                }
                                reinterpret_cast<int4*>(O_ptr + (o_idx + 128 + off_1))[0] = reinterpret_cast<int4*>(_pk_3)[0];
                            }
                        }
                        if (store_row_c != 0) {
                            if (half_c == 0) {
                                float lse_c = -CUDART_INF_F;
                                if (row_sum_c > 0.0f) {
                                    float _log2_34;
                                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_34) : "f"(row_sum_c));
                                    lse_c = smem_max[st_off + my_row_c] * softmax_scale_log2 + _log2_34 - p_scale_log2_c;
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
                        float o_part[64];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&o_part[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[1])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[2])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[3])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[4])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[5])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[6])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[7])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[8])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[9])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[10])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[11])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[12])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[13])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[14])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[15])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[16])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[17])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[18])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[19])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[20])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[21])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[22])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[23])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[24])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[25])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[26])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[27])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[28])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[29])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[30])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[31]))
                            : "r"(o_row_base));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&o_part[32])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[33])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[34])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[35])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[36])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[37])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[38])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[39])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[40])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[41])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[42])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[43])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[44])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[45])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[46])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[47])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[48])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[49])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[50])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[51])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[52])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[53])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[54])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[55])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[56])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[57])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[58])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[59])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[60])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[61])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[62])), "=r"(*reinterpret_cast<uint32_t*>(&o_part[63]))
                            : "r"(o_row_base + 32));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        if (my_row_c < 64) {
                            #pragma unroll
                            for (int off_2 = 0; off_2 < 64; off_2 += 4) {
                                {
                                    float4 _v4 = make_float4(o_part[off_2 + 0], o_part[off_2 + 1], o_part[off_2 + 2], o_part[off_2 + 3]);
                                    *reinterpret_cast<float4*>((partial_o + (p_base + off_2 * 64)) + 0) = _v4;
                                }
                            }
                        }
                        float o_part_g1[64];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[0])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[1])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[2])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[3])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[4])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[5])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[6])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[7])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[8])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[9])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[10])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[11])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[12])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[13])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[14])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[15])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[16])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[17])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[18])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[19])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[20])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[21])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[22])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[23])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[24])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[25])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[26])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[27])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[28])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[29])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[30])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[31]))
                            : "r"(o_row_base_g1));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[32])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[33])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[34])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[35])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[36])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[37])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[38])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[39])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[40])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[41])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[42])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[43])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[44])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[45])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[46])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[47])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[48])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[49])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[50])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[51])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[52])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[53])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[54])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[55])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[56])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[57])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[58])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[59])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[60])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[61])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[62])), "=r"(*reinterpret_cast<uint32_t*>(&o_part_g1[63]))
                            : "r"(o_row_base_g1 + 32));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        __syncwarp();
                        if (lane == 0) {
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cta.b64 _, [%0], %1;"
                                :: "r"(o_empty_addr), "r"((uint32_t)(32)) : "memory");
                        }
                        if (my_row_c < 64) {
                            #pragma unroll
                            for (int off1 = 0; off1 < 64; off1 += 4) {
                                {
                                    float4 _v4 = make_float4(o_part_g1[off1 + 0], o_part_g1[off1 + 1], o_part_g1[off1 + 2], o_part_g1[off1 + 3]);
                                    *reinterpret_cast<float4*>((partial_o + (p_base + (128 + off1) * 64)) + 0) = _v4;
                                }
                            }
                        }
                        if (wg_tid_c < 64) {
                            *(reinterpret_cast<float*>(partial_stats + (my_slot * 128 + wg_tid_c)) + (0)) = smem_max[st_off + wg_tid_c];
                            *(reinterpret_cast<float*>(partial_stats + (my_slot * 128 + 64 + wg_tid_c)) + (0)) = smem_sum[st_off + wg_tid_c];
                        }
                        if (n_chunks_c == 2) {
                            asm volatile("barrier.sync 11, 128;" ::: "memory");
                            if (wg_tid_c == 0) {
                                asm volatile("fence.release.gpu;" ::: "memory");
                                unsigned int arrived_old;
                                asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                                    : "=r"(arrived_old) : "l"(&tile_counters[counter_idx_c * 4]), "r"(static_cast<uint32_t>(1)) : "memory");
                                smem_corr_flag[0] = arrived_old + 1;
                            }
                            asm volatile("barrier.sync 11, 128;" ::: "memory");
                            unsigned int arrived_c = smem_corr_flag[0];
                            if ((int)arrived_c == n_chunks_c) {
                                asm volatile("fence.acquire.gpu;" ::: "memory");
                                int other_slot = slot_tile_base_c + (1 - chunk_c) * items_per_chunk_c;
                                float w_s_c = 0.0f;
                                float w_o_c = 0.0f;
                                float lse_i = -CUDART_INF_F;
                                if (my_row_c < 64) {
                                    float m_o = partial_stats[other_slot * 128 + my_row_c];
                                    float l_o = partial_stats[other_slot * 128 + 64 + my_row_c];
                                    float m_s = smem_max[st_off + my_row_c];
                                    float l_s = smem_sum[st_off + my_row_c];
                                    float m_row_i;
                                    asm("max.f32 %0, %1, %2;" : "=f"(m_row_i) : "f"(m_s), "f"(m_o));
                                    float w_s = approx_exp2((m_s - m_row_i) * softmax_scale_log2);
                                    float w_o = approx_exp2((m_o - m_row_i) * softmax_scale_log2);
                                    float den_i = __fmaf_rn(w_s, l_s, w_o * l_o);
                                    if (chunk_c != 0) {
                                        den_i = __fmaf_rn(w_o, l_o, w_s * l_s);
                                    }
                                    float _rcp_35 = approx_rcp(den_i);
                                    float inv_i = ((den_i > 0.0f) ? _rcp_35 * output_scale : 0.0f);
                                    w_s_c = w_s * inv_i;
                                    w_o_c = w_o * inv_i;
                                    if (den_i > 0.0f) {
                                        float _log2_36;
                                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_36) : "f"(den_i));
                                        lse_i = m_row_i * softmax_scale_log2 + _log2_36 - p_scale_log2_c;
                                    }
                                }
                                int oth_base = other_slot * 16384 + my_row_c * 4 + d_g0 * 64;
                                if (my_row_c < live_rows) {
                                    int j_i = q_tile_c * 4 + my_row_c / 16;
                                    int h_i = my_row_c % 16;
                                    int q_head_i = kv_head_c * 16 + h_i;
                                    int o_idx_i = ((batch_c * q_len + j_i) * num_q_heads + q_head_i) * 256 + d_g0;
                                    float o_fold[64];
                                    #pragma unroll
                                    for (int c0 = 0; c0 < 64; c0 += 4) {
                                        float o_own[4];
                                        {
                                            float4 _v4 = *reinterpret_cast<const float4*>(partial_o + (p_base + c0 * 64) + 0);
                                            o_own[0 + 0] = _v4.x;
                                            o_own[0 + 1] = _v4.y;
                                            o_own[0 + 2] = _v4.z;
                                            o_own[0 + 3] = _v4.w;
                                        }
                                        float o_oth[4];
                                        {
                                            float4 _v4 = *reinterpret_cast<const float4*>(partial_o + (oth_base + c0 * 64) + 0);
                                            o_oth[0 + 0] = _v4.x;
                                            o_oth[0 + 1] = _v4.y;
                                            o_oth[0 + 2] = _v4.z;
                                            o_oth[0 + 3] = _v4.w;
                                        }
                                        #pragma unroll
                                        for (int c = 0; c < 4; c++) {
                                            float fma_own = __fmaf_rn(o_own[c], w_s_c, o_oth[c] * w_o_c);
                                            float fma_oth = __fmaf_rn(o_oth[c], w_o_c, o_own[c] * w_s_c);
                                            float o_det = ((chunk_c == 0) ? fma_own : fma_oth);
                                            o_fold[c0 + c] = o_det;
                                        }
                                    }
                                    #pragma unroll
                                    for (int off_f = 0; off_f < 64; off_f += 8) {
                                        unsigned int _pk_4[4];
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 4; _lp++) {
                                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((o_fold + off_f)[_lp*2 + 0], (o_fold + off_f)[_lp*2+1 + 0]));
                                            _pk_4[_lp] = *(uint32_t*)&_bf2;
                                        }
                                        reinterpret_cast<int4*>(O_ptr + (o_idx_i + off_f))[0] = reinterpret_cast<int4*>(_pk_4)[0];
                                    }
                                    #pragma unroll
                                    for (int c0_1 = 0; c0_1 < 64; c0_1 += 4) {
                                        float o_own_1[4];
                                        {
                                            float4 _v4 = *reinterpret_cast<const float4*>(partial_o + (p_base + 128 * 64 + c0_1 * 64) + 0);
                                            o_own_1[0 + 0] = _v4.x;
                                            o_own_1[0 + 1] = _v4.y;
                                            o_own_1[0 + 2] = _v4.z;
                                            o_own_1[0 + 3] = _v4.w;
                                        }
                                        float o_oth_1[4];
                                        {
                                            float4 _v4 = *reinterpret_cast<const float4*>(partial_o + (oth_base + 128 * 64 + c0_1 * 64) + 0);
                                            o_oth_1[0 + 0] = _v4.x;
                                            o_oth_1[0 + 1] = _v4.y;
                                            o_oth_1[0 + 2] = _v4.z;
                                            o_oth_1[0 + 3] = _v4.w;
                                        }
                                        #pragma unroll
                                        for (int c_1 = 0; c_1 < 4; c_1++) {
                                            float fma_own_1 = __fmaf_rn(o_own_1[c_1], w_s_c, o_oth_1[c_1] * w_o_c);
                                            float fma_oth_1 = __fmaf_rn(o_oth_1[c_1], w_o_c, o_own_1[c_1] * w_s_c);
                                            float o_det_1 = ((chunk_c == 0) ? fma_own_1 : fma_oth_1);
                                            o_fold[c0_1 + c_1] = o_det_1;
                                        }
                                    }
                                    #pragma unroll
                                    for (int off1_f = 0; off1_f < 64; off1_f += 8) {
                                        unsigned int _pk_5[4];
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 4; _lp++) {
                                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((o_fold + off1_f)[_lp*2 + 0], (o_fold + off1_f)[_lp*2+1 + 0]));
                                            _pk_5[_lp] = *(uint32_t*)&_bf2;
                                        }
                                        reinterpret_cast<int4*>(O_ptr + (o_idx_i + 128 + off1_f))[0] = reinterpret_cast<int4*>(_pk_5)[0];
                                    }
                                    if (half_c == 0) {
                                        *(reinterpret_cast<float*>(LSE_ptr + ((batch_c * q_len + j_i) * num_q_heads + q_head_i)) + (0)) = lse_i;
                                    }
                                }
                                asm volatile("barrier.sync 11, 128;" ::: "memory");
                                if (elect_sync()) {
                                    mbarrier_arrive(stats_empty_addr + (st_stage_c) * 8);
                                }
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
                                asm volatile("fence.release.gpu;" ::: "memory");
                                unsigned int arrived_old_1;
                                asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                                    : "=r"(arrived_old_1) : "l"(&tile_counters[counter_idx_c * 4]), "r"(static_cast<uint32_t>(1)) : "memory");
                            }
                        }
                    }
                    st_stage_c += 1;
                    if (st_stage_c == 2) { st_stage_c = 0; _phase_stats_full ^= 1; }
                } else {
                    asm volatile("fence.acquire.gpu;" ::: "memory");
                    int r_row_1 = wg_tid_c / 4;
                    int _min_41 = ((q_len - q_tile_c * 4) < (4) ? (q_len - q_tile_c * 4) : (4));
                    int live_rows_r_1 = _min_41 * 16;
                    if (r_row_1 < live_rows_r_1) {
                        int items_r_1 = num_kv_heads * ((q_len + 4 - 1) / 4);
                        int stats_stride_r_1 = items_r_1 * 128;
                        int o_stride_r_1 = items_r_1 * 16384;
                        int stats_row_r_1 = slot_tile_base_c * 128 + r_row_1;
                        int f_r_1 = 64 >> block_end_c;
                        int d0_r_1 = block_begin_c * (4 * f_r_1) + wg_tid_c % 4 * f_r_1;
                        d0_r_1 = block_begin_c * (4 * f_r_1) + wg_tid_c % 4 * 16;
                        int o_row_r_1 = slot_tile_base_c * 16384 + r_row_1 * 4 + d0_r_1 * 64;
                        int j_r_1 = q_tile_c * 4 + r_row_1 / 16;
                        int h_r_1 = r_row_1 % 16;
                        int q_head_r_1 = kv_head_c * 16 + h_r_1;
                        int o_idx_r_1 = ((batch_c * q_len + j_r_1) * num_q_heads + q_head_r_1) * 256 + d0_r_1;
                        int store_lse_r_1 = 0;
                        if (wg_tid_c % 4 == 0 && block_begin_c == 0) {
                            store_lse_r_1 = 1;
                        }
                        int lse_idx_r_1 = (batch_c * q_len + j_r_1) * num_q_heads + q_head_r_1;
                        int narrow_r_1 = 0;
                        if (narrow_r_1 != 0) {
                            int d0_n_1 = block_begin_c * (4 * f_r_1) + wg_tid_c % 4 * f_r_1;
                            int o_row_n_1 = slot_tile_base_c * 16384 + r_row_1 * 4 + d0_n_1 * 64;
                            int o_idx_n_1 = ((batch_c * q_len + j_r_1) * num_q_heads + q_head_r_1) * 256 + d0_n_1;
                            float acc_f_1[4];
                            float out4_1[4];
                            int n_pad_r_1 = (n_chunks_c + 15) / 16 * 16;
                            #pragma unroll 1
                            for (int g_r_1 = 0; g_r_1 < 16 >> block_end_c; g_r_1++) {
                                #pragma unroll
                                for (int k0_2 = 0; k0_2 < 4; k0_2++) {
                                    acc_f_1[k0_2] = 0.0f;
                                }
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
                                    float m_new_2;
                                    asm("max.f32 %0, %1, %2;" : "=f"(m_new_2) : "f"(m_f_1), "f"(m_k_2));
                                    float a_k_2 = approx_exp2((m_f_1 - m_new_2) * softmax_scale_log2);
                                    float b_k_2 = approx_exp2((m_k_2 - m_new_2) * softmax_scale_log2);
                                    l_f_1 = __fmaf_rn(l_k_2, b_k_2, l_f_1 * a_k_2);
                                    float _vec_load_0_1[4];
                                    {
                                        float4 _v4 = *reinterpret_cast<const float4*>(partial_o + (o_col_r_1 + c_c_2 * o_stride_r_1) + 0);
                                        _vec_load_0_1[0 + 0] = _v4.x;
                                        _vec_load_0_1[0 + 1] = _v4.y;
                                        _vec_load_0_1[0 + 2] = _v4.z;
                                        _vec_load_0_1[0 + 3] = _v4.w;
                                    }
                                    #pragma unroll
                                    for (int k_1 = 0; k_1 < 4; k_1++) {
                                        acc_f_1[k_1] = __fmaf_rn(_vec_load_0_1[k_1], b_k_2, acc_f_1[k_1] * a_k_2);
                                    }
                                    m_f_1 = m_new_2;
                                }
                                float _rcp_43 = approx_rcp(l_f_1);
                                float inv_f_1 = ((l_f_1 > 0.0f) ? _rcp_43 * output_scale : 0.0f);
                                #pragma unroll
                                for (int k4_1 = 0; k4_1 < 4; k4_1++) {
                                    out4_1[k4_1] = acc_f_1[k4_1] * inv_f_1;
                                }
                                unsigned int _pk_6[2];
                                #pragma unroll
                                for (int _lp = 0; _lp < 2; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out4_1[_lp*2 + 0], out4_1[_lp*2+1 + 0]));
                                    _pk_6[_lp] = *(uint32_t*)&_bf2;
                                }
                                reinterpret_cast<int2*>(O_ptr + (o_idx_n_1 + g_r_1 * 4))[0] = reinterpret_cast<int2*>(_pk_6)[0];
                                if (store_lse_r_1 != 0) {
                                    if (g_r_1 == 0) {
                                        float lse_r_1 = -CUDART_INF_F;
                                        if (l_f_1 > 0.0f) {
                                            float _log2_44;
                                            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_44) : "f"(l_f_1));
                                            lse_r_1 = m_f_1 * softmax_scale_log2 + _log2_44 - 0.8073549f;
                                        }
                                        *(reinterpret_cast<float*>(LSE_ptr + lse_idx_r_1) + (0)) = lse_r_1;
                                    }
                                }
                            }
                        } else {
                            int n_span_r_1 = ((f_r_1 / 16) > (1) ? (f_r_1 / 16) : (1));
                            int active_r_1 = ((f_r_1 / 4) < (4) ? (f_r_1 / 4) : (4));
                            if (active_r_1 > wg_tid_c % 4) {
                                #pragma unroll 1
                                for (int w_r_1 = 0; w_r_1 < n_span_r_1; w_r_1++) {
                                    int store_lse_w_1 = ((w_r_1 == 0) ? store_lse_r_1 : 0);
                                    float acc_w_1[16];
                                    #pragma unroll
                                    for (int k0_3 = 0; k0_3 < 16; k0_3++) {
                                        acc_w_1[k0_3] = 0.0f;
                                    }
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
                                                    float4 _v4 = *reinterpret_cast<const float4*>(partial_o + o_row_r_1 + (w_r_1 * 64 * 64) + (g_1 * 4 * 64) + (c_c_3 * o_stride_r_1) + 0);
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
                                            float m_new_3;
                                            asm("max.f32 %0, %1, %2;" : "=f"(m_new_3) : "f"(m_w_1), "f"(m_k_3));
                                            float a_k_3 = approx_exp2((m_w_1 - m_new_3) * softmax_scale_log2);
                                            float b_k_3 = approx_exp2((m_k_3 - m_new_3) * softmax_scale_log2);
                                            l_w_1 = __fmaf_rn(l_k_3, b_k_3, l_w_1 * a_k_3);
                                            #pragma unroll
                                            for (int k2_1 = 0; k2_1 < 16; k2_1++) {
                                                acc_w_1[k2_1] = __fmaf_rn(o_b_1[cb2_1 * 16 + k2_1], b_k_3, acc_w_1[k2_1] * a_k_3);
                                            }
                                            m_w_1 = m_new_3;
                                        }
                                    }
                                    float _rcp_46 = approx_rcp(l_w_1);
                                    float inv_w_1 = ((l_w_1 > 0.0f) ? _rcp_46 * output_scale : 0.0f);
                                    float out_w_1[4];
                                    #pragma unroll
                                    for (int g3_1 = 0; g3_1 < 4; g3_1++) {
                                        #pragma unroll
                                        for (int k3_1 = 0; k3_1 < 4; k3_1++) {
                                            out_w_1[k3_1] = acc_w_1[g3_1 * 4 + k3_1] * inv_w_1;
                                        }
                                        unsigned int _pk_7[2];
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 2; _lp++) {
                                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_w_1[_lp*2 + 0], out_w_1[_lp*2+1 + 0]));
                                            _pk_7[_lp] = *(uint32_t*)&_bf2;
                                        }
                                        reinterpret_cast<int2*>(O_ptr + (o_idx_r_1 + w_r_1 * 64 + g3_1 * 4))[0] = reinterpret_cast<int2*>(_pk_7)[0];
                                    }
                                    if (store_lse_w_1 != 0) {
                                        float lse_w_1 = -CUDART_INF_F;
                                        if (l_w_1 > 0.0f) {
                                            float _log2_47;
                                            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_47) : "f"(l_w_1));
                                            lse_w_1 = m_w_1 * softmax_scale_log2 + _log2_47 - 0.8073549f;
                                        }
                                        *(reinterpret_cast<float*>(LSE_ptr + lse_idx_r_1) + (0)) = lse_w_1;
                                    }
                                }
                            }
                        }
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_c) * 8, _phase_work_full_1);
                unsigned int base_1_1 = work_stage_c * 16;
                unsigned int valid_1_1 = work_token_words[base_1_1];
                unsigned int kind_1_1 = work_token_words[base_1_1 + 1];
                unsigned int batch_1_1 = work_token_words[base_1_1 + 2];
                unsigned int kv_head_1_1 = work_token_words[base_1_1 + 3];
                unsigned int block_begin_1_1 = work_token_words[base_1_1 + 4];
                unsigned int block_end_1_1 = work_token_words[base_1_1 + 5];
                unsigned int seqlen_1_1 = work_token_words[base_1_1 + 6];
                unsigned int n_chunks_1_1 = work_token_words[base_1_1 + 7];
                unsigned int slot_tile_base_1_1 = work_token_words[base_1_1 + 8];
                unsigned int counter_idx_1_1 = work_token_words[base_1_1 + 9];
                unsigned int chunk_1_1 = work_token_words[base_1_1 + 10];
                unsigned int phase_1_1 = work_token_words[base_1_1 + 11];
                unsigned int q_tile_1_1 = work_token_words[base_1_1 + 12];
                unsigned int plan_flags_1_1 = work_token_words[base_1_1 + 13];
                mbarrier_arrive(work_empty_addr + (work_stage_c) * 8);
                work_stage_c += 1;
                if (work_stage_c == 4) { work_stage_c = 0; _phase_work_full_1 ^= 1; }
                valid_c = valid_1_1;
                kind_c = (int)kind_1_1;
                batch_c = (int)batch_1_1;
                kv_head_c = (int)kv_head_1_1;
                block_begin_c = (int)block_begin_1_1;
                block_end_c = (int)block_end_1_1;
                seqlen_c = (int)seqlen_1_1;
                n_chunks_c = (int)n_chunks_1_1;
                slot_tile_base_c = (int)slot_tile_base_1_1;
                counter_idx_c = (int)counter_idx_1_1;
                chunk_c = (int)chunk_1_1;
                phase_c = (int)phase_1_1;
                q_tile_c = (int)q_tile_1_1;
            }
            mbarrier_arrive(tmem_dealloc_addr);
        }
    // ---- Role: mma_warp ----
    } else if (warp == 8) {
        { // mma_warp_main
            unsigned int work_stage_m = 0;
            unsigned int k_cons_stage = 0;
            unsigned int v_cons_stage = 0;
            unsigned int qk_e = 0;
            unsigned int pf_stage = 0;
            unsigned int pv_idx_m = 0;
            unsigned int _phase_work_full_2 = 0;
            mbarrier_wait(work_full_addr + (work_stage_m) * 8, _phase_work_full_2);
            unsigned int base_2 = work_stage_m * 16;
            unsigned int valid_2 = work_token_words[base_2];
            unsigned int kind_2 = work_token_words[base_2 + 1];
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
            unsigned int plan_flags_2 = work_token_words[base_2 + 13];
            mbarrier_arrive(work_empty_addr + (work_stage_m) * 8);
            work_stage_m += 1;
            if (work_stage_m == 4) { work_stage_m = 0; _phase_work_full_2 ^= 1; }
            unsigned int valid_m = valid_2;
            int kind_m = (int)kind_2;
            int block_begin_m = (int)block_begin_2;
            int block_end_m = (int)block_end_2;
            unsigned int _phase_q_full = 0;
            unsigned int _phase_k_full = 0;
            unsigned int _phase_o_empty_0 = 1;
            unsigned int _phase_v_full = 0;
            unsigned int _phase_p_full = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_m = 0; _tile_iter_m < max_items; _tile_iter_m++) {
                if (valid_m == 0) {
                    break;
                }
                if (kind_m == 0) {
                    int cnt_m = block_end_m - block_begin_m;
                    mbarrier_wait(q_full_addr, _phase_q_full);
                    mbarrier_wait(k_full_addr + (k_cons_stage) * 8, _phase_k_full);
                    int _mma_a_lo = make_warp_uniform((((smem_q_hi_g0_addr) >> 4) & 0x3FFF) + (0) * 512);
                    int _mma_b_lo = make_warp_uniform((((smem_k_g0_addr) >> 4) & 0x3FFF) + (k_cons_stage) * 1024);
                    {
                        uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo);
                        uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_s + ((int)(qk_e % 2) * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 69206032, 0);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_s + ((int)(qk_e % 2) * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 69206032, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_s + ((int)(qk_e % 2) * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 69206032, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_s + ((int)(qk_e % 2) * 128)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 69206032, 1);
                        }
                    }
                    int _mma_a_lo_0 = make_warp_uniform((((smem_q_lo_g0_addr) >> 4) & 0x3FFF) + (0) * 512);
                    int _mma_b_lo_0 = make_warp_uniform((((smem_k_g0_addr) >> 4) & 0x3FFF) + (k_cons_stage) * 1024);
                    {
                        uint64_t _mma_ss_a_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
                        uint64_t _mma_ss_b_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_s + ((int)(qk_e % 2) * 128)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 69206032, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_s + ((int)(qk_e % 2) * 128)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 69206032, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_s + ((int)(qk_e % 2) * 128)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 69206032, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_s + ((int)(qk_e % 2) * 128)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 69206032, 1);
                        }
                    }
                    int _mma_a_lo_1 = make_warp_uniform((((smem_q_hi_g1_addr) >> 4) & 0x3FFF) + (0) * 512);
                    int _mma_b_lo_1 = make_warp_uniform((((smem_k_g1_addr) >> 4) & 0x3FFF) + (k_cons_stage) * 1024);
                    {
                        uint64_t _mma_ss_a_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_1);
                        uint64_t _mma_ss_b_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_1);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_s + ((int)(qk_e % 2) * 128)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 69206032, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_s + ((int)(qk_e % 2) * 128)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 69206032, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_s + ((int)(qk_e % 2) * 128)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 69206032, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_s + ((int)(qk_e % 2) * 128)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 69206032, 1);
                        }
                    }
                    int _mma_a_lo_2 = make_warp_uniform((((smem_q_lo_g1_addr) >> 4) & 0x3FFF) + (0) * 512);
                    int _mma_b_lo_2 = make_warp_uniform((((smem_k_g1_addr) >> 4) & 0x3FFF) + (k_cons_stage) * 1024);
                    {
                        uint64_t _mma_ss_a_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_2);
                        uint64_t _mma_ss_b_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_2);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_s + ((int)(qk_e % 2) * 128)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 69206032, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_s + ((int)(qk_e % 2) * 128)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 69206032, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_s + ((int)(qk_e % 2) * 128)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 69206032, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_s + ((int)(qk_e % 2) * 128)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 69206032, 1);
                        }
                    }
                    elect_commit(s_full_addr + (qk_e % 2) * 8);
                    elect_commit(k_empty_addr + (k_cons_stage) * 8);
                    k_cons_stage += 1;
                    if (k_cons_stage == 3) { k_cons_stage = 0; _phase_k_full ^= 1; }
                    qk_e += 1;
                    if (cnt_m == 1) {
                        elect_commit(q_empty_addr);
                    }
                    mbarrier_wait(o_empty_addr, _phase_o_empty_0);
                    _phase_o_empty_0 ^= 1;
                    int first_pv = 1;
                    #pragma unroll 1
                    for (int n_2 = 0; n_2 < cnt_m; n_2++) {
                        int next_n = n_2 + 1;
                        if (next_n < cnt_m) {
                            mbarrier_wait(k_full_addr + (k_cons_stage) * 8, _phase_k_full);
                            int _mma_a_lo_3 = make_warp_uniform((((smem_q_hi_g0_addr) >> 4) & 0x3FFF) + (0) * 512);
                            int _mma_b_lo_3 = make_warp_uniform((((smem_k_g0_addr) >> 4) & 0x3FFF) + (k_cons_stage) * 1024);
                            {
                                uint64_t _mma_ss_a_desc_4 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_3);
                                uint64_t _mma_ss_b_desc_4 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_3);
                                if (elect_sync()) {
                                    tcgen05_mma_f8f6f4((tmem_s + ((int)(qk_e % 2) * 128)), _mma_ss_a_desc_4, _mma_ss_b_desc_4, 69206032, 0);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_4, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f8f6f4((tmem_s + ((int)(qk_e % 2) * 128)), _mma_ss_a_desc_4, _mma_ss_b_desc_4, 69206032, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_4, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f8f6f4((tmem_s + ((int)(qk_e % 2) * 128)), _mma_ss_a_desc_4, _mma_ss_b_desc_4, 69206032, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_4, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f8f6f4((tmem_s + ((int)(qk_e % 2) * 128)), _mma_ss_a_desc_4, _mma_ss_b_desc_4, 69206032, 1);
                                }
                            }
                            int _mma_a_lo_4 = make_warp_uniform((((smem_q_lo_g0_addr) >> 4) & 0x3FFF) + (0) * 512);
                            int _mma_b_lo_4 = make_warp_uniform((((smem_k_g0_addr) >> 4) & 0x3FFF) + (k_cons_stage) * 1024);
                            {
                                uint64_t _mma_ss_a_desc_5 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_4);
                                uint64_t _mma_ss_b_desc_5 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_4);
                                if (elect_sync()) {
                                    tcgen05_mma_f8f6f4((tmem_s + ((int)(qk_e % 2) * 128)), _mma_ss_a_desc_5, _mma_ss_b_desc_5, 69206032, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_5, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f8f6f4((tmem_s + ((int)(qk_e % 2) * 128)), _mma_ss_a_desc_5, _mma_ss_b_desc_5, 69206032, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_5, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f8f6f4((tmem_s + ((int)(qk_e % 2) * 128)), _mma_ss_a_desc_5, _mma_ss_b_desc_5, 69206032, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_5, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f8f6f4((tmem_s + ((int)(qk_e % 2) * 128)), _mma_ss_a_desc_5, _mma_ss_b_desc_5, 69206032, 1);
                                }
                            }
                            int _mma_a_lo_5 = make_warp_uniform((((smem_q_hi_g1_addr) >> 4) & 0x3FFF) + (0) * 512);
                            int _mma_b_lo_5 = make_warp_uniform((((smem_k_g1_addr) >> 4) & 0x3FFF) + (k_cons_stage) * 1024);
                            {
                                uint64_t _mma_ss_a_desc_6 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_5);
                                uint64_t _mma_ss_b_desc_6 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_5);
                                if (elect_sync()) {
                                    tcgen05_mma_f8f6f4((tmem_s + ((int)(qk_e % 2) * 128)), _mma_ss_a_desc_6, _mma_ss_b_desc_6, 69206032, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_6, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f8f6f4((tmem_s + ((int)(qk_e % 2) * 128)), _mma_ss_a_desc_6, _mma_ss_b_desc_6, 69206032, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_6, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f8f6f4((tmem_s + ((int)(qk_e % 2) * 128)), _mma_ss_a_desc_6, _mma_ss_b_desc_6, 69206032, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_6, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f8f6f4((tmem_s + ((int)(qk_e % 2) * 128)), _mma_ss_a_desc_6, _mma_ss_b_desc_6, 69206032, 1);
                                }
                            }
                            int _mma_a_lo_6 = make_warp_uniform((((smem_q_lo_g1_addr) >> 4) & 0x3FFF) + (0) * 512);
                            int _mma_b_lo_6 = make_warp_uniform((((smem_k_g1_addr) >> 4) & 0x3FFF) + (k_cons_stage) * 1024);
                            {
                                uint64_t _mma_ss_a_desc_7 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_6);
                                uint64_t _mma_ss_b_desc_7 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_6);
                                if (elect_sync()) {
                                    tcgen05_mma_f8f6f4((tmem_s + ((int)(qk_e % 2) * 128)), _mma_ss_a_desc_7, _mma_ss_b_desc_7, 69206032, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_7, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f8f6f4((tmem_s + ((int)(qk_e % 2) * 128)), _mma_ss_a_desc_7, _mma_ss_b_desc_7, 69206032, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_7, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f8f6f4((tmem_s + ((int)(qk_e % 2) * 128)), _mma_ss_a_desc_7, _mma_ss_b_desc_7, 69206032, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_7, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f8f6f4((tmem_s + ((int)(qk_e % 2) * 128)), _mma_ss_a_desc_7, _mma_ss_b_desc_7, 69206032, 1);
                                }
                            }
                            elect_commit(s_full_addr + (qk_e % 2) * 8);
                            elect_commit(k_empty_addr + (k_cons_stage) * 8);
                            k_cons_stage += 1;
                            if (k_cons_stage == 3) { k_cons_stage = 0; _phase_k_full ^= 1; }
                            qk_e += 1;
                            if (next_n + 1 == cnt_m) {
                                elect_commit(q_empty_addr);
                            }
                        }
                        mbarrier_wait(v_full_addr + (v_cons_stage) * 8, _phase_v_full);
                        mbarrier_wait(p_full_addr + (pf_stage) * 8, _phase_p_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int first_pv_flag = first_pv;
                        int _mma_b_lo_7 = make_warp_uniform(((((smem_v_g0_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_cons_stage) * 1024);
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
                    :: "r"(tmem_o_g0), "r"(_mma_b_lo_7), "r"(tmem_s + (int)pf_stage * 128), "r"(((first_pv_flag) ? 0 : 1)) : "memory");
                        int _mma_b_lo_8 = make_warp_uniform(((((smem_v_g1_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_cons_stage) * 1024);
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
                    :: "r"(tmem_o_g1), "r"(_mma_b_lo_8), "r"(tmem_s + (int)pf_stage * 128), "r"(((first_pv_flag) ? 0 : 1)) : "memory");
                        elect_commit(v_empty_addr + (v_cons_stage) * 8);
                        elect_commit(o_ready_addr + (pv_idx_m % 2) * 8);
                        pv_idx_m += 1;
                        v_cons_stage += 1;
                        if (v_cons_stage == 2) { v_cons_stage = 0; _phase_v_full ^= 1; }
                        pf_stage += 1;
                        if (pf_stage == 2) { pf_stage = 0; _phase_p_full ^= 1; }
                        first_pv = 0;
                    }
                    _phase_q_full ^= 1;
                }
                mbarrier_wait(work_full_addr + (work_stage_m) * 8, _phase_work_full_2);
                unsigned int base_2_1 = work_stage_m * 16;
                unsigned int valid_2_1 = work_token_words[base_2_1];
                unsigned int kind_2_1 = work_token_words[base_2_1 + 1];
                unsigned int batch_2_1 = work_token_words[base_2_1 + 2];
                unsigned int kv_head_2_1 = work_token_words[base_2_1 + 3];
                unsigned int block_begin_2_1 = work_token_words[base_2_1 + 4];
                unsigned int block_end_2_1 = work_token_words[base_2_1 + 5];
                unsigned int seqlen_2_1 = work_token_words[base_2_1 + 6];
                unsigned int n_chunks_2_1 = work_token_words[base_2_1 + 7];
                unsigned int slot_tile_base_2_1 = work_token_words[base_2_1 + 8];
                unsigned int counter_idx_2_1 = work_token_words[base_2_1 + 9];
                unsigned int chunk_2_1 = work_token_words[base_2_1 + 10];
                unsigned int phase_2_1 = work_token_words[base_2_1 + 11];
                unsigned int q_tile_2_1 = work_token_words[base_2_1 + 12];
                unsigned int plan_flags_2_1 = work_token_words[base_2_1 + 13];
                mbarrier_arrive(work_empty_addr + (work_stage_m) * 8);
                work_stage_m += 1;
                if (work_stage_m == 4) { work_stage_m = 0; _phase_work_full_2 ^= 1; }
                valid_m = valid_2_1;
                kind_m = (int)kind_2_1;
                block_begin_m = (int)block_begin_2_1;
                block_end_m = (int)block_end_2_1;
            }
            unsigned int _phase_tmem_dealloc_0 = 0;
            mbarrier_wait(tmem_dealloc_addr, _phase_tmem_dealloc_0);
            _phase_tmem_dealloc_0 ^= 1;
        }
    // ---- Role: load_pgoff ----
    } else if (warp == 9) {
        { // load_pgoff_main
            int pg_blk_p = lane >> 3;
            int pg_lane_p = lane & 7;
            int page_prod_stage = 0;
            unsigned int work_stage_p = 0;
            unsigned int _phase_work_full_3 = 0;
            mbarrier_wait(work_full_addr + (work_stage_p) * 8, _phase_work_full_3);
            unsigned int base_3 = work_stage_p * 16;
            unsigned int valid_3 = work_token_words[base_3];
            unsigned int kind_3 = work_token_words[base_3 + 1];
            unsigned int batch_3 = work_token_words[base_3 + 2];
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
            unsigned int plan_flags_3 = work_token_words[base_3 + 13];
            mbarrier_arrive(work_empty_addr + (work_stage_p) * 8);
            work_stage_p += 1;
            if (work_stage_p == 4) { work_stage_p = 0; _phase_work_full_3 ^= 1; }
            unsigned int valid_p = valid_3;
            int kind_p = (int)kind_3;
            int batch_idx_p = (int)batch_3;
            int kv_head_p = (int)kv_head_3;
            int block_begin_p = (int)block_begin_3;
            int block_end_p = (int)block_end_3;
            int seqlen_kv_p = (int)seqlen_3;
            int q_tile_p = (int)q_tile_3;
            unsigned int _phase_page_offsets_empty = 1;
            unsigned int _phase_q_empty = 1;
            unsigned int _phase_q_raw_full = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_p = 0; _tile_iter_p < max_items; _tile_iter_p++) {
                if (valid_p == 0) {
                    break;
                }
                if (kind_p == 0) {
                    int cta_n_blocks_p = block_end_p - block_begin_p;
                    int max_pg_p = (((seqlen_kv_p + 64 - 1) / 64 - 1) > (0) ? ((seqlen_kv_p + 64 - 1) / 64 - 1) : (0));
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
                                int st_u_p = page_prod_stage + gc_p;
                                int st_p = ((st_u_p >= 6) ? st_u_p - 6 : st_u_p);
                                mbarrier_wait(page_offsets_empty_addr + (((page_prod_stage + gc_p >= 6) ? page_prod_stage + gc_p - 6 : page_prod_stage + gc_p)) * 8, _phase_page_offsets_empty ^ (unsigned int)(page_prod_stage + gc_p >= 6));
                                if (pg_blk_p == gc_p) {
                                    smem_page_offsets[st_p * 8 + pg_lane_p] = page_id_p;
                                }
                                __syncwarp();
                                if (elect_sync()) {
                                    mbarrier_arrive(page_offsets_full_addr + (((page_prod_stage + gc_p >= 6) ? page_prod_stage + gc_p - 6 : page_prod_stage + gc_p)) * 8);
                                }
                            }
                        }
                        #pragma unroll
                        for (int gv_p = 0; gv_p < 4; gv_p++) {
                            if (g_cnt_p > gv_p) {
                                page_prod_stage += 1;
                                if (page_prod_stage == 6) { page_prod_stage = 0; _phase_page_offsets_empty ^= 1; }
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
                                mbarrier_wait(q_empty_addr, _phase_q_empty);
                                #pragma unroll
                                for (int qh_p = 0; qh_p < 2; qh_p++) {
                                    if (qh_p == 0) {
                                        mbarrier_wait(q_raw_full_addr, _phase_q_raw_full);
                                    }
                                    if (qh_p == 1) {
                                        if (live_rows_p > 32) {
                                            mbarrier_wait(q_raw_full_addr, _phase_q_raw_full);
                                        }
                                    }
                                    int q_hi_base_s_4 = smem_q_hi_img_addr + 0 * 8192;
                                    int q_lo_base_s_4 = smem_q_lo_img_addr + 0 * 8192;
                                    int q_key_bf_s_4 = (smem_qbf_addr + 0 * 16384) / 128 % 8;
                                    int q_key_hi_s_4 = q_hi_base_s_4 / 128 % 8;
                                    int q_key_lo_s_4 = q_lo_base_s_4 / 128 % 8;
                                    unsigned int q_words_s_4[4];
                                    float q_f32_s_4[8];
                                    float q_res_s_4[8];
                                    unsigned int q_packed_s_4[2];
                                    unsigned int q_packed_lo_s_4[2];
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
                                            int q_key_row_s_6 = (q_key_bf_s_4 + q_row_s_6) % 8;
                                            int q_src_s_6 = smem_qbf_addr + 0 * 16384 + (unsigned int)(q_kg_s_6 * 4096) + (unsigned int)(q_row_s_6 * 128) + (unsigned int)((q_c16a_s_6 ^ q_key_row_s_6) * 16);
                                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                                : "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_words_s_4[(0) + 3]))
                                                : "r"(q_src_s_6));
                                            #pragma unroll
                                            for (int qw_s_6 = 0; qw_s_6 < 4; qw_s_6++) {
                                                unsigned int q_w_s_6 = q_words_s_4[qw_s_6];
                                                unsigned int q_wbits_d_6 = q_w_s_6 << 16;
                                                float q_lo_d_6 = 0.0f;
                                                float q_lo_t_d_6 = 0.0f;
                                                float q_hi_d_6 = 0.0f;
                                                float q_hi_t_d_6 = 0.0f;
                                                q_lo_d_6 = __uint_as_float(q_wbits_d_6);
                                                q_wbits_d_6 = (q_w_s_6 & 65520) << 16;
                                                q_lo_t_d_6 = __uint_as_float(q_wbits_d_6);
                                                q_wbits_d_6 = q_w_s_6 >> 16 << 16;
                                                q_hi_d_6 = __uint_as_float(q_wbits_d_6);
                                                q_wbits_d_6 = (q_w_s_6 >> 16 & 65520) << 16;
                                                q_hi_t_d_6 = __uint_as_float(q_wbits_d_6);
                                                q_f32_s_4[2 * qw_s_6] = q_lo_t_d_6;
                                                q_res_s_4[2 * qw_s_6] = q_lo_d_6 - q_lo_t_d_6;
                                                q_f32_s_4[2 * qw_s_6 + 1] = q_hi_t_d_6;
                                                q_res_s_4[2 * qw_s_6 + 1] = q_hi_d_6 - q_hi_t_d_6;
                                            }
                                        } else {
                                            #pragma unroll
                                            for (int qz_s_6 = 0; qz_s_6 < 8; qz_s_6++) {
                                                q_f32_s_4[qz_s_6] = 0.0f;
                                                q_res_s_4[qz_s_6] = 0.0f;
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
                                                : "=r"(_packed) : "f"(q_f32_s_4[0]), "f"(q_f32_s_4[1]),
                                                                   "f"(q_f32_s_4[2]), "f"(q_f32_s_4[3]));
                                            q_packed_s_4[0] = _packed;
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
                                                : "=r"(_packed) : "f"(q_f32_s_4[4]), "f"(q_f32_s_4[5]),
                                                                   "f"(q_f32_s_4[6]), "f"(q_f32_s_4[7]));
                                            q_packed_s_4[1] = _packed;
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
                                                : "=r"(_packed) : "f"(q_res_s_4[0]), "f"(q_res_s_4[1]),
                                                                   "f"(q_res_s_4[2]), "f"(q_res_s_4[3]));
                                            q_packed_lo_s_4[0] = _packed;
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
                                                : "=r"(_packed) : "f"(q_res_s_4[4]), "f"(q_res_s_4[5]),
                                                                   "f"(q_res_s_4[6]), "f"(q_res_s_4[7]));
                                            q_packed_lo_s_4[1] = _packed;
                                        }
                                        int q_row_off_s_6 = q_grp_s_6 * 8192 + q_drow_s_6 * 128 + q_half_s_6;
                                        int q_hi_addr_s_6 = q_hi_base_s_4 + q_row_off_s_6 + (q_c16_s_6 ^ (q_key_hi_s_4 + q_drow_s_6) % 8) * 16;
                                        int q_lo_addr_s_6 = q_lo_base_s_4 + q_row_off_s_6 + (q_c16_s_6 ^ (q_key_lo_s_4 + q_drow_s_6) % 8) * 16;
                                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_hi_addr_s_6), "r"(q_packed_s_4[0]));
                                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_hi_addr_s_6 + 4), "r"(q_packed_s_4[1]));
                                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_lo_addr_s_6), "r"(q_packed_lo_s_4[0]));
                                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(q_lo_addr_s_6 + 4), "r"(q_packed_lo_s_4[1]));
                                    }
                                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                    __syncwarp();
                                    if (qh_p == 0) {
                                        if (lane == 0) {
                                            mbarrier_arrive(q_raw_empty_addr);
                                        }
                                        _phase_q_raw_full ^= 1;
                                    }
                                    if (qh_p == 1) {
                                        if (live_rows_p > 32) {
                                            if (lane == 0) {
                                                mbarrier_arrive(q_raw_empty_addr);
                                            }
                                            _phase_q_raw_full ^= 1;
                                        }
                                    }
                                }
                                if (lane == 0) {
                                    mbarrier_arrive(q_full_addr);
                                }
                            }
                            if (wide_done_p != 0) {
                                int k2x_p = 0;
                                if (live_rows_p > 32) {
                                    if (block_end_p - block_begin_p >= 1) {
                                        k2x_p = 1;
                                    }
                                }
                                _phase_q_raw_full ^= 1;
                                if (live_rows_p > 32) {
                                    if (k2x_p == 0) {
                                        _phase_q_raw_full ^= 1;
                                    }
                                }
                            }
                            _phase_q_empty ^= 1;
                        }
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_p) * 8, _phase_work_full_3);
                unsigned int base_3_1 = work_stage_p * 16;
                unsigned int valid_3_1 = work_token_words[base_3_1];
                unsigned int kind_3_1 = work_token_words[base_3_1 + 1];
                unsigned int batch_3_1 = work_token_words[base_3_1 + 2];
                unsigned int kv_head_3_1 = work_token_words[base_3_1 + 3];
                unsigned int block_begin_3_1 = work_token_words[base_3_1 + 4];
                unsigned int block_end_3_1 = work_token_words[base_3_1 + 5];
                unsigned int seqlen_3_1 = work_token_words[base_3_1 + 6];
                unsigned int n_chunks_3_1 = work_token_words[base_3_1 + 7];
                unsigned int slot_tile_base_3_1 = work_token_words[base_3_1 + 8];
                unsigned int counter_idx_3_1 = work_token_words[base_3_1 + 9];
                unsigned int chunk_3_1 = work_token_words[base_3_1 + 10];
                unsigned int phase_3_1 = work_token_words[base_3_1 + 11];
                unsigned int q_tile_3_1 = work_token_words[base_3_1 + 12];
                unsigned int plan_flags_3_1 = work_token_words[base_3_1 + 13];
                mbarrier_arrive(work_empty_addr + (work_stage_p) * 8);
                work_stage_p += 1;
                if (work_stage_p == 4) { work_stage_p = 0; _phase_work_full_3 ^= 1; }
                valid_p = valid_3_1;
                kind_p = (int)kind_3_1;
                batch_idx_p = (int)batch_3_1;
                kv_head_p = (int)kv_head_3_1;
                block_begin_p = (int)block_begin_3_1;
                block_end_p = (int)block_end_3_1;
                seqlen_kv_p = (int)seqlen_3_1;
                q_tile_p = (int)q_tile_3_1;
            }
        }
    // ---- Role: scheduler ----
    } else if (warp == 10) {
        { // scheduler_main
            int lane_1 = lane;
            int num_ctas = gridDim.x;
            int n_q_tiles = (q_len + 4 - 1) / 4;
            int items_per_chunk = num_kv_heads * n_q_tiles;
            int num_groups = (batch_size + 32 - 1) / 32;
            int cp_mask = (1 << cp_world_log2) - 1;
            int small_batch_u = 0;
            #pragma unroll
            for (int gb = 0; gb < 32; gb += 8) {
                if (gb * 32 < batch_size) {
                    if (small_batch_u == 0) {
                        int seq_pre[8];
                        #pragma unroll
                        for (int gp = 0; gp < 8; gp++) {
                            seq_pre[gp] = 0;
                            int bp = (gb + gp) * 32 + lane_1;
                            if (bp < batch_size) {
                                seq_pre[gp] = causal_seqlens_kv_global[bp];
                            }
                        }
                        #pragma unroll
                        for (int gs = 0; gs < 8; gs++) {
                            int bs = (gb + gs) * 32 + lane_1;
                            if (bs < batch_size) {
                                int last_pos = seq_pre[gs] + (q_len - 1) - cp_rank;
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
                    }
                }
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            __syncwarp();
            unsigned int total_pairs = 0;
            unsigned int p_max = 0;
            unsigned int p_min = 4294967295u;
            #pragma unroll 1
            for (int g1 = 0; g1 < num_groups; g1++) {
                int b1 = g1 * 32 + lane_1;
                unsigned int pairs1 = 0;
                unsigned int pairs1_min = 4294967295u;
                if (b1 < batch_size) {
                    int s1 = sched_seq_lens[b1];
                    int _max_0 = ((s1) > (1) ? (s1) : (1));
                    pairs1 = (unsigned int)((_max_0 + 255) / 256);
                    pairs1_min = pairs1;
                }
                unsigned int _warp_redux_1;
                asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_1) : "r"(pairs1));
                total_pairs += _warp_redux_1;
                unsigned int _warp_redux_3;
                asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_3) : "r"(pairs1));
                unsigned int _max_2 = ((p_max) > (_warp_redux_3) ? (p_max) : (_warp_redux_3));
                p_max = _max_2;
                unsigned int _warp_redux_5;
                asm volatile("redux.sync.min.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_5) : "r"(pairs1_min));
                unsigned int _min_4 = ((p_min) < (_warp_redux_5) ? (p_min) : (_warp_redux_5));
                p_min = _min_4;
            }
            unsigned int total_work = total_pairs * (unsigned int)items_per_chunk;
            unsigned int uniform_u = 0;
            if (p_max == p_min) {
                uniform_u = 1;
            }
            float ctas_rcp = approx_rcp((float)num_ctas);
            float items_rcp = approx_rcp((float)items_per_chunk);
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
            unsigned int heavy_flags_u = uniform_u;
            if (ideal_pairs >= 64) {
                heavy_flags_u = uniform_u + 2;
            }
            unsigned int balance_k = (ideal_pairs + 64 - 1) / 64;
            if (balance_k < 1) {
                balance_k = 1;
            }
            if (balance_k > 8) {
                balance_k = 8;
            }
            unsigned int chunk_divisor = balance_k * (unsigned int)num_ctas;
            float _rcp_6 = approx_rcp((float)chunk_divisor);
            unsigned int q_0 = (unsigned int)((float)total_work * _rcp_6);
            if (total_work < q_0 * chunk_divisor) {
                q_0 = q_0 - 1;
            }
            if (total_work >= (q_0 + 1) * chunk_divisor) {
                q_0 = q_0 + 1;
            }
            if (total_work > q_0 * chunk_divisor) {
                q_0 = q_0 + 1;
            }
            unsigned int chunk_pairs_u = q_0;
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
                float _rcp_7 = approx_rcp((float)whole_items);
                unsigned int q_1 = (unsigned int)((float)(unsigned int)num_ctas * _rcp_7);
                if (q_1 * (unsigned int)whole_items > (unsigned int)num_ctas) {
                    q_1 = q_1 - 1;
                }
                if ((q_1 + 1) * (unsigned int)whole_items <= (unsigned int)num_ctas) {
                    q_1 = q_1 + 1;
                }
                int n_even = (int)q_1;
                if (n_even > 1) {
                    if (p_max >= 8 * (p_max - p_min)) {
                        float _rcp_8 = approx_rcp((float)n_even);
                        unsigned int q_2 = (unsigned int)((float)p_max * _rcp_8);
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
            unsigned int q_3 = (unsigned int)((float)total_work * ctas_rcp);
            if (total_work < q_3 * (unsigned int)num_ctas) {
                q_3 = q_3 - 1;
            }
            if (total_work >= (q_3 + 1) * (unsigned int)num_ctas) {
                q_3 = q_3 + 1;
            }
            if (total_work > q_3 * (unsigned int)num_ctas) {
                q_3 = q_3 + 1;
            }
            unsigned int l_one = q_3;
            unsigned int sm_floor = ((unsigned int)num_ctas * 55 + 99) / 100;
            float ctas100_rcp = approx_rcp((float)(100 * num_ctas));
            unsigned int sw_den = 145;
            float sw_rcp = approx_rcp((float)sw_den);
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
                    float _rcp_9 = approx_rcp((float)denom_f);
                    unsigned int q_4 = (unsigned int)((float)total_work * _rcp_9);
                    if (total_work < q_4 * denom_f) {
                        q_4 = q_4 - 1;
                    }
                    if (total_work >= (q_4 + 1) * denom_f) {
                        q_4 = q_4 + 1;
                    }
                    if (total_work > q_4 * denom_f) {
                        q_4 = q_4 + 1;
                    }
                    unsigned int l_hi_f = q_4;
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
                unsigned int l_lane = l_one + (unsigned int)lane_1 * step_f;
                if (l_lane > l_hi) {
                    l_lane = l_hi;
                }
                float lane_rcp = approx_rcp((float)l_lane);
                unsigned int t_lane = 0;
                unsigned int sp_lane = 0;
                unsigned int nm_lane = 0;
                #pragma unroll 4
                for (int bf = 0; bf < batch_size; bf++) {
                    int sf = sched_seq_lens[bf];
                    int _max_10 = ((sf) > (1) ? (sf) : (1));
                    unsigned int pairs_f = (unsigned int)((_max_10 + 255) / 256);
                    unsigned int q_5 = (unsigned int)((float)pairs_f * lane_rcp);
                    if (pairs_f < q_5 * l_lane) {
                        q_5 = q_5 - 1;
                    }
                    if (pairs_f >= (q_5 + 1) * l_lane) {
                        q_5 = q_5 + 1;
                    }
                    if (pairs_f > q_5 * l_lane) {
                        q_5 = q_5 + 1;
                    }
                    unsigned int nb_f = q_5;
                    t_lane += nb_f;
                    if (dense_u == 1) {
                        if (nb_f > 2) {
                            sp_lane += 1;
                        }
                        unsigned int _max_11 = ((nm_lane) > (nb_f) ? (nm_lane) : (nb_f));
                        nm_lane = _max_11;
                    }
                }
                t_lane = t_lane * (unsigned int)items_per_chunk;
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
                if (nm_lane > 2) {
                    sh_f = sh_f + 1;
                }
                if (dense_u == 1) {
                    if (nm_lane > 2) {
                        t_lane += sp_lane * (unsigned int)items_per_chunk << sh_f;
                    }
                }
                unsigned int ok_key = 32;
                if (t_lane <= (unsigned int)num_ctas) {
                    ok_key = (unsigned int)lane_1;
                }
                unsigned int first_ok;
                asm volatile("redux.sync.min.u32 %0, %1, 0xffffffff;" : "=r"(first_ok) : "r"(ok_key));
                if (first_ok < 32) {
                    unsigned int _shfl_sync_12 = __shfl_sync(0xFFFFFFFF, l_lane, (int)first_ok);
                    l_fit = _shfl_sync_12;
                    fit_valid = 1;
                }
            }
            int cand_idx = lane_1 & 15;
            int req_parity = lane_1 >> 4;
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
                float _rcp_13 = approx_rcp((float)div_c);
                unsigned int q_6 = (unsigned int)((float)total_work * _rcp_13);
                if (total_work < q_6 * div_c) {
                    q_6 = q_6 - 1;
                }
                if (total_work >= (q_6 + 1) * div_c) {
                    q_6 = q_6 + 1;
                }
                if (total_work > q_6 * div_c) {
                    q_6 = q_6 + 1;
                }
                cand = q_6;
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
            float cand_rcp = approx_rcp((float)cand);
            unsigned int tickets_c = 0;
            unsigned int nmax_c = 0;
            unsigned int splits_c = 0;
            int half_batch = (batch_size + 1) / 2;
            #pragma unroll 4
            for (int hc = 0; hc < half_batch; hc++) {
                int bc = 2 * hc + req_parity;
                unsigned int pairs_c = 0;
                if (bc < batch_size) {
                    int sc = sched_seq_lens[bc];
                    int _max_14 = ((sc) > (1) ? (sc) : (1));
                    pairs_c = (unsigned int)((_max_14 + 255) / 256);
                }
                unsigned int q_7 = (unsigned int)((float)pairs_c * cand_rcp);
                if (pairs_c < q_7 * cand) {
                    q_7 = q_7 - 1;
                }
                if (pairs_c >= (q_7 + 1) * cand) {
                    q_7 = q_7 + 1;
                }
                if (pairs_c > q_7 * cand) {
                    q_7 = q_7 + 1;
                }
                unsigned int nb_c = q_7;
                tickets_c += nb_c;
                unsigned int _max_15 = ((nmax_c) > (nb_c) ? (nmax_c) : (nb_c));
                nmax_c = _max_15;
                if (nb_c > 2) {
                    splits_c += 1;
                }
            }
            unsigned int _shfl_xor_sync_16 = __shfl_xor_sync(0xFFFFFFFF, tickets_c, 16);
            tickets_c += _shfl_xor_sync_16;
            unsigned int _shfl_xor_sync_18 = __shfl_xor_sync(0xFFFFFFFF, nmax_c, 16);
            unsigned int _max_17 = ((nmax_c) > (_shfl_xor_sync_18) ? (nmax_c) : (_shfl_xor_sync_18));
            nmax_c = _max_17;
            unsigned int _shfl_xor_sync_19 = __shfl_xor_sync(0xFFFFFFFF, splits_c, 16);
            splits_c += _shfl_xor_sync_19;
            tickets_c = tickets_c * (unsigned int)items_per_chunk;
            unsigned int q_8 = (unsigned int)((float)tickets_c * ctas_rcp);
            if (tickets_c < q_8 * (unsigned int)num_ctas) {
                q_8 = q_8 - 1;
            }
            if (tickets_c >= (q_8 + 1) * (unsigned int)num_ctas) {
                q_8 = q_8 + 1;
            }
            if (tickets_c > q_8 * (unsigned int)num_ctas) {
                q_8 = q_8 + 1;
            }
            unsigned int waves_c = q_8;
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
                sh_c = sh_c + 1;
                unsigned int one_c = 1;
                tail_c = 11 + (nmax_c + (one_c << sh_c) - 1 >> sh_c);
                if (dense_u == 1) {
                    tail_c = 6 + (4 * (nmax_c + (one_c << sh_c) - 1 >> sh_c) >> 1);
                }
                red_c = splits_c * (unsigned int)items_per_chunk << sh_c;
            }
            unsigned int a_last_c = tickets_c - (waves_c - 1) * (unsigned int)num_ctas;
            unsigned int eff_c = ((a_last_c) > (sm_floor) ? (a_last_c) : (sm_floor));
            unsigned int late_c = 0;
            if (dense_u == 1) {
                if (waves_c == 1) {
                    unsigned int q_9 = (unsigned int)((float)(100 * (unsigned int)num_ctas + 45 * tickets_c) * sw_rcp);
                    if (q_9 * sw_den > 100 * (unsigned int)num_ctas + 45 * tickets_c) {
                        q_9 = q_9 - 1;
                    }
                    if ((q_9 + 1) * sw_den <= 100 * (unsigned int)num_ctas + 45 * tickets_c) {
                        q_9 = q_9 + 1;
                    }
                    eff_c = q_9;
                }
                if (tickets_c + red_c > (unsigned int)num_ctas) {
                    late_c = 0;
                }
            }
            unsigned int cost_c = 4 * (waves_c - 1) * (cand + 20) + tail_c + late_c;
            unsigned int q_10 = (unsigned int)((float)(4 * cand * eff_c) * ctas_rcp);
            if (q_10 * (unsigned int)num_ctas > 4 * cand * eff_c) {
                q_10 = q_10 - 1;
            }
            if ((q_10 + 1) * (unsigned int)num_ctas <= 4 * cand * eff_c) {
                q_10 = q_10 + 1;
            }
            cost_c += q_10 + 80;
            unsigned int q_11 = (unsigned int)((float)(4 * cand * 5 * a_last_c) * ctas100_rcp);
            if (q_11 * (100 * (unsigned int)num_ctas) > 4 * cand * 5 * a_last_c) {
                q_11 = q_11 - 1;
            }
            if ((q_11 + 1) * (100 * (unsigned int)num_ctas) <= 4 * cand * 5 * a_last_c) {
                q_11 = q_11 + 1;
            }
            cost_c += q_11;
            unsigned int cost_key = 4294967295u;
            if (cand_valid == 1) {
                if (req_parity == 0) {
                    cost_key = cost_c * 32 + (unsigned int)lane_1;
                }
            }
            unsigned int best_key;
            asm volatile("redux.sync.min.u32 %0, %1, 0xffffffff;" : "=r"(best_key) : "r"(cost_key));
            int best_lane = (int)(best_key & 31);
            unsigned int _shfl_sync_20 = __shfl_sync(0xFFFFFFFF, cand, best_lane);
            chunk_pairs_u = _shfl_sync_20;
            int chunk_pairs = (int)chunk_pairs_u;
            float chunk_rcp = approx_rcp((float)chunk_pairs_u);
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
                unsigned int q_12 = (unsigned int)((float)p_max * chunk_rcp);
                if (p_max < q_12 * chunk_pairs_u) {
                    q_12 = q_12 - 1;
                }
                if (p_max >= (q_12 + 1) * chunk_pairs_u) {
                    q_12 = q_12 + 1;
                }
                if (p_max > q_12 * chunk_pairs_u) {
                    q_12 = q_12 + 1;
                }
                n_u = q_12;
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
                    int b2 = g2 * 32 + lane_1;
                    unsigned int full2 = 0;
                    unsigned int n2 = 0;
                    unsigned int pack2 = 0;
                    unsigned int split2 = 0;
                    unsigned int nsplit2 = 0;
                    if (b2 < batch_size) {
                        int s2 = sched_seq_lens[b2];
                        int _max_21 = ((s2) > (1) ? (s2) : (1));
                        unsigned int pairs2 = (unsigned int)((_max_21 + 255) / 256);
                        unsigned int q_13 = (unsigned int)((float)pairs2 * chunk_rcp);
                        if (pairs2 < q_13 * chunk_pairs_u) {
                            q_13 = q_13 - 1;
                        }
                        if (pairs2 >= (q_13 + 1) * chunk_pairs_u) {
                            q_13 = q_13 + 1;
                        }
                        if (pairs2 > q_13 * chunk_pairs_u) {
                            q_13 = q_13 + 1;
                        }
                        n2 = q_13;
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
                    unsigned int _warp_redux_22;
                    asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_22) : "r"(full2));
                    n_full_chunks += _warp_redux_22;
                    unsigned int _warp_redux_23;
                    asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_23) : "r"(n2));
                    n_chunks_total += _warp_redux_23;
                    unsigned int _warp_redux_24;
                    asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_24) : "r"(split2));
                    n_split_requests += _warp_redux_24;
                    unsigned int _warp_redux_26;
                    asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_26) : "r"(nsplit2));
                    unsigned int _max_25 = ((nmax_split) > (_warp_redux_26) ? (nmax_split) : (_warp_redux_26));
                    nmax_split = _max_25;
                    unsigned int pack_group;
                    asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(pack_group) : "r"(pack2));
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
            float full_rcp_u = approx_rcp((float)full_den_u);
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
            if (nmax_split > 2) {
                reduce_shift = reduce_shift + 1;
            }
            unsigned int red_base_u = n_split_requests * (unsigned int)items_per_chunk;
            unsigned int reduce_items = red_base_u << reduce_shift;
            unsigned int total_items = chunk_items + reduce_items;
            if (blockIdx.x == 0) {
                if (lane_1 == 0) {
                    int plan_off = num_ctas * 2048;
                    *(reinterpret_cast<float*>(partial_stats + plan_off) + (0)) = (float)chunk_pairs_u;
                    *(reinterpret_cast<float*>(partial_stats + plan_off + 1) + (0)) = (float)total_items;
                }
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
            unsigned int _phase_work_empty = 1;
            #pragma unroll 1
            for (unsigned int _claim = 0; _claim < max_items + 1; _claim++) {
                mbarrier_wait(work_empty_addr + (work_stage_sched) * 8, _phase_work_empty);
                unsigned int ticket_lane0 = blockIdx.x;
                ticket_lane0 = (unsigned int)(num_ctas - 1 - blockIdx.x);
                if (_claim != 0) {
                    if (total_items <= (unsigned int)num_ctas) {
                        ticket_lane0 = (unsigned int)num_ctas;
                    } else {
                        int _parity_claim_gate = (int)((unsigned int)(int)_claim % 2) ^ 1;
                        mbarrier_wait_hint(claim_gate_addr, _parity_claim_gate, 1000);
                        if (lane_1 == 0) {
                            unsigned int _atomic_fetch_add_27 = atomicAdd(&queue_counters[0], 1);
                            ticket_lane0 = _atomic_fetch_add_27 + (unsigned int)num_ctas;
                        }
                    }
                }
                unsigned int ticket = __shfl_sync(0xFFFFFFFF, ticket_lane0, 0);
                unsigned int token_base = work_stage_sched * 16;
                unsigned int valid_tok = ((ticket < total_items) ? 1 : 0);
                int counter_idx_r = 0;
                unsigned int dec_tok = valid_tok;
                unsigned int dec_ticket = ticket;
                if (valid_tok == 0) {
                    if (_claim == 0) {
                        if (chunk_items > 0) {
                            dec_tok = 1;
                            dec_ticket = 0;
                        }
                    }
                }
                if (dec_tok != 0) {
                    if (dec_ticket >= chunk_items) {
                        unsigned int r_red = dec_ticket - chunk_items;
                        unsigned int rt_idx = r_red >> reduce_shift;
                        unsigned int rq_slice = r_red - (rt_idx << reduce_shift);
                        int sel_batch_r = 0;
                        int sel_n_r = 0;
                        int item_r = 0;
                        int slot_tile_base_r = 0;
                        if (uniform_u == 1) {
                            unsigned int q_14 = (unsigned int)((float)rt_idx * items_rcp);
                            if (q_14 * (unsigned int)items_per_chunk > rt_idx) {
                                q_14 = q_14 - 1;
                            }
                            if ((q_14 + 1) * (unsigned int)items_per_chunk <= rt_idx) {
                                q_14 = q_14 + 1;
                            }
                            sel_batch_r = (int)q_14;
                            item_r = (int)(rt_idx - (unsigned int)sel_batch_r * (unsigned int)items_per_chunk);
                            sel_n_r = (int)n_u;
                            slot_tile_base_r = sel_batch_r * (int)n_u * items_per_chunk + item_r;
                            counter_idx_r = sel_batch_r * items_per_chunk + item_r;
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
                                b_r = _adv_r * 32 + lane_1;
                                n_r = 0;
                                mine_r = 0;
                                si_r = 0;
                                st_r = 0;
                                if (b_r < batch_size) {
                                    int s_r = sched_seq_lens[b_r];
                                    int _max_28 = ((s_r) > (1) ? (s_r) : (1));
                                    int pairs_r = (_max_28 + 255) / 256;
                                    unsigned int q_15 = (unsigned int)((float)(unsigned int)pairs_r * chunk_rcp);
                                    if ((unsigned int)pairs_r < q_15 * chunk_pairs_u) {
                                        q_15 = q_15 - 1;
                                    }
                                    if ((unsigned int)pairs_r >= (q_15 + 1) * chunk_pairs_u) {
                                        q_15 = q_15 + 1;
                                    }
                                    if ((unsigned int)pairs_r > q_15 * chunk_pairs_u) {
                                        q_15 = q_15 + 1;
                                    }
                                    n_r = (int)q_15;
                                    if (n_r > 1) {
                                        si_r = (unsigned int)n_r;
                                        st_r = 1;
                                    }
                                    if (n_r > 2) {
                                        mine_r = (unsigned int)items_per_chunk;
                                    }
                                }
                                uint32_t _warp_scan_sum_29 = mine_r;
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_29) : "r"(1));
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_29) : "r"(2));
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_29) : "r"(4));
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_29) : "r"(8));
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_29) : "r"(16));
                                incl_r = _warp_scan_sum_29;
                                uint32_t _warp_scan_sum_30 = si_r;
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_30) : "r"(1));
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_30) : "r"(2));
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_30) : "r"(4));
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_30) : "r"(8));
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_30) : "r"(16));
                                incl_si_r = _warp_scan_sum_30;
                                uint32_t _warp_scan_sum_31 = st_r;
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_31) : "r"(1));
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_31) : "r"(2));
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_31) : "r"(4));
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_31) : "r"(8));
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_31) : "r"(16));
                                incl_st_r = _warp_scan_sum_31;
                                unsigned int group_total_r = __shfl_sync(0xFFFFFFFF, incl_r, 31);
                                if (rt_idx < rt_before + group_total_r) {
                                    break;
                                }
                                rt_before = rt_before + group_total_r;
                                unsigned int _shfl_sync_32 = __shfl_sync(0xFFFFFFFF, incl_si_r, 31);
                                si_before_r = si_before_r + _shfl_sync_32;
                                unsigned int _shfl_sync_33 = __shfl_sync(0xFFFFFFFF, incl_st_r, 31);
                                st_before_r = st_before_r + _shfl_sync_33;
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
                            unsigned int hit_mask_r = __ballot_sync(0xFFFFFFFF, hit_r != 0);
                            int _find_first_set_34 = __ffs(hit_mask_r);
                            int hit_lane_r = _find_first_set_34 - 1;
                            int _shfl_sync_35 = __shfl_sync(0xFFFFFFFF, b_r, hit_lane_r);
                            sel_batch_r = _shfl_sync_35;
                            int _shfl_sync_36 = __shfl_sync(0xFFFFFFFF, n_r, hit_lane_r);
                            sel_n_r = _shfl_sync_36;
                            unsigned int _shfl_sync_37 = __shfl_sync(0xFFFFFFFF, excl_r, hit_lane_r);
                            int sel_excl_r = (int)_shfl_sync_37;
                            unsigned int _shfl_sync_38 = __shfl_sync(0xFFFFFFFF, excl_si_r, hit_lane_r);
                            int sel_excl_si_r = (int)_shfl_sync_38;
                            unsigned int _shfl_sync_39 = __shfl_sync(0xFFFFFFFF, excl_st_r, hit_lane_r);
                            int sel_excl_st_r = (int)_shfl_sync_39;
                            item_r = (int)in_group_r - sel_excl_r;
                            slot_tile_base_r = ((int)si_before_r + sel_excl_si_r) * items_per_chunk + item_r;
                            counter_idx_r = ((int)st_before_r + sel_excl_st_r) * items_per_chunk + item_r;
                        }
                        int q_tile_r = item_r / num_kv_heads;
                        int kv_head_r = item_r - q_tile_r * num_kv_heads;
                        if (lane_1 == 0) {
                            unsigned int arrived_r = 0;
                            #pragma unroll 1
                            for (int _poll_r = 0; _poll_r < 1073741824; _poll_r++) {
                                unsigned int _atomic_fetch_add_40;
                                asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                                    : "=r"(_atomic_fetch_add_40) : "l"(&tile_counters[counter_idx_r * 4]), "r"(static_cast<uint32_t>(0)) : "memory");
                                arrived_r = _atomic_fetch_add_40;
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
                            work_token_words[token_base + 12] = (unsigned int)q_tile_r;
                            work_token_words[token_base + 13] = heavy_flags_u;
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
                        unsigned int q_16 = (unsigned int)((float)local_items * items_rcp);
                        if (q_16 * (unsigned int)items_per_chunk > local_items) {
                            q_16 = q_16 - 1;
                        }
                        if ((q_16 + 1) * (unsigned int)items_per_chunk <= local_items) {
                            q_16 = q_16 + 1;
                        }
                        unsigned int local_chunk = q_16;
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
                                unsigned int q_17 = (unsigned int)((float)local_chunk * full_rcp_u);
                                if (q_17 * full_den_u > local_chunk) {
                                    q_17 = q_17 - 1;
                                }
                                if ((q_17 + 1) * full_den_u <= local_chunk) {
                                    q_17 = q_17 + 1;
                                }
                                sel_batch = (int)q_17;
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
                                b3 = cursor_group * 32 + lane_1;
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
                                    int _max_41 = ((s3) > (1) ? (s3) : (1));
                                    pairs3 = (_max_41 + 255) / 256;
                                    unsigned int q_18 = (unsigned int)((float)(unsigned int)pairs3 * chunk_rcp);
                                    if ((unsigned int)pairs3 < q_18 * chunk_pairs_u) {
                                        q_18 = q_18 - 1;
                                    }
                                    if ((unsigned int)pairs3 >= (q_18 + 1) * chunk_pairs_u) {
                                        q_18 = q_18 + 1;
                                    }
                                    if ((unsigned int)pairs3 > q_18 * chunk_pairs_u) {
                                        q_18 = q_18 + 1;
                                    }
                                    n3 = (int)q_18;
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
                                uint32_t _warp_scan_sum_42 = mine3;
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_42) : "r"(1));
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_42) : "r"(2));
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_42) : "r"(4));
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_42) : "r"(8));
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_42) : "r"(16));
                                incl3 = _warp_scan_sum_42;
                                uint32_t _warp_scan_sum_43 = split_items3;
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_43) : "r"(1));
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_43) : "r"(2));
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_43) : "r"(4));
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_43) : "r"(8));
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_43) : "r"(16));
                                incl_si3 = _warp_scan_sum_43;
                                uint32_t _warp_scan_sum_44 = split_tiles3;
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_44) : "r"(1));
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_44) : "r"(2));
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_44) : "r"(4));
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_44) : "r"(8));
                                asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_44) : "r"(16));
                                incl_st3 = _warp_scan_sum_44;
                                unsigned int group_total = __shfl_sync(0xFFFFFFFF, incl3, 31);
                                if (local_chunk < before + group_total) {
                                    break;
                                }
                                before = before + group_total;
                                unsigned int _shfl_sync_45 = __shfl_sync(0xFFFFFFFF, incl_si3, 31);
                                si_before = si_before + _shfl_sync_45;
                                unsigned int _shfl_sync_46 = __shfl_sync(0xFFFFFFFF, incl_st3, 31);
                                st_before = st_before + _shfl_sync_46;
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
                            unsigned int hit_mask = __ballot_sync(0xFFFFFFFF, hit3 != 0);
                            int _find_first_set_47 = __ffs(hit_mask);
                            int hit_lane = _find_first_set_47 - 1;
                            int _shfl_sync_48 = __shfl_sync(0xFFFFFFFF, b3, hit_lane);
                            sel_batch = _shfl_sync_48;
                            int _shfl_sync_49 = __shfl_sync(0xFFFFFFFF, s3, hit_lane);
                            sel_seqlen = _shfl_sync_49;
                            int _shfl_sync_50 = __shfl_sync(0xFFFFFFFF, ph3, hit_lane);
                            sel_phase = _shfl_sync_50;
                            int _shfl_sync_51 = __shfl_sync(0xFFFFFFFF, n3, hit_lane);
                            sel_n = _shfl_sync_51;
                            int _shfl_sync_52 = __shfl_sync(0xFFFFFFFF, fullc3, hit_lane);
                            sel_full = _shfl_sync_52;
                            unsigned int _shfl_sync_53 = __shfl_sync(0xFFFFFFFF, excl3, hit_lane);
                            int sel_excl = (int)_shfl_sync_53;
                            unsigned int _shfl_sync_54 = __shfl_sync(0xFFFFFFFF, excl_si3, hit_lane);
                            int sel_excl_si = (int)_shfl_sync_54;
                            unsigned int _shfl_sync_55 = __shfl_sync(0xFFFFFFFF, excl_st3, hit_lane);
                            int sel_excl_st = (int)_shfl_sync_55;
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
                        int q_tile_sel = in_chunk / num_kv_heads;
                        int kv_head_sel = in_chunk - q_tile_sel * num_kv_heads;
                        int n_blocks_tile = (((sel_seqlen + 128 - 1) / 128) > (1) ? ((sel_seqlen + 128 - 1) / 128) : (1));
                        int block_begin_4 = 2 * chunk_idx * chunk_pairs;
                        int block_end_4 = 2 * (chunk_idx + 1) * chunk_pairs;
                        if (chunk_idx + 1 == sel_n) {
                            block_end_4 = n_blocks_tile;
                        }
                        int slot_tile_base_4 = 0;
                        int counter_idx_4 = 0;
                        if (sel_n > 1) {
                            slot_tile_base_4 = si_base_s * items_per_chunk + in_chunk;
                            counter_idx_4 = st_base_s * items_per_chunk + in_chunk;
                        }
                        if (valid_tok != 0) {
                            if (lane_1 == 0) {
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
                                work_token_words[token_base + 12] = (unsigned int)q_tile_sel;
                                work_token_words[token_base + 13] = heavy_flags_u;
                            }
                        }
                    }
                }
                if (lane_1 == 0) {
                    work_token_words[token_base] = valid_tok;
                    mbarrier_arrive(work_full_addr + (work_stage_sched) * 8);
                    if (valid_tok != 0) {
                        if (ticket >= chunk_items) {
                            unsigned int one_r = 1;
                            unsigned int slices_r = one_r << reduce_shift;
                            unsigned int pops_old;
                            asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                                : "=r"(pops_old) : "l"(&tile_counters[counter_idx_r * 4 + 1]), "r"(static_cast<uint32_t>(1)) : "memory");
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
            if (lane_1 == 0) {
                uint32_t _atomic_fetch_inc_56;
                asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                    : "=r"(_atomic_fetch_inc_56) : "l"(&queue_counters[1]), "r"(static_cast<uint32_t>(num_ctas - 1)) : "memory");
                done_old = _atomic_fetch_inc_56;
            }
            unsigned int _shfl_sync_57 = __shfl_sync(0xFFFFFFFF, done_old, 0);
            done_old = _shfl_sync_57;
            if ((int)done_old == num_ctas - 1) {
                if (lane_1 == 0) {
                    *(reinterpret_cast<unsigned int*>(queue_counters) + (0)) = 0;
                }
            }
        }
    // ---- Role: load_warp ----
    } else if (warp == 11) {
        { // load_warp_main
            int page_cons_stage = 0;
            unsigned int k_prod_stage = 0;
            unsigned int v_prod_stage = 0;
            unsigned int work_stage_l = 0;
            unsigned int _phase_work_full_4 = 0;
            mbarrier_wait(work_full_addr + (work_stage_l) * 8, _phase_work_full_4);
            unsigned int base_4 = work_stage_l * 16;
            unsigned int valid_4 = work_token_words[base_4];
            unsigned int kind_4 = work_token_words[base_4 + 1];
            unsigned int batch_4 = work_token_words[base_4 + 2];
            unsigned int kv_head_4 = work_token_words[base_4 + 3];
            unsigned int block_begin_5 = work_token_words[base_4 + 4];
            unsigned int block_end_5 = work_token_words[base_4 + 5];
            unsigned int seqlen_4 = work_token_words[base_4 + 6];
            unsigned int n_chunks_4 = work_token_words[base_4 + 7];
            unsigned int slot_tile_base_5 = work_token_words[base_4 + 8];
            unsigned int counter_idx_5 = work_token_words[base_4 + 9];
            unsigned int chunk_4 = work_token_words[base_4 + 10];
            unsigned int phase_4 = work_token_words[base_4 + 11];
            unsigned int q_tile_4 = work_token_words[base_4 + 12];
            unsigned int plan_flags_4 = work_token_words[base_4 + 13];
            mbarrier_arrive(work_empty_addr + (work_stage_l) * 8);
            work_stage_l += 1;
            if (work_stage_l == 4) { work_stage_l = 0; _phase_work_full_4 ^= 1; }
            unsigned int valid_l = valid_4;
            int kind_l = (int)kind_4;
            int batch_idx_l = (int)batch_4;
            int kv_head_idx = (int)kv_head_4;
            int pf_v_l = ((q_len <= 4) ? 1 : 0);
            int block_begin_l = (int)block_begin_5;
            int block_end_l = (int)block_end_5;
            int q_tile_l = (int)q_tile_4;
            int n_chunks_l = (int)n_chunks_4;
            int uniform_l = (int)plan_flags_4;
            uniform_l = (int)plan_flags_4 & 1;
            int heavy_l = (int)plan_flags_4 >> 1 & 1;
            unsigned int _phase_q_raw_empty = 1;
            unsigned int _phase_page_offsets_full = 0;
            unsigned int _phase_k_empty = 1;
            unsigned int _phase_v_empty = 1;
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
                    int ef_on_l = 0;
                    if (n_chunks_l == 1) {
                        if (cta_n_blocks <= 64) {
                            ef_on_l = 1;
                        }
                    }
                    if (n_chunks_l >= 4) {
                        ef_on_l = 1;
                    }
                    if (heavy_l == 1) {
                        ef_on_l = 0;
                    }
                    if (uniform_l == 0) {
                        ef_on_l = 0;
                    }
                    int n_pre = ((cta_n_blocks < 3) ? cta_n_blocks : 3);
                    int gate_block = cta_n_blocks - 1 - 2;
                    if (gate_block < 0) {
                        gate_block = 0;
                    }
                    if (elect_sync()) {
                        mbarrier_wait(q_raw_empty_addr, _phase_q_raw_empty);
                        int k2x_l = 0;
                        if (_tile_iter_l == 0) {
                            if (q_len - q_tile_l * 4 > 2) {
                                if (cta_n_blocks >= 1) {
                                    k2x_l = 1;
                                }
                            }
                        }
                        if (k2x_l != 0) {
                            mbarrier_arrive_expect_tx(q_raw_full_addr, 32768);
                        } else {
                            mbarrier_arrive_expect_tx(q_raw_full_addr, 16384);
                        }
                        tma_4d_gmem2smem(smem_qbf_addr + 0 * 16384, (&Q), 0, kv_head_idx * 16, batch_idx_l * q_len + q_tile_l * 4, 0, q_raw_full_addr);
                        if (k2x_l != 0) {
                            tma_4d_gmem2smem(smem_k_g0_addr + 32768, (&Q), 0, kv_head_idx * 16, batch_idx_l * q_len + q_tile_l * 4 + 2, 0, q_raw_full_addr);
                        }
                        _phase_q_raw_empty ^= 1;
                        int q2_early_l = 0;
                        #pragma unroll 1
                        for (int ni = 0; ni < n_pre; ni++) {
                            int pre_stage_u = page_cons_stage + ni;
                            int pre_stage = ((pre_stage_u >= 6) ? pre_stage_u - 6 : pre_stage_u);
                            int pre_pg_base = pre_stage * 8;
                            mbarrier_wait(page_offsets_full_addr + (((page_cons_stage + ni >= 6) ? page_cons_stage + ni - 6 : page_cons_stage + ni)) * 8, _phase_page_offsets_full ^ (unsigned int)(page_cons_stage + ni >= 6));
                            int pg_pre[2];
                            #pragma unroll
                            for (int pg_i = 0; pg_i < 2; pg_i++) {
                                pg_pre[pg_i] = smem_page_offsets[pre_pg_base + pg_i];
                            }
                            if (k2x_l != 0) {
                                if (ni == 2) {
                                    mbarrier_wait(q_raw_empty_addr, _phase_q_raw_empty);
                                }
                            }
                            mbarrier_wait(k_empty_addr + (k_prod_stage) * 8, _phase_k_empty);
                            mbarrier_arrive_expect_tx(k_full_addr + (k_prod_stage) * 8, 32768);
                            int kdst0 = smem_k_g0_addr + k_prod_stage * 16384;
                            int kdst0_g1 = smem_k_g1_addr + k_prod_stage * 16384;
                            unsigned long long kpol0_l = ((ef_on_l == 1) ? 1364590687093260288 : 1152921504606846976);
                            #pragma unroll
                            for (int pg_i_1 = 0; pg_i_1 < 2; pg_i_1++) {
                                int kpg0e = pg_pre[pg_i_1];
                                int ktoff0e = pg_i_1 * 8192;
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                    :: "r"(kdst0 + ktoff0e), "l"((&K)), "r"(0), "r"(0), "r"(0), "r"(kv_head_idx), "r"(kpg0e),
                                       "r"(k_full_addr + (k_prod_stage) * 8), "l"(kpol0_l) : "memory");
                                asm volatile(
                                    "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                    " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                    :: "r"(kdst0_g1 + ktoff0e), "l"((&K)), "r"(0), "r"(0), "r"(1), "r"(kv_head_idx), "r"(kpg0e),
                                       "r"(k_full_addr + (k_prod_stage) * 8), "l"(kpol0_l) : "memory");
                            }
                            k_prod_stage += 1;
                            if (k_prod_stage == 3) { k_prod_stage = 0; _phase_k_empty ^= 1; }
                            if (ni < 2) {
                                mbarrier_wait(v_empty_addr + (v_prod_stage) * 8, _phase_v_empty);
                                mbarrier_arrive_expect_tx(v_full_addr + (v_prod_stage) * 8, 32768);
                                int vdst0 = smem_v_g0_addr + v_prod_stage * 16384;
                                int vdst0_g1 = smem_v_g1_addr + v_prod_stage * 16384;
                                unsigned long long vpol0_l = ((ef_on_l == 1) ? 1364590687093260288 : 1152921504606846976);
                                #pragma unroll
                                for (int pg_i_2 = 0; pg_i_2 < 2; pg_i_2++) {
                                    int vpg0e = pg_pre[pg_i_2];
                                    int vtoff0e = pg_i_2 * 8192;
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(vdst0 + vtoff0e), "l"((&V)), "r"(0), "r"(0), "r"(0), "r"(kv_head_idx), "r"(vpg0e),
                                           "r"(v_full_addr + (v_prod_stage) * 8), "l"(vpol0_l) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(vdst0_g1 + vtoff0e), "l"((&V)), "r"(0), "r"(0), "r"(1), "r"(kv_head_idx), "r"(vpg0e),
                                           "r"(v_full_addr + (v_prod_stage) * 8), "l"(vpol0_l) : "memory");
                                }
                                v_prod_stage += 1;
                                if (v_prod_stage == 2) { v_prod_stage = 0; _phase_v_empty ^= 1; }
                            }
                        }
                        if (k2x_l != 0) {
                            if (cta_n_blocks < 3) {
                                mbarrier_wait(q_raw_empty_addr, _phase_q_raw_empty);
                            }
                        }
                        int half2_late_l = 0;
                        if (k2x_l == 0) {
                            if (q_len - q_tile_l * 4 > 2) {
                                half2_late_l = 1;
                            }
                        }
                        if (half2_late_l != 0) {
                            mbarrier_wait(q_raw_empty_addr, _phase_q_raw_empty);
                            mbarrier_arrive_expect_tx(q_raw_full_addr, 16384);
                            tma_4d_gmem2smem(smem_qbf_addr + 0 * 16384, (&Q), 0, kv_head_idx * 16, batch_idx_l * q_len + q_tile_l * 4 + 2, 0, q_raw_full_addr);
                            _phase_q_raw_empty ^= 1;
                        }
                        #pragma unroll 1
                        for (int ni_1 = 0; ni_1 < cta_n_blocks; ni_1++) {
                            int nk = ni_1 + 3;
                            if (nk < cta_n_blocks) {
                                int k_page_u = page_cons_stage + 3;
                                int k_page = ((k_page_u >= 6) ? k_page_u - 6 : k_page_u);
                                int kpg_base = k_page * 8;
                                mbarrier_wait(page_offsets_full_addr + (((page_cons_stage + 3 >= 6) ? page_cons_stage + 3 - 6 : page_cons_stage + 3)) * 8, _phase_page_offsets_full ^ (unsigned int)(page_cons_stage + 3 >= 6));
                                int pg_nk[2];
                                #pragma unroll
                                for (int pg_i_3 = 0; pg_i_3 < 2; pg_i_3++) {
                                    pg_nk[pg_i_3] = smem_page_offsets[kpg_base + pg_i_3];
                                }
                                mbarrier_wait(k_empty_addr + (k_prod_stage) * 8, _phase_k_empty);
                                mbarrier_arrive_expect_tx(k_full_addr + (k_prod_stage) * 8, 32768);
                                int kdst = smem_k_g0_addr + k_prod_stage * 16384;
                                int kdst_g1 = smem_k_g1_addr + k_prod_stage * 16384;
                                unsigned long long nkpol_l = ((ef_on_l == 1) ? 1364590687093260288 : 1152921504606846976);
                                #pragma unroll
                                for (int pg_i_4 = 0; pg_i_4 < 2; pg_i_4++) {
                                    int npg0e = pg_nk[pg_i_4];
                                    int ntoffe = pg_i_4 * 8192;
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(kdst + ntoffe), "l"((&K)), "r"(0), "r"(0), "r"(0), "r"(kv_head_idx), "r"(npg0e),
                                           "r"(k_full_addr + (k_prod_stage) * 8), "l"(nkpol_l) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(kdst_g1 + ntoffe), "l"((&K)), "r"(0), "r"(0), "r"(1), "r"(kv_head_idx), "r"(npg0e),
                                           "r"(k_full_addr + (k_prod_stage) * 8), "l"(nkpol_l) : "memory");
                                }
                                k_prod_stage += 1;
                                if (k_prod_stage == 3) { k_prod_stage = 0; _phase_k_empty ^= 1; }
                            }
                            int nv = ni_1 + 2;
                            if (nv < cta_n_blocks) {
                                int v_page_u = page_cons_stage + 2;
                                int v_page = ((v_page_u >= 6) ? v_page_u - 6 : v_page_u);
                                int vpg_base = v_page * 8;
                                mbarrier_wait(page_offsets_full_addr + (((page_cons_stage + 2 >= 6) ? page_cons_stage + 2 - 6 : page_cons_stage + 2)) * 8, _phase_page_offsets_full ^ (unsigned int)(page_cons_stage + 2 >= 6));
                                int pg_nv[2];
                                #pragma unroll
                                for (int pg_i_5 = 0; pg_i_5 < 2; pg_i_5++) {
                                    pg_nv[pg_i_5] = smem_page_offsets[vpg_base + pg_i_5];
                                }
                                mbarrier_wait(v_empty_addr + (v_prod_stage) * 8, _phase_v_empty);
                                mbarrier_arrive_expect_tx(v_full_addr + (v_prod_stage) * 8, 32768);
                                int vdst = smem_v_g0_addr + v_prod_stage * 16384;
                                int vdst_g1 = smem_v_g1_addr + v_prod_stage * 16384;
                                unsigned long long vpol_l = ((ef_on_l == 1) ? 1364590687093260288 : 1152921504606846976);
                                #pragma unroll
                                for (int pg_i_6 = 0; pg_i_6 < 2; pg_i_6++) {
                                    int vpg1e = pg_nv[pg_i_6];
                                    int vtoffe = pg_i_6 * 8192;
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(vdst + vtoffe), "l"((&V)), "r"(0), "r"(0), "r"(0), "r"(kv_head_idx), "r"(vpg1e),
                                           "r"(v_full_addr + (v_prod_stage) * 8), "l"(vpol_l) : "memory");
                                    asm volatile(
                                        "cp.async.bulk.tensor.5d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
                                        :: "r"(vdst_g1 + vtoffe), "l"((&V)), "r"(0), "r"(0), "r"(1), "r"(kv_head_idx), "r"(vpg1e),
                                           "r"(v_full_addr + (v_prod_stage) * 8), "l"(vpol_l) : "memory");
                                }
                                v_prod_stage += 1;
                                if (v_prod_stage == 2) { v_prod_stage = 0; _phase_v_empty ^= 1; }
                            }
                            mbarrier_arrive(page_offsets_empty_addr + (page_cons_stage) * 8);
                            page_cons_stage += 1;
                            if (page_cons_stage == 6) { page_cons_stage = 0; _phase_page_offsets_full ^= 1; }
                            if (ni_1 == gate_block) {
                                mbarrier_arrive(claim_gate_addr);
                            }
                        }
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_l) * 8, _phase_work_full_4);
                unsigned int base_4_1 = work_stage_l * 16;
                unsigned int valid_4_1 = work_token_words[base_4_1];
                unsigned int kind_4_1 = work_token_words[base_4_1 + 1];
                unsigned int batch_4_1 = work_token_words[base_4_1 + 2];
                unsigned int kv_head_4_1 = work_token_words[base_4_1 + 3];
                unsigned int block_begin_4_1 = work_token_words[base_4_1 + 4];
                unsigned int block_end_4_1 = work_token_words[base_4_1 + 5];
                unsigned int seqlen_4_1 = work_token_words[base_4_1 + 6];
                unsigned int n_chunks_4_1 = work_token_words[base_4_1 + 7];
                unsigned int slot_tile_base_4_1 = work_token_words[base_4_1 + 8];
                unsigned int counter_idx_4_1 = work_token_words[base_4_1 + 9];
                unsigned int chunk_4_1 = work_token_words[base_4_1 + 10];
                unsigned int phase_4_1 = work_token_words[base_4_1 + 11];
                unsigned int q_tile_4_1 = work_token_words[base_4_1 + 12];
                unsigned int plan_flags_4_1 = work_token_words[base_4_1 + 13];
                mbarrier_arrive(work_empty_addr + (work_stage_l) * 8);
                work_stage_l += 1;
                if (work_stage_l == 4) { work_stage_l = 0; _phase_work_full_4 ^= 1; }
                valid_l = valid_4_1;
                kind_l = (int)kind_4_1;
                batch_idx_l = (int)batch_4_1;
                kv_head_idx = (int)kv_head_4_1;
                block_begin_l = (int)block_begin_4_1;
                block_end_l = (int)block_end_4_1;
                q_tile_l = (int)q_tile_4_1;
                n_chunks_l = (int)n_chunks_4_1;
                uniform_l = (int)plan_flags_4_1;
                uniform_l = (int)plan_flags_4_1 & 1;
                heavy_l = (int)plan_flags_4_1 >> 1 & 1;
            }
        }
    }

    // Cleanup

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
