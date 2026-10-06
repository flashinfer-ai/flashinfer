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

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 320
#define TMEM_SCORES_OFFSET 0
#define TMEM_PROBABILITIES_OFFSET 256
#define TMEM_OUTPUT_OFFSET 128
#define NUM_KV_PIPE_STAGES 3
#define NUM_WORK_PIPE_STAGES 4
#define SMEM_Q_SMEM_OFF 1024
#define SMEM_Q_SMEM_STAGE_BYTES 16384
#define SMEM_Q_SMEM_STRIDE 16384
#define SMEM_KV_SMEM_OFF 17408
#define SMEM_KV_SMEM_STAGE_BYTES 65536
#define SMEM_KV_SMEM_STRIDE 65536
#define SMEM_V_SMEM_OFF 17408
#define SMEM_V_SMEM_STAGE_BYTES 65536
#define SMEM_V_SMEM_STRIDE 65536
#define SMEM_SCALE_SMEM_OFF 214016
#define SMEM_SCALE_SMEM_STAGE_BYTES 1536
#define SMEM_SCALE_SMEM_STRIDE 1536
#define SMEM_PARTIAL_SMEM_OFF 215552
#define SMEM_PARTIAL_SMEM_STAGE_BYTES 16384
#define SMEM_PARTIAL_SMEM_STRIDE 16384
#define SMEM_WORK_TOKEN_WORDS_OFF 231936
#define SMEM_WORK_TOKEN_WORDS_STAGE_BYTES 128
#define SMEM_WORK_TOKEN_WORDS_STRIDE 128
#define SMEM_HIST_SMEM_OFF 232064
#define SMEM_HIST_SMEM_STAGE_BYTES 256
#define SMEM_HIST_SMEM_STRIDE 256
#define SMEM_TOTAL 232320
#define THREADS 384
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




union MmaSmemDesc {
    uint64_t u64;
    uint32_t u32[2];
};


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


__device__ __forceinline__ void row_max_x32_accum(const float* sv, float2& acc) {
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        if (j % 2 == 0)
            acc.x = max_noftz(acc.x, max_noftz(sv[j*2], sv[j*2+1]));
        else
            acc.y = max_noftz(acc.y, max_noftz(sv[j*2], sv[j*2+1]));
    }
}




__device__ __forceinline__ void softmax_block_sum(const float* sv, float2* acc) {
    const float2* sv2 = reinterpret_cast<const float2*>(sv);
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        asm("add.f32x2 %0, %1, %2;"
            : "+l"(reinterpret_cast<uint64_t&>(*acc))
            : "l"(reinterpret_cast<uint64_t&>(*acc)),
              "l"(reinterpret_cast<const uint64_t&>(sv2[j])));
    }
}


__device__ __forceinline__ void fma_f32x2_inplace(float2* a, float2 b, float2 c) {
    unsigned long long r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(r)
        : "l"(*(unsigned long long*)a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    *(unsigned long long*)a = r;
}

__device__ __forceinline__ void mul_f32x2_inplace(float2* a, float2 b) {
    asm("mul.rn.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}


// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)


__device__ __forceinline__ void elect_commit2(int mbar_addr0, int mbar_addr1) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];\n\t"
        "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%1];\n\t"
        "}\n"
        :: "r"(mbar_addr0), "r"(mbar_addr1) : "memory");
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



__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(384, LAUNCH_MIN_BLOCKS) void
kernel_flashinfer_vsa_blk64_balanced_m64n256_ws_sm100(const __grid_constant__ CUtensorMap q, const __grid_constant__ CUtensorMap k, const __grid_constant__ CUtensorMap v, __nv_bfloat16* __restrict__ out, float* __restrict__ lse, int* __restrict__ indptr, int* __restrict__ indices, int sequence_q, int query_blocks, int total_tiles, int num_heads, float softmax_scale_log2, int return_lse, unsigned int* __restrict__ queue_counters, int* __restrict__ order_debug, int debug_order, unsigned int* __restrict__ trace_out)
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
    #define kv_full_addr (mbar_base + 16)
    #define kv_empty_addr (mbar_base + 40)
    #define s_full_addr (mbar_base + 64)
    #define s_empty_addr (mbar_base + 72)
    #define p_full_addr (mbar_base + 80)
    #define p_lastsplit_addr (mbar_base + 88)
    #define p_empty_addr (mbar_base + 96)
    #define corr_sig_addr (mbar_base + 104)
    #define corr_done_addr (mbar_base + 112)
    #define o_full_addr (mbar_base + 120)
    #define tile_done_addr (mbar_base + 128)
    #define work_full_addr (mbar_base + 136)
    #define work_empty_addr (mbar_base + 168)
    #define claim_gate_addr (mbar_base + 200)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    __nv_bfloat16* q_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int q_smem_addr = smem + 1024;
    __nv_bfloat16* kv_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int kv_smem_addr = smem + 17408;
    __nv_bfloat16* v_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int v_smem_addr = smem + 17408;
    float* scale_smem = reinterpret_cast<float*>(smem_raw + 214016);
    const int scale_smem_addr = smem + 214016;
    unsigned int* partial_smem = reinterpret_cast<unsigned int*>(smem_raw + 215552);
    const int partial_smem_addr = smem + 215552;
    unsigned int* work_token_words = reinterpret_cast<unsigned int*>(smem_raw + 231936);
    const int work_token_words_addr = smem + 231936;
    unsigned int* hist_smem = reinterpret_cast<unsigned int*>(smem_raw + 232064);
    const int hist_smem_addr = smem + 232064;

    // Mbarrier init (16 pipeline groups, 0 ordered-sequence groups, 26 barriers)
    // Mbarriers at smem_raw[0..208)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // q_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // kv_full: 3 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            // --- pipeline 'kv_pipe' ---
            // kv_empty: 3 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // s_full: 1 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            // s_empty: 1 barriers, init_count=128
            mbarrier_init(smem + 72, 128);
            // p_full: 1 barriers, init_count=256
            mbarrier_init(smem + 80, 256);
            // p_lastsplit: 1 barriers, init_count=128
            mbarrier_init(smem + 88, 128);
            // p_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            // corr_sig: 1 barriers, init_count=128
            mbarrier_init(smem + 104, 128);
            // corr_done: 1 barriers, init_count=128
            mbarrier_init(smem + 112, 128);
            // o_full: 1 barriers, init_count=1
            mbarrier_init(smem + 120, 1);
            // tile_done: 1 barriers, init_count=128
            mbarrier_init(smem + 128, 128);
            // --- pipeline 'work_pipe' ---
            // work_full: 4 barriers, init_count=1
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            // work_empty: 4 barriers, init_count=320
            mbarrier_init(smem + 168, 320);
            mbarrier_init(smem + 176, 320);
            mbarrier_init(smem + 184, 320);
            mbarrier_init(smem + 192, 320);
            // claim_gate: 1 barriers, init_count=1
            mbarrier_init(smem + 200, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 320 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 208);
    if (warp == 0) {
        int _tmem_hold = smem + 208;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_scores = taddr;
    const int tmem_probabilities = taddr + 256;
    const int tmem_output = taddr + 128;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
    }

    // ---- Role: softmax ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
        { // softmax_main
            unsigned int work_stage_s = 0;
            unsigned int _phase_work_full = 0;
            unsigned int _phase_s_full_0 = 0;
            unsigned int _phase_p_empty_0 = 1;
            unsigned int _phase_corr_done_0 = 0;
            #pragma unroll 1
            for (int tile_iter = 0; tile_iter < total_tiles; tile_iter++) {
                unsigned int valid = 1;
                unsigned int gate_t = 1;
                int q_block_t = 0;
                int head_t = 0;
                int row_begin_t = 0;
                int count_t = 0;
                int groups_t = 0;
                if (tile_iter == 0) {
                    int bid_t = blockIdx.x;
                    q_block_t = bid_t % query_blocks;
                    head_t = bid_t / query_blocks;
                    row_begin_t = indptr[q_block_t];
                    count_t = indptr[q_block_t + 1] - row_begin_t;
                    groups_t = (count_t + 4 - 1) / 4;
                    if (bid_t >= total_tiles) {
                        valid = 0;
                    }
                } else {
                    mbarrier_wait(work_full_addr + (work_stage_s) * 8, _phase_work_full);
                    unsigned int base = work_stage_s * 8;
                    valid = work_token_words[base];
                    unsigned int q_block_u = work_token_words[base + 1];
                    unsigned int head_u = work_token_words[base + 2];
                    unsigned int row_begin_u = work_token_words[base + 3];
                    unsigned int count_u = work_token_words[base + 4];
                    unsigned int groups_u = work_token_words[base + 5];
                    gate_t = work_token_words[base + 7];
                    mbarrier_arrive(work_empty_addr + (work_stage_s) * 8);
                    work_stage_s += 1;
                    if (work_stage_s == 4) { work_stage_s = 0; _phase_work_full ^= 1; }
                    q_block_t = (int)q_block_u;
                    head_t = (int)head_u;
                    row_begin_t = (int)row_begin_u;
                    count_t = (int)count_u;
                    groups_t = (int)groups_u;
                }
                int tile_t = q_block_t + head_t * query_blocks;
                unsigned int valid_s = valid;
                if (valid_s == 0) {
                    break;
                }
                if (valid_s != 0) {
                    int q_block = q_block_t;
                    int head = head_t;
                    int query_base = q_block * 64;
                    int q_valid = sequence_q - query_base;
                    if (q_valid > 64) {
                        q_valid = 64;
                    }
                    if (q_valid < 0) {
                        q_valid = 0;
                    }
                    const int warp_in_role = warp;
                    const int tmem_row_origin = warp_in_role * 32;
                    const int warp_col = warp_in_role / 2;
                    int my_row = warp_in_role % 2 * 32 + lane;
                    int stat_slot = warp_in_role * 32 + lane;
                    int row_valid = ((my_row < q_valid) ? 1 : 0);
                    float row_max = -CAKE_INF;
                    float row_sum = 0.0f;
                    #pragma unroll 1
                    for (int group_index = 0; group_index < groups_t; group_index++) {
                        mbarrier_wait(s_full_addr, _phase_s_full_0);
                        _phase_s_full_0 ^= 1;
                        int remaining = count_t - group_index * 4;
                        int valid_lo = ((remaining > warp_col && row_valid != 0) ? 64 : 0);
                        int valid_hi = ((remaining > warp_col + 2 && row_valid != 0) ? 64 : 0);
                        int valid_cols = valid_lo + valid_hi;
                        int score_addr = taddr + (unsigned int)(tmem_row_origin << 16);
                        float _tmem_load_0[128];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                            : "r"(score_addr));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_0[32]), "=f"(_tmem_load_0[33]), "=f"(_tmem_load_0[34]), "=f"(_tmem_load_0[35]), "=f"(_tmem_load_0[36]), "=f"(_tmem_load_0[37]), "=f"(_tmem_load_0[38]), "=f"(_tmem_load_0[39]), "=f"(_tmem_load_0[40]), "=f"(_tmem_load_0[41]), "=f"(_tmem_load_0[42]), "=f"(_tmem_load_0[43]), "=f"(_tmem_load_0[44]), "=f"(_tmem_load_0[45]), "=f"(_tmem_load_0[46]), "=f"(_tmem_load_0[47]), "=f"(_tmem_load_0[48]), "=f"(_tmem_load_0[49]), "=f"(_tmem_load_0[50]), "=f"(_tmem_load_0[51]), "=f"(_tmem_load_0[52]), "=f"(_tmem_load_0[53]), "=f"(_tmem_load_0[54]), "=f"(_tmem_load_0[55]), "=f"(_tmem_load_0[56]), "=f"(_tmem_load_0[57]), "=f"(_tmem_load_0[58]), "=f"(_tmem_load_0[59]), "=f"(_tmem_load_0[60]), "=f"(_tmem_load_0[61]), "=f"(_tmem_load_0[62]), "=f"(_tmem_load_0[63])
                            : "r"(score_addr + 32));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_0[64]), "=f"(_tmem_load_0[65]), "=f"(_tmem_load_0[66]), "=f"(_tmem_load_0[67]), "=f"(_tmem_load_0[68]), "=f"(_tmem_load_0[69]), "=f"(_tmem_load_0[70]), "=f"(_tmem_load_0[71]), "=f"(_tmem_load_0[72]), "=f"(_tmem_load_0[73]), "=f"(_tmem_load_0[74]), "=f"(_tmem_load_0[75]), "=f"(_tmem_load_0[76]), "=f"(_tmem_load_0[77]), "=f"(_tmem_load_0[78]), "=f"(_tmem_load_0[79]), "=f"(_tmem_load_0[80]), "=f"(_tmem_load_0[81]), "=f"(_tmem_load_0[82]), "=f"(_tmem_load_0[83]), "=f"(_tmem_load_0[84]), "=f"(_tmem_load_0[85]), "=f"(_tmem_load_0[86]), "=f"(_tmem_load_0[87]), "=f"(_tmem_load_0[88]), "=f"(_tmem_load_0[89]), "=f"(_tmem_load_0[90]), "=f"(_tmem_load_0[91]), "=f"(_tmem_load_0[92]), "=f"(_tmem_load_0[93]), "=f"(_tmem_load_0[94]), "=f"(_tmem_load_0[95])
                            : "r"(score_addr + 64));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_0[96]), "=f"(_tmem_load_0[97]), "=f"(_tmem_load_0[98]), "=f"(_tmem_load_0[99]), "=f"(_tmem_load_0[100]), "=f"(_tmem_load_0[101]), "=f"(_tmem_load_0[102]), "=f"(_tmem_load_0[103]), "=f"(_tmem_load_0[104]), "=f"(_tmem_load_0[105]), "=f"(_tmem_load_0[106]), "=f"(_tmem_load_0[107]), "=f"(_tmem_load_0[108]), "=f"(_tmem_load_0[109]), "=f"(_tmem_load_0[110]), "=f"(_tmem_load_0[111]), "=f"(_tmem_load_0[112]), "=f"(_tmem_load_0[113]), "=f"(_tmem_load_0[114]), "=f"(_tmem_load_0[115]), "=f"(_tmem_load_0[116]), "=f"(_tmem_load_0[117]), "=f"(_tmem_load_0[118]), "=f"(_tmem_load_0[119]), "=f"(_tmem_load_0[120]), "=f"(_tmem_load_0[121]), "=f"(_tmem_load_0[122]), "=f"(_tmem_load_0[123]), "=f"(_tmem_load_0[124]), "=f"(_tmem_load_0[125]), "=f"(_tmem_load_0[126]), "=f"(_tmem_load_0[127])
                            : "r"(score_addr + 96));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        mbarrier_arrive(s_empty_addr);
                        if (valid_lo == 0) {
                            uint32_t _slice_hi_mask_0;
                            {
                                int _lim_0 = 64;
                                if (_lim_0 <= 0) { _slice_hi_mask_0 = 0u; }
                                else if (_lim_0 >= 32) { _slice_hi_mask_0 = 0xFFFFFFFFu; }
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
                                        "}" : "=r"(_slice_hi_mask_0) : "r"(_lim_0));
                                }
                            }
                            if (!(~_slice_hi_mask_0 & (1u << 0))) _tmem_load_0[0] = -CAKE_INF;
                            if (!(~_slice_hi_mask_0 & (1u << 1))) _tmem_load_0[1] = -CAKE_INF;
                            if (!(~_slice_hi_mask_0 & (1u << 2))) _tmem_load_0[2] = -CAKE_INF;
                            if (!(~_slice_hi_mask_0 & (1u << 3))) _tmem_load_0[3] = -CAKE_INF;
                            if (!(~_slice_hi_mask_0 & (1u << 4))) _tmem_load_0[4] = -CAKE_INF;
                            if (!(~_slice_hi_mask_0 & (1u << 5))) _tmem_load_0[5] = -CAKE_INF;
                            if (!(~_slice_hi_mask_0 & (1u << 6))) _tmem_load_0[6] = -CAKE_INF;
                            if (!(~_slice_hi_mask_0 & (1u << 7))) _tmem_load_0[7] = -CAKE_INF;
                            if (!(~_slice_hi_mask_0 & (1u << 8))) _tmem_load_0[8] = -CAKE_INF;
                            if (!(~_slice_hi_mask_0 & (1u << 9))) _tmem_load_0[9] = -CAKE_INF;
                            if (!(~_slice_hi_mask_0 & (1u << 10))) _tmem_load_0[10] = -CAKE_INF;
                            if (!(~_slice_hi_mask_0 & (1u << 11))) _tmem_load_0[11] = -CAKE_INF;
                            if (!(~_slice_hi_mask_0 & (1u << 12))) _tmem_load_0[12] = -CAKE_INF;
                            if (!(~_slice_hi_mask_0 & (1u << 13))) _tmem_load_0[13] = -CAKE_INF;
                            if (!(~_slice_hi_mask_0 & (1u << 14))) _tmem_load_0[14] = -CAKE_INF;
                            if (!(~_slice_hi_mask_0 & (1u << 15))) _tmem_load_0[15] = -CAKE_INF;
                            if (!(~_slice_hi_mask_0 & (1u << 16))) _tmem_load_0[16] = -CAKE_INF;
                            if (!(~_slice_hi_mask_0 & (1u << 17))) _tmem_load_0[17] = -CAKE_INF;
                            if (!(~_slice_hi_mask_0 & (1u << 18))) _tmem_load_0[18] = -CAKE_INF;
                            if (!(~_slice_hi_mask_0 & (1u << 19))) _tmem_load_0[19] = -CAKE_INF;
                            if (!(~_slice_hi_mask_0 & (1u << 20))) _tmem_load_0[20] = -CAKE_INF;
                            if (!(~_slice_hi_mask_0 & (1u << 21))) _tmem_load_0[21] = -CAKE_INF;
                            if (!(~_slice_hi_mask_0 & (1u << 22))) _tmem_load_0[22] = -CAKE_INF;
                            if (!(~_slice_hi_mask_0 & (1u << 23))) _tmem_load_0[23] = -CAKE_INF;
                            if (!(~_slice_hi_mask_0 & (1u << 24))) _tmem_load_0[24] = -CAKE_INF;
                            if (!(~_slice_hi_mask_0 & (1u << 25))) _tmem_load_0[25] = -CAKE_INF;
                            if (!(~_slice_hi_mask_0 & (1u << 26))) _tmem_load_0[26] = -CAKE_INF;
                            if (!(~_slice_hi_mask_0 & (1u << 27))) _tmem_load_0[27] = -CAKE_INF;
                            if (!(~_slice_hi_mask_0 & (1u << 28))) _tmem_load_0[28] = -CAKE_INF;
                            if (!(~_slice_hi_mask_0 & (1u << 29))) _tmem_load_0[29] = -CAKE_INF;
                            if (!(~_slice_hi_mask_0 & (1u << 30))) _tmem_load_0[30] = -CAKE_INF;
                            if (!(~_slice_hi_mask_0 & (1u << 31))) _tmem_load_0[31] = -CAKE_INF;
                            uint32_t _slice_hi_mask_1;
                            {
                                int _lim_1 = 32;
                                if (_lim_1 <= 0) { _slice_hi_mask_1 = 0u; }
                                else if (_lim_1 >= 32) { _slice_hi_mask_1 = 0xFFFFFFFFu; }
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
                                        "}" : "=r"(_slice_hi_mask_1) : "r"(_lim_1));
                                }
                            }
                            if (!(~_slice_hi_mask_1 & (1u << 0))) _tmem_load_0[32] = -CAKE_INF;
                            if (!(~_slice_hi_mask_1 & (1u << 1))) _tmem_load_0[33] = -CAKE_INF;
                            if (!(~_slice_hi_mask_1 & (1u << 2))) _tmem_load_0[34] = -CAKE_INF;
                            if (!(~_slice_hi_mask_1 & (1u << 3))) _tmem_load_0[35] = -CAKE_INF;
                            if (!(~_slice_hi_mask_1 & (1u << 4))) _tmem_load_0[36] = -CAKE_INF;
                            if (!(~_slice_hi_mask_1 & (1u << 5))) _tmem_load_0[37] = -CAKE_INF;
                            if (!(~_slice_hi_mask_1 & (1u << 6))) _tmem_load_0[38] = -CAKE_INF;
                            if (!(~_slice_hi_mask_1 & (1u << 7))) _tmem_load_0[39] = -CAKE_INF;
                            if (!(~_slice_hi_mask_1 & (1u << 8))) _tmem_load_0[40] = -CAKE_INF;
                            if (!(~_slice_hi_mask_1 & (1u << 9))) _tmem_load_0[41] = -CAKE_INF;
                            if (!(~_slice_hi_mask_1 & (1u << 10))) _tmem_load_0[42] = -CAKE_INF;
                            if (!(~_slice_hi_mask_1 & (1u << 11))) _tmem_load_0[43] = -CAKE_INF;
                            if (!(~_slice_hi_mask_1 & (1u << 12))) _tmem_load_0[44] = -CAKE_INF;
                            if (!(~_slice_hi_mask_1 & (1u << 13))) _tmem_load_0[45] = -CAKE_INF;
                            if (!(~_slice_hi_mask_1 & (1u << 14))) _tmem_load_0[46] = -CAKE_INF;
                            if (!(~_slice_hi_mask_1 & (1u << 15))) _tmem_load_0[47] = -CAKE_INF;
                            if (!(~_slice_hi_mask_1 & (1u << 16))) _tmem_load_0[48] = -CAKE_INF;
                            if (!(~_slice_hi_mask_1 & (1u << 17))) _tmem_load_0[49] = -CAKE_INF;
                            if (!(~_slice_hi_mask_1 & (1u << 18))) _tmem_load_0[50] = -CAKE_INF;
                            if (!(~_slice_hi_mask_1 & (1u << 19))) _tmem_load_0[51] = -CAKE_INF;
                            if (!(~_slice_hi_mask_1 & (1u << 20))) _tmem_load_0[52] = -CAKE_INF;
                            if (!(~_slice_hi_mask_1 & (1u << 21))) _tmem_load_0[53] = -CAKE_INF;
                            if (!(~_slice_hi_mask_1 & (1u << 22))) _tmem_load_0[54] = -CAKE_INF;
                            if (!(~_slice_hi_mask_1 & (1u << 23))) _tmem_load_0[55] = -CAKE_INF;
                            if (!(~_slice_hi_mask_1 & (1u << 24))) _tmem_load_0[56] = -CAKE_INF;
                            if (!(~_slice_hi_mask_1 & (1u << 25))) _tmem_load_0[57] = -CAKE_INF;
                            if (!(~_slice_hi_mask_1 & (1u << 26))) _tmem_load_0[58] = -CAKE_INF;
                            if (!(~_slice_hi_mask_1 & (1u << 27))) _tmem_load_0[59] = -CAKE_INF;
                            if (!(~_slice_hi_mask_1 & (1u << 28))) _tmem_load_0[60] = -CAKE_INF;
                            if (!(~_slice_hi_mask_1 & (1u << 29))) _tmem_load_0[61] = -CAKE_INF;
                            if (!(~_slice_hi_mask_1 & (1u << 30))) _tmem_load_0[62] = -CAKE_INF;
                            if (!(~_slice_hi_mask_1 & (1u << 31))) _tmem_load_0[63] = -CAKE_INF;
                            uint32_t _slice_hi_mask_2;
                            {
                                int _lim_2 = 0;
                                if (_lim_2 <= 0) { _slice_hi_mask_2 = 0u; }
                                else if (_lim_2 >= 32) { _slice_hi_mask_2 = 0xFFFFFFFFu; }
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
                                        "}" : "=r"(_slice_hi_mask_2) : "r"(_lim_2));
                                }
                            }
                            if (!(~_slice_hi_mask_2 & (1u << 0))) _tmem_load_0[64] = -CAKE_INF;
                            if (!(~_slice_hi_mask_2 & (1u << 1))) _tmem_load_0[65] = -CAKE_INF;
                            if (!(~_slice_hi_mask_2 & (1u << 2))) _tmem_load_0[66] = -CAKE_INF;
                            if (!(~_slice_hi_mask_2 & (1u << 3))) _tmem_load_0[67] = -CAKE_INF;
                            if (!(~_slice_hi_mask_2 & (1u << 4))) _tmem_load_0[68] = -CAKE_INF;
                            if (!(~_slice_hi_mask_2 & (1u << 5))) _tmem_load_0[69] = -CAKE_INF;
                            if (!(~_slice_hi_mask_2 & (1u << 6))) _tmem_load_0[70] = -CAKE_INF;
                            if (!(~_slice_hi_mask_2 & (1u << 7))) _tmem_load_0[71] = -CAKE_INF;
                            if (!(~_slice_hi_mask_2 & (1u << 8))) _tmem_load_0[72] = -CAKE_INF;
                            if (!(~_slice_hi_mask_2 & (1u << 9))) _tmem_load_0[73] = -CAKE_INF;
                            if (!(~_slice_hi_mask_2 & (1u << 10))) _tmem_load_0[74] = -CAKE_INF;
                            if (!(~_slice_hi_mask_2 & (1u << 11))) _tmem_load_0[75] = -CAKE_INF;
                            if (!(~_slice_hi_mask_2 & (1u << 12))) _tmem_load_0[76] = -CAKE_INF;
                            if (!(~_slice_hi_mask_2 & (1u << 13))) _tmem_load_0[77] = -CAKE_INF;
                            if (!(~_slice_hi_mask_2 & (1u << 14))) _tmem_load_0[78] = -CAKE_INF;
                            if (!(~_slice_hi_mask_2 & (1u << 15))) _tmem_load_0[79] = -CAKE_INF;
                            if (!(~_slice_hi_mask_2 & (1u << 16))) _tmem_load_0[80] = -CAKE_INF;
                            if (!(~_slice_hi_mask_2 & (1u << 17))) _tmem_load_0[81] = -CAKE_INF;
                            if (!(~_slice_hi_mask_2 & (1u << 18))) _tmem_load_0[82] = -CAKE_INF;
                            if (!(~_slice_hi_mask_2 & (1u << 19))) _tmem_load_0[83] = -CAKE_INF;
                            if (!(~_slice_hi_mask_2 & (1u << 20))) _tmem_load_0[84] = -CAKE_INF;
                            if (!(~_slice_hi_mask_2 & (1u << 21))) _tmem_load_0[85] = -CAKE_INF;
                            if (!(~_slice_hi_mask_2 & (1u << 22))) _tmem_load_0[86] = -CAKE_INF;
                            if (!(~_slice_hi_mask_2 & (1u << 23))) _tmem_load_0[87] = -CAKE_INF;
                            if (!(~_slice_hi_mask_2 & (1u << 24))) _tmem_load_0[88] = -CAKE_INF;
                            if (!(~_slice_hi_mask_2 & (1u << 25))) _tmem_load_0[89] = -CAKE_INF;
                            if (!(~_slice_hi_mask_2 & (1u << 26))) _tmem_load_0[90] = -CAKE_INF;
                            if (!(~_slice_hi_mask_2 & (1u << 27))) _tmem_load_0[91] = -CAKE_INF;
                            if (!(~_slice_hi_mask_2 & (1u << 28))) _tmem_load_0[92] = -CAKE_INF;
                            if (!(~_slice_hi_mask_2 & (1u << 29))) _tmem_load_0[93] = -CAKE_INF;
                            if (!(~_slice_hi_mask_2 & (1u << 30))) _tmem_load_0[94] = -CAKE_INF;
                            if (!(~_slice_hi_mask_2 & (1u << 31))) _tmem_load_0[95] = -CAKE_INF;
                            uint32_t _slice_hi_mask_3;
                            {
                                int _lim_3 = -32;
                                if (_lim_3 <= 0) { _slice_hi_mask_3 = 0u; }
                                else if (_lim_3 >= 32) { _slice_hi_mask_3 = 0xFFFFFFFFu; }
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
                                        "}" : "=r"(_slice_hi_mask_3) : "r"(_lim_3));
                                }
                            }
                            if (!(~_slice_hi_mask_3 & (1u << 0))) _tmem_load_0[96] = -CAKE_INF;
                            if (!(~_slice_hi_mask_3 & (1u << 1))) _tmem_load_0[97] = -CAKE_INF;
                            if (!(~_slice_hi_mask_3 & (1u << 2))) _tmem_load_0[98] = -CAKE_INF;
                            if (!(~_slice_hi_mask_3 & (1u << 3))) _tmem_load_0[99] = -CAKE_INF;
                            if (!(~_slice_hi_mask_3 & (1u << 4))) _tmem_load_0[100] = -CAKE_INF;
                            if (!(~_slice_hi_mask_3 & (1u << 5))) _tmem_load_0[101] = -CAKE_INF;
                            if (!(~_slice_hi_mask_3 & (1u << 6))) _tmem_load_0[102] = -CAKE_INF;
                            if (!(~_slice_hi_mask_3 & (1u << 7))) _tmem_load_0[103] = -CAKE_INF;
                            if (!(~_slice_hi_mask_3 & (1u << 8))) _tmem_load_0[104] = -CAKE_INF;
                            if (!(~_slice_hi_mask_3 & (1u << 9))) _tmem_load_0[105] = -CAKE_INF;
                            if (!(~_slice_hi_mask_3 & (1u << 10))) _tmem_load_0[106] = -CAKE_INF;
                            if (!(~_slice_hi_mask_3 & (1u << 11))) _tmem_load_0[107] = -CAKE_INF;
                            if (!(~_slice_hi_mask_3 & (1u << 12))) _tmem_load_0[108] = -CAKE_INF;
                            if (!(~_slice_hi_mask_3 & (1u << 13))) _tmem_load_0[109] = -CAKE_INF;
                            if (!(~_slice_hi_mask_3 & (1u << 14))) _tmem_load_0[110] = -CAKE_INF;
                            if (!(~_slice_hi_mask_3 & (1u << 15))) _tmem_load_0[111] = -CAKE_INF;
                            if (!(~_slice_hi_mask_3 & (1u << 16))) _tmem_load_0[112] = -CAKE_INF;
                            if (!(~_slice_hi_mask_3 & (1u << 17))) _tmem_load_0[113] = -CAKE_INF;
                            if (!(~_slice_hi_mask_3 & (1u << 18))) _tmem_load_0[114] = -CAKE_INF;
                            if (!(~_slice_hi_mask_3 & (1u << 19))) _tmem_load_0[115] = -CAKE_INF;
                            if (!(~_slice_hi_mask_3 & (1u << 20))) _tmem_load_0[116] = -CAKE_INF;
                            if (!(~_slice_hi_mask_3 & (1u << 21))) _tmem_load_0[117] = -CAKE_INF;
                            if (!(~_slice_hi_mask_3 & (1u << 22))) _tmem_load_0[118] = -CAKE_INF;
                            if (!(~_slice_hi_mask_3 & (1u << 23))) _tmem_load_0[119] = -CAKE_INF;
                            if (!(~_slice_hi_mask_3 & (1u << 24))) _tmem_load_0[120] = -CAKE_INF;
                            if (!(~_slice_hi_mask_3 & (1u << 25))) _tmem_load_0[121] = -CAKE_INF;
                            if (!(~_slice_hi_mask_3 & (1u << 26))) _tmem_load_0[122] = -CAKE_INF;
                            if (!(~_slice_hi_mask_3 & (1u << 27))) _tmem_load_0[123] = -CAKE_INF;
                            if (!(~_slice_hi_mask_3 & (1u << 28))) _tmem_load_0[124] = -CAKE_INF;
                            if (!(~_slice_hi_mask_3 & (1u << 29))) _tmem_load_0[125] = -CAKE_INF;
                            if (!(~_slice_hi_mask_3 & (1u << 30))) _tmem_load_0[126] = -CAKE_INF;
                            if (!(~_slice_hi_mask_3 & (1u << 31))) _tmem_load_0[127] = -CAKE_INF;
                        }
                        if (valid_hi == 0) {
                            uint32_t _slice_lo_mask_0;
                            {
                                int _lim_4 = 64;
                                if (_lim_4 <= 0) { _slice_lo_mask_0 = 0u; }
                                else if (_lim_4 >= 32) { _slice_lo_mask_0 = 0xFFFFFFFFu; }
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
                                        "}" : "=r"(_slice_lo_mask_0) : "r"(_lim_4));
                                }
                            }
                            if (!(_slice_lo_mask_0 & (1u << 0))) _tmem_load_0[0] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 1))) _tmem_load_0[1] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 2))) _tmem_load_0[2] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 3))) _tmem_load_0[3] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 4))) _tmem_load_0[4] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 5))) _tmem_load_0[5] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 6))) _tmem_load_0[6] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 7))) _tmem_load_0[7] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 8))) _tmem_load_0[8] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 9))) _tmem_load_0[9] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 10))) _tmem_load_0[10] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 11))) _tmem_load_0[11] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 12))) _tmem_load_0[12] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 13))) _tmem_load_0[13] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 14))) _tmem_load_0[14] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 15))) _tmem_load_0[15] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 16))) _tmem_load_0[16] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 17))) _tmem_load_0[17] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 18))) _tmem_load_0[18] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 19))) _tmem_load_0[19] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 20))) _tmem_load_0[20] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 21))) _tmem_load_0[21] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 22))) _tmem_load_0[22] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 23))) _tmem_load_0[23] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 24))) _tmem_load_0[24] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 25))) _tmem_load_0[25] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 26))) _tmem_load_0[26] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 27))) _tmem_load_0[27] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 28))) _tmem_load_0[28] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 29))) _tmem_load_0[29] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 30))) _tmem_load_0[30] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 31))) _tmem_load_0[31] = -CAKE_INF;
                            uint32_t _slice_lo_mask_1;
                            {
                                int _lim_5 = 32;
                                if (_lim_5 <= 0) { _slice_lo_mask_1 = 0u; }
                                else if (_lim_5 >= 32) { _slice_lo_mask_1 = 0xFFFFFFFFu; }
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
                                        "}" : "=r"(_slice_lo_mask_1) : "r"(_lim_5));
                                }
                            }
                            if (!(_slice_lo_mask_1 & (1u << 0))) _tmem_load_0[32] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 1))) _tmem_load_0[33] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 2))) _tmem_load_0[34] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 3))) _tmem_load_0[35] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 4))) _tmem_load_0[36] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 5))) _tmem_load_0[37] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 6))) _tmem_load_0[38] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 7))) _tmem_load_0[39] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 8))) _tmem_load_0[40] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 9))) _tmem_load_0[41] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 10))) _tmem_load_0[42] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 11))) _tmem_load_0[43] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 12))) _tmem_load_0[44] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 13))) _tmem_load_0[45] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 14))) _tmem_load_0[46] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 15))) _tmem_load_0[47] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 16))) _tmem_load_0[48] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 17))) _tmem_load_0[49] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 18))) _tmem_load_0[50] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 19))) _tmem_load_0[51] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 20))) _tmem_load_0[52] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 21))) _tmem_load_0[53] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 22))) _tmem_load_0[54] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 23))) _tmem_load_0[55] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 24))) _tmem_load_0[56] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 25))) _tmem_load_0[57] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 26))) _tmem_load_0[58] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 27))) _tmem_load_0[59] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 28))) _tmem_load_0[60] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 29))) _tmem_load_0[61] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 30))) _tmem_load_0[62] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 31))) _tmem_load_0[63] = -CAKE_INF;
                            uint32_t _slice_lo_mask_2;
                            {
                                int _lim_6 = 0;
                                if (_lim_6 <= 0) { _slice_lo_mask_2 = 0u; }
                                else if (_lim_6 >= 32) { _slice_lo_mask_2 = 0xFFFFFFFFu; }
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
                                        "}" : "=r"(_slice_lo_mask_2) : "r"(_lim_6));
                                }
                            }
                            if (!(_slice_lo_mask_2 & (1u << 0))) _tmem_load_0[64] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 1))) _tmem_load_0[65] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 2))) _tmem_load_0[66] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 3))) _tmem_load_0[67] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 4))) _tmem_load_0[68] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 5))) _tmem_load_0[69] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 6))) _tmem_load_0[70] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 7))) _tmem_load_0[71] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 8))) _tmem_load_0[72] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 9))) _tmem_load_0[73] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 10))) _tmem_load_0[74] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 11))) _tmem_load_0[75] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 12))) _tmem_load_0[76] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 13))) _tmem_load_0[77] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 14))) _tmem_load_0[78] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 15))) _tmem_load_0[79] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 16))) _tmem_load_0[80] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 17))) _tmem_load_0[81] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 18))) _tmem_load_0[82] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 19))) _tmem_load_0[83] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 20))) _tmem_load_0[84] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 21))) _tmem_load_0[85] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 22))) _tmem_load_0[86] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 23))) _tmem_load_0[87] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 24))) _tmem_load_0[88] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 25))) _tmem_load_0[89] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 26))) _tmem_load_0[90] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 27))) _tmem_load_0[91] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 28))) _tmem_load_0[92] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 29))) _tmem_load_0[93] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 30))) _tmem_load_0[94] = -CAKE_INF;
                            if (!(_slice_lo_mask_2 & (1u << 31))) _tmem_load_0[95] = -CAKE_INF;
                            uint32_t _slice_lo_mask_3;
                            {
                                int _lim_7 = -32;
                                if (_lim_7 <= 0) { _slice_lo_mask_3 = 0u; }
                                else if (_lim_7 >= 32) { _slice_lo_mask_3 = 0xFFFFFFFFu; }
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
                                        "}" : "=r"(_slice_lo_mask_3) : "r"(_lim_7));
                                }
                            }
                            if (!(_slice_lo_mask_3 & (1u << 0))) _tmem_load_0[96] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 1))) _tmem_load_0[97] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 2))) _tmem_load_0[98] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 3))) _tmem_load_0[99] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 4))) _tmem_load_0[100] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 5))) _tmem_load_0[101] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 6))) _tmem_load_0[102] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 7))) _tmem_load_0[103] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 8))) _tmem_load_0[104] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 9))) _tmem_load_0[105] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 10))) _tmem_load_0[106] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 11))) _tmem_load_0[107] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 12))) _tmem_load_0[108] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 13))) _tmem_load_0[109] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 14))) _tmem_load_0[110] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 15))) _tmem_load_0[111] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 16))) _tmem_load_0[112] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 17))) _tmem_load_0[113] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 18))) _tmem_load_0[114] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 19))) _tmem_load_0[115] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 20))) _tmem_load_0[116] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 21))) _tmem_load_0[117] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 22))) _tmem_load_0[118] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 23))) _tmem_load_0[119] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 24))) _tmem_load_0[120] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 25))) _tmem_load_0[121] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 26))) _tmem_load_0[122] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 27))) _tmem_load_0[123] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 28))) _tmem_load_0[124] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 29))) _tmem_load_0[125] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 30))) _tmem_load_0[126] = -CAKE_INF;
                            if (!(_slice_lo_mask_3 & (1u << 31))) _tmem_load_0[127] = -CAKE_INF;
                        }
                        float2 _reg_reduce_max2_8 = {-CAKE_INF, -CAKE_INF};
                        row_max_x32_accum(&_tmem_load_0[0], _reg_reduce_max2_8);
                        row_max_x32_accum(&_tmem_load_0[32], _reg_reduce_max2_8);
                        row_max_x32_accum(&_tmem_load_0[64], _reg_reduce_max2_8);
                        row_max_x32_accum(&_tmem_load_0[96], _reg_reduce_max2_8);
                        float _tmem_load_0_max = row_max_reduce(_reg_reduce_max2_8);
                        float tile_max = _tmem_load_0_max;
                        if (valid_cols <= 0) {
                            tile_max = -CAKE_INF;
                        }
                        float _max_0 = max_noftz(tile_max, row_max);
                        float new_max = _max_0;
                        float safe_max = ((new_max == -CAKE_INF) ? 0.0f : new_max);
                        float new_max_scaled = safe_max * softmax_scale_log2;
                        float _fma_0 = __fmaf_rn(row_max, softmax_scale_log2, -new_max_scaled);
                        float acc_scale_log2 = _fma_0;
                        float acc_scale;
                        float selected_max;
                        if (acc_scale_log2 >= -8.0f) {
                            selected_max = row_max;
                            safe_max = ((row_max == -CAKE_INF) ? 0.0f : row_max);
                            acc_scale = 1.0f;
                            new_max_scaled = safe_max * softmax_scale_log2;
                        } else {
                            selected_max = new_max;
                            float _exp2_0 = approx_exp2(acc_scale_log2);
                            acc_scale = ((row_max > -CAKE_INF) ? _exp2_0 : 1.0f);
                        }
                        row_max = selected_max;
                        scale_smem[stat_slot] = acc_scale;
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive(corr_sig_addr);
                        float score_bias = ((valid_cols > 0) ? -new_max_scaled : -CAKE_INF);
                        const float2 _fma_b2_9 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_10 = {score_bias, score_bias};
                        #pragma unroll
                        for (int _lf = 0; _lf < 64; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_0)[_lf], _fma_b2_9, _fma_c2_10);
                        #pragma unroll
                        for (int _le = 0; _le < 64; _le++) {
                            _tmem_load_0[_le] = approx_exp2(_tmem_load_0[_le]);
                        }
                        unsigned int packed_p_lo[32];
                        #pragma unroll
                        for (int _lp = 0; _lp < 32; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                            packed_p_lo[_lp] = *(uint32_t*)&_bf2;
                        }
                        int p_addr = taddr + (unsigned int)TMEM_PROBABILITIES_OFFSET + (unsigned int)(tmem_row_origin << 16);
                        mbarrier_wait(p_empty_addr, _phase_p_empty_0);
                        _phase_p_empty_0 ^= 1;
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x32.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                            :: "r"(p_addr), "r"(packed_p_lo[0]), "r"(packed_p_lo[1]), "r"(packed_p_lo[2]), "r"(packed_p_lo[3]), "r"(packed_p_lo[4]), "r"(packed_p_lo[5]), "r"(packed_p_lo[6]), "r"(packed_p_lo[7]), "r"(packed_p_lo[8]), "r"(packed_p_lo[9]), "r"(packed_p_lo[10]), "r"(packed_p_lo[11]), "r"(packed_p_lo[12]), "r"(packed_p_lo[13]), "r"(packed_p_lo[14]), "r"(packed_p_lo[15]), "r"(packed_p_lo[16]), "r"(packed_p_lo[17]), "r"(packed_p_lo[18]), "r"(packed_p_lo[19]), "r"(packed_p_lo[20]), "r"(packed_p_lo[21]), "r"(packed_p_lo[22]), "r"(packed_p_lo[23]), "r"(packed_p_lo[24]), "r"(packed_p_lo[25]), "r"(packed_p_lo[26]), "r"(packed_p_lo[27]), "r"(packed_p_lo[28]), "r"(packed_p_lo[29]), "r"(packed_p_lo[30]), "r"(packed_p_lo[31]));
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        mbarrier_arrive(p_full_addr);
                        #pragma unroll
                        for (int _le = 0; _le < 64; _le++) {
                            _tmem_load_0[_le + 64] = approx_exp2(_tmem_load_0[_le + 64]);
                        }
                        unsigned int packed_p_hi[32];
                        #pragma unroll
                        for (int _lp = 0; _lp < 32; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 64], _tmem_load_0[_lp*2+1 + 64]));
                            packed_p_hi[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x32.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                            :: "r"(p_addr + 32), "r"(packed_p_hi[0]), "r"(packed_p_hi[1]), "r"(packed_p_hi[2]), "r"(packed_p_hi[3]), "r"(packed_p_hi[4]), "r"(packed_p_hi[5]), "r"(packed_p_hi[6]), "r"(packed_p_hi[7]), "r"(packed_p_hi[8]), "r"(packed_p_hi[9]), "r"(packed_p_hi[10]), "r"(packed_p_hi[11]), "r"(packed_p_hi[12]), "r"(packed_p_hi[13]), "r"(packed_p_hi[14]), "r"(packed_p_hi[15]), "r"(packed_p_hi[16]), "r"(packed_p_hi[17]), "r"(packed_p_hi[18]), "r"(packed_p_hi[19]), "r"(packed_p_hi[20]), "r"(packed_p_hi[21]), "r"(packed_p_hi[22]), "r"(packed_p_hi[23]), "r"(packed_p_hi[24]), "r"(packed_p_hi[25]), "r"(packed_p_hi[26]), "r"(packed_p_hi[27]), "r"(packed_p_hi[28]), "r"(packed_p_hi[29]), "r"(packed_p_hi[30]), "r"(packed_p_hi[31]));
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        mbarrier_arrive(p_lastsplit_addr);
                        float2 _reg_reduce_sum2_11 = make_float2(0.0f, 0.0f);
                        softmax_block_sum(&_tmem_load_0[0], &_reg_reduce_sum2_11);
                        softmax_block_sum(&_tmem_load_0[32], &_reg_reduce_sum2_11);
                        softmax_block_sum(&_tmem_load_0[64], &_reg_reduce_sum2_11);
                        softmax_block_sum(&_tmem_load_0[96], &_reg_reduce_sum2_11);
                        float _tmem_load_0_sum = _reg_reduce_sum2_11.x + _reg_reduce_sum2_11.y;
                        float block_sum = _tmem_load_0_sum;
                        mbarrier_wait(corr_done_addr, _phase_corr_done_0);
                        _phase_corr_done_0 ^= 1;
                        row_sum = row_sum * acc_scale + block_sum;
                    }
                    scale_smem[128 + stat_slot] = row_sum;
                    scale_smem[256 + stat_slot] = row_max;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(corr_sig_addr);
                }
            }
        }
    }
    // ---- Role: correction ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 224;");
        { // correction_main
            unsigned int work_stage_c = 0;
            mbarrier_arrive(tile_done_addr);
            unsigned int _phase_work_full_1 = 0;
            unsigned int _phase_corr_sig_0 = 0;
            unsigned int _phase_o_full_0 = 0;
            #pragma unroll 1
            for (int tile_iter_1 = 0; tile_iter_1 < total_tiles; tile_iter_1++) {
                unsigned int valid_1 = 1;
                unsigned int gate_t_1 = 1;
                int q_block_t_1 = 0;
                int head_t_1 = 0;
                int row_begin_t_1 = 0;
                int count_t_1 = 0;
                int groups_t_1 = 0;
                if (tile_iter_1 == 0) {
                    int bid_t_1 = blockIdx.x;
                    q_block_t_1 = bid_t_1 % query_blocks;
                    head_t_1 = bid_t_1 / query_blocks;
                    row_begin_t_1 = indptr[q_block_t_1];
                    count_t_1 = indptr[q_block_t_1 + 1] - row_begin_t_1;
                    groups_t_1 = (count_t_1 + 4 - 1) / 4;
                    if (bid_t_1 >= total_tiles) {
                        valid_1 = 0;
                    }
                } else {
                    mbarrier_wait(work_full_addr + (work_stage_c) * 8, _phase_work_full_1);
                    unsigned int base_1 = work_stage_c * 8;
                    valid_1 = work_token_words[base_1];
                    unsigned int q_block_u_1 = work_token_words[base_1 + 1];
                    unsigned int head_u_1 = work_token_words[base_1 + 2];
                    unsigned int row_begin_u_1 = work_token_words[base_1 + 3];
                    unsigned int count_u_1 = work_token_words[base_1 + 4];
                    unsigned int groups_u_1 = work_token_words[base_1 + 5];
                    gate_t_1 = work_token_words[base_1 + 7];
                    mbarrier_arrive(work_empty_addr + (work_stage_c) * 8);
                    work_stage_c += 1;
                    if (work_stage_c == 4) { work_stage_c = 0; _phase_work_full_1 ^= 1; }
                    q_block_t_1 = (int)q_block_u_1;
                    head_t_1 = (int)head_u_1;
                    row_begin_t_1 = (int)row_begin_u_1;
                    count_t_1 = (int)count_u_1;
                    groups_t_1 = (int)groups_u_1;
                }
                int tile_t_1 = q_block_t_1 + head_t_1 * query_blocks;
                unsigned int valid_c = valid_1;
                if (valid_c == 0) {
                    break;
                }
                if (valid_c != 0) {
                    int q_block_1 = q_block_t_1;
                    int head_1 = head_t_1;
                    int query_base_1 = q_block_1 * 64;
                    int q_valid_1 = sequence_q - query_base_1;
                    if (q_valid_1 > 64) {
                        q_valid_1 = 64;
                    }
                    if (q_valid_1 < 0) {
                        q_valid_1 = 0;
                    }
                    const int warp_in_role_1 = warp - 4;
                    const int tmem_row_origin_1 = warp_in_role_1 * 32;
                    int my_row_1 = warp_in_role_1 % 2 * 32 + lane;
                    int stat_slot_1 = warp_in_role_1 * 32 + lane;
                    int partner_slot = (warp_in_role_1 ^ 2) * 32 + lane;
                    int row_addr = tmem_row_origin_1 << 16;
                    mbarrier_arrive(p_full_addr);
                    mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
                    _phase_corr_sig_0 ^= 1;
                    mbarrier_arrive(corr_done_addr);
                    #pragma unroll 1
                    for (int _group_index = 1; _group_index < groups_t_1; _group_index++) {
                        mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
                        _phase_corr_sig_0 ^= 1;
                        float acc_scale_1 = scale_smem[stat_slot_1];
                        int _vote_0 = __any_sync(0xFFFFFFFF, acc_scale_1 < 1.0f);
                        if (_vote_0 != 0) {
                            float _tmem_load_1[128];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31])
                                : "r"(taddr + 128 + (unsigned int)row_addr));
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(_tmem_load_1[32]), "=f"(_tmem_load_1[33]), "=f"(_tmem_load_1[34]), "=f"(_tmem_load_1[35]), "=f"(_tmem_load_1[36]), "=f"(_tmem_load_1[37]), "=f"(_tmem_load_1[38]), "=f"(_tmem_load_1[39]), "=f"(_tmem_load_1[40]), "=f"(_tmem_load_1[41]), "=f"(_tmem_load_1[42]), "=f"(_tmem_load_1[43]), "=f"(_tmem_load_1[44]), "=f"(_tmem_load_1[45]), "=f"(_tmem_load_1[46]), "=f"(_tmem_load_1[47]), "=f"(_tmem_load_1[48]), "=f"(_tmem_load_1[49]), "=f"(_tmem_load_1[50]), "=f"(_tmem_load_1[51]), "=f"(_tmem_load_1[52]), "=f"(_tmem_load_1[53]), "=f"(_tmem_load_1[54]), "=f"(_tmem_load_1[55]), "=f"(_tmem_load_1[56]), "=f"(_tmem_load_1[57]), "=f"(_tmem_load_1[58]), "=f"(_tmem_load_1[59]), "=f"(_tmem_load_1[60]), "=f"(_tmem_load_1[61]), "=f"(_tmem_load_1[62]), "=f"(_tmem_load_1[63])
                                : "r"(taddr + 128 + (unsigned int)row_addr + 32));
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(_tmem_load_1[64]), "=f"(_tmem_load_1[65]), "=f"(_tmem_load_1[66]), "=f"(_tmem_load_1[67]), "=f"(_tmem_load_1[68]), "=f"(_tmem_load_1[69]), "=f"(_tmem_load_1[70]), "=f"(_tmem_load_1[71]), "=f"(_tmem_load_1[72]), "=f"(_tmem_load_1[73]), "=f"(_tmem_load_1[74]), "=f"(_tmem_load_1[75]), "=f"(_tmem_load_1[76]), "=f"(_tmem_load_1[77]), "=f"(_tmem_load_1[78]), "=f"(_tmem_load_1[79]), "=f"(_tmem_load_1[80]), "=f"(_tmem_load_1[81]), "=f"(_tmem_load_1[82]), "=f"(_tmem_load_1[83]), "=f"(_tmem_load_1[84]), "=f"(_tmem_load_1[85]), "=f"(_tmem_load_1[86]), "=f"(_tmem_load_1[87]), "=f"(_tmem_load_1[88]), "=f"(_tmem_load_1[89]), "=f"(_tmem_load_1[90]), "=f"(_tmem_load_1[91]), "=f"(_tmem_load_1[92]), "=f"(_tmem_load_1[93]), "=f"(_tmem_load_1[94]), "=f"(_tmem_load_1[95])
                                : "r"(taddr + 128 + (unsigned int)row_addr + 64));
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(_tmem_load_1[96]), "=f"(_tmem_load_1[97]), "=f"(_tmem_load_1[98]), "=f"(_tmem_load_1[99]), "=f"(_tmem_load_1[100]), "=f"(_tmem_load_1[101]), "=f"(_tmem_load_1[102]), "=f"(_tmem_load_1[103]), "=f"(_tmem_load_1[104]), "=f"(_tmem_load_1[105]), "=f"(_tmem_load_1[106]), "=f"(_tmem_load_1[107]), "=f"(_tmem_load_1[108]), "=f"(_tmem_load_1[109]), "=f"(_tmem_load_1[110]), "=f"(_tmem_load_1[111]), "=f"(_tmem_load_1[112]), "=f"(_tmem_load_1[113]), "=f"(_tmem_load_1[114]), "=f"(_tmem_load_1[115]), "=f"(_tmem_load_1[116]), "=f"(_tmem_load_1[117]), "=f"(_tmem_load_1[118]), "=f"(_tmem_load_1[119]), "=f"(_tmem_load_1[120]), "=f"(_tmem_load_1[121]), "=f"(_tmem_load_1[122]), "=f"(_tmem_load_1[123]), "=f"(_tmem_load_1[124]), "=f"(_tmem_load_1[125]), "=f"(_tmem_load_1[126]), "=f"(_tmem_load_1[127])
                                : "r"(taddr + 128 + (unsigned int)row_addr + 96));
                            #if __CUDA_ARCH__ >= 1000
                            const float2 _scale2_0 = {acc_scale_1, acc_scale_1};
                            #pragma unroll
                            for (int _ls = 0; _ls < 64; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_1)[_ls], _scale2_0);
                            #else
                            #pragma unroll
                            for (int _ls = 0; _ls < 128; _ls++) {
                                _tmem_load_1[_ls] = _tmem_load_1[_ls] * acc_scale_1;
                            }
                            #endif
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x128.b32"
                                " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127, %128};"
                                :: "r"(taddr + 128 + (unsigned int)row_addr), "f"(_tmem_load_1[0]), "f"(_tmem_load_1[1]), "f"(_tmem_load_1[2]), "f"(_tmem_load_1[3]), "f"(_tmem_load_1[4]), "f"(_tmem_load_1[5]), "f"(_tmem_load_1[6]), "f"(_tmem_load_1[7]), "f"(_tmem_load_1[8]), "f"(_tmem_load_1[9]), "f"(_tmem_load_1[10]), "f"(_tmem_load_1[11]), "f"(_tmem_load_1[12]), "f"(_tmem_load_1[13]), "f"(_tmem_load_1[14]), "f"(_tmem_load_1[15]), "f"(_tmem_load_1[16]), "f"(_tmem_load_1[17]), "f"(_tmem_load_1[18]), "f"(_tmem_load_1[19]), "f"(_tmem_load_1[20]), "f"(_tmem_load_1[21]), "f"(_tmem_load_1[22]), "f"(_tmem_load_1[23]), "f"(_tmem_load_1[24]), "f"(_tmem_load_1[25]), "f"(_tmem_load_1[26]), "f"(_tmem_load_1[27]), "f"(_tmem_load_1[28]), "f"(_tmem_load_1[29]), "f"(_tmem_load_1[30]), "f"(_tmem_load_1[31]), "f"(_tmem_load_1[32]), "f"(_tmem_load_1[33]), "f"(_tmem_load_1[34]), "f"(_tmem_load_1[35]), "f"(_tmem_load_1[36]), "f"(_tmem_load_1[37]), "f"(_tmem_load_1[38]), "f"(_tmem_load_1[39]), "f"(_tmem_load_1[40]), "f"(_tmem_load_1[41]), "f"(_tmem_load_1[42]), "f"(_tmem_load_1[43]), "f"(_tmem_load_1[44]), "f"(_tmem_load_1[45]), "f"(_tmem_load_1[46]), "f"(_tmem_load_1[47]), "f"(_tmem_load_1[48]), "f"(_tmem_load_1[49]), "f"(_tmem_load_1[50]), "f"(_tmem_load_1[51]), "f"(_tmem_load_1[52]), "f"(_tmem_load_1[53]), "f"(_tmem_load_1[54]), "f"(_tmem_load_1[55]), "f"(_tmem_load_1[56]), "f"(_tmem_load_1[57]), "f"(_tmem_load_1[58]), "f"(_tmem_load_1[59]), "f"(_tmem_load_1[60]), "f"(_tmem_load_1[61]), "f"(_tmem_load_1[62]), "f"(_tmem_load_1[63]), "f"(_tmem_load_1[64]), "f"(_tmem_load_1[65]), "f"(_tmem_load_1[66]), "f"(_tmem_load_1[67]), "f"(_tmem_load_1[68]), "f"(_tmem_load_1[69]), "f"(_tmem_load_1[70]), "f"(_tmem_load_1[71]), "f"(_tmem_load_1[72]), "f"(_tmem_load_1[73]), "f"(_tmem_load_1[74]), "f"(_tmem_load_1[75]), "f"(_tmem_load_1[76]), "f"(_tmem_load_1[77]), "f"(_tmem_load_1[78]), "f"(_tmem_load_1[79]), "f"(_tmem_load_1[80]), "f"(_tmem_load_1[81]), "f"(_tmem_load_1[82]), "f"(_tmem_load_1[83]), "f"(_tmem_load_1[84]), "f"(_tmem_load_1[85]), "f"(_tmem_load_1[86]), "f"(_tmem_load_1[87]), "f"(_tmem_load_1[88]), "f"(_tmem_load_1[89]), "f"(_tmem_load_1[90]), "f"(_tmem_load_1[91]), "f"(_tmem_load_1[92]), "f"(_tmem_load_1[93]), "f"(_tmem_load_1[94]), "f"(_tmem_load_1[95]), "f"(_tmem_load_1[96]), "f"(_tmem_load_1[97]), "f"(_tmem_load_1[98]), "f"(_tmem_load_1[99]), "f"(_tmem_load_1[100]), "f"(_tmem_load_1[101]), "f"(_tmem_load_1[102]), "f"(_tmem_load_1[103]), "f"(_tmem_load_1[104]), "f"(_tmem_load_1[105]), "f"(_tmem_load_1[106]), "f"(_tmem_load_1[107]), "f"(_tmem_load_1[108]), "f"(_tmem_load_1[109]), "f"(_tmem_load_1[110]), "f"(_tmem_load_1[111]), "f"(_tmem_load_1[112]), "f"(_tmem_load_1[113]), "f"(_tmem_load_1[114]), "f"(_tmem_load_1[115]), "f"(_tmem_load_1[116]), "f"(_tmem_load_1[117]), "f"(_tmem_load_1[118]), "f"(_tmem_load_1[119]), "f"(_tmem_load_1[120]), "f"(_tmem_load_1[121]), "f"(_tmem_load_1[122]), "f"(_tmem_load_1[123]), "f"(_tmem_load_1[124]), "f"(_tmem_load_1[125]), "f"(_tmem_load_1[126]), "f"(_tmem_load_1[127]));
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        }
                        mbarrier_arrive(p_full_addr);
                        mbarrier_arrive(corr_done_addr);
                    }
                    mbarrier_wait(o_full_addr, _phase_o_full_0);
                    _phase_o_full_0 ^= 1;
                    mbarrier_wait(corr_sig_addr, _phase_corr_sig_0);
                    _phase_corr_sig_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float final_sum = scale_smem[128 + stat_slot_1];
                    float final_max = scale_smem[256 + stat_slot_1];
                    float partner_sum = scale_smem[128 + partner_slot];
                    float partner_max = scale_smem[256 + partner_slot];
                    float _max_1 = max_noftz(final_max, partner_max);
                    float max_total = _max_1;
                    float max_total_safe = ((max_total == -CAKE_INF) ? 0.0f : max_total);
                    float _exp2_1 = approx_exp2((final_max - max_total_safe) * softmax_scale_log2);
                    float local_scale = ((final_sum > 0.0f) ? _exp2_1 : 0.0f);
                    float _exp2_2 = approx_exp2((partner_max - max_total_safe) * softmax_scale_log2);
                    float partner_scale = ((partner_sum > 0.0f) ? _exp2_2 : 0.0f);
                    float sum_total = final_sum * local_scale + partner_sum * partner_scale;
                    float _rcp_0 = approx_rcp(sum_total);
                    float local_weight = ((sum_total > 0.0f) ? local_scale * _rcp_0 : 0.0f);
                    float _tmem_load_2[128];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_2[0]), "=f"(_tmem_load_2[1]), "=f"(_tmem_load_2[2]), "=f"(_tmem_load_2[3]), "=f"(_tmem_load_2[4]), "=f"(_tmem_load_2[5]), "=f"(_tmem_load_2[6]), "=f"(_tmem_load_2[7]), "=f"(_tmem_load_2[8]), "=f"(_tmem_load_2[9]), "=f"(_tmem_load_2[10]), "=f"(_tmem_load_2[11]), "=f"(_tmem_load_2[12]), "=f"(_tmem_load_2[13]), "=f"(_tmem_load_2[14]), "=f"(_tmem_load_2[15]), "=f"(_tmem_load_2[16]), "=f"(_tmem_load_2[17]), "=f"(_tmem_load_2[18]), "=f"(_tmem_load_2[19]), "=f"(_tmem_load_2[20]), "=f"(_tmem_load_2[21]), "=f"(_tmem_load_2[22]), "=f"(_tmem_load_2[23]), "=f"(_tmem_load_2[24]), "=f"(_tmem_load_2[25]), "=f"(_tmem_load_2[26]), "=f"(_tmem_load_2[27]), "=f"(_tmem_load_2[28]), "=f"(_tmem_load_2[29]), "=f"(_tmem_load_2[30]), "=f"(_tmem_load_2[31])
                        : "r"(taddr + 128 + (unsigned int)row_addr));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_2[32]), "=f"(_tmem_load_2[33]), "=f"(_tmem_load_2[34]), "=f"(_tmem_load_2[35]), "=f"(_tmem_load_2[36]), "=f"(_tmem_load_2[37]), "=f"(_tmem_load_2[38]), "=f"(_tmem_load_2[39]), "=f"(_tmem_load_2[40]), "=f"(_tmem_load_2[41]), "=f"(_tmem_load_2[42]), "=f"(_tmem_load_2[43]), "=f"(_tmem_load_2[44]), "=f"(_tmem_load_2[45]), "=f"(_tmem_load_2[46]), "=f"(_tmem_load_2[47]), "=f"(_tmem_load_2[48]), "=f"(_tmem_load_2[49]), "=f"(_tmem_load_2[50]), "=f"(_tmem_load_2[51]), "=f"(_tmem_load_2[52]), "=f"(_tmem_load_2[53]), "=f"(_tmem_load_2[54]), "=f"(_tmem_load_2[55]), "=f"(_tmem_load_2[56]), "=f"(_tmem_load_2[57]), "=f"(_tmem_load_2[58]), "=f"(_tmem_load_2[59]), "=f"(_tmem_load_2[60]), "=f"(_tmem_load_2[61]), "=f"(_tmem_load_2[62]), "=f"(_tmem_load_2[63])
                        : "r"(taddr + 128 + (unsigned int)row_addr + 32));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_2[64]), "=f"(_tmem_load_2[65]), "=f"(_tmem_load_2[66]), "=f"(_tmem_load_2[67]), "=f"(_tmem_load_2[68]), "=f"(_tmem_load_2[69]), "=f"(_tmem_load_2[70]), "=f"(_tmem_load_2[71]), "=f"(_tmem_load_2[72]), "=f"(_tmem_load_2[73]), "=f"(_tmem_load_2[74]), "=f"(_tmem_load_2[75]), "=f"(_tmem_load_2[76]), "=f"(_tmem_load_2[77]), "=f"(_tmem_load_2[78]), "=f"(_tmem_load_2[79]), "=f"(_tmem_load_2[80]), "=f"(_tmem_load_2[81]), "=f"(_tmem_load_2[82]), "=f"(_tmem_load_2[83]), "=f"(_tmem_load_2[84]), "=f"(_tmem_load_2[85]), "=f"(_tmem_load_2[86]), "=f"(_tmem_load_2[87]), "=f"(_tmem_load_2[88]), "=f"(_tmem_load_2[89]), "=f"(_tmem_load_2[90]), "=f"(_tmem_load_2[91]), "=f"(_tmem_load_2[92]), "=f"(_tmem_load_2[93]), "=f"(_tmem_load_2[94]), "=f"(_tmem_load_2[95])
                        : "r"(taddr + 128 + (unsigned int)row_addr + 64));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_2[96]), "=f"(_tmem_load_2[97]), "=f"(_tmem_load_2[98]), "=f"(_tmem_load_2[99]), "=f"(_tmem_load_2[100]), "=f"(_tmem_load_2[101]), "=f"(_tmem_load_2[102]), "=f"(_tmem_load_2[103]), "=f"(_tmem_load_2[104]), "=f"(_tmem_load_2[105]), "=f"(_tmem_load_2[106]), "=f"(_tmem_load_2[107]), "=f"(_tmem_load_2[108]), "=f"(_tmem_load_2[109]), "=f"(_tmem_load_2[110]), "=f"(_tmem_load_2[111]), "=f"(_tmem_load_2[112]), "=f"(_tmem_load_2[113]), "=f"(_tmem_load_2[114]), "=f"(_tmem_load_2[115]), "=f"(_tmem_load_2[116]), "=f"(_tmem_load_2[117]), "=f"(_tmem_load_2[118]), "=f"(_tmem_load_2[119]), "=f"(_tmem_load_2[120]), "=f"(_tmem_load_2[121]), "=f"(_tmem_load_2[122]), "=f"(_tmem_load_2[123]), "=f"(_tmem_load_2[124]), "=f"(_tmem_load_2[125]), "=f"(_tmem_load_2[126]), "=f"(_tmem_load_2[127])
                        : "r"(taddr + 128 + (unsigned int)row_addr + 96));
                    #if __CUDA_ARCH__ >= 1000
                    const float2 _scale2_1 = {local_weight, local_weight};
                    #pragma unroll
                    for (int _ls = 0; _ls < 64; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_2)[_ls], _scale2_1);
                    #else
                    #pragma unroll
                    for (int _ls = 0; _ls < 128; _ls++) {
                        _tmem_load_2[_ls] = _tmem_load_2[_ls] * local_weight;
                    }
                    #endif
                    int query = query_base_1 + my_row_1;
                    int output_row = (query * num_heads + head_1) * 128;
                    if (warp_in_role_1 >= 2) {
                        uint32_t _tmem_load_2_bf16[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 0], _tmem_load_2[_lp*2+1 + 0]));
                            _tmem_load_2_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(partial_smem_addr + (unsigned int)(my_row_1 * 256 + (my_row_1 & 7 ^ 0) * 16)), "r"(_tmem_load_2_bf16[0]), "r"(_tmem_load_2_bf16[1]), "r"(_tmem_load_2_bf16[2]), "r"(_tmem_load_2_bf16[3]) : "memory");
                        uint32_t _tmem_load_2_bf16_0[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 8], _tmem_load_2[_lp*2+1 + 8]));
                            _tmem_load_2_bf16_0[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(partial_smem_addr + (unsigned int)(my_row_1 * 256 + (my_row_1 & 7 ^ 1) * 16)), "r"(_tmem_load_2_bf16_0[0]), "r"(_tmem_load_2_bf16_0[1]), "r"(_tmem_load_2_bf16_0[2]), "r"(_tmem_load_2_bf16_0[3]) : "memory");
                        uint32_t _tmem_load_2_bf16_1[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 16], _tmem_load_2[_lp*2+1 + 16]));
                            _tmem_load_2_bf16_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(partial_smem_addr + (unsigned int)(my_row_1 * 256 + (my_row_1 & 7 ^ 2) * 16)), "r"(_tmem_load_2_bf16_1[0]), "r"(_tmem_load_2_bf16_1[1]), "r"(_tmem_load_2_bf16_1[2]), "r"(_tmem_load_2_bf16_1[3]) : "memory");
                        uint32_t _tmem_load_2_bf16_2[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 24], _tmem_load_2[_lp*2+1 + 24]));
                            _tmem_load_2_bf16_2[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(partial_smem_addr + (unsigned int)(my_row_1 * 256 + (my_row_1 & 7 ^ 3) * 16)), "r"(_tmem_load_2_bf16_2[0]), "r"(_tmem_load_2_bf16_2[1]), "r"(_tmem_load_2_bf16_2[2]), "r"(_tmem_load_2_bf16_2[3]) : "memory");
                        uint32_t _tmem_load_2_bf16_3[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 32], _tmem_load_2[_lp*2+1 + 32]));
                            _tmem_load_2_bf16_3[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(partial_smem_addr + (unsigned int)(my_row_1 * 256 + (my_row_1 & 7 ^ 4) * 16)), "r"(_tmem_load_2_bf16_3[0]), "r"(_tmem_load_2_bf16_3[1]), "r"(_tmem_load_2_bf16_3[2]), "r"(_tmem_load_2_bf16_3[3]) : "memory");
                        uint32_t _tmem_load_2_bf16_4[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 40], _tmem_load_2[_lp*2+1 + 40]));
                            _tmem_load_2_bf16_4[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(partial_smem_addr + (unsigned int)(my_row_1 * 256 + (my_row_1 & 7 ^ 5) * 16)), "r"(_tmem_load_2_bf16_4[0]), "r"(_tmem_load_2_bf16_4[1]), "r"(_tmem_load_2_bf16_4[2]), "r"(_tmem_load_2_bf16_4[3]) : "memory");
                        uint32_t _tmem_load_2_bf16_5[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 48], _tmem_load_2[_lp*2+1 + 48]));
                            _tmem_load_2_bf16_5[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(partial_smem_addr + (unsigned int)(my_row_1 * 256 + (my_row_1 & 7 ^ 6) * 16)), "r"(_tmem_load_2_bf16_5[0]), "r"(_tmem_load_2_bf16_5[1]), "r"(_tmem_load_2_bf16_5[2]), "r"(_tmem_load_2_bf16_5[3]) : "memory");
                        uint32_t _tmem_load_2_bf16_6[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 56], _tmem_load_2[_lp*2+1 + 56]));
                            _tmem_load_2_bf16_6[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(partial_smem_addr + (unsigned int)(my_row_1 * 256 + (my_row_1 & 7 ^ 7) * 16)), "r"(_tmem_load_2_bf16_6[0]), "r"(_tmem_load_2_bf16_6[1]), "r"(_tmem_load_2_bf16_6[2]), "r"(_tmem_load_2_bf16_6[3]) : "memory");
                        uint32_t _tmem_load_2_bf16_7[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 64], _tmem_load_2[_lp*2+1 + 64]));
                            _tmem_load_2_bf16_7[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(partial_smem_addr + (unsigned int)(my_row_1 * 256 + (my_row_1 & 7 ^ 8) * 16)), "r"(_tmem_load_2_bf16_7[0]), "r"(_tmem_load_2_bf16_7[1]), "r"(_tmem_load_2_bf16_7[2]), "r"(_tmem_load_2_bf16_7[3]) : "memory");
                        uint32_t _tmem_load_2_bf16_8[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 72], _tmem_load_2[_lp*2+1 + 72]));
                            _tmem_load_2_bf16_8[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(partial_smem_addr + (unsigned int)(my_row_1 * 256 + (my_row_1 & 7 ^ 9) * 16)), "r"(_tmem_load_2_bf16_8[0]), "r"(_tmem_load_2_bf16_8[1]), "r"(_tmem_load_2_bf16_8[2]), "r"(_tmem_load_2_bf16_8[3]) : "memory");
                        uint32_t _tmem_load_2_bf16_9[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 80], _tmem_load_2[_lp*2+1 + 80]));
                            _tmem_load_2_bf16_9[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(partial_smem_addr + (unsigned int)(my_row_1 * 256 + (my_row_1 & 7 ^ 10) * 16)), "r"(_tmem_load_2_bf16_9[0]), "r"(_tmem_load_2_bf16_9[1]), "r"(_tmem_load_2_bf16_9[2]), "r"(_tmem_load_2_bf16_9[3]) : "memory");
                        uint32_t _tmem_load_2_bf16_10[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 88], _tmem_load_2[_lp*2+1 + 88]));
                            _tmem_load_2_bf16_10[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(partial_smem_addr + (unsigned int)(my_row_1 * 256 + (my_row_1 & 7 ^ 11) * 16)), "r"(_tmem_load_2_bf16_10[0]), "r"(_tmem_load_2_bf16_10[1]), "r"(_tmem_load_2_bf16_10[2]), "r"(_tmem_load_2_bf16_10[3]) : "memory");
                        uint32_t _tmem_load_2_bf16_11[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 96], _tmem_load_2[_lp*2+1 + 96]));
                            _tmem_load_2_bf16_11[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(partial_smem_addr + (unsigned int)(my_row_1 * 256 + (my_row_1 & 7 ^ 12) * 16)), "r"(_tmem_load_2_bf16_11[0]), "r"(_tmem_load_2_bf16_11[1]), "r"(_tmem_load_2_bf16_11[2]), "r"(_tmem_load_2_bf16_11[3]) : "memory");
                        uint32_t _tmem_load_2_bf16_12[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 104], _tmem_load_2[_lp*2+1 + 104]));
                            _tmem_load_2_bf16_12[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(partial_smem_addr + (unsigned int)(my_row_1 * 256 + (my_row_1 & 7 ^ 13) * 16)), "r"(_tmem_load_2_bf16_12[0]), "r"(_tmem_load_2_bf16_12[1]), "r"(_tmem_load_2_bf16_12[2]), "r"(_tmem_load_2_bf16_12[3]) : "memory");
                        uint32_t _tmem_load_2_bf16_13[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 112], _tmem_load_2[_lp*2+1 + 112]));
                            _tmem_load_2_bf16_13[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(partial_smem_addr + (unsigned int)(my_row_1 * 256 + (my_row_1 & 7 ^ 14) * 16)), "r"(_tmem_load_2_bf16_13[0]), "r"(_tmem_load_2_bf16_13[1]), "r"(_tmem_load_2_bf16_13[2]), "r"(_tmem_load_2_bf16_13[3]) : "memory");
                        uint32_t _tmem_load_2_bf16_14[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 120], _tmem_load_2[_lp*2+1 + 120]));
                            _tmem_load_2_bf16_14[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(partial_smem_addr + (unsigned int)(my_row_1 * 256 + (my_row_1 & 7 ^ 15) * 16)), "r"(_tmem_load_2_bf16_14[0]), "r"(_tmem_load_2_bf16_14[1]), "r"(_tmem_load_2_bf16_14[2]), "r"(_tmem_load_2_bf16_14[3]) : "memory");
                    }
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                    if (warp_in_role_1 < 2 && my_row_1 < q_valid_1) {
                        #pragma unroll
                        for (int offset = 0; offset < 128; offset += 8) {
                            unsigned int _partial_smem_reg_0[4];
                            {
                                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(partial_smem);
                                #pragma unroll
                                for (int _lr = 0; _lr < 4; _lr++)
                                    _partial_smem_reg_0[_lr] = _smem_ptr[(my_row_1 * 64 + (my_row_1 & 7 ^ offset / 8) * 4) + _lr];
                            }
                            float _partial_smem_reg_0_f32[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&_partial_smem_reg_0_f32[_pair * 2])[0]), "=f"((&_partial_smem_reg_0_f32[_pair * 2])[1])
                                    : "r"(_partial_smem_reg_0[_pair]));
                            }
                            #pragma unroll
                            for (int elem = 0; elem < 8; elem++) {
                                _tmem_load_2[offset + elem] = _tmem_load_2[offset + elem] + _partial_smem_reg_0_f32[elem];
                            }
                            {
                                __nv_bfloat162 _pk[4];
                                _pk[0] = __floats2bfloat162_rn(_tmem_load_2[offset + 0], _tmem_load_2[offset + 1]);
                                _pk[1] = __floats2bfloat162_rn(_tmem_load_2[offset + 2], _tmem_load_2[offset + 3]);
                                _pk[2] = __floats2bfloat162_rn(_tmem_load_2[offset + 4], _tmem_load_2[offset + 5]);
                                _pk[3] = __floats2bfloat162_rn(_tmem_load_2[offset + 6], _tmem_load_2[offset + 7]);
                                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(out + (output_row + offset)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                            }
                        }
                        if (return_lse != 0) {
                            int stat_idx = query * num_heads + head_1;
                            float _log2_0;
                            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(sum_total));
                            lse[stat_idx] = ((sum_total > 0.0f) ? max_total_safe * softmax_scale_log2 * 0.6931471805599453f + _log2_0 * 0.6931471805599453f : -CAKE_INF);
                        }
                    }
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(tile_done_addr);
                }
            }
        }
    }
    // ---- Role: mma_warp ----
    if (warp == 8) {
        { // mma_warp_main
            unsigned int work_stage_m = 0;
            unsigned int kv_stage = 0;
            unsigned int kv_phase = 0;
            unsigned int _phase_work_full_2 = 0;
            unsigned int _phase_q_full_0 = 0;
            unsigned int _phase_tile_done_0 = 0;
            unsigned int _phase_s_empty_0 = 0;
            unsigned int _phase_p_full_0 = 0;
            unsigned int _phase_p_lastsplit_0 = 0;
            #pragma unroll 1
            for (int tile_iter_2 = 0; tile_iter_2 < total_tiles; tile_iter_2++) {
                unsigned int valid_2 = 1;
                unsigned int gate_t_2 = 1;
                int q_block_t_2 = 0;
                int head_t_2 = 0;
                int row_begin_t_2 = 0;
                int count_t_2 = 0;
                int groups_t_2 = 0;
                if (tile_iter_2 == 0) {
                    int bid_t_2 = blockIdx.x;
                    q_block_t_2 = bid_t_2 % query_blocks;
                    head_t_2 = bid_t_2 / query_blocks;
                    row_begin_t_2 = indptr[q_block_t_2];
                    count_t_2 = indptr[q_block_t_2 + 1] - row_begin_t_2;
                    groups_t_2 = (count_t_2 + 4 - 1) / 4;
                    if (bid_t_2 >= total_tiles) {
                        valid_2 = 0;
                    }
                } else {
                    mbarrier_wait(work_full_addr + (work_stage_m) * 8, _phase_work_full_2);
                    unsigned int base_2 = work_stage_m * 8;
                    valid_2 = work_token_words[base_2];
                    unsigned int q_block_u_2 = work_token_words[base_2 + 1];
                    unsigned int head_u_2 = work_token_words[base_2 + 2];
                    unsigned int row_begin_u_2 = work_token_words[base_2 + 3];
                    unsigned int count_u_2 = work_token_words[base_2 + 4];
                    unsigned int groups_u_2 = work_token_words[base_2 + 5];
                    gate_t_2 = work_token_words[base_2 + 7];
                    mbarrier_arrive(work_empty_addr + (work_stage_m) * 8);
                    work_stage_m += 1;
                    if (work_stage_m == 4) { work_stage_m = 0; _phase_work_full_2 ^= 1; }
                    q_block_t_2 = (int)q_block_u_2;
                    head_t_2 = (int)head_u_2;
                    row_begin_t_2 = (int)row_begin_u_2;
                    count_t_2 = (int)count_u_2;
                    groups_t_2 = (int)groups_u_2;
                }
                int tile_t_2 = q_block_t_2 + head_t_2 * query_blocks;
                unsigned int valid_m = valid_2;
                if (valid_m == 0) {
                    break;
                }
                if (valid_m != 0) {
                    int q_block_2 = q_block_t_2;
                    int head_2 = head_t_2;
                    int query_base_2 = q_block_2 * 64;
                    int q_valid_2 = sequence_q - query_base_2;
                    if (q_valid_2 > 64) {
                        q_valid_2 = 64;
                    }
                    if (q_valid_2 < 0) {
                        q_valid_2 = 0;
                    }
                    mbarrier_wait(q_full_addr, _phase_q_full_0);
                    _phase_q_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int first_pv = 1;
                    unsigned int k_stage = kv_stage;
                    unsigned int k_phase = kv_phase;
                    kv_stage += 1;
                    if (kv_stage == 3) { kv_stage = 0; kv_phase ^= 1; }
                    mbarrier_wait(kv_full_addr + (k_stage) * 8, k_phase);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_0 = make_warp_uniform(((q_smem_addr) >> 4) & 0x3FFF);
                    int _mma_b_lo_0 = make_warp_uniform((((kv_smem_addr) >> 4) & 0x3FFF) + (k_stage) * 4096);
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
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 71304336;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 2042;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_scores), "r"(0));
                    if (groups_t_2 == 1) {
                        elect_commit2(s_full_addr, q_empty_addr);
                    } else {
                        elect_commit(s_full_addr);
                    }
                    elect_commit(kv_empty_addr + (k_stage) * 8);
                    mbarrier_wait(tile_done_addr, _phase_tile_done_0);
                    _phase_tile_done_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    #pragma unroll 1
                    for (int group_index_1 = 0; group_index_1 < groups_t_2 - 1; group_index_1++) {
                        mbarrier_wait(s_empty_addr, _phase_s_empty_0);
                        _phase_s_empty_0 ^= 1;
                        k_stage = kv_stage;
                        k_phase = kv_phase;
                        kv_stage += 1;
                        if (kv_stage == 3) { kv_stage = 0; kv_phase ^= 1; }
                        mbarrier_wait(kv_full_addr + (k_stage) * 8, k_phase);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_1 = make_warp_uniform(((q_smem_addr) >> 4) & 0x3FFF);
                        int _mma_b_lo_1 = make_warp_uniform((((kv_smem_addr) >> 4) & 0x3FFF) + (k_stage) * 4096);
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
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 71304336;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 2042;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"(tmem_scores), "r"(0));
                        if (group_index_1 + 2 == groups_t_2) {
                            elect_commit2(s_full_addr, q_empty_addr);
                        } else {
                            elect_commit(s_full_addr);
                        }
                        elect_commit(kv_empty_addr + (k_stage) * 8);
                        unsigned int v_stage = kv_stage;
                        unsigned int v_phase = kv_phase;
                        kv_stage += 1;
                        if (kv_stage == 3) { kv_stage = 0; kv_phase ^= 1; }
                        mbarrier_wait(kv_full_addr + (v_stage) * 8, v_phase);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        mbarrier_wait(p_full_addr, _phase_p_full_0);
                        _phase_p_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_b_lo_2 = make_warp_uniform(((((v_smem_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage) * 4096);
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
                    "mov.b32 id, 71369872;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output), "r"(_mma_b_lo_2), "r"(tmem_probabilities), "r"(((first_pv) ? 0 : 1)));
                        mbarrier_wait(p_lastsplit_addr, _phase_p_lastsplit_0);
                        _phase_p_lastsplit_0 ^= 1;
                        int _mma_b_lo_3 = make_warp_uniform(((((v_smem_addr) >> 4) & 0x3FFF) | 0x4000000) + (v_stage) * 4096);
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
                    "mov.b32 id, 71369872;\n\t"
                    "add.u32 ta, %2, 32;\n\t"
                    "add.u32 blo, %1, 512;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output), "r"(_mma_b_lo_3), "r"(tmem_probabilities), "r"(1));
                        first_pv = 0;
                        elect_commit(p_empty_addr);
                        elect_commit(kv_empty_addr + (v_stage) * 8);
                    }
                    unsigned int final_v_stage = kv_stage;
                    unsigned int final_v_phase = kv_phase;
                    kv_stage += 1;
                    if (kv_stage == 3) { kv_stage = 0; kv_phase ^= 1; }
                    mbarrier_wait(kv_full_addr + (final_v_stage) * 8, final_v_phase);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    mbarrier_wait(p_full_addr, _phase_p_full_0);
                    _phase_p_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_b_lo_4 = make_warp_uniform(((((v_smem_addr) >> 4) & 0x3FFF) | 0x4000000) + (final_v_stage) * 4096);
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
                    "mov.b32 id, 71369872;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output), "r"(_mma_b_lo_4), "r"(tmem_probabilities), "r"(((first_pv) ? 0 : 1)));
                    mbarrier_wait(p_lastsplit_addr, _phase_p_lastsplit_0);
                    _phase_p_lastsplit_0 ^= 1;
                    int _mma_b_lo_5 = make_warp_uniform(((((v_smem_addr) >> 4) & 0x3FFF) | 0x4000000) + (final_v_stage) * 4096);
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
                    "mov.b32 id, 71369872;\n\t"
                    "add.u32 ta, %2, 32;\n\t"
                    "add.u32 blo, %1, 512;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.ws.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output), "r"(_mma_b_lo_5), "r"(tmem_probabilities), "r"(1));
                    elect_commit2(o_full_addr, p_empty_addr);
                    elect_commit(kv_empty_addr + (final_v_stage) * 8);
                    mbarrier_wait(s_empty_addr, _phase_s_empty_0);
                    _phase_s_empty_0 ^= 1;
                }
            }
            mbarrier_wait(tile_done_addr, _phase_tile_done_0);
            _phase_tile_done_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(512));
        }
    }
    // ---- Role: load_warp ----
    if (warp == 9) {
        { // load_warp_main
            unsigned int work_stage_l = 0;
            unsigned int load_stage = 0;
            unsigned int _phase_work_full_3 = 0;
            unsigned int _phase_q_empty_0 = 1;
            unsigned int _phase_kv_empty = 1;
            #pragma unroll 1
            for (int tile_iter_3 = 0; tile_iter_3 < total_tiles; tile_iter_3++) {
                unsigned int valid_3 = 1;
                unsigned int gate_t_3 = 1;
                int q_block_t_3 = 0;
                int head_t_3 = 0;
                int row_begin_t_3 = 0;
                int count_t_3 = 0;
                int groups_t_3 = 0;
                if (tile_iter_3 == 0) {
                    int bid_t_3 = blockIdx.x;
                    q_block_t_3 = bid_t_3 % query_blocks;
                    head_t_3 = bid_t_3 / query_blocks;
                    row_begin_t_3 = indptr[q_block_t_3];
                    count_t_3 = indptr[q_block_t_3 + 1] - row_begin_t_3;
                    groups_t_3 = (count_t_3 + 4 - 1) / 4;
                    if (bid_t_3 >= total_tiles) {
                        valid_3 = 0;
                    }
                } else {
                    mbarrier_wait(work_full_addr + (work_stage_l) * 8, _phase_work_full_3);
                    unsigned int base_3 = work_stage_l * 8;
                    valid_3 = work_token_words[base_3];
                    unsigned int q_block_u_3 = work_token_words[base_3 + 1];
                    unsigned int head_u_3 = work_token_words[base_3 + 2];
                    unsigned int row_begin_u_3 = work_token_words[base_3 + 3];
                    unsigned int count_u_3 = work_token_words[base_3 + 4];
                    unsigned int groups_u_3 = work_token_words[base_3 + 5];
                    gate_t_3 = work_token_words[base_3 + 7];
                    mbarrier_arrive(work_empty_addr + (work_stage_l) * 8);
                    work_stage_l += 1;
                    if (work_stage_l == 4) { work_stage_l = 0; _phase_work_full_3 ^= 1; }
                    q_block_t_3 = (int)q_block_u_3;
                    head_t_3 = (int)head_u_3;
                    row_begin_t_3 = (int)row_begin_u_3;
                    count_t_3 = (int)count_u_3;
                    groups_t_3 = (int)groups_u_3;
                }
                int tile_t_3 = q_block_t_3 + head_t_3 * query_blocks;
                unsigned int valid_l = valid_3;
                if (valid_l == 0) {
                    break;
                }
                if (valid_l != 0) {
                    int q_block_3 = q_block_t_3;
                    int head_3 = head_t_3;
                    int query_base_3 = q_block_3 * 64;
                    int q_valid_3 = sequence_q - query_base_3;
                    if (q_valid_3 > 64) {
                        q_valid_3 = 64;
                    }
                    if (q_valid_3 < 0) {
                        q_valid_3 = 0;
                    }
                    unsigned int gate_l = gate_t_3;
                    mbarrier_wait(q_empty_addr, _phase_q_empty_0);
                    _phase_q_empty_0 ^= 1;
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(q_full_addr, 16384);
                        tma_4d_gmem2smem(q_smem_addr, (&q), 0, head_3, query_base_3, 0, q_full_addr);
                    }
                    int entry0 = 0;
                    int entry1 = entry0 + 1;
                    int entry2 = entry0 + 2;
                    int entry3 = entry0 + 3;
                    int block0 = indices[row_begin_t_3 + entry0];
                    int block1 = block0;
                    if (entry1 < count_t_3) {
                        block1 = indices[row_begin_t_3 + entry1];
                    }
                    int block2 = block1;
                    if (entry2 < count_t_3) {
                        block2 = indices[row_begin_t_3 + entry2];
                    }
                    int block3 = block2;
                    if (entry3 < count_t_3) {
                        block3 = indices[row_begin_t_3 + entry3];
                    }
                    mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 65536);
                        tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536, (&k), 0, block0 * 64, 0, head_3, kv_full_addr + (load_stage) * 8);
                        tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 32768, (&k), 0, block0 * 64, 1, head_3, kv_full_addr + (load_stage) * 8);
                        tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 16384, (&k), 0, block1 * 64, 0, head_3, kv_full_addr + (load_stage) * 8);
                        tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 49152, (&k), 0, block1 * 64, 1, head_3, kv_full_addr + (load_stage) * 8);
                        tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 8192, (&k), 0, block2 * 64, 0, head_3, kv_full_addr + (load_stage) * 8);
                        tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 40960, (&k), 0, block2 * 64, 1, head_3, kv_full_addr + (load_stage) * 8);
                        tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 24576, (&k), 0, block3 * 64, 0, head_3, kv_full_addr + (load_stage) * 8);
                        tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 57344, (&k), 0, block3 * 64, 1, head_3, kv_full_addr + (load_stage) * 8);
                    }
                    load_stage += 1;
                    if (load_stage == 3) { load_stage = 0; _phase_kv_empty ^= 1; }
                    if (groups_t_3 > 1) {
                        int entry0_0 = 4;
                        int entry1_1 = entry0_0 + 1;
                        int entry2_2 = entry0_0 + 2;
                        int entry3_3 = entry0_0 + 3;
                        int block0_4 = indices[row_begin_t_3 + entry0_0];
                        int block1_5 = block0_4;
                        if (entry1_1 < count_t_3) {
                            block1_5 = indices[row_begin_t_3 + entry1_1];
                        }
                        int block2_6 = block1_5;
                        if (entry2_2 < count_t_3) {
                            block2_6 = indices[row_begin_t_3 + entry2_2];
                        }
                        int block3_7 = block2_6;
                        if (entry3_3 < count_t_3) {
                            block3_7 = indices[row_begin_t_3 + entry3_3];
                        }
                        mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 65536);
                            tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536, (&k), 0, block0_4 * 64, 0, head_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 32768, (&k), 0, block0_4 * 64, 1, head_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 16384, (&k), 0, block1_5 * 64, 0, head_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 49152, (&k), 0, block1_5 * 64, 1, head_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 8192, (&k), 0, block2_6 * 64, 0, head_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 40960, (&k), 0, block2_6 * 64, 1, head_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 24576, (&k), 0, block3_7 * 64, 0, head_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 57344, (&k), 0, block3_7 * 64, 1, head_3, kv_full_addr + (load_stage) * 8);
                        }
                        load_stage += 1;
                        if (load_stage == 3) { load_stage = 0; _phase_kv_empty ^= 1; }
                        #pragma unroll 1
                        for (int group_index_2 = 0; group_index_2 < groups_t_3 - 2; group_index_2++) {
                            int entry0_1 = group_index_2 * 4;
                            int entry1_2 = entry0_1 + 1;
                            int entry2_3 = entry0_1 + 2;
                            int entry3_4 = entry0_1 + 3;
                            int block0_5 = indices[row_begin_t_3 + entry0_1];
                            int block1_6 = block0_5;
                            if (entry1_2 < count_t_3) {
                                block1_6 = indices[row_begin_t_3 + entry1_2];
                            }
                            int block2_7 = block1_6;
                            if (entry2_3 < count_t_3) {
                                block2_7 = indices[row_begin_t_3 + entry2_3];
                            }
                            int block3_8 = block2_7;
                            if (entry3_4 < count_t_3) {
                                block3_8 = indices[row_begin_t_3 + entry3_4];
                            }
                            mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                            if (elect_sync()) {
                                mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 65536);
                                tma_4d_gmem2smem(v_smem_addr + load_stage * 65536, (&v), 0, block0_5 * 64, 0, head_3, kv_full_addr + (load_stage) * 8);
                                tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 16384, (&v), 0, block0_5 * 64, 1, head_3, kv_full_addr + (load_stage) * 8);
                                tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 32768, (&v), 0, block1_6 * 64, 0, head_3, kv_full_addr + (load_stage) * 8);
                                tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 49152, (&v), 0, block1_6 * 64, 1, head_3, kv_full_addr + (load_stage) * 8);
                                tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 8192, (&v), 0, block2_7 * 64, 0, head_3, kv_full_addr + (load_stage) * 8);
                                tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 24576, (&v), 0, block2_7 * 64, 1, head_3, kv_full_addr + (load_stage) * 8);
                                tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 40960, (&v), 0, block3_8 * 64, 0, head_3, kv_full_addr + (load_stage) * 8);
                                tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 57344, (&v), 0, block3_8 * 64, 1, head_3, kv_full_addr + (load_stage) * 8);
                            }
                            load_stage += 1;
                            if (load_stage == 3) { load_stage = 0; _phase_kv_empty ^= 1; }
                            int entry0_9 = (group_index_2 + 2) * 4;
                            int entry1_10 = entry0_9 + 1;
                            int entry2_11 = entry0_9 + 2;
                            int entry3_12 = entry0_9 + 3;
                            int block0_13 = indices[row_begin_t_3 + entry0_9];
                            int block1_14 = block0_13;
                            if (entry1_10 < count_t_3) {
                                block1_14 = indices[row_begin_t_3 + entry1_10];
                            }
                            int block2_15 = block1_14;
                            if (entry2_11 < count_t_3) {
                                block2_15 = indices[row_begin_t_3 + entry2_11];
                            }
                            int block3_16 = block2_15;
                            if (entry3_12 < count_t_3) {
                                block3_16 = indices[row_begin_t_3 + entry3_12];
                            }
                            mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                            if (elect_sync()) {
                                mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 65536);
                                tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536, (&k), 0, block0_13 * 64, 0, head_3, kv_full_addr + (load_stage) * 8);
                                tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 32768, (&k), 0, block0_13 * 64, 1, head_3, kv_full_addr + (load_stage) * 8);
                                tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 16384, (&k), 0, block1_14 * 64, 0, head_3, kv_full_addr + (load_stage) * 8);
                                tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 49152, (&k), 0, block1_14 * 64, 1, head_3, kv_full_addr + (load_stage) * 8);
                                tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 8192, (&k), 0, block2_15 * 64, 0, head_3, kv_full_addr + (load_stage) * 8);
                                tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 40960, (&k), 0, block2_15 * 64, 1, head_3, kv_full_addr + (load_stage) * 8);
                                tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 24576, (&k), 0, block3_16 * 64, 0, head_3, kv_full_addr + (load_stage) * 8);
                                tma_4d_gmem2smem(kv_smem_addr + load_stage * 65536 + 57344, (&k), 0, block3_16 * 64, 1, head_3, kv_full_addr + (load_stage) * 8);
                            }
                            load_stage += 1;
                            if (load_stage == 3) { load_stage = 0; _phase_kv_empty ^= 1; }
                        }
                        int entry0_8 = (groups_t_3 - 2) * 4;
                        int entry1_9 = entry0_8 + 1;
                        int entry2_10 = entry0_8 + 2;
                        int entry3_11 = entry0_8 + 3;
                        int block0_12 = indices[row_begin_t_3 + entry0_8];
                        int block1_13 = block0_12;
                        if (entry1_9 < count_t_3) {
                            block1_13 = indices[row_begin_t_3 + entry1_9];
                        }
                        int block2_14 = block1_13;
                        if (entry2_10 < count_t_3) {
                            block2_14 = indices[row_begin_t_3 + entry2_10];
                        }
                        int block3_15 = block2_14;
                        if (entry3_11 < count_t_3) {
                            block3_15 = indices[row_begin_t_3 + entry3_11];
                        }
                        mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 65536);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 65536, (&v), 0, block0_12 * 64, 0, head_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 16384, (&v), 0, block0_12 * 64, 1, head_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 32768, (&v), 0, block1_13 * 64, 0, head_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 49152, (&v), 0, block1_13 * 64, 1, head_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 8192, (&v), 0, block2_14 * 64, 0, head_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 24576, (&v), 0, block2_14 * 64, 1, head_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 40960, (&v), 0, block3_15 * 64, 0, head_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 57344, (&v), 0, block3_15 * 64, 1, head_3, kv_full_addr + (load_stage) * 8);
                        }
                        load_stage += 1;
                        if (load_stage == 3) { load_stage = 0; _phase_kv_empty ^= 1; }
                        if (gate_l != 0) {
                            if (lane == 0) {
                                mbarrier_arrive(claim_gate_addr);
                            }
                        }
                        int entry0_16 = (groups_t_3 - 1) * 4;
                        int entry1_17 = entry0_16 + 1;
                        int entry2_18 = entry0_16 + 2;
                        int entry3_19 = entry0_16 + 3;
                        int block0_20 = indices[row_begin_t_3 + entry0_16];
                        int block1_21 = block0_20;
                        if (entry1_17 < count_t_3) {
                            block1_21 = indices[row_begin_t_3 + entry1_17];
                        }
                        int block2_22 = block1_21;
                        if (entry2_18 < count_t_3) {
                            block2_22 = indices[row_begin_t_3 + entry2_18];
                        }
                        int block3_23 = block2_22;
                        if (entry3_19 < count_t_3) {
                            block3_23 = indices[row_begin_t_3 + entry3_19];
                        }
                        mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 65536);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 65536, (&v), 0, block0_20 * 64, 0, head_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 16384, (&v), 0, block0_20 * 64, 1, head_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 32768, (&v), 0, block1_21 * 64, 0, head_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 49152, (&v), 0, block1_21 * 64, 1, head_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 8192, (&v), 0, block2_22 * 64, 0, head_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 24576, (&v), 0, block2_22 * 64, 1, head_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 40960, (&v), 0, block3_23 * 64, 0, head_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 57344, (&v), 0, block3_23 * 64, 1, head_3, kv_full_addr + (load_stage) * 8);
                        }
                        load_stage += 1;
                        if (load_stage == 3) { load_stage = 0; _phase_kv_empty ^= 1; }
                    } else {
                        if (gate_l != 0) {
                            if (lane == 0) {
                                mbarrier_arrive(claim_gate_addr);
                            }
                        }
                        int entry0_0_1 = 0;
                        int entry1_1_1 = entry0_0_1 + 1;
                        int entry2_2_1 = entry0_0_1 + 2;
                        int entry3_3_1 = entry0_0_1 + 3;
                        int block0_4_1 = indices[row_begin_t_3 + entry0_0_1];
                        int block1_5_1 = block0_4_1;
                        if (entry1_1_1 < count_t_3) {
                            block1_5_1 = indices[row_begin_t_3 + entry1_1_1];
                        }
                        int block2_6_1 = block1_5_1;
                        if (entry2_2_1 < count_t_3) {
                            block2_6_1 = indices[row_begin_t_3 + entry2_2_1];
                        }
                        int block3_7_1 = block2_6_1;
                        if (entry3_3_1 < count_t_3) {
                            block3_7_1 = indices[row_begin_t_3 + entry3_3_1];
                        }
                        mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 65536);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 65536, (&v), 0, block0_4_1 * 64, 0, head_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 16384, (&v), 0, block0_4_1 * 64, 1, head_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 32768, (&v), 0, block1_5_1 * 64, 0, head_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 49152, (&v), 0, block1_5_1 * 64, 1, head_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 8192, (&v), 0, block2_6_1 * 64, 0, head_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 24576, (&v), 0, block2_6_1 * 64, 1, head_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 40960, (&v), 0, block3_7_1 * 64, 0, head_3, kv_full_addr + (load_stage) * 8);
                            tma_4d_gmem2smem(v_smem_addr + load_stage * 65536 + 57344, (&v), 0, block3_7_1 * 64, 1, head_3, kv_full_addr + (load_stage) * 8);
                        }
                        load_stage += 1;
                        if (load_stage == 3) { load_stage = 0; _phase_kv_empty ^= 1; }
                    }
                }
            }
        }
    }
    // ---- Role: scheduler ----
    if (warp == 10) {
        { // scheduler_main
            int lane_0 = lane;
            int grid_i = gridDim.x;
            int bid_1 = blockIdx.x;
            unsigned int work_stage_sched = 0;
            #pragma unroll
            for (int hb = 0; hb < 2; hb++) {
                hist_smem[hb * 32 + lane_0] = 0;
            }
            __syncwarp();
            int n_scan = (query_blocks + 31) / 32;
            int carry = indptr[0];
            int v_reg[8];
            #pragma unroll
            for (int hi = 0; hi < 8; hi++) {
                v_reg[hi] = 0;
                int qb_v = hi * 32 + lane_0;
                if (qb_v < query_blocks) {
                    v_reg[hi] = indptr[qb_v + 1];
                }
            }
            int grp_reg[8];
            #pragma unroll
            for (int hi_1 = 0; hi_1 < 8; hi_1++) {
                grp_reg[hi_1] = -1;
                int _shfl_up_0 = __shfl_up_sync(0xFFFFFFFF, v_reg[hi_1], 1, 32);
                int prev_h = _shfl_up_0;
                if (lane_0 == 0) {
                    prev_h = carry;
                }
                int qb_h = hi_1 * 32 + lane_0;
                if (qb_h < query_blocks) {
                    int cnt_h = v_reg[hi_1] - prev_h;
                    int grp_h = (cnt_h + 4 - 1) / 4;
                    if (grp_h > 63) {
                        grp_h = 63;
                    }
                    grp_reg[hi_1] = grp_h;
                    atomicAdd(&hist_smem[grp_h], 1);
                }
                int _shfl_0 = __shfl_sync(0xFFFFFFFF, v_reg[hi_1], 31);
                carry = _shfl_0;
            }
            #pragma unroll 1
            for (int hc = 8; hc < n_scan; hc += 8) {
                int v2[8];
                #pragma unroll
                for (int hj = 0; hj < 8; hj++) {
                    v2[hj] = 0;
                    int qb_c = (hc + hj) * 32 + lane_0;
                    if (qb_c < query_blocks) {
                        v2[hj] = indptr[qb_c + 1];
                    }
                }
                #pragma unroll
                for (int hj_1 = 0; hj_1 < 8; hj_1++) {
                    int _shfl_up_1 = __shfl_up_sync(0xFFFFFFFF, v2[hj_1], 1, 32);
                    int prev_c = _shfl_up_1;
                    if (lane_0 == 0) {
                        prev_c = carry;
                    }
                    int qb_c2 = (hc + hj_1) * 32 + lane_0;
                    if (qb_c2 < query_blocks) {
                        int cnt_c = v2[hj_1] - prev_c;
                        int grp_c = (cnt_c + 4 - 1) / 4;
                        if (grp_c > 63) {
                            grp_c = 63;
                        }
                        atomicAdd(&hist_smem[grp_c], 1);
                    }
                    int _shfl_1 = __shfl_sync(0xFFFFFFFF, v2[hj_1], 31);
                    carry = _shfl_1;
                }
            }
            __syncwarp();
            unsigned int h0 = hist_smem[lane_0 * 2];
            unsigned int h1 = hist_smem[lane_0 * 2 + 1];
            unsigned int lane_sum = h0 + h1;
            unsigned int _max_2 = ((h0) > (h1) ? (h0) : (h1));
            unsigned int lane_max = _max_2;
            unsigned int _vote_1 = __ballot_sync(0xFFFFFFFF, lane_max == (unsigned int)query_blocks);
            unsigned int uniform_mask = _vote_1;
            unsigned int uniform_sel = ((uniform_mask != 0) ? 1 : 0);
            unsigned int heads_u = (unsigned int)num_heads;
            uint32_t _warp_scan_sum_u32_0 = lane_sum;
            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.down.b32 t|p, %0, %1, 31, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_0) : "r"(1));
            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.down.b32 t|p, %0, %1, 31, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_0) : "r"(2));
            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.down.b32 t|p, %0, %1, 31, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_0) : "r"(4));
            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.down.b32 t|p, %0, %1, 31, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_0) : "r"(8));
            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.down.b32 t|p, %0, %1, 31, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_0) : "r"(16));
            unsigned int suffix_incl = _warp_scan_sum_u32_0;
            unsigned int excl = suffix_incl - lane_sum;
            unsigned int p1 = excl * heads_u;
            unsigned int p0 = (excl + h1) * heads_u;
            unsigned int n0 = h0 * heads_u;
            unsigned int n1 = h1 * heads_u;
            unsigned int gate_phase = 0;
            unsigned int gate_flag = 1;
            if (uniform_sel != 0) {
                gate_flag = 0;
            }
            if (grid_i >= total_tiles) {
                gate_flag = 0;
            }
            unsigned int one_u = 1;
            unsigned int below_mask = (one_u << (unsigned int)lane_0) - 1;
            unsigned int pending = 1;
            int last_pos = total_tiles + grid_i - 1;
            int n_pub = 1;
            unsigned int _phase_work_empty = 1;
            #pragma unroll 1
            for (int claim_i = 0; claim_i < total_tiles + 1; claim_i++) {
                if (pending != 0) {
                    mbarrier_wait_hint(work_empty_addr + (work_stage_sched) * 8, _phase_work_empty, 1000);
                }
                unsigned int ticket_lane0 = (unsigned int)total_tiles;
                if (grid_i < total_tiles) {
                    if (uniform_sel != 0) {
                        int t_static = bid_1 + n_pub * grid_i;
                        ticket_lane0 = (unsigned int)t_static;
                    } else {
                        if (pending != 0) {
                            mbarrier_wait_hint(claim_gate_addr, gate_phase, 1000);
                            gate_phase = gate_phase ^ 1;
                        }
                        if (lane_0 == 0) {
                            unsigned int _atomic_old_0 = atomicAdd(&queue_counters[0], 1);
                            ticket_lane0 = _atomic_old_0;
                        }
                    }
                }
                unsigned int _shfl_2 = __shfl_sync(0xFFFFFFFF, ticket_lane0, 0);
                unsigned int ticket = _shfl_2;
                if (ticket == (unsigned int)last_pos) {
                    if (lane_0 == 0) {
                        *(reinterpret_cast<unsigned int*>(queue_counters) + (0)) = 0;
                    }
                }
                unsigned int valid_tok = ((ticket < (unsigned int)total_tiles) ? 1 : 0);
                int hit = -1;
                unsigned int p_sel = p0;
                unsigned int n_sel = h0;
                if (ticket >= p0 && ticket < p0 + n0) {
                    hit = lane_0 * 2;
                }
                if (ticket >= p1 && ticket < p1 + n1) {
                    hit = lane_0 * 2 + 1;
                    p_sel = p1;
                    n_sel = h1;
                }
                unsigned int _vote_2 = __ballot_sync(0xFFFFFFFF, hit >= 0);
                unsigned int hit_mask = _vote_2;
                int _ffs_0 = __ffs(hit_mask);
                int _max_3 = ((_ffs_0 - 1) > (0) ? (_ffs_0 - 1) : (0));
                int hit_lane = _max_3;
                int _shfl_3 = __shfl_sync(0xFFFFFFFF, hit, hit_lane);
                int bucket = _shfl_3;
                unsigned int _shfl_4 = __shfl_sync(0xFFFFFFFF, p_sel, hit_lane);
                unsigned int p_b = _shfl_4;
                unsigned int _shfl_5 = __shfl_sync(0xFFFFFFFF, n_sel, hit_lane);
                unsigned int _max_4 = ((_shfl_5) > (one_u) ? (_shfl_5) : (one_u));
                unsigned int n_b = _max_4;
                unsigned int r_t = ticket - p_b;
                unsigned int head_sel = r_t / n_b;
                unsigned int j_sel = r_t - head_sel * n_b;
                unsigned int running = 0;
                int q_found = 0;
                unsigned int done_f = 0;
                #pragma unroll
                for (int fi = 0; fi < 8; fi++) {
                    if (done_f == 0) {
                        int in_b = 0;
                        if (grp_reg[fi] == bucket) {
                            in_b = 1;
                        }
                        unsigned int _vote_3 = __ballot_sync(0xFFFFFFFF, in_b != 0);
                        unsigned int m_f = _vote_3;
                        int _popc_0 = __popc(m_f);
                        unsigned int c_f = (unsigned int)_popc_0;
                        if (j_sel < running + c_f) {
                            unsigned int want_f = j_sel - running;
                            int _popc_1 = __popc(m_f & below_mask);
                            unsigned int rank_f = (unsigned int)_popc_1;
                            int mine_f = 0;
                            if (in_b != 0 && rank_f == want_f) {
                                mine_f = 1;
                            }
                            unsigned int _vote_4 = __ballot_sync(0xFFFFFFFF, mine_f != 0);
                            unsigned int sel_mask = _vote_4;
                            int _ffs_1 = __ffs(sel_mask);
                            q_found = fi * 32 + (_ffs_1 - 1);
                            done_f = 1;
                        }
                        running = running + c_f;
                    }
                }
                #pragma unroll 1
                for (int fi_1 = 8; fi_1 < n_scan; fi_1++) {
                    if (done_f == 0) {
                        int in_b_1 = 0;
                        int qb_f = fi_1 * 32 + lane_0;
                        if (qb_f < query_blocks) {
                            int cnt_f = indptr[qb_f + 1] - indptr[qb_f];
                            int grp_f = (cnt_f + 4 - 1) / 4;
                            if (grp_f > 63) {
                                grp_f = 63;
                            }
                            if (grp_f == bucket) {
                                in_b_1 = 1;
                            }
                        }
                        unsigned int _vote_5 = __ballot_sync(0xFFFFFFFF, in_b_1 != 0);
                        unsigned int m_f_1 = _vote_5;
                        int _popc_2 = __popc(m_f_1);
                        unsigned int c_f_1 = (unsigned int)_popc_2;
                        if (j_sel < running + c_f_1) {
                            unsigned int want_f_1 = j_sel - running;
                            int _popc_3 = __popc(m_f_1 & below_mask);
                            unsigned int rank_f_1 = (unsigned int)_popc_3;
                            int mine_f_1 = 0;
                            if (in_b_1 != 0 && rank_f_1 == want_f_1) {
                                mine_f_1 = 1;
                            }
                            unsigned int _vote_6 = __ballot_sync(0xFFFFFFFF, mine_f_1 != 0);
                            unsigned int sel_mask_1 = _vote_6;
                            int _ffs_2 = __ffs(sel_mask_1);
                            q_found = fi_1 * 32 + (_ffs_2 - 1);
                            done_f = 1;
                        }
                        running = running + c_f_1;
                    }
                }
                int tile_map = (int)head_sel * query_blocks + q_found;
                int tile_sel = (int)ticket;
                unsigned int skip = 0;
                if (valid_tok != 0) {
                    if (uniform_sel == 0) {
                        tile_sel = tile_map;
                        if (tile_sel < grid_i) {
                            skip = 1;
                        }
                    }
                }
                if (skip == 0) {
                    unsigned int token_base_p = work_stage_sched * 8;
                    if (valid_tok != 0) {
                        int q_block_p = tile_sel % query_blocks;
                        int head_p = tile_sel / query_blocks;
                        int row_begin_p = indptr[q_block_p];
                        int count_p = indptr[q_block_p + 1] - row_begin_p;
                        int groups_p = (count_p + 4 - 1) / 4;
                        if (lane_0 == 0) {
                            work_token_words[token_base_p + 1] = (unsigned int)q_block_p;
                            work_token_words[token_base_p + 2] = (unsigned int)head_p;
                            work_token_words[token_base_p + 3] = (unsigned int)row_begin_p;
                            work_token_words[token_base_p + 4] = (unsigned int)count_p;
                            work_token_words[token_base_p + 5] = (unsigned int)groups_p;
                            work_token_words[token_base_p + 6] = ticket;
                            work_token_words[token_base_p + 7] = gate_flag;
                        }
                    }
                    if (lane_0 == 0) {
                        work_token_words[token_base_p] = valid_tok;
                        mbarrier_arrive(work_full_addr + (work_stage_sched) * 8);
                    }
                    work_stage_sched += 1;
                    if (work_stage_sched == 4) { work_stage_sched = 0; _phase_work_empty ^= 1; }
                    if (lane_0 == 0) {
                        if (valid_tok != 0) {
                            if (debug_order != 0) {
                                *(reinterpret_cast<int*>(order_debug + (int)ticket) + (0)) = tile_sel;
                            }
                        }
                    }
                    n_pub = n_pub + 1;
                    pending = 1;
                }
                if (skip != 0) {
                    pending = 0;
                }
                if (valid_tok == 0) {
                    break;
                }
            }
            if (gate_flag == 0) {
                if (bid_1 < total_tiles) {
                    mbarrier_wait_hint(claim_gate_addr, gate_phase, 1000);
                }
            }
        }
    }
    // ---- Role: empty ----
    if (warp == 11) {
        // idle — no tasks assigned
    }

    // Cleanup
}

} // extern "C"
