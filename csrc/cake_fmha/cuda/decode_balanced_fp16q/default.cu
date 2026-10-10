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
#define TMEM_NCOLS 96
#define TMEM_TMEM_S0_OFFSET 0
#define TMEM_TMEM_S1_OFFSET 8
#define TMEM_TMEM_STATS0_OFFSET 16
#define TMEM_TMEM_STATS1_OFFSET 48
#define TMEM_TMEM_O0_OFFSET 80
#define TMEM_TMEM_O1_OFFSET 88
#define NUM_KV_PIPE_STAGES 8
#define NUM_Q_PIPE_STAGES 2
#define NUM_OP01_PIPE_STAGES 1
#define NUM_WORK_PIPE_STAGES 4
#define NUM_PG_PIPE_STAGES 6
#define SMEM_SMEM_CORR_REDUCE_OFF 1024
#define SMEM_SMEM_CORR_REDUCE_STAGE_BYTES 128
#define SMEM_SMEM_CORR_REDUCE_STRIDE 128
#define SMEM_SMEM_EXCH0_OFF 1152
#define SMEM_SMEM_EXCH0_STAGE_BYTES 256
#define SMEM_SMEM_EXCH0_STRIDE 256
#define SMEM_SMEM_EXCH1_OFF 1408
#define SMEM_SMEM_EXCH1_STAGE_BYTES 256
#define SMEM_SMEM_EXCH1_STRIDE 256
#define SMEM_SMEM_EXCH0_U32_OFF 1152
#define SMEM_SMEM_EXCH0_U32_STAGE_BYTES 256
#define SMEM_SMEM_EXCH0_U32_STRIDE 256
#define SMEM_SMEM_EXCH1_U32_OFF 1408
#define SMEM_SMEM_EXCH1_U32_STAGE_BYTES 256
#define SMEM_SMEM_EXCH1_U32_STRIDE 256
#define SMEM_SMEM_QT_OFF 1664
#define SMEM_SMEM_QT_STAGE_BYTES 1024
#define SMEM_SMEM_QT_STRIDE 1024
#define SMEM_SMEM_QT_LO_OFF 146048
#define SMEM_SMEM_QT_LO_STAGE_BYTES 1024
#define SMEM_SMEM_QT_LO_STRIDE 1024
#define SMEM_SMEM_KV_OFF 3712
#define SMEM_SMEM_KV_STAGE_BYTES 16384
#define SMEM_SMEM_KV_STRIDE 16384
#define SMEM_SMEM_V_OFF 3712
#define SMEM_SMEM_V_STAGE_BYTES 16384
#define SMEM_SMEM_V_STRIDE 16384
#define SMEM_SMEM_P0_OFF 134784
#define SMEM_SMEM_P0_STAGE_BYTES 1024
#define SMEM_SMEM_P0_STRIDE 1024
#define SMEM_SMEM_P1_OFF 136832
#define SMEM_SMEM_P1_STAGE_BYTES 1024
#define SMEM_SMEM_P1_STRIDE 1024
#define SMEM_SMEM_O_OFF 138880
#define SMEM_SMEM_O_STAGE_BYTES 1024
#define SMEM_SMEM_O_STRIDE 1024
#define SMEM_SMEM_O_U32_OFF 138880
#define SMEM_SMEM_O_U32_STAGE_BYTES 1024
#define SMEM_SMEM_O_U32_STRIDE 1024
#define SMEM_WORK_TOKEN_WORDS_OFF 139904
#define SMEM_WORK_TOKEN_WORDS_STAGE_BYTES 256
#define SMEM_WORK_TOKEN_WORDS_STRIDE 256
#define SMEM_SMEM_MERGE_FLAG_OFF 140160
#define SMEM_SMEM_MERGE_FLAG_STAGE_BYTES 16
#define SMEM_SMEM_MERGE_FLAG_STRIDE 16
#define SMEM_SCHED_SEQ_LENS_OFF 140176
#define SMEM_SCHED_SEQ_LENS_STAGE_BYTES 4096
#define SMEM_SCHED_SEQ_LENS_STRIDE 4096
#define SMEM_SMEM_PG_OFF 144384
#define SMEM_SMEM_PG_STAGE_BYTES 128
#define SMEM_SMEM_PG_STRIDE 128
#define SMEM_TOTAL 148096
#define THREADS 512
#define BLOCK_N 128
#define HEAD_DIM 128
#define TILE_Q 8
#define PAGE_SIZE 16
#define NUM_KV_STAGES 8
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


__device__ __forceinline__ void tmem_ld_x4(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x4.b32"
        " {%0, %1, %2, %3}, [%4];"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3])
        : "r"(tmem_addr));
}


__device__ __forceinline__ void tmem_st_x4_f32(int tmem_addr, const float* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x4.b32"
        " [%0], {%1, %2, %3, %4};"
        :: "r"(tmem_addr),
           "f"(src[0]), "f"(src[1]), "f"(src[2]), "f"(src[3]));
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


__device__ __forceinline__ float2 add_f32x2(float2 a, float2 b) {
    float2 r;
    asm("add.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 sub_f32x2(float2 a, float2 b) {
    float2 r;
    asm("sub.rn.ftz.f32x2 %0, %1, %2;"
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

__global__ __launch_bounds__(512, LAUNCH_MIN_BLOCKS) void
kernel_cake_fmha_decode_balanced_fp16q(unsigned int* __restrict__ Qt, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap V, __half* __restrict__ O_ptr, int* __restrict__ page_table, int* __restrict__ seq_lens_kv, float* __restrict__ partial_o, float* __restrict__ partial_stats, unsigned int* __restrict__ tile_counters, unsigned int* __restrict__ queue_counters, int max_pages_per_seq, float softmax_scale_log2, float output_scale, int num_q_heads, int num_kv_heads, int group_ratio, int batch_size, int q_len, unsigned int max_items)
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
    #define kv_full_addr (mbar_base + 32)
    #define kv_empty_addr (mbar_base + 96)
    #define s_full_0_addr (mbar_base + 160)
    #define s_full_1_addr (mbar_base + 168)
    #define s_empty_0_addr (mbar_base + 176)
    #define s_empty_1_addr (mbar_base + 184)
    #define p_full_0_addr (mbar_base + 192)
    #define p_full_1_addr (mbar_base + 200)
    #define p_empty_0_addr (mbar_base + 208)
    #define p_empty_1_addr (mbar_base + 216)
    #define o_done_0_addr (mbar_base + 224)
    #define o_done_1_addr (mbar_base + 232)
    #define corr_scale_0_addr (mbar_base + 240)
    #define corr_scale_1_addr (mbar_base + 248)
    #define corr_empty_0_addr (mbar_base + 256)
    #define corr_empty_1_addr (mbar_base + 264)
    #define stats_empty_addr (mbar_base + 272)
    #define tmem_dealloc_addr (mbar_base + 280)
    #define order_p01_0_addr (mbar_base + 288)
    #define order_p01_1_addr (mbar_base + 296)
    #define work_full_addr (mbar_base + 304)
    #define work_empty_addr (mbar_base + 336)
    #define claim_gate_addr (mbar_base + 368)
    #define pg_full_addr (mbar_base + 376)
    #define pg_empty_addr (mbar_base + 424)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    float* smem_corr_reduce = reinterpret_cast<float*>(smem_raw + 1024);
    const int smem_corr_reduce_addr = smem + 1024;
    float* smem_exch0 = reinterpret_cast<float*>(smem_raw + 1152);
    const int smem_exch0_addr = smem + 1152;
    float* smem_exch1 = reinterpret_cast<float*>(smem_raw + 1408);
    const int smem_exch1_addr = smem + 1408;
    unsigned int* smem_exch0_u32 = reinterpret_cast<unsigned int*>(smem_raw + 1152);
    const int smem_exch0_u32_addr = smem + 1152;
    unsigned int* smem_exch1_u32 = reinterpret_cast<unsigned int*>(smem_raw + 1408);
    const int smem_exch1_u32_addr = smem + 1408;
    uint8_t* smem_qt = reinterpret_cast<uint8_t*>(smem_raw + 1664);
    const int smem_qt_addr = smem + 1664;
    uint8_t* smem_qt_lo = reinterpret_cast<uint8_t*>(smem_raw + 146048);
    const int smem_qt_lo_addr = smem + 146048;
    uint8_t* smem_kv = reinterpret_cast<uint8_t*>(smem_raw + 3712);
    const int smem_kv_addr = smem + 3712;
    uint8_t* smem_v = reinterpret_cast<uint8_t*>(smem_raw + 3712);
    const int smem_v_addr = smem + 3712;
    uint8_t* smem_p0 = reinterpret_cast<uint8_t*>(smem_raw + 134784);
    const int smem_p0_addr = smem + 134784;
    uint8_t* smem_p1 = reinterpret_cast<uint8_t*>(smem_raw + 136832);
    const int smem_p1_addr = smem + 136832;
    uint8_t* smem_o = reinterpret_cast<uint8_t*>(smem_raw + 138880);
    const int smem_o_addr = smem + 138880;
    unsigned int* smem_o_u32 = reinterpret_cast<unsigned int*>(smem_raw + 138880);
    const int smem_o_u32_addr = smem + 138880;
    unsigned int* work_token_words = reinterpret_cast<unsigned int*>(smem_raw + 139904);
    const int work_token_words_addr = smem + 139904;
    unsigned int* smem_merge_flag = reinterpret_cast<unsigned int*>(smem_raw + 140160);
    const int smem_merge_flag_addr = smem + 140160;
    int* sched_seq_lens = reinterpret_cast<int*>(smem_raw + 140176);
    const int sched_seq_lens_addr = smem + 140176;
    int* smem_pg = reinterpret_cast<int*>(smem_raw + 144384);
    const int smem_pg_addr = smem + 144384;
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&K))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&V))) : "memory");

    // Mbarrier init (27 pipeline groups, 0 ordered-sequence groups, 59 barriers)
    // Mbarriers at smem_raw[0..472)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 2 barriers, init_count=32
            mbarrier_init(smem + 0, 32);
            mbarrier_init(smem + 8, 32);
            // q_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // kv_full: 8 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // kv_empty: 8 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            // s_full_0: 1 barriers, init_count=1
            mbarrier_init(smem + 160, 1);
            // s_full_1: 1 barriers, init_count=1
            mbarrier_init(smem + 168, 1);
            // s_empty_0: 1 barriers, init_count=128
            mbarrier_init(smem + 176, 128);
            // s_empty_1: 1 barriers, init_count=128
            mbarrier_init(smem + 184, 128);
            // p_full_0: 1 barriers, init_count=256
            mbarrier_init(smem + 192, 256);
            // p_full_1: 1 barriers, init_count=256
            mbarrier_init(smem + 200, 256);
            // p_empty_0: 1 barriers, init_count=1
            mbarrier_init(smem + 208, 1);
            // p_empty_1: 1 barriers, init_count=1
            mbarrier_init(smem + 216, 1);
            // o_done_0: 1 barriers, init_count=1
            mbarrier_init(smem + 224, 1);
            // o_done_1: 1 barriers, init_count=1
            mbarrier_init(smem + 232, 1);
            // corr_scale_0: 1 barriers, init_count=128
            mbarrier_init(smem + 240, 128);
            // corr_scale_1: 1 barriers, init_count=128
            mbarrier_init(smem + 248, 128);
            // corr_empty_0: 1 barriers, init_count=128
            mbarrier_init(smem + 256, 128);
            // corr_empty_1: 1 barriers, init_count=128
            mbarrier_init(smem + 264, 128);
            // stats_empty: 1 barriers, init_count=4
            mbarrier_init(smem + 272, 4);
            // tmem_dealloc: 1 barriers, init_count=128
            mbarrier_init(smem + 280, 128);
            // order_p01_0: 1 barriers, init_count=128
            mbarrier_init(smem + 288, 128);
            // order_p01_1: 1 barriers, init_count=128
            mbarrier_init(smem + 296, 128);
            // --- pipeline 'work_pipe' ---
            // work_full: 4 barriers, init_count=1
            mbarrier_init(smem + 304, 1);
            mbarrier_init(smem + 312, 1);
            mbarrier_init(smem + 320, 1);
            mbarrier_init(smem + 328, 1);
            // work_empty: 4 barriers, init_count=480
            mbarrier_init(smem + 336, 480);
            mbarrier_init(smem + 344, 480);
            mbarrier_init(smem + 352, 480);
            mbarrier_init(smem + 360, 480);
            // claim_gate: 1 barriers, init_count=1
            mbarrier_init(smem + 368, 1);
            // pg_full: 6 barriers, init_count=32
            mbarrier_init(smem + 376, 32);
            mbarrier_init(smem + 384, 32);
            mbarrier_init(smem + 392, 32);
            mbarrier_init(smem + 400, 32);
            mbarrier_init(smem + 408, 32);
            mbarrier_init(smem + 416, 32);
            // pg_empty: 6 barriers, init_count=1
            mbarrier_init(smem + 424, 1);
            mbarrier_init(smem + 432, 1);
            mbarrier_init(smem + 440, 1);
            mbarrier_init(smem + 448, 1);
            mbarrier_init(smem + 456, 1);
            mbarrier_init(smem + 464, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (128 columns, 96 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 472);
    if (warp == 0) {
        int _tmem_hold = smem + 472;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_s0 = taddr;
    const int tmem_tmem_s1 = taddr + 8;
    const int tmem_tmem_stats0 = taddr + 16;
    const int tmem_tmem_stats1 = taddr + 48;
    const int tmem_tmem_o0 = taddr + 80;
    const int tmem_tmem_o1 = taddr + 88;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
    }

    // ---- Role: softmax ----
    if (warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 184;");
        { // softmax_main
            int is_wg1 = ((warp >= 4) ? 1 : 0);
            const int tmem_row_base_v = warp % 4 * 32;
            int my_tmem_s_base = taddr + (unsigned int)(((is_wg1 != 0) ? 8 : 0));
            int my_tmem_stats = taddr + (unsigned int)(((is_wg1 != 0) ? 48 : 16)) + (unsigned int)(tmem_row_base_v << 16);
            const int warp_in_wg = warp % 4;
            const int wg_tid = warp_in_wg * 32 + lane;
            int col_pair = wg_tid % 4;
            int col_pair_base = col_pair * 2;
            float* my_exch_ptr = ((is_wg1 != 0) ? (smem_exch1) : (smem_exch0));
            unsigned int* my_exch_u32_ptr = ((is_wg1 != 0) ? (smem_exch1_u32) : (smem_exch0_u32));
            uint8_t* my_p_base = ((is_wg1 != 0) ? (smem_p1) : (smem_p0));
            int my_p_swz_base = ((is_wg1 != 0) ? smem_p1_addr : smem_p0_addr) / 128 % 8;
            float sv[8];
            float sv_lo[4];
            float sv_hi[4];
            int op01_phase = ((is_wg1 != 0) ? 0 : 1);
            int op01_stage = 0;
            unsigned int work_stage_s = 0;
            unsigned int _phase_work_full = 0;
            mbarrier_wait(work_full_addr + (work_stage_s) * 8, _phase_work_full);
            unsigned int base = work_stage_s * 16;
            unsigned int valid = work_token_words[base];
            unsigned int row_tile = work_token_words[base + 1];
            unsigned int batch = work_token_words[base + 2];
            unsigned int kv_head = work_token_words[base + 3];
            unsigned int block_begin = work_token_words[base + 4];
            unsigned int block_end = work_token_words[base + 5];
            unsigned int seqlen_row = work_token_words[base + 6];
            unsigned int n_chunks = work_token_words[base + 7];
            unsigned int slot_tile_base = work_token_words[base + 8];
            unsigned int counter_idx = work_token_words[base + 9];
            unsigned int chunk = work_token_words[base + 10];
            mbarrier_arrive(work_empty_addr + (work_stage_s) * 8);
            work_stage_s += 1;
            if (work_stage_s == 4) { work_stage_s = 0; _phase_work_full ^= 1; }
            unsigned int valid_s = valid;
            int block_begin_s = (int)block_begin;
            int block_end_s = (int)block_end;
            int seqlen_kv = (int)seqlen_row;
            unsigned int _phase_stats_empty_0 = 1;
            unsigned int _phase_s_full_1_0 = 0;
            unsigned int _phase_s_full_0_0 = 0;
            unsigned int _phase_corr_empty_1_0 = 1;
            unsigned int _phase_corr_empty_0_0 = 1;
            unsigned int _phase_p_empty_1_0 = 1;
            unsigned int _phase_p_empty_0_0 = 1;
            #pragma unroll 1
            for (unsigned int _tile_iter_s = 0; _tile_iter_s < max_items; _tile_iter_s++) {
                if (valid_s == 0) {
                    break;
                }
                mbarrier_wait(stats_empty_addr, _phase_stats_empty_0);
                _phase_stats_empty_0 ^= 1;
                int num_n_blocks_total = block_end_s - block_begin_s;
                int cta_n_blocks = num_n_blocks_total + num_n_blocks_total % 2;
                int split_start_block = block_begin_s;
                int num_pairs = cta_n_blocks / 2;
                float row_max_pair[2];
                float row_sum_pair[2];
                row_max_pair[0] = -CAKE_FMHA_INF;
                row_max_pair[1] = -CAKE_FMHA_INF;
                row_sum_pair[0] = 0.0f;
                row_sum_pair[1] = 0.0f;
                if (is_wg1 != 0) {
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                } else {
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                }
                uint32_t _amf_u_0 = __float_as_uint(-3.4028235e+38f);
                uint32_t _amf_mask_0 = -int32_t(_amf_u_0 >> 31) | 0x80000000u;
                unsigned int _amf_enc_0 = _amf_u_0 ^ _amf_mask_0;
                if (wg_tid < 8) {
                    my_exch_u32_ptr[wg_tid] = _amf_enc_0;
                }
                if (is_wg1 != 0) {
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                } else {
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                }
                if (is_wg1 != 0) {
                    mbarrier_wait(s_full_1_addr, _phase_s_full_1_0);
                    _phase_s_full_1_0 ^= 1;
                } else {
                    mbarrier_wait(s_full_0_addr, _phase_s_full_0_0);
                    _phase_s_full_0_0 ^= 1;
                }
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&sv_lo[0])), "=r"(*reinterpret_cast<uint32_t*>(&sv_lo[1])), "=r"(*reinterpret_cast<uint32_t*>(&sv_lo[2])), "=r"(*reinterpret_cast<uint32_t*>(&sv_lo[3]))
                    : "r"(my_tmem_s_base));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&sv_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&sv_hi[1])), "=r"(*reinterpret_cast<uint32_t*>(&sv_hi[2])), "=r"(*reinterpret_cast<uint32_t*>(&sv_hi[3]))
                    : "r"(my_tmem_s_base + 1048576));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                if (is_wg1 != 0) {
                    mbarrier_arrive(s_empty_1_addr);
                } else {
                    mbarrier_arrive(s_empty_0_addr);
                }
                #pragma unroll
                for (int c = 0; c < 4; c++) {
                    sv[c] = sv_lo[c];
                    sv[c + 4] = sv_hi[c];
                }
                #pragma unroll 1
                for (int pair = 0; pair < num_pairs; pair++) {
                    int my_block = split_start_block + cta_n_blocks - 1 - 2 * pair - is_wg1;
                    int ldtm_row_base = warp_in_wg * 32 + lane / 4;
                    int kv_pos0 = my_block * BLOCK_N + ldtm_row_base;
                    int kv_pos1 = kv_pos0 + 8;
                    int kv_pos2 = kv_pos0 + 16;
                    int kv_pos3 = kv_pos0 + 24;
                    if (seqlen_kv < (my_block + 1) * BLOCK_N) {
                        if (kv_pos0 >= seqlen_kv) {
                            sv[0] = -3.4028235e+38f;
                            sv[1] = -3.4028235e+38f;
                        }
                        if (kv_pos1 >= seqlen_kv) {
                            sv[2] = -3.4028235e+38f;
                            sv[3] = -3.4028235e+38f;
                        }
                        if (kv_pos2 >= seqlen_kv) {
                            sv[4] = -3.4028235e+38f;
                            sv[5] = -3.4028235e+38f;
                        }
                        if (kv_pos3 >= seqlen_kv) {
                            sv[6] = -3.4028235e+38f;
                            sv[7] = -3.4028235e+38f;
                        }
                    }
                    float pair_max[2];
                    float _max_5 = max_noftz(sv[0], sv[2]);
                    float _max_6 = max_noftz(sv[4], sv[6]);
                    float _max_7 = max_noftz(_max_5, _max_6);
                    pair_max[0] = _max_7;
                    float _max_8 = max_noftz(sv[1], sv[3]);
                    float _max_9 = max_noftz(sv[5], sv[7]);
                    float _max_10 = max_noftz(_max_8, _max_9);
                    pair_max[1] = _max_10;
                    #pragma unroll
                    for (int c_1 = 0; c_1 < 2; c_1++) {
                        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, pair_max[c_1], 16);
                        float _max_11 = max_noftz(pair_max[c_1], _shfl_xor_0);
                        pair_max[c_1] = _max_11;
                        float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, pair_max[c_1], 8);
                        float _max_12 = max_noftz(pair_max[c_1], _shfl_xor_1);
                        pair_max[c_1] = _max_12;
                    }
                    float old_max_pair[2];
                    float new_max_pair[2];
                    #pragma unroll
                    for (int c_2 = 0; c_2 < 2; c_2++) {
                        old_max_pair[c_2] = row_max_pair[c_2];
                        float _max_13 = max_noftz(row_max_pair[c_2], pair_max[c_2]);
                        new_max_pair[c_2] = _max_13;
                    }
                    if (lane < 8) {
                        uint32_t _amf_u_1 = __float_as_uint(new_max_pair[0]);
                        uint32_t _amf_mask_1 = -int32_t(_amf_u_1 >> 31) | 0x80000000u;
                        unsigned int _amf_enc_1 = _amf_u_1 ^ _amf_mask_1;
                        uint32_t _amf_u_2 = __float_as_uint(new_max_pair[1]);
                        uint32_t _amf_mask_2 = -int32_t(_amf_u_2 >> 31) | 0x80000000u;
                        unsigned int _amf_enc_2 = _amf_u_2 ^ _amf_mask_2;
                        atomicMax(&my_exch_u32_ptr[col_pair_base], _amf_enc_1);
                        atomicMax(&my_exch_u32_ptr[col_pair_base + 1], _amf_enc_2);
                    }
                    if (is_wg1 != 0) {
                        asm volatile("barrier.sync 9, 128;" ::: "memory");
                    } else {
                        asm volatile("barrier.sync 8, 128;" ::: "memory");
                    }
                    uint32_t _amf_u_3 = my_exch_u32_ptr[col_pair_base];
                    uint32_t _amf_mask_3 = ((_amf_u_3 >> 31) - 1u) | 0x80000000u;
                    float _amf_dec_0 = __uint_as_float(_amf_u_3 ^ _amf_mask_3);
                    new_max_pair[0] = _amf_dec_0;
                    uint32_t _amf_u_4 = my_exch_u32_ptr[col_pair_base + 1];
                    uint32_t _amf_mask_4 = ((_amf_u_4 >> 31) - 1u) | 0x80000000u;
                    float _amf_dec_1 = __uint_as_float(_amf_u_4 ^ _amf_mask_4);
                    new_max_pair[1] = _amf_dec_1;
                    float acc_scale_pair[2];
                    float2 _f2_0 = make_float2(row_max_pair[0], row_max_pair[1]);
                    float2 _f2_1 = make_float2(new_max_pair[0], new_max_pair[1]);
                    float2 acc_delta_pair_f2 = sub_f32x2(_f2_0, _f2_1);
                    float2 _f2_2 = make_float2(softmax_scale_log2, softmax_scale_log2);
                    float2 _mul_f32x2_0;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_0) : "l"(*(const unsigned long long*)&_f2_2), "l"(*(const unsigned long long*)&acc_delta_pair_f2));
                    float2 acc_scaled_delta_pair_f2 = _mul_f32x2_0;
                    acc_scale_pair[0] = 1.0f;
                    acc_scale_pair[1] = 1.0f;
                    int needs_acc_rescale = ((acc_delta_pair_f2.x != 0.0f) ? 1 : 0);
                    needs_acc_rescale = needs_acc_rescale | ((acc_delta_pair_f2.y != 0.0f) ? 1 : 0);
                    if (needs_acc_rescale != 0) {
                        float _exp2_0 = approx_exp2(acc_scaled_delta_pair_f2.x);
                        acc_scale_pair[0] = _exp2_0;
                        float _exp2_1 = approx_exp2(acc_scaled_delta_pair_f2.y);
                        acc_scale_pair[1] = _exp2_1;
                    }
                    float stats_pair[4];
                    stats_pair[0] = old_max_pair[0];
                    stats_pair[1] = old_max_pair[1];
                    stats_pair[2] = new_max_pair[0];
                    stats_pair[3] = new_max_pair[1];
                    if (is_wg1 != 0) {
                        mbarrier_wait(corr_empty_1_addr, _phase_corr_empty_1_0);
                        _phase_corr_empty_1_0 ^= 1;
                    } else {
                        mbarrier_wait(corr_empty_0_addr, _phase_corr_empty_0_0);
                        _phase_corr_empty_0_0 ^= 1;
                    }
                    tmem_st_x4_f32(my_tmem_stats, stats_pair);
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    if (is_wg1 != 0) {
                        mbarrier_arrive(corr_scale_1_addr);
                    } else {
                        mbarrier_arrive(corr_scale_0_addr);
                    }
                    float exp_vals[8];
                    row_max_pair[0] = new_max_pair[0];
                    row_max_pair[1] = new_max_pair[1];
                    float safe_max0 = ((new_max_pair[0] == -CAKE_FMHA_INF) ? 0.0f : new_max_pair[0]);
                    float safe_max1 = ((new_max_pair[1] == -CAKE_FMHA_INF) ? 0.0f : new_max_pair[1]);
                    float2 _f2_3 = make_float2(softmax_scale_log2, softmax_scale_log2);
                    float2 _f2_4 = make_float2(-softmax_scale_log2, -softmax_scale_log2);
                    float2 _f2_5 = make_float2(8.8073549f, 8.8073549f);
                    float2 _f2_6 = make_float2(safe_max0, safe_max1);
                    float2 neg_scaled_pair_f2 = fma_f32x2_rn_ftz(_f2_6, _f2_4, _f2_5);
                    float2 _f2_7 = make_float2(sv[0], sv[1]);
                    float2 _mul_f32x2_1;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_1) : "l"(*(const unsigned long long*)&_f2_3), "l"(*(const unsigned long long*)&_f2_7));
                    float2 scaled01_pair_f2 = _mul_f32x2_1;
                    float2 affine01_pair_f2 = add_f32x2(scaled01_pair_f2, neg_scaled_pair_f2);
                    exp_vals[0] = affine01_pair_f2.x;
                    exp_vals[1] = affine01_pair_f2.y;
                    float2 _f2_8 = make_float2(sv[2], sv[3]);
                    float2 affine23_pair_f2 = fma_f32x2_rn_ftz(_f2_3, _f2_8, neg_scaled_pair_f2);
                    exp_vals[2] = affine23_pair_f2.x;
                    exp_vals[3] = affine23_pair_f2.y;
                    if (is_wg1 != 0) {
                        mbarrier_wait(order_p01_1_addr, op01_phase);
                    } else {
                        mbarrier_wait(order_p01_0_addr, op01_phase);
                    }
                    float _exp2_2 = approx_exp2(exp_vals[0]);
                    exp_vals[0] = _exp2_2;
                    float2 _f2_9 = make_float2(sv[4], sv[5]);
                    float2 affine45_pair_f2 = fma_f32x2_rn_ftz(_f2_3, _f2_9, neg_scaled_pair_f2);
                    exp_vals[4] = affine45_pair_f2.x;
                    exp_vals[5] = affine45_pair_f2.y;
                    float _exp2_3 = approx_exp2(exp_vals[1]);
                    exp_vals[1] = _exp2_3;
                    if (is_wg1 != 0) {
                        mbarrier_arrive(order_p01_0_addr);
                    } else {
                        mbarrier_arrive(order_p01_1_addr);
                    }
                    op01_phase ^= 1;
                    float _exp2_4 = approx_exp2(exp_vals[2]);
                    exp_vals[2] = _exp2_4;
                    float2 _f2_10 = make_float2(sv[6], sv[7]);
                    float2 affine67_pair_f2 = fma_f32x2_rn_ftz(_f2_3, _f2_10, neg_scaled_pair_f2);
                    exp_vals[6] = affine67_pair_f2.x;
                    exp_vals[7] = affine67_pair_f2.y;
                    float _exp2_5 = approx_exp2(exp_vals[3]);
                    exp_vals[3] = _exp2_5;
                    float _exp2_6 = approx_exp2(exp_vals[4]);
                    exp_vals[4] = _exp2_6;
                    float _exp2_7 = approx_exp2(exp_vals[5]);
                    exp_vals[5] = _exp2_7;
                    float _exp2_8 = approx_exp2(exp_vals[6]);
                    exp_vals[6] = _exp2_8;
                    float _exp2_9 = approx_exp2(exp_vals[7]);
                    exp_vals[7] = _exp2_9;
                    unsigned int regs_p[2];
                    {
                        uint32_t _packed;
                        asm volatile("{\n\t"
                            ".reg .b16 _lo;\n\t"
                            ".reg .b16 _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}"
                            : "=r"(_packed) : "f"(exp_vals[0]), "f"(exp_vals[1]),
                                               "f"(exp_vals[2]), "f"(exp_vals[3]));
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
                            : "=r"(_packed) : "f"(exp_vals[4]), "f"(exp_vals[5]),
                                               "f"(exp_vals[6]), "f"(exp_vals[7]));
                        regs_p[1] = _packed;
                    }
                    int mtx_idx = lane / 8;
                    int thr_row_idx = lane % 8;
                    int p_swz_row = (my_p_swz_base + thr_row_idx) % 8;
                    int seg_col_idx = warp_in_wg * 2 + mtx_idx ^ p_swz_row;
                    int stsm_offset = thr_row_idx * 128 + seg_col_idx * 16;
                    if (is_wg1 != 0) {
                        mbarrier_wait(p_empty_1_addr, _phase_p_empty_1_0);
                        _phase_p_empty_1_0 ^= 1;
                    } else {
                        mbarrier_wait(p_empty_0_addr, _phase_p_empty_0_0);
                        _phase_p_empty_0_0 ^= 1;
                    }
                    const void* _stmatrix_b8_ptr_5 = reinterpret_cast<const void*>(reinterpret_cast<uint8_t*>(my_p_base) + stsm_offset);
                    uint64_t _stmatrix_b8_addr64_5;
                    asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(_stmatrix_b8_addr64_5) : "l"(_stmatrix_b8_ptr_5));
                    uint32_t _stmatrix_b8_addr_5;
                    asm volatile("cvt.u32.u64 %0, %1;" : "=r"(_stmatrix_b8_addr_5) : "l"(_stmatrix_b8_addr64_5));
                    asm volatile("stmatrix.sync.aligned.m16n8.x2.trans.shared.b8 [%0], {%1, %2};\n"
                        :: "r"(_stmatrix_b8_addr_5), "r"(regs_p[0]), "r"(regs_p[1])
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    if (is_wg1 != 0) {
                        mbarrier_arrive(p_full_1_addr);
                    } else {
                        mbarrier_arrive(p_full_0_addr);
                    }
                    float2 _f2_11 = make_float2(exp_vals[0], exp_vals[1]);
                    float2 _f2_12 = make_float2(exp_vals[2], exp_vals[3]);
                    float2 _f2_13 = make_float2(exp_vals[4], exp_vals[5]);
                    float2 _f2_14 = make_float2(exp_vals[6], exp_vals[7]);
                    float2 _f2_15 = make_float2(row_sum_pair[0], row_sum_pair[1]);
                    float2 _f2_16 = make_float2(acc_scale_pair[0], acc_scale_pair[1]);
                    float2 row_sum_p0_f2 = fma_f32x2_rn_ftz(_f2_15, _f2_16, _f2_11);
                    float2 row_sum_p1_f2 = add_f32x2(row_sum_p0_f2, _f2_12);
                    float2 row_sum_p2_f2 = add_f32x2(row_sum_p1_f2, _f2_13);
                    float2 row_sum_next_f2 = add_f32x2(row_sum_p2_f2, _f2_14);
                    row_sum_pair[0] = row_sum_next_f2.x;
                    row_sum_pair[1] = row_sum_next_f2.y;
                    if (pair < num_pairs - 1) {
                        if (is_wg1 != 0) {
                            mbarrier_wait(s_full_1_addr, _phase_s_full_1_0);
                            _phase_s_full_1_0 ^= 1;
                        } else {
                            mbarrier_wait(s_full_0_addr, _phase_s_full_0_0);
                            _phase_s_full_0_0 ^= 1;
                        }
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&sv_lo[0])), "=r"(*reinterpret_cast<uint32_t*>(&sv_lo[1])), "=r"(*reinterpret_cast<uint32_t*>(&sv_lo[2])), "=r"(*reinterpret_cast<uint32_t*>(&sv_lo[3]))
                            : "r"(my_tmem_s_base));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&sv_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&sv_hi[1])), "=r"(*reinterpret_cast<uint32_t*>(&sv_hi[2])), "=r"(*reinterpret_cast<uint32_t*>(&sv_hi[3]))
                            : "r"(my_tmem_s_base + 1048576));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        if (is_wg1 != 0) {
                            mbarrier_arrive(s_empty_1_addr);
                        } else {
                            mbarrier_arrive(s_empty_0_addr);
                        }
                        #pragma unroll
                        for (int c_3 = 0; c_3 < 4; c_3++) {
                            sv[c_3] = sv_lo[c_3];
                            sv[c_3 + 4] = sv_hi[c_3];
                        }
                    }
                }
                if (is_wg1 != 0) {
                    mbarrier_wait(corr_empty_1_addr, _phase_corr_empty_1_0);
                    _phase_corr_empty_1_0 ^= 1;
                } else {
                    mbarrier_wait(corr_empty_0_addr, _phase_corr_empty_0_0);
                    _phase_corr_empty_0_0 ^= 1;
                }
                float final_stats_pair[4];
                final_stats_pair[0] = row_sum_pair[0];
                final_stats_pair[1] = row_sum_pair[1];
                final_stats_pair[2] = row_max_pair[0];
                final_stats_pair[3] = row_max_pair[1];
                tmem_st_x4_f32(my_tmem_stats, final_stats_pair);
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                if (is_wg1 != 0) {
                    mbarrier_arrive(corr_scale_1_addr);
                } else {
                    mbarrier_arrive(corr_scale_0_addr);
                }
                mbarrier_wait(work_full_addr + (work_stage_s) * 8, _phase_work_full);
                unsigned int base_0 = work_stage_s * 16;
                unsigned int valid_1 = work_token_words[base_0];
                unsigned int row_tile_2 = work_token_words[base_0 + 1];
                unsigned int batch_3 = work_token_words[base_0 + 2];
                unsigned int kv_head_4 = work_token_words[base_0 + 3];
                unsigned int block_begin_5 = work_token_words[base_0 + 4];
                unsigned int block_end_6 = work_token_words[base_0 + 5];
                unsigned int seqlen_row_7 = work_token_words[base_0 + 6];
                unsigned int n_chunks_8 = work_token_words[base_0 + 7];
                unsigned int slot_tile_base_9 = work_token_words[base_0 + 8];
                unsigned int counter_idx_10 = work_token_words[base_0 + 9];
                unsigned int chunk_11 = work_token_words[base_0 + 10];
                mbarrier_arrive(work_empty_addr + (work_stage_s) * 8);
                work_stage_s += 1;
                if (work_stage_s == 4) { work_stage_s = 0; _phase_work_full ^= 1; }
                valid_s = valid_1;
                block_begin_s = (int)block_begin_5;
                block_end_s = (int)block_end_6;
                seqlen_kv = (int)seqlen_row_7;
            }
        }
    }
    // ---- Role: correction ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 88;");
        { // correction_main
            const int tmem_row_base_v_1 = warp % 4 * 32;
            const int corr_row = tmem_row_base_v_1 << 16;
            const int corr_tid = warp % 4 * 32 + lane;
            const int col_pair_base_c = corr_tid % 4 * 2;
            const int o_row_base = warp % 4 * 32 + lane / 4;
            int slot_stride = q_len * num_kv_heads;
            unsigned int work_stage_c = 0;
            if (corr_tid < 32) {
                int spec_ipc = q_len * num_kv_heads;
                int spec_tiles = batch_size * spec_ipc;
                int spec_ctas = gridDim.x;
                int spec_id = blockIdx.x;
                int spec_len0 = seq_lens_kv[0];
                int spec_p0 = (spec_len0 - (q_len - 1) + 255) / 256;
                int spec_l = spec_p0;
                if (spec_tiles <= spec_ctas) {
                    int spec_n_even = spec_ctas / spec_tiles;
                    if (spec_n_even > 1) {
                        int spec_l_even = (spec_p0 + spec_n_even - 1) / spec_n_even;
                        if (spec_l_even < 2) {
                            spec_l_even = 2;
                        }
                        if (spec_l_even < spec_p0) {
                            spec_l = spec_l_even;
                        }
                    }
                }
                int spec_full = spec_p0 / spec_l;
                int spec_rem = spec_p0 - spec_full * spec_l;
                int spec_n_chunks = spec_full;
                if (spec_rem > 0) {
                    spec_n_chunks = spec_full + 1;
                }
                int spec_full_tickets = spec_tiles * spec_full;
                int spec_valid = 0;
                int spec_batch = 0;
                int spec_chunk = 0;
                int spec_jh = 0;
                if (spec_id < spec_full_tickets) {
                    int spec_per_req = spec_full * spec_ipc;
                    spec_batch = spec_id / spec_per_req;
                    int spec_r = spec_id - spec_batch * spec_per_req;
                    spec_chunk = spec_r / spec_ipc;
                    spec_jh = spec_r - spec_chunk * spec_ipc;
                    spec_valid = 1;
                }
                if (spec_id >= spec_full_tickets) {
                    if (spec_rem > 0) {
                        int spec_t = spec_id - spec_full_tickets;
                        if (spec_t < spec_tiles) {
                            spec_batch = spec_t / spec_ipc;
                            spec_jh = spec_t - spec_batch * spec_ipc;
                            spec_chunk = spec_full;
                            spec_valid = 1;
                        }
                    }
                }
                if (spec_valid != 0) {
                    int spec_row = spec_jh / num_kv_heads;
                    int spec_head = spec_jh - spec_row * num_kv_heads;
                    int spec_seqlen = spec_len0 - (q_len - 1 - spec_row);
                    int spec_nblocks = (spec_seqlen + BLOCK_N - 1) / BLOCK_N;
                    int spec_begin = 2 * spec_chunk * spec_l;
                    int spec_end = spec_nblocks;
                    if (spec_chunk < spec_n_chunks - 1) {
                        spec_end = 2 * (spec_chunk + 1) * spec_l;
                    }
                    int spec_max_pg = (spec_seqlen + PAGE_SIZE - 1) / PAGE_SIZE - 1;
                    int spec_pt_base = spec_batch * max_pages_per_seq;
                    if (corr_tid < 24) {
                        int spec_nb = corr_tid / 8;
                        int spec_block = spec_end - 1 - spec_nb % 2;
                        if (spec_block >= spec_begin) {
                            int spec_page_idx = spec_block * 8 + corr_tid % 8;
                            if (spec_page_idx > spec_max_pg) {
                                spec_page_idx = spec_max_pg;
                            }
                            int spec_page = page_table[spec_pt_base + spec_page_idx];
                            if (spec_nb < 2) {
                                asm volatile("cp.async.bulk.prefetch.tensor.5d.L2.global.tile [%0, {%1, %2, %3, %4, %5}];" :: "l"((uint64_t)((&K))), "r"((int)(0)), "r"((int)(0)), "r"((int)(0)), "r"((int)(spec_head)), "r"((int)(spec_page)) : "memory");
                            }
                            if (spec_nb == 2) {
                                asm volatile("cp.async.bulk.prefetch.tensor.5d.L2.global.tile [%0, {%1, %2, %3, %4, %5}];" :: "l"((uint64_t)((&V))), "r"((int)(0)), "r"((int)(0)), "r"((int)(0)), "r"((int)(spec_head)), "r"((int)(spec_page)) : "memory");
                            }
                        }
                    }
                    if (corr_tid == 24) {
                        int spec_q_word = (spec_batch * spec_ipc + spec_jh) * (TILE_Q * HEAD_DIM / 2);
                        #pragma unroll
                        for (int spec_ln = 0; spec_ln < 16; spec_ln++) {
                            asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(Qt + (spec_q_word + spec_ln * 32))));
                        }
                    }
                    if (corr_tid > 24) {
                        if (corr_tid < 28) {
                            int spec_line_idx = (spec_end - 8) * 8 + (corr_tid - 24 - 1) * 32;
                            if (spec_line_idx < 0) {
                                spec_line_idx = 0;
                            }
                            if (spec_line_idx > spec_max_pg) {
                                spec_line_idx = spec_max_pg;
                            }
                            asm volatile("prefetch.global.L2 [%0];" :: "l"((uint64_t)(page_table + (spec_pt_base + spec_line_idx))));
                        }
                    }
                }
            }
            unsigned int _phase_work_full_1 = 0;
            mbarrier_wait(work_full_addr + (work_stage_c) * 8, _phase_work_full_1);
            unsigned int base_1 = work_stage_c * 16;
            unsigned int valid_2 = work_token_words[base_1];
            unsigned int row_tile_1 = work_token_words[base_1 + 1];
            unsigned int batch_1 = work_token_words[base_1 + 2];
            unsigned int kv_head_1 = work_token_words[base_1 + 3];
            unsigned int block_begin_1 = work_token_words[base_1 + 4];
            unsigned int block_end_1 = work_token_words[base_1 + 5];
            unsigned int seqlen_row_1 = work_token_words[base_1 + 6];
            unsigned int n_chunks_1 = work_token_words[base_1 + 7];
            unsigned int slot_tile_base_1 = work_token_words[base_1 + 8];
            unsigned int counter_idx_1 = work_token_words[base_1 + 9];
            unsigned int chunk_1 = work_token_words[base_1 + 10];
            mbarrier_arrive(work_empty_addr + (work_stage_c) * 8);
            work_stage_c += 1;
            if (work_stage_c == 4) { work_stage_c = 0; _phase_work_full_1 ^= 1; }
            unsigned int valid_c = valid_2;
            int row_tile_c = (int)row_tile_1;
            int kv_head_idx = (int)kv_head_1;
            int block_begin_c = (int)block_begin_1;
            int block_end_c = (int)block_end_1;
            int n_chunks_c = (int)n_chunks_1;
            int slot_tile_base_c = (int)slot_tile_base_1;
            int counter_idx_c = (int)counter_idx_1;
            int chunk_c = (int)chunk_1;
            unsigned int _phase_corr_scale_0_0 = 0;
            unsigned int _phase_corr_scale_1_0 = 0;
            unsigned int _phase_o_done_0_0 = 0;
            unsigned int _phase_o_done_1_0 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_c = 0; _tile_iter_c < max_items; _tile_iter_c++) {
                if (valid_c == 0) {
                    break;
                }
                int num_n_blocks_total_1 = block_end_c - block_begin_c;
                int cta_n_blocks_1 = num_n_blocks_total_1 + num_n_blocks_total_1 % 2;
                int num_pairs_1 = cta_n_blocks_1 / 2;
                int my_slot = slot_tile_base_c + chunk_c * slot_stride;
                int partial_o_base = my_slot * 1024 + o_row_base;
                int partial_stats_base = my_slot * 16;
                if (num_pairs_1 > 0) {
                    mbarrier_wait(corr_scale_0_addr, _phase_corr_scale_0_0);
                    _phase_corr_scale_0_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    mbarrier_arrive(corr_empty_0_addr);
                    mbarrier_arrive(p_full_0_addr);
                    mbarrier_wait(corr_scale_1_addr, _phase_corr_scale_1_0);
                    _phase_corr_scale_1_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    mbarrier_arrive(corr_empty_1_addr);
                    mbarrier_arrive(p_full_1_addr);
                }
                #pragma unroll 1
                for (int pair_1 = 1; pair_1 < num_pairs_1; pair_1++) {
                    mbarrier_wait(corr_scale_0_addr, _phase_corr_scale_0_0);
                    _phase_corr_scale_0_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float _tmem_load_0[4];
                    tmem_ld_x4(&_tmem_load_0[0], taddr + 16 + (unsigned int)corr_row);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    mbarrier_arrive(corr_empty_0_addr);
                    mbarrier_wait(o_done_0_addr, _phase_o_done_0_0);
                    _phase_o_done_0_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float acc0_pair[2];
                    float2 _f2_17 = make_float2(_tmem_load_0[0], _tmem_load_0[1]);
                    float2 _f2_18 = make_float2(_tmem_load_0[2], _tmem_load_0[3]);
                    float2 max_diff0_pair_f2 = sub_f32x2(_f2_17, _f2_18);
                    float2 _f2_19 = make_float2(softmax_scale_log2, softmax_scale_log2);
                    float2 _mul_f32x2_2;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_2) : "l"(*(const unsigned long long*)&_f2_19), "l"(*(const unsigned long long*)&max_diff0_pair_f2));
                    float2 scaled_diff0_pair_f2 = _mul_f32x2_2;
                    float _exp2_10 = approx_exp2(scaled_diff0_pair_f2.x);
                    acc0_pair[0] = ((max_diff0_pair_f2.x != 0.0f) ? _exp2_10 : 1.0f);
                    float _exp2_11 = approx_exp2(scaled_diff0_pair_f2.y);
                    acc0_pair[1] = ((max_diff0_pair_f2.y != 0.0f) ? _exp2_11 : 1.0f);
                    int rescale_pred_0 = ((acc0_pair[0] != 1.0f) ? 1 : 0);
                    rescale_pred_0 = rescale_pred_0 | ((acc0_pair[1] != 1.0f) ? 1 : 0);
                    int _vote_1 = __any_sync(0xFFFFFFFF, rescale_pred_0 != 0);
                    if (_vote_1 != 0) {
                        float o0_lo[4];
                        float o0_hi[4];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&o0_lo[0])), "=r"(*reinterpret_cast<uint32_t*>(&o0_lo[1])), "=r"(*reinterpret_cast<uint32_t*>(&o0_lo[2])), "=r"(*reinterpret_cast<uint32_t*>(&o0_lo[3]))
                            : "r"(taddr + 80));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&o0_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&o0_hi[1])), "=r"(*reinterpret_cast<uint32_t*>(&o0_hi[2])), "=r"(*reinterpret_cast<uint32_t*>(&o0_hi[3]))
                            : "r"(taddr + 80 + 1048576));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        float o0[8];
                        #pragma unroll
                        for (int h = 0; h < 4; h++) {
                            o0[h] = o0_lo[h];
                            o0[h + 4] = o0_hi[h];
                        }
                        float2 _f2_20 = make_float2(acc0_pair[0], acc0_pair[1]);
                        float2 _f2_21 = make_float2(o0[0], o0[1]);
                        float2 _f2_22 = make_float2(o0[2], o0[3]);
                        float2 _f2_23 = make_float2(o0[4], o0[5]);
                        float2 _f2_24 = make_float2(o0[6], o0[7]);
                        float2 _mul_f32x2_3;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_3) : "l"(*(const unsigned long long*)&_f2_21), "l"(*(const unsigned long long*)&_f2_20));
                        float2 o0_lo01_scaled_f2 = _mul_f32x2_3;
                        float2 _mul_f32x2_4;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_4) : "l"(*(const unsigned long long*)&_f2_22), "l"(*(const unsigned long long*)&_f2_20));
                        float2 o0_lo23_scaled_f2 = _mul_f32x2_4;
                        float2 _mul_f32x2_5;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_5) : "l"(*(const unsigned long long*)&_f2_23), "l"(*(const unsigned long long*)&_f2_20));
                        float2 o0_hi01_scaled_f2 = _mul_f32x2_5;
                        float2 _mul_f32x2_6;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_6) : "l"(*(const unsigned long long*)&_f2_24), "l"(*(const unsigned long long*)&_f2_20));
                        float2 o0_hi23_scaled_f2 = _mul_f32x2_6;
                        o0[0] = o0_lo01_scaled_f2.x;
                        o0[1] = o0_lo01_scaled_f2.y;
                        o0[2] = o0_lo23_scaled_f2.x;
                        o0[3] = o0_lo23_scaled_f2.y;
                        o0[4] = o0_hi01_scaled_f2.x;
                        o0[5] = o0_hi01_scaled_f2.y;
                        o0[6] = o0_hi23_scaled_f2.x;
                        o0[7] = o0_hi23_scaled_f2.y;
                        #pragma unroll
                        for (int h_1 = 0; h_1 < 4; h_1++) {
                            o0_lo[h_1] = o0[h_1];
                            o0_hi[h_1] = o0[h_1 + 4];
                        }
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x256b.x1.b32"
                            " [%0], {%1, %2, %3, %4};"
                            :: "r"(taddr + 80), "r"(*reinterpret_cast<const uint32_t*>(&o0_lo[0])), "r"(*reinterpret_cast<const uint32_t*>(&o0_lo[1])), "r"(*reinterpret_cast<const uint32_t*>(&o0_lo[2])), "r"(*reinterpret_cast<const uint32_t*>(&o0_lo[3])));
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x256b.x1.b32"
                            " [%0], {%1, %2, %3, %4};"
                            :: "r"(taddr + 80 + 1048576), "r"(*reinterpret_cast<const uint32_t*>(&o0_hi[0])), "r"(*reinterpret_cast<const uint32_t*>(&o0_hi[1])), "r"(*reinterpret_cast<const uint32_t*>(&o0_hi[2])), "r"(*reinterpret_cast<const uint32_t*>(&o0_hi[3])));
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                    mbarrier_arrive(p_full_0_addr);
                    mbarrier_wait(corr_scale_1_addr, _phase_corr_scale_1_0);
                    _phase_corr_scale_1_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float _tmem_load_1[4];
                    tmem_ld_x4(&_tmem_load_1[0], taddr + 48 + (unsigned int)corr_row);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    mbarrier_arrive(corr_empty_1_addr);
                    mbarrier_wait(o_done_1_addr, _phase_o_done_1_0);
                    _phase_o_done_1_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float acc1_pair[2];
                    float2 _f2_25 = make_float2(_tmem_load_1[0], _tmem_load_1[1]);
                    float2 _f2_26 = make_float2(_tmem_load_1[2], _tmem_load_1[3]);
                    float2 max_diff1_pair_f2 = sub_f32x2(_f2_25, _f2_26);
                    float2 _f2_27 = make_float2(softmax_scale_log2, softmax_scale_log2);
                    float2 _mul_f32x2_7;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_7) : "l"(*(const unsigned long long*)&_f2_27), "l"(*(const unsigned long long*)&max_diff1_pair_f2));
                    float2 scaled_diff1_pair_f2 = _mul_f32x2_7;
                    float _exp2_12 = approx_exp2(scaled_diff1_pair_f2.x);
                    acc1_pair[0] = ((max_diff1_pair_f2.x != 0.0f) ? _exp2_12 : 1.0f);
                    float _exp2_13 = approx_exp2(scaled_diff1_pair_f2.y);
                    acc1_pair[1] = ((max_diff1_pair_f2.y != 0.0f) ? _exp2_13 : 1.0f);
                    int rescale_pred_1 = ((acc1_pair[0] != 1.0f) ? 1 : 0);
                    rescale_pred_1 = rescale_pred_1 | ((acc1_pair[1] != 1.0f) ? 1 : 0);
                    int _vote_2 = __any_sync(0xFFFFFFFF, rescale_pred_1 != 0);
                    if (_vote_2 != 0) {
                        float o1_lo[4];
                        float o1_hi[4];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&o1_lo[0])), "=r"(*reinterpret_cast<uint32_t*>(&o1_lo[1])), "=r"(*reinterpret_cast<uint32_t*>(&o1_lo[2])), "=r"(*reinterpret_cast<uint32_t*>(&o1_lo[3]))
                            : "r"(taddr + 88));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&o1_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&o1_hi[1])), "=r"(*reinterpret_cast<uint32_t*>(&o1_hi[2])), "=r"(*reinterpret_cast<uint32_t*>(&o1_hi[3]))
                            : "r"(taddr + 88 + 1048576));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        float o1[8];
                        #pragma unroll
                        for (int h_2 = 0; h_2 < 4; h_2++) {
                            o1[h_2] = o1_lo[h_2];
                            o1[h_2 + 4] = o1_hi[h_2];
                        }
                        float2 _f2_28 = make_float2(acc1_pair[0], acc1_pair[1]);
                        float2 _f2_29 = make_float2(o1[0], o1[1]);
                        float2 _f2_30 = make_float2(o1[2], o1[3]);
                        float2 _f2_31 = make_float2(o1[4], o1[5]);
                        float2 _f2_32 = make_float2(o1[6], o1[7]);
                        float2 _mul_f32x2_8;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_8) : "l"(*(const unsigned long long*)&_f2_29), "l"(*(const unsigned long long*)&_f2_28));
                        float2 o1_lo01_scaled_f2 = _mul_f32x2_8;
                        float2 _mul_f32x2_9;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_9) : "l"(*(const unsigned long long*)&_f2_30), "l"(*(const unsigned long long*)&_f2_28));
                        float2 o1_lo23_scaled_f2 = _mul_f32x2_9;
                        float2 _mul_f32x2_10;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_10) : "l"(*(const unsigned long long*)&_f2_31), "l"(*(const unsigned long long*)&_f2_28));
                        float2 o1_hi01_scaled_f2 = _mul_f32x2_10;
                        float2 _mul_f32x2_11;
                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_11) : "l"(*(const unsigned long long*)&_f2_32), "l"(*(const unsigned long long*)&_f2_28));
                        float2 o1_hi23_scaled_f2 = _mul_f32x2_11;
                        o1[0] = o1_lo01_scaled_f2.x;
                        o1[1] = o1_lo01_scaled_f2.y;
                        o1[2] = o1_lo23_scaled_f2.x;
                        o1[3] = o1_lo23_scaled_f2.y;
                        o1[4] = o1_hi01_scaled_f2.x;
                        o1[5] = o1_hi01_scaled_f2.y;
                        o1[6] = o1_hi23_scaled_f2.x;
                        o1[7] = o1_hi23_scaled_f2.y;
                        #pragma unroll
                        for (int h_3 = 0; h_3 < 4; h_3++) {
                            o1_lo[h_3] = o1[h_3];
                            o1_hi[h_3] = o1[h_3 + 4];
                        }
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x256b.x1.b32"
                            " [%0], {%1, %2, %3, %4};"
                            :: "r"(taddr + 88), "r"(*reinterpret_cast<const uint32_t*>(&o1_lo[0])), "r"(*reinterpret_cast<const uint32_t*>(&o1_lo[1])), "r"(*reinterpret_cast<const uint32_t*>(&o1_lo[2])), "r"(*reinterpret_cast<const uint32_t*>(&o1_lo[3])));
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x256b.x1.b32"
                            " [%0], {%1, %2, %3, %4};"
                            :: "r"(taddr + 88 + 1048576), "r"(*reinterpret_cast<const uint32_t*>(&o1_hi[0])), "r"(*reinterpret_cast<const uint32_t*>(&o1_hi[1])), "r"(*reinterpret_cast<const uint32_t*>(&o1_hi[2])), "r"(*reinterpret_cast<const uint32_t*>(&o1_hi[3])));
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                    mbarrier_arrive(p_full_1_addr);
                }
                mbarrier_wait(corr_scale_0_addr, _phase_corr_scale_0_0);
                _phase_corr_scale_0_0 ^= 1;
                mbarrier_wait(corr_scale_1_addr, _phase_corr_scale_1_0);
                _phase_corr_scale_1_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float scale0_pair[2];
                float scale1_pair[2];
                float local_sum_pair[2];
                const int idx0_c = col_pair_base_c;
                const int idx1_c = col_pair_base_c + 1;
                float _tmem_load_2[4];
                tmem_ld_x4(&_tmem_load_2[0], taddr + 16 + (unsigned int)corr_row);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float _tmem_load_3[4];
                tmem_ld_x4(&_tmem_load_3[0], taddr + 48 + (unsigned int)corr_row);
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                mbarrier_arrive(corr_empty_0_addr);
                mbarrier_arrive(corr_empty_1_addr);
                float2 _f2_33 = make_float2(_tmem_load_2[0], _tmem_load_2[1]);
                float2 _f2_34 = make_float2(_tmem_load_3[0], _tmem_load_3[1]);
                float m0_0 = _tmem_load_2[2];
                float m0_1 = _tmem_load_2[3];
                float m1_0 = _tmem_load_3[2];
                float m1_1 = _tmem_load_3[3];
                float _max_14 = max_noftz(m0_0, m1_0);
                float fm0 = _max_14;
                float _max_15 = max_noftz(m0_1, m1_1);
                float fm1 = _max_15;
                float2 _f2_35 = make_float2(m0_0, m0_1);
                float2 _f2_36 = make_float2(m1_0, m1_1);
                float2 _f2_37 = make_float2(fm0, fm1);
                float2 max_diff_i0_f2 = sub_f32x2(_f2_35, _f2_37);
                float2 max_diff_i1_f2 = sub_f32x2(_f2_36, _f2_37);
                float2 _f2_38 = make_float2(softmax_scale_log2, softmax_scale_log2);
                float2 _mul_f32x2_12;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_12) : "l"(*(const unsigned long long*)&_f2_38), "l"(*(const unsigned long long*)&max_diff_i0_f2));
                float2 d0_pair_f2 = _mul_f32x2_12;
                float2 _mul_f32x2_13;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_13) : "l"(*(const unsigned long long*)&_f2_38), "l"(*(const unsigned long long*)&max_diff_i1_f2));
                float2 d1_pair_f2 = _mul_f32x2_13;
                float _exp2_14 = approx_exp2(d0_pair_f2.x);
                scale0_pair[0] = ((m0_0 == -CAKE_FMHA_INF) ? 0.0f : _exp2_14);
                float _exp2_15 = approx_exp2(d0_pair_f2.y);
                scale0_pair[1] = ((m0_1 == -CAKE_FMHA_INF) ? 0.0f : _exp2_15);
                float _exp2_16 = approx_exp2(d1_pair_f2.x);
                scale1_pair[0] = ((m1_0 == -CAKE_FMHA_INF) ? 0.0f : _exp2_16);
                float _exp2_17 = approx_exp2(d1_pair_f2.y);
                scale1_pair[1] = ((m1_1 == -CAKE_FMHA_INF) ? 0.0f : _exp2_17);
                float2 _f2_39 = make_float2(scale0_pair[0], scale0_pair[1]);
                float2 _f2_40 = make_float2(scale1_pair[0], scale1_pair[1]);
                float2 _mul_f32x2_14;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_14) : "l"(*(const unsigned long long*)&_f2_34), "l"(*(const unsigned long long*)&_f2_40));
                float2 s1_scaled_pair_f2 = _mul_f32x2_14;
                float2 local_sum_pair_f2 = fma_f32x2_rn_ftz(_f2_33, _f2_39, s1_scaled_pair_f2);
                float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, local_sum_pair_f2.x, 16);
                float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, local_sum_pair_f2.y, 16);
                float2 _f2_41 = make_float2(_shfl_xor_2, _shfl_xor_3);
                float2 sum16_pair_f2 = add_f32x2(local_sum_pair_f2, _f2_41);
                float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, sum16_pair_f2.x, 8);
                float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, sum16_pair_f2.y, 8);
                float2 _f2_42 = make_float2(_shfl_xor_4, _shfl_xor_5);
                float2 sum8_pair_f2 = add_f32x2(sum16_pair_f2, _f2_42);
                float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, sum8_pair_f2.x, 4);
                float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, sum8_pair_f2.y, 4);
                float2 _f2_43 = make_float2(_shfl_xor_6, _shfl_xor_7);
                float2 reduced_sum_pair_f2 = add_f32x2(sum8_pair_f2, _f2_43);
                if (lane < 4) {
                    const int warp_sum_base_c = warp % 4 * 8 + col_pair_base_c;
                    smem_corr_reduce[warp_sum_base_c] = reduced_sum_pair_f2.x;
                    smem_corr_reduce[warp_sum_base_c + 1] = reduced_sum_pair_f2.y;
                }
                asm volatile("barrier.sync 10, 128;" ::: "memory");
                float2 _f2_44 = make_float2(smem_corr_reduce[idx0_c], smem_corr_reduce[idx1_c]);
                float2 _f2_45 = make_float2(smem_corr_reduce[idx0_c + 8], smem_corr_reduce[idx1_c + 8]);
                float2 _f2_46 = make_float2(smem_corr_reduce[idx0_c + 16], smem_corr_reduce[idx1_c + 16]);
                float2 _f2_47 = make_float2(smem_corr_reduce[idx0_c + 24], smem_corr_reduce[idx1_c + 24]);
                float2 sum_w01_pair_f2 = add_f32x2(_f2_44, _f2_45);
                float2 sum_w23_pair_f2 = add_f32x2(_f2_46, _f2_47);
                float2 sum_reduced_pair_f2 = add_f32x2(sum_w01_pair_f2, sum_w23_pair_f2);
                local_sum_pair[0] = sum_reduced_pair_f2.x;
                local_sum_pair[1] = sum_reduced_pair_f2.y;
                mbarrier_wait(o_done_0_addr, _phase_o_done_0_0);
                _phase_o_done_0_0 ^= 1;
                mbarrier_wait(o_done_1_addr, _phase_o_done_1_0);
                _phase_o_done_1_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float inv_sum_pair[2];
                #pragma unroll
                for (int c_4 = 0; c_4 < 2; c_4++) {
                    float _rcp_1 = approx_rcp(local_sum_pair[c_4]);
                    inv_sum_pair[c_4] = _rcp_1;
                }
                float o0_lo_epi[4];
                float o0_hi_epi[4];
                float o1_lo_epi[4];
                float o1_hi_epi[4];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&o0_lo_epi[0])), "=r"(*reinterpret_cast<uint32_t*>(&o0_lo_epi[1])), "=r"(*reinterpret_cast<uint32_t*>(&o0_lo_epi[2])), "=r"(*reinterpret_cast<uint32_t*>(&o0_lo_epi[3]))
                    : "r"(taddr + 80));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&o0_hi_epi[0])), "=r"(*reinterpret_cast<uint32_t*>(&o0_hi_epi[1])), "=r"(*reinterpret_cast<uint32_t*>(&o0_hi_epi[2])), "=r"(*reinterpret_cast<uint32_t*>(&o0_hi_epi[3]))
                    : "r"(taddr + 80 + 1048576));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&o1_lo_epi[0])), "=r"(*reinterpret_cast<uint32_t*>(&o1_lo_epi[1])), "=r"(*reinterpret_cast<uint32_t*>(&o1_lo_epi[2])), "=r"(*reinterpret_cast<uint32_t*>(&o1_lo_epi[3]))
                    : "r"(taddr + 88));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                    " {%0, %1, %2, %3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&o1_hi_epi[0])), "=r"(*reinterpret_cast<uint32_t*>(&o1_hi_epi[1])), "=r"(*reinterpret_cast<uint32_t*>(&o1_hi_epi[2])), "=r"(*reinterpret_cast<uint32_t*>(&o1_hi_epi[3]))
                    : "r"(taddr + 88 + 1048576));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                if (elect_sync()) {
                    mbarrier_arrive(stats_empty_addr);
                }
                int publish_split = ((n_chunks_c > 1) ? 1 : 0);
                float norm_in0 = 1.0f;
                float norm_in1 = 1.0f;
                float norm_mult = 1.0f;
                if (publish_split == 0) {
                    norm_in0 = inv_sum_pair[0];
                    norm_in1 = inv_sum_pair[1];
                    norm_mult = output_scale;
                }
                float out_vals[8];
                float2 _f2_48 = make_float2(norm_in0, norm_in1);
                float2 _f2_49 = make_float2(norm_mult, norm_mult);
                float2 _mul_f32x2_15;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_15) : "l"(*(const unsigned long long*)&_f2_48), "l"(*(const unsigned long long*)&_f2_49));
                float2 norm_pair_f2 = _mul_f32x2_15;
                float2 _mul_f32x2_16;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_16) : "l"(*(const unsigned long long*)&_f2_39), "l"(*(const unsigned long long*)&norm_pair_f2));
                float2 final_scale0_pair_f2 = _mul_f32x2_16;
                float2 _mul_f32x2_17;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_17) : "l"(*(const unsigned long long*)&_f2_40), "l"(*(const unsigned long long*)&norm_pair_f2));
                float2 final_scale1_pair_f2 = _mul_f32x2_17;
                float2 _f2_50 = make_float2(o0_lo_epi[0], o0_lo_epi[1]);
                float2 _f2_51 = make_float2(o1_lo_epi[0], o1_lo_epi[1]);
                float2 _mul_f32x2_18;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_18) : "l"(*(const unsigned long long*)&_f2_51), "l"(*(const unsigned long long*)&final_scale1_pair_f2));
                float2 o1_lo01_scaled_f2_1 = _mul_f32x2_18;
                float2 out_lo01_f2 = fma_f32x2_rn_ftz(_f2_50, final_scale0_pair_f2, o1_lo01_scaled_f2_1);
                out_vals[0] = out_lo01_f2.x;
                out_vals[1] = out_lo01_f2.y;
                float2 _f2_52 = make_float2(o0_lo_epi[2], o0_lo_epi[3]);
                float2 _f2_53 = make_float2(o1_lo_epi[2], o1_lo_epi[3]);
                float2 _mul_f32x2_19;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_19) : "l"(*(const unsigned long long*)&_f2_53), "l"(*(const unsigned long long*)&final_scale1_pair_f2));
                float2 o1_lo23_scaled_f2_1 = _mul_f32x2_19;
                float2 out_lo23_f2 = fma_f32x2_rn_ftz(_f2_52, final_scale0_pair_f2, o1_lo23_scaled_f2_1);
                out_vals[2] = out_lo23_f2.x;
                out_vals[3] = out_lo23_f2.y;
                float2 _f2_54 = make_float2(o0_hi_epi[0], o0_hi_epi[1]);
                float2 _f2_55 = make_float2(o1_hi_epi[0], o1_hi_epi[1]);
                float2 _mul_f32x2_20;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_20) : "l"(*(const unsigned long long*)&_f2_55), "l"(*(const unsigned long long*)&final_scale1_pair_f2));
                float2 o1_hi01_scaled_f2_1 = _mul_f32x2_20;
                float2 out_hi01_f2 = fma_f32x2_rn_ftz(_f2_54, final_scale0_pair_f2, o1_hi01_scaled_f2_1);
                out_vals[4] = out_hi01_f2.x;
                out_vals[5] = out_hi01_f2.y;
                float2 _f2_56 = make_float2(o0_hi_epi[2], o0_hi_epi[3]);
                float2 _f2_57 = make_float2(o1_hi_epi[2], o1_hi_epi[3]);
                float2 _mul_f32x2_21;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_21) : "l"(*(const unsigned long long*)&_f2_57), "l"(*(const unsigned long long*)&final_scale1_pair_f2));
                float2 o1_hi23_scaled_f2_1 = _mul_f32x2_21;
                float2 out_hi23_f2 = fma_f32x2_rn_ftz(_f2_56, final_scale0_pair_f2, o1_hi23_scaled_f2_1);
                out_vals[6] = out_hi23_f2.x;
                out_vals[7] = out_hi23_f2.y;
                int store_now = 1;
                int merged_now = 0;
                if (publish_split != 0) {
                    store_now = 0;
                    #pragma unroll
                    for (int k = 0; k < 4; k++) {
                        *(reinterpret_cast<float*>(partial_o + (partial_o_base + k * 8 + col_pair_base_c * HEAD_DIM)) + (0)) = out_vals[2 * k];
                        *(reinterpret_cast<float*>(partial_o + (partial_o_base + k * 8 + (col_pair_base_c + 1) * HEAD_DIM)) + (0)) = out_vals[2 * k + 1];
                    }
                    if (corr_tid < 4) {
                        *(reinterpret_cast<float*>(partial_stats + (partial_stats_base + col_pair_base_c)) + (0)) = fm0;
                        *(reinterpret_cast<float*>(partial_stats + (partial_stats_base + col_pair_base_c + 1)) + (0)) = fm1;
                        *(reinterpret_cast<float*>(partial_stats + (partial_stats_base + 8 + col_pair_base_c)) + (0)) = local_sum_pair[0];
                        *(reinterpret_cast<float*>(partial_stats + (partial_stats_base + 8 + col_pair_base_c + 1)) + (0)) = local_sum_pair[1];
                    }
                    asm volatile("barrier.sync 12, 128;" ::: "memory");
                    if (corr_tid == 0) {
                        asm volatile("fence.release.gpu;" ::: "memory");
                        unsigned int _atomic_old_2;
                        asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                            : "=r"(_atomic_old_2) : "l"(&tile_counters[counter_idx_c]), "r"(static_cast<uint32_t>(1)) : "memory");
                        unsigned int old_count = _atomic_old_2;
                        smem_merge_flag[0] = (((int)old_count + 1 == n_chunks_c) ? 1 : 0);
                    }
                    asm volatile("barrier.sync 12, 128;" ::: "memory");
                    unsigned int merge_flag = smem_merge_flag[0];
                    if (merge_flag != 0) {
                        asm volatile("fence.acquire.gpu;" ::: "memory");
                        int m_head = corr_tid / 16;
                        int m_d0 = corr_tid % 16 * 8;
                        int n_pad_m = (n_chunks_c + 7) / 8 * 8;
                        float acc_m[8];
                        acc_m[0] = 0.0f;
                        acc_m[1] = 0.0f;
                        acc_m[2] = 0.0f;
                        acc_m[3] = 0.0f;
                        acc_m[4] = 0.0f;
                        acc_m[5] = 0.0f;
                        acc_m[6] = 0.0f;
                        acc_m[7] = 0.0f;
                        float m_run = -1e+30f;
                        float l_run = 0.0f;
                        #pragma unroll 8
                        for (int c_m = 0; c_m < n_pad_m; c_m++) {
                            int c_c = c_m;
                            if (n_chunks_c <= c_m) {
                                c_c = n_chunks_c - 1;
                            }
                            int slot_m = slot_tile_base_c + c_c * slot_stride;
                            int stats_m = slot_m * 16;
                            float m_k = partial_stats[stats_m + m_head];
                            float l_k = partial_stats[stats_m + 8 + m_head];
                            if (n_chunks_c <= c_m) {
                                m_k = -1e+30f;
                                l_k = 0.0f;
                            }
                            float _max_16 = max_noftz(m_run, m_k);
                            float m_new = _max_16;
                            float _exp2_18 = approx_exp2((m_run - m_new) * softmax_scale_log2);
                            float a_k = _exp2_18;
                            float _exp2_19 = approx_exp2((m_k - m_new) * softmax_scale_log2);
                            float b_k = _exp2_19;
                            float _fma_0 = __fmaf_rn(l_k, b_k, l_run * a_k);
                            l_run = _fma_0;
                            int o_m_base = slot_m * 1024 + m_head * HEAD_DIM + m_d0;
                            float _vec_load_0[4];
                            {
                                float4 _v4 = *reinterpret_cast<const float4*>(partial_o + o_m_base + 0);
                                _vec_load_0[0 + 0] = _v4.x;
                                _vec_load_0[0 + 1] = _v4.y;
                                _vec_load_0[0 + 2] = _v4.z;
                                _vec_load_0[0 + 3] = _v4.w;
                            }
                            float _vec_load_1[4];
                            {
                                float4 _v4 = *reinterpret_cast<const float4*>(partial_o + (o_m_base + 4) + 0);
                                _vec_load_1[0 + 0] = _v4.x;
                                _vec_load_1[0 + 1] = _v4.y;
                                _vec_load_1[0 + 2] = _v4.z;
                                _vec_load_1[0 + 3] = _v4.w;
                            }
                            #pragma unroll
                            for (int k_1 = 0; k_1 < 4; k_1++) {
                                float _fma_1 = __fmaf_rn(_vec_load_0[k_1], b_k, acc_m[k_1] * a_k);
                                acc_m[k_1] = _fma_1;
                                float _fma_2 = __fmaf_rn(_vec_load_1[k_1], b_k, acc_m[4 + k_1] * a_k);
                                acc_m[4 + k_1] = _fma_2;
                            }
                            m_run = m_new;
                        }
                        float inv_m = 1.0f / l_run * output_scale;
                        #pragma unroll
                        for (int k_2 = 0; k_2 < 8; k_2++) {
                            acc_m[k_2] = acc_m[k_2] * inv_m;
                        }
                        if (m_head < group_ratio) {
                            float acc_lo[4];
                            float acc_hi[4];
                            #pragma unroll
                            for (int k_3 = 0; k_3 < 4; k_3++) {
                                acc_lo[k_3] = acc_m[k_3];
                                acc_hi[k_3] = acc_m[4 + k_3];
                            }
                            int o_idx_m = (row_tile_c * num_q_heads + kv_head_idx * group_ratio + m_head) * HEAD_DIM + m_d0;
                            {
                                uint2 _pk2;
                                __half2* _pk = reinterpret_cast<__half2*>(&_pk2);
                                _pk[0] = __floats2half2_rn(acc_lo[0 + 0], acc_lo[0 + 1]);
                                _pk[1] = __floats2half2_rn(acc_lo[0 + 2], acc_lo[0 + 3]);
                                *reinterpret_cast<uint2*>(&((__half*)(O_ptr + o_idx_m))[0]) = _pk2;
                            }
                            {
                                uint2 _pk2;
                                __half2* _pk = reinterpret_cast<__half2*>(&_pk2);
                                _pk[0] = __floats2half2_rn(acc_hi[0 + 0], acc_hi[0 + 1]);
                                _pk[1] = __floats2half2_rn(acc_hi[0 + 2], acc_hi[0 + 3]);
                                *reinterpret_cast<uint2*>(&((__half*)(O_ptr + (o_idx_m + 4)))[0]) = _pk2;
                            }
                        }
                        if (corr_tid == 0) {
                            *(reinterpret_cast<unsigned int*>(tile_counters + counter_idx_c) + (0)) = 0;
                        }
                        merged_now = 1;
                    }
                    asm volatile("barrier.sync 12, 128;" ::: "memory");
                }
                if (store_now != 0) {
                    #pragma unroll
                    for (int o_pair = 0; o_pair < 4; o_pair++) {
                        int o_dim = o_row_base + o_pair * 8;
                        int o_q0 = col_pair_base_c;
                        int o_q1 = col_pair_base_c + 1;
                        if (o_q0 < group_ratio) {
                            int q_head0 = kv_head_idx * group_ratio + o_q0;
                            int out_idx0 = (row_tile_c * num_q_heads + q_head0) * HEAD_DIM + o_dim;
                            *(reinterpret_cast<__half*>(O_ptr + out_idx0) + (0)) = __float2half_rn(out_vals[o_pair * 2]);
                        }
                        if (o_q1 < group_ratio) {
                            int q_head1 = kv_head_idx * group_ratio + o_q1;
                            int out_idx1 = (row_tile_c * num_q_heads + q_head1) * HEAD_DIM + o_dim;
                            *(reinterpret_cast<__half*>(O_ptr + out_idx1) + (0)) = __float2half_rn(out_vals[o_pair * 2 + 1]);
                        }
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_c) * 8, _phase_work_full_1);
                unsigned int base_0_1 = work_stage_c * 16;
                unsigned int valid_1_1 = work_token_words[base_0_1];
                unsigned int row_tile_2_1 = work_token_words[base_0_1 + 1];
                unsigned int batch_3_1 = work_token_words[base_0_1 + 2];
                unsigned int kv_head_4_1 = work_token_words[base_0_1 + 3];
                unsigned int block_begin_5_1 = work_token_words[base_0_1 + 4];
                unsigned int block_end_6_1 = work_token_words[base_0_1 + 5];
                unsigned int seqlen_row_7_1 = work_token_words[base_0_1 + 6];
                unsigned int n_chunks_8_1 = work_token_words[base_0_1 + 7];
                unsigned int slot_tile_base_9_1 = work_token_words[base_0_1 + 8];
                unsigned int counter_idx_10_1 = work_token_words[base_0_1 + 9];
                unsigned int chunk_11_1 = work_token_words[base_0_1 + 10];
                mbarrier_arrive(work_empty_addr + (work_stage_c) * 8);
                work_stage_c += 1;
                if (work_stage_c == 4) { work_stage_c = 0; _phase_work_full_1 ^= 1; }
                valid_c = valid_1_1;
                row_tile_c = (int)row_tile_2_1;
                kv_head_idx = (int)kv_head_4_1;
                block_begin_c = (int)block_begin_5_1;
                block_end_c = (int)block_end_6_1;
                n_chunks_c = (int)n_chunks_8_1;
                slot_tile_base_c = (int)slot_tile_base_9_1;
                counter_idx_c = (int)counter_idx_10_1;
                chunk_c = (int)chunk_11_1;
            }
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: mma_warp ----
    if (warp == 12) {
        { // mma_warp_main
            int q_slot_m = 0;
            int q_phase_m = 0;
            int kv_base_m = 0;
            unsigned int work_stage_m = 0;
            unsigned int _phase_work_full_2 = 0;
            mbarrier_wait(work_full_addr + (work_stage_m) * 8, _phase_work_full_2);
            unsigned int base_2 = work_stage_m * 16;
            unsigned int valid_3 = work_token_words[base_2];
            unsigned int row_tile_3 = work_token_words[base_2 + 1];
            unsigned int batch_2 = work_token_words[base_2 + 2];
            unsigned int kv_head_2 = work_token_words[base_2 + 3];
            unsigned int block_begin_2 = work_token_words[base_2 + 4];
            unsigned int block_end_2 = work_token_words[base_2 + 5];
            unsigned int seqlen_row_2 = work_token_words[base_2 + 6];
            unsigned int n_chunks_2 = work_token_words[base_2 + 7];
            unsigned int slot_tile_base_2 = work_token_words[base_2 + 8];
            unsigned int counter_idx_2 = work_token_words[base_2 + 9];
            unsigned int chunk_2 = work_token_words[base_2 + 10];
            mbarrier_arrive(work_empty_addr + (work_stage_m) * 8);
            work_stage_m += 1;
            if (work_stage_m == 4) { work_stage_m = 0; _phase_work_full_2 ^= 1; }
            unsigned int valid_m = valid_3;
            int block_begin_m = (int)block_begin_2;
            int block_end_m = (int)block_end_2;
            unsigned int _phase_s_empty_0_0 = 1;
            unsigned int _phase_s_empty_1_0 = 1;
            unsigned int _phase_p_full_0_0 = 0;
            unsigned int _phase_p_full_1_0 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_m = 0; _tile_iter_m < max_items; _tile_iter_m++) {
                if (valid_m == 0) {
                    break;
                }
                int num_n_blocks_total_2 = block_end_m - block_begin_m;
                int cta_n_blocks_2 = num_n_blocks_total_2 + num_n_blocks_total_2 % 2;
                int num_pairs_2 = cta_n_blocks_2 / 2;
                int inst0_stage = kv_base_m;
                int kv_base1_m = (kv_base_m + 1) % NUM_KV_STAGES;
                int first_pv0 = 1;
                int first_pv1 = 1;
                mbarrier_wait(q_full_addr + (q_slot_m) * 8, q_phase_m);
                mbarrier_wait(s_empty_0_addr, _phase_s_empty_0_0);
                _phase_s_empty_0_0 ^= 1;
                mbarrier_wait(kv_full_addr + (kv_base_m) * 8, 0);
                int _mma_a_lo_0 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (kv_base_m) * 1024);
                int _mma_b_lo_0 = make_warp_uniform((((smem_qt_addr) >> 4) & 0x3FFF) + (q_slot_m) * 64);
                {
                    uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
                    uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_tmem_s0, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134348816, 0);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_tmem_s0, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134348816, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_tmem_s0, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134348816, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_tmem_s0, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134348816, 1);
                    }
                }
                int _mma_a_lo_1 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (kv_base_m) * 1024);
                int _mma_b_lo_1 = make_warp_uniform((((smem_qt_lo_addr) >> 4) & 0x3FFF) + (q_slot_m) * 64);
                {
                    uint64_t _mma_ss_a_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_1);
                    uint64_t _mma_ss_b_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_1);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_tmem_s0, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134348816, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_tmem_s0, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134348816, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_tmem_s0, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134348816, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_tmem_s0, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134348816, 1);
                    }
                }
                elect_commit(kv_empty_addr + (kv_base_m) * 8);
                elect_commit(s_full_0_addr);
                mbarrier_wait(s_empty_1_addr, _phase_s_empty_1_0);
                _phase_s_empty_1_0 ^= 1;
                mbarrier_wait(kv_full_addr + (kv_base1_m) * 8, 0);
                int _mma_a_lo_2 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (kv_base1_m) * 1024);
                int _mma_b_lo_2 = make_warp_uniform((((smem_qt_addr) >> 4) & 0x3FFF) + (q_slot_m) * 64);
                {
                    uint64_t _mma_ss_a_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_2);
                    uint64_t _mma_ss_b_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_2);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_tmem_s1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134348816, 0);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_tmem_s1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134348816, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_tmem_s1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134348816, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_tmem_s1, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134348816, 1);
                    }
                }
                int _mma_a_lo_3 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (kv_base1_m) * 1024);
                int _mma_b_lo_3 = make_warp_uniform((((smem_qt_lo_addr) >> 4) & 0x3FFF) + (q_slot_m) * 64);
                {
                    uint64_t _mma_ss_a_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_3);
                    uint64_t _mma_ss_b_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_3);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_tmem_s1, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134348816, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_tmem_s1, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134348816, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_tmem_s1, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134348816, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_tmem_s1, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134348816, 1);
                    }
                }
                elect_commit(kv_empty_addr + (kv_base1_m) * 8);
                elect_commit(s_full_1_addr);
                #pragma unroll 1
                for (int pair_2 = 0; pair_2 < num_pairs_2 - 1; pair_2++) {
                    int s0 = inst0_stage;
                    int s1 = (inst0_stage + 1) % NUM_KV_STAGES;
                    int s0_next = (inst0_stage + 2) % NUM_KV_STAGES;
                    int s1_next = (inst0_stage + 3) % NUM_KV_STAGES;
                    mbarrier_wait(s_empty_0_addr, _phase_s_empty_0_0);
                    _phase_s_empty_0_0 ^= 1;
                    mbarrier_wait(kv_full_addr + (s0_next) * 8, 0);
                    int _mma_a_lo_4 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (s0_next) * 1024);
                    int _mma_b_lo_4 = make_warp_uniform((((smem_qt_addr) >> 4) & 0x3FFF) + (q_slot_m) * 64);
                    {
                        uint64_t _mma_ss_a_desc_4 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_4);
                        uint64_t _mma_ss_b_desc_4 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_4);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4(tmem_tmem_s0, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134348816, 0);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4(tmem_tmem_s0, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134348816, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4(tmem_tmem_s0, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134348816, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_4, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4(tmem_tmem_s0, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134348816, 1);
                        }
                    }
                    int _mma_a_lo_5 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (s0_next) * 1024);
                    int _mma_b_lo_5 = make_warp_uniform((((smem_qt_lo_addr) >> 4) & 0x3FFF) + (q_slot_m) * 64);
                    {
                        uint64_t _mma_ss_a_desc_5 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_5);
                        uint64_t _mma_ss_b_desc_5 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_5);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4(tmem_tmem_s0, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134348816, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_5, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4(tmem_tmem_s0, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134348816, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_5, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4(tmem_tmem_s0, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134348816, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_5, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4(tmem_tmem_s0, _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134348816, 1);
                        }
                    }
                    elect_commit(kv_empty_addr + (s0_next) * 8);
                    elect_commit(s_full_0_addr);
                    mbarrier_wait(p_full_0_addr, _phase_p_full_0_0);
                    _phase_p_full_0_0 ^= 1;
                    mbarrier_wait(kv_full_addr + (s0) * 8, 1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_6 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (s0) * 1024);
                    int _mma_b_lo_6 = make_warp_uniform((((smem_p0_addr) >> 4) & 0x3FFF) | 0x400000);
                    {
                        uint64_t _mma_ss_a_desc_6 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_6);
                        uint64_t _mma_ss_b_desc_6 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_6);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4(tmem_tmem_o0, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134381584, ((first_pv0) ? 0 : 1));
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_6, 256U);
                        incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4(tmem_tmem_o0, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134381584, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_6, 256U);
                        incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4(tmem_tmem_o0, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134381584, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_6, 256U);
                        incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4(tmem_tmem_o0, _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134381584, 1);
                        }
                    }
                    elect_commit2(kv_empty_addr + (s0) * 8, o_done_0_addr);
                    elect_commit(p_empty_0_addr);
                    mbarrier_wait(s_empty_1_addr, _phase_s_empty_1_0);
                    _phase_s_empty_1_0 ^= 1;
                    mbarrier_wait(kv_full_addr + (s1_next) * 8, 0);
                    int _mma_a_lo_7 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (s1_next) * 1024);
                    int _mma_b_lo_7 = make_warp_uniform((((smem_qt_addr) >> 4) & 0x3FFF) + (q_slot_m) * 64);
                    {
                        uint64_t _mma_ss_a_desc_7 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_7);
                        uint64_t _mma_ss_b_desc_7 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_7);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4(tmem_tmem_s1, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134348816, 0);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_7, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4(tmem_tmem_s1, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134348816, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_7, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4(tmem_tmem_s1, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134348816, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_7, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4(tmem_tmem_s1, _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134348816, 1);
                        }
                    }
                    int _mma_a_lo_8 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (s1_next) * 1024);
                    int _mma_b_lo_8 = make_warp_uniform((((smem_qt_lo_addr) >> 4) & 0x3FFF) + (q_slot_m) * 64);
                    {
                        uint64_t _mma_ss_a_desc_8 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_8);
                        uint64_t _mma_ss_b_desc_8 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_8);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4(tmem_tmem_s1, _mma_ss_a_desc_8, _mma_ss_b_desc_8, 134348816, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_8, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_8, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4(tmem_tmem_s1, _mma_ss_a_desc_8, _mma_ss_b_desc_8, 134348816, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_8, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_8, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4(tmem_tmem_s1, _mma_ss_a_desc_8, _mma_ss_b_desc_8, 134348816, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_8, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_8, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4(tmem_tmem_s1, _mma_ss_a_desc_8, _mma_ss_b_desc_8, 134348816, 1);
                        }
                    }
                    elect_commit(kv_empty_addr + (s1_next) * 8);
                    elect_commit(s_full_1_addr);
                    mbarrier_wait(p_full_1_addr, _phase_p_full_1_0);
                    _phase_p_full_1_0 ^= 1;
                    mbarrier_wait(kv_full_addr + (s1) * 8, 1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_9 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (s1) * 1024);
                    int _mma_b_lo_9 = make_warp_uniform((((smem_p1_addr) >> 4) & 0x3FFF) | 0x400000);
                    {
                        uint64_t _mma_ss_a_desc_9 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_9);
                        uint64_t _mma_ss_b_desc_9 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_9);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4(tmem_tmem_o1, _mma_ss_a_desc_9, _mma_ss_b_desc_9, 134381584, ((first_pv1) ? 0 : 1));
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_9, 256U);
                        incr_smem_desc_lo(_mma_ss_b_desc_9, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4(tmem_tmem_o1, _mma_ss_a_desc_9, _mma_ss_b_desc_9, 134381584, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_9, 256U);
                        incr_smem_desc_lo(_mma_ss_b_desc_9, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4(tmem_tmem_o1, _mma_ss_a_desc_9, _mma_ss_b_desc_9, 134381584, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_9, 256U);
                        incr_smem_desc_lo(_mma_ss_b_desc_9, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4(tmem_tmem_o1, _mma_ss_a_desc_9, _mma_ss_b_desc_9, 134381584, 1);
                        }
                    }
                    elect_commit2(kv_empty_addr + (s1) * 8, o_done_1_addr);
                    elect_commit(p_empty_1_addr);
                    inst0_stage = s0_next;
                    first_pv0 = 0;
                    first_pv1 = 0;
                }
                elect_commit(q_empty_addr + (q_slot_m) * 8);
                q_slot_m += 1;
                if (q_slot_m == 2) { q_slot_m = 0; q_phase_m ^= 1; }
                int s0_last = inst0_stage;
                int s1_last = (inst0_stage + 1) % NUM_KV_STAGES;
                mbarrier_wait(p_full_0_addr, _phase_p_full_0_0);
                _phase_p_full_0_0 ^= 1;
                mbarrier_wait(kv_full_addr + (s0_last) * 8, 1);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_a_lo_10 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (s0_last) * 1024);
                int _mma_b_lo_10 = make_warp_uniform((((smem_p0_addr) >> 4) & 0x3FFF) | 0x400000);
                {
                    uint64_t _mma_ss_a_desc_10 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_10);
                    uint64_t _mma_ss_b_desc_10 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_10);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_tmem_o0, _mma_ss_a_desc_10, _mma_ss_b_desc_10, 134381584, ((first_pv0) ? 0 : 1));
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_10, 256U);
                    incr_smem_desc_lo(_mma_ss_b_desc_10, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_tmem_o0, _mma_ss_a_desc_10, _mma_ss_b_desc_10, 134381584, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_10, 256U);
                    incr_smem_desc_lo(_mma_ss_b_desc_10, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_tmem_o0, _mma_ss_a_desc_10, _mma_ss_b_desc_10, 134381584, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_10, 256U);
                    incr_smem_desc_lo(_mma_ss_b_desc_10, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_tmem_o0, _mma_ss_a_desc_10, _mma_ss_b_desc_10, 134381584, 1);
                    }
                }
                elect_commit2(kv_empty_addr + (s0_last) * 8, o_done_0_addr);
                elect_commit(p_empty_0_addr);
                mbarrier_wait(p_full_1_addr, _phase_p_full_1_0);
                _phase_p_full_1_0 ^= 1;
                mbarrier_wait(kv_full_addr + (s1_last) * 8, 1);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_a_lo_11 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (s1_last) * 1024);
                int _mma_b_lo_11 = make_warp_uniform((((smem_p1_addr) >> 4) & 0x3FFF) | 0x400000);
                {
                    uint64_t _mma_ss_a_desc_11 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_11);
                    uint64_t _mma_ss_b_desc_11 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_11);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_tmem_o1, _mma_ss_a_desc_11, _mma_ss_b_desc_11, 134381584, ((first_pv1) ? 0 : 1));
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_11, 256U);
                    incr_smem_desc_lo(_mma_ss_b_desc_11, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_tmem_o1, _mma_ss_a_desc_11, _mma_ss_b_desc_11, 134381584, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_11, 256U);
                    incr_smem_desc_lo(_mma_ss_b_desc_11, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_tmem_o1, _mma_ss_a_desc_11, _mma_ss_b_desc_11, 134381584, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_11, 256U);
                    incr_smem_desc_lo(_mma_ss_b_desc_11, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_tmem_o1, _mma_ss_a_desc_11, _mma_ss_b_desc_11, 134381584, 1);
                    }
                }
                elect_commit2(kv_empty_addr + (s1_last) * 8, o_done_1_addr);
                elect_commit(p_empty_1_addr);
                kv_base_m = (kv_base_m + cta_n_blocks_2) % NUM_KV_STAGES;
                mbarrier_wait(work_full_addr + (work_stage_m) * 8, _phase_work_full_2);
                unsigned int base_0_2 = work_stage_m * 16;
                unsigned int valid_1_2 = work_token_words[base_0_2];
                unsigned int row_tile_2_2 = work_token_words[base_0_2 + 1];
                unsigned int batch_3_2 = work_token_words[base_0_2 + 2];
                unsigned int kv_head_4_2 = work_token_words[base_0_2 + 3];
                unsigned int block_begin_5_2 = work_token_words[base_0_2 + 4];
                unsigned int block_end_6_2 = work_token_words[base_0_2 + 5];
                unsigned int seqlen_row_7_2 = work_token_words[base_0_2 + 6];
                unsigned int n_chunks_8_2 = work_token_words[base_0_2 + 7];
                unsigned int slot_tile_base_9_2 = work_token_words[base_0_2 + 8];
                unsigned int counter_idx_10_2 = work_token_words[base_0_2 + 9];
                unsigned int chunk_11_2 = work_token_words[base_0_2 + 10];
                mbarrier_arrive(work_empty_addr + (work_stage_m) * 8);
                work_stage_m += 1;
                if (work_stage_m == 4) { work_stage_m = 0; _phase_work_full_2 ^= 1; }
                valid_m = valid_1_2;
                block_begin_m = (int)block_begin_5_2;
                block_end_m = (int)block_end_6_2;
            }
            unsigned int _phase_tmem_dealloc_0 = 0;
            mbarrier_wait(tmem_dealloc_addr, _phase_tmem_dealloc_0);
            _phase_tmem_dealloc_0 ^= 1;
            int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(128));
        }
    }
    // ---- Role: load_pgoff ----
    if (warp == 13) {
        { // load_pgoff_main
            int pg_slot_p = 0;
            int pg_phase_p = 1;
            unsigned int work_stage_p = 0;
            unsigned int _phase_work_full_3 = 0;
            mbarrier_wait(work_full_addr + (work_stage_p) * 8, _phase_work_full_3);
            unsigned int base_3 = work_stage_p * 16;
            unsigned int valid_4 = work_token_words[base_3];
            unsigned int row_tile_4 = work_token_words[base_3 + 1];
            unsigned int batch_4 = work_token_words[base_3 + 2];
            unsigned int kv_head_3 = work_token_words[base_3 + 3];
            unsigned int block_begin_3 = work_token_words[base_3 + 4];
            unsigned int block_end_3 = work_token_words[base_3 + 5];
            unsigned int seqlen_row_3 = work_token_words[base_3 + 6];
            unsigned int n_chunks_3 = work_token_words[base_3 + 7];
            unsigned int slot_tile_base_3 = work_token_words[base_3 + 8];
            unsigned int counter_idx_3 = work_token_words[base_3 + 9];
            unsigned int chunk_3 = work_token_words[base_3 + 10];
            mbarrier_arrive(work_empty_addr + (work_stage_p) * 8);
            work_stage_p += 1;
            if (work_stage_p == 4) { work_stage_p = 0; _phase_work_full_3 ^= 1; }
            unsigned int valid_p = valid_4;
            int batch_idx_p = (int)batch_4;
            int block_begin_p = (int)block_begin_3;
            int block_end_p = (int)block_end_3;
            int seqlen_kv_p = (int)seqlen_row_3;
            #pragma unroll 1
            for (unsigned int _tile_iter_p = 0; _tile_iter_p < max_items; _tile_iter_p++) {
                if (valid_p == 0) {
                    break;
                }
                int num_n_blocks_total_p = block_end_p - block_begin_p;
                int cta_n_blocks_p = num_n_blocks_total_p + num_n_blocks_total_p % 2;
                int pt_base_p = batch_idx_p * max_pages_per_seq;
                int max_pg_p = (seqlen_kv_p + PAGE_SIZE - 1) / PAGE_SIZE - 1;
                #pragma unroll 1
                for (int group_base_p = 0; group_base_p < cta_n_blocks_p; group_base_p += 4) {
                    int group_rem_p = cta_n_blocks_p - group_base_p;
                    int group_blocks_p = ((group_rem_p > 4) ? 4 : group_rem_p);
                    int group_block_p = lane / 8;
                    int active_pg_p = ((group_block_p < group_blocks_p) ? 1 : 0);
                    int safe_group_block_p = ((active_pg_p != 0) ? group_block_p : 0);
                    int n_block_p = block_begin_p + cta_n_blocks_p - 1 - group_base_p - safe_group_block_p;
                    int page_idx_p = n_block_p * 8 + lane % 8;
                    if (page_idx_p > max_pg_p) {
                        page_idx_p = max_pg_p;
                    }
                    mbarrier_wait(pg_empty_addr + (pg_slot_p) * 8, pg_phase_p);
                    int pg_stage_addr_p = smem_pg_addr + (unsigned int)(pg_slot_p * 128);
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p;\n\t"
                        "setp.ne.b32 p, %0, 0;\n\t"
                        "@p cp.async.ca.shared::cta.global [%1], [%2], 4;\n\t"
                        "}"
                        :: "r"((active_pg_p) ? 1 : 0), "r"(pg_stage_addr_p + lane * 4), "l"(page_table + (pt_base_p + page_idx_p)));
                    asm volatile(
                        "{\n\t"
                        "cp.async.mbarrier.arrive.shared::cta.b64 [%0];\n\t"
                        "}"
                        :: "r"(pg_full_addr + (pg_slot_p) * 8) : "memory");
                    mbarrier_arrive(pg_full_addr + (pg_slot_p) * 8);
                    pg_slot_p += 1;
                    if (pg_slot_p == 6) { pg_slot_p = 0; pg_phase_p ^= 1; }
                }
                mbarrier_wait(work_full_addr + (work_stage_p) * 8, _phase_work_full_3);
                unsigned int base_0_3 = work_stage_p * 16;
                unsigned int valid_1_3 = work_token_words[base_0_3];
                unsigned int row_tile_2_3 = work_token_words[base_0_3 + 1];
                unsigned int batch_3_3 = work_token_words[base_0_3 + 2];
                unsigned int kv_head_4_3 = work_token_words[base_0_3 + 3];
                unsigned int block_begin_5_3 = work_token_words[base_0_3 + 4];
                unsigned int block_end_6_3 = work_token_words[base_0_3 + 5];
                unsigned int seqlen_row_7_3 = work_token_words[base_0_3 + 6];
                unsigned int n_chunks_8_3 = work_token_words[base_0_3 + 7];
                unsigned int slot_tile_base_9_3 = work_token_words[base_0_3 + 8];
                unsigned int counter_idx_10_3 = work_token_words[base_0_3 + 9];
                unsigned int chunk_11_3 = work_token_words[base_0_3 + 10];
                mbarrier_arrive(work_empty_addr + (work_stage_p) * 8);
                work_stage_p += 1;
                if (work_stage_p == 4) { work_stage_p = 0; _phase_work_full_3 ^= 1; }
                valid_p = valid_1_3;
                batch_idx_p = (int)batch_3_3;
                block_begin_p = (int)block_begin_5_3;
                block_end_p = (int)block_end_6_3;
                seqlen_kv_p = (int)seqlen_row_7_3;
            }
        }
    }
    // ---- Role: scheduler ----
    if (warp == 14) {
        { // scheduler_main
            int lane_0 = lane;
            int num_ctas = gridDim.x;
            int items_per_chunk = q_len * num_kv_heads;
            int num_groups = (batch_size + 32 - 1) / 32;
            #pragma unroll 4
            for (int gs = 0; gs < num_groups; gs++) {
                int bs = gs * 32 + lane_0;
                if (bs < batch_size) {
                    sched_seq_lens[bs] = seq_lens_kv[bs];
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
                    int s1_1 = sched_seq_lens[b1] - (q_len - 1);
                    pairs1 = (unsigned int)((s1_1 + 255) / 256);
                    pairs1_min = pairs1;
                }
                unsigned int _warp_redux_u32_0;
                asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_0) : "r"(pairs1));
                total_pairs += _warp_redux_u32_0;
                unsigned int _warp_redux_u32_1;
                asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_1) : "r"(pairs1));
                unsigned int _max_0 = ((p_max) > (_warp_redux_u32_1) ? (p_max) : (_warp_redux_u32_1));
                p_max = _max_0;
                unsigned int _warp_redux_u32_2;
                asm volatile("redux.sync.min.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_2) : "r"(pairs1_min));
                unsigned int _min_0 = ((p_min) < (_warp_redux_u32_2) ? (p_min) : (_warp_redux_u32_2));
                p_min = _min_0;
            }
            unsigned int total_work = total_pairs * (unsigned int)items_per_chunk;
            unsigned int ideal_pairs = (total_work + (unsigned int)num_ctas - 1) / (unsigned int)num_ctas;
            unsigned int balance_k = (ideal_pairs + 64 - 1) / 64;
            if (balance_k < 1) {
                balance_k = 1;
            }
            if (balance_k > 8) {
                balance_k = 8;
            }
            unsigned int chunk_divisor = balance_k * (unsigned int)num_ctas;
            unsigned int chunk_pairs_u = (total_work + chunk_divisor - 1) / chunk_divisor;
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
                int n_even = num_ctas / whole_items;
                if (n_even > 1) {
                    if (p_max >= 8 * (p_max - p_min)) {
                        unsigned int l_even = (p_max + (unsigned int)n_even - 1) / (unsigned int)n_even;
                        if (l_even < 2) {
                            l_even = 2;
                        }
                        if (l_even < p_max) {
                            chunk_pairs_u = l_even;
                        }
                    }
                }
            }
            unsigned int whole_u = (unsigned int)whole_items;
            unsigned int ctas_u = (unsigned int)num_ctas;
            if (whole_items > num_ctas) {
                if (p_max >= 8 * (p_max - p_min)) {
                    unsigned int k_max_div = 8 * ctas_u;
                    unsigned int l_min = (total_work + k_max_div - 1) / k_max_div;
                    if (l_min < 2) {
                        l_min = 2;
                    }
                    unsigned int cand = p_max;
                    if (lane_0 < 8) {
                        unsigned int k_div = (unsigned int)(lane_0 + 1) * ctas_u;
                        cand = (total_work + k_div - 1) / k_div;
                    }
                    if (lane_0 >= 8) {
                        if (lane_0 < 15) {
                            unsigned int n_div = (unsigned int)(lane_0 - 6);
                            cand = (p_max + n_div - 1) / n_div;
                        }
                    }
                    if (cand < 2) {
                        cand = 2;
                    }
                    if (cand < l_min) {
                        cand = p_max;
                    }
                    unsigned int full_per_tile = p_max / cand;
                    unsigned int remainder = p_max - full_per_tile * cand;
                    unsigned int unit = 4 * cand + 4;
                    unsigned int full_items = full_per_tile * whole_u;
                    unsigned int full_waves = full_items / ctas_u;
                    unsigned int n_last = full_items - full_waves * ctas_u;
                    unsigned int cost = full_waves * unit * ctas_u;
                    unsigned int rem_items = 0;
                    if (remainder > 0) {
                        rem_items = whole_u;
                    }
                    unsigned int rem_unit = 4 * remainder + 4;
                    unsigned int idle = ctas_u - n_last;
                    unsigned int absorbed = idle * (unit / rem_unit);
                    if (absorbed > rem_items) {
                        absorbed = rem_items;
                    }
                    unsigned int _min_1 = ((idle) < (rem_items) ? (idle) : (rem_items));
                    unsigned int active = n_last + _min_1;
                    if (n_last > 0) {
                        int _max_1 = ((96) > (active) ? (96) : (active));
                        cost += unit * (unsigned int)_max_1;
                        rem_items -= absorbed;
                    }
                    unsigned int rounds = (rem_items + ctas_u - 1) / ctas_u;
                    if (rem_items > 0) {
                        unsigned int _min_2 = ((rem_items) < (ctas_u) ? (rem_items) : (ctas_u));
                        int _max_2 = ((96) > (_min_2) ? (96) : (_min_2));
                        cost += rounds * rem_unit * (unsigned int)_max_2;
                    }
                    if (full_per_tile > 1) {
                        cost += 3 * ctas_u;
                    }
                    if (full_per_tile <= 1) {
                        if (remainder > 0) {
                            cost += 3 * ctas_u;
                        }
                    }
                    unsigned int cand_cost = cost;
                    if (cand >= p_max) {
                        cand_cost = 4294967295;
                    }
                    unsigned int full_per_tile_0 = p_max / p_max;
                    unsigned int remainder_1 = p_max - full_per_tile_0 * p_max;
                    unsigned int unit_2 = 4 * p_max + 4;
                    unsigned int full_items_3 = full_per_tile_0 * whole_u;
                    unsigned int full_waves_4 = full_items_3 / ctas_u;
                    unsigned int n_last_5 = full_items_3 - full_waves_4 * ctas_u;
                    unsigned int cost_6 = full_waves_4 * unit_2 * ctas_u;
                    unsigned int rem_items_7 = 0;
                    if (remainder_1 > 0) {
                        rem_items_7 = whole_u;
                    }
                    unsigned int rem_unit_8 = 4 * remainder_1 + 4;
                    unsigned int idle_9 = ctas_u - n_last_5;
                    unsigned int absorbed_10 = idle_9 * (unit_2 / rem_unit_8);
                    if (absorbed_10 > rem_items_7) {
                        absorbed_10 = rem_items_7;
                    }
                    unsigned int _min_3 = ((idle_9) < (rem_items_7) ? (idle_9) : (rem_items_7));
                    unsigned int active_11 = n_last_5 + _min_3;
                    if (n_last_5 > 0) {
                        int _max_3 = ((96) > (active_11) ? (96) : (active_11));
                        cost_6 += unit_2 * (unsigned int)_max_3;
                        rem_items_7 -= absorbed_10;
                    }
                    unsigned int rounds_12 = (rem_items_7 + ctas_u - 1) / ctas_u;
                    if (rem_items_7 > 0) {
                        unsigned int _min_4 = ((rem_items_7) < (ctas_u) ? (rem_items_7) : (ctas_u));
                        int _max_4 = ((96) > (_min_4) ? (96) : (_min_4));
                        cost_6 += rounds_12 * rem_unit_8 * (unsigned int)_max_4;
                    }
                    if (full_per_tile_0 > 1) {
                        cost_6 += 3 * ctas_u;
                    }
                    if (full_per_tile_0 <= 1) {
                        if (remainder_1 > 0) {
                            cost_6 += 3 * ctas_u;
                        }
                    }
                    unsigned int whole_cost = cost_6;
                    unsigned int _warp_redux_u32_3;
                    asm volatile("redux.sync.min.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_3) : "r"(cand_cost));
                    unsigned int min_cost = _warp_redux_u32_3;
                    unsigned int pick = 0;
                    if (cand_cost == min_cost) {
                        pick = cand;
                    }
                    unsigned int _warp_redux_u32_4;
                    asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_4) : "r"(pick));
                    unsigned int best_l = _warp_redux_u32_4;
                    chunk_pairs_u = p_max;
                    if (min_cost < whole_cost) {
                        if (whole_cost - min_cost >= whole_cost / 50) {
                            chunk_pairs_u = best_l;
                        }
                    }
                }
            }
            int chunk_pairs = (int)chunk_pairs_u;
            float _rcp_0 = approx_rcp((float)chunk_pairs_u);
            float chunk_rcp = _rcp_0;
            unsigned int n_full_chunks = 0;
            unsigned int n_chunks_total = 0;
            unsigned int rem_bucket_total[4];
            #pragma unroll
            for (int bi = 0; bi < 4; bi++) {
                rem_bucket_total[bi] = 0;
            }
            #pragma unroll 1
            for (int g2 = 0; g2 < num_groups; g2++) {
                int b2 = g2 * 32 + lane_0;
                unsigned int full2 = 0;
                unsigned int n2 = 0;
                unsigned int pack2 = 0;
                if (b2 < batch_size) {
                    int s2 = sched_seq_lens[b2] - (q_len - 1);
                    unsigned int pairs2 = (unsigned int)((s2 + 255) / 256);
                    unsigned int q = (unsigned int)((float)pairs2 * chunk_rcp);
                    if (q > 0) {
                        if (pairs2 < q * chunk_pairs_u) {
                            q = q - 1;
                        }
                    }
                    if (pairs2 >= (q + 1) * chunk_pairs_u) {
                        q = q + 1;
                    }
                    if (pairs2 > q * chunk_pairs_u) {
                        q = q + 1;
                    }
                    n2 = q;
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
                }
                unsigned int _warp_redux_u32_5;
                asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_5) : "r"(full2));
                n_full_chunks += _warp_redux_u32_5;
                unsigned int _warp_redux_u32_6;
                asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_6) : "r"(n2));
                n_chunks_total += _warp_redux_u32_6;
                unsigned int _warp_redux_u32_7;
                asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_7) : "r"(pack2));
                unsigned int pack_group = _warp_redux_u32_7;
                #pragma unroll
                for (int bi_1 = 1; bi_1 < 4; bi_1++) {
                    rem_bucket_total[bi_1] = rem_bucket_total[bi_1] + (pack_group >> (unsigned int)(8 * (bi_1 - 1)) & 255);
                }
            }
            unsigned int bucket_end[4];
            bucket_end[0] = n_full_chunks * (unsigned int)items_per_chunk;
            #pragma unroll
            for (int bi_2 = 1; bi_2 < 4; bi_2++) {
                bucket_end[bi_2] = bucket_end[bi_2 - 1] + rem_bucket_total[bi_2] * (unsigned int)items_per_chunk;
            }
            unsigned int total_items = n_chunks_total * (unsigned int)items_per_chunk;
            if (blockIdx.x == 0) {
                if (lane_0 == 0) {
                    int plan_off = num_ctas * 256;
                    *(reinterpret_cast<float*>(partial_stats + plan_off) + (0)) = (float)chunk_pairs_u;
                    *(reinterpret_cast<float*>(partial_stats + (plan_off + 1)) + (0)) = (float)total_items;
                }
            }
            unsigned int work_stage_sched = 0;
            int cur_group_b[4];
            unsigned int before_b[4];
            unsigned int si_before_b[4];
            unsigned int st_before_b[4];
            #pragma unroll
            for (int bi_3 = 0; bi_3 < 4; bi_3++) {
                cur_group_b[bi_3] = 0;
                before_b[bi_3] = 0;
                si_before_b[bi_3] = 0;
                st_before_b[bi_3] = 0;
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
                unsigned int _shfl_0 = __shfl_sync(0xFFFFFFFF, ticket_lane0, 0);
                unsigned int ticket = _shfl_0;
                unsigned int token_base = work_stage_sched * 16;
                unsigned int valid_tok = ((ticket < total_items) ? 1 : 0);
                if (valid_tok != 0) {
                    int bucket = 0;
                    unsigned int bucket_start = 0;
                    #pragma unroll
                    for (int bi_4 = 0; bi_4 < 3; bi_4++) {
                        if (bucket_end[bi_4] <= ticket) {
                            bucket = bi_4 + 1;
                            bucket_start = bucket_end[bi_4];
                        }
                    }
                    unsigned int local_items = ticket - bucket_start;
                    unsigned int local_chunk = local_items / (unsigned int)items_per_chunk;
                    int in_chunk = (int)(local_items - local_chunk * (unsigned int)items_per_chunk);
                    int cursor_group = 0;
                    unsigned int before = 0;
                    unsigned int si_before = 0;
                    unsigned int st_before = 0;
                    #pragma unroll
                    for (int bi_5 = 0; bi_5 < 4; bi_5++) {
                        if (bucket == bi_5) {
                            cursor_group = cur_group_b[bi_5];
                            before = before_b[bi_5];
                            si_before = si_before_b[bi_5];
                            st_before = st_before_b[bi_5];
                        }
                    }
                    int s3 = 0;
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
                        pairs3 = 0;
                        n3 = 0;
                        fullc3 = 0;
                        mine3 = 0;
                        split_items3 = 0;
                        split_tiles3 = 0;
                        if (b3 < batch_size) {
                            s3 = sched_seq_lens[b3];
                            pairs3 = (s3 - (q_len - 1) + 255) / 256;
                            unsigned int q_1 = (unsigned int)((float)(unsigned int)pairs3 * chunk_rcp);
                            if (q_1 > 0) {
                                if (q_1 * chunk_pairs_u > (unsigned int)pairs3) {
                                    q_1 = q_1 - 1;
                                }
                            }
                            if ((q_1 + 1) * chunk_pairs_u <= (unsigned int)pairs3) {
                                q_1 = q_1 + 1;
                            }
                            if (q_1 * chunk_pairs_u < (unsigned int)pairs3) {
                                q_1 = q_1 + 1;
                            }
                            n3 = (int)q_1;
                            fullc3 = n3;
                            int rem3 = 0;
                            if (pairs3 < n3 * chunk_pairs) {
                                fullc3 = n3 - 1;
                                rem3 = pairs3 - fullc3 * chunk_pairs;
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
                            if (n3 > 1) {
                                split_items3 = (unsigned int)n3;
                                split_tiles3 = 1;
                            }
                        }
                        uint32_t _warp_scan_sum_u32_0 = mine3;
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_0) : "r"(1));
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_0) : "r"(2));
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_0) : "r"(4));
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_0) : "r"(8));
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_0) : "r"(16));
                        incl3 = _warp_scan_sum_u32_0;
                        uint32_t _warp_scan_sum_u32_1 = split_items3;
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_1) : "r"(1));
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_1) : "r"(2));
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_1) : "r"(4));
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_1) : "r"(8));
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_1) : "r"(16));
                        incl_si3 = _warp_scan_sum_u32_1;
                        uint32_t _warp_scan_sum_u32_2 = split_tiles3;
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_2) : "r"(1));
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_2) : "r"(2));
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_2) : "r"(4));
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_2) : "r"(8));
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_2) : "r"(16));
                        incl_st3 = _warp_scan_sum_u32_2;
                        unsigned int _shfl_1 = __shfl_sync(0xFFFFFFFF, incl3, 31);
                        unsigned int group_total = _shfl_1;
                        if (local_chunk < before + group_total) {
                            break;
                        }
                        before = before + group_total;
                        unsigned int _shfl_2 = __shfl_sync(0xFFFFFFFF, incl_si3, 31);
                        si_before = si_before + _shfl_2;
                        unsigned int _shfl_3 = __shfl_sync(0xFFFFFFFF, incl_st3, 31);
                        st_before = st_before + _shfl_3;
                        cursor_group = cursor_group + 1;
                    }
                    #pragma unroll
                    for (int bi_6 = 0; bi_6 < 4; bi_6++) {
                        if (bucket == bi_6) {
                            cur_group_b[bi_6] = cursor_group;
                            before_b[bi_6] = before;
                            si_before_b[bi_6] = si_before;
                            st_before_b[bi_6] = st_before;
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
                    unsigned int _vote_0 = __ballot_sync(0xFFFFFFFF, hit3 != 0);
                    unsigned int hit_mask = _vote_0;
                    int _ffs_0 = __ffs(hit_mask);
                    int hit_lane = _ffs_0 - 1;
                    int _shfl_4 = __shfl_sync(0xFFFFFFFF, b3, hit_lane);
                    int sel_batch = _shfl_4;
                    int _shfl_5 = __shfl_sync(0xFFFFFFFF, s3, hit_lane);
                    int sel_seqlen = _shfl_5;
                    int _shfl_6 = __shfl_sync(0xFFFFFFFF, n3, hit_lane);
                    int sel_n = _shfl_6;
                    int _shfl_7 = __shfl_sync(0xFFFFFFFF, fullc3, hit_lane);
                    int sel_full = _shfl_7;
                    unsigned int _shfl_8 = __shfl_sync(0xFFFFFFFF, excl3, hit_lane);
                    int sel_excl = (int)_shfl_8;
                    unsigned int _shfl_9 = __shfl_sync(0xFFFFFFFF, excl_si3, hit_lane);
                    int sel_excl_si = (int)_shfl_9;
                    unsigned int _shfl_10 = __shfl_sync(0xFFFFFFFF, excl_st3, hit_lane);
                    int sel_excl_st = (int)_shfl_10;
                    int sel_chunk_off = (int)in_group - sel_excl;
                    int chunk_idx = sel_full;
                    if (bucket == 0) {
                        chunk_idx = sel_chunk_off;
                    }
                    int q_row = in_chunk / num_kv_heads;
                    int kv_head_sel = in_chunk - q_row * num_kv_heads;
                    int seqlen_row_4 = sel_seqlen - (q_len - 1 - q_row);
                    int n_blocks_row = (seqlen_row_4 + BLOCK_N - 1) / BLOCK_N;
                    int block_begin_4 = 2 * chunk_idx * chunk_pairs;
                    int block_end_4 = 2 * (chunk_idx + 1) * chunk_pairs;
                    if (chunk_idx + 1 == sel_n) {
                        block_end_4 = n_blocks_row;
                    }
                    int slot_tile_base_4 = 0;
                    int counter_idx_4 = 0;
                    if (sel_n > 1) {
                        slot_tile_base_4 = ((int)si_before + sel_excl_si) * items_per_chunk + q_row * num_kv_heads + kv_head_sel;
                        counter_idx_4 = ((int)st_before + sel_excl_st) * items_per_chunk + q_row * num_kv_heads + kv_head_sel;
                    }
                    if (lane_0 == 0) {
                        work_token_words[token_base + 1] = (unsigned int)(sel_batch * q_len + q_row);
                        work_token_words[token_base + 2] = (unsigned int)sel_batch;
                        work_token_words[token_base + 3] = (unsigned int)kv_head_sel;
                        work_token_words[token_base + 4] = (unsigned int)block_begin_4;
                        work_token_words[token_base + 5] = (unsigned int)block_end_4;
                        work_token_words[token_base + 6] = (unsigned int)seqlen_row_4;
                        work_token_words[token_base + 7] = (unsigned int)sel_n;
                        work_token_words[token_base + 8] = (unsigned int)slot_tile_base_4;
                        work_token_words[token_base + 9] = (unsigned int)counter_idx_4;
                        work_token_words[token_base + 10] = (unsigned int)chunk_idx;
                    }
                }
                if (lane_0 == 0) {
                    work_token_words[token_base] = valid_tok;
                    mbarrier_arrive(work_full_addr + (work_stage_sched) * 8);
                }
                work_stage_sched += 1;
                if (work_stage_sched == 4) { work_stage_sched = 0; _phase_work_empty ^= 1; }
                if (valid_tok == 0) {
                    break;
                }
            }
            if (lane_0 == 0) {
                uint32_t _atomic_inc_old_0;
                asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                    : "=r"(_atomic_inc_old_0) : "l"(&queue_counters[1]), "r"(static_cast<uint32_t>(num_ctas - 1)) : "memory");
                unsigned int done_old = _atomic_inc_old_0;
                if ((int)done_old == num_ctas - 1) {
                    *(reinterpret_cast<unsigned int*>(queue_counters) + (0)) = 0;
                }
            }
        }
    }
    // ---- Role: load_warp ----
    if (warp == 15) {
        { // load_warp_main
            int q_slot_l = 0;
            int q_phase_l = 1;
            int pg_slot_l = 0;
            int pg_phase_l = 0;
            unsigned int work_stage_l = 0;
            int kv_base_l = 0;
            int prefetched_l = 0;
            int pg_item_base_l = 0;
            int pg_next_base_l = 0;
            unsigned int _phase_work_full_4 = 0;
            mbarrier_wait(work_full_addr + (work_stage_l) * 8, _phase_work_full_4);
            unsigned int base_4 = work_stage_l * 16;
            unsigned int valid_5 = work_token_words[base_4];
            unsigned int row_tile_5 = work_token_words[base_4 + 1];
            unsigned int batch_5 = work_token_words[base_4 + 2];
            unsigned int kv_head_5 = work_token_words[base_4 + 3];
            unsigned int block_begin_6 = work_token_words[base_4 + 4];
            unsigned int block_end_5 = work_token_words[base_4 + 5];
            unsigned int seqlen_row_5 = work_token_words[base_4 + 6];
            unsigned int n_chunks_4 = work_token_words[base_4 + 7];
            unsigned int slot_tile_base_5 = work_token_words[base_4 + 8];
            unsigned int counter_idx_5 = work_token_words[base_4 + 9];
            unsigned int chunk_4 = work_token_words[base_4 + 10];
            mbarrier_arrive(work_empty_addr + (work_stage_l) * 8);
            work_stage_l += 1;
            if (work_stage_l == 4) { work_stage_l = 0; _phase_work_full_4 ^= 1; }
            unsigned int valid_l = valid_5;
            int row_tile_l = (int)row_tile_5;
            int batch_idx_l = (int)batch_5;
            int kv_head_idx_1 = (int)kv_head_5;
            int block_begin_l = (int)block_begin_6;
            int block_end_l = (int)block_end_5;
            unsigned int valid_n = 0;
            int row_tile_n = 0;
            int batch_idx_n = 0;
            int kv_head_n = 0;
            int block_begin_n = 0;
            int block_end_n = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_l = 0; _tile_iter_l < max_items; _tile_iter_l++) {
                if (valid_l == 0) {
                    break;
                }
                int num_n_blocks_total_3 = block_end_l - block_begin_l;
                int cta_n_blocks_3 = num_n_blocks_total_3 + num_n_blocks_total_3 % 2;
                int gate_block = cta_n_blocks_3 - 1 - 8;
                if (gate_block < 0) {
                    gate_block = 0;
                }
                int head_blocks = ((cta_n_blocks_3 < 4) ? cta_n_blocks_3 : 4);
                int tail_start = cta_n_blocks_3 - 4;
                if (tail_start < 0) {
                    tail_start = 0;
                }
                if (prefetched_l == 0) {
                    pg_item_base_l = pg_slot_l;
                    mbarrier_wait(q_empty_addr + (q_slot_l) * 8, q_phase_l);
                    int q_word_base_s = (row_tile_l * num_kv_heads + kv_head_idx_1) * (TILE_Q * HEAD_DIM / 2);
                    unsigned int q_words_s[8];
                    float q_f32_s[16];
                    float q_res_s[16];
                    unsigned int q_packed_s[4];
                    unsigned int q_packed_lo_s[4];
                    #pragma unroll
                    for (int qc_s = 0; qc_s < 2; qc_s++) {
                        int q_chunk_s = lane * 2 + qc_s;
                        int q_row_s = q_chunk_s / 8;
                        int q_col16_s = q_chunk_s % 8;
                        {
                            uint4 _uv4_0 = *reinterpret_cast<const uint4*>(Qt + q_word_base_s + q_row_s * 64 + q_col16_s * 8);
                            q_words_s[0 + 0] = _uv4_0.x;
                            q_words_s[0 + 1] = _uv4_0.y;
                            q_words_s[0 + 2] = _uv4_0.z;
                            q_words_s[0 + 3] = _uv4_0.w;
                        }
                        {
                            uint4 _uv4_1 = *reinterpret_cast<const uint4*>(Qt + q_word_base_s + q_row_s * 64 + q_col16_s * 8 + 4);
                            q_words_s[4 + 0] = _uv4_1.x;
                            q_words_s[4 + 1] = _uv4_1.y;
                            q_words_s[4 + 2] = _uv4_1.z;
                            q_words_s[4 + 3] = _uv4_1.w;
                        }
                        #pragma unroll
                        for (int qw_s = 0; qw_s < 8; qw_s++) {
                            unsigned int q_w_s = q_words_s[qw_s];
                            unsigned int q_lo_bits_d = q_w_s & 65535;
                            unsigned int q_hi_bits_d = q_w_s >> 16 & 65535;
                            unsigned int q_lo_tb_d = q_lo_bits_d & 65408;
                            unsigned int q_hi_tb_d = q_hi_bits_d & 65408;
                            float _cvt_f32_f16_0;
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_0) : "h"((uint16_t)(q_lo_bits_d)));
                            float q_lo_d = _cvt_f32_f16_0;
                            float _cvt_f32_f16_1;
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_1) : "h"((uint16_t)(q_lo_tb_d)));
                            float q_lo_t_d = _cvt_f32_f16_1;
                            float _cvt_f32_f16_2;
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_2) : "h"((uint16_t)(q_hi_bits_d)));
                            float q_hi_d = _cvt_f32_f16_2;
                            float _cvt_f32_f16_3;
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_3) : "h"((uint16_t)(q_hi_tb_d)));
                            float q_hi_t_d = _cvt_f32_f16_3;
                            q_f32_s[2 * qw_s] = q_lo_t_d;
                            q_res_s[2 * qw_s] = q_lo_d - q_lo_t_d;
                            q_f32_s[2 * qw_s + 1] = q_hi_t_d;
                            q_res_s[2 * qw_s + 1] = q_hi_d - q_hi_t_d;
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
                        int q_hi_base_s = smem_qt_addr + (unsigned int)(q_slot_l * 1024);
                        int q_lo_base_s = smem_qt_lo_addr + (unsigned int)(q_slot_l * 1024);
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
                    mbarrier_arrive(q_full_addr + (q_slot_l) * 8);
                }
                if (elect_sync()) {
                    #pragma unroll 1
                    for (int ni = prefetched_l; ni < head_blocks; ni++) {
                        int kv_stage = (kv_base_l + ni) % NUM_KV_STAGES;
                        mbarrier_wait(kv_empty_addr + (kv_stage) * 8, 1);
                        if (ni % 4 == 0) {
                            mbarrier_wait(pg_full_addr + (pg_slot_l) * 8, pg_phase_l);
                            pg_slot_l += 1;
                            if (pg_slot_l == 6) { pg_slot_l = 0; pg_phase_l ^= 1; }
                        }
                        mbarrier_arrive_expect_tx(kv_full_addr + (kv_stage) * 8, 16384);
                        int ldst = smem_kv_addr + (unsigned int)(kv_stage * 16384);
                        int pg_k[8];
                        int pg_k_stage = (pg_item_base_l + ni / 4) % 6;
                        int pg_base_addr = smem_pg_addr + (unsigned int)(pg_k_stage * 128) + (unsigned int)(ni % 4 * 32);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&pg_k[0])), "=r"(*reinterpret_cast<uint32_t*>(&pg_k[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pg_k[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pg_k[(0) + 3]))
                            : "r"(pg_base_addr));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&pg_k[4])), "=r"(*reinterpret_cast<uint32_t*>(&pg_k[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pg_k[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pg_k[(4) + 3]))
                            : "r"(pg_base_addr + 16));
                        #pragma unroll
                        for (int pg_i = 0; pg_i < 8; pg_i++) {
                            int toff = pg_i * 2048;
                            tma_5d_gmem2smem(ldst + toff, (&K), 0, 0, 0, kv_head_idx_1, pg_k[pg_i], kv_full_addr + (kv_stage) * 8);
                        }
                    }
                }
                if (elect_sync()) {
                    #pragma unroll 1
                    for (int ni_1 = 0; ni_1 < tail_start; ni_1++) {
                        int stage = (kv_base_l + ni_1) % NUM_KV_STAGES;
                        int pg_v_stage = (pg_item_base_l + ni_1 / 4) % 6;
                        int next_ni = ni_1 + 4;
                        int k_stage = (kv_base_l + next_ni) % NUM_KV_STAGES;
                        mbarrier_wait(kv_empty_addr + (k_stage) * 8, 1);
                        if (next_ni % 4 == 0) {
                            mbarrier_wait(pg_full_addr + (pg_slot_l) * 8, pg_phase_l);
                            pg_slot_l += 1;
                            if (pg_slot_l == 6) { pg_slot_l = 0; pg_phase_l ^= 1; }
                        }
                        mbarrier_arrive_expect_tx(kv_full_addr + (k_stage) * 8, 16384);
                        int kdst = smem_kv_addr + (unsigned int)(k_stage * 16384);
                        int pg_nk[8];
                        int pg_nk_stage = (pg_item_base_l + next_ni / 4) % 6;
                        int pg_base_addr_1 = smem_pg_addr + (unsigned int)(pg_nk_stage * 128) + (unsigned int)(next_ni % 4 * 32);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&pg_nk[0])), "=r"(*reinterpret_cast<uint32_t*>(&pg_nk[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pg_nk[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pg_nk[(0) + 3]))
                            : "r"(pg_base_addr_1));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&pg_nk[4])), "=r"(*reinterpret_cast<uint32_t*>(&pg_nk[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pg_nk[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pg_nk[(4) + 3]))
                            : "r"(pg_base_addr_1 + 16));
                        #pragma unroll
                        for (int pg_i_1 = 0; pg_i_1 < 8; pg_i_1++) {
                            int ntoff = pg_i_1 * 2048;
                            tma_5d_gmem2smem(kdst + ntoff, (&K), 0, 0, 0, kv_head_idx_1, pg_nk[pg_i_1], kv_full_addr + (k_stage) * 8);
                        }
                        int pg_v[8];
                        int pg_base_addr_0 = smem_pg_addr + (unsigned int)(pg_v_stage * 128) + (unsigned int)(ni_1 % 4 * 32);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&pg_v[0])), "=r"(*reinterpret_cast<uint32_t*>(&pg_v[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pg_v[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pg_v[(0) + 3]))
                            : "r"(pg_base_addr_0));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&pg_v[4])), "=r"(*reinterpret_cast<uint32_t*>(&pg_v[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pg_v[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pg_v[(4) + 3]))
                            : "r"(pg_base_addr_0 + 16));
                        if (ni_1 % 4 == 3) {
                            mbarrier_arrive(pg_empty_addr + (pg_v_stage) * 8);
                        }
                        mbarrier_wait(kv_empty_addr + (stage) * 8, 0);
                        mbarrier_arrive_expect_tx(kv_full_addr + (stage) * 8, 16384);
                        int vdst = smem_kv_addr + (unsigned int)(stage * 16384);
                        #pragma unroll
                        for (int pg_i_2 = 0; pg_i_2 < 8; pg_i_2++) {
                            int vtoff = pg_i_2 * 2048;
                            tma_5d_gmem2smem(vdst + vtoff, (&V), 0, 0, 0, kv_head_idx_1, pg_v[pg_i_2], kv_full_addr + (stage) * 8);
                        }
                        if (ni_1 == gate_block) {
                            mbarrier_arrive(claim_gate_addr);
                        }
                    }
                    if (gate_block >= tail_start) {
                        mbarrier_arrive(claim_gate_addr);
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_l) * 8, _phase_work_full_4);
                unsigned int base_0_4 = work_stage_l * 16;
                unsigned int valid_1_4 = work_token_words[base_0_4];
                unsigned int row_tile_2_4 = work_token_words[base_0_4 + 1];
                unsigned int batch_3_4 = work_token_words[base_0_4 + 2];
                unsigned int kv_head_4_4 = work_token_words[base_0_4 + 3];
                unsigned int block_begin_5_4 = work_token_words[base_0_4 + 4];
                unsigned int block_end_6_4 = work_token_words[base_0_4 + 5];
                unsigned int seqlen_row_7_4 = work_token_words[base_0_4 + 6];
                unsigned int n_chunks_8_4 = work_token_words[base_0_4 + 7];
                unsigned int slot_tile_base_9_4 = work_token_words[base_0_4 + 8];
                unsigned int counter_idx_10_4 = work_token_words[base_0_4 + 9];
                unsigned int chunk_11_4 = work_token_words[base_0_4 + 10];
                mbarrier_arrive(work_empty_addr + (work_stage_l) * 8);
                work_stage_l += 1;
                if (work_stage_l == 4) { work_stage_l = 0; _phase_work_full_4 ^= 1; }
                valid_n = valid_1_4;
                row_tile_n = (int)row_tile_2_4;
                batch_idx_n = (int)batch_3_4;
                kv_head_n = (int)kv_head_4_4;
                block_begin_n = (int)block_begin_5_4;
                block_end_n = (int)block_end_6_4;
                int n_blocks_n = block_end_n - block_begin_n;
                int cta_n_blocks_n = n_blocks_n + n_blocks_n % 2;
                int prefetch_n = cta_n_blocks_3 - tail_start;
                if (prefetch_n > cta_n_blocks_n) {
                    prefetch_n = cta_n_blocks_n;
                }
                if (valid_n == 0) {
                    prefetch_n = 0;
                }
                int q_slot_n = q_slot_l;
                int q_phase_n = q_phase_l;
                q_slot_n += 1;
                if (q_slot_n == 2) { q_slot_n = 0; q_phase_n ^= 1; }
                if (valid_n != 0) {
                    mbarrier_wait(q_empty_addr + (q_slot_n) * 8, q_phase_n);
                    int q_word_base_s_1 = (row_tile_n * num_kv_heads + kv_head_n) * (TILE_Q * HEAD_DIM / 2);
                    unsigned int q_words_s_1[8];
                    float q_f32_s_1[16];
                    float q_res_s_1[16];
                    unsigned int q_packed_s_1[4];
                    unsigned int q_packed_lo_s_1[4];
                    #pragma unroll
                    for (int qc_s_1 = 0; qc_s_1 < 2; qc_s_1++) {
                        int q_chunk_s_1 = lane * 2 + qc_s_1;
                        int q_row_s_1 = q_chunk_s_1 / 8;
                        int q_col16_s_1 = q_chunk_s_1 % 8;
                        {
                            uint4 _uv4_2 = *reinterpret_cast<const uint4*>(Qt + q_word_base_s_1 + q_row_s_1 * 64 + q_col16_s_1 * 8);
                            q_words_s_1[0 + 0] = _uv4_2.x;
                            q_words_s_1[0 + 1] = _uv4_2.y;
                            q_words_s_1[0 + 2] = _uv4_2.z;
                            q_words_s_1[0 + 3] = _uv4_2.w;
                        }
                        {
                            uint4 _uv4_3 = *reinterpret_cast<const uint4*>(Qt + q_word_base_s_1 + q_row_s_1 * 64 + q_col16_s_1 * 8 + 4);
                            q_words_s_1[4 + 0] = _uv4_3.x;
                            q_words_s_1[4 + 1] = _uv4_3.y;
                            q_words_s_1[4 + 2] = _uv4_3.z;
                            q_words_s_1[4 + 3] = _uv4_3.w;
                        }
                        #pragma unroll
                        for (int qw_s_1 = 0; qw_s_1 < 8; qw_s_1++) {
                            unsigned int q_w_s_1 = q_words_s_1[qw_s_1];
                            unsigned int q_lo_bits_d_1 = q_w_s_1 & 65535;
                            unsigned int q_hi_bits_d_1 = q_w_s_1 >> 16 & 65535;
                            unsigned int q_lo_tb_d_1 = q_lo_bits_d_1 & 65408;
                            unsigned int q_hi_tb_d_1 = q_hi_bits_d_1 & 65408;
                            float _cvt_f32_f16_4;
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_4) : "h"((uint16_t)(q_lo_bits_d_1)));
                            float q_lo_d_1 = _cvt_f32_f16_4;
                            float _cvt_f32_f16_5;
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_5) : "h"((uint16_t)(q_lo_tb_d_1)));
                            float q_lo_t_d_1 = _cvt_f32_f16_5;
                            float _cvt_f32_f16_6;
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_6) : "h"((uint16_t)(q_hi_bits_d_1)));
                            float q_hi_d_1 = _cvt_f32_f16_6;
                            float _cvt_f32_f16_7;
                            asm("cvt.f32.f16 %0, %1;" : "=f"(_cvt_f32_f16_7) : "h"((uint16_t)(q_hi_tb_d_1)));
                            float q_hi_t_d_1 = _cvt_f32_f16_7;
                            q_f32_s_1[2 * qw_s_1] = q_lo_t_d_1;
                            q_res_s_1[2 * qw_s_1] = q_lo_d_1 - q_lo_t_d_1;
                            q_f32_s_1[2 * qw_s_1 + 1] = q_hi_t_d_1;
                            q_res_s_1[2 * qw_s_1 + 1] = q_hi_d_1 - q_hi_t_d_1;
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
                        int q_hi_base_s_1 = smem_qt_addr + (unsigned int)(q_slot_n * 1024);
                        int q_lo_base_s_1 = smem_qt_lo_addr + (unsigned int)(q_slot_n * 1024);
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
                    mbarrier_arrive(q_full_addr + (q_slot_n) * 8);
                }
                if (elect_sync()) {
                    #pragma unroll 1
                    for (int ni_2 = tail_start; ni_2 < cta_n_blocks_3; ni_2++) {
                        int stage_1 = (kv_base_l + ni_2) % NUM_KV_STAGES;
                        int pg_v_stage_1 = (pg_item_base_l + ni_2 / 4) % 6;
                        if (prefetch_n > ni_2 - tail_start) {
                            int pn = ni_2 - tail_start;
                            int k_stage_1 = (kv_base_l + cta_n_blocks_3 + pn) % NUM_KV_STAGES;
                            if (pn == 0) {
                                pg_next_base_l = pg_slot_l;
                            }
                            if (pn % 4 == 0) {
                                mbarrier_wait(pg_full_addr + (pg_slot_l) * 8, pg_phase_l);
                                pg_slot_l += 1;
                                if (pg_slot_l == 6) { pg_slot_l = 0; pg_phase_l ^= 1; }
                            }
                            mbarrier_wait(kv_empty_addr + (k_stage_1) * 8, 1);
                            mbarrier_arrive_expect_tx(kv_full_addr + (k_stage_1) * 8, 16384);
                            int kdst_1 = smem_kv_addr + (unsigned int)(k_stage_1 * 16384);
                            int pg_nk_1[8];
                            int pg_nk_stage_1 = (pg_next_base_l + pn / 4) % 6;
                            int pg_base_addr_2 = smem_pg_addr + (unsigned int)(pg_nk_stage_1 * 128) + (unsigned int)(pn % 4 * 32);
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pg_nk_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&pg_nk_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pg_nk_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pg_nk_1[(0) + 3]))
                                : "r"(pg_base_addr_2));
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&pg_nk_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&pg_nk_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pg_nk_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pg_nk_1[(4) + 3]))
                                : "r"(pg_base_addr_2 + 16));
                            #pragma unroll
                            for (int pg_i_3 = 0; pg_i_3 < 8; pg_i_3++) {
                                int ntoff_1 = pg_i_3 * 2048;
                                tma_5d_gmem2smem(kdst_1 + ntoff_1, (&K), 0, 0, 0, kv_head_n, pg_nk_1[pg_i_3], kv_full_addr + (k_stage_1) * 8);
                            }
                        }
                        int pg_v_1[8];
                        int pg_base_addr_3 = smem_pg_addr + (unsigned int)(pg_v_stage_1 * 128) + (unsigned int)(ni_2 % 4 * 32);
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&pg_v_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&pg_v_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pg_v_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pg_v_1[(0) + 3]))
                            : "r"(pg_base_addr_3));
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&pg_v_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&pg_v_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pg_v_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pg_v_1[(4) + 3]))
                            : "r"(pg_base_addr_3 + 16));
                        int pg_group_done = ((ni_2 % 4 == 3) ? 1 : 0);
                        pg_group_done = pg_group_done | ((ni_2 + 1 == cta_n_blocks_3) ? 1 : 0);
                        if (pg_group_done != 0) {
                            mbarrier_arrive(pg_empty_addr + (pg_v_stage_1) * 8);
                        }
                        mbarrier_wait(kv_empty_addr + (stage_1) * 8, 0);
                        mbarrier_arrive_expect_tx(kv_full_addr + (stage_1) * 8, 16384);
                        int vdst_1 = smem_kv_addr + (unsigned int)(stage_1 * 16384);
                        #pragma unroll
                        for (int pg_i_4 = 0; pg_i_4 < 8; pg_i_4++) {
                            int vtoff_1 = pg_i_4 * 2048;
                            tma_5d_gmem2smem(vdst_1 + vtoff_1, (&V), 0, 0, 0, kv_head_idx_1, pg_v_1[pg_i_4], kv_full_addr + (stage_1) * 8);
                        }
                    }
                }
                kv_base_l = (kv_base_l + cta_n_blocks_3) % NUM_KV_STAGES;
                q_slot_l += 1;
                if (q_slot_l == 2) { q_slot_l = 0; q_phase_l ^= 1; }
                prefetched_l = prefetch_n;
                pg_item_base_l = pg_next_base_l;
                valid_l = valid_n;
                row_tile_l = row_tile_n;
                batch_idx_l = batch_idx_n;
                kv_head_idx_1 = kv_head_n;
                block_begin_l = block_begin_n;
                block_end_l = block_end_n;
            }
        }
    }

    // Cleanup
}

} // extern "C"
