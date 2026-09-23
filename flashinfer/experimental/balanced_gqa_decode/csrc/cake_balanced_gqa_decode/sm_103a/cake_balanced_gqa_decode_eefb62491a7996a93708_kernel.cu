/*
 * Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
struct __align__(128) CakeTensorMap { uint64_t opaque[16]; };
struct __align__(64) CakeTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(CakeTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(CakeTensorMap64) == 64, "64-aligned tensor-map ABI alignment");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(CakeTensorMap) >= alignof(CUtensorMap), "CakeTensorMap alignment must cover the CUtensorMap CUDA ABI");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 256
#define TMEM_TMEM_S_OFFSET 0
#define TMEM_TMEM_O_OFFSET 128
#define NUM_Q_PIPE_STAGES 2
#define NUM_KV_PIPE_STAGES 4
#define NUM_SM_PIPE_STAGES 2
#define NUM_PAGE_PIPE_STAGES 6
#define NUM_WORK_PIPE_STAGES 4
#define SMEM_SMEM_SCALE_OFF 1024
#define SMEM_SMEM_SCALE_STAGE_BYTES 512
#define SMEM_SMEM_SCALE_STRIDE 512
#define SMEM_SMEM_SUM_OFF 1536
#define SMEM_SMEM_SUM_STAGE_BYTES 256
#define SMEM_SMEM_SUM_STRIDE 256
#define SMEM_SMEM_MAX_OFF 1792
#define SMEM_SMEM_MAX_STAGE_BYTES 256
#define SMEM_SMEM_MAX_STRIDE 256
#define SMEM_SMEM_PAGE_OFFSETS_OFF 2048
#define SMEM_SMEM_PAGE_OFFSETS_STAGE_BYTES 192
#define SMEM_SMEM_PAGE_OFFSETS_STRIDE 192
#define SMEM_WORK_TOKEN_WORDS_OFF 2304
#define SMEM_WORK_TOKEN_WORDS_STAGE_BYTES 256
#define SMEM_WORK_TOKEN_WORDS_STRIDE 256
#define SMEM_SMEM_MERGE_FLAG_OFF 2560
#define SMEM_SMEM_MERGE_FLAG_STAGE_BYTES 16
#define SMEM_SMEM_MERGE_FLAG_STRIDE 16
#define SMEM_SCHED_SEQ_LENS_OFF 3072
#define SMEM_SCHED_SEQ_LENS_STAGE_BYTES 4096
#define SMEM_SCHED_SEQ_LENS_STRIDE 4096
#define SMEM_SMEM_QT_OFF 9216
#define SMEM_SMEM_QT_STAGE_BYTES 16384
#define SMEM_SMEM_QT_STRIDE 16384
#define SMEM_SMEM_KV_OFF 41984
#define SMEM_SMEM_KV_STAGE_BYTES 32768
#define SMEM_SMEM_KV_STRIDE 32768
#define SMEM_SMEM_V_OFF 41984
#define SMEM_SMEM_V_STAGE_BYTES 32768
#define SMEM_SMEM_V_STRIDE 32768
#define SMEM_SMEM_P_OFF 173056
#define SMEM_SMEM_P_STAGE_BYTES 16384
#define SMEM_SMEM_P_STRIDE 16384
#define SMEM_SMEM_ST_A_OFF 189440
#define SMEM_SMEM_ST_A_STAGE_BYTES 16512
#define SMEM_SMEM_ST_A_STRIDE 16512
#define SMEM_SMEM_ST_B_OFF 205952
#define SMEM_SMEM_ST_B_STAGE_BYTES 16512
#define SMEM_SMEM_ST_B_STRIDE 16512
#define SMEM_TOTAL 222464
#define THREADS 512
#define BLOCK_N 128
#define HEAD_DIM 128
#define PAGE_SIZE 16
#define NUM_KV_STAGES 4

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

__device__ __forceinline__ void mbarrier_init_generic(void* mbar_addr, int count) {
    asm volatile("mbarrier.init.b64 [%0], %1;"
        :: "l"(mbar_addr), "r"(count));
}


__device__ __forceinline__ uint32_t mbarrier_try_wait(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
}

__device__ __forceinline__ uint32_t mbarrier_try_wait_cluster(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
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

// Source-faithful relaxed CTA wait used only by a typed protocol that does
// not attach the PTX acquire qualifier, such as FA4's interior P-ready edge.
__device__ __forceinline__ void mbarrier_wait_relaxed(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_RELAXED:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64"
        " P1, [%0], %1, 10000000;\n\t"
        "@P1 bra.uni DONE_RELAXED;\n\t"
        "bra.uni LAB_WAIT_RELAXED;\n\t"
        "DONE_RELAXED:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase) : "memory");
}

// Exact source ports may request the PTX suspendTimeHint operand explicitly.
// The hint is expressed in nanoseconds and is kept separate from the canonical
// no-hint CTA helper so unrelated schedules retain their existing retry path.
__device__ __forceinline__ void mbarrier_wait_suspend(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_SUSPEND:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_SUSPEND;\n\t"
        "bra.uni LAB_WAIT_SUSPEND;\n\t"
        "DONE_SUSPEND:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_cluster(int mbar_addr, int phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1;\n\t"
        "@P1 bra.uni DONE_CLUSTER;\n\t"
        "bra.uni LAB_WAIT_CLUSTER;\n\t"
        "DONE_CLUSTER:\n\t"
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

// Exact unqualified CTA wait used by source schedules whose PTX intentionally
// omits the acquire qualifier while retaining a typed suspendTimeHint operand.
__device__ __forceinline__ void mbarrier_wait_relaxed_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_RELAXED_HINT:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra DONE_RELAXED_HINT;\n\t"
        "bra LAB_WAIT_RELAXED_HINT;\n\t"
        "DONE_RELAXED_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint));
}

__device__ __forceinline__ void mbarrier_wait_cluster_hint(
        int mbar_addr, int phase, uint32_t suspend_time_hint) {
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT_CLUSTER_HINT:\n\t"
        "mbarrier.try_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE_CLUSTER_HINT;\n\t"
        "bra.uni LAB_WAIT_CLUSTER_HINT;\n\t"
        "DONE_CLUSTER_HINT:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(suspend_time_hint) : "memory");
}

__device__ __forceinline__ void mbarrier_wait_token(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait(mbar_addr, phase);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_suspend(
        int mbar_addr, int phase, uint32_t token, uint32_t suspend_time_hint) {
    if (token == 0) {
        mbarrier_wait_suspend(mbar_addr, phase, suspend_time_hint);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_cluster(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        mbarrier_wait_cluster(mbar_addr, phase);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_hint(
        int mbar_addr, int phase, uint32_t token, uint32_t suspend_time_hint) {
    if (token == 0) {
        mbarrier_wait_hint(mbar_addr, phase, suspend_time_hint);
    }
}

__device__ __forceinline__ void mbarrier_wait_token_cluster_hint(
        int mbar_addr, int phase, uint32_t token, uint32_t suspend_time_hint) {
    if (token == 0) {
        mbarrier_wait_cluster_hint(mbar_addr, phase, suspend_time_hint);
    }
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


__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
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


__device__ __forceinline__ void tmem_ld_x32(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7,"
        "  %8, %9, %10, %11, %12, %13, %14, %15,"
        "  %16, %17, %18, %19, %20, %21, %22, %23,"
        "  %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
        : "=f"(dst[0]),  "=f"(dst[1]),  "=f"(dst[2]),  "=f"(dst[3]),
          "=f"(dst[4]),  "=f"(dst[5]),  "=f"(dst[6]),  "=f"(dst[7]),
          "=f"(dst[8]),  "=f"(dst[9]),  "=f"(dst[10]), "=f"(dst[11]),
          "=f"(dst[12]), "=f"(dst[13]), "=f"(dst[14]), "=f"(dst[15]),
          "=f"(dst[16]), "=f"(dst[17]), "=f"(dst[18]), "=f"(dst[19]),
          "=f"(dst[20]), "=f"(dst[21]), "=f"(dst[22]), "=f"(dst[23]),
          "=f"(dst[24]), "=f"(dst[25]), "=f"(dst[26]), "=f"(dst[27]),
          "=f"(dst[28]), "=f"(dst[29]), "=f"(dst[30]), "=f"(dst[31])
        : "r"(tmem_addr));
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


__device__ __forceinline__ void tcgen05_commit2(int mbar_addr0, int mbar_addr1) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];\n\t"
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%1];\n\t"
        :: "r"(mbar_addr0), "r"(mbar_addr1) : "memory");
}


__device__ __forceinline__ void fence_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}


__device__ __forceinline__ uint64_t make_smem_desc(int addr) {
    const int SBO = 1024;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL)
         | (2ULL << 61ULL);
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


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(512) void
kernel_cake_balanced_gqa_decode_eefb62491a7996a93708(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap V, __nv_bfloat16* __restrict__ O_ptr, int* __restrict__ page_table, int* __restrict__ seq_lens_kv, float* __restrict__ partial_o, float* __restrict__ partial_stats, unsigned int* __restrict__ tile_counters, unsigned int* __restrict__ queue_counters, int max_pages_per_seq, float softmax_scale_log2, int num_q_heads, int num_kv_heads, int batch_size, int q_len, unsigned int max_items)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define q_full_addr (mbar_base + 0)
    #define q_empty_addr (mbar_base + 16)
    #define kv_full_addr (mbar_base + 32)
    #define kv_empty_addr (mbar_base + 64)
    #define s_full_addr (mbar_base + 96)
    #define corr_scale_addr (mbar_base + 112)
    #define p_full_addr (mbar_base + 128)
    #define p_empty_addr (mbar_base + 136)
    #define o_ready_addr (mbar_base + 144)
    #define o_empty_addr (mbar_base + 152)
    #define stats_full_addr (mbar_base + 160)
    #define stats_empty_addr (mbar_base + 168)
    #define tmem_dealloc_addr (mbar_base + 176)
    #define page_offsets_full_addr (mbar_base + 184)
    #define page_offsets_empty_addr (mbar_base + 232)
    #define work_full_addr (mbar_base + 280)
    #define work_empty_addr (mbar_base + 312)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    float* smem_scale = reinterpret_cast<float*>(smem_raw + 1024);
    const int smem_scale_addr = smem + 1024;
    float* smem_sum = reinterpret_cast<float*>(smem_raw + 1536);
    const int smem_sum_addr = smem + 1536;
    float* smem_max = reinterpret_cast<float*>(smem_raw + 1792);
    const int smem_max_addr = smem + 1792;
    int* smem_page_offsets = reinterpret_cast<int*>(smem_raw + 2048);
    const int smem_page_offsets_addr = smem + 2048;
    unsigned int* work_token_words = reinterpret_cast<unsigned int*>(smem_raw + 2304);
    const int work_token_words_addr = smem + 2304;
    unsigned int* smem_merge_flag = reinterpret_cast<unsigned int*>(smem_raw + 2560);
    const int smem_merge_flag_addr = smem + 2560;
    int* sched_seq_lens = reinterpret_cast<int*>(smem_raw + 3072);
    const int sched_seq_lens_addr = smem + 3072;
    __nv_bfloat16* smem_qt = reinterpret_cast<__nv_bfloat16*>(smem_raw + 9216);
    const int smem_qt_addr = smem + 9216;
    __nv_bfloat16* smem_kv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 41984);
    const int smem_kv_addr = smem + 41984;
    __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 41984);
    const int smem_v_addr = smem + 41984;
    __nv_bfloat16* smem_p = reinterpret_cast<__nv_bfloat16*>(smem_raw + 173056);
    const int smem_p_addr = smem + 173056;
    float* smem_st_a = reinterpret_cast<float*>(smem_raw + 189440);
    const int smem_st_a_addr = smem + 189440;
    float* smem_st_b = reinterpret_cast<float*>(smem_raw + 205952);
    const int smem_st_b_addr = smem + 205952;
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Q))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&K))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&V))) : "memory");

    // Mbarrier init (17 pipeline groups, 0 ordered-sequence groups, 43 barriers)
    // Mbarriers at smem_raw[0..344)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'q_pipe' ---
            // q_full: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            // q_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // kv_full: 4 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // kv_empty: 4 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // --- pipeline 'sm_pipe' ---
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            // corr_scale: 2 barriers, init_count=256
            mbarrier_init(smem + 112, 256);
            mbarrier_init(smem + 120, 256);
            // p_full: 1 barriers, init_count=384
            mbarrier_init(smem + 128, 384);
            // p_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 136, 1);
            // o_ready: 1 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            // o_empty: 1 barriers, init_count=128
            mbarrier_init(smem + 152, 128);
            // stats_full: 1 barriers, init_count=256
            mbarrier_init(smem + 160, 256);
            // stats_empty: 1 barriers, init_count=4
            mbarrier_init(smem + 168, 4);
            // tmem_dealloc: 1 barriers, init_count=128
            mbarrier_init(smem + 176, 128);
            // --- pipeline 'page_pipe' ---
            // page_offsets_full: 6 barriers, init_count=1
            mbarrier_init(smem + 184, 1);
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            // page_offsets_empty: 6 barriers, init_count=1
            mbarrier_init(smem + 232, 1);
            mbarrier_init(smem + 240, 1);
            mbarrier_init(smem + 248, 1);
            mbarrier_init(smem + 256, 1);
            mbarrier_init(smem + 264, 1);
            mbarrier_init(smem + 272, 1);
            // --- pipeline 'work_pipe' ---
            // work_full: 4 barriers, init_count=1
            mbarrier_init(smem + 280, 1);
            mbarrier_init(smem + 288, 1);
            mbarrier_init(smem + 296, 1);
            mbarrier_init(smem + 304, 1);
            // work_empty: 4 barriers, init_count=480
            mbarrier_init(smem + 312, 480);
            mbarrier_init(smem + 320, 480);
            mbarrier_init(smem + 328, 480);
            mbarrier_init(smem + 336, 480);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 192 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 344);
    if (warp == 0) {
        int _tmem_hold = smem + 344;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_s = taddr;
    const int tmem_tmem_o = taddr + 128;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
    }

    // ---- Role: softmax ----
    if (warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 184;");
        { // softmax_main
            int wg = warp / 4;
            const int warp_in_wg = warp % 4;
            const int lane_base = warp_in_wg * 32;
            int wg_tid = warp_in_wg * 32 + lane;
            int col_local = wg_tid / 4;
            int quarter = wg_tid % 4;
            int col_base = wg * 32;
            int my_col = col_base + col_local;
            int tok_base = quarter * 8;
            int my_s_base = taddr + (unsigned int)(lane_base << 16) + (unsigned int)col_base;
            float* st_ptr = ((wg != 0) ? smem_st_b : smem_st_a);
            int row_j = my_col / 8;
            int _min_1 = ((row_j) < (q_len - 1) ? (row_j) : (q_len - 1));
            int vis_j = _min_1;
            unsigned int sm_stage = 0;
            unsigned int sm_phase = 0;
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
            mbarrier_arrive(work_empty_addr + (work_stage_s) * 8);
            work_stage_s += 1;
            if (work_stage_s == 4) { work_stage_s = 0; _phase_work_full ^= 1; }
            unsigned int valid_s = valid;
            int kind_s = (int)kind;
            int block_begin_s = (int)block_begin;
            int block_end_s = (int)block_end;
            int seqlen_s = (int)seqlen;
            unsigned int _phase_p_empty_0 = 1;
            unsigned int _phase_stats_empty_0 = 1;
            #pragma unroll 1
            for (unsigned int _tile_iter_s = 0; _tile_iter_s < max_items; _tile_iter_s++) {
                if (valid_s == 0) {
                    break;
                }
                if (kind_s == 0) {
                    int cnt_s = block_end_s - block_begin_s;
                    int vis_min = seqlen_s - (q_len - 1);
                    int vis_col = vis_min + vis_j;
                    float row_max = -CAKE_INF;
                    float psum = 0.0f;
                    #pragma unroll 1
                    for (int n = 0; n < cnt_s; n++) {
                        if (wg_tid == 0) {
                        }
                        mbarrier_wait(s_full_addr + (sm_stage) * 8, sm_phase);
                        if (wg_tid == 0) {
                        }
                        float _tmem_load_0[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                            : "r"((unsigned int)my_s_base + sm_stage * 64));
                        int tok_w = lane_base + lane;
                        #pragma unroll
                        for (int c = 0; c < 32; c++) {
                            st_ptr[c * 129 + tok_w] = _tmem_load_0[c];
                        }
                        if (wg != 0) {
                            asm volatile("barrier.sync 9, 128;" ::: "memory");
                        } else {
                            asm volatile("barrier.sync 8, 128;" ::: "memory");
                        }
                        int my_block = block_begin_s + cnt_s - 1 - n;
                        int blk_pos = my_block * BLOCK_N + tok_base;
                        float svals[32];
                        float lmax = -CAKE_INF;
                        int needs_mask = ((vis_min < (my_block + 1) * BLOCK_N) ? 1 : 0);
                        #pragma unroll
                        for (int j = 0; j < 4; j++) {
                            #pragma unroll
                            for (int i = 0; i < 8; i++) {
                                float s_ji = st_ptr[col_local * 129 + tok_base + 32 * j + i];
                                if (needs_mask != 0) {
                                    if (vis_col <= blk_pos + 32 * j + i) {
                                        s_ji = -CAKE_INF;
                                    }
                                }
                                svals[j * 8 + i] = s_ji;
                                float _max_1 = max_noftz(lmax, s_ji);
                                lmax = _max_1;
                            }
                        }
                        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, lmax, 1);
                        float _max_2 = max_noftz(lmax, _shfl_xor_0);
                        lmax = _max_2;
                        float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, lmax, 2);
                        float _max_3 = max_noftz(lmax, _shfl_xor_1);
                        lmax = _max_3;
                        float _max_5 = max_noftz(row_max, lmax);
                        float new_max = _max_5;
                        float acc_scale = 1.0f;
                        if (row_max > -CAKE_INF) {
                            float _exp2_0 = approx_exp2(softmax_scale_log2 * (row_max - new_max));
                            acc_scale = _exp2_0;
                        }
                        if (quarter == 0) {
                            smem_scale[sm_stage * 64 + (unsigned int)my_col] = acc_scale;
                        }
                        mbarrier_arrive(corr_scale_addr + (sm_stage) * 8);
                        float safe_max = ((new_max == -CAKE_INF) ? 0.0f : new_max);
                        float p_vals[32];
                        float lsum = 0.0f;
                        #pragma unroll
                        for (int k = 0; k < 32; k++) {
                            float _exp2_1 = approx_exp2((svals[k] - safe_max) * softmax_scale_log2);
                            float p_k = _exp2_1;
                            p_vals[k] = p_k;
                            lsum = lsum + p_k;
                        }
                        psum = psum * acc_scale + lsum;
                        row_max = new_max;
                        if (wg_tid == 0) {
                        }
                        mbarrier_wait(p_empty_addr, _phase_p_empty_0);
                        _phase_p_empty_0 ^= 1;
                        if (wg_tid == 0) {
                        }
                        int k_run = 0;
                        unsigned int regs_p[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p_vals[_lp*2 + 0], p_vals[_lp*2+1 + 0]));
                            regs_p[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_p_addr + (unsigned int)((k_run / 64 * 64 + my_col) * 128 + (k_run % 64 + tok_base) * 2 ^ ((k_run / 64 * 64 + my_col) * 128 + (k_run % 64 + tok_base) * 2 >> 7 & 7) << 4))), "r"(regs_p[0]), "r"(regs_p[1]), "r"(regs_p[2]), "r"(regs_p[3]) : "memory");
                        int k_run_0 = 32;
                        unsigned int regs_p_1[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p_vals[_lp*2 + 8], p_vals[_lp*2+1 + 8]));
                            regs_p_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_p_addr + (unsigned int)((k_run_0 / 64 * 64 + my_col) * 128 + (k_run_0 % 64 + tok_base) * 2 ^ ((k_run_0 / 64 * 64 + my_col) * 128 + (k_run_0 % 64 + tok_base) * 2 >> 7 & 7) << 4))), "r"(regs_p_1[0]), "r"(regs_p_1[1]), "r"(regs_p_1[2]), "r"(regs_p_1[3]) : "memory");
                        int k_run_2 = 64;
                        unsigned int regs_p_3[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p_vals[_lp*2 + 16], p_vals[_lp*2+1 + 16]));
                            regs_p_3[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_p_addr + (unsigned int)((k_run_2 / 64 * 64 + my_col) * 128 + (k_run_2 % 64 + tok_base) * 2 ^ ((k_run_2 / 64 * 64 + my_col) * 128 + (k_run_2 % 64 + tok_base) * 2 >> 7 & 7) << 4))), "r"(regs_p_3[0]), "r"(regs_p_3[1]), "r"(regs_p_3[2]), "r"(regs_p_3[3]) : "memory");
                        int k_run_4 = 96;
                        unsigned int regs_p_5[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(p_vals[_lp*2 + 24], p_vals[_lp*2+1 + 24]));
                            regs_p_5[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_p_addr + (unsigned int)((k_run_4 / 64 * 64 + my_col) * 128 + (k_run_4 % 64 + tok_base) * 2 ^ ((k_run_4 / 64 * 64 + my_col) * 128 + (k_run_4 % 64 + tok_base) * 2 >> 7 & 7) << 4))), "r"(regs_p_5[0]), "r"(regs_p_5[1]), "r"(regs_p_5[2]), "r"(regs_p_5[3]) : "memory");
                        asm volatile("fence.proxy.async;");
                        mbarrier_arrive(p_full_addr);
                        if (wg != 0) {
                            asm volatile("barrier.sync 9, 128;" ::: "memory");
                        } else {
                            asm volatile("barrier.sync 8, 128;" ::: "memory");
                        }
                        sm_stage += 1;
                        if (sm_stage == 2) { sm_stage = 0; sm_phase ^= 1; }
                        if (wg_tid == 0) {
                            if (wg == 0) {
                            }
                        }
                    }
                    float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, psum, 1);
                    float total = psum + _shfl_xor_3;
                    float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, total, 2);
                    total = total + _shfl_xor_4;
                    if (wg_tid == 0) {
                    }
                    mbarrier_wait(stats_empty_addr, _phase_stats_empty_0);
                    _phase_stats_empty_0 ^= 1;
                    if (quarter == 0) {
                        smem_sum[my_col] = total;
                        smem_max[my_col] = row_max;
                    }
                    mbarrier_arrive(stats_full_addr);
                    if (wg_tid == 0) {
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
                mbarrier_arrive(work_empty_addr + (work_stage_s) * 8);
                work_stage_s += 1;
                if (work_stage_s == 4) { work_stage_s = 0; _phase_work_full ^= 1; }
                valid_s = valid_1;
                kind_s = (int)kind_2;
                block_begin_s = (int)block_begin_5;
                block_end_s = (int)block_end_6;
                seqlen_s = (int)seqlen_7;
            }
        }
    // ---- Role: correction ----
    } else if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 88;");
        { // correction_main
            const int warp_in_wg_c = warp % 4;
            const int corr_row = warp_in_wg_c * 32 << 16;
            int wg_tid_c = warp_in_wg_c * 32 + lane;
            int d_idx = warp_in_wg_c * 32 + lane;
            int merges_per_tile_c = q_len * 8 / 4;
            int live_rows = q_len * 8;
            unsigned int sc_buf = 0;
            unsigned int sc_phase = 0;
            int n_merges_seen = 0;
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
            mbarrier_arrive(work_empty_addr + (work_stage_c) * 8);
            work_stage_c += 1;
            if (work_stage_c == 4) { work_stage_c = 0; _phase_work_full_1 ^= 1; }
            unsigned int valid_c = valid_2;
            int kind_c = (int)kind_1;
            int batch_c = (int)batch_1;
            int kv_head_c = (int)kv_head_1;
            int block_begin_c = (int)block_begin_1;
            int block_end_c = (int)block_end_1;
            int n_chunks_c = (int)n_chunks_1;
            int slot_tile_base_c = (int)slot_tile_base_1;
            int counter_idx_c = (int)counter_idx_1;
            int chunk_c = (int)chunk_1;
            unsigned int _phase_o_ready_0 = 0;
            unsigned int _phase_stats_full_0 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_c = 0; _tile_iter_c < max_items; _tile_iter_c++) {
                if (valid_c == 0) {
                    break;
                }
                if (kind_c == 0) {
                    int cnt_c = block_end_c - block_begin_c;
                    int my_slot = slot_tile_base_c + chunk_c * num_kv_heads;
                    #pragma unroll 1
                    for (int n_1 = 0; n_1 < cnt_c; n_1++) {
                        if (wg_tid_c == 0) {
                        }
                        mbarrier_wait(corr_scale_addr + (sc_buf) * 8, sc_phase);
                        if (wg_tid_c == 0) {
                        }
                        if (n_1 > 0) {
                            mbarrier_wait(o_ready_addr, _phase_o_ready_0);
                            _phase_o_ready_0 ^= 1;
                            if (wg_tid_c == 0) {
                            }
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int sc_off = (int)sc_buf * 64;
                            int need = 0;
                            #pragma unroll
                            for (int c_1 = 0; c_1 < 64; c_1++) {
                                need = need | ((smem_scale[sc_off + c_1] != 1.0f) ? 1 : 0);
                            }
                            int _vote_1 = __any_sync(0xFFFFFFFF, need != 0);
                            if (_vote_1 != 0) {
                                int o_base = taddr + 128 + (unsigned int)corr_row;
                                #pragma unroll
                                for (int chunk_2 = 0; chunk_2 < 2; chunk_2++) {
                                    float _tmem_load_1[32];
                                    asm volatile(
                                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                        : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31])
                                        : "r"(o_base + chunk_2 * 32));
                                    #pragma unroll
                                    for (int c_2 = 0; c_2 < 32; c_2++) {
                                        _tmem_load_1[c_2] = _tmem_load_1[c_2] * smem_scale[sc_off + chunk_2 * 32 + c_2];
                                    }
                                    tmem_st_x32_f32(o_base + chunk_2 * 32, _tmem_load_1);
                                }
                                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            }
                        }
                        if (wg_tid_c == 0) {
                        }
                        mbarrier_arrive(p_full_addr);
                        sc_buf += 1;
                        if (sc_buf == 2) { sc_buf = 0; sc_phase ^= 1; }
                        if (wg_tid_c == 0) {
                        }
                    }
                    mbarrier_wait(o_ready_addr, _phase_o_ready_0);
                    _phase_o_ready_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (wg_tid_c == 0) {
                    }
                    mbarrier_wait(stats_full_addr, _phase_stats_full_0);
                    _phase_stats_full_0 ^= 1;
                    if (wg_tid_c == 0) {
                    }
                    int publish_split = ((n_chunks_c > 1) ? 1 : 0);
                    int o_base_e = taddr + 128 + (unsigned int)corr_row;
                    #pragma unroll
                    for (int chunk_3 = 0; chunk_3 < 2; chunk_3++) {
                        float _tmem_load_2[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_2[0]), "=f"(_tmem_load_2[1]), "=f"(_tmem_load_2[2]), "=f"(_tmem_load_2[3]), "=f"(_tmem_load_2[4]), "=f"(_tmem_load_2[5]), "=f"(_tmem_load_2[6]), "=f"(_tmem_load_2[7]), "=f"(_tmem_load_2[8]), "=f"(_tmem_load_2[9]), "=f"(_tmem_load_2[10]), "=f"(_tmem_load_2[11]), "=f"(_tmem_load_2[12]), "=f"(_tmem_load_2[13]), "=f"(_tmem_load_2[14]), "=f"(_tmem_load_2[15]), "=f"(_tmem_load_2[16]), "=f"(_tmem_load_2[17]), "=f"(_tmem_load_2[18]), "=f"(_tmem_load_2[19]), "=f"(_tmem_load_2[20]), "=f"(_tmem_load_2[21]), "=f"(_tmem_load_2[22]), "=f"(_tmem_load_2[23]), "=f"(_tmem_load_2[24]), "=f"(_tmem_load_2[25]), "=f"(_tmem_load_2[26]), "=f"(_tmem_load_2[27]), "=f"(_tmem_load_2[28]), "=f"(_tmem_load_2[29]), "=f"(_tmem_load_2[30]), "=f"(_tmem_load_2[31])
                            : "r"(o_base_e + chunk_3 * 32));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        if (publish_split == 0) {
                            #pragma unroll
                            for (int c_3 = 0; c_3 < 32; c_3++) {
                                const int r_c = chunk_3 * 32 + c_3;
                                float row_sum_c = smem_sum[r_c];
                                float _rcp_4 = approx_rcp(row_sum_c);
                                float inv_c = ((row_sum_c > 0.0f) ? _rcp_4 : 0.0f);
                                float val_c = _tmem_load_2[c_3] * inv_c;
                                if (r_c < live_rows) {
                                    int j_c = r_c / 8;
                                    int h_c = r_c % 8;
                                    int q_head_c = kv_head_c * 8 + h_c;
                                    int o_idx = ((batch_c * q_len + j_c) * num_q_heads + q_head_c) * HEAD_DIM + d_idx;
                                    *(reinterpret_cast<__nv_bfloat16*>(O_ptr + o_idx) + (0)) = __float2bfloat16_rn(val_c);
                                }
                            }
                        } else {
                            #pragma unroll
                            for (int c_4 = 0; c_4 < 32; c_4++) {
                                const int r_p = chunk_3 * 32 + c_4;
                                int p_idx = my_slot * 8192 + r_p * HEAD_DIM + d_idx;
                                *(reinterpret_cast<float*>(partial_o + p_idx) + (0)) = _tmem_load_2[c_4];
                            }
                        }
                    }
                    if (publish_split != 0) {
                        if (wg_tid_c < 64) {
                            *(reinterpret_cast<float*>(partial_stats + (my_slot * 128 + wg_tid_c)) + (0)) = smem_max[wg_tid_c];
                            *(reinterpret_cast<float*>(partial_stats + (my_slot * 128 + 64 + wg_tid_c)) + (0)) = smem_sum[wg_tid_c];
                        }
                    }
                    mbarrier_arrive(o_empty_addr);
                    if (elect_sync()) {
                        mbarrier_arrive(stats_empty_addr);
                    }
                    if (wg_tid_c == 0) {
                        if (_tile_iter_c == 0) {
                        }
                    }
                    if (publish_split != 0) {
                        __threadfence();
                        asm volatile("barrier.sync 10, 128;" ::: "memory");
                        if (wg_tid_c == 0) {
                            unsigned int _atomic_old_1;
                            asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                                : "=r"(_atomic_old_1) : "l"(&tile_counters[counter_idx_c * 2]), "r"(static_cast<uint32_t>(1)) : "memory");
                        }
                    }
                } else {
                    int merge_slice_c = chunk_c;
                    if (wg_tid_c == 0) {
                        if (n_merges_seen == 0) {
                        }
                        #pragma unroll 1
                        for (int _poll = 0; _poll < 1073741824; _poll++) {
                            unsigned int _atomic_old_2;
                            asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                                : "=r"(_atomic_old_2) : "l"(&tile_counters[counter_idx_c * 2]), "r"(static_cast<uint32_t>(0)) : "memory");
                            unsigned int arrived = _atomic_old_2;
                            if (n_chunks_c <= (int)arrived) {
                                break;
                            }
                        }
                    }
                    asm volatile("barrier.sync 10, 128;" ::: "memory");
                    asm volatile("fence.acquire.gpu;" ::: "memory");
                    if (wg_tid_c == 0) {
                        if (n_merges_seen == 0) {
                        }
                    }
                    int r_m = merge_slice_c * 4 + warp_in_wg_c;
                    int d0_m = lane * 4;
                    int stats_row = slot_tile_base_c * 128 + r_m;
                    int stats_stride = num_kv_heads * 128;
                    float lmax_m = -CAKE_INF;
                    #pragma unroll 4
                    for (int c_a = lane; c_a < n_chunks_c; c_a += 32) {
                        float _max_6 = max_noftz(lmax_m, partial_stats[stats_row + c_a * stats_stride]);
                        lmax_m = _max_6;
                    }
                    float _warp_reduce_0 = lmax_m;
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1)
                        _warp_reduce_0 = max_noftz(_warp_reduce_0, __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_0, offset));
                    float m_row = _warp_reduce_0;
                    float lden_m = 0.0f;
                    #pragma unroll 4
                    for (int c_d = lane; c_d < n_chunks_c; c_d += 32) {
                        float m_d = partial_stats[stats_row + c_d * stats_stride];
                        float l_d = partial_stats[stats_row + 64 + c_d * stats_stride];
                        float _exp2_2 = approx_exp2((m_d - m_row) * softmax_scale_log2);
                        float _fma_0 = __fmaf_rn(_exp2_2, l_d, lden_m);
                        lden_m = _fma_0;
                    }
                    float _warp_reduce_1 = lden_m;
                    #pragma unroll
                    for (int offset = 16; offset > 0; offset >>= 1)
                        _warp_reduce_1 += __shfl_xor_sync(0xFFFFFFFF, _warp_reduce_1, offset);
                    float den = _warp_reduce_1;
                    float acc[4];
                    acc[0] = 0.0f;
                    acc[1] = 0.0f;
                    acc[2] = 0.0f;
                    acc[3] = 0.0f;
                    int o_row = slot_tile_base_c * 8192 + r_m * HEAD_DIM + d0_m;
                    int o_stride = num_kv_heads * 8192;
                    #pragma unroll 8
                    for (int c_m = 0; c_m < n_chunks_c; c_m++) {
                        float m_c = partial_stats[stats_row + c_m * stats_stride];
                        float _vec_load_2[4];
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(partial_o + (o_row + c_m * o_stride) + 0);
                            _vec_load_2[0 + 0] = _v4.x;
                            _vec_load_2[0 + 1] = _v4.y;
                            _vec_load_2[0 + 2] = _v4.z;
                            _vec_load_2[0 + 3] = _v4.w;
                        }
                        float _exp2_3 = approx_exp2((m_c - m_row) * softmax_scale_log2);
                        float w_c = _exp2_3;
                        #pragma unroll
                        for (int k_1 = 0; k_1 < 4; k_1++) {
                            float _fma_1 = __fmaf_rn(_vec_load_2[k_1], w_c, acc[k_1]);
                            acc[k_1] = _fma_1;
                        }
                    }
                    float _rcp_5 = approx_rcp(den);
                    float inv_den = ((den > 0.0f) ? _rcp_5 : 0.0f);
                    float out_m[4];
                    #pragma unroll
                    for (int k_2 = 0; k_2 < 4; k_2++) {
                        out_m[k_2] = acc[k_2] * inv_den;
                    }
                    if (r_m < live_rows) {
                        int j_m = r_m / 8;
                        int h_m = r_m % 8;
                        int q_head_m = kv_head_c * 8 + h_m;
                        int o_idx_m = ((batch_c * q_len + j_m) * num_q_heads + q_head_m) * HEAD_DIM + d0_m;
                        {
                            uint2 _pk2;
                            __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
                            _pk[0] = __floats2bfloat162_rn(out_m[0 + 0], out_m[0 + 1]);
                            _pk[1] = __floats2bfloat162_rn(out_m[0 + 2], out_m[0 + 3]);
                            *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(O_ptr + o_idx_m))[0]) = _pk2;
                        }
                    }
                    asm volatile("barrier.sync 10, 128;" ::: "memory");
                    n_merges_seen = n_merges_seen + 1;
                    if (wg_tid_c == 0) {
                    }
                    if (wg_tid_c == 0) {
                        unsigned int _atomic_old_3;
                        asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;"
                            : "=r"(_atomic_old_3) : "l"(&tile_counters[counter_idx_c * 2 + 1]), "r"(static_cast<uint32_t>(1)) : "memory");
                        unsigned int merged_old = _atomic_old_3;
                        if ((int)merged_old + 1 == merges_per_tile_c) {
                            *(reinterpret_cast<unsigned int*>(tile_counters + (counter_idx_c * 2)) + (0)) = 0;
                            *(reinterpret_cast<unsigned int*>(tile_counters + (counter_idx_c * 2 + 1)) + (0)) = 0;
                        }
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
                mbarrier_arrive(work_empty_addr + (work_stage_c) * 8);
                work_stage_c += 1;
                if (work_stage_c == 4) { work_stage_c = 0; _phase_work_full_1 ^= 1; }
                valid_c = valid_1_1;
                kind_c = (int)kind_2_1;
                batch_c = (int)batch_3_1;
                kv_head_c = (int)kv_head_4_1;
                block_begin_c = (int)block_begin_5_1;
                block_end_c = (int)block_end_6_1;
                n_chunks_c = (int)n_chunks_8_1;
                slot_tile_base_c = (int)slot_tile_base_9_1;
                counter_idx_c = (int)counter_idx_10_1;
                chunk_c = (int)chunk_11_1;
            }
            mbarrier_arrive(tmem_dealloc_addr);
        }
    // ---- Role: mma_warp ----
    } else if (warp == 12) {
        { // mma_warp_main
            unsigned int work_stage_m = 0;
            unsigned int q_cons_stage = 0;
            unsigned int q_cons_phase = 0;
            int s_buf = 0;
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
            unsigned int chunk_4 = work_token_words[base_2 + 10];
            mbarrier_arrive(work_empty_addr + (work_stage_m) * 8);
            work_stage_m += 1;
            if (work_stage_m == 4) { work_stage_m = 0; _phase_work_full_2 ^= 1; }
            unsigned int valid_m = valid_3;
            int kind_m = (int)kind_3;
            int block_begin_m = (int)block_begin_2;
            int block_end_m = (int)block_end_2;
            unsigned int _phase_o_empty_0 = 1;
            unsigned int _phase_p_full_0 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_m = 0; _tile_iter_m < max_items; _tile_iter_m++) {
                if (valid_m == 0) {
                    break;
                }
                if (kind_m == 0) {
                    int cnt_m = block_end_m - block_begin_m;
                    mbarrier_wait(q_full_addr + (q_cons_stage) * 8, q_cons_phase);
                    mbarrier_wait(kv_full_addr, 0);
                    if (_tile_iter_m == 0) {
                        if (lane == 0) {
                        }
                    }
                    int _mma_a_lo_0 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (0) * 2048);
                    int _mma_b_lo_0 = make_warp_uniform((((smem_qt_addr) >> 4) & 0x3FFF) + (q_cons_stage) * 1024);
                    {
                        uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
                        uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_tmem_s + (s_buf * 64)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135267472, 0);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_tmem_s + (s_buf * 64)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135267472, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_tmem_s + (s_buf * 64)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135267472, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_tmem_s + (s_buf * 64)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135267472, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 1018U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 506U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_tmem_s + (s_buf * 64)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135267472, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_tmem_s + (s_buf * 64)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135267472, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_tmem_s + (s_buf * 64)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135267472, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_tmem_s + (s_buf * 64)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 135267472, 1);
                        }
                    }
                    elect_commit(s_full_addr + (s_buf) * 8);
                    elect_commit(kv_empty_addr);
                    s_buf = s_buf ^ 1;
                    if (lane == 0) {
                    }
                    mbarrier_wait(o_empty_addr, _phase_o_empty_0);
                    _phase_o_empty_0 ^= 1;
                    if (lane == 0) {
                    }
                    int first_pv = 1;
                    #pragma unroll 1
                    for (int n_2 = 0; n_2 < cnt_m; n_2++) {
                        int stage = n_2 % NUM_KV_STAGES;
                        int next_n = n_2 + 1;
                        if (lane == 0) {
                        }
                        if (next_n < cnt_m) {
                            int nstage = next_n % NUM_KV_STAGES;
                            mbarrier_wait(kv_full_addr + (nstage) * 8, 0);
                            if (lane == 0) {
                            }
                            int _mma_a_lo_1 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (nstage) * 2048);
                            int _mma_b_lo_1 = make_warp_uniform((((smem_qt_addr) >> 4) & 0x3FFF) + (q_cons_stage) * 1024);
                            {
                                uint64_t _mma_ss_a_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_1);
                                uint64_t _mma_ss_b_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_1);
                                if (elect_sync()) {
                                    tcgen05_mma_f16((tmem_tmem_s + (s_buf * 64)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 135267472, 0);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f16((tmem_tmem_s + (s_buf * 64)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 135267472, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f16((tmem_tmem_s + (s_buf * 64)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 135267472, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f16((tmem_tmem_s + (s_buf * 64)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 135267472, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_1, 1018U);
                                incr_smem_desc_lo(_mma_ss_b_desc_1, 506U);
                                if (elect_sync()) {
                                    tcgen05_mma_f16((tmem_tmem_s + (s_buf * 64)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 135267472, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f16((tmem_tmem_s + (s_buf * 64)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 135267472, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f16((tmem_tmem_s + (s_buf * 64)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 135267472, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f16((tmem_tmem_s + (s_buf * 64)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 135267472, 1);
                                }
                            }
                            elect_commit(s_full_addr + (s_buf) * 8);
                            elect_commit(kv_empty_addr + (nstage) * 8);
                            s_buf = s_buf ^ 1;
                        }
                        if (lane == 0) {
                        }
                        mbarrier_wait(kv_full_addr + (stage) * 8, 1);
                        if (lane == 0) {
                        }
                        mbarrier_wait(p_full_addr, _phase_p_full_0);
                        _phase_p_full_0 ^= 1;
                        if (lane == 0) {
                        }
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int first_pv_flag = first_pv;
                        int _mma_a_lo_2 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (stage) * 2048);
                        int _mma_b_lo_2 = make_warp_uniform((((smem_p_addr) >> 4) & 0x3FFF) | 0x2000000);
                        {
                            uint64_t _mma_ss_a_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_2);
                            uint64_t _mma_ss_b_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_2);
                            if (elect_sync()) {
                                tcgen05_mma_f16(tmem_tmem_o, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 135300240, ((first_pv_flag) ? 0 : 1));
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_2, 128U);
                            incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16(tmem_tmem_o, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 135300240, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_2, 128U);
                            incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16(tmem_tmem_o, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 135300240, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_2, 128U);
                            incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16(tmem_tmem_o, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 135300240, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_2, 128U);
                            incr_smem_desc_lo(_mma_ss_b_desc_2, 506U);
                            if (elect_sync()) {
                                tcgen05_mma_f16(tmem_tmem_o, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 135300240, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_2, 128U);
                            incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16(tmem_tmem_o, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 135300240, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_2, 128U);
                            incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16(tmem_tmem_o, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 135300240, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_2, 128U);
                            incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16(tmem_tmem_o, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 135300240, 1);
                            }
                        }
                        elect_commit2(kv_empty_addr + (stage) * 8, o_ready_addr);
                        elect_commit(p_empty_addr);
                        first_pv = 0;
                        if (lane == 0) {
                        }
                    }
                    elect_commit(q_empty_addr + (q_cons_stage) * 8);
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
        }
    // ---- Role: load_pgoff ----
    } else if (warp == 13) {
        { // load_pgoff_main
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
            unsigned int chunk_5 = work_token_words[base_3 + 10];
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
                    int max_pg_p = (seqlen_kv_p + PAGE_SIZE - 1) / PAGE_SIZE - 1;
                    int pt_base_p = batch_idx_p * max_pages_per_seq;
                    #pragma unroll 1
                    for (int ni_p = 0; ni_p < cta_n_blocks_p; ni_p++) {
                        int n_block_p = block_begin_p + cta_n_blocks_p - 1 - ni_p;
                        int logical_page_base_p = n_block_p * 8;
                        mbarrier_wait(page_offsets_empty_addr + (page_prod_stage) * 8, page_prod_phase);
                        if (elect_sync()) {
                            int smem_page_base_p = page_prod_stage * 8;
                            if (max_pg_p >= logical_page_base_p + 7 && pt_base_p % 4 == 0) {
                                int _vec_load_0[4];
                                {
                                    const int4* _ivptr_0 = reinterpret_cast<const int4*>(page_table + pt_base_p + logical_page_base_p);
                                    int4 _ivld_0;
                                    _ivld_0 = *_ivptr_0;
                                    _vec_load_0[0 + 0] = _ivld_0.x;
                                    _vec_load_0[0 + 1] = _ivld_0.y;
                                    _vec_load_0[0 + 2] = _ivld_0.z;
                                    _vec_load_0[0 + 3] = _ivld_0.w;
                                }
                                int _vec_load_1[4];
                                {
                                    const int4* _ivptr_1 = reinterpret_cast<const int4*>(page_table + pt_base_p + logical_page_base_p + 4);
                                    int4 _ivld_1;
                                    _ivld_1 = *_ivptr_1;
                                    _vec_load_1[0 + 0] = _ivld_1.x;
                                    _vec_load_1[0 + 1] = _ivld_1.y;
                                    _vec_load_1[0 + 2] = _ivld_1.z;
                                    _vec_load_1[0 + 3] = _ivld_1.w;
                                }
                                #pragma unroll
                                for (int pg_i_p = 0; pg_i_p < 4; pg_i_p++) {
                                    smem_page_offsets[smem_page_base_p + pg_i_p] = _vec_load_0[pg_i_p];
                                    smem_page_offsets[smem_page_base_p + pg_i_p + 4] = _vec_load_1[pg_i_p];
                                }
                            } else {
                                #pragma unroll
                                for (int pg_i_p_1 = 0; pg_i_p_1 < 8; pg_i_p_1++) {
                                    int safe_page_idx_p = logical_page_base_p + pg_i_p_1;
                                    if (safe_page_idx_p > max_pg_p) {
                                        safe_page_idx_p = max_pg_p;
                                    }
                                    smem_page_offsets[smem_page_base_p + pg_i_p_1] = page_table[pt_base_p + safe_page_idx_p];
                                }
                            }
                            mbarrier_arrive(page_offsets_full_addr + (page_prod_stage) * 8);
                        }
                        page_prod_stage += 1;
                        if (page_prod_stage == 6) { page_prod_stage = 0; page_prod_phase ^= 1; }
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
        }
    // ---- Role: scheduler ----
    } else if (warp == 14) {
        { // scheduler_main
            int lane_0 = lane;
            int num_ctas = gridDim.x;
            if (lane_0 == 0) {
            }
            int items_per_chunk = num_kv_heads;
            int merges_per_tile = q_len * 8 / 4;
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
                    int s1 = sched_seq_lens[b1];
                    pairs1 = (unsigned int)((s1 + 255) / 256);
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
            unsigned int best_cost = 4294967295;
            unsigned int best_pairs = chunk_pairs_u;
            unsigned int prev_cand = 0;
            float _rcp_0 = approx_rcp((float)num_ctas);
            float ctas_rcp = _rcp_0;
            #pragma unroll 1
            for (int kc = 0; kc < 10; kc++) {
                unsigned int cand = chunk_pairs_u;
                if (kc == 9) {
                    cand = p_max;
                }
                if (kc < 8) {
                    unsigned int div_c = (unsigned int)(kc + 1) * (unsigned int)num_ctas;
                    float _rcp_1 = approx_rcp((float)div_c);
                    unsigned int q = (unsigned int)((float)total_work * _rcp_1);
                    if (q > 0) {
                        if (total_work < q * div_c) {
                            q = q - 1;
                        }
                    }
                    if (total_work >= (q + 1) * div_c) {
                        q = q + 1;
                    }
                    if (total_work > q * div_c) {
                        q = q + 1;
                    }
                    cand = q;
                    if (cand < 2) {
                        cand = 2;
                    }
                }
                if (cand != prev_cand) {
                    prev_cand = cand;
                    float _rcp_2 = approx_rcp((float)cand);
                    float cand_rcp = _rcp_2;
                    unsigned int tickets_c = 0;
                    unsigned int split_c = 0;
                    #pragma unroll 1
                    for (int gc = 0; gc < num_groups; gc++) {
                        int bc = gc * 32 + lane_0;
                        unsigned int nb_c = 0;
                        unsigned int sp_c = 0;
                        if (bc < batch_size) {
                            int sc = sched_seq_lens[bc];
                            unsigned int pairs_c = (unsigned int)((sc + 255) / 256);
                            unsigned int q_1 = (unsigned int)((float)pairs_c * cand_rcp);
                            if (q_1 > 0) {
                                if (pairs_c < q_1 * cand) {
                                    q_1 = q_1 - 1;
                                }
                            }
                            if (pairs_c >= (q_1 + 1) * cand) {
                                q_1 = q_1 + 1;
                            }
                            if (pairs_c > q_1 * cand) {
                                q_1 = q_1 + 1;
                            }
                            nb_c = q_1;
                            if (nb_c > 1) {
                                sp_c = 1;
                            }
                        }
                        unsigned int _warp_redux_u32_3;
                        asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_3) : "r"(nb_c));
                        tickets_c += _warp_redux_u32_3;
                        unsigned int _warp_redux_u32_4;
                        asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_4) : "r"(sp_c));
                        split_c += _warp_redux_u32_4;
                    }
                    tickets_c = tickets_c * (unsigned int)items_per_chunk;
                    unsigned int q_2 = (unsigned int)((float)tickets_c * ctas_rcp);
                    if (q_2 > 0) {
                        if (tickets_c < q_2 * (unsigned int)num_ctas) {
                            q_2 = q_2 - 1;
                        }
                    }
                    if (tickets_c >= (q_2 + 1) * (unsigned int)num_ctas) {
                        q_2 = q_2 + 1;
                    }
                    if (tickets_c > q_2 * (unsigned int)num_ctas) {
                        q_2 = q_2 + 1;
                    }
                    unsigned int waves_c = q_2;
                    unsigned int merge_c = split_c * (unsigned int)items_per_chunk * (unsigned int)merges_per_tile;
                    unsigned int q_0 = (unsigned int)((float)merge_c * ctas_rcp);
                    if (q_0 > 0) {
                        if (merge_c < q_0 * (unsigned int)num_ctas) {
                            q_0 = q_0 - 1;
                        }
                    }
                    if (merge_c >= (q_0 + 1) * (unsigned int)num_ctas) {
                        q_0 = q_0 + 1;
                    }
                    if (merge_c > q_0 * (unsigned int)num_ctas) {
                        q_0 = q_0 + 1;
                    }
                    unsigned int merge_waves_c = q_0;
                    unsigned int cost_c = waves_c * (cand + 1) + merge_waves_c;
                    if (cost_c < best_cost) {
                        best_cost = cost_c;
                        best_pairs = cand;
                    }
                }
            }
            chunk_pairs_u = best_pairs;
            int chunk_pairs = (int)chunk_pairs_u;
            float _rcp_3 = approx_rcp((float)chunk_pairs_u);
            float chunk_rcp = _rcp_3;
            unsigned int n_full_chunks = 0;
            unsigned int n_chunks_total = 0;
            unsigned int n_split_reqs = 0;
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
                unsigned int split2 = 0;
                if (b2 < batch_size) {
                    int s2 = sched_seq_lens[b2];
                    unsigned int pairs2 = (unsigned int)((s2 + 255) / 256);
                    unsigned int q_3 = (unsigned int)((float)pairs2 * chunk_rcp);
                    if (q_3 > 0) {
                        if (pairs2 < q_3 * chunk_pairs_u) {
                            q_3 = q_3 - 1;
                        }
                    }
                    if (pairs2 >= (q_3 + 1) * chunk_pairs_u) {
                        q_3 = q_3 + 1;
                    }
                    if (pairs2 > q_3 * chunk_pairs_u) {
                        q_3 = q_3 + 1;
                    }
                    n2 = q_3;
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
                    if (n2 > 1) {
                        split2 = 1;
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
                n_split_reqs += _warp_redux_u32_7;
                unsigned int _warp_redux_u32_8;
                asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_8) : "r"(pack2));
                unsigned int pack_group = _warp_redux_u32_8;
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
            unsigned int chunk_items = n_chunks_total * (unsigned int)items_per_chunk;
            unsigned int merge_items = n_split_reqs * (unsigned int)items_per_chunk * (unsigned int)merges_per_tile;
            unsigned int total_items = chunk_items + merge_items;
            if (blockIdx.x == 0) {
                if (lane_0 == 0) {
                    *(reinterpret_cast<unsigned int*>(queue_counters + 2) + (0)) = chunk_pairs_u;
                    *(reinterpret_cast<unsigned int*>(queue_counters + 3) + (0)) = total_items;
                }
            }
            if (lane_0 == 0) {
            }
            unsigned int work_stage_sched = 0;
            int cur_group_b[5];
            unsigned int before_b[5];
            unsigned int si_before_b[5];
            unsigned int st_before_b[5];
            #pragma unroll
            for (int bi_3 = 0; bi_3 < 5; bi_3++) {
                cur_group_b[bi_3] = 0;
                before_b[bi_3] = 0;
                si_before_b[bi_3] = 0;
                st_before_b[bi_3] = 0;
            }
            unsigned int first_claim = 1;
            unsigned int _phase_work_empty = 1;
            #pragma unroll 1
            for (unsigned int _claim = 0; _claim < max_items + 1; _claim++) {
                mbarrier_wait(work_empty_addr + (work_stage_sched) * 8, _phase_work_empty);
                unsigned int ticket_lane0 = blockIdx.x;
                if (first_claim == 0) {
                    if (lane_0 == 0) {
                        unsigned int _atomic_old_0 = atomicAdd(&queue_counters[0], 1);
                        ticket_lane0 = _atomic_old_0 + (unsigned int)num_ctas;
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
                    for (int bi_4 = 0; bi_4 < 4; bi_4++) {
                        if (ticket >= bucket_end[bi_4]) {
                            bucket = bi_4 + 1;
                            bucket_start = bucket_end[bi_4];
                        }
                    }
                    int is_merge = ((bucket == 4) ? 1 : 0);
                    unsigned int local_items = ticket - bucket_start;
                    unsigned int local_chunk = local_items / (unsigned int)items_per_chunk;
                    int in_chunk = (int)(local_items - local_chunk * (unsigned int)items_per_chunk);
                    int merge_slice = 0;
                    if (is_merge != 0) {
                        unsigned int tile_local = local_items / (unsigned int)merges_per_tile;
                        merge_slice = (int)(local_items - tile_local * (unsigned int)merges_per_tile);
                        local_chunk = tile_local / (unsigned int)items_per_chunk;
                        in_chunk = (int)(tile_local - local_chunk * (unsigned int)items_per_chunk);
                    }
                    int cursor_group = 0;
                    unsigned int before = 0;
                    unsigned int si_before = 0;
                    unsigned int st_before = 0;
                    #pragma unroll
                    for (int bi_5 = 0; bi_5 < 5; bi_5++) {
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
                            pairs3 = (s3 + 255) / 256;
                            unsigned int q_4 = (unsigned int)((float)(unsigned int)pairs3 * chunk_rcp);
                            if (q_4 > 0) {
                                if (q_4 * chunk_pairs_u > (unsigned int)pairs3) {
                                    q_4 = q_4 - 1;
                                }
                            }
                            if ((q_4 + 1) * chunk_pairs_u <= (unsigned int)pairs3) {
                                q_4 = q_4 + 1;
                            }
                            if (q_4 * chunk_pairs_u < (unsigned int)pairs3) {
                                q_4 = q_4 + 1;
                            }
                            n3 = (int)q_4;
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
                            } else if (is_merge != 0) {
                                mine3 = split_tiles3;
                            } else {
                                if (rem3 > 0) {
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
                    for (int bi_6 = 0; bi_6 < 5; bi_6++) {
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
                    int kv_head_sel = in_chunk;
                    int n_blocks_tile = (sel_seqlen + BLOCK_N - 1) / BLOCK_N;
                    int block_begin_4 = 2 * chunk_idx * chunk_pairs;
                    int block_end_4 = 2 * (chunk_idx + 1) * chunk_pairs;
                    if (chunk_idx + 1 == sel_n) {
                        block_end_4 = n_blocks_tile;
                    }
                    int slot_tile_base_4 = 0;
                    int counter_idx_4 = 0;
                    if (sel_n > 1) {
                        slot_tile_base_4 = ((int)si_before + sel_excl_si) * items_per_chunk + kv_head_sel;
                        counter_idx_4 = ((int)st_before + sel_excl_st) * items_per_chunk + kv_head_sel;
                    }
                    if (lane_0 == 0) {
                        work_token_words[token_base + 1] = (unsigned int)is_merge;
                        work_token_words[token_base + 2] = (unsigned int)sel_batch;
                        work_token_words[token_base + 3] = (unsigned int)kv_head_sel;
                        work_token_words[token_base + 4] = (unsigned int)block_begin_4;
                        work_token_words[token_base + 5] = (unsigned int)block_end_4;
                        work_token_words[token_base + 6] = (unsigned int)sel_seqlen;
                        work_token_words[token_base + 7] = (unsigned int)sel_n;
                        work_token_words[token_base + 8] = (unsigned int)slot_tile_base_4;
                        work_token_words[token_base + 9] = (unsigned int)counter_idx_4;
                        if (is_merge != 0) {
                            work_token_words[token_base + 10] = (unsigned int)merge_slice;
                        } else {
                            work_token_words[token_base + 10] = (unsigned int)chunk_idx;
                        }
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
    // ---- Role: load_warp ----
    } else if (warp == 15) {
        { // load_warp_main
            unsigned int q_prod_stage = 0;
            unsigned int q_prod_phase = 1;
            unsigned int page_cons_stage = 0;
            unsigned int page_cons_phase = 0;
            unsigned int work_stage_l = 0;
            {
                int spec_tiles = batch_size * num_kv_heads;
                int spec_id = blockIdx.x;
                if (spec_id < spec_tiles) {
                    if (elect_sync()) {
                        int spec_batch = spec_id / num_kv_heads;
                        int spec_head = spec_id - spec_batch * num_kv_heads;
                        int spec_seqlen = seq_lens_kv[spec_batch];
                        int spec_blocks = (spec_seqlen + BLOCK_N - 1) / BLOCK_N;
                        int spec_max_pg = (spec_seqlen + PAGE_SIZE - 1) / PAGE_SIZE - 1;
                        int spec_pt_base = spec_batch * max_pages_per_seq;
                        asm volatile("cp.async.bulk.prefetch.tensor.4d.L2.global.tile [%0, {%1, %2, %3, %4}];" :: "l"((uint64_t)((&Q))), "r"((int)(0)), "r"((int)(spec_head * 8)), "r"((int)(spec_batch * q_len)), "r"((int)(0)) : "memory");
                        #pragma unroll
                        for (int spec_nb = 0; spec_nb < 2; spec_nb++) {
                            int spec_block = spec_blocks - 1 - spec_nb;
                            if (spec_block >= 0) {
                                #pragma unroll
                                for (int spec_pg = 0; spec_pg < 8; spec_pg++) {
                                    int spec_page_idx = spec_block * 8 + spec_pg;
                                    if (spec_page_idx > spec_max_pg) {
                                        spec_page_idx = spec_max_pg;
                                    }
                                    int spec_page = page_table[spec_pt_base + spec_page_idx] * num_kv_heads + spec_head;
                                    #pragma unroll
                                    for (int spec_hg = 0; spec_hg < 2; spec_hg++) {
                                        asm volatile("cp.async.bulk.prefetch.tensor.4d.L2.global.tile [%0, {%1, %2, %3, %4}];" :: "l"((uint64_t)((&K))), "r"((int)(0)), "r"((int)(0)), "r"((int)(spec_hg)), "r"((int)(spec_page)) : "memory");
                                    }
                                }
                            }
                        }
                    }
                }
            }
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
            unsigned int chunk_6 = work_token_words[base_4 + 10];
            mbarrier_arrive(work_empty_addr + (work_stage_l) * 8);
            work_stage_l += 1;
            if (work_stage_l == 4) { work_stage_l = 0; _phase_work_full_4 ^= 1; }
            unsigned int valid_l = valid_5;
            int kind_l = (int)kind_5;
            int batch_idx_l = (int)batch_5;
            int kv_head_idx = (int)kv_head_5;
            int block_begin_l = (int)block_begin_6;
            int block_end_l = (int)block_end_5;
            #pragma unroll 1
            for (unsigned int _tile_iter_l = 0; _tile_iter_l < max_items; _tile_iter_l++) {
                if (valid_l == 0) {
                    break;
                }
                if (kind_l == 0) {
                    int cta_n_blocks = block_end_l - block_begin_l;
                    mbarrier_wait(q_empty_addr + (q_prod_stage) * 8, q_prod_phase);
                    if (_tile_iter_l == 0) {
                        if (lane == 0) {
                        }
                    }
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(q_full_addr + (q_prod_stage) * 8, 64 * HEAD_DIM * 2);
                        tma_4d_gmem2smem(smem_qt_addr + q_prod_stage * 16384, (&Q), 0, kv_head_idx * 8, batch_idx_l * q_len, 0, q_full_addr + (q_prod_stage) * 8);
                        int kv_stage = 0;
                        int kv_phase = 1;
                        int prefill = ((cta_n_blocks < 2) ? cta_n_blocks : 2);
                        #pragma unroll 1
                        for (int ni = 0; ni < prefill; ni++) {
                            int page_stage_unwrapped = page_cons_stage + (unsigned int)ni;
                            int page_stage = page_stage_unwrapped;
                            if (page_stage >= 6) {
                                page_stage = page_stage - 6;
                            }
                            int page_phase = page_cons_phase;
                            if (page_stage_unwrapped >= 6) {
                                page_phase = page_phase ^ 1;
                            }
                            int pg_base = page_stage * 8;
                            mbarrier_wait(page_offsets_full_addr + (page_stage) * 8, page_phase);
                            mbarrier_wait(kv_empty_addr + (kv_stage) * 8, kv_phase);
                            mbarrier_arrive_expect_tx(kv_full_addr + (kv_stage) * 8, 32768);
                            int ldst = smem_kv_addr + (unsigned int)(kv_stage * 32768);
                            int pg_k[8];
                            #pragma unroll
                            for (int pg_i = 0; pg_i < 8; pg_i++) {
                                pg_k[pg_i] = smem_page_offsets[pg_base + pg_i];
                            }
                            #pragma unroll
                            for (int pg_i_1 = 0; pg_i_1 < 8; pg_i_1++) {
                                int pg0 = pg_k[pg_i_1] * num_kv_heads + kv_head_idx;
                                #pragma unroll
                                for (int hg = 0; hg < 2; hg++) {
                                    int toff = hg * 16384 + pg_i_1 * 2048;
                                    tma_4d_gmem2smem(ldst + toff, (&K), 0, 0, hg, pg0, kv_full_addr + (kv_stage) * 8);
                                }
                            }
                            kv_stage += 1;
                            if (kv_stage == 4) { kv_stage = 0; kv_phase ^= 1; }
                        }
                        #pragma unroll 1
                        for (int ni_1 = 0; ni_1 < cta_n_blocks; ni_1++) {
                            int next_ni = ni_1 + 2;
                            if (next_ni < cta_n_blocks) {
                                int k_stage = next_ni % NUM_KV_STAGES;
                                int next_page_stage_unwrapped = page_cons_stage + 2;
                                int next_page_stage = next_page_stage_unwrapped;
                                if (next_page_stage >= 6) {
                                    next_page_stage = next_page_stage - 6;
                                }
                                int next_page_phase = page_cons_phase;
                                if (next_page_stage_unwrapped >= 6) {
                                    next_page_phase = next_page_phase ^ 1;
                                }
                                int npg_base = next_page_stage * 8;
                                mbarrier_wait(page_offsets_full_addr + (next_page_stage) * 8, next_page_phase);
                                mbarrier_wait(kv_empty_addr + (k_stage) * 8, 1);
                                mbarrier_arrive_expect_tx(kv_full_addr + (k_stage) * 8, 32768);
                                int kdst = smem_kv_addr + (unsigned int)(k_stage * 32768);
                                int pg_nk[8];
                                #pragma unroll
                                for (int pg_i_2 = 0; pg_i_2 < 8; pg_i_2++) {
                                    pg_nk[pg_i_2] = smem_page_offsets[npg_base + pg_i_2];
                                }
                                #pragma unroll
                                for (int pg_i_3 = 0; pg_i_3 < 8; pg_i_3++) {
                                    int npg0 = pg_nk[pg_i_3] * num_kv_heads + kv_head_idx;
                                    #pragma unroll
                                    for (int hg_1 = 0; hg_1 < 2; hg_1++) {
                                        int ntoff = hg_1 * 16384 + pg_i_3 * 2048;
                                        tma_4d_gmem2smem(kdst + ntoff, (&K), 0, 0, hg_1, npg0, kv_full_addr + (k_stage) * 8);
                                    }
                                }
                            }
                            int stage_1 = ni_1 % NUM_KV_STAGES;
                            int vpg_base = page_cons_stage * 8;
                            mbarrier_wait(kv_empty_addr + (stage_1) * 8, 0);
                            mbarrier_arrive_expect_tx(kv_full_addr + (stage_1) * 8, 32768);
                            int vdst = smem_kv_addr + (unsigned int)(stage_1 * 32768);
                            int pg_v[8];
                            #pragma unroll
                            for (int pg_i_4 = 0; pg_i_4 < 8; pg_i_4++) {
                                pg_v[pg_i_4] = smem_page_offsets[vpg_base + pg_i_4];
                            }
                            #pragma unroll
                            for (int pg_i_5 = 0; pg_i_5 < 8; pg_i_5++) {
                                int vpg0 = pg_v[pg_i_5] * num_kv_heads + kv_head_idx;
                                #pragma unroll
                                for (int hg_2 = 0; hg_2 < 2; hg_2++) {
                                    int vtoff = hg_2 * 16384 + pg_i_5 * 2048;
                                    tma_4d_gmem2smem(vdst + vtoff, (&V), 0, 0, hg_2, vpg0, kv_full_addr + (stage_1) * 8);
                                }
                            }
                            mbarrier_arrive(page_offsets_empty_addr + (page_cons_stage) * 8);
                            page_cons_stage += 1;
                            if (page_cons_stage == 6) { page_cons_stage = 0; page_cons_phase ^= 1; }
                        }
                    }
                    q_prod_stage += 1;
                    if (q_prod_stage == 2) { q_prod_stage = 0; q_prod_phase ^= 1; }
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
        }
    }

    // Cleanup

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
    }
}

} // extern "C"
