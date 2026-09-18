// Copyright (c) 2026 FlashInfer contributors.
// SPDX-License-Identifier: Apache-2.0

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
template <int N>
struct __align__(128) CakeTensorMapPack { CakeTensorMap maps[N]; };

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
#define TMEM_NCOLS 512
#define TMEM_TMEM_S0_OFFSET 0
#define TMEM_TMEM_O_HI_OFFSET 256
#define TMEM_TMEM_O_LO_OFFSET 384
#define NUM_RAW_KV_PIPE_STAGES 4
#define NUM_TRANSFORMED_KV_PIPE_STAGES 2
#define NUM_SM_PIPE_STAGES 2
#define NUM_WORK_PIPE_STAGES 2
#define NUM_THROTTLE_PIPE_STAGES 2
#define NUM_PAGE_PIPE_STAGES 6
#define NUM_CORR_PIPE_STAGES 2
#define NUM_P_PIPE_STAGES 2
#define SMEM_SMEM_CORR_OFF 227968
#define SMEM_SMEM_CORR_STAGE_BYTES 1536
#define SMEM_SMEM_CORR_STRIDE 1536
#define SMEM_SMEM_EXCH_OFF 1536
#define SMEM_SMEM_EXCH_STAGE_BYTES 384
#define SMEM_SMEM_EXCH_STRIDE 384
#define SMEM_SMEM_EXCH_U32_OFF 1536
#define SMEM_SMEM_EXCH_U32_STAGE_BYTES 384
#define SMEM_SMEM_EXCH_U32_STRIDE 384
#define SMEM_SMEM_QT_HI_OFF 2048
#define SMEM_SMEM_QT_HI_STAGE_BYTES 32768
#define SMEM_SMEM_QT_HI_STRIDE 32768
#define SMEM_SMEM_QT_LO_OFF 34816
#define SMEM_SMEM_QT_LO_STAGE_BYTES 32768
#define SMEM_SMEM_QT_LO_STRIDE 32768
#define SMEM_SMEM_KV_FP8_OFF 133120
#define SMEM_SMEM_KV_FP8_STAGE_BYTES 16384
#define SMEM_SMEM_KV_FP8_STRIDE 16384
#define SMEM_SMEM_KV_OFF 67584
#define SMEM_SMEM_KV_STAGE_BYTES 32768
#define SMEM_SMEM_KV_STRIDE 32768
#define SMEM_SMEM_V_OFF 67584
#define SMEM_SMEM_V_STAGE_BYTES 32768
#define SMEM_SMEM_V_STRIDE 32768
#define SMEM_SMEM_P_OFF 149504
#define SMEM_SMEM_P_STAGE_BYTES 24576
#define SMEM_SMEM_P_STRIDE 24576
#define SMEM_WORK_RESPONSE_VIEW_OFF 1344
#define SMEM_WORK_RESPONSE_VIEW_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_VIEW_STRIDE 16
#define SMEM_SPLIT_REDUCE_FLAG_OFF 198656
#define SMEM_SPLIT_REDUCE_FLAG_STAGE_BYTES 4
#define SMEM_SPLIT_REDUCE_FLAG_STRIDE 4
#define SMEM_SPLIT_WEIGHTS_OFF 198784
#define SMEM_SPLIT_WEIGHTS_STAGE_BYTES 29184
#define SMEM_SPLIT_WEIGHTS_STRIDE 29184
#define SMEM_SMEM_PAGE_OFFSETS_OFF 229504
#define SMEM_SMEM_PAGE_OFFSETS_STAGE_BYTES 768
#define SMEM_SMEM_PAGE_OFFSETS_STRIDE 768
#define SMEM_SMEM_Q6_SCALES_OFF 230272
#define SMEM_SMEM_Q6_SCALES_STAGE_BYTES 1024
#define SMEM_SMEM_Q6_SCALES_STRIDE 1024
#define SMEM_SMEM_Q6_FINAL_OFF 231296
#define SMEM_SMEM_Q6_FINAL_STAGE_BYTES 1024
#define SMEM_SMEM_Q6_FINAL_STRIDE 1024
#define SMEM_TOTAL 232320
#define THREADS 512
#define BLOCK_N 128
#define HEAD_DIM 256
#define HEAD_DIM_HALF 128
#define TILE_Q 96
#define Q_GROUPS_PER_KV 1
#define PAGE_SIZE 64
#define NUM_RAW_KV_STAGES 4
#define NUM_TRANSFORMED_KV_STAGES 2
#define Q_LEN 6
#define UNIFORM_KV_LEN 0
#define USE_REQUEST_ORDER 1
#define USE_SCALE_POINTERS 1
#define NUM_SPLIT 76
#define USE_SEGMENTED_CLC 0
#define USE_HIGH_BATCH_TWO_WAVE 0
#define USE_TWO_CTA_REDUCER 0
#define USE_MMA_LOOP_PEEL 1
#define USE_PAGE_OFFSET_CPASYNC 0
#define USE_LEGACY_PAGE_VEC4 0
#define WRITE_LSE 0

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
           "r"(i_desc), "r"(enable_input_d));
}


__device__ __forceinline__ uint64_t desc_encode(uint64_t x) {
    return (x & 0x3FFFFULL) >> 4ULL;
}


__device__ __forceinline__ void mma_ss_step(
    int a_lo, int b_lo, int taddr, uint32_t i_desc, int enable_d,
    uint32_t a_dhi, uint32_t b_dhi) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader, p;\n\t"
        ".reg .b32 adhi, bdhi;\n\t"
        ".reg .b64 da, db;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "mov.b32 adhi, %5;\n\t"
        "mov.b32 bdhi, %6;\n\t"
        "mov.b64 da, {%0, adhi};\n\t"
        "mov.b64 db, {%1, bdhi};\n\t"
        "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, %3, p;\n\t"
        "}\n"
        :: "r"(a_lo), "r"(b_lo), "r"(taddr), "r"(i_desc), "r"(enable_d), "r"(a_dhi), "r"(b_dhi));
}


__device__ __forceinline__ void mma_ts_step(
    int taddr_out, int taddr_a, int b_lo, uint32_t b_dhi,
    uint32_t i_desc, int enable_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader, p;\n\t"
        ".reg .b32 dhi;\n\t"
        ".reg .b64 db;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "setp.ne.b32 p, %5, 0;\n\t"
        "mov.b32 dhi, %3;\n\t"
        "mov.b64 db, {%2, dhi};\n\t"
        "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], db, %4, p;\n\t"
        "}\n"
        :: "r"(taddr_out), "r"(taddr_a), "r"(b_lo), "r"(b_dhi),
           "r"(i_desc), "r"(enable_d));
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


__device__ __forceinline__ void tmem_st_x16(int tmem_addr, uint32_t* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x16.b32"
        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8,"
        "  %9, %10, %11, %12, %13, %14, %15, %16};"
        :: "r"(tmem_addr),
           "r"(src[0]),  "r"(src[1]),  "r"(src[2]),  "r"(src[3]),
           "r"(src[4]),  "r"(src[5]),  "r"(src[6]),  "r"(src[7]),
           "r"(src[8]),  "r"(src[9]),  "r"(src[10]), "r"(src[11]),
           "r"(src[12]), "r"(src[13]), "r"(src[14]), "r"(src[15]));
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


__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = max_noftz(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    return val;
}


__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
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


__device__ __forceinline__ float2 ex2_emulation_f32x2_value(float2 value) {
    const float c0 = 1.0f, c1 = 0.695146143436431884765625f;
    const float c2 = 0.227564394474029541015625f, c3 = 0.077119089663028717041015625f;
    const float magic = 12582912.0f;
    float x0 = max_noftz(value.x, -127.0f), x1 = max_noftz(value.y, -127.0f);
    float2 xc2 = make_float2(x0, x1), magic2 = make_float2(magic, magic);
    float2 xr2;
    asm("add.rm.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xr2)
        : "l"(*(unsigned long long*)&xc2), "l"(*(unsigned long long*)&magic2));
    float2 c3_2 = make_float2(c3, c3), c2_2 = make_float2(c2, c2);
    float2 c1_2 = make_float2(c1, c1), c0_2 = make_float2(c0, c0);
    float2 xrb2, xfrac2;
    asm("sub.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xrb2)
        : "l"(*(unsigned long long*)&xr2), "l"(*(unsigned long long*)&magic2));
    asm("sub.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xfrac2)
        : "l"(*(unsigned long long*)&xc2), "l"(*(unsigned long long*)&xrb2));
    float2 poly2;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;" : "=l"(*(unsigned long long*)&poly2)
        : "l"(*(unsigned long long*)&c3_2), "l"(*(unsigned long long*)&xfrac2), "l"(*(unsigned long long*)&c2_2));
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;" : "=l"(*(unsigned long long*)&poly2)
        : "l"(*(unsigned long long*)&poly2), "l"(*(unsigned long long*)&xfrac2), "l"(*(unsigned long long*)&c1_2));
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;" : "=l"(*(unsigned long long*)&poly2)
        : "l"(*(unsigned long long*)&poly2), "l"(*(unsigned long long*)&xfrac2), "l"(*(unsigned long long*)&c0_2));
    int x0r_i, x1r_i, p0_i, p1_i;
    asm("mov.b64 {%0, %1}, %2;" : "=r"(x0r_i), "=r"(x1r_i) : "l"(*(unsigned long long*)&xr2));
    asm("mov.b64 {%0, %1}, %2;" : "=r"(p0_i), "=r"(p1_i) : "l"(*(unsigned long long*)&poly2));
    float r0, r1;
    asm("mov.b32 %0, %1;" : "=f"(r0) : "r"((x0r_i << 23) + p0_i));
    asm("mov.b32 %0, %1;" : "=f"(r1) : "r"((x1r_i << 23) + p1_i));
    return make_float2(r0, r1);
}

__device__ __forceinline__ void ex2_emulation_f32x2(float* x0_ptr, float* x1_ptr) {
    float2 result = ex2_emulation_f32x2_value(make_float2(*x0_ptr, *x1_ptr));
    *x0_ptr = result.x; *x1_ptr = result.y;
}

__device__ __forceinline__ void softmax_frag_exp2_cast(
    float* sv, uint32_t* pv, int use_emu)
{
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        if (use_emu && j >= 12)
            ex2_emulation_f32x2(&sv[j*2], &sv[j*2+1]);
        else {
            sv[j*2]   = approx_exp2(sv[j*2]);
            sv[j*2+1] = approx_exp2(sv[j*2+1]);
        }
    }
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        __nv_bfloat162 bf = __float22bfloat162_rn({sv[j*2], sv[j*2+1]});
        pv[j] = reinterpret_cast<uint32_t&>(bf);
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
    asm("mul.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void add_f32x2_inplace(float2* a, float2 b) {
    asm("add.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void sub_f32x2_inplace(float2* a, float2 b) {
    asm("sub.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
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

__device__ __forceinline__ void fma_scale_x32(
    float* sv, const float2* scale2, const float2* neg_max2)
{
    float2* sv_2 = reinterpret_cast<float2*>(sv);
    #pragma unroll
    for (int j = 0; j < 16; j++)
        fma_f32x2_inplace(&sv_2[j], *scale2, *neg_max2);
}

__device__ __forceinline__ float2 fma_f32x2(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rn.ftz.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

// ex2_emulation_f32x2 defined in softmax_frag_exp2_cast helper (or standalone)

__device__ __forceinline__ float2 add_f32x2_rn_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 add_f32x2_rn_ftz(float2 a, float2 b) {
    float2 r;
    asm("add.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 add_f32x2_rz_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.rz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 add_f32x2_rz_ftz(float2 a, float2 b) {
    float2 r;
    asm("add.rz.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 add_f32x2_rm_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.rm.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 add_f32x2_rm_ftz(float2 a, float2 b) {
    float2 r;
    asm("add.rm.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 add_f32x2_rp_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.rp.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 add_f32x2_rp_ftz(float2 a, float2 b) {
    float2 r;
    asm("add.rp.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_rn_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_rn_ftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_rz_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_rz_ftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rz.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_rm_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rm.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_rm_ftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rm.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_rp_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rp.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 mul_f32x2_rp_ftz(float2 a, float2 b) {
    float2 r;
    asm("mul.rp.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_rn_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2_rn_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rn.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_rn_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2_rn_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rn.ftz.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_rz_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2_rz_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rz.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_rz_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rz.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2_rz_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rz.ftz.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_rm_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rm.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2_rm_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rm.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_rm_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rm.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2_rm_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rm.ftz.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_rp_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rp.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2_rp_noftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rp.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_f32x2_rp_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm("fma.rp.ftz.f32x2 %0, %1, %2, %3;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
}

__device__ __forceinline__ float2 fma_sub_f32x2_rp_ftz(float2 a, float2 b, float2 c) {
    float2 r;
    asm volatile("{\n\t"
        ".reg .f32 _c0, _c1;\n\t"
        ".reg .b64 _neg_c;\n\t"
        "mov.b64 {_c0, _c1}, %3;\n\t"
        "neg.f32 _c0, _c0;\n\t"
        "neg.f32 _c1, _c1;\n\t"
        "mov.b64 _neg_c, {_c0, _c1};\n\t"
        "fma.rp.ftz.f32x2 %0, %1, %2, _neg_c;\n\t"
        "}\n"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(const unsigned long long*)&a),
          "l"(*(const unsigned long long*)&b),
          "l"(*(const unsigned long long*)&c));
    return r;
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


__device__ __forceinline__ void tma_3d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
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

__global__ __launch_bounds__(512, 1) void
kernel_cake_fmha_request_ordered_paged_decode_32q2_00be11bfbf2ea504e0aa(CakeTensorMap const* Qt, CakeTensorMap const* K, CakeTensorMap const* V, __nv_bfloat16* __restrict__ partial_O, float* __restrict__ partial_LSE, unsigned int* __restrict__ split_completion, __nv_bfloat16* __restrict__ O, float* __restrict__ LSE, int* __restrict__ page_table, int* __restrict__ seq_lens_kv, int* __restrict__ request_order, int max_pages_per_seq, int page_table_v_offset, float softmax_scale_log2, float output_scale, float* __restrict__ bmm1_scale_ptr, float* __restrict__ bmm2_scale_ptr, int bmm1_is_log2, int num_q_heads, int num_kv_heads, int batch_size, unsigned int total_tiles)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define q_full_addr (mbar_base + 0)
    #define q_empty_addr (mbar_base + 8)
    #define raw_kv_full_addr (mbar_base + 16)
    #define raw_kv_empty_addr (mbar_base + 48)
    #define kv_full_addr (mbar_base + 80)
    #define kv_empty_addr (mbar_base + 96)
    #define s_full_0_addr (mbar_base + 112)
    #define s_empty_0_addr (mbar_base + 128)
    #define p_full_0_addr (mbar_base + 144)
    #define corr_scale_0_addr (mbar_base + 160)
    #define corr_empty_0_addr (mbar_base + 176)
    #define o_full_addr (mbar_base + 192)
    #define o_empty_addr (mbar_base + 200)
    #define tmem_dealloc_addr (mbar_base + 208)
    #define work_full_addr (mbar_base + 216)
    #define work_empty_addr (mbar_base + 232)
    #define throttle_full_addr (mbar_base + 248)
    #define throttle_empty_addr (mbar_base + 264)
    #define page_offsets_full_addr (mbar_base + 280)
    #define page_offsets_empty_addr (mbar_base + 328)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 1;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 1;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(Qt)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(K)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(V)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    float* smem_corr = reinterpret_cast<float*>(smem_raw + 227968);
    const int smem_corr_addr = smem + 227968;
    float* smem_exch = reinterpret_cast<float*>(smem_raw + 1536);
    const int smem_exch_addr = smem + 1536;
    unsigned int* smem_exch_u32 = reinterpret_cast<unsigned int*>(smem_raw + 1536);
    const int smem_exch_u32_addr = smem + 1536;
    __nv_bfloat16* smem_qt_hi = reinterpret_cast<__nv_bfloat16*>(smem_raw + 2048);
    const int smem_qt_hi_addr = smem + 2048;
    __nv_bfloat16* smem_qt_lo = reinterpret_cast<__nv_bfloat16*>(smem_raw + 34816);
    const int smem_qt_lo_addr = smem + 34816;
    uint8_t* smem_kv_fp8 = reinterpret_cast<uint8_t*>(smem_raw + 133120);
    const int smem_kv_fp8_addr = smem + 133120;
    __nv_bfloat16* smem_kv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 67584);
    const int smem_kv_addr = smem + 67584;
    __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 67584);
    const int smem_v_addr = smem + 67584;
    __nv_bfloat16* smem_p = reinterpret_cast<__nv_bfloat16*>(smem_raw + 149504);
    const int smem_p_addr = smem + 149504;
    unsigned int* work_response_view = reinterpret_cast<unsigned int*>(smem_raw + 1344);
    const int work_response_view_addr = smem + 1344;
    int* split_reduce_flag = reinterpret_cast<int*>(smem_raw + 198656);
    const int split_reduce_flag_addr = smem + 198656;
    float* split_weights = reinterpret_cast<float*>(smem_raw + 198784);
    const int split_weights_addr = smem + 198784;
    int* smem_page_offsets = reinterpret_cast<int*>(smem_raw + 229504);
    const int smem_page_offsets_addr = smem + 229504;
    float* smem_q6_scales = reinterpret_cast<float*>(smem_raw + 230272);
    const int smem_q6_scales_addr = smem + 230272;
    float* smem_q6_final = reinterpret_cast<float*>(smem_raw + 231296);
    const int smem_q6_final_addr = smem + 231296;
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(Qt)) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(K)) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(V)) : "memory");

    // Mbarrier init (20 pipeline groups, 0 ordered-sequence groups, 47 barriers)
    // Mbarriers at smem_raw[0..376)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // q_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // raw_kv_full: 4 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // raw_kv_empty: 4 barriers, init_count=4
            mbarrier_init(smem + 48, 4);
            mbarrier_init(smem + 56, 4);
            mbarrier_init(smem + 64, 4);
            mbarrier_init(smem + 72, 4);
            // kv_full: 2 barriers, init_count=4
            mbarrier_init(smem + 80, 4);
            mbarrier_init(smem + 88, 4);
            // kv_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            // --- pipeline 'sm_pipe' ---
            // s_full_0: 2 barriers, init_count=1
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // s_empty_0: 2 barriers, init_count=128
            mbarrier_init(smem + 128, 128);
            mbarrier_init(smem + 136, 128);
            // --- pipeline 'p_pipe' ---
            // p_full_0: 2 barriers, init_count=256
            mbarrier_init(smem + 144, 256);
            mbarrier_init(smem + 152, 256);
            // --- pipeline 'corr_pipe' ---
            // corr_scale_0: 2 barriers, init_count=128
            mbarrier_init(smem + 160, 128);
            mbarrier_init(smem + 168, 128);
            // corr_empty_0: 2 barriers, init_count=128
            mbarrier_init(smem + 176, 128);
            mbarrier_init(smem + 184, 128);
            // o_full: 1 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            // o_empty: 1 barriers, init_count=128
            mbarrier_init(smem + 200, 128);
            // tmem_dealloc: 1 barriers, init_count=128
            mbarrier_init(smem + 208, 128);
            // --- pipeline 'work_pipe' ---
            // work_full: 2 barriers, init_count=1
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            // work_empty: 2 barriers, init_count=512
            mbarrier_init(smem + 232, 512);
            mbarrier_init(smem + 240, 512);
            // --- pipeline 'throttle_pipe' ---
            // throttle_full: 2 barriers, init_count=32
            mbarrier_init(smem + 248, 32);
            mbarrier_init(smem + 256, 32);
            // throttle_empty: 2 barriers, init_count=32
            mbarrier_init(smem + 264, 32);
            mbarrier_init(smem + 272, 32);
            // --- pipeline 'page_pipe' ---
            // page_offsets_full: 6 barriers, init_count=32
            mbarrier_init(smem + 280, 32);
            mbarrier_init(smem + 288, 32);
            mbarrier_init(smem + 296, 32);
            mbarrier_init(smem + 304, 32);
            mbarrier_init(smem + 312, 32);
            mbarrier_init(smem + 320, 32);
            // page_offsets_empty: 6 barriers, init_count=1
            mbarrier_init(smem + 328, 1);
            mbarrier_init(smem + 336, 1);
            mbarrier_init(smem + 344, 1);
            mbarrier_init(smem + 352, 1);
            mbarrier_init(smem + 360, 1);
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
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_s0 = taddr;
    const int tmem_tmem_o_hi = taddr + 256;
    const int tmem_tmem_o_lo = taddr + 384;

    // ---- Role: softmax ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 248;");
        { // softmax_main
            const int tmem_row_base_v = warp * 32;
            int my_tmem_s = taddr;
            int my_tmem_stats = taddr + (unsigned int)(tmem_row_base_v << 16);
            const int warp_in_wg = warp;
            const int wg_tid = (unsigned int)(warp_in_wg * 32) + lane;
            int col_pair = wg_tid % 4;
            int col_pair_base = col_pair * 2;
            unsigned int work_stage_s = 0;
            int sm_stage = 0;
            int sm_phase = 0;
            int corr_prod_stage = 0;
            int corr_prod_phase = 1;
            float bmm1_scale_log2_s = softmax_scale_log2;
            float bmm1_scale_log2_p = softmax_scale_log2;
            int batch_idx = 0;
            int q_row_idx = 0;
            int kv_head_idx = 0;
            int split_idx = 0;
            int part_count = NUM_SPLIT;
            int bundle_idx = 0;
            int bundle_item_idx = 0;
            {
                q_row_idx = blockIdx.x / NUM_SPLIT * 6;
                split_idx = blockIdx.x % NUM_SPLIT;
                kv_head_idx = 0;
                kv_head_idx = blockIdx.y;
                int schedule_batch_idx = blockIdx.z;
                batch_idx = schedule_batch_idx;
                {
                    batch_idx = request_order[schedule_batch_idx];
                }
            }
            int visible_keys = UNIFORM_KV_LEN - Q_LEN + q_row_idx + 1;
            {
                visible_keys = seq_lens_kv[batch_idx] - Q_LEN + q_row_idx + 1;
            }
            visible_keys = visible_keys + 6 - 1;
            if (visible_keys < 0) {
                visible_keys = 0;
            }
            int peer_visible = visible_keys;
            int _min_4 = (((peer_visible + BLOCK_N - 1) / BLOCK_N) < (NUM_SPLIT) ? ((peer_visible + BLOCK_N - 1) / BLOCK_N) : (NUM_SPLIT));
            part_count = _min_4;
            int batch_idx_s = batch_idx;
            int q_row_idx_s = q_row_idx;
            int kv_head_idx_s = kv_head_idx;
            int split_idx_s = split_idx;
            int part_count_s = part_count;
            int bundle_idx_s = bundle_idx;
            int bundle_item_idx_s = bundle_item_idx;
            unsigned int peer_tiles = 0;
            if (split_idx_s < part_count_s) {
                peer_tiles = total_tiles;
            }
            #pragma unroll 1
            for (unsigned int _tile_iter_s = 0; _tile_iter_s < peer_tiles; _tile_iter_s++) {
                int visible_keys_0 = UNIFORM_KV_LEN - Q_LEN + q_row_idx_s + 1;
                {
                    visible_keys_0 = seq_lens_kv[batch_idx_s] - Q_LEN + q_row_idx_s + 1;
                }
                visible_keys_0 = visible_keys_0 + 6 - 1;
                if (visible_keys_0 < 0) {
                    visible_keys_0 = 0;
                }
                int seqlen_kv_s = visible_keys_0;
                int peer_denominator = part_count_s * BLOCK_N;
                int peer_blocks = (seqlen_kv_s + peer_denominator - 1) / peer_denominator;
                float row_max_qm = -CAKE_INF;
                float row_sum_qm = 0.0f;
                asm volatile("griddepcontrol.wait;" ::: "memory");
                {
                    bmm1_scale_log2_s = bmm1_scale_ptr[0];
                    if (bmm1_is_log2 == 0) {
                        bmm1_scale_log2_s = bmm1_scale_log2_s * 1.4426950408889634f;
                    }
                }
                asm volatile("griddepcontrol.wait;" ::: "memory");
                {
                    bmm1_scale_log2_p = bmm1_scale_ptr[0];
                    if (bmm1_is_log2 == 0) {
                        bmm1_scale_log2_p = bmm1_scale_log2_p * 1.4426950408889634f;
                    }
                }
                #pragma unroll 1
                for (int n = 0; n < peer_blocks; n++) {
                    mbarrier_wait(s_full_0_addr + (sm_stage) * 8, sm_phase);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int score_base_qm = my_tmem_s + sm_stage * 128 + (tmem_row_base_v << 16);
                    float _tmem_load_0[128];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                        : "r"(score_base_qm));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_0[32]), "=f"(_tmem_load_0[33]), "=f"(_tmem_load_0[34]), "=f"(_tmem_load_0[35]), "=f"(_tmem_load_0[36]), "=f"(_tmem_load_0[37]), "=f"(_tmem_load_0[38]), "=f"(_tmem_load_0[39]), "=f"(_tmem_load_0[40]), "=f"(_tmem_load_0[41]), "=f"(_tmem_load_0[42]), "=f"(_tmem_load_0[43]), "=f"(_tmem_load_0[44]), "=f"(_tmem_load_0[45]), "=f"(_tmem_load_0[46]), "=f"(_tmem_load_0[47]), "=f"(_tmem_load_0[48]), "=f"(_tmem_load_0[49]), "=f"(_tmem_load_0[50]), "=f"(_tmem_load_0[51]), "=f"(_tmem_load_0[52]), "=f"(_tmem_load_0[53]), "=f"(_tmem_load_0[54]), "=f"(_tmem_load_0[55]), "=f"(_tmem_load_0[56]), "=f"(_tmem_load_0[57]), "=f"(_tmem_load_0[58]), "=f"(_tmem_load_0[59]), "=f"(_tmem_load_0[60]), "=f"(_tmem_load_0[61]), "=f"(_tmem_load_0[62]), "=f"(_tmem_load_0[63])
                        : "r"(score_base_qm + 32));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_0[64]), "=f"(_tmem_load_0[65]), "=f"(_tmem_load_0[66]), "=f"(_tmem_load_0[67]), "=f"(_tmem_load_0[68]), "=f"(_tmem_load_0[69]), "=f"(_tmem_load_0[70]), "=f"(_tmem_load_0[71]), "=f"(_tmem_load_0[72]), "=f"(_tmem_load_0[73]), "=f"(_tmem_load_0[74]), "=f"(_tmem_load_0[75]), "=f"(_tmem_load_0[76]), "=f"(_tmem_load_0[77]), "=f"(_tmem_load_0[78]), "=f"(_tmem_load_0[79]), "=f"(_tmem_load_0[80]), "=f"(_tmem_load_0[81]), "=f"(_tmem_load_0[82]), "=f"(_tmem_load_0[83]), "=f"(_tmem_load_0[84]), "=f"(_tmem_load_0[85]), "=f"(_tmem_load_0[86]), "=f"(_tmem_load_0[87]), "=f"(_tmem_load_0[88]), "=f"(_tmem_load_0[89]), "=f"(_tmem_load_0[90]), "=f"(_tmem_load_0[91]), "=f"(_tmem_load_0[92]), "=f"(_tmem_load_0[93]), "=f"(_tmem_load_0[94]), "=f"(_tmem_load_0[95])
                        : "r"(score_base_qm + 64));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_0[96]), "=f"(_tmem_load_0[97]), "=f"(_tmem_load_0[98]), "=f"(_tmem_load_0[99]), "=f"(_tmem_load_0[100]), "=f"(_tmem_load_0[101]), "=f"(_tmem_load_0[102]), "=f"(_tmem_load_0[103]), "=f"(_tmem_load_0[104]), "=f"(_tmem_load_0[105]), "=f"(_tmem_load_0[106]), "=f"(_tmem_load_0[107]), "=f"(_tmem_load_0[108]), "=f"(_tmem_load_0[109]), "=f"(_tmem_load_0[110]), "=f"(_tmem_load_0[111]), "=f"(_tmem_load_0[112]), "=f"(_tmem_load_0[113]), "=f"(_tmem_load_0[114]), "=f"(_tmem_load_0[115]), "=f"(_tmem_load_0[116]), "=f"(_tmem_load_0[117]), "=f"(_tmem_load_0[118]), "=f"(_tmem_load_0[119]), "=f"(_tmem_load_0[120]), "=f"(_tmem_load_0[121]), "=f"(_tmem_load_0[122]), "=f"(_tmem_load_0[123]), "=f"(_tmem_load_0[124]), "=f"(_tmem_load_0[125]), "=f"(_tmem_load_0[126]), "=f"(_tmem_load_0[127])
                        : "r"(score_base_qm + 96));
                    int visible_qm = seqlen_kv_s - 5 + wg_tid / 16;
                    int _min_5 = ((128) < (visible_qm - (split_idx_s * peer_blocks + n) * 128) ? (128) : (visible_qm - (split_idx_s * peer_blocks + n) * 128));
                    int _max_0 = ((0) > (_min_5) ? (0) : (_min_5));
                    int valid_qm = _max_0;
                    if (wg_tid >= 96) {
                        valid_qm = 0;
                    }
                    uint32_t _slice_lo_mask_0;
                    {
                        int _lim_0 = valid_qm;
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
                        int _lim_1 = valid_qm - 32;
                        if (_lim_1 <= 0) { _slice_lo_mask_1 = 0u; }
                        else if (_lim_1 >= 32) { _slice_lo_mask_1 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_slice_lo_mask_1) : "r"(_lim_1));
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
                        int _lim_2 = valid_qm - 64;
                        if (_lim_2 <= 0) { _slice_lo_mask_2 = 0u; }
                        else if (_lim_2 >= 32) { _slice_lo_mask_2 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_slice_lo_mask_2) : "r"(_lim_2));
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
                        int _lim_3 = valid_qm - 96;
                        if (_lim_3 <= 0) { _slice_lo_mask_3 = 0u; }
                        else if (_lim_3 >= 32) { _slice_lo_mask_3 = 0xFFFFFFFFu; }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_slice_lo_mask_3) : "r"(_lim_3));
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
                    float2 _reg_reduce_max2_4 = {-CAKE_INF, -CAKE_INF};
                    row_max_x32_accum(&_tmem_load_0[0], _reg_reduce_max2_4);
                    row_max_x32_accum(&_tmem_load_0[32], _reg_reduce_max2_4);
                    row_max_x32_accum(&_tmem_load_0[64], _reg_reduce_max2_4);
                    row_max_x32_accum(&_tmem_load_0[96], _reg_reduce_max2_4);
                    float _tmem_load_0_max = row_max_reduce(_reg_reduce_max2_4);
                    float _max_1 = max_noftz(row_max_qm, _tmem_load_0_max);
                    float new_max_qm = _max_1;
                    float delta_qm = bmm1_scale_log2_s * (row_max_qm - new_max_qm);
                    float _exp2_0 = approx_exp2(delta_qm);
                    float alpha_qm = ((row_max_qm > -CAKE_INF) ? _exp2_0 : 1.0f);
                    mbarrier_wait(corr_empty_0_addr + (corr_prod_stage) * 8, corr_prod_phase);
                    smem_q6_scales[corr_prod_stage * 128 + wg_tid] = alpha_qm;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(corr_scale_0_addr + (corr_prod_stage) * 8);
                    corr_prod_stage += 1;
                    if (corr_prod_stage == 2) { corr_prod_stage = 0; corr_prod_phase ^= 1; }
                    float safe_max_qm = ((new_max_qm == -CAKE_INF) ? 0.0f : new_max_qm);
                    float max_scaled_qm = safe_max_qm * bmm1_scale_log2_p;
                    const float2 _fma_b2_5 = {bmm1_scale_log2_p, bmm1_scale_log2_p};
                    const float2 _fma_c2_6 = {-max_scaled_qm, -max_scaled_qm};
                    #pragma unroll
                    for (int _lf = 0; _lf < 64; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_0)[_lf], _fma_b2_5, _fma_c2_6);
                    #pragma unroll
                    for (int _le = 0; _le < 128; _le++) {
                        _tmem_load_0[_le] = approx_exp2(_tmem_load_0[_le]);
                    }
                    float2 _reg_reduce_sum2_7 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&_tmem_load_0[0], &_reg_reduce_sum2_7);
                    softmax_block_sum(&_tmem_load_0[32], &_reg_reduce_sum2_7);
                    softmax_block_sum(&_tmem_load_0[64], &_reg_reduce_sum2_7);
                    softmax_block_sum(&_tmem_load_0[96], &_reg_reduce_sum2_7);
                    float _tmem_load_0_sum = _reg_reduce_sum2_7.x + _reg_reduce_sum2_7.y;
                    float block_sum_qm = _tmem_load_0_sum;
                    float _fma_0 = __fmaf_rn(row_sum_qm, alpha_qm, block_sum_qm);
                    row_sum_qm = _fma_0;
                    row_max_qm = new_max_qm;
                    int p_base_qm = score_base_qm + 32;
                    #pragma unroll
                    for (int p_chunk_qm = 0; p_chunk_qm < 4; p_chunk_qm++) {
                        uint32_t _tmem_load_0_bf16[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2((_tmem_load_0 + p_chunk_qm * 32)[_lp*2 + 0], (_tmem_load_0 + p_chunk_qm * 32)[_lp*2+1 + 0]));
                            _tmem_load_0_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x16.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"(p_base_qm + p_chunk_qm * 16), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_0_bf16[15])));
                    }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    mbarrier_arrive(p_full_0_addr + (sm_stage) * 8);
                    mbarrier_arrive(s_empty_0_addr + (sm_stage) * 8);
                    sm_stage += 1;
                    if (sm_stage == 2) { sm_stage = 0; sm_phase ^= 1; }
                }
                mbarrier_wait(corr_empty_0_addr + (corr_prod_stage) * 8, corr_prod_phase);
                smem_q6_final[wg_tid * 2] = row_sum_qm;
                smem_q6_final[wg_tid * 2 + 1] = row_max_qm;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(corr_scale_0_addr + (corr_prod_stage) * 8);
                corr_prod_stage += 1;
                if (corr_prod_stage == 2) { corr_prod_stage = 0; corr_prod_phase ^= 1; }
            }
        }
    }
    // ---- Role: correction ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 120;");
        { // correction_main
            const int tmem_row_base_v_1 = warp % 4 * 32;
            const int corr_row = tmem_row_base_v_1 << 16;
            const int warp_in_wg_c = warp % 4;
            const int corr_tid = (unsigned int)(warp_in_wg_c * 32) + lane;
            const int col_pair_c = corr_tid % 4;
            const int col_pair_base_c = col_pair_c * 2;
            unsigned int work_stage_c = 0;
            int corr_cons_stage = 0;
            int corr_cons_phase = 0;
            int p_stage_c = 0;
            int d_idx = warp % 4 * 32 + lane;
            int group_ratio_rt = num_q_heads / (num_kv_heads * Q_GROUPS_PER_KV);
            group_ratio_rt = 6 * group_ratio_rt;
            float bmm1_scale_log2_c = softmax_scale_log2;
            float bmm2_scale_c = output_scale;
            int batch_idx_1 = 0;
            int q_row_idx_1 = 0;
            int kv_head_idx_1 = 0;
            int split_idx_1 = 0;
            int part_count_1 = NUM_SPLIT;
            int bundle_idx_1 = 0;
            int bundle_item_idx_1 = 0;
            {
                q_row_idx_1 = blockIdx.x / NUM_SPLIT * 6;
                split_idx_1 = blockIdx.x % NUM_SPLIT;
                kv_head_idx_1 = 0;
                kv_head_idx_1 = blockIdx.y;
                int schedule_batch_idx_1 = blockIdx.z;
                batch_idx_1 = schedule_batch_idx_1;
                {
                    batch_idx_1 = request_order[schedule_batch_idx_1];
                }
            }
            int visible_keys_1 = UNIFORM_KV_LEN - Q_LEN + q_row_idx_1 + 1;
            {
                visible_keys_1 = seq_lens_kv[batch_idx_1] - Q_LEN + q_row_idx_1 + 1;
            }
            visible_keys_1 = visible_keys_1 + 6 - 1;
            if (visible_keys_1 < 0) {
                visible_keys_1 = 0;
            }
            int peer_visible_1 = visible_keys_1;
            int _min_6 = (((peer_visible_1 + BLOCK_N - 1) / BLOCK_N) < (NUM_SPLIT) ? ((peer_visible_1 + BLOCK_N - 1) / BLOCK_N) : (NUM_SPLIT));
            part_count_1 = _min_6;
            int batch_idx_c = batch_idx_1;
            int q_row_idx_c = q_row_idx_1;
            int kv_head_idx_c = kv_head_idx_1;
            int split_idx_c = split_idx_1;
            int part_count_c = part_count_1;
            int bundle_idx_c = bundle_idx_1;
            int bundle_item_idx_c = bundle_item_idx_1;
            unsigned int peer_tiles_1 = 0;
            if (split_idx_c < part_count_c) {
                peer_tiles_1 = total_tiles;
            }
            unsigned int _phase_o_full_0 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_c = 0; _tile_iter_c < peer_tiles_1; _tile_iter_c++) {
                int visible_keys_0_1 = UNIFORM_KV_LEN - Q_LEN + q_row_idx_c + 1;
                {
                    visible_keys_0_1 = seq_lens_kv[batch_idx_c] - Q_LEN + q_row_idx_c + 1;
                }
                visible_keys_0_1 = visible_keys_0_1 + 6 - 1;
                if (visible_keys_0_1 < 0) {
                    visible_keys_0_1 = 0;
                }
                int seqlen_kv_c = visible_keys_0_1;
                int peer_denominator_1 = part_count_c * BLOCK_N;
                int peer_blocks_1 = (seqlen_kv_c + peer_denominator_1 - 1) / peer_denominator_1;
                asm volatile("griddepcontrol.wait;" ::: "memory");
                {
                    bmm1_scale_log2_c = bmm1_scale_ptr[0];
                    if (bmm1_is_log2 == 0) {
                        bmm1_scale_log2_c = bmm1_scale_log2_c * 1.4426950408889634f;
                    }
                    bmm2_scale_c = bmm2_scale_ptr[0];
                }
                #pragma unroll 1
                for (int _n = 0; _n < peer_blocks_1; _n++) {
                    mbarrier_wait(corr_scale_0_addr + (corr_cons_stage) * 8, corr_cons_phase);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float row_scale_qm = smem_q6_scales[corr_cons_stage * 128 + corr_tid];
                    if (_n > 0) {
                        mbarrier_wait(o_full_addr, _phase_o_full_0);
                        _phase_o_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _vote_0 = __any_sync(0xFFFFFFFF, row_scale_qm != 1.0f);
                        if (_vote_0 != 0) {
                            #pragma unroll
                            for (int output_half_qm = 0; output_half_qm < 2; output_half_qm++) {
                                #pragma unroll
                                for (int output_chunk_qm = 0; output_chunk_qm < 8; output_chunk_qm++) {
                                    int correction_addr_qm = taddr + 256 + (unsigned int)(output_half_qm * 128) + (unsigned int)corr_row + (unsigned int)(output_chunk_qm * 16);
                                    float _tmem_load_1[16];
                                    tmem_ld_x16(&_tmem_load_1[0], correction_addr_qm);
                                    const float2 _scale2_0 = {row_scale_qm, row_scale_qm};
                                    #pragma unroll
                                    for (int _ls = 0; _ls < 8; _ls++)
                                        mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_1)[_ls], _scale2_0);
                                    tmem_st_x16_f32(correction_addr_qm, _tmem_load_1);
                                }
                            }
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        }
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        mbarrier_arrive(o_empty_addr);
                    }
                    mbarrier_arrive(p_full_0_addr + (p_stage_c) * 8);
                    p_stage_c += 1;
                    if (p_stage_c == 2) { p_stage_c = 0; }
                    mbarrier_arrive(corr_empty_0_addr + (corr_cons_stage) * 8);
                    corr_cons_stage += 1;
                    if (corr_cons_stage == 2) { corr_cons_stage = 0; corr_cons_phase ^= 1; }
                }
                mbarrier_wait(corr_scale_0_addr + (corr_cons_stage) * 8, corr_cons_phase);
                asm volatile("tcgen05.fence::after_thread_sync;");
                float sum_final_qm = smem_q6_final[corr_tid * 2];
                float max_final_qm = smem_q6_final[corr_tid * 2 + 1];
                mbarrier_arrive(corr_empty_0_addr + (corr_cons_stage) * 8);
                corr_cons_stage += 1;
                if (corr_cons_stage == 2) { corr_cons_stage = 0; corr_cons_phase ^= 1; }
                if (peer_blocks_1 > 0) {
                    mbarrier_wait(o_full_addr, _phase_o_full_0);
                    _phase_o_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                }
                int output_row_qm = (batch_idx_c * Q_LEN + q_row_idx_c + corr_tid / 16) * num_q_heads + kv_head_idx_c * 16 + corr_tid % 16;
                if (corr_tid < 96) {
                    int online_store_row = ((batch_idx_c * num_kv_heads + kv_head_idx_c) * NUM_SPLIT + split_idx_c) * 128 + corr_tid;
                    float online_store_stats[2];
                    online_store_stats[0] = -CAKE_INF;
                    online_store_stats[1] = sum_final_qm;
                    if (sum_final_qm > 0.0f) {
                        online_store_stats[0] = max_final_qm;
                    }
                    {
                        float2 _v2 = make_float2(online_store_stats[0 + 0], online_store_stats[0 + 1]);
                        *reinterpret_cast<float2*>(partial_LSE + (online_store_row * 2) + 0) = _v2;
                    }
                }
                float normalizer_qm = 0.0f;
                if (sum_final_qm > 0.0f) {
                    normalizer_qm = bmm2_scale_c;
                }
                #pragma unroll
                for (int output_half_qm_1 = 0; output_half_qm_1 < 2; output_half_qm_1++) {
                    #pragma unroll
                    for (int epilogue_chunk_qm = 0; epilogue_chunk_qm < 8; epilogue_chunk_qm++) {
                        int epilogue_addr_qm = taddr + 256 + (unsigned int)(output_half_qm_1 * 128) + (unsigned int)corr_row + (unsigned int)(epilogue_chunk_qm * 16);
                        float epilogue_values_qm[16];
                        epilogue_values_qm[0] = 0.0f;
                        epilogue_values_qm[1] = 0.0f;
                        epilogue_values_qm[2] = 0.0f;
                        epilogue_values_qm[3] = 0.0f;
                        epilogue_values_qm[4] = 0.0f;
                        epilogue_values_qm[5] = 0.0f;
                        epilogue_values_qm[6] = 0.0f;
                        epilogue_values_qm[7] = 0.0f;
                        epilogue_values_qm[8] = 0.0f;
                        epilogue_values_qm[9] = 0.0f;
                        epilogue_values_qm[10] = 0.0f;
                        epilogue_values_qm[11] = 0.0f;
                        epilogue_values_qm[12] = 0.0f;
                        epilogue_values_qm[13] = 0.0f;
                        epilogue_values_qm[14] = 0.0f;
                        epilogue_values_qm[15] = 0.0f;
                        if (peer_blocks_1 > 0) {
                            tmem_ld_x16(&epilogue_values_qm[0], epilogue_addr_qm);
                        }
                        const float2 _scale2_1 = {normalizer_qm, normalizer_qm};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(epilogue_values_qm)[_ls], _scale2_1);
                        if (corr_tid < 96) {
                            int partial_offset_qm = (((batch_idx_c * num_kv_heads + kv_head_idx_c) * NUM_SPLIT + split_idx_c) * 128 + corr_tid) * HEAD_DIM + output_half_qm_1 * 128 + epilogue_chunk_qm * 16;
                            {
                                __nv_bfloat162 _pk[8];
                                _pk[0] = __floats2bfloat162_rn(epilogue_values_qm[0 + 0], epilogue_values_qm[0 + 1]);
                                _pk[1] = __floats2bfloat162_rn(epilogue_values_qm[0 + 2], epilogue_values_qm[0 + 3]);
                                _pk[2] = __floats2bfloat162_rn(epilogue_values_qm[0 + 4], epilogue_values_qm[0 + 5]);
                                _pk[3] = __floats2bfloat162_rn(epilogue_values_qm[0 + 6], epilogue_values_qm[0 + 7]);
                                _pk[4] = __floats2bfloat162_rn(epilogue_values_qm[0 + 8], epilogue_values_qm[0 + 9]);
                                _pk[5] = __floats2bfloat162_rn(epilogue_values_qm[0 + 10], epilogue_values_qm[0 + 11]);
                                _pk[6] = __floats2bfloat162_rn(epilogue_values_qm[0 + 12], epilogue_values_qm[0 + 13]);
                                _pk[7] = __floats2bfloat162_rn(epilogue_values_qm[0 + 14], epilogue_values_qm[0 + 15]);
                                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(partial_O + partial_offset_qm))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(partial_O + partial_offset_qm))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                            }
                        }
                    }
                }
                mbarrier_arrive(o_empty_addr);
                if (USE_SEGMENTED_CLC != 0 && part_count_c > 1) {
                    int base_tile_idx_seg = (batch_idx_c * Q_LEN + q_row_idx_c) * (num_kv_heads * Q_GROUPS_PER_KV) + kv_head_idx_c;
                    __threadfence();
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                    if (d_idx == 0) {
                        uint32_t _atomic_inc_old_0;
                        asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                            : "=r"(_atomic_inc_old_0) : "l"(&split_completion[base_tile_idx_seg]), "r"(static_cast<uint32_t>(part_count_c - 1)) : "memory");
                        unsigned int old_count_seg = _atomic_inc_old_0;
                        split_reduce_flag[0] = (((int)old_count_seg + 1 == part_count_c) ? 1 : 0);
                    }
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                    if (split_reduce_flag[0] != 0) {
                        __threadfence();
                        int reduce_head_seg = d_idx / 8;
                        int reduce_lane_seg = d_idx % 8;
                        int reduce_head_valid_seg = 0;
                        if (reduce_head_seg < group_ratio_rt) {
                            if (reduce_head_seg < TILE_Q) {
                                reduce_head_valid_seg = 1;
                            }
                        }
                        int reduce_q_head_seg = kv_head_idx_c * group_ratio_rt + reduce_head_seg;
                        int reduce_stat_base_seg = ((batch_idx_c * Q_LEN + q_row_idx_c) * num_q_heads + reduce_q_head_seg) * 16;
                        int split0_seg = reduce_lane_seg;
                        int split1_seg = reduce_lane_seg + 8;
                        float lse0_seg = -CAKE_INF;
                        float lse1_seg = -CAKE_INF;
                        if (reduce_head_valid_seg != 0) {
                            if (split0_seg < part_count_c) {
                                lse0_seg = partial_LSE[reduce_stat_base_seg + split0_seg];
                            }
                            if (split1_seg < part_count_c) {
                                lse1_seg = partial_LSE[reduce_stat_base_seg + split1_seg];
                            }
                        }
                        float _max_2 = max_noftz(lse0_seg, lse1_seg);
                        float lane_max_seg = _max_2;
                        int subgroup_lane_base_seg = lane / 8 * 8;
                        float merged_max_seg = -CAKE_INF;
                        #pragma unroll
                        for (int source_lane_seg = 0; source_lane_seg < 8; source_lane_seg++) {
                            float _shfl_0 = __shfl_sync(0xFFFFFFFF, lane_max_seg, subgroup_lane_base_seg + source_lane_seg);
                            float source_max_seg = _shfl_0;
                            float _max_3 = max_noftz(merged_max_seg, source_max_seg);
                            merged_max_seg = _max_3;
                        }
                        float weight0_seg = 0.0f;
                        float weight1_seg = 0.0f;
                        if (lse0_seg != -CAKE_INF) {
                            float _exp2_1 = approx_exp2(lse0_seg - merged_max_seg);
                            weight0_seg = _exp2_1;
                        }
                        if (lse1_seg != -CAKE_INF) {
                            float _exp2_2 = approx_exp2(lse1_seg - merged_max_seg);
                            weight1_seg = _exp2_2;
                        }
                        float lane_weight_sum_seg = weight0_seg + weight1_seg;
                        float weight_sum_seg = 0.0f;
                        #pragma unroll
                        for (int source_lane_seg_1 = 0; source_lane_seg_1 < 8; source_lane_seg_1++) {
                            float _shfl_1 = __shfl_sync(0xFFFFFFFF, lane_weight_sum_seg, subgroup_lane_base_seg + source_lane_seg_1);
                            weight_sum_seg = weight_sum_seg + _shfl_1;
                        }
                        float _rcp_0 = approx_rcp(weight_sum_seg);
                        float inv_weight_sum_seg = ((weight_sum_seg > 0.0f) ? _rcp_0 : 0.0f);
                        if (reduce_head_valid_seg != 0) {
                            if (split0_seg < part_count_c) {
                                split_weights[split0_seg * TILE_Q + reduce_head_seg] = weight0_seg * inv_weight_sum_seg;
                            }
                            if (split1_seg < part_count_c) {
                                split_weights[split1_seg * TILE_Q + reduce_head_seg] = weight1_seg * inv_weight_sum_seg;
                            }
                            if (reduce_lane_seg == 0) {
                            }
                        }
                        asm volatile("barrier.sync 9, 128;" ::: "memory");
                        int merge_head_seg = d_idx / 8;
                        int merge_d_base_seg = d_idx % 8 * 32;
                        int merge_head_valid_seg = 0;
                        if (merge_head_seg < group_ratio_rt) {
                            if (merge_head_seg < TILE_Q) {
                                merge_head_valid_seg = 1;
                            }
                        }
                        if (merge_head_valid_seg != 0) {
                            int merge_q_head_seg = kv_head_idx_c * group_ratio_rt + merge_head_seg;
                            #pragma unroll
                            for (int vec_chunk_seg = 0; vec_chunk_seg < 4; vec_chunk_seg++) {
                                int elem_base_seg = merge_d_base_seg + vec_chunk_seg * 8;
                                int partial_o_base_seg = ((batch_idx_c * Q_LEN + q_row_idx_c) * num_q_heads + merge_q_head_seg) * 16 * HEAD_DIM + elem_base_seg;
                                int final_o_idx_seg = ((batch_idx_c * Q_LEN + q_row_idx_c) * num_q_heads + merge_q_head_seg) * HEAD_DIM + elem_base_seg;
                                float _vec_load_0[8];
                                {
                                    const uint4* _vptr_2 = reinterpret_cast<const uint4*>(partial_O + partial_o_base_seg + 0);
                                    uint4 _vld_2[1];
                                    #pragma unroll
                                    for (int _blk = 0; _blk < 1; _blk++) {
                                        _vld_2[_blk] = _vptr_2[_blk];
                                        uint32_t* _vpairs_2 = reinterpret_cast<uint32_t*>(&_vld_2[_blk]);
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 4; _pair++) {
                                            asm volatile(
                                                "{\n\t"
                                                "shl.b32 %0, %2, 16;\n\t"
                                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                                "}\n"
                                                : "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_0[0 + _blk * 8 + _pair * 2])[1])
                                                : "r"(_vpairs_2[_pair]));
                                        }
                                    }
                                }
                                float merge_weight0_seg = split_weights[merge_head_seg];
                                #pragma unroll
                                for (int elem_seg = 0; elem_seg < 8; elem_seg++) {
                                    _vec_load_0[elem_seg] = _vec_load_0[elem_seg] * merge_weight0_seg;
                                }
                                #pragma unroll 2
                                for (int reduce_part_seg = 1; reduce_part_seg < part_count_c; reduce_part_seg++) {
                                    float _vec_load_1[8];
                                    {
                                        const uint4* _vptr_3 = reinterpret_cast<const uint4*>(partial_O + (partial_o_base_seg + reduce_part_seg * HEAD_DIM) + 0);
                                        uint4 _vld_3[1];
                                        #pragma unroll
                                        for (int _blk = 0; _blk < 1; _blk++) {
                                            _vld_3[_blk] = _vptr_3[_blk];
                                            uint32_t* _vpairs_3 = reinterpret_cast<uint32_t*>(&_vld_3[_blk]);
                                            #pragma unroll
                                            for (int _pair = 0; _pair < 4; _pair++) {
                                                asm volatile(
                                                    "{\n\t"
                                                    "shl.b32 %0, %2, 16;\n\t"
                                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                                    "}\n"
                                                    : "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_1[0 + _blk * 8 + _pair * 2])[1])
                                                    : "r"(_vpairs_3[_pair]));
                                            }
                                        }
                                    }
                                    float reduce_weight_seg = split_weights[reduce_part_seg * TILE_Q + merge_head_seg];
                                    #pragma unroll
                                    for (int elem_seg_1 = 0; elem_seg_1 < 8; elem_seg_1++) {
                                        float _fma_1 = __fmaf_rn(_vec_load_1[elem_seg_1], reduce_weight_seg, _vec_load_0[elem_seg_1]);
                                        _vec_load_0[elem_seg_1] = _fma_1;
                                    }
                                }
                                {
                                    __nv_bfloat162 _pk[4];
                                    _pk[0] = __floats2bfloat162_rn(_vec_load_0[0 + 0], _vec_load_0[0 + 1]);
                                    _pk[1] = __floats2bfloat162_rn(_vec_load_0[0 + 2], _vec_load_0[0 + 3]);
                                    _pk[2] = __floats2bfloat162_rn(_vec_load_0[0 + 4], _vec_load_0[0 + 5]);
                                    _pk[3] = __floats2bfloat162_rn(_vec_load_0[0 + 6], _vec_load_0[0 + 7]);
                                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O + final_o_idx_seg))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                }
                            }
                        }
                    }
                } else if (USE_SEGMENTED_CLC == 0 && NUM_SPLIT > 1) {
                    int base_tile_idx = (batch_idx_c * Q_LEN + q_row_idx_c) * (num_kv_heads * Q_GROUPS_PER_KV) + kv_head_idx_c;
                    __threadfence();
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                    if (d_idx == 0) {
                        uint32_t _atomic_inc_old_1;
                        asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                            : "=r"(_atomic_inc_old_1) : "l"(&split_completion[base_tile_idx]), "r"(static_cast<uint32_t>(part_count_c - 1)) : "memory");
                        unsigned int old_count = _atomic_inc_old_1;
                        int peer_groups = (24 + part_count_c - 1) / part_count_c;
                        int peer_reducers = (24 + peer_groups - 1) / peer_groups;
                        split_reduce_flag[0] = 0;
                        if (peer_reducers > part_count_c - 1 - (int)old_count) {
                            split_reduce_flag[0] = part_count_c - (int)old_count;
                            if (part_count_c > (int)old_count + 1) {
                                {
                                    const uint32_t* _awe_p_4 = &split_completion[base_tile_idx];
                                    while (true) {
                                        uint32_t _awe_v_4;
                                        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_awe_v_4) : "l"(_awe_p_4) : "memory");
                                        if (_awe_v_4 == static_cast<uint32_t>(0)) break;
                                    }
                                }
                            }
                        }
                    }
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                    if (split_reduce_flag[0] != 0) {
                        __threadfence();
                        int peer_groups_per_reducer = (24 + part_count_c - 1) / part_count_c;
                        int peer_rank = split_reduce_flag[0] - 1;
                        #pragma unroll 1
                        for (int peer_local_group = 0; peer_local_group < peer_groups_per_reducer; peer_local_group++) {
                            int peer_row_group = peer_rank * peer_groups_per_reducer + peer_local_group;
                            if (peer_row_group < 24) {
                                int peer_row = peer_row_group * 4 + d_idx / 32;
                                int peer_lane = d_idx % 32;
                                int peer_q_head = kv_head_idx_c * 16 + peer_row / 16 * num_q_heads + peer_row % 16;
                                int peer_stat_base = ((batch_idx_c * Q_LEN + q_row_idx_c) * num_q_heads + peer_q_head) * NUM_SPLIT;
                                int online_group_base = (batch_idx_c * num_kv_heads + kv_head_idx_c) * NUM_SPLIT;
                                float online_max = -CAKE_INF;
                                float online_sum = 0.0f;
                                float online_o[8];
                                online_o[0] = 0.0f;
                                online_o[1] = 0.0f;
                                online_o[2] = 0.0f;
                                online_o[3] = 0.0f;
                                online_o[4] = 0.0f;
                                online_o[5] = 0.0f;
                                online_o[6] = 0.0f;
                                online_o[7] = 0.0f;
                                #pragma unroll 1
                                for (int online_base = 0; online_base < part_count_c; online_base += 4) {
                                    float online_stats[8];
                                    float online_partials[32];
                                    #pragma unroll
                                    for (int online_set = 0; online_set < 4; online_set++) {
                                        int _min_7 = ((online_base + online_set) < (part_count_c - 1) ? (online_base + online_set) : (part_count_c - 1));
                                        int online_load_split = _min_7;
                                        int online_load_row = (online_group_base + online_load_split) * 128 + peer_row;
                                        float _vec_load_2[2];
                                        {
                                            float2 _v2_5 = *reinterpret_cast<const float2*>(partial_LSE + (online_load_row * 2) + 0);
                                            _vec_load_2[0] = _v2_5.x;
                                            _vec_load_2[0 + 1] = _v2_5.y;
                                        }
                                        float _vec_load_3[8];
                                        {
                                            const uint4* _vptr_6 = reinterpret_cast<const uint4*>(partial_O + (online_load_row * HEAD_DIM + peer_lane * 8) + 0);
                                            uint4 _vld_6[1];
                                            #pragma unroll
                                            for (int _blk = 0; _blk < 1; _blk++) {
                                                _vld_6[_blk] = _vptr_6[_blk];
                                                uint32_t* _vpairs_6 = reinterpret_cast<uint32_t*>(&_vld_6[_blk]);
                                                #pragma unroll
                                                for (int _pair = 0; _pair < 4; _pair++) {
                                                    asm volatile(
                                                        "{\n\t"
                                                        "shl.b32 %0, %2, 16;\n\t"
                                                        "and.b32 %1, %2, 0xffff0000;\n\t"
                                                        "}\n"
                                                        : "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[1])
                                                        : "r"(_vpairs_6[_pair]));
                                                }
                                            }
                                        }
                                        #pragma unroll
                                        for (int online_stat = 0; online_stat < 2; online_stat++) {
                                            online_stats[online_set * 2 + online_stat] = _vec_load_2[online_stat];
                                        }
                                        #pragma unroll
                                        for (int online_element = 0; online_element < 8; online_element++) {
                                            online_partials[online_set * 8 + online_element] = _vec_load_3[online_element];
                                        }
                                    }
                                    #pragma unroll
                                    for (int online_set_1 = 0; online_set_1 < 4; online_set_1++) {
                                        float online_alpha = 1.0f;
                                        float online_beta = 0.0f;
                                        float online_new_max = online_max;
                                        if (part_count_c > online_base + online_set_1) {
                                            if (online_stats[online_set_1 * 2 + 1] > 0.0f) {
                                                float _max_4 = max_noftz(online_max, online_stats[online_set_1 * 2]);
                                                online_new_max = _max_4;
                                                online_alpha = 0.0f;
                                                if (online_sum > 0.0f) {
                                                    float _exp2_3 = approx_exp2((online_max - online_new_max) * bmm1_scale_log2_c);
                                                    online_alpha = _exp2_3;
                                                }
                                                float _exp2_4 = approx_exp2((online_stats[online_set_1 * 2] - online_new_max) * bmm1_scale_log2_c);
                                                online_beta = _exp2_4;
                                            }
                                        }
                                        float _fma_2 = __fmaf_rn(online_sum, online_alpha, online_stats[online_set_1 * 2 + 1] * online_beta);
                                        online_sum = _fma_2;
                                        float2 _f2_0 = make_float2(online_alpha, online_alpha);
                                        float2 _f2_1 = make_float2(online_beta, online_beta);
                                        #pragma unroll
                                        for (int online_pair = 0; online_pair < 4; online_pair++) {
                                            float2 _f2_2 = make_float2(online_o[online_pair * 2], online_o[online_pair * 2 + 1]);
                                            float2 _mul_f32x2_0;
                                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_0) : "l"(*(const unsigned long long*)&_f2_2), "l"(*(const unsigned long long*)&_f2_0));
                                            float2 _f2_3 = make_float2(online_partials[online_set_1 * 8 + online_pair * 2], online_partials[online_set_1 * 8 + online_pair * 2 + 1]);
                                            float2 online_merged_pair = fma_f32x2_rn_ftz(_f2_3, _f2_1, _mul_f32x2_0);
                                            online_o[online_pair * 2] = online_merged_pair.x;
                                            online_o[online_pair * 2 + 1] = online_merged_pair.y;
                                        }
                                        online_max = online_new_max;
                                    }
                                }
                                float online_inv = 0.0f;
                                if (online_sum > 0.0f) {
                                    float _rcp_1 = approx_rcp(online_sum);
                                    online_inv = _rcp_1;
                                }
                                float2 _f2_4 = make_float2(online_inv, online_inv);
                                #pragma unroll
                                for (int online_pair_1 = 0; online_pair_1 < 4; online_pair_1++) {
                                    float2 _f2_5 = make_float2(online_o[online_pair_1 * 2], online_o[online_pair_1 * 2 + 1]);
                                    float2 _mul_f32x2_1;
                                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_1) : "l"(*(const unsigned long long*)&_f2_5), "l"(*(const unsigned long long*)&_f2_4));
                                    online_o[online_pair_1 * 2] = _mul_f32x2_1.x;
                                    online_o[online_pair_1 * 2 + 1] = _mul_f32x2_1.y;
                                }
                                int online_output_base = peer_stat_base / NUM_SPLIT * HEAD_DIM + peer_lane * 8;
                                {
                                    __nv_bfloat162 _pk[4];
                                    _pk[0] = __floats2bfloat162_rn(online_o[0 + 0], online_o[0 + 1]);
                                    _pk[1] = __floats2bfloat162_rn(online_o[0 + 2], online_o[0 + 3]);
                                    _pk[2] = __floats2bfloat162_rn(online_o[0 + 4], online_o[0 + 5]);
                                    _pk[3] = __floats2bfloat162_rn(online_o[0 + 6], online_o[0 + 7]);
                                    *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O + online_output_base))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                }
                            }
                        }
                    }
                }
            }
            mbarrier_arrive(tmem_dealloc_addr);
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
    }
    // ---- Role: mma_warp ----
    if (warp == 8) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
        { // mma_warp_main
            const int tmem_s0v = taddr;
            const int tmem_o_hi_v = taddr + 256;
            const int tmem_o_lo_v = taddr + 384;
            unsigned int work_stage_m = 0;
            int sm_stage_1 = 0;
            int transformed_stage = 0;
            int transformed_phase = 0;
            int q_phase_m = 0;
            int sm_empty_phase_m = 1;
            int p_stage_m = 0;
            int p_phase_m = 0;
            mbarrier_wait(s_empty_0_addr, sm_empty_phase_m);
            mbarrier_wait(s_empty_0_addr + 8, sm_empty_phase_m);
            int batch_idx_2 = 0;
            int q_row_idx_2 = 0;
            int kv_head_idx_2 = 0;
            int split_idx_2 = 0;
            int part_count_2 = NUM_SPLIT;
            int bundle_idx_2 = 0;
            int bundle_item_idx_2 = 0;
            {
                q_row_idx_2 = blockIdx.x / NUM_SPLIT * 6;
                split_idx_2 = blockIdx.x % NUM_SPLIT;
                kv_head_idx_2 = 0;
                kv_head_idx_2 = blockIdx.y;
                int schedule_batch_idx_2 = blockIdx.z;
                batch_idx_2 = schedule_batch_idx_2;
                {
                    batch_idx_2 = request_order[schedule_batch_idx_2];
                }
            }
            int visible_keys_2 = UNIFORM_KV_LEN - Q_LEN + q_row_idx_2 + 1;
            {
                visible_keys_2 = seq_lens_kv[batch_idx_2] - Q_LEN + q_row_idx_2 + 1;
            }
            visible_keys_2 = visible_keys_2 + 6 - 1;
            if (visible_keys_2 < 0) {
                visible_keys_2 = 0;
            }
            int peer_visible_2 = visible_keys_2;
            int _min_3 = (((peer_visible_2 + BLOCK_N - 1) / BLOCK_N) < (NUM_SPLIT) ? ((peer_visible_2 + BLOCK_N - 1) / BLOCK_N) : (NUM_SPLIT));
            part_count_2 = _min_3;
            int batch_idx_m = batch_idx_2;
            int q_row_idx_m = q_row_idx_2;
            int kv_head_idx_m = kv_head_idx_2;
            int split_idx_m = split_idx_2;
            int part_count_m = part_count_2;
            int bundle_idx_m = bundle_idx_2;
            int bundle_item_idx_m = bundle_item_idx_2;
            unsigned int peer_tiles_2 = 0;
            if (split_idx_m < part_count_m) {
                peer_tiles_2 = total_tiles;
            }
            unsigned int _phase_o_empty_0 = 1;
            unsigned int _phase_q_full_0 = 0;
            unsigned int _phase_p_full_0_0 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_m = 0; _tile_iter_m < peer_tiles_2; _tile_iter_m++) {
                int visible_keys_0_2 = UNIFORM_KV_LEN - Q_LEN + q_row_idx_m + 1;
                {
                    visible_keys_0_2 = seq_lens_kv[batch_idx_m] - Q_LEN + q_row_idx_m + 1;
                }
                visible_keys_0_2 = visible_keys_0_2 + 6 - 1;
                if (visible_keys_0_2 < 0) {
                    visible_keys_0_2 = 0;
                }
                int seqlen_kv_m = visible_keys_0_2;
                int peer_denominator_2 = part_count_m * BLOCK_N;
                int peer_blocks_2 = (seqlen_kv_m + peer_denominator_2 - 1) / peer_denominator_2;
                int first_pv = 1;
                {
                    uint32_t _mbar_token_0 = mbarrier_try_wait(q_full_addr, q_phase_m);
                    mbarrier_wait_token(q_full_addr, q_phase_m, _mbar_token_0);
                    q_phase_m ^= 1;
                    if (peer_blocks_2 > 0) {
                        mbarrier_wait(s_empty_0_addr + (sm_stage_1) * 8, sm_empty_phase_m);
                        uint32_t _mbar_token_1 = mbarrier_try_wait(kv_full_addr + (transformed_stage) * 8, transformed_phase);
                        mbarrier_wait_token(kv_full_addr + (transformed_stage) * 8, transformed_phase, _mbar_token_1);
                        int _mma_a_lo_0 = make_warp_uniform(((smem_qt_hi_addr) >> 4) & 0x3FFF);
                        int _mma_b_lo_0 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (transformed_stage) * 2048);
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
                    "mov.b32 id, 136316048;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 1018;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_tmem_s0 + (sm_stage_1 * 128))), "r"(0));
                        elect_commit(kv_empty_addr + (transformed_stage) * 8);
                        transformed_stage += 1;
                        if (transformed_stage == 2) { transformed_stage = 0; transformed_phase ^= 1; }
                        uint32_t _mbar_token_2 = mbarrier_try_wait(kv_full_addr + (transformed_stage) * 8, transformed_phase);
                        mbarrier_wait_token(kv_full_addr + (transformed_stage) * 8, transformed_phase, _mbar_token_2);
                        int _mma_a_lo_1 = make_warp_uniform(((smem_qt_lo_addr) >> 4) & 0x3FFF);
                        int _mma_b_lo_1 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (transformed_stage) * 2048);
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
                    "mov.b32 id, 136316048;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 1018;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"((tmem_tmem_s0 + (sm_stage_1 * 128))), "r"(1));
                        elect_commit(s_full_0_addr + (sm_stage_1) * 8);
                        elect_commit(kv_empty_addr + (transformed_stage) * 8);
                        transformed_stage += 1;
                        if (transformed_stage == 2) { transformed_stage = 0; transformed_phase ^= 1; }
                        sm_stage_1 += 1;
                        if (sm_stage_1 == 2) { sm_stage_1 = 0; sm_empty_phase_m ^= 1; }
                        #pragma unroll 1
                        for (int _body_n_m = 0; _body_n_m < peer_blocks_2 - 1; _body_n_m++) {
                            mbarrier_wait(s_empty_0_addr + (sm_stage_1) * 8, sm_empty_phase_m);
                            uint32_t _mbar_token_3 = mbarrier_try_wait(kv_full_addr + (transformed_stage) * 8, transformed_phase);
                            mbarrier_wait_token(kv_full_addr + (transformed_stage) * 8, transformed_phase, _mbar_token_3);
                            int _mma_a_lo_2 = make_warp_uniform(((smem_qt_hi_addr) >> 4) & 0x3FFF);
                            int _mma_b_lo_2 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (transformed_stage) * 2048);
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
                    "mov.b32 id, 136316048;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 1018;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_2), "r"(_mma_b_lo_2), "r"((tmem_tmem_s0 + (sm_stage_1 * 128))), "r"(0));
                            elect_commit(kv_empty_addr + (transformed_stage) * 8);
                            transformed_stage += 1;
                            if (transformed_stage == 2) { transformed_stage = 0; transformed_phase ^= 1; }
                            uint32_t _mbar_token_4 = mbarrier_try_wait(kv_full_addr + (transformed_stage) * 8, transformed_phase);
                            mbarrier_wait_token(kv_full_addr + (transformed_stage) * 8, transformed_phase, _mbar_token_4);
                            int _mma_a_lo_3 = make_warp_uniform(((smem_qt_lo_addr) >> 4) & 0x3FFF);
                            int _mma_b_lo_3 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (transformed_stage) * 2048);
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
                    "mov.b32 id, 136316048;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 1018;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_3), "r"(_mma_b_lo_3), "r"((tmem_tmem_s0 + (sm_stage_1 * 128))), "r"(1));
                            elect_commit(s_full_0_addr + (sm_stage_1) * 8);
                            elect_commit(kv_empty_addr + (transformed_stage) * 8);
                            transformed_stage += 1;
                            if (transformed_stage == 2) { transformed_stage = 0; transformed_phase ^= 1; }
                            sm_stage_1 += 1;
                            if (sm_stage_1 == 2) { sm_stage_1 = 0; sm_empty_phase_m ^= 1; }
                            mbarrier_wait(p_full_0_addr + (p_stage_m) * 8, p_phase_m);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            mbarrier_wait(o_empty_addr, _phase_o_empty_0);
                            _phase_o_empty_0 ^= 1;
                            int first_pv_flag = first_pv;
                            uint32_t _mbar_token_5 = mbarrier_try_wait(kv_full_addr + (transformed_stage) * 8, transformed_phase);
                            mbarrier_wait_token(kv_full_addr + (transformed_stage) * 8, transformed_phase, _mbar_token_5);
                            int _mma_b_lo_4 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (transformed_stage) * 2048);
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
                    "mov.b32 id, 136381584;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
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
                    "}\n"
                    :: "r"(tmem_tmem_o_hi), "r"(_mma_b_lo_4), "r"(tmem_tmem_s0 + (p_stage_m * 128 + 32)), "r"(((first_pv_flag) ? 0 : 1)));
                            elect_commit(kv_empty_addr + (transformed_stage) * 8);
                            transformed_stage += 1;
                            if (transformed_stage == 2) { transformed_stage = 0; transformed_phase ^= 1; }
                            uint32_t _mbar_token_6 = mbarrier_try_wait(kv_full_addr + (transformed_stage) * 8, transformed_phase);
                            mbarrier_wait_token(kv_full_addr + (transformed_stage) * 8, transformed_phase, _mbar_token_6);
                            int _mma_b_lo_5 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (transformed_stage) * 2048);
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
                    "mov.b32 id, 136381584;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
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
                    "}\n"
                    :: "r"(tmem_tmem_o_lo), "r"(_mma_b_lo_5), "r"(tmem_tmem_s0 + (p_stage_m * 128 + 32)), "r"(((first_pv_flag) ? 0 : 1)));
                            elect_commit(kv_empty_addr + (transformed_stage) * 8);
                            transformed_stage += 1;
                            if (transformed_stage == 2) { transformed_stage = 0; transformed_phase ^= 1; }
                            elect_commit(o_full_addr);
                            first_pv = 0;
                            p_stage_m += 1;
                            if (p_stage_m == 2) { p_stage_m = 0; p_phase_m ^= 1; }
                        }
                        mbarrier_wait(s_empty_0_addr + (sm_stage_1) * 8, sm_empty_phase_m);
                        mbarrier_wait(s_empty_0_addr + (sm_stage_1 ^ 1) * 8, sm_empty_phase_m ^ sm_stage_1);
                        elect_commit(q_empty_addr);
                        mbarrier_wait(p_full_0_addr + (p_stage_m) * 8, p_phase_m);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        mbarrier_wait(o_empty_addr, _phase_o_empty_0);
                        _phase_o_empty_0 ^= 1;
                        int last_pv_init = first_pv;
                        uint32_t _mbar_token_7 = mbarrier_try_wait(kv_full_addr + (transformed_stage) * 8, transformed_phase);
                        mbarrier_wait_token(kv_full_addr + (transformed_stage) * 8, transformed_phase, _mbar_token_7);
                        int _mma_b_lo_6 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (transformed_stage) * 2048);
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
                    "mov.b32 id, 136381584;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
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
                    "}\n"
                    :: "r"(tmem_tmem_o_hi), "r"(_mma_b_lo_6), "r"(tmem_tmem_s0 + (p_stage_m * 128 + 32)), "r"(((last_pv_init) ? 0 : 1)));
                        elect_commit(kv_empty_addr + (transformed_stage) * 8);
                        transformed_stage += 1;
                        if (transformed_stage == 2) { transformed_stage = 0; transformed_phase ^= 1; }
                        uint32_t _mbar_token_8 = mbarrier_try_wait(kv_full_addr + (transformed_stage) * 8, transformed_phase);
                        mbarrier_wait_token(kv_full_addr + (transformed_stage) * 8, transformed_phase, _mbar_token_8);
                        int _mma_b_lo_7 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (transformed_stage) * 2048);
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
                    "mov.b32 id, 136381584;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
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
                    "}\n"
                    :: "r"(tmem_tmem_o_lo), "r"(_mma_b_lo_7), "r"(tmem_tmem_s0 + (p_stage_m * 128 + 32)), "r"(((last_pv_init) ? 0 : 1)));
                        elect_commit(kv_empty_addr + (transformed_stage) * 8);
                        transformed_stage += 1;
                        if (transformed_stage == 2) { transformed_stage = 0; transformed_phase ^= 1; }
                        elect_commit(o_full_addr);
                        p_stage_m += 1;
                        if (p_stage_m == 2) { p_stage_m = 0; p_phase_m ^= 1; }
                    } else {
                        elect_commit(q_empty_addr);
                    }
                }
            }
            elect_commit(s_full_0_addr + (sm_stage_1) * 8);
            sm_stage_1 += 1;
            if (sm_stage_1 == 2) { sm_stage_1 = 0; sm_empty_phase_m ^= 1; }
            elect_commit(s_full_0_addr + (sm_stage_1) * 8);
            sm_stage_1 += 1;
            if (sm_stage_1 == 2) { sm_stage_1 = 0; sm_empty_phase_m ^= 1; }
            unsigned int _phase_tmem_dealloc_0 = 0;
            mbarrier_wait(tmem_dealloc_addr, _phase_tmem_dealloc_0);
            _phase_tmem_dealloc_0 ^= 1;
            int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(512));
        }
    }
    // ---- Role: page_offsets ----
    if (warp == 9) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
        { // page_offsets_main
            unsigned int work_stage_p = 0;
            int page_prod_stage = 0;
            int page_prod_phase = 1;
            int batch_idx_3 = 0;
            int q_row_idx_3 = 0;
            int kv_head_idx_3 = 0;
            int split_idx_3 = 0;
            int part_count_3 = NUM_SPLIT;
            int bundle_idx_3 = 0;
            int bundle_item_idx_3 = 0;
            {
                q_row_idx_3 = blockIdx.x / NUM_SPLIT * 6;
                split_idx_3 = blockIdx.x % NUM_SPLIT;
                kv_head_idx_3 = 0;
                kv_head_idx_3 = blockIdx.y;
                int schedule_batch_idx_3 = blockIdx.z;
                batch_idx_3 = schedule_batch_idx_3;
                {
                    batch_idx_3 = request_order[schedule_batch_idx_3];
                }
            }
            int visible_keys_3 = UNIFORM_KV_LEN - Q_LEN + q_row_idx_3 + 1;
            {
                visible_keys_3 = seq_lens_kv[batch_idx_3] - Q_LEN + q_row_idx_3 + 1;
            }
            visible_keys_3 = visible_keys_3 + 6 - 1;
            if (visible_keys_3 < 0) {
                visible_keys_3 = 0;
            }
            int peer_visible_3 = visible_keys_3;
            int _min_0 = (((peer_visible_3 + BLOCK_N - 1) / BLOCK_N) < (NUM_SPLIT) ? ((peer_visible_3 + BLOCK_N - 1) / BLOCK_N) : (NUM_SPLIT));
            part_count_3 = _min_0;
            int batch_idx_p = batch_idx_3;
            int q_row_idx_p = q_row_idx_3;
            int kv_head_idx_p = kv_head_idx_3;
            int split_idx_p = split_idx_3;
            int part_count_p = part_count_3;
            int bundle_idx_p = bundle_idx_3;
            int bundle_item_idx_p = bundle_item_idx_3;
            unsigned int peer_tiles_3 = 0;
            if (split_idx_p < part_count_p) {
                peer_tiles_3 = total_tiles;
            }
            #pragma unroll 1
            for (unsigned int _tile_iter_p = 0; _tile_iter_p < peer_tiles_3; _tile_iter_p++) {
                int visible_keys_0_3 = UNIFORM_KV_LEN - Q_LEN + q_row_idx_p + 1;
                {
                    visible_keys_0_3 = seq_lens_kv[batch_idx_p] - Q_LEN + q_row_idx_p + 1;
                }
                visible_keys_0_3 = visible_keys_0_3 + 6 - 1;
                if (visible_keys_0_3 < 0) {
                    visible_keys_0_3 = 0;
                }
                int seqlen_kv_p = visible_keys_0_3;
                int peer_denominator_3 = part_count_p * BLOCK_N;
                int peer_blocks_3 = (seqlen_kv_p + peer_denominator_3 - 1) / peer_denominator_3;
                int pages_per_seq_p = (seqlen_kv_p + PAGE_SIZE - 1) / PAGE_SIZE;
                int max_page_p = pages_per_seq_p - 1;
                int pt_base_p = batch_idx_p * max_pages_per_seq;
                int pt_base_v_p = pt_base_p + page_table_v_offset;
                {
                    {
                        #pragma unroll 1
                        for (int n_p = 0; n_p < peer_blocks_3; n_p++) {
                            int n_block_p = split_idx_p * peer_blocks_3 + n_p;
                            int logical_page_base_p = n_block_p * 2;
                            mbarrier_wait(page_offsets_empty_addr + (page_prod_stage) * 8, page_prod_phase);
                            if (elect_sync()) {
                                int page_smem_base_p = page_prod_stage * 4;
                                #pragma unroll
                                for (int page_in_block_p = 0; page_in_block_p < 2; page_in_block_p++) {
                                    int logical_page_p = logical_page_base_p + page_in_block_p;
                                    int clamped_page_p = ((logical_page_p > max_page_p) ? max_page_p : logical_page_p);
                                    smem_page_offsets[page_smem_base_p + page_in_block_p] = page_table[pt_base_p + clamped_page_p];
                                    smem_page_offsets[page_smem_base_p + 2 + page_in_block_p] = page_table[pt_base_v_p + clamped_page_p];
                                }
                            }
                            mbarrier_arrive(page_offsets_full_addr + (page_prod_stage) * 8);
                            page_prod_stage += 1;
                            if (page_prod_stage == 6) { page_prod_stage = 0; page_prod_phase ^= 1; }
                        }
                    }
                }
            }
        }
    }
    // ---- Role: scheduler ----
    if (warp == 10) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
        // idle — no tasks assigned
    }
    // ---- Role: load_warp ----
    if (warp == 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 56;");
        { // load_warp_main
            unsigned int work_stage_l = 0;
            unsigned int throttle_stage_l = 0;
            int raw_stage = 0;
            int raw_phase = 1;
            int page_cons_stage = 0;
            int page_cons_phase = 0;
            int page_release_stage = 0;
            int page_release_phase = 0;
            int batch_idx_4 = 0;
            int q_row_idx_4 = 0;
            int kv_head_idx_4 = 0;
            int split_idx_4 = 0;
            int part_count_4 = NUM_SPLIT;
            int bundle_idx_4 = 0;
            int bundle_item_idx_4 = 0;
            {
                q_row_idx_4 = blockIdx.x / NUM_SPLIT * 6;
                split_idx_4 = blockIdx.x % NUM_SPLIT;
                kv_head_idx_4 = 0;
                kv_head_idx_4 = blockIdx.y;
                int schedule_batch_idx_4 = blockIdx.z;
                batch_idx_4 = schedule_batch_idx_4;
                {
                    batch_idx_4 = request_order[schedule_batch_idx_4];
                }
            }
            int visible_keys_4 = UNIFORM_KV_LEN - Q_LEN + q_row_idx_4 + 1;
            {
                visible_keys_4 = seq_lens_kv[batch_idx_4] - Q_LEN + q_row_idx_4 + 1;
            }
            visible_keys_4 = visible_keys_4 + 6 - 1;
            if (visible_keys_4 < 0) {
                visible_keys_4 = 0;
            }
            int peer_visible_4 = visible_keys_4;
            int _min_1 = (((peer_visible_4 + BLOCK_N - 1) / BLOCK_N) < (NUM_SPLIT) ? ((peer_visible_4 + BLOCK_N - 1) / BLOCK_N) : (NUM_SPLIT));
            part_count_4 = _min_1;
            int batch_idx_0 = batch_idx_4;
            int q_row_idx_1_1 = q_row_idx_4;
            int kv_head_idx_2_1 = kv_head_idx_4;
            int split_idx_l = split_idx_4;
            int part_count_l = part_count_4;
            int bundle_idx_l = bundle_idx_4;
            int bundle_item_idx_l = bundle_item_idx_4;
            unsigned int peer_tiles_4 = 0;
            if (split_idx_l < part_count_l) {
                peer_tiles_4 = total_tiles;
            }
            unsigned int _phase_q_empty_0 = 1;
            #pragma unroll 1
            for (unsigned int _tile_iter_l = 0; _tile_iter_l < peer_tiles_4; _tile_iter_l++) {
                int visible_keys_0_4 = UNIFORM_KV_LEN - Q_LEN + q_row_idx_1_1 + 1;
                {
                    visible_keys_0_4 = seq_lens_kv[batch_idx_0] - Q_LEN + q_row_idx_1_1 + 1;
                }
                visible_keys_0_4 = visible_keys_0_4 + 6 - 1;
                if (visible_keys_0_4 < 0) {
                    visible_keys_0_4 = 0;
                }
                int seqlen_kv = visible_keys_0_4;
                int peer_denominator_4 = part_count_l * BLOCK_N;
                int peer_blocks_4 = (seqlen_kv + peer_denominator_4 - 1) / peer_denominator_4;
                asm volatile("griddepcontrol.wait;" ::: "memory");
                mbarrier_wait(q_empty_addr, _phase_q_empty_0);
                _phase_q_empty_0 ^= 1;
                #pragma unroll 1
                for (int query_pad_col = 0; query_pad_col < 128; query_pad_col++) {
                    {
                        __nv_bfloat16 _bval_0 = __float2bfloat16_rn(0.0f);
                        uint16_t _bits_0 = *(uint16_t*)&_bval_0;
                        uint32_t _addr_0 = static_cast<uint32_t>((smem_qt_hi_addr + ((unsigned int)(query_pad_col / 64 * 16384) + (96 + lane) * 128 + (unsigned int)(query_pad_col % 64 * 2) ^ ((unsigned int)(query_pad_col / 64 * 16384) + (96 + lane) * 128 + (unsigned int)(query_pad_col % 64 * 2) >> 7 & 7) << 4)));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_0), "h"(_bits_0) : "memory");
                    }
                    {
                        __nv_bfloat16 _bval_1 = __float2bfloat16_rn(0.0f);
                        uint16_t _bits_1 = *(uint16_t*)&_bval_1;
                        uint32_t _addr_1 = static_cast<uint32_t>((smem_qt_lo_addr + ((unsigned int)(query_pad_col / 64 * 16384) + (96 + lane) * 128 + (unsigned int)(query_pad_col % 64 * 2) ^ ((unsigned int)(query_pad_col / 64 * 16384) + (96 + lane) * 128 + (unsigned int)(query_pad_col % 64 * 2) >> 7 & 7) << 4)));
                        asm volatile("st.shared.b16 [%0], %1;" :: "r"(_addr_1), "h"(_bits_1) : "memory");
                    }
                }
                asm volatile("fence.proxy.async;");
                __syncwarp();
                if (elect_sync()) {
                    int group_ratio_l = num_q_heads / (num_kv_heads * Q_GROUPS_PER_KV);
                    int off_qt = (batch_idx_0 * Q_LEN + q_row_idx_1_1) * num_q_heads + kv_head_idx_2_1 * group_ratio_l;
                    mbarrier_arrive_expect_tx(q_full_addr, TILE_Q * HEAD_DIM * 2);
                    #pragma unroll
                    for (int query_row_l = 0; query_row_l < 6; query_row_l++) {
                        #pragma unroll
                        for (int dim_chunk_l = 0; dim_chunk_l < 4; dim_chunk_l++) {
                            int query_dst_l = smem_qt_hi_addr;
                            if (dim_chunk_l >= 2) {
                                query_dst_l = smem_qt_lo_addr;
                            }
                            tma_3d_gmem2smem(query_dst_l + dim_chunk_l % 2 * 128 * 64 * 2 + query_row_l * 16 * 64 * 2, Qt, 0, off_qt + query_row_l * num_q_heads, dim_chunk_l, q_full_addr);
                        }
                    }
                    {
                        {
                            if (peer_blocks_4 > 0) {
                                mbarrier_wait(page_offsets_full_addr + (page_cons_stage) * 8, page_cons_phase);
                                int page_smem_base = page_cons_stage * 4;
                                #pragma unroll
                                for (int dim_half = 0; dim_half < 2; dim_half++) {
                                    mbarrier_wait(raw_kv_empty_addr + (raw_stage) * 8, raw_phase);
                                    mbarrier_arrive_expect_tx(raw_kv_full_addr + (raw_stage) * 8, BLOCK_N * HEAD_DIM_HALF);
                                    int raw_dst = smem_kv_fp8_addr + (unsigned int)(raw_stage * 16384);
                                    #pragma unroll
                                    for (int page_in_block = 0; page_in_block < 2; page_in_block++) {
                                        int physical_page = smem_page_offsets[page_smem_base + page_in_block];
                                        int page_dst = raw_dst + page_in_block * PAGE_SIZE * HEAD_DIM_HALF;
                                        {
                                            tma_5d_gmem2smem(page_dst, K, 0, 0, dim_half, kv_head_idx_2_1 / Q_GROUPS_PER_KV, physical_page, raw_kv_full_addr + (raw_stage) * 8);
                                        }
                                    }
                                    raw_stage += 1;
                                    if (raw_stage == 4) { raw_stage = 0; raw_phase ^= 1; }
                                }
                                page_cons_stage += 1;
                                if (page_cons_stage == 6) { page_cons_stage = 0; page_cons_phase ^= 1; }
                                #pragma unroll 1
                                for (int _body_n_l = 0; _body_n_l < peer_blocks_4 - 1; _body_n_l++) {
                                    mbarrier_wait(page_offsets_full_addr + (page_cons_stage) * 8, page_cons_phase);
                                    int page_smem_base_0 = page_cons_stage * 4;
                                    #pragma unroll
                                    for (int dim_half_1 = 0; dim_half_1 < 2; dim_half_1++) {
                                        mbarrier_wait(raw_kv_empty_addr + (raw_stage) * 8, raw_phase);
                                        mbarrier_arrive_expect_tx(raw_kv_full_addr + (raw_stage) * 8, BLOCK_N * HEAD_DIM_HALF);
                                        int raw_dst_1 = smem_kv_fp8_addr + (unsigned int)(raw_stage * 16384);
                                        #pragma unroll
                                        for (int page_in_block_1 = 0; page_in_block_1 < 2; page_in_block_1++) {
                                            int physical_page_1 = smem_page_offsets[page_smem_base_0 + page_in_block_1];
                                            int page_dst_1 = raw_dst_1 + page_in_block_1 * PAGE_SIZE * HEAD_DIM_HALF;
                                            {
                                                tma_5d_gmem2smem(page_dst_1, K, 0, 0, dim_half_1, kv_head_idx_2_1 / Q_GROUPS_PER_KV, physical_page_1, raw_kv_full_addr + (raw_stage) * 8);
                                            }
                                        }
                                        raw_stage += 1;
                                        if (raw_stage == 4) { raw_stage = 0; raw_phase ^= 1; }
                                    }
                                    page_cons_stage += 1;
                                    if (page_cons_stage == 6) { page_cons_stage = 0; page_cons_phase ^= 1; }
                                    int page_smem_base_1 = page_release_stage * 4;
                                    #pragma unroll
                                    for (int dim_half_2 = 0; dim_half_2 < 2; dim_half_2++) {
                                        mbarrier_wait(raw_kv_empty_addr + (raw_stage) * 8, raw_phase);
                                        mbarrier_arrive_expect_tx(raw_kv_full_addr + (raw_stage) * 8, BLOCK_N * HEAD_DIM_HALF);
                                        int raw_dst_2 = smem_kv_fp8_addr + (unsigned int)(raw_stage * 16384);
                                        #pragma unroll
                                        for (int page_in_block_2 = 0; page_in_block_2 < 2; page_in_block_2++) {
                                            int physical_page_2 = smem_page_offsets[page_smem_base_1 + 2 + page_in_block_2];
                                            int page_dst_2 = raw_dst_2 + page_in_block_2 * PAGE_SIZE * HEAD_DIM_HALF;
                                            {
                                                tma_5d_gmem2smem(page_dst_2, V, 0, 0, dim_half_2, kv_head_idx_2_1 / Q_GROUPS_PER_KV, physical_page_2, raw_kv_full_addr + (raw_stage) * 8);
                                            }
                                        }
                                        raw_stage += 1;
                                        if (raw_stage == 4) { raw_stage = 0; raw_phase ^= 1; }
                                    }
                                    mbarrier_arrive(page_offsets_empty_addr + (page_release_stage) * 8);
                                    page_release_stage += 1;
                                    if (page_release_stage == 6) { page_release_stage = 0; page_release_phase ^= 1; }
                                }
                                int page_smem_base_0_1 = page_release_stage * 4;
                                #pragma unroll
                                for (int dim_half_3 = 0; dim_half_3 < 2; dim_half_3++) {
                                    mbarrier_wait(raw_kv_empty_addr + (raw_stage) * 8, raw_phase);
                                    mbarrier_arrive_expect_tx(raw_kv_full_addr + (raw_stage) * 8, BLOCK_N * HEAD_DIM_HALF);
                                    int raw_dst_3 = smem_kv_fp8_addr + (unsigned int)(raw_stage * 16384);
                                    #pragma unroll
                                    for (int page_in_block_3 = 0; page_in_block_3 < 2; page_in_block_3++) {
                                        int physical_page_3 = smem_page_offsets[page_smem_base_0_1 + 2 + page_in_block_3];
                                        int page_dst_3 = raw_dst_3 + page_in_block_3 * PAGE_SIZE * HEAD_DIM_HALF;
                                        {
                                            tma_5d_gmem2smem(page_dst_3, V, 0, 0, dim_half_3, kv_head_idx_2_1 / Q_GROUPS_PER_KV, physical_page_3, raw_kv_full_addr + (raw_stage) * 8);
                                        }
                                    }
                                    raw_stage += 1;
                                    if (raw_stage == 4) { raw_stage = 0; raw_phase ^= 1; }
                                }
                                mbarrier_arrive(page_offsets_empty_addr + (page_release_stage) * 8);
                                page_release_stage += 1;
                                if (page_release_stage == 6) { page_release_stage = 0; page_release_phase ^= 1; }
                            }
                        }
                    }
                }
            }
        }
    }
    // ---- Role: transform ----
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 88;");
        { // transform_main
            unsigned int work_stage_t = 0;
            int raw_stage_t = 0;
            int raw_phase_t = 0;
            int transformed_stage_t = 0;
            int transformed_phase_t = 1;
            int batch_idx_5 = 0;
            int q_row_idx_5 = 0;
            int kv_head_idx_5 = 0;
            int split_idx_5 = 0;
            int part_count_5 = NUM_SPLIT;
            int bundle_idx_5 = 0;
            int bundle_item_idx_5 = 0;
            {
                q_row_idx_5 = blockIdx.x / NUM_SPLIT * 6;
                split_idx_5 = blockIdx.x % NUM_SPLIT;
                kv_head_idx_5 = 0;
                kv_head_idx_5 = blockIdx.y;
                int schedule_batch_idx_5 = blockIdx.z;
                batch_idx_5 = schedule_batch_idx_5;
                {
                    batch_idx_5 = request_order[schedule_batch_idx_5];
                }
            }
            int visible_keys_5 = UNIFORM_KV_LEN - Q_LEN + q_row_idx_5 + 1;
            {
                visible_keys_5 = seq_lens_kv[batch_idx_5] - Q_LEN + q_row_idx_5 + 1;
            }
            visible_keys_5 = visible_keys_5 + 6 - 1;
            if (visible_keys_5 < 0) {
                visible_keys_5 = 0;
            }
            int peer_visible_5 = visible_keys_5;
            int _min_2 = (((peer_visible_5 + BLOCK_N - 1) / BLOCK_N) < (NUM_SPLIT) ? ((peer_visible_5 + BLOCK_N - 1) / BLOCK_N) : (NUM_SPLIT));
            part_count_5 = _min_2;
            int batch_idx_t = batch_idx_5;
            int q_row_idx_t = q_row_idx_5;
            int kv_head_idx_t = kv_head_idx_5;
            int split_idx_t = split_idx_5;
            int part_count_t = part_count_5;
            int bundle_idx_t = bundle_idx_5;
            int bundle_item_idx_t = bundle_item_idx_5;
            unsigned int peer_tiles_5 = 0;
            if (split_idx_t < part_count_t) {
                peer_tiles_5 = total_tiles;
            }
            #pragma unroll 1
            for (unsigned int _tile_iter_t = 0; _tile_iter_t < peer_tiles_5; _tile_iter_t++) {
                int visible_keys_0_5 = UNIFORM_KV_LEN - Q_LEN + q_row_idx_t + 1;
                {
                    visible_keys_0_5 = seq_lens_kv[batch_idx_t] - Q_LEN + q_row_idx_t + 1;
                }
                visible_keys_0_5 = visible_keys_0_5 + 6 - 1;
                if (visible_keys_0_5 < 0) {
                    visible_keys_0_5 = 0;
                }
                int seqlen_kv_t = visible_keys_0_5;
                int peer_denominator_5 = part_count_t * BLOCK_N;
                int peer_blocks_5 = (seqlen_kv_t + peer_denominator_5 - 1) / peer_denominator_5;
                int total_half_items = peer_blocks_5 * 4;
                #pragma unroll 1
                for (int _item = 0; _item < total_half_items; _item++) {
                    mbarrier_wait(raw_kv_full_addr + (raw_stage_t) * 8, raw_phase_t);
                    mbarrier_wait(kv_empty_addr + (transformed_stage_t) * 8, transformed_phase_t);
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    {
                        const char* _src_ptr = smem_raw + (smem_kv_fp8_addr + (unsigned int)(raw_stage_t * 16384) - smem);
                        char* _dst_ptr = smem_raw + (smem_kv_addr + (unsigned int)(transformed_stage_t * 32768) - smem);
                        const int _tid = (int)threadIdx.x - (12) * 32;
                        uint64_t _src_buf[16];
                        #pragma unroll
                        for (int _outer = 0; _outer < 2; ++_outer) {
                            #pragma unroll
                            for (int _base = _outer * 1024; _base < (_outer + 1) * 1024; _base += 128) {
                                int _off = _base + _tid;
                                _src_buf[_base >> 7] = reinterpret_cast<const uint64_t*>(_src_ptr)[_off];
                            }
                            #pragma unroll
                            for (int _base = _outer * 1024; _base < (_outer + 1) * 1024; _base += 128) {
                                int _off = _base + _tid;
                                uint64_t _src64 = _src_buf[_base >> 7];
                                uint32_t _out_x16x2[4];
                                #pragma unroll
                                for (int _cv = 0; _cv < 4; ++_cv) {
                                    uint16_t _e4m3x2 = (uint16_t)((_src64 >> (_cv * 16)) & 0xFFFFull);
                                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_out_x16x2[_cv]) : "h"(_e4m3x2));
                                    #else
                                    uint32_t _f16x2;
                                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"(_e4m3x2));
                                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                                    float _f0;
                                    float _f1;
                                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_out_x16x2[_cv]) : "f"(_f1), "f"(_f0));
                                    #endif
                                }
                                uint4 _dst4 = make_uint4(_out_x16x2[0], _out_x16x2[1], _out_x16x2[2], _out_x16x2[3]);
                                int _elt = _off * 8;
                                int _row = (((_elt % 128) / 64) * 128) + (_elt / 128);
                                int _byte_off = (_row * 128) + (((_elt % 64) * 16) / 8);
                                int _swz_off = _byte_off ^ ((_row % 8) * 16);
                                *reinterpret_cast<uint4*>(_dst_ptr + _swz_off) = _dst4;
                            }
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    }
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(raw_kv_empty_addr + (raw_stage_t) * 8);
                        mbarrier_arrive(kv_full_addr + (transformed_stage_t) * 8);
                    }
                    raw_stage_t += 1;
                    if (raw_stage_t == 4) { raw_stage_t = 0; raw_phase_t ^= 1; }
                    transformed_stage_t += 1;
                    if (transformed_stage_t == 2) { transformed_stage_t = 0; transformed_phase_t ^= 1; }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
