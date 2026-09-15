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
#define TMEM_SCORES_OFFSET 0
#define TMEM_PROBABILITIES_OFFSET 32
#define TMEM_OUTPUT_HI_OFFSET 256
#define TMEM_OUTPUT_LO_OFFSET 384
#define NUM_Q_PIPE_STAGES 2
#define NUM_KV_PIPE_STAGES 9
#define NUM_PG_PIPE_STAGES 6
#define NUM_WORK_PIPE_STAGES 2
#define NUM_THROTTLE_PIPE_STAGES 2
#define NUM_SCORE_PIPE_STAGES 2
#define NUM_STAT_PIPE_STAGES 2
#define SMEM_SMEM_QT_OFF 0
#define SMEM_SMEM_QT_STAGE_BYTES 16384
#define SMEM_SMEM_QT_STRIDE 32768
#define SMEM_SMEM_QT_SECOND_OFF 16384
#define SMEM_SMEM_QT_SECOND_STAGE_BYTES 16384
#define SMEM_SMEM_QT_SECOND_STRIDE 32768
#define SMEM_SMEM_KV_OFF 65536
#define SMEM_SMEM_KV_STAGE_BYTES 16384
#define SMEM_SMEM_KV_STRIDE 16384
#define SMEM_SMEM_V_OFF 65536
#define SMEM_SMEM_V_STAGE_BYTES 16384
#define SMEM_SMEM_V_STRIDE 16384
#define SMEM_SMEM_PG_OFF 212992
#define SMEM_SMEM_PG_STAGE_BYTES 128
#define SMEM_SMEM_PG_STRIDE 128
#define SMEM_WORK_RESPONSE_OFF 213760
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_TOTAL 214352
#define THREADS 384
#define BATCH_SIZE 128
#define NUM_Q_HEADS 32
#define NUM_KV_HEADS 2
#define Q_LEN 6
#define BLOCK_N 128
#define HEAD_DIM 256
#define RETURN_LSE 0
#define ENABLE_PDL 0

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
        "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, %3, p;\n\t"
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
        "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], [%1], db, %4, p;\n\t"
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


__device__ __forceinline__ void tmem_st_x32(int tmem_addr, uint32_t* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x32.b32"
        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8,"
        "  %9, %10, %11, %12, %13, %14, %15, %16,"
        "  %17, %18, %19, %20, %21, %22, %23, %24,"
        "  %25, %26, %27, %28, %29, %30, %31, %32};"
        :: "r"(tmem_addr),
           "r"(src[0]),  "r"(src[1]),  "r"(src[2]),  "r"(src[3]),
           "r"(src[4]),  "r"(src[5]),  "r"(src[6]),  "r"(src[7]),
           "r"(src[8]),  "r"(src[9]),  "r"(src[10]), "r"(src[11]),
           "r"(src[12]), "r"(src[13]), "r"(src[14]), "r"(src[15]),
           "r"(src[16]), "r"(src[17]), "r"(src[18]), "r"(src[19]),
           "r"(src[20]), "r"(src[21]), "r"(src[22]), "r"(src[23]),
           "r"(src[24]), "r"(src[25]), "r"(src[26]), "r"(src[27]),
           "r"(src[28]), "r"(src[29]), "r"(src[30]), "r"(src[31]));
}


__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}


__device__ __forceinline__ float max_noftz(float a, float b) {
    float c;
    asm("max.f32 %0, %1, %2;" : "=f"(c) : "f"(a), "f"(b));
    return c;
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


__device__ __forceinline__ void tmem_ld_x8(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3]),
          "=f"(dst[4]), "=f"(dst[5]), "=f"(dst[6]), "=f"(dst[7])
        : "r"(tmem_addr));
}


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(384, 1) void
kernel_cake_fmha_request_ordered_paged_decode_fp8q_fbb3f38bece34f929238(CakeTensorMap const* Qt, CakeTensorMap const* K, CakeTensorMap const* V, __nv_bfloat16* __restrict__ O, float* __restrict__ LSE, int* __restrict__ request_order, int use_request_order, int* __restrict__ page_table, int* __restrict__ seq_lens_kv, float* __restrict__ bmm1_scale_ptr, float* __restrict__ bmm2_scale_ptr, float* __restrict__ partial_O, float* __restrict__ partial_max, float* __restrict__ partial_sum, int pt_batch_stride, int pt_v_offset, int bmm1_is_log2, int num_splits, int blocks_per_split)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem + 213872;
    #define q_full_addr (mbar_base + 16)
    #define q_empty_addr (mbar_base + 32)
    #define kv_full_addr (mbar_base + 48)
    #define kv_empty_addr (mbar_base + 120)
    #define pg_full_addr (mbar_base + 192)
    #define pg_empty_addr (mbar_base + 240)
    #define work_full_addr (mbar_base + 288)
    #define work_empty_addr (mbar_base + 304)
    #define throttle_full_addr (mbar_base + 320)
    #define throttle_empty_addr (mbar_base + 336)
    #define s_full_addr (mbar_base + 352)
    #define s_empty_addr (mbar_base + 368)
    #define stats_full_addr (mbar_base + 384)
    #define stats_empty_addr (mbar_base + 400)
    #define p_full_reserved_addr (mbar_base + 432)
    #define p_empty_reserved_addr (mbar_base + 448)
    #define o_full_addr (mbar_base + 464)
    #define o_empty_addr (mbar_base + 472)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(Qt)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(K)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(V)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    uint8_t* smem_qt = reinterpret_cast<uint8_t*>(smem_raw + 0);
    const int smem_qt_addr = smem + 0;
    uint8_t* smem_qt_second = reinterpret_cast<uint8_t*>(smem_raw + 16384);
    const int smem_qt_second_addr = smem + 16384;
    uint8_t* smem_kv = reinterpret_cast<uint8_t*>(smem_raw + 65536);
    const int smem_kv_addr = smem + 65536;
    uint8_t* smem_v = reinterpret_cast<uint8_t*>(smem_raw + 65536);
    const int smem_v_addr = smem + 65536;
    int* smem_pg = reinterpret_cast<int*>(smem_raw + 212992);
    const int smem_pg_addr = smem + 212992;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 213760);
    const int work_response_addr = smem + 213760;
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(Qt)) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(K)) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(V)) : "memory");

    // Mbarrier init (18 pipeline groups, 0 ordered-sequence groups, 56 barriers)
    // Mbarriers at smem_raw[213872..214352)

    if (warp == 11) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 2 barriers, init_count=1
            mbarrier_init(smem + 213888, 1);
            mbarrier_init(smem + 213896, 1);
            // q_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 213904, 1);
            mbarrier_init(smem + 213912, 1);
            // kv_full: 9 barriers, init_count=1
            mbarrier_init(smem + 213920, 1);
            mbarrier_init(smem + 213928, 1);
            mbarrier_init(smem + 213936, 1);
            mbarrier_init(smem + 213944, 1);
            mbarrier_init(smem + 213952, 1);
            mbarrier_init(smem + 213960, 1);
            mbarrier_init(smem + 213968, 1);
            mbarrier_init(smem + 213976, 1);
            mbarrier_init(smem + 213984, 1);
            // kv_empty: 9 barriers, init_count=1
            mbarrier_init(smem + 213992, 1);
            mbarrier_init(smem + 214000, 1);
            mbarrier_init(smem + 214008, 1);
            mbarrier_init(smem + 214016, 1);
            mbarrier_init(smem + 214024, 1);
            mbarrier_init(smem + 214032, 1);
            mbarrier_init(smem + 214040, 1);
            mbarrier_init(smem + 214048, 1);
            mbarrier_init(smem + 214056, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }
    if (warp == 9) {
        uint32_t leader = elect_sync();
        if (leader) {
            // pg_full: 6 barriers, init_count=32
            mbarrier_init(smem + 214064, 32);
            mbarrier_init(smem + 214072, 32);
            mbarrier_init(smem + 214080, 32);
            mbarrier_init(smem + 214088, 32);
            mbarrier_init(smem + 214096, 32);
            mbarrier_init(smem + 214104, 32);
            // pg_empty: 6 barriers, init_count=1
            mbarrier_init(smem + 214112, 1);
            mbarrier_init(smem + 214120, 1);
            mbarrier_init(smem + 214128, 1);
            mbarrier_init(smem + 214136, 1);
            mbarrier_init(smem + 214144, 1);
            mbarrier_init(smem + 214152, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }
    if (warp == 10) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'work_pipe' ---
            // work_full: 2 barriers, init_count=1
            mbarrier_init(smem + 214160, 1);
            mbarrier_init(smem + 214168, 1);
            // work_empty: 2 barriers, init_count=384
            mbarrier_init(smem + 214176, 384);
            mbarrier_init(smem + 214184, 384);
            // --- pipeline 'throttle_pipe' ---
            // throttle_full: 2 barriers, init_count=32
            mbarrier_init(smem + 214192, 32);
            mbarrier_init(smem + 214200, 32);
            // throttle_empty: 2 barriers, init_count=32
            mbarrier_init(smem + 214208, 32);
            mbarrier_init(smem + 214216, 32);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }
    if (warp == 1) {
        uint32_t leader = elect_sync();
        if (leader) {
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 214224, 1);
            mbarrier_init(smem + 214232, 1);
            // s_empty: 2 barriers, init_count=128
            mbarrier_init(smem + 214240, 128);
            mbarrier_init(smem + 214248, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }
    if (warp == 5) {
        uint32_t leader = elect_sync();
        if (leader) {
            // stats_full: 2 barriers, init_count=128
            mbarrier_init(smem + 214256, 128);
            mbarrier_init(smem + 214264, 128);
            // stats_empty: 2 barriers, init_count=128
            mbarrier_init(smem + 214272, 128);
            mbarrier_init(smem + 214280, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }
    if (warp == 2) {
        uint32_t leader = elect_sync();
        if (leader) {
            // p_full_reserved: 2 barriers, init_count=128
            mbarrier_init(smem + 214304, 128);
            mbarrier_init(smem + 214312, 128);
            // p_empty_reserved: 2 barriers, init_count=32
            mbarrier_init(smem + 214320, 32);
            mbarrier_init(smem + 214328, 32);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }
    if (warp == 4) {
        uint32_t leader = elect_sync();
        if (leader) {
            // o_full: 1 barriers, init_count=1
            mbarrier_init(smem + 214336, 1);
            // o_empty: 1 barriers, init_count=128
            mbarrier_init(smem + 214344, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 213872);
    if (warp == 0) {
        int _tmem_hold = smem + 213872;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_scores = taddr;
    const int tmem_probabilities = taddr + 32;
    const int tmem_output_hi = taddr + 256;
    const int tmem_output_lo = taddr + 384;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 112;");
    }

    // ---- Role: softmax ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 200;");
        { // softmax_main
            float scale_s = bmm1_scale_ptr[0] * ((bmm1_is_log2 != 0) ? 1.0f : 1.4426950408889634f);
            int logical_row_s = warp % 4 * 32 + lane;
            int first_col_s = 0;
            int score_slot_s = 0;
            int score_phase_s = 0;
            int stats_slot_s = 0;
            int stats_phase_s = 1;
            unsigned int work_stage_s = 0;
            unsigned int tile_idx_s = blockIdx.z;
            float sv[128];
            float stats_s[2];
            unsigned int _phase_work_full = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_s = 0; _tile_iter_s < BATCH_SIZE * NUM_KV_HEADS; _tile_iter_s++) {
                int batch_s = tile_idx_s / (unsigned int)NUM_KV_HEADS;
                if (use_request_order != 0) {
                    batch_s = request_order[batch_s];
                }
                int kv_len_s = seq_lens_kv[batch_s];
                int n_blocks_s = (kv_len_s + 127) / 128;
                float row_max_s = -3.4028235e+38f;
                float row_sum_s = 0.0f;
                mbarrier_wait(stats_empty_addr + (stats_slot_s) * 8, stats_phase_s);
                #pragma unroll 1
                for (int block_s = 0; block_s < n_blocks_s; block_s++) {
                    mbarrier_wait(s_full_addr + (score_slot_s) * 8, score_phase_s);
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(sv[0]), "=f"(sv[1]), "=f"(sv[2]), "=f"(sv[3]), "=f"(sv[4]), "=f"(sv[5]), "=f"(sv[6]), "=f"(sv[7]), "=f"(sv[8]), "=f"(sv[9]), "=f"(sv[10]), "=f"(sv[11]), "=f"(sv[12]), "=f"(sv[13]), "=f"(sv[14]), "=f"(sv[15]), "=f"(sv[16]), "=f"(sv[17]), "=f"(sv[18]), "=f"(sv[19]), "=f"(sv[20]), "=f"(sv[21]), "=f"(sv[22]), "=f"(sv[23]), "=f"(sv[24]), "=f"(sv[25]), "=f"(sv[26]), "=f"(sv[27]), "=f"(sv[28]), "=f"(sv[29]), "=f"(sv[30]), "=f"(sv[31])
                        : "r"(taddr + (unsigned int)(score_slot_s * 128)));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(sv[32]), "=f"(sv[33]), "=f"(sv[34]), "=f"(sv[35]), "=f"(sv[36]), "=f"(sv[37]), "=f"(sv[38]), "=f"(sv[39]), "=f"(sv[40]), "=f"(sv[41]), "=f"(sv[42]), "=f"(sv[43]), "=f"(sv[44]), "=f"(sv[45]), "=f"(sv[46]), "=f"(sv[47]), "=f"(sv[48]), "=f"(sv[49]), "=f"(sv[50]), "=f"(sv[51]), "=f"(sv[52]), "=f"(sv[53]), "=f"(sv[54]), "=f"(sv[55]), "=f"(sv[56]), "=f"(sv[57]), "=f"(sv[58]), "=f"(sv[59]), "=f"(sv[60]), "=f"(sv[61]), "=f"(sv[62]), "=f"(sv[63])
                        : "r"(taddr + (unsigned int)(score_slot_s * 128) + 32));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(sv[64]), "=f"(sv[65]), "=f"(sv[66]), "=f"(sv[67]), "=f"(sv[68]), "=f"(sv[69]), "=f"(sv[70]), "=f"(sv[71]), "=f"(sv[72]), "=f"(sv[73]), "=f"(sv[74]), "=f"(sv[75]), "=f"(sv[76]), "=f"(sv[77]), "=f"(sv[78]), "=f"(sv[79]), "=f"(sv[80]), "=f"(sv[81]), "=f"(sv[82]), "=f"(sv[83]), "=f"(sv[84]), "=f"(sv[85]), "=f"(sv[86]), "=f"(sv[87]), "=f"(sv[88]), "=f"(sv[89]), "=f"(sv[90]), "=f"(sv[91]), "=f"(sv[92]), "=f"(sv[93]), "=f"(sv[94]), "=f"(sv[95])
                        : "r"(taddr + (unsigned int)(score_slot_s * 128) + 64));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(sv[96]), "=f"(sv[97]), "=f"(sv[98]), "=f"(sv[99]), "=f"(sv[100]), "=f"(sv[101]), "=f"(sv[102]), "=f"(sv[103]), "=f"(sv[104]), "=f"(sv[105]), "=f"(sv[106]), "=f"(sv[107]), "=f"(sv[108]), "=f"(sv[109]), "=f"(sv[110]), "=f"(sv[111]), "=f"(sv[112]), "=f"(sv[113]), "=f"(sv[114]), "=f"(sv[115]), "=f"(sv[116]), "=f"(sv[117]), "=f"(sv[118]), "=f"(sv[119]), "=f"(sv[120]), "=f"(sv[121]), "=f"(sv[122]), "=f"(sv[123]), "=f"(sv[124]), "=f"(sv[125]), "=f"(sv[126]), "=f"(sv[127])
                        : "r"(taddr + (unsigned int)(score_slot_s * 128) + 96));
                    float old_max_s = row_max_s;
                    float max0_s = row_max_s;
                    float max1_s = row_max_s;
                    float max2_s = row_max_s;
                    float max3_s = row_max_s;
                    if (block_s * 128 + 128 > kv_len_s - Q_LEN) {
                        int causal_s = kv_len_s - Q_LEN + logical_row_s / 16;
                        #pragma unroll
                        for (int j_s = 0; j_s < 128; j_s++) {
                            if (causal_s < block_s * 128 + first_col_s + j_s) {
                                sv[j_s] = -3.4028235e+38f;
                            }
                        }
                    }
                    #pragma unroll
                    for (int j_s_1 = 0; j_s_1 < 128; j_s_1 += 4) {
                        float _max_0 = max_noftz(max0_s, sv[j_s_1]);
                        max0_s = _max_0;
                        float _max_1 = max_noftz(max1_s, sv[j_s_1 + 1]);
                        max1_s = _max_1;
                        float _max_2 = max_noftz(max2_s, sv[j_s_1 + 2]);
                        max2_s = _max_2;
                        float _max_3 = max_noftz(max3_s, sv[j_s_1 + 3]);
                        max3_s = _max_3;
                    }
                    float _max_4 = max_noftz(max0_s, max2_s);
                    float _max_5 = max_noftz(max1_s, max3_s);
                    float _max_6 = max_noftz(_max_4, _max_5);
                    row_max_s = _max_6;
                    stats_s[0] = old_max_s;
                    stats_s[1] = row_max_s;
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x2.b32"
                        " [%0], {%1, %2};"
                        :: "r"(taddr + (unsigned int)(stats_slot_s * 128)), "f"(stats_s[0]), "f"(stats_s[1]));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(stats_full_addr + (stats_slot_s) * 8);
                    stats_slot_s += 1;
                    if (stats_slot_s == 2) { stats_slot_s = 0; stats_phase_s ^= 1; }
                    float negmax_s = (-row_max_s) * scale_s + 8.8073549f;
                    const float2 _scale2_0 = {scale_s, scale_s};
                    #pragma unroll
                    for (int _ls = 0; _ls < 64; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(sv)[_ls], _scale2_0);
                    const float2 _add2_1 = {negmax_s, negmax_s};
                    #pragma unroll
                    for (int _la = 0; _la < 64; _la++)
                        add_f32x2_inplace(&reinterpret_cast<float2*>(sv)[_la], _add2_1);
                    #pragma unroll
                    for (int _le = 0; _le < 128; _le++) {
                        sv[_le] = approx_exp2(sv[_le]);
                    }
                    uint32_t _fp8_0[32];
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
                            : "=r"(_packed) : "f"(sv[4]), "f"(sv[5]),
                                               "f"(sv[6]), "f"(sv[7]));
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
                            : "=r"(_packed) : "f"(sv[8]), "f"(sv[9]),
                                               "f"(sv[10]), "f"(sv[11]));
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
                            : "=r"(_packed) : "f"(sv[12]), "f"(sv[13]),
                                               "f"(sv[14]), "f"(sv[15]));
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
                            : "=r"(_packed) : "f"(sv[16]), "f"(sv[17]),
                                               "f"(sv[18]), "f"(sv[19]));
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
                            : "=r"(_packed) : "f"(sv[20]), "f"(sv[21]),
                                               "f"(sv[22]), "f"(sv[23]));
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
                            : "=r"(_packed) : "f"(sv[24]), "f"(sv[25]),
                                               "f"(sv[26]), "f"(sv[27]));
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
                            : "=r"(_packed) : "f"(sv[28]), "f"(sv[29]),
                                               "f"(sv[30]), "f"(sv[31]));
                        _fp8_0[7] = _packed;
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
                            : "=r"(_packed) : "f"(sv[32]), "f"(sv[33]),
                                               "f"(sv[34]), "f"(sv[35]));
                        _fp8_0[8] = _packed;
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
                            : "=r"(_packed) : "f"(sv[36]), "f"(sv[37]),
                                               "f"(sv[38]), "f"(sv[39]));
                        _fp8_0[9] = _packed;
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
                            : "=r"(_packed) : "f"(sv[40]), "f"(sv[41]),
                                               "f"(sv[42]), "f"(sv[43]));
                        _fp8_0[10] = _packed;
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
                            : "=r"(_packed) : "f"(sv[44]), "f"(sv[45]),
                                               "f"(sv[46]), "f"(sv[47]));
                        _fp8_0[11] = _packed;
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
                            : "=r"(_packed) : "f"(sv[48]), "f"(sv[49]),
                                               "f"(sv[50]), "f"(sv[51]));
                        _fp8_0[12] = _packed;
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
                            : "=r"(_packed) : "f"(sv[52]), "f"(sv[53]),
                                               "f"(sv[54]), "f"(sv[55]));
                        _fp8_0[13] = _packed;
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
                            : "=r"(_packed) : "f"(sv[56]), "f"(sv[57]),
                                               "f"(sv[58]), "f"(sv[59]));
                        _fp8_0[14] = _packed;
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
                            : "=r"(_packed) : "f"(sv[60]), "f"(sv[61]),
                                               "f"(sv[62]), "f"(sv[63]));
                        _fp8_0[15] = _packed;
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
                            : "=r"(_packed) : "f"(sv[64]), "f"(sv[65]),
                                               "f"(sv[66]), "f"(sv[67]));
                        _fp8_0[16] = _packed;
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
                            : "=r"(_packed) : "f"(sv[68]), "f"(sv[69]),
                                               "f"(sv[70]), "f"(sv[71]));
                        _fp8_0[17] = _packed;
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
                            : "=r"(_packed) : "f"(sv[72]), "f"(sv[73]),
                                               "f"(sv[74]), "f"(sv[75]));
                        _fp8_0[18] = _packed;
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
                            : "=r"(_packed) : "f"(sv[76]), "f"(sv[77]),
                                               "f"(sv[78]), "f"(sv[79]));
                        _fp8_0[19] = _packed;
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
                            : "=r"(_packed) : "f"(sv[80]), "f"(sv[81]),
                                               "f"(sv[82]), "f"(sv[83]));
                        _fp8_0[20] = _packed;
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
                            : "=r"(_packed) : "f"(sv[84]), "f"(sv[85]),
                                               "f"(sv[86]), "f"(sv[87]));
                        _fp8_0[21] = _packed;
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
                            : "=r"(_packed) : "f"(sv[88]), "f"(sv[89]),
                                               "f"(sv[90]), "f"(sv[91]));
                        _fp8_0[22] = _packed;
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
                            : "=r"(_packed) : "f"(sv[92]), "f"(sv[93]),
                                               "f"(sv[94]), "f"(sv[95]));
                        _fp8_0[23] = _packed;
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
                            : "=r"(_packed) : "f"(sv[96]), "f"(sv[97]),
                                               "f"(sv[98]), "f"(sv[99]));
                        _fp8_0[24] = _packed;
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
                            : "=r"(_packed) : "f"(sv[100]), "f"(sv[101]),
                                               "f"(sv[102]), "f"(sv[103]));
                        _fp8_0[25] = _packed;
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
                            : "=r"(_packed) : "f"(sv[104]), "f"(sv[105]),
                                               "f"(sv[106]), "f"(sv[107]));
                        _fp8_0[26] = _packed;
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
                            : "=r"(_packed) : "f"(sv[108]), "f"(sv[109]),
                                               "f"(sv[110]), "f"(sv[111]));
                        _fp8_0[27] = _packed;
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
                            : "=r"(_packed) : "f"(sv[112]), "f"(sv[113]),
                                               "f"(sv[114]), "f"(sv[115]));
                        _fp8_0[28] = _packed;
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
                            : "=r"(_packed) : "f"(sv[116]), "f"(sv[117]),
                                               "f"(sv[118]), "f"(sv[119]));
                        _fp8_0[29] = _packed;
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
                            : "=r"(_packed) : "f"(sv[120]), "f"(sv[121]),
                                               "f"(sv[122]), "f"(sv[123]));
                        _fp8_0[30] = _packed;
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
                            : "=r"(_packed) : "f"(sv[124]), "f"(sv[125]),
                                               "f"(sv[126]), "f"(sv[127]));
                        _fp8_0[31] = _packed;
                    }
                    tmem_st_x32(taddr + (unsigned int)(score_slot_s * 128) + 32, _fp8_0);
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_wait(stats_empty_addr + (stats_slot_s) * 8, stats_phase_s);
                    if (n_blocks_s > block_s + 1) {
                        mbarrier_arrive(s_empty_addr + (score_slot_s) * 8);
                    }
                    float sum_s[8];
                    #pragma unroll
                    for (int j_s_2 = 0; j_s_2 < 8; j_s_2++) {
                        sum_s[j_s_2] = 0.0f;
                    }
                    #pragma unroll
                    for (int j_s_3 = 0; j_s_3 < 128; j_s_3 += 8) {
                        #pragma unroll
                        for (int k_s = 0; k_s < 8; k_s++) {
                            sum_s[k_s] = sum_s[k_s] + sv[j_s_3 + k_s];
                        }
                    }
                    float local_sum_s = sum_s[0] + sum_s[2] + (sum_s[4] + sum_s[6]) + (sum_s[1] + sum_s[3] + (sum_s[5] + sum_s[7]));
                    float rescale_s = 1.0f;
                    if (old_max_s != row_max_s) {
                        float _exp2_0 = approx_exp2(scale_s * (old_max_s - row_max_s));
                        rescale_s = _exp2_0;
                    }
                    row_sum_s = rescale_s * row_sum_s + local_sum_s;
                    if (block_s + 1 == n_blocks_s) {
                        stats_s[0] = row_sum_s;
                        stats_s[1] = row_max_s;
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x2.b32"
                            " [%0], {%1, %2};"
                            :: "r"(taddr + (unsigned int)(stats_slot_s * 128)), "f"(stats_s[0]), "f"(stats_s[1]));
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive(stats_full_addr + (stats_slot_s) * 8);
                        stats_slot_s += 1;
                        if (stats_slot_s == 2) { stats_slot_s = 0; stats_phase_s ^= 1; }
                        mbarrier_wait(stats_empty_addr + (stats_slot_s) * 8, stats_phase_s);
                        mbarrier_arrive(stats_full_addr + (stats_slot_s) * 8);
                        stats_slot_s += 1;
                        if (stats_slot_s == 2) { stats_slot_s = 0; stats_phase_s ^= 1; }
                        mbarrier_arrive(s_empty_addr + (score_slot_s) * 8);
                    }
                    score_slot_s += 1;
                    if (score_slot_s == 2) { score_slot_s = 0; score_phase_s ^= 1; }
                }
                mbarrier_wait(work_full_addr + (work_stage_s) * 8, _phase_work_full);
                uint32_t _clc_valid_4 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_4)
                    : "r"(work_response_addr + work_stage_s * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_4 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_4)
                    : "r"(work_response_addr + work_stage_s * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage_s) * 8);
                work_stage_s += 1;
                if (work_stage_s == 2) { work_stage_s = 0; _phase_work_full ^= 1; }
                if (_clc_valid_4 == 0) {
                    break;
                }
                tile_idx_s = _clc_ctaid_4;
            }
        }
    }
    // ---- Role: correction ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 192;");
        { // correction_main
            float scale_c = bmm1_scale_ptr[0] * ((bmm1_is_log2 != 0) ? 1.0f : 1.4426950408889634f);
            float output_scale_c = bmm2_scale_ptr[0];
            int logical_row_c = warp % 4 * 32 + lane;
            int first_col_c = 0;
            int stat_slot_c = 0;
            int stat_phase_c = 0;
            unsigned int work_stage_c = 0;
            unsigned int tile_idx_c = blockIdx.z;
            float stat_c[2];
            float corr_c[128];
            float out_c[8];
            unsigned int _phase_o_full_0 = 0;
            unsigned int _phase_work_full_1 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_c = 0; _tile_iter_c < BATCH_SIZE * NUM_KV_HEADS; _tile_iter_c++) {
                int batch_c = tile_idx_c / (unsigned int)NUM_KV_HEADS;
                int kv_head_c = tile_idx_c % (unsigned int)NUM_KV_HEADS;
                if (use_request_order != 0) {
                    batch_c = request_order[batch_c];
                }
                int kv_len_c = seq_lens_kv[batch_c];
                int n_blocks_c = (kv_len_c + 127) / 128;
                mbarrier_wait(stats_full_addr + (stat_slot_c) * 8, stat_phase_c);
                mbarrier_arrive(stats_empty_addr + (stat_slot_c) * 8);
                stat_slot_c += 1;
                if (stat_slot_c == 2) { stat_slot_c = 0; stat_phase_c ^= 1; }
                #pragma unroll 1
                for (int block_c = 0; block_c < n_blocks_c - 1; block_c++) {
                    mbarrier_wait(stats_full_addr + (stat_slot_c) * 8, stat_phase_c);
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x2.b32"
                        " {%0, %1}, [%2];"
                        : "=f"(stat_c[0]), "=f"(stat_c[1])
                        : "r"(taddr + (unsigned int)(stat_slot_c * 128)));
                    mbarrier_arrive(stats_empty_addr + (stat_slot_c) * 8);
                    stat_slot_c += 1;
                    if (stat_slot_c == 2) { stat_slot_c = 0; stat_phase_c ^= 1; }
                    mbarrier_wait(o_full_addr, _phase_o_full_0);
                    _phase_o_full_0 ^= 1;
                    float rescale_c = 1.0f;
                    if (stat_c[0] != stat_c[1]) {
                        float _exp2_1 = approx_exp2(scale_c * (stat_c[0] - stat_c[1]));
                        rescale_c = _exp2_1;
                    }
                    int _vote_0 = __any_sync(0xFFFFFFFF, rescale_c != 1.0f);
                    if (_vote_0 != 0) {
                        #pragma unroll
                        for (int half_c = 0; half_c < 2; half_c++) {
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(corr_c[0]), "=f"(corr_c[1]), "=f"(corr_c[2]), "=f"(corr_c[3]), "=f"(corr_c[4]), "=f"(corr_c[5]), "=f"(corr_c[6]), "=f"(corr_c[7]), "=f"(corr_c[8]), "=f"(corr_c[9]), "=f"(corr_c[10]), "=f"(corr_c[11]), "=f"(corr_c[12]), "=f"(corr_c[13]), "=f"(corr_c[14]), "=f"(corr_c[15]), "=f"(corr_c[16]), "=f"(corr_c[17]), "=f"(corr_c[18]), "=f"(corr_c[19]), "=f"(corr_c[20]), "=f"(corr_c[21]), "=f"(corr_c[22]), "=f"(corr_c[23]), "=f"(corr_c[24]), "=f"(corr_c[25]), "=f"(corr_c[26]), "=f"(corr_c[27]), "=f"(corr_c[28]), "=f"(corr_c[29]), "=f"(corr_c[30]), "=f"(corr_c[31])
                                : "r"(taddr + 256 + (unsigned int)(half_c * 128)));
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(corr_c[32]), "=f"(corr_c[33]), "=f"(corr_c[34]), "=f"(corr_c[35]), "=f"(corr_c[36]), "=f"(corr_c[37]), "=f"(corr_c[38]), "=f"(corr_c[39]), "=f"(corr_c[40]), "=f"(corr_c[41]), "=f"(corr_c[42]), "=f"(corr_c[43]), "=f"(corr_c[44]), "=f"(corr_c[45]), "=f"(corr_c[46]), "=f"(corr_c[47]), "=f"(corr_c[48]), "=f"(corr_c[49]), "=f"(corr_c[50]), "=f"(corr_c[51]), "=f"(corr_c[52]), "=f"(corr_c[53]), "=f"(corr_c[54]), "=f"(corr_c[55]), "=f"(corr_c[56]), "=f"(corr_c[57]), "=f"(corr_c[58]), "=f"(corr_c[59]), "=f"(corr_c[60]), "=f"(corr_c[61]), "=f"(corr_c[62]), "=f"(corr_c[63])
                                : "r"(taddr + 256 + (unsigned int)(half_c * 128) + 32));
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(corr_c[64]), "=f"(corr_c[65]), "=f"(corr_c[66]), "=f"(corr_c[67]), "=f"(corr_c[68]), "=f"(corr_c[69]), "=f"(corr_c[70]), "=f"(corr_c[71]), "=f"(corr_c[72]), "=f"(corr_c[73]), "=f"(corr_c[74]), "=f"(corr_c[75]), "=f"(corr_c[76]), "=f"(corr_c[77]), "=f"(corr_c[78]), "=f"(corr_c[79]), "=f"(corr_c[80]), "=f"(corr_c[81]), "=f"(corr_c[82]), "=f"(corr_c[83]), "=f"(corr_c[84]), "=f"(corr_c[85]), "=f"(corr_c[86]), "=f"(corr_c[87]), "=f"(corr_c[88]), "=f"(corr_c[89]), "=f"(corr_c[90]), "=f"(corr_c[91]), "=f"(corr_c[92]), "=f"(corr_c[93]), "=f"(corr_c[94]), "=f"(corr_c[95])
                                : "r"(taddr + 256 + (unsigned int)(half_c * 128) + 64));
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(corr_c[96]), "=f"(corr_c[97]), "=f"(corr_c[98]), "=f"(corr_c[99]), "=f"(corr_c[100]), "=f"(corr_c[101]), "=f"(corr_c[102]), "=f"(corr_c[103]), "=f"(corr_c[104]), "=f"(corr_c[105]), "=f"(corr_c[106]), "=f"(corr_c[107]), "=f"(corr_c[108]), "=f"(corr_c[109]), "=f"(corr_c[110]), "=f"(corr_c[111]), "=f"(corr_c[112]), "=f"(corr_c[113]), "=f"(corr_c[114]), "=f"(corr_c[115]), "=f"(corr_c[116]), "=f"(corr_c[117]), "=f"(corr_c[118]), "=f"(corr_c[119]), "=f"(corr_c[120]), "=f"(corr_c[121]), "=f"(corr_c[122]), "=f"(corr_c[123]), "=f"(corr_c[124]), "=f"(corr_c[125]), "=f"(corr_c[126]), "=f"(corr_c[127])
                                : "r"(taddr + 256 + (unsigned int)(half_c * 128) + 96));
                            const float2 _scale2_0 = {rescale_c, rescale_c};
                            #pragma unroll
                            for (int _ls = 0; _ls < 64; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(corr_c)[_ls], _scale2_0);
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x128.b32"
                                " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127, %128};"
                                :: "r"(taddr + 256 + (unsigned int)(half_c * 128)), "f"(corr_c[0]), "f"(corr_c[1]), "f"(corr_c[2]), "f"(corr_c[3]), "f"(corr_c[4]), "f"(corr_c[5]), "f"(corr_c[6]), "f"(corr_c[7]), "f"(corr_c[8]), "f"(corr_c[9]), "f"(corr_c[10]), "f"(corr_c[11]), "f"(corr_c[12]), "f"(corr_c[13]), "f"(corr_c[14]), "f"(corr_c[15]), "f"(corr_c[16]), "f"(corr_c[17]), "f"(corr_c[18]), "f"(corr_c[19]), "f"(corr_c[20]), "f"(corr_c[21]), "f"(corr_c[22]), "f"(corr_c[23]), "f"(corr_c[24]), "f"(corr_c[25]), "f"(corr_c[26]), "f"(corr_c[27]), "f"(corr_c[28]), "f"(corr_c[29]), "f"(corr_c[30]), "f"(corr_c[31]), "f"(corr_c[32]), "f"(corr_c[33]), "f"(corr_c[34]), "f"(corr_c[35]), "f"(corr_c[36]), "f"(corr_c[37]), "f"(corr_c[38]), "f"(corr_c[39]), "f"(corr_c[40]), "f"(corr_c[41]), "f"(corr_c[42]), "f"(corr_c[43]), "f"(corr_c[44]), "f"(corr_c[45]), "f"(corr_c[46]), "f"(corr_c[47]), "f"(corr_c[48]), "f"(corr_c[49]), "f"(corr_c[50]), "f"(corr_c[51]), "f"(corr_c[52]), "f"(corr_c[53]), "f"(corr_c[54]), "f"(corr_c[55]), "f"(corr_c[56]), "f"(corr_c[57]), "f"(corr_c[58]), "f"(corr_c[59]), "f"(corr_c[60]), "f"(corr_c[61]), "f"(corr_c[62]), "f"(corr_c[63]), "f"(corr_c[64]), "f"(corr_c[65]), "f"(corr_c[66]), "f"(corr_c[67]), "f"(corr_c[68]), "f"(corr_c[69]), "f"(corr_c[70]), "f"(corr_c[71]), "f"(corr_c[72]), "f"(corr_c[73]), "f"(corr_c[74]), "f"(corr_c[75]), "f"(corr_c[76]), "f"(corr_c[77]), "f"(corr_c[78]), "f"(corr_c[79]), "f"(corr_c[80]), "f"(corr_c[81]), "f"(corr_c[82]), "f"(corr_c[83]), "f"(corr_c[84]), "f"(corr_c[85]), "f"(corr_c[86]), "f"(corr_c[87]), "f"(corr_c[88]), "f"(corr_c[89]), "f"(corr_c[90]), "f"(corr_c[91]), "f"(corr_c[92]), "f"(corr_c[93]), "f"(corr_c[94]), "f"(corr_c[95]), "f"(corr_c[96]), "f"(corr_c[97]), "f"(corr_c[98]), "f"(corr_c[99]), "f"(corr_c[100]), "f"(corr_c[101]), "f"(corr_c[102]), "f"(corr_c[103]), "f"(corr_c[104]), "f"(corr_c[105]), "f"(corr_c[106]), "f"(corr_c[107]), "f"(corr_c[108]), "f"(corr_c[109]), "f"(corr_c[110]), "f"(corr_c[111]), "f"(corr_c[112]), "f"(corr_c[113]), "f"(corr_c[114]), "f"(corr_c[115]), "f"(corr_c[116]), "f"(corr_c[117]), "f"(corr_c[118]), "f"(corr_c[119]), "f"(corr_c[120]), "f"(corr_c[121]), "f"(corr_c[122]), "f"(corr_c[123]), "f"(corr_c[124]), "f"(corr_c[125]), "f"(corr_c[126]), "f"(corr_c[127]));
                        }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    }
                    mbarrier_arrive(o_empty_addr);
                }
                mbarrier_wait(stats_full_addr + (stat_slot_c) * 8, stat_phase_c);
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x2.b32"
                    " {%0, %1}, [%2];"
                    : "=f"(stat_c[0]), "=f"(stat_c[1])
                    : "r"(taddr + (unsigned int)(stat_slot_c * 128)));
                mbarrier_arrive(stats_empty_addr + (stat_slot_c) * 8);
                stat_slot_c += 1;
                if (stat_slot_c == 2) { stat_slot_c = 0; stat_phase_c ^= 1; }
                mbarrier_wait(stats_full_addr + (stat_slot_c) * 8, stat_phase_c);
                mbarrier_arrive(stats_empty_addr + (stat_slot_c) * 8);
                stat_slot_c += 1;
                if (stat_slot_c == 2) { stat_slot_c = 0; stat_phase_c ^= 1; }
                mbarrier_wait(o_full_addr, _phase_o_full_0);
                _phase_o_full_0 ^= 1;
                float total_sum_c = stat_c[0];
                float normalization_c = output_scale_c / total_sum_c;
                #pragma unroll
                for (int half_c_1 = 0; half_c_1 < 2; half_c_1++) {
                    #pragma unroll 1
                    for (int chunk_c = 0; chunk_c < 128; chunk_c += 8) {
                        tmem_ld_x8(&out_c[0], taddr + 256 + (unsigned int)(half_c_1 * 128) + (unsigned int)chunk_c);
                        const float2 _scale2_1 = {normalization_c, normalization_c};
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(out_c)[_ls], _scale2_1);
                        unsigned int packed_c[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(out_c[_lp*2 + 0], out_c[_lp*2+1 + 0]));
                            packed_c[_lp] = *(uint32_t*)&_bf2;
                        }
                        if (logical_row_c < Q_LEN * 16) {
                            int offset_c = ((batch_c * Q_LEN + logical_row_c / 16) * NUM_Q_HEADS + kv_head_c * 16 + logical_row_c % 16) * 256 + half_c_1 * 128 + first_col_c + chunk_c;
                            reinterpret_cast<int4*>(reinterpret_cast<uint8_t*>(O + offset_c))[0] = reinterpret_cast<int4*>(packed_c)[0];
                        }
                    }
                }
                mbarrier_arrive(o_empty_addr);
                mbarrier_wait(work_full_addr + (work_stage_c) * 8, _phase_work_full_1);
                uint32_t _clc_valid_5 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_5)
                    : "r"(work_response_addr + work_stage_c * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_5 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_5)
                    : "r"(work_response_addr + work_stage_c * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage_c) * 8);
                work_stage_c += 1;
                if (work_stage_c == 2) { work_stage_c = 0; _phase_work_full_1 ^= 1; }
                if (_clc_valid_5 == 0) {
                    break;
                }
                tile_idx_c = _clc_ctaid_5;
            }
            asm volatile("barrier.sync 6, 128;" ::: "memory");
            if (warp == 4) {
                int _shfl_0 = __shfl_sync(0xFFFFFFFF, taddr, 0);
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_shfl_0), "r"(512));
            }
        }
    }
    // ---- Role: mma_warp ----
    if (warp == 8) {
        { // mma_warp_main
            int q_slot_m = 0;
            int q_phase_m = 0;
            int kv_slot_m = 0;
            int kv_phase_m = 0;
            int s_slot_m = 0;
            int s_phase_m = 1;
            int p_slot_m = 0;
            unsigned int work_stage_m = 0;
            unsigned int tile_idx_m = blockIdx.z;
            mbarrier_wait(s_empty_addr, 1);
            mbarrier_wait(s_empty_addr + 8, 1);
            unsigned int _phase_o_empty_0 = 1;
            unsigned int _phase_work_full_2 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_m = 0; _tile_iter_m < BATCH_SIZE * NUM_KV_HEADS; _tile_iter_m++) {
                int batch_m = tile_idx_m / (unsigned int)NUM_KV_HEADS;
                if (use_request_order != 0) {
                    batch_m = request_order[batch_m];
                }
                int kv_len_m = seq_lens_kv[batch_m];
                int n_blocks_m = (kv_len_m + 127) / 128;
                mbarrier_wait(q_full_addr + (q_slot_m) * 8, q_phase_m);
                mbarrier_wait(kv_full_addr + (kv_slot_m) * 8, kv_phase_m);
                int _mma_a_lo_0 = make_warp_uniform((((smem_qt_addr) >> 4) & 0x3FFF) + (q_slot_m) * 2048);
                int _mma_b_lo_0 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (kv_slot_m) * 1024);
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
                    "mov.b32 id, 136314896;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_scores + (s_slot_m * 128))), "r"(0));
                elect_commit(kv_empty_addr + (kv_slot_m) * 8);
                kv_slot_m += 1;
                if (kv_slot_m == 9) { kv_slot_m = 0; kv_phase_m ^= 1; }
                mbarrier_wait(kv_full_addr + (kv_slot_m) * 8, kv_phase_m);
                int _mma_a_lo_1 = make_warp_uniform((((smem_qt_second_addr) >> 4) & 0x3FFF) + (q_slot_m) * 2048);
                int _mma_b_lo_1 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (kv_slot_m) * 1024);
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
                    "mov.b32 id, 136314896;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"((tmem_scores + (s_slot_m * 128))), "r"(1));
                elect_commit2(kv_empty_addr + (kv_slot_m) * 8, s_full_addr + (s_slot_m) * 8);
                kv_slot_m += 1;
                if (kv_slot_m == 9) { kv_slot_m = 0; kv_phase_m ^= 1; }
                s_slot_m += 1;
                if (s_slot_m == 2) { s_slot_m = 0; s_phase_m ^= 1; }
                #pragma unroll 1
                for (int i_m = 0; i_m < n_blocks_m - 1; i_m++) {
                    mbarrier_wait(kv_full_addr + (kv_slot_m) * 8, kv_phase_m);
                    int _mma_a_lo_2 = make_warp_uniform((((smem_qt_addr) >> 4) & 0x3FFF) + (q_slot_m) * 2048);
                    int _mma_b_lo_2 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (kv_slot_m) * 1024);
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
                    "mov.b32 id, 136314896;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_2), "r"(_mma_b_lo_2), "r"((tmem_scores + (s_slot_m * 128))), "r"(0));
                    elect_commit(kv_empty_addr + (kv_slot_m) * 8);
                    kv_slot_m += 1;
                    if (kv_slot_m == 9) { kv_slot_m = 0; kv_phase_m ^= 1; }
                    mbarrier_wait(kv_full_addr + (kv_slot_m) * 8, kv_phase_m);
                    int _mma_a_lo_3 = make_warp_uniform((((smem_qt_second_addr) >> 4) & 0x3FFF) + (q_slot_m) * 2048);
                    int _mma_b_lo_3 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (kv_slot_m) * 1024);
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
                    "mov.b32 id, 136314896;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f8f6f4 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_3), "r"(_mma_b_lo_3), "r"((tmem_scores + (s_slot_m * 128))), "r"(1));
                    elect_commit2(kv_empty_addr + (kv_slot_m) * 8, s_full_addr + (s_slot_m) * 8);
                    kv_slot_m += 1;
                    if (kv_slot_m == 9) { kv_slot_m = 0; kv_phase_m ^= 1; }
                    s_slot_m += 1;
                    if (s_slot_m == 2) { s_slot_m = 0; s_phase_m ^= 1; }
                    mbarrier_wait(s_empty_addr + (s_slot_m) * 8, s_phase_m);
                    mbarrier_wait(o_empty_addr, _phase_o_empty_0);
                    _phase_o_empty_0 ^= 1;
                    mbarrier_wait(kv_full_addr + (kv_slot_m) * 8, kv_phase_m);
                    int _mma_b_lo_4 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (kv_slot_m) * 1024);
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
                    "mov.b32 id, 136380432;\n\t"
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
                    :: "r"(tmem_output_hi), "r"(_mma_b_lo_4), "r"(tmem_probabilities + p_slot_m * 128), "r"(((i_m == 0) ? 0 : 1)));
                    elect_commit(kv_empty_addr + (kv_slot_m) * 8);
                    kv_slot_m += 1;
                    if (kv_slot_m == 9) { kv_slot_m = 0; kv_phase_m ^= 1; }
                    mbarrier_wait(kv_full_addr + (kv_slot_m) * 8, kv_phase_m);
                    int _mma_b_lo_5 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (kv_slot_m) * 1024);
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
                    "mov.b32 id, 136380432;\n\t"
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
                    :: "r"(tmem_output_lo), "r"(_mma_b_lo_5), "r"(tmem_probabilities + p_slot_m * 128), "r"(((i_m == 0) ? 0 : 1)));
                    elect_commit2(kv_empty_addr + (kv_slot_m) * 8, o_full_addr);
                    kv_slot_m += 1;
                    if (kv_slot_m == 9) { kv_slot_m = 0; kv_phase_m ^= 1; }
                    p_slot_m = (p_slot_m + 1) % 2;
                }
                elect_commit(q_empty_addr + (q_slot_m) * 8);
                q_slot_m += 1;
                if (q_slot_m == 2) { q_slot_m = 0; q_phase_m ^= 1; }
                int tail_slot_m = (s_slot_m + 1) % 2;
                int tail_phase_m = s_phase_m ^ ((s_slot_m == 1) ? 1 : 0);
                mbarrier_wait(s_empty_addr + (tail_slot_m) * 8, tail_phase_m);
                mbarrier_wait(o_empty_addr, _phase_o_empty_0);
                _phase_o_empty_0 ^= 1;
                mbarrier_wait(kv_full_addr + (kv_slot_m) * 8, kv_phase_m);
                int _mma_b_lo_6 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (kv_slot_m) * 1024);
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
                    "mov.b32 id, 136380432;\n\t"
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
                    :: "r"(tmem_output_hi), "r"(_mma_b_lo_6), "r"(tmem_probabilities + p_slot_m * 128), "r"(((n_blocks_m == 1) ? 0 : 1)));
                elect_commit(kv_empty_addr + (kv_slot_m) * 8);
                kv_slot_m += 1;
                if (kv_slot_m == 9) { kv_slot_m = 0; kv_phase_m ^= 1; }
                mbarrier_wait(kv_full_addr + (kv_slot_m) * 8, kv_phase_m);
                int _mma_b_lo_7 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (kv_slot_m) * 1024);
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
                    "mov.b32 id, 136380432;\n\t"
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
                    :: "r"(tmem_output_lo), "r"(_mma_b_lo_7), "r"(tmem_probabilities + p_slot_m * 128), "r"(((n_blocks_m == 1) ? 0 : 1)));
                elect_commit2(kv_empty_addr + (kv_slot_m) * 8, o_full_addr);
                kv_slot_m += 1;
                if (kv_slot_m == 9) { kv_slot_m = 0; kv_phase_m ^= 1; }
                p_slot_m = (p_slot_m + 1) % 2;
                mbarrier_wait(work_full_addr + (work_stage_m) * 8, _phase_work_full_2);
                uint32_t _clc_valid_3 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_3)
                    : "r"(work_response_addr + work_stage_m * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_3 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_3)
                    : "r"(work_response_addr + work_stage_m * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage_m) * 8);
                work_stage_m += 1;
                if (work_stage_m == 2) { work_stage_m = 0; _phase_work_full_2 ^= 1; }
                if (_clc_valid_3 == 0) {
                    break;
                }
                tile_idx_m = _clc_ctaid_3;
            }
            elect_commit(s_full_addr + (s_slot_m) * 8);
            s_slot_m += 1;
            if (s_slot_m == 2) { s_slot_m = 0; s_phase_m ^= 1; }
            elect_commit(s_full_addr + (s_slot_m) * 8);
        }
    }
    // ---- Role: load_pgoff ----
    if (warp == 9) {
        { // load_pgoff_main
            int pg_slot_p = 0;
            int pg_phase_p = 1;
            unsigned int work_stage_p = 0;
            unsigned int tile_idx_p = blockIdx.z;
            unsigned int _phase_work_full_3 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_p = 0; _tile_iter_p < BATCH_SIZE * NUM_KV_HEADS; _tile_iter_p++) {
                int batch_idx_p = tile_idx_p / (unsigned int)NUM_KV_HEADS;
                if (use_request_order != 0) {
                    batch_idx_p = request_order[batch_idx_p];
                }
                int seqlen_kv_p = seq_lens_kv[batch_idx_p];
                int n_blocks_p = (seqlen_kv_p + 127) / 128;
                int cta_blocks_p = n_blocks_p;
                int page_upper_p = (seqlen_kv_p + 63) / 64 - 1;
                #pragma unroll 1
                for (int group_base_p = 0; group_base_p < cta_blocks_p; group_base_p += 16) {
                    int page_idx_p = (unsigned int)(group_base_p * 2) + lane;
                    page_idx_p = ((page_idx_p > page_upper_p) ? page_upper_p : page_idx_p);
                    #pragma unroll
                    for (int side_p = 0; side_p < 2; side_p++) {
                        mbarrier_wait(pg_empty_addr + (pg_slot_p) * 8, pg_phase_p);
                        asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4;"
                            :: "r"(smem_pg_addr + (unsigned int)(pg_slot_p * 128) + lane * 4), "l"(page_table + (batch_idx_p * pt_batch_stride + side_p * pt_v_offset + page_idx_p)));
                        asm volatile(
                            "{\n\t"
                            "cp.async.mbarrier.arrive.shared::cta.b64 [%0];\n\t"
                            "}"
                            :: "r"(pg_full_addr + (pg_slot_p) * 8) : "memory");
                        mbarrier_arrive(pg_full_addr + (pg_slot_p) * 8);
                        pg_slot_p += 1;
                        if (pg_slot_p == 6) { pg_slot_p = 0; pg_phase_p ^= 1; }
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_p) * 8, _phase_work_full_3);
                uint32_t _clc_valid_0 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_0)
                    : "r"(work_response_addr + work_stage_p * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_0 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_0)
                    : "r"(work_response_addr + work_stage_p * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage_p) * 8);
                work_stage_p += 1;
                if (work_stage_p == 2) { work_stage_p = 0; _phase_work_full_3 ^= 1; }
                if (_clc_valid_0 == 0) {
                    break;
                }
                tile_idx_p = _clc_ctaid_0;
            }
        }
    }
    // ---- Role: scheduler ----
    if (warp == 10) {
        { // scheduler_main
            unsigned int work_stage_h = 0;
            unsigned int throttle_stage_h = 0;
            unsigned int tile_idx_h = blockIdx.z;
            unsigned int total_tiles_h = BATCH_SIZE * NUM_KV_HEADS;
            unsigned int _phase_throttle_full = 0;
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_work_full_4 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_h = 0; _tile_iter_h < total_tiles_h; _tile_iter_h++) {
                mbarrier_wait(throttle_full_addr + (throttle_stage_h) * 8, _phase_throttle_full);
                mbarrier_arrive(throttle_empty_addr + (throttle_stage_h) * 8);
                throttle_stage_h += 1;
                if (throttle_stage_h == 2) { throttle_stage_h = 0; _phase_throttle_full ^= 1; }
                if (elect_sync()) {
                    mbarrier_wait(work_empty_addr + (work_stage_h) * 8, _phase_work_empty);
                    mbarrier_arrive_expect_tx(work_full_addr + (work_stage_h) * 8, 16);
                    asm volatile(
                        "fence.proxy.async.shared::cta;\n\t"
                        "clusterlaunchcontrol.try_cancel.async.shared::cta"
                            ".mbarrier::complete_tx::bytes.b128"
                            " [%0], [%1];"
                        :: "r"(work_response_addr + work_stage_h * 16 + 0 * 16), "r"(work_full_addr + work_stage_h * 8)
                        : "memory");
                }
                mbarrier_wait(work_full_addr + (work_stage_h) * 8, _phase_work_full_4);
                uint32_t _clc_valid_1 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_1)
                    : "r"(work_response_addr + work_stage_h * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_1 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_1)
                    : "r"(work_response_addr + work_stage_h * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage_h) * 8);
                work_stage_h += 1;
                if (work_stage_h == 2) { work_stage_h = 0; _phase_work_empty ^= 1; _phase_work_full_4 ^= 1; }
                if (_clc_valid_1 == 0) {
                    break;
                }
                tile_idx_h = _clc_ctaid_1;
            }
        }
    }
    // ---- Role: load_warp ----
    if (warp == 11) {
        { // load_warp_main
            int q_slot_l = 0;
            int q_phase_l = 1;
            int pg_slot_l = 0;
            int pg_phase_l = 0;
            int kv_slot_l = 0;
            int kv_phase_l = 1;
            unsigned int work_stage_l = 0;
            unsigned int throttle_stage_l = 0;
            unsigned int tile_idx_l = blockIdx.z;
            unsigned int total_tiles_l = BATCH_SIZE * NUM_KV_HEADS;
            unsigned int _phase_throttle_empty = 1;
            unsigned int _phase_work_full_5 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_l = 0; _tile_iter_l < total_tiles_l; _tile_iter_l++) {
                mbarrier_wait(throttle_empty_addr + (throttle_stage_l) * 8, _phase_throttle_empty);
                mbarrier_arrive(throttle_full_addr + (throttle_stage_l) * 8);
                throttle_stage_l += 1;
                if (throttle_stage_l == 2) { throttle_stage_l = 0; _phase_throttle_empty ^= 1; }
                int split_l = 0;
                int bh_l = tile_idx_l;
                int batch_idx = bh_l / NUM_KV_HEADS;
                if (use_request_order != 0) {
                    batch_idx = request_order[batch_idx];
                }
                int kv_head_idx = bh_l % NUM_KV_HEADS;
                int seqlen_kv = seq_lens_kv[batch_idx];
                int num_n_blocks_total = (seqlen_kv + BLOCK_N - 1) / BLOCK_N;
                int cta_n_blocks = num_n_blocks_total;
                mbarrier_wait(q_empty_addr + (q_slot_l) * 8, q_phase_l);
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(q_full_addr + (q_slot_l) * 8, 32768);
                    tma_4d_gmem2smem(smem_qt_addr + (unsigned int)(q_slot_l * 32768), Qt, 0, 0, kv_head_idx, batch_idx * Q_LEN, q_full_addr + (q_slot_l) * 8);
                    tma_4d_gmem2smem(smem_qt_second_addr + (unsigned int)(q_slot_l * 32768), Qt, 128, 0, kv_head_idx, batch_idx * Q_LEN, q_full_addr + (q_slot_l) * 8);
                    int pg_k_slot_l = 0;
                    int pg_v_slot_l = 0;
                    #pragma unroll 1
                    for (int action_l = 0; action_l < 2 * cta_n_blocks; action_l++) {
                        int is_v_l = ((action_l == 0) ? 0 : (action_l + 1) % 2);
                        int block_l = ((action_l == 0) ? 0 : (action_l + 1) / 2);
                        if (action_l == 2 * cta_n_blocks - 1) {
                            is_v_l = 1;
                            block_l = cta_n_blocks - 1;
                        } else if (is_v_l != 0) {
                            block_l = block_l - 1;
                        }
                        if (block_l % 16 == 0) {
                            mbarrier_wait(pg_full_addr + (pg_slot_l) * 8, pg_phase_l);
                            if (is_v_l != 0) {
                                pg_v_slot_l = pg_slot_l;
                            } else {
                                pg_k_slot_l = pg_slot_l;
                            }
                            pg_slot_l += 1;
                            if (pg_slot_l == 6) { pg_slot_l = 0; pg_phase_l ^= 1; }
                        }
                        int pg_read_slot_l = ((is_v_l != 0) ? pg_v_slot_l : pg_k_slot_l);
                        int pages_l[2];
                        asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&pages_l[0])), "=r"(*reinterpret_cast<uint32_t*>(&pages_l[(0) + 1]))
                            : "r"(smem_pg_addr + (unsigned int)(pg_read_slot_l * 128) + (unsigned int)(block_l % 16 * 8)) : "memory");
                        if (block_l % 16 == 15 || block_l + 1 == cta_n_blocks) {
                            mbarrier_arrive(pg_empty_addr + (pg_read_slot_l) * 8);
                        }
                        #pragma unroll
                        for (int half_l = 0; half_l < 2; half_l++) {
                            mbarrier_wait(kv_empty_addr + (kv_slot_l) * 8, kv_phase_l);
                            mbarrier_arrive_expect_tx(kv_full_addr + (kv_slot_l) * 8, 16384);
                            #pragma unroll
                            for (int page_l = 0; page_l < 2; page_l++) {
                                if (is_v_l != 0) {
                                    tma_4d_gmem2smem(smem_kv_addr + (unsigned int)(kv_slot_l * 16384) + (unsigned int)(page_l * 8192), V, 128 * half_l, 0, kv_head_idx, pages_l[page_l], kv_full_addr + (kv_slot_l) * 8);
                                } else {
                                    tma_4d_gmem2smem(smem_kv_addr + (unsigned int)(kv_slot_l * 16384) + (unsigned int)(page_l * 8192), K, 128 * half_l, 0, kv_head_idx, pages_l[page_l], kv_full_addr + (kv_slot_l) * 8);
                                }
                            }
                            kv_slot_l += 1;
                            if (kv_slot_l == 9) { kv_slot_l = 0; kv_phase_l ^= 1; }
                        }
                    }
                }
                q_slot_l += 1;
                if (q_slot_l == 2) { q_slot_l = 0; q_phase_l ^= 1; }
                mbarrier_wait(work_full_addr + (work_stage_l) * 8, _phase_work_full_5);
                uint32_t _clc_valid_2 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_2)
                    : "r"(work_response_addr + work_stage_l * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_2 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_2)
                    : "r"(work_response_addr + work_stage_l * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage_l) * 8);
                work_stage_l += 1;
                if (work_stage_l == 2) { work_stage_l = 0; _phase_work_full_5 ^= 1; }
                if (_clc_valid_2 == 0) {
                    break;
                }
                tile_idx_l = _clc_ctaid_2;
            }
        }
    }

    // Cleanup
}

} // extern "C"
