/*
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
static_assert(sizeof(uint64_t) == 8, "Sm110 requires an LP64 CUDA host ABI");
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) Sm110TensorMap { uint64_t opaque[16]; };
struct __align__(64) Sm110TensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(Sm110TensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(Sm110TensorMap64) == 64, "64-aligned tensor-map ABI alignment");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128-byte aligned");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define SM110_INF CUDART_INF_F
#define TMEM_NCOLS 128
#define TMEM_SCORES_OFFSET 64
#define TMEM_OUTPUT_OFFSET 0
#define NUM_KV_PIPE_STAGES 3
#define SMEM_Q_SMEM_OFF 1024
#define SMEM_Q_SMEM_STAGE_BYTES 16384
#define SMEM_Q_SMEM_STRIDE 16384
#define SMEM_KV_SMEM_OFF 17408
#define SMEM_KV_SMEM_STAGE_BYTES 8192
#define SMEM_KV_SMEM_STRIDE 8192
#define SMEM_V_SMEM_OFF 17408
#define SMEM_V_SMEM_STAGE_BYTES 8192
#define SMEM_V_SMEM_STRIDE 8192
#define SMEM_V_LOW_OFF 17408
#define SMEM_V_LOW_STAGE_BYTES 4096
#define SMEM_V_LOW_STRIDE 8192
#define SMEM_V_HIGH_OFF 21504
#define SMEM_V_HIGH_STAGE_BYTES 4096
#define SMEM_V_HIGH_STRIDE 8192
#define SMEM_TOTAL 41984
#define THREADS 128

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


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tmem_st_x8_u32(int addr, const uint32_t* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x8.b32"
        " [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
        :: "r"(addr),
           "r"(src[0]), "r"(src[1]), "r"(src[2]), "r"(src[3]),
           "r"(src[4]), "r"(src[5]), "r"(src[6]), "r"(src[7]));
}

extern "C" {

__global__ __launch_bounds__(128, 1) void
kernel_sm110_gqa_decode_n32_b4_direct(const __grid_constant__ Sm110TensorMap64 Q, const __grid_constant__ Sm110TensorMap64 K, const __grid_constant__ Sm110TensorMap64 V, __half* __restrict__ O, int* __restrict__ sequence_lengths, float softmax_scale_log2)
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
    #define kv_full_addr (mbar_base + 8)
    #define kv_empty_addr (mbar_base + 32)
    #define s_full_addr (mbar_base + 56)
    #define padding_ready_addr (mbar_base + 64)
    #define p_full_addr (mbar_base + 72)
    #define o_full_addr (mbar_base + 80)
    #define tile_done_addr (mbar_base + 88)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    __half* q_smem = reinterpret_cast<__half*>(smem_raw + 1024);
    const int q_smem_addr = smem + 1024;
    __half* kv_smem = reinterpret_cast<__half*>(smem_raw + 17408);
    const int kv_smem_addr = smem + 17408;
    __half* v_smem = reinterpret_cast<__half*>(smem_raw + 17408);
    const int v_smem_addr = smem + 17408;
    __half* v_low = reinterpret_cast<__half*>(smem_raw + 17408);
    const int v_low_addr = smem + 17408;
    __half* v_high = reinterpret_cast<__half*>(smem_raw + 21504);
    const int v_high_addr = smem + 21504;

    // Mbarrier init (8 pipeline groups, 0 ordered-sequence groups, 12 barriers)
    // Mbarriers at smem_raw[0..96)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // kv_full: 3 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // --- pipeline 'kv_pipe' ---
            // kv_empty: 3 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            // s_full: 1 barriers, init_count=1
            mbarrier_init(smem + 56, 1);
            // padding_ready: 1 barriers, init_count=96
            mbarrier_init(smem + 64, 96);
            // p_full: 1 barriers, init_count=32
            mbarrier_init(smem + 72, 32);
            // o_full: 1 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            // tile_done: 1 barriers, init_count=32
            mbarrier_init(smem + 88, 32);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (128 columns, 128 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 96);
    if (warp == 0) {
        int _tmem_hold = smem + 96;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(128) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_scores = taddr + 64;
    const int tmem_output = taddr;

    // ---- Role: softmax ----
    if (warp == 0) {
        { // softmax_main
            int split = 0;
            int tile = blockIdx.x;
            int batch = tile / 8;
            int kv_head = tile - batch * 8;
            int seqlen_kv = sequence_lengths[batch];
            int total_blocks = (seqlen_kv + 32 - 1) / 32;
            int first_block = total_blocks * split;
            int end_block = total_blocks * (split + 1);
            int num_n_blocks = end_block - first_block;
            if (num_n_blocks < 1) {
                num_n_blocks = 1;
                first_block = total_blocks;
            }
            const int warp_in_role = warp;
            const int tmem_row_origin = warp_in_role * 32;
            const int logical_row_origin = warp_in_role * 16;
            int my_row = (unsigned int)logical_row_origin + lane % 16;
            int col_half = lane / 16;
            int row_valid = ((my_row < 4) ? 1 : 0);
            float row_max = -SM110_INF;
            float row_sum = 0.0f;
            unsigned int _phase_s_full_0 = 0;
            #pragma unroll 1
            for (int n_block = 0; n_block < num_n_blocks; n_block++) {
                mbarrier_wait(s_full_addr, _phase_s_full_0);
                _phase_s_full_0 ^= 1;
                int valid_cols = seqlen_kv - (first_block + n_block) * 32;
                if (valid_cols > 32) {
                    valid_cols = 32;
                }
                if (row_valid == 0) {
                    valid_cols = 0;
                }
                int score_addr = taddr + 64 + (unsigned int)(tmem_row_origin << 16);
                float _tmem_load_0[16];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x16.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16], 16;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15]))
                    : "r"(score_addr));
                int half_valid = valid_cols - col_half * 16;
                if (half_valid < 0) {
                    half_valid = 0;
                }
                if (half_valid > 16) {
                    half_valid = 16;
                }
                if (half_valid <= 0) {
                    if (!(0 & (1u << 0))) _tmem_load_0[0] = -SM110_INF;
                    if (!(0 & (1u << 1))) _tmem_load_0[1] = -SM110_INF;
                    if (!(0 & (1u << 2))) _tmem_load_0[2] = -SM110_INF;
                    if (!(0 & (1u << 3))) _tmem_load_0[3] = -SM110_INF;
                    if (!(0 & (1u << 4))) _tmem_load_0[4] = -SM110_INF;
                    if (!(0 & (1u << 5))) _tmem_load_0[5] = -SM110_INF;
                    if (!(0 & (1u << 6))) _tmem_load_0[6] = -SM110_INF;
                    if (!(0 & (1u << 7))) _tmem_load_0[7] = -SM110_INF;
                    if (!(0 & (1u << 8))) _tmem_load_0[8] = -SM110_INF;
                    if (!(0 & (1u << 9))) _tmem_load_0[9] = -SM110_INF;
                    if (!(0 & (1u << 10))) _tmem_load_0[10] = -SM110_INF;
                    if (!(0 & (1u << 11))) _tmem_load_0[11] = -SM110_INF;
                    if (!(0 & (1u << 12))) _tmem_load_0[12] = -SM110_INF;
                    if (!(0 & (1u << 13))) _tmem_load_0[13] = -SM110_INF;
                    if (!(0 & (1u << 14))) _tmem_load_0[14] = -SM110_INF;
                    if (!(0 & (1u << 15))) _tmem_load_0[15] = -SM110_INF;
                } else if (half_valid < 16) {
                    uint32_t _slice_lo_mask_0;
                    {
                        int _lim_0 = half_valid;
                        if (_lim_0 <= 0) { _slice_lo_mask_0 = 0u; }
                        else if (_lim_0 >= 16) { _slice_lo_mask_0 = ((1u << 16) - 1u); }
                        else {
                            asm volatile("{"
                                ".reg .u32 t;\n\t"
                                "shl.b32 t, 1, %1;\n\t"
                                "add.u32 %0, t, -1;\n\t"
                                "}" : "=r"(_slice_lo_mask_0) : "r"(_lim_0));
                        }
                    }
                    if (!(_slice_lo_mask_0 & (1u << 0))) _tmem_load_0[0] = -SM110_INF;
                    if (!(_slice_lo_mask_0 & (1u << 1))) _tmem_load_0[1] = -SM110_INF;
                    if (!(_slice_lo_mask_0 & (1u << 2))) _tmem_load_0[2] = -SM110_INF;
                    if (!(_slice_lo_mask_0 & (1u << 3))) _tmem_load_0[3] = -SM110_INF;
                    if (!(_slice_lo_mask_0 & (1u << 4))) _tmem_load_0[4] = -SM110_INF;
                    if (!(_slice_lo_mask_0 & (1u << 5))) _tmem_load_0[5] = -SM110_INF;
                    if (!(_slice_lo_mask_0 & (1u << 6))) _tmem_load_0[6] = -SM110_INF;
                    if (!(_slice_lo_mask_0 & (1u << 7))) _tmem_load_0[7] = -SM110_INF;
                    if (!(_slice_lo_mask_0 & (1u << 8))) _tmem_load_0[8] = -SM110_INF;
                    if (!(_slice_lo_mask_0 & (1u << 9))) _tmem_load_0[9] = -SM110_INF;
                    if (!(_slice_lo_mask_0 & (1u << 10))) _tmem_load_0[10] = -SM110_INF;
                    if (!(_slice_lo_mask_0 & (1u << 11))) _tmem_load_0[11] = -SM110_INF;
                    if (!(_slice_lo_mask_0 & (1u << 12))) _tmem_load_0[12] = -SM110_INF;
                    if (!(_slice_lo_mask_0 & (1u << 13))) _tmem_load_0[13] = -SM110_INF;
                    if (!(_slice_lo_mask_0 & (1u << 14))) _tmem_load_0[14] = -SM110_INF;
                    if (!(_slice_lo_mask_0 & (1u << 15))) _tmem_load_0[15] = -SM110_INF;
                }
                float _tmem_load_0_max = _tmem_load_0[0];
                #pragma unroll
                for (int _lr = 1; _lr < 16; _lr++) {
                    _tmem_load_0_max = max_noftz(_tmem_load_0_max, _tmem_load_0[_lr]);
                }
                float tile_max = _tmem_load_0_max;
                if (half_valid <= 0) {
                    tile_max = -SM110_INF;
                }
                float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, tile_max, 16);
                float _max_0 = max_noftz(tile_max, _shfl_xor_0);
                tile_max = _max_0;
                float _max_1 = max_noftz(tile_max, row_max);
                float new_max = _max_1;
                float safe_max = ((new_max == -SM110_INF) ? 0.0f : new_max);
                float new_max_scaled = safe_max * softmax_scale_log2;
                float _fma_0 = __fmaf_rn(row_max, softmax_scale_log2, -new_max_scaled);
                float acc_scale_log2 = _fma_0;
                float acc_scale;
                float selected_max;
                if (acc_scale_log2 >= -8.0f) {
                    selected_max = row_max;
                    safe_max = ((row_max == -SM110_INF) ? 0.0f : row_max);
                    acc_scale = 1.0f;
                    new_max_scaled = safe_max * softmax_scale_log2;
                } else {
                    selected_max = new_max;
                    float _exp2_0 = approx_exp2(acc_scale_log2);
                    acc_scale = ((row_max > -SM110_INF) ? _exp2_0 : 1.0f);
                }
                row_max = selected_max;
                float score_bias = ((valid_cols > 0) ? -new_max_scaled : -SM110_INF);
                const float2 _fma_b2_1 = {softmax_scale_log2, softmax_scale_log2};
                const float2 _fma_c2_2 = {score_bias, score_bias};
                #pragma unroll
                for (int _lf = 0; _lf < 8; _lf++)
                    fma_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_0)[_lf], _fma_b2_1, _fma_c2_2);
                #pragma unroll
                for (int _le = 0; _le < 16; _le++) {
                    _tmem_load_0[_le] = approx_exp2(_tmem_load_0[_le]);
                }
                float _tmem_load_0_sum = 0.0f;
                #pragma unroll
                for (int _lr = 0; _lr < 16; _lr++) {
                    _tmem_load_0_sum += _tmem_load_0[_lr];
                }
                float block_half = _tmem_load_0_sum;
                float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, block_half, 16);
                float block_sum = block_half + _shfl_xor_1;
                unsigned int packed_p[8];
                #pragma unroll
                for (int _lp = 0; _lp < 8; _lp++) {
                    __half2 _h2 = __float22half2_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                    packed_p[_lp] = *(uint32_t*)&_h2;
                }
                int p_addr = taddr + 96 + (unsigned int)(tmem_row_origin << 16);
                asm volatile(
                    "tcgen05.st.sync.aligned.16x32bx2.x8.b32"
                    " [%0], 8, {%1, %2, %3, %4, %5, %6, %7, %8};"
                    :: "r"(p_addr), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[3])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[4])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[5])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[6])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[7])));
                asm volatile(
                    "tcgen05.st.sync.aligned.16x32bx2.x8.b32"
                    " [%0], 8, {%1, %2, %3, %4, %5, %6, %7, %8};"
                    :: "r"(taddr + 96 + (unsigned int)(tmem_row_origin + 16 << 16)), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[3])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[4])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[5])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[6])), "r"(*reinterpret_cast<const uint32_t*>(&packed_p[7])));
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                int _vote_0 = __any_sync(0xFFFFFFFF, acc_scale < 1.0f);
                if (n_block > 0 && _vote_0 != 0) {
                    #pragma unroll
                    for (int output_half = 0; output_half < 2; output_half++) {
                        float _tmem_load_1[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 32;"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[31]))
                            : "r"(taddr + (unsigned int)(tmem_row_origin + output_half * 16 << 16)));
                        const float2 _scale2_3 = {acc_scale, acc_scale};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_1)[_ls], _scale2_3);
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x32bx2.x32.b32"
                            " [%0], 32, {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                            :: "r"(taddr + (unsigned int)(tmem_row_origin + output_half * 16 << 16)), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[15])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[16])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[17])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[18])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[19])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[20])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[21])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[22])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[23])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[24])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[25])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[26])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[27])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[28])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[29])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[30])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[31])));
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                }
                row_sum = row_sum * acc_scale + block_sum;
                mbarrier_arrive(p_full_addr);
            }
            unsigned int _phase_o_full_0 = 0;
            mbarrier_wait(o_full_addr, _phase_o_full_0);
            _phase_o_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float _rcp_0 = approx_rcp(row_sum);
            float inv_sum = ((row_sum > 0.0f) ? _rcp_0 : 0.0f);
            int q_head = kv_head * 4 + my_row;
            int output_row = (batch * 32 + q_head) * 128;
            #pragma unroll
            for (int output_half_1 = 0; output_half_1 < 2; output_half_1++) {
                float _tmem_load_2[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 32;"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[31]))
                    : "r"(taddr + (unsigned int)(tmem_row_origin + output_half_1 * 16 << 16)));
                int col_base = output_half_1 * 64 + col_half * 32;
                if (my_row < 4) {
                    #pragma unroll
                    for (int offset = 0; offset < 32; offset += 8) {
                        {
                            const float2 _prescale2_4 = {inv_sum, inv_sum};
                            #if __CUDA_ARCH__ >= 1000
                            #pragma unroll
                            for (int _ps = 0; _ps < 4; _ps++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_2[offset])[_ps], _prescale2_4);
                            #else
                            #pragma unroll
                            for (int _ps = 0; _ps < 8; _ps++)
                                _tmem_load_2[offset + _ps] *= inv_sum;
                            #endif
                            __half2 _pk[4];
                            _pk[0] = __floats2half2_rn(_tmem_load_2[offset + 0], _tmem_load_2[offset + 1]);
                            _pk[1] = __floats2half2_rn(_tmem_load_2[offset + 2], _tmem_load_2[offset + 3]);
                            _pk[2] = __floats2half2_rn(_tmem_load_2[offset + 4], _tmem_load_2[offset + 5]);
                            _pk[3] = __floats2half2_rn(_tmem_load_2[offset + 6], _tmem_load_2[offset + 7]);
                            *reinterpret_cast<uint4*>(&((__half*)(O + (output_row + col_base + offset)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        }
                    }
                }
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(tile_done_addr);
            asm volatile("barrier.sync 2, 128;" ::: "memory");
        }
    }
    // ---- Role: mma_warp ----
    if (warp == 1) {
        { // mma_warp_main
            int split_1 = 0;
            int tile_1 = blockIdx.x;
            int batch_1 = tile_1 / 8;
            int kv_head_1 = tile_1 - batch_1 * 8;
            int seqlen_kv_1 = sequence_lengths[batch_1];
            int total_blocks_1 = (seqlen_kv_1 + 32 - 1) / 32;
            int first_block_1 = total_blocks_1 * split_1;
            int end_block_1 = total_blocks_1 * (split_1 + 1);
            int num_n_blocks_1 = end_block_1 - first_block_1;
            if (num_n_blocks_1 < 1) {
                num_n_blocks_1 = 1;
                first_block_1 = total_blocks_1;
            }
            unsigned int inactive_p[8];
            if (!(0 & (1u << 0))) inactive_p[0] = 0;
            if (!(0 & (1u << 1))) inactive_p[1] = 0;
            if (!(0 & (1u << 2))) inactive_p[2] = 0;
            if (!(0 & (1u << 3))) inactive_p[3] = 0;
            if (!(0 & (1u << 4))) inactive_p[4] = 0;
            if (!(0 & (1u << 5))) inactive_p[5] = 0;
            if (!(0 & (1u << 6))) inactive_p[6] = 0;
            if (!(0 & (1u << 7))) inactive_p[7] = 0;
            const int inactive_row_origin = warp * 32;
            asm volatile(
                "tcgen05.st.sync.aligned.16x32bx2.x8.b32"
                " [%0], 8, {%1, %2, %3, %4, %5, %6, %7, %8};"
                :: "r"(taddr + 96 + (unsigned int)(inactive_row_origin << 16)), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p[0])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p[1])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p[2])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p[3])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p[4])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p[5])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p[6])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p[7])));
            asm volatile(
                "tcgen05.st.sync.aligned.16x32bx2.x8.b32"
                " [%0], 8, {%1, %2, %3, %4, %5, %6, %7, %8};"
                :: "r"(taddr + 96 + (unsigned int)(inactive_row_origin + 16 << 16)), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p[0])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p[1])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p[2])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p[3])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p[4])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p[5])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p[6])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p[7])));
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            mbarrier_arrive(padding_ready_addr);
            unsigned int kv_stage = 0;
            unsigned int kv_phase = 0;
            int first_pv = 1;
            unsigned int _phase_q_full_0 = 0;
            mbarrier_wait(q_full_addr, _phase_q_full_0);
            _phase_q_full_0 ^= 1;
            unsigned int _phase_padding_ready_0 = 0;
            mbarrier_wait(padding_ready_addr, _phase_padding_ready_0);
            _phase_padding_ready_0 ^= 1;
            unsigned int _phase_p_full_0 = 0;
            #pragma unroll 1
            for (int n_block_1 = 0; n_block_1 < num_n_blocks_1; n_block_1++) {
                unsigned int k_stage = kv_stage;
                unsigned int k_phase = kv_phase;
                kv_stage += 1;
                if (kv_stage == 3) { kv_stage = 0; kv_phase ^= 1; }
                mbarrier_wait(kv_full_addr + (k_stage) * 8, k_phase);
                if (elect_sync()) {
                    int _mma_a_lo_0 = ((q_smem_addr) >> 4) & 0x3FFF;
                    int _mma_b_lo_0 = (((kv_smem_addr) >> 4) & 0x3FFF) + (k_stage) * 512;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 67633168;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 250;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_scores), "r"(0));
                    tcgen05_commit(s_full_addr);
                    tcgen05_commit(kv_empty_addr + (k_stage) * 8);
                }
                unsigned int v_stage = kv_stage;
                unsigned int v_phase = kv_phase;
                kv_stage += 1;
                if (kv_stage == 3) { kv_stage = 0; kv_phase ^= 1; }
                mbarrier_wait(kv_full_addr + (v_stage) * 8, v_phase);
                mbarrier_wait(p_full_addr, _phase_p_full_0);
                _phase_p_full_0 ^= 1;
                if (elect_sync()) {
                    int _mma_b_lo_1 = ((((v_low_addr) >> 4) & 0x3FFF) | 0x1000000) + (v_stage) * 512;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 68222992;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output), "r"(_mma_b_lo_1), "r"(tmem_scores + 32), "r"(((first_pv) ? 0 : 1)));
                    int _mma_b_lo_2 = ((((v_high_addr) >> 4) & 0x3FFF) | 0x1000000) + (v_stage) * 512;
                    asm volatile(
                    "{\n\t"
                    ".reg .pred p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    ""
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 68222992;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output + 1048576), "r"(_mma_b_lo_2), "r"(tmem_scores + 32 + 1048576), "r"(((first_pv) ? 0 : 1)));
                    if (n_block_1 + 1 == num_n_blocks_1) {
                        tcgen05_commit(o_full_addr);
                    }
                    tcgen05_commit(kv_empty_addr + (v_stage) * 8);
                }
                first_pv = 0;
            }
            unsigned int _phase_tile_done_0 = 0;
            mbarrier_wait(tile_done_addr, _phase_tile_done_0);
            _phase_tile_done_0 ^= 1;
            int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(128));
            asm volatile("barrier.sync 2, 128;" ::: "memory");
        }
    }
    // ---- Role: load_warp ----
    if (warp == 2) {
        { // load_warp_main
            int split_2 = 0;
            int tile_2 = blockIdx.x;
            int batch_2 = tile_2 / 8;
            int kv_head_2 = tile_2 - batch_2 * 8;
            int seqlen_kv_2 = sequence_lengths[batch_2];
            int total_blocks_2 = (seqlen_kv_2 + 32 - 1) / 32;
            int first_block_2 = total_blocks_2 * split_2;
            int end_block_2 = total_blocks_2 * (split_2 + 1);
            int num_n_blocks_2 = end_block_2 - first_block_2;
            if (num_n_blocks_2 < 1) {
                num_n_blocks_2 = 1;
                first_block_2 = total_blocks_2;
            }
            unsigned int inactive_p_1[8];
            if (!(0 & (1u << 0))) inactive_p_1[0] = 0;
            if (!(0 & (1u << 1))) inactive_p_1[1] = 0;
            if (!(0 & (1u << 2))) inactive_p_1[2] = 0;
            if (!(0 & (1u << 3))) inactive_p_1[3] = 0;
            if (!(0 & (1u << 4))) inactive_p_1[4] = 0;
            if (!(0 & (1u << 5))) inactive_p_1[5] = 0;
            if (!(0 & (1u << 6))) inactive_p_1[6] = 0;
            if (!(0 & (1u << 7))) inactive_p_1[7] = 0;
            const int inactive_row_origin_1 = warp * 32;
            asm volatile(
                "tcgen05.st.sync.aligned.16x32bx2.x8.b32"
                " [%0], 8, {%1, %2, %3, %4, %5, %6, %7, %8};"
                :: "r"(taddr + 96 + (unsigned int)(inactive_row_origin_1 << 16)), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_1[7])));
            asm volatile(
                "tcgen05.st.sync.aligned.16x32bx2.x8.b32"
                " [%0], 8, {%1, %2, %3, %4, %5, %6, %7, %8};"
                :: "r"(taddr + 96 + (unsigned int)(inactive_row_origin_1 + 16 << 16)), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_1[7])));
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            mbarrier_arrive(padding_ready_addr);
            int total_blocks_0 = (seqlen_kv_2 + 32 - 1) / 32;
            unsigned int load_stage = 0;
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(q_full_addr, 16384);
                tma_5d_gmem2smem(q_smem_addr, (&Q), 0, 0, 0, kv_head_2, batch_2, q_full_addr);
                tma_5d_gmem2smem(q_smem_addr + 8192, (&Q), 64, 0, 0, kv_head_2, batch_2, q_full_addr);
            }
            unsigned int _phase_kv_empty = 1;
            #pragma unroll 1
            for (int n_block_2 = 0; n_block_2 < num_n_blocks_2; n_block_2++) {
                int token_base = (first_block_2 + n_block_2) * 32;
                mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 8192);
                    if (first_block_2 + n_block_2 >= total_blocks_0 - 32 && total_blocks_0 > first_block_2 + n_block_2) {
                        asm volatile(
                            "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                            :: "r"(kv_smem_addr + load_stage * 8192), "l"((&K)), "r"(0), "r"(token_base), "r"(kv_head_2), "r"(batch_2),
                               "r"(kv_full_addr + (load_stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                            :: "r"(kv_smem_addr + load_stage * 8192 + 4096), "l"((&K)), "r"(64), "r"(token_base), "r"(kv_head_2), "r"(batch_2),
                               "r"(kv_full_addr + (load_stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    } else {
                        tma_4d_gmem2smem(kv_smem_addr + load_stage * 8192, (&K), 0, token_base, kv_head_2, batch_2, kv_full_addr + (load_stage) * 8);
                        tma_4d_gmem2smem(kv_smem_addr + load_stage * 8192 + 4096, (&K), 64, token_base, kv_head_2, batch_2, kv_full_addr + (load_stage) * 8);
                    }
                }
                load_stage += 1;
                if (load_stage == 3) { load_stage = 0; _phase_kv_empty ^= 1; }
                mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(kv_full_addr + (load_stage) * 8, 8192);
                    if (first_block_2 + n_block_2 >= total_blocks_0 - 32 && total_blocks_0 > first_block_2 + n_block_2) {
                        asm volatile(
                            "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                            :: "r"(kv_smem_addr + load_stage * 8192), "l"((&V)), "r"(0), "r"(token_base), "r"(kv_head_2), "r"(batch_2),
                               "r"(kv_full_addr + (load_stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                        asm volatile(
                            "cp.async.bulk.tensor.4d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                            " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                            :: "r"(kv_smem_addr + load_stage * 8192 + 4096), "l"((&V)), "r"(64), "r"(token_base), "r"(kv_head_2), "r"(batch_2),
                               "r"(kv_full_addr + (load_stage) * 8), "l"(0x14F0000000000000ULL) : "memory");
                    } else {
                        tma_4d_gmem2smem(kv_smem_addr + load_stage * 8192, (&V), 0, token_base, kv_head_2, batch_2, kv_full_addr + (load_stage) * 8);
                        tma_4d_gmem2smem(kv_smem_addr + load_stage * 8192 + 4096, (&V), 64, token_base, kv_head_2, batch_2, kv_full_addr + (load_stage) * 8);
                    }
                }
                load_stage += 1;
                if (load_stage == 3) { load_stage = 0; _phase_kv_empty ^= 1; }
            }
            asm volatile("barrier.sync 2, 128;" ::: "memory");
        }
    }
    // ---- Role: zero_warp ----
    if (warp == 3) {
        { // zero_warp_main
            unsigned int inactive_p_2[8];
            if (!(0 & (1u << 0))) inactive_p_2[0] = 0;
            if (!(0 & (1u << 1))) inactive_p_2[1] = 0;
            if (!(0 & (1u << 2))) inactive_p_2[2] = 0;
            if (!(0 & (1u << 3))) inactive_p_2[3] = 0;
            if (!(0 & (1u << 4))) inactive_p_2[4] = 0;
            if (!(0 & (1u << 5))) inactive_p_2[5] = 0;
            if (!(0 & (1u << 6))) inactive_p_2[6] = 0;
            if (!(0 & (1u << 7))) inactive_p_2[7] = 0;
            const int inactive_row_origin_2 = warp * 32;
            asm volatile(
                "tcgen05.st.sync.aligned.16x32bx2.x8.b32"
                " [%0], 8, {%1, %2, %3, %4, %5, %6, %7, %8};"
                :: "r"(taddr + 96 + (unsigned int)(inactive_row_origin_2 << 16)), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_2[0])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_2[1])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_2[2])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_2[3])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_2[4])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_2[5])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_2[6])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_2[7])));
            asm volatile(
                "tcgen05.st.sync.aligned.16x32bx2.x8.b32"
                " [%0], 8, {%1, %2, %3, %4, %5, %6, %7, %8};"
                :: "r"(taddr + 96 + (unsigned int)(inactive_row_origin_2 + 16 << 16)), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_2[0])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_2[1])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_2[2])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_2[3])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_2[4])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_2[5])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_2[6])), "r"(*reinterpret_cast<const uint32_t*>(&inactive_p_2[7])));
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            mbarrier_arrive(padding_ready_addr);
            asm volatile("barrier.sync 2, 128;" ::: "memory");
        }
    }

    // Cleanup
}

} // extern "C"
