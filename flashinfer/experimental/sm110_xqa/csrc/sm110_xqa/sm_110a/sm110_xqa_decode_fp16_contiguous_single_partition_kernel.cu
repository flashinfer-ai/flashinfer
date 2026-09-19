/*
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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
static_assert(sizeof(uint64_t) == 8, "Sm110Xqa requires an LP64 CUDA host ABI");
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) Sm110XqaTensorMap { uint64_t opaque[16]; };
struct __align__(64) Sm110XqaTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(Sm110XqaTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(Sm110XqaTensorMap64) == 64, "64-aligned tensor-map ABI alignment");
template <int N>
struct __align__(128) Sm110XqaTensorMapPack { Sm110XqaTensorMap maps[N]; };

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(Sm110XqaTensorMap) >= alignof(CUtensorMap), "Sm110XqaTensorMap alignment must cover the CUtensorMap CUDA ABI");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define SM110_XQA_INF CUDART_INF_F
#define TMEM_NCOLS 256
#define TMEM_SCORES_OFFSET 0
#define TMEM_ACCUM_OFFSET 128
#define NUM_MAIN_STAGES 1
#define SMEM_SQ_OFF 0
#define SMEM_SQ_STAGE_BYTES 16384
#define SMEM_SQ_STRIDE 16384
#define SMEM_SK_OFF 16384
#define SMEM_SK_STAGE_BYTES 16384
#define SMEM_SK_STRIDE 16384
#define SMEM_SV_OFF 32768
#define SMEM_SV_STAGE_BYTES 16384
#define SMEM_SV_STRIDE 16384
#define SMEM_MERGE_FLAG_OFF 0
#define SMEM_MERGE_FLAG_STAGE_BYTES 4
#define SMEM_MERGE_FLAG_STRIDE 4
#define SMEM_TOTAL 50176
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

__device__ __forceinline__ void fma_f32x2_noftz_inplace(float2* a, float2 b, float2 c) {
    unsigned long long r;
    asm("fma.rn.f32x2 %0, %1, %2, %3;"
        : "=l"(r)
        : "l"(*(unsigned long long*)a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    *(unsigned long long*)a = r;
}

__device__ __forceinline__ void mul_f32x2_inplace(float2* a, float2 b) {
    asm("mul.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void mul_f32x2_noftz_inplace(float2* a, float2 b) {
    asm("mul.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void add_f32x2_inplace(float2* a, float2 b) {
    asm("add.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void add_f32x2_noftz_inplace(float2* a, float2 b) {
    asm("add.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void sub_f32x2_inplace(float2* a, float2 b) {
    asm("sub.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void sub_f32x2_noftz_inplace(float2* a, float2 b) {
    asm("sub.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ float2 add_f32x2(float2 a, float2 b) {
    float2 r;
    asm("add.rn.ftz.f32x2 %0, %1, %2;"
        : "=l"(*(unsigned long long*)&r)
        : "l"(*(unsigned long long*)&a), "l"(*(unsigned long long*)&b));
    return r;
}

__device__ __forceinline__ float2 add_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("add.f32x2 %0, %1, %2;"
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

__device__ __forceinline__ float2 sub_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("sub.f32x2 %0, %1, %2;"
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

__device__ __forceinline__ float2 mul_f32x2_noftz(float2 a, float2 b) {
    float2 r;
    asm("mul.f32x2 %0, %1, %2;"
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


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}

extern "C" {

__global__ __launch_bounds__(128, 1) void
kernel_sm110_xqa_decode_fp16_contiguous_single_partition(const __half* __restrict__ q, const __half* __restrict__ kv, const int* __restrict__ sequence_lengths, __half* __restrict__ output, float* __restrict__ partial, float* __restrict__ statistics, unsigned int* __restrict__ counters, unsigned int heads, unsigned int ratio, unsigned int capacity, unsigned int partitions, unsigned int partition_tokens, float attention_scale)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem + 49152;
    #define mma_done_addr (mbar_base + 0)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    __half* sq = reinterpret_cast<__half*>(smem_raw + 0);
    const int sq_addr = smem + 0;
    __half* sk = reinterpret_cast<__half*>(smem_raw + 16384);
    const int sk_addr = smem + 16384;
    __half* sv = reinterpret_cast<__half*>(smem_raw + 32768);
    const int sv_addr = smem + 32768;
    unsigned int* merge_flag = reinterpret_cast<unsigned int*>(smem_raw + 0);
    const int merge_flag_addr = smem + 0;

    // Mbarrier init (1 pipeline groups, 0 ordered-sequence groups, 1 barriers)
    // Mbarriers at smem_raw[49152..49160)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // mma_done: 1 barriers, init_count=1
            mbarrier_init(smem + 49152, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 49160);
    if (warp == 0) {
        int _tmem_hold = smem + 49160;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_scores = taddr;
    const int tmem_accum = taddr + 128;

    // === Task calls (dependency order) ===
    unsigned int batch = blockIdx.z;
    unsigned int head = blockIdx.y;
    unsigned int partition = blockIdx.x;
    unsigned int partition_begin = partition * partition_tokens;
    unsigned int length = sequence_lengths[batch];
    unsigned int _min_0 = ((length) < (partition_begin + partition_tokens) ? (length) : (partition_begin + partition_tokens));
    unsigned int partition_end = _min_0;
    unsigned int partition_length = ((partition_end > partition_begin) ? partition_end - partition_begin : (unsigned int)0);
    unsigned int _max_0 = (((partition_length + 63) / 64) > ((unsigned int)1) ? ((partition_length + 63) / 64) : ((unsigned int)1));
    unsigned int tile_count = _max_0;
    unsigned int query_base = (batch * heads + head) * ratio * 128;
    unsigned int k_base = (batch * 2 * heads + head) * capacity * 128;
    unsigned int v_base = ((batch * 2 + 1) * heads + head) * capacity * 128;
    float scale = attention_scale * 1.4426950408889634f;
    #pragma unroll 1
    for (int copy = 0; copy < 8; copy++) {
        unsigned int grain = copy * 128 + tid;
        unsigned int row = grain / 16;
        unsigned int col = grain % 16 * 8;
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(sq_addr + col / 64 * 8192 + row * 128 + (unsigned int)(col % 64 * 2 ^ (row & 7) << 4)), "l"(q + (query_base + row * 128 + col)), "r"((row < ratio) ? 16 : 0));
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    asm volatile("barrier.sync 8, 128;" ::: "memory");
    unsigned int my_row = warp * 16 + lane % 16;
    unsigned int row_addr = warp * 32 << 16;
    unsigned int col_half = lane / 16;
    float row_max = -SM110_XQA_INF;
    float row_sum = 0.0f;
    #pragma unroll 1
    for (unsigned int tile = 0; tile < tile_count; tile++) {
        #pragma unroll 1
        for (int copy_1 = 0; copy_1 < 8; copy_1++) {
            unsigned int grain_1 = copy_1 * 128 + tid;
            unsigned int row_1 = grain_1 / 16;
            unsigned int col_1 = grain_1 % 16 * 8;
            unsigned int token = partition_begin + tile * 64 + row_1;
            unsigned int relative_address = col_1 / 64 * 8192 + row_1 * 128 + (unsigned int)(col_1 % 64 * 2 ^ (row_1 & 7) << 4);
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(sk_addr + relative_address), "l"(kv + (k_base + token * 128 + col_1)), "r"((token < partition_end) ? 16 : 0));
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(sv_addr + relative_address), "l"(kv + (v_base + token * 128 + col_1)), "r"((token < partition_end) ? 16 : 0));
        }
        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        asm volatile("barrier.sync 8, 128;" ::: "memory");
        if (warp == 0) {
            if (elect_sync()) {
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_a_lo_0 = ((sq_addr) >> 4) & 0x3FFF;
                int _mma_b_lo_0 = ((sk_addr) >> 4) & 0x3FFF;
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
            "mov.b32 id, 68157456;\n\t"
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
            "add.u32 blo, blo, 506;\n\t"
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
                tcgen05_commit(mma_done_addr);
            }
        }
        mbarrier_wait(mma_done_addr, 0);
        asm volatile("tcgen05.fence::after_thread_sync;");
        float _tmem_load_0[32];
        asm volatile(
            "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 32;"
            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[31]))
            : "r"(taddr + row_addr));
        const float2 _scale2_0 = {scale, scale};
        #pragma unroll
        for (int _ls = 0; _ls < 16; _ls++)
            mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_0)[_ls], _scale2_0);
        #pragma unroll
        for (int col_2 = 0; col_2 < 32; col_2++) {
            unsigned int token_1 = tile * 64 + col_half * 32 + (unsigned int)col_2;
            if (my_row >= ratio || token_1 >= partition_length) {
                _tmem_load_0[col_2] = -SM110_XQA_INF;
            }
        }
        float2 _reg_reduce_max2_1 = {-SM110_XQA_INF, -SM110_XQA_INF};
        _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(_tmem_load_0[0], _tmem_load_0[1]));
        _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(_tmem_load_0[2], _tmem_load_0[3]));
        _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(_tmem_load_0[4], _tmem_load_0[5]));
        _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(_tmem_load_0[6], _tmem_load_0[7]));
        _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(_tmem_load_0[8], _tmem_load_0[9]));
        _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(_tmem_load_0[10], _tmem_load_0[11]));
        _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(_tmem_load_0[12], _tmem_load_0[13]));
        _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(_tmem_load_0[14], _tmem_load_0[15]));
        _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(_tmem_load_0[16], _tmem_load_0[17]));
        _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(_tmem_load_0[18], _tmem_load_0[19]));
        _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(_tmem_load_0[20], _tmem_load_0[21]));
        _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(_tmem_load_0[22], _tmem_load_0[23]));
        _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(_tmem_load_0[24], _tmem_load_0[25]));
        _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(_tmem_load_0[26], _tmem_load_0[27]));
        _reg_reduce_max2_1.x = max_noftz(_reg_reduce_max2_1.x, max_noftz(_tmem_load_0[28], _tmem_load_0[29]));
        _reg_reduce_max2_1.y = max_noftz(_reg_reduce_max2_1.y, max_noftz(_tmem_load_0[30], _tmem_load_0[31]));
        float _tmem_load_0_max = row_max_reduce(_reg_reduce_max2_1);
        float maximum = _tmem_load_0_max;
        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, maximum, 16);
        float _max_1 = max_noftz(maximum, _shfl_xor_0);
        maximum = _max_1;
        float _max_2 = max_noftz(maximum, row_max);
        float new_max = _max_2;
        float safe_max = ((new_max == -SM110_XQA_INF) ? 0.0f : new_max);
        float _exp2_0 = approx_exp2(row_max - safe_max);
        float correction = _exp2_0;
        const float2 _fma_b2_2 = {1.0f, 1.0f};
        const float2 _fma_c2_3 = {-safe_max, -safe_max};
        float2 _fma_pair_4 = fma_f32x2(make_float2(_tmem_load_0[0], _tmem_load_0[1]), _fma_b2_2, _fma_c2_3);
        _tmem_load_0[0] = _fma_pair_4.x;
        _tmem_load_0[1] = _fma_pair_4.y;
        float2 _fma_pair_5 = fma_f32x2(make_float2(_tmem_load_0[2], _tmem_load_0[3]), _fma_b2_2, _fma_c2_3);
        _tmem_load_0[2] = _fma_pair_5.x;
        _tmem_load_0[3] = _fma_pair_5.y;
        float2 _fma_pair_6 = fma_f32x2(make_float2(_tmem_load_0[4], _tmem_load_0[5]), _fma_b2_2, _fma_c2_3);
        _tmem_load_0[4] = _fma_pair_6.x;
        _tmem_load_0[5] = _fma_pair_6.y;
        float2 _fma_pair_7 = fma_f32x2(make_float2(_tmem_load_0[6], _tmem_load_0[7]), _fma_b2_2, _fma_c2_3);
        _tmem_load_0[6] = _fma_pair_7.x;
        _tmem_load_0[7] = _fma_pair_7.y;
        float2 _fma_pair_8 = fma_f32x2(make_float2(_tmem_load_0[8], _tmem_load_0[9]), _fma_b2_2, _fma_c2_3);
        _tmem_load_0[8] = _fma_pair_8.x;
        _tmem_load_0[9] = _fma_pair_8.y;
        float2 _fma_pair_9 = fma_f32x2(make_float2(_tmem_load_0[10], _tmem_load_0[11]), _fma_b2_2, _fma_c2_3);
        _tmem_load_0[10] = _fma_pair_9.x;
        _tmem_load_0[11] = _fma_pair_9.y;
        float2 _fma_pair_10 = fma_f32x2(make_float2(_tmem_load_0[12], _tmem_load_0[13]), _fma_b2_2, _fma_c2_3);
        _tmem_load_0[12] = _fma_pair_10.x;
        _tmem_load_0[13] = _fma_pair_10.y;
        float2 _fma_pair_11 = fma_f32x2(make_float2(_tmem_load_0[14], _tmem_load_0[15]), _fma_b2_2, _fma_c2_3);
        _tmem_load_0[14] = _fma_pair_11.x;
        _tmem_load_0[15] = _fma_pair_11.y;
        float2 _fma_pair_12 = fma_f32x2(make_float2(_tmem_load_0[16], _tmem_load_0[17]), _fma_b2_2, _fma_c2_3);
        _tmem_load_0[16] = _fma_pair_12.x;
        _tmem_load_0[17] = _fma_pair_12.y;
        float2 _fma_pair_13 = fma_f32x2(make_float2(_tmem_load_0[18], _tmem_load_0[19]), _fma_b2_2, _fma_c2_3);
        _tmem_load_0[18] = _fma_pair_13.x;
        _tmem_load_0[19] = _fma_pair_13.y;
        float2 _fma_pair_14 = fma_f32x2(make_float2(_tmem_load_0[20], _tmem_load_0[21]), _fma_b2_2, _fma_c2_3);
        _tmem_load_0[20] = _fma_pair_14.x;
        _tmem_load_0[21] = _fma_pair_14.y;
        float2 _fma_pair_15 = fma_f32x2(make_float2(_tmem_load_0[22], _tmem_load_0[23]), _fma_b2_2, _fma_c2_3);
        _tmem_load_0[22] = _fma_pair_15.x;
        _tmem_load_0[23] = _fma_pair_15.y;
        float2 _fma_pair_16 = fma_f32x2(make_float2(_tmem_load_0[24], _tmem_load_0[25]), _fma_b2_2, _fma_c2_3);
        _tmem_load_0[24] = _fma_pair_16.x;
        _tmem_load_0[25] = _fma_pair_16.y;
        float2 _fma_pair_17 = fma_f32x2(make_float2(_tmem_load_0[26], _tmem_load_0[27]), _fma_b2_2, _fma_c2_3);
        _tmem_load_0[26] = _fma_pair_17.x;
        _tmem_load_0[27] = _fma_pair_17.y;
        float2 _fma_pair_18 = fma_f32x2(make_float2(_tmem_load_0[28], _tmem_load_0[29]), _fma_b2_2, _fma_c2_3);
        _tmem_load_0[28] = _fma_pair_18.x;
        _tmem_load_0[29] = _fma_pair_18.y;
        float2 _fma_pair_19 = fma_f32x2(make_float2(_tmem_load_0[30], _tmem_load_0[31]), _fma_b2_2, _fma_c2_3);
        _tmem_load_0[30] = _fma_pair_19.x;
        _tmem_load_0[31] = _fma_pair_19.y;
        #pragma unroll
        for (int _le = 0; _le < 32; _le++) {
            _tmem_load_0[_le] = approx_exp2(_tmem_load_0[_le]);
        }
        float2 _reg_reduce_sum2_20 = make_float2(0.0f, 0.0f);
        _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(_tmem_load_0[0], _tmem_load_0[1]));
        _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(_tmem_load_0[2], _tmem_load_0[3]));
        _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(_tmem_load_0[4], _tmem_load_0[5]));
        _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(_tmem_load_0[6], _tmem_load_0[7]));
        _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(_tmem_load_0[8], _tmem_load_0[9]));
        _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(_tmem_load_0[10], _tmem_load_0[11]));
        _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(_tmem_load_0[12], _tmem_load_0[13]));
        _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(_tmem_load_0[14], _tmem_load_0[15]));
        _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(_tmem_load_0[16], _tmem_load_0[17]));
        _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(_tmem_load_0[18], _tmem_load_0[19]));
        _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(_tmem_load_0[20], _tmem_load_0[21]));
        _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(_tmem_load_0[22], _tmem_load_0[23]));
        _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(_tmem_load_0[24], _tmem_load_0[25]));
        _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(_tmem_load_0[26], _tmem_load_0[27]));
        _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(_tmem_load_0[28], _tmem_load_0[29]));
        _reg_reduce_sum2_20 = add_f32x2(_reg_reduce_sum2_20, make_float2(_tmem_load_0[30], _tmem_load_0[31]));
        float _tmem_load_0_sum = _reg_reduce_sum2_20.x + _reg_reduce_sum2_20.y;
        float half_sum = _tmem_load_0_sum;
        float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, half_sum, 16);
        float tile_sum = half_sum + _shfl_xor_1;
        row_sum = row_sum * correction + tile_sum;
        row_max = new_max;
        unsigned int packed[16];
        #pragma unroll
        for (int _lp = 0; _lp < 16; _lp++) {
            __half2 _h2 = __float22half2_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
            packed[_lp] = *(uint32_t*)&_h2;
        }
        asm volatile(
            "tcgen05.st.sync.aligned.16x32bx2.x16.b32"
            " [%0], 16, {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
            :: "r"(taddr + 32 + row_addr), "r"(*reinterpret_cast<const uint32_t*>(&packed[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed[3])), "r"(*reinterpret_cast<const uint32_t*>(&packed[4])), "r"(*reinterpret_cast<const uint32_t*>(&packed[5])), "r"(*reinterpret_cast<const uint32_t*>(&packed[6])), "r"(*reinterpret_cast<const uint32_t*>(&packed[7])), "r"(*reinterpret_cast<const uint32_t*>(&packed[8])), "r"(*reinterpret_cast<const uint32_t*>(&packed[9])), "r"(*reinterpret_cast<const uint32_t*>(&packed[10])), "r"(*reinterpret_cast<const uint32_t*>(&packed[11])), "r"(*reinterpret_cast<const uint32_t*>(&packed[12])), "r"(*reinterpret_cast<const uint32_t*>(&packed[13])), "r"(*reinterpret_cast<const uint32_t*>(&packed[14])), "r"(*reinterpret_cast<const uint32_t*>(&packed[15])));
        if (tile > 0) {
            float _tmem_load_1[64];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[31]))
                : "r"(taddr + 128 + row_addr));
            asm volatile(
                "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[32])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[33])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[34])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[35])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[36])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[37])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[38])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[39])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[40])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[41])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[42])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[43])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[44])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[45])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[46])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[47])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[48])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[49])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[50])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[51])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[52])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[53])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[54])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[55])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[56])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[57])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[58])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[59])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[60])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[61])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[62])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[63]))
                : "r"(taddr + 128 + row_addr + 32));
            const float2 _scale2_21 = {correction, correction};
            #pragma unroll
            for (int _ls = 0; _ls < 32; _ls++)
                mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_1)[_ls], _scale2_21);
            asm volatile(
                "tcgen05.st.sync.aligned.16x32bx2.x64.b32"
                " [%0], 64, {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64};"
                :: "r"(taddr + 128 + row_addr), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[15])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[16])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[17])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[18])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[19])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[20])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[21])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[22])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[23])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[24])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[25])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[26])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[27])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[28])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[29])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[30])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[31])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[32])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[33])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[34])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[35])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[36])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[37])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[38])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[39])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[40])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[41])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[42])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[43])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[44])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[45])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[46])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[47])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[48])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[49])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[50])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[51])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[52])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[53])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[54])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[55])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[56])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[57])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[58])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[59])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[60])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[61])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[62])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[63])));
        }
        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
        asm volatile("tcgen05.fence::before_thread_sync;");
        asm volatile("barrier.sync 8, 128;" ::: "memory");
        if (warp == 0) {
            if (elect_sync()) {
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_b_lo_1 = (((sv_addr) >> 4) & 0x3FFF) | 0x2000000;
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
            "mov.b32 id, 69271568;\n\t"
            "mov.b32 ta, %2;\n\t"
            "mov.b32 blo, %1;\n\t"
            "mov.b64 db, {blo, dhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
            "add.u32 ta, ta, 8;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 db, {blo, dhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
            "add.u32 ta, ta, 8;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 db, {blo, dhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
            "add.u32 ta, ta, 8;\n\t"
            "add.u32 blo, blo, 128;\n\t"
            "mov.b64 db, {blo, dhi};\n\t"
            "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
            "}\n"
            :: "r"(tmem_accum), "r"(_mma_b_lo_1), "r"(tmem_scores + 32), "r"(((tile == 0) ? 0 : 1)));
                tcgen05_commit(mma_done_addr);
            }
        }
        mbarrier_wait(mma_done_addr, 1);
        asm volatile("tcgen05.fence::after_thread_sync;");
        asm volatile("barrier.sync 8, 128;" ::: "memory");
    }
    float _tmem_load_2[64];
    asm volatile(
        "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[31]))
        : "r"(taddr + 128 + row_addr));
    asm volatile(
        "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[32])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[33])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[34])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[35])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[36])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[37])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[38])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[39])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[40])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[41])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[42])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[43])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[44])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[45])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[46])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[47])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[48])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[49])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[50])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[51])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[52])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[53])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[54])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[55])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[56])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[57])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[58])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[59])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[60])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[61])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[62])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[63]))
        : "r"(taddr + 128 + row_addr + 32));
    if (my_row < ratio) {
        unsigned int query_row = (batch * heads + head) * ratio + my_row;
        if (partitions == 1) {
            float _rcp_0 = __frcp_rn(row_sum);
            float inverse = ((row_sum > 0.0f) ? _rcp_0 : 0.0f);
            #pragma unroll
            for (int col_3 = 0; col_3 < 64; col_3 += 8) {
                {
                    const float2 _prescale2_22 = {inverse, inverse};
                    #if __CUDA_ARCH__ >= 1000
                    #pragma unroll
                    for (int _ps = 0; _ps < 4; _ps++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_2[col_3])[_ps], _prescale2_22);
                    #else
                    #pragma unroll
                    for (int _ps = 0; _ps < 8; _ps++)
                        _tmem_load_2[col_3 + _ps] *= inverse;
                    #endif
                    __half2 _pk[4];
                    _pk[0] = __floats2half2_rn(_tmem_load_2[col_3 + 0], _tmem_load_2[col_3 + 1]);
                    _pk[1] = __floats2half2_rn(_tmem_load_2[col_3 + 2], _tmem_load_2[col_3 + 3]);
                    _pk[2] = __floats2half2_rn(_tmem_load_2[col_3 + 4], _tmem_load_2[col_3 + 5]);
                    _pk[3] = __floats2half2_rn(_tmem_load_2[col_3 + 6], _tmem_load_2[col_3 + 7]);
                    *reinterpret_cast<uint4*>(&((__half*)(output + (query_row * 128 + col_half * 64 + (unsigned int)col_3)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                }
            }
        } else {
            unsigned int partial_row = query_row * partitions + partition;
            #pragma unroll
            for (int col_4 = 0; col_4 < 64; col_4 += 4) {
                {
                    float4 _v4 = make_float4(_tmem_load_2[col_4 + 0], _tmem_load_2[col_4 + 1], _tmem_load_2[col_4 + 2], _tmem_load_2[col_4 + 3]);
                    *reinterpret_cast<float4*>(partial + (partial_row * 128 + col_half * 64 + (unsigned int)col_4) + 0) = _v4;
                }
            }
            if (col_half == 0) {
                statistics[partial_row * 2] = row_max;
                statistics[partial_row * 2 + 1] = row_sum;
            }
        }
    }
    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::before_thread_sync;");
    asm volatile("barrier.sync 8, 128;" ::: "memory");
    if (warp == 0) {
        int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(256));
    }
    if (partitions > 1) {
        if (tid == 0) {
            unsigned int last_old = partitions - 1;
            uint32_t _atomic_inc_old_0;
            asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                : "=r"(_atomic_inc_old_0) : "l"(&counters[batch * heads + head]), "r"(static_cast<uint32_t>(last_old)) : "memory");
            unsigned int old = _atomic_inc_old_0;
            merge_flag[0] = (unsigned int)(old == last_old);
        }
        asm volatile("barrier.sync 8, 128;" ::: "memory");
        unsigned int is_last = merge_flag[0];
        if (is_last != 0) {
            #pragma unroll 1
            for (unsigned int group = 0; group < (ratio + 7) / 8; group++) {
                unsigned int local_row = group * 8 + warp * 2 + lane / 16;
                if (local_row < ratio) {
                    unsigned int merge_row = (batch * heads + head) * ratio + local_row;
                    unsigned int merge_col = lane % 16 * 8;
                    float numerator[8];
                    numerator[0] = 0.0f;
                    numerator[1] = 0.0f;
                    numerator[2] = 0.0f;
                    numerator[3] = 0.0f;
                    numerator[4] = 0.0f;
                    numerator[5] = 0.0f;
                    numerator[6] = 0.0f;
                    numerator[7] = 0.0f;
                    float denominator = 0.0f;
                    if (!0 || partitions > 4) {
                        float merge_max = -SM110_XQA_INF;
                        #pragma unroll 1
                        for (unsigned int shard = 0; shard < partitions; shard++) {
                            float candidate = statistics[(merge_row * partitions + shard) * 2];
                            float _max_3 = max_noftz(merge_max, candidate);
                            merge_max = _max_3;
                        }
                        float safe_merge_max = ((merge_max == -SM110_XQA_INF) ? 0.0f : merge_max);
                        #pragma unroll 1
                        for (unsigned int shard_1 = 0; shard_1 < partitions; shard_1++) {
                            unsigned int merge_partial_row = merge_row * partitions + shard_1;
                            float shard_max = statistics[merge_partial_row * 2];
                            float shard_sum = statistics[merge_partial_row * 2 + 1];
                            float _exp2_1 = approx_exp2(shard_max - safe_merge_max);
                            float weight = _exp2_1;
                            float _vec_load_0[4];
                            {
                                float4 _v4 = *reinterpret_cast<const float4*>(partial + merge_partial_row * 128 + merge_col);
                                _vec_load_0[0 + 0] = _v4.x;
                                _vec_load_0[0 + 1] = _v4.y;
                                _vec_load_0[0 + 2] = _v4.z;
                                _vec_load_0[0 + 3] = _v4.w;
                            }
                            float _vec_load_1[4];
                            {
                                float4 _v4 = *reinterpret_cast<const float4*>(partial + merge_partial_row * 128 + merge_col + 4);
                                _vec_load_1[0 + 0] = _v4.x;
                                _vec_load_1[0 + 1] = _v4.y;
                                _vec_load_1[0 + 2] = _v4.z;
                                _vec_load_1[0 + 3] = _v4.w;
                            }
                            #pragma unroll
                            for (int element = 0; element < 4; element++) {
                                float _fma_0 = __fmaf_rn(_vec_load_0[element], weight, numerator[element]);
                                numerator[element] = _fma_0;
                                float _fma_1 = __fmaf_rn(_vec_load_1[element], weight, numerator[element + 4]);
                                numerator[element + 4] = _fma_1;
                            }
                            float _fma_2 = __fmaf_rn(shard_sum, weight, denominator);
                            denominator = _fma_2;
                        }
                    }
                    float _rcp_1 = __frcp_rn(denominator);
                    float inverse_1 = ((denominator > 0.0f) ? _rcp_1 : 0.0f);
                    {
                        const float2 _prescale2_25 = {inverse_1, inverse_1};
                        #if __CUDA_ARCH__ >= 1000
                        #pragma unroll
                        for (int _ps = 0; _ps < 4; _ps++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(&numerator[0])[_ps], _prescale2_25);
                        #else
                        #pragma unroll
                        for (int _ps = 0; _ps < 8; _ps++)
                            numerator[0 + _ps] *= inverse_1;
                        #endif
                        __half2 _pk[4];
                        _pk[0] = __floats2half2_rn(numerator[0 + 0], numerator[0 + 1]);
                        _pk[1] = __floats2half2_rn(numerator[0 + 2], numerator[0 + 3]);
                        _pk[2] = __floats2half2_rn(numerator[0 + 4], numerator[0 + 5]);
                        _pk[3] = __floats2half2_rn(numerator[0 + 6], numerator[0 + 7]);
                        *reinterpret_cast<uint4*>(&((__half*)(output + (merge_row * 128 + merge_col)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                    }
                }
            }
        }
    }

    // Cleanup
    __syncthreads();
}

} // extern "C"
