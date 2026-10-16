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
#define TMEM_NCOLS 512
#define TMEM_SCORES_OFFSET 0
#define TMEM_ACCUM_0_OFFSET 256
#define TMEM_ACCUM_1_OFFSET 384
#define NUM_MAIN_STAGES 1
#define SMEM_SQ_OFF 0
#define SMEM_SQ_STAGE_BYTES 65536
#define SMEM_SQ_STRIDE 65536
#define SMEM_SK_OFF 65536
#define SMEM_SK_STAGE_BYTES 65536
#define SMEM_SK_STRIDE 65536
#define SMEM_SV_0_OFF 131072
#define SMEM_SV_0_STAGE_BYTES 16384
#define SMEM_SV_0_STRIDE 16384
#define SMEM_SV_1_OFF 147456
#define SMEM_SV_1_STAGE_BYTES 16384
#define SMEM_SV_1_STRIDE 16384
#define SMEM_TOTAL 164864

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

__global__ __launch_bounds__(256, 1) void
kernel_sm110_xqa_tree_fp16_paged(unsigned int q_seq_len, unsigned int num_kv_heads, unsigned int head_group_size, const unsigned int* __restrict__ q_cu_seq_lens, float attention_scale, __half* __restrict__ output, const __half* __restrict__ q, const unsigned int* __restrict__ mask, const uint8_t* __restrict__ cache, const int* __restrict__ sequence_lengths, const int* __restrict__ page_list, unsigned int capacity, unsigned int max_pages, float k_cache_scale, float v_cache_scale)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem + 163840;
    #define mma_done_addr (mbar_base + 0)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    __half* sq = reinterpret_cast<__half*>(smem_raw + 0);
    const int sq_addr = smem + 0;
    __half* sk = reinterpret_cast<__half*>(smem_raw + 65536);
    const int sk_addr = smem + 65536;
    __half* sv_0 = reinterpret_cast<__half*>(smem_raw + 131072);
    const int sv_0_addr = smem + 131072;
    __half* sv_1 = reinterpret_cast<__half*>(smem_raw + 147456);
    const int sv_1_addr = smem + 147456;

    // Mbarrier init (1 pipeline groups, 0 ordered-sequence groups, 1 barriers)
    // Mbarriers at smem_raw[163840..163848)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // mma_done: 1 barriers, init_count=1
            mbarrier_init(smem + 163840, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 163848);
    if (warp == 0) {
        int _tmem_hold = smem + 163848;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_scores = taddr;
    const int tmem_accum_0 = taddr + 256;
    const int tmem_accum_1 = taddr + 384;

    // === Task calls (dependency order) ===
    unsigned int actual_q = q_seq_len;
    unsigned int request_offset = q_seq_len * (unsigned int)blockIdx.z;
    if ((unsigned long long)q_cu_seq_lens != 0) {
        request_offset = q_cu_seq_lens[blockIdx.z];
        actual_q = q_cu_seq_lens[blockIdx.z + 1] - request_offset;
    }
    unsigned int rows_per_head = (q_seq_len * head_group_size + 63) / 64;
    unsigned int head = (unsigned int)blockIdx.y / rows_per_head;
    unsigned int row_begin = (unsigned int)blockIdx.y % rows_per_head * 64;
    unsigned int q_heads = num_kv_heads * head_group_size;
    unsigned int output_col = blockIdx.x * 256;
    unsigned int length = sequence_lengths[blockIdx.z];
    unsigned int num_blocks = (length + 63) / 64;
    unsigned int prefix = length - actual_q;
    unsigned int mask_stride = (q_seq_len + 31) / 32;
    float scale = attention_scale * 1.4426950408889634f;
    float value_scale = 1.0f;
    #pragma unroll 1
    for (int copy = 0; copy < 16; copy++) {
        unsigned int grain = copy * 256 + tid;
        unsigned int row = grain / 64;
        unsigned int col = grain % 64 * 8;
        unsigned int head_token = row_begin + row;
        unsigned int source_head = (request_offset + head_token / head_group_size) * q_heads + head * head_group_size + head_token % head_group_size;
        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
            :: "r"(sq_addr + col / 64 * 8192 + row * 128 + (unsigned int)(col % 64 * 2 ^ (row & 7) << 4)), "l"(q + (source_head * 512 + col)), "r"((head_token < actual_q * head_group_size) ? 16 : 0));
    }
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group 0;");
    asm volatile("barrier.sync 8, 256;" ::: "memory");
    unsigned int my_row = warp * 16 + lane % 16;
    unsigned int row_addr = warp * 32 << 16;
    unsigned int col_half = lane / 16;
    unsigned int query_token = (row_begin + my_row) / head_group_size;
    bool row_valid = row_begin + my_row < actual_q * head_group_size;
    float row_max = -SM110_XQA_INF;
    float row_sum = 0.0f;
    if (num_blocks > 0) {
        int tile_page = 0;
        if (length > 0) {
            tile_page = page_list[(unsigned int)(blockIdx.z * 2) * max_pages];
        }
        #pragma unroll 1
        for (int copy_1 = 0; copy_1 < 16; copy_1++) {
            unsigned int grain_1 = copy_1 * 256 + tid;
            unsigned int row_1 = grain_1 / 64;
            unsigned int col_1 = grain_1 % 64 * 8;
            unsigned int token = row_1;
            bool valid = token < length;
            unsigned int source_row = 0;
            if (token < length) {
                source_row = ((unsigned int)tile_page * 128 + row_1) * num_kv_heads + head;
            }
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(sk_addr + col_1 / 64 * 8192 + row_1 * 128 + (unsigned int)(col_1 % 64 * 2 ^ (row_1 & 7) << 4)), "l"(reinterpret_cast<const __half*>(cache) + (source_row * 512 + col_1)), "r"((valid) ? 16 : 0));
        }
        asm volatile("cp.async.commit_group;");
    }
    #pragma unroll 1
    for (unsigned int tile = 0; tile < num_blocks; tile++) {
        asm volatile("cp.async.wait_group 0;");
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        asm volatile("barrier.sync 8, 256;" ::: "memory");
        if (warp == 0) {
            if (elect_sync()) {
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_a_lo_0 = (((sq_addr) >> 4) & 0x3FFF) + (0) * 4096;
                int _mma_b_lo_0 = (((sk_addr) >> 4) & 0x3FFF) + (0) * 4096;
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
                int _mma_a_lo_1 = (((sq_addr + 16384) >> 4) & 0x3FFF) + (0) * 4096;
                int _mma_b_lo_1 = (((sk_addr + 16384) >> 4) & 0x3FFF) + (0) * 4096;
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
            :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"(tmem_scores), "r"(1));
                int _mma_a_lo_2 = (((sq_addr + 32768) >> 4) & 0x3FFF) + (0) * 4096;
                int _mma_b_lo_2 = (((sk_addr + 32768) >> 4) & 0x3FFF) + (0) * 4096;
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
            :: "r"(_mma_a_lo_2), "r"(_mma_b_lo_2), "r"(tmem_scores), "r"(1));
                int _mma_a_lo_3 = (((sq_addr + 49152) >> 4) & 0x3FFF) + (0) * 4096;
                int _mma_b_lo_3 = (((sk_addr + 49152) >> 4) & 0x3FFF) + (0) * 4096;
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
            :: "r"(_mma_a_lo_3), "r"(_mma_b_lo_3), "r"(tmem_scores), "r"(1));
                tcgen05_commit(mma_done_addr);
            }
        }
        int tile_page_1 = 0;
        if (length > tile * 64) {
            tile_page_1 = page_list[(unsigned int)(blockIdx.z * 2 + 1) * max_pages + tile * 64 / 128];
        }
        #pragma unroll 1
        for (int copy_2 = 0; copy_2 < 8; copy_2++) {
            unsigned int grain_2 = copy_2 * 256 + tid;
            unsigned int row_2 = grain_2 / 32;
            unsigned int col_2 = grain_2 % 32 * 8;
            unsigned int token_1 = tile * 64 + row_2;
            bool valid_1 = token_1 < length;
            unsigned int source_row_1 = 0;
            if (token_1 < length) {
                source_row_1 = ((unsigned int)tile_page_1 * 128 + tile * 64 % 128 + row_2) * num_kv_heads + head;
            }
            asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                :: "r"(sv_0_addr + col_2 / 64 * 8192 + row_2 * 128 + (unsigned int)(col_2 % 64 * 2 ^ (row_2 & 7) << 4)), "l"(reinterpret_cast<const __half*>(cache) + (source_row_1 * 512 + output_col + col_2)), "r"((valid_1) ? 16 : 0));
        }
        asm volatile("cp.async.commit_group;");
        mbarrier_wait(mma_done_addr, 0);
        asm volatile("tcgen05.fence::after_thread_sync;");
        asm volatile("cp.async.wait_group 0;");
        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
        if (num_blocks > tile + 1) {
            int tile_page_0 = 0;
            if (length > (tile + 1) * 64) {
                tile_page_0 = page_list[(unsigned int)(blockIdx.z * 2) * max_pages + (tile + 1) * 64 / 128];
            }
            #pragma unroll 1
            for (int copy_3 = 0; copy_3 < 16; copy_3++) {
                unsigned int grain_3 = copy_3 * 256 + tid;
                unsigned int row_3 = grain_3 / 64;
                unsigned int col_3 = grain_3 % 64 * 8;
                unsigned int token_2 = (tile + 1) * 64 + row_3;
                bool valid_2 = token_2 < length;
                unsigned int source_row_2 = 0;
                if (token_2 < length) {
                    source_row_2 = ((unsigned int)tile_page_0 * 128 + (tile + 1) * 64 % 128 + row_3) * num_kv_heads + head;
                }
                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                    :: "r"(sk_addr + col_3 / 64 * 8192 + row_3 * 128 + (unsigned int)(col_3 % 64 * 2 ^ (row_3 & 7) << 4)), "l"(reinterpret_cast<const __half*>(cache) + (source_row_2 * 512 + col_3)), "r"((valid_2) ? 16 : 0));
            }
            asm volatile("cp.async.commit_group;");
        }
        if (warp < 4) {
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
            if (prefix > tile * 64 + 63) {
                if (row_begin + my_row >= actual_q * head_group_size) {
                    _tmem_load_0[0] = -SM110_XQA_INF;
                    _tmem_load_0[1] = -SM110_XQA_INF;
                    _tmem_load_0[2] = -SM110_XQA_INF;
                    _tmem_load_0[3] = -SM110_XQA_INF;
                    _tmem_load_0[4] = -SM110_XQA_INF;
                    _tmem_load_0[5] = -SM110_XQA_INF;
                    _tmem_load_0[6] = -SM110_XQA_INF;
                    _tmem_load_0[7] = -SM110_XQA_INF;
                    _tmem_load_0[8] = -SM110_XQA_INF;
                    _tmem_load_0[9] = -SM110_XQA_INF;
                    _tmem_load_0[10] = -SM110_XQA_INF;
                    _tmem_load_0[11] = -SM110_XQA_INF;
                    _tmem_load_0[12] = -SM110_XQA_INF;
                    _tmem_load_0[13] = -SM110_XQA_INF;
                    _tmem_load_0[14] = -SM110_XQA_INF;
                    _tmem_load_0[15] = -SM110_XQA_INF;
                    _tmem_load_0[16] = -SM110_XQA_INF;
                    _tmem_load_0[17] = -SM110_XQA_INF;
                    _tmem_load_0[18] = -SM110_XQA_INF;
                    _tmem_load_0[19] = -SM110_XQA_INF;
                    _tmem_load_0[20] = -SM110_XQA_INF;
                    _tmem_load_0[21] = -SM110_XQA_INF;
                    _tmem_load_0[22] = -SM110_XQA_INF;
                    _tmem_load_0[23] = -SM110_XQA_INF;
                    _tmem_load_0[24] = -SM110_XQA_INF;
                    _tmem_load_0[25] = -SM110_XQA_INF;
                    _tmem_load_0[26] = -SM110_XQA_INF;
                    _tmem_load_0[27] = -SM110_XQA_INF;
                    _tmem_load_0[28] = -SM110_XQA_INF;
                    _tmem_load_0[29] = -SM110_XQA_INF;
                    _tmem_load_0[30] = -SM110_XQA_INF;
                    _tmem_load_0[31] = -SM110_XQA_INF;
                }
            } else {
                #pragma unroll
                for (int col_4 = 0; col_4 < 32; col_4++) {
                    unsigned int token_3 = tile * 64 + col_half * 32 + (unsigned int)col_4;
                    bool visible = row_valid && token_3 < length;
                    if (visible && token_3 >= prefix) {
                        unsigned int tree_col = token_3 - prefix;
                        unsigned int bits = mask[(request_offset + query_token) * mask_stride + tree_col / 32];
                        visible = (bits >> tree_col % 32 & 1) != 0;
                    }
                    if (!visible) {
                        _tmem_load_0[col_4] = -SM110_XQA_INF;
                    }
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
            float tile_max = _tmem_load_0_max;
            float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, tile_max, 16);
            float _max_0 = max_noftz(tile_max, _shfl_xor_0);
            tile_max = _max_0;
            float _max_1 = max_noftz(tile_max, row_max);
            float new_max = _max_1;
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
                {
                    float _tmem_load_1[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[31]))
                        : "r"(taddr + 256 + row_addr));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[32])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[33])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[34])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[35])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[36])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[37])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[38])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[39])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[40])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[41])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[42])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[43])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[44])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[45])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[46])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[47])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[48])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[49])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[50])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[51])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[52])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[53])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[54])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[55])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[56])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[57])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[58])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[59])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[60])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[61])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[62])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[63]))
                        : "r"(taddr + 256 + row_addr + 32));
                    const float2 _scale2_21 = {correction, correction};
                    #pragma unroll
                    for (int _ls = 0; _ls < 32; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_1)[_ls], _scale2_21);
                    asm volatile(
                        "tcgen05.st.sync.aligned.16x32bx2.x64.b32"
                        " [%0], 64, {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64};"
                        :: "r"(taddr + 256 + row_addr), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[15])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[16])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[17])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[18])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[19])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[20])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[21])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[22])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[23])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[24])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[25])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[26])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[27])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[28])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[29])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[30])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[31])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[32])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[33])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[34])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[35])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[36])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[37])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[38])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[39])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[40])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[41])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[42])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[43])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[44])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[45])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[46])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[47])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[48])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[49])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[50])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[51])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[52])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[53])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[54])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[55])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[56])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[57])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[58])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[59])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[60])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[61])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[62])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_1[63])));
                }
                {
                    float _tmem_load_2[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[31]))
                        : "r"(taddr + 256 + 128 + row_addr));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[32])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[33])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[34])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[35])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[36])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[37])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[38])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[39])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[40])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[41])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[42])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[43])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[44])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[45])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[46])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[47])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[48])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[49])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[50])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[51])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[52])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[53])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[54])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[55])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[56])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[57])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[58])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[59])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[60])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[61])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[62])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[63]))
                        : "r"(taddr + 256 + 128 + row_addr + 32));
                    const float2 _scale2_22 = {correction, correction};
                    #pragma unroll
                    for (int _ls = 0; _ls < 32; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_2)[_ls], _scale2_22);
                    asm volatile(
                        "tcgen05.st.sync.aligned.16x32bx2.x64.b32"
                        " [%0], 64, {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64};"
                        :: "r"(taddr + 256 + 128 + row_addr), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[15])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[16])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[17])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[18])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[19])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[20])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[21])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[22])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[23])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[24])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[25])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[26])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[27])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[28])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[29])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[30])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[31])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[32])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[33])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[34])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[35])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[36])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[37])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[38])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[39])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[40])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[41])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[42])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[43])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[44])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[45])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[46])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[47])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[48])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[49])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[50])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[51])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[52])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[53])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[54])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[55])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[56])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[57])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[58])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[59])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[60])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[61])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[62])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_2[63])));
                }
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
        }
        asm volatile("barrier.sync 8, 256;" ::: "memory");
        if (warp == 0) {
            if (elect_sync()) {
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_b_lo_4 = (((sv_0_addr) >> 4) & 0x3FFF) | 0x2000000;
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
            :: "r"(tmem_accum_0), "r"(_mma_b_lo_4), "r"(tmem_scores + 32), "r"(((tile == 0) ? 0 : 1)));
                int _mma_b_lo_5 = (((sv_1_addr) >> 4) & 0x3FFF) | 0x2000000;
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
            :: "r"(tmem_accum_1), "r"(_mma_b_lo_5), "r"(tmem_scores + 32), "r"(((tile == 0) ? 0 : 1)));
                tcgen05_commit(mma_done_addr);
            }
        }
        mbarrier_wait(mma_done_addr, 1);
        asm volatile("tcgen05.fence::after_thread_sync;");
        asm volatile("barrier.sync 8, 256;" ::: "memory");
    }
    if (warp < 4) {
        {
            float _tmem_load_3[64];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[31]))
                : "r"(taddr + 256 + row_addr));
            asm volatile(
                "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[32])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[33])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[34])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[35])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[36])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[37])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[38])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[39])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[40])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[41])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[42])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[43])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[44])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[45])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[46])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[47])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[48])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[49])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[50])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[51])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[52])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[53])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[54])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[55])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[56])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[57])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[58])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[59])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[60])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[61])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[62])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[63]))
                : "r"(taddr + 256 + row_addr + 32));
            float _rcp_0 = __frcp_rn(row_sum);
            float inverse = ((row_sum > 0.0f) ? _rcp_0 : 0.0f);
            inverse = inverse * value_scale;
            if (row_valid) {
                unsigned int head_token_1 = row_begin + my_row;
                unsigned int out_head = (request_offset + head_token_1 / head_group_size) * q_heads + head * head_group_size + head_token_1 % head_group_size;
                #pragma unroll
                for (int col_5 = 0; col_5 < 64; col_5 += 8) {
                    {
                        const float2 _prescale2_23 = {inverse, inverse};
                        #if __CUDA_ARCH__ >= 1000
                        #pragma unroll
                        for (int _ps = 0; _ps < 4; _ps++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_3[col_5])[_ps], _prescale2_23);
                        #else
                        #pragma unroll
                        for (int _ps = 0; _ps < 8; _ps++)
                            _tmem_load_3[col_5 + _ps] *= inverse;
                        #endif
                        __half2 _pk[4];
                        _pk[0] = __floats2half2_rn(_tmem_load_3[col_5 + 0], _tmem_load_3[col_5 + 1]);
                        _pk[1] = __floats2half2_rn(_tmem_load_3[col_5 + 2], _tmem_load_3[col_5 + 3]);
                        _pk[2] = __floats2half2_rn(_tmem_load_3[col_5 + 4], _tmem_load_3[col_5 + 5]);
                        _pk[3] = __floats2half2_rn(_tmem_load_3[col_5 + 6], _tmem_load_3[col_5 + 7]);
                        *reinterpret_cast<uint4*>(&((__half*)(output + (out_head * 512 + output_col + col_half * 64 + (unsigned int)col_5)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                    }
                }
            }
        }
        {
            float _tmem_load_4[64];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[31]))
                : "r"(taddr + 256 + 128 + row_addr));
            asm volatile(
                "tcgen05.ld.sync.aligned.16x32bx2.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32], 64;"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[32])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[33])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[34])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[35])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[36])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[37])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[38])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[39])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[40])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[41])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[42])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[43])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[44])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[45])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[46])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[47])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[48])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[49])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[50])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[51])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[52])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[53])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[54])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[55])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[56])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[57])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[58])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[59])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[60])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[61])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[62])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[63]))
                : "r"(taddr + 256 + 128 + row_addr + 32));
            float _rcp_1 = __frcp_rn(row_sum);
            float inverse_1 = ((row_sum > 0.0f) ? _rcp_1 : 0.0f);
            inverse_1 = inverse_1 * value_scale;
            if (row_valid) {
                unsigned int head_token_2 = row_begin + my_row;
                unsigned int out_head_1 = (request_offset + head_token_2 / head_group_size) * q_heads + head * head_group_size + head_token_2 % head_group_size;
                #pragma unroll
                for (int col_6 = 0; col_6 < 64; col_6 += 8) {
                    {
                        const float2 _prescale2_24 = {inverse_1, inverse_1};
                        #if __CUDA_ARCH__ >= 1000
                        #pragma unroll
                        for (int _ps = 0; _ps < 4; _ps++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_4[col_6])[_ps], _prescale2_24);
                        #else
                        #pragma unroll
                        for (int _ps = 0; _ps < 8; _ps++)
                            _tmem_load_4[col_6 + _ps] *= inverse_1;
                        #endif
                        __half2 _pk[4];
                        _pk[0] = __floats2half2_rn(_tmem_load_4[col_6 + 0], _tmem_load_4[col_6 + 1]);
                        _pk[1] = __floats2half2_rn(_tmem_load_4[col_6 + 2], _tmem_load_4[col_6 + 3]);
                        _pk[2] = __floats2half2_rn(_tmem_load_4[col_6 + 4], _tmem_load_4[col_6 + 5]);
                        _pk[3] = __floats2half2_rn(_tmem_load_4[col_6 + 6], _tmem_load_4[col_6 + 7]);
                        *reinterpret_cast<uint4*>(&((__half*)(output + (out_head_1 * 512 + output_col + 128 + col_half * 64 + col_6)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                    }
                }
            }
        }
        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
        asm volatile("tcgen05.fence::before_thread_sync;");
    }
    asm volatile("barrier.sync 8, 256;" ::: "memory");
    if (warp == 0) {
        int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(512));
    }

    // Cleanup
    __syncthreads();
}

} // extern "C"
