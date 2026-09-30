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
static_assert(sizeof(uint64_t) == 8, "Cake requires an LP64 CUDA host ABI");
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) Sm110XqaTensorMap { uint64_t opaque[16]; };
struct __align__(64) Sm110XqaTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(Sm110XqaTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(Sm110XqaTensorMap64) == 64, "64-aligned tensor-map ABI alignment");

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
#define TMEM_S0_OFFSET 0
#define TMEM_S1_OFFSET 64
#define TMEM_P0_OFFSET 128
#define TMEM_P1_OFFSET 160
#define TMEM_O0_OFFSET 256
#define TMEM_O1_OFFSET 384
#define NUM_RING_STAGES 6
#define NUM_Q_PIPE_STAGES 4
#define NUM_S_PIPE_STAGES 2
#define NUM_P_PIPE_STAGES 2
#define SMEM_SQ_OFF 1024
#define SMEM_SQ_STAGE_BYTES 131072
#define SMEM_SQ_STRIDE 131072
#define SMEM_SK_OFF 132096
#define SMEM_SK_STAGE_BYTES 16384
#define SMEM_SK_STRIDE 16384
#define SMEM_SV_OFF 132096
#define SMEM_SV_STAGE_BYTES 16384
#define SMEM_SV_STRIDE 16384
#define SMEM_SSTAGE_OFF 1024
#define SMEM_SSTAGE_STAGE_BYTES 67584
#define SMEM_SSTAGE_STRIDE 67584
#define SMEM_TOTAL 230400

#if !defined(__CUDACC_RTC__)
#include <stddef.h>
#endif
struct __align__(8) KVCacheList {
    void* data;
    const int* sequence_lengths;
    unsigned int capacity;
};
static_assert(sizeof(KVCacheList) == 24, "KVCacheList size");
static_assert(__alignof__(KVCacheList) == 8, "KVCacheList alignment");
#if !defined(__CUDACC_RTC__)
static_assert(offsetof(KVCacheList, data) == 0, "KVCacheList.data offset");
static_assert(offsetof(KVCacheList, sequence_lengths) == 8, "KVCacheList.sequence_lengths offset");
static_assert(offsetof(KVCacheList, capacity) == 16, "KVCacheList.capacity offset");
#endif

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
        :: "l"(mbar_addr), "r"(count) : "memory");
}


__device__ __forceinline__ uint32_t mbarrier_try_wait_plain(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
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


__device__ __forceinline__ void elect_commit_cg2_multicast(int mbar_addr, uint16_t cta_mask) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "@leader tcgen05.commit.cta_group::2.mbarrier::arrive::one"
        ".shared::cluster.multicast::cluster.b64 [%0], %1;\n\t"
        "}\n"
        :: "r"(mbar_addr), "h"(cta_mask) : "memory");
}


__device__ __forceinline__ void elect_commit_cg1_multicast(int mbar_addr, uint16_t cta_mask) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.multicast::cluster.b64 [%0], %1;\n\t"
        "}\n"
        :: "r"(mbar_addr), "h"(cta_mask) : "memory");
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


__device__ __forceinline__ uint32_t smem_addr(const void* ptr) {
    uint32_t addr;
    asm("{\n\t"
        ".reg .u64 u64addr;\n\t"
        "cvta.to.shared.u64 u64addr, %1;\n\t"
        "cvt.u32.u64 %0, u64addr;\n\t"
        "}\n" : "=r"(addr) : "l"(ptr));
    return addr;
}


__device__ __forceinline__ uint32_t mapa_to_rank(uint32_t local_addr, uint32_t rank) {
    uint32_t remote;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
        : "=r"(remote) : "r"(local_addr), "r"(rank));
    return remote;
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

__device__ __forceinline__ void fma_f32x2_noftz_inplace(float2* a, float2 b, float2 c) {
    unsigned long long r;
    asm("fma.rn.f32x2 %0, %1, %2, %3;"
        : "=l"(r)
        : "l"(*(unsigned long long*)a), "l"(*(unsigned long long*)&b),
          "l"(*(unsigned long long*)&c));
    *(unsigned long long*)a = r;
}

__device__ __forceinline__ void mul_f32x2_inplace(float2* a, float2 b) {
    asm("mul.rn.f32x2 %0, %0, %1;"
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
    asm("mul.rn.f32x2 %0, %1, %2;"
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


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tcgen05_commit_cg2_multicast(int mbar_addr, uint16_t cta_mask) {
    asm volatile(
        "{\n\t"
        ".reg .b16 lo, hi;\n\t"
        "mov.b32 {lo, hi}, %1;\n\t"
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one"
        ".shared::cluster.multicast::cluster.b64 [%0], lo;\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"((uint32_t)cta_mask) : "memory");
}


__device__ __forceinline__ void tcgen05_commit_cg1_multicast(int mbar_addr, uint16_t cta_mask) {
    asm volatile(
        "{\n\t"
        ".reg .b16 lo, hi;\n\t"
        "mov.b32 {lo, hi}, %1;\n\t"
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.multicast::cluster.b64 [%0], lo;\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"((uint32_t)cta_mask) : "memory");
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


__device__ __forceinline__ void tmem_ld_x16_wait(float* dst, int addr) {
    tmem_ld_x16(dst, addr);
    asm volatile("tcgen05.wait::ld.sync.aligned;");
}

extern "C" {

__global__ __launch_bounds__(512, 1) __cluster_dims__(2,2,1) void
kernel_sm110_xqa_tree_fp16_contiguous_tmem_r2(const __grid_constant__ Sm110XqaTensorMap64 Q, const __grid_constant__ Sm110XqaTensorMap64 KV, unsigned int q_seq_len, unsigned int num_kv_heads, unsigned int head_group_size, const unsigned int* __restrict__ q_cu_seq_lens, float attention_scale, __half* __restrict__ output, const unsigned int* __restrict__ mask, KVCacheList kv_cache_list, unsigned int batch_size, float k_cache_scale, float v_cache_scale)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane = static_cast<uint32_t>(tid) & 31u;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define landed_addr (mbar_base + 0)
    #define ready_addr (mbar_base + 48)
    #define free_addr (mbar_base + 96)
    #define q_ready_addr (mbar_base + 144)
    #define s_full_addr (mbar_base + 176)
    #define s_empty_addr (mbar_base + 192)
    #define p_full_addr (mbar_base + 208)
    #define p_empty_addr (mbar_base + 224)
    #define tmem_retire_addr (mbar_base + 240)
    #define o_ready_addr (mbar_base + 248)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * (gridDim.y / 2) + (blockIdx.y / 2)) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * (gridDim.y / 2) * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    __half* sq = reinterpret_cast<__half*>(smem_raw + 1024);
    const int sq_addr = smem + 1024;
    __half* sk = reinterpret_cast<__half*>(smem_raw + 132096);
    const int sk_addr = smem + 132096;
    __half* sv = reinterpret_cast<__half*>(smem_raw + 132096);
    const int sv_addr = smem + 132096;
    unsigned int* sstage = reinterpret_cast<unsigned int*>(smem_raw + 1024);
    const int sstage_addr = smem + 1024;
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Q))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&KV))) : "memory");
    int _mma_base_lo_0 = ((sq_addr) >> 4) & 0x3FFF;
    int _mma_base_lo_1 = ((sk_addr) >> 4) & 0x3FFF;
    int _mma_base_lo_2 = ((sq_addr + 32768) >> 4) & 0x3FFF;
    int _mma_base_lo_3 = ((sq_addr + 65536) >> 4) & 0x3FFF;
    int _mma_base_lo_4 = ((sq_addr + 98304) >> 4) & 0x3FFF;
    int _mma_base_lo_5 = (((sv_addr) >> 4) & 0x3FFF) | 0x2000000;
    int _mma_base_lo_6 = (((sv_addr + 2048) >> 4) & 0x3FFF) | 0x2000000;
    int _mma_base_lo_7 = (((sv_addr + 4096) >> 4) & 0x3FFF) | 0x2000000;
    int _mma_base_lo_8 = (((sv_addr + 6144) >> 4) & 0x3FFF) | 0x2000000;

    // Mbarrier init (10 pipeline groups, 0 ordered-sequence groups, 32 barriers)
    // Mbarriers at smem_raw[0..256)


    // Mbarrier init group 1
    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'ring' ---
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 8);
            mbarrier_init(smem + 56, 8);
            mbarrier_init(smem + 64, 8);
            mbarrier_init(smem + 72, 8);
            mbarrier_init(smem + 80, 8);
            mbarrier_init(smem + 88, 8);
            mbarrier_init(smem + 96, 4);
            mbarrier_init(smem + 104, 4);
            mbarrier_init(smem + 112, 4);
            mbarrier_init(smem + 120, 4);
            mbarrier_init(smem + 128, 4);
            mbarrier_init(smem + 136, 4);
            // --- pipeline 'q_pipe' ---
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            // --- pipeline 's_pipe' ---
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            mbarrier_init(smem + 192, 128);
            mbarrier_init(smem + 200, 128);
            // --- pipeline 'p_pipe' ---
            mbarrier_init(smem + 208, 128);
            mbarrier_init(smem + 216, 128);
            mbarrier_init(smem + 224, 1);
            mbarrier_init(smem + 232, 1);
            mbarrier_init(smem + 240, 128);
            mbarrier_init(smem + 248, 1);
            asm volatile("fence.mbarrier_init.release.cluster;");
        }
    }
    __syncthreads();

    __syncwarp();

    // Source-retained mbarrier initialization completion
    asm volatile("barrier.sync 1, 512;" ::: "memory");

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 256);
    if (warp == 12) {
        int _tmem_hold = smem + 256;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    // barrier.cluster.wait deferred (WarpConfig.cluster_init_wait_warps): role entries, or the schedule's ClusterSyncWait when the tuple is empty

    const int taddr = 0;

    // Kernel post-init ops
    const int tmem_s0 = taddr;
    const int tmem_s1 = taddr + 64;
    const int tmem_p0 = taddr + 128;
    const int tmem_p1 = taddr + 160;
    const int tmem_o0 = taddr + 256;
    const int tmem_o1 = taddr + 384;

    // ---- Role: softmax ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 192;");
        // Deferred post-initialization cluster wait (WarpConfig.cluster_init_wait_warps)
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        { // softmax_main
            unsigned int actual_q = q_seq_len;
            unsigned int request_offset = q_seq_len * (unsigned int)blockIdx.z;
            if ((unsigned long long)q_cu_seq_lens != 0) {
                request_offset = q_cu_seq_lens[blockIdx.z];
                actual_q = q_cu_seq_lens[blockIdx.z + 1] - request_offset;
            }
            unsigned int blocks_per_head = (unsigned int)gridDim.y / num_kv_heads;
            unsigned int head = (unsigned int)blockIdx.y / blocks_per_head;
            unsigned int row_begin = (unsigned int)blockIdx.y % blocks_per_head * 128;
            unsigned int output_col = blockIdx.x * 256;
            int _vec_load_3[1];
            {
                _vec_load_3[0] = *reinterpret_cast<const int*>(kv_cache_list.sequence_lengths + blockIdx.z);
            }
            unsigned int length = (unsigned int)_vec_load_3[0];
            unsigned int num_tiles = (length + 63) / 64;
            int row_base = warp % 4 * 32 << 16;
            unsigned int my_row = warp % 4 * 32 + lane;
            unsigned int head_token = row_begin + my_row;
            int row_valid = ((head_token < actual_q * 2) ? 1 : 0);
            unsigned int query_token = head_token / 2;
            unsigned int prefix = length - actual_q;
            unsigned int mask_stride = (q_seq_len + 31) / 32;
            unsigned int _min_3 = ((query_token) < (actual_q - 1) ? (query_token) : (actual_q - 1));
            unsigned int mask_row = (request_offset + _min_3) * mask_stride;
            unsigned int _vec_load_4[1];
            {
                _vec_load_4[0] = *reinterpret_cast<const unsigned int*>(mask + mask_row);
            }
            unsigned int mw0 = _vec_load_4[0];
            unsigned int mw1 = 0;
            unsigned int mw2 = 0;
            unsigned int mw3 = 0;
            unsigned int zero32 = 0;
            if (mask_stride > 1) {
                unsigned int _vec_load_5[1];
                {
                    _vec_load_5[0] = *reinterpret_cast<const unsigned int*>(mask + (mask_row + 1));
                }
                mw1 = _vec_load_5[0];
            }
            if (mask_stride > 2) {
                unsigned int _vec_load_6[1];
                {
                    _vec_load_6[0] = *reinterpret_cast<const unsigned int*>(mask + (mask_row + 2));
                }
                mw2 = _vec_load_6[0];
            }
            if (mask_stride > 3) {
                unsigned int _vec_load_7[1];
                {
                    _vec_load_7[0] = *reinterpret_cast<const unsigned int*>(mask + (mask_row + 3));
                }
                mw3 = _vec_load_7[0];
            }
            unsigned int pk[16];
            float scale = attention_scale * 1.4426950408889634f * k_cache_scale;
            float m_used = -SM110_XQA_INF;
            float row_sum = 0.0f;
            unsigned int sm_s_stage = 0;
            unsigned int sm_p_stage = 0;
            unsigned int packed[32];
            unsigned int _phase_s_full = 0;
            unsigned int _phase_p_empty = 1;
            #pragma unroll 1
            for (unsigned int tile = 0; tile < num_tiles; tile++) {
                mbarrier_wait_hint(s_full_addr + (sm_s_stage) * 8, _phase_s_full, 10000000);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int s_addr = taddr + sm_s_stage * 64 + (unsigned int)row_base;
                float _tmem_load_0[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                    : "r"(s_addr));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                float _tmem_load_1[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31])
                    : "r"(s_addr + 32));
                asm volatile("tcgen05.wait::ld.sync.aligned;");
                mbarrier_arrive(s_empty_addr + (sm_s_stage) * 8);
                sm_s_stage += 1;
                if (sm_s_stage == 2) { sm_s_stage = 0; _phase_s_full ^= 1; }
                unsigned int tile_token0 = tile * 64;
                if (row_valid == 0) {
                    if (!(0 & (1u << 0))) _tmem_load_0[0] = -SM110_XQA_INF;
                    if (!(0 & (1u << 1))) _tmem_load_0[1] = -SM110_XQA_INF;
                    if (!(0 & (1u << 2))) _tmem_load_0[2] = -SM110_XQA_INF;
                    if (!(0 & (1u << 3))) _tmem_load_0[3] = -SM110_XQA_INF;
                    if (!(0 & (1u << 4))) _tmem_load_0[4] = -SM110_XQA_INF;
                    if (!(0 & (1u << 5))) _tmem_load_0[5] = -SM110_XQA_INF;
                    if (!(0 & (1u << 6))) _tmem_load_0[6] = -SM110_XQA_INF;
                    if (!(0 & (1u << 7))) _tmem_load_0[7] = -SM110_XQA_INF;
                    if (!(0 & (1u << 8))) _tmem_load_0[8] = -SM110_XQA_INF;
                    if (!(0 & (1u << 9))) _tmem_load_0[9] = -SM110_XQA_INF;
                    if (!(0 & (1u << 10))) _tmem_load_0[10] = -SM110_XQA_INF;
                    if (!(0 & (1u << 11))) _tmem_load_0[11] = -SM110_XQA_INF;
                    if (!(0 & (1u << 12))) _tmem_load_0[12] = -SM110_XQA_INF;
                    if (!(0 & (1u << 13))) _tmem_load_0[13] = -SM110_XQA_INF;
                    if (!(0 & (1u << 14))) _tmem_load_0[14] = -SM110_XQA_INF;
                    if (!(0 & (1u << 15))) _tmem_load_0[15] = -SM110_XQA_INF;
                    if (!(0 & (1u << 16))) _tmem_load_0[16] = -SM110_XQA_INF;
                    if (!(0 & (1u << 17))) _tmem_load_0[17] = -SM110_XQA_INF;
                    if (!(0 & (1u << 18))) _tmem_load_0[18] = -SM110_XQA_INF;
                    if (!(0 & (1u << 19))) _tmem_load_0[19] = -SM110_XQA_INF;
                    if (!(0 & (1u << 20))) _tmem_load_0[20] = -SM110_XQA_INF;
                    if (!(0 & (1u << 21))) _tmem_load_0[21] = -SM110_XQA_INF;
                    if (!(0 & (1u << 22))) _tmem_load_0[22] = -SM110_XQA_INF;
                    if (!(0 & (1u << 23))) _tmem_load_0[23] = -SM110_XQA_INF;
                    if (!(0 & (1u << 24))) _tmem_load_0[24] = -SM110_XQA_INF;
                    if (!(0 & (1u << 25))) _tmem_load_0[25] = -SM110_XQA_INF;
                    if (!(0 & (1u << 26))) _tmem_load_0[26] = -SM110_XQA_INF;
                    if (!(0 & (1u << 27))) _tmem_load_0[27] = -SM110_XQA_INF;
                    if (!(0 & (1u << 28))) _tmem_load_0[28] = -SM110_XQA_INF;
                    if (!(0 & (1u << 29))) _tmem_load_0[29] = -SM110_XQA_INF;
                    if (!(0 & (1u << 30))) _tmem_load_0[30] = -SM110_XQA_INF;
                    if (!(0 & (1u << 31))) _tmem_load_0[31] = -SM110_XQA_INF;
                    if (!(0 & (1u << 0))) _tmem_load_1[0] = -SM110_XQA_INF;
                    if (!(0 & (1u << 1))) _tmem_load_1[1] = -SM110_XQA_INF;
                    if (!(0 & (1u << 2))) _tmem_load_1[2] = -SM110_XQA_INF;
                    if (!(0 & (1u << 3))) _tmem_load_1[3] = -SM110_XQA_INF;
                    if (!(0 & (1u << 4))) _tmem_load_1[4] = -SM110_XQA_INF;
                    if (!(0 & (1u << 5))) _tmem_load_1[5] = -SM110_XQA_INF;
                    if (!(0 & (1u << 6))) _tmem_load_1[6] = -SM110_XQA_INF;
                    if (!(0 & (1u << 7))) _tmem_load_1[7] = -SM110_XQA_INF;
                    if (!(0 & (1u << 8))) _tmem_load_1[8] = -SM110_XQA_INF;
                    if (!(0 & (1u << 9))) _tmem_load_1[9] = -SM110_XQA_INF;
                    if (!(0 & (1u << 10))) _tmem_load_1[10] = -SM110_XQA_INF;
                    if (!(0 & (1u << 11))) _tmem_load_1[11] = -SM110_XQA_INF;
                    if (!(0 & (1u << 12))) _tmem_load_1[12] = -SM110_XQA_INF;
                    if (!(0 & (1u << 13))) _tmem_load_1[13] = -SM110_XQA_INF;
                    if (!(0 & (1u << 14))) _tmem_load_1[14] = -SM110_XQA_INF;
                    if (!(0 & (1u << 15))) _tmem_load_1[15] = -SM110_XQA_INF;
                    if (!(0 & (1u << 16))) _tmem_load_1[16] = -SM110_XQA_INF;
                    if (!(0 & (1u << 17))) _tmem_load_1[17] = -SM110_XQA_INF;
                    if (!(0 & (1u << 18))) _tmem_load_1[18] = -SM110_XQA_INF;
                    if (!(0 & (1u << 19))) _tmem_load_1[19] = -SM110_XQA_INF;
                    if (!(0 & (1u << 20))) _tmem_load_1[20] = -SM110_XQA_INF;
                    if (!(0 & (1u << 21))) _tmem_load_1[21] = -SM110_XQA_INF;
                    if (!(0 & (1u << 22))) _tmem_load_1[22] = -SM110_XQA_INF;
                    if (!(0 & (1u << 23))) _tmem_load_1[23] = -SM110_XQA_INF;
                    if (!(0 & (1u << 24))) _tmem_load_1[24] = -SM110_XQA_INF;
                    if (!(0 & (1u << 25))) _tmem_load_1[25] = -SM110_XQA_INF;
                    if (!(0 & (1u << 26))) _tmem_load_1[26] = -SM110_XQA_INF;
                    if (!(0 & (1u << 27))) _tmem_load_1[27] = -SM110_XQA_INF;
                    if (!(0 & (1u << 28))) _tmem_load_1[28] = -SM110_XQA_INF;
                    if (!(0 & (1u << 29))) _tmem_load_1[29] = -SM110_XQA_INF;
                    if (!(0 & (1u << 30))) _tmem_load_1[30] = -SM110_XQA_INF;
                    if (!(0 & (1u << 31))) _tmem_load_1[31] = -SM110_XQA_INF;
                } else {
                    if (length < tile_token0 + 64) {
                        int valid = length - tile_token0;
                        if (valid <= 32) {
                            if (!(0 & (1u << 0))) _tmem_load_1[0] = -SM110_XQA_INF;
                            if (!(0 & (1u << 1))) _tmem_load_1[1] = -SM110_XQA_INF;
                            if (!(0 & (1u << 2))) _tmem_load_1[2] = -SM110_XQA_INF;
                            if (!(0 & (1u << 3))) _tmem_load_1[3] = -SM110_XQA_INF;
                            if (!(0 & (1u << 4))) _tmem_load_1[4] = -SM110_XQA_INF;
                            if (!(0 & (1u << 5))) _tmem_load_1[5] = -SM110_XQA_INF;
                            if (!(0 & (1u << 6))) _tmem_load_1[6] = -SM110_XQA_INF;
                            if (!(0 & (1u << 7))) _tmem_load_1[7] = -SM110_XQA_INF;
                            if (!(0 & (1u << 8))) _tmem_load_1[8] = -SM110_XQA_INF;
                            if (!(0 & (1u << 9))) _tmem_load_1[9] = -SM110_XQA_INF;
                            if (!(0 & (1u << 10))) _tmem_load_1[10] = -SM110_XQA_INF;
                            if (!(0 & (1u << 11))) _tmem_load_1[11] = -SM110_XQA_INF;
                            if (!(0 & (1u << 12))) _tmem_load_1[12] = -SM110_XQA_INF;
                            if (!(0 & (1u << 13))) _tmem_load_1[13] = -SM110_XQA_INF;
                            if (!(0 & (1u << 14))) _tmem_load_1[14] = -SM110_XQA_INF;
                            if (!(0 & (1u << 15))) _tmem_load_1[15] = -SM110_XQA_INF;
                            if (!(0 & (1u << 16))) _tmem_load_1[16] = -SM110_XQA_INF;
                            if (!(0 & (1u << 17))) _tmem_load_1[17] = -SM110_XQA_INF;
                            if (!(0 & (1u << 18))) _tmem_load_1[18] = -SM110_XQA_INF;
                            if (!(0 & (1u << 19))) _tmem_load_1[19] = -SM110_XQA_INF;
                            if (!(0 & (1u << 20))) _tmem_load_1[20] = -SM110_XQA_INF;
                            if (!(0 & (1u << 21))) _tmem_load_1[21] = -SM110_XQA_INF;
                            if (!(0 & (1u << 22))) _tmem_load_1[22] = -SM110_XQA_INF;
                            if (!(0 & (1u << 23))) _tmem_load_1[23] = -SM110_XQA_INF;
                            if (!(0 & (1u << 24))) _tmem_load_1[24] = -SM110_XQA_INF;
                            if (!(0 & (1u << 25))) _tmem_load_1[25] = -SM110_XQA_INF;
                            if (!(0 & (1u << 26))) _tmem_load_1[26] = -SM110_XQA_INF;
                            if (!(0 & (1u << 27))) _tmem_load_1[27] = -SM110_XQA_INF;
                            if (!(0 & (1u << 28))) _tmem_load_1[28] = -SM110_XQA_INF;
                            if (!(0 & (1u << 29))) _tmem_load_1[29] = -SM110_XQA_INF;
                            if (!(0 & (1u << 30))) _tmem_load_1[30] = -SM110_XQA_INF;
                            if (!(0 & (1u << 31))) _tmem_load_1[31] = -SM110_XQA_INF;
                            if (valid <= 0) {
                                if (!(0 & (1u << 0))) _tmem_load_0[0] = -SM110_XQA_INF;
                                if (!(0 & (1u << 1))) _tmem_load_0[1] = -SM110_XQA_INF;
                                if (!(0 & (1u << 2))) _tmem_load_0[2] = -SM110_XQA_INF;
                                if (!(0 & (1u << 3))) _tmem_load_0[3] = -SM110_XQA_INF;
                                if (!(0 & (1u << 4))) _tmem_load_0[4] = -SM110_XQA_INF;
                                if (!(0 & (1u << 5))) _tmem_load_0[5] = -SM110_XQA_INF;
                                if (!(0 & (1u << 6))) _tmem_load_0[6] = -SM110_XQA_INF;
                                if (!(0 & (1u << 7))) _tmem_load_0[7] = -SM110_XQA_INF;
                                if (!(0 & (1u << 8))) _tmem_load_0[8] = -SM110_XQA_INF;
                                if (!(0 & (1u << 9))) _tmem_load_0[9] = -SM110_XQA_INF;
                                if (!(0 & (1u << 10))) _tmem_load_0[10] = -SM110_XQA_INF;
                                if (!(0 & (1u << 11))) _tmem_load_0[11] = -SM110_XQA_INF;
                                if (!(0 & (1u << 12))) _tmem_load_0[12] = -SM110_XQA_INF;
                                if (!(0 & (1u << 13))) _tmem_load_0[13] = -SM110_XQA_INF;
                                if (!(0 & (1u << 14))) _tmem_load_0[14] = -SM110_XQA_INF;
                                if (!(0 & (1u << 15))) _tmem_load_0[15] = -SM110_XQA_INF;
                                if (!(0 & (1u << 16))) _tmem_load_0[16] = -SM110_XQA_INF;
                                if (!(0 & (1u << 17))) _tmem_load_0[17] = -SM110_XQA_INF;
                                if (!(0 & (1u << 18))) _tmem_load_0[18] = -SM110_XQA_INF;
                                if (!(0 & (1u << 19))) _tmem_load_0[19] = -SM110_XQA_INF;
                                if (!(0 & (1u << 20))) _tmem_load_0[20] = -SM110_XQA_INF;
                                if (!(0 & (1u << 21))) _tmem_load_0[21] = -SM110_XQA_INF;
                                if (!(0 & (1u << 22))) _tmem_load_0[22] = -SM110_XQA_INF;
                                if (!(0 & (1u << 23))) _tmem_load_0[23] = -SM110_XQA_INF;
                                if (!(0 & (1u << 24))) _tmem_load_0[24] = -SM110_XQA_INF;
                                if (!(0 & (1u << 25))) _tmem_load_0[25] = -SM110_XQA_INF;
                                if (!(0 & (1u << 26))) _tmem_load_0[26] = -SM110_XQA_INF;
                                if (!(0 & (1u << 27))) _tmem_load_0[27] = -SM110_XQA_INF;
                                if (!(0 & (1u << 28))) _tmem_load_0[28] = -SM110_XQA_INF;
                                if (!(0 & (1u << 29))) _tmem_load_0[29] = -SM110_XQA_INF;
                                if (!(0 & (1u << 30))) _tmem_load_0[30] = -SM110_XQA_INF;
                                if (!(0 & (1u << 31))) _tmem_load_0[31] = -SM110_XQA_INF;
                            } else {
                                uint32_t _slice_lo_mask_0;
                                {
                                    int _lim_0 = valid;
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
                                if (!(_slice_lo_mask_0 & (1u << 0))) _tmem_load_0[0] = -SM110_XQA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 1))) _tmem_load_0[1] = -SM110_XQA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 2))) _tmem_load_0[2] = -SM110_XQA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 3))) _tmem_load_0[3] = -SM110_XQA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 4))) _tmem_load_0[4] = -SM110_XQA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 5))) _tmem_load_0[5] = -SM110_XQA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 6))) _tmem_load_0[6] = -SM110_XQA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 7))) _tmem_load_0[7] = -SM110_XQA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 8))) _tmem_load_0[8] = -SM110_XQA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 9))) _tmem_load_0[9] = -SM110_XQA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 10))) _tmem_load_0[10] = -SM110_XQA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 11))) _tmem_load_0[11] = -SM110_XQA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 12))) _tmem_load_0[12] = -SM110_XQA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 13))) _tmem_load_0[13] = -SM110_XQA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 14))) _tmem_load_0[14] = -SM110_XQA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 15))) _tmem_load_0[15] = -SM110_XQA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 16))) _tmem_load_0[16] = -SM110_XQA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 17))) _tmem_load_0[17] = -SM110_XQA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 18))) _tmem_load_0[18] = -SM110_XQA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 19))) _tmem_load_0[19] = -SM110_XQA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 20))) _tmem_load_0[20] = -SM110_XQA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 21))) _tmem_load_0[21] = -SM110_XQA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 22))) _tmem_load_0[22] = -SM110_XQA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 23))) _tmem_load_0[23] = -SM110_XQA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 24))) _tmem_load_0[24] = -SM110_XQA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 25))) _tmem_load_0[25] = -SM110_XQA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 26))) _tmem_load_0[26] = -SM110_XQA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 27))) _tmem_load_0[27] = -SM110_XQA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 28))) _tmem_load_0[28] = -SM110_XQA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 29))) _tmem_load_0[29] = -SM110_XQA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 30))) _tmem_load_0[30] = -SM110_XQA_INF;
                                if (!(_slice_lo_mask_0 & (1u << 31))) _tmem_load_0[31] = -SM110_XQA_INF;
                            }
                        } else {
                            uint32_t _slice_lo_mask_1;
                            {
                                int _lim_1 = valid - 32;
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
                            if (!(_slice_lo_mask_1 & (1u << 0))) _tmem_load_1[0] = -SM110_XQA_INF;
                            if (!(_slice_lo_mask_1 & (1u << 1))) _tmem_load_1[1] = -SM110_XQA_INF;
                            if (!(_slice_lo_mask_1 & (1u << 2))) _tmem_load_1[2] = -SM110_XQA_INF;
                            if (!(_slice_lo_mask_1 & (1u << 3))) _tmem_load_1[3] = -SM110_XQA_INF;
                            if (!(_slice_lo_mask_1 & (1u << 4))) _tmem_load_1[4] = -SM110_XQA_INF;
                            if (!(_slice_lo_mask_1 & (1u << 5))) _tmem_load_1[5] = -SM110_XQA_INF;
                            if (!(_slice_lo_mask_1 & (1u << 6))) _tmem_load_1[6] = -SM110_XQA_INF;
                            if (!(_slice_lo_mask_1 & (1u << 7))) _tmem_load_1[7] = -SM110_XQA_INF;
                            if (!(_slice_lo_mask_1 & (1u << 8))) _tmem_load_1[8] = -SM110_XQA_INF;
                            if (!(_slice_lo_mask_1 & (1u << 9))) _tmem_load_1[9] = -SM110_XQA_INF;
                            if (!(_slice_lo_mask_1 & (1u << 10))) _tmem_load_1[10] = -SM110_XQA_INF;
                            if (!(_slice_lo_mask_1 & (1u << 11))) _tmem_load_1[11] = -SM110_XQA_INF;
                            if (!(_slice_lo_mask_1 & (1u << 12))) _tmem_load_1[12] = -SM110_XQA_INF;
                            if (!(_slice_lo_mask_1 & (1u << 13))) _tmem_load_1[13] = -SM110_XQA_INF;
                            if (!(_slice_lo_mask_1 & (1u << 14))) _tmem_load_1[14] = -SM110_XQA_INF;
                            if (!(_slice_lo_mask_1 & (1u << 15))) _tmem_load_1[15] = -SM110_XQA_INF;
                            if (!(_slice_lo_mask_1 & (1u << 16))) _tmem_load_1[16] = -SM110_XQA_INF;
                            if (!(_slice_lo_mask_1 & (1u << 17))) _tmem_load_1[17] = -SM110_XQA_INF;
                            if (!(_slice_lo_mask_1 & (1u << 18))) _tmem_load_1[18] = -SM110_XQA_INF;
                            if (!(_slice_lo_mask_1 & (1u << 19))) _tmem_load_1[19] = -SM110_XQA_INF;
                            if (!(_slice_lo_mask_1 & (1u << 20))) _tmem_load_1[20] = -SM110_XQA_INF;
                            if (!(_slice_lo_mask_1 & (1u << 21))) _tmem_load_1[21] = -SM110_XQA_INF;
                            if (!(_slice_lo_mask_1 & (1u << 22))) _tmem_load_1[22] = -SM110_XQA_INF;
                            if (!(_slice_lo_mask_1 & (1u << 23))) _tmem_load_1[23] = -SM110_XQA_INF;
                            if (!(_slice_lo_mask_1 & (1u << 24))) _tmem_load_1[24] = -SM110_XQA_INF;
                            if (!(_slice_lo_mask_1 & (1u << 25))) _tmem_load_1[25] = -SM110_XQA_INF;
                            if (!(_slice_lo_mask_1 & (1u << 26))) _tmem_load_1[26] = -SM110_XQA_INF;
                            if (!(_slice_lo_mask_1 & (1u << 27))) _tmem_load_1[27] = -SM110_XQA_INF;
                            if (!(_slice_lo_mask_1 & (1u << 28))) _tmem_load_1[28] = -SM110_XQA_INF;
                            if (!(_slice_lo_mask_1 & (1u << 29))) _tmem_load_1[29] = -SM110_XQA_INF;
                            if (!(_slice_lo_mask_1 & (1u << 30))) _tmem_load_1[30] = -SM110_XQA_INF;
                            if (!(_slice_lo_mask_1 & (1u << 31))) _tmem_load_1[31] = -SM110_XQA_INF;
                        }
                    }
                    if (prefix < tile_token0 + 64) {
                        int tree0 = tile_token0 - prefix;
                        if (mask_stride <= 4) {
                            unsigned int vis_lo = 4294967295;
                            unsigned int vis_hi = 4294967295;
                            if (tree0 >= 0) {
                                unsigned int widx = (unsigned int)tree0 >> 5;
                                unsigned int shift = (unsigned int)tree0 & 31;
                                unsigned int pj0 = ((widx == 0) ? mw0 : ((widx == 1) ? mw1 : ((widx == 2) ? mw2 : ((widx == 3) ? mw3 : zero32))));
                                unsigned int pj1 = ((widx + 1 == 0) ? mw0 : ((widx + 1 == 1) ? mw1 : ((widx + 1 == 2) ? mw2 : ((widx + 1 == 3) ? mw3 : zero32))));
                                unsigned int pj2 = ((widx + 2 == 0) ? mw0 : ((widx + 2 == 1) ? mw1 : ((widx + 2 == 2) ? mw2 : ((widx + 2 == 3) ? mw3 : zero32))));
                                vis_lo = pj0 >> shift | pj1 << 31 - shift << 1;
                                vis_hi = pj1 >> shift | pj2 << 31 - shift << 1;
                            } else {
                                unsigned int lead = (unsigned int)(-tree0);
                                if (lead < 32) {
                                    vis_lo = mw0 << lead | ((unsigned int)1 << lead) - 1;
                                    vis_hi = mw1 << lead | mw0 >> 32 - lead;
                                } else {
                                    unsigned int lead32 = lead - 32;
                                    vis_hi = mw0 << lead32 | ((unsigned int)1 << lead32) - 1;
                                }
                            }
                            _tmem_load_0[0] = (((vis_lo & 1) != 0) ? _tmem_load_0[0] : -SM110_XQA_INF);
                            _tmem_load_1[0] = (((vis_hi & 1) != 0) ? _tmem_load_1[0] : -SM110_XQA_INF);
                            _tmem_load_0[1] = (((vis_lo >> 1 & 1) != 0) ? _tmem_load_0[1] : -SM110_XQA_INF);
                            _tmem_load_1[1] = (((vis_hi >> 1 & 1) != 0) ? _tmem_load_1[1] : -SM110_XQA_INF);
                            _tmem_load_0[2] = (((vis_lo >> 2 & 1) != 0) ? _tmem_load_0[2] : -SM110_XQA_INF);
                            _tmem_load_1[2] = (((vis_hi >> 2 & 1) != 0) ? _tmem_load_1[2] : -SM110_XQA_INF);
                            _tmem_load_0[3] = (((vis_lo >> 3 & 1) != 0) ? _tmem_load_0[3] : -SM110_XQA_INF);
                            _tmem_load_1[3] = (((vis_hi >> 3 & 1) != 0) ? _tmem_load_1[3] : -SM110_XQA_INF);
                            _tmem_load_0[4] = (((vis_lo >> 4 & 1) != 0) ? _tmem_load_0[4] : -SM110_XQA_INF);
                            _tmem_load_1[4] = (((vis_hi >> 4 & 1) != 0) ? _tmem_load_1[4] : -SM110_XQA_INF);
                            _tmem_load_0[5] = (((vis_lo >> 5 & 1) != 0) ? _tmem_load_0[5] : -SM110_XQA_INF);
                            _tmem_load_1[5] = (((vis_hi >> 5 & 1) != 0) ? _tmem_load_1[5] : -SM110_XQA_INF);
                            _tmem_load_0[6] = (((vis_lo >> 6 & 1) != 0) ? _tmem_load_0[6] : -SM110_XQA_INF);
                            _tmem_load_1[6] = (((vis_hi >> 6 & 1) != 0) ? _tmem_load_1[6] : -SM110_XQA_INF);
                            _tmem_load_0[7] = (((vis_lo >> 7 & 1) != 0) ? _tmem_load_0[7] : -SM110_XQA_INF);
                            _tmem_load_1[7] = (((vis_hi >> 7 & 1) != 0) ? _tmem_load_1[7] : -SM110_XQA_INF);
                            _tmem_load_0[8] = (((vis_lo >> 8 & 1) != 0) ? _tmem_load_0[8] : -SM110_XQA_INF);
                            _tmem_load_1[8] = (((vis_hi >> 8 & 1) != 0) ? _tmem_load_1[8] : -SM110_XQA_INF);
                            _tmem_load_0[9] = (((vis_lo >> 9 & 1) != 0) ? _tmem_load_0[9] : -SM110_XQA_INF);
                            _tmem_load_1[9] = (((vis_hi >> 9 & 1) != 0) ? _tmem_load_1[9] : -SM110_XQA_INF);
                            _tmem_load_0[10] = (((vis_lo >> 10 & 1) != 0) ? _tmem_load_0[10] : -SM110_XQA_INF);
                            _tmem_load_1[10] = (((vis_hi >> 10 & 1) != 0) ? _tmem_load_1[10] : -SM110_XQA_INF);
                            _tmem_load_0[11] = (((vis_lo >> 11 & 1) != 0) ? _tmem_load_0[11] : -SM110_XQA_INF);
                            _tmem_load_1[11] = (((vis_hi >> 11 & 1) != 0) ? _tmem_load_1[11] : -SM110_XQA_INF);
                            _tmem_load_0[12] = (((vis_lo >> 12 & 1) != 0) ? _tmem_load_0[12] : -SM110_XQA_INF);
                            _tmem_load_1[12] = (((vis_hi >> 12 & 1) != 0) ? _tmem_load_1[12] : -SM110_XQA_INF);
                            _tmem_load_0[13] = (((vis_lo >> 13 & 1) != 0) ? _tmem_load_0[13] : -SM110_XQA_INF);
                            _tmem_load_1[13] = (((vis_hi >> 13 & 1) != 0) ? _tmem_load_1[13] : -SM110_XQA_INF);
                            _tmem_load_0[14] = (((vis_lo >> 14 & 1) != 0) ? _tmem_load_0[14] : -SM110_XQA_INF);
                            _tmem_load_1[14] = (((vis_hi >> 14 & 1) != 0) ? _tmem_load_1[14] : -SM110_XQA_INF);
                            _tmem_load_0[15] = (((vis_lo >> 15 & 1) != 0) ? _tmem_load_0[15] : -SM110_XQA_INF);
                            _tmem_load_1[15] = (((vis_hi >> 15 & 1) != 0) ? _tmem_load_1[15] : -SM110_XQA_INF);
                            _tmem_load_0[16] = (((vis_lo >> 16 & 1) != 0) ? _tmem_load_0[16] : -SM110_XQA_INF);
                            _tmem_load_1[16] = (((vis_hi >> 16 & 1) != 0) ? _tmem_load_1[16] : -SM110_XQA_INF);
                            _tmem_load_0[17] = (((vis_lo >> 17 & 1) != 0) ? _tmem_load_0[17] : -SM110_XQA_INF);
                            _tmem_load_1[17] = (((vis_hi >> 17 & 1) != 0) ? _tmem_load_1[17] : -SM110_XQA_INF);
                            _tmem_load_0[18] = (((vis_lo >> 18 & 1) != 0) ? _tmem_load_0[18] : -SM110_XQA_INF);
                            _tmem_load_1[18] = (((vis_hi >> 18 & 1) != 0) ? _tmem_load_1[18] : -SM110_XQA_INF);
                            _tmem_load_0[19] = (((vis_lo >> 19 & 1) != 0) ? _tmem_load_0[19] : -SM110_XQA_INF);
                            _tmem_load_1[19] = (((vis_hi >> 19 & 1) != 0) ? _tmem_load_1[19] : -SM110_XQA_INF);
                            _tmem_load_0[20] = (((vis_lo >> 20 & 1) != 0) ? _tmem_load_0[20] : -SM110_XQA_INF);
                            _tmem_load_1[20] = (((vis_hi >> 20 & 1) != 0) ? _tmem_load_1[20] : -SM110_XQA_INF);
                            _tmem_load_0[21] = (((vis_lo >> 21 & 1) != 0) ? _tmem_load_0[21] : -SM110_XQA_INF);
                            _tmem_load_1[21] = (((vis_hi >> 21 & 1) != 0) ? _tmem_load_1[21] : -SM110_XQA_INF);
                            _tmem_load_0[22] = (((vis_lo >> 22 & 1) != 0) ? _tmem_load_0[22] : -SM110_XQA_INF);
                            _tmem_load_1[22] = (((vis_hi >> 22 & 1) != 0) ? _tmem_load_1[22] : -SM110_XQA_INF);
                            _tmem_load_0[23] = (((vis_lo >> 23 & 1) != 0) ? _tmem_load_0[23] : -SM110_XQA_INF);
                            _tmem_load_1[23] = (((vis_hi >> 23 & 1) != 0) ? _tmem_load_1[23] : -SM110_XQA_INF);
                            _tmem_load_0[24] = (((vis_lo >> 24 & 1) != 0) ? _tmem_load_0[24] : -SM110_XQA_INF);
                            _tmem_load_1[24] = (((vis_hi >> 24 & 1) != 0) ? _tmem_load_1[24] : -SM110_XQA_INF);
                            _tmem_load_0[25] = (((vis_lo >> 25 & 1) != 0) ? _tmem_load_0[25] : -SM110_XQA_INF);
                            _tmem_load_1[25] = (((vis_hi >> 25 & 1) != 0) ? _tmem_load_1[25] : -SM110_XQA_INF);
                            _tmem_load_0[26] = (((vis_lo >> 26 & 1) != 0) ? _tmem_load_0[26] : -SM110_XQA_INF);
                            _tmem_load_1[26] = (((vis_hi >> 26 & 1) != 0) ? _tmem_load_1[26] : -SM110_XQA_INF);
                            _tmem_load_0[27] = (((vis_lo >> 27 & 1) != 0) ? _tmem_load_0[27] : -SM110_XQA_INF);
                            _tmem_load_1[27] = (((vis_hi >> 27 & 1) != 0) ? _tmem_load_1[27] : -SM110_XQA_INF);
                            _tmem_load_0[28] = (((vis_lo >> 28 & 1) != 0) ? _tmem_load_0[28] : -SM110_XQA_INF);
                            _tmem_load_1[28] = (((vis_hi >> 28 & 1) != 0) ? _tmem_load_1[28] : -SM110_XQA_INF);
                            _tmem_load_0[29] = (((vis_lo >> 29 & 1) != 0) ? _tmem_load_0[29] : -SM110_XQA_INF);
                            _tmem_load_1[29] = (((vis_hi >> 29 & 1) != 0) ? _tmem_load_1[29] : -SM110_XQA_INF);
                            _tmem_load_0[30] = (((vis_lo >> 30 & 1) != 0) ? _tmem_load_0[30] : -SM110_XQA_INF);
                            _tmem_load_1[30] = (((vis_hi >> 30 & 1) != 0) ? _tmem_load_1[30] : -SM110_XQA_INF);
                            _tmem_load_0[31] = (((vis_lo >> 31 & 1) != 0) ? _tmem_load_0[31] : -SM110_XQA_INF);
                            _tmem_load_1[31] = (((vis_hi >> 31 & 1) != 0) ? _tmem_load_1[31] : -SM110_XQA_INF);
                        } else {
                            int _max_0 = ((tree0) > (0) ? (tree0) : (0));
                            unsigned int word0 = _max_0 >> 5;
                            unsigned int _vec_load_8[1];
                            {
                                _vec_load_8[0] = *reinterpret_cast<const unsigned int*>(mask + (mask_row + word0));
                            }
                            unsigned int w0 = _vec_load_8[0];
                            unsigned int w1 = 0;
                            unsigned int w2 = 0;
                            if (mask_stride > word0 + 1) {
                                unsigned int _vec_load_9[1];
                                {
                                    _vec_load_9[0] = *reinterpret_cast<const unsigned int*>(mask + (mask_row + word0 + 1));
                                }
                                w1 = _vec_load_9[0];
                            }
                            if (mask_stride > word0 + 2) {
                                unsigned int _vec_load_10[1];
                                {
                                    _vec_load_10[0] = *reinterpret_cast<const unsigned int*>(mask + (mask_row + word0 + 2));
                                }
                                w2 = _vec_load_10[0];
                            }
                            int _max_1 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[0] = ((tree0 < 0 || ((((tree0 >> 5) - (_max_1 >> 5) == 0) ? w0 : (((tree0 >> 5) - (_max_1 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 & 31) & 1) != 0) ? _tmem_load_0[0] : -SM110_XQA_INF);
                            int _max_2 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[1] = ((tree0 + 1 < 0 || ((((tree0 + 1 >> 5) - (_max_2 >> 5) == 0) ? w0 : (((tree0 + 1 >> 5) - (_max_2 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 1 & 31) & 1) != 0) ? _tmem_load_0[1] : -SM110_XQA_INF);
                            int _max_3 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[2] = ((tree0 + 2 < 0 || ((((tree0 + 2 >> 5) - (_max_3 >> 5) == 0) ? w0 : (((tree0 + 2 >> 5) - (_max_3 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 2 & 31) & 1) != 0) ? _tmem_load_0[2] : -SM110_XQA_INF);
                            int _max_4 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[3] = ((tree0 + 3 < 0 || ((((tree0 + 3 >> 5) - (_max_4 >> 5) == 0) ? w0 : (((tree0 + 3 >> 5) - (_max_4 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 3 & 31) & 1) != 0) ? _tmem_load_0[3] : -SM110_XQA_INF);
                            int _max_5 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[4] = ((tree0 + 4 < 0 || ((((tree0 + 4 >> 5) - (_max_5 >> 5) == 0) ? w0 : (((tree0 + 4 >> 5) - (_max_5 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 4 & 31) & 1) != 0) ? _tmem_load_0[4] : -SM110_XQA_INF);
                            int _max_6 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[5] = ((tree0 + 5 < 0 || ((((tree0 + 5 >> 5) - (_max_6 >> 5) == 0) ? w0 : (((tree0 + 5 >> 5) - (_max_6 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 5 & 31) & 1) != 0) ? _tmem_load_0[5] : -SM110_XQA_INF);
                            int _max_7 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[6] = ((tree0 + 6 < 0 || ((((tree0 + 6 >> 5) - (_max_7 >> 5) == 0) ? w0 : (((tree0 + 6 >> 5) - (_max_7 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 6 & 31) & 1) != 0) ? _tmem_load_0[6] : -SM110_XQA_INF);
                            int _max_8 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[7] = ((tree0 + 7 < 0 || ((((tree0 + 7 >> 5) - (_max_8 >> 5) == 0) ? w0 : (((tree0 + 7 >> 5) - (_max_8 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 7 & 31) & 1) != 0) ? _tmem_load_0[7] : -SM110_XQA_INF);
                            int _max_9 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[8] = ((tree0 + 8 < 0 || ((((tree0 + 8 >> 5) - (_max_9 >> 5) == 0) ? w0 : (((tree0 + 8 >> 5) - (_max_9 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 8 & 31) & 1) != 0) ? _tmem_load_0[8] : -SM110_XQA_INF);
                            int _max_10 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[9] = ((tree0 + 9 < 0 || ((((tree0 + 9 >> 5) - (_max_10 >> 5) == 0) ? w0 : (((tree0 + 9 >> 5) - (_max_10 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 9 & 31) & 1) != 0) ? _tmem_load_0[9] : -SM110_XQA_INF);
                            int _max_11 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[10] = ((tree0 + 10 < 0 || ((((tree0 + 10 >> 5) - (_max_11 >> 5) == 0) ? w0 : (((tree0 + 10 >> 5) - (_max_11 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 10 & 31) & 1) != 0) ? _tmem_load_0[10] : -SM110_XQA_INF);
                            int _max_12 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[11] = ((tree0 + 11 < 0 || ((((tree0 + 11 >> 5) - (_max_12 >> 5) == 0) ? w0 : (((tree0 + 11 >> 5) - (_max_12 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 11 & 31) & 1) != 0) ? _tmem_load_0[11] : -SM110_XQA_INF);
                            int _max_13 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[12] = ((tree0 + 12 < 0 || ((((tree0 + 12 >> 5) - (_max_13 >> 5) == 0) ? w0 : (((tree0 + 12 >> 5) - (_max_13 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 12 & 31) & 1) != 0) ? _tmem_load_0[12] : -SM110_XQA_INF);
                            int _max_14 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[13] = ((tree0 + 13 < 0 || ((((tree0 + 13 >> 5) - (_max_14 >> 5) == 0) ? w0 : (((tree0 + 13 >> 5) - (_max_14 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 13 & 31) & 1) != 0) ? _tmem_load_0[13] : -SM110_XQA_INF);
                            int _max_15 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[14] = ((tree0 + 14 < 0 || ((((tree0 + 14 >> 5) - (_max_15 >> 5) == 0) ? w0 : (((tree0 + 14 >> 5) - (_max_15 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 14 & 31) & 1) != 0) ? _tmem_load_0[14] : -SM110_XQA_INF);
                            int _max_16 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[15] = ((tree0 + 15 < 0 || ((((tree0 + 15 >> 5) - (_max_16 >> 5) == 0) ? w0 : (((tree0 + 15 >> 5) - (_max_16 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 15 & 31) & 1) != 0) ? _tmem_load_0[15] : -SM110_XQA_INF);
                            int _max_17 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[16] = ((tree0 + 16 < 0 || ((((tree0 + 16 >> 5) - (_max_17 >> 5) == 0) ? w0 : (((tree0 + 16 >> 5) - (_max_17 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 16 & 31) & 1) != 0) ? _tmem_load_0[16] : -SM110_XQA_INF);
                            int _max_18 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[17] = ((tree0 + 17 < 0 || ((((tree0 + 17 >> 5) - (_max_18 >> 5) == 0) ? w0 : (((tree0 + 17 >> 5) - (_max_18 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 17 & 31) & 1) != 0) ? _tmem_load_0[17] : -SM110_XQA_INF);
                            int _max_19 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[18] = ((tree0 + 18 < 0 || ((((tree0 + 18 >> 5) - (_max_19 >> 5) == 0) ? w0 : (((tree0 + 18 >> 5) - (_max_19 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 18 & 31) & 1) != 0) ? _tmem_load_0[18] : -SM110_XQA_INF);
                            int _max_20 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[19] = ((tree0 + 19 < 0 || ((((tree0 + 19 >> 5) - (_max_20 >> 5) == 0) ? w0 : (((tree0 + 19 >> 5) - (_max_20 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 19 & 31) & 1) != 0) ? _tmem_load_0[19] : -SM110_XQA_INF);
                            int _max_21 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[20] = ((tree0 + 20 < 0 || ((((tree0 + 20 >> 5) - (_max_21 >> 5) == 0) ? w0 : (((tree0 + 20 >> 5) - (_max_21 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 20 & 31) & 1) != 0) ? _tmem_load_0[20] : -SM110_XQA_INF);
                            int _max_22 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[21] = ((tree0 + 21 < 0 || ((((tree0 + 21 >> 5) - (_max_22 >> 5) == 0) ? w0 : (((tree0 + 21 >> 5) - (_max_22 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 21 & 31) & 1) != 0) ? _tmem_load_0[21] : -SM110_XQA_INF);
                            int _max_23 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[22] = ((tree0 + 22 < 0 || ((((tree0 + 22 >> 5) - (_max_23 >> 5) == 0) ? w0 : (((tree0 + 22 >> 5) - (_max_23 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 22 & 31) & 1) != 0) ? _tmem_load_0[22] : -SM110_XQA_INF);
                            int _max_24 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[23] = ((tree0 + 23 < 0 || ((((tree0 + 23 >> 5) - (_max_24 >> 5) == 0) ? w0 : (((tree0 + 23 >> 5) - (_max_24 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 23 & 31) & 1) != 0) ? _tmem_load_0[23] : -SM110_XQA_INF);
                            int _max_25 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[24] = ((tree0 + 24 < 0 || ((((tree0 + 24 >> 5) - (_max_25 >> 5) == 0) ? w0 : (((tree0 + 24 >> 5) - (_max_25 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 24 & 31) & 1) != 0) ? _tmem_load_0[24] : -SM110_XQA_INF);
                            int _max_26 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[25] = ((tree0 + 25 < 0 || ((((tree0 + 25 >> 5) - (_max_26 >> 5) == 0) ? w0 : (((tree0 + 25 >> 5) - (_max_26 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 25 & 31) & 1) != 0) ? _tmem_load_0[25] : -SM110_XQA_INF);
                            int _max_27 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[26] = ((tree0 + 26 < 0 || ((((tree0 + 26 >> 5) - (_max_27 >> 5) == 0) ? w0 : (((tree0 + 26 >> 5) - (_max_27 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 26 & 31) & 1) != 0) ? _tmem_load_0[26] : -SM110_XQA_INF);
                            int _max_28 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[27] = ((tree0 + 27 < 0 || ((((tree0 + 27 >> 5) - (_max_28 >> 5) == 0) ? w0 : (((tree0 + 27 >> 5) - (_max_28 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 27 & 31) & 1) != 0) ? _tmem_load_0[27] : -SM110_XQA_INF);
                            int _max_29 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[28] = ((tree0 + 28 < 0 || ((((tree0 + 28 >> 5) - (_max_29 >> 5) == 0) ? w0 : (((tree0 + 28 >> 5) - (_max_29 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 28 & 31) & 1) != 0) ? _tmem_load_0[28] : -SM110_XQA_INF);
                            int _max_30 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[29] = ((tree0 + 29 < 0 || ((((tree0 + 29 >> 5) - (_max_30 >> 5) == 0) ? w0 : (((tree0 + 29 >> 5) - (_max_30 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 29 & 31) & 1) != 0) ? _tmem_load_0[29] : -SM110_XQA_INF);
                            int _max_31 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[30] = ((tree0 + 30 < 0 || ((((tree0 + 30 >> 5) - (_max_31 >> 5) == 0) ? w0 : (((tree0 + 30 >> 5) - (_max_31 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 30 & 31) & 1) != 0) ? _tmem_load_0[30] : -SM110_XQA_INF);
                            int _max_32 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_0[31] = ((tree0 + 31 < 0 || ((((tree0 + 31 >> 5) - (_max_32 >> 5) == 0) ? w0 : (((tree0 + 31 >> 5) - (_max_32 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 31 & 31) & 1) != 0) ? _tmem_load_0[31] : -SM110_XQA_INF);
                            int _max_33 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_1[0] = ((tree0 + 32 < 0 || ((((tree0 + 32 >> 5) - (_max_33 >> 5) == 0) ? w0 : (((tree0 + 32 >> 5) - (_max_33 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 32 & 31) & 1) != 0) ? _tmem_load_1[0] : -SM110_XQA_INF);
                            int _max_34 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_1[1] = ((tree0 + 33 < 0 || ((((tree0 + 33 >> 5) - (_max_34 >> 5) == 0) ? w0 : (((tree0 + 33 >> 5) - (_max_34 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 33 & 31) & 1) != 0) ? _tmem_load_1[1] : -SM110_XQA_INF);
                            int _max_35 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_1[2] = ((tree0 + 34 < 0 || ((((tree0 + 34 >> 5) - (_max_35 >> 5) == 0) ? w0 : (((tree0 + 34 >> 5) - (_max_35 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 34 & 31) & 1) != 0) ? _tmem_load_1[2] : -SM110_XQA_INF);
                            int _max_36 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_1[3] = ((tree0 + 35 < 0 || ((((tree0 + 35 >> 5) - (_max_36 >> 5) == 0) ? w0 : (((tree0 + 35 >> 5) - (_max_36 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 35 & 31) & 1) != 0) ? _tmem_load_1[3] : -SM110_XQA_INF);
                            int _max_37 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_1[4] = ((tree0 + 36 < 0 || ((((tree0 + 36 >> 5) - (_max_37 >> 5) == 0) ? w0 : (((tree0 + 36 >> 5) - (_max_37 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 36 & 31) & 1) != 0) ? _tmem_load_1[4] : -SM110_XQA_INF);
                            int _max_38 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_1[5] = ((tree0 + 37 < 0 || ((((tree0 + 37 >> 5) - (_max_38 >> 5) == 0) ? w0 : (((tree0 + 37 >> 5) - (_max_38 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 37 & 31) & 1) != 0) ? _tmem_load_1[5] : -SM110_XQA_INF);
                            int _max_39 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_1[6] = ((tree0 + 38 < 0 || ((((tree0 + 38 >> 5) - (_max_39 >> 5) == 0) ? w0 : (((tree0 + 38 >> 5) - (_max_39 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 38 & 31) & 1) != 0) ? _tmem_load_1[6] : -SM110_XQA_INF);
                            int _max_40 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_1[7] = ((tree0 + 39 < 0 || ((((tree0 + 39 >> 5) - (_max_40 >> 5) == 0) ? w0 : (((tree0 + 39 >> 5) - (_max_40 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 39 & 31) & 1) != 0) ? _tmem_load_1[7] : -SM110_XQA_INF);
                            int _max_41 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_1[8] = ((tree0 + 40 < 0 || ((((tree0 + 40 >> 5) - (_max_41 >> 5) == 0) ? w0 : (((tree0 + 40 >> 5) - (_max_41 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 40 & 31) & 1) != 0) ? _tmem_load_1[8] : -SM110_XQA_INF);
                            int _max_42 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_1[9] = ((tree0 + 41 < 0 || ((((tree0 + 41 >> 5) - (_max_42 >> 5) == 0) ? w0 : (((tree0 + 41 >> 5) - (_max_42 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 41 & 31) & 1) != 0) ? _tmem_load_1[9] : -SM110_XQA_INF);
                            int _max_43 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_1[10] = ((tree0 + 42 < 0 || ((((tree0 + 42 >> 5) - (_max_43 >> 5) == 0) ? w0 : (((tree0 + 42 >> 5) - (_max_43 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 42 & 31) & 1) != 0) ? _tmem_load_1[10] : -SM110_XQA_INF);
                            int _max_44 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_1[11] = ((tree0 + 43 < 0 || ((((tree0 + 43 >> 5) - (_max_44 >> 5) == 0) ? w0 : (((tree0 + 43 >> 5) - (_max_44 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 43 & 31) & 1) != 0) ? _tmem_load_1[11] : -SM110_XQA_INF);
                            int _max_45 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_1[12] = ((tree0 + 44 < 0 || ((((tree0 + 44 >> 5) - (_max_45 >> 5) == 0) ? w0 : (((tree0 + 44 >> 5) - (_max_45 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 44 & 31) & 1) != 0) ? _tmem_load_1[12] : -SM110_XQA_INF);
                            int _max_46 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_1[13] = ((tree0 + 45 < 0 || ((((tree0 + 45 >> 5) - (_max_46 >> 5) == 0) ? w0 : (((tree0 + 45 >> 5) - (_max_46 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 45 & 31) & 1) != 0) ? _tmem_load_1[13] : -SM110_XQA_INF);
                            int _max_47 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_1[14] = ((tree0 + 46 < 0 || ((((tree0 + 46 >> 5) - (_max_47 >> 5) == 0) ? w0 : (((tree0 + 46 >> 5) - (_max_47 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 46 & 31) & 1) != 0) ? _tmem_load_1[14] : -SM110_XQA_INF);
                            int _max_48 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_1[15] = ((tree0 + 47 < 0 || ((((tree0 + 47 >> 5) - (_max_48 >> 5) == 0) ? w0 : (((tree0 + 47 >> 5) - (_max_48 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 47 & 31) & 1) != 0) ? _tmem_load_1[15] : -SM110_XQA_INF);
                            int _max_49 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_1[16] = ((tree0 + 48 < 0 || ((((tree0 + 48 >> 5) - (_max_49 >> 5) == 0) ? w0 : (((tree0 + 48 >> 5) - (_max_49 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 48 & 31) & 1) != 0) ? _tmem_load_1[16] : -SM110_XQA_INF);
                            int _max_50 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_1[17] = ((tree0 + 49 < 0 || ((((tree0 + 49 >> 5) - (_max_50 >> 5) == 0) ? w0 : (((tree0 + 49 >> 5) - (_max_50 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 49 & 31) & 1) != 0) ? _tmem_load_1[17] : -SM110_XQA_INF);
                            int _max_51 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_1[18] = ((tree0 + 50 < 0 || ((((tree0 + 50 >> 5) - (_max_51 >> 5) == 0) ? w0 : (((tree0 + 50 >> 5) - (_max_51 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 50 & 31) & 1) != 0) ? _tmem_load_1[18] : -SM110_XQA_INF);
                            int _max_52 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_1[19] = ((tree0 + 51 < 0 || ((((tree0 + 51 >> 5) - (_max_52 >> 5) == 0) ? w0 : (((tree0 + 51 >> 5) - (_max_52 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 51 & 31) & 1) != 0) ? _tmem_load_1[19] : -SM110_XQA_INF);
                            int _max_53 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_1[20] = ((tree0 + 52 < 0 || ((((tree0 + 52 >> 5) - (_max_53 >> 5) == 0) ? w0 : (((tree0 + 52 >> 5) - (_max_53 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 52 & 31) & 1) != 0) ? _tmem_load_1[20] : -SM110_XQA_INF);
                            int _max_54 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_1[21] = ((tree0 + 53 < 0 || ((((tree0 + 53 >> 5) - (_max_54 >> 5) == 0) ? w0 : (((tree0 + 53 >> 5) - (_max_54 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 53 & 31) & 1) != 0) ? _tmem_load_1[21] : -SM110_XQA_INF);
                            int _max_55 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_1[22] = ((tree0 + 54 < 0 || ((((tree0 + 54 >> 5) - (_max_55 >> 5) == 0) ? w0 : (((tree0 + 54 >> 5) - (_max_55 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 54 & 31) & 1) != 0) ? _tmem_load_1[22] : -SM110_XQA_INF);
                            int _max_56 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_1[23] = ((tree0 + 55 < 0 || ((((tree0 + 55 >> 5) - (_max_56 >> 5) == 0) ? w0 : (((tree0 + 55 >> 5) - (_max_56 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 55 & 31) & 1) != 0) ? _tmem_load_1[23] : -SM110_XQA_INF);
                            int _max_57 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_1[24] = ((tree0 + 56 < 0 || ((((tree0 + 56 >> 5) - (_max_57 >> 5) == 0) ? w0 : (((tree0 + 56 >> 5) - (_max_57 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 56 & 31) & 1) != 0) ? _tmem_load_1[24] : -SM110_XQA_INF);
                            int _max_58 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_1[25] = ((tree0 + 57 < 0 || ((((tree0 + 57 >> 5) - (_max_58 >> 5) == 0) ? w0 : (((tree0 + 57 >> 5) - (_max_58 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 57 & 31) & 1) != 0) ? _tmem_load_1[25] : -SM110_XQA_INF);
                            int _max_59 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_1[26] = ((tree0 + 58 < 0 || ((((tree0 + 58 >> 5) - (_max_59 >> 5) == 0) ? w0 : (((tree0 + 58 >> 5) - (_max_59 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 58 & 31) & 1) != 0) ? _tmem_load_1[26] : -SM110_XQA_INF);
                            int _max_60 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_1[27] = ((tree0 + 59 < 0 || ((((tree0 + 59 >> 5) - (_max_60 >> 5) == 0) ? w0 : (((tree0 + 59 >> 5) - (_max_60 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 59 & 31) & 1) != 0) ? _tmem_load_1[27] : -SM110_XQA_INF);
                            int _max_61 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_1[28] = ((tree0 + 60 < 0 || ((((tree0 + 60 >> 5) - (_max_61 >> 5) == 0) ? w0 : (((tree0 + 60 >> 5) - (_max_61 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 60 & 31) & 1) != 0) ? _tmem_load_1[28] : -SM110_XQA_INF);
                            int _max_62 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_1[29] = ((tree0 + 61 < 0 || ((((tree0 + 61 >> 5) - (_max_62 >> 5) == 0) ? w0 : (((tree0 + 61 >> 5) - (_max_62 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 61 & 31) & 1) != 0) ? _tmem_load_1[29] : -SM110_XQA_INF);
                            int _max_63 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_1[30] = ((tree0 + 62 < 0 || ((((tree0 + 62 >> 5) - (_max_63 >> 5) == 0) ? w0 : (((tree0 + 62 >> 5) - (_max_63 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 62 & 31) & 1) != 0) ? _tmem_load_1[30] : -SM110_XQA_INF);
                            int _max_64 = ((tree0) > (0) ? (tree0) : (0));
                            _tmem_load_1[31] = ((tree0 + 63 < 0 || ((((tree0 + 63 >> 5) - (_max_64 >> 5) == 0) ? w0 : (((tree0 + 63 >> 5) - (_max_64 >> 5) == 1) ? w1 : w2)) >> (unsigned int)(tree0 + 63 & 31) & 1) != 0) ? _tmem_load_1[31] : -SM110_XQA_INF);
                        }
                    }
                }
                float2 _reg_reduce_max2_2 = {-SM110_XQA_INF, -SM110_XQA_INF};
                _reg_reduce_max2_2.x = max_noftz(_reg_reduce_max2_2.x, max_noftz(_tmem_load_0[0], _tmem_load_0[1]));
                _reg_reduce_max2_2.y = max_noftz(_reg_reduce_max2_2.y, max_noftz(_tmem_load_0[2], _tmem_load_0[3]));
                _reg_reduce_max2_2.x = max_noftz(_reg_reduce_max2_2.x, max_noftz(_tmem_load_0[4], _tmem_load_0[5]));
                _reg_reduce_max2_2.y = max_noftz(_reg_reduce_max2_2.y, max_noftz(_tmem_load_0[6], _tmem_load_0[7]));
                _reg_reduce_max2_2.x = max_noftz(_reg_reduce_max2_2.x, max_noftz(_tmem_load_0[8], _tmem_load_0[9]));
                _reg_reduce_max2_2.y = max_noftz(_reg_reduce_max2_2.y, max_noftz(_tmem_load_0[10], _tmem_load_0[11]));
                _reg_reduce_max2_2.x = max_noftz(_reg_reduce_max2_2.x, max_noftz(_tmem_load_0[12], _tmem_load_0[13]));
                _reg_reduce_max2_2.y = max_noftz(_reg_reduce_max2_2.y, max_noftz(_tmem_load_0[14], _tmem_load_0[15]));
                _reg_reduce_max2_2.x = max_noftz(_reg_reduce_max2_2.x, max_noftz(_tmem_load_0[16], _tmem_load_0[17]));
                _reg_reduce_max2_2.y = max_noftz(_reg_reduce_max2_2.y, max_noftz(_tmem_load_0[18], _tmem_load_0[19]));
                _reg_reduce_max2_2.x = max_noftz(_reg_reduce_max2_2.x, max_noftz(_tmem_load_0[20], _tmem_load_0[21]));
                _reg_reduce_max2_2.y = max_noftz(_reg_reduce_max2_2.y, max_noftz(_tmem_load_0[22], _tmem_load_0[23]));
                _reg_reduce_max2_2.x = max_noftz(_reg_reduce_max2_2.x, max_noftz(_tmem_load_0[24], _tmem_load_0[25]));
                _reg_reduce_max2_2.y = max_noftz(_reg_reduce_max2_2.y, max_noftz(_tmem_load_0[26], _tmem_load_0[27]));
                _reg_reduce_max2_2.x = max_noftz(_reg_reduce_max2_2.x, max_noftz(_tmem_load_0[28], _tmem_load_0[29]));
                _reg_reduce_max2_2.y = max_noftz(_reg_reduce_max2_2.y, max_noftz(_tmem_load_0[30], _tmem_load_0[31]));
                float _tmem_load_0_max = row_max_reduce(_reg_reduce_max2_2);
                float2 _reg_reduce_max2_3 = {-SM110_XQA_INF, -SM110_XQA_INF};
                _reg_reduce_max2_3.x = max_noftz(_reg_reduce_max2_3.x, max_noftz(_tmem_load_1[0], _tmem_load_1[1]));
                _reg_reduce_max2_3.y = max_noftz(_reg_reduce_max2_3.y, max_noftz(_tmem_load_1[2], _tmem_load_1[3]));
                _reg_reduce_max2_3.x = max_noftz(_reg_reduce_max2_3.x, max_noftz(_tmem_load_1[4], _tmem_load_1[5]));
                _reg_reduce_max2_3.y = max_noftz(_reg_reduce_max2_3.y, max_noftz(_tmem_load_1[6], _tmem_load_1[7]));
                _reg_reduce_max2_3.x = max_noftz(_reg_reduce_max2_3.x, max_noftz(_tmem_load_1[8], _tmem_load_1[9]));
                _reg_reduce_max2_3.y = max_noftz(_reg_reduce_max2_3.y, max_noftz(_tmem_load_1[10], _tmem_load_1[11]));
                _reg_reduce_max2_3.x = max_noftz(_reg_reduce_max2_3.x, max_noftz(_tmem_load_1[12], _tmem_load_1[13]));
                _reg_reduce_max2_3.y = max_noftz(_reg_reduce_max2_3.y, max_noftz(_tmem_load_1[14], _tmem_load_1[15]));
                _reg_reduce_max2_3.x = max_noftz(_reg_reduce_max2_3.x, max_noftz(_tmem_load_1[16], _tmem_load_1[17]));
                _reg_reduce_max2_3.y = max_noftz(_reg_reduce_max2_3.y, max_noftz(_tmem_load_1[18], _tmem_load_1[19]));
                _reg_reduce_max2_3.x = max_noftz(_reg_reduce_max2_3.x, max_noftz(_tmem_load_1[20], _tmem_load_1[21]));
                _reg_reduce_max2_3.y = max_noftz(_reg_reduce_max2_3.y, max_noftz(_tmem_load_1[22], _tmem_load_1[23]));
                _reg_reduce_max2_3.x = max_noftz(_reg_reduce_max2_3.x, max_noftz(_tmem_load_1[24], _tmem_load_1[25]));
                _reg_reduce_max2_3.y = max_noftz(_reg_reduce_max2_3.y, max_noftz(_tmem_load_1[26], _tmem_load_1[27]));
                _reg_reduce_max2_3.x = max_noftz(_reg_reduce_max2_3.x, max_noftz(_tmem_load_1[28], _tmem_load_1[29]));
                _reg_reduce_max2_3.y = max_noftz(_reg_reduce_max2_3.y, max_noftz(_tmem_load_1[30], _tmem_load_1[31]));
                float _tmem_load_1_max = row_max_reduce(_reg_reduce_max2_3);
                float _max_65 = max_noftz(_tmem_load_0_max, _tmem_load_1_max);
                float tile_max = _max_65;
                float tile_max_scaled = ((tile_max != -SM110_XQA_INF) ? tile_max * scale : -SM110_XQA_INF);
                float _max_66 = max_noftz(m_used, tile_max_scaled);
                float m_new = _max_66;
                float need_rescale = ((m_used != -SM110_XQA_INF && m_new > m_used + 8.0f) ? 1.0f : 0.0f);
                float alpha = 1.0f;
                if (need_rescale != 0.0f || m_used == -SM110_XQA_INF) {
                    float _exp2_0 = approx_exp2(m_used - m_new);
                    alpha = ((m_used != -SM110_XQA_INF) ? _exp2_0 : 1.0f);
                    m_used = m_new;
                }
                mbarrier_wait_hint(p_empty_addr + (sm_p_stage) * 8, _phase_p_empty, 10000000);
                float any_rescale = need_rescale;
                float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, any_rescale, 1);
                float _max_67 = max_noftz(any_rescale, _shfl_xor_0);
                any_rescale = _max_67;
                float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, any_rescale, 2);
                float _max_68 = max_noftz(any_rescale, _shfl_xor_1);
                any_rescale = _max_68;
                float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, any_rescale, 4);
                float _max_69 = max_noftz(any_rescale, _shfl_xor_2);
                any_rescale = _max_69;
                float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, any_rescale, 8);
                float _max_70 = max_noftz(any_rescale, _shfl_xor_3);
                any_rescale = _max_70;
                float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, any_rescale, 16);
                float _max_71 = max_noftz(any_rescale, _shfl_xor_4);
                any_rescale = _max_71;
                if (any_rescale != 0.0f) {
                    mbarrier_wait_hint(p_empty_addr + (tile - 1 & 1) * 8, tile - 1 >> 1 & 1, 10000000);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float _tmem_load_2[16];
                    tmem_ld_x16(&_tmem_load_2[0], taddr + 256 + (unsigned int)row_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    const float2 _scale2_4 = {alpha, alpha};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_2)[_ls], _scale2_4);
                    tmem_st_x16_f32(taddr + 256 + (unsigned int)row_base, _tmem_load_2);
                    float _tmem_load_3[16];
                    tmem_ld_x16(&_tmem_load_3[0], taddr + 272 + (unsigned int)row_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    const float2 _scale2_5 = {alpha, alpha};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_3)[_ls], _scale2_5);
                    tmem_st_x16_f32(taddr + 272 + (unsigned int)row_base, _tmem_load_3);
                    float _tmem_load_4[16];
                    tmem_ld_x16(&_tmem_load_4[0], taddr + 288 + (unsigned int)row_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    const float2 _scale2_6 = {alpha, alpha};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_4)[_ls], _scale2_6);
                    tmem_st_x16_f32(taddr + 288 + (unsigned int)row_base, _tmem_load_4);
                    float _tmem_load_5[16];
                    tmem_ld_x16(&_tmem_load_5[0], taddr + 304 + (unsigned int)row_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    const float2 _scale2_7 = {alpha, alpha};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_5)[_ls], _scale2_7);
                    tmem_st_x16_f32(taddr + 304 + (unsigned int)row_base, _tmem_load_5);
                    float _tmem_load_6[16];
                    tmem_ld_x16(&_tmem_load_6[0], taddr + 320 + (unsigned int)row_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    const float2 _scale2_8 = {alpha, alpha};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_6)[_ls], _scale2_8);
                    tmem_st_x16_f32(taddr + 320 + (unsigned int)row_base, _tmem_load_6);
                    float _tmem_load_7[16];
                    tmem_ld_x16(&_tmem_load_7[0], taddr + 336 + (unsigned int)row_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    const float2 _scale2_9 = {alpha, alpha};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_7)[_ls], _scale2_9);
                    tmem_st_x16_f32(taddr + 336 + (unsigned int)row_base, _tmem_load_7);
                    float _tmem_load_8[16];
                    tmem_ld_x16(&_tmem_load_8[0], taddr + 352 + (unsigned int)row_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    const float2 _scale2_10 = {alpha, alpha};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_8)[_ls], _scale2_10);
                    tmem_st_x16_f32(taddr + 352 + (unsigned int)row_base, _tmem_load_8);
                    float _tmem_load_9[16];
                    tmem_ld_x16(&_tmem_load_9[0], taddr + 368 + (unsigned int)row_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    const float2 _scale2_11 = {alpha, alpha};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_9)[_ls], _scale2_11);
                    tmem_st_x16_f32(taddr + 368 + (unsigned int)row_base, _tmem_load_9);
                    float _tmem_load_10[16];
                    tmem_ld_x16(&_tmem_load_10[0], taddr + 384 + (unsigned int)row_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    const float2 _scale2_12 = {alpha, alpha};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_10)[_ls], _scale2_12);
                    tmem_st_x16_f32(taddr + 384 + (unsigned int)row_base, _tmem_load_10);
                    float _tmem_load_11[16];
                    tmem_ld_x16(&_tmem_load_11[0], taddr + 400 + (unsigned int)row_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    const float2 _scale2_13 = {alpha, alpha};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_11)[_ls], _scale2_13);
                    tmem_st_x16_f32(taddr + 400 + (unsigned int)row_base, _tmem_load_11);
                    float _tmem_load_12[16];
                    tmem_ld_x16(&_tmem_load_12[0], taddr + 416 + (unsigned int)row_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    const float2 _scale2_14 = {alpha, alpha};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_12)[_ls], _scale2_14);
                    tmem_st_x16_f32(taddr + 416 + (unsigned int)row_base, _tmem_load_12);
                    float _tmem_load_13[16];
                    tmem_ld_x16(&_tmem_load_13[0], taddr + 432 + (unsigned int)row_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    const float2 _scale2_15 = {alpha, alpha};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_13)[_ls], _scale2_15);
                    tmem_st_x16_f32(taddr + 432 + (unsigned int)row_base, _tmem_load_13);
                    float _tmem_load_14[16];
                    tmem_ld_x16(&_tmem_load_14[0], taddr + 448 + (unsigned int)row_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    const float2 _scale2_16 = {alpha, alpha};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_14)[_ls], _scale2_16);
                    tmem_st_x16_f32(taddr + 448 + (unsigned int)row_base, _tmem_load_14);
                    float _tmem_load_15[16];
                    tmem_ld_x16(&_tmem_load_15[0], taddr + 464 + (unsigned int)row_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    const float2 _scale2_17 = {alpha, alpha};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_15)[_ls], _scale2_17);
                    tmem_st_x16_f32(taddr + 464 + (unsigned int)row_base, _tmem_load_15);
                    float _tmem_load_16[16];
                    tmem_ld_x16(&_tmem_load_16[0], taddr + 480 + (unsigned int)row_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    const float2 _scale2_18 = {alpha, alpha};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_16)[_ls], _scale2_18);
                    tmem_st_x16_f32(taddr + 480 + (unsigned int)row_base, _tmem_load_16);
                    float _tmem_load_17[16];
                    tmem_ld_x16(&_tmem_load_17[0], taddr + 496 + (unsigned int)row_base);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    const float2 _scale2_19 = {alpha, alpha};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_17)[_ls], _scale2_19);
                    tmem_st_x16_f32(taddr + 496 + (unsigned int)row_base, _tmem_load_17);
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                }
                float bias = ((m_used != -SM110_XQA_INF) ? -m_used : 0.0f);
                const float2 _fma_b2_20 = {scale, scale};
                const float2 _fma_c2_21 = {bias, bias};
                float2 _fma_pair_22 = fma_f32x2(make_float2(_tmem_load_0[0], _tmem_load_0[1]), _fma_b2_20, _fma_c2_21);
                _tmem_load_0[0] = _fma_pair_22.x;
                _tmem_load_0[1] = _fma_pair_22.y;
                float2 _fma_pair_23 = fma_f32x2(make_float2(_tmem_load_0[2], _tmem_load_0[3]), _fma_b2_20, _fma_c2_21);
                _tmem_load_0[2] = _fma_pair_23.x;
                _tmem_load_0[3] = _fma_pair_23.y;
                float2 _fma_pair_24 = fma_f32x2(make_float2(_tmem_load_0[4], _tmem_load_0[5]), _fma_b2_20, _fma_c2_21);
                _tmem_load_0[4] = _fma_pair_24.x;
                _tmem_load_0[5] = _fma_pair_24.y;
                float2 _fma_pair_25 = fma_f32x2(make_float2(_tmem_load_0[6], _tmem_load_0[7]), _fma_b2_20, _fma_c2_21);
                _tmem_load_0[6] = _fma_pair_25.x;
                _tmem_load_0[7] = _fma_pair_25.y;
                float2 _fma_pair_26 = fma_f32x2(make_float2(_tmem_load_0[8], _tmem_load_0[9]), _fma_b2_20, _fma_c2_21);
                _tmem_load_0[8] = _fma_pair_26.x;
                _tmem_load_0[9] = _fma_pair_26.y;
                float2 _fma_pair_27 = fma_f32x2(make_float2(_tmem_load_0[10], _tmem_load_0[11]), _fma_b2_20, _fma_c2_21);
                _tmem_load_0[10] = _fma_pair_27.x;
                _tmem_load_0[11] = _fma_pair_27.y;
                float2 _fma_pair_28 = fma_f32x2(make_float2(_tmem_load_0[12], _tmem_load_0[13]), _fma_b2_20, _fma_c2_21);
                _tmem_load_0[12] = _fma_pair_28.x;
                _tmem_load_0[13] = _fma_pair_28.y;
                float2 _fma_pair_29 = fma_f32x2(make_float2(_tmem_load_0[14], _tmem_load_0[15]), _fma_b2_20, _fma_c2_21);
                _tmem_load_0[14] = _fma_pair_29.x;
                _tmem_load_0[15] = _fma_pair_29.y;
                float2 _fma_pair_30 = fma_f32x2(make_float2(_tmem_load_0[16], _tmem_load_0[17]), _fma_b2_20, _fma_c2_21);
                _tmem_load_0[16] = _fma_pair_30.x;
                _tmem_load_0[17] = _fma_pair_30.y;
                float2 _fma_pair_31 = fma_f32x2(make_float2(_tmem_load_0[18], _tmem_load_0[19]), _fma_b2_20, _fma_c2_21);
                _tmem_load_0[18] = _fma_pair_31.x;
                _tmem_load_0[19] = _fma_pair_31.y;
                float2 _fma_pair_32 = fma_f32x2(make_float2(_tmem_load_0[20], _tmem_load_0[21]), _fma_b2_20, _fma_c2_21);
                _tmem_load_0[20] = _fma_pair_32.x;
                _tmem_load_0[21] = _fma_pair_32.y;
                float2 _fma_pair_33 = fma_f32x2(make_float2(_tmem_load_0[22], _tmem_load_0[23]), _fma_b2_20, _fma_c2_21);
                _tmem_load_0[22] = _fma_pair_33.x;
                _tmem_load_0[23] = _fma_pair_33.y;
                float2 _fma_pair_34 = fma_f32x2(make_float2(_tmem_load_0[24], _tmem_load_0[25]), _fma_b2_20, _fma_c2_21);
                _tmem_load_0[24] = _fma_pair_34.x;
                _tmem_load_0[25] = _fma_pair_34.y;
                float2 _fma_pair_35 = fma_f32x2(make_float2(_tmem_load_0[26], _tmem_load_0[27]), _fma_b2_20, _fma_c2_21);
                _tmem_load_0[26] = _fma_pair_35.x;
                _tmem_load_0[27] = _fma_pair_35.y;
                float2 _fma_pair_36 = fma_f32x2(make_float2(_tmem_load_0[28], _tmem_load_0[29]), _fma_b2_20, _fma_c2_21);
                _tmem_load_0[28] = _fma_pair_36.x;
                _tmem_load_0[29] = _fma_pair_36.y;
                float2 _fma_pair_37 = fma_f32x2(make_float2(_tmem_load_0[30], _tmem_load_0[31]), _fma_b2_20, _fma_c2_21);
                _tmem_load_0[30] = _fma_pair_37.x;
                _tmem_load_0[31] = _fma_pair_37.y;
                const float2 _fma_b2_38 = {scale, scale};
                const float2 _fma_c2_39 = {bias, bias};
                float2 _fma_pair_40 = fma_f32x2(make_float2(_tmem_load_1[0], _tmem_load_1[1]), _fma_b2_38, _fma_c2_39);
                _tmem_load_1[0] = _fma_pair_40.x;
                _tmem_load_1[1] = _fma_pair_40.y;
                float2 _fma_pair_41 = fma_f32x2(make_float2(_tmem_load_1[2], _tmem_load_1[3]), _fma_b2_38, _fma_c2_39);
                _tmem_load_1[2] = _fma_pair_41.x;
                _tmem_load_1[3] = _fma_pair_41.y;
                float2 _fma_pair_42 = fma_f32x2(make_float2(_tmem_load_1[4], _tmem_load_1[5]), _fma_b2_38, _fma_c2_39);
                _tmem_load_1[4] = _fma_pair_42.x;
                _tmem_load_1[5] = _fma_pair_42.y;
                float2 _fma_pair_43 = fma_f32x2(make_float2(_tmem_load_1[6], _tmem_load_1[7]), _fma_b2_38, _fma_c2_39);
                _tmem_load_1[6] = _fma_pair_43.x;
                _tmem_load_1[7] = _fma_pair_43.y;
                float2 _fma_pair_44 = fma_f32x2(make_float2(_tmem_load_1[8], _tmem_load_1[9]), _fma_b2_38, _fma_c2_39);
                _tmem_load_1[8] = _fma_pair_44.x;
                _tmem_load_1[9] = _fma_pair_44.y;
                float2 _fma_pair_45 = fma_f32x2(make_float2(_tmem_load_1[10], _tmem_load_1[11]), _fma_b2_38, _fma_c2_39);
                _tmem_load_1[10] = _fma_pair_45.x;
                _tmem_load_1[11] = _fma_pair_45.y;
                float2 _fma_pair_46 = fma_f32x2(make_float2(_tmem_load_1[12], _tmem_load_1[13]), _fma_b2_38, _fma_c2_39);
                _tmem_load_1[12] = _fma_pair_46.x;
                _tmem_load_1[13] = _fma_pair_46.y;
                float2 _fma_pair_47 = fma_f32x2(make_float2(_tmem_load_1[14], _tmem_load_1[15]), _fma_b2_38, _fma_c2_39);
                _tmem_load_1[14] = _fma_pair_47.x;
                _tmem_load_1[15] = _fma_pair_47.y;
                float2 _fma_pair_48 = fma_f32x2(make_float2(_tmem_load_1[16], _tmem_load_1[17]), _fma_b2_38, _fma_c2_39);
                _tmem_load_1[16] = _fma_pair_48.x;
                _tmem_load_1[17] = _fma_pair_48.y;
                float2 _fma_pair_49 = fma_f32x2(make_float2(_tmem_load_1[18], _tmem_load_1[19]), _fma_b2_38, _fma_c2_39);
                _tmem_load_1[18] = _fma_pair_49.x;
                _tmem_load_1[19] = _fma_pair_49.y;
                float2 _fma_pair_50 = fma_f32x2(make_float2(_tmem_load_1[20], _tmem_load_1[21]), _fma_b2_38, _fma_c2_39);
                _tmem_load_1[20] = _fma_pair_50.x;
                _tmem_load_1[21] = _fma_pair_50.y;
                float2 _fma_pair_51 = fma_f32x2(make_float2(_tmem_load_1[22], _tmem_load_1[23]), _fma_b2_38, _fma_c2_39);
                _tmem_load_1[22] = _fma_pair_51.x;
                _tmem_load_1[23] = _fma_pair_51.y;
                float2 _fma_pair_52 = fma_f32x2(make_float2(_tmem_load_1[24], _tmem_load_1[25]), _fma_b2_38, _fma_c2_39);
                _tmem_load_1[24] = _fma_pair_52.x;
                _tmem_load_1[25] = _fma_pair_52.y;
                float2 _fma_pair_53 = fma_f32x2(make_float2(_tmem_load_1[26], _tmem_load_1[27]), _fma_b2_38, _fma_c2_39);
                _tmem_load_1[26] = _fma_pair_53.x;
                _tmem_load_1[27] = _fma_pair_53.y;
                float2 _fma_pair_54 = fma_f32x2(make_float2(_tmem_load_1[28], _tmem_load_1[29]), _fma_b2_38, _fma_c2_39);
                _tmem_load_1[28] = _fma_pair_54.x;
                _tmem_load_1[29] = _fma_pair_54.y;
                float2 _fma_pair_55 = fma_f32x2(make_float2(_tmem_load_1[30], _tmem_load_1[31]), _fma_b2_38, _fma_c2_39);
                _tmem_load_1[30] = _fma_pair_55.x;
                _tmem_load_1[31] = _fma_pair_55.y;
                #pragma unroll
                for (int _le = 0; _le < 32; _le++) {
                    _tmem_load_0[_le] = approx_exp2(_tmem_load_0[_le]);
                }
                #pragma unroll
                for (int _le = 0; _le < 32; _le++) {
                    _tmem_load_1[_le] = approx_exp2(_tmem_load_1[_le]);
                }
                float2 _reg_reduce_sum2_56 = make_float2(0.0f, 0.0f);
                _reg_reduce_sum2_56 = add_f32x2(_reg_reduce_sum2_56, make_float2(_tmem_load_0[0], _tmem_load_0[1]));
                _reg_reduce_sum2_56 = add_f32x2(_reg_reduce_sum2_56, make_float2(_tmem_load_0[2], _tmem_load_0[3]));
                _reg_reduce_sum2_56 = add_f32x2(_reg_reduce_sum2_56, make_float2(_tmem_load_0[4], _tmem_load_0[5]));
                _reg_reduce_sum2_56 = add_f32x2(_reg_reduce_sum2_56, make_float2(_tmem_load_0[6], _tmem_load_0[7]));
                _reg_reduce_sum2_56 = add_f32x2(_reg_reduce_sum2_56, make_float2(_tmem_load_0[8], _tmem_load_0[9]));
                _reg_reduce_sum2_56 = add_f32x2(_reg_reduce_sum2_56, make_float2(_tmem_load_0[10], _tmem_load_0[11]));
                _reg_reduce_sum2_56 = add_f32x2(_reg_reduce_sum2_56, make_float2(_tmem_load_0[12], _tmem_load_0[13]));
                _reg_reduce_sum2_56 = add_f32x2(_reg_reduce_sum2_56, make_float2(_tmem_load_0[14], _tmem_load_0[15]));
                _reg_reduce_sum2_56 = add_f32x2(_reg_reduce_sum2_56, make_float2(_tmem_load_0[16], _tmem_load_0[17]));
                _reg_reduce_sum2_56 = add_f32x2(_reg_reduce_sum2_56, make_float2(_tmem_load_0[18], _tmem_load_0[19]));
                _reg_reduce_sum2_56 = add_f32x2(_reg_reduce_sum2_56, make_float2(_tmem_load_0[20], _tmem_load_0[21]));
                _reg_reduce_sum2_56 = add_f32x2(_reg_reduce_sum2_56, make_float2(_tmem_load_0[22], _tmem_load_0[23]));
                _reg_reduce_sum2_56 = add_f32x2(_reg_reduce_sum2_56, make_float2(_tmem_load_0[24], _tmem_load_0[25]));
                _reg_reduce_sum2_56 = add_f32x2(_reg_reduce_sum2_56, make_float2(_tmem_load_0[26], _tmem_load_0[27]));
                _reg_reduce_sum2_56 = add_f32x2(_reg_reduce_sum2_56, make_float2(_tmem_load_0[28], _tmem_load_0[29]));
                _reg_reduce_sum2_56 = add_f32x2(_reg_reduce_sum2_56, make_float2(_tmem_load_0[30], _tmem_load_0[31]));
                float _tmem_load_0_sum = _reg_reduce_sum2_56.x + _reg_reduce_sum2_56.y;
                float2 _reg_reduce_sum2_57 = make_float2(0.0f, 0.0f);
                _reg_reduce_sum2_57 = add_f32x2(_reg_reduce_sum2_57, make_float2(_tmem_load_1[0], _tmem_load_1[1]));
                _reg_reduce_sum2_57 = add_f32x2(_reg_reduce_sum2_57, make_float2(_tmem_load_1[2], _tmem_load_1[3]));
                _reg_reduce_sum2_57 = add_f32x2(_reg_reduce_sum2_57, make_float2(_tmem_load_1[4], _tmem_load_1[5]));
                _reg_reduce_sum2_57 = add_f32x2(_reg_reduce_sum2_57, make_float2(_tmem_load_1[6], _tmem_load_1[7]));
                _reg_reduce_sum2_57 = add_f32x2(_reg_reduce_sum2_57, make_float2(_tmem_load_1[8], _tmem_load_1[9]));
                _reg_reduce_sum2_57 = add_f32x2(_reg_reduce_sum2_57, make_float2(_tmem_load_1[10], _tmem_load_1[11]));
                _reg_reduce_sum2_57 = add_f32x2(_reg_reduce_sum2_57, make_float2(_tmem_load_1[12], _tmem_load_1[13]));
                _reg_reduce_sum2_57 = add_f32x2(_reg_reduce_sum2_57, make_float2(_tmem_load_1[14], _tmem_load_1[15]));
                _reg_reduce_sum2_57 = add_f32x2(_reg_reduce_sum2_57, make_float2(_tmem_load_1[16], _tmem_load_1[17]));
                _reg_reduce_sum2_57 = add_f32x2(_reg_reduce_sum2_57, make_float2(_tmem_load_1[18], _tmem_load_1[19]));
                _reg_reduce_sum2_57 = add_f32x2(_reg_reduce_sum2_57, make_float2(_tmem_load_1[20], _tmem_load_1[21]));
                _reg_reduce_sum2_57 = add_f32x2(_reg_reduce_sum2_57, make_float2(_tmem_load_1[22], _tmem_load_1[23]));
                _reg_reduce_sum2_57 = add_f32x2(_reg_reduce_sum2_57, make_float2(_tmem_load_1[24], _tmem_load_1[25]));
                _reg_reduce_sum2_57 = add_f32x2(_reg_reduce_sum2_57, make_float2(_tmem_load_1[26], _tmem_load_1[27]));
                _reg_reduce_sum2_57 = add_f32x2(_reg_reduce_sum2_57, make_float2(_tmem_load_1[28], _tmem_load_1[29]));
                _reg_reduce_sum2_57 = add_f32x2(_reg_reduce_sum2_57, make_float2(_tmem_load_1[30], _tmem_load_1[31]));
                float _tmem_load_1_sum = _reg_reduce_sum2_57.x + _reg_reduce_sum2_57.y;
                row_sum = row_sum * alpha + _tmem_load_0_sum + _tmem_load_1_sum;
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __half2 _h2 = __float22half2_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                    packed[_lp] = *(uint32_t*)&_h2;
                }
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __half2 _h2 = __float22half2_rn(make_float2(_tmem_load_1[_lp*2 + 0], _tmem_load_1[_lp*2+1 + 0]));
                    packed[_lp + 16] = *(uint32_t*)&_h2;
                }
                asm volatile("tcgen05.fence::after_thread_sync;");
                int p_addr = taddr + 128 + sm_p_stage * 32 + (unsigned int)row_base;
                tmem_st_x32(p_addr, packed);
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                mbarrier_arrive(p_full_addr + (sm_p_stage) * 8);
                sm_p_stage += 1;
                if (sm_p_stage == 2) { sm_p_stage = 0; _phase_p_empty ^= 1; }
            }
            unsigned int _phase_o_ready_0 = 0;
            mbarrier_wait_hint(o_ready_addr, _phase_o_ready_0, 10000000);
            _phase_o_ready_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float _rcp_0 = approx_rcp(row_sum);
            float inverse = ((row_sum > 0.0f) ? _rcp_0 : 0.0f);
            inverse = inverse * v_cache_scale;
            float _tmem_load_18[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(_tmem_load_18[0]), "=f"(_tmem_load_18[1]), "=f"(_tmem_load_18[2]), "=f"(_tmem_load_18[3]), "=f"(_tmem_load_18[4]), "=f"(_tmem_load_18[5]), "=f"(_tmem_load_18[6]), "=f"(_tmem_load_18[7]), "=f"(_tmem_load_18[8]), "=f"(_tmem_load_18[9]), "=f"(_tmem_load_18[10]), "=f"(_tmem_load_18[11]), "=f"(_tmem_load_18[12]), "=f"(_tmem_load_18[13]), "=f"(_tmem_load_18[14]), "=f"(_tmem_load_18[15]), "=f"(_tmem_load_18[16]), "=f"(_tmem_load_18[17]), "=f"(_tmem_load_18[18]), "=f"(_tmem_load_18[19]), "=f"(_tmem_load_18[20]), "=f"(_tmem_load_18[21]), "=f"(_tmem_load_18[22]), "=f"(_tmem_load_18[23]), "=f"(_tmem_load_18[24]), "=f"(_tmem_load_18[25]), "=f"(_tmem_load_18[26]), "=f"(_tmem_load_18[27]), "=f"(_tmem_load_18[28]), "=f"(_tmem_load_18[29]), "=f"(_tmem_load_18[30]), "=f"(_tmem_load_18[31])
                : "r"(taddr + 256 + (unsigned int)row_base));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            const float2 _scale2_58 = {inverse, inverse};
            #pragma unroll
            for (int _ls = 0; _ls < 16; _ls++)
                mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_18)[_ls], _scale2_58);
            #pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
                __half2 _h2 = __float22half2_rn(make_float2(_tmem_load_18[_lp*2 + 0], _tmem_load_18[_lp*2+1 + 0]));
                pk[_lp] = *(uint32_t*)&_h2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sstage_addr + my_row * 528), "r"(pk[0]), "r"(pk[1]), "r"(pk[2]), "r"(pk[3]) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sstage_addr + (my_row * 528 + 16)), "r"(pk[4]), "r"(pk[5]), "r"(pk[6]), "r"(pk[7]) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sstage_addr + (my_row * 528 + 32)), "r"(pk[8]), "r"(pk[9]), "r"(pk[10]), "r"(pk[11]) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sstage_addr + (my_row * 528 + 48)), "r"(pk[12]), "r"(pk[13]), "r"(pk[14]), "r"(pk[15]) : "memory");
            float _tmem_load_19[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(_tmem_load_19[0]), "=f"(_tmem_load_19[1]), "=f"(_tmem_load_19[2]), "=f"(_tmem_load_19[3]), "=f"(_tmem_load_19[4]), "=f"(_tmem_load_19[5]), "=f"(_tmem_load_19[6]), "=f"(_tmem_load_19[7]), "=f"(_tmem_load_19[8]), "=f"(_tmem_load_19[9]), "=f"(_tmem_load_19[10]), "=f"(_tmem_load_19[11]), "=f"(_tmem_load_19[12]), "=f"(_tmem_load_19[13]), "=f"(_tmem_load_19[14]), "=f"(_tmem_load_19[15]), "=f"(_tmem_load_19[16]), "=f"(_tmem_load_19[17]), "=f"(_tmem_load_19[18]), "=f"(_tmem_load_19[19]), "=f"(_tmem_load_19[20]), "=f"(_tmem_load_19[21]), "=f"(_tmem_load_19[22]), "=f"(_tmem_load_19[23]), "=f"(_tmem_load_19[24]), "=f"(_tmem_load_19[25]), "=f"(_tmem_load_19[26]), "=f"(_tmem_load_19[27]), "=f"(_tmem_load_19[28]), "=f"(_tmem_load_19[29]), "=f"(_tmem_load_19[30]), "=f"(_tmem_load_19[31])
                : "r"(taddr + 288 + (unsigned int)row_base));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            const float2 _scale2_59 = {inverse, inverse};
            #pragma unroll
            for (int _ls = 0; _ls < 16; _ls++)
                mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_19)[_ls], _scale2_59);
            #pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
                __half2 _h2 = __float22half2_rn(make_float2(_tmem_load_19[_lp*2 + 0], _tmem_load_19[_lp*2+1 + 0]));
                pk[_lp] = *(uint32_t*)&_h2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sstage_addr + (my_row * 528 + 64)), "r"(pk[0]), "r"(pk[1]), "r"(pk[2]), "r"(pk[3]) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sstage_addr + (my_row * 528 + 64 + 16)), "r"(pk[4]), "r"(pk[5]), "r"(pk[6]), "r"(pk[7]) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sstage_addr + (my_row * 528 + 64 + 32)), "r"(pk[8]), "r"(pk[9]), "r"(pk[10]), "r"(pk[11]) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sstage_addr + (my_row * 528 + 64 + 48)), "r"(pk[12]), "r"(pk[13]), "r"(pk[14]), "r"(pk[15]) : "memory");
            float _tmem_load_20[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(_tmem_load_20[0]), "=f"(_tmem_load_20[1]), "=f"(_tmem_load_20[2]), "=f"(_tmem_load_20[3]), "=f"(_tmem_load_20[4]), "=f"(_tmem_load_20[5]), "=f"(_tmem_load_20[6]), "=f"(_tmem_load_20[7]), "=f"(_tmem_load_20[8]), "=f"(_tmem_load_20[9]), "=f"(_tmem_load_20[10]), "=f"(_tmem_load_20[11]), "=f"(_tmem_load_20[12]), "=f"(_tmem_load_20[13]), "=f"(_tmem_load_20[14]), "=f"(_tmem_load_20[15]), "=f"(_tmem_load_20[16]), "=f"(_tmem_load_20[17]), "=f"(_tmem_load_20[18]), "=f"(_tmem_load_20[19]), "=f"(_tmem_load_20[20]), "=f"(_tmem_load_20[21]), "=f"(_tmem_load_20[22]), "=f"(_tmem_load_20[23]), "=f"(_tmem_load_20[24]), "=f"(_tmem_load_20[25]), "=f"(_tmem_load_20[26]), "=f"(_tmem_load_20[27]), "=f"(_tmem_load_20[28]), "=f"(_tmem_load_20[29]), "=f"(_tmem_load_20[30]), "=f"(_tmem_load_20[31])
                : "r"(taddr + 320 + (unsigned int)row_base));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            const float2 _scale2_60 = {inverse, inverse};
            #pragma unroll
            for (int _ls = 0; _ls < 16; _ls++)
                mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_20)[_ls], _scale2_60);
            #pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
                __half2 _h2 = __float22half2_rn(make_float2(_tmem_load_20[_lp*2 + 0], _tmem_load_20[_lp*2+1 + 0]));
                pk[_lp] = *(uint32_t*)&_h2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sstage_addr + (my_row * 528 + 128)), "r"(pk[0]), "r"(pk[1]), "r"(pk[2]), "r"(pk[3]) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sstage_addr + (my_row * 528 + 128 + 16)), "r"(pk[4]), "r"(pk[5]), "r"(pk[6]), "r"(pk[7]) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sstage_addr + (my_row * 528 + 128 + 32)), "r"(pk[8]), "r"(pk[9]), "r"(pk[10]), "r"(pk[11]) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sstage_addr + (my_row * 528 + 128 + 48)), "r"(pk[12]), "r"(pk[13]), "r"(pk[14]), "r"(pk[15]) : "memory");
            float _tmem_load_21[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(_tmem_load_21[0]), "=f"(_tmem_load_21[1]), "=f"(_tmem_load_21[2]), "=f"(_tmem_load_21[3]), "=f"(_tmem_load_21[4]), "=f"(_tmem_load_21[5]), "=f"(_tmem_load_21[6]), "=f"(_tmem_load_21[7]), "=f"(_tmem_load_21[8]), "=f"(_tmem_load_21[9]), "=f"(_tmem_load_21[10]), "=f"(_tmem_load_21[11]), "=f"(_tmem_load_21[12]), "=f"(_tmem_load_21[13]), "=f"(_tmem_load_21[14]), "=f"(_tmem_load_21[15]), "=f"(_tmem_load_21[16]), "=f"(_tmem_load_21[17]), "=f"(_tmem_load_21[18]), "=f"(_tmem_load_21[19]), "=f"(_tmem_load_21[20]), "=f"(_tmem_load_21[21]), "=f"(_tmem_load_21[22]), "=f"(_tmem_load_21[23]), "=f"(_tmem_load_21[24]), "=f"(_tmem_load_21[25]), "=f"(_tmem_load_21[26]), "=f"(_tmem_load_21[27]), "=f"(_tmem_load_21[28]), "=f"(_tmem_load_21[29]), "=f"(_tmem_load_21[30]), "=f"(_tmem_load_21[31])
                : "r"(taddr + 352 + (unsigned int)row_base));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            const float2 _scale2_61 = {inverse, inverse};
            #pragma unroll
            for (int _ls = 0; _ls < 16; _ls++)
                mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_21)[_ls], _scale2_61);
            #pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
                __half2 _h2 = __float22half2_rn(make_float2(_tmem_load_21[_lp*2 + 0], _tmem_load_21[_lp*2+1 + 0]));
                pk[_lp] = *(uint32_t*)&_h2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sstage_addr + (my_row * 528 + 192)), "r"(pk[0]), "r"(pk[1]), "r"(pk[2]), "r"(pk[3]) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sstage_addr + (my_row * 528 + 192 + 16)), "r"(pk[4]), "r"(pk[5]), "r"(pk[6]), "r"(pk[7]) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sstage_addr + (my_row * 528 + 192 + 32)), "r"(pk[8]), "r"(pk[9]), "r"(pk[10]), "r"(pk[11]) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sstage_addr + (my_row * 528 + 192 + 48)), "r"(pk[12]), "r"(pk[13]), "r"(pk[14]), "r"(pk[15]) : "memory");
            float _tmem_load_22[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(_tmem_load_22[0]), "=f"(_tmem_load_22[1]), "=f"(_tmem_load_22[2]), "=f"(_tmem_load_22[3]), "=f"(_tmem_load_22[4]), "=f"(_tmem_load_22[5]), "=f"(_tmem_load_22[6]), "=f"(_tmem_load_22[7]), "=f"(_tmem_load_22[8]), "=f"(_tmem_load_22[9]), "=f"(_tmem_load_22[10]), "=f"(_tmem_load_22[11]), "=f"(_tmem_load_22[12]), "=f"(_tmem_load_22[13]), "=f"(_tmem_load_22[14]), "=f"(_tmem_load_22[15]), "=f"(_tmem_load_22[16]), "=f"(_tmem_load_22[17]), "=f"(_tmem_load_22[18]), "=f"(_tmem_load_22[19]), "=f"(_tmem_load_22[20]), "=f"(_tmem_load_22[21]), "=f"(_tmem_load_22[22]), "=f"(_tmem_load_22[23]), "=f"(_tmem_load_22[24]), "=f"(_tmem_load_22[25]), "=f"(_tmem_load_22[26]), "=f"(_tmem_load_22[27]), "=f"(_tmem_load_22[28]), "=f"(_tmem_load_22[29]), "=f"(_tmem_load_22[30]), "=f"(_tmem_load_22[31])
                : "r"(taddr + 384 + (unsigned int)row_base));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            const float2 _scale2_62 = {inverse, inverse};
            #pragma unroll
            for (int _ls = 0; _ls < 16; _ls++)
                mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_22)[_ls], _scale2_62);
            #pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
                __half2 _h2 = __float22half2_rn(make_float2(_tmem_load_22[_lp*2 + 0], _tmem_load_22[_lp*2+1 + 0]));
                pk[_lp] = *(uint32_t*)&_h2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sstage_addr + (my_row * 528 + 256)), "r"(pk[0]), "r"(pk[1]), "r"(pk[2]), "r"(pk[3]) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sstage_addr + (my_row * 528 + 256 + 16)), "r"(pk[4]), "r"(pk[5]), "r"(pk[6]), "r"(pk[7]) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sstage_addr + (my_row * 528 + 256 + 32)), "r"(pk[8]), "r"(pk[9]), "r"(pk[10]), "r"(pk[11]) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sstage_addr + (my_row * 528 + 256 + 48)), "r"(pk[12]), "r"(pk[13]), "r"(pk[14]), "r"(pk[15]) : "memory");
            float _tmem_load_23[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(_tmem_load_23[0]), "=f"(_tmem_load_23[1]), "=f"(_tmem_load_23[2]), "=f"(_tmem_load_23[3]), "=f"(_tmem_load_23[4]), "=f"(_tmem_load_23[5]), "=f"(_tmem_load_23[6]), "=f"(_tmem_load_23[7]), "=f"(_tmem_load_23[8]), "=f"(_tmem_load_23[9]), "=f"(_tmem_load_23[10]), "=f"(_tmem_load_23[11]), "=f"(_tmem_load_23[12]), "=f"(_tmem_load_23[13]), "=f"(_tmem_load_23[14]), "=f"(_tmem_load_23[15]), "=f"(_tmem_load_23[16]), "=f"(_tmem_load_23[17]), "=f"(_tmem_load_23[18]), "=f"(_tmem_load_23[19]), "=f"(_tmem_load_23[20]), "=f"(_tmem_load_23[21]), "=f"(_tmem_load_23[22]), "=f"(_tmem_load_23[23]), "=f"(_tmem_load_23[24]), "=f"(_tmem_load_23[25]), "=f"(_tmem_load_23[26]), "=f"(_tmem_load_23[27]), "=f"(_tmem_load_23[28]), "=f"(_tmem_load_23[29]), "=f"(_tmem_load_23[30]), "=f"(_tmem_load_23[31])
                : "r"(taddr + 416 + (unsigned int)row_base));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            const float2 _scale2_63 = {inverse, inverse};
            #pragma unroll
            for (int _ls = 0; _ls < 16; _ls++)
                mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_23)[_ls], _scale2_63);
            #pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
                __half2 _h2 = __float22half2_rn(make_float2(_tmem_load_23[_lp*2 + 0], _tmem_load_23[_lp*2+1 + 0]));
                pk[_lp] = *(uint32_t*)&_h2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sstage_addr + (my_row * 528 + 320)), "r"(pk[0]), "r"(pk[1]), "r"(pk[2]), "r"(pk[3]) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sstage_addr + (my_row * 528 + 320 + 16)), "r"(pk[4]), "r"(pk[5]), "r"(pk[6]), "r"(pk[7]) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sstage_addr + (my_row * 528 + 320 + 32)), "r"(pk[8]), "r"(pk[9]), "r"(pk[10]), "r"(pk[11]) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sstage_addr + (my_row * 528 + 320 + 48)), "r"(pk[12]), "r"(pk[13]), "r"(pk[14]), "r"(pk[15]) : "memory");
            float _tmem_load_24[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(_tmem_load_24[0]), "=f"(_tmem_load_24[1]), "=f"(_tmem_load_24[2]), "=f"(_tmem_load_24[3]), "=f"(_tmem_load_24[4]), "=f"(_tmem_load_24[5]), "=f"(_tmem_load_24[6]), "=f"(_tmem_load_24[7]), "=f"(_tmem_load_24[8]), "=f"(_tmem_load_24[9]), "=f"(_tmem_load_24[10]), "=f"(_tmem_load_24[11]), "=f"(_tmem_load_24[12]), "=f"(_tmem_load_24[13]), "=f"(_tmem_load_24[14]), "=f"(_tmem_load_24[15]), "=f"(_tmem_load_24[16]), "=f"(_tmem_load_24[17]), "=f"(_tmem_load_24[18]), "=f"(_tmem_load_24[19]), "=f"(_tmem_load_24[20]), "=f"(_tmem_load_24[21]), "=f"(_tmem_load_24[22]), "=f"(_tmem_load_24[23]), "=f"(_tmem_load_24[24]), "=f"(_tmem_load_24[25]), "=f"(_tmem_load_24[26]), "=f"(_tmem_load_24[27]), "=f"(_tmem_load_24[28]), "=f"(_tmem_load_24[29]), "=f"(_tmem_load_24[30]), "=f"(_tmem_load_24[31])
                : "r"(taddr + 448 + (unsigned int)row_base));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            const float2 _scale2_64 = {inverse, inverse};
            #pragma unroll
            for (int _ls = 0; _ls < 16; _ls++)
                mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_24)[_ls], _scale2_64);
            #pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
                __half2 _h2 = __float22half2_rn(make_float2(_tmem_load_24[_lp*2 + 0], _tmem_load_24[_lp*2+1 + 0]));
                pk[_lp] = *(uint32_t*)&_h2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sstage_addr + (my_row * 528 + 384)), "r"(pk[0]), "r"(pk[1]), "r"(pk[2]), "r"(pk[3]) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sstage_addr + (my_row * 528 + 384 + 16)), "r"(pk[4]), "r"(pk[5]), "r"(pk[6]), "r"(pk[7]) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sstage_addr + (my_row * 528 + 384 + 32)), "r"(pk[8]), "r"(pk[9]), "r"(pk[10]), "r"(pk[11]) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sstage_addr + (my_row * 528 + 384 + 48)), "r"(pk[12]), "r"(pk[13]), "r"(pk[14]), "r"(pk[15]) : "memory");
            float _tmem_load_25[32];
            asm volatile(
                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                : "=f"(_tmem_load_25[0]), "=f"(_tmem_load_25[1]), "=f"(_tmem_load_25[2]), "=f"(_tmem_load_25[3]), "=f"(_tmem_load_25[4]), "=f"(_tmem_load_25[5]), "=f"(_tmem_load_25[6]), "=f"(_tmem_load_25[7]), "=f"(_tmem_load_25[8]), "=f"(_tmem_load_25[9]), "=f"(_tmem_load_25[10]), "=f"(_tmem_load_25[11]), "=f"(_tmem_load_25[12]), "=f"(_tmem_load_25[13]), "=f"(_tmem_load_25[14]), "=f"(_tmem_load_25[15]), "=f"(_tmem_load_25[16]), "=f"(_tmem_load_25[17]), "=f"(_tmem_load_25[18]), "=f"(_tmem_load_25[19]), "=f"(_tmem_load_25[20]), "=f"(_tmem_load_25[21]), "=f"(_tmem_load_25[22]), "=f"(_tmem_load_25[23]), "=f"(_tmem_load_25[24]), "=f"(_tmem_load_25[25]), "=f"(_tmem_load_25[26]), "=f"(_tmem_load_25[27]), "=f"(_tmem_load_25[28]), "=f"(_tmem_load_25[29]), "=f"(_tmem_load_25[30]), "=f"(_tmem_load_25[31])
                : "r"(taddr + 480 + (unsigned int)row_base));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            const float2 _scale2_65 = {inverse, inverse};
            #pragma unroll
            for (int _ls = 0; _ls < 16; _ls++)
                mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_25)[_ls], _scale2_65);
            #pragma unroll
            for (int _lp = 0; _lp < 16; _lp++) {
                __half2 _h2 = __float22half2_rn(make_float2(_tmem_load_25[_lp*2 + 0], _tmem_load_25[_lp*2+1 + 0]));
                pk[_lp] = *(uint32_t*)&_h2;
            }
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sstage_addr + (my_row * 528 + 448)), "r"(pk[0]), "r"(pk[1]), "r"(pk[2]), "r"(pk[3]) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sstage_addr + (my_row * 528 + 448 + 16)), "r"(pk[4]), "r"(pk[5]), "r"(pk[6]), "r"(pk[7]) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sstage_addr + (my_row * 528 + 448 + 32)), "r"(pk[8]), "r"(pk[9]), "r"(pk[10]), "r"(pk[11]) : "memory");
            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sstage_addr + (my_row * 528 + 448 + 48)), "r"(pk[12]), "r"(pk[13]), "r"(pk[14]), "r"(pk[15]) : "memory");
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(tmem_retire_addr);
            asm volatile("barrier.sync 4, 128;" ::: "memory");
            unsigned int valid_rows = actual_q * 2 - row_begin;
            unsigned int q_heads = num_kv_heads * head_group_size;
            #pragma unroll 4
            for (unsigned int i = 0; i < 32; i++) {
                unsigned int r = warp % 4 * 32 + i;
                if (r < valid_rows) {
                    unsigned int ht = row_begin + r;
                    unsigned int orow = (request_offset + ht / 2) * q_heads + head * 2 + ht % 2;
                    uint32_t _sstage_reg_0[4];
                    __int128_t _smem_b128_66;
                    asm volatile("ld.shared.b128 %0, [%1];" : "=q"(_smem_b128_66) : "r"(sstage_addr + (r * 132 + lane * 4) * 4));
                    _sstage_reg_0[0] = reinterpret_cast<const uint32_t*>(&_smem_b128_66)[0];
                    _sstage_reg_0[1] = reinterpret_cast<const uint32_t*>(&_smem_b128_66)[1];
                    _sstage_reg_0[2] = reinterpret_cast<const uint32_t*>(&_smem_b128_66)[2];
                    _sstage_reg_0[3] = reinterpret_cast<const uint32_t*>(&_smem_b128_66)[3];
                    reinterpret_cast<int4*>(output + (orow * 512 + output_col + lane * 8))[0] = reinterpret_cast<int4*>(_sstage_reg_0)[0];
                }
            }
            asm volatile("barrier.cluster.arrive.release;" ::: "memory");
            asm volatile("barrier.cluster.wait.acquire;" ::: "memory");
        }
    }
    // ---- Role: convert ----
    if (warp >= 4 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 96;");
        // Deferred post-initialization cluster wait (WarpConfig.cluster_init_wait_warps)
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        { // convert_main
            unsigned int actual_q_1 = q_seq_len;
            unsigned int request_offset_1 = q_seq_len * (unsigned int)blockIdx.z;
            if ((unsigned long long)q_cu_seq_lens != 0) {
                request_offset_1 = q_cu_seq_lens[blockIdx.z];
                actual_q_1 = q_cu_seq_lens[blockIdx.z + 1] - request_offset_1;
            }
            unsigned int blocks_per_head_1 = (unsigned int)gridDim.y / num_kv_heads;
            unsigned int head_1 = (unsigned int)blockIdx.y / blocks_per_head_1;
            unsigned int row_begin_1 = (unsigned int)blockIdx.y % blocks_per_head_1 * 128;
            unsigned int output_col_1 = blockIdx.x * 256;
            int _vec_load_1[1];
            {
                _vec_load_1[0] = *reinterpret_cast<const int*>(kv_cache_list.sequence_lengths + blockIdx.z);
            }
            unsigned int length_1 = (unsigned int)_vec_load_1[0];
            unsigned int num_tiles_1 = (length_1 + 63) / 64;
            int warp_id_in_role = (warp - 4);
            unsigned int tid_1 = (unsigned int)(warp_id_in_role * 32) + lane;
            unsigned int zero = 0;
            unsigned int conv_stage = 0;
            unsigned int _phase_landed = 0;
            #pragma unroll 1
            for (unsigned int tile_1 = 0; tile_1 < num_tiles_1; tile_1++) {
                #pragma unroll
                for (int c = 0; c < 4; c++) {
                    mbarrier_wait_hint(landed_addr + (conv_stage) * 8, _phase_landed, 10000000);
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(ready_addr + (conv_stage) * 8);
                    }
                    conv_stage += 1;
                    if (conv_stage == 6) { conv_stage = 0; _phase_landed ^= 1; }
                }
                if (tile_1 > 0) {
                    #pragma unroll
                    for (int v = 0; v < 2; v++) {
                        mbarrier_wait_hint(landed_addr + (conv_stage) * 8, _phase_landed, 10000000);
                        {
                            unsigned int _min_1 = ((length_1 - (tile_1 - 1) * 64) < ((unsigned int)64) ? (length_1 - (tile_1 - 1) * 64) : ((unsigned int)64));
                            unsigned int valid_1 = _min_1;
                            if (valid_1 < 64) {
                                unsigned int f16_base = conv_stage * 16384;
                                for (unsigned int g = valid_1 * 8 + tid_1; g < 512; g += 256) {
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sk_addr + (f16_base + g * 16)), "r"(zero), "r"(zero), "r"(zero), "r"(zero) : "memory");
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sk_addr + (f16_base + 8192 + g * 16)), "r"(zero), "r"(zero), "r"(zero), "r"(zero) : "memory");
                                }
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            }
                        }
                        __syncwarp();
                        if (elect_sync()) {
                            mbarrier_arrive(ready_addr + (conv_stage) * 8);
                        }
                        conv_stage += 1;
                        if (conv_stage == 6) { conv_stage = 0; _phase_landed ^= 1; }
                    }
                }
            }
            #pragma unroll
            for (int v_1 = 0; v_1 < 2; v_1++) {
                mbarrier_wait_hint(landed_addr + (conv_stage) * 8, _phase_landed, 10000000);
                {
                    unsigned int _min_2 = ((length_1 - (num_tiles_1 - 1) * 64) < ((unsigned int)64) ? (length_1 - (num_tiles_1 - 1) * 64) : ((unsigned int)64));
                    unsigned int valid_2 = _min_2;
                    if (valid_2 < 64) {
                        unsigned int f16_base_1 = conv_stage * 16384;
                        for (unsigned int g_1 = valid_2 * 8 + tid_1; g_1 < 512; g_1 += 256) {
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sk_addr + (f16_base_1 + g_1 * 16)), "r"(zero), "r"(zero), "r"(zero), "r"(zero) : "memory");
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(sk_addr + (f16_base_1 + 8192 + g_1 * 16)), "r"(zero), "r"(zero), "r"(zero), "r"(zero) : "memory");
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    }
                }
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(ready_addr + (conv_stage) * 8);
                }
                conv_stage += 1;
                if (conv_stage == 6) { conv_stage = 0; _phase_landed ^= 1; }
            }
            asm volatile("barrier.cluster.arrive.release;" ::: "memory");
            asm volatile("barrier.cluster.wait.acquire;" ::: "memory");
        }
    }
    // ---- Role: mma ----
    if (warp == 12) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 32;");
        // Deferred post-initialization cluster wait (WarpConfig.cluster_init_wait_warps)
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        { // mma_main
            asm volatile("barrier.sync 2, 32;" ::: "memory");
            unsigned int actual_q_2 = q_seq_len;
            unsigned int request_offset_2 = q_seq_len * (unsigned int)blockIdx.z;
            if ((unsigned long long)q_cu_seq_lens != 0) {
                request_offset_2 = q_cu_seq_lens[blockIdx.z];
                actual_q_2 = q_cu_seq_lens[blockIdx.z + 1] - request_offset_2;
            }
            unsigned int blocks_per_head_2 = (unsigned int)gridDim.y / num_kv_heads;
            unsigned int head_2 = (unsigned int)blockIdx.y / blocks_per_head_2;
            unsigned int row_begin_2 = (unsigned int)blockIdx.y % blocks_per_head_2 * 128;
            unsigned int output_col_2 = blockIdx.x * 256;
            int _vec_load_2[1];
            {
                _vec_load_2[0] = *reinterpret_cast<const int*>(kv_cache_list.sequence_lengths + blockIdx.z);
            }
            unsigned int length_2 = (unsigned int)_vec_load_2[0];
            unsigned int num_tiles_2 = (length_2 + 63) / 64;
            unsigned int mma_stage = 0;
            unsigned int qk_stage = 0;
            unsigned int pv_stage = 0;
            unsigned int _phase_s_empty = 1;
            unsigned int _phase_q_ready_0 = 0;
            unsigned int _phase_ready = 0;
            unsigned int _phase_q_ready_1 = 0;
            unsigned int _phase_q_ready_2 = 0;
            unsigned int _phase_q_ready_3 = 0;
            unsigned int _phase_p_full = 0;
            #pragma unroll 1
            for (unsigned int tile_2 = 0; tile_2 < num_tiles_2; tile_2++) {
                mbarrier_wait_hint(s_empty_addr + (qk_stage) * 8, _phase_s_empty, 10000000);
                if (tile_2 == 0) {
                    mbarrier_wait_hint(q_ready_addr, _phase_q_ready_0, 10000000);
                    _phase_q_ready_0 ^= 1;
                    mbarrier_wait_hint(ready_addr + (mma_stage) * 8, _phase_ready, 10000000);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (elect_sync()) {
                        int _mma_a_lo_0 = (_mma_base_lo_0) + (0) * 8192;
                        int _mma_b_lo_0 = (_mma_base_lo_1) + (mma_stage) * 1024;
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
                    "mov.b32 id, 135266320;\n\t"
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
                    "add.u32 alo, alo, 1018;\n\t"
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
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_s0), "r"(0));
                        tcgen05_commit_cg1_multicast(free_addr + (mma_stage) * 8, (uint16_t)(15));
                    }
                    mma_stage += 1;
                    if (mma_stage == 6) { mma_stage = 0; _phase_ready ^= 1; }
                    mbarrier_wait_hint(q_ready_addr + 8, _phase_q_ready_1, 10000000);
                    _phase_q_ready_1 ^= 1;
                    mbarrier_wait_hint(ready_addr + (mma_stage) * 8, _phase_ready, 10000000);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (elect_sync()) {
                        int _mma_a_lo_1 = (_mma_base_lo_2) + (0) * 8192;
                        int _mma_b_lo_1 = (_mma_base_lo_1) + (mma_stage) * 1024;
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
                    "mov.b32 id, 135266320;\n\t"
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
                    "add.u32 alo, alo, 1018;\n\t"
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
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"(tmem_s0), "r"(1));
                        tcgen05_commit_cg1_multicast(free_addr + (mma_stage) * 8, (uint16_t)(15));
                    }
                    mma_stage += 1;
                    if (mma_stage == 6) { mma_stage = 0; _phase_ready ^= 1; }
                    mbarrier_wait_hint(q_ready_addr + 16, _phase_q_ready_2, 10000000);
                    _phase_q_ready_2 ^= 1;
                    mbarrier_wait_hint(ready_addr + (mma_stage) * 8, _phase_ready, 10000000);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (elect_sync()) {
                        int _mma_a_lo_2 = (_mma_base_lo_3) + (0) * 8192;
                        int _mma_b_lo_2 = (_mma_base_lo_1) + (mma_stage) * 1024;
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
                    "mov.b32 id, 135266320;\n\t"
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
                    "add.u32 alo, alo, 1018;\n\t"
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
                    :: "r"(_mma_a_lo_2), "r"(_mma_b_lo_2), "r"(tmem_s0), "r"(1));
                        tcgen05_commit_cg1_multicast(free_addr + (mma_stage) * 8, (uint16_t)(15));
                    }
                    mma_stage += 1;
                    if (mma_stage == 6) { mma_stage = 0; _phase_ready ^= 1; }
                    mbarrier_wait_hint(q_ready_addr + 24, _phase_q_ready_3, 10000000);
                    _phase_q_ready_3 ^= 1;
                    mbarrier_wait_hint(ready_addr + (mma_stage) * 8, _phase_ready, 10000000);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (elect_sync()) {
                        int _mma_a_lo_3 = (_mma_base_lo_4) + (0) * 8192;
                        int _mma_b_lo_3 = (_mma_base_lo_1) + (mma_stage) * 1024;
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
                    "mov.b32 id, 135266320;\n\t"
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
                    "add.u32 alo, alo, 1018;\n\t"
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
                    :: "r"(_mma_a_lo_3), "r"(_mma_b_lo_3), "r"(tmem_s0), "r"(1));
                        tcgen05_commit_cg1_multicast(free_addr + (mma_stage) * 8, (uint16_t)(15));
                    }
                    mma_stage += 1;
                    if (mma_stage == 6) { mma_stage = 0; _phase_ready ^= 1; }
                } else if (qk_stage == 0) {
                    mbarrier_wait_hint(ready_addr + (mma_stage) * 8, _phase_ready, 10000000);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (elect_sync()) {
                        int _mma_a_lo_4 = (_mma_base_lo_0) + (0) * 8192;
                        int _mma_b_lo_4 = (_mma_base_lo_1) + (mma_stage) * 1024;
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
                    "mov.b32 id, 135266320;\n\t"
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
                    "add.u32 alo, alo, 1018;\n\t"
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
                    :: "r"(_mma_a_lo_4), "r"(_mma_b_lo_4), "r"(tmem_s0), "r"(0));
                        tcgen05_commit_cg1_multicast(free_addr + (mma_stage) * 8, (uint16_t)(15));
                    }
                    mma_stage += 1;
                    if (mma_stage == 6) { mma_stage = 0; _phase_ready ^= 1; }
                    mbarrier_wait_hint(ready_addr + (mma_stage) * 8, _phase_ready, 10000000);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (elect_sync()) {
                        int _mma_a_lo_5 = (_mma_base_lo_2) + (0) * 8192;
                        int _mma_b_lo_5 = (_mma_base_lo_1) + (mma_stage) * 1024;
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
                    "mov.b32 id, 135266320;\n\t"
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
                    "add.u32 alo, alo, 1018;\n\t"
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
                    :: "r"(_mma_a_lo_5), "r"(_mma_b_lo_5), "r"(tmem_s0), "r"(1));
                        tcgen05_commit_cg1_multicast(free_addr + (mma_stage) * 8, (uint16_t)(15));
                    }
                    mma_stage += 1;
                    if (mma_stage == 6) { mma_stage = 0; _phase_ready ^= 1; }
                    mbarrier_wait_hint(ready_addr + (mma_stage) * 8, _phase_ready, 10000000);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (elect_sync()) {
                        int _mma_a_lo_6 = (_mma_base_lo_3) + (0) * 8192;
                        int _mma_b_lo_6 = (_mma_base_lo_1) + (mma_stage) * 1024;
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
                    "mov.b32 id, 135266320;\n\t"
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
                    "add.u32 alo, alo, 1018;\n\t"
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
                    :: "r"(_mma_a_lo_6), "r"(_mma_b_lo_6), "r"(tmem_s0), "r"(1));
                        tcgen05_commit_cg1_multicast(free_addr + (mma_stage) * 8, (uint16_t)(15));
                    }
                    mma_stage += 1;
                    if (mma_stage == 6) { mma_stage = 0; _phase_ready ^= 1; }
                    mbarrier_wait_hint(ready_addr + (mma_stage) * 8, _phase_ready, 10000000);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (elect_sync()) {
                        int _mma_a_lo_7 = (_mma_base_lo_4) + (0) * 8192;
                        int _mma_b_lo_7 = (_mma_base_lo_1) + (mma_stage) * 1024;
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
                    "mov.b32 id, 135266320;\n\t"
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
                    "add.u32 alo, alo, 1018;\n\t"
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
                    :: "r"(_mma_a_lo_7), "r"(_mma_b_lo_7), "r"(tmem_s0), "r"(1));
                        tcgen05_commit_cg1_multicast(free_addr + (mma_stage) * 8, (uint16_t)(15));
                    }
                    mma_stage += 1;
                    if (mma_stage == 6) { mma_stage = 0; _phase_ready ^= 1; }
                } else {
                    mbarrier_wait_hint(ready_addr + (mma_stage) * 8, _phase_ready, 10000000);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (elect_sync()) {
                        int _mma_a_lo_8 = (_mma_base_lo_0) + (0) * 8192;
                        int _mma_b_lo_8 = (_mma_base_lo_1) + (mma_stage) * 1024;
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
                    "mov.b32 id, 135266320;\n\t"
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
                    "add.u32 alo, alo, 1018;\n\t"
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
                    :: "r"(_mma_a_lo_8), "r"(_mma_b_lo_8), "r"(tmem_s1), "r"(0));
                        tcgen05_commit_cg1_multicast(free_addr + (mma_stage) * 8, (uint16_t)(15));
                    }
                    mma_stage += 1;
                    if (mma_stage == 6) { mma_stage = 0; _phase_ready ^= 1; }
                    mbarrier_wait_hint(ready_addr + (mma_stage) * 8, _phase_ready, 10000000);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (elect_sync()) {
                        int _mma_a_lo_9 = (_mma_base_lo_2) + (0) * 8192;
                        int _mma_b_lo_9 = (_mma_base_lo_1) + (mma_stage) * 1024;
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
                    "mov.b32 id, 135266320;\n\t"
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
                    "add.u32 alo, alo, 1018;\n\t"
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
                    :: "r"(_mma_a_lo_9), "r"(_mma_b_lo_9), "r"(tmem_s1), "r"(1));
                        tcgen05_commit_cg1_multicast(free_addr + (mma_stage) * 8, (uint16_t)(15));
                    }
                    mma_stage += 1;
                    if (mma_stage == 6) { mma_stage = 0; _phase_ready ^= 1; }
                    mbarrier_wait_hint(ready_addr + (mma_stage) * 8, _phase_ready, 10000000);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (elect_sync()) {
                        int _mma_a_lo_10 = (_mma_base_lo_3) + (0) * 8192;
                        int _mma_b_lo_10 = (_mma_base_lo_1) + (mma_stage) * 1024;
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
                    "mov.b32 id, 135266320;\n\t"
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
                    "add.u32 alo, alo, 1018;\n\t"
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
                    :: "r"(_mma_a_lo_10), "r"(_mma_b_lo_10), "r"(tmem_s1), "r"(1));
                        tcgen05_commit_cg1_multicast(free_addr + (mma_stage) * 8, (uint16_t)(15));
                    }
                    mma_stage += 1;
                    if (mma_stage == 6) { mma_stage = 0; _phase_ready ^= 1; }
                    mbarrier_wait_hint(ready_addr + (mma_stage) * 8, _phase_ready, 10000000);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (elect_sync()) {
                        int _mma_a_lo_11 = (_mma_base_lo_4) + (0) * 8192;
                        int _mma_b_lo_11 = (_mma_base_lo_1) + (mma_stage) * 1024;
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
                    "mov.b32 id, 135266320;\n\t"
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
                    "add.u32 alo, alo, 1018;\n\t"
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
                    :: "r"(_mma_a_lo_11), "r"(_mma_b_lo_11), "r"(tmem_s1), "r"(1));
                        tcgen05_commit_cg1_multicast(free_addr + (mma_stage) * 8, (uint16_t)(15));
                    }
                    mma_stage += 1;
                    if (mma_stage == 6) { mma_stage = 0; _phase_ready ^= 1; }
                }
                if (elect_sync()) {
                    tcgen05_commit(s_full_addr + (qk_stage) * 8);
                }
                qk_stage += 1;
                if (qk_stage == 2) { qk_stage = 0; _phase_s_empty ^= 1; }
                if (tile_2 > 0) {
                    mbarrier_wait_hint(p_full_addr + (pv_stage) * 8, _phase_p_full, 10000000);
                    const bool _selected_wait_cond_0 = pv_stage == 0;
                    mbarrier_wait((_selected_wait_cond_0 ? (ready_addr + (mma_stage) * 8) : (ready_addr + (mma_stage) * 8)), (_selected_wait_cond_0 ? (_phase_ready) : (_phase_ready)));
                    if (_selected_wait_cond_0) {
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        if (elect_sync()) {
                            int _mma_b_lo_12 = (_mma_base_lo_5) + (mma_stage) * 1024;
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
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o0), "r"(_mma_b_lo_12), "r"(tmem_p0), "r"(((tile_2 - 1 == 0) ? 0 : 1)));
                            int _mma_b_lo_13 = (_mma_base_lo_6) + (mma_stage) * 1024;
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
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o0), "r"(_mma_b_lo_13), "r"(tmem_p0 + 8), "r"(1));
                            int _mma_b_lo_14 = (_mma_base_lo_7) + (mma_stage) * 1024;
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
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o0), "r"(_mma_b_lo_14), "r"(tmem_p0 + 16), "r"(1));
                            int _mma_b_lo_15 = (_mma_base_lo_8) + (mma_stage) * 1024;
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
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o0), "r"(_mma_b_lo_15), "r"(tmem_p0 + 24), "r"(1));
                            tcgen05_commit_cg1_multicast(free_addr + (mma_stage) * 8, (uint16_t)(15));
                        }
                        mma_stage += 1;
                        if (mma_stage == 6) { mma_stage = 0; _phase_ready ^= 1; }
                        mbarrier_wait_hint(ready_addr + (mma_stage) * 8, _phase_ready, 10000000);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        if (elect_sync()) {
                            int _mma_b_lo_16 = (_mma_base_lo_5) + (mma_stage) * 1024;
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
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o1), "r"(_mma_b_lo_16), "r"(tmem_p0), "r"(((tile_2 - 1 == 0) ? 0 : 1)));
                            int _mma_b_lo_17 = (_mma_base_lo_6) + (mma_stage) * 1024;
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
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o1), "r"(_mma_b_lo_17), "r"(tmem_p0 + 8), "r"(1));
                            int _mma_b_lo_18 = (_mma_base_lo_7) + (mma_stage) * 1024;
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
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o1), "r"(_mma_b_lo_18), "r"(tmem_p0 + 16), "r"(1));
                            int _mma_b_lo_19 = (_mma_base_lo_8) + (mma_stage) * 1024;
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
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o1), "r"(_mma_b_lo_19), "r"(tmem_p0 + 24), "r"(1));
                            tcgen05_commit_cg1_multicast(free_addr + (mma_stage) * 8, (uint16_t)(15));
                        }
                        mma_stage += 1;
                        if (mma_stage == 6) { mma_stage = 0; _phase_ready ^= 1; }
                    } else {
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        if (elect_sync()) {
                            int _mma_b_lo_20 = (_mma_base_lo_5) + (mma_stage) * 1024;
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
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o0), "r"(_mma_b_lo_20), "r"(tmem_p1), "r"(((tile_2 - 1 == 0) ? 0 : 1)));
                            int _mma_b_lo_21 = (_mma_base_lo_6) + (mma_stage) * 1024;
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
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o0), "r"(_mma_b_lo_21), "r"(tmem_p1 + 8), "r"(1));
                            int _mma_b_lo_22 = (_mma_base_lo_7) + (mma_stage) * 1024;
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
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o0), "r"(_mma_b_lo_22), "r"(tmem_p1 + 16), "r"(1));
                            int _mma_b_lo_23 = (_mma_base_lo_8) + (mma_stage) * 1024;
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
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o0), "r"(_mma_b_lo_23), "r"(tmem_p1 + 24), "r"(1));
                            tcgen05_commit_cg1_multicast(free_addr + (mma_stage) * 8, (uint16_t)(15));
                        }
                        mma_stage += 1;
                        if (mma_stage == 6) { mma_stage = 0; _phase_ready ^= 1; }
                        mbarrier_wait_hint(ready_addr + (mma_stage) * 8, _phase_ready, 10000000);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        if (elect_sync()) {
                            int _mma_b_lo_24 = (_mma_base_lo_5) + (mma_stage) * 1024;
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
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o1), "r"(_mma_b_lo_24), "r"(tmem_p1), "r"(((tile_2 - 1 == 0) ? 0 : 1)));
                            int _mma_b_lo_25 = (_mma_base_lo_6) + (mma_stage) * 1024;
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
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o1), "r"(_mma_b_lo_25), "r"(tmem_p1 + 8), "r"(1));
                            int _mma_b_lo_26 = (_mma_base_lo_7) + (mma_stage) * 1024;
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
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o1), "r"(_mma_b_lo_26), "r"(tmem_p1 + 16), "r"(1));
                            int _mma_b_lo_27 = (_mma_base_lo_8) + (mma_stage) * 1024;
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
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o1), "r"(_mma_b_lo_27), "r"(tmem_p1 + 24), "r"(1));
                            tcgen05_commit_cg1_multicast(free_addr + (mma_stage) * 8, (uint16_t)(15));
                        }
                        mma_stage += 1;
                        if (mma_stage == 6) { mma_stage = 0; _phase_ready ^= 1; }
                    }
                    if (elect_sync()) {
                        tcgen05_commit(p_empty_addr + (pv_stage) * 8);
                    }
                    pv_stage += 1;
                    if (pv_stage == 2) { pv_stage = 0; _phase_p_full ^= 1; }
                }
            }
            mbarrier_wait_hint(p_full_addr + (pv_stage) * 8, _phase_p_full, 10000000);
            const bool _selected_wait_cond_1 = pv_stage == 0;
            mbarrier_wait((_selected_wait_cond_1 ? (ready_addr + (mma_stage) * 8) : (ready_addr + (mma_stage) * 8)), (_selected_wait_cond_1 ? (_phase_ready) : (_phase_ready)));
            if (_selected_wait_cond_1) {
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (elect_sync()) {
                    int _mma_b_lo_28 = (_mma_base_lo_5) + (mma_stage) * 1024;
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
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o0), "r"(_mma_b_lo_28), "r"(tmem_p0), "r"(((num_tiles_2 - 1 == 0) ? 0 : 1)));
                    int _mma_b_lo_29 = (_mma_base_lo_6) + (mma_stage) * 1024;
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
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o0), "r"(_mma_b_lo_29), "r"(tmem_p0 + 8), "r"(1));
                    int _mma_b_lo_30 = (_mma_base_lo_7) + (mma_stage) * 1024;
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
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o0), "r"(_mma_b_lo_30), "r"(tmem_p0 + 16), "r"(1));
                    int _mma_b_lo_31 = (_mma_base_lo_8) + (mma_stage) * 1024;
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
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o0), "r"(_mma_b_lo_31), "r"(tmem_p0 + 24), "r"(1));
                    tcgen05_commit_cg1_multicast(free_addr + (mma_stage) * 8, (uint16_t)(15));
                }
                mma_stage += 1;
                if (mma_stage == 6) { mma_stage = 0; _phase_ready ^= 1; }
                mbarrier_wait_hint(ready_addr + (mma_stage) * 8, _phase_ready, 10000000);
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (elect_sync()) {
                    int _mma_b_lo_32 = (_mma_base_lo_5) + (mma_stage) * 1024;
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
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o1), "r"(_mma_b_lo_32), "r"(tmem_p0), "r"(((num_tiles_2 - 1 == 0) ? 0 : 1)));
                    int _mma_b_lo_33 = (_mma_base_lo_6) + (mma_stage) * 1024;
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
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o1), "r"(_mma_b_lo_33), "r"(tmem_p0 + 8), "r"(1));
                    int _mma_b_lo_34 = (_mma_base_lo_7) + (mma_stage) * 1024;
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
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o1), "r"(_mma_b_lo_34), "r"(tmem_p0 + 16), "r"(1));
                    int _mma_b_lo_35 = (_mma_base_lo_8) + (mma_stage) * 1024;
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
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o1), "r"(_mma_b_lo_35), "r"(tmem_p0 + 24), "r"(1));
                    tcgen05_commit_cg1_multicast(free_addr + (mma_stage) * 8, (uint16_t)(15));
                }
                mma_stage += 1;
                if (mma_stage == 6) { mma_stage = 0; _phase_ready ^= 1; }
            } else {
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (elect_sync()) {
                    int _mma_b_lo_36 = (_mma_base_lo_5) + (mma_stage) * 1024;
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
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o0), "r"(_mma_b_lo_36), "r"(tmem_p1), "r"(((num_tiles_2 - 1 == 0) ? 0 : 1)));
                    int _mma_b_lo_37 = (_mma_base_lo_6) + (mma_stage) * 1024;
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
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o0), "r"(_mma_b_lo_37), "r"(tmem_p1 + 8), "r"(1));
                    int _mma_b_lo_38 = (_mma_base_lo_7) + (mma_stage) * 1024;
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
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o0), "r"(_mma_b_lo_38), "r"(tmem_p1 + 16), "r"(1));
                    int _mma_b_lo_39 = (_mma_base_lo_8) + (mma_stage) * 1024;
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
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o0), "r"(_mma_b_lo_39), "r"(tmem_p1 + 24), "r"(1));
                    tcgen05_commit_cg1_multicast(free_addr + (mma_stage) * 8, (uint16_t)(15));
                }
                mma_stage += 1;
                if (mma_stage == 6) { mma_stage = 0; _phase_ready ^= 1; }
                mbarrier_wait_hint(ready_addr + (mma_stage) * 8, _phase_ready, 10000000);
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (elect_sync()) {
                    int _mma_b_lo_40 = (_mma_base_lo_5) + (mma_stage) * 1024;
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
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o1), "r"(_mma_b_lo_40), "r"(tmem_p1), "r"(((num_tiles_2 - 1 == 0) ? 0 : 1)));
                    int _mma_b_lo_41 = (_mma_base_lo_6) + (mma_stage) * 1024;
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
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o1), "r"(_mma_b_lo_41), "r"(tmem_p1 + 8), "r"(1));
                    int _mma_b_lo_42 = (_mma_base_lo_7) + (mma_stage) * 1024;
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
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o1), "r"(_mma_b_lo_42), "r"(tmem_p1 + 16), "r"(1));
                    int _mma_b_lo_43 = (_mma_base_lo_8) + (mma_stage) * 1024;
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
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "}\n"
                    :: "r"(tmem_o1), "r"(_mma_b_lo_43), "r"(tmem_p1 + 24), "r"(1));
                    tcgen05_commit_cg1_multicast(free_addr + (mma_stage) * 8, (uint16_t)(15));
                }
                mma_stage += 1;
                if (mma_stage == 6) { mma_stage = 0; _phase_ready ^= 1; }
            }
            if (elect_sync()) {
                tcgen05_commit(p_empty_addr + (pv_stage) * 8);
            }
            pv_stage += 1;
            if (pv_stage == 2) { pv_stage = 0; _phase_p_full ^= 1; }
            if (elect_sync()) {
                tcgen05_commit(o_ready_addr);
            }
            unsigned int _phase_tmem_retire_0 = 0;
            mbarrier_wait_hint(tmem_retire_addr, _phase_tmem_retire_0, 10000000);
            _phase_tmem_retire_0 ^= 1;
            int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(512));
            asm volatile("barrier.cluster.arrive.release;" ::: "memory");
            asm volatile("barrier.cluster.wait.acquire;" ::: "memory");
        }
    }
    // ---- Role: load ----
    if (warp == 13) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 32;");
        // Deferred post-initialization cluster wait (WarpConfig.cluster_init_wait_warps)
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        { // load_main
            unsigned int actual_q_3 = q_seq_len;
            unsigned int request_offset_3 = q_seq_len * (unsigned int)blockIdx.z;
            if ((unsigned long long)q_cu_seq_lens != 0) {
                request_offset_3 = q_cu_seq_lens[blockIdx.z];
                actual_q_3 = q_cu_seq_lens[blockIdx.z + 1] - request_offset_3;
            }
            unsigned int blocks_per_head_3 = (unsigned int)gridDim.y / num_kv_heads;
            unsigned int head_3 = (unsigned int)blockIdx.y / blocks_per_head_3;
            unsigned int row_begin_3 = (unsigned int)blockIdx.y % blocks_per_head_3 * 128;
            unsigned int output_col_3 = blockIdx.x * 256;
            int _vec_load_0[1];
            {
                _vec_load_0[0] = *reinterpret_cast<const int*>(kv_cache_list.sequence_lengths + blockIdx.z);
            }
            unsigned int length_3 = (unsigned int)_vec_load_0[0];
            unsigned int num_tiles_3 = (length_3 + 63) / 64;
            unsigned int token0 = request_offset_3 + row_begin_3 / 2;
            unsigned int rank = (unsigned int)cta_rank;
            #pragma unroll
            for (int pair = 0; pair < 4; pair++) {
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(q_ready_addr + (pair) * 8, 32768);
                    if ((rank & 1) == (unsigned int)(pair & 1)) {
                        #pragma unroll
                        for (int slab = 2 * pair; slab < 2 * pair + 2; slab++) {
                            if (rank >> 1 == 0) {
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                    :: "r"(sq_addr + (unsigned int)(slab * 16384)), "l"((&Q)), "r"(slab * 64), "r"(head_3 * 2), "r"(token0),
                                       "r"(q_ready_addr + (pair) * 8), "h"((uint16_t)(3)) : "memory");
                            } else {
                                asm volatile(
                                    "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
                                    " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                    :: "r"(sq_addr + (unsigned int)(slab * 16384)), "l"((&Q)), "r"(slab * 64), "r"(head_3 * 2), "r"(token0),
                                       "r"(q_ready_addr + (pair) * 8), "h"((uint16_t)(12)) : "memory");
                            }
                        }
                    }
                }
            }
            unsigned int load_stage = 0;
            unsigned int _phase_free = 1;
            #pragma unroll 1
            for (unsigned int tile_3 = 0; tile_3 < num_tiles_3; tile_3++) {
                #pragma unroll
                for (int c_1 = 0; c_1 < 4; c_1++) {
                    unsigned int kv_index = (unsigned int)(blockIdx.z * 2) * num_kv_heads + head_3;
                    unsigned int token = tile_3 * 64;
                    mbarrier_wait_hint(free_addr + (load_stage) * 8, _phase_free, 10000000);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(landed_addr + (load_stage) * 8, 16384);
                        {
                            #pragma unroll
                            for (int h = 0; h < 2; h++) {
                                if (rank == (unsigned int)(2 * c_1 + h & 3)) {
                                    asm volatile(
                                        "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
                                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                        :: "r"(sk_addr + load_stage * 16384 + (unsigned int)(h * 8192)), "l"((&KV)), "r"(c_1 * 128 + h * 64), "r"(token), "r"(kv_index),
                                           "r"(landed_addr + (load_stage) * 8), "h"((uint16_t)(15)) : "memory");
                                }
                            }
                        }
                    }
                    load_stage += 1;
                    if (load_stage == 6) { load_stage = 0; _phase_free ^= 1; }
                }
                if (tile_3 > 0) {
                    #pragma unroll
                    for (int v_2 = 0; v_2 < 2; v_2++) {
                        unsigned int kv_index_1 = (unsigned int)(blockIdx.z * 2 + 1) * num_kv_heads + head_3;
                        unsigned int token_1 = (tile_3 - 1) * 64;
                        mbarrier_wait_hint(free_addr + (load_stage) * 8, _phase_free, 10000000);
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(landed_addr + (load_stage) * 8, 16384);
                            {
                                #pragma unroll
                                for (int h_1 = 0; h_1 < 2; h_1++) {
                                    if (rank >> 1 == (unsigned int)(v_2 + h_1 & 1)) {
                                        if ((rank & 1) == 0) {
                                            asm volatile(
                                                "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
                                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                :: "r"(sk_addr + load_stage * 16384 + (unsigned int)(h_1 * 8192)), "l"((&KV)), "r"(output_col_3 + (unsigned int)(v_2 * 128) + (unsigned int)(h_1 * 64)), "r"(token_1), "r"(kv_index_1),
                                                   "r"(landed_addr + (load_stage) * 8), "h"((uint16_t)(5)) : "memory");
                                        } else {
                                            asm volatile(
                                                "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
                                                " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                                :: "r"(sk_addr + load_stage * 16384 + (unsigned int)(h_1 * 8192)), "l"((&KV)), "r"(output_col_3 + (unsigned int)(v_2 * 128) + (unsigned int)(h_1 * 64)), "r"(token_1), "r"(kv_index_1),
                                                   "r"(landed_addr + (load_stage) * 8), "h"((uint16_t)(10)) : "memory");
                                        }
                                    }
                                }
                            }
                        }
                        load_stage += 1;
                        if (load_stage == 6) { load_stage = 0; _phase_free ^= 1; }
                    }
                }
            }
            #pragma unroll
            for (int v_3 = 0; v_3 < 2; v_3++) {
                unsigned int kv_index_2 = (unsigned int)(blockIdx.z * 2 + 1) * num_kv_heads + head_3;
                unsigned int token_2 = (num_tiles_3 - 1) * 64;
                mbarrier_wait_hint(free_addr + (load_stage) * 8, _phase_free, 10000000);
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(landed_addr + (load_stage) * 8, 16384);
                    {
                        #pragma unroll
                        for (int h_2 = 0; h_2 < 2; h_2++) {
                            if (rank >> 1 == (unsigned int)(v_3 + h_2 & 1)) {
                                if ((rank & 1) == 0) {
                                    asm volatile(
                                        "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
                                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                        :: "r"(sk_addr + load_stage * 16384 + (unsigned int)(h_2 * 8192)), "l"((&KV)), "r"(output_col_3 + (unsigned int)(v_3 * 128) + (unsigned int)(h_2 * 64)), "r"(token_2), "r"(kv_index_2),
                                           "r"(landed_addr + (load_stage) * 8), "h"((uint16_t)(5)) : "memory");
                                } else {
                                    asm volatile(
                                        "cp.async.bulk.tensor.3d.shared::cluster.global.tile.mbarrier::complete_tx::bytes.multicast::cluster"
                                        " [%0], [%1, {%2, %3, %4}], [%5], %6;"
                                        :: "r"(sk_addr + load_stage * 16384 + (unsigned int)(h_2 * 8192)), "l"((&KV)), "r"(output_col_3 + (unsigned int)(v_3 * 128) + (unsigned int)(h_2 * 64)), "r"(token_2), "r"(kv_index_2),
                                           "r"(landed_addr + (load_stage) * 8), "h"((uint16_t)(10)) : "memory");
                                }
                            }
                        }
                    }
                }
                load_stage += 1;
                if (load_stage == 6) { load_stage = 0; _phase_free ^= 1; }
            }
            asm volatile("barrier.cluster.arrive.release;" ::: "memory");
            asm volatile("barrier.cluster.wait.acquire;" ::: "memory");
        }
    }
    // ---- Role: idle ----
    if (warp >= 14 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 32;");
        // Deferred post-initialization cluster wait (WarpConfig.cluster_init_wait_warps)
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        { // idle_main
            __syncwarp();
            asm volatile("barrier.cluster.arrive.release;" ::: "memory");
            asm volatile("barrier.cluster.wait.acquire;" ::: "memory");
        }
    }

    // Cleanup
}

} // extern "C"
