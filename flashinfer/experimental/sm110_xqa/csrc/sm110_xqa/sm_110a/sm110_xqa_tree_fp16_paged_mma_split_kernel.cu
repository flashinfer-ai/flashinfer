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
#define NUM_MAIN_STAGES 1
#define SMEM_POOL_OFF 128
#define SMEM_POOL_STAGE_BYTES 221184
#define SMEM_POOL_STRIDE 221184
#define SMEM_STATS0_OFF 123008
#define SMEM_STATS0_STAGE_BYTES 4096
#define SMEM_STATS0_STRIDE 4096
#define SMEM_STATS1_OFF 217216
#define SMEM_STATS1_STAGE_BYTES 4096
#define SMEM_STATS1_STRIDE 4096
#define SMEM_MERGE_OFF 127104
#define SMEM_MERGE_STAGE_BYTES 65536
#define SMEM_MERGE_STRIDE 65536
#define SMEM_TOTAL 221312

#if !defined(__CUDACC_RTC__)
#include <stddef.h>
#endif
struct __align__(8) KVCacheList {
    void* pool;
    const int* page_list;
    const int* sequence_lengths;
    unsigned int max_pages;
};
static_assert(sizeof(KVCacheList) == 32, "KVCacheList size");
static_assert(__alignof__(KVCacheList) == 8, "KVCacheList alignment");
#if !defined(__CUDACC_RTC__)
static_assert(offsetof(KVCacheList, pool) == 0, "KVCacheList.pool offset");
static_assert(offsetof(KVCacheList, page_list) == 8, "KVCacheList.page_list offset");
static_assert(offsetof(KVCacheList, sequence_lengths) == 16, "KVCacheList.sequence_lengths offset");
static_assert(offsetof(KVCacheList, max_pages) == 24, "KVCacheList.max_pages offset");
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

extern "C" {

__global__ __launch_bounds__(512, 1) void
kernel_sm110_xqa_tree_fp16_paged_mma_split(unsigned int q_seq_len, unsigned int num_kv_heads, unsigned int head_group_size, const unsigned int* __restrict__ q_cu_seq_lens, float attention_scale, __half* __restrict__ output, const __half* __restrict__ q, const unsigned int* __restrict__ mask, const float* __restrict__ attention_sinks, KVCacheList kv_cache_list, unsigned int batch_size, float k_cache_scale, float v_cache_scale, unsigned int* __restrict__ semaphores, void* __restrict__ scratch)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(128) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define q_ready_addr (mbar_base + 0)
    #define x_produced0_addr (mbar_base + 8)
    #define x_consumed0_addr (mbar_base + 16)
    #define x_produced1_addr (mbar_base + 24)
    #define x_consumed1_addr (mbar_base + 32)
    #define merge_ready_addr (mbar_base + 40)
    #define q_reordered_addr (mbar_base + 48)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;

    // Kernel setup ops
    uint8_t* pool = reinterpret_cast<uint8_t*>(smem_raw + 128);
    const int pool_addr = smem + 128;
    float* stats0 = reinterpret_cast<float*>(smem_raw + 123008);
    const int stats0_addr = smem + 123008;
    float* stats1 = reinterpret_cast<float*>(smem_raw + 217216);
    const int stats1_addr = smem + 217216;
    float* merge = reinterpret_cast<float*>(smem_raw + 127104);
    const int merge_addr = smem + 127104;
    unsigned int actual_q = q_seq_len;
    unsigned int request_offset = q_seq_len * (unsigned int)blockIdx.z;
    if ((unsigned long long)q_cu_seq_lens != 0) {
        request_offset = q_cu_seq_lens[blockIdx.z];
        actual_q = q_cu_seq_lens[blockIdx.z + 1] - request_offset;
    }
    unsigned int q_heads = num_kv_heads * head_group_size;
    unsigned int blocks_per_group = (unsigned int)gridDim.y / num_kv_heads;
    unsigned int head_group = (unsigned int)blockIdx.y / blocks_per_group;
    unsigned int row_begin = (unsigned int)blockIdx.y % blocks_per_group * 32;
    unsigned int _min_0 = ((actual_q * head_group_size - row_begin) < ((unsigned int)32) ? (actual_q * head_group_size - row_begin) : ((unsigned int)32));
    unsigned int valid_rows = ((row_begin <= actual_q * head_group_size) ? _min_0 : (unsigned int)0);
    unsigned int mask_row_words = (q_seq_len + 31) / 32;
    unsigned int _vec_load_0[1];
    {
        uint32_t _scalar_bits_0;
        uint64_t _l2_evict_last_policy_1;
        asm("createpolicy.fractional.L2::evict_last.b64 %0;"
            : "=l"(_l2_evict_last_policy_1));
        asm("ld.global.nc.L1::evict_last.L2::cache_hint.L2::256B.b32 %0, [%1], %2;"
            : "=r"(_scalar_bits_0)
            : "l"((const void*)(kv_cache_list.sequence_lengths + (blockIdx.z))), "l"(_l2_evict_last_policy_1));
        _vec_load_0[0] = (unsigned int)_scalar_bits_0;
    }
    unsigned int length = _vec_load_0[0];
    unsigned int nblocks = (length + 16 - 1) / 16;
    unsigned int b_split = ((nblocks + 1) / 2 + 1) / 2 * 2;
    unsigned int passes0 = (b_split + 8 - 1) / 8;
    unsigned int passes1 = (nblocks - b_split + 8 - 1) / 8;
    unsigned int k_contig_base = (unsigned int)0;
    unsigned int v_contig_base = (unsigned int)0;
    float qk_scale = attention_scale;
    unsigned int sq = pool_addr;

    // Mbarrier init (7 pipeline groups, 0 ordered-sequence groups, 7 barriers)
    // Mbarriers at smem_raw[0..56)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_ready: 1 barriers, init_count=512
            mbarrier_init(smem + 0, 512);
            // x_produced0: 1 barriers, init_count=256
            mbarrier_init(smem + 8, 256);
            // x_consumed0: 1 barriers, init_count=256
            mbarrier_init(smem + 16, 256);
            // x_produced1: 1 barriers, init_count=256
            mbarrier_init(smem + 24, 256);
            // x_consumed1: 1 barriers, init_count=256
            mbarrier_init(smem + 32, 256);
            // merge_ready: 1 barriers, init_count=256
            mbarrier_init(smem + 40, 256);
            // q_reordered: 1 barriers, init_count=256
            mbarrier_init(smem + 48, 256);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // ---- Role: group0 ----
    if (warp <= 7) {
        { // group0_main
            int warp_id_in_role = (warp - 0);
            unsigned int warp_0 = warp_id_in_role;
            unsigned int row = (warp_0 * 32 + (unsigned int)lane) / 8;
            unsigned int line = (warp_0 * 32 + (unsigned int)lane) % 8;
            unsigned int head_token = row_begin + row;
            unsigned int source_head = (request_offset + head_token / head_group_size) * q_heads + head_group * head_group_size + head_token % head_group_size;
            if (row < valid_rows) { asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"((uint64_t)(q + (source_head * 512 + line * 64)))); }
            unsigned long long _mbarrier_arrival_token_0;
            asm volatile(
                "mbarrier.arrive.release.cta.b64 %0, [%1], %2;"
                : "=l"(_mbarrier_arrival_token_0)
                : "l"(smem_raw + ((x_consumed0_addr) - smem)), "r"((uint32_t)(1)) : "memory");
            if (valid_rows == 32) {
                #pragma unroll
                for (int copy = 0; copy < 4; copy++) {
                    unsigned int grain = (unsigned int)(copy * 512) + warp_0 * 32 + (unsigned int)lane;
                    unsigned int row_0 = grain / 64;
                    unsigned int col = grain % 64;
                    unsigned int head_token_1 = row_begin + row_0;
                    unsigned int source_head_2 = (request_offset + head_token_1 / head_group_size) * q_heads + head_group * head_group_size + head_token_1 % head_group_size;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(sq + row_0 * 1024 + (unsigned int)(col * 16 ^ (row_0 & 7) << 4)), "l"(q + (source_head_2 * 512 + col * 8)));
                }
            } else {
                #pragma unroll
                for (int copy_1 = 0; copy_1 < 4; copy_1++) {
                    unsigned int grain_1 = (unsigned int)(copy_1 * 512) + warp_0 * 32 + (unsigned int)lane;
                    unsigned int row_0_1 = grain_1 / 64;
                    unsigned int col_1 = grain_1 % 64;
                    unsigned int head_token_1_1 = row_begin + row_0_1;
                    unsigned int source_head_2_1 = (request_offset + head_token_1_1 / head_group_size) * q_heads + head_group * head_group_size + head_token_1_1 % head_group_size;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                        :: "r"(sq + row_0_1 * 1024 + (unsigned int)(col_1 * 16 ^ (row_0_1 & 7) << 4)), "l"(q + (source_head_2_1 * 512 + col_1 * 8)), "r"((row_0_1 < valid_rows) ? 16 : 0));
                }
            }
            asm volatile(
                "{\n\t"
                "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n\t"
                "}"
                :: "r"(q_ready_addr) : "memory");
            unsigned int info[3];
            unsigned int first_block = (unsigned int)0 + warp_0;
            unsigned int first_available = (unsigned int)0;
            if (first_block < b_split) {
                unsigned int slot = (unsigned int)(blockIdx.z * 2) * kv_cache_list.max_pages + first_block * 16 / 128;
                int _vec_load_1[1];
                {
                    _vec_load_1[0] = *reinterpret_cast<const int*>(kv_cache_list.page_list + slot);
                }
                int page = _vec_load_1[0];
                bool valid = page >= 0;
                info[0] = ((valid) ? ((unsigned int)page * 128 + first_block * 16 % 128) * num_kv_heads + head_group : (unsigned int)0);
                info[1] = num_kv_heads;
                info[2] = ((valid) ? (unsigned int)1 : (unsigned int)0);
                unsigned int _min_1 = ((length - first_block * 16) < ((unsigned int)16) ? (length - first_block * 16) : ((unsigned int)16));
                first_available = _min_1;
            }
            #pragma unroll
            for (int part = 0; part < 5; part++) {
                if (first_block < b_split) {
                    bool valid_1 = info[2] != 0;
                    #pragma unroll
                    for (int copy_2 = 0; copy_2 < 2; copy_2++) {
                        unsigned int row_0_2 = 8 * copy_2 + lane / 4;
                        unsigned int col_2 = lane % 4;
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"(pool_addr + 32768 + warp_0 * 6 * 1024 + (unsigned int)(part * 1024) + row_0_2 * 64 + (unsigned int)(col_2 * 16 ^ (row_0_2 >> 1 & 3) << 4)), "l"(reinterpret_cast<const __half*>(kv_cache_list.pool) + ((info[0] + row_0_2 * info[1]) * 512 + (unsigned int)(part * 32) + col_2 * 8)), "r"((valid_1 && row_0_2 < first_available) ? 16 : 0));
                    }
                }
                asm volatile("cp.async.commit_group;");
            }
            #pragma unroll
            for (int s = 0; s < 2; s++) {
                unsigned int slot_block = (unsigned int)0 + (unsigned int)s;
                if (slot_block < b_split) {
                    unsigned int slot_1 = (unsigned int)(blockIdx.z * 2 + 1) * kv_cache_list.max_pages + slot_block * 16 / 128;
                    int _vec_load_2[1];
                    {
                        _vec_load_2[0] = *reinterpret_cast<const int*>(kv_cache_list.page_list + slot_1);
                    }
                    int page_1 = _vec_load_2[0];
                    bool valid_2 = page_1 >= 0;
                    info[0] = ((valid_2) ? ((unsigned int)page_1 * 128 + slot_block * 16 % 128) * num_kv_heads + head_group : (unsigned int)0);
                    info[1] = num_kv_heads;
                    info[2] = ((valid_2) ? (unsigned int)1 : (unsigned int)0);
                    unsigned int _min_2 = ((length - slot_block * 16) < ((unsigned int)16) ? (length - slot_block * 16) : ((unsigned int)16));
                    unsigned int slot_available = _min_2;
                    bool valid_0 = info[2] != 0;
                    #pragma unroll
                    for (int copy_3 = 0; copy_3 < 4; copy_3++) {
                        unsigned int row16 = 4 * copy_3 + lane / 8;
                        unsigned int col_3 = lane % 8;
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"(pool_addr + 90112 + warp_0 * 4096 + (unsigned int)(s * 2048) + row16 * 128 + (unsigned int)(col_3 * 16 ^ (row16 & 7) << 4)), "l"(reinterpret_cast<const __half*>(kv_cache_list.pool) + ((info[0] + row16 * info[1]) * 512 + warp_0 * 64 + col_3 * 8)), "r"((valid_0 && row16 < slot_available) ? 16 : 0));
                    }
                }
                asm volatile("cp.async.commit_group;");
            }
            unsigned int slice_rows = (length + blocks_per_group - 1) / blocks_per_group;
            unsigned int slice_begin = (unsigned int)blockIdx.y % blocks_per_group * slice_rows;
            unsigned int _min_3 = ((slice_begin + slice_rows) < (length) ? (slice_begin + slice_rows) : (length));
            unsigned int slice_end = ((slice_begin < length) ? _min_3 : slice_begin);
            unsigned int slice_lines = (slice_end - slice_begin) * (unsigned int)(512 * ((0) ? 1 : 2) / 128);
            for (unsigned int it = 0; it < (slice_lines + 255) / 256; it++) {
                unsigned int index = it * 256 + (warp_0 * 32 + (unsigned int)lane);
                unsigned int row_0_3 = slice_begin + index / (unsigned int)(512 * ((0) ? 1 : 2) / 128);
                unsigned int line_1 = index % (unsigned int)(512 * ((0) ? 1 : 2) / 128);
                bool valid_3 = index < slice_lines;
                unsigned int head_row = k_contig_base + row_0_3;
                unsigned int page_slot = (unsigned int)(blockIdx.z * 2) * kv_cache_list.max_pages + row_0_3 / 128;
                int _vec_load_3[1];
                {
                    _vec_load_3[0] = *reinterpret_cast<const int*>(kv_cache_list.page_list + ((valid_3) ? page_slot : (unsigned int)0));
                }
                int page_id = _vec_load_3[0];
                valid_3 = valid_3 && page_id >= 0;
                head_row = ((valid_3) ? ((unsigned int)page_id * 128 + row_0_3 % 128) * num_kv_heads + head_group : (unsigned int)0);
                if (valid_3) { asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"((uint64_t)(reinterpret_cast<const __half*>(kv_cache_list.pool) + (head_row * 512 + line_1 * (unsigned int)(128 / ((0) ? 1 : 2)))))); }
            }
            mbarrier_wait_hint(q_ready_addr, 0, 4294967295u);
            float acc[64];
            acc[0] = 0.0f;
            acc[1] = 0.0f;
            acc[2] = 0.0f;
            acc[3] = 0.0f;
            acc[4] = 0.0f;
            acc[5] = 0.0f;
            acc[6] = 0.0f;
            acc[7] = 0.0f;
            acc[8] = 0.0f;
            acc[9] = 0.0f;
            acc[10] = 0.0f;
            acc[11] = 0.0f;
            acc[12] = 0.0f;
            acc[13] = 0.0f;
            acc[14] = 0.0f;
            acc[15] = 0.0f;
            acc[16] = 0.0f;
            acc[17] = 0.0f;
            acc[18] = 0.0f;
            acc[19] = 0.0f;
            acc[20] = 0.0f;
            acc[21] = 0.0f;
            acc[22] = 0.0f;
            acc[23] = 0.0f;
            acc[24] = 0.0f;
            acc[25] = 0.0f;
            acc[26] = 0.0f;
            acc[27] = 0.0f;
            acc[28] = 0.0f;
            acc[29] = 0.0f;
            acc[30] = 0.0f;
            acc[31] = 0.0f;
            acc[32] = 0.0f;
            acc[33] = 0.0f;
            acc[34] = 0.0f;
            acc[35] = 0.0f;
            acc[36] = 0.0f;
            acc[37] = 0.0f;
            acc[38] = 0.0f;
            acc[39] = 0.0f;
            acc[40] = 0.0f;
            acc[41] = 0.0f;
            acc[42] = 0.0f;
            acc[43] = 0.0f;
            acc[44] = 0.0f;
            acc[45] = 0.0f;
            acc[46] = 0.0f;
            acc[47] = 0.0f;
            acc[48] = 0.0f;
            acc[49] = 0.0f;
            acc[50] = 0.0f;
            acc[51] = 0.0f;
            acc[52] = 0.0f;
            acc[53] = 0.0f;
            acc[54] = 0.0f;
            acc[55] = 0.0f;
            acc[56] = 0.0f;
            acc[57] = 0.0f;
            acc[58] = 0.0f;
            acc[59] = 0.0f;
            acc[60] = 0.0f;
            acc[61] = 0.0f;
            acc[62] = 0.0f;
            acc[63] = 0.0f;
            float m_run[4];
            float l_run[4];
            #pragma unroll
            for (int r = 0; r < 4; r++) {
                m_run[r] = -1e+30f;
                l_run[r] = 0.0f;
            }
            unsigned int cslot = (unsigned int)0;
            unsigned int islot = (unsigned int)5;
            for (unsigned int p = 0; p < passes0; p++) {
                unsigned int block = (unsigned int)0 + 8 * p + warp_0;
                if (block < b_split) {
                    float s_acc[16];
                    s_acc[0] = 0.0f;
                    s_acc[1] = 0.0f;
                    s_acc[2] = 0.0f;
                    s_acc[3] = 0.0f;
                    s_acc[4] = 0.0f;
                    s_acc[5] = 0.0f;
                    s_acc[6] = 0.0f;
                    s_acc[7] = 0.0f;
                    s_acc[8] = 0.0f;
                    s_acc[9] = 0.0f;
                    s_acc[10] = 0.0f;
                    s_acc[11] = 0.0f;
                    s_acc[12] = 0.0f;
                    s_acc[13] = 0.0f;
                    s_acc[14] = 0.0f;
                    s_acc[15] = 0.0f;
                    unsigned int token0 = block * 16;
                    unsigned int _min_4 = ((length - token0) < ((unsigned int)16) ? (length - token0) : ((unsigned int)16));
                    unsigned int available = _min_4;
                    unsigned int slot_2 = (unsigned int)(blockIdx.z * 2) * kv_cache_list.max_pages + token0 / 128;
                    int _vec_load_4[1];
                    {
                        _vec_load_4[0] = *reinterpret_cast<const int*>(kv_cache_list.page_list + slot_2);
                    }
                    int page_2 = _vec_load_4[0];
                    bool valid_4 = page_2 >= 0;
                    info[0] = ((valid_4) ? ((unsigned int)page_2 * 128 + token0 % 128) * num_kv_heads + head_group : (unsigned int)0);
                    info[1] = num_kv_heads;
                    info[2] = ((valid_4) ? (unsigned int)1 : (unsigned int)0);
                    unsigned int next_info[3];
                    unsigned int next_block = block + 8;
                    unsigned int next_available = (unsigned int)0;
                    if (next_block < b_split) {
                        unsigned int slot_0 = (unsigned int)(blockIdx.z * 2) * kv_cache_list.max_pages + next_block * 16 / 128;
                        int _vec_load_5[1];
                        {
                            _vec_load_5[0] = *reinterpret_cast<const int*>(kv_cache_list.page_list + slot_0);
                        }
                        int page_1_1 = _vec_load_5[0];
                        bool valid_2_1 = page_1_1 >= 0;
                        next_info[0] = ((valid_2_1) ? ((unsigned int)page_1_1 * 128 + next_block * 16 % 128) * num_kv_heads + head_group : (unsigned int)0);
                        next_info[1] = num_kv_heads;
                        next_info[2] = ((valid_2_1) ? (unsigned int)1 : (unsigned int)0);
                        unsigned int _min_5 = ((length - next_block * 16) < ((unsigned int)16) ? (length - next_block * 16) : ((unsigned int)16));
                        next_available = _min_5;
                    }
                    #pragma unroll 1
                    for (int part_1 = 0; part_1 < 16; part_1++) {
                        if (part_1 + 6 - 1 < 16) {
                            bool valid_0_1 = info[2] != 0;
                            #pragma unroll
                            for (int copy_4 = 0; copy_4 < 2; copy_4++) {
                                unsigned int row_0_4 = 8 * copy_4 + lane / 4;
                                unsigned int col_4 = lane % 4;
                                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                                    :: "r"(pool_addr + 32768 + warp_0 * 6 * 1024 + islot * 1024 + row_0_4 * 64 + (unsigned int)(col_4 * 16 ^ (row_0_4 >> 1 & 3) << 4)), "l"(reinterpret_cast<const __half*>(kv_cache_list.pool) + ((info[0] + row_0_4 * info[1]) * 512 + (unsigned int)((part_1 + 6 - 1) * 32) + col_4 * 8)), "r"((valid_0_1 && row_0_4 < available) ? 16 : 0));
                            }
                        } else if (next_block < b_split) {
                            bool valid_0_2 = next_info[2] != 0;
                            #pragma unroll
                            for (int copy_5 = 0; copy_5 < 2; copy_5++) {
                                unsigned int row_0_5 = 8 * copy_5 + lane / 4;
                                unsigned int col_5 = lane % 4;
                                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                                    :: "r"(pool_addr + 32768 + warp_0 * 6 * 1024 + islot * 1024 + row_0_5 * 64 + (unsigned int)(col_5 * 16 ^ (row_0_5 >> 1 & 3) << 4)), "l"(reinterpret_cast<const __half*>(kv_cache_list.pool) + ((next_info[0] + row_0_5 * next_info[1]) * 512 + (unsigned int)((part_1 + 6 - 1 - 16) * 32) + col_5 * 8)), "r"((valid_0_2 && row_0_5 < next_available) ? 16 : 0));
                            }
                        }
                        asm volatile("cp.async.commit_group;");
                        asm volatile("cp.async.wait_group 5;");
                        #pragma unroll
                        for (int split = 0; split < 2; split++) {
                            unsigned int qa16[8];
                            unsigned int kb16[4];
                            #pragma unroll
                            for (int m = 0; m < 2; m++) {
                                unsigned int qrow16 = m * 16 + lane % 16;
                                unsigned int qgrain16 = part_1 * 4 + split * 2 + lane / 16;
                                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                    : "=r"(qa16[m * 4]), "=r"(qa16[m * 4 + 1]), "=r"(qa16[m * 4 + 2]), "=r"(qa16[m * 4 + 3])
                                    : "r"(sq + qrow16 * 1024 + (unsigned int)(qgrain16 * 16 ^ (qrow16 & 7) << 4))
                                    : "memory");
                            }
                            unsigned int krow16 = lane % 8 + lane / 16 * 8;
                            unsigned int kgrain16 = split * 2 + lane / 8 % 2;
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(kb16[0]), "=r"(kb16[1]), "=r"(kb16[2]), "=r"(kb16[3])
                                : "r"(pool_addr + 32768 + warp_0 * 6 * 1024 + cslot * 1024 + krow16 * 64 + (unsigned int)(kgrain16 * 16 ^ (krow16 >> 1 & 3) << 4))
                                : "memory");
                            #pragma unroll
                            for (int m_1 = 0; m_1 < 2; m_1++) {
                                #pragma unroll
                                for (int n = 0; n < 2; n++) {
                                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                        : "+f"((s_acc + (m_1 * 2 + n) * 4)[0]), "+f"((s_acc + (m_1 * 2 + n) * 4)[1]), "+f"((s_acc + (m_1 * 2 + n) * 4)[2]), "+f"((s_acc + (m_1 * 2 + n) * 4)[3])
                                        : "r"((qa16 + m_1 * 4)[0]), "r"((qa16 + m_1 * 4)[1]), "r"((qa16 + m_1 * 4)[2]), "r"((qa16 + m_1 * 4)[3]), "r"((kb16 + n * 2)[0]), "r"((kb16 + n * 2)[1]));
                                }
                            }
                        }
                        cslot = (cslot + 1) % 6;
                        islot = (islot + 1) % 6;
                    }
                    const float2 _scale2_1 = {qk_scale, qk_scale};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(s_acc)[_ls], _scale2_1);
                    if (token0 + 16 > length - actual_q) {
                        unsigned int prefix = length - actual_q;
                        #pragma unroll
                        for (int m_2 = 0; m_2 < 2; m_2++) {
                            #pragma unroll
                            for (int i = 0; i < 2; i++) {
                                unsigned int _min_6 = (((row_begin + (unsigned int)(m_2 * 16) + (unsigned int)(lane / 4) + (unsigned int)(i * 8)) / head_group_size) < (actual_q - 1) ? ((row_begin + (unsigned int)(m_2 * 16) + (unsigned int)(lane / 4) + (unsigned int)(i * 8)) / head_group_size) : (actual_q - 1));
                                unsigned int token_row = _min_6;
                                unsigned int row_words = (request_offset + token_row) * mask_row_words;
                                #pragma unroll
                                for (int n_1 = 0; n_1 < 2; n_1++) {
                                    #pragma unroll
                                    for (int j = 0; j < 2; j++) {
                                        unsigned int token = token0 + (unsigned int)(n_1 * 8) + (unsigned int)(lane % 4 * 2) + (unsigned int)j;
                                        unsigned int _min_7 = ((token - prefix) < (actual_q - 1) ? (token - prefix) : (actual_q - 1));
                                        unsigned int pos = ((token >= prefix) ? _min_7 : (unsigned int)0);
                                        unsigned int _vec_load_6[1];
                                        {
                                            _vec_load_6[0] = *reinterpret_cast<const unsigned int*>(mask + (row_words + pos / 32));
                                        }
                                        unsigned int word = _vec_load_6[0];
                                        bool visible = token < prefix || (word & (unsigned int)1 << pos % 32) != 0;
                                        s_acc[(m_2 * 2 + n_1) * 4 + (i * 2 + j)] = ((visible && token < length) ? s_acc[(m_2 * 2 + n_1) * 4 + (i * 2 + j)] : -SM110_XQA_INF);
                                    }
                                }
                            }
                        }
                    }
                    float row_max[4];
                    float row_sum[4];
                    #pragma unroll
                    for (int r_1 = 0; r_1 < 4; r_1++) {
                        row_max[r_1] = -1e+30f;
                        row_sum[r_1] = 0.0f;
                    }
                    #pragma unroll
                    for (int m_3 = 0; m_3 < 2; m_3++) {
                        #pragma unroll
                        for (int i_1 = 0; i_1 < 2; i_1++) {
                            #pragma unroll
                            for (int n_2 = 0; n_2 < 2; n_2++) {
                                #pragma unroll
                                for (int j_1 = 0; j_1 < 2; j_1++) {
                                    float _fmax_0 = fmaxf(row_max[m_3 * 2 + i_1], s_acc[(m_3 * 2 + n_2) * 4 + (i_1 * 2 + j_1)]);
                                    row_max[m_3 * 2 + i_1] = _fmax_0;
                                }
                            }
                        }
                    }
                    #pragma unroll
                    for (int r_2 = 0; r_2 < 4; r_2++) {
                        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, row_max[r_2], 2);
                        float _fmax_1 = fmaxf(row_max[r_2], _shfl_xor_0);
                        row_max[r_2] = _fmax_1;
                    }
                    #pragma unroll
                    for (int r_3 = 0; r_3 < 4; r_3++) {
                        float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, row_max[r_3], 1);
                        float _fmax_2 = fmaxf(row_max[r_3], _shfl_xor_1);
                        row_max[r_3] = _fmax_2;
                    }
                    #pragma unroll
                    for (int m_4 = 0; m_4 < 2; m_4++) {
                        #pragma unroll
                        for (int i_2 = 0; i_2 < 2; i_2++) {
                            float bias = row_max[m_4 * 2 + i_2] * 1.4426950408889634f;
                            #pragma unroll
                            for (int n_3 = 0; n_3 < 2; n_3++) {
                                #pragma unroll
                                for (int j_2 = 0; j_2 < 2; j_2++) {
                                    float _exp2_0 = approx_exp2(s_acc[(m_4 * 2 + n_3) * 4 + (i_2 * 2 + j_2)] * 1.4426950408889634f - bias);
                                    s_acc[(m_4 * 2 + n_3) * 4 + (i_2 * 2 + j_2)] = _exp2_0;
                                    row_sum[m_4 * 2 + i_2] = row_sum[m_4 * 2 + i_2] + s_acc[(m_4 * 2 + n_3) * 4 + (i_2 * 2 + j_2)];
                                }
                            }
                        }
                    }
                    #pragma unroll
                    for (int r_4 = 0; r_4 < 4; r_4++) {
                        float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, row_sum[r_4], 2);
                        row_sum[r_4] = row_sum[r_4] + _shfl_xor_2;
                    }
                    #pragma unroll
                    for (int r_5 = 0; r_5 < 4; r_5++) {
                        float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, row_sum[r_5], 1);
                        row_sum[r_5] = row_sum[r_5] + _shfl_xor_3;
                    }
                    mbarrier_wait_hint(x_consumed0_addr, p % 2, 4294967295u);
                    unsigned int packed[8];
                    #pragma unroll
                    for (int m_5 = 0; m_5 < 2; m_5++) {
                        #pragma unroll
                        for (int i_3 = 0; i_3 < 2; i_3++) {
                            #pragma unroll
                            for (int n_4 = 0; n_4 < 2; n_4++) {
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __half2 _h2 = __float22half2_rn(make_float2((s_acc + (m_5 * 2 + n_4) * 4 + i_3 * 2)[_lp*2 + 0], (s_acc + (m_5 * 2 + n_4) * 4 + i_3 * 2)[_lp*2+1 + 0]));
                                    packed[((m_5 * 2 + i_3) * 2 + n_4) + _lp] = *(uint32_t*)&_h2;
                                }
                            }
                        }
                    }
                    #pragma unroll
                    for (int m_6 = 0; m_6 < 2; m_6++) {
                        unsigned int row_0_6 = m_6 * 16 + lane / 16 * 8 + lane % 8;
                        unsigned int grain_2 = lane / 8 % 2;
                        uint32_t _stmatrix_addr_2 = static_cast<uint32_t>(pool_addr + 81920 + warp_0 * 1024 + row_0_6 * 32 + (unsigned int)(grain_2 * 16 ^ (row_0_6 >> 2 & 1) << 4));
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_2), "r"(*reinterpret_cast<const uint32_t*>(&packed[m_6 * 2 * 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed[m_6 * 2 * 2 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed[(m_6 * 2 + 1) * 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed[(m_6 * 2 + 1) * 2 + 1]))
                            : "memory");
                    }
                    if (lane % 4 == 0) {
                        #pragma unroll
                        for (int r_6 = 0; r_6 < 4; r_6++) {
                            unsigned int stat_row = r_6 / 2 * 16 + lane / 4 + r_6 % 2 * 8;
                            stats0[warp_0 * 32 + stat_row] = row_max[r_6];
                            stats0[256 + warp_0 * 32 + stat_row] = row_sum[r_6];
                        }
                    }
                }
                unsigned long long _mbarrier_arrival_token_3;
                asm volatile(
                    "mbarrier.arrive.release.cta.b64 %0, [%1], %2;"
                    : "=l"(_mbarrier_arrival_token_3)
                    : "l"(smem_raw + ((x_produced0_addr) - smem)), "r"((uint32_t)(1)) : "memory");
                mbarrier_wait_hint(x_produced0_addr, p % 2, 4294967295u);
                float m_new[4];
                #pragma unroll
                for (int r_7 = 0; r_7 < 4; r_7++) {
                    m_new[r_7] = m_run[r_7];
                }
                #pragma unroll
                for (int t = 0; t < 8; t++) {
                    if (b_split > (unsigned int)0 + 8 * p + (unsigned int)t) {
                        #pragma unroll
                        for (int r_8 = 0; r_8 < 4; r_8++) {
                            unsigned int stat_row_1 = r_8 / 2 * 16 + lane / 4 + r_8 % 2 * 8;
                            float tile_max[1];
                            tile_max[0] = stats0[(unsigned int)(t * 32) + stat_row_1];
                            float _fmax_3 = fmaxf(m_new[r_8], tile_max[0]);
                            m_new[r_8] = _fmax_3;
                        }
                    }
                }
                float scale[4];
                #pragma unroll
                for (int r_9 = 0; r_9 < 4; r_9++) {
                    float _expf_0 = __expf(m_run[r_9] - m_new[r_9]);
                    scale[r_9] = _expf_0;
                    l_run[r_9] = l_run[r_9] * scale[r_9];
                    m_run[r_9] = m_new[r_9];
                }
                #pragma unroll
                for (int m_7 = 0; m_7 < 2; m_7++) {
                    #pragma unroll
                    for (int i_4 = 0; i_4 < 2; i_4++) {
                        #pragma unroll
                        for (int n_5 = 0; n_5 < 8; n_5++) {
                            #pragma unroll
                            for (int j_3 = 0; j_3 < 2; j_3++) {
                                acc[(m_7 * 8 + n_5) * 4 + (i_4 * 2 + j_3)] = acc[(m_7 * 8 + n_5) * 4 + (i_4 * 2 + j_3)] * scale[m_7 * 2 + i_4];
                            }
                        }
                    }
                }
                #pragma unroll
                for (int t_1 = 0; t_1 < 8; t_1++) {
                    unsigned int tile_block = (unsigned int)0 + 8 * p + (unsigned int)t_1;
                    {
                        asm volatile("cp.async.wait_group 1;");
                    }
                    if (tile_block < b_split) {
                        unsigned int xscale[4];
                        #pragma unroll
                        for (int r_10 = 0; r_10 < 4; r_10++) {
                            unsigned int stat_row_2 = r_10 / 2 * 16 + lane / 4 + r_10 % 2 * 8;
                            float tile_stat[2];
                            tile_stat[0] = stats0[(unsigned int)(t_1 * 32) + stat_row_2];
                            tile_stat[1] = stats0[(unsigned int)(256 + t_1 * 32) + stat_row_2];
                            float _expf_1 = __expf(tile_stat[0] - m_new[r_10]);
                            float xs = _expf_1;
                            l_run[r_10] = l_run[r_10] + tile_stat[1] * xs;
                            float pair[2];
                            pair[0] = xs;
                            pair[1] = xs;
                            unsigned int packed_scale[1];
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __half2 _h2 = __float22half2_rn(make_float2(pair[_lp*2 + 0], pair[_lp*2+1 + 0]));
                                packed_scale[_lp] = *(uint32_t*)&_h2;
                            }
                            xscale[r_10] = packed_scale[0];
                        }
                        unsigned int xa[8];
                        unsigned int vb[16];
                        #pragma unroll
                        for (int m_8 = 0; m_8 < 2; m_8++) {
                            unsigned int row_0_7 = m_8 * 16 + lane % 16;
                            unsigned int grain_3 = lane / 16;
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(xa[m_8 * 4]), "=r"(xa[m_8 * 4 + 1]), "=r"(xa[m_8 * 4 + 2]), "=r"(xa[m_8 * 4 + 3])
                                : "r"(pool_addr + 81920 + (unsigned int)(t_1 * 1024) + row_0_7 * 32 + (unsigned int)(grain_3 * 16 ^ (row_0_7 >> 2 & 1) << 4))
                                : "memory");
                            #pragma unroll
                            for (int i_5 = 0; i_5 < 2; i_5++) {
                                #pragma unroll
                                for (int j_4 = 0; j_4 < 2; j_4++) {
                                    uint32_t _f16x2_mul_0;
                                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_0) : "r"(xa[m_8 * 4 + (j_4 * 2 + i_5)]), "r"(xscale[m_8 * 2 + i_5]));
                                    xa[m_8 * 4 + (j_4 * 2 + i_5)] = _f16x2_mul_0;
                                }
                            }
                        }
                        #pragma unroll
                        for (int n_6 = 0; n_6 < 4; n_6++) {
                            unsigned int vrow = lane % 16;
                            unsigned int vgrain = n_6 * 2 + lane / 16;
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(vb[n_6 * 4]), "=r"(vb[n_6 * 4 + 1]), "=r"(vb[n_6 * 4 + 2]), "=r"(vb[n_6 * 4 + 3])
                                : "r"(pool_addr + 90112 + warp_0 * 4096 + (unsigned int)(t_1 % 2 * 2048) + vrow * 128 + (unsigned int)(vgrain * 16 ^ (vrow & 7) << 4))
                                : "memory");
                        }
                        #pragma unroll
                        for (int m_9 = 0; m_9 < 2; m_9++) {
                            #pragma unroll
                            for (int n_7 = 0; n_7 < 8; n_7++) {
                                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                    : "+f"((acc + (m_9 * 8 + n_7) * 4)[0]), "+f"((acc + (m_9 * 8 + n_7) * 4)[1]), "+f"((acc + (m_9 * 8 + n_7) * 4)[2]), "+f"((acc + (m_9 * 8 + n_7) * 4)[3])
                                    : "r"((xa + m_9 * 4)[0]), "r"((xa + m_9 * 4)[1]), "r"((xa + m_9 * 4)[2]), "r"((xa + m_9 * 4)[3]), "r"((vb + n_7 * 2)[0]), "r"((vb + n_7 * 2)[1]));
                            }
                        }
                    }
                    {
                        unsigned int refill_block = tile_block + 2;
                        if (refill_block < b_split) {
                            unsigned int slot_3 = (unsigned int)(blockIdx.z * 2 + 1) * kv_cache_list.max_pages + refill_block * 16 / 128;
                            int _vec_load_7[1];
                            {
                                _vec_load_7[0] = *reinterpret_cast<const int*>(kv_cache_list.page_list + slot_3);
                            }
                            int page_3 = _vec_load_7[0];
                            bool valid_5 = page_3 >= 0;
                            info[0] = ((valid_5) ? ((unsigned int)page_3 * 128 + refill_block * 16 % 128) * num_kv_heads + head_group : (unsigned int)0);
                            info[1] = num_kv_heads;
                            info[2] = ((valid_5) ? (unsigned int)1 : (unsigned int)0);
                            unsigned int _min_8 = ((length - refill_block * 16) < ((unsigned int)16) ? (length - refill_block * 16) : ((unsigned int)16));
                            unsigned int refill_available = _min_8;
                            bool valid_0_3 = info[2] != 0;
                            #pragma unroll
                            for (int copy_6 = 0; copy_6 < 4; copy_6++) {
                                unsigned int row16_1 = 4 * copy_6 + lane / 8;
                                unsigned int col_6 = lane % 8;
                                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                                    :: "r"(pool_addr + 90112 + warp_0 * 4096 + (unsigned int)(t_1 % 2 * 2048) + row16_1 * 128 + (unsigned int)(col_6 * 16 ^ (row16_1 & 7) << 4)), "l"(reinterpret_cast<const __half*>(kv_cache_list.pool) + ((info[0] + row16_1 * info[1]) * 512 + warp_0 * 64 + col_6 * 8)), "r"((valid_0_3 && row16_1 < refill_available) ? 16 : 0));
                            }
                        }
                        asm volatile("cp.async.commit_group;");
                    }
                }
                unsigned long long _mbarrier_arrival_token_4;
                asm volatile(
                    "mbarrier.arrive.release.cta.b64 %0, [%1], %2;"
                    : "=l"(_mbarrier_arrival_token_4)
                    : "l"(smem_raw + ((x_consumed0_addr) - smem)), "r"((uint32_t)(1)) : "memory");
            }
            asm volatile("cp.async.wait_group 0;");
            mbarrier_wait_hint(merge_ready_addr, 0, 4294967295u);
            float a0[4];
            float a1[4];
            float inv[4];
            #pragma unroll
            for (int r_11 = 0; r_11 < 4; r_11++) {
                unsigned int stat_row_3 = r_11 / 2 * 16 + lane / 4 + r_11 % 2 * 8;
                float other[2];
                other[0] = stats1[512 + stat_row_3];
                other[1] = stats1[544 + stat_row_3];
                float _fmax_4 = fmaxf(m_run[r_11], other[0]);
                float m_total = _fmax_4;
                float _expf_2 = __expf(m_run[r_11] - m_total);
                a0[r_11] = _expf_2;
                float _expf_3 = __expf(other[0] - m_total);
                a1[r_11] = _expf_3;
                float denominator = l_run[r_11] * a0[r_11] + other[1] * a1[r_11];
                float _rcp_0 = __frcp_rn(denominator);
                inv[r_11] = _rcp_0;
            }
            #pragma unroll
            for (int m_10 = 0; m_10 < 2; m_10++) {
                #pragma unroll
                for (int n_8 = 0; n_8 < 8; n_8++) {
                    float part_2[4];
                    reinterpret_cast<int4*>(part_2 + 0)[0] = reinterpret_cast<int4*>(merge + (warp_0 * 2048 + (unsigned int)((m_10 * 8 + n_8) * 128) + (unsigned int)(lane * 4)))[0];
                    #pragma unroll
                    for (int i_6 = 0; i_6 < 2; i_6++) {
                        #pragma unroll
                        for (int j_5 = 0; j_5 < 2; j_5++) {
                            acc[(m_10 * 8 + n_8) * 4 + (i_6 * 2 + j_5)] = (acc[(m_10 * 8 + n_8) * 4 + (i_6 * 2 + j_5)] * a0[m_10 * 2 + i_6] + part_2[i_6 * 2 + j_5] * a1[m_10 * 2 + i_6]) * inv[m_10 * 2 + i_6];
                        }
                    }
                }
            }
            unsigned int out_packed[32];
            #pragma unroll
            for (int m_11 = 0; m_11 < 2; m_11++) {
                #pragma unroll
                for (int i_7 = 0; i_7 < 2; i_7++) {
                    #pragma unroll
                    for (int n_9 = 0; n_9 < 8; n_9++) {
                        #pragma unroll
                        for (int _lp = 0; _lp < 1; _lp++) {
                            __half2 _h2 = __float22half2_rn(make_float2((acc + (m_11 * 8 + n_9) * 4 + i_7 * 2)[_lp*2 + 0], (acc + (m_11 * 8 + n_9) * 4 + i_7 * 2)[_lp*2+1 + 0]));
                            out_packed[((m_11 * 2 + i_7) * 8 + n_9) + _lp] = *(uint32_t*)&_h2;
                        }
                    }
                }
            }
            unsigned int stage = pool_addr + 90112 + warp_0 * 4096;
            #pragma unroll
            for (int m_12 = 0; m_12 < 4; m_12++) {
                #pragma unroll
                for (int n_10 = 0; n_10 < 2; n_10++) {
                    unsigned int row_0_8 = m_12 * 8 + lane % 8;
                    unsigned int col_7 = n_10 * 4 + lane / 8;
                    uint32_t _stmatrix_addr_5 = static_cast<uint32_t>(stage + row_0_8 * 128 + (unsigned int)(col_7 * 16 ^ (row_0_8 & 7) << 4));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_5), "r"(*reinterpret_cast<const uint32_t*>(&out_packed[m_12 * 8 + n_10 * 4])), "r"(*reinterpret_cast<const uint32_t*>(&out_packed[m_12 * 8 + (n_10 * 4 + 1)])), "r"(*reinterpret_cast<const uint32_t*>(&out_packed[m_12 * 8 + (n_10 * 4 + 2)])), "r"(*reinterpret_cast<const uint32_t*>(&out_packed[m_12 * 8 + (n_10 * 4 + 3)]))
                        : "memory");
                }
            }
            __syncwarp();
            #pragma unroll
            for (int copy_7 = 0; copy_7 < 8; copy_7++) {
                unsigned int row_0_9 = (copy_7 * 32 + lane) / 8;
                unsigned int col_8 = (copy_7 * 32 + lane) % 8;
                unsigned int words[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&words[0])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&words[(0) + 3]))
                    : "r"(stage + row_0_9 * 128 + (unsigned int)(col_8 * 16 ^ (row_0_9 & 7) << 4)));
                if (row_0_9 < valid_rows) {
                    unsigned int head_token_0 = row_begin + row_0_9;
                    unsigned int out_head = (request_offset + head_token_0 / head_group_size) * q_heads + head_group * head_group_size + head_token_0 % head_group_size;
                    reinterpret_cast<int4*>(output + (out_head * 512 + warp_0 * 64 + col_8 * 8))[0] = reinterpret_cast<int4*>(words)[0];
                }
            }
        }
    }
    // ---- Role: group1 ----
    if (warp >= 8 && warp <= 15) {
        { // group1_main
            int warp_id_in_role_1 = (warp - 8);
            unsigned int warp_0_1 = warp_id_in_role_1;
            unsigned int out_row = (warp_0_1 * 32 + (unsigned int)lane) / 8;
            unsigned int out_line = (warp_0_1 * 32 + (unsigned int)lane) % 8;
            unsigned int out_token = row_begin + out_row;
            unsigned int out_head_row = (request_offset + out_token / head_group_size) * q_heads + head_group * head_group_size + out_token % head_group_size;
            if (out_row < valid_rows) { asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"((uint64_t)(output + (out_head_row * 512 + out_line * 64)))); }
            unsigned long long _mbarrier_arrival_token_0;
            asm volatile(
                "mbarrier.arrive.release.cta.b64 %0, [%1], %2;"
                : "=l"(_mbarrier_arrival_token_0)
                : "l"(smem_raw + ((x_consumed1_addr) - smem)), "r"((uint32_t)(1)) : "memory");
            if (valid_rows == 32) {
                #pragma unroll
                for (int copy_8 = 0; copy_8 < 4; copy_8++) {
                    unsigned int grain_4 = (unsigned int)(copy_8 * 512) + (warp_0_1 + 8) * 32 + (unsigned int)lane;
                    unsigned int row_1 = grain_4 / 64;
                    unsigned int col_9 = grain_4 % 64;
                    unsigned int head_token_2 = row_begin + row_1;
                    unsigned int source_head_1 = (request_offset + head_token_2 / head_group_size) * q_heads + head_group * head_group_size + head_token_2 % head_group_size;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16;"
                        :: "r"(sq + row_1 * 1024 + (unsigned int)(col_9 * 16 ^ (row_1 & 7) << 4)), "l"(q + (source_head_1 * 512 + col_9 * 8)));
                }
            } else {
                #pragma unroll
                for (int copy_9 = 0; copy_9 < 4; copy_9++) {
                    unsigned int grain_5 = (unsigned int)(copy_9 * 512) + (warp_0_1 + 8) * 32 + (unsigned int)lane;
                    unsigned int row_2 = grain_5 / 64;
                    unsigned int col_10 = grain_5 % 64;
                    unsigned int head_token_3 = row_begin + row_2;
                    unsigned int source_head_3 = (request_offset + head_token_3 / head_group_size) * q_heads + head_group * head_group_size + head_token_3 % head_group_size;
                    asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                        :: "r"(sq + row_2 * 1024 + (unsigned int)(col_10 * 16 ^ (row_2 & 7) << 4)), "l"(q + (source_head_3 * 512 + col_10 * 8)), "r"((row_2 < valid_rows) ? 16 : 0));
                }
            }
            asm volatile(
                "{\n\t"
                "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n\t"
                "}"
                :: "r"(q_ready_addr) : "memory");
            unsigned int info_1[3];
            unsigned int first_block_1 = b_split + warp_0_1;
            unsigned int first_available_1 = (unsigned int)0;
            if (first_block_1 < nblocks) {
                unsigned int slot_4 = (unsigned int)(blockIdx.z * 2) * kv_cache_list.max_pages + first_block_1 * 16 / 128;
                int _vec_load_8[1];
                {
                    _vec_load_8[0] = *reinterpret_cast<const int*>(kv_cache_list.page_list + slot_4);
                }
                int page_4 = _vec_load_8[0];
                bool valid_6 = page_4 >= 0;
                info_1[0] = ((valid_6) ? ((unsigned int)page_4 * 128 + first_block_1 * 16 % 128) * num_kv_heads + head_group : (unsigned int)0);
                info_1[1] = num_kv_heads;
                info_1[2] = ((valid_6) ? (unsigned int)1 : (unsigned int)0);
                unsigned int _min_9 = ((length - first_block_1 * 16) < ((unsigned int)16) ? (length - first_block_1 * 16) : ((unsigned int)16));
                first_available_1 = _min_9;
            }
            #pragma unroll
            for (int part_3 = 0; part_3 < 5; part_3++) {
                if (first_block_1 < nblocks) {
                    bool valid_7 = info_1[2] != 0;
                    #pragma unroll
                    for (int copy_10 = 0; copy_10 < 2; copy_10++) {
                        unsigned int row_3 = 8 * copy_10 + lane / 4;
                        unsigned int col_11 = lane % 4;
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"(pool_addr + 126976 + warp_0_1 * 6 * 1024 + (unsigned int)(part_3 * 1024) + row_3 * 64 + (unsigned int)(col_11 * 16 ^ (row_3 >> 1 & 3) << 4)), "l"(reinterpret_cast<const __half*>(kv_cache_list.pool) + ((info_1[0] + row_3 * info_1[1]) * 512 + (unsigned int)(part_3 * 32) + col_11 * 8)), "r"((valid_7 && row_3 < first_available_1) ? 16 : 0));
                    }
                }
                asm volatile("cp.async.commit_group;");
            }
            #pragma unroll
            for (int s_1 = 0; s_1 < 2; s_1++) {
                unsigned int slot_block_1 = b_split + (unsigned int)s_1;
                if (slot_block_1 < nblocks) {
                    unsigned int slot_5 = (unsigned int)(blockIdx.z * 2 + 1) * kv_cache_list.max_pages + slot_block_1 * 16 / 128;
                    int _vec_load_9[1];
                    {
                        _vec_load_9[0] = *reinterpret_cast<const int*>(kv_cache_list.page_list + slot_5);
                    }
                    int page_5 = _vec_load_9[0];
                    bool valid_8 = page_5 >= 0;
                    info_1[0] = ((valid_8) ? ((unsigned int)page_5 * 128 + slot_block_1 * 16 % 128) * num_kv_heads + head_group : (unsigned int)0);
                    info_1[1] = num_kv_heads;
                    info_1[2] = ((valid_8) ? (unsigned int)1 : (unsigned int)0);
                    unsigned int _min_10 = ((length - slot_block_1 * 16) < ((unsigned int)16) ? (length - slot_block_1 * 16) : ((unsigned int)16));
                    unsigned int slot_available_1 = _min_10;
                    bool valid_0_4 = info_1[2] != 0;
                    #pragma unroll
                    for (int copy_11 = 0; copy_11 < 4; copy_11++) {
                        unsigned int row16_2 = 4 * copy_11 + lane / 8;
                        unsigned int col_12 = lane % 8;
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"(pool_addr + 184320 + warp_0_1 * 4096 + (unsigned int)(s_1 * 2048) + row16_2 * 128 + (unsigned int)(col_12 * 16 ^ (row16_2 & 7) << 4)), "l"(reinterpret_cast<const __half*>(kv_cache_list.pool) + ((info_1[0] + row16_2 * info_1[1]) * 512 + warp_0_1 * 64 + col_12 * 8)), "r"((valid_0_4 && row16_2 < slot_available_1) ? 16 : 0));
                    }
                }
                asm volatile("cp.async.commit_group;");
            }
            unsigned int slice_rows_1 = (length + blocks_per_group - 1) / blocks_per_group;
            unsigned int slice_begin_1 = (unsigned int)blockIdx.y % blocks_per_group * slice_rows_1;
            unsigned int _min_11 = ((slice_begin_1 + slice_rows_1) < (length) ? (slice_begin_1 + slice_rows_1) : (length));
            unsigned int slice_end_1 = ((slice_begin_1 < length) ? _min_11 : slice_begin_1);
            unsigned int slice_lines_1 = (slice_end_1 - slice_begin_1) * (unsigned int)(512 * ((0) ? 1 : 2) / 128);
            for (unsigned int it_1 = 0; it_1 < (slice_lines_1 + 255) / 256; it_1++) {
                unsigned int index_1 = it_1 * 256 + (warp_0_1 * 32 + (unsigned int)lane);
                unsigned int row_4 = slice_begin_1 + index_1 / (unsigned int)(512 * ((0) ? 1 : 2) / 128);
                unsigned int line_2 = index_1 % (unsigned int)(512 * ((0) ? 1 : 2) / 128);
                bool valid_9 = index_1 < slice_lines_1;
                unsigned int head_row_1 = v_contig_base + row_4;
                unsigned int page_slot_1 = (unsigned int)(blockIdx.z * 2 + 1) * kv_cache_list.max_pages + row_4 / 128;
                int _vec_load_10[1];
                {
                    _vec_load_10[0] = *reinterpret_cast<const int*>(kv_cache_list.page_list + ((valid_9) ? page_slot_1 : (unsigned int)0));
                }
                int page_id_1 = _vec_load_10[0];
                valid_9 = valid_9 && page_id_1 >= 0;
                head_row_1 = ((valid_9) ? ((unsigned int)page_id_1 * 128 + row_4 % 128) * num_kv_heads + head_group : (unsigned int)0);
                if (valid_9) { asm volatile("prefetch.global.L2::evict_last [%0];" :: "l"((uint64_t)(reinterpret_cast<const __half*>(kv_cache_list.pool) + (head_row_1 * 512 + line_2 * (unsigned int)(128 / ((0) ? 1 : 2)))))); }
            }
            mbarrier_wait_hint(q_ready_addr, 0, 4294967295u);
            float acc_1[64];
            acc_1[0] = 0.0f;
            acc_1[1] = 0.0f;
            acc_1[2] = 0.0f;
            acc_1[3] = 0.0f;
            acc_1[4] = 0.0f;
            acc_1[5] = 0.0f;
            acc_1[6] = 0.0f;
            acc_1[7] = 0.0f;
            acc_1[8] = 0.0f;
            acc_1[9] = 0.0f;
            acc_1[10] = 0.0f;
            acc_1[11] = 0.0f;
            acc_1[12] = 0.0f;
            acc_1[13] = 0.0f;
            acc_1[14] = 0.0f;
            acc_1[15] = 0.0f;
            acc_1[16] = 0.0f;
            acc_1[17] = 0.0f;
            acc_1[18] = 0.0f;
            acc_1[19] = 0.0f;
            acc_1[20] = 0.0f;
            acc_1[21] = 0.0f;
            acc_1[22] = 0.0f;
            acc_1[23] = 0.0f;
            acc_1[24] = 0.0f;
            acc_1[25] = 0.0f;
            acc_1[26] = 0.0f;
            acc_1[27] = 0.0f;
            acc_1[28] = 0.0f;
            acc_1[29] = 0.0f;
            acc_1[30] = 0.0f;
            acc_1[31] = 0.0f;
            acc_1[32] = 0.0f;
            acc_1[33] = 0.0f;
            acc_1[34] = 0.0f;
            acc_1[35] = 0.0f;
            acc_1[36] = 0.0f;
            acc_1[37] = 0.0f;
            acc_1[38] = 0.0f;
            acc_1[39] = 0.0f;
            acc_1[40] = 0.0f;
            acc_1[41] = 0.0f;
            acc_1[42] = 0.0f;
            acc_1[43] = 0.0f;
            acc_1[44] = 0.0f;
            acc_1[45] = 0.0f;
            acc_1[46] = 0.0f;
            acc_1[47] = 0.0f;
            acc_1[48] = 0.0f;
            acc_1[49] = 0.0f;
            acc_1[50] = 0.0f;
            acc_1[51] = 0.0f;
            acc_1[52] = 0.0f;
            acc_1[53] = 0.0f;
            acc_1[54] = 0.0f;
            acc_1[55] = 0.0f;
            acc_1[56] = 0.0f;
            acc_1[57] = 0.0f;
            acc_1[58] = 0.0f;
            acc_1[59] = 0.0f;
            acc_1[60] = 0.0f;
            acc_1[61] = 0.0f;
            acc_1[62] = 0.0f;
            acc_1[63] = 0.0f;
            float m_run_1[4];
            float l_run_1[4];
            #pragma unroll
            for (int r_12 = 0; r_12 < 4; r_12++) {
                m_run_1[r_12] = -1e+30f;
                l_run_1[r_12] = 0.0f;
            }
            unsigned int cslot_1 = (unsigned int)0;
            unsigned int islot_1 = (unsigned int)5;
            for (unsigned int p_1 = 0; p_1 < passes1; p_1++) {
                unsigned int block_1 = b_split + 8 * p_1 + warp_0_1;
                if (block_1 < nblocks) {
                    float s_acc_1[16];
                    s_acc_1[0] = 0.0f;
                    s_acc_1[1] = 0.0f;
                    s_acc_1[2] = 0.0f;
                    s_acc_1[3] = 0.0f;
                    s_acc_1[4] = 0.0f;
                    s_acc_1[5] = 0.0f;
                    s_acc_1[6] = 0.0f;
                    s_acc_1[7] = 0.0f;
                    s_acc_1[8] = 0.0f;
                    s_acc_1[9] = 0.0f;
                    s_acc_1[10] = 0.0f;
                    s_acc_1[11] = 0.0f;
                    s_acc_1[12] = 0.0f;
                    s_acc_1[13] = 0.0f;
                    s_acc_1[14] = 0.0f;
                    s_acc_1[15] = 0.0f;
                    unsigned int token0_1 = block_1 * 16;
                    unsigned int _min_12 = ((length - token0_1) < ((unsigned int)16) ? (length - token0_1) : ((unsigned int)16));
                    unsigned int available_1 = _min_12;
                    unsigned int slot_6 = (unsigned int)(blockIdx.z * 2) * kv_cache_list.max_pages + token0_1 / 128;
                    int _vec_load_11[1];
                    {
                        _vec_load_11[0] = *reinterpret_cast<const int*>(kv_cache_list.page_list + slot_6);
                    }
                    int page_6 = _vec_load_11[0];
                    bool valid_10 = page_6 >= 0;
                    info_1[0] = ((valid_10) ? ((unsigned int)page_6 * 128 + token0_1 % 128) * num_kv_heads + head_group : (unsigned int)0);
                    info_1[1] = num_kv_heads;
                    info_1[2] = ((valid_10) ? (unsigned int)1 : (unsigned int)0);
                    unsigned int next_info_1[3];
                    unsigned int next_block_1 = block_1 + 8;
                    unsigned int next_available_1 = (unsigned int)0;
                    if (next_block_1 < nblocks) {
                        unsigned int slot_0_1 = (unsigned int)(blockIdx.z * 2) * kv_cache_list.max_pages + next_block_1 * 16 / 128;
                        int _vec_load_12[1];
                        {
                            _vec_load_12[0] = *reinterpret_cast<const int*>(kv_cache_list.page_list + slot_0_1);
                        }
                        int page_1_2 = _vec_load_12[0];
                        bool valid_2_2 = page_1_2 >= 0;
                        next_info_1[0] = ((valid_2_2) ? ((unsigned int)page_1_2 * 128 + next_block_1 * 16 % 128) * num_kv_heads + head_group : (unsigned int)0);
                        next_info_1[1] = num_kv_heads;
                        next_info_1[2] = ((valid_2_2) ? (unsigned int)1 : (unsigned int)0);
                        unsigned int _min_13 = ((length - next_block_1 * 16) < ((unsigned int)16) ? (length - next_block_1 * 16) : ((unsigned int)16));
                        next_available_1 = _min_13;
                    }
                    #pragma unroll 1
                    for (int part_4 = 0; part_4 < 16; part_4++) {
                        if (part_4 + 6 - 1 < 16) {
                            bool valid_0_5 = info_1[2] != 0;
                            #pragma unroll
                            for (int copy_12 = 0; copy_12 < 2; copy_12++) {
                                unsigned int row_5 = 8 * copy_12 + lane / 4;
                                unsigned int col_13 = lane % 4;
                                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                                    :: "r"(pool_addr + 126976 + warp_0_1 * 6 * 1024 + islot_1 * 1024 + row_5 * 64 + (unsigned int)(col_13 * 16 ^ (row_5 >> 1 & 3) << 4)), "l"(reinterpret_cast<const __half*>(kv_cache_list.pool) + ((info_1[0] + row_5 * info_1[1]) * 512 + (unsigned int)((part_4 + 6 - 1) * 32) + col_13 * 8)), "r"((valid_0_5 && row_5 < available_1) ? 16 : 0));
                            }
                        } else if (next_block_1 < nblocks) {
                            bool valid_0_6 = next_info_1[2] != 0;
                            #pragma unroll
                            for (int copy_13 = 0; copy_13 < 2; copy_13++) {
                                unsigned int row_6 = 8 * copy_13 + lane / 4;
                                unsigned int col_14 = lane % 4;
                                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                                    :: "r"(pool_addr + 126976 + warp_0_1 * 6 * 1024 + islot_1 * 1024 + row_6 * 64 + (unsigned int)(col_14 * 16 ^ (row_6 >> 1 & 3) << 4)), "l"(reinterpret_cast<const __half*>(kv_cache_list.pool) + ((next_info_1[0] + row_6 * next_info_1[1]) * 512 + (unsigned int)((part_4 + 6 - 1 - 16) * 32) + col_14 * 8)), "r"((valid_0_6 && row_6 < next_available_1) ? 16 : 0));
                            }
                        }
                        asm volatile("cp.async.commit_group;");
                        asm volatile("cp.async.wait_group 5;");
                        #pragma unroll
                        for (int split_1 = 0; split_1 < 2; split_1++) {
                            unsigned int qa16_1[8];
                            unsigned int kb16_1[4];
                            #pragma unroll
                            for (int m_13 = 0; m_13 < 2; m_13++) {
                                unsigned int qrow16_1 = m_13 * 16 + lane % 16;
                                unsigned int qgrain16_1 = part_4 * 4 + split_1 * 2 + lane / 16;
                                asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                    : "=r"(qa16_1[m_13 * 4]), "=r"(qa16_1[m_13 * 4 + 1]), "=r"(qa16_1[m_13 * 4 + 2]), "=r"(qa16_1[m_13 * 4 + 3])
                                    : "r"(sq + qrow16_1 * 1024 + (unsigned int)(qgrain16_1 * 16 ^ (qrow16_1 & 7) << 4))
                                    : "memory");
                            }
                            unsigned int krow16_1 = lane % 8 + lane / 16 * 8;
                            unsigned int kgrain16_1 = split_1 * 2 + lane / 8 % 2;
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(kb16_1[0]), "=r"(kb16_1[1]), "=r"(kb16_1[2]), "=r"(kb16_1[3])
                                : "r"(pool_addr + 126976 + warp_0_1 * 6 * 1024 + cslot_1 * 1024 + krow16_1 * 64 + (unsigned int)(kgrain16_1 * 16 ^ (krow16_1 >> 1 & 3) << 4))
                                : "memory");
                            #pragma unroll
                            for (int m_14 = 0; m_14 < 2; m_14++) {
                                #pragma unroll
                                for (int n_11 = 0; n_11 < 2; n_11++) {
                                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                        : "+f"((s_acc_1 + (m_14 * 2 + n_11) * 4)[0]), "+f"((s_acc_1 + (m_14 * 2 + n_11) * 4)[1]), "+f"((s_acc_1 + (m_14 * 2 + n_11) * 4)[2]), "+f"((s_acc_1 + (m_14 * 2 + n_11) * 4)[3])
                                        : "r"((qa16_1 + m_14 * 4)[0]), "r"((qa16_1 + m_14 * 4)[1]), "r"((qa16_1 + m_14 * 4)[2]), "r"((qa16_1 + m_14 * 4)[3]), "r"((kb16_1 + n_11 * 2)[0]), "r"((kb16_1 + n_11 * 2)[1]));
                                }
                            }
                        }
                        cslot_1 = (cslot_1 + 1) % 6;
                        islot_1 = (islot_1 + 1) % 6;
                    }
                    const float2 _scale2_1 = {qk_scale, qk_scale};
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(s_acc_1)[_ls], _scale2_1);
                    if (token0_1 + 16 > length - actual_q) {
                        unsigned int prefix_1 = length - actual_q;
                        #pragma unroll
                        for (int m_15 = 0; m_15 < 2; m_15++) {
                            #pragma unroll
                            for (int i_8 = 0; i_8 < 2; i_8++) {
                                unsigned int _min_14 = (((row_begin + (unsigned int)(m_15 * 16) + (unsigned int)(lane / 4) + (unsigned int)(i_8 * 8)) / head_group_size) < (actual_q - 1) ? ((row_begin + (unsigned int)(m_15 * 16) + (unsigned int)(lane / 4) + (unsigned int)(i_8 * 8)) / head_group_size) : (actual_q - 1));
                                unsigned int token_row_1 = _min_14;
                                unsigned int row_words_1 = (request_offset + token_row_1) * mask_row_words;
                                #pragma unroll
                                for (int n_12 = 0; n_12 < 2; n_12++) {
                                    #pragma unroll
                                    for (int j_6 = 0; j_6 < 2; j_6++) {
                                        unsigned int token_1 = token0_1 + (unsigned int)(n_12 * 8) + (unsigned int)(lane % 4 * 2) + (unsigned int)j_6;
                                        unsigned int _min_15 = ((token_1 - prefix_1) < (actual_q - 1) ? (token_1 - prefix_1) : (actual_q - 1));
                                        unsigned int pos_1 = ((token_1 >= prefix_1) ? _min_15 : (unsigned int)0);
                                        unsigned int _vec_load_13[1];
                                        {
                                            _vec_load_13[0] = *reinterpret_cast<const unsigned int*>(mask + (row_words_1 + pos_1 / 32));
                                        }
                                        unsigned int word_1 = _vec_load_13[0];
                                        bool visible_1 = token_1 < prefix_1 || (word_1 & (unsigned int)1 << pos_1 % 32) != 0;
                                        s_acc_1[(m_15 * 2 + n_12) * 4 + (i_8 * 2 + j_6)] = ((visible_1 && token_1 < length) ? s_acc_1[(m_15 * 2 + n_12) * 4 + (i_8 * 2 + j_6)] : -SM110_XQA_INF);
                                    }
                                }
                            }
                        }
                    }
                    float row_max_1[4];
                    float row_sum_1[4];
                    #pragma unroll
                    for (int r_13 = 0; r_13 < 4; r_13++) {
                        row_max_1[r_13] = -1e+30f;
                        row_sum_1[r_13] = 0.0f;
                    }
                    #pragma unroll
                    for (int m_16 = 0; m_16 < 2; m_16++) {
                        #pragma unroll
                        for (int i_9 = 0; i_9 < 2; i_9++) {
                            #pragma unroll
                            for (int n_13 = 0; n_13 < 2; n_13++) {
                                #pragma unroll
                                for (int j_7 = 0; j_7 < 2; j_7++) {
                                    float _fmax_5 = fmaxf(row_max_1[m_16 * 2 + i_9], s_acc_1[(m_16 * 2 + n_13) * 4 + (i_9 * 2 + j_7)]);
                                    row_max_1[m_16 * 2 + i_9] = _fmax_5;
                                }
                            }
                        }
                    }
                    #pragma unroll
                    for (int r_14 = 0; r_14 < 4; r_14++) {
                        float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, row_max_1[r_14], 2);
                        float _fmax_6 = fmaxf(row_max_1[r_14], _shfl_xor_4);
                        row_max_1[r_14] = _fmax_6;
                    }
                    #pragma unroll
                    for (int r_15 = 0; r_15 < 4; r_15++) {
                        float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, row_max_1[r_15], 1);
                        float _fmax_7 = fmaxf(row_max_1[r_15], _shfl_xor_5);
                        row_max_1[r_15] = _fmax_7;
                    }
                    #pragma unroll
                    for (int m_17 = 0; m_17 < 2; m_17++) {
                        #pragma unroll
                        for (int i_10 = 0; i_10 < 2; i_10++) {
                            float bias_1 = row_max_1[m_17 * 2 + i_10] * 1.4426950408889634f;
                            #pragma unroll
                            for (int n_14 = 0; n_14 < 2; n_14++) {
                                #pragma unroll
                                for (int j_8 = 0; j_8 < 2; j_8++) {
                                    float _exp2_1 = approx_exp2(s_acc_1[(m_17 * 2 + n_14) * 4 + (i_10 * 2 + j_8)] * 1.4426950408889634f - bias_1);
                                    s_acc_1[(m_17 * 2 + n_14) * 4 + (i_10 * 2 + j_8)] = _exp2_1;
                                    row_sum_1[m_17 * 2 + i_10] = row_sum_1[m_17 * 2 + i_10] + s_acc_1[(m_17 * 2 + n_14) * 4 + (i_10 * 2 + j_8)];
                                }
                            }
                        }
                    }
                    #pragma unroll
                    for (int r_16 = 0; r_16 < 4; r_16++) {
                        float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, row_sum_1[r_16], 2);
                        row_sum_1[r_16] = row_sum_1[r_16] + _shfl_xor_6;
                    }
                    #pragma unroll
                    for (int r_17 = 0; r_17 < 4; r_17++) {
                        float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, row_sum_1[r_17], 1);
                        row_sum_1[r_17] = row_sum_1[r_17] + _shfl_xor_7;
                    }
                    mbarrier_wait_hint(x_consumed1_addr, p_1 % 2, 4294967295u);
                    unsigned int packed_1[8];
                    #pragma unroll
                    for (int m_18 = 0; m_18 < 2; m_18++) {
                        #pragma unroll
                        for (int i_11 = 0; i_11 < 2; i_11++) {
                            #pragma unroll
                            for (int n_15 = 0; n_15 < 2; n_15++) {
                                #pragma unroll
                                for (int _lp = 0; _lp < 1; _lp++) {
                                    __half2 _h2 = __float22half2_rn(make_float2((s_acc_1 + (m_18 * 2 + n_15) * 4 + i_11 * 2)[_lp*2 + 0], (s_acc_1 + (m_18 * 2 + n_15) * 4 + i_11 * 2)[_lp*2+1 + 0]));
                                    packed_1[((m_18 * 2 + i_11) * 2 + n_15) + _lp] = *(uint32_t*)&_h2;
                                }
                            }
                        }
                    }
                    #pragma unroll
                    for (int m_19 = 0; m_19 < 2; m_19++) {
                        unsigned int row_7 = m_19 * 16 + lane / 16 * 8 + lane % 8;
                        unsigned int grain_6 = lane / 8 % 2;
                        uint32_t _stmatrix_addr_2 = static_cast<uint32_t>(pool_addr + 176128 + warp_0_1 * 1024 + row_7 * 32 + (unsigned int)(grain_6 * 16 ^ (row_7 >> 2 & 1) << 4));
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_2), "r"(*reinterpret_cast<const uint32_t*>(&packed_1[m_19 * 2 * 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_1[m_19 * 2 * 2 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_1[(m_19 * 2 + 1) * 2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_1[(m_19 * 2 + 1) * 2 + 1]))
                            : "memory");
                    }
                    if (lane % 4 == 0) {
                        #pragma unroll
                        for (int r_18 = 0; r_18 < 4; r_18++) {
                            unsigned int stat_row_4 = r_18 / 2 * 16 + lane / 4 + r_18 % 2 * 8;
                            stats1[warp_0_1 * 32 + stat_row_4] = row_max_1[r_18];
                            stats1[256 + warp_0_1 * 32 + stat_row_4] = row_sum_1[r_18];
                        }
                    }
                }
                unsigned long long _mbarrier_arrival_token_3;
                asm volatile(
                    "mbarrier.arrive.release.cta.b64 %0, [%1], %2;"
                    : "=l"(_mbarrier_arrival_token_3)
                    : "l"(smem_raw + ((x_produced1_addr) - smem)), "r"((uint32_t)(1)) : "memory");
                mbarrier_wait_hint(x_produced1_addr, p_1 % 2, 4294967295u);
                float m_new_1[4];
                #pragma unroll
                for (int r_19 = 0; r_19 < 4; r_19++) {
                    m_new_1[r_19] = m_run_1[r_19];
                }
                #pragma unroll
                for (int t_2 = 0; t_2 < 8; t_2++) {
                    if (nblocks > b_split + 8 * p_1 + (unsigned int)t_2) {
                        #pragma unroll
                        for (int r_20 = 0; r_20 < 4; r_20++) {
                            unsigned int stat_row_5 = r_20 / 2 * 16 + lane / 4 + r_20 % 2 * 8;
                            float tile_max_1[1];
                            tile_max_1[0] = stats1[(unsigned int)(t_2 * 32) + stat_row_5];
                            float _fmax_8 = fmaxf(m_new_1[r_20], tile_max_1[0]);
                            m_new_1[r_20] = _fmax_8;
                        }
                    }
                }
                float scale_1[4];
                #pragma unroll
                for (int r_21 = 0; r_21 < 4; r_21++) {
                    float _expf_4 = __expf(m_run_1[r_21] - m_new_1[r_21]);
                    scale_1[r_21] = _expf_4;
                    l_run_1[r_21] = l_run_1[r_21] * scale_1[r_21];
                    m_run_1[r_21] = m_new_1[r_21];
                }
                #pragma unroll
                for (int m_20 = 0; m_20 < 2; m_20++) {
                    #pragma unroll
                    for (int i_12 = 0; i_12 < 2; i_12++) {
                        #pragma unroll
                        for (int n_16 = 0; n_16 < 8; n_16++) {
                            #pragma unroll
                            for (int j_9 = 0; j_9 < 2; j_9++) {
                                acc_1[(m_20 * 8 + n_16) * 4 + (i_12 * 2 + j_9)] = acc_1[(m_20 * 8 + n_16) * 4 + (i_12 * 2 + j_9)] * scale_1[m_20 * 2 + i_12];
                            }
                        }
                    }
                }
                #pragma unroll
                for (int t_3 = 0; t_3 < 8; t_3++) {
                    unsigned int tile_block_1 = b_split + 8 * p_1 + (unsigned int)t_3;
                    {
                        asm volatile("cp.async.wait_group 1;");
                    }
                    if (tile_block_1 < nblocks) {
                        unsigned int xscale_1[4];
                        #pragma unroll
                        for (int r_22 = 0; r_22 < 4; r_22++) {
                            unsigned int stat_row_6 = r_22 / 2 * 16 + lane / 4 + r_22 % 2 * 8;
                            float tile_stat_1[2];
                            tile_stat_1[0] = stats1[(unsigned int)(t_3 * 32) + stat_row_6];
                            tile_stat_1[1] = stats1[(unsigned int)(256 + t_3 * 32) + stat_row_6];
                            float _expf_5 = __expf(tile_stat_1[0] - m_new_1[r_22]);
                            float xs_1 = _expf_5;
                            l_run_1[r_22] = l_run_1[r_22] + tile_stat_1[1] * xs_1;
                            float pair_1[2];
                            pair_1[0] = xs_1;
                            pair_1[1] = xs_1;
                            unsigned int packed_scale_1[1];
                            #pragma unroll
                            for (int _lp = 0; _lp < 1; _lp++) {
                                __half2 _h2 = __float22half2_rn(make_float2(pair_1[_lp*2 + 0], pair_1[_lp*2+1 + 0]));
                                packed_scale_1[_lp] = *(uint32_t*)&_h2;
                            }
                            xscale_1[r_22] = packed_scale_1[0];
                        }
                        unsigned int xa_1[8];
                        unsigned int vb_1[16];
                        #pragma unroll
                        for (int m_21 = 0; m_21 < 2; m_21++) {
                            unsigned int row_8 = m_21 * 16 + lane % 16;
                            unsigned int grain_7 = lane / 16;
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(xa_1[m_21 * 4]), "=r"(xa_1[m_21 * 4 + 1]), "=r"(xa_1[m_21 * 4 + 2]), "=r"(xa_1[m_21 * 4 + 3])
                                : "r"(pool_addr + 176128 + (unsigned int)(t_3 * 1024) + row_8 * 32 + (unsigned int)(grain_7 * 16 ^ (row_8 >> 2 & 1) << 4))
                                : "memory");
                            #pragma unroll
                            for (int i_13 = 0; i_13 < 2; i_13++) {
                                #pragma unroll
                                for (int j_10 = 0; j_10 < 2; j_10++) {
                                    uint32_t _f16x2_mul_1;
                                    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(_f16x2_mul_1) : "r"(xa_1[m_21 * 4 + (j_10 * 2 + i_13)]), "r"(xscale_1[m_21 * 2 + i_13]));
                                    xa_1[m_21 * 4 + (j_10 * 2 + i_13)] = _f16x2_mul_1;
                                }
                            }
                        }
                        #pragma unroll
                        for (int n_17 = 0; n_17 < 4; n_17++) {
                            unsigned int vrow_1 = lane % 16;
                            unsigned int vgrain_1 = n_17 * 2 + lane / 16;
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                : "=r"(vb_1[n_17 * 4]), "=r"(vb_1[n_17 * 4 + 1]), "=r"(vb_1[n_17 * 4 + 2]), "=r"(vb_1[n_17 * 4 + 3])
                                : "r"(pool_addr + 184320 + warp_0_1 * 4096 + (unsigned int)(t_3 % 2 * 2048) + vrow_1 * 128 + (unsigned int)(vgrain_1 * 16 ^ (vrow_1 & 7) << 4))
                                : "memory");
                        }
                        #pragma unroll
                        for (int m_22 = 0; m_22 < 2; m_22++) {
                            #pragma unroll
                            for (int n_18 = 0; n_18 < 8; n_18++) {
                                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                                    : "+f"((acc_1 + (m_22 * 8 + n_18) * 4)[0]), "+f"((acc_1 + (m_22 * 8 + n_18) * 4)[1]), "+f"((acc_1 + (m_22 * 8 + n_18) * 4)[2]), "+f"((acc_1 + (m_22 * 8 + n_18) * 4)[3])
                                    : "r"((xa_1 + m_22 * 4)[0]), "r"((xa_1 + m_22 * 4)[1]), "r"((xa_1 + m_22 * 4)[2]), "r"((xa_1 + m_22 * 4)[3]), "r"((vb_1 + n_18 * 2)[0]), "r"((vb_1 + n_18 * 2)[1]));
                            }
                        }
                    }
                    {
                        unsigned int refill_block_1 = tile_block_1 + 2;
                        if (refill_block_1 < nblocks) {
                            unsigned int slot_7 = (unsigned int)(blockIdx.z * 2 + 1) * kv_cache_list.max_pages + refill_block_1 * 16 / 128;
                            int _vec_load_14[1];
                            {
                                _vec_load_14[0] = *reinterpret_cast<const int*>(kv_cache_list.page_list + slot_7);
                            }
                            int page_7 = _vec_load_14[0];
                            bool valid_11 = page_7 >= 0;
                            info_1[0] = ((valid_11) ? ((unsigned int)page_7 * 128 + refill_block_1 * 16 % 128) * num_kv_heads + head_group : (unsigned int)0);
                            info_1[1] = num_kv_heads;
                            info_1[2] = ((valid_11) ? (unsigned int)1 : (unsigned int)0);
                            unsigned int _min_16 = ((length - refill_block_1 * 16) < ((unsigned int)16) ? (length - refill_block_1 * 16) : ((unsigned int)16));
                            unsigned int refill_available_1 = _min_16;
                            bool valid_0_7 = info_1[2] != 0;
                            #pragma unroll
                            for (int copy_14 = 0; copy_14 < 4; copy_14++) {
                                unsigned int row16_3 = 4 * copy_14 + lane / 8;
                                unsigned int col_15 = lane % 8;
                                asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                                    :: "r"(pool_addr + 184320 + warp_0_1 * 4096 + (unsigned int)(t_3 % 2 * 2048) + row16_3 * 128 + (unsigned int)(col_15 * 16 ^ (row16_3 & 7) << 4)), "l"(reinterpret_cast<const __half*>(kv_cache_list.pool) + ((info_1[0] + row16_3 * info_1[1]) * 512 + warp_0_1 * 64 + col_15 * 8)), "r"((valid_0_7 && row16_3 < refill_available_1) ? 16 : 0));
                            }
                        }
                        asm volatile("cp.async.commit_group;");
                    }
                }
                unsigned long long _mbarrier_arrival_token_4;
                asm volatile(
                    "mbarrier.arrive.release.cta.b64 %0, [%1], %2;"
                    : "=l"(_mbarrier_arrival_token_4)
                    : "l"(smem_raw + ((x_consumed1_addr) - smem)), "r"((uint32_t)(1)) : "memory");
            }
            asm volatile("cp.async.wait_group 0;");
            mbarrier_wait_hint(x_consumed1_addr, passes1 % 2, 4294967295u);
            #pragma unroll
            for (int m_23 = 0; m_23 < 2; m_23++) {
                #pragma unroll
                for (int n_19 = 0; n_19 < 8; n_19++) {
                    reinterpret_cast<int4*>(merge + (warp_0_1 * 2048 + (unsigned int)((m_23 * 8 + n_19) * 128) + (unsigned int)(lane * 4)))[0] = reinterpret_cast<int4*>(acc_1 + (m_23 * 8 + n_19) * 4)[0];
                }
            }
            if (warp_0_1 == 0) {
                if (lane % 4 == 0) {
                    #pragma unroll
                    for (int r_23 = 0; r_23 < 4; r_23++) {
                        unsigned int stat_row_7 = r_23 / 2 * 16 + lane / 4 + r_23 % 2 * 8;
                        stats1[512 + stat_row_7] = m_run_1[r_23];
                        stats1[544 + stat_row_7] = l_run_1[r_23];
                    }
                }
            }
            unsigned long long _mbarrier_arrival_token_5;
            asm volatile(
                "mbarrier.arrive.release.cta.b64 %0, [%1], %2;"
                : "=l"(_mbarrier_arrival_token_5)
                : "l"(smem_raw + ((merge_ready_addr) - smem)), "r"((uint32_t)(1)) : "memory");
        }
    }

    // Cleanup
}

} // extern "C"
