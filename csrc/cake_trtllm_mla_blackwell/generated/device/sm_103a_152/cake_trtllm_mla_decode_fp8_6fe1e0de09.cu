/*
 * Copyright (c) 2026 by FlashInfer team.
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
// Generated source; do not edit manually.
typedef signed char        int8_t;
typedef unsigned char      uint8_t;
typedef unsigned short     uint16_t;
typedef unsigned int       uint32_t;
#if defined(__CUDACC_RTC__)
typedef unsigned long long uint64_t;
#else
typedef unsigned long      uint64_t;
#endif
static_assert(sizeof(uint64_t) == 8, "MLA requires an LP64 CUDA host ABI");
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) MlaTensorMap { uint64_t opaque[16]; };
struct __align__(64) MlaTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(MlaTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(MlaTensorMap64) == 64, "64-aligned tensor-map ABI alignment");
template <int N>
struct __align__(128) MlaTensorMapPack { MlaTensorMap maps[N]; };

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(MlaTensorMap) >= alignof(CUtensorMap), "MlaTensorMap alignment must cover the CUtensorMap CUDA ABI");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define MLA_INF CUDART_INF_F
#define TMEM_NCOLS 512
#define TMEM_TMEM_SCRATCH_OFFSET 0
#define NUM_K_PIPE_STAGES 2
#define NUM_V_PIPE_STAGES 2
#define SMEM_SMEM_Q_OFF 1024
#define SMEM_SMEM_Q_STAGE_BYTES 40960
#define SMEM_SMEM_Q_STRIDE 40960
#define SMEM_SMEM_K_OFF 41984
#define SMEM_SMEM_K_STAGE_BYTES 40960
#define SMEM_SMEM_K_STRIDE 40960
#define SMEM_SMEM_V_OFF 123904
#define SMEM_SMEM_V_STAGE_BYTES 32768
#define SMEM_SMEM_V_STRIDE 32768
#define SMEM_SMEM_STATS_MAX_OFF 189440
#define SMEM_SMEM_STATS_MAX_STAGE_BYTES 1024
#define SMEM_SMEM_STATS_MAX_STRIDE 1024
#define SMEM_SMEM_STATS_SUM_OFF 190464
#define SMEM_SMEM_STATS_SUM_STAGE_BYTES 512
#define SMEM_SMEM_STATS_SUM_STRIDE 512
#define SMEM_TOTAL 190976
#define THREADS 384
#define USE_PDL 1
#define SM103_SCORE_LD_RED_MAX 1

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


__device__ __forceinline__ uint32_t mbarrier_test_wait(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.test_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
}

__device__ __forceinline__ uint32_t mbarrier_test_wait_cluster(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.test_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
}


__device__ __forceinline__ void mbarrier_wait_token_test(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        asm volatile(
            "{\n\t"
            ".reg .pred P1;\n\t"
            "LAB_WAIT_TEST:\n\t"
            "mbarrier.test_wait.parity.acquire.cta.shared::cta.b64"
            " P1, [%0], %1;\n\t"
            "@P1 bra.uni DONE_TEST;\n\t"
            "bra.uni LAB_WAIT_TEST;\n\t"
            "DONE_TEST:\n\t"
            "}\n"
            :: "r"(mbar_addr), "r"(phase) : "memory");
    }
}

__device__ __forceinline__ void mbarrier_wait_token_test_cluster(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        asm volatile(
            "{\n\t"
            ".reg .pred P1;\n\t"
            "LAB_WAIT_TEST_CLUSTER:\n\t"
            "mbarrier.test_wait.parity.acquire.cluster.shared::cta.b64"
            " P1, [%0], %1;\n\t"
            "@P1 bra.uni DONE_TEST_CLUSTER;\n\t"
            "bra.uni LAB_WAIT_TEST_CLUSTER;\n\t"
            "DONE_TEST_CLUSTER:\n\t"
            "}\n"
            :: "r"(mbar_addr), "r"(phase) : "memory");
    }
}


__device__ __forceinline__ void tcgen05_mma_f8f6f4_cta2(
    int taddr, uint64_t a_desc, uint64_t b_desc,
    uint32_t i_desc, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        ".reg .b32 m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\t"
        "mov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], %1, %2, %3, {m0, m1, m2, m3, m4, m5, m6, m7}, p;\n\t"
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


__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_2d_gmem2smem_cta2(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
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


__device__ __forceinline__ void tmem_st_x8_u32(int addr, const uint32_t* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x8.b32"
        " [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
        :: "r"(addr),
           "r"(src[0]), "r"(src[1]), "r"(src[2]), "r"(src[3]),
           "r"(src[4]), "r"(src[5]), "r"(src[6]), "r"(src[7]));
}

extern "C" {

__global__ __launch_bounds__(384, 1) __cluster_dims__(2,1,1) void
kernel_trtllm_mla_decode_fp8(MlaTensorMap const* tmap_q, MlaTensorMap const* tmap_k, MlaTensorMap const* tmap_v, __nv_bfloat16* __restrict__ partial_O, float* __restrict__ partial_stats, __nv_bfloat16* __restrict__ O, float* __restrict__ LSE, int* __restrict__ seq_lens_kv, int* __restrict__ work_batch_indices, int* __restrict__ page_table, float softmax_scale_log2, float bmm2_scale, int num_split, int num_queries, int max_pages_per_seq)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane = static_cast<uint32_t>(tid) & 31u;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define q_full_addr (mbar_base + 0)
    #define q_empty_addr (mbar_base + 8)
    #define k_full_addr (mbar_base + 16)
    #define k_empty_addr (mbar_base + 32)
    #define v_full_addr (mbar_base + 48)
    #define v_empty_addr (mbar_base + 64)
    #define s_full_addr (mbar_base + 80)
    #define s_empty_addr (mbar_base + 96)
    #define p_full_addr (mbar_base + 112)
    #define p_empty_addr (mbar_base + 128)
    #define o_empty_addr (mbar_base + 144)
    #define stats_addr (mbar_base + 152)
    #define stats_empty_addr (mbar_base + 168)
    #define sum_ready_addr (mbar_base + 184)
    #define o_full_addr (mbar_base + 192)
    #define partial_results_ready_addr (mbar_base + 200)
    #define q_pair_ready_addr (mbar_base + 208)
    #define tmem_dealloc_peer_addr (mbar_base + 216)
    #define tmem_epoch_ready_addr (mbar_base + 224)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(tmap_q)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(tmap_k)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(tmap_v)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    uint8_t* smem_q = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_q_addr = smem + 1024;
    uint8_t* smem_k = reinterpret_cast<uint8_t*>(smem_raw + 41984);
    const int smem_k_addr = smem + 41984;
    uint8_t* smem_v = reinterpret_cast<uint8_t*>(smem_raw + 123904);
    const int smem_v_addr = smem + 123904;
    float* smem_stats_max = reinterpret_cast<float*>(smem_raw + 189440);
    const int smem_stats_max_addr = smem + 189440;
    float* smem_stats_sum = reinterpret_cast<float*>(smem_raw + 190464);
    const int smem_stats_sum_addr = smem + 190464;

    // Mbarrier init (19 pipeline groups, 0 ordered-sequence groups, 29 barriers)
    // Mbarriers at smem_raw[0..232)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // q_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // --- pipeline 'k_pipe' ---
            // k_full: 2 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // k_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // --- pipeline 'v_pipe' ---
            // v_full: 2 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // v_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // s_empty: 2 barriers, init_count=256
            mbarrier_init(smem + 96, 256);
            mbarrier_init(smem + 104, 256);
            // p_full: 2 barriers, init_count=256
            mbarrier_init(smem + 112, 256);
            mbarrier_init(smem + 120, 256);
            // p_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            // o_empty: 1 barriers, init_count=256
            mbarrier_init(smem + 144, 256);
            // stats: 2 barriers, init_count=128
            mbarrier_init(smem + 152, 128);
            mbarrier_init(smem + 160, 128);
            // stats_empty: 2 barriers, init_count=128
            mbarrier_init(smem + 168, 128);
            mbarrier_init(smem + 176, 128);
            // sum_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 184, 128);
            // o_full: 1 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            // partial_results_ready: 1 barriers, init_count=256
            mbarrier_init(smem + 200, 256);
            // q_pair_ready: 1 barriers, init_count=64
            mbarrier_init(smem + 208, 64);
            // tmem_dealloc_peer: 1 barriers, init_count=32
            mbarrier_init(smem + 216, 32);
            // tmem_epoch_ready: 1 barriers, init_count=256
            mbarrier_init(smem + 224, 256);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 232);
    if (warp == 0) {
        int _tmem_hold = smem + 232;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_scratch = taddr;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 112;");
    }

    // ---- Role: softmax_wg ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 200;");
        { // softmax_wg_main
            uint32_t _mbar_token_0 = mbarrier_test_wait(q_full_addr, 0);
            const int wg_dummy_inc = 0;
            const int tmem_row_base = warp % 2 * 32;
            const int score_half = warp / 2;
            const int my_row = (unsigned int)tmem_row_base + lane;
            int split_idx = blockIdx.x / 2;
            int batch_idx = blockIdx.y;
            int seqlen_kv_b = seq_lens_kv[batch_idx];
            int num_kv_tiles_total = (seqlen_kv_b + 128 - 1) / 128;
            int blocks_per_split = (num_kv_tiles_total + num_split - 1) / num_split;
            int my_start_block = split_idx * blocks_per_split;
            int my_end_block_raw = my_start_block + blocks_per_split;
            int my_end_block = ((my_end_block_raw < num_kv_tiles_total) ? my_end_block_raw : num_kv_tiles_total);
            int my_n_tiles = my_end_block - my_start_block;
            mbarrier_wait_token_test(q_full_addr, 0, _mbar_token_0);
            {
                asm volatile("griddepcontrol.wait;" ::: "memory");
                asm volatile("griddepcontrol.wait;" ::: "memory");
            }
            float row_max_val = -MLA_INF;
            float row_sum_val = 0.0f;
            #pragma unroll 1
            for (int tile = 0; tile < my_n_tiles; tile++) {
                int phase = tile & 1;
                int s_wait_phase = tile >> 1 & 1;
                float sv[64];
                float local_max = -MLA_INF;
                {
                    uint32_t _mbar_token_1 = mbarrier_test_wait(s_full_addr + (phase) * 8, s_wait_phase);
                    int s_off = ((phase != 0) ? 128 : 0);
                    int s_base = taddr + (unsigned int)s_off + (unsigned int)(score_half * 64) + (unsigned int)(tmem_row_base << 16);
                    int abs_tile = my_start_block + tile;
                    int tail_valid = seqlen_kv_b - abs_tile * 128;
                    int local_tail_raw = tail_valid - score_half * 64;
                    int local_tail_hi = ((local_tail_raw > 64) ? 64 : local_tail_raw);
                    int local_tail = ((local_tail_hi < 0) ? 0 : local_tail_hi);
                    if (local_tail < 64) {
                        mbarrier_wait_token(s_full_addr + (phase) * 8, s_wait_phase, _mbar_token_1);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                        #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                        #endif
                        asm volatile(
                            "tcgen05.ld.red.sync.aligned.32x32b.x64.max.f32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, [%65];"
                            : "=f"(sv[0]), "=f"(sv[1]), "=f"(sv[2]), "=f"(sv[3]), "=f"(sv[4]), "=f"(sv[5]), "=f"(sv[6]), "=f"(sv[7]), "=f"(sv[8]), "=f"(sv[9]), "=f"(sv[10]), "=f"(sv[11]), "=f"(sv[12]), "=f"(sv[13]), "=f"(sv[14]), "=f"(sv[15]), "=f"(sv[16]), "=f"(sv[17]), "=f"(sv[18]), "=f"(sv[19]), "=f"(sv[20]), "=f"(sv[21]), "=f"(sv[22]), "=f"(sv[23]), "=f"(sv[24]), "=f"(sv[25]), "=f"(sv[26]), "=f"(sv[27]), "=f"(sv[28]), "=f"(sv[29]), "=f"(sv[30]), "=f"(sv[31]), "=f"(sv[32]), "=f"(sv[33]), "=f"(sv[34]), "=f"(sv[35]), "=f"(sv[36]), "=f"(sv[37]), "=f"(sv[38]), "=f"(sv[39]), "=f"(sv[40]), "=f"(sv[41]), "=f"(sv[42]), "=f"(sv[43]), "=f"(sv[44]), "=f"(sv[45]), "=f"(sv[46]), "=f"(sv[47]), "=f"(sv[48]), "=f"(sv[49]), "=f"(sv[50]), "=f"(sv[51]), "=f"(sv[52]), "=f"(sv[53]), "=f"(sv[54]), "=f"(sv[55]), "=f"(sv[56]), "=f"(sv[57]), "=f"(sv[58]), "=f"(sv[59]), "=f"(sv[60]), "=f"(sv[61]), "=f"(sv[62]), "=f"(sv[63]), "=f"(local_max)
                            : "r"(s_base));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        uint32_t _slice_lo_mask_0;
                        {
                            int _lim_0 = local_tail;
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
                        if (!(_slice_lo_mask_0 & (1u << 0))) sv[0] = -MLA_INF;
                        if (!(_slice_lo_mask_0 & (1u << 1))) sv[1] = -MLA_INF;
                        if (!(_slice_lo_mask_0 & (1u << 2))) sv[2] = -MLA_INF;
                        if (!(_slice_lo_mask_0 & (1u << 3))) sv[3] = -MLA_INF;
                        if (!(_slice_lo_mask_0 & (1u << 4))) sv[4] = -MLA_INF;
                        if (!(_slice_lo_mask_0 & (1u << 5))) sv[5] = -MLA_INF;
                        if (!(_slice_lo_mask_0 & (1u << 6))) sv[6] = -MLA_INF;
                        if (!(_slice_lo_mask_0 & (1u << 7))) sv[7] = -MLA_INF;
                        if (!(_slice_lo_mask_0 & (1u << 8))) sv[8] = -MLA_INF;
                        if (!(_slice_lo_mask_0 & (1u << 9))) sv[9] = -MLA_INF;
                        if (!(_slice_lo_mask_0 & (1u << 10))) sv[10] = -MLA_INF;
                        if (!(_slice_lo_mask_0 & (1u << 11))) sv[11] = -MLA_INF;
                        if (!(_slice_lo_mask_0 & (1u << 12))) sv[12] = -MLA_INF;
                        if (!(_slice_lo_mask_0 & (1u << 13))) sv[13] = -MLA_INF;
                        if (!(_slice_lo_mask_0 & (1u << 14))) sv[14] = -MLA_INF;
                        if (!(_slice_lo_mask_0 & (1u << 15))) sv[15] = -MLA_INF;
                        if (!(_slice_lo_mask_0 & (1u << 16))) sv[16] = -MLA_INF;
                        if (!(_slice_lo_mask_0 & (1u << 17))) sv[17] = -MLA_INF;
                        if (!(_slice_lo_mask_0 & (1u << 18))) sv[18] = -MLA_INF;
                        if (!(_slice_lo_mask_0 & (1u << 19))) sv[19] = -MLA_INF;
                        if (!(_slice_lo_mask_0 & (1u << 20))) sv[20] = -MLA_INF;
                        if (!(_slice_lo_mask_0 & (1u << 21))) sv[21] = -MLA_INF;
                        if (!(_slice_lo_mask_0 & (1u << 22))) sv[22] = -MLA_INF;
                        if (!(_slice_lo_mask_0 & (1u << 23))) sv[23] = -MLA_INF;
                        if (!(_slice_lo_mask_0 & (1u << 24))) sv[24] = -MLA_INF;
                        if (!(_slice_lo_mask_0 & (1u << 25))) sv[25] = -MLA_INF;
                        if (!(_slice_lo_mask_0 & (1u << 26))) sv[26] = -MLA_INF;
                        if (!(_slice_lo_mask_0 & (1u << 27))) sv[27] = -MLA_INF;
                        if (!(_slice_lo_mask_0 & (1u << 28))) sv[28] = -MLA_INF;
                        if (!(_slice_lo_mask_0 & (1u << 29))) sv[29] = -MLA_INF;
                        if (!(_slice_lo_mask_0 & (1u << 30))) sv[30] = -MLA_INF;
                        if (!(_slice_lo_mask_0 & (1u << 31))) sv[31] = -MLA_INF;
                        uint32_t _slice_lo_mask_1;
                        {
                            int _lim_1 = local_tail - 32;
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
                        if (!(_slice_lo_mask_1 & (1u << 0))) sv[32] = -MLA_INF;
                        if (!(_slice_lo_mask_1 & (1u << 1))) sv[33] = -MLA_INF;
                        if (!(_slice_lo_mask_1 & (1u << 2))) sv[34] = -MLA_INF;
                        if (!(_slice_lo_mask_1 & (1u << 3))) sv[35] = -MLA_INF;
                        if (!(_slice_lo_mask_1 & (1u << 4))) sv[36] = -MLA_INF;
                        if (!(_slice_lo_mask_1 & (1u << 5))) sv[37] = -MLA_INF;
                        if (!(_slice_lo_mask_1 & (1u << 6))) sv[38] = -MLA_INF;
                        if (!(_slice_lo_mask_1 & (1u << 7))) sv[39] = -MLA_INF;
                        if (!(_slice_lo_mask_1 & (1u << 8))) sv[40] = -MLA_INF;
                        if (!(_slice_lo_mask_1 & (1u << 9))) sv[41] = -MLA_INF;
                        if (!(_slice_lo_mask_1 & (1u << 10))) sv[42] = -MLA_INF;
                        if (!(_slice_lo_mask_1 & (1u << 11))) sv[43] = -MLA_INF;
                        if (!(_slice_lo_mask_1 & (1u << 12))) sv[44] = -MLA_INF;
                        if (!(_slice_lo_mask_1 & (1u << 13))) sv[45] = -MLA_INF;
                        if (!(_slice_lo_mask_1 & (1u << 14))) sv[46] = -MLA_INF;
                        if (!(_slice_lo_mask_1 & (1u << 15))) sv[47] = -MLA_INF;
                        if (!(_slice_lo_mask_1 & (1u << 16))) sv[48] = -MLA_INF;
                        if (!(_slice_lo_mask_1 & (1u << 17))) sv[49] = -MLA_INF;
                        if (!(_slice_lo_mask_1 & (1u << 18))) sv[50] = -MLA_INF;
                        if (!(_slice_lo_mask_1 & (1u << 19))) sv[51] = -MLA_INF;
                        if (!(_slice_lo_mask_1 & (1u << 20))) sv[52] = -MLA_INF;
                        if (!(_slice_lo_mask_1 & (1u << 21))) sv[53] = -MLA_INF;
                        if (!(_slice_lo_mask_1 & (1u << 22))) sv[54] = -MLA_INF;
                        if (!(_slice_lo_mask_1 & (1u << 23))) sv[55] = -MLA_INF;
                        if (!(_slice_lo_mask_1 & (1u << 24))) sv[56] = -MLA_INF;
                        if (!(_slice_lo_mask_1 & (1u << 25))) sv[57] = -MLA_INF;
                        if (!(_slice_lo_mask_1 & (1u << 26))) sv[58] = -MLA_INF;
                        if (!(_slice_lo_mask_1 & (1u << 27))) sv[59] = -MLA_INF;
                        if (!(_slice_lo_mask_1 & (1u << 28))) sv[60] = -MLA_INF;
                        if (!(_slice_lo_mask_1 & (1u << 29))) sv[61] = -MLA_INF;
                        if (!(_slice_lo_mask_1 & (1u << 30))) sv[62] = -MLA_INF;
                        if (!(_slice_lo_mask_1 & (1u << 31))) sv[63] = -MLA_INF;
                        local_max = -MLA_INF;
                        #pragma unroll
                        for (int i = 0; i < 64; i++) {
                            float _max_0 = max_noftz(local_max, sv[i]);
                            local_max = _max_0;
                        }
                    } else {
                        mbarrier_wait_token_test(s_full_addr + (phase) * 8, s_wait_phase, _mbar_token_1);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                        #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                        #endif
                        asm volatile(
                            "tcgen05.ld.red.sync.aligned.32x32b.x64.max.f32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, [%65];"
                            : "=f"(sv[0]), "=f"(sv[1]), "=f"(sv[2]), "=f"(sv[3]), "=f"(sv[4]), "=f"(sv[5]), "=f"(sv[6]), "=f"(sv[7]), "=f"(sv[8]), "=f"(sv[9]), "=f"(sv[10]), "=f"(sv[11]), "=f"(sv[12]), "=f"(sv[13]), "=f"(sv[14]), "=f"(sv[15]), "=f"(sv[16]), "=f"(sv[17]), "=f"(sv[18]), "=f"(sv[19]), "=f"(sv[20]), "=f"(sv[21]), "=f"(sv[22]), "=f"(sv[23]), "=f"(sv[24]), "=f"(sv[25]), "=f"(sv[26]), "=f"(sv[27]), "=f"(sv[28]), "=f"(sv[29]), "=f"(sv[30]), "=f"(sv[31]), "=f"(sv[32]), "=f"(sv[33]), "=f"(sv[34]), "=f"(sv[35]), "=f"(sv[36]), "=f"(sv[37]), "=f"(sv[38]), "=f"(sv[39]), "=f"(sv[40]), "=f"(sv[41]), "=f"(sv[42]), "=f"(sv[43]), "=f"(sv[44]), "=f"(sv[45]), "=f"(sv[46]), "=f"(sv[47]), "=f"(sv[48]), "=f"(sv[49]), "=f"(sv[50]), "=f"(sv[51]), "=f"(sv[52]), "=f"(sv[53]), "=f"(sv[54]), "=f"(sv[55]), "=f"(sv[56]), "=f"(sv[57]), "=f"(sv[58]), "=f"(sv[59]), "=f"(sv[60]), "=f"(sv[61]), "=f"(sv[62]), "=f"(sv[63]), "=f"(local_max)
                            : "r"(s_base));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                    }
                }
                int stats_empty_phase = tile >> 1 & 1 ^ 1;
                mbarrier_wait(stats_empty_addr + (phase) * 8, stats_empty_phase);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int pair_max_idx = phase * 128 + score_half * 64 + my_row;
                smem_stats_max[pair_max_idx] = local_max;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                if (warp % 2 == 0) {
                    asm volatile("barrier.sync 1, 64;" ::: "memory");
                } else {
                    asm volatile("barrier.sync 2, 64;" ::: "memory");
                }
                asm volatile("tcgen05.fence::after_thread_sync;");
                int partner_max_idx = phase * 128 + (1 - score_half) * 64 + my_row;
                float partner_max = smem_stats_max[partner_max_idx];
                float _max_2 = max_noftz(local_max, partner_max);
                float new_max = _max_2;
                float _max_3 = max_noftz(new_max, row_max_val);
                new_max = _max_3;
                float delta = softmax_scale_log2 * (row_max_val - new_max);
                float _exp2_0 = approx_exp2(delta);
                float exp_delta = _exp2_0;
                float acc_scale = ((row_max_val > -MLA_INF) ? exp_delta : 1.0f);
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                if (warp >= 2) {
                    smem_stats_max[phase * 128 + my_row] = acc_scale;
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(stats_addr + (phase) * 8);
                row_max_val = new_max;
                float safe_max = ((new_max == -MLA_INF) ? 0.0f : new_max);
                float max_scaled = safe_max * softmax_scale_log2;
                #pragma unroll
                for (int i_1 = 0; i_1 < 64; i_1++) {
                    float _exp2_1 = approx_exp2(sv[i_1] * softmax_scale_log2 - max_scaled);
                    sv[i_1] = _exp2_1;
                }
                float local_sum = 0.0f;
                #pragma unroll
                for (int i_2 = 0; i_2 < 64; i_2++) {
                    local_sum = local_sum + sv[i_2];
                }
                row_sum_val = row_sum_val * acc_scale + local_sum;
                int p_empty_phase = tile >> 1 & 1 ^ 1;
                mbarrier_wait(p_empty_addr + (phase) * 8, p_empty_phase);
                int p_off = ((phase != 0) ? 224 : 96);
                int p_base = taddr + (unsigned int)p_off + (unsigned int)(score_half * 16) + (unsigned int)(tmem_row_base << 16);
                {
                    uint32_t _pv_packed[8];
                    #pragma unroll
                    for (int _j = 0; _j < 8; _j++) {
                        uint32_t _pk;
                        asm("{\n\t"
                            ".reg .b16 _lo, _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}\n"
                            : "=r"(_pk) : "f"(sv[0 + _j * 4]), "f"(sv[0 + _j * 4 + 1]),
                              "f"(sv[0 + _j * 4 + 2]), "f"(sv[0 + _j * 4 + 3]));
                        _pv_packed[_j] = _pk;
                    }
                    tmem_st_x8_u32(p_base, _pv_packed);
                }
                {
                    uint32_t _pv_packed[8];
                    #pragma unroll
                    for (int _j = 0; _j < 8; _j++) {
                        uint32_t _pk;
                        asm("{\n\t"
                            ".reg .b16 _lo, _hi;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                            "mov.b32 %0, {_lo, _hi};\n\t"
                            "}\n"
                            : "=r"(_pk) : "f"(sv[32 + _j * 4]), "f"(sv[32 + _j * 4 + 1]),
                              "f"(sv[32 + _j * 4 + 2]), "f"(sv[32 + _j * 4 + 3]));
                        _pv_packed[_j] = _pk;
                    }
                    tmem_st_x8_u32(p_base + 8, _pv_packed);
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(p_full_addr + (unsigned int)(phase * 8)), "r"(0) : "memory");
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(s_empty_addr + (unsigned int)(phase * 8)), "r"(0) : "memory");
            }
            int final_sum_slot = my_row * 2 + score_half;
            smem_stats_sum[final_sum_slot] = row_sum_val;
            if (score_half == 0) {
                smem_stats_max[64 + my_row] = row_max_val;
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(sum_ready_addr);
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(partial_results_ready_addr);
            unsigned int _phase_partial_results_ready_0 = 0;
            mbarrier_wait(partial_results_ready_addr, _phase_partial_results_ready_0);
            _phase_partial_results_ready_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
        }
    }
    // ---- Role: correction_wg ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 192;");
        { // correction_wg_main
            const int wg_dummy_inc_1 = 0;
            const int tmem_row_base_1 = warp % 2 * 32;
            const int o_tmem_row_base = warp % 4 * 32;
            const int n_half = warp % 4 / 2;
            const int my_row_1 = (unsigned int)tmem_row_base_1 + lane;
            const int corr_row = o_tmem_row_base << 16;
            float scrub_zero[128];
            #pragma unroll
            for (int scrub_i = 0; scrub_i < 128; scrub_i++) {
                scrub_zero[scrub_i] = 0.0f;
            }
            const int scrub_row = (unsigned int)o_tmem_row_base + lane;
            #pragma unroll
            for (int scrub_col = 0; scrub_col < 512; scrub_col += 128) {
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x128.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127, %128};"
                    :: "r"(taddr + (unsigned int)scrub_col + (unsigned int)(scrub_row << 16)), "f"(scrub_zero[0]), "f"(scrub_zero[1]), "f"(scrub_zero[2]), "f"(scrub_zero[3]), "f"(scrub_zero[4]), "f"(scrub_zero[5]), "f"(scrub_zero[6]), "f"(scrub_zero[7]), "f"(scrub_zero[8]), "f"(scrub_zero[9]), "f"(scrub_zero[10]), "f"(scrub_zero[11]), "f"(scrub_zero[12]), "f"(scrub_zero[13]), "f"(scrub_zero[14]), "f"(scrub_zero[15]), "f"(scrub_zero[16]), "f"(scrub_zero[17]), "f"(scrub_zero[18]), "f"(scrub_zero[19]), "f"(scrub_zero[20]), "f"(scrub_zero[21]), "f"(scrub_zero[22]), "f"(scrub_zero[23]), "f"(scrub_zero[24]), "f"(scrub_zero[25]), "f"(scrub_zero[26]), "f"(scrub_zero[27]), "f"(scrub_zero[28]), "f"(scrub_zero[29]), "f"(scrub_zero[30]), "f"(scrub_zero[31]), "f"(scrub_zero[32]), "f"(scrub_zero[33]), "f"(scrub_zero[34]), "f"(scrub_zero[35]), "f"(scrub_zero[36]), "f"(scrub_zero[37]), "f"(scrub_zero[38]), "f"(scrub_zero[39]), "f"(scrub_zero[40]), "f"(scrub_zero[41]), "f"(scrub_zero[42]), "f"(scrub_zero[43]), "f"(scrub_zero[44]), "f"(scrub_zero[45]), "f"(scrub_zero[46]), "f"(scrub_zero[47]), "f"(scrub_zero[48]), "f"(scrub_zero[49]), "f"(scrub_zero[50]), "f"(scrub_zero[51]), "f"(scrub_zero[52]), "f"(scrub_zero[53]), "f"(scrub_zero[54]), "f"(scrub_zero[55]), "f"(scrub_zero[56]), "f"(scrub_zero[57]), "f"(scrub_zero[58]), "f"(scrub_zero[59]), "f"(scrub_zero[60]), "f"(scrub_zero[61]), "f"(scrub_zero[62]), "f"(scrub_zero[63]), "f"(scrub_zero[64]), "f"(scrub_zero[65]), "f"(scrub_zero[66]), "f"(scrub_zero[67]), "f"(scrub_zero[68]), "f"(scrub_zero[69]), "f"(scrub_zero[70]), "f"(scrub_zero[71]), "f"(scrub_zero[72]), "f"(scrub_zero[73]), "f"(scrub_zero[74]), "f"(scrub_zero[75]), "f"(scrub_zero[76]), "f"(scrub_zero[77]), "f"(scrub_zero[78]), "f"(scrub_zero[79]), "f"(scrub_zero[80]), "f"(scrub_zero[81]), "f"(scrub_zero[82]), "f"(scrub_zero[83]), "f"(scrub_zero[84]), "f"(scrub_zero[85]), "f"(scrub_zero[86]), "f"(scrub_zero[87]), "f"(scrub_zero[88]), "f"(scrub_zero[89]), "f"(scrub_zero[90]), "f"(scrub_zero[91]), "f"(scrub_zero[92]), "f"(scrub_zero[93]), "f"(scrub_zero[94]), "f"(scrub_zero[95]), "f"(scrub_zero[96]), "f"(scrub_zero[97]), "f"(scrub_zero[98]), "f"(scrub_zero[99]), "f"(scrub_zero[100]), "f"(scrub_zero[101]), "f"(scrub_zero[102]), "f"(scrub_zero[103]), "f"(scrub_zero[104]), "f"(scrub_zero[105]), "f"(scrub_zero[106]), "f"(scrub_zero[107]), "f"(scrub_zero[108]), "f"(scrub_zero[109]), "f"(scrub_zero[110]), "f"(scrub_zero[111]), "f"(scrub_zero[112]), "f"(scrub_zero[113]), "f"(scrub_zero[114]), "f"(scrub_zero[115]), "f"(scrub_zero[116]), "f"(scrub_zero[117]), "f"(scrub_zero[118]), "f"(scrub_zero[119]), "f"(scrub_zero[120]), "f"(scrub_zero[121]), "f"(scrub_zero[122]), "f"(scrub_zero[123]), "f"(scrub_zero[124]), "f"(scrub_zero[125]), "f"(scrub_zero[126]), "f"(scrub_zero[127]));
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(tmem_epoch_ready_addr);
            asm volatile(
                "{\n\t"
                ".reg .b32 remAddr32;\n\t"
                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                "}"
                :: "r"(tmem_epoch_ready_addr), "r"(cta_rank ^ 1) : "memory");
            int split_idx_1 = blockIdx.x / 2;
            int batch_idx_1 = blockIdx.y;
            int seqlen_kv_b_1 = seq_lens_kv[batch_idx_1];
            int num_kv_tiles_total_1 = (seqlen_kv_b_1 + 128 - 1) / 128;
            int blocks_per_split_1 = (num_kv_tiles_total_1 + num_split - 1) / num_split;
            int my_start_block_1 = split_idx_1 * blocks_per_split_1;
            int my_end_block_raw_1 = my_start_block_1 + blocks_per_split_1;
            int my_end_block_1 = ((my_end_block_raw_1 < num_kv_tiles_total_1) ? my_end_block_raw_1 : num_kv_tiles_total_1);
            int my_n_tiles_1 = my_end_block_1 - my_start_block_1;
            unsigned int _phase_q_full_0 = 0;
            mbarrier_wait(q_full_addr, _phase_q_full_0);
            _phase_q_full_0 ^= 1;
            {
                asm volatile("griddepcontrol.wait;" ::: "memory");
            }
            unsigned int _phase_o_full_0 = 0;
            #pragma unroll 1
            for (int tile_1 = 0; tile_1 < my_n_tiles_1; tile_1++) {
                int phase_1 = tile_1 & 1;
                int stats_wait_phase = tile_1 >> 1 & 1;
                float acc_scale_1 = 1.0f;
                {
                    uint32_t _mbar_token_2 = mbarrier_test_wait(stats_addr + (phase_1) * 8, stats_wait_phase);
                    int stats_max_idx = phase_1 * 128 + my_row_1;
                    if (tile_1 > 0) {
                        int o_ready_phase = tile_1 - 1 & 1;
                        uint32_t _mbar_token_3 = mbarrier_test_wait(o_full_addr, o_ready_phase);
                        mbarrier_wait_token(stats_addr + (phase_1) * 8, stats_wait_phase, _mbar_token_2);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        acc_scale_1 = smem_stats_max[stats_max_idx];
                        mbarrier_wait_token(o_full_addr, o_ready_phase, _mbar_token_3);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                    } else {
                        mbarrier_wait_token(stats_addr + (phase_1) * 8, stats_wait_phase, _mbar_token_2);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        acc_scale_1 = smem_stats_max[stats_max_idx];
                    }
                }
                if (tile_1 > 0) {
                    int _vote_0 = __any_sync(0xFFFFFFFF, acc_scale_1 < 1.0f);
                    int any_rescale = _vote_0;
                    if (any_rescale != 0) {
                        #pragma unroll
                        for (int vs_local = 0; vs_local < 2; vs_local++) {
                            int vs = vs_local;
                            int o_base = taddr + 256 + (unsigned int)(vs * 128) + (unsigned int)corr_row;
                            #pragma unroll
                            for (int c = 0; c < 128; c += 32) {
                                float _tmem_load_0[32];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                    : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                                    : "r"(o_base + c));
                                #pragma unroll
                                for (int j = 0; j < 32; j++) {
                                    _tmem_load_0[j] = _tmem_load_0[j] * acc_scale_1;
                                }
                                tmem_st_x32_f32(o_base + c, _tmem_load_0);
                                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            }
                        }
                    }
                }
                asm volatile("tcgen05.fence::before_thread_sync;");
                if (tile_1 > 0) {
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(o_empty_addr), "r"(0) : "memory");
                }
                mbarrier_arrive(stats_empty_addr + (phase_1) * 8);
            }
            {
                int o_final_phase = ((my_n_tiles_1 > 0) ? my_n_tiles_1 - 1 & 1 : 0);
                mbarrier_wait(o_full_addr, o_final_phase);
            }
            unsigned int _phase_sum_ready_0 = 0;
            mbarrier_wait(sum_ready_addr, _phase_sum_ready_0);
            _phase_sum_ready_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            int final_sum_slot_1 = my_row_1 * 2 + n_half;
            float local_final_sum = smem_stats_sum[final_sum_slot_1];
            smem_stats_sum[final_sum_slot_1] = local_final_sum;
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            if (warp % 2 == 0) {
                asm volatile("barrier.sync 4, 64;" ::: "memory");
            } else {
                asm volatile("barrier.sync 5, 64;" ::: "memory");
            }
            asm volatile("tcgen05.fence::after_thread_sync;");
            int partner_final_sum_slot = my_row_1 * 2 + (1 - n_half);
            float partner_final_sum = smem_stats_sum[partner_final_sum_slot];
            float total_sum = local_final_sum + partner_final_sum;
            float final_row_max = smem_stats_max[64 + my_row_1];
            float safe_sum = ((total_sum > 0.0f) ? total_sum : 1.0f);
            float _rcp_0 = approx_rcp(safe_sum);
            float output_scale = ((num_split == 1) ? _rcp_0 : 1.0f);
            int head_idx = cta_rank * 64 + my_row_1;
            if (n_half == 0) {
                int stat_off = ((batch_idx_1 * num_split + split_idx_1) * 128 + head_idx) * 2;
                float row_max_scaled = final_row_max * softmax_scale_log2;
                float stored_max = ((final_row_max == -MLA_INF) ? -MLA_INF : row_max_scaled);
                float packed_stats[2];
                packed_stats[0] = stored_max;
                packed_stats[1] = total_sum;
                {
                    float2 _v2 = make_float2(packed_stats[0 + 0], packed_stats[0 + 1]);
                    *reinterpret_cast<float2*>(partial_stats + stat_off + 0) = _v2;
                }
                if (num_split == 1) {
                    float _log2_0;
                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(total_sum));
                    float direct_lse = ((total_sum > 0.0f) ? stored_max + _log2_0 : -MLA_INF);
                    LSE[batch_idx_1 * 128 + head_idx] = direct_lse;
                }
            }
            int po_offset = ((batch_idx_1 * num_split + split_idx_1) * 128 + head_idx) * 512 + n_half * 256;
            int direct_offset = (batch_idx_1 * 128 + head_idx) * 512 + n_half * 256;
            #pragma unroll
            for (int vs_local_1 = 0; vs_local_1 < 2; vs_local_1++) {
                int vs_1 = vs_local_1;
                int o_base_epi = taddr + 256 + (unsigned int)(vs_1 * 128) + (unsigned int)corr_row;
                #pragma unroll
                for (int c_1 = 0; c_1 < 128; c_1 += 64) {
                    float _tmem_load_1[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31])
                        : "r"(o_base_epi + c_1));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_1[32]), "=f"(_tmem_load_1[33]), "=f"(_tmem_load_1[34]), "=f"(_tmem_load_1[35]), "=f"(_tmem_load_1[36]), "=f"(_tmem_load_1[37]), "=f"(_tmem_load_1[38]), "=f"(_tmem_load_1[39]), "=f"(_tmem_load_1[40]), "=f"(_tmem_load_1[41]), "=f"(_tmem_load_1[42]), "=f"(_tmem_load_1[43]), "=f"(_tmem_load_1[44]), "=f"(_tmem_load_1[45]), "=f"(_tmem_load_1[46]), "=f"(_tmem_load_1[47]), "=f"(_tmem_load_1[48]), "=f"(_tmem_load_1[49]), "=f"(_tmem_load_1[50]), "=f"(_tmem_load_1[51]), "=f"(_tmem_load_1[52]), "=f"(_tmem_load_1[53]), "=f"(_tmem_load_1[54]), "=f"(_tmem_load_1[55]), "=f"(_tmem_load_1[56]), "=f"(_tmem_load_1[57]), "=f"(_tmem_load_1[58]), "=f"(_tmem_load_1[59]), "=f"(_tmem_load_1[60]), "=f"(_tmem_load_1[61]), "=f"(_tmem_load_1[62]), "=f"(_tmem_load_1[63])
                        : "r"(o_base_epi + c_1 + 32));
                    int gmem_base = po_offset + vs_1 * 128 + c_1;
                    {
                        unsigned int packed_epi[16];
                        if (num_split == 1) {
                            const float2 _scale2_0 = {output_scale * bmm2_scale, output_scale * bmm2_scale};
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++)
                                reinterpret_cast<float2*>((_tmem_load_1 + 0))[_ls] = mul_f32x2_rn_ftz(reinterpret_cast<float2*>((_tmem_load_1 + 0))[_ls], _scale2_0);
                        } else {
                            const float2 _scale2_1 = {output_scale, output_scale};
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++)
                                reinterpret_cast<float2*>((_tmem_load_1 + 0))[_ls] = mul_f32x2_rn_ftz(reinterpret_cast<float2*>((_tmem_load_1 + 0))[_ls], _scale2_1);
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_1[_lp*2 + 0], _tmem_load_1[_lp*2+1 + 0]));
                            packed_epi[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int j_1 = 0; j_1 < 16; j_1 += 4) {
                            if (num_split == 1) {
                                int direct_base = direct_offset + vs_1 * 128 + c_1;
                                reinterpret_cast<int4*>(O + (direct_base + j_1 * 2))[0] = reinterpret_cast<int4*>(packed_epi + j_1)[0];
                            } else {
                                reinterpret_cast<int4*>(partial_O + (gmem_base + j_1 * 2))[0] = reinterpret_cast<int4*>(packed_epi + j_1)[0];
                            }
                        }
                        if (num_split == 1) {
                            const float2 _scale2_2 = {output_scale * bmm2_scale, output_scale * bmm2_scale};
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++)
                                reinterpret_cast<float2*>((_tmem_load_1 + 32))[_ls] = mul_f32x2_rn_ftz(reinterpret_cast<float2*>((_tmem_load_1 + 32))[_ls], _scale2_2);
                        } else {
                            const float2 _scale2_3 = {output_scale, output_scale};
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++)
                                reinterpret_cast<float2*>((_tmem_load_1 + 32))[_ls] = mul_f32x2_rn_ftz(reinterpret_cast<float2*>((_tmem_load_1 + 32))[_ls], _scale2_3);
                        }
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_1[_lp*2 + 32], _tmem_load_1[_lp*2+1 + 32]));
                            packed_epi[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int j_2 = 0; j_2 < 16; j_2 += 4) {
                            if (num_split == 1) {
                                int direct_base_1 = direct_offset + vs_1 * 128 + c_1;
                                reinterpret_cast<int4*>(O + (direct_base_1 + 32 + j_2 * 2))[0] = reinterpret_cast<int4*>(packed_epi + j_2)[0];
                            } else {
                                reinterpret_cast<int4*>(partial_O + (gmem_base + 32 + j_2 * 2))[0] = reinterpret_cast<int4*>(packed_epi + j_2)[0];
                            }
                        }
                    }
                }
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile(
                "{\n\t"
                ".reg .b32 remAddr32;\n\t"
                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                "}"
                :: "r"(o_empty_addr), "r"(0) : "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(partial_results_ready_addr);
            unsigned int _phase_partial_results_ready_0_1 = 0;
            mbarrier_wait(partial_results_ready_addr, _phase_partial_results_ready_0_1);
            _phase_partial_results_ready_0_1 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            if (USE_PDL && num_split > 1) {
                __threadfence();
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
            asm volatile("barrier.sync 13, 160;" ::: "memory");
            if (warp == 4) {
                int tmem_dealloc_peer_rank = cta_rank ^ 1;
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(tmem_dealloc_peer_addr), "r"(tmem_dealloc_peer_rank) : "memory");
                mbarrier_wait(tmem_dealloc_peer_addr, 0);
                int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
                asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(512));
            }
        }
    }
    // ---- Role: mma_warp ----
    if (warp == 8) {
        { // mma_warp_main
            const int wg3_dummy = 0;
            unsigned int mma_k_stage = 0;
            unsigned int mma_v_stage = 0;
            int split_idx_2 = blockIdx.x / 2;
            int batch_idx_2 = blockIdx.y;
            int seqlen_kv_b_2 = seq_lens_kv[batch_idx_2];
            int num_kv_tiles_total_2 = (seqlen_kv_b_2 + 128 - 1) / 128;
            int blocks_per_split_2 = (num_kv_tiles_total_2 + num_split - 1) / num_split;
            int my_start_block_2 = split_idx_2 * blocks_per_split_2;
            int my_end_block_raw_2 = my_start_block_2 + blocks_per_split_2;
            int my_end_block_2 = ((my_end_block_raw_2 < num_kv_tiles_total_2) ? my_end_block_raw_2 : num_kv_tiles_total_2);
            int my_n_tiles_2 = my_end_block_2 - my_start_block_2;
            unsigned int _phase_tmem_epoch_ready_0 = 0;
            mbarrier_wait(tmem_epoch_ready_addr, _phase_tmem_epoch_ready_0);
            _phase_tmem_epoch_ready_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            unsigned int _phase_q_full_0_1 = 0;
            mbarrier_wait(q_full_addr, _phase_q_full_0_1);
            _phase_q_full_0_1 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            asm volatile(
                "{\n\t"
                ".reg .b32 remAddr32;\n\t"
                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                "}"
                :: "r"(q_pair_ready_addr), "r"(0) : "memory");
            unsigned int _phase_q_pair_ready_0 = 0;
            if (cta_rank == 0) {
                mbarrier_wait_cluster_hint(q_pair_ready_addr, _phase_q_pair_ready_0, 10000000);
                _phase_q_pair_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
            }
            int first_pv = 1;
            unsigned int _phase_k_full = 0;
            unsigned int _phase_o_empty_0 = 1;
            unsigned int _phase_v_full = 0;
            #pragma unroll 1
            for (int tile_2 = 0; tile_2 < my_n_tiles_2; tile_2++) {
                int phase_2 = tile_2 & 1;
                int s_empty_phase = tile_2 >> 1 & 1 ^ 1;
                int score_col = ((phase_2 != 0) ? 128 : 0);
                if (cta_rank == 0) {
                    mbarrier_wait(s_empty_addr + (phase_2) * 8, s_empty_phase);
                    mbarrier_wait(k_full_addr + (mma_k_stage) * 8, _phase_k_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    {
                        int _mma_a_lo_0 = (((smem_q_addr) >> 4) & 0x3FFF) + (0) * 2560;
                        int _mma_b_lo_0 = (((smem_k_addr) >> 4) & 0x3FFF) + (mma_k_stage) * 2560;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 136314896;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_tmem_scratch + (score_col))), "r"(0));
                    }
                    {
                        int _mma_a_lo_2 = (((smem_q_addr + 8192) >> 4) & 0x3FFF) + (0) * 2560;
                        int _mma_b_lo_2 = (((smem_k_addr + 8192) >> 4) & 0x3FFF) + (mma_k_stage) * 2560;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 136314896;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_2), "r"(_mma_b_lo_2), "r"((tmem_tmem_scratch + (score_col))), "r"(1));
                    }
                    {
                        int _mma_a_lo_4 = (((smem_q_addr + 16384) >> 4) & 0x3FFF) + (0) * 2560;
                        int _mma_b_lo_4 = (((smem_k_addr + 16384) >> 4) & 0x3FFF) + (mma_k_stage) * 2560;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 136314896;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_4), "r"(_mma_b_lo_4), "r"((tmem_tmem_scratch + (score_col))), "r"(1));
                    }
                    {
                        int _mma_a_lo_6 = (((smem_q_addr + 24576) >> 4) & 0x3FFF) + (0) * 2560;
                        int _mma_b_lo_6 = (((smem_k_addr + 24576) >> 4) & 0x3FFF) + (mma_k_stage) * 2560;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 136314896;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_6), "r"(_mma_b_lo_6), "r"((tmem_tmem_scratch + (score_col))), "r"(1));
                    }
                    {
                        int _mma_a_lo_9 = (((smem_q_addr + 32768) >> 4) & 0x3FFF) + (0) * 2560;
                        int _mma_b_lo_9 = (((smem_k_addr + 32768) >> 4) & 0x3FFF) + (mma_k_stage) * 2560;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 136314896;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_9), "r"(_mma_b_lo_9), "r"((tmem_tmem_scratch + (score_col))), "r"(1));
                    }
                    elect_commit_cg2_multicast(s_full_addr + (phase_2) * 8, (uint16_t)(3));
                    elect_commit_cg2_multicast(k_empty_addr + (mma_k_stage) * 8, (uint16_t)(3));
                }
                mma_k_stage += 1;
                if (mma_k_stage == 2) { mma_k_stage = 0; _phase_k_full ^= 1; }
                if (tile_2 > 0) {
                    int prev_phase = tile_2 - 1 & 1;
                    int prev_tile = tile_2 - 1;
                    int pv_wait_phase = prev_tile >> 1 & 1;
                    if (cta_rank == 0) {
                        mbarrier_wait(o_empty_addr, _phase_o_empty_0);
                        _phase_o_empty_0 ^= 1;
                        mbarrier_wait(p_full_addr + (prev_phase) * 8, pv_wait_phase);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int p_col = ((prev_phase != 0) ? 224 : 96);
                        mbarrier_wait(v_full_addr + (mma_v_stage) * 8, _phase_v_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_b_lo_10 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (mma_v_stage) * 2048;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_tmem_scratch + (256))), "r"(_mma_b_lo_10), "r"(tmem_tmem_scratch + p_col), "r"(((first_pv) ? 0 : 1)));
                        int _mma_b_lo_11 = ((((smem_v_addr + 16384) >> 4) & 0x3FFF) | 0x4000000) + (mma_v_stage) * 2048;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_tmem_scratch + (384))), "r"(_mma_b_lo_11), "r"(tmem_tmem_scratch + p_col), "r"(((first_pv) ? 0 : 1)));
                        elect_commit_cg2_multicast(v_empty_addr + (mma_v_stage) * 8, (uint16_t)(3));
                        mma_v_stage += 1;
                        if (mma_v_stage == 2) { mma_v_stage = 0; _phase_v_full ^= 1; }
                        elect_commit_cg2_multicast(p_empty_addr + (prev_phase) * 8, (uint16_t)(3));
                        elect_commit_cg2_multicast(o_full_addr, (uint16_t)(3));
                    }
                    first_pv = 0;
                }
            }
            int last_phase = my_n_tiles_2 - 1 & 1;
            int last_tile = my_n_tiles_2 - 1;
            int drain_wait_phase = last_tile >> 1 & 1;
            if (cta_rank == 0) {
                mbarrier_wait(o_empty_addr, _phase_o_empty_0);
                _phase_o_empty_0 ^= 1;
                mbarrier_wait(p_full_addr + (last_phase) * 8, drain_wait_phase);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int p_col_last = ((last_phase != 0) ? 224 : 96);
                mbarrier_wait(v_full_addr + (mma_v_stage) * 8, _phase_v_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_b_lo_12 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (mma_v_stage) * 2048;
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_tmem_scratch + (256))), "r"(_mma_b_lo_12), "r"(tmem_tmem_scratch + p_col_last), "r"(((first_pv) ? 0 : 1)));
                int _mma_b_lo_13 = ((((smem_v_addr + 16384) >> 4) & 0x3FFF) | 0x4000000) + (mma_v_stage) * 2048;
                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 dhi, 0x40004040;\n\t"
                    "mov.b32 id, 136380432;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_tmem_scratch + (384))), "r"(_mma_b_lo_13), "r"(tmem_tmem_scratch + p_col_last), "r"(((first_pv) ? 0 : 1)));
                elect_commit_cg2_multicast(v_empty_addr + (mma_v_stage) * 8, (uint16_t)(3));
                mma_v_stage += 1;
                if (mma_v_stage == 2) { mma_v_stage = 0; _phase_v_full ^= 1; }
                elect_commit_cg2_multicast(p_empty_addr + (last_phase) * 8, (uint16_t)(3));
                elect_commit_cg2_multicast(o_full_addr, (uint16_t)(3));
                elect_commit_cg2_multicast(q_empty_addr, (uint16_t)(3));
                mbarrier_wait(o_empty_addr, _phase_o_empty_0);
                _phase_o_empty_0 ^= 1;
            }
            asm volatile("barrier.arrive 13, 160;" ::: "memory");
        }
    }
    // ---- Role: page_offsets_warp ----
    if (warp == 9) {
        { // page_offsets_warp_main
            const int wg3_dummy_1 = 0;
            unsigned int _phase_q_full_0_2 = 0;
            mbarrier_wait(q_full_addr, _phase_q_full_0_2);
            _phase_q_full_0_2 ^= 1;
        }
    }
    // ---- Role: padding_warp ----
    if (warp == 10) {
        { // padding_warp_main
            const int wg3_dummy_2 = 0;
            unsigned int _phase_q_full_0_3 = 0;
            mbarrier_wait(q_full_addr, _phase_q_full_0_3);
            _phase_q_full_0_3 ^= 1;
        }
    }
    // ---- Role: load_warp ----
    if (warp == 11) {
        { // load_warp_main
            const int wg3_dummy_3 = 0;
            unsigned int load_k_stage = 0;
            unsigned int load_v_stage = 0;
            int split_idx_3 = blockIdx.x / 2;
            int batch_idx_3 = blockIdx.y;
            int seqlen_kv_b_3 = seq_lens_kv[batch_idx_3];
            int num_kv_tiles_total_3 = (seqlen_kv_b_3 + 128 - 1) / 128;
            int blocks_per_split_3 = (num_kv_tiles_total_3 + num_split - 1) / num_split;
            int my_start_block_3 = split_idx_3 * blocks_per_split_3;
            int my_end_block_raw_3 = my_start_block_3 + blocks_per_split_3;
            int my_end_block_3 = ((my_end_block_raw_3 < num_kv_tiles_total_3) ? my_end_block_raw_3 : num_kv_tiles_total_3);
            int my_n_tiles_3 = my_end_block_3 - my_start_block_3;
            int page_batch_idx = work_batch_indices[batch_idx_3];
            int pt_base = page_batch_idx * max_pages_per_seq;
            int q_row_global = batch_idx_3 * 128 + cta_rank * 64;
            unsigned int _phase_q_empty_0 = 1;
            mbarrier_wait(q_empty_addr, _phase_q_empty_0);
            _phase_q_empty_0 ^= 1;
            {
                asm volatile("griddepcontrol.wait;" ::: "memory");
            }
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(q_full_addr, 40960);
                #pragma unroll
                for (int s = 0; s < 5; s++) {
                    int q_dst = smem_q_addr + (unsigned int)(s * 8192);
                    tma_2d_gmem2smem(q_dst, tmap_q, s * 128, q_row_global, q_full_addr);
                }
            }
            int pg_k0 = page_table[pt_base + 2 * my_start_block_3 + cta_rank];
            unsigned int _phase_k_empty = 1;
            mbarrier_wait(k_empty_addr + (load_k_stage) * 8, _phase_k_empty);
            if (cta_rank == 0) {
                if (elect_sync()) {
                    asm volatile(
                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                        :: "r"((k_full_addr + (load_k_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(81920)) : "memory");
                }
            }
            if (elect_sync()) {
                int k0_dst = smem_k_addr + load_k_stage * 40960;
                #pragma unroll
                for (int n = 0; n < 5; n++) {
                    tma_2d_gmem2smem_cta2(k0_dst + n * 8192, tmap_k, n * 128, pg_k0 * 64, ((k_full_addr + (load_k_stage) * 8) & 0xFEFFFFFF));
                }
            }
            load_k_stage += 1;
            if (load_k_stage == 2) { load_k_stage = 0; _phase_k_empty ^= 1; }
            unsigned int _phase_v_empty = 1;
            #pragma unroll 1
            for (int tile_3 = 1; tile_3 < my_n_tiles_3; tile_3++) {
                int abs_tile_k = my_start_block_3 + tile_3;
                int pg_k = page_table[pt_base + 2 * abs_tile_k + cta_rank];
                mbarrier_wait(k_empty_addr + (load_k_stage) * 8, _phase_k_empty);
                if (cta_rank == 0) {
                    if (elect_sync()) {
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((k_full_addr + (load_k_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(81920)) : "memory");
                    }
                }
                if (elect_sync()) {
                    int dst_k = smem_k_addr + load_k_stage * 40960;
                    #pragma unroll
                    for (int n_1 = 0; n_1 < 5; n_1++) {
                        tma_2d_gmem2smem_cta2(dst_k + n_1 * 8192, tmap_k, n_1 * 128, pg_k * 64, ((k_full_addr + (load_k_stage) * 8) & 0xFEFFFFFF));
                    }
                }
                load_k_stage += 1;
                if (load_k_stage == 2) { load_k_stage = 0; _phase_k_empty ^= 1; }
                int abs_tile_v = my_start_block_3 + (tile_3 - 1);
                int pg_v0 = page_table[pt_base + 2 * abs_tile_v];
                int pg_v1 = page_table[pt_base + 2 * abs_tile_v + 1];
                mbarrier_wait(v_empty_addr + (load_v_stage) * 8, _phase_v_empty);
                if (cta_rank == 0) {
                    if (elect_sync()) {
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((v_full_addr + (load_v_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(65536)) : "memory");
                    }
                }
                if (elect_sync()) {
                    int v_dst = smem_v_addr + load_v_stage * 32768;
                    int v0_z = cta_rank;
                    int v1_z = 2 + cta_rank;
                    tma_2d_gmem2smem_cta2(v_dst, tmap_v, v0_z * 128, pg_v0 * 64, ((v_full_addr + (load_v_stage) * 8) & 0xFEFFFFFF));
                    tma_2d_gmem2smem_cta2(v_dst + 16384, tmap_v, v1_z * 128, pg_v0 * 64, ((v_full_addr + (load_v_stage) * 8) & 0xFEFFFFFF));
                    tma_2d_gmem2smem_cta2(v_dst + 8192, tmap_v, v0_z * 128, pg_v1 * 64, ((v_full_addr + (load_v_stage) * 8) & 0xFEFFFFFF));
                    tma_2d_gmem2smem_cta2(v_dst + 16384 + 8192, tmap_v, v1_z * 128, pg_v1 * 64, ((v_full_addr + (load_v_stage) * 8) & 0xFEFFFFFF));
                }
                load_v_stage += 1;
                if (load_v_stage == 2) { load_v_stage = 0; _phase_v_empty ^= 1; }
            }
            int abs_tile_last = my_start_block_3 + (my_n_tiles_3 - 1);
            int pg_vl0 = page_table[pt_base + 2 * abs_tile_last];
            int pg_vl1 = page_table[pt_base + 2 * abs_tile_last + 1];
            mbarrier_wait(v_empty_addr + (load_v_stage) * 8, _phase_v_empty);
            if (cta_rank == 0) {
                if (elect_sync()) {
                    asm volatile(
                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                        :: "r"((v_full_addr + (load_v_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(65536)) : "memory");
                }
            }
            if (elect_sync()) {
                int v_dst_last = smem_v_addr + load_v_stage * 32768;
                int v0_z_last = cta_rank;
                int v1_z_last = 2 + cta_rank;
                tma_2d_gmem2smem_cta2(v_dst_last, tmap_v, v0_z_last * 128, pg_vl0 * 64, ((v_full_addr + (load_v_stage) * 8) & 0xFEFFFFFF));
                tma_2d_gmem2smem_cta2(v_dst_last + 16384, tmap_v, v1_z_last * 128, pg_vl0 * 64, ((v_full_addr + (load_v_stage) * 8) & 0xFEFFFFFF));
                tma_2d_gmem2smem_cta2(v_dst_last + 8192, tmap_v, v0_z_last * 128, pg_vl1 * 64, ((v_full_addr + (load_v_stage) * 8) & 0xFEFFFFFF));
                tma_2d_gmem2smem_cta2(v_dst_last + 16384 + 8192, tmap_v, v1_z_last * 128, pg_vl1 * 64, ((v_full_addr + (load_v_stage) * 8) & 0xFEFFFFFF));
            }
            load_v_stage += 1;
            if (load_v_stage == 2) { load_v_stage = 0; _phase_v_empty ^= 1; }
        }
    }

    // Cleanup
}

} // extern "C"

