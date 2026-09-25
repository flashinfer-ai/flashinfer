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
#define TMEM_TMEM_K16_OFFSET 64
#define TMEM_TMEM_O_OFFSET 192
#define NUM_RAW_RING_STAGES 11
#define NUM_S_RING_STAGES 2
#define NUM_P_RING_STAGES 2
#define NUM_ITEM_RING_STAGES 2
#define SMEM_SMEM_U32_OFF 1024
#define SMEM_SMEM_U32_STAGE_BYTES 226304
#define SMEM_SMEM_U32_STRIDE 226304
#define SMEM_SMEM_F32_OFF 1024
#define SMEM_SMEM_F32_STAGE_BYTES 512
#define SMEM_SMEM_F32_STRIDE 512
#define SMEM_SMEM_PAGE_OFF 1664
#define SMEM_SMEM_PAGE_STAGE_BYTES 128
#define SMEM_SMEM_PAGE_STRIDE 128
#define SMEM_SMEM_VALID_OFF 1792
#define SMEM_SMEM_VALID_STAGE_BYTES 128
#define SMEM_SMEM_VALID_STRIDE 128
#define SMEM_SMEM_NPAIRS_OFF 1920
#define SMEM_SMEM_NPAIRS_STAGE_BYTES 8
#define SMEM_SMEM_NPAIRS_STRIDE 8
#define SMEM_SMEM_SPLIT_FLAG_OFF 1936
#define SMEM_SMEM_SPLIT_FLAG_STAGE_BYTES 4
#define SMEM_SMEM_SPLIT_FLAG_STRIDE 4
#define SMEM_SMEM_PART_OFF 3072
#define SMEM_SMEM_PART_STAGE_BYTES 8192
#define SMEM_SMEM_PART_STRIDE 8192
#define SMEM_SMEM_PSTAT_OFF 2048
#define SMEM_SMEM_PSTAT_STAGE_BYTES 128
#define SMEM_SMEM_PSTAT_STRIDE 128
#define SMEM_SMEM_QT_OFF 3072
#define SMEM_SMEM_QT_STAGE_BYTES 4096
#define SMEM_SMEM_QT_STRIDE 4096
#define SMEM_SMEM_QH_OFF 3072
#define SMEM_SMEM_QH_STAGE_BYTES 2048
#define SMEM_SMEM_QH_STRIDE 2048
#define SMEM_SMEM_P_OFF 11264
#define SMEM_SMEM_P_STAGE_BYTES 4096
#define SMEM_SMEM_P_STRIDE 4096
#define SMEM_SMEM_SCALE_OFF 27648
#define SMEM_SMEM_SCALE_STAGE_BYTES 1024
#define SMEM_SMEM_SCALE_STRIDE 1024
#define SMEM_SMEM_RAW_OFF 38912
#define SMEM_SMEM_RAW_STAGE_BYTES 8192
#define SMEM_SMEM_RAW_STRIDE 8192
#define SMEM_SMEM_V_OFF 129024
#define SMEM_SMEM_V_STAGE_BYTES 32768
#define SMEM_SMEM_V_STRIDE 32768
#define SMEM_SMEM_V_MN_OFF 129024
#define SMEM_SMEM_V_MN_STAGE_BYTES 32768
#define SMEM_SMEM_V_MN_STRIDE 32768
#define SMEM_TOTAL 227328
#define THREADS 512

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


__device__ __forceinline__ void tma_3d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
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


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(512, 1) __cluster_dims__(2,1,1) void
kernel_cake_msa_nvfp4_decode_3e69c5a37956de9de9fd(const __grid_constant__ CUtensorMap Q, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap K_scale, const __grid_constant__ CUtensorMap V, const __grid_constant__ CUtensorMap V_scale, __nv_bfloat16* __restrict__ O, float* __restrict__ msa_lse, float* __restrict__ partial_O, float* __restrict__ partial_M, float* __restrict__ partial_D, int* __restrict__ split_completion, int* __restrict__ kv_indices, int* __restrict__ kv_indptr, int* __restrict__ task_kind, int* __restrict__ task_request, int* __restrict__ task_kv_head, int total_q, int seqlen_q, int num_q_heads, int num_kv_heads, float softmax_scale_log2, float output_scale, int msa_max_pages)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);

    const int mbar_base = smem;
    #define q_full_addr (mbar_base + 0)
    #define q_empty_addr (mbar_base + 16)
    #define meta_full_addr (mbar_base + 32)
    #define kv_raw_full_addr (mbar_base + 48)
    #define kv_raw_empty_addr (mbar_base + 304)
    #define k16_full_addr (mbar_base + 392)
    #define k16_empty_addr (mbar_base + 408)
    #define v_full_addr (mbar_base + 424)
    #define v_empty_addr (mbar_base + 448)
    #define s_full_addr (mbar_base + 472)
    #define s_empty_addr (mbar_base + 488)
    #define p_full_addr (mbar_base + 504)
    #define pv_done_addr (mbar_base + 520)
    #define decode_done_addr (mbar_base + 584)
    #define meta_empty_addr (mbar_base + 592)
    #define part_ready_addr (mbar_base + 608)
    #define q_dead_addr (mbar_base + 616)
    #define tmem_dead_addr (mbar_base + 624)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    unsigned int* smem_u32 = reinterpret_cast<unsigned int*>(smem_raw + 1024);
    const int smem_u32_addr = smem + 1024;
    float* smem_f32 = reinterpret_cast<float*>(smem_raw + 1024);
    const int smem_f32_addr = smem + 1024;
    int* smem_page = reinterpret_cast<int*>(smem_raw + 1664);
    const int smem_page_addr = smem + 1664;
    int* smem_valid = reinterpret_cast<int*>(smem_raw + 1792);
    const int smem_valid_addr = smem + 1792;
    int* smem_npairs = reinterpret_cast<int*>(smem_raw + 1920);
    const int smem_npairs_addr = smem + 1920;
    int* smem_split_flag = reinterpret_cast<int*>(smem_raw + 1936);
    const int smem_split_flag_addr = smem + 1936;
    float* smem_part = reinterpret_cast<float*>(smem_raw + 3072);
    const int smem_part_addr = smem + 3072;
    float* smem_pstat = reinterpret_cast<float*>(smem_raw + 2048);
    const int smem_pstat_addr = smem + 2048;
    __nv_bfloat16* smem_qt = reinterpret_cast<__nv_bfloat16*>(smem_raw + 3072);
    const int smem_qt_addr = smem + 3072;
    __nv_bfloat16* smem_qh = reinterpret_cast<__nv_bfloat16*>(smem_raw + 3072);
    const int smem_qh_addr = smem + 3072;
    __nv_bfloat16* smem_p = reinterpret_cast<__nv_bfloat16*>(smem_raw + 11264);
    const int smem_p_addr = smem + 11264;
    uint8_t* smem_scale = reinterpret_cast<uint8_t*>(smem_raw + 27648);
    const int smem_scale_addr = smem + 27648;
    uint8_t* smem_raw_1 = reinterpret_cast<uint8_t*>(smem_raw + 38912);
    const int smem_raw_addr = smem + 38912;
    __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 129024);
    const int smem_v_addr = smem + 129024;
    __nv_bfloat16* smem_v_mn = reinterpret_cast<__nv_bfloat16*>(smem_raw + 129024);
    const int smem_v_mn_addr = smem + 129024;

    // Mbarrier init (18 pipeline groups, 0 ordered-sequence groups, 79 barriers)
    // Mbarriers at smem_raw[0..632)

    if (warp == 0) {
        // q_full: 2 barriers, init_count=1
        // q_empty: 2 barriers, init_count=1
        // meta_full: 2 barriers, init_count=1
        // kv_raw_full: 32 barriers, init_count=1
        // --- pipeline 'raw_ring' ---
        // kv_raw_empty: 11 barriers, init_count=4
        // k16_full: 2 barriers, init_count=4
        // k16_empty: 2 barriers, init_count=1
        // v_full: 3 barriers, init_count=4
        // v_empty: 3 barriers, init_count=1
        // s_full: 2 barriers, init_count=1
        // s_empty: 2 barriers, init_count=4
        // p_full: 2 barriers, init_count=4
        // pv_done: 8 barriers, init_count=1
        // decode_done: 1 barriers, init_count=4
        // meta_empty: 2 barriers, init_count=4
        // part_ready: 1 barriers, init_count=1
        // q_dead: 1 barriers, init_count=2
        // tmem_dead: 1 barriers, init_count=4
        // Warp-cooperative initialization in physical record order.
        mbarrier_init(smem + 0 + lane * 8, 1);
        uint32_t _mbarrier_init_count_0_32 = 4;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(29), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(24), "r"((uint32_t)(4)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(21), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(19), "r"((uint32_t)(4)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(6), "r"((uint32_t)(1)));
        mbarrier_init(smem + 256 + lane * 8, _mbarrier_init_count_0_32);
        uint32_t _mbarrier_init_count_0_64 = 4;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_64) : "r"(lane), "n"(14), "r"((uint32_t)(2)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_64) : "r"(lane), "n"(13), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_64) : "r"(lane), "n"(12), "r"((uint32_t)(4)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_64) : "r"(lane), "n"(9), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_64) : "r"(lane), "n"(1), "r"((uint32_t)(4)));
        if (lane < 15) {
            mbarrier_init(smem + 512 + lane * 8, _mbarrier_init_count_0_64);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    __syncwarp();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 632);
    if (warp == 0) {
        int _tmem_hold = smem + 632;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    // barrier.cluster.wait deferred to the role entries listed in WarpConfig.cluster_init_wait_warps

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_s = taddr;
    const int tmem_tmem_k16 = taddr + 64;
    const int tmem_tmem_o = taddr + 192;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
    }

    // ---- Role: softmax ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 168;");
        // Deferred post-initialization cluster wait (WarpConfig.cluster_init_wait_warps)
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        { // softmax_main
            float tau = 8.0f;
            unsigned int total_items = total_q * num_kv_heads * 2;
            const int warp_in_role = warp % 4;
            const int token = warp_in_role * 32 + lane;
            const int tmem_lane_base = warp % 4 * 32 << 16;
            int item_stage = 0;
            int item_phase = 0;
            int s_stage = 0;
            int s_phase = 0;
            int p_stage = 0;
            int item_ctr = 0;
            float sv0[16];
            float sv1[16];
            float nv0[16];
            float nv1[16];
            #pragma unroll 1
            for (unsigned int work_idx = blockIdx.x; work_idx < total_items; work_idx += gridDim.x) {
                int item = work_idx / 2;
                int split = work_idx % 2;
                int query = item / num_kv_heads;
                int kv_head = item % num_kv_heads;
                int group_size = num_q_heads / num_kv_heads;
                mbarrier_wait(meta_full_addr + (item_stage) * 8, item_phase);
                int npairs = smem_npairs[item_stage];
                if (warp_in_role == 0) {
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(part_ready_addr, 8320);
                    }
                }
                int meta_base = item_stage * 16 + 8 * split;
                int item_par = item_stage;
                item_stage += 1;
                if (item_stage == 2) { item_stage = 0; item_phase ^= 1; }
                float origin[16];
                float neg_ms[16];
                float lane_sum[16];
                #pragma unroll
                for (int c = 0; c < 16; c++) {
                    origin[c] = -CAKE_INF;
                    neg_ms[c] = 0.0f;
                    lane_sum[c] = 0.0f;
                }
                int origins_set = 0;
                if (npairs > 0) {
                    mbarrier_wait(s_full_addr + (s_stage) * 8, s_phase);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int s_addr0 = taddr + (unsigned int)(s_stage * 32) + (unsigned int)tmem_lane_base;
                    tmem_ld_x16(&sv0[0], s_addr0);
                    tmem_ld_x16(&sv1[0], s_addr0 + 16);
                }
                #pragma unroll 1
                for (int p = 0; p < 8; p++) {
                    if (npairs <= p) {
                        break;
                    }
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(s_empty_addr + (s_stage) * 8);
                    }
                    s_stage += 1;
                    if (s_stage == 2) { s_stage = 0; s_phase ^= 1; }
                    int valid0 = smem_valid[meta_base + 2 * p];
                    int valid1 = smem_valid[meta_base + 2 * p + 1];
                    if (token >= valid0) {
                        #pragma unroll
                        for (int c_1 = 0; c_1 < 16; c_1++) {
                            sv0[c_1] = -CAKE_INF;
                        }
                    }
                    if (token >= valid1) {
                        #pragma unroll
                        for (int c_2 = 0; c_2 < 16; c_2++) {
                            sv1[c_2] = -CAKE_INF;
                        }
                    }
                    if (origins_set == 0) {
                        float wmax[16];
                        #pragma unroll
                        for (int c_3 = 0; c_3 < 16; c_3++) {
                            float _max_0 = max_noftz(sv0[c_3], sv1[c_3]);
                            float _warp_redux_f32_0;
                            asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_f32_0) : "f"(_max_0));
                            wmax[c_3] = _warp_redux_f32_0;
                        }
                        if (wmax[0] > -CAKE_INF) {
                            #pragma unroll
                            for (int c_4 = 0; c_4 < 16; c_4++) {
                                origin[c_4] = wmax[c_4];
                                neg_ms[c_4] = (-wmax[c_4]) * softmax_scale_log2;
                            }
                            origins_set = 1;
                        }
                    }
                    if (p >= 2) {
                        mbarrier_wait(pv_done_addr + (p - 2) * 8, item_par);
                    }
                    #pragma unroll
                    for (int c_5 = 0; c_5 < 16; c_5++) {
                        float _fma_0 = __fmaf_rn(sv0[c_5], softmax_scale_log2, neg_ms[c_5]);
                        float _exp2_0 = approx_exp2(_fma_0);
                        sv0[c_5] = _exp2_0;
                        float _fma_1 = __fmaf_rn(sv1[c_5], softmax_scale_log2, neg_ms[c_5]);
                        float _exp2_1 = approx_exp2(_fma_1);
                        sv1[c_5] = _exp2_1;
                        lane_sum[c_5] = lane_sum[c_5] + sv0[c_5] + sv1[c_5];
                    }
                    int p0_base = smem_p_addr + (unsigned int)(p_stage * 2 * 4096);
                    int p1_base = smem_p_addr + (unsigned int)((p_stage * 2 + 1) * 4096);
                    unsigned int x0[8];
                    unsigned int x1[8];
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sv0[_lp*2 + 0], sv0[_lp*2+1 + 0]));
                        x0[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 8; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sv1[_lp*2 + 0], sv1[_lp*2+1 + 0]));
                        x1[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p0_base + (token * 32 ^ (token * 32 >> 7 & 1) << 4))), "r"(x0[0]), "r"(x0[1]), "r"(x0[2]), "r"(x0[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p0_base + (token * 32 + 16 ^ (token * 32 + 16 >> 7 & 1) << 4))), "r"(x0[4]), "r"(x0[5]), "r"(x0[6]), "r"(x0[7]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p1_base + (token * 32 ^ (token * 32 >> 7 & 1) << 4))), "r"(x1[0]), "r"(x1[1]), "r"(x1[2]), "r"(x1[3]) : "memory");
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p1_base + (token * 32 + 16 ^ (token * 32 + 16 >> 7 & 1) << 4))), "r"(x1[4]), "r"(x1[5]), "r"(x1[6]), "r"(x1[7]) : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(p_full_addr + (p_stage) * 8);
                    }
                    p_stage += 1;
                    if (p_stage == 2) { p_stage = 0; }
                    if (npairs > p + 1) {
                        mbarrier_wait(s_full_addr + (s_stage) * 8, s_phase);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int s_addr_n = taddr + (unsigned int)(s_stage * 32) + (unsigned int)tmem_lane_base;
                        tmem_ld_x16(&nv0[0], s_addr_n);
                        tmem_ld_x16(&nv1[0], s_addr_n + 16);
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        #pragma unroll
                        for (int c_6 = 0; c_6 < 16; c_6++) {
                            sv0[c_6] = nv0[c_6];
                            sv1[c_6] = nv1[c_6];
                        }
                    }
                }
                __syncwarp();
                if (elect_sync()) {
                    mbarrier_arrive(meta_empty_addr + (item_par) * 8);
                }
                int b4 = lane / 16 % 2;
                int b3 = lane / 8 % 2;
                int b2 = lane / 4 % 2;
                int b1 = lane / 2 % 2;
                float r8[8];
                #pragma unroll
                for (int c_7 = 0; c_7 < 8; c_7++) {
                    float snd8 = ((b4 == 0) ? lane_sum[c_7 + 8] : lane_sum[c_7]);
                    float kp8 = ((b4 == 0) ? lane_sum[c_7] : lane_sum[c_7 + 8]);
                    float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, snd8, 16);
                    r8[c_7] = kp8 + _shfl_xor_0;
                }
                float r4[4];
                #pragma unroll
                for (int c_8 = 0; c_8 < 4; c_8++) {
                    float snd4 = ((b3 == 0) ? r8[c_8 + 4] : r8[c_8]);
                    float kp4 = ((b3 == 0) ? r8[c_8] : r8[c_8 + 4]);
                    float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, snd4, 8);
                    r4[c_8] = kp4 + _shfl_xor_1;
                }
                float r2[2];
                #pragma unroll
                for (int c_9 = 0; c_9 < 2; c_9++) {
                    float snd2 = ((b2 == 0) ? r4[c_9 + 2] : r4[c_9]);
                    float kp2 = ((b2 == 0) ? r4[c_9] : r4[c_9 + 2]);
                    float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, snd2, 4);
                    r2[c_9] = kp2 + _shfl_xor_2;
                }
                float snd1 = ((b1 == 0) ? r2[1] : r2[0]);
                float kp1 = ((b1 == 0) ? r2[0] : r2[1]);
                float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, snd1, 2);
                float r1 = kp1 + _shfl_xor_3;
                float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, r1, 1);
                r1 = r1 + _shfl_xor_4;
                float head_sum_w = r1;
                float orig_hold = -CAKE_INF;
                #pragma unroll
                for (int c_10 = 0; c_10 < 16; c_10++) {
                    if (lane == c_10) {
                        orig_hold = origin[c_10];
                    }
                }
                if (lane % 2 == 0) {
                    smem_f32[warp_in_role * 32 + lane / 2] = head_sum_w;
                }
                if (lane < 16) {
                    smem_f32[warp_in_role * 32 + 16 + lane] = orig_hold;
                }
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                float m_ab = -CAKE_INF;
                float wgt0 = 0.0f;
                float wgt1 = 0.0f;
                float wgt2 = 0.0f;
                float wgt3 = 0.0f;
                float head_sum = 0.0f;
                if (lane < 16) {
                    float og0 = smem_f32[16 + lane];
                    float og1 = smem_f32[48 + lane];
                    float og2 = smem_f32[80 + lane];
                    float og3 = smem_f32[112 + lane];
                    float _max_1 = max_noftz(og0, og1);
                    float _max_2 = max_noftz(og2, og3);
                    float _max_3 = max_noftz(_max_1, _max_2);
                    m_ab = _max_3;
                    float _exp2_2 = approx_exp2((og0 - m_ab) * softmax_scale_log2);
                    wgt0 = ((og0 > -CAKE_INF) ? _exp2_2 : 0.0f);
                    float _exp2_3 = approx_exp2((og1 - m_ab) * softmax_scale_log2);
                    wgt1 = ((og1 > -CAKE_INF) ? _exp2_3 : 0.0f);
                    float _exp2_4 = approx_exp2((og2 - m_ab) * softmax_scale_log2);
                    wgt2 = ((og2 > -CAKE_INF) ? _exp2_4 : 0.0f);
                    float _exp2_5 = approx_exp2((og3 - m_ab) * softmax_scale_log2);
                    wgt3 = ((og3 > -CAKE_INF) ? _exp2_5 : 0.0f);
                    head_sum = smem_f32[lane] * wgt0 + smem_f32[32 + lane] * wgt1 + smem_f32[64 + lane] * wgt2 + smem_f32[96 + lane] * wgt3;
                }
                if (npairs > 0) {
                    mbarrier_wait(pv_done_addr + (npairs - 1) * 8, item_par);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                }
                float o_epi[16];
                #pragma unroll
                for (int h = 0; h < 16; h++) {
                    o_epi[h] = 0.0f;
                }
                if (npairs > 0) {
                    #pragma unroll
                    for (int w = 0; w < 4; w++) {
                        float _tmem_load_0[16];
                        tmem_ld_x16(&_tmem_load_0[0], taddr + 192 + (unsigned int)(16 * w) + (unsigned int)tmem_lane_base);
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        float wsel = ((w == 0) ? wgt0 : ((w == 1) ? wgt1 : ((w == 2) ? wgt2 : wgt3)));
                        #pragma unroll
                        for (int h_1 = 0; h_1 < 16; h_1++) {
                            float _shfl_0;
                            asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_0) : "f"(wsel), "r"(h_1));
                            float _fma_2 = __fmaf_rn(_tmem_load_0[h_1], _shfl_0, o_epi[h_1]);
                            o_epi[h_1] = _fma_2;
                        }
                    }
                }
                if (total_items <= work_idx + (unsigned int)gridDim.x) {
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(tmem_dead_addr);
                    }
                }
                {
                    int rank_c = cta_rank;
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(decode_done_addr);
                    }
                    mbarrier_wait_cluster_hint(q_dead_addr, 0, 10000000);
                    #pragma unroll
                    for (int o = 0; o < 2; o++) {
                        uint32_t _mapa_0;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mapa_0) : "r"(smem_part_addr + (unsigned int)(4 * ((rank_c * 128 + token) * 8))), "r"(o));
                        uint32_t _mapa_1;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mapa_1) : "r"(part_ready_addr), "r"(o));
                        #pragma unroll
                        for (int c4 = 0; c4 < 2; c4++) {
                            asm volatile(
                                "st.async.weak.shared::cluster.mbarrier::complete_tx::bytes.v4.b32 [%0], {%1, %2, %3, %4}, [%5];"
                                :: "r"(_mapa_0 + (unsigned int)(16 * c4)), "r"(__float_as_uint(o_epi[o * 8 + 4 * c4])), "r"(__float_as_uint(o_epi[o * 8 + 4 * c4 + 1])), "r"(__float_as_uint(o_epi[o * 8 + 4 * c4 + 2])), "r"(__float_as_uint(o_epi[o * 8 + 4 * c4 + 3])), "r"(_mapa_1) : "memory");
                        }
                    }
                    if (warp_in_role == 0) {
                        if (lane < 16) {
                            int o_st = lane / 8;
                            int hl_st = lane % 8;
                            uint32_t _mapa_2;
                            asm volatile(
                                "mapa.shared::cluster.u32 %0, %1, %2;"
                                : "=r"(_mapa_2) : "r"(smem_pstat_addr + (unsigned int)(4 * ((rank_c * 8 + hl_st) * 2))), "r"(o_st));
                            uint32_t _mapa_3;
                            asm volatile(
                                "mapa.shared::cluster.u32 %0, %1, %2;"
                                : "=r"(_mapa_3) : "r"(part_ready_addr), "r"(o_st));
                            asm volatile(
                                "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];"
                                :: "r"(_mapa_2), "f"(m_ab), "r"(_mapa_3) : "memory");
                            asm volatile(
                                "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];"
                                :: "r"(_mapa_2 + 4), "f"(head_sum), "r"(_mapa_3) : "memory");
                        }
                    }
                    mbarrier_wait_cluster_hint(part_ready_addr, 0, 10000000);
                    int tid128 = warp_in_role * 32 + lane;
                    int hl = tid128 % 8;
                    int g = tid128 / 8 * 2;
                    int m_head = rank_c * 8 + hl;
                    if (m_head < group_size) {
                        float st_m[2];
                        float st_d[2];
                        float merge_m = -CAKE_INF;
                        #pragma unroll
                        for (int rs = 0; rs < 2; rs++) {
                            st_m[rs] = smem_pstat[(rs * 8 + hl) * 2];
                            st_d[rs] = smem_pstat[(rs * 8 + hl) * 2 + 1];
                            float _max_4 = max_noftz(merge_m, st_m[rs]);
                            merge_m = _max_4;
                        }
                        float merge_d = 0.0f;
                        float merge_w[2];
                        #pragma unroll
                        for (int rs_1 = 0; rs_1 < 2; rs_1++) {
                            float _exp2_6 = approx_exp2(softmax_scale_log2 * (st_m[rs_1] - merge_m));
                            merge_w[rs_1] = ((st_m[rs_1] == -CAKE_INF) ? 0.0f : _exp2_6);
                            float _fma_3 = __fmaf_rn(st_d[rs_1], merge_w[rs_1], merge_d);
                            merge_d = _fma_3;
                        }
                        float _rcp_1 = approx_rcp(merge_d);
                        float merge_inv = ((merge_d > 0.0f) ? output_scale * _rcp_1 : 0.0f);
                        int m_row = query * num_q_heads + kv_head * group_size + m_head;
                        float merge_o[8];
                        #pragma unroll
                        for (int e = 0; e < 8; e++) {
                            merge_o[e] = 0.0f;
                        }
                        #pragma unroll
                        for (int rs_2 = 0; rs_2 < 2; rs_2++) {
                            #pragma unroll
                            for (int ep = 0; ep < 4; ep++) {
                                float _fma_4 = __fmaf_rn(smem_part[(rs_2 * 128 + g + ep * 32) * 8 + hl], merge_w[rs_2], merge_o[2 * ep]);
                                merge_o[2 * ep] = _fma_4;
                                float _fma_5 = __fmaf_rn(smem_part[(rs_2 * 128 + g + ep * 32 + 1) * 8 + hl], merge_w[rs_2], merge_o[2 * ep + 1]);
                                merge_o[2 * ep + 1] = _fma_5;
                            }
                        }
                        #pragma unroll
                        for (int ep_1 = 0; ep_1 < 4; ep_1++) {
                            {
                                const float2 _prescale2_0 = {merge_inv, merge_inv};
                                #if __CUDA_ARCH__ >= 1000
                                #pragma unroll
                                for (int _ps = 0; _ps < 1; _ps++)
                                    mul_f32x2_inplace(&reinterpret_cast<float2*>(&merge_o[2 * ep_1])[_ps], _prescale2_0);
                                #else
                                #pragma unroll
                                for (int _ps = 0; _ps < 2; _ps++)
                                    merge_o[2 * ep_1 + _ps] *= merge_inv;
                                #endif
                                __nv_bfloat162 _pk = __floats2bfloat162_rn(merge_o[2 * ep_1 + 0], merge_o[2 * ep_1 + 1]);
                                *reinterpret_cast<__nv_bfloat162*>(&((__nv_bfloat16*)(O + (m_row * 128 + g + ep_1 * 32)))[0]) = _pk;
                            }
                        }
                        if (g == 0) {
                            float merge_lse = -CAKE_INF;
                            if (merge_d > 0.0f) {
                                float _log2_1;
                                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_1) : "f"(merge_d));
                                merge_lse = merge_m * softmax_scale_log2 * 0.6931471805599453f + _log2_1 * 0.6931471805599453f;
                            }
                            *(reinterpret_cast<float*>(msa_lse + m_row) + (0)) = merge_lse;
                        }
                    }
                }
                item_ctr = item_ctr + 1;
            }
        }
    }
    // ---- Role: transform_k ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 80;");
        { // transform_k_main
            int role_tid = warp % 4 * 32 + lane;
            unsigned int total_items_k = total_q * num_kv_heads * 2;
            int item_stage_k = 0;
            int item_phase_k = 0;
            int kt_k = 0;
            int tile_base_k = 0;
            int item_ctr_k = 0;
            #pragma unroll 1
            for (unsigned int work_idx_k = blockIdx.x; work_idx_k < total_items_k; work_idx_k += gridDim.x) {
                mbarrier_wait(meta_full_addr + (item_stage_k) * 8, item_phase_k);
                int npairs_k = smem_npairs[item_stage_k];
                item_stage_k += 1;
                if (item_stage_k == 2) { item_stage_k = 0; item_phase_k ^= 1; }
                #pragma unroll 1
                for (int sp = 0; sp < 8; sp++) {
                    if (npairs_k <= sp) {
                        break;
                    }
                    int kl_k = 2 * sp;
                    if (sp >= 2) {
                        kl_k = 4 * sp - 4;
                    }
                    {
                        #pragma unroll 1
                        for (int jv = 0; jv < 2; jv++) {
                            int st = (tile_base_k + kl_k + jv) % 11;
                            int fst = kt_k % 16;
                            int fph = kt_k / 16 & 1;
                            int slot = kt_k % 2;
                            int use = kt_k / 2 & 1;
                            mbarrier_wait(kv_raw_full_addr + (fst) * 8, fph);
                            mbarrier_wait(k16_empty_addr + (slot) * 8, 1 - use);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int t = role_tid;
                            int row_base = smem_raw_addr + (unsigned int)(st * 8192) + (unsigned int)(t * 64);
                            int swz = t >> 1 & 3;
                            const int lane_base = warp % 4 * 32 << 16;
                            int dst_base = taddr + 64 + (unsigned int)(slot * 64) + (unsigned int)lane_base;
                            {
                                unsigned int sw[2];
                                asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&sw[0])), "=r"(*reinterpret_cast<uint32_t*>(&sw[(0) + 1]))
                                    : "r"(smem_scale_addr + (unsigned int)(st * 1024) + (unsigned int)(t * 8)));
                                unsigned int sc[8];
                                #pragma unroll
                                for (int q = 0; q < 4; q++) {
                                    unsigned int pair = sw[q >> 1] >> (unsigned int)(16 * (q & 1)) & 65535;
                                    uint32_t _fp8_bf16x2_0;
                                    {
                                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_fp8_bf16x2_0) : "h"((uint16_t)(pair)));
                                    #else
                                    uint32_t _f16x2;
                                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"((uint16_t)(pair)));
                                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                                    float _f0;
                                    float _f1;
                                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp8_bf16x2_0) : "f"(_f1), "f"(_f0));
                                    #endif
                                    }
                                    unsigned int sf = _fp8_bf16x2_0;
                                    uint32_t _prmt_b32_0;
                                    asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(_prmt_b32_0) : "r"(sf), "r"(sf));
                                    sc[2 * q] = _prmt_b32_0;
                                    uint32_t _prmt_b32_1;
                                    asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_b32_1) : "r"(sf), "r"(sf));
                                    sc[2 * q + 1] = _prmt_b32_1;
                                }
                                unsigned int raw[16];
                                #pragma unroll
                                for (int q_1 = 0; q_1 < 4; q_1++) {
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&raw[4 * q_1])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(4 * q_1) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(4 * q_1) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw[(4 * q_1) + 3]))
                                        : "r"(row_base + ((q_1 ^ swz) << 4)));
                                }
                                #pragma unroll
                                for (int q_2 = 0; q_2 < 4; q_2++) {
                                    #pragma unroll
                                    for (int h_2 = 0; h_2 < 2; h_2++) {
                                        unsigned int w_1[8];
                                        #pragma unroll
                                        for (int i = 0; i < 2; i++) {
                                            uint32_t _fp4_bf16x8_0[4];
                                            {
                                            #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                            asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                                                "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                                                "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                                                "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                                                "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                                                "}"
                                                : "=r"(_fp4_bf16x8_0[0]), "=r"(_fp4_bf16x8_0[1]), "=r"(_fp4_bf16x8_0[2]), "=r"(_fp4_bf16x8_0[3]) : "r"((uint32_t)(raw[4 * q_2 + 2 * h_2 + i])));
                                            #else
                                            asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                                                "cvt.rn.f16x2.e2m1x2 %0, b0;"
                                                "cvt.rn.f16x2.e2m1x2 %1, b1;"
                                                "cvt.rn.f16x2.e2m1x2 %2, b2;"
                                                "cvt.rn.f16x2.e2m1x2 %3, b3;"
                                                "}"
                                                : "=r"(_fp4_bf16x8_0[0]), "=r"(_fp4_bf16x8_0[1]), "=r"(_fp4_bf16x8_0[2]), "=r"(_fp4_bf16x8_0[3]) : "r"((uint32_t)(raw[4 * q_2 + 2 * h_2 + i])));
                                            {
                                            uint16_t lo = (uint16_t)_fp4_bf16x8_0[0], hi = (uint16_t)(_fp4_bf16x8_0[0] >> 16);
                                            float f0, f1;
                                            asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                                            asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                                            asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_0[0]) : "f"(f1), "f"(f0));
                                            }
                                            {
                                            uint16_t lo = (uint16_t)_fp4_bf16x8_0[1], hi = (uint16_t)(_fp4_bf16x8_0[1] >> 16);
                                            float f0, f1;
                                            asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                                            asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                                            asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_0[1]) : "f"(f1), "f"(f0));
                                            }
                                            {
                                            uint16_t lo = (uint16_t)_fp4_bf16x8_0[2], hi = (uint16_t)(_fp4_bf16x8_0[2] >> 16);
                                            float f0, f1;
                                            asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                                            asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                                            asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_0[2]) : "f"(f1), "f"(f0));
                                            }
                                            {
                                            uint16_t lo = (uint16_t)_fp4_bf16x8_0[3], hi = (uint16_t)(_fp4_bf16x8_0[3] >> 16);
                                            float f0, f1;
                                            asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                                            asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                                            asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_0[3]) : "f"(f1), "f"(f0));
                                            }
                                            #endif
                                            }
                                            #pragma unroll
                                            for (int j = 0; j < 4; j++) {
                                                uint32_t _bf16x2_mul_0;
                                                asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_0) : "r"(_fp4_bf16x8_0[j]), "r"(sc[2 * q_2 + h_2]));
                                                w_1[4 * i + j] = _bf16x2_mul_0;
                                            }
                                        }
                                        asm volatile(
                                            "tcgen05.st.sync.aligned.32x32b.x8.b32"
                                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                            :: "r"(dst_base + 16 * q_2 + 8 * h_2), "r"(*reinterpret_cast<const uint32_t*>(&w_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&w_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&w_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&w_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&w_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&w_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&w_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&w_1[7])));
                                    }
                                }
                            }
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            asm volatile("tcgen05.fence::before_thread_sync;");
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            __syncwarp();
                            if (elect_sync()) {
                                mbarrier_arrive(k16_full_addr + (slot) * 8);
                                mbarrier_arrive(kv_raw_empty_addr + (st) * 8);
                            }
                            kt_k = kt_k + 1;
                        }
                    }
                }
                tile_base_k = tile_base_k + 4 * npairs_k;
                item_ctr_k = item_ctr_k + 1;
            }
        }
    }
    // ---- Role: transform_v ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 80;");
        { // transform_v_main
            int role_tid_v = warp % 4 * 32 + lane;
            unsigned int total_items_t = total_q * num_kv_heads * 2;
            int item_stage_t = 0;
            int item_phase_t = 0;
            int tile_n = 0;
            int vtile_t = 0;
            int item_ctr_t = 0;
            int xf_ctr = 0;
            #pragma unroll 1
            for (unsigned int work_idx_t = blockIdx.x; work_idx_t < total_items_t; work_idx_t += gridDim.x) {
                mbarrier_wait(meta_full_addr + (item_stage_t) * 8, item_phase_t);
                int npairs_t = smem_npairs[item_stage_t];
                item_stage_t += 1;
                if (item_stage_t == 2) { item_stage_t = 0; item_phase_t ^= 1; }
                xf_ctr = 0;
                #pragma unroll 1
                for (int pair_1 = 0; pair_1 < 8; pair_1++) {
                    if (npairs_t <= pair_1) {
                        break;
                    }
                    if (pair_1 == 0) {
                        tile_n = tile_n + 2;
                        if (npairs_t > 1) {
                            tile_n = tile_n + 2;
                        }
                    }
                    if (npairs_t > pair_1 + 2) {
                        tile_n = tile_n + 2;
                    }
                    {
                        #pragma unroll 1
                        for (int v_tile_r = 0; v_tile_r < 2; v_tile_r++) {
                            int st_1 = tile_n % 11;
                            int fst_1 = 16 + vtile_t % 16;
                            int fph_1 = vtile_t / 16 & 1;
                            int vs = vtile_t % 3;
                            int vph = 1 - (vtile_t / 3 & 1);
                            mbarrier_wait(kv_raw_full_addr + (fst_1) * 8, fph_1);
                            mbarrier_wait(v_empty_addr + (vs) * 8, vph);
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            int t_1 = role_tid_v;
                            int row_base_1 = smem_raw_addr + (unsigned int)(st_1 * 8192) + (unsigned int)(t_1 * 64);
                            int swz_1 = t_1 >> 1 & 3;
                            int tm7 = t_1 & 7;
                            int dst_row = smem_v_addr + (unsigned int)(vs * 32768) + (unsigned int)((t_1 >> 3) * 1024) + (unsigned int)(tm7 * 128);
                            unsigned int sel = (t_1 & 3) * 17 + 64;
                            int scale_row = smem_scale_addr + (unsigned int)(st_1 * 1024) + (unsigned int)((t_1 >> 2) * 32);
                            {
                                unsigned int sw_1[8];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&sw_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&sw_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sw_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sw_1[(0) + 3]))
                                    : "r"(scale_row));
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&sw_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&sw_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&sw_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&sw_1[(4) + 3]))
                                    : "r"(scale_row + 16));
                                unsigned int sc_1[8];
                                #pragma unroll
                                for (int q_3 = 0; q_3 < 4; q_3++) {
                                    uint32_t _prmt_b32_2;
                                    asm("prmt.b32 %0, %1, %2, %3;" : "=r"(_prmt_b32_2) : "r"(sw_1[2 * q_3]), "r"(sw_1[2 * q_3 + 1]), "r"(sel));
                                    unsigned int pair_0 = _prmt_b32_2;
                                    uint32_t _fp8_bf16x2_1;
                                    {
                                    #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                    asm volatile("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(_fp8_bf16x2_1) : "h"((uint16_t)(pair_0)));
                                    #else
                                    uint32_t _f16x2;
                                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2) : "h"((uint16_t)(pair_0)));
                                    uint16_t _h0 = (uint16_t)(_f16x2 & 0xFFFFu);
                                    uint16_t _h1 = (uint16_t)((_f16x2 >> 16) & 0xFFFFu);
                                    float _f0;
                                    float _f1;
                                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f0) : "h"(_h0));
                                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_f1) : "h"(_h1));
                                    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp8_bf16x2_1) : "f"(_f1), "f"(_f0));
                                    #endif
                                    }
                                    unsigned int sf_1 = _fp8_bf16x2_1;
                                    uint32_t _prmt_b32_3;
                                    asm("prmt.b32 %0, %1, %2, 0x1010;" : "=r"(_prmt_b32_3) : "r"(sf_1), "r"(sf_1));
                                    sc_1[2 * q_3] = _prmt_b32_3;
                                    uint32_t _prmt_b32_4;
                                    asm("prmt.b32 %0, %1, %2, 0x3232;" : "=r"(_prmt_b32_4) : "r"(sf_1), "r"(sf_1));
                                    sc_1[2 * q_3 + 1] = _prmt_b32_4;
                                }
                                unsigned int raw_1[16];
                                #pragma unroll
                                for (int q_4 = 0; q_4 < 4; q_4++) {
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&raw_1[4 * q_4])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1[(4 * q_4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1[(4 * q_4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_1[(4 * q_4) + 3]))
                                        : "r"(row_base_1 + ((q_4 ^ swz_1) << 4)));
                                }
                                #pragma unroll
                                for (int q_5 = 0; q_5 < 4; q_5++) {
                                    int plane = dst_row + (q_5 >> 1) * 16384;
                                    #pragma unroll
                                    for (int i_1 = 0; i_1 < 4; i_1++) {
                                        uint32_t _fp4_bf16x8_1[4];
                                        {
                                        #if defined(__CUDACC_VER_MAJOR__) && (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2)) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
                                        asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                                            "cvt.rn.bf16x2.e2m1x2 %0, b0;"
                                            "cvt.rn.bf16x2.e2m1x2 %1, b1;"
                                            "cvt.rn.bf16x2.e2m1x2 %2, b2;"
                                            "cvt.rn.bf16x2.e2m1x2 %3, b3;"
                                            "}"
                                            : "=r"(_fp4_bf16x8_1[0]), "=r"(_fp4_bf16x8_1[1]), "=r"(_fp4_bf16x8_1[2]), "=r"(_fp4_bf16x8_1[3]) : "r"((uint32_t)(raw_1[4 * q_5 + i_1])));
                                        #else
                                        asm("{ .reg .b8 b0,b1,b2,b3; mov.b32 {b0,b1,b2,b3}, %4;"
                                            "cvt.rn.f16x2.e2m1x2 %0, b0;"
                                            "cvt.rn.f16x2.e2m1x2 %1, b1;"
                                            "cvt.rn.f16x2.e2m1x2 %2, b2;"
                                            "cvt.rn.f16x2.e2m1x2 %3, b3;"
                                            "}"
                                            : "=r"(_fp4_bf16x8_1[0]), "=r"(_fp4_bf16x8_1[1]), "=r"(_fp4_bf16x8_1[2]), "=r"(_fp4_bf16x8_1[3]) : "r"((uint32_t)(raw_1[4 * q_5 + i_1])));
                                        {
                                        uint16_t lo = (uint16_t)_fp4_bf16x8_1[0], hi = (uint16_t)(_fp4_bf16x8_1[0] >> 16);
                                        float f0, f1;
                                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_1[0]) : "f"(f1), "f"(f0));
                                        }
                                        {
                                        uint16_t lo = (uint16_t)_fp4_bf16x8_1[1], hi = (uint16_t)(_fp4_bf16x8_1[1] >> 16);
                                        float f0, f1;
                                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_1[1]) : "f"(f1), "f"(f0));
                                        }
                                        {
                                        uint16_t lo = (uint16_t)_fp4_bf16x8_1[2], hi = (uint16_t)(_fp4_bf16x8_1[2] >> 16);
                                        float f0, f1;
                                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_1[2]) : "f"(f1), "f"(f0));
                                        }
                                        {
                                        uint16_t lo = (uint16_t)_fp4_bf16x8_1[3], hi = (uint16_t)(_fp4_bf16x8_1[3] >> 16);
                                        float f0, f1;
                                        asm("cvt.f32.f16 %0, %1;" : "=f"(f0) : "h"(lo));
                                        asm("cvt.f32.f16 %0, %1;" : "=f"(f1) : "h"(hi));
                                        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(_fp4_bf16x8_1[3]) : "f"(f1), "f"(f0));
                                        }
                                        #endif
                                        }
                                        unsigned int w_2[4];
                                        #pragma unroll
                                        for (int j_1 = 0; j_1 < 4; j_1++) {
                                            uint32_t _bf16x2_mul_1;
                                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_1) : "r"(_fp4_bf16x8_1[j_1]), "r"(sc_1[2 * q_5 + (i_1 >> 1)]));
                                            w_2[j_1] = _bf16x2_mul_1;
                                        }
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                            "r"(plane + ((4 * (q_5 & 1) + i_1 ^ tm7) << 4)), "r"(*reinterpret_cast<uint32_t*>(&w_2[0])), "r"(*reinterpret_cast<uint32_t*>(&w_2[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&w_2[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&w_2[(0) + 3])));
                                    }
                                }
                            }
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            __syncwarp();
                            if (elect_sync()) {
                                mbarrier_arrive(v_full_addr + (vs) * 8);
                                mbarrier_arrive(kv_raw_empty_addr + (st_1) * 8);
                            }
                            tile_n = tile_n + 1;
                            vtile_t = vtile_t + 1;
                        }
                    }
                }
                item_ctr_t = item_ctr_t + 1;
            }
        }
    }
    // ---- Role: producer ----
    if (warp == 12) {
        { // producer_main
            unsigned int total_items_l = total_q * num_kv_heads * 2;
            int load_stage = 0;
            int load_phase = 1;
            int kt_l = 0;
            int vt_l = 0;
            int item_par_1 = 0;
            int me_phase = 1;
            int item_ctr_l = 0;
            int cur_page = 0;
            int cur_valid = 0;
            unsigned int first_idx = blockIdx.x;
            if (first_idx < total_items_l) {
                int item_f = first_idx / 2;
                int query_f = item_f / num_kv_heads;
                int kv_head_f = item_f % num_kv_heads;
                if (lane < 16) {
                    int tok_f = 0;
                    int batch = query_f / seqlen_q;
                    int query_in_batch = query_f - batch * seqlen_q;
                    int selected_block = task_kind[(kv_head_f * total_q + query_f) * 16 + lane];
                    int kv_len = task_kv_head[batch];
                    int valid_cols = 0;
                    if (selected_block >= 0) {
                        int block_start = selected_block * 128;
                        valid_cols = kv_len - block_start;
                        if (valid_cols > 128) {
                            valid_cols = 128;
                        }
                        if (valid_cols < 0) {
                            valid_cols = 0;
                        }
                        {
                            int query_position = kv_len - seqlen_q + query_in_batch;
                            int causal_cols = query_position - block_start + 1;
                            if (valid_cols > causal_cols) {
                                valid_cols = causal_cols;
                            }
                            if (valid_cols < 0) {
                                valid_cols = 0;
                            }
                        }
                    }
                    int token_base = 0;
                    int page_head = 0;
                    {
                        int physical_page = 0;
                        if (selected_block >= 0) {
                            physical_page = kv_indices[batch * msa_max_pages + selected_block];
                            if (physical_page < 0) {
                                valid_cols = 0;
                                physical_page = 0;
                            }
                        }
                        page_head = physical_page * num_kv_heads + kv_head_f;
                    }
                    tok_f = token_base;
                    cur_page = page_head;
                    cur_valid = valid_cols;
                }
            }
            #pragma unroll 1
            for (unsigned int work_idx_l = blockIdx.x; work_idx_l < total_items_l; work_idx_l += gridDim.x) {
                int item_l = work_idx_l / 2;
                int split_l = work_idx_l % 2;
                int query_l = item_l / num_kv_heads;
                int kv_head_l = item_l % num_kv_heads;
                int group_size_l = num_q_heads / num_kv_heads;
                int slot_page_head = cur_page;
                int slot_valid = cur_valid;
                __syncwarp();
                mbarrier_wait(meta_empty_addr + (item_par_1) * 8, me_phase);
                int meta_base_l = item_par_1 * 16;
                unsigned int _vote_0 = __ballot_sync(0xFFFFFFFF, slot_valid > 0);
                unsigned int valid_mask = _vote_0;
                unsigned int lane_one = 1;
                unsigned int below_mask = (lane_one << (unsigned int)lane) - 1;
                int _popc_0 = __popc(valid_mask);
                int count_l = _popc_0;
                int _popc_1 = __popc(valid_mask & below_mask);
                int pos_l = _popc_1;
                if (slot_valid > 0) {
                    smem_page[meta_base_l + pos_l] = slot_page_head;
                    smem_valid[meta_base_l + pos_l] = slot_valid;
                    if (pos_l + 1 == count_l) {
                        if (count_l % 2 == 1) {
                            smem_page[meta_base_l + count_l] = slot_page_head;
                            smem_valid[meta_base_l + count_l] = 0;
                        }
                    }
                }
                int npairs_all = (count_l + count_l % 2) / 2;
                int npairs_l = npairs_all - split_l * 4;
                if (npairs_l > 4) {
                    npairs_l = 4;
                }
                if (npairs_l < 0) {
                    npairs_l = 0;
                }
                int slot0_l = meta_base_l + 8 * split_l;
                __syncwarp();
                if (elect_sync()) {
                    if (npairs_l > 0) {
                        #pragma unroll
                        for (int k_tile = 0; k_tile < 2; k_tile++) {
                            mbarrier_wait(kv_raw_empty_addr + (load_stage) * 8, load_phase);
                            mbarrier_arrive_expect_tx(kv_raw_full_addr + (kt_l % 16) * 8, 9216);
                            int physical_page_1 = smem_page[slot0_l + k_tile] / num_kv_heads;
                            int kv_head_1 = smem_page[slot0_l + k_tile] - physical_page_1 * num_kv_heads;
                            tma_4d_gmem2smem(smem_raw_addr + (unsigned int)(load_stage * 8192), (&K), 0, 0, kv_head_1, physical_page_1, kv_raw_full_addr + (kt_l % 16) * 8);
                            tma_4d_gmem2smem(smem_scale_addr + (unsigned int)(load_stage * 1024), (&K_scale), 0, 0, kv_head_1, physical_page_1, kv_raw_full_addr + (kt_l % 16) * 8);
                            load_stage += 1;
                            if (load_stage == 11) { load_stage = 0; load_phase ^= 1; }
                            kt_l = kt_l + 1;
                        }
                    }
                }
                mbarrier_wait(q_empty_addr + (item_par_1) * 8, me_phase);
                if (elect_sync()) {
                    int q_row_l = query_l * num_q_heads + kv_head_l * group_size_l;
                    mbarrier_arrive_expect_tx(q_full_addr + (item_par_1) * 8, 4096);
                    tma_3d_gmem2smem(smem_qt_addr + (unsigned int)(item_par_1 * 4096), (&Q), 0, q_row_l, 0, q_full_addr + (item_par_1) * 8);
                }
                if (elect_sync()) {
                    if (npairs_l > 1) {
                        #pragma unroll
                        for (int k_tile_1 = 0; k_tile_1 < 2; k_tile_1++) {
                            mbarrier_wait(kv_raw_empty_addr + (load_stage) * 8, load_phase);
                            mbarrier_arrive_expect_tx(kv_raw_full_addr + (kt_l % 16) * 8, 9216);
                            int physical_page_2 = smem_page[slot0_l + 2 + k_tile_1] / num_kv_heads;
                            int kv_head_2 = smem_page[slot0_l + 2 + k_tile_1] - physical_page_2 * num_kv_heads;
                            tma_4d_gmem2smem(smem_raw_addr + (unsigned int)(load_stage * 8192), (&K), 0, 0, kv_head_2, physical_page_2, kv_raw_full_addr + (kt_l % 16) * 8);
                            tma_4d_gmem2smem(smem_scale_addr + (unsigned int)(load_stage * 1024), (&K_scale), 0, 0, kv_head_2, physical_page_2, kv_raw_full_addr + (kt_l % 16) * 8);
                            load_stage += 1;
                            if (load_stage == 11) { load_stage = 0; load_phase ^= 1; }
                            kt_l = kt_l + 1;
                        }
                    }
                }
                if (elect_sync()) {
                    smem_npairs[item_par_1] = npairs_l;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(meta_full_addr + (item_par_1) * 8);
                }
                unsigned int nxt_idx = work_idx_l + (unsigned int)gridDim.x;
                cur_page = 0;
                cur_valid = 0;
                if (nxt_idx < total_items_l) {
                    int item_n = nxt_idx / 2;
                    int query_n = item_n / num_kv_heads;
                    int kv_head_n = item_n % num_kv_heads;
                    if (lane < 16) {
                        int tok_n = 0;
                        int batch_1 = query_n / seqlen_q;
                        int query_in_batch_1 = query_n - batch_1 * seqlen_q;
                        int selected_block_1 = task_kind[(kv_head_n * total_q + query_n) * 16 + lane];
                        int kv_len_1 = task_kv_head[batch_1];
                        int valid_cols_1 = 0;
                        if (selected_block_1 >= 0) {
                            int block_start_1 = selected_block_1 * 128;
                            valid_cols_1 = kv_len_1 - block_start_1;
                            if (valid_cols_1 > 128) {
                                valid_cols_1 = 128;
                            }
                            if (valid_cols_1 < 0) {
                                valid_cols_1 = 0;
                            }
                            {
                                int query_position_1 = kv_len_1 - seqlen_q + query_in_batch_1;
                                int causal_cols_1 = query_position_1 - block_start_1 + 1;
                                if (valid_cols_1 > causal_cols_1) {
                                    valid_cols_1 = causal_cols_1;
                                }
                                if (valid_cols_1 < 0) {
                                    valid_cols_1 = 0;
                                }
                            }
                        }
                        int token_base_1 = 0;
                        int page_head_1 = 0;
                        {
                            int physical_page_3 = 0;
                            if (selected_block_1 >= 0) {
                                physical_page_3 = kv_indices[batch_1 * msa_max_pages + selected_block_1];
                                if (physical_page_3 < 0) {
                                    valid_cols_1 = 0;
                                    physical_page_3 = 0;
                                }
                            }
                            page_head_1 = physical_page_3 * num_kv_heads + kv_head_n;
                        }
                        tok_n = token_base_1;
                        cur_page = page_head_1;
                        cur_valid = valid_cols_1;
                    }
                }
                __syncwarp();
                if (elect_sync()) {
                    #pragma unroll 1
                    for (int pair_2 = 0; pair_2 < 8; pair_2++) {
                        if (npairs_l <= pair_2) {
                            break;
                        }
                        if (npairs_l > pair_2 + 2) {
                            #pragma unroll
                            for (int k_tile_2 = 0; k_tile_2 < 2; k_tile_2++) {
                                mbarrier_wait(kv_raw_empty_addr + (load_stage) * 8, load_phase);
                                mbarrier_arrive_expect_tx(kv_raw_full_addr + (kt_l % 16) * 8, 9216);
                                int physical_page_4 = smem_page[slot0_l + pair_2 * 2 + 4 + k_tile_2] / num_kv_heads;
                                int kv_head_3 = smem_page[slot0_l + pair_2 * 2 + 4 + k_tile_2] - physical_page_4 * num_kv_heads;
                                tma_4d_gmem2smem(smem_raw_addr + (unsigned int)(load_stage * 8192), (&K), 0, 0, kv_head_3, physical_page_4, kv_raw_full_addr + (kt_l % 16) * 8);
                                tma_4d_gmem2smem(smem_scale_addr + (unsigned int)(load_stage * 1024), (&K_scale), 0, 0, kv_head_3, physical_page_4, kv_raw_full_addr + (kt_l % 16) * 8);
                                load_stage += 1;
                                if (load_stage == 11) { load_stage = 0; load_phase ^= 1; }
                                kt_l = kt_l + 1;
                            }
                        }
                        #pragma unroll
                        for (int v_tile = 0; v_tile < 2; v_tile++) {
                            mbarrier_wait(kv_raw_empty_addr + (load_stage) * 8, load_phase);
                            mbarrier_arrive_expect_tx(kv_raw_full_addr + (16 + vt_l % 16) * 8, 9216);
                            int physical_page_5 = smem_page[slot0_l + pair_2 * 2 + v_tile] / num_kv_heads;
                            int kv_head_4 = smem_page[slot0_l + pair_2 * 2 + v_tile] - physical_page_5 * num_kv_heads;
                            tma_4d_gmem2smem(smem_raw_addr + (unsigned int)(load_stage * 8192), (&V), 0, 0, kv_head_4, physical_page_5, kv_raw_full_addr + (16 + vt_l % 16) * 8);
                            tma_4d_gmem2smem(smem_scale_addr + (unsigned int)(load_stage * 1024), (&V_scale), 0, 0, kv_head_4, physical_page_5, kv_raw_full_addr + (16 + vt_l % 16) * 8);
                            load_stage += 1;
                            if (load_stage == 11) { load_stage = 0; load_phase ^= 1; }
                            vt_l = vt_l + 1;
                        }
                    }
                }
                item_par_1 = 1 - item_par_1;
                if (item_par_1 == 0) {
                    me_phase = 1 - me_phase;
                }
                item_ctr_l = item_ctr_l + 1;
            }
        }
    }
    // ---- Role: mma_s ----
    if (warp == 13) {
        // Deferred post-initialization cluster wait (WarpConfig.cluster_init_wait_warps)
        asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
        asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
        { // mma_s_main
            unsigned int total_items_s = total_q * num_kv_heads * 2;
            int item_stage_s = 0;
            int item_phase_s = 0;
            int s_stage_m = 0;
            int s_phase_m = 1;
            int ktile_n = 0;
            int item_ctr_s = 0;
            #pragma unroll 1
            for (unsigned int work_idx_s = blockIdx.x; work_idx_s < total_items_s; work_idx_s += gridDim.x) {
                mbarrier_wait(meta_full_addr + (item_stage_s) * 8, item_phase_s);
                int npairs_s = smem_npairs[item_stage_s];
                int q_buf_s = item_stage_s;
                int q_phase_s = item_phase_s;
                item_stage_s += 1;
                if (item_stage_s == 2) { item_stage_s = 0; item_phase_s ^= 1; }
                mbarrier_wait(q_full_addr + (q_buf_s) * 8, q_phase_s);
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (npairs_s == 0) {
                    if (elect_sync()) {
                        mbarrier_arrive(q_empty_addr + (q_buf_s) * 8);
                    }
                    elect_commit_cg1_multicast(q_dead_addr, (uint16_t)(3));
                }
                #pragma unroll 1
                for (int sp_1 = 0; sp_1 < 8; sp_1++) {
                    if (npairs_s <= sp_1) {
                        break;
                    }
                    mbarrier_wait(s_empty_addr + (s_stage_m) * 8, s_phase_m);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int slot_1 = ktile_n % 2;
                    int use_1 = ktile_n / 2 & 1;
                    mbarrier_wait(k16_full_addr + (slot_1) * 8, use_1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    {
                        int _mma_b_lo_0 = make_warp_uniform((((smem_qh_addr) >> 4) & 0x3FFF) + (2 * q_buf_s) * 128);
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
                    "mov.b32 id, 134481040;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_tmem_s + (s_stage_m * 32))), "r"(_mma_b_lo_0), "r"(tmem_tmem_k16 + 2 * slot_1 * 32), "r"(0));
                        int _mma_b_lo_1 = make_warp_uniform((((smem_qh_addr) >> 4) & 0x3FFF) + (2 * q_buf_s + 1) * 128);
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
                    "mov.b32 id, 134481040;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_tmem_s + (s_stage_m * 32))), "r"(_mma_b_lo_1), "r"(tmem_tmem_k16 + (2 * slot_1 + 1) * 32), "r"(1));
                    }
                    elect_commit(k16_empty_addr + (slot_1) * 8);
                    ktile_n = ktile_n + 1;
                    int slot_0 = ktile_n % 2;
                    int use_1_1 = ktile_n / 2 & 1;
                    mbarrier_wait(k16_full_addr + (slot_0) * 8, use_1_1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    {
                        int _mma_b_lo_2 = make_warp_uniform((((smem_qh_addr) >> 4) & 0x3FFF) + (2 * q_buf_s) * 128);
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
                    "mov.b32 id, 134481040;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_tmem_s + (s_stage_m * 32 + 16))), "r"(_mma_b_lo_2), "r"(tmem_tmem_k16 + 2 * slot_0 * 32), "r"(0));
                        int _mma_b_lo_3 = make_warp_uniform((((smem_qh_addr) >> 4) & 0x3FFF) + (2 * q_buf_s + 1) * 128);
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
                    "mov.b32 id, 134481040;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_tmem_s + (s_stage_m * 32 + 16))), "r"(_mma_b_lo_3), "r"(tmem_tmem_k16 + (2 * slot_0 + 1) * 32), "r"(1));
                    }
                    elect_commit(k16_empty_addr + (slot_0) * 8);
                    ktile_n = ktile_n + 1;
                    if (sp_1 == npairs_s - 1) {
                        elect_commit(q_empty_addr + (q_buf_s) * 8);
                    }
                    elect_commit(s_full_addr + (s_stage_m) * 8);
                    if (sp_1 == npairs_s - 1) {
                        elect_commit_cg1_multicast(q_dead_addr, (uint16_t)(3));
                    }
                    s_stage_m += 1;
                    if (s_stage_m == 2) { s_stage_m = 0; s_phase_m ^= 1; }
                }
                item_ctr_s = item_ctr_s + 1;
            }
        }
    }
    // ---- Role: mma_pv ----
    if (warp == 14) {
        { // mma_pv_main
            unsigned int total_items_v = total_q * num_kv_heads * 2;
            int item_stage_v = 0;
            int item_phase_v = 0;
            int pf_stage_v = 0;
            int pf_phase_v = 0;
            int vtile_n = 0;
            int have_prev_v = 0;
            int item_ctr_v = 0;
            unsigned int _phase_decode_done_0 = 0;
            #pragma unroll 1
            for (unsigned int work_idx_v = blockIdx.x; work_idx_v < total_items_v; work_idx_v += gridDim.x) {
                mbarrier_wait(meta_full_addr + (item_stage_v) * 8, item_phase_v);
                int npairs_v = smem_npairs[item_stage_v];
                item_stage_v += 1;
                if (item_stage_v == 2) { item_stage_v = 0; item_phase_v ^= 1; }
                int first_pv = 1;
                int dd_waited = 0;
                #pragma unroll 1
                for (int pv = 0; pv < 8; pv++) {
                    if (npairs_v <= pv) {
                        break;
                    }
                    mbarrier_wait(p_full_addr + (pf_stage_v) * 8, pf_phase_v);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (dd_waited == 0) {
                        if (have_prev_v != 0) {
                            mbarrier_wait(decode_done_addr, _phase_decode_done_0);
                            _phase_decode_done_0 ^= 1;
                            asm volatile("tcgen05.fence::after_thread_sync;");
                        }
                        dd_waited = 1;
                    }
                    int vt = vtile_n;
                    int vs_1 = vt % 3;
                    int vph_1 = vt / 3 & 1;
                    mbarrier_wait(v_full_addr + (vs_1) * 8, vph_1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int init_j = ((1) ? first_pv : 0);
                    {
                        int _mma_a_lo_4 = make_warp_uniform(((((smem_v_mn_addr) >> 4) & 0x3FFF) | 0x4000000) + (vs_1) * 2048);
                        int _mma_b_lo_4 = make_warp_uniform(((((smem_p_addr) >> 4) & 0x3FFF) | 0x1000000) + (pf_stage_v * 2) * 256);
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
                    "mov.b32 bdhi, 0xC0004010;\n\t"
                    "mov.b32 id, 134579344;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 32;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_4), "r"(_mma_b_lo_4), "r"(tmem_tmem_o), "r"(((init_j) ? 0 : 1)));
                        int _mma_a_lo_5 = make_warp_uniform(((((smem_v_mn_addr) >> 4) & 0x3FFF) | 0x4000000) + (vs_1) * 2048);
                        int _mma_b_lo_5 = make_warp_uniform(((((smem_p_addr) >> 4) & 0x3FFF) | 0x1000000) + (pf_stage_v * 2) * 256);
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
                    "mov.b32 bdhi, 0xC0004010;\n\t"
                    "mov.b32 id, 134579344;\n\t"
                    "add.u32 alo, %0, 256;\n\t"
                    "add.u32 blo, %1, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 32;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_5), "r"(_mma_b_lo_5), "r"((tmem_tmem_o + (16))), "r"(((init_j) ? 0 : 1)));
                        int _mma_a_lo_6 = make_warp_uniform(((((smem_v_mn_addr) >> 4) & 0x3FFF) | 0x4000000) + (vs_1) * 2048);
                        int _mma_b_lo_6 = make_warp_uniform(((((smem_p_addr) >> 4) & 0x3FFF) | 0x1000000) + (pf_stage_v * 2) * 256);
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
                    "mov.b32 bdhi, 0xC0004010;\n\t"
                    "mov.b32 id, 134579344;\n\t"
                    "add.u32 alo, %0, 512;\n\t"
                    "add.u32 blo, %1, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 32;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_6), "r"(_mma_b_lo_6), "r"((tmem_tmem_o + (32))), "r"(((init_j) ? 0 : 1)));
                        int _mma_a_lo_7 = make_warp_uniform(((((smem_v_mn_addr) >> 4) & 0x3FFF) | 0x4000000) + (vs_1) * 2048);
                        int _mma_b_lo_7 = make_warp_uniform(((((smem_p_addr) >> 4) & 0x3FFF) | 0x1000000) + (pf_stage_v * 2) * 256);
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
                    "mov.b32 bdhi, 0xC0004010;\n\t"
                    "mov.b32 id, 134579344;\n\t"
                    "add.u32 alo, %0, 768;\n\t"
                    "add.u32 blo, %1, 192;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 32;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_7), "r"(_mma_b_lo_7), "r"((tmem_tmem_o + (48))), "r"(((init_j) ? 0 : 1)));
                    }
                    elect_commit(v_empty_addr + (vs_1) * 8);
                    int vt_0 = vtile_n + 1;
                    int vs_1_1 = vt_0 % 3;
                    int vph_2 = vt_0 / 3 & 1;
                    mbarrier_wait(v_full_addr + (vs_1_1) * 8, vph_2);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int init_j_3 = ((0) ? first_pv : 0);
                    {
                        int _mma_a_lo_8 = make_warp_uniform(((((smem_v_mn_addr) >> 4) & 0x3FFF) | 0x4000000) + (vs_1_1) * 2048);
                        int _mma_b_lo_8 = make_warp_uniform(((((smem_p_addr) >> 4) & 0x3FFF) | 0x1000000) + (pf_stage_v * 2 + 1) * 256);
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
                    "mov.b32 bdhi, 0xC0004010;\n\t"
                    "mov.b32 id, 134579344;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 32;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_8), "r"(_mma_b_lo_8), "r"(tmem_tmem_o), "r"(((init_j_3) ? 0 : 1)));
                        int _mma_a_lo_9 = make_warp_uniform(((((smem_v_mn_addr) >> 4) & 0x3FFF) | 0x4000000) + (vs_1_1) * 2048);
                        int _mma_b_lo_9 = make_warp_uniform(((((smem_p_addr) >> 4) & 0x3FFF) | 0x1000000) + (pf_stage_v * 2 + 1) * 256);
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
                    "mov.b32 bdhi, 0xC0004010;\n\t"
                    "mov.b32 id, 134579344;\n\t"
                    "add.u32 alo, %0, 256;\n\t"
                    "add.u32 blo, %1, 64;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 32;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_9), "r"(_mma_b_lo_9), "r"((tmem_tmem_o + (16))), "r"(((init_j_3) ? 0 : 1)));
                        int _mma_a_lo_10 = make_warp_uniform(((((smem_v_mn_addr) >> 4) & 0x3FFF) | 0x4000000) + (vs_1_1) * 2048);
                        int _mma_b_lo_10 = make_warp_uniform(((((smem_p_addr) >> 4) & 0x3FFF) | 0x1000000) + (pf_stage_v * 2 + 1) * 256);
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
                    "mov.b32 bdhi, 0xC0004010;\n\t"
                    "mov.b32 id, 134579344;\n\t"
                    "add.u32 alo, %0, 512;\n\t"
                    "add.u32 blo, %1, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 32;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_10), "r"(_mma_b_lo_10), "r"((tmem_tmem_o + (32))), "r"(((init_j_3) ? 0 : 1)));
                        int _mma_a_lo_11 = make_warp_uniform(((((smem_v_mn_addr) >> 4) & 0x3FFF) | 0x4000000) + (vs_1_1) * 2048);
                        int _mma_b_lo_11 = make_warp_uniform(((((smem_p_addr) >> 4) & 0x3FFF) | 0x1000000) + (pf_stage_v * 2 + 1) * 256);
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
                    "mov.b32 bdhi, 0xC0004010;\n\t"
                    "mov.b32 id, 134579344;\n\t"
                    "add.u32 alo, %0, 768;\n\t"
                    "add.u32 blo, %1, 192;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 128;\n\t"
                    "add.u32 blo, blo, 32;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_11), "r"(_mma_b_lo_11), "r"((tmem_tmem_o + (48))), "r"(((init_j_3) ? 0 : 1)));
                    }
                    elect_commit(v_empty_addr + (vs_1_1) * 8);
                    first_pv = 0;
                    vtile_n = vtile_n + 2;
                    elect_commit(pv_done_addr + (pv) * 8);
                    pf_stage_v += 1;
                    if (pf_stage_v == 2) { pf_stage_v = 0; pf_phase_v ^= 1; }
                }
                if (dd_waited == 0) {
                    if (have_prev_v != 0) {
                        mbarrier_wait(decode_done_addr, _phase_decode_done_0);
                        _phase_decode_done_0 ^= 1;
                    }
                }
                #pragma unroll 1
                for (int spare = npairs_v; spare < 8; spare++) {
                    if (elect_sync()) {
                        mbarrier_arrive(pv_done_addr + (spare) * 8);
                    }
                }
                have_prev_v = 1;
                item_ctr_v = item_ctr_v + 1;
            }
        }
    }
    // ---- Role: idle ----
    if (warp == 15) {
        { // idle_main
            unsigned int _phase_tmem_dead_0 = 0;
            mbarrier_wait(tmem_dead_addr, _phase_tmem_dead_0);
            _phase_tmem_dead_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(256));
        }
    }

    // Cleanup
}

} // extern "C"
