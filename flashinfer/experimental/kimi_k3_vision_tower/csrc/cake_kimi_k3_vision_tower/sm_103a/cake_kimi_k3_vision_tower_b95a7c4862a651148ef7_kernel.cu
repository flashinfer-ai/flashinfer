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
#define TMEM_NCOLS 512
#define TMEM_SCORES_OFFSET 0
#define TMEM_PROBS_0_OFFSET 64
#define TMEM_PROBS_1_OFFSET 192
#define TMEM_OUTPUT_0_OFFSET 256
#define TMEM_OUTPUT_1_OFFSET 384
#define NUM_KV_STAGES 7
#define NUM_QP_STAGES 2
#define SMEM_SCALES_OFF 1024
#define SMEM_SCALES_STAGE_BYTES 2048
#define SMEM_SCALES_STRIDE 2048
#define SMEM_SMEM_QA_OFF 3072
#define SMEM_SMEM_QA_STAGE_BYTES 32768
#define SMEM_SMEM_QA_STRIDE 32768
#define SMEM_SMEM_KV_OFF 68608
#define SMEM_SMEM_KV_STAGE_BYTES 16384
#define SMEM_SMEM_KV_STRIDE 16384
#define SMEM_SMEM_V_OFF 68608
#define SMEM_SMEM_V_STAGE_BYTES 16384
#define SMEM_SMEM_V_STRIDE 16384
#define SMEM_SMEM_O_OFF 3072
#define SMEM_SMEM_O_STAGE_BYTES 16384
#define SMEM_SMEM_O_STRIDE 16384
#define SMEM_TOTAL 183296
#define USE_TMEM_LD_RED 1
#define BLOCK_M 128
#define BLOCK_N 128
#define HEAD_DIM 128

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


__device__ __forceinline__ void tcgen05_mma_f16_cta2(
    int taddr, uint64_t a_desc, uint64_t b_desc,
    uint32_t i_desc, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        ".reg .b32 m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\t"
        "mov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::f16 [%0], %1, %2, %3, {m0, m1, m2, m3, m4, m5, m6, m7}, p;\n\t"
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


__device__ __forceinline__ void tma_4d_gmem2smem_cta2(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_store_4d(
    const void *tmap, int x, int y, int z, int w, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3, %4}], [%5];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(w), "r"(smem_addr) : "memory");
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


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(512, 1) __cluster_dims__(2,1,1) void
kernel_cake_kimi_k3_vision_tower_b95a7c4862a651148ef7(const __grid_constant__ CUtensorMap Q, __nv_bfloat16* __restrict__ Q_raw, const __grid_constant__ CUtensorMap K, const __grid_constant__ CUtensorMap V, __nv_bfloat16* __restrict__ O, const __grid_constant__ CUtensorMap O_tma, int* __restrict__ seg_begin, int* __restrict__ seg_len, int* __restrict__ unit_table, unsigned long long* __restrict__ probe, unsigned int total_tiles, int num_heads, float softmax_scale_log2)
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
    #define kv_empty_addr (mbar_base + 88)
    #define s_full_addr (mbar_base + 144)
    #define p_full_addr (mbar_base + 160)
    #define p_full_2_addr (mbar_base + 176)
    #define corr_done_addr (mbar_base + 192)
    #define scale_full_addr (mbar_base + 208)
    #define o_full_addr (mbar_base + 224)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    float* scales = reinterpret_cast<float*>(smem_raw + 1024);
    const int scales_addr = smem + 1024;
    __nv_bfloat16* smem_qa = reinterpret_cast<__nv_bfloat16*>(smem_raw + 3072);
    const int smem_qa_addr = smem + 3072;
    __nv_bfloat16* smem_kv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 68608);
    const int smem_kv_addr = smem + 68608;
    __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 68608);
    const int smem_v_addr = smem + 68608;
    __nv_bfloat16* smem_o = reinterpret_cast<__nv_bfloat16*>(smem_raw + 3072);
    const int smem_o_addr = smem + 3072;

    // Mbarrier init (10 pipeline groups, 0 ordered-sequence groups, 30 barriers)
    // Mbarriers at smem_raw[0..240)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'qp' ---
            // q_full: 2 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            mbarrier_init(smem + 8, 2);
            // q_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // --- pipeline 'kv' ---
            // kv_full: 7 barriers, init_count=2
            mbarrier_init(smem + 32, 2);
            mbarrier_init(smem + 40, 2);
            mbarrier_init(smem + 48, 2);
            mbarrier_init(smem + 56, 2);
            mbarrier_init(smem + 64, 2);
            mbarrier_init(smem + 72, 2);
            mbarrier_init(smem + 80, 2);
            // kv_empty: 7 barriers, init_count=1
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            // p_full: 2 barriers, init_count=512
            mbarrier_init(smem + 160, 512);
            mbarrier_init(smem + 168, 512);
            // p_full_2: 2 barriers, init_count=256
            mbarrier_init(smem + 176, 256);
            mbarrier_init(smem + 184, 256);
            // corr_done: 2 barriers, init_count=128
            mbarrier_init(smem + 192, 128);
            mbarrier_init(smem + 200, 128);
            // scale_full: 2 barriers, init_count=128
            mbarrier_init(smem + 208, 128);
            mbarrier_init(smem + 216, 128);
            // o_full: 2 barriers, init_count=1
            mbarrier_init(smem + 224, 1);
            mbarrier_init(smem + 232, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 240);
    if (warp == 0) {
        int _tmem_hold = smem + 240;
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
    const int tmem_scores = taddr;
    const int tmem_probs_0 = taddr + 64;
    const int tmem_probs_1 = taddr + 192;
    const int tmem_output_0 = taddr + 256;
    const int tmem_output_1 = taddr + 384;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;");
    }

    // ---- Role: softmax ----
    if (warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 192;");
        { // softmax_main
            unsigned int stage = make_warp_uniform(warp / 4);
            uint32_t warp_group_idx = static_cast<uint32_t>(tid) / 128u;
            int sync_group = make_warp_uniform(warp_group_idx);
            int tmem_s_off = make_warp_uniform(stage * 128);
            int tmem_p_off = make_warp_uniform(stage * 128 + 64);
            int scale_off = make_warp_uniform(stage * (unsigned int)BLOCK_M);
            unsigned int q_slot_s = 0;
            unsigned int _phase_s_full = 0;
            unsigned int _phase_corr_done = 0;
            #pragma unroll 1
            for (unsigned int tile_idx = cluster_id; tile_idx < total_tiles; tile_idx += num_clusters) {
                int rec = tile_idx * 2;
                int seg = unit_table[rec];
                int packed = unit_table[rec + 1];
                int head = packed >> 16;
                int c = packed & 65535;
                int doc_begin = seg_begin[seg];
                int doc_len = seg_len[seg];
                int m_block = c * 2 + cta_rank;
                int num_n_blocks = (doc_len + BLOCK_N - 1) / BLOCK_N;
                int n_count = (num_n_blocks + 1) / 2;
                float row_max = -CAKE_INF;
                float row_sum = 0.0f;
                #pragma unroll 1
                for (unsigned int n_iter = 0; n_iter < n_count; n_iter++) {
                    int n_block = (unsigned int)(2 * n_count - 1) - stage - 2 * n_iter;
                    mbarrier_wait(s_full_addr + (stage) * 8, _phase_s_full);
                    _phase_s_full ^= 1;
                    int s_addr = taddr + (unsigned int)tmem_s_off + (unsigned int)(warp % 4 * 32 << 16);
                    float sv[128];
                    float tile_max = -CAKE_INF;
                    {
                        float tile_max_lo = -CAKE_INF;
                        float tile_max_hi = -CAKE_INF;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                        #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                        #endif
                        asm volatile(
                            "tcgen05.ld.red.sync.aligned.32x32b.x64.max.f32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, [%65];"
                            : "=f"(sv[0]), "=f"(sv[1]), "=f"(sv[2]), "=f"(sv[3]), "=f"(sv[4]), "=f"(sv[5]), "=f"(sv[6]), "=f"(sv[7]), "=f"(sv[8]), "=f"(sv[9]), "=f"(sv[10]), "=f"(sv[11]), "=f"(sv[12]), "=f"(sv[13]), "=f"(sv[14]), "=f"(sv[15]), "=f"(sv[16]), "=f"(sv[17]), "=f"(sv[18]), "=f"(sv[19]), "=f"(sv[20]), "=f"(sv[21]), "=f"(sv[22]), "=f"(sv[23]), "=f"(sv[24]), "=f"(sv[25]), "=f"(sv[26]), "=f"(sv[27]), "=f"(sv[28]), "=f"(sv[29]), "=f"(sv[30]), "=f"(sv[31]), "=f"(sv[32]), "=f"(sv[33]), "=f"(sv[34]), "=f"(sv[35]), "=f"(sv[36]), "=f"(sv[37]), "=f"(sv[38]), "=f"(sv[39]), "=f"(sv[40]), "=f"(sv[41]), "=f"(sv[42]), "=f"(sv[43]), "=f"(sv[44]), "=f"(sv[45]), "=f"(sv[46]), "=f"(sv[47]), "=f"(sv[48]), "=f"(sv[49]), "=f"(sv[50]), "=f"(sv[51]), "=f"(sv[52]), "=f"(sv[53]), "=f"(sv[54]), "=f"(sv[55]), "=f"(sv[56]), "=f"(sv[57]), "=f"(sv[58]), "=f"(sv[59]), "=f"(sv[60]), "=f"(sv[61]), "=f"(sv[62]), "=f"(sv[63]), "=f"(tile_max_lo)
                            : "r"((unsigned int)tmem_scores + stage * 128 + (unsigned int)(warp % 4 * 32 << 16)));
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                        #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                        #endif
                        asm volatile(
                            "tcgen05.ld.red.sync.aligned.32x32b.x64.max.f32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, [%65];"
                            : "=f"(sv[64]), "=f"(sv[65]), "=f"(sv[66]), "=f"(sv[67]), "=f"(sv[68]), "=f"(sv[69]), "=f"(sv[70]), "=f"(sv[71]), "=f"(sv[72]), "=f"(sv[73]), "=f"(sv[74]), "=f"(sv[75]), "=f"(sv[76]), "=f"(sv[77]), "=f"(sv[78]), "=f"(sv[79]), "=f"(sv[80]), "=f"(sv[81]), "=f"(sv[82]), "=f"(sv[83]), "=f"(sv[84]), "=f"(sv[85]), "=f"(sv[86]), "=f"(sv[87]), "=f"(sv[88]), "=f"(sv[89]), "=f"(sv[90]), "=f"(sv[91]), "=f"(sv[92]), "=f"(sv[93]), "=f"(sv[94]), "=f"(sv[95]), "=f"(sv[96]), "=f"(sv[97]), "=f"(sv[98]), "=f"(sv[99]), "=f"(sv[100]), "=f"(sv[101]), "=f"(sv[102]), "=f"(sv[103]), "=f"(sv[104]), "=f"(sv[105]), "=f"(sv[106]), "=f"(sv[107]), "=f"(sv[108]), "=f"(sv[109]), "=f"(sv[110]), "=f"(sv[111]), "=f"(sv[112]), "=f"(sv[113]), "=f"(sv[114]), "=f"(sv[115]), "=f"(sv[116]), "=f"(sv[117]), "=f"(sv[118]), "=f"(sv[119]), "=f"(sv[120]), "=f"(sv[121]), "=f"(sv[122]), "=f"(sv[123]), "=f"(sv[124]), "=f"(sv[125]), "=f"(sv[126]), "=f"(sv[127]), "=f"(tile_max_hi)
                            : "r"((unsigned int)tmem_scores + (stage * 128 + 64) + (unsigned int)(warp % 4 * 32 << 16)));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        float _max_0 = max_noftz(tile_max_lo, tile_max_hi);
                        tile_max = _max_0;
                    }
                    int tail_valid = doc_len - n_block * BLOCK_N;
                    if (tail_valid < BLOCK_N) {
                        uint32_t _slice_lo_mask_0;
                        {
                            int _lim_0 = tail_valid;
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
                        if (!(_slice_lo_mask_0 & (1u << 0))) sv[0] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 1))) sv[1] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 2))) sv[2] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 3))) sv[3] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 4))) sv[4] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 5))) sv[5] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 6))) sv[6] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 7))) sv[7] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 8))) sv[8] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 9))) sv[9] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 10))) sv[10] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 11))) sv[11] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 12))) sv[12] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 13))) sv[13] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 14))) sv[14] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 15))) sv[15] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 16))) sv[16] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 17))) sv[17] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 18))) sv[18] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 19))) sv[19] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 20))) sv[20] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 21))) sv[21] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 22))) sv[22] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 23))) sv[23] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 24))) sv[24] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 25))) sv[25] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 26))) sv[26] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 27))) sv[27] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 28))) sv[28] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 29))) sv[29] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 30))) sv[30] = -CAKE_INF;
                        if (!(_slice_lo_mask_0 & (1u << 31))) sv[31] = -CAKE_INF;
                        uint32_t _slice_lo_mask_1;
                        {
                            int _lim_1 = tail_valid - 32;
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
                        if (!(_slice_lo_mask_1 & (1u << 0))) sv[32] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 1))) sv[33] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 2))) sv[34] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 3))) sv[35] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 4))) sv[36] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 5))) sv[37] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 6))) sv[38] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 7))) sv[39] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 8))) sv[40] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 9))) sv[41] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 10))) sv[42] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 11))) sv[43] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 12))) sv[44] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 13))) sv[45] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 14))) sv[46] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 15))) sv[47] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 16))) sv[48] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 17))) sv[49] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 18))) sv[50] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 19))) sv[51] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 20))) sv[52] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 21))) sv[53] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 22))) sv[54] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 23))) sv[55] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 24))) sv[56] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 25))) sv[57] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 26))) sv[58] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 27))) sv[59] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 28))) sv[60] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 29))) sv[61] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 30))) sv[62] = -CAKE_INF;
                        if (!(_slice_lo_mask_1 & (1u << 31))) sv[63] = -CAKE_INF;
                        uint32_t _slice_lo_mask_2;
                        {
                            int _lim_2 = tail_valid - 64;
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
                        if (!(_slice_lo_mask_2 & (1u << 0))) sv[64] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 1))) sv[65] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 2))) sv[66] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 3))) sv[67] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 4))) sv[68] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 5))) sv[69] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 6))) sv[70] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 7))) sv[71] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 8))) sv[72] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 9))) sv[73] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 10))) sv[74] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 11))) sv[75] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 12))) sv[76] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 13))) sv[77] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 14))) sv[78] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 15))) sv[79] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 16))) sv[80] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 17))) sv[81] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 18))) sv[82] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 19))) sv[83] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 20))) sv[84] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 21))) sv[85] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 22))) sv[86] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 23))) sv[87] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 24))) sv[88] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 25))) sv[89] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 26))) sv[90] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 27))) sv[91] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 28))) sv[92] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 29))) sv[93] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 30))) sv[94] = -CAKE_INF;
                        if (!(_slice_lo_mask_2 & (1u << 31))) sv[95] = -CAKE_INF;
                        uint32_t _slice_lo_mask_3;
                        {
                            int _lim_3 = tail_valid - 96;
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
                        if (!(_slice_lo_mask_3 & (1u << 0))) sv[96] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 1))) sv[97] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 2))) sv[98] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 3))) sv[99] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 4))) sv[100] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 5))) sv[101] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 6))) sv[102] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 7))) sv[103] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 8))) sv[104] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 9))) sv[105] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 10))) sv[106] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 11))) sv[107] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 12))) sv[108] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 13))) sv[109] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 14))) sv[110] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 15))) sv[111] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 16))) sv[112] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 17))) sv[113] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 18))) sv[114] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 19))) sv[115] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 20))) sv[116] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 21))) sv[117] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 22))) sv[118] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 23))) sv[119] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 24))) sv[120] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 25))) sv[121] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 26))) sv[122] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 27))) sv[123] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 28))) sv[124] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 29))) sv[125] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 30))) sv[126] = -CAKE_INF;
                        if (!(_slice_lo_mask_3 & (1u << 31))) sv[127] = -CAKE_INF;
                        float2 _reg_reduce_max2_4 = {-CAKE_INF, -CAKE_INF};
                        row_max_x32_accum(&sv[0], _reg_reduce_max2_4);
                        row_max_x32_accum(&sv[32], _reg_reduce_max2_4);
                        row_max_x32_accum(&sv[64], _reg_reduce_max2_4);
                        row_max_x32_accum(&sv[96], _reg_reduce_max2_4);
                        float sv_max = row_max_reduce(_reg_reduce_max2_4);
                        tile_max = sv_max;
                    }
                    float _max_1 = max_noftz(tile_max, row_max);
                    float new_max = _max_1;
                    float safe_max = ((new_max == -CAKE_INF) ? 0.0f : new_max);
                    float new_max_scaled = safe_max * softmax_scale_log2;
                    float _fma_0 = __fmaf_rn(row_max, softmax_scale_log2, -new_max_scaled);
                    float acc_scale_log2 = _fma_0;
                    float acc_scale;
                    if (acc_scale_log2 >= -8.0f) {
                        safe_max = ((row_max == -CAKE_INF) ? 0.0f : row_max);
                        acc_scale = 1.0f;
                        new_max_scaled = safe_max * softmax_scale_log2;
                    } else {
                        float _exp2_0 = approx_exp2(acc_scale_log2);
                        acc_scale = ((row_max > -CAKE_INF) ? _exp2_0 : 1.0f);
                        row_max = new_max;
                    }
                    const float2 _fma_b2_5 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_6 = {-new_max_scaled, -new_max_scaled};
                    #pragma unroll
                    for (int _lf = 0; _lf < 8; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 0))[_lf], _fma_b2_5, _fma_c2_6);
                    #pragma unroll
                    for (int _le = 0; _le < 1; _le++) {
                        if (1 && _le >= 0) {
                            float2 _exp2_pair_7 = ex2_emulation_f32x2_value(make_float2(sv[_le*2], sv[_le*2 + 1]));
                            sv[_le*2] = _exp2_pair_7.x;
                            sv[_le*2 + 1] = _exp2_pair_7.y;
                        } else {
                            sv[_le*2] = approx_exp2(sv[_le*2]);
                            sv[_le*2 + 1] = approx_exp2(sv[_le*2 + 1]);
                        }
                    }
                    sv[2] = approx_exp2(sv[2]);
                    float scale_value[1];
                    scale_value[0] = acc_scale;
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(s_addr), "f"(scale_value[0]));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(scale_full_addr + (stage) * 8);
                    #pragma unroll
                    for (int _le = 0; _le < 13; _le++) {
                        sv[_le + 3] = approx_exp2(sv[_le + 3]);
                    }
                    const float2 _fma_b2_8 = {softmax_scale_log2, softmax_scale_log2};
                    const float2 _fma_c2_9 = {-new_max_scaled, -new_max_scaled};
                    #pragma unroll
                    for (int _lf = 0; _lf < 56; _lf++)
                        fma_f32x2_inplace(&reinterpret_cast<float2*>((sv + 16))[_lf], _fma_b2_8, _fma_c2_9);
                    #pragma unroll
                    for (int _le = 0; _le < 112; _le++) {
                        sv[_le + 16] = approx_exp2(sv[_le + 16]);
                    }
                    int p_addr = taddr + (unsigned int)tmem_p_off + (unsigned int)(warp % 4 * 32 << 16);
                    float2 _f2_0 = make_float2(sv[0], sv[1]);
                    float2 partial = _f2_0;
                    #pragma unroll
                    for (int pair = 2; pair < 32; pair += 2) {
                        float2 _f2_1 = make_float2(sv[pair], sv[pair + 1]);
                        partial = add_f32x2(partial, _f2_1);
                    }
                    uint32_t sv_bf16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sv[_lp*2 + 0], sv[_lp*2+1 + 0]));
                        sv_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(p_addr), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[3])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[4])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[5])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[6])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[7])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[8])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[9])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[10])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[11])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[12])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[13])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[14])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16[15])));
                    float2 _f2_2 = make_float2(sv[32], sv[33]);
                    float2 partial_0 = _f2_2;
                    #pragma unroll
                    for (int pair_1 = 34; pair_1 < 64; pair_1 += 2) {
                        float2 _f2_3 = make_float2(sv[pair_1], sv[pair_1 + 1]);
                        partial_0 = add_f32x2(partial_0, _f2_3);
                    }
                    uint32_t sv_bf16_1[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sv[_lp*2 + 32], sv[_lp*2+1 + 32]));
                        sv_bf16_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(p_addr + 16), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[0])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[1])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[2])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[3])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[4])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[5])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[6])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[7])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[8])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[9])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[10])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[11])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[12])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[13])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[14])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_1[15])));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((p_full_addr + (stage) * 8) & 0xFEFFFFFF) : "memory");
                    float2 _f2_4 = make_float2(sv[64], sv[65]);
                    float2 partial_2 = _f2_4;
                    #pragma unroll
                    for (int pair_2 = 66; pair_2 < 96; pair_2 += 2) {
                        float2 _f2_5 = make_float2(sv[pair_2], sv[pair_2 + 1]);
                        partial_2 = add_f32x2(partial_2, _f2_5);
                    }
                    uint32_t sv_bf16_3[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sv[_lp*2 + 64], sv[_lp*2+1 + 64]));
                        sv_bf16_3[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(p_addr + 32), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_3[0])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_3[1])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_3[2])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_3[3])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_3[4])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_3[5])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_3[6])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_3[7])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_3[8])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_3[9])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_3[10])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_3[11])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_3[12])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_3[13])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_3[14])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_3[15])));
                    float2 _f2_6 = make_float2(sv[96], sv[97]);
                    float2 partial_4 = _f2_6;
                    #pragma unroll
                    for (int pair_3 = 98; pair_3 < 128; pair_3 += 2) {
                        float2 _f2_7 = make_float2(sv[pair_3], sv[pair_3 + 1]);
                        partial_4 = add_f32x2(partial_4, _f2_7);
                    }
                    uint32_t sv_bf16_5[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sv[_lp*2 + 96], sv[_lp*2+1 + 96]));
                        sv_bf16_5[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(p_addr + 48), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_5[0])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_5[1])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_5[2])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_5[3])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_5[4])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_5[5])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_5[6])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_5[7])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_5[8])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_5[9])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_5[10])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_5[11])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_5[12])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_5[13])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_5[14])), "r"(*reinterpret_cast<const uint32_t*>(&sv_bf16_5[15])));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((p_full_2_addr + (stage) * 8) & 0xFEFFFFFF) : "memory");
                    mbarrier_wait(corr_done_addr + (stage) * 8, _phase_corr_done);
                    _phase_corr_done ^= 1;
                    float2 sum01 = add_f32x2(partial, partial_0);
                    float2 sum23 = add_f32x2(partial_2, partial_4);
                    float2 total_sum = add_f32x2(sum01, sum23);
                    row_sum = row_sum * acc_scale + total_sum.x + total_sum.y;
                }
                scales[warp % 4 * 32 + lane + scale_off + 2 * BLOCK_M] = row_sum;
                scales[warp % 4 * 32 + lane + scale_off] = row_max;
                if (sync_group == 0) {
                    asm volatile("barrier.sync 1, 256;" ::: "memory");
                } else {
                    asm volatile("barrier.sync 2, 256;" ::: "memory");
                }
            }
        }
    }
    // ---- Role: correction ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 88;");
        { // correction_main
            asm volatile(
                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                :: "r"((p_full_addr) & 0xFEFFFFFF) : "memory");
            asm volatile(
                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                :: "r"((p_full_addr + 8) & 0xFEFFFFFF) : "memory");
            unsigned int q_slot_c = 0;
            unsigned int _phase_scale_full_0 = 0;
            unsigned int _phase_scale_full_1 = 0;
            unsigned int _phase_o_full_0 = 0;
            unsigned int _phase_o_full_1 = 0;
            #pragma unroll 1
            for (unsigned int tile_idx_1 = cluster_id; tile_idx_1 < total_tiles; tile_idx_1 += num_clusters) {
                int rec_1 = tile_idx_1 * 2;
                int seg_1 = unit_table[rec_1];
                int packed_1 = unit_table[rec_1 + 1];
                int head_1 = packed_1 >> 16;
                int c_1 = packed_1 & 65535;
                int doc_begin_1 = seg_begin[seg_1];
                int doc_len_1 = seg_len[seg_1];
                int m_block_1 = c_1 * 2 + cta_rank;
                int num_n_blocks_1 = (doc_len_1 + BLOCK_N - 1) / BLOCK_N;
                int n_count_1 = (num_n_blocks_1 + 1) / 2;
                mbarrier_wait(scale_full_addr, _phase_scale_full_0);
                _phase_scale_full_0 ^= 1;
                mbarrier_arrive(corr_done_addr);
                mbarrier_wait(scale_full_addr + 8, _phase_scale_full_1);
                _phase_scale_full_1 ^= 1;
                #pragma unroll 1
                for (unsigned int n_iter_1 = 1; n_iter_1 < n_count_1; n_iter_1++) {
                    mbarrier_wait(scale_full_addr, _phase_scale_full_0);
                    _phase_scale_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float _tmem_load_0[1];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x1.b32"
                        " {%0}, [%1];"
                        : "=f"(_tmem_load_0[0])
                        : "r"(taddr + (unsigned int)(warp % 4 * 32 << 16)));
                    float scale0 = _tmem_load_0[0];
                    int _vote_0 = __all_sync(0xFFFFFFFF, scale0 == 1.0f);
                    int skip_rescale0 = _vote_0;
                    if (skip_rescale0 == 0) {
                        #pragma unroll
                        for (int col = 0; col < HEAD_DIM / 16; col++) {
                            int addr0 = taddr + (unsigned int)TMEM_OUTPUT_0_OFFSET + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(col * 16);
                            float _tmem_load_1[16];
                            tmem_ld_x16(&_tmem_load_1[0], addr0);
                            const float2 _scale2_0 = {scale0, scale0};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_1)[_ls], _scale2_0);
                            tmem_st_x16_f32(addr0, _tmem_load_1);
                        }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((p_full_addr) & 0xFEFFFFFF) : "memory");
                    mbarrier_arrive(corr_done_addr + 8);
                    mbarrier_wait(scale_full_addr + 8, _phase_scale_full_1);
                    _phase_scale_full_1 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float _tmem_load_2[1];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x1.b32"
                        " {%0}, [%1];"
                        : "=f"(_tmem_load_2[0])
                        : "r"(taddr + (unsigned int)BLOCK_M + (unsigned int)(warp % 4 * 32 << 16)));
                    float scale1 = _tmem_load_2[0];
                    int _vote_1 = __all_sync(0xFFFFFFFF, scale1 == 1.0f);
                    int skip_rescale1 = _vote_1;
                    if (skip_rescale1 == 0) {
                        #pragma unroll
                        for (int col_1 = 0; col_1 < HEAD_DIM / 16; col_1++) {
                            int addr1 = taddr + (unsigned int)TMEM_OUTPUT_1_OFFSET + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(col_1 * 16);
                            float _tmem_load_3[16];
                            tmem_ld_x16(&_tmem_load_3[0], addr1);
                            const float2 _scale2_1 = {scale1, scale1};
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_3)[_ls], _scale2_1);
                            tmem_st_x16_f32(addr1, _tmem_load_3);
                        }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((p_full_addr + 8) & 0xFEFFFFFF) : "memory");
                    mbarrier_arrive(corr_done_addr);
                }
                mbarrier_arrive(corr_done_addr + 8);
                mbarrier_wait(o_full_addr, _phase_o_full_0);
                _phase_o_full_0 ^= 1;
                mbarrier_wait(o_full_addr + 8, _phase_o_full_1);
                _phase_o_full_1 ^= 1;
                asm volatile("barrier.sync 1, 256;" ::: "memory");
                asm volatile("barrier.sync 2, 256;" ::: "memory");
                unsigned int slot0 = q_slot_c;
                unsigned int slot1 = q_slot_c + 1;
                int tile_row = m_block_1 * BLOCK_M;
                int tile_valid = doc_len_1 - tile_row;
                float m0 = scales[warp % 4 * 32 + lane];
                float m1 = scales[warp % 4 * 32 + lane + BLOCK_M];
                float l0 = scales[warp % 4 * 32 + lane + 2 * BLOCK_M];
                float l1 = scales[warp % 4 * 32 + lane + 3 * BLOCK_M];
                float _max_2 = max_noftz(m0, m1);
                float m_ref = _max_2;
                float safe_ref = ((m_ref == -CAKE_INF) ? 0.0f : m_ref);
                float ref_scaled = safe_ref * softmax_scale_log2;
                float a0 = 0.0f;
                float a1 = 0.0f;
                if (m0 > -CAKE_INF) {
                    float _fma_1 = __fmaf_rn(m0, softmax_scale_log2, -ref_scaled);
                    float _exp2_1 = approx_exp2(_fma_1);
                    a0 = _exp2_1;
                }
                if (m1 > -CAKE_INF) {
                    float _fma_2 = __fmaf_rn(m1, softmax_scale_log2, -ref_scaled);
                    float _exp2_2 = approx_exp2(_fma_2);
                    a1 = _exp2_2;
                }
                float final_sum = l0 * a0 + l1 * a1;
                float inv_sum;
                if (final_sum != 0.0f && final_sum == final_sum) {
                    float _rcp_0 = approx_rcp(final_sum);
                    inv_sum = _rcp_0;
                } else {
                    inv_sum = 0.0f;
                }
                float f0 = a0 * inv_sum;
                float f1 = a1 * inv_sum;
                if (tile_valid >= BLOCK_M) {
                    asm volatile("cp.async.bulk.wait_group.read 0;");
                    asm volatile("barrier.sync 3, 128;" ::: "memory");
                    #pragma unroll
                    for (int col_2 = 0; col_2 < HEAD_DIM / 16; col_2++) {
                        int addr0_1 = taddr + (unsigned int)TMEM_OUTPUT_0_OFFSET + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(col_2 * 16);
                        int addr1_1 = taddr + (unsigned int)TMEM_OUTPUT_1_OFFSET + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(col_2 * 16);
                        float _tmem_load_4[16];
                        tmem_ld_x16(&_tmem_load_4[0], addr0_1);
                        float _tmem_load_5[16];
                        tmem_ld_x16(&_tmem_load_5[0], addr1_1);
                        const float2 _scale2_2 = {f1, f1};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_5)[_ls], _scale2_2);
                        #pragma unroll
                        for (int _lf = 0; _lf < 16; _lf++) {
                            _tmem_load_4[_lf] = fmaf(_tmem_load_4[_lf], f0, _tmem_load_5[_lf]);
                        }
                        uint32_t _tmem_load_4_bf16[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 8; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_4[_lp*2 + 0], _tmem_load_4[_lp*2+1 + 0]));
                            _tmem_load_4_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        int srow = (unsigned int)(col_2 / 4 * 128) + slot0 * 256 + (unsigned int)(warp % 4 * 32 + lane);
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o_addr + (unsigned int)(srow * 128 + col_2 % 4 * 32 ^ (srow * 128 + col_2 % 4 * 32 >> 7 & 7) << 4))), "r"(_tmem_load_4_bf16[0]), "r"(_tmem_load_4_bf16[1]), "r"(_tmem_load_4_bf16[2]), "r"(_tmem_load_4_bf16[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_o_addr + (unsigned int)(srow * 128 + (col_2 % 4 * 32 + 16) ^ (srow * 128 + (col_2 % 4 * 32 + 16) >> 7 & 7) << 4))), "r"(_tmem_load_4_bf16[4]), "r"(_tmem_load_4_bf16[5]), "r"(_tmem_load_4_bf16[6]), "r"(_tmem_load_4_bf16[7]) : "memory");
                    }
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((p_full_addr) & 0xFEFFFFFF) : "memory");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((p_full_addr + 8) & 0xFEFFFFFF) : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync 3, 128;" ::: "memory");
                    if (warp == 8) {
                        if (elect_sync()) {
                            tma_store_4d((&O_tma), 0, doc_begin_1 + tile_row, head_1, 0, smem_o_addr + 2 * slot0 * 16384);
                        }
                    }
                    asm volatile("cp.async.bulk.commit_group;");
                } else {
                    float m0_0 = scales[warp % 4 * 32 + lane];
                    float m1_1 = scales[warp % 4 * 32 + lane + BLOCK_M];
                    float l0_2 = scales[warp % 4 * 32 + lane + 2 * BLOCK_M];
                    float l1_3 = scales[warp % 4 * 32 + lane + 3 * BLOCK_M];
                    float _max_3 = max_noftz(m0_0, m1_1);
                    float m_ref_4 = _max_3;
                    float safe_ref_5 = ((m_ref_4 == -CAKE_INF) ? 0.0f : m_ref_4);
                    float ref_scaled_6 = safe_ref_5 * softmax_scale_log2;
                    float a0_7 = 0.0f;
                    float a1_8 = 0.0f;
                    if (m0_0 > -CAKE_INF) {
                        float _fma_3 = __fmaf_rn(m0_0, softmax_scale_log2, -ref_scaled_6);
                        float _exp2_3 = approx_exp2(_fma_3);
                        a0_7 = _exp2_3;
                    }
                    if (m1_1 > -CAKE_INF) {
                        float _fma_4 = __fmaf_rn(m1_1, softmax_scale_log2, -ref_scaled_6);
                        float _exp2_4 = approx_exp2(_fma_4);
                        a1_8 = _exp2_4;
                    }
                    float final_sum_9 = l0_2 * a0_7 + l1_3 * a1_8;
                    float inv_sum_10;
                    if (final_sum_9 != 0.0f && final_sum_9 == final_sum_9) {
                        float _rcp_1 = approx_rcp(final_sum_9);
                        inv_sum_10 = _rcp_1;
                    } else {
                        inv_sum_10 = 0.0f;
                    }
                    float f0_11 = a0_7 * inv_sum_10;
                    float f1_12 = a1_8 * inv_sum_10;
                    int local_row = m_block_1 * BLOCK_M + (warp % 4 * 32 + lane);
                    int out_row = (doc_begin_1 + local_row) * num_heads + head_1;
                    #pragma unroll
                    for (int col_3 = 0; col_3 < HEAD_DIM / 16; col_3++) {
                        int addr0_2 = taddr + (unsigned int)TMEM_OUTPUT_0_OFFSET + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(col_3 * 16);
                        int addr1_2 = taddr + (unsigned int)TMEM_OUTPUT_1_OFFSET + (unsigned int)(warp % 4 * 32 << 16) + (unsigned int)(col_3 * 16);
                        float _tmem_load_6[16];
                        tmem_ld_x16(&_tmem_load_6[0], addr0_2);
                        float _tmem_load_7[16];
                        tmem_ld_x16(&_tmem_load_7[0], addr1_2);
                        const float2 _scale2_3 = {f1_12, f1_12};
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_7)[_ls], _scale2_3);
                        #pragma unroll
                        for (int _lf = 0; _lf < 16; _lf++) {
                            _tmem_load_6[_lf] = fmaf(_tmem_load_6[_lf], f0_11, _tmem_load_7[_lf]);
                        }
                        if (local_row < doc_len_1) {
                            {
                                __nv_bfloat162 _pk[8];
                                _pk[0] = __floats2bfloat162_rn(_tmem_load_6[0 + 0], _tmem_load_6[0 + 1]);
                                _pk[1] = __floats2bfloat162_rn(_tmem_load_6[0 + 2], _tmem_load_6[0 + 3]);
                                _pk[2] = __floats2bfloat162_rn(_tmem_load_6[0 + 4], _tmem_load_6[0 + 5]);
                                _pk[3] = __floats2bfloat162_rn(_tmem_load_6[0 + 6], _tmem_load_6[0 + 7]);
                                _pk[4] = __floats2bfloat162_rn(_tmem_load_6[0 + 8], _tmem_load_6[0 + 9]);
                                _pk[5] = __floats2bfloat162_rn(_tmem_load_6[0 + 10], _tmem_load_6[0 + 11]);
                                _pk[6] = __floats2bfloat162_rn(_tmem_load_6[0 + 12], _tmem_load_6[0 + 13]);
                                _pk[7] = __floats2bfloat162_rn(_tmem_load_6[0 + 14], _tmem_load_6[0 + 15]);
                                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O + (out_row * HEAD_DIM + col_3 * 16)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O + (out_row * HEAD_DIM + col_3 * 16)))[8]) = *reinterpret_cast<uint4*>(&_pk[4]);
                            }
                        }
                    }
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((p_full_addr) & 0xFEFFFFFF) : "memory");
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((p_full_addr + 8) & 0xFEFFFFFF) : "memory");
                }
                asm volatile("cp.async.bulk.wait_group.read 0;");
                asm volatile("barrier.sync 3, 128;" ::: "memory");
                if (warp == 8) {
                    if (elect_sync()) {
                        mbarrier_arrive(q_empty_addr + (q_slot_c) * 8);
                    }
                }
                q_slot_c += 1;
                if (q_slot_c == 2) { q_slot_c = 0; }
            }
            asm volatile("cp.async.bulk.wait_group 0;");
        }
    }
    // ---- Role: mma ----
    if (warp == 12) {
        { // mma_main
            unsigned int _phase_p_full_0 = 0;
            unsigned int _phase_p_full_2_0 = 0;
            unsigned int _phase_p_full_1 = 0;
            unsigned int _phase_p_full_2_1 = 0;
            if (cta_rank == 0) {
                unsigned int kv_stage = 0;
                unsigned int kv_phase = 0;
                unsigned int q_stage = 0;
                unsigned int q_phase = 0;
                #pragma unroll 1
                for (unsigned int tile_idx_2 = cluster_id; tile_idx_2 < total_tiles; tile_idx_2 += num_clusters) {
                    int rec_2 = tile_idx_2 * 2;
                    int seg_2 = unit_table[rec_2];
                    int packed_2 = unit_table[rec_2 + 1];
                    int head_2 = packed_2 >> 16;
                    int c_2 = packed_2 & 65535;
                    int doc_begin_2 = seg_begin[seg_2];
                    int doc_len_2 = seg_len[seg_2];
                    int m_block_2 = c_2 * 2 + cta_rank;
                    int num_n_blocks_2 = (doc_len_2 + BLOCK_N - 1) / BLOCK_N;
                    int n_count_2 = (num_n_blocks_2 + 1) / 2;
                    mbarrier_wait_cluster_hint(q_full_addr + (q_stage) * 8, q_phase, 10000000);
                    mbarrier_wait(kv_full_addr + (kv_stage) * 8, kv_phase);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_0 = (((smem_qa_addr) >> 4) & 0x3FFF) + (q_stage) * 2048;
                    int _mma_b_lo_0 = (((smem_kv_addr) >> 4) & 0x3FFF) + (kv_stage) * 1024;
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
                    "mov.b32 id, 270533776;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"(tmem_scores), "r"(0));
                    elect_commit_cg2_multicast(s_full_addr, (uint16_t)(3));
                    elect_commit_cg2_multicast(kv_empty_addr + (kv_stage) * 8, (uint16_t)(3));
                    kv_stage += 1;
                    if (kv_stage == 7) { kv_stage = 0; kv_phase ^= 1; }
                    mbarrier_wait(kv_full_addr + (kv_stage) * 8, kv_phase);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_1 = (((smem_qa_addr) >> 4) & 0x3FFF) + (q_stage) * 2048;
                    int _mma_b_lo_1 = (((smem_kv_addr) >> 4) & 0x3FFF) + (kv_stage) * 1024;
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
                    "mov.b32 id, 270533776;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"((tmem_scores + (128))), "r"(0));
                    elect_commit_cg2_multicast(s_full_addr + 8, (uint16_t)(3));
                    elect_commit_cg2_multicast(kv_empty_addr + (kv_stage) * 8, (uint16_t)(3));
                    kv_stage += 1;
                    if (kv_stage == 7) { kv_stage = 0; kv_phase ^= 1; }
                    unsigned int first_pv = 1;
                    #pragma unroll 1
                    for (unsigned int n_iter_2 = 0; n_iter_2 < n_count_2 - 1; n_iter_2++) {
                        unsigned int va_stage = kv_stage;
                        unsigned int va_phase = kv_phase;
                        kv_stage += 1;
                        if (kv_stage == 7) { kv_stage = 0; kv_phase ^= 1; }
                        int first_pv_flag = first_pv;
                        mbarrier_wait(kv_full_addr + (va_stage) * 8, va_phase);
                        mbarrier_wait(p_full_addr, _phase_p_full_0);
                        _phase_p_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_b_lo_2 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (va_stage) * 1024;
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
                    "mov.b32 id, 270599312;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_0), "r"(_mma_b_lo_2), "r"(tmem_probs_0), "r"(((first_pv_flag) ? 0 : 1)));
                        mbarrier_wait(p_full_2_addr, _phase_p_full_2_0);
                        _phase_p_full_2_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_b_lo_3 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (va_stage) * 1024;
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
                    "mov.b32 id, 270599312;\n\t"
                    "add.u32 ta, %2, 32;\n\t"
                    "add.u32 blo, %1, 512;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_0), "r"(_mma_b_lo_3), "r"(tmem_probs_0), "r"(1));
                        elect_commit_cg2_multicast(kv_empty_addr + (va_stage) * 8, (uint16_t)(3));
                        unsigned int ka_stage = kv_stage;
                        unsigned int ka_phase = kv_phase;
                        kv_stage += 1;
                        if (kv_stage == 7) { kv_stage = 0; kv_phase ^= 1; }
                        mbarrier_wait(kv_full_addr + (ka_stage) * 8, ka_phase);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_4 = (((smem_qa_addr) >> 4) & 0x3FFF) + (q_stage) * 2048;
                        int _mma_b_lo_4 = (((smem_kv_addr) >> 4) & 0x3FFF) + (ka_stage) * 1024;
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
                    "mov.b32 id, 270533776;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_4), "r"(_mma_b_lo_4), "r"(tmem_scores), "r"(0));
                        elect_commit_cg2_multicast(s_full_addr, (uint16_t)(3));
                        elect_commit_cg2_multicast(kv_empty_addr + (ka_stage) * 8, (uint16_t)(3));
                        unsigned int vb_stage = kv_stage;
                        unsigned int vb_phase = kv_phase;
                        kv_stage += 1;
                        if (kv_stage == 7) { kv_stage = 0; kv_phase ^= 1; }
                        mbarrier_wait(kv_full_addr + (vb_stage) * 8, vb_phase);
                        mbarrier_wait(p_full_addr + 8, _phase_p_full_1);
                        _phase_p_full_1 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_b_lo_5 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (vb_stage) * 1024;
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
                    "mov.b32 id, 270599312;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_1), "r"(_mma_b_lo_5), "r"(tmem_probs_1), "r"(((first_pv_flag) ? 0 : 1)));
                        mbarrier_wait(p_full_2_addr + 8, _phase_p_full_2_1);
                        _phase_p_full_2_1 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_b_lo_6 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (vb_stage) * 1024;
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
                    "mov.b32 id, 270599312;\n\t"
                    "add.u32 ta, %2, 32;\n\t"
                    "add.u32 blo, %1, 512;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_1), "r"(_mma_b_lo_6), "r"(tmem_probs_1), "r"(1));
                        elect_commit_cg2_multicast(kv_empty_addr + (vb_stage) * 8, (uint16_t)(3));
                        unsigned int kb_stage = kv_stage;
                        unsigned int kb_phase = kv_phase;
                        kv_stage += 1;
                        if (kv_stage == 7) { kv_stage = 0; kv_phase ^= 1; }
                        mbarrier_wait(kv_full_addr + (kb_stage) * 8, kb_phase);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_7 = (((smem_qa_addr) >> 4) & 0x3FFF) + (q_stage) * 2048;
                        int _mma_b_lo_7 = (((smem_kv_addr) >> 4) & 0x3FFF) + (kb_stage) * 1024;
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
                    "mov.b32 id, 270533776;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 1018;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_7), "r"(_mma_b_lo_7), "r"((tmem_scores + (128))), "r"(0));
                        elect_commit_cg2_multicast(s_full_addr + 8, (uint16_t)(3));
                        elect_commit_cg2_multicast(kv_empty_addr + (kb_stage) * 8, (uint16_t)(3));
                        first_pv = 0;
                    }
                    q_stage += 1;
                    if (q_stage == 2) { q_stage = 0; q_phase ^= 1; }
                    int first_pv_flag_1 = first_pv;
                    unsigned int va_stage_1 = kv_stage;
                    unsigned int va_phase_1 = kv_phase;
                    kv_stage += 1;
                    if (kv_stage == 7) { kv_stage = 0; kv_phase ^= 1; }
                    mbarrier_wait(kv_full_addr + (va_stage_1) * 8, va_phase_1);
                    mbarrier_wait(p_full_addr, _phase_p_full_0);
                    _phase_p_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_b_lo_8 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (va_stage_1) * 1024;
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
                    "mov.b32 id, 270599312;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_0), "r"(_mma_b_lo_8), "r"(tmem_probs_0), "r"(((first_pv_flag_1) ? 0 : 1)));
                    mbarrier_wait(p_full_2_addr, _phase_p_full_2_0);
                    _phase_p_full_2_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_b_lo_9 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (va_stage_1) * 1024;
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
                    "mov.b32 id, 270599312;\n\t"
                    "add.u32 ta, %2, 32;\n\t"
                    "add.u32 blo, %1, 512;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_0), "r"(_mma_b_lo_9), "r"(tmem_probs_0), "r"(1));
                    elect_commit_cg2_multicast(kv_empty_addr + (va_stage_1) * 8, (uint16_t)(3));
                    unsigned int vb_stage_1 = kv_stage;
                    unsigned int vb_phase_1 = kv_phase;
                    kv_stage += 1;
                    if (kv_stage == 7) { kv_stage = 0; kv_phase ^= 1; }
                    mbarrier_wait(kv_full_addr + (vb_stage_1) * 8, vb_phase_1);
                    mbarrier_wait(p_full_addr + 8, _phase_p_full_1);
                    _phase_p_full_1 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_b_lo_10 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (vb_stage_1) * 1024;
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
                    "mov.b32 id, 270599312;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_1), "r"(_mma_b_lo_10), "r"(tmem_probs_1), "r"(((first_pv_flag_1) ? 0 : 1)));
                    mbarrier_wait(p_full_2_addr + 8, _phase_p_full_2_1);
                    _phase_p_full_2_1 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_b_lo_11 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (vb_stage_1) * 1024;
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
                    "mov.b32 id, 270599312;\n\t"
                    "add.u32 ta, %2, 32;\n\t"
                    "add.u32 blo, %1, 512;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%0], [ta], db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_output_1), "r"(_mma_b_lo_11), "r"(tmem_probs_1), "r"(1));
                    elect_commit_cg2_multicast(kv_empty_addr + (vb_stage_1) * 8, (uint16_t)(3));
                    elect_commit_cg2_multicast(o_full_addr, (uint16_t)(3));
                    elect_commit_cg2_multicast(o_full_addr + 8, (uint16_t)(3));
                }
            }
        }
    }
    // ---- Role: load ----
    if (warp == 13) {
        { // load_main
            unsigned int load_stage = 0;
            unsigned int q_load_stage = 0;
            unsigned int _phase_q_empty = 1;
            unsigned int _phase_kv_empty = 1;
            #pragma unroll 1
            for (unsigned int tile_idx_3 = cluster_id; tile_idx_3 < total_tiles; tile_idx_3 += num_clusters) {
                int rec_3 = tile_idx_3 * 2;
                int seg_3 = unit_table[rec_3];
                int packed_3 = unit_table[rec_3 + 1];
                int head_3 = packed_3 >> 16;
                int c_3 = packed_3 & 65535;
                int doc_begin_3 = seg_begin[seg_3];
                int doc_len_3 = seg_len[seg_3];
                int m_block_3 = c_3 * 2 + cta_rank;
                int num_n_blocks_3 = (doc_len_3 + BLOCK_N - 1) / BLOCK_N;
                int n_count_3 = (num_n_blocks_3 + 1) / 2;
                int q_local = m_block_3 * BLOCK_M;
                int q_row = doc_begin_3 + q_local;
                int q_remaining = doc_len_3 - q_local;
                mbarrier_wait(q_empty_addr + (q_load_stage) * 8, _phase_q_empty);
                if (elect_sync()) {
                    asm volatile(
                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                        :: "r"((q_full_addr + (q_load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                    tma_4d_gmem2smem_cta2(smem_qa_addr + q_load_stage * 32768, (&Q), 0, q_row, head_3, 0, ((q_full_addr + (q_load_stage) * 8) & 0xFEFFFFFF));
                }
                q_load_stage += 1;
                if (q_load_stage == 2) { q_load_stage = 0; _phase_q_empty ^= 1; }
                int top = 2 * n_count_3 - 1;
                mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                if (elect_sync()) {
                    asm volatile(
                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                        :: "r"((kv_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                    tma_4d_gmem2smem_cta2(smem_kv_addr + load_stage * 16384, (&K), 0, doc_begin_3 + top * BLOCK_N + cta_rank * 64, head_3, 0, ((kv_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                }
                load_stage += 1;
                if (load_stage == 7) { load_stage = 0; _phase_kv_empty ^= 1; }
                mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                if (elect_sync()) {
                    asm volatile(
                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                        :: "r"((kv_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                    tma_4d_gmem2smem_cta2(smem_kv_addr + load_stage * 16384, (&K), 0, doc_begin_3 + (top - 1) * BLOCK_N + cta_rank * 64, head_3, 0, ((kv_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                }
                load_stage += 1;
                if (load_stage == 7) { load_stage = 0; _phase_kv_empty ^= 1; }
                #pragma unroll 1
                for (unsigned int ni = 0; ni < n_count_3 - 1; ni++) {
                    int a_row = (unsigned int)doc_begin_3 + ((unsigned int)top - 2 * ni) * (unsigned int)BLOCK_N;
                    mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                    if (elect_sync()) {
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((kv_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                        tma_4d_gmem2smem_cta2(smem_v_addr + load_stage * 16384, (&V), 0, a_row, cta_rank, head_3, ((kv_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                    }
                    load_stage += 1;
                    if (load_stage == 7) { load_stage = 0; _phase_kv_empty ^= 1; }
                    mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                    if (elect_sync()) {
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((kv_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                        tma_4d_gmem2smem_cta2(smem_kv_addr + load_stage * 16384, (&K), 0, a_row - 2 * BLOCK_N + cta_rank * 64, head_3, 0, ((kv_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                    }
                    load_stage += 1;
                    if (load_stage == 7) { load_stage = 0; _phase_kv_empty ^= 1; }
                    mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                    if (elect_sync()) {
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((kv_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                        tma_4d_gmem2smem_cta2(smem_v_addr + load_stage * 16384, (&V), 0, a_row - BLOCK_N, cta_rank, head_3, ((kv_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                    }
                    load_stage += 1;
                    if (load_stage == 7) { load_stage = 0; _phase_kv_empty ^= 1; }
                    mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                    if (elect_sync()) {
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((kv_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                        tma_4d_gmem2smem_cta2(smem_kv_addr + load_stage * 16384, (&K), 0, a_row - 3 * BLOCK_N + cta_rank * 64, head_3, 0, ((kv_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                    }
                    load_stage += 1;
                    if (load_stage == 7) { load_stage = 0; _phase_kv_empty ^= 1; }
                }
                mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                if (elect_sync()) {
                    asm volatile(
                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                        :: "r"((kv_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                    tma_4d_gmem2smem_cta2(smem_v_addr + load_stage * 16384, (&V), 0, doc_begin_3 + BLOCK_N, cta_rank, head_3, ((kv_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                }
                load_stage += 1;
                if (load_stage == 7) { load_stage = 0; _phase_kv_empty ^= 1; }
                mbarrier_wait(kv_empty_addr + (load_stage) * 8, _phase_kv_empty);
                if (elect_sync()) {
                    asm volatile(
                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                        :: "r"((kv_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                    tma_4d_gmem2smem_cta2(smem_v_addr + load_stage * 16384, (&V), 0, doc_begin_3, cta_rank, head_3, ((kv_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                }
                load_stage += 1;
                if (load_stage == 7) { load_stage = 0; _phase_kv_empty ^= 1; }
            }
        }
    }
    // ---- Role: empty ----
    if (warp >= 14 && warp <= 15) {
        // idle — no tasks assigned
    }

    // Cleanup
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
