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
struct CakeFastDivmod { int32_t divisor; uint32_t multiplier; uint32_t shift_right; };
static_assert(sizeof(CakeFastDivmod) == 12, "CakeFastDivmod CUDA ABI must be 12 bytes");
static_assert(alignof(CakeFastDivmod) == 4, "CakeFastDivmod CUDA ABI must be 4-byte aligned");

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(CakeTensorMap) >= alignof(CUtensorMap), "CakeTensorMap alignment must cover the CUtensorMap CUDA ABI");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 512
#define TMEM_TMEM_SCRATCH_OFFSET 0
#define NUM_KV_PIPE_STAGES 15
#define NUM_WORK_PIPE_STAGES 2
#define SMEM_SMEM_Q_OFF 1024
#define SMEM_SMEM_Q_STAGE_BYTES 8192
#define SMEM_SMEM_Q_STRIDE 8192
#define SMEM_SMEM_KV_OFF 74752
#define SMEM_SMEM_KV_STAGE_BYTES 8192
#define SMEM_SMEM_KV_STRIDE 8192
#define SMEM_SMEM_V_OFF 74752
#define SMEM_SMEM_V_STAGE_BYTES 8192
#define SMEM_SMEM_V_STRIDE 8192
#define SMEM_SMEM_P_OFF 197632
#define SMEM_SMEM_P_STAGE_BYTES 4096
#define SMEM_SMEM_P_STRIDE 4096
#define SMEM_SMEM_OSTAGE_OFF 197632
#define SMEM_SMEM_OSTAGE_STAGE_BYTES 8192
#define SMEM_SMEM_OSTAGE_STRIDE 8192
#define SMEM_SMEM_SOFTMAX_EXCHANGE_OFF 230400
#define SMEM_SMEM_SOFTMAX_EXCHANGE_STAGE_BYTES 512
#define SMEM_SMEM_SOFTMAX_EXCHANGE_STRIDE 512
#define SMEM_SMEM_EPILOGUE_EXCHANGE_OFF 230912
#define SMEM_SMEM_EPILOGUE_EXCHANGE_STAGE_BYTES 512
#define SMEM_SMEM_EPILOGUE_EXCHANGE_STRIDE 512
#define SMEM_WORK_TOKEN_WORDS_OFF 231424
#define SMEM_WORK_TOKEN_WORDS_STAGE_BYTES 64
#define SMEM_WORK_TOKEN_WORDS_STRIDE 64
#define SMEM_PLAN_WORDS_OFF 231488
#define SMEM_PLAN_WORDS_STAGE_BYTES 16
#define SMEM_PLAN_WORDS_STRIDE 16
#define SMEM_PUB_WORDS_OFF 231504
#define SMEM_PUB_WORDS_STAGE_BYTES 32
#define SMEM_PUB_WORDS_STRIDE 32
#define SMEM_TOTAL 231552
#define THREADS 384
#define USE_PDL 1

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


__device__ __forceinline__ void tmem_ld_x4(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x4.b32"
        " {%0, %1, %2, %3}, [%4];"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3])
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


__device__ __forceinline__ void tmem_st_x4_f32(int tmem_addr, const float* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x4.b32"
        " [%0], {%1, %2, %3, %4};"
        :: "r"(tmem_addr),
           "f"(src[0]), "f"(src[1]), "f"(src[2]), "f"(src[3]));
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


__device__ __forceinline__ void tma_3d_gmem2smem_cta2(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
        " [%0], [%1, {%2, %3, %4}], [%5];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z),
           "r"(mbar_addr) : "memory");
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


__device__ __forceinline__ void tma_4d_gmem2smem_cta2(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_store_2d(
    const void *tmap, int x, int y, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2}], [%3];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(smem_addr) : "memory");
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

extern "C" {

__global__ __launch_bounds__(384, 1) __cluster_dims__(2,1,1) void
kernel_cake_mla_varq_dcp_decode_151fa207eae09b6acf4c(const __grid_constant__ CUtensorMap tmap_q, const __grid_constant__ CUtensorMap tmap_k, const __grid_constant__ CUtensorMap tmap_v, const __grid_constant__ CUtensorMap tmap_o, const __grid_constant__ CUtensorMap tmap_po, __nv_bfloat16* __restrict__ O, float* __restrict__ LSE, __nv_bfloat16* __restrict__ partial_O, float* __restrict__ partial_lse, int* __restrict__ page_table, int* __restrict__ seq_lens, int* __restrict__ cum_seq_lens_q, int* __restrict__ causal_global, unsigned int* __restrict__ sched_counters, unsigned int* __restrict__ unit_flags, unsigned int* __restrict__ split_meta, unsigned int* __restrict__ merge_ctl, float softmax_scale_log2, int tiles_max, int num_heads, int max_pages, int cp_world, int cp_rank, int num_items, int unit_min, int static_tiles, int unit_num, int unit_den, int max_units, int static_only, unsigned long long* __restrict__ dbg, int partial_slots, CakeFastDivmod fd_tiles_max, CakeFastDivmod fd_num_heads, CakeFastDivmod fd_cp_world, CakeFastDivmod fd_num_items, CakeFastDivmod fd_unit_min, CakeFastDivmod fd_static_tiles, CakeFastDivmod fd_unit_den, CakeFastDivmod fd_clusters)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);

    const int mbar_base = smem;
    #define work_full_addr (mbar_base + 0)
    #define work_empty_addr (mbar_base + 16)
    #define plan_full_addr (mbar_base + 32)
    #define unit_done_addr (mbar_base + 40)
    #define merge_ready_addr (mbar_base + 48)
    #define pub_req_addr (mbar_base + 56)
    #define pub_free_addr (mbar_base + 64)
    #define p_epi_free_addr (mbar_base + 72)
    #define q_full_addr (mbar_base + 80)
    #define q_empty_addr (mbar_base + 88)
    #define kv_full_addr (mbar_base + 96)
    #define kv_empty_addr (mbar_base + 216)
    #define s_full_addr (mbar_base + 336)
    #define s_empty_addr (mbar_base + 352)
    #define p_full_addr (mbar_base + 368)
    #define p_empty_addr (mbar_base + 384)
    #define o_empty_addr (mbar_base + 400)
    #define stats_addr (mbar_base + 408)
    #define stats_empty_addr (mbar_base + 424)
    #define o_full_addr (mbar_base + 440)
    #define o_first_slice_seeded_addr (mbar_base + 448)
    #define tmem_dealloc_peer_addr (mbar_base + 456)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    __nv_bfloat16* smem_q = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_q_addr = smem + 1024;
    __nv_bfloat16* smem_kv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 74752);
    const int smem_kv_addr = smem + 74752;
    __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 74752);
    const int smem_v_addr = smem + 74752;
    __nv_bfloat16* smem_p = reinterpret_cast<__nv_bfloat16*>(smem_raw + 197632);
    const int smem_p_addr = smem + 197632;
    __nv_bfloat16* smem_ostage = reinterpret_cast<__nv_bfloat16*>(smem_raw + 197632);
    const int smem_ostage_addr = smem + 197632;
    float* smem_softmax_exchange = reinterpret_cast<float*>(smem_raw + 230400);
    const int smem_softmax_exchange_addr = smem + 230400;
    float* smem_epilogue_exchange = reinterpret_cast<float*>(smem_raw + 230912);
    const int smem_epilogue_exchange_addr = smem + 230912;
    unsigned int* work_token_words = reinterpret_cast<unsigned int*>(smem_raw + 231424);
    const int work_token_words_addr = smem + 231424;
    unsigned int* plan_words = reinterpret_cast<unsigned int*>(smem_raw + 231488);
    const int plan_words_addr = smem + 231488;
    unsigned int* pub_words = reinterpret_cast<unsigned int*>(smem_raw + 231504);
    const int pub_words_addr = smem + 231504;
    if (warp == 8) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&tmap_q))) : "memory"); }
    if (warp == 8) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&tmap_k))) : "memory"); }
    if (warp == 8) { asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&tmap_v))) : "memory"); }

    // Mbarrier init (22 pipeline groups, 0 ordered-sequence groups, 58 barriers)
    // Mbarriers at smem_raw[0..464)

    if (warp == 0) {
        // --- pipeline 'work_pipe' ---
        // work_full: 2 barriers, init_count=1
        // work_empty: 2 barriers, init_count=20
        // plan_full: 1 barriers, init_count=1
        // unit_done: 1 barriers, init_count=1
        // merge_ready: 1 barriers, init_count=256
        // pub_req: 1 barriers, init_count=1
        // pub_free: 1 barriers, init_count=1
        // p_epi_free: 1 barriers, init_count=1
        // q_full: 1 barriers, init_count=1
        // q_empty: 1 barriers, init_count=1
        // --- pipeline 'kv_pipe' ---
        // kv_full: 15 barriers, init_count=1
        // kv_empty: 15 barriers, init_count=1
        // s_full: 2 barriers, init_count=1
        // s_empty: 2 barriers, init_count=256
        // p_full: 2 barriers, init_count=256
        // p_empty: 2 barriers, init_count=1
        // o_empty: 1 barriers, init_count=256
        // stats: 2 barriers, init_count=128
        // stats_empty: 2 barriers, init_count=128
        // o_full: 1 barriers, init_count=1
        // o_first_slice_seeded: 1 barriers, init_count=256
        // tmem_dealloc_peer: 1 barriers, init_count=32
        // Warp-cooperative initialization in physical record order.
        uint32_t _mbarrier_init_count_0_0 = 1;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(7), "r"((uint32_t)(256)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(6), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(4), "r"((uint32_t)(20)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(2), "r"((uint32_t)(1)));
        mbarrier_init(smem + 0 + lane * 8, _mbarrier_init_count_0_0);
        uint32_t _mbarrier_init_count_0_32 = 32;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(25), "r"((uint32_t)(256)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(24), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(23), "r"((uint32_t)(128)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(19), "r"((uint32_t)(256)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(18), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(16), "r"((uint32_t)(256)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(12), "r"((uint32_t)(1)));
        if (lane < 26) {
            mbarrier_init(smem + 256 + lane * 8, _mbarrier_init_count_0_32);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    __syncwarp();

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 464);
    if (warp == 0) {
        int _tmem_hold = smem + 464;
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
        asm volatile("setmaxnreg.dec.sync.aligned.u32 96;");
    }

    // ---- Role: softmax_wg ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 192;");
        { // softmax_wg_main
            const int wg_dummy_inc = 0;
            {
                asm volatile("griddepcontrol.wait;" ::: "memory");
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
            const int tmem_row_base = warp % 2 * 32;
            const int n_half = warp % 4 / 2;
            const int my_row = tmem_row_base + lane;
            const int exchange_idx = n_half * 64 + my_row;
            const int seed_row_base = warp % 4 * 32;
            int row_in_tile = cta_rank * 64 + my_row;
            int softmax_tile_cursor = 0;
            float seed_zero[4];
            #pragma unroll
            for (int seed_i = 0; seed_i < 4; seed_i++) {
                seed_zero[seed_i] = 0.0f;
            }
            #pragma unroll
            for (int seed_col = 128; seed_col < 192; seed_col += 4) {
                int seed_addr = taddr + (unsigned int)seed_col + (unsigned int)(seed_row_base << 16);
                tmem_st_x4_f32(seed_addr, seed_zero);
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(o_first_slice_seeded_addr);
            int seed_peer_rank = cta_rank ^ 1;
            asm volatile(
                "{\n\t"
                ".reg .b32 remAddr32;\n\t"
                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                "}"
                :: "r"(o_first_slice_seeded_addr), "r"(seed_peer_rank) : "memory");
            unsigned int work_stage_softmax = 0;
            unsigned int first_s_seen = 1;
            uint32_t _fast_div_q_10 = (fd_num_items.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)cluster_id), fd_num_items.multiplier) >> fd_num_items.shift_right) : (uint32_t)((unsigned int)cluster_id);
            uint32_t _fast_div_r_10 = (uint32_t)((unsigned int)cluster_id) - _fast_div_q_10 * (uint32_t)(fd_num_items.divisor);
            int k_st = (int)_fast_div_q_10;
            int it_st = cluster_id - (unsigned int)(k_st * num_items);
            uint32_t _fast_div_q_11 = (fd_tiles_max.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)it_st), fd_tiles_max.multiplier) >> fd_tiles_max.shift_right) : (uint32_t)((unsigned int)it_st);
            uint32_t _fast_div_r_11 = (uint32_t)((unsigned int)it_st) - _fast_div_q_11 * (uint32_t)(fd_tiles_max.divisor);
            int sb = (int)_fast_div_q_11;
            int sm = it_st - sb * tiles_max;
            int q_begin = cum_seq_lens_q[sb];
            int q_len = cum_seq_lens_q[sb + 1] - q_begin;
            int rows_remaining = q_len * num_heads - sm * 128;
            int valid_rows = ((rows_remaining < 128) ? rows_remaining : 128);
            int k_local = seq_lens[sb];
            int g_bound = causal_global[sb];
            int n_t = -1;
            if (valid_rows > 0) {
                uint32_t _fast_div_q_12 = (fd_num_heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(sm * 128 + valid_rows - 1)), fd_num_heads.multiplier) >> fd_num_heads.shift_right) : (uint32_t)((unsigned int)(sm * 128 + valid_rows - 1));
                uint32_t _fast_div_r_12 = (uint32_t)((unsigned int)(sm * 128 + valid_rows - 1)) - _fast_div_q_12 * (uint32_t)(fd_num_heads.divisor);
                int last_q_t = (int)_fast_div_q_12;
                int lim_num_t = g_bound - q_len + last_q_t + 1 - cp_rank;
                lim_num_t = ((lim_num_t > 0) ? lim_num_t : 0);
                uint32_t _fast_div_q_13 = (fd_cp_world.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(lim_num_t + cp_world - 1)), fd_cp_world.multiplier) >> fd_cp_world.shift_right) : (uint32_t)((unsigned int)(lim_num_t + cp_world - 1));
                uint32_t _fast_div_r_13 = (uint32_t)((unsigned int)(lim_num_t + cp_world - 1)) - _fast_div_q_13 * (uint32_t)(fd_cp_world.divisor);
                int lim_ceil_t = (int)_fast_div_q_13;
                int key_lim_t = ((lim_ceil_t < k_local) ? lim_ceil_t : k_local);
                key_lim_t = ((key_lim_t > 0) ? key_lim_t : 0);
                n_t = (key_lim_t + 128 - 1) / 128;
            }
            int n_all = n_t;
            int n_st = n_all - k_st * static_tiles;
            n_st = ((n_st < static_tiles) ? n_st : static_tiles);
            n_st = ((n_st > 0) ? n_st : 0);
            int has_st = 1;
            if (k_st > 0) {
                has_st = ((n_st > 0) ? 1 : 0);
            }
            if (k_st >= 8) {
                has_st = 0;
            }
            int has_static_softmax = has_st;
            int no_token_softmax = ((0) ? static_only : 0);
            int par_softmax = 0;
            unsigned int _phase_work_full = 0;
            unsigned int _phase_p_epi_free_0 = 1;
            #pragma unroll 1
            for (unsigned int _unit_iter_softmax = 0; _unit_iter_softmax < max_units + 1; _unit_iter_softmax++) {
                unsigned int tok_valid = 1;
                int batch_idx = 0;
                int m_tile = 0;
                int start_tile = 0;
                int num_kv_tiles = 0;
                int split_idx = 0;
                int item_units = 1;
                int slot_base = 0;
                int is_static = has_static_softmax;
                has_static_softmax = 0;
                if (is_static != 0) {
                    batch_idx = sb;
                    m_tile = sm;
                    num_kv_tiles = n_st;
                    start_tile = k_st * static_tiles;
                    split_idx = k_st;
                } else if (no_token_softmax != 0) {
                    tok_valid = 0;
                } else {
                    mbarrier_wait_cluster(work_full_addr + (work_stage_softmax) * 8, _phase_work_full);
                    unsigned int tok_words[8];
                    #pragma unroll
                    for (int w = 0; w < 8; w++) {
                        tok_words[w] = 0;
                    }
                    if (lane == 0) {
                        uint32_t _mapa_2;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mapa_2) : "r"(work_token_words_addr + work_stage_softmax * 32), "r"(0));
                        asm volatile("ld.shared::cluster.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tok_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&tok_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tok_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tok_words[(0) + 3]))
                            : "r"(_mapa_2));
                        asm volatile("ld.shared::cluster.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tok_words[4])), "=r"(*reinterpret_cast<uint32_t*>(&tok_words[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tok_words[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tok_words[(4) + 3]))
                            : "r"(_mapa_2 + 16));
                    }
                    unsigned int _shfl_16 = __shfl_sync(0xFFFFFFFF, tok_words[0], 0);
                    unsigned int valid = _shfl_16;
                    unsigned int _shfl_17 = __shfl_sync(0xFFFFFFFF, tok_words[1], 0);
                    int batch_idx_0 = (int)_shfl_17;
                    unsigned int _shfl_18 = __shfl_sync(0xFFFFFFFF, tok_words[2], 0);
                    int m_tile_1 = (int)_shfl_18;
                    unsigned int _shfl_19 = __shfl_sync(0xFFFFFFFF, tok_words[3], 0);
                    int start_tile_2 = (int)_shfl_19;
                    unsigned int _shfl_20 = __shfl_sync(0xFFFFFFFF, tok_words[4], 0);
                    int num_kv_tiles_3 = (int)_shfl_20;
                    unsigned int _shfl_21 = __shfl_sync(0xFFFFFFFF, tok_words[5], 0);
                    int unit_idx = (int)_shfl_21;
                    unsigned int _shfl_22 = __shfl_sync(0xFFFFFFFF, tok_words[6], 0);
                    int units = (int)_shfl_22;
                    unsigned int _shfl_23 = __shfl_sync(0xFFFFFFFF, tok_words[7], 0);
                    int slot_base_4 = (int)_shfl_23;
                    if (lane == 0) {
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                            "}"
                            :: "r"(work_empty_addr + work_stage_softmax * 8), "r"(0) : "memory");
                    }
                    work_stage_softmax += 1;
                    if (work_stage_softmax == 2) { work_stage_softmax = 0; _phase_work_full ^= 1; }
                    tok_valid = valid;
                    batch_idx = batch_idx_0;
                    m_tile = m_tile_1;
                    start_tile = start_tile_2;
                    num_kv_tiles = num_kv_tiles_3;
                    split_idx = unit_idx;
                    item_units = units;
                    slot_base = slot_base_4;
                }
                if (tok_valid == 0) {
                    break;
                }
                int q_begin_0 = cum_seq_lens_q[batch_idx];
                int q_len_1 = cum_seq_lens_q[batch_idx + 1] - q_begin_0;
                int rows_remaining_2 = q_len_1 * num_heads - m_tile * 128;
                int valid_rows_3 = ((rows_remaining_2 < 128) ? rows_remaining_2 : 128);
                int k_local_4 = seq_lens[batch_idx];
                int g_bound_5 = causal_global[batch_idx];
                if (num_kv_tiles > 0) {
                    mbarrier_wait(p_epi_free_addr, _phase_p_epi_free_0);
                    _phase_p_epi_free_0 ^= 1;
                    int flat_row = m_tile * 128 + row_in_tile;
                    uint32_t _fast_div_q_14 = (fd_num_heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)flat_row), fd_num_heads.multiplier) >> fd_num_heads.shift_right) : (uint32_t)((unsigned int)flat_row);
                    uint32_t _fast_div_r_14 = (uint32_t)((unsigned int)flat_row) - _fast_div_q_14 * (uint32_t)(fd_num_heads.divisor);
                    int q_idx_row = (int)_fast_div_q_14;
                    int lim_row_num = g_bound_5 - q_len_1 + q_idx_row + 1 - cp_rank;
                    lim_row_num = ((lim_row_num > 0) ? lim_row_num : 0);
                    uint32_t _fast_div_q_15 = (fd_cp_world.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(lim_row_num + cp_world - 1)), fd_cp_world.multiplier) >> fd_cp_world.shift_right) : (uint32_t)((unsigned int)(lim_row_num + cp_world - 1));
                    uint32_t _fast_div_r_15 = (uint32_t)((unsigned int)(lim_row_num + cp_world - 1)) - _fast_div_q_15 * (uint32_t)(fd_cp_world.divisor);
                    int lim_row_ceil = (int)_fast_div_q_15;
                    int key_lim_row = ((lim_row_ceil < k_local_4) ? lim_row_ceil : k_local_4);
                    key_lim_row = ((key_lim_row > 0) ? key_lim_row : 0);
                    int half_col0 = n_half * 64;
                    float row_max_val = -CAKE_INF;
                    float row_sum_val = 0.0f;
                    #pragma unroll 1
                    for (int tile = 0; tile < num_kv_tiles; tile++) {
                        int pipeline_tile = softmax_tile_cursor + tile;
                        int phase = pipeline_tile & 1;
                        int s_wait_phase = pipeline_tile >> 1 & 1;
                        mbarrier_wait(s_full_addr + (phase) * 8, s_wait_phase);
                        int s_off = ((phase != 0) ? 64 : 0);
                        int s_base = taddr + (unsigned int)s_off + (unsigned int)(n_half * 64) + (unsigned int)(tmem_row_base << 16);
                        float _tmem_load_0[64];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                            : "r"(s_base));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_0[32]), "=f"(_tmem_load_0[33]), "=f"(_tmem_load_0[34]), "=f"(_tmem_load_0[35]), "=f"(_tmem_load_0[36]), "=f"(_tmem_load_0[37]), "=f"(_tmem_load_0[38]), "=f"(_tmem_load_0[39]), "=f"(_tmem_load_0[40]), "=f"(_tmem_load_0[41]), "=f"(_tmem_load_0[42]), "=f"(_tmem_load_0[43]), "=f"(_tmem_load_0[44]), "=f"(_tmem_load_0[45]), "=f"(_tmem_load_0[46]), "=f"(_tmem_load_0[47]), "=f"(_tmem_load_0[48]), "=f"(_tmem_load_0[49]), "=f"(_tmem_load_0[50]), "=f"(_tmem_load_0[51]), "=f"(_tmem_load_0[52]), "=f"(_tmem_load_0[53]), "=f"(_tmem_load_0[54]), "=f"(_tmem_load_0[55]), "=f"(_tmem_load_0[56]), "=f"(_tmem_load_0[57]), "=f"(_tmem_load_0[58]), "=f"(_tmem_load_0[59]), "=f"(_tmem_load_0[60]), "=f"(_tmem_load_0[61]), "=f"(_tmem_load_0[62]), "=f"(_tmem_load_0[63])
                            : "r"(s_base + 32));
                        int valid_cols = key_lim_row - (start_tile + tile) * 128 - half_col0;
                        int _vote_0 = __any_sync(0xFFFFFFFF, valid_cols < 64);
                        int need_mask = _vote_0;
                        if (need_mask != 0) {
                            uint32_t _slice_lo_mask_0;
                            {
                                int _lim_0 = valid_cols;
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
                                int _lim_1 = valid_cols - 32;
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
                        }
                        float2 _reg_reduce_max2_2 = {-CAKE_INF, -CAKE_INF};
                        row_max_x32_accum(&_tmem_load_0[0], _reg_reduce_max2_2);
                        row_max_x32_accum(&_tmem_load_0[32], _reg_reduce_max2_2);
                        float _tmem_load_0_max = row_max_reduce(_reg_reduce_max2_2);
                        float _max_0 = max_noftz(_tmem_load_0_max, row_max_val);
                        smem_softmax_exchange[exchange_idx] = _max_0;
                        asm volatile("barrier.sync 2, 128;" ::: "memory");
                        float _max_1 = max_noftz(_max_0, smem_softmax_exchange[exchange_idx ^ 64]);
                        asm volatile("barrier.sync 2, 128;" ::: "memory");
                        float growth = (_max_1 - row_max_val) * softmax_scale_log2;
                        int keep_stale = ((row_max_val > -CAKE_INF) ? 1 : 0);
                        if (keep_stale != 0) {
                            if (growth > 8.0f) {
                                keep_stale = 0;
                            }
                        }
                        float no_correction = ((keep_stale != 0) ? 1.0f : 0.0f);
                        float delta = softmax_scale_log2 * (row_max_val - _max_1);
                        float _exp2_0 = approx_exp2(delta);
                        float exp_delta = _exp2_0;
                        float acc_scale = ((row_max_val > -CAKE_INF) ? exp_delta : 1.0f);
                        if (keep_stale != 0) {
                            acc_scale = 1.0f;
                        } else {
                            row_max_val = _max_1;
                        }
                        float safe_max = ((row_max_val == -CAKE_INF) ? 0.0f : row_max_val);
                        float max_scaled = safe_max * softmax_scale_log2;
                        const float2 _fma_b2_3 = {softmax_scale_log2, softmax_scale_log2};
                        const float2 _fma_c2_4 = {-max_scaled, -max_scaled};
                        #pragma unroll
                        for (int _lf = 0; _lf < 32; _lf++)
                            fma_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_0)[_lf], _fma_b2_3, _fma_c2_4);
                        #pragma unroll
                        for (int _le = 0; _le < 64; _le++) {
                            _tmem_load_0[_le] = approx_exp2(_tmem_load_0[_le]);
                        }
                        float2 _reg_reduce_sum2_5 = make_float2(0.0f, 0.0f);
                        softmax_block_sum(&_tmem_load_0[0], &_reg_reduce_sum2_5);
                        softmax_block_sum(&_tmem_load_0[32], &_reg_reduce_sum2_5);
                        float _tmem_load_0_sum = _reg_reduce_sum2_5.x + _reg_reduce_sum2_5.y;
                        row_sum_val = row_sum_val * acc_scale + _tmem_load_0_sum;
                        int p_cor_empty_phase = pipeline_tile >> 1 & 1 ^ 1;
                        mbarrier_wait(stats_empty_addr + (phase) * 8, p_cor_empty_phase);
                        float meta[4];
                        meta[0] = row_sum_val;
                        meta[1] = row_max_val;
                        meta[2] = acc_scale;
                        meta[3] = no_correction;
                        int meta_addr = taddr + 384 + (unsigned int)(phase * 8) + (unsigned int)(n_half * 4) + (unsigned int)(tmem_row_base << 16);
                        tmem_st_x4_f32(meta_addr, meta);
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        mbarrier_arrive(stats_addr + (phase) * 8);
                        int p_empty_phase = pipeline_tile >> 1 & 1 ^ 1;
                        mbarrier_wait(p_empty_addr + (phase) * 8, p_empty_phase);
                        const int p_row = my_row;
                        uint32_t _tmem_load_0_bf16[32];
                        #pragma unroll
                        for (int _lp = 0; _lp < 32; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                            _tmem_load_0_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int ps_local = 0; ps_local < 2; ps_local++) {
                            int ps = n_half * 2 + ps_local;
                            int p_stage = phase * 4 + ps;
                            int p_base_smem = smem_p_addr + (unsigned int)(p_stage * 4096);
                            #pragma unroll
                            for (int vec = 0; vec < 4; vec++) {
                                int pv_off = (ps_local * 4 + vec) * 4;
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p_base_smem + (p_row * 64 + vec * 16 ^ (p_row * 64 + vec * 16 >> 7 & 3) << 4))), "r"(_tmem_load_0_bf16[pv_off]), "r"(_tmem_load_0_bf16[pv_off + 1]), "r"(_tmem_load_0_bf16[pv_off + 2]), "r"(_tmem_load_0_bf16[pv_off + 3]) : "memory");
                            }
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
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
                }
                softmax_tile_cursor = softmax_tile_cursor + num_kv_tiles;
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            if (warp == 0) {
            }
        }
    }
    // ---- Role: correction_wg ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 208;");
        { // correction_wg_main
            const int wg_dummy_inc_1 = 0;
            {
                asm volatile("griddepcontrol.wait;" ::: "memory");
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
            const int tmem_row_base_1 = warp % 2 * 32;
            const int n_half_1 = warp % 4 / 2;
            const int my_row_1 = tmem_row_base_1 + lane;
            const int corr_row = tmem_row_base_1 << 16;
            int row_in_tile_1 = cta_rank * 64 + my_row_1;
            const int epilogue_exchange_idx = n_half_1 * 64 + my_row_1;
            int correction_tile_cursor = 0;
            float neg_inf = -CAKE_INF;
            float zero_vec[8];
            #pragma unroll
            for (int z_i = 0; z_i < 8; z_i++) {
                zero_vec[z_i] = 0.0f;
            }
            unsigned int work_stage_corr = 0;
            unsigned int first_epi_done = 1;
            int unit_counter_corr = 0;
            unsigned int first_pubh_corr = 1;
            uint32_t _fast_div_q_16 = (fd_num_items.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)cluster_id), fd_num_items.multiplier) >> fd_num_items.shift_right) : (uint32_t)((unsigned int)cluster_id);
            uint32_t _fast_div_r_16 = (uint32_t)((unsigned int)cluster_id) - _fast_div_q_16 * (uint32_t)(fd_num_items.divisor);
            int k_st_1 = (int)_fast_div_q_16;
            int it_st_1 = cluster_id - (unsigned int)(k_st_1 * num_items);
            uint32_t _fast_div_q_17 = (fd_tiles_max.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)it_st_1), fd_tiles_max.multiplier) >> fd_tiles_max.shift_right) : (uint32_t)((unsigned int)it_st_1);
            uint32_t _fast_div_r_17 = (uint32_t)((unsigned int)it_st_1) - _fast_div_q_17 * (uint32_t)(fd_tiles_max.divisor);
            int sb_1 = (int)_fast_div_q_17;
            int sm_1 = it_st_1 - sb_1 * tiles_max;
            int q_begin_1 = cum_seq_lens_q[sb_1];
            int q_len_2 = cum_seq_lens_q[sb_1 + 1] - q_begin_1;
            int rows_remaining_1 = q_len_2 * num_heads - sm_1 * 128;
            int valid_rows_1 = ((rows_remaining_1 < 128) ? rows_remaining_1 : 128);
            int k_local_1 = seq_lens[sb_1];
            int g_bound_1 = causal_global[sb_1];
            int n_t_1 = -1;
            if (valid_rows_1 > 0) {
                uint32_t _fast_div_q_18 = (fd_num_heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(sm_1 * 128 + valid_rows_1 - 1)), fd_num_heads.multiplier) >> fd_num_heads.shift_right) : (uint32_t)((unsigned int)(sm_1 * 128 + valid_rows_1 - 1));
                uint32_t _fast_div_r_18 = (uint32_t)((unsigned int)(sm_1 * 128 + valid_rows_1 - 1)) - _fast_div_q_18 * (uint32_t)(fd_num_heads.divisor);
                int last_q_t_1 = (int)_fast_div_q_18;
                int lim_num_t_1 = g_bound_1 - q_len_2 + last_q_t_1 + 1 - cp_rank;
                lim_num_t_1 = ((lim_num_t_1 > 0) ? lim_num_t_1 : 0);
                uint32_t _fast_div_q_19 = (fd_cp_world.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(lim_num_t_1 + cp_world - 1)), fd_cp_world.multiplier) >> fd_cp_world.shift_right) : (uint32_t)((unsigned int)(lim_num_t_1 + cp_world - 1));
                uint32_t _fast_div_r_19 = (uint32_t)((unsigned int)(lim_num_t_1 + cp_world - 1)) - _fast_div_q_19 * (uint32_t)(fd_cp_world.divisor);
                int lim_ceil_t_1 = (int)_fast_div_q_19;
                int key_lim_t_1 = ((lim_ceil_t_1 < k_local_1) ? lim_ceil_t_1 : k_local_1);
                key_lim_t_1 = ((key_lim_t_1 > 0) ? key_lim_t_1 : 0);
                n_t_1 = (key_lim_t_1 + 128 - 1) / 128;
            }
            int n_all_1 = n_t_1;
            int n_st_1 = n_all_1 - k_st_1 * static_tiles;
            n_st_1 = ((n_st_1 < static_tiles) ? n_st_1 : static_tiles);
            n_st_1 = ((n_st_1 > 0) ? n_st_1 : 0);
            int has_st_1 = 1;
            if (k_st_1 > 0) {
                has_st_1 = ((n_st_1 > 0) ? 1 : 0);
            }
            if (k_st_1 >= 8) {
                has_st_1 = 0;
            }
            int has_static_corr = has_st_1;
            int no_token_corr = ((0) ? static_only : 0);
            int par_corr = 0;
            unsigned int _phase_work_full_1 = 0;
            unsigned int _phase_o_full_0 = 0;
            unsigned int _phase_plan_full_0 = 0;
            unsigned int _phase_pub_free_0 = 1;
            #pragma unroll 1
            for (unsigned int _unit_iter_corr = 0; _unit_iter_corr < max_units + 1; _unit_iter_corr++) {
                unsigned int tok_valid_1 = 1;
                int batch_idx_1 = 0;
                int m_tile_2 = 0;
                int start_tile_1 = 0;
                int num_kv_tiles_1 = 0;
                int split_idx_1 = 0;
                int item_units_1 = 1;
                int slot_base_1 = 0;
                int is_static_1 = has_static_corr;
                has_static_corr = 0;
                if (is_static_1 != 0) {
                    batch_idx_1 = sb_1;
                    m_tile_2 = sm_1;
                    num_kv_tiles_1 = n_st_1;
                    start_tile_1 = k_st_1 * static_tiles;
                    split_idx_1 = k_st_1;
                } else if (no_token_corr != 0) {
                    tok_valid_1 = 0;
                } else {
                    mbarrier_wait_cluster(work_full_addr + (work_stage_corr) * 8, _phase_work_full_1);
                    unsigned int tok_words_1[8];
                    #pragma unroll
                    for (int w_1 = 0; w_1 < 8; w_1++) {
                        tok_words_1[w_1] = 0;
                    }
                    if (lane == 0) {
                        uint32_t _mapa_3;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mapa_3) : "r"(work_token_words_addr + work_stage_corr * 32), "r"(0));
                        asm volatile("ld.shared::cluster.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tok_words_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&tok_words_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tok_words_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tok_words_1[(0) + 3]))
                            : "r"(_mapa_3));
                        asm volatile("ld.shared::cluster.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tok_words_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&tok_words_1[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tok_words_1[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tok_words_1[(4) + 3]))
                            : "r"(_mapa_3 + 16));
                    }
                    unsigned int _shfl_24 = __shfl_sync(0xFFFFFFFF, tok_words_1[0], 0);
                    unsigned int valid_1 = _shfl_24;
                    unsigned int _shfl_25 = __shfl_sync(0xFFFFFFFF, tok_words_1[1], 0);
                    int batch_idx_0_1 = (int)_shfl_25;
                    unsigned int _shfl_26 = __shfl_sync(0xFFFFFFFF, tok_words_1[2], 0);
                    int m_tile_1_1 = (int)_shfl_26;
                    unsigned int _shfl_27 = __shfl_sync(0xFFFFFFFF, tok_words_1[3], 0);
                    int start_tile_2_1 = (int)_shfl_27;
                    unsigned int _shfl_28 = __shfl_sync(0xFFFFFFFF, tok_words_1[4], 0);
                    int num_kv_tiles_3_1 = (int)_shfl_28;
                    unsigned int _shfl_29 = __shfl_sync(0xFFFFFFFF, tok_words_1[5], 0);
                    int unit_idx_1 = (int)_shfl_29;
                    unsigned int _shfl_30 = __shfl_sync(0xFFFFFFFF, tok_words_1[6], 0);
                    int units_1 = (int)_shfl_30;
                    unsigned int _shfl_31 = __shfl_sync(0xFFFFFFFF, tok_words_1[7], 0);
                    int slot_base_4_1 = (int)_shfl_31;
                    if (lane == 0) {
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                            "}"
                            :: "r"(work_empty_addr + work_stage_corr * 8), "r"(0) : "memory");
                    }
                    work_stage_corr += 1;
                    if (work_stage_corr == 2) { work_stage_corr = 0; _phase_work_full_1 ^= 1; }
                    tok_valid_1 = valid_1;
                    batch_idx_1 = batch_idx_0_1;
                    m_tile_2 = m_tile_1_1;
                    start_tile_1 = start_tile_2_1;
                    num_kv_tiles_1 = num_kv_tiles_3_1;
                    split_idx_1 = unit_idx_1;
                    item_units_1 = units_1;
                    slot_base_1 = slot_base_4_1;
                }
                if (tok_valid_1 == 0) {
                    break;
                }
                int q_begin_0_1 = cum_seq_lens_q[batch_idx_1];
                int q_len_1_1 = cum_seq_lens_q[batch_idx_1 + 1] - q_begin_0_1;
                int rows_remaining_2_1 = q_len_1_1 * num_heads - m_tile_2 * 128;
                int valid_rows_3_1 = ((rows_remaining_2_1 < 128) ? rows_remaining_2_1 : 128);
                int k_local_4_1 = seq_lens[batch_idx_1];
                int g_bound_5_1 = causal_global[batch_idx_1];
                int flat_row_1 = m_tile_2 * 128 + row_in_tile_1;
                int compact_row = q_begin_0_1 * num_heads + flat_row_1;
                int slot_row = (slot_base_1 + split_idx_1) * 128 + row_in_tile_1;
                if (num_kv_tiles_1 > 0) {
                    float final_sum_val = 0.0f;
                    float final_max_val = -CAKE_INF;
                    #pragma unroll 1
                    for (int tile_1 = 0; tile_1 < num_kv_tiles_1; tile_1++) {
                        int pipeline_tile_1 = correction_tile_cursor + tile_1;
                        int phase_1 = pipeline_tile_1 & 1;
                        int stats_wait_phase = pipeline_tile_1 >> 1 & 1;
                        mbarrier_wait(stats_addr + (phase_1) * 8, stats_wait_phase);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int meta_addr_1 = taddr + 384 + (unsigned int)(phase_1 * 8) + (unsigned int)(n_half_1 * 4) + (unsigned int)corr_row;
                        float _tmem_load_1[4];
                        tmem_ld_x4(&_tmem_load_1[0], meta_addr_1);
                        final_sum_val = _tmem_load_1[0];
                        final_max_val = _tmem_load_1[1];
                        float acc_scale_1 = _tmem_load_1[2];
                        float no_correction_1 = _tmem_load_1[3];
                        if (tile_1 > 0) {
                            mbarrier_wait(o_full_addr, _phase_o_full_0);
                            _phase_o_full_0 ^= 1;
                            asm volatile("tcgen05.fence::after_thread_sync;");
                        }
                        if (tile_1 > 0) {
                            int _vote_1 = __all_sync(0xFFFFFFFF, no_correction_1 == 1.0f);
                            int skip_correction = _vote_1;
                            if (skip_correction == 0) {
                                #pragma unroll
                                for (int vs_local = 0; vs_local < 2; vs_local++) {
                                    #pragma unroll
                                    for (int acc_stage = 0; acc_stage < 2; acc_stage++) {
                                        int o_base = taddr + 128 + (unsigned int)(acc_stage * 128) + (unsigned int)(vs_local * 64) + (unsigned int)corr_row;
                                        #pragma unroll
                                        for (int c = 0; c < 64; c += 16) {
                                            float _tmem_load_2[16];
                                            tmem_ld_x16(&_tmem_load_2[0], o_base + c);
                                            const float2 _scale2_0 = {acc_scale_1, acc_scale_1};
                                            #pragma unroll
                                            for (int _ls = 0; _ls < 8; _ls++)
                                                mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_2)[_ls], _scale2_0);
                                            tmem_st_x16_f32(o_base + c, _tmem_load_2);
                                        }
                                    }
                                }
                            }
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
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
                    mbarrier_wait(o_full_addr, _phase_o_full_0);
                    _phase_o_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    smem_epilogue_exchange[epilogue_exchange_idx] = final_sum_val;
                    asm volatile("barrier.sync 3, 128;" ::: "memory");
                    final_sum_val = final_sum_val + smem_epilogue_exchange[epilogue_exchange_idx ^ 64];
                    asm volatile("barrier.sync 3, 128;" ::: "memory");
                    float _rcp_0 = approx_rcp(final_sum_val);
                    float inv_sum = ((final_sum_val > 0.0f) ? _rcp_0 : 0.0f);
                    float _log2_0;
                    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(final_sum_val));
                    float lse_log2 = final_max_val * softmax_scale_log2 + _log2_0;
                    lse_log2 = ((final_sum_val > 0.0f) ? lse_log2 : neg_inf);
                    if (is_static_1 != 0) {
                        if (n_all_1 > static_tiles) {
                            mbarrier_wait_cluster(plan_full_addr, _phase_plan_full_0);
                            _phase_plan_full_0 ^= 1;
                            unsigned int plan_w[4];
                            #pragma unroll
                            for (int pw_i = 0; pw_i < 4; pw_i++) {
                                plan_w[pw_i] = 0;
                            }
                            if (lane == 0) {
                                uint32_t _mapa_4;
                                asm volatile(
                                    "mapa.shared::cluster.u32 %0, %1, %2;"
                                    : "=r"(_mapa_4) : "r"(plan_words_addr), "r"(0));
                                asm volatile("ld.shared::cluster.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&plan_w[0])), "=r"(*reinterpret_cast<uint32_t*>(&plan_w[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&plan_w[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&plan_w[(0) + 3]))
                                    : "r"(_mapa_4));
                            }
                            unsigned int _shfl_32 = __shfl_sync(0xFFFFFFFF, plan_w[0], 0);
                            item_units_1 = (int)_shfl_32;
                            unsigned int _shfl_33 = __shfl_sync(0xFFFFFFFF, plan_w[1], 0);
                            slot_base_1 = (int)_shfl_33;
                            slot_row = (slot_base_1 + split_idx_1) * 128 + row_in_tile_1;
                        }
                    }
                    if (item_units_1 == 1) {
                        int row_valid = row_in_tile_1 < valid_rows_3_1;
                        if (row_valid != 0) {
                            *(reinterpret_cast<float*>(LSE + compact_row) + (0)) = lse_log2 * 0.6931471805599453f;
                        }
                        int o_offset = compact_row * 512;
                        int cta_full = ((valid_rows_3_1 >= (cta_rank + 1) * 64) ? 1 : 0);
                        if (cta_full != 0) {
                            #pragma unroll
                            for (int h0 = 0; h0 < 2; h0++) {
                                int acc_e = h0 / 2;
                                int sl_e = h0 - acc_e * 2;
                                int o_base_e = taddr + 128 + (unsigned int)(acc_e * 128) + (unsigned int)(sl_e * 64) + (unsigned int)corr_row;
                                float _tmem_load_3[32];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                    : "=f"(_tmem_load_3[0]), "=f"(_tmem_load_3[1]), "=f"(_tmem_load_3[2]), "=f"(_tmem_load_3[3]), "=f"(_tmem_load_3[4]), "=f"(_tmem_load_3[5]), "=f"(_tmem_load_3[6]), "=f"(_tmem_load_3[7]), "=f"(_tmem_load_3[8]), "=f"(_tmem_load_3[9]), "=f"(_tmem_load_3[10]), "=f"(_tmem_load_3[11]), "=f"(_tmem_load_3[12]), "=f"(_tmem_load_3[13]), "=f"(_tmem_load_3[14]), "=f"(_tmem_load_3[15]), "=f"(_tmem_load_3[16]), "=f"(_tmem_load_3[17]), "=f"(_tmem_load_3[18]), "=f"(_tmem_load_3[19]), "=f"(_tmem_load_3[20]), "=f"(_tmem_load_3[21]), "=f"(_tmem_load_3[22]), "=f"(_tmem_load_3[23]), "=f"(_tmem_load_3[24]), "=f"(_tmem_load_3[25]), "=f"(_tmem_load_3[26]), "=f"(_tmem_load_3[27]), "=f"(_tmem_load_3[28]), "=f"(_tmem_load_3[29]), "=f"(_tmem_load_3[30]), "=f"(_tmem_load_3[31])
                                    : "r"(o_base_e));
                                float _tmem_load_4[32];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                    : "=f"(_tmem_load_4[0]), "=f"(_tmem_load_4[1]), "=f"(_tmem_load_4[2]), "=f"(_tmem_load_4[3]), "=f"(_tmem_load_4[4]), "=f"(_tmem_load_4[5]), "=f"(_tmem_load_4[6]), "=f"(_tmem_load_4[7]), "=f"(_tmem_load_4[8]), "=f"(_tmem_load_4[9]), "=f"(_tmem_load_4[10]), "=f"(_tmem_load_4[11]), "=f"(_tmem_load_4[12]), "=f"(_tmem_load_4[13]), "=f"(_tmem_load_4[14]), "=f"(_tmem_load_4[15]), "=f"(_tmem_load_4[16]), "=f"(_tmem_load_4[17]), "=f"(_tmem_load_4[18]), "=f"(_tmem_load_4[19]), "=f"(_tmem_load_4[20]), "=f"(_tmem_load_4[21]), "=f"(_tmem_load_4[22]), "=f"(_tmem_load_4[23]), "=f"(_tmem_load_4[24]), "=f"(_tmem_load_4[25]), "=f"(_tmem_load_4[26]), "=f"(_tmem_load_4[27]), "=f"(_tmem_load_4[28]), "=f"(_tmem_load_4[29]), "=f"(_tmem_load_4[30]), "=f"(_tmem_load_4[31])
                                    : "r"(o_base_e + 32));
                                float sa_e[32];
                                float sb_e[32];
                                #pragma unroll
                                for (int ie = 0; ie < 32; ie++) {
                                    sa_e[ie] = _tmem_load_3[ie] * inv_sum;
                                    sb_e[ie] = _tmem_load_4[ie] * inv_sum;
                                }
                                uint32_t sa_e_bf16[16];
                                #pragma unroll
                                for (int _lp = 0; _lp < 16; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sa_e[_lp*2 + 0], sa_e[_lp*2+1 + 0]));
                                    sa_e_bf16[_lp] = *(uint32_t*)&_bf2;
                                }
                                uint32_t sb_e_bf16[16];
                                #pragma unroll
                                for (int _lp = 0; _lp < 16; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sb_e[_lp*2 + 0], sb_e[_lp*2+1 + 0]));
                                    sb_e_bf16[_lp] = *(uint32_t*)&_bf2;
                                }
                                int hb_e = h0 % 2;
                                int stage_e = hb_e * 2 + n_half_1;
                                int sbase_e = smem_ostage_addr + (unsigned int)(stage_e * 8192);
                                #pragma unroll
                                for (int ce = 0; ce < 4; ce++) {
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sbase_e + (my_row_1 * 128 + ce * 16 ^ (my_row_1 * 128 + ce * 16 >> 7 & 7) << 4))), "r"(sa_e_bf16[ce * 4]), "r"(sa_e_bf16[ce * 4 + 1]), "r"(sa_e_bf16[ce * 4 + 2]), "r"(sa_e_bf16[ce * 4 + 3]) : "memory");
                                }
                                #pragma unroll
                                for (int ce2 = 0; ce2 < 4; ce2++) {
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sbase_e + (my_row_1 * 128 + (64 + ce2 * 16) ^ (my_row_1 * 128 + (64 + ce2 * 16) >> 7 & 7) << 4))), "r"(sb_e_bf16[ce2 * 4]), "r"(sb_e_bf16[ce2 * 4 + 1]), "r"(sb_e_bf16[ce2 * 4 + 2]), "r"(sb_e_bf16[ce2 * 4 + 3]) : "memory");
                                }
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile("barrier.sync 3, 128;" ::: "memory");
                                if (warp == 4) {
                                    if (elect_sync()) {
                                        #pragma unroll
                                        for (int nh_b = 0; nh_b < 2; nh_b++) {
                                            int col_b = (acc_e * 4 + nh_b * 2 + sl_e) * 64;
                                            tma_store_2d((&tmap_o), col_b, q_begin_0_1 * num_heads + m_tile_2 * 128 + cta_rank * 64, smem_ostage_addr + (unsigned int)((hb_e * 2 + nh_b) * 8192));
                                        }
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                }
                            }
                            #pragma unroll
                            for (int h1 = 2; h1 < 4; h1++) {
                                int acc_e_1 = h1 / 2;
                                int sl_e_1 = h1 - acc_e_1 * 2;
                                int o_base_e_1 = taddr + 128 + (unsigned int)(acc_e_1 * 128) + (unsigned int)(sl_e_1 * 64) + (unsigned int)corr_row;
                                float _tmem_load_5[32];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                    : "=f"(_tmem_load_5[0]), "=f"(_tmem_load_5[1]), "=f"(_tmem_load_5[2]), "=f"(_tmem_load_5[3]), "=f"(_tmem_load_5[4]), "=f"(_tmem_load_5[5]), "=f"(_tmem_load_5[6]), "=f"(_tmem_load_5[7]), "=f"(_tmem_load_5[8]), "=f"(_tmem_load_5[9]), "=f"(_tmem_load_5[10]), "=f"(_tmem_load_5[11]), "=f"(_tmem_load_5[12]), "=f"(_tmem_load_5[13]), "=f"(_tmem_load_5[14]), "=f"(_tmem_load_5[15]), "=f"(_tmem_load_5[16]), "=f"(_tmem_load_5[17]), "=f"(_tmem_load_5[18]), "=f"(_tmem_load_5[19]), "=f"(_tmem_load_5[20]), "=f"(_tmem_load_5[21]), "=f"(_tmem_load_5[22]), "=f"(_tmem_load_5[23]), "=f"(_tmem_load_5[24]), "=f"(_tmem_load_5[25]), "=f"(_tmem_load_5[26]), "=f"(_tmem_load_5[27]), "=f"(_tmem_load_5[28]), "=f"(_tmem_load_5[29]), "=f"(_tmem_load_5[30]), "=f"(_tmem_load_5[31])
                                    : "r"(o_base_e_1));
                                float _tmem_load_6[32];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                    : "=f"(_tmem_load_6[0]), "=f"(_tmem_load_6[1]), "=f"(_tmem_load_6[2]), "=f"(_tmem_load_6[3]), "=f"(_tmem_load_6[4]), "=f"(_tmem_load_6[5]), "=f"(_tmem_load_6[6]), "=f"(_tmem_load_6[7]), "=f"(_tmem_load_6[8]), "=f"(_tmem_load_6[9]), "=f"(_tmem_load_6[10]), "=f"(_tmem_load_6[11]), "=f"(_tmem_load_6[12]), "=f"(_tmem_load_6[13]), "=f"(_tmem_load_6[14]), "=f"(_tmem_load_6[15]), "=f"(_tmem_load_6[16]), "=f"(_tmem_load_6[17]), "=f"(_tmem_load_6[18]), "=f"(_tmem_load_6[19]), "=f"(_tmem_load_6[20]), "=f"(_tmem_load_6[21]), "=f"(_tmem_load_6[22]), "=f"(_tmem_load_6[23]), "=f"(_tmem_load_6[24]), "=f"(_tmem_load_6[25]), "=f"(_tmem_load_6[26]), "=f"(_tmem_load_6[27]), "=f"(_tmem_load_6[28]), "=f"(_tmem_load_6[29]), "=f"(_tmem_load_6[30]), "=f"(_tmem_load_6[31])
                                    : "r"(o_base_e_1 + 32));
                                {
                                    if (warp == 4) {
                                        if (elect_sync()) {
                                            asm volatile("cp.async.bulk.wait_group.read 1;");
                                        }
                                    }
                                    asm volatile("barrier.sync 3, 128;" ::: "memory");
                                }
                                float sa_e_1[32];
                                float sb_e_1[32];
                                #pragma unroll
                                for (int ie_1 = 0; ie_1 < 32; ie_1++) {
                                    sa_e_1[ie_1] = _tmem_load_5[ie_1] * inv_sum;
                                    sb_e_1[ie_1] = _tmem_load_6[ie_1] * inv_sum;
                                }
                                uint32_t sa_e_bf16_1[16];
                                #pragma unroll
                                for (int _lp = 0; _lp < 16; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sa_e_1[_lp*2 + 0], sa_e_1[_lp*2+1 + 0]));
                                    sa_e_bf16_1[_lp] = *(uint32_t*)&_bf2;
                                }
                                uint32_t sb_e_bf16_1[16];
                                #pragma unroll
                                for (int _lp = 0; _lp < 16; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sb_e_1[_lp*2 + 0], sb_e_1[_lp*2+1 + 0]));
                                    sb_e_bf16_1[_lp] = *(uint32_t*)&_bf2;
                                }
                                int hb_e_1 = h1 % 2;
                                int stage_e_1 = hb_e_1 * 2 + n_half_1;
                                int sbase_e_1 = smem_ostage_addr + (unsigned int)(stage_e_1 * 8192);
                                #pragma unroll
                                for (int ce_1 = 0; ce_1 < 4; ce_1++) {
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sbase_e_1 + (my_row_1 * 128 + ce_1 * 16 ^ (my_row_1 * 128 + ce_1 * 16 >> 7 & 7) << 4))), "r"(sa_e_bf16_1[ce_1 * 4]), "r"(sa_e_bf16_1[ce_1 * 4 + 1]), "r"(sa_e_bf16_1[ce_1 * 4 + 2]), "r"(sa_e_bf16_1[ce_1 * 4 + 3]) : "memory");
                                }
                                #pragma unroll
                                for (int ce2_1 = 0; ce2_1 < 4; ce2_1++) {
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sbase_e_1 + (my_row_1 * 128 + (64 + ce2_1 * 16) ^ (my_row_1 * 128 + (64 + ce2_1 * 16) >> 7 & 7) << 4))), "r"(sb_e_bf16_1[ce2_1 * 4]), "r"(sb_e_bf16_1[ce2_1 * 4 + 1]), "r"(sb_e_bf16_1[ce2_1 * 4 + 2]), "r"(sb_e_bf16_1[ce2_1 * 4 + 3]) : "memory");
                                }
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile("barrier.sync 3, 128;" ::: "memory");
                                if (warp == 4) {
                                    if (elect_sync()) {
                                        #pragma unroll
                                        for (int nh_b_1 = 0; nh_b_1 < 2; nh_b_1++) {
                                            int col_b_1 = (acc_e_1 * 4 + nh_b_1 * 2 + sl_e_1) * 64;
                                            tma_store_2d((&tmap_o), col_b_1, q_begin_0_1 * num_heads + m_tile_2 * 128 + cta_rank * 64, smem_ostage_addr + (unsigned int)((hb_e_1 * 2 + nh_b_1) * 8192));
                                        }
                                        asm volatile("cp.async.bulk.commit_group;");
                                    }
                                }
                            }
                            if (warp == 4) {
                                if (elect_sync()) {
                                    asm volatile("cp.async.bulk.wait_group.read 0;");
                                }
                            }
                            asm volatile("barrier.sync 3, 128;" ::: "memory");
                        } else {
                            #pragma unroll
                            for (int acc_stage_1 = 0; acc_stage_1 < 2; acc_stage_1++) {
                                int o_base_e0 = taddr + 128 + (unsigned int)(acc_stage_1 * 128) + (unsigned int)corr_row;
                                int lvs_base = acc_stage_1 * 2 * 2 + n_half_1 * 2;
                                int gmem_base_0 = o_offset + lvs_base * 64;
                                float _tmem_load_7[32];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                    : "=f"(_tmem_load_7[0]), "=f"(_tmem_load_7[1]), "=f"(_tmem_load_7[2]), "=f"(_tmem_load_7[3]), "=f"(_tmem_load_7[4]), "=f"(_tmem_load_7[5]), "=f"(_tmem_load_7[6]), "=f"(_tmem_load_7[7]), "=f"(_tmem_load_7[8]), "=f"(_tmem_load_7[9]), "=f"(_tmem_load_7[10]), "=f"(_tmem_load_7[11]), "=f"(_tmem_load_7[12]), "=f"(_tmem_load_7[13]), "=f"(_tmem_load_7[14]), "=f"(_tmem_load_7[15]), "=f"(_tmem_load_7[16]), "=f"(_tmem_load_7[17]), "=f"(_tmem_load_7[18]), "=f"(_tmem_load_7[19]), "=f"(_tmem_load_7[20]), "=f"(_tmem_load_7[21]), "=f"(_tmem_load_7[22]), "=f"(_tmem_load_7[23]), "=f"(_tmem_load_7[24]), "=f"(_tmem_load_7[25]), "=f"(_tmem_load_7[26]), "=f"(_tmem_load_7[27]), "=f"(_tmem_load_7[28]), "=f"(_tmem_load_7[29]), "=f"(_tmem_load_7[30]), "=f"(_tmem_load_7[31])
                                    : "r"(o_base_e0));
                                float _tmem_load_8[32];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                    : "=f"(_tmem_load_8[0]), "=f"(_tmem_load_8[1]), "=f"(_tmem_load_8[2]), "=f"(_tmem_load_8[3]), "=f"(_tmem_load_8[4]), "=f"(_tmem_load_8[5]), "=f"(_tmem_load_8[6]), "=f"(_tmem_load_8[7]), "=f"(_tmem_load_8[8]), "=f"(_tmem_load_8[9]), "=f"(_tmem_load_8[10]), "=f"(_tmem_load_8[11]), "=f"(_tmem_load_8[12]), "=f"(_tmem_load_8[13]), "=f"(_tmem_load_8[14]), "=f"(_tmem_load_8[15]), "=f"(_tmem_load_8[16]), "=f"(_tmem_load_8[17]), "=f"(_tmem_load_8[18]), "=f"(_tmem_load_8[19]), "=f"(_tmem_load_8[20]), "=f"(_tmem_load_8[21]), "=f"(_tmem_load_8[22]), "=f"(_tmem_load_8[23]), "=f"(_tmem_load_8[24]), "=f"(_tmem_load_8[25]), "=f"(_tmem_load_8[26]), "=f"(_tmem_load_8[27]), "=f"(_tmem_load_8[28]), "=f"(_tmem_load_8[29]), "=f"(_tmem_load_8[30]), "=f"(_tmem_load_8[31])
                                    : "r"(o_base_e0 + 32));
                                if (row_valid != 0) {
                                    {
                                        #pragma unroll
                                        for (int j = 0; j < 32; j += 16) {
                                            {
                                                const float2 _prescale2_1 = {inv_sum, inv_sum};
                                                #if __CUDA_ARCH__ >= 1000
                                                #pragma unroll
                                                for (int _ps = 0; _ps < 8; _ps++)
                                                    mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_7[j])[_ps], _prescale2_1);
                                                #else
                                                #pragma unroll
                                                for (int _ps = 0; _ps < 16; _ps++)
                                                    _tmem_load_7[j + _ps] *= inv_sum;
                                                #endif
                                                {
                                                    __nv_bfloat162 _pk0 = __floats2bfloat162_rn(_tmem_load_7[j + 0], _tmem_load_7[j + 1]);
                                                    unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                                                    __nv_bfloat162 _pk1 = __floats2bfloat162_rn(_tmem_load_7[j + 2], _tmem_load_7[j + 3]);
                                                    unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                                                    __nv_bfloat162 _pk2 = __floats2bfloat162_rn(_tmem_load_7[j + 4], _tmem_load_7[j + 5]);
                                                    unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                                                    __nv_bfloat162 _pk3 = __floats2bfloat162_rn(_tmem_load_7[j + 6], _tmem_load_7[j + 7]);
                                                    unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                                                    __nv_bfloat162 _pk4 = __floats2bfloat162_rn(_tmem_load_7[j + 8], _tmem_load_7[j + 9]);
                                                    unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                                                    __nv_bfloat162 _pk5 = __floats2bfloat162_rn(_tmem_load_7[j + 10], _tmem_load_7[j + 11]);
                                                    unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                                                    __nv_bfloat162 _pk6 = __floats2bfloat162_rn(_tmem_load_7[j + 12], _tmem_load_7[j + 13]);
                                                    unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                                                    __nv_bfloat162 _pk7 = __floats2bfloat162_rn(_tmem_load_7[j + 14], _tmem_load_7[j + 15]);
                                                    unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                                                    asm volatile(
                                                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                                        :: "l"((void*)(&((__nv_bfloat16*)(O + (gmem_base_0 + j)))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                                                }
                                            }
                                        }
                                        #pragma unroll
                                        for (int j2 = 0; j2 < 32; j2 += 16) {
                                            {
                                                const float2 _prescale2_2 = {inv_sum, inv_sum};
                                                #if __CUDA_ARCH__ >= 1000
                                                #pragma unroll
                                                for (int _ps = 0; _ps < 8; _ps++)
                                                    mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_8[j2])[_ps], _prescale2_2);
                                                #else
                                                #pragma unroll
                                                for (int _ps = 0; _ps < 16; _ps++)
                                                    _tmem_load_8[j2 + _ps] *= inv_sum;
                                                #endif
                                                {
                                                    __nv_bfloat162 _pk0 = __floats2bfloat162_rn(_tmem_load_8[j2 + 0], _tmem_load_8[j2 + 1]);
                                                    unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                                                    __nv_bfloat162 _pk1 = __floats2bfloat162_rn(_tmem_load_8[j2 + 2], _tmem_load_8[j2 + 3]);
                                                    unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                                                    __nv_bfloat162 _pk2 = __floats2bfloat162_rn(_tmem_load_8[j2 + 4], _tmem_load_8[j2 + 5]);
                                                    unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                                                    __nv_bfloat162 _pk3 = __floats2bfloat162_rn(_tmem_load_8[j2 + 6], _tmem_load_8[j2 + 7]);
                                                    unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                                                    __nv_bfloat162 _pk4 = __floats2bfloat162_rn(_tmem_load_8[j2 + 8], _tmem_load_8[j2 + 9]);
                                                    unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                                                    __nv_bfloat162 _pk5 = __floats2bfloat162_rn(_tmem_load_8[j2 + 10], _tmem_load_8[j2 + 11]);
                                                    unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                                                    __nv_bfloat162 _pk6 = __floats2bfloat162_rn(_tmem_load_8[j2 + 12], _tmem_load_8[j2 + 13]);
                                                    unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                                                    __nv_bfloat162 _pk7 = __floats2bfloat162_rn(_tmem_load_8[j2 + 14], _tmem_load_8[j2 + 15]);
                                                    unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                                                    asm volatile(
                                                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                                        :: "l"((void*)(&((__nv_bfloat16*)(O + (gmem_base_0 + 32 + j2)))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                                                }
                                            }
                                        }
                                    }
                                }
                                float _tmem_load_9[32];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                    : "=f"(_tmem_load_9[0]), "=f"(_tmem_load_9[1]), "=f"(_tmem_load_9[2]), "=f"(_tmem_load_9[3]), "=f"(_tmem_load_9[4]), "=f"(_tmem_load_9[5]), "=f"(_tmem_load_9[6]), "=f"(_tmem_load_9[7]), "=f"(_tmem_load_9[8]), "=f"(_tmem_load_9[9]), "=f"(_tmem_load_9[10]), "=f"(_tmem_load_9[11]), "=f"(_tmem_load_9[12]), "=f"(_tmem_load_9[13]), "=f"(_tmem_load_9[14]), "=f"(_tmem_load_9[15]), "=f"(_tmem_load_9[16]), "=f"(_tmem_load_9[17]), "=f"(_tmem_load_9[18]), "=f"(_tmem_load_9[19]), "=f"(_tmem_load_9[20]), "=f"(_tmem_load_9[21]), "=f"(_tmem_load_9[22]), "=f"(_tmem_load_9[23]), "=f"(_tmem_load_9[24]), "=f"(_tmem_load_9[25]), "=f"(_tmem_load_9[26]), "=f"(_tmem_load_9[27]), "=f"(_tmem_load_9[28]), "=f"(_tmem_load_9[29]), "=f"(_tmem_load_9[30]), "=f"(_tmem_load_9[31])
                                    : "r"(o_base_e0 + 64));
                                float _tmem_load_10[32];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                    : "=f"(_tmem_load_10[0]), "=f"(_tmem_load_10[1]), "=f"(_tmem_load_10[2]), "=f"(_tmem_load_10[3]), "=f"(_tmem_load_10[4]), "=f"(_tmem_load_10[5]), "=f"(_tmem_load_10[6]), "=f"(_tmem_load_10[7]), "=f"(_tmem_load_10[8]), "=f"(_tmem_load_10[9]), "=f"(_tmem_load_10[10]), "=f"(_tmem_load_10[11]), "=f"(_tmem_load_10[12]), "=f"(_tmem_load_10[13]), "=f"(_tmem_load_10[14]), "=f"(_tmem_load_10[15]), "=f"(_tmem_load_10[16]), "=f"(_tmem_load_10[17]), "=f"(_tmem_load_10[18]), "=f"(_tmem_load_10[19]), "=f"(_tmem_load_10[20]), "=f"(_tmem_load_10[21]), "=f"(_tmem_load_10[22]), "=f"(_tmem_load_10[23]), "=f"(_tmem_load_10[24]), "=f"(_tmem_load_10[25]), "=f"(_tmem_load_10[26]), "=f"(_tmem_load_10[27]), "=f"(_tmem_load_10[28]), "=f"(_tmem_load_10[29]), "=f"(_tmem_load_10[30]), "=f"(_tmem_load_10[31])
                                    : "r"(o_base_e0 + 64 + 32));
                                if (row_valid != 0) {
                                    {
                                        #pragma unroll
                                        for (int j_1 = 0; j_1 < 32; j_1 += 16) {
                                            {
                                                const float2 _prescale2_3 = {inv_sum, inv_sum};
                                                #if __CUDA_ARCH__ >= 1000
                                                #pragma unroll
                                                for (int _ps = 0; _ps < 8; _ps++)
                                                    mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_9[j_1])[_ps], _prescale2_3);
                                                #else
                                                #pragma unroll
                                                for (int _ps = 0; _ps < 16; _ps++)
                                                    _tmem_load_9[j_1 + _ps] *= inv_sum;
                                                #endif
                                                {
                                                    __nv_bfloat162 _pk0 = __floats2bfloat162_rn(_tmem_load_9[j_1 + 0], _tmem_load_9[j_1 + 1]);
                                                    unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                                                    __nv_bfloat162 _pk1 = __floats2bfloat162_rn(_tmem_load_9[j_1 + 2], _tmem_load_9[j_1 + 3]);
                                                    unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                                                    __nv_bfloat162 _pk2 = __floats2bfloat162_rn(_tmem_load_9[j_1 + 4], _tmem_load_9[j_1 + 5]);
                                                    unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                                                    __nv_bfloat162 _pk3 = __floats2bfloat162_rn(_tmem_load_9[j_1 + 6], _tmem_load_9[j_1 + 7]);
                                                    unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                                                    __nv_bfloat162 _pk4 = __floats2bfloat162_rn(_tmem_load_9[j_1 + 8], _tmem_load_9[j_1 + 9]);
                                                    unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                                                    __nv_bfloat162 _pk5 = __floats2bfloat162_rn(_tmem_load_9[j_1 + 10], _tmem_load_9[j_1 + 11]);
                                                    unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                                                    __nv_bfloat162 _pk6 = __floats2bfloat162_rn(_tmem_load_9[j_1 + 12], _tmem_load_9[j_1 + 13]);
                                                    unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                                                    __nv_bfloat162 _pk7 = __floats2bfloat162_rn(_tmem_load_9[j_1 + 14], _tmem_load_9[j_1 + 15]);
                                                    unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                                                    asm volatile(
                                                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                                        :: "l"((void*)(&((__nv_bfloat16*)(O + (gmem_base_0 + 64 + j_1)))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                                                }
                                            }
                                        }
                                        #pragma unroll
                                        for (int j2_1 = 0; j2_1 < 32; j2_1 += 16) {
                                            {
                                                const float2 _prescale2_4 = {inv_sum, inv_sum};
                                                #if __CUDA_ARCH__ >= 1000
                                                #pragma unroll
                                                for (int _ps = 0; _ps < 8; _ps++)
                                                    mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_10[j2_1])[_ps], _prescale2_4);
                                                #else
                                                #pragma unroll
                                                for (int _ps = 0; _ps < 16; _ps++)
                                                    _tmem_load_10[j2_1 + _ps] *= inv_sum;
                                                #endif
                                                {
                                                    __nv_bfloat162 _pk0 = __floats2bfloat162_rn(_tmem_load_10[j2_1 + 0], _tmem_load_10[j2_1 + 1]);
                                                    unsigned _pk_u0 = *reinterpret_cast<unsigned*>(&_pk0);
                                                    __nv_bfloat162 _pk1 = __floats2bfloat162_rn(_tmem_load_10[j2_1 + 2], _tmem_load_10[j2_1 + 3]);
                                                    unsigned _pk_u1 = *reinterpret_cast<unsigned*>(&_pk1);
                                                    __nv_bfloat162 _pk2 = __floats2bfloat162_rn(_tmem_load_10[j2_1 + 4], _tmem_load_10[j2_1 + 5]);
                                                    unsigned _pk_u2 = *reinterpret_cast<unsigned*>(&_pk2);
                                                    __nv_bfloat162 _pk3 = __floats2bfloat162_rn(_tmem_load_10[j2_1 + 6], _tmem_load_10[j2_1 + 7]);
                                                    unsigned _pk_u3 = *reinterpret_cast<unsigned*>(&_pk3);
                                                    __nv_bfloat162 _pk4 = __floats2bfloat162_rn(_tmem_load_10[j2_1 + 8], _tmem_load_10[j2_1 + 9]);
                                                    unsigned _pk_u4 = *reinterpret_cast<unsigned*>(&_pk4);
                                                    __nv_bfloat162 _pk5 = __floats2bfloat162_rn(_tmem_load_10[j2_1 + 10], _tmem_load_10[j2_1 + 11]);
                                                    unsigned _pk_u5 = *reinterpret_cast<unsigned*>(&_pk5);
                                                    __nv_bfloat162 _pk6 = __floats2bfloat162_rn(_tmem_load_10[j2_1 + 12], _tmem_load_10[j2_1 + 13]);
                                                    unsigned _pk_u6 = *reinterpret_cast<unsigned*>(&_pk6);
                                                    __nv_bfloat162 _pk7 = __floats2bfloat162_rn(_tmem_load_10[j2_1 + 14], _tmem_load_10[j2_1 + 15]);
                                                    unsigned _pk_u7 = *reinterpret_cast<unsigned*>(&_pk7);
                                                    asm volatile(
                                                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                                        :: "l"((void*)(&((__nv_bfloat16*)(O + (gmem_base_0 + 64 + 32 + j2_1)))[0])), "r"(_pk_u0), "r"(_pk_u1), "r"(_pk_u2), "r"(_pk_u3), "r"(_pk_u4), "r"(_pk_u5), "r"(_pk_u6), "r"(_pk_u7) : "memory");
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    } else {
                        *(reinterpret_cast<float*>(partial_lse + slot_row) + (0)) = lse_log2;
                        #pragma unroll
                        for (int h0_1 = 0; h0_1 < 2; h0_1++) {
                            int acc_e_2 = h0_1 / 2;
                            int sl_e_2 = h0_1 - acc_e_2 * 2;
                            int o_base_e_2 = taddr + 128 + (unsigned int)(acc_e_2 * 128) + (unsigned int)(sl_e_2 * 64) + (unsigned int)corr_row;
                            float _tmem_load_11[32];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(_tmem_load_11[0]), "=f"(_tmem_load_11[1]), "=f"(_tmem_load_11[2]), "=f"(_tmem_load_11[3]), "=f"(_tmem_load_11[4]), "=f"(_tmem_load_11[5]), "=f"(_tmem_load_11[6]), "=f"(_tmem_load_11[7]), "=f"(_tmem_load_11[8]), "=f"(_tmem_load_11[9]), "=f"(_tmem_load_11[10]), "=f"(_tmem_load_11[11]), "=f"(_tmem_load_11[12]), "=f"(_tmem_load_11[13]), "=f"(_tmem_load_11[14]), "=f"(_tmem_load_11[15]), "=f"(_tmem_load_11[16]), "=f"(_tmem_load_11[17]), "=f"(_tmem_load_11[18]), "=f"(_tmem_load_11[19]), "=f"(_tmem_load_11[20]), "=f"(_tmem_load_11[21]), "=f"(_tmem_load_11[22]), "=f"(_tmem_load_11[23]), "=f"(_tmem_load_11[24]), "=f"(_tmem_load_11[25]), "=f"(_tmem_load_11[26]), "=f"(_tmem_load_11[27]), "=f"(_tmem_load_11[28]), "=f"(_tmem_load_11[29]), "=f"(_tmem_load_11[30]), "=f"(_tmem_load_11[31])
                                : "r"(o_base_e_2));
                            float _tmem_load_12[32];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(_tmem_load_12[0]), "=f"(_tmem_load_12[1]), "=f"(_tmem_load_12[2]), "=f"(_tmem_load_12[3]), "=f"(_tmem_load_12[4]), "=f"(_tmem_load_12[5]), "=f"(_tmem_load_12[6]), "=f"(_tmem_load_12[7]), "=f"(_tmem_load_12[8]), "=f"(_tmem_load_12[9]), "=f"(_tmem_load_12[10]), "=f"(_tmem_load_12[11]), "=f"(_tmem_load_12[12]), "=f"(_tmem_load_12[13]), "=f"(_tmem_load_12[14]), "=f"(_tmem_load_12[15]), "=f"(_tmem_load_12[16]), "=f"(_tmem_load_12[17]), "=f"(_tmem_load_12[18]), "=f"(_tmem_load_12[19]), "=f"(_tmem_load_12[20]), "=f"(_tmem_load_12[21]), "=f"(_tmem_load_12[22]), "=f"(_tmem_load_12[23]), "=f"(_tmem_load_12[24]), "=f"(_tmem_load_12[25]), "=f"(_tmem_load_12[26]), "=f"(_tmem_load_12[27]), "=f"(_tmem_load_12[28]), "=f"(_tmem_load_12[29]), "=f"(_tmem_load_12[30]), "=f"(_tmem_load_12[31])
                                : "r"(o_base_e_2 + 32));
                            float sa_e_2[32];
                            float sb_e_2[32];
                            #pragma unroll
                            for (int ie_2 = 0; ie_2 < 32; ie_2++) {
                                sa_e_2[ie_2] = _tmem_load_11[ie_2] * inv_sum;
                                sb_e_2[ie_2] = _tmem_load_12[ie_2] * inv_sum;
                            }
                            uint32_t sa_e_bf16_2[16];
                            #pragma unroll
                            for (int _lp = 0; _lp < 16; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sa_e_2[_lp*2 + 0], sa_e_2[_lp*2+1 + 0]));
                                sa_e_bf16_2[_lp] = *(uint32_t*)&_bf2;
                            }
                            uint32_t sb_e_bf16_2[16];
                            #pragma unroll
                            for (int _lp = 0; _lp < 16; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sb_e_2[_lp*2 + 0], sb_e_2[_lp*2+1 + 0]));
                                sb_e_bf16_2[_lp] = *(uint32_t*)&_bf2;
                            }
                            int hb_e_2 = h0_1 % 2;
                            int stage_e_2 = hb_e_2 * 2 + n_half_1;
                            int sbase_e_2 = smem_ostage_addr + (unsigned int)(stage_e_2 * 8192);
                            #pragma unroll
                            for (int ce_2 = 0; ce_2 < 4; ce_2++) {
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sbase_e_2 + (my_row_1 * 128 + ce_2 * 16 ^ (my_row_1 * 128 + ce_2 * 16 >> 7 & 7) << 4))), "r"(sa_e_bf16_2[ce_2 * 4]), "r"(sa_e_bf16_2[ce_2 * 4 + 1]), "r"(sa_e_bf16_2[ce_2 * 4 + 2]), "r"(sa_e_bf16_2[ce_2 * 4 + 3]) : "memory");
                            }
                            #pragma unroll
                            for (int ce2_2 = 0; ce2_2 < 4; ce2_2++) {
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sbase_e_2 + (my_row_1 * 128 + (64 + ce2_2 * 16) ^ (my_row_1 * 128 + (64 + ce2_2 * 16) >> 7 & 7) << 4))), "r"(sb_e_bf16_2[ce2_2 * 4]), "r"(sb_e_bf16_2[ce2_2 * 4 + 1]), "r"(sb_e_bf16_2[ce2_2 * 4 + 2]), "r"(sb_e_bf16_2[ce2_2 * 4 + 3]) : "memory");
                            }
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile("barrier.sync 3, 128;" ::: "memory");
                            if (warp == 4) {
                                if (elect_sync()) {
                                    #pragma unroll
                                    for (int nh_b_2 = 0; nh_b_2 < 2; nh_b_2++) {
                                        int col_b_2 = (acc_e_2 * 4 + nh_b_2 * 2 + sl_e_2) * 64;
                                        tma_store_2d((&tmap_po), col_b_2, (slot_base_1 + split_idx_1) * 128 + cta_rank * 64, smem_ostage_addr + (unsigned int)((hb_e_2 * 2 + nh_b_2) * 8192));
                                    }
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                        }
                        #pragma unroll
                        for (int h1_1 = 2; h1_1 < 4; h1_1++) {
                            int acc_e_3 = h1_1 / 2;
                            int sl_e_3 = h1_1 - acc_e_3 * 2;
                            int o_base_e_3 = taddr + 128 + (unsigned int)(acc_e_3 * 128) + (unsigned int)(sl_e_3 * 64) + (unsigned int)corr_row;
                            float _tmem_load_13[32];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(_tmem_load_13[0]), "=f"(_tmem_load_13[1]), "=f"(_tmem_load_13[2]), "=f"(_tmem_load_13[3]), "=f"(_tmem_load_13[4]), "=f"(_tmem_load_13[5]), "=f"(_tmem_load_13[6]), "=f"(_tmem_load_13[7]), "=f"(_tmem_load_13[8]), "=f"(_tmem_load_13[9]), "=f"(_tmem_load_13[10]), "=f"(_tmem_load_13[11]), "=f"(_tmem_load_13[12]), "=f"(_tmem_load_13[13]), "=f"(_tmem_load_13[14]), "=f"(_tmem_load_13[15]), "=f"(_tmem_load_13[16]), "=f"(_tmem_load_13[17]), "=f"(_tmem_load_13[18]), "=f"(_tmem_load_13[19]), "=f"(_tmem_load_13[20]), "=f"(_tmem_load_13[21]), "=f"(_tmem_load_13[22]), "=f"(_tmem_load_13[23]), "=f"(_tmem_load_13[24]), "=f"(_tmem_load_13[25]), "=f"(_tmem_load_13[26]), "=f"(_tmem_load_13[27]), "=f"(_tmem_load_13[28]), "=f"(_tmem_load_13[29]), "=f"(_tmem_load_13[30]), "=f"(_tmem_load_13[31])
                                : "r"(o_base_e_3));
                            float _tmem_load_14[32];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(_tmem_load_14[0]), "=f"(_tmem_load_14[1]), "=f"(_tmem_load_14[2]), "=f"(_tmem_load_14[3]), "=f"(_tmem_load_14[4]), "=f"(_tmem_load_14[5]), "=f"(_tmem_load_14[6]), "=f"(_tmem_load_14[7]), "=f"(_tmem_load_14[8]), "=f"(_tmem_load_14[9]), "=f"(_tmem_load_14[10]), "=f"(_tmem_load_14[11]), "=f"(_tmem_load_14[12]), "=f"(_tmem_load_14[13]), "=f"(_tmem_load_14[14]), "=f"(_tmem_load_14[15]), "=f"(_tmem_load_14[16]), "=f"(_tmem_load_14[17]), "=f"(_tmem_load_14[18]), "=f"(_tmem_load_14[19]), "=f"(_tmem_load_14[20]), "=f"(_tmem_load_14[21]), "=f"(_tmem_load_14[22]), "=f"(_tmem_load_14[23]), "=f"(_tmem_load_14[24]), "=f"(_tmem_load_14[25]), "=f"(_tmem_load_14[26]), "=f"(_tmem_load_14[27]), "=f"(_tmem_load_14[28]), "=f"(_tmem_load_14[29]), "=f"(_tmem_load_14[30]), "=f"(_tmem_load_14[31])
                                : "r"(o_base_e_3 + 32));
                            {
                                if (warp == 4) {
                                    if (elect_sync()) {
                                        asm volatile("cp.async.bulk.wait_group.read 1;");
                                    }
                                }
                                asm volatile("barrier.sync 3, 128;" ::: "memory");
                            }
                            float sa_e_3[32];
                            float sb_e_3[32];
                            #pragma unroll
                            for (int ie_3 = 0; ie_3 < 32; ie_3++) {
                                sa_e_3[ie_3] = _tmem_load_13[ie_3] * inv_sum;
                                sb_e_3[ie_3] = _tmem_load_14[ie_3] * inv_sum;
                            }
                            uint32_t sa_e_bf16_3[16];
                            #pragma unroll
                            for (int _lp = 0; _lp < 16; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sa_e_3[_lp*2 + 0], sa_e_3[_lp*2+1 + 0]));
                                sa_e_bf16_3[_lp] = *(uint32_t*)&_bf2;
                            }
                            uint32_t sb_e_bf16_3[16];
                            #pragma unroll
                            for (int _lp = 0; _lp < 16; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sb_e_3[_lp*2 + 0], sb_e_3[_lp*2+1 + 0]));
                                sb_e_bf16_3[_lp] = *(uint32_t*)&_bf2;
                            }
                            int hb_e_3 = h1_1 % 2;
                            int stage_e_3 = hb_e_3 * 2 + n_half_1;
                            int sbase_e_3 = smem_ostage_addr + (unsigned int)(stage_e_3 * 8192);
                            #pragma unroll
                            for (int ce_3 = 0; ce_3 < 4; ce_3++) {
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sbase_e_3 + (my_row_1 * 128 + ce_3 * 16 ^ (my_row_1 * 128 + ce_3 * 16 >> 7 & 7) << 4))), "r"(sa_e_bf16_3[ce_3 * 4]), "r"(sa_e_bf16_3[ce_3 * 4 + 1]), "r"(sa_e_bf16_3[ce_3 * 4 + 2]), "r"(sa_e_bf16_3[ce_3 * 4 + 3]) : "memory");
                            }
                            #pragma unroll
                            for (int ce2_3 = 0; ce2_3 < 4; ce2_3++) {
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((sbase_e_3 + (my_row_1 * 128 + (64 + ce2_3 * 16) ^ (my_row_1 * 128 + (64 + ce2_3 * 16) >> 7 & 7) << 4))), "r"(sb_e_bf16_3[ce2_3 * 4]), "r"(sb_e_bf16_3[ce2_3 * 4 + 1]), "r"(sb_e_bf16_3[ce2_3 * 4 + 2]), "r"(sb_e_bf16_3[ce2_3 * 4 + 3]) : "memory");
                            }
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile("barrier.sync 3, 128;" ::: "memory");
                            if (warp == 4) {
                                if (elect_sync()) {
                                    #pragma unroll
                                    for (int nh_b_3 = 0; nh_b_3 < 2; nh_b_3++) {
                                        int col_b_3 = (acc_e_3 * 4 + nh_b_3 * 2 + sl_e_3) * 64;
                                        tma_store_2d((&tmap_po), col_b_3, (slot_base_1 + split_idx_1) * 128 + cta_rank * 64, smem_ostage_addr + (unsigned int)((hb_e_3 * 2 + nh_b_3) * 8192));
                                    }
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                        }
                        if (warp == 4) {
                            if (elect_sync()) {
                                asm volatile("cp.async.bulk.wait_group.read 0;");
                            }
                        }
                        asm volatile("barrier.sync 3, 128;" ::: "memory");
                    }
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(o_empty_addr), "r"(0) : "memory");
                    if (warp == 4) {
                        if (lane == 0) {
                            mbarrier_arrive(p_epi_free_addr);
                        }
                    }
                    if (item_units_1 > 1) {
                        if (warp == 4) {
                            if (elect_sync()) {
                                asm volatile("cp.async.bulk.wait_group 0;");
                            }
                        }
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                            "}"
                            :: "r"(merge_ready_addr), "r"(0) : "memory");
                        if (cta_rank == 0) {
                            if (warp == 4) {
                                mbarrier_wait(pub_free_addr, _phase_pub_free_0);
                                _phase_pub_free_0 ^= 1;
                                if (lane == 0) {
                                    pub_words[0] = 1;
                                    pub_words[1] = (unsigned int)slot_base_1;
                                    pub_words[2] = (unsigned int)split_idx_1;
                                    mbarrier_arrive(pub_req_addr);
                                }
                            }
                        }
                    }
                } else if (row_in_tile_1 < valid_rows_3_1) {
                    *(reinterpret_cast<float*>(LSE + compact_row) + (0)) = neg_inf;
                    int o_zero_base = compact_row * 512;
                    #pragma unroll
                    for (int zc = 0; zc < 512; zc += 8) {
                        {
                            __nv_bfloat162 _pk[4];
                            _pk[0] = __floats2bfloat162_rn(zero_vec[0 + 0], zero_vec[0 + 1]);
                            _pk[1] = __floats2bfloat162_rn(zero_vec[0 + 2], zero_vec[0 + 3]);
                            _pk[2] = __floats2bfloat162_rn(zero_vec[0 + 4], zero_vec[0 + 5]);
                            _pk[3] = __floats2bfloat162_rn(zero_vec[0 + 6], zero_vec[0 + 7]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O + (o_zero_base + zc)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        }
                    }
                }
                correction_tile_cursor = correction_tile_cursor + num_kv_tiles_1;
            }
            if (cta_rank == 0) {
                if (warp == 4) {
                    mbarrier_wait(pub_free_addr, _phase_pub_free_0);
                    _phase_pub_free_0 ^= 1;
                    if (lane == 0) {
                        pub_words[0] = 0;
                        mbarrier_arrive(pub_req_addr);
                    }
                }
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            if (warp == 4) {
            }
        }
    }
    // ---- Role: mma_warp ----
    if (warp == 8) {
        { // mma_warp_main
            const int wg2_dummy = 0;
            {
                asm volatile("griddepcontrol.wait;" ::: "memory");
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
            unsigned int mma_kv_stage = 0;
            int mma_tile_cursor = 0;
            unsigned int _phase_o_first_slice_seeded_0 = 0;
            mbarrier_wait(o_first_slice_seeded_addr, _phase_o_first_slice_seeded_0);
            _phase_o_first_slice_seeded_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            unsigned int work_stage_mma = 0;
            unsigned int first_q_mma = 1;
            uint32_t _fast_div_q_6 = (fd_num_items.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)cluster_id), fd_num_items.multiplier) >> fd_num_items.shift_right) : (uint32_t)((unsigned int)cluster_id);
            uint32_t _fast_div_r_6 = (uint32_t)((unsigned int)cluster_id) - _fast_div_q_6 * (uint32_t)(fd_num_items.divisor);
            int k_st_2 = (int)_fast_div_q_6;
            int it_st_2 = cluster_id - (unsigned int)(k_st_2 * num_items);
            uint32_t _fast_div_q_7 = (fd_tiles_max.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)it_st_2), fd_tiles_max.multiplier) >> fd_tiles_max.shift_right) : (uint32_t)((unsigned int)it_st_2);
            uint32_t _fast_div_r_7 = (uint32_t)((unsigned int)it_st_2) - _fast_div_q_7 * (uint32_t)(fd_tiles_max.divisor);
            int sb_2 = (int)_fast_div_q_7;
            int sm_2 = it_st_2 - sb_2 * tiles_max;
            int q_begin_2 = cum_seq_lens_q[sb_2];
            int q_len_3 = cum_seq_lens_q[sb_2 + 1] - q_begin_2;
            int rows_remaining_3 = q_len_3 * num_heads - sm_2 * 128;
            int valid_rows_2 = ((rows_remaining_3 < 128) ? rows_remaining_3 : 128);
            int k_local_2 = seq_lens[sb_2];
            int g_bound_2 = causal_global[sb_2];
            int n_t_2 = -1;
            if (valid_rows_2 > 0) {
                uint32_t _fast_div_q_8 = (fd_num_heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(sm_2 * 128 + valid_rows_2 - 1)), fd_num_heads.multiplier) >> fd_num_heads.shift_right) : (uint32_t)((unsigned int)(sm_2 * 128 + valid_rows_2 - 1));
                uint32_t _fast_div_r_8 = (uint32_t)((unsigned int)(sm_2 * 128 + valid_rows_2 - 1)) - _fast_div_q_8 * (uint32_t)(fd_num_heads.divisor);
                int last_q_t_2 = (int)_fast_div_q_8;
                int lim_num_t_2 = g_bound_2 - q_len_3 + last_q_t_2 + 1 - cp_rank;
                lim_num_t_2 = ((lim_num_t_2 > 0) ? lim_num_t_2 : 0);
                uint32_t _fast_div_q_9 = (fd_cp_world.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(lim_num_t_2 + cp_world - 1)), fd_cp_world.multiplier) >> fd_cp_world.shift_right) : (uint32_t)((unsigned int)(lim_num_t_2 + cp_world - 1));
                uint32_t _fast_div_r_9 = (uint32_t)((unsigned int)(lim_num_t_2 + cp_world - 1)) - _fast_div_q_9 * (uint32_t)(fd_cp_world.divisor);
                int lim_ceil_t_2 = (int)_fast_div_q_9;
                int key_lim_t_2 = ((lim_ceil_t_2 < k_local_2) ? lim_ceil_t_2 : k_local_2);
                key_lim_t_2 = ((key_lim_t_2 > 0) ? key_lim_t_2 : 0);
                n_t_2 = (key_lim_t_2 + 128 - 1) / 128;
            }
            int n_all_2 = n_t_2;
            int n_st_2 = n_all_2 - k_st_2 * static_tiles;
            n_st_2 = ((n_st_2 < static_tiles) ? n_st_2 : static_tiles);
            n_st_2 = ((n_st_2 > 0) ? n_st_2 : 0);
            int has_st_2 = 1;
            if (k_st_2 > 0) {
                has_st_2 = ((n_st_2 > 0) ? 1 : 0);
            }
            if (k_st_2 >= 8) {
                has_st_2 = 0;
            }
            int has_static_mma = has_st_2;
            int no_token_mma = ((0) ? static_only : 0);
            unsigned int _phase_work_full_2 = 0;
            unsigned int _phase_q_full_0 = 0;
            unsigned int _phase_kv_full = 0;
            unsigned int _phase_o_empty_0 = 1;
            #pragma unroll 1
            for (unsigned int _unit_iter_mma = 0; _unit_iter_mma < max_units + 1; _unit_iter_mma++) {
                unsigned int tok_valid_2 = 1;
                int batch_idx_2 = 0;
                int m_tile_3 = 0;
                int start_tile_3 = 0;
                int num_kv_tiles_2 = 0;
                int split_idx_2 = 0;
                int item_units_2 = 1;
                int slot_base_2 = 0;
                int is_static_2 = has_static_mma;
                has_static_mma = 0;
                if (is_static_2 != 0) {
                    batch_idx_2 = sb_2;
                    m_tile_3 = sm_2;
                    num_kv_tiles_2 = n_st_2;
                    start_tile_3 = k_st_2 * static_tiles;
                    split_idx_2 = k_st_2;
                } else if (no_token_mma != 0) {
                    tok_valid_2 = 0;
                } else {
                    mbarrier_wait_cluster(work_full_addr + (work_stage_mma) * 8, _phase_work_full_2);
                    unsigned int tok_words_2[8];
                    #pragma unroll
                    for (int w_2 = 0; w_2 < 8; w_2++) {
                        tok_words_2[w_2] = 0;
                    }
                    if (lane == 0) {
                        uint32_t _mapa_1;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mapa_1) : "r"(work_token_words_addr + work_stage_mma * 32), "r"(0));
                        asm volatile("ld.shared::cluster.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tok_words_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&tok_words_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tok_words_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tok_words_2[(0) + 3]))
                            : "r"(_mapa_1));
                        asm volatile("ld.shared::cluster.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tok_words_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&tok_words_2[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tok_words_2[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tok_words_2[(4) + 3]))
                            : "r"(_mapa_1 + 16));
                    }
                    unsigned int _shfl_8 = __shfl_sync(0xFFFFFFFF, tok_words_2[0], 0);
                    unsigned int valid_2 = _shfl_8;
                    unsigned int _shfl_9 = __shfl_sync(0xFFFFFFFF, tok_words_2[1], 0);
                    int batch_idx_0_2 = (int)_shfl_9;
                    unsigned int _shfl_10 = __shfl_sync(0xFFFFFFFF, tok_words_2[2], 0);
                    int m_tile_1_2 = (int)_shfl_10;
                    unsigned int _shfl_11 = __shfl_sync(0xFFFFFFFF, tok_words_2[3], 0);
                    int start_tile_2_2 = (int)_shfl_11;
                    unsigned int _shfl_12 = __shfl_sync(0xFFFFFFFF, tok_words_2[4], 0);
                    int num_kv_tiles_3_2 = (int)_shfl_12;
                    unsigned int _shfl_13 = __shfl_sync(0xFFFFFFFF, tok_words_2[5], 0);
                    int unit_idx_2 = (int)_shfl_13;
                    unsigned int _shfl_14 = __shfl_sync(0xFFFFFFFF, tok_words_2[6], 0);
                    int units_2 = (int)_shfl_14;
                    unsigned int _shfl_15 = __shfl_sync(0xFFFFFFFF, tok_words_2[7], 0);
                    int slot_base_4_2 = (int)_shfl_15;
                    if (lane == 0) {
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                            "}"
                            :: "r"(work_empty_addr + work_stage_mma * 8), "r"(0) : "memory");
                    }
                    work_stage_mma += 1;
                    if (work_stage_mma == 2) { work_stage_mma = 0; _phase_work_full_2 ^= 1; }
                    tok_valid_2 = valid_2;
                    batch_idx_2 = batch_idx_0_2;
                    m_tile_3 = m_tile_1_2;
                    start_tile_3 = start_tile_2_2;
                    num_kv_tiles_2 = num_kv_tiles_3_2;
                    split_idx_2 = unit_idx_2;
                    item_units_2 = units_2;
                    slot_base_2 = slot_base_4_2;
                }
                if (tok_valid_2 == 0) {
                    break;
                }
                int q_begin_0_2 = cum_seq_lens_q[batch_idx_2];
                int q_len_1_2 = cum_seq_lens_q[batch_idx_2 + 1] - q_begin_0_2;
                int rows_remaining_2_2 = q_len_1_2 * num_heads - m_tile_3 * 128;
                int valid_rows_3_2 = ((rows_remaining_2_2 < 128) ? rows_remaining_2_2 : 128);
                int k_local_4_2 = seq_lens[batch_idx_2];
                int g_bound_5_2 = causal_global[batch_idx_2];
                if (num_kv_tiles_2 > 0) {
                    if (cta_rank == 0) {
                        mbarrier_wait(q_full_addr, _phase_q_full_0);
                        _phase_q_full_0 ^= 1;
                        int first_pv = 1;
                        int pt_s = mma_tile_cursor;
                        int ph_s = pt_s & 1;
                        int se_ph = pt_s >> 1 & 1 ^ 1;
                        mbarrier_wait(s_empty_addr + (ph_s) * 8, se_ph);
                        int sc_col = ((ph_s != 0) ? 64 : 0);
                        #pragma unroll
                        for (int n_m = 0; n_m < 8; n_m++) {
                            mbarrier_wait(kv_full_addr + (mma_kv_stage) * 8, _phase_kv_full);
                            int _mma_a_lo_0 = (((smem_q_addr) >> 4) & 0x3FFF) + (n_m) * 512;
                            int _mma_b_lo_0 = (((smem_kv_addr) >> 4) & 0x3FFF) + (mma_kv_stage) * 512;
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
                    "mov.b32 id, 136316048;\n\t"
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
                    "}\n"
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_tmem_scratch + (sc_col))), "r"(((n_m == 0) ? 0 : 1)));
                            elect_commit_cg2_multicast(kv_empty_addr + (mma_kv_stage) * 8, (uint16_t)(3));
                            mma_kv_stage += 1;
                            if (mma_kv_stage == 15) { mma_kv_stage = 0; _phase_kv_full ^= 1; }
                        }
                        mbarrier_wait(kv_full_addr + (mma_kv_stage) * 8, _phase_kv_full);
                        int _mma_a_lo_1 = (((smem_q_addr) >> 4) & 0x3FFF) + (8) * 512;
                        int _mma_b_lo_1 = (((smem_kv_addr) >> 4) & 0x3FFF) + (mma_kv_stage) * 512;
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
                    "mov.b32 id, 136316048;\n\t"
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
                    "}\n"
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"((tmem_tmem_scratch + (sc_col))), "r"(1));
                        elect_commit_cg2_multicast(kv_empty_addr + (mma_kv_stage) * 8, (uint16_t)(3));
                        elect_commit_cg2_multicast(s_full_addr + (ph_s) * 8, (uint16_t)(3));
                        mma_kv_stage += 1;
                        if (mma_kv_stage == 15) { mma_kv_stage = 0; _phase_kv_full ^= 1; }
                        #pragma unroll 1
                        for (int tile_2 = 0; tile_2 < num_kv_tiles_2; tile_2++) {
                            if (num_kv_tiles_2 > tile_2 + 1) {
                                int pt_s_0 = mma_tile_cursor + (tile_2 + 1);
                                int ph_s_1 = pt_s_0 & 1;
                                int se_ph_2 = pt_s_0 >> 1 & 1 ^ 1;
                                mbarrier_wait(s_empty_addr + (ph_s_1) * 8, se_ph_2);
                                int sc_col_3 = ((ph_s_1 != 0) ? 64 : 0);
                                #pragma unroll
                                for (int n_m_1 = 0; n_m_1 < 8; n_m_1++) {
                                    mbarrier_wait(kv_full_addr + (mma_kv_stage) * 8, _phase_kv_full);
                                    int _mma_a_lo_2 = (((smem_q_addr) >> 4) & 0x3FFF) + (n_m_1) * 512;
                                    int _mma_b_lo_2 = (((smem_kv_addr) >> 4) & 0x3FFF) + (mma_kv_stage) * 512;
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
                    "mov.b32 id, 136316048;\n\t"
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
                    "}\n"
                    :: "r"(_mma_a_lo_2), "r"(_mma_b_lo_2), "r"((tmem_tmem_scratch + (sc_col_3))), "r"(((n_m_1 == 0) ? 0 : 1)));
                                    elect_commit_cg2_multicast(kv_empty_addr + (mma_kv_stage) * 8, (uint16_t)(3));
                                    mma_kv_stage += 1;
                                    if (mma_kv_stage == 15) { mma_kv_stage = 0; _phase_kv_full ^= 1; }
                                }
                                mbarrier_wait(kv_full_addr + (mma_kv_stage) * 8, _phase_kv_full);
                                int _mma_a_lo_3 = (((smem_q_addr) >> 4) & 0x3FFF) + (8) * 512;
                                int _mma_b_lo_3 = (((smem_kv_addr) >> 4) & 0x3FFF) + (mma_kv_stage) * 512;
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
                    "mov.b32 id, 136316048;\n\t"
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
                    "}\n"
                    :: "r"(_mma_a_lo_3), "r"(_mma_b_lo_3), "r"((tmem_tmem_scratch + (sc_col_3))), "r"(1));
                                elect_commit_cg2_multicast(kv_empty_addr + (mma_kv_stage) * 8, (uint16_t)(3));
                                elect_commit_cg2_multicast(s_full_addr + (ph_s_1) * 8, (uint16_t)(3));
                                mma_kv_stage += 1;
                                if (mma_kv_stage == 15) { mma_kv_stage = 0; _phase_kv_full ^= 1; }
                            }
                            int pt_p = mma_tile_cursor + tile_2;
                            int ph_p = pt_p & 1;
                            int pv_ph = pt_p >> 1 & 1;
                            mbarrier_wait(o_empty_addr, _phase_o_empty_0);
                            _phase_o_empty_0 ^= 1;
                            mbarrier_wait(p_full_addr + (ph_p) * 8, pv_ph);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            #pragma unroll
                            for (int ps_m = 0; ps_m < 4; ps_m++) {
                                int p_stage_m = ph_p * 4 + ps_m;
                                #pragma unroll
                                for (int acc_m = 0; acc_m < 2; acc_m++) {
                                    mbarrier_wait(kv_full_addr + (mma_kv_stage) * 8, _phase_kv_full);
                                    int out_col_m = 128 + acc_m * 128;
                                    int _mma_a_lo_4 = (((smem_p_addr) >> 4) & 0x3FFF) + (p_stage_m) * 256;
                                    int _mma_b_lo_4 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x1000000) + (mma_kv_stage) * 512;
                                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x80004020;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 138478736;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f16 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_4), "r"(_mma_b_lo_4), "r"((tmem_tmem_scratch + (out_col_m))), "r"(((((ps_m == 0) ? first_pv : 0)) ? 0 : 1)));
                                    elect_commit_cg2_multicast(kv_empty_addr + (mma_kv_stage) * 8, (uint16_t)(3));
                                    mma_kv_stage += 1;
                                    if (mma_kv_stage == 15) { mma_kv_stage = 0; _phase_kv_full ^= 1; }
                                }
                            }
                            elect_commit_cg2_multicast(p_empty_addr + (ph_p) * 8, (uint16_t)(3));
                            elect_commit_cg2_multicast(o_full_addr, (uint16_t)(3));
                            first_pv = 0;
                        }
                        elect_commit_cg2_multicast(q_empty_addr, (uint16_t)(3));
                    }
                }
                mma_tile_cursor = mma_tile_cursor + num_kv_tiles_2;
            }
            if (cta_rank == 0) {
                #pragma unroll
                for (int tail_offset = 0; tail_offset < 2; tail_offset++) {
                    int tail_tile = mma_tile_cursor + tail_offset;
                    int tail_stage = tail_tile & 1;
                    int tail_phase = tail_tile >> 1 & 1 ^ 1;
                    mbarrier_wait(s_empty_addr + (tail_stage) * 8, tail_phase);
                }
                mbarrier_wait(o_empty_addr, _phase_o_empty_0);
                _phase_o_empty_0 ^= 1;
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            int tmem_dealloc_peer_rank = cta_rank ^ 1;
            asm volatile(
                "{\n\t"
                ".reg .b32 remAddr32;\n\t"
                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                "}"
                :: "r"(tmem_dealloc_peer_addr), "r"(tmem_dealloc_peer_rank) : "memory");
            mbarrier_wait(tmem_dealloc_peer_addr, 0);
            int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
            asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(512));
        }
    }
    // ---- Role: load_warp ----
    if (warp == 9) {
        { // load_warp_main
            const int wg2_dummy_1 = 0;
            {
                asm volatile("griddepcontrol.wait;" ::: "memory");
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
            unsigned int load_kv_stage = 0;
            int pg_pf[1];
            int spec_left = 0;
            int valid_rows_sp = 0;
            uint32_t _fast_div_q_0 = (fd_num_items.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)cluster_id), fd_num_items.multiplier) >> fd_num_items.shift_right) : (uint32_t)((unsigned int)cluster_id);
            uint32_t _fast_div_r_0 = (uint32_t)((unsigned int)cluster_id) - _fast_div_q_0 * (uint32_t)(fd_num_items.divisor);
            int k_pf = (int)_fast_div_q_0;
            int it_pf = cluster_id - (unsigned int)(k_pf * num_items);
            uint32_t _fast_div_q_1 = (fd_tiles_max.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)it_pf), fd_tiles_max.multiplier) >> fd_tiles_max.shift_right) : (uint32_t)((unsigned int)it_pf);
            uint32_t _fast_div_r_1 = (uint32_t)((unsigned int)it_pf) - _fast_div_q_1 * (uint32_t)(fd_tiles_max.divisor);
            int sb_pf = (int)_fast_div_q_1;
            int tok_pf = k_pf * static_tiles * 128 + cta_rank * 64;
            #pragma unroll
            for (int kbp = 0; kbp < 1; kbp++) {
                int pidx_pf = (tok_pf + kbp * 64) / 64;
                pidx_pf = ((pidx_pf < max_pages) ? pidx_pf : max_pages - 1);
                pg_pf[kbp] = page_table[sb_pf * max_pages + pidx_pf];
            }
            unsigned int work_stage_load = 0;
            unsigned int first_tok_load = 1;
            unsigned int first_ud_load = 1;
            uint32_t _fast_div_q_2 = (fd_num_items.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)cluster_id), fd_num_items.multiplier) >> fd_num_items.shift_right) : (uint32_t)((unsigned int)cluster_id);
            uint32_t _fast_div_r_2 = (uint32_t)((unsigned int)cluster_id) - _fast_div_q_2 * (uint32_t)(fd_num_items.divisor);
            int k_st_3 = (int)_fast_div_q_2;
            int it_st_3 = cluster_id - (unsigned int)(k_st_3 * num_items);
            uint32_t _fast_div_q_3 = (fd_tiles_max.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)it_st_3), fd_tiles_max.multiplier) >> fd_tiles_max.shift_right) : (uint32_t)((unsigned int)it_st_3);
            uint32_t _fast_div_r_3 = (uint32_t)((unsigned int)it_st_3) - _fast_div_q_3 * (uint32_t)(fd_tiles_max.divisor);
            int sb_3 = (int)_fast_div_q_3;
            int sm_3 = it_st_3 - sb_3 * tiles_max;
            int q_begin_3 = cum_seq_lens_q[sb_3];
            int q_len_4 = cum_seq_lens_q[sb_3 + 1] - q_begin_3;
            int rows_remaining_4 = q_len_4 * num_heads - sm_3 * 128;
            int valid_rows_4 = ((rows_remaining_4 < 128) ? rows_remaining_4 : 128);
            int k_local_3 = seq_lens[sb_3];
            int g_bound_3 = causal_global[sb_3];
            int n_t_3 = -1;
            if (valid_rows_4 > 0) {
                uint32_t _fast_div_q_4 = (fd_num_heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(sm_3 * 128 + valid_rows_4 - 1)), fd_num_heads.multiplier) >> fd_num_heads.shift_right) : (uint32_t)((unsigned int)(sm_3 * 128 + valid_rows_4 - 1));
                uint32_t _fast_div_r_4 = (uint32_t)((unsigned int)(sm_3 * 128 + valid_rows_4 - 1)) - _fast_div_q_4 * (uint32_t)(fd_num_heads.divisor);
                int last_q_t_3 = (int)_fast_div_q_4;
                int lim_num_t_3 = g_bound_3 - q_len_4 + last_q_t_3 + 1 - cp_rank;
                lim_num_t_3 = ((lim_num_t_3 > 0) ? lim_num_t_3 : 0);
                uint32_t _fast_div_q_5 = (fd_cp_world.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(lim_num_t_3 + cp_world - 1)), fd_cp_world.multiplier) >> fd_cp_world.shift_right) : (uint32_t)((unsigned int)(lim_num_t_3 + cp_world - 1));
                uint32_t _fast_div_r_5 = (uint32_t)((unsigned int)(lim_num_t_3 + cp_world - 1)) - _fast_div_q_5 * (uint32_t)(fd_cp_world.divisor);
                int lim_ceil_t_3 = (int)_fast_div_q_5;
                int key_lim_t_3 = ((lim_ceil_t_3 < k_local_3) ? lim_ceil_t_3 : k_local_3);
                key_lim_t_3 = ((key_lim_t_3 > 0) ? key_lim_t_3 : 0);
                n_t_3 = (key_lim_t_3 + 128 - 1) / 128;
            }
            int n_all_3 = n_t_3;
            int n_st_3 = n_all_3 - k_st_3 * static_tiles;
            n_st_3 = ((n_st_3 < static_tiles) ? n_st_3 : static_tiles);
            n_st_3 = ((n_st_3 > 0) ? n_st_3 : 0);
            int has_st_3 = 1;
            if (k_st_3 > 0) {
                has_st_3 = ((n_st_3 > 0) ? 1 : 0);
            }
            if (k_st_3 >= 8) {
                has_st_3 = 0;
            }
            int has_static_load = has_st_3;
            unsigned int _phase_work_full_3 = 0;
            unsigned int _phase_q_empty_0 = 1;
            unsigned int _phase_kv_empty = 1;
            #pragma unroll 1
            for (unsigned int _unit_iter_load = 0; _unit_iter_load < max_units + 1; _unit_iter_load++) {
                unsigned int tok_valid_3 = 1;
                int batch_idx_3 = 0;
                int m_tile_4 = 0;
                int start_tile_4 = 0;
                int num_kv_tiles_4 = 0;
                int split_idx_3 = 0;
                int item_units_3 = 1;
                int slot_base_3 = 0;
                int is_static_3 = has_static_load;
                has_static_load = 0;
                if (is_static_3 != 0) {
                    batch_idx_3 = sb_3;
                    m_tile_4 = sm_3;
                    num_kv_tiles_4 = n_st_3;
                    start_tile_4 = k_st_3 * static_tiles;
                    split_idx_3 = k_st_3;
                } else {
                    mbarrier_wait_cluster(work_full_addr + (work_stage_load) * 8, _phase_work_full_3);
                    unsigned int tok_words_3[8];
                    #pragma unroll
                    for (int w_3 = 0; w_3 < 8; w_3++) {
                        tok_words_3[w_3] = 0;
                    }
                    if (lane == 0) {
                        uint32_t _mapa_0;
                        asm volatile(
                            "mapa.shared::cluster.u32 %0, %1, %2;"
                            : "=r"(_mapa_0) : "r"(work_token_words_addr + work_stage_load * 32), "r"(0));
                        asm volatile("ld.shared::cluster.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tok_words_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&tok_words_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tok_words_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tok_words_3[(0) + 3]))
                            : "r"(_mapa_0));
                        asm volatile("ld.shared::cluster.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&tok_words_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&tok_words_3[(4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&tok_words_3[(4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&tok_words_3[(4) + 3]))
                            : "r"(_mapa_0 + 16));
                    }
                    unsigned int _shfl_0 = __shfl_sync(0xFFFFFFFF, tok_words_3[0], 0);
                    unsigned int valid_3 = _shfl_0;
                    unsigned int _shfl_1 = __shfl_sync(0xFFFFFFFF, tok_words_3[1], 0);
                    int batch_idx_0_3 = (int)_shfl_1;
                    unsigned int _shfl_2 = __shfl_sync(0xFFFFFFFF, tok_words_3[2], 0);
                    int m_tile_1_3 = (int)_shfl_2;
                    unsigned int _shfl_3 = __shfl_sync(0xFFFFFFFF, tok_words_3[3], 0);
                    int start_tile_2_3 = (int)_shfl_3;
                    unsigned int _shfl_4 = __shfl_sync(0xFFFFFFFF, tok_words_3[4], 0);
                    int num_kv_tiles_3_3 = (int)_shfl_4;
                    unsigned int _shfl_5 = __shfl_sync(0xFFFFFFFF, tok_words_3[5], 0);
                    int unit_idx_3 = (int)_shfl_5;
                    unsigned int _shfl_6 = __shfl_sync(0xFFFFFFFF, tok_words_3[6], 0);
                    int units_3 = (int)_shfl_6;
                    unsigned int _shfl_7 = __shfl_sync(0xFFFFFFFF, tok_words_3[7], 0);
                    int slot_base_4_3 = (int)_shfl_7;
                    if (lane == 0) {
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                            "}"
                            :: "r"(work_empty_addr + work_stage_load * 8), "r"(0) : "memory");
                    }
                    work_stage_load += 1;
                    if (work_stage_load == 2) { work_stage_load = 0; _phase_work_full_3 ^= 1; }
                    tok_valid_3 = valid_3;
                    batch_idx_3 = batch_idx_0_3;
                    m_tile_4 = m_tile_1_3;
                    start_tile_4 = start_tile_2_3;
                    num_kv_tiles_4 = num_kv_tiles_3_3;
                    split_idx_3 = unit_idx_3;
                    item_units_3 = units_3;
                    slot_base_3 = slot_base_4_3;
                }
                if (tok_valid_3 == 0) {
                    break;
                }
                int q_begin_0_3 = cum_seq_lens_q[batch_idx_3];
                int q_len_1_3 = cum_seq_lens_q[batch_idx_3 + 1] - q_begin_0_3;
                int rows_remaining_2_3 = q_len_1_3 * num_heads - m_tile_4 * 128;
                int valid_rows_3_3 = ((rows_remaining_2_3 < 128) ? rows_remaining_2_3 : 128);
                int k_local_4_3 = seq_lens[batch_idx_3];
                int g_bound_5_3 = causal_global[batch_idx_3];
                valid_rows_sp = valid_rows_3_3;
                if (num_kv_tiles_4 > 0) {
                    int pt_base = batch_idx_3 * max_pages;
                    int q_row_global = q_begin_0_3 * num_heads + m_tile_4 * 128 + cta_rank * 64;
                    if (spec_left == 0) {
                        mbarrier_wait(q_empty_addr, _phase_q_empty_0);
                        _phase_q_empty_0 ^= 1;
                        if (cta_rank == 0) {
                            if (elect_sync()) {
                                asm volatile(
                                    "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                    :: "r"((q_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(147456)) : "memory");
                            }
                        }
                        if (elect_sync()) {
                            #pragma unroll
                            for (int s = 0; s < 9; s++) {
                                int q_dst = smem_q_addr + (unsigned int)(s * 8192);
                                tma_3d_gmem2smem_cta2(q_dst, (&tmap_q), 0, q_row_global, s, ((q_full_addr) & 0xFEFFFFFF));
                            }
                        }
                        int pg_k0[1];
                        int off_k0[1];
                        int tok_kl = start_tile_4 * 128 + cta_rank * 64;
                        #pragma unroll
                        for (int kbl = 0; kbl < 1; kbl++) {
                            int tok_kbl = tok_kl + kbl * 64;
                            off_k0[kbl] = tok_kbl % 64;
                            if (is_static_3 != 0) {
                                pg_k0[kbl] = pg_pf[kbl];
                            } else {
                                pg_k0[kbl] = page_table[pt_base + tok_kbl / 64];
                            }
                        }
                        #pragma unroll
                        for (int n_l = 0; n_l < 9; n_l++) {
                            mbarrier_wait(kv_empty_addr + (load_kv_stage) * 8, _phase_kv_empty);
                            if (cta_rank == 0) {
                                if (elect_sync()) {
                                    asm volatile(
                                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                        :: "r"((kv_full_addr + (load_kv_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                                }
                            }
                            if (elect_sync()) {
                                int dst_kl = smem_kv_addr + load_kv_stage * 8192;
                                #pragma unroll
                                for (int kb_l = 0; kb_l < 1; kb_l++) {
                                    int pg_kbl = pg_k0[kb_l];
                                    int off_kbl = off_k0[kb_l];
                                    tma_4d_gmem2smem_cta2(dst_kl + kb_l * 8192, (&tmap_k), 0, off_kbl, n_l, pg_kbl, ((kv_full_addr + (load_kv_stage) * 8) & 0xFEFFFFFF));
                                }
                            }
                            load_kv_stage += 1;
                            if (load_kv_stage == 15) { load_kv_stage = 0; _phase_kv_empty ^= 1; }
                        }
                    }
                    spec_left = 0;
                    int pg_kn[1];
                    int off_kn[1];
                    int pg_vn[4];
                    int off_vn[4];
                    if (num_kv_tiles_4 > 1) {
                        int tok_kl_1 = (start_tile_4 + 1) * 128 + cta_rank * 64;
                        #pragma unroll
                        for (int kbl_1 = 0; kbl_1 < 1; kbl_1++) {
                            int tok_kbl_1 = tok_kl_1 + kbl_1 * 64;
                            off_kn[kbl_1] = tok_kbl_1 % 64;
                            {
                                pg_kn[kbl_1] = page_table[pt_base + tok_kbl_1 / 64];
                            }
                        }
                    }
                    if (num_kv_tiles_4 > 0) {
                        int tok_vb = start_tile_4 * 128;
                        #pragma unroll
                        for (int psv_l = 0; psv_l < 4; psv_l++) {
                            #pragma unroll
                            for (int vbv_l = 0; vbv_l < 1; vbv_l++) {
                                int tok_vl = tok_vb + psv_l * 32 + vbv_l * 32;
                                pg_vn[psv_l + vbv_l] = page_table[pt_base + tok_vl / 64];
                                off_vn[psv_l + vbv_l] = tok_vl % 64;
                            }
                        }
                    }
                    #pragma unroll 1
                    for (int tile_3 = 0; tile_3 < num_kv_tiles_4; tile_3++) {
                        int has_k = ((num_kv_tiles_4 > tile_3 + 1) ? 1 : 0);
                        int early_t = ((tile_3 <= 1) ? 1 : 0);
                        if (item_units_3 <= 1) {
                            early_t = 0;
                        }
                        int pg_kt[1];
                        int off_kt[1];
                        #pragma unroll
                        for (int kc_l = 0; kc_l < 1; kc_l++) {
                            pg_kt[kc_l] = pg_kn[kc_l];
                            off_kt[kc_l] = off_kn[kc_l];
                        }
                        int pg_vt[4];
                        int off_vt[4];
                        #pragma unroll
                        for (int vc_l = 0; vc_l < 4; vc_l++) {
                            pg_vt[vc_l] = pg_vn[vc_l];
                            off_vt[vc_l] = off_vn[vc_l];
                        }
                        if (num_kv_tiles_4 > tile_3 + 1 + 1) {
                            int tok_kl_2 = (start_tile_4 + tile_3 + 1 + 1) * 128 + cta_rank * 64;
                            #pragma unroll
                            for (int kbl_2 = 0; kbl_2 < 1; kbl_2++) {
                                int tok_kbl_2 = tok_kl_2 + kbl_2 * 64;
                                off_kn[kbl_2] = tok_kbl_2 % 64;
                                {
                                    pg_kn[kbl_2] = page_table[pt_base + tok_kbl_2 / 64];
                                }
                            }
                        }
                        if (num_kv_tiles_4 > tile_3 + 1) {
                            int tok_vb_1 = (start_tile_4 + tile_3 + 1) * 128;
                            #pragma unroll
                            for (int psv_l_1 = 0; psv_l_1 < 4; psv_l_1++) {
                                #pragma unroll
                                for (int vbv_l_1 = 0; vbv_l_1 < 1; vbv_l_1++) {
                                    int tok_vl_1 = tok_vb_1 + psv_l_1 * 32 + vbv_l_1 * 32;
                                    pg_vn[psv_l_1 + vbv_l_1] = page_table[pt_base + tok_vl_1 / 64];
                                    off_vn[psv_l_1 + vbv_l_1] = tok_vl_1 % 64;
                                }
                            }
                        }
                        if (has_k != 0) {
                            #pragma unroll
                            for (int n_l_1 = 0; n_l_1 < 9; n_l_1++) {
                                mbarrier_wait(kv_empty_addr + (load_kv_stage) * 8, _phase_kv_empty);
                                if (cta_rank == 0) {
                                    if (elect_sync()) {
                                        asm volatile(
                                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                            :: "r"((kv_full_addr + (load_kv_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                                    }
                                }
                                if (elect_sync()) {
                                    int dst_kl_1 = smem_kv_addr + load_kv_stage * 8192;
                                    #pragma unroll
                                    for (int kb_l_1 = 0; kb_l_1 < 1; kb_l_1++) {
                                        int pg_kbl_1 = pg_kt[kb_l_1];
                                        int off_kbl_1 = off_kt[kb_l_1];
                                        tma_4d_gmem2smem_cta2(dst_kl_1 + kb_l_1 * 8192, (&tmap_k), 0, off_kbl_1, n_l_1, pg_kbl_1, ((kv_full_addr + (load_kv_stage) * 8) & 0xFEFFFFFF));
                                    }
                                }
                                load_kv_stage += 1;
                                if (load_kv_stage == 15) { load_kv_stage = 0; _phase_kv_empty ^= 1; }
                            }
                        }
                        #pragma unroll
                        for (int ps_l = 0; ps_l < 4; ps_l++) {
                            #pragma unroll
                            for (int acc_l = 0; acc_l < 2; acc_l++) {
                                mbarrier_wait(kv_empty_addr + (load_kv_stage) * 8, _phase_kv_empty);
                                if (cta_rank == 0) {
                                    if (elect_sync()) {
                                        asm volatile(
                                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                            :: "r"((kv_full_addr + (load_kv_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                                    }
                                }
                                if (elect_sync()) {
                                    int dst_vl = smem_v_addr + load_kv_stage * 8192;
                                    #pragma unroll
                                    for (int vb_l = 0; vb_l < 1; vb_l++) {
                                        int pg_vl = pg_vt[ps_l + vb_l];
                                        int off_vl = off_vt[ps_l + vb_l];
                                        int v_group_l = acc_l * 4 + cta_rank * 2;
                                        tma_4d_gmem2smem_cta2(dst_vl + vb_l * 8192, (&tmap_v), 0, off_vl, v_group_l, pg_vl, ((kv_full_addr + (load_kv_stage) * 8) & 0xFEFFFFFF));
                                    }
                                }
                                load_kv_stage += 1;
                                if (load_kv_stage == 15) { load_kv_stage = 0; _phase_kv_empty ^= 1; }
                            }
                        }
                    }
                }
                if (cta_rank == 0) {
                    if (lane == 0) {
                        mbarrier_arrive(unit_done_addr);
                    }
                }
            }
            mbarrier_wait(q_empty_addr, _phase_q_empty_0);
            _phase_q_empty_0 ^= 1;
        }
    }
    // ---- Role: sched_warp ----
    if (warp == 10) {
        { // sched_warp_main
            const int wg2_dummy_2 = 0;
            {
                asm volatile("griddepcontrol.wait;" ::: "memory");
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_work_empty_1 = 1;
            if (cta_rank == 0) {
                if (static_only != 0) {
                    int lane_s = lane;
                    unsigned int stage_s = 0;
                    mbarrier_wait_cluster(work_empty_addr + (stage_s) * 8, _phase_work_empty);
                    if (lane_s == 0) {
                        work_token_words[0] = 0;
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                            "}"
                            :: "r"(work_full_addr), "r"(0) : "memory");
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                            "}"
                            :: "r"(work_full_addr), "r"(1) : "memory");
                    }
                    stage_s += 1;
                    if (stage_s == 2) { stage_s = 0; _phase_work_empty ^= 1; }
                    #pragma unroll
                    for (int _drain_s = 0; _drain_s < 2; _drain_s++) {
                        mbarrier_wait_cluster(work_empty_addr + (stage_s) * 8, _phase_work_empty);
                        stage_s += 1;
                        if (stage_s == 2) { stage_s = 0; _phase_work_empty ^= 1; }
                    }
                } else {
                    int lane_0 = lane;
                    int num_cl = num_clusters;
                    int my_cluster = cluster_id;
                    unsigned int fin_l0 = 0;
                    if (lane_0 == 0) {
                        unsigned int _load_acquire_0;
                        asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_load_acquire_0) : "l"((reinterpret_cast<unsigned int*>(sched_counters) + (1))) : "memory");
                        fin_l0 = _load_acquire_0;
                    }
                    unsigned int _shfl_34 = __shfl_sync(0xFFFFFFFF, fin_l0, 0);
                    unsigned int fin_s = _shfl_34;
                    uint32_t _fast_div_q_20 = (fd_clusters.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)fin_s), fd_clusters.multiplier) >> fd_clusters.shift_right) : (uint32_t)((unsigned int)fin_s);
                    uint32_t _fast_div_r_20 = (uint32_t)((unsigned int)fin_s) - _fast_div_q_20 * (uint32_t)(fd_clusters.divisor);
                    int par = (int)_fast_div_q_20 & 1;
                    int cnt_off = par * 4;
                    if (my_cluster == 0) {
                        int other_off = (1 - par) * 4;
                        if (lane_0 == 0) {
                            *(reinterpret_cast<unsigned int*>(sched_counters + other_off) + (0)) = 0;
                            *(reinterpret_cast<unsigned int*>(sched_counters + (other_off + 2)) + (0)) = 0;
                        }
                    }
                    int num_groups = (num_items + 31) / 32;
                    int num_chunks = (num_groups + 4 - 1) / 4;
                    int n_cache[4];
                    unsigned int total_tiles = 0;
                    unsigned int max_tiles = 0;
                    #pragma unroll 1
                    for (int chunk = 0; chunk < num_chunks; chunk++) {
                        int n_chunk[4];
                        #pragma unroll
                        for (int gi = 0; gi < 4; gi++) {
                            int g = chunk * 4 + gi;
                            int n_g = -1;
                            if (g < num_groups) {
                                int item_g = g * 32 + lane_0;
                                if (item_g < num_items) {
                                    uint32_t _fast_div_q_21 = (fd_tiles_max.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)item_g), fd_tiles_max.multiplier) >> fd_tiles_max.shift_right) : (uint32_t)((unsigned int)item_g);
                                    uint32_t _fast_div_r_21 = (uint32_t)((unsigned int)item_g) - _fast_div_q_21 * (uint32_t)(fd_tiles_max.divisor);
                                    int b_g = (int)_fast_div_q_21;
                                    int m_g = item_g - b_g * tiles_max;
                                    int q_begin_4 = cum_seq_lens_q[b_g];
                                    int q_len_5 = cum_seq_lens_q[b_g + 1] - q_begin_4;
                                    int rows_remaining_5 = q_len_5 * num_heads - m_g * 128;
                                    int valid_rows_5 = ((rows_remaining_5 < 128) ? rows_remaining_5 : 128);
                                    int k_local_5 = seq_lens[b_g];
                                    int g_bound_4 = causal_global[b_g];
                                    int n_t_4 = -1;
                                    if (valid_rows_5 > 0) {
                                        uint32_t _fast_div_q_22 = (fd_num_heads.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(m_g * 128 + valid_rows_5 - 1)), fd_num_heads.multiplier) >> fd_num_heads.shift_right) : (uint32_t)((unsigned int)(m_g * 128 + valid_rows_5 - 1));
                                        uint32_t _fast_div_r_22 = (uint32_t)((unsigned int)(m_g * 128 + valid_rows_5 - 1)) - _fast_div_q_22 * (uint32_t)(fd_num_heads.divisor);
                                        int last_q_t_4 = (int)_fast_div_q_22;
                                        int lim_num_t_4 = g_bound_4 - q_len_5 + last_q_t_4 + 1 - cp_rank;
                                        lim_num_t_4 = ((lim_num_t_4 > 0) ? lim_num_t_4 : 0);
                                        uint32_t _fast_div_q_23 = (fd_cp_world.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(lim_num_t_4 + cp_world - 1)), fd_cp_world.multiplier) >> fd_cp_world.shift_right) : (uint32_t)((unsigned int)(lim_num_t_4 + cp_world - 1));
                                        uint32_t _fast_div_r_23 = (uint32_t)((unsigned int)(lim_num_t_4 + cp_world - 1)) - _fast_div_q_23 * (uint32_t)(fd_cp_world.divisor);
                                        int lim_ceil_t_4 = (int)_fast_div_q_23;
                                        int key_lim_t_4 = ((lim_ceil_t_4 < k_local_5) ? lim_ceil_t_4 : k_local_5);
                                        key_lim_t_4 = ((key_lim_t_4 > 0) ? key_lim_t_4 : 0);
                                        n_t_4 = (key_lim_t_4 + 128 - 1) / 128;
                                    }
                                    n_g = n_t_4;
                                }
                            }
                            n_chunk[gi] = n_g;
                        }
                        #pragma unroll
                        for (int gi2 = 0; gi2 < 4; gi2++) {
                            int g2 = chunk * 4 + gi2;
                            if (g2 < num_groups) {
                                int n_g2 = n_chunk[gi2];
                                n_cache[g2] = n_g2;
                                int n_pos_g = ((n_g2 > 0) ? n_g2 : 0);
                                unsigned int n_pos_u = (unsigned int)n_pos_g;
                                unsigned int _warp_redux_u32_0;
                                asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_0) : "r"(n_pos_u));
                                total_tiles = total_tiles + _warp_redux_u32_0;
                                unsigned int _warp_redux_u32_1;
                                asm volatile("redux.sync.max.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_1) : "r"(n_pos_u));
                                unsigned int _max_2 = ((max_tiles) > (_warp_redux_u32_1) ? (max_tiles) : (_warp_redux_u32_1));
                                max_tiles = _max_2;
                            }
                        }
                    }
                    uint32_t _fast_div_q_24 = (fd_clusters.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)((int)total_tiles + num_cl - 1)), fd_clusters.multiplier) >> fd_clusters.shift_right) : (uint32_t)((unsigned int)((int)total_tiles + num_cl - 1));
                    uint32_t _fast_div_r_24 = (uint32_t)((unsigned int)((int)total_tiles + num_cl - 1)) - _fast_div_q_24 * (uint32_t)(fd_clusters.divisor);
                    int per_cluster = (int)_fast_div_q_24;
                    uint32_t _fast_div_q_25 = (fd_unit_den.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(per_cluster * unit_num + unit_den - 1)), fd_unit_den.multiplier) >> fd_unit_den.shift_right) : (uint32_t)((unsigned int)(per_cluster * unit_num + unit_den - 1));
                    uint32_t _fast_div_r_25 = (uint32_t)((unsigned int)(per_cluster * unit_num + unit_den - 1)) - _fast_div_q_25 * (uint32_t)(fd_unit_den.divisor);
                    int unit_size = (int)_fast_div_q_25;
                    unit_size = ((unit_size > unit_min) ? unit_size : unit_min);
                    int min_by_splits = ((int)max_tiles + 247 - 1) / 247;
                    unit_size = ((unit_size > min_by_splits) ? unit_size : min_by_splits);
                    unit_size = ((unit_size > 0) ? unit_size : 1);
                    unsigned int grp_b0 = 0;
                    unsigned int grp_b1 = 0;
                    unsigned int grp_b2 = 0;
                    unsigned int grp_b3 = 0;
                    unsigned int grp_b4 = 0;
                    unsigned int grp_b5 = 0;
                    unsigned int grp_units = 0;
                    unsigned int units_prefix = 0;
                    unsigned int split_total = 0;
                    uint32_t _fast_div_q_26 = (fd_num_items.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)my_cluster), fd_num_items.multiplier) >> fd_num_items.shift_right) : (uint32_t)((unsigned int)my_cluster);
                    uint32_t _fast_div_r_26 = (uint32_t)((unsigned int)my_cluster) - _fast_div_q_26 * (uint32_t)(fd_num_items.divisor);
                    int k_me = (int)_fast_div_q_26;
                    int item_me = my_cluster - k_me * num_items;
                    int my_group = item_me / 32;
                    int my_lane = item_me - my_group * 32;
                    int n_me = n_cache[my_group];
                    int _shfl_35 = __shfl_sync(0xFFFFFFFF, n_me, my_lane);
                    n_me = _shfl_35;
                    int has_static_sched = 1;
                    if (k_me > 0) {
                        has_static_sched = ((n_me - k_me * static_tiles > 0) ? 1 : 0);
                    }
                    if (k_me >= 8) {
                        has_static_sched = 0;
                    }
                    unsigned int my_units = 0;
                    unsigned int my_slot = 0;
                    #pragma unroll 1
                    for (int gp = 0; gp < num_groups; gp++) {
                        int n2 = n_cache[gp];
                        int item2 = gp * 32 + lane_0;
                        int static_i = 0;
                        if ((unsigned int)item2 < num_clusters) {
                            uint32_t _fast_div_q_27 = (fd_num_items.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(num_clusters - (unsigned int)item2 + (unsigned int)num_items - 1)), fd_num_items.multiplier) >> fd_num_items.shift_right) : (uint32_t)((unsigned int)(num_clusters - (unsigned int)item2 + (unsigned int)num_items - 1));
                            uint32_t _fast_div_r_27 = (uint32_t)((unsigned int)(num_clusters - (unsigned int)item2 + (unsigned int)num_items - 1)) - _fast_div_q_27 * (uint32_t)(fd_num_items.divisor);
                            int rounds_i = (int)_fast_div_q_27;
                            rounds_i = ((rounds_i < 8) ? rounds_i : 8);
                            if (n2 >= 0) {
                                static_i = 1;
                                if (n2 > static_tiles) {
                                    uint32_t _fast_div_q_28 = (fd_static_tiles.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(n2 - 1)), fd_static_tiles.multiplier) >> fd_static_tiles.shift_right) : (uint32_t)((unsigned int)(n2 - 1));
                                    uint32_t _fast_div_r_28 = (uint32_t)((unsigned int)(n2 - 1)) - _fast_div_q_28 * (uint32_t)(fd_static_tiles.divisor);
                                    int extra_i = (int)_fast_div_q_28;
                                    extra_i = ((extra_i < rounds_i - 1) ? extra_i : rounds_i - 1);
                                    static_i = 1 + extra_i;
                                }
                            }
                        }
                        int r_i = 0;
                        if (n2 > 0) {
                            r_i = n2 - static_i * static_tiles;
                            r_i = ((r_i > 0) ? r_i : 0);
                        }
                        int full_i = 0;
                        int rem_i = 0;
                        if (r_i > 0) {
                            full_i = r_i / unit_size;
                            rem_i = r_i - full_i * unit_size;
                        }
                        int tail_i = 0;
                        if (rem_i > 0) {
                            if (full_i > 0) {
                                if (unit_size > rem_i * 4) {
                                    tail_i = rem_i;
                                    rem_i = 0;
                                }
                            }
                        }
                        int empty_i = 0;
                        if (static_i == 0) {
                            if (n2 == 0) {
                                empty_i = 1;
                            }
                        }
                        int units_i = static_i + full_i + ((rem_i > 0) ? 1 : 0) + empty_i;
                        int cls2 = 0;
                        if (rem_i > 0) {
                            cls2 = 3;
                            if (unit_size <= rem_i * 4) {
                                cls2 = 2;
                            }
                            if (unit_size <= rem_i * 2) {
                                cls2 = 1;
                            }
                        }
                        unsigned int in_b1 = ((cls2 == 1) ? 1 : 0);
                        unsigned int in_b2 = ((cls2 == 2) ? 1 : 0);
                        unsigned int in_b3 = ((cls2 == 3) ? 1 : 0);
                        unsigned int units2 = (unsigned int)units_i;
                        unsigned int split2 = ((units_i > 1) ? 1 : 0);
                        unsigned int _warp_redux_u32_2;
                        asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_2) : "r"((unsigned int)full_i));
                        unsigned int tot_b0 = _warp_redux_u32_2;
                        unsigned int _warp_redux_u32_3;
                        asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_3) : "r"(in_b1));
                        unsigned int tot_b1 = _warp_redux_u32_3;
                        unsigned int _warp_redux_u32_4;
                        asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_4) : "r"(in_b2));
                        unsigned int tot_b2 = _warp_redux_u32_4;
                        unsigned int _warp_redux_u32_5;
                        asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_5) : "r"(in_b3));
                        unsigned int tot_b3 = _warp_redux_u32_5;
                        unsigned int _warp_redux_u32_6;
                        asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_6) : "r"((unsigned int)empty_i));
                        unsigned int tot_b4 = _warp_redux_u32_6;
                        unsigned int tot_b5 = 0;
                        unsigned int _warp_redux_u32_7;
                        asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_7) : "r"(units2));
                        unsigned int tot_units = _warp_redux_u32_7;
                        unsigned int _warp_redux_u32_8;
                        asm volatile("redux.sync.add.u32 %0, %1, 0xffffffff;" : "=r"(_warp_redux_u32_8) : "r"(split2));
                        unsigned int tot_split = _warp_redux_u32_8;
                        if (lane_0 == gp) {
                            grp_b0 = tot_b0;
                            grp_b1 = tot_b1;
                            grp_b2 = tot_b2;
                            grp_b3 = tot_b3;
                            grp_b4 = tot_b4;
                            grp_b5 = tot_b5;
                            grp_units = tot_units;
                        }
                        uint32_t _warp_scan_sum_u32_0 = units2;
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_0) : "r"(1));
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_0) : "r"(2));
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_0) : "r"(4));
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_0) : "r"(8));
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_0) : "r"(16));
                        unsigned int incl_units2 = _warp_scan_sum_u32_0;
                        if (gp == my_group) {
                            unsigned int _shfl_36 = __shfl_sync(0xFFFFFFFF, units2, my_lane);
                            my_units = _shfl_36;
                            unsigned int _shfl_37 = __shfl_sync(0xFFFFFFFF, incl_units2 - units2, my_lane);
                            my_slot = units_prefix + _shfl_37;
                        }
                        if (my_cluster == 0) {
                            uint32_t _warp_scan_sum_u32_1 = split2;
                            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_1) : "r"(1));
                            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_1) : "r"(2));
                            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_1) : "r"(4));
                            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_1) : "r"(8));
                            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_1) : "r"(16));
                            unsigned int incl_split2 = _warp_scan_sum_u32_1;
                            if (split2 != 0) {
                                int meta_at = (int)(split_total + incl_split2 - split2) * 2;
                                unsigned int meta_val = units2 | units_prefix + incl_units2 - units2 << 9;
                                *(reinterpret_cast<unsigned int*>(split_meta + meta_at) + (0)) = meta_val;
                                *(reinterpret_cast<unsigned int*>(split_meta + (meta_at + 1)) + (0)) = (unsigned int)item2;
                            }
                        }
                        units_prefix = units_prefix + tot_units;
                        split_total = split_total + tot_split;
                    }
                    if (lane_0 == 0) {
                        plan_words[0] = my_units;
                        plan_words[1] = my_slot;
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                            "}"
                            :: "r"(plan_full_addr), "r"(0) : "memory");
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                            "}"
                            :: "r"(plan_full_addr), "r"(1) : "memory");
                    }
                    if (my_cluster == 0) {
                        __syncwarp();
                        if (lane_0 == 0) {
                            *(reinterpret_cast<unsigned int*>(merge_ctl + 3) + (0)) = units_prefix;
                            asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(merge_ctl) + (1))), "r"(static_cast<unsigned int>(split_total + 1)) : "memory");
                        }
                    }
                    unsigned int pref_b0 = grp_b0;
                    unsigned int pref_b1 = grp_b1;
                    unsigned int pref_b2 = grp_b2;
                    unsigned int pref_b3 = grp_b3;
                    unsigned int pref_b4 = grp_b4;
                    unsigned int pref_b5 = grp_b5;
                    unsigned int pref_units = grp_units;
                    int total_lane = 0;
                    if (num_groups > 1) {
                        uint32_t _warp_scan_sum_u32_2 = grp_b0;
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_2) : "r"(1));
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_2) : "r"(2));
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_2) : "r"(4));
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_2) : "r"(8));
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_2) : "r"(16));
                        pref_b0 = _warp_scan_sum_u32_2;
                        uint32_t _warp_scan_sum_u32_3 = grp_b1;
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_3) : "r"(1));
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_3) : "r"(2));
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_3) : "r"(4));
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_3) : "r"(8));
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_3) : "r"(16));
                        pref_b1 = _warp_scan_sum_u32_3;
                        uint32_t _warp_scan_sum_u32_4 = grp_b2;
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_4) : "r"(1));
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_4) : "r"(2));
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_4) : "r"(4));
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_4) : "r"(8));
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_4) : "r"(16));
                        pref_b2 = _warp_scan_sum_u32_4;
                        uint32_t _warp_scan_sum_u32_5 = grp_b3;
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_5) : "r"(1));
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_5) : "r"(2));
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_5) : "r"(4));
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_5) : "r"(8));
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_5) : "r"(16));
                        pref_b3 = _warp_scan_sum_u32_5;
                        uint32_t _warp_scan_sum_u32_6 = grp_b4;
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_6) : "r"(1));
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_6) : "r"(2));
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_6) : "r"(4));
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_6) : "r"(8));
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_6) : "r"(16));
                        pref_b4 = _warp_scan_sum_u32_6;
                        uint32_t _warp_scan_sum_u32_7 = grp_b5;
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_7) : "r"(1));
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_7) : "r"(2));
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_7) : "r"(4));
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_7) : "r"(8));
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_7) : "r"(16));
                        pref_b5 = _warp_scan_sum_u32_7;
                        uint32_t _warp_scan_sum_u32_8 = grp_units;
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_8) : "r"(1));
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_8) : "r"(2));
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_8) : "r"(4));
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_8) : "r"(8));
                        asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_8) : "r"(16));
                        pref_units = _warp_scan_sum_u32_8;
                        total_lane = 31;
                    }
                    unsigned int bucket_end[6];
                    unsigned int _shfl_38 = __shfl_sync(0xFFFFFFFF, pref_b0, total_lane);
                    bucket_end[0] = _shfl_38;
                    unsigned int _shfl_39 = __shfl_sync(0xFFFFFFFF, pref_b1, total_lane);
                    bucket_end[1] = bucket_end[0] + _shfl_39;
                    unsigned int _shfl_40 = __shfl_sync(0xFFFFFFFF, pref_b2, total_lane);
                    bucket_end[2] = bucket_end[1] + _shfl_40;
                    unsigned int _shfl_41 = __shfl_sync(0xFFFFFFFF, pref_b3, total_lane);
                    bucket_end[3] = bucket_end[2] + _shfl_41;
                    unsigned int _shfl_42 = __shfl_sync(0xFFFFFFFF, pref_b4, total_lane);
                    bucket_end[4] = bucket_end[3] + _shfl_42;
                    unsigned int _shfl_43 = __shfl_sync(0xFFFFFFFF, pref_b5, total_lane);
                    bucket_end[5] = bucket_end[4] + _shfl_43;
                    unsigned int total_tickets = bucket_end[5];
                    unsigned int work_stage_sched = 0;
                    int claim_wait = has_static_sched;
                    int unit_wait_phase = 0;
                    unsigned int first_pub = 1;
                    unsigned int no_more = ((total_tickets == 0) ? 1 : 0);
                    #pragma unroll 1
                    for (unsigned int _claim = 0; _claim < max_units + 1; _claim++) {
                        if (claim_wait != 0) {
                            if (no_more == 0) {
                                mbarrier_wait(unit_done_addr, unit_wait_phase);
                                unit_wait_phase = unit_wait_phase ^ 1;
                            }
                        }
                        claim_wait = 1;
                        mbarrier_wait_cluster(work_empty_addr + (work_stage_sched) * 8, _phase_work_empty_1);
                        unsigned int ticket_lane0 = total_tickets;
                        if (no_more == 0) {
                            if (lane_0 == 0) {
                                unsigned int _atomic_old_0 = atomicAdd(&sched_counters[cnt_off], 1);
                                ticket_lane0 = _atomic_old_0;
                            }
                        }
                        unsigned int _shfl_44 = __shfl_sync(0xFFFFFFFF, ticket_lane0, 0);
                        unsigned int ticket = _shfl_44;
                        if (total_tickets <= ticket + 1) {
                            no_more = 1;
                        }
                        unsigned int token_base = work_stage_sched * 8;
                        unsigned int valid_tok = ((ticket < total_tickets) ? 1 : 0);
                        unsigned int kind_tok = valid_tok;
                        if (valid_tok != 0) {
                            int bucket = 0;
                            unsigned int bucket_start = 0;
                            #pragma unroll
                            for (int bi = 0; bi < 5; bi++) {
                                if (ticket >= bucket_end[bi]) {
                                    bucket = bi + 1;
                                    bucket_start = bucket_end[bi];
                                }
                            }
                            unsigned int t_local = ticket - bucket_start;
                            unsigned int pref_sel = pref_b0;
                            unsigned int cnt_sel = grp_b0;
                            if (bucket == 1) {
                                pref_sel = pref_b1;
                                cnt_sel = grp_b1;
                            }
                            if (bucket == 2) {
                                pref_sel = pref_b2;
                                cnt_sel = grp_b2;
                            }
                            if (bucket == 3) {
                                pref_sel = pref_b3;
                                cnt_sel = grp_b3;
                            }
                            if (bucket == 4) {
                                pref_sel = pref_b4;
                                cnt_sel = grp_b4;
                            }
                            if (bucket == 5) {
                                pref_sel = pref_b5;
                                cnt_sel = grp_b5;
                            }
                            unsigned int _vote_2 = __ballot_sync(0xFFFFFFFF, t_local < pref_sel);
                            unsigned int grp_mask = _vote_2;
                            int _ffs_0 = __ffs(grp_mask);
                            int hit_group = _ffs_0 - 1;
                            unsigned int _shfl_45 = __shfl_sync(0xFFFFFFFF, pref_sel - cnt_sel, hit_group);
                            unsigned int before = _shfl_45;
                            unsigned int _shfl_46 = __shfl_sync(0xFFFFFFFF, pref_units - grp_units, hit_group);
                            unsigned int ubefore = _shfl_46;
                            unsigned int in_group = t_local - before;
                            int n3 = n_cache[hit_group];
                            int item3 = hit_group * 32 + lane_0;
                            int static_i_1 = 0;
                            if ((unsigned int)item3 < num_clusters) {
                                uint32_t _fast_div_q_29 = (fd_num_items.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(num_clusters - (unsigned int)item3 + (unsigned int)num_items - 1)), fd_num_items.multiplier) >> fd_num_items.shift_right) : (uint32_t)((unsigned int)(num_clusters - (unsigned int)item3 + (unsigned int)num_items - 1));
                                uint32_t _fast_div_r_29 = (uint32_t)((unsigned int)(num_clusters - (unsigned int)item3 + (unsigned int)num_items - 1)) - _fast_div_q_29 * (uint32_t)(fd_num_items.divisor);
                                int rounds_i_1 = (int)_fast_div_q_29;
                                rounds_i_1 = ((rounds_i_1 < 8) ? rounds_i_1 : 8);
                                if (n3 >= 0) {
                                    static_i_1 = 1;
                                    if (n3 > static_tiles) {
                                        uint32_t _fast_div_q_30 = (fd_static_tiles.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)(n3 - 1)), fd_static_tiles.multiplier) >> fd_static_tiles.shift_right) : (uint32_t)((unsigned int)(n3 - 1));
                                        uint32_t _fast_div_r_30 = (uint32_t)((unsigned int)(n3 - 1)) - _fast_div_q_30 * (uint32_t)(fd_static_tiles.divisor);
                                        int extra_i_1 = (int)_fast_div_q_30;
                                        extra_i_1 = ((extra_i_1 < rounds_i_1 - 1) ? extra_i_1 : rounds_i_1 - 1);
                                        static_i_1 = 1 + extra_i_1;
                                    }
                                }
                            }
                            int r_i_1 = 0;
                            if (n3 > 0) {
                                r_i_1 = n3 - static_i_1 * static_tiles;
                                r_i_1 = ((r_i_1 > 0) ? r_i_1 : 0);
                            }
                            int full_i_1 = 0;
                            int rem_i_1 = 0;
                            if (r_i_1 > 0) {
                                full_i_1 = r_i_1 / unit_size;
                                rem_i_1 = r_i_1 - full_i_1 * unit_size;
                            }
                            int tail_i_1 = 0;
                            if (rem_i_1 > 0) {
                                if (full_i_1 > 0) {
                                    if (unit_size > rem_i_1 * 4) {
                                        tail_i_1 = rem_i_1;
                                        rem_i_1 = 0;
                                    }
                                }
                            }
                            int empty_i_1 = 0;
                            if (static_i_1 == 0) {
                                if (n3 == 0) {
                                    empty_i_1 = 1;
                                }
                            }
                            int units_i_1 = static_i_1 + full_i_1 + ((rem_i_1 > 0) ? 1 : 0) + empty_i_1;
                            int cls3 = 0;
                            if (rem_i_1 > 0) {
                                cls3 = 3;
                                if (unit_size <= rem_i_1 * 4) {
                                    cls3 = 2;
                                }
                                if (unit_size <= rem_i_1 * 2) {
                                    cls3 = 1;
                                }
                            }
                            unsigned int mine3 = 0;
                            if (bucket == 0) {
                                mine3 = (unsigned int)full_i_1;
                            } else if (bucket == 4) {
                                mine3 = (unsigned int)empty_i_1;
                            } else {
                                mine3 = ((cls3 == bucket) ? 1 : 0);
                            }
                            unsigned int units3 = (unsigned int)units_i_1;
                            uint32_t _warp_scan_sum_u32_9 = mine3;
                            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_9) : "r"(1));
                            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_9) : "r"(2));
                            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_9) : "r"(4));
                            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_9) : "r"(8));
                            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_9) : "r"(16));
                            unsigned int incl3 = _warp_scan_sum_u32_9;
                            uint32_t _warp_scan_sum_u32_10 = units3;
                            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_10) : "r"(1));
                            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_10) : "r"(2));
                            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_10) : "r"(4));
                            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_10) : "r"(8));
                            asm volatile("{ .reg .pred p; .reg .b32 t; shfl.sync.up.b32 t|p, %0, %1, 0, 0xffffffff; @p add.u32 %0, %0, t; }" : "+r"(_warp_scan_sum_u32_10) : "r"(16));
                            unsigned int incl_u3 = _warp_scan_sum_u32_10;
                            unsigned int excl3 = incl3 - mine3;
                            int hit3 = 0;
                            if (mine3 > 0) {
                                if (excl3 <= in_group) {
                                    if (in_group < incl3) {
                                        hit3 = 1;
                                    }
                                }
                            }
                            unsigned int _vote_3 = __ballot_sync(0xFFFFFFFF, hit3 != 0);
                            unsigned int hit_mask = _vote_3;
                            int _ffs_1 = __ffs(hit_mask);
                            int hit_lane = _ffs_1 - 1;
                            int sel_item = hit_group * 32 + hit_lane;
                            int _shfl_47 = __shfl_sync(0xFFFFFFFF, static_i_1, hit_lane);
                            int sel_static = _shfl_47;
                            int _shfl_48 = __shfl_sync(0xFFFFFFFF, full_i_1, hit_lane);
                            int sel_full = _shfl_48;
                            int _shfl_49 = __shfl_sync(0xFFFFFFFF, rem_i_1, hit_lane);
                            int sel_rem = _shfl_49;
                            int _shfl_50 = __shfl_sync(0xFFFFFFFF, tail_i_1, hit_lane);
                            int sel_tail = _shfl_50;
                            int _shfl_51 = __shfl_sync(0xFFFFFFFF, units_i_1, hit_lane);
                            int sel_units = _shfl_51;
                            unsigned int _shfl_52 = __shfl_sync(0xFFFFFFFF, excl3, hit_lane);
                            unsigned int sel_excl = _shfl_52;
                            unsigned int _shfl_53 = __shfl_sync(0xFFFFFFFF, incl_u3 - units3, hit_lane);
                            unsigned int sel_excl_units = _shfl_53;
                            int unit_idx_sel = sel_static + sel_full;
                            int start_sel = sel_static * static_tiles + sel_full * unit_size;
                            int n_tiles_sel = sel_rem;
                            if (bucket == 0) {
                                int k_sel = (int)(in_group - sel_excl);
                                unit_idx_sel = sel_static + k_sel;
                                start_sel = sel_static * static_tiles + k_sel * unit_size;
                                n_tiles_sel = unit_size;
                                if (k_sel == sel_full - 1) {
                                    n_tiles_sel = unit_size + sel_tail;
                                }
                            }
                            if (bucket == 4) {
                                unit_idx_sel = 0;
                                start_sel = 0;
                                n_tiles_sel = 0;
                            }
                            uint32_t _fast_div_q_31 = (fd_tiles_max.divisor != 1) ? (__umulhi((uint32_t)((unsigned int)sel_item), fd_tiles_max.multiplier) >> fd_tiles_max.shift_right) : (uint32_t)((unsigned int)sel_item);
                            uint32_t _fast_div_r_31 = (uint32_t)((unsigned int)sel_item) - _fast_div_q_31 * (uint32_t)(fd_tiles_max.divisor);
                            int sel_batch = (int)_fast_div_q_31;
                            int sel_mtile = sel_item - sel_batch * tiles_max;
                            unsigned int slot_word = ubefore + sel_excl_units;
                            if (lane_0 == 0) {
                                work_token_words[token_base + 1] = (unsigned int)sel_batch;
                                work_token_words[token_base + 2] = (unsigned int)sel_mtile;
                                work_token_words[token_base + 3] = (unsigned int)start_sel;
                                work_token_words[token_base + 4] = (unsigned int)n_tiles_sel;
                                work_token_words[token_base + 5] = (unsigned int)unit_idx_sel;
                                work_token_words[token_base + 6] = (unsigned int)sel_units;
                                work_token_words[token_base + 7] = slot_word;
                            }
                        }
                        if (lane_0 == 0) {
                            work_token_words[token_base] = kind_tok;
                            asm volatile(
                                "{\n\t"
                                ".reg .b32 remAddr32;\n\t"
                                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                                "}"
                                :: "r"(work_full_addr + work_stage_sched * 8), "r"(0) : "memory");
                            asm volatile(
                                "{\n\t"
                                ".reg .b32 remAddr32;\n\t"
                                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                                "}"
                                :: "r"(work_full_addr + work_stage_sched * 8), "r"(1) : "memory");
                        }
                        work_stage_sched += 1;
                        if (work_stage_sched == 2) { work_stage_sched = 0; _phase_work_empty_1 ^= 1; }
                        first_pub = 0;
                        if (valid_tok == 0) {
                            break;
                        }
                    }
                    #pragma unroll
                    for (int _drain = 0; _drain < 2; _drain++) {
                        mbarrier_wait_cluster(work_empty_addr + (work_stage_sched) * 8, _phase_work_empty_1);
                        work_stage_sched += 1;
                        if (work_stage_sched == 2) { work_stage_sched = 0; _phase_work_empty_1 ^= 1; }
                    }
                    if (lane_0 == 0) {
                        asm volatile("red.release.gpu.global.add.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(sched_counters) + (1))), "r"(static_cast<unsigned int>(1)) : "memory");
                    }
                }
            }
        }
    }
    // ---- Role: empty1 ----
    if (warp == 11) {
        { // empty1_main
            const int wg2_dummy_3 = 0;
            {
                asm volatile("griddepcontrol.wait;" ::: "memory");
                asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            }
            asm volatile("tcgen05.fence::after_thread_sync;");
            unsigned int _phase_pub_req_0 = 0;
            unsigned int _phase_merge_ready_0 = 0;
            if (cta_rank == 0) {
                unsigned int first_pub_iter = 1;
                unsigned int flag_one = 1;
                int flag_off_p = 0;
                #pragma unroll 1
                for (unsigned int _pub_iter = 0; _pub_iter < max_units + 1; _pub_iter++) {
                    mbarrier_wait(pub_req_addr, _phase_pub_req_0);
                    _phase_pub_req_0 ^= 1;
                    unsigned int pub_kind = pub_words[0];
                    if (pub_kind == 0) {
                        break;
                    }
                    unsigned int pub_slot = pub_words[1];
                    unsigned int pub_unit = pub_words[2];
                    mbarrier_wait_cluster(merge_ready_addr, _phase_merge_ready_0);
                    _phase_merge_ready_0 ^= 1;
                    if (lane == 0) {
                        mbarrier_arrive(pub_free_addr);
                        asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"((reinterpret_cast<unsigned int*>(unit_flags) + (flag_off_p + (int)(pub_slot + pub_unit)))), "r"(static_cast<unsigned int>(flag_one)) : "memory");
                    }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
