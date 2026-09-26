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
#define NUM_KV_PIPE_STAGES 15
#define NUM_WORK_PIPE_STAGES 2
#define NUM_THROTTLE_PIPE_STAGES 2
#define SMEM_SMEM_Q_OFF 0
#define SMEM_SMEM_Q_STAGE_BYTES 8192
#define SMEM_SMEM_Q_STRIDE 8192
#define SMEM_SMEM_KV_OFF 73728
#define SMEM_SMEM_KV_STAGE_BYTES 8192
#define SMEM_SMEM_KV_STRIDE 8192
#define SMEM_SMEM_VHALF_OFF 73728
#define SMEM_SMEM_VHALF_STAGE_BYTES 8192
#define SMEM_SMEM_VHALF_STRIDE 8192
#define SMEM_SMEM_VQUARTER_OFF 73728
#define SMEM_SMEM_VQUARTER_STAGE_BYTES 4096
#define SMEM_SMEM_VQUARTER_STRIDE 8192
#define SMEM_SMEM_P_OFF 196608
#define SMEM_SMEM_P_STAGE_BYTES 4096
#define SMEM_SMEM_P_STRIDE 4096
#define SMEM_SMEM_SOFTMAX_EXCHANGE_OFF 229376
#define SMEM_SMEM_SOFTMAX_EXCHANGE_STAGE_BYTES 512
#define SMEM_SMEM_SOFTMAX_EXCHANGE_STRIDE 512
#define SMEM_SMEM_EPILOGUE_EXCHANGE_OFF 229888
#define SMEM_SMEM_EPILOGUE_EXCHANGE_STAGE_BYTES 512
#define SMEM_SMEM_EPILOGUE_EXCHANGE_STRIDE 512
#define SMEM_WORK_RESPONSE_OFF 230400
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_TOTAL 230912
#define THREADS 384
#define Q4_B1024 1

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


__device__ __forceinline__ void tma_4d_gmem2smem_cta2(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w),
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


__device__ __forceinline__ void tmem_ld_x16_wait(float* dst, int addr) {
    tmem_ld_x16(dst, addr);
    asm volatile("tcgen05.wait::ld.sync.aligned;");
}


__device__ __forceinline__ void tmem_ld_x8(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3]),
          "=f"(dst[4]), "=f"(dst[5]), "=f"(dst[6]), "=f"(dst[7])
        : "r"(tmem_addr));
}


__device__ __forceinline__ unsigned int __as_u32(float v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "f"(v));
    return u;
}
__device__ __forceinline__ unsigned int __as_u32(__nv_bfloat162 v) {
    return *reinterpret_cast<const unsigned int*>(&v);
}
__device__ __forceinline__ unsigned int __as_u32(unsigned int v) { return v; }
__device__ __forceinline__ unsigned int __as_u32(int v) {
    unsigned int u;
    asm("mov.b32 %0, %1;" : "=r"(u) : "r"(v));
    return u;
}

extern "C" {

__global__ __launch_bounds__(384, 1) __cluster_dims__(2,1,1) void
kernel_mla_decode_exact_live_bf16_clc(const __grid_constant__ CUtensorMap tmap_q, const __grid_constant__ CUtensorMap tmap_k, const __grid_constant__ CUtensorMap tmap_v, __nv_bfloat16* __restrict__ O, int* __restrict__ seq_lens_kv, int* __restrict__ page_table, float* __restrict__ sinks, float softmax_scale_log2, float bmm2_scale, int total_work_items, int value_split_count, int max_pages_per_seq, int enable_sink)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem + 230432;
    #define q_full_addr (mbar_base + 0)
    #define q_empty_addr (mbar_base + 8)
    #define kv_full_addr (mbar_base + 16)
    #define kv_empty_addr (mbar_base + 136)
    #define s_full_addr (mbar_base + 256)
    #define s_empty_addr (mbar_base + 272)
    #define p_full_addr (mbar_base + 288)
    #define p_empty_addr (mbar_base + 304)
    #define o_empty_addr (mbar_base + 320)
    #define stats_addr (mbar_base + 328)
    #define stats_empty_addr (mbar_base + 344)
    #define o_full_addr (mbar_base + 360)
    #define tmem_scrubbed_addr (mbar_base + 368)
    #define tmem_dealloc_peer_addr (mbar_base + 376)
    #define work_full_addr (mbar_base + 384)
    #define work_empty_addr (mbar_base + 400)
    #define throttle_full_addr (mbar_base + 416)
    #define throttle_empty_addr (mbar_base + 432)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    __nv_bfloat16* smem_q = reinterpret_cast<__nv_bfloat16*>(smem_raw + 0);
    const int smem_q_addr = smem + 0;
    __nv_bfloat16* smem_kv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 73728);
    const int smem_kv_addr = smem + 73728;
    __nv_bfloat16* smem_vhalf = reinterpret_cast<__nv_bfloat16*>(smem_raw + 73728);
    const int smem_vhalf_addr = smem + 73728;
    __nv_bfloat16* smem_vquarter = reinterpret_cast<__nv_bfloat16*>(smem_raw + 73728);
    const int smem_vquarter_addr = smem + 73728;
    __nv_bfloat16* smem_p = reinterpret_cast<__nv_bfloat16*>(smem_raw + 196608);
    const int smem_p_addr = smem + 196608;
    float* smem_softmax_exchange = reinterpret_cast<float*>(smem_raw + 229376);
    const int smem_softmax_exchange_addr = smem + 229376;
    float* smem_epilogue_exchange = reinterpret_cast<float*>(smem_raw + 229888);
    const int smem_epilogue_exchange_addr = smem + 229888;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 230400);
    const int work_response_addr = smem + 230400;

    // Mbarrier init (18 pipeline groups, 0 ordered-sequence groups, 56 barriers)
    // Mbarriers at smem_raw[230432..230880)

    if (warp == 0) {
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
        // tmem_scrubbed: 1 barriers, init_count=256
        // tmem_dealloc_peer: 1 barriers, init_count=32
        // --- pipeline 'work_pipe' ---
        // work_full: 2 barriers, init_count=1
        // work_empty: 2 barriers, init_count=704
        // --- pipeline 'throttle_pipe' ---
        // throttle_full: 2 barriers, init_count=32
        // throttle_empty: 2 barriers, init_count=32
        // Warp-cooperative initialization in physical record order.
        mbarrier_init(smem + 230432 + lane * 8, 1);
        uint32_t _mbarrier_init_count_0_32 = 32;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(20), "r"((uint32_t)(704)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(18), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(16), "r"((uint32_t)(32)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(15), "r"((uint32_t)(256)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(14), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(13), "r"((uint32_t)(128)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(9), "r"((uint32_t)(256)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(8), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(6), "r"((uint32_t)(256)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(2), "r"((uint32_t)(1)));
        if (lane < 24) {
            mbarrier_init(smem + 230688 + lane * 8, _mbarrier_init_count_0_32);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 230880);
    if (warp == 0) {
        int _tmem_hold = smem + 230880;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    // Partial post-allocation TMEM rendezvous (384 threads on named barrier 0)
    if (warp <= 11) {
        asm volatile("barrier.sync.aligned %0, %1;" :: "r"(0), "r"(384) : "memory");
        asm volatile("tcgen05.fence::after_thread_sync;");
    }

    const int taddr = (warp <= 11) ? tmem_addr_storage[0] : 0;

    // Kernel post-init ops
    const int tmem_tmem_scratch = taddr;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 96;");
    }

    // ---- Role: load_warp ----
    if (warp == 9) {
        { // load_warp_main
            const int wg2_dummy = 0;
            unsigned int load_kv_stage = 0;
            unsigned int load_kv_phase = 1;
            unsigned int load_kv_token = 0;
            unsigned int load_work_idx = blockIdx.z * ((Q4_B1024) ? 4 : value_split_count) + blockIdx.x / 2;
            unsigned int load_work_stage = 0;
            unsigned int load_throttle_stage = 0;
            unsigned int _phase_throttle_empty = 1;
            unsigned int _phase_q_empty_0 = 1;
            unsigned int _phase_kv_empty = 1;
            unsigned int _phase_work_full = 0;
            #pragma unroll 1
            for (unsigned int work_idx = 0; work_idx < ((Q4_B1024) ? 4096 : total_work_items); work_idx++) {
                unsigned int load_ordered_work_idx = load_work_idx;
                if (((Q4_B1024) ? 4096 : total_work_items) > 76) {
                    if (((Q4_B1024) ? 4096 : total_work_items) <= 1024) {
                        load_ordered_work_idx = (unsigned int)page_table[((Q4_B1024) ? 4096 : total_work_items) * ((Q4_B1024) ? 32 : max_pages_per_seq) + (int)load_work_idx];
                    }
                }
                if (cta_rank == 0) {
                    mbarrier_wait(throttle_empty_addr + (load_throttle_stage) * 8, _phase_throttle_empty);
                    mbarrier_arrive(throttle_full_addr + (load_throttle_stage) * 8);
                    load_throttle_stage += 1;
                    if (load_throttle_stage == 2) { load_throttle_stage = 0; _phase_throttle_empty ^= 1; }
                }
                int batch_idx = load_ordered_work_idx;
                int value_split = 0;
                int seqlen_kv_b = seq_lens_kv[batch_idx];
                int total_kv_tiles = (seqlen_kv_b + 128 - 1) / 128;
                int start_tile = 0;
                int num_kv_tiles = total_kv_tiles;
                int pt_base = batch_idx * ((Q4_B1024) ? 32 : max_pages_per_seq);
                int q_row_global = batch_idx * 128 + cta_rank * 64;
                mbarrier_wait(q_empty_addr, _phase_q_empty_0);
                _phase_q_empty_0 ^= 1;
                asm volatile("griddepcontrol.wait;" ::: "memory");
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
                int k0_page_base = pt_base + 4 * start_tile + 2 * cta_rank;
                int pg_k00 = 0;
                int pg_k01 = 0;
                {
                    pg_k00 = page_table[k0_page_base];
                    pg_k01 = page_table[k0_page_base + 1];
                }
                #pragma unroll
                for (int n = 0; n < 8; n++) {
                    int dst = 0;
                    {
                        uint32_t _mbar_token_0 = mbarrier_try_wait(kv_empty_addr + (load_kv_stage) * 8, load_kv_phase);
                        load_kv_token = _mbar_token_0;
                        dst = smem_kv_addr + load_kv_stage * 8192;
                        mbarrier_wait_token(kv_empty_addr + (load_kv_stage) * 8, load_kv_phase, load_kv_token);
                    }
                    if (cta_rank == 0) {
                        if (elect_sync()) {
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((kv_full_addr + (load_kv_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                        }
                    }
                    if (elect_sync()) {
                        tma_4d_gmem2smem_cta2(dst, (&tmap_k), 0, 0, n, pg_k00, ((kv_full_addr + (load_kv_stage) * 8) & 0xFEFFFFFF));
                        tma_4d_gmem2smem_cta2(dst + 4096, (&tmap_k), 0, 0, n, pg_k01, ((kv_full_addr + (load_kv_stage) * 8) & 0xFEFFFFFF));
                    }
                    {
                        load_kv_stage += 1;
                        if (load_kv_stage == 15) { load_kv_stage = 0; load_kv_phase ^= 1; }
                    }
                }
                int dst_r = 0;
                {
                    uint32_t _mbar_token_1 = mbarrier_try_wait(kv_empty_addr + (load_kv_stage) * 8, load_kv_phase);
                    load_kv_token = _mbar_token_1;
                    dst_r = smem_kv_addr + load_kv_stage * 8192;
                    mbarrier_wait_token(kv_empty_addr + (load_kv_stage) * 8, load_kv_phase, load_kv_token);
                }
                if (cta_rank == 0) {
                    if (elect_sync()) {
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((kv_full_addr + (load_kv_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                    }
                }
                if (elect_sync()) {
                    tma_4d_gmem2smem_cta2(dst_r, (&tmap_k), 0, 0, 8, pg_k00, ((kv_full_addr + (load_kv_stage) * 8) & 0xFEFFFFFF));
                    tma_4d_gmem2smem_cta2(dst_r + 4096, (&tmap_k), 0, 0, 8, pg_k01, ((kv_full_addr + (load_kv_stage) * 8) & 0xFEFFFFFF));
                }
                {
                    load_kv_stage += 1;
                    if (load_kv_stage == 15) { load_kv_stage = 0; load_kv_phase ^= 1; }
                }
                #pragma unroll 1
                for (int tile = 1; tile < num_kv_tiles; tile++) {
                    int abs_tile_k = start_tile + tile;
                    int k_page_base = pt_base + 4 * abs_tile_k + 2 * cta_rank;
                    int pg_k0 = 0;
                    int pg_k1 = 0;
                    {
                        pg_k0 = page_table[k_page_base];
                        pg_k1 = page_table[k_page_base + 1];
                    }
                    #pragma unroll
                    for (int n_1 = 0; n_1 < 8; n_1++) {
                        int dst_k = 0;
                        {
                            uint32_t _mbar_token_2 = mbarrier_try_wait(kv_empty_addr + (load_kv_stage) * 8, load_kv_phase);
                            load_kv_token = _mbar_token_2;
                            dst_k = smem_kv_addr + load_kv_stage * 8192;
                            mbarrier_wait_token(kv_empty_addr + (load_kv_stage) * 8, load_kv_phase, load_kv_token);
                        }
                        if (cta_rank == 0) {
                            if (elect_sync()) {
                                asm volatile(
                                    "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                    :: "r"((kv_full_addr + (load_kv_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                            }
                        }
                        if (elect_sync()) {
                            tma_4d_gmem2smem_cta2(dst_k, (&tmap_k), 0, 0, n_1, pg_k0, ((kv_full_addr + (load_kv_stage) * 8) & 0xFEFFFFFF));
                            tma_4d_gmem2smem_cta2(dst_k + 4096, (&tmap_k), 0, 0, n_1, pg_k1, ((kv_full_addr + (load_kv_stage) * 8) & 0xFEFFFFFF));
                        }
                        {
                            load_kv_stage += 1;
                            if (load_kv_stage == 15) { load_kv_stage = 0; load_kv_phase ^= 1; }
                        }
                    }
                    int dst_kr = 0;
                    {
                        uint32_t _mbar_token_3 = mbarrier_try_wait(kv_empty_addr + (load_kv_stage) * 8, load_kv_phase);
                        load_kv_token = _mbar_token_3;
                        dst_kr = smem_kv_addr + load_kv_stage * 8192;
                        mbarrier_wait_token(kv_empty_addr + (load_kv_stage) * 8, load_kv_phase, load_kv_token);
                    }
                    if (cta_rank == 0) {
                        if (elect_sync()) {
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((kv_full_addr + (load_kv_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                        }
                    }
                    if (elect_sync()) {
                        tma_4d_gmem2smem_cta2(dst_kr, (&tmap_k), 0, 0, 8, pg_k0, ((kv_full_addr + (load_kv_stage) * 8) & 0xFEFFFFFF));
                        tma_4d_gmem2smem_cta2(dst_kr + 4096, (&tmap_k), 0, 0, 8, pg_k1, ((kv_full_addr + (load_kv_stage) * 8) & 0xFEFFFFFF));
                    }
                    {
                        load_kv_stage += 1;
                        if (load_kv_stage == 15) { load_kv_stage = 0; load_kv_phase ^= 1; }
                    }
                    int abs_tile_v = start_tile + tile - 1;
                    #pragma unroll
                    for (int ps = 0; ps < 4; ps++) {
                        int pg_v = page_table[pt_base + 4 * abs_tile_v + ps];
                        #pragma unroll
                        for (int acc_stage = 0; acc_stage < ((1) ? 2 : 1); acc_stage++) {
                            int v_group_half = 0;
                            int dst_vhalf = 0;
                            {
                                uint32_t _mbar_token_4 = mbarrier_try_wait(kv_empty_addr + (load_kv_stage) * 8, load_kv_phase);
                                load_kv_token = _mbar_token_4;
                                v_group_half = acc_stage * 4 + cta_rank * 2;
                                dst_vhalf = smem_vhalf_addr + load_kv_stage * 8192;
                                mbarrier_wait_token(kv_empty_addr + (load_kv_stage) * 8, load_kv_phase, load_kv_token);
                            }
                            {
                                if (cta_rank == 0) {
                                    if (elect_sync()) {
                                        asm volatile(
                                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                            :: "r"((kv_full_addr + (load_kv_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                                    }
                                }
                                if (elect_sync()) {
                                    tma_4d_gmem2smem_cta2(dst_vhalf, (&tmap_v), 0, 0, v_group_half, pg_v, ((kv_full_addr + (load_kv_stage) * 8) & 0xFEFFFFFF));
                                    tma_4d_gmem2smem_cta2(dst_vhalf + 4096, (&tmap_v), 0, 0, v_group_half + 1, pg_v, ((kv_full_addr + (load_kv_stage) * 8) & 0xFEFFFFFF));
                                }
                            }
                            {
                                load_kv_stage += 1;
                                if (load_kv_stage == 15) { load_kv_stage = 0; load_kv_phase ^= 1; }
                            }
                        }
                    }
                }
                int abs_tile_last = start_tile + num_kv_tiles - 1;
                #pragma unroll
                for (int ps_1 = 0; ps_1 < 4; ps_1++) {
                    int pg_v_last = page_table[pt_base + 4 * abs_tile_last + ps_1];
                    #pragma unroll
                    for (int acc_stage_1 = 0; acc_stage_1 < ((1) ? 2 : 1); acc_stage_1++) {
                        int v_group_half_last = 0;
                        int dst_vhalf_last = 0;
                        {
                            uint32_t _mbar_token_5 = mbarrier_try_wait(kv_empty_addr + (load_kv_stage) * 8, load_kv_phase);
                            load_kv_token = _mbar_token_5;
                            v_group_half_last = acc_stage_1 * 4 + cta_rank * 2;
                            dst_vhalf_last = smem_vhalf_addr + load_kv_stage * 8192;
                            mbarrier_wait_token(kv_empty_addr + (load_kv_stage) * 8, load_kv_phase, load_kv_token);
                        }
                        {
                            if (cta_rank == 0) {
                                if (elect_sync()) {
                                    asm volatile(
                                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                        :: "r"((kv_full_addr + (load_kv_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(16384)) : "memory");
                                }
                            }
                            if (elect_sync()) {
                                tma_4d_gmem2smem_cta2(dst_vhalf_last, (&tmap_v), 0, 0, v_group_half_last, pg_v_last, ((kv_full_addr + (load_kv_stage) * 8) & 0xFEFFFFFF));
                                tma_4d_gmem2smem_cta2(dst_vhalf_last + 4096, (&tmap_v), 0, 0, v_group_half_last + 1, pg_v_last, ((kv_full_addr + (load_kv_stage) * 8) & 0xFEFFFFFF));
                            }
                        }
                        {
                            load_kv_stage += 1;
                            if (load_kv_stage == 15) { load_kv_stage = 0; load_kv_phase ^= 1; }
                        }
                    }
                }
                mbarrier_wait(work_full_addr + (load_work_stage) * 8, _phase_work_full);
                uint32_t _clc_valid_0 = 0;
                uint32_t _clc_ctaid_x_0;
                uint32_t _clc_ctaid_y_0;
                uint32_t _clc_ctaid_z_0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%4];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %3, 1, 0, p1;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, %1, %2, _}, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_x_0), "=r"(_clc_ctaid_y_0), "=r"(_clc_ctaid_z_0), "=r"(_clc_valid_0)
                    : "r"(work_response_addr + load_work_stage * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + load_work_stage * 8), "r"(0) : "memory");
                load_work_stage += 1;
                if (load_work_stage == 2) { load_work_stage = 0; _phase_work_full ^= 1; }
                unsigned int load_work_valid = _clc_valid_0;
                if (load_work_valid == 0) {
                    break;
                }
                load_work_idx = _clc_ctaid_z_0 * (unsigned int)(((Q4_B1024) ? 4 : value_split_count)) + _clc_ctaid_x_0 / 2;
            }
            mbarrier_wait(q_empty_addr, _phase_q_empty_0);
            _phase_q_empty_0 ^= 1;
        }
    // ---- Role: empty1 ----
    } else if (warp == 11) {
        { // empty1_main
            const int wg2_dummy_1 = 0;
            unsigned int empty1_work_idx = blockIdx.z * ((Q4_B1024) ? 4 : value_split_count) + blockIdx.x / 2;
            unsigned int empty1_work_stage = 0;
            unsigned int _phase_work_full_1 = 0;
            #pragma unroll 1
            for (unsigned int _work_iter = 0; _work_iter < ((Q4_B1024) ? 4096 : total_work_items); _work_iter++) {
                mbarrier_wait(work_full_addr + (empty1_work_stage) * 8, _phase_work_full_1);
                uint32_t _clc_valid_5 = 0;
                uint32_t _clc_ctaid_x_5;
                uint32_t _clc_ctaid_y_5;
                uint32_t _clc_ctaid_z_5;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%4];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %3, 1, 0, p1;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, %1, %2, _}, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_x_5), "=r"(_clc_ctaid_y_5), "=r"(_clc_ctaid_z_5), "=r"(_clc_valid_5)
                    : "r"(work_response_addr + empty1_work_stage * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + empty1_work_stage * 8), "r"(0) : "memory");
                empty1_work_stage += 1;
                if (empty1_work_stage == 2) { empty1_work_stage = 0; _phase_work_full_1 ^= 1; }
                unsigned int empty1_work_valid = _clc_valid_5;
                if (empty1_work_valid == 0) {
                    break;
                }
                empty1_work_idx = _clc_ctaid_z_5 * (unsigned int)(((Q4_B1024) ? 4 : value_split_count)) + _clc_ctaid_x_5 / 2;
            }
        }
    // ---- Role: softmax_wg ----
    } else if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 176;");
        { // softmax_wg_main
            const int wg_dummy_inc = 0;
            const int tmem_row_base = warp % 2 * 32;
            const int n_half = warp % 4 / 2;
            const int my_row = tmem_row_base + lane;
            const int exchange_idx = n_half * 64 + my_row;
            const int seed_row_base = warp % 4 * 32;
            int softmax_tile_cursor = 0;
            unsigned int softmax_work_idx = blockIdx.z * ((Q4_B1024) ? 4 : value_split_count) + blockIdx.x / 2;
            unsigned int softmax_work_stage = 0;
            float seed_zero[4];
            #pragma unroll
            for (int seed_i = 0; seed_i < 4; seed_i++) {
                seed_zero[seed_i] = 0.0f;
            }
            #pragma unroll
            for (int seed_col = 0; seed_col < 512; seed_col += 4) {
                int seed_addr = taddr + (unsigned int)seed_col + (unsigned int)(seed_row_base << 16);
                tmem_st_x4_f32(seed_addr, seed_zero);
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(tmem_scrubbed_addr);
            int seed_peer_rank = cta_rank ^ 1;
            asm volatile(
                "{\n\t"
                ".reg .b32 remAddr32;\n\t"
                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                "}"
                :: "r"(tmem_scrubbed_addr), "r"(seed_peer_rank) : "memory");
            unsigned int _phase_work_full_2 = 0;
            #pragma unroll 1
            for (unsigned int work_idx_1 = 0; work_idx_1 < ((Q4_B1024) ? 4096 : total_work_items); work_idx_1++) {
                unsigned int softmax_ordered_work_idx = softmax_work_idx;
                if (((Q4_B1024) ? 4096 : total_work_items) > 76) {
                    if (((Q4_B1024) ? 4096 : total_work_items) <= 1024) {
                        softmax_ordered_work_idx = (unsigned int)page_table[((Q4_B1024) ? 4096 : total_work_items) * ((Q4_B1024) ? 32 : max_pages_per_seq) + (int)softmax_work_idx];
                    }
                }
                int batch_idx_1 = softmax_ordered_work_idx;
                int seqlen_kv_b_1 = seq_lens_kv[batch_idx_1];
                int total_kv_tiles_1 = (seqlen_kv_b_1 + 128 - 1) / 128;
                int start_tile_1 = 0;
                int num_kv_tiles_1 = total_kv_tiles_1;
                float row_max_val = -MLA_INF;
                float row_sum_val = 0.0f;
                asm volatile("griddepcontrol.wait;" ::: "memory");
                asm volatile("griddepcontrol.wait;" ::: "memory");
                #pragma unroll 1
                for (int tile_1 = 0; tile_1 < num_kv_tiles_1; tile_1++) {
                    int pipeline_tile = softmax_tile_cursor + tile_1;
                    int phase = pipeline_tile & 1;
                    int s_wait_phase = pipeline_tile >> 1 & 1;
                    mbarrier_wait(s_full_addr + (phase) * 8, s_wait_phase);
                    int s_off = ((phase != 0) ? 64 : 0);
                    int s_base = taddr + (unsigned int)s_off + (unsigned int)(tmem_row_base << 16);
                    float sv[64];
                    float tile_max = -MLA_INF;
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                    #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                    #endif
                    asm volatile(
                        "tcgen05.ld.red.sync.aligned.32x32b.x64.max.f32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, %64, [%65];"
                        : "=f"(sv[0]), "=f"(sv[1]), "=f"(sv[2]), "=f"(sv[3]), "=f"(sv[4]), "=f"(sv[5]), "=f"(sv[6]), "=f"(sv[7]), "=f"(sv[8]), "=f"(sv[9]), "=f"(sv[10]), "=f"(sv[11]), "=f"(sv[12]), "=f"(sv[13]), "=f"(sv[14]), "=f"(sv[15]), "=f"(sv[16]), "=f"(sv[17]), "=f"(sv[18]), "=f"(sv[19]), "=f"(sv[20]), "=f"(sv[21]), "=f"(sv[22]), "=f"(sv[23]), "=f"(sv[24]), "=f"(sv[25]), "=f"(sv[26]), "=f"(sv[27]), "=f"(sv[28]), "=f"(sv[29]), "=f"(sv[30]), "=f"(sv[31]), "=f"(sv[32]), "=f"(sv[33]), "=f"(sv[34]), "=f"(sv[35]), "=f"(sv[36]), "=f"(sv[37]), "=f"(sv[38]), "=f"(sv[39]), "=f"(sv[40]), "=f"(sv[41]), "=f"(sv[42]), "=f"(sv[43]), "=f"(sv[44]), "=f"(sv[45]), "=f"(sv[46]), "=f"(sv[47]), "=f"(sv[48]), "=f"(sv[49]), "=f"(sv[50]), "=f"(sv[51]), "=f"(sv[52]), "=f"(sv[53]), "=f"(sv[54]), "=f"(sv[55]), "=f"(sv[56]), "=f"(sv[57]), "=f"(sv[58]), "=f"(sv[59]), "=f"(sv[60]), "=f"(sv[61]), "=f"(sv[62]), "=f"(sv[63]), "=f"(tile_max)
                        : "r"(s_base));
                    int abs_tile = start_tile_1 + tile_1;
                    bool score_tile_full = 0;
                    bool all_tiles_are_complete_k = seqlen_kv_b_1 % 128 == 0;
                    int tile_end_k = (abs_tile + 1) * 128;
                    bool is_full_tile_k = tile_end_k < seqlen_kv_b_1;
                    score_tile_full = all_tiles_are_complete_k || is_full_tile_k;
                    if (!score_tile_full) {
                        int tail_valid = seqlen_kv_b_1 - abs_tile * 128 - n_half * 64;
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
                    }
                    float new_max = tile_max;
                    if (!score_tile_full) {
                        float2 _reg_reduce_max2_2 = {-MLA_INF, -MLA_INF};
                        row_max_x32_accum(&sv[0], _reg_reduce_max2_2);
                        row_max_x32_accum(&sv[32], _reg_reduce_max2_2);
                        float sv_max = row_max_reduce(_reg_reduce_max2_2);
                        new_max = sv_max;
                    }
                    float _max_0 = max_noftz(new_max, row_max_val);
                    new_max = _max_0;
                    smem_softmax_exchange[exchange_idx] = new_max;
                    asm volatile("barrier.sync 2, 128;" ::: "memory");
                    float _max_1 = max_noftz(new_max, smem_softmax_exchange[exchange_idx ^ 64]);
                    asm volatile("barrier.sync 2, 128;" ::: "memory");
                    new_max = _max_1;
                    float exact_meta[4];
                    int exact_meta_addr = taddr + 384 + (unsigned int)(phase * 8) + (unsigned int)(n_half * 4) + (unsigned int)(tmem_row_base << 16);
                    mbarrier_wait(stats_empty_addr + (phase) * 8, (unsigned int)(pipeline_tile >> 1 & 1) ^ softmax_work_stage & 1 ^ 1);
                    exact_meta[0] = row_max_val;
                    exact_meta[1] = new_max;
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x2.b32"
                        " [%0], {%1, %2};"
                        :: "r"(exact_meta_addr), "f"(exact_meta[0]), "f"(exact_meta[1]));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(stats_addr + (phase) * 8);
                    float _tmem_load_0[1];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x1.b32"
                        " {%0}, [%1];"
                        : "=f"(_tmem_load_0[0])
                        : "r"(exact_meta_addr + 1));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    new_max = _tmem_load_0[0];
                    float no_correction = (((new_max - row_max_val) * softmax_scale_log2 <= 0.0f) ? 1.0f : 0.0f);
                    float delta = softmax_scale_log2 * (row_max_val - new_max);
                    float _exp2_0 = approx_exp2(delta);
                    float exp_delta = _exp2_0;
                    float acc_scale = ((row_max_val > -MLA_INF) ? exp_delta : 1.0f);
                    row_max_val = new_max;
                    float safe_max = ((new_max == -MLA_INF) ? 0.0f : new_max);
                    float max_scaled = safe_max * softmax_scale_log2;
                    float2 _f2_0 = make_float2(softmax_scale_log2, softmax_scale_log2);
                    float2 scale_pair = _f2_0;
                    float2 _f2_1 = make_float2(-max_scaled, -max_scaled);
                    float2 neg_max_pair = _f2_1;
                    #pragma unroll
                    for (int i = 0; i < 64; i += 2) {
                        float2 _f2_2 = make_float2(sv[i], sv[i + 1]);
                        float2 score_pair = _f2_2;
                        float2 affine_pair = fma_f32x2_rn_ftz(score_pair, scale_pair, neg_max_pair);
                        float _exp2_1 = approx_exp2(affine_pair.x);
                        sv[i] = _exp2_1;
                        float _exp2_2 = approx_exp2(affine_pair.y);
                        sv[i + 1] = _exp2_2;
                    }
                    float2 _reg_reduce_sum2_3 = make_float2(0.0f, 0.0f);
                    softmax_block_sum(&sv[0], &_reg_reduce_sum2_3);
                    softmax_block_sum(&sv[32], &_reg_reduce_sum2_3);
                    float sv_sum = _reg_reduce_sum2_3.x + _reg_reduce_sum2_3.y;
                    row_sum_val = row_sum_val * acc_scale + sv_sum;
                    int p_cor_empty_phase = pipeline_tile >> 1 & 1 ^ 1;
                    float meta[4];
                    int meta_addr = taddr + 384 + (unsigned int)(phase * 8) + (unsigned int)(n_half * 4) + (unsigned int)(tmem_row_base << 16);
                    int p_empty_phase = pipeline_tile >> 1 & 1 ^ 1;
                    mbarrier_wait(p_empty_addr + (phase) * 8, p_empty_phase);
                    const int p_row = my_row;
                    uint32_t sv_bf16[32];
                    #pragma unroll
                    for (int _lp = 0; _lp < 32; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(sv[_lp*2 + 0], sv[_lp*2+1 + 0]));
                        sv_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int ps_local = 0; ps_local < 2; ps_local++) {
                        int ps_2 = n_half * 2 + ps_local;
                        int p_stage = phase * 4 + ps_2;
                        int p_base_smem = smem_p_addr + (unsigned int)(p_stage * 4096);
                        #pragma unroll
                        for (int vec = 0; vec < 4; vec++) {
                            int pv_off = ps_local * 16 + vec * 4;
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p_base_smem + (p_row * 64 + vec * 16 ^ (p_row * 64 + vec * 16 >> 7 & 3) << 4))), "r"(__as_u32(sv_bf16[pv_off])), "r"(__as_u32(sv_bf16[pv_off + 1])), "r"(__as_u32(sv_bf16[pv_off + 2])), "r"(__as_u32(sv_bf16[pv_off + 3])) : "memory");
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
                mbarrier_wait(stats_empty_addr + (softmax_tile_cursor + num_kv_tiles_1 & 1) * 8, (unsigned int)(softmax_tile_cursor + num_kv_tiles_1 >> 1 & 1) ^ softmax_work_stage & 1 ^ 1);
                seed_zero[0] = row_sum_val;
                seed_zero[1] = row_max_val;
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x2.b32"
                    " [%0], {%1, %2};"
                    :: "r"(taddr + 384 + (unsigned int)((softmax_tile_cursor + num_kv_tiles_1 & 1) * 8) + (unsigned int)(n_half * 4) + (unsigned int)(tmem_row_base << 16)), "f"(seed_zero[0]), "f"(seed_zero[1]));
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                mbarrier_arrive(stats_addr + (softmax_tile_cursor + num_kv_tiles_1 & 1) * 8);
                mbarrier_wait(stats_empty_addr + (softmax_tile_cursor + num_kv_tiles_1 + 1 & 1) * 8, (unsigned int)(softmax_tile_cursor + num_kv_tiles_1 + 1 >> 1 & 1) ^ softmax_work_stage & 1 ^ 1);
                mbarrier_arrive(stats_addr + (softmax_tile_cursor + num_kv_tiles_1 + 1 & 1) * 8);
                softmax_tile_cursor = softmax_tile_cursor + num_kv_tiles_1;
                mbarrier_wait(work_full_addr + (softmax_work_stage) * 8, _phase_work_full_2);
                uint32_t _clc_valid_2 = 0;
                uint32_t _clc_ctaid_x_2;
                uint32_t _clc_ctaid_y_2;
                uint32_t _clc_ctaid_z_2;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%4];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %3, 1, 0, p1;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, %1, %2, _}, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_x_2), "=r"(_clc_ctaid_y_2), "=r"(_clc_ctaid_z_2), "=r"(_clc_valid_2)
                    : "r"(work_response_addr + softmax_work_stage * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + softmax_work_stage * 8), "r"(0) : "memory");
                softmax_work_stage += 1;
                if (softmax_work_stage == 2) { softmax_work_stage = 0; _phase_work_full_2 ^= 1; }
                unsigned int softmax_work_valid = _clc_valid_2;
                if (softmax_work_valid == 0) {
                    break;
                }
                softmax_work_idx = _clc_ctaid_z_2 * (unsigned int)(((Q4_B1024) ? 4 : value_split_count)) + _clc_ctaid_x_2 / 2;
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
        }
    // ---- Role: correction_wg ----
    } else if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 208;");
        { // correction_wg_main
            const int wg_dummy_inc_1 = 0;
            const int tmem_row_base_1 = warp % 2 * 32;
            const int n_half_1 = warp % 4 / 2;
            const int my_row_1 = tmem_row_base_1 + lane;
            const int corr_row = tmem_row_base_1 << 16;
            int correction_tile_cursor = 0;
            unsigned int correction_work_idx = blockIdx.z * ((Q4_B1024) ? 4 : value_split_count) + blockIdx.x / 2;
            unsigned int correction_work_stage = 0;
            unsigned int _phase_o_full_0 = 0;
            unsigned int _phase_work_full_3 = 0;
            #pragma unroll 1
            for (unsigned int work_idx_2 = 0; work_idx_2 < ((Q4_B1024) ? 4096 : total_work_items); work_idx_2++) {
                unsigned int correction_ordered_work_idx = correction_work_idx;
                if (((Q4_B1024) ? 4096 : total_work_items) > 76) {
                    if (((Q4_B1024) ? 4096 : total_work_items) <= 1024) {
                        correction_ordered_work_idx = (unsigned int)page_table[((Q4_B1024) ? 4096 : total_work_items) * ((Q4_B1024) ? 32 : max_pages_per_seq) + (int)correction_work_idx];
                    }
                }
                int batch_idx_2 = correction_ordered_work_idx;
                int value_split_1 = 0;
                int seqlen_kv_b_2 = seq_lens_kv[batch_idx_2];
                int total_kv_tiles_2 = (seqlen_kv_b_2 + 128 - 1) / 128;
                int start_tile_2 = 0;
                int num_kv_tiles_2 = total_kv_tiles_2;
                float final_sum_val = 0.0f;
                float final_max_val = -MLA_INF;
                asm volatile("griddepcontrol.wait;" ::: "memory");
                mbarrier_wait(stats_addr + (correction_tile_cursor & 1) * 8, (unsigned int)(correction_tile_cursor >> 1 & 1) ^ correction_work_stage & 1);
                mbarrier_arrive(stats_empty_addr + (correction_tile_cursor & 1) * 8);
                #pragma unroll 1
                for (int tile_2 = 1; tile_2 < num_kv_tiles_2; tile_2++) {
                    int pipeline_tile_1 = correction_tile_cursor + tile_2;
                    int phase_1 = pipeline_tile_1 & 1;
                    int stats_wait_phase = (unsigned int)(pipeline_tile_1 >> 1 & 1) ^ correction_work_stage & 1;
                    mbarrier_wait(stats_addr + (phase_1) * 8, stats_wait_phase);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int meta_addr_1 = taddr + 384 + (unsigned int)(phase_1 * 8) + (unsigned int)(n_half_1 * 4) + (unsigned int)corr_row;
                    float _tmem_load_1[2];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x2.b32"
                        " {%0, %1}, [%2];"
                        : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1])
                        : "r"(meta_addr_1));
                    final_sum_val = final_sum_val;
                    final_max_val = final_max_val;
                    float _exp2_3 = approx_exp2(softmax_scale_log2 * (_tmem_load_1[0] - _tmem_load_1[1]));
                    float acc_scale_1 = _exp2_3;
                    float no_correction_1 = 0.0f;
                    {
                        mbarrier_wait(o_full_addr, _phase_o_full_0);
                        _phase_o_full_0 ^= 1;
                        asm volatile("tcgen05.fence::after_thread_sync;");
                    }
                    {
                        int _vote_0 = __all_sync(0xFFFFFFFF, _tmem_load_1[1] <= _tmem_load_1[0]);
                        int skip_correction = _vote_0;
                        if (skip_correction == 0) {
                            {
                                #pragma unroll
                                for (int acc_stage_2 = 0; acc_stage_2 < 2; acc_stage_2++) {
                                    #pragma unroll
                                    for (int vs_local = 0; vs_local < 2; vs_local++) {
                                        int o_base = taddr + 128 + (unsigned int)(acc_stage_2 * 128) + (unsigned int)(vs_local * 64) + (unsigned int)corr_row;
                                        #pragma unroll
                                        for (int c = 0; c < 64; c += 16) {
                                            float _tmem_load_3[16];
                                            tmem_ld_x16(&_tmem_load_3[0], o_base + c);
                                            const float2 _scale2_0 = {acc_scale_1, acc_scale_1};
                                            #pragma unroll
                                            for (int _ls = 0; _ls < 8; _ls++)
                                                mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_3)[_ls], _scale2_0);
                                            tmem_st_x16_f32(o_base + c, _tmem_load_3);
                                        }
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
                mbarrier_wait(stats_addr + (correction_tile_cursor + num_kv_tiles_2 & 1) * 8, (unsigned int)(correction_tile_cursor + num_kv_tiles_2 >> 1 & 1) ^ correction_work_stage & 1);
                asm volatile("tcgen05.fence::after_thread_sync;");
                float _tmem_load_5[2];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x2.b32"
                    " {%0, %1}, [%2];"
                    : "=f"(_tmem_load_5[0]), "=f"(_tmem_load_5[1])
                    : "r"(taddr + 384 + (unsigned int)((correction_tile_cursor + num_kv_tiles_2 & 1) * 8) + (unsigned int)(n_half_1 * 4) + (unsigned int)corr_row));
                mbarrier_arrive(stats_empty_addr + (correction_tile_cursor + num_kv_tiles_2 & 1) * 8);
                mbarrier_wait(stats_addr + (correction_tile_cursor + num_kv_tiles_2 + 1 & 1) * 8, (unsigned int)(correction_tile_cursor + num_kv_tiles_2 + 1 >> 1 & 1) ^ correction_work_stage & 1);
                mbarrier_arrive(stats_empty_addr + (correction_tile_cursor + num_kv_tiles_2 + 1 & 1) * 8);
                final_sum_val = _tmem_load_5[0];
                final_max_val = _tmem_load_5[1];
                mbarrier_wait(o_full_addr, _phase_o_full_0);
                _phase_o_full_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                const int epilogue_exchange_idx = n_half_1 * 64 + my_row_1;
                smem_epilogue_exchange[epilogue_exchange_idx] = final_sum_val;
                asm volatile("barrier.sync 3, 128;" ::: "memory");
                final_sum_val = final_sum_val + smem_epilogue_exchange[epilogue_exchange_idx ^ 64];
                asm volatile("barrier.sync 3, 128;" ::: "memory");
                float _rcp_0 = approx_rcp(final_sum_val);
                float inv_sum = _rcp_0;
                int head_idx = cta_rank * 64 + my_row_1;
                float _log2_0;
                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(final_sum_val));
                float lse_log2 = final_max_val * softmax_scale_log2 + _log2_0;
                int o_offset = (batch_idx_2 * 128 + head_idx) * 512;
                float output_scale = 1.0f;
                {
                    #pragma unroll 1
                    for (int acc_stage_3 = 0; acc_stage_3 < 2; acc_stage_3++) {
                        #pragma unroll 1
                        for (int vs_local_1 = 0; vs_local_1 < 2; vs_local_1++) {
                            int logical_vs = acc_stage_3 * 2 * 2 + n_half_1 * 2 + vs_local_1;
                            int o_base_epi = taddr + 128 + (unsigned int)(acc_stage_3 * 128) + (unsigned int)(vs_local_1 * 64) + (unsigned int)corr_row;
                            {
                                #pragma unroll
                                for (int c_1 = 0; c_1 < 64; c_1 += 64) {
                                    float _tmem_load_7[32];
                                    tmem_ld_x8(&_tmem_load_7[0], o_base_epi + c_1);
                                    tmem_ld_x8(&_tmem_load_7[8], o_base_epi + c_1 + 8);
                                    tmem_ld_x8(&_tmem_load_7[16], o_base_epi + c_1 + 16);
                                    tmem_ld_x8(&_tmem_load_7[24], o_base_epi + c_1 + 24);
                                    float _tmem_load_8[32];
                                    tmem_ld_x8(&_tmem_load_8[0], o_base_epi + c_1 + 32);
                                    tmem_ld_x8(&_tmem_load_8[8], o_base_epi + c_1 + 32 + 8);
                                    tmem_ld_x8(&_tmem_load_8[16], o_base_epi + c_1 + 32 + 16);
                                    tmem_ld_x8(&_tmem_load_8[24], o_base_epi + c_1 + 32 + 24);
                                    unsigned int epilogue_packed_first[16];
                                    unsigned int epilogue_packed_next[16];
                                    const float2 _scale2_1 = {inv_sum * output_scale, inv_sum * output_scale};
                                    #pragma unroll
                                    for (int _ls = 0; _ls < 16; _ls++)
                                        mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_7)[_ls], _scale2_1);
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 16; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_7[_lp*2 + 0], _tmem_load_7[_lp*2+1 + 0]));
                                        epilogue_packed_first[_lp] = *(uint32_t*)&_bf2;
                                    }
                                    const float2 _scale2_2 = {inv_sum * output_scale, inv_sum * output_scale};
                                    #pragma unroll
                                    for (int _ls = 0; _ls < 16; _ls++)
                                        mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_8)[_ls], _scale2_2);
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 16; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_8[_lp*2 + 0], _tmem_load_8[_lp*2+1 + 0]));
                                        epilogue_packed_next[_lp] = *(uint32_t*)&_bf2;
                                    }
                                    int gmem_base = o_offset + logical_vs * 64 + c_1;
                                    if (((unsigned long long)O & 31) == 0) {
                                        #pragma unroll
                                        for (int j = 0; j < 32; j += 16) {
                                            {
                                                unsigned _stv8_3_0 = __float_as_uint(reinterpret_cast<float*>(epilogue_packed_first)[j / 2 + 0]);
                                                unsigned _stv8_3_1 = __float_as_uint(reinterpret_cast<float*>(epilogue_packed_first)[j / 2 + 1]);
                                                unsigned _stv8_3_2 = __float_as_uint(reinterpret_cast<float*>(epilogue_packed_first)[j / 2 + 2]);
                                                unsigned _stv8_3_3 = __float_as_uint(reinterpret_cast<float*>(epilogue_packed_first)[j / 2 + 3]);
                                                unsigned _stv8_3_4 = __float_as_uint(reinterpret_cast<float*>(epilogue_packed_first)[j / 2 + 4]);
                                                unsigned _stv8_3_5 = __float_as_uint(reinterpret_cast<float*>(epilogue_packed_first)[j / 2 + 5]);
                                                unsigned _stv8_3_6 = __float_as_uint(reinterpret_cast<float*>(epilogue_packed_first)[j / 2 + 6]);
                                                unsigned _stv8_3_7 = __float_as_uint(reinterpret_cast<float*>(epilogue_packed_first)[j / 2 + 7]);
                                                asm volatile(
                                                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                                    :: "l"((void*)(O + (gmem_base + j) + (0))), "r"(_stv8_3_0), "r"(_stv8_3_1), "r"(_stv8_3_2), "r"(_stv8_3_3), "r"(_stv8_3_4), "r"(_stv8_3_5), "r"(_stv8_3_6), "r"(_stv8_3_7) : "memory");
                                            }
                                        }
                                        #pragma unroll
                                        for (int j_1 = 0; j_1 < 32; j_1 += 16) {
                                            {
                                                unsigned _stv8_4_0 = __float_as_uint(reinterpret_cast<float*>(epilogue_packed_next)[j_1 / 2 + 0]);
                                                unsigned _stv8_4_1 = __float_as_uint(reinterpret_cast<float*>(epilogue_packed_next)[j_1 / 2 + 1]);
                                                unsigned _stv8_4_2 = __float_as_uint(reinterpret_cast<float*>(epilogue_packed_next)[j_1 / 2 + 2]);
                                                unsigned _stv8_4_3 = __float_as_uint(reinterpret_cast<float*>(epilogue_packed_next)[j_1 / 2 + 3]);
                                                unsigned _stv8_4_4 = __float_as_uint(reinterpret_cast<float*>(epilogue_packed_next)[j_1 / 2 + 4]);
                                                unsigned _stv8_4_5 = __float_as_uint(reinterpret_cast<float*>(epilogue_packed_next)[j_1 / 2 + 5]);
                                                unsigned _stv8_4_6 = __float_as_uint(reinterpret_cast<float*>(epilogue_packed_next)[j_1 / 2 + 6]);
                                                unsigned _stv8_4_7 = __float_as_uint(reinterpret_cast<float*>(epilogue_packed_next)[j_1 / 2 + 7]);
                                                asm volatile(
                                                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                                    :: "l"((void*)(O + (gmem_base + 32 + j_1) + (0))), "r"(_stv8_4_0), "r"(_stv8_4_1), "r"(_stv8_4_2), "r"(_stv8_4_3), "r"(_stv8_4_4), "r"(_stv8_4_5), "r"(_stv8_4_6), "r"(_stv8_4_7) : "memory");
                                            }
                                        }
                                    } else {
                                        #pragma unroll
                                        for (int j_2 = 0; j_2 < 32; j_2 += 8) {
                                            reinterpret_cast<int4*>(O + (gmem_base + j_2))[0] = reinterpret_cast<int4*>(epilogue_packed_first + j_2 / 2)[0];
                                        }
                                        #pragma unroll
                                        for (int j_3 = 0; j_3 < 32; j_3 += 8) {
                                            reinterpret_cast<int4*>(O + (gmem_base + 32 + j_3))[0] = reinterpret_cast<int4*>(epilogue_packed_next + j_3 / 2)[0];
                                        }
                                    }
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
                correction_tile_cursor = correction_tile_cursor + num_kv_tiles_2;
                mbarrier_wait(work_full_addr + (correction_work_stage) * 8, _phase_work_full_3);
                uint32_t _clc_valid_3 = 0;
                uint32_t _clc_ctaid_x_3;
                uint32_t _clc_ctaid_y_3;
                uint32_t _clc_ctaid_z_3;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%4];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %3, 1, 0, p1;\n\t"
                    "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, %1, %2, _}, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_x_3), "=r"(_clc_ctaid_y_3), "=r"(_clc_ctaid_z_3), "=r"(_clc_valid_3)
                    : "r"(work_response_addr + correction_work_stage * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + correction_work_stage * 8), "r"(0) : "memory");
                correction_work_stage += 1;
                if (correction_work_stage == 2) { correction_work_stage = 0; _phase_work_full_3 ^= 1; }
                unsigned int correction_work_valid = _clc_valid_3;
                if (correction_work_valid == 0) {
                    break;
                }
                correction_work_idx = _clc_ctaid_z_3 * (unsigned int)(((Q4_B1024) ? 4 : value_split_count)) + _clc_ctaid_x_3 / 2;
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
        }
    // ---- Role: mma_warp ----
    } else if (warp == 8) {
        { // mma_warp_main
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&tmap_q))) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&tmap_k))) : "memory");
            asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&tmap_v))) : "memory");
            const int wg2_dummy_2 = 0;
            unsigned int mma_kv_stage = 0;
            unsigned int mma_kv_phase = 0;
            unsigned int mma_kv_token = 0;
            int mma_tile_cursor = 0;
            unsigned int mma_work_idx = blockIdx.z * ((Q4_B1024) ? 4 : value_split_count) + blockIdx.x / 2;
            unsigned int mma_work_stage = 0;
            unsigned int _phase_tmem_scrubbed_0 = 0;
            mbarrier_wait(tmem_scrubbed_addr, _phase_tmem_scrubbed_0);
            _phase_tmem_scrubbed_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            unsigned int _phase_q_full_0 = 0;
            unsigned int _phase_kv_full = 0;
            unsigned int _phase_o_empty_0 = 1;
            unsigned int _phase_work_full_4 = 0;
            #pragma unroll 1
            for (unsigned int work_idx_3 = 0; work_idx_3 < ((Q4_B1024) ? 4096 : total_work_items); work_idx_3++) {
                unsigned int mma_ordered_work_idx = mma_work_idx;
                if (((Q4_B1024) ? 4096 : total_work_items) > 76) {
                    if (((Q4_B1024) ? 4096 : total_work_items) <= 1024) {
                        mma_ordered_work_idx = (unsigned int)page_table[((Q4_B1024) ? 4096 : total_work_items) * ((Q4_B1024) ? 32 : max_pages_per_seq) + (int)mma_work_idx];
                    }
                }
                if (cta_rank != 0) {
                    break;
                }
                int batch_idx_3 = mma_ordered_work_idx;
                int seqlen_kv_b_3 = seq_lens_kv[batch_idx_3];
                int total_kv_tiles_3 = (seqlen_kv_b_3 + 128 - 1) / 128;
                int start_tile_3 = 0;
                int num_kv_tiles_3 = total_kv_tiles_3;
                if (cta_rank == 0) {
                    mbarrier_wait(q_full_addr, _phase_q_full_0);
                    _phase_q_full_0 ^= 1;
                    int first_pv = 1;
                    #pragma unroll 1
                    for (int tile_3 = 0; tile_3 < num_kv_tiles_3; tile_3++) {
                        int pipeline_tile_2 = mma_tile_cursor + tile_3;
                        int phase_2 = pipeline_tile_2 & 1;
                        int s_empty_phase = pipeline_tile_2 >> 1 & 1 ^ 1;
                        mbarrier_wait(s_empty_addr + (phase_2) * 8, s_empty_phase);
                        int score_col = ((phase_2 != 0) ? 64 : 0);
                        #pragma unroll
                        for (int n_2 = 0; n_2 < 8; n_2++) {
                            {
                                uint32_t _mbar_token_6 = mbarrier_try_wait(kv_full_addr + (mma_kv_stage) * 8, mma_kv_phase);
                                mma_kv_token = _mbar_token_6;
                                mbarrier_wait_token(kv_full_addr + (mma_kv_stage) * 8, mma_kv_phase, mma_kv_token);
                                int _mma_a_lo_0 = (((smem_q_addr) >> 4) & 0x3FFF) + (n_2) * 512;
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
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_tmem_scratch + (score_col))), "r"(((n_2 == 0) ? 0 : 1)));
                            }
                            elect_commit_cg2_multicast(kv_empty_addr + (mma_kv_stage) * 8, (uint16_t)(3));
                            {
                                mma_kv_stage += 1;
                                if (mma_kv_stage == 15) { mma_kv_stage = 0; mma_kv_phase ^= 1; }
                            }
                        }
                        {
                            uint32_t _mbar_token_7 = mbarrier_try_wait(kv_full_addr + (mma_kv_stage) * 8, mma_kv_phase);
                            mma_kv_token = _mbar_token_7;
                            mbarrier_wait_token(kv_full_addr + (mma_kv_stage) * 8, mma_kv_phase, mma_kv_token);
                            int _mma_a_lo_2 = (((smem_q_addr) >> 4) & 0x3FFF) + (8) * 512;
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
                    :: "r"(_mma_a_lo_2), "r"(_mma_b_lo_2), "r"((tmem_tmem_scratch + (score_col))), "r"(1));
                        }
                        elect_commit_cg2_multicast(kv_empty_addr + (mma_kv_stage) * 8, (uint16_t)(3));
                        elect_commit_cg2_multicast(s_full_addr + (phase_2) * 8, (uint16_t)(3));
                        {
                            mma_kv_stage += 1;
                            if (mma_kv_stage == 15) { mma_kv_stage = 0; mma_kv_phase ^= 1; }
                        }
                        if (tile_3 > 0) {
                            int prev_pipeline_tile = pipeline_tile_2 - 1;
                            int prev_phase = prev_pipeline_tile & 1;
                            int pv_wait_phase = prev_pipeline_tile >> 1 & 1;
                            mbarrier_wait(o_empty_addr, _phase_o_empty_0);
                            _phase_o_empty_0 ^= 1;
                            mbarrier_wait(p_full_addr + (prev_phase) * 8, pv_wait_phase);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            #pragma unroll
                            for (int ps_3 = 0; ps_3 < 4; ps_3++) {
                                int p_stage_1 = prev_phase * 4 + ps_3;
                                #pragma unroll
                                for (int acc_stage_4 = 0; acc_stage_4 < ((1) ? 2 : 1); acc_stage_4++) {
                                    {
                                        uint32_t _mbar_token_8 = mbarrier_try_wait(kv_full_addr + (mma_kv_stage) * 8, mma_kv_phase);
                                        mma_kv_token = _mbar_token_8;
                                        int output_col = 128 + acc_stage_4 * 128;
                                        mbarrier_wait_token(kv_full_addr + (mma_kv_stage) * 8, mma_kv_phase, mma_kv_token);
                                        int _mma_a_lo_4 = (((smem_p_addr) >> 4) & 0x3FFF) + (p_stage_1) * 256;
                                        int _mma_b_lo_4 = ((((smem_vhalf_addr) >> 4) & 0x3FFF) | 0x1000000) + (mma_kv_stage) * 512;
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
                    :: "r"(_mma_a_lo_4), "r"(_mma_b_lo_4), "r"((tmem_tmem_scratch + (output_col))), "r"(((((ps_3 == 0) ? first_pv : 0)) ? 0 : 1)));
                                    }
                                    elect_commit_cg2_multicast(kv_empty_addr + (mma_kv_stage) * 8, (uint16_t)(3));
                                    {
                                        mma_kv_stage += 1;
                                        if (mma_kv_stage == 15) { mma_kv_stage = 0; mma_kv_phase ^= 1; }
                                    }
                                }
                            }
                            first_pv = 0;
                            elect_commit_cg2_multicast(p_empty_addr + (prev_phase) * 8, (uint16_t)(3));
                            elect_commit_cg2_multicast(o_full_addr, (uint16_t)(3));
                        }
                    }
                    int last_pipeline_tile = mma_tile_cursor + num_kv_tiles_3 - 1;
                    int last_phase = last_pipeline_tile & 1;
                    int drain_wait_phase = last_pipeline_tile >> 1 & 1;
                    mbarrier_wait(o_empty_addr, _phase_o_empty_0);
                    _phase_o_empty_0 ^= 1;
                    mbarrier_wait(p_full_addr + (last_phase) * 8, drain_wait_phase);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    #pragma unroll
                    for (int ps_4 = 0; ps_4 < 4; ps_4++) {
                        int p_stage_last = last_phase * 4 + ps_4;
                        #pragma unroll
                        for (int acc_stage_5 = 0; acc_stage_5 < ((1) ? 2 : 1); acc_stage_5++) {
                            {
                                uint32_t _mbar_token_9 = mbarrier_try_wait(kv_full_addr + (mma_kv_stage) * 8, mma_kv_phase);
                                mma_kv_token = _mbar_token_9;
                                int output_col_d = 128 + acc_stage_5 * 128;
                                mbarrier_wait_token(kv_full_addr + (mma_kv_stage) * 8, mma_kv_phase, mma_kv_token);
                                int _mma_a_lo_7 = (((smem_p_addr) >> 4) & 0x3FFF) + (p_stage_last) * 256;
                                int _mma_b_lo_7 = ((((smem_vhalf_addr) >> 4) & 0x3FFF) | 0x1000000) + (mma_kv_stage) * 512;
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
                    :: "r"(_mma_a_lo_7), "r"(_mma_b_lo_7), "r"((tmem_tmem_scratch + (output_col_d))), "r"(((((ps_4 == 0) ? first_pv : 0)) ? 0 : 1)));
                            }
                            elect_commit_cg2_multicast(kv_empty_addr + (mma_kv_stage) * 8, (uint16_t)(3));
                            {
                                mma_kv_stage += 1;
                                if (mma_kv_stage == 15) { mma_kv_stage = 0; mma_kv_phase ^= 1; }
                            }
                        }
                    }
                    elect_commit_cg2_multicast(p_empty_addr + (last_phase) * 8, (uint16_t)(3));
                    elect_commit_cg2_multicast(o_full_addr, (uint16_t)(3));
                    elect_commit_cg2_multicast(q_empty_addr, (uint16_t)(3));
                    mma_tile_cursor = mma_tile_cursor + num_kv_tiles_3;
                    mbarrier_wait(work_full_addr + (mma_work_stage) * 8, _phase_work_full_4);
                    uint32_t _clc_valid_1 = 0;
                    uint32_t _clc_ctaid_x_1;
                    uint32_t _clc_ctaid_y_1;
                    uint32_t _clc_ctaid_z_1;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%4];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %3, 1, 0, p1;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, %1, %2, _}, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_x_1), "=r"(_clc_ctaid_y_1), "=r"(_clc_ctaid_z_1), "=r"(_clc_valid_1)
                        : "r"(work_response_addr + mma_work_stage * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(work_empty_addr + mma_work_stage * 8), "r"(0) : "memory");
                    mma_work_stage += 1;
                    if (mma_work_stage == 2) { mma_work_stage = 0; _phase_work_full_4 ^= 1; }
                    unsigned int mma_work_valid = _clc_valid_1;
                    if (mma_work_valid == 0) {
                        break;
                    }
                    mma_work_idx = _clc_ctaid_z_1 * (unsigned int)(((Q4_B1024) ? 4 : value_split_count)) + _clc_ctaid_x_1 / 2;
                }
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
    // ---- Role: empty0 ----
    } else if (warp == 10) {
        { // empty0_main
            const int wg2_dummy_3 = 0;
            unsigned int _phase_throttle_full = 0;
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_work_full_5 = 0;
            if (cta_rank == 0) {
                unsigned int scheduler_work_idx = blockIdx.z * ((Q4_B1024) ? 4 : value_split_count) + blockIdx.x / 2;
                unsigned int scheduler_prod_stage = 0;
                unsigned int scheduler_cons_stage = 0;
                unsigned int scheduler_throttle_stage = 0;
                #pragma unroll 1
                for (unsigned int _work_iter_1 = 0; _work_iter_1 < ((Q4_B1024) ? 4096 : total_work_items); _work_iter_1++) {
                    mbarrier_wait(throttle_full_addr + (scheduler_throttle_stage) * 8, _phase_throttle_full);
                    mbarrier_arrive(throttle_empty_addr + (scheduler_throttle_stage) * 8);
                    scheduler_throttle_stage += 1;
                    if (scheduler_throttle_stage == 2) { scheduler_throttle_stage = 0; _phase_throttle_full ^= 1; }
                    mbarrier_wait(work_empty_addr + (scheduler_prod_stage) * 8, _phase_work_empty);
                    if (lane < 2) {
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                            "}"
                            :: "r"(work_full_addr + scheduler_prod_stage * 8), "r"(lane), "r"((uint32_t)(16)) : "memory");
                    }
                    if (elect_sync()) {
                        asm volatile(
                            "fence.proxy.async.shared::cta;\n\t"
                            "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                                " [%0], [%1];"
                            :: "r"(work_response_addr + scheduler_prod_stage * 16 + 0 * 16), "r"(work_full_addr + scheduler_prod_stage * 8)
                            : "memory");
                    }
                    scheduler_prod_stage += 1;
                    if (scheduler_prod_stage == 2) { scheduler_prod_stage = 0; _phase_work_empty ^= 1; }
                    mbarrier_wait(work_full_addr + (scheduler_cons_stage) * 8, _phase_work_full_5);
                    uint32_t _clc_valid_4 = 0;
                    uint32_t _clc_ctaid_x_4;
                    uint32_t _clc_ctaid_y_4;
                    uint32_t _clc_ctaid_z_4;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%4];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %3, 1, 0, p1;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {%0, %1, %2, _}, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_x_4), "=r"(_clc_ctaid_y_4), "=r"(_clc_ctaid_z_4), "=r"(_clc_valid_4)
                        : "r"(work_response_addr + scheduler_cons_stage * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(work_empty_addr + scheduler_cons_stage * 8), "r"(0) : "memory");
                    scheduler_cons_stage += 1;
                    if (scheduler_cons_stage == 2) { scheduler_cons_stage = 0; _phase_work_full_5 ^= 1; }
                    unsigned int scheduler_work_valid = _clc_valid_4;
                    if (scheduler_work_valid == 0) {
                        break;
                    }
                    scheduler_work_idx = _clc_ctaid_z_4 * (unsigned int)(((Q4_B1024) ? 4 : value_split_count)) + _clc_ctaid_x_4 / 2;
                }
                #pragma unroll
                for (int _tail = 0; _tail < 2; _tail++) {
                    mbarrier_wait(work_empty_addr + (scheduler_prod_stage) * 8, _phase_work_empty);
                    scheduler_prod_stage += 1;
                    if (scheduler_prod_stage == 2) { scheduler_prod_stage = 0; _phase_work_empty ^= 1; }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"

