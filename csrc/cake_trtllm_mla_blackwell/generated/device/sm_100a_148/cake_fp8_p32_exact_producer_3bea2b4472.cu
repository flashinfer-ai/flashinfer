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

#define MLA_INF CUDART_INF_F
#define TMEM_NCOLS 512
#define TMEM_TMEM_OFFSET 0
#define NUM_Q_PIPE_STAGES 2
#define NUM_KV_PIPE_STAGES 9
#define NUM_PAGE_PIPE_STAGES 6
#define NUM_ORDER_PIPE_STAGES 1
#define SMEM_SMEM_Q0_OFF 0
#define SMEM_SMEM_Q0_STAGE_BYTES 2048
#define SMEM_SMEM_Q0_STRIDE 10240
#define SMEM_SMEM_Q1_OFF 2048
#define SMEM_SMEM_Q1_STAGE_BYTES 2048
#define SMEM_SMEM_Q1_STRIDE 10240
#define SMEM_SMEM_Q2_OFF 4096
#define SMEM_SMEM_Q2_STAGE_BYTES 2048
#define SMEM_SMEM_Q2_STRIDE 10240
#define SMEM_SMEM_Q3_OFF 6144
#define SMEM_SMEM_Q3_STAGE_BYTES 2048
#define SMEM_SMEM_Q3_STRIDE 10240
#define SMEM_SMEM_Q4_OFF 8192
#define SMEM_SMEM_Q4_STAGE_BYTES 2048
#define SMEM_SMEM_Q4_STRIDE 10240
#define SMEM_SMEM_KV_OFF 20480
#define SMEM_SMEM_KV_STAGE_BYTES 16384
#define SMEM_SMEM_KV_STRIDE 16384
#define SMEM_SMEM_V_OFF 20480
#define SMEM_SMEM_V_STAGE_BYTES 16384
#define SMEM_SMEM_V_STRIDE 16384
#define SMEM_SMEM_P0_OFF 167936
#define SMEM_SMEM_P0_STAGE_BYTES 2048
#define SMEM_SMEM_P0_STRIDE 2048
#define SMEM_SMEM_P1_OFF 169984
#define SMEM_SMEM_P1_STAGE_BYTES 2048
#define SMEM_SMEM_P1_STRIDE 2048
#define SMEM_PAGE_OFFSETS_OFF 172032
#define SMEM_PAGE_OFFSETS_STAGE_BYTES 128
#define SMEM_PAGE_OFFSETS_STRIDE 128
#define SMEM_EXCHANGE0_OFF 172800
#define SMEM_EXCHANGE0_STAGE_BYTES 256
#define SMEM_EXCHANGE0_STRIDE 256
#define SMEM_EXCHANGE1_OFF 173056
#define SMEM_EXCHANGE1_STAGE_BYTES 256
#define SMEM_EXCHANGE1_STRIDE 256
#define SMEM_EXCHANGE0_U32_OFF 172800
#define SMEM_EXCHANGE0_U32_STAGE_BYTES 256
#define SMEM_EXCHANGE0_U32_STRIDE 256
#define SMEM_EXCHANGE1_U32_OFF 173056
#define SMEM_EXCHANGE1_U32_STAGE_BYTES 256
#define SMEM_EXCHANGE1_U32_STRIDE 256
#define SMEM_CORR_STATS_REDUCE_OFF 173312
#define SMEM_CORR_STATS_REDUCE_STAGE_BYTES 256
#define SMEM_CORR_STATS_REDUCE_STRIDE 256
#define SMEM_REDUCTION_TICKET_OFF 173696
#define SMEM_REDUCTION_TICKET_STAGE_BYTES 4
#define SMEM_REDUCTION_TICKET_STRIDE 4
#define SMEM_SMEM_O_OFF 173824
#define SMEM_SMEM_O_STAGE_BYTES 4096
#define SMEM_SMEM_O_STRIDE 4096
#define SMEM_TOTAL 178560
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


__device__ __forceinline__ void tcgen05_mma_f8f6f4(
    int taddr, uint64_t a_desc, uint64_t b_desc,
    uint32_t i_desc, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, p;\n\t"
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


__device__ __forceinline__ void tmem_st_x8_f32(int tmem_addr, const float* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x8.b32"
        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
        :: "r"(tmem_addr),
           "f"(src[0]), "f"(src[1]), "f"(src[2]), "f"(src[3]),
           "f"(src[4]), "f"(src[5]), "f"(src[6]), "f"(src[7]));
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


__device__ __forceinline__ void elect_commit2(int mbar_addr0, int mbar_addr1) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];\n\t"
        "@leader tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%1];\n\t"
        "}\n"
        :: "r"(mbar_addr0), "r"(mbar_addr1) : "memory");
}


__device__ __forceinline__ void tcgen05_commit2(int mbar_addr0, int mbar_addr1) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];\n\t"
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%1];\n\t"
        :: "r"(mbar_addr0), "r"(mbar_addr1) : "memory");
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


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tmem_ld_x8(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x8.b32"
        " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3]),
          "=f"(dst[4]), "=f"(dst[5]), "=f"(dst[6]), "=f"(dst[7])
        : "r"(tmem_addr));
}


__device__ __forceinline__ void tmem_ld_x8_wait(float* dst, int addr) {
    tmem_ld_x8(dst, addr);
    asm volatile("tcgen05.wait::ld.sync.aligned;");
}


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(512, 1) void
kernel_fp8_p32_exact_producer(const __grid_constant__ CUtensorMap Q_map, const __grid_constant__ CUtensorMap K_map, const __grid_constant__ CUtensorMap V_map, int* __restrict__ page_table, int* __restrict__ seq_lens, __nv_bfloat16* __restrict__ partial_O, float* __restrict__ partial_stats, __nv_bfloat16* __restrict__ O, unsigned int* __restrict__ completion, int q_len, int page_table_stride, int max_num_ctas_q, int max_num_ctas_kv, float bmm1_scale_log2, float bmm2_scale)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);

    const int mbar_base = smem + 177920;
    #define q_full_addr (mbar_base + 0)
    #define q_empty_addr (mbar_base + 16)
    #define kv_full_addr (mbar_base + 32)
    #define kv_empty_addr (mbar_base + 104)
    #define s_full0_addr (mbar_base + 176)
    #define s_full1_addr (mbar_base + 184)
    #define _source_p_empty0_addr (mbar_base + 192)
    #define _source_p_empty1_addr (mbar_base + 200)
    #define o_full0_addr (mbar_base + 208)
    #define o_full1_addr (mbar_base + 216)
    #define page_full_addr (mbar_base + 224)
    #define page_empty_addr (mbar_base + 272)
    #define s_empty0_addr (mbar_base + 320)
    #define scale_ready0_addr (mbar_base + 328)
    #define stats_empty0_addr (mbar_base + 336)
    #define s_empty1_addr (mbar_base + 344)
    #define scale_ready1_addr (mbar_base + 352)
    #define stats_empty1_addr (mbar_base + 360)
    #define order_p01_0_addr (mbar_base + 368)
    #define order_p01_1_addr (mbar_base + 376)
    #define _source_p_ready0_addr (mbar_base + 384)
    #define _source_p_ready1_addr (mbar_base + 392)
    #define o_empty0_addr (mbar_base + 400)
    #define o_empty1_addr (mbar_base + 408)
    #define _source_barrier_reservation_addr (mbar_base + 416)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    uint8_t* smem_q0 = reinterpret_cast<uint8_t*>(smem_raw + 0);
    const int smem_q0_addr = smem + 0;
    uint8_t* smem_q1 = reinterpret_cast<uint8_t*>(smem_raw + 2048);
    const int smem_q1_addr = smem + 2048;
    uint8_t* smem_q2 = reinterpret_cast<uint8_t*>(smem_raw + 4096);
    const int smem_q2_addr = smem + 4096;
    uint8_t* smem_q3 = reinterpret_cast<uint8_t*>(smem_raw + 6144);
    const int smem_q3_addr = smem + 6144;
    uint8_t* smem_q4 = reinterpret_cast<uint8_t*>(smem_raw + 8192);
    const int smem_q4_addr = smem + 8192;
    uint8_t* smem_kv = reinterpret_cast<uint8_t*>(smem_raw + 20480);
    const int smem_kv_addr = smem + 20480;
    uint8_t* smem_v = reinterpret_cast<uint8_t*>(smem_raw + 20480);
    const int smem_v_addr = smem + 20480;
    uint8_t* smem_p0 = reinterpret_cast<uint8_t*>(smem_raw + 167936);
    const int smem_p0_addr = smem + 167936;
    uint8_t* smem_p1 = reinterpret_cast<uint8_t*>(smem_raw + 169984);
    const int smem_p1_addr = smem + 169984;
    int* page_offsets = reinterpret_cast<int*>(smem_raw + 172032);
    const int page_offsets_addr = smem + 172032;
    float* exchange0 = reinterpret_cast<float*>(smem_raw + 172800);
    const int exchange0_addr = smem + 172800;
    float* exchange1 = reinterpret_cast<float*>(smem_raw + 173056);
    const int exchange1_addr = smem + 173056;
    unsigned int* exchange0_u32 = reinterpret_cast<unsigned int*>(smem_raw + 172800);
    const int exchange0_u32_addr = smem + 172800;
    unsigned int* exchange1_u32 = reinterpret_cast<unsigned int*>(smem_raw + 173056);
    const int exchange1_u32_addr = smem + 173056;
    float* corr_stats_reduce = reinterpret_cast<float*>(smem_raw + 173312);
    const int corr_stats_reduce_addr = smem + 173312;
    int* reduction_ticket = reinterpret_cast<int*>(smem_raw + 173696);
    const int reduction_ticket_addr = smem + 173696;
    uint8_t* smem_o = reinterpret_cast<uint8_t*>(smem_raw + 173824);
    const int smem_o_addr = smem + 173824;
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&Q_map))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&K_map))) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)((&V_map))) : "memory");

    // Mbarrier init (25 pipeline groups, 0 ordered-sequence groups, 64 barriers)
    // Mbarriers at smem_raw[177920..178432)

    if (warp == 0) {
        // --- pipeline 'q_pipe' ---
        // q_full: 2 barriers, init_count=1
        // q_empty: 2 barriers, init_count=1
        // --- pipeline 'kv_pipe' ---
        // kv_full: 9 barriers, init_count=1
        // kv_empty: 9 barriers, init_count=1
        // s_full0: 1 barriers, init_count=1
        // s_full1: 1 barriers, init_count=1
        // _source_p_empty0: 1 barriers, init_count=1
        // _source_p_empty1: 1 barriers, init_count=1
        // o_full0: 1 barriers, init_count=1
        // o_full1: 1 barriers, init_count=1
        // --- pipeline 'page_pipe' ---
        // page_full: 6 barriers, init_count=32
        // page_empty: 6 barriers, init_count=32
        // s_empty0: 1 barriers, init_count=128
        // scale_ready0: 1 barriers, init_count=128
        // stats_empty0: 1 barriers, init_count=128
        // s_empty1: 1 barriers, init_count=128
        // scale_ready1: 1 barriers, init_count=128
        // stats_empty1: 1 barriers, init_count=128
        // order_p01_0: 1 barriers, init_count=128
        // order_p01_1: 1 barriers, init_count=128
        // _source_p_ready0: 1 barriers, init_count=128
        // _source_p_ready1: 1 barriers, init_count=128
        // o_empty0: 1 barriers, init_count=128
        // o_empty1: 1 barriers, init_count=128
        // _source_barrier_reservation: 12 barriers, init_count=128
        // Warp-cooperative initialization in physical record order.
        uint32_t _mbarrier_init_count_0_0 = 32;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(28), "r"((uint32_t)(1)));
        mbarrier_init(smem + 177920 + lane * 8, _mbarrier_init_count_0_0);
        uint32_t _mbarrier_init_count_0_32 = 128;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_32) : "r"(lane), "n"(8), "r"((uint32_t)(32)));
        mbarrier_init(smem + 178176 + lane * 8, _mbarrier_init_count_0_32);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 178432);
    if (warp == 0) {
        int _tmem_hold = smem + 178432;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem = taddr;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 96;");
    }

    // ---- Role: softmax0 ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 176;");
        { // softmax0_main
            int cta_q_s0 = blockIdx.x / 2;
            int cta_kv_s0 = blockIdx.x % 2;
            int batch_s0 = blockIdx.z;
            int seq_len_s0 = seq_lens[batch_s0] - (q_len - 1 - cta_q_s0);
            int num_ctas_s0 = 2;
            int num_steps_s0 = 2;
            const int warp_in_group = warp % 4;
            const int wg_tid = warp_in_group * 32 + lane;
            const int col_group = wg_tid % 4;
            const int col_group_base = col_group * 4;
            float row_max[4];
            float row_sum[4];
            row_max[0] = -3.4028235e+38f;
            row_max[1] = -3.4028235e+38f;
            row_max[2] = -3.4028235e+38f;
            row_max[3] = -3.4028235e+38f;
            row_sum[0] = 0.0f;
            row_sum[1] = 0.0f;
            row_sum[2] = 0.0f;
            row_sum[3] = 0.0f;
            uint32_t _amf_u_0 = __float_as_uint(-3.4028235e+38f);
            uint32_t _amf_mask_0 = -int32_t(_amf_u_0 >> 31) | 0x80000000u;
            unsigned int _amf_enc_0 = _amf_u_0 ^ _amf_mask_0;
            if (wg_tid < 16) {
                exchange0_u32[wg_tid] = _amf_enc_0;
            }
            asm volatile("barrier.sync 8, 128;" ::: "memory");
            int order_phase = 1;
            int order_stage = 0;
            unsigned int _phase_s_full0_0 = 0;
            unsigned int _phase_stats_empty0_0 = 1;
            #pragma unroll 1
            for (int tile = 0; tile < num_steps_s0 * 2; tile += 2) {
                mbarrier_wait(s_full0_addr, _phase_s_full0_0);
                _phase_s_full0_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float scores[16];
                float scores_lo[8];
                float scores_hi[8];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&scores_lo[0])), "=r"(*reinterpret_cast<uint32_t*>(&scores_lo[1])), "=r"(*reinterpret_cast<uint32_t*>(&scores_lo[2])), "=r"(*reinterpret_cast<uint32_t*>(&scores_lo[3])), "=r"(*reinterpret_cast<uint32_t*>(&scores_lo[4])), "=r"(*reinterpret_cast<uint32_t*>(&scores_lo[5])), "=r"(*reinterpret_cast<uint32_t*>(&scores_lo[6])), "=r"(*reinterpret_cast<uint32_t*>(&scores_lo[7]))
                    : "r"(taddr));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&scores_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&scores_hi[1])), "=r"(*reinterpret_cast<uint32_t*>(&scores_hi[2])), "=r"(*reinterpret_cast<uint32_t*>(&scores_hi[3])), "=r"(*reinterpret_cast<uint32_t*>(&scores_hi[4])), "=r"(*reinterpret_cast<uint32_t*>(&scores_hi[5])), "=r"(*reinterpret_cast<uint32_t*>(&scores_hi[6])), "=r"(*reinterpret_cast<uint32_t*>(&scores_hi[7]))
                    : "r"(taddr + 1048576));
                #pragma unroll
                for (int value_idx = 0; value_idx < 8; value_idx++) {
                    scores[value_idx] = scores_lo[value_idx];
                    scores[value_idx + 8] = scores_hi[value_idx];
                }
                int tile_token_base = (cta_kv_s0 * (num_steps_s0 * 2) + tile) * 128;
                int local_token_base = warp_in_group * 32 + lane / 4;
                if (seq_len_s0 < tile_token_base + 128) {
                    if (seq_len_s0 <= tile_token_base + local_token_base) {
                        scores[0] = -3.4028235e+38f;
                        scores[1] = -3.4028235e+38f;
                        scores[4] = -3.4028235e+38f;
                        scores[5] = -3.4028235e+38f;
                    }
                    if (seq_len_s0 <= tile_token_base + local_token_base + 8) {
                        scores[2] = -3.4028235e+38f;
                        scores[3] = -3.4028235e+38f;
                        scores[6] = -3.4028235e+38f;
                        scores[7] = -3.4028235e+38f;
                    }
                    if (seq_len_s0 <= tile_token_base + local_token_base + 16) {
                        scores[8] = -3.4028235e+38f;
                        scores[9] = -3.4028235e+38f;
                        scores[12] = -3.4028235e+38f;
                        scores[13] = -3.4028235e+38f;
                    }
                    if (seq_len_s0 <= tile_token_base + local_token_base + 24) {
                        scores[10] = -3.4028235e+38f;
                        scores[11] = -3.4028235e+38f;
                        scores[14] = -3.4028235e+38f;
                        scores[15] = -3.4028235e+38f;
                    }
                }
                float tile_max[4];
                float _max_0 = max_noftz(scores[0], scores[2]);
                float _max_1 = max_noftz(scores[8], scores[10]);
                float _max_2 = max_noftz(_max_0, _max_1);
                tile_max[0] = _max_2;
                float _max_3 = max_noftz(scores[1], scores[3]);
                float _max_4 = max_noftz(scores[9], scores[11]);
                float _max_5 = max_noftz(_max_3, _max_4);
                tile_max[1] = _max_5;
                float _max_6 = max_noftz(scores[4], scores[6]);
                float _max_7 = max_noftz(scores[12], scores[14]);
                float _max_8 = max_noftz(_max_6, _max_7);
                tile_max[2] = _max_8;
                float _max_9 = max_noftz(scores[5], scores[7]);
                float _max_10 = max_noftz(scores[13], scores[15]);
                float _max_11 = max_noftz(_max_9, _max_10);
                tile_max[3] = _max_11;
                const int local_row_idx = lane / 4 % 4;
                float pair01_left = ((local_row_idx % 2 == 0) ? tile_max[1] : tile_max[0]);
                float pair01_right = ((local_row_idx % 2 == 0) ? tile_max[0] : tile_max[1]);
                float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, pair01_left, 4);
                float _max_12 = max_noftz(_shfl_xor_0, pair01_right);
                float pair01 = _max_12;
                float pair23_left = ((local_row_idx % 2 == 0) ? tile_max[3] : tile_max[2]);
                float pair23_right = ((local_row_idx % 2 == 0) ? tile_max[2] : tile_max[3]);
                float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, pair23_left, 4);
                float _max_13 = max_noftz(_shfl_xor_1, pair23_right);
                float pair23 = _max_13;
                float group_left = ((local_row_idx < 2) ? pair23 : pair01);
                float group_right = ((local_row_idx < 2) ? pair01 : pair23);
                float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, group_left, 8);
                float _max_14 = max_noftz(_shfl_xor_2, group_right);
                float reduced_max = _max_14;
                uint32_t _amf_u_1 = __float_as_uint(reduced_max);
                uint32_t _amf_mask_1 = -int32_t(_amf_u_1 >> 31) | 0x80000000u;
                unsigned int _amf_enc_1 = _amf_u_1 ^ _amf_mask_1;
                atomicMax(&exchange0_u32[col_group_base + local_row_idx], _amf_enc_1);
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                float old_max[4];
                float new_max[4];
                #pragma unroll
                for (int stat = 0; stat < 4; stat++) {
                    old_max[stat] = row_max[stat];
                    uint32_t _amf_u_2 = exchange0_u32[col_group_base + stat];
                    uint32_t _amf_mask_2 = ((_amf_u_2 >> 31) - 1u) | 0x80000000u;
                    float _amf_dec_0 = __uint_as_float(_amf_u_2 ^ _amf_mask_2);
                    new_max[stat] = _amf_dec_0;
                    row_max[stat] = new_max[stat];
                }
                mbarrier_wait(stats_empty0_addr, _phase_stats_empty0_0);
                _phase_stats_empty0_0 ^= 1;
                float local_stats[8];
                #pragma unroll
                for (int stat_1 = 0; stat_1 < 4; stat_1++) {
                    local_stats[stat_1] = old_max[stat_1];
                    local_stats[stat_1 + 4] = new_max[stat_1];
                }
                tmem_st_x8_f32(taddr + 32, local_stats);
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                mbarrier_arrive(scale_ready0_addr);
                float safe_max0 = ((new_max[0] == -3.4028235e+38f) ? 0.0f : new_max[0]);
                float safe_max1 = ((new_max[1] == -3.4028235e+38f) ? 0.0f : new_max[1]);
                float safe_max2 = ((new_max[2] == -3.4028235e+38f) ? 0.0f : new_max[2]);
                float safe_max3 = ((new_max[3] == -3.4028235e+38f) ? 0.0f : new_max[3]);
                float2 _f2_0 = make_float2(bmm1_scale_log2, bmm1_scale_log2);
                float2 _f2_1 = make_float2(-bmm1_scale_log2, -bmm1_scale_log2);
                float2 _f2_2 = make_float2(8.8073549f, 8.8073549f);
                float2 _f2_3 = make_float2(safe_max0, safe_max1);
                float2 _f2_4 = make_float2(safe_max2, safe_max3);
                float2 neg_max01 = fma_f32x2_rn_ftz(_f2_3, _f2_1, _f2_2);
                float2 neg_max23 = fma_f32x2_rn_ftz(_f2_4, _f2_1, _f2_2);
                float probabilities[16];
                float _fma_0 = __fmaf_rn(bmm1_scale_log2, scores[0], neg_max01.x);
                probabilities[0] = _fma_0;
                float _fma_1 = __fmaf_rn(bmm1_scale_log2, scores[1], neg_max01.y);
                probabilities[1] = _fma_1;
                float2 _f2_5 = make_float2(scores[2], scores[3]);
                float2 _mul_f32x2_0;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_0) : "l"(*(const unsigned long long*)&_f2_0), "l"(*(const unsigned long long*)&_f2_5));
                float2 scaled23 = _mul_f32x2_0;
                float2 prob23 = add_f32x2(scaled23, neg_max01);
                probabilities[2] = prob23.x;
                probabilities[3] = prob23.y;
                mbarrier_wait(order_p01_0_addr, order_phase);
                float _exp2_0 = approx_exp2(probabilities[0]);
                probabilities[0] = _exp2_0;
                float2 _f2_6 = make_float2(scores[4], scores[5]);
                float2 _mul_f32x2_1;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_1) : "l"(*(const unsigned long long*)&_f2_0), "l"(*(const unsigned long long*)&_f2_6));
                float2 scaled45 = _mul_f32x2_1;
                float2 prob45 = add_f32x2(scaled45, neg_max23);
                probabilities[4] = prob45.x;
                probabilities[5] = prob45.y;
                float _exp2_1 = approx_exp2(probabilities[1]);
                probabilities[1] = _exp2_1;
                float _exp2_2 = approx_exp2(probabilities[2]);
                probabilities[2] = _exp2_2;
                float2 _f2_7 = make_float2(scores[6], scores[7]);
                float2 _mul_f32x2_2;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_2) : "l"(*(const unsigned long long*)&_f2_0), "l"(*(const unsigned long long*)&_f2_7));
                float2 scaled67 = _mul_f32x2_2;
                float2 prob67 = add_f32x2(scaled67, neg_max23);
                probabilities[6] = prob67.x;
                probabilities[7] = prob67.y;
                float _exp2_3 = approx_exp2(probabilities[3]);
                probabilities[3] = _exp2_3;
                float _exp2_4 = approx_exp2(probabilities[4]);
                probabilities[4] = _exp2_4;
                float2 _f2_8 = make_float2(scores[8], scores[9]);
                float2 _mul_f32x2_3;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_3) : "l"(*(const unsigned long long*)&_f2_0), "l"(*(const unsigned long long*)&_f2_8));
                float2 scaled89 = _mul_f32x2_3;
                float2 prob89 = add_f32x2(scaled89, neg_max01);
                probabilities[8] = prob89.x;
                probabilities[9] = prob89.y;
                float _exp2_5 = approx_exp2(probabilities[5]);
                probabilities[5] = _exp2_5;
                float _exp2_6 = approx_exp2(probabilities[6]);
                probabilities[6] = _exp2_6;
                float2 _f2_9 = make_float2(scores[10], scores[11]);
                float2 _mul_f32x2_4;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_4) : "l"(*(const unsigned long long*)&_f2_0), "l"(*(const unsigned long long*)&_f2_9));
                float2 scaled1011 = _mul_f32x2_4;
                float2 prob1011 = add_f32x2(scaled1011, neg_max01);
                probabilities[10] = prob1011.x;
                probabilities[11] = prob1011.y;
                float _exp2_7 = approx_exp2(probabilities[7]);
                probabilities[7] = _exp2_7;
                float _exp2_8 = approx_exp2(probabilities[8]);
                probabilities[8] = _exp2_8;
                float2 _f2_10 = make_float2(scores[12], scores[13]);
                float2 _mul_f32x2_5;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_5) : "l"(*(const unsigned long long*)&_f2_0), "l"(*(const unsigned long long*)&_f2_10));
                float2 scaled1213 = _mul_f32x2_5;
                float2 prob1213 = add_f32x2(scaled1213, neg_max23);
                probabilities[12] = prob1213.x;
                probabilities[13] = prob1213.y;
                float _exp2_9 = approx_exp2(probabilities[9]);
                probabilities[9] = _exp2_9;
                mbarrier_arrive(order_p01_1_addr);
                order_phase ^= 1;
                float _exp2_10 = approx_exp2(probabilities[10]);
                probabilities[10] = _exp2_10;
                float2 _f2_11 = make_float2(scores[14], scores[15]);
                float2 _mul_f32x2_6;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_6) : "l"(*(const unsigned long long*)&_f2_0), "l"(*(const unsigned long long*)&_f2_11));
                float2 scaled1415 = _mul_f32x2_6;
                float2 prob1415 = add_f32x2(scaled1415, neg_max23);
                probabilities[14] = prob1415.x;
                probabilities[15] = prob1415.y;
                float _exp2_11 = approx_exp2(probabilities[11]);
                probabilities[11] = _exp2_11;
                unsigned int regs_p[4];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(probabilities[0]), "f"(probabilities[1]),
                                           "f"(probabilities[2]), "f"(probabilities[3]));
                    regs_p[0] = _packed;
                }
                #pragma unroll
                for (int value_idx_1 = 12; value_idx_1 < 16; value_idx_1++) {
                    float _exp2_12 = approx_exp2(probabilities[value_idx_1]);
                    probabilities[value_idx_1] = _exp2_12;
                }
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(probabilities[4]), "f"(probabilities[5]),
                                           "f"(probabilities[6]), "f"(probabilities[7]));
                    regs_p[1] = _packed;
                }
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(probabilities[8]), "f"(probabilities[9]),
                                           "f"(probabilities[10]), "f"(probabilities[11]));
                    regs_p[2] = _packed;
                }
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(probabilities[12]), "f"(probabilities[13]),
                                           "f"(probabilities[14]), "f"(probabilities[15]));
                    regs_p[3] = _packed;
                }
                const int matrix_idx = lane / 8;
                const int matrix_row_idx = matrix_idx % 2;
                const int matrix_col_idx = matrix_idx / 2;
                const int thread_row_idx = lane % 8;
                const int segment_col_idx = warp_in_group * 2 + matrix_col_idx ^ thread_row_idx;
                const int stsm_offset = (matrix_row_idx * 8 + thread_row_idx) * 128 + segment_col_idx * 16;
                const void* _stmatrix_b8_ptr_3 = reinterpret_cast<const void*>(reinterpret_cast<uint8_t*>(smem_p0) + stsm_offset);
                uint64_t _stmatrix_b8_addr64_3;
                asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(_stmatrix_b8_addr64_3) : "l"(_stmatrix_b8_ptr_3));
                uint32_t _stmatrix_b8_addr_3;
                asm volatile("cvt.u32.u64 %0, %1;" : "=r"(_stmatrix_b8_addr_3) : "l"(_stmatrix_b8_addr64_3));
                asm volatile("stmatrix.sync.aligned.m16n8.x4.trans.shared.b8 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_b8_addr_3), "r"(regs_p[0]), "r"(regs_p[1]), "r"(regs_p[2]), "r"(regs_p[3])
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                if (tile + 2 >= num_steps_s0 * 2) {
                    mbarrier_wait(stats_empty0_addr, _phase_stats_empty0_0);
                    _phase_stats_empty0_0 ^= 1;
                }
                mbarrier_arrive(s_empty0_addr);
                float acc_scale[4];
                #pragma unroll
                for (int stat_2 = 0; stat_2 < 4; stat_2++) {
                    float max_delta = old_max[stat_2] - new_max[stat_2];
                    acc_scale[stat_2] = 1.0f;
                    if (max_delta != 0.0f) {
                        float _exp2_13 = approx_exp2(max_delta * bmm1_scale_log2);
                        acc_scale[stat_2] = _exp2_13;
                    }
                }
                float2 _f2_12 = make_float2(row_sum[0], row_sum[1]);
                float2 _f2_13 = make_float2(acc_scale[0], acc_scale[1]);
                float2 _f2_14 = make_float2(probabilities[0], probabilities[1]);
                float2 _f2_15 = make_float2(probabilities[2], probabilities[3]);
                float2 _f2_16 = make_float2(probabilities[8], probabilities[9]);
                float2 _f2_17 = make_float2(probabilities[10], probabilities[11]);
                float2 next_sum01 = fma_f32x2_rn_ftz(_f2_12, _f2_13, _f2_14);
                next_sum01 = add_f32x2(next_sum01, _f2_15);
                next_sum01 = add_f32x2(next_sum01, _f2_16);
                next_sum01 = add_f32x2(next_sum01, _f2_17);
                row_sum[0] = next_sum01.x;
                row_sum[1] = next_sum01.y;
                float2 _f2_18 = make_float2(row_sum[2], row_sum[3]);
                float2 _f2_19 = make_float2(acc_scale[2], acc_scale[3]);
                float2 _f2_20 = make_float2(probabilities[4], probabilities[5]);
                float2 _f2_21 = make_float2(probabilities[6], probabilities[7]);
                float2 _f2_22 = make_float2(probabilities[12], probabilities[13]);
                float2 _f2_23 = make_float2(probabilities[14], probabilities[15]);
                float2 next_sum23 = fma_f32x2_rn_ftz(_f2_18, _f2_19, _f2_20);
                next_sum23 = add_f32x2(next_sum23, _f2_21);
                next_sum23 = add_f32x2(next_sum23, _f2_22);
                next_sum23 = add_f32x2(next_sum23, _f2_23);
                row_sum[2] = next_sum23.x;
                row_sum[3] = next_sum23.y;
                if (tile + 2 >= num_steps_s0 * 2) {
                    float final_stats[8];
                    #pragma unroll
                    for (int stat_3 = 0; stat_3 < 4; stat_3++) {
                        final_stats[stat_3] = row_sum[stat_3];
                        final_stats[stat_3 + 4] = row_max[stat_3];
                    }
                    tmem_st_x8_f32(taddr + 32, final_stats);
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(scale_ready0_addr);
                }
            }
        }
    // ---- Role: softmax1 ----
    } else if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 176;");
        { // softmax1_main
            int cta_q_s1 = blockIdx.x / 2;
            int cta_kv_s1 = blockIdx.x % 2;
            int batch_s1 = blockIdx.z;
            int seq_len_s1 = seq_lens[batch_s1] - (q_len - 1 - cta_q_s1);
            int num_ctas_s1 = 2;
            int num_steps_s1 = 2;
            const int warp_in_group_1 = warp % 4;
            const int wg_tid_1 = warp_in_group_1 * 32 + lane;
            const int col_group_1 = wg_tid_1 % 4;
            const int col_group_base_1 = col_group_1 * 4;
            float row_max_1[4];
            float row_sum_1[4];
            row_max_1[0] = -3.4028235e+38f;
            row_max_1[1] = -3.4028235e+38f;
            row_max_1[2] = -3.4028235e+38f;
            row_max_1[3] = -3.4028235e+38f;
            row_sum_1[0] = 0.0f;
            row_sum_1[1] = 0.0f;
            row_sum_1[2] = 0.0f;
            row_sum_1[3] = 0.0f;
            uint32_t _amf_u_0 = __float_as_uint(-3.4028235e+38f);
            uint32_t _amf_mask_0 = -int32_t(_amf_u_0 >> 31) | 0x80000000u;
            unsigned int _amf_enc_2 = _amf_u_0 ^ _amf_mask_0;
            if (wg_tid_1 < 16) {
                exchange1_u32[wg_tid_1] = _amf_enc_2;
            }
            asm volatile("barrier.sync 9, 128;" ::: "memory");
            int order_phase_1 = 0;
            int order_stage_1 = 0;
            unsigned int _phase_s_full1_0 = 0;
            unsigned int _phase_stats_empty1_0 = 1;
            #pragma unroll 1
            for (int tile_1 = 1; tile_1 < num_steps_s1 * 2; tile_1 += 2) {
                mbarrier_wait(s_full1_addr, _phase_s_full1_0);
                _phase_s_full1_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                float scores_1[16];
                float scores_lo_1[8];
                float scores_hi_1[8];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&scores_lo_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&scores_lo_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&scores_lo_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&scores_lo_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&scores_lo_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&scores_lo_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&scores_lo_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&scores_lo_1[7]))
                    : "r"(taddr + 16));
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&scores_hi_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&scores_hi_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&scores_hi_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&scores_hi_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&scores_hi_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&scores_hi_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&scores_hi_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&scores_hi_1[7]))
                    : "r"(taddr + 16 + 1048576));
                #pragma unroll
                for (int value_idx_2 = 0; value_idx_2 < 8; value_idx_2++) {
                    scores_1[value_idx_2] = scores_lo_1[value_idx_2];
                    scores_1[value_idx_2 + 8] = scores_hi_1[value_idx_2];
                }
                int tile_token_base_1 = (cta_kv_s1 * (num_steps_s1 * 2) + tile_1) * 128;
                int local_token_base_1 = warp_in_group_1 * 32 + lane / 4;
                if (seq_len_s1 < tile_token_base_1 + 128) {
                    if (seq_len_s1 <= tile_token_base_1 + local_token_base_1) {
                        scores_1[0] = -3.4028235e+38f;
                        scores_1[1] = -3.4028235e+38f;
                        scores_1[4] = -3.4028235e+38f;
                        scores_1[5] = -3.4028235e+38f;
                    }
                    if (seq_len_s1 <= tile_token_base_1 + local_token_base_1 + 8) {
                        scores_1[2] = -3.4028235e+38f;
                        scores_1[3] = -3.4028235e+38f;
                        scores_1[6] = -3.4028235e+38f;
                        scores_1[7] = -3.4028235e+38f;
                    }
                    if (seq_len_s1 <= tile_token_base_1 + local_token_base_1 + 16) {
                        scores_1[8] = -3.4028235e+38f;
                        scores_1[9] = -3.4028235e+38f;
                        scores_1[12] = -3.4028235e+38f;
                        scores_1[13] = -3.4028235e+38f;
                    }
                    if (seq_len_s1 <= tile_token_base_1 + local_token_base_1 + 24) {
                        scores_1[10] = -3.4028235e+38f;
                        scores_1[11] = -3.4028235e+38f;
                        scores_1[14] = -3.4028235e+38f;
                        scores_1[15] = -3.4028235e+38f;
                    }
                }
                float tile_max_1[4];
                float _max_15 = max_noftz(scores_1[0], scores_1[2]);
                float _max_16 = max_noftz(scores_1[8], scores_1[10]);
                float _max_17 = max_noftz(_max_15, _max_16);
                tile_max_1[0] = _max_17;
                float _max_18 = max_noftz(scores_1[1], scores_1[3]);
                float _max_19 = max_noftz(scores_1[9], scores_1[11]);
                float _max_20 = max_noftz(_max_18, _max_19);
                tile_max_1[1] = _max_20;
                float _max_21 = max_noftz(scores_1[4], scores_1[6]);
                float _max_22 = max_noftz(scores_1[12], scores_1[14]);
                float _max_23 = max_noftz(_max_21, _max_22);
                tile_max_1[2] = _max_23;
                float _max_24 = max_noftz(scores_1[5], scores_1[7]);
                float _max_25 = max_noftz(scores_1[13], scores_1[15]);
                float _max_26 = max_noftz(_max_24, _max_25);
                tile_max_1[3] = _max_26;
                const int local_row_idx_1 = lane / 4 % 4;
                float pair01_left_1 = ((local_row_idx_1 % 2 == 0) ? tile_max_1[1] : tile_max_1[0]);
                float pair01_right_1 = ((local_row_idx_1 % 2 == 0) ? tile_max_1[0] : tile_max_1[1]);
                float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, pair01_left_1, 4);
                float _max_27 = max_noftz(_shfl_xor_3, pair01_right_1);
                float pair01_1 = _max_27;
                float pair23_left_1 = ((local_row_idx_1 % 2 == 0) ? tile_max_1[3] : tile_max_1[2]);
                float pair23_right_1 = ((local_row_idx_1 % 2 == 0) ? tile_max_1[2] : tile_max_1[3]);
                float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, pair23_left_1, 4);
                float _max_28 = max_noftz(_shfl_xor_4, pair23_right_1);
                float pair23_1 = _max_28;
                float group_left_1 = ((local_row_idx_1 < 2) ? pair23_1 : pair01_1);
                float group_right_1 = ((local_row_idx_1 < 2) ? pair01_1 : pair23_1);
                float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, group_left_1, 8);
                float _max_29 = max_noftz(_shfl_xor_5, group_right_1);
                float reduced_max_1 = _max_29;
                uint32_t _amf_u_1 = __float_as_uint(reduced_max_1);
                uint32_t _amf_mask_1 = -int32_t(_amf_u_1 >> 31) | 0x80000000u;
                unsigned int _amf_enc_3 = _amf_u_1 ^ _amf_mask_1;
                atomicMax(&exchange1_u32[col_group_base_1 + local_row_idx_1], _amf_enc_3);
                asm volatile("barrier.sync 9, 128;" ::: "memory");
                float old_max_1[4];
                float new_max_1[4];
                #pragma unroll
                for (int stat_4 = 0; stat_4 < 4; stat_4++) {
                    old_max_1[stat_4] = row_max_1[stat_4];
                    uint32_t _amf_u_2 = exchange1_u32[col_group_base_1 + stat_4];
                    uint32_t _amf_mask_2 = ((_amf_u_2 >> 31) - 1u) | 0x80000000u;
                    float _amf_dec_1 = __uint_as_float(_amf_u_2 ^ _amf_mask_2);
                    new_max_1[stat_4] = _amf_dec_1;
                    row_max_1[stat_4] = new_max_1[stat_4];
                }
                mbarrier_wait(stats_empty1_addr, _phase_stats_empty1_0);
                _phase_stats_empty1_0 ^= 1;
                float local_stats_1[8];
                #pragma unroll
                for (int stat_5 = 0; stat_5 < 4; stat_5++) {
                    local_stats_1[stat_5] = old_max_1[stat_5];
                    local_stats_1[stat_5 + 4] = new_max_1[stat_5];
                }
                tmem_st_x8_f32(taddr + 64, local_stats_1);
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                mbarrier_arrive(scale_ready1_addr);
                float safe_max0_1 = ((new_max_1[0] == -3.4028235e+38f) ? 0.0f : new_max_1[0]);
                float safe_max1_1 = ((new_max_1[1] == -3.4028235e+38f) ? 0.0f : new_max_1[1]);
                float safe_max2_1 = ((new_max_1[2] == -3.4028235e+38f) ? 0.0f : new_max_1[2]);
                float safe_max3_1 = ((new_max_1[3] == -3.4028235e+38f) ? 0.0f : new_max_1[3]);
                float2 _f2_24 = make_float2(bmm1_scale_log2, bmm1_scale_log2);
                float2 _f2_25 = make_float2(-bmm1_scale_log2, -bmm1_scale_log2);
                float2 _f2_26 = make_float2(8.8073549f, 8.8073549f);
                float2 _f2_27 = make_float2(safe_max0_1, safe_max1_1);
                float2 _f2_28 = make_float2(safe_max2_1, safe_max3_1);
                float2 neg_max01_1 = fma_f32x2_rn_ftz(_f2_27, _f2_25, _f2_26);
                float2 neg_max23_1 = fma_f32x2_rn_ftz(_f2_28, _f2_25, _f2_26);
                float probabilities_1[16];
                float _fma_2 = __fmaf_rn(bmm1_scale_log2, scores_1[0], neg_max01_1.x);
                probabilities_1[0] = _fma_2;
                float _fma_3 = __fmaf_rn(bmm1_scale_log2, scores_1[1], neg_max01_1.y);
                probabilities_1[1] = _fma_3;
                float2 _f2_29 = make_float2(scores_1[2], scores_1[3]);
                float2 _mul_f32x2_7;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_7) : "l"(*(const unsigned long long*)&_f2_24), "l"(*(const unsigned long long*)&_f2_29));
                float2 scaled23_1 = _mul_f32x2_7;
                float2 prob23_1 = add_f32x2(scaled23_1, neg_max01_1);
                probabilities_1[2] = prob23_1.x;
                probabilities_1[3] = prob23_1.y;
                mbarrier_wait(order_p01_1_addr, order_phase_1);
                float _exp2_14 = approx_exp2(probabilities_1[0]);
                probabilities_1[0] = _exp2_14;
                float2 _f2_30 = make_float2(scores_1[4], scores_1[5]);
                float2 _mul_f32x2_8;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_8) : "l"(*(const unsigned long long*)&_f2_24), "l"(*(const unsigned long long*)&_f2_30));
                float2 scaled45_1 = _mul_f32x2_8;
                float2 prob45_1 = add_f32x2(scaled45_1, neg_max23_1);
                probabilities_1[4] = prob45_1.x;
                probabilities_1[5] = prob45_1.y;
                float _exp2_15 = approx_exp2(probabilities_1[1]);
                probabilities_1[1] = _exp2_15;
                float _exp2_16 = approx_exp2(probabilities_1[2]);
                probabilities_1[2] = _exp2_16;
                float2 _f2_31 = make_float2(scores_1[6], scores_1[7]);
                float2 _mul_f32x2_9;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_9) : "l"(*(const unsigned long long*)&_f2_24), "l"(*(const unsigned long long*)&_f2_31));
                float2 scaled67_1 = _mul_f32x2_9;
                float2 prob67_1 = add_f32x2(scaled67_1, neg_max23_1);
                probabilities_1[6] = prob67_1.x;
                probabilities_1[7] = prob67_1.y;
                float _exp2_17 = approx_exp2(probabilities_1[3]);
                probabilities_1[3] = _exp2_17;
                float _exp2_18 = approx_exp2(probabilities_1[4]);
                probabilities_1[4] = _exp2_18;
                float2 _f2_32 = make_float2(scores_1[8], scores_1[9]);
                float2 _mul_f32x2_10;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_10) : "l"(*(const unsigned long long*)&_f2_24), "l"(*(const unsigned long long*)&_f2_32));
                float2 scaled89_1 = _mul_f32x2_10;
                float2 prob89_1 = add_f32x2(scaled89_1, neg_max01_1);
                probabilities_1[8] = prob89_1.x;
                probabilities_1[9] = prob89_1.y;
                float _exp2_19 = approx_exp2(probabilities_1[5]);
                probabilities_1[5] = _exp2_19;
                float _exp2_20 = approx_exp2(probabilities_1[6]);
                probabilities_1[6] = _exp2_20;
                float2 _f2_33 = make_float2(scores_1[10], scores_1[11]);
                float2 _mul_f32x2_11;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_11) : "l"(*(const unsigned long long*)&_f2_24), "l"(*(const unsigned long long*)&_f2_33));
                float2 scaled1011_1 = _mul_f32x2_11;
                float2 prob1011_1 = add_f32x2(scaled1011_1, neg_max01_1);
                probabilities_1[10] = prob1011_1.x;
                probabilities_1[11] = prob1011_1.y;
                float _exp2_21 = approx_exp2(probabilities_1[7]);
                probabilities_1[7] = _exp2_21;
                float _exp2_22 = approx_exp2(probabilities_1[8]);
                probabilities_1[8] = _exp2_22;
                float2 _f2_34 = make_float2(scores_1[12], scores_1[13]);
                float2 _mul_f32x2_12;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_12) : "l"(*(const unsigned long long*)&_f2_24), "l"(*(const unsigned long long*)&_f2_34));
                float2 scaled1213_1 = _mul_f32x2_12;
                float2 prob1213_1 = add_f32x2(scaled1213_1, neg_max23_1);
                probabilities_1[12] = prob1213_1.x;
                probabilities_1[13] = prob1213_1.y;
                float _exp2_23 = approx_exp2(probabilities_1[9]);
                probabilities_1[9] = _exp2_23;
                mbarrier_arrive(order_p01_0_addr);
                order_phase_1 ^= 1;
                float _exp2_24 = approx_exp2(probabilities_1[10]);
                probabilities_1[10] = _exp2_24;
                float2 _f2_35 = make_float2(scores_1[14], scores_1[15]);
                float2 _mul_f32x2_13;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_13) : "l"(*(const unsigned long long*)&_f2_24), "l"(*(const unsigned long long*)&_f2_35));
                float2 scaled1415_1 = _mul_f32x2_13;
                float2 prob1415_1 = add_f32x2(scaled1415_1, neg_max23_1);
                probabilities_1[14] = prob1415_1.x;
                probabilities_1[15] = prob1415_1.y;
                float _exp2_25 = approx_exp2(probabilities_1[11]);
                probabilities_1[11] = _exp2_25;
                unsigned int regs_p_1[4];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(probabilities_1[0]), "f"(probabilities_1[1]),
                                           "f"(probabilities_1[2]), "f"(probabilities_1[3]));
                    regs_p_1[0] = _packed;
                }
                #pragma unroll
                for (int value_idx_3 = 12; value_idx_3 < 16; value_idx_3++) {
                    float _exp2_26 = approx_exp2(probabilities_1[value_idx_3]);
                    probabilities_1[value_idx_3] = _exp2_26;
                }
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(probabilities_1[4]), "f"(probabilities_1[5]),
                                           "f"(probabilities_1[6]), "f"(probabilities_1[7]));
                    regs_p_1[1] = _packed;
                }
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(probabilities_1[8]), "f"(probabilities_1[9]),
                                           "f"(probabilities_1[10]), "f"(probabilities_1[11]));
                    regs_p_1[2] = _packed;
                }
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(probabilities_1[12]), "f"(probabilities_1[13]),
                                           "f"(probabilities_1[14]), "f"(probabilities_1[15]));
                    regs_p_1[3] = _packed;
                }
                const int matrix_idx_1 = lane / 8;
                const int matrix_row_idx_1 = matrix_idx_1 % 2;
                const int matrix_col_idx_1 = matrix_idx_1 / 2;
                const int thread_row_idx_1 = lane % 8;
                const int segment_col_idx_1 = warp_in_group_1 * 2 + matrix_col_idx_1 ^ thread_row_idx_1;
                const int stsm_offset_1 = (matrix_row_idx_1 * 8 + thread_row_idx_1) * 128 + segment_col_idx_1 * 16;
                const void* _stmatrix_b8_ptr_3 = reinterpret_cast<const void*>(reinterpret_cast<uint8_t*>(smem_p1) + stsm_offset_1);
                uint64_t _stmatrix_b8_addr64_3;
                asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(_stmatrix_b8_addr64_3) : "l"(_stmatrix_b8_ptr_3));
                uint32_t _stmatrix_b8_addr_3;
                asm volatile("cvt.u32.u64 %0, %1;" : "=r"(_stmatrix_b8_addr_3) : "l"(_stmatrix_b8_addr64_3));
                asm volatile("stmatrix.sync.aligned.m16n8.x4.trans.shared.b8 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_b8_addr_3), "r"(regs_p_1[0]), "r"(regs_p_1[1]), "r"(regs_p_1[2]), "r"(regs_p_1[3])
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                if (tile_1 + 2 >= num_steps_s1 * 2) {
                    mbarrier_wait(stats_empty1_addr, _phase_stats_empty1_0);
                    _phase_stats_empty1_0 ^= 1;
                }
                mbarrier_arrive(s_empty1_addr);
                float acc_scale_1[4];
                #pragma unroll
                for (int stat_6 = 0; stat_6 < 4; stat_6++) {
                    float max_delta_1 = old_max_1[stat_6] - new_max_1[stat_6];
                    acc_scale_1[stat_6] = 1.0f;
                    if (max_delta_1 != 0.0f) {
                        float _exp2_27 = approx_exp2(max_delta_1 * bmm1_scale_log2);
                        acc_scale_1[stat_6] = _exp2_27;
                    }
                }
                float2 _f2_36 = make_float2(row_sum_1[0], row_sum_1[1]);
                float2 _f2_37 = make_float2(acc_scale_1[0], acc_scale_1[1]);
                float2 _f2_38 = make_float2(probabilities_1[0], probabilities_1[1]);
                float2 _f2_39 = make_float2(probabilities_1[2], probabilities_1[3]);
                float2 _f2_40 = make_float2(probabilities_1[8], probabilities_1[9]);
                float2 _f2_41 = make_float2(probabilities_1[10], probabilities_1[11]);
                float2 next_sum01_1 = fma_f32x2_rn_ftz(_f2_36, _f2_37, _f2_38);
                next_sum01_1 = add_f32x2(next_sum01_1, _f2_39);
                next_sum01_1 = add_f32x2(next_sum01_1, _f2_40);
                next_sum01_1 = add_f32x2(next_sum01_1, _f2_41);
                row_sum_1[0] = next_sum01_1.x;
                row_sum_1[1] = next_sum01_1.y;
                float2 _f2_42 = make_float2(row_sum_1[2], row_sum_1[3]);
                float2 _f2_43 = make_float2(acc_scale_1[2], acc_scale_1[3]);
                float2 _f2_44 = make_float2(probabilities_1[4], probabilities_1[5]);
                float2 _f2_45 = make_float2(probabilities_1[6], probabilities_1[7]);
                float2 _f2_46 = make_float2(probabilities_1[12], probabilities_1[13]);
                float2 _f2_47 = make_float2(probabilities_1[14], probabilities_1[15]);
                float2 next_sum23_1 = fma_f32x2_rn_ftz(_f2_42, _f2_43, _f2_44);
                next_sum23_1 = add_f32x2(next_sum23_1, _f2_45);
                next_sum23_1 = add_f32x2(next_sum23_1, _f2_46);
                next_sum23_1 = add_f32x2(next_sum23_1, _f2_47);
                row_sum_1[2] = next_sum23_1.x;
                row_sum_1[3] = next_sum23_1.y;
                if (tile_1 + 2 >= num_steps_s1 * 2) {
                    float final_stats_1[8];
                    #pragma unroll
                    for (int stat_7 = 0; stat_7 < 4; stat_7++) {
                        final_stats_1[stat_7] = row_sum_1[stat_7];
                        final_stats_1[stat_7 + 4] = row_max_1[stat_7];
                    }
                    tmem_st_x8_f32(taddr + 64, final_stats_1);
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    mbarrier_arrive(scale_ready1_addr);
                }
            }
        }
    // ---- Role: correction ----
    } else if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 64;");
        { // correction_main
            int cta_q_c = blockIdx.x / 2;
            int cta_kv_c = blockIdx.x % 2;
            int y_c = blockIdx.y;
            int head_group_c = y_c / 4;
            int hv_partition_c = y_c % 4;
            int batch_c = blockIdx.z;
            int seq_len_c = seq_lens[batch_c] - (q_len - 1 - cta_q_c);
            int num_ctas_c = 2;
            int num_steps_c = 2;
            int num_tiles_c = num_steps_c * 2;
            const int warp_c = warp % 4;
            const int corr_tid = warp_c * 32 + lane;
            const int tmem_row_c = warp_c * 32 << 16;
            const int corr_head_base_c = corr_tid % 4 * 4;
            unsigned int _phase_scale_ready0_0 = 0;
            unsigned int _phase_o_full0_0 = 0;
            unsigned int _phase_scale_ready1_0 = 0;
            unsigned int _phase_o_full1_0 = 0;
            #pragma unroll 1
            for (int tile_c = 0; tile_c < num_tiles_c; tile_c++) {
                if (tile_c % 2 == 0) {
                    mbarrier_wait(scale_ready0_addr, _phase_scale_ready0_0);
                    _phase_scale_ready0_0 ^= 1;
                    float corr_scale0[4];
                    if (tile_c > 0) {
                        float _tmem_load_0[8];
                        tmem_ld_x8(&_tmem_load_0[0], taddr + 32);
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        #pragma unroll
                        for (int head_local_c = 0; head_local_c < 4; head_local_c++) {
                            float corr_delta0_c = _tmem_load_0[head_local_c] - _tmem_load_0[head_local_c + 4];
                            float _exp2_28 = approx_exp2(corr_delta0_c * bmm1_scale_log2);
                            corr_scale0[head_local_c] = ((corr_delta0_c != 0.0f) ? _exp2_28 : 1.0f);
                        }
                    }
                    mbarrier_arrive(stats_empty0_addr);
                    if (tile_c > 0) {
                        mbarrier_wait(o_full0_addr, _phase_o_full0_0);
                        _phase_o_full0_0 ^= 1;
                        float previous0_lo[8];
                        float previous0_hi[8];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&previous0_lo[0])), "=r"(*reinterpret_cast<uint32_t*>(&previous0_lo[1])), "=r"(*reinterpret_cast<uint32_t*>(&previous0_lo[2])), "=r"(*reinterpret_cast<uint32_t*>(&previous0_lo[3])), "=r"(*reinterpret_cast<uint32_t*>(&previous0_lo[4])), "=r"(*reinterpret_cast<uint32_t*>(&previous0_lo[5])), "=r"(*reinterpret_cast<uint32_t*>(&previous0_lo[6])), "=r"(*reinterpret_cast<uint32_t*>(&previous0_lo[7]))
                            : "r"(taddr + 96));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&previous0_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&previous0_hi[1])), "=r"(*reinterpret_cast<uint32_t*>(&previous0_hi[2])), "=r"(*reinterpret_cast<uint32_t*>(&previous0_hi[3])), "=r"(*reinterpret_cast<uint32_t*>(&previous0_hi[4])), "=r"(*reinterpret_cast<uint32_t*>(&previous0_hi[5])), "=r"(*reinterpret_cast<uint32_t*>(&previous0_hi[6])), "=r"(*reinterpret_cast<uint32_t*>(&previous0_hi[7]))
                            : "r"(taddr + 96 + 1048576));
                        float2 _f2_48 = make_float2(corr_scale0[0], corr_scale0[1]);
                        float2 _f2_49 = make_float2(corr_scale0[2], corr_scale0[3]);
                        int corr_pred0 = 0;
                        #pragma unroll
                        for (int head_local_c_1 = 0; head_local_c_1 < 4; head_local_c_1++) {
                            if (corr_scale0[head_local_c_1] != 1.0f) {
                                corr_pred0 = 1;
                            }
                        }
                        int _vote_0 = __any_sync(0xFFFFFFFF, corr_pred0 != 0);
                        if (_vote_0 != 0) {
                            #pragma unroll
                            for (int reg_base_c = 0; reg_base_c < 8; reg_base_c += 4) {
                                float2 _f2_50 = make_float2(previous0_lo[reg_base_c], previous0_lo[reg_base_c + 1]);
                                float2 _f2_51 = make_float2(previous0_lo[reg_base_c + 2], previous0_lo[reg_base_c + 3]);
                                float2 _f2_52 = make_float2(previous0_hi[reg_base_c], previous0_hi[reg_base_c + 1]);
                                float2 _f2_53 = make_float2(previous0_hi[reg_base_c + 2], previous0_hi[reg_base_c + 3]);
                                float2 _f32x2_mul_b_0 = ((reg_base_c == 0) ? _f2_48 : _f2_49);
                                float2 _mul_f32x2_14;
                                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_14) : "l"(*(const unsigned long long*)&_f2_50), "l"(*(const unsigned long long*)&_f32x2_mul_b_0));
                                float2 prev0_lo01_scaled_f2 = _mul_f32x2_14;
                                float2 _f32x2_mul_b_1 = ((reg_base_c == 0) ? _f2_48 : _f2_49);
                                float2 _mul_f32x2_15;
                                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_15) : "l"(*(const unsigned long long*)&_f2_51), "l"(*(const unsigned long long*)&_f32x2_mul_b_1));
                                float2 prev0_lo23_scaled_f2 = _mul_f32x2_15;
                                float2 _f32x2_mul_b_2 = ((reg_base_c == 0) ? _f2_48 : _f2_49);
                                float2 _mul_f32x2_16;
                                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_16) : "l"(*(const unsigned long long*)&_f2_52), "l"(*(const unsigned long long*)&_f32x2_mul_b_2));
                                float2 prev0_hi01_scaled_f2 = _mul_f32x2_16;
                                float2 _f32x2_mul_b_3 = ((reg_base_c == 0) ? _f2_48 : _f2_49);
                                float2 _mul_f32x2_17;
                                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_17) : "l"(*(const unsigned long long*)&_f2_53), "l"(*(const unsigned long long*)&_f32x2_mul_b_3));
                                float2 prev0_hi23_scaled_f2 = _mul_f32x2_17;
                                previous0_lo[reg_base_c] = prev0_lo01_scaled_f2.x;
                                previous0_lo[reg_base_c + 1] = prev0_lo01_scaled_f2.y;
                                previous0_lo[reg_base_c + 2] = prev0_lo23_scaled_f2.x;
                                previous0_lo[reg_base_c + 3] = prev0_lo23_scaled_f2.y;
                                previous0_hi[reg_base_c] = prev0_hi01_scaled_f2.x;
                                previous0_hi[reg_base_c + 1] = prev0_hi01_scaled_f2.y;
                                previous0_hi[reg_base_c + 2] = prev0_hi23_scaled_f2.x;
                                previous0_hi[reg_base_c + 3] = prev0_hi23_scaled_f2.y;
                            }
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x256b.x2.b32"
                                " [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "r"(taddr + 96), "r"(*reinterpret_cast<const uint32_t*>(&previous0_lo[0])), "r"(*reinterpret_cast<const uint32_t*>(&previous0_lo[1])), "r"(*reinterpret_cast<const uint32_t*>(&previous0_lo[2])), "r"(*reinterpret_cast<const uint32_t*>(&previous0_lo[3])), "r"(*reinterpret_cast<const uint32_t*>(&previous0_lo[4])), "r"(*reinterpret_cast<const uint32_t*>(&previous0_lo[5])), "r"(*reinterpret_cast<const uint32_t*>(&previous0_lo[6])), "r"(*reinterpret_cast<const uint32_t*>(&previous0_lo[7])));
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x256b.x2.b32"
                                " [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "r"(taddr + 96 + 1048576), "r"(*reinterpret_cast<const uint32_t*>(&previous0_hi[0])), "r"(*reinterpret_cast<const uint32_t*>(&previous0_hi[1])), "r"(*reinterpret_cast<const uint32_t*>(&previous0_hi[2])), "r"(*reinterpret_cast<const uint32_t*>(&previous0_hi[3])), "r"(*reinterpret_cast<const uint32_t*>(&previous0_hi[4])), "r"(*reinterpret_cast<const uint32_t*>(&previous0_hi[5])), "r"(*reinterpret_cast<const uint32_t*>(&previous0_hi[6])), "r"(*reinterpret_cast<const uint32_t*>(&previous0_hi[7])));
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        }
                        mbarrier_arrive(o_empty0_addr);
                    }
                } else {
                    mbarrier_wait(scale_ready1_addr, _phase_scale_ready1_0);
                    _phase_scale_ready1_0 ^= 1;
                    float corr_scale1[4];
                    if (tile_c > 1) {
                        float _tmem_load_1[8];
                        tmem_ld_x8(&_tmem_load_1[0], taddr + 64);
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        #pragma unroll
                        for (int head_local_c_2 = 0; head_local_c_2 < 4; head_local_c_2++) {
                            float corr_delta1_c = _tmem_load_1[head_local_c_2] - _tmem_load_1[head_local_c_2 + 4];
                            float _exp2_29 = approx_exp2(corr_delta1_c * bmm1_scale_log2);
                            corr_scale1[head_local_c_2] = ((corr_delta1_c != 0.0f) ? _exp2_29 : 1.0f);
                        }
                    }
                    mbarrier_arrive(stats_empty1_addr);
                    if (tile_c > 1) {
                        mbarrier_wait(o_full1_addr, _phase_o_full1_0);
                        _phase_o_full1_0 ^= 1;
                        float previous1_lo[8];
                        float previous1_hi[8];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&previous1_lo[0])), "=r"(*reinterpret_cast<uint32_t*>(&previous1_lo[1])), "=r"(*reinterpret_cast<uint32_t*>(&previous1_lo[2])), "=r"(*reinterpret_cast<uint32_t*>(&previous1_lo[3])), "=r"(*reinterpret_cast<uint32_t*>(&previous1_lo[4])), "=r"(*reinterpret_cast<uint32_t*>(&previous1_lo[5])), "=r"(*reinterpret_cast<uint32_t*>(&previous1_lo[6])), "=r"(*reinterpret_cast<uint32_t*>(&previous1_lo[7]))
                            : "r"(taddr + 112));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&previous1_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&previous1_hi[1])), "=r"(*reinterpret_cast<uint32_t*>(&previous1_hi[2])), "=r"(*reinterpret_cast<uint32_t*>(&previous1_hi[3])), "=r"(*reinterpret_cast<uint32_t*>(&previous1_hi[4])), "=r"(*reinterpret_cast<uint32_t*>(&previous1_hi[5])), "=r"(*reinterpret_cast<uint32_t*>(&previous1_hi[6])), "=r"(*reinterpret_cast<uint32_t*>(&previous1_hi[7]))
                            : "r"(taddr + 112 + 1048576));
                        float2 _f2_54 = make_float2(corr_scale1[0], corr_scale1[1]);
                        float2 _f2_55 = make_float2(corr_scale1[2], corr_scale1[3]);
                        int corr_pred1 = 0;
                        #pragma unroll
                        for (int head_local_c_3 = 0; head_local_c_3 < 4; head_local_c_3++) {
                            if (corr_scale1[head_local_c_3] != 1.0f) {
                                corr_pred1 = 1;
                            }
                        }
                        int _vote_1 = __any_sync(0xFFFFFFFF, corr_pred1 != 0);
                        if (_vote_1 != 0) {
                            #pragma unroll
                            for (int reg_base_c_1 = 0; reg_base_c_1 < 8; reg_base_c_1 += 4) {
                                float2 _f2_56 = make_float2(previous1_lo[reg_base_c_1], previous1_lo[reg_base_c_1 + 1]);
                                float2 _f2_57 = make_float2(previous1_lo[reg_base_c_1 + 2], previous1_lo[reg_base_c_1 + 3]);
                                float2 _f2_58 = make_float2(previous1_hi[reg_base_c_1], previous1_hi[reg_base_c_1 + 1]);
                                float2 _f2_59 = make_float2(previous1_hi[reg_base_c_1 + 2], previous1_hi[reg_base_c_1 + 3]);
                                float2 _f32x2_mul_b_4 = ((reg_base_c_1 == 0) ? _f2_54 : _f2_55);
                                float2 _mul_f32x2_18;
                                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_18) : "l"(*(const unsigned long long*)&_f2_56), "l"(*(const unsigned long long*)&_f32x2_mul_b_4));
                                float2 prev1_lo01_scaled_f2 = _mul_f32x2_18;
                                float2 _f32x2_mul_b_5 = ((reg_base_c_1 == 0) ? _f2_54 : _f2_55);
                                float2 _mul_f32x2_19;
                                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_19) : "l"(*(const unsigned long long*)&_f2_57), "l"(*(const unsigned long long*)&_f32x2_mul_b_5));
                                float2 prev1_lo23_scaled_f2 = _mul_f32x2_19;
                                float2 _f32x2_mul_b_6 = ((reg_base_c_1 == 0) ? _f2_54 : _f2_55);
                                float2 _mul_f32x2_20;
                                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_20) : "l"(*(const unsigned long long*)&_f2_58), "l"(*(const unsigned long long*)&_f32x2_mul_b_6));
                                float2 prev1_hi01_scaled_f2 = _mul_f32x2_20;
                                float2 _f32x2_mul_b_7 = ((reg_base_c_1 == 0) ? _f2_54 : _f2_55);
                                float2 _mul_f32x2_21;
                                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_21) : "l"(*(const unsigned long long*)&_f2_59), "l"(*(const unsigned long long*)&_f32x2_mul_b_7));
                                float2 prev1_hi23_scaled_f2 = _mul_f32x2_21;
                                previous1_lo[reg_base_c_1] = prev1_lo01_scaled_f2.x;
                                previous1_lo[reg_base_c_1 + 1] = prev1_lo01_scaled_f2.y;
                                previous1_lo[reg_base_c_1 + 2] = prev1_lo23_scaled_f2.x;
                                previous1_lo[reg_base_c_1 + 3] = prev1_lo23_scaled_f2.y;
                                previous1_hi[reg_base_c_1] = prev1_hi01_scaled_f2.x;
                                previous1_hi[reg_base_c_1 + 1] = prev1_hi01_scaled_f2.y;
                                previous1_hi[reg_base_c_1 + 2] = prev1_hi23_scaled_f2.x;
                                previous1_hi[reg_base_c_1 + 3] = prev1_hi23_scaled_f2.y;
                            }
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x256b.x2.b32"
                                " [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "r"(taddr + 112), "r"(*reinterpret_cast<const uint32_t*>(&previous1_lo[0])), "r"(*reinterpret_cast<const uint32_t*>(&previous1_lo[1])), "r"(*reinterpret_cast<const uint32_t*>(&previous1_lo[2])), "r"(*reinterpret_cast<const uint32_t*>(&previous1_lo[3])), "r"(*reinterpret_cast<const uint32_t*>(&previous1_lo[4])), "r"(*reinterpret_cast<const uint32_t*>(&previous1_lo[5])), "r"(*reinterpret_cast<const uint32_t*>(&previous1_lo[6])), "r"(*reinterpret_cast<const uint32_t*>(&previous1_lo[7])));
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x256b.x2.b32"
                                " [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "r"(taddr + 112 + 1048576), "r"(*reinterpret_cast<const uint32_t*>(&previous1_hi[0])), "r"(*reinterpret_cast<const uint32_t*>(&previous1_hi[1])), "r"(*reinterpret_cast<const uint32_t*>(&previous1_hi[2])), "r"(*reinterpret_cast<const uint32_t*>(&previous1_hi[3])), "r"(*reinterpret_cast<const uint32_t*>(&previous1_hi[4])), "r"(*reinterpret_cast<const uint32_t*>(&previous1_hi[5])), "r"(*reinterpret_cast<const uint32_t*>(&previous1_hi[6])), "r"(*reinterpret_cast<const uint32_t*>(&previous1_hi[7])));
                            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        }
                        mbarrier_arrive(o_empty1_addr);
                    }
                }
            }
            mbarrier_wait(scale_ready0_addr, _phase_scale_ready0_0);
            _phase_scale_ready0_0 ^= 1;
            float _tmem_load_2[8];
            tmem_ld_x8(&_tmem_load_2[0], taddr + 32);
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            mbarrier_arrive(stats_empty0_addr);
            mbarrier_wait(scale_ready1_addr, _phase_scale_ready1_0);
            _phase_scale_ready1_0 ^= 1;
            float _tmem_load_3[8];
            tmem_ld_x8(&_tmem_load_3[0], taddr + 64);
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            mbarrier_arrive(stats_empty1_addr);
            mbarrier_wait(o_full0_addr, _phase_o_full0_0);
            _phase_o_full0_0 ^= 1;
            if (num_tiles_c > 1) {
                mbarrier_wait(o_full1_addr, _phase_o_full1_0);
                _phase_o_full1_0 ^= 1;
            }
            asm volatile("tcgen05.fence::after_thread_sync;");
            float out0_lo[8];
            float out0_hi[8];
            float out1_lo[8];
            float out1_hi[8];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(*reinterpret_cast<uint32_t*>(&out0_lo[0])), "=r"(*reinterpret_cast<uint32_t*>(&out0_lo[1])), "=r"(*reinterpret_cast<uint32_t*>(&out0_lo[2])), "=r"(*reinterpret_cast<uint32_t*>(&out0_lo[3])), "=r"(*reinterpret_cast<uint32_t*>(&out0_lo[4])), "=r"(*reinterpret_cast<uint32_t*>(&out0_lo[5])), "=r"(*reinterpret_cast<uint32_t*>(&out0_lo[6])), "=r"(*reinterpret_cast<uint32_t*>(&out0_lo[7]))
                : "r"(taddr + 96));
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(*reinterpret_cast<uint32_t*>(&out0_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&out0_hi[1])), "=r"(*reinterpret_cast<uint32_t*>(&out0_hi[2])), "=r"(*reinterpret_cast<uint32_t*>(&out0_hi[3])), "=r"(*reinterpret_cast<uint32_t*>(&out0_hi[4])), "=r"(*reinterpret_cast<uint32_t*>(&out0_hi[5])), "=r"(*reinterpret_cast<uint32_t*>(&out0_hi[6])), "=r"(*reinterpret_cast<uint32_t*>(&out0_hi[7]))
                : "r"(taddr + 96 + 1048576));
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(*reinterpret_cast<uint32_t*>(&out1_lo[0])), "=r"(*reinterpret_cast<uint32_t*>(&out1_lo[1])), "=r"(*reinterpret_cast<uint32_t*>(&out1_lo[2])), "=r"(*reinterpret_cast<uint32_t*>(&out1_lo[3])), "=r"(*reinterpret_cast<uint32_t*>(&out1_lo[4])), "=r"(*reinterpret_cast<uint32_t*>(&out1_lo[5])), "=r"(*reinterpret_cast<uint32_t*>(&out1_lo[6])), "=r"(*reinterpret_cast<uint32_t*>(&out1_lo[7]))
                : "r"(taddr + 112));
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x2.b32"
                " {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(*reinterpret_cast<uint32_t*>(&out1_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&out1_hi[1])), "=r"(*reinterpret_cast<uint32_t*>(&out1_hi[2])), "=r"(*reinterpret_cast<uint32_t*>(&out1_hi[3])), "=r"(*reinterpret_cast<uint32_t*>(&out1_hi[4])), "=r"(*reinterpret_cast<uint32_t*>(&out1_hi[5])), "=r"(*reinterpret_cast<uint32_t*>(&out1_hi[6])), "=r"(*reinterpret_cast<uint32_t*>(&out1_hi[7]))
                : "r"(taddr + 112 + 1048576));
            int logical_cta_c = ((batch_c * 32 + y_c) * max_num_ctas_q + cta_q_c) * max_num_ctas_kv + cta_kv_c;
            int partial_o_base_c = logical_cta_c * 16 * 128;
            int stats_base_c = logical_cta_c * 16;
            float final_merged_max_c[4];
            float final_local_sum_c[4];
            float final_scale0_c[4];
            float final_scale1_c[4];
            #pragma unroll
            for (int head_local_c_4 = 0; head_local_c_4 < 4; head_local_c_4++) {
                float _max_30 = max_noftz(_tmem_load_2[head_local_c_4 + 4], _tmem_load_3[head_local_c_4 + 4]);
                float merged_max_c = _max_30;
                float _exp2_30 = approx_exp2((_tmem_load_2[head_local_c_4 + 4] - merged_max_c) * bmm1_scale_log2);
                float weight0_c = ((_tmem_load_2[head_local_c_4 + 4] == -3.4028235e+38f) ? 0.0f : _exp2_30);
                float _exp2_31 = approx_exp2((_tmem_load_3[head_local_c_4 + 4] - merged_max_c) * bmm1_scale_log2);
                float weight1_c = ((_tmem_load_3[head_local_c_4 + 4] == -3.4028235e+38f) ? 0.0f : _exp2_31);
                final_merged_max_c[head_local_c_4] = merged_max_c;
                final_local_sum_c[head_local_c_4] = _tmem_load_2[head_local_c_4] * weight0_c + _tmem_load_3[head_local_c_4] * weight1_c;
                final_scale0_c[head_local_c_4] = weight0_c * bmm2_scale / 448.0f;
                final_scale1_c[head_local_c_4] = weight1_c * bmm2_scale / 448.0f;
            }
            #pragma unroll
            for (int head_local_c_5 = 0; head_local_c_5 < 4; head_local_c_5++) {
                float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, final_local_sum_c[head_local_c_5], 16);
                final_local_sum_c[head_local_c_5] = final_local_sum_c[head_local_c_5] + _shfl_xor_6;
                float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, final_local_sum_c[head_local_c_5], 8);
                final_local_sum_c[head_local_c_5] = final_local_sum_c[head_local_c_5] + _shfl_xor_7;
                float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, final_local_sum_c[head_local_c_5], 4);
                final_local_sum_c[head_local_c_5] = final_local_sum_c[head_local_c_5] + _shfl_xor_8;
                if (lane < 4) {
                    corr_stats_reduce[warp_c * 16 + corr_head_base_c + head_local_c_5] = final_local_sum_c[head_local_c_5];
                }
            }
            asm volatile("barrier.sync 3, 128;" ::: "memory");
            float final_global_sum_c[4];
            #pragma unroll
            for (int head_local_c_6 = 0; head_local_c_6 < 4; head_local_c_6++) {
                int head_offset_c = corr_head_base_c + head_local_c_6;
                final_global_sum_c[head_local_c_6] = corr_stats_reduce[head_offset_c] + corr_stats_reduce[16 + head_offset_c] + corr_stats_reduce[32 + head_offset_c] + corr_stats_reduce[48 + head_offset_c];
            }
            float merged_o_c[16];
            #pragma unroll
            for (int reg_base_c_2 = 0; reg_base_c_2 < 8; reg_base_c_2 += 4) {
                const int scale_idx_c = ((reg_base_c_2 == 0) ? 0 : 2);
                float2 _f2_60 = make_float2(final_scale0_c[scale_idx_c], final_scale0_c[scale_idx_c + 1]);
                float2 _f2_61 = make_float2(final_scale1_c[scale_idx_c], final_scale1_c[scale_idx_c + 1]);
                #pragma unroll
                for (int pair_off_c = 0; pair_off_c < 4; pair_off_c += 2) {
                    const int reg_c = reg_base_c_2 + pair_off_c;
                    float2 _f2_62 = make_float2(out0_lo[reg_c], out0_lo[reg_c + 1]);
                    float2 _f2_63 = make_float2(out1_lo[reg_c], out1_lo[reg_c + 1]);
                    float2 _f2_64 = make_float2(out0_hi[reg_c], out0_hi[reg_c + 1]);
                    float2 _f2_65 = make_float2(out1_hi[reg_c], out1_hi[reg_c + 1]);
                    float2 _mul_f32x2_22;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_22) : "l"(*(const unsigned long long*)&_f2_63), "l"(*(const unsigned long long*)&_f2_61));
                    float2 out1_lo_scaled_c = _mul_f32x2_22;
                    float2 _mul_f32x2_23;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_23) : "l"(*(const unsigned long long*)&_f2_65), "l"(*(const unsigned long long*)&_f2_61));
                    float2 out1_hi_scaled_c = _mul_f32x2_23;
                    float2 merged_lo_pair_c = fma_f32x2_rn_ftz(_f2_62, _f2_60, out1_lo_scaled_c);
                    float2 merged_hi_pair_c = fma_f32x2_rn_ftz(_f2_64, _f2_60, out1_hi_scaled_c);
                    merged_o_c[reg_c] = merged_lo_pair_c.x;
                    merged_o_c[reg_c + 1] = merged_lo_pair_c.y;
                    merged_o_c[reg_c + 8] = merged_hi_pair_c.x;
                    merged_o_c[reg_c + 9] = merged_hi_pair_c.y;
                }
            }
            unsigned int regs_o_c[8];
            #pragma unroll
            for (int _lp = 0; _lp < 8; _lp++) {
                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(merged_o_c[_lp*2 + 0], merged_o_c[_lp*2+1 + 0]));
                regs_o_c[_lp] = *(uint32_t*)&_bf2;
            }
            int slice_idx_c = warp_c / 2;
            int warp_idx_in_slice_c = warp_c % 2;
            int mtx_idx_c = lane / 8;
            int mtx_row_idx_c = mtx_idx_c / 2;
            int mtx_col_idx_c = warp_idx_in_slice_c * 4 + mtx_idx_c % 2;
            int thr_row_idx_c = lane % 8;
            #pragma unroll
            for (int stsm_idx_c = 0; stsm_idx_c < 2; stsm_idx_c++) {
                int seg_col_idx_c = mtx_col_idx_c + stsm_idx_c * 2 ^ thr_row_idx_c;
                int stsm_offset_c = slice_idx_c * 16 * 128 + (mtx_row_idx_c * 8 + thr_row_idx_c) * 128 + seg_col_idx_c * 16;
                const int stsm_reg_c = stsm_idx_c * 4;
                const void* _stmatrix_ptr_8 = reinterpret_cast<const void*>(reinterpret_cast<uint8_t*>(smem_o) + stsm_offset_c);
                uint64_t _stmatrix_addr64_8;
                asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(_stmatrix_addr64_8) : "l"(_stmatrix_ptr_8));
                uint32_t _stmatrix_addr_8;
                asm volatile("cvt.u32.u64 %0, %1;" : "=r"(_stmatrix_addr_8) : "l"(_stmatrix_addr64_8));
                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                    :: "r"(_stmatrix_addr_8), "r"(*reinterpret_cast<const uint32_t*>(&regs_o_c[stsm_reg_c])), "r"(*reinterpret_cast<const uint32_t*>(&regs_o_c[stsm_reg_c + 1])), "r"(*reinterpret_cast<const uint32_t*>(&regs_o_c[stsm_reg_c + 2])), "r"(*reinterpret_cast<const uint32_t*>(&regs_o_c[stsm_reg_c + 3]))
                    : "memory");
            }
            asm volatile("barrier.sync 3, 128;" ::: "memory");
            #pragma unroll
            for (int copy_idx_c = 0; copy_idx_c < 2; copy_idx_c++) {
                int copy_base_c = corr_tid * 16 + copy_idx_c * 128 * 16;
                int copy_smem_row_c = copy_base_c / 128;
                int copy_dst_row_c = copy_smem_row_c % 16;
                int copy_dst_col_bytes_c = copy_smem_row_c / 16 * 128 + copy_base_c % 128;
                int copy_smem_offset_c = copy_base_c ^ copy_smem_row_c % 8 * 16;
                unsigned int copy_vec_c[4];
                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&copy_vec_c[0])), "=r"(*reinterpret_cast<uint32_t*>(&copy_vec_c[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&copy_vec_c[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&copy_vec_c[(0) + 3]))
                    : "r"(smem_o_addr + (unsigned int)copy_smem_offset_c));
                reinterpret_cast<int4*>(partial_O + (partial_o_base_c + copy_dst_row_c * 128 + copy_dst_col_bytes_c / 2))[0] = reinterpret_cast<int4*>(copy_vec_c)[0];
            }
            if (corr_tid < 4) {
                #pragma unroll
                for (int head_local_c_7 = 0; head_local_c_7 < 4; head_local_c_7++) {
                    int head_c = 2 * (corr_tid % 4) + 8 * (head_local_c_7 / 2) + head_local_c_7 % 2;
                    partial_stats[(stats_base_c + head_c) * 2] = final_merged_max_c[head_local_c_7];
                    partial_stats[(stats_base_c + head_c) * 2 + 1] = final_global_sum_c[head_local_c_7];
                }
            }
            mbarrier_arrive(o_empty0_addr);
            if (num_tiles_c > 1) {
                mbarrier_arrive(o_empty1_addr);
            }
            int reduction_group_c = (batch_c * 32 + y_c) * max_num_ctas_q + cta_q_c;
            asm volatile("barrier.sync 3, 128;" ::: "memory");
            if (corr_tid == 0) {
                uint32_t _atomic_inc_old_0;
                asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;"
                    : "=r"(_atomic_inc_old_0) : "l"(&completion[reduction_group_c]), "r"(static_cast<uint32_t>(1)) : "memory");
                unsigned int old_ticket_c = _atomic_inc_old_0;
                reduction_ticket[0] = 1 - (int)old_ticket_c;
                if (old_ticket_c == 0) {
                    {
                        unsigned int* _global_wait_zero_p_9 = (reinterpret_cast<unsigned int*>(completion) + (reduction_group_c));
                        unsigned int _global_wait_zero_v_9;
                        do {
                            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(_global_wait_zero_v_9) : "l"(_global_wait_zero_p_9) : "memory");
                        } while (_global_wait_zero_v_9 != 0u);
                    }
                }
            }
            asm volatile("barrier.sync 3, 128;" ::: "memory");
            int reduction_cta_c = reduction_ticket[0];
            int reduction_row_c = reduction_cta_c * 8 + corr_tid / 16;
            int reduction_col_c = corr_tid * 8 % 128;
            int stats_group_base_c = reduction_group_c * max_num_ctas_kv * 16 * 2;
            int stats_row_c = reduction_row_c * 2;
            float _vec_load_0[2];
            {
                float2 _v2_10 = *reinterpret_cast<const float2*>(partial_stats + stats_group_base_c + stats_row_c);
                _vec_load_0[0] = _v2_10.x;
                _vec_load_0[0 + 1] = _v2_10.y;
            }
            float _vec_load_1[2];
            {
                float2 _v2_11 = *reinterpret_cast<const float2*>(partial_stats + stats_group_base_c + 32 + stats_row_c);
                _vec_load_1[0] = _v2_11.x;
                _vec_load_1[0 + 1] = _v2_11.y;
            }
            float max0_c = _vec_load_0[0];
            float sum0_c = _vec_load_0[1];
            float max1_c = _vec_load_1[0];
            float sum1_c = _vec_load_1[1];
            float _max_31 = max_noftz(max0_c, max1_c);
            float global_max_c = _max_31;
            float _exp2_32 = approx_exp2((max0_c - global_max_c) * bmm1_scale_log2);
            float cta_scale0_c = _exp2_32;
            float _exp2_33 = approx_exp2((max1_c - global_max_c) * bmm1_scale_log2);
            float cta_scale1_c = _exp2_33;
            float global_sum_c = sum0_c * cta_scale0_c + sum1_c * cta_scale1_c;
            float normalized_scale_c = 448.0f / global_sum_c;
            int partial_group_base_c = reduction_group_c * max_num_ctas_kv * 16 * 128;
            int partial_row_col_c = reduction_row_c * 128 + reduction_col_c;
            float _vec_load_2[8];
            {
                const uint4* _vptr_12 = reinterpret_cast<const uint4*>(partial_O + (partial_group_base_c + partial_row_col_c) + 0);
                uint4 _vld_12[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_12[_blk] = _vptr_12[_blk];
                    uint32_t* _vpairs_12 = reinterpret_cast<uint32_t*>(&_vld_12[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_2[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_12[_pair]));
                    }
                }
            }
            float _vec_load_3[8];
            {
                const uint4* _vptr_13 = reinterpret_cast<const uint4*>(partial_O + (partial_group_base_c + 2048 + partial_row_col_c) + 0);
                uint4 _vld_13[1];
                #pragma unroll
                for (int _blk = 0; _blk < 1; _blk++) {
                    _vld_13[_blk] = _vptr_13[_blk];
                    uint32_t* _vpairs_13 = reinterpret_cast<uint32_t*>(&_vld_13[_blk]);
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[0]), "=f"((&_vec_load_3[0 + _blk * 8 + _pair * 2])[1])
                            : "r"(_vpairs_13[_pair]));
                    }
                }
            }
            float reduced_c[8];
            float2 _f2_66 = make_float2(cta_scale0_c, cta_scale0_c);
            float2 _f2_67 = make_float2(cta_scale1_c, cta_scale1_c);
            float2 _f2_68 = make_float2(normalized_scale_c, normalized_scale_c);
            #pragma unroll
            for (int elem_c = 0; elem_c < 8; elem_c += 2) {
                float2 _f2_69 = make_float2(_vec_load_2[elem_c], _vec_load_2[elem_c + 1]);
                float2 _f2_70 = make_float2(_vec_load_3[elem_c], _vec_load_3[elem_c + 1]);
                float2 _mul_f32x2_24;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_24) : "l"(*(const unsigned long long*)&_f2_70), "l"(*(const unsigned long long*)&_f2_67));
                float2 merged_pair_c = fma_f32x2_rn_ftz(_f2_69, _f2_66, _mul_f32x2_24);
                float2 _mul_f32x2_25;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_25) : "l"(*(const unsigned long long*)&merged_pair_c), "l"(*(const unsigned long long*)&_f2_68));
                float2 reduced_pair_c = _mul_f32x2_25;
                reduced_c[elem_c] = reduced_pair_c.x;
                reduced_c[elem_c + 1] = reduced_pair_c.y;
            }
            int output_head_c = head_group_c * 16 + reduction_row_c;
            int output_base_c = ((batch_c * q_len + cta_q_c) * 128 + output_head_c) * 512 + hv_partition_c * 128 + reduction_col_c;
            {
                __nv_bfloat162 _pk[4];
                _pk[0] = __floats2bfloat162_rn(reduced_c[0 + 0], reduced_c[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(reduced_c[0 + 2], reduced_c[0 + 3]);
                _pk[2] = __floats2bfloat162_rn(reduced_c[0 + 4], reduced_c[0 + 5]);
                _pk[3] = __floats2bfloat162_rn(reduced_c[0 + 6], reduced_c[0 + 7]);
                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O + output_base_c))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
            }
            asm volatile("barrier.sync 13, 128;" ::: "memory");
            if (warp_c == 0) {
                int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(512));
            }
        }
    // ---- Role: mma_warp ----
    } else if (warp == 12) {
        { // mma_warp_main
            int cta_q_m = blockIdx.x / 2;
            int batch_m = blockIdx.z;
            int seq_len_m = seq_lens[batch_m] - (q_len - 1 - cta_q_m);
            int num_ctas_m = 2;
            int num_steps_m = 2;
            int kv_stage_m = 0;
            unsigned int _phase_q_full_0 = 0;
            mbarrier_wait(q_full_addr, _phase_q_full_0);
            _phase_q_full_0 ^= 1;
            unsigned int _phase_s_empty0_0 = 1;
            mbarrier_wait(s_empty0_addr, _phase_s_empty0_0);
            _phase_s_empty0_0 ^= 1;
            unsigned int _phase_kv_full = 0;
            mbarrier_wait(kv_full_addr + (kv_stage_m) * 8, _phase_kv_full);
            int _mma_a_lo_0 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (kv_stage_m) * 1024);
            int _mma_b_lo_0 = make_warp_uniform((((smem_q0_addr) >> 4) & 0x3FFF) + (0) * 640);
            {
                uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
                uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                if (elect_sync()) {
                    tcgen05_mma_f8f6f4(tmem_tmem, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134479888, 0);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f8f6f4(tmem_tmem, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134479888, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f8f6f4(tmem_tmem, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134479888, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f8f6f4(tmem_tmem, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134479888, 1);
                }
            }
            elect_commit(kv_empty_addr + (kv_stage_m) * 8);
            kv_stage_m += 1;
            if (kv_stage_m == 9) { kv_stage_m = 0; _phase_kv_full ^= 1; }
            mbarrier_wait(kv_full_addr + (kv_stage_m) * 8, _phase_kv_full);
            int _mma_a_lo_1 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (kv_stage_m) * 1024);
            int _mma_b_lo_1 = make_warp_uniform((((smem_q1_addr) >> 4) & 0x3FFF) + (0) * 640);
            {
                uint64_t _mma_ss_a_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_1);
                uint64_t _mma_ss_b_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_1);
                if (elect_sync()) {
                    tcgen05_mma_f8f6f4(tmem_tmem, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134479888, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f8f6f4(tmem_tmem, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134479888, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f8f6f4(tmem_tmem, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134479888, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f8f6f4(tmem_tmem, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134479888, 1);
                }
            }
            elect_commit(kv_empty_addr + (kv_stage_m) * 8);
            kv_stage_m += 1;
            if (kv_stage_m == 9) { kv_stage_m = 0; _phase_kv_full ^= 1; }
            mbarrier_wait(kv_full_addr + (kv_stage_m) * 8, _phase_kv_full);
            int _mma_a_lo_2 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (kv_stage_m) * 1024);
            int _mma_b_lo_2 = make_warp_uniform((((smem_q2_addr) >> 4) & 0x3FFF) + (0) * 640);
            {
                uint64_t _mma_ss_a_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_2);
                uint64_t _mma_ss_b_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_2);
                if (elect_sync()) {
                    tcgen05_mma_f8f6f4(tmem_tmem, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134479888, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f8f6f4(tmem_tmem, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134479888, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f8f6f4(tmem_tmem, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134479888, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_2, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f8f6f4(tmem_tmem, _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134479888, 1);
                }
            }
            elect_commit(kv_empty_addr + (kv_stage_m) * 8);
            kv_stage_m += 1;
            if (kv_stage_m == 9) { kv_stage_m = 0; _phase_kv_full ^= 1; }
            mbarrier_wait(kv_full_addr + (kv_stage_m) * 8, _phase_kv_full);
            int _mma_a_lo_3 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (kv_stage_m) * 1024);
            int _mma_b_lo_3 = make_warp_uniform((((smem_q3_addr) >> 4) & 0x3FFF) + (0) * 640);
            {
                uint64_t _mma_ss_a_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_3);
                uint64_t _mma_ss_b_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_3);
                if (elect_sync()) {
                    tcgen05_mma_f8f6f4(tmem_tmem, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134479888, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f8f6f4(tmem_tmem, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134479888, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f8f6f4(tmem_tmem, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134479888, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_3, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f8f6f4(tmem_tmem, _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134479888, 1);
                }
            }
            elect_commit(kv_empty_addr + (kv_stage_m) * 8);
            kv_stage_m += 1;
            if (kv_stage_m == 9) { kv_stage_m = 0; _phase_kv_full ^= 1; }
            mbarrier_wait(kv_full_addr + (kv_stage_m) * 8, _phase_kv_full);
            int _mma_a_lo_4 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (kv_stage_m) * 1024);
            int _mma_b_lo_4 = make_warp_uniform((((smem_q4_addr) >> 4) & 0x3FFF) + (0) * 640);
            {
                uint64_t _mma_ss_a_desc_4 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_4);
                uint64_t _mma_ss_b_desc_4 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_4);
                if (elect_sync()) {
                    tcgen05_mma_f8f6f4(tmem_tmem, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134479888, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_4, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f8f6f4(tmem_tmem, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134479888, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_4, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f8f6f4(tmem_tmem, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134479888, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_4, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_4, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f8f6f4(tmem_tmem, _mma_ss_a_desc_4, _mma_ss_b_desc_4, 134479888, 1);
                }
            }
            elect_commit(kv_empty_addr + (kv_stage_m) * 8);
            kv_stage_m += 1;
            if (kv_stage_m == 9) { kv_stage_m = 0; _phase_kv_full ^= 1; }
            elect_commit(s_full0_addr);
            unsigned int _phase_s_empty1_0 = 1;
            mbarrier_wait(s_empty1_addr, _phase_s_empty1_0);
            _phase_s_empty1_0 ^= 1;
            mbarrier_wait(kv_full_addr + (kv_stage_m) * 8, _phase_kv_full);
            int _mma_a_lo_5 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (kv_stage_m) * 1024);
            int _mma_b_lo_5 = make_warp_uniform((((smem_q0_addr) >> 4) & 0x3FFF) + (0) * 640);
            {
                uint64_t _mma_ss_a_desc_5 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_5);
                uint64_t _mma_ss_b_desc_5 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_5);
                if (elect_sync()) {
                    tcgen05_mma_f8f6f4((tmem_tmem + (16)), _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134479888, 0);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_5, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f8f6f4((tmem_tmem + (16)), _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134479888, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_5, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f8f6f4((tmem_tmem + (16)), _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134479888, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_5, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_5, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f8f6f4((tmem_tmem + (16)), _mma_ss_a_desc_5, _mma_ss_b_desc_5, 134479888, 1);
                }
            }
            elect_commit(kv_empty_addr + (kv_stage_m) * 8);
            kv_stage_m += 1;
            if (kv_stage_m == 9) { kv_stage_m = 0; _phase_kv_full ^= 1; }
            mbarrier_wait(kv_full_addr + (kv_stage_m) * 8, _phase_kv_full);
            int _mma_a_lo_6 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (kv_stage_m) * 1024);
            int _mma_b_lo_6 = make_warp_uniform((((smem_q1_addr) >> 4) & 0x3FFF) + (0) * 640);
            {
                uint64_t _mma_ss_a_desc_6 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_6);
                uint64_t _mma_ss_b_desc_6 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_6);
                if (elect_sync()) {
                    tcgen05_mma_f8f6f4((tmem_tmem + (16)), _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134479888, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_6, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f8f6f4((tmem_tmem + (16)), _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134479888, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_6, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f8f6f4((tmem_tmem + (16)), _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134479888, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_6, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_6, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f8f6f4((tmem_tmem + (16)), _mma_ss_a_desc_6, _mma_ss_b_desc_6, 134479888, 1);
                }
            }
            elect_commit(kv_empty_addr + (kv_stage_m) * 8);
            kv_stage_m += 1;
            if (kv_stage_m == 9) { kv_stage_m = 0; _phase_kv_full ^= 1; }
            mbarrier_wait(kv_full_addr + (kv_stage_m) * 8, _phase_kv_full);
            int _mma_a_lo_7 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (kv_stage_m) * 1024);
            int _mma_b_lo_7 = make_warp_uniform((((smem_q2_addr) >> 4) & 0x3FFF) + (0) * 640);
            {
                uint64_t _mma_ss_a_desc_7 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_7);
                uint64_t _mma_ss_b_desc_7 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_7);
                if (elect_sync()) {
                    tcgen05_mma_f8f6f4((tmem_tmem + (16)), _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134479888, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_7, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f8f6f4((tmem_tmem + (16)), _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134479888, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_7, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f8f6f4((tmem_tmem + (16)), _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134479888, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_7, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_7, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f8f6f4((tmem_tmem + (16)), _mma_ss_a_desc_7, _mma_ss_b_desc_7, 134479888, 1);
                }
            }
            elect_commit(kv_empty_addr + (kv_stage_m) * 8);
            kv_stage_m += 1;
            if (kv_stage_m == 9) { kv_stage_m = 0; _phase_kv_full ^= 1; }
            mbarrier_wait(kv_full_addr + (kv_stage_m) * 8, _phase_kv_full);
            int _mma_a_lo_8 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (kv_stage_m) * 1024);
            int _mma_b_lo_8 = make_warp_uniform((((smem_q3_addr) >> 4) & 0x3FFF) + (0) * 640);
            {
                uint64_t _mma_ss_a_desc_8 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_8);
                uint64_t _mma_ss_b_desc_8 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_8);
                if (elect_sync()) {
                    tcgen05_mma_f8f6f4((tmem_tmem + (16)), _mma_ss_a_desc_8, _mma_ss_b_desc_8, 134479888, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_8, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_8, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f8f6f4((tmem_tmem + (16)), _mma_ss_a_desc_8, _mma_ss_b_desc_8, 134479888, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_8, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_8, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f8f6f4((tmem_tmem + (16)), _mma_ss_a_desc_8, _mma_ss_b_desc_8, 134479888, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_8, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_8, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f8f6f4((tmem_tmem + (16)), _mma_ss_a_desc_8, _mma_ss_b_desc_8, 134479888, 1);
                }
            }
            elect_commit(kv_empty_addr + (kv_stage_m) * 8);
            kv_stage_m += 1;
            if (kv_stage_m == 9) { kv_stage_m = 0; _phase_kv_full ^= 1; }
            mbarrier_wait(kv_full_addr + (kv_stage_m) * 8, _phase_kv_full);
            int _mma_a_lo_9 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (kv_stage_m) * 1024);
            int _mma_b_lo_9 = make_warp_uniform((((smem_q4_addr) >> 4) & 0x3FFF) + (0) * 640);
            {
                uint64_t _mma_ss_a_desc_9 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_9);
                uint64_t _mma_ss_b_desc_9 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_9);
                if (elect_sync()) {
                    tcgen05_mma_f8f6f4((tmem_tmem + (16)), _mma_ss_a_desc_9, _mma_ss_b_desc_9, 134479888, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_9, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_9, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f8f6f4((tmem_tmem + (16)), _mma_ss_a_desc_9, _mma_ss_b_desc_9, 134479888, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_9, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_9, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f8f6f4((tmem_tmem + (16)), _mma_ss_a_desc_9, _mma_ss_b_desc_9, 134479888, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_9, 2U);
                incr_smem_desc_lo(_mma_ss_b_desc_9, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f8f6f4((tmem_tmem + (16)), _mma_ss_a_desc_9, _mma_ss_b_desc_9, 134479888, 1);
                }
            }
            elect_commit(kv_empty_addr + (kv_stage_m) * 8);
            kv_stage_m += 1;
            if (kv_stage_m == 9) { kv_stage_m = 0; _phase_kv_full ^= 1; }
            elect_commit(s_full1_addr);
            unsigned int _phase_o_empty0_0 = 1;
            unsigned int _phase_o_empty1_0 = 1;
            #pragma unroll 1
            for (int pair_m = 0; pair_m < num_steps_m - 1; pair_m++) {
                mbarrier_wait(s_empty0_addr, _phase_s_empty0_0);
                _phase_s_empty0_0 ^= 1;
                mbarrier_wait(o_empty0_addr, _phase_o_empty0_0);
                _phase_o_empty0_0 ^= 1;
                mbarrier_wait(kv_full_addr + (kv_stage_m) * 8, _phase_kv_full);
                int _mma_a_lo_10 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (kv_stage_m) * 1024);
                int _mma_b_lo_10 = make_warp_uniform((((smem_p0_addr) >> 4) & 0x3FFF) | 0x800000);
                {
                    uint64_t _mma_ss_a_desc_10 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_10);
                    uint64_t _mma_ss_b_desc_10 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_10);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_tmem + (96)), _mma_ss_a_desc_10, _mma_ss_b_desc_10, 134512656, ((pair_m == 0) ? 0 : 1));
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_10, 256U);
                    incr_smem_desc_lo(_mma_ss_b_desc_10, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_tmem + (96)), _mma_ss_a_desc_10, _mma_ss_b_desc_10, 134512656, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_10, 256U);
                    incr_smem_desc_lo(_mma_ss_b_desc_10, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_tmem + (96)), _mma_ss_a_desc_10, _mma_ss_b_desc_10, 134512656, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_10, 256U);
                    incr_smem_desc_lo(_mma_ss_b_desc_10, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_tmem + (96)), _mma_ss_a_desc_10, _mma_ss_b_desc_10, 134512656, 1);
                    }
                }
                elect_commit2(kv_empty_addr + (kv_stage_m) * 8, o_full0_addr);
                kv_stage_m += 1;
                if (kv_stage_m == 9) { kv_stage_m = 0; _phase_kv_full ^= 1; }
                mbarrier_wait(kv_full_addr + (kv_stage_m) * 8, _phase_kv_full);
                int _mma_a_lo_11 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (kv_stage_m) * 1024);
                int _mma_b_lo_11 = make_warp_uniform((((smem_q0_addr) >> 4) & 0x3FFF) + (0) * 640);
                {
                    uint64_t _mma_ss_a_desc_11 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_11);
                    uint64_t _mma_ss_b_desc_11 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_11);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_tmem, _mma_ss_a_desc_11, _mma_ss_b_desc_11, 134479888, 0);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_11, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_11, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_tmem, _mma_ss_a_desc_11, _mma_ss_b_desc_11, 134479888, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_11, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_11, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_tmem, _mma_ss_a_desc_11, _mma_ss_b_desc_11, 134479888, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_11, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_11, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_tmem, _mma_ss_a_desc_11, _mma_ss_b_desc_11, 134479888, 1);
                    }
                }
                elect_commit(kv_empty_addr + (kv_stage_m) * 8);
                kv_stage_m += 1;
                if (kv_stage_m == 9) { kv_stage_m = 0; _phase_kv_full ^= 1; }
                mbarrier_wait(kv_full_addr + (kv_stage_m) * 8, _phase_kv_full);
                int _mma_a_lo_12 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (kv_stage_m) * 1024);
                int _mma_b_lo_12 = make_warp_uniform((((smem_q1_addr) >> 4) & 0x3FFF) + (0) * 640);
                {
                    uint64_t _mma_ss_a_desc_12 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_12);
                    uint64_t _mma_ss_b_desc_12 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_12);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_tmem, _mma_ss_a_desc_12, _mma_ss_b_desc_12, 134479888, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_12, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_12, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_tmem, _mma_ss_a_desc_12, _mma_ss_b_desc_12, 134479888, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_12, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_12, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_tmem, _mma_ss_a_desc_12, _mma_ss_b_desc_12, 134479888, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_12, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_12, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_tmem, _mma_ss_a_desc_12, _mma_ss_b_desc_12, 134479888, 1);
                    }
                }
                elect_commit(kv_empty_addr + (kv_stage_m) * 8);
                kv_stage_m += 1;
                if (kv_stage_m == 9) { kv_stage_m = 0; _phase_kv_full ^= 1; }
                mbarrier_wait(kv_full_addr + (kv_stage_m) * 8, _phase_kv_full);
                int _mma_a_lo_13 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (kv_stage_m) * 1024);
                int _mma_b_lo_13 = make_warp_uniform((((smem_q2_addr) >> 4) & 0x3FFF) + (0) * 640);
                {
                    uint64_t _mma_ss_a_desc_13 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_13);
                    uint64_t _mma_ss_b_desc_13 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_13);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_tmem, _mma_ss_a_desc_13, _mma_ss_b_desc_13, 134479888, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_13, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_13, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_tmem, _mma_ss_a_desc_13, _mma_ss_b_desc_13, 134479888, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_13, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_13, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_tmem, _mma_ss_a_desc_13, _mma_ss_b_desc_13, 134479888, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_13, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_13, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_tmem, _mma_ss_a_desc_13, _mma_ss_b_desc_13, 134479888, 1);
                    }
                }
                elect_commit(kv_empty_addr + (kv_stage_m) * 8);
                kv_stage_m += 1;
                if (kv_stage_m == 9) { kv_stage_m = 0; _phase_kv_full ^= 1; }
                mbarrier_wait(kv_full_addr + (kv_stage_m) * 8, _phase_kv_full);
                int _mma_a_lo_14 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (kv_stage_m) * 1024);
                int _mma_b_lo_14 = make_warp_uniform((((smem_q3_addr) >> 4) & 0x3FFF) + (0) * 640);
                {
                    uint64_t _mma_ss_a_desc_14 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_14);
                    uint64_t _mma_ss_b_desc_14 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_14);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_tmem, _mma_ss_a_desc_14, _mma_ss_b_desc_14, 134479888, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_14, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_14, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_tmem, _mma_ss_a_desc_14, _mma_ss_b_desc_14, 134479888, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_14, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_14, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_tmem, _mma_ss_a_desc_14, _mma_ss_b_desc_14, 134479888, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_14, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_14, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_tmem, _mma_ss_a_desc_14, _mma_ss_b_desc_14, 134479888, 1);
                    }
                }
                elect_commit(kv_empty_addr + (kv_stage_m) * 8);
                kv_stage_m += 1;
                if (kv_stage_m == 9) { kv_stage_m = 0; _phase_kv_full ^= 1; }
                mbarrier_wait(kv_full_addr + (kv_stage_m) * 8, _phase_kv_full);
                int _mma_a_lo_15 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (kv_stage_m) * 1024);
                int _mma_b_lo_15 = make_warp_uniform((((smem_q4_addr) >> 4) & 0x3FFF) + (0) * 640);
                {
                    uint64_t _mma_ss_a_desc_15 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_15);
                    uint64_t _mma_ss_b_desc_15 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_15);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_tmem, _mma_ss_a_desc_15, _mma_ss_b_desc_15, 134479888, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_15, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_15, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_tmem, _mma_ss_a_desc_15, _mma_ss_b_desc_15, 134479888, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_15, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_15, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_tmem, _mma_ss_a_desc_15, _mma_ss_b_desc_15, 134479888, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_15, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_15, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4(tmem_tmem, _mma_ss_a_desc_15, _mma_ss_b_desc_15, 134479888, 1);
                    }
                }
                elect_commit(kv_empty_addr + (kv_stage_m) * 8);
                kv_stage_m += 1;
                if (kv_stage_m == 9) { kv_stage_m = 0; _phase_kv_full ^= 1; }
                elect_commit(s_full0_addr);
                mbarrier_wait(s_empty1_addr, _phase_s_empty1_0);
                _phase_s_empty1_0 ^= 1;
                mbarrier_wait(o_empty1_addr, _phase_o_empty1_0);
                _phase_o_empty1_0 ^= 1;
                mbarrier_wait(kv_full_addr + (kv_stage_m) * 8, _phase_kv_full);
                int _mma_a_lo_16 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (kv_stage_m) * 1024);
                int _mma_b_lo_16 = make_warp_uniform((((smem_p1_addr) >> 4) & 0x3FFF) | 0x800000);
                {
                    uint64_t _mma_ss_a_desc_16 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_16);
                    uint64_t _mma_ss_b_desc_16 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_16);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_tmem + (112)), _mma_ss_a_desc_16, _mma_ss_b_desc_16, 134512656, ((pair_m == 0) ? 0 : 1));
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_16, 256U);
                    incr_smem_desc_lo(_mma_ss_b_desc_16, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_tmem + (112)), _mma_ss_a_desc_16, _mma_ss_b_desc_16, 134512656, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_16, 256U);
                    incr_smem_desc_lo(_mma_ss_b_desc_16, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_tmem + (112)), _mma_ss_a_desc_16, _mma_ss_b_desc_16, 134512656, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_16, 256U);
                    incr_smem_desc_lo(_mma_ss_b_desc_16, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_tmem + (112)), _mma_ss_a_desc_16, _mma_ss_b_desc_16, 134512656, 1);
                    }
                }
                elect_commit2(kv_empty_addr + (kv_stage_m) * 8, o_full1_addr);
                kv_stage_m += 1;
                if (kv_stage_m == 9) { kv_stage_m = 0; _phase_kv_full ^= 1; }
                mbarrier_wait(kv_full_addr + (kv_stage_m) * 8, _phase_kv_full);
                int _mma_a_lo_17 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (kv_stage_m) * 1024);
                int _mma_b_lo_17 = make_warp_uniform((((smem_q0_addr) >> 4) & 0x3FFF) + (0) * 640);
                {
                    uint64_t _mma_ss_a_desc_17 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_17);
                    uint64_t _mma_ss_b_desc_17 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_17);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_tmem + (16)), _mma_ss_a_desc_17, _mma_ss_b_desc_17, 134479888, 0);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_17, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_17, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_tmem + (16)), _mma_ss_a_desc_17, _mma_ss_b_desc_17, 134479888, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_17, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_17, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_tmem + (16)), _mma_ss_a_desc_17, _mma_ss_b_desc_17, 134479888, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_17, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_17, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_tmem + (16)), _mma_ss_a_desc_17, _mma_ss_b_desc_17, 134479888, 1);
                    }
                }
                elect_commit(kv_empty_addr + (kv_stage_m) * 8);
                kv_stage_m += 1;
                if (kv_stage_m == 9) { kv_stage_m = 0; _phase_kv_full ^= 1; }
                mbarrier_wait(kv_full_addr + (kv_stage_m) * 8, _phase_kv_full);
                int _mma_a_lo_18 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (kv_stage_m) * 1024);
                int _mma_b_lo_18 = make_warp_uniform((((smem_q1_addr) >> 4) & 0x3FFF) + (0) * 640);
                {
                    uint64_t _mma_ss_a_desc_18 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_18);
                    uint64_t _mma_ss_b_desc_18 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_18);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_tmem + (16)), _mma_ss_a_desc_18, _mma_ss_b_desc_18, 134479888, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_18, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_18, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_tmem + (16)), _mma_ss_a_desc_18, _mma_ss_b_desc_18, 134479888, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_18, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_18, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_tmem + (16)), _mma_ss_a_desc_18, _mma_ss_b_desc_18, 134479888, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_18, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_18, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_tmem + (16)), _mma_ss_a_desc_18, _mma_ss_b_desc_18, 134479888, 1);
                    }
                }
                elect_commit(kv_empty_addr + (kv_stage_m) * 8);
                kv_stage_m += 1;
                if (kv_stage_m == 9) { kv_stage_m = 0; _phase_kv_full ^= 1; }
                mbarrier_wait(kv_full_addr + (kv_stage_m) * 8, _phase_kv_full);
                int _mma_a_lo_19 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (kv_stage_m) * 1024);
                int _mma_b_lo_19 = make_warp_uniform((((smem_q2_addr) >> 4) & 0x3FFF) + (0) * 640);
                {
                    uint64_t _mma_ss_a_desc_19 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_19);
                    uint64_t _mma_ss_b_desc_19 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_19);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_tmem + (16)), _mma_ss_a_desc_19, _mma_ss_b_desc_19, 134479888, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_19, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_19, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_tmem + (16)), _mma_ss_a_desc_19, _mma_ss_b_desc_19, 134479888, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_19, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_19, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_tmem + (16)), _mma_ss_a_desc_19, _mma_ss_b_desc_19, 134479888, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_19, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_19, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_tmem + (16)), _mma_ss_a_desc_19, _mma_ss_b_desc_19, 134479888, 1);
                    }
                }
                elect_commit(kv_empty_addr + (kv_stage_m) * 8);
                kv_stage_m += 1;
                if (kv_stage_m == 9) { kv_stage_m = 0; _phase_kv_full ^= 1; }
                mbarrier_wait(kv_full_addr + (kv_stage_m) * 8, _phase_kv_full);
                int _mma_a_lo_20 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (kv_stage_m) * 1024);
                int _mma_b_lo_20 = make_warp_uniform((((smem_q3_addr) >> 4) & 0x3FFF) + (0) * 640);
                {
                    uint64_t _mma_ss_a_desc_20 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_20);
                    uint64_t _mma_ss_b_desc_20 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_20);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_tmem + (16)), _mma_ss_a_desc_20, _mma_ss_b_desc_20, 134479888, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_20, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_20, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_tmem + (16)), _mma_ss_a_desc_20, _mma_ss_b_desc_20, 134479888, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_20, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_20, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_tmem + (16)), _mma_ss_a_desc_20, _mma_ss_b_desc_20, 134479888, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_20, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_20, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_tmem + (16)), _mma_ss_a_desc_20, _mma_ss_b_desc_20, 134479888, 1);
                    }
                }
                elect_commit(kv_empty_addr + (kv_stage_m) * 8);
                kv_stage_m += 1;
                if (kv_stage_m == 9) { kv_stage_m = 0; _phase_kv_full ^= 1; }
                mbarrier_wait(kv_full_addr + (kv_stage_m) * 8, _phase_kv_full);
                int _mma_a_lo_21 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (kv_stage_m) * 1024);
                int _mma_b_lo_21 = make_warp_uniform((((smem_q4_addr) >> 4) & 0x3FFF) + (0) * 640);
                {
                    uint64_t _mma_ss_a_desc_21 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_21);
                    uint64_t _mma_ss_b_desc_21 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_21);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_tmem + (16)), _mma_ss_a_desc_21, _mma_ss_b_desc_21, 134479888, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_21, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_21, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_tmem + (16)), _mma_ss_a_desc_21, _mma_ss_b_desc_21, 134479888, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_21, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_21, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_tmem + (16)), _mma_ss_a_desc_21, _mma_ss_b_desc_21, 134479888, 1);
                    }
                    incr_smem_desc_lo(_mma_ss_a_desc_21, 2U);
                    incr_smem_desc_lo(_mma_ss_b_desc_21, 2U);
                    if (elect_sync()) {
                        tcgen05_mma_f8f6f4((tmem_tmem + (16)), _mma_ss_a_desc_21, _mma_ss_b_desc_21, 134479888, 1);
                    }
                }
                elect_commit(kv_empty_addr + (kv_stage_m) * 8);
                kv_stage_m += 1;
                if (kv_stage_m == 9) { kv_stage_m = 0; _phase_kv_full ^= 1; }
                elect_commit(s_full1_addr);
            }
            mbarrier_wait(s_empty0_addr, _phase_s_empty0_0);
            _phase_s_empty0_0 ^= 1;
            mbarrier_wait(o_empty0_addr, _phase_o_empty0_0);
            _phase_o_empty0_0 ^= 1;
            mbarrier_wait(kv_full_addr + (kv_stage_m) * 8, _phase_kv_full);
            int _mma_a_lo_22 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (kv_stage_m) * 1024);
            int _mma_b_lo_22 = make_warp_uniform((((smem_p0_addr) >> 4) & 0x3FFF) | 0x800000);
            {
                uint64_t _mma_ss_a_desc_22 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_22);
                uint64_t _mma_ss_b_desc_22 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_22);
                if (elect_sync()) {
                    tcgen05_mma_f8f6f4((tmem_tmem + (96)), _mma_ss_a_desc_22, _mma_ss_b_desc_22, 134512656, ((num_steps_m == 1) ? 0 : 1));
                }
                incr_smem_desc_lo(_mma_ss_a_desc_22, 256U);
                incr_smem_desc_lo(_mma_ss_b_desc_22, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f8f6f4((tmem_tmem + (96)), _mma_ss_a_desc_22, _mma_ss_b_desc_22, 134512656, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_22, 256U);
                incr_smem_desc_lo(_mma_ss_b_desc_22, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f8f6f4((tmem_tmem + (96)), _mma_ss_a_desc_22, _mma_ss_b_desc_22, 134512656, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_22, 256U);
                incr_smem_desc_lo(_mma_ss_b_desc_22, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f8f6f4((tmem_tmem + (96)), _mma_ss_a_desc_22, _mma_ss_b_desc_22, 134512656, 1);
                }
            }
            elect_commit2(kv_empty_addr + (kv_stage_m) * 8, o_full0_addr);
            kv_stage_m += 1;
            if (kv_stage_m == 9) { kv_stage_m = 0; _phase_kv_full ^= 1; }
            mbarrier_wait(s_empty1_addr, _phase_s_empty1_0);
            _phase_s_empty1_0 ^= 1;
            mbarrier_wait(o_empty1_addr, _phase_o_empty1_0);
            _phase_o_empty1_0 ^= 1;
            mbarrier_wait(kv_full_addr + (kv_stage_m) * 8, _phase_kv_full);
            int _mma_a_lo_23 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (kv_stage_m) * 1024);
            int _mma_b_lo_23 = make_warp_uniform((((smem_p1_addr) >> 4) & 0x3FFF) | 0x800000);
            {
                uint64_t _mma_ss_a_desc_23 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_23);
                uint64_t _mma_ss_b_desc_23 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_23);
                if (elect_sync()) {
                    tcgen05_mma_f8f6f4((tmem_tmem + (112)), _mma_ss_a_desc_23, _mma_ss_b_desc_23, 134512656, ((num_steps_m == 1) ? 0 : 1));
                }
                incr_smem_desc_lo(_mma_ss_a_desc_23, 256U);
                incr_smem_desc_lo(_mma_ss_b_desc_23, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f8f6f4((tmem_tmem + (112)), _mma_ss_a_desc_23, _mma_ss_b_desc_23, 134512656, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_23, 256U);
                incr_smem_desc_lo(_mma_ss_b_desc_23, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f8f6f4((tmem_tmem + (112)), _mma_ss_a_desc_23, _mma_ss_b_desc_23, 134512656, 1);
                }
                incr_smem_desc_lo(_mma_ss_a_desc_23, 256U);
                incr_smem_desc_lo(_mma_ss_b_desc_23, 2U);
                if (elect_sync()) {
                    tcgen05_mma_f8f6f4((tmem_tmem + (112)), _mma_ss_a_desc_23, _mma_ss_b_desc_23, 134512656, 1);
                }
            }
            elect_commit2(kv_empty_addr + (kv_stage_m) * 8, o_full1_addr);
            kv_stage_m += 1;
            if (kv_stage_m == 9) { kv_stage_m = 0; _phase_kv_full ^= 1; }
            elect_commit(q_empty_addr);
        }
    // ---- Role: page_warp ----
    } else if (warp == 13) {
        { // page_warp_main
            int cta_q_p = blockIdx.x / 2;
            int cta_kv_p = blockIdx.x % 2;
            int batch_p = blockIdx.z;
            int seq_len_p = seq_lens[batch_p] - (q_len - 1 - cta_q_p);
            int num_ctas_p = 2;
            int num_steps_p = 2;
            int num_tiles_p = num_steps_p * 2;
            int max_page_p = (seq_len_p + 32 - 1) / 32 - 1;
            int page_stage_p = 0;
            unsigned int _phase_page_empty = 1;
            #pragma unroll 1
            for (int tile_p = 0; tile_p < num_tiles_p; tile_p++) {
                mbarrier_wait(page_empty_addr + (page_stage_p) * 8, _phase_page_empty);
                int global_page_p = (cta_kv_p * num_tiles_p + tile_p) * 4 + lane;
                if (global_page_p > max_page_p) {
                    global_page_p = max_page_p;
                }
                asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4;"
                    :: "r"(page_offsets_addr + (unsigned int)(page_stage_p * 128) + (unsigned int)(lane * 4)), "l"(page_table + (batch_p * page_table_stride + global_page_p)));
                asm volatile(
                    "{\n\t"
                    "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n\t"
                    "}"
                    :: "r"(page_full_addr + (page_stage_p) * 8) : "memory");
                page_stage_p += 1;
                if (page_stage_p == 6) { page_stage_p = 0; _phase_page_empty ^= 1; }
            }
        }
    // ---- Role: padding_warp ----
    } else if (warp == 14) {
        // idle — no tasks assigned
    // ---- Role: load_warp ----
    } else if (warp == 15) {
        { // load_warp_main
            int cta_q_l = blockIdx.x / 2;
            int cta_kv_l = blockIdx.x % 2;
            int y_l = blockIdx.y;
            int head_group_l = y_l / 4;
            int hv_partition_l = y_l % 4;
            int batch_l = blockIdx.z;
            int seq_len_l = seq_lens[batch_l] - (q_len - 1 - cta_q_l);
            int num_ctas_l = 2;
            int num_steps_l = 2;
            int num_tiles_l = num_steps_l * 2;
            int q_stage_l = 0;
            int kv_stage_l = 0;
            int page_stage_l = 0;
            unsigned int _phase_q_empty = 1;
            mbarrier_wait(q_empty_addr + (q_stage_l) * 8, _phase_q_empty);
            if (elect_sync()) {
                int q_row_l = (batch_l * q_len + cta_q_l) * 128 + head_group_l * 16;
                mbarrier_arrive_expect_tx(q_full_addr + (q_stage_l) * 8, 10240);
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1, {%2, %3}], [%4], %5;"
                    :: "r"(smem_q0_addr + (unsigned int)(q_stage_l * 10240)), "l"((&Q_map)), "r"(0), "r"(q_row_l),
                       "r"(q_full_addr + (q_stage_l) * 8), "l"(0x14F0000000000000ULL) : "memory");
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1, {%2, %3}], [%4], %5;"
                    :: "r"(smem_q1_addr + (unsigned int)(q_stage_l * 10240)), "l"((&Q_map)), "r"(128), "r"(q_row_l),
                       "r"(q_full_addr + (q_stage_l) * 8), "l"(0x14F0000000000000ULL) : "memory");
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1, {%2, %3}], [%4], %5;"
                    :: "r"(smem_q2_addr + (unsigned int)(q_stage_l * 10240)), "l"((&Q_map)), "r"(256), "r"(q_row_l),
                       "r"(q_full_addr + (q_stage_l) * 8), "l"(0x14F0000000000000ULL) : "memory");
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1, {%2, %3}], [%4], %5;"
                    :: "r"(smem_q3_addr + (unsigned int)(q_stage_l * 10240)), "l"((&Q_map)), "r"(384), "r"(q_row_l),
                       "r"(q_full_addr + (q_stage_l) * 8), "l"(0x14F0000000000000ULL) : "memory");
                asm volatile(
                    "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                    " [%0], [%1, {%2, %3}], [%4], %5;"
                    :: "r"(smem_q4_addr + (unsigned int)(q_stage_l * 10240)), "l"((&Q_map)), "r"(512), "r"(q_row_l),
                       "r"(q_full_addr + (q_stage_l) * 8), "l"(0x14F0000000000000ULL) : "memory");
            }
            unsigned int _phase_kv_empty = 1;
            #pragma unroll 1
            for (int aligned_tile_l = 0; aligned_tile_l < num_tiles_l; aligned_tile_l++) {
                mbarrier_wait(page_full_addr + (aligned_tile_l) * 8, 0);
                #pragma unroll
                for (int head_stage = 0; head_stage < 5; head_stage++) {
                    mbarrier_wait(kv_empty_addr + (kv_stage_l) * 8, _phase_kv_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(kv_full_addr + (kv_stage_l) * 8, 16384);
                        #pragma unroll
                        for (int page = 0; page < 4; page++) {
                            int physical_page = page_offsets[aligned_tile_l * 32 + page];
                            asm volatile(
                                "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes.L2::cache_hint"
                                " [%0], [%1, {%2, %3}], [%4], %5;"
                                :: "r"(smem_kv_addr + (unsigned int)(kv_stage_l * 16384) + (unsigned int)(page * 32 * 128)), "l"((&K_map)), "r"(head_stage * 128), "r"(physical_page * 32),
                                   "r"(kv_full_addr + (kv_stage_l) * 8), "l"(0x14F0000000000000ULL) : "memory");
                        }
                    }
                    kv_stage_l += 1;
                    if (kv_stage_l == 9) { kv_stage_l = 0; _phase_kv_empty ^= 1; }
                }
                if (aligned_tile_l > 0) {
                    mbarrier_wait(kv_empty_addr + (kv_stage_l) * 8, _phase_kv_empty);
                    if (elect_sync()) {
                        mbarrier_arrive_expect_tx(kv_full_addr + (kv_stage_l) * 8, 16384);
                        #pragma unroll
                        for (int page_1 = 0; page_1 < 4; page_1++) {
                            int physical_page_1 = page_offsets[(aligned_tile_l - 1) * 32 + page_1];
                            tma_2d_gmem2smem(smem_kv_addr + (unsigned int)(kv_stage_l * 16384) + (unsigned int)(page_1 * 32 * 128), (&K_map), hv_partition_l * 128, physical_page_1 * 32, kv_full_addr + (kv_stage_l) * 8);
                        }
                    }
                    kv_stage_l += 1;
                    if (kv_stage_l == 9) { kv_stage_l = 0; _phase_kv_empty ^= 1; }
                    mbarrier_arrive(page_empty_addr + (aligned_tile_l - 1) * 8);
                }
            }
            mbarrier_wait(kv_empty_addr + (kv_stage_l) * 8, _phase_kv_empty);
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(kv_full_addr + (kv_stage_l) * 8, 16384);
                #pragma unroll
                for (int page_2 = 0; page_2 < 4; page_2++) {
                    int physical_page_2 = page_offsets[(num_tiles_l - 1) * 32 + page_2];
                    tma_2d_gmem2smem(smem_kv_addr + (unsigned int)(kv_stage_l * 16384) + (unsigned int)(page_2 * 32 * 128), (&K_map), hv_partition_l * 128, physical_page_2 * 32, kv_full_addr + (kv_stage_l) * 8);
                }
            }
            kv_stage_l += 1;
            if (kv_stage_l == 9) { kv_stage_l = 0; _phase_kv_empty ^= 1; }
            mbarrier_arrive(page_empty_addr + (num_tiles_l - 1) * 8);
        }
    }

    // Cleanup
}

} // extern "C"

