/*
 * Copyright (c) 2023 by FlashInfer team.
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
template <int N>
struct __align__(128) CakeTensorMapPack { CakeTensorMap maps[N]; };

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128-byte aligned");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 512
#define TMEM_SOURCE_EXACT_SCRATCH_OFFSET 0
#define NUM_Q_PIPE_STAGES 1
#define NUM_KV_PIPE_STAGES 4
#define NUM_PAGE_PIPE_STAGES 6
#define NUM_SCORE_PIPE_STAGES 2
#define NUM_STATS_PIPE_STAGES 2
#define NUM_P_PIPE_STAGES 2
#define NUM_O_PIPE_STAGES 1
#define NUM_WORK_PIPE_STAGES 2
#define NUM_THROTTLE_PIPE_STAGES 2
#define SMEM_SMEM_Q_OFF 1024
#define SMEM_SMEM_Q_STAGE_BYTES 2048
#define SMEM_SMEM_Q_STRIDE 2048
#define SMEM_SMEM_KV_OFF 9216
#define SMEM_SMEM_KV_STAGE_BYTES 32768
#define SMEM_SMEM_KV_STRIDE 32768
#define SMEM_SMEM_V_OFF 9216
#define SMEM_SMEM_V_STAGE_BYTES 32768
#define SMEM_SMEM_V_STRIDE 32768
#define SMEM_SMEM_P_OFF 140288
#define SMEM_SMEM_P_STAGE_BYTES 2048
#define SMEM_SMEM_P_STRIDE 2048
#define SMEM_SMEM_PAGE_OFFSETS_OFF 144384
#define SMEM_SMEM_PAGE_OFFSETS_STAGE_BYTES 1024
#define SMEM_SMEM_PAGE_OFFSETS_STRIDE 1024
#define SMEM_SMEM_O_OFF 150528
#define SMEM_SMEM_O_STAGE_BYTES 2048
#define SMEM_SMEM_O_STRIDE 2048
#define SMEM_SMEM_SOFTMAX_OFF 152576
#define SMEM_SMEM_SOFTMAX_STAGE_BYTES 128
#define SMEM_SMEM_SOFTMAX_STRIDE 128
#define SMEM_SMEM_SOFTMAX_U32_OFF 152576
#define SMEM_SMEM_SOFTMAX_U32_STAGE_BYTES 128
#define SMEM_SMEM_SOFTMAX_U32_STRIDE 128
#define SMEM_SMEM_SOFTMAX_UNUSED_OFF 152704
#define SMEM_SMEM_SOFTMAX_UNUSED_STAGE_BYTES 128
#define SMEM_SMEM_SOFTMAX_UNUSED_STRIDE 128
#define SMEM_SMEM_CORR_OFF 152832
#define SMEM_SMEM_CORR_STAGE_BYTES 128
#define SMEM_SMEM_CORR_STRIDE 128
#define SMEM_SMEM_STATS_OFF 152960
#define SMEM_SMEM_STATS_STAGE_BYTES 96
#define SMEM_SMEM_STATS_STRIDE 96
#define SMEM_WORK_RESPONSE_OFF 153152
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_TOTAL 153216
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


__device__ __forceinline__ void tmem_ld_x4(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x4.b32"
        " {%0, %1, %2, %3}, [%4];"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3])
        : "r"(tmem_addr));
}


__device__ __forceinline__ void tmem_st_x4_f32(int tmem_addr, const float* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x4.b32"
        " [%0], {%1, %2, %3, %4};"
        :: "r"(tmem_addr),
           "f"(src[0]), "f"(src[1]), "f"(src[2]), "f"(src[3]));
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

__device__ __forceinline__ void mul_f32x2_inplace(float2* a, float2 b) {
    asm("mul.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void add_f32x2_inplace(float2* a, float2 b) {
    asm("add.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void sub_f32x2_inplace(float2* a, float2 b) {
    asm("sub.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ float2 add_f32x2(float2 a, float2 b) {
    float2 r;
    asm("add.rn.ftz.f32x2 %0, %1, %2;"
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


__device__ __forceinline__ void tma_4d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_gather4_gmem2smem(
    int dst, const void *tmap_ptr,
    int col_idx, int row0, int row1, int row2, int row3,
    int mbar_addr) {
    // Canonical .shared::cta form for non-multicast gather4, matching
    // trtllm-gen / cuda_ptx and the PTX ISA qualifier order
    // (dim.dst.src.load_mode.completion_mechanism). Per the PTX grammar,
    // .shared::cluster is reserved for the multicast variant (ctaMask).
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
        :: "r"(dst), "l"(tmap_ptr), "r"(col_idx),
           "r"(row0), "r"(row1), "r"(row2), "r"(row3),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tmem_ld_x4_wait(float* dst, int addr) {
    tmem_ld_x4(dst, addr);
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
kernel_cake_dsv4_bf16_h8_h16_source_exact(CakeTensorMap const* tmap_q, CakeTensorMap const* tmap_swa_kv, CakeTensorMap const* tmap_compressed_kv, __nv_bfloat16* __restrict__ O, int* __restrict__ sparse_indices, int* __restrict__ sparse_topk_lens, int* __restrict__ seq_lens, int* __restrict__ cum_seq_lens_q, float* __restrict__ sinks, float* __restrict__ bmm1_scale, float* __restrict__ bmm2_scale, int sparse_topk, int max_q_len, int batch_size, int has_sinks)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define q_full_addr (mbar_base + 0)
    #define q_empty_addr (mbar_base + 8)
    #define kv_full_addr (mbar_base + 16)
    #define kv_empty_addr (mbar_base + 48)
    #define page_full_addr (mbar_base + 80)
    #define page_empty_addr (mbar_base + 128)
    #define score_full_addr (mbar_base + 176)
    #define score_empty_addr (mbar_base + 192)
    #define stats_full_addr (mbar_base + 208)
    #define stats_empty_addr (mbar_base + 224)
    #define p_full_addr (mbar_base + 240)
    #define p_empty_addr (mbar_base + 256)
    #define o_full_addr (mbar_base + 272)
    #define o_empty_addr (mbar_base + 280)
    #define work_full_addr (mbar_base + 288)
    #define work_empty_addr (mbar_base + 304)
    #define throttle_full_addr (mbar_base + 320)
    #define throttle_empty_addr (mbar_base + 336)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(tmap_q)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(tmap_swa_kv)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(tmap_compressed_kv)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    __nv_bfloat16* smem_q = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_q_addr = smem + 1024;
    __nv_bfloat16* smem_kv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 9216);
    const int smem_kv_addr = smem + 9216;
    __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 9216);
    const int smem_v_addr = smem + 9216;
    __nv_bfloat16* smem_p = reinterpret_cast<__nv_bfloat16*>(smem_raw + 140288);
    const int smem_p_addr = smem + 140288;
    int* smem_page_offsets = reinterpret_cast<int*>(smem_raw + 144384);
    const int smem_page_offsets_addr = smem + 144384;
    __nv_bfloat16* smem_o = reinterpret_cast<__nv_bfloat16*>(smem_raw + 150528);
    const int smem_o_addr = smem + 150528;
    float* smem_softmax = reinterpret_cast<float*>(smem_raw + 152576);
    const int smem_softmax_addr = smem + 152576;
    unsigned int* smem_softmax_u32 = reinterpret_cast<unsigned int*>(smem_raw + 152576);
    const int smem_softmax_u32_addr = smem + 152576;
    float* smem_softmax_unused = reinterpret_cast<float*>(smem_raw + 152704);
    const int smem_softmax_unused_addr = smem + 152704;
    float* smem_corr = reinterpret_cast<float*>(smem_raw + 152832);
    const int smem_corr_addr = smem + 152832;
    float* smem_stats = reinterpret_cast<float*>(smem_raw + 152960);
    const int smem_stats_addr = smem + 152960;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 153152);
    const int work_response_addr = smem + 153152;
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(tmap_q)) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(tmap_swa_kv)) : "memory");
    asm volatile("prefetch.tensormap [%0];" :: "l"((uint64_t)(tmap_compressed_kv)) : "memory");

    // Mbarrier init (18 pipeline groups, 0 ordered-sequence groups, 44 barriers)
    // Mbarriers at smem_raw[0..352)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'q_pipe' ---
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // q_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // --- pipeline 'kv_pipe' ---
            // kv_full: 4 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // kv_empty: 4 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // --- pipeline 'page_pipe' ---
            // page_full: 6 barriers, init_count=32
            mbarrier_init(smem + 80, 32);
            mbarrier_init(smem + 88, 32);
            mbarrier_init(smem + 96, 32);
            mbarrier_init(smem + 104, 32);
            mbarrier_init(smem + 112, 32);
            mbarrier_init(smem + 120, 32);
            // page_empty: 6 barriers, init_count=128
            mbarrier_init(smem + 128, 128);
            mbarrier_init(smem + 136, 128);
            mbarrier_init(smem + 144, 128);
            mbarrier_init(smem + 152, 128);
            mbarrier_init(smem + 160, 128);
            mbarrier_init(smem + 168, 128);
            // --- pipeline 'score_pipe' ---
            // score_full: 2 barriers, init_count=1
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            // score_empty: 2 barriers, init_count=128
            mbarrier_init(smem + 192, 128);
            mbarrier_init(smem + 200, 128);
            // --- pipeline 'stats_pipe' ---
            // stats_full: 2 barriers, init_count=128
            mbarrier_init(smem + 208, 128);
            mbarrier_init(smem + 216, 128);
            // stats_empty: 2 barriers, init_count=128
            mbarrier_init(smem + 224, 128);
            mbarrier_init(smem + 232, 128);
            // --- pipeline 'p_pipe' ---
            // p_full: 2 barriers, init_count=128
            mbarrier_init(smem + 240, 128);
            mbarrier_init(smem + 248, 128);
            // p_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 256, 1);
            mbarrier_init(smem + 264, 1);
            // --- pipeline 'o_pipe' ---
            // o_full: 1 barriers, init_count=1
            mbarrier_init(smem + 272, 1);
            // o_empty: 1 barriers, init_count=128
            mbarrier_init(smem + 280, 128);
            // --- pipeline 'work_pipe' ---
            // work_full: 2 barriers, init_count=1
            mbarrier_init(smem + 288, 1);
            mbarrier_init(smem + 296, 1);
            // work_empty: 2 barriers, init_count=512
            mbarrier_init(smem + 304, 512);
            mbarrier_init(smem + 312, 512);
            // --- pipeline 'throttle_pipe' ---
            // throttle_full: 2 barriers, init_count=128
            mbarrier_init(smem + 320, 128);
            mbarrier_init(smem + 328, 128);
            // throttle_empty: 2 barriers, init_count=32
            mbarrier_init(smem + 336, 32);
            mbarrier_init(smem + 344, 32);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 352);
    if (warp == 0) {
        int _tmem_hold = smem + 352;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_source_exact_scratch = taddr;

    // ---- Ordered hardware-WG register redistribution ----
    // Inc phase consumes the registers released above.
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 136;");
    }

    // ---- Role: softmax ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 152;");
        { // softmax_main
            unsigned int work_stage = 0;
            unsigned int score_stage = 0;
            unsigned int score_phase = 0;
            unsigned int stats_stage = 0;
            unsigned int stats_phase = 1;
            unsigned int p_stage = 0;
            unsigned int p_phase = 1;
            unsigned int work_x = blockIdx.x;
            unsigned int work_y = blockIdx.y;
            unsigned int work_z = blockIdx.z;
            const int warp_in_wg = warp;
            const int wg_tid = (unsigned int)(warp_in_wg * 32) + lane;
            const int col_pair_base = wg_tid % 4 * 2;
            const int ldtm_row_base = (unsigned int)(warp_in_wg * 32) + lane / 4;
            float softmax_scale_log2 = bmm1_scale[0] * 1.4426950408889634f;
            unsigned int _phase_work_full = 0;
            #pragma unroll 1
            for (unsigned int _work = 0; _work < 120; _work++) {
                int q_begin = cum_seq_lens_q[work_z];
                int q_end = cum_seq_lens_q[work_z + 1];
                int q_len = q_end - q_begin;
                int query_idx = (unsigned int)q_begin + work_x;
                int active_topk = 0;
                if (work_x < (unsigned int)q_len) {
                    active_topk = sparse_topk_lens[query_idx];
                }
                int num_steps = (active_topk + 128 - 1) / 128;
                float row_max_pair[2];
                float row_sum_pair[2];
                row_max_pair[0] = -CAKE_INF;
                row_max_pair[1] = -CAKE_INF;
                row_sum_pair[0] = 0.0f;
                row_sum_pair[1] = 0.0f;
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                uint32_t _amf_u_0 = __float_as_uint(-3.4028235e+38f);
                uint32_t _amf_mask_0 = -int32_t(_amf_u_0 >> 31) | 0x80000000u;
                unsigned int _amf_enc_0 = _amf_u_0 ^ _amf_mask_0;
                if (wg_tid < 8) {
                    smem_softmax_u32[wg_tid] = _amf_enc_0;
                }
                asm volatile("barrier.sync 1, 128;" ::: "memory");
                #pragma unroll 1
                for (int tile = 0; tile < num_steps; tile++) {
                    mbarrier_wait(score_full_addr + (score_stage) * 8, score_phase);
                    int score_col = ((score_stage == 0) ? 0 : 8);
                    float scores[8];
                    float score_lo[4];
                    float score_hi[4];
                    int tile_end = (tile + 1) * 128;
                    int is_full_tile = ((active_topk % 128 == 0) ? 1 : 0);
                    is_full_tile = is_full_tile | ((tile_end < active_topk) ? 1 : 0);
                    if (is_full_tile != 0) {
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&score_lo[0])), "=r"(*reinterpret_cast<uint32_t*>(&score_lo[1])), "=r"(*reinterpret_cast<uint32_t*>(&score_lo[2])), "=r"(*reinterpret_cast<uint32_t*>(&score_lo[3]))
                            : "r"(taddr + (unsigned int)score_col));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&score_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&score_hi[1])), "=r"(*reinterpret_cast<uint32_t*>(&score_hi[2])), "=r"(*reinterpret_cast<uint32_t*>(&score_hi[3]))
                            : "r"(taddr + (unsigned int)score_col + 1048576));
                        #pragma unroll
                        for (int c = 0; c < 4; c++) {
                            scores[c] = score_lo[c];
                            scores[c + 4] = score_hi[c];
                        }
                    } else {
                        asm volatile(".pragma \"set knob ColdBlock\";\n" ::: "memory");
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&score_lo[0])), "=r"(*reinterpret_cast<uint32_t*>(&score_lo[1])), "=r"(*reinterpret_cast<uint32_t*>(&score_lo[2])), "=r"(*reinterpret_cast<uint32_t*>(&score_lo[3]))
                            : "r"(taddr + (unsigned int)score_col));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&score_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&score_hi[1])), "=r"(*reinterpret_cast<uint32_t*>(&score_hi[2])), "=r"(*reinterpret_cast<uint32_t*>(&score_hi[3]))
                            : "r"(taddr + (unsigned int)score_col + 1048576));
                        #pragma unroll
                        for (int c_1 = 0; c_1 < 4; c_1++) {
                            scores[c_1] = score_lo[c_1];
                            scores[c_1 + 4] = score_hi[c_1];
                        }
                        if (active_topk <= tile * 128 + ldtm_row_base) {
                            scores[0] = -CAKE_INF;
                            scores[1] = -CAKE_INF;
                        }
                        if (active_topk <= tile * 128 + ldtm_row_base + 8) {
                            scores[2] = -CAKE_INF;
                            scores[3] = -CAKE_INF;
                        }
                        if (active_topk <= tile * 128 + ldtm_row_base + 16) {
                            scores[4] = -CAKE_INF;
                            scores[5] = -CAKE_INF;
                        }
                        if (active_topk <= tile * 128 + ldtm_row_base + 24) {
                            scores[6] = -CAKE_INF;
                            scores[7] = -CAKE_INF;
                        }
                        asm volatile(".pragma \"reset knob ColdBlock\";\n" ::: "memory");
                    }
                    float pair_max[2];
                    float _max_0 = max_noftz(scores[0], scores[2]);
                    float _max_1 = max_noftz(scores[4], scores[6]);
                    float _max_2 = max_noftz(_max_0, _max_1);
                    pair_max[0] = _max_2;
                    float _max_3 = max_noftz(scores[1], scores[3]);
                    float _max_4 = max_noftz(scores[5], scores[7]);
                    float _max_5 = max_noftz(_max_3, _max_4);
                    pair_max[1] = _max_5;
                    #pragma unroll
                    for (int c_2 = 0; c_2 < 2; c_2++) {
                        float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, pair_max[c_2], 16);
                        float _max_6 = max_noftz(pair_max[c_2], _shfl_xor_0);
                        pair_max[c_2] = _max_6;
                        float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, pair_max[c_2], 8);
                        float _max_7 = max_noftz(pair_max[c_2], _shfl_xor_1);
                        pair_max[c_2] = _max_7;
                    }
                    float old_max_pair[2];
                    float new_max_pair[2];
                    #pragma unroll
                    for (int c_3 = 0; c_3 < 2; c_3++) {
                        old_max_pair[c_3] = row_max_pair[c_3];
                        float _max_8 = max_noftz(row_max_pair[c_3], pair_max[c_3]);
                        new_max_pair[c_3] = _max_8;
                    }
                    if (lane < 8) {
                        uint32_t _amf_u_1 = __float_as_uint(new_max_pair[0]);
                        uint32_t _amf_mask_1 = -int32_t(_amf_u_1 >> 31) | 0x80000000u;
                        unsigned int _amf_enc_1 = _amf_u_1 ^ _amf_mask_1;
                        uint32_t _amf_u_2 = __float_as_uint(new_max_pair[1]);
                        uint32_t _amf_mask_2 = -int32_t(_amf_u_2 >> 31) | 0x80000000u;
                        unsigned int _amf_enc_2 = _amf_u_2 ^ _amf_mask_2;
                        atomicMax(&smem_softmax_u32[col_pair_base], _amf_enc_1);
                        atomicMax(&smem_softmax_u32[col_pair_base + 1], _amf_enc_2);
                    }
                    asm volatile("barrier.sync 1, 128;" ::: "memory");
                    uint32_t _amf_u_3 = smem_softmax_u32[col_pair_base];
                    uint32_t _amf_mask_3 = ((_amf_u_3 >> 31) - 1u) | 0x80000000u;
                    float _amf_dec_0 = __uint_as_float(_amf_u_3 ^ _amf_mask_3);
                    new_max_pair[0] = _amf_dec_0;
                    uint32_t _amf_u_4 = smem_softmax_u32[col_pair_base + 1];
                    uint32_t _amf_mask_4 = ((_amf_u_4 >> 31) - 1u) | 0x80000000u;
                    float _amf_dec_1 = __uint_as_float(_amf_u_4 ^ _amf_mask_4);
                    new_max_pair[1] = _amf_dec_1;
                    mbarrier_wait(stats_empty_addr + (stats_stage) * 8, stats_phase);
                    float stats_pair[4];
                    stats_pair[0] = old_max_pair[0];
                    stats_pair[1] = old_max_pair[1];
                    stats_pair[2] = new_max_pair[0];
                    stats_pair[3] = new_max_pair[1];
                    int stats_col = ((stats_stage == 0) ? 16 : 48);
                    tmem_st_x4_f32(taddr + (unsigned int)stats_col, stats_pair);
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(stats_full_addr + (stats_stage) * 8);
                    stats_stage += 1;
                    if (stats_stage == 2) { stats_stage = 0; stats_phase ^= 1; }
                    float acc_scale_pair[2];
                    #pragma unroll
                    for (int c_4 = 0; c_4 < 2; c_4++) {
                        float delta = softmax_scale_log2 * (old_max_pair[c_4] - new_max_pair[c_4]);
                        float _exp2_0 = approx_exp2(delta);
                        acc_scale_pair[c_4] = ((old_max_pair[c_4] > -CAKE_INF) ? _exp2_0 : 1.0f);
                        row_max_pair[c_4] = new_max_pair[c_4];
                    }
                    asm volatile(".pragma \"set knob SchedResBusyXU64=1\";\n" ::: "memory");
                    float exp_values[8];
                    #pragma unroll
                    for (int c_5 = 0; c_5 < 8; c_5++) {
                        float safe_max = ((new_max_pair[c_5 % 2] == -CAKE_INF) ? 0.0f : new_max_pair[c_5 % 2]);
                        float _exp2_1 = approx_exp2(scores[c_5] * softmax_scale_log2 - safe_max * softmax_scale_log2);
                        exp_values[c_5] = _exp2_1;
                    }
                    mbarrier_wait(p_empty_addr + (p_stage) * 8, p_phase);
                    unsigned int regs_p[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(exp_values[_lp*2 + 0], exp_values[_lp*2+1 + 0]));
                        regs_p[_lp] = *(uint32_t*)&_bf2;
                    }
                    int slice_idx = warp_in_wg / 2;
                    int warp_idx_in_slice = warp_in_wg % 2;
                    int mtx_idx = lane / 8;
                    int thr_row_idx = lane % 8;
                    int seg_col_idx = warp_idx_in_slice * 4 + mtx_idx ^ thr_row_idx;
                    int stsm_offset = slice_idx * 8 * 128 + thr_row_idx * 128 + seg_col_idx * 16;
                    const void* _stmatrix_ptr_5 = reinterpret_cast<const void*>(reinterpret_cast<uint8_t*>(smem_p) + (p_stage * 2048 + (unsigned int)stsm_offset));
                    uint64_t _stmatrix_addr64_5;
                    asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(_stmatrix_addr64_5) : "l"(_stmatrix_ptr_5));
                    uint32_t _stmatrix_addr_5;
                    asm volatile("cvt.u32.u64 %0, %1;" : "=r"(_stmatrix_addr_5) : "l"(_stmatrix_addr64_5));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_5), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[0])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[1])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[2])), "r"(*reinterpret_cast<const uint32_t*>(&regs_p[3]))
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(p_full_addr + (p_stage) * 8);
                    p_stage += 1;
                    if (p_stage == 2) { p_stage = 0; p_phase ^= 1; }
                    float2 _f2_0 = make_float2(exp_values[0], exp_values[1]);
                    float2 _f2_1 = make_float2(exp_values[2], exp_values[3]);
                    float2 _f2_2 = make_float2(exp_values[4], exp_values[5]);
                    float2 _f2_3 = make_float2(exp_values[6], exp_values[7]);
                    float2 _f2_4 = make_float2(row_sum_pair[0], row_sum_pair[1]);
                    float2 _f2_5 = make_float2(acc_scale_pair[0], acc_scale_pair[1]);
                    float2 sum_p0_pair = fma_f32x2_rn_ftz(_f2_4, _f2_5, _f2_0);
                    float2 sum_p1_pair = add_f32x2(sum_p0_pair, _f2_1);
                    float2 sum_p2_pair = add_f32x2(sum_p1_pair, _f2_2);
                    float2 next_sum_pair = add_f32x2(sum_p2_pair, _f2_3);
                    row_sum_pair[0] = next_sum_pair.x;
                    row_sum_pair[1] = next_sum_pair.y;
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    mbarrier_arrive(score_empty_addr + (score_stage) * 8);
                    score_stage += 1;
                    if (score_stage == 2) { score_stage = 0; score_phase ^= 1; }
                }
                if (num_steps > 0) {
                    mbarrier_wait(stats_empty_addr + (stats_stage) * 8, stats_phase);
                    float final_stats_pair[4];
                    final_stats_pair[0] = row_sum_pair[0];
                    final_stats_pair[1] = row_sum_pair[1];
                    final_stats_pair[2] = row_max_pair[0];
                    final_stats_pair[3] = row_max_pair[1];
                    int final_stats_col = ((stats_stage == 0) ? 16 : 48);
                    tmem_st_x4_f32(taddr + (unsigned int)final_stats_col, final_stats_pair);
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(stats_full_addr + (stats_stage) * 8);
                    stats_stage += 1;
                    if (stats_stage == 2) { stats_stage = 0; stats_phase ^= 1; }
                }
                mbarrier_wait(work_full_addr + (work_stage) * 8, _phase_work_full);
                uint32_t _clc_valid_3 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_3)
                    : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_9 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_9)
                    : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_10 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_10)
                    : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_11 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_11)
                    : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + work_stage * 8), "r"(0) : "memory");
                work_stage += 1;
                if (work_stage == 2) { work_stage = 0; _phase_work_full ^= 1; }
                if (_clc_valid_3 == 0) {
                    break;
                }
                work_x = _clc_ctaid_9;
                work_y = _clc_ctaid_10;
                work_z = _clc_ctaid_11;
            }
        }
    }
    // ---- Role: correction ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 136;");
        { // correction_main
            unsigned int work_stage_1 = 0;
            unsigned int stats_stage_1 = 0;
            unsigned int stats_phase_1 = 0;
            unsigned int work_x_1 = blockIdx.x;
            unsigned int work_y_1 = blockIdx.y;
            unsigned int work_z_1 = blockIdx.z;
            const int corr_rank = warp - 4;
            const int corr_tid = (unsigned int)(corr_rank * 32) + lane;
            const int col_pair_base_1 = corr_tid % 4 * 2;
            float softmax_scale_log2_1 = bmm1_scale[0] * 1.4426950408889634f;
            float output_scale = bmm2_scale[0];
            unsigned int _phase_o_full_0 = 0;
            unsigned int _phase_work_full_1 = 0;
            #pragma unroll 1
            for (unsigned int _work_1 = 0; _work_1 < 120; _work_1++) {
                int q_begin_1 = cum_seq_lens_q[work_z_1];
                int q_end_1 = cum_seq_lens_q[work_z_1 + 1];
                int q_len_1 = q_end_1 - q_begin_1;
                int query_idx_1 = (unsigned int)q_begin_1 + work_x_1;
                int active_topk_1 = 0;
                if (work_x_1 < (unsigned int)q_len_1) {
                    active_topk_1 = sparse_topk_lens[query_idx_1];
                }
                int num_steps_1 = (active_topk_1 + 128 - 1) / 128;
                int head_group = work_y_1 / 4;
                int value_quarter = work_y_1 % 4;
                if (num_steps_1 > 0) {
                    mbarrier_wait(stats_full_addr + (stats_stage_1) * 8, stats_phase_1);
                    mbarrier_arrive(stats_empty_addr + (stats_stage_1) * 8);
                    stats_stage_1 += 1;
                    if (stats_stage_1 == 2) { stats_stage_1 = 0; stats_phase_1 ^= 1; }
                }
                #pragma unroll 1
                for (int tile_1 = 1; tile_1 < num_steps_1; tile_1++) {
                    mbarrier_wait(stats_full_addr + (stats_stage_1) * 8, stats_phase_1);
                    int stats_col_1 = ((stats_stage_1 == 0) ? 16 : 48);
                    float _tmem_load_0[4];
                    tmem_ld_x4(&_tmem_load_0[0], taddr + (unsigned int)stats_col_1);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    float scale_pair[2];
                    #pragma unroll
                    for (int c_6 = 0; c_6 < 2; c_6++) {
                        float max_diff = _tmem_load_0[c_6] - _tmem_load_0[c_6 + 2];
                        float _exp2_2 = approx_exp2(softmax_scale_log2_1 * max_diff);
                        scale_pair[c_6] = ((max_diff != 0.0f) ? _exp2_2 : 1.0f);
                    }
                    mbarrier_arrive(stats_empty_addr + (stats_stage_1) * 8);
                    stats_stage_1 += 1;
                    if (stats_stage_1 == 2) { stats_stage_1 = 0; stats_phase_1 ^= 1; }
                    mbarrier_wait(o_full_addr, _phase_o_full_0);
                    _phase_o_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int needs_scale = ((scale_pair[0] != 1.0f) ? 1 : 0);
                    needs_scale = needs_scale | ((scale_pair[1] != 1.0f) ? 1 : 0);
                    int _vote_0 = __any_sync(0xFFFFFFFF, needs_scale != 0);
                    needs_scale = _vote_0;
                    if (needs_scale != 0) {
                        float values_lo[4];
                        float values_hi[4];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&values_lo[0])), "=r"(*reinterpret_cast<uint32_t*>(&values_lo[1])), "=r"(*reinterpret_cast<uint32_t*>(&values_lo[2])), "=r"(*reinterpret_cast<uint32_t*>(&values_lo[3]))
                            : "r"(taddr + 80));
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                            " {%0, %1, %2, %3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&values_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&values_hi[1])), "=r"(*reinterpret_cast<uint32_t*>(&values_hi[2])), "=r"(*reinterpret_cast<uint32_t*>(&values_hi[3]))
                            : "r"(taddr + 80 + 1048576));
                        #pragma unroll
                        for (int pair = 0; pair < 4; pair++) {
                            asm volatile(".pragma \"next knob WarpOpexPrev=1\";\n" ::: "memory");
                            values_lo[pair] = values_lo[pair] * scale_pair[pair % 2];
                            values_hi[pair] = values_hi[pair] * scale_pair[pair % 2];
                        }
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x256b.x1.b32"
                            " [%0], {%1, %2, %3, %4};"
                            :: "r"(taddr + 80), "r"(*reinterpret_cast<const uint32_t*>(&values_lo[0])), "r"(*reinterpret_cast<const uint32_t*>(&values_lo[1])), "r"(*reinterpret_cast<const uint32_t*>(&values_lo[2])), "r"(*reinterpret_cast<const uint32_t*>(&values_lo[3])));
                        asm volatile(
                            "tcgen05.st.sync.aligned.16x256b.x1.b32"
                            " [%0], {%1, %2, %3, %4};"
                            :: "r"(taddr + 80 + 1048576), "r"(*reinterpret_cast<const uint32_t*>(&values_hi[0])), "r"(*reinterpret_cast<const uint32_t*>(&values_hi[1])), "r"(*reinterpret_cast<const uint32_t*>(&values_hi[2])), "r"(*reinterpret_cast<const uint32_t*>(&values_hi[3])));
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    }
                    mbarrier_arrive(o_empty_addr);
                }
                if (num_steps_1 > 0) {
                    mbarrier_wait(stats_full_addr + (stats_stage_1) * 8, stats_phase_1);
                    int final_stats_col_1 = ((stats_stage_1 == 0) ? 16 : 48);
                    float _tmem_load_1[4];
                    tmem_ld_x4(&_tmem_load_1[0], taddr + (unsigned int)final_stats_col_1);
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    mbarrier_arrive(stats_empty_addr + (stats_stage_1) * 8);
                    stats_stage_1 += 1;
                    if (stats_stage_1 == 2) { stats_stage_1 = 0; stats_phase_1 ^= 1; }
                    float pair_sum[2];
                    pair_sum[0] = _tmem_load_1[0];
                    pair_sum[1] = _tmem_load_1[1];
                    float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, pair_sum[0], 16);
                    pair_sum[0] = pair_sum[0] + _shfl_xor_2;
                    float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, pair_sum[1], 16);
                    pair_sum[1] = pair_sum[1] + _shfl_xor_3;
                    float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, pair_sum[0], 8);
                    pair_sum[0] = pair_sum[0] + _shfl_xor_4;
                    float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, pair_sum[1], 8);
                    pair_sum[1] = pair_sum[1] + _shfl_xor_5;
                    float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, pair_sum[0], 4);
                    pair_sum[0] = pair_sum[0] + _shfl_xor_6;
                    float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, pair_sum[1], 4);
                    pair_sum[1] = pair_sum[1] + _shfl_xor_7;
                    if (lane < 4) {
                        smem_corr[corr_rank * 8 + col_pair_base_1] = pair_sum[0];
                        smem_corr[corr_rank * 8 + col_pair_base_1 + 1] = pair_sum[1];
                    }
                    asm volatile("barrier.sync 2, 128;" ::: "memory");
                    float sum0 = smem_corr[col_pair_base_1] + smem_corr[8 + col_pair_base_1] + smem_corr[16 + col_pair_base_1] + smem_corr[24 + col_pair_base_1];
                    float sum1 = smem_corr[col_pair_base_1 + 1] + smem_corr[8 + col_pair_base_1 + 1] + smem_corr[16 + col_pair_base_1 + 1] + smem_corr[24 + col_pair_base_1 + 1];
                    if (has_sinks != 0) {
                        float _exp2_3 = approx_exp2(sinks[head_group * 8 + col_pair_base_1] * 1.4426950408889634f - _tmem_load_1[2] * softmax_scale_log2_1);
                        sum0 = sum0 + _exp2_3;
                        float _exp2_4 = approx_exp2(sinks[head_group * 8 + col_pair_base_1 + 1] * 1.4426950408889634f - _tmem_load_1[3] * softmax_scale_log2_1);
                        sum1 = sum1 + _exp2_4;
                    }
                    float _rcp_0 = approx_rcp(sum0);
                    float scale0 = ((sum0 > 0.0f) ? output_scale * _rcp_0 : 0.0f);
                    float _rcp_1 = approx_rcp(sum1);
                    float scale1 = ((sum1 > 0.0f) ? output_scale * _rcp_1 : 0.0f);
                    mbarrier_wait(o_full_addr, _phase_o_full_0);
                    _phase_o_full_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float output_lo[4];
                    float output_hi[4];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                        " {%0, %1, %2, %3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&output_lo[0])), "=r"(*reinterpret_cast<uint32_t*>(&output_lo[1])), "=r"(*reinterpret_cast<uint32_t*>(&output_lo[2])), "=r"(*reinterpret_cast<uint32_t*>(&output_lo[3]))
                        : "r"(taddr + 80));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                        " {%0, %1, %2, %3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&output_hi[0])), "=r"(*reinterpret_cast<uint32_t*>(&output_hi[1])), "=r"(*reinterpret_cast<uint32_t*>(&output_hi[2])), "=r"(*reinterpret_cast<uint32_t*>(&output_hi[3]))
                        : "r"(taddr + 80 + 1048576));
                    unsigned int packed_o[4];
                    #pragma unroll
                    for (int pair_1 = 0; pair_1 < 4; pair_1++) {
                        asm volatile(".pragma \"next knob WarpOpexPrev=1\";\n" ::: "memory");
                        output_lo[pair_1] = output_lo[pair_1] * ((pair_1 % 2 == 0) ? scale0 : scale1);
                        output_hi[pair_1] = output_hi[pair_1] * ((pair_1 % 2 == 0) ? scale0 : scale1);
                    }
                    float output_values[8];
                    #pragma unroll
                    for (int c_7 = 0; c_7 < 4; c_7++) {
                        output_values[c_7] = output_lo[c_7];
                        output_values[c_7 + 4] = output_hi[c_7];
                    }
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(output_values[_lp*2 + 0], output_values[_lp*2+1 + 0]));
                        packed_o[_lp] = *(uint32_t*)&_bf2;
                    }
                    int slice_idx_1 = corr_rank / 2;
                    int warp_idx_in_slice_1 = corr_rank % 2;
                    int mtx_idx_1 = lane / 8;
                    int thr_row_idx_1 = lane % 8;
                    int seg_col_idx_1 = warp_idx_in_slice_1 * 4 + mtx_idx_1 ^ thr_row_idx_1;
                    int stsm_offset_1 = slice_idx_1 * 8 * 128 + thr_row_idx_1 * 128 + seg_col_idx_1 * 16;
                    const void* _stmatrix_ptr_0 = reinterpret_cast<const void*>(reinterpret_cast<uint8_t*>(smem_o) + stsm_offset_1);
                    uint64_t _stmatrix_addr64_0;
                    asm volatile("cvta.to.shared.u64 %0, %1;" : "=l"(_stmatrix_addr64_0) : "l"(_stmatrix_ptr_0));
                    uint32_t _stmatrix_addr_0;
                    asm volatile("cvt.u32.u64 %0, %1;" : "=r"(_stmatrix_addr_0) : "l"(_stmatrix_addr64_0));
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_0), "r"(*reinterpret_cast<const uint32_t*>(&packed_o[0])), "r"(*reinterpret_cast<const uint32_t*>(&packed_o[1])), "r"(*reinterpret_cast<const uint32_t*>(&packed_o[2])), "r"(*reinterpret_cast<const uint32_t*>(&packed_o[3]))
                        : "memory");
                    asm volatile("barrier.sync 2, 128;" ::: "memory");
                    int copy_base = corr_tid * 16;
                    int smem_row = copy_base / 128;
                    int copy_smem_offset = copy_base ^ smem_row % 8 * 16;
                    int copy_row = smem_row % 8;
                    int copy_col = (smem_row / 8 * 128 + copy_base % 128) / 2;
                    unsigned int copy_vec[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&copy_vec[0])), "=r"(*reinterpret_cast<uint32_t*>(&copy_vec[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&copy_vec[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&copy_vec[(0) + 3]))
                        : "r"(smem_o_addr + (unsigned int)copy_smem_offset));
                    int head_groups = gridDim.y / 4;
                    int output_base = query_idx_1 * head_groups * 8 * 512 + head_group * 8 * 512 + value_quarter * 128;
                    reinterpret_cast<int4*>(O + (output_base + copy_row * 512 + copy_col))[0] = reinterpret_cast<int4*>(copy_vec)[0];
                    mbarrier_arrive(o_empty_addr);
                }
                mbarrier_wait(work_full_addr + (work_stage_1) * 8, _phase_work_full_1);
                uint32_t _clc_valid_4 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_4)
                    : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_12 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_12)
                    : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_13 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_13)
                    : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_14 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_14)
                    : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + work_stage_1 * 8), "r"(0) : "memory");
                work_stage_1 += 1;
                if (work_stage_1 == 2) { work_stage_1 = 0; _phase_work_full_1 ^= 1; }
                if (_clc_valid_4 == 0) {
                    break;
                }
                work_x_1 = _clc_ctaid_12;
                work_y_1 = _clc_ctaid_13;
                work_z_1 = _clc_ctaid_14;
            }
            asm volatile("barrier.sync 11, 128;" ::: "memory");
            if (corr_rank == 0) {
                int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(512));
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 8) {
        { // mma_main
            unsigned int work_stage_2 = 0;
            unsigned int q_stage = 0;
            unsigned int q_phase = 0;
            unsigned int kv_stage = 0;
            unsigned int kv_phase = 0;
            unsigned int score_stage_1 = 0;
            unsigned int score_phase_1 = 1;
            unsigned int p_stage_1 = 0;
            unsigned int p_phase_1 = 0;
            unsigned int work_x_2 = blockIdx.x;
            unsigned int work_y_2 = blockIdx.y;
            unsigned int work_z_2 = blockIdx.z;
            unsigned int _phase_o_empty_0 = 1;
            unsigned int _phase_work_full_2 = 0;
            #pragma unroll 1
            for (unsigned int _work_2 = 0; _work_2 < 120; _work_2++) {
                int q_begin_2 = cum_seq_lens_q[work_z_2];
                int q_end_2 = cum_seq_lens_q[work_z_2 + 1];
                int q_len_2 = q_end_2 - q_begin_2;
                int query_idx_2 = (unsigned int)q_begin_2 + work_x_2;
                int active_topk_2 = 0;
                if (work_x_2 < (unsigned int)q_len_2) {
                    active_topk_2 = sparse_topk_lens[query_idx_2];
                }
                int num_steps_2 = (active_topk_2 + 128 - 1) / 128;
                if (num_steps_2 > 0) {
                    mbarrier_wait(q_full_addr + (q_stage) * 8, q_phase);
                    int first_output = 1;
                    mbarrier_wait(score_empty_addr + (score_stage_1) * 8, score_phase_1);
                    int score_col_1 = ((score_stage_1 == 0) ? 0 : 8);
                    #pragma unroll
                    for (int hs = 0; hs < 4; hs++) {
                        mbarrier_wait(kv_full_addr + (kv_stage) * 8, kv_phase);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_0 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (kv_stage) * 2048);
                        int _mma_b_lo_0 = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (hs) * 128);
                        {
                            uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
                            uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_source_exact_scratch + (score_col_1)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134349968, ((hs == 0) ? 0 : 1));
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_source_exact_scratch + (score_col_1)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134349968, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_source_exact_scratch + (score_col_1)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134349968, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_source_exact_scratch + (score_col_1)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134349968, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_0, 1018U);
                            incr_smem_desc_lo(_mma_ss_b_desc_0, 58U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_source_exact_scratch + (score_col_1)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134349968, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_source_exact_scratch + (score_col_1)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134349968, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_source_exact_scratch + (score_col_1)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134349968, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                            incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_source_exact_scratch + (score_col_1)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 134349968, 1);
                            }
                        }
                        elect_commit(kv_empty_addr + (kv_stage) * 8);
                        kv_stage += 1;
                        if (kv_stage == 4) { kv_stage = 0; kv_phase ^= 1; }
                    }
                    elect_commit(score_full_addr + (score_stage_1) * 8);
                    score_stage_1 += 1;
                    if (score_stage_1 == 2) { score_stage_1 = 0; score_phase_1 ^= 1; }
                    #pragma unroll 1
                    for (int _tile = 0; _tile < num_steps_2 - 1; _tile++) {
                        mbarrier_wait(score_empty_addr + (score_stage_1) * 8, score_phase_1);
                        int score_col_0 = ((score_stage_1 == 0) ? 0 : 8);
                        #pragma unroll
                        for (int hs_1 = 0; hs_1 < 4; hs_1++) {
                            mbarrier_wait(kv_full_addr + (kv_stage) * 8, kv_phase);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                            int _mma_a_lo_1 = make_warp_uniform((((smem_kv_addr) >> 4) & 0x3FFF) + (kv_stage) * 2048);
                            int _mma_b_lo_1 = make_warp_uniform((((smem_q_addr) >> 4) & 0x3FFF) + (hs_1) * 128);
                            {
                                uint64_t _mma_ss_a_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_1);
                                uint64_t _mma_ss_b_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_1);
                                if (elect_sync()) {
                                    tcgen05_mma_f16((tmem_source_exact_scratch + (score_col_0)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134349968, ((hs_1 == 0) ? 0 : 1));
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f16((tmem_source_exact_scratch + (score_col_0)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134349968, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f16((tmem_source_exact_scratch + (score_col_0)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134349968, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f16((tmem_source_exact_scratch + (score_col_0)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134349968, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_1, 1018U);
                                incr_smem_desc_lo(_mma_ss_b_desc_1, 58U);
                                if (elect_sync()) {
                                    tcgen05_mma_f16((tmem_source_exact_scratch + (score_col_0)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134349968, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f16((tmem_source_exact_scratch + (score_col_0)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134349968, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f16((tmem_source_exact_scratch + (score_col_0)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134349968, 1);
                                }
                                incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                                incr_smem_desc_lo(_mma_ss_b_desc_1, 2U);
                                if (elect_sync()) {
                                    tcgen05_mma_f16((tmem_source_exact_scratch + (score_col_0)), _mma_ss_a_desc_1, _mma_ss_b_desc_1, 134349968, 1);
                                }
                            }
                            elect_commit(kv_empty_addr + (kv_stage) * 8);
                            kv_stage += 1;
                            if (kv_stage == 4) { kv_stage = 0; kv_phase ^= 1; }
                        }
                        elect_commit(score_full_addr + (score_stage_1) * 8);
                        score_stage_1 += 1;
                        if (score_stage_1 == 2) { score_stage_1 = 0; score_phase_1 ^= 1; }
                        mbarrier_wait(p_full_addr + (p_stage_1) * 8, p_phase_1);
                        mbarrier_wait(o_empty_addr, _phase_o_empty_0);
                        _phase_o_empty_0 ^= 1;
                        mbarrier_wait(kv_full_addr + (kv_stage) * 8, kv_phase);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_2 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (kv_stage) * 2048);
                        int _mma_b_lo_2 = make_warp_uniform(((((smem_p_addr) >> 4) & 0x3FFF) | 0x400000) + (p_stage_1) * 128);
                        {
                            uint64_t _mma_ss_a_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_2);
                            uint64_t _mma_ss_b_desc_2 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_2);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_source_exact_scratch + (80)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134382736, ((first_output) ? 0 : 1));
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_2, 128U);
                            incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_source_exact_scratch + (80)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134382736, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_2, 128U);
                            incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_source_exact_scratch + (80)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134382736, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_2, 128U);
                            incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_source_exact_scratch + (80)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134382736, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_2, 128U);
                            incr_smem_desc_lo(_mma_ss_b_desc_2, 58U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_source_exact_scratch + (80)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134382736, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_2, 128U);
                            incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_source_exact_scratch + (80)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134382736, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_2, 128U);
                            incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_source_exact_scratch + (80)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134382736, 1);
                            }
                            incr_smem_desc_lo(_mma_ss_a_desc_2, 128U);
                            incr_smem_desc_lo(_mma_ss_b_desc_2, 2U);
                            if (elect_sync()) {
                                tcgen05_mma_f16((tmem_source_exact_scratch + (80)), _mma_ss_a_desc_2, _mma_ss_b_desc_2, 134382736, 1);
                            }
                        }
                        elect_commit(o_full_addr);
                        elect_commit(kv_empty_addr + (kv_stage) * 8);
                        kv_stage += 1;
                        if (kv_stage == 4) { kv_stage = 0; kv_phase ^= 1; }
                        elect_commit(p_empty_addr + (p_stage_1) * 8);
                        p_stage_1 += 1;
                        if (p_stage_1 == 2) { p_stage_1 = 0; p_phase_1 ^= 1; }
                        first_output = 0;
                    }
                    mbarrier_wait(p_full_addr + (p_stage_1) * 8, p_phase_1);
                    mbarrier_wait(o_empty_addr, _phase_o_empty_0);
                    _phase_o_empty_0 ^= 1;
                    mbarrier_wait(kv_full_addr + (kv_stage) * 8, kv_phase);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_3 = make_warp_uniform(((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (kv_stage) * 2048);
                    int _mma_b_lo_3 = make_warp_uniform(((((smem_p_addr) >> 4) & 0x3FFF) | 0x400000) + (p_stage_1) * 128);
                    {
                        uint64_t _mma_ss_a_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_3);
                        uint64_t _mma_ss_b_desc_3 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_3);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_source_exact_scratch + (80)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134382736, ((first_output) ? 0 : 1));
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_3, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_source_exact_scratch + (80)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134382736, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_3, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_source_exact_scratch + (80)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134382736, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_3, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_source_exact_scratch + (80)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134382736, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_3, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_3, 58U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_source_exact_scratch + (80)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134382736, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_3, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_source_exact_scratch + (80)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134382736, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_3, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_source_exact_scratch + (80)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134382736, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_3, 128U);
                        incr_smem_desc_lo(_mma_ss_b_desc_3, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f16((tmem_source_exact_scratch + (80)), _mma_ss_a_desc_3, _mma_ss_b_desc_3, 134382736, 1);
                        }
                    }
                    elect_commit(o_full_addr);
                    elect_commit(kv_empty_addr + (kv_stage) * 8);
                    kv_stage += 1;
                    if (kv_stage == 4) { kv_stage = 0; kv_phase ^= 1; }
                    elect_commit(p_empty_addr + (p_stage_1) * 8);
                    p_stage_1 += 1;
                    if (p_stage_1 == 2) { p_stage_1 = 0; p_phase_1 ^= 1; }
                    elect_commit(q_empty_addr + (q_stage) * 8);
                    q_phase ^= 1;
                }
                mbarrier_wait(work_full_addr + (work_stage_2) * 8, _phase_work_full_2);
                uint32_t _clc_valid_2 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_2)
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_6 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_6)
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_7 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_7)
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_8 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_8)
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + work_stage_2 * 8), "r"(0) : "memory");
                work_stage_2 += 1;
                if (work_stage_2 == 2) { work_stage_2 = 0; _phase_work_full_2 ^= 1; }
                if (_clc_valid_2 == 0) {
                    break;
                }
                work_x_2 = _clc_ctaid_6;
                work_y_2 = _clc_ctaid_7;
                work_z_2 = _clc_ctaid_8;
            }
            #pragma unroll
            for (int tail = 0; tail < 2; tail++) {
                elect_commit(score_full_addr + (tail) * 8);
            }
        }
    }
    // ---- Role: page_loader ----
    if (warp == 9) {
        { // page_loader_main
            unsigned int work_stage_3 = 0;
            unsigned int page_stage = 0;
            unsigned int page_phase = 1;
            unsigned int work_x_3 = blockIdx.x;
            unsigned int work_y_3 = blockIdx.y;
            unsigned int work_z_3 = blockIdx.z;
            unsigned int _phase_work_full_3 = 0;
            #pragma unroll 1
            for (unsigned int _work_3 = 0; _work_3 < 120; _work_3++) {
                int q_begin_3 = cum_seq_lens_q[work_z_3];
                int q_end_3 = cum_seq_lens_q[work_z_3 + 1];
                int q_len_3 = q_end_3 - q_begin_3;
                int query_idx_3 = (unsigned int)q_begin_3 + work_x_3;
                int active_topk_3 = 0;
                if (work_x_3 < (unsigned int)q_len_3) {
                    active_topk_3 = sparse_topk_lens[query_idx_3];
                }
                int num_steps_3 = (active_topk_3 + 128 - 1) / 128;
                int num_pairs = (num_steps_3 + 1) / 2;
                int last_aligned = active_topk_3 - 1 & -4;
                #pragma unroll 1
                for (int pair_2 = 0; pair_2 < num_pairs; pair_2++) {
                    int global_base = query_idx_3 * sparse_topk + pair_2 * 2 * 128;
                    int lane_base = lane * 4;
                    #pragma unroll
                    for (int duplicate = 0; duplicate < 2; duplicate++) {
                        mbarrier_wait(page_empty_addr + (page_stage) * 8, page_phase);
                        int page0 = pair_2 * 2 * 128 + lane_base;
                        int page1 = page0 + 128;
                        int safe0 = ((page0 <= last_aligned) ? page0 : last_aligned);
                        int safe1 = ((page1 <= last_aligned) ? page1 : last_aligned);
                        asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 16;"
                            :: "r"(smem_page_offsets_addr + page_stage * 1024 + (unsigned int)(lane_base * 4)), "l"(sparse_indices + (query_idx_3 * sparse_topk + safe0)));
                        asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 16;"
                            :: "r"(smem_page_offsets_addr + page_stage * 1024 + (unsigned int)((128 + lane_base) * 4)), "l"(sparse_indices + (query_idx_3 * sparse_topk + safe1)));
                        asm volatile(
                            "{\n\t"
                            "cp.async.mbarrier.arrive.shared::cta.b64 [%0];\n\t"
                            "}"
                            :: "r"(page_full_addr + (page_stage) * 8) : "memory");
                        mbarrier_arrive(page_full_addr + (page_stage) * 8);
                        page_stage += 1;
                        if (page_stage == 6) { page_stage = 0; page_phase ^= 1; }
                    }
                }
                mbarrier_wait(work_full_addr + (work_stage_3) * 8, _phase_work_full_3);
                uint32_t _clc_valid_0 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_0)
                    : "r"(work_response_addr + work_stage_3 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_0 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_0)
                    : "r"(work_response_addr + work_stage_3 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_1 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_1)
                    : "r"(work_response_addr + work_stage_3 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_2 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_2)
                    : "r"(work_response_addr + work_stage_3 * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + work_stage_3 * 8), "r"(0) : "memory");
                work_stage_3 += 1;
                if (work_stage_3 == 2) { work_stage_3 = 0; _phase_work_full_3 ^= 1; }
                if (_clc_valid_0 == 0) {
                    break;
                }
                work_x_3 = _clc_ctaid_0;
                work_y_3 = _clc_ctaid_1;
                work_z_3 = _clc_ctaid_2;
            }
        }
    }
    // ---- Role: scheduler ----
    if (warp == 10) {
        { // scheduler_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int work_stage_4 = 0;
            unsigned int throttle_stage = 0;
            unsigned int throttle_phase = 0;
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_work_full_4 = 0;
            #pragma unroll 1
            for (unsigned int _work_4 = 0; _work_4 < 120; _work_4++) {
                mbarrier_wait(throttle_full_addr + (throttle_stage) * 8, throttle_phase);
                mbarrier_arrive(throttle_empty_addr + (throttle_stage) * 8);
                throttle_stage += 1;
                if (throttle_stage == 2) { throttle_stage = 0; throttle_phase ^= 1; }
                mbarrier_wait(work_empty_addr + (work_stage_4) * 8, _phase_work_empty);
                if (elect_sync()) {
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                        "}"
                        :: "r"(work_full_addr + work_stage_4 * 8), "r"(0), "r"((uint32_t)(16)) : "memory");
                    asm volatile(
                        "fence.proxy.async.shared::cta;\n\t"
                        "clusterlaunchcontrol.try_cancel.async.shared::cta"
                            ".mbarrier::complete_tx::bytes.b128"
                            " [%0], [%1];"
                        :: "r"(work_response_addr + work_stage_4 * 16 + 0 * 16), "r"(work_full_addr + work_stage_4 * 8)
                        : "memory");
                }
                mbarrier_wait(work_full_addr + (work_stage_4) * 8, _phase_work_full_4);
                uint32_t _clc_valid_5 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_5)
                    : "r"(work_response_addr + work_stage_4 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_15 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_15)
                    : "r"(work_response_addr + work_stage_4 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_16 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_16)
                    : "r"(work_response_addr + work_stage_4 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_17 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_17)
                    : "r"(work_response_addr + work_stage_4 * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + work_stage_4 * 8), "r"(0) : "memory");
                work_stage_4 += 1;
                if (work_stage_4 == 2) { work_stage_4 = 0; _phase_work_empty ^= 1; _phase_work_full_4 ^= 1; }
                if (_clc_valid_5 == 0) {
                    break;
                }
            }
        }
    }
    // ---- Role: padding ----
    if (warp == 11) {
        { // padding_main
            unsigned int work_stage_5 = 0;
            unsigned int work_x_4 = blockIdx.x;
            unsigned int work_y_4 = blockIdx.y;
            unsigned int work_z_4 = blockIdx.z;
            unsigned int _phase_work_full_5 = 0;
            #pragma unroll 1
            for (unsigned int _work_5 = 0; _work_5 < 120; _work_5++) {
                mbarrier_wait(work_full_addr + (work_stage_5) * 8, _phase_work_full_5);
                uint32_t _clc_valid_6 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_6)
                    : "r"(work_response_addr + work_stage_5 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_18 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_18)
                    : "r"(work_response_addr + work_stage_5 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_19 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_19)
                    : "r"(work_response_addr + work_stage_5 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_20 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_20)
                    : "r"(work_response_addr + work_stage_5 * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + work_stage_5 * 8), "r"(0) : "memory");
                work_stage_5 += 1;
                if (work_stage_5 == 2) { work_stage_5 = 0; _phase_work_full_5 ^= 1; }
                if (_clc_valid_6 == 0) {
                    break;
                }
                work_x_4 = _clc_ctaid_18;
                work_y_4 = _clc_ctaid_19;
                work_z_4 = _clc_ctaid_20;
            }
        }
    }
    // ---- Role: loader ----
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 88;");
        { // loader_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            const int load_rank = warp - 12;
            unsigned int work_stage_6 = 0;
            unsigned int throttle_stage_1 = 0;
            unsigned int throttle_phase_1 = 1;
            unsigned int q_stage_1 = 0;
            unsigned int q_phase_1 = 1;
            unsigned int kv_stage_1 = 0;
            unsigned int kv_phase_1 = 1;
            unsigned int page_k_stage = 0;
            unsigned int page_k_phase = 0;
            unsigned int page_v_stage = 1;
            unsigned int page_v_phase = 0;
            unsigned int work_x_5 = blockIdx.x;
            unsigned int work_y_5 = blockIdx.y;
            unsigned int work_z_5 = blockIdx.z;
            unsigned int _phase_work_full_6 = 0;
            #pragma unroll 1
            for (unsigned int _work_6 = 0; _work_6 < 120; _work_6++) {
                int q_begin_4 = cum_seq_lens_q[work_z_5];
                int q_end_4 = cum_seq_lens_q[work_z_5 + 1];
                int q_len_4 = q_end_4 - q_begin_4;
                int query_idx_4 = (unsigned int)q_begin_4 + work_x_5;
                int active_topk_4 = 0;
                if (work_x_5 < (unsigned int)q_len_4) {
                    active_topk_4 = sparse_topk_lens[query_idx_4];
                }
                int num_steps_4 = (active_topk_4 + 128 - 1) / 128;
                int head_group_1 = work_y_5 / 4;
                int value_quarter_1 = work_y_5 % 4;
                mbarrier_wait(throttle_empty_addr + (throttle_stage_1) * 8, throttle_phase_1);
                mbarrier_arrive(throttle_full_addr + (throttle_stage_1) * 8);
                throttle_stage_1 += 1;
                if (throttle_stage_1 == 2) { throttle_stage_1 = 0; throttle_phase_1 ^= 1; }
                if (num_steps_4 > 0) {
                    mbarrier_wait(q_empty_addr + (q_stage_1) * 8, q_phase_1);
                    if (load_rank == 0) {
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(q_full_addr + (q_stage_1) * 8, 8192);
                            #pragma unroll
                            for (int hs_2 = 0; hs_2 < 4; hs_2++) {
                                tma_4d_gmem2smem(smem_q_addr + (unsigned int)(hs_2 * 2048), tmap_q, 0, head_group_1 * 8, hs_2 * 2, query_idx_4, q_full_addr + (q_stage_1) * 8);
                            }
                        }
                    }
                    mbarrier_wait(page_full_addr + (page_k_stage) * 8, page_k_phase);
                    int page_base = 0;
                    #pragma unroll
                    for (int hs_3 = 0; hs_3 < 4; hs_3++) {
                        mbarrier_wait(kv_empty_addr + (kv_stage_1) * 8, kv_phase_1);
                        if (load_rank == 0) {
                            if (elect_sync()) {
                                mbarrier_arrive_expect_tx(kv_full_addr + (kv_stage_1) * 8, 32768);
                            }
                        }
                        asm volatile("barrier.sync 9, 128;" ::: "memory");
                        #pragma unroll
                        for (int group = 0; group < 8; group++) {
                            const int row_base = load_rank * 4 + group * 16;
                            int rows[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&rows[0])), "=r"(*reinterpret_cast<uint32_t*>(&rows[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rows[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rows[(0) + 3]))
                                : "r"(smem_page_offsets_addr + page_k_stage * 1024 + (unsigned int)((page_base + row_base) * 4)));
                            int dst_k = smem_kv_addr + kv_stage_1 * 32768 + (unsigned int)(row_base * 64 * 2);
                            if (elect_sync()) {
                                {
                                    tma_gather4_gmem2smem(dst_k, tmap_swa_kv, hs_3 * 128, rows[0], rows[1], rows[2], rows[3], kv_full_addr + (kv_stage_1) * 8);
                                    tma_gather4_gmem2smem(dst_k + 16384, tmap_swa_kv, hs_3 * 128 + 64, rows[0], rows[1], rows[2], rows[3], kv_full_addr + (kv_stage_1) * 8);
                                }
                            }
                        }
                        kv_stage_1 += 1;
                        if (kv_stage_1 == 4) { kv_stage_1 = 0; kv_phase_1 ^= 1; }
                    }
                    #pragma unroll 1
                    for (int tile_2 = 0; tile_2 < num_steps_4 - 1; tile_2++) {
                        int next_tile = tile_2 + 1;
                        if ((next_tile & 1) == 0) {
                            mbarrier_wait(page_full_addr + (page_k_stage) * 8, page_k_phase);
                        }
                        int page_base_0 = (next_tile & 1) * 128;
                        #pragma unroll
                        for (int hs_4 = 0; hs_4 < 4; hs_4++) {
                            mbarrier_wait(kv_empty_addr + (kv_stage_1) * 8, kv_phase_1);
                            if (load_rank == 0) {
                                if (elect_sync()) {
                                    mbarrier_arrive_expect_tx(kv_full_addr + (kv_stage_1) * 8, 32768);
                                }
                            }
                            asm volatile("barrier.sync 9, 128;" ::: "memory");
                            #pragma unroll
                            for (int group_1 = 0; group_1 < 8; group_1++) {
                                const int row_base_1 = load_rank * 4 + group_1 * 16;
                                int rows_1[4];
                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                    : "=r"(*reinterpret_cast<uint32_t*>(&rows_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&rows_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rows_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rows_1[(0) + 3]))
                                    : "r"(smem_page_offsets_addr + page_k_stage * 1024 + (unsigned int)((page_base_0 + row_base_1) * 4)));
                                int dst_k_1 = smem_kv_addr + kv_stage_1 * 32768 + (unsigned int)(row_base_1 * 64 * 2);
                                if (elect_sync()) {
                                    if (next_tile == 0) {
                                        tma_gather4_gmem2smem(dst_k_1, tmap_swa_kv, hs_4 * 128, rows_1[0], rows_1[1], rows_1[2], rows_1[3], kv_full_addr + (kv_stage_1) * 8);
                                        tma_gather4_gmem2smem(dst_k_1 + 16384, tmap_swa_kv, hs_4 * 128 + 64, rows_1[0], rows_1[1], rows_1[2], rows_1[3], kv_full_addr + (kv_stage_1) * 8);
                                    } else {
                                        tma_gather4_gmem2smem(dst_k_1, tmap_compressed_kv, hs_4 * 128, rows_1[0], rows_1[1], rows_1[2], rows_1[3], kv_full_addr + (kv_stage_1) * 8);
                                        tma_gather4_gmem2smem(dst_k_1 + 16384, tmap_compressed_kv, hs_4 * 128 + 64, rows_1[0], rows_1[1], rows_1[2], rows_1[3], kv_full_addr + (kv_stage_1) * 8);
                                    }
                                }
                            }
                            kv_stage_1 += 1;
                            if (kv_stage_1 == 4) { kv_stage_1 = 0; kv_phase_1 ^= 1; }
                        }
                        if ((next_tile & 1) != 0 && next_tile < num_steps_4 - 1) {
                            mbarrier_arrive(page_empty_addr + (page_k_stage) * 8);
                            page_k_stage += 1;
                            if (page_k_stage == 6) { page_k_stage = 0; page_k_phase ^= 1; }
                            page_k_stage += 1;
                            if (page_k_stage == 6) { page_k_stage = 0; page_k_phase ^= 1; }
                        }
                        if ((tile_2 & 1) == 0) {
                            mbarrier_wait(page_full_addr + (page_v_stage) * 8, page_v_phase);
                        }
                        int page_base_1 = (tile_2 & 1) * 128;
                        mbarrier_wait(kv_empty_addr + (kv_stage_1) * 8, kv_phase_1);
                        if (load_rank == 0) {
                            if (elect_sync()) {
                                mbarrier_arrive_expect_tx(kv_full_addr + (kv_stage_1) * 8, 32768);
                            }
                        }
                        asm volatile("barrier.sync 9, 128;" ::: "memory");
                        #pragma unroll
                        for (int group_2 = 0; group_2 < 8; group_2++) {
                            const int row_base_2 = load_rank * 4 + group_2 * 16;
                            int rows_2[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&rows_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&rows_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rows_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rows_2[(0) + 3]))
                                : "r"(smem_page_offsets_addr + page_v_stage * 1024 + (unsigned int)((page_base_1 + row_base_2) * 4)));
                            int dst_v = smem_v_addr + kv_stage_1 * 32768 + (unsigned int)(row_base_2 * 64 * 2);
                            if (elect_sync()) {
                                if (tile_2 == 0) {
                                    tma_gather4_gmem2smem(dst_v, tmap_swa_kv, value_quarter_1 * 128, rows_2[0], rows_2[1], rows_2[2], rows_2[3], kv_full_addr + (kv_stage_1) * 8);
                                    tma_gather4_gmem2smem(dst_v + 16384, tmap_swa_kv, value_quarter_1 * 128 + 64, rows_2[0], rows_2[1], rows_2[2], rows_2[3], kv_full_addr + (kv_stage_1) * 8);
                                } else {
                                    tma_gather4_gmem2smem(dst_v, tmap_compressed_kv, value_quarter_1 * 128, rows_2[0], rows_2[1], rows_2[2], rows_2[3], kv_full_addr + (kv_stage_1) * 8);
                                    tma_gather4_gmem2smem(dst_v + 16384, tmap_compressed_kv, value_quarter_1 * 128 + 64, rows_2[0], rows_2[1], rows_2[2], rows_2[3], kv_full_addr + (kv_stage_1) * 8);
                                }
                            }
                        }
                        kv_stage_1 += 1;
                        if (kv_stage_1 == 4) { kv_stage_1 = 0; kv_phase_1 ^= 1; }
                        if ((tile_2 & 1) != 0) {
                            mbarrier_arrive(page_empty_addr + (page_v_stage) * 8);
                            page_v_stage += 1;
                            if (page_v_stage == 6) { page_v_stage = 0; page_v_phase ^= 1; }
                            page_v_stage += 1;
                            if (page_v_stage == 6) { page_v_stage = 0; page_v_phase ^= 1; }
                        }
                    }
                    int last_tile = num_steps_4 - 1;
                    mbarrier_arrive(page_empty_addr + (page_k_stage) * 8);
                    page_k_stage += 1;
                    if (page_k_stage == 6) { page_k_stage = 0; page_k_phase ^= 1; }
                    page_k_stage += 1;
                    if (page_k_stage == 6) { page_k_stage = 0; page_k_phase ^= 1; }
                    if ((last_tile & 1) == 0) {
                        mbarrier_wait(page_full_addr + (page_v_stage) * 8, page_v_phase);
                    }
                    int page_base_0_1 = (last_tile & 1) * 128;
                    mbarrier_wait(kv_empty_addr + (kv_stage_1) * 8, kv_phase_1);
                    if (load_rank == 0) {
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(kv_full_addr + (kv_stage_1) * 8, 32768);
                        }
                    }
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                    #pragma unroll
                    for (int group_3 = 0; group_3 < 8; group_3++) {
                        const int row_base_3 = load_rank * 4 + group_3 * 16;
                        int rows_3[4];
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&rows_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&rows_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&rows_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&rows_3[(0) + 3]))
                            : "r"(smem_page_offsets_addr + page_v_stage * 1024 + (unsigned int)((page_base_0_1 + row_base_3) * 4)));
                        int dst_v_1 = smem_v_addr + kv_stage_1 * 32768 + (unsigned int)(row_base_3 * 64 * 2);
                        if (elect_sync()) {
                            if (last_tile == 0) {
                                tma_gather4_gmem2smem(dst_v_1, tmap_swa_kv, value_quarter_1 * 128, rows_3[0], rows_3[1], rows_3[2], rows_3[3], kv_full_addr + (kv_stage_1) * 8);
                                tma_gather4_gmem2smem(dst_v_1 + 16384, tmap_swa_kv, value_quarter_1 * 128 + 64, rows_3[0], rows_3[1], rows_3[2], rows_3[3], kv_full_addr + (kv_stage_1) * 8);
                            } else {
                                tma_gather4_gmem2smem(dst_v_1, tmap_compressed_kv, value_quarter_1 * 128, rows_3[0], rows_3[1], rows_3[2], rows_3[3], kv_full_addr + (kv_stage_1) * 8);
                                tma_gather4_gmem2smem(dst_v_1 + 16384, tmap_compressed_kv, value_quarter_1 * 128 + 64, rows_3[0], rows_3[1], rows_3[2], rows_3[3], kv_full_addr + (kv_stage_1) * 8);
                            }
                        }
                    }
                    kv_stage_1 += 1;
                    if (kv_stage_1 == 4) { kv_stage_1 = 0; kv_phase_1 ^= 1; }
                    mbarrier_arrive(page_empty_addr + (page_v_stage) * 8);
                    page_v_stage += 1;
                    if (page_v_stage == 6) { page_v_stage = 0; page_v_phase ^= 1; }
                    page_v_stage += 1;
                    if (page_v_stage == 6) { page_v_stage = 0; page_v_phase ^= 1; }
                }
                mbarrier_wait(work_full_addr + (work_stage_6) * 8, _phase_work_full_6);
                uint32_t _clc_valid_1 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_1)
                    : "r"(work_response_addr + work_stage_6 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_3 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_3)
                    : "r"(work_response_addr + work_stage_6 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_4 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_4)
                    : "r"(work_response_addr + work_stage_6 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_5 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_5)
                    : "r"(work_response_addr + work_stage_6 * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + work_stage_6 * 8), "r"(0) : "memory");
                work_stage_6 += 1;
                if (work_stage_6 == 2) { work_stage_6 = 0; _phase_work_full_6 ^= 1; }
                if (_clc_valid_1 == 0) {
                    break;
                }
                work_x_5 = _clc_ctaid_3;
                work_y_5 = _clc_ctaid_4;
                work_z_5 = _clc_ctaid_5;
            }
        }
    }

    // Cleanup
}

} // extern "C"
