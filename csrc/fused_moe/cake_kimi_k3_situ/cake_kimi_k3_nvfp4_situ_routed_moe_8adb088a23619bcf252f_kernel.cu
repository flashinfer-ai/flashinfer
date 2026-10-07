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
#define TMEM_NCOLS 248
#define TMEM_ACCUM_OFFSET 0
#define TMEM_SFA_OFFSET 8
#define TMEM_SFB_OFFSET 168
#define NUM_K_PIPE_STAGES 5
#define NUM_OUT_PIPE_STAGES 1
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 32768
#define SMEM_SMEM_A_STRIDE 32768
#define SMEM_SMEM_B_OFF 164864
#define SMEM_SMEM_B_STAGE_BYTES 2048
#define SMEM_SMEM_B_STRIDE 2048
#define SMEM_EPI_STAGING_OFF 175104
#define SMEM_EPI_STAGING_STAGE_BYTES 256
#define SMEM_EPI_STAGING_STRIDE 256
#define SMEM_SMEM_SFA_OFF 177152
#define SMEM_SMEM_SFA_STAGE_BYTES 4096
#define SMEM_SMEM_SFA_STRIDE 4096
#define SMEM_SMEM_SFB_OFF 197632
#define SMEM_SMEM_SFB_STAGE_BYTES 256
#define SMEM_SMEM_SFB_STRIDE 256
#define SMEM_SMEM_QTOK_OFF 198912
#define SMEM_SMEM_QTOK_STAGE_BYTES 1792
#define SMEM_SMEM_QTOK_STRIDE 1792
#define SMEM_SMEM_QSF_OFF 200704
#define SMEM_SMEM_QSF_STAGE_BYTES 256
#define SMEM_SMEM_QSF_STRIDE 256
#define SMEM_SMEM_QROUTE_OFF 200960
#define SMEM_SMEM_QROUTE_STAGE_BYTES 16
#define SMEM_SMEM_QROUTE_STRIDE 16
#define SMEM_TOTAL 201088
#define THREADS 512
#define BLOCK_M 128
#define BLOCK_N 8
#define BLOCK_K 512
#define WEIGHTS_SHUFFLED 1

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


__device__ __forceinline__ void tcgen05_mma_mxf4nvf4_bs(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
    int sfa_taddr, int sfb_taddr, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.scale_vec::4X"
        " [%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(sfa_taddr), "r"(sfb_taddr),
           "r"(enable_input_d));
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


__device__ __forceinline__ void tma_4d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
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


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tmem_ld_x4(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x4.b32"
        " {%0, %1, %2, %3}, [%4];"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3])
        : "r"(tmem_addr));
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
kernel_cake_kimi_k3_nvfp4_situ_routed_moe_8adb088a23619bcf252f(const __grid_constant__ CUtensorMap A, uint8_t* __restrict__ B, const __grid_constant__ CUtensorMap SFA, uint8_t* __restrict__ SFB, const __grid_constant__ CUtensorMap C, uint8_t* __restrict__ SFC, int* __restrict__ total_tiles, __nv_bfloat16* __restrict__ x, float* __restrict__ qx, int* __restrict__ topk_ids, int* __restrict__ token_to_permuted, int* __restrict__ expert_counts, int* __restrict__ expert_tile_offsets, int* __restrict__ expert_scatter_offsets, int* __restrict__ route_map, int* __restrict__ tile_expert, int* __restrict__ tile_mn_limit, float* __restrict__ scale_c, float* __restrict__ scale_gate, float* __restrict__ clamp_limit, float* __restrict__ act_alpha, float* __restrict__ act_beta, int M_out, int K, int grid_m, int grid_n, int K_tiles)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane = static_cast<uint32_t>(tid) & 31u;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define a_full_addr (mbar_base + 0)
    #define b_full_addr (mbar_base + 40)
    #define sfa_full_addr (mbar_base + 80)
    #define sfa_free_addr (mbar_base + 120)
    #define tmem_sfa_full_addr (mbar_base + 160)
    #define tmem_sfb_full_addr (mbar_base + 200)
    #define k_done_addr (mbar_base + 240)
    #define mma_full_addr (mbar_base + 280)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    // Kernel setup ops
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + 164864);
    const int smem_b_addr = smem + 164864;
    uint8_t* epi_staging = reinterpret_cast<uint8_t*>(smem_raw + 175104);
    const int epi_staging_addr = smem + 175104;
    uint8_t* smem_sfa = reinterpret_cast<uint8_t*>(smem_raw + 177152);
    const int smem_sfa_addr = smem + 177152;
    uint8_t* smem_sfb = reinterpret_cast<uint8_t*>(smem_raw + 197632);
    const int smem_sfb_addr = smem + 197632;
    uint8_t* smem_qtok = reinterpret_cast<uint8_t*>(smem_raw + 198912);
    const int smem_qtok_addr = smem + 198912;
    uint8_t* smem_qsf = reinterpret_cast<uint8_t*>(smem_raw + 200704);
    const int smem_qsf_addr = smem + 200704;
    uint8_t* smem_qroute = reinterpret_cast<uint8_t*>(smem_raw + 200960);
    const int smem_qroute_addr = smem + 200960;

    // Mbarrier init (8 pipeline groups, 0 ordered-sequence groups, 36 barriers)
    // Mbarriers at smem_raw[0..288)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'k_pipe' ---
            // a_full: 5 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            // b_full: 5 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // sfa_full: 5 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            // sfa_free: 5 barriers, init_count=1
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            // tmem_sfa_full: 5 barriers, init_count=1
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            mbarrier_init(smem + 192, 1);
            // tmem_sfb_full: 5 barriers, init_count=1
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            mbarrier_init(smem + 232, 1);
            // k_done: 5 barriers, init_count=1
            mbarrier_init(smem + 240, 1);
            mbarrier_init(smem + 248, 1);
            mbarrier_init(smem + 256, 1);
            mbarrier_init(smem + 264, 1);
            mbarrier_init(smem + 272, 1);
            // --- pipeline 'out_pipe' ---
            // mma_full: 1 barriers, init_count=1
            mbarrier_init(smem + 280, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (256 columns, 248 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 288);
    if (warp == 0) {
        int _tmem_hold = smem + 288;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;
    const int tmem_sfa = taddr + 8;
    const int tmem_sfb = taddr + 168;
    asm volatile("griddepcontrol.wait;" ::: "memory");
    int s2a_cta00 = 0;
    if (blockIdx.x == 0 && blockIdx.y == 0) {
        s2a_cta00 = 1;
    }
    int s2a_n_tile = blockIdx.y;
    int group = tid - 32;
    if (tid >= 32 && group < 224) {
        unsigned int packed_source[8];
        float source[16];
        {
            const uint4* _vptr_0 = reinterpret_cast<const uint4*>(x + (group * 16) + 0);
            uint4* _vdst_0 = reinterpret_cast<uint4*>(&packed_source[0]);
            #pragma unroll
            for (int _blk = 0; _blk < 2; _blk++) {
                _vdst_0[_blk] = _vptr_0[_blk];
            }
        }
        #pragma unroll
        for (int _pair = 0; _pair < 8; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&source[_pair * 2])[0]), "=f"((&source[_pair * 2])[1])
                : "r"(packed_source[_pair]));
        }
        uint32_t _bf16x2_abs_0;
        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_0) : "r"(packed_source[0]));
        uint32_t _bf16x2_abs_1;
        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_1) : "r"(packed_source[1]));
        uint32_t _bf16x2_max_0;
        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_0) : "r"(_bf16x2_abs_0), "r"(_bf16x2_abs_1));
        uint32_t _bf16x2_abs_2;
        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_2) : "r"(packed_source[2]));
        uint32_t _bf16x2_max_1;
        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_1) : "r"(_bf16x2_max_0), "r"(_bf16x2_abs_2));
        uint32_t _bf16x2_abs_3;
        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_3) : "r"(packed_source[3]));
        uint32_t _bf16x2_max_2;
        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_2) : "r"(_bf16x2_max_1), "r"(_bf16x2_abs_3));
        uint32_t _bf16x2_abs_4;
        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_4) : "r"(packed_source[4]));
        uint32_t _bf16x2_max_3;
        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_3) : "r"(_bf16x2_max_2), "r"(_bf16x2_abs_4));
        uint32_t _bf16x2_abs_5;
        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_5) : "r"(packed_source[5]));
        uint32_t _bf16x2_max_4;
        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_4) : "r"(_bf16x2_max_3), "r"(_bf16x2_abs_5));
        uint32_t _bf16x2_abs_6;
        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_6) : "r"(packed_source[6]));
        uint32_t _bf16x2_max_5;
        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_5) : "r"(_bf16x2_max_4), "r"(_bf16x2_abs_6));
        uint32_t _bf16x2_abs_7;
        asm("abs.bf16x2 %0, %1;" : "=r"(_bf16x2_abs_7) : "r"(packed_source[7]));
        uint32_t _bf16x2_max_6;
        asm("max.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_max_6) : "r"(_bf16x2_max_5), "r"(_bf16x2_abs_7));
        uint16_t _bf16_max_0;
        asm("max.bf16 %0, %1, %2;" : "=h"(_bf16_max_0) : "h"((uint16_t)(_bf16x2_max_6 & 65535)), "h"((uint16_t)(_bf16x2_max_6 >> 16)));
        float _cvt_f32_bf16_0;
        asm("cvt.f32.bf16 %0, %1;" : "=f"(_cvt_f32_bf16_0) : "h"((uint16_t)(_bf16_max_0)));
        float block_max = _cvt_f32_bf16_0;
        float global_encode = qx[0];
        float scale_value = 0.0f;
        if (block_max != 0.0f) {
            scale_value = block_max * (global_encode * 0.16666666666666666f);
        }
        float _fp8_rt_0;
        uint16_t _e4m3x2_1;
        uint32_t _f16x2_1;
        asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_1) : "f"(0.0f), "f"(scale_value));
        asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1) : "h"(_e4m3x2_1));
        uint16_t _fp8_h0_1 = (uint16_t)(_f16x2_1 & 0xFFFFu);
        asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_0) : "h"(_fp8_h0_1));
        float rounded_scale = _fp8_rt_0;
        float output_scale = 0.0f;
        if (block_max != 0.0f) {
            float _fdiv_rn_0 = __fdiv_rn(1.0f, global_encode);
            float global_decode = _fdiv_rn_0;
            float _fdiv_rn_1 = __fdiv_rn(1.0f, rounded_scale * global_decode);
            float _min_0 = fminf(_fdiv_rn_1, 3.4028234663852886e+38f);
            output_scale = _min_0;
        }
        const float2 _scale2_2 = {output_scale, output_scale};
        #pragma unroll
        for (int _ls = 0; _ls < 8; _ls++)
            mul_f32x2_inplace(&reinterpret_cast<float2*>(source)[_ls], _scale2_2);
        uint32_t _fp4_0[2];
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_0[0]) : "f"(source[0]), "f"(source[1]), "f"(source[2]), "f"(source[3]), "f"(source[4]), "f"(source[5]), "f"(source[6]), "f"(source[7]));
        asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_0[1]) : "f"(source[8]), "f"(source[9]), "f"(source[10]), "f"(source[11]), "f"(source[12]), "f"(source[13]), "f"(source[14]), "f"(source[15]));
        asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_qtok_addr + (unsigned int)(group * 8)), "r"((_fp4_0[0])));
        asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_qtok_addr + (unsigned int)(group * 8) + 4), "r"((_fp4_0[1])));
        float _shfl_down_0 = __shfl_down_sync(0xFFFFFFFF, scale_value, 1, 32);
        float s2a_scale_hi = _shfl_down_0;
        if (group % 2 == 0) {
            uint16_t _e4m3x2_f32_0;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_f32_0) : "f"(s2a_scale_hi), "f"(scale_value));
            asm volatile("st.shared.b16 [%0], %1;" :: "r"(smem_qsf_addr + (unsigned int)group), "h"((uint16_t)(_e4m3x2_f32_0)));
        }
        if (s2a_cta00 == 1) {
            *(reinterpret_cast<int*>(B + (group * 8)) + (0)) = _fp4_0[0];
            *(reinterpret_cast<int*>(B + (group * 8 + 4)) + (0)) = _fp4_0[1];
            {
                unsigned short _fp8_pair;
                asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(scale_value));
                *(reinterpret_cast<unsigned char*>(SFB + group) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
            }
        }
    }
    if (s2a_cta00 == 1) {
        if (tid < 129) {
            route_map[tid] = 0;
        }
        if (tid < 16) {
            tile_expert[tid] = 0;
            tile_mn_limit[tid] = tid * 8;
        }
        if (warp == 0) {
            __syncwarp();
        }
    }
    int s2a_expert = 0;
    int s2a_valid = 0;
    if (warp == 0) {
        int route_expert = 896;
        if (lane < 16) {
            route_expert = topk_ids[lane];
        }
        int occurrences_before = 0;
        int route_count = 0;
        #pragma unroll
        for (int slot = 0; slot < 16; slot++) {
            int _shfl_0 = __shfl_sync(0xFFFFFFFF, route_expert, slot);
            int peer_expert = _shfl_0;
            if (peer_expert == route_expert) {
                route_count = route_count + 1;
                if ((unsigned int)slot < lane) {
                    occurrences_before = occurrences_before + 1;
                }
            }
        }
        int tile_begin = 0;
        if (lane < 16 && occurrences_before % 8 == 0) {
            tile_begin = 1;
        }
        int route_offset = 0;
        int tile_total = 0;
        #pragma unroll
        for (int slot_1 = 0; slot_1 < 16; slot_1++) {
            int _shfl_1 = __shfl_sync(0xFFFFFFFF, route_expert, slot_1);
            int prefix_expert = _shfl_1;
            int _shfl_2 = __shfl_sync(0xFFFFFFFF, tile_begin, slot_1);
            int prefix_tiles = _shfl_2;
            tile_total = tile_total + prefix_tiles;
            if (prefix_expert < route_expert) {
                route_offset = route_offset + prefix_tiles;
            }
        }
        int s2a_tile_sel = -1;
        int s2a_valid_sel = 0;
        if (lane < 16 && occurrences_before % 8 == 0) {
            s2a_tile_sel = route_offset + occurrences_before / 8;
            s2a_valid_sel = route_count - occurrences_before;
            if (s2a_valid_sel > 8) {
                s2a_valid_sel = 8;
            }
        }
        #pragma unroll
        for (int slot_2 = 0; slot_2 < 16; slot_2++) {
            int _shfl_3 = __shfl_sync(0xFFFFFFFF, s2a_tile_sel, slot_2);
            int slot_tile = _shfl_3;
            int _shfl_4 = __shfl_sync(0xFFFFFFFF, route_expert, slot_2);
            int slot_expert = _shfl_4;
            int _shfl_5 = __shfl_sync(0xFFFFFFFF, s2a_valid_sel, slot_2);
            int slot_valid = _shfl_5;
            if (slot_tile == s2a_n_tile) {
                s2a_expert = slot_expert;
                s2a_valid = slot_valid;
            }
        }
        if (s2a_cta00 == 1) {
            if (tid == 0) {
                total_tiles[0] = tile_total;
            }
            if (tid < 16) {
                int tile = route_offset + occurrences_before / 8;
                int grouped_row = tile * 8 + occurrences_before % 8;
                token_to_permuted[tid] = grouped_row;
                if (occurrences_before % 8 == 0) {
                    int valid = route_count - occurrences_before;
                    if (valid > 8) {
                        valid = 8;
                    }
                    tile_expert[tile] = route_expert;
                    tile_mn_limit[tile] = tile * 8 + valid;
                }
            }
        }
        if (lane == 0) {
            unsigned int s2a_route_w[2];
            s2a_route_w[0] = (unsigned int)s2a_expert;
            s2a_route_w[1] = (unsigned int)s2a_valid;
            asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_qroute_addr), "r"((s2a_route_w[0])));
            asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_qroute_addr + 4), "r"((s2a_route_w[1])));
        }
    }
    if (s2a_cta00 == 1 && tid >= 256) {
        int d_ids[16];
        for (int d_s = 0; d_s < 16; d_s++) {
            d_ids[d_s] = topk_ids[d_s];
        }
        int d_tb[16];
        for (int d_s_1 = 0; d_s_1 < 16; d_s_1++) {
            int d_occ = 0;
            for (int d_j = 0; d_j < d_s_1; d_j++) {
                if (d_ids[d_j] == d_ids[d_s_1]) {
                    d_occ = d_occ + 1;
                }
            }
            d_tb[d_s_1] = 0;
            if (d_occ % 8 == 0) {
                d_tb[d_s_1] = 1;
            }
        }
        for (int d_chunk = 0; d_chunk < 4; d_chunk++) {
            int dense_expert = tid - 256 + d_chunk * 256;
            int d_cnt = 0;
            int d_off = 0;
            for (int d_s_2 = 0; d_s_2 < 16; d_s_2++) {
                if (d_ids[d_s_2] == dense_expert) {
                    d_cnt = d_cnt + 1;
                }
                if (dense_expert > d_ids[d_s_2]) {
                    d_off = d_off + d_tb[d_s_2];
                }
            }
            if (dense_expert < 896) {
                expert_counts[dense_expert] = d_cnt;
                expert_scatter_offsets[dense_expert] = d_cnt;
                expert_tile_offsets[dense_expert] = d_off;
            }
        }
    }
    __syncthreads();
    unsigned int s2a_route_r[2];
    asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];"
        : "=r"(*reinterpret_cast<uint32_t*>(&s2a_route_r[0])), "=r"(*reinterpret_cast<uint32_t*>(&s2a_route_r[(0) + 1]))
        : "r"(smem_qroute_addr));
    s2a_expert = (int)s2a_route_r[0];
    s2a_valid = (int)s2a_route_r[1];

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 8 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
    }

    // ---- Role: epilogue ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 160;");
        { // epilogue_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            const int warp_0 = warp;
            const int lane_1 = lane;
            int n_tile = blockIdx.y;
            int m_tile = blockIdx.x;
            int expert = s2a_expert;
            int valid_rows = s2a_valid;
            float sc = scale_c[expert];
            float sg = scale_gate[expert];
            float al = act_alpha[expert];
            float be = act_beta[expert];
            float quant_pair[8] = {0};
            unsigned int _phase_mma_full_0 = 0;
            mbarrier_wait(mma_full_addr, _phase_mma_full_0);
            _phase_mma_full_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float _tmem_load_0[4];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                " {%0, %1, %2, %3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3]))
                : "r"(taddr));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            float _tmem_load_1[4];
            asm volatile(
                "tcgen05.ld.sync.aligned.16x256b.x1.b32"
                " {%0, %1, %2, %3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3]))
                : "r"(taddr + 1048576));
            asm volatile("tcgen05.wait::ld.sync.aligned;");
            int base_row = warp_0 * 16 + lane_1 / 4 * 2;
            asm volatile("cp.async.bulk.wait_group.read 0;");
            asm volatile("barrier.sync 7, 128;" ::: "memory");
            for (int token_group = 0; token_group < 1; token_group++) {
                int token0 = lane_1 % 4 * 2 + token_group * 8;
                int token1 = token0 + 1;
                float value00 = 0.0f;
                float value01 = 0.0f;
                float value10 = 0.0f;
                float value11 = 0.0f;
                {
                    float up00 = _tmem_load_0[token_group * 4];
                    float up01 = _tmem_load_0[token_group * 4 + 1];
                    float gate00 = _tmem_load_0[token_group * 4 + 2];
                    float gate01 = _tmem_load_0[token_group * 4 + 3];
                    float up10 = _tmem_load_1[token_group * 4];
                    float up11 = _tmem_load_1[token_group * 4 + 1];
                    float gate10 = _tmem_load_1[token_group * 4 + 2];
                    float gate11 = _tmem_load_1[token_group * 4 + 3];
                    float _rcp_0 = approx_rcp(al);
                    float inv_alpha = _rcp_0;
                    float _rcp_1 = approx_rcp(be);
                    float inv_beta = _rcp_1;
                    float gate_tanh_scale = sg * inv_alpha;
                    float up_tanh_scale = sg * inv_beta;
                    float neg_gate_sigmoid_scale = -(sg * 1.4426950216293335f);
                    float2 _f2_0 = make_float2(gate_tanh_scale, gate_tanh_scale);
                    float2 gate_tanh_scale2 = _f2_0;
                    float2 _f2_1 = make_float2(up_tanh_scale, up_tanh_scale);
                    float2 up_tanh_scale2 = _f2_1;
                    float2 _f2_2 = make_float2(neg_gate_sigmoid_scale, neg_gate_sigmoid_scale);
                    float2 neg_gate_sigmoid_scale2 = _f2_2;
                    float2 _f2_3 = make_float2(al, al);
                    float2 alpha2 = _f2_3;
                    float2 _f2_4 = make_float2(be, be);
                    float2 beta2 = _f2_4;
                    float2 _f2_5 = make_float2(1.0f, 1.0f);
                    float2 one2 = _f2_5;
                    float2 _f2_6 = make_float2(up00, up01);
                    float2 up0 = _f2_6;
                    float2 _f2_7 = make_float2(up10, up11);
                    float2 up1 = _f2_7;
                    float2 _f2_8 = make_float2(gate00, gate01);
                    float2 gate0 = _f2_8;
                    float2 _f2_9 = make_float2(gate10, gate11);
                    float2 gate1 = _f2_9;
                    float2 _mul_f32x2_0;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_0) : "l"(*(const unsigned long long*)&up0), "l"(*(const unsigned long long*)&up_tanh_scale2));
                    float2 up0_norm = _mul_f32x2_0;
                    float2 _mul_f32x2_1;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_1) : "l"(*(const unsigned long long*)&up1), "l"(*(const unsigned long long*)&up_tanh_scale2));
                    float2 up1_norm = _mul_f32x2_1;
                    float2 _mul_f32x2_2;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_2) : "l"(*(const unsigned long long*)&gate0), "l"(*(const unsigned long long*)&gate_tanh_scale2));
                    float2 gate0_norm = _mul_f32x2_2;
                    float2 _mul_f32x2_3;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_3) : "l"(*(const unsigned long long*)&gate1), "l"(*(const unsigned long long*)&gate_tanh_scale2));
                    float2 gate1_norm = _mul_f32x2_3;
                    float2 _mul_f32x2_4;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_4) : "l"(*(const unsigned long long*)&gate0), "l"(*(const unsigned long long*)&neg_gate_sigmoid_scale2));
                    float2 gate0_exp_arg = _mul_f32x2_4;
                    float2 _mul_f32x2_5;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_5) : "l"(*(const unsigned long long*)&gate1), "l"(*(const unsigned long long*)&neg_gate_sigmoid_scale2));
                    float2 gate1_exp_arg = _mul_f32x2_5;
                    float _tanh_approx_0;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_0) : "f"(up0_norm.x));
                    float _tanh_approx_1;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_1) : "f"(up0_norm.y));
                    float2 _f2_10 = make_float2(_tanh_approx_0, _tanh_approx_1);
                    float2 up0_tanh = _f2_10;
                    float _tanh_approx_2;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_2) : "f"(up1_norm.x));
                    float _tanh_approx_3;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_3) : "f"(up1_norm.y));
                    float2 _f2_11 = make_float2(_tanh_approx_2, _tanh_approx_3);
                    float2 up1_tanh = _f2_11;
                    float _tanh_approx_4;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_4) : "f"(gate0_norm.x));
                    float _tanh_approx_5;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_5) : "f"(gate0_norm.y));
                    float2 _f2_12 = make_float2(_tanh_approx_4, _tanh_approx_5);
                    float2 gate0_tanh = _f2_12;
                    float _tanh_approx_6;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_6) : "f"(gate1_norm.x));
                    float _tanh_approx_7;
                    asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_7) : "f"(gate1_norm.y));
                    float2 _f2_13 = make_float2(_tanh_approx_6, _tanh_approx_7);
                    float2 gate1_tanh = _f2_13;
                    float _exp2_0 = approx_exp2(gate0_exp_arg.x);
                    float _exp2_1 = approx_exp2(gate0_exp_arg.y);
                    float2 _f2_14 = make_float2(_exp2_0, _exp2_1);
                    float2 gate0_exp = _f2_14;
                    float _exp2_2 = approx_exp2(gate1_exp_arg.x);
                    float _exp2_3 = approx_exp2(gate1_exp_arg.y);
                    float2 _f2_15 = make_float2(_exp2_2, _exp2_3);
                    float2 gate1_exp = _f2_15;
                    float2 gate0_denom = add_f32x2(gate0_exp, one2);
                    float2 gate1_denom = add_f32x2(gate1_exp, one2);
                    float _rcp_2 = approx_rcp(gate0_denom.x);
                    float _rcp_3 = approx_rcp(gate0_denom.y);
                    float2 _f2_16 = make_float2(_rcp_2, _rcp_3);
                    float2 gate0_sigmoid = _f2_16;
                    float _rcp_4 = approx_rcp(gate1_denom.x);
                    float _rcp_5 = approx_rcp(gate1_denom.y);
                    float2 _f2_17 = make_float2(_rcp_4, _rcp_5);
                    float2 gate1_sigmoid = _f2_17;
                    float2 _mul_f32x2_6;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_6) : "l"(*(const unsigned long long*)&beta2), "l"(*(const unsigned long long*)&up0_tanh));
                    float2 left0 = _mul_f32x2_6;
                    float2 _mul_f32x2_7;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_7) : "l"(*(const unsigned long long*)&beta2), "l"(*(const unsigned long long*)&up1_tanh));
                    float2 left1 = _mul_f32x2_7;
                    float2 _mul_f32x2_8;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_8) : "l"(*(const unsigned long long*)&alpha2), "l"(*(const unsigned long long*)&gate0_tanh));
                    float2 right0 = _mul_f32x2_8;
                    float2 _mul_f32x2_9;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_9) : "l"(*(const unsigned long long*)&alpha2), "l"(*(const unsigned long long*)&gate1_tanh));
                    float2 right1 = _mul_f32x2_9;
                    float2 _mul_f32x2_10;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_10) : "l"(*(const unsigned long long*)&right0), "l"(*(const unsigned long long*)&gate0_sigmoid));
                    right0 = _mul_f32x2_10;
                    float2 _mul_f32x2_11;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_11) : "l"(*(const unsigned long long*)&right1), "l"(*(const unsigned long long*)&gate1_sigmoid));
                    right1 = _mul_f32x2_11;
                    float2 _mul_f32x2_12;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_12) : "l"(*(const unsigned long long*)&left0), "l"(*(const unsigned long long*)&right0));
                    float2 value0 = _mul_f32x2_12;
                    float2 _mul_f32x2_13;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_13) : "l"(*(const unsigned long long*)&left1), "l"(*(const unsigned long long*)&right1));
                    float2 value1 = _mul_f32x2_13;
                    value00 = value0.x;
                    value01 = value0.y;
                    value10 = value1.x;
                    value11 = value1.y;
                }
                float _fabs_0 = fabsf(value00);
                float _fabs_1 = fabsf(value10);
                float _max_4 = max_noftz(_fabs_0, _fabs_1);
                float block_max0 = _max_4;
                float _fabs_2 = fabsf(value01);
                float _fabs_3 = fabsf(value11);
                float _max_5 = max_noftz(_fabs_2, _fabs_3);
                float block_max1 = _max_5;
                float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, block_max0, 4);
                float _max_6 = max_noftz(block_max0, _shfl_xor_0);
                block_max0 = _max_6;
                float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, block_max1, 4);
                float _max_7 = max_noftz(block_max1, _shfl_xor_1);
                block_max1 = _max_7;
                float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, block_max0, 8);
                float _max_8 = max_noftz(block_max0, _shfl_xor_2);
                block_max0 = _max_8;
                float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, block_max1, 8);
                float _max_9 = max_noftz(block_max1, _shfl_xor_3);
                block_max1 = _max_9;
                float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, block_max0, 16);
                float _max_10 = max_noftz(block_max0, _shfl_xor_4);
                block_max0 = _max_10;
                float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, block_max1, 16);
                float _max_11 = max_noftz(block_max1, _shfl_xor_5);
                block_max1 = _max_11;
                float scale0 = 0.0f;
                float scale1 = 0.0f;
                {
                    float2 _f2_18 = make_float2(block_max0, block_max1);
                    float2 _f2_19 = make_float2(sc, sc);
                    float2 _mul_f32x2_14;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_14) : "l"(*(const unsigned long long*)&_f2_18), "l"(*(const unsigned long long*)&_f2_19));
                    float2 scaled_max = _mul_f32x2_14;
                    float2 _f2_20 = make_float2(0.16666666666666666f, 0.16666666666666666f);
                    float2 _mul_f32x2_15;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_15) : "l"(*(const unsigned long long*)&scaled_max), "l"(*(const unsigned long long*)&_f2_20));
                    scaled_max = _mul_f32x2_15;
                    float _fp8_rt_1;
                    uint16_t _e4m3x2_0;
                    uint32_t _f16x2_0;
                    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_0) : "f"(0.0f), "f"(scaled_max.x));
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_0) : "h"(_e4m3x2_0));
                    uint16_t _fp8_h0_0 = (uint16_t)(_f16x2_0 & 0xFFFFu);
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_1) : "h"(_fp8_h0_0));
                    scale0 = _fp8_rt_1;
                    float _fp8_rt_2;
                    uint16_t _e4m3x2_1;
                    uint32_t _f16x2_1;
                    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_1) : "f"(0.0f), "f"(scaled_max.y));
                    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_1) : "h"(_e4m3x2_1));
                    uint16_t _fp8_h0_1 = (uint16_t)(_f16x2_1 & 0xFFFFu);
                    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_2) : "h"(_fp8_h0_1));
                    scale1 = _fp8_rt_2;
                }
                float inv_scale0 = 0.0f;
                float inv_scale1 = 0.0f;
                if (scale0 != 0.0f) {
                    {
                        float _rcp_6 = approx_rcp(scale0);
                        inv_scale0 = _rcp_6;
                    }
                }
                if (scale1 != 0.0f) {
                    {
                        float _rcp_7 = approx_rcp(scale1);
                        inv_scale1 = _rcp_7;
                    }
                }
                float2 _f2_21 = make_float2(0.0f, 0.0f);
                float2 quant0 = _f2_21;
                float2 _f2_22 = make_float2(0.0f, 0.0f);
                float2 quant1 = _f2_22;
                {
                    float2 _f2_23 = make_float2(value00, value10);
                    float2 _f2_24 = make_float2(inv_scale0, inv_scale0);
                    float2 _mul_f32x2_16;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_16) : "l"(*(const unsigned long long*)&_f2_23), "l"(*(const unsigned long long*)&_f2_24));
                    quant0 = _mul_f32x2_16;
                    float2 _f2_25 = make_float2(value01, value11);
                    float2 _f2_26 = make_float2(inv_scale1, inv_scale1);
                    float2 _mul_f32x2_17;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_17) : "l"(*(const unsigned long long*)&_f2_25), "l"(*(const unsigned long long*)&_f2_26));
                    quant1 = _mul_f32x2_17;
                    float2 _f2_27 = make_float2(sc, sc);
                    float2 scale_c2 = _f2_27;
                    float2 _mul_f32x2_18;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_18) : "l"(*(const unsigned long long*)&quant0), "l"(*(const unsigned long long*)&scale_c2));
                    quant0 = _mul_f32x2_18;
                    float2 _mul_f32x2_19;
                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_19) : "l"(*(const unsigned long long*)&quant1), "l"(*(const unsigned long long*)&scale_c2));
                    quant1 = _mul_f32x2_19;
                    quant_pair[0] = quant0.x;
                    quant_pair[1] = quant0.y;
                }
                uint32_t _slice_lo_mask_0;
                {
                    int _lim_2 = 2;
                    if (_lim_2 <= 0) { _slice_lo_mask_0 = 0u; }
                    else if (_lim_2 >= 8) { _slice_lo_mask_0 = ((1u << 8) - 1u); }
                    else {
                        asm volatile("{"
                            ".reg .u32 t;\n\t"
                            "shl.b32 t, 1, %1;\n\t"
                            "add.u32 %0, t, -1;\n\t"
                            "}" : "=r"(_slice_lo_mask_0) : "r"(_lim_2));
                    }
                }
                uint32_t _slice_hi_mask_0;
                {
                    int _lim_3 = 8;
                    if (_lim_3 <= 0) { _slice_hi_mask_0 = 0u; }
                    else if (_lim_3 >= 8) { _slice_hi_mask_0 = ((1u << 8) - 1u); }
                    else {
                        asm volatile("{"
                            ".reg .u32 t;\n\t"
                            "shl.b32 t, 1, %1;\n\t"
                            "add.u32 %0, t, -1;\n\t"
                            "}" : "=r"(_slice_hi_mask_0) : "r"(_lim_3));
                    }
                }
                if (!(_slice_lo_mask_0 | ~_slice_hi_mask_0 & (1u << 0))) quant_pair[0] = 0.0f;
                if (!(_slice_lo_mask_0 | ~_slice_hi_mask_0 & (1u << 1))) quant_pair[1] = 0.0f;
                if (!(_slice_lo_mask_0 | ~_slice_hi_mask_0 & (1u << 2))) quant_pair[2] = 0.0f;
                if (!(_slice_lo_mask_0 | ~_slice_hi_mask_0 & (1u << 3))) quant_pair[3] = 0.0f;
                if (!(_slice_lo_mask_0 | ~_slice_hi_mask_0 & (1u << 4))) quant_pair[4] = 0.0f;
                if (!(_slice_lo_mask_0 | ~_slice_hi_mask_0 & (1u << 5))) quant_pair[5] = 0.0f;
                if (!(_slice_lo_mask_0 | ~_slice_hi_mask_0 & (1u << 6))) quant_pair[6] = 0.0f;
                if (!(_slice_lo_mask_0 | ~_slice_hi_mask_0 & (1u << 7))) quant_pair[7] = 0.0f;
                uint32_t _fp4_1[1];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_1[0]) : "f"(quant_pair[0]), "f"(quant_pair[1]), "f"(quant_pair[2]), "f"(quant_pair[3]), "f"(quant_pair[4]), "f"(quant_pair[5]), "f"(quant_pair[6]), "f"(quant_pair[7]));
                {
                    quant_pair[0] = quant1.x;
                    quant_pair[1] = quant1.y;
                }
                uint32_t _fp4_2[1];
                asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(_fp4_2[0]) : "f"(quant_pair[0]), "f"(quant_pair[1]), "f"(quant_pair[2]), "f"(quant_pair[3]), "f"(quant_pair[4]), "f"(quant_pair[5]), "f"(quant_pair[6]), "f"(quant_pair[7]));
                if (lane_1 < 4) {
                    int sf_feature = m_tile * 4 + warp_0;
                    int sf_token_group = token0 / 8;
                    int sf_tile_stride = M_out / 64 * 32;
                    int sf_base = n_tile * sf_tile_stride + sf_token_group * (M_out / 64) * 32 + sf_feature / 4 * 32 + token0 % 8 * 4 + sf_feature % 4;
                    if (token0 < valid_rows) {
                        {
                            unsigned short _fp8_pair;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(scale0));
                            *(reinterpret_cast<unsigned char*>(SFC + sf_base) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                        }
                    }
                    if (token1 < valid_rows) {
                        {
                            unsigned short _fp8_pair;
                            asm("cvt.rn.satfinite.e4m3x2.f32 %0, 0f00000000, %1;" : "=h"(_fp8_pair) : "f"(scale1));
                            *(reinterpret_cast<unsigned char*>(SFC + (sf_base + 4)) + (0)) = (unsigned char)(_fp8_pair & 0xFF);
                        }
                    }
                }
                int smem_flat0 = token0 * 64 + base_row;
                int smem_flat1 = token1 * 64 + base_row;
                int smem_index0 = smem_flat0 / 2 ^ smem_flat0 / 256 % 2 * 16;
                int smem_index1 = smem_flat1 / 2 ^ smem_flat1 / 256 % 2 * 16;
                epi_staging[smem_index0] = _fp4_1[0];
                epi_staging[smem_index1] = _fp4_2[0];
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("barrier.sync 7, 128;" ::: "memory");
            if (warp == 0) {
                if (elect_sync()) {
                    int padding_rows = (8 - valid_rows % 8) % 8;
                    tma_store_4d((&C), m_tile * 64, padding_rows, 1073741824, n_tile * 8 - padding_rows + 1073741824, epi_staging_addr);
                }
            }
            asm volatile("cp.async.bulk.commit_group;");
            asm volatile("barrier.sync 7, 128;" ::: "memory");
        }
    }
    // ---- Role: copy_sfb ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 72;");
        { // copy_sfb_main
            unsigned int stage = 0;
            int valid_rows_c = s2a_valid;
            const int lane_0 = lane;
            unsigned int word[1];
            unsigned int _phase_k_done = 1;
            #pragma unroll 1
            for (int _iter_k = 0; _iter_k < K_tiles; _iter_k++) {
                mbarrier_wait(k_done_addr + (stage) * 8, _phase_k_done);
                asm volatile("tcgen05.fence::after_thread_sync;");
                #pragma unroll
                for (int q = 0; q < 8; q++) {
                    word[0] = 0;
                    if (lane_0 < valid_rows_c) {
                        asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&word[0])) : "r"(smem_qsf_addr + (unsigned int)(_iter_k * (BLOCK_K / 16)) + (unsigned int)(q * 4)));
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(taddr + 8 + 160 + stage * 16 + (unsigned int)(q * 2)), "r"(word[0]));
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                asm volatile("tcgen05.fence::before_thread_sync;");
                asm volatile("barrier.sync 4, 128;" ::: "memory");
                if (warp == 4) {
                    if (elect_sync()) {
                        mbarrier_arrive(tmem_sfb_full_addr + (stage) * 8);
                    }
                }
                stage += 1;
                if (stage == 5) { stage = 0; _phase_k_done ^= 1; }
            }
        }
    }
    // ---- Role: load_b ----
    if (warp >= 8 && warp <= 9) {
        { // load_b_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int stage_1 = 0;
            int n_tile_1 = blockIdx.y;
            int local_thread = (warp - 8) * 32 + lane;
            int valid_rows_1 = s2a_valid;
            unsigned int _phase_k_done_1 = 1;
            #pragma unroll 1
            for (int iter_k = 0; iter_k < K_tiles; iter_k++) {
                mbarrier_wait(k_done_addr + (stage_1) * 8, _phase_k_done_1);
                int dst_base = smem_b_addr + stage_1 * 2048;
                for (int row_group = 0; row_group < 1; row_group++) {
                    int elt_offset = local_thread * 32 + row_group * 2048;
                    int row = elt_offset / 256;
                    int col = elt_offset % 256;
                    int src_base = iter_k * (BLOCK_K / 2) + col / 2;
                    int dst_chunk = elt_offset / 2 ^ row % 8 * 16;
                    unsigned int s2a_bw[4];
                    s2a_bw[0] = 0;
                    s2a_bw[1] = 0;
                    s2a_bw[2] = 0;
                    s2a_bw[3] = 0;
                    if (row < valid_rows_1) {
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&s2a_bw[0])), "=r"(*reinterpret_cast<uint32_t*>(&s2a_bw[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&s2a_bw[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&s2a_bw[(0) + 3]))
                            : "r"(smem_qtok_addr + (unsigned int)src_base));
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(dst_base + dst_chunk), "r"(*reinterpret_cast<uint32_t*>(&s2a_bw[0])), "r"(*reinterpret_cast<uint32_t*>(&s2a_bw[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&s2a_bw[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&s2a_bw[(0) + 3])));
                    unsigned int s2a_bw_0[4];
                    s2a_bw_0[0] = 0;
                    s2a_bw_0[1] = 0;
                    s2a_bw_0[2] = 0;
                    s2a_bw_0[3] = 0;
                    if (row < valid_rows_1) {
                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&s2a_bw_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&s2a_bw_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&s2a_bw_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&s2a_bw_0[(0) + 3]))
                            : "r"(smem_qtok_addr + (unsigned int)src_base + 128));
                    }
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                        "r"(dst_base + 1024 + dst_chunk), "r"(*reinterpret_cast<uint32_t*>(&s2a_bw_0[0])), "r"(*reinterpret_cast<uint32_t*>(&s2a_bw_0[(0) + 1])), "r"(*reinterpret_cast<uint32_t*>(&s2a_bw_0[(0) + 2])), "r"(*reinterpret_cast<uint32_t*>(&s2a_bw_0[(0) + 3])));
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 8, 64;" ::: "memory");
                if (warp == 8) {
                    if (elect_sync()) {
                        mbarrier_arrive(b_full_addr + (stage_1) * 8);
                    }
                }
                stage_1 += 1;
                if (stage_1 == 5) { stage_1 = 0; _phase_k_done_1 ^= 1; }
            }
        }
    }
    // ---- Role: load_sfb ----
    if (warp == 10) {
        // idle — no tasks assigned
    }
    // ---- Role: load_a ----
    if (warp == 11) {
        { // load_a_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int stage_2 = 0;
            int m_tile_1 = blockIdx.x;
            int n_tile_2 = blockIdx.y;
            int expert_1 = s2a_expert;
            unsigned int _phase_k_done_2 = 1;
            #pragma unroll 1
            for (int iter_k_1 = 0; iter_k_1 < K_tiles; iter_k_1++) {
                mbarrier_wait(k_done_addr + (stage_2) * 8, _phase_k_done_2);
                int sf_unused = expert_1;
                if (elect_sync()) {
                    tma_4d_gmem2smem(smem_a_addr + stage_2 * 32768, (&A), 0, m_tile_1 * 128, iter_k_1 * 2, sf_unused, a_full_addr + (stage_2) * 8);
                    tma_4d_gmem2smem(smem_a_addr + stage_2 * 32768 + 16384, (&A), 0, m_tile_1 * 128, iter_k_1 * 2 + 1, sf_unused, a_full_addr + (stage_2) * 8);
                    mbarrier_arrive_expect_tx(a_full_addr + (stage_2) * 8, 32768);
                }
                stage_2 += 1;
                if (stage_2 == 5) { stage_2 = 0; _phase_k_done_2 ^= 1; }
            }
        }
    }
    // ---- Role: load_sfa ----
    if (warp == 12) {
        { // load_sfa_main
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int stage_3 = 0;
            int m_tile_2 = blockIdx.x;
            int n_tile_3 = blockIdx.y;
            int expert_2 = s2a_expert;
            unsigned int _phase_sfa_free = 1;
            #pragma unroll 1
            for (int iter_k_2 = 0; iter_k_2 < K_tiles; iter_k_2++) {
                mbarrier_wait(sfa_free_addr + (stage_3) * 8, _phase_sfa_free);
                int sf_tile = (expert_2 * grid_m + m_tile_2) * K_tiles + iter_k_2;
                if (elect_sync()) {
                    tma_4d_gmem2smem(smem_sfa_addr + stage_3 * 4096, (&SFA), 0, 0, iter_k_2 * 8, expert_2 * grid_m + m_tile_2, sfa_full_addr + (stage_3) * 8);
                    mbarrier_arrive_expect_tx(sfa_full_addr + (stage_3) * 8, 4096);
                }
                stage_3 += 1;
                if (stage_3 == 5) { stage_3 = 0; _phase_sfa_free ^= 1; }
            }
        }
    }
    // ---- Role: copy_sfa ----
    if (warp == 13) {
        { // copy_sfa_main
            unsigned int stage_4 = 0;
            unsigned int _phase_sfa_full = 0;
            unsigned int _phase_k_done_3 = 1;
            #pragma unroll 1
            for (int _iter_k_1 = 0; _iter_k_1 < K_tiles; _iter_k_1++) {
                mbarrier_wait(sfa_full_addr + (stage_4) * 8, _phase_sfa_full);
                mbarrier_wait(k_done_addr + (stage_4) * 8, _phase_k_done_3);
                asm volatile("tcgen05.fence::after_thread_sync;");
                if (elect_sync()) {
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_0 = ((((uint64_t)(smem_sfa_addr + stage_4 * 4096)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                        asm volatile(
                            "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                            :: "r"((uint32_t)((unsigned int)tmem_sfa + stage_4 * 32)), "l"(_tcgen05_cp_desc_0)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_1 = ((((uint64_t)(smem_sfa_addr + stage_4 * 4096 + 512)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                        asm volatile(
                            "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                            :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_4 * 32 + 4))), "l"(_tcgen05_cp_desc_1)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_2 = ((((uint64_t)(smem_sfa_addr + stage_4 * 4096 + 1024)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                        asm volatile(
                            "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                            :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_4 * 32 + 8))), "l"(_tcgen05_cp_desc_2)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_3 = ((((uint64_t)(smem_sfa_addr + stage_4 * 4096 + 1536)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                        asm volatile(
                            "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                            :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_4 * 32 + 12))), "l"(_tcgen05_cp_desc_3)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_4 = ((((uint64_t)(smem_sfa_addr + stage_4 * 4096 + 2048)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                        asm volatile(
                            "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                            :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_4 * 32 + 16))), "l"(_tcgen05_cp_desc_4)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_5 = ((((uint64_t)(smem_sfa_addr + stage_4 * 4096 + 2560)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                        asm volatile(
                            "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                            :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_4 * 32 + 20))), "l"(_tcgen05_cp_desc_5)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_6 = ((((uint64_t)(smem_sfa_addr + stage_4 * 4096 + 3072)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                        asm volatile(
                            "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                            :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_4 * 32 + 24))), "l"(_tcgen05_cp_desc_6)
                            : "memory");
                    }
                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                    #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                    #endif
                    {
                        uint64_t _tcgen05_cp_desc_7 = ((((uint64_t)(smem_sfa_addr + stage_4 * 4096 + 3584)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                        asm volatile(
                            "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                            :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_4 * 32 + 28))), "l"(_tcgen05_cp_desc_7)
                            : "memory");
                    }
                }
                elect_commit2(tmem_sfa_full_addr + (stage_4) * 8, sfa_free_addr + (stage_4) * 8);
                stage_4 += 1;
                if (stage_4 == 5) { stage_4 = 0; _phase_sfa_full ^= 1; _phase_k_done_3 ^= 1; }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 14) {
        { // mma_main
            unsigned int stage_5 = 0;
            unsigned int _phase_a_full = 0;
            unsigned int _phase_b_full = 0;
            unsigned int _phase_tmem_sfa_full = 0;
            unsigned int _phase_tmem_sfb_full = 0;
            #pragma unroll 1
            for (int iter_k_3 = 0; iter_k_3 < K_tiles; iter_k_3++) {
                mbarrier_wait(a_full_addr + (stage_5) * 8, _phase_a_full);
                mbarrier_wait(b_full_addr + (stage_5) * 8, _phase_b_full);
                mbarrier_wait(tmem_sfa_full_addr + (stage_5) * 8, _phase_tmem_sfa_full);
                mbarrier_wait(tmem_sfb_full_addr + (stage_5) * 8, _phase_tmem_sfb_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_a_lo_0 = make_warp_uniform((((smem_a_addr) >> 4) & 0x3FFF) + (stage_5) * 2048);
                int _mma_b_lo_0 = make_warp_uniform((((smem_b_addr) >> 4) & 0x3FFF) + (stage_5) * 128);
                if (elect_sync()) {
                    {
                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                        uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                        tcgen05_mma_mxf4nvf4_bs(tmem_accum, a_desc + 0, b_desc + 0,
                            0x8020480U, (unsigned int)tmem_sfa + stage_5 * 32 + 0, (unsigned int)tmem_sfb + stage_5 * 16 + 0, ((((1) ? ((iter_k_3 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                    }
                }
                int _mma_a_lo_1 = make_warp_uniform((((smem_a_addr + 32) >> 4) & 0x3FFF) + (stage_5) * 2048);
                int _mma_b_lo_1 = make_warp_uniform((((smem_b_addr + 32) >> 4) & 0x3FFF) + (stage_5) * 128);
                if (elect_sync()) {
                    {
                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                        uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);

                        tcgen05_mma_mxf4nvf4_bs(tmem_accum, a_desc + 0, b_desc + 0,
                            0x8020480U, (unsigned int)tmem_sfa + (stage_5 * 32 + 4) + 0, (unsigned int)tmem_sfb + (stage_5 * 16 + 2) + 0, ((((0) ? ((iter_k_3 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                    }
                }
                int _mma_a_lo_2 = make_warp_uniform((((smem_a_addr + 64) >> 4) & 0x3FFF) + (stage_5) * 2048);
                int _mma_b_lo_2 = make_warp_uniform((((smem_b_addr + 64) >> 4) & 0x3FFF) + (stage_5) * 128);
                if (elect_sync()) {
                    {
                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_2) | ((uint64_t)0x40004040 << 32);
                        uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_2) | ((uint64_t)0x40004040 << 32);

                        tcgen05_mma_mxf4nvf4_bs(tmem_accum, a_desc + 0, b_desc + 0,
                            0x8020480U, (unsigned int)tmem_sfa + (stage_5 * 32 + 8) + 0, (unsigned int)tmem_sfb + (stage_5 * 16 + 4) + 0, ((((0) ? ((iter_k_3 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                    }
                }
                int _mma_a_lo_3 = make_warp_uniform((((smem_a_addr + 96) >> 4) & 0x3FFF) + (stage_5) * 2048);
                int _mma_b_lo_3 = make_warp_uniform((((smem_b_addr + 96) >> 4) & 0x3FFF) + (stage_5) * 128);
                if (elect_sync()) {
                    {
                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_3) | ((uint64_t)0x40004040 << 32);
                        uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_3) | ((uint64_t)0x40004040 << 32);

                        tcgen05_mma_mxf4nvf4_bs(tmem_accum, a_desc + 0, b_desc + 0,
                            0x8020480U, (unsigned int)tmem_sfa + (stage_5 * 32 + 12) + 0, (unsigned int)tmem_sfb + (stage_5 * 16 + 6) + 0, ((((0) ? ((iter_k_3 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                    }
                }
                int _mma_a_lo_4 = make_warp_uniform((((smem_a_addr + 16384) >> 4) & 0x3FFF) + (stage_5) * 2048);
                int _mma_b_lo_4 = make_warp_uniform((((smem_b_addr + 1024) >> 4) & 0x3FFF) + (stage_5) * 128);
                if (elect_sync()) {
                    {
                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_4) | ((uint64_t)0x40004040 << 32);
                        uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_4) | ((uint64_t)0x40004040 << 32);

                        tcgen05_mma_mxf4nvf4_bs(tmem_accum, a_desc + 0, b_desc + 0,
                            0x8020480U, (unsigned int)tmem_sfa + (stage_5 * 32 + 16) + 0, (unsigned int)tmem_sfb + (stage_5 * 16 + 8) + 0, ((((0) ? ((iter_k_3 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                    }
                }
                int _mma_a_lo_5 = make_warp_uniform((((smem_a_addr + 16416) >> 4) & 0x3FFF) + (stage_5) * 2048);
                int _mma_b_lo_5 = make_warp_uniform((((smem_b_addr + 1056) >> 4) & 0x3FFF) + (stage_5) * 128);
                if (elect_sync()) {
                    {
                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_5) | ((uint64_t)0x40004040 << 32);
                        uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_5) | ((uint64_t)0x40004040 << 32);

                        tcgen05_mma_mxf4nvf4_bs(tmem_accum, a_desc + 0, b_desc + 0,
                            0x8020480U, (unsigned int)tmem_sfa + (stage_5 * 32 + 20) + 0, (unsigned int)tmem_sfb + (stage_5 * 16 + 10) + 0, ((((0) ? ((iter_k_3 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                    }
                }
                int _mma_a_lo_6 = make_warp_uniform((((smem_a_addr + 16448) >> 4) & 0x3FFF) + (stage_5) * 2048);
                int _mma_b_lo_6 = make_warp_uniform((((smem_b_addr + 1088) >> 4) & 0x3FFF) + (stage_5) * 128);
                if (elect_sync()) {
                    {
                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_6) | ((uint64_t)0x40004040 << 32);
                        uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_6) | ((uint64_t)0x40004040 << 32);

                        tcgen05_mma_mxf4nvf4_bs(tmem_accum, a_desc + 0, b_desc + 0,
                            0x8020480U, (unsigned int)tmem_sfa + (stage_5 * 32 + 24) + 0, (unsigned int)tmem_sfb + (stage_5 * 16 + 12) + 0, ((((0) ? ((iter_k_3 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                    }
                }
                int _mma_a_lo_7 = make_warp_uniform((((smem_a_addr + 16480) >> 4) & 0x3FFF) + (stage_5) * 2048);
                int _mma_b_lo_7 = make_warp_uniform((((smem_b_addr + 1120) >> 4) & 0x3FFF) + (stage_5) * 128);
                if (elect_sync()) {
                    {
                        uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_7) | ((uint64_t)0x40004040 << 32);
                        uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_7) | ((uint64_t)0x40004040 << 32);

                        tcgen05_mma_mxf4nvf4_bs(tmem_accum, a_desc + 0, b_desc + 0,
                            0x8020480U, (unsigned int)tmem_sfa + (stage_5 * 32 + 28) + 0, (unsigned int)tmem_sfb + (stage_5 * 16 + 14) + 0, ((((0) ? ((iter_k_3 == 0) ? 1 : 0) : 0)) ? 0 : 1));
                    }
                }
                if (iter_k_3 + 1 == K_tiles) {
                    elect_commit2(k_done_addr + (stage_5) * 8, mma_full_addr);
                } else {
                    elect_commit(k_done_addr + (stage_5) * 8);
                }
                stage_5 += 1;
                if (stage_5 == 5) { stage_5 = 0; _phase_a_full ^= 1; _phase_b_full ^= 1; _phase_tmem_sfa_full ^= 1; _phase_tmem_sfb_full ^= 1; }
            }
        }
    }
    // ---- Role: padding ----
    if (warp == 15) {
        // idle — no tasks assigned
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(256));
    }
}

} // extern "C"
