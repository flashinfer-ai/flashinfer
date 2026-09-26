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
template <int N>
struct __align__(128) CakeTensorMapPack { CakeTensorMap maps[N]; };

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
#define TMEM_PARTIALS_OFFSET 0
#define NUM_AB_PIPE_STAGES 7
#define NUM_PARTIAL_PIPE_STAGES 4
#define NUM_SCALE_PIPE_STAGES 1
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 16384
#define SMEM_SMEM_B_OFF 115712
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 16384
#define SMEM_SMEM_B_HALF0_OFF 115712
#define SMEM_SMEM_B_HALF0_STAGE_BYTES 8192
#define SMEM_SMEM_B_HALF0_STRIDE 16384
#define SMEM_SMEM_B_HALF1_OFF 123904
#define SMEM_SMEM_B_HALF1_STAGE_BYTES 8192
#define SMEM_SMEM_B_HALF1_STRIDE 16384
#define SMEM_ROWMAX_XCHG_OFF 1024
#define SMEM_ROWMAX_XCHG_STAGE_BYTES 114688
#define SMEM_ROWMAX_XCHG_STRIDE 114688
#define SMEM_SMEM_ASCALE_OFF 230400
#define SMEM_SMEM_ASCALE_STAGE_BYTES 2048
#define SMEM_SMEM_ASCALE_STRIDE 2048
#define SMEM_TOTAL 232448
#define THREADS 384

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

__global__ __launch_bounds__(384, 1) __cluster_dims__(2,1,1) void
kernel_cake_grouped_fp8_fused_silu_quant_c5c857d1d7fd38445fd2(CakeTensorMap const* A, CakeTensorMap const* B, uint8_t* __restrict__ out_q, float* __restrict__ out_s, float* __restrict__ a_scale, float* __restrict__ b_scale, int* __restrict__ m_indices, int M, int N, int K, int G)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);

    const int mbar_base = smem;
    #define ab_full_addr (mbar_base + 0)
    #define ab_free_addr (mbar_base + 56)
    #define partial_full_addr (mbar_base + 112)
    #define partial_free_addr (mbar_base + 144)
    #define scale_full_addr (mbar_base + 176)
    #define scale_free_addr (mbar_base + 184)
    #define producers_done_addr (mbar_base + 192)
    #define pair_exit_addr (mbar_base + 200)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(A)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(B)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + 115712);
    const int smem_b_addr = smem + 115712;
    uint8_t* smem_b_half0 = reinterpret_cast<uint8_t*>(smem_raw + 115712);
    const int smem_b_half0_addr = smem + 115712;
    uint8_t* smem_b_half1 = reinterpret_cast<uint8_t*>(smem_raw + 123904);
    const int smem_b_half1_addr = smem + 123904;
    float* rowmax_xchg = reinterpret_cast<float*>(smem_raw + 1024);
    const int rowmax_xchg_addr = smem + 1024;
    float* smem_ascale = reinterpret_cast<float*>(smem_raw + 230400);
    const int smem_ascale_addr = smem + 230400;
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 208);
    int taddr;
    int tmem_partials;

    // Mbarrier init (8 pipeline groups, 0 ordered-sequence groups, 26 barriers)
    // Mbarriers at smem_raw[0..208)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'ab_pipe' ---
            // ab_full: 7 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            mbarrier_init(smem + 8, 2);
            mbarrier_init(smem + 16, 2);
            mbarrier_init(smem + 24, 2);
            mbarrier_init(smem + 32, 2);
            mbarrier_init(smem + 40, 2);
            mbarrier_init(smem + 48, 2);
            // ab_free: 7 barriers, init_count=1
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            // --- pipeline 'partial_pipe' ---
            // partial_full: 4 barriers, init_count=1
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            // partial_free: 4 barriers, init_count=16
            mbarrier_init(smem + 144, 16);
            mbarrier_init(smem + 152, 16);
            mbarrier_init(smem + 160, 16);
            mbarrier_init(smem + 168, 16);
            // --- pipeline 'scale_pipe' ---
            // scale_full: 1 barriers, init_count=32
            mbarrier_init(smem + 176, 32);
            // scale_free: 1 barriers, init_count=256
            mbarrier_init(smem + 184, 256);
            // producers_done: 1 barriers, init_count=3
            mbarrier_init(smem + 192, 3);
            // pair_exit: 1 barriers, init_count=1
            mbarrier_init(smem + 200, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 64;");
    }

    // ---- Role: acc_epi ----
    if (warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 216;");
        { // acc_epi_main
            if (warp == 0) {
                int _tmem_hold_0 = smem + 208;
                asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold_0), "r"(512) : "memory");
            }
            asm volatile("barrier.sync 9, 288;" ::: "memory");
            asm volatile("tcgen05.fence::after_thread_sync;");
            taddr = tmem_addr_storage[0];
            tmem_partials = taddr;
            if (warp == 0) {
                asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
            }
            unsigned int partial_stage = 0;
            unsigned int consumer_ab_cursor = 0;
            unsigned int epi_panel_cursor = 0;
            float b_scale_prefetch[4];
            float b_scale_prefetch_u[4];
            int b_scale_primed = 0;
            unsigned int scale_stage = 0;
            const int acc_wg = warp / 4;
            const int acc_warp_in_wg = warp % 4;
            int wg_col_base = acc_wg * 128;
            int row_base = acc_warp_in_wg * 32;
            int row = row_base + lane;
            int m_tiles = (M - 1) / 256 + 1;
            int n_tiles = N / 256;
            int total_tiles = m_tiles * n_tiles;
            int nblk = (M - 1) / 128 + 1;
            int g_blk_lo = -1;
            int g_blk_hi = -1;
            if (nblk > lane) {
                g_blk_lo = m_indices[lane * 128];
            }
            if (nblk > lane + 32) {
                g_blk_hi = m_indices[(lane + 32) * 128];
            }
            int pair_lo = 0;
            int pair_hi = 0;
            int pairs_total = 0;
            int cur_e = -1;
            int local_i = 0;
            #pragma unroll 1
            for (int kb = 0; kb < nblk; kb++) {
                int _shfl_32 = __shfl_sync(0xFFFFFFFF, g_blk_lo, kb & 31);
                int gk = _shfl_32;
                int _shfl_33 = __shfl_sync(0xFFFFFFFF, g_blk_hi, kb & 31);
                int gk_hi = _shfl_33;
                if (kb >= 32) {
                    gk = gk_hi;
                }
                if (gk != cur_e) {
                    cur_e = gk;
                    local_i = -1;
                }
                local_i += 1;
                if ((local_i & 1) == 0) {
                    if (lane == (pairs_total & 31)) {
                        if (pairs_total < 32) {
                            pair_lo = kb;
                        }
                        if (pairs_total >= 32) {
                            pair_hi = kb;
                        }
                    }
                    pairs_total += 1;
                }
            }
            total_tiles = pairs_total * n_tiles;
            unsigned int _phase_scale_full = 0;
            unsigned int _phase_partial_full = 0;
            #pragma unroll 1
            for (int tile_id = cluster_id; tile_id < total_tiles; tile_id += num_clusters) {
                int pair_id = tile_id / n_tiles;
                int n_tile = tile_id % n_tiles;
                int _shfl_34 = __shfl_sync(0xFFFFFFFF, pair_lo, pair_id & 31);
                int b0 = _shfl_34;
                int _shfl_35 = __shfl_sync(0xFFFFFFFF, pair_hi, pair_id & 31);
                int b0_hi = _shfl_35;
                if (pair_id >= 32) {
                    b0 = b0_hi;
                }
                int _shfl_36 = __shfl_sync(0xFFFFFFFF, g_blk_lo, b0 & 31);
                int group = _shfl_36;
                int _shfl_37 = __shfl_sync(0xFFFFFFFF, g_blk_hi, b0 & 31);
                int group_hi = _shfl_37;
                if (b0 >= 32) {
                    group = group_hi;
                }
                int bn = b0 + 1;
                int _shfl_38 = __shfl_sync(0xFFFFFFFF, g_blk_lo, bn & 31);
                int gn = _shfl_38;
                int _shfl_39 = __shfl_sync(0xFFFFFFFF, g_blk_hi, bn & 31);
                int gn_hi = _shfl_39;
                if (bn >= 32) {
                    gn = gn_hi;
                }
                int b1 = -1;
                if (bn < nblk) {
                    if (gn == group) {
                        b1 = bn;
                    }
                }
                int my_blk = b0;
                if (cta_rank == 1) {
                    my_blk = b1;
                }
                int m_tile = b0;
                int my_rows = 0;
                if (my_blk >= 0) {
                    m_tile = my_blk;
                    my_rows = M - my_blk * 128;
                    if (my_rows > 128) {
                        my_rows = 128;
                    }
                }
                int cluster_m_base = m_tile * 128 - cta_rank * 128;
                int tile_rows = 256;
                int run_begin = cta_rank * 128;
                int run_end = run_begin + my_rows;
                int next_tile = (unsigned int)tile_id + num_clusters;
                int has_next = 0;
                if (next_tile < total_tiles) {
                    has_next = 1;
                }
                int next_pair = next_tile / n_tiles;
                int next_n_tile = next_tile % n_tiles;
                int _shfl_40 = __shfl_sync(0xFFFFFFFF, pair_lo, next_pair & 31);
                int nb0 = _shfl_40;
                int _shfl_41 = __shfl_sync(0xFFFFFFFF, pair_hi, next_pair & 31);
                int nb0_hi = _shfl_41;
                if (next_pair >= 32) {
                    nb0 = nb0_hi;
                }
                int _shfl_42 = __shfl_sync(0xFFFFFFFF, g_blk_lo, nb0 & 31);
                int next_group = _shfl_42;
                int _shfl_43 = __shfl_sync(0xFFFFFFFF, g_blk_hi, nb0 & 31);
                int next_group_hi = _shfl_43;
                if (nb0 >= 32) {
                    next_group = next_group_hi;
                }
                float acc0[32];
                float acc1[32];
                float acc2[32];
                float acc3[32];
                acc0[0] = 0.0f;
                acc0[1] = 0.0f;
                acc0[2] = 0.0f;
                acc0[3] = 0.0f;
                acc0[4] = 0.0f;
                acc0[5] = 0.0f;
                acc0[6] = 0.0f;
                acc0[7] = 0.0f;
                acc0[8] = 0.0f;
                acc0[9] = 0.0f;
                acc0[10] = 0.0f;
                acc0[11] = 0.0f;
                acc0[12] = 0.0f;
                acc0[13] = 0.0f;
                acc0[14] = 0.0f;
                acc0[15] = 0.0f;
                acc0[16] = 0.0f;
                acc0[17] = 0.0f;
                acc0[18] = 0.0f;
                acc0[19] = 0.0f;
                acc0[20] = 0.0f;
                acc0[21] = 0.0f;
                acc0[22] = 0.0f;
                acc0[23] = 0.0f;
                acc0[24] = 0.0f;
                acc0[25] = 0.0f;
                acc0[26] = 0.0f;
                acc0[27] = 0.0f;
                acc0[28] = 0.0f;
                acc0[29] = 0.0f;
                acc0[30] = 0.0f;
                acc0[31] = 0.0f;
                acc1[0] = 0.0f;
                acc1[1] = 0.0f;
                acc1[2] = 0.0f;
                acc1[3] = 0.0f;
                acc1[4] = 0.0f;
                acc1[5] = 0.0f;
                acc1[6] = 0.0f;
                acc1[7] = 0.0f;
                acc1[8] = 0.0f;
                acc1[9] = 0.0f;
                acc1[10] = 0.0f;
                acc1[11] = 0.0f;
                acc1[12] = 0.0f;
                acc1[13] = 0.0f;
                acc1[14] = 0.0f;
                acc1[15] = 0.0f;
                acc1[16] = 0.0f;
                acc1[17] = 0.0f;
                acc1[18] = 0.0f;
                acc1[19] = 0.0f;
                acc1[20] = 0.0f;
                acc1[21] = 0.0f;
                acc1[22] = 0.0f;
                acc1[23] = 0.0f;
                acc1[24] = 0.0f;
                acc1[25] = 0.0f;
                acc1[26] = 0.0f;
                acc1[27] = 0.0f;
                acc1[28] = 0.0f;
                acc1[29] = 0.0f;
                acc1[30] = 0.0f;
                acc1[31] = 0.0f;
                acc2[0] = 0.0f;
                acc2[1] = 0.0f;
                acc2[2] = 0.0f;
                acc2[3] = 0.0f;
                acc2[4] = 0.0f;
                acc2[5] = 0.0f;
                acc2[6] = 0.0f;
                acc2[7] = 0.0f;
                acc2[8] = 0.0f;
                acc2[9] = 0.0f;
                acc2[10] = 0.0f;
                acc2[11] = 0.0f;
                acc2[12] = 0.0f;
                acc2[13] = 0.0f;
                acc2[14] = 0.0f;
                acc2[15] = 0.0f;
                acc2[16] = 0.0f;
                acc2[17] = 0.0f;
                acc2[18] = 0.0f;
                acc2[19] = 0.0f;
                acc2[20] = 0.0f;
                acc2[21] = 0.0f;
                acc2[22] = 0.0f;
                acc2[23] = 0.0f;
                acc2[24] = 0.0f;
                acc2[25] = 0.0f;
                acc2[26] = 0.0f;
                acc2[27] = 0.0f;
                acc2[28] = 0.0f;
                acc2[29] = 0.0f;
                acc2[30] = 0.0f;
                acc2[31] = 0.0f;
                acc3[0] = 0.0f;
                acc3[1] = 0.0f;
                acc3[2] = 0.0f;
                acc3[3] = 0.0f;
                acc3[4] = 0.0f;
                acc3[5] = 0.0f;
                acc3[6] = 0.0f;
                acc3[7] = 0.0f;
                acc3[8] = 0.0f;
                acc3[9] = 0.0f;
                acc3[10] = 0.0f;
                acc3[11] = 0.0f;
                acc3[12] = 0.0f;
                acc3[13] = 0.0f;
                acc3[14] = 0.0f;
                acc3[15] = 0.0f;
                acc3[16] = 0.0f;
                acc3[17] = 0.0f;
                acc3[18] = 0.0f;
                acc3[19] = 0.0f;
                acc3[20] = 0.0f;
                acc3[21] = 0.0f;
                acc3[22] = 0.0f;
                acc3[23] = 0.0f;
                acc3[24] = 0.0f;
                acc3[25] = 0.0f;
                acc3[26] = 0.0f;
                acc3[27] = 0.0f;
                acc3[28] = 0.0f;
                acc3[29] = 0.0f;
                acc3[30] = 0.0f;
                acc3[31] = 0.0f;
                float partial_fragment[16];
                int k_blocks = K / 128;
                int scale_groups = k_blocks / 4;
                long long b_scale_base = group;
                b_scale_base = (b_scale_base * (long long)(N / 128) + (long long)n_tile) * (long long)k_blocks;
                long long b_scale_base_u = b_scale_base + (long long)(N / 2 / 128 * k_blocks);
                if (b_scale_primed == 0) {
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(b_scale + b_scale_base);
                        b_scale_prefetch[0 + 0] = _v4.x;
                        b_scale_prefetch[0 + 1] = _v4.y;
                        b_scale_prefetch[0 + 2] = _v4.z;
                        b_scale_prefetch[0 + 3] = _v4.w;
                    }
                    {
                        float4 _v4 = *reinterpret_cast<const float4*>(b_scale + b_scale_base_u);
                        b_scale_prefetch_u[0 + 0] = _v4.x;
                        b_scale_prefetch_u[0 + 1] = _v4.y;
                        b_scale_prefetch_u[0 + 2] = _v4.z;
                        b_scale_prefetch_u[0 + 3] = _v4.w;
                    }
                }
                b_scale_primed = 0;
                #pragma unroll 1
                for (int scale_group = 0; scale_group < scale_groups; scale_group++) {
                    float scale_products[4];
                    float scale_products_u[4];
                    mbarrier_wait(scale_full_addr + (scale_stage) * 8, _phase_scale_full);
                    int scale_base = scale_stage * 128 * 4 + (unsigned int)(row * 4);
                    #pragma unroll
                    for (int preload_k = 0; preload_k < 4; preload_k++) {
                        float a_s = smem_ascale[scale_base + preload_k];
                        scale_products[preload_k] = a_s * b_scale_prefetch[preload_k];
                        scale_products_u[preload_k] = a_s * b_scale_prefetch_u[preload_k];
                    }
                    mbarrier_arrive(scale_free_addr + (scale_stage) * 8);
                    _phase_scale_full ^= 1;
                    if (scale_groups > scale_group + 1) {
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(b_scale + b_scale_base + (long long)((scale_group + 1) * 4));
                            b_scale_prefetch[0 + 0] = _v4.x;
                            b_scale_prefetch[0 + 1] = _v4.y;
                            b_scale_prefetch[0 + 2] = _v4.z;
                            b_scale_prefetch[0 + 3] = _v4.w;
                        }
                        {
                            float4 _v4 = *reinterpret_cast<const float4*>(b_scale + b_scale_base_u + (long long)((scale_group + 1) * 4));
                            b_scale_prefetch_u[0 + 0] = _v4.x;
                            b_scale_prefetch_u[0 + 1] = _v4.y;
                            b_scale_prefetch_u[0 + 2] = _v4.z;
                            b_scale_prefetch_u[0 + 3] = _v4.w;
                        }
                    }
                    if (scale_groups <= scale_group + 1) {
                        if (has_next != 0) {
                            long long next_b_scale_base = next_group;
                            next_b_scale_base = (next_b_scale_base * (long long)(N / 128) + (long long)next_n_tile) * (long long)k_blocks;
                            {
                                float4 _v4 = *reinterpret_cast<const float4*>(b_scale + next_b_scale_base);
                                b_scale_prefetch[0 + 0] = _v4.x;
                                b_scale_prefetch[0 + 1] = _v4.y;
                                b_scale_prefetch[0 + 2] = _v4.z;
                                b_scale_prefetch[0 + 3] = _v4.w;
                            }
                            {
                                float4 _v4 = *reinterpret_cast<const float4*>(b_scale + next_b_scale_base + (long long)(N / 2 / 128 * k_blocks));
                                b_scale_prefetch_u[0 + 0] = _v4.x;
                                b_scale_prefetch_u[0 + 1] = _v4.y;
                                b_scale_prefetch_u[0 + 2] = _v4.z;
                                b_scale_prefetch_u[0 + 3] = _v4.w;
                            }
                            b_scale_primed = 1;
                        }
                    }
                    #pragma unroll
                    for (int k_inner = 0; k_inner < 4; k_inner++) {
                        mbarrier_wait(partial_full_addr + (partial_stage) * 8, _phase_partial_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        float combined = scale_products[k_inner];
                        float combined_u = scale_products_u[k_inner];
                        int _trl_addr_7 = tmem_partials + (partial_stage * 128 + (unsigned int)(acc_wg * 64)) + (row_base << 16);
                        tmem_ld_x16_wait(&partial_fragment[0], _trl_addr_7);
                        {
                            unsigned long long _fma_acc_scale2_8;
                            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_8) : "f"(combined));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[0]), "+f"(acc0[1]) : "f"(partial_fragment[0]), "f"(partial_fragment[1]), "l"(_fma_acc_scale2_8));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[2]), "+f"(acc0[3]) : "f"(partial_fragment[2]), "f"(partial_fragment[3]), "l"(_fma_acc_scale2_8));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[4]), "+f"(acc0[5]) : "f"(partial_fragment[4]), "f"(partial_fragment[5]), "l"(_fma_acc_scale2_8));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[6]), "+f"(acc0[7]) : "f"(partial_fragment[6]), "f"(partial_fragment[7]), "l"(_fma_acc_scale2_8));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[8]), "+f"(acc0[9]) : "f"(partial_fragment[8]), "f"(partial_fragment[9]), "l"(_fma_acc_scale2_8));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[10]), "+f"(acc0[11]) : "f"(partial_fragment[10]), "f"(partial_fragment[11]), "l"(_fma_acc_scale2_8));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[12]), "+f"(acc0[13]) : "f"(partial_fragment[12]), "f"(partial_fragment[13]), "l"(_fma_acc_scale2_8));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[14]), "+f"(acc0[15]) : "f"(partial_fragment[14]), "f"(partial_fragment[15]), "l"(_fma_acc_scale2_8));
                        }
                        int _trl_addr_9 = tmem_partials + (partial_stage * 128 + (unsigned int)(acc_wg * 64) + 16) + (row_base << 16);
                        tmem_ld_x16_wait(&partial_fragment[0], _trl_addr_9);
                        {
                            unsigned long long _fma_acc_scale2_10;
                            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_10) : "f"(combined));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[16]), "+f"(acc0[17]) : "f"(partial_fragment[0]), "f"(partial_fragment[1]), "l"(_fma_acc_scale2_10));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[18]), "+f"(acc0[19]) : "f"(partial_fragment[2]), "f"(partial_fragment[3]), "l"(_fma_acc_scale2_10));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[20]), "+f"(acc0[21]) : "f"(partial_fragment[4]), "f"(partial_fragment[5]), "l"(_fma_acc_scale2_10));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[22]), "+f"(acc0[23]) : "f"(partial_fragment[6]), "f"(partial_fragment[7]), "l"(_fma_acc_scale2_10));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[24]), "+f"(acc0[25]) : "f"(partial_fragment[8]), "f"(partial_fragment[9]), "l"(_fma_acc_scale2_10));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[26]), "+f"(acc0[27]) : "f"(partial_fragment[10]), "f"(partial_fragment[11]), "l"(_fma_acc_scale2_10));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[28]), "+f"(acc0[29]) : "f"(partial_fragment[12]), "f"(partial_fragment[13]), "l"(_fma_acc_scale2_10));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc0[30]), "+f"(acc0[31]) : "f"(partial_fragment[14]), "f"(partial_fragment[15]), "l"(_fma_acc_scale2_10));
                        }
                        int _trl_addr_11 = tmem_partials + (partial_stage * 128 + (unsigned int)(acc_wg * 64) + 32) + (row_base << 16);
                        tmem_ld_x16_wait(&partial_fragment[0], _trl_addr_11);
                        {
                            unsigned long long _fma_acc_scale2_12;
                            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_12) : "f"(combined));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[0]), "+f"(acc1[1]) : "f"(partial_fragment[0]), "f"(partial_fragment[1]), "l"(_fma_acc_scale2_12));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[2]), "+f"(acc1[3]) : "f"(partial_fragment[2]), "f"(partial_fragment[3]), "l"(_fma_acc_scale2_12));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[4]), "+f"(acc1[5]) : "f"(partial_fragment[4]), "f"(partial_fragment[5]), "l"(_fma_acc_scale2_12));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[6]), "+f"(acc1[7]) : "f"(partial_fragment[6]), "f"(partial_fragment[7]), "l"(_fma_acc_scale2_12));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[8]), "+f"(acc1[9]) : "f"(partial_fragment[8]), "f"(partial_fragment[9]), "l"(_fma_acc_scale2_12));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[10]), "+f"(acc1[11]) : "f"(partial_fragment[10]), "f"(partial_fragment[11]), "l"(_fma_acc_scale2_12));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[12]), "+f"(acc1[13]) : "f"(partial_fragment[12]), "f"(partial_fragment[13]), "l"(_fma_acc_scale2_12));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[14]), "+f"(acc1[15]) : "f"(partial_fragment[14]), "f"(partial_fragment[15]), "l"(_fma_acc_scale2_12));
                        }
                        int _trl_addr_13 = tmem_partials + (partial_stage * 128 + (unsigned int)(acc_wg * 64) + 32 + 16) + (row_base << 16);
                        tmem_ld_x16_wait(&partial_fragment[0], _trl_addr_13);
                        {
                            unsigned long long _fma_acc_scale2_14;
                            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_14) : "f"(combined));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[16]), "+f"(acc1[17]) : "f"(partial_fragment[0]), "f"(partial_fragment[1]), "l"(_fma_acc_scale2_14));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[18]), "+f"(acc1[19]) : "f"(partial_fragment[2]), "f"(partial_fragment[3]), "l"(_fma_acc_scale2_14));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[20]), "+f"(acc1[21]) : "f"(partial_fragment[4]), "f"(partial_fragment[5]), "l"(_fma_acc_scale2_14));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[22]), "+f"(acc1[23]) : "f"(partial_fragment[6]), "f"(partial_fragment[7]), "l"(_fma_acc_scale2_14));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[24]), "+f"(acc1[25]) : "f"(partial_fragment[8]), "f"(partial_fragment[9]), "l"(_fma_acc_scale2_14));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[26]), "+f"(acc1[27]) : "f"(partial_fragment[10]), "f"(partial_fragment[11]), "l"(_fma_acc_scale2_14));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[28]), "+f"(acc1[29]) : "f"(partial_fragment[12]), "f"(partial_fragment[13]), "l"(_fma_acc_scale2_14));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc1[30]), "+f"(acc1[31]) : "f"(partial_fragment[14]), "f"(partial_fragment[15]), "l"(_fma_acc_scale2_14));
                        }
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        if (elect_sync()) {
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                :: "r"((partial_free_addr + (partial_stage) * 8) & 0xFEFFFFFF) : "memory");
                        }
                        partial_stage += 1;
                        if (partial_stage == 4) { partial_stage = 0; _phase_partial_full ^= 1; }
                        mbarrier_wait(partial_full_addr + (partial_stage) * 8, _phase_partial_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _trl_addr_15 = tmem_partials + (partial_stage * 128 + (unsigned int)(acc_wg * 64)) + (row_base << 16);
                        tmem_ld_x16_wait(&partial_fragment[0], _trl_addr_15);
                        {
                            unsigned long long _fma_acc_scale2_16;
                            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_16) : "f"(combined_u));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc2[0]), "+f"(acc2[1]) : "f"(partial_fragment[0]), "f"(partial_fragment[1]), "l"(_fma_acc_scale2_16));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc2[2]), "+f"(acc2[3]) : "f"(partial_fragment[2]), "f"(partial_fragment[3]), "l"(_fma_acc_scale2_16));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc2[4]), "+f"(acc2[5]) : "f"(partial_fragment[4]), "f"(partial_fragment[5]), "l"(_fma_acc_scale2_16));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc2[6]), "+f"(acc2[7]) : "f"(partial_fragment[6]), "f"(partial_fragment[7]), "l"(_fma_acc_scale2_16));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc2[8]), "+f"(acc2[9]) : "f"(partial_fragment[8]), "f"(partial_fragment[9]), "l"(_fma_acc_scale2_16));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc2[10]), "+f"(acc2[11]) : "f"(partial_fragment[10]), "f"(partial_fragment[11]), "l"(_fma_acc_scale2_16));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc2[12]), "+f"(acc2[13]) : "f"(partial_fragment[12]), "f"(partial_fragment[13]), "l"(_fma_acc_scale2_16));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc2[14]), "+f"(acc2[15]) : "f"(partial_fragment[14]), "f"(partial_fragment[15]), "l"(_fma_acc_scale2_16));
                        }
                        int _trl_addr_17 = tmem_partials + (partial_stage * 128 + (unsigned int)(acc_wg * 64) + 16) + (row_base << 16);
                        tmem_ld_x16_wait(&partial_fragment[0], _trl_addr_17);
                        {
                            unsigned long long _fma_acc_scale2_18;
                            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_18) : "f"(combined_u));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc2[16]), "+f"(acc2[17]) : "f"(partial_fragment[0]), "f"(partial_fragment[1]), "l"(_fma_acc_scale2_18));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc2[18]), "+f"(acc2[19]) : "f"(partial_fragment[2]), "f"(partial_fragment[3]), "l"(_fma_acc_scale2_18));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc2[20]), "+f"(acc2[21]) : "f"(partial_fragment[4]), "f"(partial_fragment[5]), "l"(_fma_acc_scale2_18));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc2[22]), "+f"(acc2[23]) : "f"(partial_fragment[6]), "f"(partial_fragment[7]), "l"(_fma_acc_scale2_18));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc2[24]), "+f"(acc2[25]) : "f"(partial_fragment[8]), "f"(partial_fragment[9]), "l"(_fma_acc_scale2_18));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc2[26]), "+f"(acc2[27]) : "f"(partial_fragment[10]), "f"(partial_fragment[11]), "l"(_fma_acc_scale2_18));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc2[28]), "+f"(acc2[29]) : "f"(partial_fragment[12]), "f"(partial_fragment[13]), "l"(_fma_acc_scale2_18));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc2[30]), "+f"(acc2[31]) : "f"(partial_fragment[14]), "f"(partial_fragment[15]), "l"(_fma_acc_scale2_18));
                        }
                        int _trl_addr_19 = tmem_partials + (partial_stage * 128 + (unsigned int)(acc_wg * 64) + 32) + (row_base << 16);
                        tmem_ld_x16_wait(&partial_fragment[0], _trl_addr_19);
                        {
                            unsigned long long _fma_acc_scale2_20;
                            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_20) : "f"(combined_u));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc3[0]), "+f"(acc3[1]) : "f"(partial_fragment[0]), "f"(partial_fragment[1]), "l"(_fma_acc_scale2_20));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc3[2]), "+f"(acc3[3]) : "f"(partial_fragment[2]), "f"(partial_fragment[3]), "l"(_fma_acc_scale2_20));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc3[4]), "+f"(acc3[5]) : "f"(partial_fragment[4]), "f"(partial_fragment[5]), "l"(_fma_acc_scale2_20));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc3[6]), "+f"(acc3[7]) : "f"(partial_fragment[6]), "f"(partial_fragment[7]), "l"(_fma_acc_scale2_20));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc3[8]), "+f"(acc3[9]) : "f"(partial_fragment[8]), "f"(partial_fragment[9]), "l"(_fma_acc_scale2_20));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc3[10]), "+f"(acc3[11]) : "f"(partial_fragment[10]), "f"(partial_fragment[11]), "l"(_fma_acc_scale2_20));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc3[12]), "+f"(acc3[13]) : "f"(partial_fragment[12]), "f"(partial_fragment[13]), "l"(_fma_acc_scale2_20));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc3[14]), "+f"(acc3[15]) : "f"(partial_fragment[14]), "f"(partial_fragment[15]), "l"(_fma_acc_scale2_20));
                        }
                        int _trl_addr_21 = tmem_partials + (partial_stage * 128 + (unsigned int)(acc_wg * 64) + 32 + 16) + (row_base << 16);
                        tmem_ld_x16_wait(&partial_fragment[0], _trl_addr_21);
                        {
                            unsigned long long _fma_acc_scale2_22;
                            asm volatile("mov.b64 %0, {%1, %1};" : "=l"(_fma_acc_scale2_22) : "f"(combined_u));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc3[16]), "+f"(acc3[17]) : "f"(partial_fragment[0]), "f"(partial_fragment[1]), "l"(_fma_acc_scale2_22));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc3[18]), "+f"(acc3[19]) : "f"(partial_fragment[2]), "f"(partial_fragment[3]), "l"(_fma_acc_scale2_22));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc3[20]), "+f"(acc3[21]) : "f"(partial_fragment[4]), "f"(partial_fragment[5]), "l"(_fma_acc_scale2_22));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc3[22]), "+f"(acc3[23]) : "f"(partial_fragment[6]), "f"(partial_fragment[7]), "l"(_fma_acc_scale2_22));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc3[24]), "+f"(acc3[25]) : "f"(partial_fragment[8]), "f"(partial_fragment[9]), "l"(_fma_acc_scale2_22));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc3[26]), "+f"(acc3[27]) : "f"(partial_fragment[10]), "f"(partial_fragment[11]), "l"(_fma_acc_scale2_22));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc3[28]), "+f"(acc3[29]) : "f"(partial_fragment[12]), "f"(partial_fragment[13]), "l"(_fma_acc_scale2_22));
                            asm volatile("{\n\t" ".reg .b64 _src2, _acc2, _out2;\n\t" "mov.b64 _src2, {%2, %3};\n\t" "mov.b64 _acc2, {%0, %1};\n\t" "fma.rn.f32x2 _out2, _src2, %4, _acc2;\n\t" "mov.b64 {%0, %1}, _out2;\n\t" "}" : "+f"(acc3[30]), "+f"(acc3[31]) : "f"(partial_fragment[14]), "f"(partial_fragment[15]), "l"(_fma_acc_scale2_22));
                        }
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        if (elect_sync()) {
                            asm volatile(
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                                :: "r"((partial_free_addr + (partial_stage) * 8) & 0xFEFFFFFF) : "memory");
                        }
                        partial_stage += 1;
                        if (partial_stage == 4) { partial_stage = 0; _phase_partial_full ^= 1; }
                    }
                }
                unsigned int output_ab_stage = (consumer_ab_cursor + (unsigned int)k_blocks - 1) % 7;
                consumer_ab_cursor = (consumer_ab_cursor + (unsigned int)k_blocks) % 7;
                int cluster_row = cta_rank * 128 + row;
                int owns = 0;
                if (cluster_row >= run_begin) {
                    if (cluster_row < run_end) {
                        owns = 1;
                    }
                }
                float hvals[64];
                #pragma unroll
                for (int j = 0; j < 32; j++) {
                    __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(acc0[j]);
                    float _cvt_f32_0 = __bfloat162float(_cvt_bf16_0);
                    float g0 = _cvt_f32_0;
                    __nv_bfloat16 _cvt_bf16_1 = __float2bfloat16(acc2[j]);
                    float _cvt_f32_1 = __bfloat162float(_cvt_bf16_1);
                    float u0 = _cvt_f32_1;
                    float _exp2_0 = approx_exp2((-g0) * 1.4426950408889634f);
                    float _rcp_0 = approx_rcp(1.0f + _exp2_0);
                    float s0 = _rcp_0;
                    __nv_bfloat16 _cvt_bf16_2 = __float2bfloat16(g0 * s0 * u0);
                    float _cvt_f32_2 = __bfloat162float(_cvt_bf16_2);
                    hvals[j] = _cvt_f32_2;
                    __nv_bfloat16 _cvt_bf16_3 = __float2bfloat16(acc1[j]);
                    float _cvt_f32_3 = __bfloat162float(_cvt_bf16_3);
                    float g1 = _cvt_f32_3;
                    __nv_bfloat16 _cvt_bf16_4 = __float2bfloat16(acc3[j]);
                    float _cvt_f32_4 = __bfloat162float(_cvt_bf16_4);
                    float u1 = _cvt_f32_4;
                    float _exp2_1 = approx_exp2((-g1) * 1.4426950408889634f);
                    float _rcp_1 = approx_rcp(1.0f + _exp2_1);
                    float s1 = _rcp_1;
                    __nv_bfloat16 _cvt_bf16_5 = __float2bfloat16(g1 * s1 * u1);
                    float _cvt_f32_5 = __bfloat162float(_cvt_bf16_5);
                    hvals[32 + j] = _cvt_f32_5;
                }
                float mags[64];
                #pragma unroll
                for (int j_1 = 0; j_1 < 64; j_1++) {
                    mags[j_1] = hvals[j_1];
                }
                float _fabs_0 = fabsf(mags[0]);
                mags[0] = _fabs_0;
                float _fabs_1 = fabsf(mags[1]);
                mags[1] = _fabs_1;
                float _fabs_2 = fabsf(mags[2]);
                mags[2] = _fabs_2;
                float _fabs_3 = fabsf(mags[3]);
                mags[3] = _fabs_3;
                float _fabs_4 = fabsf(mags[4]);
                mags[4] = _fabs_4;
                float _fabs_5 = fabsf(mags[5]);
                mags[5] = _fabs_5;
                float _fabs_6 = fabsf(mags[6]);
                mags[6] = _fabs_6;
                float _fabs_7 = fabsf(mags[7]);
                mags[7] = _fabs_7;
                float _fabs_8 = fabsf(mags[8]);
                mags[8] = _fabs_8;
                float _fabs_9 = fabsf(mags[9]);
                mags[9] = _fabs_9;
                float _fabs_10 = fabsf(mags[10]);
                mags[10] = _fabs_10;
                float _fabs_11 = fabsf(mags[11]);
                mags[11] = _fabs_11;
                float _fabs_12 = fabsf(mags[12]);
                mags[12] = _fabs_12;
                float _fabs_13 = fabsf(mags[13]);
                mags[13] = _fabs_13;
                float _fabs_14 = fabsf(mags[14]);
                mags[14] = _fabs_14;
                float _fabs_15 = fabsf(mags[15]);
                mags[15] = _fabs_15;
                float _fabs_16 = fabsf(mags[16]);
                mags[16] = _fabs_16;
                float _fabs_17 = fabsf(mags[17]);
                mags[17] = _fabs_17;
                float _fabs_18 = fabsf(mags[18]);
                mags[18] = _fabs_18;
                float _fabs_19 = fabsf(mags[19]);
                mags[19] = _fabs_19;
                float _fabs_20 = fabsf(mags[20]);
                mags[20] = _fabs_20;
                float _fabs_21 = fabsf(mags[21]);
                mags[21] = _fabs_21;
                float _fabs_22 = fabsf(mags[22]);
                mags[22] = _fabs_22;
                float _fabs_23 = fabsf(mags[23]);
                mags[23] = _fabs_23;
                float _fabs_24 = fabsf(mags[24]);
                mags[24] = _fabs_24;
                float _fabs_25 = fabsf(mags[25]);
                mags[25] = _fabs_25;
                float _fabs_26 = fabsf(mags[26]);
                mags[26] = _fabs_26;
                float _fabs_27 = fabsf(mags[27]);
                mags[27] = _fabs_27;
                float _fabs_28 = fabsf(mags[28]);
                mags[28] = _fabs_28;
                float _fabs_29 = fabsf(mags[29]);
                mags[29] = _fabs_29;
                float _fabs_30 = fabsf(mags[30]);
                mags[30] = _fabs_30;
                float _fabs_31 = fabsf(mags[31]);
                mags[31] = _fabs_31;
                float _fabs_32 = fabsf(mags[32]);
                mags[32] = _fabs_32;
                float _fabs_33 = fabsf(mags[33]);
                mags[33] = _fabs_33;
                float _fabs_34 = fabsf(mags[34]);
                mags[34] = _fabs_34;
                float _fabs_35 = fabsf(mags[35]);
                mags[35] = _fabs_35;
                float _fabs_36 = fabsf(mags[36]);
                mags[36] = _fabs_36;
                float _fabs_37 = fabsf(mags[37]);
                mags[37] = _fabs_37;
                float _fabs_38 = fabsf(mags[38]);
                mags[38] = _fabs_38;
                float _fabs_39 = fabsf(mags[39]);
                mags[39] = _fabs_39;
                float _fabs_40 = fabsf(mags[40]);
                mags[40] = _fabs_40;
                float _fabs_41 = fabsf(mags[41]);
                mags[41] = _fabs_41;
                float _fabs_42 = fabsf(mags[42]);
                mags[42] = _fabs_42;
                float _fabs_43 = fabsf(mags[43]);
                mags[43] = _fabs_43;
                float _fabs_44 = fabsf(mags[44]);
                mags[44] = _fabs_44;
                float _fabs_45 = fabsf(mags[45]);
                mags[45] = _fabs_45;
                float _fabs_46 = fabsf(mags[46]);
                mags[46] = _fabs_46;
                float _fabs_47 = fabsf(mags[47]);
                mags[47] = _fabs_47;
                float _fabs_48 = fabsf(mags[48]);
                mags[48] = _fabs_48;
                float _fabs_49 = fabsf(mags[49]);
                mags[49] = _fabs_49;
                float _fabs_50 = fabsf(mags[50]);
                mags[50] = _fabs_50;
                float _fabs_51 = fabsf(mags[51]);
                mags[51] = _fabs_51;
                float _fabs_52 = fabsf(mags[52]);
                mags[52] = _fabs_52;
                float _fabs_53 = fabsf(mags[53]);
                mags[53] = _fabs_53;
                float _fabs_54 = fabsf(mags[54]);
                mags[54] = _fabs_54;
                float _fabs_55 = fabsf(mags[55]);
                mags[55] = _fabs_55;
                float _fabs_56 = fabsf(mags[56]);
                mags[56] = _fabs_56;
                float _fabs_57 = fabsf(mags[57]);
                mags[57] = _fabs_57;
                float _fabs_58 = fabsf(mags[58]);
                mags[58] = _fabs_58;
                float _fabs_59 = fabsf(mags[59]);
                mags[59] = _fabs_59;
                float _fabs_60 = fabsf(mags[60]);
                mags[60] = _fabs_60;
                float _fabs_61 = fabsf(mags[61]);
                mags[61] = _fabs_61;
                float _fabs_62 = fabsf(mags[62]);
                mags[62] = _fabs_62;
                float _fabs_63 = fabsf(mags[63]);
                mags[63] = _fabs_63;
                float2 _reg_reduce_max2_23 = {-CAKE_INF, -CAKE_INF};
                row_max_x32_accum(&mags[0], _reg_reduce_max2_23);
                row_max_x32_accum(&mags[32], _reg_reduce_max2_23);
                float mags_max = row_max_reduce(_reg_reduce_max2_23);
                float absmax = mags_max;
                int xchg_base = output_ab_stage * 4096;
                rowmax_xchg[xchg_base + acc_wg * 128 + row] = absmax;
                asm volatile("barrier.sync 11, 256;" ::: "memory");
                float other = rowmax_xchg[xchg_base + (1 - acc_wg) * 128 + row];
                float _fmax_0 = fmaxf(absmax, other);
                float group_max = _fmax_0;
                float _fmax_1 = fmaxf(group_max, 1e-10f);
                group_max = _fmax_1;
                float scale = group_max / 448.0f;
                float inv_scale = 1.0f / scale;
                const float2 _scale2_24 = {inv_scale, inv_scale};
                #pragma unroll
                for (int _ls = 0; _ls < 32; _ls++)
                    mul_f32x2_inplace(&reinterpret_cast<float2*>(hvals)[_ls], _scale2_24);
                #pragma unroll
                for (int j_2 = 0; j_2 < 64; j_2++) {
                    float _min_0 = fminf(hvals[j_2], 448.0f);
                    float _fmax_2 = fmaxf(_min_0, -448.0f);
                    hvals[j_2] = _fmax_2;
                }
                if (owns != 0) {
                    long long out_row = cluster_m_base;
                    out_row += cluster_row;
                    long long q_base = out_row * (long long)(N / 2) + (long long)(n_tile * 128) + (long long)(acc_wg * 64);
                    #pragma unroll
                    for (int vec = 0; vec < 64; vec += 16) {
                        {
                            unsigned int _fp8_pk[4];
                            asm("{\n\t"
                                ".reg .b16 _lo, _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}\n"
                                : "=r"(_fp8_pk[0]) : "f"(hvals[vec + 0]), "f"(hvals[vec + 1]), "f"(hvals[vec + 2]), "f"(hvals[vec + 3]));
                            asm("{\n\t"
                                ".reg .b16 _lo, _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}\n"
                                : "=r"(_fp8_pk[1]) : "f"(hvals[vec + 4]), "f"(hvals[vec + 5]), "f"(hvals[vec + 6]), "f"(hvals[vec + 7]));
                            asm("{\n\t"
                                ".reg .b16 _lo, _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}\n"
                                : "=r"(_fp8_pk[2]) : "f"(hvals[vec + 8]), "f"(hvals[vec + 9]), "f"(hvals[vec + 10]), "f"(hvals[vec + 11]));
                            asm("{\n\t"
                                ".reg .b16 _lo, _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}\n"
                                : "=r"(_fp8_pk[3]) : "f"(hvals[vec + 12]), "f"(hvals[vec + 13]), "f"(hvals[vec + 14]), "f"(hvals[vec + 15]));
                            *reinterpret_cast<uint4*>(reinterpret_cast<unsigned char*>(out_q + (q_base + (long long)vec)) + (0)) = *reinterpret_cast<uint4*>(_fp8_pk);
                        }
                    }
                    if (acc_wg == 0) {
                        *(reinterpret_cast<float*>(out_s + (out_row * (long long)(N / 2 / 128) + (long long)n_tile)) + (0)) = scale;
                    }
                }
                asm volatile("barrier.sync 11, 256;" ::: "memory");
                if (warp == 0) {
                    if (elect_sync()) {
                        mbarrier_arrive(ab_free_addr + (output_ab_stage) * 8);
                    }
                }
            }
            asm volatile("barrier.sync 9, 288;" ::: "memory");
            asm volatile("tcgen05.fence::after_thread_sync;");
            unsigned int _phase_producers_done_0 = 0;
            unsigned int _phase_pair_exit_0 = 0;
            if (warp == 0) {
                mbarrier_wait(producers_done_addr, _phase_producers_done_0);
                _phase_producers_done_0 ^= 1;
                if (elect_sync()) {
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(pair_exit_addr), "r"(1 - cta_rank) : "memory");
                }
                mbarrier_wait(pair_exit_addr, _phase_pair_exit_0);
                _phase_pair_exit_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
                asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(512));
            }
        }
    }
    // ---- Role: mma_role ----
    if (warp == 8) {
        { // mma_role_main
            asm volatile("barrier.sync 9, 288;" ::: "memory");
            asm volatile("tcgen05.fence::after_thread_sync;");
            taddr = tmem_addr_storage[0];
            tmem_partials = taddr;
            unsigned int ab_stage = 0;
            unsigned int partial_stage_1 = 0;
            int m_tiles_1 = (M - 1) / 256 + 1;
            int n_tiles_1 = N / 256;
            int total_tiles_1 = m_tiles_1 * n_tiles_1;
            unsigned int _phase_partial_free = 1;
            unsigned int _phase_ab_full = 0;
            if (cta_rank == 0) {
                int nblk_1 = (M - 1) / 128 + 1;
                int g_blk_lo_1 = -1;
                int g_blk_hi_1 = -1;
                if (nblk_1 > lane) {
                    g_blk_lo_1 = m_indices[lane * 128];
                }
                if (nblk_1 > lane + 32) {
                    g_blk_hi_1 = m_indices[(lane + 32) * 128];
                }
                int pair_lo_1 = 0;
                int pair_hi_1 = 0;
                int pairs_total_1 = 0;
                int cur_e_1 = -1;
                int local_i_1 = 0;
                #pragma unroll 1
                for (int kb_1 = 0; kb_1 < nblk_1; kb_1++) {
                    int _shfl_24 = __shfl_sync(0xFFFFFFFF, g_blk_lo_1, kb_1 & 31);
                    int gk_1 = _shfl_24;
                    int _shfl_25 = __shfl_sync(0xFFFFFFFF, g_blk_hi_1, kb_1 & 31);
                    int gk_hi_1 = _shfl_25;
                    if (kb_1 >= 32) {
                        gk_1 = gk_hi_1;
                    }
                    if (gk_1 != cur_e_1) {
                        cur_e_1 = gk_1;
                        local_i_1 = -1;
                    }
                    local_i_1 += 1;
                    if ((local_i_1 & 1) == 0) {
                        if (lane == (pairs_total_1 & 31)) {
                            if (pairs_total_1 < 32) {
                                pair_lo_1 = kb_1;
                            }
                            if (pairs_total_1 >= 32) {
                                pair_hi_1 = kb_1;
                            }
                        }
                        pairs_total_1 += 1;
                    }
                }
                total_tiles_1 = pairs_total_1 * n_tiles_1;
                #pragma unroll 1
                for (int tile_id_1 = cluster_id; tile_id_1 < total_tiles_1; tile_id_1 += num_clusters) {
                    int pair_id_1 = tile_id_1 / n_tiles_1;
                    int n_tile_1 = tile_id_1 % n_tiles_1;
                    int _shfl_26 = __shfl_sync(0xFFFFFFFF, pair_lo_1, pair_id_1 & 31);
                    int b0_1 = _shfl_26;
                    int _shfl_27 = __shfl_sync(0xFFFFFFFF, pair_hi_1, pair_id_1 & 31);
                    int b0_hi_1 = _shfl_27;
                    if (pair_id_1 >= 32) {
                        b0_1 = b0_hi_1;
                    }
                    int _shfl_28 = __shfl_sync(0xFFFFFFFF, g_blk_lo_1, b0_1 & 31);
                    int group_1 = _shfl_28;
                    int _shfl_29 = __shfl_sync(0xFFFFFFFF, g_blk_hi_1, b0_1 & 31);
                    int group_hi_1 = _shfl_29;
                    if (b0_1 >= 32) {
                        group_1 = group_hi_1;
                    }
                    int bn_1 = b0_1 + 1;
                    int _shfl_30 = __shfl_sync(0xFFFFFFFF, g_blk_lo_1, bn_1 & 31);
                    int gn_1 = _shfl_30;
                    int _shfl_31 = __shfl_sync(0xFFFFFFFF, g_blk_hi_1, bn_1 & 31);
                    int gn_hi_1 = _shfl_31;
                    if (bn_1 >= 32) {
                        gn_1 = gn_hi_1;
                    }
                    int b1_1 = -1;
                    if (bn_1 < nblk_1) {
                        if (gn_1 == group_1) {
                            b1_1 = bn_1;
                        }
                    }
                    int my_blk_1 = b0_1;
                    if (cta_rank == 1) {
                        my_blk_1 = b1_1;
                    }
                    int m_tile_1 = b0_1;
                    int my_rows_1 = 0;
                    if (my_blk_1 >= 0) {
                        m_tile_1 = my_blk_1;
                        my_rows_1 = M - my_blk_1 * 128;
                        if (my_rows_1 > 128) {
                            my_rows_1 = 128;
                        }
                    }
                    int cluster_m_base_1 = m_tile_1 * 128 - cta_rank * 128;
                    int tile_rows_1 = 256;
                    int run_begin_1 = cta_rank * 128;
                    int run_end_1 = run_begin_1 + my_rows_1;
                    int k_blocks_1 = K / 128;
                    #pragma unroll 1
                    for (int iter_k = 0; iter_k < k_blocks_1; iter_k++) {
                        mbarrier_wait(partial_free_addr + (partial_stage_1) * 8, _phase_partial_free);
                        mbarrier_wait(ab_full_addr + (ab_stage) * 8, _phase_ab_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_0 = (((smem_a_addr) >> 4) & 0x3FFF) + (ab_stage) * 1024;
                        int _mma_b_lo_0 = (((smem_b_half0_addr) >> 4) & 0x3FFF) + (ab_stage) * 1024;
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
                    "mov.b32 id, 270532624;\n\t"
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
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_partials + (partial_stage_1 * 128))), "r"(0));
                        elect_commit_cg2_multicast(partial_full_addr + (partial_stage_1) * 8, (uint16_t)(3));
                        partial_stage_1 += 1;
                        if (partial_stage_1 == 4) { partial_stage_1 = 0; _phase_partial_free ^= 1; }
                        mbarrier_wait(partial_free_addr + (partial_stage_1) * 8, _phase_partial_free);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_1 = (((smem_a_addr) >> 4) & 0x3FFF) + (ab_stage) * 1024;
                        int _mma_b_lo_1 = (((smem_b_half1_addr) >> 4) & 0x3FFF) + (ab_stage) * 1024;
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
                    "mov.b32 id, 270532624;\n\t"
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
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"((tmem_partials + (partial_stage_1 * 128))), "r"(0));
                        int release_ab = 1;
                        if (iter_k == k_blocks_1 - 1) {
                            release_ab = 0;
                        }
                        if (release_ab != 0) {
                            elect_commit_cg2_multicast(ab_free_addr + (ab_stage) * 8, (uint16_t)(3));
                        }
                        elect_commit_cg2_multicast(partial_full_addr + (partial_stage_1) * 8, (uint16_t)(3));
                        ab_stage += 1;
                        if (ab_stage == 7) { ab_stage = 0; _phase_ab_full ^= 1; }
                        partial_stage_1 += 1;
                        if (partial_stage_1 == 4) { partial_stage_1 = 0; _phase_partial_free ^= 1; }
                    }
                }
            }
            asm volatile("barrier.sync 9, 288;" ::: "memory");
        }
    }
    // ---- Role: tma_role ----
    if (warp == 9) {
        { // tma_role_main
            unsigned int ab_stage_1 = 0;
            int m_tiles_2 = (M - 1) / 256 + 1;
            int n_tiles_2 = N / 256;
            int total_tiles_2 = m_tiles_2 * n_tiles_2;
            int nblk_2 = (M - 1) / 128 + 1;
            int g_blk_lo_2 = -1;
            int g_blk_hi_2 = -1;
            if (nblk_2 > lane) {
                g_blk_lo_2 = m_indices[lane * 128];
            }
            if (nblk_2 > lane + 32) {
                g_blk_hi_2 = m_indices[(lane + 32) * 128];
            }
            int pair_lo_2 = 0;
            int pair_hi_2 = 0;
            int pairs_total_2 = 0;
            int cur_e_2 = -1;
            int local_i_2 = 0;
            #pragma unroll 1
            for (int kb_2 = 0; kb_2 < nblk_2; kb_2++) {
                int _shfl_0 = __shfl_sync(0xFFFFFFFF, g_blk_lo_2, kb_2 & 31);
                int gk_2 = _shfl_0;
                int _shfl_1 = __shfl_sync(0xFFFFFFFF, g_blk_hi_2, kb_2 & 31);
                int gk_hi_2 = _shfl_1;
                if (kb_2 >= 32) {
                    gk_2 = gk_hi_2;
                }
                if (gk_2 != cur_e_2) {
                    cur_e_2 = gk_2;
                    local_i_2 = -1;
                }
                local_i_2 += 1;
                if ((local_i_2 & 1) == 0) {
                    if (lane == (pairs_total_2 & 31)) {
                        if (pairs_total_2 < 32) {
                            pair_lo_2 = kb_2;
                        }
                        if (pairs_total_2 >= 32) {
                            pair_hi_2 = kb_2;
                        }
                    }
                    pairs_total_2 += 1;
                }
            }
            total_tiles_2 = pairs_total_2 * n_tiles_2;
            unsigned int _phase_ab_free = 1;
            #pragma unroll 1
            for (int tile_id_2 = cluster_id; tile_id_2 < total_tiles_2; tile_id_2 += num_clusters) {
                int pair_id_2 = tile_id_2 / n_tiles_2;
                int n_tile_2 = tile_id_2 % n_tiles_2;
                int _shfl_2 = __shfl_sync(0xFFFFFFFF, pair_lo_2, pair_id_2 & 31);
                int b0_2 = _shfl_2;
                int _shfl_3 = __shfl_sync(0xFFFFFFFF, pair_hi_2, pair_id_2 & 31);
                int b0_hi_2 = _shfl_3;
                if (pair_id_2 >= 32) {
                    b0_2 = b0_hi_2;
                }
                int _shfl_4 = __shfl_sync(0xFFFFFFFF, g_blk_lo_2, b0_2 & 31);
                int group_2 = _shfl_4;
                int _shfl_5 = __shfl_sync(0xFFFFFFFF, g_blk_hi_2, b0_2 & 31);
                int group_hi_2 = _shfl_5;
                if (b0_2 >= 32) {
                    group_2 = group_hi_2;
                }
                int bn_2 = b0_2 + 1;
                int _shfl_6 = __shfl_sync(0xFFFFFFFF, g_blk_lo_2, bn_2 & 31);
                int gn_2 = _shfl_6;
                int _shfl_7 = __shfl_sync(0xFFFFFFFF, g_blk_hi_2, bn_2 & 31);
                int gn_hi_2 = _shfl_7;
                if (bn_2 >= 32) {
                    gn_2 = gn_hi_2;
                }
                int b1_2 = -1;
                if (bn_2 < nblk_2) {
                    if (gn_2 == group_2) {
                        b1_2 = bn_2;
                    }
                }
                int my_blk_2 = b0_2;
                if (cta_rank == 1) {
                    my_blk_2 = b1_2;
                }
                int m_tile_2 = b0_2;
                int my_rows_2 = 0;
                if (my_blk_2 >= 0) {
                    m_tile_2 = my_blk_2;
                    my_rows_2 = M - my_blk_2 * 128;
                    if (my_rows_2 > 128) {
                        my_rows_2 = 128;
                    }
                }
                int cluster_m_base_2 = m_tile_2 * 128 - cta_rank * 128;
                int tile_rows_2 = 256;
                int run_begin_2 = cta_rank * 128;
                int run_end_2 = run_begin_2 + my_rows_2;
                int k_blocks_2 = K / 128;
                if (elect_sync()) {
                    #pragma unroll 1
                    for (int iter_k_1 = 0; iter_k_1 < k_blocks_2; iter_k_1++) {
                        mbarrier_wait(ab_free_addr + (ab_stage_1) * 8, _phase_ab_free);
                        tma_3d_gmem2smem_cta2(smem_a_addr + ab_stage_1 * 16384, A, 0, m_tile_2 * 128, iter_k_1, ((ab_full_addr + (ab_stage_1) * 8) & 0xFEFFFFFF));
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((ab_full_addr + (ab_stage_1) * 8) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                        ab_stage_1 += 1;
                        if (ab_stage_1 == 7) { ab_stage_1 = 0; _phase_ab_free ^= 1; }
                    }
                }
            }
            if (elect_sync()) {
                #pragma unroll
                for (int _ab_drain = 0; _ab_drain < 7; _ab_drain++) {
                    mbarrier_wait(ab_free_addr + (ab_stage_1) * 8, _phase_ab_free);
                    ab_stage_1 += 1;
                    if (ab_stage_1 == 7) { ab_stage_1 = 0; _phase_ab_free ^= 1; }
                }
                mbarrier_arrive(producers_done_addr);
            }
        }
    }
    // ---- Role: scale_role ----
    if (warp == 10) {
        { // scale_role_main
            unsigned int scale_stage_1 = 0;
            int m_tiles_3 = (M - 1) / 256 + 1;
            int n_tiles_3 = N / 256;
            int total_tiles_3 = m_tiles_3 * n_tiles_3;
            int nblk_3 = (M - 1) / 128 + 1;
            int g_blk_lo_3 = -1;
            int g_blk_hi_3 = -1;
            if (nblk_3 > lane) {
                g_blk_lo_3 = m_indices[lane * 128];
            }
            if (nblk_3 > lane + 32) {
                g_blk_hi_3 = m_indices[(lane + 32) * 128];
            }
            int pair_lo_3 = 0;
            int pair_hi_3 = 0;
            int pairs_total_3 = 0;
            int cur_e_3 = -1;
            int local_i_3 = 0;
            #pragma unroll 1
            for (int kb_3 = 0; kb_3 < nblk_3; kb_3++) {
                int _shfl_16 = __shfl_sync(0xFFFFFFFF, g_blk_lo_3, kb_3 & 31);
                int gk_3 = _shfl_16;
                int _shfl_17 = __shfl_sync(0xFFFFFFFF, g_blk_hi_3, kb_3 & 31);
                int gk_hi_3 = _shfl_17;
                if (kb_3 >= 32) {
                    gk_3 = gk_hi_3;
                }
                if (gk_3 != cur_e_3) {
                    cur_e_3 = gk_3;
                    local_i_3 = -1;
                }
                local_i_3 += 1;
                if ((local_i_3 & 1) == 0) {
                    if (lane == (pairs_total_3 & 31)) {
                        if (pairs_total_3 < 32) {
                            pair_lo_3 = kb_3;
                        }
                        if (pairs_total_3 >= 32) {
                            pair_hi_3 = kb_3;
                        }
                    }
                    pairs_total_3 += 1;
                }
            }
            total_tiles_3 = pairs_total_3 * n_tiles_3;
            unsigned int _phase_scale_free = 1;
            #pragma unroll 1
            for (int tile_id_3 = cluster_id; tile_id_3 < total_tiles_3; tile_id_3 += num_clusters) {
                int pair_id_3 = tile_id_3 / n_tiles_3;
                int n_tile_3 = tile_id_3 % n_tiles_3;
                int _shfl_18 = __shfl_sync(0xFFFFFFFF, pair_lo_3, pair_id_3 & 31);
                int b0_3 = _shfl_18;
                int _shfl_19 = __shfl_sync(0xFFFFFFFF, pair_hi_3, pair_id_3 & 31);
                int b0_hi_3 = _shfl_19;
                if (pair_id_3 >= 32) {
                    b0_3 = b0_hi_3;
                }
                int _shfl_20 = __shfl_sync(0xFFFFFFFF, g_blk_lo_3, b0_3 & 31);
                int group_3 = _shfl_20;
                int _shfl_21 = __shfl_sync(0xFFFFFFFF, g_blk_hi_3, b0_3 & 31);
                int group_hi_3 = _shfl_21;
                if (b0_3 >= 32) {
                    group_3 = group_hi_3;
                }
                int bn_3 = b0_3 + 1;
                int _shfl_22 = __shfl_sync(0xFFFFFFFF, g_blk_lo_3, bn_3 & 31);
                int gn_3 = _shfl_22;
                int _shfl_23 = __shfl_sync(0xFFFFFFFF, g_blk_hi_3, bn_3 & 31);
                int gn_hi_3 = _shfl_23;
                if (bn_3 >= 32) {
                    gn_3 = gn_hi_3;
                }
                int b1_3 = -1;
                if (bn_3 < nblk_3) {
                    if (gn_3 == group_3) {
                        b1_3 = bn_3;
                    }
                }
                int my_blk_3 = b0_3;
                if (cta_rank == 1) {
                    my_blk_3 = b1_3;
                }
                int m_tile_3 = b0_3;
                int my_rows_3 = 0;
                if (my_blk_3 >= 0) {
                    m_tile_3 = my_blk_3;
                    my_rows_3 = M - my_blk_3 * 128;
                    if (my_rows_3 > 128) {
                        my_rows_3 = 128;
                    }
                }
                int cluster_m_base_3 = m_tile_3 * 128 - cta_rank * 128;
                int tile_rows_3 = 256;
                int run_begin_3 = cta_rank * 128;
                int run_end_3 = run_begin_3 + my_rows_3;
                int k_blocks_3 = K / 128;
                int scale_groups_1 = k_blocks_3 / 4;
                #pragma unroll 1
                for (int scale_group_1 = 0; scale_group_1 < scale_groups_1; scale_group_1++) {
                    mbarrier_wait(scale_free_addr + (scale_stage_1) * 8, _phase_scale_free);
                    int stage_base = scale_stage_1 * 128 * 4;
                    #pragma unroll
                    for (int chunk = 0; chunk < 4; chunk++) {
                        int row_1 = chunk * 32 + lane;
                        int g_row = m_tile_3 * 128 + row_1;
                        asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"(smem_ascale_addr + (unsigned int)((stage_base + row_1 * 4) * 4)), "l"(a_scale + (g_row * k_blocks_3 + scale_group_1 * 4)), "r"((g_row < M) ? 16 : 0));
                    }
                    asm volatile(
                        "{\n\t"
                        "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n\t"
                        "}"
                        :: "r"(scale_full_addr + (scale_stage_1) * 8) : "memory");
                    _phase_scale_free ^= 1;
                }
            }
            #pragma unroll
            for (int _drain = 0; _drain < 1; _drain++) {
                mbarrier_wait(scale_free_addr + (scale_stage_1) * 8, _phase_scale_free);
                _phase_scale_free ^= 1;
            }
            if (elect_sync()) {
                mbarrier_arrive(producers_done_addr);
            }
        }
    }
    // ---- Role: tma_b_role ----
    if (warp == 11) {
        { // tma_b_role_main
            unsigned int ab_stage_b = 0;
            int m_tiles_b = (M - 1) / 256 + 1;
            int n_tiles_b = N / 256;
            int total_tiles_b = m_tiles_b * n_tiles_b;
            int nblk_4 = (M - 1) / 128 + 1;
            int g_blk_lo_4 = -1;
            int g_blk_hi_4 = -1;
            if (nblk_4 > lane) {
                g_blk_lo_4 = m_indices[lane * 128];
            }
            if (nblk_4 > lane + 32) {
                g_blk_hi_4 = m_indices[(lane + 32) * 128];
            }
            int pair_lo_4 = 0;
            int pair_hi_4 = 0;
            int pairs_total_4 = 0;
            int cur_e_4 = -1;
            int local_i_4 = 0;
            #pragma unroll 1
            for (int kb_4 = 0; kb_4 < nblk_4; kb_4++) {
                int _shfl_8 = __shfl_sync(0xFFFFFFFF, g_blk_lo_4, kb_4 & 31);
                int gk_4 = _shfl_8;
                int _shfl_9 = __shfl_sync(0xFFFFFFFF, g_blk_hi_4, kb_4 & 31);
                int gk_hi_4 = _shfl_9;
                if (kb_4 >= 32) {
                    gk_4 = gk_hi_4;
                }
                if (gk_4 != cur_e_4) {
                    cur_e_4 = gk_4;
                    local_i_4 = -1;
                }
                local_i_4 += 1;
                if ((local_i_4 & 1) == 0) {
                    if (lane == (pairs_total_4 & 31)) {
                        if (pairs_total_4 < 32) {
                            pair_lo_4 = kb_4;
                        }
                        if (pairs_total_4 >= 32) {
                            pair_hi_4 = kb_4;
                        }
                    }
                    pairs_total_4 += 1;
                }
            }
            total_tiles_b = pairs_total_4 * n_tiles_b;
            unsigned int _phase_ab_free_1 = 1;
            #pragma unroll 1
            for (int tile_id_b = cluster_id; tile_id_b < total_tiles_b; tile_id_b += num_clusters) {
                int pair_id_4 = tile_id_b / n_tiles_b;
                int n_tile_b = tile_id_b % n_tiles_b;
                int _shfl_10 = __shfl_sync(0xFFFFFFFF, pair_lo_4, pair_id_4 & 31);
                int b0_4 = _shfl_10;
                int _shfl_11 = __shfl_sync(0xFFFFFFFF, pair_hi_4, pair_id_4 & 31);
                int b0_hi_4 = _shfl_11;
                if (pair_id_4 >= 32) {
                    b0_4 = b0_hi_4;
                }
                int _shfl_12 = __shfl_sync(0xFFFFFFFF, g_blk_lo_4, b0_4 & 31);
                int group_b = _shfl_12;
                int _shfl_13 = __shfl_sync(0xFFFFFFFF, g_blk_hi_4, b0_4 & 31);
                int group_hi_4 = _shfl_13;
                if (b0_4 >= 32) {
                    group_b = group_hi_4;
                }
                int bn_4 = b0_4 + 1;
                int _shfl_14 = __shfl_sync(0xFFFFFFFF, g_blk_lo_4, bn_4 & 31);
                int gn_4 = _shfl_14;
                int _shfl_15 = __shfl_sync(0xFFFFFFFF, g_blk_hi_4, bn_4 & 31);
                int gn_hi_4 = _shfl_15;
                if (bn_4 >= 32) {
                    gn_4 = gn_hi_4;
                }
                int b1_4 = -1;
                if (bn_4 < nblk_4) {
                    if (gn_4 == group_b) {
                        b1_4 = bn_4;
                    }
                }
                int my_blk_4 = b0_4;
                if (cta_rank == 1) {
                    my_blk_4 = b1_4;
                }
                int m_tile_b = b0_4;
                int my_rows_4 = 0;
                if (my_blk_4 >= 0) {
                    m_tile_b = my_blk_4;
                    my_rows_4 = M - my_blk_4 * 128;
                    if (my_rows_4 > 128) {
                        my_rows_4 = 128;
                    }
                }
                int cluster_m_base_4 = m_tile_b * 128 - cta_rank * 128;
                int tile_rows_4 = 256;
                int run_begin_4 = cta_rank * 128;
                int run_end_4 = run_begin_4 + my_rows_4;
                int k_blocks_b = K / 128;
                if (elect_sync()) {
                    #pragma unroll 1
                    for (int iter_k_b = 0; iter_k_b < k_blocks_b; iter_k_b++) {
                        mbarrier_wait(ab_free_addr + (ab_stage_b) * 8, _phase_ab_free_1);
                        tma_4d_gmem2smem_cta2(smem_b_addr + ab_stage_b * 16384, B, 0, n_tile_b * 128 + cta_rank * 64, iter_k_b, group_b, ((ab_full_addr + (ab_stage_b) * 8) & 0xFEFFFFFF));
                        tma_4d_gmem2smem_cta2(smem_b_addr + ab_stage_b * 16384 + 8192, B, 0, N / 2 + n_tile_b * 128 + cta_rank * 64, iter_k_b, group_b, ((ab_full_addr + (ab_stage_b) * 8) & 0xFEFFFFFF));
                        ab_stage_b += 1;
                        if (ab_stage_b == 7) { ab_stage_b = 0; _phase_ab_free_1 ^= 1; }
                    }
                }
            }
            if (elect_sync()) {
                #pragma unroll
                for (int _ab_drain_b = 0; _ab_drain_b < 7; _ab_drain_b++) {
                    mbarrier_wait(ab_free_addr + (ab_stage_b) * 8, _phase_ab_free_1);
                    ab_stage_b += 1;
                    if (ab_stage_b == 7) { ab_stage_b = 0; _phase_ab_free_1 ^= 1; }
                }
                mbarrier_arrive(producers_done_addr);
            }
        }
    }

    // Cleanup
}

} // extern "C"
