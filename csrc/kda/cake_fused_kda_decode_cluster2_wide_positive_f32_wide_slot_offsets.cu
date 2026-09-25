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
static_assert(sizeof(uint64_t) == 8, "FlashInfer requires an LP64 CUDA host ABI");
typedef signed int         int32_t;
typedef short int          int16_t;
struct __align__(128) FlashInferTensorMap { uint64_t opaque[16]; };
struct __align__(64) FlashInferTensorMap64 { uint64_t opaque[16]; };
static_assert(sizeof(FlashInferTensorMap64) == 128, "64-aligned tensor-map ABI size");
static_assert(alignof(FlashInferTensorMap64) == 64, "64-aligned tensor-map ABI alignment");
template <int N>
struct __align__(128) FlashInferTensorMapPack { FlashInferTensorMap maps[N]; };

#if defined(__CUDACC_RTC__)
typedef struct __align__(128) { uint64_t opaque[16]; } CUtensorMap;
#else
#include <cuda.h>
#endif

static_assert(sizeof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128 bytes");
static_assert(alignof(FlashInferTensorMap) >= alignof(CUtensorMap), "FlashInferTensorMap alignment must cover the CUtensorMap CUDA ABI");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define FLASHINFER_INF CUDART_INF_F
#define NUM_MAIN_STAGES 1
#define SMEM_SMIXED_OFF 1024
#define SMEM_SMIXED_STAGE_BYTES 2048
#define SMEM_SMIXED_STRIDE 2048
#define SMEM_SRECURRENCE_OFF 3072
#define SMEM_SRECURRENCE_STAGE_BYTES 256
#define SMEM_SRECURRENCE_STRIDE 256
#define SMEM_SOUTPUTSCALE_OFF 3328
#define SMEM_SOUTPUTSCALE_STAGE_BYTES 512
#define SMEM_SOUTPUTSCALE_STRIDE 512
#define SMEM_SGATEDECAY_OFF 3840
#define SMEM_SGATEDECAY_STAGE_BYTES 768
#define SMEM_SGATEDECAY_STRIDE 768
#define SMEM_SBETA_OFF 4608
#define SMEM_SBETA_STAGE_BYTES 12
#define SMEM_SBETA_STRIDE 12
#define SMEM_SFIRSTSLABRMSPARTIAL_OFF 1024
#define SMEM_SFIRSTSLABRMSPARTIAL_STAGE_BYTES 4
#define SMEM_SFIRSTSLABRMSPARTIAL_STRIDE 4
#define SMEM_SASYNCSTATE_OFF 1024
#define SMEM_SASYNCSTATE_STAGE_BYTES 4
#define SMEM_SASYNCSTATE_STRIDE 4
#define SMEM_SCLUSTERRMS_OFF 4624
#define SMEM_SCLUSTERRMS_STAGE_BYTES 16
#define SMEM_SCLUSTERRMS_STRIDE 16
#define SMEM_SCONVNEW_OFF 4640
#define SMEM_SCONVNEW_STAGE_BYTES 4608
#define SMEM_SCONVNEW_STRIDE 4608
#define SMEM_TOTAL 9344
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

__global__ __launch_bounds__(512, 2) __cluster_dims__(2,1,1) void
kernel_cake_fused_kda_decode_cluster2_wide_positive_f32_wide_slot_offsets(__nv_bfloat16* __restrict__ x, float* __restrict__ weight, __nv_bfloat16* __restrict__ conv_state, __nv_bfloat16* __restrict__ raw_gate, __nv_bfloat16* __restrict__ raw_beta, float* __restrict__ A_log, float* __restrict__ dt_bias, int* __restrict__ state_indices, float* __restrict__ state, __nv_bfloat16* __restrict__ output_gate, float* __restrict__ norm_weight, __nv_bfloat16* __restrict__ output, int x_row_stride, int conv_slot_stride, int beta_row_stride, int state_slot_stride, int output_gate_row_stride, int H, int use_lower_bound, float lower_bound_log2, float norm_eps)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    asm volatile("{ .reg .u64 smem_ptr; cvta.to.shared.u64 smem_ptr, %1; cvt.u32.u64 %0, smem_ptr; }" : "=r"(smem) : "l"(smem_raw));
    smem = make_warp_uniform(smem);

    const int mbar_base = smem;
    #define cluster_rms_partial_addr (mbar_base + 0)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    float* sMixed = reinterpret_cast<float*>(smem_raw + 1024);
    const int sMixed_addr = smem + 1024;
    __nv_bfloat16* sRecurrence = reinterpret_cast<__nv_bfloat16*>(smem_raw + 3072);
    const int sRecurrence_addr = smem + 3072;
    float* sOutputScale = reinterpret_cast<float*>(smem_raw + 3328);
    const int sOutputScale_addr = smem + 3328;
    float* sGateDecay = reinterpret_cast<float*>(smem_raw + 3840);
    const int sGateDecay_addr = smem + 3840;
    float* sBeta = reinterpret_cast<float*>(smem_raw + 4608);
    const int sBeta_addr = smem + 4608;
    float* sFirstSlabRmsPartial = reinterpret_cast<float*>(smem_raw + 1024);
    const int sFirstSlabRmsPartial_addr = smem + 1024;
    unsigned int* sAsyncState = reinterpret_cast<unsigned int*>(smem_raw + 1024);
    const int sAsyncState_addr = smem + 1024;
    float* sClusterRms = reinterpret_cast<float*>(smem_raw + 4624);
    const int sClusterRms_addr = smem + 4624;
    float* sConvNew = reinterpret_cast<float*>(smem_raw + 4640);
    const int sConvNew_addr = smem + 4640;

    // Mbarrier init (1 pipeline groups, 0 ordered-sequence groups, 1 barriers)
    // Mbarriers at smem_raw[0..8)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // cluster_rms_partial: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    // === Task calls (dependency order) ===
    int row = blockIdx.y;
    int head = blockIdx.x / 2;
    int tid_0 = tid;
    int lane_1 = lane;
    int group = tid_0 / 16;
    int lane_group = tid_0 - group * 16;
    int k_start = lane_group * 8;
    int qk_smem_start = lane_group * 12;
    int state_owner_row_base = group * 2;
    int cluster_rank = cta_rank;
    int cluster_tile_base = cluster_rank * 64;
    int hidden = H * 128;
    int qkv_size = 3 * hidden;
    int requested_slot = state_indices[row];
    int slot = requested_slot;
    int is_live = 1;
    float state_regs[32];
    unsigned int state_carriers[4];
    float r_q[8];
    float r_k[8];
    float r_decay[8];
    if (tid_0 < 192) {
        int qkv_idx = tid_0 / 64;
        int channel_pair = tid_0 - qkv_idx * 64;
        int channel = channel_pair * 2;
        int channel_base = qkv_idx * hidden + head * 128 + channel;
        long long conv_base = (long long)slot * (long long)conv_slot_stride + (long long)channel_base;
        {
            uint32_t _bf16x2_bits_0;
            _bf16x2_bits_0 = *reinterpret_cast<const uint32_t*>(conv_state + conv_base);
            state_carriers[0] = _bf16x2_bits_0;
        }
        {
            uint32_t _bf16x2_bits_1;
            _bf16x2_bits_1 = *reinterpret_cast<const uint32_t*>(conv_state + conv_base + (long long)qkv_size);
            state_carriers[1] = _bf16x2_bits_1;
        }
        {
            uint32_t _bf16x2_bits_2;
            _bf16x2_bits_2 = *reinterpret_cast<const uint32_t*>(conv_state + conv_base + (long long)(2 * qkv_size));
            state_carriers[2] = _bf16x2_bits_2;
        }
        int x_base = row * x_row_stride + channel_base;
        r_q[6] = (float)x[x_base];
        r_q[7] = (float)x[x_base + 1];
        #pragma unroll
        for (int _pair = 0; _pair < 3; _pair++) {
            asm volatile(
                "{\n\t"
                "shl.b32 %0, %2, 16;\n\t"
                "and.b32 %1, %2, 0xffff0000;\n\t"
                "}\n"
                : "=f"((&r_q[_pair * 2])[0]), "=f"((&r_q[_pair * 2])[1])
                : "r"(state_carriers[_pair]));
        }
        float weight0_c0 = weight[qkv_idx * 4 * hidden + head * 128 + channel];
        float weight0_c1 = weight[qkv_idx * 4 * hidden + head * 128 + channel + 1];
        float weight1_c0 = weight[(qkv_idx * 4 + 1) * hidden + head * 128 + channel];
        float weight1_c1 = weight[(qkv_idx * 4 + 1) * hidden + head * 128 + channel + 1];
        float weight2_c0 = weight[(qkv_idx * 4 + 2) * hidden + head * 128 + channel];
        float weight2_c1 = weight[(qkv_idx * 4 + 2) * hidden + head * 128 + channel + 1];
        float weight3_c0 = weight[(qkv_idx * 4 + 3) * hidden + head * 128 + channel];
        float weight3_c1 = weight[(qkv_idx * 4 + 3) * hidden + head * 128 + channel + 1];
        float mixed0 = r_q[0] * weight0_c0;
        float mixed1 = r_q[1] * weight0_c1;
        mixed0 += r_q[2] * weight1_c0;
        mixed1 += r_q[3] * weight1_c1;
        mixed0 += r_q[4] * weight2_c0;
        mixed1 += r_q[5] * weight2_c1;
        mixed0 += r_q[6] * weight3_c0;
        mixed1 += r_q[7] * weight3_c1;
        float _tanh_approx_0;
        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_0) : "f"(mixed0 * 0.5f));
        float silu0 = mixed0 * (_tanh_approx_0 * 0.5f + 0.5f);
        float _tanh_approx_1;
        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_1) : "f"(mixed1 * 0.5f));
        float silu1 = mixed1 * (_tanh_approx_1 * 0.5f + 0.5f);
        int qk_segment = (3 - qkv_idx) / 2;
        int smem_channel = channel + channel / 8 * 4 * qk_segment;
        sMixed[qkv_idx * 192 + smem_channel] = (float)(__nv_bfloat16)silu0;
        sMixed[qkv_idx * 192 + smem_channel + 1] = (float)(__nv_bfloat16)silu1;
        #pragma unroll
        for (int tap_idx = 0; tap_idx < 6; tap_idx++) {
            sConvNew[tid_0 * 6 + tap_idx] = r_q[2 + tap_idx];
        }
    }
    if (tid_0 >= 384) {
        int k_idx = tid_0 - 384;
        int gate_idx = (row * H + head) * 128 + k_idx;
        float _expf_0 = __expf(A_log[head]);
        float A = _expf_0;
        float gate = (float)raw_gate[gate_idx] + dt_bias[head * 128 + k_idx];
        float decay_log2 = 0.0f;
        if (use_lower_bound != 0) {
            float _tanh_approx_2;
            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_2) : "f"(A * gate * 0.5f));
            decay_log2 = lower_bound_log2 * (_tanh_approx_2 * 0.5f + 0.5f);
        } else {
            float softplus = gate;
            if (gate <= 20.0f) {
                float _expf_1 = __expf(gate);
                float _log2_0;
                asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(1.0f + _expf_1));
                softplus = _log2_0 * 0.6931471805599453f;
            }
            float log_decay = (-A) * softplus;
            decay_log2 = log_decay * 1.4426950408889634f;
        }
        int gate_smem_idx = k_idx + k_idx / 8 * 4;
        float _exp2_0 = approx_exp2(decay_log2);
        sGateDecay[gate_smem_idx] = _exp2_0;
        float output_gate_value = (float)output_gate[row * output_gate_row_stride + head * 128 + k_idx];
        float _tanh_approx_3;
        asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_3) : "f"(output_gate_value * 0.5f));
        sOutputScale[k_idx] = norm_weight[k_idx] * (_tanh_approx_3 * 0.5f + 0.5f);
        if (tid_0 == 384) {
            float beta_raw = (float)raw_beta[row * beta_row_stride + head];
            float _tanh_approx_4;
            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_4) : "f"(beta_raw * 0.5f));
            sBeta[0] = _tanh_approx_4 * 0.5f + 0.5f;
        }
    }
    long long state_head_base = (long long)slot * (long long)state_slot_stride + (long long)(head * 128 * 128);
    long long state_group_base = state_head_base + (long long)((state_owner_row_base + cluster_tile_base) * 128) + (long long)k_start;
    #pragma unroll
    for (int local_row = 0; local_row < 2; local_row++) {
        {
            unsigned _ldv8_3_0;
            unsigned _ldv8_3_1;
            unsigned _ldv8_3_2;
            unsigned _ldv8_3_3;
            unsigned _ldv8_3_4;
            unsigned _ldv8_3_5;
            unsigned _ldv8_3_6;
            unsigned _ldv8_3_7;
            asm volatile("ld.global.L1::no_allocate.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                : "=r"(_ldv8_3_0), "=r"(_ldv8_3_1), "=r"(_ldv8_3_2), "=r"(_ldv8_3_3), "=r"(_ldv8_3_4), "=r"(_ldv8_3_5), "=r"(_ldv8_3_6), "=r"(_ldv8_3_7) : "l"((const void*)(state + (state_group_base + (long long)(local_row * 128)))) : "memory");
            state_regs[local_row * 8 + 0] = __uint_as_float(_ldv8_3_0);
            state_regs[local_row * 8 + 1] = __uint_as_float(_ldv8_3_1);
            state_regs[local_row * 8 + 2] = __uint_as_float(_ldv8_3_2);
            state_regs[local_row * 8 + 3] = __uint_as_float(_ldv8_3_3);
            state_regs[local_row * 8 + 4] = __uint_as_float(_ldv8_3_4);
            state_regs[local_row * 8 + 5] = __uint_as_float(_ldv8_3_5);
            state_regs[local_row * 8 + 6] = __uint_as_float(_ldv8_3_6);
            state_regs[local_row * 8 + 7] = __uint_as_float(_ldv8_3_7);
        }
    }
    asm volatile("barrier.cluster.arrive.relaxed.aligned;" ::: "memory");
    __syncthreads();
    float2 _f2_0 = make_float2(0.0f, 0.0f);
    float2 q_sq_pair = _f2_0;
    float2 _f2_1 = make_float2(0.0f, 0.0f);
    float2 k_sq_pair = _f2_1;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        r_q[i] = 0.0f;
        r_k[i] = 0.0f;
    }
    #pragma unroll
    for (int i_pair = 0; i_pair < 4; i_pair++) {
        int i0 = i_pair * 2;
        int i1 = i0 + 1;
        if (lane_1 < 16) {
            float2 _f2_2 = make_float2(sMixed[qk_smem_start + i0], sMixed[qk_smem_start + i1]);
            float2 q_pair = _f2_2;
            float2 _f2_3 = make_float2(sMixed[192 + qk_smem_start + i0], sMixed[192 + qk_smem_start + i1]);
            float2 k_pair = _f2_3;
            r_q[i0] = q_pair.x;
            r_q[i1] = q_pair.y;
            r_k[i0] = k_pair.x;
            r_k[i1] = k_pair.y;
            q_sq_pair = fma_f32x2_rn_ftz(q_pair, q_pair, q_sq_pair);
            k_sq_pair = fma_f32x2_rn_ftz(k_pair, k_pair, k_sq_pair);
        }
    }
    float q_sq = q_sq_pair.x + q_sq_pair.y;
    float k_sq = k_sq_pair.x + k_sq_pair.y;
    float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, q_sq, 8);
    q_sq += _shfl_xor_0;
    float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, k_sq, 8);
    k_sq += _shfl_xor_1;
    float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, q_sq, 4);
    q_sq += _shfl_xor_2;
    float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, k_sq, 4);
    k_sq += _shfl_xor_3;
    float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, q_sq, 2);
    q_sq += _shfl_xor_4;
    float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, k_sq, 2);
    k_sq += _shfl_xor_5;
    float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, q_sq, 1);
    q_sq += _shfl_xor_6;
    float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, k_sq, 1);
    k_sq += _shfl_xor_7;
    float q_scale = 0.0f;
    float k_scale = 0.0f;
    if (lane_1 < 16) {
        float _rsqrt_0 = rsqrtf(q_sq + 1e-06f);
        q_scale = _rsqrt_0 * 0.08838834764831845f;
        float _rsqrt_1 = rsqrtf(k_sq + 1e-06f);
        k_scale = _rsqrt_1;
    }
    float _shfl_0;
    asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_0) : "f"(q_scale), "r"(lane_group));
    q_scale = _shfl_0;
    float _shfl_1;
    asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_1) : "f"(k_scale), "r"(lane_group));
    k_scale = _shfl_1;
    #pragma unroll
    for (int i_1 = 0; i_1 < 8; i_1++) {
        float _shfl_2;
        asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_2) : "f"(r_q[i_1]), "r"(lane_group));
        r_q[i_1] = _shfl_2;
        float _shfl_3;
        asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_3) : "f"(r_k[i_1]), "r"(lane_group));
        r_k[i_1] = _shfl_3;
    }
    float beta = sBeta[0];
    #pragma unroll
    for (int i_pair_1 = 0; i_pair_1 < 4; i_pair_1++) {
        int i0_1 = i_pair_1 * 2;
        int i1_1 = i0_1 + 1;
        r_decay[i0_1] = sGateDecay[qk_smem_start + i0_1];
        r_decay[i1_1] = sGateDecay[qk_smem_start + i1_1];
    }
    #pragma unroll
    for (int value_tile = 0; value_tile < 1; value_tile++) {
        int tile_base = value_tile * 64 + cluster_tile_base;
        long long state_tile_base = state_group_base + (long long)(value_tile * 64 * 128);
        if (value_tile > 0) {
            #pragma unroll
            for (int local_row_1 = 0; local_row_1 < 2; local_row_1++) {
                {
                    unsigned _ldv8_4_0;
                    unsigned _ldv8_4_1;
                    unsigned _ldv8_4_2;
                    unsigned _ldv8_4_3;
                    unsigned _ldv8_4_4;
                    unsigned _ldv8_4_5;
                    unsigned _ldv8_4_6;
                    unsigned _ldv8_4_7;
                    asm volatile("ld.global.L1::no_allocate.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                        : "=r"(_ldv8_4_0), "=r"(_ldv8_4_1), "=r"(_ldv8_4_2), "=r"(_ldv8_4_3), "=r"(_ldv8_4_4), "=r"(_ldv8_4_5), "=r"(_ldv8_4_6), "=r"(_ldv8_4_7) : "l"((const void*)(state + (state_tile_base + (long long)(local_row_1 * 128)))) : "memory");
                    state_regs[local_row_1 * 8 + 0] = __uint_as_float(_ldv8_4_0);
                    state_regs[local_row_1 * 8 + 1] = __uint_as_float(_ldv8_4_1);
                    state_regs[local_row_1 * 8 + 2] = __uint_as_float(_ldv8_4_2);
                    state_regs[local_row_1 * 8 + 3] = __uint_as_float(_ldv8_4_3);
                    state_regs[local_row_1 * 8 + 4] = __uint_as_float(_ldv8_4_4);
                    state_regs[local_row_1 * 8 + 5] = __uint_as_float(_ldv8_4_5);
                    state_regs[local_row_1 * 8 + 6] = __uint_as_float(_ldv8_4_6);
                    state_regs[local_row_1 * 8 + 7] = __uint_as_float(_ldv8_4_7);
                }
            }
        }
        #pragma unroll
        for (int row_group = 0; row_group < 2 / ((0) ? 4 : 2); row_group++) {
            int local_row_a = row_group * ((0) ? 4 : 2);
            int local_row_b = local_row_a + 1;
            float2 _f2_4 = make_float2(0.0f, 0.0f);
            float2 state_key_pair_a = _f2_4;
            float2 _f2_5 = make_float2(0.0f, 0.0f);
            float2 state_key_pair_b = _f2_5;
            #pragma unroll
            for (int i_pair_2 = 0; i_pair_2 < 4; i_pair_2++) {
                int i0_2 = i_pair_2 * 2;
                int i1_2 = i0_2 + 1;
                int reg_offset_a = local_row_a * 8 + i0_2;
                int reg_offset_b = local_row_b * 8 + i0_2;
                float2 _f2_6 = make_float2(r_decay[i0_2], r_decay[i1_2]);
                float2 decay_pair = _f2_6;
                float2 _f2_7 = make_float2(r_k[i0_2], r_k[i1_2]);
                float2 key_pair = _f2_7;
                float2 _f2_8 = make_float2(state_regs[reg_offset_a], state_regs[reg_offset_a + 1]);
                float2 _mul_f32x2_0;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_0) : "l"(*(const unsigned long long*)&_f2_8), "l"(*(const unsigned long long*)&decay_pair));
                float2 state_pair_a = _mul_f32x2_0;
                float2 _f2_9 = make_float2(state_regs[reg_offset_b], state_regs[reg_offset_b + 1]);
                float2 _mul_f32x2_1;
                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_1) : "l"(*(const unsigned long long*)&_f2_9), "l"(*(const unsigned long long*)&decay_pair));
                float2 state_pair_b = _mul_f32x2_1;
                state_regs[reg_offset_a] = state_pair_a.x;
                state_regs[reg_offset_a + 1] = state_pair_a.y;
                state_regs[reg_offset_b] = state_pair_b.x;
                state_regs[reg_offset_b + 1] = state_pair_b.y;
                state_key_pair_a = fma_f32x2_rn_ftz(state_pair_a, key_pair, state_key_pair_a);
                state_key_pair_b = fma_f32x2_rn_ftz(state_pair_b, key_pair, state_key_pair_b);
            }
            float state_key_dot_a = state_key_pair_a.x + state_key_pair_a.y;
            float state_key_dot_b = state_key_pair_b.x + state_key_pair_b.y;
            float _shfl_xor_8 = __shfl_xor_sync(0xFFFFFFFF, state_key_dot_a, 8);
            state_key_dot_a += _shfl_xor_8;
            float _shfl_xor_9 = __shfl_xor_sync(0xFFFFFFFF, state_key_dot_b, 8);
            state_key_dot_b += _shfl_xor_9;
            float _shfl_xor_10 = __shfl_xor_sync(0xFFFFFFFF, state_key_dot_a, 4);
            state_key_dot_a += _shfl_xor_10;
            float _shfl_xor_11 = __shfl_xor_sync(0xFFFFFFFF, state_key_dot_b, 4);
            state_key_dot_b += _shfl_xor_11;
            float _shfl_xor_12 = __shfl_xor_sync(0xFFFFFFFF, state_key_dot_a, 2);
            state_key_dot_a += _shfl_xor_12;
            float _shfl_xor_13 = __shfl_xor_sync(0xFFFFFFFF, state_key_dot_b, 2);
            state_key_dot_b += _shfl_xor_13;
            float _shfl_xor_14 = __shfl_xor_sync(0xFFFFFFFF, state_key_dot_a, 1);
            state_key_dot_a += _shfl_xor_14;
            float _shfl_xor_15 = __shfl_xor_sync(0xFFFFFFFF, state_key_dot_b, 1);
            state_key_dot_b += _shfl_xor_15;
            state_key_dot_a *= k_scale;
            state_key_dot_b *= k_scale;
            int value_row_a = tile_base + state_owner_row_base + local_row_a;
            int value_row_b = tile_base + state_owner_row_base + local_row_b;
            int value_smem_row_a = value_row_a;
            int value_smem_row_b = value_row_b;
            float value_a = sMixed[384 + value_smem_row_a];
            float value_b = sMixed[384 + value_smem_row_b];
            float delta_a = (value_a - state_key_dot_a) * beta;
            float delta_b = (value_b - state_key_dot_b) * beta;
            float delta_key_scale_a = delta_a * k_scale;
            float delta_key_scale_b = delta_b * k_scale;
            float recurrence_value_a = 0.0f;
            float recurrence_value_b = 0.0f;
            float2 _f2_10 = make_float2(0.0f, 0.0f);
            float2 state_query_pair_a = _f2_10;
            float2 _f2_11 = make_float2(0.0f, 0.0f);
            float2 state_query_pair_b = _f2_11;
            #pragma unroll
            for (int i_pair_3 = 0; i_pair_3 < 4; i_pair_3++) {
                int i0_3 = i_pair_3 * 2;
                int i1_3 = i0_3 + 1;
                int reg_offset_a_1 = local_row_a * 8 + i0_3;
                int reg_offset_b_1 = local_row_b * 8 + i0_3;
                float2 _f2_12 = make_float2(r_k[i0_3], r_k[i1_3]);
                float2 key_pair_1 = _f2_12;
                float2 _f2_13 = make_float2(delta_key_scale_a, delta_key_scale_a);
                float2 _f2_14 = make_float2(state_regs[reg_offset_a_1], state_regs[reg_offset_a_1 + 1]);
                float2 updated_pair_a = fma_f32x2_rn_ftz(_f2_13, key_pair_1, _f2_14);
                float2 _f2_15 = make_float2(delta_key_scale_b, delta_key_scale_b);
                float2 _f2_16 = make_float2(state_regs[reg_offset_b_1], state_regs[reg_offset_b_1 + 1]);
                float2 updated_pair_b = fma_f32x2_rn_ftz(_f2_15, key_pair_1, _f2_16);
                state_regs[reg_offset_a_1] = updated_pair_a.x;
                state_regs[reg_offset_a_1 + 1] = updated_pair_a.y;
                state_regs[reg_offset_b_1] = updated_pair_b.x;
                state_regs[reg_offset_b_1 + 1] = updated_pair_b.y;
                float2 _f2_17 = make_float2(r_q[i0_3], r_q[i1_3]);
                float2 query_pair = _f2_17;
                state_query_pair_a = fma_f32x2_rn_ftz(updated_pair_a, query_pair, state_query_pair_a);
                state_query_pair_b = fma_f32x2_rn_ftz(updated_pair_b, query_pair, state_query_pair_b);
            }
            if (is_live != 0) {
                {
                    unsigned _stv8_5_0 = __float_as_uint(state_regs[local_row_a * 8 + 0]);
                    unsigned _stv8_5_1 = __float_as_uint(state_regs[local_row_a * 8 + 1]);
                    unsigned _stv8_5_2 = __float_as_uint(state_regs[local_row_a * 8 + 2]);
                    unsigned _stv8_5_3 = __float_as_uint(state_regs[local_row_a * 8 + 3]);
                    unsigned _stv8_5_4 = __float_as_uint(state_regs[local_row_a * 8 + 4]);
                    unsigned _stv8_5_5 = __float_as_uint(state_regs[local_row_a * 8 + 5]);
                    unsigned _stv8_5_6 = __float_as_uint(state_regs[local_row_a * 8 + 6]);
                    unsigned _stv8_5_7 = __float_as_uint(state_regs[local_row_a * 8 + 7]);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(state + (state_tile_base + (long long)(local_row_a * 128)))), "r"(_stv8_5_0), "r"(_stv8_5_1), "r"(_stv8_5_2), "r"(_stv8_5_3), "r"(_stv8_5_4), "r"(_stv8_5_5), "r"(_stv8_5_6), "r"(_stv8_5_7) : "memory");
                }
                {
                    unsigned _stv8_6_0 = __float_as_uint(state_regs[local_row_b * 8 + 0]);
                    unsigned _stv8_6_1 = __float_as_uint(state_regs[local_row_b * 8 + 1]);
                    unsigned _stv8_6_2 = __float_as_uint(state_regs[local_row_b * 8 + 2]);
                    unsigned _stv8_6_3 = __float_as_uint(state_regs[local_row_b * 8 + 3]);
                    unsigned _stv8_6_4 = __float_as_uint(state_regs[local_row_b * 8 + 4]);
                    unsigned _stv8_6_5 = __float_as_uint(state_regs[local_row_b * 8 + 5]);
                    unsigned _stv8_6_6 = __float_as_uint(state_regs[local_row_b * 8 + 6]);
                    unsigned _stv8_6_7 = __float_as_uint(state_regs[local_row_b * 8 + 7]);
                    asm volatile(
                        "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                        :: "l"((void*)(state + (state_tile_base + (long long)(local_row_b * 128)))), "r"(_stv8_6_0), "r"(_stv8_6_1), "r"(_stv8_6_2), "r"(_stv8_6_3), "r"(_stv8_6_4), "r"(_stv8_6_5), "r"(_stv8_6_6), "r"(_stv8_6_7) : "memory");
                }
            }
            recurrence_value_a = state_query_pair_a.x + state_query_pair_a.y;
            recurrence_value_b = state_query_pair_b.x + state_query_pair_b.y;
            float _shfl_xor_16 = __shfl_xor_sync(0xFFFFFFFF, recurrence_value_a, 8);
            recurrence_value_a += _shfl_xor_16;
            float _shfl_xor_17 = __shfl_xor_sync(0xFFFFFFFF, recurrence_value_b, 8);
            recurrence_value_b += _shfl_xor_17;
            float _shfl_xor_18 = __shfl_xor_sync(0xFFFFFFFF, recurrence_value_a, 4);
            recurrence_value_a += _shfl_xor_18;
            float _shfl_xor_19 = __shfl_xor_sync(0xFFFFFFFF, recurrence_value_b, 4);
            recurrence_value_b += _shfl_xor_19;
            float _shfl_xor_20 = __shfl_xor_sync(0xFFFFFFFF, recurrence_value_a, 2);
            recurrence_value_a += _shfl_xor_20;
            float _shfl_xor_21 = __shfl_xor_sync(0xFFFFFFFF, recurrence_value_b, 2);
            recurrence_value_b += _shfl_xor_21;
            float _shfl_xor_22 = __shfl_xor_sync(0xFFFFFFFF, recurrence_value_a, 1);
            recurrence_value_a += _shfl_xor_22;
            float _shfl_xor_23 = __shfl_xor_sync(0xFFFFFFFF, recurrence_value_b, 1);
            recurrence_value_b += _shfl_xor_23;
            recurrence_value_a *= q_scale;
            recurrence_value_b *= q_scale;
            if (lane_group == 0) {
                sRecurrence[value_row_a] = recurrence_value_a;
                sRecurrence[value_row_b] = recurrence_value_b;
            }
        }
    }
    __syncthreads();
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    int conv_writer_producer = tid_0 - 96;
    if (tid_0 / 96 == cluster_rank + 1) {
        int spread_qkv_idx = conv_writer_producer / 64;
        int spread_channel = (conv_writer_producer - spread_qkv_idx * 64) * 2;
        int spread_channel_base = spread_qkv_idx * hidden + head * 128 + spread_channel;
        long long spread_store_base = (long long)slot * (long long)conv_slot_stride + (long long)spread_channel_base;
        if (is_live != 0) {
            conv_state[spread_store_base] = sConvNew[conv_writer_producer * 6];
            conv_state[spread_store_base + 1] = sConvNew[conv_writer_producer * 6 + 1];
            conv_state[spread_store_base + (long long)qkv_size] = sConvNew[conv_writer_producer * 6 + 2];
            conv_state[spread_store_base + (long long)qkv_size + 1] = sConvNew[conv_writer_producer * 6 + 3];
            conv_state[spread_store_base + (long long)(2 * qkv_size)] = sConvNew[conv_writer_producer * 6 + 4];
            conv_state[spread_store_base + (long long)(2 * qkv_size) + 1] = sConvNew[conv_writer_producer * 6 + 5];
        }
    }
    if (warp == 0) {
        float tile_partial = 0.0f;
        if (lane_1 < 16) {
            __nv_bfloat162 partial_pair0 = reinterpret_cast<const __nv_bfloat162*>(sRecurrence)[cluster_tile_base / 2 + lane_1 * 2];
            __nv_bfloat162 partial_pair1 = reinterpret_cast<const __nv_bfloat162*>(sRecurrence)[cluster_tile_base / 2 + lane_1 * 2 + 1];
            float partial_v0 = (float)partial_pair0.x;
            float partial_v1 = (float)partial_pair0.y;
            float partial_v2 = (float)partial_pair1.x;
            float partial_v3 = (float)partial_pair1.y;
            tile_partial = partial_v0 * partial_v0 + partial_v1 * partial_v1 + partial_v2 * partial_v2 + partial_v3 * partial_v3;
        }
        float _shfl_xor_24 = __shfl_xor_sync(0xFFFFFFFF, tile_partial, 8);
        tile_partial += _shfl_xor_24;
        float _shfl_xor_25 = __shfl_xor_sync(0xFFFFFFFF, tile_partial, 4);
        tile_partial += _shfl_xor_25;
        float _shfl_xor_26 = __shfl_xor_sync(0xFFFFFFFF, tile_partial, 2);
        tile_partial += _shfl_xor_26;
        float _shfl_xor_27 = __shfl_xor_sync(0xFFFFFFFF, tile_partial, 1);
        tile_partial += _shfl_xor_27;
        if (lane_1 == 0) {
            mbarrier_arrive_expect_tx(cluster_rms_partial_addr, 4);
            uint32_t _mapa_0;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_0) : "r"(sClusterRms_addr + 4), "r"(1 - cluster_rank));
            uint32_t _mapa_1;
            asm volatile(
                "mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(_mapa_1) : "r"(cluster_rms_partial_addr), "r"(1 - cluster_rank));
            asm volatile(
                "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [%0], %1, [%2];"
                :: "r"(_mapa_0), "f"(tile_partial), "r"(_mapa_1) : "memory");
        }
        mbarrier_wait(cluster_rms_partial_addr, 0);
        float pair_sum_squares = tile_partial + sClusterRms[1];
        float _rsqrt_2 = rsqrtf(pair_sum_squares * 0.0078125f + norm_eps);
        float pair_inverse_rms = _rsqrt_2;
        if (lane_1 < 16) {
            int tile_column = cluster_tile_base + lane_1 * 4;
            float pair_output_values[4];
            __nv_bfloat162 output_pair0 = reinterpret_cast<const __nv_bfloat162*>(sRecurrence)[cluster_tile_base / 2 + lane_1 * 2];
            __nv_bfloat162 output_pair1 = reinterpret_cast<const __nv_bfloat162*>(sRecurrence)[cluster_tile_base / 2 + lane_1 * 2 + 1];
            pair_output_values[0] = (float)output_pair0.x;
            pair_output_values[1] = (float)output_pair0.y;
            pair_output_values[2] = (float)output_pair1.x;
            pair_output_values[3] = (float)output_pair1.y;
            float pair_output_scales[4];
            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                : "=r"(*reinterpret_cast<uint32_t*>(&pair_output_scales[0])), "=r"(*reinterpret_cast<uint32_t*>(&pair_output_scales[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&pair_output_scales[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&pair_output_scales[(0) + 3]))
                : "r"(sOutputScale_addr + (unsigned int)(tile_column * 4)));
            #pragma unroll
            for (int channel_idx = 0; channel_idx < 4; channel_idx++) {
                if (is_live != 0) {
                    pair_output_values[channel_idx] = pair_output_values[channel_idx] * pair_inverse_rms * pair_output_scales[channel_idx];
                } else {
                    pair_output_values[channel_idx] = 0.0f;
                }
            }
            int pair_output_base = (row * H + head) * 128 + tile_column;
            {
                uint2 _pk2;
                __nv_bfloat162* _pk = reinterpret_cast<__nv_bfloat162*>(&_pk2);
                _pk[0] = __floats2bfloat162_rn(pair_output_values[0 + 0], pair_output_values[0 + 1]);
                _pk[1] = __floats2bfloat162_rn(pair_output_values[0 + 2], pair_output_values[0 + 3]);
                *reinterpret_cast<uint2*>(&((__nv_bfloat16*)(output))[pair_output_base]) = _pk2;
            }
        }
    }
}

} // extern "C"

