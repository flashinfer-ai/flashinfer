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

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 320
#define TMEM_TMEM_M_OFFSET 0
#define TMEM_TMEM_INP_OFFSET 128
#define TMEM_TMEM_M_ALT_OFFSET 192
#define NUM_R_PIPE_STAGES 3
#define SMEM_SMEM_R_OFF 1024
#define SMEM_SMEM_R_STAGE_BYTES 32768
#define SMEM_SMEM_R_STRIDE 32768
#define SMEM_SMEM_D_OFF 99328
#define SMEM_SMEM_D_STAGE_BYTES 512
#define SMEM_SMEM_D_STRIDE 1024
#define SMEM_SMEM_D_ALL_OFF 99328
#define SMEM_SMEM_D_ALL_STAGE_BYTES 2560
#define SMEM_SMEM_D_ALL_STRIDE 2560
#define SMEM_SMEM_PANEL_OFF 102400
#define SMEM_SMEM_PANEL_STAGE_BYTES 8192
#define SMEM_SMEM_PANEL_STRIDE 8192
#define SMEM_TOTAL 118784
#define THREADS 320

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
        :: "l"(mbar_addr), "r"(count) : "memory");
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


__device__ __forceinline__ void tma_store_3d(
    const void *tmap, int x, int y, int z, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3}], [%4];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(smem_addr) : "memory");
}


__device__ __forceinline__ void cp_async_bulk_gmem2smem(
    unsigned smem_addr, const void* gmem_ptr, unsigned bytes, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1], %2, [%3];"
        :: "r"(smem_addr), "l"(gmem_ptr), "r"(bytes), "r"(mbar_addr)
        : "memory");
}


__device__ __forceinline__ void tcgen05_commit(int mbar_addr) {
    asm volatile(
        "tcgen05.commit.cta_group::1.mbarrier::arrive::one"
        ".shared::cluster.b64 [%0];"
        :: "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(320) void
kernel_cake_kda_bf16_4a1e3751aebc32ccd4531543bd6b73a38a9ea2f8094f113d0aa7fb76f28c266d(int* __restrict__ items, int num_heads, __nv_bfloat16* __restrict__ pair, CakeTensorMap const* pair_tma, float* __restrict__ dpair, __nv_bfloat16* __restrict__ maps, CakeTensorMap const* maps_tma, __nv_bfloat16* __restrict__ map_final, CakeTensorMap const* map_final_tma)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define r_full_addr (mbar_base + 0)
    #define r_free_addr (mbar_base + 24)
    #define init_ready_addr (mbar_base + 48)
    #define m_ready_addr (mbar_base + 56)
    #define inp_ready_addr (mbar_base + 72)
    #define snap_done_addr (mbar_base + 80)
    #define tmem_dealloc_ready_addr (mbar_base + 96)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

    const int cta_rank = 0;
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(pair_tma)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(maps_tma)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(map_final_tma)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    __nv_bfloat16* smem_r = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_r_addr = smem + 1024;
    float* smem_d = reinterpret_cast<float*>(smem_raw + 99328);
    const int smem_d_addr = smem + 99328;
    float* smem_d_all = reinterpret_cast<float*>(smem_raw + 99328);
    const int smem_d_all_addr = smem + 99328;
    __nv_bfloat16* smem_panel = reinterpret_cast<__nv_bfloat16*>(smem_raw + 102400);
    const int smem_panel_addr = smem + 102400;

    // Mbarrier init (7 pipeline groups, 0 ordered-sequence groups, 13 barriers)
    // Mbarriers at smem_raw[0..104)

    if (warp == 0) {
        // --- pipeline 'r_pipe' ---
        // r_full: 3 barriers, init_count=1
        // r_free: 3 barriers, init_count=1
        // init_ready: 1 barriers, init_count=4
        // m_ready: 2 barriers, init_count=1
        // inp_ready: 1 barriers, init_count=4
        // snap_done: 2 barriers, init_count=4
        // tmem_dealloc_ready: 1 barriers, init_count=2
        // Warp-cooperative initialization in physical record order.
        uint32_t _mbarrier_init_count_0_0 = 2;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(12), "r"((uint32_t)(4)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(9), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(7), "r"((uint32_t)(4)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(6), "r"((uint32_t)(1)));
        if (lane < 13) {
            mbarrier_init(smem + 0 + lane * 8, _mbarrier_init_count_0_0);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    __syncwarp();

    // TMEM alloc (512 columns, 320 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 104);
    if (warp == 0) {
        int _tmem_hold = smem + 104;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_m = taddr;
    const int tmem_tmem_inp = taddr + 128;
    const int tmem_tmem_m_alt = taddr + 192;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
    }

    // ---- Role: compute ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 152;");
        { // compute_main
            int item_base = blockIdx.x * 4;
            int snap_outer = items[item_base];
            int num_blocks = items[item_base + 1];
            int final_outer = items[item_base + 2];
            int warp_in_wg = warp % 4;
            const int tmem_row_base = warp_in_wg * 32 << 16;
            int state_row = warp_in_wg * 32 + lane;
            int m_base = taddr + (unsigned int)tmem_row_base;
            int inp_base = taddr + 128 + (unsigned int)tmem_row_base;
            unsigned int compute_stage = 0;
            #pragma unroll
            for (int eye_col_block = 0; eye_col_block < 4; eye_col_block++) {
                float eye[32];
                int eye_diag = state_row - eye_col_block * 32;
                #pragma unroll
                for (int eye_col = 0; eye_col < 32; eye_col++) {
                    eye[eye_col] = ((eye_diag == eye_col) ? 1.0f : 0.0f);
                }
                tmem_st_x32_f32(m_base + eye_col_block * 32, eye);
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            if (elect_sync()) {
                mbarrier_arrive(init_ready_addr);
            }
            unsigned int _phase_init_ready_0 = 0;
            mbarrier_wait(init_ready_addr, _phase_init_ready_0);
            _phase_init_ready_0 ^= 1;
            unsigned int _phase_r_full = 0;
            #pragma unroll 1
            for (int p = 0; p < num_blocks; p++) {
                int is_first = ((p == 0) ? 1 : 0);
                int src_sel = p % 2 * 192;
                int dst_sel = 192 - src_sel;
                if (is_first == 0) {
                    int prev_block = p - 1;
                    int prev_stage = prev_block & 1;
                    int prev_phase = prev_block >> 1 & 1;
                    mbarrier_wait(m_ready_addr + (prev_stage) * 8, prev_phase);
                    mbarrier_wait(snap_done_addr + (prev_stage) * 8, prev_phase);
                }
                mbarrier_wait(r_full_addr + (compute_stage) * 8, _phase_r_full);
                int d_base = (int)compute_stage * 256;
                float m_frag0[32];
                float m_frag1[32];
                float m_frag2[32];
                float m_frag3[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(m_frag0[0]), "=f"(m_frag0[1]), "=f"(m_frag0[2]), "=f"(m_frag0[3]), "=f"(m_frag0[4]), "=f"(m_frag0[5]), "=f"(m_frag0[6]), "=f"(m_frag0[7]), "=f"(m_frag0[8]), "=f"(m_frag0[9]), "=f"(m_frag0[10]), "=f"(m_frag0[11]), "=f"(m_frag0[12]), "=f"(m_frag0[13]), "=f"(m_frag0[14]), "=f"(m_frag0[15]), "=f"(m_frag0[16]), "=f"(m_frag0[17]), "=f"(m_frag0[18]), "=f"(m_frag0[19]), "=f"(m_frag0[20]), "=f"(m_frag0[21]), "=f"(m_frag0[22]), "=f"(m_frag0[23]), "=f"(m_frag0[24]), "=f"(m_frag0[25]), "=f"(m_frag0[26]), "=f"(m_frag0[27]), "=f"(m_frag0[28]), "=f"(m_frag0[29]), "=f"(m_frag0[30]), "=f"(m_frag0[31])
                    : "r"(m_base + src_sel));
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(m_frag1[0]), "=f"(m_frag1[1]), "=f"(m_frag1[2]), "=f"(m_frag1[3]), "=f"(m_frag1[4]), "=f"(m_frag1[5]), "=f"(m_frag1[6]), "=f"(m_frag1[7]), "=f"(m_frag1[8]), "=f"(m_frag1[9]), "=f"(m_frag1[10]), "=f"(m_frag1[11]), "=f"(m_frag1[12]), "=f"(m_frag1[13]), "=f"(m_frag1[14]), "=f"(m_frag1[15]), "=f"(m_frag1[16]), "=f"(m_frag1[17]), "=f"(m_frag1[18]), "=f"(m_frag1[19]), "=f"(m_frag1[20]), "=f"(m_frag1[21]), "=f"(m_frag1[22]), "=f"(m_frag1[23]), "=f"(m_frag1[24]), "=f"(m_frag1[25]), "=f"(m_frag1[26]), "=f"(m_frag1[27]), "=f"(m_frag1[28]), "=f"(m_frag1[29]), "=f"(m_frag1[30]), "=f"(m_frag1[31])
                    : "r"(m_base + src_sel + 32));
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(m_frag2[0]), "=f"(m_frag2[1]), "=f"(m_frag2[2]), "=f"(m_frag2[3]), "=f"(m_frag2[4]), "=f"(m_frag2[5]), "=f"(m_frag2[6]), "=f"(m_frag2[7]), "=f"(m_frag2[8]), "=f"(m_frag2[9]), "=f"(m_frag2[10]), "=f"(m_frag2[11]), "=f"(m_frag2[12]), "=f"(m_frag2[13]), "=f"(m_frag2[14]), "=f"(m_frag2[15]), "=f"(m_frag2[16]), "=f"(m_frag2[17]), "=f"(m_frag2[18]), "=f"(m_frag2[19]), "=f"(m_frag2[20]), "=f"(m_frag2[21]), "=f"(m_frag2[22]), "=f"(m_frag2[23]), "=f"(m_frag2[24]), "=f"(m_frag2[25]), "=f"(m_frag2[26]), "=f"(m_frag2[27]), "=f"(m_frag2[28]), "=f"(m_frag2[29]), "=f"(m_frag2[30]), "=f"(m_frag2[31])
                    : "r"(m_base + src_sel + 64));
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(m_frag3[0]), "=f"(m_frag3[1]), "=f"(m_frag3[2]), "=f"(m_frag3[3]), "=f"(m_frag3[4]), "=f"(m_frag3[5]), "=f"(m_frag3[6]), "=f"(m_frag3[7]), "=f"(m_frag3[8]), "=f"(m_frag3[9]), "=f"(m_frag3[10]), "=f"(m_frag3[11]), "=f"(m_frag3[12]), "=f"(m_frag3[13]), "=f"(m_frag3[14]), "=f"(m_frag3[15]), "=f"(m_frag3[16]), "=f"(m_frag3[17]), "=f"(m_frag3[18]), "=f"(m_frag3[19]), "=f"(m_frag3[20]), "=f"(m_frag3[21]), "=f"(m_frag3[22]), "=f"(m_frag3[23]), "=f"(m_frag3[24]), "=f"(m_frag3[25]), "=f"(m_frag3[26]), "=f"(m_frag3[27]), "=f"(m_frag3[28]), "=f"(m_frag3[29]), "=f"(m_frag3[30]), "=f"(m_frag3[31])
                    : "r"(m_base + src_sel + 96));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                uint32_t m_frag0_bf16[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(m_frag0[_lp*2 + 0], m_frag0[_lp*2+1 + 0]));
                    m_frag0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x16.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(inp_base), "r"(m_frag0_bf16[0]), "r"(m_frag0_bf16[1]), "r"(m_frag0_bf16[2]), "r"(m_frag0_bf16[3]), "r"(m_frag0_bf16[4]), "r"(m_frag0_bf16[5]), "r"(m_frag0_bf16[6]), "r"(m_frag0_bf16[7]), "r"(m_frag0_bf16[8]), "r"(m_frag0_bf16[9]), "r"(m_frag0_bf16[10]), "r"(m_frag0_bf16[11]), "r"(m_frag0_bf16[12]), "r"(m_frag0_bf16[13]), "r"(m_frag0_bf16[14]), "r"(m_frag0_bf16[15]));
                float d_scale[16];
                #pragma unroll
                for (int half = 0; half < 2; half++) {
                    #pragma unroll
                    for (int col = 0; col < 16; col++) {
                        d_scale[col] = smem_d_all[d_base + half * 16 + col];
                    }
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>((m_frag0 + half * 16))[_ls], reinterpret_cast<const float2*>(d_scale)[_ls]);
                }
                tmem_st_x32_f32(m_base + dst_sel, m_frag0);
                uint32_t m_frag1_bf16[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(m_frag1[_lp*2 + 0], m_frag1[_lp*2+1 + 0]));
                    m_frag1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x16.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(inp_base + 16), "r"(m_frag1_bf16[0]), "r"(m_frag1_bf16[1]), "r"(m_frag1_bf16[2]), "r"(m_frag1_bf16[3]), "r"(m_frag1_bf16[4]), "r"(m_frag1_bf16[5]), "r"(m_frag1_bf16[6]), "r"(m_frag1_bf16[7]), "r"(m_frag1_bf16[8]), "r"(m_frag1_bf16[9]), "r"(m_frag1_bf16[10]), "r"(m_frag1_bf16[11]), "r"(m_frag1_bf16[12]), "r"(m_frag1_bf16[13]), "r"(m_frag1_bf16[14]), "r"(m_frag1_bf16[15]));
                float d_scale_0[16];
                #pragma unroll
                for (int half_1 = 0; half_1 < 2; half_1++) {
                    #pragma unroll
                    for (int col_1 = 0; col_1 < 16; col_1++) {
                        d_scale_0[col_1] = smem_d_all[d_base + 32 + half_1 * 16 + col_1];
                    }
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>((m_frag1 + half_1 * 16))[_ls], reinterpret_cast<const float2*>(d_scale_0)[_ls]);
                }
                tmem_st_x32_f32(m_base + dst_sel + 32, m_frag1);
                uint32_t m_frag2_bf16[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(m_frag2[_lp*2 + 0], m_frag2[_lp*2+1 + 0]));
                    m_frag2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x16.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(inp_base + 32), "r"(m_frag2_bf16[0]), "r"(m_frag2_bf16[1]), "r"(m_frag2_bf16[2]), "r"(m_frag2_bf16[3]), "r"(m_frag2_bf16[4]), "r"(m_frag2_bf16[5]), "r"(m_frag2_bf16[6]), "r"(m_frag2_bf16[7]), "r"(m_frag2_bf16[8]), "r"(m_frag2_bf16[9]), "r"(m_frag2_bf16[10]), "r"(m_frag2_bf16[11]), "r"(m_frag2_bf16[12]), "r"(m_frag2_bf16[13]), "r"(m_frag2_bf16[14]), "r"(m_frag2_bf16[15]));
                float d_scale_1[16];
                #pragma unroll
                for (int half_2 = 0; half_2 < 2; half_2++) {
                    #pragma unroll
                    for (int col_2 = 0; col_2 < 16; col_2++) {
                        d_scale_1[col_2] = smem_d_all[d_base + 64 + half_2 * 16 + col_2];
                    }
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>((m_frag2 + half_2 * 16))[_ls], reinterpret_cast<const float2*>(d_scale_1)[_ls]);
                }
                tmem_st_x32_f32(m_base + dst_sel + 64, m_frag2);
                uint32_t m_frag3_bf16[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(m_frag3[_lp*2 + 0], m_frag3[_lp*2+1 + 0]));
                    m_frag3_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x16.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(inp_base + 48), "r"(m_frag3_bf16[0]), "r"(m_frag3_bf16[1]), "r"(m_frag3_bf16[2]), "r"(m_frag3_bf16[3]), "r"(m_frag3_bf16[4]), "r"(m_frag3_bf16[5]), "r"(m_frag3_bf16[6]), "r"(m_frag3_bf16[7]), "r"(m_frag3_bf16[8]), "r"(m_frag3_bf16[9]), "r"(m_frag3_bf16[10]), "r"(m_frag3_bf16[11]), "r"(m_frag3_bf16[12]), "r"(m_frag3_bf16[13]), "r"(m_frag3_bf16[14]), "r"(m_frag3_bf16[15]));
                float d_scale_2[16];
                #pragma unroll
                for (int half_3 = 0; half_3 < 2; half_3++) {
                    #pragma unroll
                    for (int col_3 = 0; col_3 < 16; col_3++) {
                        d_scale_2[col_3] = smem_d_all[d_base + 96 + half_3 * 16 + col_3];
                    }
                    #pragma unroll
                    for (int _ls = 0; _ls < 8; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>((m_frag3 + half_3 * 16))[_ls], reinterpret_cast<const float2*>(d_scale_2)[_ls]);
                }
                tmem_st_x32_f32(m_base + dst_sel + 96, m_frag3);
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                if (elect_sync()) {
                    mbarrier_arrive(inp_ready_addr);
                }
                compute_stage += 1;
                if (compute_stage == 3) { compute_stage = 0; _phase_r_full ^= 1; }
            }
            if (warp_in_wg == 0) {
                if (elect_sync()) {
                    mbarrier_arrive(tmem_dealloc_ready_addr);
                }
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 72;");
        { // epilogue_main
            int item_base_1 = blockIdx.x * 4;
            int snap_outer_1 = items[item_base_1];
            int num_blocks_1 = items[item_base_1 + 1];
            int final_outer_1 = items[item_base_1 + 2];
            int warp_in_wg_1 = warp % 4;
            const int tmem_row_base_1 = warp_in_wg_1 * 32 << 16;
            int state_row_1 = warp_in_wg_1 * 32 + lane;
            int epilogue_local_warp = warp_in_wg_1;
            int m_base_1 = taddr + (unsigned int)tmem_row_base_1;
            unsigned int _phase_init_ready_0_1 = 0;
            #pragma unroll 1
            for (int p_1 = 0; p_1 < num_blocks_1 + 1; p_1++) {
                int is_first_1 = ((p_1 == 0) ? 1 : 0);
                int is_last = ((num_blocks_1 <= p_1) ? 1 : 0);
                if (is_first_1 != 0) {
                    mbarrier_wait(init_ready_addr, _phase_init_ready_0_1);
                    _phase_init_ready_0_1 ^= 1;
                } else {
                    int prev_block_1 = p_1 - 1;
                    int prev_stage_1 = prev_block_1 & 1;
                    int prev_phase_1 = prev_block_1 >> 1 & 1;
                    mbarrier_wait(m_ready_addr + (prev_stage_1) * 8, prev_phase_1);
                }
                int snap_sel = p_1 % 2 * 192;
                float frag0[32];
                float frag1[32];
                float frag2[32];
                float frag3[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(frag0[0]), "=f"(frag0[1]), "=f"(frag0[2]), "=f"(frag0[3]), "=f"(frag0[4]), "=f"(frag0[5]), "=f"(frag0[6]), "=f"(frag0[7]), "=f"(frag0[8]), "=f"(frag0[9]), "=f"(frag0[10]), "=f"(frag0[11]), "=f"(frag0[12]), "=f"(frag0[13]), "=f"(frag0[14]), "=f"(frag0[15]), "=f"(frag0[16]), "=f"(frag0[17]), "=f"(frag0[18]), "=f"(frag0[19]), "=f"(frag0[20]), "=f"(frag0[21]), "=f"(frag0[22]), "=f"(frag0[23]), "=f"(frag0[24]), "=f"(frag0[25]), "=f"(frag0[26]), "=f"(frag0[27]), "=f"(frag0[28]), "=f"(frag0[29]), "=f"(frag0[30]), "=f"(frag0[31])
                    : "r"(m_base_1 + snap_sel));
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(frag1[0]), "=f"(frag1[1]), "=f"(frag1[2]), "=f"(frag1[3]), "=f"(frag1[4]), "=f"(frag1[5]), "=f"(frag1[6]), "=f"(frag1[7]), "=f"(frag1[8]), "=f"(frag1[9]), "=f"(frag1[10]), "=f"(frag1[11]), "=f"(frag1[12]), "=f"(frag1[13]), "=f"(frag1[14]), "=f"(frag1[15]), "=f"(frag1[16]), "=f"(frag1[17]), "=f"(frag1[18]), "=f"(frag1[19]), "=f"(frag1[20]), "=f"(frag1[21]), "=f"(frag1[22]), "=f"(frag1[23]), "=f"(frag1[24]), "=f"(frag1[25]), "=f"(frag1[26]), "=f"(frag1[27]), "=f"(frag1[28]), "=f"(frag1[29]), "=f"(frag1[30]), "=f"(frag1[31])
                    : "r"(m_base_1 + snap_sel + 32));
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(frag2[0]), "=f"(frag2[1]), "=f"(frag2[2]), "=f"(frag2[3]), "=f"(frag2[4]), "=f"(frag2[5]), "=f"(frag2[6]), "=f"(frag2[7]), "=f"(frag2[8]), "=f"(frag2[9]), "=f"(frag2[10]), "=f"(frag2[11]), "=f"(frag2[12]), "=f"(frag2[13]), "=f"(frag2[14]), "=f"(frag2[15]), "=f"(frag2[16]), "=f"(frag2[17]), "=f"(frag2[18]), "=f"(frag2[19]), "=f"(frag2[20]), "=f"(frag2[21]), "=f"(frag2[22]), "=f"(frag2[23]), "=f"(frag2[24]), "=f"(frag2[25]), "=f"(frag2[26]), "=f"(frag2[27]), "=f"(frag2[28]), "=f"(frag2[29]), "=f"(frag2[30]), "=f"(frag2[31])
                    : "r"(m_base_1 + snap_sel + 64));
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(frag3[0]), "=f"(frag3[1]), "=f"(frag3[2]), "=f"(frag3[3]), "=f"(frag3[4]), "=f"(frag3[5]), "=f"(frag3[6]), "=f"(frag3[7]), "=f"(frag3[8]), "=f"(frag3[9]), "=f"(frag3[10]), "=f"(frag3[11]), "=f"(frag3[12]), "=f"(frag3[13]), "=f"(frag3[14]), "=f"(frag3[15]), "=f"(frag3[16]), "=f"(frag3[17]), "=f"(frag3[18]), "=f"(frag3[19]), "=f"(frag3[20]), "=f"(frag3[21]), "=f"(frag3[22]), "=f"(frag3[23]), "=f"(frag3[24]), "=f"(frag3[25]), "=f"(frag3[26]), "=f"(frag3[27]), "=f"(frag3[28]), "=f"(frag3[29]), "=f"(frag3[30]), "=f"(frag3[31])
                    : "r"(m_base_1 + snap_sel + 96));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                int snap_arrive_stage = p_1 & 1;
                if (elect_sync()) {
                    mbarrier_arrive(snap_done_addr + (snap_arrive_stage) * 8);
                }
                if (epilogue_local_warp == 0) {
                    asm volatile("cp.async.bulk.wait_group.read 0;");
                }
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                uint32_t frag0_bf16[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(frag0[_lp*2 + 0], frag0[_lp*2+1 + 0]));
                    frag0_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_panel_addr + (unsigned int)(state_row_1 * 128 ^ (state_row_1 * 128 >> 7 & 7) << 4))), "r"(frag0_bf16[0]), "r"(frag0_bf16[1]), "r"(frag0_bf16[2]), "r"(frag0_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_panel_addr + (unsigned int)(state_row_1 * 128 + 16 ^ (state_row_1 * 128 + 16 >> 7 & 7) << 4))), "r"(frag0_bf16[4]), "r"(frag0_bf16[5]), "r"(frag0_bf16[6]), "r"(frag0_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_panel_addr + (unsigned int)(state_row_1 * 128 + 32 ^ (state_row_1 * 128 + 32 >> 7 & 7) << 4))), "r"(frag0_bf16[8]), "r"(frag0_bf16[9]), "r"(frag0_bf16[10]), "r"(frag0_bf16[11]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_panel_addr + (unsigned int)(state_row_1 * 128 + 48 ^ (state_row_1 * 128 + 48 >> 7 & 7) << 4))), "r"(frag0_bf16[12]), "r"(frag0_bf16[13]), "r"(frag0_bf16[14]), "r"(frag0_bf16[15]) : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                if (epilogue_local_warp == 0) {
                    if (elect_sync()) {
                        if (is_last != 0) {
                            tma_store_3d(map_final_tma, 0, 0, final_outer_1, smem_panel_addr);
                        } else {
                            tma_store_3d(maps_tma, 0, 0, snap_outer_1 + p_1 * num_heads, smem_panel_addr);
                        }
                    }
                    asm volatile("cp.async.bulk.commit_group;");
                }
                if (epilogue_local_warp == 0) {
                    asm volatile("cp.async.bulk.wait_group.read 0;");
                }
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                uint32_t frag1_bf16[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(frag1[_lp*2 + 0], frag1[_lp*2+1 + 0]));
                    frag1_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_panel_addr + (unsigned int)(state_row_1 * 128 ^ (state_row_1 * 128 >> 7 & 7) << 4))), "r"(frag1_bf16[0]), "r"(frag1_bf16[1]), "r"(frag1_bf16[2]), "r"(frag1_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_panel_addr + (unsigned int)(state_row_1 * 128 + 16 ^ (state_row_1 * 128 + 16 >> 7 & 7) << 4))), "r"(frag1_bf16[4]), "r"(frag1_bf16[5]), "r"(frag1_bf16[6]), "r"(frag1_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_panel_addr + (unsigned int)(state_row_1 * 128 + 32 ^ (state_row_1 * 128 + 32 >> 7 & 7) << 4))), "r"(frag1_bf16[8]), "r"(frag1_bf16[9]), "r"(frag1_bf16[10]), "r"(frag1_bf16[11]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_panel_addr + (unsigned int)(state_row_1 * 128 + 48 ^ (state_row_1 * 128 + 48 >> 7 & 7) << 4))), "r"(frag1_bf16[12]), "r"(frag1_bf16[13]), "r"(frag1_bf16[14]), "r"(frag1_bf16[15]) : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                if (epilogue_local_warp == 0) {
                    if (elect_sync()) {
                        if (is_last != 0) {
                            tma_store_3d(map_final_tma, 32, 0, final_outer_1, smem_panel_addr);
                        } else {
                            tma_store_3d(maps_tma, 32, 0, snap_outer_1 + p_1 * num_heads, smem_panel_addr);
                        }
                    }
                    asm volatile("cp.async.bulk.commit_group;");
                }
                if (epilogue_local_warp == 0) {
                    asm volatile("cp.async.bulk.wait_group.read 0;");
                }
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                uint32_t frag2_bf16[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(frag2[_lp*2 + 0], frag2[_lp*2+1 + 0]));
                    frag2_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_panel_addr + (unsigned int)(state_row_1 * 128 ^ (state_row_1 * 128 >> 7 & 7) << 4))), "r"(frag2_bf16[0]), "r"(frag2_bf16[1]), "r"(frag2_bf16[2]), "r"(frag2_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_panel_addr + (unsigned int)(state_row_1 * 128 + 16 ^ (state_row_1 * 128 + 16 >> 7 & 7) << 4))), "r"(frag2_bf16[4]), "r"(frag2_bf16[5]), "r"(frag2_bf16[6]), "r"(frag2_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_panel_addr + (unsigned int)(state_row_1 * 128 + 32 ^ (state_row_1 * 128 + 32 >> 7 & 7) << 4))), "r"(frag2_bf16[8]), "r"(frag2_bf16[9]), "r"(frag2_bf16[10]), "r"(frag2_bf16[11]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_panel_addr + (unsigned int)(state_row_1 * 128 + 48 ^ (state_row_1 * 128 + 48 >> 7 & 7) << 4))), "r"(frag2_bf16[12]), "r"(frag2_bf16[13]), "r"(frag2_bf16[14]), "r"(frag2_bf16[15]) : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                if (epilogue_local_warp == 0) {
                    if (elect_sync()) {
                        if (is_last != 0) {
                            tma_store_3d(map_final_tma, 64, 0, final_outer_1, smem_panel_addr);
                        } else {
                            tma_store_3d(maps_tma, 64, 0, snap_outer_1 + p_1 * num_heads, smem_panel_addr);
                        }
                    }
                    asm volatile("cp.async.bulk.commit_group;");
                }
                if (epilogue_local_warp == 0) {
                    asm volatile("cp.async.bulk.wait_group.read 0;");
                }
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                uint32_t frag3_bf16[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(frag3[_lp*2 + 0], frag3[_lp*2+1 + 0]));
                    frag3_bf16[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_panel_addr + (unsigned int)(state_row_1 * 128 ^ (state_row_1 * 128 >> 7 & 7) << 4))), "r"(frag3_bf16[0]), "r"(frag3_bf16[1]), "r"(frag3_bf16[2]), "r"(frag3_bf16[3]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_panel_addr + (unsigned int)(state_row_1 * 128 + 16 ^ (state_row_1 * 128 + 16 >> 7 & 7) << 4))), "r"(frag3_bf16[4]), "r"(frag3_bf16[5]), "r"(frag3_bf16[6]), "r"(frag3_bf16[7]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_panel_addr + (unsigned int)(state_row_1 * 128 + 32 ^ (state_row_1 * 128 + 32 >> 7 & 7) << 4))), "r"(frag3_bf16[8]), "r"(frag3_bf16[9]), "r"(frag3_bf16[10]), "r"(frag3_bf16[11]) : "memory");
                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_panel_addr + (unsigned int)(state_row_1 * 128 + 48 ^ (state_row_1 * 128 + 48 >> 7 & 7) << 4))), "r"(frag3_bf16[12]), "r"(frag3_bf16[13]), "r"(frag3_bf16[14]), "r"(frag3_bf16[15]) : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                if (epilogue_local_warp == 0) {
                    if (elect_sync()) {
                        if (is_last != 0) {
                            tma_store_3d(map_final_tma, 96, 0, final_outer_1, smem_panel_addr);
                        } else {
                            tma_store_3d(maps_tma, 96, 0, snap_outer_1 + p_1 * num_heads, smem_panel_addr);
                        }
                    }
                    asm volatile("cp.async.bulk.commit_group;");
                }
            }
            if (epilogue_local_warp == 0) {
                asm volatile("cp.async.bulk.wait_group 0;");
                if (elect_sync()) {
                    mbarrier_arrive(tmem_dealloc_ready_addr);
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 8) {
        { // mma_main
            int item_base_2 = blockIdx.x * 4;
            int snap_outer_2 = items[item_base_2];
            int num_blocks_2 = items[item_base_2 + 1];
            int final_outer_2 = items[item_base_2 + 2];
            unsigned int mma_stage = 0;
            unsigned int _phase_r_full_1 = 0;
            unsigned int _phase_inp_ready_0 = 0;
            #pragma unroll 1
            for (int p_2 = 0; p_2 < num_blocks_2; p_2++) {
                mbarrier_wait(r_full_addr + (mma_stage) * 8, _phase_r_full_1);
                mbarrier_wait(inp_ready_addr, _phase_inp_ready_0);
                _phase_inp_ready_0 ^= 1;
                int parity = p_2 % 2;
                if (parity == 0) {
                    int _mma_b_lo_0 = make_warp_uniform(((((smem_r_addr) >> 4) & 0x3FFF) | 0x4000000) + (mma_stage) * 2048);
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
                    "mov.b32 id, 136381584;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_m_alt), "r"(_mma_b_lo_0), "r"(tmem_tmem_inp), "r"(1));
                } else {
                    int _mma_b_lo_1 = make_warp_uniform(((((smem_r_addr) >> 4) & 0x3FFF) | 0x4000000) + (mma_stage) * 2048);
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
                    "mov.b32 id, 136381584;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_m), "r"(_mma_b_lo_1), "r"(tmem_tmem_inp), "r"(1));
                }
                elect_commit(r_free_addr + (mma_stage) * 8);
                int m_ready_stage = p_2 & 1;
                elect_commit(m_ready_addr + (m_ready_stage) * 8);
                mma_stage += 1;
                if (mma_stage == 3) { mma_stage = 0; _phase_r_full_1 ^= 1; }
            }
            unsigned int _phase_tmem_dealloc_ready_0 = 0;
            mbarrier_wait(tmem_dealloc_ready_addr, _phase_tmem_dealloc_ready_0);
            _phase_tmem_dealloc_ready_0 ^= 1;
            int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(512));
        }
    }
    // ---- Role: load ----
    if (warp == 9) {
        { // load_main
            int item_base_3 = blockIdx.x * 4;
            int snap_outer_3 = items[item_base_3];
            int num_blocks_3 = items[item_base_3 + 1];
            int final_outer_3 = items[item_base_3 + 2];
            unsigned int _phase_r_free = 1;
            if (elect_sync()) {
                unsigned int load_stage = 0;
                #pragma unroll 1
                for (int p_3 = 0; p_3 < num_blocks_3; p_3++) {
                    mbarrier_wait(r_free_addr + (load_stage) * 8, _phase_r_free);
                    mbarrier_arrive_expect_tx(r_full_addr + (load_stage) * 8, 33280);
                    tma_4d_gmem2smem(smem_r_addr + load_stage * 32768, pair_tma, 0, (snap_outer_3 + p_3 * num_heads) * 128, 0, 0, r_full_addr + (load_stage) * 8);
                    cp_async_bulk_gmem2smem(smem_d_addr + (unsigned int)((int)load_stage * 256 * 4), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(dpair) + ((unsigned long long)(long long)((snap_outer_3 + p_3 * num_heads) * 128) * (unsigned long long)4)), 512, r_full_addr + (load_stage) * 8);
                    load_stage += 1;
                    if (load_stage == 3) { load_stage = 0; _phase_r_free ^= 1; }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
