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
#define TMEM_NCOLS 512
#define TMEM_ACC_OFFSET 0
#define NUM_AB_PIPE_STAGES 2
#define NUM_ACC_PIPE_STAGES 2
#define NUM_SCALE_PIPE_STAGES 3
#define NUM_TILE_PIPE_STAGES 16
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 16384
#define SMEM_SMEM_B_OFF 33792
#define SMEM_SMEM_B_STAGE_BYTES 32768
#define SMEM_SMEM_B_STRIDE 32768
#define SMEM_EPI_STAGING_OFF 99328
#define SMEM_EPI_STAGING_STAGE_BYTES 131072
#define SMEM_EPI_STAGING_STRIDE 131072
#define SMEM_SMEM_ASCALE_OFF 230400
#define SMEM_SMEM_ASCALE_STAGE_BYTES 1536
#define SMEM_SMEM_ASCALE_STRIDE 1536
#define SMEM_SMEM_BSCALE_OFF 231936
#define SMEM_SMEM_BSCALE_STAGE_BYTES 48
#define SMEM_SMEM_BSCALE_STRIDE 48
#define SMEM_TILE_RING_OFF 231984
#define SMEM_TILE_RING_STAGE_BYTES 256
#define SMEM_TILE_RING_STRIDE 256
#define SMEM_TOTAL 232320
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


__device__ __forceinline__ uint32_t mbarrier_test_wait(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.test_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%1], %2;\n\t"
        "selp.u32 %0, 1, 0, P1;\n\t"
        "}\n"
        : "=r"(token)
        : "r"(mbar_addr), "r"(phase) : "memory");
    return token;
}

__device__ __forceinline__ uint32_t mbarrier_test_wait_cluster(int mbar_addr, int phase) {
    uint32_t token;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "mbarrier.test_wait.parity.acquire.cluster.shared::cta.b64"
        " P1, [%1], %2;\n\t"
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


__device__ __forceinline__ void tma_store_2d(
    const void *tmap, int x, int y, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2}], [%3];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(smem_addr) : "memory");
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

__global__ __launch_bounds__(384) void
kernel_cake_grouped_fp8_gemm_d7f8d5132ca62707fae1(CakeTensorMap const* A, CakeTensorMap const* B, CakeTensorMap const* C_tma, __nv_bfloat16* __restrict__ C, float* __restrict__ a_scale, float* __restrict__ b_scale, int* __restrict__ m_indices, int M, int N, int K, int G)
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
    #define mma_done_addr (mbar_base + 16)
    #define acc_free_addr (mbar_base + 32)
    #define scale_full_addr (mbar_base + 48)
    #define scale_free_addr (mbar_base + 72)
    #define tile_full_addr (mbar_base + 96)
    #define tile_free_addr (mbar_base + 224)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(A)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(B)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(C_tma)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int smem_b_addr = smem + 33792;
    __nv_bfloat16* epi_staging = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
    const int epi_staging_addr = smem + 99328;
    float* smem_ascale = reinterpret_cast<float*>(smem_raw + 230400);
    const int smem_ascale_addr = smem + 230400;
    float* smem_bscale = reinterpret_cast<float*>(smem_raw + 231936);
    const int smem_bscale_addr = smem + 231936;
    int* tile_ring = reinterpret_cast<int*>(smem_raw + 231984);
    const int tile_ring_addr = smem + 231984;
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 352);
    int taddr;
    int tmem_acc;

    // Mbarrier init (7 pipeline groups, 0 ordered-sequence groups, 44 barriers)
    // Mbarriers at smem_raw[0..352)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'ab_pipe' ---
            // ab_full: 2 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            // mma_done: 2 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // --- pipeline 'acc_pipe' ---
            // acc_free: 2 barriers, init_count=8
            mbarrier_init(smem + 32, 8);
            mbarrier_init(smem + 40, 8);
            // --- pipeline 'scale_pipe' ---
            // scale_full: 3 barriers, init_count=32
            mbarrier_init(smem + 48, 32);
            mbarrier_init(smem + 56, 32);
            mbarrier_init(smem + 64, 32);
            // scale_free: 3 barriers, init_count=256
            mbarrier_init(smem + 72, 256);
            mbarrier_init(smem + 80, 256);
            mbarrier_init(smem + 88, 256);
            // --- pipeline 'tile_pipe' ---
            // tile_full: 16 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
            // tile_free: 16 barriers, init_count=259
            mbarrier_init(smem + 224, 259);
            mbarrier_init(smem + 232, 259);
            mbarrier_init(smem + 240, 259);
            mbarrier_init(smem + 248, 259);
            mbarrier_init(smem + 256, 259);
            mbarrier_init(smem + 264, 259);
            mbarrier_init(smem + 272, 259);
            mbarrier_init(smem + 280, 259);
            mbarrier_init(smem + 288, 259);
            mbarrier_init(smem + 296, 259);
            mbarrier_init(smem + 304, 259);
            mbarrier_init(smem + 312, 259);
            mbarrier_init(smem + 320, 259);
            mbarrier_init(smem + 328, 259);
            mbarrier_init(smem + 336, 259);
            mbarrier_init(smem + 344, 259);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 64;");
    }

    // ---- Role: acc_epi ----
    if (warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 216;");
        { // acc_epi_main
            int max_nonempty_groups = G;
            if (max_nonempty_groups > M) {
                max_nonempty_groups = M;
            }
            int max_segments = ((M + 128 - 1) / 128 + max_nonempty_groups - 1) * ((N + 256 - 1) / 256);
            if (warp == 0) {
                int _tmem_hold_0 = smem + 352;
                asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold_0), "r"(512) : "memory");
            }
            asm volatile("barrier.sync 9, 288;" ::: "memory");
            asm volatile("tcgen05.fence::after_thread_sync;");
            taddr = tmem_addr_storage[0];
            tmem_acc = taddr;
            if (warp == 0) {
                asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
            }
            unsigned int ring_stage = 0;
            unsigned int s_stage = 0;
            unsigned int acc_stage = 0;
            unsigned int acc_done_phase = 0;
            unsigned int epi_publish_stage = 0;
            const int epi_wg = warp / 4;
            const int epi_warp = warp % 4;
            unsigned int _phase_tile_full = 0;
            unsigned int _phase_scale_full = 0;
            #pragma unroll 1
            for (int ring_iter = 0; ring_iter < max_segments + 1; ring_iter++) {
                mbarrier_wait(tile_full_addr + (ring_stage) * 8, _phase_tile_full);
                int valid = tile_ring[ring_stage * 4 + 3];
                if (valid == 0) {
                    mbarrier_arrive(tile_free_addr + (ring_stage) * 8);
                    ring_stage += 1;
                    if (ring_stage == 16) { ring_stage = 0; _phase_tile_full ^= 1; }
                    break;
                }
                int m_tile = tile_ring[ring_stage * 4];
                int n_tile = tile_ring[ring_stage * 4 + 1];
                int run_begin = valid >> 8;
                int run_end = valid & 255;
                int tile_rows = 128;
                if (m_tile * 128 + tile_rows > M) {
                    tile_rows = M - m_tile * 128;
                }
                int homogeneous = 0;
                if (valid == tile_rows) {
                    homogeneous = 1;
                }
                int k_blocks = 1;
                #pragma unroll 1
                for (int iter_k = 0; iter_k < k_blocks; iter_k++) {
                    mbarrier_wait(scale_full_addr + (s_stage) * 8, _phase_scale_full);
                    mbarrier_wait(mma_done_addr + (acc_stage) * 8, acc_done_phase);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int row_base = epi_warp * 32;
                    int row = row_base + lane;
                    float a_s = smem_ascale[s_stage * 128 + (unsigned int)row];
                    unsigned int epi_stage = epi_publish_stage;
                    if (homogeneous != 0) {
                        int h_first_addr = taddr + (unsigned int)(row_base << 16) + acc_stage * 256 + (unsigned int)(epi_wg * 128);
                        float _tmem_load_0[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                            : "r"(h_first_addr));
                        if (warp == 0) {
                            if (elect_sync()) {
                                asm volatile("cp.async.bulk.wait_group.read 2;");
                            }
                        }
                        asm volatile("barrier.sync 8, 256;" ::: "memory");
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        float h_first_b_s = smem_bscale[s_stage * 4 + (unsigned int)epi_wg];
                        float h_first_combined = a_s * h_first_b_s;
                        const float2 _scale2_1 = {h_first_combined, h_first_combined};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            reinterpret_cast<float2*>(_tmem_load_0)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>(_tmem_load_0)[_ls], _scale2_1);
                        uint32_t _tmem_load_0_bf16[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                            _tmem_load_0_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        int h_first_c_stage_row = ((unsigned int)(epi_wg * 4 * 2) + epi_stage) * 128 + (unsigned int)row;
                        __nv_bfloat16* _sv_ptr_0 = reinterpret_cast<__nv_bfloat16*>(epi_staging + (h_first_c_stage_row * 32));
                        reinterpret_cast<int4*>(_sv_ptr_0 + 0)[0] = reinterpret_cast<int4*>(_tmem_load_0_bf16)[0];
                        reinterpret_cast<int4*>(_sv_ptr_0 + 8)[0] = reinterpret_cast<int4*>(_tmem_load_0_bf16 + 4)[0];
                        reinterpret_cast<int4*>(_sv_ptr_0 + 16)[0] = reinterpret_cast<int4*>(_tmem_load_0_bf16 + 8)[0];
                        reinterpret_cast<int4*>(_sv_ptr_0 + 24)[0] = reinterpret_cast<int4*>(_tmem_load_0_bf16 + 12)[0];
                        #pragma unroll
                        for (int h_sub = 1; h_sub < 4; h_sub++) {
                            int h_tmem_addr = taddr + (unsigned int)(row_base << 16) + acc_stage * 256 + (unsigned int)(epi_wg * 128) + (unsigned int)(h_sub * 32);
                            float _tmem_load_1[32];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31])
                                : "r"(h_tmem_addr));
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            float h_b_s = smem_bscale[s_stage * 4 + (unsigned int)epi_wg + (unsigned int)(h_sub / 4)];
                            float h_combined = a_s * h_b_s;
                            const float2 _scale2_2 = {h_combined, h_combined};
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++)
                                reinterpret_cast<float2*>(_tmem_load_1)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>(_tmem_load_1)[_ls], _scale2_2);
                            uint32_t _tmem_load_1_bf16[16];
                            #pragma unroll
                            for (int _lp = 0; _lp < 16; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_1[_lp*2 + 0], _tmem_load_1[_lp*2+1 + 0]));
                                _tmem_load_1_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            int h_c_stage_row = ((unsigned int)((epi_wg * 4 + h_sub) * 2) + epi_stage) * 128 + (unsigned int)row;
                            __nv_bfloat16* _sv_ptr_1 = reinterpret_cast<__nv_bfloat16*>(epi_staging + (h_c_stage_row * 32));
                            reinterpret_cast<int4*>(_sv_ptr_1 + 0)[0] = reinterpret_cast<int4*>(_tmem_load_1_bf16)[0];
                            reinterpret_cast<int4*>(_sv_ptr_1 + 8)[0] = reinterpret_cast<int4*>(_tmem_load_1_bf16 + 4)[0];
                            reinterpret_cast<int4*>(_sv_ptr_1 + 16)[0] = reinterpret_cast<int4*>(_tmem_load_1_bf16 + 8)[0];
                            reinterpret_cast<int4*>(_sv_ptr_1 + 24)[0] = reinterpret_cast<int4*>(_tmem_load_1_bf16 + 12)[0];
                        }
                    } else {
                        #pragma unroll
                        for (int sub = 0; sub < 4; sub++) {
                            int tmem_addr = taddr + (unsigned int)(row_base << 16) + acc_stage * 256 + (unsigned int)(epi_wg * 128) + (unsigned int)(sub * 32);
                            float _tmem_load_2[32];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(_tmem_load_2[0]), "=f"(_tmem_load_2[1]), "=f"(_tmem_load_2[2]), "=f"(_tmem_load_2[3]), "=f"(_tmem_load_2[4]), "=f"(_tmem_load_2[5]), "=f"(_tmem_load_2[6]), "=f"(_tmem_load_2[7]), "=f"(_tmem_load_2[8]), "=f"(_tmem_load_2[9]), "=f"(_tmem_load_2[10]), "=f"(_tmem_load_2[11]), "=f"(_tmem_load_2[12]), "=f"(_tmem_load_2[13]), "=f"(_tmem_load_2[14]), "=f"(_tmem_load_2[15]), "=f"(_tmem_load_2[16]), "=f"(_tmem_load_2[17]), "=f"(_tmem_load_2[18]), "=f"(_tmem_load_2[19]), "=f"(_tmem_load_2[20]), "=f"(_tmem_load_2[21]), "=f"(_tmem_load_2[22]), "=f"(_tmem_load_2[23]), "=f"(_tmem_load_2[24]), "=f"(_tmem_load_2[25]), "=f"(_tmem_load_2[26]), "=f"(_tmem_load_2[27]), "=f"(_tmem_load_2[28]), "=f"(_tmem_load_2[29]), "=f"(_tmem_load_2[30]), "=f"(_tmem_load_2[31])
                                : "r"(tmem_addr));
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            float b_s = smem_bscale[s_stage * 4 + (unsigned int)epi_wg + (unsigned int)(sub / 4)];
                            float combined = a_s * b_s;
                            const float2 _scale2_3 = {combined, combined};
                            #pragma unroll
                            for (int _ls = 0; _ls < 16; _ls++)
                                reinterpret_cast<float2*>(_tmem_load_2)[_ls] = mul_f32x2_rn_noftz(reinterpret_cast<float2*>(_tmem_load_2)[_ls], _scale2_3);
                            if (row >= run_begin) {
                                if (row < run_end) {
                                    long long output_row = m_tile * 128 + row;
                                    #pragma unroll
                                    for (int vec = 0; vec < 32; vec += 8) {
                                        int out_col = n_tile * 256 + epi_wg * 128 + sub * 32 + vec;
                                        if (out_col < N) {
                                            {
                                                __nv_bfloat162 _pk[4];
                                                _pk[0] = __floats2bfloat162_rn(_tmem_load_2[vec + 0], _tmem_load_2[vec + 1]);
                                                _pk[1] = __floats2bfloat162_rn(_tmem_load_2[vec + 2], _tmem_load_2[vec + 3]);
                                                _pk[2] = __floats2bfloat162_rn(_tmem_load_2[vec + 4], _tmem_load_2[vec + 5]);
                                                _pk[3] = __floats2bfloat162_rn(_tmem_load_2[vec + 6], _tmem_load_2[vec + 7]);
                                                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(C + (output_row * (long long)N + (long long)out_col)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    if (homogeneous != 0) {
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 8, 256;" ::: "memory");
                        if (warp == 0) {
                            if (elect_sync()) {
                                int staging_offset = epi_stage * 8192;
                                tma_store_2d(C_tma, n_tile * 256, m_tile * 128, epi_staging_addr + (unsigned int)staging_offset);
                                tma_store_2d(C_tma, n_tile * 256 + 32, m_tile * 128, epi_staging_addr + 16384 + (unsigned int)staging_offset);
                                tma_store_2d(C_tma, n_tile * 256 + 64, m_tile * 128, epi_staging_addr + 32768 + (unsigned int)staging_offset);
                                tma_store_2d(C_tma, n_tile * 256 + 96, m_tile * 128, epi_staging_addr + 49152 + (unsigned int)staging_offset);
                                asm volatile("cp.async.bulk.commit_group;");
                                tma_store_2d(C_tma, n_tile * 256 + 128, m_tile * 128, epi_staging_addr + 65536 + (unsigned int)staging_offset);
                                tma_store_2d(C_tma, n_tile * 256 + 160, m_tile * 128, epi_staging_addr + 81920 + (unsigned int)staging_offset);
                                tma_store_2d(C_tma, n_tile * 256 + 192, m_tile * 128, epi_staging_addr + 98304 + (unsigned int)staging_offset);
                                tma_store_2d(C_tma, n_tile * 256 + 224, m_tile * 128, epi_staging_addr + 114688 + (unsigned int)staging_offset);
                                asm volatile("cp.async.bulk.commit_group;");
                            }
                        }
                        epi_publish_stage = epi_publish_stage ^ 1;
                    }
                    mbarrier_arrive(scale_free_addr + (s_stage) * 8);
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    if (elect_sync()) {
                        mbarrier_arrive(acc_free_addr + (acc_stage) * 8);
                    }
                    acc_stage += 1;
                    if (acc_stage == 2) { acc_stage = 0; acc_done_phase ^= 1; }
                    s_stage += 1;
                    if (s_stage == 3) { s_stage = 0; _phase_scale_full ^= 1; }
                }
                mbarrier_arrive(tile_free_addr + (ring_stage) * 8);
                ring_stage += 1;
                if (ring_stage == 16) { ring_stage = 0; _phase_tile_full ^= 1; }
            }
            if (warp == 0) {
                if (elect_sync()) {
                    asm volatile("cp.async.bulk.wait_group.read 0;");
                }
            }
            asm volatile("barrier.sync 8, 256;" ::: "memory");
            asm volatile("barrier.sync 9, 288;" ::: "memory");
            asm volatile("tcgen05.fence::after_thread_sync;");
            if (warp == 0) {
                int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(512));
            }
        }
    }
    // ---- Role: mma_role ----
    if (warp == 8) {
        { // mma_role_main
            int max_nonempty_groups_1 = G;
            if (max_nonempty_groups_1 > M) {
                max_nonempty_groups_1 = M;
            }
            int max_segments_1 = ((M + 128 - 1) / 128 + max_nonempty_groups_1 - 1) * ((N + 256 - 1) / 256);
            asm volatile("barrier.sync 9, 288;" ::: "memory");
            asm volatile("tcgen05.fence::after_thread_sync;");
            taddr = tmem_addr_storage[0];
            tmem_acc = taddr;
            unsigned int ring_stage_1 = 0;
            unsigned int ab_stage = 0;
            unsigned int acc_stage_1 = 0;
            unsigned int _phase_tile_full_1 = 0;
            unsigned int _phase_acc_free = 1;
            unsigned int _phase_ab_full = 0;
            #pragma unroll 1
            for (int ring_iter_1 = 0; ring_iter_1 < max_segments_1 + 1; ring_iter_1++) {
                mbarrier_wait(tile_full_addr + (ring_stage_1) * 8, _phase_tile_full_1);
                int valid_1 = tile_ring[ring_stage_1 * 4 + 3];
                if (valid_1 == 0) {
                    if (elect_sync()) {
                        mbarrier_arrive(tile_free_addr + (ring_stage_1) * 8);
                    }
                    ring_stage_1 += 1;
                    if (ring_stage_1 == 16) { ring_stage_1 = 0; _phase_tile_full_1 ^= 1; }
                    break;
                }
                int k_blocks_1 = 1;
                #pragma unroll 1
                for (int iter_k_1 = 0; iter_k_1 < k_blocks_1; iter_k_1++) {
                    mbarrier_wait(acc_free_addr + (acc_stage_1) * 8, _phase_acc_free);
                    mbarrier_wait(ab_full_addr + (ab_stage) * 8, _phase_ab_full);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int _mma_a_lo_0 = make_warp_uniform((((smem_a_addr) >> 4) & 0x3FFF) + (ab_stage) * 1024);
                    int _mma_b_lo_0 = make_warp_uniform((((smem_b_addr) >> 4) & 0x3FFF) + (ab_stage) * 2048);
                    {
                        uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
                        uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_acc + (acc_stage_1 * 256)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 138412048, 0);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_acc + (acc_stage_1 * 256)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 138412048, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_acc + (acc_stage_1 * 256)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 138412048, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 2U);
                        if (elect_sync()) {
                            tcgen05_mma_f8f6f4((tmem_acc + (acc_stage_1 * 256)), _mma_ss_a_desc_0, _mma_ss_b_desc_0, 138412048, 1);
                        }
                    }
                    elect_commit(mma_done_addr + (acc_stage_1) * 8);
                    ab_stage += 1;
                    if (ab_stage == 2) { ab_stage = 0; _phase_ab_full ^= 1; }
                    acc_stage_1 += 1;
                    if (acc_stage_1 == 2) { acc_stage_1 = 0; _phase_acc_free ^= 1; }
                }
                if (elect_sync()) {
                    mbarrier_arrive(tile_free_addr + (ring_stage_1) * 8);
                }
                ring_stage_1 += 1;
                if (ring_stage_1 == 16) { ring_stage_1 = 0; _phase_tile_full_1 ^= 1; }
            }
            asm volatile("barrier.sync 9, 288;" ::: "memory");
        }
    }
    // ---- Role: tma_role ----
    if (warp == 9) {
        { // tma_role_main
            int max_nonempty_groups_2 = G;
            if (max_nonempty_groups_2 > M) {
                max_nonempty_groups_2 = M;
            }
            int max_segments_2 = ((M + 128 - 1) / 128 + max_nonempty_groups_2 - 1) * ((N + 256 - 1) / 256);
            unsigned int ring_stage_2 = 0;
            unsigned int ab_stage_1 = 0;
            unsigned int ab_done_phase = 1;
            unsigned int _phase_tile_full_2 = 0;
            #pragma unroll 1
            for (int ring_iter_2 = 0; ring_iter_2 < max_segments_2 + 1; ring_iter_2++) {
                mbarrier_wait(tile_full_addr + (ring_stage_2) * 8, _phase_tile_full_2);
                int valid_2 = tile_ring[ring_stage_2 * 4 + 3];
                if (valid_2 == 0) {
                    if (elect_sync()) {
                        mbarrier_arrive(tile_free_addr + (ring_stage_2) * 8);
                    }
                    ring_stage_2 += 1;
                    if (ring_stage_2 == 16) { ring_stage_2 = 0; _phase_tile_full_2 ^= 1; }
                    break;
                }
                int m_tile_1 = tile_ring[ring_stage_2 * 4];
                int n_tile_1 = tile_ring[ring_stage_2 * 4 + 1];
                int group = tile_ring[ring_stage_2 * 4 + 2];
                int k_blocks_2 = 1;
                #pragma unroll 1
                for (int iter_k_2 = 0; iter_k_2 < k_blocks_2; iter_k_2++) {
                    mbarrier_wait(mma_done_addr + (ab_stage_1) * 8, ab_done_phase);
                    if (elect_sync()) {
                        tma_3d_gmem2smem(smem_a_addr + ab_stage_1 * 16384, A, 0, m_tile_1 * 128, iter_k_2, ab_full_addr + (ab_stage_1) * 8);
                        tma_4d_gmem2smem(smem_b_addr + ab_stage_1 * 32768, B, 0, n_tile_1 * 256, iter_k_2, group, ab_full_addr + (ab_stage_1) * 8);
                        mbarrier_arrive_expect_tx(ab_full_addr + (ab_stage_1) * 8, 49152);
                    }
                    ab_stage_1 += 1;
                    if (ab_stage_1 == 2) { ab_stage_1 = 0; ab_done_phase ^= 1; }
                }
                if (elect_sync()) {
                    mbarrier_arrive(tile_free_addr + (ring_stage_2) * 8);
                }
                ring_stage_2 += 1;
                if (ring_stage_2 == 16) { ring_stage_2 = 0; _phase_tile_full_2 ^= 1; }
            }
        }
    }
    // ---- Role: scale_role ----
    if (warp == 10) {
        { // scale_role_main
            int max_nonempty_groups_3 = G;
            if (max_nonempty_groups_3 > M) {
                max_nonempty_groups_3 = M;
            }
            int max_segments_3 = ((M + 128 - 1) / 128 + max_nonempty_groups_3 - 1) * ((N + 256 - 1) / 256);
            unsigned int ring_stage_3 = 0;
            unsigned int s_stage_1 = 0;
            unsigned int _phase_tile_full_3 = 0;
            unsigned int _phase_scale_free = 1;
            #pragma unroll 1
            for (int ring_iter_3 = 0; ring_iter_3 < max_segments_3 + 1; ring_iter_3++) {
                mbarrier_wait(tile_full_addr + (ring_stage_3) * 8, _phase_tile_full_3);
                int valid_3 = tile_ring[ring_stage_3 * 4 + 3];
                if (valid_3 == 0) {
                    if (elect_sync()) {
                        mbarrier_arrive(tile_free_addr + (ring_stage_3) * 8);
                    }
                    ring_stage_3 += 1;
                    if (ring_stage_3 == 16) { ring_stage_3 = 0; _phase_tile_full_3 ^= 1; }
                    break;
                }
                int m_tile_2 = tile_ring[ring_stage_3 * 4];
                int n_tile_2 = tile_ring[ring_stage_3 * 4 + 1];
                int group_1 = tile_ring[ring_stage_3 * 4 + 2];
                int k_blocks_3 = 1;
                int n_blocks = N / 128;
                #pragma unroll 1
                for (int iter_k_3 = 0; iter_k_3 < k_blocks_3; iter_k_3++) {
                    mbarrier_wait(scale_free_addr + (s_stage_1) * 8, _phase_scale_free);
                    int base = s_stage_1 * 128;
                    #pragma unroll
                    for (int chunk = 0; chunk < 4; chunk++) {
                        int row_1 = chunk * 32 + lane;
                        int g_row = m_tile_2 * 128 + row_1;
                        asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4, %2;"
                            :: "r"(smem_ascale_addr + (unsigned int)((base + row_1) * 4)), "l"(a_scale + (g_row * k_blocks_3 + iter_k_3)), "r"((g_row < M) ? 4 : 0));
                    }
                    if (elect_sync()) {
                        #pragma unroll
                        for (int b_half = 0; b_half < 2; b_half++) {
                            int scale_block = n_tile_2 * 2 + b_half;
                            asm volatile("cp.async.ca.shared::cta.global [%0], [%1], 4, %2;"
                                :: "r"(smem_bscale_addr + s_stage_1 * 16 + (unsigned int)(b_half * 4)), "l"(b_scale + ((group_1 * n_blocks + scale_block) * k_blocks_3 + iter_k_3)), "r"((scale_block < n_blocks) ? 4 : 0));
                        }
                    }
                    asm volatile(
                        "{\n\t"
                        "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [%0];\n\t"
                        "}"
                        :: "r"(scale_full_addr + (s_stage_1) * 8) : "memory");
                    s_stage_1 += 1;
                    if (s_stage_1 == 3) { s_stage_1 = 0; _phase_scale_free ^= 1; }
                }
                if (elect_sync()) {
                    mbarrier_arrive(tile_free_addr + (ring_stage_3) * 8);
                }
                ring_stage_3 += 1;
                if (ring_stage_3 == 16) { ring_stage_3 = 0; _phase_tile_full_3 ^= 1; }
            }
        }
    }
    // ---- Role: sched ----
    if (warp == 11) {
        { // sched_main
            unsigned int ring_stage_4 = 0;
            int m_tiles = (M + 128 - 1) / 128;
            int n_tiles = (N + 256 - 1) / 256;
            int total_tiles = m_tiles * n_tiles;
            int num_ctas = num_bids;
            int m_tile_3 = bid / n_tiles;
            int n_tile_3 = bid % n_tiles;
            int step_m = num_ctas / n_tiles;
            int step_n = num_ctas % n_tiles;
            unsigned int _phase_tile_free = 1;
            #pragma unroll 1
            for (int tile_id = bid; tile_id < total_tiles; tile_id += num_bids) {
                int tile_rows_1 = 128;
                if (m_tile_3 * 128 + tile_rows_1 > M) {
                    tile_rows_1 = M - m_tile_3 * 128;
                }
                int last_group = 0;
                if (lane == 0) {
                    last_group = m_indices[m_tile_3 * 128 + tile_rows_1 - 1];
                }
                int run_begin_1 = 0;
                #pragma unroll 1
                for (int segment_iter = 0; segment_iter < tile_rows_1; segment_iter++) {
                    if (run_begin_1 >= tile_rows_1) {
                        break;
                    }
                    mbarrier_wait(tile_free_addr + (ring_stage_4) * 8, _phase_tile_free);
                    int run_end_1 = tile_rows_1;
                    if (lane == 0) {
                        int group_2 = m_indices[m_tile_3 * 128 + run_begin_1];
                        if (group_2 != last_group) {
                            #pragma unroll 1
                            for (int probe = run_begin_1 + 1; probe < tile_rows_1; probe++) {
                                int next_group = m_indices[m_tile_3 * 128 + probe];
                                if (next_group != group_2) {
                                    run_end_1 = probe;
                                    break;
                                }
                            }
                        }
                        tile_ring[ring_stage_4 * 4] = m_tile_3;
                        tile_ring[ring_stage_4 * 4 + 1] = n_tile_3;
                        tile_ring[ring_stage_4 * 4 + 2] = group_2;
                        tile_ring[ring_stage_4 * 4 + 3] = run_begin_1 << 8 | run_end_1;
                    }
                    int _shfl_0 = __shfl_sync(0xFFFFFFFF, run_end_1, 0);
                    run_end_1 = _shfl_0;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync 10, 32;" ::: "memory");
                    if (elect_sync()) {
                        mbarrier_arrive(tile_full_addr + (ring_stage_4) * 8);
                    }
                    ring_stage_4 += 1;
                    if (ring_stage_4 == 16) { ring_stage_4 = 0; _phase_tile_free ^= 1; }
                    run_begin_1 = run_end_1;
                }
                m_tile_3 += step_m;
                n_tile_3 += step_n;
                if (n_tile_3 >= n_tiles) {
                    n_tile_3 -= n_tiles;
                    m_tile_3 += 1;
                }
            }
            mbarrier_wait(tile_free_addr + (ring_stage_4) * 8, _phase_tile_free);
            if (elect_sync()) {
                tile_ring[ring_stage_4 * 4] = 0;
                tile_ring[ring_stage_4 * 4 + 1] = 0;
                tile_ring[ring_stage_4 * 4 + 2] = 0;
                tile_ring[ring_stage_4 * 4 + 3] = 0;
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            asm volatile("barrier.sync 10, 32;" ::: "memory");
            if (elect_sync()) {
                mbarrier_arrive(tile_full_addr + (ring_stage_4) * 8);
            }
        }
    }

    // Cleanup
}

} // extern "C"
