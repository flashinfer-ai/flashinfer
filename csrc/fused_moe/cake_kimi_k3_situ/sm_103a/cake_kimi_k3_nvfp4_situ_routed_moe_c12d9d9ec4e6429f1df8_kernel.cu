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
#define TMEM_NCOLS 384
#define TMEM_ACCUM_OFFSET 0
#define TMEM_SFA_OFFSET 256
#define TMEM_SFB_OFFSET 320
#define NUM_K_PIPE_STAGES 4
#define NUM_MMA_PIPE_STAGES 2
#define NUM_WORK_PIPE_STAGES 3
#define NUM_THROTTLE_PIPE_STAGES 3
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 16384
#define SMEM_SMEM_B_OFF 66560
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 16384
#define SMEM_SMEM_A_MMA0_OFF 1024
#define SMEM_SMEM_A_MMA0_STAGE_BYTES 4096
#define SMEM_SMEM_A_MMA0_STRIDE 16384
#define SMEM_SMEM_A_MMA1_OFF 1056
#define SMEM_SMEM_A_MMA1_STAGE_BYTES 4096
#define SMEM_SMEM_A_MMA1_STRIDE 16384
#define SMEM_SMEM_A_MMA2_OFF 1088
#define SMEM_SMEM_A_MMA2_STAGE_BYTES 4096
#define SMEM_SMEM_A_MMA2_STRIDE 16384
#define SMEM_SMEM_A_MMA3_OFF 1120
#define SMEM_SMEM_A_MMA3_STAGE_BYTES 4096
#define SMEM_SMEM_A_MMA3_STRIDE 16384
#define SMEM_SMEM_A_MMA4_OFF 1024
#define SMEM_SMEM_A_MMA4_STAGE_BYTES 4096
#define SMEM_SMEM_A_MMA4_STRIDE 16384
#define SMEM_SMEM_A_MMA5_OFF 1024
#define SMEM_SMEM_A_MMA5_STAGE_BYTES 4096
#define SMEM_SMEM_A_MMA5_STRIDE 16384
#define SMEM_SMEM_A_MMA6_OFF 1024
#define SMEM_SMEM_A_MMA6_STAGE_BYTES 4096
#define SMEM_SMEM_A_MMA6_STRIDE 16384
#define SMEM_SMEM_A_MMA7_OFF 1024
#define SMEM_SMEM_A_MMA7_STAGE_BYTES 4096
#define SMEM_SMEM_A_MMA7_STRIDE 16384
#define SMEM_SMEM_B_MMA0_OFF 66560
#define SMEM_SMEM_B_MMA0_STAGE_BYTES 4096
#define SMEM_SMEM_B_MMA0_STRIDE 16384
#define SMEM_SMEM_B_MMA1_OFF 66592
#define SMEM_SMEM_B_MMA1_STAGE_BYTES 4096
#define SMEM_SMEM_B_MMA1_STRIDE 16384
#define SMEM_SMEM_B_MMA2_OFF 66624
#define SMEM_SMEM_B_MMA2_STAGE_BYTES 4096
#define SMEM_SMEM_B_MMA2_STRIDE 16384
#define SMEM_SMEM_B_MMA3_OFF 66656
#define SMEM_SMEM_B_MMA3_STAGE_BYTES 4096
#define SMEM_SMEM_B_MMA3_STRIDE 16384
#define SMEM_SMEM_B_MMA4_OFF 66560
#define SMEM_SMEM_B_MMA4_STAGE_BYTES 4096
#define SMEM_SMEM_B_MMA4_STRIDE 16384
#define SMEM_SMEM_B_MMA5_OFF 66560
#define SMEM_SMEM_B_MMA5_STAGE_BYTES 4096
#define SMEM_SMEM_B_MMA5_STRIDE 16384
#define SMEM_SMEM_B_MMA6_OFF 66560
#define SMEM_SMEM_B_MMA6_STAGE_BYTES 4096
#define SMEM_SMEM_B_MMA6_STRIDE 16384
#define SMEM_SMEM_B_MMA7_OFF 66560
#define SMEM_SMEM_B_MMA7_STAGE_BYTES 4096
#define SMEM_SMEM_B_MMA7_STRIDE 16384
#define SMEM_SMEM_A_MMA0_NEXT_OFF 1024
#define SMEM_SMEM_A_MMA0_NEXT_STAGE_BYTES 4096
#define SMEM_SMEM_A_MMA0_NEXT_STRIDE 16384
#define SMEM_SMEM_B_MMA0_NEXT_OFF 66560
#define SMEM_SMEM_B_MMA0_NEXT_STAGE_BYTES 4096
#define SMEM_SMEM_B_MMA0_NEXT_STRIDE 16384
#define SMEM_SMEM_A_MMA2_NEXT_OFF 1024
#define SMEM_SMEM_A_MMA2_NEXT_STAGE_BYTES 4096
#define SMEM_SMEM_A_MMA2_NEXT_STRIDE 16384
#define SMEM_SMEM_B_MMA2_NEXT_OFF 66560
#define SMEM_SMEM_B_MMA2_NEXT_STAGE_BYTES 4096
#define SMEM_SMEM_B_MMA2_NEXT_STRIDE 16384
#define SMEM_SMEM_A_MMA5_NEXT_OFF 1024
#define SMEM_SMEM_A_MMA5_NEXT_STAGE_BYTES 4096
#define SMEM_SMEM_A_MMA5_NEXT_STRIDE 16384
#define SMEM_SMEM_B_MMA5_NEXT_OFF 66560
#define SMEM_SMEM_B_MMA5_NEXT_STAGE_BYTES 4096
#define SMEM_SMEM_B_MMA5_NEXT_STRIDE 16384
#define SMEM_SMEM_SFA_OFF 167936
#define SMEM_SMEM_SFA_STAGE_BYTES 2048
#define SMEM_SMEM_SFA_STRIDE 2048
#define SMEM_SMEM_SFB_OFF 176128
#define SMEM_SMEM_SFB_STAGE_BYTES 2048
#define SMEM_SMEM_SFB_STRIDE 2048
#define SMEM_EPI_STAGING_OFF 132096
#define SMEM_EPI_STAGING_STAGE_BYTES 32768
#define SMEM_EPI_STAGING_STRIDE 32768
#define SMEM_EPI_STAGING_U64_OFF 132096
#define SMEM_EPI_STAGING_U64_STAGE_BYTES 32768
#define SMEM_EPI_STAGING_U64_STRIDE 32768
#define SMEM_WORK_RESPONSE_OFF 184320
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_TOTAL 184448
#define THREADS 384
#define BLOCK_M 128
#define BLOCK_N 128
#define BLOCK_K 256
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
        "@leader tcgen05.mma.cta_group::1.kind::mxf4nvf4 [%2], da, db, %3, p;\n\t"
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
    asm("mul.rn.ftz.f32x2 %0, %0, %1;"
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
    asm("mul.rn.ftz.f32x2 %0, %1, %2;"
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


__device__ __forceinline__ uint32_t make_warp_uniform(uint32_t val) {
    uint32_t result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1f, 0xffffffff;"
        : "=r"(result) : "r"(val));
    return result;
}

extern "C" {

__global__ __launch_bounds__(384, 1) void
kernel_cake_kimi_k3_nvfp4_situ_routed_moe_c12d9d9ec4e6429f1df8(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, const __grid_constant__ CUtensorMap C_tma, __nv_bfloat16* __restrict__ C, float* __restrict__ scale_c, int* __restrict__ tile_expert, int* __restrict__ tile_mn_limit, int M, int K, int grid_m, int grid_n, int K_tiles, int* __restrict__ total_tiles)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define a_full_addr (mbar_base + 0)
    #define b_full_addr (mbar_base + 32)
    #define sfa_full_addr (mbar_base + 64)
    #define sfb_full_addr (mbar_base + 96)
    #define sfa_smem_free_addr (mbar_base + 128)
    #define sfb_smem_free_addr (mbar_base + 160)
    #define tmem_sfab_full_addr (mbar_base + 192)
    #define k_done_addr (mbar_base + 224)
    #define mma_full_addr (mbar_base + 256)
    #define mma_free_addr (mbar_base + 272)
    #define work_full_addr (mbar_base + 288)
    #define work_empty_addr (mbar_base + 312)
    #define throttle_full_addr (mbar_base + 336)
    #define throttle_empty_addr (mbar_base + 360)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 384);

    // Kernel setup ops
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + 66560);
    const int smem_b_addr = smem + 66560;
    uint8_t* smem_a_mma0 = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_mma0_addr = smem + 1024;
    uint8_t* smem_a_mma1 = reinterpret_cast<uint8_t*>(smem_raw + 1056);
    const int smem_a_mma1_addr = smem + 1056;
    uint8_t* smem_a_mma2 = reinterpret_cast<uint8_t*>(smem_raw + 1088);
    const int smem_a_mma2_addr = smem + 1088;
    uint8_t* smem_a_mma3 = reinterpret_cast<uint8_t*>(smem_raw + 1120);
    const int smem_a_mma3_addr = smem + 1120;
    uint8_t* smem_a_mma4 = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_mma4_addr = smem + 1024;
    uint8_t* smem_a_mma5 = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_mma5_addr = smem + 1024;
    uint8_t* smem_a_mma6 = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_mma6_addr = smem + 1024;
    uint8_t* smem_a_mma7 = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_mma7_addr = smem + 1024;
    uint8_t* smem_b_mma0 = reinterpret_cast<uint8_t*>(smem_raw + 66560);
    const int smem_b_mma0_addr = smem + 66560;
    uint8_t* smem_b_mma1 = reinterpret_cast<uint8_t*>(smem_raw + 66592);
    const int smem_b_mma1_addr = smem + 66592;
    uint8_t* smem_b_mma2 = reinterpret_cast<uint8_t*>(smem_raw + 66624);
    const int smem_b_mma2_addr = smem + 66624;
    uint8_t* smem_b_mma3 = reinterpret_cast<uint8_t*>(smem_raw + 66656);
    const int smem_b_mma3_addr = smem + 66656;
    uint8_t* smem_b_mma4 = reinterpret_cast<uint8_t*>(smem_raw + 66560);
    const int smem_b_mma4_addr = smem + 66560;
    uint8_t* smem_b_mma5 = reinterpret_cast<uint8_t*>(smem_raw + 66560);
    const int smem_b_mma5_addr = smem + 66560;
    uint8_t* smem_b_mma6 = reinterpret_cast<uint8_t*>(smem_raw + 66560);
    const int smem_b_mma6_addr = smem + 66560;
    uint8_t* smem_b_mma7 = reinterpret_cast<uint8_t*>(smem_raw + 66560);
    const int smem_b_mma7_addr = smem + 66560;
    uint8_t* smem_a_mma0_next = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_mma0_next_addr = smem + 1024;
    uint8_t* smem_b_mma0_next = reinterpret_cast<uint8_t*>(smem_raw + 66560);
    const int smem_b_mma0_next_addr = smem + 66560;
    uint8_t* smem_a_mma2_next = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_mma2_next_addr = smem + 1024;
    uint8_t* smem_b_mma2_next = reinterpret_cast<uint8_t*>(smem_raw + 66560);
    const int smem_b_mma2_next_addr = smem + 66560;
    uint8_t* smem_a_mma5_next = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_mma5_next_addr = smem + 1024;
    uint8_t* smem_b_mma5_next = reinterpret_cast<uint8_t*>(smem_raw + 66560);
    const int smem_b_mma5_next_addr = smem + 66560;
    uint8_t* smem_sfa = reinterpret_cast<uint8_t*>(smem_raw + 167936);
    const int smem_sfa_addr = smem + 167936;
    uint8_t* smem_sfb = reinterpret_cast<uint8_t*>(smem_raw + 176128);
    const int smem_sfb_addr = smem + 176128;
    __nv_bfloat16* epi_staging = reinterpret_cast<__nv_bfloat16*>(smem_raw + 132096);
    const int epi_staging_addr = smem + 132096;
    unsigned long long* epi_staging_u64 = reinterpret_cast<unsigned long long*>(smem_raw + 132096);
    const int epi_staging_u64_addr = smem + 132096;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 184320);
    const int work_response_addr = smem + 184320;
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if ((int)blockIdx.y >= total_tiles[0]) return;

    // Mbarrier init (14 pipeline groups, 0 ordered-sequence groups, 48 barriers)
    // Mbarriers at smem_raw[0..384)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'k_pipe' ---
            // a_full: 4 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            // b_full: 4 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // sfa_full: 4 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // sfb_full: 4 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // sfa_smem_free: 4 barriers, init_count=1
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            // sfb_smem_free: 4 barriers, init_count=1
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            // tmem_sfab_full: 4 barriers, init_count=1
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
            // k_done: 4 barriers, init_count=1
            mbarrier_init(smem + 224, 1);
            mbarrier_init(smem + 232, 1);
            mbarrier_init(smem + 240, 1);
            mbarrier_init(smem + 248, 1);
            // --- pipeline 'mma_pipe' ---
            // mma_full: 2 barriers, init_count=1
            mbarrier_init(smem + 256, 1);
            mbarrier_init(smem + 264, 1);
            // mma_free: 2 barriers, init_count=4
            mbarrier_init(smem + 272, 4);
            mbarrier_init(smem + 280, 4);
            // --- pipeline 'work_pipe' ---
            // work_full: 3 barriers, init_count=1
            mbarrier_init(smem + 288, 1);
            mbarrier_init(smem + 296, 1);
            mbarrier_init(smem + 304, 1);
            // work_empty: 3 barriers, init_count=384
            mbarrier_init(smem + 312, 384);
            mbarrier_init(smem + 320, 384);
            mbarrier_init(smem + 328, 384);
            // --- pipeline 'throttle_pipe' ---
            // throttle_full: 3 barriers, init_count=32
            mbarrier_init(smem + 336, 32);
            mbarrier_init(smem + 344, 32);
            mbarrier_init(smem + 352, 32);
            // throttle_empty: 3 barriers, init_count=32
            mbarrier_init(smem + 360, 32);
            mbarrier_init(smem + 368, 32);
            mbarrier_init(smem + 376, 32);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 384 used)
    if (warp == 0) {
        int _tmem_hold = smem + 384;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;
    const int tmem_sfa = taddr + 256;
    const int tmem_sfb = taddr + 320;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 4 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 24;");
    }

    // ---- Role: epilogue ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 160;");
        { // epilogue_main
            unsigned int acc_stage = 0;
            unsigned int work_stage = 0;
            unsigned int m_tile = blockIdx.x;
            unsigned int n_tile = blockIdx.y;
            asm volatile("griddepcontrol.wait;" ::: "memory");
            asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
            const int epi_warp = warp;
            int row = epi_warp * 32;
            int feature = (unsigned int)row + lane % 8 * 4 + lane / 8;
            int wide_feature = (unsigned int)row + lane / 4 * 4;
            int wide_token = lane % 4 * 2;
            __nv_bfloat16 bf = 0.0f;
            float wide_values[4];
            unsigned int wide_packed[2];
            unsigned long long wide_word = 0;
            unsigned int _phase_work_full = 0;
            unsigned int _phase_mma_full = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter = 0; _tile_iter < grid_m * grid_n; _tile_iter++) {
                unsigned int inactive_valid = 1;
                #pragma unroll 1
                for (unsigned int _inactive_iter = 0; _inactive_iter < grid_m * grid_n; _inactive_iter++) {
                    if (n_tile < (unsigned int)total_tiles[0]) {
                        break;
                    }
                    {
                        mbarrier_wait_hint(work_full_addr + (work_stage) * 8, _phase_work_full, 10000000);
                    }
                    unsigned int valid = 0;
                    unsigned int next_x = 0;
                    unsigned int next_y = 0;
                    uint32_t _clc_valid_12 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %0, 1, 0, p1;\n\t"
                        "}\n"
                        : "=r"(_clc_valid_12)
                        : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                        : "memory");
                    valid = _clc_valid_12;
                    uint32_t _clc_ctaid_24 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_24)
                        : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                        : "memory");
                    next_x = _clc_ctaid_24;
                    uint32_t _clc_ctaid_25 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_25)
                        : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                        : "memory");
                    next_y = _clc_ctaid_25;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(work_empty_addr + (work_stage) * 8);
                    work_stage += 1;
                    if (work_stage == 3) { work_stage = 0; _phase_work_full ^= 1; }
                    inactive_valid = valid;
                    m_tile = next_x;
                    n_tile = next_y;
                    if (inactive_valid == 0) {
                        break;
                    }
                }
                if (inactive_valid == 0) {
                    break;
                }
                if (m_tile >= (unsigned int)grid_m || n_tile >= (unsigned int)grid_n) {
                    break;
                }
                int token_base = n_tile * (unsigned int)BLOCK_N;
                int off_m = m_tile * (unsigned int)BLOCK_M;
                int expert = tile_expert[n_tile];
                float output_scale = scale_c[expert];
                int _mn_limit = tile_mn_limit[n_tile];
                mbarrier_wait(mma_full_addr + (acc_stage) * 8, _phase_mma_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                float _tmem_load_0[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_0[31]))
                    : "r"(taddr + (unsigned int)(row << 16) + acc_stage * 128));
                float _tmem_load_1[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_1[31]))
                    : "r"(taddr + (unsigned int)(row << 16) + acc_stage * 128 + 64));
                float _tmem_load_2[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[31]))
                    : "r"(taddr + (unsigned int)(row + 16 << 16) + acc_stage * 128));
                float _tmem_load_3[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_3[31]))
                    : "r"(taddr + (unsigned int)(row + 16 << 16) + acc_stage * 128 + 64));
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                const float2 _scale2_0 = {output_scale, output_scale};
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
                    mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_0)[_ls], _scale2_0);
                const float2 _scale2_1 = {output_scale, output_scale};
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
                    mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_1)[_ls], _scale2_1);
                const float2 _scale2_2 = {output_scale, output_scale};
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
                    mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_2)[_ls], _scale2_2);
                const float2 _scale2_3 = {output_scale, output_scale};
                #pragma unroll
                for (int _ls = 0; _ls < 16; _ls++)
                    mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_3)[_ls], _scale2_3);
                asm volatile("cp.async.bulk.wait_group.read 0;");
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                int token = wide_token;
                {
                    wide_values[0] = _tmem_load_0[0];
                    wide_values[1] = _tmem_load_0[2];
                    wide_values[2] = _tmem_load_2[0];
                    wide_values[3] = _tmem_load_2[2];
                }
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wide_values[_lp*2 + 0], wide_values[_lp*2+1 + 0]));
                    wide_packed[_lp] = *(uint32_t*)&_bf2;
                }
                wide_word = (unsigned long long)wide_packed[0] | (unsigned long long)wide_packed[1] << 32;
                if (wide_feature < 64) {
                    epi_staging_u64[(token * 64 + wide_feature) / 4] = wide_word;
                } else {
                    epi_staging_u64[(8192 + token * 64 + wide_feature - 64) / 4] = wide_word;
                }
                int token_0 = wide_token + 1;
                {
                    wide_values[0] = _tmem_load_0[1];
                    wide_values[1] = _tmem_load_0[3];
                    wide_values[2] = _tmem_load_2[1];
                    wide_values[3] = _tmem_load_2[3];
                }
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wide_values[_lp*2 + 0], wide_values[_lp*2+1 + 0]));
                    wide_packed[_lp] = *(uint32_t*)&_bf2;
                }
                wide_word = (unsigned long long)wide_packed[0] | (unsigned long long)wide_packed[1] << 32;
                if (wide_feature < 64) {
                    epi_staging_u64[(token_0 * 64 + wide_feature) / 4] = wide_word;
                } else {
                    epi_staging_u64[(8192 + token_0 * 64 + wide_feature - 64) / 4] = wide_word;
                }
                int token_1 = wide_token + 8;
                {
                    wide_values[0] = _tmem_load_0[4];
                    wide_values[1] = _tmem_load_0[6];
                    wide_values[2] = _tmem_load_2[4];
                    wide_values[3] = _tmem_load_2[6];
                }
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wide_values[_lp*2 + 0], wide_values[_lp*2+1 + 0]));
                    wide_packed[_lp] = *(uint32_t*)&_bf2;
                }
                wide_word = (unsigned long long)wide_packed[0] | (unsigned long long)wide_packed[1] << 32;
                if (wide_feature < 64) {
                    epi_staging_u64[(token_1 * 64 + wide_feature) / 4] = wide_word;
                } else {
                    epi_staging_u64[(8192 + token_1 * 64 + wide_feature - 64) / 4] = wide_word;
                }
                int token_2 = wide_token + 8 + 1;
                {
                    wide_values[0] = _tmem_load_0[5];
                    wide_values[1] = _tmem_load_0[7];
                    wide_values[2] = _tmem_load_2[5];
                    wide_values[3] = _tmem_load_2[7];
                }
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wide_values[_lp*2 + 0], wide_values[_lp*2+1 + 0]));
                    wide_packed[_lp] = *(uint32_t*)&_bf2;
                }
                wide_word = (unsigned long long)wide_packed[0] | (unsigned long long)wide_packed[1] << 32;
                if (wide_feature < 64) {
                    epi_staging_u64[(token_2 * 64 + wide_feature) / 4] = wide_word;
                } else {
                    epi_staging_u64[(8192 + token_2 * 64 + wide_feature - 64) / 4] = wide_word;
                }
                int token_3 = wide_token + 16;
                {
                    wide_values[0] = _tmem_load_0[8];
                    wide_values[1] = _tmem_load_0[10];
                    wide_values[2] = _tmem_load_2[8];
                    wide_values[3] = _tmem_load_2[10];
                }
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wide_values[_lp*2 + 0], wide_values[_lp*2+1 + 0]));
                    wide_packed[_lp] = *(uint32_t*)&_bf2;
                }
                wide_word = (unsigned long long)wide_packed[0] | (unsigned long long)wide_packed[1] << 32;
                if (wide_feature < 64) {
                    epi_staging_u64[(token_3 * 64 + wide_feature) / 4] = wide_word;
                } else {
                    epi_staging_u64[(8192 + token_3 * 64 + wide_feature - 64) / 4] = wide_word;
                }
                int token_4 = wide_token + 16 + 1;
                {
                    wide_values[0] = _tmem_load_0[9];
                    wide_values[1] = _tmem_load_0[11];
                    wide_values[2] = _tmem_load_2[9];
                    wide_values[3] = _tmem_load_2[11];
                }
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wide_values[_lp*2 + 0], wide_values[_lp*2+1 + 0]));
                    wide_packed[_lp] = *(uint32_t*)&_bf2;
                }
                wide_word = (unsigned long long)wide_packed[0] | (unsigned long long)wide_packed[1] << 32;
                if (wide_feature < 64) {
                    epi_staging_u64[(token_4 * 64 + wide_feature) / 4] = wide_word;
                } else {
                    epi_staging_u64[(8192 + token_4 * 64 + wide_feature - 64) / 4] = wide_word;
                }
                int token_5 = wide_token + 24;
                {
                    wide_values[0] = _tmem_load_0[12];
                    wide_values[1] = _tmem_load_0[14];
                    wide_values[2] = _tmem_load_2[12];
                    wide_values[3] = _tmem_load_2[14];
                }
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wide_values[_lp*2 + 0], wide_values[_lp*2+1 + 0]));
                    wide_packed[_lp] = *(uint32_t*)&_bf2;
                }
                wide_word = (unsigned long long)wide_packed[0] | (unsigned long long)wide_packed[1] << 32;
                if (wide_feature < 64) {
                    epi_staging_u64[(token_5 * 64 + wide_feature) / 4] = wide_word;
                } else {
                    epi_staging_u64[(8192 + token_5 * 64 + wide_feature - 64) / 4] = wide_word;
                }
                int token_6 = wide_token + 24 + 1;
                {
                    wide_values[0] = _tmem_load_0[13];
                    wide_values[1] = _tmem_load_0[15];
                    wide_values[2] = _tmem_load_2[13];
                    wide_values[3] = _tmem_load_2[15];
                }
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wide_values[_lp*2 + 0], wide_values[_lp*2+1 + 0]));
                    wide_packed[_lp] = *(uint32_t*)&_bf2;
                }
                wide_word = (unsigned long long)wide_packed[0] | (unsigned long long)wide_packed[1] << 32;
                if (wide_feature < 64) {
                    epi_staging_u64[(token_6 * 64 + wide_feature) / 4] = wide_word;
                } else {
                    epi_staging_u64[(8192 + token_6 * 64 + wide_feature - 64) / 4] = wide_word;
                }
                int token_7 = wide_token + 32;
                {
                    wide_values[0] = _tmem_load_0[16];
                    wide_values[1] = _tmem_load_0[18];
                    wide_values[2] = _tmem_load_2[16];
                    wide_values[3] = _tmem_load_2[18];
                }
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wide_values[_lp*2 + 0], wide_values[_lp*2+1 + 0]));
                    wide_packed[_lp] = *(uint32_t*)&_bf2;
                }
                wide_word = (unsigned long long)wide_packed[0] | (unsigned long long)wide_packed[1] << 32;
                if (wide_feature < 64) {
                    epi_staging_u64[(token_7 * 64 + wide_feature) / 4] = wide_word;
                } else {
                    epi_staging_u64[(8192 + token_7 * 64 + wide_feature - 64) / 4] = wide_word;
                }
                int token_8 = wide_token + 32 + 1;
                {
                    wide_values[0] = _tmem_load_0[17];
                    wide_values[1] = _tmem_load_0[19];
                    wide_values[2] = _tmem_load_2[17];
                    wide_values[3] = _tmem_load_2[19];
                }
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wide_values[_lp*2 + 0], wide_values[_lp*2+1 + 0]));
                    wide_packed[_lp] = *(uint32_t*)&_bf2;
                }
                wide_word = (unsigned long long)wide_packed[0] | (unsigned long long)wide_packed[1] << 32;
                if (wide_feature < 64) {
                    epi_staging_u64[(token_8 * 64 + wide_feature) / 4] = wide_word;
                } else {
                    epi_staging_u64[(8192 + token_8 * 64 + wide_feature - 64) / 4] = wide_word;
                }
                int token_9 = wide_token + 40;
                {
                    wide_values[0] = _tmem_load_0[20];
                    wide_values[1] = _tmem_load_0[22];
                    wide_values[2] = _tmem_load_2[20];
                    wide_values[3] = _tmem_load_2[22];
                }
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wide_values[_lp*2 + 0], wide_values[_lp*2+1 + 0]));
                    wide_packed[_lp] = *(uint32_t*)&_bf2;
                }
                wide_word = (unsigned long long)wide_packed[0] | (unsigned long long)wide_packed[1] << 32;
                if (wide_feature < 64) {
                    epi_staging_u64[(token_9 * 64 + wide_feature) / 4] = wide_word;
                } else {
                    epi_staging_u64[(8192 + token_9 * 64 + wide_feature - 64) / 4] = wide_word;
                }
                int token_10 = wide_token + 40 + 1;
                {
                    wide_values[0] = _tmem_load_0[21];
                    wide_values[1] = _tmem_load_0[23];
                    wide_values[2] = _tmem_load_2[21];
                    wide_values[3] = _tmem_load_2[23];
                }
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wide_values[_lp*2 + 0], wide_values[_lp*2+1 + 0]));
                    wide_packed[_lp] = *(uint32_t*)&_bf2;
                }
                wide_word = (unsigned long long)wide_packed[0] | (unsigned long long)wide_packed[1] << 32;
                if (wide_feature < 64) {
                    epi_staging_u64[(token_10 * 64 + wide_feature) / 4] = wide_word;
                } else {
                    epi_staging_u64[(8192 + token_10 * 64 + wide_feature - 64) / 4] = wide_word;
                }
                int token_11 = wide_token + 48;
                {
                    wide_values[0] = _tmem_load_0[24];
                    wide_values[1] = _tmem_load_0[26];
                    wide_values[2] = _tmem_load_2[24];
                    wide_values[3] = _tmem_load_2[26];
                }
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wide_values[_lp*2 + 0], wide_values[_lp*2+1 + 0]));
                    wide_packed[_lp] = *(uint32_t*)&_bf2;
                }
                wide_word = (unsigned long long)wide_packed[0] | (unsigned long long)wide_packed[1] << 32;
                if (wide_feature < 64) {
                    epi_staging_u64[(token_11 * 64 + wide_feature) / 4] = wide_word;
                } else {
                    epi_staging_u64[(8192 + token_11 * 64 + wide_feature - 64) / 4] = wide_word;
                }
                int token_12 = wide_token + 48 + 1;
                {
                    wide_values[0] = _tmem_load_0[25];
                    wide_values[1] = _tmem_load_0[27];
                    wide_values[2] = _tmem_load_2[25];
                    wide_values[3] = _tmem_load_2[27];
                }
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wide_values[_lp*2 + 0], wide_values[_lp*2+1 + 0]));
                    wide_packed[_lp] = *(uint32_t*)&_bf2;
                }
                wide_word = (unsigned long long)wide_packed[0] | (unsigned long long)wide_packed[1] << 32;
                if (wide_feature < 64) {
                    epi_staging_u64[(token_12 * 64 + wide_feature) / 4] = wide_word;
                } else {
                    epi_staging_u64[(8192 + token_12 * 64 + wide_feature - 64) / 4] = wide_word;
                }
                int token_13 = wide_token + 56;
                {
                    wide_values[0] = _tmem_load_0[28];
                    wide_values[1] = _tmem_load_0[30];
                    wide_values[2] = _tmem_load_2[28];
                    wide_values[3] = _tmem_load_2[30];
                }
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wide_values[_lp*2 + 0], wide_values[_lp*2+1 + 0]));
                    wide_packed[_lp] = *(uint32_t*)&_bf2;
                }
                wide_word = (unsigned long long)wide_packed[0] | (unsigned long long)wide_packed[1] << 32;
                if (wide_feature < 64) {
                    epi_staging_u64[(token_13 * 64 + wide_feature) / 4] = wide_word;
                } else {
                    epi_staging_u64[(8192 + token_13 * 64 + wide_feature - 64) / 4] = wide_word;
                }
                int token_14 = wide_token + 56 + 1;
                {
                    wide_values[0] = _tmem_load_0[29];
                    wide_values[1] = _tmem_load_0[31];
                    wide_values[2] = _tmem_load_2[29];
                    wide_values[3] = _tmem_load_2[31];
                }
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wide_values[_lp*2 + 0], wide_values[_lp*2+1 + 0]));
                    wide_packed[_lp] = *(uint32_t*)&_bf2;
                }
                wide_word = (unsigned long long)wide_packed[0] | (unsigned long long)wide_packed[1] << 32;
                if (wide_feature < 64) {
                    epi_staging_u64[(token_14 * 64 + wide_feature) / 4] = wide_word;
                } else {
                    epi_staging_u64[(8192 + token_14 * 64 + wide_feature - 64) / 4] = wide_word;
                }
                int token_15 = wide_token + 64;
                {
                    wide_values[0] = _tmem_load_1[0];
                    wide_values[1] = _tmem_load_1[2];
                    wide_values[2] = _tmem_load_3[0];
                    wide_values[3] = _tmem_load_3[2];
                }
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wide_values[_lp*2 + 0], wide_values[_lp*2+1 + 0]));
                    wide_packed[_lp] = *(uint32_t*)&_bf2;
                }
                wide_word = (unsigned long long)wide_packed[0] | (unsigned long long)wide_packed[1] << 32;
                if (wide_feature < 64) {
                    epi_staging_u64[(token_15 * 64 + wide_feature) / 4] = wide_word;
                } else {
                    epi_staging_u64[(8192 + token_15 * 64 + wide_feature - 64) / 4] = wide_word;
                }
                int token_16 = wide_token + 64 + 1;
                {
                    wide_values[0] = _tmem_load_1[1];
                    wide_values[1] = _tmem_load_1[3];
                    wide_values[2] = _tmem_load_3[1];
                    wide_values[3] = _tmem_load_3[3];
                }
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wide_values[_lp*2 + 0], wide_values[_lp*2+1 + 0]));
                    wide_packed[_lp] = *(uint32_t*)&_bf2;
                }
                wide_word = (unsigned long long)wide_packed[0] | (unsigned long long)wide_packed[1] << 32;
                if (wide_feature < 64) {
                    epi_staging_u64[(token_16 * 64 + wide_feature) / 4] = wide_word;
                } else {
                    epi_staging_u64[(8192 + token_16 * 64 + wide_feature - 64) / 4] = wide_word;
                }
                int token_17 = wide_token + 64 + 8;
                {
                    wide_values[0] = _tmem_load_1[4];
                    wide_values[1] = _tmem_load_1[6];
                    wide_values[2] = _tmem_load_3[4];
                    wide_values[3] = _tmem_load_3[6];
                }
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wide_values[_lp*2 + 0], wide_values[_lp*2+1 + 0]));
                    wide_packed[_lp] = *(uint32_t*)&_bf2;
                }
                wide_word = (unsigned long long)wide_packed[0] | (unsigned long long)wide_packed[1] << 32;
                if (wide_feature < 64) {
                    epi_staging_u64[(token_17 * 64 + wide_feature) / 4] = wide_word;
                } else {
                    epi_staging_u64[(8192 + token_17 * 64 + wide_feature - 64) / 4] = wide_word;
                }
                int token_18 = wide_token + 64 + 8 + 1;
                {
                    wide_values[0] = _tmem_load_1[5];
                    wide_values[1] = _tmem_load_1[7];
                    wide_values[2] = _tmem_load_3[5];
                    wide_values[3] = _tmem_load_3[7];
                }
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wide_values[_lp*2 + 0], wide_values[_lp*2+1 + 0]));
                    wide_packed[_lp] = *(uint32_t*)&_bf2;
                }
                wide_word = (unsigned long long)wide_packed[0] | (unsigned long long)wide_packed[1] << 32;
                if (wide_feature < 64) {
                    epi_staging_u64[(token_18 * 64 + wide_feature) / 4] = wide_word;
                } else {
                    epi_staging_u64[(8192 + token_18 * 64 + wide_feature - 64) / 4] = wide_word;
                }
                int token_19 = wide_token + 64 + 16;
                {
                    wide_values[0] = _tmem_load_1[8];
                    wide_values[1] = _tmem_load_1[10];
                    wide_values[2] = _tmem_load_3[8];
                    wide_values[3] = _tmem_load_3[10];
                }
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wide_values[_lp*2 + 0], wide_values[_lp*2+1 + 0]));
                    wide_packed[_lp] = *(uint32_t*)&_bf2;
                }
                wide_word = (unsigned long long)wide_packed[0] | (unsigned long long)wide_packed[1] << 32;
                if (wide_feature < 64) {
                    epi_staging_u64[(token_19 * 64 + wide_feature) / 4] = wide_word;
                } else {
                    epi_staging_u64[(8192 + token_19 * 64 + wide_feature - 64) / 4] = wide_word;
                }
                int token_20 = wide_token + 64 + 16 + 1;
                {
                    wide_values[0] = _tmem_load_1[9];
                    wide_values[1] = _tmem_load_1[11];
                    wide_values[2] = _tmem_load_3[9];
                    wide_values[3] = _tmem_load_3[11];
                }
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wide_values[_lp*2 + 0], wide_values[_lp*2+1 + 0]));
                    wide_packed[_lp] = *(uint32_t*)&_bf2;
                }
                wide_word = (unsigned long long)wide_packed[0] | (unsigned long long)wide_packed[1] << 32;
                if (wide_feature < 64) {
                    epi_staging_u64[(token_20 * 64 + wide_feature) / 4] = wide_word;
                } else {
                    epi_staging_u64[(8192 + token_20 * 64 + wide_feature - 64) / 4] = wide_word;
                }
                int token_21 = wide_token + 64 + 24;
                {
                    wide_values[0] = _tmem_load_1[12];
                    wide_values[1] = _tmem_load_1[14];
                    wide_values[2] = _tmem_load_3[12];
                    wide_values[3] = _tmem_load_3[14];
                }
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wide_values[_lp*2 + 0], wide_values[_lp*2+1 + 0]));
                    wide_packed[_lp] = *(uint32_t*)&_bf2;
                }
                wide_word = (unsigned long long)wide_packed[0] | (unsigned long long)wide_packed[1] << 32;
                if (wide_feature < 64) {
                    epi_staging_u64[(token_21 * 64 + wide_feature) / 4] = wide_word;
                } else {
                    epi_staging_u64[(8192 + token_21 * 64 + wide_feature - 64) / 4] = wide_word;
                }
                int token_22 = wide_token + 64 + 24 + 1;
                {
                    wide_values[0] = _tmem_load_1[13];
                    wide_values[1] = _tmem_load_1[15];
                    wide_values[2] = _tmem_load_3[13];
                    wide_values[3] = _tmem_load_3[15];
                }
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wide_values[_lp*2 + 0], wide_values[_lp*2+1 + 0]));
                    wide_packed[_lp] = *(uint32_t*)&_bf2;
                }
                wide_word = (unsigned long long)wide_packed[0] | (unsigned long long)wide_packed[1] << 32;
                if (wide_feature < 64) {
                    epi_staging_u64[(token_22 * 64 + wide_feature) / 4] = wide_word;
                } else {
                    epi_staging_u64[(8192 + token_22 * 64 + wide_feature - 64) / 4] = wide_word;
                }
                int token_23 = wide_token + 64 + 32;
                {
                    wide_values[0] = _tmem_load_1[16];
                    wide_values[1] = _tmem_load_1[18];
                    wide_values[2] = _tmem_load_3[16];
                    wide_values[3] = _tmem_load_3[18];
                }
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wide_values[_lp*2 + 0], wide_values[_lp*2+1 + 0]));
                    wide_packed[_lp] = *(uint32_t*)&_bf2;
                }
                wide_word = (unsigned long long)wide_packed[0] | (unsigned long long)wide_packed[1] << 32;
                if (wide_feature < 64) {
                    epi_staging_u64[(token_23 * 64 + wide_feature) / 4] = wide_word;
                } else {
                    epi_staging_u64[(8192 + token_23 * 64 + wide_feature - 64) / 4] = wide_word;
                }
                int token_24 = wide_token + 64 + 32 + 1;
                {
                    wide_values[0] = _tmem_load_1[17];
                    wide_values[1] = _tmem_load_1[19];
                    wide_values[2] = _tmem_load_3[17];
                    wide_values[3] = _tmem_load_3[19];
                }
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wide_values[_lp*2 + 0], wide_values[_lp*2+1 + 0]));
                    wide_packed[_lp] = *(uint32_t*)&_bf2;
                }
                wide_word = (unsigned long long)wide_packed[0] | (unsigned long long)wide_packed[1] << 32;
                if (wide_feature < 64) {
                    epi_staging_u64[(token_24 * 64 + wide_feature) / 4] = wide_word;
                } else {
                    epi_staging_u64[(8192 + token_24 * 64 + wide_feature - 64) / 4] = wide_word;
                }
                int token_25 = wide_token + 64 + 40;
                {
                    wide_values[0] = _tmem_load_1[20];
                    wide_values[1] = _tmem_load_1[22];
                    wide_values[2] = _tmem_load_3[20];
                    wide_values[3] = _tmem_load_3[22];
                }
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wide_values[_lp*2 + 0], wide_values[_lp*2+1 + 0]));
                    wide_packed[_lp] = *(uint32_t*)&_bf2;
                }
                wide_word = (unsigned long long)wide_packed[0] | (unsigned long long)wide_packed[1] << 32;
                if (wide_feature < 64) {
                    epi_staging_u64[(token_25 * 64 + wide_feature) / 4] = wide_word;
                } else {
                    epi_staging_u64[(8192 + token_25 * 64 + wide_feature - 64) / 4] = wide_word;
                }
                int token_26 = wide_token + 64 + 40 + 1;
                {
                    wide_values[0] = _tmem_load_1[21];
                    wide_values[1] = _tmem_load_1[23];
                    wide_values[2] = _tmem_load_3[21];
                    wide_values[3] = _tmem_load_3[23];
                }
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wide_values[_lp*2 + 0], wide_values[_lp*2+1 + 0]));
                    wide_packed[_lp] = *(uint32_t*)&_bf2;
                }
                wide_word = (unsigned long long)wide_packed[0] | (unsigned long long)wide_packed[1] << 32;
                if (wide_feature < 64) {
                    epi_staging_u64[(token_26 * 64 + wide_feature) / 4] = wide_word;
                } else {
                    epi_staging_u64[(8192 + token_26 * 64 + wide_feature - 64) / 4] = wide_word;
                }
                int token_27 = wide_token + 64 + 48;
                {
                    wide_values[0] = _tmem_load_1[24];
                    wide_values[1] = _tmem_load_1[26];
                    wide_values[2] = _tmem_load_3[24];
                    wide_values[3] = _tmem_load_3[26];
                }
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wide_values[_lp*2 + 0], wide_values[_lp*2+1 + 0]));
                    wide_packed[_lp] = *(uint32_t*)&_bf2;
                }
                wide_word = (unsigned long long)wide_packed[0] | (unsigned long long)wide_packed[1] << 32;
                if (wide_feature < 64) {
                    epi_staging_u64[(token_27 * 64 + wide_feature) / 4] = wide_word;
                } else {
                    epi_staging_u64[(8192 + token_27 * 64 + wide_feature - 64) / 4] = wide_word;
                }
                int token_28 = wide_token + 64 + 48 + 1;
                {
                    wide_values[0] = _tmem_load_1[25];
                    wide_values[1] = _tmem_load_1[27];
                    wide_values[2] = _tmem_load_3[25];
                    wide_values[3] = _tmem_load_3[27];
                }
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wide_values[_lp*2 + 0], wide_values[_lp*2+1 + 0]));
                    wide_packed[_lp] = *(uint32_t*)&_bf2;
                }
                wide_word = (unsigned long long)wide_packed[0] | (unsigned long long)wide_packed[1] << 32;
                if (wide_feature < 64) {
                    epi_staging_u64[(token_28 * 64 + wide_feature) / 4] = wide_word;
                } else {
                    epi_staging_u64[(8192 + token_28 * 64 + wide_feature - 64) / 4] = wide_word;
                }
                int token_29 = wide_token + 64 + 56;
                {
                    wide_values[0] = _tmem_load_1[28];
                    wide_values[1] = _tmem_load_1[30];
                    wide_values[2] = _tmem_load_3[28];
                    wide_values[3] = _tmem_load_3[30];
                }
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wide_values[_lp*2 + 0], wide_values[_lp*2+1 + 0]));
                    wide_packed[_lp] = *(uint32_t*)&_bf2;
                }
                wide_word = (unsigned long long)wide_packed[0] | (unsigned long long)wide_packed[1] << 32;
                if (wide_feature < 64) {
                    epi_staging_u64[(token_29 * 64 + wide_feature) / 4] = wide_word;
                } else {
                    epi_staging_u64[(8192 + token_29 * 64 + wide_feature - 64) / 4] = wide_word;
                }
                int token_30 = wide_token + 64 + 56 + 1;
                {
                    wide_values[0] = _tmem_load_1[29];
                    wide_values[1] = _tmem_load_1[31];
                    wide_values[2] = _tmem_load_3[29];
                    wide_values[3] = _tmem_load_3[31];
                }
                #pragma unroll
                for (int _lp = 0; _lp < 2; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(wide_values[_lp*2 + 0], wide_values[_lp*2+1 + 0]));
                    wide_packed[_lp] = *(uint32_t*)&_bf2;
                }
                wide_word = (unsigned long long)wide_packed[0] | (unsigned long long)wide_packed[1] << 32;
                if (wide_feature < 64) {
                    epi_staging_u64[(token_30 * 64 + wide_feature) / 4] = wide_word;
                } else {
                    epi_staging_u64[(8192 + token_30 * 64 + wide_feature - 64) / 4] = wide_word;
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                if (warp == 0) {
                    if (elect_sync()) {
                        int padding_rows = (128 - _mn_limit % 128) % 128;
                        int local_token = padding_rows;
                        tma_store_4d((&C_tma), off_m, local_token, 1073741824, token_base - padding_rows + 1073741824, epi_staging_addr);
                        tma_store_4d((&C_tma), off_m + 64, local_token, 1073741824, token_base - padding_rows + 1073741824, epi_staging_addr + 16384);
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                }
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                if (elect_sync()) {
                    mbarrier_arrive(mma_free_addr + (acc_stage) * 8);
                }
                acc_stage += 1;
                if (acc_stage == 2) { acc_stage = 0; _phase_mma_full ^= 1; }
                {
                    mbarrier_wait_hint(work_full_addr + (work_stage) * 8, _phase_work_full, 10000000);
                }
                unsigned int valid_1 = 0;
                unsigned int next_x_1 = 0;
                unsigned int next_y_1 = 0;
                uint32_t _clc_valid_13 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_13)
                    : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                    : "memory");
                valid_1 = _clc_valid_13;
                uint32_t _clc_ctaid_26 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_26)
                    : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                    : "memory");
                next_x_1 = _clc_ctaid_26;
                uint32_t _clc_ctaid_27 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_27)
                    : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                    : "memory");
                next_y_1 = _clc_ctaid_27;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage) * 8);
                work_stage += 1;
                if (work_stage == 3) { work_stage = 0; _phase_work_full ^= 1; }
                unsigned int valid_31 = valid_1;
                m_tile = next_x_1;
                n_tile = next_y_1;
                if (valid_31 == 0) {
                    break;
                }
            }
        }
    }
    // ---- Role: load_b ----
    if (warp == 4) {
        { // load_b_main
            unsigned int stage = 0;
            unsigned int work_stage_1 = 0;
            unsigned int m_tile_1 = blockIdx.x;
            unsigned int n_tile_1 = blockIdx.y;
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int _phase_work_full_1 = 0;
            unsigned int _phase_k_done = 1;
            #pragma unroll 1
            for (unsigned int _tile_iter_1 = 0; _tile_iter_1 < grid_m * grid_n; _tile_iter_1++) {
                unsigned int inactive_valid_1 = 1;
                #pragma unroll 1
                for (unsigned int _inactive_iter_1 = 0; _inactive_iter_1 < grid_m * grid_n; _inactive_iter_1++) {
                    if (n_tile_1 < (unsigned int)total_tiles[0]) {
                        break;
                    }
                    {
                        mbarrier_wait_hint(work_full_addr + (work_stage_1) * 8, _phase_work_full_1, 10000000);
                    }
                    unsigned int valid_2 = 0;
                    unsigned int next_x_2 = 0;
                    unsigned int next_y_2 = 0;
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
                        : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    valid_2 = _clc_valid_2;
                    uint32_t _clc_ctaid_4 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_4)
                        : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    next_x_2 = _clc_ctaid_4;
                    uint32_t _clc_ctaid_5 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_5)
                        : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    next_y_2 = _clc_ctaid_5;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(work_empty_addr + (work_stage_1) * 8);
                    work_stage_1 += 1;
                    if (work_stage_1 == 3) { work_stage_1 = 0; _phase_work_full_1 ^= 1; }
                    inactive_valid_1 = valid_2;
                    m_tile_1 = next_x_2;
                    n_tile_1 = next_y_2;
                    if (inactive_valid_1 == 0) {
                        break;
                    }
                }
                if (inactive_valid_1 == 0) {
                    break;
                }
                if (m_tile_1 >= (unsigned int)grid_m || n_tile_1 >= (unsigned int)grid_n) {
                    break;
                }
                int route_k_extent = K_tiles;
                #pragma unroll 1
                for (int route_k = 0; route_k < route_k_extent; route_k++) {
                    int iter_k = route_k;
                    int route_tile = n_tile_1;
                    mbarrier_wait(k_done_addr + (stage) * 8, _phase_k_done);
                    if (elect_sync()) {
                        tma_3d_gmem2smem(smem_b_addr + stage * 16384, (&B), iter_k * 256, 0, route_tile, b_full_addr + (stage) * 8);
                        mbarrier_arrive_expect_tx(b_full_addr + (stage) * 8, 16384);
                    }
                    stage += 1;
                    if (stage == 4) { stage = 0; _phase_k_done ^= 1; }
                }
                {
                    mbarrier_wait_hint(work_full_addr + (work_stage_1) * 8, _phase_work_full_1, 10000000);
                }
                unsigned int valid_3 = 0;
                unsigned int next_x_3 = 0;
                unsigned int next_y_3 = 0;
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
                    : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                    : "memory");
                valid_3 = _clc_valid_3;
                uint32_t _clc_ctaid_6 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_6)
                    : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                    : "memory");
                next_x_3 = _clc_ctaid_6;
                uint32_t _clc_ctaid_7 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_7)
                    : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                    : "memory");
                next_y_3 = _clc_ctaid_7;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage_1) * 8);
                work_stage_1 += 1;
                if (work_stage_1 == 3) { work_stage_1 = 0; _phase_work_full_1 ^= 1; }
                unsigned int valid_0 = valid_3;
                m_tile_1 = next_x_3;
                n_tile_1 = next_y_3;
                if (valid_0 == 0) {
                    break;
                }
            }
        }
    }
    // ---- Role: load_sfb ----
    if (warp == 5) {
        { // load_sfb_main
            unsigned int stage_1 = 0;
            unsigned int work_stage_2 = 0;
            unsigned int m_tile_2 = blockIdx.x;
            unsigned int n_tile_2 = blockIdx.y;
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int _phase_work_full_2 = 0;
            unsigned int _phase_sfb_smem_free = 1;
            #pragma unroll 1
            for (unsigned int _tile_iter_2 = 0; _tile_iter_2 < grid_m * grid_n; _tile_iter_2++) {
                unsigned int inactive_valid_2 = 1;
                #pragma unroll 1
                for (unsigned int _inactive_iter_2 = 0; _inactive_iter_2 < grid_m * grid_n; _inactive_iter_2++) {
                    if (n_tile_2 < (unsigned int)total_tiles[0]) {
                        break;
                    }
                    {
                        mbarrier_wait_hint(work_full_addr + (work_stage_2) * 8, _phase_work_full_2, 10000000);
                    }
                    unsigned int valid_4 = 0;
                    unsigned int next_x_4 = 0;
                    unsigned int next_y_4 = 0;
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
                        : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                        : "memory");
                    valid_4 = _clc_valid_6;
                    uint32_t _clc_ctaid_12 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_12)
                        : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                        : "memory");
                    next_x_4 = _clc_ctaid_12;
                    uint32_t _clc_ctaid_13 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_13)
                        : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                        : "memory");
                    next_y_4 = _clc_ctaid_13;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(work_empty_addr + (work_stage_2) * 8);
                    work_stage_2 += 1;
                    if (work_stage_2 == 3) { work_stage_2 = 0; _phase_work_full_2 ^= 1; }
                    inactive_valid_2 = valid_4;
                    m_tile_2 = next_x_4;
                    n_tile_2 = next_y_4;
                    if (inactive_valid_2 == 0) {
                        break;
                    }
                }
                if (inactive_valid_2 == 0) {
                    break;
                }
                if (m_tile_2 >= (unsigned int)grid_m || n_tile_2 >= (unsigned int)grid_n) {
                    break;
                }
                int route_k_extent_1 = K_tiles;
                #pragma unroll 1
                for (int route_k_1 = 0; route_k_1 < route_k_extent_1; route_k_1++) {
                    int iter_k_1 = route_k_1;
                    int route_tile_1 = n_tile_2;
                    mbarrier_wait(sfb_smem_free_addr + (stage_1) * 8, _phase_sfb_smem_free);
                    if (elect_sync()) {
                        tma_4d_gmem2smem(smem_sfb_addr + stage_1 * 2048, (&SFB), 0, 0, iter_k_1 * 4, route_tile_1, sfb_full_addr + (stage_1) * 8);
                        mbarrier_arrive_expect_tx(sfb_full_addr + (stage_1) * 8, 2048);
                    }
                    stage_1 += 1;
                    if (stage_1 == 4) { stage_1 = 0; _phase_sfb_smem_free ^= 1; }
                }
                {
                    mbarrier_wait_hint(work_full_addr + (work_stage_2) * 8, _phase_work_full_2, 10000000);
                }
                unsigned int valid_5 = 0;
                unsigned int next_x_5 = 0;
                unsigned int next_y_5 = 0;
                uint32_t _clc_valid_7 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_7)
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                    : "memory");
                valid_5 = _clc_valid_7;
                uint32_t _clc_ctaid_14 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_14)
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                    : "memory");
                next_x_5 = _clc_ctaid_14;
                uint32_t _clc_ctaid_15 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_15)
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                    : "memory");
                next_y_5 = _clc_ctaid_15;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage_2) * 8);
                work_stage_2 += 1;
                if (work_stage_2 == 3) { work_stage_2 = 0; _phase_work_full_2 ^= 1; }
                unsigned int valid_0_1 = valid_5;
                m_tile_2 = next_x_5;
                n_tile_2 = next_y_5;
                if (valid_0_1 == 0) {
                    break;
                }
            }
        }
    }
    // ---- Role: load_a ----
    if (warp == 6) {
        { // load_a_main
            unsigned int stage_2 = 0;
            unsigned int work_stage_3 = 0;
            unsigned int throttle_stage = 0;
            unsigned int m_tile_3 = blockIdx.x;
            unsigned int n_tile_3 = blockIdx.y;
            unsigned int _phase_work_full_3 = 0;
            unsigned int _phase_throttle_empty = 1;
            unsigned int _phase_k_done_1 = 1;
            #pragma unroll 1
            for (unsigned int _tile_iter_3 = 0; _tile_iter_3 < grid_m * grid_n; _tile_iter_3++) {
                unsigned int inactive_valid_3 = 1;
                #pragma unroll 1
                for (unsigned int _inactive_iter_3 = 0; _inactive_iter_3 < grid_m * grid_n; _inactive_iter_3++) {
                    if (n_tile_3 < (unsigned int)total_tiles[0]) {
                        break;
                    }
                    {
                        mbarrier_wait_hint(work_full_addr + (work_stage_3) * 8, _phase_work_full_3, 10000000);
                    }
                    unsigned int valid_6 = 0;
                    unsigned int next_x_6 = 0;
                    unsigned int next_y_6 = 0;
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
                    valid_6 = _clc_valid_0;
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
                    next_x_6 = _clc_ctaid_0;
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
                    next_y_6 = _clc_ctaid_1;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(work_empty_addr + (work_stage_3) * 8);
                    work_stage_3 += 1;
                    if (work_stage_3 == 3) { work_stage_3 = 0; _phase_work_full_3 ^= 1; }
                    inactive_valid_3 = valid_6;
                    m_tile_3 = next_x_6;
                    n_tile_3 = next_y_6;
                    if (inactive_valid_3 == 0) {
                        break;
                    }
                }
                if (inactive_valid_3 == 0) {
                    break;
                }
                if (m_tile_3 >= (unsigned int)grid_m || n_tile_3 >= (unsigned int)grid_n) {
                    break;
                }
                mbarrier_wait(throttle_empty_addr + (throttle_stage) * 8, _phase_throttle_empty);
                mbarrier_arrive(throttle_full_addr + (throttle_stage) * 8);
                throttle_stage += 1;
                if (throttle_stage == 3) { throttle_stage = 0; _phase_throttle_empty ^= 1; }
                int route_k_extent_2 = K_tiles;
                #pragma unroll 1
                for (int route_k_2 = 0; route_k_2 < route_k_extent_2; route_k_2++) {
                    int iter_k_2 = route_k_2;
                    int expert_1 = tile_expert[n_tile_3];
                    mbarrier_wait(k_done_addr + (stage_2) * 8, _phase_k_done_1);
                    if (elect_sync()) {
                        tma_3d_gmem2smem(smem_a_addr + stage_2 * 16384, (&A), iter_k_2 * 256, m_tile_3 * (unsigned int)BLOCK_M, expert_1, a_full_addr + (stage_2) * 8);
                        mbarrier_arrive_expect_tx(a_full_addr + (stage_2) * 8, 16384);
                    }
                    stage_2 += 1;
                    if (stage_2 == 4) { stage_2 = 0; _phase_k_done_1 ^= 1; }
                }
                {
                    mbarrier_wait_hint(work_full_addr + (work_stage_3) * 8, _phase_work_full_3, 10000000);
                }
                unsigned int valid_7 = 0;
                unsigned int next_x_7 = 0;
                unsigned int next_y_7 = 0;
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
                    : "r"(work_response_addr + work_stage_3 * 16 + 0 * 16)
                    : "memory");
                valid_7 = _clc_valid_1;
                uint32_t _clc_ctaid_2 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_2)
                    : "r"(work_response_addr + work_stage_3 * 16 + 0 * 16)
                    : "memory");
                next_x_7 = _clc_ctaid_2;
                uint32_t _clc_ctaid_3 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_3)
                    : "r"(work_response_addr + work_stage_3 * 16 + 0 * 16)
                    : "memory");
                next_y_7 = _clc_ctaid_3;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage_3) * 8);
                work_stage_3 += 1;
                if (work_stage_3 == 3) { work_stage_3 = 0; _phase_work_full_3 ^= 1; }
                unsigned int valid_0_2 = valid_7;
                m_tile_3 = next_x_7;
                n_tile_3 = next_y_7;
                if (valid_0_2 == 0) {
                    break;
                }
            }
        }
    }
    // ---- Role: load_sfa ----
    if (warp == 7) {
        { // load_sfa_main
            unsigned int stage_3 = 0;
            unsigned int work_stage_4 = 0;
            unsigned int m_tile_4 = blockIdx.x;
            unsigned int n_tile_4 = blockIdx.y;
            unsigned int _phase_work_full_4 = 0;
            unsigned int _phase_sfa_smem_free = 1;
            #pragma unroll 1
            for (unsigned int _tile_iter_4 = 0; _tile_iter_4 < grid_m * grid_n; _tile_iter_4++) {
                unsigned int inactive_valid_4 = 1;
                #pragma unroll 1
                for (unsigned int _inactive_iter_4 = 0; _inactive_iter_4 < grid_m * grid_n; _inactive_iter_4++) {
                    if (n_tile_4 < (unsigned int)total_tiles[0]) {
                        break;
                    }
                    {
                        mbarrier_wait_hint(work_full_addr + (work_stage_4) * 8, _phase_work_full_4, 10000000);
                    }
                    unsigned int valid_8 = 0;
                    unsigned int next_x_8 = 0;
                    unsigned int next_y_8 = 0;
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
                        : "r"(work_response_addr + work_stage_4 * 16 + 0 * 16)
                        : "memory");
                    valid_8 = _clc_valid_4;
                    uint32_t _clc_ctaid_8 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_8)
                        : "r"(work_response_addr + work_stage_4 * 16 + 0 * 16)
                        : "memory");
                    next_x_8 = _clc_ctaid_8;
                    uint32_t _clc_ctaid_9 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_9)
                        : "r"(work_response_addr + work_stage_4 * 16 + 0 * 16)
                        : "memory");
                    next_y_8 = _clc_ctaid_9;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(work_empty_addr + (work_stage_4) * 8);
                    work_stage_4 += 1;
                    if (work_stage_4 == 3) { work_stage_4 = 0; _phase_work_full_4 ^= 1; }
                    inactive_valid_4 = valid_8;
                    m_tile_4 = next_x_8;
                    n_tile_4 = next_y_8;
                    if (inactive_valid_4 == 0) {
                        break;
                    }
                }
                if (inactive_valid_4 == 0) {
                    break;
                }
                if (m_tile_4 >= (unsigned int)grid_m || n_tile_4 >= (unsigned int)grid_n) {
                    break;
                }
                int route_k_extent_3 = K_tiles;
                #pragma unroll 1
                for (int route_k_3 = 0; route_k_3 < route_k_extent_3; route_k_3++) {
                    int iter_k_3 = route_k_3;
                    int expert_2 = tile_expert[n_tile_4];
                    mbarrier_wait(sfa_smem_free_addr + (stage_3) * 8, _phase_sfa_smem_free);
                    if (elect_sync()) {
                        tma_4d_gmem2smem(smem_sfa_addr + stage_3 * 2048, (&SFA), 0, 0, iter_k_3 * 4, (unsigned int)(expert_2 * grid_m) + m_tile_4, sfa_full_addr + (stage_3) * 8);
                        mbarrier_arrive_expect_tx(sfa_full_addr + (stage_3) * 8, 2048);
                    }
                    stage_3 += 1;
                    if (stage_3 == 4) { stage_3 = 0; _phase_sfa_smem_free ^= 1; }
                }
                {
                    mbarrier_wait_hint(work_full_addr + (work_stage_4) * 8, _phase_work_full_4, 10000000);
                }
                unsigned int valid_9 = 0;
                unsigned int next_x_9 = 0;
                unsigned int next_y_9 = 0;
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
                valid_9 = _clc_valid_5;
                uint32_t _clc_ctaid_10 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_10)
                    : "r"(work_response_addr + work_stage_4 * 16 + 0 * 16)
                    : "memory");
                next_x_9 = _clc_ctaid_10;
                uint32_t _clc_ctaid_11 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_11)
                    : "r"(work_response_addr + work_stage_4 * 16 + 0 * 16)
                    : "memory");
                next_y_9 = _clc_ctaid_11;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage_4) * 8);
                work_stage_4 += 1;
                if (work_stage_4 == 3) { work_stage_4 = 0; _phase_work_full_4 ^= 1; }
                unsigned int valid_0_3 = valid_9;
                m_tile_4 = next_x_9;
                n_tile_4 = next_y_9;
                if (valid_0_3 == 0) {
                    break;
                }
            }
        }
    }
    // ---- Role: copy_sfab ----
    if (warp == 8) {
        { // copy_sfa_main
            unsigned int stage_4 = 0;
            unsigned int work_stage_5 = 0;
            unsigned int m_tile_5 = blockIdx.x;
            unsigned int n_tile_5 = blockIdx.y;
            unsigned int _phase_work_full_5 = 0;
            unsigned int _phase_sfa_full = 0;
            unsigned int _phase_sfb_full = 0;
            unsigned int _phase_k_done_2 = 1;
            #pragma unroll 1
            for (unsigned int _tile_iter_5 = 0; _tile_iter_5 < grid_m * grid_n; _tile_iter_5++) {
                unsigned int inactive_valid_5 = 1;
                #pragma unroll 1
                for (unsigned int _inactive_iter_5 = 0; _inactive_iter_5 < grid_m * grid_n; _inactive_iter_5++) {
                    if (n_tile_5 < (unsigned int)total_tiles[0]) {
                        break;
                    }
                    {
                        mbarrier_wait_hint(work_full_addr + (work_stage_5) * 8, _phase_work_full_5, 10000000);
                    }
                    unsigned int valid_10 = 0;
                    unsigned int next_x_10 = 0;
                    unsigned int next_y_10 = 0;
                    uint32_t _clc_valid_8 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %0, 1, 0, p1;\n\t"
                        "}\n"
                        : "=r"(_clc_valid_8)
                        : "r"(work_response_addr + work_stage_5 * 16 + 0 * 16)
                        : "memory");
                    valid_10 = _clc_valid_8;
                    uint32_t _clc_ctaid_16 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_16)
                        : "r"(work_response_addr + work_stage_5 * 16 + 0 * 16)
                        : "memory");
                    next_x_10 = _clc_ctaid_16;
                    uint32_t _clc_ctaid_17 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_17)
                        : "r"(work_response_addr + work_stage_5 * 16 + 0 * 16)
                        : "memory");
                    next_y_10 = _clc_ctaid_17;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(work_empty_addr + (work_stage_5) * 8);
                    work_stage_5 += 1;
                    if (work_stage_5 == 3) { work_stage_5 = 0; _phase_work_full_5 ^= 1; }
                    inactive_valid_5 = valid_10;
                    m_tile_5 = next_x_10;
                    n_tile_5 = next_y_10;
                    if (inactive_valid_5 == 0) {
                        break;
                    }
                }
                if (inactive_valid_5 == 0) {
                    break;
                }
                if (m_tile_5 >= (unsigned int)grid_m || n_tile_5 >= (unsigned int)grid_n) {
                    break;
                }
                int route_k_extent_4 = K_tiles;
                #pragma unroll 1
                for (int _route_k = 0; _route_k < route_k_extent_4; _route_k++) {
                    mbarrier_wait(sfa_full_addr + (stage_4) * 8, _phase_sfa_full);
                    mbarrier_wait(sfb_full_addr + (stage_4) * 8, _phase_sfb_full);
                    mbarrier_wait(k_done_addr + (stage_4) * 8, _phase_k_done_2);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (elect_sync()) {
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_0 = ((((uint64_t)(smem_sfa_addr + stage_4 * 2048)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                :: "r"((uint32_t)((unsigned int)tmem_sfa + stage_4 * 16)), "l"(_tcgen05_cp_desc_0)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_1 = ((((uint64_t)(smem_sfa_addr + stage_4 * 2048 + 512)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_4 * 16 + 4))), "l"(_tcgen05_cp_desc_1)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_2 = ((((uint64_t)(smem_sfa_addr + stage_4 * 2048 + 1024)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_4 * 16 + 8))), "l"(_tcgen05_cp_desc_2)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_3 = ((((uint64_t)(smem_sfa_addr + stage_4 * 2048 + 1536)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_4 * 16 + 12))), "l"(_tcgen05_cp_desc_3)
                                : "memory");
                        }
                        for (int k_set = 0; k_set < 4; k_set++) {
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                            #endif
                            {
                                uint64_t _tcgen05_cp_desc_4 = ((((uint64_t)(smem_sfb_addr + stage_4 * 2048 + (unsigned int)(k_set * 512))) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL);
                                asm volatile(
                                    "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                    :: "r"((uint32_t)((unsigned int)tmem_sfb + (stage_4 * 16 + (unsigned int)(k_set * 4)))), "l"(_tcgen05_cp_desc_4)
                                    : "memory");
                            }
                        }
                    }
                    elect_commit2(tmem_sfab_full_addr + (stage_4) * 8, sfa_smem_free_addr + (stage_4) * 8);
                    if (elect_sync()) {
                        mbarrier_arrive(sfb_smem_free_addr + (stage_4) * 8);
                    }
                    stage_4 += 1;
                    if (stage_4 == 4) { stage_4 = 0; _phase_sfa_full ^= 1; _phase_sfb_full ^= 1; _phase_k_done_2 ^= 1; }
                }
                {
                    mbarrier_wait_hint(work_full_addr + (work_stage_5) * 8, _phase_work_full_5, 10000000);
                }
                unsigned int valid_11 = 0;
                unsigned int next_x_11 = 0;
                unsigned int next_y_11 = 0;
                uint32_t _clc_valid_9 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_9)
                    : "r"(work_response_addr + work_stage_5 * 16 + 0 * 16)
                    : "memory");
                valid_11 = _clc_valid_9;
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
                next_x_11 = _clc_ctaid_18;
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
                next_y_11 = _clc_ctaid_19;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage_5) * 8);
                work_stage_5 += 1;
                if (work_stage_5 == 3) { work_stage_5 = 0; _phase_work_full_5 ^= 1; }
                unsigned int valid_0_4 = valid_11;
                m_tile_5 = next_x_11;
                n_tile_5 = next_y_11;
                if (valid_0_4 == 0) {
                    break;
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 9) {
        { // mma_main
            unsigned int k_stage = 0;
            unsigned int acc_stage_1 = 0;
            unsigned int work_stage_6 = 0;
            unsigned int k_phase = 0;
            unsigned int a_token = 0;
            unsigned int b_token = 0;
            unsigned int sfa_token = 0;
            unsigned int sfb_token = 0;
            unsigned int m_tile_6 = blockIdx.x;
            unsigned int n_tile_6 = blockIdx.y;
            unsigned int _phase_work_full_6 = 0;
            unsigned int _phase_mma_free = 1;
            unsigned int _phase_a_full = 0;
            unsigned int _phase_b_full = 0;
            unsigned int _phase_tmem_sfab_full = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_6 = 0; _tile_iter_6 < grid_m * grid_n; _tile_iter_6++) {
                unsigned int inactive_valid_6 = 1;
                #pragma unroll 1
                for (unsigned int _inactive_iter_6 = 0; _inactive_iter_6 < grid_m * grid_n; _inactive_iter_6++) {
                    if (n_tile_6 < (unsigned int)total_tiles[0]) {
                        break;
                    }
                    {
                        mbarrier_wait_hint(work_full_addr + (work_stage_6) * 8, _phase_work_full_6, 10000000);
                    }
                    unsigned int valid_12 = 0;
                    unsigned int next_x_12 = 0;
                    unsigned int next_y_12 = 0;
                    uint32_t _clc_valid_10 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %0, 1, 0, p1;\n\t"
                        "}\n"
                        : "=r"(_clc_valid_10)
                        : "r"(work_response_addr + work_stage_6 * 16 + 0 * 16)
                        : "memory");
                    valid_12 = _clc_valid_10;
                    uint32_t _clc_ctaid_20 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_20)
                        : "r"(work_response_addr + work_stage_6 * 16 + 0 * 16)
                        : "memory");
                    next_x_12 = _clc_ctaid_20;
                    uint32_t _clc_ctaid_21 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_21)
                        : "r"(work_response_addr + work_stage_6 * 16 + 0 * 16)
                        : "memory");
                    next_y_12 = _clc_ctaid_21;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(work_empty_addr + (work_stage_6) * 8);
                    work_stage_6 += 1;
                    if (work_stage_6 == 3) { work_stage_6 = 0; _phase_work_full_6 ^= 1; }
                    inactive_valid_6 = valid_12;
                    m_tile_6 = next_x_12;
                    n_tile_6 = next_y_12;
                    if (inactive_valid_6 == 0) {
                        break;
                    }
                }
                if (inactive_valid_6 == 0) {
                    break;
                }
                if (m_tile_6 >= (unsigned int)grid_m || n_tile_6 >= (unsigned int)grid_n) {
                    break;
                }
                mbarrier_wait(mma_free_addr + (acc_stage_1) * 8, _phase_mma_free);
                int route_k_extent_5 = K_tiles;
                #pragma unroll 1
                for (int route_k_4 = 0; route_k_4 < route_k_extent_5; route_k_4++) {
                    int route_slot = ((0) ? route_k_4 / K_tiles : 0);
                    {
                        mbarrier_wait(a_full_addr + (k_stage) * 8, _phase_a_full);
                        mbarrier_wait(b_full_addr + (k_stage) * 8, _phase_b_full);
                        mbarrier_wait(tmem_sfab_full_addr + (k_stage) * 8, _phase_tmem_sfab_full);
                    }
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int sfa_base_col = 0;
                    int sfb_base_col = 0;
                    int _mma_a_lo_0 = make_warp_uniform((((smem_a_mma0_addr) >> 4) & 0x3FFF) + (k_stage) * 1024);
                    int _mma_b_lo_0 = make_warp_uniform((((smem_b_mma0_addr) >> 4) & 0x3FFF) + (k_stage) * 1024);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_accum + ((acc_stage_1 + (unsigned int)route_slot) * 128)), a_desc + 0, b_desc + 0,
                                0x8200480U, (unsigned int)tmem_sfa + (k_stage * 16 + (unsigned int)sfa_base_col) + 0, (unsigned int)tmem_sfb + (k_stage * 16 + (unsigned int)sfb_base_col) + 0, ((((1) ? ((0) ? ((route_k_4 % K_tiles == 0) ? 1 : 0) : ((route_k_4 == 0) ? 1 : 0)) : 0)) ? 0 : 1));
                        }
                    }
                    int sfa_base_col_0 = 4;
                    int sfb_base_col_1 = 4;
                    int _mma_a_lo_1 = make_warp_uniform((((smem_a_mma1_addr) >> 4) & 0x3FFF) + (k_stage) * 1024);
                    int _mma_b_lo_1 = make_warp_uniform((((smem_b_mma1_addr) >> 4) & 0x3FFF) + (k_stage) * 1024);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_accum + ((acc_stage_1 + (unsigned int)route_slot) * 128)), a_desc + 0, b_desc + 0,
                                0x8200480U, (unsigned int)tmem_sfa + (k_stage * 16 + (unsigned int)sfa_base_col_0) + 0, (unsigned int)tmem_sfb + (k_stage * 16 + (unsigned int)sfb_base_col_1) + 0, ((((0) ? ((0) ? ((route_k_4 % K_tiles == 0) ? 1 : 0) : ((route_k_4 == 0) ? 1 : 0)) : 0)) ? 0 : 1));
                        }
                    }
                    int sfa_base_col_2 = 8;
                    int sfb_base_col_3 = 8;
                    int _mma_a_lo_2 = make_warp_uniform((((smem_a_mma2_addr) >> 4) & 0x3FFF) + (k_stage) * 1024);
                    int _mma_b_lo_2 = make_warp_uniform((((smem_b_mma2_addr) >> 4) & 0x3FFF) + (k_stage) * 1024);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_2) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_2) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_accum + ((acc_stage_1 + (unsigned int)route_slot) * 128)), a_desc + 0, b_desc + 0,
                                0x8200480U, (unsigned int)tmem_sfa + (k_stage * 16 + (unsigned int)sfa_base_col_2) + 0, (unsigned int)tmem_sfb + (k_stage * 16 + (unsigned int)sfb_base_col_3) + 0, ((((0) ? ((0) ? ((route_k_4 % K_tiles == 0) ? 1 : 0) : ((route_k_4 == 0) ? 1 : 0)) : 0)) ? 0 : 1));
                        }
                    }
                    int sfa_base_col_4 = 12;
                    int sfb_base_col_5 = 12;
                    int _mma_a_lo_3 = make_warp_uniform((((smem_a_mma3_addr) >> 4) & 0x3FFF) + (k_stage) * 1024);
                    int _mma_b_lo_3 = make_warp_uniform((((smem_b_mma3_addr) >> 4) & 0x3FFF) + (k_stage) * 1024);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_3) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_3) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_accum + ((acc_stage_1 + (unsigned int)route_slot) * 128)), a_desc + 0, b_desc + 0,
                                0x8200480U, (unsigned int)tmem_sfa + (k_stage * 16 + (unsigned int)sfa_base_col_4) + 0, (unsigned int)tmem_sfb + (k_stage * 16 + (unsigned int)sfb_base_col_5) + 0, ((((0) ? ((0) ? ((route_k_4 % K_tiles == 0) ? 1 : 0) : ((route_k_4 == 0) ? 1 : 0)) : 0)) ? 0 : 1));
                        }
                    }
                    if (route_k_4 + 1 == route_k_extent_5) {
                        elect_commit2(k_done_addr + (k_stage) * 8, mma_full_addr + (acc_stage_1) * 8);
                    } else {
                        elect_commit(k_done_addr + (k_stage) * 8);
                    }
                    {
                        k_stage += 1;
                        if (k_stage == 4) { k_stage = 0; _phase_a_full ^= 1; _phase_b_full ^= 1; _phase_tmem_sfab_full ^= 1; }
                    }
                }
                acc_stage_1 += 1;
                if (acc_stage_1 == 2) { acc_stage_1 = 0; _phase_mma_free ^= 1; }
                {
                    mbarrier_wait_hint(work_full_addr + (work_stage_6) * 8, _phase_work_full_6, 10000000);
                }
                unsigned int valid_13 = 0;
                unsigned int next_x_13 = 0;
                unsigned int next_y_13 = 0;
                uint32_t _clc_valid_11 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_11)
                    : "r"(work_response_addr + work_stage_6 * 16 + 0 * 16)
                    : "memory");
                valid_13 = _clc_valid_11;
                uint32_t _clc_ctaid_22 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_22)
                    : "r"(work_response_addr + work_stage_6 * 16 + 0 * 16)
                    : "memory");
                next_x_13 = _clc_ctaid_22;
                uint32_t _clc_ctaid_23 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_23)
                    : "r"(work_response_addr + work_stage_6 * 16 + 0 * 16)
                    : "memory");
                next_y_13 = _clc_ctaid_23;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage_6) * 8);
                work_stage_6 += 1;
                if (work_stage_6 == 3) { work_stage_6 = 0; _phase_work_full_6 ^= 1; }
                unsigned int valid_0_5 = valid_13;
                m_tile_6 = next_x_13;
                n_tile_6 = next_y_13;
                if (valid_0_5 == 0) {
                    break;
                }
            }
        }
    }
    // ---- Role: work_id ----
    if (warp == 10) {
        { // work_id_main
            unsigned int work_stage_7 = 0;
            unsigned int throttle_stage_1 = 0;
            unsigned int m_tile_7 = blockIdx.x;
            unsigned int n_tile_7 = blockIdx.y;
            {
                asm volatile("griddepcontrol.wait;" ::: "memory");
            }
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_work_full_7 = 0;
            unsigned int _phase_throttle_full = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_7 = 0; _tile_iter_7 < grid_m * grid_n; _tile_iter_7++) {
                unsigned int inactive_valid_7 = 1;
                #pragma unroll 1
                for (unsigned int _inactive_iter_7 = 0; _inactive_iter_7 < grid_m * grid_n; _inactive_iter_7++) {
                    if (n_tile_7 < (unsigned int)total_tiles[0]) {
                        break;
                    }
                    if (elect_sync()) {
                        mbarrier_wait(work_empty_addr + (work_stage_7) * 8, _phase_work_empty);
                        mbarrier_arrive_expect_tx(work_full_addr + (work_stage_7) * 8, 16);
                        asm volatile(
                            "fence.proxy.async.shared::cta;\n\t"
                            "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                ".mbarrier::complete_tx::bytes.b128"
                                " [%0], [%1];"
                            :: "r"(work_response_addr + work_stage_7 * 16 + 0 * 16), "r"(work_full_addr + work_stage_7 * 8)
                            : "memory");
                    }
                    {
                        mbarrier_wait_hint(work_full_addr + (work_stage_7) * 8, _phase_work_full_7, 10000000);
                    }
                    unsigned int valid_14 = 0;
                    unsigned int next_x_14 = 0;
                    unsigned int next_y_14 = 0;
                    uint32_t _clc_valid_16 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %0, 1, 0, p1;\n\t"
                        "}\n"
                        : "=r"(_clc_valid_16)
                        : "r"(work_response_addr + work_stage_7 * 16 + 0 * 16)
                        : "memory");
                    valid_14 = _clc_valid_16;
                    uint32_t _clc_ctaid_32 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_32)
                        : "r"(work_response_addr + work_stage_7 * 16 + 0 * 16)
                        : "memory");
                    next_x_14 = _clc_ctaid_32;
                    uint32_t _clc_ctaid_33 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_33)
                        : "r"(work_response_addr + work_stage_7 * 16 + 0 * 16)
                        : "memory");
                    next_y_14 = _clc_ctaid_33;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(work_empty_addr + (work_stage_7) * 8);
                    work_stage_7 += 1;
                    if (work_stage_7 == 3) { work_stage_7 = 0; _phase_work_empty ^= 1; _phase_work_full_7 ^= 1; }
                    inactive_valid_7 = valid_14;
                    m_tile_7 = next_x_14;
                    n_tile_7 = next_y_14;
                    if (inactive_valid_7 == 0) {
                        break;
                    }
                }
                if (inactive_valid_7 == 0) {
                    break;
                }
                if (m_tile_7 >= (unsigned int)grid_m || n_tile_7 >= (unsigned int)grid_n) {
                    break;
                }
                mbarrier_wait(throttle_full_addr + (throttle_stage_1) * 8, _phase_throttle_full);
                mbarrier_arrive(throttle_empty_addr + (throttle_stage_1) * 8);
                throttle_stage_1 += 1;
                if (throttle_stage_1 == 3) { throttle_stage_1 = 0; _phase_throttle_full ^= 1; }
                if (elect_sync()) {
                    mbarrier_wait(work_empty_addr + (work_stage_7) * 8, _phase_work_empty);
                    mbarrier_arrive_expect_tx(work_full_addr + (work_stage_7) * 8, 16);
                    asm volatile(
                        "fence.proxy.async.shared::cta;\n\t"
                        "clusterlaunchcontrol.try_cancel.async.shared::cta"
                            ".mbarrier::complete_tx::bytes.b128"
                            " [%0], [%1];"
                        :: "r"(work_response_addr + work_stage_7 * 16 + 0 * 16), "r"(work_full_addr + work_stage_7 * 8)
                        : "memory");
                }
                {
                    mbarrier_wait_hint(work_full_addr + (work_stage_7) * 8, _phase_work_full_7, 10000000);
                }
                unsigned int valid_15 = 0;
                unsigned int next_x_15 = 0;
                unsigned int next_y_15 = 0;
                uint32_t _clc_valid_17 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_17)
                    : "r"(work_response_addr + work_stage_7 * 16 + 0 * 16)
                    : "memory");
                valid_15 = _clc_valid_17;
                uint32_t _clc_ctaid_34 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_34)
                    : "r"(work_response_addr + work_stage_7 * 16 + 0 * 16)
                    : "memory");
                next_x_15 = _clc_ctaid_34;
                uint32_t _clc_ctaid_35 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_35)
                    : "r"(work_response_addr + work_stage_7 * 16 + 0 * 16)
                    : "memory");
                next_y_15 = _clc_ctaid_35;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage_7) * 8);
                work_stage_7 += 1;
                if (work_stage_7 == 3) { work_stage_7 = 0; _phase_work_empty ^= 1; _phase_work_full_7 ^= 1; }
                unsigned int valid_0_6 = valid_15;
                m_tile_7 = next_x_15;
                n_tile_7 = next_y_15;
                if (valid_0_6 == 0) {
                    break;
                }
            }
        }
    }
    // ---- Role: padding ----
    if (warp == 11) {
        { // padding_main
            unsigned int work_stage_8 = 0;
            unsigned int m_tile_8 = blockIdx.x;
            unsigned int n_tile_8 = blockIdx.y;
            unsigned int _phase_work_full_8 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_8 = 0; _tile_iter_8 < grid_m * grid_n; _tile_iter_8++) {
                unsigned int inactive_valid_8 = 1;
                #pragma unroll 1
                for (unsigned int _inactive_iter_8 = 0; _inactive_iter_8 < grid_m * grid_n; _inactive_iter_8++) {
                    if (n_tile_8 < (unsigned int)total_tiles[0]) {
                        break;
                    }
                    {
                        mbarrier_wait_hint(work_full_addr + (work_stage_8) * 8, _phase_work_full_8, 10000000);
                    }
                    unsigned int valid_16 = 0;
                    unsigned int next_x_16 = 0;
                    unsigned int next_y_16 = 0;
                    uint32_t _clc_valid_14 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %0, 1, 0, p1;\n\t"
                        "}\n"
                        : "=r"(_clc_valid_14)
                        : "r"(work_response_addr + work_stage_8 * 16 + 0 * 16)
                        : "memory");
                    valid_16 = _clc_valid_14;
                    uint32_t _clc_ctaid_28 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_28)
                        : "r"(work_response_addr + work_stage_8 * 16 + 0 * 16)
                        : "memory");
                    next_x_16 = _clc_ctaid_28;
                    uint32_t _clc_ctaid_29 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_29)
                        : "r"(work_response_addr + work_stage_8 * 16 + 0 * 16)
                        : "memory");
                    next_y_16 = _clc_ctaid_29;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(work_empty_addr + (work_stage_8) * 8);
                    work_stage_8 += 1;
                    if (work_stage_8 == 3) { work_stage_8 = 0; _phase_work_full_8 ^= 1; }
                    inactive_valid_8 = valid_16;
                    m_tile_8 = next_x_16;
                    n_tile_8 = next_y_16;
                    if (inactive_valid_8 == 0) {
                        break;
                    }
                }
                if (inactive_valid_8 == 0) {
                    break;
                }
                if (m_tile_8 >= (unsigned int)grid_m || n_tile_8 >= (unsigned int)grid_n) {
                    break;
                }
                {
                    mbarrier_wait_hint(work_full_addr + (work_stage_8) * 8, _phase_work_full_8, 10000000);
                }
                unsigned int valid_17 = 0;
                unsigned int next_x_17 = 0;
                unsigned int next_y_17 = 0;
                uint32_t _clc_valid_15 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_15)
                    : "r"(work_response_addr + work_stage_8 * 16 + 0 * 16)
                    : "memory");
                valid_17 = _clc_valid_15;
                uint32_t _clc_ctaid_30 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_30)
                    : "r"(work_response_addr + work_stage_8 * 16 + 0 * 16)
                    : "memory");
                next_x_17 = _clc_ctaid_30;
                uint32_t _clc_ctaid_31 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_31)
                    : "r"(work_response_addr + work_stage_8 * 16 + 0 * 16)
                    : "memory");
                next_y_17 = _clc_ctaid_31;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage_8) * 8);
                work_stage_8 += 1;
                if (work_stage_8 == 3) { work_stage_8 = 0; _phase_work_full_8 ^= 1; }
                unsigned int valid_0_7 = valid_17;
                m_tile_8 = next_x_17;
                n_tile_8 = next_y_17;
                if (valid_0_7 == 0) {
                    break;
                }
            }
        }
    }

    // Cleanup
    __syncthreads(); // barrier before TMEM dealloc

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
