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
#define NUM_WORK_PIPE_STAGES 5
#define NUM_THROTTLE_PIPE_STAGES 5
#define NUM_DRAIN_PIPE_STAGES 1
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
#define SMEM_EPI_STAGING_U32_OFF 132096
#define SMEM_EPI_STAGING_U32_STAGE_BYTES 32768
#define SMEM_EPI_STAGING_U32_STRIDE 32768
#define SMEM_AMAX_SLOTS_OFF 164928
#define SMEM_AMAX_SLOTS_STAGE_BYTES 16
#define SMEM_AMAX_SLOTS_STRIDE 16
#define SMEM_WORK_RESPONSE_OFF 184320
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_DRAIN_RESPONSE_OFF 164864
#define SMEM_DRAIN_RESPONSE_STAGE_BYTES 64
#define SMEM_DRAIN_RESPONSE_STRIDE 64
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

__global__ __launch_bounds__(384, 1) void
kernel_cake_kimi_k3_nvfp4_situ_routed_moe_ad5623664c0619d269a9(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, const __grid_constant__ CUtensorMap C_tma, uint8_t* __restrict__ C, float* __restrict__ scale_c, int* __restrict__ tile_expert, int* __restrict__ tile_mn_limit, float* __restrict__ partial_scale, int M, int K, int grid_m, int grid_n, int K_tiles, int* __restrict__ total_tiles)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane = static_cast<uint32_t>(tid) & 31u;

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
    #define work_empty_addr (mbar_base + 328)
    #define throttle_full_addr (mbar_base + 368)
    #define throttle_empty_addr (mbar_base + 408)
    #define drain_full_addr (mbar_base + 448)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;

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
    unsigned int* epi_staging_u32 = reinterpret_cast<unsigned int*>(smem_raw + 132096);
    const int epi_staging_u32_addr = smem + 132096;
    float* amax_slots = reinterpret_cast<float*>(smem_raw + 164928);
    const int amax_slots_addr = smem + 164928;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 184320);
    const int work_response_addr = smem + 184320;
    unsigned int* drain_response = reinterpret_cast<unsigned int*>(smem_raw + 164864);
    const int drain_response_addr = smem + 164864;
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if ((int)blockIdx.y >= total_tiles[0]) return;

    // Mbarrier init (15 pipeline groups, 0 ordered-sequence groups, 57 barriers)
    // Mbarriers at smem_raw[0..456)

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
            // work_full: 5 barriers, init_count=1
            mbarrier_init(smem + 288, 1);
            mbarrier_init(smem + 296, 1);
            mbarrier_init(smem + 304, 1);
            mbarrier_init(smem + 312, 1);
            mbarrier_init(smem + 320, 1);
            // work_empty: 5 barriers, init_count=384
            mbarrier_init(smem + 328, 384);
            mbarrier_init(smem + 336, 384);
            mbarrier_init(smem + 344, 384);
            mbarrier_init(smem + 352, 384);
            mbarrier_init(smem + 360, 384);
            // --- pipeline 'throttle_pipe' ---
            // throttle_full: 5 barriers, init_count=32
            mbarrier_init(smem + 368, 32);
            mbarrier_init(smem + 376, 32);
            mbarrier_init(smem + 384, 32);
            mbarrier_init(smem + 392, 32);
            mbarrier_init(smem + 400, 32);
            // throttle_empty: 5 barriers, init_count=32
            mbarrier_init(smem + 408, 32);
            mbarrier_init(smem + 416, 32);
            mbarrier_init(smem + 424, 32);
            mbarrier_init(smem + 432, 32);
            mbarrier_init(smem + 440, 32);
            // --- pipeline 'drain_pipe' ---
            // drain_full: 1 barriers, init_count=1
            mbarrier_init(smem + 448, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 384 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 456);
    if (warp == 0) {
        int _tmem_hold = smem + 456;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

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
                        mbarrier_wait_hint(work_full_addr + (work_stage) * 8, _phase_work_full, 1000);
                    }
                    unsigned int valid = 0;
                    unsigned int next_x = 0;
                    unsigned int next_y = 0;
                    int response_index_fd = work_stage * 4;
                    valid = work_response[response_index_fd];
                    next_x = work_response[response_index_fd + 1];
                    next_y = work_response[response_index_fd + 2];
                    mbarrier_arrive(work_empty_addr + (work_stage) * 8);
                    work_stage += 1;
                    if (work_stage == 5) { work_stage = 0; _phase_work_full ^= 1; }
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
                int valid_tokens = _mn_limit - token_base;
                float tmax = 0.0f;
                int amax_token = wide_token;
                if (amax_token < valid_tokens) {
                    {
                        float _fabs_0 = fabsf(_tmem_load_0[0]);
                        float _fabs_1 = fabsf(_tmem_load_0[2]);
                        float _max3_0;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_0) : "f"(tmax), "f"(_fabs_0), "f"(_fabs_1));
                        tmax = _max3_0;
                        float _fabs_2 = fabsf(_tmem_load_2[0]);
                        float _fabs_3 = fabsf(_tmem_load_2[2]);
                        float _max3_1;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_1) : "f"(tmax), "f"(_fabs_2), "f"(_fabs_3));
                        tmax = _max3_1;
                    }
                }
                int amax_token_0 = wide_token + 1;
                if (amax_token_0 < valid_tokens) {
                    {
                        float _fabs_8 = fabsf(_tmem_load_0[1]);
                        float _fabs_9 = fabsf(_tmem_load_0[3]);
                        float _max3_4;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_4) : "f"(tmax), "f"(_fabs_8), "f"(_fabs_9));
                        tmax = _max3_4;
                        float _fabs_10 = fabsf(_tmem_load_2[1]);
                        float _fabs_11 = fabsf(_tmem_load_2[3]);
                        float _max3_5;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_5) : "f"(tmax), "f"(_fabs_10), "f"(_fabs_11));
                        tmax = _max3_5;
                    }
                }
                int amax_token_1 = wide_token + 8;
                if (amax_token_1 < valid_tokens) {
                    {
                        float _fabs_16 = fabsf(_tmem_load_0[4]);
                        float _fabs_17 = fabsf(_tmem_load_0[6]);
                        float _max3_8;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_8) : "f"(tmax), "f"(_fabs_16), "f"(_fabs_17));
                        tmax = _max3_8;
                        float _fabs_18 = fabsf(_tmem_load_2[4]);
                        float _fabs_19 = fabsf(_tmem_load_2[6]);
                        float _max3_9;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_9) : "f"(tmax), "f"(_fabs_18), "f"(_fabs_19));
                        tmax = _max3_9;
                    }
                }
                int amax_token_2 = wide_token + 8 + 1;
                if (amax_token_2 < valid_tokens) {
                    {
                        float _fabs_24 = fabsf(_tmem_load_0[5]);
                        float _fabs_25 = fabsf(_tmem_load_0[7]);
                        float _max3_12;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_12) : "f"(tmax), "f"(_fabs_24), "f"(_fabs_25));
                        tmax = _max3_12;
                        float _fabs_26 = fabsf(_tmem_load_2[5]);
                        float _fabs_27 = fabsf(_tmem_load_2[7]);
                        float _max3_13;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_13) : "f"(tmax), "f"(_fabs_26), "f"(_fabs_27));
                        tmax = _max3_13;
                    }
                }
                int amax_token_3 = wide_token + 16;
                if (amax_token_3 < valid_tokens) {
                    {
                        float _fabs_32 = fabsf(_tmem_load_0[8]);
                        float _fabs_33 = fabsf(_tmem_load_0[10]);
                        float _max3_16;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_16) : "f"(tmax), "f"(_fabs_32), "f"(_fabs_33));
                        tmax = _max3_16;
                        float _fabs_34 = fabsf(_tmem_load_2[8]);
                        float _fabs_35 = fabsf(_tmem_load_2[10]);
                        float _max3_17;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_17) : "f"(tmax), "f"(_fabs_34), "f"(_fabs_35));
                        tmax = _max3_17;
                    }
                }
                int amax_token_4 = wide_token + 16 + 1;
                if (amax_token_4 < valid_tokens) {
                    {
                        float _fabs_40 = fabsf(_tmem_load_0[9]);
                        float _fabs_41 = fabsf(_tmem_load_0[11]);
                        float _max3_20;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_20) : "f"(tmax), "f"(_fabs_40), "f"(_fabs_41));
                        tmax = _max3_20;
                        float _fabs_42 = fabsf(_tmem_load_2[9]);
                        float _fabs_43 = fabsf(_tmem_load_2[11]);
                        float _max3_21;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_21) : "f"(tmax), "f"(_fabs_42), "f"(_fabs_43));
                        tmax = _max3_21;
                    }
                }
                int amax_token_5 = wide_token + 24;
                if (amax_token_5 < valid_tokens) {
                    {
                        float _fabs_48 = fabsf(_tmem_load_0[12]);
                        float _fabs_49 = fabsf(_tmem_load_0[14]);
                        float _max3_24;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_24) : "f"(tmax), "f"(_fabs_48), "f"(_fabs_49));
                        tmax = _max3_24;
                        float _fabs_50 = fabsf(_tmem_load_2[12]);
                        float _fabs_51 = fabsf(_tmem_load_2[14]);
                        float _max3_25;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_25) : "f"(tmax), "f"(_fabs_50), "f"(_fabs_51));
                        tmax = _max3_25;
                    }
                }
                int amax_token_6 = wide_token + 24 + 1;
                if (amax_token_6 < valid_tokens) {
                    {
                        float _fabs_56 = fabsf(_tmem_load_0[13]);
                        float _fabs_57 = fabsf(_tmem_load_0[15]);
                        float _max3_28;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_28) : "f"(tmax), "f"(_fabs_56), "f"(_fabs_57));
                        tmax = _max3_28;
                        float _fabs_58 = fabsf(_tmem_load_2[13]);
                        float _fabs_59 = fabsf(_tmem_load_2[15]);
                        float _max3_29;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_29) : "f"(tmax), "f"(_fabs_58), "f"(_fabs_59));
                        tmax = _max3_29;
                    }
                }
                int amax_token_7 = wide_token + 32;
                if (amax_token_7 < valid_tokens) {
                    {
                        float _fabs_64 = fabsf(_tmem_load_0[16]);
                        float _fabs_65 = fabsf(_tmem_load_0[18]);
                        float _max3_32;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_32) : "f"(tmax), "f"(_fabs_64), "f"(_fabs_65));
                        tmax = _max3_32;
                        float _fabs_66 = fabsf(_tmem_load_2[16]);
                        float _fabs_67 = fabsf(_tmem_load_2[18]);
                        float _max3_33;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_33) : "f"(tmax), "f"(_fabs_66), "f"(_fabs_67));
                        tmax = _max3_33;
                    }
                }
                int amax_token_8 = wide_token + 32 + 1;
                if (amax_token_8 < valid_tokens) {
                    {
                        float _fabs_72 = fabsf(_tmem_load_0[17]);
                        float _fabs_73 = fabsf(_tmem_load_0[19]);
                        float _max3_36;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_36) : "f"(tmax), "f"(_fabs_72), "f"(_fabs_73));
                        tmax = _max3_36;
                        float _fabs_74 = fabsf(_tmem_load_2[17]);
                        float _fabs_75 = fabsf(_tmem_load_2[19]);
                        float _max3_37;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_37) : "f"(tmax), "f"(_fabs_74), "f"(_fabs_75));
                        tmax = _max3_37;
                    }
                }
                int amax_token_9 = wide_token + 40;
                if (amax_token_9 < valid_tokens) {
                    {
                        float _fabs_80 = fabsf(_tmem_load_0[20]);
                        float _fabs_81 = fabsf(_tmem_load_0[22]);
                        float _max3_40;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_40) : "f"(tmax), "f"(_fabs_80), "f"(_fabs_81));
                        tmax = _max3_40;
                        float _fabs_82 = fabsf(_tmem_load_2[20]);
                        float _fabs_83 = fabsf(_tmem_load_2[22]);
                        float _max3_41;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_41) : "f"(tmax), "f"(_fabs_82), "f"(_fabs_83));
                        tmax = _max3_41;
                    }
                }
                int amax_token_10 = wide_token + 40 + 1;
                if (amax_token_10 < valid_tokens) {
                    {
                        float _fabs_88 = fabsf(_tmem_load_0[21]);
                        float _fabs_89 = fabsf(_tmem_load_0[23]);
                        float _max3_44;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_44) : "f"(tmax), "f"(_fabs_88), "f"(_fabs_89));
                        tmax = _max3_44;
                        float _fabs_90 = fabsf(_tmem_load_2[21]);
                        float _fabs_91 = fabsf(_tmem_load_2[23]);
                        float _max3_45;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_45) : "f"(tmax), "f"(_fabs_90), "f"(_fabs_91));
                        tmax = _max3_45;
                    }
                }
                int amax_token_11 = wide_token + 48;
                if (amax_token_11 < valid_tokens) {
                    {
                        float _fabs_96 = fabsf(_tmem_load_0[24]);
                        float _fabs_97 = fabsf(_tmem_load_0[26]);
                        float _max3_48;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_48) : "f"(tmax), "f"(_fabs_96), "f"(_fabs_97));
                        tmax = _max3_48;
                        float _fabs_98 = fabsf(_tmem_load_2[24]);
                        float _fabs_99 = fabsf(_tmem_load_2[26]);
                        float _max3_49;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_49) : "f"(tmax), "f"(_fabs_98), "f"(_fabs_99));
                        tmax = _max3_49;
                    }
                }
                int amax_token_12 = wide_token + 48 + 1;
                if (amax_token_12 < valid_tokens) {
                    {
                        float _fabs_104 = fabsf(_tmem_load_0[25]);
                        float _fabs_105 = fabsf(_tmem_load_0[27]);
                        float _max3_52;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_52) : "f"(tmax), "f"(_fabs_104), "f"(_fabs_105));
                        tmax = _max3_52;
                        float _fabs_106 = fabsf(_tmem_load_2[25]);
                        float _fabs_107 = fabsf(_tmem_load_2[27]);
                        float _max3_53;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_53) : "f"(tmax), "f"(_fabs_106), "f"(_fabs_107));
                        tmax = _max3_53;
                    }
                }
                int amax_token_13 = wide_token + 56;
                if (amax_token_13 < valid_tokens) {
                    {
                        float _fabs_112 = fabsf(_tmem_load_0[28]);
                        float _fabs_113 = fabsf(_tmem_load_0[30]);
                        float _max3_56;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_56) : "f"(tmax), "f"(_fabs_112), "f"(_fabs_113));
                        tmax = _max3_56;
                        float _fabs_114 = fabsf(_tmem_load_2[28]);
                        float _fabs_115 = fabsf(_tmem_load_2[30]);
                        float _max3_57;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_57) : "f"(tmax), "f"(_fabs_114), "f"(_fabs_115));
                        tmax = _max3_57;
                    }
                }
                int amax_token_14 = wide_token + 56 + 1;
                if (amax_token_14 < valid_tokens) {
                    {
                        float _fabs_120 = fabsf(_tmem_load_0[29]);
                        float _fabs_121 = fabsf(_tmem_load_0[31]);
                        float _max3_60;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_60) : "f"(tmax), "f"(_fabs_120), "f"(_fabs_121));
                        tmax = _max3_60;
                        float _fabs_122 = fabsf(_tmem_load_2[29]);
                        float _fabs_123 = fabsf(_tmem_load_2[31]);
                        float _max3_61;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_61) : "f"(tmax), "f"(_fabs_122), "f"(_fabs_123));
                        tmax = _max3_61;
                    }
                }
                int amax_token_15 = wide_token + 64;
                if (amax_token_15 < valid_tokens) {
                    {
                        float _fabs_132 = fabsf(_tmem_load_1[0]);
                        float _fabs_133 = fabsf(_tmem_load_1[2]);
                        float _max3_66;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_66) : "f"(tmax), "f"(_fabs_132), "f"(_fabs_133));
                        tmax = _max3_66;
                        float _fabs_134 = fabsf(_tmem_load_3[0]);
                        float _fabs_135 = fabsf(_tmem_load_3[2]);
                        float _max3_67;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_67) : "f"(tmax), "f"(_fabs_134), "f"(_fabs_135));
                        tmax = _max3_67;
                    }
                }
                int amax_token_16 = wide_token + 64 + 1;
                if (amax_token_16 < valid_tokens) {
                    {
                        float _fabs_140 = fabsf(_tmem_load_1[1]);
                        float _fabs_141 = fabsf(_tmem_load_1[3]);
                        float _max3_70;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_70) : "f"(tmax), "f"(_fabs_140), "f"(_fabs_141));
                        tmax = _max3_70;
                        float _fabs_142 = fabsf(_tmem_load_3[1]);
                        float _fabs_143 = fabsf(_tmem_load_3[3]);
                        float _max3_71;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_71) : "f"(tmax), "f"(_fabs_142), "f"(_fabs_143));
                        tmax = _max3_71;
                    }
                }
                int amax_token_17 = wide_token + 64 + 8;
                if (amax_token_17 < valid_tokens) {
                    {
                        float _fabs_148 = fabsf(_tmem_load_1[4]);
                        float _fabs_149 = fabsf(_tmem_load_1[6]);
                        float _max3_74;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_74) : "f"(tmax), "f"(_fabs_148), "f"(_fabs_149));
                        tmax = _max3_74;
                        float _fabs_150 = fabsf(_tmem_load_3[4]);
                        float _fabs_151 = fabsf(_tmem_load_3[6]);
                        float _max3_75;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_75) : "f"(tmax), "f"(_fabs_150), "f"(_fabs_151));
                        tmax = _max3_75;
                    }
                }
                int amax_token_18 = wide_token + 64 + 8 + 1;
                if (amax_token_18 < valid_tokens) {
                    {
                        float _fabs_156 = fabsf(_tmem_load_1[5]);
                        float _fabs_157 = fabsf(_tmem_load_1[7]);
                        float _max3_78;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_78) : "f"(tmax), "f"(_fabs_156), "f"(_fabs_157));
                        tmax = _max3_78;
                        float _fabs_158 = fabsf(_tmem_load_3[5]);
                        float _fabs_159 = fabsf(_tmem_load_3[7]);
                        float _max3_79;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_79) : "f"(tmax), "f"(_fabs_158), "f"(_fabs_159));
                        tmax = _max3_79;
                    }
                }
                int amax_token_19 = wide_token + 64 + 16;
                if (amax_token_19 < valid_tokens) {
                    {
                        float _fabs_164 = fabsf(_tmem_load_1[8]);
                        float _fabs_165 = fabsf(_tmem_load_1[10]);
                        float _max3_82;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_82) : "f"(tmax), "f"(_fabs_164), "f"(_fabs_165));
                        tmax = _max3_82;
                        float _fabs_166 = fabsf(_tmem_load_3[8]);
                        float _fabs_167 = fabsf(_tmem_load_3[10]);
                        float _max3_83;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_83) : "f"(tmax), "f"(_fabs_166), "f"(_fabs_167));
                        tmax = _max3_83;
                    }
                }
                int amax_token_20 = wide_token + 64 + 16 + 1;
                if (amax_token_20 < valid_tokens) {
                    {
                        float _fabs_172 = fabsf(_tmem_load_1[9]);
                        float _fabs_173 = fabsf(_tmem_load_1[11]);
                        float _max3_86;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_86) : "f"(tmax), "f"(_fabs_172), "f"(_fabs_173));
                        tmax = _max3_86;
                        float _fabs_174 = fabsf(_tmem_load_3[9]);
                        float _fabs_175 = fabsf(_tmem_load_3[11]);
                        float _max3_87;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_87) : "f"(tmax), "f"(_fabs_174), "f"(_fabs_175));
                        tmax = _max3_87;
                    }
                }
                int amax_token_21 = wide_token + 64 + 24;
                if (amax_token_21 < valid_tokens) {
                    {
                        float _fabs_180 = fabsf(_tmem_load_1[12]);
                        float _fabs_181 = fabsf(_tmem_load_1[14]);
                        float _max3_90;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_90) : "f"(tmax), "f"(_fabs_180), "f"(_fabs_181));
                        tmax = _max3_90;
                        float _fabs_182 = fabsf(_tmem_load_3[12]);
                        float _fabs_183 = fabsf(_tmem_load_3[14]);
                        float _max3_91;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_91) : "f"(tmax), "f"(_fabs_182), "f"(_fabs_183));
                        tmax = _max3_91;
                    }
                }
                int amax_token_22 = wide_token + 64 + 24 + 1;
                if (amax_token_22 < valid_tokens) {
                    {
                        float _fabs_188 = fabsf(_tmem_load_1[13]);
                        float _fabs_189 = fabsf(_tmem_load_1[15]);
                        float _max3_94;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_94) : "f"(tmax), "f"(_fabs_188), "f"(_fabs_189));
                        tmax = _max3_94;
                        float _fabs_190 = fabsf(_tmem_load_3[13]);
                        float _fabs_191 = fabsf(_tmem_load_3[15]);
                        float _max3_95;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_95) : "f"(tmax), "f"(_fabs_190), "f"(_fabs_191));
                        tmax = _max3_95;
                    }
                }
                int amax_token_23 = wide_token + 64 + 32;
                if (amax_token_23 < valid_tokens) {
                    {
                        float _fabs_196 = fabsf(_tmem_load_1[16]);
                        float _fabs_197 = fabsf(_tmem_load_1[18]);
                        float _max3_98;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_98) : "f"(tmax), "f"(_fabs_196), "f"(_fabs_197));
                        tmax = _max3_98;
                        float _fabs_198 = fabsf(_tmem_load_3[16]);
                        float _fabs_199 = fabsf(_tmem_load_3[18]);
                        float _max3_99;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_99) : "f"(tmax), "f"(_fabs_198), "f"(_fabs_199));
                        tmax = _max3_99;
                    }
                }
                int amax_token_24 = wide_token + 64 + 32 + 1;
                if (amax_token_24 < valid_tokens) {
                    {
                        float _fabs_204 = fabsf(_tmem_load_1[17]);
                        float _fabs_205 = fabsf(_tmem_load_1[19]);
                        float _max3_102;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_102) : "f"(tmax), "f"(_fabs_204), "f"(_fabs_205));
                        tmax = _max3_102;
                        float _fabs_206 = fabsf(_tmem_load_3[17]);
                        float _fabs_207 = fabsf(_tmem_load_3[19]);
                        float _max3_103;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_103) : "f"(tmax), "f"(_fabs_206), "f"(_fabs_207));
                        tmax = _max3_103;
                    }
                }
                int amax_token_25 = wide_token + 64 + 40;
                if (amax_token_25 < valid_tokens) {
                    {
                        float _fabs_212 = fabsf(_tmem_load_1[20]);
                        float _fabs_213 = fabsf(_tmem_load_1[22]);
                        float _max3_106;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_106) : "f"(tmax), "f"(_fabs_212), "f"(_fabs_213));
                        tmax = _max3_106;
                        float _fabs_214 = fabsf(_tmem_load_3[20]);
                        float _fabs_215 = fabsf(_tmem_load_3[22]);
                        float _max3_107;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_107) : "f"(tmax), "f"(_fabs_214), "f"(_fabs_215));
                        tmax = _max3_107;
                    }
                }
                int amax_token_26 = wide_token + 64 + 40 + 1;
                if (amax_token_26 < valid_tokens) {
                    {
                        float _fabs_220 = fabsf(_tmem_load_1[21]);
                        float _fabs_221 = fabsf(_tmem_load_1[23]);
                        float _max3_110;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_110) : "f"(tmax), "f"(_fabs_220), "f"(_fabs_221));
                        tmax = _max3_110;
                        float _fabs_222 = fabsf(_tmem_load_3[21]);
                        float _fabs_223 = fabsf(_tmem_load_3[23]);
                        float _max3_111;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_111) : "f"(tmax), "f"(_fabs_222), "f"(_fabs_223));
                        tmax = _max3_111;
                    }
                }
                int amax_token_27 = wide_token + 64 + 48;
                if (amax_token_27 < valid_tokens) {
                    {
                        float _fabs_228 = fabsf(_tmem_load_1[24]);
                        float _fabs_229 = fabsf(_tmem_load_1[26]);
                        float _max3_114;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_114) : "f"(tmax), "f"(_fabs_228), "f"(_fabs_229));
                        tmax = _max3_114;
                        float _fabs_230 = fabsf(_tmem_load_3[24]);
                        float _fabs_231 = fabsf(_tmem_load_3[26]);
                        float _max3_115;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_115) : "f"(tmax), "f"(_fabs_230), "f"(_fabs_231));
                        tmax = _max3_115;
                    }
                }
                int amax_token_28 = wide_token + 64 + 48 + 1;
                if (amax_token_28 < valid_tokens) {
                    {
                        float _fabs_236 = fabsf(_tmem_load_1[25]);
                        float _fabs_237 = fabsf(_tmem_load_1[27]);
                        float _max3_118;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_118) : "f"(tmax), "f"(_fabs_236), "f"(_fabs_237));
                        tmax = _max3_118;
                        float _fabs_238 = fabsf(_tmem_load_3[25]);
                        float _fabs_239 = fabsf(_tmem_load_3[27]);
                        float _max3_119;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_119) : "f"(tmax), "f"(_fabs_238), "f"(_fabs_239));
                        tmax = _max3_119;
                    }
                }
                int amax_token_29 = wide_token + 64 + 56;
                if (amax_token_29 < valid_tokens) {
                    {
                        float _fabs_244 = fabsf(_tmem_load_1[28]);
                        float _fabs_245 = fabsf(_tmem_load_1[30]);
                        float _max3_122;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_122) : "f"(tmax), "f"(_fabs_244), "f"(_fabs_245));
                        tmax = _max3_122;
                        float _fabs_246 = fabsf(_tmem_load_3[28]);
                        float _fabs_247 = fabsf(_tmem_load_3[30]);
                        float _max3_123;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_123) : "f"(tmax), "f"(_fabs_246), "f"(_fabs_247));
                        tmax = _max3_123;
                    }
                }
                int amax_token_30 = wide_token + 64 + 56 + 1;
                if (amax_token_30 < valid_tokens) {
                    {
                        float _fabs_252 = fabsf(_tmem_load_1[29]);
                        float _fabs_253 = fabsf(_tmem_load_1[31]);
                        float _max3_126;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_126) : "f"(tmax), "f"(_fabs_252), "f"(_fabs_253));
                        tmax = _max3_126;
                        float _fabs_254 = fabsf(_tmem_load_3[29]);
                        float _fabs_255 = fabsf(_tmem_load_3[31]);
                        float _max3_127;
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                        #endif
                        asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_127) : "f"(tmax), "f"(_fabs_254), "f"(_fabs_255));
                        tmax = _max3_127;
                    }
                }
                float _warp_redux_f32_0;
                asm volatile("redux.sync.max.NaN.f32 %0, %1, 0xffffffff;" : "=f"(_warp_redux_f32_0) : "f"(tmax));
                tmax = _warp_redux_f32_0;
                if (lane == 0) {
                    amax_slots[epi_warp] = tmax;
                }
                asm volatile("cp.async.bulk.wait_group.read 0;");
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                float _fmax_0 = fmaxf(amax_slots[0], amax_slots[1]);
                float _fmax_1 = fmaxf(amax_slots[2], amax_slots[3]);
                float _fmax_2 = fmaxf(_fmax_0, _fmax_1);
                float tile_max = _fmax_2;
                float q8_inv = 0.0f;
                float q8_scale = 0.0f;
                if (tile_max > 0.0f) {
                    float _fdiv_rn_0 = __fdiv_rn(127.0f, tile_max);
                    q8_inv = _fdiv_rn_0;
                    float _fdiv_rn_1 = __fdiv_rn(tile_max, 127.0f);
                    q8_scale = _fdiv_rn_1;
                }
                unsigned int q8_word = 0;
                float q8_r0 = 0.0f;
                float q8_r1 = 0.0f;
                float q8_r2 = 0.0f;
                float q8_r3 = 0.0f;
                int q8_i0 = 0;
                int q8_i1 = 0;
                int q8_i2 = 0;
                int q8_i3 = 0;
                unsigned int q8_b01 = 0;
                unsigned int q8_b23 = 0;
                int q8_swz = 0;
                int token = wide_token;
                {
                    wide_values[0] = _tmem_load_0[0];
                    wide_values[1] = _tmem_load_0[2];
                    wide_values[2] = _tmem_load_2[0];
                    wide_values[3] = _tmem_load_2[2];
                }
                {
                    float _fma_0 = __fmaf_rn(wide_values[0], q8_inv, 8388736.0f);
                    q8_r0 = _fma_0;
                    float _fma_1 = __fmaf_rn(wide_values[1], q8_inv, 8388736.0f);
                    q8_r1 = _fma_1;
                    float _fma_2 = __fmaf_rn(wide_values[2], q8_inv, 8388736.0f);
                    q8_r2 = _fma_2;
                    float _fma_3 = __fmaf_rn(wide_values[3], q8_inv, 8388736.0f);
                    q8_r3 = _fma_3;
                    uint32_t _prmt_b32_0;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_0) : "r"(__as_u32(q8_r0)), "r"(__as_u32(q8_r1)));
                    q8_b01 = _prmt_b32_0;
                    uint32_t _prmt_b32_1;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_1) : "r"(__as_u32(q8_r2)), "r"(__as_u32(q8_r3)));
                    q8_b23 = _prmt_b32_1;
                    uint32_t _prmt_b32_2;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_b32_2) : "r"(q8_b01), "r"(q8_b23));
                    q8_word = _prmt_b32_2;
                }
                q8_swz = wide_feature ^ token % 8 * 16;
                epi_staging_u32[(token * 128 + q8_swz) / 4] = q8_word;
                int token_31 = wide_token + 1;
                {
                    wide_values[0] = _tmem_load_0[1];
                    wide_values[1] = _tmem_load_0[3];
                    wide_values[2] = _tmem_load_2[1];
                    wide_values[3] = _tmem_load_2[3];
                }
                {
                    float _fma_4 = __fmaf_rn(wide_values[0], q8_inv, 8388736.0f);
                    q8_r0 = _fma_4;
                    float _fma_5 = __fmaf_rn(wide_values[1], q8_inv, 8388736.0f);
                    q8_r1 = _fma_5;
                    float _fma_6 = __fmaf_rn(wide_values[2], q8_inv, 8388736.0f);
                    q8_r2 = _fma_6;
                    float _fma_7 = __fmaf_rn(wide_values[3], q8_inv, 8388736.0f);
                    q8_r3 = _fma_7;
                    uint32_t _prmt_b32_6;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_6) : "r"(__as_u32(q8_r0)), "r"(__as_u32(q8_r1)));
                    q8_b01 = _prmt_b32_6;
                    uint32_t _prmt_b32_7;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_7) : "r"(__as_u32(q8_r2)), "r"(__as_u32(q8_r3)));
                    q8_b23 = _prmt_b32_7;
                    uint32_t _prmt_b32_8;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_b32_8) : "r"(q8_b01), "r"(q8_b23));
                    q8_word = _prmt_b32_8;
                }
                q8_swz = wide_feature ^ token_31 % 8 * 16;
                epi_staging_u32[(token_31 * 128 + q8_swz) / 4] = q8_word;
                int token_32 = wide_token + 8;
                {
                    wide_values[0] = _tmem_load_0[4];
                    wide_values[1] = _tmem_load_0[6];
                    wide_values[2] = _tmem_load_2[4];
                    wide_values[3] = _tmem_load_2[6];
                }
                {
                    float _fma_8 = __fmaf_rn(wide_values[0], q8_inv, 8388736.0f);
                    q8_r0 = _fma_8;
                    float _fma_9 = __fmaf_rn(wide_values[1], q8_inv, 8388736.0f);
                    q8_r1 = _fma_9;
                    float _fma_10 = __fmaf_rn(wide_values[2], q8_inv, 8388736.0f);
                    q8_r2 = _fma_10;
                    float _fma_11 = __fmaf_rn(wide_values[3], q8_inv, 8388736.0f);
                    q8_r3 = _fma_11;
                    uint32_t _prmt_b32_12;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_12) : "r"(__as_u32(q8_r0)), "r"(__as_u32(q8_r1)));
                    q8_b01 = _prmt_b32_12;
                    uint32_t _prmt_b32_13;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_13) : "r"(__as_u32(q8_r2)), "r"(__as_u32(q8_r3)));
                    q8_b23 = _prmt_b32_13;
                    uint32_t _prmt_b32_14;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_b32_14) : "r"(q8_b01), "r"(q8_b23));
                    q8_word = _prmt_b32_14;
                }
                q8_swz = wide_feature ^ token_32 % 8 * 16;
                epi_staging_u32[(token_32 * 128 + q8_swz) / 4] = q8_word;
                int token_33 = wide_token + 8 + 1;
                {
                    wide_values[0] = _tmem_load_0[5];
                    wide_values[1] = _tmem_load_0[7];
                    wide_values[2] = _tmem_load_2[5];
                    wide_values[3] = _tmem_load_2[7];
                }
                {
                    float _fma_12 = __fmaf_rn(wide_values[0], q8_inv, 8388736.0f);
                    q8_r0 = _fma_12;
                    float _fma_13 = __fmaf_rn(wide_values[1], q8_inv, 8388736.0f);
                    q8_r1 = _fma_13;
                    float _fma_14 = __fmaf_rn(wide_values[2], q8_inv, 8388736.0f);
                    q8_r2 = _fma_14;
                    float _fma_15 = __fmaf_rn(wide_values[3], q8_inv, 8388736.0f);
                    q8_r3 = _fma_15;
                    uint32_t _prmt_b32_18;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_18) : "r"(__as_u32(q8_r0)), "r"(__as_u32(q8_r1)));
                    q8_b01 = _prmt_b32_18;
                    uint32_t _prmt_b32_19;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_19) : "r"(__as_u32(q8_r2)), "r"(__as_u32(q8_r3)));
                    q8_b23 = _prmt_b32_19;
                    uint32_t _prmt_b32_20;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_b32_20) : "r"(q8_b01), "r"(q8_b23));
                    q8_word = _prmt_b32_20;
                }
                q8_swz = wide_feature ^ token_33 % 8 * 16;
                epi_staging_u32[(token_33 * 128 + q8_swz) / 4] = q8_word;
                int token_34 = wide_token + 16;
                {
                    wide_values[0] = _tmem_load_0[8];
                    wide_values[1] = _tmem_load_0[10];
                    wide_values[2] = _tmem_load_2[8];
                    wide_values[3] = _tmem_load_2[10];
                }
                {
                    float _fma_16 = __fmaf_rn(wide_values[0], q8_inv, 8388736.0f);
                    q8_r0 = _fma_16;
                    float _fma_17 = __fmaf_rn(wide_values[1], q8_inv, 8388736.0f);
                    q8_r1 = _fma_17;
                    float _fma_18 = __fmaf_rn(wide_values[2], q8_inv, 8388736.0f);
                    q8_r2 = _fma_18;
                    float _fma_19 = __fmaf_rn(wide_values[3], q8_inv, 8388736.0f);
                    q8_r3 = _fma_19;
                    uint32_t _prmt_b32_24;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_24) : "r"(__as_u32(q8_r0)), "r"(__as_u32(q8_r1)));
                    q8_b01 = _prmt_b32_24;
                    uint32_t _prmt_b32_25;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_25) : "r"(__as_u32(q8_r2)), "r"(__as_u32(q8_r3)));
                    q8_b23 = _prmt_b32_25;
                    uint32_t _prmt_b32_26;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_b32_26) : "r"(q8_b01), "r"(q8_b23));
                    q8_word = _prmt_b32_26;
                }
                q8_swz = wide_feature ^ token_34 % 8 * 16;
                epi_staging_u32[(token_34 * 128 + q8_swz) / 4] = q8_word;
                int token_35 = wide_token + 16 + 1;
                {
                    wide_values[0] = _tmem_load_0[9];
                    wide_values[1] = _tmem_load_0[11];
                    wide_values[2] = _tmem_load_2[9];
                    wide_values[3] = _tmem_load_2[11];
                }
                {
                    float _fma_20 = __fmaf_rn(wide_values[0], q8_inv, 8388736.0f);
                    q8_r0 = _fma_20;
                    float _fma_21 = __fmaf_rn(wide_values[1], q8_inv, 8388736.0f);
                    q8_r1 = _fma_21;
                    float _fma_22 = __fmaf_rn(wide_values[2], q8_inv, 8388736.0f);
                    q8_r2 = _fma_22;
                    float _fma_23 = __fmaf_rn(wide_values[3], q8_inv, 8388736.0f);
                    q8_r3 = _fma_23;
                    uint32_t _prmt_b32_30;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_30) : "r"(__as_u32(q8_r0)), "r"(__as_u32(q8_r1)));
                    q8_b01 = _prmt_b32_30;
                    uint32_t _prmt_b32_31;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_31) : "r"(__as_u32(q8_r2)), "r"(__as_u32(q8_r3)));
                    q8_b23 = _prmt_b32_31;
                    uint32_t _prmt_b32_32;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_b32_32) : "r"(q8_b01), "r"(q8_b23));
                    q8_word = _prmt_b32_32;
                }
                q8_swz = wide_feature ^ token_35 % 8 * 16;
                epi_staging_u32[(token_35 * 128 + q8_swz) / 4] = q8_word;
                int token_36 = wide_token + 24;
                {
                    wide_values[0] = _tmem_load_0[12];
                    wide_values[1] = _tmem_load_0[14];
                    wide_values[2] = _tmem_load_2[12];
                    wide_values[3] = _tmem_load_2[14];
                }
                {
                    float _fma_24 = __fmaf_rn(wide_values[0], q8_inv, 8388736.0f);
                    q8_r0 = _fma_24;
                    float _fma_25 = __fmaf_rn(wide_values[1], q8_inv, 8388736.0f);
                    q8_r1 = _fma_25;
                    float _fma_26 = __fmaf_rn(wide_values[2], q8_inv, 8388736.0f);
                    q8_r2 = _fma_26;
                    float _fma_27 = __fmaf_rn(wide_values[3], q8_inv, 8388736.0f);
                    q8_r3 = _fma_27;
                    uint32_t _prmt_b32_36;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_36) : "r"(__as_u32(q8_r0)), "r"(__as_u32(q8_r1)));
                    q8_b01 = _prmt_b32_36;
                    uint32_t _prmt_b32_37;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_37) : "r"(__as_u32(q8_r2)), "r"(__as_u32(q8_r3)));
                    q8_b23 = _prmt_b32_37;
                    uint32_t _prmt_b32_38;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_b32_38) : "r"(q8_b01), "r"(q8_b23));
                    q8_word = _prmt_b32_38;
                }
                q8_swz = wide_feature ^ token_36 % 8 * 16;
                epi_staging_u32[(token_36 * 128 + q8_swz) / 4] = q8_word;
                int token_37 = wide_token + 24 + 1;
                {
                    wide_values[0] = _tmem_load_0[13];
                    wide_values[1] = _tmem_load_0[15];
                    wide_values[2] = _tmem_load_2[13];
                    wide_values[3] = _tmem_load_2[15];
                }
                {
                    float _fma_28 = __fmaf_rn(wide_values[0], q8_inv, 8388736.0f);
                    q8_r0 = _fma_28;
                    float _fma_29 = __fmaf_rn(wide_values[1], q8_inv, 8388736.0f);
                    q8_r1 = _fma_29;
                    float _fma_30 = __fmaf_rn(wide_values[2], q8_inv, 8388736.0f);
                    q8_r2 = _fma_30;
                    float _fma_31 = __fmaf_rn(wide_values[3], q8_inv, 8388736.0f);
                    q8_r3 = _fma_31;
                    uint32_t _prmt_b32_42;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_42) : "r"(__as_u32(q8_r0)), "r"(__as_u32(q8_r1)));
                    q8_b01 = _prmt_b32_42;
                    uint32_t _prmt_b32_43;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_43) : "r"(__as_u32(q8_r2)), "r"(__as_u32(q8_r3)));
                    q8_b23 = _prmt_b32_43;
                    uint32_t _prmt_b32_44;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_b32_44) : "r"(q8_b01), "r"(q8_b23));
                    q8_word = _prmt_b32_44;
                }
                q8_swz = wide_feature ^ token_37 % 8 * 16;
                epi_staging_u32[(token_37 * 128 + q8_swz) / 4] = q8_word;
                int token_38 = wide_token + 32;
                {
                    wide_values[0] = _tmem_load_0[16];
                    wide_values[1] = _tmem_load_0[18];
                    wide_values[2] = _tmem_load_2[16];
                    wide_values[3] = _tmem_load_2[18];
                }
                {
                    float _fma_32 = __fmaf_rn(wide_values[0], q8_inv, 8388736.0f);
                    q8_r0 = _fma_32;
                    float _fma_33 = __fmaf_rn(wide_values[1], q8_inv, 8388736.0f);
                    q8_r1 = _fma_33;
                    float _fma_34 = __fmaf_rn(wide_values[2], q8_inv, 8388736.0f);
                    q8_r2 = _fma_34;
                    float _fma_35 = __fmaf_rn(wide_values[3], q8_inv, 8388736.0f);
                    q8_r3 = _fma_35;
                    uint32_t _prmt_b32_48;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_48) : "r"(__as_u32(q8_r0)), "r"(__as_u32(q8_r1)));
                    q8_b01 = _prmt_b32_48;
                    uint32_t _prmt_b32_49;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_49) : "r"(__as_u32(q8_r2)), "r"(__as_u32(q8_r3)));
                    q8_b23 = _prmt_b32_49;
                    uint32_t _prmt_b32_50;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_b32_50) : "r"(q8_b01), "r"(q8_b23));
                    q8_word = _prmt_b32_50;
                }
                q8_swz = wide_feature ^ token_38 % 8 * 16;
                epi_staging_u32[(token_38 * 128 + q8_swz) / 4] = q8_word;
                int token_39 = wide_token + 32 + 1;
                {
                    wide_values[0] = _tmem_load_0[17];
                    wide_values[1] = _tmem_load_0[19];
                    wide_values[2] = _tmem_load_2[17];
                    wide_values[3] = _tmem_load_2[19];
                }
                {
                    float _fma_36 = __fmaf_rn(wide_values[0], q8_inv, 8388736.0f);
                    q8_r0 = _fma_36;
                    float _fma_37 = __fmaf_rn(wide_values[1], q8_inv, 8388736.0f);
                    q8_r1 = _fma_37;
                    float _fma_38 = __fmaf_rn(wide_values[2], q8_inv, 8388736.0f);
                    q8_r2 = _fma_38;
                    float _fma_39 = __fmaf_rn(wide_values[3], q8_inv, 8388736.0f);
                    q8_r3 = _fma_39;
                    uint32_t _prmt_b32_54;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_54) : "r"(__as_u32(q8_r0)), "r"(__as_u32(q8_r1)));
                    q8_b01 = _prmt_b32_54;
                    uint32_t _prmt_b32_55;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_55) : "r"(__as_u32(q8_r2)), "r"(__as_u32(q8_r3)));
                    q8_b23 = _prmt_b32_55;
                    uint32_t _prmt_b32_56;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_b32_56) : "r"(q8_b01), "r"(q8_b23));
                    q8_word = _prmt_b32_56;
                }
                q8_swz = wide_feature ^ token_39 % 8 * 16;
                epi_staging_u32[(token_39 * 128 + q8_swz) / 4] = q8_word;
                int token_40 = wide_token + 40;
                {
                    wide_values[0] = _tmem_load_0[20];
                    wide_values[1] = _tmem_load_0[22];
                    wide_values[2] = _tmem_load_2[20];
                    wide_values[3] = _tmem_load_2[22];
                }
                {
                    float _fma_40 = __fmaf_rn(wide_values[0], q8_inv, 8388736.0f);
                    q8_r0 = _fma_40;
                    float _fma_41 = __fmaf_rn(wide_values[1], q8_inv, 8388736.0f);
                    q8_r1 = _fma_41;
                    float _fma_42 = __fmaf_rn(wide_values[2], q8_inv, 8388736.0f);
                    q8_r2 = _fma_42;
                    float _fma_43 = __fmaf_rn(wide_values[3], q8_inv, 8388736.0f);
                    q8_r3 = _fma_43;
                    uint32_t _prmt_b32_60;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_60) : "r"(__as_u32(q8_r0)), "r"(__as_u32(q8_r1)));
                    q8_b01 = _prmt_b32_60;
                    uint32_t _prmt_b32_61;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_61) : "r"(__as_u32(q8_r2)), "r"(__as_u32(q8_r3)));
                    q8_b23 = _prmt_b32_61;
                    uint32_t _prmt_b32_62;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_b32_62) : "r"(q8_b01), "r"(q8_b23));
                    q8_word = _prmt_b32_62;
                }
                q8_swz = wide_feature ^ token_40 % 8 * 16;
                epi_staging_u32[(token_40 * 128 + q8_swz) / 4] = q8_word;
                int token_41 = wide_token + 40 + 1;
                {
                    wide_values[0] = _tmem_load_0[21];
                    wide_values[1] = _tmem_load_0[23];
                    wide_values[2] = _tmem_load_2[21];
                    wide_values[3] = _tmem_load_2[23];
                }
                {
                    float _fma_44 = __fmaf_rn(wide_values[0], q8_inv, 8388736.0f);
                    q8_r0 = _fma_44;
                    float _fma_45 = __fmaf_rn(wide_values[1], q8_inv, 8388736.0f);
                    q8_r1 = _fma_45;
                    float _fma_46 = __fmaf_rn(wide_values[2], q8_inv, 8388736.0f);
                    q8_r2 = _fma_46;
                    float _fma_47 = __fmaf_rn(wide_values[3], q8_inv, 8388736.0f);
                    q8_r3 = _fma_47;
                    uint32_t _prmt_b32_66;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_66) : "r"(__as_u32(q8_r0)), "r"(__as_u32(q8_r1)));
                    q8_b01 = _prmt_b32_66;
                    uint32_t _prmt_b32_67;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_67) : "r"(__as_u32(q8_r2)), "r"(__as_u32(q8_r3)));
                    q8_b23 = _prmt_b32_67;
                    uint32_t _prmt_b32_68;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_b32_68) : "r"(q8_b01), "r"(q8_b23));
                    q8_word = _prmt_b32_68;
                }
                q8_swz = wide_feature ^ token_41 % 8 * 16;
                epi_staging_u32[(token_41 * 128 + q8_swz) / 4] = q8_word;
                int token_42 = wide_token + 48;
                {
                    wide_values[0] = _tmem_load_0[24];
                    wide_values[1] = _tmem_load_0[26];
                    wide_values[2] = _tmem_load_2[24];
                    wide_values[3] = _tmem_load_2[26];
                }
                {
                    float _fma_48 = __fmaf_rn(wide_values[0], q8_inv, 8388736.0f);
                    q8_r0 = _fma_48;
                    float _fma_49 = __fmaf_rn(wide_values[1], q8_inv, 8388736.0f);
                    q8_r1 = _fma_49;
                    float _fma_50 = __fmaf_rn(wide_values[2], q8_inv, 8388736.0f);
                    q8_r2 = _fma_50;
                    float _fma_51 = __fmaf_rn(wide_values[3], q8_inv, 8388736.0f);
                    q8_r3 = _fma_51;
                    uint32_t _prmt_b32_72;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_72) : "r"(__as_u32(q8_r0)), "r"(__as_u32(q8_r1)));
                    q8_b01 = _prmt_b32_72;
                    uint32_t _prmt_b32_73;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_73) : "r"(__as_u32(q8_r2)), "r"(__as_u32(q8_r3)));
                    q8_b23 = _prmt_b32_73;
                    uint32_t _prmt_b32_74;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_b32_74) : "r"(q8_b01), "r"(q8_b23));
                    q8_word = _prmt_b32_74;
                }
                q8_swz = wide_feature ^ token_42 % 8 * 16;
                epi_staging_u32[(token_42 * 128 + q8_swz) / 4] = q8_word;
                int token_43 = wide_token + 48 + 1;
                {
                    wide_values[0] = _tmem_load_0[25];
                    wide_values[1] = _tmem_load_0[27];
                    wide_values[2] = _tmem_load_2[25];
                    wide_values[3] = _tmem_load_2[27];
                }
                {
                    float _fma_52 = __fmaf_rn(wide_values[0], q8_inv, 8388736.0f);
                    q8_r0 = _fma_52;
                    float _fma_53 = __fmaf_rn(wide_values[1], q8_inv, 8388736.0f);
                    q8_r1 = _fma_53;
                    float _fma_54 = __fmaf_rn(wide_values[2], q8_inv, 8388736.0f);
                    q8_r2 = _fma_54;
                    float _fma_55 = __fmaf_rn(wide_values[3], q8_inv, 8388736.0f);
                    q8_r3 = _fma_55;
                    uint32_t _prmt_b32_78;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_78) : "r"(__as_u32(q8_r0)), "r"(__as_u32(q8_r1)));
                    q8_b01 = _prmt_b32_78;
                    uint32_t _prmt_b32_79;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_79) : "r"(__as_u32(q8_r2)), "r"(__as_u32(q8_r3)));
                    q8_b23 = _prmt_b32_79;
                    uint32_t _prmt_b32_80;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_b32_80) : "r"(q8_b01), "r"(q8_b23));
                    q8_word = _prmt_b32_80;
                }
                q8_swz = wide_feature ^ token_43 % 8 * 16;
                epi_staging_u32[(token_43 * 128 + q8_swz) / 4] = q8_word;
                int token_44 = wide_token + 56;
                {
                    wide_values[0] = _tmem_load_0[28];
                    wide_values[1] = _tmem_load_0[30];
                    wide_values[2] = _tmem_load_2[28];
                    wide_values[3] = _tmem_load_2[30];
                }
                {
                    float _fma_56 = __fmaf_rn(wide_values[0], q8_inv, 8388736.0f);
                    q8_r0 = _fma_56;
                    float _fma_57 = __fmaf_rn(wide_values[1], q8_inv, 8388736.0f);
                    q8_r1 = _fma_57;
                    float _fma_58 = __fmaf_rn(wide_values[2], q8_inv, 8388736.0f);
                    q8_r2 = _fma_58;
                    float _fma_59 = __fmaf_rn(wide_values[3], q8_inv, 8388736.0f);
                    q8_r3 = _fma_59;
                    uint32_t _prmt_b32_84;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_84) : "r"(__as_u32(q8_r0)), "r"(__as_u32(q8_r1)));
                    q8_b01 = _prmt_b32_84;
                    uint32_t _prmt_b32_85;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_85) : "r"(__as_u32(q8_r2)), "r"(__as_u32(q8_r3)));
                    q8_b23 = _prmt_b32_85;
                    uint32_t _prmt_b32_86;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_b32_86) : "r"(q8_b01), "r"(q8_b23));
                    q8_word = _prmt_b32_86;
                }
                q8_swz = wide_feature ^ token_44 % 8 * 16;
                epi_staging_u32[(token_44 * 128 + q8_swz) / 4] = q8_word;
                int token_45 = wide_token + 56 + 1;
                {
                    wide_values[0] = _tmem_load_0[29];
                    wide_values[1] = _tmem_load_0[31];
                    wide_values[2] = _tmem_load_2[29];
                    wide_values[3] = _tmem_load_2[31];
                }
                {
                    float _fma_60 = __fmaf_rn(wide_values[0], q8_inv, 8388736.0f);
                    q8_r0 = _fma_60;
                    float _fma_61 = __fmaf_rn(wide_values[1], q8_inv, 8388736.0f);
                    q8_r1 = _fma_61;
                    float _fma_62 = __fmaf_rn(wide_values[2], q8_inv, 8388736.0f);
                    q8_r2 = _fma_62;
                    float _fma_63 = __fmaf_rn(wide_values[3], q8_inv, 8388736.0f);
                    q8_r3 = _fma_63;
                    uint32_t _prmt_b32_90;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_90) : "r"(__as_u32(q8_r0)), "r"(__as_u32(q8_r1)));
                    q8_b01 = _prmt_b32_90;
                    uint32_t _prmt_b32_91;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_91) : "r"(__as_u32(q8_r2)), "r"(__as_u32(q8_r3)));
                    q8_b23 = _prmt_b32_91;
                    uint32_t _prmt_b32_92;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_b32_92) : "r"(q8_b01), "r"(q8_b23));
                    q8_word = _prmt_b32_92;
                }
                q8_swz = wide_feature ^ token_45 % 8 * 16;
                epi_staging_u32[(token_45 * 128 + q8_swz) / 4] = q8_word;
                int token_46 = wide_token + 64;
                {
                    wide_values[0] = _tmem_load_1[0];
                    wide_values[1] = _tmem_load_1[2];
                    wide_values[2] = _tmem_load_3[0];
                    wide_values[3] = _tmem_load_3[2];
                }
                {
                    float _fma_64 = __fmaf_rn(wide_values[0], q8_inv, 8388736.0f);
                    q8_r0 = _fma_64;
                    float _fma_65 = __fmaf_rn(wide_values[1], q8_inv, 8388736.0f);
                    q8_r1 = _fma_65;
                    float _fma_66 = __fmaf_rn(wide_values[2], q8_inv, 8388736.0f);
                    q8_r2 = _fma_66;
                    float _fma_67 = __fmaf_rn(wide_values[3], q8_inv, 8388736.0f);
                    q8_r3 = _fma_67;
                    uint32_t _prmt_b32_96;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_96) : "r"(__as_u32(q8_r0)), "r"(__as_u32(q8_r1)));
                    q8_b01 = _prmt_b32_96;
                    uint32_t _prmt_b32_97;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_97) : "r"(__as_u32(q8_r2)), "r"(__as_u32(q8_r3)));
                    q8_b23 = _prmt_b32_97;
                    uint32_t _prmt_b32_98;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_b32_98) : "r"(q8_b01), "r"(q8_b23));
                    q8_word = _prmt_b32_98;
                }
                q8_swz = wide_feature ^ token_46 % 8 * 16;
                epi_staging_u32[(token_46 * 128 + q8_swz) / 4] = q8_word;
                int token_47 = wide_token + 64 + 1;
                {
                    wide_values[0] = _tmem_load_1[1];
                    wide_values[1] = _tmem_load_1[3];
                    wide_values[2] = _tmem_load_3[1];
                    wide_values[3] = _tmem_load_3[3];
                }
                {
                    float _fma_68 = __fmaf_rn(wide_values[0], q8_inv, 8388736.0f);
                    q8_r0 = _fma_68;
                    float _fma_69 = __fmaf_rn(wide_values[1], q8_inv, 8388736.0f);
                    q8_r1 = _fma_69;
                    float _fma_70 = __fmaf_rn(wide_values[2], q8_inv, 8388736.0f);
                    q8_r2 = _fma_70;
                    float _fma_71 = __fmaf_rn(wide_values[3], q8_inv, 8388736.0f);
                    q8_r3 = _fma_71;
                    uint32_t _prmt_b32_102;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_102) : "r"(__as_u32(q8_r0)), "r"(__as_u32(q8_r1)));
                    q8_b01 = _prmt_b32_102;
                    uint32_t _prmt_b32_103;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_103) : "r"(__as_u32(q8_r2)), "r"(__as_u32(q8_r3)));
                    q8_b23 = _prmt_b32_103;
                    uint32_t _prmt_b32_104;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_b32_104) : "r"(q8_b01), "r"(q8_b23));
                    q8_word = _prmt_b32_104;
                }
                q8_swz = wide_feature ^ token_47 % 8 * 16;
                epi_staging_u32[(token_47 * 128 + q8_swz) / 4] = q8_word;
                int token_48 = wide_token + 64 + 8;
                {
                    wide_values[0] = _tmem_load_1[4];
                    wide_values[1] = _tmem_load_1[6];
                    wide_values[2] = _tmem_load_3[4];
                    wide_values[3] = _tmem_load_3[6];
                }
                {
                    float _fma_72 = __fmaf_rn(wide_values[0], q8_inv, 8388736.0f);
                    q8_r0 = _fma_72;
                    float _fma_73 = __fmaf_rn(wide_values[1], q8_inv, 8388736.0f);
                    q8_r1 = _fma_73;
                    float _fma_74 = __fmaf_rn(wide_values[2], q8_inv, 8388736.0f);
                    q8_r2 = _fma_74;
                    float _fma_75 = __fmaf_rn(wide_values[3], q8_inv, 8388736.0f);
                    q8_r3 = _fma_75;
                    uint32_t _prmt_b32_108;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_108) : "r"(__as_u32(q8_r0)), "r"(__as_u32(q8_r1)));
                    q8_b01 = _prmt_b32_108;
                    uint32_t _prmt_b32_109;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_109) : "r"(__as_u32(q8_r2)), "r"(__as_u32(q8_r3)));
                    q8_b23 = _prmt_b32_109;
                    uint32_t _prmt_b32_110;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_b32_110) : "r"(q8_b01), "r"(q8_b23));
                    q8_word = _prmt_b32_110;
                }
                q8_swz = wide_feature ^ token_48 % 8 * 16;
                epi_staging_u32[(token_48 * 128 + q8_swz) / 4] = q8_word;
                int token_49 = wide_token + 64 + 8 + 1;
                {
                    wide_values[0] = _tmem_load_1[5];
                    wide_values[1] = _tmem_load_1[7];
                    wide_values[2] = _tmem_load_3[5];
                    wide_values[3] = _tmem_load_3[7];
                }
                {
                    float _fma_76 = __fmaf_rn(wide_values[0], q8_inv, 8388736.0f);
                    q8_r0 = _fma_76;
                    float _fma_77 = __fmaf_rn(wide_values[1], q8_inv, 8388736.0f);
                    q8_r1 = _fma_77;
                    float _fma_78 = __fmaf_rn(wide_values[2], q8_inv, 8388736.0f);
                    q8_r2 = _fma_78;
                    float _fma_79 = __fmaf_rn(wide_values[3], q8_inv, 8388736.0f);
                    q8_r3 = _fma_79;
                    uint32_t _prmt_b32_114;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_114) : "r"(__as_u32(q8_r0)), "r"(__as_u32(q8_r1)));
                    q8_b01 = _prmt_b32_114;
                    uint32_t _prmt_b32_115;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_115) : "r"(__as_u32(q8_r2)), "r"(__as_u32(q8_r3)));
                    q8_b23 = _prmt_b32_115;
                    uint32_t _prmt_b32_116;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_b32_116) : "r"(q8_b01), "r"(q8_b23));
                    q8_word = _prmt_b32_116;
                }
                q8_swz = wide_feature ^ token_49 % 8 * 16;
                epi_staging_u32[(token_49 * 128 + q8_swz) / 4] = q8_word;
                int token_50 = wide_token + 64 + 16;
                {
                    wide_values[0] = _tmem_load_1[8];
                    wide_values[1] = _tmem_load_1[10];
                    wide_values[2] = _tmem_load_3[8];
                    wide_values[3] = _tmem_load_3[10];
                }
                {
                    float _fma_80 = __fmaf_rn(wide_values[0], q8_inv, 8388736.0f);
                    q8_r0 = _fma_80;
                    float _fma_81 = __fmaf_rn(wide_values[1], q8_inv, 8388736.0f);
                    q8_r1 = _fma_81;
                    float _fma_82 = __fmaf_rn(wide_values[2], q8_inv, 8388736.0f);
                    q8_r2 = _fma_82;
                    float _fma_83 = __fmaf_rn(wide_values[3], q8_inv, 8388736.0f);
                    q8_r3 = _fma_83;
                    uint32_t _prmt_b32_120;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_120) : "r"(__as_u32(q8_r0)), "r"(__as_u32(q8_r1)));
                    q8_b01 = _prmt_b32_120;
                    uint32_t _prmt_b32_121;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_121) : "r"(__as_u32(q8_r2)), "r"(__as_u32(q8_r3)));
                    q8_b23 = _prmt_b32_121;
                    uint32_t _prmt_b32_122;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_b32_122) : "r"(q8_b01), "r"(q8_b23));
                    q8_word = _prmt_b32_122;
                }
                q8_swz = wide_feature ^ token_50 % 8 * 16;
                epi_staging_u32[(token_50 * 128 + q8_swz) / 4] = q8_word;
                int token_51 = wide_token + 64 + 16 + 1;
                {
                    wide_values[0] = _tmem_load_1[9];
                    wide_values[1] = _tmem_load_1[11];
                    wide_values[2] = _tmem_load_3[9];
                    wide_values[3] = _tmem_load_3[11];
                }
                {
                    float _fma_84 = __fmaf_rn(wide_values[0], q8_inv, 8388736.0f);
                    q8_r0 = _fma_84;
                    float _fma_85 = __fmaf_rn(wide_values[1], q8_inv, 8388736.0f);
                    q8_r1 = _fma_85;
                    float _fma_86 = __fmaf_rn(wide_values[2], q8_inv, 8388736.0f);
                    q8_r2 = _fma_86;
                    float _fma_87 = __fmaf_rn(wide_values[3], q8_inv, 8388736.0f);
                    q8_r3 = _fma_87;
                    uint32_t _prmt_b32_126;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_126) : "r"(__as_u32(q8_r0)), "r"(__as_u32(q8_r1)));
                    q8_b01 = _prmt_b32_126;
                    uint32_t _prmt_b32_127;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_127) : "r"(__as_u32(q8_r2)), "r"(__as_u32(q8_r3)));
                    q8_b23 = _prmt_b32_127;
                    uint32_t _prmt_b32_128;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_b32_128) : "r"(q8_b01), "r"(q8_b23));
                    q8_word = _prmt_b32_128;
                }
                q8_swz = wide_feature ^ token_51 % 8 * 16;
                epi_staging_u32[(token_51 * 128 + q8_swz) / 4] = q8_word;
                int token_52 = wide_token + 64 + 24;
                {
                    wide_values[0] = _tmem_load_1[12];
                    wide_values[1] = _tmem_load_1[14];
                    wide_values[2] = _tmem_load_3[12];
                    wide_values[3] = _tmem_load_3[14];
                }
                {
                    float _fma_88 = __fmaf_rn(wide_values[0], q8_inv, 8388736.0f);
                    q8_r0 = _fma_88;
                    float _fma_89 = __fmaf_rn(wide_values[1], q8_inv, 8388736.0f);
                    q8_r1 = _fma_89;
                    float _fma_90 = __fmaf_rn(wide_values[2], q8_inv, 8388736.0f);
                    q8_r2 = _fma_90;
                    float _fma_91 = __fmaf_rn(wide_values[3], q8_inv, 8388736.0f);
                    q8_r3 = _fma_91;
                    uint32_t _prmt_b32_132;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_132) : "r"(__as_u32(q8_r0)), "r"(__as_u32(q8_r1)));
                    q8_b01 = _prmt_b32_132;
                    uint32_t _prmt_b32_133;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_133) : "r"(__as_u32(q8_r2)), "r"(__as_u32(q8_r3)));
                    q8_b23 = _prmt_b32_133;
                    uint32_t _prmt_b32_134;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_b32_134) : "r"(q8_b01), "r"(q8_b23));
                    q8_word = _prmt_b32_134;
                }
                q8_swz = wide_feature ^ token_52 % 8 * 16;
                epi_staging_u32[(token_52 * 128 + q8_swz) / 4] = q8_word;
                int token_53 = wide_token + 64 + 24 + 1;
                {
                    wide_values[0] = _tmem_load_1[13];
                    wide_values[1] = _tmem_load_1[15];
                    wide_values[2] = _tmem_load_3[13];
                    wide_values[3] = _tmem_load_3[15];
                }
                {
                    float _fma_92 = __fmaf_rn(wide_values[0], q8_inv, 8388736.0f);
                    q8_r0 = _fma_92;
                    float _fma_93 = __fmaf_rn(wide_values[1], q8_inv, 8388736.0f);
                    q8_r1 = _fma_93;
                    float _fma_94 = __fmaf_rn(wide_values[2], q8_inv, 8388736.0f);
                    q8_r2 = _fma_94;
                    float _fma_95 = __fmaf_rn(wide_values[3], q8_inv, 8388736.0f);
                    q8_r3 = _fma_95;
                    uint32_t _prmt_b32_138;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_138) : "r"(__as_u32(q8_r0)), "r"(__as_u32(q8_r1)));
                    q8_b01 = _prmt_b32_138;
                    uint32_t _prmt_b32_139;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_139) : "r"(__as_u32(q8_r2)), "r"(__as_u32(q8_r3)));
                    q8_b23 = _prmt_b32_139;
                    uint32_t _prmt_b32_140;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_b32_140) : "r"(q8_b01), "r"(q8_b23));
                    q8_word = _prmt_b32_140;
                }
                q8_swz = wide_feature ^ token_53 % 8 * 16;
                epi_staging_u32[(token_53 * 128 + q8_swz) / 4] = q8_word;
                int token_54 = wide_token + 64 + 32;
                {
                    wide_values[0] = _tmem_load_1[16];
                    wide_values[1] = _tmem_load_1[18];
                    wide_values[2] = _tmem_load_3[16];
                    wide_values[3] = _tmem_load_3[18];
                }
                {
                    float _fma_96 = __fmaf_rn(wide_values[0], q8_inv, 8388736.0f);
                    q8_r0 = _fma_96;
                    float _fma_97 = __fmaf_rn(wide_values[1], q8_inv, 8388736.0f);
                    q8_r1 = _fma_97;
                    float _fma_98 = __fmaf_rn(wide_values[2], q8_inv, 8388736.0f);
                    q8_r2 = _fma_98;
                    float _fma_99 = __fmaf_rn(wide_values[3], q8_inv, 8388736.0f);
                    q8_r3 = _fma_99;
                    uint32_t _prmt_b32_144;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_144) : "r"(__as_u32(q8_r0)), "r"(__as_u32(q8_r1)));
                    q8_b01 = _prmt_b32_144;
                    uint32_t _prmt_b32_145;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_145) : "r"(__as_u32(q8_r2)), "r"(__as_u32(q8_r3)));
                    q8_b23 = _prmt_b32_145;
                    uint32_t _prmt_b32_146;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_b32_146) : "r"(q8_b01), "r"(q8_b23));
                    q8_word = _prmt_b32_146;
                }
                q8_swz = wide_feature ^ token_54 % 8 * 16;
                epi_staging_u32[(token_54 * 128 + q8_swz) / 4] = q8_word;
                int token_55 = wide_token + 64 + 32 + 1;
                {
                    wide_values[0] = _tmem_load_1[17];
                    wide_values[1] = _tmem_load_1[19];
                    wide_values[2] = _tmem_load_3[17];
                    wide_values[3] = _tmem_load_3[19];
                }
                {
                    float _fma_100 = __fmaf_rn(wide_values[0], q8_inv, 8388736.0f);
                    q8_r0 = _fma_100;
                    float _fma_101 = __fmaf_rn(wide_values[1], q8_inv, 8388736.0f);
                    q8_r1 = _fma_101;
                    float _fma_102 = __fmaf_rn(wide_values[2], q8_inv, 8388736.0f);
                    q8_r2 = _fma_102;
                    float _fma_103 = __fmaf_rn(wide_values[3], q8_inv, 8388736.0f);
                    q8_r3 = _fma_103;
                    uint32_t _prmt_b32_150;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_150) : "r"(__as_u32(q8_r0)), "r"(__as_u32(q8_r1)));
                    q8_b01 = _prmt_b32_150;
                    uint32_t _prmt_b32_151;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_151) : "r"(__as_u32(q8_r2)), "r"(__as_u32(q8_r3)));
                    q8_b23 = _prmt_b32_151;
                    uint32_t _prmt_b32_152;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_b32_152) : "r"(q8_b01), "r"(q8_b23));
                    q8_word = _prmt_b32_152;
                }
                q8_swz = wide_feature ^ token_55 % 8 * 16;
                epi_staging_u32[(token_55 * 128 + q8_swz) / 4] = q8_word;
                int token_56 = wide_token + 64 + 40;
                {
                    wide_values[0] = _tmem_load_1[20];
                    wide_values[1] = _tmem_load_1[22];
                    wide_values[2] = _tmem_load_3[20];
                    wide_values[3] = _tmem_load_3[22];
                }
                {
                    float _fma_104 = __fmaf_rn(wide_values[0], q8_inv, 8388736.0f);
                    q8_r0 = _fma_104;
                    float _fma_105 = __fmaf_rn(wide_values[1], q8_inv, 8388736.0f);
                    q8_r1 = _fma_105;
                    float _fma_106 = __fmaf_rn(wide_values[2], q8_inv, 8388736.0f);
                    q8_r2 = _fma_106;
                    float _fma_107 = __fmaf_rn(wide_values[3], q8_inv, 8388736.0f);
                    q8_r3 = _fma_107;
                    uint32_t _prmt_b32_156;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_156) : "r"(__as_u32(q8_r0)), "r"(__as_u32(q8_r1)));
                    q8_b01 = _prmt_b32_156;
                    uint32_t _prmt_b32_157;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_157) : "r"(__as_u32(q8_r2)), "r"(__as_u32(q8_r3)));
                    q8_b23 = _prmt_b32_157;
                    uint32_t _prmt_b32_158;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_b32_158) : "r"(q8_b01), "r"(q8_b23));
                    q8_word = _prmt_b32_158;
                }
                q8_swz = wide_feature ^ token_56 % 8 * 16;
                epi_staging_u32[(token_56 * 128 + q8_swz) / 4] = q8_word;
                int token_57 = wide_token + 64 + 40 + 1;
                {
                    wide_values[0] = _tmem_load_1[21];
                    wide_values[1] = _tmem_load_1[23];
                    wide_values[2] = _tmem_load_3[21];
                    wide_values[3] = _tmem_load_3[23];
                }
                {
                    float _fma_108 = __fmaf_rn(wide_values[0], q8_inv, 8388736.0f);
                    q8_r0 = _fma_108;
                    float _fma_109 = __fmaf_rn(wide_values[1], q8_inv, 8388736.0f);
                    q8_r1 = _fma_109;
                    float _fma_110 = __fmaf_rn(wide_values[2], q8_inv, 8388736.0f);
                    q8_r2 = _fma_110;
                    float _fma_111 = __fmaf_rn(wide_values[3], q8_inv, 8388736.0f);
                    q8_r3 = _fma_111;
                    uint32_t _prmt_b32_162;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_162) : "r"(__as_u32(q8_r0)), "r"(__as_u32(q8_r1)));
                    q8_b01 = _prmt_b32_162;
                    uint32_t _prmt_b32_163;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_163) : "r"(__as_u32(q8_r2)), "r"(__as_u32(q8_r3)));
                    q8_b23 = _prmt_b32_163;
                    uint32_t _prmt_b32_164;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_b32_164) : "r"(q8_b01), "r"(q8_b23));
                    q8_word = _prmt_b32_164;
                }
                q8_swz = wide_feature ^ token_57 % 8 * 16;
                epi_staging_u32[(token_57 * 128 + q8_swz) / 4] = q8_word;
                int token_58 = wide_token + 64 + 48;
                {
                    wide_values[0] = _tmem_load_1[24];
                    wide_values[1] = _tmem_load_1[26];
                    wide_values[2] = _tmem_load_3[24];
                    wide_values[3] = _tmem_load_3[26];
                }
                {
                    float _fma_112 = __fmaf_rn(wide_values[0], q8_inv, 8388736.0f);
                    q8_r0 = _fma_112;
                    float _fma_113 = __fmaf_rn(wide_values[1], q8_inv, 8388736.0f);
                    q8_r1 = _fma_113;
                    float _fma_114 = __fmaf_rn(wide_values[2], q8_inv, 8388736.0f);
                    q8_r2 = _fma_114;
                    float _fma_115 = __fmaf_rn(wide_values[3], q8_inv, 8388736.0f);
                    q8_r3 = _fma_115;
                    uint32_t _prmt_b32_168;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_168) : "r"(__as_u32(q8_r0)), "r"(__as_u32(q8_r1)));
                    q8_b01 = _prmt_b32_168;
                    uint32_t _prmt_b32_169;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_169) : "r"(__as_u32(q8_r2)), "r"(__as_u32(q8_r3)));
                    q8_b23 = _prmt_b32_169;
                    uint32_t _prmt_b32_170;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_b32_170) : "r"(q8_b01), "r"(q8_b23));
                    q8_word = _prmt_b32_170;
                }
                q8_swz = wide_feature ^ token_58 % 8 * 16;
                epi_staging_u32[(token_58 * 128 + q8_swz) / 4] = q8_word;
                int token_59 = wide_token + 64 + 48 + 1;
                {
                    wide_values[0] = _tmem_load_1[25];
                    wide_values[1] = _tmem_load_1[27];
                    wide_values[2] = _tmem_load_3[25];
                    wide_values[3] = _tmem_load_3[27];
                }
                {
                    float _fma_116 = __fmaf_rn(wide_values[0], q8_inv, 8388736.0f);
                    q8_r0 = _fma_116;
                    float _fma_117 = __fmaf_rn(wide_values[1], q8_inv, 8388736.0f);
                    q8_r1 = _fma_117;
                    float _fma_118 = __fmaf_rn(wide_values[2], q8_inv, 8388736.0f);
                    q8_r2 = _fma_118;
                    float _fma_119 = __fmaf_rn(wide_values[3], q8_inv, 8388736.0f);
                    q8_r3 = _fma_119;
                    uint32_t _prmt_b32_174;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_174) : "r"(__as_u32(q8_r0)), "r"(__as_u32(q8_r1)));
                    q8_b01 = _prmt_b32_174;
                    uint32_t _prmt_b32_175;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_175) : "r"(__as_u32(q8_r2)), "r"(__as_u32(q8_r3)));
                    q8_b23 = _prmt_b32_175;
                    uint32_t _prmt_b32_176;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_b32_176) : "r"(q8_b01), "r"(q8_b23));
                    q8_word = _prmt_b32_176;
                }
                q8_swz = wide_feature ^ token_59 % 8 * 16;
                epi_staging_u32[(token_59 * 128 + q8_swz) / 4] = q8_word;
                int token_60 = wide_token + 64 + 56;
                {
                    wide_values[0] = _tmem_load_1[28];
                    wide_values[1] = _tmem_load_1[30];
                    wide_values[2] = _tmem_load_3[28];
                    wide_values[3] = _tmem_load_3[30];
                }
                {
                    float _fma_120 = __fmaf_rn(wide_values[0], q8_inv, 8388736.0f);
                    q8_r0 = _fma_120;
                    float _fma_121 = __fmaf_rn(wide_values[1], q8_inv, 8388736.0f);
                    q8_r1 = _fma_121;
                    float _fma_122 = __fmaf_rn(wide_values[2], q8_inv, 8388736.0f);
                    q8_r2 = _fma_122;
                    float _fma_123 = __fmaf_rn(wide_values[3], q8_inv, 8388736.0f);
                    q8_r3 = _fma_123;
                    uint32_t _prmt_b32_180;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_180) : "r"(__as_u32(q8_r0)), "r"(__as_u32(q8_r1)));
                    q8_b01 = _prmt_b32_180;
                    uint32_t _prmt_b32_181;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_181) : "r"(__as_u32(q8_r2)), "r"(__as_u32(q8_r3)));
                    q8_b23 = _prmt_b32_181;
                    uint32_t _prmt_b32_182;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_b32_182) : "r"(q8_b01), "r"(q8_b23));
                    q8_word = _prmt_b32_182;
                }
                q8_swz = wide_feature ^ token_60 % 8 * 16;
                epi_staging_u32[(token_60 * 128 + q8_swz) / 4] = q8_word;
                int token_61 = wide_token + 64 + 56 + 1;
                {
                    wide_values[0] = _tmem_load_1[29];
                    wide_values[1] = _tmem_load_1[31];
                    wide_values[2] = _tmem_load_3[29];
                    wide_values[3] = _tmem_load_3[31];
                }
                {
                    float _fma_124 = __fmaf_rn(wide_values[0], q8_inv, 8388736.0f);
                    q8_r0 = _fma_124;
                    float _fma_125 = __fmaf_rn(wide_values[1], q8_inv, 8388736.0f);
                    q8_r1 = _fma_125;
                    float _fma_126 = __fmaf_rn(wide_values[2], q8_inv, 8388736.0f);
                    q8_r2 = _fma_126;
                    float _fma_127 = __fmaf_rn(wide_values[3], q8_inv, 8388736.0f);
                    q8_r3 = _fma_127;
                    uint32_t _prmt_b32_186;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_186) : "r"(__as_u32(q8_r0)), "r"(__as_u32(q8_r1)));
                    q8_b01 = _prmt_b32_186;
                    uint32_t _prmt_b32_187;
                    asm("prmt.b32 %0, %1, %2, 0x0040;" : "=r"(_prmt_b32_187) : "r"(__as_u32(q8_r2)), "r"(__as_u32(q8_r3)));
                    q8_b23 = _prmt_b32_187;
                    uint32_t _prmt_b32_188;
                    asm("prmt.b32 %0, %1, %2, 0x5410;" : "=r"(_prmt_b32_188) : "r"(q8_b01), "r"(q8_b23));
                    q8_word = _prmt_b32_188;
                }
                q8_swz = wide_feature ^ token_61 % 8 * 16;
                epi_staging_u32[(token_61 * 128 + q8_swz) / 4] = q8_word;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                if (warp == 0) {
                    if (elect_sync()) {
                        int padding_rows = (128 - _mn_limit % 128) % 128;
                        int local_token = padding_rows;
                        partial_scale[n_tile * (unsigned int)grid_m + m_tile] = q8_scale;
                        tma_store_4d((&C_tma), off_m, local_token, 1073741824, token_base - padding_rows + 1073741824, epi_staging_addr);
                    }
                }
                asm volatile("cp.async.bulk.commit_group;");
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                if (elect_sync()) {
                    mbarrier_arrive(mma_free_addr + (acc_stage) * 8);
                }
                acc_stage += 1;
                if (acc_stage == 2) { acc_stage = 0; _phase_mma_full ^= 1; }
                {
                    mbarrier_wait_hint(work_full_addr + (work_stage) * 8, _phase_work_full, 1000);
                }
                unsigned int valid_1 = 0;
                unsigned int next_x_1 = 0;
                unsigned int next_y_1 = 0;
                int response_index_fd_1 = work_stage * 4;
                valid_1 = work_response[response_index_fd_1];
                next_x_1 = work_response[response_index_fd_1 + 1];
                next_y_1 = work_response[response_index_fd_1 + 2];
                mbarrier_arrive(work_empty_addr + (work_stage) * 8);
                work_stage += 1;
                if (work_stage == 5) { work_stage = 0; _phase_work_full ^= 1; }
                unsigned int valid_62 = valid_1;
                m_tile = next_x_1;
                n_tile = next_y_1;
                if (valid_62 == 0) {
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
                        mbarrier_wait_hint(work_full_addr + (work_stage_1) * 8, _phase_work_full_1, 1000);
                    }
                    unsigned int valid_2 = 0;
                    unsigned int next_x_2 = 0;
                    unsigned int next_y_2 = 0;
                    int response_index_fd_2 = work_stage_1 * 4;
                    valid_2 = work_response[response_index_fd_2];
                    next_x_2 = work_response[response_index_fd_2 + 1];
                    next_y_2 = work_response[response_index_fd_2 + 2];
                    mbarrier_arrive(work_empty_addr + (work_stage_1) * 8);
                    work_stage_1 += 1;
                    if (work_stage_1 == 5) { work_stage_1 = 0; _phase_work_full_1 ^= 1; }
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
                    mbarrier_wait_hint(work_full_addr + (work_stage_1) * 8, _phase_work_full_1, 1000);
                }
                unsigned int valid_3 = 0;
                unsigned int next_x_3 = 0;
                unsigned int next_y_3 = 0;
                int response_index_fd_3 = work_stage_1 * 4;
                valid_3 = work_response[response_index_fd_3];
                next_x_3 = work_response[response_index_fd_3 + 1];
                next_y_3 = work_response[response_index_fd_3 + 2];
                mbarrier_arrive(work_empty_addr + (work_stage_1) * 8);
                work_stage_1 += 1;
                if (work_stage_1 == 5) { work_stage_1 = 0; _phase_work_full_1 ^= 1; }
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
                        mbarrier_wait_hint(work_full_addr + (work_stage_2) * 8, _phase_work_full_2, 1000);
                    }
                    unsigned int valid_4 = 0;
                    unsigned int next_x_4 = 0;
                    unsigned int next_y_4 = 0;
                    int response_index_fd_4 = work_stage_2 * 4;
                    valid_4 = work_response[response_index_fd_4];
                    next_x_4 = work_response[response_index_fd_4 + 1];
                    next_y_4 = work_response[response_index_fd_4 + 2];
                    mbarrier_arrive(work_empty_addr + (work_stage_2) * 8);
                    work_stage_2 += 1;
                    if (work_stage_2 == 5) { work_stage_2 = 0; _phase_work_full_2 ^= 1; }
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
                    mbarrier_wait_hint(work_full_addr + (work_stage_2) * 8, _phase_work_full_2, 1000);
                }
                unsigned int valid_5 = 0;
                unsigned int next_x_5 = 0;
                unsigned int next_y_5 = 0;
                int response_index_fd_5 = work_stage_2 * 4;
                valid_5 = work_response[response_index_fd_5];
                next_x_5 = work_response[response_index_fd_5 + 1];
                next_y_5 = work_response[response_index_fd_5 + 2];
                mbarrier_arrive(work_empty_addr + (work_stage_2) * 8);
                work_stage_2 += 1;
                if (work_stage_2 == 5) { work_stage_2 = 0; _phase_work_full_2 ^= 1; }
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
                        mbarrier_wait_hint(work_full_addr + (work_stage_3) * 8, _phase_work_full_3, 1000);
                    }
                    unsigned int valid_6 = 0;
                    unsigned int next_x_6 = 0;
                    unsigned int next_y_6 = 0;
                    int response_index_fd_6 = work_stage_3 * 4;
                    valid_6 = work_response[response_index_fd_6];
                    next_x_6 = work_response[response_index_fd_6 + 1];
                    next_y_6 = work_response[response_index_fd_6 + 2];
                    mbarrier_arrive(work_empty_addr + (work_stage_3) * 8);
                    work_stage_3 += 1;
                    if (work_stage_3 == 5) { work_stage_3 = 0; _phase_work_full_3 ^= 1; }
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
                mbarrier_wait_hint(throttle_empty_addr + (throttle_stage) * 8, _phase_throttle_empty, 10000000);
                mbarrier_arrive(throttle_full_addr + (throttle_stage) * 8);
                throttle_stage += 1;
                if (throttle_stage == 5) { throttle_stage = 0; _phase_throttle_empty ^= 1; }
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
                    mbarrier_wait_hint(work_full_addr + (work_stage_3) * 8, _phase_work_full_3, 1000);
                }
                unsigned int valid_7 = 0;
                unsigned int next_x_7 = 0;
                unsigned int next_y_7 = 0;
                int response_index_fd_7 = work_stage_3 * 4;
                valid_7 = work_response[response_index_fd_7];
                next_x_7 = work_response[response_index_fd_7 + 1];
                next_y_7 = work_response[response_index_fd_7 + 2];
                mbarrier_arrive(work_empty_addr + (work_stage_3) * 8);
                work_stage_3 += 1;
                if (work_stage_3 == 5) { work_stage_3 = 0; _phase_work_full_3 ^= 1; }
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
                        mbarrier_wait_hint(work_full_addr + (work_stage_4) * 8, _phase_work_full_4, 1000);
                    }
                    unsigned int valid_8 = 0;
                    unsigned int next_x_8 = 0;
                    unsigned int next_y_8 = 0;
                    int response_index_fd_8 = work_stage_4 * 4;
                    valid_8 = work_response[response_index_fd_8];
                    next_x_8 = work_response[response_index_fd_8 + 1];
                    next_y_8 = work_response[response_index_fd_8 + 2];
                    mbarrier_arrive(work_empty_addr + (work_stage_4) * 8);
                    work_stage_4 += 1;
                    if (work_stage_4 == 5) { work_stage_4 = 0; _phase_work_full_4 ^= 1; }
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
                    mbarrier_wait_hint(work_full_addr + (work_stage_4) * 8, _phase_work_full_4, 1000);
                }
                unsigned int valid_9 = 0;
                unsigned int next_x_9 = 0;
                unsigned int next_y_9 = 0;
                int response_index_fd_9 = work_stage_4 * 4;
                valid_9 = work_response[response_index_fd_9];
                next_x_9 = work_response[response_index_fd_9 + 1];
                next_y_9 = work_response[response_index_fd_9 + 2];
                mbarrier_arrive(work_empty_addr + (work_stage_4) * 8);
                work_stage_4 += 1;
                if (work_stage_4 == 5) { work_stage_4 = 0; _phase_work_full_4 ^= 1; }
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
                        mbarrier_wait_hint(work_full_addr + (work_stage_5) * 8, _phase_work_full_5, 1000);
                    }
                    unsigned int valid_10 = 0;
                    unsigned int next_x_10 = 0;
                    unsigned int next_y_10 = 0;
                    int response_index_fd_10 = work_stage_5 * 4;
                    valid_10 = work_response[response_index_fd_10];
                    next_x_10 = work_response[response_index_fd_10 + 1];
                    next_y_10 = work_response[response_index_fd_10 + 2];
                    mbarrier_arrive(work_empty_addr + (work_stage_5) * 8);
                    work_stage_5 += 1;
                    if (work_stage_5 == 5) { work_stage_5 = 0; _phase_work_full_5 ^= 1; }
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
                            uint64_t _tcgen05_cp_desc_0 = ((((uint64_t)(smem_sfa_addr + stage_4 * 2048)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                :: "r"((uint32_t)((unsigned int)tmem_sfa + stage_4 * 16)), "l"(_tcgen05_cp_desc_0)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_1 = ((((uint64_t)(smem_sfa_addr + stage_4 * 2048 + 512)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_4 * 16 + 4))), "l"(_tcgen05_cp_desc_1)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_2 = ((((uint64_t)(smem_sfa_addr + stage_4 * 2048 + 1024)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_4 * 16 + 8))), "l"(_tcgen05_cp_desc_2)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_3 = ((((uint64_t)(smem_sfa_addr + stage_4 * 2048 + 1536)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
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
                                uint64_t _tcgen05_cp_desc_4 = ((((uint64_t)(smem_sfb_addr + stage_4 * 2048 + (unsigned int)(k_set * 512))) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL) | (0ULL << 61ULL);
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
                    mbarrier_wait_hint(work_full_addr + (work_stage_5) * 8, _phase_work_full_5, 1000);
                }
                unsigned int valid_11 = 0;
                unsigned int next_x_11 = 0;
                unsigned int next_y_11 = 0;
                int response_index_fd_11 = work_stage_5 * 4;
                valid_11 = work_response[response_index_fd_11];
                next_x_11 = work_response[response_index_fd_11 + 1];
                next_y_11 = work_response[response_index_fd_11 + 2];
                mbarrier_arrive(work_empty_addr + (work_stage_5) * 8);
                work_stage_5 += 1;
                if (work_stage_5 == 5) { work_stage_5 = 0; _phase_work_full_5 ^= 1; }
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
                        mbarrier_wait_hint(work_full_addr + (work_stage_6) * 8, _phase_work_full_6, 1000);
                    }
                    unsigned int valid_12 = 0;
                    unsigned int next_x_12 = 0;
                    unsigned int next_y_12 = 0;
                    int response_index_fd_12 = work_stage_6 * 4;
                    valid_12 = work_response[response_index_fd_12];
                    next_x_12 = work_response[response_index_fd_12 + 1];
                    next_y_12 = work_response[response_index_fd_12 + 2];
                    mbarrier_arrive(work_empty_addr + (work_stage_6) * 8);
                    work_stage_6 += 1;
                    if (work_stage_6 == 5) { work_stage_6 = 0; _phase_work_full_6 ^= 1; }
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
                    {
                        int sfa_base_col = 0;
                        int sfb_base_col = 0;
                        {
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
                        }
                    }
                    {
                        int sfa_base_col_1 = 4;
                        int sfb_base_col_1 = 4;
                        {
                            int _mma_a_lo_1 = make_warp_uniform((((smem_a_mma1_addr) >> 4) & 0x3FFF) + (k_stage) * 1024);
                            int _mma_b_lo_1 = make_warp_uniform((((smem_b_mma1_addr) >> 4) & 0x3FFF) + (k_stage) * 1024);
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs((tmem_accum + ((acc_stage_1 + (unsigned int)route_slot) * 128)), a_desc + 0, b_desc + 0,
                                        0x8200480U, (unsigned int)tmem_sfa + (k_stage * 16 + (unsigned int)sfa_base_col_1) + 0, (unsigned int)tmem_sfb + (k_stage * 16 + (unsigned int)sfb_base_col_1) + 0, ((((0) ? ((0) ? ((route_k_4 % K_tiles == 0) ? 1 : 0) : ((route_k_4 == 0) ? 1 : 0)) : 0)) ? 0 : 1));
                                }
                            }
                        }
                    }
                    {
                        int sfa_base_col_2 = 8;
                        int sfb_base_col_2 = 8;
                        if (!1 || route_k_4 == 0) {
                            int _mma_a_lo_2 = make_warp_uniform((((smem_a_mma2_addr) >> 4) & 0x3FFF) + (k_stage) * 1024);
                            int _mma_b_lo_2 = make_warp_uniform((((smem_b_mma2_addr) >> 4) & 0x3FFF) + (k_stage) * 1024);
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_2) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_2) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs((tmem_accum + ((acc_stage_1 + (unsigned int)route_slot) * 128)), a_desc + 0, b_desc + 0,
                                        0x8200480U, (unsigned int)tmem_sfa + (k_stage * 16 + (unsigned int)sfa_base_col_2) + 0, (unsigned int)tmem_sfb + (k_stage * 16 + (unsigned int)sfb_base_col_2) + 0, ((((0) ? ((0) ? ((route_k_4 % K_tiles == 0) ? 1 : 0) : ((route_k_4 == 0) ? 1 : 0)) : 0)) ? 0 : 1));
                                }
                            }
                        }
                    }
                    {
                        int sfa_base_col_3 = 12;
                        int sfb_base_col_3 = 12;
                        if (!1 || route_k_4 == 0) {
                            int _mma_a_lo_3 = make_warp_uniform((((smem_a_mma3_addr) >> 4) & 0x3FFF) + (k_stage) * 1024);
                            int _mma_b_lo_3 = make_warp_uniform((((smem_b_mma3_addr) >> 4) & 0x3FFF) + (k_stage) * 1024);
                            if (elect_sync()) {
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_3) | ((uint64_t)0x40004040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_3) | ((uint64_t)0x40004040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs((tmem_accum + ((acc_stage_1 + (unsigned int)route_slot) * 128)), a_desc + 0, b_desc + 0,
                                        0x8200480U, (unsigned int)tmem_sfa + (k_stage * 16 + (unsigned int)sfa_base_col_3) + 0, (unsigned int)tmem_sfb + (k_stage * 16 + (unsigned int)sfb_base_col_3) + 0, ((((0) ? ((0) ? ((route_k_4 % K_tiles == 0) ? 1 : 0) : ((route_k_4 == 0) ? 1 : 0)) : 0)) ? 0 : 1));
                                }
                            }
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
                    mbarrier_wait_hint(work_full_addr + (work_stage_6) * 8, _phase_work_full_6, 1000);
                }
                unsigned int valid_13 = 0;
                unsigned int next_x_13 = 0;
                unsigned int next_y_13 = 0;
                int response_index_fd_13 = work_stage_6 * 4;
                valid_13 = work_response[response_index_fd_13];
                next_x_13 = work_response[response_index_fd_13 + 1];
                next_y_13 = work_response[response_index_fd_13 + 2];
                mbarrier_arrive(work_empty_addr + (work_stage_6) * 8);
                work_stage_6 += 1;
                if (work_stage_6 == 5) { work_stage_6 = 0; _phase_work_full_6 ^= 1; }
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
            unsigned int _phase_throttle_full = 0;
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_drain_full_0 = 0;
            unsigned int _phase_work_full_7 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_7 = 0; _tile_iter_7 < grid_m * grid_n; _tile_iter_7++) {
                if (m_tile_7 >= (unsigned int)grid_m || n_tile_7 >= (unsigned int)grid_n) {
                    break;
                }
                mbarrier_wait_hint(throttle_full_addr + (throttle_stage_1) * 8, _phase_throttle_full, 10000000);
                mbarrier_arrive(throttle_empty_addr + (throttle_stage_1) * 8);
                throttle_stage_1 += 1;
                if (throttle_stage_1 == 5) { throttle_stage_1 = 0; _phase_throttle_full ^= 1; }
                mbarrier_wait(work_empty_addr + (work_stage_7) * 8, _phase_work_empty);
                if (elect_sync()) {
                    mbarrier_arrive_expect_tx(drain_full_addr, 16);
                    asm volatile(
                        "fence.proxy.async.shared::cta;\n\t"
                        "clusterlaunchcontrol.try_cancel.async.shared::cta"
                            ".mbarrier::complete_tx::bytes.b128"
                            " [%0], [%1];"
                        :: "r"(drain_response_addr + 0 * 64 + 0 * 16), "r"(drain_full_addr + 0 * 8)
                        : "memory");
                }
                mbarrier_wait(drain_full_addr, _phase_drain_full_0);
                _phase_drain_full_0 ^= 1;
                unsigned int slot_valid = 0;
                unsigned int slot_x = 0;
                unsigned int slot_y = 0;
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
                    : "r"(drain_response_addr + 0 * 64 + 0 * 16)
                    : "memory");
                slot_valid = _clc_valid_0;
                if (slot_valid != 0) {
                    uint32_t _clc_ctaid_0 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_0)
                        : "r"(drain_response_addr + 0 * 64 + 0 * 16)
                        : "memory");
                    slot_x = _clc_ctaid_0;
                    uint32_t _clc_ctaid_1 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_1)
                        : "r"(drain_response_addr + 0 * 64 + 0 * 16)
                        : "memory");
                    slot_y = _clc_ctaid_1;
                }
                unsigned int pub_valid = slot_valid;
                unsigned int pub_x = slot_x;
                unsigned int pub_y = slot_y;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                unsigned int excess = 0;
                if (pub_valid != 0) {
                    if (pub_y >= (unsigned int)total_tiles[0]) {
                        excess = 1;
                    }
                }
                unsigned int publish_pending = 1;
                if (excess != 0) {
                    unsigned int drained = 0;
                    unsigned int credit = 1;
                    #pragma unroll 1
                    for (unsigned int _drain_round = 0; _drain_round < grid_m * grid_n; _drain_round++) {
                        unsigned int real_seen = 0;
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(drain_full_addr, 64);
                            asm volatile(
                                "fence.proxy.async.shared::cta;\n\t"
                                "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                    ".mbarrier::complete_tx::bytes.b128"
                                    " [%0], [%1];"
                                :: "r"(drain_response_addr + 0 * 64 + 0 * 16), "r"(drain_full_addr + 0 * 8)
                                : "memory");
                            asm volatile(
                                "fence.proxy.async.shared::cta;\n\t"
                                "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                    ".mbarrier::complete_tx::bytes.b128"
                                    " [%0], [%1];"
                                :: "r"(drain_response_addr + 0 * 64 + 1 * 16), "r"(drain_full_addr + 0 * 8)
                                : "memory");
                            asm volatile(
                                "fence.proxy.async.shared::cta;\n\t"
                                "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                    ".mbarrier::complete_tx::bytes.b128"
                                    " [%0], [%1];"
                                :: "r"(drain_response_addr + 0 * 64 + 2 * 16), "r"(drain_full_addr + 0 * 8)
                                : "memory");
                            asm volatile(
                                "fence.proxy.async.shared::cta;\n\t"
                                "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                    ".mbarrier::complete_tx::bytes.b128"
                                    " [%0], [%1];"
                                :: "r"(drain_response_addr + 0 * 64 + 3 * 16), "r"(drain_full_addr + 0 * 8)
                                : "memory");
                        }
                        mbarrier_wait(drain_full_addr, _phase_drain_full_0);
                        _phase_drain_full_0 ^= 1;
                        unsigned int slot_valid_0 = 0;
                        unsigned int slot_x_1 = 0;
                        unsigned int slot_y_2 = 0;
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
                            : "r"(drain_response_addr + 0 * 64 + 0 * 16)
                            : "memory");
                        slot_valid_0 = _clc_valid_1;
                        if (slot_valid_0 != 0) {
                            uint32_t _clc_ctaid_2 = 0;
                            asm volatile(
                                "{\n\t"
                                ".reg .pred p1;\n\t"
                                ".reg .b128 clc_r;\n\t"
                                "ld.shared.b128 clc_r, [%1];\n\t"
                                "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                                "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                                "}\n"
                                : "=r"(_clc_ctaid_2)
                                : "r"(drain_response_addr + 0 * 64 + 0 * 16)
                                : "memory");
                            slot_x_1 = _clc_ctaid_2;
                            uint32_t _clc_ctaid_3 = 0;
                            asm volatile(
                                "{\n\t"
                                ".reg .pred p1;\n\t"
                                ".reg .b128 clc_r;\n\t"
                                "ld.shared.b128 clc_r, [%1];\n\t"
                                "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                                "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                                "}\n"
                                : "=r"(_clc_ctaid_3)
                                : "r"(drain_response_addr + 0 * 64 + 0 * 16)
                                : "memory");
                            slot_y_2 = _clc_ctaid_3;
                        }
                        if (slot_valid_0 == 0) {
                            drained = 1;
                        } else if (slot_y_2 < (unsigned int)total_tiles[0]) {
                            if (credit == 0) {
                                mbarrier_wait_hint(throttle_full_addr + (throttle_stage_1) * 8, _phase_throttle_full, 10000000);
                                mbarrier_arrive(throttle_empty_addr + (throttle_stage_1) * 8);
                                throttle_stage_1 += 1;
                                if (throttle_stage_1 == 5) { throttle_stage_1 = 0; _phase_throttle_full ^= 1; }
                                mbarrier_wait(work_empty_addr + (work_stage_7) * 8, _phase_work_empty);
                            }
                            credit = 0;
                            if (elect_sync()) {
                                int response_index = work_stage_7 * 4;
                                work_response[response_index] = 1;
                                work_response[response_index + 1] = slot_x_1;
                                work_response[response_index + 2] = slot_y_2;
                                work_response[response_index + 3] = 0;
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                mbarrier_arrive(work_full_addr + (work_stage_7) * 8);
                            }
                            {
                                mbarrier_wait_hint(work_full_addr + (work_stage_7) * 8, _phase_work_full_7, 1000);
                            }
                            unsigned int valid_14 = 0;
                            unsigned int next_x_14 = 0;
                            unsigned int next_y_14 = 0;
                            int response_index_fd_14 = work_stage_7 * 4;
                            valid_14 = work_response[response_index_fd_14];
                            next_x_14 = work_response[response_index_fd_14 + 1];
                            next_y_14 = work_response[response_index_fd_14 + 2];
                            mbarrier_arrive(work_empty_addr + (work_stage_7) * 8);
                            work_stage_7 += 1;
                            if (work_stage_7 == 5) { work_stage_7 = 0; _phase_work_empty ^= 1; _phase_work_full_7 ^= 1; }
                            m_tile_7 = next_x_14;
                            n_tile_7 = next_y_14;
                            real_seen = 1;
                        }
                        unsigned int slot_valid_3 = 0;
                        unsigned int slot_x_4 = 0;
                        unsigned int slot_y_5 = 0;
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
                            : "r"(drain_response_addr + 0 * 64 + 1 * 16)
                            : "memory");
                        slot_valid_3 = _clc_valid_2;
                        if (slot_valid_3 != 0) {
                            uint32_t _clc_ctaid_4 = 0;
                            asm volatile(
                                "{\n\t"
                                ".reg .pred p1;\n\t"
                                ".reg .b128 clc_r;\n\t"
                                "ld.shared.b128 clc_r, [%1];\n\t"
                                "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                                "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                                "}\n"
                                : "=r"(_clc_ctaid_4)
                                : "r"(drain_response_addr + 0 * 64 + 1 * 16)
                                : "memory");
                            slot_x_4 = _clc_ctaid_4;
                            uint32_t _clc_ctaid_5 = 0;
                            asm volatile(
                                "{\n\t"
                                ".reg .pred p1;\n\t"
                                ".reg .b128 clc_r;\n\t"
                                "ld.shared.b128 clc_r, [%1];\n\t"
                                "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                                "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                                "}\n"
                                : "=r"(_clc_ctaid_5)
                                : "r"(drain_response_addr + 0 * 64 + 1 * 16)
                                : "memory");
                            slot_y_5 = _clc_ctaid_5;
                        }
                        if (slot_valid_3 == 0) {
                            drained = 1;
                        } else if (slot_y_5 < (unsigned int)total_tiles[0]) {
                            if (credit == 0) {
                                mbarrier_wait_hint(throttle_full_addr + (throttle_stage_1) * 8, _phase_throttle_full, 10000000);
                                mbarrier_arrive(throttle_empty_addr + (throttle_stage_1) * 8);
                                throttle_stage_1 += 1;
                                if (throttle_stage_1 == 5) { throttle_stage_1 = 0; _phase_throttle_full ^= 1; }
                                mbarrier_wait(work_empty_addr + (work_stage_7) * 8, _phase_work_empty);
                            }
                            credit = 0;
                            if (elect_sync()) {
                                int response_index_1 = work_stage_7 * 4;
                                work_response[response_index_1] = 1;
                                work_response[response_index_1 + 1] = slot_x_4;
                                work_response[response_index_1 + 2] = slot_y_5;
                                work_response[response_index_1 + 3] = 0;
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                mbarrier_arrive(work_full_addr + (work_stage_7) * 8);
                            }
                            {
                                mbarrier_wait_hint(work_full_addr + (work_stage_7) * 8, _phase_work_full_7, 1000);
                            }
                            unsigned int valid_15 = 0;
                            unsigned int next_x_15 = 0;
                            unsigned int next_y_15 = 0;
                            int response_index_fd_15 = work_stage_7 * 4;
                            valid_15 = work_response[response_index_fd_15];
                            next_x_15 = work_response[response_index_fd_15 + 1];
                            next_y_15 = work_response[response_index_fd_15 + 2];
                            mbarrier_arrive(work_empty_addr + (work_stage_7) * 8);
                            work_stage_7 += 1;
                            if (work_stage_7 == 5) { work_stage_7 = 0; _phase_work_empty ^= 1; _phase_work_full_7 ^= 1; }
                            m_tile_7 = next_x_15;
                            n_tile_7 = next_y_15;
                            real_seen = 1;
                        }
                        unsigned int slot_valid_6 = 0;
                        unsigned int slot_x_7 = 0;
                        unsigned int slot_y_8 = 0;
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
                            : "r"(drain_response_addr + 0 * 64 + 2 * 16)
                            : "memory");
                        slot_valid_6 = _clc_valid_3;
                        if (slot_valid_6 != 0) {
                            uint32_t _clc_ctaid_6 = 0;
                            asm volatile(
                                "{\n\t"
                                ".reg .pred p1;\n\t"
                                ".reg .b128 clc_r;\n\t"
                                "ld.shared.b128 clc_r, [%1];\n\t"
                                "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                                "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                                "}\n"
                                : "=r"(_clc_ctaid_6)
                                : "r"(drain_response_addr + 0 * 64 + 2 * 16)
                                : "memory");
                            slot_x_7 = _clc_ctaid_6;
                            uint32_t _clc_ctaid_7 = 0;
                            asm volatile(
                                "{\n\t"
                                ".reg .pred p1;\n\t"
                                ".reg .b128 clc_r;\n\t"
                                "ld.shared.b128 clc_r, [%1];\n\t"
                                "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                                "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                                "}\n"
                                : "=r"(_clc_ctaid_7)
                                : "r"(drain_response_addr + 0 * 64 + 2 * 16)
                                : "memory");
                            slot_y_8 = _clc_ctaid_7;
                        }
                        if (slot_valid_6 == 0) {
                            drained = 1;
                        } else if (slot_y_8 < (unsigned int)total_tiles[0]) {
                            if (credit == 0) {
                                mbarrier_wait_hint(throttle_full_addr + (throttle_stage_1) * 8, _phase_throttle_full, 10000000);
                                mbarrier_arrive(throttle_empty_addr + (throttle_stage_1) * 8);
                                throttle_stage_1 += 1;
                                if (throttle_stage_1 == 5) { throttle_stage_1 = 0; _phase_throttle_full ^= 1; }
                                mbarrier_wait(work_empty_addr + (work_stage_7) * 8, _phase_work_empty);
                            }
                            credit = 0;
                            if (elect_sync()) {
                                int response_index_2 = work_stage_7 * 4;
                                work_response[response_index_2] = 1;
                                work_response[response_index_2 + 1] = slot_x_7;
                                work_response[response_index_2 + 2] = slot_y_8;
                                work_response[response_index_2 + 3] = 0;
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                mbarrier_arrive(work_full_addr + (work_stage_7) * 8);
                            }
                            {
                                mbarrier_wait_hint(work_full_addr + (work_stage_7) * 8, _phase_work_full_7, 1000);
                            }
                            unsigned int valid_16 = 0;
                            unsigned int next_x_16 = 0;
                            unsigned int next_y_16 = 0;
                            int response_index_fd_16 = work_stage_7 * 4;
                            valid_16 = work_response[response_index_fd_16];
                            next_x_16 = work_response[response_index_fd_16 + 1];
                            next_y_16 = work_response[response_index_fd_16 + 2];
                            mbarrier_arrive(work_empty_addr + (work_stage_7) * 8);
                            work_stage_7 += 1;
                            if (work_stage_7 == 5) { work_stage_7 = 0; _phase_work_empty ^= 1; _phase_work_full_7 ^= 1; }
                            m_tile_7 = next_x_16;
                            n_tile_7 = next_y_16;
                            real_seen = 1;
                        }
                        unsigned int slot_valid_9 = 0;
                        unsigned int slot_x_10 = 0;
                        unsigned int slot_y_11 = 0;
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
                            : "r"(drain_response_addr + 0 * 64 + 3 * 16)
                            : "memory");
                        slot_valid_9 = _clc_valid_4;
                        if (slot_valid_9 != 0) {
                            uint32_t _clc_ctaid_8 = 0;
                            asm volatile(
                                "{\n\t"
                                ".reg .pred p1;\n\t"
                                ".reg .b128 clc_r;\n\t"
                                "ld.shared.b128 clc_r, [%1];\n\t"
                                "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                                "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                                "}\n"
                                : "=r"(_clc_ctaid_8)
                                : "r"(drain_response_addr + 0 * 64 + 3 * 16)
                                : "memory");
                            slot_x_10 = _clc_ctaid_8;
                            uint32_t _clc_ctaid_9 = 0;
                            asm volatile(
                                "{\n\t"
                                ".reg .pred p1;\n\t"
                                ".reg .b128 clc_r;\n\t"
                                "ld.shared.b128 clc_r, [%1];\n\t"
                                "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                                "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                                "}\n"
                                : "=r"(_clc_ctaid_9)
                                : "r"(drain_response_addr + 0 * 64 + 3 * 16)
                                : "memory");
                            slot_y_11 = _clc_ctaid_9;
                        }
                        if (slot_valid_9 == 0) {
                            drained = 1;
                        } else if (slot_y_11 < (unsigned int)total_tiles[0]) {
                            if (credit == 0) {
                                mbarrier_wait_hint(throttle_full_addr + (throttle_stage_1) * 8, _phase_throttle_full, 10000000);
                                mbarrier_arrive(throttle_empty_addr + (throttle_stage_1) * 8);
                                throttle_stage_1 += 1;
                                if (throttle_stage_1 == 5) { throttle_stage_1 = 0; _phase_throttle_full ^= 1; }
                                mbarrier_wait(work_empty_addr + (work_stage_7) * 8, _phase_work_empty);
                            }
                            credit = 0;
                            if (elect_sync()) {
                                int response_index_3 = work_stage_7 * 4;
                                work_response[response_index_3] = 1;
                                work_response[response_index_3 + 1] = slot_x_10;
                                work_response[response_index_3 + 2] = slot_y_11;
                                work_response[response_index_3 + 3] = 0;
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                mbarrier_arrive(work_full_addr + (work_stage_7) * 8);
                            }
                            {
                                mbarrier_wait_hint(work_full_addr + (work_stage_7) * 8, _phase_work_full_7, 1000);
                            }
                            unsigned int valid_17 = 0;
                            unsigned int next_x_17 = 0;
                            unsigned int next_y_17 = 0;
                            int response_index_fd_17 = work_stage_7 * 4;
                            valid_17 = work_response[response_index_fd_17];
                            next_x_17 = work_response[response_index_fd_17 + 1];
                            next_y_17 = work_response[response_index_fd_17 + 2];
                            mbarrier_arrive(work_empty_addr + (work_stage_7) * 8);
                            work_stage_7 += 1;
                            if (work_stage_7 == 5) { work_stage_7 = 0; _phase_work_empty ^= 1; _phase_work_full_7 ^= 1; }
                            m_tile_7 = next_x_17;
                            n_tile_7 = next_y_17;
                            real_seen = 1;
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        if (drained != 0) {
                            break;
                        }
                        if (real_seen != 0) {
                            break;
                        }
                    }
                    if (drained != 0) {
                        if (credit == 0) {
                            mbarrier_wait(work_empty_addr + (work_stage_7) * 8, _phase_work_empty);
                        }
                        pub_valid = 0;
                    } else {
                        publish_pending = 0;
                    }
                }
                if (publish_pending != 0) {
                    if (elect_sync()) {
                        int response_index_4 = work_stage_7 * 4;
                        work_response[response_index_4] = pub_valid;
                        work_response[response_index_4 + 1] = pub_x;
                        work_response[response_index_4 + 2] = pub_y;
                        work_response[response_index_4 + 3] = 0;
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive(work_full_addr + (work_stage_7) * 8);
                    }
                    {
                        mbarrier_wait_hint(work_full_addr + (work_stage_7) * 8, _phase_work_full_7, 1000);
                    }
                    unsigned int valid_18 = 0;
                    unsigned int next_x_18 = 0;
                    unsigned int next_y_18 = 0;
                    int response_index_fd_18 = work_stage_7 * 4;
                    valid_18 = work_response[response_index_fd_18];
                    next_x_18 = work_response[response_index_fd_18 + 1];
                    next_y_18 = work_response[response_index_fd_18 + 2];
                    mbarrier_arrive(work_empty_addr + (work_stage_7) * 8);
                    work_stage_7 += 1;
                    if (work_stage_7 == 5) { work_stage_7 = 0; _phase_work_empty ^= 1; _phase_work_full_7 ^= 1; }
                    unsigned int valid_0_6 = valid_18;
                    m_tile_7 = next_x_18;
                    n_tile_7 = next_y_18;
                    if (valid_0_6 == 0) {
                        break;
                    }
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
                unsigned int inactive_valid_7 = 1;
                #pragma unroll 1
                for (unsigned int _inactive_iter_7 = 0; _inactive_iter_7 < grid_m * grid_n; _inactive_iter_7++) {
                    if (n_tile_8 < (unsigned int)total_tiles[0]) {
                        break;
                    }
                    {
                        mbarrier_wait_hint(work_full_addr + (work_stage_8) * 8, _phase_work_full_8, 1000);
                    }
                    unsigned int valid_19 = 0;
                    unsigned int next_x_19 = 0;
                    unsigned int next_y_19 = 0;
                    int response_index_fd_19 = work_stage_8 * 4;
                    valid_19 = work_response[response_index_fd_19];
                    next_x_19 = work_response[response_index_fd_19 + 1];
                    next_y_19 = work_response[response_index_fd_19 + 2];
                    mbarrier_arrive(work_empty_addr + (work_stage_8) * 8);
                    work_stage_8 += 1;
                    if (work_stage_8 == 5) { work_stage_8 = 0; _phase_work_full_8 ^= 1; }
                    inactive_valid_7 = valid_19;
                    m_tile_8 = next_x_19;
                    n_tile_8 = next_y_19;
                    if (inactive_valid_7 == 0) {
                        break;
                    }
                }
                if (inactive_valid_7 == 0) {
                    break;
                }
                if (m_tile_8 >= (unsigned int)grid_m || n_tile_8 >= (unsigned int)grid_n) {
                    break;
                }
                {
                    mbarrier_wait_hint(work_full_addr + (work_stage_8) * 8, _phase_work_full_8, 1000);
                }
                unsigned int valid_20 = 0;
                unsigned int next_x_20 = 0;
                unsigned int next_y_20 = 0;
                int response_index_fd_20 = work_stage_8 * 4;
                valid_20 = work_response[response_index_fd_20];
                next_x_20 = work_response[response_index_fd_20 + 1];
                next_y_20 = work_response[response_index_fd_20 + 2];
                mbarrier_arrive(work_empty_addr + (work_stage_8) * 8);
                work_stage_8 += 1;
                if (work_stage_8 == 5) { work_stage_8 = 0; _phase_work_full_8 ^= 1; }
                unsigned int valid_0_7 = valid_20;
                m_tile_8 = next_x_20;
                n_tile_8 = next_y_20;
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
