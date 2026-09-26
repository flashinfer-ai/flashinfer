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
#define TMEM_TMEM_STATE_0_OFFSET 64
#define TMEM_TMEM_STATE_INP_0_OFFSET 0
#define TMEM_TMEM_U_ACC_0_OFFSET 224
#define TMEM_TMEM_U2_INP_0_OFFSET 224
#define TMEM_TMEM_U2_ACC_0_OFFSET 0
#define TMEM_TMEM_OUT_0_OFFSET 192
#define TMEM_TMEM_STATE_OUT_0_OFFSET 64
#define TMEM_TMEM_STATE_1_OFFSET 320
#define TMEM_TMEM_STATE_INP_1_OFFSET 256
#define TMEM_TMEM_U_ACC_1_OFFSET 480
#define TMEM_TMEM_U2_INP_1_OFFSET 480
#define TMEM_TMEM_U2_ACC_1_OFFSET 256
#define TMEM_TMEM_OUT_1_OFFSET 448
#define TMEM_TMEM_STATE_OUT_1_OFFSET 320
#define NUM_M_PIPE_0_STAGES 1
#define NUM_OPS_PIPE_0_STAGES 1
#define NUM_M_PIPE_1_STAGES 1
#define NUM_OPS_PIPE_1_STAGES 1
#define SMEM_SMEM_CARRY_HI_OFF 1024
#define SMEM_SMEM_CARRY_HI_STAGE_BYTES 32768
#define SMEM_SMEM_CARRY_HI_STRIDE 32768
#define SMEM_SMEM_CARRY_LO_OFF 33792
#define SMEM_SMEM_CARRY_LO_STAGE_BYTES 32768
#define SMEM_SMEM_CARRY_LO_STRIDE 32768
#define SMEM_SMEM_QD_0_OFF 66560
#define SMEM_SMEM_QD_0_STAGE_BYTES 8192
#define SMEM_SMEM_QD_0_STRIDE 32768
#define SMEM_SMEM_KD_0_OFF 74752
#define SMEM_SMEM_KD_0_STAGE_BYTES 8192
#define SMEM_SMEM_KD_0_STRIDE 32768
#define SMEM_SMEM_FT_0_OFF 82944
#define SMEM_SMEM_FT_0_STAGE_BYTES 12288
#define SMEM_SMEM_FT_0_STRIDE 32768
#define SMEM_SMEM_MQK_0_OFF 91136
#define SMEM_SMEM_MQK_0_STAGE_BYTES 4096
#define SMEM_SMEM_MQK_0_STRIDE 32768
#define SMEM_SMEM_INV_0_OFF 95232
#define SMEM_SMEM_INV_0_STAGE_BYTES 2048
#define SMEM_SMEM_INV_0_STRIDE 32768
#define SMEM_SMEM_VEC_0_OFF 97280
#define SMEM_SMEM_VEC_0_STAGE_BYTES 1168
#define SMEM_SMEM_VEC_0_STRIDE 32768
#define SMEM_SMEM_VEC_ALL_0_OFF 97280
#define SMEM_SMEM_VEC_ALL_0_STAGE_BYTES 1168
#define SMEM_SMEM_VEC_ALL_0_STRIDE 1168
#define SMEM_SMEM_MAP_0_OFF 99328
#define SMEM_SMEM_MAP_0_STAGE_BYTES 32768
#define SMEM_SMEM_MAP_0_STRIDE 32768
#define SMEM_SMEM_OUT_0_OFF 132096
#define SMEM_SMEM_OUT_0_STAGE_BYTES 8192
#define SMEM_SMEM_OUT_0_STRIDE 8192
#define SMEM_SMEM_PANEL_0_OFF 132096
#define SMEM_SMEM_PANEL_0_STAGE_BYTES 16384
#define SMEM_SMEM_PANEL_0_STRIDE 16384
#define SMEM_SMEM_PANEL_BF16_0_OFF 132096
#define SMEM_SMEM_PANEL_BF16_0_STAGE_BYTES 8192
#define SMEM_SMEM_PANEL_BF16_0_STRIDE 8192
#define SMEM_SMEM_DPAIR_0_OFF 148480
#define SMEM_SMEM_DPAIR_0_STAGE_BYTES 512
#define SMEM_SMEM_DPAIR_0_STRIDE 512
#define SMEM_SMEM_QD_1_OFF 149504
#define SMEM_SMEM_QD_1_STAGE_BYTES 8192
#define SMEM_SMEM_QD_1_STRIDE 32768
#define SMEM_SMEM_KD_1_OFF 157696
#define SMEM_SMEM_KD_1_STAGE_BYTES 8192
#define SMEM_SMEM_KD_1_STRIDE 32768
#define SMEM_SMEM_FT_1_OFF 165888
#define SMEM_SMEM_FT_1_STAGE_BYTES 12288
#define SMEM_SMEM_FT_1_STRIDE 32768
#define SMEM_SMEM_MQK_1_OFF 174080
#define SMEM_SMEM_MQK_1_STAGE_BYTES 4096
#define SMEM_SMEM_MQK_1_STRIDE 32768
#define SMEM_SMEM_INV_1_OFF 178176
#define SMEM_SMEM_INV_1_STAGE_BYTES 2048
#define SMEM_SMEM_INV_1_STRIDE 32768
#define SMEM_SMEM_VEC_1_OFF 180224
#define SMEM_SMEM_VEC_1_STAGE_BYTES 1168
#define SMEM_SMEM_VEC_1_STRIDE 32768
#define SMEM_SMEM_VEC_ALL_1_OFF 180224
#define SMEM_SMEM_VEC_ALL_1_STAGE_BYTES 1168
#define SMEM_SMEM_VEC_ALL_1_STRIDE 1168
#define SMEM_SMEM_MAP_1_OFF 182272
#define SMEM_SMEM_MAP_1_STAGE_BYTES 32768
#define SMEM_SMEM_MAP_1_STRIDE 32768
#define SMEM_SMEM_OUT_1_OFF 215040
#define SMEM_SMEM_OUT_1_STAGE_BYTES 8192
#define SMEM_SMEM_OUT_1_STRIDE 8192
#define SMEM_SMEM_PANEL_1_OFF 215040
#define SMEM_SMEM_PANEL_1_STAGE_BYTES 16384
#define SMEM_SMEM_PANEL_1_STRIDE 16384
#define SMEM_SMEM_PANEL_BF16_1_OFF 215040
#define SMEM_SMEM_PANEL_BF16_1_STAGE_BYTES 8192
#define SMEM_SMEM_PANEL_BF16_1_STRIDE 8192
#define SMEM_SMEM_DPAIR_1_OFF 231424
#define SMEM_SMEM_DPAIR_1_STAGE_BYTES 512
#define SMEM_SMEM_DPAIR_1_STRIDE 512
#define SMEM_TOTAL 232448
#define THREADS 640

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


__device__ __forceinline__ void tmem_st_x8_u32(int addr, const uint32_t* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x8.b32"
        " [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
        :: "r"(addr),
           "r"(src[0]), "r"(src[1]), "r"(src[2]), "r"(src[3]),
           "r"(src[4]), "r"(src[5]), "r"(src[6]), "r"(src[7]));
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

__global__ __launch_bounds__(640) void
kernel_cake_kda_bf16_8c48c7b4062dd4c9d6370d42faf203d0d3759260600fa28ae603532efb43ad93(int* __restrict__ items, int num_heads, __nv_bfloat16* __restrict__ carry_hi, CakeTensorMap const* carry_hi_tma, __nv_bfloat16* __restrict__ carry_lo, CakeTensorMap const* carry_lo_tma, __nv_bfloat16* __restrict__ maps, CakeTensorMap const* maps_tma, __nv_bfloat16* __restrict__ map_final, CakeTensorMap const* map_final_tma, __nv_bfloat16* __restrict__ op_qd, __nv_bfloat16* __restrict__ op_kd, __nv_bfloat16* __restrict__ op_ft, __nv_bfloat16* __restrict__ op_inv, float* __restrict__ op_vec, __nv_bfloat16* __restrict__ out, CakeTensorMap const* out_tma, float* __restrict__ rows, CakeTensorMap const* rows_tma, long long* __restrict__ row_starts, float* __restrict__ final, CakeTensorMap const* final_tma, __nv_bfloat16* __restrict__ pair, CakeTensorMap const* pair_tma, float* __restrict__ dpair)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define m_full_0_addr (mbar_base + 0)
    #define m_free_0_addr (mbar_base + 8)
    #define ops_full_0_addr (mbar_base + 16)
    #define ops_free_0_addr (mbar_base + 24)
    #define state_ready_0_addr (mbar_base + 32)
    #define snapshot_done_0_addr (mbar_base + 40)
    #define state_read_done_0_addr (mbar_base + 48)
    #define dpair_ready_0_addr (mbar_base + 56)
    #define state_inp_ready_0_addr (mbar_base + 64)
    #define u_ready_0_addr (mbar_base + 72)
    #define u_inp_ready_0_addr (mbar_base + 80)
    #define u2_acc_ready_0_addr (mbar_base + 88)
    #define u2_inp_ready_0_addr (mbar_base + 96)
    #define final_ready_0_addr (mbar_base + 104)
    #define out_empty_0_addr (mbar_base + 112)
    #define m_full_1_addr (mbar_base + 120)
    #define m_free_1_addr (mbar_base + 128)
    #define ops_full_1_addr (mbar_base + 136)
    #define ops_free_1_addr (mbar_base + 144)
    #define state_ready_1_addr (mbar_base + 152)
    #define snapshot_done_1_addr (mbar_base + 160)
    #define state_read_done_1_addr (mbar_base + 168)
    #define dpair_ready_1_addr (mbar_base + 176)
    #define state_inp_ready_1_addr (mbar_base + 184)
    #define u_ready_1_addr (mbar_base + 192)
    #define u_inp_ready_1_addr (mbar_base + 200)
    #define u2_acc_ready_1_addr (mbar_base + 208)
    #define u2_inp_ready_1_addr (mbar_base + 216)
    #define final_ready_1_addr (mbar_base + 224)
    #define out_empty_1_addr (mbar_base + 232)
    #define carry_full_addr (mbar_base + 240)
    #define tmem_dealloc_ready_addr (mbar_base + 248)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(carry_hi_tma)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(carry_lo_tma)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(maps_tma)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(map_final_tma)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(out_tma)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(rows_tma)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(final_tma)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(pair_tma)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    __nv_bfloat16* smem_carry_hi = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_carry_hi_addr = smem + 1024;
    __nv_bfloat16* smem_carry_lo = reinterpret_cast<__nv_bfloat16*>(smem_raw + 33792);
    const int smem_carry_lo_addr = smem + 33792;
    __nv_bfloat16* smem_qd_0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 66560);
    const int smem_qd_0_addr = smem + 66560;
    __nv_bfloat16* smem_kd_0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 74752);
    const int smem_kd_0_addr = smem + 74752;
    __nv_bfloat16* smem_ft_0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 82944);
    const int smem_ft_0_addr = smem + 82944;
    __nv_bfloat16* smem_mqk_0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 91136);
    const int smem_mqk_0_addr = smem + 91136;
    __nv_bfloat16* smem_inv_0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 95232);
    const int smem_inv_0_addr = smem + 95232;
    float* smem_vec_0 = reinterpret_cast<float*>(smem_raw + 97280);
    const int smem_vec_0_addr = smem + 97280;
    float* smem_vec_all_0 = reinterpret_cast<float*>(smem_raw + 97280);
    const int smem_vec_all_0_addr = smem + 97280;
    __nv_bfloat16* smem_map_0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 99328);
    const int smem_map_0_addr = smem + 99328;
    __nv_bfloat16* smem_out_0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 132096);
    const int smem_out_0_addr = smem + 132096;
    float* smem_panel_0 = reinterpret_cast<float*>(smem_raw + 132096);
    const int smem_panel_0_addr = smem + 132096;
    __nv_bfloat16* smem_panel_bf16_0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 132096);
    const int smem_panel_bf16_0_addr = smem + 132096;
    float* smem_dpair_0 = reinterpret_cast<float*>(smem_raw + 148480);
    const int smem_dpair_0_addr = smem + 148480;
    __nv_bfloat16* smem_qd_1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 149504);
    const int smem_qd_1_addr = smem + 149504;
    __nv_bfloat16* smem_kd_1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 157696);
    const int smem_kd_1_addr = smem + 157696;
    __nv_bfloat16* smem_ft_1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 165888);
    const int smem_ft_1_addr = smem + 165888;
    __nv_bfloat16* smem_mqk_1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 174080);
    const int smem_mqk_1_addr = smem + 174080;
    __nv_bfloat16* smem_inv_1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 178176);
    const int smem_inv_1_addr = smem + 178176;
    float* smem_vec_1 = reinterpret_cast<float*>(smem_raw + 180224);
    const int smem_vec_1_addr = smem + 180224;
    float* smem_vec_all_1 = reinterpret_cast<float*>(smem_raw + 180224);
    const int smem_vec_all_1_addr = smem + 180224;
    __nv_bfloat16* smem_map_1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 182272);
    const int smem_map_1_addr = smem + 182272;
    __nv_bfloat16* smem_out_1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 215040);
    const int smem_out_1_addr = smem + 215040;
    float* smem_panel_1 = reinterpret_cast<float*>(smem_raw + 215040);
    const int smem_panel_1_addr = smem + 215040;
    __nv_bfloat16* smem_panel_bf16_1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 215040);
    const int smem_panel_bf16_1_addr = smem + 215040;
    float* smem_dpair_1 = reinterpret_cast<float*>(smem_raw + 231424);
    const int smem_dpair_1_addr = smem + 231424;

    // Mbarrier init (32 pipeline groups, 0 ordered-sequence groups, 32 barriers)
    // Mbarriers at smem_raw[0..256)

    if (warp == 0) {
        // --- pipeline 'm_pipe_0' ---
        // m_full_0: 1 barriers, init_count=1
        // m_free_0: 1 barriers, init_count=1
        // --- pipeline 'ops_pipe_0' ---
        // ops_full_0: 1 barriers, init_count=1
        // ops_free_0: 1 barriers, init_count=1
        // --- pipeline 'm_pipe_0' ---
        // state_ready_0: 1 barriers, init_count=1
        // snapshot_done_0: 1 barriers, init_count=4
        // state_read_done_0: 1 barriers, init_count=4
        // dpair_ready_0: 1 barriers, init_count=4
        // --- pipeline 'ops_pipe_0' ---
        // state_inp_ready_0: 1 barriers, init_count=4
        // u_ready_0: 1 barriers, init_count=1
        // u_inp_ready_0: 1 barriers, init_count=4
        // u2_acc_ready_0: 1 barriers, init_count=1
        // u2_inp_ready_0: 1 barriers, init_count=4
        // final_ready_0: 1 barriers, init_count=1
        // out_empty_0: 1 barriers, init_count=1
        // --- pipeline 'm_pipe_1' ---
        // m_full_1: 1 barriers, init_count=1
        // m_free_1: 1 barriers, init_count=1
        // --- pipeline 'ops_pipe_1' ---
        // ops_full_1: 1 barriers, init_count=1
        // ops_free_1: 1 barriers, init_count=1
        // --- pipeline 'm_pipe_1' ---
        // state_ready_1: 1 barriers, init_count=1
        // snapshot_done_1: 1 barriers, init_count=4
        // state_read_done_1: 1 barriers, init_count=4
        // dpair_ready_1: 1 barriers, init_count=4
        // --- pipeline 'ops_pipe_1' ---
        // state_inp_ready_1: 1 barriers, init_count=4
        // u_ready_1: 1 barriers, init_count=1
        // u_inp_ready_1: 1 barriers, init_count=4
        // u2_acc_ready_1: 1 barriers, init_count=1
        // u2_inp_ready_1: 1 barriers, init_count=4
        // final_ready_1: 1 barriers, init_count=1
        // out_empty_1: 1 barriers, init_count=1
        // carry_full: 1 barriers, init_count=1
        // tmem_dealloc_ready: 1 barriers, init_count=4
        // Warp-cooperative initialization in physical record order.
        uint32_t _mbarrier_init_count_0_0 = 4;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(31), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(28), "r"((uint32_t)(4)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(27), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(26), "r"((uint32_t)(4)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(25), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(24), "r"((uint32_t)(4)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(20), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(13), "r"((uint32_t)(4)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(12), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(11), "r"((uint32_t)(4)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(10), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(9), "r"((uint32_t)(4)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(5), "r"((uint32_t)(1)));
        mbarrier_init(smem + 0 + lane * 8, _mbarrier_init_count_0_0);
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    __syncwarp();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 256);
    if (warp == 0) {
        int _tmem_hold = smem + 256;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_state_0 = taddr + 64;
    const int tmem_tmem_state_inp_0 = taddr;
    const int tmem_tmem_u_acc_0 = taddr + 224;
    const int tmem_tmem_u2_inp_0 = taddr + 224;
    const int tmem_tmem_u2_acc_0 = taddr;
    const int tmem_tmem_out_0 = taddr + 192;
    const int tmem_tmem_state_out_0 = taddr + 64;
    const int tmem_tmem_state_1 = taddr + 320;
    const int tmem_tmem_state_inp_1 = taddr + 256;
    const int tmem_tmem_u_acc_1 = taddr + 480;
    const int tmem_tmem_u2_inp_1 = taddr + 480;
    const int tmem_tmem_u2_acc_1 = taddr + 256;
    const int tmem_tmem_out_1 = taddr + 448;
    const int tmem_tmem_state_out_1 = taddr + 320;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 16 && warp <= 19) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
    }

    // ---- Role: compute_0 ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 144;");
        { // compute_0_main
            int item_base = blockIdx.x * 12;
            int carry_slab = items[item_base];
            int snap_outer = items[item_base + 1];
            int ops_slab = items[item_base + 2];
            int p_begin = items[item_base + 3];
            int p_end = items[item_base + 4];
            int window_chunks = items[item_base + 5];
            int token_base = items[item_base + 6];
            int part_idx = items[item_base + 7];
            int final_outer = items[item_base + 8];
            int head_idx = items[item_base + 9];
            int window_tokens = items[item_base + 10];
            int rows_enabled = items[item_base + 11];
            int num_blocks = p_end - p_begin;
            int has_final = ((final_outer >= 0) ? 1 : 0);
            int num_steps = num_blocks + has_final;
            int warp_in_wg = warp % 4;
            const int tmem_row_base = warp_in_wg * 32 << 16;
            int state_row = warp_in_wg * 32 + lane;
            int pairmap_mode = 0;
            int state_base = taddr + 64 + (unsigned int)tmem_row_base;
            int inp_base = taddr + (unsigned int)tmem_row_base;
            int u_base = taddr + 224 + (unsigned int)tmem_row_base;
            int u2_acc_base = taddr + (unsigned int)tmem_row_base;
            int u2_inp_base = taddr + 224 + (unsigned int)tmem_row_base;
            unsigned int compute_m_stage = 0;
            unsigned int compute_ops_stage = 0;
            unsigned int _phase_state_read_done_0_0 = 1;
            unsigned int _phase_state_ready_0 = 0;
            unsigned int _phase_ops_full_0 = 0;
            unsigned int _phase_u_ready_0 = 0;
            unsigned int _phase_u2_acc_ready_0 = 0;
            unsigned int _phase_snapshot_done_0 = 0;
            unsigned int _phase_final_ready_0 = 0;
            #pragma unroll 1
            for (int step = 0; step < num_steps; step += 2) {
                int remaining = window_chunks - (p_begin + step) * 2;
                int count = 2;
                if (remaining < 2) {
                    count = remaining;
                }
                if (num_blocks <= step) {
                    count = 0;
                }
                int step_chunks = count;
                float dpair_acc = 1.0f;
                if (pairmap_mode != 0) {
                    mbarrier_wait(state_read_done_0_addr, _phase_state_read_done_0_0);
                    _phase_state_read_done_0_0 ^= 1;
                    #pragma unroll
                    for (int eye_col_block = 0; eye_col_block < 4; eye_col_block++) {
                        float eye[32];
                        int eye_diag = state_row - eye_col_block * 32;
                        #pragma unroll
                        for (int eye_col = 0; eye_col < 32; eye_col++) {
                            eye[eye_col] = ((eye_diag == eye_col) ? 1.0f : 0.0f);
                        }
                        tmem_st_x32_f32(state_base + eye_col_block * 32, eye);
                    }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    if (elect_sync()) {
                        mbarrier_arrive(state_ready_0_addr + (compute_m_stage) * 8);
                    }
                }
                if (step_chunks > 0) {
                    mbarrier_wait(state_ready_0_addr + (compute_m_stage) * 8, _phase_state_ready_0);
                    #pragma unroll
                    for (int state_col_block = 0; state_col_block < 4; state_col_block++) {
                        float _tmem_load_0[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                            : "r"(state_base + state_col_block * 32));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        uint32_t _tmem_load_0_bf16[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_0[_lp*2 + 0], _tmem_load_0[_lp*2+1 + 0]));
                            _tmem_load_0_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x16.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"(inp_base + state_col_block * 16), "r"(_tmem_load_0_bf16[0]), "r"(_tmem_load_0_bf16[1]), "r"(_tmem_load_0_bf16[2]), "r"(_tmem_load_0_bf16[3]), "r"(_tmem_load_0_bf16[4]), "r"(_tmem_load_0_bf16[5]), "r"(_tmem_load_0_bf16[6]), "r"(_tmem_load_0_bf16[7]), "r"(_tmem_load_0_bf16[8]), "r"(_tmem_load_0_bf16[9]), "r"(_tmem_load_0_bf16[10]), "r"(_tmem_load_0_bf16[11]), "r"(_tmem_load_0_bf16[12]), "r"(_tmem_load_0_bf16[13]), "r"(_tmem_load_0_bf16[14]), "r"(_tmem_load_0_bf16[15]));
                    }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    if (elect_sync()) {
                        mbarrier_arrive(state_inp_ready_0_addr + (compute_ops_stage) * 8);
                    }
                }
                #pragma unroll 1
                for (int chunk = 0; chunk < step_chunks; chunk++) {
                    mbarrier_wait(ops_full_0_addr + (compute_ops_stage) * 8, _phase_ops_full_0);
                    int vec_base = (int)compute_ops_stage * 8192;
                    mbarrier_wait(u_ready_0_addr + (compute_ops_stage) * 8, _phase_u_ready_0);
                    float prediction[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(prediction[0]), "=f"(prediction[1]), "=f"(prediction[2]), "=f"(prediction[3]), "=f"(prediction[4]), "=f"(prediction[5]), "=f"(prediction[6]), "=f"(prediction[7]), "=f"(prediction[8]), "=f"(prediction[9]), "=f"(prediction[10]), "=f"(prediction[11]), "=f"(prediction[12]), "=f"(prediction[13]), "=f"(prediction[14]), "=f"(prediction[15]), "=f"(prediction[16]), "=f"(prediction[17]), "=f"(prediction[18]), "=f"(prediction[19]), "=f"(prediction[20]), "=f"(prediction[21]), "=f"(prediction[22]), "=f"(prediction[23]), "=f"(prediction[24]), "=f"(prediction[25]), "=f"(prediction[26]), "=f"(prediction[27]), "=f"(prediction[28]), "=f"(prediction[29]), "=f"(prediction[30]), "=f"(prediction[31])
                        : "r"(u_base));
                    #pragma unroll
                    for (int residual_half = 0; residual_half < 2; residual_half++) {
                        float residual[16];
                        #pragma unroll
                        for (int residual_col = 0; residual_col < 16; residual_col++) {
                            float neg_beta = -smem_vec_all_0[vec_base + 257 + residual_half * 16 + residual_col];
                            residual[residual_col] = prediction[residual_half * 16 + residual_col] * neg_beta;
                        }
                        uint32_t residual_bf16[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 8; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual[_lp*2 + 0], residual[_lp*2+1 + 0]));
                            residual_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        tmem_st_x8_u32(u_base + residual_half * 8, (const uint32_t*)residual_bf16);
                    }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    if (elect_sync()) {
                        mbarrier_arrive(u_inp_ready_0_addr + (compute_ops_stage) * 8);
                    }
                    mbarrier_wait(u2_acc_ready_0_addr + (compute_ops_stage) * 8, _phase_u2_acc_ready_0);
                    float _tmem_load_1[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31])
                        : "r"(u2_acc_base));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    unsigned int u2_packed[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_1[_lp*2 + 0], _tmem_load_1[_lp*2+1 + 0]));
                        u2_packed[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(u2_inp_base), "r"(u2_packed[0]), "r"(u2_packed[1]), "r"(u2_packed[2]), "r"(u2_packed[3]), "r"(u2_packed[4]), "r"(u2_packed[5]), "r"(u2_packed[6]), "r"(u2_packed[7]), "r"(u2_packed[8]), "r"(u2_packed[9]), "r"(u2_packed[10]), "r"(u2_packed[11]), "r"(u2_packed[12]), "r"(u2_packed[13]), "r"(u2_packed[14]), "r"(u2_packed[15]));
                    int do_decay = ((step_chunks > chunk + 1) ? 1 : 0);
                    if (do_decay != 0) {
                        if (pairmap_mode == 0) {
                            mbarrier_wait(snapshot_done_0_addr + (compute_m_stage) * 8, _phase_snapshot_done_0);
                        } else {
                            float d_row = smem_vec_all_0[vec_base + state_row];
                            dpair_acc = dpair_acc * d_row;
                            int publish_dpair = ((step_chunks <= chunk + 1) ? 1 : 0);
                            if (publish_dpair != 0) {
                                smem_dpair_0[state_row] = dpair_acc;
                                if (elect_sync()) {
                                    mbarrier_arrive(dpair_ready_0_addr);
                                }
                            }
                        }
                        #pragma unroll
                        for (int state_col_block_1 = 0; state_col_block_1 < 4; state_col_block_1++) {
                            int state_addr = state_base + state_col_block_1 * 32;
                            float state_frag[32];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(state_frag[0]), "=f"(state_frag[1]), "=f"(state_frag[2]), "=f"(state_frag[3]), "=f"(state_frag[4]), "=f"(state_frag[5]), "=f"(state_frag[6]), "=f"(state_frag[7]), "=f"(state_frag[8]), "=f"(state_frag[9]), "=f"(state_frag[10]), "=f"(state_frag[11]), "=f"(state_frag[12]), "=f"(state_frag[13]), "=f"(state_frag[14]), "=f"(state_frag[15]), "=f"(state_frag[16]), "=f"(state_frag[17]), "=f"(state_frag[18]), "=f"(state_frag[19]), "=f"(state_frag[20]), "=f"(state_frag[21]), "=f"(state_frag[22]), "=f"(state_frag[23]), "=f"(state_frag[24]), "=f"(state_frag[25]), "=f"(state_frag[26]), "=f"(state_frag[27]), "=f"(state_frag[28]), "=f"(state_frag[29]), "=f"(state_frag[30]), "=f"(state_frag[31])
                                : "r"(state_addr));
                            float state_scale[16];
                            #pragma unroll
                            for (int state_half = 0; state_half < 2; state_half++) {
                                #pragma unroll
                                for (int state_col = 0; state_col < 16; state_col++) {
                                    state_scale[state_col] = smem_vec_all_0[vec_base + state_col_block_1 * 32 + state_half * 16 + state_col];
                                }
                                #pragma unroll
                                for (int _ls = 0; _ls < 8; _ls++)
                                    mul_f32x2_inplace(&reinterpret_cast<float2*>((state_frag + state_half * 16))[_ls], reinterpret_cast<const float2*>(state_scale)[_ls]);
                            }
                            tmem_st_x32_f32(state_addr, state_frag);
                        }
                    }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    if (elect_sync()) {
                        mbarrier_arrive(u2_inp_ready_0_addr + (compute_ops_stage) * 8);
                    }
                    if (step_chunks > chunk + 1) {
                        mbarrier_wait(final_ready_0_addr + (compute_ops_stage) * 8, _phase_final_ready_0);
                        #pragma unroll
                        for (int state_col_block_2 = 0; state_col_block_2 < 4; state_col_block_2++) {
                            float _tmem_load_2[32];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(_tmem_load_2[0]), "=f"(_tmem_load_2[1]), "=f"(_tmem_load_2[2]), "=f"(_tmem_load_2[3]), "=f"(_tmem_load_2[4]), "=f"(_tmem_load_2[5]), "=f"(_tmem_load_2[6]), "=f"(_tmem_load_2[7]), "=f"(_tmem_load_2[8]), "=f"(_tmem_load_2[9]), "=f"(_tmem_load_2[10]), "=f"(_tmem_load_2[11]), "=f"(_tmem_load_2[12]), "=f"(_tmem_load_2[13]), "=f"(_tmem_load_2[14]), "=f"(_tmem_load_2[15]), "=f"(_tmem_load_2[16]), "=f"(_tmem_load_2[17]), "=f"(_tmem_load_2[18]), "=f"(_tmem_load_2[19]), "=f"(_tmem_load_2[20]), "=f"(_tmem_load_2[21]), "=f"(_tmem_load_2[22]), "=f"(_tmem_load_2[23]), "=f"(_tmem_load_2[24]), "=f"(_tmem_load_2[25]), "=f"(_tmem_load_2[26]), "=f"(_tmem_load_2[27]), "=f"(_tmem_load_2[28]), "=f"(_tmem_load_2[29]), "=f"(_tmem_load_2[30]), "=f"(_tmem_load_2[31])
                                : "r"(state_base + state_col_block_2 * 32));
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            uint32_t _tmem_load_2_bf16[16];
                            #pragma unroll
                            for (int _lp = 0; _lp < 16; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 0], _tmem_load_2[_lp*2+1 + 0]));
                                _tmem_load_2_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x16.b32"
                                " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                                :: "r"(inp_base + state_col_block_2 * 16), "r"(_tmem_load_2_bf16[0]), "r"(_tmem_load_2_bf16[1]), "r"(_tmem_load_2_bf16[2]), "r"(_tmem_load_2_bf16[3]), "r"(_tmem_load_2_bf16[4]), "r"(_tmem_load_2_bf16[5]), "r"(_tmem_load_2_bf16[6]), "r"(_tmem_load_2_bf16[7]), "r"(_tmem_load_2_bf16[8]), "r"(_tmem_load_2_bf16[9]), "r"(_tmem_load_2_bf16[10]), "r"(_tmem_load_2_bf16[11]), "r"(_tmem_load_2_bf16[12]), "r"(_tmem_load_2_bf16[13]), "r"(_tmem_load_2_bf16[14]), "r"(_tmem_load_2_bf16[15]));
                        }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        _phase_ops_full_0 ^= 1;
                        _phase_u_ready_0 ^= 1;
                        _phase_u2_acc_ready_0 ^= 1;
                        _phase_final_ready_0 ^= 1;
                        if (elect_sync()) {
                            mbarrier_arrive(state_inp_ready_0_addr + (compute_ops_stage) * 8);
                        }
                    } else {
                        _phase_ops_full_0 ^= 1;
                        _phase_u_ready_0 ^= 1;
                        _phase_u2_acc_ready_0 ^= 1;
                        _phase_final_ready_0 ^= 1;
                    }
                }
                _phase_state_ready_0 ^= 1;
                _phase_snapshot_done_0 ^= 1;
            }
            if (warp_in_wg == 0) {
                if (elect_sync()) {
                    mbarrier_arrive(tmem_dealloc_ready_addr);
                }
            }
        }
    }
    // ---- Role: epilogue_0 ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 72;");
        { // epilogue_0_main
            int item_base_1 = blockIdx.x * 12;
            int carry_slab_1 = items[item_base_1];
            int snap_outer_1 = items[item_base_1 + 1];
            int ops_slab_1 = items[item_base_1 + 2];
            int p_begin_1 = items[item_base_1 + 3];
            int p_end_1 = items[item_base_1 + 4];
            int window_chunks_1 = items[item_base_1 + 5];
            int token_base_1 = items[item_base_1 + 6];
            int part_idx_1 = items[item_base_1 + 7];
            int final_outer_1 = items[item_base_1 + 8];
            int head_idx_1 = items[item_base_1 + 9];
            int window_tokens_1 = items[item_base_1 + 10];
            int rows_enabled_1 = items[item_base_1 + 11];
            int num_blocks_1 = p_end_1 - p_begin_1;
            int has_final_1 = ((final_outer_1 >= 0) ? 1 : 0);
            int num_steps_1 = num_blocks_1 + has_final_1;
            int warp_in_wg_1 = warp % 4;
            const int tmem_row_base_1 = warp_in_wg_1 * 32 << 16;
            int state_row_1 = warp_in_wg_1 * 32 + lane;
            int epilogue_local_warp = warp_in_wg_1;
            int state_base_1 = taddr + 64 + (unsigned int)tmem_row_base_1;
            int out_base = taddr + 192 + (unsigned int)tmem_row_base_1;
            unsigned int epilogue_m_stage = 0;
            unsigned int epilogue_ops_stage = 0;
            int apply_mode = 1;
            unsigned int _phase_state_ready_0_1 = 0;
            unsigned int _phase_final_ready_0_1 = 0;
            unsigned int _phase_dpair_ready_0_0 = 0;
            #pragma unroll 1
            for (int step_1 = 0; step_1 < num_steps_1; step_1 += 2) {
                int block = p_begin_1 + step_1;
                int remaining_1 = window_chunks_1 - (p_begin_1 + step_1) * 2;
                int count_1 = 2;
                if (remaining_1 < 2) {
                    count_1 = remaining_1;
                }
                if (num_blocks_1 <= step_1) {
                    count_1 = 0;
                }
                int step_chunks_1 = count_1;
                int is_final = ((num_blocks_1 <= step_1) ? 1 : 0);
                int writes_row = ((rows_enabled_1 != 0 && block > 0 && is_final == 0) ? 1 : 0);
                mbarrier_wait(state_ready_0_addr + (epilogue_m_stage) * 8, _phase_state_ready_0_1);
                if (writes_row != 0 || is_final != 0) {
                    int panel_outer = final_outer_1;
                    if (is_final == 0) {
                        long long row_start = row_starts[part_idx_1];
                        panel_outer = (int)((row_start + (long long)block) * (long long)num_heads + (long long)head_idx_1);
                    }
                    #pragma unroll
                    for (int panel = 0; panel < 4; panel++) {
                        float _tmem_load_3[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_3[0]), "=f"(_tmem_load_3[1]), "=f"(_tmem_load_3[2]), "=f"(_tmem_load_3[3]), "=f"(_tmem_load_3[4]), "=f"(_tmem_load_3[5]), "=f"(_tmem_load_3[6]), "=f"(_tmem_load_3[7]), "=f"(_tmem_load_3[8]), "=f"(_tmem_load_3[9]), "=f"(_tmem_load_3[10]), "=f"(_tmem_load_3[11]), "=f"(_tmem_load_3[12]), "=f"(_tmem_load_3[13]), "=f"(_tmem_load_3[14]), "=f"(_tmem_load_3[15]), "=f"(_tmem_load_3[16]), "=f"(_tmem_load_3[17]), "=f"(_tmem_load_3[18]), "=f"(_tmem_load_3[19]), "=f"(_tmem_load_3[20]), "=f"(_tmem_load_3[21]), "=f"(_tmem_load_3[22]), "=f"(_tmem_load_3[23]), "=f"(_tmem_load_3[24]), "=f"(_tmem_load_3[25]), "=f"(_tmem_load_3[26]), "=f"(_tmem_load_3[27]), "=f"(_tmem_load_3[28]), "=f"(_tmem_load_3[29]), "=f"(_tmem_load_3[30]), "=f"(_tmem_load_3[31])
                            : "r"(state_base_1 + panel * 32));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        if (epilogue_local_warp == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                        }
                        asm volatile("barrier.sync 8, 128;" ::: "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_panel_0_addr + (unsigned int)(state_row_1 * 128 ^ (state_row_1 * 128 >> 7 & 7) << 4))), "r"(__as_u32(_tmem_load_3[0])), "r"(__as_u32(_tmem_load_3[1])), "r"(__as_u32(_tmem_load_3[2])), "r"(__as_u32(_tmem_load_3[3])) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_panel_0_addr + (unsigned int)(state_row_1 * 128 + 16 ^ (state_row_1 * 128 + 16 >> 7 & 7) << 4))), "r"(__as_u32(_tmem_load_3[4])), "r"(__as_u32(_tmem_load_3[5])), "r"(__as_u32(_tmem_load_3[6])), "r"(__as_u32(_tmem_load_3[7])) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_panel_0_addr + (unsigned int)(state_row_1 * 128 + 32 ^ (state_row_1 * 128 + 32 >> 7 & 7) << 4))), "r"(__as_u32(_tmem_load_3[8])), "r"(__as_u32(_tmem_load_3[9])), "r"(__as_u32(_tmem_load_3[10])), "r"(__as_u32(_tmem_load_3[11])) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_panel_0_addr + (unsigned int)(state_row_1 * 128 + 48 ^ (state_row_1 * 128 + 48 >> 7 & 7) << 4))), "r"(__as_u32(_tmem_load_3[12])), "r"(__as_u32(_tmem_load_3[13])), "r"(__as_u32(_tmem_load_3[14])), "r"(__as_u32(_tmem_load_3[15])) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_panel_0_addr + (unsigned int)(state_row_1 * 128 + 64 ^ (state_row_1 * 128 + 64 >> 7 & 7) << 4))), "r"(__as_u32(_tmem_load_3[16])), "r"(__as_u32(_tmem_load_3[17])), "r"(__as_u32(_tmem_load_3[18])), "r"(__as_u32(_tmem_load_3[19])) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_panel_0_addr + (unsigned int)(state_row_1 * 128 + 80 ^ (state_row_1 * 128 + 80 >> 7 & 7) << 4))), "r"(__as_u32(_tmem_load_3[20])), "r"(__as_u32(_tmem_load_3[21])), "r"(__as_u32(_tmem_load_3[22])), "r"(__as_u32(_tmem_load_3[23])) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_panel_0_addr + (unsigned int)(state_row_1 * 128 + 96 ^ (state_row_1 * 128 + 96 >> 7 & 7) << 4))), "r"(__as_u32(_tmem_load_3[24])), "r"(__as_u32(_tmem_load_3[25])), "r"(__as_u32(_tmem_load_3[26])), "r"(__as_u32(_tmem_load_3[27])) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_panel_0_addr + (unsigned int)(state_row_1 * 128 + 112 ^ (state_row_1 * 128 + 112 >> 7 & 7) << 4))), "r"(__as_u32(_tmem_load_3[28])), "r"(__as_u32(_tmem_load_3[29])), "r"(__as_u32(_tmem_load_3[30])), "r"(__as_u32(_tmem_load_3[31])) : "memory");
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 8, 128;" ::: "memory");
                        if (epilogue_local_warp == 0) {
                            if (elect_sync()) {
                                if (is_final != 0) {
                                    tma_store_3d(final_tma, panel * 32, 0, panel_outer, smem_panel_0_addr);
                                } else {
                                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 900)
                                    #error "TmaReduceAdd3d requires SM90 or newer"
                                    #endif
                                    asm volatile(
                                        "cp.reduce.async.bulk.tensor.3d.global.shared::cta.add.tile.bulk_group"
                                        " [%0, {%1, %2, %3}], [%4];"
                                        :: "l"(rows_tma), "r"(panel * 32), "r"(0), "r"(panel_outer), "r"(smem_panel_0_addr) : "memory");
                                }
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    }
                    if (epilogue_local_warp == 0) {
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                    }
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                }
                if (apply_mode != 0) {
                    if (elect_sync()) {
                        mbarrier_arrive(snapshot_done_0_addr + (epilogue_m_stage) * 8);
                        mbarrier_arrive(state_read_done_0_addr);
                    }
                }
                #pragma unroll 1
                for (int chunk_1 = 0; chunk_1 < step_chunks_1; chunk_1++) {
                    int chunk_idx = block * 2 + chunk_1;
                    int chunk_is_full = apply_mode;
                    int valid_tokens = window_tokens_1 - chunk_idx * 32;
                    if (valid_tokens > 32) {
                        valid_tokens = 32;
                    }
                    mbarrier_wait(final_ready_0_addr + (epilogue_ops_stage) * 8, _phase_final_ready_0_1);
                    if (chunk_is_full != 0) {
                        float _tmem_load_4[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_4[15]))
                            : "r"(out_base));
                        float _tmem_load_5[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[15]))
                            : "r"(out_base + 1048576));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        asm volatile("barrier.sync 8, 128;" ::: "memory");
                        if (epilogue_local_warp == 0) {
                            if (elect_sync()) {
                                mbarrier_arrive(out_empty_0_addr);
                            }
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                        }
                        asm volatile("barrier.sync 8, 128;" ::: "memory");
                        int out_stage_addr = smem_out_0_addr;
                        #pragma unroll
                        for (int dim_half = 0; dim_half < 2; dim_half++) {
                            unsigned int out_packed[8];
                            if (dim_half == 0) {
                                #pragma unroll
                                for (int _lp = 0; _lp < 8; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_4[_lp*2 + 0], _tmem_load_4[_lp*2+1 + 0]));
                                    out_packed[_lp] = *(uint32_t*)&_bf2;
                                }
                            } else {
                                #pragma unroll
                                for (int _lp = 0; _lp < 8; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_5[_lp*2 + 0], _tmem_load_5[_lp*2+1 + 0]));
                                    out_packed[_lp] = *(uint32_t*)&_bf2;
                                }
                            }
                            #pragma unroll
                            for (int token_group = 0; token_group < 2; token_group++) {
                                int mtx_idx = lane / 8;
                                int row_addr = lane & 7;
                                int dim_base = epilogue_local_warp * 32 + dim_half * 16 + (mtx_idx & 1) * 8;
                                int token_base_mtx = token_group * 16 + mtx_idx / 2 * 8;
                                int token_addr = token_base_mtx + row_addr;
                                int token_pair = token_addr / 2;
                                int token_parity = token_addr & 1;
                                int raw_row = token_pair + dim_base / 64 * 16;
                                int raw_col = (dim_base & 63 ^ (token_pair & 3) << 4 ^ token_parity << 3) + token_parity * 64;
                                int stsm_offset = (raw_row * 128 + raw_col) * 2;
                                const int pack_base = token_group * 4;
                                uint32_t _stmatrix_addr_0 = static_cast<uint32_t>((unsigned long long)(out_stage_addr + stsm_offset));
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_0), "r"(*reinterpret_cast<const uint32_t*>(&out_packed[pack_base])), "r"(*reinterpret_cast<const uint32_t*>(&out_packed[pack_base + 1])), "r"(*reinterpret_cast<const uint32_t*>(&out_packed[pack_base + 2])), "r"(*reinterpret_cast<const uint32_t*>(&out_packed[pack_base + 3]))
                                    : "memory");
                            }
                        }
                        int zero_rows = 32 - valid_tokens;
                        if (zero_rows > 0) {
                            unsigned int zero_word = 0;
                            int zero_chunks = zero_rows * 16;
                            #pragma unroll
                            for (int zero_k = 0; zero_k < 4; zero_k++) {
                                int zero_idx = state_row_1 + zero_k * 128;
                                if (zero_idx < zero_chunks) {
                                    int zero_half = zero_idx / (zero_rows * 8);
                                    int zero_within = zero_idx - zero_half * (zero_rows * 8);
                                    int zero_offset = zero_half * 4096 + (valid_tokens + zero_within / 8) * 128 + (zero_within & 7) * 16;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_out_0_addr + (unsigned int)zero_offset), "r"(zero_word), "r"(zero_word), "r"(zero_word), "r"(zero_word) : "memory");
                                }
                            }
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 8, 128;" ::: "memory");
                        if (epilogue_local_warp == 0) {
                            if (elect_sync()) {
                                #pragma unroll
                                for (int dim_half_1 = 0; dim_half_1 < 2; dim_half_1++) {
                                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 900)
                                    #error "TmaReduceAdd3d requires SM90 or newer"
                                    #endif
                                    asm volatile(
                                        "cp.reduce.async.bulk.tensor.3d.global.shared::cta.add.tile.bulk_group"
                                        " [%0, {%1, %2, %3}], [%4];"
                                        :: "l"(out_tma), "r"(dim_half_1 * 64), "r"(head_idx_1), "r"(token_base_1 + chunk_idx * 32), "r"(smem_out_0_addr + (unsigned int)(dim_half_1 * 4096)) : "memory");
                                }
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    }
                    _phase_final_ready_0_1 ^= 1;
                }
                if (apply_mode == 0) {
                    mbarrier_wait(dpair_ready_0_addr, _phase_dpair_ready_0_0);
                    _phase_dpair_ready_0_0 ^= 1;
                    float dpair_row = smem_dpair_0[state_row_1];
                    int pair_outer = snap_outer_1 + block * num_heads;
                    #pragma unroll
                    for (int panel_1 = 0; panel_1 < 4; panel_1++) {
                        float pair_frag[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(pair_frag[0]), "=f"(pair_frag[1]), "=f"(pair_frag[2]), "=f"(pair_frag[3]), "=f"(pair_frag[4]), "=f"(pair_frag[5]), "=f"(pair_frag[6]), "=f"(pair_frag[7]), "=f"(pair_frag[8]), "=f"(pair_frag[9]), "=f"(pair_frag[10]), "=f"(pair_frag[11]), "=f"(pair_frag[12]), "=f"(pair_frag[13]), "=f"(pair_frag[14]), "=f"(pair_frag[15]), "=f"(pair_frag[16]), "=f"(pair_frag[17]), "=f"(pair_frag[18]), "=f"(pair_frag[19]), "=f"(pair_frag[20]), "=f"(pair_frag[21]), "=f"(pair_frag[22]), "=f"(pair_frag[23]), "=f"(pair_frag[24]), "=f"(pair_frag[25]), "=f"(pair_frag[26]), "=f"(pair_frag[27]), "=f"(pair_frag[28]), "=f"(pair_frag[29]), "=f"(pair_frag[30]), "=f"(pair_frag[31])
                            : "r"(state_base_1 + panel_1 * 32));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        int diag_local = state_row_1 - panel_1 * 32;
                        #pragma unroll
                        for (int pair_col = 0; pair_col < 32; pair_col++) {
                            pair_frag[pair_col] = pair_frag[pair_col] - ((diag_local == pair_col) ? dpair_row : 0.0f);
                        }
                        if (epilogue_local_warp == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                        }
                        asm volatile("barrier.sync 8, 128;" ::: "memory");
                        uint32_t pair_frag_bf16[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair_frag[_lp*2 + 0], pair_frag[_lp*2+1 + 0]));
                            pair_frag_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_panel_bf16_0_addr + (unsigned int)(state_row_1 * 128 ^ (state_row_1 * 128 >> 7 & 7) << 4))), "r"(pair_frag_bf16[0]), "r"(pair_frag_bf16[1]), "r"(pair_frag_bf16[2]), "r"(pair_frag_bf16[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_panel_bf16_0_addr + (unsigned int)(state_row_1 * 128 + 16 ^ (state_row_1 * 128 + 16 >> 7 & 7) << 4))), "r"(pair_frag_bf16[4]), "r"(pair_frag_bf16[5]), "r"(pair_frag_bf16[6]), "r"(pair_frag_bf16[7]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_panel_bf16_0_addr + (unsigned int)(state_row_1 * 128 + 32 ^ (state_row_1 * 128 + 32 >> 7 & 7) << 4))), "r"(pair_frag_bf16[8]), "r"(pair_frag_bf16[9]), "r"(pair_frag_bf16[10]), "r"(pair_frag_bf16[11]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_panel_bf16_0_addr + (unsigned int)(state_row_1 * 128 + 48 ^ (state_row_1 * 128 + 48 >> 7 & 7) << 4))), "r"(pair_frag_bf16[12]), "r"(pair_frag_bf16[13]), "r"(pair_frag_bf16[14]), "r"(pair_frag_bf16[15]) : "memory");
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 8, 128;" ::: "memory");
                        if (epilogue_local_warp == 0) {
                            if (elect_sync()) {
                                tma_store_3d(pair_tma, panel_1 * 32, 0, pair_outer, smem_panel_bf16_0_addr);
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    }
                    dpair[(long long)(pair_outer * 128 + state_row_1)] = dpair_row;
                    if (epilogue_local_warp == 0) {
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                    }
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                    if (elect_sync()) {
                        mbarrier_arrive(state_read_done_0_addr);
                    }
                }
                _phase_state_ready_0_1 ^= 1;
            }
            if (epilogue_local_warp == 0) {
                asm volatile("cp.async.bulk.wait_group 0;");
                if (elect_sync()) {
                    mbarrier_arrive(tmem_dealloc_ready_addr);
                }
            }
        }
    }
    // ---- Role: mma_0 ----
    if (warp == 16) {
        { // mma_0_main
            int item_base_2 = blockIdx.x * 12;
            int carry_slab_2 = items[item_base_2];
            int snap_outer_2 = items[item_base_2 + 1];
            int ops_slab_2 = items[item_base_2 + 2];
            int p_begin_2 = items[item_base_2 + 3];
            int p_end_2 = items[item_base_2 + 4];
            int window_chunks_2 = items[item_base_2 + 5];
            int token_base_2 = items[item_base_2 + 6];
            int part_idx_2 = items[item_base_2 + 7];
            int final_outer_2 = items[item_base_2 + 8];
            int head_idx_2 = items[item_base_2 + 9];
            int window_tokens_2 = items[item_base_2 + 10];
            int rows_enabled_2 = items[item_base_2 + 11];
            int num_blocks_2 = p_end_2 - p_begin_2;
            int has_final_2 = ((final_outer_2 >= 0) ? 1 : 0);
            int num_steps_2 = num_blocks_2 + has_final_2;
            unsigned int mma_m_stage = 0;
            unsigned int mma_ops_stage = 0;
            int apply_mode_1 = 1;
            unsigned int _phase_carry_full_0 = 0;
            if (apply_mode_1 != 0) {
                mbarrier_wait(carry_full_addr, _phase_carry_full_0);
                _phase_carry_full_0 ^= 1;
            }
            unsigned int _phase_m_full_0 = 0;
            unsigned int _phase_state_read_done_0_0_1 = 1;
            unsigned int _phase_ops_full_0_1 = 0;
            unsigned int _phase_state_inp_ready_0 = 0;
            unsigned int _phase_out_empty_0_0 = 1;
            unsigned int _phase_u_inp_ready_0 = 0;
            unsigned int _phase_u2_inp_ready_0 = 0;
            #pragma unroll 1
            for (int step_2 = 0; step_2 < num_steps_2; step_2 += 2) {
                if (apply_mode_1 != 0) {
                    mbarrier_wait(m_full_0_addr + (mma_m_stage) * 8, _phase_m_full_0);
                    mbarrier_wait(state_read_done_0_addr, _phase_state_read_done_0_0_1);
                    _phase_state_read_done_0_0_1 ^= 1;
                    int _mma_a_lo_0 = make_warp_uniform(((smem_carry_hi_addr) >> 4) & 0x3FFF);
                    int _mma_b_lo_0 = make_warp_uniform(((((smem_map_0_addr) >> 4) & 0x3FFF) | 0x4000000) + (mma_m_stage) * 2048);
                    {
                        uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_0);
                        uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_0);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_state_0, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136381584, 0);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 128U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_state_0, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136381584, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 128U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_state_0, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136381584, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 128U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_state_0, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136381584, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 1018U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 128U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_state_0, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136381584, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 128U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_state_0, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136381584, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 128U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_state_0, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136381584, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 128U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_state_0, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136381584, 1);
                        }
                    }
                    int _mma_a_lo_1 = make_warp_uniform(((smem_carry_lo_addr) >> 4) & 0x3FFF);
                    int _mma_b_lo_1 = make_warp_uniform(((((smem_map_0_addr) >> 4) & 0x3FFF) | 0x4000000) + (mma_m_stage) * 2048);
                    {
                        uint64_t _mma_ss_a_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_1);
                        uint64_t _mma_ss_b_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_1);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_state_0, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 136381584, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 128U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_state_0, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 136381584, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 128U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_state_0, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 136381584, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 128U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_state_0, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 136381584, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 1018U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 128U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_state_0, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 136381584, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 128U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_state_0, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 136381584, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 128U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_state_0, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 136381584, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 128U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_state_0, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 136381584, 1);
                        }
                    }
                    elect_commit2(state_ready_0_addr + (mma_m_stage) * 8, m_free_0_addr + (mma_m_stage) * 8);
                }
                int remaining_2 = window_chunks_2 - (p_begin_2 + step_2) * 2;
                int count_2 = 2;
                if (remaining_2 < 2) {
                    count_2 = remaining_2;
                }
                if (num_blocks_2 <= step_2) {
                    count_2 = 0;
                }
                int step_chunks_2 = count_2;
                #pragma unroll 1
                for (int chunk_2 = 0; chunk_2 < step_chunks_2; chunk_2++) {
                    mbarrier_wait(ops_full_0_addr + (mma_ops_stage) * 8, _phase_ops_full_0_1);
                    mbarrier_wait(state_inp_ready_0_addr + (mma_ops_stage) * 8, _phase_state_inp_ready_0);
                    if (apply_mode_1 != 0) {
                        mbarrier_wait(out_empty_0_addr, _phase_out_empty_0_0);
                        _phase_out_empty_0_0 ^= 1;
                        int _mma_b_lo_2 = make_warp_uniform((((smem_qd_0_addr) >> 4) & 0x3FFF) + (mma_ops_stage) * 2048);
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
                    "mov.b32 id, 134743184;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 250;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_out_0), "r"(_mma_b_lo_2), "r"(tmem_tmem_state_inp_0), "r"(0));
                    }
                    int _mma_b_lo_3 = make_warp_uniform((((smem_kd_0_addr) >> 4) & 0x3FFF) + (mma_ops_stage) * 2048);
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
                    "mov.b32 id, 134743184;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 250;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_u_acc_0), "r"(_mma_b_lo_3), "r"(tmem_tmem_state_inp_0), "r"(0));
                    elect_commit(u_ready_0_addr + (mma_ops_stage) * 8);
                    mbarrier_wait(u_inp_ready_0_addr + (mma_ops_stage) * 8, _phase_u_inp_ready_0);
                    int _mma_b_lo_4 = make_warp_uniform((((smem_inv_0_addr) >> 4) & 0x3FFF) + (mma_ops_stage) * 2048);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0xC0004010;\n\t"
                    "mov.b32 id, 134743184;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_u2_acc_0), "r"(_mma_b_lo_4), "r"(tmem_tmem_u2_inp_0), "r"(0));
                    elect_commit(u2_acc_ready_0_addr + (mma_ops_stage) * 8);
                    mbarrier_wait(u2_inp_ready_0_addr + (mma_ops_stage) * 8, _phase_u2_inp_ready_0);
                    int want_state = ((step_chunks_2 > chunk_2 + 1) ? 1 : 0);
                    if (want_state != 0) {
                        int _mma_b_lo_5 = make_warp_uniform(((((smem_ft_0_addr) >> 4) & 0x3FFF) | 0x1000000) + (mma_ops_stage) * 2048);
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
                    "mov.b32 id, 136905872;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_state_out_0), "r"(_mma_b_lo_5), "r"(tmem_tmem_u2_inp_0), "r"(1));
                    } else {
                        int _mma_b_lo_6 = make_warp_uniform(((((smem_mqk_0_addr) >> 4) & 0x3FFF) | 0x1000000) + (mma_ops_stage) * 2048);
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
                    "mov.b32 id, 134808720;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_out_0), "r"(_mma_b_lo_6), "r"(tmem_tmem_u2_inp_0), "r"(1));
                    }
                    elect_commit2(final_ready_0_addr + (mma_ops_stage) * 8, ops_free_0_addr + (mma_ops_stage) * 8);
                    _phase_ops_full_0_1 ^= 1;
                    _phase_state_inp_ready_0 ^= 1;
                    _phase_u_inp_ready_0 ^= 1;
                    _phase_u2_inp_ready_0 ^= 1;
                }
                _phase_m_full_0 ^= 1;
            }
            unsigned int _phase_tmem_dealloc_ready_0 = 0;
            {
                mbarrier_wait(tmem_dealloc_ready_addr, _phase_tmem_dealloc_ready_0);
                _phase_tmem_dealloc_ready_0 ^= 1;
                int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(512));
            }
        }
    }
    // ---- Role: load_0 ----
    if (warp == 17) {
        { // load_0_main
            int item_base_3 = blockIdx.x * 12;
            int carry_slab_3 = items[item_base_3];
            int snap_outer_3 = items[item_base_3 + 1];
            int ops_slab_3 = items[item_base_3 + 2];
            int p_begin_3 = items[item_base_3 + 3];
            int p_end_3 = items[item_base_3 + 4];
            int window_chunks_3 = items[item_base_3 + 5];
            int token_base_3 = items[item_base_3 + 6];
            int part_idx_3 = items[item_base_3 + 7];
            int final_outer_3 = items[item_base_3 + 8];
            int head_idx_3 = items[item_base_3 + 9];
            int window_tokens_3 = items[item_base_3 + 10];
            int rows_enabled_3 = items[item_base_3 + 11];
            int num_blocks_3 = p_end_3 - p_begin_3;
            int has_final_3 = ((final_outer_3 >= 0) ? 1 : 0);
            int num_steps_3 = num_blocks_3 + has_final_3;
            int apply_mode_2 = 1;
            int carry_pipe = 0;
            unsigned int _phase_m_free_0 = 1;
            unsigned int _phase_ops_free_0 = 1;
            if (elect_sync()) {
                if (carry_pipe == 0) {
                    mbarrier_arrive_expect_tx(carry_full_addr, 65536);
                    tma_4d_gmem2smem(smem_carry_hi_addr, carry_hi_tma, 0, carry_slab_3 * 128, 0, 0, carry_full_addr);
                    tma_4d_gmem2smem(smem_carry_lo_addr, carry_lo_tma, 0, carry_slab_3 * 128, 0, 0, carry_full_addr);
                }
                unsigned int load_m_stage = 0;
                unsigned int load_ops_stage = 0;
                #pragma unroll 1
                for (int step_3 = 0; step_3 < num_steps_3; step_3 += 2) {
                    int block_1 = p_begin_3 + step_3;
                    if (apply_mode_2 != 0) {
                        mbarrier_wait(m_free_0_addr + (load_m_stage) * 8, _phase_m_free_0);
                        mbarrier_arrive_expect_tx(m_full_0_addr + (load_m_stage) * 8, 32768);
                        if (num_blocks_3 <= step_3) {
                            tma_4d_gmem2smem(smem_map_0_addr + load_m_stage * 32768, map_final_tma, 0, final_outer_3 * 128, 0, 0, m_full_0_addr + (load_m_stage) * 8);
                        } else {
                            tma_4d_gmem2smem(smem_map_0_addr + load_m_stage * 32768, maps_tma, 0, (snap_outer_3 + block_1 * num_heads) * 128, 0, 0, m_full_0_addr + (load_m_stage) * 8);
                        }
                    }
                    int remaining_3 = window_chunks_3 - (p_begin_3 + step_3) * 2;
                    int count_3 = 2;
                    if (remaining_3 < 2) {
                        count_3 = remaining_3;
                    }
                    if (num_blocks_3 <= step_3) {
                        count_3 = 0;
                    }
                    int step_chunks_3 = count_3;
                    #pragma unroll 1
                    for (int chunk_3 = 0; chunk_3 < step_chunks_3; chunk_3++) {
                        long long slab = (long long)(ops_slab_3 + (block_1 * 2 + chunk_3) * num_heads);
                        mbarrier_wait(ops_free_0_addr + (load_ops_stage) * 8, _phase_ops_free_0);
                        mbarrier_arrive_expect_tx(ops_full_0_addr + (load_ops_stage) * 8, 31888);
                        int slot_bf16 = (int)load_ops_stage * 16384;
                        cp_async_bulk_gmem2smem(smem_qd_0_addr + (unsigned int)(slot_bf16 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(op_qd) + ((unsigned long long)(slab * 4096) * (unsigned long long)2)), 8192, ops_full_0_addr + (load_ops_stage) * 8);
                        cp_async_bulk_gmem2smem(smem_kd_0_addr + (unsigned int)(slot_bf16 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(op_kd) + ((unsigned long long)(slab * 4096) * (unsigned long long)2)), 8192, ops_full_0_addr + (load_ops_stage) * 8);
                        cp_async_bulk_gmem2smem(smem_ft_0_addr + (unsigned int)(slot_bf16 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(op_ft) + ((unsigned long long)(slab * 6144) * (unsigned long long)2)), 12288, ops_full_0_addr + (load_ops_stage) * 8);
                        cp_async_bulk_gmem2smem(smem_inv_0_addr + (unsigned int)(slot_bf16 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(op_inv) + ((unsigned long long)(slab * 1024) * (unsigned long long)2)), 2048, ops_full_0_addr + (load_ops_stage) * 8);
                        cp_async_bulk_gmem2smem(smem_vec_0_addr + (unsigned int)((int)load_ops_stage * 8192 * 4), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(op_vec) + ((unsigned long long)(slab * 292) * (unsigned long long)4)), 1168, ops_full_0_addr + (load_ops_stage) * 8);
                        _phase_ops_free_0 ^= 1;
                    }
                    _phase_m_free_0 ^= 1;
                }
            }
        }
    }
    // ---- Role: compute_1 ----
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 144;");
        { // compute_1_main
            int item_base_4 = blockIdx.x * 12;
            int carry_slab_4 = items[item_base_4];
            int snap_outer_4 = items[item_base_4 + 1];
            int ops_slab_4 = items[item_base_4 + 2];
            int p_begin_4 = items[item_base_4 + 3];
            int p_end_4 = items[item_base_4 + 4];
            int window_chunks_4 = items[item_base_4 + 5];
            int token_base_4 = items[item_base_4 + 6];
            int part_idx_4 = items[item_base_4 + 7];
            int final_outer_4 = items[item_base_4 + 8];
            int head_idx_4 = items[item_base_4 + 9];
            int window_tokens_4 = items[item_base_4 + 10];
            int rows_enabled_4 = items[item_base_4 + 11];
            int num_blocks_4 = p_end_4 - p_begin_4;
            int has_final_4 = ((final_outer_4 >= 0) ? 1 : 0);
            int num_steps_4 = num_blocks_4 + has_final_4;
            int warp_in_wg_2 = warp % 4;
            const int tmem_row_base_2 = warp_in_wg_2 * 32 << 16;
            int state_row_2 = warp_in_wg_2 * 32 + lane;
            int pairmap_mode_1 = 0;
            int state_base_2 = taddr + 256 + 64 + (unsigned int)tmem_row_base_2;
            int inp_base_1 = taddr + 256 + (unsigned int)tmem_row_base_2;
            int u_base_1 = taddr + 256 + 224 + (unsigned int)tmem_row_base_2;
            int u2_acc_base_1 = taddr + 256 + (unsigned int)tmem_row_base_2;
            int u2_inp_base_1 = taddr + 256 + 224 + (unsigned int)tmem_row_base_2;
            unsigned int compute_m_stage_1 = 0;
            unsigned int compute_ops_stage_1 = 0;
            unsigned int _phase_state_read_done_1_0 = 1;
            unsigned int _phase_state_ready_1 = 0;
            unsigned int _phase_ops_full_1 = 0;
            unsigned int _phase_u_ready_1 = 0;
            unsigned int _phase_u2_acc_ready_1 = 0;
            unsigned int _phase_snapshot_done_1 = 0;
            unsigned int _phase_final_ready_1 = 0;
            #pragma unroll 1
            for (int step_4 = 1; step_4 < num_steps_4; step_4 += 2) {
                int remaining_4 = window_chunks_4 - (p_begin_4 + step_4) * 2;
                int count_4 = 2;
                if (remaining_4 < 2) {
                    count_4 = remaining_4;
                }
                if (num_blocks_4 <= step_4) {
                    count_4 = 0;
                }
                int step_chunks_4 = count_4;
                float dpair_acc_1 = 1.0f;
                if (pairmap_mode_1 != 0) {
                    mbarrier_wait(state_read_done_1_addr, _phase_state_read_done_1_0);
                    _phase_state_read_done_1_0 ^= 1;
                    #pragma unroll
                    for (int eye_col_block_1 = 0; eye_col_block_1 < 4; eye_col_block_1++) {
                        float eye_1[32];
                        int eye_diag_1 = state_row_2 - eye_col_block_1 * 32;
                        #pragma unroll
                        for (int eye_col_1 = 0; eye_col_1 < 32; eye_col_1++) {
                            eye_1[eye_col_1] = ((eye_diag_1 == eye_col_1) ? 1.0f : 0.0f);
                        }
                        tmem_st_x32_f32(state_base_2 + eye_col_block_1 * 32, eye_1);
                    }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    if (elect_sync()) {
                        mbarrier_arrive(state_ready_1_addr + (compute_m_stage_1) * 8);
                    }
                }
                if (step_chunks_4 > 0) {
                    mbarrier_wait(state_ready_1_addr + (compute_m_stage_1) * 8, _phase_state_ready_1);
                    #pragma unroll
                    for (int state_col_block_3 = 0; state_col_block_3 < 4; state_col_block_3++) {
                        float _tmem_load_6[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_6[0]), "=f"(_tmem_load_6[1]), "=f"(_tmem_load_6[2]), "=f"(_tmem_load_6[3]), "=f"(_tmem_load_6[4]), "=f"(_tmem_load_6[5]), "=f"(_tmem_load_6[6]), "=f"(_tmem_load_6[7]), "=f"(_tmem_load_6[8]), "=f"(_tmem_load_6[9]), "=f"(_tmem_load_6[10]), "=f"(_tmem_load_6[11]), "=f"(_tmem_load_6[12]), "=f"(_tmem_load_6[13]), "=f"(_tmem_load_6[14]), "=f"(_tmem_load_6[15]), "=f"(_tmem_load_6[16]), "=f"(_tmem_load_6[17]), "=f"(_tmem_load_6[18]), "=f"(_tmem_load_6[19]), "=f"(_tmem_load_6[20]), "=f"(_tmem_load_6[21]), "=f"(_tmem_load_6[22]), "=f"(_tmem_load_6[23]), "=f"(_tmem_load_6[24]), "=f"(_tmem_load_6[25]), "=f"(_tmem_load_6[26]), "=f"(_tmem_load_6[27]), "=f"(_tmem_load_6[28]), "=f"(_tmem_load_6[29]), "=f"(_tmem_load_6[30]), "=f"(_tmem_load_6[31])
                            : "r"(state_base_2 + state_col_block_3 * 32));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        uint32_t _tmem_load_6_bf16[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_6[_lp*2 + 0], _tmem_load_6[_lp*2+1 + 0]));
                            _tmem_load_6_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile(
                            "tcgen05.st.sync.aligned.32x32b.x16.b32"
                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                            :: "r"(inp_base_1 + state_col_block_3 * 16), "r"(_tmem_load_6_bf16[0]), "r"(_tmem_load_6_bf16[1]), "r"(_tmem_load_6_bf16[2]), "r"(_tmem_load_6_bf16[3]), "r"(_tmem_load_6_bf16[4]), "r"(_tmem_load_6_bf16[5]), "r"(_tmem_load_6_bf16[6]), "r"(_tmem_load_6_bf16[7]), "r"(_tmem_load_6_bf16[8]), "r"(_tmem_load_6_bf16[9]), "r"(_tmem_load_6_bf16[10]), "r"(_tmem_load_6_bf16[11]), "r"(_tmem_load_6_bf16[12]), "r"(_tmem_load_6_bf16[13]), "r"(_tmem_load_6_bf16[14]), "r"(_tmem_load_6_bf16[15]));
                    }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    if (elect_sync()) {
                        mbarrier_arrive(state_inp_ready_1_addr + (compute_ops_stage_1) * 8);
                    }
                }
                #pragma unroll 1
                for (int chunk_4 = 0; chunk_4 < step_chunks_4; chunk_4++) {
                    mbarrier_wait(ops_full_1_addr + (compute_ops_stage_1) * 8, _phase_ops_full_1);
                    int vec_base_1 = (int)compute_ops_stage_1 * 8192;
                    mbarrier_wait(u_ready_1_addr + (compute_ops_stage_1) * 8, _phase_u_ready_1);
                    float prediction_1[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(prediction_1[0]), "=f"(prediction_1[1]), "=f"(prediction_1[2]), "=f"(prediction_1[3]), "=f"(prediction_1[4]), "=f"(prediction_1[5]), "=f"(prediction_1[6]), "=f"(prediction_1[7]), "=f"(prediction_1[8]), "=f"(prediction_1[9]), "=f"(prediction_1[10]), "=f"(prediction_1[11]), "=f"(prediction_1[12]), "=f"(prediction_1[13]), "=f"(prediction_1[14]), "=f"(prediction_1[15]), "=f"(prediction_1[16]), "=f"(prediction_1[17]), "=f"(prediction_1[18]), "=f"(prediction_1[19]), "=f"(prediction_1[20]), "=f"(prediction_1[21]), "=f"(prediction_1[22]), "=f"(prediction_1[23]), "=f"(prediction_1[24]), "=f"(prediction_1[25]), "=f"(prediction_1[26]), "=f"(prediction_1[27]), "=f"(prediction_1[28]), "=f"(prediction_1[29]), "=f"(prediction_1[30]), "=f"(prediction_1[31])
                        : "r"(u_base_1));
                    #pragma unroll
                    for (int residual_half_1 = 0; residual_half_1 < 2; residual_half_1++) {
                        float residual_1[16];
                        #pragma unroll
                        for (int residual_col_1 = 0; residual_col_1 < 16; residual_col_1++) {
                            float neg_beta_1 = -smem_vec_all_1[vec_base_1 + 257 + residual_half_1 * 16 + residual_col_1];
                            residual_1[residual_col_1] = prediction_1[residual_half_1 * 16 + residual_col_1] * neg_beta_1;
                        }
                        uint32_t residual_bf16_1[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 8; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_1[_lp*2 + 0], residual_1[_lp*2+1 + 0]));
                            residual_bf16_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        tmem_st_x8_u32(u_base_1 + residual_half_1 * 8, (const uint32_t*)residual_bf16_1);
                    }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    if (elect_sync()) {
                        mbarrier_arrive(u_inp_ready_1_addr + (compute_ops_stage_1) * 8);
                    }
                    mbarrier_wait(u2_acc_ready_1_addr + (compute_ops_stage_1) * 8, _phase_u2_acc_ready_1);
                    float _tmem_load_7[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_7[0]), "=f"(_tmem_load_7[1]), "=f"(_tmem_load_7[2]), "=f"(_tmem_load_7[3]), "=f"(_tmem_load_7[4]), "=f"(_tmem_load_7[5]), "=f"(_tmem_load_7[6]), "=f"(_tmem_load_7[7]), "=f"(_tmem_load_7[8]), "=f"(_tmem_load_7[9]), "=f"(_tmem_load_7[10]), "=f"(_tmem_load_7[11]), "=f"(_tmem_load_7[12]), "=f"(_tmem_load_7[13]), "=f"(_tmem_load_7[14]), "=f"(_tmem_load_7[15]), "=f"(_tmem_load_7[16]), "=f"(_tmem_load_7[17]), "=f"(_tmem_load_7[18]), "=f"(_tmem_load_7[19]), "=f"(_tmem_load_7[20]), "=f"(_tmem_load_7[21]), "=f"(_tmem_load_7[22]), "=f"(_tmem_load_7[23]), "=f"(_tmem_load_7[24]), "=f"(_tmem_load_7[25]), "=f"(_tmem_load_7[26]), "=f"(_tmem_load_7[27]), "=f"(_tmem_load_7[28]), "=f"(_tmem_load_7[29]), "=f"(_tmem_load_7[30]), "=f"(_tmem_load_7[31])
                        : "r"(u2_acc_base_1));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    unsigned int u2_packed_1[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_7[_lp*2 + 0], _tmem_load_7[_lp*2+1 + 0]));
                        u2_packed_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(u2_inp_base_1), "r"(u2_packed_1[0]), "r"(u2_packed_1[1]), "r"(u2_packed_1[2]), "r"(u2_packed_1[3]), "r"(u2_packed_1[4]), "r"(u2_packed_1[5]), "r"(u2_packed_1[6]), "r"(u2_packed_1[7]), "r"(u2_packed_1[8]), "r"(u2_packed_1[9]), "r"(u2_packed_1[10]), "r"(u2_packed_1[11]), "r"(u2_packed_1[12]), "r"(u2_packed_1[13]), "r"(u2_packed_1[14]), "r"(u2_packed_1[15]));
                    int do_decay_1 = ((step_chunks_4 > chunk_4 + 1) ? 1 : 0);
                    if (do_decay_1 != 0) {
                        if (pairmap_mode_1 == 0) {
                            mbarrier_wait(snapshot_done_1_addr + (compute_m_stage_1) * 8, _phase_snapshot_done_1);
                        } else {
                            float d_row_1 = smem_vec_all_1[vec_base_1 + state_row_2];
                            dpair_acc_1 = dpair_acc_1 * d_row_1;
                            int publish_dpair_1 = ((step_chunks_4 <= chunk_4 + 1) ? 1 : 0);
                            if (publish_dpair_1 != 0) {
                                smem_dpair_1[state_row_2] = dpair_acc_1;
                                if (elect_sync()) {
                                    mbarrier_arrive(dpair_ready_1_addr);
                                }
                            }
                        }
                        #pragma unroll
                        for (int state_col_block_4 = 0; state_col_block_4 < 4; state_col_block_4++) {
                            int state_addr_1 = state_base_2 + state_col_block_4 * 32;
                            float state_frag_1[32];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(state_frag_1[0]), "=f"(state_frag_1[1]), "=f"(state_frag_1[2]), "=f"(state_frag_1[3]), "=f"(state_frag_1[4]), "=f"(state_frag_1[5]), "=f"(state_frag_1[6]), "=f"(state_frag_1[7]), "=f"(state_frag_1[8]), "=f"(state_frag_1[9]), "=f"(state_frag_1[10]), "=f"(state_frag_1[11]), "=f"(state_frag_1[12]), "=f"(state_frag_1[13]), "=f"(state_frag_1[14]), "=f"(state_frag_1[15]), "=f"(state_frag_1[16]), "=f"(state_frag_1[17]), "=f"(state_frag_1[18]), "=f"(state_frag_1[19]), "=f"(state_frag_1[20]), "=f"(state_frag_1[21]), "=f"(state_frag_1[22]), "=f"(state_frag_1[23]), "=f"(state_frag_1[24]), "=f"(state_frag_1[25]), "=f"(state_frag_1[26]), "=f"(state_frag_1[27]), "=f"(state_frag_1[28]), "=f"(state_frag_1[29]), "=f"(state_frag_1[30]), "=f"(state_frag_1[31])
                                : "r"(state_addr_1));
                            float state_scale_1[16];
                            #pragma unroll
                            for (int state_half_1 = 0; state_half_1 < 2; state_half_1++) {
                                #pragma unroll
                                for (int state_col_1 = 0; state_col_1 < 16; state_col_1++) {
                                    state_scale_1[state_col_1] = smem_vec_all_1[vec_base_1 + state_col_block_4 * 32 + state_half_1 * 16 + state_col_1];
                                }
                                #pragma unroll
                                for (int _ls = 0; _ls < 8; _ls++)
                                    mul_f32x2_inplace(&reinterpret_cast<float2*>((state_frag_1 + state_half_1 * 16))[_ls], reinterpret_cast<const float2*>(state_scale_1)[_ls]);
                            }
                            tmem_st_x32_f32(state_addr_1, state_frag_1);
                        }
                    }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    if (elect_sync()) {
                        mbarrier_arrive(u2_inp_ready_1_addr + (compute_ops_stage_1) * 8);
                    }
                    if (step_chunks_4 > chunk_4 + 1) {
                        mbarrier_wait(final_ready_1_addr + (compute_ops_stage_1) * 8, _phase_final_ready_1);
                        #pragma unroll
                        for (int state_col_block_5 = 0; state_col_block_5 < 4; state_col_block_5++) {
                            float _tmem_load_8[32];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(_tmem_load_8[0]), "=f"(_tmem_load_8[1]), "=f"(_tmem_load_8[2]), "=f"(_tmem_load_8[3]), "=f"(_tmem_load_8[4]), "=f"(_tmem_load_8[5]), "=f"(_tmem_load_8[6]), "=f"(_tmem_load_8[7]), "=f"(_tmem_load_8[8]), "=f"(_tmem_load_8[9]), "=f"(_tmem_load_8[10]), "=f"(_tmem_load_8[11]), "=f"(_tmem_load_8[12]), "=f"(_tmem_load_8[13]), "=f"(_tmem_load_8[14]), "=f"(_tmem_load_8[15]), "=f"(_tmem_load_8[16]), "=f"(_tmem_load_8[17]), "=f"(_tmem_load_8[18]), "=f"(_tmem_load_8[19]), "=f"(_tmem_load_8[20]), "=f"(_tmem_load_8[21]), "=f"(_tmem_load_8[22]), "=f"(_tmem_load_8[23]), "=f"(_tmem_load_8[24]), "=f"(_tmem_load_8[25]), "=f"(_tmem_load_8[26]), "=f"(_tmem_load_8[27]), "=f"(_tmem_load_8[28]), "=f"(_tmem_load_8[29]), "=f"(_tmem_load_8[30]), "=f"(_tmem_load_8[31])
                                : "r"(state_base_2 + state_col_block_5 * 32));
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            uint32_t _tmem_load_8_bf16[16];
                            #pragma unroll
                            for (int _lp = 0; _lp < 16; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_8[_lp*2 + 0], _tmem_load_8[_lp*2+1 + 0]));
                                _tmem_load_8_bf16[_lp] = *(uint32_t*)&_bf2;
                            }
                            asm volatile(
                                "tcgen05.st.sync.aligned.32x32b.x16.b32"
                                " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                                :: "r"(inp_base_1 + state_col_block_5 * 16), "r"(_tmem_load_8_bf16[0]), "r"(_tmem_load_8_bf16[1]), "r"(_tmem_load_8_bf16[2]), "r"(_tmem_load_8_bf16[3]), "r"(_tmem_load_8_bf16[4]), "r"(_tmem_load_8_bf16[5]), "r"(_tmem_load_8_bf16[6]), "r"(_tmem_load_8_bf16[7]), "r"(_tmem_load_8_bf16[8]), "r"(_tmem_load_8_bf16[9]), "r"(_tmem_load_8_bf16[10]), "r"(_tmem_load_8_bf16[11]), "r"(_tmem_load_8_bf16[12]), "r"(_tmem_load_8_bf16[13]), "r"(_tmem_load_8_bf16[14]), "r"(_tmem_load_8_bf16[15]));
                        }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        _phase_ops_full_1 ^= 1;
                        _phase_u_ready_1 ^= 1;
                        _phase_u2_acc_ready_1 ^= 1;
                        _phase_final_ready_1 ^= 1;
                        if (elect_sync()) {
                            mbarrier_arrive(state_inp_ready_1_addr + (compute_ops_stage_1) * 8);
                        }
                    } else {
                        _phase_ops_full_1 ^= 1;
                        _phase_u_ready_1 ^= 1;
                        _phase_u2_acc_ready_1 ^= 1;
                        _phase_final_ready_1 ^= 1;
                    }
                }
                _phase_state_ready_1 ^= 1;
                _phase_snapshot_done_1 ^= 1;
            }
            if (warp_in_wg_2 == 0) {
                if (elect_sync()) {
                    mbarrier_arrive(tmem_dealloc_ready_addr);
                }
            }
        }
    }
    // ---- Role: epilogue_1 ----
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 72;");
        { // epilogue_1_main
            int item_base_5 = blockIdx.x * 12;
            int carry_slab_5 = items[item_base_5];
            int snap_outer_5 = items[item_base_5 + 1];
            int ops_slab_5 = items[item_base_5 + 2];
            int p_begin_5 = items[item_base_5 + 3];
            int p_end_5 = items[item_base_5 + 4];
            int window_chunks_5 = items[item_base_5 + 5];
            int token_base_5 = items[item_base_5 + 6];
            int part_idx_5 = items[item_base_5 + 7];
            int final_outer_5 = items[item_base_5 + 8];
            int head_idx_5 = items[item_base_5 + 9];
            int window_tokens_5 = items[item_base_5 + 10];
            int rows_enabled_5 = items[item_base_5 + 11];
            int num_blocks_5 = p_end_5 - p_begin_5;
            int has_final_5 = ((final_outer_5 >= 0) ? 1 : 0);
            int num_steps_5 = num_blocks_5 + has_final_5;
            int warp_in_wg_3 = warp % 4;
            const int tmem_row_base_3 = warp_in_wg_3 * 32 << 16;
            int state_row_3 = warp_in_wg_3 * 32 + lane;
            int epilogue_local_warp_1 = warp_in_wg_3;
            int state_base_3 = taddr + 256 + 64 + (unsigned int)tmem_row_base_3;
            int out_base_1 = taddr + 256 + 192 + (unsigned int)tmem_row_base_3;
            unsigned int epilogue_m_stage_1 = 0;
            unsigned int epilogue_ops_stage_1 = 0;
            int apply_mode_3 = 1;
            unsigned int _phase_state_ready_1_1 = 0;
            unsigned int _phase_final_ready_1_1 = 0;
            unsigned int _phase_dpair_ready_1_0 = 0;
            #pragma unroll 1
            for (int step_5 = 1; step_5 < num_steps_5; step_5 += 2) {
                int block_2 = p_begin_5 + step_5;
                int remaining_5 = window_chunks_5 - (p_begin_5 + step_5) * 2;
                int count_5 = 2;
                if (remaining_5 < 2) {
                    count_5 = remaining_5;
                }
                if (num_blocks_5 <= step_5) {
                    count_5 = 0;
                }
                int step_chunks_5 = count_5;
                int is_final_1 = ((num_blocks_5 <= step_5) ? 1 : 0);
                int writes_row_1 = ((rows_enabled_5 != 0 && block_2 > 0 && is_final_1 == 0) ? 1 : 0);
                mbarrier_wait(state_ready_1_addr + (epilogue_m_stage_1) * 8, _phase_state_ready_1_1);
                if (writes_row_1 != 0 || is_final_1 != 0) {
                    int panel_outer_1 = final_outer_5;
                    if (is_final_1 == 0) {
                        long long row_start_1 = row_starts[part_idx_5];
                        panel_outer_1 = (int)((row_start_1 + (long long)block_2) * (long long)num_heads + (long long)head_idx_5);
                    }
                    #pragma unroll
                    for (int panel_2 = 0; panel_2 < 4; panel_2++) {
                        float _tmem_load_9[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(_tmem_load_9[0]), "=f"(_tmem_load_9[1]), "=f"(_tmem_load_9[2]), "=f"(_tmem_load_9[3]), "=f"(_tmem_load_9[4]), "=f"(_tmem_load_9[5]), "=f"(_tmem_load_9[6]), "=f"(_tmem_load_9[7]), "=f"(_tmem_load_9[8]), "=f"(_tmem_load_9[9]), "=f"(_tmem_load_9[10]), "=f"(_tmem_load_9[11]), "=f"(_tmem_load_9[12]), "=f"(_tmem_load_9[13]), "=f"(_tmem_load_9[14]), "=f"(_tmem_load_9[15]), "=f"(_tmem_load_9[16]), "=f"(_tmem_load_9[17]), "=f"(_tmem_load_9[18]), "=f"(_tmem_load_9[19]), "=f"(_tmem_load_9[20]), "=f"(_tmem_load_9[21]), "=f"(_tmem_load_9[22]), "=f"(_tmem_load_9[23]), "=f"(_tmem_load_9[24]), "=f"(_tmem_load_9[25]), "=f"(_tmem_load_9[26]), "=f"(_tmem_load_9[27]), "=f"(_tmem_load_9[28]), "=f"(_tmem_load_9[29]), "=f"(_tmem_load_9[30]), "=f"(_tmem_load_9[31])
                            : "r"(state_base_3 + panel_2 * 32));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        if (epilogue_local_warp_1 == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                        }
                        asm volatile("barrier.sync 9, 128;" ::: "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_panel_1_addr + (unsigned int)(state_row_3 * 128 ^ (state_row_3 * 128 >> 7 & 7) << 4))), "r"(__as_u32(_tmem_load_9[0])), "r"(__as_u32(_tmem_load_9[1])), "r"(__as_u32(_tmem_load_9[2])), "r"(__as_u32(_tmem_load_9[3])) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_panel_1_addr + (unsigned int)(state_row_3 * 128 + 16 ^ (state_row_3 * 128 + 16 >> 7 & 7) << 4))), "r"(__as_u32(_tmem_load_9[4])), "r"(__as_u32(_tmem_load_9[5])), "r"(__as_u32(_tmem_load_9[6])), "r"(__as_u32(_tmem_load_9[7])) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_panel_1_addr + (unsigned int)(state_row_3 * 128 + 32 ^ (state_row_3 * 128 + 32 >> 7 & 7) << 4))), "r"(__as_u32(_tmem_load_9[8])), "r"(__as_u32(_tmem_load_9[9])), "r"(__as_u32(_tmem_load_9[10])), "r"(__as_u32(_tmem_load_9[11])) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_panel_1_addr + (unsigned int)(state_row_3 * 128 + 48 ^ (state_row_3 * 128 + 48 >> 7 & 7) << 4))), "r"(__as_u32(_tmem_load_9[12])), "r"(__as_u32(_tmem_load_9[13])), "r"(__as_u32(_tmem_load_9[14])), "r"(__as_u32(_tmem_load_9[15])) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_panel_1_addr + (unsigned int)(state_row_3 * 128 + 64 ^ (state_row_3 * 128 + 64 >> 7 & 7) << 4))), "r"(__as_u32(_tmem_load_9[16])), "r"(__as_u32(_tmem_load_9[17])), "r"(__as_u32(_tmem_load_9[18])), "r"(__as_u32(_tmem_load_9[19])) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_panel_1_addr + (unsigned int)(state_row_3 * 128 + 80 ^ (state_row_3 * 128 + 80 >> 7 & 7) << 4))), "r"(__as_u32(_tmem_load_9[20])), "r"(__as_u32(_tmem_load_9[21])), "r"(__as_u32(_tmem_load_9[22])), "r"(__as_u32(_tmem_load_9[23])) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_panel_1_addr + (unsigned int)(state_row_3 * 128 + 96 ^ (state_row_3 * 128 + 96 >> 7 & 7) << 4))), "r"(__as_u32(_tmem_load_9[24])), "r"(__as_u32(_tmem_load_9[25])), "r"(__as_u32(_tmem_load_9[26])), "r"(__as_u32(_tmem_load_9[27])) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_panel_1_addr + (unsigned int)(state_row_3 * 128 + 112 ^ (state_row_3 * 128 + 112 >> 7 & 7) << 4))), "r"(__as_u32(_tmem_load_9[28])), "r"(__as_u32(_tmem_load_9[29])), "r"(__as_u32(_tmem_load_9[30])), "r"(__as_u32(_tmem_load_9[31])) : "memory");
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 9, 128;" ::: "memory");
                        if (epilogue_local_warp_1 == 0) {
                            if (elect_sync()) {
                                if (is_final_1 != 0) {
                                    tma_store_3d(final_tma, panel_2 * 32, 0, panel_outer_1, smem_panel_1_addr);
                                } else {
                                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 900)
                                    #error "TmaReduceAdd3d requires SM90 or newer"
                                    #endif
                                    asm volatile(
                                        "cp.reduce.async.bulk.tensor.3d.global.shared::cta.add.tile.bulk_group"
                                        " [%0, {%1, %2, %3}], [%4];"
                                        :: "l"(rows_tma), "r"(panel_2 * 32), "r"(0), "r"(panel_outer_1), "r"(smem_panel_1_addr) : "memory");
                                }
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    }
                    if (epilogue_local_warp_1 == 0) {
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                    }
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                }
                if (apply_mode_3 != 0) {
                    if (elect_sync()) {
                        mbarrier_arrive(snapshot_done_1_addr + (epilogue_m_stage_1) * 8);
                        mbarrier_arrive(state_read_done_1_addr);
                    }
                }
                #pragma unroll 1
                for (int chunk_5 = 0; chunk_5 < step_chunks_5; chunk_5++) {
                    int chunk_idx_1 = block_2 * 2 + chunk_5;
                    int chunk_is_full_1 = apply_mode_3;
                    int valid_tokens_1 = window_tokens_5 - chunk_idx_1 * 32;
                    if (valid_tokens_1 > 32) {
                        valid_tokens_1 = 32;
                    }
                    mbarrier_wait(final_ready_1_addr + (epilogue_ops_stage_1) * 8, _phase_final_ready_1_1);
                    if (chunk_is_full_1 != 0) {
                        float _tmem_load_10[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_10[15]))
                            : "r"(out_base_1));
                        float _tmem_load_11[16];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_11[15]))
                            : "r"(out_base_1 + 1048576));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        asm volatile("barrier.sync 9, 128;" ::: "memory");
                        if (epilogue_local_warp_1 == 0) {
                            if (elect_sync()) {
                                mbarrier_arrive(out_empty_1_addr);
                            }
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                        }
                        asm volatile("barrier.sync 9, 128;" ::: "memory");
                        int out_stage_addr_1 = smem_out_1_addr;
                        #pragma unroll
                        for (int dim_half_2 = 0; dim_half_2 < 2; dim_half_2++) {
                            unsigned int out_packed_1[8];
                            if (dim_half_2 == 0) {
                                #pragma unroll
                                for (int _lp = 0; _lp < 8; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_10[_lp*2 + 0], _tmem_load_10[_lp*2+1 + 0]));
                                    out_packed_1[_lp] = *(uint32_t*)&_bf2;
                                }
                            } else {
                                #pragma unroll
                                for (int _lp = 0; _lp < 8; _lp++) {
                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_11[_lp*2 + 0], _tmem_load_11[_lp*2+1 + 0]));
                                    out_packed_1[_lp] = *(uint32_t*)&_bf2;
                                }
                            }
                            #pragma unroll
                            for (int token_group_1 = 0; token_group_1 < 2; token_group_1++) {
                                int mtx_idx_1 = lane / 8;
                                int row_addr_1 = lane & 7;
                                int dim_base_1 = epilogue_local_warp_1 * 32 + dim_half_2 * 16 + (mtx_idx_1 & 1) * 8;
                                int token_base_mtx_1 = token_group_1 * 16 + mtx_idx_1 / 2 * 8;
                                int token_addr_1 = token_base_mtx_1 + row_addr_1;
                                int token_pair_1 = token_addr_1 / 2;
                                int token_parity_1 = token_addr_1 & 1;
                                int raw_row_1 = token_pair_1 + dim_base_1 / 64 * 16;
                                int raw_col_1 = (dim_base_1 & 63 ^ (token_pair_1 & 3) << 4 ^ token_parity_1 << 3) + token_parity_1 * 64;
                                int stsm_offset_1 = (raw_row_1 * 128 + raw_col_1) * 2;
                                const int pack_base_1 = token_group_1 * 4;
                                uint32_t _stmatrix_addr_0 = static_cast<uint32_t>((unsigned long long)(out_stage_addr_1 + stsm_offset_1));
                                asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                    :: "r"(_stmatrix_addr_0), "r"(*reinterpret_cast<const uint32_t*>(&out_packed_1[pack_base_1])), "r"(*reinterpret_cast<const uint32_t*>(&out_packed_1[pack_base_1 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&out_packed_1[pack_base_1 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&out_packed_1[pack_base_1 + 3]))
                                    : "memory");
                            }
                        }
                        int zero_rows_1 = 32 - valid_tokens_1;
                        if (zero_rows_1 > 0) {
                            unsigned int zero_word_1 = 0;
                            int zero_chunks_1 = zero_rows_1 * 16;
                            #pragma unroll
                            for (int zero_k_1 = 0; zero_k_1 < 4; zero_k_1++) {
                                int zero_idx_1 = state_row_3 + zero_k_1 * 128;
                                if (zero_idx_1 < zero_chunks_1) {
                                    int zero_half_1 = zero_idx_1 / (zero_rows_1 * 8);
                                    int zero_within_1 = zero_idx_1 - zero_half_1 * (zero_rows_1 * 8);
                                    int zero_offset_1 = zero_half_1 * 4096 + (valid_tokens_1 + zero_within_1 / 8) * 128 + (zero_within_1 & 7) * 16;
                                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_out_1_addr + (unsigned int)zero_offset_1), "r"(zero_word_1), "r"(zero_word_1), "r"(zero_word_1), "r"(zero_word_1) : "memory");
                                }
                            }
                        }
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 9, 128;" ::: "memory");
                        if (epilogue_local_warp_1 == 0) {
                            if (elect_sync()) {
                                #pragma unroll
                                for (int dim_half_3 = 0; dim_half_3 < 2; dim_half_3++) {
                                    #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 900)
                                    #error "TmaReduceAdd3d requires SM90 or newer"
                                    #endif
                                    asm volatile(
                                        "cp.reduce.async.bulk.tensor.3d.global.shared::cta.add.tile.bulk_group"
                                        " [%0, {%1, %2, %3}], [%4];"
                                        :: "l"(out_tma), "r"(dim_half_3 * 64), "r"(head_idx_5), "r"(token_base_5 + chunk_idx_1 * 32), "r"(smem_out_1_addr + (unsigned int)(dim_half_3 * 4096)) : "memory");
                                }
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    }
                    _phase_final_ready_1_1 ^= 1;
                }
                if (apply_mode_3 == 0) {
                    mbarrier_wait(dpair_ready_1_addr, _phase_dpair_ready_1_0);
                    _phase_dpair_ready_1_0 ^= 1;
                    float dpair_row_1 = smem_dpair_1[state_row_3];
                    int pair_outer_1 = snap_outer_5 + block_2 * num_heads;
                    #pragma unroll
                    for (int panel_3 = 0; panel_3 < 4; panel_3++) {
                        float pair_frag_1[32];
                        asm volatile(
                            "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                            : "=f"(pair_frag_1[0]), "=f"(pair_frag_1[1]), "=f"(pair_frag_1[2]), "=f"(pair_frag_1[3]), "=f"(pair_frag_1[4]), "=f"(pair_frag_1[5]), "=f"(pair_frag_1[6]), "=f"(pair_frag_1[7]), "=f"(pair_frag_1[8]), "=f"(pair_frag_1[9]), "=f"(pair_frag_1[10]), "=f"(pair_frag_1[11]), "=f"(pair_frag_1[12]), "=f"(pair_frag_1[13]), "=f"(pair_frag_1[14]), "=f"(pair_frag_1[15]), "=f"(pair_frag_1[16]), "=f"(pair_frag_1[17]), "=f"(pair_frag_1[18]), "=f"(pair_frag_1[19]), "=f"(pair_frag_1[20]), "=f"(pair_frag_1[21]), "=f"(pair_frag_1[22]), "=f"(pair_frag_1[23]), "=f"(pair_frag_1[24]), "=f"(pair_frag_1[25]), "=f"(pair_frag_1[26]), "=f"(pair_frag_1[27]), "=f"(pair_frag_1[28]), "=f"(pair_frag_1[29]), "=f"(pair_frag_1[30]), "=f"(pair_frag_1[31])
                            : "r"(state_base_3 + panel_3 * 32));
                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                        int diag_local_1 = state_row_3 - panel_3 * 32;
                        #pragma unroll
                        for (int pair_col_1 = 0; pair_col_1 < 32; pair_col_1++) {
                            pair_frag_1[pair_col_1] = pair_frag_1[pair_col_1] - ((diag_local_1 == pair_col_1) ? dpair_row_1 : 0.0f);
                        }
                        if (epilogue_local_warp_1 == 0) {
                            asm volatile("cp.async.bulk.wait_group.read 0;");
                        }
                        asm volatile("barrier.sync 9, 128;" ::: "memory");
                        uint32_t pair_frag_bf16_1[16];
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(pair_frag_1[_lp*2 + 0], pair_frag_1[_lp*2+1 + 0]));
                            pair_frag_bf16_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_panel_bf16_1_addr + (unsigned int)(state_row_3 * 128 ^ (state_row_3 * 128 >> 7 & 7) << 4))), "r"(pair_frag_bf16_1[0]), "r"(pair_frag_bf16_1[1]), "r"(pair_frag_bf16_1[2]), "r"(pair_frag_bf16_1[3]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_panel_bf16_1_addr + (unsigned int)(state_row_3 * 128 + 16 ^ (state_row_3 * 128 + 16 >> 7 & 7) << 4))), "r"(pair_frag_bf16_1[4]), "r"(pair_frag_bf16_1[5]), "r"(pair_frag_bf16_1[6]), "r"(pair_frag_bf16_1[7]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_panel_bf16_1_addr + (unsigned int)(state_row_3 * 128 + 32 ^ (state_row_3 * 128 + 32 >> 7 & 7) << 4))), "r"(pair_frag_bf16_1[8]), "r"(pair_frag_bf16_1[9]), "r"(pair_frag_bf16_1[10]), "r"(pair_frag_bf16_1[11]) : "memory");
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_panel_bf16_1_addr + (unsigned int)(state_row_3 * 128 + 48 ^ (state_row_3 * 128 + 48 >> 7 & 7) << 4))), "r"(pair_frag_bf16_1[12]), "r"(pair_frag_bf16_1[13]), "r"(pair_frag_bf16_1[14]), "r"(pair_frag_bf16_1[15]) : "memory");
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile("barrier.sync 9, 128;" ::: "memory");
                        if (epilogue_local_warp_1 == 0) {
                            if (elect_sync()) {
                                tma_store_3d(pair_tma, panel_3 * 32, 0, pair_outer_1, smem_panel_bf16_1_addr);
                            }
                            asm volatile("cp.async.bulk.commit_group;");
                        }
                    }
                    dpair[(long long)(pair_outer_1 * 128 + state_row_3)] = dpair_row_1;
                    if (epilogue_local_warp_1 == 0) {
                        asm volatile("cp.async.bulk.wait_group.read 0;");
                    }
                    asm volatile("barrier.sync 9, 128;" ::: "memory");
                    if (elect_sync()) {
                        mbarrier_arrive(state_read_done_1_addr);
                    }
                }
                _phase_state_ready_1_1 ^= 1;
            }
            if (epilogue_local_warp_1 == 0) {
                asm volatile("cp.async.bulk.wait_group 0;");
                if (elect_sync()) {
                    mbarrier_arrive(tmem_dealloc_ready_addr);
                }
            }
        }
    }
    // ---- Role: mma_1 ----
    if (warp == 18) {
        { // mma_1_main
            int item_base_6 = blockIdx.x * 12;
            int carry_slab_6 = items[item_base_6];
            int snap_outer_6 = items[item_base_6 + 1];
            int ops_slab_6 = items[item_base_6 + 2];
            int p_begin_6 = items[item_base_6 + 3];
            int p_end_6 = items[item_base_6 + 4];
            int window_chunks_6 = items[item_base_6 + 5];
            int token_base_6 = items[item_base_6 + 6];
            int part_idx_6 = items[item_base_6 + 7];
            int final_outer_6 = items[item_base_6 + 8];
            int head_idx_6 = items[item_base_6 + 9];
            int window_tokens_6 = items[item_base_6 + 10];
            int rows_enabled_6 = items[item_base_6 + 11];
            int num_blocks_6 = p_end_6 - p_begin_6;
            int has_final_6 = ((final_outer_6 >= 0) ? 1 : 0);
            int num_steps_6 = num_blocks_6 + has_final_6;
            unsigned int mma_m_stage_1 = 0;
            unsigned int mma_ops_stage_1 = 0;
            int apply_mode_4 = 1;
            unsigned int _phase_carry_full_0_1 = 0;
            if (apply_mode_4 != 0) {
                mbarrier_wait(carry_full_addr, _phase_carry_full_0_1);
                _phase_carry_full_0_1 ^= 1;
            }
            unsigned int _phase_m_full_1 = 0;
            unsigned int _phase_state_read_done_1_0_1 = 1;
            unsigned int _phase_ops_full_1_1 = 0;
            unsigned int _phase_state_inp_ready_1 = 0;
            unsigned int _phase_out_empty_1_0 = 1;
            unsigned int _phase_u_inp_ready_1 = 0;
            unsigned int _phase_u2_inp_ready_1 = 0;
            #pragma unroll 1
            for (int step_6 = 1; step_6 < num_steps_6; step_6 += 2) {
                if (apply_mode_4 != 0) {
                    mbarrier_wait(m_full_1_addr + (mma_m_stage_1) * 8, _phase_m_full_1);
                    mbarrier_wait(state_read_done_1_addr, _phase_state_read_done_1_0_1);
                    _phase_state_read_done_1_0_1 ^= 1;
                    int _mma_a_lo_7 = make_warp_uniform(((smem_carry_hi_addr) >> 4) & 0x3FFF);
                    int _mma_b_lo_7 = make_warp_uniform(((((smem_map_1_addr) >> 4) & 0x3FFF) | 0x4000000) + (mma_m_stage_1) * 2048);
                    {
                        uint64_t _mma_ss_a_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_7);
                        uint64_t _mma_ss_b_desc_0 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_7);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_state_1, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136381584, 0);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 128U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_state_1, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136381584, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 128U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_state_1, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136381584, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 128U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_state_1, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136381584, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 1018U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 128U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_state_1, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136381584, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 128U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_state_1, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136381584, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 128U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_state_1, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136381584, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_0, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_0, 128U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_state_1, _mma_ss_a_desc_0, _mma_ss_b_desc_0, 136381584, 1);
                        }
                    }
                    int _mma_a_lo_8 = make_warp_uniform(((smem_carry_lo_addr) >> 4) & 0x3FFF);
                    int _mma_b_lo_8 = make_warp_uniform(((((smem_map_1_addr) >> 4) & 0x3FFF) | 0x4000000) + (mma_m_stage_1) * 2048);
                    {
                        uint64_t _mma_ss_a_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_a_lo_8);
                        uint64_t _mma_ss_b_desc_1 = (static_cast<uint64_t>(0x40004040U) << 32) | static_cast<uint32_t>(_mma_b_lo_8);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_state_1, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 136381584, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 128U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_state_1, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 136381584, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 128U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_state_1, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 136381584, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 128U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_state_1, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 136381584, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 1018U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 128U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_state_1, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 136381584, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 128U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_state_1, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 136381584, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 128U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_state_1, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 136381584, 1);
                        }
                        incr_smem_desc_lo(_mma_ss_a_desc_1, 2U);
                        incr_smem_desc_lo(_mma_ss_b_desc_1, 128U);
                        if (elect_sync()) {
                            tcgen05_mma_f16(tmem_tmem_state_1, _mma_ss_a_desc_1, _mma_ss_b_desc_1, 136381584, 1);
                        }
                    }
                    elect_commit2(state_ready_1_addr + (mma_m_stage_1) * 8, m_free_1_addr + (mma_m_stage_1) * 8);
                }
                int remaining_6 = window_chunks_6 - (p_begin_6 + step_6) * 2;
                int count_6 = 2;
                if (remaining_6 < 2) {
                    count_6 = remaining_6;
                }
                if (num_blocks_6 <= step_6) {
                    count_6 = 0;
                }
                int step_chunks_6 = count_6;
                #pragma unroll 1
                for (int chunk_6 = 0; chunk_6 < step_chunks_6; chunk_6++) {
                    mbarrier_wait(ops_full_1_addr + (mma_ops_stage_1) * 8, _phase_ops_full_1_1);
                    mbarrier_wait(state_inp_ready_1_addr + (mma_ops_stage_1) * 8, _phase_state_inp_ready_1);
                    if (apply_mode_4 != 0) {
                        mbarrier_wait(out_empty_1_addr, _phase_out_empty_1_0);
                        _phase_out_empty_1_0 ^= 1;
                        int _mma_b_lo_9 = make_warp_uniform((((smem_qd_1_addr) >> 4) & 0x3FFF) + (mma_ops_stage_1) * 2048);
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
                    "mov.b32 id, 134743184;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 250;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_out_1), "r"(_mma_b_lo_9), "r"(tmem_tmem_state_inp_1), "r"(0));
                    }
                    int _mma_b_lo_10 = make_warp_uniform((((smem_kd_1_addr) >> 4) & 0x3FFF) + (mma_ops_stage_1) * 2048);
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
                    "mov.b32 id, 134743184;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 250;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_u_acc_1), "r"(_mma_b_lo_10), "r"(tmem_tmem_state_inp_1), "r"(0));
                    elect_commit(u_ready_1_addr + (mma_ops_stage_1) * 8);
                    mbarrier_wait(u_inp_ready_1_addr + (mma_ops_stage_1) * 8, _phase_u_inp_ready_1);
                    int _mma_b_lo_11 = make_warp_uniform((((smem_inv_1_addr) >> 4) & 0x3FFF) + (mma_ops_stage_1) * 2048);
                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0xC0004010;\n\t"
                    "mov.b32 id, 134743184;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_u2_acc_1), "r"(_mma_b_lo_11), "r"(tmem_tmem_u2_inp_1), "r"(0));
                    elect_commit(u2_acc_ready_1_addr + (mma_ops_stage_1) * 8);
                    mbarrier_wait(u2_inp_ready_1_addr + (mma_ops_stage_1) * 8, _phase_u2_inp_ready_1);
                    int want_state_1 = ((step_chunks_6 > chunk_6 + 1) ? 1 : 0);
                    if (want_state_1 != 0) {
                        int _mma_b_lo_12 = make_warp_uniform(((((smem_ft_1_addr) >> 4) & 0x3FFF) | 0x1000000) + (mma_ops_stage_1) * 2048);
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
                    "mov.b32 id, 136905872;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_state_out_1), "r"(_mma_b_lo_12), "r"(tmem_tmem_u2_inp_1), "r"(1));
                    } else {
                        int _mma_b_lo_13 = make_warp_uniform(((((smem_mqk_1_addr) >> 4) & 0x3FFF) | 0x1000000) + (mma_ops_stage_1) * 2048);
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
                    "mov.b32 id, 134808720;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 128;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::f16 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_out_1), "r"(_mma_b_lo_13), "r"(tmem_tmem_u2_inp_1), "r"(1));
                    }
                    elect_commit2(final_ready_1_addr + (mma_ops_stage_1) * 8, ops_free_1_addr + (mma_ops_stage_1) * 8);
                    _phase_ops_full_1_1 ^= 1;
                    _phase_state_inp_ready_1 ^= 1;
                    _phase_u_inp_ready_1 ^= 1;
                    _phase_u2_inp_ready_1 ^= 1;
                }
                _phase_m_full_1 ^= 1;
            }
            unsigned int _phase_tmem_dealloc_ready_0_1 = 0;
        }
    }
    // ---- Role: load_1 ----
    if (warp == 19) {
        { // load_1_main
            int item_base_7 = blockIdx.x * 12;
            int carry_slab_7 = items[item_base_7];
            int snap_outer_7 = items[item_base_7 + 1];
            int ops_slab_7 = items[item_base_7 + 2];
            int p_begin_7 = items[item_base_7 + 3];
            int p_end_7 = items[item_base_7 + 4];
            int window_chunks_7 = items[item_base_7 + 5];
            int token_base_7 = items[item_base_7 + 6];
            int part_idx_7 = items[item_base_7 + 7];
            int final_outer_7 = items[item_base_7 + 8];
            int head_idx_7 = items[item_base_7 + 9];
            int window_tokens_7 = items[item_base_7 + 10];
            int rows_enabled_7 = items[item_base_7 + 11];
            int num_blocks_7 = p_end_7 - p_begin_7;
            int has_final_7 = ((final_outer_7 >= 0) ? 1 : 0);
            int num_steps_7 = num_blocks_7 + has_final_7;
            int apply_mode_5 = 1;
            int carry_pipe_1 = 1;
            unsigned int _phase_m_free_1 = 1;
            unsigned int _phase_ops_free_1 = 1;
            if (elect_sync()) {
                if (carry_pipe_1 == 0) {
                    mbarrier_arrive_expect_tx(carry_full_addr, 65536);
                    tma_4d_gmem2smem(smem_carry_hi_addr, carry_hi_tma, 0, carry_slab_7 * 128, 0, 0, carry_full_addr);
                    tma_4d_gmem2smem(smem_carry_lo_addr, carry_lo_tma, 0, carry_slab_7 * 128, 0, 0, carry_full_addr);
                }
                unsigned int load_m_stage_1 = 0;
                unsigned int load_ops_stage_1 = 0;
                #pragma unroll 1
                for (int step_7 = 1; step_7 < num_steps_7; step_7 += 2) {
                    int block_3 = p_begin_7 + step_7;
                    if (apply_mode_5 != 0) {
                        mbarrier_wait(m_free_1_addr + (load_m_stage_1) * 8, _phase_m_free_1);
                        mbarrier_arrive_expect_tx(m_full_1_addr + (load_m_stage_1) * 8, 32768);
                        if (num_blocks_7 <= step_7) {
                            tma_4d_gmem2smem(smem_map_1_addr + load_m_stage_1 * 32768, map_final_tma, 0, final_outer_7 * 128, 0, 0, m_full_1_addr + (load_m_stage_1) * 8);
                        } else {
                            tma_4d_gmem2smem(smem_map_1_addr + load_m_stage_1 * 32768, maps_tma, 0, (snap_outer_7 + block_3 * num_heads) * 128, 0, 0, m_full_1_addr + (load_m_stage_1) * 8);
                        }
                    }
                    int remaining_7 = window_chunks_7 - (p_begin_7 + step_7) * 2;
                    int count_7 = 2;
                    if (remaining_7 < 2) {
                        count_7 = remaining_7;
                    }
                    if (num_blocks_7 <= step_7) {
                        count_7 = 0;
                    }
                    int step_chunks_7 = count_7;
                    #pragma unroll 1
                    for (int chunk_7 = 0; chunk_7 < step_chunks_7; chunk_7++) {
                        long long slab_1 = (long long)(ops_slab_7 + (block_3 * 2 + chunk_7) * num_heads);
                        mbarrier_wait(ops_free_1_addr + (load_ops_stage_1) * 8, _phase_ops_free_1);
                        mbarrier_arrive_expect_tx(ops_full_1_addr + (load_ops_stage_1) * 8, 31888);
                        int slot_bf16_1 = (int)load_ops_stage_1 * 16384;
                        cp_async_bulk_gmem2smem(smem_qd_1_addr + (unsigned int)(slot_bf16_1 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(op_qd) + ((unsigned long long)(slab_1 * 4096) * (unsigned long long)2)), 8192, ops_full_1_addr + (load_ops_stage_1) * 8);
                        cp_async_bulk_gmem2smem(smem_kd_1_addr + (unsigned int)(slot_bf16_1 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(op_kd) + ((unsigned long long)(slab_1 * 4096) * (unsigned long long)2)), 8192, ops_full_1_addr + (load_ops_stage_1) * 8);
                        cp_async_bulk_gmem2smem(smem_ft_1_addr + (unsigned int)(slot_bf16_1 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(op_ft) + ((unsigned long long)(slab_1 * 6144) * (unsigned long long)2)), 12288, ops_full_1_addr + (load_ops_stage_1) * 8);
                        cp_async_bulk_gmem2smem(smem_inv_1_addr + (unsigned int)(slot_bf16_1 * 2), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(op_inv) + ((unsigned long long)(slab_1 * 1024) * (unsigned long long)2)), 2048, ops_full_1_addr + (load_ops_stage_1) * 8);
                        cp_async_bulk_gmem2smem(smem_vec_1_addr + (unsigned int)((int)load_ops_stage_1 * 8192 * 4), reinterpret_cast<const void*>(reinterpret_cast<const uint8_t*>(op_vec) + ((unsigned long long)(slab_1 * 292) * (unsigned long long)4)), 1168, ops_full_1_addr + (load_ops_stage_1) * 8);
                        _phase_ops_free_1 ^= 1;
                    }
                    _phase_m_free_1 ^= 1;
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
