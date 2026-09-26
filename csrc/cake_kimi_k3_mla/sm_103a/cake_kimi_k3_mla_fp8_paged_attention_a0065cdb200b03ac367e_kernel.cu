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
#define TMEM_TMEM_SCRATCH_OFFSET 0
#define NUM_K_PIPE_STAGES 3
#define NUM_V_PIPE_STAGES 2
#define SMEM_SMEM_Q_OFF 1024
#define SMEM_SMEM_Q_STAGE_BYTES 32768
#define SMEM_SMEM_Q_STRIDE 32768
#define SMEM_SMEM_QR_OFF 33792
#define SMEM_SMEM_QR_STAGE_BYTES 4096
#define SMEM_SMEM_QR_STRIDE 4096
#define SMEM_SMEM_K_OFF 37888
#define SMEM_SMEM_K_STAGE_BYTES 32768
#define SMEM_SMEM_K_STRIDE 36864
#define SMEM_SMEM_KR_OFF 70656
#define SMEM_SMEM_KR_STAGE_BYTES 4096
#define SMEM_SMEM_KR_STRIDE 36864
#define SMEM_SMEM_V_OFF 148480
#define SMEM_SMEM_V_STAGE_BYTES 32768
#define SMEM_SMEM_V_STRIDE 32768
#define SMEM_SMEM_STATS_MAX_OFF 230400
#define SMEM_SMEM_STATS_MAX_STAGE_BYTES 1280
#define SMEM_SMEM_STATS_MAX_STRIDE 1280
#define SMEM_SMEM_STATS_SUM_OFF 231424
#define SMEM_SMEM_STATS_SUM_STAGE_BYTES 1024
#define SMEM_SMEM_STATS_SUM_STRIDE 1024
#define SMEM_SMEM_STATS_RUN_OFF 231680
#define SMEM_SMEM_STATS_RUN_STAGE_BYTES 256
#define SMEM_SMEM_STATS_RUN_STRIDE 256
#define SMEM_SMEM_P_OFF 214016
#define SMEM_SMEM_P_STAGE_BYTES 8192
#define SMEM_SMEM_P_STRIDE 8192
#define SMEM_SMEM_EPI_OFF 1024
#define SMEM_SMEM_EPI_STAGE_BYTES 16384
#define SMEM_SMEM_EPI_STRIDE 16384
#define SMEM_TOTAL 232448
#define THREADS 384
#define SM103_SCORE_LD_RED_MAX 1
#define SM100_SCORE_TREE_MAX 0
#define SM100_ROW_SUM_CHAINS 0
#define SM100_DEFER_ROW_SUM 0
#define SM100_CORRECTION_BATCH 0
#define SM100_F16X2_EXP 0
#define SM100_EXP2_EMU25 0
#define SM100_EXP2_EMU44 0
#define SM100_EXP2_EMU62 0
#define SM100_DROP_PROXY_FENCES 0
#define SM100_EXP2_EMU31 0
#define SM100_EXP2_EMU41 0
#define SM100_PACKED_MATH 0
#define SM100_KO_NO_EXP 0
#define SM100_KO_NO_RESCALE 0
#define SM100_KO_NO_EXCHANGE 0
#define SM100_KO_NO_P_STORE 0
#define SM100_KO_NO_SOFTMAX 0
#define SM100_KO_FREE_MMA 0
#define SM100_KO_NO_K_LOAD 0
#define SM100_KO_NO_V_LOAD 0
#define SM100_KO_HALF_V 0
#define SM100_KO_RESCALE_NEVER 0
#define SM100_E4M3_GRID_SUM 0
#define PP_KO_REF_HANDOFF 0
#define PP_KO_RESCALE 0
#define PP_KO_PAIR_SYNC 0
#define PP_BAR_IDS_HI 0

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


__device__ __forceinline__ void mbarrier_wait_token_test(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        asm volatile(
            "{\n\t"
            ".reg .pred P1;\n\t"
            "LAB_WAIT_TEST:\n\t"
            "mbarrier.test_wait.parity.acquire.cta.shared::cta.b64"
            " P1, [%0], %1;\n\t"
            "@P1 bra.uni DONE_TEST;\n\t"
            "bra.uni LAB_WAIT_TEST;\n\t"
            "DONE_TEST:\n\t"
            "}\n"
            :: "r"(mbar_addr), "r"(phase) : "memory");
    }
}

__device__ __forceinline__ void mbarrier_wait_token_test_cluster(int mbar_addr, int phase, uint32_t token) {
    if (token == 0) {
        asm volatile(
            "{\n\t"
            ".reg .pred P1;\n\t"
            "LAB_WAIT_TEST_CLUSTER:\n\t"
            "mbarrier.test_wait.parity.acquire.cluster.shared::cta.b64"
            " P1, [%0], %1;\n\t"
            "@P1 bra.uni DONE_TEST_CLUSTER;\n\t"
            "bra.uni LAB_WAIT_TEST_CLUSTER;\n\t"
            "DONE_TEST_CLUSTER:\n\t"
            "}\n"
            :: "r"(mbar_addr), "r"(phase) : "memory");
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


__device__ __forceinline__ float2 ex2_emulation_f32x2_value(float2 value) {
    const float c0 = 1.0f, c1 = 0.695146143436431884765625f;
    const float c2 = 0.227564394474029541015625f, c3 = 0.077119089663028717041015625f;
    const float magic = 12582912.0f;
    float x0 = max_noftz(value.x, -127.0f), x1 = max_noftz(value.y, -127.0f);
    float2 xc2 = make_float2(x0, x1), magic2 = make_float2(magic, magic);
    float2 xr2;
    asm("add.rm.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xr2)
        : "l"(*(unsigned long long*)&xc2), "l"(*(unsigned long long*)&magic2));
    float2 c3_2 = make_float2(c3, c3), c2_2 = make_float2(c2, c2);
    float2 c1_2 = make_float2(c1, c1), c0_2 = make_float2(c0, c0);
    float2 xrb2, xfrac2;
    asm("sub.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xrb2)
        : "l"(*(unsigned long long*)&xr2), "l"(*(unsigned long long*)&magic2));
    asm("sub.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&xfrac2)
        : "l"(*(unsigned long long*)&xc2), "l"(*(unsigned long long*)&xrb2));
    float2 poly2;
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;" : "=l"(*(unsigned long long*)&poly2)
        : "l"(*(unsigned long long*)&c3_2), "l"(*(unsigned long long*)&xfrac2), "l"(*(unsigned long long*)&c2_2));
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;" : "=l"(*(unsigned long long*)&poly2)
        : "l"(*(unsigned long long*)&poly2), "l"(*(unsigned long long*)&xfrac2), "l"(*(unsigned long long*)&c1_2));
    asm("fma.rn.ftz.f32x2 %0, %1, %2, %3;" : "=l"(*(unsigned long long*)&poly2)
        : "l"(*(unsigned long long*)&poly2), "l"(*(unsigned long long*)&xfrac2), "l"(*(unsigned long long*)&c0_2));
    int x0r_i, x1r_i, p0_i, p1_i;
    asm("mov.b64 {%0, %1}, %2;" : "=r"(x0r_i), "=r"(x1r_i) : "l"(*(unsigned long long*)&xr2));
    asm("mov.b64 {%0, %1}, %2;" : "=r"(p0_i), "=r"(p1_i) : "l"(*(unsigned long long*)&poly2));
    float r0, r1;
    asm("mov.b32 %0, %1;" : "=f"(r0) : "r"((x0r_i << 23) + p0_i));
    asm("mov.b32 %0, %1;" : "=f"(r1) : "r"((x1r_i << 23) + p1_i));
    return make_float2(r0, r1);
}

__device__ __forceinline__ void ex2_emulation_f32x2(float* x0_ptr, float* x1_ptr) {
    float2 result = ex2_emulation_f32x2_value(make_float2(*x0_ptr, *x1_ptr));
    *x0_ptr = result.x; *x1_ptr = result.y;
}

__device__ __forceinline__ void softmax_frag_exp2_cast(
    float* sv, uint32_t* pv, int use_emu)
{
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        if (use_emu && j >= 12)
            ex2_emulation_f32x2(&sv[j*2], &sv[j*2+1]);
        else {
            sv[j*2]   = approx_exp2(sv[j*2]);
            sv[j*2+1] = approx_exp2(sv[j*2+1]);
        }
    }
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        __nv_bfloat162 bf = __float22bfloat162_rn({sv[j*2], sv[j*2+1]});
        pv[j] = reinterpret_cast<uint32_t&>(bf);
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


__device__ __forceinline__ void tma_2d_gmem2smem(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_2d_gmem2smem_cta2(
    int dst, const void *tmap_ptr, int x, int y, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
        " [%0], [%1, {%2, %3}], [%4];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y),
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

extern "C" {

__global__ __launch_bounds__(384, 1) __cluster_dims__(2,1,1) void
kernel_cake_kimi_k3_mla_fp8_paged_attention_a0065cdb200b03ac367e(const __grid_constant__ CUtensorMap tmap_q, const __grid_constant__ CUtensorMap tmap_k, const __grid_constant__ CUtensorMap tmap_qr, const __grid_constant__ CUtensorMap tmap_kr, const __grid_constant__ CUtensorMap tmap_v, __nv_bfloat16* __restrict__ partial_O, float* __restrict__ partial_max, float* __restrict__ partial_sum, int* __restrict__ seq_lens, int* __restrict__ cum_seq_lens_q, int* __restrict__ page_table, float softmax_scale_log2, float bmm2_scale, int num_heads, int num_split, int max_pages_per_seq)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define q_full_addr (mbar_base + 0)
    #define q_empty_addr (mbar_base + 8)
    #define k_full_addr (mbar_base + 16)
    #define k_empty_addr (mbar_base + 40)
    #define v_full_addr (mbar_base + 64)
    #define v_empty_addr (mbar_base + 80)
    #define s_full_addr (mbar_base + 96)
    #define s_empty_addr (mbar_base + 112)
    #define p_full_addr (mbar_base + 128)
    #define ref_full_addr (mbar_base + 144)
    #define p_empty_addr (mbar_base + 160)
    #define o_empty_addr (mbar_base + 176)
    #define sum_ready_addr (mbar_base + 192)
    #define o_full_addr (mbar_base + 200)
    #define o_done_addr (mbar_base + 216)
    #define partial_results_ready_addr (mbar_base + 224)
    #define q_pair_ready_addr (mbar_base + 232)
    #define tmem_dealloc_peer_addr (mbar_base + 240)
    #define tmem_epoch_ready_addr (mbar_base + 248)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));

    // Kernel setup ops
    uint8_t* smem_q = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_q_addr = smem + 1024;
    uint8_t* smem_qr = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int smem_qr_addr = smem + 33792;
    uint8_t* smem_k = reinterpret_cast<uint8_t*>(smem_raw + 37888);
    const int smem_k_addr = smem + 37888;
    uint8_t* smem_kr = reinterpret_cast<uint8_t*>(smem_raw + 70656);
    const int smem_kr_addr = smem + 70656;
    uint8_t* smem_v = reinterpret_cast<uint8_t*>(smem_raw + 148480);
    const int smem_v_addr = smem + 148480;
    float* smem_stats_max = reinterpret_cast<float*>(smem_raw + 230400);
    const int smem_stats_max_addr = smem + 230400;
    float* smem_stats_sum = reinterpret_cast<float*>(smem_raw + 231424);
    const int smem_stats_sum_addr = smem + 231424;
    float* smem_stats_run = reinterpret_cast<float*>(smem_raw + 231680);
    const int smem_stats_run_addr = smem + 231680;
    uint8_t* smem_p = reinterpret_cast<uint8_t*>(smem_raw + 214016);
    const int smem_p_addr = smem + 214016;
    unsigned int* smem_epi = reinterpret_cast<unsigned int*>(smem_raw + 1024);
    const int smem_epi_addr = smem + 1024;

    // Mbarrier init (19 pipeline groups, 0 ordered-sequence groups, 32 barriers)
    // Mbarriers at smem_raw[0..256)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            // q_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // --- pipeline 'k_pipe' ---
            // k_full: 3 barriers, init_count=1
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            // k_empty: 3 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // --- pipeline 'v_pipe' ---
            // v_full: 2 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // v_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            // s_empty: 2 barriers, init_count=8
            mbarrier_init(smem + 112, 8);
            mbarrier_init(smem + 120, 8);
            // p_full: 2 barriers, init_count=8
            mbarrier_init(smem + 128, 8);
            mbarrier_init(smem + 136, 8);
            // ref_full: 2 barriers, init_count=4
            mbarrier_init(smem + 144, 4);
            mbarrier_init(smem + 152, 4);
            // p_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            // o_empty: 2 barriers, init_count=8
            mbarrier_init(smem + 176, 8);
            mbarrier_init(smem + 184, 8);
            // sum_ready: 1 barriers, init_count=256
            mbarrier_init(smem + 192, 256);
            // o_full: 2 barriers, init_count=1
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            // o_done: 1 barriers, init_count=1
            mbarrier_init(smem + 216, 1);
            // partial_results_ready: 1 barriers, init_count=256
            mbarrier_init(smem + 224, 256);
            // q_pair_ready: 1 barriers, init_count=64
            mbarrier_init(smem + 232, 64);
            // tmem_dealloc_peer: 1 barriers, init_count=32
            mbarrier_init(smem + 240, 32);
            // tmem_epoch_ready: 1 barriers, init_count=256
            mbarrier_init(smem + 248, 256);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 256);
    if (warp == 0) {
        int _tmem_hold = smem + 256;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    __syncthreads();
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_scratch = taddr;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 112;");
    }

    // ---- Role: softmax_wg ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 200;");
        { // softmax_wg_main
            const int wg_dummy_inc = 0;
            const int tmem_row_base = warp % 2 * 32;
            const int half = warp % 4 / 2;
            const int o_tmem_row_base = warp % 4 * 32;
            const int corr_row = o_tmem_row_base << 16;
            const int n_half = half;
            const int my_row = tmem_row_base + lane;
            int split_idx = blockIdx.x / 2;
            int m_tile = gridDim.y - 1 - blockIdx.y;
            int b = blockIdx.z;
            int q_start = cum_seq_lens_q[b];
            int q_len_b = cum_seq_lens_q[b + 1] - q_start;
            int kv_len = seq_lens[b];
            int rows_b = q_len_b * num_heads;
            int row0 = m_tile * 128;
            int rows_left = rows_b - row0;
            int rows_pos = ((rows_left < 0) ? 0 : rows_left);
            int rows_valid = ((rows_pos > 128) ? 128 : rows_pos);
            int row_base_global = q_start * num_heads + row0;
            int last_row = row0 + rows_valid - 1;
            int t_last = last_row / num_heads;
            int kv_end_raw = kv_len - q_len_b + t_last + 1;
            int kv_end = ((rows_valid == 0) ? 0 : kv_end_raw);
            int n_pages = (kv_len + 64 - 1) / 64;
            int n_tiles_total = (kv_end + 128 - 1) / 128;
            int tiles_per_split = (n_tiles_total + num_split - 1) / num_split;
            int my_start_raw = split_idx * tiles_per_split;
            int my_end_raw = my_start_raw + tiles_per_split;
            int my_end = ((my_end_raw > n_tiles_total) ? n_tiles_total : my_end_raw);
            int my_n_raw = my_end - my_start_raw;
            int empty = ((my_n_raw < 1) ? 1 : 0);
            int my_n_tiles = ((my_n_raw < 1) ? 1 : my_n_raw);
            int my_start_block = ((my_n_raw < 1) ? 0 : my_start_raw);
            int pt_base = b * max_pages_per_seq;
            int r_local = cta_rank * 64 + my_row;
            int row_tok_raw = (row0 + r_local) / num_heads;
            int q_last = q_len_b - 1;
            int row_tok = ((row_tok_raw > q_last) ? q_last : row_tok_raw);
            int row_kv_end_raw = kv_len - q_len_b + row_tok + 1;
            int row_kv_end = ((empty != 0) ? 0 : row_kv_end_raw);
            int out_row = row_base_global + r_local;
            unsigned int _phase_q_full_0 = 0;
            mbarrier_wait(q_full_addr, _phase_q_full_0);
            _phase_q_full_0 ^= 1;
            float row_max_val = -CAKE_INF;
            float row_sum_val = 0.0f;
            int n_my_tiles = (my_n_tiles + 1) / 2;
            #pragma unroll 1
            for (int k = 0; k < n_my_tiles; k++) {
                int tile = 2 * k;
                int s_wait_phase = k & 1;
                float sv[64];
                float local_max = -CAKE_INF;
                int s_base = taddr + (unsigned int)(o_tmem_row_base << 16);
                int abs_tile = my_start_block + tile;
                int tail_valid = row_kv_end - abs_tile * 128;
                int local_tail_raw = tail_valid - half * 64;
                int local_tail_hi = ((local_tail_raw > 64) ? 64 : local_tail_raw);
                int local_tail = ((local_tail_hi < 0) ? 0 : local_tail_hi);
                {
                    uint32_t _mbar_token_0 = mbarrier_test_wait(s_full_addr, s_wait_phase);
                    int _vote_0 = __any_sync(0xFFFFFFFF, local_tail < 64);
                    int tail_any = _vote_0;
                    float lmax_a = -CAKE_INF;
                    float lmax_b = -CAKE_INF;
                    if (tail_any != 0) {
                        mbarrier_wait_token(s_full_addr, s_wait_phase, _mbar_token_0);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                        #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                        #endif
                        asm volatile(
                            "tcgen05.ld.red.sync.aligned.32x32b.x32.max.f32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];"
                            : "=f"(sv[0]), "=f"(sv[1]), "=f"(sv[2]), "=f"(sv[3]), "=f"(sv[4]), "=f"(sv[5]), "=f"(sv[6]), "=f"(sv[7]), "=f"(sv[8]), "=f"(sv[9]), "=f"(sv[10]), "=f"(sv[11]), "=f"(sv[12]), "=f"(sv[13]), "=f"(sv[14]), "=f"(sv[15]), "=f"(sv[16]), "=f"(sv[17]), "=f"(sv[18]), "=f"(sv[19]), "=f"(sv[20]), "=f"(sv[21]), "=f"(sv[22]), "=f"(sv[23]), "=f"(sv[24]), "=f"(sv[25]), "=f"(sv[26]), "=f"(sv[27]), "=f"(sv[28]), "=f"(sv[29]), "=f"(sv[30]), "=f"(sv[31]), "=f"(lmax_a)
                            : "r"(s_base));
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                        #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                        #endif
                        asm volatile(
                            "tcgen05.ld.red.sync.aligned.32x32b.x32.max.f32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];"
                            : "=f"(sv[32]), "=f"(sv[33]), "=f"(sv[34]), "=f"(sv[35]), "=f"(sv[36]), "=f"(sv[37]), "=f"(sv[38]), "=f"(sv[39]), "=f"(sv[40]), "=f"(sv[41]), "=f"(sv[42]), "=f"(sv[43]), "=f"(sv[44]), "=f"(sv[45]), "=f"(sv[46]), "=f"(sv[47]), "=f"(sv[48]), "=f"(sv[49]), "=f"(sv[50]), "=f"(sv[51]), "=f"(sv[52]), "=f"(sv[53]), "=f"(sv[54]), "=f"(sv[55]), "=f"(sv[56]), "=f"(sv[57]), "=f"(sv[58]), "=f"(sv[59]), "=f"(sv[60]), "=f"(sv[61]), "=f"(sv[62]), "=f"(sv[63]), "=f"(lmax_b)
                            : "r"(s_base + 32));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        float _max_0 = max_noftz(lmax_a, lmax_b);
                        local_max = _max_0;
                        if (local_tail < 64) {
                            uint32_t _slice_lo_mask_0;
                            {
                                int _lim_0 = local_tail;
                                if (_lim_0 <= 0) { _slice_lo_mask_0 = 0u; }
                                else if (_lim_0 >= 32) { _slice_lo_mask_0 = 0xFFFFFFFFu; }
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
                                        "}" : "=r"(_slice_lo_mask_0) : "r"(_lim_0));
                                }
                            }
                            if (!(_slice_lo_mask_0 & (1u << 0))) sv[0] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 1))) sv[1] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 2))) sv[2] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 3))) sv[3] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 4))) sv[4] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 5))) sv[5] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 6))) sv[6] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 7))) sv[7] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 8))) sv[8] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 9))) sv[9] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 10))) sv[10] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 11))) sv[11] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 12))) sv[12] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 13))) sv[13] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 14))) sv[14] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 15))) sv[15] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 16))) sv[16] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 17))) sv[17] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 18))) sv[18] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 19))) sv[19] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 20))) sv[20] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 21))) sv[21] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 22))) sv[22] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 23))) sv[23] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 24))) sv[24] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 25))) sv[25] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 26))) sv[26] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 27))) sv[27] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 28))) sv[28] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 29))) sv[29] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 30))) sv[30] = -CAKE_INF;
                            if (!(_slice_lo_mask_0 & (1u << 31))) sv[31] = -CAKE_INF;
                            uint32_t _slice_lo_mask_1;
                            {
                                int _lim_1 = local_tail - 32;
                                if (_lim_1 <= 0) { _slice_lo_mask_1 = 0u; }
                                else if (_lim_1 >= 32) { _slice_lo_mask_1 = 0xFFFFFFFFu; }
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
                                        "}" : "=r"(_slice_lo_mask_1) : "r"(_lim_1));
                                }
                            }
                            if (!(_slice_lo_mask_1 & (1u << 0))) sv[32] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 1))) sv[33] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 2))) sv[34] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 3))) sv[35] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 4))) sv[36] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 5))) sv[37] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 6))) sv[38] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 7))) sv[39] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 8))) sv[40] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 9))) sv[41] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 10))) sv[42] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 11))) sv[43] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 12))) sv[44] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 13))) sv[45] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 14))) sv[46] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 15))) sv[47] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 16))) sv[48] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 17))) sv[49] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 18))) sv[50] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 19))) sv[51] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 20))) sv[52] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 21))) sv[53] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 22))) sv[54] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 23))) sv[55] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 24))) sv[56] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 25))) sv[57] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 26))) sv[58] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 27))) sv[59] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 28))) sv[60] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 29))) sv[61] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 30))) sv[62] = -CAKE_INF;
                            if (!(_slice_lo_mask_1 & (1u << 31))) sv[63] = -CAKE_INF;
                            float _max3_0;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_0) : "f"(sv[0]), "f"(sv[1]), "f"(sv[2]));
                            float _max3_1;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_1) : "f"(sv[3]), "f"(sv[4]), "f"(sv[5]));
                            float _max3_2;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_2) : "f"(sv[6]), "f"(sv[7]), "f"(sv[8]));
                            float _max3_3;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_3) : "f"(sv[9]), "f"(sv[10]), "f"(sv[11]));
                            float _max3_4;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_4) : "f"(sv[12]), "f"(sv[13]), "f"(sv[14]));
                            float _max3_5;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_5) : "f"(sv[15]), "f"(sv[16]), "f"(sv[17]));
                            float _max3_6;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_6) : "f"(sv[18]), "f"(sv[19]), "f"(sv[20]));
                            float _max3_7;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_7) : "f"(sv[21]), "f"(sv[22]), "f"(sv[23]));
                            float _max3_8;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_8) : "f"(sv[24]), "f"(sv[25]), "f"(sv[26]));
                            float _max3_9;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_9) : "f"(sv[27]), "f"(sv[28]), "f"(sv[29]));
                            float _max3_10;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_10) : "f"(sv[30]), "f"(sv[31]), "f"(sv[32]));
                            float _max3_11;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_11) : "f"(sv[33]), "f"(sv[34]), "f"(sv[35]));
                            float _max3_12;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_12) : "f"(sv[36]), "f"(sv[37]), "f"(sv[38]));
                            float _max3_13;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_13) : "f"(sv[39]), "f"(sv[40]), "f"(sv[41]));
                            float _max3_14;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_14) : "f"(sv[42]), "f"(sv[43]), "f"(sv[44]));
                            float _max3_15;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_15) : "f"(sv[45]), "f"(sv[46]), "f"(sv[47]));
                            float _max3_16;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_16) : "f"(sv[48]), "f"(sv[49]), "f"(sv[50]));
                            float _max3_17;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_17) : "f"(sv[51]), "f"(sv[52]), "f"(sv[53]));
                            float _max3_18;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_18) : "f"(sv[54]), "f"(sv[55]), "f"(sv[56]));
                            float _max3_19;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_19) : "f"(sv[57]), "f"(sv[58]), "f"(sv[59]));
                            float _max3_20;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_20) : "f"(sv[60]), "f"(sv[61]), "f"(sv[62]));
                            float _max3_21;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_21) : "f"(_max3_0), "f"(_max3_1), "f"(_max3_2));
                            float _max3_22;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_22) : "f"(_max3_3), "f"(_max3_4), "f"(_max3_5));
                            float _max3_23;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_23) : "f"(_max3_6), "f"(_max3_7), "f"(_max3_8));
                            float _max3_24;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_24) : "f"(_max3_9), "f"(_max3_10), "f"(_max3_11));
                            float _max3_25;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_25) : "f"(_max3_12), "f"(_max3_13), "f"(_max3_14));
                            float _max3_26;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_26) : "f"(_max3_15), "f"(_max3_16), "f"(_max3_17));
                            float _max3_27;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_27) : "f"(_max3_18), "f"(_max3_19), "f"(_max3_20));
                            float _max3_28;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_28) : "f"(_max3_21), "f"(_max3_22), "f"(_max3_23));
                            float _max3_29;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_29) : "f"(_max3_24), "f"(_max3_25), "f"(_max3_26));
                            float _max_1 = max_noftz(_max3_27, sv[63]);
                            float _max3_30;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_30) : "f"(_max3_28), "f"(_max3_29), "f"(_max_1));
                            local_max = _max3_30;
                        }
                    } else {
                        mbarrier_wait_token_test(s_full_addr, s_wait_phase, _mbar_token_0);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                        #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                        #endif
                        asm volatile(
                            "tcgen05.ld.red.sync.aligned.32x32b.x32.max.f32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];"
                            : "=f"(sv[0]), "=f"(sv[1]), "=f"(sv[2]), "=f"(sv[3]), "=f"(sv[4]), "=f"(sv[5]), "=f"(sv[6]), "=f"(sv[7]), "=f"(sv[8]), "=f"(sv[9]), "=f"(sv[10]), "=f"(sv[11]), "=f"(sv[12]), "=f"(sv[13]), "=f"(sv[14]), "=f"(sv[15]), "=f"(sv[16]), "=f"(sv[17]), "=f"(sv[18]), "=f"(sv[19]), "=f"(sv[20]), "=f"(sv[21]), "=f"(sv[22]), "=f"(sv[23]), "=f"(sv[24]), "=f"(sv[25]), "=f"(sv[26]), "=f"(sv[27]), "=f"(sv[28]), "=f"(sv[29]), "=f"(sv[30]), "=f"(sv[31]), "=f"(lmax_a)
                            : "r"(s_base));
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                        #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                        #endif
                        asm volatile(
                            "tcgen05.ld.red.sync.aligned.32x32b.x32.max.f32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];"
                            : "=f"(sv[32]), "=f"(sv[33]), "=f"(sv[34]), "=f"(sv[35]), "=f"(sv[36]), "=f"(sv[37]), "=f"(sv[38]), "=f"(sv[39]), "=f"(sv[40]), "=f"(sv[41]), "=f"(sv[42]), "=f"(sv[43]), "=f"(sv[44]), "=f"(sv[45]), "=f"(sv[46]), "=f"(sv[47]), "=f"(sv[48]), "=f"(sv[49]), "=f"(sv[50]), "=f"(sv[51]), "=f"(sv[52]), "=f"(sv[53]), "=f"(sv[54]), "=f"(sv[55]), "=f"(sv[56]), "=f"(sv[57]), "=f"(sv[58]), "=f"(sv[59]), "=f"(sv[60]), "=f"(sv[61]), "=f"(sv[62]), "=f"(sv[63]), "=f"(lmax_b)
                            : "r"(s_base + 32));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        float _max_2 = max_noftz(lmax_a, lmax_b);
                        local_max = _max_2;
                    }
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                __syncwarp();
                if (elect_sync()) {
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(s_empty_addr), "r"(0) : "memory");
                }
                smem_stats_max[half * 64 + my_row] = local_max;
                float ref_prev = -CAKE_INF;
                float run_prev = -CAKE_INF;
                if (tile > 0) {
                    mbarrier_wait(ref_full_addr + 8, tile - 1 >> 1 & 1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    ref_prev = smem_stats_max[256 + my_row];
                    run_prev = smem_stats_run[my_row];
                }
                float partner_max = local_max;
                {
                    if (warp % 2 == 0) {
                        asm volatile("barrier.sync 1, 64;" ::: "memory");
                    } else {
                        asm volatile("barrier.sync 2, 64;" ::: "memory");
                    }
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    partner_max = smem_stats_max[(1 - half) * 64 + my_row];
                }
                float _max3_62;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_62) : "f"(local_max), "f"(partner_max), "f"(ref_prev));
                float cand_max = _max3_62;
                float _max_5 = max_noftz(cand_max, run_prev);
                float run_new = _max_5;
                float lazy_delta = softmax_scale_log2 * (ref_prev - run_new);
                int need_move = ((lazy_delta < -2.0f) ? 1 : 0);
                int _vote_1 = __any_sync(0xFFFFFFFF, need_move != 0);
                int any_need = _vote_1;
                int grew = ((run_new > ref_prev) ? 1 : 0);
                int do_move = any_need * grew;
                float ref_new = ((do_move != 0) ? run_new : ref_prev);
                {
                    if (half == 0) {
                        smem_stats_max[256 + my_row] = ref_new;
                        smem_stats_run[my_row] = run_new;
                    }
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(ref_full_addr);
                    }
                }
                float delta_mine = softmax_scale_log2 * (row_max_val - ref_new);
                float _exp2_0 = approx_exp2(delta_mine);
                float exp_delta_mine = _exp2_0;
                float acc_scale = ((row_max_val > -CAKE_INF) ? exp_delta_mine : 1.0f);
                row_max_val = ref_new;
                float o_delta = softmax_scale_log2 * (ref_prev - ref_new);
                float _exp2_1 = approx_exp2(o_delta);
                float o_scale = _exp2_1;
                int o_moved = ((ref_new > ref_prev) ? 1 : 0);
                int _vote_2 = __any_sync(0xFFFFFFFF, o_moved != 0);
                int any_rescale = _vote_2;
                float safe_max = ((ref_new == -CAKE_INF) ? 0.0f : ref_new);
                float max_scaled = safe_max * softmax_scale_log2 - 6.8073549220576f;
                {
                    #pragma unroll
                    for (int i = 0; i < 64; i++) {
                        float _exp2_2 = approx_exp2(sv[i] * softmax_scale_log2 - max_scaled);
                        sv[i] = _exp2_2;
                    }
                }
                float sum_c0 = sv[0];
                float sum_c1 = sv[1];
                float sum_c2 = sv[2];
                float sum_c3 = sv[3];
                float sum_c4 = sv[4];
                float sum_c5 = sv[5];
                float sum_c6 = sv[6];
                float sum_c7 = sv[7];
                #pragma unroll
                for (int i_1 = 8; i_1 < 64; i_1 += 8) {
                    sum_c0 = sum_c0 + sv[i_1];
                    sum_c1 = sum_c1 + sv[i_1 + 1];
                    sum_c2 = sum_c2 + sv[i_1 + 2];
                    sum_c3 = sum_c3 + sv[i_1 + 3];
                    sum_c4 = sum_c4 + sv[i_1 + 4];
                    sum_c5 = sum_c5 + sv[i_1 + 5];
                    sum_c6 = sum_c6 + sv[i_1 + 6];
                    sum_c7 = sum_c7 + sv[i_1 + 7];
                }
                float sum_c01 = sum_c0 + sum_c1;
                float sum_c23 = sum_c2 + sum_c3;
                float sum_c45 = sum_c4 + sum_c5;
                float sum_c67 = sum_c6 + sum_c7;
                float sum_c0123 = sum_c01 + sum_c23;
                float sum_c4567 = sum_c45 + sum_c67;
                float local_sum = sum_c0123 + sum_c4567;
                row_sum_val = row_sum_val * acc_scale + local_sum;
                unsigned int p_pack[16];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(sv[0]), "f"(sv[1]),
                                           "f"(sv[2]), "f"(sv[3]));
                    p_pack[0] = _packed;
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
                        : "=r"(_packed) : "f"(sv[4]), "f"(sv[5]),
                                           "f"(sv[6]), "f"(sv[7]));
                    p_pack[1] = _packed;
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
                        : "=r"(_packed) : "f"(sv[8]), "f"(sv[9]),
                                           "f"(sv[10]), "f"(sv[11]));
                    p_pack[2] = _packed;
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
                        : "=r"(_packed) : "f"(sv[12]), "f"(sv[13]),
                                           "f"(sv[14]), "f"(sv[15]));
                    p_pack[3] = _packed;
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
                        : "=r"(_packed) : "f"(sv[16]), "f"(sv[17]),
                                           "f"(sv[18]), "f"(sv[19]));
                    p_pack[4] = _packed;
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
                        : "=r"(_packed) : "f"(sv[20]), "f"(sv[21]),
                                           "f"(sv[22]), "f"(sv[23]));
                    p_pack[5] = _packed;
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
                        : "=r"(_packed) : "f"(sv[24]), "f"(sv[25]),
                                           "f"(sv[26]), "f"(sv[27]));
                    p_pack[6] = _packed;
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
                        : "=r"(_packed) : "f"(sv[28]), "f"(sv[29]),
                                           "f"(sv[30]), "f"(sv[31]));
                    p_pack[7] = _packed;
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
                        : "=r"(_packed) : "f"(sv[32]), "f"(sv[33]),
                                           "f"(sv[34]), "f"(sv[35]));
                    p_pack[8] = _packed;
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
                        : "=r"(_packed) : "f"(sv[36]), "f"(sv[37]),
                                           "f"(sv[38]), "f"(sv[39]));
                    p_pack[9] = _packed;
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
                        : "=r"(_packed) : "f"(sv[40]), "f"(sv[41]),
                                           "f"(sv[42]), "f"(sv[43]));
                    p_pack[10] = _packed;
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
                        : "=r"(_packed) : "f"(sv[44]), "f"(sv[45]),
                                           "f"(sv[46]), "f"(sv[47]));
                    p_pack[11] = _packed;
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
                        : "=r"(_packed) : "f"(sv[48]), "f"(sv[49]),
                                           "f"(sv[50]), "f"(sv[51]));
                    p_pack[12] = _packed;
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
                        : "=r"(_packed) : "f"(sv[52]), "f"(sv[53]),
                                           "f"(sv[54]), "f"(sv[55]));
                    p_pack[13] = _packed;
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
                        : "=r"(_packed) : "f"(sv[56]), "f"(sv[57]),
                                           "f"(sv[58]), "f"(sv[59]));
                    p_pack[14] = _packed;
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
                        : "=r"(_packed) : "f"(sv[60]), "f"(sv[61]),
                                           "f"(sv[62]), "f"(sv[63]));
                    p_pack[15] = _packed;
                }
                if (tile > 0 && !PP_KO_RESCALE) {
                    if (any_rescale != 0) {
                        mbarrier_wait(o_full_addr + 8, tile - 1 >> 1 & 1);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int o_rs_base = taddr + 256 + (unsigned int)corr_row;
                        #pragma unroll
                        for (int oc = 0; oc < 256; oc += 64) {
                            float _tmem_load_0[32];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                                : "r"(o_rs_base + oc));
                            float _tmem_load_1[32];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31])
                                : "r"(o_rs_base + oc + 32));
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            {
                                #pragma unroll
                                for (int j = 0; j < 32; j++) {
                                    _tmem_load_0[j] = _tmem_load_0[j] * o_scale;
                                    _tmem_load_1[j] = _tmem_load_1[j] * o_scale;
                                }
                            }
                            tmem_st_x32_f32(o_rs_base + oc, _tmem_load_0);
                            tmem_st_x32_f32(o_rs_base + oc + 32, _tmem_load_1);
                        }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                    }
                }
                int p_empty_phase = k & 1 ^ 1;
                mbarrier_wait(p_empty_addr, p_empty_phase);
                #pragma unroll
                for (int px_c = 0; px_c < 4; px_c++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_p_addr + (unsigned int)(my_row * 128 + (half * 64 + px_c * 16) ^ (my_row * 128 + (half * 64 + px_c * 16) >> 7 & 7) << 4))), "r"(p_pack[4 * px_c]), "r"(p_pack[4 * px_c + 1]), "r"(p_pack[4 * px_c + 2]), "r"(p_pack[4 * px_c + 3]) : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncwarp();
                if (elect_sync()) {
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(p_full_addr), "r"(0) : "memory");
                }
            }
            mbarrier_wait(o_done_addr, 0);
            smem_stats_sum[my_row * 4 + half] = row_sum_val;
            if (half == 0) {
                smem_stats_max[my_row] = row_max_val;
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(sum_ready_addr);
            unsigned int _phase_sum_ready_0 = 0;
            mbarrier_wait(sum_ready_addr, _phase_sum_ready_0);
            _phase_sum_ready_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float fs00 = smem_stats_sum[my_row * 4];
            float fs01 = smem_stats_sum[my_row * 4 + 1];
            float fs10 = smem_stats_sum[my_row * 4 + 2];
            float fs11 = smem_stats_sum[my_row * 4 + 3];
            float ref0 = smem_stats_max[my_row];
            float ref1 = smem_stats_max[128 + my_row];
            float _max_6 = max_noftz(ref0, ref1);
            float final_row_max = _max_6;
            float _exp2_3 = approx_exp2(softmax_scale_log2 * (ref0 - final_row_max));
            float w0_raw = _exp2_3;
            float _exp2_4 = approx_exp2(softmax_scale_log2 * (ref1 - final_row_max));
            float w1_raw = _exp2_4;
            float w0 = ((ref0 > -CAKE_INF) ? w0_raw : 0.0f);
            float w1 = ((ref1 > -CAKE_INF) ? w1_raw : 0.0f);
            float fs0 = fs00 + fs01;
            float fs1 = fs10 + fs11;
            float total_sum = fs0 * w0 + fs1 * w1;
            float safe_sum = ((total_sum > 0.0f) ? total_sum : 1.0f);
            float out_mul = ((num_split == 1) ? bmm2_scale : 1.0f);
            float _rcp_0 = approx_rcp(safe_sum);
            float output_scale = _rcp_0 * out_mul;
            int epi_base_w = warp * 512;
            int epi_wbase = warp * 2048 + lane * 64;
            int epi_wkey = lane >> 1 & 3;
            int epi_rq = lane >> 2;
            int epi_s = lane & 3;
            int row_stride = num_split * 512;
            int out_base = (row_base_global * num_split + split_idx) * 512 + n_half * 128;
            int lane_row = cta_rank * 64 + tmem_row_base + lane;
            int lane_out = out_base + lane_row * row_stride;
            #pragma unroll
            for (int vs_local = 1; vs_local < 2; vs_local++) {
                int vs = vs_local;
                int o_base_epi = taddr + 256 + (unsigned int)(vs * 128) + (unsigned int)corr_row;
                #pragma unroll
                for (int c2 = 0; c2 < 128; c2 += 128) {
                    float _tmem_load_2[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_2[0]), "=f"(_tmem_load_2[1]), "=f"(_tmem_load_2[2]), "=f"(_tmem_load_2[3]), "=f"(_tmem_load_2[4]), "=f"(_tmem_load_2[5]), "=f"(_tmem_load_2[6]), "=f"(_tmem_load_2[7]), "=f"(_tmem_load_2[8]), "=f"(_tmem_load_2[9]), "=f"(_tmem_load_2[10]), "=f"(_tmem_load_2[11]), "=f"(_tmem_load_2[12]), "=f"(_tmem_load_2[13]), "=f"(_tmem_load_2[14]), "=f"(_tmem_load_2[15]), "=f"(_tmem_load_2[16]), "=f"(_tmem_load_2[17]), "=f"(_tmem_load_2[18]), "=f"(_tmem_load_2[19]), "=f"(_tmem_load_2[20]), "=f"(_tmem_load_2[21]), "=f"(_tmem_load_2[22]), "=f"(_tmem_load_2[23]), "=f"(_tmem_load_2[24]), "=f"(_tmem_load_2[25]), "=f"(_tmem_load_2[26]), "=f"(_tmem_load_2[27]), "=f"(_tmem_load_2[28]), "=f"(_tmem_load_2[29]), "=f"(_tmem_load_2[30]), "=f"(_tmem_load_2[31])
                        : "r"(o_base_epi + c2));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_2[32]), "=f"(_tmem_load_2[33]), "=f"(_tmem_load_2[34]), "=f"(_tmem_load_2[35]), "=f"(_tmem_load_2[36]), "=f"(_tmem_load_2[37]), "=f"(_tmem_load_2[38]), "=f"(_tmem_load_2[39]), "=f"(_tmem_load_2[40]), "=f"(_tmem_load_2[41]), "=f"(_tmem_load_2[42]), "=f"(_tmem_load_2[43]), "=f"(_tmem_load_2[44]), "=f"(_tmem_load_2[45]), "=f"(_tmem_load_2[46]), "=f"(_tmem_load_2[47]), "=f"(_tmem_load_2[48]), "=f"(_tmem_load_2[49]), "=f"(_tmem_load_2[50]), "=f"(_tmem_load_2[51]), "=f"(_tmem_load_2[52]), "=f"(_tmem_load_2[53]), "=f"(_tmem_load_2[54]), "=f"(_tmem_load_2[55]), "=f"(_tmem_load_2[56]), "=f"(_tmem_load_2[57]), "=f"(_tmem_load_2[58]), "=f"(_tmem_load_2[59]), "=f"(_tmem_load_2[60]), "=f"(_tmem_load_2[61]), "=f"(_tmem_load_2[62]), "=f"(_tmem_load_2[63])
                        : "r"(o_base_epi + c2 + 32));
                    float _tmem_load_3[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_3[0]), "=f"(_tmem_load_3[1]), "=f"(_tmem_load_3[2]), "=f"(_tmem_load_3[3]), "=f"(_tmem_load_3[4]), "=f"(_tmem_load_3[5]), "=f"(_tmem_load_3[6]), "=f"(_tmem_load_3[7]), "=f"(_tmem_load_3[8]), "=f"(_tmem_load_3[9]), "=f"(_tmem_load_3[10]), "=f"(_tmem_load_3[11]), "=f"(_tmem_load_3[12]), "=f"(_tmem_load_3[13]), "=f"(_tmem_load_3[14]), "=f"(_tmem_load_3[15]), "=f"(_tmem_load_3[16]), "=f"(_tmem_load_3[17]), "=f"(_tmem_load_3[18]), "=f"(_tmem_load_3[19]), "=f"(_tmem_load_3[20]), "=f"(_tmem_load_3[21]), "=f"(_tmem_load_3[22]), "=f"(_tmem_load_3[23]), "=f"(_tmem_load_3[24]), "=f"(_tmem_load_3[25]), "=f"(_tmem_load_3[26]), "=f"(_tmem_load_3[27]), "=f"(_tmem_load_3[28]), "=f"(_tmem_load_3[29]), "=f"(_tmem_load_3[30]), "=f"(_tmem_load_3[31])
                        : "r"(o_base_epi + c2 + 64));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_3[32]), "=f"(_tmem_load_3[33]), "=f"(_tmem_load_3[34]), "=f"(_tmem_load_3[35]), "=f"(_tmem_load_3[36]), "=f"(_tmem_load_3[37]), "=f"(_tmem_load_3[38]), "=f"(_tmem_load_3[39]), "=f"(_tmem_load_3[40]), "=f"(_tmem_load_3[41]), "=f"(_tmem_load_3[42]), "=f"(_tmem_load_3[43]), "=f"(_tmem_load_3[44]), "=f"(_tmem_load_3[45]), "=f"(_tmem_load_3[46]), "=f"(_tmem_load_3[47]), "=f"(_tmem_load_3[48]), "=f"(_tmem_load_3[49]), "=f"(_tmem_load_3[50]), "=f"(_tmem_load_3[51]), "=f"(_tmem_load_3[52]), "=f"(_tmem_load_3[53]), "=f"(_tmem_load_3[54]), "=f"(_tmem_load_3[55]), "=f"(_tmem_load_3[56]), "=f"(_tmem_load_3[57]), "=f"(_tmem_load_3[58]), "=f"(_tmem_load_3[59]), "=f"(_tmem_load_3[60]), "=f"(_tmem_load_3[61]), "=f"(_tmem_load_3[62]), "=f"(_tmem_load_3[63])
                        : "r"(o_base_epi + c2 + 64 + 32));
                    {
                        unsigned int packed_epi[16];
                        const float2 _scale2_2 = {output_scale, output_scale};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>((_tmem_load_2 + 0))[_ls], _scale2_2);
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 0], _tmem_load_2[_lp*2+1 + 0]));
                            packed_epi[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int j_1 = 0; j_1 < 16; j_1 += 4) {
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_epi_addr + (unsigned int)(epi_wbase + (epi_wkey ^ j_1 / 4) * 16)), "r"(packed_epi[j_1]), "r"(packed_epi[j_1 + 1]), "r"(packed_epi[j_1 + 2]), "r"(packed_epi[j_1 + 3]) : "memory");
                        }
                        __syncwarp();
                        #pragma unroll
                        for (int i_2 = 0; i_2 < 4; i_2++) {
                            int epi_r = i_2 * 8 + epi_rq;
                            unsigned int _smem_epi_reg_0[4];
                            {
                                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(smem_epi);
                                #pragma unroll
                                for (int _lr = 0; _lr < 4; _lr++)
                                    _smem_epi_reg_0[_lr] = _smem_ptr[(epi_base_w + epi_r * 16 + (epi_s ^ epi_r >> 1 & 3) * 4) + _lr];
                            }
                            int epi_row = cta_rank * 64 + tmem_row_base + epi_r;
                            int epi_col = vs * 256 + c2 + epi_s * 8;
                            if (epi_row < rows_valid) {
                                reinterpret_cast<int4*>(partial_O + (out_base + epi_row * row_stride + epi_col))[0] = reinterpret_cast<int4*>(_smem_epi_reg_0)[0];
                            }
                        }
                        __syncwarp();
                        const float2 _scale2_3 = {output_scale, output_scale};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>((_tmem_load_2 + 32))[_ls], _scale2_3);
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_2[_lp*2 + 32], _tmem_load_2[_lp*2+1 + 32]));
                            packed_epi[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int j_2 = 0; j_2 < 16; j_2 += 4) {
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_epi_addr + (unsigned int)(epi_wbase + (epi_wkey ^ j_2 / 4) * 16)), "r"(packed_epi[j_2]), "r"(packed_epi[j_2 + 1]), "r"(packed_epi[j_2 + 2]), "r"(packed_epi[j_2 + 3]) : "memory");
                        }
                        __syncwarp();
                        #pragma unroll
                        for (int i_3 = 0; i_3 < 4; i_3++) {
                            int epi_r_1 = i_3 * 8 + epi_rq;
                            unsigned int _smem_epi_reg_1[4];
                            {
                                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(smem_epi);
                                #pragma unroll
                                for (int _lr = 0; _lr < 4; _lr++)
                                    _smem_epi_reg_1[_lr] = _smem_ptr[(epi_base_w + epi_r_1 * 16 + (epi_s ^ epi_r_1 >> 1 & 3) * 4) + _lr];
                            }
                            int epi_row_1 = cta_rank * 64 + tmem_row_base + epi_r_1;
                            int epi_col_1 = vs * 256 + c2 + 32 + epi_s * 8;
                            if (epi_row_1 < rows_valid) {
                                reinterpret_cast<int4*>(partial_O + (out_base + epi_row_1 * row_stride + epi_col_1))[0] = reinterpret_cast<int4*>(_smem_epi_reg_1)[0];
                            }
                        }
                        __syncwarp();
                    }
                    {
                        unsigned int packed_epi_1[16];
                        const float2 _scale2_4 = {output_scale, output_scale};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>((_tmem_load_3 + 0))[_ls], _scale2_4);
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 0], _tmem_load_3[_lp*2+1 + 0]));
                            packed_epi_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int j_3 = 0; j_3 < 16; j_3 += 4) {
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_epi_addr + (unsigned int)(epi_wbase + (epi_wkey ^ j_3 / 4) * 16)), "r"(packed_epi_1[j_3]), "r"(packed_epi_1[j_3 + 1]), "r"(packed_epi_1[j_3 + 2]), "r"(packed_epi_1[j_3 + 3]) : "memory");
                        }
                        __syncwarp();
                        #pragma unroll
                        for (int i_4 = 0; i_4 < 4; i_4++) {
                            int epi_r_2 = i_4 * 8 + epi_rq;
                            unsigned int _smem_epi_reg_2[4];
                            {
                                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(smem_epi);
                                #pragma unroll
                                for (int _lr = 0; _lr < 4; _lr++)
                                    _smem_epi_reg_2[_lr] = _smem_ptr[(epi_base_w + epi_r_2 * 16 + (epi_s ^ epi_r_2 >> 1 & 3) * 4) + _lr];
                            }
                            int epi_row_2 = cta_rank * 64 + tmem_row_base + epi_r_2;
                            int epi_col_2 = vs * 256 + (c2 + 64) + epi_s * 8;
                            if (epi_row_2 < rows_valid) {
                                reinterpret_cast<int4*>(partial_O + (out_base + epi_row_2 * row_stride + epi_col_2))[0] = reinterpret_cast<int4*>(_smem_epi_reg_2)[0];
                            }
                        }
                        __syncwarp();
                        const float2 _scale2_5 = {output_scale, output_scale};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>((_tmem_load_3 + 32))[_ls], _scale2_5);
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 32], _tmem_load_3[_lp*2+1 + 32]));
                            packed_epi_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int j_4 = 0; j_4 < 16; j_4 += 4) {
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_epi_addr + (unsigned int)(epi_wbase + (epi_wkey ^ j_4 / 4) * 16)), "r"(packed_epi_1[j_4]), "r"(packed_epi_1[j_4 + 1]), "r"(packed_epi_1[j_4 + 2]), "r"(packed_epi_1[j_4 + 3]) : "memory");
                        }
                        __syncwarp();
                        #pragma unroll
                        for (int i_5 = 0; i_5 < 4; i_5++) {
                            int epi_r_3 = i_5 * 8 + epi_rq;
                            unsigned int _smem_epi_reg_3[4];
                            {
                                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(smem_epi);
                                #pragma unroll
                                for (int _lr = 0; _lr < 4; _lr++)
                                    _smem_epi_reg_3[_lr] = _smem_ptr[(epi_base_w + epi_r_3 * 16 + (epi_s ^ epi_r_3 >> 1 & 3) * 4) + _lr];
                            }
                            int epi_row_3 = cta_rank * 64 + tmem_row_base + epi_r_3;
                            int epi_col_3 = vs * 256 + (c2 + 64) + 32 + epi_s * 8;
                            if (epi_row_3 < rows_valid) {
                                reinterpret_cast<int4*>(partial_O + (out_base + epi_row_3 * row_stride + epi_col_3))[0] = reinterpret_cast<int4*>(_smem_epi_reg_3)[0];
                            }
                        }
                        __syncwarp();
                    }
                }
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(partial_results_ready_addr);
            unsigned int _phase_partial_results_ready_0 = 0;
            mbarrier_wait(partial_results_ready_addr, _phase_partial_results_ready_0);
            _phase_partial_results_ready_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
        }
    }
    // ---- Role: correction_wg ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 192;");
        { // correction_wg_main
            const int wg_dummy_inc_1 = 0;
            const int tmem_row_base_1 = warp % 2 * 32;
            const int half_1 = warp % 4 / 2;
            const int o_tmem_row_base_1 = warp % 4 * 32;
            const int corr_row_1 = o_tmem_row_base_1 << 16;
            const int n_half_1 = half_1;
            const int my_row_1 = tmem_row_base_1 + lane;
            float scrub_zero[128];
            #pragma unroll
            for (int scrub_i = 0; scrub_i < 128; scrub_i++) {
                scrub_zero[scrub_i] = 0.0f;
            }
            const int scrub_row = o_tmem_row_base_1 + lane;
            #pragma unroll
            for (int scrub_col = 0; scrub_col < 512; scrub_col += 128) {
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x128.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63, %64, %65, %66, %67, %68, %69, %70, %71, %72, %73, %74, %75, %76, %77, %78, %79, %80, %81, %82, %83, %84, %85, %86, %87, %88, %89, %90, %91, %92, %93, %94, %95, %96, %97, %98, %99, %100, %101, %102, %103, %104, %105, %106, %107, %108, %109, %110, %111, %112, %113, %114, %115, %116, %117, %118, %119, %120, %121, %122, %123, %124, %125, %126, %127, %128};"
                    :: "r"(taddr + (unsigned int)scrub_col + (unsigned int)(scrub_row << 16)), "f"(scrub_zero[0]), "f"(scrub_zero[1]), "f"(scrub_zero[2]), "f"(scrub_zero[3]), "f"(scrub_zero[4]), "f"(scrub_zero[5]), "f"(scrub_zero[6]), "f"(scrub_zero[7]), "f"(scrub_zero[8]), "f"(scrub_zero[9]), "f"(scrub_zero[10]), "f"(scrub_zero[11]), "f"(scrub_zero[12]), "f"(scrub_zero[13]), "f"(scrub_zero[14]), "f"(scrub_zero[15]), "f"(scrub_zero[16]), "f"(scrub_zero[17]), "f"(scrub_zero[18]), "f"(scrub_zero[19]), "f"(scrub_zero[20]), "f"(scrub_zero[21]), "f"(scrub_zero[22]), "f"(scrub_zero[23]), "f"(scrub_zero[24]), "f"(scrub_zero[25]), "f"(scrub_zero[26]), "f"(scrub_zero[27]), "f"(scrub_zero[28]), "f"(scrub_zero[29]), "f"(scrub_zero[30]), "f"(scrub_zero[31]), "f"(scrub_zero[32]), "f"(scrub_zero[33]), "f"(scrub_zero[34]), "f"(scrub_zero[35]), "f"(scrub_zero[36]), "f"(scrub_zero[37]), "f"(scrub_zero[38]), "f"(scrub_zero[39]), "f"(scrub_zero[40]), "f"(scrub_zero[41]), "f"(scrub_zero[42]), "f"(scrub_zero[43]), "f"(scrub_zero[44]), "f"(scrub_zero[45]), "f"(scrub_zero[46]), "f"(scrub_zero[47]), "f"(scrub_zero[48]), "f"(scrub_zero[49]), "f"(scrub_zero[50]), "f"(scrub_zero[51]), "f"(scrub_zero[52]), "f"(scrub_zero[53]), "f"(scrub_zero[54]), "f"(scrub_zero[55]), "f"(scrub_zero[56]), "f"(scrub_zero[57]), "f"(scrub_zero[58]), "f"(scrub_zero[59]), "f"(scrub_zero[60]), "f"(scrub_zero[61]), "f"(scrub_zero[62]), "f"(scrub_zero[63]), "f"(scrub_zero[64]), "f"(scrub_zero[65]), "f"(scrub_zero[66]), "f"(scrub_zero[67]), "f"(scrub_zero[68]), "f"(scrub_zero[69]), "f"(scrub_zero[70]), "f"(scrub_zero[71]), "f"(scrub_zero[72]), "f"(scrub_zero[73]), "f"(scrub_zero[74]), "f"(scrub_zero[75]), "f"(scrub_zero[76]), "f"(scrub_zero[77]), "f"(scrub_zero[78]), "f"(scrub_zero[79]), "f"(scrub_zero[80]), "f"(scrub_zero[81]), "f"(scrub_zero[82]), "f"(scrub_zero[83]), "f"(scrub_zero[84]), "f"(scrub_zero[85]), "f"(scrub_zero[86]), "f"(scrub_zero[87]), "f"(scrub_zero[88]), "f"(scrub_zero[89]), "f"(scrub_zero[90]), "f"(scrub_zero[91]), "f"(scrub_zero[92]), "f"(scrub_zero[93]), "f"(scrub_zero[94]), "f"(scrub_zero[95]), "f"(scrub_zero[96]), "f"(scrub_zero[97]), "f"(scrub_zero[98]), "f"(scrub_zero[99]), "f"(scrub_zero[100]), "f"(scrub_zero[101]), "f"(scrub_zero[102]), "f"(scrub_zero[103]), "f"(scrub_zero[104]), "f"(scrub_zero[105]), "f"(scrub_zero[106]), "f"(scrub_zero[107]), "f"(scrub_zero[108]), "f"(scrub_zero[109]), "f"(scrub_zero[110]), "f"(scrub_zero[111]), "f"(scrub_zero[112]), "f"(scrub_zero[113]), "f"(scrub_zero[114]), "f"(scrub_zero[115]), "f"(scrub_zero[116]), "f"(scrub_zero[117]), "f"(scrub_zero[118]), "f"(scrub_zero[119]), "f"(scrub_zero[120]), "f"(scrub_zero[121]), "f"(scrub_zero[122]), "f"(scrub_zero[123]), "f"(scrub_zero[124]), "f"(scrub_zero[125]), "f"(scrub_zero[126]), "f"(scrub_zero[127]));
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(tmem_epoch_ready_addr);
            asm volatile(
                "{\n\t"
                ".reg .b32 remAddr32;\n\t"
                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                "}"
                :: "r"(tmem_epoch_ready_addr), "r"(cta_rank ^ 1) : "memory");
            int split_idx_1 = blockIdx.x / 2;
            int m_tile_1 = gridDim.y - 1 - blockIdx.y;
            int b_1 = blockIdx.z;
            int q_start_1 = cum_seq_lens_q[b_1];
            int q_len_b_1 = cum_seq_lens_q[b_1 + 1] - q_start_1;
            int kv_len_1 = seq_lens[b_1];
            int rows_b_1 = q_len_b_1 * num_heads;
            int row0_1 = m_tile_1 * 128;
            int rows_left_1 = rows_b_1 - row0_1;
            int rows_pos_1 = ((rows_left_1 < 0) ? 0 : rows_left_1);
            int rows_valid_1 = ((rows_pos_1 > 128) ? 128 : rows_pos_1);
            int row_base_global_1 = q_start_1 * num_heads + row0_1;
            int last_row_1 = row0_1 + rows_valid_1 - 1;
            int t_last_1 = last_row_1 / num_heads;
            int kv_end_raw_1 = kv_len_1 - q_len_b_1 + t_last_1 + 1;
            int kv_end_1 = ((rows_valid_1 == 0) ? 0 : kv_end_raw_1);
            int n_pages_1 = (kv_len_1 + 64 - 1) / 64;
            int n_tiles_total_1 = (kv_end_1 + 128 - 1) / 128;
            int tiles_per_split_1 = (n_tiles_total_1 + num_split - 1) / num_split;
            int my_start_raw_1 = split_idx_1 * tiles_per_split_1;
            int my_end_raw_1 = my_start_raw_1 + tiles_per_split_1;
            int my_end_1 = ((my_end_raw_1 > n_tiles_total_1) ? n_tiles_total_1 : my_end_raw_1);
            int my_n_raw_1 = my_end_1 - my_start_raw_1;
            int empty_1 = ((my_n_raw_1 < 1) ? 1 : 0);
            int my_n_tiles_1 = ((my_n_raw_1 < 1) ? 1 : my_n_raw_1);
            int my_start_block_1 = ((my_n_raw_1 < 1) ? 0 : my_start_raw_1);
            int pt_base_1 = b_1 * max_pages_per_seq;
            int r_local_1 = cta_rank * 64 + my_row_1;
            int row_tok_raw_1 = (row0_1 + r_local_1) / num_heads;
            int q_last_1 = q_len_b_1 - 1;
            int row_tok_1 = ((row_tok_raw_1 > q_last_1) ? q_last_1 : row_tok_raw_1);
            int row_kv_end_raw_1 = kv_len_1 - q_len_b_1 + row_tok_1 + 1;
            int row_kv_end_1 = ((empty_1 != 0) ? 0 : row_kv_end_raw_1);
            int out_row_1 = row_base_global_1 + r_local_1;
            unsigned int _phase_q_full_0_1 = 0;
            mbarrier_wait(q_full_addr, _phase_q_full_0_1);
            _phase_q_full_0_1 ^= 1;
            float row_max_val_1 = -CAKE_INF;
            float row_sum_val_1 = 0.0f;
            int n_my_tiles_1 = (my_n_tiles_1 - 1 + 1) / 2;
            #pragma unroll 1
            for (int k_1 = 0; k_1 < n_my_tiles_1; k_1++) {
                int tile_1 = 2 * k_1 + 1;
                int s_wait_phase_1 = k_1 & 1;
                float sv_1[64];
                float local_max_1 = -CAKE_INF;
                int s_base_1 = taddr + 128 + (unsigned int)(o_tmem_row_base_1 << 16);
                int abs_tile_1 = my_start_block_1 + tile_1;
                int tail_valid_1 = row_kv_end_1 - abs_tile_1 * 128;
                int local_tail_raw_1 = tail_valid_1 - half_1 * 64;
                int local_tail_hi_1 = ((local_tail_raw_1 > 64) ? 64 : local_tail_raw_1);
                int local_tail_1 = ((local_tail_hi_1 < 0) ? 0 : local_tail_hi_1);
                {
                    uint32_t _mbar_token_2 = mbarrier_test_wait(s_full_addr + 8, s_wait_phase_1);
                    int _vote_3 = __any_sync(0xFFFFFFFF, local_tail_1 < 64);
                    int tail_any_1 = _vote_3;
                    float lmax_a_1 = -CAKE_INF;
                    float lmax_b_1 = -CAKE_INF;
                    if (tail_any_1 != 0) {
                        mbarrier_wait_token(s_full_addr + 8, s_wait_phase_1, _mbar_token_2);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                        #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                        #endif
                        asm volatile(
                            "tcgen05.ld.red.sync.aligned.32x32b.x32.max.f32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];"
                            : "=f"(sv_1[0]), "=f"(sv_1[1]), "=f"(sv_1[2]), "=f"(sv_1[3]), "=f"(sv_1[4]), "=f"(sv_1[5]), "=f"(sv_1[6]), "=f"(sv_1[7]), "=f"(sv_1[8]), "=f"(sv_1[9]), "=f"(sv_1[10]), "=f"(sv_1[11]), "=f"(sv_1[12]), "=f"(sv_1[13]), "=f"(sv_1[14]), "=f"(sv_1[15]), "=f"(sv_1[16]), "=f"(sv_1[17]), "=f"(sv_1[18]), "=f"(sv_1[19]), "=f"(sv_1[20]), "=f"(sv_1[21]), "=f"(sv_1[22]), "=f"(sv_1[23]), "=f"(sv_1[24]), "=f"(sv_1[25]), "=f"(sv_1[26]), "=f"(sv_1[27]), "=f"(sv_1[28]), "=f"(sv_1[29]), "=f"(sv_1[30]), "=f"(sv_1[31]), "=f"(lmax_a_1)
                            : "r"(s_base_1));
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                        #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                        #endif
                        asm volatile(
                            "tcgen05.ld.red.sync.aligned.32x32b.x32.max.f32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];"
                            : "=f"(sv_1[32]), "=f"(sv_1[33]), "=f"(sv_1[34]), "=f"(sv_1[35]), "=f"(sv_1[36]), "=f"(sv_1[37]), "=f"(sv_1[38]), "=f"(sv_1[39]), "=f"(sv_1[40]), "=f"(sv_1[41]), "=f"(sv_1[42]), "=f"(sv_1[43]), "=f"(sv_1[44]), "=f"(sv_1[45]), "=f"(sv_1[46]), "=f"(sv_1[47]), "=f"(sv_1[48]), "=f"(sv_1[49]), "=f"(sv_1[50]), "=f"(sv_1[51]), "=f"(sv_1[52]), "=f"(sv_1[53]), "=f"(sv_1[54]), "=f"(sv_1[55]), "=f"(sv_1[56]), "=f"(sv_1[57]), "=f"(sv_1[58]), "=f"(sv_1[59]), "=f"(sv_1[60]), "=f"(sv_1[61]), "=f"(sv_1[62]), "=f"(sv_1[63]), "=f"(lmax_b_1)
                            : "r"(s_base_1 + 32));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        float _max_7 = max_noftz(lmax_a_1, lmax_b_1);
                        local_max_1 = _max_7;
                        if (local_tail_1 < 64) {
                            uint32_t _slice_lo_mask_6;
                            {
                                int _lim_0 = local_tail_1;
                                if (_lim_0 <= 0) { _slice_lo_mask_6 = 0u; }
                                else if (_lim_0 >= 32) { _slice_lo_mask_6 = 0xFFFFFFFFu; }
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
                                        "}" : "=r"(_slice_lo_mask_6) : "r"(_lim_0));
                                }
                            }
                            if (!(_slice_lo_mask_6 & (1u << 0))) sv_1[0] = -CAKE_INF;
                            if (!(_slice_lo_mask_6 & (1u << 1))) sv_1[1] = -CAKE_INF;
                            if (!(_slice_lo_mask_6 & (1u << 2))) sv_1[2] = -CAKE_INF;
                            if (!(_slice_lo_mask_6 & (1u << 3))) sv_1[3] = -CAKE_INF;
                            if (!(_slice_lo_mask_6 & (1u << 4))) sv_1[4] = -CAKE_INF;
                            if (!(_slice_lo_mask_6 & (1u << 5))) sv_1[5] = -CAKE_INF;
                            if (!(_slice_lo_mask_6 & (1u << 6))) sv_1[6] = -CAKE_INF;
                            if (!(_slice_lo_mask_6 & (1u << 7))) sv_1[7] = -CAKE_INF;
                            if (!(_slice_lo_mask_6 & (1u << 8))) sv_1[8] = -CAKE_INF;
                            if (!(_slice_lo_mask_6 & (1u << 9))) sv_1[9] = -CAKE_INF;
                            if (!(_slice_lo_mask_6 & (1u << 10))) sv_1[10] = -CAKE_INF;
                            if (!(_slice_lo_mask_6 & (1u << 11))) sv_1[11] = -CAKE_INF;
                            if (!(_slice_lo_mask_6 & (1u << 12))) sv_1[12] = -CAKE_INF;
                            if (!(_slice_lo_mask_6 & (1u << 13))) sv_1[13] = -CAKE_INF;
                            if (!(_slice_lo_mask_6 & (1u << 14))) sv_1[14] = -CAKE_INF;
                            if (!(_slice_lo_mask_6 & (1u << 15))) sv_1[15] = -CAKE_INF;
                            if (!(_slice_lo_mask_6 & (1u << 16))) sv_1[16] = -CAKE_INF;
                            if (!(_slice_lo_mask_6 & (1u << 17))) sv_1[17] = -CAKE_INF;
                            if (!(_slice_lo_mask_6 & (1u << 18))) sv_1[18] = -CAKE_INF;
                            if (!(_slice_lo_mask_6 & (1u << 19))) sv_1[19] = -CAKE_INF;
                            if (!(_slice_lo_mask_6 & (1u << 20))) sv_1[20] = -CAKE_INF;
                            if (!(_slice_lo_mask_6 & (1u << 21))) sv_1[21] = -CAKE_INF;
                            if (!(_slice_lo_mask_6 & (1u << 22))) sv_1[22] = -CAKE_INF;
                            if (!(_slice_lo_mask_6 & (1u << 23))) sv_1[23] = -CAKE_INF;
                            if (!(_slice_lo_mask_6 & (1u << 24))) sv_1[24] = -CAKE_INF;
                            if (!(_slice_lo_mask_6 & (1u << 25))) sv_1[25] = -CAKE_INF;
                            if (!(_slice_lo_mask_6 & (1u << 26))) sv_1[26] = -CAKE_INF;
                            if (!(_slice_lo_mask_6 & (1u << 27))) sv_1[27] = -CAKE_INF;
                            if (!(_slice_lo_mask_6 & (1u << 28))) sv_1[28] = -CAKE_INF;
                            if (!(_slice_lo_mask_6 & (1u << 29))) sv_1[29] = -CAKE_INF;
                            if (!(_slice_lo_mask_6 & (1u << 30))) sv_1[30] = -CAKE_INF;
                            if (!(_slice_lo_mask_6 & (1u << 31))) sv_1[31] = -CAKE_INF;
                            uint32_t _slice_lo_mask_7;
                            {
                                int _lim_1 = local_tail_1 - 32;
                                if (_lim_1 <= 0) { _slice_lo_mask_7 = 0u; }
                                else if (_lim_1 >= 32) { _slice_lo_mask_7 = 0xFFFFFFFFu; }
                                else {
                                    asm volatile("{"
                                        ".reg .u32 t;\n\t"
                                        "shl.b32 t, 1, %1;\n\t"
                                        "add.u32 %0, t, -1;\n\t"
                                        "}" : "=r"(_slice_lo_mask_7) : "r"(_lim_1));
                                }
                            }
                            if (!(_slice_lo_mask_7 & (1u << 0))) sv_1[32] = -CAKE_INF;
                            if (!(_slice_lo_mask_7 & (1u << 1))) sv_1[33] = -CAKE_INF;
                            if (!(_slice_lo_mask_7 & (1u << 2))) sv_1[34] = -CAKE_INF;
                            if (!(_slice_lo_mask_7 & (1u << 3))) sv_1[35] = -CAKE_INF;
                            if (!(_slice_lo_mask_7 & (1u << 4))) sv_1[36] = -CAKE_INF;
                            if (!(_slice_lo_mask_7 & (1u << 5))) sv_1[37] = -CAKE_INF;
                            if (!(_slice_lo_mask_7 & (1u << 6))) sv_1[38] = -CAKE_INF;
                            if (!(_slice_lo_mask_7 & (1u << 7))) sv_1[39] = -CAKE_INF;
                            if (!(_slice_lo_mask_7 & (1u << 8))) sv_1[40] = -CAKE_INF;
                            if (!(_slice_lo_mask_7 & (1u << 9))) sv_1[41] = -CAKE_INF;
                            if (!(_slice_lo_mask_7 & (1u << 10))) sv_1[42] = -CAKE_INF;
                            if (!(_slice_lo_mask_7 & (1u << 11))) sv_1[43] = -CAKE_INF;
                            if (!(_slice_lo_mask_7 & (1u << 12))) sv_1[44] = -CAKE_INF;
                            if (!(_slice_lo_mask_7 & (1u << 13))) sv_1[45] = -CAKE_INF;
                            if (!(_slice_lo_mask_7 & (1u << 14))) sv_1[46] = -CAKE_INF;
                            if (!(_slice_lo_mask_7 & (1u << 15))) sv_1[47] = -CAKE_INF;
                            if (!(_slice_lo_mask_7 & (1u << 16))) sv_1[48] = -CAKE_INF;
                            if (!(_slice_lo_mask_7 & (1u << 17))) sv_1[49] = -CAKE_INF;
                            if (!(_slice_lo_mask_7 & (1u << 18))) sv_1[50] = -CAKE_INF;
                            if (!(_slice_lo_mask_7 & (1u << 19))) sv_1[51] = -CAKE_INF;
                            if (!(_slice_lo_mask_7 & (1u << 20))) sv_1[52] = -CAKE_INF;
                            if (!(_slice_lo_mask_7 & (1u << 21))) sv_1[53] = -CAKE_INF;
                            if (!(_slice_lo_mask_7 & (1u << 22))) sv_1[54] = -CAKE_INF;
                            if (!(_slice_lo_mask_7 & (1u << 23))) sv_1[55] = -CAKE_INF;
                            if (!(_slice_lo_mask_7 & (1u << 24))) sv_1[56] = -CAKE_INF;
                            if (!(_slice_lo_mask_7 & (1u << 25))) sv_1[57] = -CAKE_INF;
                            if (!(_slice_lo_mask_7 & (1u << 26))) sv_1[58] = -CAKE_INF;
                            if (!(_slice_lo_mask_7 & (1u << 27))) sv_1[59] = -CAKE_INF;
                            if (!(_slice_lo_mask_7 & (1u << 28))) sv_1[60] = -CAKE_INF;
                            if (!(_slice_lo_mask_7 & (1u << 29))) sv_1[61] = -CAKE_INF;
                            if (!(_slice_lo_mask_7 & (1u << 30))) sv_1[62] = -CAKE_INF;
                            if (!(_slice_lo_mask_7 & (1u << 31))) sv_1[63] = -CAKE_INF;
                            float _max3_63;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_63) : "f"(sv_1[0]), "f"(sv_1[1]), "f"(sv_1[2]));
                            float _max3_64;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_64) : "f"(sv_1[3]), "f"(sv_1[4]), "f"(sv_1[5]));
                            float _max3_65;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_65) : "f"(sv_1[6]), "f"(sv_1[7]), "f"(sv_1[8]));
                            float _max3_66;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_66) : "f"(sv_1[9]), "f"(sv_1[10]), "f"(sv_1[11]));
                            float _max3_67;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_67) : "f"(sv_1[12]), "f"(sv_1[13]), "f"(sv_1[14]));
                            float _max3_68;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_68) : "f"(sv_1[15]), "f"(sv_1[16]), "f"(sv_1[17]));
                            float _max3_69;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_69) : "f"(sv_1[18]), "f"(sv_1[19]), "f"(sv_1[20]));
                            float _max3_70;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_70) : "f"(sv_1[21]), "f"(sv_1[22]), "f"(sv_1[23]));
                            float _max3_71;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_71) : "f"(sv_1[24]), "f"(sv_1[25]), "f"(sv_1[26]));
                            float _max3_72;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_72) : "f"(sv_1[27]), "f"(sv_1[28]), "f"(sv_1[29]));
                            float _max3_73;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_73) : "f"(sv_1[30]), "f"(sv_1[31]), "f"(sv_1[32]));
                            float _max3_74;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_74) : "f"(sv_1[33]), "f"(sv_1[34]), "f"(sv_1[35]));
                            float _max3_75;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_75) : "f"(sv_1[36]), "f"(sv_1[37]), "f"(sv_1[38]));
                            float _max3_76;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_76) : "f"(sv_1[39]), "f"(sv_1[40]), "f"(sv_1[41]));
                            float _max3_77;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_77) : "f"(sv_1[42]), "f"(sv_1[43]), "f"(sv_1[44]));
                            float _max3_78;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_78) : "f"(sv_1[45]), "f"(sv_1[46]), "f"(sv_1[47]));
                            float _max3_79;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_79) : "f"(sv_1[48]), "f"(sv_1[49]), "f"(sv_1[50]));
                            float _max3_80;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_80) : "f"(sv_1[51]), "f"(sv_1[52]), "f"(sv_1[53]));
                            float _max3_81;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_81) : "f"(sv_1[54]), "f"(sv_1[55]), "f"(sv_1[56]));
                            float _max3_82;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_82) : "f"(sv_1[57]), "f"(sv_1[58]), "f"(sv_1[59]));
                            float _max3_83;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_83) : "f"(sv_1[60]), "f"(sv_1[61]), "f"(sv_1[62]));
                            float _max3_84;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_84) : "f"(_max3_63), "f"(_max3_64), "f"(_max3_65));
                            float _max3_85;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_85) : "f"(_max3_66), "f"(_max3_67), "f"(_max3_68));
                            float _max3_86;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_86) : "f"(_max3_69), "f"(_max3_70), "f"(_max3_71));
                            float _max3_87;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_87) : "f"(_max3_72), "f"(_max3_73), "f"(_max3_74));
                            float _max3_88;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_88) : "f"(_max3_75), "f"(_max3_76), "f"(_max3_77));
                            float _max3_89;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_89) : "f"(_max3_78), "f"(_max3_79), "f"(_max3_80));
                            float _max3_90;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_90) : "f"(_max3_81), "f"(_max3_82), "f"(_max3_83));
                            float _max3_91;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_91) : "f"(_max3_84), "f"(_max3_85), "f"(_max3_86));
                            float _max3_92;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_92) : "f"(_max3_87), "f"(_max3_88), "f"(_max3_89));
                            float _max_8 = max_noftz(_max3_90, sv_1[63]);
                            float _max3_93;
                            #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                            #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                            #endif
                            asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_93) : "f"(_max3_91), "f"(_max3_92), "f"(_max_8));
                            local_max_1 = _max3_93;
                        }
                    } else {
                        mbarrier_wait_token_test(s_full_addr + 8, s_wait_phase_1, _mbar_token_2);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                        #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                        #endif
                        asm volatile(
                            "tcgen05.ld.red.sync.aligned.32x32b.x32.max.f32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];"
                            : "=f"(sv_1[0]), "=f"(sv_1[1]), "=f"(sv_1[2]), "=f"(sv_1[3]), "=f"(sv_1[4]), "=f"(sv_1[5]), "=f"(sv_1[6]), "=f"(sv_1[7]), "=f"(sv_1[8]), "=f"(sv_1[9]), "=f"(sv_1[10]), "=f"(sv_1[11]), "=f"(sv_1[12]), "=f"(sv_1[13]), "=f"(sv_1[14]), "=f"(sv_1[15]), "=f"(sv_1[16]), "=f"(sv_1[17]), "=f"(sv_1[18]), "=f"(sv_1[19]), "=f"(sv_1[20]), "=f"(sv_1[21]), "=f"(sv_1[22]), "=f"(sv_1[23]), "=f"(sv_1[24]), "=f"(sv_1[25]), "=f"(sv_1[26]), "=f"(sv_1[27]), "=f"(sv_1[28]), "=f"(sv_1[29]), "=f"(sv_1[30]), "=f"(sv_1[31]), "=f"(lmax_a_1)
                            : "r"(s_base_1));
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                        #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                        #endif
                        asm volatile(
                            "tcgen05.ld.red.sync.aligned.32x32b.x32.max.f32"
                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];"
                            : "=f"(sv_1[32]), "=f"(sv_1[33]), "=f"(sv_1[34]), "=f"(sv_1[35]), "=f"(sv_1[36]), "=f"(sv_1[37]), "=f"(sv_1[38]), "=f"(sv_1[39]), "=f"(sv_1[40]), "=f"(sv_1[41]), "=f"(sv_1[42]), "=f"(sv_1[43]), "=f"(sv_1[44]), "=f"(sv_1[45]), "=f"(sv_1[46]), "=f"(sv_1[47]), "=f"(sv_1[48]), "=f"(sv_1[49]), "=f"(sv_1[50]), "=f"(sv_1[51]), "=f"(sv_1[52]), "=f"(sv_1[53]), "=f"(sv_1[54]), "=f"(sv_1[55]), "=f"(sv_1[56]), "=f"(sv_1[57]), "=f"(sv_1[58]), "=f"(sv_1[59]), "=f"(sv_1[60]), "=f"(sv_1[61]), "=f"(sv_1[62]), "=f"(sv_1[63]), "=f"(lmax_b_1)
                            : "r"(s_base_1 + 32));
                        asm volatile("tcgen05.wait::ld.sync.aligned;");
                        float _max_9 = max_noftz(lmax_a_1, lmax_b_1);
                        local_max_1 = _max_9;
                    }
                }
                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                __syncwarp();
                if (elect_sync()) {
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(s_empty_addr + 8), "r"(0) : "memory");
                }
                smem_stats_max[128 + half_1 * 64 + my_row_1] = local_max_1;
                float ref_prev_1 = -CAKE_INF;
                float run_prev_1 = -CAKE_INF;
                if (tile_1 > 0) {
                    mbarrier_wait(ref_full_addr, tile_1 - 1 >> 1 & 1);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    ref_prev_1 = smem_stats_max[256 + my_row_1];
                    run_prev_1 = smem_stats_run[my_row_1];
                }
                float partner_max_1 = local_max_1;
                {
                    if (warp % 2 == 0) {
                        asm volatile("barrier.sync 4, 64;" ::: "memory");
                    } else {
                        asm volatile("barrier.sync 5, 64;" ::: "memory");
                    }
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    partner_max_1 = smem_stats_max[128 + (1 - half_1) * 64 + my_row_1];
                }
                float _max3_125;
                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                #error "Max3 requires PTX three-input max.f32 support on sm_100+"
                #endif
                asm volatile("max.f32 %0, %1, %2, %3;" : "=f"(_max3_125) : "f"(local_max_1), "f"(partner_max_1), "f"(ref_prev_1));
                float cand_max_1 = _max3_125;
                float _max_12 = max_noftz(cand_max_1, run_prev_1);
                float run_new_1 = _max_12;
                float lazy_delta_1 = softmax_scale_log2 * (ref_prev_1 - run_new_1);
                int need_move_1 = ((lazy_delta_1 < -2.0f) ? 1 : 0);
                int _vote_4 = __any_sync(0xFFFFFFFF, need_move_1 != 0);
                int any_need_1 = _vote_4;
                int grew_1 = ((run_new_1 > ref_prev_1) ? 1 : 0);
                int do_move_1 = any_need_1 * grew_1;
                float ref_new_1 = ((do_move_1 != 0) ? run_new_1 : ref_prev_1);
                {
                    if (half_1 == 0) {
                        smem_stats_max[256 + my_row_1] = ref_new_1;
                        smem_stats_run[my_row_1] = run_new_1;
                    }
                    __syncwarp();
                    if (elect_sync()) {
                        mbarrier_arrive(ref_full_addr + 8);
                    }
                }
                float delta_mine_1 = softmax_scale_log2 * (row_max_val_1 - ref_new_1);
                float _exp2_5 = approx_exp2(delta_mine_1);
                float exp_delta_mine_1 = _exp2_5;
                float acc_scale_1 = ((row_max_val_1 > -CAKE_INF) ? exp_delta_mine_1 : 1.0f);
                row_max_val_1 = ref_new_1;
                float o_delta_1 = softmax_scale_log2 * (ref_prev_1 - ref_new_1);
                float _exp2_6 = approx_exp2(o_delta_1);
                float o_scale_1 = _exp2_6;
                int o_moved_1 = ((ref_new_1 > ref_prev_1) ? 1 : 0);
                int _vote_5 = __any_sync(0xFFFFFFFF, o_moved_1 != 0);
                int any_rescale_1 = _vote_5;
                float safe_max_1 = ((ref_new_1 == -CAKE_INF) ? 0.0f : ref_new_1);
                float max_scaled_1 = safe_max_1 * softmax_scale_log2 - 6.8073549220576f;
                {
                    #pragma unroll
                    for (int i_6 = 0; i_6 < 64; i_6++) {
                        float _exp2_7 = approx_exp2(sv_1[i_6] * softmax_scale_log2 - max_scaled_1);
                        sv_1[i_6] = _exp2_7;
                    }
                }
                float sum_c0_1 = sv_1[0];
                float sum_c1_1 = sv_1[1];
                float sum_c2_1 = sv_1[2];
                float sum_c3_1 = sv_1[3];
                float sum_c4_1 = sv_1[4];
                float sum_c5_1 = sv_1[5];
                float sum_c6_1 = sv_1[6];
                float sum_c7_1 = sv_1[7];
                #pragma unroll
                for (int i_7 = 8; i_7 < 64; i_7 += 8) {
                    sum_c0_1 = sum_c0_1 + sv_1[i_7];
                    sum_c1_1 = sum_c1_1 + sv_1[i_7 + 1];
                    sum_c2_1 = sum_c2_1 + sv_1[i_7 + 2];
                    sum_c3_1 = sum_c3_1 + sv_1[i_7 + 3];
                    sum_c4_1 = sum_c4_1 + sv_1[i_7 + 4];
                    sum_c5_1 = sum_c5_1 + sv_1[i_7 + 5];
                    sum_c6_1 = sum_c6_1 + sv_1[i_7 + 6];
                    sum_c7_1 = sum_c7_1 + sv_1[i_7 + 7];
                }
                float sum_c01_1 = sum_c0_1 + sum_c1_1;
                float sum_c23_1 = sum_c2_1 + sum_c3_1;
                float sum_c45_1 = sum_c4_1 + sum_c5_1;
                float sum_c67_1 = sum_c6_1 + sum_c7_1;
                float sum_c0123_1 = sum_c01_1 + sum_c23_1;
                float sum_c4567_1 = sum_c45_1 + sum_c67_1;
                float local_sum_1 = sum_c0123_1 + sum_c4567_1;
                row_sum_val_1 = row_sum_val_1 * acc_scale_1 + local_sum_1;
                unsigned int p_pack_1[16];
                {
                    uint32_t _packed;
                    asm volatile("{\n\t"
                        ".reg .b16 _lo;\n\t"
                        ".reg .b16 _hi;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                        "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                        "mov.b32 %0, {_lo, _hi};\n\t"
                        "}"
                        : "=r"(_packed) : "f"(sv_1[0]), "f"(sv_1[1]),
                                           "f"(sv_1[2]), "f"(sv_1[3]));
                    p_pack_1[0] = _packed;
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
                        : "=r"(_packed) : "f"(sv_1[4]), "f"(sv_1[5]),
                                           "f"(sv_1[6]), "f"(sv_1[7]));
                    p_pack_1[1] = _packed;
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
                        : "=r"(_packed) : "f"(sv_1[8]), "f"(sv_1[9]),
                                           "f"(sv_1[10]), "f"(sv_1[11]));
                    p_pack_1[2] = _packed;
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
                        : "=r"(_packed) : "f"(sv_1[12]), "f"(sv_1[13]),
                                           "f"(sv_1[14]), "f"(sv_1[15]));
                    p_pack_1[3] = _packed;
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
                        : "=r"(_packed) : "f"(sv_1[16]), "f"(sv_1[17]),
                                           "f"(sv_1[18]), "f"(sv_1[19]));
                    p_pack_1[4] = _packed;
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
                        : "=r"(_packed) : "f"(sv_1[20]), "f"(sv_1[21]),
                                           "f"(sv_1[22]), "f"(sv_1[23]));
                    p_pack_1[5] = _packed;
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
                        : "=r"(_packed) : "f"(sv_1[24]), "f"(sv_1[25]),
                                           "f"(sv_1[26]), "f"(sv_1[27]));
                    p_pack_1[6] = _packed;
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
                        : "=r"(_packed) : "f"(sv_1[28]), "f"(sv_1[29]),
                                           "f"(sv_1[30]), "f"(sv_1[31]));
                    p_pack_1[7] = _packed;
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
                        : "=r"(_packed) : "f"(sv_1[32]), "f"(sv_1[33]),
                                           "f"(sv_1[34]), "f"(sv_1[35]));
                    p_pack_1[8] = _packed;
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
                        : "=r"(_packed) : "f"(sv_1[36]), "f"(sv_1[37]),
                                           "f"(sv_1[38]), "f"(sv_1[39]));
                    p_pack_1[9] = _packed;
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
                        : "=r"(_packed) : "f"(sv_1[40]), "f"(sv_1[41]),
                                           "f"(sv_1[42]), "f"(sv_1[43]));
                    p_pack_1[10] = _packed;
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
                        : "=r"(_packed) : "f"(sv_1[44]), "f"(sv_1[45]),
                                           "f"(sv_1[46]), "f"(sv_1[47]));
                    p_pack_1[11] = _packed;
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
                        : "=r"(_packed) : "f"(sv_1[48]), "f"(sv_1[49]),
                                           "f"(sv_1[50]), "f"(sv_1[51]));
                    p_pack_1[12] = _packed;
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
                        : "=r"(_packed) : "f"(sv_1[52]), "f"(sv_1[53]),
                                           "f"(sv_1[54]), "f"(sv_1[55]));
                    p_pack_1[13] = _packed;
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
                        : "=r"(_packed) : "f"(sv_1[56]), "f"(sv_1[57]),
                                           "f"(sv_1[58]), "f"(sv_1[59]));
                    p_pack_1[14] = _packed;
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
                        : "=r"(_packed) : "f"(sv_1[60]), "f"(sv_1[61]),
                                           "f"(sv_1[62]), "f"(sv_1[63]));
                    p_pack_1[15] = _packed;
                }
                if (tile_1 > 0 && !PP_KO_RESCALE) {
                    if (any_rescale_1 != 0) {
                        mbarrier_wait(o_full_addr, tile_1 - 1 >> 1 & 1);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int o_rs_base_1 = taddr + 256 + (unsigned int)corr_row_1;
                        #pragma unroll
                        for (int oc_1 = 0; oc_1 < 256; oc_1 += 64) {
                            float _tmem_load_4[32];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(_tmem_load_4[0]), "=f"(_tmem_load_4[1]), "=f"(_tmem_load_4[2]), "=f"(_tmem_load_4[3]), "=f"(_tmem_load_4[4]), "=f"(_tmem_load_4[5]), "=f"(_tmem_load_4[6]), "=f"(_tmem_load_4[7]), "=f"(_tmem_load_4[8]), "=f"(_tmem_load_4[9]), "=f"(_tmem_load_4[10]), "=f"(_tmem_load_4[11]), "=f"(_tmem_load_4[12]), "=f"(_tmem_load_4[13]), "=f"(_tmem_load_4[14]), "=f"(_tmem_load_4[15]), "=f"(_tmem_load_4[16]), "=f"(_tmem_load_4[17]), "=f"(_tmem_load_4[18]), "=f"(_tmem_load_4[19]), "=f"(_tmem_load_4[20]), "=f"(_tmem_load_4[21]), "=f"(_tmem_load_4[22]), "=f"(_tmem_load_4[23]), "=f"(_tmem_load_4[24]), "=f"(_tmem_load_4[25]), "=f"(_tmem_load_4[26]), "=f"(_tmem_load_4[27]), "=f"(_tmem_load_4[28]), "=f"(_tmem_load_4[29]), "=f"(_tmem_load_4[30]), "=f"(_tmem_load_4[31])
                                : "r"(o_rs_base_1 + oc_1));
                            float _tmem_load_5[32];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                : "=f"(_tmem_load_5[0]), "=f"(_tmem_load_5[1]), "=f"(_tmem_load_5[2]), "=f"(_tmem_load_5[3]), "=f"(_tmem_load_5[4]), "=f"(_tmem_load_5[5]), "=f"(_tmem_load_5[6]), "=f"(_tmem_load_5[7]), "=f"(_tmem_load_5[8]), "=f"(_tmem_load_5[9]), "=f"(_tmem_load_5[10]), "=f"(_tmem_load_5[11]), "=f"(_tmem_load_5[12]), "=f"(_tmem_load_5[13]), "=f"(_tmem_load_5[14]), "=f"(_tmem_load_5[15]), "=f"(_tmem_load_5[16]), "=f"(_tmem_load_5[17]), "=f"(_tmem_load_5[18]), "=f"(_tmem_load_5[19]), "=f"(_tmem_load_5[20]), "=f"(_tmem_load_5[21]), "=f"(_tmem_load_5[22]), "=f"(_tmem_load_5[23]), "=f"(_tmem_load_5[24]), "=f"(_tmem_load_5[25]), "=f"(_tmem_load_5[26]), "=f"(_tmem_load_5[27]), "=f"(_tmem_load_5[28]), "=f"(_tmem_load_5[29]), "=f"(_tmem_load_5[30]), "=f"(_tmem_load_5[31])
                                : "r"(o_rs_base_1 + oc_1 + 32));
                            asm volatile("tcgen05.wait::ld.sync.aligned;");
                            {
                                #pragma unroll
                                for (int j_5 = 0; j_5 < 32; j_5++) {
                                    _tmem_load_4[j_5] = _tmem_load_4[j_5] * o_scale_1;
                                    _tmem_load_5[j_5] = _tmem_load_5[j_5] * o_scale_1;
                                }
                            }
                            tmem_st_x32_f32(o_rs_base_1 + oc_1, _tmem_load_4);
                            tmem_st_x32_f32(o_rs_base_1 + oc_1 + 32, _tmem_load_5);
                        }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                    }
                }
                int p_empty_phase_1 = k_1 & 1 ^ 1;
                mbarrier_wait(p_empty_addr + 8, p_empty_phase_1);
                #pragma unroll
                for (int px_c_1 = 0; px_c_1 < 4; px_c_1++) {
                    asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_p_addr + 8192 + (unsigned int)(my_row_1 * 128 + (half_1 * 64 + px_c_1 * 16) ^ (my_row_1 * 128 + (half_1 * 64 + px_c_1 * 16) >> 7 & 7) << 4))), "r"(p_pack_1[4 * px_c_1]), "r"(p_pack_1[4 * px_c_1 + 1]), "r"(p_pack_1[4 * px_c_1 + 2]), "r"(p_pack_1[4 * px_c_1 + 3]) : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                __syncwarp();
                if (elect_sync()) {
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(p_full_addr + 8), "r"(0) : "memory");
                }
            }
            mbarrier_wait(o_done_addr, 0);
            smem_stats_sum[my_row_1 * 4 + 2 + half_1] = row_sum_val_1;
            if (half_1 == 0) {
                smem_stats_max[128 + my_row_1] = row_max_val_1;
            }
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            mbarrier_arrive(sum_ready_addr);
            unsigned int _phase_sum_ready_0_1 = 0;
            mbarrier_wait(sum_ready_addr, _phase_sum_ready_0_1);
            _phase_sum_ready_0_1 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            float fs00_1 = smem_stats_sum[my_row_1 * 4];
            float fs01_1 = smem_stats_sum[my_row_1 * 4 + 1];
            float fs10_1 = smem_stats_sum[my_row_1 * 4 + 2];
            float fs11_1 = smem_stats_sum[my_row_1 * 4 + 3];
            float ref0_1 = smem_stats_max[my_row_1];
            float ref1_1 = smem_stats_max[128 + my_row_1];
            float _max_13 = max_noftz(ref0_1, ref1_1);
            float final_row_max_1 = _max_13;
            float _exp2_8 = approx_exp2(softmax_scale_log2 * (ref0_1 - final_row_max_1));
            float w0_raw_1 = _exp2_8;
            float _exp2_9 = approx_exp2(softmax_scale_log2 * (ref1_1 - final_row_max_1));
            float w1_raw_1 = _exp2_9;
            float w0_1 = ((ref0_1 > -CAKE_INF) ? w0_raw_1 : 0.0f);
            float w1_1 = ((ref1_1 > -CAKE_INF) ? w1_raw_1 : 0.0f);
            float fs0_1 = fs00_1 + fs01_1;
            float fs1_1 = fs10_1 + fs11_1;
            float total_sum_1 = fs0_1 * w0_1 + fs1_1 * w1_1;
            float safe_sum_1 = ((total_sum_1 > 0.0f) ? total_sum_1 : 1.0f);
            float out_mul_1 = ((num_split == 1) ? bmm2_scale : 1.0f);
            float _rcp_1 = approx_rcp(safe_sum_1);
            float output_scale_1 = _rcp_1 * out_mul_1;
            if (n_half_1 == 0) {
                if (r_local_1 < rows_valid_1) {
                    int stat_off = out_row_1 * num_split + split_idx_1;
                    float row_max_scaled = final_row_max_1 * softmax_scale_log2;
                    float stored_max = ((final_row_max_1 == -CAKE_INF) ? -CAKE_INF : row_max_scaled);
                    *(reinterpret_cast<float*>(partial_max + stat_off) + (0)) = stored_max;
                    *(reinterpret_cast<float*>(partial_sum + stat_off) + (0)) = total_sum_1;
                }
            }
            int epi_base_w_1 = warp * 512;
            int epi_wbase_1 = warp * 2048 + lane * 64;
            int epi_wkey_1 = lane >> 1 & 3;
            int epi_rq_1 = lane >> 2;
            int epi_s_1 = lane & 3;
            int row_stride_1 = num_split * 512;
            int out_base_1 = (row_base_global_1 * num_split + split_idx_1) * 512 + n_half_1 * 128;
            int lane_row_1 = cta_rank * 64 + tmem_row_base_1 + lane;
            int lane_out_1 = out_base_1 + lane_row_1 * row_stride_1;
            #pragma unroll
            for (int vs_local_1 = 0; vs_local_1 < 1; vs_local_1++) {
                int vs_1 = vs_local_1;
                int o_base_epi_1 = taddr + 256 + (unsigned int)(vs_1 * 128) + (unsigned int)corr_row_1;
                #pragma unroll
                for (int c2_1 = 0; c2_1 < 128; c2_1 += 128) {
                    float _tmem_load_6[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_6[0]), "=f"(_tmem_load_6[1]), "=f"(_tmem_load_6[2]), "=f"(_tmem_load_6[3]), "=f"(_tmem_load_6[4]), "=f"(_tmem_load_6[5]), "=f"(_tmem_load_6[6]), "=f"(_tmem_load_6[7]), "=f"(_tmem_load_6[8]), "=f"(_tmem_load_6[9]), "=f"(_tmem_load_6[10]), "=f"(_tmem_load_6[11]), "=f"(_tmem_load_6[12]), "=f"(_tmem_load_6[13]), "=f"(_tmem_load_6[14]), "=f"(_tmem_load_6[15]), "=f"(_tmem_load_6[16]), "=f"(_tmem_load_6[17]), "=f"(_tmem_load_6[18]), "=f"(_tmem_load_6[19]), "=f"(_tmem_load_6[20]), "=f"(_tmem_load_6[21]), "=f"(_tmem_load_6[22]), "=f"(_tmem_load_6[23]), "=f"(_tmem_load_6[24]), "=f"(_tmem_load_6[25]), "=f"(_tmem_load_6[26]), "=f"(_tmem_load_6[27]), "=f"(_tmem_load_6[28]), "=f"(_tmem_load_6[29]), "=f"(_tmem_load_6[30]), "=f"(_tmem_load_6[31])
                        : "r"(o_base_epi_1 + c2_1));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_6[32]), "=f"(_tmem_load_6[33]), "=f"(_tmem_load_6[34]), "=f"(_tmem_load_6[35]), "=f"(_tmem_load_6[36]), "=f"(_tmem_load_6[37]), "=f"(_tmem_load_6[38]), "=f"(_tmem_load_6[39]), "=f"(_tmem_load_6[40]), "=f"(_tmem_load_6[41]), "=f"(_tmem_load_6[42]), "=f"(_tmem_load_6[43]), "=f"(_tmem_load_6[44]), "=f"(_tmem_load_6[45]), "=f"(_tmem_load_6[46]), "=f"(_tmem_load_6[47]), "=f"(_tmem_load_6[48]), "=f"(_tmem_load_6[49]), "=f"(_tmem_load_6[50]), "=f"(_tmem_load_6[51]), "=f"(_tmem_load_6[52]), "=f"(_tmem_load_6[53]), "=f"(_tmem_load_6[54]), "=f"(_tmem_load_6[55]), "=f"(_tmem_load_6[56]), "=f"(_tmem_load_6[57]), "=f"(_tmem_load_6[58]), "=f"(_tmem_load_6[59]), "=f"(_tmem_load_6[60]), "=f"(_tmem_load_6[61]), "=f"(_tmem_load_6[62]), "=f"(_tmem_load_6[63])
                        : "r"(o_base_epi_1 + c2_1 + 32));
                    float _tmem_load_7[64];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_7[0]), "=f"(_tmem_load_7[1]), "=f"(_tmem_load_7[2]), "=f"(_tmem_load_7[3]), "=f"(_tmem_load_7[4]), "=f"(_tmem_load_7[5]), "=f"(_tmem_load_7[6]), "=f"(_tmem_load_7[7]), "=f"(_tmem_load_7[8]), "=f"(_tmem_load_7[9]), "=f"(_tmem_load_7[10]), "=f"(_tmem_load_7[11]), "=f"(_tmem_load_7[12]), "=f"(_tmem_load_7[13]), "=f"(_tmem_load_7[14]), "=f"(_tmem_load_7[15]), "=f"(_tmem_load_7[16]), "=f"(_tmem_load_7[17]), "=f"(_tmem_load_7[18]), "=f"(_tmem_load_7[19]), "=f"(_tmem_load_7[20]), "=f"(_tmem_load_7[21]), "=f"(_tmem_load_7[22]), "=f"(_tmem_load_7[23]), "=f"(_tmem_load_7[24]), "=f"(_tmem_load_7[25]), "=f"(_tmem_load_7[26]), "=f"(_tmem_load_7[27]), "=f"(_tmem_load_7[28]), "=f"(_tmem_load_7[29]), "=f"(_tmem_load_7[30]), "=f"(_tmem_load_7[31])
                        : "r"(o_base_epi_1 + c2_1 + 64));
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_7[32]), "=f"(_tmem_load_7[33]), "=f"(_tmem_load_7[34]), "=f"(_tmem_load_7[35]), "=f"(_tmem_load_7[36]), "=f"(_tmem_load_7[37]), "=f"(_tmem_load_7[38]), "=f"(_tmem_load_7[39]), "=f"(_tmem_load_7[40]), "=f"(_tmem_load_7[41]), "=f"(_tmem_load_7[42]), "=f"(_tmem_load_7[43]), "=f"(_tmem_load_7[44]), "=f"(_tmem_load_7[45]), "=f"(_tmem_load_7[46]), "=f"(_tmem_load_7[47]), "=f"(_tmem_load_7[48]), "=f"(_tmem_load_7[49]), "=f"(_tmem_load_7[50]), "=f"(_tmem_load_7[51]), "=f"(_tmem_load_7[52]), "=f"(_tmem_load_7[53]), "=f"(_tmem_load_7[54]), "=f"(_tmem_load_7[55]), "=f"(_tmem_load_7[56]), "=f"(_tmem_load_7[57]), "=f"(_tmem_load_7[58]), "=f"(_tmem_load_7[59]), "=f"(_tmem_load_7[60]), "=f"(_tmem_load_7[61]), "=f"(_tmem_load_7[62]), "=f"(_tmem_load_7[63])
                        : "r"(o_base_epi_1 + c2_1 + 64 + 32));
                    {
                        unsigned int packed_epi_2[16];
                        const float2 _scale2_2 = {output_scale_1, output_scale_1};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>((_tmem_load_6 + 0))[_ls], _scale2_2);
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_6[_lp*2 + 0], _tmem_load_6[_lp*2+1 + 0]));
                            packed_epi_2[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int j_6 = 0; j_6 < 16; j_6 += 4) {
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_epi_addr + (unsigned int)(epi_wbase_1 + (epi_wkey_1 ^ j_6 / 4) * 16)), "r"(packed_epi_2[j_6]), "r"(packed_epi_2[j_6 + 1]), "r"(packed_epi_2[j_6 + 2]), "r"(packed_epi_2[j_6 + 3]) : "memory");
                        }
                        __syncwarp();
                        #pragma unroll
                        for (int i_8 = 0; i_8 < 4; i_8++) {
                            int epi_r_4 = i_8 * 8 + epi_rq_1;
                            unsigned int _smem_epi_reg_4[4];
                            {
                                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(smem_epi);
                                #pragma unroll
                                for (int _lr = 0; _lr < 4; _lr++)
                                    _smem_epi_reg_4[_lr] = _smem_ptr[(epi_base_w_1 + epi_r_4 * 16 + (epi_s_1 ^ epi_r_4 >> 1 & 3) * 4) + _lr];
                            }
                            int epi_row_4 = cta_rank * 64 + tmem_row_base_1 + epi_r_4;
                            int epi_col_4 = vs_1 * 256 + c2_1 + epi_s_1 * 8;
                            if (epi_row_4 < rows_valid_1) {
                                reinterpret_cast<int4*>(partial_O + (out_base_1 + epi_row_4 * row_stride_1 + epi_col_4))[0] = reinterpret_cast<int4*>(_smem_epi_reg_4)[0];
                            }
                        }
                        __syncwarp();
                        const float2 _scale2_3 = {output_scale_1, output_scale_1};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>((_tmem_load_6 + 32))[_ls], _scale2_3);
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_6[_lp*2 + 32], _tmem_load_6[_lp*2+1 + 32]));
                            packed_epi_2[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int j_7 = 0; j_7 < 16; j_7 += 4) {
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_epi_addr + (unsigned int)(epi_wbase_1 + (epi_wkey_1 ^ j_7 / 4) * 16)), "r"(packed_epi_2[j_7]), "r"(packed_epi_2[j_7 + 1]), "r"(packed_epi_2[j_7 + 2]), "r"(packed_epi_2[j_7 + 3]) : "memory");
                        }
                        __syncwarp();
                        #pragma unroll
                        for (int i_9 = 0; i_9 < 4; i_9++) {
                            int epi_r_5 = i_9 * 8 + epi_rq_1;
                            unsigned int _smem_epi_reg_5[4];
                            {
                                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(smem_epi);
                                #pragma unroll
                                for (int _lr = 0; _lr < 4; _lr++)
                                    _smem_epi_reg_5[_lr] = _smem_ptr[(epi_base_w_1 + epi_r_5 * 16 + (epi_s_1 ^ epi_r_5 >> 1 & 3) * 4) + _lr];
                            }
                            int epi_row_5 = cta_rank * 64 + tmem_row_base_1 + epi_r_5;
                            int epi_col_5 = vs_1 * 256 + c2_1 + 32 + epi_s_1 * 8;
                            if (epi_row_5 < rows_valid_1) {
                                reinterpret_cast<int4*>(partial_O + (out_base_1 + epi_row_5 * row_stride_1 + epi_col_5))[0] = reinterpret_cast<int4*>(_smem_epi_reg_5)[0];
                            }
                        }
                        __syncwarp();
                    }
                    {
                        unsigned int packed_epi_3[16];
                        const float2 _scale2_4 = {output_scale_1, output_scale_1};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>((_tmem_load_7 + 0))[_ls], _scale2_4);
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_7[_lp*2 + 0], _tmem_load_7[_lp*2+1 + 0]));
                            packed_epi_3[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int j_8 = 0; j_8 < 16; j_8 += 4) {
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_epi_addr + (unsigned int)(epi_wbase_1 + (epi_wkey_1 ^ j_8 / 4) * 16)), "r"(packed_epi_3[j_8]), "r"(packed_epi_3[j_8 + 1]), "r"(packed_epi_3[j_8 + 2]), "r"(packed_epi_3[j_8 + 3]) : "memory");
                        }
                        __syncwarp();
                        #pragma unroll
                        for (int i_10 = 0; i_10 < 4; i_10++) {
                            int epi_r_6 = i_10 * 8 + epi_rq_1;
                            unsigned int _smem_epi_reg_6[4];
                            {
                                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(smem_epi);
                                #pragma unroll
                                for (int _lr = 0; _lr < 4; _lr++)
                                    _smem_epi_reg_6[_lr] = _smem_ptr[(epi_base_w_1 + epi_r_6 * 16 + (epi_s_1 ^ epi_r_6 >> 1 & 3) * 4) + _lr];
                            }
                            int epi_row_6 = cta_rank * 64 + tmem_row_base_1 + epi_r_6;
                            int epi_col_6 = vs_1 * 256 + (c2_1 + 64) + epi_s_1 * 8;
                            if (epi_row_6 < rows_valid_1) {
                                reinterpret_cast<int4*>(partial_O + (out_base_1 + epi_row_6 * row_stride_1 + epi_col_6))[0] = reinterpret_cast<int4*>(_smem_epi_reg_6)[0];
                            }
                        }
                        __syncwarp();
                        const float2 _scale2_5 = {output_scale_1, output_scale_1};
                        #pragma unroll
                        for (int _ls = 0; _ls < 16; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>((_tmem_load_7 + 32))[_ls], _scale2_5);
                        #pragma unroll
                        for (int _lp = 0; _lp < 16; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_7[_lp*2 + 32], _tmem_load_7[_lp*2+1 + 32]));
                            packed_epi_3[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int j_9 = 0; j_9 < 16; j_9 += 4) {
                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"(smem_epi_addr + (unsigned int)(epi_wbase_1 + (epi_wkey_1 ^ j_9 / 4) * 16)), "r"(packed_epi_3[j_9]), "r"(packed_epi_3[j_9 + 1]), "r"(packed_epi_3[j_9 + 2]), "r"(packed_epi_3[j_9 + 3]) : "memory");
                        }
                        __syncwarp();
                        #pragma unroll
                        for (int i_11 = 0; i_11 < 4; i_11++) {
                            int epi_r_7 = i_11 * 8 + epi_rq_1;
                            unsigned int _smem_epi_reg_7[4];
                            {
                                const unsigned int* _smem_ptr = reinterpret_cast<const unsigned int*>(smem_epi);
                                #pragma unroll
                                for (int _lr = 0; _lr < 4; _lr++)
                                    _smem_epi_reg_7[_lr] = _smem_ptr[(epi_base_w_1 + epi_r_7 * 16 + (epi_s_1 ^ epi_r_7 >> 1 & 3) * 4) + _lr];
                            }
                            int epi_row_7 = cta_rank * 64 + tmem_row_base_1 + epi_r_7;
                            int epi_col_7 = vs_1 * 256 + (c2_1 + 64) + 32 + epi_s_1 * 8;
                            if (epi_row_7 < rows_valid_1) {
                                reinterpret_cast<int4*>(partial_O + (out_base_1 + epi_row_7 * row_stride_1 + epi_col_7))[0] = reinterpret_cast<int4*>(_smem_epi_reg_7)[0];
                            }
                        }
                        __syncwarp();
                    }
                }
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            __syncwarp();
            if (elect_sync()) {
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(o_empty_addr), "r"(0) : "memory");
            }
            __syncwarp();
            if (elect_sync()) {
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(o_empty_addr + 8), "r"(0) : "memory");
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(partial_results_ready_addr);
            unsigned int _phase_partial_results_ready_0_1 = 0;
            mbarrier_wait(partial_results_ready_addr, _phase_partial_results_ready_0_1);
            _phase_partial_results_ready_0_1 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            asm volatile("barrier.sync 13, 160;" ::: "memory");
            if (warp == 4) {
                int tmem_dealloc_peer_rank = cta_rank ^ 1;
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(tmem_dealloc_peer_addr), "r"(tmem_dealloc_peer_rank) : "memory");
                mbarrier_wait(tmem_dealloc_peer_addr, 0);
                int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
                asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(512));
            }
        }
    }
    // ---- Role: mma_warp ----
    if (warp == 8) {
        { // mma_warp_main
            const int wg3_dummy = 0;
            unsigned int mma_k_stage = 0;
            unsigned int mma_v_stage = 0;
            int split_idx_2 = blockIdx.x / 2;
            int m_tile_2 = gridDim.y - 1 - blockIdx.y;
            int b_2 = blockIdx.z;
            int q_start_2 = cum_seq_lens_q[b_2];
            int q_len_b_2 = cum_seq_lens_q[b_2 + 1] - q_start_2;
            int kv_len_2 = seq_lens[b_2];
            int rows_b_2 = q_len_b_2 * num_heads;
            int row0_2 = m_tile_2 * 128;
            int rows_left_2 = rows_b_2 - row0_2;
            int rows_pos_2 = ((rows_left_2 < 0) ? 0 : rows_left_2);
            int rows_valid_2 = ((rows_pos_2 > 128) ? 128 : rows_pos_2);
            int row_base_global_2 = q_start_2 * num_heads + row0_2;
            int last_row_2 = row0_2 + rows_valid_2 - 1;
            int t_last_2 = last_row_2 / num_heads;
            int kv_end_raw_2 = kv_len_2 - q_len_b_2 + t_last_2 + 1;
            int kv_end_2 = ((rows_valid_2 == 0) ? 0 : kv_end_raw_2);
            int n_pages_2 = (kv_len_2 + 64 - 1) / 64;
            int n_tiles_total_2 = (kv_end_2 + 128 - 1) / 128;
            int tiles_per_split_2 = (n_tiles_total_2 + num_split - 1) / num_split;
            int my_start_raw_2 = split_idx_2 * tiles_per_split_2;
            int my_end_raw_2 = my_start_raw_2 + tiles_per_split_2;
            int my_end_2 = ((my_end_raw_2 > n_tiles_total_2) ? n_tiles_total_2 : my_end_raw_2);
            int my_n_raw_2 = my_end_2 - my_start_raw_2;
            int empty_2 = ((my_n_raw_2 < 1) ? 1 : 0);
            int my_n_tiles_2 = ((my_n_raw_2 < 1) ? 1 : my_n_raw_2);
            int my_start_block_2 = ((my_n_raw_2 < 1) ? 0 : my_start_raw_2);
            int pt_base_2 = b_2 * max_pages_per_seq;
            unsigned int _phase_tmem_epoch_ready_0 = 0;
            mbarrier_wait(tmem_epoch_ready_addr, _phase_tmem_epoch_ready_0);
            _phase_tmem_epoch_ready_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            unsigned int _phase_q_full_0_2 = 0;
            mbarrier_wait(q_full_addr, _phase_q_full_0_2);
            _phase_q_full_0_2 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            asm volatile(
                "{\n\t"
                ".reg .b32 remAddr32;\n\t"
                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                "mbarrier.arrive.release.cluster.shared::cluster.b64 _, [remAddr32];\n\t"
                "}"
                :: "r"(q_pair_ready_addr), "r"(0) : "memory");
            unsigned int _phase_q_pair_ready_0 = 0;
            if (cta_rank == 0) {
                mbarrier_wait_cluster_hint(q_pair_ready_addr, _phase_q_pair_ready_0, 10000000);
                _phase_q_pair_ready_0 ^= 1;
                asm volatile("tcgen05.fence::after_thread_sync;");
            }
            int first_pv = 1;
            unsigned int _phase_k_full = 0;
            unsigned int _phase_v_full = 0;
            #pragma unroll 1
            for (int tile_2 = 0; tile_2 < my_n_tiles_2 + 1; tile_2++) {
                int phase = tile_2 & 1;
                int s_empty_phase = tile_2 >> 1 & 1 ^ 1;
                if (my_n_tiles_2 > tile_2) {
                    int score_col = ((phase != 0) ? 128 : 0);
                    if (cta_rank == 0) {
                        {
                            mbarrier_wait(s_empty_addr + (phase) * 8, s_empty_phase);
                        }
                    }
                    if (cta_rank == 0) {
                        mbarrier_wait(k_full_addr + (mma_k_stage) * 8, _phase_k_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        {
                            int _mma_a_lo_0 = (((smem_q_addr) >> 4) & 0x3FFF) + (0) * 2048;
                            int _mma_b_lo_0 = (((smem_k_addr) >> 4) & 0x3FFF) + (mma_k_stage) * 2304;
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
                    "mov.b32 id, 136314896;\n\t"
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
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_tmem_scratch + (score_col))), "r"(0));
                        }
                        {
                            int _mma_a_lo_2 = (((smem_q_addr + 8192) >> 4) & 0x3FFF) + (0) * 2048;
                            int _mma_b_lo_2 = (((smem_k_addr + 8192) >> 4) & 0x3FFF) + (mma_k_stage) * 2304;
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
                    "mov.b32 id, 136314896;\n\t"
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
                    :: "r"(_mma_a_lo_2), "r"(_mma_b_lo_2), "r"((tmem_tmem_scratch + (score_col))), "r"(1));
                        }
                        {
                            int _mma_a_lo_4 = (((smem_q_addr + 16384) >> 4) & 0x3FFF) + (0) * 2048;
                            int _mma_b_lo_4 = (((smem_k_addr + 16384) >> 4) & 0x3FFF) + (mma_k_stage) * 2304;
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
                    "mov.b32 id, 136314896;\n\t"
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
                    :: "r"(_mma_a_lo_4), "r"(_mma_b_lo_4), "r"((tmem_tmem_scratch + (score_col))), "r"(1));
                        }
                        {
                            int _mma_a_lo_6 = (((smem_q_addr + 24576) >> 4) & 0x3FFF) + (0) * 2048;
                            int _mma_b_lo_6 = (((smem_k_addr + 24576) >> 4) & 0x3FFF) + (mma_k_stage) * 2304;
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
                    "mov.b32 id, 136314896;\n\t"
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
                    :: "r"(_mma_a_lo_6), "r"(_mma_b_lo_6), "r"((tmem_tmem_scratch + (score_col))), "r"(1));
                        }
                        int _mma_a_lo_8 = ((smem_qr_addr) >> 4) & 0x3FFF;
                        int _mma_b_lo_8 = (((smem_kr_addr) >> 4) & 0x3FFF) + (mma_k_stage) * 2304;
                        asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\tmov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
                    "mov.b32 adhi, 0x80004020;\n\t"
                    "mov.b32 bdhi, 0x80004020;\n\t"
                    "mov.b32 id, 136314896;\n\t"
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
                    "}\n"
                    :: "r"(_mma_a_lo_8), "r"(_mma_b_lo_8), "r"((tmem_tmem_scratch + (score_col))), "r"(1));
                        elect_commit_cg2_multicast(k_empty_addr + (mma_k_stage) * 8, (uint16_t)(3));
                    }
                    mma_k_stage += 1;
                    if (mma_k_stage == 3) { mma_k_stage = 0; _phase_k_full ^= 1; }
                    if (cta_rank == 0) {
                        elect_commit_cg2_multicast(s_full_addr + (phase) * 8, (uint16_t)(3));
                    }
                }
                if (tile_2 > 1) {
                    int prev_phase = tile_2 - 2 & 1;
                    int prev_tile = tile_2 - 2;
                    int pv_wait_phase = prev_tile >> 1 & 1;
                    if (cta_rank == 0) {
                        {
                            mbarrier_wait(p_full_addr + (prev_phase) * 8, pv_wait_phase);
                        }
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        mbarrier_wait(v_full_addr + (mma_v_stage) * 8, _phase_v_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_9 = (((smem_p_addr) >> 4) & 0x3FFF) + (prev_phase) * 512;
                        int _mma_b_lo_9 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (mma_v_stage) * 2048;
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
                    "mov.b32 id, 138477584;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_9), "r"(_mma_b_lo_9), "r"((tmem_tmem_scratch + (256))), "r"(((first_pv) ? 0 : 1)));
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        int _mma_a_lo_10 = (((smem_p_addr) >> 4) & 0x3FFF) + (prev_phase) * 512;
                        int _mma_b_lo_10 = ((((smem_v_addr + 16384) >> 4) & 0x3FFF) | 0x4000000) + (mma_v_stage) * 2048;
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
                    "mov.b32 id, 138477584;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_10), "r"(_mma_b_lo_10), "r"((tmem_tmem_scratch + (384))), "r"(((first_pv) ? 0 : 1)));
                        elect_commit_cg2_multicast(v_empty_addr + (mma_v_stage) * 8, (uint16_t)(3));
                        mma_v_stage += 1;
                        if (mma_v_stage == 2) { mma_v_stage = 0; _phase_v_full ^= 1; }
                        elect_commit_cg2_multicast(p_empty_addr + (prev_phase) * 8, (uint16_t)(3));
                        if (prev_tile >= 2) {
                            mbarrier_wait(o_full_addr + (prev_phase) * 8, prev_tile - 2 >> 1 & 1);
                        }
                        elect_commit_cg2_multicast(o_full_addr + (prev_phase) * 8, (uint16_t)(3));
                    }
                    first_pv = 0;
                }
            }
            int last_phase = my_n_tiles_2 - 1 & 1;
            int last_tile = my_n_tiles_2 - 1;
            int drain_wait_phase = last_tile >> 1 & 1;
            if (cta_rank == 0) {
                {
                    mbarrier_wait(p_full_addr + (last_phase) * 8, drain_wait_phase);
                }
                asm volatile("tcgen05.fence::after_thread_sync;");
                mbarrier_wait(v_full_addr + (mma_v_stage) * 8, _phase_v_full);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_a_lo_11 = (((smem_p_addr) >> 4) & 0x3FFF) + (last_phase) * 512;
                int _mma_b_lo_11 = ((((smem_v_addr) >> 4) & 0x3FFF) | 0x4000000) + (mma_v_stage) * 2048;
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
                    "mov.b32 id, 138477584;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_11), "r"(_mma_b_lo_11), "r"((tmem_tmem_scratch + (256))), "r"(((first_pv) ? 0 : 1)));
                asm volatile("tcgen05.fence::after_thread_sync;");
                int _mma_a_lo_12 = (((smem_p_addr) >> 4) & 0x3FFF) + (last_phase) * 512;
                int _mma_b_lo_12 = ((((smem_v_addr + 16384) >> 4) & 0x3FFF) | 0x4000000) + (mma_v_stage) * 2048;
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
                    "mov.b32 id, 138477584;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 256;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, id, {m0, m1, m2, m3, m4, m5, m6, m7}, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_12), "r"(_mma_b_lo_12), "r"((tmem_tmem_scratch + (384))), "r"(((first_pv) ? 0 : 1)));
                elect_commit_cg2_multicast(v_empty_addr + (mma_v_stage) * 8, (uint16_t)(3));
                mma_v_stage += 1;
                if (mma_v_stage == 2) { mma_v_stage = 0; _phase_v_full ^= 1; }
                elect_commit_cg2_multicast(p_empty_addr + (last_phase) * 8, (uint16_t)(3));
                if (last_tile >= 2) {
                    mbarrier_wait(o_full_addr + (last_phase) * 8, last_tile - 2 >> 1 & 1);
                }
                elect_commit_cg2_multicast(o_full_addr + (last_phase) * 8, (uint16_t)(3));
                elect_commit_cg2_multicast(o_done_addr, (uint16_t)(3));
                elect_commit_cg2_multicast(q_empty_addr, (uint16_t)(3));
                mbarrier_wait(o_empty_addr, 0);
                mbarrier_wait(o_empty_addr + 8, 0);
            }
            asm volatile("barrier.arrive 13, 160;" ::: "memory");
        }
    }
    // ---- Role: page_offsets_warp ----
    if (warp == 9) {
        { // page_offsets_warp_main
            const int wg3_dummy_1 = 0;
            unsigned int _phase_q_full_0_3 = 0;
            mbarrier_wait(q_full_addr, _phase_q_full_0_3);
            _phase_q_full_0_3 ^= 1;
        }
    }
    // ---- Role: v_load_warp ----
    if (warp == 10) {
        { // v_load_warp_main
            const int wg3_dummy_2 = 0;
            unsigned int load_v_stage = 0;
            int split_idx_3 = blockIdx.x / 2;
            int m_tile_3 = gridDim.y - 1 - blockIdx.y;
            int b_3 = blockIdx.z;
            int q_start_3 = cum_seq_lens_q[b_3];
            int q_len_b_3 = cum_seq_lens_q[b_3 + 1] - q_start_3;
            int kv_len_3 = seq_lens[b_3];
            int rows_b_3 = q_len_b_3 * num_heads;
            int row0_3 = m_tile_3 * 128;
            int rows_left_3 = rows_b_3 - row0_3;
            int rows_pos_3 = ((rows_left_3 < 0) ? 0 : rows_left_3);
            int rows_valid_3 = ((rows_pos_3 > 128) ? 128 : rows_pos_3);
            int row_base_global_3 = q_start_3 * num_heads + row0_3;
            int last_row_3 = row0_3 + rows_valid_3 - 1;
            int t_last_3 = last_row_3 / num_heads;
            int kv_end_raw_3 = kv_len_3 - q_len_b_3 + t_last_3 + 1;
            int kv_end_3 = ((rows_valid_3 == 0) ? 0 : kv_end_raw_3);
            int n_pages_3 = (kv_len_3 + 64 - 1) / 64;
            int n_tiles_total_3 = (kv_end_3 + 128 - 1) / 128;
            int tiles_per_split_3 = (n_tiles_total_3 + num_split - 1) / num_split;
            int my_start_raw_3 = split_idx_3 * tiles_per_split_3;
            int my_end_raw_3 = my_start_raw_3 + tiles_per_split_3;
            int my_end_3 = ((my_end_raw_3 > n_tiles_total_3) ? n_tiles_total_3 : my_end_raw_3);
            int my_n_raw_3 = my_end_3 - my_start_raw_3;
            int empty_3 = ((my_n_raw_3 < 1) ? 1 : 0);
            int my_n_tiles_3 = ((my_n_raw_3 < 1) ? 1 : my_n_raw_3);
            int my_start_block_3 = ((my_n_raw_3 < 1) ? 0 : my_start_raw_3);
            int pt_base_3 = b_3 * max_pages_per_seq;
            unsigned int _phase_q_full_0_4 = 0;
            mbarrier_wait(q_full_addr, _phase_q_full_0_4);
            _phase_q_full_0_4 ^= 1;
            unsigned int _phase_v_empty = 1;
            #pragma unroll 1
            for (int tile_3 = 0; tile_3 < my_n_tiles_3; tile_3++) {
                int abs_tile_v = my_start_block_3 + tile_3;
                int slot_ok = ((n_pages_3 > 2 * abs_tile_v) ? 2 * abs_tile_v : 0);
                int pg_v0 = page_table[pt_base_3 + slot_ok];
                int slot_ok_0 = ((n_pages_3 > 2 * abs_tile_v + 1) ? 2 * abs_tile_v + 1 : 0);
                int pg_v1 = page_table[pt_base_3 + slot_ok_0];
                mbarrier_wait(v_empty_addr + (load_v_stage) * 8, _phase_v_empty);
                {
                    if (cta_rank == 0) {
                        if (elect_sync()) {
                            {
                                asm volatile(
                                    "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                    :: "r"((v_full_addr + (load_v_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(65536)) : "memory");
                            }
                        }
                    }
                    if (elect_sync()) {
                        int v_dst = smem_v_addr + load_v_stage * 32768;
                        int v0_z = cta_rank;
                        int v1_z = 2 + cta_rank;
                        {
                            tma_2d_gmem2smem_cta2(v_dst, (&tmap_v), v0_z * 128, pg_v0 * 64, ((v_full_addr + (load_v_stage) * 8) & 0xFEFFFFFF));
                        }
                        {
                            tma_2d_gmem2smem_cta2(v_dst + 16384, (&tmap_v), v1_z * 128, pg_v0 * 64, ((v_full_addr + (load_v_stage) * 8) & 0xFEFFFFFF));
                        }
                        tma_2d_gmem2smem_cta2(v_dst + 8192, (&tmap_v), v0_z * 128, pg_v1 * 64, ((v_full_addr + (load_v_stage) * 8) & 0xFEFFFFFF));
                        tma_2d_gmem2smem_cta2(v_dst + 16384 + 8192, (&tmap_v), v1_z * 128, pg_v1 * 64, ((v_full_addr + (load_v_stage) * 8) & 0xFEFFFFFF));
                    }
                }
                load_v_stage += 1;
                if (load_v_stage == 2) { load_v_stage = 0; _phase_v_empty ^= 1; }
            }
        }
    }
    // ---- Role: load_warp ----
    if (warp == 11) {
        { // load_warp_main
            const int wg3_dummy_3 = 0;
            unsigned int load_k_stage = 0;
            int split_idx_4 = blockIdx.x / 2;
            int m_tile_4 = gridDim.y - 1 - blockIdx.y;
            int b_4 = blockIdx.z;
            int q_start_4 = cum_seq_lens_q[b_4];
            int q_len_b_4 = cum_seq_lens_q[b_4 + 1] - q_start_4;
            int kv_len_4 = seq_lens[b_4];
            int rows_b_4 = q_len_b_4 * num_heads;
            int row0_4 = m_tile_4 * 128;
            int rows_left_4 = rows_b_4 - row0_4;
            int rows_pos_4 = ((rows_left_4 < 0) ? 0 : rows_left_4);
            int rows_valid_4 = ((rows_pos_4 > 128) ? 128 : rows_pos_4);
            int row_base_global_4 = q_start_4 * num_heads + row0_4;
            int last_row_4 = row0_4 + rows_valid_4 - 1;
            int t_last_4 = last_row_4 / num_heads;
            int kv_end_raw_4 = kv_len_4 - q_len_b_4 + t_last_4 + 1;
            int kv_end_4 = ((rows_valid_4 == 0) ? 0 : kv_end_raw_4);
            int n_pages_4 = (kv_len_4 + 64 - 1) / 64;
            int n_tiles_total_4 = (kv_end_4 + 128 - 1) / 128;
            int tiles_per_split_4 = (n_tiles_total_4 + num_split - 1) / num_split;
            int my_start_raw_4 = split_idx_4 * tiles_per_split_4;
            int my_end_raw_4 = my_start_raw_4 + tiles_per_split_4;
            int my_end_4 = ((my_end_raw_4 > n_tiles_total_4) ? n_tiles_total_4 : my_end_raw_4);
            int my_n_raw_4 = my_end_4 - my_start_raw_4;
            int empty_4 = ((my_n_raw_4 < 1) ? 1 : 0);
            int my_n_tiles_4 = ((my_n_raw_4 < 1) ? 1 : my_n_raw_4);
            int my_start_block_4 = ((my_n_raw_4 < 1) ? 0 : my_start_raw_4);
            int pt_base_4 = b_4 * max_pages_per_seq;
            int q_row_global = row_base_global_4 + cta_rank * 64;
            unsigned int _phase_q_empty_0 = 1;
            mbarrier_wait(q_empty_addr, _phase_q_empty_0);
            _phase_q_empty_0 ^= 1;
            if (elect_sync()) {
                mbarrier_arrive_expect_tx(q_full_addr, 36864);
                #pragma unroll
                for (int s = 0; s < 4; s++) {
                    int q_dst = smem_q_addr + (unsigned int)(s * 8192);
                    tma_2d_gmem2smem(q_dst, (&tmap_q), s * 128, q_row_global, q_full_addr);
                }
                tma_2d_gmem2smem(smem_qr_addr, (&tmap_qr), 512, q_row_global, q_full_addr);
            }
            int slot_ok_1 = ((n_pages_4 > 2 * my_start_block_4 + cta_rank) ? 2 * my_start_block_4 + cta_rank : 0);
            int pg_k = page_table[pt_base_4 + slot_ok_1];
            unsigned int _phase_k_empty = 1;
            mbarrier_wait(k_empty_addr + (load_k_stage) * 8, _phase_k_empty);
            {
                if (cta_rank == 0) {
                    if (elect_sync()) {
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((k_full_addr + (load_k_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(73728)) : "memory");
                    }
                }
                if (elect_sync()) {
                    int k_dst = smem_k_addr + load_k_stage * 36864;
                    tma_2d_gmem2smem_cta2(k_dst, (&tmap_k), 0, pg_k * 64, ((k_full_addr + (load_k_stage) * 8) & 0xFEFFFFFF));
                    tma_2d_gmem2smem_cta2(k_dst + 8192, (&tmap_k), 128, pg_k * 64, ((k_full_addr + (load_k_stage) * 8) & 0xFEFFFFFF));
                    tma_2d_gmem2smem_cta2(k_dst + 16384, (&tmap_k), 256, pg_k * 64, ((k_full_addr + (load_k_stage) * 8) & 0xFEFFFFFF));
                    tma_2d_gmem2smem_cta2(k_dst + 24576, (&tmap_k), 384, pg_k * 64, ((k_full_addr + (load_k_stage) * 8) & 0xFEFFFFFF));
                    tma_2d_gmem2smem_cta2(smem_kr_addr + load_k_stage * 36864, (&tmap_kr), 512, pg_k * 64, ((k_full_addr + (load_k_stage) * 8) & 0xFEFFFFFF));
                }
            }
            load_k_stage += 1;
            if (load_k_stage == 3) { load_k_stage = 0; _phase_k_empty ^= 1; }
            if (my_n_tiles_4 > 1) {
                int slot_ok_0_1 = ((n_pages_4 > 2 * (my_start_block_4 + 1) + cta_rank) ? 2 * (my_start_block_4 + 1) + cta_rank : 0);
                int pg_k_1 = page_table[pt_base_4 + slot_ok_0_1];
                mbarrier_wait(k_empty_addr + (load_k_stage) * 8, _phase_k_empty);
                {
                    if (cta_rank == 0) {
                        if (elect_sync()) {
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((k_full_addr + (load_k_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(73728)) : "memory");
                        }
                    }
                    if (elect_sync()) {
                        int k_dst_1 = smem_k_addr + load_k_stage * 36864;
                        tma_2d_gmem2smem_cta2(k_dst_1, (&tmap_k), 0, pg_k_1 * 64, ((k_full_addr + (load_k_stage) * 8) & 0xFEFFFFFF));
                        tma_2d_gmem2smem_cta2(k_dst_1 + 8192, (&tmap_k), 128, pg_k_1 * 64, ((k_full_addr + (load_k_stage) * 8) & 0xFEFFFFFF));
                        tma_2d_gmem2smem_cta2(k_dst_1 + 16384, (&tmap_k), 256, pg_k_1 * 64, ((k_full_addr + (load_k_stage) * 8) & 0xFEFFFFFF));
                        tma_2d_gmem2smem_cta2(k_dst_1 + 24576, (&tmap_k), 384, pg_k_1 * 64, ((k_full_addr + (load_k_stage) * 8) & 0xFEFFFFFF));
                        tma_2d_gmem2smem_cta2(smem_kr_addr + load_k_stage * 36864, (&tmap_kr), 512, pg_k_1 * 64, ((k_full_addr + (load_k_stage) * 8) & 0xFEFFFFFF));
                    }
                }
                load_k_stage += 1;
                if (load_k_stage == 3) { load_k_stage = 0; _phase_k_empty ^= 1; }
            }
            if (my_n_tiles_4 > 2) {
                int slot_ok_0_2 = ((n_pages_4 > 2 * (my_start_block_4 + 2) + cta_rank) ? 2 * (my_start_block_4 + 2) + cta_rank : 0);
                int pg_k_1_1 = page_table[pt_base_4 + slot_ok_0_2];
                mbarrier_wait(k_empty_addr + (load_k_stage) * 8, _phase_k_empty);
                {
                    if (cta_rank == 0) {
                        if (elect_sync()) {
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((k_full_addr + (load_k_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(73728)) : "memory");
                        }
                    }
                    if (elect_sync()) {
                        int k_dst_2 = smem_k_addr + load_k_stage * 36864;
                        tma_2d_gmem2smem_cta2(k_dst_2, (&tmap_k), 0, pg_k_1_1 * 64, ((k_full_addr + (load_k_stage) * 8) & 0xFEFFFFFF));
                        tma_2d_gmem2smem_cta2(k_dst_2 + 8192, (&tmap_k), 128, pg_k_1_1 * 64, ((k_full_addr + (load_k_stage) * 8) & 0xFEFFFFFF));
                        tma_2d_gmem2smem_cta2(k_dst_2 + 16384, (&tmap_k), 256, pg_k_1_1 * 64, ((k_full_addr + (load_k_stage) * 8) & 0xFEFFFFFF));
                        tma_2d_gmem2smem_cta2(k_dst_2 + 24576, (&tmap_k), 384, pg_k_1_1 * 64, ((k_full_addr + (load_k_stage) * 8) & 0xFEFFFFFF));
                        tma_2d_gmem2smem_cta2(smem_kr_addr + load_k_stage * 36864, (&tmap_kr), 512, pg_k_1_1 * 64, ((k_full_addr + (load_k_stage) * 8) & 0xFEFFFFFF));
                    }
                }
                load_k_stage += 1;
                if (load_k_stage == 3) { load_k_stage = 0; _phase_k_empty ^= 1; }
            }
            #pragma unroll 1
            for (int tile_4 = 3; tile_4 < my_n_tiles_4; tile_4++) {
                int slot_ok_0_3 = ((n_pages_4 > 2 * (my_start_block_4 + tile_4) + cta_rank) ? 2 * (my_start_block_4 + tile_4) + cta_rank : 0);
                int pg_k_1_2 = page_table[pt_base_4 + slot_ok_0_3];
                mbarrier_wait(k_empty_addr + (load_k_stage) * 8, _phase_k_empty);
                {
                    if (cta_rank == 0) {
                        if (elect_sync()) {
                            asm volatile(
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                :: "r"((k_full_addr + (load_k_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(73728)) : "memory");
                        }
                    }
                    if (elect_sync()) {
                        int k_dst_3 = smem_k_addr + load_k_stage * 36864;
                        tma_2d_gmem2smem_cta2(k_dst_3, (&tmap_k), 0, pg_k_1_2 * 64, ((k_full_addr + (load_k_stage) * 8) & 0xFEFFFFFF));
                        tma_2d_gmem2smem_cta2(k_dst_3 + 8192, (&tmap_k), 128, pg_k_1_2 * 64, ((k_full_addr + (load_k_stage) * 8) & 0xFEFFFFFF));
                        tma_2d_gmem2smem_cta2(k_dst_3 + 16384, (&tmap_k), 256, pg_k_1_2 * 64, ((k_full_addr + (load_k_stage) * 8) & 0xFEFFFFFF));
                        tma_2d_gmem2smem_cta2(k_dst_3 + 24576, (&tmap_k), 384, pg_k_1_2 * 64, ((k_full_addr + (load_k_stage) * 8) & 0xFEFFFFFF));
                        tma_2d_gmem2smem_cta2(smem_kr_addr + load_k_stage * 36864, (&tmap_kr), 512, pg_k_1_2 * 64, ((k_full_addr + (load_k_stage) * 8) & 0xFEFFFFFF));
                    }
                }
                load_k_stage += 1;
                if (load_k_stage == 3) { load_k_stage = 0; _phase_k_empty ^= 1; }
            }
        }
    }

    // Cleanup
}

} // extern "C"
