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
#define TMEM_NCOLS 480
#define TMEM_TMEM_SOLVED_OFFSET 352
#define TMEM_TMEM_PAIR_OFFSET 384
#define TMEM_TMEM_STATE_OFFSET 0
#define TMEM_TMEM_PREDICTION_OFFSET 256
#define TMEM_TMEM_RESIDUAL_OFFSET 288
#define TMEM_TMEM_OUTPUT_OFFSET 320
#define NUM_RAW_PIPE_STAGES 6
#define SMEM_PACKET_SMEM_OFF 1024
#define SMEM_PACKET_SMEM_STAGE_BYTES 74752
#define SMEM_PACKET_SMEM_STRIDE 74752
#define SMEM_PACKET_TAIL_SMEM_OFF 70656
#define SMEM_PACKET_TAIL_SMEM_STAGE_BYTES 5120
#define SMEM_PACKET_TAIL_SMEM_STRIDE 74752
#define SMEM_SMEM_CHECKPOINT_OFF 1024
#define SMEM_SMEM_CHECKPOINT_STAGE_BYTES 32768
#define SMEM_SMEM_CHECKPOINT_STRIDE 32768
#define SMEM_SMEM_OUTPUT_OFF 58368
#define SMEM_SMEM_OUTPUT_STAGE_BYTES 8192
#define SMEM_SMEM_OUTPUT_STRIDE 74752
#define SMEM_SMEM_QD_OFF 1024
#define SMEM_SMEM_QD_STAGE_BYTES 32768
#define SMEM_SMEM_QD_STRIDE 74752
#define SMEM_SMEM_Q_RAW_OFF 1024
#define SMEM_SMEM_Q_RAW_STAGE_BYTES 8192
#define SMEM_SMEM_Q_RAW_STRIDE 74752
#define SMEM_SMEM_KD_OFF 5120
#define SMEM_SMEM_KD_STAGE_BYTES 32768
#define SMEM_SMEM_KD_STRIDE 74752
#define SMEM_SMEM_K_RAW_OFF 5120
#define SMEM_SMEM_K_RAW_STAGE_BYTES 8192
#define SMEM_SMEM_K_RAW_STRIDE 74752
#define SMEM_SMEM_KI_OFF 33792
#define SMEM_SMEM_KI_STAGE_BYTES 16384
#define SMEM_SMEM_KI_STRIDE 74752
#define SMEM_SMEM_GATE_RAW_OFF 54272
#define SMEM_SMEM_GATE_RAW_STAGE_BYTES 8192
#define SMEM_SMEM_GATE_RAW_STRIDE 74752
#define SMEM_SMEM_W_OUT_OFF 33792
#define SMEM_SMEM_W_OUT_STAGE_BYTES 16384
#define SMEM_SMEM_W_OUT_STRIDE 74752
#define SMEM_SMEM_KR_STORE_OFF 33792
#define SMEM_SMEM_KR_STORE_STAGE_BYTES 16384
#define SMEM_SMEM_KR_STORE_STRIDE 74752
#define SMEM_SMEM_QK_PLAIN_OFF 50176
#define SMEM_SMEM_QK_PLAIN_STAGE_BYTES 4096
#define SMEM_SMEM_QK_PLAIN_STRIDE 74752
#define SMEM_SMEM_BETA_RAW_OFF 74880
#define SMEM_SMEM_BETA_RAW_STAGE_BYTES 512
#define SMEM_SMEM_BETA_RAW_STRIDE 74752
#define SMEM_SMEM_GATE_OFF 54272
#define SMEM_SMEM_GATE_STAGE_BYTES 16384
#define SMEM_SMEM_GATE_STRIDE 74752
#define SMEM_SMEM_BETA_OFF 74752
#define SMEM_SMEM_BETA_STAGE_BYTES 128
#define SMEM_SMEM_BETA_STRIDE 74752
#define SMEM_SMEM_ABT_OFF 70656
#define SMEM_SMEM_ABT_STAGE_BYTES 4096
#define SMEM_SMEM_ABT_STRIDE 74752
#define SMEM_SMEM_INVERSE_CROSS_OFF 54272
#define SMEM_SMEM_INVERSE_CROSS_STAGE_BYTES 4096
#define SMEM_SMEM_INVERSE_CROSS_STRIDE 74752
#define SMEM_SMEM_QK_OUT_OFF 50176
#define SMEM_SMEM_QK_OUT_STAGE_BYTES 4096
#define SMEM_SMEM_QK_OUT_STRIDE 74752
#define SMEM_SMEM_DIAG_OFF 74880
#define SMEM_SMEM_DIAG_STAGE_BYTES 512
#define SMEM_SMEM_DIAG_STRIDE 74752
#define SMEM_SMEM_INVERSE_OUT_OFF 70656
#define SMEM_SMEM_INVERSE_OUT_STAGE_BYTES 4096
#define SMEM_SMEM_INVERSE_OUT_STRIDE 74752
#define SMEM_SMEM_QDKD_OFF 1024
#define SMEM_SMEM_QDKD_STAGE_BYTES 32768
#define SMEM_SMEM_QDKD_STRIDE 74752
#define SMEM_SMEM_V_RAW_OFF 58368
#define SMEM_SMEM_V_RAW_STAGE_BYTES 8192
#define SMEM_SMEM_V_RAW_STRIDE 74752
#define SMEM_TOTAL 225280

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

// Match CUTLASS ClusterBarrier::wait: a large suspendTimeHint lets the hardware
// take the blocking phase-check slowpath instead of spinning on TRYWAIT misses.
__device__ __forceinline__ void mbarrier_wait(int mbar_addr, int phase) {
    uint32_t ticks = 0x989680;
    asm volatile(
        "{\n\t"
        ".reg .pred P1;\n\t"
        "LAB_WAIT:\n\t"
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64"
        " P1, [%0], %1, %2;\n\t"
        "@P1 bra.uni DONE;\n\t"
        "bra.uni LAB_WAIT;\n\t"
        "DONE:\n\t"
        "}\n"
        :: "r"(mbar_addr), "r"(phase), "r"(ticks) : "memory");
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


__device__ __forceinline__ void tcgen05_mma_tf32(
    int taddr, uint64_t a_desc, uint64_t b_desc,
    uint32_t i_desc, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "tcgen05.mma.cta_group::1.kind::tf32 [%0], %1, %2, %3, p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(enable_input_d));
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
        "@leader tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, %3, p;\n\t"
        "}\n"
        :: "r"(a_lo), "r"(b_lo), "r"(taddr), "r"(i_desc), "r"(enable_d), "r"(a_dhi), "r"(b_dhi));
}


__device__ __forceinline__ void mma_ts_step(
    int taddr_out, int taddr_a, int b_lo, uint32_t b_dhi,
    uint32_t i_desc, int enable_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader, p;\n\t"
        ".reg .b32 dhi;\n\t"
        ".reg .b64 db;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "setp.ne.b32 p, %5, 0;\n\t"
        "mov.b32 dhi, %3;\n\t"
        "mov.b64 db, {%2, dhi};\n\t"
        "@leader tcgen05.mma.cta_group::1.kind::tf32 [%0], [%1], db, %4, p;\n\t"
        "}\n"
        :: "r"(taddr_out), "r"(taddr_a), "r"(b_lo), "r"(b_dhi),
           "r"(i_desc), "r"(enable_d));
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


__device__ __forceinline__ void tmem_st_x16_f32(int tmem_addr, const float* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x16.b32"
        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8,"
        "  %9, %10, %11, %12, %13, %14, %15, %16};"
        :: "r"(tmem_addr),
           "f"(src[0]),  "f"(src[1]),  "f"(src[2]),  "f"(src[3]),
           "f"(src[4]),  "f"(src[5]),  "f"(src[6]),  "f"(src[7]),
           "f"(src[8]),  "f"(src[9]),  "f"(src[10]), "f"(src[11]),
           "f"(src[12]), "f"(src[13]), "f"(src[14]), "f"(src[15]));
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

__global__ __launch_bounds__(512, 1) void
kernel_cake_kda_tf32_0aaa66f8b0fff559171a9d75b17859de6cf328b991c01274a3419eb656c327bd(__nv_bfloat16* __restrict__ q, CakeTensorMap const* q_tma, __nv_bfloat16* __restrict__ k, CakeTensorMap const* k_tma, __nv_bfloat16* __restrict__ raw_gate, CakeTensorMap const* raw_gate_tma, __nv_bfloat16* __restrict__ beta_logits, float* __restrict__ beta_active_f32, long long affine_cache_token_offset, int affine_cache_part_offset, CakeTensorMap const* beta_logits_tma, float* __restrict__ a_log, float* __restrict__ dt_bias, long long* __restrict__ cu_seqlens, int* __restrict__ seq_order, __nv_bfloat16* __restrict__ v, CakeTensorMap const* v_tma, __nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ initial_state, __nv_bfloat16* __restrict__ final_state, float* __restrict__ initial_state_f32, float* __restrict__ final_state_f32, unsigned long long state_indices_addr, long long state_slot_stride, int use_state_indices, int use_initial_state, int store_final_state, CakeTensorMap const* state_checkpoints_tma, __nv_bfloat16* __restrict__ state_checkpoints, long long* __restrict__ checkpoint_cu_starts, int checkpoint_every_n_tokens, float scale, int num_heads, float gate_lower_bound, long long beta_token_stride, int* __restrict__ task_ids, int* __restrict__ task_offsets, int* __restrict__ task_token_starts, int* __restrict__ task_token_counts, int* __restrict__ task_state_sources, int* __restrict__ task_state_destinations, float* __restrict__ mid_state_f32, unsigned int* __restrict__ mid_state_ready, CakeTensorMap const* owner_packet_tma, CakeTensorMap const* owner_packet_tail_tma, float* __restrict__ map_output_f32)
{
    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define pair_done_addr (mbar_base + 0)
    #define v_full_addr (mbar_base + 24)
    #define mma_done_addr (mbar_base + 48)
    #define factor_full_addr (mbar_base + 56)
    #define center_ready_addr (mbar_base + 80)
    #define factor_free_addr (mbar_base + 104)
    #define gate_raw_full_addr (mbar_base + 128)
    #define qk_raw_full_addr (mbar_base + 176)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 224);
    if (warp == 3) {
        uint64_t __cake_tensormap_acquire_addr = (uint64_t)(q_tma);
        if (lane == 1) __cake_tensormap_acquire_addr = (uint64_t)(k_tma);
        if (lane == 2) __cake_tensormap_acquire_addr = (uint64_t)(raw_gate_tma);
        if (lane == 3) __cake_tensormap_acquire_addr = (uint64_t)(beta_logits_tma);
        if (lane == 4) __cake_tensormap_acquire_addr = (uint64_t)(v_tma);
        if (lane == 5) __cake_tensormap_acquire_addr = (uint64_t)(state_checkpoints_tma);
        if (lane == 6) __cake_tensormap_acquire_addr = (uint64_t)(owner_packet_tma);
        if (lane == 7) __cake_tensormap_acquire_addr = (uint64_t)(owner_packet_tail_tma);
        if (lane < 8) {
            asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"(__cake_tensormap_acquire_addr) : "memory");
        }
    }


    // Kernel setup ops
    float* packet_smem = reinterpret_cast<float*>(smem_raw + 1024);
    const int packet_smem_addr = smem + 1024;
    float* packet_tail_smem = reinterpret_cast<float*>(smem_raw + 70656);
    const int packet_tail_smem_addr = smem + 70656;
    __nv_bfloat16* smem_checkpoint = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_checkpoint_addr = smem + 1024;
    __nv_bfloat16* smem_output = reinterpret_cast<__nv_bfloat16*>(smem_raw + 58368);
    const int smem_output_addr = smem + 58368;
    float* smem_qd = reinterpret_cast<float*>(smem_raw + 1024);
    const int smem_qd_addr = smem + 1024;
    __nv_bfloat16* smem_q_raw = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_q_raw_addr = smem + 1024;
    float* smem_kd = reinterpret_cast<float*>(smem_raw + 5120);
    const int smem_kd_addr = smem + 5120;
    __nv_bfloat16* smem_k_raw = reinterpret_cast<__nv_bfloat16*>(smem_raw + 5120);
    const int smem_k_raw_addr = smem + 5120;
    float* smem_ki = reinterpret_cast<float*>(smem_raw + 33792);
    const int smem_ki_addr = smem + 33792;
    __nv_bfloat16* smem_gate_raw = reinterpret_cast<__nv_bfloat16*>(smem_raw + 54272);
    const int smem_gate_raw_addr = smem + 54272;
    float* smem_w_out = reinterpret_cast<float*>(smem_raw + 33792);
    const int smem_w_out_addr = smem + 33792;
    float* smem_kr_store = reinterpret_cast<float*>(smem_raw + 33792);
    const int smem_kr_store_addr = smem + 33792;
    float* smem_qk_plain = reinterpret_cast<float*>(smem_raw + 50176);
    const int smem_qk_plain_addr = smem + 50176;
    __nv_bfloat16* smem_beta_raw = reinterpret_cast<__nv_bfloat16*>(smem_raw + 74880);
    const int smem_beta_raw_addr = smem + 74880;
    float* smem_gate = reinterpret_cast<float*>(smem_raw + 54272);
    const int smem_gate_addr = smem + 54272;
    float* smem_beta = reinterpret_cast<float*>(smem_raw + 74752);
    const int smem_beta_addr = smem + 74752;
    float* smem_abt = reinterpret_cast<float*>(smem_raw + 70656);
    const int smem_abt_addr = smem + 70656;
    float* smem_inverse_cross = reinterpret_cast<float*>(smem_raw + 54272);
    const int smem_inverse_cross_addr = smem + 54272;
    float* smem_qk_out = reinterpret_cast<float*>(smem_raw + 50176);
    const int smem_qk_out_addr = smem + 50176;
    float* smem_diag = reinterpret_cast<float*>(smem_raw + 74880);
    const int smem_diag_addr = smem + 74880;
    float* smem_inverse_out = reinterpret_cast<float*>(smem_raw + 70656);
    const int smem_inverse_out_addr = smem + 70656;
    float* smem_qdkd = reinterpret_cast<float*>(smem_raw + 1024);
    const int smem_qdkd_addr = smem + 1024;
    __nv_bfloat16* smem_v_raw = reinterpret_cast<__nv_bfloat16*>(smem_raw + 58368);
    const int smem_v_raw_addr = smem + 58368;

    // Mbarrier init (8 pipeline groups, 0 ordered-sequence groups, 28 barriers)
    // Mbarriers at smem_raw[0..224)

    if (warp == 0) {
        // pair_done: 3 barriers, init_count=1
        // v_full: 3 barriers, init_count=1
        // mma_done: 1 barriers, init_count=1
        // factor_full: 3 barriers, init_count=128
        // center_ready: 3 barriers, init_count=128
        // factor_free: 3 barriers, init_count=128
        // --- pipeline 'raw_pipe' ---
        // gate_raw_full: 6 barriers, init_count=1
        // qk_raw_full: 6 barriers, init_count=1
        // Warp-cooperative initialization, grouped by equal arrival count.
        for (int _bar = lane; _bar < 7; _bar += 32) {
            mbarrier_init(smem + 0 + _bar * 8, 1);
        }
        for (int _bar = lane; _bar < 9; _bar += 32) {
            mbarrier_init(smem + 56 + _bar * 8, 128);
        }
        for (int _bar = lane; _bar < 12; _bar += 32) {
            mbarrier_init(smem + 128 + _bar * 8, 1);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    __syncwarp();

    // TMEM alloc (512 columns, 480 used)
    if (warp == 0) {
        int _tmem_hold = smem + 224;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_solved = taddr + 352;
    const int tmem_tmem_pair = taddr + 384;
    const int tmem_tmem_state = taddr;
    const int tmem_tmem_prediction = taddr + 256;
    const int tmem_tmem_residual = taddr + 288;
    const int tmem_tmem_output = taddr + 320;

    // ---- Role: prepare_role ----
    if (warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 112;");
        { // prepare_role_main
            int raw_epoch = 0;
            {
                {
                    {
                        int value_offset = 0;
                        int seq_idx = seq_order[blockIdx.x / num_heads];
                        int head_idx = blockIdx.x % num_heads;
                        long long sequence_bos = cu_seqlens[seq_idx];
                        long long bos = sequence_bos;
                        long long sequence_eos = cu_seqlens[seq_idx + 1];
                        int my_chunks = (int)((sequence_eos - bos + 32 - 1) / 32);
                        int prep_instance = tid / 128;
                        int prep_tid = tid % 128;
                        int prep_warp = prep_tid / 32;
                        int col = prep_tid % 128;
                        float _exp2_18 = approx_exp2(a_log[head_idx] * 1.4426950408889634f);
                        float gate_rate = _exp2_18;
                        float gate_rate_half = gate_rate * 0.5f;
                        float gate_bias = dt_bias[head_idx * 128 + col];
                        float gate_half_scale = gate_lower_bound * 0.7213475204444817f;
                        int first_chunk = prep_instance;
                        long long current_token_base = bos + (long long)(first_chunk * 32);
                        long long current_eos = sequence_eos;
                        int packet_slot = first_chunk;
                        #pragma unroll 1
                        for (int cta_chunk = first_chunk; cta_chunk < my_chunks; cta_chunk += 3 * ((0) ? 0 : 1)) {
                            long long token_base = current_token_base;
                            long long eos = current_eos;
                            int chunk_is_full = ((eos >= token_base + 32) ? 1 : 0);
                            unsigned int raw_stage = (unsigned int)prep_instance * 2 + ((unsigned int)raw_epoch & 1);
                            unsigned int raw_phase = (unsigned int)raw_epoch / 2 & 1;
                            {
                                mbarrier_wait(factor_free_addr + (prep_instance) * 8, (unsigned int)(raw_epoch & 1) ^ 1);
                            }
                            if (prep_warp == 0) {
                                if (elect_sync()) {
                                    int gate_tx_bytes = 8192;
                                    mbarrier_arrive_expect_tx(gate_raw_full_addr + (raw_stage) * 8, gate_tx_bytes);
                                    tma_3d_gmem2smem(smem_gate_raw_addr + (unsigned int)(tid / 128 * 74752), raw_gate_tma, 0, head_idx, (int)token_base, gate_raw_full_addr + (raw_stage) * 8);
                                    mbarrier_arrive_expect_tx(qk_raw_full_addr + (raw_stage) * 8, 16384);
                                    #pragma unroll
                                    for (int half = 0; half < 2; half++) {
                                        tma_4d_gmem2smem(smem_q_raw_addr + (unsigned int)(tid / 128 * 74752) + (unsigned int)(half * 8192), q_tma, 0, (int)token_base, head_idx, half, qk_raw_full_addr + (raw_stage) * 8);
                                        tma_4d_gmem2smem(smem_k_raw_addr + (unsigned int)(tid / 128 * 74752) + (unsigned int)(half * 8192), k_tma, 0, (int)token_base, head_idx, half, qk_raw_full_addr + (raw_stage) * 8);
                                    }
                                }
                            }
                            float beta_value = 0.0f;
                            if (prep_tid < 32) {
                                {
                                    long long beta_token = token_base + (long long)prep_tid;
                                    if (beta_token < eos) {
                                        long long beta_index = beta_token * beta_token_stride + (long long)head_idx;
                                        {
                                            float _tanh_approx_11;
                                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_11) : "f"((float)beta_logits[beta_index] * 0.5f));
                                            beta_value = _tanh_approx_11 * 0.5f + 0.5f;
                                        }
                                    }
                                }
                            }
                            mbarrier_wait(gate_raw_full_addr + (raw_stage) * 8, raw_phase);
                            if (prep_tid < 32) {
                                smem_beta[prep_instance * 18688 + prep_tid] = beta_value;
                            }
                            if (chunk_is_full == 0) {
                                mbarrier_wait(qk_raw_full_addr + (raw_stage) * 8, raw_phase);
                                #pragma unroll
                                for (int tail_tile = 0; tail_tile < 2; tail_tile++) {
                                    int tail_row = (unsigned int)(tail_tile * 16 + prep_warp * 4) + lane / 8;
                                    int tail_lane_in_row = lane % 8;
                                    if (eos <= token_base + (long long)tail_row) {
                                        float tail_zero[8];
                                        tail_zero[0] = 0.0f;
                                        tail_zero[1] = 0.0f;
                                        tail_zero[2] = 0.0f;
                                        tail_zero[3] = 0.0f;
                                        tail_zero[4] = 0.0f;
                                        tail_zero[5] = 0.0f;
                                        tail_zero[6] = 0.0f;
                                        tail_zero[7] = 0.0f;
                                        #pragma unroll
                                        for (int dim_half = 0; dim_half < 2; dim_half++) {
                                            int tail_segment = dim_half * 8 + tail_lane_in_row;
                                            unsigned int packed[4];
                                            #pragma unroll
                                            for (int _lp = 0; _lp < 4; _lp++) {
                                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(tail_zero[_lp*2 + 0], tail_zero[_lp*2+1 + 0]));
                                                packed[_lp] = *(uint32_t*)&_bf2;
                                            }
                                            #pragma unroll
                                            for (int word = 0; word < 4; word++) {
                                                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_q_raw_addr + (unsigned int)(tid / 128 * 74752) + (unsigned int)(tail_segment * 8 / 64 * 8192 + tail_row * 128 + tail_segment * 8 % 64 * 2 ^ (tail_segment * 8 / 64 * 8192 + tail_row * 128 + tail_segment * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word * 4)), "r"((packed[word])));
                                            }
                                            unsigned int packed_0[4];
                                            #pragma unroll
                                            for (int _lp = 0; _lp < 4; _lp++) {
                                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(tail_zero[_lp*2 + 0], tail_zero[_lp*2+1 + 0]));
                                                packed_0[_lp] = *(uint32_t*)&_bf2;
                                            }
                                            #pragma unroll
                                            for (int word_1 = 0; word_1 < 4; word_1++) {
                                                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_k_raw_addr + (unsigned int)(tid / 128 * 74752) + (unsigned int)(tail_segment * 8 / 64 * 8192 + tail_row * 128 + tail_segment * 8 % 64 * 2 ^ (tail_segment * 8 / 64 * 8192 + tail_row * 128 + tail_segment * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_1 * 4)), "r"((packed_0[word_1])));
                                            }
                                        }
                                    }
                                }
                                if (prep_tid < 128) {
                                    #pragma unroll
                                    for (int tail_gate_row = 0; tail_gate_row < 32; tail_gate_row++) {
                                        if (eos <= token_base + (long long)tail_gate_row) {
                                            smem_gate_raw[prep_instance * 37376 + tail_gate_row * 128 + col] = 0.0f;
                                        }
                                    }
                                }
                                asm volatile("barrier.sync %0, 128;" :: "r"(prep_instance + 1) : "memory");
                            }
                            int wide_gate = 0;
                            int tensor_pair = 1;
                            float prefix_log2 = 0.0f;
                            float gate_decay[32];
                            if (prep_tid < 128) {
                                if (chunk_is_full != 0) {
                                    #pragma unroll
                                    for (int row = 0; row < 32; row++) {
                                        float _cvt_f32_4 = __bfloat162float(smem_gate_raw[prep_instance * 37376 + row * 128 + col]);
                                        float gate_raw_value = _cvt_f32_4;
                                        float gate_increment = 0.0f;
                                        {
                                            float gate_arg = gate_rate_half * (gate_raw_value + gate_bias);
                                            float _tanh_approx_13;
                                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_13) : "f"(gate_arg));
                                            float _fma_40 = __fmaf_rn(_tanh_approx_13, gate_half_scale, gate_half_scale);
                                            gate_increment = _fma_40;
                                        }
                                        prefix_log2 += gate_increment;
                                        {
                                            gate_decay[row] = prefix_log2;
                                        }
                                    }
                                } else {
                                    #pragma unroll
                                    for (int row_1 = 0; row_1 < 32; row_1++) {
                                        float gate_increment_1 = 0.0f;
                                        if (eos > token_base + (long long)row_1) {
                                            float _cvt_f32_5 = __bfloat162float(smem_gate_raw[prep_instance * 37376 + row_1 * 128 + col]);
                                            float gate_raw_value_1 = _cvt_f32_5;
                                            {
                                                float gate_arg_1 = gate_rate_half * (gate_raw_value_1 + gate_bias);
                                                float _tanh_approx_14;
                                                asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_14) : "f"(gate_arg_1));
                                                float _fma_45 = __fmaf_rn(_tanh_approx_14, gate_half_scale, gate_half_scale);
                                                gate_increment_1 = _fma_45;
                                            }
                                        }
                                        prefix_log2 += gate_increment_1;
                                        {
                                            gate_decay[row_1] = prefix_log2;
                                        }
                                    }
                                }
                            }
                            asm volatile("barrier.sync %0, 128;" :: "r"(prep_instance + 1) : "memory");
                            {
                                if (prep_tid < 128) {
                                    #pragma unroll
                                    for (int row_2 = 0; row_2 < 32; row_2++) {
                                        {
                                            float _exp2_24 = approx_exp2(gate_decay[row_2] + 112.0f);
                                            gate_decay[row_2] = _exp2_24;
                                        }
                                        smem_gate[prep_instance * 18688 + row_2 * 128 + col] = gate_decay[row_2];
                                    }
                                    float total_decay = gate_decay[31];
                                    smem_gate[prep_instance * 18688 + 3968 + col] = total_decay;
                                    {
                                        float _exp2_26 = approx_exp2(prefix_log2);
                                        smem_diag[prep_instance * 18688 + col] = _exp2_26;
                                    }
                                }
                            }
                            {
                                mbarrier_arrive(center_ready_addr + (prep_instance) * 8);
                            }
                            asm volatile("barrier.sync %0, 128;" :: "r"(prep_instance + 1) : "memory");
                            if (chunk_is_full != 0) {
                                mbarrier_wait(qk_raw_full_addr + (raw_stage) * 8, raw_phase);
                            }
                            int lane_in_row = lane % 8;
                            float kr_saved[32];
                            float qk_self[2];
                            unsigned int q_raw_cache[16];
                            unsigned int k_raw_cache[16];
                            {
                                #pragma unroll
                                for (int raw_tile = 0; raw_tile < 2; raw_tile++) {
                                    int raw_row = (unsigned int)(raw_tile * 16 + prep_warp * 4) + lane / 8;
                                    #pragma unroll
                                    for (int raw_half = 0; raw_half < 2; raw_half++) {
                                        int raw_segment = raw_half * 8 + lane_in_row;
                                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                            : "=r"(*reinterpret_cast<uint32_t*>(&q_raw_cache[raw_tile * 8 + raw_half * 4])), "=r"(*reinterpret_cast<uint32_t*>(&q_raw_cache[(raw_tile * 8 + raw_half * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&q_raw_cache[(raw_tile * 8 + raw_half * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&q_raw_cache[(raw_tile * 8 + raw_half * 4) + 3]))
                                            : "r"((smem_q_raw_addr + (unsigned int)(tid / 128 * 74752) + (unsigned int)(raw_segment * 8 / 64 * 8192 + raw_row * 128 + raw_segment * 8 % 64 * 2 ^ (raw_segment * 8 / 64 * 8192 + raw_row * 128 + raw_segment * 8 % 64 * 2 >> 7 & 7) << 4))));
                                        asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                            : "=r"(*reinterpret_cast<uint32_t*>(&k_raw_cache[raw_tile * 8 + raw_half * 4])), "=r"(*reinterpret_cast<uint32_t*>(&k_raw_cache[(raw_tile * 8 + raw_half * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&k_raw_cache[(raw_tile * 8 + raw_half * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&k_raw_cache[(raw_tile * 8 + raw_half * 4) + 3]))
                                            : "r"((smem_k_raw_addr + (unsigned int)(tid / 128 * 74752) + (unsigned int)(raw_segment * 8 / 64 * 8192 + raw_row * 128 + raw_segment * 8 % 64 * 2 ^ (raw_segment * 8 / 64 * 8192 + raw_row * 128 + raw_segment * 8 % 64 * 2 >> 7 & 7) << 4))));
                                    }
                                }
                                asm volatile("barrier.sync %0, 128;" :: "r"(prep_instance + 1) : "memory");
                            }
                            #pragma unroll
                            for (int norm_tile = 0; norm_tile < 2; norm_tile++) {
                                int row_3 = (unsigned int)(norm_tile * 16 + prep_warp * 4) + lane / 8;
                                float q_raw[16];
                                float k_raw[16];
                                unsigned int q_raw_packed[8];
                                unsigned int k_raw_packed[8];
                                q_raw[0] = 0.0f;
                                q_raw[1] = 0.0f;
                                q_raw[2] = 0.0f;
                                q_raw[3] = 0.0f;
                                q_raw[4] = 0.0f;
                                q_raw[5] = 0.0f;
                                q_raw[6] = 0.0f;
                                q_raw[7] = 0.0f;
                                q_raw[8] = 0.0f;
                                q_raw[9] = 0.0f;
                                q_raw[10] = 0.0f;
                                q_raw[11] = 0.0f;
                                q_raw[12] = 0.0f;
                                q_raw[13] = 0.0f;
                                q_raw[14] = 0.0f;
                                q_raw[15] = 0.0f;
                                k_raw[0] = 0.0f;
                                k_raw[1] = 0.0f;
                                k_raw[2] = 0.0f;
                                k_raw[3] = 0.0f;
                                k_raw[4] = 0.0f;
                                k_raw[5] = 0.0f;
                                k_raw[6] = 0.0f;
                                k_raw[7] = 0.0f;
                                k_raw[8] = 0.0f;
                                k_raw[9] = 0.0f;
                                k_raw[10] = 0.0f;
                                k_raw[11] = 0.0f;
                                k_raw[12] = 0.0f;
                                k_raw[13] = 0.0f;
                                k_raw[14] = 0.0f;
                                k_raw[15] = 0.0f;
                                {
                                    #pragma unroll
                                    for (int raw_word = 0; raw_word < 8; raw_word++) {
                                        q_raw_packed[raw_word] = q_raw_cache[norm_tile * 8 + raw_word];
                                        k_raw_packed[raw_word] = k_raw_cache[norm_tile * 8 + raw_word];
                                    }
                                }
                                float q_raw_packed_f32[16];
                                #pragma unroll
                                for (int _pair = 0; _pair < 8; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        "shl.b32 %0, %2, 16;\n\t"
                                        "and.b32 %1, %2, 0xffff0000;\n\t"
                                        "}\n"
                                        : "=f"((&q_raw_packed_f32[_pair * 2])[0]), "=f"((&q_raw_packed_f32[_pair * 2])[1])
                                        : "r"(q_raw_packed[_pair]));
                                }
                                float k_raw_packed_f32[16];
                                #pragma unroll
                                for (int _pair = 0; _pair < 8; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        "shl.b32 %0, %2, 16;\n\t"
                                        "and.b32 %1, %2, 0xffff0000;\n\t"
                                        "}\n"
                                        : "=f"((&k_raw_packed_f32[_pair * 2])[0]), "=f"((&k_raw_packed_f32[_pair * 2])[1])
                                        : "r"(k_raw_packed[_pair]));
                                }
                                #pragma unroll
                                for (int elem = 0; elem < 16; elem++) {
                                    q_raw[elem] = q_raw_packed_f32[elem];
                                    k_raw[elem] = k_raw_packed_f32[elem];
                                }
                                float q_sum = 0.0f;
                                float k_sum = 0.0f;
                                #pragma unroll
                                for (int elem_pair = 0; elem_pair < 8; elem_pair++) {
                                    float _bf16x2_dot_f32_6;
                                    asm volatile(
                                        "{\n\t"
                                        ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                                        "mov.b32 {a_lo, a_hi}, %1;\n\t"
                                        "mov.b32 {b_lo, b_hi}, %2;\n\t"
                                        "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                                        "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                                        "}\n"
                                        : "=f"(_bf16x2_dot_f32_6) : "r"(q_raw_packed[elem_pair]), "r"(q_raw_packed[elem_pair]), "f"(q_sum));
                                    q_sum = _bf16x2_dot_f32_6;
                                    float _bf16x2_dot_f32_7;
                                    asm volatile(
                                        "{\n\t"
                                        ".reg .b16 a_lo, a_hi, b_lo, b_hi;\n\t"
                                        "mov.b32 {a_lo, a_hi}, %1;\n\t"
                                        "mov.b32 {b_lo, b_hi}, %2;\n\t"
                                        "fma.rn.f32.bf16 %0, a_lo, b_lo, %3;\n\t"
                                        "fma.rn.f32.bf16 %0, a_hi, b_hi, %0;\n\t"
                                        "}\n"
                                        : "=f"(_bf16x2_dot_f32_7) : "r"(k_raw_packed[elem_pair]), "r"(k_raw_packed[elem_pair]), "f"(k_sum));
                                    k_sum = _bf16x2_dot_f32_7;
                                }
                                float _shfl_xor_64 = __shfl_xor_sync(0xFFFFFFFF, q_sum, 4);
                                q_sum += _shfl_xor_64;
                                float _shfl_xor_65 = __shfl_xor_sync(0xFFFFFFFF, k_sum, 4);
                                k_sum += _shfl_xor_65;
                                float _shfl_xor_66 = __shfl_xor_sync(0xFFFFFFFF, q_sum, 2);
                                q_sum += _shfl_xor_66;
                                float _shfl_xor_67 = __shfl_xor_sync(0xFFFFFFFF, k_sum, 2);
                                k_sum += _shfl_xor_67;
                                float _shfl_xor_68 = __shfl_xor_sync(0xFFFFFFFF, q_sum, 1);
                                q_sum += _shfl_xor_68;
                                float _shfl_xor_69 = __shfl_xor_sync(0xFFFFFFFF, k_sum, 1);
                                k_sum += _shfl_xor_69;
                                float _rsqrt_4 = rsqrtf(q_sum + 1e-06f);
                                float q_inv = _rsqrt_4;
                                float _rsqrt_5 = rsqrtf(k_sum + 1e-06f);
                                float k_inv_norm = _rsqrt_5;
                                if (norm_tile == 1) {
                                    raw_epoch = raw_epoch + 1;
                                    current_token_base = token_base + (long long)(3 * ((0) ? 0 : 1) * 32);
                                    current_eos = sequence_eos;
                                }
                                #pragma unroll
                                for (int dim_half_1 = 0; dim_half_1 < 2; dim_half_1++) {
                                    int segment = dim_half_1 * 8 + lane_in_row;
                                    int reg_base = dim_half_1 * 8;
                                    float qd_values[8];
                                    float kd_values[8];
                                    float ki_values[8];
                                    float kr_values[8];
                                    float2 _f2_32 = make_float2(q_inv, q_inv);
                                    float2 q_norm_pair = _f2_32;
                                    float2 _f2_33 = make_float2(k_inv_norm, k_inv_norm);
                                    float2 k_norm_pair = _f2_33;
                                    #pragma unroll
                                    for (int elem_pair_1 = 0; elem_pair_1 < 4; elem_pair_1++) {
                                        int elem0 = elem_pair_1 * 2;
                                        int elem1 = elem0 + 1;
                                        int this_col0 = segment * 8 + elem0;
                                        int this_col1 = this_col0 + 1;
                                        float decay0 = smem_gate[prep_instance * 18688 + row_3 * 128 + this_col0];
                                        float decay1 = smem_gate[prep_instance * 18688 + row_3 * 128 + this_col1];
                                        int feature0_wide = 0;
                                        int feature1_wide = 0;
                                        if (wide_gate != 0) {
                                            float center0 = smem_diag[prep_instance * 18688 + this_col0];
                                            float center1 = smem_diag[prep_instance * 18688 + this_col1];
                                            feature0_wide = (int)(center0 < 0.0f);
                                            feature1_wide = (int)(center1 < 0.0f);
                                            if (feature0_wide != 0) {
                                                decay0 = 1.0f;
                                            }
                                            if (feature1_wide != 0) {
                                                decay1 = 1.0f;
                                            }
                                        }
                                        float _rcp_4 = approx_rcp(decay0);
                                        float inv_decay0 = _rcp_4;
                                        float _rcp_5 = approx_rcp(decay1);
                                        float inv_decay1 = _rcp_5;
                                        float2 _f2_34 = make_float2(q_raw[reg_base + elem0], q_raw[reg_base + elem1]);
                                        float2 raw_q_pair = _f2_34;
                                        float2 _f2_35 = make_float2(k_raw[reg_base + elem0], k_raw[reg_base + elem1]);
                                        float2 raw_k_pair = _f2_35;
                                        float2 _mul_f32x2_14;
                                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_14) : "l"(*(const unsigned long long*)&raw_q_pair), "l"(*(const unsigned long long*)&q_norm_pair));
                                        float2 q_value_pair = _mul_f32x2_14;
                                        float2 _mul_f32x2_15;
                                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_15) : "l"(*(const unsigned long long*)&raw_k_pair), "l"(*(const unsigned long long*)&k_norm_pair));
                                        float2 k_value_pair = _mul_f32x2_15;
                                        float normalized_values[4];
                                        normalized_values[0] = q_value_pair.x;
                                        normalized_values[1] = q_value_pair.y;
                                        normalized_values[2] = k_value_pair.x;
                                        normalized_values[3] = k_value_pair.y;
                                        unsigned int normalized_words[2];
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 2; _lp++) {
                                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(normalized_values[_lp*2 + 0], normalized_values[_lp*2+1 + 0]));
                                            normalized_words[_lp] = *(uint32_t*)&_bf2;
                                        }
                                        float normalized_words_f32[4];
                                        #pragma unroll
                                        for (int _pair = 0; _pair < 2; _pair++) {
                                            asm volatile(
                                                "{\n\t"
                                                "shl.b32 %0, %2, 16;\n\t"
                                                "and.b32 %1, %2, 0xffff0000;\n\t"
                                                "}\n"
                                                : "=f"((&normalized_words_f32[_pair * 2])[0]), "=f"((&normalized_words_f32[_pair * 2])[1])
                                                : "r"(normalized_words[_pair]));
                                        }
                                        float2 _f2_36 = make_float2(normalized_words_f32[0], normalized_words_f32[1]);
                                        q_value_pair = _f2_36;
                                        float2 _f2_37 = make_float2(normalized_words_f32[2], normalized_words_f32[3]);
                                        k_value_pair = _f2_37;
                                        float2 _f2_38 = make_float2(decay0, decay1);
                                        float2 decay_pair = _f2_38;
                                        float2 _f2_39 = make_float2(inv_decay0, inv_decay1);
                                        float2 inv_decay_pair = _f2_39;
                                        float2 _mul_f32x2_16;
                                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_16) : "l"(*(const unsigned long long*)&q_value_pair), "l"(*(const unsigned long long*)&decay_pair));
                                        float2 qd_pair = _mul_f32x2_16;
                                        float2 _mul_f32x2_17;
                                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_17) : "l"(*(const unsigned long long*)&k_value_pair), "l"(*(const unsigned long long*)&decay_pair));
                                        float2 kd_pair = _mul_f32x2_17;
                                        float2 _mul_f32x2_18;
                                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_18) : "l"(*(const unsigned long long*)&k_value_pair), "l"(*(const unsigned long long*)&inv_decay_pair));
                                        float2 ki_pair = _mul_f32x2_18;
                                        qd_values[elem0] = qd_pair.x;
                                        qd_values[elem1] = qd_pair.y;
                                        kd_values[elem0] = kd_pair.x;
                                        kd_values[elem1] = kd_pair.y;
                                        ki_values[elem0] = ki_pair.x;
                                        ki_values[elem1] = ki_pair.y;
                                        if (wide_gate != 0) {
                                            if (feature0_wide != 0) {
                                                ki_values[elem0] = 0.0f;
                                            }
                                            if (feature1_wide != 0) {
                                                ki_values[elem1] = 0.0f;
                                            }
                                        }
                                        float total_decay0 = smem_gate[prep_instance * 18688 + 3968 + this_col0];
                                        float total_decay1 = smem_gate[prep_instance * 18688 + 3968 + this_col1];
                                        float2 _f2_40 = make_float2(total_decay0, total_decay1);
                                        float2 _mul_f32x2_19;
                                        asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_19) : "l"(*(const unsigned long long*)&ki_pair), "l"(*(const unsigned long long*)&_f2_40));
                                        float2 kr_pair = _mul_f32x2_19;
                                        {
                                            float2 _f2_41 = make_float2(3.552713678800501e-15f, 3.552713678800501e-15f);
                                            float2 _mul_f32x2_20;
                                            asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_20) : "l"(*(const unsigned long long*)&kr_pair), "l"(*(const unsigned long long*)&_f2_41));
                                            kr_pair = _mul_f32x2_20;
                                        }
                                        kr_values[elem0] = kr_pair.x;
                                        kr_values[elem1] = kr_pair.y;
                                    }
                                    {
                                        unsigned int words[8];
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 8; _lp++) {
                                            words[_lp] = __float_as_uint(qd_values[_lp + 0]);
                                        }
                                        #pragma unroll
                                        for (int vector = 0; vector < 2; vector++) {
                                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                                "r"((smem_qd_addr + (unsigned int)(tid / 128 * 74752) + (unsigned int)((segment * 8 + vector * 4) / 32 * 8192 + row_3 * 128 + (segment * 8 + vector * 4) % 32 * 4 ^ ((segment * 8 + vector * 4) / 32 * 8192 + row_3 * 128 + (segment * 8 + vector * 4) % 32 * 4 >> 7 & 7) << 4))), "r"(*reinterpret_cast<uint32_t*>(&words[vector * 4])), "r"(*reinterpret_cast<uint32_t*>(&words[(vector * 4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&words[(vector * 4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&words[(vector * 4) + 3])));
                                        }
                                    }
                                    {
                                        unsigned int words_1[8];
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 8; _lp++) {
                                            words_1[_lp] = __float_as_uint(kd_values[_lp + 0]);
                                        }
                                        #pragma unroll
                                        for (int vector_1 = 0; vector_1 < 2; vector_1++) {
                                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                                "r"((smem_kd_addr + (unsigned int)(tid / 128 * 74752) + (unsigned int)((segment * 8 + vector_1 * 4) / 32 * 8192 + row_3 * 128 + (segment * 8 + vector_1 * 4) % 32 * 4 ^ ((segment * 8 + vector_1 * 4) / 32 * 8192 + row_3 * 128 + (segment * 8 + vector_1 * 4) % 32 * 4 >> 7 & 7) << 4))), "r"(*reinterpret_cast<uint32_t*>(&words_1[vector_1 * 4])), "r"(*reinterpret_cast<uint32_t*>(&words_1[(vector_1 * 4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&words_1[(vector_1 * 4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&words_1[(vector_1 * 4) + 3])));
                                        }
                                    }
                                    if (tensor_pair != 0) {
                                        unsigned int words_2[8];
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 8; _lp++) {
                                            asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(words_2[_lp]) : "f"(ki_values[_lp + 0]));
                                        }
                                        #pragma unroll
                                        for (int vector_2 = 0; vector_2 < 2; vector_2++) {
                                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                                "r"((smem_ki_addr + (unsigned int)(tid / 128 * 74752) + (unsigned int)((segment * 8 + vector_2 * 4) / 32 * 4096 + row_3 * 128 + (segment * 8 + vector_2 * 4) % 32 * 4 ^ ((segment * 8 + vector_2 * 4) / 32 * 4096 + row_3 * 128 + (segment * 8 + vector_2 * 4) % 32 * 4 >> 7 & 7) << 4))), "r"(*reinterpret_cast<uint32_t*>(&words_2[vector_2 * 4])), "r"(*reinterpret_cast<uint32_t*>(&words_2[(vector_2 * 4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&words_2[(vector_2 * 4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&words_2[(vector_2 * 4) + 3])));
                                        }
                                    }
                                    #pragma unroll
                                    for (int elem_1 = 0; elem_1 < 8; elem_1++) {
                                        kr_saved[norm_tile * 16 + dim_half_1 * 8 + elem_1] = kr_values[elem_1];
                                    }
                                }
                            }
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            asm volatile("barrier.sync %0, 128;" :: "r"(prep_instance + 1) : "memory");
                            if (tensor_pair != 0) {
                                if (prep_warp == 0) {
                                    int _mma_a_lo_2 = make_warp_uniform((((smem_qdkd_addr) >> 4) & 0x3FFF) + (prep_instance) * 4672);
                                    int _mma_b_lo_2 = make_warp_uniform((((smem_ki_addr) >> 4) & 0x3FFF) + (prep_instance) * 4672);
                                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 adhi, bdhi, alo, blo, id;\n\t"
                    ".reg .b64 da, db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 adhi, 0x40004040;\n\t"
                    "mov.b32 bdhi, 0x40004040;\n\t"
                    "mov.b32 id, 67635472;\n\t"
                    "mov.b32 alo, %0;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p0;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 250;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 250;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 506;\n\t"
                    "add.u32 blo, blo, 250;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "add.u32 alo, alo, 2;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 da, {alo, adhi};\n\t"
                    "mov.b64 db, {blo, bdhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%2], da, db, id, p1;\n\t"
                    "}\n"
                    :: "r"(_mma_a_lo_2), "r"(_mma_b_lo_2), "r"((tmem_tmem_pair + (prep_instance * 32))), "r"(0));
                                    elect_commit(pair_done_addr + (prep_instance) * 8);
                                }
                                mbarrier_wait(pair_done_addr + (prep_instance) * 8, (unsigned int)(raw_epoch - 1 & 1));
                                if (prep_warp < 4) {
                                    float _tmem_load_2[16];
                                    asm volatile(
                                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_2[15]))
                                        : "r"(taddr + 384 + (unsigned int)(prep_instance * 32) + (unsigned int)(prep_warp * 32 << 16)));
                                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                    if (prep_warp < 2) {
                                        unsigned int pair_rounded[16];
                                        {
                                            #pragma unroll
                                            for (int _lp = 0; _lp < 16; _lp++) {
                                                pair_rounded[_lp] = __float_as_uint(_tmem_load_2[_lp + 0]);
                                            }
                                        }
                                        #pragma unroll
                                        for (int word_2 = 0; word_2 < 16; word_2++) {
                                            int row_4 = (unsigned int)(prep_warp * 16) + lane / 4 + (unsigned int)(word_2 % 4 / 2 * 8);
                                            int col_0 = lane % 4 * 2 + (unsigned int)(word_2 % 2) + (unsigned int)(word_2 / 4 * 8);
                                            float value = 0.0f;
                                            if (row_4 >= col_0) {
                                                value = reinterpret_cast<float*>(pair_rounded)[word_2];
                                            }
                                            {
                                                value = value * 1.8446744073709552e+19f;
                                            }
                                            smem_qk_plain[(unsigned int)(tid / 128 * 18688) + (col_0 / 16 * 2048 + row_4 * 64 + col_0 % 16 * 4 ^ (col_0 / 16 * 2048 + row_4 * 64 + col_0 % 16 * 4 >> 7 & 3) << 4) / 4] = value;
                                        }
                                    } else {
                                        #pragma unroll
                                        for (int word_3 = 0; word_3 < 16; word_3++) {
                                            int row_5 = (unsigned int)((prep_warp - 2) * 16) + lane / 4 + (unsigned int)(word_3 % 4 / 2 * 8);
                                            int col_0_1 = lane % 4 * 2 + (unsigned int)(word_3 % 2) + (unsigned int)(word_3 / 4 * 8);
                                            if (wide_gate != 0) {
                                                smem_abt[(unsigned int)(tid / 128 * 18688) + (col_0_1 / 32 * 4096 + row_5 * 128 + col_0_1 % 32 * 4 ^ (col_0_1 / 32 * 4096 + row_5 * 128 + col_0_1 % 32 * 4 >> 7 & 3) << 5) / 4] = _tmem_load_2[word_3];
                                            } else {
                                                smem_inverse_cross[(unsigned int)(tid / 128 * 18688) + (col_0_1 / 16 * 2048 + row_5 * 64 + col_0_1 % 16 * 4 ^ (col_0_1 / 16 * 2048 + row_5 * 64 + col_0_1 % 16 * 4 >> 7 & 3) << 4) / 4] = _tmem_load_2[word_3];
                                            }
                                        }
                                    }
                                }
                            } else {
                                if (prep_warp == 0) {
                                    if (elect_sync()) {
                                        mbarrier_arrive(pair_done_addr + (prep_instance) * 8);
                                    }
                                }
                                mbarrier_wait(pair_done_addr + (prep_instance) * 8, (unsigned int)(raw_epoch - 1 & 1));
                            }
                            if (tensor_pair != 0) {
                                #pragma unroll
                                for (int tile = 0; tile < 2; tile++) {
                                    int kr_row = (unsigned int)(tile * 16 + prep_warp * 4) + lane / 8;
                                    #pragma unroll
                                    for (int half_1 = 0; half_1 < 2; half_1++) {
                                        unsigned int kr_words[8];
                                        float kr_values_1[8];
                                        #pragma unroll
                                        for (int elem_2 = 0; elem_2 < 8; elem_2++) {
                                            kr_values_1[elem_2] = kr_saved[tile * 16 + half_1 * 8 + elem_2];
                                        }
                                        {
                                            unsigned int words_3[8];
                                            #pragma unroll
                                            for (int _lp = 0; _lp < 8; _lp++) {
                                                words_3[_lp] = __float_as_uint(kr_values_1[_lp + 0]);
                                            }
                                            #pragma unroll
                                            for (int vector_3 = 0; vector_3 < 2; vector_3++) {
                                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" ::
                                                    "r"((smem_kr_store_addr + (unsigned int)(tid / 128 * 74752) + ((((unsigned int)(half_1 * 8) + lane % 8) * 8 + (unsigned int)(vector_3 * 4)) / 32 * 4096 + (unsigned int)(kr_row * 128) + (((unsigned int)(half_1 * 8) + lane % 8) * 8 + (unsigned int)(vector_3 * 4)) % 32 * 4 ^ ((((unsigned int)(half_1 * 8) + lane % 8) * 8 + (unsigned int)(vector_3 * 4)) / 32 * 4096 + (unsigned int)(kr_row * 128) + (((unsigned int)(half_1 * 8) + lane % 8) * 8 + (unsigned int)(vector_3 * 4)) % 32 * 4 >> 7 & 3) << 5))), "r"(*reinterpret_cast<uint32_t*>(&words_3[vector_3 * 4])), "r"(*reinterpret_cast<uint32_t*>(&words_3[(vector_3 * 4) + 1])), "r"(*reinterpret_cast<uint32_t*>(&words_3[(vector_3 * 4) + 2])), "r"(*reinterpret_cast<uint32_t*>(&words_3[(vector_3 * 4) + 3])));
                                            }
                                        }
                                    }
                                }
                            }
                            if (wide_gate != 0) {
                                if (tensor_pair == 0) {
                                    #pragma unroll 1
                                    for (int element = prep_tid; element < 1024; element += 128) {
                                        smem_qk_plain[(unsigned int)(tid / 128 * 18688) + (element % 32 / 16 * 2048 + element / 32 * 64 + element % 32 % 16 * 4 ^ (element % 32 / 16 * 2048 + element / 32 * 64 + element % 32 % 16 * 4 >> 7 & 3) << 4) / 4] = 0.0f;
                                        smem_abt[(unsigned int)(tid / 128 * 18688) + (element % 32 / 32 * 4096 + element / 32 * 128 + element % 32 % 32 * 4 ^ (element % 32 / 32 * 4096 + element / 32 * 128 + element % 32 % 32 * 4 >> 7 & 3) << 5) / 4] = 0.0f;
                                    }
                                }
                                asm volatile("barrier.sync %0, 128;" :: "r"(prep_instance + 1) : "memory");
                                unsigned int wide_masks[4];
                                int wide_count = 0;
                                #pragma unroll
                                for (int mask_half = 0; mask_half < 4; mask_half++) {
                                    float mask_center = smem_diag[(unsigned int)(prep_instance * 18688) + lane + (unsigned int)(mask_half * 32)];
                                    unsigned int _vote_4 = __ballot_sync(0xFFFFFFFF, mask_center < 0.0f);
                                    wide_masks[mask_half] = _vote_4;
                                    int _popc_6 = __popc(wide_masks[mask_half]);
                                    wide_count += _popc_6;
                                }
                                if (wide_count <= 8) {
                                    #pragma unroll 1
                                    for (int row_wave = 0; row_wave < 8; row_wave++) {
                                        int target_row = row_wave * 4 + prep_warp;
                                        float2 _f2_42 = make_float2(0.0f, 0.0f);
                                        float2 column_pair = _f2_42;
                                        #pragma unroll
                                        for (int mask_half_1 = 0; mask_half_1 < 4; mask_half_1++) {
                                            unsigned int remaining_mask = wide_masks[mask_half_1];
                                            int _popc_7 = __popc(remaining_mask);
                                            int mask_count = _popc_7;
                                            #pragma unroll 1
                                            for (int selected = 0; selected < mask_count; selected++) {
                                                int _ffs_4 = __ffs(remaining_mask);
                                                int selected_feature = mask_half_1 * 32 + _ffs_4 - 1;
                                                remaining_mask = remaining_mask & remaining_mask - 1;
                                                float scalar_q = smem_qd[(unsigned int)(tid / 128 * 18688) + (selected_feature / 32 * 8192 + target_row * 128 + selected_feature % 32 * 4 ^ (selected_feature / 32 * 8192 + target_row * 128 + selected_feature % 32 * 4 >> 7 & 7) << 4) / 4];
                                                float scalar_k = smem_kd[(unsigned int)(tid / 128 * 18688) + (selected_feature / 32 * 8192 + target_row * 128 + selected_feature % 32 * 4 ^ (selected_feature / 32 * 8192 + target_row * 128 + selected_feature % 32 * 4 >> 7 & 7) << 4) / 4];
                                                float column_k = smem_kd[(unsigned int)(tid / 128 * 18688) + ((unsigned int)(selected_feature / 32 * 8192) + lane * 128 + (unsigned int)(selected_feature % 32 * 4) ^ ((unsigned int)(selected_feature / 32 * 8192) + lane * 128 + (unsigned int)(selected_feature % 32 * 4) >> 7 & 7) << 4) / 4];
                                                float relative_decay = 1.0f;
                                                if ((unsigned int)target_row > lane) {
                                                    relative_decay = smem_gate[(unsigned int)(prep_instance * 18688) + (lane + 1) * 128 + (unsigned int)selected_feature];
                                                }
                                                float _shfl_down_22 = __shfl_down_sync(0xFFFFFFFF, relative_decay, 1, 32);
                                                float later_decay = _shfl_down_22;
                                                if ((unsigned int)target_row >= lane + 1) {
                                                    relative_decay *= later_decay;
                                                }
                                                float _shfl_down_23 = __shfl_down_sync(0xFFFFFFFF, relative_decay, 2, 32);
                                                float later_decay_0 = _shfl_down_23;
                                                if ((unsigned int)target_row >= lane + 2) {
                                                    relative_decay *= later_decay_0;
                                                }
                                                float _shfl_down_24 = __shfl_down_sync(0xFFFFFFFF, relative_decay, 4, 32);
                                                float later_decay_1 = _shfl_down_24;
                                                if ((unsigned int)target_row >= lane + 4) {
                                                    relative_decay *= later_decay_1;
                                                }
                                                float _shfl_down_25 = __shfl_down_sync(0xFFFFFFFF, relative_decay, 8, 32);
                                                float later_decay_2 = _shfl_down_25;
                                                if ((unsigned int)target_row >= lane + 8) {
                                                    relative_decay *= later_decay_2;
                                                }
                                                float _shfl_down_26 = __shfl_down_sync(0xFFFFFFFF, relative_decay, 16, 32);
                                                float later_decay_3 = _shfl_down_26;
                                                if ((unsigned int)target_row >= lane + 16) {
                                                    relative_decay *= later_decay_3;
                                                }
                                                float weighted_column = column_k * relative_decay;
                                                float2 _f2_43 = make_float2(scalar_q, scalar_k);
                                                float2 _f2_44 = make_float2(weighted_column, weighted_column);
                                                column_pair = fma_f32x2_rn_ftz(_f2_43, _f2_44, column_pair);
                                            }
                                        }
                                        if ((unsigned int)target_row >= lane) {
                                            int qk_index = (unsigned int)(tid / 128 * 18688) + (lane / 16 * 2048 + (unsigned int)(target_row * 64) + lane % 16 * 4 ^ (lane / 16 * 2048 + (unsigned int)(target_row * 64) + lane % 16 * 4 >> 7 & 3) << 4) / 4;
                                            int kk_index = (unsigned int)(tid / 128 * 18688) + (lane / 32 * 4096 + (unsigned int)(target_row * 128) + lane % 32 * 4 ^ (lane / 32 * 4096 + (unsigned int)(target_row * 128) + lane % 32 * 4 >> 7 & 3) << 5) / 4;
                                            smem_qk_plain[qk_index] = smem_qk_plain[qk_index] + column_pair.x;
                                            smem_abt[kk_index] = smem_abt[kk_index] + column_pair.y;
                                        }
                                    }
                                } else {
                                    unsigned int lane_mask0 = wide_masks[0];
                                    unsigned int lane_mask1 = wide_masks[2];
                                    if (lane % 16 >= 8) {
                                        lane_mask0 = wide_masks[1];
                                        lane_mask1 = wide_masks[3];
                                    }
                                    int feature_base = lane % 16 * 4;
                                    #pragma unroll 1
                                    for (int row_wave_1 = 0; row_wave_1 < 4; row_wave_1++) {
                                        int pair_base = row_wave_1 * 8 + prep_warp * 2;
                                        int target_row_1 = (unsigned int)pair_base + lane / 16;
                                        float row_q[8];
                                        float row_k[8];
                                        float weight[8];
                                        float other_k[8];
                                        float row_gate[8];
                                        #pragma unroll
                                        for (int vector_4 = 0; vector_4 < 2; vector_4++) {
                                            int feature = feature_base + vector_4 * 64;
                                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                                : "=r"(*reinterpret_cast<uint32_t*>(&row_q[vector_4 * 4])), "=r"(*reinterpret_cast<uint32_t*>(&row_q[(vector_4 * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&row_q[(vector_4 * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&row_q[(vector_4 * 4) + 3]))
                                                : "r"((smem_qd_addr + (unsigned int)(prep_instance * 74752) + (unsigned int)(feature / 32 * 8192 + target_row_1 * 128 + feature % 32 * 4 ^ (feature / 32 * 8192 + target_row_1 * 128 + feature % 32 * 4 >> 7 & 7) << 4))));
                                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                                : "=r"(*reinterpret_cast<uint32_t*>(&row_k[vector_4 * 4])), "=r"(*reinterpret_cast<uint32_t*>(&row_k[(vector_4 * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&row_k[(vector_4 * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&row_k[(vector_4 * 4) + 3]))
                                                : "r"((smem_kd_addr + (unsigned int)(prep_instance * 74752) + (unsigned int)(feature / 32 * 8192 + target_row_1 * 128 + feature % 32 * 4 ^ (feature / 32 * 8192 + target_row_1 * 128 + feature % 32 * 4 >> 7 & 7) << 4))));
                                            int gate_col = feature;
                                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                                : "=r"(*reinterpret_cast<uint32_t*>(&row_gate[vector_4 * 4])), "=r"(*reinterpret_cast<uint32_t*>(&row_gate[(vector_4 * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&row_gate[(vector_4 * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&row_gate[(vector_4 * 4) + 3]))
                                                : "r"(smem_gate_addr + (unsigned int)(prep_instance * 74752) + (unsigned int)((target_row_1 * 128 + gate_col) * 4)));
                                            unsigned int lane_mask = lane_mask0;
                                            if (vector_4 != 0) {
                                                lane_mask = lane_mask1;
                                            }
                                            #pragma unroll
                                            for (int half_2 = 0; half_2 < 4; half_2++) {
                                                float decay = row_gate[vector_4 * 4 + half_2];
                                                weight[vector_4 * 4 + half_2] = (float)(lane_mask >> lane % 8 * 4 + (unsigned int)half_2 & 1) * decay;
                                            }
                                        }
                                        #pragma unroll 1
                                        for (int distance = 1; distance < pair_base + 2; distance++) {
                                            int target_col = target_row_1 - distance;
                                            if (target_col < 0) {
                                                target_col = 0;
                                            }
                                            #pragma unroll
                                            for (int vector_5 = 0; vector_5 < 2; vector_5++) {
                                                int feature_1 = feature_base + vector_5 * 64;
                                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                                    : "=r"(*reinterpret_cast<uint32_t*>(&other_k[vector_5 * 4])), "=r"(*reinterpret_cast<uint32_t*>(&other_k[(vector_5 * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&other_k[(vector_5 * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&other_k[(vector_5 * 4) + 3]))
                                                    : "r"((smem_kd_addr + (unsigned int)(prep_instance * 74752) + (unsigned int)(feature_1 / 32 * 8192 + target_col * 128 + feature_1 % 32 * 4 ^ (feature_1 / 32 * 8192 + target_col * 128 + feature_1 % 32 * 4 >> 7 & 7) << 4))));
                                                int gate_col_1 = feature_1;
                                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                                    : "=r"(*reinterpret_cast<uint32_t*>(&row_gate[vector_5 * 4])), "=r"(*reinterpret_cast<uint32_t*>(&row_gate[(vector_5 * 4) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&row_gate[(vector_5 * 4) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&row_gate[(vector_5 * 4) + 3]))
                                                    : "r"(smem_gate_addr + (unsigned int)(prep_instance * 74752) + (unsigned int)((target_col * 128 + gate_col_1) * 4)));
                                            }
                                            float2 _f2_45 = make_float2(0.0f, 0.0f);
                                            float2 dot_pair = _f2_45;
                                            #pragma unroll
                                            for (int half_3 = 0; half_3 < 8; half_3++) {
                                                if (target_row_1 < distance) {
                                                    weight[half_3] = 0.0f;
                                                }
                                                float weighted_k = other_k[half_3] * weight[half_3];
                                                float2 _f2_46 = make_float2(row_q[half_3], row_k[half_3]);
                                                float2 _f2_47 = make_float2(weighted_k, weighted_k);
                                                dot_pair = fma_f32x2_rn_ftz(_f2_46, _f2_47, dot_pair);
                                                float decay_1 = row_gate[half_3];
                                                weight[half_3] = weight[half_3] * decay_1;
                                            }
                                            float dot_q = dot_pair.x;
                                            float dot_k = dot_pair.y;
                                            float _shfl_xor_70 = __shfl_xor_sync(0xFFFFFFFF, dot_q, 8);
                                            dot_q += _shfl_xor_70;
                                            float _shfl_xor_71 = __shfl_xor_sync(0xFFFFFFFF, dot_k, 8);
                                            dot_k += _shfl_xor_71;
                                            float _shfl_xor_72 = __shfl_xor_sync(0xFFFFFFFF, dot_q, 4);
                                            dot_q += _shfl_xor_72;
                                            float _shfl_xor_73 = __shfl_xor_sync(0xFFFFFFFF, dot_k, 4);
                                            dot_k += _shfl_xor_73;
                                            float _shfl_xor_74 = __shfl_xor_sync(0xFFFFFFFF, dot_q, 2);
                                            dot_q += _shfl_xor_74;
                                            float _shfl_xor_75 = __shfl_xor_sync(0xFFFFFFFF, dot_k, 2);
                                            dot_k += _shfl_xor_75;
                                            float _shfl_xor_76 = __shfl_xor_sync(0xFFFFFFFF, dot_q, 1);
                                            dot_q += _shfl_xor_76;
                                            float _shfl_xor_77 = __shfl_xor_sync(0xFFFFFFFF, dot_k, 1);
                                            dot_k += _shfl_xor_77;
                                            if (lane % 16 == 0 && target_row_1 >= distance) {
                                                int qk_index_1 = (unsigned int)(tid / 128 * 18688) + (target_col / 16 * 2048 + target_row_1 * 64 + target_col % 16 * 4 ^ (target_col / 16 * 2048 + target_row_1 * 64 + target_col % 16 * 4 >> 7 & 3) << 4) / 4;
                                                int kk_index_1 = (unsigned int)(tid / 128 * 18688) + (target_col / 32 * 4096 + target_row_1 * 128 + target_col % 32 * 4 ^ (target_col / 32 * 4096 + target_row_1 * 128 + target_col % 32 * 4 >> 7 & 3) << 5) / 4;
                                                smem_qk_plain[qk_index_1] = smem_qk_plain[qk_index_1] + dot_q;
                                                smem_abt[kk_index_1] = smem_abt[kk_index_1] + dot_k;
                                            }
                                            int _vote_5 = __all_sync(0xFFFFFFFF, weight[0] == 0.0f && weight[1] == 0.0f && weight[2] == 0.0f && weight[3] == 0.0f && weight[4] == 0.0f && weight[5] == 0.0f && weight[6] == 0.0f && weight[7] == 0.0f);
                                            int exhausted = _vote_5;
                                            if (exhausted != 0) {
                                                break;
                                            }
                                        }
                                    }
                                }
                                asm volatile("barrier.sync %0, 128;" :: "r"(prep_instance + 1) : "memory");
                                if (wide_count <= 8) {
                                    int feature_ordinal = 0;
                                    #pragma unroll
                                    for (int mask_half_2 = 0; mask_half_2 < 4; mask_half_2++) {
                                        unsigned int restore_mask = wide_masks[mask_half_2];
                                        int _popc_8 = __popc(restore_mask);
                                        int restore_count = _popc_8;
                                        #pragma unroll 1
                                        for (int selected_1 = 0; selected_1 < restore_count; selected_1++) {
                                            int _ffs_5 = __ffs(restore_mask);
                                            int restore_feature = mask_half_2 * 32 + _ffs_5 - 1;
                                            restore_mask = restore_mask & restore_mask - 1;
                                            if (feature_ordinal % 4 == prep_warp) {
                                                float row_decay = smem_gate[(unsigned int)(prep_instance * 18688) + lane * 128 + (unsigned int)restore_feature];
                                                float prefix = row_decay;
                                                float _shfl_down_27 = __shfl_down_sync(0xFFFFFFFF, row_decay, 1, 32);
                                                float suffix = _shfl_down_27;
                                                if (lane == 31) {
                                                    suffix = 1.0f;
                                                }
                                                float _shfl_up_10 = __shfl_up_sync(0xFFFFFFFF, prefix, 1, 32);
                                                float earlier = _shfl_up_10;
                                                float _shfl_down_28 = __shfl_down_sync(0xFFFFFFFF, suffix, 1, 32);
                                                float later = _shfl_down_28;
                                                if (lane >= 1) {
                                                    prefix *= earlier;
                                                }
                                                if (lane + 1 < 32) {
                                                    suffix *= later;
                                                }
                                                float _shfl_up_11 = __shfl_up_sync(0xFFFFFFFF, prefix, 2, 32);
                                                float earlier_0 = _shfl_up_11;
                                                float _shfl_down_29 = __shfl_down_sync(0xFFFFFFFF, suffix, 2, 32);
                                                float later_1 = _shfl_down_29;
                                                if (lane >= 2) {
                                                    prefix *= earlier_0;
                                                }
                                                if (lane + 2 < 32) {
                                                    suffix *= later_1;
                                                }
                                                float _shfl_up_12 = __shfl_up_sync(0xFFFFFFFF, prefix, 4, 32);
                                                float earlier_2 = _shfl_up_12;
                                                float _shfl_down_30 = __shfl_down_sync(0xFFFFFFFF, suffix, 4, 32);
                                                float later_3 = _shfl_down_30;
                                                if (lane >= 4) {
                                                    prefix *= earlier_2;
                                                }
                                                if (lane + 4 < 32) {
                                                    suffix *= later_3;
                                                }
                                                float _shfl_up_13 = __shfl_up_sync(0xFFFFFFFF, prefix, 8, 32);
                                                float earlier_4 = _shfl_up_13;
                                                float _shfl_down_31 = __shfl_down_sync(0xFFFFFFFF, suffix, 8, 32);
                                                float later_5 = _shfl_down_31;
                                                if (lane >= 8) {
                                                    prefix *= earlier_4;
                                                }
                                                if (lane + 8 < 32) {
                                                    suffix *= later_5;
                                                }
                                                float _shfl_up_14 = __shfl_up_sync(0xFFFFFFFF, prefix, 16, 32);
                                                float earlier_6 = _shfl_up_14;
                                                float _shfl_down_32 = __shfl_down_sync(0xFFFFFFFF, suffix, 16, 32);
                                                float later_7 = _shfl_down_32;
                                                if (lane >= 16) {
                                                    prefix *= earlier_6;
                                                }
                                                if (lane + 16 < 32) {
                                                    suffix *= later_7;
                                                }
                                                int q_index = (unsigned int)(tid / 128 * 18688) + ((unsigned int)(restore_feature / 32 * 8192) + lane * 128 + (unsigned int)(restore_feature % 32 * 4) ^ ((unsigned int)(restore_feature / 32 * 8192) + lane * 128 + (unsigned int)(restore_feature % 32 * 4) >> 7 & 7) << 4) / 4;
                                                int k_index = (unsigned int)(tid / 128 * 18688) + ((unsigned int)(restore_feature / 32 * 8192) + lane * 128 + (unsigned int)(restore_feature % 32 * 4) ^ ((unsigned int)(restore_feature / 32 * 8192) + lane * 128 + (unsigned int)(restore_feature % 32 * 4) >> 7 & 7) << 4) / 4;
                                                float normalized_k = smem_kd[k_index];
                                                float wide_kr = normalized_k * suffix;
                                                smem_kr_store[(unsigned int)(tid / 128 * 18688) + ((unsigned int)(restore_feature / 32 * 4096) + lane * 128 + (unsigned int)(restore_feature % 32 * 4) ^ ((unsigned int)(restore_feature / 32 * 4096) + lane * 128 + (unsigned int)(restore_feature % 32 * 4) >> 7 & 3) << 5) / 4] = wide_kr;
                                                smem_qd[q_index] = smem_qd[q_index] * prefix;
                                                smem_kd[k_index] = normalized_k * prefix;
                                            }
                                            feature_ordinal += 1;
                                        }
                                    }
                                } else if (prep_tid < 128) {
                                    float center = smem_diag[prep_instance * 18688 + prep_tid];
                                    if (center < 0.0f) {
                                        float wide_suffix = 1.0f;
                                        #pragma unroll 4
                                        for (int reverse_row = 0; reverse_row < 32; reverse_row++) {
                                            int gate_row = 31 - reverse_row;
                                            float normalized_k_1 = smem_kd[(unsigned int)(tid / 128 * 18688) + (prep_tid / 32 * 8192 + gate_row * 128 + prep_tid % 32 * 4 ^ (prep_tid / 32 * 8192 + gate_row * 128 + prep_tid % 32 * 4 >> 7 & 7) << 4) / 4];
                                            float wide_kr_1 = normalized_k_1 * wide_suffix;
                                            smem_kr_store[(unsigned int)(tid / 128 * 18688) + (prep_tid / 32 * 4096 + gate_row * 128 + prep_tid % 32 * 4 ^ (prep_tid / 32 * 4096 + gate_row * 128 + prep_tid % 32 * 4 >> 7 & 3) << 5) / 4] = wide_kr_1;
                                            wide_suffix *= smem_gate[prep_instance * 18688 + gate_row * 128 + prep_tid];
                                        }
                                        float wide_prefix = 1.0f;
                                        #pragma unroll 4
                                        for (int gate_row_1 = 0; gate_row_1 < 32; gate_row_1++) {
                                            wide_prefix *= smem_gate[prep_instance * 18688 + gate_row_1 * 128 + prep_tid];
                                            int q_index_1 = (unsigned int)(tid / 128 * 18688) + (prep_tid / 32 * 8192 + gate_row_1 * 128 + prep_tid % 32 * 4 ^ (prep_tid / 32 * 8192 + gate_row_1 * 128 + prep_tid % 32 * 4 >> 7 & 7) << 4) / 4;
                                            int k_index_1 = (unsigned int)(tid / 128 * 18688) + (prep_tid / 32 * 8192 + gate_row_1 * 128 + prep_tid % 32 * 4 ^ (prep_tid / 32 * 8192 + gate_row_1 * 128 + prep_tid % 32 * 4 >> 7 & 7) << 4) / 4;
                                            smem_qd[q_index_1] = smem_qd[q_index_1] * wide_prefix;
                                            smem_kd[k_index_1] = smem_kd[k_index_1] * wide_prefix;
                                        }
                                    }
                                }
                                asm volatile("barrier.sync %0, 128;" :: "r"(prep_instance + 1) : "memory");
                                #pragma unroll 1
                                for (int element_1 = prep_tid; element_1 < 1024; element_1 += 128) {
                                    smem_inverse_cross[(unsigned int)(tid / 128 * 18688) + (element_1 % 32 / 16 * 2048 + element_1 / 32 * 64 + element_1 % 32 % 16 * 4 ^ (element_1 % 32 / 16 * 2048 + element_1 / 32 * 64 + element_1 % 32 % 16 * 4 >> 7 & 3) << 4) / 4] = smem_abt[(unsigned int)(tid / 128 * 18688) + (element_1 % 32 / 32 * 4096 + element_1 / 32 * 128 + element_1 % 32 % 32 * 4 ^ (element_1 % 32 / 32 * 4096 + element_1 / 32 * 128 + element_1 % 32 % 32 * 4 >> 7 & 3) << 5) / 4];
                                }
                            }
                            asm volatile("barrier.sync %0, 128;" :: "r"(prep_instance + 1) : "memory");
                            float kk_acc[8];
                            if (prep_warp < 3) {
                                int kk_row = ((prep_warp != 0) ? 16 : 0);
                                int kk_col = ((prep_warp == 1) ? 16 : 0);
                                int kk_row_0 = ((prep_warp == 0) ? 0 : 16);
                                int kk_col_1 = ((prep_warp == 1) ? 16 : 0);
                                #pragma unroll
                                for (int word_4 = 0; word_4 < 8; word_4++) {
                                    int row_6 = (unsigned int)kk_row_0 + lane / 4 + (unsigned int)(word_4 % 4 / 2 * 8);
                                    int col_0_2 = (unsigned int)kk_col_1 + lane % 4 * 2 + (unsigned int)(word_4 % 2) + (unsigned int)(word_4 / 4 * 8);
                                    kk_acc[word_4] = smem_inverse_cross[(unsigned int)(tid / 128 * 18688) + (col_0_2 / 16 * 2048 + row_6 * 64 + col_0_2 % 16 * 4 ^ (col_0_2 / 16 * 2048 + row_6 * 64 + col_0_2 % 16 * 4 >> 7 & 3) << 4) / 4];
                                }
                            }
                            asm volatile("barrier.sync %0, 128;" :: "r"(prep_instance + 1) : "memory");
                            if (prep_warp < 3) {
                                if (prep_warp < 2) {
                                    int row0 = lane / 4;
                                    int row1 = row0 + 8;
                                    int col0 = lane % 4 * 2;
                                    float beta0 = smem_beta[tid / 128 * 18688 + prep_warp * 16 + row0];
                                    float beta1 = smem_beta[tid / 128 * 18688 + prep_warp * 16 + row1];
                                    float l_values[8];
                                    l_values[0] = 0.0f;
                                    l_values[1] = 0.0f;
                                    l_values[2] = 0.0f;
                                    l_values[3] = 0.0f;
                                    l_values[4] = 0.0f;
                                    l_values[5] = 0.0f;
                                    l_values[6] = 0.0f;
                                    l_values[7] = 0.0f;
                                    if (row0 > col0) {
                                        l_values[0] = kk_acc[0] * beta0;
                                    }
                                    if (row0 > col0 + 1) {
                                        l_values[1] = kk_acc[1] * beta0;
                                    }
                                    if (row1 > col0) {
                                        l_values[2] = kk_acc[2] * beta1;
                                    }
                                    if (row1 > col0 + 1) {
                                        l_values[3] = kk_acc[3] * beta1;
                                    }
                                    if (row0 > col0 + 8) {
                                        l_values[4] = kk_acc[4] * beta0;
                                    }
                                    if (row0 > col0 + 9) {
                                        l_values[5] = kk_acc[5] * beta0;
                                    }
                                    if (row1 > col0 + 8) {
                                        l_values[6] = kk_acc[6] * beta1;
                                    }
                                    if (row1 > col0 + 9) {
                                        l_values[7] = kk_acc[7] * beta1;
                                    }
                                    float inverse_low[2];
                                    float inverse_high[2];
                                    {
                                        #pragma unroll
                                        for (int word_5 = 0; word_5 < 2; word_5++) {
                                            inverse_low[word_5] = 0.0f;
                                            inverse_high[word_5] = 0.0f;
                                            if (lane / 4 == lane % 4 * 2 + (unsigned int)word_5) {
                                                inverse_low[word_5] = 1.0f;
                                                inverse_high[word_5] = 1.0f;
                                            }
                                        }
                                        if (lane % 4 == 0 && lane / 4 > 0) {
                                            inverse_low[0] = -l_values[0];
                                            inverse_high[0] = -l_values[6];
                                        }
                                        #pragma unroll
                                        for (int inner = 1; inner < 7; inner++) {
                                            float _shfl_40 = __shfl_sync(0xFFFFFFFF, (l_values + 0)[inner % 2], lane / 4 * 4 + (unsigned int)(inner / 2));
                                            float low_coeff = _shfl_40;
                                            float _shfl_41 = __shfl_sync(0xFFFFFFFF, (l_values + 6)[inner % 2], lane / 4 * 4 + (unsigned int)(inner / 2));
                                            float high_coeff = _shfl_41;
                                            float _shfl_42 = __shfl_sync(0xFFFFFFFF, inverse_low[0], (unsigned int)(inner * 4) + lane % 4);
                                            float low_prior0 = _shfl_42;
                                            float _shfl_43 = __shfl_sync(0xFFFFFFFF, inverse_high[0], (unsigned int)(inner * 4) + lane % 4);
                                            float high_prior0 = _shfl_43;
                                            float _shfl_44 = __shfl_sync(0xFFFFFFFF, inverse_low[1], (unsigned int)(inner * 4) + lane % 4);
                                            float low_prior1 = _shfl_44;
                                            float _shfl_45 = __shfl_sync(0xFFFFFFFF, inverse_high[1], (unsigned int)(inner * 4) + lane % 4);
                                            float high_prior1 = _shfl_45;
                                            if (lane / 4 > (unsigned int)inner) {
                                                float _fma_46 = __fmaf_rn(-low_coeff, low_prior0, inverse_low[0]);
                                                inverse_low[0] = _fma_46;
                                                float _fma_47 = __fmaf_rn(-high_coeff, high_prior0, inverse_high[0]);
                                                inverse_high[0] = _fma_47;
                                                float _fma_48 = __fmaf_rn(-low_coeff, low_prior1, inverse_low[1]);
                                                inverse_low[1] = _fma_48;
                                                float _fma_49 = __fmaf_rn(-high_coeff, high_prior1, inverse_high[1]);
                                                inverse_high[1] = _fma_49;
                                            }
                                        }
                                    }
                                    float inverse_cross_tmp[2];
                                    float inverse_cross[2];
                                    unsigned int inv8_lhs_words[2];
                                    unsigned int inv8_rhs_words[2];
                                    {
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 2; _lp++) {
                                            inv8_lhs_words[_lp] = __float_as_uint(inverse_high[_lp + 0]);
                                        }
                                    }
                                    {
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 2; _lp++) {
                                            inv8_rhs_words[_lp] = __float_as_uint(l_values[_lp + 2]);
                                        }
                                    }
                                    float inv8_a[4];
                                    float inv8_b[2];
                                    unsigned int inv8_a_words[4];
                                    unsigned int inv8_b_words[2];
                                    float inv8_acc[4];
                                    #pragma unroll
                                    for (int word_6 = 0; word_6 < 2; word_6++) {
                                        int col_0_3 = lane % 4 + (unsigned int)(word_6 * 4);
                                        int a_lane = lane / 4 * 4 + (unsigned int)(col_0_3 / 2);
                                        float _shfl_52 = __shfl_sync(0xFFFFFFFF, reinterpret_cast<float*>(inv8_lhs_words)[0], a_lane);
                                        float a_low = _shfl_52;
                                        float _shfl_53 = __shfl_sync(0xFFFFFFFF, reinterpret_cast<float*>(inv8_lhs_words)[1], a_lane);
                                        float a_high = _shfl_53;
                                        float a_value = a_low;
                                        if ((lane & 1) != 0) {
                                            a_value = a_high;
                                        }
                                        inv8_a[word_6 * 2] = a_value;
                                        inv8_a[word_6 * 2 + 1] = a_value;
                                        int b_lane = (unsigned int)(col_0_3 * 4) + lane / 4 / 2;
                                        float _shfl_54 = __shfl_sync(0xFFFFFFFF, reinterpret_cast<float*>(inv8_rhs_words)[0], b_lane);
                                        float b_low = _shfl_54;
                                        float _shfl_55 = __shfl_sync(0xFFFFFFFF, reinterpret_cast<float*>(inv8_rhs_words)[1], b_lane);
                                        float b_high = _shfl_55;
                                        inv8_b[word_6] = b_low;
                                        if ((lane / 4 & 1) != 0) {
                                            inv8_b[word_6] = b_high;
                                        }
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 4; _lp++) {
                                        inv8_a_words[_lp] = __float_as_uint(inv8_a[_lp + 0]);
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 2; _lp++) {
                                        inv8_b_words[_lp] = __float_as_uint(inv8_b[_lp + 0]);
                                    }
                                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                                        : "=f"(inv8_acc[0]), "=f"(inv8_acc[1]), "=f"(inv8_acc[2]), "=f"(inv8_acc[3])
                                        : "r"(inv8_a_words[0]), "r"(inv8_a_words[1]), "r"(inv8_a_words[2]), "r"(inv8_a_words[3]), "r"(inv8_b_words[0]), "r"(inv8_b_words[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                                    inverse_cross_tmp[0] = inv8_acc[0];
                                    inverse_cross_tmp[1] = inv8_acc[1];
                                    unsigned int inv8_lhs_words_0[2];
                                    unsigned int inv8_rhs_words_1[2];
                                    {
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 2; _lp++) {
                                            inv8_lhs_words_0[_lp] = __float_as_uint(inverse_cross_tmp[_lp + 0]);
                                        }
                                    }
                                    {
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 2; _lp++) {
                                            inv8_rhs_words_1[_lp] = __float_as_uint(inverse_low[_lp + 0]);
                                        }
                                    }
                                    float inv8_a_2[4];
                                    float inv8_b_3[2];
                                    unsigned int inv8_a_words_4[4];
                                    unsigned int inv8_b_words_5[2];
                                    float inv8_acc_6[4];
                                    #pragma unroll
                                    for (int word_7 = 0; word_7 < 2; word_7++) {
                                        int col_0_4 = lane % 4 + (unsigned int)(word_7 * 4);
                                        int a_lane_1 = lane / 4 * 4 + (unsigned int)(col_0_4 / 2);
                                        float _shfl_56 = __shfl_sync(0xFFFFFFFF, reinterpret_cast<float*>(inv8_lhs_words_0)[0], a_lane_1);
                                        float a_low_1 = _shfl_56;
                                        float _shfl_57 = __shfl_sync(0xFFFFFFFF, reinterpret_cast<float*>(inv8_lhs_words_0)[1], a_lane_1);
                                        float a_high_1 = _shfl_57;
                                        float a_value_1 = a_low_1;
                                        if ((lane & 1) != 0) {
                                            a_value_1 = a_high_1;
                                        }
                                        inv8_a_2[word_7 * 2] = a_value_1;
                                        inv8_a_2[word_7 * 2 + 1] = a_value_1;
                                        int b_lane_1 = (unsigned int)(col_0_4 * 4) + lane / 4 / 2;
                                        float _shfl_58 = __shfl_sync(0xFFFFFFFF, reinterpret_cast<float*>(inv8_rhs_words_1)[0], b_lane_1);
                                        float b_low_1 = _shfl_58;
                                        float _shfl_59 = __shfl_sync(0xFFFFFFFF, reinterpret_cast<float*>(inv8_rhs_words_1)[1], b_lane_1);
                                        float b_high_1 = _shfl_59;
                                        inv8_b_3[word_7] = b_low_1;
                                        if ((lane / 4 & 1) != 0) {
                                            inv8_b_3[word_7] = b_high_1;
                                        }
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 4; _lp++) {
                                        inv8_a_words_4[_lp] = __float_as_uint(inv8_a_2[_lp + 0]);
                                    }
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 2; _lp++) {
                                        inv8_b_words_5[_lp] = __float_as_uint(inv8_b_3[_lp + 0]);
                                    }
                                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                                        : "=f"(inv8_acc_6[0]), "=f"(inv8_acc_6[1]), "=f"(inv8_acc_6[2]), "=f"(inv8_acc_6[3])
                                        : "r"(inv8_a_words_4[0]), "r"(inv8_a_words_4[1]), "r"(inv8_a_words_4[2]), "r"(inv8_a_words_4[3]), "r"(inv8_b_words_5[0]), "r"(inv8_b_words_5[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                                    inverse_cross[0] = inv8_acc_6[0];
                                    inverse_cross[1] = inv8_acc_6[1];
                                    float inverse[8];
                                    inverse[0] = 0.0f;
                                    inverse[1] = 0.0f;
                                    inverse[2] = 0.0f;
                                    inverse[3] = 0.0f;
                                    inverse[4] = 0.0f;
                                    inverse[5] = 0.0f;
                                    inverse[6] = 0.0f;
                                    inverse[7] = 0.0f;
                                    inverse[0] = inverse_low[0];
                                    inverse[1] = inverse_low[1];
                                    inverse[6] = inverse_high[0];
                                    inverse[7] = inverse_high[1];
                                    inverse[2] = -inverse_cross[0];
                                    inverse[3] = -inverse_cross[1];
                                    float result[8];
                                    #pragma unroll
                                    for (int word_8 = 0; word_8 < 8; word_8++) {
                                        int col_0_5 = lane % 4 * 2 + (unsigned int)(word_8 % 2) + (unsigned int)(word_8 / 4 * 8);
                                        float beta_col = smem_beta[tid / 128 * 18688 + prep_warp * 16 + col_0_5];
                                        result[word_8] = inverse[word_8] * beta_col;
                                    }
                                    unsigned int inverse_packed[8];
                                    {
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 8; _lp++) {
                                            inverse_packed[_lp] = __float_as_uint(result[_lp + 0]);
                                        }
                                    }
                                    #pragma unroll
                                    for (int word_9 = 0; word_9 < 8; word_9++) {
                                        int row_7 = lane / 4 + (unsigned int)(word_9 % 4 / 2 * 8);
                                        int col_0_6 = lane % 4 * 2 + (unsigned int)(word_9 % 2) + (unsigned int)(word_9 / 4 * 8);
                                        smem_abt[(unsigned int)(tid / 128 * 18688) + ((prep_warp * 16 + row_7) / 32 * 4096 + (prep_warp * 16 + col_0_6) * 128 + (prep_warp * 16 + row_7) % 32 * 4 ^ ((prep_warp * 16 + row_7) / 32 * 4096 + (prep_warp * 16 + col_0_6) * 128 + (prep_warp * 16 + row_7) % 32 * 4 >> 7 & 3) << 5) / 4] = reinterpret_cast<float*>(inverse_packed)[word_9];
                                    }
                                    __syncwarp();
                                } else {
                                    unsigned int words_4[8];
                                    {
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 8; _lp++) {
                                            words_4[_lp] = __float_as_uint(kk_acc[_lp + 0]);
                                        }
                                    }
                                    #pragma unroll
                                    for (int i = 0; i < 8; i++) {
                                        int row_8 = 16 + lane / 4 + (unsigned int)(i % 4 / 2 * 8);
                                        int col_0_7 = lane % 4 * 2 + (unsigned int)(i % 2) + (unsigned int)(i / 4 * 8);
                                        smem_inverse_cross[(unsigned int)(tid / 128 * 18688) + (col_0_7 / 16 * 2048 + row_8 * 64 + col_0_7 % 16 * 4 ^ (col_0_7 / 16 * 2048 + row_8 * 64 + col_0_7 % 16 * 4 >> 7 & 3) << 4) / 4] = reinterpret_cast<float*>(words_4)[i];
                                    }
                                }
                            }
                            if (prep_warp == 3) {
                                float zeros[8];
                                zeros[0] = 0.0f;
                                zeros[1] = 0.0f;
                                zeros[2] = 0.0f;
                                zeros[3] = 0.0f;
                                zeros[4] = 0.0f;
                                zeros[5] = 0.0f;
                                zeros[6] = 0.0f;
                                zeros[7] = 0.0f;
                                unsigned int words_5[8];
                                {
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 8; _lp++) {
                                        words_5[_lp] = __float_as_uint(zeros[_lp + 0]);
                                    }
                                }
                                #pragma unroll
                                for (int i_1 = 0; i_1 < 8; i_1++) {
                                    int row_9 = lane / 4 + (unsigned int)(i_1 % 4 / 2 * 8);
                                    int col_0_8 = 16 + lane % 4 * 2 + (unsigned int)(i_1 % 2) + (unsigned int)(i_1 / 4 * 8);
                                    smem_qk_plain[(unsigned int)(tid / 128 * 18688) + (col_0_8 / 16 * 2048 + row_9 * 64 + col_0_8 % 16 * 4 ^ (col_0_8 / 16 * 2048 + row_9 * 64 + col_0_8 % 16 * 4 >> 7 & 3) << 4) / 4] = reinterpret_cast<float*>(words_5)[i_1];
                                }
                                unsigned int words_0[8];
                                {
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 8; _lp++) {
                                        words_0[_lp] = __float_as_uint(zeros[_lp + 0]);
                                    }
                                }
                                #pragma unroll
                                for (int i_2 = 0; i_2 < 8; i_2++) {
                                    int row_10 = 16 + lane / 4 + (unsigned int)(i_2 % 4 / 2 * 8);
                                    int col_0_9 = lane % 4 * 2 + (unsigned int)(i_2 % 2) + (unsigned int)(i_2 / 4 * 8);
                                    smem_abt[(unsigned int)(tid / 128 * 18688) + (col_0_9 / 32 * 4096 + row_10 * 128 + col_0_9 % 32 * 4 ^ (col_0_9 / 32 * 4096 + row_10 * 128 + col_0_9 % 32 * 4 >> 7 & 3) << 5) / 4] = reinterpret_cast<float*>(words_0)[i_2];
                                }
                            }
                            asm volatile("barrier.sync %0, 128;" :: "r"(prep_instance + 1) : "memory");
                            if (prep_warp == 2) {
                                float coupling[8];
                                float a_values[4];
                                float b_values[2];
                                unsigned int a_regs[4];
                                unsigned int b_regs[2];
                                #pragma unroll
                                for (int kk = 0; kk < 2; kk++) {
                                    {
                                        int a_ld_row = lane & 15;
                                        int a_ld_col = (unsigned int)(kk * 8) + lane / 16 * 4;
                                        int a_ld_addr = (smem_abt_addr + (unsigned int)(tid / 128 * 74752) + (unsigned int)(a_ld_col / 32 * 4096 + a_ld_row * 128 + a_ld_col % 32 * 4 ^ (a_ld_col / 32 * 4096 + a_ld_row * 128 + a_ld_col % 32 * 4 >> 7 & 3) << 5));
                                        unsigned int a_ldsm_bits[4];
                                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                            : "=r"(a_ldsm_bits[0]), "=r"(a_ldsm_bits[1]), "=r"(a_ldsm_bits[2]), "=r"(a_ldsm_bits[3])
                                            : "r"(a_ld_addr)
                                            : "memory");
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 4; _lp++) {
                                            a_regs[_lp] = __float_as_uint((reinterpret_cast<float*>(a_ldsm_bits))[_lp + 0]);
                                        }
                                    }
                                    #pragma unroll
                                    for (int nn = 0; nn < 2; nn++) {
                                        {
                                            int b_ld_row = (unsigned int)(16 + nn * 8) + (lane & 7);
                                            int b_ld_col = (unsigned int)(kk * 8) + (lane / 8 & 1) * 4;
                                            int b_ld_addr = (smem_inverse_cross_addr + (unsigned int)(tid / 128 * 74752) + (unsigned int)(b_ld_col / 16 * 2048 + b_ld_row * 64 + b_ld_col % 16 * 4 ^ (b_ld_col / 16 * 2048 + b_ld_row * 64 + b_ld_col % 16 * 4 >> 7 & 3) << 4));
                                            unsigned int b_ldsm_bits[2];
                                            asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                                                : "=r"(b_ldsm_bits[0]), "=r"(b_ldsm_bits[1])
                                                : "r"(b_ld_addr)
                                                : "memory");
                                            #pragma unroll
                                            for (int _lp = 0; _lp < 2; _lp++) {
                                                b_regs[_lp] = __float_as_uint((reinterpret_cast<float*>(b_ldsm_bits))[_lp + 0]);
                                            }
                                        }
                                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                                            : "=f"((coupling + nn * 4)[0]), "=f"((coupling + nn * 4)[1]), "=f"((coupling + nn * 4)[2]), "=f"((coupling + nn * 4)[3])
                                            : "r"(a_regs[0]), "r"(a_regs[1]), "r"(a_regs[2]), "r"(a_regs[3]), "r"(b_regs[0]), "r"(b_regs[1]), "f"(((((kk == 0) ? 1 : 0)) ? 0.0f : (coupling + nn * 4)[0])), "f"(((((kk == 0) ? 1 : 0)) ? 0.0f : (coupling + nn * 4)[1])), "f"(((((kk == 0) ? 1 : 0)) ? 0.0f : (coupling + nn * 4)[2])), "f"(((((kk == 0) ? 1 : 0)) ? 0.0f : (coupling + nn * 4)[3])));
                                    }
                                }
                                unsigned int words_6[8];
                                {
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 8; _lp++) {
                                        words_6[_lp] = __float_as_uint(coupling[_lp + 0]);
                                    }
                                }
                                #pragma unroll
                                for (int i_3 = 0; i_3 < 8; i_3++) {
                                    int row_11 = lane / 4 + (unsigned int)(i_3 % 4 / 2 * 8);
                                    int col_0_10 = 16 + lane % 4 * 2 + (unsigned int)(i_3 % 2) + (unsigned int)(i_3 / 4 * 8);
                                    smem_inverse_cross[(unsigned int)(tid / 128 * 18688) + (col_0_10 / 16 * 2048 + row_11 * 64 + col_0_10 % 16 * 4 ^ (col_0_10 / 16 * 2048 + row_11 * 64 + col_0_10 % 16 * 4 >> 7 & 3) << 4) / 4] = reinterpret_cast<float*>(words_6)[i_3];
                                }
                                __syncwarp();
                                float a_values_0[4];
                                float b_values_1[2];
                                unsigned int a_regs_2[4];
                                unsigned int b_regs_3[2];
                                #pragma unroll
                                for (int kk_1 = 0; kk_1 < 2; kk_1++) {
                                    {
                                        int a_ld_row_1 = lane & 15;
                                        int a_ld_col_1 = (unsigned int)(16 + kk_1 * 8) + lane / 16 * 4;
                                        int a_ld_addr_1 = (smem_inverse_cross_addr + (unsigned int)(tid / 128 * 74752) + (unsigned int)(a_ld_col_1 / 16 * 2048 + a_ld_row_1 * 64 + a_ld_col_1 % 16 * 4 ^ (a_ld_col_1 / 16 * 2048 + a_ld_row_1 * 64 + a_ld_col_1 % 16 * 4 >> 7 & 3) << 4));
                                        unsigned int a_ldsm_bits_1[4];
                                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                            : "=r"(a_ldsm_bits_1[0]), "=r"(a_ldsm_bits_1[1]), "=r"(a_ldsm_bits_1[2]), "=r"(a_ldsm_bits_1[3])
                                            : "r"(a_ld_addr_1)
                                            : "memory");
                                        #pragma unroll
                                        for (int _lp = 0; _lp < 4; _lp++) {
                                            a_regs_2[_lp] = __float_as_uint((reinterpret_cast<float*>(a_ldsm_bits_1))[_lp + 0]);
                                        }
                                    }
                                    #pragma unroll
                                    for (int nn_1 = 0; nn_1 < 2; nn_1++) {
                                        {
                                            #pragma unroll
                                            for (int i_4 = 0; i_4 < 2; i_4++) {
                                                int bk = (unsigned int)(16 + kk_1 * 8) + lane % 4 + (unsigned int)(i_4 * 4);
                                                int bn = (unsigned int)(16 + nn_1 * 8) + lane / 4;
                                                {
                                                    b_values_1[i_4] = smem_abt[(unsigned int)(tid / 128 * 18688) + (bn / 32 * 4096 + bk * 128 + bn % 32 * 4 ^ (bn / 32 * 4096 + bk * 128 + bn % 32 * 4 >> 7 & 3) << 5) / 4];
                                                }
                                            }
                                            #pragma unroll
                                            for (int _lp = 0; _lp < 2; _lp++) {
                                                b_regs_3[_lp] = __float_as_uint(b_values_1[_lp + 0]);
                                            }
                                        }
                                        asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                                            : "=f"((coupling + nn_1 * 4)[0]), "=f"((coupling + nn_1 * 4)[1]), "=f"((coupling + nn_1 * 4)[2]), "=f"((coupling + nn_1 * 4)[3])
                                            : "r"(a_regs_2[0]), "r"(a_regs_2[1]), "r"(a_regs_2[2]), "r"(a_regs_2[3]), "r"(b_regs_3[0]), "r"(b_regs_3[1]), "f"(((((kk_1 == 0) ? 1 : 0)) ? 0.0f : (coupling + nn_1 * 4)[0])), "f"(((((kk_1 == 0) ? 1 : 0)) ? 0.0f : (coupling + nn_1 * 4)[1])), "f"(((((kk_1 == 0) ? 1 : 0)) ? 0.0f : (coupling + nn_1 * 4)[2])), "f"(((((kk_1 == 0) ? 1 : 0)) ? 0.0f : (coupling + nn_1 * 4)[3])));
                                    }
                                }
                                #pragma unroll
                                for (int i_5 = 0; i_5 < 8; i_5++) {
                                    coupling[i_5] = -coupling[i_5];
                                }
                                unsigned int words_4_1[8];
                                {
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 8; _lp++) {
                                        words_4_1[_lp] = __float_as_uint(coupling[_lp + 0]);
                                    }
                                }
                                #pragma unroll
                                for (int i_6 = 0; i_6 < 8; i_6++) {
                                    int row_12 = lane / 4 + (unsigned int)(i_6 % 4 / 2 * 8);
                                    int col_0_11 = 16 + lane % 4 * 2 + (unsigned int)(i_6 % 2) + (unsigned int)(i_6 / 4 * 8);
                                    smem_abt[(unsigned int)(tid / 128 * 18688) + (col_0_11 / 32 * 4096 + row_12 * 128 + col_0_11 % 32 * 4 ^ (col_0_11 / 32 * 4096 + row_12 * 128 + col_0_11 % 32 * 4 >> 7 & 3) << 5) / 4] = reinterpret_cast<float*>(words_4_1)[i_6];
                                }
                            }
                            asm volatile("barrier.sync %0, 128;" :: "r"(prep_instance + 1) : "memory");
                            {
                                if (prep_warp == 0) {
                                    if (elect_sync()) {
                                        mbarrier_arrive_expect_tx(v_full_addr + (prep_instance) * 8, 8192);
                                        tma_4d_gmem2smem(smem_v_raw_addr + (unsigned int)(prep_instance * 74752), v_tma, 0, (int)token_base, head_idx, 0, v_full_addr + (prep_instance) * 8);
                                    }
                                }
                            }
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            {
                                {
                                    mbarrier_arrive(factor_full_addr + (prep_instance) * 8);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    // ---- Role: recurrence_role ----
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 176;");
        { // recurrence_role_main
            unsigned int mma_phase = 0;
            unsigned int factor_epoch_bits = 0;
            {
                {
                    unsigned int solve_phase = 0;
                    int value_offset_1 = 0;
                    int seq_idx_1 = seq_order[blockIdx.x / num_heads];
                    int head_idx_1 = blockIdx.x % num_heads;
                    long long sequence_bos_1 = cu_seqlens[seq_idx_1];
                    long long bos_1 = sequence_bos_1;
                    long long sequence_eos_1 = cu_seqlens[seq_idx_1 + 1];
                    int my_chunks_1 = (int)((sequence_eos_1 - bos_1 + 32 - 1) / 32);
                    long long state_head = ((long long)seq_idx_1 * (long long)num_heads + (long long)head_idx_1) * 128 * 128;
                    long long initial_head = state_head;
                    int initial_enabled = use_initial_state;
                    {
                        int state_slot = seq_idx_1;
                        if (use_state_indices != 0) {
                            state_slot = reinterpret_cast<int*>(state_indices_addr)[seq_idx_1];
                        }
                        state_head = (long long)state_slot * state_slot_stride + (long long)head_idx_1 * 128 * 128;
                        initial_head = state_head;
                    }
                    int task_source = -1;
                    int task_destination = -1;
                    int warp_in_wg = warp % 4;
                    const int tmem_row = warp_in_wg * 32 << 16;
                    int lane_column = (lane & 3) * 2;
                    int row_in_warp = lane / 4;
                    int row_top = value_offset_1 + warp_in_wg * 32 + row_in_warp;
                    int state_buffer = 0;
                    {
                        #pragma unroll
                        for (int io_part = 0; io_part < 4; io_part++) {
                            int io_row = (unsigned int)(value_offset_1 + warp_in_wg * 32) + lane % 32;
                            int io_col = (unsigned int)(io_part * 32) + lane / 32 * 32;
                            long long io_base = initial_head + (long long)io_row * 128 + (long long)io_col;
                            float initial[32];
                            initial[0] = 0.0f;
                            initial[1] = 0.0f;
                            initial[2] = 0.0f;
                            initial[3] = 0.0f;
                            initial[4] = 0.0f;
                            initial[5] = 0.0f;
                            initial[6] = 0.0f;
                            initial[7] = 0.0f;
                            initial[8] = 0.0f;
                            initial[9] = 0.0f;
                            initial[10] = 0.0f;
                            initial[11] = 0.0f;
                            initial[12] = 0.0f;
                            initial[13] = 0.0f;
                            initial[14] = 0.0f;
                            initial[15] = 0.0f;
                            initial[16] = 0.0f;
                            initial[17] = 0.0f;
                            initial[18] = 0.0f;
                            initial[19] = 0.0f;
                            initial[20] = 0.0f;
                            initial[21] = 0.0f;
                            initial[22] = 0.0f;
                            initial[23] = 0.0f;
                            initial[24] = 0.0f;
                            initial[25] = 0.0f;
                            initial[26] = 0.0f;
                            initial[27] = 0.0f;
                            initial[28] = 0.0f;
                            initial[29] = 0.0f;
                            initial[30] = 0.0f;
                            initial[31] = 0.0f;
                            if (initial_enabled != 0) {
                                {
                                    if (((unsigned long long)initial_state_f32 + (unsigned long long)io_base * 4 & 15) == 0) {
                                        {
                                            {
                                                #pragma unroll
                                                for (int vector_6 = 0; vector_6 < 8; vector_6++) {
                                                    {
                                                        float4 _v4 = *reinterpret_cast<const float4*>(initial_state_f32 + io_base + (long long)(vector_6 % 4 * 128) - (long long)(lane % 4) * 128 + (long long)(vector_6 / 4 * 16) + (long long)(lane % 4 * 4));
                                                        initial[vector_6 * 4 + 0] = _v4.x;
                                                        initial[vector_6 * 4 + 1] = _v4.y;
                                                        initial[vector_6 * 4 + 2] = _v4.z;
                                                        initial[vector_6 * 4 + 3] = _v4.w;
                                                    }
                                                }
                                                #pragma unroll
                                                for (int segment_1 = 0; segment_1 < 2; segment_1++) {
                                                    #pragma unroll
                                                    for (int word_10 = 0; word_10 < 4; word_10++) {
                                                        unsigned int mask = -(lane & 1);
                                                        unsigned int send = reinterpret_cast<unsigned int*>(initial)[segment_1 * 16 + word_10] & mask | reinterpret_cast<unsigned int*>(initial)[segment_1 * 16 + 4 + word_10] & ~mask;
                                                        unsigned int _shfl_xor_99 = __shfl_xor_sync(0xFFFFFFFF, send, 1);
                                                        unsigned int recv = _shfl_xor_99;
                                                        reinterpret_cast<unsigned int*>(initial)[segment_1 * 16 + word_10] = recv & mask | reinterpret_cast<unsigned int*>(initial)[segment_1 * 16 + word_10] & ~mask;
                                                        reinterpret_cast<unsigned int*>(initial)[segment_1 * 16 + 4 + word_10] = reinterpret_cast<unsigned int*>(initial)[segment_1 * 16 + 4 + word_10] & mask | recv & ~mask;
                                                    }
                                                    #pragma unroll
                                                    for (int word_11 = 0; word_11 < 4; word_11++) {
                                                        unsigned int mask_1 = -(lane & 1);
                                                        unsigned int send_1 = reinterpret_cast<unsigned int*>(initial)[segment_1 * 16 + 8 + word_11] & mask_1 | reinterpret_cast<unsigned int*>(initial)[segment_1 * 16 + 12 + word_11] & ~mask_1;
                                                        unsigned int _shfl_xor_100 = __shfl_xor_sync(0xFFFFFFFF, send_1, 1);
                                                        unsigned int recv_1 = _shfl_xor_100;
                                                        reinterpret_cast<unsigned int*>(initial)[segment_1 * 16 + 8 + word_11] = recv_1 & mask_1 | reinterpret_cast<unsigned int*>(initial)[segment_1 * 16 + 8 + word_11] & ~mask_1;
                                                        reinterpret_cast<unsigned int*>(initial)[segment_1 * 16 + 12 + word_11] = reinterpret_cast<unsigned int*>(initial)[segment_1 * 16 + 12 + word_11] & mask_1 | recv_1 & ~mask_1;
                                                    }
                                                }
                                                #pragma unroll
                                                for (int segment_2 = 0; segment_2 < 2; segment_2++) {
                                                    #pragma unroll
                                                    for (int word_12 = 0; word_12 < 4; word_12++) {
                                                        unsigned int mask_2 = -(lane / 2 & 1);
                                                        unsigned int send_2 = reinterpret_cast<unsigned int*>(initial)[segment_2 * 16 + word_12] & mask_2 | reinterpret_cast<unsigned int*>(initial)[segment_2 * 16 + 8 + word_12] & ~mask_2;
                                                        unsigned int _shfl_xor_101 = __shfl_xor_sync(0xFFFFFFFF, send_2, 2);
                                                        unsigned int recv_2 = _shfl_xor_101;
                                                        reinterpret_cast<unsigned int*>(initial)[segment_2 * 16 + word_12] = recv_2 & mask_2 | reinterpret_cast<unsigned int*>(initial)[segment_2 * 16 + word_12] & ~mask_2;
                                                        reinterpret_cast<unsigned int*>(initial)[segment_2 * 16 + 8 + word_12] = reinterpret_cast<unsigned int*>(initial)[segment_2 * 16 + 8 + word_12] & mask_2 | recv_2 & ~mask_2;
                                                    }
                                                    #pragma unroll
                                                    for (int word_13 = 0; word_13 < 4; word_13++) {
                                                        unsigned int mask_3 = -(lane / 2 & 1);
                                                        unsigned int send_3 = reinterpret_cast<unsigned int*>(initial)[segment_2 * 16 + 4 + word_13] & mask_3 | reinterpret_cast<unsigned int*>(initial)[segment_2 * 16 + 12 + word_13] & ~mask_3;
                                                        unsigned int _shfl_xor_102 = __shfl_xor_sync(0xFFFFFFFF, send_3, 2);
                                                        unsigned int recv_3 = _shfl_xor_102;
                                                        reinterpret_cast<unsigned int*>(initial)[segment_2 * 16 + 4 + word_13] = recv_3 & mask_3 | reinterpret_cast<unsigned int*>(initial)[segment_2 * 16 + 4 + word_13] & ~mask_3;
                                                        reinterpret_cast<unsigned int*>(initial)[segment_2 * 16 + 12 + word_13] = reinterpret_cast<unsigned int*>(initial)[segment_2 * 16 + 12 + word_13] & mask_3 | recv_3 & ~mask_3;
                                                    }
                                                }
                                            }
                                        }
                                    } else {
                                        #pragma unroll
                                        for (int word_14 = 0; word_14 < 32; word_14++) {
                                            initial[word_14] = initial_state_f32[io_base + (long long)word_14];
                                        }
                                    }
                                }
                            }
                            {
                                #pragma unroll
                                for (int word_15 = 0; word_15 < 32; word_15++) {
                                    initial[word_15] = initial[word_15] * 3.552713678800501e-15f;
                                }
                            }
                            tmem_st_x32_f32(taddr + (unsigned int)tmem_row + (unsigned int)(io_part * 32), initial);
                        }
                    }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("barrier.sync 4, 128;" ::: "memory");
                    #pragma unroll 1
                    for (int cta_chunk_1 = 0; cta_chunk_1 < my_chunks_1; cta_chunk_1++) {
                        long long token_base_1 = bos_1 + (long long)(cta_chunk_1 * 32);
                        long long eos_1 = sequence_eos_1;
                        int factor_stage = cta_chunk_1 % 3;
                        unsigned int factor_phase = (unsigned int)(cta_chunk_1 / 3 & 1);
                        {
                            mbarrier_wait(center_ready_addr + (factor_stage) * 8, factor_phase);
                        }
                        int next_state_buffer = 0;
                        {
                            next_state_buffer = 1 - state_buffer;
                        }
                        {
                            mbarrier_wait(factor_full_addr + (factor_stage) * 8, factor_phase);
                        }
                        if (warp_in_wg == 0) {
                            int _mma_b_lo_19 = make_warp_uniform((((smem_kd_addr) >> 4) & 0x3FFF) + (factor_stage) * 4672);
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
                    "mov.b32 id, 134744336;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_prediction), "r"(_mma_b_lo_19), "r"(tmem_tmem_state + state_buffer * 128), "r"(0));
                            {
                                int _mma_b_lo_20 = make_warp_uniform((((smem_qd_addr) >> 4) & 0x3FFF) + (factor_stage) * 4672);
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
                    "mov.b32 id, 134744336;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 506;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_output), "r"(_mma_b_lo_20), "r"(tmem_tmem_state + state_buffer * 128), "r"(0));
                            }
                            elect_commit(mma_done_addr);
                        }
                        {
                            {
                                int state_row = (unsigned int)(warp_in_wg * 32) + lane;
                                #pragma unroll 1
                                for (int state_part = 0; state_part < 4; state_part++) {
                                    float _tmem_load_37[32];
                                    asm volatile(
                                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                        : "=f"(_tmem_load_37[0]), "=f"(_tmem_load_37[1]), "=f"(_tmem_load_37[2]), "=f"(_tmem_load_37[3]), "=f"(_tmem_load_37[4]), "=f"(_tmem_load_37[5]), "=f"(_tmem_load_37[6]), "=f"(_tmem_load_37[7]), "=f"(_tmem_load_37[8]), "=f"(_tmem_load_37[9]), "=f"(_tmem_load_37[10]), "=f"(_tmem_load_37[11]), "=f"(_tmem_load_37[12]), "=f"(_tmem_load_37[13]), "=f"(_tmem_load_37[14]), "=f"(_tmem_load_37[15]), "=f"(_tmem_load_37[16]), "=f"(_tmem_load_37[17]), "=f"(_tmem_load_37[18]), "=f"(_tmem_load_37[19]), "=f"(_tmem_load_37[20]), "=f"(_tmem_load_37[21]), "=f"(_tmem_load_37[22]), "=f"(_tmem_load_37[23]), "=f"(_tmem_load_37[24]), "=f"(_tmem_load_37[25]), "=f"(_tmem_load_37[26]), "=f"(_tmem_load_37[27]), "=f"(_tmem_load_37[28]), "=f"(_tmem_load_37[29]), "=f"(_tmem_load_37[30]), "=f"(_tmem_load_37[31])
                                        : "r"(taddr + (unsigned int)(warp_in_wg * 32 << 16) + (unsigned int)(state_buffer * 128) + (unsigned int)(state_part * 32)));
                                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                    {
                                        int entering_token = cta_chunk_1 * 32;
                                        if (entering_token % checkpoint_every_n_tokens == 0) {
                                            float cp_unscaled[32];
                                            {
                                                #pragma unroll
                                                for (int word_16 = 0; word_16 < 32; word_16++) {
                                                    cp_unscaled[word_16] = _tmem_load_37[word_16] * 281474976710656.0f;
                                                }
                                            }
                                            long long cp_row = checkpoint_cu_starts[seq_idx_1] + (long long)(entering_token / checkpoint_every_n_tokens);
                                            long long cp_base = ((cp_row * (long long)num_heads + (long long)head_idx_1) * 128 + (long long)state_row) * 128 + (long long)(state_part * 32);
                                            if (((unsigned long long)state_checkpoints & 15) == 0) {
                                                unsigned int cp_words[16];
                                                #pragma unroll
                                                for (int _lp = 0; _lp < 16; _lp++) {
                                                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(cp_unscaled[_lp*2 + 0], cp_unscaled[_lp*2+1 + 0]));
                                                    cp_words[_lp] = *(uint32_t*)&_bf2;
                                                }
                                                #pragma unroll
                                                for (int cp_bit = 0; cp_bit < 1; cp_bit++) {
                                                    #pragma unroll
                                                    for (int cp_pair = 0; cp_pair < 2; cp_pair++) {
                                                        const int cp_low = cp_pair / (1 << cp_bit) * (2 << cp_bit) + cp_pair % (1 << cp_bit);
                                                        const int cp_high = cp_low + (1 << cp_bit);
                                                        #pragma unroll
                                                        for (int cp_word = 0; cp_word < 4; cp_word++) {
                                                            unsigned int cp_mask = -(lane >> (unsigned int)cp_bit & 1);
                                                            unsigned int cp_send = cp_words[cp_low * 4 + cp_word] & cp_mask | cp_words[cp_high * 4 + cp_word] & ~cp_mask;
                                                            unsigned int _shfl_xor_105 = __shfl_xor_sync(0xFFFFFFFF, cp_send, 1 << cp_bit);
                                                            unsigned int cp_recv = _shfl_xor_105;
                                                            cp_words[cp_low * 4 + cp_word] = cp_recv & cp_mask | cp_words[cp_low * 4 + cp_word] & ~cp_mask;
                                                            cp_words[cp_high * 4 + cp_word] = cp_words[cp_high * 4 + cp_word] & cp_mask | cp_recv & ~cp_mask;
                                                        }
                                                    }
                                                }
                                                #pragma unroll
                                                for (int vector_7 = 0; vector_7 < 4; vector_7++) {
                                                    long long cp_offset = cp_base + (long long)(vector_7 % 2 * 128) - (long long)(lane & 1) * 128 + (long long)(vector_7 / 2 * 16) + (long long)((lane & 1) * 8);
                                                    {
                                                        int4 _iv4 = make_int4(reinterpret_cast<int*>(cp_words)[vector_7 * 4 + 0], reinterpret_cast<int*>(cp_words)[vector_7 * 4 + 1], reinterpret_cast<int*>(cp_words)[vector_7 * 4 + 2], reinterpret_cast<int*>(cp_words)[vector_7 * 4 + 3]);
                                                        *reinterpret_cast<int4*>(reinterpret_cast<int*>((unsigned long long)state_checkpoints + (unsigned long long)cp_offset * 2) + 0) = _iv4;
                                                    }
                                                }
                                            } else {
                                                #pragma unroll
                                                for (int word_17 = 0; word_17 < 32; word_17++) {
                                                    state_checkpoints[cp_base + (long long)word_17] = cp_unscaled[word_17];
                                                }
                                            }
                                        }
                                    }
                                    unsigned int carried_decay_bits[4];
                                    #pragma unroll
                                    for (int pair = 0; pair < 16; pair++) {
                                        int column = state_part * 32 + pair * 2;
                                        float2 _f2_182 = make_float2(0.0f, 0.0f);
                                        float2 center_1 = _f2_182;
                                        {
                                            if (pair % 2 == 0) {
                                                asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                                    : "=r"(*reinterpret_cast<uint32_t*>(&carried_decay_bits[0])), "=r"(*reinterpret_cast<uint32_t*>(&carried_decay_bits[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&carried_decay_bits[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&carried_decay_bits[(0) + 3]))
                                                    : "r"(smem_diag_addr + (unsigned int)(factor_stage * 74752) + (unsigned int)(column * 4)));
                                            }
                                            float2 _f2_183 = make_float2(reinterpret_cast<float*>(carried_decay_bits)[pair % 2 * 2], reinterpret_cast<float*>(carried_decay_bits)[pair % 2 * 2 + 1]);
                                            center_1 = _f2_183;
                                        }
                                        float2 _f2_186 = make_float2(_tmem_load_37[pair * 2], _tmem_load_37[pair * 2 + 1]);
                                        float2 _mul_f32x2_106;
                                        asm("mul.rn.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_106) : "l"(*(const unsigned long long*)&_f2_186), "l"(*(const unsigned long long*)&center_1));
                                        float2 value_1 = _mul_f32x2_106;
                                        _tmem_load_37[pair * 2] = value_1.x;
                                        _tmem_load_37[pair * 2 + 1] = value_1.y;
                                    }
                                    tmem_st_x32_f32(taddr + (unsigned int)(warp_in_wg * 32 << 16) + (unsigned int)(next_state_buffer * 128) + (unsigned int)(state_part * 32), _tmem_load_37);
                                }
                                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                            }
                        }
                        mbarrier_wait(mma_done_addr, mma_phase);
                        mma_phase = mma_phase ^ 1;
                        mbarrier_wait(v_full_addr + (factor_stage) * 8, factor_phase);
                        #pragma unroll 1
                        for (int row_band = 0; row_band < 2; row_band++) {
                            float v_prefetch_values[16];
                            {
                                unsigned int v_prefetch_bits[8];
                                #pragma unroll
                                for (int tile_1 = 0; tile_1 < 2; tile_1++) {
                                    int v_ld_token = (unsigned int)(tile_1 * 16) + lane / 16 * 8 + (lane & 7);
                                    int v_ld_feature = (unsigned int)(value_offset_1 + warp_in_wg * 32 + row_band * 16) + (lane / 8 & 1) * 8;
                                    int v_ld_addr = (smem_v_raw_addr + (unsigned int)(factor_stage * 74752) + (unsigned int)(v_ld_feature / 64 * 4096 + v_ld_token * 128 + v_ld_feature % 64 * 2 ^ (v_ld_feature / 64 * 4096 + v_ld_token * 128 + v_ld_feature % 64 * 2 >> 7 & 7) << 4));
                                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                                        : "=r"(v_prefetch_bits[tile_1 * 4]), "=r"(v_prefetch_bits[tile_1 * 4 + 1]), "=r"(v_prefetch_bits[tile_1 * 4 + 2]), "=r"(v_prefetch_bits[tile_1 * 4 + 3])
                                        : "r"(v_ld_addr)
                                        : "memory");
                                }
                                float v_prefetch_bits_f32[16];
                                #pragma unroll
                                for (int _pair = 0; _pair < 8; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        "shl.b32 %0, %2, 16;\n\t"
                                        "and.b32 %1, %2, 0xffff0000;\n\t"
                                        "}\n"
                                        : "=f"((&v_prefetch_bits_f32[_pair * 2])[0]), "=f"((&v_prefetch_bits_f32[_pair * 2])[1])
                                        : "r"(v_prefetch_bits[_pair]));
                                }
                                #pragma unroll
                                for (int v_word = 0; v_word < 16; v_word++) {
                                    v_prefetch_values[v_word] = v_prefetch_bits_f32[v_word];
                                }
                            }
                            float _tmem_load_41[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_41[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_41[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_41[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_41[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_41[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_41[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_41[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_41[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_41[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_41[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_41[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_41[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_41[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_41[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_41[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_41[15]))
                                : "r"(taddr + (unsigned int)tmem_row + (unsigned int)(row_band * 16 << 16) + 256));
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            float residual[16];
                            #pragma unroll
                            for (int group = 0; group < 4; group++) {
                                unsigned int v_pair_bits[2];
                                float v_pair_values[4];
                                #pragma unroll
                                for (int pair_row = 0; pair_row < 2; pair_row++) {
                                    int sr = row_top + row_band * 16 + pair_row * 8;
                                    #pragma unroll
                                    for (int adjacent = 0; adjacent < 2; adjacent++) {
                                        long long token = token_base_1 + (long long)(group * 8) + (long long)lane_column + (long long)adjacent;
                                        float value_2 = 0.0f;
                                        if (token < eos_1) {
                                            {
                                                value_2 = v_prefetch_values[group * 4 + pair_row * 2 + adjacent];
                                            }
                                        }
                                        {
                                            float _fma_56 = __fmaf_rn(_tmem_load_41[group * 4 + pair_row * 2 + adjacent], -5.421010862427522e-20f, value_2);
                                            residual[group * 4 + pair_row * 2 + adjacent] = _fma_56;
                                        }
                                    }
                                }
                            }
                            unsigned int residual_words[16];
                            {
                                #pragma unroll
                                for (int _lp = 0; _lp < 16; _lp++) {
                                    residual_words[_lp] = __float_as_uint(residual[_lp + 0]);
                                }
                            }
                            asm volatile(
                                "tcgen05.st.sync.aligned.16x256b.x4.b32"
                                " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                                :: "r"(taddr + (unsigned int)tmem_row + (unsigned int)(row_band * 16 << 16) + 256 + 32), "r"(*reinterpret_cast<const uint32_t*>(&(reinterpret_cast<float*>(residual_words))[0])), "r"(*reinterpret_cast<const uint32_t*>(&(reinterpret_cast<float*>(residual_words))[1])), "r"(*reinterpret_cast<const uint32_t*>(&(reinterpret_cast<float*>(residual_words))[2])), "r"(*reinterpret_cast<const uint32_t*>(&(reinterpret_cast<float*>(residual_words))[3])), "r"(*reinterpret_cast<const uint32_t*>(&(reinterpret_cast<float*>(residual_words))[4])), "r"(*reinterpret_cast<const uint32_t*>(&(reinterpret_cast<float*>(residual_words))[5])), "r"(*reinterpret_cast<const uint32_t*>(&(reinterpret_cast<float*>(residual_words))[6])), "r"(*reinterpret_cast<const uint32_t*>(&(reinterpret_cast<float*>(residual_words))[7])), "r"(*reinterpret_cast<const uint32_t*>(&(reinterpret_cast<float*>(residual_words))[8])), "r"(*reinterpret_cast<const uint32_t*>(&(reinterpret_cast<float*>(residual_words))[9])), "r"(*reinterpret_cast<const uint32_t*>(&(reinterpret_cast<float*>(residual_words))[10])), "r"(*reinterpret_cast<const uint32_t*>(&(reinterpret_cast<float*>(residual_words))[11])), "r"(*reinterpret_cast<const uint32_t*>(&(reinterpret_cast<float*>(residual_words))[12])), "r"(*reinterpret_cast<const uint32_t*>(&(reinterpret_cast<float*>(residual_words))[13])), "r"(*reinterpret_cast<const uint32_t*>(&(reinterpret_cast<float*>(residual_words))[14])), "r"(*reinterpret_cast<const uint32_t*>(&(reinterpret_cast<float*>(residual_words))[15])));
                        }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile("barrier.sync 4, 128;" ::: "memory");
                        {
                            if (warp_in_wg == 0) {
                                int _mma_b_lo_22 = make_warp_uniform(((((smem_inverse_out_addr) >> 4) & 0x3FFF) | 0x1000000) + (factor_stage) * 4672);
                                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x20004020;\n\t"
                    "mov.b32 id, 134809872;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_solved), "r"(_mma_b_lo_22), "r"(tmem_tmem_residual), "r"(0));
                                elect_commit(mma_done_addr);
                            }
                            mbarrier_wait(mma_done_addr, mma_phase);
                            mma_phase = mma_phase ^ 1;
                        }
                        {
                            asm volatile("barrier.sync 4, 128;" ::: "memory");
                        }
                        if (warp_in_wg == 0) {
                            {
                                int _mma_b_lo_23 = make_warp_uniform(((((smem_w_out_addr) >> 4) & 0x3FFF) | 0x1000000) + (factor_stage) * 4672);
                                asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x20004020;\n\t"
                    "mov.b32 id, 136382736;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 64;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"((tmem_tmem_state + (next_state_buffer * 128))), "r"(_mma_b_lo_23), "r"(tmem_tmem_solved), "r"(1));
                                {
                                    int _mma_b_lo_24 = make_warp_uniform((((smem_qk_out_addr) >> 4) & 0x3FFF) + (factor_stage) * 4672);
                                    asm volatile(
                    "{\n\t"
                    ".reg .pred leader, p0, p1;\n\t"
                    ".reg .b32 dhi, blo, ta, id;\n\t"
                    ".reg .b64 db;\n\t"
                    "elect.sync _|leader, 0xFFFFFFFF;\n\t"
                    "setp.ne.b32 p0, %3, 0;\n\t"
                    "setp.ne.b32 p1, 1, 0;\n\t"
                    ""
                    "mov.b32 dhi, 0x80004020;\n\t"
                    "mov.b32 id, 134744336;\n\t"
                    "mov.b32 ta, %2;\n\t"
                    "mov.b32 blo, %1;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p0;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 126;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "add.u32 ta, ta, 8;\n\t"
                    "add.u32 blo, blo, 2;\n\t"
                    "mov.b64 db, {blo, dhi};\n\t"
                    "@leader tcgen05.mma.cta_group::1.kind::tf32 [%0], [ta], db, id, p1;\n\t"
                    "}\n"
                    :: "r"(tmem_tmem_output), "r"(_mma_b_lo_24), "r"(tmem_tmem_solved), "r"(1));
                                }
                            }
                            elect_commit(mma_done_addr);
                        }
                        mbarrier_wait(mma_done_addr, mma_phase);
                        mma_phase = mma_phase ^ 1;
                        {
                            float output_scale = scale;
                            {
                                output_scale = scale * 5.421010862427522e-20f;
                            }
                            int output_stage = 0;
                            {
                                output_stage = factor_stage;
                            }
                            if (eos_1 >= token_base_1 + 32 && ((unsigned long long)out & 15) == 0) {
                                #pragma unroll
                                for (int output_band = 0; output_band < 2; output_band++) {
                                    float _tmem_load_43[16];
                                    asm volatile(
                                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_43[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_43[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_43[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_43[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_43[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_43[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_43[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_43[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_43[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_43[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_43[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_43[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_43[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_43[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_43[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_43[15]))
                                        : "r"(taddr + (unsigned int)tmem_row + (unsigned int)(output_band * 16 << 16) + 256 + 64));
                                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                    const float2 _scale2_1 = {output_scale, output_scale};
                                    #pragma unroll
                                    for (int _ls = 0; _ls < 8; _ls++)
                                        mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_43)[_ls], _scale2_1);
                                    unsigned int result_packed[8];
                                    #pragma unroll
                                    for (int _lp = 0; _lp < 8; _lp++) {
                                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_43[_lp*2 + 0], _tmem_load_43[_lp*2+1 + 0]));
                                        result_packed[_lp] = *(uint32_t*)&_bf2;
                                    }
                                    #pragma unroll
                                    for (int output_group = 0; output_group < 2; output_group++) {
                                        int matrix = lane / 8;
                                        int output_dim = warp_in_wg * 32 + output_band * 16 + (matrix & 1) * 8;
                                        int output_tok = (unsigned int)(output_group * 16 + matrix / 2 * 8) + (lane & 7);
                                        int output_addr = (smem_output_addr + (unsigned int)(output_stage * 74752) + (unsigned int)(output_dim / 64 * 4096 + output_tok * 128 + output_dim % 64 * 2 ^ (output_dim / 64 * 4096 + output_tok * 128 + output_dim % 64 * 2 >> 7 & 7) << 4));
                                        uint32_t _stmatrix_addr_2 = static_cast<uint32_t>((unsigned long long)output_addr);
                                        asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                                            :: "r"(_stmatrix_addr_2), "r"(*reinterpret_cast<const uint32_t*>(&result_packed[output_group * 4])), "r"(*reinterpret_cast<const uint32_t*>(&result_packed[output_group * 4 + 1])), "r"(*reinterpret_cast<const uint32_t*>(&result_packed[output_group * 4 + 2])), "r"(*reinterpret_cast<const uint32_t*>(&result_packed[output_group * 4 + 3]))
                                            : "memory");
                                    }
                                }
                                asm volatile("barrier.sync 4, 128;" ::: "memory");
                                int output_lane = (unsigned int)(warp_in_wg * 32) + lane;
                                #pragma unroll
                                for (int output_part = 0; output_part < 4; output_part++) {
                                    int output_tok_1 = output_part * 8 + output_lane / 16;
                                    int output_dim_1 = output_lane % 16 * 8;
                                    int output_addr_1 = (smem_output_addr + (unsigned int)(output_stage * 74752) + (unsigned int)(output_dim_1 / 64 * 4096 + output_tok_1 * 128 + output_dim_1 % 64 * 2 ^ (output_dim_1 / 64 * 4096 + output_tok_1 * 128 + output_dim_1 % 64 * 2 >> 7 & 7) << 4));
                                    unsigned int output_words[4];
                                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&output_words[0])), "=r"(*reinterpret_cast<uint32_t*>(&output_words[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&output_words[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&output_words[(0) + 3]))
                                        : "r"(output_addr_1));
                                    long long output_offset = ((token_base_1 + (long long)output_tok_1) * (long long)num_heads + (long long)head_idx_1) * 128 + (long long)output_dim_1;
                                    {
                                        int4 _iv4 = make_int4(reinterpret_cast<int*>(output_words)[0 + 0], reinterpret_cast<int*>(output_words)[0 + 1], reinterpret_cast<int*>(output_words)[0 + 2], reinterpret_cast<int*>(output_words)[0 + 3]);
                                        *reinterpret_cast<int4*>(reinterpret_cast<int*>((unsigned long long)out + (unsigned long long)output_offset * 2) + 0) = _iv4;
                                    }
                                }
                            } else if (0) {
                                float _tmem_load_44[32];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                    : "=f"(_tmem_load_44[0]), "=f"(_tmem_load_44[1]), "=f"(_tmem_load_44[2]), "=f"(_tmem_load_44[3]), "=f"(_tmem_load_44[4]), "=f"(_tmem_load_44[5]), "=f"(_tmem_load_44[6]), "=f"(_tmem_load_44[7]), "=f"(_tmem_load_44[8]), "=f"(_tmem_load_44[9]), "=f"(_tmem_load_44[10]), "=f"(_tmem_load_44[11]), "=f"(_tmem_load_44[12]), "=f"(_tmem_load_44[13]), "=f"(_tmem_load_44[14]), "=f"(_tmem_load_44[15]), "=f"(_tmem_load_44[16]), "=f"(_tmem_load_44[17]), "=f"(_tmem_load_44[18]), "=f"(_tmem_load_44[19]), "=f"(_tmem_load_44[20]), "=f"(_tmem_load_44[21]), "=f"(_tmem_load_44[22]), "=f"(_tmem_load_44[23]), "=f"(_tmem_load_44[24]), "=f"(_tmem_load_44[25]), "=f"(_tmem_load_44[26]), "=f"(_tmem_load_44[27]), "=f"(_tmem_load_44[28]), "=f"(_tmem_load_44[29]), "=f"(_tmem_load_44[30]), "=f"(_tmem_load_44[31])
                                    : "r"(taddr + (unsigned int)(warp_in_wg * 32 << 16) + 256 + 64));
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                int output_row = (unsigned int)(value_offset_1 + warp_in_wg * 32) + lane;
                                #pragma unroll
                                for (int output_col = 0; output_col < 32; output_col++) {
                                    long long token_1 = token_base_1 + (long long)output_col;
                                    if (token_1 < eos_1) {
                                        {
                                            out[(token_1 * (long long)num_heads + (long long)head_idx_1) * 128 + (long long)output_row] = _tmem_load_44[output_col] * output_scale;
                                        }
                                    }
                                }
                            } else {
                                #pragma unroll 1
                                for (int row_band_1 = 0; row_band_1 < 2; row_band_1++) {
                                    float _tmem_load_45[16];
                                    asm volatile(
                                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_45[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_45[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_45[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_45[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_45[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_45[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_45[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_45[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_45[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_45[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_45[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_45[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_45[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_45[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_45[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_45[15]))
                                        : "r"(taddr + (unsigned int)tmem_row + (unsigned int)(row_band_1 * 16 << 16) + 256 + 64));
                                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                    #pragma unroll
                                    for (int group_1 = 0; group_1 < 4; group_1++) {
                                        #pragma unroll
                                        for (int pair_row_1 = 0; pair_row_1 < 2; pair_row_1++) {
                                            int sr_1 = row_top + row_band_1 * 16 + pair_row_1 * 8;
                                            #pragma unroll
                                            for (int adjacent_1 = 0; adjacent_1 < 2; adjacent_1++) {
                                                long long token_2 = token_base_1 + (long long)(group_1 * 8) + (long long)lane_column + (long long)adjacent_1;
                                                if (token_2 < eos_1) {
                                                    {
                                                        out[(token_2 * (long long)num_heads + (long long)head_idx_1) * 128 + (long long)sr_1] = _tmem_load_45[group_1 * 4 + pair_row_1 * 2 + adjacent_1] * output_scale;
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        state_buffer = next_state_buffer;
                        asm volatile("barrier.sync 4, 128;" ::: "memory");
                        {
                            mbarrier_arrive(factor_free_addr + (factor_stage) * 8);
                        }
                    }
                    if (store_final_state != 0 || task_destination >= 0) {
                        {
                            #pragma unroll
                            for (int io_part_1 = 0; io_part_1 < 4; io_part_1++) {
                                int io_row_1 = (unsigned int)(value_offset_1 + warp_in_wg * 32) + lane % 32;
                                int io_col_1 = (unsigned int)(io_part_1 * 32) + lane / 32 * 32;
                                long long io_base_1 = state_head + (long long)io_row_1 * 128 + (long long)io_col_1;
                                float _tmem_load_46[32];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                    : "=f"(_tmem_load_46[0]), "=f"(_tmem_load_46[1]), "=f"(_tmem_load_46[2]), "=f"(_tmem_load_46[3]), "=f"(_tmem_load_46[4]), "=f"(_tmem_load_46[5]), "=f"(_tmem_load_46[6]), "=f"(_tmem_load_46[7]), "=f"(_tmem_load_46[8]), "=f"(_tmem_load_46[9]), "=f"(_tmem_load_46[10]), "=f"(_tmem_load_46[11]), "=f"(_tmem_load_46[12]), "=f"(_tmem_load_46[13]), "=f"(_tmem_load_46[14]), "=f"(_tmem_load_46[15]), "=f"(_tmem_load_46[16]), "=f"(_tmem_load_46[17]), "=f"(_tmem_load_46[18]), "=f"(_tmem_load_46[19]), "=f"(_tmem_load_46[20]), "=f"(_tmem_load_46[21]), "=f"(_tmem_load_46[22]), "=f"(_tmem_load_46[23]), "=f"(_tmem_load_46[24]), "=f"(_tmem_load_46[25]), "=f"(_tmem_load_46[26]), "=f"(_tmem_load_46[27]), "=f"(_tmem_load_46[28]), "=f"(_tmem_load_46[29]), "=f"(_tmem_load_46[30]), "=f"(_tmem_load_46[31])
                                    : "r"(taddr + (unsigned int)tmem_row + (unsigned int)(state_buffer * 128) + (unsigned int)(io_part_1 * 32)));
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                {
                                    #pragma unroll
                                    for (int word_18 = 0; word_18 < 32; word_18++) {
                                        _tmem_load_46[word_18] = _tmem_load_46[word_18] * 281474976710656.0f;
                                    }
                                }
                                {
                                    if (((unsigned long long)final_state_f32 + (unsigned long long)io_base_1 * 4 & 15) == 0) {
                                        {
                                            #pragma unroll
                                            for (int ep_pair = 0; ep_pair < 4; ep_pair++) {
                                                #pragma unroll
                                                for (int ep_word = 0; ep_word < 4; ep_word++) {
                                                    unsigned int ep_mask = -(lane & 1);
                                                    unsigned int ep_send = reinterpret_cast<unsigned int*>(_tmem_load_46)[ep_pair * 8 + ep_word] & ep_mask | reinterpret_cast<unsigned int*>(_tmem_load_46)[ep_pair * 8 + 4 + ep_word] & ~ep_mask;
                                                    unsigned int _shfl_xor_107 = __shfl_xor_sync(0xFFFFFFFF, ep_send, 1);
                                                    unsigned int ep_recv = _shfl_xor_107;
                                                    reinterpret_cast<unsigned int*>(_tmem_load_46)[ep_pair * 8 + ep_word] = ep_recv & ep_mask | reinterpret_cast<unsigned int*>(_tmem_load_46)[ep_pair * 8 + ep_word] & ~ep_mask;
                                                    reinterpret_cast<unsigned int*>(_tmem_load_46)[ep_pair * 8 + 4 + ep_word] = reinterpret_cast<unsigned int*>(_tmem_load_46)[ep_pair * 8 + 4 + ep_word] & ep_mask | ep_recv & ~ep_mask;
                                                }
                                            }
                                            #pragma unroll
                                            for (int vector_8 = 0; vector_8 < 8; vector_8++) {
                                                {
                                                    float4 _v4 = make_float4(_tmem_load_46[vector_8 * 4 + 0], _tmem_load_46[vector_8 * 4 + 1], _tmem_load_46[vector_8 * 4 + 2], _tmem_load_46[vector_8 * 4 + 3]);
                                                    *reinterpret_cast<float4*>(final_state_f32 + (io_base_1 + (long long)(vector_8 % 2 * 128) + (-((long long)(lane & 1) * 128)) + (long long)(vector_8 / 2 * 8) + (long long)((lane & 1) * 4)) + 0) = _v4;
                                                }
                                            }
                                        }
                                    } else {
                                        #pragma unroll
                                        for (int word_19 = 0; word_19 < 32; word_19++) {
                                            final_state_f32[io_base_1 + (long long)word_19] = _tmem_load_46[word_19];
                                        }
                                    }
                                }
                            }
                        }
                    }
                    asm volatile("barrier.sync 4, 128;" ::: "memory");
                }
            }
            int warp_in_wg_1 = warp % 4;
            if (warp_in_wg_1 == 0) {
                int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
                asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(512));
            }
        }
    }

    // Cleanup
}

} // extern "C"
