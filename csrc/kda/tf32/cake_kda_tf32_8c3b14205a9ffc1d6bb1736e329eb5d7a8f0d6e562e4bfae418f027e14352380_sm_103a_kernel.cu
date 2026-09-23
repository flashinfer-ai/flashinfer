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
#define SMEM_PACKET_SMEM_STAGE_BYTES 53248
#define SMEM_PACKET_SMEM_STRIDE 74752
#define SMEM_PACKET_TAIL_SMEM_OFF 70656
#define SMEM_PACKET_TAIL_SMEM_STAGE_BYTES 5120
#define SMEM_PACKET_TAIL_SMEM_STRIDE 74752
#define SMEM_SMEM_CHECKPOINT_OFF 1024
#define SMEM_SMEM_CHECKPOINT_STAGE_BYTES 32768
#define SMEM_SMEM_CHECKPOINT_STRIDE 32768
#define SMEM_SMEM_OUTPUT_OFF 1024
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
kernel_cake_kda_tf32_8c3b14205a9ffc1d6bb1736e329eb5d7a8f0d6e562e4bfae418f027e14352380(__nv_bfloat16* __restrict__ q, CakeTensorMap const* q_tma, __nv_bfloat16* __restrict__ k, CakeTensorMap const* k_tma, __nv_bfloat16* __restrict__ raw_gate, CakeTensorMap const* raw_gate_tma, __nv_bfloat16* __restrict__ beta_logits, float* __restrict__ beta_active_f32, long long affine_cache_token_offset, int affine_cache_part_offset, CakeTensorMap const* beta_logits_tma, float* __restrict__ a_log, float* __restrict__ dt_bias, long long* __restrict__ cu_seqlens, int* __restrict__ seq_order, __nv_bfloat16* __restrict__ v, CakeTensorMap const* v_tma, __nv_bfloat16* __restrict__ out, __nv_bfloat16* __restrict__ initial_state, __nv_bfloat16* __restrict__ final_state, float* __restrict__ initial_state_f32, float* __restrict__ final_state_f32, unsigned long long state_indices_addr, long long state_slot_stride, int use_state_indices, int use_initial_state, int store_final_state, CakeTensorMap const* state_checkpoints_tma, __nv_bfloat16* __restrict__ state_checkpoints, long long* __restrict__ checkpoint_cu_starts, int checkpoint_every_n_tokens, float scale, int num_heads, float gate_lower_bound, long long beta_token_stride, int* __restrict__ task_ids, int* __restrict__ task_offsets, int* __restrict__ task_token_starts, int* __restrict__ task_token_counts, int* __restrict__ task_state_sources, int* __restrict__ task_state_destinations, float* __restrict__ mid_state_f32, unsigned int* __restrict__ mid_state_ready, CakeTensorMap const* owner_packet_tma, CakeTensorMap const* owner_packet_tail_tma, float* __restrict__ map_output_f32)
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
    #define packet_full_addr (mbar_base + 24)
    #define v_full_addr (mbar_base + 48)
    #define mma_done_addr (mbar_base + 72)
    #define factor_full_addr (mbar_base + 80)
    #define factor_free_addr (mbar_base + 104)
    #define gate_raw_full_addr (mbar_base + 128)
    #define qk_raw_full_addr (mbar_base + 176)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
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
    __nv_bfloat16* smem_output = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_output_addr = smem + 1024;
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
        // packet_full: 3 barriers, init_count=1
        // v_full: 3 barriers, init_count=1
        // mma_done: 1 barriers, init_count=1
        // factor_full: 3 barriers, init_count=128
        // factor_free: 3 barriers, init_count=128
        // --- pipeline 'raw_pipe' ---
        // gate_raw_full: 6 barriers, init_count=1
        // qk_raw_full: 6 barriers, init_count=1
        // Warp-cooperative initialization in physical record order.
        uint32_t _mbarrier_init_count_0_0 = 1;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(16), "r"((uint32_t)(128)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_0_0) : "r"(lane), "n"(10), "r"((uint32_t)(1)));
        if (lane < 28) {
            mbarrier_init(smem + 0 + lane * 8, _mbarrier_init_count_0_0);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    __syncwarp();

    // TMEM alloc (512 columns, 480 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 224);
    if (warp == 0) {
        int _tmem_hold = smem + 224;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

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
        asm volatile("setmaxnreg.dec.sync.aligned.u32 128;");
        { // prepare_role_main
            int raw_epoch = 0;
            {
                int task = blockIdx.x;
                int seq = seq_order[task / num_heads];
                int head = task % num_heads;
                long long bos = cu_seqlens[seq];
                int chunks = (int)((cu_seqlens[seq + 1] - bos + 32 - 1) / 32);
                int stage = tid / 128;
                int tid_0 = tid % 128;
                #pragma unroll 1
                for (int chunk = stage; chunk < chunks; chunk += 3) {
                    int epoch = chunk / 3;
                    mbarrier_wait(factor_free_addr + (stage) * 8, (unsigned int)(epoch & 1) ^ 1);
                    if (tid_0 == 0) {
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive_expect_tx(packet_full_addr + (stage) * 8, 58368);
                        tma_2d_gmem2smem(packet_smem_addr + (unsigned int)(stage * 74752), owner_packet_tma, 0, (int)((((bos + affine_cache_token_offset) / 32 + (long long)chunk + (long long)seq + (long long)affine_cache_part_offset) * (long long)num_heads + (long long)head) * 114), packet_full_addr + (stage) * 8);
                        tma_2d_gmem2smem(packet_tail_smem_addr + (unsigned int)(stage * 74752), owner_packet_tail_tma, 0, (int)((((bos + affine_cache_token_offset) / 32 + (long long)chunk + (long long)seq + (long long)affine_cache_part_offset) * (long long)num_heads + (long long)head) * 114 + 104), packet_full_addr + (stage) * 8);
                    }
                    mbarrier_wait(packet_full_addr + (stage) * 8, (unsigned int)(epoch & 1));
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    if (tid_0 == 0) {
                        mbarrier_arrive(v_full_addr + (stage) * 8);
                    }
                    mbarrier_arrive(factor_full_addr + (stage) * 8);
                }
            }
        }
    }
    // ---- Role: recurrence_role ----
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 128;");
        { // recurrence_role_main
            unsigned int mma_phase = 0;
            unsigned int factor_epoch_bits = 0;
            {
                {
                    unsigned int solve_phase = 0;
                    int value_offset = 0;
                    int seq_idx = seq_order[blockIdx.x / num_heads];
                    int head_idx = blockIdx.x % num_heads;
                    long long sequence_bos = cu_seqlens[seq_idx];
                    long long bos_1 = sequence_bos;
                    long long sequence_eos = cu_seqlens[seq_idx + 1];
                    int my_chunks = (int)((sequence_eos - bos_1 + 32 - 1) / 32);
                    long long state_head = ((long long)seq_idx * (long long)num_heads + (long long)head_idx) * 128 * 128;
                    long long initial_head = state_head;
                    int initial_enabled = use_initial_state;
                    {
                        int state_slot = seq_idx;
                        if (use_state_indices != 0) {
                            state_slot = reinterpret_cast<int*>(state_indices_addr)[seq_idx];
                        }
                        state_head = (long long)state_slot * state_slot_stride + (long long)head_idx * 128 * 128;
                        initial_head = state_head;
                    }
                    int task_source = -1;
                    int task_destination = -1;
                    int warp_in_wg = warp % 4;
                    const int tmem_row = warp_in_wg * 32 << 16;
                    int lane_column = (lane & 3) * 2;
                    int row_in_warp = lane / 4;
                    int row_top = value_offset + warp_in_wg * 32 + row_in_warp;
                    int state_buffer = 0;
                    {
                        #pragma unroll 1
                        for (int row_band = 0; row_band < 2; row_band++) {
                            #pragma unroll
                            for (int half = 0; half < 2; half++) {
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
                                    #pragma unroll
                                    for (int group = 0; group < 8; group++) {
                                        int column = half * 64 + group * 8 + lane_column;
                                        #pragma unroll
                                        for (int pair_row = 0; pair_row < 2; pair_row++) {
                                            int sr = row_top + row_band * 16 + pair_row * 8;
                                            {
                                                {
                                                    #pragma unroll
                                                    for (int adjacent = 0; adjacent < 2; adjacent++) {
                                                        long long si = initial_head + (long long)sr * 128 + (long long)column + (long long)adjacent;
                                                        if (task_source >= 0) {
                                                            initial[group * 4 + pair_row * 2 + adjacent] = mid_state_f32[(long long)task_source * 128 * 128 + (long long)sr * 128 + (long long)column + (long long)adjacent];
                                                        } else if (0) {
                                                            initial[group * 4 + pair_row * 2 + adjacent] = (float)initial_state[si];
                                                        } else {
                                                            {
                                                                initial[group * 4 + pair_row * 2 + adjacent] = initial_state_f32[si];
                                                            }
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                asm volatile(
                                    "tcgen05.st.sync.aligned.16x256b.x8.b32"
                                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                                    :: "r"(taddr + (unsigned int)tmem_row + (unsigned int)(row_band * 16 << 16) + (unsigned int)(half * 64)), "r"(*reinterpret_cast<const uint32_t*>(&initial[0])), "r"(*reinterpret_cast<const uint32_t*>(&initial[1])), "r"(*reinterpret_cast<const uint32_t*>(&initial[2])), "r"(*reinterpret_cast<const uint32_t*>(&initial[3])), "r"(*reinterpret_cast<const uint32_t*>(&initial[4])), "r"(*reinterpret_cast<const uint32_t*>(&initial[5])), "r"(*reinterpret_cast<const uint32_t*>(&initial[6])), "r"(*reinterpret_cast<const uint32_t*>(&initial[7])), "r"(*reinterpret_cast<const uint32_t*>(&initial[8])), "r"(*reinterpret_cast<const uint32_t*>(&initial[9])), "r"(*reinterpret_cast<const uint32_t*>(&initial[10])), "r"(*reinterpret_cast<const uint32_t*>(&initial[11])), "r"(*reinterpret_cast<const uint32_t*>(&initial[12])), "r"(*reinterpret_cast<const uint32_t*>(&initial[13])), "r"(*reinterpret_cast<const uint32_t*>(&initial[14])), "r"(*reinterpret_cast<const uint32_t*>(&initial[15])), "r"(*reinterpret_cast<const uint32_t*>(&initial[16])), "r"(*reinterpret_cast<const uint32_t*>(&initial[17])), "r"(*reinterpret_cast<const uint32_t*>(&initial[18])), "r"(*reinterpret_cast<const uint32_t*>(&initial[19])), "r"(*reinterpret_cast<const uint32_t*>(&initial[20])), "r"(*reinterpret_cast<const uint32_t*>(&initial[21])), "r"(*reinterpret_cast<const uint32_t*>(&initial[22])), "r"(*reinterpret_cast<const uint32_t*>(&initial[23])), "r"(*reinterpret_cast<const uint32_t*>(&initial[24])), "r"(*reinterpret_cast<const uint32_t*>(&initial[25])), "r"(*reinterpret_cast<const uint32_t*>(&initial[26])), "r"(*reinterpret_cast<const uint32_t*>(&initial[27])), "r"(*reinterpret_cast<const uint32_t*>(&initial[28])), "r"(*reinterpret_cast<const uint32_t*>(&initial[29])), "r"(*reinterpret_cast<const uint32_t*>(&initial[30])), "r"(*reinterpret_cast<const uint32_t*>(&initial[31])));
                            }
                        }
                    }
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("barrier.sync 4, 128;" ::: "memory");
                    #pragma unroll 1
                    for (int cta_chunk = 0; cta_chunk < my_chunks; cta_chunk++) {
                        long long token_base = bos_1 + (long long)(cta_chunk * 32);
                        long long eos = sequence_eos;
                        int factor_stage = cta_chunk % 3;
                        unsigned int factor_phase = (unsigned int)(cta_chunk / 3 & 1);
                        {
                        }
                        {
                            mbarrier_wait(factor_full_addr + (factor_stage) * 8, factor_phase);
                        }
                        int next_state_buffer = 0;
                        {
                            next_state_buffer = 1 - state_buffer;
                        }
                        {
                            {
                                #pragma unroll 1
                                for (int row_band_1 = 0; row_band_1 < 2; row_band_1++) {
                                    #pragma unroll
                                    for (int half_1 = 0; half_1 < 2; half_1++) {
                                        float _tmem_load_36[32];
                                        asm volatile(
                                            "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_36[31]))
                                            : "r"(taddr + (unsigned int)tmem_row + (unsigned int)(row_band_1 * 16 << 16) + (unsigned int)(state_buffer * 128) + (unsigned int)(half_1 * 64)));
                                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                        #pragma unroll
                                        for (int group_1 = 0; group_1 < 8; group_1++) {
                                            int column_1 = half_1 * 64 + group_1 * 8 + lane_column;
                                            float2 _f2_171 = make_float2(smem_diag[factor_stage * 18688 + column_1], smem_diag[factor_stage * 18688 + column_1 + 1]);
                                            float2 decay_pair = _f2_171;
                                            #pragma unroll
                                            for (int pair_row_1 = 0; pair_row_1 < 2; pair_row_1++) {
                                                float2 _f2_174 = make_float2(_tmem_load_36[group_1 * 4 + pair_row_1 * 2], _tmem_load_36[group_1 * 4 + pair_row_1 * 2 + 1]);
                                                float2 state_pair = _f2_174;
                                                {
                                                    float2 _mul_f32x2_100;
                                                    asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_100) : "l"(*(const unsigned long long*)&state_pair), "l"(*(const unsigned long long*)&decay_pair));
                                                    state_pair = _mul_f32x2_100;
                                                }
                                                _tmem_load_36[group_1 * 4 + pair_row_1 * 2] = state_pair.x;
                                                _tmem_load_36[group_1 * 4 + pair_row_1 * 2 + 1] = state_pair.y;
                                            }
                                        }
                                        asm volatile(
                                            "tcgen05.st.sync.aligned.16x256b.x8.b32"
                                            " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32};"
                                            :: "r"(taddr + (unsigned int)tmem_row + (unsigned int)(row_band_1 * 16 << 16) + (unsigned int)(next_state_buffer * 128) + (unsigned int)(half_1 * 64)), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_36[0])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_36[1])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_36[2])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_36[3])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_36[4])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_36[5])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_36[6])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_36[7])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_36[8])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_36[9])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_36[10])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_36[11])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_36[12])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_36[13])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_36[14])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_36[15])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_36[16])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_36[17])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_36[18])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_36[19])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_36[20])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_36[21])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_36[22])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_36[23])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_36[24])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_36[25])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_36[26])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_36[27])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_36[28])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_36[29])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_36[30])), "r"(*reinterpret_cast<const uint32_t*>(&_tmem_load_36[31])));
                                    }
                                }
                                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                                asm volatile("barrier.sync 4, 128;" ::: "memory");
                            }
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
                        mbarrier_wait(mma_done_addr, mma_phase);
                        mma_phase = mma_phase ^ 1;
                        mbarrier_wait(v_full_addr + (factor_stage) * 8, factor_phase);
                        #pragma unroll 1
                        for (int row_band_2 = 0; row_band_2 < 2; row_band_2++) {
                            float v_prefetch_values[16];
                            float _tmem_load_41[16];
                            asm volatile(
                                "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_41[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_41[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_41[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_41[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_41[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_41[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_41[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_41[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_41[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_41[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_41[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_41[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_41[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_41[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_41[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_41[15]))
                                : "r"(taddr + (unsigned int)tmem_row + (unsigned int)(row_band_2 * 16 << 16) + 256));
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            float residual[16];
                            #pragma unroll
                            for (int group_2 = 0; group_2 < 4; group_2++) {
                                unsigned int v_pair_bits[2];
                                float v_pair_values[4];
                                #pragma unroll
                                for (int pair_row_2 = 0; pair_row_2 < 2; pair_row_2++) {
                                    int sr_1 = row_top + row_band_2 * 16 + pair_row_2 * 8;
                                    #pragma unroll
                                    for (int adjacent_1 = 0; adjacent_1 < 2; adjacent_1++) {
                                        long long token = token_base + (long long)(group_2 * 8) + (long long)lane_column + (long long)adjacent_1;
                                        float value = 0.0f;
                                        {
                                            residual[group_2 * 4 + pair_row_2 * 2 + adjacent_1] = value - _tmem_load_41[group_2 * 4 + pair_row_2 * 2 + adjacent_1];
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
                                :: "r"(taddr + (unsigned int)tmem_row + (unsigned int)(row_band_2 * 16 << 16) + 256 + 32), "r"(*reinterpret_cast<const uint32_t*>(&(reinterpret_cast<float*>(residual_words))[0])), "r"(*reinterpret_cast<const uint32_t*>(&(reinterpret_cast<float*>(residual_words))[1])), "r"(*reinterpret_cast<const uint32_t*>(&(reinterpret_cast<float*>(residual_words))[2])), "r"(*reinterpret_cast<const uint32_t*>(&(reinterpret_cast<float*>(residual_words))[3])), "r"(*reinterpret_cast<const uint32_t*>(&(reinterpret_cast<float*>(residual_words))[4])), "r"(*reinterpret_cast<const uint32_t*>(&(reinterpret_cast<float*>(residual_words))[5])), "r"(*reinterpret_cast<const uint32_t*>(&(reinterpret_cast<float*>(residual_words))[6])), "r"(*reinterpret_cast<const uint32_t*>(&(reinterpret_cast<float*>(residual_words))[7])), "r"(*reinterpret_cast<const uint32_t*>(&(reinterpret_cast<float*>(residual_words))[8])), "r"(*reinterpret_cast<const uint32_t*>(&(reinterpret_cast<float*>(residual_words))[9])), "r"(*reinterpret_cast<const uint32_t*>(&(reinterpret_cast<float*>(residual_words))[10])), "r"(*reinterpret_cast<const uint32_t*>(&(reinterpret_cast<float*>(residual_words))[11])), "r"(*reinterpret_cast<const uint32_t*>(&(reinterpret_cast<float*>(residual_words))[12])), "r"(*reinterpret_cast<const uint32_t*>(&(reinterpret_cast<float*>(residual_words))[13])), "r"(*reinterpret_cast<const uint32_t*>(&(reinterpret_cast<float*>(residual_words))[14])), "r"(*reinterpret_cast<const uint32_t*>(&(reinterpret_cast<float*>(residual_words))[15])));
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
                            int output_stage = 0;
                            {
                                output_stage = factor_stage;
                            }
                            {
                                {
                                    #pragma unroll 1
                                    for (int row_band_3 = 0; row_band_3 < 2; row_band_3++) {
                                        float _tmem_load_45[16];
                                        asm volatile(
                                            "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                                            " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                                            : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_45[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_45[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_45[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_45[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_45[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_45[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_45[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_45[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_45[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_45[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_45[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_45[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_45[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_45[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_45[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_45[15]))
                                            : "r"(taddr + (unsigned int)tmem_row + (unsigned int)(row_band_3 * 16 << 16) + 256 + 64));
                                        asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                        #pragma unroll
                                        for (int group_3 = 0; group_3 < 4; group_3++) {
                                            #pragma unroll
                                            for (int pair_row_3 = 0; pair_row_3 < 2; pair_row_3++) {
                                                int sr_2 = row_top + row_band_3 * 16 + pair_row_3 * 8;
                                                #pragma unroll
                                                for (int adjacent_2 = 0; adjacent_2 < 2; adjacent_2++) {
                                                    long long token_1 = token_base + (long long)(group_3 * 8) + (long long)lane_column + (long long)adjacent_2;
                                                    if (token_1 < eos) {
                                                        {
                                                            map_output_f32[(token_1 * (long long)num_heads + (long long)head_idx) * 128 + (long long)sr_2] = _tmem_load_45[group_3 * 4 + pair_row_3 * 2 + adjacent_2] * output_scale;
                                                        }
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
                            #pragma unroll 1
                            for (int row_band_4 = 0; row_band_4 < 2; row_band_4++) {
                                #pragma unroll
                                for (int half_2 = 0; half_2 < 2; half_2++) {
                                    float _tmem_load_47[32];
                                    asm volatile(
                                        "tcgen05.ld.sync.aligned.16x256b.x8.b32"
                                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_47[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_47[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_47[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_47[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_47[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_47[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_47[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_47[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_47[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_47[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_47[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_47[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_47[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_47[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_47[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_47[15])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_47[16])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_47[17])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_47[18])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_47[19])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_47[20])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_47[21])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_47[22])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_47[23])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_47[24])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_47[25])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_47[26])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_47[27])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_47[28])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_47[29])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_47[30])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_47[31]))
                                        : "r"(taddr + (unsigned int)tmem_row + (unsigned int)(row_band_4 * 16 << 16) + (unsigned int)(state_buffer * 128) + (unsigned int)(half_2 * 64)));
                                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                    #pragma unroll
                                    for (int group_4 = 0; group_4 < 8; group_4++) {
                                        int column_2 = half_2 * 64 + group_4 * 8 + lane_column;
                                        #pragma unroll
                                        for (int pair_row_4 = 0; pair_row_4 < 2; pair_row_4++) {
                                            int sr_3 = row_top + row_band_4 * 16 + pair_row_4 * 8;
                                            {
                                                {
                                                    #pragma unroll
                                                    for (int adjacent_3 = 0; adjacent_3 < 2; adjacent_3++) {
                                                        long long si_1 = state_head + (long long)sr_3 * 128 + (long long)column_2 + (long long)adjacent_3;
                                                        if (task_destination >= 0) {
                                                            mid_state_f32[(long long)task_destination * 128 * 128 + (long long)sr_3 * 128 + (long long)column_2 + (long long)adjacent_3] = _tmem_load_47[group_4 * 4 + pair_row_4 * 2 + adjacent_3];
                                                        } else if (1) {
                                                            final_state_f32[si_1] = _tmem_load_47[group_4 * 4 + pair_row_4 * 2 + adjacent_3];
                                                        } else {
                                                            final_state[si_1] = _tmem_load_47[group_4 * 4 + pair_row_4 * 2 + adjacent_3];
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                    asm volatile("barrier.sync 4, 128;" ::: "memory");
                    __threadfence();
                    asm volatile("griddepcontrol.launch_dependents;" ::: "memory");
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
