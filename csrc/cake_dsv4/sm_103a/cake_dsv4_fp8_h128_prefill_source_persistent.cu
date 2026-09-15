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
static_assert(alignof(CUtensorMap) == 128, "CUtensorMap CUDA ABI must be 128-byte aligned");
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 512
#define TMEM_TMEM_SCRATCH_OFFSET 0
#define NUM_K_PIPE_STAGES 2
#define NUM_V_PIPE_STAGES 2
#define NUM_KV_PIPE_STAGES 1
#define NUM_INDEX_PIPE_STAGES 6
#define NUM_SOURCE_WORK_PIPE_STAGES 2
#define NUM_SOURCE_THROTTLE_PIPE_STAGES 2
#define SMEM_SMEM_Q_OFF 1024
#define SMEM_SMEM_Q_STAGE_BYTES 8192
#define SMEM_SMEM_Q_STRIDE 8192
#define SMEM_SMEM_KV_OFF 33792
#define SMEM_SMEM_KV_STAGE_BYTES 16384
#define SMEM_SMEM_KV_STRIDE 16384
#define SMEM_SMEM_V_OFF 33792
#define SMEM_SMEM_V_STAGE_BYTES 16384
#define SMEM_SMEM_V_STRIDE 16384
#define SMEM_SMEM_Q_FULL_OFF 1024
#define SMEM_SMEM_Q_FULL_STAGE_BYTES 32768
#define SMEM_SMEM_Q_FULL_STRIDE 32768
#define SMEM_SMEM_K_FULL_OFF 33792
#define SMEM_SMEM_K_FULL_STAGE_BYTES 32768
#define SMEM_SMEM_K_FULL_STRIDE 32768
#define SMEM_SMEM_V_FULL_OFF 99328
#define SMEM_SMEM_V_FULL_STAGE_BYTES 32768
#define SMEM_SMEM_V_FULL_STRIDE 32768
#define SMEM_SMEM_STATS_MAX_OFF 164864
#define SMEM_SMEM_STATS_MAX_STAGE_BYTES 1024
#define SMEM_SMEM_STATS_MAX_STRIDE 1024
#define SMEM_SMEM_STATS_SUM_OFF 165888
#define SMEM_SMEM_STATS_SUM_STAGE_BYTES 512
#define SMEM_SMEM_STATS_SUM_STRIDE 512
#define SMEM_SMEM_STATS_FINAL_MAX_OFF 166400
#define SMEM_SMEM_STATS_FINAL_MAX_STAGE_BYTES 512
#define SMEM_SMEM_STATS_FINAL_MAX_STRIDE 512
#define SMEM_SMEM_SOFTMAX_WARP_PAIR_EXCHANGE_OFF 166912
#define SMEM_SMEM_SOFTMAX_WARP_PAIR_EXCHANGE_STAGE_BYTES 1024
#define SMEM_SMEM_SOFTMAX_WARP_PAIR_EXCHANGE_STRIDE 1024
#define SMEM_SMEM_CORR_WARP_PAIR_EXCHANGE_OFF 167936
#define SMEM_SMEM_CORR_WARP_PAIR_EXCHANGE_STAGE_BYTES 512
#define SMEM_SMEM_CORR_WARP_PAIR_EXCHANGE_STRIDE 512
#define SMEM_SMEM_SPARSE_INDICES_OFF 168448
#define SMEM_SMEM_SPARSE_INDICES_STAGE_BYTES 1024
#define SMEM_SMEM_SPARSE_INDICES_STRIDE 1024
#define SMEM_SMEM_P_FP8_OFF 174592
#define SMEM_SMEM_P_FP8_STAGE_BYTES 8192
#define SMEM_SMEM_P_FP8_STRIDE 8192
#define SMEM_WORK_RESPONSE_OFF 190976
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_TOTAL 191104
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


__device__ __forceinline__ void mma_ss_step_cg2(
    int a_lo, int b_lo, int taddr, uint32_t i_desc, int enable_d,
    uint32_t a_dhi, uint32_t b_dhi) {
    asm volatile(
        "{\n\t"
        ".reg .pred leader, p;\n\t"
        ".reg .b32 adhi, bdhi, m0, m1, m2, m3, m4, m5, m6, m7;\n\t"
        ".reg .b64 da, db;\n\t"
        "elect.sync _|leader, 0xFFFFFFFF;\n\t"
        "setp.ne.b32 p, %4, 0;\n\t"
        "mov.b32 m0, 0; mov.b32 m1, 0; mov.b32 m2, 0; mov.b32 m3, 0;\n\t"
        "mov.b32 m4, 0; mov.b32 m5, 0; mov.b32 m6, 0; mov.b32 m7, 0;\n\t"
        "mov.b32 adhi, %5;\n\t"
        "mov.b32 bdhi, %6;\n\t"
        "mov.b64 da, {%0, adhi};\n\t"
        "mov.b64 db, {%1, bdhi};\n\t"
        "@leader tcgen05.mma.cta_group::2.kind::f8f6f4 [%2], da, db, %3, "
        "{m0, m1, m2, m3, m4, m5, m6, m7}, p;\n\t"
        "}\n"
        :: "r"(a_lo), "r"(b_lo), "r"(taddr), "r"(i_desc), "r"(enable_d), "r"(a_dhi), "r"(b_dhi));
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


__device__ __forceinline__ void tmem_ld_x4(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x4.b32"
        " {%0, %1, %2, %3}, [%4];"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3])
        : "r"(tmem_addr));
}


__device__ __forceinline__ void tmem_st_x16(int tmem_addr, uint32_t* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x16.b32"
        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8,"
        "  %9, %10, %11, %12, %13, %14, %15, %16};"
        :: "r"(tmem_addr),
           "r"(src[0]),  "r"(src[1]),  "r"(src[2]),  "r"(src[3]),
           "r"(src[4]),  "r"(src[5]),  "r"(src[6]),  "r"(src[7]),
           "r"(src[8]),  "r"(src[9]),  "r"(src[10]), "r"(src[11]),
           "r"(src[12]), "r"(src[13]), "r"(src[14]), "r"(src[15]));
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


__device__ __forceinline__ void tmem_st_x4_f32(int tmem_addr, const float* src) {
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x4.b32"
        " [%0], {%1, %2, %3, %4};"
        :: "r"(tmem_addr),
           "f"(src[0]), "f"(src[1]), "f"(src[2]), "f"(src[3]));
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



__device__ __forceinline__ void softmax_block_sum(const float* sv, float2* acc) {
    const float2* sv2 = reinterpret_cast<const float2*>(sv);
    #pragma unroll
    for (int j = 0; j < 16; j++) {
        asm("add.f32x2 %0, %1, %2;"
            : "+l"(reinterpret_cast<uint64_t&>(*acc))
            : "l"(reinterpret_cast<uint64_t&>(*acc)),
              "l"(reinterpret_cast<const uint64_t&>(sv2[j])));
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

__device__ __forceinline__ void mul_f32x2_inplace(float2* a, float2 b) {
    asm("mul.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void add_f32x2_inplace(float2* a, float2 b) {
    asm("add.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ void sub_f32x2_inplace(float2* a, float2 b) {
    asm("sub.rn.ftz.f32x2 %0, %0, %1;"
        : "+l"(*(unsigned long long*)a) : "l"(*(unsigned long long*)&b));
}

__device__ __forceinline__ float2 add_f32x2(float2 a, float2 b) {
    float2 r;
    asm("add.rn.ftz.f32x2 %0, %1, %2;"
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


__device__ __forceinline__ void tma_4d_gmem2smem_cta2(
    int dst, const void *tmap_ptr, int x, int y, int z, int w, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
        " [%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(dst), "l"(tmap_ptr), "r"(x), "r"(y), "r"(z), "r"(w),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_gather4_gmem2smem(
    int dst, const void *tmap_ptr,
    int col_idx, int row0, int row1, int row2, int row3,
    int mbar_addr) {
    // Canonical .shared::cta form for non-multicast gather4, matching
    // trtllm-gen / cuda_ptx and the PTX ISA qualifier order
    // (dim.dst.src.load_mode.completion_mechanism). Per the PTX grammar,
    // .shared::cluster is reserved for the multicast variant (ctaMask).
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
        ".mbarrier::complete_tx::bytes"
        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
        :: "r"(dst), "l"(tmap_ptr), "r"(col_idx),
           "r"(row0), "r"(row1), "r"(row2), "r"(row3),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_gather4_gmem2smem_cta2(
    int dst, const void *tmap_ptr,
    int col_idx, int row0, int row1, int row2, int row3,
    int mbar_addr) {
    // Canonical .shared::cta form; see tma_gather4_gmem2smem above.
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.tile::gather4"
        ".mbarrier::complete_tx::bytes.cta_group::2"
        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
        :: "r"(dst), "l"(tmap_ptr), "r"(col_idx),
           "r"(row0), "r"(row1), "r"(row2), "r"(row3),
           "r"(mbar_addr) : "memory");
}


__device__ __forceinline__ void tma_gather4_gmem2smem_mc(
    int dst, const void *tmap_ptr,
    int col_idx, int row0, int row1, int row2, int row3,
    int mbar_addr, unsigned short cta_mask) {
    // Multicast variant: the PTX grammar ties the .shared::cluster
    // destination to .multicast::cluster + ctaMask (cf. cuda_ptx /
    // SM100_TMA_LOAD_MULTICAST_2D_GATHER4 in CUTLASS).
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
        ".mbarrier::complete_tx::bytes.multicast::cluster"
        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
        :: "r"(dst), "l"(tmap_ptr), "r"(col_idx),
           "r"(row0), "r"(row1), "r"(row2), "r"(row3),
           "r"(mbar_addr), "h"(cta_mask) : "memory");
}


__device__ __forceinline__ void tma_gather4_gmem2smem_mc_cta2(
    int dst, const void *tmap_ptr,
    int col_idx, int row0, int row1, int row2, int row3,
    int mbar_addr, unsigned short cta_mask) {
    // Multicast + cta_group::2 variant; see tma_gather4_gmem2smem_mc.
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4"
        ".mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
        " [%0], [%1, {%2, %3, %4, %5, %6}], [%7], %8;"
        :: "r"(dst), "l"(tmap_ptr), "r"(col_idx),
           "r"(row0), "r"(row1), "r"(row2), "r"(row3),
           "r"(mbar_addr), "h"(cta_mask) : "memory");
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

extern "C" {

__global__ __launch_bounds__(512, 1) __cluster_dims__(2,1,1) void
kernel_cake_dsv4_fp8_h128_prefill_source_persistent(const __grid_constant__ CUtensorMap tmap_q, const __grid_constant__ CUtensorMap tmap_swa_kv, const __grid_constant__ CUtensorMap tmap_compressed_kv, __nv_bfloat16* __restrict__ O, float* __restrict__ partial_lse, int* __restrict__ sparse_indices, int* __restrict__ sparse_topk_lens, int* __restrict__ seq_lens, int* __restrict__ cum_seq_lens_q, float* __restrict__ sinks, float* __restrict__ bmm1_scale, float* __restrict__ bmm2_scale, int num_heads, int num_query_tokens, int sparse_topk, int has_sinks, int total_work_items, int max_q_len, int batch_size)
{
    // PTX global compiler scheduling controls
    asm volatile(".pragma \"global knob ForceLateCommoning=1\";\n" : : : "memory");
    asm volatile(".pragma \"global knob HoistLate=3\";\n" : : : "memory");
    asm volatile(".pragma \"global knob MbarrierInitRegMapping=1\";\n" : : : "memory");

    const int tid = threadIdx.x;
    const uint32_t warp = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    uint32_t lane;
    asm("mov.u32 %0, %%laneid;" : "=r"(lane));

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define q_full_addr (mbar_base + 0)
    #define q_empty_addr (mbar_base + 8)
    #define k_full_addr (mbar_base + 16)
    #define k_empty_addr (mbar_base + 32)
    #define v_full_addr (mbar_base + 48)
    #define v_empty_addr (mbar_base + 64)
    #define index_full_addr (mbar_base + 80)
    #define index_empty_addr (mbar_base + 128)
    #define s_full_addr (mbar_base + 176)
    #define p_full_addr (mbar_base + 192)
    #define split_pv_p_empty_addr (mbar_base + 208)
    #define s_empty_addr (mbar_base + 224)
    #define stats_addr (mbar_base + 240)
    #define sum_ready_addr (mbar_base + 256)
    #define o_empty_addr (mbar_base + 264)
    #define o_full_addr (mbar_base + 272)
    #define s_seeded_addr (mbar_base + 280)
    #define q_pair_ready_addr (mbar_base + 288)
    #define kv_pair_ready_addr (mbar_base + 296)
    #define pv_pair_ready_addr (mbar_base + 304)
    #define tmem_dealloc_addr (mbar_base + 320)
    #define tmem_dealloc_peer_addr (mbar_base + 328)
    #define source_work_full_addr (mbar_base + 336)
    #define source_work_empty_addr (mbar_base + 352)
    #define source_throttle_full_addr (mbar_base + 368)
    #define source_throttle_empty_addr (mbar_base + 384)

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
    uint8_t* smem_kv = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int smem_kv_addr = smem + 33792;
    uint8_t* smem_v = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int smem_v_addr = smem + 33792;
    uint8_t* smem_q_full = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_q_full_addr = smem + 1024;
    uint8_t* smem_k_full = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int smem_k_full_addr = smem + 33792;
    uint8_t* smem_v_full = reinterpret_cast<uint8_t*>(smem_raw + 99328);
    const int smem_v_full_addr = smem + 99328;
    float* smem_stats_max = reinterpret_cast<float*>(smem_raw + 164864);
    const int smem_stats_max_addr = smem + 164864;
    float* smem_stats_sum = reinterpret_cast<float*>(smem_raw + 165888);
    const int smem_stats_sum_addr = smem + 165888;
    float* smem_stats_final_max = reinterpret_cast<float*>(smem_raw + 166400);
    const int smem_stats_final_max_addr = smem + 166400;
    float* smem_softmax_warp_pair_exchange = reinterpret_cast<float*>(smem_raw + 166912);
    const int smem_softmax_warp_pair_exchange_addr = smem + 166912;
    float* smem_corr_warp_pair_exchange = reinterpret_cast<float*>(smem_raw + 167936);
    const int smem_corr_warp_pair_exchange_addr = smem + 167936;
    int* smem_sparse_indices = reinterpret_cast<int*>(smem_raw + 168448);
    const int smem_sparse_indices_addr = smem + 168448;
    uint8_t* smem_p_fp8 = reinterpret_cast<uint8_t*>(smem_raw + 174592);
    const int smem_p_fp8_addr = smem + 174592;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 190976);
    const int work_response_addr = smem + 190976;
    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // Mbarrier init (26 pipeline groups, 0 ordered-sequence groups, 50 barriers)
    // Mbarriers at smem_raw[0..400)

    if (warp == 12) {
        uint32_t leader = elect_sync();
        if (leader) {
            // q_full: 1 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            // q_empty: 1 barriers, init_count=1
            mbarrier_init(smem + 8, 1);
            // --- pipeline 'k_pipe' ---
            // k_full: 2 barriers, init_count=2
            mbarrier_init(smem + 16, 2);
            mbarrier_init(smem + 24, 2);
            // k_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            // --- pipeline 'v_pipe' ---
            // v_full: 2 barriers, init_count=1
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            // v_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }
    if (warp == 9) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'index_pipe' ---
            // index_full: 6 barriers, init_count=32
            mbarrier_init(smem + 80, 32);
            mbarrier_init(smem + 88, 32);
            mbarrier_init(smem + 96, 32);
            mbarrier_init(smem + 104, 32);
            mbarrier_init(smem + 112, 32);
            mbarrier_init(smem + 120, 32);
            // index_empty: 6 barriers, init_count=128
            mbarrier_init(smem + 128, 128);
            mbarrier_init(smem + 136, 128);
            mbarrier_init(smem + 144, 128);
            mbarrier_init(smem + 152, 128);
            mbarrier_init(smem + 160, 128);
            mbarrier_init(smem + 168, 128);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }
    if (warp == 1) {
        uint32_t leader = elect_sync();
        if (leader) {
            // s_full: 2 barriers, init_count=1
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            // p_full: 2 barriers, init_count=256
            mbarrier_init(smem + 192, 256);
            mbarrier_init(smem + 200, 256);
            // split_pv_p_empty: 2 barriers, init_count=1
            mbarrier_init(smem + 208, 1);
            mbarrier_init(smem + 216, 1);
            // s_empty: 2 barriers, init_count=256
            mbarrier_init(smem + 224, 256);
            mbarrier_init(smem + 232, 256);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }
    if (warp == 5) {
        uint32_t leader = elect_sync();
        if (leader) {
            // stats: 2 barriers, init_count=128
            mbarrier_init(smem + 240, 128);
            mbarrier_init(smem + 248, 128);
            // sum_ready: 1 barriers, init_count=128
            mbarrier_init(smem + 256, 128);
            // tmem_dealloc_peer: 1 barriers, init_count=32
            mbarrier_init(smem + 328, 32);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }
    if (warp == 4) {
        uint32_t leader = elect_sync();
        if (leader) {
            // o_empty: 1 barriers, init_count=256
            mbarrier_init(smem + 264, 256);
            // o_full: 1 barriers, init_count=1
            mbarrier_init(smem + 272, 1);
            // pv_pair_ready: stages (1,), init_count=64
            mbarrier_init(smem + 312, 64);
            // tmem_dealloc: 1 barriers, init_count=448
            mbarrier_init(smem + 320, 448);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }
    if (warp == 2) {
        uint32_t leader = elect_sync();
        if (leader) {
            // s_seeded: 1 barriers, init_count=256
            mbarrier_init(smem + 280, 256);
            // q_pair_ready: 1 barriers, init_count=64
            mbarrier_init(smem + 288, 64);
            // --- pipeline 'kv_pipe' ---
            // kv_pair_ready: 1 barriers, init_count=64
            mbarrier_init(smem + 296, 64);
            // pv_pair_ready: stages (0,), init_count=64
            mbarrier_init(smem + 304, 64);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }
    if (warp == 10) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'source_work_pipe' ---
            // source_work_full: 2 barriers, init_count=1
            mbarrier_init(smem + 336, 1);
            mbarrier_init(smem + 344, 1);
            // source_work_empty: 2 barriers, init_count=960
            mbarrier_init(smem + 352, 960);
            mbarrier_init(smem + 360, 960);
            // --- pipeline 'source_throttle_pipe' ---
            // source_throttle_full: 2 barriers, init_count=128
            mbarrier_init(smem + 368, 128);
            mbarrier_init(smem + 376, 128);
            // source_throttle_empty: 2 barriers, init_count=32
            mbarrier_init(smem + 384, 32);
            mbarrier_init(smem + 392, 32);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncthreads();

    // TMEM alloc (512 columns, 512 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 400);
    if (warp == 0) {
        int _tmem_hold = smem + 400;
        asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;");
    }

    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_scratch = taddr;

    // ---- Ordered hardware-WG register redistribution ----
    // Inc phase consumes the registers released above.
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 136;");
    }

    // ---- Role: index_warp ----
    if (warp == 9) {
        { // index_warp_main
            const int index_dummy = 0;
            unsigned int _phase_index_empty = 1;
            unsigned int _phase_source_work_full = 0;
            unsigned int _phase_q_full_0 = 0;
            {
                int all_num_kv_tiles = (sparse_topk + 128 - 1) / 128;
                unsigned int index_prod_stage = 0;
                unsigned int source_work_stage = 0;
                unsigned int source_work_x = blockIdx.x;
                unsigned int source_work_z = blockIdx.z;
                #pragma unroll 1
                for (unsigned int static_work = 0; static_work < 514; static_work++) {
                    int split_idx = 0;
                    int query_idx = source_work_x >> 1;
                    int source_work_valid = source_work_z * (unsigned int)max_q_len + (source_work_x >> 1) < (unsigned int)num_query_tokens;
                    int mapped_query_idx = source_work_z * (unsigned int)max_q_len + (source_work_x >> 1);
                    if (source_work_valid != 0) {
                        int active_topk = sparse_topk_lens[mapped_query_idx];
                        all_num_kv_tiles = (active_topk + 128 - 1) / 128;
                        int tiles_per_split = all_num_kv_tiles + 1 - 1;
                        int first_tile = split_idx * tiles_per_split;
                        int sparse_base = mapped_query_idx * sparse_topk;
                        int num_index_passes = (tiles_per_split + 1) / 2;
                        #pragma unroll 1
                        for (int index_pass = 0; index_pass < num_index_passes; index_pass++) {
                            mbarrier_wait(index_empty_addr + (index_prod_stage) * 8, _phase_index_empty);
                            int index_stage_base = smem_sparse_indices_addr + index_prod_stage * 1024;
                            int stage_row_base = index_pass * 256;
                            #pragma unroll
                            for (int index_half = 0; index_half < 2; index_half++) {
                                int index_offset = (unsigned int)(index_half * 128) + lane * 4;
                                int global_index_offset = first_tile * 128 + stage_row_base + index_offset;
                                asm volatile("cp.async.cg.shared::cta.global.L2::128B [%0], [%1], 16;"
                                    :: "r"(index_stage_base + index_offset * 4), "l"(sparse_indices + (sparse_base + global_index_offset)));
                            }
                            asm volatile(
                                "{\n\t"
                                "cp.async.mbarrier.arrive.shared::cta.b64 [%0];\n\t"
                                "}"
                                :: "r"(index_full_addr + (index_prod_stage) * 8) : "memory");
                            mbarrier_arrive(index_full_addr + (index_prod_stage) * 8);
                            index_prod_stage += 1;
                            if (index_prod_stage == 6) { index_prod_stage = 0; _phase_index_empty ^= 1; }
                        }
                    }
                    {
                        mbarrier_wait(source_work_full_addr + (source_work_stage) * 8, _phase_source_work_full);
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
                            : "r"(work_response_addr + source_work_stage * 16 + 0 * 16)
                            : "memory");
                        uint32_t _clc_ctaid_0 = 0;
                        asm volatile(
                            "{\n\t"
                            ".reg .b128 clc_r;\n\t"
                            "ld.shared.b128 clc_r, [%1];\n\t"
                            "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                            "}\n"
                            : "=r"(_clc_ctaid_0)
                            : "r"(work_response_addr + source_work_stage * 16 + 0 * 16)
                            : "memory");
                        uint32_t _clc_ctaid_1 = 0;
                        asm volatile(
                            "{\n\t"
                            ".reg .b128 clc_r;\n\t"
                            "ld.shared.b128 clc_r, [%1];\n\t"
                            "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                            "}\n"
                            : "=r"(_clc_ctaid_1)
                            : "r"(work_response_addr + source_work_stage * 16 + 0 * 16)
                            : "memory");
                        uint32_t _clc_ctaid_2 = 0;
                        asm volatile(
                            "{\n\t"
                            ".reg .b128 clc_r;\n\t"
                            "ld.shared.b128 clc_r, [%1];\n\t"
                            "clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                            "}\n"
                            : "=r"(_clc_ctaid_2)
                            : "r"(work_response_addr + source_work_stage * 16 + 0 * 16)
                            : "memory");
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                            "}"
                            :: "r"(source_work_empty_addr + source_work_stage * 8), "r"(0) : "memory");
                        source_work_stage += 1;
                        if (source_work_stage == 2) { source_work_stage = 0; _phase_source_work_full ^= 1; }
                        if (_clc_valid_0 == 0) {
                            break;
                        }
                        source_work_x = _clc_ctaid_0 + (unsigned int)cta_rank;
                        source_work_z = _clc_ctaid_2;
                    }
                }
            }
        }
    }
    // ---- Role: load_warp ----
    if (warp >= 12 && warp <= 15) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 72;");
        { // load_warp_main
            const int wg2_dummy = 0;
            const int load_warp_rank = warp - (unsigned int)(((1) ? 12 : 9));
            int all_num_kv_tiles_1 = (sparse_topk + 128 - 1) / 128;
            unsigned int load_k_stage = 0;
            unsigned int load_v_stage = 0;
            int load_v_tile_cursor = 0;
            unsigned int load_kv_stage = 0;
            unsigned int load_k_index_stage = 0;
            unsigned int load_v_index_stage = 0;
            int k_cta_offset = bid % 2 * 64;
            unsigned int source_work_stage_1 = 0;
            unsigned int source_throttle_stage = 0;
            unsigned int source_work_x_1 = blockIdx.x;
            unsigned int source_work_z_1 = blockIdx.z;
            unsigned int _phase_source_throttle_empty = 1;
            unsigned int _phase_q_empty_0 = 1;
            unsigned int _phase_index_full = 0;
            unsigned int _phase_k_empty = 1;
            unsigned int _phase_index_full_1 = 0;
            unsigned int _phase_source_work_full_1 = 0;
            #pragma unroll 1
            for (unsigned int work_idx = 0; work_idx < 514; work_idx++) {
                int split_idx_1 = 0;
                int query_idx_1 = source_work_x_1 >> 1 >> 1;
                int v_chunk = source_work_x_1 >> 1 & 1;
                {
                    split_idx_1 = 0;
                    query_idx_1 = source_work_x_1 >> 1;
                    v_chunk = bid % 2;
                }
                {
                    if (bid % 2 == 0) {
                        mbarrier_wait(source_throttle_empty_addr + (source_throttle_stage) * 8, _phase_source_throttle_empty);
                        mbarrier_arrive(source_throttle_full_addr + (source_throttle_stage) * 8);
                        source_throttle_stage += 1;
                        if (source_throttle_stage == 2) { source_throttle_stage = 0; _phase_source_throttle_empty ^= 1; }
                    }
                }
                int source_work_valid_1 = source_work_z_1 * (unsigned int)max_q_len + (source_work_x_1 >> 1) < (unsigned int)num_query_tokens;
                int mapped_query_idx_1 = source_work_z_1 * (unsigned int)max_q_len + (source_work_x_1 >> 1);
                if (source_work_valid_1 != 0) {
                    {
                        int active_topk_1 = sparse_topk_lens[mapped_query_idx_1];
                        all_num_kv_tiles_1 = (active_topk_1 + 128 - 1) / 128;
                    }
                    int tiles_per_split_1 = all_num_kv_tiles_1 + 1 - 1;
                    int first_tile_1 = split_idx_1 * tiles_per_split_1;
                    int num_kv_tiles = tiles_per_split_1;
                    int sparse_extent = num_kv_tiles * 128;
                    if (load_warp_rank < 2) {
                        mbarrier_wait(q_empty_addr, _phase_q_empty_0);
                        _phase_q_empty_0 ^= 1;
                        if (load_warp_rank == 0) {
                            if (elect_sync()) {
                                asm volatile(
                                    "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                    :: "r"((q_full_addr) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                                #pragma unroll
                                for (int q_stage = 0; q_stage < 4; q_stage++) {
                                    asm volatile(
                                        "cp.async.bulk.tensor.4d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster.cta_group::2"
                                        " [%0], [%1, {%2, %3, %4, %5}], [%6], %7;"
                                        :: "r"(smem_q_full_addr + (unsigned int)(q_stage * 8192)), "l"((&tmap_q)), "r"(0), "r"(bid % 2 * 64), "r"(q_stage), "r"(mapped_query_idx_1),
                                           "r"(((q_full_addr) & 0xFEFFFFFF)), "h"((uint16_t)(1 << bid % 2)) : "memory");
                                }
                            }
                        }
                        #pragma unroll 1
                        for (int tile = 0; tile < num_kv_tiles; tile++) {
                            if ((tile & 1) == 0) {
                                mbarrier_wait(index_full_addr + (load_k_index_stage) * 8, _phase_index_full);
                                asm volatile("tcgen05.fence::after_thread_sync;");
                            }
                            int group = lane & 15;
                            int feature_chunk = (unsigned int)(load_warp_rank * 2) + (lane >> 4);
                            int group_offset = (tile & 1) * 128 + k_cta_offset + group * 4;
                            int raw_rows[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&raw_rows[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows[(0) + 3]))
                                : "r"(smem_sparse_indices_addr + load_k_index_stage * 1024 + (unsigned int)(group_offset * 4)));
                            mbarrier_wait(k_empty_addr + (load_k_stage) * 8, _phase_k_empty);
                            if (load_warp_rank == 0) {
                                if (elect_sync()) {
                                    asm volatile(
                                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                        :: "r"((k_full_addr + (load_k_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(32768)) : "memory");
                                }
                            }
                            int k_dst = smem_k_full_addr + load_k_stage * 32768 + (unsigned int)(feature_chunk * 8192);
                            if (first_tile_1 + tile == 0) {
                                tma_gather4_gmem2smem_mc_cta2(k_dst + group * 512, (&tmap_swa_kv), feature_chunk * 128, ((raw_rows[0] >= 0) ? raw_rows[0] : 0), ((raw_rows[1] >= 0) ? raw_rows[1] : 0), ((raw_rows[2] >= 0) ? raw_rows[2] : 0), ((raw_rows[3] >= 0) ? raw_rows[3] : 0), ((k_full_addr + (load_k_stage) * 8) & 0xFEFFFFFF), 1 << bid % 2);
                            } else {
                                tma_gather4_gmem2smem_mc_cta2(k_dst + group * 512, (&tmap_compressed_kv), feature_chunk * 128, ((raw_rows[0] >= 0) ? raw_rows[0] : 0), ((raw_rows[1] >= 0) ? raw_rows[1] : 0), ((raw_rows[2] >= 0) ? raw_rows[2] : 0), ((raw_rows[3] >= 0) ? raw_rows[3] : 0), ((k_full_addr + (load_k_stage) * 8) & 0xFEFFFFFF), 1 << bid % 2);
                            }
                            load_k_stage += 1;
                            if (load_k_stage == 2) { load_k_stage = 0; _phase_k_empty ^= 1; }
                            if ((tile & 1) != 0 || tile + 1 == num_kv_tiles) {
                                asm volatile("tcgen05.fence::before_thread_sync;");
                                mbarrier_arrive(index_empty_addr + (load_k_index_stage) * 8);
                                load_k_index_stage += 1;
                                if (load_k_index_stage == 6) { load_k_index_stage = 0; _phase_index_full ^= 1; }
                            }
                        }
                    } else {
                        int v_warp_rank = load_warp_rank - 2;
                        #pragma unroll 1
                        for (int tile_1 = 0; tile_1 < num_kv_tiles; tile_1++) {
                            if ((tile_1 & 1) == 0) {
                                mbarrier_wait(index_full_addr + (load_v_index_stage) * 8, _phase_index_full_1);
                                asm volatile("tcgen05.fence::after_thread_sync;");
                            }
                            int v_dst = smem_v_full_addr + load_v_stage * 32768 + (unsigned int)(v_warp_rank * 16384);
                            int v_col = bid % 2 * 256 + v_warp_rank * 128;
                            int group_1 = lane;
                            int group_offset_1 = (tile_1 & 1) * 128 + group_1 * 4;
                            int raw_rows_1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&raw_rows_1[(0) + 3]))
                                : "r"(smem_sparse_indices_addr + load_v_index_stage * 1024 + (unsigned int)(group_offset_1 * 4)));
                            int v_empty_phase = load_v_tile_cursor >> 1 & 1 ^ 1;
                            uint32_t _mbar_token_0 = mbarrier_try_wait(v_empty_addr + (load_v_stage) * 8, v_empty_phase);
                            mbarrier_wait_token(v_empty_addr + (load_v_stage) * 8, v_empty_phase, _mbar_token_0);
                            if (v_warp_rank == 0 && bid % 2 == 0) {
                                if (elect_sync()) {
                                    asm volatile(
                                        "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                                        :: "r"((v_full_addr + (load_v_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(65536)) : "memory");
                                }
                            }
                            if (first_tile_1 + tile_1 == 0) {
                                tma_gather4_gmem2smem_mc_cta2(v_dst + group_1 * 512, (&tmap_swa_kv), v_col, ((raw_rows_1[0] >= 0) ? raw_rows_1[0] : 0), ((raw_rows_1[1] >= 0) ? raw_rows_1[1] : 0), ((raw_rows_1[2] >= 0) ? raw_rows_1[2] : 0), ((raw_rows_1[3] >= 0) ? raw_rows_1[3] : 0), ((v_full_addr + (load_v_stage) * 8) & 0xFEFFFFFF), 1 << bid % 2);
                            } else {
                                tma_gather4_gmem2smem_mc_cta2(v_dst + group_1 * 512, (&tmap_compressed_kv), v_col, ((raw_rows_1[0] >= 0) ? raw_rows_1[0] : 0), ((raw_rows_1[1] >= 0) ? raw_rows_1[1] : 0), ((raw_rows_1[2] >= 0) ? raw_rows_1[2] : 0), ((raw_rows_1[3] >= 0) ? raw_rows_1[3] : 0), ((v_full_addr + (load_v_stage) * 8) & 0xFEFFFFFF), 1 << bid % 2);
                            }
                            load_v_stage += 1;
                            if (load_v_stage == 2) { load_v_stage = 0; }
                            load_v_tile_cursor = load_v_tile_cursor + 1;
                            if ((tile_1 & 1) != 0 || tile_1 + 1 == num_kv_tiles) {
                                asm volatile("tcgen05.fence::before_thread_sync;");
                                mbarrier_arrive(index_empty_addr + (load_v_index_stage) * 8);
                                load_v_index_stage += 1;
                                if (load_v_index_stage == 6) { load_v_index_stage = 0; _phase_index_full ^= 1; _phase_index_full_1 ^= 1; }
                            }
                        }
                    }
                }
                {
                    mbarrier_wait(source_work_full_addr + (source_work_stage_1) * 8, _phase_source_work_full_1);
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
                        : "r"(work_response_addr + source_work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_3 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_3)
                        : "r"(work_response_addr + source_work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_4 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_4)
                        : "r"(work_response_addr + source_work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_5 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_5)
                        : "r"(work_response_addr + source_work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(source_work_empty_addr + source_work_stage_1 * 8), "r"(0) : "memory");
                    source_work_stage_1 += 1;
                    if (source_work_stage_1 == 2) { source_work_stage_1 = 0; _phase_source_work_full_1 ^= 1; }
                    if (_clc_valid_1 == 0) {
                        break;
                    }
                    source_work_x_1 = _clc_ctaid_3 + (unsigned int)cta_rank;
                    source_work_z_1 = _clc_ctaid_5;
                }
            }
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: softmax_wg ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 152;");
        { // softmax_wg_main
            float softmax_scale_log2 = bmm1_scale[0] * 1.4426950408889634f;
            const int wg_dummy_inc = 0;
            int all_num_kv_tiles_2 = (sparse_topk + 128 - 1) / 128;
            const int tmem_row_base = ((1) ? warp % 2 * 32 : warp % 4 * 32);
            const int tmem_score_row_base = ((1) ? (int)(warp % 4 * 32) : tmem_row_base);
            const int n_half = ((1) ? (int)(warp % 4 / 2) : 0);
            const int my_row = (unsigned int)tmem_row_base + lane;
            const int stats_row = n_half * 64 + my_row;
            int softmax_tile_cursor = 0;
            unsigned int source_work_stage_2 = 0;
            unsigned int source_work_x_2 = blockIdx.x;
            unsigned int source_work_z_2 = blockIdx.z;
            unsigned int _phase_q_full_0_1 = 0;
            unsigned int _phase_source_work_full_2 = 0;
            #pragma unroll 1
            for (unsigned int work_idx_1 = 0; work_idx_1 < 514; work_idx_1++) {
                int split_idx_2 = 0;
                int query_idx_2 = source_work_x_2 >> 1 >> 1;
                {
                    split_idx_2 = 0;
                    query_idx_2 = source_work_x_2 >> 1;
                }
                int source_work_valid_2 = source_work_z_2 * (unsigned int)max_q_len + (source_work_x_2 >> 1) < (unsigned int)num_query_tokens;
                int mapped_query_idx_2 = source_work_z_2 * (unsigned int)max_q_len + (source_work_x_2 >> 1);
                if (source_work_valid_2 != 0) {
                    int active_topk_2 = sparse_topk_lens[mapped_query_idx_2];
                    {
                        all_num_kv_tiles_2 = (active_topk_2 + 128 - 1) / 128;
                    }
                    int tiles_per_split_2 = all_num_kv_tiles_2 + 1 - 1;
                    int first_tile_2 = split_idx_2 * tiles_per_split_2;
                    int num_kv_tiles_1 = tiles_per_split_2;
                    float row_max_val = -CAKE_INF;
                    float row_sum_val = 0.0f;
                    int sink_head = ((1) ? bid % 2 * 64 + my_row : my_row);
                    if (has_sinks != 0 && sink_head < num_heads && split_idx_2 == 0) {
                        row_max_val = sinks[sink_head] * 1.4426950408889634f / softmax_scale_log2;
                        row_sum_val = 1.0f;
                    }
                    #pragma unroll 1
                    for (int tile_2 = 0; tile_2 < num_kv_tiles_1; tile_2++) {
                        int pipeline_tile = softmax_tile_cursor + tile_2;
                        int phase = pipeline_tile & 1;
                        int s_wait_phase = pipeline_tile >> 1 & 1;
                        int s_off = ((phase != 0) ? 128 : 0);
                        int s_base = taddr + (unsigned int)s_off + (unsigned int)(tmem_score_row_base << 16);
                        float new_max = row_max_val;
                        int valid_sparse_cols = ((active_topk_2 < sparse_topk) ? active_topk_2 : sparse_topk);
                        valid_sparse_cols = valid_sparse_cols - (first_tile_2 + tile_2) * 128 - n_half * 64;
                        if (valid_sparse_cols < 0) {
                            valid_sparse_cols = 0;
                        }
                        if (valid_sparse_cols > 64) {
                            valid_sparse_cols = 64;
                        }
                        {
                            uint32_t _mbar_token_5 = mbarrier_try_wait(s_full_addr + (phase) * 8, s_wait_phase);
                            mbarrier_wait_token(s_full_addr + (phase) * 8, s_wait_phase, _mbar_token_5);
                            asm volatile("tcgen05.fence::after_thread_sync;");
                        }
                        float _tmem_load_0[4];
                        tmem_ld_x4(&_tmem_load_0[0], s_base);
                        float score_tile_max = -CAKE_INF;
                        float score_tile_max_hi = -CAKE_INF;
                        float sv_split[64];
                        {
                            if (valid_sparse_cols == 64) {
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                                #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                                #endif
                                asm volatile(
                                    "tcgen05.ld.red.sync.aligned.32x32b.x32.max.f32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];"
                                    : "=f"(sv_split[0]), "=f"(sv_split[1]), "=f"(sv_split[2]), "=f"(sv_split[3]), "=f"(sv_split[4]), "=f"(sv_split[5]), "=f"(sv_split[6]), "=f"(sv_split[7]), "=f"(sv_split[8]), "=f"(sv_split[9]), "=f"(sv_split[10]), "=f"(sv_split[11]), "=f"(sv_split[12]), "=f"(sv_split[13]), "=f"(sv_split[14]), "=f"(sv_split[15]), "=f"(sv_split[16]), "=f"(sv_split[17]), "=f"(sv_split[18]), "=f"(sv_split[19]), "=f"(sv_split[20]), "=f"(sv_split[21]), "=f"(sv_split[22]), "=f"(sv_split[23]), "=f"(sv_split[24]), "=f"(sv_split[25]), "=f"(sv_split[26]), "=f"(sv_split[27]), "=f"(sv_split[28]), "=f"(sv_split[29]), "=f"(sv_split[30]), "=f"(sv_split[31]), "=f"(score_tile_max)
                                    : "r"(s_base));
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1000)
                                #error "TmemLoadRed requires tcgen05.ld.red support (sm_103/sm_101-sm_110 family), not sm_100"
                                #endif
                                asm volatile(
                                    "tcgen05.ld.red.sync.aligned.32x32b.x32.max.f32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, %32, [%33];"
                                    : "=f"(sv_split[32]), "=f"(sv_split[33]), "=f"(sv_split[34]), "=f"(sv_split[35]), "=f"(sv_split[36]), "=f"(sv_split[37]), "=f"(sv_split[38]), "=f"(sv_split[39]), "=f"(sv_split[40]), "=f"(sv_split[41]), "=f"(sv_split[42]), "=f"(sv_split[43]), "=f"(sv_split[44]), "=f"(sv_split[45]), "=f"(sv_split[46]), "=f"(sv_split[47]), "=f"(sv_split[48]), "=f"(sv_split[49]), "=f"(sv_split[50]), "=f"(sv_split[51]), "=f"(sv_split[52]), "=f"(sv_split[53]), "=f"(sv_split[54]), "=f"(sv_split[55]), "=f"(sv_split[56]), "=f"(sv_split[57]), "=f"(sv_split[58]), "=f"(sv_split[59]), "=f"(sv_split[60]), "=f"(sv_split[61]), "=f"(sv_split[62]), "=f"(sv_split[63]), "=f"(score_tile_max_hi)
                                    : "r"(s_base + 32));
                                asm volatile("tcgen05.wait::ld.sync.aligned;");
                                float _max_0 = max_noftz(score_tile_max, score_tile_max_hi);
                                score_tile_max = _max_0;
                            } else if (valid_sparse_cols != 0) {
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                    : "=f"(sv_split[0]), "=f"(sv_split[1]), "=f"(sv_split[2]), "=f"(sv_split[3]), "=f"(sv_split[4]), "=f"(sv_split[5]), "=f"(sv_split[6]), "=f"(sv_split[7]), "=f"(sv_split[8]), "=f"(sv_split[9]), "=f"(sv_split[10]), "=f"(sv_split[11]), "=f"(sv_split[12]), "=f"(sv_split[13]), "=f"(sv_split[14]), "=f"(sv_split[15]), "=f"(sv_split[16]), "=f"(sv_split[17]), "=f"(sv_split[18]), "=f"(sv_split[19]), "=f"(sv_split[20]), "=f"(sv_split[21]), "=f"(sv_split[22]), "=f"(sv_split[23]), "=f"(sv_split[24]), "=f"(sv_split[25]), "=f"(sv_split[26]), "=f"(sv_split[27]), "=f"(sv_split[28]), "=f"(sv_split[29]), "=f"(sv_split[30]), "=f"(sv_split[31])
                                    : "r"(s_base));
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                    : "=f"(sv_split[32]), "=f"(sv_split[33]), "=f"(sv_split[34]), "=f"(sv_split[35]), "=f"(sv_split[36]), "=f"(sv_split[37]), "=f"(sv_split[38]), "=f"(sv_split[39]), "=f"(sv_split[40]), "=f"(sv_split[41]), "=f"(sv_split[42]), "=f"(sv_split[43]), "=f"(sv_split[44]), "=f"(sv_split[45]), "=f"(sv_split[46]), "=f"(sv_split[47]), "=f"(sv_split[48]), "=f"(sv_split[49]), "=f"(sv_split[50]), "=f"(sv_split[51]), "=f"(sv_split[52]), "=f"(sv_split[53]), "=f"(sv_split[54]), "=f"(sv_split[55]), "=f"(sv_split[56]), "=f"(sv_split[57]), "=f"(sv_split[58]), "=f"(sv_split[59]), "=f"(sv_split[60]), "=f"(sv_split[61]), "=f"(sv_split[62]), "=f"(sv_split[63])
                                    : "r"(s_base + 32));
                                uint32_t _slice_lo_mask_0;
                                {
                                    int _lim_0 = valid_sparse_cols;
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
                                if (!(_slice_lo_mask_0 & (1u << 0))) sv_split[0] = -CAKE_INF;
                                if (!(_slice_lo_mask_0 & (1u << 1))) sv_split[1] = -CAKE_INF;
                                if (!(_slice_lo_mask_0 & (1u << 2))) sv_split[2] = -CAKE_INF;
                                if (!(_slice_lo_mask_0 & (1u << 3))) sv_split[3] = -CAKE_INF;
                                if (!(_slice_lo_mask_0 & (1u << 4))) sv_split[4] = -CAKE_INF;
                                if (!(_slice_lo_mask_0 & (1u << 5))) sv_split[5] = -CAKE_INF;
                                if (!(_slice_lo_mask_0 & (1u << 6))) sv_split[6] = -CAKE_INF;
                                if (!(_slice_lo_mask_0 & (1u << 7))) sv_split[7] = -CAKE_INF;
                                if (!(_slice_lo_mask_0 & (1u << 8))) sv_split[8] = -CAKE_INF;
                                if (!(_slice_lo_mask_0 & (1u << 9))) sv_split[9] = -CAKE_INF;
                                if (!(_slice_lo_mask_0 & (1u << 10))) sv_split[10] = -CAKE_INF;
                                if (!(_slice_lo_mask_0 & (1u << 11))) sv_split[11] = -CAKE_INF;
                                if (!(_slice_lo_mask_0 & (1u << 12))) sv_split[12] = -CAKE_INF;
                                if (!(_slice_lo_mask_0 & (1u << 13))) sv_split[13] = -CAKE_INF;
                                if (!(_slice_lo_mask_0 & (1u << 14))) sv_split[14] = -CAKE_INF;
                                if (!(_slice_lo_mask_0 & (1u << 15))) sv_split[15] = -CAKE_INF;
                                if (!(_slice_lo_mask_0 & (1u << 16))) sv_split[16] = -CAKE_INF;
                                if (!(_slice_lo_mask_0 & (1u << 17))) sv_split[17] = -CAKE_INF;
                                if (!(_slice_lo_mask_0 & (1u << 18))) sv_split[18] = -CAKE_INF;
                                if (!(_slice_lo_mask_0 & (1u << 19))) sv_split[19] = -CAKE_INF;
                                if (!(_slice_lo_mask_0 & (1u << 20))) sv_split[20] = -CAKE_INF;
                                if (!(_slice_lo_mask_0 & (1u << 21))) sv_split[21] = -CAKE_INF;
                                if (!(_slice_lo_mask_0 & (1u << 22))) sv_split[22] = -CAKE_INF;
                                if (!(_slice_lo_mask_0 & (1u << 23))) sv_split[23] = -CAKE_INF;
                                if (!(_slice_lo_mask_0 & (1u << 24))) sv_split[24] = -CAKE_INF;
                                if (!(_slice_lo_mask_0 & (1u << 25))) sv_split[25] = -CAKE_INF;
                                if (!(_slice_lo_mask_0 & (1u << 26))) sv_split[26] = -CAKE_INF;
                                if (!(_slice_lo_mask_0 & (1u << 27))) sv_split[27] = -CAKE_INF;
                                if (!(_slice_lo_mask_0 & (1u << 28))) sv_split[28] = -CAKE_INF;
                                if (!(_slice_lo_mask_0 & (1u << 29))) sv_split[29] = -CAKE_INF;
                                if (!(_slice_lo_mask_0 & (1u << 30))) sv_split[30] = -CAKE_INF;
                                if (!(_slice_lo_mask_0 & (1u << 31))) sv_split[31] = -CAKE_INF;
                                uint32_t _slice_lo_mask_1;
                                {
                                    int _lim_1 = valid_sparse_cols - 32;
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
                                if (!(_slice_lo_mask_1 & (1u << 0))) sv_split[32] = -CAKE_INF;
                                if (!(_slice_lo_mask_1 & (1u << 1))) sv_split[33] = -CAKE_INF;
                                if (!(_slice_lo_mask_1 & (1u << 2))) sv_split[34] = -CAKE_INF;
                                if (!(_slice_lo_mask_1 & (1u << 3))) sv_split[35] = -CAKE_INF;
                                if (!(_slice_lo_mask_1 & (1u << 4))) sv_split[36] = -CAKE_INF;
                                if (!(_slice_lo_mask_1 & (1u << 5))) sv_split[37] = -CAKE_INF;
                                if (!(_slice_lo_mask_1 & (1u << 6))) sv_split[38] = -CAKE_INF;
                                if (!(_slice_lo_mask_1 & (1u << 7))) sv_split[39] = -CAKE_INF;
                                if (!(_slice_lo_mask_1 & (1u << 8))) sv_split[40] = -CAKE_INF;
                                if (!(_slice_lo_mask_1 & (1u << 9))) sv_split[41] = -CAKE_INF;
                                if (!(_slice_lo_mask_1 & (1u << 10))) sv_split[42] = -CAKE_INF;
                                if (!(_slice_lo_mask_1 & (1u << 11))) sv_split[43] = -CAKE_INF;
                                if (!(_slice_lo_mask_1 & (1u << 12))) sv_split[44] = -CAKE_INF;
                                if (!(_slice_lo_mask_1 & (1u << 13))) sv_split[45] = -CAKE_INF;
                                if (!(_slice_lo_mask_1 & (1u << 14))) sv_split[46] = -CAKE_INF;
                                if (!(_slice_lo_mask_1 & (1u << 15))) sv_split[47] = -CAKE_INF;
                                if (!(_slice_lo_mask_1 & (1u << 16))) sv_split[48] = -CAKE_INF;
                                if (!(_slice_lo_mask_1 & (1u << 17))) sv_split[49] = -CAKE_INF;
                                if (!(_slice_lo_mask_1 & (1u << 18))) sv_split[50] = -CAKE_INF;
                                if (!(_slice_lo_mask_1 & (1u << 19))) sv_split[51] = -CAKE_INF;
                                if (!(_slice_lo_mask_1 & (1u << 20))) sv_split[52] = -CAKE_INF;
                                if (!(_slice_lo_mask_1 & (1u << 21))) sv_split[53] = -CAKE_INF;
                                if (!(_slice_lo_mask_1 & (1u << 22))) sv_split[54] = -CAKE_INF;
                                if (!(_slice_lo_mask_1 & (1u << 23))) sv_split[55] = -CAKE_INF;
                                if (!(_slice_lo_mask_1 & (1u << 24))) sv_split[56] = -CAKE_INF;
                                if (!(_slice_lo_mask_1 & (1u << 25))) sv_split[57] = -CAKE_INF;
                                if (!(_slice_lo_mask_1 & (1u << 26))) sv_split[58] = -CAKE_INF;
                                if (!(_slice_lo_mask_1 & (1u << 27))) sv_split[59] = -CAKE_INF;
                                if (!(_slice_lo_mask_1 & (1u << 28))) sv_split[60] = -CAKE_INF;
                                if (!(_slice_lo_mask_1 & (1u << 29))) sv_split[61] = -CAKE_INF;
                                if (!(_slice_lo_mask_1 & (1u << 30))) sv_split[62] = -CAKE_INF;
                                if (!(_slice_lo_mask_1 & (1u << 31))) sv_split[63] = -CAKE_INF;
                                float2 _reg_reduce_max2_2 = {-CAKE_INF, -CAKE_INF};
                                row_max_x32_accum(&sv_split[0], _reg_reduce_max2_2);
                                row_max_x32_accum(&sv_split[32], _reg_reduce_max2_2);
                                float sv_split_max = row_max_reduce(_reg_reduce_max2_2);
                                score_tile_max = sv_split_max;
                            }
                        }
                        {
                            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                            asm volatile(
                                "{\n\t"
                                ".reg .b32 remAddr32;\n\t"
                                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                                "}"
                                :: "r"(s_empty_addr + phase * 8), "r"(0) : "memory");
                            float _max_1 = max_noftz(new_max, score_tile_max);
                            new_max = _max_1;
                            mbarrier_wait(split_pv_p_empty_addr + (phase) * 8, s_wait_phase ^ 1);
                            smem_softmax_warp_pair_exchange[phase * 128 + stats_row] = new_max;
                            asm volatile("barrier.sync %0, 64;" :: "r"(2 + warp % 2) : "memory");
                            float _max_2 = max_noftz(new_max, smem_softmax_warp_pair_exchange[phase * 128 + (stats_row ^ 64)]);
                            new_max = _max_2;
                        }
                        float _fma_0 = __fmaf_rn(row_max_val, softmax_scale_log2, (-new_max) * softmax_scale_log2);
                        float delta = _fma_0;
                        float _exp2_0 = approx_exp2(delta);
                        float exp_delta = _exp2_0;
                        float acc_scale = ((row_max_val > -CAKE_INF) ? exp_delta : 1.0f);
                        smem_stats_max[phase * 128 + stats_row] = acc_scale;
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        mbarrier_arrive(stats_addr + (phase) * 8);
                        row_max_val = new_max;
                        float safe_max = ((new_max == -CAKE_INF) ? 0.0f : new_max);
                        float max_scaled = safe_max * softmax_scale_log2;
                        float block_sum = 0.0f;
                        {
                            {
                                if (valid_sparse_cols != 0) {
                                    const float2 _fma_b2_3 = {softmax_scale_log2, softmax_scale_log2};
                                    const float2 _fma_c2_4 = {-max_scaled, -max_scaled};
                                    #pragma unroll
                                    for (int _lf = 0; _lf < 32; _lf++)
                                        fma_f32x2_inplace(&reinterpret_cast<float2*>(sv_split)[_lf], _fma_b2_3, _fma_c2_4);
                                    #pragma unroll
                                    for (int _le = 0; _le < 64; _le++) {
                                        sv_split[_le] = approx_exp2(sv_split[_le]);
                                    }
                                    uint32_t _fp8_0[16];
                                    {
                                        uint32_t _packed;
                                        asm volatile("{\n\t"
                                            ".reg .b16 _lo;\n\t"
                                            ".reg .b16 _hi;\n\t"
                                            "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                            "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                            "mov.b32 %0, {_lo, _hi};\n\t"
                                            "}"
                                            : "=r"(_packed) : "f"(sv_split[0]), "f"(sv_split[1]),
                                                               "f"(sv_split[2]), "f"(sv_split[3]));
                                        _fp8_0[0] = _packed;
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
                                            : "=r"(_packed) : "f"(sv_split[4]), "f"(sv_split[5]),
                                                               "f"(sv_split[6]), "f"(sv_split[7]));
                                        _fp8_0[1] = _packed;
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
                                            : "=r"(_packed) : "f"(sv_split[8]), "f"(sv_split[9]),
                                                               "f"(sv_split[10]), "f"(sv_split[11]));
                                        _fp8_0[2] = _packed;
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
                                            : "=r"(_packed) : "f"(sv_split[12]), "f"(sv_split[13]),
                                                               "f"(sv_split[14]), "f"(sv_split[15]));
                                        _fp8_0[3] = _packed;
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
                                            : "=r"(_packed) : "f"(sv_split[16]), "f"(sv_split[17]),
                                                               "f"(sv_split[18]), "f"(sv_split[19]));
                                        _fp8_0[4] = _packed;
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
                                            : "=r"(_packed) : "f"(sv_split[20]), "f"(sv_split[21]),
                                                               "f"(sv_split[22]), "f"(sv_split[23]));
                                        _fp8_0[5] = _packed;
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
                                            : "=r"(_packed) : "f"(sv_split[24]), "f"(sv_split[25]),
                                                               "f"(sv_split[26]), "f"(sv_split[27]));
                                        _fp8_0[6] = _packed;
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
                                            : "=r"(_packed) : "f"(sv_split[28]), "f"(sv_split[29]),
                                                               "f"(sv_split[30]), "f"(sv_split[31]));
                                        _fp8_0[7] = _packed;
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
                                            : "=r"(_packed) : "f"(sv_split[32]), "f"(sv_split[33]),
                                                               "f"(sv_split[34]), "f"(sv_split[35]));
                                        _fp8_0[8] = _packed;
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
                                            : "=r"(_packed) : "f"(sv_split[36]), "f"(sv_split[37]),
                                                               "f"(sv_split[38]), "f"(sv_split[39]));
                                        _fp8_0[9] = _packed;
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
                                            : "=r"(_packed) : "f"(sv_split[40]), "f"(sv_split[41]),
                                                               "f"(sv_split[42]), "f"(sv_split[43]));
                                        _fp8_0[10] = _packed;
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
                                            : "=r"(_packed) : "f"(sv_split[44]), "f"(sv_split[45]),
                                                               "f"(sv_split[46]), "f"(sv_split[47]));
                                        _fp8_0[11] = _packed;
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
                                            : "=r"(_packed) : "f"(sv_split[48]), "f"(sv_split[49]),
                                                               "f"(sv_split[50]), "f"(sv_split[51]));
                                        _fp8_0[12] = _packed;
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
                                            : "=r"(_packed) : "f"(sv_split[52]), "f"(sv_split[53]),
                                                               "f"(sv_split[54]), "f"(sv_split[55]));
                                        _fp8_0[13] = _packed;
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
                                            : "=r"(_packed) : "f"(sv_split[56]), "f"(sv_split[57]),
                                                               "f"(sv_split[58]), "f"(sv_split[59]));
                                        _fp8_0[14] = _packed;
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
                                            : "=r"(_packed) : "f"(sv_split[60]), "f"(sv_split[61]),
                                                               "f"(sv_split[62]), "f"(sv_split[63]));
                                        _fp8_0[15] = _packed;
                                    }
                                    {
                                        int p_stage_base = smem_p_fp8_addr + (unsigned int)(phase * 8192);
                                        #pragma unroll
                                        for (int p_vec = 0; p_vec < 4; p_vec++) {
                                            asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((p_stage_base + (my_row * 128 + (n_half * 4 + p_vec) * 16 ^ (my_row * 128 + (n_half * 4 + p_vec) * 16 >> 7 & 7) << 4))), "r"(_fp8_0[p_vec * 4]), "r"(_fp8_0[p_vec * 4 + 1]), "r"(_fp8_0[p_vec * 4 + 2]), "r"(_fp8_0[p_vec * 4 + 3]) : "memory");
                                        }
                                    }
                                } else {
                                    int zero_p_stage = smem_p_fp8_addr + (unsigned int)(phase * 8192);
                                    #pragma unroll
                                    for (int zero_p_vec = 0; zero_p_vec < 4; zero_p_vec++) {
                                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((zero_p_stage + (my_row * 128 + (n_half * 4 + zero_p_vec) * 16 ^ (my_row * 128 + (n_half * 4 + zero_p_vec) * 16 >> 7 & 7) << 4))), "r"(0), "r"(0), "r"(0), "r"(0) : "memory");
                                    }
                                }
                            }
                        }
                        {
                            {
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            }
                        }
                        asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                        asm volatile("tcgen05.fence::before_thread_sync;");
                        {
                            asm volatile(".pragma \"next knob FenceCode\";\n" ::: "memory");
                        }
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                            "}"
                            :: "r"(p_full_addr + phase * 8), "r"(0) : "memory");
                        {
                            {
                                asm volatile(".pragma \"next knob FenceCode\";\n" ::: "memory");
                                if (valid_sparse_cols != 0) {
                                    const float2* _reg_reduce_src2_5 = reinterpret_cast<const float2*>(&sv_split[0]);
                                    float2 _reg_reduce_sum2_5_0 = make_float2(0.0f, 0.0f);
                                    float2 _reg_reduce_sum2_5_1 = make_float2(0.0f, 0.0f);
                                    float2 _reg_reduce_sum2_5_2 = make_float2(0.0f, 0.0f);
                                    float2 _reg_reduce_sum2_5_3 = make_float2(0.0f, 0.0f);
                                    #pragma unroll
                                    for (int _rr = 0; _rr < 8; _rr++) {
                                        add_f32x2_inplace(&_reg_reduce_sum2_5_0, _reg_reduce_src2_5[_rr * 4 + 0]);
                                        add_f32x2_inplace(&_reg_reduce_sum2_5_1, _reg_reduce_src2_5[_rr * 4 + 1]);
                                        add_f32x2_inplace(&_reg_reduce_sum2_5_2, _reg_reduce_src2_5[_rr * 4 + 2]);
                                        add_f32x2_inplace(&_reg_reduce_sum2_5_3, _reg_reduce_src2_5[_rr * 4 + 3]);
                                    }
                                    add_f32x2_inplace(&_reg_reduce_sum2_5_0, _reg_reduce_sum2_5_1);
                                    add_f32x2_inplace(&_reg_reduce_sum2_5_2, _reg_reduce_sum2_5_3);
                                    add_f32x2_inplace(&_reg_reduce_sum2_5_0, _reg_reduce_sum2_5_2);
                                    float sv_split_sum = _reg_reduce_sum2_5_0.x + _reg_reduce_sum2_5_0.y;
                                    block_sum = sv_split_sum;
                                }
                                float _fma_2 = __fmaf_rn(row_sum_val, acc_scale, block_sum);
                                row_sum_val = _fma_2;
                            }
                        }
                    }
                    smem_stats_sum[stats_row] = row_sum_val;
                    smem_stats_final_max[stats_row] = row_max_val;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(sum_ready_addr);
                    softmax_tile_cursor = softmax_tile_cursor + num_kv_tiles_1;
                }
                {
                    mbarrier_wait(source_work_full_addr + (source_work_stage_2) * 8, _phase_source_work_full_2);
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
                        : "r"(work_response_addr + source_work_stage_2 * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_9 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_9)
                        : "r"(work_response_addr + source_work_stage_2 * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_10 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_10)
                        : "r"(work_response_addr + source_work_stage_2 * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_11 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_11)
                        : "r"(work_response_addr + source_work_stage_2 * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(source_work_empty_addr + source_work_stage_2 * 8), "r"(0) : "memory");
                    source_work_stage_2 += 1;
                    if (source_work_stage_2 == 2) { source_work_stage_2 = 0; _phase_source_work_full_2 ^= 1; }
                    if (_clc_valid_3 == 0) {
                        break;
                    }
                    source_work_x_2 = _clc_ctaid_9 + (unsigned int)cta_rank;
                    source_work_z_2 = _clc_ctaid_11;
                }
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: correction_wg ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 144;");
        { // correction_wg_main
            float softmax_scale_log2_1 = bmm1_scale[0] * 1.4426950408889634f;
            float output_scale = bmm2_scale[0];
            const int wg_dummy_inc_1 = 0;
            int all_num_kv_tiles_3 = (sparse_topk + 128 - 1) / 128;
            const int tmem_row_base_1 = ((1) ? warp % 2 * 32 : warp % 4 * 32);
            const int n_half_1 = ((1) ? (int)(warp % 4 / 2) : 0);
            const int my_row_1 = (unsigned int)tmem_row_base_1 + lane;
            const int stats_row_1 = n_half_1 * 64 + my_row_1;
            const int corr_row = tmem_row_base_1 << 16;
            int correction_tile_cursor = 0;
            unsigned int source_work_stage_3 = 0;
            unsigned int source_work_x_3 = blockIdx.x;
            unsigned int source_work_z_3 = blockIdx.z;
            unsigned int _phase_q_full_0_2 = 0;
            unsigned int _phase_o_full_0 = 0;
            unsigned int _phase_o_empty_0 = 1;
            unsigned int _phase_sum_ready_0 = 0;
            unsigned int _phase_source_work_full_3 = 0;
            #pragma unroll 1
            for (unsigned int work_idx_2 = 0; work_idx_2 < 514; work_idx_2++) {
                int split_idx_3 = 0;
                int query_idx_3 = source_work_x_3 >> 1 >> 1;
                int v_chunk_1 = source_work_x_3 >> 1 & 1;
                {
                    split_idx_3 = 0;
                    query_idx_3 = source_work_x_3 >> 1;
                    v_chunk_1 = bid % 2;
                }
                int source_work_valid_3 = source_work_z_3 * (unsigned int)max_q_len + (source_work_x_3 >> 1) < (unsigned int)num_query_tokens;
                int mapped_query_idx_3 = source_work_z_3 * (unsigned int)max_q_len + (source_work_x_3 >> 1);
                if (source_work_valid_3 != 0) {
                    {
                        int active_topk_3 = sparse_topk_lens[mapped_query_idx_3];
                        all_num_kv_tiles_3 = (active_topk_3 + 128 - 1) / 128;
                    }
                    int tiles_per_split_3 = all_num_kv_tiles_3 + 1 - 1;
                    int first_tile_3 = split_idx_3 * tiles_per_split_3;
                    int num_kv_tiles_2 = tiles_per_split_3;
                    {
                        int first_stats_tile = correction_tile_cursor;
                        int first_stats_phase = first_stats_tile & 1;
                        int first_stats_wait_phase = first_stats_tile >> 1 & 1;
                        mbarrier_wait(stats_addr + (first_stats_phase) * 8, first_stats_wait_phase);
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                            "}"
                            :: "r"(o_empty_addr), "r"(0) : "memory");
                    }
                    #pragma unroll 1
                    for (int tile_3 = 1; tile_3 < num_kv_tiles_2; tile_3++) {
                        int pipeline_tile_1 = correction_tile_cursor + tile_3;
                        int phase_1 = pipeline_tile_1 & 1;
                        int stats_wait_phase = pipeline_tile_1 >> 1 & 1;
                        mbarrier_wait(stats_addr + (phase_1) * 8, stats_wait_phase);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        float acc_scale_1 = smem_stats_max[phase_1 * 128 + stats_row_1];
                        {
                            {
                                int prev_output_tile = pipeline_tile_1 - 1;
                                int o_full_phase = prev_output_tile & 1;
                                uint32_t _mbar_token_6 = mbarrier_try_wait(o_full_addr, o_full_phase);
                                mbarrier_wait_token(o_full_addr, o_full_phase, _mbar_token_6);
                            }
                            asm volatile("tcgen05.fence::after_thread_sync;");
                        }
                        {
                            int _vote_0 = __any_sync(0xFFFFFFFF, acc_scale_1 < 1.0f);
                            int any_rescale = _vote_0;
                            if (any_rescale != 0) {
                                #pragma unroll
                                for (int vs = 0; vs < 2; vs++) {
                                    int o_base = taddr + 256 + (unsigned int)(vs * 128) + (unsigned int)corr_row;
                                    #pragma unroll
                                    for (int c = 0; c < 128; c += 16) {
                                        float _tmem_load_3[16];
                                        tmem_ld_x16(&_tmem_load_3[0], o_base + c);
                                        const float2 _scale2_0 = {acc_scale_1, acc_scale_1};
                                        #pragma unroll
                                        for (int _ls = 0; _ls < 8; _ls++)
                                            mul_f32x2_inplace(&reinterpret_cast<float2*>(_tmem_load_3)[_ls], _scale2_0);
                                        tmem_st_x16_f32(o_base + c, _tmem_load_3);
                                    }
                                }
                                {
                                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                                }
                            }
                        }
                        {
                            asm volatile(
                                "{\n\t"
                                ".reg .b32 remAddr32;\n\t"
                                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                                "}"
                                :: "r"(o_empty_addr), "r"(0) : "memory");
                        }
                    }
                    {
                        int last_output_tile = correction_tile_cursor + num_kv_tiles_2 - 1;
                        int o_full_phase_1 = last_output_tile & 1;
                        uint32_t _mbar_token_7 = mbarrier_try_wait(o_full_addr, o_full_phase_1);
                        mbarrier_wait_token(o_full_addr, o_full_phase_1, _mbar_token_7);
                    }
                    mbarrier_wait(sum_ready_addr, _phase_sum_ready_0);
                    _phase_sum_ready_0 ^= 1;
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    float total_sum = smem_stats_sum[stats_row_1];
                    float final_max = smem_stats_final_max[stats_row_1];
                    {
                        smem_corr_warp_pair_exchange[stats_row_1] = total_sum;
                        asm volatile("barrier.sync %0, 64;" :: "r"(4 + warp % 2) : "memory");
                        total_sum = total_sum + smem_corr_warp_pair_exchange[stats_row_1 ^ 64];
                    }
                    float _rcp_0 = approx_rcp(total_sum);
                    float inv_sum = ((total_sum > 0.0f) ? _rcp_0 : 0.0f);
                    int head_idx = my_row_1;
                    {
                        head_idx = bid % 2 * 64 + my_row_1;
                    }
                    int direct_o_offset = (mapped_query_idx_3 * num_heads + head_idx) * 512 + v_chunk_1 * 256;
                    int partial_o_offset = (mapped_query_idx_3 * num_heads + head_idx + split_idx_3) * 512 + ((0) ? v_chunk_1 : n_half_1) * 256;
                    int o_offset = partial_o_offset;
                    #pragma unroll 1
                    for (int vs_1 = 0; vs_1 < 2; vs_1++) {
                        int o_base_epi = taddr + 256 + (unsigned int)(vs_1 * 128) + (unsigned int)corr_row;
                        #pragma unroll 1
                        for (int c_1 = 0; c_1 < 128; c_1 += 16) {
                            float _tmem_load_4[16];
                            tmem_ld_x16(&_tmem_load_4[0], o_base_epi + c_1);
                            int gmem_base = o_offset + vs_1 * 128 + c_1;
                            if (head_idx < num_heads) {
                                #pragma unroll
                                for (int j = 0; j < 16; j += 8) {
                                    {
                                        const float2 _prescale2_1 = {inv_sum * output_scale, inv_sum * output_scale};
                                        #if __CUDA_ARCH__ >= 1000
                                        #pragma unroll
                                        for (int _ps = 0; _ps < 4; _ps++)
                                            mul_f32x2_inplace(&reinterpret_cast<float2*>(&_tmem_load_4[j])[_ps], _prescale2_1);
                                        #else
                                        #pragma unroll
                                        for (int _ps = 0; _ps < 8; _ps++)
                                            _tmem_load_4[j + _ps] *= inv_sum * output_scale;
                                        #endif
                                        __nv_bfloat162 _pk[4];
                                        _pk[0] = __floats2bfloat162_rn(_tmem_load_4[j + 0], _tmem_load_4[j + 1]);
                                        _pk[1] = __floats2bfloat162_rn(_tmem_load_4[j + 2], _tmem_load_4[j + 3]);
                                        _pk[2] = __floats2bfloat162_rn(_tmem_load_4[j + 4], _tmem_load_4[j + 5]);
                                        _pk[3] = __floats2bfloat162_rn(_tmem_load_4[j + 6], _tmem_load_4[j + 7]);
                                        *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(O + (gmem_base + j)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                                    }
                                }
                            }
                        }
                    }
                    {
                        asm volatile("tcgen05.fence::before_thread_sync;");
                    }
                    correction_tile_cursor = correction_tile_cursor + num_kv_tiles_2;
                }
                {
                    mbarrier_wait(source_work_full_addr + (source_work_stage_3) * 8, _phase_source_work_full_3);
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
                        : "r"(work_response_addr + source_work_stage_3 * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_12 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_12)
                        : "r"(work_response_addr + source_work_stage_3 * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_13 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_13)
                        : "r"(work_response_addr + source_work_stage_3 * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_14 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_14)
                        : "r"(work_response_addr + source_work_stage_3 * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(source_work_empty_addr + source_work_stage_3 * 8), "r"(0) : "memory");
                    source_work_stage_3 += 1;
                    if (source_work_stage_3 == 2) { source_work_stage_3 = 0; _phase_source_work_full_3 ^= 1; }
                    if (_clc_valid_4 == 0) {
                        break;
                    }
                    source_work_x_3 = _clc_ctaid_12 + (unsigned int)cta_rank;
                    source_work_z_3 = _clc_ctaid_14;
                }
            }
            asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }
    // ---- Role: mma_warp ----
    if (warp == 8) {
        { // mma_warp_main
            const int wg2_dummy_1 = 0;
            int all_num_kv_tiles_4 = (sparse_topk + 128 - 1) / 128;
            unsigned int mma_k_stage = 0;
            unsigned int mma_v_stage = 0;
            unsigned int mma_kv_stage = 0;
            int mma_tile_cursor = 0;
            unsigned int source_work_stage_4 = 0;
            unsigned int source_work_x_4 = blockIdx.x;
            unsigned int source_work_z_4 = blockIdx.z;
            {
                if (cta_rank == 0) {
                    uint32_t _mbar_token_1 = mbarrier_try_wait(s_empty_addr, 1);
                    mbarrier_wait_token(s_empty_addr, 1, _mbar_token_1);
                    uint32_t _mbar_token_2 = mbarrier_try_wait(s_empty_addr + 8, 1);
                    mbarrier_wait_token(s_empty_addr + 8, 1, _mbar_token_2);
                }
            }
            unsigned int _phase_s_seeded_0 = 0;
            unsigned int _phase_q_full_0_3 = 0;
            unsigned int _phase_q_pair_ready_0 = 0;
            unsigned int _phase_k_full = 0;
            unsigned int _phase_k_full_1 = 0;
            unsigned int _phase_kv_pair_ready = 0;
            unsigned int _phase_source_work_full_4 = 0;
            #pragma unroll 1
            for (unsigned int work_idx_3 = 0; work_idx_3 < 514; work_idx_3++) {
                int split_idx_4 = 0;
                {
                    split_idx_4 = 0;
                }
                {
                    int query_idx_4 = source_work_x_4 >> 1;
                }
                int source_work_valid_4 = source_work_z_4 * (unsigned int)max_q_len + (source_work_x_4 >> 1) < (unsigned int)num_query_tokens;
                int mapped_query_idx_4 = source_work_z_4 * (unsigned int)max_q_len + (source_work_x_4 >> 1);
                if (source_work_valid_4 != 0) {
                    {
                        int active_topk_4 = sparse_topk_lens[mapped_query_idx_4];
                        all_num_kv_tiles_4 = (active_topk_4 + 128 - 1) / 128;
                    }
                    int tiles_per_split_4 = all_num_kv_tiles_4 + 1 - 1;
                    int first_tile_4 = split_idx_4 * tiles_per_split_4;
                    int num_kv_tiles_3 = tiles_per_split_4;
                    {
                        if (cta_rank == 0) {
                            mbarrier_wait(q_full_addr, _phase_q_full_0_3);
                            _phase_q_full_0_3 ^= 1;
                        }
                    }
                    int first_pv = 1;
                    {
                        int first_pipeline_tile = mma_tile_cursor;
                        int first_phase = first_pipeline_tile & 1;
                        int first_score_col = ((first_phase != 0) ? 128 : 0);
                        if (cta_rank == 0) {
                            mbarrier_wait(k_full_addr + (mma_k_stage) * 8, _phase_k_full);
                            int _mma_a_lo_0 = (((smem_q_full_addr) >> 4) & 0x3FFF) + (0) * 2048;
                            int _mma_b_lo_0 = (((smem_k_full_addr) >> 4) & 0x3FFF) + (mma_k_stage) * 2048;
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
                    :: "r"(_mma_a_lo_0), "r"(_mma_b_lo_0), "r"((tmem_tmem_scratch + (first_score_col))), "r"(0));
                            int _mma_a_lo_1 = (((smem_q_full_addr + 8192) >> 4) & 0x3FFF) + (0) * 2048;
                            int _mma_b_lo_1 = (((smem_k_full_addr + 8192) >> 4) & 0x3FFF) + (mma_k_stage) * 2048;
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
                    :: "r"(_mma_a_lo_1), "r"(_mma_b_lo_1), "r"((tmem_tmem_scratch + (first_score_col))), "r"(1));
                            int _mma_a_lo_2 = (((smem_q_full_addr + 16384) >> 4) & 0x3FFF) + (0) * 2048;
                            int _mma_b_lo_2 = (((smem_k_full_addr + 16384) >> 4) & 0x3FFF) + (mma_k_stage) * 2048;
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
                    :: "r"(_mma_a_lo_2), "r"(_mma_b_lo_2), "r"((tmem_tmem_scratch + (first_score_col))), "r"(1));
                            int _mma_a_lo_3 = (((smem_q_full_addr + 24576) >> 4) & 0x3FFF) + (0) * 2048;
                            int _mma_b_lo_3 = (((smem_k_full_addr + 24576) >> 4) & 0x3FFF) + (mma_k_stage) * 2048;
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
                    :: "r"(_mma_a_lo_3), "r"(_mma_b_lo_3), "r"((tmem_tmem_scratch + (first_score_col))), "r"(1));
                            if (cta_rank == 0) {
                                elect_commit_cg2_multicast(k_empty_addr + (mma_k_stage) * 8, (uint16_t)(3));
                            }
                            mma_k_stage += 1;
                            if (mma_k_stage == 2) { mma_k_stage = 0; _phase_k_full ^= 1; }
                            if (cta_rank == 0) {
                                elect_commit_cg2_multicast(s_full_addr + (first_phase) * 8, (uint16_t)(3));
                            }
                        }
                    }
                    #pragma unroll 1
                    for (int tile_4 = 1; tile_4 < num_kv_tiles_3; tile_4++) {
                        int pipeline_tile_2 = mma_tile_cursor + tile_4;
                        int phase_2 = pipeline_tile_2 & 1;
                        int score_col = ((phase_2 != 0) ? 128 : 0);
                        {
                            if (cta_rank == 0) {
                                mbarrier_wait(k_full_addr + (mma_k_stage) * 8, _phase_k_full);
                                int _mma_a_lo_4 = (((smem_q_full_addr) >> 4) & 0x3FFF) + (0) * 2048;
                                int _mma_b_lo_4 = (((smem_k_full_addr) >> 4) & 0x3FFF) + (mma_k_stage) * 2048;
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
                    :: "r"(_mma_a_lo_4), "r"(_mma_b_lo_4), "r"((tmem_tmem_scratch + (score_col))), "r"(0));
                                int _mma_a_lo_5 = (((smem_q_full_addr + 8192) >> 4) & 0x3FFF) + (0) * 2048;
                                int _mma_b_lo_5 = (((smem_k_full_addr + 8192) >> 4) & 0x3FFF) + (mma_k_stage) * 2048;
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
                    :: "r"(_mma_a_lo_5), "r"(_mma_b_lo_5), "r"((tmem_tmem_scratch + (score_col))), "r"(1));
                                int _mma_a_lo_6 = (((smem_q_full_addr + 16384) >> 4) & 0x3FFF) + (0) * 2048;
                                int _mma_b_lo_6 = (((smem_k_full_addr + 16384) >> 4) & 0x3FFF) + (mma_k_stage) * 2048;
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
                                int _mma_a_lo_7 = (((smem_q_full_addr + 24576) >> 4) & 0x3FFF) + (0) * 2048;
                                int _mma_b_lo_7 = (((smem_k_full_addr + 24576) >> 4) & 0x3FFF) + (mma_k_stage) * 2048;
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
                    :: "r"(_mma_a_lo_7), "r"(_mma_b_lo_7), "r"((tmem_tmem_scratch + (score_col))), "r"(1));
                                if (cta_rank == 0) {
                                    elect_commit_cg2_multicast(k_empty_addr + (mma_k_stage) * 8, (uint16_t)(3));
                                }
                                mma_k_stage += 1;
                                if (mma_k_stage == 2) { mma_k_stage = 0; _phase_k_full ^= 1; }
                                if (cta_rank == 0) {
                                    elect_commit_cg2_multicast(s_full_addr + (phase_2) * 8, (uint16_t)(3));
                                }
                                int next_s_tile = pipeline_tile_2 + 1;
                                int next_s_stage = next_s_tile & 1;
                                int next_s_empty_phase = next_s_tile >> 1 & 1 ^ 1;
                                uint32_t _mbar_token_3 = mbarrier_try_wait(s_empty_addr + (next_s_stage) * 8, next_s_empty_phase);
                                mbarrier_wait_token(s_empty_addr + (next_s_stage) * 8, next_s_empty_phase, _mbar_token_3);
                            }
                        }
                    }
                    {
                        if (cta_rank == 0) {
                            elect_commit_cg2_multicast(q_empty_addr, (uint16_t)(3));
                        }
                    }
                    int last_pipeline_tile = mma_tile_cursor + num_kv_tiles_3 - 1;
                    int last_phase = last_pipeline_tile & 1;
                    int drain_wait_phase = last_pipeline_tile >> 1 & 1;
                    {
                        int final_s_tile = last_pipeline_tile + 2;
                        int final_s_stage = final_s_tile & 1;
                        int final_s_empty_phase = final_s_tile >> 1 & 1 ^ 1;
                        if (cta_rank == 0) {
                            uint32_t _mbar_token_4 = mbarrier_try_wait(s_empty_addr + (final_s_stage) * 8, final_s_empty_phase);
                            mbarrier_wait_token(s_empty_addr + (final_s_stage) * 8, final_s_empty_phase, _mbar_token_4);
                        }
                    }
                    mma_tile_cursor = mma_tile_cursor + num_kv_tiles_3;
                }
                {
                    mbarrier_wait(source_work_full_addr + (source_work_stage_4) * 8, _phase_source_work_full_4);
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
                        : "r"(work_response_addr + source_work_stage_4 * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_6 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_6)
                        : "r"(work_response_addr + source_work_stage_4 * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_7 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_7)
                        : "r"(work_response_addr + source_work_stage_4 * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_8 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_8)
                        : "r"(work_response_addr + source_work_stage_4 * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(source_work_empty_addr + source_work_stage_4 * 8), "r"(0) : "memory");
                    source_work_stage_4 += 1;
                    if (source_work_stage_4 == 2) { source_work_stage_4 = 0; _phase_source_work_full_4 ^= 1; }
                    if (_clc_valid_2 == 0) {
                        break;
                    }
                    source_work_x_4 = _clc_ctaid_6 + (unsigned int)cta_rank;
                    source_work_z_4 = _clc_ctaid_8;
                }
            }
            mbarrier_arrive(tmem_dealloc_addr);
            unsigned int _phase_tmem_dealloc_0 = 0;
            mbarrier_wait(tmem_dealloc_addr, _phase_tmem_dealloc_0);
            _phase_tmem_dealloc_0 ^= 1;
            asm volatile("tcgen05.fence::after_thread_sync;");
            {
                int tmem_dealloc_peer_rank = bid % 2 ^ 1;
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
    // ---- Role: scheduler ----
    if (warp == 10) {
        { // scheduler_main
            const int wg2_dummy_2 = 0;
            unsigned int _phase_source_throttle_full = 0;
            unsigned int _phase_source_work_empty = 1;
            unsigned int _phase_source_work_full_5 = 0;
            unsigned int _phase_q_full_0_4 = 0;
            {
                unsigned int source_work_stage_5 = 0;
                unsigned int source_throttle_stage_1 = 0;
                if (cta_rank == 0) {
                    #pragma unroll 1
                    for (unsigned int _source_work = 0; _source_work < 514; _source_work++) {
                        mbarrier_wait(source_throttle_full_addr + (source_throttle_stage_1) * 8, _phase_source_throttle_full);
                        mbarrier_arrive(source_throttle_empty_addr + (source_throttle_stage_1) * 8);
                        source_throttle_stage_1 += 1;
                        if (source_throttle_stage_1 == 2) { source_throttle_stage_1 = 0; _phase_source_throttle_full ^= 1; }
                        mbarrier_wait(source_work_empty_addr + (source_work_stage_5) * 8, _phase_source_work_empty);
                        if (lane < 2) {
                            asm volatile(
                                "{\n\t"
                                ".reg .b32 remAddr32;\n\t"
                                "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                                "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                                "}"
                                :: "r"(source_work_full_addr + source_work_stage_5 * 8), "r"(lane), "r"((uint32_t)(16)) : "memory");
                        }
                        if (elect_sync()) {
                            asm volatile(
                                "fence.proxy.async.shared::cta;\n\t"
                                "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                    ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                                    " [%0], [%1];"
                                :: "r"(work_response_addr + source_work_stage_5 * 16 + 0 * 16), "r"(source_work_full_addr + source_work_stage_5 * 8)
                                : "memory");
                        }
                        mbarrier_wait(source_work_full_addr + (source_work_stage_5) * 8, _phase_source_work_full_5);
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
                            : "r"(work_response_addr + source_work_stage_5 * 16 + 0 * 16)
                            : "memory");
                        uint32_t _clc_ctaid_15 = 0;
                        asm volatile(
                            "{\n\t"
                            ".reg .b128 clc_r;\n\t"
                            "ld.shared.b128 clc_r, [%1];\n\t"
                            "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                            "}\n"
                            : "=r"(_clc_ctaid_15)
                            : "r"(work_response_addr + source_work_stage_5 * 16 + 0 * 16)
                            : "memory");
                        uint32_t _clc_ctaid_16 = 0;
                        asm volatile(
                            "{\n\t"
                            ".reg .b128 clc_r;\n\t"
                            "ld.shared.b128 clc_r, [%1];\n\t"
                            "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                            "}\n"
                            : "=r"(_clc_ctaid_16)
                            : "r"(work_response_addr + source_work_stage_5 * 16 + 0 * 16)
                            : "memory");
                        uint32_t _clc_ctaid_17 = 0;
                        asm volatile(
                            "{\n\t"
                            ".reg .b128 clc_r;\n\t"
                            "ld.shared.b128 clc_r, [%1];\n\t"
                            "clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                            "}\n"
                            : "=r"(_clc_ctaid_17)
                            : "r"(work_response_addr + source_work_stage_5 * 16 + 0 * 16)
                            : "memory");
                        asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                        source_work_stage_5 += 1;
                        if (source_work_stage_5 == 2) { source_work_stage_5 = 0; _phase_source_work_empty ^= 1; _phase_source_work_full_5 ^= 1; }
                        if (_clc_valid_5 == 0) {
                            break;
                        }
                    }
                    #pragma unroll
                    for (unsigned int _source_tail = 0; _source_tail < 2; _source_tail++) {
                        mbarrier_wait(source_work_empty_addr + (source_work_stage_5) * 8, _phase_source_work_empty);
                        source_work_stage_5 += 1;
                        if (source_work_stage_5 == 2) { source_work_stage_5 = 0; _phase_source_work_empty ^= 1; _phase_source_work_full_5 ^= 1; }
                    }
                }
            }
        }
    }
    // ---- Role: padding ----
    if (warp == 11) {
        { // padding_main
            const int wg3_dummy = 0;
            unsigned int source_work_stage_6 = 0;
            unsigned int source_work_x_5 = blockIdx.x;
            unsigned int source_work_z_5 = blockIdx.z;
            int pv_tile_cursor = 0;
            unsigned int pv_v_stage = 0;
            unsigned int _phase_source_work_full_6 = 0;
            #pragma unroll 1
            for (unsigned int _pv_work = 0; _pv_work < 514; _pv_work++) {
                int pv_valid = source_work_z_5 * (unsigned int)max_q_len + (source_work_x_5 >> 1) < (unsigned int)num_query_tokens;
                if (pv_valid != 0) {
                    int pv_query = source_work_z_5 * (unsigned int)max_q_len + (source_work_x_5 >> 1);
                    int pv_active = sparse_topk_lens[pv_query];
                    int pv_tiles = (pv_active + 128 - 1) / 128;
                    int pv_first = 1;
                    #pragma unroll 1
                    for (int pv_tile = 0; pv_tile < pv_tiles; pv_tile++) {
                        int pv_cursor = pv_tile_cursor + pv_tile;
                        int pv_phase = pv_cursor & 1;
                        int pv_wait_phase = pv_cursor >> 1 & 1;
                        if (cta_rank == 0) {
                            mbarrier_wait(p_full_addr + (pv_phase) * 8, pv_wait_phase);
                            mbarrier_wait(o_empty_addr, pv_cursor & 1);
                            mbarrier_wait(v_full_addr + (pv_v_stage) * 8, pv_wait_phase);
                            int _mma_a_lo_10 = (((smem_p_fp8_addr) >> 4) & 0x3FFF) + (pv_phase) * 512;
                            int _mma_b_lo_10 = ((((smem_v_full_addr) >> 4) & 0x3FFF) | 0x4000000) + (pv_v_stage) * 2048;
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
                    :: "r"(_mma_a_lo_10), "r"(_mma_b_lo_10), "r"((tmem_tmem_scratch + (256))), "r"(((pv_first) ? 0 : 1)));
                            int _mma_a_lo_11 = (((smem_p_fp8_addr) >> 4) & 0x3FFF) + (pv_phase) * 512;
                            int _mma_b_lo_11 = ((((smem_v_full_addr + 16384) >> 4) & 0x3FFF) | 0x4000000) + (pv_v_stage) * 2048;
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
                    :: "r"(_mma_a_lo_11), "r"(_mma_b_lo_11), "r"((tmem_tmem_scratch + (384))), "r"(((pv_first) ? 0 : 1)));
                            if (cta_rank == 0) {
                                elect_commit_cg2_multicast(v_empty_addr + (pv_v_stage) * 8, (uint16_t)(3));
                            }
                            pv_v_stage += 1;
                            if (pv_v_stage == 2) { pv_v_stage = 0; }
                            if (cta_rank == 0) {
                                elect_commit_cg2_multicast(split_pv_p_empty_addr + (pv_phase) * 8, (uint16_t)(3));
                            }
                            if (cta_rank == 0) {
                                elect_commit_cg2_multicast(o_full_addr, (uint16_t)(3));
                            }
                        }
                        pv_first = 0;
                    }
                    pv_tile_cursor = pv_tile_cursor + pv_tiles;
                }
                mbarrier_wait(source_work_full_addr + (source_work_stage_6) * 8, _phase_source_work_full_6);
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
                    : "r"(work_response_addr + source_work_stage_6 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_18 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_18)
                    : "r"(work_response_addr + source_work_stage_6 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_19 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_19)
                    : "r"(work_response_addr + source_work_stage_6 * 16 + 0 * 16)
                    : "memory");
                uint32_t _clc_ctaid_20 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::z.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_20)
                    : "r"(work_response_addr + source_work_stage_6 * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(source_work_empty_addr + source_work_stage_6 * 8), "r"(0) : "memory");
                source_work_stage_6 += 1;
                if (source_work_stage_6 == 2) { source_work_stage_6 = 0; _phase_source_work_full_6 ^= 1; }
                if (_clc_valid_6 == 0) {
                    break;
                }
                source_work_x_5 = _clc_ctaid_18 + (unsigned int)cta_rank;
                source_work_z_5 = _clc_ctaid_20;
            }
            asm volatile("tcgen05.fence::before_thread_sync;");
            mbarrier_arrive(tmem_dealloc_addr);
        }
    }

    // Cleanup
}

} // extern "C"
