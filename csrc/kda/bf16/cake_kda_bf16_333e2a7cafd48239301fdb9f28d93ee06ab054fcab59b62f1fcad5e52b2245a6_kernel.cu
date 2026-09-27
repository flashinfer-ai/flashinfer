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
#define TMEM_NCOLS 256
#define TMEM_TMEM_STATE_OFFSET 64
#define TMEM_TMEM_STATE_DECAY0_OFFSET 64
#define TMEM_TMEM_STATE_DECAY1_OFFSET 80
#define TMEM_TMEM_STATE_DECAY2_OFFSET 96
#define TMEM_TMEM_STATE_DECAY3_OFFSET 112
#define TMEM_TMEM_STATE_DECAY4_OFFSET 128
#define TMEM_TMEM_STATE_DECAY5_OFFSET 144
#define TMEM_TMEM_STATE_DECAY6_OFFSET 160
#define TMEM_TMEM_STATE_DECAY7_OFFSET 176
#define TMEM_TMEM_STATE_INP_OFFSET 0
#define TMEM_TMEM_STATE_INP_DECAY0_OFFSET 0
#define TMEM_TMEM_STATE_INP_DECAY1_OFFSET 8
#define TMEM_TMEM_STATE_INP_DECAY2_OFFSET 16
#define TMEM_TMEM_STATE_INP_DECAY3_OFFSET 24
#define TMEM_TMEM_STATE_INP_DECAY4_OFFSET 32
#define TMEM_TMEM_STATE_INP_DECAY5_OFFSET 40
#define TMEM_TMEM_STATE_INP_DECAY6_OFFSET 48
#define TMEM_TMEM_STATE_INP_DECAY7_OFFSET 56
#define TMEM_TMEM_U_ACC_OFFSET 224
#define TMEM_TMEM_U2_INP_OFFSET 224
#define TMEM_TMEM_U2_ACC_OFFSET 0
#define TMEM_TMEM_OUT_OFFSET 192
#define TMEM_TMEM_STATE_OUT_OFFSET 64
#define NUM_CHUNK_PIPE_STAGES 5
#define NUM_CHECKPOINT_PIPE_STAGES 2
#define NUM_SNAPSHOT_PIPE_STAGES 4
#define SMEM_SMEM_QD_OFF 1024
#define SMEM_SMEM_QD_STAGE_BYTES 8192
#define SMEM_SMEM_QD_STRIDE 41984
#define SMEM_SMEM_G_RAW_OFF 1024
#define SMEM_SMEM_G_RAW_STAGE_BYTES 8192
#define SMEM_SMEM_G_RAW_STRIDE 41984
#define SMEM_SMEM_G_RAW_ALL_OFF 1024
#define SMEM_SMEM_G_RAW_ALL_STAGE_BYTES 176128
#define SMEM_SMEM_G_RAW_ALL_STRIDE 176128
#define SMEM_SMEM_KD_OFF 9216
#define SMEM_SMEM_KD_STAGE_BYTES 8192
#define SMEM_SMEM_KD_STRIDE 41984
#define SMEM_SMEM_Q_RAW_PREFETCH_OFF 17408
#define SMEM_SMEM_Q_RAW_PREFETCH_STAGE_BYTES 8192
#define SMEM_SMEM_Q_RAW_PREFETCH_STRIDE 41984
#define SMEM_SMEM_FINAL_TRANS_OFF 17408
#define SMEM_SMEM_FINAL_TRANS_STAGE_BYTES 12288
#define SMEM_SMEM_FINAL_TRANS_STRIDE 41984
#define SMEM_SMEM_KR_TRANS_OFF 17408
#define SMEM_SMEM_KR_TRANS_STAGE_BYTES 8192
#define SMEM_SMEM_KR_TRANS_STRIDE 41984
#define SMEM_SMEM_MQK_TRANS_OFF 25600
#define SMEM_SMEM_MQK_TRANS_STAGE_BYTES 2048
#define SMEM_SMEM_MQK_TRANS_STRIDE 41984
#define SMEM_SMEM_FINAL_MQK_SLAB_OFF 25600
#define SMEM_SMEM_FINAL_MQK_SLAB_STAGE_BYTES 4096
#define SMEM_SMEM_FINAL_MQK_SLAB_STRIDE 41984
#define SMEM_SMEM_INV_OFF 29696
#define SMEM_SMEM_INV_STAGE_BYTES 2048
#define SMEM_SMEM_INV_STRIDE 41984
#define SMEM_SMEM_V_OFF 32384
#define SMEM_SMEM_V_STAGE_BYTES 8192
#define SMEM_SMEM_V_STRIDE 41984
#define SMEM_SMEM_SHORT_N32_V_OFF 168960
#define SMEM_SMEM_SHORT_N32_V_STAGE_BYTES 32768
#define SMEM_SMEM_SHORT_N32_V_STRIDE 32768
#define SMEM_SMEM_KI_OFF 17408
#define SMEM_SMEM_KI_STAGE_BYTES 8192
#define SMEM_SMEM_KI_STRIDE 41984
#define SMEM_SMEM_GATE_OFF 25600
#define SMEM_SMEM_GATE_STAGE_BYTES 16384
#define SMEM_SMEM_GATE_STRIDE 41984
#define SMEM_SMEM_BETA_RAW_OFF 41984
#define SMEM_SMEM_BETA_RAW_STAGE_BYTES 512
#define SMEM_SMEM_BETA_RAW_STRIDE 41984
#define SMEM_SMEM_BETA_RAW_ALL_OFF 41984
#define SMEM_SMEM_BETA_RAW_ALL_STAGE_BYTES 168448
#define SMEM_SMEM_BETA_RAW_ALL_STRIDE 168448
#define SMEM_SMEM_INV_WORK_OFF 32384
#define SMEM_SMEM_INV_WORK_STAGE_BYTES 4096
#define SMEM_SMEM_INV_WORK_STRIDE 41984
#define SMEM_SMEM_OUT_OFF 210944
#define SMEM_SMEM_OUT_STAGE_BYTES 8192
#define SMEM_SMEM_OUT_STRIDE 8192
#define SMEM_SMEM_V18_OFF 210944
#define SMEM_SMEM_V18_STAGE_BYTES 16384
#define SMEM_SMEM_V18_STRIDE 16384
#define SMEM_SMEM_CHECKPOINT_OFF 228352
#define SMEM_SMEM_CHECKPOINT_STAGE_BYTES 32768
#define SMEM_SMEM_CHECKPOINT_STRIDE 32768
#define SMEM_SMEM_RESTORE_FACTOR_ALL_OFF 41984
#define SMEM_SMEM_RESTORE_FACTOR_ALL_STAGE_BYTES 168452
#define SMEM_SMEM_RESTORE_FACTOR_ALL_STRIDE 168452
#define SMEM_SMEM_V21_OFF 41984
#define SMEM_SMEM_V21_STAGE_BYTES 168448
#define SMEM_SMEM_V21_STRIDE 168448
#define SMEM_SMEM_GT_PREFIX_ALL_OFF 41472
#define SMEM_SMEM_GT_PREFIX_ALL_STAGE_BYTES 168448
#define SMEM_SMEM_GT_PREFIX_ALL_STRIDE 168448
#define SMEM_SMEM_GT_ALL_OFF 31744
#define SMEM_SMEM_GT_ALL_STAGE_BYTES 168448
#define SMEM_SMEM_GT_ALL_STRIDE 168448
#define SMEM_SMEM_PREP_BETA_ALL_OFF 42500
#define SMEM_SMEM_PREP_BETA_ALL_STAGE_BYTES 168064
#define SMEM_SMEM_PREP_BETA_ALL_STRIDE 168064
#define SMEM_SMEM_PREP_BETA_BF16_ALL_OFF 42500
#define SMEM_SMEM_PREP_BETA_BF16_ALL_STAGE_BYTES 168000
#define SMEM_SMEM_PREP_BETA_BF16_ALL_STRIDE 168000
#define SMEM_SMEM_PREP_BETA_U32_ALL_OFF 42500
#define SMEM_SMEM_PREP_BETA_U32_ALL_STAGE_BYTES 168000
#define SMEM_SMEM_PREP_BETA_U32_ALL_STRIDE 168000
#define SMEM_SMEM_GATE_RATE_ALL_OFF 42628
#define SMEM_SMEM_GATE_RATE_ALL_STAGE_BYTES 167940
#define SMEM_SMEM_GATE_RATE_ALL_STRIDE 167940
#define SMEM_SMEM_GATE_BIAS_ALL_OFF 227408
#define SMEM_SMEM_GATE_BIAS_ALL_STAGE_BYTES 512
#define SMEM_SMEM_GATE_BIAS_ALL_STRIDE 512
#define SMEM_SMEM_STATE_DECAY_DIAG_RAW_OFF 1024
#define SMEM_SMEM_STATE_DECAY_DIAG_RAW_STAGE_BYTES 4096
#define SMEM_SMEM_STATE_DECAY_DIAG_RAW_STRIDE 4096
#define SMEM_SMEM_STATE_DECAY_DIAG0_OFF 1024
#define SMEM_SMEM_STATE_DECAY_DIAG0_STAGE_BYTES 512
#define SMEM_SMEM_STATE_DECAY_DIAG0_STRIDE 512
#define SMEM_SMEM_STATE_DECAY_DIAG1_OFF 1536
#define SMEM_SMEM_STATE_DECAY_DIAG1_STAGE_BYTES 512
#define SMEM_SMEM_STATE_DECAY_DIAG1_STRIDE 512
#define SMEM_SMEM_STATE_DECAY_DIAG2_OFF 2048
#define SMEM_SMEM_STATE_DECAY_DIAG2_STAGE_BYTES 512
#define SMEM_SMEM_STATE_DECAY_DIAG2_STRIDE 512
#define SMEM_SMEM_STATE_DECAY_DIAG3_OFF 2560
#define SMEM_SMEM_STATE_DECAY_DIAG3_STAGE_BYTES 512
#define SMEM_SMEM_STATE_DECAY_DIAG3_STRIDE 512
#define SMEM_SMEM_STATE_DECAY_DIAG4_OFF 3072
#define SMEM_SMEM_STATE_DECAY_DIAG4_STAGE_BYTES 512
#define SMEM_SMEM_STATE_DECAY_DIAG4_STRIDE 512
#define SMEM_SMEM_STATE_DECAY_DIAG5_OFF 3584
#define SMEM_SMEM_STATE_DECAY_DIAG5_STAGE_BYTES 512
#define SMEM_SMEM_STATE_DECAY_DIAG5_STRIDE 512
#define SMEM_SMEM_STATE_DECAY_DIAG6_OFF 4096
#define SMEM_SMEM_STATE_DECAY_DIAG6_STAGE_BYTES 512
#define SMEM_SMEM_STATE_DECAY_DIAG6_STRIDE 512
#define SMEM_SMEM_STATE_DECAY_DIAG7_OFF 4608
#define SMEM_SMEM_STATE_DECAY_DIAG7_STAGE_BYTES 512
#define SMEM_SMEM_STATE_DECAY_DIAG7_STRIDE 512
#define SMEM_SMEM_V_ALL_OFF 32384
#define SMEM_SMEM_V_ALL_STAGE_BYTES 176128
#define SMEM_SMEM_V_ALL_STRIDE 176128
#define SMEM_SMEM_GATE_ALL_OFF 25600
#define SMEM_SMEM_GATE_ALL_STAGE_BYTES 184320
#define SMEM_SMEM_GATE_ALL_STRIDE 184320
#define SMEM_SMEM_STATE_CHECKPOINT_NEEDED_OFF 227328
#define SMEM_SMEM_STATE_CHECKPOINT_NEEDED_STAGE_BYTES 80
#define SMEM_SMEM_STATE_CHECKPOINT_NEEDED_STRIDE 80
#define SMEM_SMEM_WORK_ITEM_WARP_MAX_OFF 227328
#define SMEM_SMEM_WORK_ITEM_WARP_MAX_STAGE_BYTES 16
#define SMEM_SMEM_WORK_ITEM_WARP_MAX_STRIDE 16
#define SMEM_SMEM_WORK_ITEM_COMPUTE_START_OFF 227344
#define SMEM_SMEM_WORK_ITEM_COMPUTE_START_STAGE_BYTES 4
#define SMEM_SMEM_WORK_ITEM_COMPUTE_START_STRIDE 4
#define SMEM_SMEM_WORK_ITEM_RESOLVED_OFF 227348
#define SMEM_SMEM_WORK_ITEM_RESOLVED_STAGE_BYTES 4
#define SMEM_SMEM_WORK_ITEM_RESOLVED_STRIDE 4
#define SMEM_TOTAL 227968
#define STORE_BACKWARD_TAPE 0
#define STORE_E_TAPE 1
#define SPLIT_WORK_ITEMS 0

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


__device__ __forceinline__ float approx_exp2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
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


__device__ __forceinline__ void tma_store_3d(
    const void *tmap, int x, int y, int z, unsigned smem_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
        " [%0, {%1, %2, %3}], [%4];"
        :: "l"(tmap), "r"(x), "r"(y), "r"(z), "r"(smem_addr) : "memory");
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

__global__ __launch_bounds__(1024) void
kernel_cake_kda_bf16_333e2a7cafd48239301fdb9f28d93ee06ab054fcab59b62f1fcad5e52b2245a6(__nv_bfloat16* __restrict__ q, CakeTensorMap const* q_tma, __nv_bfloat16* __restrict__ k, CakeTensorMap const* k_tma, __nv_bfloat16* __restrict__ v, CakeTensorMap const* v_tma, __nv_bfloat16* __restrict__ g, CakeTensorMap const* g_tma, __nv_bfloat16* __restrict__ beta, CakeTensorMap const* beta_tma, float* __restrict__ A_log, float* __restrict__ dt_bias, long long* __restrict__ cu_seqlens, int* __restrict__ seq_order, __nv_bfloat16* __restrict__ initial_state, __nv_bfloat16* __restrict__ out, CakeTensorMap const* out_tma, __nv_bfloat16* __restrict__ final_state, int num_heads, int use_initial_state, int store_final_state, float scale, float lower_bound, unsigned long long state_indices_addr, unsigned long long state_checkpoints_addr, unsigned long long checkpoint_cu_starts_addr, long long beta_token_stride, long long state_slot_stride, int use_state_indices, int checkpoint_every_n_tokens, long long* __restrict__ cu_chunk_offsets, __nv_bfloat16* __restrict__ chunk_state, unsigned int* __restrict__ state_checkpoint_needed, __nv_bfloat16* __restrict__ tape_qd, __nv_bfloat16* __restrict__ tape_kd, __nv_bfloat16* __restrict__ tape_kr, __nv_bfloat16* __restrict__ tape_j, float* __restrict__ tape_restore_factor, __nv_bfloat16* __restrict__ tape_e, __nv_bfloat16* __restrict__ tape_x, __nv_bfloat16* __restrict__ tape_r, float* __restrict__ norm_inv_out, __nv_bfloat16* __restrict__ decay_out, float* __restrict__ beta_active_out, float* __restrict__ initial_state_f32, unsigned int* __restrict__ zero_workspace, int zero_words, int num_sequences, CakeTensorMap const* state_checkpoints_tma, float* __restrict__ final_state_f32, long long g_token_stride)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define qk_full_addr (mbar_base + 0)
    #define gate_raw_full_addr (mbar_base + 40)
    #define qk_raw_full_addr (mbar_base + 80)
    #define v_full_addr (mbar_base + 120)
    #define v_free_addr (mbar_base + 160)
    #define smem_free_addr (mbar_base + 200)
    #define raw_inputs_free_addr (mbar_base + 240)
    #define state_inp_ready_addr (mbar_base + 280)
    #define old_out_ready_addr (mbar_base + 320)
    #define u_inp_ready_addr (mbar_base + 360)
    #define u2_acc_ready_addr (mbar_base + 400)
    #define u2_inp_ready_addr (mbar_base + 440)
    #define final_ready_addr (mbar_base + 480)
    #define out_empty_addr (mbar_base + 520)
    #define tmem_dealloc_ready_addr (mbar_base + 528)
    #define checkpoint_ready_addr (mbar_base + 536)
    #define checkpoint_free_addr (mbar_base + 552)
    #define checkpoint_snapshot_done_addr (mbar_base + 568)
    #define prep_diag_ready_addr (mbar_base + 600)
    #define prep_inv16_ready_addr (mbar_base + 640)
    #define work_item_ready_addr (mbar_base + 680)
    #define short_beta_ready_addr (mbar_base + 688)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    if (warp == 8 && lane == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(q_tma)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(k_tma)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(v_tma)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(g_tma)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(beta_tma)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(out_tma)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(state_checkpoints_tma)) : "memory");
    }


    // Kernel setup ops
    __nv_bfloat16* smem_qd = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_qd_addr = smem + 1024;
    __nv_bfloat16* smem_g_raw = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_g_raw_addr = smem + 1024;
    __nv_bfloat16* smem_g_raw_all = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_g_raw_all_addr = smem + 1024;
    __nv_bfloat16* smem_kd = reinterpret_cast<__nv_bfloat16*>(smem_raw + 9216);
    const int smem_kd_addr = smem + 9216;
    __nv_bfloat16* smem_q_raw_prefetch = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int smem_q_raw_prefetch_addr = smem + 17408;
    __nv_bfloat16* smem_final_trans = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int smem_final_trans_addr = smem + 17408;
    __nv_bfloat16* smem_kr_trans = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int smem_kr_trans_addr = smem + 17408;
    __nv_bfloat16* smem_mqk_trans = reinterpret_cast<__nv_bfloat16*>(smem_raw + 25600);
    const int smem_mqk_trans_addr = smem + 25600;
    __nv_bfloat16* smem_final_mqk_slab = reinterpret_cast<__nv_bfloat16*>(smem_raw + 25600);
    const int smem_final_mqk_slab_addr = smem + 25600;
    __nv_bfloat16* smem_inv = reinterpret_cast<__nv_bfloat16*>(smem_raw + 29696);
    const int smem_inv_addr = smem + 29696;
    __nv_bfloat16* smem_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 32384);
    const int smem_v_addr = smem + 32384;
    __nv_bfloat16* smem_short_n32_v = reinterpret_cast<__nv_bfloat16*>(smem_raw + 168960);
    const int smem_short_n32_v_addr = smem + 168960;
    __nv_bfloat16* smem_ki = reinterpret_cast<__nv_bfloat16*>(smem_raw + 17408);
    const int smem_ki_addr = smem + 17408;
    float* smem_gate = reinterpret_cast<float*>(smem_raw + 25600);
    const int smem_gate_addr = smem + 25600;
    __nv_bfloat16* smem_beta_raw = reinterpret_cast<__nv_bfloat16*>(smem_raw + 41984);
    const int smem_beta_raw_addr = smem + 41984;
    __nv_bfloat16* smem_beta_raw_all = reinterpret_cast<__nv_bfloat16*>(smem_raw + 41984);
    const int smem_beta_raw_all_addr = smem + 41984;
    __nv_bfloat16* smem_inv_work = reinterpret_cast<__nv_bfloat16*>(smem_raw + 32384);
    const int smem_inv_work_addr = smem + 32384;
    __nv_bfloat16* smem_out = reinterpret_cast<__nv_bfloat16*>(smem_raw + 210944);
    const int smem_out_addr = smem + 210944;
    float* smem_v18 = reinterpret_cast<float*>(smem_raw + 210944);
    const int smem_v18_addr = smem + 210944;
    __nv_bfloat16* smem_checkpoint = reinterpret_cast<__nv_bfloat16*>(smem_raw + 228352);
    const int smem_checkpoint_addr = smem + 228352;
    float* smem_restore_factor_all = reinterpret_cast<float*>(smem_raw + 41984);
    const int smem_restore_factor_all_addr = smem + 41984;
    int* smem_v21 = reinterpret_cast<int*>(smem_raw + 41984);
    const int smem_v21_addr = smem + 41984;
    float* smem_gt_prefix_all = reinterpret_cast<float*>(smem_raw + 41472);
    const int smem_gt_prefix_all_addr = smem + 41472;
    float* smem_gt_all = reinterpret_cast<float*>(smem_raw + 31744);
    const int smem_gt_all_addr = smem + 31744;
    float* smem_prep_beta_all = reinterpret_cast<float*>(smem_raw + 42500);
    const int smem_prep_beta_all_addr = smem + 42500;
    __nv_bfloat16* smem_prep_beta_bf16_all = reinterpret_cast<__nv_bfloat16*>(smem_raw + 42500);
    const int smem_prep_beta_bf16_all_addr = smem + 42500;
    unsigned int* smem_prep_beta_u32_all = reinterpret_cast<unsigned int*>(smem_raw + 42500);
    const int smem_prep_beta_u32_all_addr = smem + 42500;
    float* smem_gate_rate_all = reinterpret_cast<float*>(smem_raw + 42628);
    const int smem_gate_rate_all_addr = smem + 42628;
    float* smem_gate_bias_all = reinterpret_cast<float*>(smem_raw + 227408);
    const int smem_gate_bias_all_addr = smem + 227408;
    unsigned int* smem_state_decay_diag_raw = reinterpret_cast<unsigned int*>(smem_raw + 1024);
    const int smem_state_decay_diag_raw_addr = smem + 1024;
    __nv_bfloat16* smem_state_decay_diag0 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1024);
    const int smem_state_decay_diag0_addr = smem + 1024;
    __nv_bfloat16* smem_state_decay_diag1 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 1536);
    const int smem_state_decay_diag1_addr = smem + 1536;
    __nv_bfloat16* smem_state_decay_diag2 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 2048);
    const int smem_state_decay_diag2_addr = smem + 2048;
    __nv_bfloat16* smem_state_decay_diag3 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 2560);
    const int smem_state_decay_diag3_addr = smem + 2560;
    __nv_bfloat16* smem_state_decay_diag4 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 3072);
    const int smem_state_decay_diag4_addr = smem + 3072;
    __nv_bfloat16* smem_state_decay_diag5 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 3584);
    const int smem_state_decay_diag5_addr = smem + 3584;
    __nv_bfloat16* smem_state_decay_diag6 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 4096);
    const int smem_state_decay_diag6_addr = smem + 4096;
    __nv_bfloat16* smem_state_decay_diag7 = reinterpret_cast<__nv_bfloat16*>(smem_raw + 4608);
    const int smem_state_decay_diag7_addr = smem + 4608;
    __nv_bfloat16* smem_v_all = reinterpret_cast<__nv_bfloat16*>(smem_raw + 32384);
    const int smem_v_all_addr = smem + 32384;
    float* smem_gate_all = reinterpret_cast<float*>(smem_raw + 25600);
    const int smem_gate_all_addr = smem + 25600;
    unsigned int* smem_state_checkpoint_needed = reinterpret_cast<unsigned int*>(smem_raw + 227328);
    const int smem_state_checkpoint_needed_addr = smem + 227328;
    float* smem_work_item_warp_max = reinterpret_cast<float*>(smem_raw + 227328);
    const int smem_work_item_warp_max_addr = smem + 227328;
    int* smem_work_item_compute_start = reinterpret_cast<int*>(smem_raw + 227344);
    const int smem_work_item_compute_start_addr = smem + 227344;
    unsigned int* smem_work_item_resolved = reinterpret_cast<unsigned int*>(smem_raw + 227348);
    const int smem_work_item_resolved_addr = smem + 227348;

    // Mbarrier init (22 pipeline groups, 0 ordered-sequence groups, 91 barriers)
    // Mbarriers at smem_raw[0..728)

    if (warp == 10) {
        // --- pipeline 'chunk_pipe' ---
        // qk_full: 5 barriers, init_count=128
        // gate_raw_full: 5 barriers, init_count=1
        // qk_raw_full: 5 barriers, init_count=1
        // v_full: 5 barriers, init_count=1
        // v_free: 5 barriers, init_count=128
        // smem_free: 5 barriers, init_count=128
        // raw_inputs_free: 5 barriers, init_count=1
        // state_inp_ready: 5 barriers, init_count=4
        // old_out_ready: 5 barriers, init_count=1
        // u_inp_ready: 5 barriers, init_count=4
        // u2_acc_ready: 5 barriers, init_count=1
        // u2_inp_ready: 5 barriers, init_count=4
        // final_ready: 5 barriers, init_count=1
        // out_empty: 1 barriers, init_count=1
        // tmem_dealloc_ready: 1 barriers, init_count=2
        // --- pipeline 'checkpoint_pipe' ---
        // checkpoint_ready: 2 barriers, init_count=4
        // checkpoint_free: 2 barriers, init_count=1
        // --- pipeline 'snapshot_pipe' ---
        // checkpoint_snapshot_done: 4 barriers, init_count=4
        // --- pipeline 'chunk_pipe' ---
        // prep_diag_ready: 5 barriers, init_count=64
        // prep_inv16_ready: 5 barriers, init_count=64
        // work_item_ready: 1 barriers, init_count=1
        // short_beta_ready: 5 barriers, init_count=1
        // Warp-cooperative initialization in physical record order.
        uint32_t _mbarrier_init_count_10_0 = 1;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_0) : "r"(lane), "n"(30), "r"((uint32_t)(128)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_0) : "r"(lane), "n"(20), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_0) : "r"(lane), "n"(5), "r"((uint32_t)(128)));
        mbarrier_init(smem + 0 + lane * 8, _mbarrier_init_count_10_0);
        uint32_t _mbarrier_init_count_10_32 = 1;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_32) : "r"(lane), "n"(28), "r"((uint32_t)(4)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_32) : "r"(lane), "n"(23), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_32) : "r"(lane), "n"(18), "r"((uint32_t)(4)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_32) : "r"(lane), "n"(13), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_32) : "r"(lane), "n"(8), "r"((uint32_t)(4)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_32) : "r"(lane), "n"(3), "r"((uint32_t)(1)));
        mbarrier_init(smem + 256 + lane * 8, _mbarrier_init_count_10_32);
        uint32_t _mbarrier_init_count_10_64 = 1;
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_64) : "r"(lane), "n"(21), "r"((uint32_t)(64)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_64) : "r"(lane), "n"(11), "r"((uint32_t)(4)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_64) : "r"(lane), "n"(7), "r"((uint32_t)(1)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_64) : "r"(lane), "n"(5), "r"((uint32_t)(4)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_64) : "r"(lane), "n"(3), "r"((uint32_t)(2)));
        asm volatile("{ .reg .pred p; setp.lt.u32 p, %1, %2; selp.u32 %0, %3, %0, p; }" : "+r"(_mbarrier_init_count_10_64) : "r"(lane), "n"(2), "r"((uint32_t)(1)));
        if (lane < 27) {
            mbarrier_init(smem + 512 + lane * 8, _mbarrier_init_count_10_64);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }

    __syncwarp();

    // TMEM alloc (256 columns, 256 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 728);
    if (warp == 0) {
        int _tmem_hold = smem + 728;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(256) : "memory");
        __syncwarp();
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_tmem_state = taddr + 64;
    const int tmem_tmem_state_decay0 = taddr + 64;
    const int tmem_tmem_state_decay1 = taddr + 80;
    const int tmem_tmem_state_decay2 = taddr + 96;
    const int tmem_tmem_state_decay3 = taddr + 112;
    const int tmem_tmem_state_decay4 = taddr + 128;
    const int tmem_tmem_state_decay5 = taddr + 144;
    const int tmem_tmem_state_decay6 = taddr + 160;
    const int tmem_tmem_state_decay7 = taddr + 176;
    const int tmem_tmem_state_inp = taddr;
    const int tmem_tmem_state_inp_decay0 = taddr;
    const int tmem_tmem_state_inp_decay1 = taddr + 8;
    const int tmem_tmem_state_inp_decay2 = taddr + 16;
    const int tmem_tmem_state_inp_decay3 = taddr + 24;
    const int tmem_tmem_state_inp_decay4 = taddr + 32;
    const int tmem_tmem_state_inp_decay5 = taddr + 40;
    const int tmem_tmem_state_inp_decay6 = taddr + 48;
    const int tmem_tmem_state_inp_decay7 = taddr + 56;
    const int tmem_tmem_u_acc = taddr + 224;
    const int tmem_tmem_u2_inp = taddr + 224;
    const int tmem_tmem_u2_acc = taddr;
    const int tmem_tmem_out = taddr + 192;
    const int tmem_tmem_state_out = taddr + 64;
    asm volatile("griddepcontrol.wait;" ::: "memory");

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 8 && warp <= 11) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
    }

    // ---- Role: compute ----
    if (warp <= 3) {
        asm volatile("setmaxnreg.inc.sync.aligned.u32 168;");
        { // compute_main
            int task_idx = blockIdx.x;
            int warp_id_in_role = (warp - 0);
            int compute_local_warp = warp_id_in_role;
            int warp_in_wg = warp % 4;
            int state_row = warp_in_wg * 32 + lane;
            int split_compute_start = 0;
            int seq_idx = seq_order[task_idx / num_heads];
            int head_idx = task_idx % num_heads;
            long long bos = cu_seqlens[seq_idx];
            long long eos = cu_seqlens[seq_idx + 1];
            int num_chunks = ((int)(eos - bos) + 32 - 1) / 32;
            int seq_len = (int)(eos - bos);
            int num_chunks_0 = (seq_len + 32 - 1) / 32;
            int num_pages = 0;
            long long total_chunks = cu_chunk_offsets[num_sequences];
            long long fallback_head = total_chunks * (long long)num_heads + (long long)seq_idx * (long long)num_heads + (long long)head_idx;
            const int tmem_row_base = warp_in_wg * 32 << 16;
            long long state_base = (((long long)seq_idx * (long long)num_heads + (long long)head_idx) * 128 + (long long)state_row) * 128;
            {
                int state_slot = seq_idx;
                if (use_state_indices != 0) {
                    state_slot = reinterpret_cast<int*>(state_indices_addr)[seq_idx];
                }
                state_base = (long long)state_slot * state_slot_stride + ((long long)head_idx * 128 + (long long)state_row) * 128;
            }
            long long checkpoint_row_start = 0;
            if (checkpoint_every_n_tokens != 0) {
                checkpoint_row_start = reinterpret_cast<long long*>(checkpoint_cu_starts_addr)[seq_idx];
            }
            long long initial_state_base = state_base;
            int initial_state_enabled = (int)(use_initial_state != 0);
            #pragma unroll
            for (int state_col_block = 0; state_col_block < 4; state_col_block++) {
                float state_frag[32];
                state_frag[0] = 0.0f;
                state_frag[1] = 0.0f;
                state_frag[2] = 0.0f;
                state_frag[3] = 0.0f;
                state_frag[4] = 0.0f;
                state_frag[5] = 0.0f;
                state_frag[6] = 0.0f;
                state_frag[7] = 0.0f;
                state_frag[8] = 0.0f;
                state_frag[9] = 0.0f;
                state_frag[10] = 0.0f;
                state_frag[11] = 0.0f;
                state_frag[12] = 0.0f;
                state_frag[13] = 0.0f;
                state_frag[14] = 0.0f;
                state_frag[15] = 0.0f;
                state_frag[16] = 0.0f;
                state_frag[17] = 0.0f;
                state_frag[18] = 0.0f;
                state_frag[19] = 0.0f;
                state_frag[20] = 0.0f;
                state_frag[21] = 0.0f;
                state_frag[22] = 0.0f;
                state_frag[23] = 0.0f;
                state_frag[24] = 0.0f;
                state_frag[25] = 0.0f;
                state_frag[26] = 0.0f;
                state_frag[27] = 0.0f;
                state_frag[28] = 0.0f;
                state_frag[29] = 0.0f;
                state_frag[30] = 0.0f;
                state_frag[31] = 0.0f;
                if (initial_state_enabled != 0) {
                    {
                        float initial_values[8];
                        #pragma unroll
                        for (int initial_quarter = 0; initial_quarter < 4; initial_quarter++) {
                            {
                                unsigned _ldv8_0_0;
                                unsigned _ldv8_0_1;
                                unsigned _ldv8_0_2;
                                unsigned _ldv8_0_3;
                                unsigned _ldv8_0_4;
                                unsigned _ldv8_0_5;
                                unsigned _ldv8_0_6;
                                unsigned _ldv8_0_7;
                                asm volatile(
                                    "ld.global.v8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
                                    : "=r"(_ldv8_0_0), "=r"(_ldv8_0_1), "=r"(_ldv8_0_2), "=r"(_ldv8_0_3), "=r"(_ldv8_0_4), "=r"(_ldv8_0_5), "=r"(_ldv8_0_6), "=r"(_ldv8_0_7) : "l"((const void*)(initial_state_f32 + (initial_state_base + (long long)(state_col_block * 32) + (long long)(initial_quarter * 8)))) : "memory");
                                initial_values[0 + 0] = __uint_as_float(_ldv8_0_0);
                                initial_values[0 + 1] = __uint_as_float(_ldv8_0_1);
                                initial_values[0 + 2] = __uint_as_float(_ldv8_0_2);
                                initial_values[0 + 3] = __uint_as_float(_ldv8_0_3);
                                initial_values[0 + 4] = __uint_as_float(_ldv8_0_4);
                                initial_values[0 + 5] = __uint_as_float(_ldv8_0_5);
                                initial_values[0 + 6] = __uint_as_float(_ldv8_0_6);
                                initial_values[0 + 7] = __uint_as_float(_ldv8_0_7);
                            }
                            #pragma unroll
                            for (int initial_item = 0; initial_item < 8; initial_item++) {
                                {
                                    state_frag[initial_quarter * 8 + initial_item] = initial_values[initial_item];
                                }
                            }
                        }
                    }
                }
                tmem_st_x32_f32(taddr + 64 + (unsigned int)tmem_row_base + (unsigned int)(state_col_block * 32), state_frag);
            }
            asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
            if (checkpoint_every_n_tokens != 0) {
                long long checkpoint_base = ((checkpoint_row_start * (long long)num_heads + (long long)head_idx) * 128 + (long long)state_row) * 128;
                #pragma unroll
                for (int state_col_block_1 = 0; state_col_block_1 < 4; state_col_block_1++) {
                    float _tmem_load_0[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                        : "r"(taddr + 64 + (unsigned int)tmem_row_base + (unsigned int)(state_col_block_1 * 32)));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    {
                        unsigned _stv8_1_0 = __float_as_uint(_tmem_load_0[0 + 0]);
                        unsigned _stv8_1_1 = __float_as_uint(_tmem_load_0[0 + 1]);
                        unsigned _stv8_1_2 = __float_as_uint(_tmem_load_0[0 + 2]);
                        unsigned _stv8_1_3 = __float_as_uint(_tmem_load_0[0 + 3]);
                        unsigned _stv8_1_4 = __float_as_uint(_tmem_load_0[0 + 4]);
                        unsigned _stv8_1_5 = __float_as_uint(_tmem_load_0[0 + 5]);
                        unsigned _stv8_1_6 = __float_as_uint(_tmem_load_0[0 + 6]);
                        unsigned _stv8_1_7 = __float_as_uint(_tmem_load_0[0 + 7]);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(reinterpret_cast<float*>(state_checkpoints_addr) + (checkpoint_base + (long long)(state_col_block_1 * 32)) + (0))), "r"(_stv8_1_0), "r"(_stv8_1_1), "r"(_stv8_1_2), "r"(_stv8_1_3), "r"(_stv8_1_4), "r"(_stv8_1_5), "r"(_stv8_1_6), "r"(_stv8_1_7) : "memory");
                    }
                    {
                        unsigned _stv8_2_0 = __float_as_uint(_tmem_load_0[8 + 0]);
                        unsigned _stv8_2_1 = __float_as_uint(_tmem_load_0[8 + 1]);
                        unsigned _stv8_2_2 = __float_as_uint(_tmem_load_0[8 + 2]);
                        unsigned _stv8_2_3 = __float_as_uint(_tmem_load_0[8 + 3]);
                        unsigned _stv8_2_4 = __float_as_uint(_tmem_load_0[8 + 4]);
                        unsigned _stv8_2_5 = __float_as_uint(_tmem_load_0[8 + 5]);
                        unsigned _stv8_2_6 = __float_as_uint(_tmem_load_0[8 + 6]);
                        unsigned _stv8_2_7 = __float_as_uint(_tmem_load_0[8 + 7]);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(reinterpret_cast<float*>(state_checkpoints_addr) + (checkpoint_base + (long long)(state_col_block_1 * 32) + 8) + (0))), "r"(_stv8_2_0), "r"(_stv8_2_1), "r"(_stv8_2_2), "r"(_stv8_2_3), "r"(_stv8_2_4), "r"(_stv8_2_5), "r"(_stv8_2_6), "r"(_stv8_2_7) : "memory");
                    }
                    {
                        unsigned _stv8_3_0 = __float_as_uint(_tmem_load_0[16 + 0]);
                        unsigned _stv8_3_1 = __float_as_uint(_tmem_load_0[16 + 1]);
                        unsigned _stv8_3_2 = __float_as_uint(_tmem_load_0[16 + 2]);
                        unsigned _stv8_3_3 = __float_as_uint(_tmem_load_0[16 + 3]);
                        unsigned _stv8_3_4 = __float_as_uint(_tmem_load_0[16 + 4]);
                        unsigned _stv8_3_5 = __float_as_uint(_tmem_load_0[16 + 5]);
                        unsigned _stv8_3_6 = __float_as_uint(_tmem_load_0[16 + 6]);
                        unsigned _stv8_3_7 = __float_as_uint(_tmem_load_0[16 + 7]);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(reinterpret_cast<float*>(state_checkpoints_addr) + (checkpoint_base + (long long)(state_col_block_1 * 32) + 16) + (0))), "r"(_stv8_3_0), "r"(_stv8_3_1), "r"(_stv8_3_2), "r"(_stv8_3_3), "r"(_stv8_3_4), "r"(_stv8_3_5), "r"(_stv8_3_6), "r"(_stv8_3_7) : "memory");
                    }
                    {
                        unsigned _stv8_4_0 = __float_as_uint(_tmem_load_0[24 + 0]);
                        unsigned _stv8_4_1 = __float_as_uint(_tmem_load_0[24 + 1]);
                        unsigned _stv8_4_2 = __float_as_uint(_tmem_load_0[24 + 2]);
                        unsigned _stv8_4_3 = __float_as_uint(_tmem_load_0[24 + 3]);
                        unsigned _stv8_4_4 = __float_as_uint(_tmem_load_0[24 + 4]);
                        unsigned _stv8_4_5 = __float_as_uint(_tmem_load_0[24 + 5]);
                        unsigned _stv8_4_6 = __float_as_uint(_tmem_load_0[24 + 6]);
                        unsigned _stv8_4_7 = __float_as_uint(_tmem_load_0[24 + 7]);
                        asm volatile(
                            "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                            :: "l"((void*)(reinterpret_cast<float*>(state_checkpoints_addr) + (checkpoint_base + (long long)(state_col_block_1 * 32) + 24) + (0))), "r"(_stv8_4_0), "r"(_stv8_4_1), "r"(_stv8_4_2), "r"(_stv8_4_3), "r"(_stv8_4_4), "r"(_stv8_4_5), "r"(_stv8_4_6), "r"(_stv8_4_7) : "memory");
                    }
                }
            }
            unsigned int compute_stage = 0;
            unsigned int checkpoint_stage_compute = 0;
            unsigned int snapshot_stage_compute = 0;
            unsigned int _phase_checkpoint_free = 1;
            unsigned int _phase_qk_full = 0;
            unsigned int _phase_checkpoint_snapshot_done = 0;
            unsigned int _phase_v_full = 0;
            unsigned int _phase_old_out_ready = 0;
            unsigned int _phase_u2_acc_ready = 0;
            unsigned int _phase_final_ready = 0;
            #pragma unroll 1
            for (int chunk_idx = 0; chunk_idx < num_chunks_0; chunk_idx++) {
                int chunk_global_local = chunk_idx;
                int owned_chunk = chunk_global_local >= 0 && chunk_global_local < num_chunks;
                int checkpoint_token_entering = chunk_idx * 32;
                int checkpoint_entering = checkpoint_every_n_tokens != 0 && checkpoint_token_entering % checkpoint_every_n_tokens == 0;
                float state_panel0[32];
                float state_panel1[32];
                float state_panel2[32];
                float state_panel3[32];
                unsigned int state_packed0[16];
                unsigned int state_packed1[16];
                unsigned int state_packed2[16];
                unsigned int state_packed3[16];
                mbarrier_wait(qk_full_addr + (compute_stage) * 8, _phase_qk_full);
                #pragma unroll 1
                for (int state_col_block_2 = 0; state_col_block_2 < ((0) ? 4 : 3); state_col_block_2++) {
                    int state_addr = taddr + 64 + (unsigned int)tmem_row_base + (unsigned int)(state_col_block_2 * 32);
                    float _tmem_load_1[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_1[0]), "=f"(_tmem_load_1[1]), "=f"(_tmem_load_1[2]), "=f"(_tmem_load_1[3]), "=f"(_tmem_load_1[4]), "=f"(_tmem_load_1[5]), "=f"(_tmem_load_1[6]), "=f"(_tmem_load_1[7]), "=f"(_tmem_load_1[8]), "=f"(_tmem_load_1[9]), "=f"(_tmem_load_1[10]), "=f"(_tmem_load_1[11]), "=f"(_tmem_load_1[12]), "=f"(_tmem_load_1[13]), "=f"(_tmem_load_1[14]), "=f"(_tmem_load_1[15]), "=f"(_tmem_load_1[16]), "=f"(_tmem_load_1[17]), "=f"(_tmem_load_1[18]), "=f"(_tmem_load_1[19]), "=f"(_tmem_load_1[20]), "=f"(_tmem_load_1[21]), "=f"(_tmem_load_1[22]), "=f"(_tmem_load_1[23]), "=f"(_tmem_load_1[24]), "=f"(_tmem_load_1[25]), "=f"(_tmem_load_1[26]), "=f"(_tmem_load_1[27]), "=f"(_tmem_load_1[28]), "=f"(_tmem_load_1[29]), "=f"(_tmem_load_1[30]), "=f"(_tmem_load_1[31])
                        : "r"(state_addr));
                    uint32_t _tmem_load_1_bf16[16];
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_1[_lp*2 + 0], _tmem_load_1[_lp*2+1 + 0]));
                        _tmem_load_1_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(taddr + (unsigned int)tmem_row_base + (unsigned int)(state_col_block_2 * 16)), "r"(_tmem_load_1_bf16[0]), "r"(_tmem_load_1_bf16[1]), "r"(_tmem_load_1_bf16[2]), "r"(_tmem_load_1_bf16[3]), "r"(_tmem_load_1_bf16[4]), "r"(_tmem_load_1_bf16[5]), "r"(_tmem_load_1_bf16[6]), "r"(_tmem_load_1_bf16[7]), "r"(_tmem_load_1_bf16[8]), "r"(_tmem_load_1_bf16[9]), "r"(_tmem_load_1_bf16[10]), "r"(_tmem_load_1_bf16[11]), "r"(_tmem_load_1_bf16[12]), "r"(_tmem_load_1_bf16[13]), "r"(_tmem_load_1_bf16[14]), "r"(_tmem_load_1_bf16[15]));
                    if (!1 && checkpoint_entering != 0 && chunk_idx != 0) {
                        long long entering_checkpoint_idx = checkpoint_row_start + (long long)(chunk_idx * 32 / checkpoint_every_n_tokens);
                        long long entering_checkpoint_col = ((entering_checkpoint_idx * (long long)num_heads + (long long)head_idx) * 128 + (long long)state_row) * 128 + (long long)(state_col_block_2 * 32);
                        {
                            unsigned _stv8_5_0 = __float_as_uint(_tmem_load_1[0 + 0]);
                            unsigned _stv8_5_1 = __float_as_uint(_tmem_load_1[0 + 1]);
                            unsigned _stv8_5_2 = __float_as_uint(_tmem_load_1[0 + 2]);
                            unsigned _stv8_5_3 = __float_as_uint(_tmem_load_1[0 + 3]);
                            unsigned _stv8_5_4 = __float_as_uint(_tmem_load_1[0 + 4]);
                            unsigned _stv8_5_5 = __float_as_uint(_tmem_load_1[0 + 5]);
                            unsigned _stv8_5_6 = __float_as_uint(_tmem_load_1[0 + 6]);
                            unsigned _stv8_5_7 = __float_as_uint(_tmem_load_1[0 + 7]);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(reinterpret_cast<float*>(state_checkpoints_addr) + entering_checkpoint_col + (0))), "r"(_stv8_5_0), "r"(_stv8_5_1), "r"(_stv8_5_2), "r"(_stv8_5_3), "r"(_stv8_5_4), "r"(_stv8_5_5), "r"(_stv8_5_6), "r"(_stv8_5_7) : "memory");
                        }
                        {
                            unsigned _stv8_6_0 = __float_as_uint(_tmem_load_1[8 + 0]);
                            unsigned _stv8_6_1 = __float_as_uint(_tmem_load_1[8 + 1]);
                            unsigned _stv8_6_2 = __float_as_uint(_tmem_load_1[8 + 2]);
                            unsigned _stv8_6_3 = __float_as_uint(_tmem_load_1[8 + 3]);
                            unsigned _stv8_6_4 = __float_as_uint(_tmem_load_1[8 + 4]);
                            unsigned _stv8_6_5 = __float_as_uint(_tmem_load_1[8 + 5]);
                            unsigned _stv8_6_6 = __float_as_uint(_tmem_load_1[8 + 6]);
                            unsigned _stv8_6_7 = __float_as_uint(_tmem_load_1[8 + 7]);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(reinterpret_cast<float*>(state_checkpoints_addr) + (entering_checkpoint_col + 8) + (0))), "r"(_stv8_6_0), "r"(_stv8_6_1), "r"(_stv8_6_2), "r"(_stv8_6_3), "r"(_stv8_6_4), "r"(_stv8_6_5), "r"(_stv8_6_6), "r"(_stv8_6_7) : "memory");
                        }
                        {
                            unsigned _stv8_7_0 = __float_as_uint(_tmem_load_1[16 + 0]);
                            unsigned _stv8_7_1 = __float_as_uint(_tmem_load_1[16 + 1]);
                            unsigned _stv8_7_2 = __float_as_uint(_tmem_load_1[16 + 2]);
                            unsigned _stv8_7_3 = __float_as_uint(_tmem_load_1[16 + 3]);
                            unsigned _stv8_7_4 = __float_as_uint(_tmem_load_1[16 + 4]);
                            unsigned _stv8_7_5 = __float_as_uint(_tmem_load_1[16 + 5]);
                            unsigned _stv8_7_6 = __float_as_uint(_tmem_load_1[16 + 6]);
                            unsigned _stv8_7_7 = __float_as_uint(_tmem_load_1[16 + 7]);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(reinterpret_cast<float*>(state_checkpoints_addr) + (entering_checkpoint_col + 16) + (0))), "r"(_stv8_7_0), "r"(_stv8_7_1), "r"(_stv8_7_2), "r"(_stv8_7_3), "r"(_stv8_7_4), "r"(_stv8_7_5), "r"(_stv8_7_6), "r"(_stv8_7_7) : "memory");
                        }
                        {
                            unsigned _stv8_8_0 = __float_as_uint(_tmem_load_1[24 + 0]);
                            unsigned _stv8_8_1 = __float_as_uint(_tmem_load_1[24 + 1]);
                            unsigned _stv8_8_2 = __float_as_uint(_tmem_load_1[24 + 2]);
                            unsigned _stv8_8_3 = __float_as_uint(_tmem_load_1[24 + 3]);
                            unsigned _stv8_8_4 = __float_as_uint(_tmem_load_1[24 + 4]);
                            unsigned _stv8_8_5 = __float_as_uint(_tmem_load_1[24 + 5]);
                            unsigned _stv8_8_6 = __float_as_uint(_tmem_load_1[24 + 6]);
                            unsigned _stv8_8_7 = __float_as_uint(_tmem_load_1[24 + 7]);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(reinterpret_cast<float*>(state_checkpoints_addr) + (entering_checkpoint_col + 24) + (0))), "r"(_stv8_8_0), "r"(_stv8_8_1), "r"(_stv8_8_2), "r"(_stv8_8_3), "r"(_stv8_8_4), "r"(_stv8_8_5), "r"(_stv8_8_6), "r"(_stv8_8_7) : "memory");
                        }
                    }
                    {
                        float state_scale[16];
                        #pragma unroll
                        for (int state_half = 0; state_half < 2; state_half++) {
                            #pragma unroll
                            for (int state_col = 0; state_col < 16; state_col++) {
                                state_scale[state_col] = smem_gt_all[compute_stage * 10496 + (unsigned int)(state_col_block_2 * 32) + (unsigned int)(state_half * 16) + (unsigned int)state_col];
                            }
                            #pragma unroll
                            for (int _ls = 0; _ls < 8; _ls++)
                                mul_f32x2_inplace(&reinterpret_cast<float2*>((_tmem_load_1 + state_half * 16))[_ls], reinterpret_cast<const float2*>(state_scale)[_ls]);
                        }
                        if (STORE_BACKWARD_TAPE == 0 && checkpoint_entering != 0 && chunk_idx != 0) {
                            mbarrier_wait(checkpoint_snapshot_done_addr + (snapshot_stage_compute) * 8, _phase_checkpoint_snapshot_done);
                            snapshot_stage_compute += 1;
                            if (snapshot_stage_compute == 4) { snapshot_stage_compute = 0; _phase_checkpoint_snapshot_done ^= 1; }
                        }
                        tmem_st_x32_f32(state_addr, _tmem_load_1);
                    }
                }
                int state_tail_addr = taddr + 64 + (unsigned int)tmem_row_base + 96;
                float state_tail_frag[32];
                unsigned int state_tail_packed[16];
                {
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(state_tail_frag[0]), "=f"(state_tail_frag[1]), "=f"(state_tail_frag[2]), "=f"(state_tail_frag[3]), "=f"(state_tail_frag[4]), "=f"(state_tail_frag[5]), "=f"(state_tail_frag[6]), "=f"(state_tail_frag[7]), "=f"(state_tail_frag[8]), "=f"(state_tail_frag[9]), "=f"(state_tail_frag[10]), "=f"(state_tail_frag[11]), "=f"(state_tail_frag[12]), "=f"(state_tail_frag[13]), "=f"(state_tail_frag[14]), "=f"(state_tail_frag[15]), "=f"(state_tail_frag[16]), "=f"(state_tail_frag[17]), "=f"(state_tail_frag[18]), "=f"(state_tail_frag[19]), "=f"(state_tail_frag[20]), "=f"(state_tail_frag[21]), "=f"(state_tail_frag[22]), "=f"(state_tail_frag[23]), "=f"(state_tail_frag[24]), "=f"(state_tail_frag[25]), "=f"(state_tail_frag[26]), "=f"(state_tail_frag[27]), "=f"(state_tail_frag[28]), "=f"(state_tail_frag[29]), "=f"(state_tail_frag[30]), "=f"(state_tail_frag[31])
                        : "r"(state_tail_addr));
                    #pragma unroll
                    for (int _lp = 0; _lp < 16; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(state_tail_frag[_lp*2 + 0], state_tail_frag[_lp*2+1 + 0]));
                        state_tail_packed[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x16.b32"
                        " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                        :: "r"(taddr + (unsigned int)tmem_row_base + 48), "r"(state_tail_packed[0]), "r"(state_tail_packed[1]), "r"(state_tail_packed[2]), "r"(state_tail_packed[3]), "r"(state_tail_packed[4]), "r"(state_tail_packed[5]), "r"(state_tail_packed[6]), "r"(state_tail_packed[7]), "r"(state_tail_packed[8]), "r"(state_tail_packed[9]), "r"(state_tail_packed[10]), "r"(state_tail_packed[11]), "r"(state_tail_packed[12]), "r"(state_tail_packed[13]), "r"(state_tail_packed[14]), "r"(state_tail_packed[15]));
                    if (!1 && checkpoint_entering != 0 && chunk_idx != 0) {
                        long long entering_tail_idx = checkpoint_row_start + (long long)(chunk_idx * 32 / checkpoint_every_n_tokens);
                        long long entering_tail_col = ((entering_tail_idx * (long long)num_heads + (long long)head_idx) * 128 + (long long)state_row) * 128 + 96;
                        {
                            unsigned _stv8_9_0 = __float_as_uint(state_tail_frag[0 + 0]);
                            unsigned _stv8_9_1 = __float_as_uint(state_tail_frag[0 + 1]);
                            unsigned _stv8_9_2 = __float_as_uint(state_tail_frag[0 + 2]);
                            unsigned _stv8_9_3 = __float_as_uint(state_tail_frag[0 + 3]);
                            unsigned _stv8_9_4 = __float_as_uint(state_tail_frag[0 + 4]);
                            unsigned _stv8_9_5 = __float_as_uint(state_tail_frag[0 + 5]);
                            unsigned _stv8_9_6 = __float_as_uint(state_tail_frag[0 + 6]);
                            unsigned _stv8_9_7 = __float_as_uint(state_tail_frag[0 + 7]);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(reinterpret_cast<float*>(state_checkpoints_addr) + entering_tail_col + (0))), "r"(_stv8_9_0), "r"(_stv8_9_1), "r"(_stv8_9_2), "r"(_stv8_9_3), "r"(_stv8_9_4), "r"(_stv8_9_5), "r"(_stv8_9_6), "r"(_stv8_9_7) : "memory");
                        }
                        {
                            unsigned _stv8_10_0 = __float_as_uint(state_tail_frag[8 + 0]);
                            unsigned _stv8_10_1 = __float_as_uint(state_tail_frag[8 + 1]);
                            unsigned _stv8_10_2 = __float_as_uint(state_tail_frag[8 + 2]);
                            unsigned _stv8_10_3 = __float_as_uint(state_tail_frag[8 + 3]);
                            unsigned _stv8_10_4 = __float_as_uint(state_tail_frag[8 + 4]);
                            unsigned _stv8_10_5 = __float_as_uint(state_tail_frag[8 + 5]);
                            unsigned _stv8_10_6 = __float_as_uint(state_tail_frag[8 + 6]);
                            unsigned _stv8_10_7 = __float_as_uint(state_tail_frag[8 + 7]);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(reinterpret_cast<float*>(state_checkpoints_addr) + (entering_tail_col + 8) + (0))), "r"(_stv8_10_0), "r"(_stv8_10_1), "r"(_stv8_10_2), "r"(_stv8_10_3), "r"(_stv8_10_4), "r"(_stv8_10_5), "r"(_stv8_10_6), "r"(_stv8_10_7) : "memory");
                        }
                        {
                            unsigned _stv8_11_0 = __float_as_uint(state_tail_frag[16 + 0]);
                            unsigned _stv8_11_1 = __float_as_uint(state_tail_frag[16 + 1]);
                            unsigned _stv8_11_2 = __float_as_uint(state_tail_frag[16 + 2]);
                            unsigned _stv8_11_3 = __float_as_uint(state_tail_frag[16 + 3]);
                            unsigned _stv8_11_4 = __float_as_uint(state_tail_frag[16 + 4]);
                            unsigned _stv8_11_5 = __float_as_uint(state_tail_frag[16 + 5]);
                            unsigned _stv8_11_6 = __float_as_uint(state_tail_frag[16 + 6]);
                            unsigned _stv8_11_7 = __float_as_uint(state_tail_frag[16 + 7]);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(reinterpret_cast<float*>(state_checkpoints_addr) + (entering_tail_col + 16) + (0))), "r"(_stv8_11_0), "r"(_stv8_11_1), "r"(_stv8_11_2), "r"(_stv8_11_3), "r"(_stv8_11_4), "r"(_stv8_11_5), "r"(_stv8_11_6), "r"(_stv8_11_7) : "memory");
                        }
                        {
                            unsigned _stv8_12_0 = __float_as_uint(state_tail_frag[24 + 0]);
                            unsigned _stv8_12_1 = __float_as_uint(state_tail_frag[24 + 1]);
                            unsigned _stv8_12_2 = __float_as_uint(state_tail_frag[24 + 2]);
                            unsigned _stv8_12_3 = __float_as_uint(state_tail_frag[24 + 3]);
                            unsigned _stv8_12_4 = __float_as_uint(state_tail_frag[24 + 4]);
                            unsigned _stv8_12_5 = __float_as_uint(state_tail_frag[24 + 5]);
                            unsigned _stv8_12_6 = __float_as_uint(state_tail_frag[24 + 6]);
                            unsigned _stv8_12_7 = __float_as_uint(state_tail_frag[24 + 7]);
                            asm volatile(
                                "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                :: "l"((void*)(reinterpret_cast<float*>(state_checkpoints_addr) + (entering_tail_col + 24) + (0))), "r"(_stv8_12_0), "r"(_stv8_12_1), "r"(_stv8_12_2), "r"(_stv8_12_3), "r"(_stv8_12_4), "r"(_stv8_12_5), "r"(_stv8_12_6), "r"(_stv8_12_7) : "memory");
                        }
                    }
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                if (elect_sync()) {
                    mbarrier_arrive(state_inp_ready_addr + (compute_stage) * 8);
                }
                {
                    float state_tail_scale[16];
                    #pragma unroll
                    for (int state_half_1 = 0; state_half_1 < 2; state_half_1++) {
                        #pragma unroll
                        for (int state_col_1 = 0; state_col_1 < 16; state_col_1++) {
                            state_tail_scale[state_col_1] = smem_gt_all[compute_stage * 10496 + 96 + (unsigned int)(state_half_1 * 16) + (unsigned int)state_col_1];
                        }
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>((state_tail_frag + state_half_1 * 16))[_ls], reinterpret_cast<const float2*>(state_tail_scale)[_ls]);
                    }
                    if (STORE_BACKWARD_TAPE == 0 && checkpoint_entering != 0 && chunk_idx != 0) {
                        mbarrier_wait(checkpoint_snapshot_done_addr + (snapshot_stage_compute) * 8, _phase_checkpoint_snapshot_done);
                        snapshot_stage_compute += 1;
                        if (snapshot_stage_compute == 4) { snapshot_stage_compute = 0; _phase_checkpoint_snapshot_done ^= 1; }
                    }
                    tmem_st_x32_f32(state_tail_addr, state_tail_frag);
                }
                mbarrier_wait(v_full_addr + (compute_stage) * 8, _phase_v_full);
                unsigned int v_prefetch_bits[8];
                mbarrier_wait(old_out_ready_addr + (compute_stage) * 8, _phase_old_out_ready);
                float prediction[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(prediction[0]), "=f"(prediction[1]), "=f"(prediction[2]), "=f"(prediction[3]), "=f"(prediction[4]), "=f"(prediction[5]), "=f"(prediction[6]), "=f"(prediction[7]), "=f"(prediction[8]), "=f"(prediction[9]), "=f"(prediction[10]), "=f"(prediction[11]), "=f"(prediction[12]), "=f"(prediction[13]), "=f"(prediction[14]), "=f"(prediction[15]), "=f"(prediction[16]), "=f"(prediction[17]), "=f"(prediction[18]), "=f"(prediction[19]), "=f"(prediction[20]), "=f"(prediction[21]), "=f"(prediction[22]), "=f"(prediction[23]), "=f"(prediction[24]), "=f"(prediction[25]), "=f"(prediction[26]), "=f"(prediction[27]), "=f"(prediction[28]), "=f"(prediction[29]), "=f"(prediction[30]), "=f"(prediction[31])
                    : "r"(taddr + 224 + (unsigned int)tmem_row_base));
                long long chunk_global_e = cu_chunk_offsets[seq_idx] + (long long)chunk_global_local;
                long long tape_ex_base = ((chunk_global_e * (long long)num_heads + (long long)head_idx) * 128 + (long long)state_row) * 32;
                #pragma unroll
                for (int residual_half = 0; residual_half < 2; residual_half++) {
                    float residual_v[16];
                    float residual_beta[16];
                    #pragma unroll
                    for (int residual_col = 0; residual_col < 16; residual_col++) {
                        int token_col = residual_half * 16 + residual_col;
                        {
                            __nv_bfloat16 v_value = smem_v_all[compute_stage * 20992 + (unsigned int)(token_col * 128) + (unsigned int)state_row];
                            float _cvt_f32_40 = __bfloat162float(v_value);
                            residual_v[residual_col] = _cvt_f32_40;
                            residual_beta[residual_col] = smem_prep_beta_all[compute_stage * 10496 + (unsigned int)token_col];
                        }
                    }
                    {
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            sub_f32x2_inplace(&reinterpret_cast<float2*>(residual_v)[_ls], reinterpret_cast<const float2*>((prediction + residual_half * 16))[_ls]);
                    }
                    if (STORE_BACKWARD_TAPE != 0 && STORE_E_TAPE != 0 && owned_chunk != 0) {
                        {
                            __nv_bfloat162 _pk[4];
                            _pk[0] = __floats2bfloat162_rn(residual_v[0 + 0], residual_v[0 + 1]);
                            _pk[1] = __floats2bfloat162_rn(residual_v[0 + 2], residual_v[0 + 3]);
                            _pk[2] = __floats2bfloat162_rn(residual_v[0 + 4], residual_v[0 + 5]);
                            _pk[3] = __floats2bfloat162_rn(residual_v[0 + 6], residual_v[0 + 7]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(tape_e + (tape_ex_base + (long long)(residual_half * 16))))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        }
                        {
                            __nv_bfloat162 _pk[4];
                            _pk[0] = __floats2bfloat162_rn(residual_v[8 + 0], residual_v[8 + 1]);
                            _pk[1] = __floats2bfloat162_rn(residual_v[8 + 2], residual_v[8 + 3]);
                            _pk[2] = __floats2bfloat162_rn(residual_v[8 + 4], residual_v[8 + 5]);
                            _pk[3] = __floats2bfloat162_rn(residual_v[8 + 6], residual_v[8 + 7]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(tape_e + (tape_ex_base + (long long)(residual_half * 16) + 8)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        }
                    }
                    {
                        #pragma unroll
                        for (int _ls = 0; _ls < 8; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(residual_v)[_ls], reinterpret_cast<const float2*>(residual_beta)[_ls]);
                        if (STORE_BACKWARD_TAPE != 0 && owned_chunk != 0) {
                            {
                                __nv_bfloat162 _pk[4];
                                _pk[0] = __floats2bfloat162_rn(residual_v[0 + 0], residual_v[0 + 1]);
                                _pk[1] = __floats2bfloat162_rn(residual_v[0 + 2], residual_v[0 + 3]);
                                _pk[2] = __floats2bfloat162_rn(residual_v[0 + 4], residual_v[0 + 5]);
                                _pk[3] = __floats2bfloat162_rn(residual_v[0 + 6], residual_v[0 + 7]);
                                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(tape_x + (tape_ex_base + (long long)(residual_half * 16))))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                            }
                            {
                                __nv_bfloat162 _pk[4];
                                _pk[0] = __floats2bfloat162_rn(residual_v[8 + 0], residual_v[8 + 1]);
                                _pk[1] = __floats2bfloat162_rn(residual_v[8 + 2], residual_v[8 + 3]);
                                _pk[2] = __floats2bfloat162_rn(residual_v[8 + 4], residual_v[8 + 5]);
                                _pk[3] = __floats2bfloat162_rn(residual_v[8 + 6], residual_v[8 + 7]);
                                *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(tape_x + (tape_ex_base + (long long)(residual_half * 16) + 8)))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                            }
                        }
                        uint32_t residual_v_bf16[8];
                        #pragma unroll
                        for (int _lp = 0; _lp < 8; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(residual_v[_lp*2 + 0], residual_v[_lp*2+1 + 0]));
                            residual_v_bf16[_lp] = *(uint32_t*)&_bf2;
                        }
                        tmem_st_x8_u32(taddr + 224 + (unsigned int)tmem_row_base + (unsigned int)(residual_half * 8), (const uint32_t*)residual_v_bf16);
                    }
                }
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                mbarrier_arrive(v_free_addr + (compute_stage) * 8);
                if (elect_sync()) {
                    mbarrier_arrive(u_inp_ready_addr + (compute_stage) * 8);
                }
                mbarrier_wait(u2_acc_ready_addr + (compute_stage) * 8, _phase_u2_acc_ready);
                float _tmem_load_3[32];
                asm volatile(
                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                    : "=f"(_tmem_load_3[0]), "=f"(_tmem_load_3[1]), "=f"(_tmem_load_3[2]), "=f"(_tmem_load_3[3]), "=f"(_tmem_load_3[4]), "=f"(_tmem_load_3[5]), "=f"(_tmem_load_3[6]), "=f"(_tmem_load_3[7]), "=f"(_tmem_load_3[8]), "=f"(_tmem_load_3[9]), "=f"(_tmem_load_3[10]), "=f"(_tmem_load_3[11]), "=f"(_tmem_load_3[12]), "=f"(_tmem_load_3[13]), "=f"(_tmem_load_3[14]), "=f"(_tmem_load_3[15]), "=f"(_tmem_load_3[16]), "=f"(_tmem_load_3[17]), "=f"(_tmem_load_3[18]), "=f"(_tmem_load_3[19]), "=f"(_tmem_load_3[20]), "=f"(_tmem_load_3[21]), "=f"(_tmem_load_3[22]), "=f"(_tmem_load_3[23]), "=f"(_tmem_load_3[24]), "=f"(_tmem_load_3[25]), "=f"(_tmem_load_3[26]), "=f"(_tmem_load_3[27]), "=f"(_tmem_load_3[28]), "=f"(_tmem_load_3[29]), "=f"(_tmem_load_3[30]), "=f"(_tmem_load_3[31])
                    : "r"(taddr + (unsigned int)tmem_row_base));
                if (STORE_BACKWARD_TAPE != 0 && owned_chunk != 0) {
                    long long chunk_global_r = cu_chunk_offsets[seq_idx] + (long long)chunk_global_local;
                    long long tape_r_base = ((chunk_global_r * (long long)num_heads + (long long)head_idx) * 128 + (long long)state_row) * 32;
                    #pragma unroll
                    for (int tape_r_vec = 0; tape_r_vec < 4; tape_r_vec++) {
                        {
                            __nv_bfloat162 _pk[4];
                            _pk[0] = __floats2bfloat162_rn(_tmem_load_3[tape_r_vec * 8 + 0], _tmem_load_3[tape_r_vec * 8 + 1]);
                            _pk[1] = __floats2bfloat162_rn(_tmem_load_3[tape_r_vec * 8 + 2], _tmem_load_3[tape_r_vec * 8 + 3]);
                            _pk[2] = __floats2bfloat162_rn(_tmem_load_3[tape_r_vec * 8 + 4], _tmem_load_3[tape_r_vec * 8 + 5]);
                            _pk[3] = __floats2bfloat162_rn(_tmem_load_3[tape_r_vec * 8 + 6], _tmem_load_3[tape_r_vec * 8 + 7]);
                            *reinterpret_cast<uint4*>(&((__nv_bfloat16*)(tape_r + (tape_r_base + (long long)(tape_r_vec * 8))))[0]) = *reinterpret_cast<uint4*>(&_pk[0]);
                        }
                    }
                }
                unsigned int u2_packed[16];
                #pragma unroll
                for (int _lp = 0; _lp < 16; _lp++) {
                    __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_3[_lp*2 + 0], _tmem_load_3[_lp*2+1 + 0]));
                    u2_packed[_lp] = *(uint32_t*)&_bf2;
                }
                asm volatile(
                    "tcgen05.st.sync.aligned.32x32b.x16.b32"
                    " [%0], {%1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16};"
                    :: "r"(taddr + 224 + (unsigned int)tmem_row_base), "r"(u2_packed[0]), "r"(u2_packed[1]), "r"(u2_packed[2]), "r"(u2_packed[3]), "r"(u2_packed[4]), "r"(u2_packed[5]), "r"(u2_packed[6]), "r"(u2_packed[7]), "r"(u2_packed[8]), "r"(u2_packed[9]), "r"(u2_packed[10]), "r"(u2_packed[11]), "r"(u2_packed[12]), "r"(u2_packed[13]), "r"(u2_packed[14]), "r"(u2_packed[15]));
                asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                if (elect_sync()) {
                    mbarrier_arrive(u2_inp_ready_addr + (compute_stage) * 8);
                }
                mbarrier_wait(final_ready_addr + (compute_stage) * 8, _phase_final_ready);
                {
                    mbarrier_arrive(smem_free_addr + (compute_stage) * 8);
                }
                compute_stage += 1;
                if (compute_stage == 5) { compute_stage = 0; _phase_qk_full ^= 1; _phase_v_full ^= 1; _phase_old_out_ready ^= 1; _phase_u2_acc_ready ^= 1; _phase_final_ready ^= 1; }
            }
            if (store_final_state != 0) {
                #pragma unroll
                for (int state_col_block_3 = 0; state_col_block_3 < 4; state_col_block_3++) {
                    float _tmem_load_4[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_4[0]), "=f"(_tmem_load_4[1]), "=f"(_tmem_load_4[2]), "=f"(_tmem_load_4[3]), "=f"(_tmem_load_4[4]), "=f"(_tmem_load_4[5]), "=f"(_tmem_load_4[6]), "=f"(_tmem_load_4[7]), "=f"(_tmem_load_4[8]), "=f"(_tmem_load_4[9]), "=f"(_tmem_load_4[10]), "=f"(_tmem_load_4[11]), "=f"(_tmem_load_4[12]), "=f"(_tmem_load_4[13]), "=f"(_tmem_load_4[14]), "=f"(_tmem_load_4[15]), "=f"(_tmem_load_4[16]), "=f"(_tmem_load_4[17]), "=f"(_tmem_load_4[18]), "=f"(_tmem_load_4[19]), "=f"(_tmem_load_4[20]), "=f"(_tmem_load_4[21]), "=f"(_tmem_load_4[22]), "=f"(_tmem_load_4[23]), "=f"(_tmem_load_4[24]), "=f"(_tmem_load_4[25]), "=f"(_tmem_load_4[26]), "=f"(_tmem_load_4[27]), "=f"(_tmem_load_4[28]), "=f"(_tmem_load_4[29]), "=f"(_tmem_load_4[30]), "=f"(_tmem_load_4[31])
                        : "r"(taddr + 64 + (unsigned int)tmem_row_base + (unsigned int)(state_col_block_3 * 32)));
                    {
                        #pragma unroll
                        for (int state_vec = 0; state_vec < 4; state_vec++) {
                            {
                                unsigned _stv8_13_0 = __float_as_uint(_tmem_load_4[state_vec * 8 + 0]);
                                unsigned _stv8_13_1 = __float_as_uint(_tmem_load_4[state_vec * 8 + 1]);
                                unsigned _stv8_13_2 = __float_as_uint(_tmem_load_4[state_vec * 8 + 2]);
                                unsigned _stv8_13_3 = __float_as_uint(_tmem_load_4[state_vec * 8 + 3]);
                                unsigned _stv8_13_4 = __float_as_uint(_tmem_load_4[state_vec * 8 + 4]);
                                unsigned _stv8_13_5 = __float_as_uint(_tmem_load_4[state_vec * 8 + 5]);
                                unsigned _stv8_13_6 = __float_as_uint(_tmem_load_4[state_vec * 8 + 6]);
                                unsigned _stv8_13_7 = __float_as_uint(_tmem_load_4[state_vec * 8 + 7]);
                                asm volatile(
                                    "st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                                    :: "l"((void*)(final_state_f32 + (state_base + (long long)(state_col_block_3 * 32) + (long long)(state_vec * 8)) + (0))), "r"(_stv8_13_0), "r"(_stv8_13_1), "r"(_stv8_13_2), "r"(_stv8_13_3), "r"(_stv8_13_4), "r"(_stv8_13_5), "r"(_stv8_13_6), "r"(_stv8_13_7) : "memory");
                            }
                        }
                    }
                }
            }
            asm volatile("barrier.sync 9, 128;" ::: "memory");
            if (compute_local_warp == 0) {
                if (elect_sync()) {
                    mbarrier_arrive(tmem_dealloc_ready_addr);
                }
            }
        }
    // ---- Role: epilogue ----
    } else if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
        { // epilogue_main
            int task_idx_1 = blockIdx.x;
            int split_compute_start_1 = 0;
            unsigned int _phase_work_item_ready_0 = 0;
            int seq_idx_1 = seq_order[task_idx_1 / num_heads];
            int head_idx_1 = task_idx_1 % num_heads;
            long long bos_1 = cu_seqlens[seq_idx_1];
            long long eos_1 = cu_seqlens[seq_idx_1 + 1];
            int num_chunks_1 = ((int)(eos_1 - bos_1) + 32 - 1) / 32;
            int seq_len_1 = (int)(eos_1 - bos_1);
            int num_chunks_0_1 = (seq_len_1 + 32 - 1) / 32;
            int warp_id_in_role_1 = (warp - 4);
            int epilogue_local_warp = warp_id_in_role_1;
            int warp_in_wg_1 = warp % 4;
            const int tmem_row_base_1 = warp_in_wg_1 * 32 << 16;
            int state_row_1 = warp_in_wg_1 * 32 + lane;
            unsigned int epilogue_stage = 0;
            unsigned int output_stage = 0;
            unsigned int checkpoint_stage_epilogue = 0;
            unsigned int snapshot_stage_epilogue = 0;
            long long epilogue_checkpoint_row_start = 0;
            if (checkpoint_every_n_tokens != 0) {
                epilogue_checkpoint_row_start = reinterpret_cast<long long*>(checkpoint_cu_starts_addr)[seq_idx_1];
            }
            int epilogue_chunks = num_chunks_0_1;
            unsigned int _phase_checkpoint_ready = 0;
            unsigned int _phase_final_ready_1 = 0;
            #pragma unroll 1
            for (int chunk_idx_1 = 0; chunk_idx_1 < epilogue_chunks; chunk_idx_1++) {
                int checkpoint_token_epilogue = chunk_idx_1 * 32;
                int checkpoint_entering_epilogue = checkpoint_every_n_tokens != 0 && checkpoint_token_epilogue % checkpoint_every_n_tokens == 0;
                int chunk_is_full = ((seq_len_1 >= (chunk_idx_1 + 1) * 32) ? 1 : 0);
                if (chunk_is_full != 0) {
                    mbarrier_wait(final_ready_addr + (epilogue_stage) * 8, _phase_final_ready_1);
                    float _tmem_load_5[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_5[15]))
                        : "r"(taddr + 192 + (unsigned int)tmem_row_base_1));
                    float _tmem_load_6[16];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.16x256b.x4.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[1])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[2])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[3])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[4])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[5])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[6])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[7])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[8])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[9])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[10])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[11])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[12])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[13])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[14])), "=r"(*reinterpret_cast<uint32_t*>(&_tmem_load_6[15]))
                        : "r"(taddr + 192 + (unsigned int)tmem_row_base_1 + 1048576));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                    if (epilogue_local_warp == 0) {
                        if (elect_sync()) {
                            mbarrier_arrive(out_empty_addr);
                        }
                    }
                    if (epilogue_local_warp == 0) {
                        if (chunk_idx_1 >= 2) {
                            asm volatile("cp.async.bulk.wait_group.read 1;");
                        }
                    }
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                    int out_stage_addr = smem_out_addr + output_stage * 8192;
                    #pragma unroll
                    for (int dim_half = 0; dim_half < 2; dim_half++) {
                        unsigned int out_packed[8];
                        if (dim_half == 0) {
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_5[_lp*2 + 0], _tmem_load_5[_lp*2+1 + 0]));
                                out_packed[_lp] = *(uint32_t*)&_bf2;
                            }
                        } else {
                            #pragma unroll
                            for (int _lp = 0; _lp < 8; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(_tmem_load_6[_lp*2 + 0], _tmem_load_6[_lp*2+1 + 0]));
                                out_packed[_lp] = *(uint32_t*)&_bf2;
                            }
                        }
                        #pragma unroll
                        for (int token_group = 0; token_group < 2; token_group++) {
                            int mtx_idx = lane / 8;
                            int row_addr = lane & 7;
                            int dim_base = epilogue_local_warp * 32 + dim_half * 16 + (mtx_idx & 1) * 8;
                            int token_base = token_group * 16 + mtx_idx / 2 * 8;
                            int token_addr = token_base + row_addr;
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
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                    if (epilogue_local_warp == 0) {
                        if (elect_sync()) {
                            tma_store_4d(out_tma, 0, (int)(bos_1 + (long long)(chunk_idx_1 * 32)), head_idx_1, 0, smem_out_addr + output_stage * 8192);
                        }
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                    output_stage = output_stage ^ 1;
                    {
                        int snapshot_chunk = chunk_idx_1 + 1;
                        int snapshot_entering = checkpoint_every_n_tokens != 0 && snapshot_chunk * 32 % checkpoint_every_n_tokens == 0 && snapshot_chunk < num_chunks_0_1;
                        if (snapshot_entering != 0) {
                            long long snapshot_row_idx = epilogue_checkpoint_row_start + (long long)(snapshot_chunk * 32 / checkpoint_every_n_tokens);
                            int snapshot_outer = (int)(snapshot_row_idx * (long long)num_heads + (long long)head_idx_1);
                            #pragma unroll
                            for (int snapshot_panel = 0; snapshot_panel < 4; snapshot_panel++) {
                                float _tmem_load_7[32];
                                asm volatile(
                                    "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                                    " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                                    : "=f"(_tmem_load_7[0]), "=f"(_tmem_load_7[1]), "=f"(_tmem_load_7[2]), "=f"(_tmem_load_7[3]), "=f"(_tmem_load_7[4]), "=f"(_tmem_load_7[5]), "=f"(_tmem_load_7[6]), "=f"(_tmem_load_7[7]), "=f"(_tmem_load_7[8]), "=f"(_tmem_load_7[9]), "=f"(_tmem_load_7[10]), "=f"(_tmem_load_7[11]), "=f"(_tmem_load_7[12]), "=f"(_tmem_load_7[13]), "=f"(_tmem_load_7[14]), "=f"(_tmem_load_7[15]), "=f"(_tmem_load_7[16]), "=f"(_tmem_load_7[17]), "=f"(_tmem_load_7[18]), "=f"(_tmem_load_7[19]), "=f"(_tmem_load_7[20]), "=f"(_tmem_load_7[21]), "=f"(_tmem_load_7[22]), "=f"(_tmem_load_7[23]), "=f"(_tmem_load_7[24]), "=f"(_tmem_load_7[25]), "=f"(_tmem_load_7[26]), "=f"(_tmem_load_7[27]), "=f"(_tmem_load_7[28]), "=f"(_tmem_load_7[29]), "=f"(_tmem_load_7[30]), "=f"(_tmem_load_7[31])
                                    : "r"(taddr + 64 + (unsigned int)tmem_row_base_1 + (unsigned int)(snapshot_panel * 32)));
                                asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                                if (elect_sync()) {
                                    mbarrier_arrive(checkpoint_snapshot_done_addr + (snapshot_stage_epilogue) * 8);
                                }
                                snapshot_stage_epilogue += 1;
                                if (snapshot_stage_epilogue == 4) { snapshot_stage_epilogue = 0; }
                                if (epilogue_local_warp == 0) {
                                    asm volatile("cp.async.bulk.wait_group.read 0;");
                                }
                                asm volatile("barrier.sync 8, 128;" ::: "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_v18_addr + (unsigned int)(state_row_1 * 128 ^ (state_row_1 * 128 >> 7 & 7) << 4))), "r"(__as_u32(_tmem_load_7[0])), "r"(__as_u32(_tmem_load_7[1])), "r"(__as_u32(_tmem_load_7[2])), "r"(__as_u32(_tmem_load_7[3])) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_v18_addr + (unsigned int)(state_row_1 * 128 + 16 ^ (state_row_1 * 128 + 16 >> 7 & 7) << 4))), "r"(__as_u32(_tmem_load_7[4])), "r"(__as_u32(_tmem_load_7[5])), "r"(__as_u32(_tmem_load_7[6])), "r"(__as_u32(_tmem_load_7[7])) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_v18_addr + (unsigned int)(state_row_1 * 128 + 32 ^ (state_row_1 * 128 + 32 >> 7 & 7) << 4))), "r"(__as_u32(_tmem_load_7[8])), "r"(__as_u32(_tmem_load_7[9])), "r"(__as_u32(_tmem_load_7[10])), "r"(__as_u32(_tmem_load_7[11])) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_v18_addr + (unsigned int)(state_row_1 * 128 + 48 ^ (state_row_1 * 128 + 48 >> 7 & 7) << 4))), "r"(__as_u32(_tmem_load_7[12])), "r"(__as_u32(_tmem_load_7[13])), "r"(__as_u32(_tmem_load_7[14])), "r"(__as_u32(_tmem_load_7[15])) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_v18_addr + (unsigned int)(state_row_1 * 128 + 64 ^ (state_row_1 * 128 + 64 >> 7 & 7) << 4))), "r"(__as_u32(_tmem_load_7[16])), "r"(__as_u32(_tmem_load_7[17])), "r"(__as_u32(_tmem_load_7[18])), "r"(__as_u32(_tmem_load_7[19])) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_v18_addr + (unsigned int)(state_row_1 * 128 + 80 ^ (state_row_1 * 128 + 80 >> 7 & 7) << 4))), "r"(__as_u32(_tmem_load_7[20])), "r"(__as_u32(_tmem_load_7[21])), "r"(__as_u32(_tmem_load_7[22])), "r"(__as_u32(_tmem_load_7[23])) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_v18_addr + (unsigned int)(state_row_1 * 128 + 96 ^ (state_row_1 * 128 + 96 >> 7 & 7) << 4))), "r"(__as_u32(_tmem_load_7[24])), "r"(__as_u32(_tmem_load_7[25])), "r"(__as_u32(_tmem_load_7[26])), "r"(__as_u32(_tmem_load_7[27])) : "memory");
                                asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((smem_v18_addr + (unsigned int)(state_row_1 * 128 + 112 ^ (state_row_1 * 128 + 112 >> 7 & 7) << 4))), "r"(__as_u32(_tmem_load_7[28])), "r"(__as_u32(_tmem_load_7[29])), "r"(__as_u32(_tmem_load_7[30])), "r"(__as_u32(_tmem_load_7[31])) : "memory");
                                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                                asm volatile("barrier.sync 8, 128;" ::: "memory");
                                if (epilogue_local_warp == 0) {
                                    if (elect_sync()) {
                                        tma_store_3d(state_checkpoints_tma, snapshot_panel * 32, 0, snapshot_outer, smem_v18_addr);
                                    }
                                    asm volatile("cp.async.bulk.commit_group;");
                                }
                            }
                            if (epilogue_local_warp == 0) {
                                asm volatile("cp.async.bulk.wait_group.read 0;");
                            }
                            asm volatile("barrier.sync 8, 128;" ::: "memory");
                        }
                    }
                } else {
                    mbarrier_wait(final_ready_addr + (epilogue_stage) * 8, _phase_final_ready_1);
                    float _tmem_load_8[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_8[0]), "=f"(_tmem_load_8[1]), "=f"(_tmem_load_8[2]), "=f"(_tmem_load_8[3]), "=f"(_tmem_load_8[4]), "=f"(_tmem_load_8[5]), "=f"(_tmem_load_8[6]), "=f"(_tmem_load_8[7]), "=f"(_tmem_load_8[8]), "=f"(_tmem_load_8[9]), "=f"(_tmem_load_8[10]), "=f"(_tmem_load_8[11]), "=f"(_tmem_load_8[12]), "=f"(_tmem_load_8[13]), "=f"(_tmem_load_8[14]), "=f"(_tmem_load_8[15]), "=f"(_tmem_load_8[16]), "=f"(_tmem_load_8[17]), "=f"(_tmem_load_8[18]), "=f"(_tmem_load_8[19]), "=f"(_tmem_load_8[20]), "=f"(_tmem_load_8[21]), "=f"(_tmem_load_8[22]), "=f"(_tmem_load_8[23]), "=f"(_tmem_load_8[24]), "=f"(_tmem_load_8[25]), "=f"(_tmem_load_8[26]), "=f"(_tmem_load_8[27]), "=f"(_tmem_load_8[28]), "=f"(_tmem_load_8[29]), "=f"(_tmem_load_8[30]), "=f"(_tmem_load_8[31])
                        : "r"(taddr + 192 + (unsigned int)tmem_row_base_1));
                    asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                    if (epilogue_local_warp == 0) {
                        if (elect_sync()) {
                            mbarrier_arrive(out_empty_addr);
                        }
                    }
                    #pragma unroll
                    for (int token_col_1 = 0; token_col_1 < 32; token_col_1++) {
                        long long out_token = bos_1 + (long long)(chunk_idx_1 * 32 + token_col_1);
                        if (out_token < eos_1) {
                            long long out_idx = (out_token * (long long)num_heads + (long long)head_idx_1) * 128 + (long long)state_row_1;
                            out[out_idx] = _tmem_load_8[token_col_1];
                        }
                    }
                }
                epilogue_stage += 1;
                if (epilogue_stage == 5) { epilogue_stage = 0; _phase_final_ready_1 ^= 1; }
            }
            {
                if (epilogue_local_warp == 0) {
                    asm volatile("cp.async.bulk.wait_group 0;");
                }
                asm volatile("barrier.sync 8, 128;" ::: "memory");
            }
            if (epilogue_local_warp == 0) {
                if (elect_sync()) {
                    mbarrier_arrive(tmem_dealloc_ready_addr);
                }
            }
        }
    // ---- Role: beta_prefetch ----
    } else if (warp == 8) {
        { // beta_prefetch_main

        }
    // ---- Role: aux_mma ----
    } else if (warp >= 10 && warp <= 11) {
        { // aux_mma_main
            unsigned int _phase_work_item_ready_0_1 = 0;
        }
    // ---- Role: mma ----
    } else if (warp == 9) {
        { // mma_main
            int task_idx_2 = blockIdx.x;
            int split_compute_start_2 = 0;
            unsigned int _phase_work_item_ready_0_2 = 0;
            int seq_idx_2 = seq_order[task_idx_2 / num_heads];
            int head_idx_2 = task_idx_2 % num_heads;
            long long bos_2 = cu_seqlens[seq_idx_2];
            long long eos_2 = cu_seqlens[seq_idx_2 + 1];
            int num_chunks_2 = ((int)(eos_2 - bos_2) + 32 - 1) / 32;
            int seq_len_2 = (int)(eos_2 - bos_2);
            int num_chunks_0_2 = (seq_len_2 + 32 - 1) / 32;
            unsigned int mma_stage = 0;
            unsigned int _phase_qk_full_1 = 0;
            unsigned int _phase_out_empty_0 = 1;
            unsigned int _phase_state_inp_ready = 0;
            unsigned int _phase_u_inp_ready = 0;
            unsigned int _phase_u2_inp_ready = 0;
            unsigned int _phase_final_ready_2 = 0;
            #pragma unroll 1
            for (int _chunk_idx = 0; _chunk_idx < num_chunks_0_2; _chunk_idx++) {
                mbarrier_wait(qk_full_addr + (mma_stage) * 8, _phase_qk_full_1);
                {
                    mbarrier_wait(out_empty_addr, _phase_out_empty_0);
                    _phase_out_empty_0 ^= 1;
                }
                {
                    mbarrier_wait(state_inp_ready_addr + (mma_stage) * 8, _phase_state_inp_ready);
                    {
                        {
                            int _mma_b_lo_6 = make_warp_uniform((((smem_qd_addr) >> 4) & 0x3FFF) + (mma_stage) * 2624);
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
                    :: "r"(tmem_tmem_out), "r"(_mma_b_lo_6), "r"(tmem_tmem_state_inp), "r"(0));
                        }
                        int _mma_b_lo_7 = make_warp_uniform((((smem_kd_addr) >> 4) & 0x3FFF) + (mma_stage) * 2624);
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
                    :: "r"(tmem_tmem_u_acc), "r"(_mma_b_lo_7), "r"(tmem_tmem_state_inp), "r"(0));
                    }
                }
                {
                    elect_commit2(old_out_ready_addr + (mma_stage) * 8, raw_inputs_free_addr + (mma_stage) * 8);
                }
                mbarrier_wait(u_inp_ready_addr + (mma_stage) * 8, _phase_u_inp_ready);
                int _mma_b_lo_8 = make_warp_uniform((((smem_inv_addr) >> 4) & 0x3FFF) + (mma_stage) * 2624);
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
                    :: "r"(tmem_tmem_u2_acc), "r"(_mma_b_lo_8), "r"(tmem_tmem_u2_inp), "r"(0));
                elect_commit(u2_acc_ready_addr + (mma_stage) * 8);
                mbarrier_wait(u2_inp_ready_addr + (mma_stage) * 8, _phase_u2_inp_ready);
                int _mma_b_lo_9 = make_warp_uniform(((((smem_kr_trans_addr) >> 4) & 0x3FFF) | 0x1000000) + (mma_stage) * 2624);
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
                    "}\n"
                    :: "r"(tmem_tmem_state), "r"(_mma_b_lo_9), "r"(tmem_tmem_u2_inp), "r"(1));
                int _mma_b_lo_10 = make_warp_uniform(((((smem_final_mqk_slab_addr) >> 4) & 0x3FFF) | 0x1000000) + (mma_stage) * 2624);
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
                    :: "r"(tmem_tmem_out), "r"(_mma_b_lo_10), "r"(tmem_tmem_u2_inp), "r"(1));
                elect_commit(final_ready_addr + (mma_stage) * 8);
                mma_stage += 1;
                if (mma_stage == 5) { mma_stage = 0; _phase_qk_full_1 ^= 1; _phase_state_inp_ready ^= 1; _phase_u_inp_ready ^= 1; _phase_u2_inp_ready ^= 1; _phase_final_ready_2 ^= 1; }
            }
            unsigned int _phase_tmem_dealloc_ready_0 = 0;
            mbarrier_wait(tmem_dealloc_ready_addr, _phase_tmem_dealloc_ready_0);
            _phase_tmem_dealloc_ready_0 ^= 1;
            int _tmem_dealloc_addr = *((volatile int*)tmem_addr_storage);
            asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(_tmem_dealloc_addr), "r"(256));
        }
    // ---- Role: prep ----
    } else if (warp >= 12 && warp <= 31) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 48;");
        { // prep_main
            int task_idx_3 = blockIdx.x;
            int split_compute_start_3 = 0;
            unsigned int _phase_work_item_ready_0_3 = 0;
            int seq_idx_3 = seq_order[task_idx_3 / num_heads];
            int head_idx_3 = task_idx_3 % num_heads;
            long long bos_3 = cu_seqlens[seq_idx_3];
            long long eos_3 = cu_seqlens[seq_idx_3 + 1];
            int num_chunks_3 = ((int)(eos_3 - bos_3) + 32 - 1) / 32;
            int seq_len_3 = (int)(eos_3 - bos_3);
            int num_chunks_0_3 = (seq_len_3 + 32 - 1) / 32;
            int instance_id = (warp - 12) / 4;
            int prep_instance = instance_id;
            int warp_id_in_role_2 = (warp - 12);
            int prep_local_warp = warp_id_in_role_2 - prep_instance * 4;
            int prep_tid = prep_local_warp * 32 + lane;
            int num_prep_iters = (num_chunks_0_3 + 5 - 1 - prep_instance) / 5;
            unsigned int prep_stage = (unsigned int)prep_instance;
            int gate_rate_stage_f32 = prep_instance * 10496;
            int prep_global_tid = warp_id_in_role_2 * 32 + lane;
            if (prep_tid == 0) {
                float _expf_0 = __expf(A_log[head_idx_3]);
                smem_gate_rate_all[gate_rate_stage_f32] = _expf_0;
            }
            if (prep_global_tid < 128) {
                smem_gate_bias_all[prep_global_tid] = dt_bias[head_idx_3 * 128 + prep_global_tid];
            }
            asm volatile("barrier.sync 15, 640;" ::: "memory");
            unsigned int _phase_raw_inputs_free = 1;
            unsigned int _phase_gate_raw_full = 0;
            unsigned int _phase_smem_free = 1;
            unsigned int _phase_v_free = 1;
            unsigned int _phase_qk_raw_full = 0;
            unsigned int _phase_short_beta_ready = 0;
            unsigned int _phase_prep_diag_ready = 0;
            unsigned int _phase_prep_inv16_ready = 0;
            #pragma unroll 1
            for (int prep_iter = 0; prep_iter < num_prep_iters; prep_iter++) {
                int chunk_idx_2 = prep_iter * 5 + prep_instance;
                int chunk_global_local_1 = chunk_idx_2;
                int owned_chunk_1 = chunk_global_local_1 >= 0 && chunk_global_local_1 < num_chunks_3;
                int stage_f32 = prep_stage * 10496;
                int stage_bf16 = prep_stage * 20992;
                int chunk_is_full_1 = ((seq_len_3 >= (chunk_idx_2 + 1) * 32) ? 1 : 0);
                float early_beta_value = 0.0f;
                float early_gate0 = 0.0f;
                if (chunk_is_full_1 != 0 || prep_iter != 0) {
                    mbarrier_wait(raw_inputs_free_addr + (prep_stage) * 8, _phase_raw_inputs_free);
                }
                if (chunk_is_full_1 != 0) {
                    if (prep_local_warp == 0) {
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(gate_raw_full_addr + (prep_stage) * 8, 8704);
                            tma_3d_gmem2smem(smem_g_raw_addr + prep_stage * 41984, g_tma, 0, head_idx_3, (int)(bos_3 + (long long)(chunk_idx_2 * 32)), gate_raw_full_addr + (prep_stage) * 8);
                            {
                                tma_2d_gmem2smem(smem_beta_raw_addr + prep_stage * 41984, beta_tma, ((0) ? 0 : head_idx_3 / 8 * 8), (int)(((0) ? (bos_3 + (long long)(chunk_idx_2 * 32)) / 2 : bos_3 + (long long)(chunk_idx_2 * 32))), gate_raw_full_addr + (prep_stage) * 8);
                            }
                            mbarrier_arrive_expect_tx(qk_raw_full_addr + (prep_stage) * 8, 16384);
                            tma_4d_gmem2smem(smem_kd_addr + prep_stage * 41984, k_tma, 0, (int)(bos_3 + (long long)(chunk_idx_2 * 32)), head_idx_3, 0, qk_raw_full_addr + (prep_stage) * 8);
                        }
                    }
                    mbarrier_wait(gate_raw_full_addr + (prep_stage) * 8, _phase_gate_raw_full);
                    if (prep_local_warp == 2 && lane < 32) {
                        {
                            float beta_logit = 0.0f;
                            {
                                unsigned int beta_raw_pair[1];
                                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&beta_raw_pair[0])) : "r"(smem_beta_raw_addr + prep_stage * 41984 + (unsigned int)(lane * 16) + (unsigned int)(head_idx_3 % 8 / 2 * 4)));
                                float beta_raw_pair_f32[2];
                                #pragma unroll
                                for (int _pair = 0; _pair < 1; _pair++) {
                                    asm volatile(
                                        "{\n\t"
                                        "shl.b32 %0, %2, 16;\n\t"
                                        "and.b32 %1, %2, 0xffff0000;\n\t"
                                        "}\n"
                                        : "=f"((&beta_raw_pair_f32[_pair * 2])[0]), "=f"((&beta_raw_pair_f32[_pair * 2])[1])
                                        : "r"(beta_raw_pair[_pair]));
                                }
                                beta_logit = beta_raw_pair_f32[0];
                                if (head_idx_3 % 2 != 0) {
                                    beta_logit = beta_raw_pair_f32[1];
                                }
                            }
                            float _tanh_approx_1;
                            asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_1) : "f"(beta_logit * 0.5f));
                            early_beta_value = _tanh_approx_1 * 0.5f + 0.5f;
                        }
                    }
                    if (prep_tid < 128) {
                        float early_gate_rate = smem_gate_rate_all[stage_f32];
                        float early_gate_bias = smem_gate_bias_all[prep_tid];
                        __nv_bfloat16 early_gate_raw = smem_g_raw_all[stage_bf16 + prep_tid];
                        float _cvt_f32_1 = __bfloat162float(early_gate_raw);
                        float unbounded_biased = _cvt_f32_1 + early_gate_bias;
                        float _exp2_0 = approx_exp2(unbounded_biased * 1.4426950408889634f);
                        float unbounded_exponential = _exp2_0;
                        float _fma_0 = __fmaf_rn(unbounded_exponential, 0.2f, -0.25f);
                        float _fma_1 = __fmaf_rn(unbounded_exponential, _fma_0, 0.3333333333333333f);
                        float _fma_2 = __fmaf_rn(unbounded_exponential, _fma_1, -0.5f);
                        float _fma_3 = __fmaf_rn(unbounded_exponential * unbounded_exponential, _fma_2, unbounded_exponential);
                        float unbounded_series_log2 = _fma_3 * 1.4426950408889634f;
                        float _log2_0;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_0) : "f"(1.0f + unbounded_exponential));
                        float unbounded_direct_log2 = _log2_0;
                        float unbounded_softplus_log2 = ((unbounded_biased <= 20.0f) ? ((unbounded_biased < -4.0f) ? unbounded_series_log2 : unbounded_direct_log2) : unbounded_biased * 1.4426950408889634f);
                        early_gate0 = (-early_gate_rate) * unbounded_softplus_log2;
                    }
                }
                mbarrier_wait(smem_free_addr + (prep_stage) * 8, _phase_smem_free);
                mbarrier_wait(v_free_addr + (prep_stage) * 8, _phase_v_free);
                if (chunk_is_full_1 != 0) {
                    if (prep_local_warp == 0) {
                        if (elect_sync()) {
                            tma_4d_gmem2smem(smem_q_raw_prefetch_addr + prep_stage * 41984, q_tma, 0, (int)(bos_3 + (long long)(chunk_idx_2 * 32)), head_idx_3, 0, qk_raw_full_addr + (prep_stage) * 8);
                        }
                    }
                }
                if (chunk_is_full_1 == 0) {
                    #pragma unroll
                    for (int gate_load_pass = 0; gate_load_pass < 4; gate_load_pass++) {
                        int gate_load_item = gate_load_pass * 128 + prep_tid;
                        int gate_load_row = gate_load_item / 16;
                        int gate_load_segment = gate_load_item % 16;
                        long long gate_load_token = bos_3 + (long long)(chunk_idx_2 * 32 + gate_load_row);
                        long long gate_load_base = gate_load_token * g_token_stride + (long long)head_idx_3 * 128 + (long long)(gate_load_segment * 8);
                        long long qk_load_base = (gate_load_token * (long long)num_heads + (long long)head_idx_3) * 128 + (long long)(gate_load_segment * 8);
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"(smem_g_raw_addr + prep_stage * 41984 + (unsigned int)(gate_load_item * 16)), "l"(g + gate_load_base), "r"((gate_load_token < eos_3) ? 16 : 0));
                        int q_tail_addr = (smem_q_raw_prefetch_addr + prep_stage * 41984 + (unsigned int)(gate_load_segment * 8 / 64 * 4096 + gate_load_row * 128 + gate_load_segment * 8 % 64 * 2 ^ (gate_load_segment * 8 / 64 * 4096 + gate_load_row * 128 + gate_load_segment * 8 % 64 * 2 >> 7 & 7) << 4));
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"(q_tail_addr), "l"(q + qk_load_base), "r"((gate_load_token < eos_3) ? 16 : 0));
                        int k_tail_addr = (smem_kd_addr + prep_stage * 41984 + (unsigned int)(gate_load_segment * 8 / 64 * 4096 + gate_load_row * 128 + gate_load_segment * 8 % 64 * 2 ^ (gate_load_segment * 8 / 64 * 4096 + gate_load_row * 128 + gate_load_segment * 8 % 64 * 2 >> 7 & 7) << 4));
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"(k_tail_addr), "l"(k + qk_load_base), "r"((gate_load_token < eos_3) ? 16 : 0));
                    }
                }
                if (chunk_is_full_1 == 0) {
                    asm volatile("cp.async.commit_group;");
                    asm volatile("cp.async.wait_group 0;");
                    asm volatile("barrier.sync %0, 128;" :: "r"(10 + prep_instance) : "memory");
                }
                if (prep_local_warp == 2 && lane < 32) {
                    long long beta_token = bos_3 + (long long)(chunk_idx_2 * 32 + lane);
                    float beta_value = early_beta_value;
                    if (chunk_is_full_1 == 0) {
                        if (beta_token < eos_3) {
                            {
                                float beta_logit_1 = (float)beta[beta_token * (long long)num_heads + (long long)head_idx_3];
                                beta_logit_1 = (float)beta[beta_token * beta_token_stride + (long long)head_idx_3];
                                float _tanh_approx_2;
                                asm volatile("tanh.approx.f32 %0, %1;" : "=f"(_tanh_approx_2) : "f"(beta_logit_1 * 0.5f));
                                beta_value = _tanh_approx_2 * 0.5f + 0.5f;
                            }
                        }
                    }
                    {
                        __nv_bfloat16 _cvt_bf16_1 = __float2bfloat16(beta_value);
                        float _cvt_f32_2 = __bfloat162float(_cvt_bf16_1);
                        smem_prep_beta_all[stage_f32 + lane] = _cvt_f32_2;
                    }
                }
                float unbounded_total_log2 = 0.0f;
                if (prep_tid < 128) {
                    int gate_col = prep_tid;
                    float gate_rate = smem_gate_rate_all[stage_f32];
                    float gate_bias = smem_gate_bias_all[gate_col];
                    float prefix_log2 = 0.0f;
                    {
                        float2 _f2_7 = make_float2(-gate_rate, -gate_rate);
                        float2 scan_neg_rate_pair = _f2_7;
                        float2 _f2_8 = make_float2(gate_bias, gate_bias);
                        float2 scan_bias_pair = _f2_8;
                        long long first_gate_token = bos_3 + (long long)(chunk_idx_2 * 32);
                        __nv_bfloat16 first_gate_raw = smem_g_raw_all[stage_bf16 + gate_col];
                        float first_gate_log2 = early_gate0;
                        if (chunk_is_full_1 == 0) {
                            float _cvt_f32_7 = __bfloat162float(first_gate_raw);
                            float unbounded_biased_1 = _cvt_f32_7 + gate_bias;
                            float _exp2_1 = approx_exp2(unbounded_biased_1 * 1.4426950408889634f);
                            float unbounded_exponential_1 = _exp2_1;
                            float _fma_4 = __fmaf_rn(unbounded_exponential_1, 0.2f, -0.25f);
                            float _fma_5 = __fmaf_rn(unbounded_exponential_1, _fma_4, 0.3333333333333333f);
                            float _fma_6 = __fmaf_rn(unbounded_exponential_1, _fma_5, -0.5f);
                            float _fma_7 = __fmaf_rn(unbounded_exponential_1 * unbounded_exponential_1, _fma_6, unbounded_exponential_1);
                            float unbounded_series_log2_1 = _fma_7 * 1.4426950408889634f;
                            float _log2_1;
                            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_1) : "f"(1.0f + unbounded_exponential_1));
                            float unbounded_direct_log2_1 = _log2_1;
                            float unbounded_softplus_log2_1 = ((unbounded_biased_1 <= 20.0f) ? ((unbounded_biased_1 < -4.0f) ? unbounded_series_log2_1 : unbounded_direct_log2_1) : unbounded_biased_1 * 1.4426950408889634f);
                            float first_gate_value = (-gate_rate) * unbounded_softplus_log2_1;
                            first_gate_log2 = ((first_gate_token < eos_3) ? first_gate_value : 0.0f);
                        }
                        unbounded_total_log2 = unbounded_total_log2 + first_gate_log2;
                        float _max_0 = max_noftz(first_gate_log2, -15.0f);
                        prefix_log2 += _max_0;
                        smem_gate_all[stage_f32 + gate_col] = prefix_log2;
                        for (int gate_pair = 0; gate_pair < 15; gate_pair++) {
                            int pair_row0 = gate_pair * 2 + 1;
                            int pair_row1 = pair_row0 + 1;
                            long long pair_token0 = bos_3 + (long long)(chunk_idx_2 * 32 + pair_row0);
                            long long pair_token1 = pair_token0 + 1;
                            __nv_bfloat16 pair_raw0 = smem_g_raw_all[stage_bf16 + pair_row0 * 128 + gate_col];
                            __nv_bfloat16 pair_raw1 = smem_g_raw_all[stage_bf16 + pair_row1 * 128 + gate_col];
                            float _cvt_f32_8 = __bfloat162float(pair_raw0);
                            float _cvt_f32_9 = __bfloat162float(pair_raw1);
                            float2 _f2_9 = make_float2(_cvt_f32_8, _cvt_f32_9);
                            float2 unbounded_biased_pair = add_f32x2(_f2_9, scan_bias_pair);
                            float2 _f2_10 = make_float2(1.4426950408889634f, 1.4426950408889634f);
                            float2 _mul_f32x2_0;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_0) : "l"(*(const unsigned long long*)&unbounded_biased_pair), "l"(*(const unsigned long long*)&_f2_10));
                            float _exp2_2 = approx_exp2(_mul_f32x2_0.x);
                            float _exp2_3 = approx_exp2(_mul_f32x2_0.y);
                            float2 _f2_11 = make_float2(_exp2_2, _exp2_3);
                            float2 _f2_12 = make_float2(0.2f, 0.2f);
                            float2 _f2_13 = make_float2(-0.25f, -0.25f);
                            float2 _f2_14 = make_float2(0.3333333333333333f, 0.3333333333333333f);
                            float2 _f2_15 = make_float2(-0.5f, -0.5f);
                            float2 _mul_f32x2_1;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_1) : "l"(*(const unsigned long long*)&_f2_11), "l"(*(const unsigned long long*)&_f2_11));
                            float2 _f2_16 = make_float2(1.4426950408889634f, 1.4426950408889634f);
                            float2 _f32x2_mul_a_0 = fma_f32x2_rn_ftz(_mul_f32x2_1, fma_f32x2_rn_ftz(_f2_11, fma_f32x2_rn_ftz(_f2_11, fma_f32x2_rn_ftz(_f2_11, _f2_12, _f2_13), _f2_14), _f2_15), _f2_11);
                            float2 _mul_f32x2_2;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_2) : "l"(*(const unsigned long long*)&_f32x2_mul_a_0), "l"(*(const unsigned long long*)&_f2_16));
                            float2 _f2_17 = make_float2(1.0f, 1.0f);
                            float2 unbounded_one_plus_pair = add_f32x2(_f2_11, _f2_17);
                            float _log2_2;
                            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_2) : "f"(unbounded_one_plus_pair.x));
                            float _log2_3;
                            asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_3) : "f"(unbounded_one_plus_pair.y));
                            float2 _f2_18 = make_float2(_log2_2, _log2_3);
                            float2 _f2_19 = make_float2(((unbounded_biased_pair.x <= 20.0f) ? ((unbounded_biased_pair.x < -4.0f) ? _mul_f32x2_2.x : _f2_18.x) : _mul_f32x2_0.x), ((unbounded_biased_pair.y <= 20.0f) ? ((unbounded_biased_pair.y < -4.0f) ? _mul_f32x2_2.y : _f2_18.y) : _mul_f32x2_0.y));
                            float2 _mul_f32x2_3;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_3) : "l"(*(const unsigned long long*)&scan_neg_rate_pair), "l"(*(const unsigned long long*)&_f2_19));
                            float pair_log0 = ((pair_token0 < eos_3) ? _mul_f32x2_3.x : 0.0f);
                            float pair_log1 = ((pair_token1 < eos_3) ? _mul_f32x2_3.y : 0.0f);
                            unbounded_total_log2 = unbounded_total_log2 + pair_log0;
                            float _max_1 = max_noftz(pair_log0, -15.0f);
                            prefix_log2 += _max_1;
                            smem_gate_all[stage_f32 + pair_row0 * 128 + gate_col] = prefix_log2;
                            unbounded_total_log2 = unbounded_total_log2 + pair_log1;
                            float _max_2 = max_noftz(pair_log1, -15.0f);
                            prefix_log2 += _max_2;
                            smem_gate_all[stage_f32 + pair_row1 * 128 + gate_col] = prefix_log2;
                        }
                        long long last_gate_token = bos_3 + (long long)(chunk_idx_2 * 32 + 31);
                        __nv_bfloat16 last_gate_raw = smem_g_raw_all[stage_bf16 + 3968 + gate_col];
                        float _cvt_f32_10 = __bfloat162float(last_gate_raw);
                        float unbounded_biased_2 = _cvt_f32_10 + gate_bias;
                        float _exp2_4 = approx_exp2(unbounded_biased_2 * 1.4426950408889634f);
                        float unbounded_exponential_2 = _exp2_4;
                        float _fma_8 = __fmaf_rn(unbounded_exponential_2, 0.2f, -0.25f);
                        float _fma_9 = __fmaf_rn(unbounded_exponential_2, _fma_8, 0.3333333333333333f);
                        float _fma_10 = __fmaf_rn(unbounded_exponential_2, _fma_9, -0.5f);
                        float _fma_11 = __fmaf_rn(unbounded_exponential_2 * unbounded_exponential_2, _fma_10, unbounded_exponential_2);
                        float unbounded_series_log2_2 = _fma_11 * 1.4426950408889634f;
                        float _log2_4;
                        asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(_log2_4) : "f"(1.0f + unbounded_exponential_2));
                        float unbounded_direct_log2_2 = _log2_4;
                        float unbounded_softplus_log2_2 = ((unbounded_biased_2 <= 20.0f) ? ((unbounded_biased_2 < -4.0f) ? unbounded_series_log2_2 : unbounded_direct_log2_2) : unbounded_biased_2 * 1.4426950408889634f);
                        float last_gate_value = (-gate_rate) * unbounded_softplus_log2_2;
                        float last_gate_log2 = ((last_gate_token < eos_3) ? last_gate_value : 0.0f);
                        unbounded_total_log2 = unbounded_total_log2 + last_gate_log2;
                        float _max_3 = max_noftz(last_gate_log2, -15.0f);
                        prefix_log2 += _max_3;
                        smem_gate_all[stage_f32 + 3968 + gate_col] = prefix_log2;
                    }
                }
                asm volatile("barrier.sync %0, 128;" :: "r"(10 + prep_instance) : "memory");
                float page_total_log2 = 0.0f;
                if (chunk_is_full_1 != 0) {
                    mbarrier_wait(qk_raw_full_addr + (prep_stage) * 8, _phase_qk_raw_full);
                }
                if (prep_tid < 128) {
                    float total_log2 = smem_gt_prefix_all[stage_f32 + prep_tid];
                    float restore_factor_value = 1.0f;
                    float unbounded_cross_anchor = (float)(int)smem_gate_all[stage_f32 + 1920 + prep_tid];
                    float unbounded_tile0_half = (float)(int)((smem_gate_all[stage_f32 + prep_tid] - smem_gate_all[stage_f32 + 1920 + prep_tid]) * 0.5f);
                    float _min_0 = fminf((float)(int)((smem_gate_all[stage_f32 + 1920 + prep_tid] - smem_gate_all[stage_f32 + 2048 + prep_tid] + (smem_gate_all[stage_f32 + 1920 + prep_tid] - smem_gate_all[stage_f32 + 3968 + prep_tid])) * 0.5f), 126.0f);
                    float unbounded_tile1_half = _min_0;
                    float unbounded_tile0_anchor = unbounded_cross_anchor + unbounded_tile0_half;
                    float unbounded_tile1_anchor = unbounded_cross_anchor - unbounded_tile1_half;
                    int _max_4 = ((-(int)unbounded_cross_anchor) > (0) ? (-(int)unbounded_cross_anchor) : (0));
                    int unbounded_cross_magnitude = _max_4;
                    int unbounded_row_scale_bits = (127 - (int)unbounded_tile1_half << 7) + (unbounded_cross_magnitude & 127);
                    int unbounded_col_scale_bits = (127 - (int)unbounded_tile0_half << 7) + (unbounded_cross_magnitude >> 7 & 127);
                    int unbounded_anchor_word = unbounded_row_scale_bits + (unbounded_col_scale_bits << 16);
                    smem_v21[stage_f32 + prep_tid] = unbounded_anchor_word;
                }
                if (prep_tid == 0) {
                    smem_restore_factor_all[stage_f32 + 128] = 1.0f;
                }
                float unbounded_anchor_lo[8];
                float unbounded_anchor_hi[8];
                int anchor_segment = prep_tid % 16;
                #pragma unroll
                for (int anchor_elem = 0; anchor_elem < 8; anchor_elem++) {
                    int anchor_col = anchor_segment * 8 + anchor_elem;
                    float unbounded_cross_anchor_1 = (float)(int)smem_gate_all[stage_f32 + 1920 + anchor_col];
                    float unbounded_tile0_half_1 = (float)(int)((smem_gate_all[stage_f32 + anchor_col] - smem_gate_all[stage_f32 + 1920 + anchor_col]) * 0.5f);
                    float _min_1 = fminf((float)(int)((smem_gate_all[stage_f32 + 1920 + anchor_col] - smem_gate_all[stage_f32 + 2048 + anchor_col] + (smem_gate_all[stage_f32 + 1920 + anchor_col] - smem_gate_all[stage_f32 + 3968 + anchor_col])) * 0.5f), 126.0f);
                    float unbounded_tile1_half_1 = _min_1;
                    float unbounded_tile0_anchor_1 = unbounded_cross_anchor_1 + unbounded_tile0_half_1;
                    float unbounded_tile1_anchor_1 = unbounded_cross_anchor_1 - unbounded_tile1_half_1;
                    unbounded_anchor_lo[anchor_elem] = unbounded_tile0_anchor_1;
                    unbounded_anchor_hi[anchor_elem] = unbounded_tile1_anchor_1;
                }
                #pragma unroll 1
                for (int work_pass = 0; work_pass < 4; work_pass++) {
                    int work_item = work_pass * 128 + prep_tid;
                    int row = work_item / 16;
                    int segment = work_item % 16;
                    long long token = bos_3 + (long long)(chunk_idx_2 * 32 + row);
                    int token_valid = ((token < eos_3) ? 1 : 0);
                    long long gmem_base = (token * (long long)num_heads + (long long)head_idx_3) * 128 + (long long)(segment * 8);
                    float q_raw_vec[8];
                    float k_raw_vec[8];
                    unsigned int packed[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&packed[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed[(0) + 3]))
                        : "r"((smem_q_raw_prefetch_addr + prep_stage * 41984 + (unsigned int)(segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 ^ (segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 >> 7 & 7) << 4))));
                    float packed_f32[8];
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&packed_f32[_pair * 2])[0]), "=f"((&packed_f32[_pair * 2])[1])
                            : "r"(packed[_pair]));
                    }
                    #pragma unroll
                    for (int value_idx = 0; value_idx < 8; value_idx++) {
                        q_raw_vec[value_idx] = packed_f32[value_idx];
                    }
                    unsigned int packed_0[4];
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&packed_0[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0[(0) + 3]))
                        : "r"((smem_kd_addr + prep_stage * 41984 + (unsigned int)(segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 ^ (segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 >> 7 & 7) << 4))));
                    float packed_0_f32[8];
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&packed_0_f32[_pair * 2])[0]), "=f"((&packed_0_f32[_pair * 2])[1])
                            : "r"(packed_0[_pair]));
                    }
                    #pragma unroll
                    for (int value_idx_1 = 0; value_idx_1 < 8; value_idx_1++) {
                        k_raw_vec[value_idx_1] = packed_0_f32[value_idx_1];
                    }
                    float2 _f2_20 = make_float2(0.0f, 0.0f);
                    float2 q_sum_pair = _f2_20;
                    float2 _f2_21 = make_float2(0.0f, 0.0f);
                    float2 k_sum_pair = _f2_21;
                    for (int elem_pair = 0; elem_pair < 4; elem_pair++) {
                        float2 _f2_22 = make_float2(q_raw_vec[elem_pair * 2], q_raw_vec[elem_pair * 2 + 1]);
                        float2 q_raw_pair = _f2_22;
                        float2 _f2_23 = make_float2(k_raw_vec[elem_pair * 2], k_raw_vec[elem_pair * 2 + 1]);
                        float2 k_raw_pair = _f2_23;
                        q_sum_pair = fma_f32x2_rn_ftz(q_raw_pair, q_raw_pair, q_sum_pair);
                        k_sum_pair = fma_f32x2_rn_ftz(k_raw_pair, k_raw_pair, k_sum_pair);
                    }
                    float q_sum = q_sum_pair.x + q_sum_pair.y;
                    float k_sum = k_sum_pair.x + k_sum_pair.y;
                    float _shfl_xor_0 = __shfl_xor_sync(0xFFFFFFFF, q_sum, 8);
                    q_sum += _shfl_xor_0;
                    float _shfl_xor_1 = __shfl_xor_sync(0xFFFFFFFF, k_sum, 8);
                    k_sum += _shfl_xor_1;
                    float _shfl_xor_2 = __shfl_xor_sync(0xFFFFFFFF, q_sum, 4);
                    q_sum += _shfl_xor_2;
                    float _shfl_xor_3 = __shfl_xor_sync(0xFFFFFFFF, k_sum, 4);
                    k_sum += _shfl_xor_3;
                    float _shfl_xor_4 = __shfl_xor_sync(0xFFFFFFFF, q_sum, 2);
                    q_sum += _shfl_xor_4;
                    float _shfl_xor_5 = __shfl_xor_sync(0xFFFFFFFF, k_sum, 2);
                    k_sum += _shfl_xor_5;
                    float _shfl_xor_6 = __shfl_xor_sync(0xFFFFFFFF, q_sum, 1);
                    q_sum += _shfl_xor_6;
                    float _shfl_xor_7 = __shfl_xor_sync(0xFFFFFFFF, k_sum, 1);
                    k_sum += _shfl_xor_7;
                    float _rsqrt_0 = rsqrtf(q_sum + 1e-06f);
                    float q_inv = _rsqrt_0;
                    float _rsqrt_1 = rsqrtf(k_sum + 1e-06f);
                    float k_inv = _rsqrt_1;
                    const float2 _scale2_1 = {q_inv, q_inv};
                    #pragma unroll
                    for (int _ls = 0; _ls < 4; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(q_raw_vec)[_ls], _scale2_1);
                    const float2 _scale2_2 = {k_inv, k_inv};
                    #pragma unroll
                    for (int _ls = 0; _ls < 4; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(k_raw_vec)[_ls], _scale2_2);
                    __nv_bfloat16 _cvt_bf16_2 = __float2bfloat16(q_raw_vec[0]);
                    float _cvt_f32_11 = __bfloat162float(_cvt_bf16_2);
                    q_raw_vec[0] = _cvt_f32_11;
                    __nv_bfloat16 _cvt_bf16_3 = __float2bfloat16(k_raw_vec[0]);
                    float _cvt_f32_12 = __bfloat162float(_cvt_bf16_3);
                    k_raw_vec[0] = _cvt_f32_12;
                    __nv_bfloat16 _cvt_bf16_4 = __float2bfloat16(q_raw_vec[1]);
                    float _cvt_f32_13 = __bfloat162float(_cvt_bf16_4);
                    q_raw_vec[1] = _cvt_f32_13;
                    __nv_bfloat16 _cvt_bf16_5 = __float2bfloat16(k_raw_vec[1]);
                    float _cvt_f32_14 = __bfloat162float(_cvt_bf16_5);
                    k_raw_vec[1] = _cvt_f32_14;
                    __nv_bfloat16 _cvt_bf16_6 = __float2bfloat16(q_raw_vec[2]);
                    float _cvt_f32_15 = __bfloat162float(_cvt_bf16_6);
                    q_raw_vec[2] = _cvt_f32_15;
                    __nv_bfloat16 _cvt_bf16_7 = __float2bfloat16(k_raw_vec[2]);
                    float _cvt_f32_16 = __bfloat162float(_cvt_bf16_7);
                    k_raw_vec[2] = _cvt_f32_16;
                    __nv_bfloat16 _cvt_bf16_8 = __float2bfloat16(q_raw_vec[3]);
                    float _cvt_f32_17 = __bfloat162float(_cvt_bf16_8);
                    q_raw_vec[3] = _cvt_f32_17;
                    __nv_bfloat16 _cvt_bf16_9 = __float2bfloat16(k_raw_vec[3]);
                    float _cvt_f32_18 = __bfloat162float(_cvt_bf16_9);
                    k_raw_vec[3] = _cvt_f32_18;
                    __nv_bfloat16 _cvt_bf16_10 = __float2bfloat16(q_raw_vec[4]);
                    float _cvt_f32_19 = __bfloat162float(_cvt_bf16_10);
                    q_raw_vec[4] = _cvt_f32_19;
                    __nv_bfloat16 _cvt_bf16_11 = __float2bfloat16(k_raw_vec[4]);
                    float _cvt_f32_20 = __bfloat162float(_cvt_bf16_11);
                    k_raw_vec[4] = _cvt_f32_20;
                    __nv_bfloat16 _cvt_bf16_12 = __float2bfloat16(q_raw_vec[5]);
                    float _cvt_f32_21 = __bfloat162float(_cvt_bf16_12);
                    q_raw_vec[5] = _cvt_f32_21;
                    __nv_bfloat16 _cvt_bf16_13 = __float2bfloat16(k_raw_vec[5]);
                    float _cvt_f32_22 = __bfloat162float(_cvt_bf16_13);
                    k_raw_vec[5] = _cvt_f32_22;
                    __nv_bfloat16 _cvt_bf16_14 = __float2bfloat16(q_raw_vec[6]);
                    float _cvt_f32_23 = __bfloat162float(_cvt_bf16_14);
                    q_raw_vec[6] = _cvt_f32_23;
                    __nv_bfloat16 _cvt_bf16_15 = __float2bfloat16(k_raw_vec[6]);
                    float _cvt_f32_24 = __bfloat162float(_cvt_bf16_15);
                    k_raw_vec[6] = _cvt_f32_24;
                    __nv_bfloat16 _cvt_bf16_16 = __float2bfloat16(q_raw_vec[7]);
                    float _cvt_f32_25 = __bfloat162float(_cvt_bf16_16);
                    q_raw_vec[7] = _cvt_f32_25;
                    __nv_bfloat16 _cvt_bf16_17 = __float2bfloat16(k_raw_vec[7]);
                    float _cvt_f32_26 = __bfloat162float(_cvt_bf16_17);
                    k_raw_vec[7] = _cvt_f32_26;
                    float qd_vec[8];
                    float kd_vec[8];
                    float ki_vec[8];
                    for (int elem_in_segment = 0; elem_in_segment < 8; elem_in_segment++) {
                        int col = segment * 8 + elem_in_segment;
                        float prefix = smem_gate_all[stage_f32 + row * 128 + col];
                        float tile_anchor_log2 = unbounded_anchor_lo[elem_in_segment];
                        if (row >= 16) {
                            tile_anchor_log2 = unbounded_anchor_hi[elem_in_segment];
                        }
                        float _exp2_5 = approx_exp2(prefix - tile_anchor_log2);
                        float decay = _exp2_5;
                        qd_vec[elem_in_segment] = decay;
                        kd_vec[elem_in_segment] = decay;
                        ki_vec[elem_in_segment] = k_raw_vec[elem_in_segment] / decay;
                    }
                    #pragma unroll
                    for (int _ls = 0; _ls < 4; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(qd_vec)[_ls], reinterpret_cast<const float2*>(q_raw_vec)[_ls]);
                    {
                        const float2 _scale2_3 = {scale, scale};
                        #pragma unroll
                        for (int _ls = 0; _ls < 4; _ls++)
                            mul_f32x2_inplace(&reinterpret_cast<float2*>(qd_vec)[_ls], _scale2_3);
                    }
                    #pragma unroll
                    for (int _ls = 0; _ls < 4; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(kd_vec)[_ls], reinterpret_cast<const float2*>(k_raw_vec)[_ls]);
                    unsigned int packed_1[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(qd_vec[_lp*2 + 0], qd_vec[_lp*2+1 + 0]));
                        packed_1[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int word = 0; word < 4; word++) {
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_qd_addr + prep_stage * 41984 + (unsigned int)(segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 ^ (segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word * 4)), "r"((packed_1[word])));
                    }
                    unsigned int packed_2[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(kd_vec[_lp*2 + 0], kd_vec[_lp*2+1 + 0]));
                        packed_2[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int word_1 = 0; word_1 < 4; word_1++) {
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_kd_addr + prep_stage * 41984 + (unsigned int)(segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 ^ (segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_1 * 4)), "r"((packed_2[word_1])));
                    }
                    unsigned int packed_3[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(ki_vec[_lp*2 + 0], ki_vec[_lp*2+1 + 0]));
                        packed_3[_lp] = *(uint32_t*)&_bf2;
                    }
                    #pragma unroll
                    for (int word_2 = 0; word_2 < 4; word_2++) {
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_ki_addr + prep_stage * 41984 + (unsigned int)(segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 ^ (segment * 8 / 64 * 4096 + row * 128 + segment * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_2 * 4)), "r"((packed_3[word_2])));
                    }
                }
                asm volatile("barrier.sync %0, 128;" :: "r"(10 + prep_instance) : "memory");
                unsigned int a_frag[4];
                unsigned int b_frag[4];
                float acc[8];
                {
                    int pair_row_base = prep_local_warp / 2 * 16;
                    int pair_col_base = prep_local_warp % 2 * 16;
                    if (pair_row_base >= pair_col_base) {
                        int cross_scale_col = lane % 4 * 2;
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_kd_addr + prep_stage * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16) * 16))
                            : "memory");
                        if (pair_row_base - pair_col_base != 0) {
                            int cross_word_lo0 = smem_v21[stage_f32 + cross_scale_col];
                            int cross_word_lo1 = smem_v21[stage_f32 + cross_scale_col + 1];
                            unsigned int cross_a_lo = (unsigned int)((cross_word_lo0 & 65408) + ((cross_word_lo1 & 65408) << 16));
                            unsigned int cross_b_lo = (unsigned int)((cross_word_lo0 >> 16 & 65408) + (cross_word_lo1 & 4286578688));
                            int cross_word_hi0 = smem_v21[stage_f32 + cross_scale_col + 8];
                            int cross_word_hi1 = smem_v21[stage_f32 + cross_scale_col + 9];
                            unsigned int cross_a_hi = (unsigned int)((cross_word_hi0 & 65408) + ((cross_word_hi1 & 65408) << 16));
                            unsigned int cross_b_hi = (unsigned int)((cross_word_hi0 >> 16 & 65408) + (cross_word_hi1 & 4286578688));
                            uint32_t _bf16x2_mul_0;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_0) : "r"(a_frag[0]), "r"(cross_a_lo));
                            a_frag[0] = _bf16x2_mul_0;
                            uint32_t _bf16x2_mul_1;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_1) : "r"(a_frag[1]), "r"(cross_a_lo));
                            a_frag[1] = _bf16x2_mul_1;
                            uint32_t _bf16x2_mul_2;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_2) : "r"(a_frag[2]), "r"(cross_a_hi));
                            a_frag[2] = _bf16x2_mul_2;
                            uint32_t _bf16x2_mul_3;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_3) : "r"(a_frag[3]), "r"(cross_a_hi));
                            a_frag[3] = _bf16x2_mul_3;
                            uint32_t _bf16x2_mul_4;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_4) : "r"(b_frag[0]), "r"(cross_b_lo));
                            b_frag[0] = _bf16x2_mul_4;
                            uint32_t _bf16x2_mul_5;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_5) : "r"(b_frag[1]), "r"(cross_b_hi));
                            b_frag[1] = _bf16x2_mul_5;
                            uint32_t _bf16x2_mul_6;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_6) : "r"(b_frag[2]), "r"(cross_b_lo));
                            b_frag[2] = _bf16x2_mul_6;
                            uint32_t _bf16x2_mul_7;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_7) : "r"(b_frag[3]), "r"(cross_b_hi));
                            b_frag[3] = _bf16x2_mul_7;
                        }
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(acc[4]), "=f"(acc[(4) + 1]), "=f"(acc[(4) + 2]), "=f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_kd_addr + prep_stage * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)(((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256) * 16))
                            : "memory");
                        if (pair_row_base - pair_col_base != 0) {
                            int cross_word_lo0_1 = smem_v21[stage_f32 + 16 + cross_scale_col];
                            int cross_word_lo1_1 = smem_v21[stage_f32 + 16 + cross_scale_col + 1];
                            unsigned int cross_a_lo_1 = (unsigned int)((cross_word_lo0_1 & 65408) + ((cross_word_lo1_1 & 65408) << 16));
                            unsigned int cross_b_lo_1 = (unsigned int)((cross_word_lo0_1 >> 16 & 65408) + (cross_word_lo1_1 & 4286578688));
                            int cross_word_hi0_1 = smem_v21[stage_f32 + 16 + cross_scale_col + 8];
                            int cross_word_hi1_1 = smem_v21[stage_f32 + 16 + cross_scale_col + 9];
                            unsigned int cross_a_hi_1 = (unsigned int)((cross_word_hi0_1 & 65408) + ((cross_word_hi1_1 & 65408) << 16));
                            unsigned int cross_b_hi_1 = (unsigned int)((cross_word_hi0_1 >> 16 & 65408) + (cross_word_hi1_1 & 4286578688));
                            uint32_t _bf16x2_mul_8;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_8) : "r"(a_frag[0]), "r"(cross_a_lo_1));
                            a_frag[0] = _bf16x2_mul_8;
                            uint32_t _bf16x2_mul_9;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_9) : "r"(a_frag[1]), "r"(cross_a_lo_1));
                            a_frag[1] = _bf16x2_mul_9;
                            uint32_t _bf16x2_mul_10;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_10) : "r"(a_frag[2]), "r"(cross_a_hi_1));
                            a_frag[2] = _bf16x2_mul_10;
                            uint32_t _bf16x2_mul_11;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_11) : "r"(a_frag[3]), "r"(cross_a_hi_1));
                            a_frag[3] = _bf16x2_mul_11;
                            uint32_t _bf16x2_mul_12;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_12) : "r"(b_frag[0]), "r"(cross_b_lo_1));
                            b_frag[0] = _bf16x2_mul_12;
                            uint32_t _bf16x2_mul_13;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_13) : "r"(b_frag[1]), "r"(cross_b_hi_1));
                            b_frag[1] = _bf16x2_mul_13;
                            uint32_t _bf16x2_mul_14;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_14) : "r"(b_frag[2]), "r"(cross_b_lo_1));
                            b_frag[2] = _bf16x2_mul_14;
                            uint32_t _bf16x2_mul_15;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_15) : "r"(b_frag[3]), "r"(cross_b_hi_1));
                            b_frag[3] = _bf16x2_mul_15;
                        }
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_kd_addr + prep_stage * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)((((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256) * 16))
                            : "memory");
                        if (pair_row_base - pair_col_base != 0) {
                            int cross_word_lo0_2 = smem_v21[stage_f32 + 32 + cross_scale_col];
                            int cross_word_lo1_2 = smem_v21[stage_f32 + 32 + cross_scale_col + 1];
                            unsigned int cross_a_lo_2 = (unsigned int)((cross_word_lo0_2 & 65408) + ((cross_word_lo1_2 & 65408) << 16));
                            unsigned int cross_b_lo_2 = (unsigned int)((cross_word_lo0_2 >> 16 & 65408) + (cross_word_lo1_2 & 4286578688));
                            int cross_word_hi0_2 = smem_v21[stage_f32 + 32 + cross_scale_col + 8];
                            int cross_word_hi1_2 = smem_v21[stage_f32 + 32 + cross_scale_col + 9];
                            unsigned int cross_a_hi_2 = (unsigned int)((cross_word_hi0_2 & 65408) + ((cross_word_hi1_2 & 65408) << 16));
                            unsigned int cross_b_hi_2 = (unsigned int)((cross_word_hi0_2 >> 16 & 65408) + (cross_word_hi1_2 & 4286578688));
                            uint32_t _bf16x2_mul_16;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_16) : "r"(a_frag[0]), "r"(cross_a_lo_2));
                            a_frag[0] = _bf16x2_mul_16;
                            uint32_t _bf16x2_mul_17;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_17) : "r"(a_frag[1]), "r"(cross_a_lo_2));
                            a_frag[1] = _bf16x2_mul_17;
                            uint32_t _bf16x2_mul_18;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_18) : "r"(a_frag[2]), "r"(cross_a_hi_2));
                            a_frag[2] = _bf16x2_mul_18;
                            uint32_t _bf16x2_mul_19;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_19) : "r"(a_frag[3]), "r"(cross_a_hi_2));
                            a_frag[3] = _bf16x2_mul_19;
                            uint32_t _bf16x2_mul_20;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_20) : "r"(b_frag[0]), "r"(cross_b_lo_2));
                            b_frag[0] = _bf16x2_mul_20;
                            uint32_t _bf16x2_mul_21;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_21) : "r"(b_frag[1]), "r"(cross_b_hi_2));
                            b_frag[1] = _bf16x2_mul_21;
                            uint32_t _bf16x2_mul_22;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_22) : "r"(b_frag[2]), "r"(cross_b_lo_2));
                            b_frag[2] = _bf16x2_mul_22;
                            uint32_t _bf16x2_mul_23;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_23) : "r"(b_frag[3]), "r"(cross_b_hi_2));
                            b_frag[3] = _bf16x2_mul_23;
                        }
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_kd_addr + prep_stage * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)(((((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256) * 16))
                            : "memory");
                        if (pair_row_base - pair_col_base != 0) {
                            int cross_word_lo0_3 = smem_v21[stage_f32 + 48 + cross_scale_col];
                            int cross_word_lo1_3 = smem_v21[stage_f32 + 48 + cross_scale_col + 1];
                            unsigned int cross_a_lo_3 = (unsigned int)((cross_word_lo0_3 & 65408) + ((cross_word_lo1_3 & 65408) << 16));
                            unsigned int cross_b_lo_3 = (unsigned int)((cross_word_lo0_3 >> 16 & 65408) + (cross_word_lo1_3 & 4286578688));
                            int cross_word_hi0_3 = smem_v21[stage_f32 + 48 + cross_scale_col + 8];
                            int cross_word_hi1_3 = smem_v21[stage_f32 + 48 + cross_scale_col + 9];
                            unsigned int cross_a_hi_3 = (unsigned int)((cross_word_hi0_3 & 65408) + ((cross_word_hi1_3 & 65408) << 16));
                            unsigned int cross_b_hi_3 = (unsigned int)((cross_word_hi0_3 >> 16 & 65408) + (cross_word_hi1_3 & 4286578688));
                            uint32_t _bf16x2_mul_24;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_24) : "r"(a_frag[0]), "r"(cross_a_lo_3));
                            a_frag[0] = _bf16x2_mul_24;
                            uint32_t _bf16x2_mul_25;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_25) : "r"(a_frag[1]), "r"(cross_a_lo_3));
                            a_frag[1] = _bf16x2_mul_25;
                            uint32_t _bf16x2_mul_26;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_26) : "r"(a_frag[2]), "r"(cross_a_hi_3));
                            a_frag[2] = _bf16x2_mul_26;
                            uint32_t _bf16x2_mul_27;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_27) : "r"(a_frag[3]), "r"(cross_a_hi_3));
                            a_frag[3] = _bf16x2_mul_27;
                            uint32_t _bf16x2_mul_28;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_28) : "r"(b_frag[0]), "r"(cross_b_lo_3));
                            b_frag[0] = _bf16x2_mul_28;
                            uint32_t _bf16x2_mul_29;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_29) : "r"(b_frag[1]), "r"(cross_b_hi_3));
                            b_frag[1] = _bf16x2_mul_29;
                            uint32_t _bf16x2_mul_30;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_30) : "r"(b_frag[2]), "r"(cross_b_lo_3));
                            b_frag[2] = _bf16x2_mul_30;
                            uint32_t _bf16x2_mul_31;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_31) : "r"(b_frag[3]), "r"(cross_b_hi_3));
                            b_frag[3] = _bf16x2_mul_31;
                        }
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_kd_addr + prep_stage * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)((((((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256) * 16))
                            : "memory");
                        if (pair_row_base - pair_col_base != 0) {
                            int cross_word_lo0_4 = smem_v21[stage_f32 + 64 + cross_scale_col];
                            int cross_word_lo1_4 = smem_v21[stage_f32 + 64 + cross_scale_col + 1];
                            unsigned int cross_a_lo_4 = (unsigned int)((cross_word_lo0_4 & 65408) + ((cross_word_lo1_4 & 65408) << 16));
                            unsigned int cross_b_lo_4 = (unsigned int)((cross_word_lo0_4 >> 16 & 65408) + (cross_word_lo1_4 & 4286578688));
                            int cross_word_hi0_4 = smem_v21[stage_f32 + 64 + cross_scale_col + 8];
                            int cross_word_hi1_4 = smem_v21[stage_f32 + 64 + cross_scale_col + 9];
                            unsigned int cross_a_hi_4 = (unsigned int)((cross_word_hi0_4 & 65408) + ((cross_word_hi1_4 & 65408) << 16));
                            unsigned int cross_b_hi_4 = (unsigned int)((cross_word_hi0_4 >> 16 & 65408) + (cross_word_hi1_4 & 4286578688));
                            uint32_t _bf16x2_mul_32;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_32) : "r"(a_frag[0]), "r"(cross_a_lo_4));
                            a_frag[0] = _bf16x2_mul_32;
                            uint32_t _bf16x2_mul_33;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_33) : "r"(a_frag[1]), "r"(cross_a_lo_4));
                            a_frag[1] = _bf16x2_mul_33;
                            uint32_t _bf16x2_mul_34;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_34) : "r"(a_frag[2]), "r"(cross_a_hi_4));
                            a_frag[2] = _bf16x2_mul_34;
                            uint32_t _bf16x2_mul_35;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_35) : "r"(a_frag[3]), "r"(cross_a_hi_4));
                            a_frag[3] = _bf16x2_mul_35;
                            uint32_t _bf16x2_mul_36;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_36) : "r"(b_frag[0]), "r"(cross_b_lo_4));
                            b_frag[0] = _bf16x2_mul_36;
                            uint32_t _bf16x2_mul_37;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_37) : "r"(b_frag[1]), "r"(cross_b_hi_4));
                            b_frag[1] = _bf16x2_mul_37;
                            uint32_t _bf16x2_mul_38;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_38) : "r"(b_frag[2]), "r"(cross_b_lo_4));
                            b_frag[2] = _bf16x2_mul_38;
                            uint32_t _bf16x2_mul_39;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_39) : "r"(b_frag[3]), "r"(cross_b_hi_4));
                            b_frag[3] = _bf16x2_mul_39;
                        }
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_kd_addr + prep_stage * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)(((((((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256) * 16))
                            : "memory");
                        if (pair_row_base - pair_col_base != 0) {
                            int cross_word_lo0_5 = smem_v21[stage_f32 + 80 + cross_scale_col];
                            int cross_word_lo1_5 = smem_v21[stage_f32 + 80 + cross_scale_col + 1];
                            unsigned int cross_a_lo_5 = (unsigned int)((cross_word_lo0_5 & 65408) + ((cross_word_lo1_5 & 65408) << 16));
                            unsigned int cross_b_lo_5 = (unsigned int)((cross_word_lo0_5 >> 16 & 65408) + (cross_word_lo1_5 & 4286578688));
                            int cross_word_hi0_5 = smem_v21[stage_f32 + 80 + cross_scale_col + 8];
                            int cross_word_hi1_5 = smem_v21[stage_f32 + 80 + cross_scale_col + 9];
                            unsigned int cross_a_hi_5 = (unsigned int)((cross_word_hi0_5 & 65408) + ((cross_word_hi1_5 & 65408) << 16));
                            unsigned int cross_b_hi_5 = (unsigned int)((cross_word_hi0_5 >> 16 & 65408) + (cross_word_hi1_5 & 4286578688));
                            uint32_t _bf16x2_mul_40;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_40) : "r"(a_frag[0]), "r"(cross_a_lo_5));
                            a_frag[0] = _bf16x2_mul_40;
                            uint32_t _bf16x2_mul_41;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_41) : "r"(a_frag[1]), "r"(cross_a_lo_5));
                            a_frag[1] = _bf16x2_mul_41;
                            uint32_t _bf16x2_mul_42;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_42) : "r"(a_frag[2]), "r"(cross_a_hi_5));
                            a_frag[2] = _bf16x2_mul_42;
                            uint32_t _bf16x2_mul_43;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_43) : "r"(a_frag[3]), "r"(cross_a_hi_5));
                            a_frag[3] = _bf16x2_mul_43;
                            uint32_t _bf16x2_mul_44;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_44) : "r"(b_frag[0]), "r"(cross_b_lo_5));
                            b_frag[0] = _bf16x2_mul_44;
                            uint32_t _bf16x2_mul_45;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_45) : "r"(b_frag[1]), "r"(cross_b_hi_5));
                            b_frag[1] = _bf16x2_mul_45;
                            uint32_t _bf16x2_mul_46;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_46) : "r"(b_frag[2]), "r"(cross_b_lo_5));
                            b_frag[2] = _bf16x2_mul_46;
                            uint32_t _bf16x2_mul_47;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_47) : "r"(b_frag[3]), "r"(cross_b_hi_5));
                            b_frag[3] = _bf16x2_mul_47;
                        }
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_kd_addr + prep_stage * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2 ^ 6) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)((((((((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256 + 256 ^ 6) - 256) * 16))
                            : "memory");
                        if (pair_row_base - pair_col_base != 0) {
                            int cross_word_lo0_6 = smem_v21[stage_f32 + 96 + cross_scale_col];
                            int cross_word_lo1_6 = smem_v21[stage_f32 + 96 + cross_scale_col + 1];
                            unsigned int cross_a_lo_6 = (unsigned int)((cross_word_lo0_6 & 65408) + ((cross_word_lo1_6 & 65408) << 16));
                            unsigned int cross_b_lo_6 = (unsigned int)((cross_word_lo0_6 >> 16 & 65408) + (cross_word_lo1_6 & 4286578688));
                            int cross_word_hi0_6 = smem_v21[stage_f32 + 96 + cross_scale_col + 8];
                            int cross_word_hi1_6 = smem_v21[stage_f32 + 96 + cross_scale_col + 9];
                            unsigned int cross_a_hi_6 = (unsigned int)((cross_word_hi0_6 & 65408) + ((cross_word_hi1_6 & 65408) << 16));
                            unsigned int cross_b_hi_6 = (unsigned int)((cross_word_hi0_6 >> 16 & 65408) + (cross_word_hi1_6 & 4286578688));
                            uint32_t _bf16x2_mul_48;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_48) : "r"(a_frag[0]), "r"(cross_a_lo_6));
                            a_frag[0] = _bf16x2_mul_48;
                            uint32_t _bf16x2_mul_49;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_49) : "r"(a_frag[1]), "r"(cross_a_lo_6));
                            a_frag[1] = _bf16x2_mul_49;
                            uint32_t _bf16x2_mul_50;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_50) : "r"(a_frag[2]), "r"(cross_a_hi_6));
                            a_frag[2] = _bf16x2_mul_50;
                            uint32_t _bf16x2_mul_51;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_51) : "r"(a_frag[3]), "r"(cross_a_hi_6));
                            a_frag[3] = _bf16x2_mul_51;
                            uint32_t _bf16x2_mul_52;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_52) : "r"(b_frag[0]), "r"(cross_b_lo_6));
                            b_frag[0] = _bf16x2_mul_52;
                            uint32_t _bf16x2_mul_53;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_53) : "r"(b_frag[1]), "r"(cross_b_hi_6));
                            b_frag[1] = _bf16x2_mul_53;
                            uint32_t _bf16x2_mul_54;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_54) : "r"(b_frag[2]), "r"(cross_b_lo_6));
                            b_frag[2] = _bf16x2_mul_54;
                            uint32_t _bf16x2_mul_55;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_55) : "r"(b_frag[3]), "r"(cross_b_hi_6));
                            b_frag[3] = _bf16x2_mul_55;
                        }
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_kd_addr + prep_stage * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (pair_row_base + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (pair_row_base + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2 ^ 6 ^ 2) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)(((((((((lane % 16 / 8 / 8 * 256 + (pair_col_base + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (pair_col_base + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256) * 16))
                            : "memory");
                        if (pair_row_base - pair_col_base != 0) {
                            int cross_word_lo0_7 = smem_v21[stage_f32 + 112 + cross_scale_col];
                            int cross_word_lo1_7 = smem_v21[stage_f32 + 112 + cross_scale_col + 1];
                            unsigned int cross_a_lo_7 = (unsigned int)((cross_word_lo0_7 & 65408) + ((cross_word_lo1_7 & 65408) << 16));
                            unsigned int cross_b_lo_7 = (unsigned int)((cross_word_lo0_7 >> 16 & 65408) + (cross_word_lo1_7 & 4286578688));
                            int cross_word_hi0_7 = smem_v21[stage_f32 + 112 + cross_scale_col + 8];
                            int cross_word_hi1_7 = smem_v21[stage_f32 + 112 + cross_scale_col + 9];
                            unsigned int cross_a_hi_7 = (unsigned int)((cross_word_hi0_7 & 65408) + ((cross_word_hi1_7 & 65408) << 16));
                            unsigned int cross_b_hi_7 = (unsigned int)((cross_word_hi0_7 >> 16 & 65408) + (cross_word_hi1_7 & 4286578688));
                            uint32_t _bf16x2_mul_56;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_56) : "r"(a_frag[0]), "r"(cross_a_lo_7));
                            a_frag[0] = _bf16x2_mul_56;
                            uint32_t _bf16x2_mul_57;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_57) : "r"(a_frag[1]), "r"(cross_a_lo_7));
                            a_frag[1] = _bf16x2_mul_57;
                            uint32_t _bf16x2_mul_58;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_58) : "r"(a_frag[2]), "r"(cross_a_hi_7));
                            a_frag[2] = _bf16x2_mul_58;
                            uint32_t _bf16x2_mul_59;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_59) : "r"(a_frag[3]), "r"(cross_a_hi_7));
                            a_frag[3] = _bf16x2_mul_59;
                            uint32_t _bf16x2_mul_60;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_60) : "r"(b_frag[0]), "r"(cross_b_lo_7));
                            b_frag[0] = _bf16x2_mul_60;
                            uint32_t _bf16x2_mul_61;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_61) : "r"(b_frag[1]), "r"(cross_b_hi_7));
                            b_frag[1] = _bf16x2_mul_61;
                            uint32_t _bf16x2_mul_62;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_62) : "r"(b_frag[2]), "r"(cross_b_lo_7));
                            b_frag[2] = _bf16x2_mul_62;
                            uint32_t _bf16x2_mul_63;
                            asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_63) : "r"(b_frag[3]), "r"(cross_b_hi_7));
                            b_frag[3] = _bf16x2_mul_63;
                        }
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        int row0 = pair_row_base + lane / 4;
                        int row1 = row0 + 8;
                        int col0 = pair_col_base + lane % 4 * 2;
                        float beta0 = 0.0f;
                        float beta1 = 0.0f;
                        {
                            beta0 = smem_prep_beta_all[stage_f32 + row0];
                            beta1 = smem_prep_beta_all[stage_f32 + row1];
                        }
                        float seed[8];
                        seed[0] = 0.0f;
                        seed[1] = 0.0f;
                        seed[2] = 0.0f;
                        seed[3] = 0.0f;
                        seed[4] = 0.0f;
                        seed[5] = 0.0f;
                        seed[6] = 0.0f;
                        seed[7] = 0.0f;
                        if (row0 > col0) {
                            seed[0] = acc[0] * beta0;
                        }
                        if (row0 > col0 + 1) {
                            seed[1] = acc[1] * beta0;
                        }
                        if (row1 > col0) {
                            seed[2] = acc[2] * beta1;
                        }
                        if (row1 > col0 + 1) {
                            seed[3] = acc[3] * beta1;
                        }
                        if (row0 > col0 + 8) {
                            seed[4] = acc[4] * beta0;
                        }
                        if (row0 > col0 + 9) {
                            seed[5] = acc[5] * beta0;
                        }
                        if (row1 > col0 + 8) {
                            seed[6] = acc[6] * beta1;
                        }
                        if (row1 > col0 + 9) {
                            seed[7] = acc[7] * beta1;
                        }
                        unsigned int seed_packed[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(seed[_lp*2 + 0], seed[_lp*2+1 + 0]));
                            seed_packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        int seed_lane_row = lane % 16;
                        int seed_lane_col = lane / 16 * 8;
                        int byte_off = (int)prep_stage * 41984 + (pair_row_base + seed_lane_row) * 128 + (pair_col_base + seed_lane_col) * 2;
                        int swizzled_off = byte_off ^ (byte_off >> 7 & 7) << 4;
                        int seed_addr = smem_inv_work_addr + (unsigned int)swizzled_off;
                        uint32_t _stmatrix_addr_4 = static_cast<uint32_t>((unsigned long long)seed_addr);
                        asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                            :: "r"(_stmatrix_addr_4), "r"(*reinterpret_cast<const uint32_t*>(&seed_packed[0])), "r"(*reinterpret_cast<const uint32_t*>(&seed_packed[1])), "r"(*reinterpret_cast<const uint32_t*>(&seed_packed[2])), "r"(*reinterpret_cast<const uint32_t*>(&seed_packed[3]))
                            : "memory");
                    }
                    if (prep_local_warp == 1) {
                        acc[0] = 0.0f;
                        acc[1] = 0.0f;
                        acc[2] = 0.0f;
                        acc[3] = 0.0f;
                        acc[4] = 0.0f;
                        acc[5] = 0.0f;
                        acc[6] = 0.0f;
                        acc[7] = 0.0f;
                        int row0_1 = lane / 4;
                        int row1_1 = row0_1 + 8;
                        int col0_1 = 16 + lane % 4 * 2;
                        float mqk[8];
                        mqk[0] = 0.0f;
                        mqk[1] = 0.0f;
                        mqk[2] = 0.0f;
                        mqk[3] = 0.0f;
                        mqk[4] = 0.0f;
                        mqk[5] = 0.0f;
                        mqk[6] = 0.0f;
                        mqk[7] = 0.0f;
                        if (row0_1 >= col0_1) {
                            mqk[0] = acc[0];
                        }
                        if (row0_1 >= col0_1 + 1) {
                            mqk[1] = acc[1];
                        }
                        if (row1_1 >= col0_1) {
                            mqk[2] = acc[2];
                        }
                        if (row1_1 >= col0_1 + 1) {
                            mqk[3] = acc[3];
                        }
                        if (row0_1 >= col0_1 + 8) {
                            mqk[4] = acc[4];
                        }
                        if (row0_1 >= col0_1 + 9) {
                            mqk[5] = acc[5];
                        }
                        if (row1_1 >= col0_1 + 8) {
                            mqk[6] = acc[6];
                        }
                        if (row1_1 >= col0_1 + 9) {
                            mqk[7] = acc[7];
                        }
                        unsigned int mqk_packed[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(mqk[_lp*2 + 0], mqk[_lp*2+1 + 0]));
                            mqk_packed[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int publish_pair = 0; publish_pair < 2; publish_pair++) {
                            int publish_row = 16 + publish_pair * 8 + (lane & 7);
                            int publish_col = 128 + lane / 8 * 8;
                            uint32_t _stmatrix_addr_5 = static_cast<uint32_t>((unsigned long long)(smem_final_trans_addr + prep_stage * 41984 + (unsigned int)(publish_col / 64 * 4096 + publish_row * 128 + publish_col % 64 * 2 ^ (publish_col / 64 * 4096 + publish_row * 128 + publish_col % 64 * 2 >> 7 & 7) << 4)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1, %2};\n"
                                :: "r"(_stmatrix_addr_5), "r"(*reinterpret_cast<const uint32_t*>(&mqk_packed[publish_pair * 2])), "r"(*reinterpret_cast<const uint32_t*>(&mqk_packed[publish_pair * 2 + 1]))
                                : "memory");
                        }
                        int cross_scale_col_1 = lane % 4 * 2;
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + prep_stage * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)((lane % 16 / 8 / 8 * 256 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16) * 16))
                            : "memory");
                        int cross_word_lo0_8 = smem_v21[stage_f32 + cross_scale_col_1];
                        int cross_word_lo1_8 = smem_v21[stage_f32 + cross_scale_col_1 + 1];
                        unsigned int cross_a_lo_8 = (unsigned int)((cross_word_lo0_8 & 65408) + ((cross_word_lo1_8 & 65408) << 16));
                        unsigned int cross_b_lo_8 = (unsigned int)((cross_word_lo0_8 >> 16 & 65408) + (cross_word_lo1_8 & 4286578688));
                        int cross_word_hi0_8 = smem_v21[stage_f32 + cross_scale_col_1 + 8];
                        int cross_word_hi1_8 = smem_v21[stage_f32 + cross_scale_col_1 + 9];
                        unsigned int cross_a_hi_8 = (unsigned int)((cross_word_hi0_8 & 65408) + ((cross_word_hi1_8 & 65408) << 16));
                        unsigned int cross_b_hi_8 = (unsigned int)((cross_word_hi0_8 >> 16 & 65408) + (cross_word_hi1_8 & 4286578688));
                        uint32_t _bf16x2_mul_64;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_64) : "r"(a_frag[0]), "r"(cross_a_lo_8));
                        a_frag[0] = _bf16x2_mul_64;
                        uint32_t _bf16x2_mul_65;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_65) : "r"(a_frag[1]), "r"(cross_a_lo_8));
                        a_frag[1] = _bf16x2_mul_65;
                        uint32_t _bf16x2_mul_66;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_66) : "r"(a_frag[2]), "r"(cross_a_hi_8));
                        a_frag[2] = _bf16x2_mul_66;
                        uint32_t _bf16x2_mul_67;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_67) : "r"(a_frag[3]), "r"(cross_a_hi_8));
                        a_frag[3] = _bf16x2_mul_67;
                        uint32_t _bf16x2_mul_68;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_68) : "r"(b_frag[0]), "r"(cross_b_lo_8));
                        b_frag[0] = _bf16x2_mul_68;
                        uint32_t _bf16x2_mul_69;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_69) : "r"(b_frag[1]), "r"(cross_b_hi_8));
                        b_frag[1] = _bf16x2_mul_69;
                        uint32_t _bf16x2_mul_70;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_70) : "r"(b_frag[2]), "r"(cross_b_lo_8));
                        b_frag[2] = _bf16x2_mul_70;
                        uint32_t _bf16x2_mul_71;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_71) : "r"(b_frag[3]), "r"(cross_b_hi_8));
                        b_frag[3] = _bf16x2_mul_71;
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(acc[4]), "=f"(acc[(4) + 1]), "=f"(acc[(4) + 2]), "=f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + prep_stage * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16 ^ 2) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)(((lane % 16 / 8 / 8 * 256 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256) * 16))
                            : "memory");
                        int cross_word_lo0_0 = smem_v21[stage_f32 + 16 + cross_scale_col_1];
                        int cross_word_lo1_1_1 = smem_v21[stage_f32 + 16 + cross_scale_col_1 + 1];
                        unsigned int cross_a_lo_2_1 = (unsigned int)((cross_word_lo0_0 & 65408) + ((cross_word_lo1_1_1 & 65408) << 16));
                        unsigned int cross_b_lo_3_1 = (unsigned int)((cross_word_lo0_0 >> 16 & 65408) + (cross_word_lo1_1_1 & 4286578688));
                        int cross_word_hi0_4_1 = smem_v21[stage_f32 + 16 + cross_scale_col_1 + 8];
                        int cross_word_hi1_5_1 = smem_v21[stage_f32 + 16 + cross_scale_col_1 + 9];
                        unsigned int cross_a_hi_6_1 = (unsigned int)((cross_word_hi0_4_1 & 65408) + ((cross_word_hi1_5_1 & 65408) << 16));
                        unsigned int cross_b_hi_7_1 = (unsigned int)((cross_word_hi0_4_1 >> 16 & 65408) + (cross_word_hi1_5_1 & 4286578688));
                        uint32_t _bf16x2_mul_72;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_72) : "r"(a_frag[0]), "r"(cross_a_lo_2_1));
                        a_frag[0] = _bf16x2_mul_72;
                        uint32_t _bf16x2_mul_73;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_73) : "r"(a_frag[1]), "r"(cross_a_lo_2_1));
                        a_frag[1] = _bf16x2_mul_73;
                        uint32_t _bf16x2_mul_74;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_74) : "r"(a_frag[2]), "r"(cross_a_hi_6_1));
                        a_frag[2] = _bf16x2_mul_74;
                        uint32_t _bf16x2_mul_75;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_75) : "r"(a_frag[3]), "r"(cross_a_hi_6_1));
                        a_frag[3] = _bf16x2_mul_75;
                        uint32_t _bf16x2_mul_76;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_76) : "r"(b_frag[0]), "r"(cross_b_lo_3_1));
                        b_frag[0] = _bf16x2_mul_76;
                        uint32_t _bf16x2_mul_77;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_77) : "r"(b_frag[1]), "r"(cross_b_hi_7_1));
                        b_frag[1] = _bf16x2_mul_77;
                        uint32_t _bf16x2_mul_78;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_78) : "r"(b_frag[2]), "r"(cross_b_lo_3_1));
                        b_frag[2] = _bf16x2_mul_78;
                        uint32_t _bf16x2_mul_79;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_79) : "r"(b_frag[3]), "r"(cross_b_hi_7_1));
                        b_frag[3] = _bf16x2_mul_79;
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + prep_stage * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)((((lane % 16 / 8 / 8 * 256 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256) * 16))
                            : "memory");
                        int cross_word_lo0_8_1 = smem_v21[stage_f32 + 32 + cross_scale_col_1];
                        int cross_word_lo1_9 = smem_v21[stage_f32 + 32 + cross_scale_col_1 + 1];
                        unsigned int cross_a_lo_10 = (unsigned int)((cross_word_lo0_8_1 & 65408) + ((cross_word_lo1_9 & 65408) << 16));
                        unsigned int cross_b_lo_11 = (unsigned int)((cross_word_lo0_8_1 >> 16 & 65408) + (cross_word_lo1_9 & 4286578688));
                        int cross_word_hi0_12 = smem_v21[stage_f32 + 32 + cross_scale_col_1 + 8];
                        int cross_word_hi1_13 = smem_v21[stage_f32 + 32 + cross_scale_col_1 + 9];
                        unsigned int cross_a_hi_14 = (unsigned int)((cross_word_hi0_12 & 65408) + ((cross_word_hi1_13 & 65408) << 16));
                        unsigned int cross_b_hi_15 = (unsigned int)((cross_word_hi0_12 >> 16 & 65408) + (cross_word_hi1_13 & 4286578688));
                        uint32_t _bf16x2_mul_80;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_80) : "r"(a_frag[0]), "r"(cross_a_lo_10));
                        a_frag[0] = _bf16x2_mul_80;
                        uint32_t _bf16x2_mul_81;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_81) : "r"(a_frag[1]), "r"(cross_a_lo_10));
                        a_frag[1] = _bf16x2_mul_81;
                        uint32_t _bf16x2_mul_82;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_82) : "r"(a_frag[2]), "r"(cross_a_hi_14));
                        a_frag[2] = _bf16x2_mul_82;
                        uint32_t _bf16x2_mul_83;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_83) : "r"(a_frag[3]), "r"(cross_a_hi_14));
                        a_frag[3] = _bf16x2_mul_83;
                        uint32_t _bf16x2_mul_84;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_84) : "r"(b_frag[0]), "r"(cross_b_lo_11));
                        b_frag[0] = _bf16x2_mul_84;
                        uint32_t _bf16x2_mul_85;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_85) : "r"(b_frag[1]), "r"(cross_b_hi_15));
                        b_frag[1] = _bf16x2_mul_85;
                        uint32_t _bf16x2_mul_86;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_86) : "r"(b_frag[2]), "r"(cross_b_lo_11));
                        b_frag[2] = _bf16x2_mul_86;
                        uint32_t _bf16x2_mul_87;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_87) : "r"(b_frag[3]), "r"(cross_b_hi_15));
                        b_frag[3] = _bf16x2_mul_87;
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + prep_stage * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)(((((lane % 16 / 8 / 8 * 256 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256) * 16))
                            : "memory");
                        int cross_word_lo0_16 = smem_v21[stage_f32 + 48 + cross_scale_col_1];
                        int cross_word_lo1_17 = smem_v21[stage_f32 + 48 + cross_scale_col_1 + 1];
                        unsigned int cross_a_lo_18 = (unsigned int)((cross_word_lo0_16 & 65408) + ((cross_word_lo1_17 & 65408) << 16));
                        unsigned int cross_b_lo_19 = (unsigned int)((cross_word_lo0_16 >> 16 & 65408) + (cross_word_lo1_17 & 4286578688));
                        int cross_word_hi0_20 = smem_v21[stage_f32 + 48 + cross_scale_col_1 + 8];
                        int cross_word_hi1_21 = smem_v21[stage_f32 + 48 + cross_scale_col_1 + 9];
                        unsigned int cross_a_hi_22 = (unsigned int)((cross_word_hi0_20 & 65408) + ((cross_word_hi1_21 & 65408) << 16));
                        unsigned int cross_b_hi_23 = (unsigned int)((cross_word_hi0_20 >> 16 & 65408) + (cross_word_hi1_21 & 4286578688));
                        uint32_t _bf16x2_mul_88;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_88) : "r"(a_frag[0]), "r"(cross_a_lo_18));
                        a_frag[0] = _bf16x2_mul_88;
                        uint32_t _bf16x2_mul_89;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_89) : "r"(a_frag[1]), "r"(cross_a_lo_18));
                        a_frag[1] = _bf16x2_mul_89;
                        uint32_t _bf16x2_mul_90;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_90) : "r"(a_frag[2]), "r"(cross_a_hi_22));
                        a_frag[2] = _bf16x2_mul_90;
                        uint32_t _bf16x2_mul_91;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_91) : "r"(a_frag[3]), "r"(cross_a_hi_22));
                        a_frag[3] = _bf16x2_mul_91;
                        uint32_t _bf16x2_mul_92;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_92) : "r"(b_frag[0]), "r"(cross_b_lo_19));
                        b_frag[0] = _bf16x2_mul_92;
                        uint32_t _bf16x2_mul_93;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_93) : "r"(b_frag[1]), "r"(cross_b_hi_23));
                        b_frag[1] = _bf16x2_mul_93;
                        uint32_t _bf16x2_mul_94;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_94) : "r"(b_frag[2]), "r"(cross_b_lo_19));
                        b_frag[2] = _bf16x2_mul_94;
                        uint32_t _bf16x2_mul_95;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_95) : "r"(b_frag[3]), "r"(cross_b_hi_23));
                        b_frag[3] = _bf16x2_mul_95;
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + prep_stage * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)((((((lane % 16 / 8 / 8 * 256 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256) * 16))
                            : "memory");
                        int cross_word_lo0_24 = smem_v21[stage_f32 + 64 + cross_scale_col_1];
                        int cross_word_lo1_25 = smem_v21[stage_f32 + 64 + cross_scale_col_1 + 1];
                        unsigned int cross_a_lo_26 = (unsigned int)((cross_word_lo0_24 & 65408) + ((cross_word_lo1_25 & 65408) << 16));
                        unsigned int cross_b_lo_27 = (unsigned int)((cross_word_lo0_24 >> 16 & 65408) + (cross_word_lo1_25 & 4286578688));
                        int cross_word_hi0_28 = smem_v21[stage_f32 + 64 + cross_scale_col_1 + 8];
                        int cross_word_hi1_29 = smem_v21[stage_f32 + 64 + cross_scale_col_1 + 9];
                        unsigned int cross_a_hi_30 = (unsigned int)((cross_word_hi0_28 & 65408) + ((cross_word_hi1_29 & 65408) << 16));
                        unsigned int cross_b_hi_31 = (unsigned int)((cross_word_hi0_28 >> 16 & 65408) + (cross_word_hi1_29 & 4286578688));
                        uint32_t _bf16x2_mul_96;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_96) : "r"(a_frag[0]), "r"(cross_a_lo_26));
                        a_frag[0] = _bf16x2_mul_96;
                        uint32_t _bf16x2_mul_97;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_97) : "r"(a_frag[1]), "r"(cross_a_lo_26));
                        a_frag[1] = _bf16x2_mul_97;
                        uint32_t _bf16x2_mul_98;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_98) : "r"(a_frag[2]), "r"(cross_a_hi_30));
                        a_frag[2] = _bf16x2_mul_98;
                        uint32_t _bf16x2_mul_99;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_99) : "r"(a_frag[3]), "r"(cross_a_hi_30));
                        a_frag[3] = _bf16x2_mul_99;
                        uint32_t _bf16x2_mul_100;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_100) : "r"(b_frag[0]), "r"(cross_b_lo_27));
                        b_frag[0] = _bf16x2_mul_100;
                        uint32_t _bf16x2_mul_101;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_101) : "r"(b_frag[1]), "r"(cross_b_hi_31));
                        b_frag[1] = _bf16x2_mul_101;
                        uint32_t _bf16x2_mul_102;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_102) : "r"(b_frag[2]), "r"(cross_b_lo_27));
                        b_frag[2] = _bf16x2_mul_102;
                        uint32_t _bf16x2_mul_103;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_103) : "r"(b_frag[3]), "r"(cross_b_hi_31));
                        b_frag[3] = _bf16x2_mul_103;
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + prep_stage * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)(((((((lane % 16 / 8 / 8 * 256 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256) * 16))
                            : "memory");
                        int cross_word_lo0_32 = smem_v21[stage_f32 + 80 + cross_scale_col_1];
                        int cross_word_lo1_33 = smem_v21[stage_f32 + 80 + cross_scale_col_1 + 1];
                        unsigned int cross_a_lo_34 = (unsigned int)((cross_word_lo0_32 & 65408) + ((cross_word_lo1_33 & 65408) << 16));
                        unsigned int cross_b_lo_35 = (unsigned int)((cross_word_lo0_32 >> 16 & 65408) + (cross_word_lo1_33 & 4286578688));
                        int cross_word_hi0_36 = smem_v21[stage_f32 + 80 + cross_scale_col_1 + 8];
                        int cross_word_hi1_37 = smem_v21[stage_f32 + 80 + cross_scale_col_1 + 9];
                        unsigned int cross_a_hi_38 = (unsigned int)((cross_word_hi0_36 & 65408) + ((cross_word_hi1_37 & 65408) << 16));
                        unsigned int cross_b_hi_39 = (unsigned int)((cross_word_hi0_36 >> 16 & 65408) + (cross_word_hi1_37 & 4286578688));
                        uint32_t _bf16x2_mul_104;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_104) : "r"(a_frag[0]), "r"(cross_a_lo_34));
                        a_frag[0] = _bf16x2_mul_104;
                        uint32_t _bf16x2_mul_105;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_105) : "r"(a_frag[1]), "r"(cross_a_lo_34));
                        a_frag[1] = _bf16x2_mul_105;
                        uint32_t _bf16x2_mul_106;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_106) : "r"(a_frag[2]), "r"(cross_a_hi_38));
                        a_frag[2] = _bf16x2_mul_106;
                        uint32_t _bf16x2_mul_107;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_107) : "r"(a_frag[3]), "r"(cross_a_hi_38));
                        a_frag[3] = _bf16x2_mul_107;
                        uint32_t _bf16x2_mul_108;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_108) : "r"(b_frag[0]), "r"(cross_b_lo_35));
                        b_frag[0] = _bf16x2_mul_108;
                        uint32_t _bf16x2_mul_109;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_109) : "r"(b_frag[1]), "r"(cross_b_hi_39));
                        b_frag[1] = _bf16x2_mul_109;
                        uint32_t _bf16x2_mul_110;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_110) : "r"(b_frag[2]), "r"(cross_b_lo_35));
                        b_frag[2] = _bf16x2_mul_110;
                        uint32_t _bf16x2_mul_111;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_111) : "r"(b_frag[3]), "r"(cross_b_hi_39));
                        b_frag[3] = _bf16x2_mul_111;
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + prep_stage * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2 ^ 6) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)((((((((lane % 16 / 8 / 8 * 256 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256 + 256 ^ 6) - 256) * 16))
                            : "memory");
                        int cross_word_lo0_40 = smem_v21[stage_f32 + 96 + cross_scale_col_1];
                        int cross_word_lo1_41 = smem_v21[stage_f32 + 96 + cross_scale_col_1 + 1];
                        unsigned int cross_a_lo_42 = (unsigned int)((cross_word_lo0_40 & 65408) + ((cross_word_lo1_41 & 65408) << 16));
                        unsigned int cross_b_lo_43 = (unsigned int)((cross_word_lo0_40 >> 16 & 65408) + (cross_word_lo1_41 & 4286578688));
                        int cross_word_hi0_44 = smem_v21[stage_f32 + 96 + cross_scale_col_1 + 8];
                        int cross_word_hi1_45 = smem_v21[stage_f32 + 96 + cross_scale_col_1 + 9];
                        unsigned int cross_a_hi_46 = (unsigned int)((cross_word_hi0_44 & 65408) + ((cross_word_hi1_45 & 65408) << 16));
                        unsigned int cross_b_hi_47 = (unsigned int)((cross_word_hi0_44 >> 16 & 65408) + (cross_word_hi1_45 & 4286578688));
                        uint32_t _bf16x2_mul_112;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_112) : "r"(a_frag[0]), "r"(cross_a_lo_42));
                        a_frag[0] = _bf16x2_mul_112;
                        uint32_t _bf16x2_mul_113;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_113) : "r"(a_frag[1]), "r"(cross_a_lo_42));
                        a_frag[1] = _bf16x2_mul_113;
                        uint32_t _bf16x2_mul_114;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_114) : "r"(a_frag[2]), "r"(cross_a_hi_46));
                        a_frag[2] = _bf16x2_mul_114;
                        uint32_t _bf16x2_mul_115;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_115) : "r"(a_frag[3]), "r"(cross_a_hi_46));
                        a_frag[3] = _bf16x2_mul_115;
                        uint32_t _bf16x2_mul_116;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_116) : "r"(b_frag[0]), "r"(cross_b_lo_43));
                        b_frag[0] = _bf16x2_mul_116;
                        uint32_t _bf16x2_mul_117;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_117) : "r"(b_frag[1]), "r"(cross_b_hi_47));
                        b_frag[1] = _bf16x2_mul_117;
                        uint32_t _bf16x2_mul_118;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_118) : "r"(b_frag[2]), "r"(cross_b_lo_43));
                        b_frag[2] = _bf16x2_mul_118;
                        uint32_t _bf16x2_mul_119;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_119) : "r"(b_frag[3]), "r"(cross_b_hi_47));
                        b_frag[3] = _bf16x2_mul_119;
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + prep_stage * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2 ^ 6 ^ 2) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)(((((((((lane % 16 / 8 / 8 * 256 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256) * 16))
                            : "memory");
                        int cross_word_lo0_48 = smem_v21[stage_f32 + 112 + cross_scale_col_1];
                        int cross_word_lo1_49 = smem_v21[stage_f32 + 112 + cross_scale_col_1 + 1];
                        unsigned int cross_a_lo_50 = (unsigned int)((cross_word_lo0_48 & 65408) + ((cross_word_lo1_49 & 65408) << 16));
                        unsigned int cross_b_lo_51 = (unsigned int)((cross_word_lo0_48 >> 16 & 65408) + (cross_word_lo1_49 & 4286578688));
                        int cross_word_hi0_52 = smem_v21[stage_f32 + 112 + cross_scale_col_1 + 8];
                        int cross_word_hi1_53 = smem_v21[stage_f32 + 112 + cross_scale_col_1 + 9];
                        unsigned int cross_a_hi_54 = (unsigned int)((cross_word_hi0_52 & 65408) + ((cross_word_hi1_53 & 65408) << 16));
                        unsigned int cross_b_hi_55 = (unsigned int)((cross_word_hi0_52 >> 16 & 65408) + (cross_word_hi1_53 & 4286578688));
                        uint32_t _bf16x2_mul_120;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_120) : "r"(a_frag[0]), "r"(cross_a_lo_50));
                        a_frag[0] = _bf16x2_mul_120;
                        uint32_t _bf16x2_mul_121;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_121) : "r"(a_frag[1]), "r"(cross_a_lo_50));
                        a_frag[1] = _bf16x2_mul_121;
                        uint32_t _bf16x2_mul_122;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_122) : "r"(a_frag[2]), "r"(cross_a_hi_54));
                        a_frag[2] = _bf16x2_mul_122;
                        uint32_t _bf16x2_mul_123;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_123) : "r"(a_frag[3]), "r"(cross_a_hi_54));
                        a_frag[3] = _bf16x2_mul_123;
                        uint32_t _bf16x2_mul_124;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_124) : "r"(b_frag[0]), "r"(cross_b_lo_51));
                        b_frag[0] = _bf16x2_mul_124;
                        uint32_t _bf16x2_mul_125;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_125) : "r"(b_frag[1]), "r"(cross_b_hi_55));
                        b_frag[1] = _bf16x2_mul_125;
                        uint32_t _bf16x2_mul_126;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_126) : "r"(b_frag[2]), "r"(cross_b_lo_51));
                        b_frag[2] = _bf16x2_mul_126;
                        uint32_t _bf16x2_mul_127;
                        asm volatile("mul.rn.bf16x2 %0, %1, %2;" : "=r"(_bf16x2_mul_127) : "r"(b_frag[3]), "r"(cross_b_hi_55));
                        b_frag[3] = _bf16x2_mul_127;
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        int row0_56 = 16 + lane / 4;
                        int row1_57 = row0_56 + 8;
                        int col0_58 = lane % 4 * 2;
                        float mqk_59[8];
                        mqk_59[0] = 0.0f;
                        mqk_59[1] = 0.0f;
                        mqk_59[2] = 0.0f;
                        mqk_59[3] = 0.0f;
                        mqk_59[4] = 0.0f;
                        mqk_59[5] = 0.0f;
                        mqk_59[6] = 0.0f;
                        mqk_59[7] = 0.0f;
                        if (row0_56 >= col0_58) {
                            mqk_59[0] = acc[0];
                        }
                        if (row0_56 >= col0_58 + 1) {
                            mqk_59[1] = acc[1];
                        }
                        if (row1_57 >= col0_58) {
                            mqk_59[2] = acc[2];
                        }
                        if (row1_57 >= col0_58 + 1) {
                            mqk_59[3] = acc[3];
                        }
                        if (row0_56 >= col0_58 + 8) {
                            mqk_59[4] = acc[4];
                        }
                        if (row0_56 >= col0_58 + 9) {
                            mqk_59[5] = acc[5];
                        }
                        if (row1_57 >= col0_58 + 8) {
                            mqk_59[6] = acc[6];
                        }
                        if (row1_57 >= col0_58 + 9) {
                            mqk_59[7] = acc[7];
                        }
                        unsigned int mqk_packed_60[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(mqk_59[_lp*2 + 0], mqk_59[_lp*2+1 + 0]));
                            mqk_packed_60[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int publish_pair_1 = 0; publish_pair_1 < 2; publish_pair_1++) {
                            int publish_row_1 = publish_pair_1 * 8 + (lane & 7);
                            int publish_col_1 = 144 + lane / 8 * 8;
                            uint32_t _stmatrix_addr_6 = static_cast<uint32_t>((unsigned long long)(smem_final_trans_addr + prep_stage * 41984 + (unsigned int)(publish_col_1 / 64 * 4096 + publish_row_1 * 128 + publish_col_1 % 64 * 2 ^ (publish_col_1 / 64 * 4096 + publish_row_1 * 128 + publish_col_1 % 64 * 2 >> 7 & 7) << 4)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1, %2};\n"
                                :: "r"(_stmatrix_addr_6), "r"(*reinterpret_cast<const uint32_t*>(&mqk_packed_60[publish_pair_1 * 2])), "r"(*reinterpret_cast<const uint32_t*>(&mqk_packed_60[publish_pair_1 * 2 + 1]))
                                : "memory");
                        }
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + prep_stage * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)((lane % 16 / 8 / 8 * 256 + (16 + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (16 + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(acc[4]), "=f"(acc[(4) + 1]), "=f"(acc[(4) + 2]), "=f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + prep_stage * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16 ^ 2) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)(((lane % 16 / 8 / 8 * 256 + (16 + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (16 + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + prep_stage * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)((((lane % 16 / 8 / 8 * 256 + (16 + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (16 + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + prep_stage * 41984 + (unsigned int)((lane / 16 / 8 * 256 + (16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)(((((lane % 16 / 8 / 8 * 256 + (16 + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (16 + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + prep_stage * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)((((((lane % 16 / 8 / 8 * 256 + (16 + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (16 + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + prep_stage * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)(((((((lane % 16 / 8 / 8 * 256 + (16 + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (16 + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + prep_stage * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2 ^ 6) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)((((((((lane % 16 / 8 / 8 * 256 + (16 + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (16 + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256 + 256 ^ 6) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + prep_stage * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + (16 + lane % 16) * 8 + (lane / 16 % 8 * 16 ^ (16 + lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2 ^ 6 ^ 2) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)(((((((((lane % 16 / 8 / 8 * 256 + (16 + 8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (16 + 8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        int row0_61 = 16 + lane / 4;
                        int row1_62 = row0_61 + 8;
                        int col0_63 = 16 + lane % 4 * 2;
                        float mqk_64[8];
                        mqk_64[0] = 0.0f;
                        mqk_64[1] = 0.0f;
                        mqk_64[2] = 0.0f;
                        mqk_64[3] = 0.0f;
                        mqk_64[4] = 0.0f;
                        mqk_64[5] = 0.0f;
                        mqk_64[6] = 0.0f;
                        mqk_64[7] = 0.0f;
                        if (row0_61 >= col0_63) {
                            mqk_64[0] = acc[0];
                        }
                        if (row0_61 >= col0_63 + 1) {
                            mqk_64[1] = acc[1];
                        }
                        if (row1_62 >= col0_63) {
                            mqk_64[2] = acc[2];
                        }
                        if (row1_62 >= col0_63 + 1) {
                            mqk_64[3] = acc[3];
                        }
                        if (row0_61 >= col0_63 + 8) {
                            mqk_64[4] = acc[4];
                        }
                        if (row0_61 >= col0_63 + 9) {
                            mqk_64[5] = acc[5];
                        }
                        if (row1_62 >= col0_63 + 8) {
                            mqk_64[6] = acc[6];
                        }
                        if (row1_62 >= col0_63 + 9) {
                            mqk_64[7] = acc[7];
                        }
                        unsigned int mqk_packed_65[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(mqk_64[_lp*2 + 0], mqk_64[_lp*2+1 + 0]));
                            mqk_packed_65[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int publish_pair_2 = 0; publish_pair_2 < 2; publish_pair_2++) {
                            int publish_row_2 = 16 + publish_pair_2 * 8 + (lane & 7);
                            int publish_col_2 = 144 + lane / 8 * 8;
                            uint32_t _stmatrix_addr_7 = static_cast<uint32_t>((unsigned long long)(smem_final_trans_addr + prep_stage * 41984 + (unsigned int)(publish_col_2 / 64 * 4096 + publish_row_2 * 128 + publish_col_2 % 64 * 2 ^ (publish_col_2 / 64 * 4096 + publish_row_2 * 128 + publish_col_2 % 64 * 2 >> 7 & 7) << 4)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1, %2};\n"
                                :: "r"(_stmatrix_addr_7), "r"(*reinterpret_cast<const uint32_t*>(&mqk_packed_65[publish_pair_2 * 2])), "r"(*reinterpret_cast<const uint32_t*>(&mqk_packed_65[publish_pair_2 * 2 + 1]))
                                : "memory");
                        }
                    } else if (prep_local_warp == 2) {
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + prep_stage * 41984 + (unsigned int)((lane / 16 / 8 * 256 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)((lane % 16 / 8 / 8 * 256 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]), "=f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                            : "=f"(acc[4]), "=f"(acc[(4) + 1]), "=f"(acc[(4) + 2]), "=f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + prep_stage * 41984 + (unsigned int)((lane / 16 / 8 * 256 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)(((lane % 16 / 8 / 8 * 256 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + prep_stage * 41984 + (unsigned int)((lane / 16 / 8 * 256 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)((((lane % 16 / 8 / 8 * 256 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + prep_stage * 41984 + (unsigned int)((lane / 16 / 8 * 256 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)(((((lane % 16 / 8 / 8 * 256 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + prep_stage * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)((((((lane % 16 / 8 / 8 * 256 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + prep_stage * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)(((((((lane % 16 / 8 / 8 * 256 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + prep_stage * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2 ^ 6) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)((((((((lane % 16 / 8 / 8 * 256 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256 + 256 ^ 6) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(a_frag[0]), "=r"(a_frag[1]), "=r"(a_frag[2]), "=r"(a_frag[3])
                            : "r"(smem_qd_addr + prep_stage * 41984 + (unsigned int)(((lane / 16 / 8 * 256 + lane % 16 * 8 + (lane / 16 % 8 * 16 ^ (lane % 16 & 7) << 4) / 16 ^ 2 ^ 6 ^ 2 ^ 6) + 256 ^ 2 ^ 6 ^ 2) * 16))
                            : "memory");
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                            : "=r"(b_frag[0]), "=r"(b_frag[1]), "=r"(b_frag[2]), "=r"(b_frag[3])
                            : "r"(smem_ki_addr + prep_stage * 41984 + (unsigned int)(((((((((lane % 16 / 8 / 8 * 256 + (8 * (lane / 16) + lane % 8) * 8 + (lane % 16 / 8 % 8 * 16 ^ (8 * (lane / 16) + lane % 8 & 7) << 4) / 16 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256 + 256 ^ 6) + 256 - 256 + 256 ^ 2) - 256 + 256 ^ 6) - 256 + 256 ^ 2) - 256) * 16))
                            : "memory");
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[0]), "r"(b_frag[1]));
                        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3};\n"
                            : "+f"(acc[4]), "+f"(acc[(4) + 1]), "+f"(acc[(4) + 2]), "+f"(acc[(4) + 3])
                            : "r"(a_frag[0]), "r"(a_frag[1]), "r"(a_frag[2]), "r"(a_frag[3]), "r"(b_frag[2]), "r"(b_frag[(2) + 1]));
                        int row0_2 = lane / 4;
                        int row1_2 = row0_2 + 8;
                        int col0_2 = lane % 4 * 2;
                        float mqk_1[8];
                        mqk_1[0] = 0.0f;
                        mqk_1[1] = 0.0f;
                        mqk_1[2] = 0.0f;
                        mqk_1[3] = 0.0f;
                        mqk_1[4] = 0.0f;
                        mqk_1[5] = 0.0f;
                        mqk_1[6] = 0.0f;
                        mqk_1[7] = 0.0f;
                        if (row0_2 >= col0_2) {
                            mqk_1[0] = acc[0];
                        }
                        if (row0_2 >= col0_2 + 1) {
                            mqk_1[1] = acc[1];
                        }
                        if (row1_2 >= col0_2) {
                            mqk_1[2] = acc[2];
                        }
                        if (row1_2 >= col0_2 + 1) {
                            mqk_1[3] = acc[3];
                        }
                        if (row0_2 >= col0_2 + 8) {
                            mqk_1[4] = acc[4];
                        }
                        if (row0_2 >= col0_2 + 9) {
                            mqk_1[5] = acc[5];
                        }
                        if (row1_2 >= col0_2 + 8) {
                            mqk_1[6] = acc[6];
                        }
                        if (row1_2 >= col0_2 + 9) {
                            mqk_1[7] = acc[7];
                        }
                        unsigned int mqk_packed_1[4];
                        #pragma unroll
                        for (int _lp = 0; _lp < 4; _lp++) {
                            __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(mqk_1[_lp*2 + 0], mqk_1[_lp*2+1 + 0]));
                            mqk_packed_1[_lp] = *(uint32_t*)&_bf2;
                        }
                        #pragma unroll
                        for (int publish_pair_3 = 0; publish_pair_3 < 2; publish_pair_3++) {
                            int publish_row_3 = publish_pair_3 * 8 + (lane & 7);
                            int publish_col_3 = 128 + lane / 8 * 8;
                            uint32_t _stmatrix_addr_8 = static_cast<uint32_t>((unsigned long long)(smem_final_trans_addr + prep_stage * 41984 + (unsigned int)(publish_col_3 / 64 * 4096 + publish_row_3 * 128 + publish_col_3 % 64 * 2 ^ (publish_col_3 / 64 * 4096 + publish_row_3 * 128 + publish_col_3 % 64 * 2 >> 7 & 7) << 4)));
                            asm volatile("stmatrix.sync.aligned.m8n8.x2.trans.shared.b16 [%0], {%1, %2};\n"
                                :: "r"(_stmatrix_addr_8), "r"(*reinterpret_cast<const uint32_t*>(&mqk_packed_1[publish_pair_3 * 2])), "r"(*reinterpret_cast<const uint32_t*>(&mqk_packed_1[publish_pair_3 * 2 + 1]))
                                : "memory");
                        }
                    }
                    asm volatile("barrier.sync %0, 128;" :: "r"(10 + prep_instance) : "memory");
                }
                long long tape_scaled_base = 0;
                if (prep_tid < 128) {
                    float total_log2_1 = smem_gt_prefix_all[stage_f32 + prep_tid];
                    float _exp2_6 = approx_exp2(unbounded_total_log2);
                    smem_gt_all[stage_f32 + prep_tid] = _exp2_6;
                }
                {
                    if (prep_local_warp == 1) {
                        int stage_f32_0 = prep_stage * 10496;
                        float restore_scale = smem_restore_factor_all[stage_f32_0 + 128];
                        float restore_factor[8];
                        int restore_segment = lane & 15;
                        float qk_factor[8];
                        float qk_tail[8];
                        float kr_factor[8];
                        float kr_tail[8];
                        #pragma unroll
                        for (int factor_elem = 0; factor_elem < 8; factor_elem++) {
                            int factor_col = restore_segment * 8 + factor_elem;
                            int unbounded_restore_word = smem_v21[stage_f32_0 + factor_col];
                            int unbounded_restore_anchor_int = -((unbounded_restore_word & 127) + ((unbounded_restore_word >> 16 & 127) << 7)) + (127 - (unbounded_restore_word >> 23 & 255));
                            float unbounded_restore_anchor = (float)unbounded_restore_anchor_int;
                            float unbounded_kr_log2 = smem_gt_prefix_all[stage_f32_0 + factor_col] - unbounded_restore_anchor;
                            int _max_5 = ((unbounded_restore_anchor_int) > (-126) ? (unbounded_restore_anchor_int) : (-126));
                            int unbounded_qk_factor_bits = _max_5 + 127 << 23;
                            qk_factor[factor_elem] = __uint_as_float((unsigned int)unbounded_qk_factor_bits);
                            int _min_2 = ((unbounded_restore_anchor_int + 126) < (0) ? (unbounded_restore_anchor_int + 126) : (0));
                            int _max_6 = ((_min_2) > (-126) ? (_min_2) : (-126));
                            int unbounded_qk_tail_bits = _max_6 + 127 << 23;
                            qk_tail[factor_elem] = __uint_as_float((unsigned int)unbounded_qk_tail_bits);
                            float _max_7 = max_noftz(unbounded_kr_log2, -126.0f);
                            float _exp2_7 = approx_exp2(_max_7);
                            kr_factor[factor_elem] = _exp2_7;
                            float _max_8 = max_noftz(unbounded_kr_log2, -126.0f);
                            float unbounded_restore_head = _max_8;
                            float _max_9 = max_noftz(unbounded_kr_log2 - unbounded_restore_head, -126.0f);
                            float _exp2_8 = approx_exp2(_max_9);
                            kr_tail[factor_elem] = _exp2_8;
                        }
                        #pragma unroll 1
                        for (int restore_pass = 0; restore_pass < 4; restore_pass++) {
                            int restore_row = restore_pass * 2 + (lane >> 4);
                            float restore_qd_values[8];
                            float restore_kd_values[8];
                            float restore_ki_values[8];
                            float restore_kr_values[8];
                            unsigned int packed_4[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&packed_4[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_4[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_4[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_4[(0) + 3]))
                                : "r"((smem_qd_addr + prep_stage * 41984 + (unsigned int)(restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 ^ (restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 >> 7 & 7) << 4))));
                            float packed_f32_1[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&packed_f32_1[_pair * 2])[0]), "=f"((&packed_f32_1[_pair * 2])[1])
                                    : "r"(packed_4[_pair]));
                            }
                            #pragma unroll
                            for (int value_idx_2 = 0; value_idx_2 < 8; value_idx_2++) {
                                restore_qd_values[value_idx_2] = packed_f32_1[value_idx_2];
                            }
                            unsigned int packed_0_1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&packed_0_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0_1[(0) + 3]))
                                : "r"((smem_kd_addr + prep_stage * 41984 + (unsigned int)(restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 ^ (restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 >> 7 & 7) << 4))));
                            float packed_0_f32_1[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&packed_0_f32_1[_pair * 2])[0]), "=f"((&packed_0_f32_1[_pair * 2])[1])
                                    : "r"(packed_0_1[_pair]));
                            }
                            #pragma unroll
                            for (int value_idx_3 = 0; value_idx_3 < 8; value_idx_3++) {
                                restore_kd_values[value_idx_3] = packed_0_f32_1[value_idx_3];
                            }
                            unsigned int packed_1_1[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&packed_1_1[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_1_1[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_1_1[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_1_1[(0) + 3]))
                                : "r"((smem_ki_addr + prep_stage * 41984 + (unsigned int)(restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 ^ (restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 >> 7 & 7) << 4))));
                            float packed_1_f32[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&packed_1_f32[_pair * 2])[0]), "=f"((&packed_1_f32[_pair * 2])[1])
                                    : "r"(packed_1_1[_pair]));
                            }
                            #pragma unroll
                            for (int value_idx_4 = 0; value_idx_4 < 8; value_idx_4++) {
                                restore_ki_values[value_idx_4] = packed_1_f32[value_idx_4];
                            }
                            #pragma unroll
                            for (int restore_elem = 0; restore_elem < 8; restore_elem++) {
                                restore_qd_values[restore_elem] = restore_qd_values[restore_elem] * qk_factor[restore_elem];
                                restore_kd_values[restore_elem] = restore_kd_values[restore_elem] * qk_factor[restore_elem];
                                restore_kr_values[restore_elem] = restore_ki_values[restore_elem] * kr_factor[restore_elem];
                            }
                            #pragma unroll
                            for (int restore_elem_1 = 0; restore_elem_1 < 8; restore_elem_1++) {
                                restore_qd_values[restore_elem_1] = restore_qd_values[restore_elem_1] * qk_tail[restore_elem_1];
                                restore_kd_values[restore_elem_1] = restore_kd_values[restore_elem_1] * qk_tail[restore_elem_1];
                                restore_kr_values[restore_elem_1] = restore_kr_values[restore_elem_1] * kr_tail[restore_elem_1];
                            }
                            unsigned int packed_2_1[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(restore_qd_values[_lp*2 + 0], restore_qd_values[_lp*2+1 + 0]));
                                packed_2_1[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int word_3 = 0; word_3 < 4; word_3++) {
                                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_qd_addr + prep_stage * 41984 + (unsigned int)(restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 ^ (restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_3 * 4)), "r"((packed_2_1[word_3])));
                            }
                            unsigned int packed_3_1[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(restore_kd_values[_lp*2 + 0], restore_kd_values[_lp*2+1 + 0]));
                                packed_3_1[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int word_4 = 0; word_4 < 4; word_4++) {
                                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_kd_addr + prep_stage * 41984 + (unsigned int)(restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 ^ (restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_4 * 4)), "r"((packed_3_1[word_4])));
                            }
                            unsigned int packed_4_1[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(restore_kr_values[_lp*2 + 0], restore_kr_values[_lp*2+1 + 0]));
                                packed_4_1[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int word_5 = 0; word_5 < 4; word_5++) {
                                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_kr_trans_addr + prep_stage * 41984 + (unsigned int)(restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 ^ (restore_segment * 8 / 64 * 4096 + restore_row * 128 + restore_segment * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_5 * 4)), "r"((packed_4_1[word_5])));
                            }
                        }
                    }
                    if (prep_local_warp == 2) {
                        int stage_f32_0_1 = prep_stage * 10496;
                        float restore_scale_1 = smem_restore_factor_all[stage_f32_0_1 + 128];
                        float restore_factor_1[8];
                        int restore_segment_1 = lane & 15;
                        float qk_factor_1[8];
                        float qk_tail_1[8];
                        float kr_factor_1[8];
                        float kr_tail_1[8];
                        #pragma unroll
                        for (int factor_elem_1 = 0; factor_elem_1 < 8; factor_elem_1++) {
                            int factor_col_1 = restore_segment_1 * 8 + factor_elem_1;
                            int unbounded_restore_word_1 = smem_v21[stage_f32_0_1 + factor_col_1];
                            int unbounded_restore_anchor_int_1 = -((unbounded_restore_word_1 & 127) + ((unbounded_restore_word_1 >> 16 & 127) << 7)) + (127 - (unbounded_restore_word_1 >> 23 & 255));
                            float unbounded_restore_anchor_1 = (float)unbounded_restore_anchor_int_1;
                            float unbounded_kr_log2_1 = smem_gt_prefix_all[stage_f32_0_1 + factor_col_1] - unbounded_restore_anchor_1;
                            int _max_10 = ((unbounded_restore_anchor_int_1) > (-126) ? (unbounded_restore_anchor_int_1) : (-126));
                            int unbounded_qk_factor_bits_1 = _max_10 + 127 << 23;
                            qk_factor_1[factor_elem_1] = __uint_as_float((unsigned int)unbounded_qk_factor_bits_1);
                            int _min_3 = ((unbounded_restore_anchor_int_1 + 126) < (0) ? (unbounded_restore_anchor_int_1 + 126) : (0));
                            int _max_11 = ((_min_3) > (-126) ? (_min_3) : (-126));
                            int unbounded_qk_tail_bits_1 = _max_11 + 127 << 23;
                            qk_tail_1[factor_elem_1] = __uint_as_float((unsigned int)unbounded_qk_tail_bits_1);
                            float _max_12 = max_noftz(unbounded_kr_log2_1, -126.0f);
                            float _exp2_9 = approx_exp2(_max_12);
                            kr_factor_1[factor_elem_1] = _exp2_9;
                            float _max_13 = max_noftz(unbounded_kr_log2_1, -126.0f);
                            float unbounded_restore_head_1 = _max_13;
                            float _max_14 = max_noftz(unbounded_kr_log2_1 - unbounded_restore_head_1, -126.0f);
                            float _exp2_10 = approx_exp2(_max_14);
                            kr_tail_1[factor_elem_1] = _exp2_10;
                        }
                        #pragma unroll 1
                        for (int restore_pass_1 = 0; restore_pass_1 < 4; restore_pass_1++) {
                            int restore_row_1 = 8 + restore_pass_1 * 2 + (lane >> 4);
                            float restore_qd_values_1[8];
                            float restore_kd_values_1[8];
                            float restore_ki_values_1[8];
                            float restore_kr_values_1[8];
                            unsigned int packed_5[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&packed_5[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_5[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_5[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_5[(0) + 3]))
                                : "r"((smem_qd_addr + prep_stage * 41984 + (unsigned int)(restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 ^ (restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 >> 7 & 7) << 4))));
                            float packed_f32_2[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&packed_f32_2[_pair * 2])[0]), "=f"((&packed_f32_2[_pair * 2])[1])
                                    : "r"(packed_5[_pair]));
                            }
                            #pragma unroll
                            for (int value_idx_5 = 0; value_idx_5 < 8; value_idx_5++) {
                                restore_qd_values_1[value_idx_5] = packed_f32_2[value_idx_5];
                            }
                            unsigned int packed_0_2[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&packed_0_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0_2[(0) + 3]))
                                : "r"((smem_kd_addr + prep_stage * 41984 + (unsigned int)(restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 ^ (restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 >> 7 & 7) << 4))));
                            float packed_0_f32_2[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&packed_0_f32_2[_pair * 2])[0]), "=f"((&packed_0_f32_2[_pair * 2])[1])
                                    : "r"(packed_0_2[_pair]));
                            }
                            #pragma unroll
                            for (int value_idx_6 = 0; value_idx_6 < 8; value_idx_6++) {
                                restore_kd_values_1[value_idx_6] = packed_0_f32_2[value_idx_6];
                            }
                            unsigned int packed_1_2[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&packed_1_2[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_1_2[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_1_2[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_1_2[(0) + 3]))
                                : "r"((smem_ki_addr + prep_stage * 41984 + (unsigned int)(restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 ^ (restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 >> 7 & 7) << 4))));
                            float packed_1_f32_1[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&packed_1_f32_1[_pair * 2])[0]), "=f"((&packed_1_f32_1[_pair * 2])[1])
                                    : "r"(packed_1_2[_pair]));
                            }
                            #pragma unroll
                            for (int value_idx_7 = 0; value_idx_7 < 8; value_idx_7++) {
                                restore_ki_values_1[value_idx_7] = packed_1_f32_1[value_idx_7];
                            }
                            #pragma unroll
                            for (int restore_elem_2 = 0; restore_elem_2 < 8; restore_elem_2++) {
                                restore_qd_values_1[restore_elem_2] = restore_qd_values_1[restore_elem_2] * qk_factor_1[restore_elem_2];
                                restore_kd_values_1[restore_elem_2] = restore_kd_values_1[restore_elem_2] * qk_factor_1[restore_elem_2];
                                restore_kr_values_1[restore_elem_2] = restore_ki_values_1[restore_elem_2] * kr_factor_1[restore_elem_2];
                            }
                            #pragma unroll
                            for (int restore_elem_3 = 0; restore_elem_3 < 8; restore_elem_3++) {
                                restore_qd_values_1[restore_elem_3] = restore_qd_values_1[restore_elem_3] * qk_tail_1[restore_elem_3];
                                restore_kd_values_1[restore_elem_3] = restore_kd_values_1[restore_elem_3] * qk_tail_1[restore_elem_3];
                                restore_kr_values_1[restore_elem_3] = restore_kr_values_1[restore_elem_3] * kr_tail_1[restore_elem_3];
                            }
                            unsigned int packed_2_2[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(restore_qd_values_1[_lp*2 + 0], restore_qd_values_1[_lp*2+1 + 0]));
                                packed_2_2[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int word_6 = 0; word_6 < 4; word_6++) {
                                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_qd_addr + prep_stage * 41984 + (unsigned int)(restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 ^ (restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_6 * 4)), "r"((packed_2_2[word_6])));
                            }
                            unsigned int packed_3_2[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(restore_kd_values_1[_lp*2 + 0], restore_kd_values_1[_lp*2+1 + 0]));
                                packed_3_2[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int word_7 = 0; word_7 < 4; word_7++) {
                                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_kd_addr + prep_stage * 41984 + (unsigned int)(restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 ^ (restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_7 * 4)), "r"((packed_3_2[word_7])));
                            }
                            unsigned int packed_4_2[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(restore_kr_values_1[_lp*2 + 0], restore_kr_values_1[_lp*2+1 + 0]));
                                packed_4_2[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int word_8 = 0; word_8 < 4; word_8++) {
                                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_kr_trans_addr + prep_stage * 41984 + (unsigned int)(restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 ^ (restore_segment_1 * 8 / 64 * 4096 + restore_row_1 * 128 + restore_segment_1 * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_8 * 4)), "r"((packed_4_2[word_8])));
                            }
                        }
                    }
                    if (prep_local_warp == 3) {
                        int stage_f32_0_2 = prep_stage * 10496;
                        float restore_scale_2 = smem_restore_factor_all[stage_f32_0_2 + 128];
                        float restore_factor_2[8];
                        int restore_segment_2 = lane & 15;
                        float qk_factor_2[8];
                        float qk_tail_2[8];
                        float kr_factor_2[8];
                        float kr_tail_2[8];
                        #pragma unroll
                        for (int factor_elem_2 = 0; factor_elem_2 < 8; factor_elem_2++) {
                            int factor_col_2 = restore_segment_2 * 8 + factor_elem_2;
                            int unbounded_restore_word_2 = smem_v21[stage_f32_0_2 + factor_col_2];
                            int unbounded_restore_anchor_int_2 = -((unbounded_restore_word_2 & 127) + ((unbounded_restore_word_2 >> 16 & 127) << 7)) - (127 - (unbounded_restore_word_2 >> 7 & 255));
                            float unbounded_restore_anchor_2 = (float)unbounded_restore_anchor_int_2;
                            float unbounded_kr_log2_2 = smem_gt_prefix_all[stage_f32_0_2 + factor_col_2] - unbounded_restore_anchor_2;
                            int _max_15 = ((unbounded_restore_anchor_int_2) > (-126) ? (unbounded_restore_anchor_int_2) : (-126));
                            int unbounded_qk_factor_bits_2 = _max_15 + 127 << 23;
                            qk_factor_2[factor_elem_2] = __uint_as_float((unsigned int)unbounded_qk_factor_bits_2);
                            int _min_4 = ((unbounded_restore_anchor_int_2 + 126) < (0) ? (unbounded_restore_anchor_int_2 + 126) : (0));
                            int _max_16 = ((_min_4) > (-126) ? (_min_4) : (-126));
                            int unbounded_qk_tail_bits_2 = _max_16 + 127 << 23;
                            qk_tail_2[factor_elem_2] = __uint_as_float((unsigned int)unbounded_qk_tail_bits_2);
                            float _max_17 = max_noftz(unbounded_kr_log2_2, -126.0f);
                            float _exp2_11 = approx_exp2(_max_17);
                            kr_factor_2[factor_elem_2] = _exp2_11;
                            float _max_18 = max_noftz(unbounded_kr_log2_2, -126.0f);
                            float unbounded_restore_head_2 = _max_18;
                            float _max_19 = max_noftz(unbounded_kr_log2_2 - unbounded_restore_head_2, -126.0f);
                            float _exp2_12 = approx_exp2(_max_19);
                            kr_tail_2[factor_elem_2] = _exp2_12;
                        }
                        #pragma unroll 1
                        for (int restore_pass_2 = 0; restore_pass_2 < 8; restore_pass_2++) {
                            int restore_row_2 = 16 + restore_pass_2 * 2 + (lane >> 4);
                            float restore_qd_values_2[8];
                            float restore_kd_values_2[8];
                            float restore_ki_values_2[8];
                            float restore_kr_values_2[8];
                            unsigned int packed_6[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&packed_6[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_6[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_6[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_6[(0) + 3]))
                                : "r"((smem_qd_addr + prep_stage * 41984 + (unsigned int)(restore_segment_2 * 8 / 64 * 4096 + restore_row_2 * 128 + restore_segment_2 * 8 % 64 * 2 ^ (restore_segment_2 * 8 / 64 * 4096 + restore_row_2 * 128 + restore_segment_2 * 8 % 64 * 2 >> 7 & 7) << 4))));
                            float packed_f32_3[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&packed_f32_3[_pair * 2])[0]), "=f"((&packed_f32_3[_pair * 2])[1])
                                    : "r"(packed_6[_pair]));
                            }
                            #pragma unroll
                            for (int value_idx_8 = 0; value_idx_8 < 8; value_idx_8++) {
                                restore_qd_values_2[value_idx_8] = packed_f32_3[value_idx_8];
                            }
                            unsigned int packed_0_3[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&packed_0_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_0_3[(0) + 3]))
                                : "r"((smem_kd_addr + prep_stage * 41984 + (unsigned int)(restore_segment_2 * 8 / 64 * 4096 + restore_row_2 * 128 + restore_segment_2 * 8 % 64 * 2 ^ (restore_segment_2 * 8 / 64 * 4096 + restore_row_2 * 128 + restore_segment_2 * 8 % 64 * 2 >> 7 & 7) << 4))));
                            float packed_0_f32_3[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&packed_0_f32_3[_pair * 2])[0]), "=f"((&packed_0_f32_3[_pair * 2])[1])
                                    : "r"(packed_0_3[_pair]));
                            }
                            #pragma unroll
                            for (int value_idx_9 = 0; value_idx_9 < 8; value_idx_9++) {
                                restore_kd_values_2[value_idx_9] = packed_0_f32_3[value_idx_9];
                            }
                            unsigned int packed_1_3[4];
                            asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                                : "=r"(*reinterpret_cast<uint32_t*>(&packed_1_3[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_1_3[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_1_3[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_1_3[(0) + 3]))
                                : "r"((smem_ki_addr + prep_stage * 41984 + (unsigned int)(restore_segment_2 * 8 / 64 * 4096 + restore_row_2 * 128 + restore_segment_2 * 8 % 64 * 2 ^ (restore_segment_2 * 8 / 64 * 4096 + restore_row_2 * 128 + restore_segment_2 * 8 % 64 * 2 >> 7 & 7) << 4))));
                            float packed_1_f32_2[8];
                            #pragma unroll
                            for (int _pair = 0; _pair < 4; _pair++) {
                                asm volatile(
                                    "{\n\t"
                                    "shl.b32 %0, %2, 16;\n\t"
                                    "and.b32 %1, %2, 0xffff0000;\n\t"
                                    "}\n"
                                    : "=f"((&packed_1_f32_2[_pair * 2])[0]), "=f"((&packed_1_f32_2[_pair * 2])[1])
                                    : "r"(packed_1_3[_pair]));
                            }
                            #pragma unroll
                            for (int value_idx_10 = 0; value_idx_10 < 8; value_idx_10++) {
                                restore_ki_values_2[value_idx_10] = packed_1_f32_2[value_idx_10];
                            }
                            #pragma unroll
                            for (int restore_elem_4 = 0; restore_elem_4 < 8; restore_elem_4++) {
                                restore_qd_values_2[restore_elem_4] = restore_qd_values_2[restore_elem_4] * qk_factor_2[restore_elem_4];
                                restore_kd_values_2[restore_elem_4] = restore_kd_values_2[restore_elem_4] * qk_factor_2[restore_elem_4];
                                restore_kr_values_2[restore_elem_4] = restore_ki_values_2[restore_elem_4] * kr_factor_2[restore_elem_4];
                            }
                            #pragma unroll
                            for (int restore_elem_5 = 0; restore_elem_5 < 8; restore_elem_5++) {
                                restore_qd_values_2[restore_elem_5] = restore_qd_values_2[restore_elem_5] * qk_tail_2[restore_elem_5];
                                restore_kd_values_2[restore_elem_5] = restore_kd_values_2[restore_elem_5] * qk_tail_2[restore_elem_5];
                                restore_kr_values_2[restore_elem_5] = restore_kr_values_2[restore_elem_5] * kr_tail_2[restore_elem_5];
                            }
                            unsigned int packed_2_3[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(restore_qd_values_2[_lp*2 + 0], restore_qd_values_2[_lp*2+1 + 0]));
                                packed_2_3[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int word_9 = 0; word_9 < 4; word_9++) {
                                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_qd_addr + prep_stage * 41984 + (unsigned int)(restore_segment_2 * 8 / 64 * 4096 + restore_row_2 * 128 + restore_segment_2 * 8 % 64 * 2 ^ (restore_segment_2 * 8 / 64 * 4096 + restore_row_2 * 128 + restore_segment_2 * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_9 * 4)), "r"((packed_2_3[word_9])));
                            }
                            unsigned int packed_3_3[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(restore_kd_values_2[_lp*2 + 0], restore_kd_values_2[_lp*2+1 + 0]));
                                packed_3_3[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int word_10 = 0; word_10 < 4; word_10++) {
                                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_kd_addr + prep_stage * 41984 + (unsigned int)(restore_segment_2 * 8 / 64 * 4096 + restore_row_2 * 128 + restore_segment_2 * 8 % 64 * 2 ^ (restore_segment_2 * 8 / 64 * 4096 + restore_row_2 * 128 + restore_segment_2 * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_10 * 4)), "r"((packed_3_3[word_10])));
                            }
                            unsigned int packed_4_3[4];
                            #pragma unroll
                            for (int _lp = 0; _lp < 4; _lp++) {
                                __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(restore_kr_values_2[_lp*2 + 0], restore_kr_values_2[_lp*2+1 + 0]));
                                packed_4_3[_lp] = *(uint32_t*)&_bf2;
                            }
                            #pragma unroll
                            for (int word_11 = 0; word_11 < 4; word_11++) {
                                asm volatile("st.shared.b32 [%0], %1;" :: "r"((smem_kr_trans_addr + prep_stage * 41984 + (unsigned int)(restore_segment_2 * 8 / 64 * 4096 + restore_row_2 * 128 + restore_segment_2 * 8 % 64 * 2 ^ (restore_segment_2 * 8 / 64 * 4096 + restore_row_2 * 128 + restore_segment_2 * 8 % 64 * 2 >> 7 & 7) << 4)) + (unsigned int)(word_11 * 4)), "r"((packed_4_3[word_11])));
                            }
                        }
                    }
                }
                if (prep_local_warp == 0) {
                    int inverse_row = lane;
                    int diag_block = inverse_row / 8;
                    int lane_in_diag = lane & 7;
                    float inv_row[8];
                    unsigned int packed_7[4];
                    int byte_off_1 = (int)prep_stage * 41984 + inverse_row * 128 + diag_block * 8 * 2;
                    int swizzled_off_1 = byte_off_1 ^ (byte_off_1 >> 7 & 7) << 4;
                    asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];"
                        : "=r"(*reinterpret_cast<uint32_t*>(&packed_7[0])), "=r"(*reinterpret_cast<uint32_t*>(&packed_7[(0) + 1])), "=r"(*reinterpret_cast<uint32_t*>(&packed_7[(0) + 2])), "=r"(*reinterpret_cast<uint32_t*>(&packed_7[(0) + 3]))
                        : "r"(smem_inv_work_addr + (unsigned int)swizzled_off_1));
                    float packed_f32_4[8];
                    #pragma unroll
                    for (int _pair = 0; _pair < 4; _pair++) {
                        asm volatile(
                            "{\n\t"
                            "shl.b32 %0, %2, 16;\n\t"
                            "and.b32 %1, %2, 0xffff0000;\n\t"
                            "}\n"
                            : "=f"((&packed_f32_4[_pair * 2])[0]), "=f"((&packed_f32_4[_pair * 2])[1])
                            : "r"(packed_7[_pair]));
                    }
                    #pragma unroll
                    for (int value_idx_11 = 0; value_idx_11 < 8; value_idx_11++) {
                        inv_row[value_idx_11] = packed_f32_4[value_idx_11];
                    }
                    #pragma unroll
                    for (int diag_elem = 0; diag_elem < 8; diag_elem++) {
                        if (lane_in_diag == diag_elem) {
                            inv_row[diag_elem] = 1.0f;
                        }
                    }
                    int diag_group_base = lane - lane_in_diag;
                    #pragma unroll
                    for (int src_row = 0; src_row < 7; src_row++) {
                        float row_scale = -inv_row[src_row];
                        #pragma unroll
                        for (int prev_col = 0; prev_col < src_row; prev_col++) {
                            int pivot_lane = diag_group_base + src_row;
                            float _shfl_0;
                            asm volatile("shfl.sync.idx.b32 %0, %1, %2, 0x1f, 0xffffffff;" : "=f"(_shfl_0) : "f"(inv_row[prev_col]), "r"(pivot_lane));
                            float pivot = _shfl_0;
                            if (lane_in_diag > src_row) {
                                float _fma_12 = __fmaf_rn(row_scale, pivot, inv_row[prev_col]);
                                inv_row[prev_col] = _fma_12;
                            }
                        }
                        if (lane_in_diag > src_row) {
                            inv_row[src_row] = row_scale;
                        }
                    }
                    unsigned int packed_0_4[4];
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(inv_row[_lp*2 + 0], inv_row[_lp*2+1 + 0]));
                        packed_0_4[_lp] = *(uint32_t*)&_bf2;
                    }
                    int byte_off_1_1 = (int)prep_stage * 41984 + inverse_row * 128 + diag_block * 8 * 2;
                    int swizzled_off_2 = byte_off_1_1 ^ (byte_off_1_1 >> 7 & 7) << 4;
                    #pragma unroll
                    for (int word_12 = 0; word_12 < 4; word_12++) {
                        asm volatile("st.shared.b32 [%0], %1;" :: "r"(smem_inv_work_addr + (unsigned int)swizzled_off_2 + (unsigned int)(word_12 * 4)), "r"((packed_0_4[word_12])));
                    }
                }
                if (prep_local_warp < 2) {
                    __syncwarp();
                    mbarrier_arrive(prep_diag_ready_addr + (prep_stage) * 8);
                    mbarrier_wait(prep_diag_ready_addr + (prep_stage) * 8, _phase_prep_diag_ready);
                }
                if (prep_local_warp < 2) {
                    int lane_row = lane & 7;
                    int byte_off_2 = (int)prep_stage * 41984 + (prep_local_warp * 16 + 8 + lane_row) * 128 + (prep_local_warp * 16 + 8) * 2;
                    int swizzled_off_3 = byte_off_2 ^ (byte_off_2 >> 7 & 7) << 4;
                    int d_addr = smem_inv_work_addr + (unsigned int)swizzled_off_3;
                    int byte_off_0 = (int)prep_stage * 41984 + (prep_local_warp * 16 + 8 + lane_row) * 128 + prep_local_warp * 16 * 2;
                    int swizzled_off_1_1 = byte_off_0 ^ (byte_off_0 >> 7 & 7) << 4;
                    int c_addr = smem_inv_work_addr + (unsigned int)swizzled_off_1_1;
                    int byte_off_2_1 = (int)prep_stage * 41984 + (prep_local_warp * 16 + lane_row) * 128 + prep_local_warp * 16 * 2;
                    int swizzled_off_3_1 = byte_off_2_1 ^ (byte_off_2_1 >> 7 & 7) << 4;
                    int a_addr = smem_inv_work_addr + (unsigned int)swizzled_off_3_1;
                    unsigned int d_frag[2];
                    unsigned int c_frag[1];
                    float dc_acc[4];
                    unsigned int dc_bf16[2];
                    unsigned int inv_a_frag[1];
                    float o_acc[4];
                    unsigned int o_bf16[2];
                    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                        : "=r"(d_frag[0])
                        : "r"(d_addr)
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x1.shared.b16 {%0}, [%1];\n"
                        : "=r"(d_frag[1])
                        : "r"(d_addr)
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];\n"
                        : "=r"(c_frag[0])
                        : "r"(c_addr)
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
                        : "=f"(dc_acc[0]), "=f"(dc_acc[1]), "=f"(dc_acc[2]), "=f"(dc_acc[3])
                        : "r"(d_frag[0]), "r"(d_frag[1]), "r"(c_frag[0]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    const float2 _scale2_9 = {-1.0f, -1.0f};
                    #pragma unroll
                    for (int _ls = 0; _ls < 2; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(dc_acc)[_ls], _scale2_9);
                    #pragma unroll
                    for (int _lp = 0; _lp < 2; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dc_acc[_lp*2 + 0], dc_acc[_lp*2+1 + 0]));
                        dc_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16 {%0}, [%1];\n"
                        : "=r"(inv_a_frag[0])
                        : "r"(a_addr)
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5}, {%6}, {%7, %8, %9, %10};\n"
                        : "=f"(o_acc[0]), "=f"(o_acc[1]), "=f"(o_acc[2]), "=f"(o_acc[3])
                        : "r"(dc_bf16[0]), "r"(dc_bf16[1]), "r"(inv_a_frag[0]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    #pragma unroll
                    for (int _lp = 0; _lp < 2; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o_acc[_lp*2 + 0], o_acc[_lp*2+1 + 0]));
                        o_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    int byte_off_4 = (int)prep_stage * 41984 + (prep_local_warp * 16 + 8 + lane_row) * 128 + prep_local_warp * 16 * 2;
                    int swizzled_off_5 = byte_off_4 ^ (byte_off_4 >> 7 & 7) << 4;
                    int o_addr = smem_inv_work_addr + (unsigned int)swizzled_off_5;
                    uint32_t _stmatrix_addr_10 = static_cast<uint32_t>((unsigned long long)o_addr);
                    asm volatile("stmatrix.sync.aligned.m8n8.x1.shared.b16 [%0], {%1};\n"
                        :: "r"(_stmatrix_addr_10), "r"(*reinterpret_cast<const uint32_t*>(&o_bf16[0]))
                        : "memory");
                    __syncwarp();
                    mbarrier_arrive(prep_inv16_ready_addr + (prep_stage) * 8);
                    mbarrier_wait(prep_inv16_ready_addr + (prep_stage) * 8, _phase_prep_inv16_ready);
                }
                if (prep_local_warp == 0) {
                    int lane_row_1 = lane % 16;
                    int lane_col = lane / 16 * 8;
                    int byte_off_3 = (int)prep_stage * 41984 + (16 + lane_row_1) * 128 + (16 + lane_col) * 2;
                    int swizzled_off_4 = byte_off_3 ^ (byte_off_3 >> 7 & 7) << 4;
                    int d_addr_1 = smem_inv_work_addr + (unsigned int)swizzled_off_4;
                    int byte_off_0_1 = (int)prep_stage * 41984 + (16 + lane_row_1) * 128 + lane_col * 2;
                    int swizzled_off_1_2 = byte_off_0_1 ^ (byte_off_0_1 >> 7 & 7) << 4;
                    int c_addr_1 = smem_inv_work_addr + (unsigned int)swizzled_off_1_2;
                    int byte_off_2_2 = (int)prep_stage * 41984 + lane_row_1 * 128 + lane_col * 2;
                    int swizzled_off_3_2 = byte_off_2_2 ^ (byte_off_2_2 >> 7 & 7) << 4;
                    int a_addr_1 = smem_inv_work_addr + (unsigned int)swizzled_off_3_2;
                    unsigned int d32_frag[4];
                    unsigned int c32_frag[4];
                    float dc32_acc[8];
                    unsigned int dc32_bf16[4];
                    unsigned int a32_frag[4];
                    float o32_acc[8];
                    unsigned int o32_bf16[4];
                    unsigned int zero32_bf16[4];
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(d32_frag[0]), "=r"(d32_frag[1]), "=r"(d32_frag[2]), "=r"(d32_frag[3])
                        : "r"(d_addr_1)
                        : "memory");
                    int d_publish_addr = (smem_inv_addr + prep_stage * 41984 + (unsigned int)((16 + lane_col) / 16 * 1024 + (16 + lane_row_1) * 32 + (16 + lane_col) % 16 * 2 ^ ((16 + lane_col) / 16 * 1024 + (16 + lane_row_1) * 32 + (16 + lane_col) % 16 * 2 >> 7 & 1) << 4));
                    uint32_t _stmatrix_addr_11 = static_cast<uint32_t>((unsigned long long)d_publish_addr);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_11), "r"(*reinterpret_cast<const uint32_t*>(&d32_frag[0])), "r"(*reinterpret_cast<const uint32_t*>(&d32_frag[1])), "r"(*reinterpret_cast<const uint32_t*>(&d32_frag[2])), "r"(*reinterpret_cast<const uint32_t*>(&d32_frag[3]))
                        : "memory");
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(c32_frag[0]), "=r"(c32_frag[1]), "=r"(c32_frag[2]), "=r"(c32_frag[3])
                        : "r"(c_addr_1)
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(dc32_acc[0]), "=f"(dc32_acc[1]), "=f"(dc32_acc[2]), "=f"(dc32_acc[3])
                        : "r"(d32_frag[0]), "r"(d32_frag[1]), "r"(d32_frag[2]), "r"(d32_frag[3]), "r"(c32_frag[0]), "r"(c32_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(dc32_acc[4]), "=f"(dc32_acc[(4) + 1]), "=f"(dc32_acc[(4) + 2]), "=f"(dc32_acc[(4) + 3])
                        : "r"(d32_frag[0]), "r"(d32_frag[1]), "r"(d32_frag[2]), "r"(d32_frag[3]), "r"(c32_frag[2]), "r"(c32_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    const float2 _scale2_12 = {-1.0f, -1.0f};
                    #pragma unroll
                    for (int _ls = 0; _ls < 4; _ls++)
                        mul_f32x2_inplace(&reinterpret_cast<float2*>(dc32_acc)[_ls], _scale2_12);
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(dc32_acc[_lp*2 + 0], dc32_acc[_lp*2+1 + 0]));
                        dc32_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                        : "=r"(a32_frag[0]), "=r"(a32_frag[1]), "=r"(a32_frag[2]), "=r"(a32_frag[3])
                        : "r"(a_addr_1)
                        : "memory");
                    int a_publish_addr = (smem_inv_addr + prep_stage * 41984 + (unsigned int)(lane_col / 16 * 1024 + lane_row_1 * 32 + lane_col % 16 * 2 ^ (lane_col / 16 * 1024 + lane_row_1 * 32 + lane_col % 16 * 2 >> 7 & 1) << 4));
                    uint32_t _stmatrix_addr_13 = static_cast<uint32_t>((unsigned long long)a_publish_addr);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_13), "r"(*reinterpret_cast<const uint32_t*>(&a32_frag[0])), "r"(*reinterpret_cast<const uint32_t*>(&a32_frag[1])), "r"(*reinterpret_cast<const uint32_t*>(&a32_frag[2])), "r"(*reinterpret_cast<const uint32_t*>(&a32_frag[3]))
                        : "memory");
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(o32_acc[0]), "=f"(o32_acc[1]), "=f"(o32_acc[2]), "=f"(o32_acc[3])
                        : "r"(dc32_bf16[0]), "r"(dc32_bf16[1]), "r"(dc32_bf16[2]), "r"(dc32_bf16[3]), "r"(a32_frag[0]), "r"(a32_frag[1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                        : "=f"(o32_acc[4]), "=f"(o32_acc[(4) + 1]), "=f"(o32_acc[(4) + 2]), "=f"(o32_acc[(4) + 3])
                        : "r"(dc32_bf16[0]), "r"(dc32_bf16[1]), "r"(dc32_bf16[2]), "r"(dc32_bf16[3]), "r"(a32_frag[2]), "r"(a32_frag[(2) + 1]), "f"(0.0f), "f"(0.0f), "f"(0.0f), "f"(0.0f));
                    #pragma unroll
                    for (int _lp = 0; _lp < 4; _lp++) {
                        __nv_bfloat162 _bf2 = __float22bfloat162_rn(make_float2(o32_acc[_lp*2 + 0], o32_acc[_lp*2+1 + 0]));
                        o32_bf16[_lp] = *(uint32_t*)&_bf2;
                    }
                    int o_publish_addr = (smem_inv_addr + prep_stage * 41984 + (unsigned int)(lane_col / 16 * 1024 + (16 + lane_row_1) * 32 + lane_col % 16 * 2 ^ (lane_col / 16 * 1024 + (16 + lane_row_1) * 32 + lane_col % 16 * 2 >> 7 & 1) << 4));
                    uint32_t _stmatrix_addr_14 = static_cast<uint32_t>((unsigned long long)o_publish_addr);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_14), "r"(*reinterpret_cast<const uint32_t*>(&o32_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&o32_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&o32_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&o32_bf16[3]))
                        : "memory");
                    #pragma unroll
                    for (int zero_word = 0; zero_word < 4; zero_word++) {
                        zero32_bf16[zero_word] = 0;
                    }
                    int zero_publish_addr = (smem_inv_addr + prep_stage * 41984 + (unsigned int)((16 + lane_col) / 16 * 1024 + lane_row_1 * 32 + (16 + lane_col) % 16 * 2 ^ ((16 + lane_col) / 16 * 1024 + lane_row_1 * 32 + (16 + lane_col) % 16 * 2 >> 7 & 1) << 4));
                    uint32_t _stmatrix_addr_15 = static_cast<uint32_t>((unsigned long long)zero_publish_addr);
                    asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n"
                        :: "r"(_stmatrix_addr_15), "r"(*reinterpret_cast<const uint32_t*>(&zero32_bf16[0])), "r"(*reinterpret_cast<const uint32_t*>(&zero32_bf16[1])), "r"(*reinterpret_cast<const uint32_t*>(&zero32_bf16[2])), "r"(*reinterpret_cast<const uint32_t*>(&zero32_bf16[3]))
                        : "memory");
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync %0, 128;" :: "r"(10 + prep_instance) : "memory");
                mbarrier_arrive(qk_full_addr + (prep_stage) * 8);
                if (chunk_is_full_1 != 0) {
                    if (prep_local_warp == 0) {
                        if (elect_sync()) {
                            mbarrier_arrive_expect_tx(v_full_addr + (prep_stage) * 8, 8192);
                            {
                                tma_3d_gmem2smem(smem_v_addr + prep_stage * 41984, v_tma, 0, head_idx_3, (int)(bos_3 + (long long)(chunk_idx_2 * 32)), v_full_addr + (prep_stage) * 8);
                            }
                        }
                    }
                } else {
                    #pragma unroll
                    for (int v_load_iter = 0; v_load_iter < 4; v_load_iter++) {
                        int v_item = v_load_iter * 128 + prep_tid;
                        int row_1 = v_item / 16;
                        int segment_1 = v_item % 16;
                        long long token_1 = bos_3 + (long long)(chunk_idx_2 * 32 + row_1);
                        int token_valid_1 = ((token_1 < eos_3) ? 1 : 0);
                        long long v_src = (token_1 * (long long)num_heads + (long long)head_idx_3) * 128 + (long long)(segment_1 * 8);
                        int v_dst = smem_v_addr + prep_stage * 41984 + (unsigned int)((row_1 * 128 + segment_1 * 8) * 2);
                        asm volatile("cp.async.cg.shared::cta.global [%0], [%1], 16, %2;"
                            :: "r"(v_dst), "l"(v + v_src), "r"((token_valid_1 != 0) ? 16 : 0));
                    }
                    asm volatile("cp.async.commit_group;");
                    asm volatile("cp.async.wait_group 0;");
                    asm volatile("barrier.sync %0, 128;" :: "r"(10 + prep_instance) : "memory");
                    if (prep_local_warp == 0) {
                        if (elect_sync()) {
                            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                            mbarrier_arrive(v_full_addr + (prep_stage) * 8);
                        }
                    }
                }
                for (int _advance = 0; _advance < 5; _advance++) {
                    prep_stage += 1;
                    if (prep_stage == 5) { prep_stage = 0; _phase_raw_inputs_free ^= 1; _phase_gate_raw_full ^= 1; _phase_smem_free ^= 1; _phase_v_free ^= 1; _phase_qk_raw_full ^= 1; _phase_short_beta_ready ^= 1; _phase_prep_diag_ready ^= 1; _phase_prep_inv16_ready ^= 1; }
                }
            }
        }
    }

    // Cleanup
}

} // extern "C"
