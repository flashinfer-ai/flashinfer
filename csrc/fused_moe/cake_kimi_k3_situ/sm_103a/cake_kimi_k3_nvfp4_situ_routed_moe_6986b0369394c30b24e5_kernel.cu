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
#define TMEM_NCOLS 280
#define TMEM_ACCUM_OFFSET 0
#define TMEM_SFA_OFFSET 64
#define TMEM_SFB_OFFSET 208
#define NUM_K_PIPE_STAGES 9
#define NUM_MMA_PIPE_STAGES 2
#define NUM_WORK_PIPE_STAGES 3
#define NUM_THROTTLE_PIPE_STAGES 3
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 16384
#define SMEM_SMEM_B_OFF 148480
#define SMEM_SMEM_B_STAGE_BYTES 4096
#define SMEM_SMEM_B_STRIDE 4096
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
#define SMEM_SMEM_B_MMA0_OFF 148480
#define SMEM_SMEM_B_MMA0_STAGE_BYTES 1024
#define SMEM_SMEM_B_MMA0_STRIDE 4096
#define SMEM_SMEM_B_MMA1_OFF 148512
#define SMEM_SMEM_B_MMA1_STAGE_BYTES 1024
#define SMEM_SMEM_B_MMA1_STRIDE 4096
#define SMEM_SMEM_B_MMA2_OFF 148544
#define SMEM_SMEM_B_MMA2_STAGE_BYTES 1024
#define SMEM_SMEM_B_MMA2_STRIDE 4096
#define SMEM_SMEM_B_MMA3_OFF 148576
#define SMEM_SMEM_B_MMA3_STAGE_BYTES 1024
#define SMEM_SMEM_B_MMA3_STRIDE 4096
#define SMEM_SMEM_B_MMA4_OFF 148480
#define SMEM_SMEM_B_MMA4_STAGE_BYTES 1024
#define SMEM_SMEM_B_MMA4_STRIDE 4096
#define SMEM_SMEM_B_MMA5_OFF 148480
#define SMEM_SMEM_B_MMA5_STAGE_BYTES 1024
#define SMEM_SMEM_B_MMA5_STRIDE 4096
#define SMEM_SMEM_B_MMA6_OFF 148480
#define SMEM_SMEM_B_MMA6_STAGE_BYTES 1024
#define SMEM_SMEM_B_MMA6_STRIDE 4096
#define SMEM_SMEM_B_MMA7_OFF 148480
#define SMEM_SMEM_B_MMA7_STAGE_BYTES 1024
#define SMEM_SMEM_B_MMA7_STRIDE 4096
#define SMEM_SMEM_A_MMA0_NEXT_OFF 1024
#define SMEM_SMEM_A_MMA0_NEXT_STAGE_BYTES 4096
#define SMEM_SMEM_A_MMA0_NEXT_STRIDE 16384
#define SMEM_SMEM_B_MMA0_NEXT_OFF 148480
#define SMEM_SMEM_B_MMA0_NEXT_STAGE_BYTES 1024
#define SMEM_SMEM_B_MMA0_NEXT_STRIDE 4096
#define SMEM_SMEM_A_MMA2_NEXT_OFF 1024
#define SMEM_SMEM_A_MMA2_NEXT_STAGE_BYTES 4096
#define SMEM_SMEM_A_MMA2_NEXT_STRIDE 16384
#define SMEM_SMEM_B_MMA2_NEXT_OFF 148480
#define SMEM_SMEM_B_MMA2_NEXT_STAGE_BYTES 1024
#define SMEM_SMEM_B_MMA2_NEXT_STRIDE 4096
#define SMEM_SMEM_A_MMA5_NEXT_OFF 1024
#define SMEM_SMEM_A_MMA5_NEXT_STAGE_BYTES 4096
#define SMEM_SMEM_A_MMA5_NEXT_STRIDE 16384
#define SMEM_SMEM_B_MMA5_NEXT_OFF 148480
#define SMEM_SMEM_B_MMA5_NEXT_STAGE_BYTES 1024
#define SMEM_SMEM_B_MMA5_NEXT_STRIDE 4096
#define SMEM_SMEM_SFA_OFF 185344
#define SMEM_SMEM_SFA_STAGE_BYTES 2048
#define SMEM_SMEM_SFA_STRIDE 2048
#define SMEM_SMEM_SFB_OFF 203776
#define SMEM_SMEM_SFB_STAGE_BYTES 512
#define SMEM_SMEM_SFB_STRIDE 512
#define SMEM_EPI_STAGING_OFF 208384
#define SMEM_EPI_STAGING_STAGE_BYTES 8192
#define SMEM_EPI_STAGING_STRIDE 8192
#define SMEM_EPI_STAGING_U64_OFF 208384
#define SMEM_EPI_STAGING_U64_STAGE_BYTES 8192
#define SMEM_EPI_STAGING_U64_STRIDE 8192
#define SMEM_WORK_RESPONSE_OFF 216576
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_TOTAL 216704
#define THREADS 512
#define BLOCK_M 128
#define BLOCK_N 32
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


__device__ __forceinline__ void tmem_ld_x4(float* dst, int tmem_addr) {
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x4.b32"
        " {%0, %1, %2, %3}, [%4];"
        : "=f"(dst[0]), "=f"(dst[1]), "=f"(dst[2]), "=f"(dst[3])
        : "r"(tmem_addr));
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

__global__ __launch_bounds__(512, 1) void
kernel_cake_kimi_k3_nvfp4_situ_routed_moe_6986b0369394c30b24e5(const __grid_constant__ CUtensorMap A, const __grid_constant__ CUtensorMap B, const __grid_constant__ CUtensorMap SFA, const __grid_constant__ CUtensorMap SFB, const __grid_constant__ CUtensorMap C_tma, __nv_bfloat16* __restrict__ C, float* __restrict__ scale_c, int* __restrict__ tile_expert, int* __restrict__ tile_mn_limit, int M, int K, int grid_m, int grid_n, int K_tiles, int* __restrict__ total_tiles)
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
    #define b_full_addr (mbar_base + 72)
    #define sfa_full_addr (mbar_base + 144)
    #define sfb_full_addr (mbar_base + 216)
    #define sfa_smem_free_addr (mbar_base + 288)
    #define sfb_smem_free_addr (mbar_base + 360)
    #define tmem_sfa_full_addr (mbar_base + 432)
    #define tmem_sfb_full_addr (mbar_base + 504)
    #define k_done_addr (mbar_base + 576)
    #define mma_full_addr (mbar_base + 648)
    #define mma_free_addr (mbar_base + 664)
    #define work_full_addr (mbar_base + 680)
    #define work_empty_addr (mbar_base + 704)
    #define throttle_full_addr (mbar_base + 728)
    #define throttle_empty_addr (mbar_base + 752)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 776);

    // Kernel setup ops
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + 148480);
    const int smem_b_addr = smem + 148480;
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
    uint8_t* smem_b_mma0 = reinterpret_cast<uint8_t*>(smem_raw + 148480);
    const int smem_b_mma0_addr = smem + 148480;
    uint8_t* smem_b_mma1 = reinterpret_cast<uint8_t*>(smem_raw + 148512);
    const int smem_b_mma1_addr = smem + 148512;
    uint8_t* smem_b_mma2 = reinterpret_cast<uint8_t*>(smem_raw + 148544);
    const int smem_b_mma2_addr = smem + 148544;
    uint8_t* smem_b_mma3 = reinterpret_cast<uint8_t*>(smem_raw + 148576);
    const int smem_b_mma3_addr = smem + 148576;
    uint8_t* smem_b_mma4 = reinterpret_cast<uint8_t*>(smem_raw + 148480);
    const int smem_b_mma4_addr = smem + 148480;
    uint8_t* smem_b_mma5 = reinterpret_cast<uint8_t*>(smem_raw + 148480);
    const int smem_b_mma5_addr = smem + 148480;
    uint8_t* smem_b_mma6 = reinterpret_cast<uint8_t*>(smem_raw + 148480);
    const int smem_b_mma6_addr = smem + 148480;
    uint8_t* smem_b_mma7 = reinterpret_cast<uint8_t*>(smem_raw + 148480);
    const int smem_b_mma7_addr = smem + 148480;
    uint8_t* smem_a_mma0_next = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_mma0_next_addr = smem + 1024;
    uint8_t* smem_b_mma0_next = reinterpret_cast<uint8_t*>(smem_raw + 148480);
    const int smem_b_mma0_next_addr = smem + 148480;
    uint8_t* smem_a_mma2_next = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_mma2_next_addr = smem + 1024;
    uint8_t* smem_b_mma2_next = reinterpret_cast<uint8_t*>(smem_raw + 148480);
    const int smem_b_mma2_next_addr = smem + 148480;
    uint8_t* smem_a_mma5_next = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_mma5_next_addr = smem + 1024;
    uint8_t* smem_b_mma5_next = reinterpret_cast<uint8_t*>(smem_raw + 148480);
    const int smem_b_mma5_next_addr = smem + 148480;
    uint8_t* smem_sfa = reinterpret_cast<uint8_t*>(smem_raw + 185344);
    const int smem_sfa_addr = smem + 185344;
    uint8_t* smem_sfb = reinterpret_cast<uint8_t*>(smem_raw + 203776);
    const int smem_sfb_addr = smem + 203776;
    __nv_bfloat16* epi_staging = reinterpret_cast<__nv_bfloat16*>(smem_raw + 208384);
    const int epi_staging_addr = smem + 208384;
    unsigned long long* epi_staging_u64 = reinterpret_cast<unsigned long long*>(smem_raw + 208384);
    const int epi_staging_u64_addr = smem + 208384;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 216576);
    const int work_response_addr = smem + 216576;
    asm volatile("griddepcontrol.wait;" ::: "memory");
    if ((int)blockIdx.y >= total_tiles[0]) return;

    // Mbarrier init (15 pipeline groups, 0 ordered-sequence groups, 97 barriers)
    // Mbarriers at smem_raw[0..776)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'k_pipe' ---
            // a_full: 9 barriers, init_count=1
            mbarrier_init(smem + 0, 1);
            mbarrier_init(smem + 8, 1);
            mbarrier_init(smem + 16, 1);
            mbarrier_init(smem + 24, 1);
            mbarrier_init(smem + 32, 1);
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            // b_full: 9 barriers, init_count=1
            mbarrier_init(smem + 72, 1);
            mbarrier_init(smem + 80, 1);
            mbarrier_init(smem + 88, 1);
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            mbarrier_init(smem + 128, 1);
            mbarrier_init(smem + 136, 1);
            // sfa_full: 9 barriers, init_count=1
            mbarrier_init(smem + 144, 1);
            mbarrier_init(smem + 152, 1);
            mbarrier_init(smem + 160, 1);
            mbarrier_init(smem + 168, 1);
            mbarrier_init(smem + 176, 1);
            mbarrier_init(smem + 184, 1);
            mbarrier_init(smem + 192, 1);
            mbarrier_init(smem + 200, 1);
            mbarrier_init(smem + 208, 1);
            // sfb_full: 9 barriers, init_count=1
            mbarrier_init(smem + 216, 1);
            mbarrier_init(smem + 224, 1);
            mbarrier_init(smem + 232, 1);
            mbarrier_init(smem + 240, 1);
            mbarrier_init(smem + 248, 1);
            mbarrier_init(smem + 256, 1);
            mbarrier_init(smem + 264, 1);
            mbarrier_init(smem + 272, 1);
            mbarrier_init(smem + 280, 1);
            // sfa_smem_free: 9 barriers, init_count=1
            mbarrier_init(smem + 288, 1);
            mbarrier_init(smem + 296, 1);
            mbarrier_init(smem + 304, 1);
            mbarrier_init(smem + 312, 1);
            mbarrier_init(smem + 320, 1);
            mbarrier_init(smem + 328, 1);
            mbarrier_init(smem + 336, 1);
            mbarrier_init(smem + 344, 1);
            mbarrier_init(smem + 352, 1);
            // sfb_smem_free: 9 barriers, init_count=1
            mbarrier_init(smem + 360, 1);
            mbarrier_init(smem + 368, 1);
            mbarrier_init(smem + 376, 1);
            mbarrier_init(smem + 384, 1);
            mbarrier_init(smem + 392, 1);
            mbarrier_init(smem + 400, 1);
            mbarrier_init(smem + 408, 1);
            mbarrier_init(smem + 416, 1);
            mbarrier_init(smem + 424, 1);
            // tmem_sfa_full: 9 barriers, init_count=1
            mbarrier_init(smem + 432, 1);
            mbarrier_init(smem + 440, 1);
            mbarrier_init(smem + 448, 1);
            mbarrier_init(smem + 456, 1);
            mbarrier_init(smem + 464, 1);
            mbarrier_init(smem + 472, 1);
            mbarrier_init(smem + 480, 1);
            mbarrier_init(smem + 488, 1);
            mbarrier_init(smem + 496, 1);
            // tmem_sfb_full: 9 barriers, init_count=1
            mbarrier_init(smem + 504, 1);
            mbarrier_init(smem + 512, 1);
            mbarrier_init(smem + 520, 1);
            mbarrier_init(smem + 528, 1);
            mbarrier_init(smem + 536, 1);
            mbarrier_init(smem + 544, 1);
            mbarrier_init(smem + 552, 1);
            mbarrier_init(smem + 560, 1);
            mbarrier_init(smem + 568, 1);
            // k_done: 9 barriers, init_count=1
            mbarrier_init(smem + 576, 1);
            mbarrier_init(smem + 584, 1);
            mbarrier_init(smem + 592, 1);
            mbarrier_init(smem + 600, 1);
            mbarrier_init(smem + 608, 1);
            mbarrier_init(smem + 616, 1);
            mbarrier_init(smem + 624, 1);
            mbarrier_init(smem + 632, 1);
            mbarrier_init(smem + 640, 1);
            // --- pipeline 'mma_pipe' ---
            // mma_full: 2 barriers, init_count=1
            mbarrier_init(smem + 648, 1);
            mbarrier_init(smem + 656, 1);
            // mma_free: 2 barriers, init_count=4
            mbarrier_init(smem + 664, 4);
            mbarrier_init(smem + 672, 4);
            // --- pipeline 'work_pipe' ---
            // work_full: 3 barriers, init_count=1
            mbarrier_init(smem + 680, 1);
            mbarrier_init(smem + 688, 1);
            mbarrier_init(smem + 696, 1);
            // work_empty: 3 barriers, init_count=512
            mbarrier_init(smem + 704, 512);
            mbarrier_init(smem + 712, 512);
            mbarrier_init(smem + 720, 512);
            // --- pipeline 'throttle_pipe' ---
            // throttle_full: 3 barriers, init_count=32
            mbarrier_init(smem + 728, 32);
            mbarrier_init(smem + 736, 32);
            mbarrier_init(smem + 744, 32);
            // throttle_empty: 3 barriers, init_count=32
            mbarrier_init(smem + 752, 32);
            mbarrier_init(smem + 760, 32);
            mbarrier_init(smem + 768, 32);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    // TMEM alloc (512 columns, 280 used)
    if (warp == 0) {
        int _tmem_hold = smem + 776;
        asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"(_tmem_hold), "r"(512) : "memory");
        asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
    }

    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;");

    const int taddr = tmem_addr_storage[0];

    // Kernel post-init ops
    const int tmem_accum = taddr;
    const int tmem_sfa = taddr + 64;
    const int tmem_sfb = taddr + 208;

    // ---- Ordered hardware-WG register redistribution ----
    // Dec phase frees registers before any WG attempts inc.
    if (warp >= 8 && warp <= 15) {
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
                        : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                        : "memory");
                    valid = _clc_valid_14;
                    uint32_t _clc_ctaid_28 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_28)
                        : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                        : "memory");
                    next_x = _clc_ctaid_28;
                    uint32_t _clc_ctaid_29 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_29)
                        : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                        : "memory");
                    next_y = _clc_ctaid_29;
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
                {
                    float _tmem_load_2[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_2[0]), "=f"(_tmem_load_2[1]), "=f"(_tmem_load_2[2]), "=f"(_tmem_load_2[3]), "=f"(_tmem_load_2[4]), "=f"(_tmem_load_2[5]), "=f"(_tmem_load_2[6]), "=f"(_tmem_load_2[7]), "=f"(_tmem_load_2[8]), "=f"(_tmem_load_2[9]), "=f"(_tmem_load_2[10]), "=f"(_tmem_load_2[11]), "=f"(_tmem_load_2[12]), "=f"(_tmem_load_2[13]), "=f"(_tmem_load_2[14]), "=f"(_tmem_load_2[15]), "=f"(_tmem_load_2[16]), "=f"(_tmem_load_2[17]), "=f"(_tmem_load_2[18]), "=f"(_tmem_load_2[19]), "=f"(_tmem_load_2[20]), "=f"(_tmem_load_2[21]), "=f"(_tmem_load_2[22]), "=f"(_tmem_load_2[23]), "=f"(_tmem_load_2[24]), "=f"(_tmem_load_2[25]), "=f"(_tmem_load_2[26]), "=f"(_tmem_load_2[27]), "=f"(_tmem_load_2[28]), "=f"(_tmem_load_2[29]), "=f"(_tmem_load_2[30]), "=f"(_tmem_load_2[31])
                        : "r"(taddr + (unsigned int)(row << 16) + acc_stage * 32));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    asm volatile("cp.async.bulk.wait_group.read 0;");
                    asm volatile("barrier.sync 8, 128;" ::: "memory");
                    #pragma unroll
                    for (int token = 0; token < 32; token++) {
                        __nv_bfloat16 _cvt_bf16_0 = __float2bfloat16(_tmem_load_2[token] * output_scale);
                        bf = _cvt_bf16_0;
                        if (feature < 64) {
                            epi_staging[token * 64 + feature] = bf;
                        } else {
                            epi_staging[2048 + token * 64 + feature - 64] = bf;
                        }
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync 8, 128;" ::: "memory");
                if (warp == 0) {
                    if (elect_sync()) {
                        int padding_rows = (32 - _mn_limit % 32) % 32;
                        int local_token = padding_rows;
                        tma_store_4d((&C_tma), off_m, local_token, 1073741824, token_base - padding_rows + 1073741824, epi_staging_addr);
                        tma_store_4d((&C_tma), off_m + 64, local_token, 1073741824, token_base - padding_rows + 1073741824, epi_staging_addr + 4096);
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
                    : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                    : "memory");
                valid_1 = _clc_valid_15;
                uint32_t _clc_ctaid_30 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_30)
                    : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                    : "memory");
                next_x_1 = _clc_ctaid_30;
                uint32_t _clc_ctaid_31 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_31)
                    : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                    : "memory");
                next_y_1 = _clc_ctaid_31;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage) * 8);
                work_stage += 1;
                if (work_stage == 3) { work_stage = 0; _phase_work_full ^= 1; }
                unsigned int valid_0 = valid_1;
                m_tile = next_x_1;
                n_tile = next_y_1;
                if (valid_0 == 0) {
                    break;
                }
            }
        }
    }
    // ---- Role: copy_sfb ----
    if (warp >= 4 && warp <= 7) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 72;");
        { // copy_sfb_main
            unsigned int stage = 0;
            unsigned int work_stage_1 = 0;
            unsigned int m_tile_1 = blockIdx.x;
            unsigned int n_tile_1 = blockIdx.y;
            const int lane_0 = lane;
            unsigned int word[1];
            unsigned int _phase_work_full_1 = 0;
            unsigned int _phase_sfb_full = 0;
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
                        : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    valid_2 = _clc_valid_10;
                    uint32_t _clc_ctaid_20 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_20)
                        : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    next_x_2 = _clc_ctaid_20;
                    uint32_t _clc_ctaid_21 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_21)
                        : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    next_y_2 = _clc_ctaid_21;
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
                for (int _route_k = 0; _route_k < route_k_extent; _route_k++) {
                    mbarrier_wait(sfb_full_addr + (stage) * 8, _phase_sfb_full);
                    mbarrier_wait(k_done_addr + (stage) * 8, _phase_k_done);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    const int q = 0;
                    for (int reg = 0; reg < 1; reg++) {
                        word[reg] = 0;
                        if (lane_0 + reg * 32 < 32) {
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&word[reg])) : "r"(smem_sfb_addr + stage * 512 + (unsigned int)((reg * 4 + lane_0 / 8) * 128) + (unsigned int)(q * 32) + (unsigned int)(lane_0 % 8 * 4)));
                        }
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(taddr + 64 + 144 + stage * 8), "r"(word[0]));
                    const int q_0 = 1;
                    for (int reg_1 = 0; reg_1 < 1; reg_1++) {
                        word[reg_1] = 0;
                        if (lane_0 + reg_1 * 32 < 32) {
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&word[reg_1])) : "r"(smem_sfb_addr + stage * 512 + (unsigned int)((reg_1 * 4 + lane_0 / 8) * 128) + (unsigned int)(q_0 * 32) + (unsigned int)(lane_0 % 8 * 4)));
                        }
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(taddr + 64 + 144 + stage * 8 + 2), "r"(word[0]));
                    const int q_1 = 2;
                    for (int reg_2 = 0; reg_2 < 1; reg_2++) {
                        word[reg_2] = 0;
                        if (lane_0 + reg_2 * 32 < 32) {
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&word[reg_2])) : "r"(smem_sfb_addr + stage * 512 + (unsigned int)((reg_2 * 4 + lane_0 / 8) * 128) + (unsigned int)(q_1 * 32) + (unsigned int)(lane_0 % 8 * 4)));
                        }
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(taddr + 64 + 144 + stage * 8 + 4), "r"(word[0]));
                    const int q_2 = 3;
                    for (int reg_3 = 0; reg_3 < 1; reg_3++) {
                        word[reg_3] = 0;
                        if (lane_0 + reg_3 * 32 < 32) {
                            asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<uint32_t*>(&word[reg_3])) : "r"(smem_sfb_addr + stage * 512 + (unsigned int)((reg_3 * 4 + lane_0 / 8) * 128) + (unsigned int)(q_2 * 32) + (unsigned int)(lane_0 % 8 * 4)));
                        }
                    }
                    asm volatile(
                        "tcgen05.st.sync.aligned.32x32b.x1.b32"
                        " [%0], {%1};"
                        :: "r"(taddr + 64 + 144 + stage * 8 + 6), "r"(word[0]));
                    asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
                    asm volatile("tcgen05.fence::before_thread_sync;");
                    asm volatile("barrier.sync 4, 128;" ::: "memory");
                    if (warp == 4) {
                        if (elect_sync()) {
                            mbarrier_arrive(tmem_sfb_full_addr + (stage) * 8);
                            mbarrier_arrive(sfb_smem_free_addr + (stage) * 8);
                        }
                    }
                    stage += 1;
                    if (stage == 9) { stage = 0; _phase_sfb_full ^= 1; _phase_k_done ^= 1; }
                }
                {
                    mbarrier_wait_hint(work_full_addr + (work_stage_1) * 8, _phase_work_full_1, 10000000);
                }
                unsigned int valid_3 = 0;
                unsigned int next_x_3 = 0;
                unsigned int next_y_3 = 0;
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
                    : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                    : "memory");
                valid_3 = _clc_valid_11;
                uint32_t _clc_ctaid_22 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_22)
                    : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                    : "memory");
                next_x_3 = _clc_ctaid_22;
                uint32_t _clc_ctaid_23 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_23)
                    : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                    : "memory");
                next_y_3 = _clc_ctaid_23;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage_1) * 8);
                work_stage_1 += 1;
                if (work_stage_1 == 3) { work_stage_1 = 0; _phase_work_full_1 ^= 1; }
                unsigned int valid_0_1 = valid_3;
                m_tile_1 = next_x_3;
                n_tile_1 = next_y_3;
                if (valid_0_1 == 0) {
                    break;
                }
            }
        }
    }
    // ---- Role: load_b ----
    if (warp == 8) {
        { // load_b_main
            unsigned int stage_1 = 0;
            unsigned int work_stage_2 = 0;
            unsigned int m_tile_2 = blockIdx.x;
            unsigned int n_tile_2 = blockIdx.y;
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int _phase_work_full_2 = 0;
            unsigned int _phase_k_done_1 = 1;
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
                        : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                        : "memory");
                    valid_4 = _clc_valid_2;
                    uint32_t _clc_ctaid_4 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_4)
                        : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                        : "memory");
                    next_x_4 = _clc_ctaid_4;
                    uint32_t _clc_ctaid_5 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_5)
                        : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                        : "memory");
                    next_y_4 = _clc_ctaid_5;
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
                for (int route_k = 0; route_k < route_k_extent_1; route_k++) {
                    int iter_k = route_k;
                    int route_tile = n_tile_2;
                    mbarrier_wait(k_done_addr + (stage_1) * 8, _phase_k_done_1);
                    if (elect_sync()) {
                        tma_3d_gmem2smem(smem_b_addr + stage_1 * 4096, (&B), iter_k * 256, 0, route_tile, b_full_addr + (stage_1) * 8);
                        mbarrier_arrive_expect_tx(b_full_addr + (stage_1) * 8, 4096);
                    }
                    stage_1 += 1;
                    if (stage_1 == 9) { stage_1 = 0; _phase_k_done_1 ^= 1; }
                }
                {
                    mbarrier_wait_hint(work_full_addr + (work_stage_2) * 8, _phase_work_full_2, 10000000);
                }
                unsigned int valid_5 = 0;
                unsigned int next_x_5 = 0;
                unsigned int next_y_5 = 0;
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
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                    : "memory");
                valid_5 = _clc_valid_3;
                uint32_t _clc_ctaid_6 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_6)
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                    : "memory");
                next_x_5 = _clc_ctaid_6;
                uint32_t _clc_ctaid_7 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_7)
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                    : "memory");
                next_y_5 = _clc_ctaid_7;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage_2) * 8);
                work_stage_2 += 1;
                if (work_stage_2 == 3) { work_stage_2 = 0; _phase_work_full_2 ^= 1; }
                unsigned int valid_0_2 = valid_5;
                m_tile_2 = next_x_5;
                n_tile_2 = next_y_5;
                if (valid_0_2 == 0) {
                    break;
                }
            }
        }
    }
    // ---- Role: load_sfb ----
    if (warp == 9) {
        { // load_sfb_main
            unsigned int stage_2 = 0;
            unsigned int work_stage_3 = 0;
            unsigned int m_tile_3 = blockIdx.x;
            unsigned int n_tile_3 = blockIdx.y;
            asm volatile("griddepcontrol.wait;" ::: "memory");
            unsigned int _phase_work_full_3 = 0;
            unsigned int _phase_sfb_smem_free = 1;
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
                        : "r"(work_response_addr + work_stage_3 * 16 + 0 * 16)
                        : "memory");
                    valid_6 = _clc_valid_6;
                    uint32_t _clc_ctaid_12 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_12)
                        : "r"(work_response_addr + work_stage_3 * 16 + 0 * 16)
                        : "memory");
                    next_x_6 = _clc_ctaid_12;
                    uint32_t _clc_ctaid_13 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_13)
                        : "r"(work_response_addr + work_stage_3 * 16 + 0 * 16)
                        : "memory");
                    next_y_6 = _clc_ctaid_13;
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
                int route_k_extent_2 = K_tiles;
                #pragma unroll 1
                for (int route_k_1 = 0; route_k_1 < route_k_extent_2; route_k_1++) {
                    int iter_k_1 = route_k_1;
                    int route_tile_1 = n_tile_3;
                    mbarrier_wait(sfb_smem_free_addr + (stage_2) * 8, _phase_sfb_smem_free);
                    if (elect_sync()) {
                        tma_3d_gmem2smem(smem_sfb_addr + stage_2 * 512, (&SFB), 0, iter_k_1 * 4, route_tile_1 * 4, sfb_full_addr + (stage_2) * 8);
                        mbarrier_arrive_expect_tx(sfb_full_addr + (stage_2) * 8, 512);
                    }
                    stage_2 += 1;
                    if (stage_2 == 9) { stage_2 = 0; _phase_sfb_smem_free ^= 1; }
                }
                {
                    mbarrier_wait_hint(work_full_addr + (work_stage_3) * 8, _phase_work_full_3, 10000000);
                }
                unsigned int valid_7 = 0;
                unsigned int next_x_7 = 0;
                unsigned int next_y_7 = 0;
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
                    : "r"(work_response_addr + work_stage_3 * 16 + 0 * 16)
                    : "memory");
                valid_7 = _clc_valid_7;
                uint32_t _clc_ctaid_14 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_14)
                    : "r"(work_response_addr + work_stage_3 * 16 + 0 * 16)
                    : "memory");
                next_x_7 = _clc_ctaid_14;
                uint32_t _clc_ctaid_15 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_15)
                    : "r"(work_response_addr + work_stage_3 * 16 + 0 * 16)
                    : "memory");
                next_y_7 = _clc_ctaid_15;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage_3) * 8);
                work_stage_3 += 1;
                if (work_stage_3 == 3) { work_stage_3 = 0; _phase_work_full_3 ^= 1; }
                unsigned int valid_0_3 = valid_7;
                m_tile_3 = next_x_7;
                n_tile_3 = next_y_7;
                if (valid_0_3 == 0) {
                    break;
                }
            }
        }
    }
    // ---- Role: load_a ----
    if (warp == 10) {
        { // load_a_main
            unsigned int stage_3 = 0;
            unsigned int work_stage_4 = 0;
            unsigned int throttle_stage = 0;
            unsigned int m_tile_4 = blockIdx.x;
            unsigned int n_tile_4 = blockIdx.y;
            unsigned int _phase_work_full_4 = 0;
            unsigned int _phase_throttle_empty = 1;
            unsigned int _phase_k_done_2 = 1;
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
                        : "r"(work_response_addr + work_stage_4 * 16 + 0 * 16)
                        : "memory");
                    valid_8 = _clc_valid_0;
                    uint32_t _clc_ctaid_0 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_0)
                        : "r"(work_response_addr + work_stage_4 * 16 + 0 * 16)
                        : "memory");
                    next_x_8 = _clc_ctaid_0;
                    uint32_t _clc_ctaid_1 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_1)
                        : "r"(work_response_addr + work_stage_4 * 16 + 0 * 16)
                        : "memory");
                    next_y_8 = _clc_ctaid_1;
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
                mbarrier_wait(throttle_empty_addr + (throttle_stage) * 8, _phase_throttle_empty);
                mbarrier_arrive(throttle_full_addr + (throttle_stage) * 8);
                throttle_stage += 1;
                if (throttle_stage == 3) { throttle_stage = 0; _phase_throttle_empty ^= 1; }
                int route_k_extent_3 = K_tiles;
                #pragma unroll 1
                for (int route_k_2 = 0; route_k_2 < route_k_extent_3; route_k_2++) {
                    int iter_k_2 = route_k_2;
                    int expert_1 = tile_expert[n_tile_4];
                    mbarrier_wait(k_done_addr + (stage_3) * 8, _phase_k_done_2);
                    if (elect_sync()) {
                        tma_3d_gmem2smem(smem_a_addr + stage_3 * 16384, (&A), iter_k_2 * 256, m_tile_4 * (unsigned int)BLOCK_M, expert_1, a_full_addr + (stage_3) * 8);
                        mbarrier_arrive_expect_tx(a_full_addr + (stage_3) * 8, 16384);
                    }
                    stage_3 += 1;
                    if (stage_3 == 9) { stage_3 = 0; _phase_k_done_2 ^= 1; }
                }
                {
                    mbarrier_wait_hint(work_full_addr + (work_stage_4) * 8, _phase_work_full_4, 10000000);
                }
                unsigned int valid_9 = 0;
                unsigned int next_x_9 = 0;
                unsigned int next_y_9 = 0;
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
                    : "r"(work_response_addr + work_stage_4 * 16 + 0 * 16)
                    : "memory");
                valid_9 = _clc_valid_1;
                uint32_t _clc_ctaid_2 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_2)
                    : "r"(work_response_addr + work_stage_4 * 16 + 0 * 16)
                    : "memory");
                next_x_9 = _clc_ctaid_2;
                uint32_t _clc_ctaid_3 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_3)
                    : "r"(work_response_addr + work_stage_4 * 16 + 0 * 16)
                    : "memory");
                next_y_9 = _clc_ctaid_3;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage_4) * 8);
                work_stage_4 += 1;
                if (work_stage_4 == 3) { work_stage_4 = 0; _phase_work_full_4 ^= 1; }
                unsigned int valid_0_4 = valid_9;
                m_tile_4 = next_x_9;
                n_tile_4 = next_y_9;
                if (valid_0_4 == 0) {
                    break;
                }
            }
        }
    }
    // ---- Role: load_sfa ----
    if (warp == 11) {
        { // load_sfa_main
            unsigned int stage_4 = 0;
            unsigned int work_stage_5 = 0;
            unsigned int m_tile_5 = blockIdx.x;
            unsigned int n_tile_5 = blockIdx.y;
            unsigned int _phase_work_full_5 = 0;
            unsigned int _phase_sfa_smem_free = 1;
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
                        : "r"(work_response_addr + work_stage_5 * 16 + 0 * 16)
                        : "memory");
                    valid_10 = _clc_valid_4;
                    uint32_t _clc_ctaid_8 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_8)
                        : "r"(work_response_addr + work_stage_5 * 16 + 0 * 16)
                        : "memory");
                    next_x_10 = _clc_ctaid_8;
                    uint32_t _clc_ctaid_9 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_9)
                        : "r"(work_response_addr + work_stage_5 * 16 + 0 * 16)
                        : "memory");
                    next_y_10 = _clc_ctaid_9;
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
                for (int route_k_3 = 0; route_k_3 < route_k_extent_4; route_k_3++) {
                    int iter_k_3 = route_k_3;
                    int expert_2 = tile_expert[n_tile_5];
                    mbarrier_wait(sfa_smem_free_addr + (stage_4) * 8, _phase_sfa_smem_free);
                    if (elect_sync()) {
                        tma_4d_gmem2smem(smem_sfa_addr + stage_4 * 2048, (&SFA), 0, 0, iter_k_3 * 4, (unsigned int)(expert_2 * grid_m) + m_tile_5, sfa_full_addr + (stage_4) * 8);
                        mbarrier_arrive_expect_tx(sfa_full_addr + (stage_4) * 8, 2048);
                    }
                    stage_4 += 1;
                    if (stage_4 == 9) { stage_4 = 0; _phase_sfa_smem_free ^= 1; }
                }
                {
                    mbarrier_wait_hint(work_full_addr + (work_stage_5) * 8, _phase_work_full_5, 10000000);
                }
                unsigned int valid_11 = 0;
                unsigned int next_x_11 = 0;
                unsigned int next_y_11 = 0;
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
                    : "r"(work_response_addr + work_stage_5 * 16 + 0 * 16)
                    : "memory");
                valid_11 = _clc_valid_5;
                uint32_t _clc_ctaid_10 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_10)
                    : "r"(work_response_addr + work_stage_5 * 16 + 0 * 16)
                    : "memory");
                next_x_11 = _clc_ctaid_10;
                uint32_t _clc_ctaid_11 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_11)
                    : "r"(work_response_addr + work_stage_5 * 16 + 0 * 16)
                    : "memory");
                next_y_11 = _clc_ctaid_11;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage_5) * 8);
                work_stage_5 += 1;
                if (work_stage_5 == 3) { work_stage_5 = 0; _phase_work_full_5 ^= 1; }
                unsigned int valid_0_5 = valid_11;
                m_tile_5 = next_x_11;
                n_tile_5 = next_y_11;
                if (valid_0_5 == 0) {
                    break;
                }
            }
        }
    }
    // ---- Role: copy_sfa ----
    if (warp == 12) {
        { // copy_sfa_main
            unsigned int stage_5 = 0;
            unsigned int work_stage_6 = 0;
            unsigned int m_tile_6 = blockIdx.x;
            unsigned int n_tile_6 = blockIdx.y;
            unsigned int _phase_work_full_6 = 0;
            unsigned int _phase_sfa_full = 0;
            unsigned int _phase_k_done_3 = 1;
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
                        : "r"(work_response_addr + work_stage_6 * 16 + 0 * 16)
                        : "memory");
                    valid_12 = _clc_valid_8;
                    uint32_t _clc_ctaid_16 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_16)
                        : "r"(work_response_addr + work_stage_6 * 16 + 0 * 16)
                        : "memory");
                    next_x_12 = _clc_ctaid_16;
                    uint32_t _clc_ctaid_17 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_17)
                        : "r"(work_response_addr + work_stage_6 * 16 + 0 * 16)
                        : "memory");
                    next_y_12 = _clc_ctaid_17;
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
                int route_k_extent_5 = K_tiles;
                #pragma unroll 1
                for (int _route_k_1 = 0; _route_k_1 < route_k_extent_5; _route_k_1++) {
                    mbarrier_wait(sfa_full_addr + (stage_5) * 8, _phase_sfa_full);
                    mbarrier_wait(k_done_addr + (stage_5) * 8, _phase_k_done_3);
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    if (elect_sync()) {
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_0 = ((((uint64_t)(smem_sfa_addr + stage_5 * 2048)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                :: "r"((uint32_t)((unsigned int)tmem_sfa + stage_5 * 16)), "l"(_tcgen05_cp_desc_0)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_1 = ((((uint64_t)(smem_sfa_addr + stage_5 * 2048 + 512)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_5 * 16 + 4))), "l"(_tcgen05_cp_desc_1)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_2 = ((((uint64_t)(smem_sfa_addr + stage_5 * 2048 + 1024)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_5 * 16 + 8))), "l"(_tcgen05_cp_desc_2)
                                : "memory");
                        }
                        #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 1000)
                        #error "Tcgen05Cp requires Blackwell tcgen05.cp support"
                        #endif
                        {
                            uint64_t _tcgen05_cp_desc_3 = ((((uint64_t)(smem_sfa_addr + stage_5 * 2048 + 1536)) & 0x3FFFFULL) >> 4ULL) | (((((uint64_t)(0)) & 0x3FFFFULL) >> 4ULL) << 16ULL) | (((((uint64_t)(128)) & 0x3FFFFULL) >> 4ULL) << 32ULL) | (1ULL << 46ULL);
                            asm volatile(
                                "tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;"
                                :: "r"((uint32_t)((unsigned int)tmem_sfa + (stage_5 * 16 + 12))), "l"(_tcgen05_cp_desc_3)
                                : "memory");
                        }
                    }
                    elect_commit2(tmem_sfa_full_addr + (stage_5) * 8, sfa_smem_free_addr + (stage_5) * 8);
                    stage_5 += 1;
                    if (stage_5 == 9) { stage_5 = 0; _phase_sfa_full ^= 1; _phase_k_done_3 ^= 1; }
                }
                {
                    mbarrier_wait_hint(work_full_addr + (work_stage_6) * 8, _phase_work_full_6, 10000000);
                }
                unsigned int valid_13 = 0;
                unsigned int next_x_13 = 0;
                unsigned int next_y_13 = 0;
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
                    : "r"(work_response_addr + work_stage_6 * 16 + 0 * 16)
                    : "memory");
                valid_13 = _clc_valid_9;
                uint32_t _clc_ctaid_18 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_18)
                    : "r"(work_response_addr + work_stage_6 * 16 + 0 * 16)
                    : "memory");
                next_x_13 = _clc_ctaid_18;
                uint32_t _clc_ctaid_19 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_19)
                    : "r"(work_response_addr + work_stage_6 * 16 + 0 * 16)
                    : "memory");
                next_y_13 = _clc_ctaid_19;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage_6) * 8);
                work_stage_6 += 1;
                if (work_stage_6 == 3) { work_stage_6 = 0; _phase_work_full_6 ^= 1; }
                unsigned int valid_0_6 = valid_13;
                m_tile_6 = next_x_13;
                n_tile_6 = next_y_13;
                if (valid_0_6 == 0) {
                    break;
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 13) {
        { // mma_main
            unsigned int k_stage = 0;
            unsigned int acc_stage_1 = 0;
            unsigned int work_stage_7 = 0;
            unsigned int k_phase = 0;
            unsigned int a_token = 0;
            unsigned int b_token = 0;
            unsigned int sfa_token = 0;
            unsigned int sfb_token = 0;
            unsigned int m_tile_7 = blockIdx.x;
            unsigned int n_tile_7 = blockIdx.y;
            unsigned int _phase_work_full_7 = 0;
            unsigned int _phase_mma_free = 1;
            unsigned int _phase_a_full = 0;
            unsigned int _phase_b_full = 0;
            unsigned int _phase_tmem_sfa_full = 0;
            unsigned int _phase_tmem_sfb_full = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_7 = 0; _tile_iter_7 < grid_m * grid_n; _tile_iter_7++) {
                unsigned int inactive_valid_7 = 1;
                #pragma unroll 1
                for (unsigned int _inactive_iter_7 = 0; _inactive_iter_7 < grid_m * grid_n; _inactive_iter_7++) {
                    if (n_tile_7 < (unsigned int)total_tiles[0]) {
                        break;
                    }
                    {
                        mbarrier_wait_hint(work_full_addr + (work_stage_7) * 8, _phase_work_full_7, 10000000);
                    }
                    unsigned int valid_14 = 0;
                    unsigned int next_x_14 = 0;
                    unsigned int next_y_14 = 0;
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
                        : "r"(work_response_addr + work_stage_7 * 16 + 0 * 16)
                        : "memory");
                    valid_14 = _clc_valid_12;
                    uint32_t _clc_ctaid_24 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_24)
                        : "r"(work_response_addr + work_stage_7 * 16 + 0 * 16)
                        : "memory");
                    next_x_14 = _clc_ctaid_24;
                    uint32_t _clc_ctaid_25 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_25)
                        : "r"(work_response_addr + work_stage_7 * 16 + 0 * 16)
                        : "memory");
                    next_y_14 = _clc_ctaid_25;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(work_empty_addr + (work_stage_7) * 8);
                    work_stage_7 += 1;
                    if (work_stage_7 == 3) { work_stage_7 = 0; _phase_work_full_7 ^= 1; }
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
                mbarrier_wait(mma_free_addr + (acc_stage_1) * 8, _phase_mma_free);
                int route_k_extent_6 = K_tiles;
                #pragma unroll 1
                for (int route_k_4 = 0; route_k_4 < route_k_extent_6; route_k_4++) {
                    int route_slot = ((0) ? route_k_4 / K_tiles : 0);
                    {
                        mbarrier_wait(a_full_addr + (k_stage) * 8, _phase_a_full);
                        mbarrier_wait(b_full_addr + (k_stage) * 8, _phase_b_full);
                        mbarrier_wait(tmem_sfa_full_addr + (k_stage) * 8, _phase_tmem_sfa_full);
                        mbarrier_wait(tmem_sfb_full_addr + (k_stage) * 8, _phase_tmem_sfb_full);
                    }
                    asm volatile("tcgen05.fence::after_thread_sync;");
                    int sfa_base_col = 0;
                    int sfb_base_col = 0;
                    int _mma_a_lo_0 = make_warp_uniform((((smem_a_mma0_addr) >> 4) & 0x3FFF) + (k_stage) * 1024);
                    int _mma_b_lo_0 = make_warp_uniform((((smem_b_mma0_addr) >> 4) & 0x3FFF) + (k_stage) * 256);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_accum + ((acc_stage_1 + (unsigned int)route_slot) * 32)), a_desc + 0, b_desc + 0,
                                0x8080480U, (unsigned int)tmem_sfa + (k_stage * 16 + (unsigned int)sfa_base_col) + 0, (unsigned int)tmem_sfb + (k_stage * 8 + (unsigned int)sfb_base_col) + 0, ((((1) ? ((0) ? ((route_k_4 % K_tiles == 0) ? 1 : 0) : ((route_k_4 == 0) ? 1 : 0)) : 0)) ? 0 : 1));
                        }
                    }
                    int sfa_base_col_0 = 4;
                    int sfb_base_col_1 = 2;
                    int _mma_a_lo_1 = make_warp_uniform((((smem_a_mma1_addr) >> 4) & 0x3FFF) + (k_stage) * 1024);
                    int _mma_b_lo_1 = make_warp_uniform((((smem_b_mma1_addr) >> 4) & 0x3FFF) + (k_stage) * 256);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_accum + ((acc_stage_1 + (unsigned int)route_slot) * 32)), a_desc + 0, b_desc + 0,
                                0x8080480U, (unsigned int)tmem_sfa + (k_stage * 16 + (unsigned int)sfa_base_col_0) + 0, (unsigned int)tmem_sfb + (k_stage * 8 + (unsigned int)sfb_base_col_1) + 0, ((((0) ? ((0) ? ((route_k_4 % K_tiles == 0) ? 1 : 0) : ((route_k_4 == 0) ? 1 : 0)) : 0)) ? 0 : 1));
                        }
                    }
                    int sfa_base_col_2 = 8;
                    int sfb_base_col_3 = 4;
                    int _mma_a_lo_2 = make_warp_uniform((((smem_a_mma2_addr) >> 4) & 0x3FFF) + (k_stage) * 1024);
                    int _mma_b_lo_2 = make_warp_uniform((((smem_b_mma2_addr) >> 4) & 0x3FFF) + (k_stage) * 256);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_2) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_2) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_accum + ((acc_stage_1 + (unsigned int)route_slot) * 32)), a_desc + 0, b_desc + 0,
                                0x8080480U, (unsigned int)tmem_sfa + (k_stage * 16 + (unsigned int)sfa_base_col_2) + 0, (unsigned int)tmem_sfb + (k_stage * 8 + (unsigned int)sfb_base_col_3) + 0, ((((0) ? ((0) ? ((route_k_4 % K_tiles == 0) ? 1 : 0) : ((route_k_4 == 0) ? 1 : 0)) : 0)) ? 0 : 1));
                        }
                    }
                    int sfa_base_col_4 = 12;
                    int sfb_base_col_5 = 6;
                    int _mma_a_lo_3 = make_warp_uniform((((smem_a_mma3_addr) >> 4) & 0x3FFF) + (k_stage) * 1024);
                    int _mma_b_lo_3 = make_warp_uniform((((smem_b_mma3_addr) >> 4) & 0x3FFF) + (k_stage) * 256);
                    if (elect_sync()) {
                        {
                            uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_3) | ((uint64_t)0x40004040 << 32);
                            uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_3) | ((uint64_t)0x40004040 << 32);

                            tcgen05_mma_mxf4nvf4_bs((tmem_accum + ((acc_stage_1 + (unsigned int)route_slot) * 32)), a_desc + 0, b_desc + 0,
                                0x8080480U, (unsigned int)tmem_sfa + (k_stage * 16 + (unsigned int)sfa_base_col_4) + 0, (unsigned int)tmem_sfb + (k_stage * 8 + (unsigned int)sfb_base_col_5) + 0, ((((0) ? ((0) ? ((route_k_4 % K_tiles == 0) ? 1 : 0) : ((route_k_4 == 0) ? 1 : 0)) : 0)) ? 0 : 1));
                        }
                    }
                    if (route_k_4 + 1 == route_k_extent_6) {
                        elect_commit2(k_done_addr + (k_stage) * 8, mma_full_addr + (acc_stage_1) * 8);
                    } else {
                        elect_commit(k_done_addr + (k_stage) * 8);
                    }
                    {
                        k_stage += 1;
                        if (k_stage == 9) { k_stage = 0; _phase_a_full ^= 1; _phase_b_full ^= 1; _phase_tmem_sfa_full ^= 1; _phase_tmem_sfb_full ^= 1; }
                    }
                }
                acc_stage_1 += 1;
                if (acc_stage_1 == 2) { acc_stage_1 = 0; _phase_mma_free ^= 1; }
                {
                    mbarrier_wait_hint(work_full_addr + (work_stage_7) * 8, _phase_work_full_7, 10000000);
                }
                unsigned int valid_15 = 0;
                unsigned int next_x_15 = 0;
                unsigned int next_y_15 = 0;
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
                    : "r"(work_response_addr + work_stage_7 * 16 + 0 * 16)
                    : "memory");
                valid_15 = _clc_valid_13;
                uint32_t _clc_ctaid_26 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_26)
                    : "r"(work_response_addr + work_stage_7 * 16 + 0 * 16)
                    : "memory");
                next_x_15 = _clc_ctaid_26;
                uint32_t _clc_ctaid_27 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_27)
                    : "r"(work_response_addr + work_stage_7 * 16 + 0 * 16)
                    : "memory");
                next_y_15 = _clc_ctaid_27;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage_7) * 8);
                work_stage_7 += 1;
                if (work_stage_7 == 3) { work_stage_7 = 0; _phase_work_full_7 ^= 1; }
                unsigned int valid_0_7 = valid_15;
                m_tile_7 = next_x_15;
                n_tile_7 = next_y_15;
                if (valid_0_7 == 0) {
                    break;
                }
            }
        }
    }
    // ---- Role: work_id ----
    if (warp == 14) {
        { // work_id_main
            unsigned int work_stage_8 = 0;
            unsigned int throttle_stage_1 = 0;
            unsigned int m_tile_8 = blockIdx.x;
            unsigned int n_tile_8 = blockIdx.y;
            {
                asm volatile("griddepcontrol.wait;" ::: "memory");
            }
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_work_full_8 = 0;
            unsigned int _phase_throttle_full = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_8 = 0; _tile_iter_8 < grid_m * grid_n; _tile_iter_8++) {
                unsigned int inactive_valid_8 = 1;
                #pragma unroll 1
                for (unsigned int _inactive_iter_8 = 0; _inactive_iter_8 < grid_m * grid_n; _inactive_iter_8++) {
                    if (n_tile_8 < (unsigned int)total_tiles[0]) {
                        break;
                    }
                    if (elect_sync()) {
                        mbarrier_wait(work_empty_addr + (work_stage_8) * 8, _phase_work_empty);
                        mbarrier_arrive_expect_tx(work_full_addr + (work_stage_8) * 8, 16);
                        asm volatile(
                            "fence.proxy.async.shared::cta;\n\t"
                            "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                ".mbarrier::complete_tx::bytes.b128"
                                " [%0], [%1];"
                            :: "r"(work_response_addr + work_stage_8 * 16 + 0 * 16), "r"(work_full_addr + work_stage_8 * 8)
                            : "memory");
                    }
                    {
                        mbarrier_wait_hint(work_full_addr + (work_stage_8) * 8, _phase_work_full_8, 10000000);
                    }
                    unsigned int valid_16 = 0;
                    unsigned int next_x_16 = 0;
                    unsigned int next_y_16 = 0;
                    uint32_t _clc_valid_18 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "selp.u32 %0, 1, 0, p1;\n\t"
                        "}\n"
                        : "=r"(_clc_valid_18)
                        : "r"(work_response_addr + work_stage_8 * 16 + 0 * 16)
                        : "memory");
                    valid_16 = _clc_valid_18;
                    uint32_t _clc_ctaid_36 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_36)
                        : "r"(work_response_addr + work_stage_8 * 16 + 0 * 16)
                        : "memory");
                    next_x_16 = _clc_ctaid_36;
                    uint32_t _clc_ctaid_37 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_37)
                        : "r"(work_response_addr + work_stage_8 * 16 + 0 * 16)
                        : "memory");
                    next_y_16 = _clc_ctaid_37;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(work_empty_addr + (work_stage_8) * 8);
                    work_stage_8 += 1;
                    if (work_stage_8 == 3) { work_stage_8 = 0; _phase_work_empty ^= 1; _phase_work_full_8 ^= 1; }
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
                mbarrier_wait(throttle_full_addr + (throttle_stage_1) * 8, _phase_throttle_full);
                mbarrier_arrive(throttle_empty_addr + (throttle_stage_1) * 8);
                throttle_stage_1 += 1;
                if (throttle_stage_1 == 3) { throttle_stage_1 = 0; _phase_throttle_full ^= 1; }
                if (elect_sync()) {
                    mbarrier_wait(work_empty_addr + (work_stage_8) * 8, _phase_work_empty);
                    mbarrier_arrive_expect_tx(work_full_addr + (work_stage_8) * 8, 16);
                    asm volatile(
                        "fence.proxy.async.shared::cta;\n\t"
                        "clusterlaunchcontrol.try_cancel.async.shared::cta"
                            ".mbarrier::complete_tx::bytes.b128"
                            " [%0], [%1];"
                        :: "r"(work_response_addr + work_stage_8 * 16 + 0 * 16), "r"(work_full_addr + work_stage_8 * 8)
                        : "memory");
                }
                {
                    mbarrier_wait_hint(work_full_addr + (work_stage_8) * 8, _phase_work_full_8, 10000000);
                }
                unsigned int valid_17 = 0;
                unsigned int next_x_17 = 0;
                unsigned int next_y_17 = 0;
                uint32_t _clc_valid_19 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .pred p1;\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                    "selp.u32 %0, 1, 0, p1;\n\t"
                    "}\n"
                    : "=r"(_clc_valid_19)
                    : "r"(work_response_addr + work_stage_8 * 16 + 0 * 16)
                    : "memory");
                valid_17 = _clc_valid_19;
                uint32_t _clc_ctaid_38 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_38)
                    : "r"(work_response_addr + work_stage_8 * 16 + 0 * 16)
                    : "memory");
                next_x_17 = _clc_ctaid_38;
                uint32_t _clc_ctaid_39 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_39)
                    : "r"(work_response_addr + work_stage_8 * 16 + 0 * 16)
                    : "memory");
                next_y_17 = _clc_ctaid_39;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage_8) * 8);
                work_stage_8 += 1;
                if (work_stage_8 == 3) { work_stage_8 = 0; _phase_work_empty ^= 1; _phase_work_full_8 ^= 1; }
                unsigned int valid_0_8 = valid_17;
                m_tile_8 = next_x_17;
                n_tile_8 = next_y_17;
                if (valid_0_8 == 0) {
                    break;
                }
            }
        }
    }
    // ---- Role: padding ----
    if (warp == 15) {
        { // padding_main
            unsigned int work_stage_9 = 0;
            unsigned int m_tile_9 = blockIdx.x;
            unsigned int n_tile_9 = blockIdx.y;
            unsigned int _phase_work_full_9 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_9 = 0; _tile_iter_9 < grid_m * grid_n; _tile_iter_9++) {
                unsigned int inactive_valid_9 = 1;
                #pragma unroll 1
                for (unsigned int _inactive_iter_9 = 0; _inactive_iter_9 < grid_m * grid_n; _inactive_iter_9++) {
                    if (n_tile_9 < (unsigned int)total_tiles[0]) {
                        break;
                    }
                    {
                        mbarrier_wait_hint(work_full_addr + (work_stage_9) * 8, _phase_work_full_9, 10000000);
                    }
                    unsigned int valid_18 = 0;
                    unsigned int next_x_18 = 0;
                    unsigned int next_y_18 = 0;
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
                        : "r"(work_response_addr + work_stage_9 * 16 + 0 * 16)
                        : "memory");
                    valid_18 = _clc_valid_16;
                    uint32_t _clc_ctaid_32 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_32)
                        : "r"(work_response_addr + work_stage_9 * 16 + 0 * 16)
                        : "memory");
                    next_x_18 = _clc_ctaid_32;
                    uint32_t _clc_ctaid_33 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_33)
                        : "r"(work_response_addr + work_stage_9 * 16 + 0 * 16)
                        : "memory");
                    next_y_18 = _clc_ctaid_33;
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    mbarrier_arrive(work_empty_addr + (work_stage_9) * 8);
                    work_stage_9 += 1;
                    if (work_stage_9 == 3) { work_stage_9 = 0; _phase_work_full_9 ^= 1; }
                    inactive_valid_9 = valid_18;
                    m_tile_9 = next_x_18;
                    n_tile_9 = next_y_18;
                    if (inactive_valid_9 == 0) {
                        break;
                    }
                }
                if (inactive_valid_9 == 0) {
                    break;
                }
                if (m_tile_9 >= (unsigned int)grid_m || n_tile_9 >= (unsigned int)grid_n) {
                    break;
                }
                {
                    mbarrier_wait_hint(work_full_addr + (work_stage_9) * 8, _phase_work_full_9, 10000000);
                }
                unsigned int valid_19 = 0;
                unsigned int next_x_19 = 0;
                unsigned int next_y_19 = 0;
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
                    : "r"(work_response_addr + work_stage_9 * 16 + 0 * 16)
                    : "memory");
                valid_19 = _clc_valid_17;
                uint32_t _clc_ctaid_34 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_34)
                    : "r"(work_response_addr + work_stage_9 * 16 + 0 * 16)
                    : "memory");
                next_x_19 = _clc_ctaid_34;
                uint32_t _clc_ctaid_35 = 0;
                asm volatile(
                    "{\n\t"
                    ".reg .b128 clc_r;\n\t"
                    "ld.shared.b128 clc_r, [%1];\n\t"
                    "clusterlaunchcontrol.query_cancel.get_first_ctaid::y.b32.b128 %0, clc_r;\n\t"
                    "}\n"
                    : "=r"(_clc_ctaid_35)
                    : "r"(work_response_addr + work_stage_9 * 16 + 0 * 16)
                    : "memory");
                next_y_19 = _clc_ctaid_35;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                mbarrier_arrive(work_empty_addr + (work_stage_9) * 8);
                work_stage_9 += 1;
                if (work_stage_9 == 3) { work_stage_9 = 0; _phase_work_full_9 ^= 1; }
                unsigned int valid_0_9 = valid_19;
                m_tile_9 = next_x_19;
                n_tile_9 = next_y_19;
                if (valid_0_9 == 0) {
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
