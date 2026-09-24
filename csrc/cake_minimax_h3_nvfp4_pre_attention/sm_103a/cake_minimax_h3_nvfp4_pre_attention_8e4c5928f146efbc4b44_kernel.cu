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

__device__ __forceinline__ int make_warp_uniform(int x) {
    int result;
    asm volatile("shfl.sync.idx.b32 %0, %1, 0, 0x1F, 0xFFFFFFFF;"
                 : "=r"(result) : "r"(x));
    return result;
}

#define CAKE_INF CUDART_INF_F
#define TMEM_NCOLS 400
#define TMEM_ACCUM_OFFSET 0
#define TMEM_TMEM_SFA_OFFSET 256
#define TMEM_TMEM_SFB_OFFSET 304
#define NUM_TMA_PIPE_STAGES 5
#define NUM_MAINLOOP_PIPE_STAGES 1
#define NUM_WORK_PIPE_STAGES 4
#define SMEM_SMEM_A_OFF 1024
#define SMEM_SMEM_A_STAGE_BYTES 16384
#define SMEM_SMEM_A_STRIDE 38912
#define SMEM_SMEM_B_OFF 17408
#define SMEM_SMEM_B_STAGE_BYTES 16384
#define SMEM_SMEM_B_STRIDE 38912
#define SMEM_SMEM_SFA_ALL_OFF 33792
#define SMEM_SMEM_SFA_ALL_STAGE_BYTES 2048
#define SMEM_SMEM_SFA_ALL_STRIDE 38912
#define SMEM_SMEM_SFB_ALL_OFF 35840
#define SMEM_SMEM_SFB_ALL_STAGE_BYTES 4096
#define SMEM_SMEM_SFB_ALL_STRIDE 38912
#define SMEM_SMEM_V4_OFF 33792
#define SMEM_SMEM_V4_STAGE_BYTES 512
#define SMEM_SMEM_V4_STRIDE 38912
#define SMEM_SMEM_V5_OFF 34304
#define SMEM_SMEM_V5_STAGE_BYTES 512
#define SMEM_SMEM_V5_STRIDE 38912
#define SMEM_SMEM_V6_OFF 34816
#define SMEM_SMEM_V6_STAGE_BYTES 512
#define SMEM_SMEM_V6_STRIDE 38912
#define SMEM_SMEM_V7_OFF 35328
#define SMEM_SMEM_V7_STAGE_BYTES 512
#define SMEM_SMEM_V7_STRIDE 38912
#define SMEM_SMEM_V8_OFF 35840
#define SMEM_SMEM_V8_STAGE_BYTES 1024
#define SMEM_SMEM_V8_STRIDE 38912
#define SMEM_SMEM_V9_OFF 36864
#define SMEM_SMEM_V9_STAGE_BYTES 1024
#define SMEM_SMEM_V9_STRIDE 38912
#define SMEM_SMEM_V10_OFF 37888
#define SMEM_SMEM_V10_STAGE_BYTES 1024
#define SMEM_SMEM_V10_STRIDE 38912
#define SMEM_SMEM_V11_OFF 38912
#define SMEM_SMEM_V11_STAGE_BYTES 1024
#define SMEM_SMEM_V11_STRIDE 38912
#define SMEM_EPI_STAGING_OFF 195584
#define SMEM_EPI_STAGING_STAGE_BYTES 8192
#define SMEM_EPI_STAGING_STRIDE 8192
#define SMEM_WORK_RESPONSE_OFF 228352
#define SMEM_WORK_RESPONSE_STAGE_BYTES 16
#define SMEM_WORK_RESPONSE_STRIDE 16
#define SMEM_SMEM_V14_OFF 1024
#define SMEM_SMEM_V14_STAGE_BYTES 6144
#define SMEM_SMEM_V14_STRIDE 38912
#define SMEM_SMEM_V15_OFF 1072
#define SMEM_SMEM_V15_STAGE_BYTES 6144
#define SMEM_SMEM_V15_STRIDE 38912
#define SMEM_SMEM_V16_OFF 1120
#define SMEM_SMEM_V16_STAGE_BYTES 6144
#define SMEM_SMEM_V16_STRIDE 38912
#define SMEM_SMEM_V17_OFF 1040
#define SMEM_SMEM_V17_STAGE_BYTES 6144
#define SMEM_SMEM_V17_STRIDE 38912
#define SMEM_SMEM_V18_OFF 1088
#define SMEM_SMEM_V18_STAGE_BYTES 6144
#define SMEM_SMEM_V18_STRIDE 38912
#define SMEM_SMEM_V19_OFF 1136
#define SMEM_SMEM_V19_STAGE_BYTES 6144
#define SMEM_SMEM_V19_STRIDE 38912
#define SMEM_SMEM_V20_OFF 1056
#define SMEM_SMEM_V20_STAGE_BYTES 6144
#define SMEM_SMEM_V20_STRIDE 38912
#define SMEM_SMEM_V21_OFF 1104
#define SMEM_SMEM_V21_STAGE_BYTES 6144
#define SMEM_SMEM_V21_STRIDE 38912
#define SMEM_SMEM_V22_OFF 1024
#define SMEM_SMEM_V22_STAGE_BYTES 6144
#define SMEM_SMEM_V22_STRIDE 38912
#define SMEM_SMEM_V23_OFF 17408
#define SMEM_SMEM_V23_STAGE_BYTES 6144
#define SMEM_SMEM_V23_STRIDE 38912
#define SMEM_SMEM_V24_OFF 17456
#define SMEM_SMEM_V24_STAGE_BYTES 6144
#define SMEM_SMEM_V24_STRIDE 38912
#define SMEM_SMEM_V25_OFF 17504
#define SMEM_SMEM_V25_STAGE_BYTES 6144
#define SMEM_SMEM_V25_STRIDE 38912
#define SMEM_SMEM_V26_OFF 17424
#define SMEM_SMEM_V26_STAGE_BYTES 6144
#define SMEM_SMEM_V26_STRIDE 38912
#define SMEM_SMEM_V27_OFF 17472
#define SMEM_SMEM_V27_STAGE_BYTES 6144
#define SMEM_SMEM_V27_STRIDE 38912
#define SMEM_SMEM_V28_OFF 17520
#define SMEM_SMEM_V28_STAGE_BYTES 6144
#define SMEM_SMEM_V28_STRIDE 38912
#define SMEM_SMEM_V29_OFF 17440
#define SMEM_SMEM_V29_STAGE_BYTES 6144
#define SMEM_SMEM_V29_STRIDE 38912
#define SMEM_SMEM_V30_OFF 17488
#define SMEM_SMEM_V30_STAGE_BYTES 6144
#define SMEM_SMEM_V30_STRIDE 38912
#define SMEM_SMEM_V31_OFF 17408
#define SMEM_SMEM_V31_STAGE_BYTES 6144
#define SMEM_SMEM_V31_STRIDE 38912
#define SMEM_TOTAL 228480
#define BLOCK_M 128
#define BLOCK_N 256
#define B_HALF_N 128
#define BLOCK_K 256
#define CTA_GROUP 2
#define NUM_STAGES 5
#define GROUP_M 64
#define WORK_STAGES 4
#define WORK_CONSUMERS 546
#define NUM_K_ITERS 21
#define N_TILES 84
#define SF_K_TILES 84
#define num_cluster_tiles ((m_tiles / CTA_GROUP) * N_TILES)
#define tiles_per_group (GROUP_M * N_TILES)

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


__device__ __forceinline__ void tcgen05_mma_mxf4nvf4_bs_cta2(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
    int sfa_taddr, int sfb_taddr, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.scale_vec::4X"
        " [%0], %1, %2, %3, [%4], [%5], p;\n\t"
        "}\n"
        :: "r"(taddr), "l"(a_desc), "l"(b_desc),
           "r"(i_desc), "r"(sfa_taddr), "r"(sfb_taddr),
           "r"(enable_input_d));
}


__device__ __forceinline__ void tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2(
    int taddr, uint64_t a_desc, uint64_t b_desc, uint32_t i_desc,
    int sfa_taddr, int sfb_taddr, int enable_input_d) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, %6, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::mxf4nvf4.block_scale.block16"
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


__device__ __forceinline__ void fence_async_shared() {
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
}


__device__ __forceinline__ uint64_t make_sf_cp_desc_sbo128(int addr) {
    const int SBO = 128;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL);
}


__device__ __forceinline__ uint64_t make_sf_cp_desc_lo_sbo128(int lo) {
    const int SBO = 128;
    return (uint64_t)(uint32_t)lo
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL);
}


__device__ __forceinline__ void tcgen05_cp_32x128b_warpx4_cta2(
    int taddr, uint64_t s_desc) {
    asm volatile(
        "tcgen05.cp.cta_group::2.32x128b.warpx4 [%0], %1;"
        :: "r"(taddr), "l"(s_desc));
}


__device__ __forceinline__ uint64_t make_smem_desc(int addr) {
    const int SBO = 1024;
    return desc_encode(addr)
         | (desc_encode(SBO) << 32ULL)
         | (1ULL << 46ULL)
         | (2ULL << 61ULL);
}


__device__ __forceinline__ void tma_3d_gmem2smem_cta2(
    int dst, const void *tmap_ptr, int x, int y, int z, int mbar_addr) {
    asm volatile(
        "cp.async.bulk.tensor.3d.shared::cluster.global"
        ".mbarrier::complete_tx::bytes.cta_group::2"
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

__global__ __launch_bounds__(320) __cluster_dims__(2,1,1) void
kernel_cake_minimax_h3_nvfp4_pre_attention_8e4c5928f146efbc4b44(CakeTensorMap const* A, CakeTensorMap const* B, CakeTensorMap const* SFA, CakeTensorMap const* SFB, float* __restrict__ alpha, __nv_bfloat16* __restrict__ q_norm_weight, __nv_bfloat16* __restrict__ k_norm_weight, __nv_bfloat16* __restrict__ rope_cos_sin, float* __restrict__ out_global_scale, uint8_t* __restrict__ out_q, uint8_t* __restrict__ out_sf, CakeTensorMap const* OUTQ, unsigned int* __restrict__ qkv_words, unsigned int* __restrict__ debug_q_words, unsigned int* __restrict__ debug_k_words, int write_debug, float eps, int M, int m_tiles, int HEADS_PER_DESTINATION, int ROWS_PER_DESTINATION, int SCALE_STRIDE)
{
    const int tid = threadIdx.x;
    const int warp = make_warp_uniform(tid / 32);
    const int lane = tid % 32;

    extern __shared__ __align__(1024) char smem_raw[];
    int smem;
    smem = (int)(unsigned long long)__cvta_generic_to_shared(smem_raw);

    const int mbar_base = smem;
    #define tma_full_addr (mbar_base + 0)
    #define mma_done_addr (mbar_base + 40)
    #define mainloop_done_addr (mbar_base + 80)
    #define epilogue_done_addr (mbar_base + 88)
    #define work_full_addr (mbar_base + 96)
    #define work_empty_addr (mbar_base + 128)

    const int bid = blockIdx.x;
    const int num_bids = gridDim.x;
    const unsigned int clusters_x = gridDim.x / 2;
    const unsigned int cluster_id = ((blockIdx.z * gridDim.y + blockIdx.y) * clusters_x) + blockIdx.x / 2;
    const unsigned int num_clusters = clusters_x * gridDim.y * gridDim.z;

    int cta_rank;
    asm volatile("mov.b32 %0, %%cluster_ctarank;" : "=r"(cta_rank));
    if (tid == 0) {
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(A)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(B)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(SFA)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(SFB)) : "memory");
        asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"((uint64_t)(OUTQ)) : "memory");
    }
    __syncthreads();


    // Kernel setup ops
    uint8_t* smem_a = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_a_addr = smem + 1024;
    uint8_t* smem_b = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int smem_b_addr = smem + 17408;
    uint8_t* smem_sfa_all = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int smem_sfa_all_addr = smem + 33792;
    uint8_t* smem_sfb_all = reinterpret_cast<uint8_t*>(smem_raw + 35840);
    const int smem_sfb_all_addr = smem + 35840;
    uint8_t* smem_v4 = reinterpret_cast<uint8_t*>(smem_raw + 33792);
    const int smem_v4_addr = smem + 33792;
    uint8_t* smem_v5 = reinterpret_cast<uint8_t*>(smem_raw + 34304);
    const int smem_v5_addr = smem + 34304;
    uint8_t* smem_v6 = reinterpret_cast<uint8_t*>(smem_raw + 34816);
    const int smem_v6_addr = smem + 34816;
    uint8_t* smem_v7 = reinterpret_cast<uint8_t*>(smem_raw + 35328);
    const int smem_v7_addr = smem + 35328;
    uint8_t* smem_v8 = reinterpret_cast<uint8_t*>(smem_raw + 35840);
    const int smem_v8_addr = smem + 35840;
    uint8_t* smem_v9 = reinterpret_cast<uint8_t*>(smem_raw + 36864);
    const int smem_v9_addr = smem + 36864;
    uint8_t* smem_v10 = reinterpret_cast<uint8_t*>(smem_raw + 37888);
    const int smem_v10_addr = smem + 37888;
    uint8_t* smem_v11 = reinterpret_cast<uint8_t*>(smem_raw + 38912);
    const int smem_v11_addr = smem + 38912;
    uint8_t* epi_staging = reinterpret_cast<uint8_t*>(smem_raw + 195584);
    const int epi_staging_addr = smem + 195584;
    unsigned int* work_response = reinterpret_cast<unsigned int*>(smem_raw + 228352);
    const int work_response_addr = smem + 228352;
    uint8_t* smem_v14 = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_v14_addr = smem + 1024;
    uint8_t* smem_v15 = reinterpret_cast<uint8_t*>(smem_raw + 1072);
    const int smem_v15_addr = smem + 1072;
    uint8_t* smem_v16 = reinterpret_cast<uint8_t*>(smem_raw + 1120);
    const int smem_v16_addr = smem + 1120;
    uint8_t* smem_v17 = reinterpret_cast<uint8_t*>(smem_raw + 1040);
    const int smem_v17_addr = smem + 1040;
    uint8_t* smem_v18 = reinterpret_cast<uint8_t*>(smem_raw + 1088);
    const int smem_v18_addr = smem + 1088;
    uint8_t* smem_v19 = reinterpret_cast<uint8_t*>(smem_raw + 1136);
    const int smem_v19_addr = smem + 1136;
    uint8_t* smem_v20 = reinterpret_cast<uint8_t*>(smem_raw + 1056);
    const int smem_v20_addr = smem + 1056;
    uint8_t* smem_v21 = reinterpret_cast<uint8_t*>(smem_raw + 1104);
    const int smem_v21_addr = smem + 1104;
    uint8_t* smem_v22 = reinterpret_cast<uint8_t*>(smem_raw + 1024);
    const int smem_v22_addr = smem + 1024;
    uint8_t* smem_v23 = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int smem_v23_addr = smem + 17408;
    uint8_t* smem_v24 = reinterpret_cast<uint8_t*>(smem_raw + 17456);
    const int smem_v24_addr = smem + 17456;
    uint8_t* smem_v25 = reinterpret_cast<uint8_t*>(smem_raw + 17504);
    const int smem_v25_addr = smem + 17504;
    uint8_t* smem_v26 = reinterpret_cast<uint8_t*>(smem_raw + 17424);
    const int smem_v26_addr = smem + 17424;
    uint8_t* smem_v27 = reinterpret_cast<uint8_t*>(smem_raw + 17472);
    const int smem_v27_addr = smem + 17472;
    uint8_t* smem_v28 = reinterpret_cast<uint8_t*>(smem_raw + 17520);
    const int smem_v28_addr = smem + 17520;
    uint8_t* smem_v29 = reinterpret_cast<uint8_t*>(smem_raw + 17440);
    const int smem_v29_addr = smem + 17440;
    uint8_t* smem_v30 = reinterpret_cast<uint8_t*>(smem_raw + 17488);
    const int smem_v30_addr = smem + 17488;
    uint8_t* smem_v31 = reinterpret_cast<uint8_t*>(smem_raw + 17408);
    const int smem_v31_addr = smem + 17408;

    // Mbarrier init (6 pipeline groups, 0 ordered-sequence groups, 20 barriers)
    // Mbarriers at smem_raw[0..160)

    if (warp == 0) {
        uint32_t leader = elect_sync();
        if (leader) {
            // --- pipeline 'tma_pipe' ---
            // tma_full: 5 barriers, init_count=2
            mbarrier_init(smem + 0, 2);
            mbarrier_init(smem + 8, 2);
            mbarrier_init(smem + 16, 2);
            mbarrier_init(smem + 24, 2);
            mbarrier_init(smem + 32, 2);
            // mma_done: 5 barriers, init_count=1
            mbarrier_init(smem + 40, 1);
            mbarrier_init(smem + 48, 1);
            mbarrier_init(smem + 56, 1);
            mbarrier_init(smem + 64, 1);
            mbarrier_init(smem + 72, 1);
            // --- pipeline 'mainloop_pipe' ---
            // mainloop_done: 1 barriers, init_count=1
            mbarrier_init(smem + 80, 1);
            // epilogue_done: 1 barriers, init_count=16
            mbarrier_init(smem + 88, 16);
            // --- pipeline 'work_pipe' ---
            // work_full: 4 barriers, init_count=1
            mbarrier_init(smem + 96, 1);
            mbarrier_init(smem + 104, 1);
            mbarrier_init(smem + 112, 1);
            mbarrier_init(smem + 120, 1);
            // work_empty: 4 barriers, init_count=546
            mbarrier_init(smem + 128, 546);
            mbarrier_init(smem + 136, 546);
            mbarrier_init(smem + 144, 546);
            mbarrier_init(smem + 152, 546);
            asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        }
    }

    __syncwarp();

    asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory");

    // TMEM alloc (512 columns, 400 used)
    volatile int* tmem_addr_storage = (volatile int*)(smem_raw + 160);
    if (warp == 0) {
        int _tmem_hold = smem + 160;
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
    const int tmem_accum = taddr;
    const int tmem_tmem_sfa = taddr + 256;
    const int tmem_tmem_sfb = taddr + 304;

    // ---- Role: load ----
    if (warp == 0) {
        { // load_main
            unsigned int load_stage = 0;
            unsigned int work_stage = 0;
            int weight_row_base = cta_rank * B_HALF_N;
            unsigned int _phase_work_empty = 1;
            unsigned int _phase_mma_done = 1;
            unsigned int _phase_work_full = 0;
            if (elect_sync()) {
                unsigned int this_bid = bid;
                #pragma unroll 1
                for (unsigned int _tile_iter = 0; _tile_iter < num_cluster_tiles; _tile_iter++) {
                    if (cta_rank == 0) {
                        mbarrier_wait_cluster_hint(work_empty_addr + (work_stage) * 8, _phase_work_empty, 10000000);
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                            "}"
                            :: "r"(work_full_addr + work_stage * 8), "r"(0), "r"((uint32_t)(16)) : "memory");
                        asm volatile(
                            "{\n\t"
                            ".reg .b32 remAddr32;\n\t"
                            "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [remAddr32], %2;\n\t"
                            "}"
                            :: "r"(work_full_addr + work_stage * 8), "r"(1), "r"((uint32_t)(16)) : "memory");
                        asm volatile(
                            "fence.proxy.async.shared::cta;\n\t"
                            "clusterlaunchcontrol.try_cancel.async.shared::cta"
                                ".mbarrier::complete_tx::bytes.multicast::cluster::all.b128"
                                " [%0], [%1];"
                            :: "r"(work_response_addr + work_stage * 16 + 0 * 16), "r"(work_full_addr + work_stage * 8)
                            : "memory");
                    }
                    int group = this_bid / (unsigned int)tiles_per_group;
                    int first_m = group * GROUP_M;
                    int remaining = m_tiles - first_m;
                    int group_size = ((remaining >= GROUP_M) ? GROUP_M : remaining);
                    int local = this_bid % (unsigned int)tiles_per_group;
                    int bid_m = first_m + local % group_size;
                    int bid_n = local / group_size;
                    int off_m = bid_m * BLOCK_M;
                    int off_n = bid_n * BLOCK_N;
                    int sfa_tile_row = bid_m * SF_K_TILES;
                    int sfb_tile_row = bid_n * SF_K_TILES;
                    #pragma unroll 1
                    for (int iter_k = 0; iter_k < NUM_K_ITERS; iter_k++) {
                        mbarrier_wait(mma_done_addr + (load_stage) * 8, _phase_mma_done);
                        tma_3d_gmem2smem_cta2(smem_a_addr + load_stage * 38912, A, 0, off_m, iter_k, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_b_addr + load_stage * 38912, B, 0, weight_row_base + off_n, iter_k, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        int k_set_base = iter_k * 4;
                        tma_3d_gmem2smem_cta2(smem_sfa_all_addr + load_stage * 38912, SFA, 0, 0, sfa_tile_row + k_set_base, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        tma_3d_gmem2smem_cta2(smem_sfb_all_addr + load_stage * 38912, SFB, 0, 0, sfb_tile_row + k_set_base, ((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF));
                        asm volatile(
                            "mbarrier.arrive.expect_tx.release.cta.shared::cluster.b64 _, [%0], %1;"
                            :: "r"((tma_full_addr + (load_stage) * 8) & 0xFEFFFFFF), "r"((uint32_t)(38912)) : "memory");
                        load_stage += 1;
                        if (load_stage == 5) { load_stage = 0; _phase_mma_done ^= 1; }
                    }
                    mbarrier_wait(work_full_addr + (work_stage) * 8, _phase_work_full);
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
                        : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                        : "memory");
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
                        : "r"(work_response_addr + work_stage * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(work_empty_addr + work_stage * 8), "r"(0) : "memory");
                    work_stage += 1;
                    if (work_stage == 4) { work_stage = 0; _phase_work_empty ^= 1; _phase_work_full ^= 1; }
                    if (_clc_valid_0 == 0) {
                        break;
                    }
                    this_bid = _clc_ctaid_0 + (unsigned int)cta_rank;
                }
            }
        }
    }
    // ---- Role: mma ----
    if (warp == 1) {
        { // mma_main
            unsigned int mma_tma_stage = 0;
            unsigned int acc_stage = 0;
            unsigned int work_stage_1 = 0;
            unsigned int _phase_epilogue_done = 1;
            unsigned int _phase_tma_full = 0;
            unsigned int _phase_work_full_1 = 0;
            if (cta_rank == 0) {
                #pragma unroll 1
                for (unsigned int _tile_iter_1 = 0; _tile_iter_1 < num_cluster_tiles; _tile_iter_1++) {
                    mbarrier_wait(epilogue_done_addr + (acc_stage) * 8, _phase_epilogue_done);
                    #pragma unroll 1
                    for (int super_k = 0; super_k < 7; super_k++) {
                        mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        unsigned int stage_p0 = mma_tma_stage;
                        int init_flag = ((super_k == 0) ? 1 : 0);
                        if (elect_sync()) {
                            {
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa, make_sf_cp_desc_lo_sbo128((((smem_v4_addr) >> 4) + (stage_p0) * 2432)));
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb, make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (stage_p0) * 2432)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 4), make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (stage_p0) * 2432 + 32)));
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + 4, make_sf_cp_desc_lo_sbo128((((smem_v5_addr) >> 4) + (stage_p0) * 2432)));
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + 8, make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (stage_p0) * 2432)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 8 + 4), make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (stage_p0) * 2432 + 32)));
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + 8, make_sf_cp_desc_lo_sbo128((((smem_v6_addr) >> 4) + (stage_p0) * 2432)));
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + 16, make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (stage_p0) * 2432)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 16 + 4), make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (stage_p0) * 2432 + 32)));
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + 12, make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (stage_p0) * 2432)));
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + 24, make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (stage_p0) * 2432)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 24 + 4), make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (stage_p0) * 2432 + 32)));
                            }
                            {
                                int _mma_a_lo_0 = (((smem_v14_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                                int _mma_b_lo_0 = (((smem_v23_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                                int _mma_a_next_lo_0 = (((smem_v22_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                                int _mma_b_next_lo_0 = (((smem_v31_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_0 | ((uint64_t)((uint32_t)_mma_a_next_lo_0 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_0 | ((uint64_t)((uint32_t)_mma_b_next_lo_0 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2(tmem_accum, a_desc + 0, b_desc + 0,
                                        0x90400480U, tmem_tmem_sfa + 0, tmem_tmem_sfb + 0, ((init_flag) ? 0 : 1));
                                }
                            }
                            {
                                int _mma_a_lo_1 = (((smem_v15_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                                int _mma_b_lo_1 = (((smem_v24_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                                int _mma_a_next_lo_1 = (((smem_v22_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                                int _mma_b_next_lo_1 = (((smem_v31_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_1 | ((uint64_t)((uint32_t)_mma_a_next_lo_1 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_1 | ((uint64_t)((uint32_t)_mma_b_next_lo_1 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2(tmem_accum, a_desc + 0, b_desc + 0,
                                        0xd04004a0U, tmem_tmem_sfa + 4 + 0, tmem_tmem_sfb + 8 + 0, 1);
                                }
                            }
                        }
                        mma_tma_stage += 1;
                        if (mma_tma_stage == 5) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                        mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        unsigned int stage_p1 = mma_tma_stage;
                        if (elect_sync()) {
                            {
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + 16, make_sf_cp_desc_lo_sbo128((((smem_v4_addr) >> 4) + (stage_p1) * 2432)));
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + 32, make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (stage_p1) * 2432)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 32 + 4), make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (stage_p1) * 2432 + 32)));
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + 20, make_sf_cp_desc_lo_sbo128((((smem_v5_addr) >> 4) + (stage_p1) * 2432)));
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + 40, make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (stage_p1) * 2432)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 40 + 4), make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (stage_p1) * 2432 + 32)));
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + 24, make_sf_cp_desc_lo_sbo128((((smem_v6_addr) >> 4) + (stage_p1) * 2432)));
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + 48, make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (stage_p1) * 2432)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 48 + 4), make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (stage_p1) * 2432 + 32)));
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + 28, make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (stage_p1) * 2432)));
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + 56, make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (stage_p1) * 2432)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 56 + 4), make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (stage_p1) * 2432 + 32)));
                            }
                            {
                                int _mma_a_lo_2 = (((smem_v16_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                                int _mma_b_lo_2 = (((smem_v25_addr) >> 4) & 0x3FFF) + (stage_p0) * 2432;
                                int _mma_a_next_lo_2 = (((smem_v22_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                                int _mma_b_next_lo_2 = (((smem_v31_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_2 | ((uint64_t)((uint32_t)_mma_a_next_lo_2 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_2 | ((uint64_t)((uint32_t)_mma_b_next_lo_2 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2(tmem_accum, a_desc + 0, b_desc + 0,
                                        0x90400480U, tmem_tmem_sfa + 12 + 0, tmem_tmem_sfb + 24 + 0, 1);
                                }
                            }
                        }
                        elect_commit_cg2_multicast(mma_done_addr + (stage_p0) * 8, (uint16_t)(3));
                        if (elect_sync()) {
                            {
                                int _mma_a_lo_3 = (((smem_v17_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                                int _mma_b_lo_3 = (((smem_v26_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                                int _mma_a_next_lo_3 = (((smem_v22_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                                int _mma_b_next_lo_3 = (((smem_v31_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_3 | ((uint64_t)((uint32_t)_mma_a_next_lo_3 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_3 | ((uint64_t)((uint32_t)_mma_b_next_lo_3 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2(tmem_accum, a_desc + 0, b_desc + 0,
                                        0xd04004a0U, tmem_tmem_sfa + 16 + 0, tmem_tmem_sfb + 32 + 0, 1);
                                }
                            }
                            {
                                int _mma_a_lo_4 = (((smem_v18_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                                int _mma_b_lo_4 = (((smem_v27_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                                int _mma_a_next_lo_4 = (((smem_v22_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                                int _mma_b_next_lo_4 = (((smem_v31_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_4 | ((uint64_t)((uint32_t)_mma_a_next_lo_4 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_4 | ((uint64_t)((uint32_t)_mma_b_next_lo_4 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2(tmem_accum, a_desc + 0, b_desc + 0,
                                        0x90400480U, tmem_tmem_sfa + 24 + 0, tmem_tmem_sfb + 48 + 0, 1);
                                }
                            }
                        }
                        mma_tma_stage += 1;
                        if (mma_tma_stage == 5) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                        mbarrier_wait(tma_full_addr + (mma_tma_stage) * 8, _phase_tma_full);
                        asm volatile("tcgen05.fence::after_thread_sync;");
                        unsigned int stage_p2 = mma_tma_stage;
                        if (elect_sync()) {
                            {
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + 32, make_sf_cp_desc_lo_sbo128((((smem_v4_addr) >> 4) + (stage_p2) * 2432)));
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + 64, make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (stage_p2) * 2432)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 64 + 4), make_sf_cp_desc_lo_sbo128((((smem_v8_addr) >> 4) + (stage_p2) * 2432 + 32)));
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + 36, make_sf_cp_desc_lo_sbo128((((smem_v5_addr) >> 4) + (stage_p2) * 2432)));
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + 72, make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (stage_p2) * 2432)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 72 + 4), make_sf_cp_desc_lo_sbo128((((smem_v9_addr) >> 4) + (stage_p2) * 2432 + 32)));
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + 40, make_sf_cp_desc_lo_sbo128((((smem_v6_addr) >> 4) + (stage_p2) * 2432)));
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + 80, make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (stage_p2) * 2432)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 80 + 4), make_sf_cp_desc_lo_sbo128((((smem_v10_addr) >> 4) + (stage_p2) * 2432 + 32)));
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfa + 44, make_sf_cp_desc_lo_sbo128((((smem_v7_addr) >> 4) + (stage_p2) * 2432)));
                                tcgen05_cp_32x128b_warpx4_cta2(tmem_tmem_sfb + 88, make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (stage_p2) * 2432)));
                                tcgen05_cp_32x128b_warpx4_cta2((tmem_tmem_sfb + 88 + 4), make_sf_cp_desc_lo_sbo128((((smem_v11_addr) >> 4) + (stage_p2) * 2432 + 32)));
                            }
                            {
                                int _mma_a_lo_5 = (((smem_v19_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                                int _mma_b_lo_5 = (((smem_v28_addr) >> 4) & 0x3FFF) + (stage_p1) * 2432;
                                int _mma_a_next_lo_5 = (((smem_v22_addr) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                                int _mma_b_next_lo_5 = (((smem_v31_addr) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_5 | ((uint64_t)((uint32_t)_mma_a_next_lo_5 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_5 | ((uint64_t)((uint32_t)_mma_b_next_lo_5 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2(tmem_accum, a_desc + 0, b_desc + 0,
                                        0xd04004a0U, tmem_tmem_sfa + 28 + 0, tmem_tmem_sfb + 56 + 0, 1);
                                }
                            }
                        }
                        elect_commit_cg2_multicast(mma_done_addr + (stage_p1) * 8, (uint16_t)(3));
                        if (elect_sync()) {
                            {
                                int _mma_a_lo_6 = (((smem_v20_addr) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                                int _mma_b_lo_6 = (((smem_v29_addr) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                                int _mma_a_next_lo_6 = (((smem_v22_addr) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                                int _mma_b_next_lo_6 = (((smem_v31_addr) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_6 | ((uint64_t)((uint32_t)_mma_a_next_lo_6 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_6 | ((uint64_t)((uint32_t)_mma_b_next_lo_6 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2(tmem_accum, a_desc + 0, b_desc + 0,
                                        0x90400480U, tmem_tmem_sfa + 36 + 0, tmem_tmem_sfb + 72 + 0, 1);
                                }
                            }
                            {
                                int _mma_a_lo_7 = (((smem_v21_addr) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                                int _mma_b_lo_7 = (((smem_v30_addr) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                                int _mma_a_next_lo_7 = (((smem_v22_addr) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                                int _mma_b_next_lo_7 = (((smem_v31_addr) >> 4) & 0x3FFF) + (stage_p2) * 2432;
                                {
                                    uint64_t a_desc = ((uint64_t)(uint32_t)_mma_a_lo_7 | ((uint64_t)((uint32_t)_mma_a_next_lo_7 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);
                                    uint64_t b_desc = ((uint64_t)(uint32_t)_mma_b_lo_7 | ((uint64_t)((uint32_t)_mma_b_next_lo_7 & ~7u) << 16)) | ((uint64_t)0x40104040 << 32);

                                    tcgen05_mma_mxf4nvf4_bs_block16_k96_cta2(tmem_accum, a_desc + 0, b_desc + 0,
                                        0xd04004a0U, tmem_tmem_sfa + 40 + 0, tmem_tmem_sfb + 80 + 0, 1);
                                }
                            }
                        }
                        elect_commit_cg2_multicast(mma_done_addr + (stage_p2) * 8, (uint16_t)(3));
                        mma_tma_stage += 1;
                        if (mma_tma_stage == 5) { mma_tma_stage = 0; _phase_tma_full ^= 1; }
                    }
                    elect_commit_cg2_multicast(mainloop_done_addr + (acc_stage) * 8, (uint16_t)(3));
                    _phase_epilogue_done ^= 1;
                    mbarrier_wait(work_full_addr + (work_stage_1) * 8, _phase_work_full_1);
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
                        : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    uint32_t _clc_ctaid_1 = 0;
                    asm volatile(
                        "{\n\t"
                        ".reg .pred p1;\n\t"
                        ".reg .b128 clc_r;\n\t"
                        "ld.shared.b128 clc_r, [%1];\n\t"
                        "clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_r;\n\t"
                        "@p1 clusterlaunchcontrol.query_cancel.get_first_ctaid::x.b32.b128 %0, clc_r;\n\t"
                        "}\n"
                        : "=r"(_clc_ctaid_1)
                        : "r"(work_response_addr + work_stage_1 * 16 + 0 * 16)
                        : "memory");
                    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                    asm volatile(
                        "{\n\t"
                        ".reg .b32 remAddr32;\n\t"
                        "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                        "}"
                        :: "r"(work_empty_addr + work_stage_1 * 8), "r"(0) : "memory");
                    work_stage_1 += 1;
                    if (work_stage_1 == 4) { work_stage_1 = 0; _phase_work_full_1 ^= 1; }
                    if (_clc_valid_1 == 0) {
                        break;
                    }
                }
            }
        }
    }
    // ---- Role: epilogue ----
    if (warp >= 2 && warp <= 9) {
        { // epilogue_main
            unsigned int acc_stage_1 = 0;
            unsigned int work_stage_2 = 0;
            const int epi_warp = warp % 4;
            const int head_half = (warp - 2) / 4;
            const int local_row = epi_warp * 32 + lane;
            float alpha_v = alpha[0];
            float global_scale = out_global_scale[0];
            float _rcp_0 = approx_rcp(global_scale);
            float global_scale_rcp = _rcp_0;
            int rows_per_token = HEADS_PER_DESTINATION * 3;
            unsigned int this_bid_1 = bid;
            unsigned int tile_parity = 0;
            int store_bar = 14 + head_half;
            unsigned int _phase_mainloop_done = 0;
            unsigned int _phase_work_full_2 = 0;
            #pragma unroll 1
            for (unsigned int _tile_iter_2 = 0; _tile_iter_2 < num_cluster_tiles; _tile_iter_2++) {
                mbarrier_wait(mainloop_done_addr + (acc_stage_1) * 8, _phase_mainloop_done);
                asm volatile("tcgen05.fence::after_thread_sync;");
                int group_1 = this_bid_1 / (unsigned int)tiles_per_group;
                int first_m_1 = group_1 * GROUP_M;
                int remaining_1 = m_tiles - first_m_1;
                int group_size_1 = ((remaining_1 >= GROUP_M) ? GROUP_M : remaining_1);
                int local_1 = this_bid_1 % (unsigned int)tiles_per_group;
                int bid_m_1 = first_m_1 + local_1 % group_size_1;
                int bid_n_1 = local_1 / group_size_1;
                int off_m_1 = bid_m_1 * BLOCK_M;
                int off_n_1 = bid_n_1 * BLOCK_N;
                int token = off_m_1 + local_row;
                int head_kind = bid_n_1 * 2 + head_half;
                int head = head_kind / 3;
                int kind = head_kind % 3;
                int lane_addr = taddr + (unsigned int)(epi_warp * 32 << 16) + (unsigned int)(head_half * B_HALF_N);
                unsigned int words[64];
                #pragma unroll
                for (int c = 0; c < 4; c++) {
                    float _tmem_load_0[32];
                    asm volatile(
                        "tcgen05.ld.sync.aligned.32x32b.x32.b32"
                        " {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, [%32];"
                        : "=f"(_tmem_load_0[0]), "=f"(_tmem_load_0[1]), "=f"(_tmem_load_0[2]), "=f"(_tmem_load_0[3]), "=f"(_tmem_load_0[4]), "=f"(_tmem_load_0[5]), "=f"(_tmem_load_0[6]), "=f"(_tmem_load_0[7]), "=f"(_tmem_load_0[8]), "=f"(_tmem_load_0[9]), "=f"(_tmem_load_0[10]), "=f"(_tmem_load_0[11]), "=f"(_tmem_load_0[12]), "=f"(_tmem_load_0[13]), "=f"(_tmem_load_0[14]), "=f"(_tmem_load_0[15]), "=f"(_tmem_load_0[16]), "=f"(_tmem_load_0[17]), "=f"(_tmem_load_0[18]), "=f"(_tmem_load_0[19]), "=f"(_tmem_load_0[20]), "=f"(_tmem_load_0[21]), "=f"(_tmem_load_0[22]), "=f"(_tmem_load_0[23]), "=f"(_tmem_load_0[24]), "=f"(_tmem_load_0[25]), "=f"(_tmem_load_0[26]), "=f"(_tmem_load_0[27]), "=f"(_tmem_load_0[28]), "=f"(_tmem_load_0[29]), "=f"(_tmem_load_0[30]), "=f"(_tmem_load_0[31])
                        : "r"(lane_addr + c * 32));
                    asm volatile("tcgen05.wait::ld.sync.aligned;");
                    #pragma unroll
                    for (int j = 0; j < 16; j++) {
                        __nv_bfloat162 _bf16x2_0 = __float22bfloat162_rn(make_float2(_tmem_load_0[2 * j] * alpha_v, _tmem_load_0[2 * j + 1] * alpha_v));
                        words[c * 16 + j] = __as_u32(_bf16x2_0);
                    }
                }
                if (elect_sync()) {
                    asm volatile(
                        "mbarrier.arrive.release.cta.shared::cluster.b64 _, [%0];"
                        :: "r"((epilogue_done_addr + (acc_stage_1) * 8) & 0xFEFFFFFF) : "memory");
                }
                _phase_mainloop_done ^= 1;
                int destination = head / HEADS_PER_DESTINATION;
                int local_head = head % HEADS_PER_DESTINATION;
                int head_kind_local = local_head * 3 + kind;
                int row_in_destination = token * rows_per_token + head_kind_local;
                unsigned long long output_row = (unsigned long long)destination * (unsigned long long)ROWS_PER_DESTINATION + (unsigned long long)row_in_destination;
                unsigned long long q_row_base = output_row * 64;
                unsigned int sr = (unsigned int)row_in_destination;
                unsigned int scale_swizzle = sr >> 7 << 10 | (sr & 31) << 4 | (sr >> 5 & 3) << 2;
                unsigned long long scale_row_base = (unsigned long long)destination * (unsigned long long)SCALE_STRIDE + (unsigned long long)scale_swizzle;
                unsigned int staging_buf = tile_parity << 1 | (unsigned int)head_half;
                int staging_row = (int)staging_buf * 128 + local_row;
                if (epi_warp == 0) {
                    if (elect_sync()) {
                        asm volatile("cp.async.bulk.wait_group.read 1;");
                    }
                }
                asm volatile("barrier.sync %0, 128;" :: "r"(store_bar) : "memory");
                if (kind < 2) {
                    float partial[8];
                    #pragma unroll
                    for (int b = 0; b < 8; b++) {
                        float sum_lo = 0.0f;
                        #pragma unroll
                        for (int j_1 = 0; j_1 < 4; j_1++) {
                            float v_lo = __uint_as_float(words[b * 8 + j_1] << 16);
                            float v_hi = __uint_as_float(words[b * 8 + j_1] & 4294901760);
                            float _fma_0 = __fmaf_rn(v_lo, v_lo, sum_lo);
                            sum_lo = _fma_0;
                            float _fma_1 = __fmaf_rn(v_hi, v_hi, sum_lo);
                            sum_lo = _fma_1;
                        }
                        float sum_hi = 0.0f;
                        #pragma unroll
                        for (int j_2 = 4; j_2 < 8; j_2++) {
                            float v_lo2 = __uint_as_float(words[b * 8 + j_2] << 16);
                            float v_hi2 = __uint_as_float(words[b * 8 + j_2] & 4294901760);
                            float _fma_2 = __fmaf_rn(v_lo2, v_lo2, sum_hi);
                            sum_hi = _fma_2;
                            float _fma_3 = __fmaf_rn(v_hi2, v_hi2, sum_hi);
                            sum_hi = _fma_3;
                        }
                        partial[b] = sum_lo + sum_hi;
                    }
                    float sum_01 = partial[0] + partial[1];
                    float sum_23 = partial[2] + partial[3];
                    float sum_45 = partial[4] + partial[5];
                    float sum_67 = partial[6] + partial[7];
                    float sum_sq = sum_01 + sum_23 + (sum_45 + sum_67);
                    float _fdiv_rn_0 = __fdiv_rn(sum_sq, 128.0f);
                    float mean_sq = _fdiv_rn_0;
                    float _rsqrt_0 = rsqrtf(mean_sq + eps);
                    float rstd = _rsqrt_0;
                    float2 _f2_0 = make_float2(rstd, rstd);
                    unsigned int normalized[64];
                    #pragma unroll
                    for (int b_1 = 0; b_1 < 8; b_1++) {
                        unsigned int weight_words[8];
                        #pragma unroll
                        for (int h = 0; h < 2; h++) {
                            {
                                const uint4* _vptr_0 = reinterpret_cast<const uint4*>(((kind == 0) ? q_norm_weight : k_norm_weight) + b_1 * 16 + h * 8);
                                uint4* _vdst_0 = reinterpret_cast<uint4*>(&weight_words[4 * h]);
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vdst_0[_blk] = _vptr_0[_blk];
                                }
                            }
                        }
                        #pragma unroll
                        for (int j_3 = 0; j_3 < 8; j_3++) {
                            float2 _f2_1 = make_float2(__uint_as_float(words[b_1 * 8 + j_3] << 16), __uint_as_float(words[b_1 * 8 + j_3] & 4294901760));
                            float2 _mul_f32x2_0;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_0) : "l"(*(const unsigned long long*)&_f2_1), "l"(*(const unsigned long long*)&_f2_0));
                            float2 _f2_2 = make_float2(__uint_as_float(weight_words[j_3] << 16), __uint_as_float(weight_words[j_3] & 4294901760));
                            float2 _mul_f32x2_1;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_1) : "l"(*(const unsigned long long*)&_mul_f32x2_0), "l"(*(const unsigned long long*)&_f2_2));
                            __nv_bfloat162 _bf16x2_1 = __float22bfloat162_rn(make_float2(_mul_f32x2_1.x, _mul_f32x2_1.y));
                            normalized[b_1 * 8 + j_3] = __as_u32(_bf16x2_1);
                        }
                    }
                    #pragma unroll
                    for (int b_2 = 0; b_2 < 3; b_2++) {
                        unsigned int cos_words[8];
                        unsigned int sin_words[8];
                        #pragma unroll
                        for (int h_1 = 0; h_1 < 2; h_1++) {
                            {
                                const uint4* _vptr_1 = reinterpret_cast<const uint4*>(rope_cos_sin + token * 96 + b_2 * 16 + h_1 * 8);
                                uint4* _vdst_1 = reinterpret_cast<uint4*>(&cos_words[4 * h_1]);
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vdst_1[_blk] = _vptr_1[_blk];
                                }
                            }
                            {
                                const uint4* _vptr_2 = reinterpret_cast<const uint4*>(rope_cos_sin + token * 96 + 48 + b_2 * 16 + h_1 * 8);
                                uint4* _vdst_2 = reinterpret_cast<uint4*>(&sin_words[4 * h_1]);
                                #pragma unroll
                                for (int _blk = 0; _blk < 1; _blk++) {
                                    _vdst_2[_blk] = _vptr_2[_blk];
                                }
                            }
                        }
                        #pragma unroll
                        for (int j_4 = 0; j_4 < 8; j_4++) {
                            unsigned int n_lo_word = normalized[b_2 * 8 + j_4];
                            unsigned int n_hi_word = normalized[(b_2 + 3) * 8 + j_4];
                            float2 _f2_3 = make_float2(__uint_as_float(cos_words[j_4] << 16), __uint_as_float(cos_words[j_4] & 4294901760));
                            float2 _f2_4 = make_float2(__uint_as_float(sin_words[j_4] << 16 ^ 2147483648), __uint_as_float(sin_words[j_4] & 4294901760 ^ 2147483648));
                            float2 _f2_5 = make_float2(__uint_as_float(sin_words[j_4] << 16), __uint_as_float(sin_words[j_4] & 4294901760));
                            float2 _f2_6 = make_float2(__uint_as_float(n_hi_word << 16), __uint_as_float(n_hi_word & 4294901760));
                            float2 _mul_f32x2_2;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_2) : "l"(*(const unsigned long long*)&_f2_6), "l"(*(const unsigned long long*)&_f2_4));
                            float2 _f2_7 = make_float2(__uint_as_float(n_lo_word << 16), __uint_as_float(n_lo_word & 4294901760));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_fma_f32x2_0;
                            {
                                float2 _packed_f32x2_3_0 = _f2_3;
                                float2 _packed_f32x2_3_1 = _f2_7;
                                float2 _packed_f32x2_3_2 = _mul_f32x2_2;
                                asm volatile("fma.rn.ftz.f32x2 %0, %1, %2, %3;" : "=l"(reinterpret_cast<uint64_t&>(_packed_fma_f32x2_0)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_3_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_3_1)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_3_2)));
                            }
                            float2 _f2_8 = make_float2(__uint_as_float(n_lo_word << 16), __uint_as_float(n_lo_word & 4294901760));
                            float2 _mul_f32x2_3;
                            asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_3) : "l"(*(const unsigned long long*)&_f2_8), "l"(*(const unsigned long long*)&_f2_5));
                            float2 _f2_9 = make_float2(__uint_as_float(n_hi_word << 16), __uint_as_float(n_hi_word & 4294901760));
                            #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 1000
                            #error "Packed FP32x2 arithmetic requires SM100 or newer"
                            #endif
                            float2 _packed_fma_f32x2_1;
                            {
                                float2 _packed_f32x2_4_0 = _f2_3;
                                float2 _packed_f32x2_4_1 = _f2_9;
                                float2 _packed_f32x2_4_2 = _mul_f32x2_3;
                                asm volatile("fma.rn.ftz.f32x2 %0, %1, %2, %3;" : "=l"(reinterpret_cast<uint64_t&>(_packed_fma_f32x2_1)) : "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_4_0)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_4_1)), "l"(reinterpret_cast<const uint64_t&>(_packed_f32x2_4_2)));
                            }
                            __nv_bfloat162 _bf16x2_2 = __float22bfloat162_rn(make_float2(_packed_fma_f32x2_0.x, _packed_fma_f32x2_0.y));
                            normalized[b_2 * 8 + j_4] = __as_u32(_bf16x2_2);
                            __nv_bfloat162 _bf16x2_3 = __float22bfloat162_rn(make_float2(_packed_fma_f32x2_1.x, _packed_fma_f32x2_1.y));
                            normalized[(b_2 + 3) * 8 + j_4] = __as_u32(_bf16x2_3);
                        }
                    }
                    if (write_debug != 0) {
                        if (token < M) {
                            unsigned long long debug_row = ((unsigned long long)token * 56 + (unsigned long long)head) * 64;
                            #pragma unroll
                            for (int c_1 = 0; c_1 < 16; c_1++) {
                                unsigned int debug_chunk[4];
                                #pragma unroll
                                for (int j_5 = 0; j_5 < 4; j_5++) {
                                    debug_chunk[j_5] = normalized[4 * c_1 + j_5];
                                }
                                {
                                    int4 _iv4 = make_int4(debug_chunk[0 + 0], debug_chunk[0 + 1], debug_chunk[0 + 2], debug_chunk[0 + 3]);
                                    *reinterpret_cast<int4*>(((kind == 0) ? debug_q_words : debug_k_words) + (debug_row + (unsigned long long)(4 * c_1)) + 0) = _iv4;
                                }
                            }
                        }
                    }
                    unsigned int row_out[16];
                    #pragma unroll
                    for (int g = 0; g < 2; g++) {
                        float sf_values[4];
                        #pragma unroll
                        for (int bb = 0; bb < 4; bb++) {
                            float values[16];
                            float absolute[16];
                            float quant_values[16];
                            unsigned int packed[2];
                            #pragma unroll
                            for (int j_6 = 0; j_6 < 8; j_6++) {
                                values[2 * j_6] = __uint_as_float(normalized[(4 * g + bb) * 8 + j_6] << 16);
                                values[2 * j_6 + 1] = __uint_as_float(normalized[(4 * g + bb) * 8 + j_6] & 4294901760);
                            }
                            #pragma unroll
                            for (int j_7 = 0; j_7 < 16; j_7++) {
                                absolute[j_7] = values[j_7];
                            }
                            float _fabs_0 = fabsf(absolute[0]);
                            absolute[0] = _fabs_0;
                            float _fabs_1 = fabsf(absolute[1]);
                            absolute[1] = _fabs_1;
                            float _fabs_2 = fabsf(absolute[2]);
                            absolute[2] = _fabs_2;
                            float _fabs_3 = fabsf(absolute[3]);
                            absolute[3] = _fabs_3;
                            float _fabs_4 = fabsf(absolute[4]);
                            absolute[4] = _fabs_4;
                            float _fabs_5 = fabsf(absolute[5]);
                            absolute[5] = _fabs_5;
                            float _fabs_6 = fabsf(absolute[6]);
                            absolute[6] = _fabs_6;
                            float _fabs_7 = fabsf(absolute[7]);
                            absolute[7] = _fabs_7;
                            float _fabs_8 = fabsf(absolute[8]);
                            absolute[8] = _fabs_8;
                            float _fabs_9 = fabsf(absolute[9]);
                            absolute[9] = _fabs_9;
                            float _fabs_10 = fabsf(absolute[10]);
                            absolute[10] = _fabs_10;
                            float _fabs_11 = fabsf(absolute[11]);
                            absolute[11] = _fabs_11;
                            float _fabs_12 = fabsf(absolute[12]);
                            absolute[12] = _fabs_12;
                            float _fabs_13 = fabsf(absolute[13]);
                            absolute[13] = _fabs_13;
                            float _fabs_14 = fabsf(absolute[14]);
                            absolute[14] = _fabs_14;
                            float _fabs_15 = fabsf(absolute[15]);
                            absolute[15] = _fabs_15;
                            float absolute_max = absolute[0];
                            #pragma unroll
                            for (int _lr = 1; _lr < 16; _lr++) {
                                absolute_max = max_noftz(absolute_max, absolute[_lr]);
                            }
                            float amax = absolute_max;
                            float _rcp_1 = approx_rcp(6.0f);
                            float sf_value = global_scale * (amax * _rcp_1);
                            float _fp8_rt_0;
                            uint16_t _e4m3x2_5;
                            uint32_t _f16x2_5;
                            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_5) : "f"(0.0f), "f"(sf_value));
                            asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_5) : "h"(_e4m3x2_5));
                            uint16_t _fp8_h0_5 = (uint16_t)(_f16x2_5 & 0xFFFFu);
                            asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_0) : "h"(_fp8_h0_5));
                            float sf_rounded = _fp8_rt_0;
                            float _rcp_2 = approx_rcp(sf_rounded * global_scale_rcp);
                            float _min_0 = fminf(_rcp_2, 3.4028234663852886e+38f);
                            float output_scale = _min_0;
                            float2 _f2_10 = make_float2(output_scale, output_scale);
                            #pragma unroll
                            for (int j_8 = 0; j_8 < 8; j_8++) {
                                float2 _f2_11 = make_float2(values[2 * j_8], values[2 * j_8 + 1]);
                                float2 _mul_f32x2_4;
                                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_4) : "l"(*(const unsigned long long*)&_f2_11), "l"(*(const unsigned long long*)&_f2_10));
                                quant_values[2 * j_8] = _mul_f32x2_4.x;
                                quant_values[2 * j_8 + 1] = _mul_f32x2_4.y;
                            }
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[0]) : "f"(quant_values[0]), "f"(quant_values[1]), "f"(quant_values[2]), "f"(quant_values[3]), "f"(quant_values[4]), "f"(quant_values[5]), "f"(quant_values[6]), "f"(quant_values[7]));
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed[1]) : "f"(quant_values[8]), "f"(quant_values[9]), "f"(quant_values[10]), "f"(quant_values[11]), "f"(quant_values[12]), "f"(quant_values[13]), "f"(quant_values[14]), "f"(quant_values[15]));
                            sf_values[bb] = sf_value;
                            row_out[2 * (4 * g + bb)] = packed[0];
                            row_out[2 * (4 * g + bb) + 1] = packed[1];
                        }
                        unsigned int sf_packed[1];
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(sf_values[0]), "f"(sf_values[1]),
                                                   "f"(sf_values[2]), "f"(sf_values[3]));
                            sf_packed[0] = _packed;
                        }
                        if (token < M) {
                            *(reinterpret_cast<int*>(out_sf + (scale_row_base + (unsigned long long)(g * 512))) + (0)) = sf_packed[0];
                        }
                    }
                    #pragma unroll
                    for (int c_2 = 0; c_2 < 4; c_2++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((epi_staging_addr + (unsigned int)(staging_row * 64 + c_2 * 16 ^ (staging_row * 64 + c_2 * 16 >> 7 & 3) << 4))), "r"(row_out[4 * c_2]), "r"(row_out[4 * c_2 + 1]), "r"(row_out[4 * c_2 + 2]), "r"(row_out[4 * c_2 + 3]) : "memory");
                    }
                } else {
                    unsigned int row_out_1[16];
                    #pragma unroll
                    for (int g_1 = 0; g_1 < 2; g_1++) {
                        float sf_values_1[4];
                        #pragma unroll
                        for (int bb_1 = 0; bb_1 < 4; bb_1++) {
                            float values_1[16];
                            float absolute_1[16];
                            float quant_values_1[16];
                            unsigned int packed_1[2];
                            #pragma unroll
                            for (int j_9 = 0; j_9 < 8; j_9++) {
                                values_1[2 * j_9] = __uint_as_float(words[(4 * g_1 + bb_1) * 8 + j_9] << 16);
                                values_1[2 * j_9 + 1] = __uint_as_float(words[(4 * g_1 + bb_1) * 8 + j_9] & 4294901760);
                            }
                            #pragma unroll
                            for (int j_10 = 0; j_10 < 16; j_10++) {
                                absolute_1[j_10] = values_1[j_10];
                            }
                            float _fabs_16 = fabsf(absolute_1[0]);
                            absolute_1[0] = _fabs_16;
                            float _fabs_17 = fabsf(absolute_1[1]);
                            absolute_1[1] = _fabs_17;
                            float _fabs_18 = fabsf(absolute_1[2]);
                            absolute_1[2] = _fabs_18;
                            float _fabs_19 = fabsf(absolute_1[3]);
                            absolute_1[3] = _fabs_19;
                            float _fabs_20 = fabsf(absolute_1[4]);
                            absolute_1[4] = _fabs_20;
                            float _fabs_21 = fabsf(absolute_1[5]);
                            absolute_1[5] = _fabs_21;
                            float _fabs_22 = fabsf(absolute_1[6]);
                            absolute_1[6] = _fabs_22;
                            float _fabs_23 = fabsf(absolute_1[7]);
                            absolute_1[7] = _fabs_23;
                            float _fabs_24 = fabsf(absolute_1[8]);
                            absolute_1[8] = _fabs_24;
                            float _fabs_25 = fabsf(absolute_1[9]);
                            absolute_1[9] = _fabs_25;
                            float _fabs_26 = fabsf(absolute_1[10]);
                            absolute_1[10] = _fabs_26;
                            float _fabs_27 = fabsf(absolute_1[11]);
                            absolute_1[11] = _fabs_27;
                            float _fabs_28 = fabsf(absolute_1[12]);
                            absolute_1[12] = _fabs_28;
                            float _fabs_29 = fabsf(absolute_1[13]);
                            absolute_1[13] = _fabs_29;
                            float _fabs_30 = fabsf(absolute_1[14]);
                            absolute_1[14] = _fabs_30;
                            float _fabs_31 = fabsf(absolute_1[15]);
                            absolute_1[15] = _fabs_31;
                            float absolute_max_1 = absolute_1[0];
                            #pragma unroll
                            for (int _lr = 1; _lr < 16; _lr++) {
                                absolute_max_1 = max_noftz(absolute_max_1, absolute_1[_lr]);
                            }
                            float amax_1 = absolute_max_1;
                            float _rcp_3 = approx_rcp(6.0f);
                            float sf_value_1 = global_scale * (amax_1 * _rcp_3);
                            float _fp8_rt_1;
                            uint16_t _e4m3x2_6;
                            uint32_t _f16x2_6;
                            asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(_e4m3x2_6) : "f"(0.0f), "f"(sf_value_1));
                            asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(_f16x2_6) : "h"(_e4m3x2_6));
                            uint16_t _fp8_h0_6 = (uint16_t)(_f16x2_6 & 0xFFFFu);
                            asm volatile("cvt.f32.f16 %0, %1;" : "=f"(_fp8_rt_1) : "h"(_fp8_h0_6));
                            float sf_rounded_1 = _fp8_rt_1;
                            float _rcp_4 = approx_rcp(sf_rounded_1 * global_scale_rcp);
                            float _min_1 = fminf(_rcp_4, 3.4028234663852886e+38f);
                            float output_scale_1 = _min_1;
                            float2 _f2_12 = make_float2(output_scale_1, output_scale_1);
                            #pragma unroll
                            for (int j_11 = 0; j_11 < 8; j_11++) {
                                float2 _f2_13 = make_float2(values_1[2 * j_11], values_1[2 * j_11 + 1]);
                                float2 _mul_f32x2_5;
                                asm("mul.rn.ftz.f32x2 %0, %1, %2;" : "=l"(*(unsigned long long*)&_mul_f32x2_5) : "l"(*(const unsigned long long*)&_f2_13), "l"(*(const unsigned long long*)&_f2_12));
                                quant_values_1[2 * j_11] = _mul_f32x2_5.x;
                                quant_values_1[2 * j_11 + 1] = _mul_f32x2_5.y;
                            }
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_1[0]) : "f"(quant_values_1[0]), "f"(quant_values_1[1]), "f"(quant_values_1[2]), "f"(quant_values_1[3]), "f"(quant_values_1[4]), "f"(quant_values_1[5]), "f"(quant_values_1[6]), "f"(quant_values_1[7]));
                            asm volatile(" { .reg .b8 __b0, __b1, __b2, __b3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b0, %2, %1; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b1, %4, %3; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b2, %6, %5; \n"             " cvt.rn.satfinite.e2m1x2.f32 __b3, %8, %7; \n"             " mov.b32 %0, {__b0, __b1, __b2, __b3}; \n"             " } \n"             : "=r"(packed_1[1]) : "f"(quant_values_1[8]), "f"(quant_values_1[9]), "f"(quant_values_1[10]), "f"(quant_values_1[11]), "f"(quant_values_1[12]), "f"(quant_values_1[13]), "f"(quant_values_1[14]), "f"(quant_values_1[15]));
                            sf_values_1[bb_1] = sf_value_1;
                            row_out_1[2 * (4 * g_1 + bb_1)] = packed_1[0];
                            row_out_1[2 * (4 * g_1 + bb_1) + 1] = packed_1[1];
                        }
                        unsigned int sf_packed_1[1];
                        {
                            uint32_t _packed;
                            asm volatile("{\n\t"
                                ".reg .b16 _lo;\n\t"
                                ".reg .b16 _hi;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _lo, %2, %1;\n\t"
                                "cvt.rn.satfinite.e4m3x2.f32 _hi, %4, %3;\n\t"
                                "mov.b32 %0, {_lo, _hi};\n\t"
                                "}"
                                : "=r"(_packed) : "f"(sf_values_1[0]), "f"(sf_values_1[1]),
                                                   "f"(sf_values_1[2]), "f"(sf_values_1[3]));
                            sf_packed_1[0] = _packed;
                        }
                        if (token < M) {
                            *(reinterpret_cast<int*>(out_sf + (scale_row_base + (unsigned long long)(g_1 * 512))) + (0)) = sf_packed_1[0];
                        }
                    }
                    #pragma unroll
                    for (int c_3 = 0; c_3 < 4; c_3++) {
                        asm volatile("st.shared.v4.b32 [%0], {%1,%2,%3,%4};" :: "r"((epi_staging_addr + (unsigned int)(staging_row * 64 + c_3 * 16 ^ (staging_row * 64 + c_3 * 16 >> 7 & 3) << 4))), "r"(row_out_1[4 * c_3]), "r"(row_out_1[4 * c_3 + 1]), "r"(row_out_1[4 * c_3 + 2]), "r"(row_out_1[4 * c_3 + 3]) : "memory");
                    }
                }
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile("barrier.sync %0, 128;" :: "r"(store_bar) : "memory");
                if (epi_warp == 0) {
                    if (elect_sync()) {
                        tma_store_4d(OUTQ, 0, head_kind_local, off_m_1, destination, epi_staging_addr + staging_buf * 8192);
                        asm volatile("cp.async.bulk.commit_group;");
                    }
                }
                tile_parity = tile_parity ^ 1;
                mbarrier_wait(work_full_addr + (work_stage_2) * 8, _phase_work_full_2);
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
                    : "r"(work_response_addr + work_stage_2 * 16 + 0 * 16)
                    : "memory");
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                asm volatile(
                    "{\n\t"
                    ".reg .b32 remAddr32;\n\t"
                    "mapa.shared::cluster.u32 remAddr32, %0, %1;\n\t"
                    "mbarrier.arrive.release.cta.shared::cluster.b64 _, [remAddr32];\n\t"
                    "}"
                    :: "r"(work_empty_addr + work_stage_2 * 8), "r"(0) : "memory");
                work_stage_2 += 1;
                if (work_stage_2 == 4) { work_stage_2 = 0; _phase_work_full_2 ^= 1; }
                if (_clc_valid_2 == 0) {
                    break;
                }
                this_bid_1 = _clc_ctaid_2 + (unsigned int)cta_rank;
            }
            {
                if (epi_warp == 0) {
                    if (elect_sync()) {
                        asm volatile("cp.async.bulk.wait_group 0;");
                    }
                }
            }
        }
    }

    // Cleanup
    asm volatile("barrier.cluster.arrive.release.aligned;");
    asm volatile("barrier.cluster.wait.acquire.aligned;");

    if (warp == 0) {
        asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" :: "r"(tmem_addr_storage[0]), "r"(512));
    }
}

} // extern "C"
